"""Shape-conditioned physics-informed neural network for the thermal field.

Steady-state AXISYMMETRIC heat conduction on the meridian half-plane (r >= 0):

    k ( T_rr + T_r / r + T_zz ) + Q''' = 0        inside the body
    -k dT/dn = h_eff (T - T_inf)                  on the outer surface
    dT/dr = 0                                     on the symmetry axis

Heat sources are axisymmetric rings normalized by integral over the meridian
plane weighted by 2 pi r, so integral Q''' 2 pi r dr dz = P_i exactly. The
domain is parameterized by the profile spline: collocation points live in a
fixed reference square (rho, s) and are mapped by r = rho * r(s), so the PDE
residual is differentiable with respect to the design vector. The 1/r term is
evaluated with r clamped at R_AXIS_MIN (collocation is kept outside a thin
axis cylinder, and a symmetry loss enforces dT/dr = 0 at the axis).

The network is warm-started across outer optimization iterations
(continuation). The same formulation is solved independently by
src/verify_fdm.py (annular finite volumes) to verify PINN accuracy.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .geometry import profile_radius

# physical constants (SI)
T_INF = 25.0            # deg C ambient
H_EFF = 11.5            # W/(m^2 K) natural convection + linearized radiation
K_PLA = 0.13            # W/(m K) shell
K_CAV = 0.50            # W/(m K) lumped cavity (air + radiation + mixing)
R_REF = 0.06            # m, fixed input normalisation
R_AXIS_MIN = 1.5e-3     # m, clamp for the 1/r term

# heat sources: name, power [W], r offset [m], sigma_r [m], sigma_z [m], key
SOURCES = [
    ("pcb", 2.2, 0.010, 0.014, 0.005, "zpcb"),
    ("battery", 0.3, 0.008, 0.013, 0.008, "zbat"),
    ("speaker", 1.0, 0.008, 0.010, 0.006, "spk"),
]
P_TOT = sum(s[1] for s in SOURCES)

RES_SCALE = 1.05e4      # ~mean source density, O(1) loss balancing [W/m^3]
BC_SCALE = 1.5e2        # ~h_eff * typical dT [W/m^2]; bc term weighted x2.5
BC_WEIGHT = 2.5


class ThermalPINN(nn.Module):
    """MLP (r_n, z_n) -> dT [K], added to T_INF."""

    def __init__(self, hidden: int = 64, depth: int = 4):
        super().__init__()
        layers = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, r_m: torch.Tensor, z_m: torch.Tensor, geo) -> torch.Tensor:
        zn = (z_m - geo.clr * 1e-3) / (geo.H * 1e-3)
        rn = r_m / R_REF
        return self.net(torch.cat([rn, zn], dim=-1)) * 15.0  # dT scale 15 K


def _source_z(key: str, geo) -> torch.Tensor:
    if key == "spk":
        return geo.clr * 1e-3 + 0.86 * geo.H * 1e-3
    return geo.clr * 1e-3 + geo.d[key] * geo.H * 1e-3


def _r_profile_m(s: torch.Tensor, geo) -> torch.Tensor:
    return profile_radius(s, geo.r_ctrl) * 1e-3


def _source_norm_int(geo, dtype=torch.float32) -> torch.Tensor:
    """Integral of each gaussian over the meridian plane, 2 pi r weighted [m^3]."""
    s_g = torch.linspace(0.0, 1.0, 81, dtype=dtype)
    rho_g = torch.linspace(0.0, 1.0, 41, dtype=dtype)
    r_g = _r_profile_m(s_g, geo)                      # (81,)
    z_g = geo.clr * 1e-3 + s_g * geo.H * 1e-3         # (81,)
    ints = []
    for _, _p, r_off, sr, sz, key in SOURCES:
        z_c = _source_z(key, geo)
        phi = torch.exp(-(((rho_g[:, None] * r_g[None, :] - r_off) / sr) ** 2)
                        - (((z_g[None, :] - z_c) / sz) ** 2))
        # annular volume element: dV = 2 pi (rho r(s)) r(s) d(rho) dz
        vol_elem = rho_g[:, None] * r_g[None, :] ** 2
        # int_i = integral of phi * 2 pi rho r(s)^2 drho dz  [m^3]
        ints.append(2.0 * torch.pi * torch.trapz(
            torch.trapz(phi * vol_elem, rho_g, dim=0), s_g) * geo.H * 1e-3)
    return torch.stack(ints)   # (n_src,) [m^3]


def _source_field(r_m, z_m, geo) -> torch.Tensor:
    """Q''' [W/m^3]: sum_i P_i * phi_i / integral(phi_i * 2 pi r)."""
    ints = _source_norm_int(geo, r_m.dtype)
    q = torch.zeros_like(r_m)
    for i, (_, p, r_off, sr, sz, key) in enumerate(SOURCES):
        z_c = _source_z(key, geo)
        phi = torch.exp(-(((r_m - r_off) / sr) ** 2)
                        - (((z_m - z_c) / sz) ** 2))
        q = q + p * phi / ints[i]
    return q


def _k_field(r_m, r_in_m):
    """Piecewise conductivity, smoothly blended at the inner wall."""
    w = torch.sigmoid((r_m - r_in_m) / 8e-4)   # ~1 in the wall
    return K_CAV + (K_PLA - K_CAV) * w


def _collocation(n: int, geo, dtype=torch.float32, src_frac: float = 0.3):
    """Reference-square samples mapped into the physical domain.

    A fraction `src_frac` of points is sampled preferentially around the heat
    sources (residual-guided sampling in the spirit of RAR), the rest
    uniformly over the full radius INCLUDING the axis: below R_AXIS_MIN the
    residual switches to the limit form 2 k T_rr, so no region of the domain
    is left unconstrained (a previous version sampled only r >= 0.02 r(s),
    which allowed an absorbing artifact column at the axis).
    """
    n_src = int(n * src_frac)
    n_uni = n - n_src
    rho_u = torch.rand(n_uni, 1, dtype=dtype)
    s_u = torch.rand(n_uni, 1, dtype=dtype)

    pw = torch.tensor([s[1] for s in SOURCES], dtype=dtype)
    pw = pw / pw.sum()
    pick = torch.multinomial(pw, n_src, replacement=True)
    s_b = torch.rand(n_src, 1, dtype=dtype)
    rho_b = torch.rand(n_src, 1, dtype=dtype)
    for i, (_, _p, r_off, sr, sz, key) in enumerate(SOURCES):
        m = (pick == i)
        if m.any():
            z_c = _source_z(key, geo)
            s_mean = ((z_c - geo.clr * 1e-3) / (geo.H * 1e-3)).reshape(1, 1)
            s_i = torch.clamp(s_mean + torch.randn(m.sum(), 1, dtype=dtype)
                              * (2.0 * sz) / (geo.H * 1e-3), 0.0, 1.0)
            r_i = (r_off + torch.randn(m.sum(), 1, dtype=dtype) * 2.0 * sr) \
                .clamp(min=0.0)
            rho_i = (r_i / _r_profile_m(s_i, geo)).clamp(0.0, 1.0)
            s_b[m] = s_i
            rho_b[m] = rho_i
    rho = torch.cat([rho_u, rho_b], dim=0)
    s = torch.cat([s_u, s_b], dim=0)
    r_m = (rho * _r_profile_m(s, geo)).requires_grad_(True)
    z_m = (geo.clr * 1e-3 + s * geo.H * 1e-3).requires_grad_(True)
    return r_m, z_m, s
    r_m = (rho * _r_profile_m(s, geo)).requires_grad_(True)
    z_m = (geo.clr * 1e-3 + s * geo.H * 1e-3).requires_grad_(True)
    return r_m, z_m, s


def _boundary_points(n_curve: int, n_cap: int, geo, dtype=torch.float32):
    """Outer-surface points with outward normals: side curve + top + bottom."""
    s_b = torch.rand(n_curve, 1, dtype=dtype)
    r_mm, dr_mm = profile_radius(s_b, geo.r_ctrl, with_deriv=True)
    r_b = r_mm * 1e-3
    z_b = geo.clr * 1e-3 + s_b * geo.H * 1e-3
    dr_m, H_m = dr_mm * 1e-3, geo.H * 1e-3
    nrm = torch.sqrt(H_m ** 2 + dr_m ** 2)
    n_r, n_z = H_m / nrm, -dr_m / nrm

    r_top = (torch.rand(n_cap, 1, dtype=dtype)
             * _r_profile_m(torch.ones(1), geo)[0])
    z_top = (r_top.new_full((n_cap, 1), 0.0) + geo.clr * 1e-3 + geo.H * 1e-3)
    r_bot = (torch.rand(n_cap, 1, dtype=dtype)
             * _r_profile_m(torch.zeros(1), geo)[0])
    z_bot = (r_bot.new_full((n_cap, 1), 0.0) + geo.clr * 1e-3)

    r_all = torch.cat([r_b, r_top, r_bot], dim=0).requires_grad_(True)
    z_all = torch.cat([z_b, z_top, z_bot], dim=0).requires_grad_(True)
    n_all = torch.cat([
        torch.cat([n_r, torch.zeros_like(r_top), torch.zeros_like(r_bot)], dim=0),
        torch.cat([n_z, torch.ones_like(r_top), -torch.ones_like(r_bot)], dim=0),
    ], dim=-1)
    return r_all, z_all, n_all


def solve_thermal(net: ThermalPINN, geo, steps: int, lr: float = 5e-3,
                  n_col: int = 1200, n_bc_curve: int = 320, n_cap: int = 80,
                  n_axis: int = 120, resample_every: int = 150,
                  record_every: int = 100, verbose: bool = False,
                  lbfgs_steps: int = 0):
    """Train (or continue training) the thermal PINN on the current shape.

    `geo` must be built from DETACHED design tensors (leaves): the inner loop
    trains network parameters only; design gradients flow through
    thermal_metrics at the outer level. Returns (net, history).
    """
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, lr * 0.12)
    history = []
    for it in range(steps):
        if it % resample_every == 0 or it == steps - 1:
            # fresh batch: 2/3 fresh (uniform + source-biased), 1/3 RAR:
            # highest-|residual|-volume points from a large candidate pool
            n_fresh = (n_col * 2) // 3
            n_rar = n_col - n_fresh
            r_f, z_f, s_f = _collocation(n_fresh, geo)
            r_p, z_p, s_p = _collocation(max(n_rar * 4, 200), geo,
                                         src_frac=0.10)
            with torch.enable_grad():
                r_p = r_p.detach().requires_grad_(True)
                z_p = z_p.detach().requires_grad_(True)
                # second derivatives need the first-pass graph: score with
                # create_graph=True and detach afterwards
                res_p, vol_p = _pde_residual(net, geo, r_p, z_p, s_p)
            score = (res_p.detach().abs() * vol_p).flatten()
            take = torch.topk(score, n_rar).indices
            r_c = torch.cat([r_f, r_p[take]]).detach().requires_grad_(True)
            z_c = torch.cat([z_f, z_p[take]]).detach().requires_grad_(True)
            s_c = torch.cat([s_f, s_p[take]]).detach()
            r_b, z_b, n_b = _boundary_points(n_bc_curve, n_cap, geo)
            r_b = r_b.detach().requires_grad_(True)
            z_b = z_b.detach().requires_grad_(True)
            n_b = n_b.detach()
            r_ax = torch.zeros(n_axis, 1, requires_grad=True)
            z_ax = (geo.clr * 1e-3
                    + torch.rand(n_axis, 1) * geo.H * 1e-3).detach() \
                .requires_grad_(True)
        loss = _loss_on(net, geo, r_c, z_c, s_c, r_b, z_b, n_b, r_ax, z_ax)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if it % record_every == 0 or it == steps - 1:
            history.append(float(loss.detach()))
            if verbose:
                print(f"  [pinn] step {it:5d}  loss {history[-1]:.4e}")

    # ---- L-BFGS refinement on a fixed large batch (Raissi-style stage 2) --
    if lbfgs_steps > 0:
        r_c, z_c, s_c = _collocation(n_col, geo)
        r_b, z_b, n_b = _boundary_points(n_bc_curve * 2, n_cap * 2, geo)
        r_ax = torch.zeros(n_axis, 1, requires_grad=True)
        z_ax = (geo.clr * 1e-3
                + torch.rand(n_axis, 1) * geo.H * 1e-3).detach() \
            .requires_grad_(True)

        def closure():
            opt_lb.zero_grad()
            loss = _loss_on(net, geo, r_c, z_c, s_c, r_b, z_b, n_b, r_ax, z_ax)
            loss.backward()
            return loss

        opt_lb = torch.optim.LBFGS(net.parameters(), max_iter=lbfgs_steps,
                                   history_size=60, line_search_fn="strong_wolfe",
                                   tolerance_grad=1e-12, tolerance_change=1e-14)
        opt_lb.step(closure)
        history.append(float(_loss_on(net, geo, r_c, z_c, s_c, r_b, z_b,
                                      n_b, r_ax, z_ax).detach()))
    return net, history


def _k_field_full(r_m, z_m, geo):
    """Conductivity as a function of position (differentiable w.r.t. r, z).

    s is recovered from z so the wall blend follows the spline profile; the
    resulting k has a steep but smooth transition at the inner wall, whose
    gradient k enters the PDE residual via the k . grad(T) term of
    div(k grad T).
    """
    s = ((z_m - geo.clr * 1e-3) / (geo.H * 1e-3)).clamp(0.0, 1.0)
    r_in = (_r_profile_m(s, geo) - geo.t * 1e-3).clamp(min=2.5e-3)
    w = torch.sigmoid((r_m - r_in) / 8e-4)
    return K_CAV + (K_PLA - K_CAV) * w


def _pde_residual(net, geo, r_c, z_c, s_c, create_graph: bool = True):
    """Raw + scaled PDE residual; axis limit form 2 k T_rr below R_AXIS_MIN.

    The strong form used is div(k grad T) + Q = k (T_rr + T_r/r + T_zz)
    + grad(k) . grad(T) + Q: omitting grad(k).grad(T) makes the interface
    between wall and cavity an unconstrained flux source/sink and the field
    then violates global energy conservation while the pointwise loss stays
    small (observed and fixed during verification against the FV solver).

    Returns (res_scaled, vol_w); vol_w is the annular volume element per
    reference-square point, normalized to mean 1 - the loss weights points by
    volume so the MSE approximates the domain integral of the residual
    squared.
    """
    q = _source_field(r_c, z_c, geo)
    dT = net(r_c, z_c, geo)
    g1 = torch.autograd.grad(dT.sum(), [r_c, z_c], create_graph=create_graph)
    dT_r, dT_z = g1
    lap_r = torch.autograd.grad(dT_r.sum(), r_c, create_graph=create_graph)[0]
    lap_z = torch.autograd.grad(dT_z.sum(), z_c, create_graph=create_graph)[0]
    k = _k_field_full(r_c, z_c, geo)
    gk = torch.autograd.grad(k.sum(), [r_c, z_c], create_graph=create_graph)
    dk_dr, dk_dz = gk
    near_axis = (r_c < R_AXIS_MIN).float()
    radial = torch.where(near_axis > 0.5, 2.0 * lap_r,
                         lap_r + dT_r / r_c.clamp(min=R_AXIS_MIN))
    res = (k * (radial + lap_z) + dk_dr * dT_r + dk_dz * dT_z + q)
    rho = (r_c.detach() / _r_profile_m(s_c, geo).detach()).clamp(max=1.0)
    vol = 2.0 * torch.pi * rho * _r_profile_m(s_c, geo).detach() ** 2 \
        * geo.H * 1e-3
    vol_w = (vol / vol.mean()).detach()
    return res / RES_SCALE, vol_w


def _loss_on(net, geo, r_c, z_c, s_c, r_b, z_b, n_b, r_ax, z_ax):
    res, vol_w = _pde_residual(net, geo, r_c, z_c, s_c)

    Tb = T_INF + net(r_b, z_b, geo)
    dT_b = Tb - T_INF
    gbn = torch.autograd.grad(Tb.sum(), [r_b, z_b], create_graph=True)
    dTn = gbn[0] * n_b[..., 0:1] + gbn[1] * n_b[..., 1:2]
    bc = (K_PLA * dTn + H_EFF * dT_b) / BC_SCALE

    Tax = T_INF + net(r_ax, z_ax, geo)
    gax = torch.autograd.grad(Tax.sum(), r_ax, create_graph=True)[0]
    sym = gax * 1e-4
    return ((vol_w * res ** 2).mean() + BC_WEIGHT * (bc ** 2).mean()
            + (sym ** 2).mean())


def thermal_metrics(net: ThermalPINN, geo, n_eval: int = 240) -> dict:
    """Differentiable field metrics on a dense grid of the current shape."""
    s = torch.linspace(0, 1, n_eval).unsqueeze(-1)      # (n_eval, 1)
    rho = torch.linspace(0, 1, 60).unsqueeze(0)         # (1, 60)
    r_m = (rho * _r_profile_m(s, geo)).reshape(-1, 1)   # (n_eval*60, 1)
    z_m = (geo.clr * 1e-3 + s * geo.H * 1e-3).repeat(1, 60).reshape(-1, 1)
    mask = (rho.expand(n_eval, 60) < 0.9).reshape(-1)
    dT = net(r_m, z_m, geo).reshape(-1)
    comp = {"T_max": T_INF + dT.max(),
            "T_cav_mean": T_INF + dT[mask].mean()}
    for name, _, r_off, _, _, key in SOURCES:
        z_c = _source_z(key, geo).reshape(1, 1)
        t_c = net(torch.full((1, 1), r_off), z_c, geo)
        comp[f"T_{name}"] = T_INF + t_c.squeeze()
    return comp
