"""Shape-conditioned physics-informed neural network for the thermal field.

Steady-state heat equation on the meridian half cross-section of the body
(planar per-unit-depth equivalent of the axisymmetric problem, W_eff = 30 mm
effective depth):

    k(x) Laplacian(T) + q'''(x) = 0                  inside the domain
    -k dT/dn = h_eff (T - T_inf)                     on the outer surface
    dT/dr = 0                                        on the symmetry axis

The domain is parameterized by the profile spline: collocation points live in
a fixed reference square (rho, s) and are mapped by r = rho * r(s), so the PDE
residual is differentiable with respect to the design vector. The network is
warm-started across outer optimization iterations (continuation), which is
what makes the outer shape loop tractable on CPU.

The same formulation is solved independently by src/verify_fdm.py (finite
volume) to verify PINN accuracy - the optimization claim rests on the field
solver, so it is cross-validated.
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
W_DEPTH = 0.03          # m effective depth of the equivalent 2D section
R_REF = 0.06            # m, fixed input normalisation

# heat sources: name, power [W], r offset [m], sigma_r [m], sigma_z [m], key
SOURCES = [
    ("pcb", 2.2, 0.010, 0.014, 0.005, "zpcb"),
    ("battery", 0.3, 0.008, 0.013, 0.008, "zbat"),
    ("speaker", 1.0, 0.008, 0.010, 0.006, "spk"),
]
P_TOT = sum(s[1] for s in SOURCES)

RES_SCALE = 6.2e3       # ~mean source density, O(1) loss balancing [W/m^3]
BC_SCALE = 2.3e2        # ~h_eff * typical dT [W/m^2]


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
    """Integral of each gaussian over the half domain (differentiable, m^2)."""
    s_g = torch.linspace(0.0, 1.0, 81, dtype=dtype)
    rho_g = torch.linspace(0.0, 1.0, 41, dtype=dtype)
    r_g = _r_profile_m(s_g, geo)                      # (81,)
    z_g = geo.clr * 1e-3 + s_g * geo.H * 1e-3         # (81,)
    ints = []
    for _, _p, r_off, sr, sz, key in SOURCES:
        z_c = _source_z(key, geo)
        phi = torch.exp(-(((rho_g[:, None] * r_g[None, :] - r_off) / sr) ** 2)
                        - (((z_g[None, :] - z_c) / sz) ** 2))
        # dA = r(s) d(rho) dz  ->  integrate rho then s
        ints.append(torch.trapz(torch.trapz(phi * r_g[None, :], rho_g, dim=0),
                                s_g) * geo.H * 1e-3)
    return torch.stack(ints)   # (n_src,)


def _source_field(r_m, z_m, geo) -> torch.Tensor:
    """q''' [W/m^3]: sum_i (P_i/2)/W_eff * phi_i / int(phi_i)."""
    ints = _source_norm_int(geo, r_m.dtype)
    q = torch.zeros_like(r_m)
    for i, (_, p, r_off, sr, sz, key) in enumerate(SOURCES):
        z_c = _source_z(key, geo)
        phi = torch.exp(-(((r_m - r_off) / sr) ** 2)
                        - (((z_m - z_c) / sz) ** 2))
        q = q + (p / 2.0 / W_DEPTH) * phi / ints[i]
    return q


def _k_field(r_m, r_in_m):
    """Piecewise conductivity, smoothly blended at the inner wall."""
    w = torch.sigmoid((r_m - r_in_m) / 8e-4)   # ~1 in the wall
    return K_CAV + (K_PLA - K_CAV) * w


def _collocation(n: int, geo, dtype=torch.float32):
    """Reference-square samples mapped into the physical domain."""
    rho = torch.rand(n, 1, dtype=dtype)
    s = torch.rand(n, 1, dtype=dtype)
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
    z_top = (r_top.new_full((n_cap, 1), 0.0)
             + geo.clr * 1e-3 + geo.H * 1e-3)
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
                  record_every: int = 100, verbose: bool = False):
    """Train (or continue training) the thermal PINN on the current shape.

    Returns (net, history) - the net keeps its weights for warm starting.
    """
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, lr * 0.12)
    history = []
    for it in range(steps):
        if it % resample_every == 0 or it == steps - 1:
            r_c, z_c, s_c = _collocation(n_col, geo)
            # detach to leaves: the inner loss trains network params only;
            # design gradients flow through thermal_metrics at the outer level
            r_c = r_c.detach().requires_grad_(True)
            z_c = z_c.detach().requires_grad_(True)
            s_c = s_c.detach()
            r_b, z_b, n_b = _boundary_points(n_bc_curve, n_cap, geo)
            r_b = r_b.detach().requires_grad_(True)
            z_b = z_b.detach().requires_grad_(True)
            n_b = n_b.detach()
            r_ax = torch.zeros(n_axis, 1, requires_grad=True)
            z_ax = (geo.clr * 1e-3
                    + torch.rand(n_axis, 1) * geo.H * 1e-3).detach() \
                .requires_grad_(True)
        q = _source_field(r_c, z_c, geo)
        dT = net(r_c, z_c, geo)
        T = T_INF + dT
        g1 = torch.autograd.grad(dT.sum(), [r_c, z_c], create_graph=True)
        dT_r, dT_z = g1
        lap_r = torch.autograd.grad(dT_r.sum(), r_c, create_graph=True)[0]
        lap_z = torch.autograd.grad(dT_z.sum(), z_c, create_graph=True)[0]
        r_in_at = (_r_profile_m(s_c, geo) - geo.t * 1e-3).clamp(min=2.5e-3)
        k = _k_field(r_c, r_in_at)
        res = (k * (lap_r + lap_z) + q) / RES_SCALE

        Tb = T_INF + net(r_b, z_b, geo)
        dT_b = Tb - T_INF
        gbn = torch.autograd.grad(Tb.sum(), [r_b, z_b], create_graph=True)
        dTn = gbn[0] * n_b[..., 0:1] + gbn[1] * n_b[..., 1:2]
        bc = (K_PLA * dTn + H_EFF * dT_b) / BC_SCALE

        Tax = T_INF + net(r_ax, z_ax, geo)
        gax = torch.autograd.grad(Tax.sum(), r_ax, create_graph=True)[0]
        sym = gax * 1e-4

        loss = (res ** 2).mean() + (bc ** 2).mean() + (sym ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if it % record_every == 0 or it == steps - 1:
            history.append(float(loss.detach()))
            if verbose:
                print(f"  [pinn] step {it:5d}  loss {history[-1]:.4e}")
    return net, history


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
