"""Parametric robot-body geometry (PhORM framework).

The body is a solid of revolution. Its meridian profile r(s), s in [0, 1] is
the height fraction, is a natural cubic spline through K control radii. The
design vector collects geometric control points, overall size, wall thickness,
wheel geometry, and the vertical placement/mass of internal components.

Everything is a torch tensor with a live autograd graph, so the chain
    design vector -> geometry -> physics -> loss
is differentiable end-to-end (Hoyer et al. 2019 style neural reparameterization
spirit, but with explicit geometric control points instead of per-pixel fields).

Units: mm, g, s  (SI conversions happen at use sites).
"""
from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# Design space definition
# ---------------------------------------------------------------------------

N_KNOTS = 7
# height fractions of the profile control points (fixed knots)
S_KNOTS = torch.tensor([0.00, 0.15, 0.32, 0.52, 0.72, 0.88, 1.00])

# name -> (lower bound, upper bound, unit)
SPEC = {
    "r0":   (10.0, 45.0, "mm"),   # base radius control point
    "r1":   (10.0, 45.0, "mm"),
    "r2":   (10.0, 45.0, "mm"),
    "r3":   (10.0, 45.0, "mm"),
    "r4":   (10.0, 45.0, "mm"),
    "r5":   (10.0, 45.0, "mm"),
    "r6":   (8.0,  35.0, "mm"),   # top control point (rounded crown)
    "H":    (85.0, 135.0, "mm"),  # body height (excl. wheels)
    "t":    (1.6, 3.2, "mm"),     # shell wall thickness (printability)
    "rw":   (10.0, 20.0, "mm"),   # wheel radius
    "wt":   (34.0, 96.0, "mm"),   # track width (wheel centre to centre)
    "mbat": (80.0, 200.0, "g"),   # battery mass
    "zbat": (0.05, 0.45, "frac"), # battery height fraction
    "zpcb": (0.35, 0.75, "frac"), # PCB height fraction
    "clr":  (3.0, 8.0, "mm"),     # ground clearance under the shell
}
PARAM_NAMES = list(SPEC.keys())
N_PARAMS = len(PARAM_NAMES)
LB = torch.tensor([SPEC[k][0] for k in PARAM_NAMES])
UB = torch.tensor([SPEC[k][1] for k in PARAM_NAMES])

# fixed component masses (g) and mounts
M_PCB = 45.0        # main board + compute + mic array
M_SPK = 32.0        # speaker assembly, fixed at head
Z_SPK = 0.86        # speaker height fraction (fixed: acoustic path to mouth)
M_HEAD = 12.0       # camera/LED/ear electronics at the crown
Z_HEAD = 0.95
M_DRIVE = 90.0      # both wheel pods: motors + wheels + pods, at axle height

RHO_PLA = 1.24e-3   # g/mm^3 (PLA)
# volumetric footprints (cm^3) for the packing constraint
V_COMP_PER_G = {"battery": 0.40}
V_COMP_FIXED = {"pcb": 22.5, "speaker": 19.2, "head": 6.0, "drive": 31.5}

# stock "dtiny-like" reference design
STOCK = {
    "r0": 34.0, "r1": 40.0, "r2": 36.0, "r3": 30.0, "r4": 33.0, "r5": 29.0,
    "r6": 14.0, "H": 105.0, "t": 2.4, "rw": 15.0, "wt": 44.0,
    "mbat": 140.0, "zbat": 0.18, "zpcb": 0.55, "clr": 5.0,
}


def design_from_flat(theta: torch.Tensor) -> dict:
    """Flat tensor (in physical units, bounds-order) -> design dict."""
    return {k: theta[..., i] for i, k in enumerate(PARAM_NAMES)}


def flat_from_design(d: dict) -> torch.Tensor:
    return torch.stack([d[k] for k in PARAM_NAMES], dim=-1)


def normalize(theta: torch.Tensor) -> torch.Tensor:
    return (theta - LB) / (UB - LB)


def denormalize(x: torch.Tensor) -> torch.Tensor:
    return LB + (UB - LB) * x


# ---------------------------------------------------------------------------
# Natural cubic spline profile
# ---------------------------------------------------------------------------

def _spline_second_derivs(s_knots: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Second derivatives M_i of the natural cubic spline through (s_i, y_i)."""
    n = s_knots.shape[0]
    h = s_knots[1:] - s_knots[:-1]
    A = torch.zeros((n, n), dtype=y.dtype)
    b = torch.zeros(n, dtype=y.dtype)
    A[0, 0] = 1.0
    A[-1, -1] = 1.0
    for i in range(1, n - 1):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2.0 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        b[i] = 6.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
    return torch.linalg.solve(A, b)


def profile_radius(s: torch.Tensor, r_ctrl: torch.Tensor,
                   s_knots: torch.Tensor = S_KNOTS,
                   with_deriv: bool = False):
    """Evaluate the meridian profile r(s) (and dr/ds) anywhere in [0, 1].

    s may be scalar or any shape; r_ctrl is a tensor of N_KNOTS radii.
    """
    M = _spline_second_derivs(s_knots, r_ctrl)
    h = s_knots[1:] - s_knots[:-1]
    s_c = s.clamp(s_knots[0], s_knots[-1])
    idx = torch.searchsorted(s_knots, s_c, right=True) - 1
    idx = idx.clamp(0, r_ctrl.shape[0] - 2)
    hi = h[idx]
    a = (s_knots[idx + 1] - s_c) / hi
    b_ = (s_c - s_knots[idx]) / hi
    r = (a * r_ctrl[idx] + b_ * r_ctrl[idx + 1]
         + ((a ** 3 - a) * M[idx] + (b_ ** 3 - b_) * M[idx + 1]) * hi ** 2 / 6.0)
    if not with_deriv:
        return r
    dr = ((r_ctrl[idx + 1] - r_ctrl[idx]) / hi
          + (-(3 * a ** 2 - 1) * M[idx] + (3 * b_ ** 2 - 1) * M[idx + 1]) * hi / 6.0)
    return r, dr


# ---------------------------------------------------------------------------
# Derived geometry / mass properties
# ---------------------------------------------------------------------------

N_GRID = 64  # integration resolution along the height


class RobotGeometry:
    """Differentiable derived quantities of one design."""

    def __init__(self, d: dict):
        self.d = d
        self.H = d["H"]
        self.t = d["t"]
        self.clr = d["clr"]
        r_ctrl = torch.stack([d[f"r{i}"] for i in range(N_KNOTS)])
        self.r_ctrl = r_ctrl

        s = torch.linspace(0.0, 1.0, N_GRID + 1)
        r, dr = profile_radius(s, r_ctrl, with_deriv=True)
        self.s_grid, self.r_grid, self.dr_grid = s, r, dr
        self.r_in = (r - d["t"]).clamp(min=2.5)
        dz = self.H / N_GRID
        self.dz = dz

        # volumes (mm^3), trapezoid rule
        self.v_out = torch.trapz(torch.pi * r ** 2, s) * self.H
        self.v_in = torch.trapz(torch.pi * self.r_in ** 2, s) * self.H
        self.v_wall = (self.v_out - self.v_in
                       + torch.pi * r[0] ** 2 * d["t"]
                       + torch.pi * r[-1] ** 2 * d["t"])
        self.v_cav = self.v_in  # cavity volume (mm^3)

        # frontal silhouette area of a solid of revolution = int 2 r dz (mm^2)
        self.a_frontal = (torch.trapz(2.0 * r, s) * self.H
                          + 1.2 * torch.pi * d["rw"] ** 2)
        self.surface = torch.trapz(2.0 * torch.pi * r, s) * self.H  # mm^2

        self.r_max = r.max()
        self.r_base = r[0]
        self.r_in_min = self.r_in.min()

        # mass breakdown (g); z measured from the ground (mm)
        z_body = self.clr + s * self.H
        m_shell = RHO_PLA * self.v_wall
        self.masses = {
            # trapz ratio already gives the mean height of the shell in mm
            "shell": (m_shell, torch.trapz(z_body * r ** 2, s)
                      / torch.trapz(r ** 2, s)),
            "battery": (d["mbat"], self.clr + d["zbat"] * self.H),
            "pcb": (torch.as_tensor(M_PCB), self.clr + d["zpcb"] * self.H),
            "speaker": (torch.as_tensor(M_SPK), self.clr + Z_SPK * self.H),
            "head": (torch.as_tensor(M_HEAD), self.clr + Z_HEAD * self.H),
            "drive": (torch.as_tensor(M_DRIVE), d["rw"]),  # axles at wheel centres
        }
        self.mass = sum(m for m, _ in self.masses.values())
        self.z_cm = sum(m * z for m, z in self.masses.values()) / self.mass
        self.h_cm = self.z_cm  # tipover lever

        # support polygon: drive wheels at +-wt/2 (x), casters fore/aft at +-L/2
        r_low = profile_radius(torch.tensor([0.06]), r_ctrl)[0]
        self.castor_span = (2.0 * (r_low - 8.0)).clamp(min=36.0, max=96.0)
        self.a_x = self.castor_span / 2.0   # half-length (front/back tip axis)
        self.a_y = d["wt"] / 2.0            # half-width (side tip axis)

        # wheel axle height fraction (axles sit at z = rw above ground)
        self.s_axle = ((d["rw"] - self.clr) / self.H).clamp(0.01, 0.5)

        # head cavity above the speaker baffle (for the acoustic model), mm^3
        mask = (s >= Z_SPK).float()
        v_head = torch.trapz(torch.pi * self.r_in ** 2 * mask, s) * self.H
        self.v_head = v_head
        d_port = (2.0 * profile_radius(torch.tensor([Z_SPK]), r_ctrl)[0]
                  - 2.0 * d["t"]).clamp(min=8.0, max=26.0)
        self.d_port = d_port
        self.a_port = torch.pi * (d_port / 2.0) ** 2  # mm^2

        # component volumes (cm^3) for packing
        self.v_comp = (V_COMP_PER_G["battery"] * d["mbat"]
                       + sum(V_COMP_FIXED.values()))  # cm^3

    # ---- radial room available at a height fraction (mm) -------------------
    def inner_radius_at(self, s_val: float) -> torch.Tensor:
        sv = torch.tensor([float(s_val)])
        r_at, _ = profile_radius(sv, self.r_ctrl, with_deriv=True)
        return (r_at - self.t).clamp(min=2.5)
