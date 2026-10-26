"""Quasi-static physics for a small wheeled companion robot.

Differentiable models of the physical requirements that dominate a ~15 cm,
~400 g desktop robot that moves, talks and hears:

1. Tipover stability - zero-moment-point (ZMP) margins over a maneuver set
   (hard braking, acceleration, side slope, fore/aft grades).
2. Locomotion energy per metre - rolling resistance + aerodynamic drag +
   grade force (cost-of-transport style objective).
3. Packing / fit constraints - components must fit inside the cavity.

Formulations follow standard mobile-robotics practice (Vukobratovic & Borovac
2004 for the ZMP; Alexander 2002 for cost of transport), adapted to
quasi-static planar maneuvers.
"""
from __future__ import annotations

import math

import torch

from .geometry import N_GRID

G = 9.81                    # m/s^2
RHO_AIR = 1.2               # kg/m^3
C_D = 0.95                  # bluff rounded shell
C_RR = 0.03                 # rubber on hardwood
V_CRUISE = 0.35             # m/s

# maneuver set: (name, longitudinal inertial accel m/s^2, slope deg, axis)
# axis is the direction of the ZMP shift caused by the slope ("x" = fore/aft,
# "y" = lateral)
MANEUVERS = [
    ("brake", 2.0, 0.0, "x"),
    ("accel", -1.5, 0.0, "x"),
    ("side_slope", 0.0, 15.0, "y"),
    ("fwd_slope", 0.0, 20.0, "x"),
    ("rear_slope", 0.0, -20.0, "x"),
]


def softplus_sq(x: torch.Tensor) -> torch.Tensor:
    """Smooth one-sided penalty; zero when x <= 0."""
    return torch.nn.functional.softplus(x / 1e-3) ** 2 * 1e-6


def stability(geo) -> dict:
    """ZMP tipover margins (mm) over the maneuver set.

    In a quasi-static maneuver the centre of mass projection shifts by
    dx = h_cm * a / g (inertial) or dx = h_cm * tan(theta) (slope). The margin
    is the distance from the shifted projection to the nearest edge of the
    rectangular support polygon (half-axes a_x = castor span / 2,
    a_y = track / 2). All differentiable w.r.t. the design.
    """
    h_cm = geo.h_cm  # mm
    margins = []
    for _, ax, deg, axis in MANEUVERS:
        sx = h_cm * ax / G if ax != 0.0 else torch.zeros_like(h_cm)
        s_slope = h_cm * math.tan(math.radians(abs(deg))) if deg != 0.0 \
            else torch.zeros_like(h_cm)
        if axis == "x":
            sx = sx + s_slope
            sy = torch.zeros_like(h_cm)
        else:
            sy = s_slope
            sx = sx
        mx = (geo.a_x - sx.abs()).clamp(min=0.0)
        my = (geo.a_y - sy.abs()).clamp(min=0.0)
        margins.append(torch.minimum(mx, my))
    margins = torch.stack(margins)
    min_margin = margins.min()
    tip_angle = torch.rad2deg(torch.arctan(torch.minimum(geo.a_x, geo.a_y) / h_cm))
    return {"min_margin_mm": min_margin, "margins": margins,
            "tip_angle_deg": tip_angle}


def locomotion(geo) -> dict:
    """Specific resistance / energy per metre (J/m = N)."""
    m_kg = geo.mass * 1e-3
    a_f = geo.a_frontal * 1e-6          # m^2
    f_roll = C_RR * m_kg * G
    f_drag = 0.5 * RHO_AIR * C_D * a_f * V_CRUISE ** 2
    e_flat = f_roll + f_drag                                   # J/m
    e_slope = (f_roll * math.cos(math.radians(10))
               + m_kg * G * math.sin(math.radians(10)) + f_drag)
    # cost of transport (dimensionless): E/(m g)
    cot = e_flat / (m_kg * G)
    return {"e_flat_Jm": e_flat, "e_slope10_Jm": e_slope, "cot": cot}


def packing_and_fit(geo) -> dict:
    """Soft feasibility penalties (all zero when the design is feasible)."""
    r_bat = geo.inner_radius_at(float(geo.d["zbat"].detach()))
    r_pcb = geo.inner_radius_at(float(geo.d["zpcb"].detach()))
    r_spk = geo.inner_radius_at(0.86)

    pack_frac = geo.v_comp * 1e3 / geo.v_cav           # cm^3 -> mm^3
    pens = {
        "packing": softplus_sq(pack_frac - 0.55),
        "fit_battery": softplus_sq(21.0 - r_bat),
        "fit_pcb": softplus_sq(23.0 - r_pcb),
        "fit_speaker": softplus_sq(19.0 - r_spk),
        "min_wall_cavity": softplus_sq(6.0 - geo.r_in_min),
        "total_width": softplus_sq(geo.d["wt"] + 16.0 - 110.0),
        "wheel_vs_body": softplus_sq(geo.d["clr"] + 2.0 - geo.d["rw"]),
    }
    # spline overshoot control (manufacturability): the natural spline can
    # oscillate between control points; cap the second derivative of the
    # profile. Stock ~970, the unconstrained optimum reached ~3100 (wiggle).
    ds2 = float(N_GRID) ** 2
    r_ss = (geo.r_grid[2:] - 2.0 * geo.r_grid[1:-1]
            + geo.r_grid[:-2]) * ds2
    pens["profile_curvature"] = softplus_sq((r_ss.abs().max() - 1500.0)
                                            / 300.0)
    pens["pack_frac"] = pack_frac  # report even though not a penalty itself
    return pens
