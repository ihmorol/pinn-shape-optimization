"""Multi-objective aggregation: raw metrics, normalization, and score.

The scalar objective used by the outer optimizer is a weighted, baseline-
normalized sum plus constraint penalties with a continuation schedule:

    J(theta) = w_E E/E0 + w_S E_s/E0s + w_st S0/S + w_T Tmax/T0
               + w_A A/A0 + mu * sum(penalties)

where the *0 references are the stock-design values evaluated with the same
pipeline, so J(stock) ~= sum of weights (plus its own penalties, which are
zero for the feasible stock design). Lower is better.
"""
from __future__ import annotations

import torch

from . import acoustic as _acoustic
from . import physics
from .pinn_thermal import P_TOT, H_EFF, K_PLA

WEIGHTS = {"energy": 0.25, "slope": 0.15, "stability": 0.25,
           "thermal": 0.20, "acoustic": 0.15}
MARGIN_FLOOR_MM = 10.0


def evaluate_fast(geo) -> dict:
    """All metrics that do not need a PDE solve (thermal uses a lumped proxy)."""
    st = physics.stability(geo)
    lo = physics.locomotion(geo)
    ac = _acoustic.acoustic(geo)
    a_out = geo.surface * 1e-6 + 2.0 * 3.14 * (geo.r_base * 1e-3) ** 2
    dT_lump = P_TOT * (1.0 / (H_EFF * a_out)
                       + (geo.t * 1e-3) / (K_PLA * a_out))
    return {
        "e_flat_Jm": lo["e_flat_Jm"],
        "e_slope10_Jm": lo["e_slope10_Jm"],
        "cot": lo["cot"],
        "min_margin_mm": st["min_margin_mm"],
        "tip_angle_deg": st["tip_angle_deg"],
        "T_max": 25.0 + dT_lump,
        "f_helmholtz_Hz": ac["f_helmholtz_Hz"],
        "acoustic_pen": ac["acoustic_pen"],
        "mass_g": geo.mass,
        "v_cav_cm3": geo.v_cav * 1e-3,
        "pack_frac": physics.packing_and_fit(geo)["pack_frac"],
        "a_frontal_cm2": geo.a_frontal * 1e-2,
        "z_cm_mm": geo.z_cm,
    }


def evaluate(geo, thermal: dict | None = None) -> dict:
    """Full evaluation; `thermal` (from the PINN) replaces the lumped proxy."""
    m = evaluate_fast(geo)
    if thermal is not None:
        m["T_max"] = thermal["T_max"]
        m["T_pcb"] = thermal["T_pcb"]
        m["T_battery"] = thermal["T_battery"]
        m["T_speaker"] = thermal["T_speaker"]
    return m


def penalties(geo) -> dict:
    pens = physics.packing_and_fit(geo)
    st = physics.stability(geo)
    extra = {
        "stability_floor": physics.softplus_sq(
            (MARGIN_FLOOR_MM - st["min_margin_mm"]) / 1.0),
    }
    pens.update(extra)
    pens.pop("pack_frac", None)
    return pens


def _clamp_min(x, v):
    return x.clamp(min=v) if torch.is_tensor(x) else max(float(x), v)


def score(metrics: dict, refs: dict, mu: float = 0.0,
          geo=None, total_pen: torch.Tensor | None = None) -> torch.Tensor:
    w = WEIGHTS
    margin = _clamp_min(metrics["min_margin_mm"], 0.5)
    j = (w["energy"] * metrics["e_flat_Jm"] / refs["e_flat_Jm"]
         + w["slope"] * metrics["e_slope10_Jm"] / refs["e_slope10_Jm"]
         + w["stability"] * refs["min_margin_mm"] / margin
         + w["thermal"] * metrics["T_max"] / refs["T_max"]
         + w["acoustic"] * metrics["acoustic_pen"] / refs["acoustic_pen"])
    if total_pen is not None:
        j = j + mu * total_pen
    return j


def summary_table(metrics: dict) -> str:
    keys = ["e_flat_Jm", "e_slope10_Jm", "min_margin_mm", "tip_angle_deg",
            "T_max", "f_helmholtz_Hz", "acoustic_pen", "mass_g", "z_cm_mm"]
    return "  ".join(f"{k}={float(metrics[k]):.3f}" for k in keys)
