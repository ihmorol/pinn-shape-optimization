"""Acoustic model of the speaker cavity (Helmholtz resonator approximation).

The head cavity above the speaker baffle acts as a Helmholtz resonator:

    f_H = (c / 2 pi) sqrt(A_port / (V L_eff)),  L_eff = t_baffle + 0.85 r_port

Speech-band tuning penalty: the resonance should sit near the vocal-cavity
warmth band (~520 Hz) rather than inside the sibilance band, and the cavity
volume sets the bass rolloff (SPL ~ sqrt(V) below f_H). A shape-conditioned
Helmholtz PINN upgrade path is documented in docs/methodology.md (planned
iteration); this closed model keeps the loop fast.
"""
from __future__ import annotations

import math

import torch

C_SOUND = 343.0          # m/s
F_TARGET = 520.0         # Hz, target tuning frequency
V_REF = 3.0e-5           # m^3, reference head cavity volume


def acoustic(geo) -> dict:
    v_head = geo.v_head * 1e-9                    # m^3
    a_port = geo.a_port * 1e-6                    # m^2
    r_port = torch.sqrt(a_port / math.pi)
    l_eff = 2.0 * geo.t * 1e-3 + 0.85 * r_port    # m
    f_h = C_SOUND / (2.0 * math.pi) * torch.sqrt(a_port / (v_head * l_eff))
    pen = ((f_h - F_TARGET) / F_TARGET) ** 2 \
        + 0.35 * (V_REF / v_head) ** 2
    return {"f_helmholtz_Hz": f_h, "v_head_cm3": v_head * 1e6,
            "acoustic_pen": pen}
