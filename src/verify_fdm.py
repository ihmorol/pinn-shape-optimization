"""Independent finite-volume solver for the same axisymmetric thermal problem.

Cell-centred FV discretisation of the axisymmetric equation
    k (T_rr + T_r/r + T_zz) + Q''' = 0,   -k dT/dn = h (T - T_inf)
on the meridian half-plane with ANNULAR cells (volume 2 pi r dr dz, face
areas 2 pi r dz / 2 pi r dr), so the r = 0 face has zero area (natural
symmetry). The spline boundary is approximated as a staircase (documented
error source; fine grids keep it small). Used to verify the PINN field
solution - the optimizer's gradients are only as trustworthy as the field
solver.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from .geometry import profile_radius
from .pinn_thermal import H_EFF, K_CAV, K_PLA, SOURCES, T_INF, _source_z


def _profile_np(geo, n: int = 512):
    s = torch.linspace(0, 1, n)
    with torch.no_grad():
        r = profile_radius(s, geo.r_ctrl).numpy() * 1e-3
        r_in = (profile_radius(s, geo.r_ctrl) - geo.t).clamp(min=2.5).numpy() \
            * 1e-3
        z = (float(geo.clr) + s.numpy() * float(geo.H)) * 1e-3
    return z, r, r_in


def fdm_thermal(geo, nr: int = 96, nz: int = 192) -> dict:
    """Solve the axisymmetric thermal field; returns metric dict (floats)."""
    z_p, r_p, r_in_p = _profile_np(geo)
    z_lo, z_hi = z_p[0], z_p[-1]
    r_max = r_p.max() * 1.001
    r_edges = np.linspace(0.0, r_max, nr + 1)
    z_edges = np.linspace(z_lo, z_hi, nz + 1)
    dr, dz = r_edges[1] - r_edges[0], z_edges[1] - z_edges[0]
    rc = 0.5 * (r_edges[:-1] + r_edges[1:])
    zc = 0.5 * (z_edges[:-1] + z_edges[1:])
    r_at_z = np.interp(zc, z_p, r_p)
    r_in_at_z = np.interp(zc, z_p, r_in_p)
    mask = rc[None, :] <= r_at_z[:, None]          # (nz, nr)
    is_wall = rc[None, :] > r_in_at_z[:, None]
    k_cell = np.where(is_wall, K_PLA, K_CAV)
    vol = 2.0 * np.pi * rc[None, :] * dr * dz      # annular cell volumes

    # ring sources, normalized by integral phi 2 pi r dr dz
    RR, ZZ = np.meshgrid(rc, zc)
    q_inj = np.zeros_like(k_cell)                  # W per cell
    for _, p, r_off, sr, sz, key in SOURCES:
        z_c = float(_source_z(key, geo))
        phi = np.exp(-(((RR - r_off) / sr) ** 2) - (((ZZ - z_c) / sz) ** 2))
        phi[~mask] = 0.0
        total = float((phi * vol * mask).sum())
        q_inj += p * phi * vol / total

    n = int(mask.sum())
    idx = -np.ones_like(mask, dtype=np.int64)
    idx[mask] = np.arange(n)
    rows, cols, vals = [], [], []
    b = np.zeros(n)

    def k_harm(a, b_):
        return 2.0 / (1.0 / a + 1.0 / b_)

    for jz in range(nz):
        for jr in range(nr):
            if not mask[jz, jr]:
                continue
            i = idx[jz, jr]
            diag = 0.0
            r_face_in = r_edges[jr]
            r_face_out = r_edges[jr + 1]
            # -r face: axis (zero area) or neighbour
            if jr > 0 and mask[jz, jr - 1]:
                g = k_harm(k_cell[jz, jr], k_cell[jz, jr - 1]) \
                    * (2.0 * np.pi * r_face_in * dz) / dr
                rows.append(i); cols.append(idx[jz, jr - 1]); vals.append(-g)
                diag += g
            # +r face: neighbour or Robin on the outer shell
            if jr + 1 < nr and mask[jz, jr + 1]:
                g = k_harm(k_cell[jz, jr], k_cell[jz, jr + 1]) \
                    * (2.0 * np.pi * r_face_out * dz) / dr
                rows.append(i); cols.append(idx[jz, jr + 1]); vals.append(-g)
                diag += g
            else:
                a_ro = H_EFF * (2.0 * np.pi * r_face_out * dz)
                diag += a_ro
                b[i] += a_ro * T_INF
            # -z face: neighbour or Robin at the bottom plate
            if jz > 0 and mask[jz - 1, jr]:
                g = k_harm(k_cell[jz, jr], k_cell[jz - 1, jr]) \
                    * (2.0 * np.pi * rc[jr] * dr) / dz
                rows.append(i); cols.append(idx[jz - 1, jr]); vals.append(-g)
                diag += g
            else:
                a_ro = H_EFF * (2.0 * np.pi * rc[jr] * dr)
                diag += a_ro
                b[i] += a_ro * T_INF
            # +z face: neighbour or Robin at the top
            if jz + 1 < nz and mask[jz + 1, jr]:
                g = k_harm(k_cell[jz, jr], k_cell[jz + 1, jr]) \
                    * (2.0 * np.pi * rc[jr] * dr) / dz
                rows.append(i); cols.append(idx[jz + 1, jr]); vals.append(-g)
                diag += g
            else:
                a_ro = H_EFF * (2.0 * np.pi * rc[jr] * dr)
                diag += a_ro
                b[i] += a_ro * T_INF
            b[i] += q_inj[jz, jr]
            rows.append(i); cols.append(i); vals.append(diag)

    A = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    T = spsolve(A.tocsc(), b)
    Tf = np.full(mask.shape, np.nan)
    Tf[mask] = T

    out = {"T_max": float(np.nanmax(Tf))}
    for name, _, r_off, _, _, key in SOURCES:
        z_c = float(_source_z(key, geo))
        jz = int(np.clip(np.searchsorted(zc, z_c), 0, nz - 1))
        jr = int(np.clip(np.searchsorted(rc, r_off), 0, nr - 1))
        out[f"T_{name}"] = float(Tf[jz, jr])
    return out


if __name__ == "__main__":
    import json
    import time

    from src.geometry import RobotGeometry, design_from_flat
    from src.optimizer import stock_design
    from src.pinn_thermal import ThermalPINN, solve_thermal, thermal_metrics

    geo = RobotGeometry(design_from_flat(stock_design()))
    t0 = time.time()
    fdm = fdm_thermal(geo)
    t1 = time.time()
    net = ThermalPINN()
    net, hist = solve_thermal(net, geo, steps=2500, lbfgs_steps=250)
    pin = {k: float(v.detach()) for k, v in thermal_metrics(net, geo).items()}
    diffs = {k: 100.0 * (pin[k] - fdm[k]) / fdm[k]
             for k in fdm if k in pin}
    out = {"design": "stock", "fdm": fdm, "pinn": pin, "diff_pct": diffs,
           "fdm_wall_s": t1 - t0, "pinn_final_loss": hist[-1]}
    print(f"FDM  ({t1-t0:.1f}s): ", {k: round(v, 2) for k, v in fdm.items()})
    print(f"PINN (2500+250):    ",
          {k: round(v, 2) for k, v in pin.items()},
          f"final loss {hist[-1]:.3e}")
    for k in fdm:
        if k in pin:
            print(f"{k:12s} fdm={fdm[k]:7.2f}  pinn={pin[k]:7.2f}  "
                  f"diff={100*(pin[k]-fdm[k])/fdm[k]:+5.1f}%")
    (Path(__file__).resolve().parents[1] / "results"
     / "verify_stock.json").write_text(json.dumps(out, indent=2))
    print("saved results/verify_stock.json")
