"""Generate publication figures from results/ JSONs.

Data charts only (matplotlib, Workflow 2 of the academic-plotting skill):
profiles, convergence traces, metric bars, thermal fields. Outputs PDF+PNG
to results/plots/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.geometry import (PARAM_NAMES, RobotGeometry,  # noqa: E402
                          design_from_flat, profile_radius)
from src.optimizer import stock_design  # noqa: E402
from src.pinn_thermal import ThermalPINN, solve_thermal  # noqa: E402

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 10, "legend.fontsize": 8.5, "legend.frameon": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.15, "grid.linestyle": "-",
    "lines.linewidth": 1.8, "lines.markersize": 5,
})
C_STOCK = "#B0BEC5"     # cool gray - recedes
C_OPT = "#E76F51"       # coral - our method
C_ALT = "#264653"       # deep teal
C_ALT2 = "#2A9D8F"
OUT = REPO / "results" / "plots"
OUT.mkdir(parents=True, exist_ok=True)


def _load(name):
    p = REPO / "results" / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _best_params():
    best = _load("best")
    if best is None:
        return None
    return [best["params"][k] for k in PARAM_NAMES]


def fig_profiles():
    base = _load("iter0_baseline")
    theta_b = stock_design().numpy()
    theta_o = _best_params()
    if theta_o is None:
        print("no best design yet, skipping profiles")
        return
    s = torch.linspace(0, 1, 200)
    r_b, dr_b = profile_radius(s, torch.tensor(theta_b, dtype=torch.float32),
                               with_deriv=True)
    r_o, _ = profile_radius(s, torch.tensor(theta_o, dtype=torch.float32),
                            with_deriv=True)
    zb = float(theta_b[7])
    zo = float(theta_o[7])
    fig, ax = plt.subplots(figsize=(3.25, 2.9))
    for r, z, c, lbl in [(r_b, zb, C_STOCK, "stock"),
                         (r_o, zo, C_OPT, "PhORM optimized")]:
        zz = z * s.numpy()
        ax.fill_betweenx(zz, -r.numpy(), r.numpy(), color=c, alpha=0.18,
                         lw=0)
        ax.plot(r.numpy(), zz, color=c, label=lbl)
        ax.plot(-r.numpy(), zz, color=c)
    sc = torch.tensor(theta_o, dtype=torch.float32)[None, :7] \
        .repeat(s.shape[0] // 7 + 1, 1).flatten()[:7]
    s_knots = torch.tensor([0.0, 0.15, 0.32, 0.52, 0.72, 0.88, 1.0])
    r_k, _ = profile_radius(s_knots, torch.tensor(theta_o,
                                                  dtype=torch.float32),
                            with_deriv=True)
    ax.plot(r_k.numpy(), zo * s_knots.numpy(), "o", ms=4, color=C_OPT,
            zorder=5, label="control points")
    ax.set_xlabel("radius r (mm)")
    ax.set_ylabel("height z (mm)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    ax.set_title("Meridian profile: stock vs optimized")
    for ext in ["pdf", "png"]:
        fig.savefig(OUT / f"fig_profiles.{ext}")
    plt.close(fig)


def fig_convergence():
    fig, ax = plt.subplots(figsize=(3.25, 2.5))
    shown = False
    for name, lbl, c in [("iter1_geomopt_fast", "fast physics (iter1)",
                          C_ALT2),
                         ("iter2_pinn_coopt", "PINN-in-loop (iter2)",
                          C_OPT)]:
        res = _load(name)
        if res is None or "trace" not in res:
            continue
        tr = res["trace"]
        it = [t["iter"] for t in tr]
        J = [t["J"] for t in tr]
        ax.plot(it, J, label=lbl, color=c)
        shown = True
    if not shown:
        plt.close(fig)
        return
    ax.axhline(1.0, color="#888", lw=1, ls="--")
    ax.text(ax.get_xlim()[1] * 0.98, 1.0, "stock", va="bottom", ha="right",
            fontsize=8, color="#666")
    ax.set_xlabel("outer iteration")
    ax.set_ylabel(r"penalty-free objective $J$")
    ax.set_title("Co-optimization convergence")
    ax.legend(loc="upper right")
    for ext in ["pdf", "png"]:
        fig.savefig(OUT / f"fig_convergence.{ext}")
    plt.close(fig)


def fig_metrics():
    base = _load("iter0_baseline")
    best = _load("best")
    if base is None or best is None:
        print("need iter0_baseline + best; skipping metrics")
        return
    ob = _load(best["name"]) or {}
    opt_m = ob.get("optimized", base)
    keys = ["e_flat_Jm", "min_margin_mm", "T_max", "mass_g", "z_cm_mm"]
    labels = ["energy\n(J/m)", "stability\nmargin (mm)", r"$T_{max}$ (°C)",
              "mass (g)", "CoM height\n(mm)"]
    base_v = [base["baseline"][k] for k in keys]
    opt_v = [opt_m.get(k, base["baseline"][k]) for k in keys]
    # normalize: for margin, higher is better -> invert so bars show improvement
    norm_b = [1.0] * len(keys)
    norm_o = [opt_v[i] / base_v[i] if k != "min_margin_mm"
              else base_v[i] / max(opt_v[i], 1e-6)
              for i, k in enumerate(keys)]
    x = np.arange(len(keys))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.75, 2.6))
    ax.bar(x - w / 2, norm_b, w, label="stock", color=C_STOCK,
           edgecolor="white", linewidth=0.5)
    ax.bar(x + w / 2, norm_o, w, label="optimized", color=C_OPT,
           edgecolor="white", linewidth=0.5)
    for xi, (b, o) in enumerate(zip(base_v, opt_v)):
        ax.text(xi - w / 2, 1.02, f"{b:.4g}" if b < 10 else f"{b:.3g}",
                ha="center", va="bottom", fontsize=7, color="#555")
        ax.text(xi + w / 2, norm_o[xi] + 0.02,
                f"{o:.4g}" if o < 10 else f"{o:.3g}",
                ha="center", va="bottom", fontsize=7, color=C_OPT,
                fontweight="bold")
    ax.axhline(1.0, color="#888", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("normalized to stock\n(margin inverted: lower = better)")
    ax.set_ylim(0, max(1.35, max(norm_o) * 1.25))
    ax.legend(ncol=2, loc="upper right")
    ax.set_title(f"Multi-physics metrics: {best['name']} "
                 f"({best.get('improvement_pct', 0):+.1f}% objective)")
    for ext in ["pdf", "png"]:
        fig.savefig(OUT / f"fig_metrics.{ext}")
    plt.close(fig)


def fig_thermal_field():
    """PINN meridian temperature field for stock and best design."""
    theta_o = _best_params()
    if theta_o is None:
        print("no best design; skipping thermal field")
        return
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.0), sharey=True)
    for ax, theta, lbl in [(axes[0], stock_design().numpy(), "stock"),
                           (axes[1], np.array(theta_o), "optimized")]:
        torch.manual_seed(0)
        theta_t = torch.tensor(theta, dtype=torch.float32)
        geo = RobotGeometry(design_from_flat(theta_t))
        net = ThermalPINN()
        net, _ = solve_thermal(net, geo, steps=2200, n_col=1200,
                               lbfgs_steps=150)
        ns, nr = 160, 44
        ss = torch.linspace(0, 1, ns).unsqueeze(-1)
        rho = torch.linspace(0, 1, nr).unsqueeze(0)
        with torch.no_grad():
            r_m = (rho * profile_radius(ss, geo.r_ctrl) * 1e-3)
            z_m = geo.clr * 1e-3 + ss * geo.H * 1e-3
            zn = ((z_m - geo.clr * 1e-3) / (geo.H * 1e-3)).repeat(1, nr)
            rn = (r_m / 0.06)
            dT = net.net(torch.stack([rn, zn], dim=-1)).squeeze(-1) * 15.0
        T = 25.0 + dT.numpy()
        zz = (geo.clr + ss.squeeze(-1).numpy() * float(geo.H))
        rr = r_m.numpy() * 1e3
        im = ax.contourf(rr, zz, T, levels=18, cmap="inferno")
        ax.plot(np.concatenate([rr[:, -1], -rr[::-1, -1]]),
                np.concatenate([zz, zz[::-1]]), color="white", lw=1.2)
        ax.set_title(f"{lbl}  ($T_{{max}}$={T.max():.1f} °C)")
        ax.set_xlabel("r (mm)")
        ax.set_aspect("equal")
    axes[0].set_ylabel("z (mm)")
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label("temperature (°C)")
    for ext in ["pdf", "png"]:
        fig.savefig(OUT / f"fig_thermal_field.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    fig_profiles()
    fig_convergence()
    fig_metrics()
    fig_thermal_field()
    print(f"figures written to {OUT}")
