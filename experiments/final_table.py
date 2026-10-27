"""Consolidated results table for README / paper (reads results/*.json)."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"

ORDER = ["iter0_baseline", "iter1_geomopt_fast", "iter2_pinn_coopt",
         "iter3_fdm_crossval", "iter4_multiseed", "iter5_smooth",
         "iter6_locomotion"]


def load(name):
    p = RES / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def fmt(v, nd=3):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}" if abs(v) < 100 else f"{v:.0f}"
    return str(v)


def main():
    rows = []
    for name in ORDER:
        res = load(name)
        if res is None:
            continue
        cfg = res.get("config", {})
        mode = cfg.get("mode", "optimize")
        if mode == "baseline":
            b = res["baseline"]
            rows.append(("iter0 stock (reference)", "pinn", "1.0000",
                         "-", fmt(b.get("e_flat_Jm")), fmt(b.get("min_margin_mm")),
                         fmt(b.get("T_max"), 1), fmt(b.get("mass_g"), 0),
                         fmt(b.get("z_cm_mm"), 1)))
        elif mode == "crossval":
            cv = res.get("crossval_optimized", {})
            rows.append(("iter3 FDM certification", "fdm", "-",
                         "-", "-", "-",
                         f"{cv.get('fdm', {}).get('T_max', 0):.1f} (PINN "
                         f"{cv.get('pinn', {}).get('T_max', 0):.1f})",
                         "-", "-"))
        else:
            opt = res.get("optimized", {})
            j = res.get("J_opt")
            imp = res.get("improvement_pct")
            basis = "pinn" if cfg.get("use_pinn", True) else "proxy"
            rows.append((name, basis, fmt(j, 4),
                         f"{imp:+.1f}%" if imp is not None else "-",
                         fmt(opt.get("e_flat_Jm")),
                         fmt(opt.get("min_margin_mm")),
                         fmt(opt.get("T_max"), 1),
                         fmt(opt.get("mass_g"), 0),
                         fmt(opt.get("z_cm_mm"), 1)))
    print("| run | basis | J | ΔJ | energy J/m | margin mm | Tmax °C | mass g | CoM mm |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print("| " + " | ".join(r) + " |")
    cv = load("iter3_fdm_crossval")
    if cv:
        cvv = cv.get("crossval_optimized", {})
        print("\nPINN vs finite-volume on the optimized design:")
        for k, d in cvv.get("diff_pct", {}).items():
            print(f"- {k}: FDM {cvv['fdm'][k]:.2f} / PINN {cvv['pinn'][k]:.2f}"
                  f" ({d:+.1f}%)")


if __name__ == "__main__":
    main()
