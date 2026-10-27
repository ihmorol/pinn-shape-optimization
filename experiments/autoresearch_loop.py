"""Karpathy-style autoresearch loop for the shape-optimization project.

Inspired by github.com/karpathy/autoresearch: run short, self-contained
experiments; keep a design change only if its scalar objective beats the
current best; log every iteration (hypothesis, deltas, accept/reject) so the
improvement curve is auditable.

Usage:
    python experiments/autoresearch_loop.py            # run all pending configs
    python experiments/autoresearch_loop.py --list     # show registry state
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CFG_DIR = REPO / "experiments" / "configs"
RES_DIR = REPO / "results"
REGISTRY = RES_DIR / "registry.json"
BEST = RES_DIR / "best.json"
LOG = REPO / "docs" / "RESEARCH_LOG.md"

METRIC_KEYS = ["e_flat_Jm", "min_margin_mm", "T_max", "f_helmholtz_Hz",
               "acoustic_pen", "mass_g", "z_cm_mm"]
METRIC_LABEL = {"e_flat_Jm": "energy J/m", "min_margin_mm": "stab mm",
                "T_max": "Tmax C", "f_helmholtz_Hz": "f_H Hz",
                "acoustic_pen": "acou pen", "mass_g": "mass g",
                "z_cm_mm": "z_cm mm"}


def load_json(p: Path, default):
    return json.loads(p.read_text()) if p.exists() else default


def append_log(entry: str):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(entry + "\n")


def fmt_metrics(m):
    return "  ".join(f"{METRIC_LABEL[k]}={m.get(k, float('nan')):.2f}"
                     for k in METRIC_KEYS)


def run_pending(only: str | None = None):
    reg = load_json(REGISTRY, {})
    best = load_json(BEST, None)
    cfgs = sorted(CFG_DIR.glob("iter*.json"))
    if not cfgs:
        print("no configs found")
        return
    for cfg_path in cfgs:
        cfg = json.loads(cfg_path.read_text())
        name = cfg["name"]
        if only and only != name:
            continue
        if name in reg and not only:
            print(f"skip {name} (already in registry)")
            continue
        print(f"\n=== running {name} ===")
        print(f"hypothesis: {cfg.get('hypothesis', '-')}")
        t0 = time.time()
        r = subprocess.run([sys.executable, str(REPO / "experiments" /
                                                "run_experiment.py"),
                            "--config", str(cfg_path)])
        if r.returncode != 0:
            print(f"EXPERIMENT FAILED: {name} (rc={r.returncode})")
            reg[name] = {"status": "failed", "wall_s": time.time() - t0}
            REGISTRY.write_text(json.dumps(reg, indent=2))
            continue
        res = load_json(RES_DIR / f"{name}.json", {})
        j_opt = res.get("J_opt")
        mode = cfg.get("mode", "optimize")
        status = mode if mode in ("baseline", "crossval") else "optimize"
        # the metric basis (pinn vs lumped proxy) must match to compare J
        basis = "pinn" if cfg.get("use_pinn", True) else "proxy"
        accepted = False
        basis_note = ""
        if status == "optimize" and j_opt is not None:
            if best is None:
                accepted = True
                best = {"name": name, "J_opt": j_opt, "basis": basis,
                        "improvement_pct": res.get("improvement_pct"),
                        "params": res.get("params")}
                BEST.write_text(json.dumps(best, indent=2))
            elif best.get("basis") != basis:
                # J values under different objective weightings/physics are
                # not comparable: record, keep the existing per-basis best,
                # and never overwrite it silently (results/best.json must
                # stay a meaningful "current champion" for downstream tools)
                basis_note = " (different basis - not compared)"
            elif j_opt < best["J_opt"] - 1e-4:
                accepted = True
                best = {"name": name, "J_opt": j_opt, "basis": basis,
                        "improvement_pct": res.get("improvement_pct"),
                        "params": res.get("params")}
                BEST.write_text(json.dumps(best, indent=2))
        reg[name] = {"status": status, "accepted": accepted, "basis": basis,
                     "J_opt": j_opt,
                     "improvement_pct": res.get("improvement_pct"),
                     "wall_s": time.time() - t0}
        REGISTRY.write_text(json.dumps(reg, indent=2))

        base = res.get("baseline", {})
        opt = res.get("optimized", {})
        if status == "crossval":
            lines = [f"\n## {name} — CERTIFICATION"]
            lines.append(f"- Hypothesis: {cfg.get('hypothesis', '-')}")
            cv = res.get("crossval_optimized", {})
            for k, d in cv.get("diff_pct", {}).items():
                lines.append(f"- {k}: fdm={cv['fdm'][k]:.2f} "
                             f"pinn={cv['pinn'][k]:.2f} ({d:+.1f}%)")
            lines.append(f"- FDM-certified T_max: stock "
                         f"{res.get('crossval_stock_fdm_T_max'):.2f} C -> "
                         f"optimized {cv.get('fdm', {}).get('T_max', float('nan')):.2f} C")
        elif status == "optimize":
            lines = [f"\n## {name} — {'ACCEPTED (new best)' if accepted else 'REJECTED'}"]
            lines.append(f"- Hypothesis: {cfg.get('hypothesis', '-')}")
            lines.append(f"- J (basis={basis}): {res.get('J_stock'):.4f} "
                         f"(stock) -> {j_opt:.4f}  "
                         f"({res.get('improvement_pct'):+.1f}%)"
                         f"  {'**new best**' if accepted else ''}")
            lines.append(f"- stock:    {fmt_metrics(base)}")
            lines.append(f"- optimized:{fmt_metrics(opt)}")
            lines.append(f"- seed {res.get('best_seed')} @ iter "
                         f"{res.get('best_iter')}, wall {reg[name]['wall_s']:.0f}s")
        else:
            lines = [f"\n## {name} — BASELINE"]
            lines.append(f"- Hypothesis: {cfg.get('hypothesis', '-')}")
            lines.append(f"- stock reference metrics (basis={basis}): "
                         f"{fmt_metrics(base)}")
            lines.append(f"- J(stock) = {res.get('J_stock'):.4f}")
        lines.append(f"- Config: `{cfg_path.name}` "
                     f"(use_pinn={cfg.get('use_pinn')}, "
                     f"outer={cfg.get('outer_steps')}, "
                     f"warm={cfg.get('steps_warm')})")
        append_log("\n".join(lines))
        print("\n".join(lines))

    print("\n=== registry ===")
    for k, v in reg.items():
        j = v.get("J_opt")
        j_str = f"{j:.4f}" if isinstance(j, (int, float)) else str(j)
        print(f"{k:28s} {v.get('status'):8s} basis={v.get('basis', '-'):5s} "
              f"J={j_str} "
              f"{'ACCEPTED' if v.get('accepted') else ''}")
    if best:
        print(f"best ({best.get('basis')} basis): {best['name']}  "
              f"J={best['J_opt']:.4f} "
              f"({best.get('improvement_pct'):+.1f}% vs stock)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    if args.list:
        reg = load_json(REGISTRY, {})
        for k, v in reg.items():
            print(k, v)
        return
    run_pending(args.only)


if __name__ == "__main__":
    main()
