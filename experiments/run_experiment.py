"""Run one experiment from a JSON config.

Modes:
  baseline  - evaluate the stock design with the full pipeline (also produces
              the reference values used to normalize the objective).
  optimize  - evaluate stock -> co-optimize -> evaluate winner -> report.

Writes results/<name>.json and prints a summary table.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from src import objectives  # noqa: E402
from src.geometry import (PARAM_NAMES, STOCK, RobotGeometry,  # noqa: E402
                          design_from_flat)
from src.optimizer import run_optimization, stock_design  # noqa: E402
from src.pinn_thermal import (ThermalPINN, solve_thermal,  # noqa: E402
                              thermal_metrics)


def to_float(d):
    return {k: float(v) for k, v in d.items()}


def evaluate_stock(use_pinn: bool, steps: int = 1500):
    geo = RobotGeometry(design_from_flat(stock_design()))
    th = None
    if use_pinn:
        net = ThermalPINN()
        net, _ = solve_thermal(net, geo, steps=steps)
        th = thermal_metrics(net, geo)
    m = objectives.evaluate(geo, th)
    refs = {k: m[k] for k in
            ["e_flat_Jm", "e_slope10_Jm", "min_margin_mm", "T_max",
             "acoustic_pen"]}
    return to_float(m), refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg_path = Path(args.config)
    cfg = json.loads(cfg_path.read_text())
    name = cfg["name"]
    t0 = time.time()
    torch.set_num_threads(torch.get_num_threads())  # leave default

    use_pinn = bool(cfg.get("use_pinn", True))
    print(f"[{name}] mode={cfg.get('mode')} use_pinn={use_pinn} "
          f"seeds={cfg.get('seeds')}")
    print(f"[{name}] hypothesis: {cfg.get('hypothesis', '-')}")

    base_m, refs = evaluate_stock(use_pinn,
                                  steps=int(cfg.get("steps_init", 1200)))
    j_stock = float(objectives.score(
        {**base_m}, refs, 0.0).detach())
    print(f"[{name}] stock: {objectives.summary_table(base_m)}")
    print(f"[{name}] J(stock) = {j_stock:.4f}")

    out = {"name": name, "hypothesis": cfg.get("hypothesis", ""),
           "config": cfg, "baseline": base_m, "refs": to_float(refs),
           "J_stock": j_stock}

    if cfg.get("mode", "optimize") == "optimize":
        best = run_optimization(cfg, refs, log=lambda s: print(f"[{name}]{s}"))
        j_opt = best["J"]
        out.update({
            "optimized": best["metrics"],
            "J_opt": j_opt,
            "improvement_pct": (j_stock - j_opt) / abs(j_stock) * 100.0,
            "best_seed": best["seed"], "best_iter": best["iter"],
            "params": dict(zip(PARAM_NAMES, best["theta"])),
            "trace": best["trace"], "wall_s": best["wall_s"],
        })
        print(f"[{name}] optimized: "
              f"{ {k: round(v, 3) for k, v in best['metrics'].items()} }")
        print(f"[{name}] J_opt = {j_opt:.4f}  "
              f"({out['improvement_pct']:+.1f}% vs stock)")

    out["wall_s_total"] = time.time() - t0
    res_dir = REPO / "results"
    res_dir.mkdir(exist_ok=True)
    (res_dir / f"{name}.json").write_text(json.dumps(out, indent=2))
    print(f"[{name}] saved results/{name}.json  ({out['wall_s_total']:.0f}s)")


if __name__ == "__main__":
    main()
