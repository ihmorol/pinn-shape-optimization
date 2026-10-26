"""Minimal repro of the outer-loop double-backward crash; bisects causes."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src import objectives
from src.geometry import (N_PARAMS, RobotGeometry, design_from_flat,
                          denormalize, normalize)
from src.optimizer import stock_design
from src.pinn_thermal import ThermalPINN, solve_thermal, thermal_metrics

CFG = dict(outer=3, warm=200, use_lbfgs=True, geo_det_once=True)

x0 = torch.logit(normalize(stock_design()).clamp(1e-4, 1 - 1e-4))
x = torch.nn.Parameter((x0 + 0.10 * torch.randn(N_PARAMS)).detach().clone())
# make refs from stock once
geo0 = RobotGeometry(design_from_flat(stock_design()))
refs = {k: float(v) for k, v in objectives.evaluate_fast(geo0).items()}
net = ThermalPINN()

# exact run_optimization sequence: init solve first
design_live = denormalize(torch.sigmoid(x))
geo_det = RobotGeometry(design_from_flat(design_live.detach()))
net, _ = solve_thermal(net, geo_det, steps=60, lr=5e-3, lbfgs_steps=50)

opt = torch.optim.Adam([x], lr=0.05)
for t in range(CFG["outer"]):
    design = denormalize(torch.sigmoid(x))
    geo = RobotGeometry(design_from_flat(design))
    if not CFG["geo_det_once"]:
        geo_det = RobotGeometry(design_from_flat(design.detach()))
    pens = objectives.penalties(geo)
    total_pen = sum(pens.values())
    net, _ = solve_thermal(net, geo_det, steps=CFG["warm"], lr=1.2e-3,
                           lbfgs_steps=200 if CFG["use_lbfgs"] else 0)
    th = thermal_metrics(net, geo)
    m = objectives.evaluate(geo, th)
    j = objectives.score(m, refs, 2.0 + t, total_pen=total_pen)
    try:
        opt.zero_grad()
        j.backward()
        opt.step()
        print(f"outer {t}: backward ok")
    except RuntimeError as e:
        print(f"outer {t}: FAILED: {e}")
        break
