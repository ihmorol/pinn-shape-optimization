"""Quick smoke test: geometry -> fast physics -> gradients."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src import objectives, physics
from src.geometry import RobotGeometry, design_from_flat
from src.optimizer import stock_design

theta = stock_design()
geo = RobotGeometry(design_from_flat(theta))
m = objectives.evaluate_fast(geo)
for k, v in m.items():
    print(f"{k:18s} {float(v):10.4f}")
pens = physics.packing_and_fit(geo)
print("penalties:", {k: float(v) for k, v in pens.items()})

theta = stock_design().requires_grad_(True)
geo = RobotGeometry(design_from_flat(theta))
m = objectives.evaluate_fast(geo)
pens = physics.packing_and_fit(geo)
J = objectives.score(m, m, 10.0, total_pen=sum(pens.values()))
J.backward()
print("J =", float(J))
print("grad norm =", float(theta.grad.norm()),
      " nan:", bool(theta.grad.isnan().any()))
