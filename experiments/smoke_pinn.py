"""Smoke test the thermal PINN: training, metrics, design gradients."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.geometry import RobotGeometry, design_from_flat
from src.optimizer import stock_design
from src.pinn_thermal import ThermalPINN, solve_thermal, thermal_metrics

t0 = time.time()
theta = stock_design().requires_grad_(True)
geo = RobotGeometry(design_from_flat(theta))
net = ThermalPINN()
net, hist = solve_thermal(net, geo, steps=300, record_every=50)
print("loss history:", [f"{v:.3e}" for v in hist])
m = thermal_metrics(net, geo)
for k, v in m.items():
    print(f"{k:14s} {float(v):8.3f}")
J = sum(v for v in m.values())
J.backward()
print("dJ/dtheta norm:", float(theta.grad.norm()),
      "nan:", bool(theta.grad.isnan().any()))
print(f"wall: {time.time()-t0:.1f}s for 300 steps")
