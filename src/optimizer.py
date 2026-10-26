"""Outer shape optimizer: differentiable co-optimization of geometry + mass.

The design vector lives in unconstrained space and is mapped through a
sigmoid into the bounded design box (bounds guarantee manufacturable /
assemblable designs). Each outer step:
    design -> geometry -> fast physics -> PINN thermal field -> J -> step
with the PINN warm-started across outer iterations (continuation) and the
constraint penalty weight mu ramped up (penalty continuation method).
"""
from __future__ import annotations

import math
import time

import torch

from . import objectives
from .geometry import (PARAM_NAMES, STOCK, RobotGeometry, design_from_flat,
                       denormalize, flat_from_design, normalize)
from .pinn_thermal import ThermalPINN, solve_thermal, thermal_metrics


def _logit(p: torch.Tensor) -> torch.Tensor:
    return torch.log(p.clamp(1e-4, 1 - 1e-4))


def stock_design() -> torch.Tensor:
    return torch.tensor([STOCK[k] for k in PARAM_NAMES])


def run_optimization(cfg: dict, refs: dict, log=print) -> dict:
    """Optimize the design. cfg keys: seeds, outer_steps, steps_init,
    steps_warm, lr, mu0, mu_end, noise, use_pinn, feas_tol."""
    t_start = time.time()
    use_pinn = cfg.get("use_pinn", True)
    seeds = cfg.get("seeds", [0])
    outer = int(cfg.get("outer_steps", 40))
    mu0, mu1 = float(cfg.get("mu0", 2.0)), float(cfg.get("mu_end", 60.0))
    feas_tol = float(cfg.get("feas_tol", 5e-3))
    noise = float(cfg.get("noise", 0.10))
    x0 = _logit(normalize(stock_design()))

    best = None
    for seed in seeds:
        torch.manual_seed(int(seed))
        if cfg.get("init", "perturb") == "perturb":
            x = (x0 + noise * torch.randn_like(x0)).detach().clone()
        elif cfg.get("init") == "random":
            x = _logit(torch.rand_like(x0) * 0.9 + 0.05).detach().clone()
        else:
            x = x0.clone()
        x.requires_grad_(True)
        opt = torch.optim.Adam([x], lr=float(cfg.get("lr", 0.06)))

        net = None
        geo0 = RobotGeometry(design_from_flat(denormalize(torch.sigmoid(x))))
        if use_pinn:
            net = ThermalPINN()
            net, _ = solve_thermal(net, geo0, steps=int(cfg.get("steps_init", 1200)))

        trace = []
        best_seed = None
        for it in range(outer):
            mu = mu0 * (mu1 / mu0) ** (it / max(1, outer - 1))
            design = denormalize(torch.sigmoid(x))
            geo = RobotGeometry(design_from_flat(design))
            pens = objectives.penalties(geo)
            total_pen = sum(pens.values())
            if use_pinn:
                net, _ = solve_thermal(net, geo,
                                       steps=int(cfg.get("steps_warm", 220)),
                                       lr=1.2e-3)
                th = thermal_metrics(net, geo)
            else:
                th = None
            m = objectives.evaluate(geo, th)
            j = objectives.score(m, refs, mu, total_pen=total_pen)
            opt.zero_grad()
            j.backward()
            opt.step()

            j_clean = float(objectives.score(m, refs, 0.0).detach())
            pen_val = float(total_pen.detach())
            trace.append({"iter": it, "J": j_clean, "pen": pen_val,
                          "mu": mu})
            if pen_val < feas_tol and (best_seed is None
                                       or j_clean < best_seed["J"]):
                best_seed = {"J": j_clean, "theta": [float(v) for v in
                              denormalize(torch.sigmoid(x)).detach()],
                             "iter": it}
            if it % 10 == 0 or it == outer - 1:
                log(f"  seed{seed} it{it:3d} J={j_clean:.4f} pen={pen_val:.2e} "
                    f"mu={mu:.1f}")
        if best_seed is not None and (best is None
                                      or best_seed["J"] < best["J"]):
            best_seed["seed"] = seed
            best = best_seed

    # final clean evaluation of the winning design
    theta = torch.tensor(best["theta"])
    geo_b = RobotGeometry(design_from_flat(theta))
    th = None
    if use_pinn:
        net = ThermalPINN()
        net, _ = solve_thermal(net, geo_b,
                               steps=int(cfg.get("steps_final", 1500)))
        th = thermal_metrics(net, geo_b)
    m_b = objectives.evaluate(geo_b, th)
    best["metrics"] = {k: float(v) for k, v in m_b.items()}
    best["wall_s"] = time.time() - t_start
    best["trace"] = trace
    return best
