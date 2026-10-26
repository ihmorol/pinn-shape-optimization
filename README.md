# PhORM — Physics-informed Optimization of Robot Morphology

**Shape optimization of a small mobile companion robot with a
physics-informed neural network in the loop.**

This repository is the research companion for a Q1-journal submission on
PINN-based shape optimization of small mobile robots (move / talk / hear).
Instead of optimizing CAD, it optimizes the *geometry and mass layout* of the
robot body — spline control points of the body-of-revolution profile, wall
thickness, wheel geometry, battery mass and height, PCB height, ground
clearance — against a differentiable multi-physics objective:

| Physics | Model | Role in the objective |
|---|---|---|
| Tipover stability | quasi-static ZMP margins over 5 maneuvers (brake, accel, side slope, fore/aft grades) | maximize minimum margin |
| Locomotion energy | rolling resistance + drag + grade force (cost of transport) | minimize J/m |
| Thermal | **shape-conditioned PINN** for steady heat conduction with Robin BCs, warm-started across outer iterations (verified against an independent finite-volume solver) | minimize T_max |
| Acoustics | Helmholtz-resonator model of the speaker cavity (speech-band tuning) | minimize detuning penalty |
| Packaging | cavity volume, component fit, printability wall thickness | hard constraints via penalties |

The design vector is 16 bounded parameters mapped through a sigmoid
(manufacturability-preserving reparameterization); the outer loop is Adam on
the scalar objective with penalty continuation; the inner PINN solves the
thermal field in a reference domain mapped by the profile spline, so PDE
residuals are differentiable with respect to the shape.

## Results so far (autoresearch loop)

See [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md) for the full audited trail
(Karpathy-style: each iteration = hypothesis → experiment → keep only if
better). Note the PINN field solve is stochastic at the ~0.5 °C level
(stock T_max: 54.96 vs 54.32 across full re-runs) — treat third decimals as
noise; the finite-volume solver is the deterministic reference.

| run | basis | J (lower better) | ΔJ | note |
|---|---|---|---|---|
| iter0 stock (reference) | pinn | 1.0000 | — | T_max 55.0 °C (PINN) / 56.3 °C (FV) |
| iter1 geometry+mass co-opt | proxy | 0.8278 | +17.2% | fast physics only, 120 outer steps |
| iter2 PINN in the outer loop | pinn | **0.8499** | **+15.0%** | mass −12%, margin +24%, energy −11% |
| iter3 FDM certification | fdm | — | — | optimized T_max 54.6 °C vs stock 56.3 °C (finite volume) |
| iter4 multi-seed probe | pinn | 0.9801 | +2.0% | fresh seeds σ=0.15 don't beat seed-0 at this budget (honest) |
| iter5 smoothness-constrained | pinn | 0.9094 | +9.1% | curvature cap (max r'' 3114→1246); structural gains kept, acoustic tuning traded |

Verification (stock design, `results/verify_stock.json`): PINN vs
finite-volume T_max −2.0%, T_pcb −1.9%, T_speaker −3.1%, T_battery −5.9%.
On the optimized design (`results/iter3_fdm_crossval.json`): −3.1% /
−2.9% / −4.1% / −10.1%.

## Repository layout

```
src/
  geometry.py       parametric body-of-revolution, spline profile, mass properties
  physics.py        ZMP stability, locomotion energy, packing/fit penalties
  pinn_thermal.py   shape-conditioned thermal PINN (reference-domain mapping)
  verify_fdm.py     independent finite-volume solver for PINN verification
  acoustic.py       Helmholtz resonator model of the speaker cavity
  objectives.py     multi-objective aggregation, normalization, penalties
  optimizer.py      outer differentiable co-optimization loop
experiments/
  run_experiment.py      one config -> baseline + optimization -> results JSON
  autoresearch_loop.py   Karpathy-style loop: run pending, keep only improvements
  configs/iter*.json     iteration hypotheses and hyperparameters
  smoke_test.py          fast-physics sanity check
  smoke_pinn.py          PINN sanity check
docs/
  RESEARCH_LOG.md        auto-generated experiment ledger
  literature_review.md   survey of the field (verified references)
  methodology.md         full mathematical formulation
  perspectives/          raw literature-scout reports
results/                 JSON results, registry, best design, plots
paper/                   draft manuscript
```

## Usage

```bash
pip install -r requirements.txt
python experiments/autoresearch_loop.py          # run all pending iterations
python experiments/autoresearch_loop.py --list   # show registry
python -m src.verify_fdm                         # PINN vs finite-volume check
python experiments/make_figures.py               # publication figures
python experiments/final_table.py                # consolidated results table
```

## Status

Research in progress. Baseline + fast-physics optimization are validated;
PINN-in-the-loop optimization, PINN-vs-FDM verification, multi-seed studies,
figures and the manuscript are being produced by the ongoing autoresearch
loop. Target venues: Structural and Multidisciplinary Optimization,
Engineering with Computers, Mechanical Systems and Signal Processing.
