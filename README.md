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
better).

| Iteration | Change | J (lower better) | Basis |
|---|---|---|---|
| iter0 | stock dtiny-like reference | 1.000 | pinn |
| iter1 | geometry + mass co-opt (fast physics) | 0.828 (+17.2%) | proxy |
| iter2 | PINN thermal in the outer loop | see log | pinn |

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
python src/verify_fdm.py                         # PINN vs finite-volume check
```

## Status

Research in progress. Baseline + fast-physics optimization are validated;
PINN-in-the-loop optimization, PINN-vs-FDM verification, multi-seed studies,
figures and the manuscript are being produced by the ongoing autoresearch
loop. Target venues: Structural and Multidisciplinary Optimization,
Engineering with Computers, Mechanical Systems and Signal Processing.
