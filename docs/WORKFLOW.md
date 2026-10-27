# Working Process — how this research was produced

The auditable record of the full pipeline: what was done, in what order,
what broke, and how each decision was made. Companion to
[INDEX.md](INDEX.md) (artifact map) and [RESEARCH_LOG.md](RESEARCH_LOG.md)
(experiment ledger).

## Phase 0 — Brief and protocol (frozen before any work)

- Research questions: RQ1 how is physics embedded in neural shape
  optimization; RQ2 which variables/physics dominate small mobile robots;
  RQ3 how do open small-robot projects simulate physics and what gap
  remains. (See [literature_review.md](literature_review.md) §1.)
- Method: deep-research protocol (perspectives → verify → synthesize) for
  literature; Karpathy-style autoresearch (hypothesis → short experiment →
  keep only improvements → ledger) for the implementation — protocol:
  https://github.com/karpathy/autoresearch

## Phase 1 — Literature (4 parallel scout agents, serially re-run on slot limits)

Four general-purpose agents searched and **verified** every reference
(arXiv/publisher confirmation; unconfirmable excluded). Reports:
perspectives A/B/C/D in [perspectives/](perspectives/). Synthesis:
[literature_review.md](literature_review.md). Key external input that
shaped the design: X-CAD (control points through a non-conforming solver),
hPINN (hard constraints + certification), Sun et al. 2023
(surrogate-backprop design loops), the RAR/sampling studies, and the
morphology-co-design gap (bodies are iterated manually — Branch C).

## Phase 2 — Framework implementation

Order of construction (each module smoke-tested with gradient checks before
the next was built on it):

1. `src/geometry.py` — design space, spline profile, mass properties.
2. `src/physics.py` — stability/energy/packing.
3. `src/pinn_thermal.py` — thermal PINN with reference-domain mapping.
4. `src/acoustic.py`, `src/objectives.py`, `src/optimizer.py`.
5. `experiments/run_experiment.py` + `autoresearch_loop.py`.

## Phase 3 — Debugging trail (each bug found by a test, not by luck)

| # | Symptom | Root cause | Fix | Guard added |
|---|---|---|---|---|
| 1 | CoM ≈ 1030 mm, zero margins | spurious `×H` in shell CoM integral | removed | smoke test |
| 2 | margins 7.6 mm vs hand calc 11.4 | fore/aft slopes shifted ZMP laterally | explicit maneuver axis | smoke test |
| 3 | inner-PINN crash "backward twice" | collocation points carried the design graph into the inner loss | detach points to leaves in `solve_thermal` | smoke test |
| 4 | outer loop crash at iteration 2 | `refs` returned as **tensors** → stock-eval graph re-traversed by every outer backward | refs → floats | anomaly detection + removed debug script |
| 5 | **PINN field −17.5% vs FDM, perfectly reproducible** | (a) wrong annulus Jacobian in source normalization; (b) **missing ∇k·∇T interface flux**; (c) unsampled axis region absorbing watts | Jacobian `2πρ r²` term; div-form residual with autograd ∇k; full-ρ sampling + 2kT_rr limit form | FDM cross-verification, global energy-balance checks — `src/verify_fdm.py`, `results/verify_stock.json` |
| 6 | iter2 crash: no feasible design tracked | strict feas_tol with short μ ramp | fallback best-by-(J+100·pen) | optimizer |
| 7 | outer loop stale-geometry training | `geo_det` built once per seed | rebuilt per outer step | — |
| 8 | figure exposed spline wiggle (max r'' 3114) | natural-spline overshoot between control points | profile-curvature penalty (cap 1500) → iter5 | penalty in `physics.py` |

Bugs 3–5 are the scientifically important ones — they are exactly the PINN
pathologies the failure-mode literature warns about, and they were **silent**
(without the independent FV solver, the −17.5% error would have shipped).
The full account is in [methodology.md](methodology.md) §4.1/§4.4 and
paper §6.

## Phase 4 — Autoresearch iterations (the ledger is the record)

| iter | hypothesis (from its config) | outcome |
|---|---|---|
| 0 | stock reference under the full pipeline | J=1.000; T_max 55.0 °C (PINN) / 56.3 °C (FV) |
| 1 | geometry+mass co-opt, fast physics | +17.2% (proxy basis) |
| 2 | PINN thermal in the outer loop changes the shape | **+15.0% (PINN basis) — champion** |
| 3 | champion's field is FDM-certifiable | −3.1% T_max PINN-vs-FDM; 56.28→54.64 °C certified |
| 4 | fresh seeds converge to the same region | J=0.980 — honest negative (budget-limited) |
| 5 | curvature cap costs <2% | costs ~7% (acoustic), smooth profile — reported trade-off |
| 6 | locomotion re-weighting gives a "move-around" optimum | +14.7% under its own weights; energy −13%, margin +31% |

## Phase 5 — Review and reconciliation

A fresh-eyes audit agent cross-checked every number in docs vs
`results/*.json`, claim-vs-evidence gaps, code-doc contradictions, and repo
hygiene. All findings were fixed (verification artifact saved, ledger
patched, parameter count reconciled, stale docstrings, README usage).
Acceptance rule for the loop was further tightened: results on a different
metric basis are recorded but never silently overwrite the champion.

## Phase 6 — How to continue (documented entry points)

1. New hypothesis → drop `experiments/configs/iterN_*.json` (copy any
   config; `weights` are overridable per config) →
   `python experiments/autoresearch_loop.py`.
2. Regenerate figures/tables → `python experiments/make_figures.py`,
   `python experiments/final_table.py`.
3. Certification → `python -m src.verify_fdm` (stock) or mode `crossval`
   (champion).
4. Planned next: shape-conditioned Helmholtz PINN (acoustics), Pareto sweep
   over the objective weights, multi-start + L-BFGS polish for a global
   basin check, and a FreeCAD export of the chosen design.
