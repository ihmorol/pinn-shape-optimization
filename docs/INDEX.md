# PhORM — Master Index

**One-page map of everything in this repository.** Start here; every artifact
is linked from this file. Repository:
https://github.com/ihmorol/pinn-shape-optimization

## 1. Start reading here

| Read | File | What it gives you |
|---|---|---|
| 1st | [../README.md](../README.md) | Project overview, results table, usage |
| 2nd | [WORKFLOW.md](WORKFLOW.md) | The working process: brief → literature → implementation → debugging trail → iterations → how to continue |
| 3rd | [RESEARCH_LOG.md](RESEARCH_LOG.md) | The auditable experiment ledger (hypothesis → metrics → accept/reject, per iteration) |
| 4th | [methodology.md](methodology.md) | Full mathematical formulation of everything implemented |

## 2. Research question & deliverable chain

**Brief:** physics-informed neural shape optimization of a small mobile
companion robot (moves / talks / hears) over **geometric control points and
mass layout — not CAD**, targeting a Q1 journal.

```
research brief (RQ1-3, frozen)
   → 4 literature scout reports (verified references)      docs/perspectives/
   → survey synthesis (survey paper structure)             docs/literature_review.md
   → implementation: geometry/physics/PINN/optimizer       src/
   → autoresearch loop (Karpathy-style ledger)             experiments/ + docs/RESEARCH_LOG.md
   → FDM certification of the winning design               src/verify_fdm.py + results/iter3_fdm_crossval.json
   → manuscript draft + figures                            paper/draft.md, results/plots/
```

## 3. Literature (all references verified Oct 2026, grades A/B)

| File | Perspective | Contents |
|---|---|---|
| [perspectives/perspective_A_pinn_shape_opt.md](perspectives/perspective_A_pinn_shape_opt.md) | Mainstream PINN shape/design optimization | 18 verified papers: Raissi 2019, Deep Ritz, hPINN, Sun 2023 airfoil surrogate+backprop, SDF-conditioned surrogates, Δ-PINNs, GINNs, thermal PINN for power modules |
| [perspectives/perspective_B_neural_topology_optimization.md](perspectives/perspective_B_neural_topology_optimization.md) | Neural topology optimization | 16 verified: Hoyer 2019, TOuNN, NTopo, X-CAD (control points + XFEM), DiffAqua, NeuralTOMO, OpenTO; transferable lessons |
| [perspectives/perspective_C_morphology_codesign.md](perspectives/perspective_C_morphology_codesign.md) | Robot morphology co-design | 19 verified: Lipson & Pollack, RoboGrammar, DiffTaichi, SoftZoo, Dojo, MJX, Papadopoulos & Rey ZMP, Mini Cheetah, Open Duck Mini, Reachy Mini; dominant design variables; gaps |
| [perspectives/perspective_D_pinn_methodology.md](perspectives/perspective_D_pinn_methodology.md) | PINN failure modes + thermal/acoustic practice | 18 verified: Krishnapriyan, Wang×3, RAR, Fourier features, PINNacle/PDEBench, Helmholtz PINNs; practical recipes |
| [literature_review.md](literature_review.md) | Synthesis | Survey paper answering RQ1-3; the open niche PhORM occupies |

## 4. Source code (all differentiable end-to-end, torch CPU)

| Module | Role |
|---|---|
| [../src/geometry.py](../src/geometry.py) | 15-parameter design space; natural-cubic-spline body of revolution; volumes/masses/CoM/support polygon |
| [../src/physics.py](../src/physics.py) | ZMP stability (5 maneuvers), locomotion energy, packing/fit/curvature penalties |
| [../src/pinn_thermal.py](../src/pinn_thermal.py) | **Shape-conditioned axisymmetric thermal PINN** (div-form residual, reference-domain mapping, volume weighting, RAR, warm-start continuation) |
| [../src/verify_fdm.py](../src/verify_fdm.py) | Independent annular finite-volume solver (PINN certification) |
| [../src/acoustic.py](../src/acoustic.py) | Helmholtz speaker-cavity model |
| [../src/objectives.py](../src/objectives.py) | Multi-objective aggregation, baseline normalization, penalties |
| [../src/optimizer.py](../src/optimizer.py) | Outer Adam loop: sigmoid bounds, penalty continuation, feasibility-tracked selection, multi-seed |

## 5. Experiments

| File | Role |
|---|---|
| [../experiments/autoresearch_loop.py](../experiments/autoresearch_loop.py) | Karpathy-style loop: run pending configs, keep per-basis best, write the ledger |
| [../experiments/run_experiment.py](../experiments/run_experiment.py) | One config → stock eval + optimization + report; supports `baseline` / `optimize` / `crossval` modes and per-config `weights` |
| [../experiments/make_figures.py](../experiments/make_figures.py) | Publication figures (profiles, convergence, metrics, thermal fields) |
| [../experiments/final_table.py](../experiments/final_table.py) | Consolidated results table (README/paper source of truth) |
| [../experiments/smoke_test.py](../experiments/smoke_test.py) | Fast-physics sanity check (geometry → physics → gradients) |
| [../experiments/smoke_pinn.py](../experiments/smoke_pinn.py) | PINN sanity check (training, metrics, design gradients) |
| configs: [iter0_baseline](../experiments/configs/iter0_baseline.json) · [iter1_geomopt_fast](../experiments/configs/iter1_geomopt_fast.json) · [iter2_pinn_coopt](../experiments/configs/iter2_pinn_coopt.json) · [iter3_fdm_crossval](../experiments/configs/iter3_fdm_crossval.json) · [iter4_multiseed](../experiments/configs/iter4_multiseed.json) · [iter5_smooth](../experiments/configs/iter5_smooth.json) · [iter6_locomotion](../experiments/configs/iter6_locomotion.json) | Each config = one hypothesis (its text is the research intent) |

## 6. Results & figures

| Artifact | Contents |
|---|---|
| [../results/registry.json](../results/registry.json) | Machine ledger: per-iteration status/basis/J/accepted |
| [../results/best.json](../results/best.json) | Current champion design + parameters (iter2, PINN basis) |
| [../results/verify_stock.json](../results/verify_stock.json) | Stock PINN-vs-FVM verification (−2.0% T_max) |
| [../results/iter0_baseline.json](../results/iter0_baseline.json) | Stock reference metrics |
| [../results/iter1_geomopt_fast.json](../results/iter1_geomopt_fast.json) | +17.2% (proxy basis), 120 outer steps |
| [../results/iter2_pinn_coopt.json](../results/iter2_pinn_coopt.json) | **+15.0% (PINN basis) — champion**; full metric set + trace |
| [../results/iter3_fdm_crossval.json](../results/iter3_fdm_crossval.json) | FDM certification of the champion (56.28→54.64 °C) |
| [../results/iter4_multiseed.json](../results/iter4_multiseed.json) | Multi-seed robustness probe (J 0.980, honest negative) |
| [../results/iter5_smooth.json](../results/iter5_smooth.json) | Smoothness-constrained redesign (max r'' 1246, J 0.909) |
| [../results/iter6_locomotion.json](../results/iter6_locomotion.json) | **Locomotion-weighted optimum** (energy −13%, margin +31%) |
| [../results/plots/fig_profiles.pdf](../results/plots/fig_profiles.pdf) | Meridian profiles: stock vs smooth vs unconstrained optimum |
| [../results/plots/fig_convergence.pdf](../results/plots/fig_convergence.pdf) | Outer-loop objective convergence |
| [../results/plots/fig_metrics.pdf](../results/plots/fig_metrics.pdf) | Normalized multi-physics metric bars |
| [../results/plots/fig_thermal_field.pdf](../results/plots/fig_thermal_field.pdf) | PINN temperature fields, stock vs optimized |

## 7. Manuscript

| File | Contents |
|---|---|
| [../paper/draft.md](../paper/draft.md) | Full Q1-style draft (abstract → related work → formulation → PINN → verification traps → results → limitations). Target venues: Struct. Multidisc. Optim., Eng. with Computers, MSSP |

## 8. External sources consulted during the work

- Karpathy's autoresearch loop (the protocol this repo mirrors):
  https://github.com/karpathy/autoresearch ·
  https://thenewstack.io/karpathy-autonomous-experiment-loop ·
  https://www.datacamp.com/tutorial/guide-to-autoresearch
- Open-hardware robot practice: https://github.com/apirrone/Open_Duck_Mini ·
  https://github.com/pollen-robotics/reachy_mini
- All ~70 verified academic references are inside the four perspective files
  (§3) with arXiv/DOI links and verification grades.
