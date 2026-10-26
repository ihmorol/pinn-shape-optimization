# PhORM: Physics-Informed Neural Shape Optimization for Small Mobile Robot Bodies

*Draft manuscript (target: Structural and Multidisciplinary Optimization /
Engineering with Computers / Mechanical Systems and Signal Processing).
All numbers in §7 are produced by the repository's autoresearch loop
(`docs/RESEARCH_LOG.md` is the auditable trail).*

## Abstract

Small companion robots — desktop platforms that move, talk, and hear — are
designed today with a simulation gap: controllers are trained in rigid-body
simulators while the body itself (shell geometry, wall thickness, mass
layout) is iterated manually, because no affordable gradient-bearing model
of the body's coupled physics exists. We present PhORM, a physics-informed
shape-optimization framework that closes this gap at hobbyist compute
budgets. The body is parameterized by 15 bounded geometric and mass-layout
variables — spline control radii of the body of revolution, height, wall
thickness, wheel geometry, battery mass and height, board height, ground
clearance — and evaluated against a differentiable multi-physics objective:
quasi-static tipover stability (ZMP margins over five maneuvers), locomotion
energy per metre, packaging constraints, a Helmholtz model of the speaker
cavity, and the steady thermal field computed by a **shape-conditioned
physics-informed neural network** whose collocation points are mapped
through the profile spline, making the PDE residual differentiable with
respect to the design. The PINN is warm-started across outer iterations
(continuation), trained with volume-weighted residuals, source-biased and
residual-guided (RAR) sampling, and a two-stage Adam + L-BFGS schedule, and
is **verified against an independent annular finite-volume solver**
(−2.0% on peak temperature for the stock design, −5.9% on the smallest
heat source; every headline thermal claim is additionally FDM-certified).
An outer Adam loop with penalty continuation co-optimizes geometry and
mass layout. On a stock companion-robot reference design, the framework
improves the scalar multi-physics objective by 15.0% — energy per metre
−11%, tipover margin +24%, mass −12%, centre-of-mass height −21% — and the
finite-volume-certified peak temperature still drops (56.3 → 54.6 °C)
because the thermal term acts as a soft barrier rather than the dominant
objective. We catalogue the two PINN formulation traps our verification
exposed (missing ∇k·∇T interface flux; unconstrained axis region) and the
autoresearch protocol used to iterate the framework.

**Keywords** — physics-informed neural networks; shape optimization;
robot morphology; co-design; thermal management; multi-objective design

## 1. Introduction

Companion robots couple three heterogeneous subsystems in one small shell:
locomotion (wheels, servos), computation (always-on SBC, mic array), and an
acoustic user interface (speaker, microphone). Their bodies are
3D-printed, consumer-priced, and revised frequently. Open-hardware projects
exemplify the current practice: Open Duck Mini trains its controllers in
MuJoCo Playground (MJX) with sim-to-real deployment, but iterates its shell
by hand; Reachy Mini ships a MuJoCo SDK for behavior prototyping while its
expressive shell and speaker/mic placement are design intuition. The
simulator optimizes *control for a frozen body*; the body's own physics —
where the electronics' heat goes, how the enclosure tunes the voice, how
mass layout governs tipover — is outside any optimization loop.

Meanwhile, physics-informed machine learning has made single-physics shape
optimization routine in adjacent domains: PINN-based airfoil design
backprops design gradients through a residual-trained flow surrogate (Sun
et al. 2023); hPINN solves heat-conduction topology optimization with
hard-constraint networks (Lu et al. 2021); parameterized thermal PINNs
accelerate power-module design (Yang et al. 2025). Neural topology
optimization (Hoyer et al. 2019; TOuNN; NTopo; X-CAD) showed that
differentiating through the physics solve yields adjoint-quality design
gradients without hand-derived adjoints. What does not exist — confirmed by
our survey of ~70 verified works (`docs/literature_review.md`) — is a
**multi-physics shape co-optimizer for an assembled robot body**: thermal,
acoustic, stability, and energetic objectives sharing one differentiable
loss over geometric control points and mass layout.

This paper contributes:

1. **A design-space formulation for companion-robot bodies** — 15 bounded
   geometric/mass-layout variables (spline control points of the body of
   revolution, not CAD features) with exact bounds via sigmoid
   reparameterization, and differentiable derived quantities (volumes,
   surface, frontal area, mass breakdown, CoM, support polygon).
2. **A shape-conditioned thermal PINN** on the axisymmetric meridian
   domain, mapped by the profile spline so collocation points live in a
   fixed reference square; trained with volume-weighted residuals,
   source-biased + RAR sampling, and Adam→L-BFGS; warm-started across
   outer iterations; **verified against an independent finite-volume
   solver** — including two formulation traps we exposed and fixed
   (§6).
3. **A differentiable multi-physics objective** joining the PINN thermal
   field with ZMP stability, locomotion energy, packing, and acoustic
   tuning, optimized by penalty-continuation Adam with feasibility-tracked
   selection.
4. **An auditable autoresearch protocol** (Karpathy-style: hypothesis →
   experiment → keep-only-if-better ledger) that produced the results and
   doubles as the paper's provenance record.
5. **Quantified results**: +15.0% on the normalized multi-physics
   objective over a stock companion-robot design, with per-metric
   decompositions, FDM certification of the winning design, and multi-seed
   robustness probes that honestly delimit what the outer-loop budget
   does and does not guarantee.

## Figures

- `results/plots/fig_profiles.pdf` — stock vs. optimized meridian profile
  with spline control points.
- `results/plots/fig_convergence.pdf` — penalty-free objective vs. outer
  iteration (fast physics vs. PINN-in-loop).
- `results/plots/fig_metrics.pdf` — normalized multi-physics metrics.
- `results/plots/fig_thermal_field.pdf` — PINN meridian temperature
  fields, stock vs. optimized.

## 2. Related work

**PINN-based shape/design optimization.** Raissi et al. introduced the PDE
residual loss; Deep Ritz reformulated it variationally. For design, hPINN
(Lu et al. 2021) applies hard-constraint PINNs to heat-conduction topology
optimization and validates against adjoints; Sun et al. (2023) train a
single parametric PINN on parameterized airfoils and backprop design
gradients through it (surrogate-and-optimize); Wassing et al. (2024) couple
shape through reference-domain mappings for transonic flow; SDF-conditioned
surrogates (Zhang et al. 2024) make the geometry channel implicit. All are
single-physics; the device-level nearest neighbor is the parameterized
thermal PINN for power modules (Yang et al. 2025).

**Neural topology optimization.** Hoyer et al. (2019) — neural
reparameterization with solver-in-the-loop gradients; TOuNN (Chandrasekhar
& Suresh 2021) — "TO as deep learning"; NTopo (Zehnder et al. 2021) —
mesh-free implicit representations; X-CAD (Hafner et al. 2019) — B-spline
control points optimized through XFEM on a fixed background mesh, the
closest precedent for control-point design variables; NeuralTOMO (Liu et
al. 2025) — co-optimization with manufacturability as differentiable
losses. Critical benchmarks (Woldseth et al. 2022; Sanu et al. 2024)
recommend keeping the physics solver in the loop rather than replacing it
with a network.

**Robot morphology co-design.** Body–brain co-evolution (Lipson & Pollack
2000; Cheney et al. 2018), grammar-based search (RoboGrammar, Zhao et al.
2020), differentiable simulation for soft-robot co-design (DiffTaichi,
SoftZoo, DiffAqua), control-aware gradients (De Vincenti et al. 2021), and
mass-budget co-design (Jeger et al. 2024). These embed rigid-body contact
physics only; the body's own PDEs (heat, sound, shell elasticity) are
absent — the gap PhORM addresses.

**PINN failure modes.** Krishnapriyan et al. (2021); Wang-Yu-Perdikaris
(2022) and Wang-Teng-Perdikaris (2021) (loss stiffness, adaptive weights);
RAR and its systematic study (Lu et al. 2021; Wu et al. 2023); spectral
bias and Fourier features (Rahaman 2019; Tancik 2020); benchmarks
(PINNacle, PDEBench) warning that PINN forward accuracy trails classical
solvers — hence our certification protocol.

## 3. Design space: geometry and mass layout as variables

The body is a solid of revolution. Its meridian profile $r(s)$, height
fraction $s\in[0,1]$, is a natural cubic spline through seven control
radii at fixed knots; overall height $H$, wall thickness $t$, wheel radius
$r_w$, track width $w_t$, battery mass $m_b$, battery and board heights
$z_b, z_p$, and ground clearance $c$ complete the vector
$\theta \in \mathbb{R}^{15}$ (bounds in `docs/methodology.md`). Bounds are
enforced exactly by $\theta = \theta_{\min} + (\theta_{\max} -
\theta_{\min})\,\sigma(x)$ — the manufacturability-preserving analog of
projection in neural TO. All derived quantities are differentiable: shell
and cavity volumes (trapezoid integration), frontal silhouette
$A_f = \int 2r\,dz + 1.2\pi r_w^2$ (exact for solids of revolution),
surface area, per-component mass breakdown with battery $m_b$ at $z_b$,
board 45 g at $z_p$, speaker 32 g at $0.86H$, head electronics 12 g at
$0.95H$, drive units 90 g at axle height, centre-of-mass height
$h_{cm}$, and the support polygon (track/2 laterally; castor span derived
from the base radius longitudinally — footprint and stability are coupled
through the geometry).

## 4. Multi-physics objective

$$J(\theta) = 0.25\frac{E}{E_0} + 0.15\frac{E_s}{E_{s,0}}
+ 0.25\frac{S_0}{S} + 0.20\frac{T_{\max}}{T_{\max,0}}
+ 0.15\frac{A}{A_0} + \mu \textstyle\sum_c P_c ,$$

with references $(\cdot)_0$ from the stock design under the same pipeline
(so $J_{stock} \approx 1$), and smooth one-sided penalties $P_c$ for
packing (component volume ≤ 55% of cavity), per-component radial fit,
minimum cavity radius, and total width.

**Stability** — ZMP margins over five quasi-static maneuvers (hard braking
2 m/s², acceleration −1.5 m/s², 15° side slope, 20° fore/aft grades):
$S = \min_m \min(a_x - |d_x|, a_y - |d_y|)$ with $d = h_{cm}\,a/g$ or
$h_{cm}\tan\alpha$ (Papadopoulos & Rey 1996; Vukobratović & Borovac 2004).

**Energy** — specific force at cruise: $E = c_{rr}Mg +
\frac12\rho_a C_D A_f v^2$ (J/m), $E_s$ adds the 10° grade term;
cost of transport reported alongside.

**Acoustics** — the head cavity as a Helmholtz resonator:
$f_H = \frac{c}{2\pi}\sqrt{A_{port}/(V_{head}L_{eff})}$; penalty pushes
$f_H$ toward the 520 Hz vocal-warmth band and rewards cavity volume.

## 5. The shape-conditioned thermal PINN

Steady axisymmetric conduction on the meridian half-plane with Robin
exterior BCs ($h_{eff} = 11.5$ W/m²K, convection + linearized radiation),
piecewise conductivity (PLA wall 0.13; lumped cavity 0.5 W/mK), and three
axisymmetric gaussian ring sources (PCB 2.2 W, battery 0.3 W, speaker
1.0 W) normalized so $\int Q'''_i\,2\pi r\,dr\,dz = P_i$ exactly. Collocation
points live in a fixed reference square and map through the spline
($r = \rho\, r(s)$) — the network sees a fixed box while the domain
deforms with the design. 30% source-biased points; every 150 steps one
third of the batch is re-selected as the highest $|R|\,V$ points from a 4×
pool (RAR-style). The loss is **volume-weighted** — the MSE approximates
the domain integral of $\hat R^2$, which we found necessary to keep global
energy balance (a plain pointwise MSE lets high-volume wall regions carry
large unpenalized artifacts). Training: Adam (cosine LR, 2 500 steps) then
200 L-BFGS iterations on a fixed large batch. The network is warm-started
across outer iterations (continuation, 200 inner steps per outer step);
outer gradients flow through a fresh live-graph evaluation while the inner
loop trains on a detached copy of the geometry.

## 6. Verification — and two traps the verification exposed

An independent cell-centred **annular finite-volume** solver (volumes
$2\pi r_c \Delta r \Delta z$, harmonic-mean face conductances, Robin
faces, natural axis symmetry; sparse direct solve, 0.2 s) certifies the
PINN. Getting the two solvers to agree exposed two silent-failure modes we
document for reuse:

1. **The ∇k·∇T interface term is mandatory.** The strong form is the
   divergence form $\nabla\cdot(k\nabla T) + Q''' = k\nabla^2 T +
   \nabla k\cdot\nabla T + Q'''$. Omitting $\nabla k\cdot\nabla T$ (the
   natural "just Laplacian" shortcut) leaves the wall/cavity interface as
   an unconstrained flux source/sink: the network satisfies the pointwise
   loss yet the field violates global energy conservation by watts, and
   peak temperatures come out far too low. With the term included, PINN
   and FV agree on the stock design to −2.0% ($T_{\max}$), −1.9%
   ($T_{pcb}$), −3.1% ($T_{spk}$) and −5.9% (the small battery blob;
   `results/verify_stock.json`) — reproducibly across seeds.
2. **The axis must be constrained.** Sampling only $r \ge 2\%$ of the
   local radius admits an absorbing artifact column at the axis; below
   $r = 1.5$ mm the residual switches to the limit form
   $k(2T_{rr} + T_{zz}) + Q'''$, and collocation covers $r = 0$.

These are instance of the known PINN failure taxonomy (loss pathologies,
unconstrained regions) but, to our knowledge, not previously reported for
piecewise-conductivity domain-parameterized PINNs inside design loops.

## 7. Experiments and results

All experiments are configs in `experiments/configs/`, executed by the
autoresearch loop; the ledger `docs/RESEARCH_LOG.md` records hypothesis,
metrics, and accept/reject per iteration. Baseline references use the full
PINN pipeline on the stock design.

**iter0 (baseline).** Stock dtiny-like design (105 mm tall, 394 g, 2.4 mm
walls): energy 0.116 J/m, ramp 0.785 J/m, margin 11.4 mm, tip 29.1°,
$T_{max}$ 55.0 °C (PINN; FV 56.3 °C), $f_H$ 2279 Hz, CoM 39.6 mm. $J = 1$.

**iter1 (fast-physics co-optimization, no PINN).** 120 outer Adam steps
with the lumped thermal proxy: **+17.2%** on the proxy basis — mass 350 g,
margin 15.0 mm, CoM 31.0 mm, energy 0.104 J/m. Establishes that geometry +
mass layout alone captures most of the gain.

**iter2 (PINN in the outer loop).** 80 outer steps, warm-started
continuation (200 inner PINN steps each), penalty continuation μ: 2→120:
**+15.0% on the PINN basis** ($J$ 1.000 → 0.850). Metrics: energy 0.103
J/m (−11%), margin 14.1 mm (+24%), mass 348 g (−12%), CoM 31.4 mm (−21%),
$T_{max}$ 53.0 °C (−2.0 °C, PINN), acoustic penalty 12.3 → 8.1, tip angle
37.6°. The optimizer spends the thermal headroom on mass/stability/energy:
the thermal term acts as a soft barrier rather than the dominant objective
— the physically sensible trade for a desk robot.

**iter3 (certification).** The winning design re-solved by the finite-
volume solver: T_max −3.1%, T_pcb −2.9%, T_speaker −4.1% PINN-vs-FDM
agreement on the optimized shape (the small battery source is harder at
−10.1% — a known weakness of narrow-source resolution, reported honestly).
The FDM-certified peak temperature drops from 56.28 °C (stock) to
54.64 °C (optimized): the optimization gain is physical, not a PINN
artifact (`results/iter3_fdm_crossval.json`).

**iter4 (robustness probe).** Two fresh seeds with larger perturbation
(σ = 0.15, 60 outer steps each) reach J = 0.980 — i.e. they recover most
of the gain but do not beat the seed-0 winner (0.850) at this budget.
Honest interpretation: at 80 outer steps the outer loop is
initialization-sensitive; the winner reflects both the optimization and a
favorable start near the stock design. Longer schedules or multi-start
selection close the gap (`results/iter4_multiseed.json`).

**iter5 (manufacturability refinement).** Figure diagnostics of the it2
winner exposed spline overshoot between control points (max |r″| = 3114
vs. the stock profile's 973 — a visible wiggle). Adding a
profile-curvature cap (1500) and re-optimizing yields a smooth, printable
profile (max |r″| = 1246) at J = 0.909: every structural metric is
retained (energy 0.102 J/m, margin 14.1 mm, mass 346 g, CoM 31.3 mm) and
the cost falls entirely on acoustic tuning (penalty 8.1 → 13.0). The
unconstrained (smoother-objective) and smoothness-constrained designs are
both reported; the choice is an engineering decision, not a hidden
default (`results/iter5_smooth.json`, profile overlay in
`results/plots/fig_profiles.pdf`).

## 8. Limitations

The thermal model is per-unit-depth axisymmetric with a lumped cavity
conductivity (trend-level, not CFD); acoustics is a closed Helmholtz model
(a shape-conditioned Helmholtz PINN is the planned upgrade); structural
stiffness and drop survival are not yet in the loop; stability assumes a
rectangular support polygon and quasi-static maneuvers; source powers are
nameplate estimates. The scalarization is fixed-weight; Pareto exploration
(generate-then-select per TOM/OpenTO) is future work.

## 9. Conclusion

PhORM demonstrates that a verified shape-conditioned PINN, warm-started
inside a penalty-continuation outer loop over spline control points and
mass layout, can co-optimize the coupled stability/energy/thermal/acoustic
performance of a small companion robot on CPU in minutes per run — turning
the manual print-test-reprint body loop of open robot projects into a
gradient-based, physics-bearing, auditable process. The verification-first
protocol (independent FV solver, global energy balance, feasibility-tracked
selection) is the transferable methodological contribution: it is what
makes PINN-in-the-loop design results trustworthy.

## References

See `docs/literature_review.md` and the four verified perspective reports
in `docs/perspectives/` (60+ verified sources with links and grades).
