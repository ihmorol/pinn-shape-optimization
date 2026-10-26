# Physics-Informed Neural Networks for Shape Optimization of Small Mobile Robot Bodies — A Survey

*Synthesis of four verified literature-scout reports (`docs/perspectives/`),
compiled October 2026. Every reference in the perspective files was verified
against arXiv, publisher, or project pages (grades A/B; unconfirmable works
excluded). This document follows the survey structure: questions first,
evidence second, conclusions that answer the questions.*

## Abstract

Shape optimization with physics-informed neural networks (PINNs) has matured
in single-physics, single-surface domains — airfoils, heat-conduction
topologies, power modules — while robot morphology co-design has matured with
a different toolset: differentiable rigid-body simulators and
reinforcement-learning-in-the-loop search. We survey both lines and the
bridge between them, organized around three questions: (RQ1) how do current
methods embed physical law into neural shape optimization; (RQ2) which design
variables and physics dominate small mobile robot performance; (RQ3) how do
open small-robot projects introduce the physical world into their design
loops, and what gap remains. The synthesis finds a concrete, unoccupied
niche — multi-physics (thermal, acoustic, stability, energetic) shape
co-optimization of assembled robot bodies with geometric control points as
design variables and a shape-conditioned PINN as the differentiable field
solver — and characterizes the methodological safeguards (verification
against classical solvers, adaptive sampling, warm-started continuation)
that the PINN-failure literature shows are mandatory.

## 1. Introduction and research questions

Small companion robots (10–50 cm, consumer-priced, always-on: they move,
talk, hear) are built today by open-hardware communities whose design loop
for the *body* is manual: print, test, reprint. The controller, by contrast,
is trained in physics simulators (MuJoCo/MJX, Isaac). The body's coupled
physics — where heat from always-on compute goes, how the speaker's enclosure
tunes the voice, how mass layout governs tipover — is either ignored or
handled by rule-of-thumb. This survey maps the research landscape that could
close that loop, guided by:

- **RQ1.** How do current methods embed physical laws (PDEs, dynamics,
  stability) into neural shape optimization, and which formulations are most
  effective?
- **RQ2.** Which design variables and physical phenomena dominate small
  mobile robot performance, and how has morphology been co-optimized with
  behavior?
- **RQ3.** How do open small-robot projects introduce physics into
  simulation, and what gap can PINN-based shape optimization fill?

## 2. Method

Four scouting perspectives ran independently with wide-then-narrow searches
and per-reference verification (grade A = confirmed on arXiv/publisher;
B = two independent secondary confirmations; unconfirmable excluded):
(A) the mainstream PINN-based shape/design optimization school; (B) the
adjacent field of neural topology optimization and generative structural
design; (C) the application school of robot morphology co-design and
differentiable simulation; (D) the methodology critics — PINN failure modes
— plus thermal/acoustic PINN practice and benchmark evidence. The verified
corpus spans ~70 works from 1996–2026. Full annotated tables with links:
`docs/perspectives/*.md`.

## 3. Branch I — How physics enters neural shape optimization (RQ1)

**Three embedding mechanisms recur.** (i) *PDE-residual losses*: the network
ansatz is trained so the PDE residual, evaluated by autodiff at collocation
points, vanishes (Raissi et al. 2019); in inverse/design mode the unknown
parameters join the network weights in one gradient descent. (ii)
*Variational/energy forms*: the Deep Ritz method (E & Yu 2018) and
hPINN (Lu et al. 2021) minimize an energy or compliance integral — better
conditioned than pointwise residuals and the preferred route for structural
problems. (iii) *Solver-in-the-loop autodiff*: neural topology optimization
(Hoyer et al. 2019; TOuNN; NTopo) and differentiable simulation (DiffTaichi,
Brax, Dojo, MJX) differentiate *through* a real FEA or multibody solve —
exact adjoint-quality gradients without hand-derived adjoints.

**The shape-to-network coupling is the pivotal design choice.** Four
mainstream patterns: shape parameters as extra network inputs (CST/Bézier
airfoil families; Sun et al. 2023's simultaneous surrogate-and-optimize);
reference-domain mappings so the network sees a fixed box while the domain
deforms (Wassing et al. 2024 — the pattern PhORM uses); SDF/DeepSDF-style
implicit geometry conditioning (Zhang et al. 2024; Kraus 2024); and
geometry-aware encodings (Fourier features, Laplace–Beltrami eigenfunctions
in Δ-PINNs). Hard-constraint ansätze (hPINN's distance-function BCs) remove
loss-balancing pathologies that otherwise stall gradient-based design
iterations.

**Evidence quality is uneven.** The PINN shape-optimization evidence base is
2-D, single-physics, and airfoil-dominated; the closest device-level analog
is the parameterized 3-D thermal PINN for power modules (Yang et al. 2025,
IEEE T-PEL). Neural-TO benchmarks (Sanu et al. 2024 "good, bad, ugly";
Woldseth et al. 2022) caution that NN parameterization reshapes the
optimization landscape and that replacing the state solver with a network
remains fragile — solver-in-the-loop is the defensible lane.

## 4. Branch II — What matters on a small mobile robot (RQ2)

The morphology co-design literature establishes the dominant variables and
their physics:

- **Support geometry and CoM height** govern tipover: the ZMP/force-angle
  family (Papadopoulos & Rey 1996; Vukobratović & Borovac 2004) makes the
  margin an explicit, differentiable function of track width, castor span,
  and mass layout — battery placement is the strongest lever because it is
  the heaviest movable mass.
- **Mass and mass distribution** enter every energetic metric; cost of
  transport (specific resistance) ε = P̄/(mgv̄) sits at 0.1–0.5 for wheeled
  platforms, and mass-budget co-design is explicit in space-robot
  differentiable co-design (Jeger et al. 2024) and the MIT Cheetah actuator
  methodology (Wensing et al. 2017).
- **Wheel radius trades mobility against CoM height and torque** (r vs. step
  height vs. motor torque) — recurring in rover design studies (DLR 2013;
  Li et al. 2022).
- **Shell surface and drag** matter at speed (AUV hull studies: 3.7–4.2%
  drag reduction), but at companion-robot speeds drag is negligible while
  the *same surface area governs convective cooling* — a thermal-PDE
  variable the co-design literature ignores.
- **Sensor placement** (camera FOV, microphone arrays) is task-critical and
  non-convex; head-shape baffle effects directly modulate acoustic
  localization (DLR Justin), and MEMS mic performance depends on cavity
  design — formalizable as acoustic-structure objectives.
- **Manufacturability** (printability, min wall, BOM cost) bounds
  realizable morphologies; NeuralTOMO (SIGGRAPH 2025) shows these enter as
  differentiable losses.

Co-optimization methods that join morphology with behavior span evolutionary
co-evolution (Lipson & Pollack 2000; Cheney et al. 2018), graph-grammar
search with RL scoring (RoboGrammar 2020), differentiable simulation
(SoftZoo 2023; DiffAqua 2021), and control-aware optimal control gradients
(De Vincenti et al. 2021). All of them embed *rigid-body contact* physics —
none embeds the PDEs of the body itself (heat, sound, elasticity of the
shell).

## 5. Branch III — Practice, benchmarks, and failure modes (RQ1/RQ3)

**PINN failure modes with established fixes** (Krishnapriyan 2021;
Wang-Yu-Perdikaris 2022; Wang-Teng-Perdikaris 2021; McClenny & Braga-Neto
2023; Wu et al. 2023 sampling study): (1) loss-term stiffness → adaptive
weighting (learning-rate annealing, NTK balancing, self-adaptive point
weights); (2) spectral bias → Fourier features matched to the wavenumber;
(3) collocation coverage → residual-based adaptive refinement (RAR), with
the caveat that uniform/grid sampling remains best for smooth fields and
RAR pays off near sources/singularities; (4) BC neglect → hard-constraint
ansätze or PML losses; (5) the honest ceiling — PINNacle/PDEBench show PINNs
still trail classical solvers on most *forward* problems, so the defensible
role inside optimization is meshless differentiability, certified by a
classical solver.

**What the small-robot practice does today** (RQ3): Open Duck Mini (42 cm
bipedal duck, <$400 BOM) trains controllers in MuJoCo Playground/MJX with
sim-to-real ONNX deployment, but iterates its body by hand; Reachy Mini ships
a MuJoCo SDK simulation for behavior prototyping while its expressive shell,
speaker enclosure, and mic placement are design intuition; MIT Mini Cheetah
and ODRI/Solo established open, simulation-friendly hardware platforms. In
every case the simulator optimizes *control for a frozen body* — the
co-design loop is broken at the geometry step, and no affordable
gradient-bearing physics model of the shell exists at all.

**Benchmarks.** PDEBench (2022) and PINNacle (2024) score forward-solve
accuracy only; no benchmark measures design-trajectory quality or
∂J/∂shape consistency of PINN-based loops vs. classical adjoints — the
metrics that decide whether PINNs belong in a design loop. This is both a
gap and an opportunity for any new framework to publish its own verification
protocol.

## 6. Cross-branch synthesis: the open niche

Intersection of the three branches:

1. Mainstream PINN shape optimization = single-physics, single-surface,
   mostly aero/thermal (Branch I).
2. Robot co-design = rigid-body contact physics only; body PDEs absent
   (Branch II).
3. PINN methodology says: usable inside loops only with adaptive weighting,
   warm-started continuation, residual-guided sampling, and classical-solver
   certification (Branch III).

The unoccupied intersection is a **multi-physics body co-optimizer**: a
companion-robot shell whose geometric control points and mass layout are
optimized against quasi-static stability + locomotion energy + a
shape-conditioned thermal PINN (Robin BCs, verified against finite volumes)
+ enclosure acoustics, with the PINN warm-started across designs and every
accepted design re-certified. No surveyed work assembles these; the nearest
neighbors are the parameterized thermal PINN for power modules (single
physics, fixed topology), hPINN (single-physics heat topology optimization),
and X-CAD (control points + elasticity only). This is precisely the PhORM
framework implemented in this repository.

## 7. Open problems

- Gradient quality: PINN-derived ∂J/∂shape vs. classical adjoints is
  unquantified (no benchmark); our FV-verification experiment is a step.
- Frequency-domain acoustics at speech-band k·D needs multi-scale Fourier
  features + PML — an open solver problem before it can enter a loop.
- Coupled thermo-acoustics (electronics heat → sound speed → mic response)
  and data assimilation from on-board sensors (thermistors, mic arrays) are
  unexplored.
- Compliant-body effects (footpad deformation, shell flexure) break analytic
  stability margins; a PINN elasticity layer is the natural bridge.
- Design-space (not single-design) outputs: latent-conditioned fields
  (TOM/OpenTO pattern) applied to robot families.

## 8. Conclusions

**RQ1.** Physics enters neural shape optimization as residual losses,
energy forms, or solver-in-the-loop autodiff; the most effective documented
pattern couples a shape-parameterized network (reference-domain mapping or
SDF conditioning) with adaptive loss weighting, warm-started continuation,
and certification against a classical solver. **RQ2.** Small mobile robot
performance is dominated by support geometry + CoM height (stability), total
mass (energy), and — ignored by the literature — shell surface/cavity
geometry (thermal and acoustic), all controllable through the same small set
of geometric control points and mass-layout variables. **RQ3.** Open
small-robot projects simulate control around a frozen body and iterate the
body manually; a PINN-based, verified, multi-physics shape co-optimizer
closes that loop at hobbyist compute budgets, which is the contribution this
repository implements and measures.

## Key references (selection; full annotated lists in docs/perspectives/)

Raissi, Perdikaris & Karniadakis (2019) JCP; E & Yu (2018) Commun. Math.
Stat.; Lu et al. (2021) SIAM J. Sci. Comput. (hPINN) & SIAM Review (DeepXDE);
Sun, Sengupta & Juniper (2023) CMAME; Wassing et al. (2024) Phys. Fluids;
Zhang et al. (2024) JCP (SDF surrogates); Tancik et al. (2020) NeurIPS;
Hoyer et al. (2019) arXiv:1909.04240; Chandrasekhar & Suresh (2021) SMO
(TOuNN); Zehnder et al. (2021) NeurIPS (NTopo); Hafner et al. (2019) ACM TOG
(X-CAD); Liu et al. (2025) SIGGRAPH (NeuralTOMO); Sanu et al. (2024)
arXiv:2407.13954; Woldseth et al. (2022) SMO; Papadopoulos & Rey (1996)
ICRA; Vukobratović & Borovac (2004) Mechanism and Machine Theory; Lipson &
Pollack (2000) Nature; Zhao et al. (2020) ACM TOG (RoboGrammar); Wang et al.
(2023) ICLR (SoftZoo); Hu et al. (2020) ICLR (DiffTaichi); Howell et al.
(2022) RSS (Dojo); Zakka et al. (2025) arXiv (MuJoCo Playground); Katz,
Di Carlo & Kim (2019) ICRA; Cai et al. (2021) ASME JHT; Krishnapriyan et al.
(2021) NeurIPS; Wang, Yu & Perdikaris (2022) JCP; Wang, Teng & Perdikaris
(2021) SIAM J. Sci. Comput.; McClenny & Braga-Neto (2023) JCP; Wu et al.
(2023) CMAME (sampling); Hao et al. (2024) NeurIPS D&B (PINNacle); Takamoto
et al. (2022) NeurIPS D&B (PDEBench); Yang et al. (2025) IEEE T-PEL;
Open Duck Mini (GitHub, 2024–2026); Reachy Mini (Pollen Robotics, 2025–2026).
