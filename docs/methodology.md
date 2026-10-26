# PhORM Methodology — Physics-Informed Shape Optimization of a Small Mobile Robot Body

This document is the complete mathematical specification of the framework as
implemented in `src/`. Units: mm/g/s in the design space; SI internally for
the field problems.

## 1. Problem statement

We seek the body geometry and internal mass layout $\theta$ of a small
wheeled companion robot (moves, talks, hears) that minimizes a multi-physics
objective

$$
\min_{\theta \in \Theta}\;
J(\theta) = w_E \frac{E(\theta)}{E_0} + w_S \frac{E_s(\theta)}{E_{s,0}}
+ w_{st} \frac{S_0}{S(\theta)} + w_T \frac{T_{\max}(\theta)}{T_{\max,0}}
+ w_A \frac{A(\theta)}{A_0} + \mu \sum_c P_c(\theta),
$$

where $E$ is the locomotion energy per metre, $E_s$ the same on a 10° ramp,
$S$ the worst-case ZMP tipover margin, $T_{\max}$ the peak steady-state
temperature (from the thermal PINN), $A$ the acoustic detuning penalty of the
speaker cavity, and $P_c$ smooth constraint penalties with continuation
weight $\mu$. The reference values $(\cdot)_0$ are the stock design's metrics
evaluated with the same pipeline, so $J(\theta_0) \approx \sum w_i$ for the
feasible stock design. Weights: $w_E{=}0.25, w_S{=}0.15, w_{st}{=}0.25,
w_T{=}0.20, w_A{=}0.15$.

The design vector (15 parameters, all bounded in a manufacturable box):

| group | parameters | bounds |
|---|---|---|
| profile control radii $r_0..r_6$ | 7 spline knots at fixed heights $s_i$ | $r_0..r_5$: 10–45 mm, $r_6$: 8–35 mm |
| body height $H$ | 1 | 85–135 mm |
| wall thickness $t$ | 1 | 1.6–3.2 mm |
| wheel radius $r_w$, track width $w_t$ | 2 | 10–20 mm, 34–96 mm |
| battery mass $m_b$, heights $z_b, z_{pcb}$, clearance $c$ | 4 | per component |

Bounds are enforced exactly by a sigmoid reparameterization
$\theta = \theta_{\min} + (\theta_{\max}-\theta_{\min})\,\sigma(x)$ with
unconstrained $x \in \mathbb{R}^{16}$ (the optimization variable) — the same
"bounds-by-construction" role projection plays in neural topology
optimization (Hoyer et al. 2019; Chandrasekhar & Suresh 2021).

## 2. Geometry: solid of revolution from spline control points

The body is a solid of revolution. Its meridian profile $r(s)$, height
fraction $s\in[0,1]$, is a **natural cubic spline** through the 7 control
radii at fixed knots $s_i = \{0, 0.15, 0.32, 0.52, 0.72, 0.88, 1\}$. The
second derivatives $M_i$ solve the standard tridiagonal natural-spline
system; evaluation is the exact cubic piecewise polynomial (fully
differentiable w.r.t. $r_i$, implemented with `torch.linalg.solve`).

Derived differentiable quantities (trapezoid integration over 64 height
steps):

- shell volume $V_{wall} = \pi\!\int (r^2 - (r-t)^2)\,dz + \pi r(0)^2 t + \pi r(1)^2 t$ (mm³),
- cavity volume $V_{cav} = \pi \int (r-t)^2 dz$,
- frontal silhouette $A_f = \int 2 r\,dz + 1.2\pi r_w^2$ (exact for solids of revolution),
- shell surface $A_{shell} = \int 2\pi r\,dz$,
- mass $M = \rho_{PLA} V_{wall} + \sum_j m_j$ with components battery
  $m_b$ at $z_b$, PCB (45 g) at $z_{pcb}$, speaker (32 g) at $0.86H$, head
  electronics (12 g) at $0.95H$, drive units (90 g) at axle height $r_w$;
- centre of mass height $z_{cm} = \sum_j m_j z_j / M$;
- support polygon half-axes: lateral $a_y = w_t/2$, longitudinal
  $a_x = \big(r(0.06) - 8\,\mathrm{mm}\big)$ clamped to $[18, 48]$ mm
  (castor span derived from the base radius — footprint and stability are
  coupled through the geometry, not independent).

## 3. Fast physics (quasi-static models)

**Tipover stability (ZMP margins).** For each maneuver the centre-of-mass
projection shifts by $d = h_{cm}\,a/g$ (inertial, $a \in \{\pm2, -1.5\}$
m/s²) or $d = h_{cm}\tan\alpha$ (slope, $\alpha \in \{15°_{side}, 20°_{fwd},
20°_{rear}\}$) in the maneuver's axis. The margin is the distance from the
shifted projection to the nearest edge of the rectangular support polygon:

$$
S(\theta) = \min_{\text{maneuvers}} \min\big(a_x - |d_x|,\; a_y - |d_y|\big),
$$

and the static tip angle is $\arctan(\min(a_x,a_y)/h_{cm})$. This is the
rectangular-polygon specialization of the force–angle/ ZMP stability family
(Papadopoulos & Rey 1996; Vukobratović & Borovac 2004).

**Locomotion energy.** Per metre (J/m = N) at cruise $v = 0.35$ m/s:

$$
E = \underbrace{c_{rr} M g}_{\text{rolling}} +
\underbrace{\tfrac12 \rho_a C_D A_f v^2}_{\text{drag}}, \qquad
E_s = c_{rr} M g \cos 10° + M g \sin 10° + \tfrac12\rho_a C_D A_f v^2 ,
$$

with $c_{rr} = 0.03$ (rubber on hardwood), $C_D = 0.95$. Cost of transport
$\mathrm{CoT} = E/(Mg)$.

**Packing and fit.** Component volumes (density-adjusted) must fit in
$\le 55\%$ of $V_{cav}$; the battery/PCB/speaker need radial room at their
heights ($r(s)-t \ge$ half-width + margin); minimum cavity radius
$r - t \ge 6$ mm; total width $\le 110$ mm; wheel radius vs. clearance
($c + 2 \le r_w$); a stability floor ($S \ge 10$ mm, enforced in the
objective); and a **profile-curvature cap** — the natural spline can
oscillate between control points (the unconstrained optimum reached
$\max|r''| \approx 3100$ vs. the stock profile's $\approx 970$, a visible
manufacturability wiggle), so $\max |r''| \le 1500$ is penalized. All
constraints are smooth one-sided penalties
$\mathrm{softplus}(v/s)^2 \cdot 10^{-6}$ (zero when feasible).

## 4. Thermal field: shape-conditioned PINN (the core novelty)

### 4.1 Strong form

Steady **axisymmetric** heat conduction on the meridian half-plane
$(r, z)$, $z$ from the body underside to the crown:

$$
\nabla\cdot\big(k\,\nabla T\big) + Q''' = 0
\quad\Longleftrightarrow\quad
k\Big(T_{rr} + \tfrac{1}{r}T_r + T_{zz}\Big) + \nabla k\!\cdot\!\nabla T + Q''' = 0 ,
$$

$$
-k\,\frac{\partial T}{\partial n} = h_{eff}\,(T - T_\infty)
\ \text{ on the outer surface}, \qquad T_r = 0 \ \text{ on the axis},
$$

with $T_\infty = 25°$C, $h_{eff} = 11.5$ W/(m²K) (natural convection plus
linearized radiation for PLA enclosures), $k = 0.13$ W/(mK) in the wall
band $r_{in}(s) = r(s) - t \le r \le r(s)$ and $k = 0.5$ W/(mK) in the
cavity (lumped air + radiation + mixing), blended smoothly over 0.8 mm.
Heat sources are axisymmetric gaussian rings at the three dissipaters — PCB
2.2 W, battery 0.3 W, speaker 1.0 W — normalized so that
$\int Q'''_i\, 2\pi r\,dr\,dz = P_i$ **exactly**.

Two implementation-critical points (both discovered by verification against
the finite-volume solver, §4.4):

1. **The $\nabla k \cdot \nabla T$ interface term is mandatory.** With a
   piecewise $k$, the strong form is the divergence form; omitting
   $\nabla k\cdot\nabla T$ leaves the wall/cavity interface as an
   unconstrained flux source/sink. Networks trained without it satisfy the
   pointwise loss yet violate global energy conservation by watts.
2. **The axis needs the limit form.** Below $r = 1.5$ mm the residual
   switches to $k(2T_{rr} + T_{zz}) + Q''' = 0$ (the $T_r/r \to T_{rr}$
   limit), and collocation covers $\rho \in [0, 1]$ including $r=0$; leaving
   the axis unsampled admits an absorbing artifact column that absorbs
   watts while the pointwise loss stays small.

### 4.2 Reference-domain mapping (shape conditioning)

Collocation points live in a fixed reference square $(\rho, s) \in [0,1]^2$
and are mapped by $r = \rho\, r(s)$, $z = c + sH$ — the network always sees
a fixed box while the physical domain deforms with the design. 30% of points
are source-biased (gaussian around each dissipater, RAR-style residual-guided
sampling in the spirit of Lu et al. / McClenny & Braga-Neto); at each
resampling (every 150 steps) one third of the batch is re-selected as the
highest $|R|\cdot V$ points from a 4× candidate pool. Boundary points sample
the profile curve (with analytic outward normals $\mathbf{n} \propto (H,
-r'(s))$), the top disc, and the bottom disc.

### 4.3 Network, loss, and continuation

$T_\theta(r,z) = T_\infty + 15\,\mathrm{K}\cdot \mathrm{MLP}_\theta(r/0.06,
z/H)$ with a 4×64 tanh MLP (zero-initialized output layer, so training
starts from the ambient state). The loss is **volume-weighted** so the MSE
approximates the domain integral:

$$
\mathcal{L} = \overline{V_i \hat R_i^2}
+ 2.5\,\overline{\hat B_j^2} + \overline{\hat C_k^2},
$$

where $\hat R = (k\nabla^2 T + \nabla k\cdot\nabla T + Q''')/10.5\,\mathrm{kW/m^3}$,
$\hat B = (k\,\partial_n T + h_{eff}\Delta T)/150$ W/m², $\hat C =
\partial_r T|_{axis}$, and $V_i$ are annular volume elements normalized to
mean 1. Optimization: Adam (cosine LR $5\cdot10^{-3} \to 6\cdot10^{-4}$,
2 500 steps, resampling every 150) followed by a 200-iteration L-BFGS
polish on a fixed large batch (Raissi-style two-stage training).

**Differentiability to the design.** The field metrics
($T_{\max}$, $T_{pcb}$, $T_{bat}$, $T_{spk}$) are evaluated on a dense grid
mapped by the current spline, so $\partial T/\partial\theta$ flows through
the input mapping and source positions. The inner loop trains on a detached
copy of the geometry (network parameters only); the outer loop re-evaluates
the metrics through the live graph. Across outer iterations the network is
**warm-started** (continuation): shape changes between outer steps are
small, so 200 inner steps per outer step re-converge the field — the
discretize-then-optimize pattern with the network as the state solver
(Sun et al. 2023 use the same surrogate-backprop pattern; Lu et al. 2021
validate it against adjoints).

### 4.4 Verification against an independent solver

`src/verify_fdm.py` solves the identical problem with cell-centred **annular
finite volumes** (volumes $2\pi r_c\,\Delta r\,\Delta z$, harmonic-mean face
conductances, Robin faces at the staircase boundary, natural symmetry at
$r=0$), solved exactly with a sparse direct solver in ~0.2 s. On the stock
design (2 500 Adam + 250 L-BFGS; saved as `results/verify_stock.json`) the
PINN reproduces the FV field to $-2.0\%$ ($T_{\max}$), $-1.9\%$
($T_{pcb}$), $-3.1\%$ ($T_{spk}$) and $-5.9\%$ (the small battery blob); on
the optimized design (`results/iter3_fdm_crossval.json`) the agreement is
$-3.1\%$ / $-2.9\%$ / $-4.1\%$ / $-10.1\%$ respectively. The PINN
consistently under-reads by a few percent (tightest near narrow sources);
the FV solver is the reference and every headline thermal claim in this
work is FDM-certified. The PINN's role is to make the field differentiable
w.r.t. shape inside the optimization loop; the FV solver certifies its
accuracy.

## 5. Acoustics: speaker-cavity tuning (closed model)

The head cavity above the speaker baffle is a Helmholtz resonator:

$$
f_H = \frac{c}{2\pi}\sqrt{\frac{A_{port}}{V_{head}\,L_{eff}}},
\qquad L_{eff} = 2t + 0.85\,r_{port},
$$

with the port diameter derived from the profile at the speaker height
($\le 26$ mm). The penalty
$A = ((f_H - 520)/520)^2 + 0.35\,(V_{ref}/V_{head})^2$ pushes the resonance
toward the vocal warmth band (~520 Hz) and rewards cavity volume (bass
rolloff $\sim\sqrt{V}$). A shape-conditioned Helmholtz PINN (Neumann
cavity walls, piston-velocity source) is the planned upgrade path; the
closed model keeps the outer loop fast and is the honest baseline the PINN
upgrade will be compared against.

## 6. Outer optimizer

Adam on $x$ (the unconstrained 15-vector), lr 0.05–0.06, with:
- **penalty continuation**: $\mu(t) = \mu_0 (\mu_1/\mu_0)^{t/T}$, 2 → 120
  (config-dependent);
- **feasibility-tracked best**: the best design is the lowest
  penalty-free $J$ among outer iterations with total penalty below
  $5\cdot10^{-3}$; if no iteration reaches feasibility, the design with
  the lowest $J + 100\,P$ is used and flagged infeasible (its
  `pack_frac` may exceed the target by a few percent — reported).
- multi-seed restarts (configurable; each seed perturbs the stock design by
  $\sigma = 0.1$ in logit space, or explores uniformly);
- final re-evaluation of the winner with a fresh full-length PINN solve.

## 7. Autoresearch protocol

Experiments are configs (`experiments/configs/iter*.json`) with an explicit
hypothesis; `experiments/autoresearch_loop.py` runs pending configs, scores
them, and **accepts a design change only if it beats the current best** on
the same metric basis (pinn vs fast-proxy are tracked separately), then
appends the hypothesis, metric deltas, and accept/reject decision to
`docs/RESEARCH_LOG.md`. The protocol mirrors Karpathy's autoresearch loop
(short experiments, keep-only-improvements, auditable ledger) applied to
design optimization: the "loss" is the multi-physics objective $J$, and the
agent's code/hyperparameter changes are the mutations.

## 8. Known limitations (honest list)

- The thermal model is per-unit-depth axisymmetric with a lumped cavity
  conductivity — right for trend and trade-off analysis, not CFD-grade.
- Acoustics is a closed Helmholtz-resonator model (PINN upgrade planned).
- Structural stiffness (shell flexure, drop survival) is not yet in the
  loop; the packing/fit and wall-thickness bounds stand in for
  manufacturability.
- Stability uses a rectangular support polygon and quasi-static maneuvers.
- $Q'''$ magnitudes are nameplate estimates, not measured thermal budgets.
