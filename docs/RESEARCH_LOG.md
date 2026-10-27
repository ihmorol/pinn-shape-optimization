
## iter0_baseline — BASELINE
- Hypothesis: Establish the stock dtiny-like reference design and its objective value J(stock) with the full physics pipeline (stability, locomotion, packing, PINN thermal, acoustics). All later iterations are measured against this baseline.
- stock reference metrics (basis=pinn): energy J/m=0.12  stab mm=11.40  Tmax C=54.96  f_H Hz=2279.50  acou pen=12.30  mass g=393.59  z_cm mm=39.57
- J(stock) = 1.0000
- Config: `iter0_baseline.json` (use_pinn=True, outer=None, warm=None)

## iter1_geomopt_fast — ACCEPTED (new best)
- Hypothesis: Geometry + mass co-optimization with fast physics only (stability, locomotion, packing, acoustics, lumped thermal proxy) already improves the scalar objective substantially: the optimizer should flatten the crown, slim the upper body, drop the battery to the floor and widen the track for tipover margin, while shrinking shell volume for mass/energy.
- J (basis=proxy): 1.0000 (stock) -> 0.8278  (+17.2%)  **new best**
- stock:    energy J/m=0.12  stab mm=11.40  Tmax C=37.78  f_H Hz=2279.50  acou pen=12.30  mass g=393.59  z_cm mm=39.57
- optimized:energy J/m=0.10  stab mm=14.98  Tmax C=39.57  f_H Hz=1753.74  acou pen=5.89  mass g=350.30  z_cm mm=31.04
- seed 0 @ iter 119, wall 7s
- Config: `iter1_geomopt_fast.json` (use_pinn=False, outer=120, warm=None)

## iter2_pinn_coopt — ACCEPTED (new best)
- Hypothesis: Putting the shape-conditioned thermal PINN inside the outer loop changes the optimal shape relative to the lumped proxy: the true field concentrates heat near the PCB/speaker blobs, so the optimizer should trade some compactness for shell surface (flatter/wider crown region, thinner walls where permissible) to keep T_max down while still gaining stability and mass. Warm-started PINN (continuation) keeps each outer step at ~200 inner steps.
- J (basis=pinn): 1.0000 (stock) -> 0.8499  (+15.0%)  **new best**
- stock:    energy J/m=0.12  stab mm=11.40  Tmax C=54.32  f_H Hz=2279.50  acou pen=12.30  mass g=393.59  z_cm mm=39.57
- optimized:energy J/m=0.10  stab mm=14.14  Tmax C=53.03  f_H Hz=1958.30  acou pen=8.05  mass g=347.84  z_cm mm=31.35
- seed 0 @ iter 79, wall 514s
- Config: `iter2_pinn_coopt.json` (use_pinn=True, outer=80, warm=200)

## iter4_multiseed — REJECTED
- Hypothesis: Robustness: restarting the PINN-in-loop co-optimization from perturbed designs (larger noise, two fresh seeds) converges to the same trade-off region (stability floor active, low CoM, slimmer crown) with objective values within a few percent of the it2 winner - the framework is seed-robust rather than a lucky single run.
- J (basis=pinn): 1.0000 (stock) -> 0.9801  (+2.0%)  
- stock:    energy J/m=0.12  stab mm=11.40  Tmax C=54.98  f_H Hz=2279.50  acou pen=12.30  mass g=393.59  z_cm mm=39.57
- optimized:energy J/m=0.10  stab mm=14.69  Tmax C=53.65  f_H Hz=2695.54  acou pen=18.93  mass g=351.73  z_cm mm=29.76
- seed 1 @ iter 59, wall 671s
- Config: `iter4_multiseed.json` (use_pinn=True, outer=60, warm=180)

## iter3_fdm_crossval — CERTIFICATION
- Hypothesis: Certification experiment: the shape-conditioned PINN field solution on the OPTIMIZED design agrees with the independent annular finite-volume solver within a few percent for every thermal metric, and the FDM-certified T_max reduction (optimized vs stock) confirms the optimization gain was physical, not a PINN artifact.
- T_max: fdm=54.64 pinn=52.96 (-3.1%)
- T_pcb: fdm=53.61 pinn=52.06 (-2.9%)
- T_battery: fdm=37.36 pinn=33.60 (-10.1%)
- T_speaker: fdm=43.72 pinn=41.90 (-4.1%)
- FDM-certified T_max: stock 56.28 C -> optimized 54.64 C
- Config: iter3_fdm_crossval.json (mode=crossval, 2500 Adam + 250 L-BFGS)

## iter5_smooth — REJECTED
- Hypothesis: Manufacturability refinement: figure analysis of the it2 winner exposed spline overshoot (max |r''| = 3114 vs stock 973 - a wiggle between control points). Adding a profile-curvature penalty (cap 1500) should produce a smooth, printable profile at a small objective cost (<2%), closing the loop from figure diagnostics back into the constraint set - the auto-research loop improving its own design rules.
- J (basis=pinn): 1.0000 (stock) -> 0.9094  (+9.1%)  
- stock:    energy J/m=0.12  stab mm=11.40  Tmax C=55.01  f_H Hz=2279.50  acou pen=12.30  mass g=393.59  z_cm mm=39.57
- optimized:energy J/m=0.10  stab mm=14.11  Tmax C=53.36  f_H Hz=2337.92  acou pen=13.03  mass g=346.19  z_cm mm=31.26
- seed 0 @ iter 79, wall 538s
- Config: `iter5_smooth.json` (use_pinn=True, outer=80, warm=200)

## iter6_locomotion — RECORDED (proxy basis; not comparable to the PINN-basis champion)
- Hypothesis: Locomotion-focused re-weighting (energy 0.35 + slope 0.30 + stability 0.25, thermal/acoustic 0.05 each): 'most optimized to move around'. Expect the lightest feasible shell (mass dominates rolling and grade energy), battery pinned to the floor, widest track allowed by the width cap, and the stability floor active - i.e., a low, wide, light rover profile rather than a tall companion silhouette. Thermal weight is tiny so the lumped proxy is adequate; smoothness cap stays on for printability. NOTE: J values across different weightings are not comparable - only the improvement over this run's own stock baseline (same weights) is meaningful.
- J (basis=proxy): 1.0000 (stock) -> 0.8534  (+14.7% vs its own stock baseline under locomotion weights)
- stock:    energy J/m=0.12  stab mm=11.40  Tmax C=37.78  f_H Hz=2279.50  acou pen=12.30  mass g=393.59  z_cm mm=39.57
- optimized:energy J/m=0.10  stab mm=14.86  Tmax C=39.39  f_H Hz=2157.05  acou pen=10.50  mass g=343.21  z_cm mm=31.22
- seed 0 @ iter 149, wall 7s
- Config: `iter6_locomotion.json` (use_pinn=False, outer=150, warm=None)
