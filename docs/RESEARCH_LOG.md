
## iter0_baseline — BASELINE
- Hypothesis: Establish the stock dtiny-like reference design and its objective value J(stock) with the full physics pipeline (stability, locomotion, packing, PINN thermal, acoustics). All later iterations are measured against this baseline.
- stock reference metrics: energy J/m=0.12  stab mm=11.40  Tmax C=50.85  f_H Hz=2279.50  acou pen=12.30  mass g=393.59  z_cm mm=39.57
- J(stock) = 1.0000
- Config: `iter0_baseline.json` (use_pinn=True, outer=None, warm=None)

## iter1_geomopt_fast — ACCEPTED (new best)
- Hypothesis: Geometry + mass co-optimization with fast physics only (stability, locomotion, packing, acoustics, lumped thermal proxy) already improves the scalar objective substantially: the optimizer should flatten the crown, slim the upper body, drop the battery to the floor and widen the track for tipover margin, while shrinking shell volume for mass/energy.
- J: 1.0000 (stock) -> 0.8278  (+17.2%)  **new best**
- stock:    energy J/m=0.12  stab mm=11.40  Tmax C=37.78  f_H Hz=2279.50  acou pen=12.30  mass g=393.59  z_cm mm=39.57
- optimized:energy J/m=0.10  stab mm=14.98  Tmax C=39.57  f_H Hz=1753.74  acou pen=5.89  mass g=350.30  z_cm mm=31.04
- seed 0 @ iter 119, wall 7s
- Config: `iter1_geomopt_fast.json` (use_pinn=False, outer=120, warm=None)
