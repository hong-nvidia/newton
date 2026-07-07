# Granular Impact Force Law — Newton Validation

This documents `example_mpm_impact.py` and how well it reproduces the paper it
is based on. Run it with:

```bash
python -m newton.examples mpm_impact                       # single impact (viewer)
python -m newton.examples mpm_impact --force-law --viewer null   # force-law fit
```

## 1. The paper

**Katsuragi & Durian, "Unified force law for granular impact cratering,"
*Nature Physics* 3, 420 (2007).** [[Link to paper](https://arxiv.org/pdf/cond-mat/0703072)]

**Problem it solves.** Prior low-speed granular-impact studies had proposed
*four mutually contradictory* stopping-force laws (constant force; linear in
speed; proportional to depth; a product of powers of depth and speed). The
paper resolves the confusion by measuring the projectile's motion precisely
enough to isolate the underlying force.

**Method.** A 1-inch (`D_b = 2.54 cm`) steel sphere (mass 69.2 g, effective
density 8.07 g/cm³) is dropped from a wide range of heights (impact speeds
`v0 = 0–4 m/s`) into dry, noncohesive glass beads (250–350 μm,
`ρ_g = 1.52 g/cm³`, `μ = tan 24° = 0.45`, packing fraction 0.59). A line-scan
camera tracks a striped rod on the ball, giving 100 nm / 20 μs resolution —
enough to extract `z(t)`, `v(t)`, `a(t)` cleanly.

**Central result — the unified force law:**

```
F = -m g + k|z| + m v^2 / d1
        \_______/   \_________/
   depth-dependent   velocity-dependent
      friction        inertial drag
```

- Inertial drag `m v^2 / d1` with `d1 = 8.7 ± 0.7 cm`, **independent of
depth** — interpretable as `0.8 ρ_g D_b^2 v^2` (force to mobilize a
ball-sized volume of grains).
- Friction: linear `k|z|` with `k/m = 1040 ± 10 s⁻²`, i.e.
`k ≈ 20 μ ρ_g g D_b^2`.

**Signature findings.** All four prior laws emerge as limiting cases. The
velocity vanishes with an **acceleration discontinuity** at stopping and —
counterintuitively — the **stopping time *decreases* as impact speed
increases**. Characteristic scales are set by ball size and gravity
(`L_c = D_b`, `T_c = √(D_b/g)`, `V_c = √(D_b g)`), explaining why penetration
is always of order `D_b`.

## 2. What the simulation does (`--force-law` mode)

- Models the sand as **implicit MPM** (Drucker–Prager) in a **walled
container**, with a rigid sphere impactor and **two-way coupling** (the sand
exerts a reaction force on the ball each step).
- Starts the ball a short distance above the surface and launches it with a
reduced velocity so gravity accelerates it to exactly the target impact
speed `v0 = √(2 g H)` at the surface (a brief, visible descent that leaves
the impact conditions unchanged).
- Uses **fine time sampling** (500 fps) and **stiff sand** (yield pressure
1e6 Pa, Young's modulus 1e7 Pa) appropriate for the force-law regime.
- Records the ball's `z(t)`, `v(t)`, and the **sand reaction force**
`F_sand(t) = m(a + g)` — taken directly from the coupling rather than by
differentiating velocity (much lower noise).
- Reproduces the paper's **Fig. 3 analysis**: samples `(v, F)` at several
fixed penetration depths across all drop heights, then does a joint
least-squares fit with a **shared** `d1` and per-depth friction intercepts.
This is essential — along a single drop, depth and `v²` are collinear, so
the two force terms can only be separated across multiple heights.
- Reports `d1`, the drag coefficient `C`, `k/m`, `F(z_i)/m` vs depth, the
stopping-time-vs-`v0` table, and (optionally) a 3-panel plot mirroring the
paper's Fig. 3.



## 3. How well it tracks the paper's claims

Using the paper's actual system (2.54 cm steel sphere, glass beads), a
6-height sweep with `v0 = 1.4–4.2 m/s`:


| Claim from paper                                 | Paper value        | Simulation                  | Match                |
| ------------------------------------------------ | ------------------ | --------------------------- | -------------------- |
| Force decomposes into friction + inertial drag   | qualitative        | fit R² = 0.91               | ✅ form reproduced    |
| Inertial drag `d1` constant across depths/speeds | 8.7 ± 0.7 cm       | 8.4 cm (constant)           | ✅ within uncertainty |
| `d1 ≈ m/(0.8 ρ_g D_b²)`                          | 8.7 cm (pred 8.83) | 8.4 cm (pred 8.83)          | ✅                    |
| Drag coefficient `C` in `C ρ_g D_b² v²`          | ~0.8               | 0.84                        | ✅                    |
| Friction grows **linearly** with depth           | yes                | 22.2 → 89.9 s⁻² over 2–6 cm | ✅ linear form        |
| Friction magnitude `k/m`                         | 1040 s⁻²           | 1735 s⁻²                    | ⚠️ ~1.7× high        |
| Penetration ~ order `D_b`                        | ~1–3 `D_b`         | 1.4–2.8 `D_b`               | ✅                    |
| **Stopping time decreases with impact speed**    | yes (hallmark)     | 0.044 → 0.042 s             | ✅ reproduced         |


**Overall:** the simulation reproduces the *entire structure* of the unified
force law — both terms, a depth-independent `d1`, linear-in-depth friction,
order-`D_b` penetration, and the counterintuitive decreasing stopping time.
The inertial-drag length `d1` and drag coefficient `C` now match the paper
quantitatively; the friction magnitude `k/m` remains ~1.7× high.

## 4. Gaps and caveats

- **Friction magnitude ~1.7× high.** `d1` and `C` match the paper, but `k/m`
(the depth-friction slope) comes out ~~1.7× too large. Likely contributors:
continuum MPM ≠ discrete grains; a fairly narrow bed (4 ball-diameters
half-width) adds some wall drag that inflates the apparent forces; 5 mm
voxels (~~5 cells across the ball) limit fidelity. Widening the bed and
refining resolution should pull it toward the paper's value.
- **Acceleration discontinuity at stopping not verified.** The paper's sharp
`a`-jump at `t_stop` needs the differentiated `a(t)`, which is noisy in the
coupled solve; the simulation fits the (clean) reaction force instead. This
specific signature is untested.
- **Sensitive to sand stiffness.** With loose sand (yield pressure 8e3 Pa) the
friction term goes flat (pressure-capped) *and* stopping time trends the
*wrong* way. Only with stiffer sand (yield 1e6 Pa) do the depth-linear
friction and decreasing-stopping-time signatures appear. The example
therefore defaults to the stiffer rheology.
- **Force measurement is per-frame and lightly smoothed.** `F_sand` from
two-way coupling is somewhat noisy; the simulation applies a 3-point moving
average. Cleaner coupling or sub-step averaging could improve the fit.
- **Forward validation, not calibration.** The current implementation does
not fit MPM parameters to *match* the paper's coefficients (which would be
the natural next step to close the ~2× gap).
- **No automated test for the fit.** The `--force-law` mode is a batch
analysis and has no `test_final()`; the single-impact default does.

