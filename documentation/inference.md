# Inference

This note explains how the code turns a point estimate into standard errors and a
specification test. It is written for an economist: the estimator, the two noise
sources, how the Jacobian is taken, the two variance formulas, and the reports that
come out. The implementation is in `tools.jl` (`build_step3_weight_matrix`,
`compute_jacobian`, `compute_smm_inference`).

---

## 1. Setup

The estimator matches a vector of moments. Write `m(θ)` for the model moments at
parameters `θ` and `m̂` for the empirical moments. θ̂ minimizes the weighted distance

```
   θ̂ = argmin_θ  g(θ)' W g(θ),        g(θ) = m̂ − m(θ),
```

with weight matrix `W`. Inference is carried out on the **β+γ subsystem** — the
distance-regression moments (`reg_coef`) and the sourcing shares (`γ_ls`) — against
the parameters those moments identify, namely the trade-cost elasticity α and the
comparative-advantage scales T. (The other moments pin down the level parameters Ω, A
in earlier steps and are not part of the efficient-weighting subsystem.)

---

## 2. Two sources of noise: data vs simulation

The estimator's sampling variability has **two independent origins**, and the code
keeps them separate. This is the one place where SMM differs from textbook GMM.

- **Data noise `Σ_data`.** The empirical moments `m̂` are themselves estimated from a
  finite sample, so they carry sampling error. `Σ_data` is their covariance, taken from
  a **bootstrap** (`Sigma_beta_gamma*.npy`). It is non-zero only on the `reg_coef` and
  `γ_ls` blocks.

- **Simulation noise `Σ_sim`.** In SMM the model moments `m(θ)` are not computed in
  closed form; they are *simulated* with a finite number of Fréchet draws, so `m(θ)`
  is itself random given θ. `Σ_sim` is the covariance of that simulation error. The
  code estimates it directly (`build_step3_weight_matrix`): it re-evaluates the
  simulator `K` times at θ̂₁, each time with an **independent** draw set, and takes the
  sample covariance of the resulting moment vectors.

The total variance of the matched discrepancy `g(θ) = m̂ − m(θ)` is the sum, since the
two are independent:

```
   Ω = Σ_data + Σ_sim.
```

`Ω` is the object that drives both the weight matrix and the standard errors. In the
**GMM** path there is no simulation, so `Σ_sim = 0` and `Ω = Σ_data`; that is the only
formal difference between the two estimators' inference.

**Efficient weighting.** The efficient choice is `W = Ω^{-1}`, built once in Step 2 as
`W = (Σ_data + Σ_sim)^{-1}` and reused in Step 3's optimization and in inference. (The
code guards `Ω` against ill-conditioning with an eigenvalue floor before inverting.)

---

## 3. The Jacobian `G = ∂m/∂θ`

All the variance formulas need the sensitivity of the moments to the parameters,
`G = ∂m(θ)/∂θ`, evaluated at the estimate. How it is computed depends on the estimator.

### SMM — finite differences, at fixed draws, averaged over replications

`compute_jacobian` takes **central finite differences** of the simulated moment vector.
Two details matter:

- **Fixed draws inside each difference.** For a given replication the `+` and `−`
  evaluations use the *same* draw set, so the simulation noise cancels in the
  difference and the derivative is smooth — the same trick the optimizer relies on.
  The whole Jacobian is then **averaged over `K` independent replications** (fresh
  draws each), and the per-entry standard deviation across replications, `J_sd`, is
  reported as a direct read-out of how much simulation noise remains in `G`.

- **The perturbation scale.** The step size is chosen per parameter type:
  - Most parameters (Ω, A, α) use an **additive** central step
    `h_j = max(|θ_j| · step_rel, step_abs)` with `step_rel = 1e-4`, `step_abs = 1e-9`.
    Relative where the parameter is sizable, with a small absolute floor near zero.
  - The **T scales use a log-space step**: they are strictly positive and enter
    multiplicatively, so they are perturbed as `θ_j · exp(±δ)` with `δ = step_rel`, and
    the log-space slope `(m₊ − m₋)/(2δ)` is converted back to raw units by dividing by
    `θ_j`. A log step is **scale-invariant** — it never crosses zero, never hits the
    additive floor, and never trips the Fréchet `max(T, ε)` clamp — which matters
    because T entries span several orders of magnitude. The stored column is still the
    raw derivative `∂m/∂θ_j`, so the variance formulas are unaffected in units.

### GMM — exact derivatives by automatic differentiation

In the analytical path the moments are a closed function, so `compute_jacobian` returns
the **exact** `∂m/∂θ` by forward-mode automatic differentiation (`ForwardDiff`) — no
step size, no truncation error, and `J_sd ≡ 0`. This is why GMM standard errors are
"exact" relative to SMM's.

---

## 4. Two variance formulas: efficient and sandwich

With `G`, `W`, and `Ω` in hand, the delta method gives the parameter covariance. The
code reports **both** standard formulas.

- **Efficient variance** (valid when `W = Ω^{-1}`):

  ```
     V_eff = (G' W G)^{-1}.
  ```

- **Sandwich variance** (valid for *any* `W`):

  ```
     V_sw = (G' W G)^{-1} · (G' W Ω W G) · (G' W G)^{-1}.
  ```

The two coincide exactly when `W = Ω^{-1}` — substitute and the middle factor cancels.
So the **ratio `se_sw / se_eff`** is a built-in diagnostic: a value near 1 confirms the
weight matrix really is the efficient `Ω^{-1}` (as intended after Step 2); a value far
from 1 flags that `W` and `Ω` have drifted apart (e.g. because `Ω` was re-estimated, or
an eigenvalue floor was applied), in which case the **sandwich** SE is the one to
trust. Standard errors are the square roots of the diagonals; t-statistics use `se_sw`,
and 95% confidence intervals are `θ̂ ± 1.96 · se_sw`.

**Fitted-moment and residual SEs.** Propagating the parameter covariance back through
`G` gives the covariance of the *fitted* moments, `V_m = G V_sw G'`, and hence a
standard error for each fitted moment. Subtracting it from the total, `diag(Ω) −
diag(V_m)`, gives a **residual SE** per moment: a residual share near 0 means the moment
is essentially pinned down by the fit; near 1 means it is barely used.

**Caveat (recorded in the report).** These SEs are conditional on the draws used to
estimate `Σ_sim`; a Murphy–Topel-type correction for the sequential estimation of `Σ_sim`
and θ̂ is not applied.

---

## 5. Over-identification: the Hansen J-test

When there are more moments than parameters (`df = N_moments − N_params > 0`), the
weighted residual at the optimum is a specification test:

```
   J = g(θ̂)' W g(θ̂)   ~   χ²(df)   under the null that the model is correctly specified.
```

The code reports `J`, its degrees of freedom, and the p-value; a small p-value rejects
the model's over-identifying restrictions. This is meaningful precisely in the
over-identified configurations — e.g. `--n_tau=1 --n_coef=4`, where one α parameter
faces four distance moments (`df = 3` on that block). When the system is exactly
identified (`df = 0`) the test is vacuous and the report says so.

---

## 6. What gets written

Inference runs at θ̂₁ (into `step2/inference/`) and at θ̂₂ (into `step3/inference/`).
Each produces:

**Arrays (`.npy`):** `var_theta_efficient`, `var_theta_sandwich`, `se_theta`,
`se_theta_sandwich`, `t_stats`, `ci_95`, `se_moments_fitted`, `se_moment_residuals`.

**`inference_summary.txt`** — the human-readable report:
- an **identification check** (`λ_min`, `λ_max`, condition number, and rank of `G'WG` —
  flags weak or singular directions);
- a **parameter table**: estimate, `se_eff`, `se_sw`, their ratio, t-stat, and 95% CI,
  one row per active parameter (labelled);
- the **sandwich/efficient ratio** summary (mean and max — should sit near 1);
- **per-block and per-moment residual SEs** (how well each moment block is fit);
- the **Hansen J-test** block and a verdict;
- explicit **caveats**.

**`J_stat.txt`** — the Hansen J statistic, df, p-value, and verdict, on their own.

**`gamma_ls_fitted_se.png`** — fitted vs. empirical sourcing shares with ±1 SE bars,
including the per-sector reference regions reconstructed from the adding-up constraint.

**Step 2 also saves** the ingredients of `Ω`: `M_sim` (the K re-evaluations),
`Sigma_data`, `Sigma_sim`, `Omega`, `W_step3`, and a `diagnostics.txt` with `Ω`'s
conditioning.

### Diagnostics that print but do not change results

- **`screen_T_identification`** runs after every Jacobian and prints the conditioning of
  `H = G'WG`, its T-only sub-block, and a per-sector identification screen — an early
  warning for weakly identified T directions. Print-only.
- **`run_2x2_inference_test`** (optional, off by default) decomposes the parameter
  variance along two axes — Jacobian channel (simulated finite-difference vs. exact
  analytical) × weighting channel (data-only `Σ_data` vs. data+simulation `Ω`) — to
  attribute where the SMM standard errors' noise comes from. It writes to its own
  `inference_2x2_test/` tree and never touches the production `inference/` outputs.

---

## 7. In one paragraph

The empirical moments carry sampling error `Σ_data` (bootstrapped) and, in SMM, the
model moments carry simulation error `Σ_sim` (estimated by re-simulation); their sum
`Ω = Σ_data + Σ_sim` is both the efficient weight `W = Ω^{-1}` and the meat of the
variance. The moment sensitivities `G = ∂m/∂θ` are taken by fixed-draw central
differences (log-step for the positive, multiplicative T scales; exact AD in GMM). The
delta method then gives the efficient variance `(G'WG)^{-1}` and the robust sandwich
`(G'WG)^{-1} G'WΩWG (G'WG)^{-1}`, which agree when `W = Ω^{-1}`; the Hansen J tests the
over-identifying restrictions. All of it is written to `inference/` as arrays plus a
readable summary.
