# Validation gates and open assumptions — granular / attraction-area version

Companion to `plan_granular_aa.md` (what to build) and `finite_sample2.tex` (the model).
**Nothing here is part of the implementation.** These are the checks to run afterwards, one at a
time, and the assumptions whose cost has been measured but not removed.

---

## Part I — The assumption that matters most

### A.1 `N_s` is estimated without its feedback into the value moments

**What is assumed.** The variety count is located by a closed-form root-find on the count moment
(`plan_granular_aa.md` D5) while the value block — labour share, `π_s`, `π_r`, the sourcing shares
— is computed at the **certainty-equivalent** price index, i.e. from the pooled draws rather than
from the `N̂_s` prefix. Downstream firms optimise against `E[P^{1−ν_s}]^{1/(1−ν_s)}`, neglecting
granular price randomness.

**Why it is needed.** If the realised prefix price index drove the general equilibrium, `N_s`
would enter every value moment, the closed-form inner loop would no longer be the profiled
optimum, and `N_s` would have to be searched on the full loss.

**What it costs — measured.** `E[P^{1−ν_s}]` is genuinely invariant to the variety count, because
the self-normalised weights make it an average of i.i.d. terms. But `P = (P^{1−ν_s})^{1/(1−ν_s)}`
is a *convex* transform, so the realised index exceeds its continuum limit by Jensen. At the
calibration `θ = 1`, `ν_s = 1.5` the finite-variance condition `θ > 2(ν_s − 1)` reads `1 > 1` —
it **fails at equality** — so `(c*)^{1−ν_s}` has infinite variance and the average converges at
`√(N/log N)`:

| `N_s` | 25 | 50 | 100 | 200 | 400 | 1000 | 4000 |
|---|---|---|---|---|---|---|---|
| `E[P̂]` vs continuum | +18.6% | +11.1% | **+6.5%** | +3.8% | +2.2% | +1.0% | +0.3% |
| `sd(P̂)/P` | 0.48 | 0.35 | 0.27 | 0.20 | 0.15 | 0.11 | 0.06 |

So at `N_s ≈ 10²` the price level is off by ~6%, and that propagates to `c_r`, `y_r`, `E_sr` and
hence to `agg_industry_share` and `π_r`.

**Why it is defensible anyway.** Two reasons, and they are not rhetorical.

1. Most of the 6% would not have been identifying variation. A price-level shift is
   observationally equivalent to a technology shift (`finite_sample2.tex`, App. B), so `Ω^s` and
   `A` absorb it; only the *cross-sector* pattern of `N_s` would survive as signal. The assumption
   records that, instead of letting a slowly-converging Jensen term leak into `Ω^s` and be
   mistaken for technology.
2. By the `q ⊥ N_s` lemma none of it touches `q_ls` or the extensive margin, so the identification
   of `α` and `N_s` is unaffected either way.

**What must be undone.** The love-of-variety propagation channel lives in the *realised* index. Any
counterfactual that moves the variety set has to use it; running counterfactuals under the
certainty-equivalent reading silences the channel that motivates the whole exercise.

**A calibration note worth revisiting independently.** `θ = 1` sits exactly on the boundary of
finite variance. The value commented out in `load_parameters.jl:64` is `1.768`, which gives
`1.768 > 1` and makes this entire issue second order.

### A.2 Firm counts per cell

The firm-level reduced form identifies each observed firm with a (cell, variety) champion, so the
model gives every cell exactly `N_s` firms while observed counts vary by orders of magnitude.
Three ways out, in `plan_granular_aa.md` §1; v1 takes the first (accept the sample mismatch) and
gate **V6** measures the damage. The model's `z` are maxima over a continuum and are therefore
shifted right relative to the data's firms, so the two `log z` coefficients are estimated over
different supports.

### A.3 `𝒜⁺_s` is taken as given

The set of attraction areas hosting at least one supplier is a choice of which areas to study, not
a model prediction. The model is never asked to explain why an entire area is empty — which is the
same boundary the fixed effect draws.

---

## Part II — Validation gates

Ordered so each one can be run alone. **V0 and V1 first**: they check that the refactor changed
nothing before any modelling change is switched on.

| # | Gate | What it checks | Pass criterion |
|---|---|---|---|
| **V0** | **Legacy reproduction** | `--granular=false --ca_level=ze` reproduces the current estimates | identical moment vector and loss to floating-point noise; this is the constraint the whole implementation is built around |
| **V1** | AA map | `attraction_area_linkages.npy`: shape `(R, R_downstream)`, rows sum to 1, and `argmax_col(·) == CLOSEST_DOWNSTREAM_REGION` | exact, **asserted at load time** — a mismatch means the model's fixed effect and the empirical one are different partitions |
| **V1b** | Filter containment | every cell with `filter_N_upstream == 1` lies in an attraction area active in its own sector | if it fires, the filter is broader than `𝒜⁺`; intersect and warn rather than proceed |
| **V2** | `N_s` root-find | on synthetic `q̂`: `G(s,·)` monotone decreasing; bisection recovers a planted integer; clamps fire at both bounds | exact integer recovery |
| **V3** | AA-level Sinkhorn | round-trip recovery of a planted `T_aa` from its own area aggregates | ~1e−8, mirroring `test/test_ge_inversion.jl` |
| **V4** | Closed form vs simulation | at `θ̂`, `G(s, N̂_s)` from `q̂` against the realised empty share | agree to Monte-Carlo error; a gap points at a bug in the winner accounting |
| **V5** | Prefix stability | winners for the first `n` varieties are identical whether the solve is run at width `n` or at pool width | exact — this is what licenses one solve per replication |
| **V6** | Firm↔champion mapping | distribution of model `log z` (champions) against the empirical firms' `log z`; and `b_logz ≈ −θ = −1.0` on **both** | report the overlap and the coefficient; a large gap invalidates the coefficient comparison and calls for A.2 option 2 |
| **V7** | Two routes to `N_s` | `N̂_s` from `G_s(0)` against `N^count_s = N_supplier_s / Σ_l q̂_ls` | same order of magnitude; a large gap is a mechanism finding, not a bug |
| **V8** | Whole count curve | simulated `G_s(K)` for `K ≥ 1` against `G_K.csv` (only `K=0` is targeted) | reported max deviation — a genuinely untargeted fit check |
| **V9** | Bounds not binding | `clamped == :none` for every sector at the optimum | a persistent clamp is a rejection signal for the mechanism, not a numerical nuisance |
| **V10** | `N_s`-invariance of block 4 | drift of `reg_coef` across `N_s` at `θ̂` | should be ~0: the firm-level index contains no `N_s` term. A visible drift means the union-over-buyers approximation or the firm mapping is doing something unintended |
| **V11** | **Value-block sensitivity (tests A.1)** | recompute `π_s`, `π_r`, labour share at `N̂_s` and `2N̂_s` under the *realised* prefix price index | quantifies exactly what A.1 assumes away; the table above predicts a few percent |
| **V12** | Binding-function bias | at `θ̂`, `E[β̂(sim)]` against `β(E[y])` | measured at 2–4% of `β` in configurations matching the data; report it with `α̂` or add it as a local offset |
| **V13** | Group-emptying rate | share of `𝒜⁺` groups that come out empty in a realised economy | governed by suppliers per group: 0.7% at 5, 3.7% at 3.3, 14% at 2, 37% at 1. Count the active groups first |

### Phase 0 — before any of this

Two measurements that cost nothing and size everything else.

1. **Time one loss evaluation.** `n_good` goes from ~142 to ~1161, and the regression from
   `n_good × N_rho` rows to `n_cells × N̂_s × R_rep`. The net is not predictable a priori.
2. **Bracket `N̂_s` from `G_K.csv` directly**, using the bounds `[N_supplier_s / R_downstream,
   N_supplier_s]` and the empty-ZE share. It decides the cost of everything downstream and it is
   pure arithmetic.

---

## Part III — Why the simulate-`N_s` design was chosen

Recorded so the choice is not relitigated. **(a)** simulate `N_s` varieties on realised outcomes;
**(b)** estimate `q_ls` from a large pool and apply `Pr(K = 0) = (1−q)^{N_s}` analytically.

**The settling argument.** Under (b) the estimation never solves a finite-variety network: the
value block sits at the certainty-equivalent index while the extensive margin is granular. That
leaves "how do you solve the economy at `N̂_s`?" unanswered — and that economy is the object of
interest, since comovement is propagation through the *realised* network. The solver must exist
regardless, so estimating on a different economy than the counterfactuals is a consistency
liability for no gain.

**Secondary arguments.** (a) matches the binding function, cancelling the auxiliary estimator's
finite-sample bias — 2–4% of `β` at data-consistent group sizes (V12). And there is no `q̂` to
resolve in the moment path: a cell with a tiny win probability simply wins zero varieties.

**What is *not* an argument.** Variance at equal draw budget is a wash — estimating `E[G(0)]` with
20,000 variety-draws gives bias −0.00000 / sd 0.00029 for (a) against +0.00007 / 0.00023 for (b).
And the "support mismatch" objection to (a) does not stand: `𝒜⁺` is a choice of which areas to
study, within a retained area (a) draws firms in every ZE and lets Ricardian competition decide,
and a degenerate group contributes nothing to `β̂` — which is what the fixed-effect cloglog does on
either side. What survives is V13, a diagnostic.

**What (b) keeps.** The `N_s` inner loop, where profiling on realised economies would need
`R ×` (number of candidates) regressions per evaluation, and the cross-check V4.

---

## Part IV — Empirical-target script

The script producing the β target must run the same regression, on the same unit, with the same
support and the same regressors as the model.

**Correct as they stand** — no regeneration needed:

* the outcome is `not_supply`, which is the exact firm-level form;
* the firm-level unit, with `log z` as control;
* the four distance bins and their extraction order match `distance_bin(d, 4)`:
  `(50,100] → 1`, `(100,150] → 2`, `(150,200] → 3`, `d > 200 → 4`, `d ≤ 50 → 0` = reference.

**To fix:**

1. **`nunique() == 2` is the wrong filter.** It conditions on a *realised* outcome, which cannot be
   reproduced in the model. Use `groupby("A129_AA")["supplier"].transform("max") == 1` — the same
   `𝒜⁺_s` rule as `AA_ACTIVE` and the `G(K)` denominator. Define `𝒜⁺_s` **once**, export it, use it
   for the regression, the `G(K)` denominator and `CELL_MASK`.
2. **The distance floor.** Julia uses `log(max(d, 1.0))` (`load_parameters.jl:510`); the draft has
   no floor, so a ZE hosting its own downstream plant gives `−inf`.
3. **Which distance.** Julia's regressor is the distance to the **nearest** downstream region.
   Confirm `dist_com_w_arithmetic` is that, not an arithmetic mean over all downstream plants.
   The same partition must underlie `attraction_area_linkages.npy` and the `A129_AA` grouping.
4. **Make the bins explicit rather than lucky:**
   ```python
   df["dist_bin"] = pd.cut(d, [0, 50, 100, 150, 200, np.inf],
                           labels=["0-50","50-100","100-150","150-200","200+"], right=True)
   # "... + C(dist_bin, Treatment(reference='0-50')) + ..."
   ```
   `right=True` reproduces Julia's `50 < d <= 100`.
5. **Report the `log z` coefficient** — it should be near `−θ = −1` (V6).
6. `cov_type="HC1"` is fine for eyeballing but must not be the source of the β block of Σ, which
   comes from the joint bootstrap that also produces the γ and G blocks and their
   cross-covariances. Cluster at `A129_AA` when reporting.
7. Confirm `A129_AA` concatenates **sector × AA**, matching
   `(s−1)*R_downstream + CLOSEST_DOWNSTREAM_REGION[r]`.
