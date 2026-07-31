# Validation gates and open assumptions — granular / attraction-area version

Companion to `plan_granular_aa.md` (what to build) and `finite_sample2.tex` (the model).
**Nothing here is part of the implementation.** These are the checks to run afterwards, one at a
time, and the assumptions whose cost has been measured but not removed.

---

## Part I — The assumption that matters most

### A.1 `N_s` is estimated without its feedback into the value moments

**What is assumed.** The variety count is located by a closed-form root-find on the count moment
(`plan_granular_aa.md` D5) while the value block — labour share, `π_s`, `π_r`, the sourcing shares
— is computed at the **certainty-equivalent** price index, i.e. from the full draw set rather than
from a realised economy of `N̂_s` varieties. Downstream firms optimise against `E[P^{1−ν_s}]^{1/(1−ν_s)}`, neglecting
granular price randomness.

**Why it is needed.** If the price index of a realised `N̂_s`-variety economy drove the general
equilibrium, `N_s`
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
| **V1a** | Σ layout | every Σ file has `size = N_REG + n_γ + S`; legacy mode slices to the leading `N_REG + n_γ` block; granular keeps all three | assert on load — a wrong size means the file was regenerated with a different block set |
| **V1b** | Filter containment | every cell with `filter_N_upstream == 1` lies in an attraction area active in its own sector | if it fires, the filter is broader than `𝒜⁺`; intersect and warn rather than proceed |
| **V2** | `N_s` root-find | on synthetic `q̂`: `G(s,·)` monotone decreasing; bisection recovers a planted integer; clamps fire at both bounds | exact integer recovery |
| **V3** | AA-level Sinkhorn | round-trip recovery of a planted `T_aa` from its own area aggregates | ~1e−8, mirroring `test/test_ge_inversion.jl` |
| **V6** | Firm↔champion mapping | distribution of model `log z` (champions) against the empirical firms' `log z`; and `b_logz ≈ −θ = −1.0` on **both** | report the overlap and the coefficient; a large gap invalidates the coefficient comparison and calls for A.2 option 2 |
| **V7** | Two routes to `N_s` | `N̂_s` from `G_s(0)` against `N^count_s = N_supplier_s / Σ_l q̂_ls` | same order of magnitude; a large gap is a mechanism finding, not a bug |
| **V8** | Whole count curve | simulated `G_s(K)` for `K ≥ 1` against `G_K.csv` (only `K=0` is targeted) | reported max deviation — a genuinely untargeted fit check |
| **V9** | Bounds not binding | `clamped == :none` for every sector at the optimum | a persistent clamp is a rejection signal for the mechanism, not a numerical nuisance |
| **V10** | `N_s`-invariance of block 4 | drift of `reg_coef` across `N_s` at `θ̂` | **exactly** 0. The regression runs once on the ordinary draws and `N̂_s` never enters it, so this is a structural assertion, not a tolerance: any drift is a coding error |
| **V11** | **Value-block sensitivity (tests A.1)** | recompute `π_s`, `π_r`, labour share at `N̂_s` and `2N̂_s` under the *realised* price index of a drawn `N̂_s`-variety economy | quantifies exactly what A.1 assumes away; the table above predicts a few percent |
| **V12** | Binding-function bias | at `θ̂`, `E[β̂(sim)]` against `β(E[y])` | **the price of the design** (Part III): block 4 is `β(E[y])`, so this bias is not differenced out. Measured at 2–4% of `β` in configurations matching the data; report it with `α̂` or add it as a local offset |
| **V13** | Group-emptying rate | implied share of `𝒜⁺` groups with no supplier, `∏_{l∈a}(1−q̂_ls)^{N̂_s}` | governed by suppliers per group: 0.7% at 5, 3.7% at 3.3, 14% at 2, 37% at 1. Count the active groups first — a high rate means much of `β̂`'s support is degenerate |

### Phase 0 — before any of this

Two measurements that cost nothing and size everything else.

1. **Time one loss evaluation.** `n_good` goes from ~142 to ~1161, so the regression goes from
   `n_good × N_rho` rows to `n_cells × N_rho`. That is the whole cost change — granularity
   itself adds nothing, since `N_s` never enters the simulation.
2. **Bracket `N̂_s` from `G_K.csv` directly**, using the bounds `[N_supplier_s / R_downstream,
   N_supplier_s]` and the empty-ZE share. It decides the cost of everything downstream and it is
   pure arithmetic.

---

## Part III — Why `N_s` is never simulated

Recorded so the choice is not relitigated. Two designs were on the table: **(a)** simulate `N_s`
varieties and compute the moments on realised economies; **(b)** estimate `q_ls` from the full
draw set and apply `Pr(K = 0) = (1−q)^{N_s}` analytically. **(b) is what is implemented.**

**The settling argument.** Under (a), `N_s` reaches the moment vector through exactly one block
that (b) does not already handle in closed form — block 4 — and it reaches it only through the
finite-sample bias of the auxiliary cloglog. Everything else is identical: by Lemma 2 `q_ls` is
free of `N_s`, so `q̂` and hence `N̂_s` are the same object under both; by Proposition 1 the
firm-level index carries no `N_s` term, so the *estimand* of block 4 is the same under both; and
by D2 the value block already sits at the certainty-equivalent index computed on the full draw
set, so (a) was never solving a finite-variety economy for blocks 1, 2, 3 or 5 either.

So the question reduces to: is `E[β̂(N_s)]` or `β(E[y])` the better target for block 4? Design (a)
buys the first, which cancels the auxiliary estimator's finite-sample bias against the identical
bias in the data — worth 2–4% of `β` (V12). It pays for it by treating simulation noise on one
block with a bespoke replication device while every other block relies on the accuracy of the draw
design, and by costing `R_rep ×` more per loss evaluation. That asymmetry has no counterpart
elsewhere in the estimator, and the closed form removes the need for it entirely.

**Two things (b) gets for free.** The count moment `Ḡ_s(n) = mean_l (1−q̂_ls)^n` *is* the
expectation of the realised empty-cell share — by linearity over cells, the dependence between
cells does not affect a mean — so it is unbiased **and** noise-free, and the `N_s` inner loop is a
bisection on it with no re-simulation. And block 4 becomes exactly `N_s`-invariant rather than
approximately so, which turns V10 from a tolerance into an assertion.

**What is *not* an argument.** Variance at equal draw budget is a wash — estimating `E[G(0)]` with
20,000 variety-draws gives bias −0.00000 / sd 0.00029 for (a) against +0.00007 / 0.00023 for (b).
And the "support mismatch" objection to (a) does not stand either: `𝒜⁺` is a choice of which areas
to study, and a degenerate group contributes nothing to `β̂` on either side.

**What must be reported.** The V12 bias, alongside `α̂`. It is the one thing design (a) would have
removed, and it is now an assumed, measured cost rather than a differenced-out one.

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
