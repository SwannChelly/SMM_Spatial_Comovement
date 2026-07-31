# `test/` — validation and diagnostic scripts

Scripts that check the estimator is doing what it should. None of them is required
for a production run — they are run by hand when you want to *validate* a result or
*diagnose* an identification problem. Each one re-uses the same model and data
loading as `main.jl`, so it sees the exact same moments and masks.

| File | Role |
|------|------|
| `run_test.sh` | Launcher for the internal-validity Monte-Carlo (`run_internal_validity.jl`), mirroring `run.sh`. Usage: `./run_test.sh aero --n_coef=4 --n_tau=1 --beta0="0.5"`. |
| `run_internal_validity.jl` | **Does the estimator recover a known truth?** Builds a synthetic economy from a chosen parameter vector θ₀, overwrites the empirical targets with moments generated at θ₀, then re-estimates. Reports point-recovery error and confidence-interval coverage over many Monte-Carlo repetitions. |
| `run_untargeted_validation.jl` | **Out-of-sample check.** Feeds calibrated parameters into `../extras/untargeted_moments.jl` to simulate demand shocks and reproduce the paper's Table 2 comovement regression — moments that were *not* used in estimation. |
| `test_extensive_margin.jl` | Geometry screen (Phase-2 "GATE G2") for the analytical `reg_coef` moment — see the detailed note below. |
| `test_t_reorder.jl` | Guard test for the s-major flattening convention of the T parameters. Asserts the T-parameter axis and the γ-moment axis enumerate the (sector, region) pairs in the same order — a silent mismatch would fit the wrong T. |
| `test_ge_inversion.jl` | Phase-0 feasibility gate for T-profiling (`invert_T_ge`): round-trip recovery, GE-Sinkhorn convergence (`ρ_full`, `κ_S`, `‖J_GE‖`), uniqueness, cost. |
| `test_cloglog_verify.jl` | **Correctness gate for the hand-rolled cloglog IRLS kernel.** Checks `_cloglog_irls` (model_CP.jl — cloglog GLM fit by IRLS with one FE absorbed via weighted FWL demeaning) against GLM.jl's `glm(..., Binomial(), CloglogLink())` on synthetic data (continuous regressors, distance-bin dummies, and frequency weights verified by row-expansion). Asserts the slope coefficients match to ~1e-5. Requires GLM.jl. Usage: `julia test/test_cloglog_verify.jl`. |
| `test_control_group_regression.jl` | **What do the exogenous extensive-margin zeros change?** At a fixed θ̂ (loaded from an estimation run) it solves the network once and evaluates the distance regression twice via `fast_weighted_regression(...; include_control=…)`: WITH the control group (filter==2 y=0 rows, the production moment) vs WITHOUT (supplier pairs only). Both drop the log-z size control; only the control-group rows toggle. Prints the two coefficient vectors, their difference, the empirical target, and the per-bin supplier/control pair composition. Usage: `julia test/test_control_group_regression.jl auto 4 1` (or pass the run folder as a 4th arg). |
| `test_profile_alpha_sweep.jl` | **Why does profiling push α → 0?** Sweeps α on a grid at a fixed head (Ω,A); at each α contrasts the reg_coef / γ_ls / β+γ loss under the PROFILED regime (`T=invert_T_ge(α)`, γ_ls pinned ∀α) vs the FIXED-T regime (`T=T̂` constant). Shows that reg_coef has an interior α-optimum with T fixed but collapses to the α→0 boundary once T is profiled — the exact-γ_ls T*(α) adjustment cancels α's τ-channel leverage on reg_coef. **Also contrasts reg_coef WITH vs WITHOUT the control-group y=0 rows** (`include_control` true/false) at each α: since `invert_T_ge` pins γ_ls from supplier pairs only, T*(α) is identical in both variants, so this isolates whether the far-distance control zeros restore an interior α* under profiling. Usage: `julia test/test_profile_alpha_sweep.jl aero 4 1` (anchors θ̂ on the non-profiled PSO run `reporting_<industry>_pso`; pass a 4th arg to anchor on another run, e.g. `reporting_<industry>_profiled_pso`). |

## `test_extensive_margin.jl` in depth

**What it is.** A standalone, print-only diagnostic (`julia test_extensive_margin.jl aero 4 1`)
that includes the model files and `load_parameters.jl` exactly the way `main_gmm.jl` does,
then *measures* the extensive-margin geometry of the analytical `reg_coef` block without
touching any production path. It exists to answer a single design question: the closed-form
`reg_coef` (`compute_regression_quadrature`) currently approximates each variety's
"win-at-any-destination" probability with the **FKG product** `1 − ∏_dr (1 − ρ_dr)`. That
product treats the per-destination win events as independent, but the competitor Fréchet
draws are *shared* across destinations, so the events are positively correlated and the
product **over-states** the true union. This script quantifies that bias and decides whether
replacing FKG with the exact inclusion–exclusion (the never-merged "Phase 3" refactor) is
both *worth it* (bias large enough to matter) and *feasible* (exact cost tractable).

**The underlying object.** For an origin variety `(sector s, region r_p)` with own
productivity `z`, `r_p` wins destination `dr` against competitor `r'` iff
`z_{r'} ≤ z·(w_{r'}τ_{r',dr})/(w_{r_p}τ_{r_p,dr})`. Collecting the competition into
`Q[r',dr] = T_{r'}(w_{r'}τ_{r',dr})^{−θ}(w_{r_p}τ_{r_p,dr})^{θ} ≥ 0`, the single-destination
log-win probability is `−zinv·Φ({dr})` with `Φ({dr}) = Σ_{r'} Q[r',dr]`, and a self-check
asserts `Φ({dr})` reproduces the code's `coef_dr = T_val − Φ[s,dr](w_pτ_p)^θ` to ~1e-8 (the
`r_p` self-term cancels). Because `Win(dr) ⊆ Win(dr')` exactly when `Q[·,dr] ≥ Q[·,dr']`
componentwise, the union of win-events only needs the **Pareto-minimal** ("least-competition")
destinations `D*` — dominated destinations are already contained.

**The five things it prints.**

1. **`|D*|` histogram across all goods.** How large is the Pareto-minimal destination set?
   The exact inclusion–exclusion costs `2^|D*|` per node, so `max|D*|` sets the worst-case
   cost. Decision rule: `max|D*| ≲ 12` → exact everywhere; otherwise a hybrid (exact below a
   cutoff, pairwise above).

2. **MC vs FKG vs exact-IE vs pairwise, on a handful of `(s, r_p, z)` nodes.** With *shared*
   competitor draws it computes the Monte-Carlo win-anywhere probability (ground truth), the
   FKG product (what the code does), the exact inclusion–exclusion
   `Σ_{∅≠S⊆D*} (−1)^{|S|+1} exp(−zinv·Φ(S))` with `Φ(S) = Σ_{r'} max_{dr∈S} Q[r',dr]`, and
   the pairwise (Bonferroni, `|S|≤2`) truncation. The `z` nodes are chosen at the **low-`u`,
   high-productivity tail** (`u∈{0.02,0.1,0.3,0.6}`, since `z ∝ (−log(1−u))^{−1/θ}` decreases
   in `u`) — the regime where the extensive margin is actually active and the FKG bias bites.
   Exact-IE must track MC to MC noise (`≈1/√N`); the `FKG − MC` gap (level and ratio) is the
   *measured* bias.

3. **Exact union-of-boxes via a dominance-pruned, memoized DAG** (`build_union_dag` /
   `eval_union_dag`). The exact union measure is built with
   `μ(∪rest ∪ B) = μ(∪rest) + μ(B) − μ(∪_i(B∩B_i))`, each box a bitmask over the `D*` columns
   (intersection = mask OR, `μ(box)=exp(−zinv·Σ_c qvec_c)`), boxes contained in another pruned
   at every level, canonical pruned mask-sets memoized. Crucially the recursion *structure*
   lives entirely in z-free q-space, so the DAG is **built once per good and evaluated per
   quadrature node** — exactly the reuse Phase 3 would exploit. `nsub` (distinct subproblems)
   is the true exact cost; the run reports the `nsub` distribution and the `2^|D*|/nsub`
   reduction to answer *does pruning tame the worst `2^|D*|`?* The DAG is validated against the
   brute-force IE (`@assert max|union − IE| < 1e-9`).

4. **`|D*|` stability under β-perturbation** (binned-τ runs only): recomputes `max|D*|` under
   `β×{0.8, 1.25}` to confirm the dominance geometry does not swing with the trade-cost
   parameters. (For power-law `N_TAU=1`, dominance is monotone in distance ⇒ β-stable.)

5. **The reg_coef coefficient-bias gate** (`reg_coef_quad`) — *the number that actually
   decides Phase 3.* It replicates `compute_regression_quadrature` exactly (same
   Gauss-Legendre nodes, regressors, fixed effects, weights) and swaps **only** the regressand:
   FKG product vs the exact DAG union. It then prints `α_FKG` vs `α_exact` per bin with the
   max relative Δ — i.e. the FKG bias on the *estimated distance coefficients*, integrated over
   the whole productivity range (a large per-node ratio can still integrate to a small
   coefficient bias, or not). Goods whose `nsub` exceeds `CAP_SUB` fall back to FKG in this
   pass (count reported), so the run always completes.

**Status.** Purely diagnostic: it changes no estimate, weight matrix, or file. It informs the
still-unmerged decision of whether to replace the FKG `reg_coef` with the exact
inclusion–exclusion; the production `reg_coef` continues to use the FKG product.

---

## `test_granular_aa.jl` — validation gates for the granular / attraction-area estimator

Standalone, print-only gate runner for the model of `documentation/finite_sample2.tex`,
built per `documentation/plan_granular_aa.md` and numbered as in
`documentation/granular_validation.md` Part II.

```
julia test/test_granular_aa.jl aero 4 1 true  aa      # granular + AA
julia test/test_granular_aa.jl aero 4 1 false ze      # legacy reference
```

Args: `industry n_coef n_tau granular ca_level [n_rep]`.

**Enforced gates** (a failure means the implementation or the inputs are wrong):

* **V1 — AA map.** `attraction_area_linkages.npy` has shape `(R, R_downstream)`, rows sum to
  1, and `argmax_col == CLOSEST_DOWNSTREAM_REGION`. The last is decisive: the model's fixed
  effect and the empirical `A129_AA` grouping must be the SAME partition, or the alignment
  argument of `finite_sample2.tex` §1.2 fails. Also asserted at load time.
* **V1a — Σ layout.** Every Σ file carries `N_REG + n_γ + S` rows (β → γ → G).
* **V1b — filter containment.** Every `CELL_MASK` cell lies in an attraction area active in
  its own sector (`𝒜⁺`).
* **V2 — `N_s` root-find.** `G(s,·)` is strictly decreasing on `[N_LO, N_HI]`; the bisection
  recovers a planted integer exactly; the clamps fire at both bounds.
* **V3 — AA-level Sinkhorn.** Round-trip recovery of a planted `T` from its own area
  aggregates (~1e−13 in practice), mirroring `test_ge_inversion.jl`.
* **V5 — prefix stability.** The winners of the first `n` varieties are identical whether the
  Ricardian solve runs at width `n` or at pool width. This is the property that licenses ONE
  solve per replication (plan D1); without it every candidate `N_s` would need its own solve.
* **`Ḡ_s(0)` monotone in `N_s`** — the property the bisection relies on.

**Reported diagnostics** (informative, not implementation bugs):

* **V6 — firm ↔ champion.** `b_logz` against `−θ` (Prop. 1(c)). A large gap calls for
  `granular_validation.md` §A.2 option 2.
* **V4 — closed form vs realised.** `report_granular` (and `report.txt`) print the closed-form
  `Ḡ_s(N̂) = mean_l (1−q̂)^{N̂}` the bisection targets beside the REALISED empty share. A
  material gap points at the winner accounting, or at a stratified/QMC draw design making the
  prefix counts under-dispersed relative to `Binomial(N̂_s, q)` — in which case use `--draws=mc`.
* **V7 — two routes to `N_s`.** `N̂_s` from `Ḡ_s(0)` against `N^count_s = N_supplier_s / Σ_l q̂`.
  A large gap is a mechanism finding, not a bug.
* **V9 — bounds not binding.** A persistent clamp is a rejection signal for the mechanism:
  `:hi` means the model cannot generate enough sparsity even when every variety is sourced
  from a single origin.
* **V10 — `N_s`-invariance of block 4.** `reg_coef` drift between `N̂` and `2N̂` should be ~0,
  since the firm-level index carries no `N_s` term. A visible drift means the union-over-buyers
  approximation or the firm mapping is doing something unintended — or that `R_REP` is too
  small for the binding-function average to have settled.

Gates needing a fitted `θ̂` or an external reference (V0, V4, V8, V11, V12, V13) are not run
here; see `granular_validation.md`.

**Status.** Purely diagnostic — it changes no estimate, weight matrix, or production file.
