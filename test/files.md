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
| `test_analysis_granular.py` | **Gate for the Python reporting library** `utils.py` + `report_lib.py` (granular + attraction-area), which is what `model_report.ipynb` calls. Not Julia and needs no data: it writes a SYNTHETIC baseline + reporting tree with the exact file layout `main.jl --granular=true --ca_level=aa` produces, then runs the **shipped modules** against it (it used to execute the notebook's cells, and had to guess which of them defined and which ran). Checks the AA mapping and area names (`X_dr.query('downstream').ze2010_downstream`, in the model's AA-column order), the AA-level γ aggregation and its s-major flattening, MOMENT_MASK (first industry / first π_r / inactive γ / per-sector reference area dropped, block 6 never masked), the block split of `best_simulated_moments.npy`, the μ₁/μ₂ folder routing (step1 + step2/inference vs step3 + step3/inference), the β→γ→G standard-error split of `Sigma_data.npy` / `se_moments_fitted.npy`, the reference-area reconstruction `γ_ref,s = c_s − Σ_{a≠ref} γ_{s,a}` (planted so it must return the simulated reference value exactly), the WLS-through-origin fit against the Julia dashboard's formula, the joint (two-industry) LaTeX table, both Jacobian axes against the shape of `jacobian_all[_step3].npy` — including the `S`
variety-count columns Julia appends on the right, whose entries must be EXACTLY zero
outside the count-moment block — the noise-to-signal ratio and mask, the Σ/Ω panels and
their correlation companions, and that `globals().update(data)` binds the analysis.ipynb names. Usage: `python3 test/test_analysis_granular.py` (needs numpy/pandas/matplotlib). |
| `test_modules.py` | **Gate for the module layout and the two properties the refactor promised.** (1) ONE definition per name across `utils.py`, `report_lib.py`, `diffusion_lib.py` and `granular_lib.py` — the notebook layout carried 75 duplicated definitions and the gates policed the copies for drift; this makes the duplication impossible instead. (2) The SPECIALISED loader: a run tree with `best_simulated_moments.npy`, `Sigma_*`, `jacobian_*` and `G_K.csv` DELETED must open under `parts=("core","geography")`, and asking for a part whose files are missing must fail. (3) The PARQUET-FREE path: with `data["suppliers"]` absent, every regime is built from `theta+`, `D_r` closes against the value block, and the two counterfactual routes are separated — `simulate_economy` re-solves the whole map so `D_r` MOVES with the regime, while `counterfactual_diffusion_frame` reallocates at a fixed sector spend so it cannot. Usage: `python3 test/test_modules.py`. |
| `test_notebooks.py` | **Gate for the two notebooks.** Statically: every code cell of `model_report.ipynb` and `tests_counterfactuals.ipynb` parses, and every free name it uses is bound by the libraries it star-imports or by an earlier cell — the property the old single-namespace layout gave for free and that a notebook of imports has to earn. Functionally: the imports, the Constants cell and the economy run cell execute against a run tree carrying no `suppliers.parquet` and produce every regime at the notebook's own replication count. Usage: `python3 test/test_notebooks.py`. |
| `_nbmod.py` / `_fixture_tree.py` | Helpers, not gates. `_nbmod.install` loads the libraries into a gate's namespace with the fixture's stubs installed as MODULE attributes — the module equivalent of the old `exec(code, globals())`, and what retired the three hacks that sliced a function body out of a cell by string index. `_fixture_tree.build()` writes the minimal run tree `test_modules.py` and `test_notebooks.py` share. |
| `test_extensive_margin.jl` | Geometry screen (Phase-2 "GATE G2") for the analytical `reg_coef` moment — see the detailed note below. |
| `test_t_reorder.jl` | Guard test for the s-major flattening convention of the T parameters. Asserts the T-parameter axis and the γ-moment axis enumerate the (sector, region) pairs in the same order — a silent mismatch would fit the wrong T. |
| `test_ge_inversion.jl` | Phase-0 feasibility gate for T-profiling (`invert_T_ge`): round-trip recovery, GE-Sinkhorn convergence (`ρ_full`, `κ_S`, `‖J_GE‖`), uniqueness, cost. |
| `test_cloglog_verify.jl` | **Correctness gate for the hand-rolled cloglog IRLS kernel.** Checks `_cloglog_irls` (model_CP.jl — cloglog GLM fit by IRLS with one FE absorbed via weighted FWL demeaning) against GLM.jl's `glm(..., Binomial(), CloglogLink())` on synthetic data (continuous regressors, distance-bin dummies, and frequency weights verified by row-expansion). Asserts the slope coefficients match to ~1e-5. Requires GLM.jl. Usage: `julia test/test_cloglog_verify.jl`. |
| `test_cloglog_streaming.jl` | **Correctness gate for the STREAMING extensive-margin kernels** — `_cloglog_irls_cells` and `_wls_cells`, the memory-light replacements that are the production path for both links, and which never materialise the `n_cell × N_rho` design. Section 1 compares the cloglog kernel against the dense `_cloglog_irls` and against GLM.jl; section 1b compares the LPM kernel against the dense FWL/QR kernel and against an independent explicit-dummy OLS. Both cover all four production design shapes (bin dummies / continuous log-distance × size control on / off), with and without control cells, plus a bin-homogeneous FE group (the worst case for the closed-form within transform's cancellation). Section 3 runs the full `fast_cloglog_regression` and `fast_weighted_regression` on both paths (`REG_STREAMING[]` toggled), which also exercises the geography → cell mapping and the `log z = logz_const + logz_resid` decomposition; section 4 prints the design/allocation/time comparison at production shape. **Tolerances are deliberately asymmetric**: ~1e-10 against the dense kernels (same fixed point, different summation order — this is the gate that catches an algebra error) but 1e-5 against GLM.jl, which stops on the deviance and so pins β only to ~1e-6. Standalone (defines the geography globals itself — no data or worker pool). Requires GLM.jl. Usage: `julia test/test_cloglog_streaming.jl`. |
| `test_control_group_regression.jl` | **What do the exogenous extensive-margin zeros change?** At a fixed θ̂ (loaded from an estimation run) it solves the network once and evaluates the distance regression twice via `fast_weighted_regression(...; include_control=…)`: WITH the control group (filter==2 y=0 rows, the production moment) vs WITHOUT (supplier pairs only). Both drop the log-z size control; only the control-group rows toggle. Prints the two coefficient vectors, their difference, the empirical target, and the per-bin supplier/control pair composition. Usage: `julia test/test_control_group_regression.jl auto 4 1` (or pass the run folder as a 4th arg). |
| `test_T_convergence_map.jl` | **WHERE does the GE-Sinkhorn T inversion fail?** The PSO log's "T non-converged: n/N particles" is a pooled number that cannot say why. This maps the convergence indicator over the same fresh-start head box `train_stage` builds, and separates two explanations: (H1) the α box reaches too high — τ = d^α spreads trade costs over many orders of magnitude and `invert_T_ge` has to push remote-origin T correspondingly hard to hold γ_ls fixed; (H2) the failures live elsewhere in the head (Ω^L, Ω^s, A), in which case narrowing α will not help. Section 1 sweeps α alone at the warm-start head (a clean threshold there means `ALPHA_MAX` is the whole story and gives the number to set it to); section 2 samples the full box and reports the rate by α bin; **section 3 is the "are there other regions" test** — it conditions on the low-α half and asks whether the failures are drawn from a different part of any *other* block's box (each coordinate mapped to its position in its own box so blocks on different scales are comparable), plus one-at-a-time corner probes; section 4 dissects the failures into "slow" (residual just above `tol`, fixed by 4× the iterations) vs "oscillating" (fixed only by heavier `damping`) vs "not contracting" (fixed by neither — a region the box should not contain). Writes `T_convergence_map.npz` and a rate-vs-α figure. Usage: `julia test/test_T_convergence_map.jl auto 4 1 true aa` (add `-p N` to parallelise the box sample). |
| `test_pipeline_timing.jl` | **Which part of the SMM pipeline is longest, and where does the memory go?** Breaks one `full_SMM` evaluation into its real phases — `unpack_params`, `build_tau`, `solve_network`, the block-4 regression, the rest of `compute_moments`, `moments_to_vec`, `loss_function` — reporting wall time *and* bytes allocated for each, with the phase sum reconciled against `full_SMM` end to end. Section 2 times `invert_T_ge` separately, since under `profile_T` it is paid before every particle and scales with the iteration count the convergence diagnostic reports. Section 3 sweeps `N_rho` at ×0.5/×1/×2 to show which phases are linear in it and which are not. **Section 4 measures the `pmap` round trip**, contrasting a closure that captures `U_DRAWS` by value against one that lets the `@everywhere const` resolve on the worker — the difference is pure serialization of data every worker already had, and is the churn behind a worker RSS that climbs across a stage. Section 5 extrapolates to a 200-iteration stage and states the allocation-turnover vs retained-memory distinction. Usage: `julia -p 8 test/test_pipeline_timing.jl auto 4 1 true aa true`. |
| `test_profile_alpha_sweep.jl` | **Why does profiling push α → 0?** Sweeps α on a grid at a fixed head (Ω,A); at each α contrasts the reg_coef / γ_ls / β+γ loss under the PROFILED regime (`T=invert_T_ge(α)`, γ_ls pinned ∀α) vs the FIXED-T regime (`T=T̂` constant). Shows that reg_coef has an interior α-optimum with T fixed but collapses to the α→0 boundary once T is profiled — the exact-γ_ls T*(α) adjustment cancels α's τ-channel leverage on reg_coef. **Also contrasts reg_coef WITH vs WITHOUT the control-group y=0 rows** (`include_control` true/false) at each α: since `invert_T_ge` pins γ_ls from supplier pairs only, T*(α) is identical in both variants, so this isolates whether the far-distance control zeros restore an interior α* under profiling. **Granular/AA aware:** the 7th/8th args are `granular` and `ca_level` and they are **not optional decoration** — `load_parameters.jl` probes them with `@isdefined`, so omitting them silently loads the LEGACY continuum model (`granular=false, ca_level=:ze`) *and*, because `gb_indices` then mis-sizes the on-disk `W_step3`, silently falls back to identity weighting. Pass the same flags used for the run. Under `GRANULAR` the moment set comes from `inference_moment_indices()` (so block 6, the count moment `Ḡ_s(0)`, is included), the T blocks are handled in the **T-COLUMN space** via `unpack_T_par` (`unpack_params` returns the gathered `(S,R)` ZE matrix, which is a `DimensionMismatch` for `invert_T_ge`/`assemble_theta` under `:aa`), and each α additionally reports N̂_s, the number of sectors **clamped** at a variety bound, and the `Ḡ_s(0)` error. The α grid is extended with α̂ and the production box floor (`BOUND_LO × TAU_PRIOR`) so the boundary the optimizer converges onto is actually visited. **Key output: the weighted criterion `r'Wr` decomposed into per-block diagonal contributions `r_b'W_bb r_b`** (cross terms reported as the remainder) with each block's own α-minimiser — under profiling γ is pinned at every α, so this is what separates a reg_coef-driven α from a count-moment-driven one. The with/without-control arms are **skipped** under `CA_LEVEL=:aa`, where `REG_INCLUDE_CONTROL` is forced false and the two arms coincide. **9th arg `relax_n_lo`** (same `@isdefined`-before-include rule) forces `N_LO = 1`, so the α sweep can be re-run with the variety-count clamp removed and the clamp's share of the α gradient read off directly. **New `medΣq` column** reports the median over sectors of `Σ_l q̂_ls = E[#origins winning a variety somewhere] ∈ [1, min(N_d,R)]`: at α=0, `τ ≡ 1` makes `Φ_dr` common across buyers so one cell wins a variety for all of them and `Σ_l q̂ = 1` exactly; raising α decorrelates winners and pushes it up, which lowers `Ḡ_s(n) = mean_l (1−q̂_l)^n` in LEVEL. Rising `medΣq` with falling `medN̂` and growing `G0err` identifies that union channel — as opposed to the within-area concentration (convexity) channel, which pushes `Ḡ` the other way. Usage: `julia test/test_profile_alpha_sweep.jl aero 4 1` (legacy; anchors θ̂ on the non-profiled PSO run `reporting_<industry>_pso`), `... ./reporting_aero_profiled_aa_gran_pso cloglog false true aa` (granular/AA), or `... ./reporting_aero_profiled_aa_gran_nlo1_pso cloglog false true aa true` (granular/AA with `N_LO = 1`). |

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

Args: `industry n_coef n_tau granular ca_level`.

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
* **`Ḡ_s(0)` monotone in `N_s`** — the property the bisection relies on.
* **V10 — `N_s`-invariance of block 4.** Structural, not statistical: block 4 is computed once
  on the ordinary draws and `N̂_s` never enters it, so `reg_coef` must be **exactly** invariant.
  The gate confirms the code path really is `N_s`-free.

**Reported diagnostics** (informative, not implementation bugs):

* **V6 — firm ↔ champion.** `b_logz` against `−θ` (Prop. 1(c)). A large gap calls for
  `granular_validation.md` §A.2 option 2.
* **V7 — two routes to `N_s`.** `N̂_s` from `Ḡ_s(0)` against `N^count_s = N_supplier_s / Σ_l q̂`.
  A large gap is a mechanism finding, not a bug.
* **V9 — bounds not binding.** A persistent clamp is a rejection signal for the mechanism:
  `:hi` means the model cannot generate enough sparsity even when every variety is sourced
  from a single origin.
(V4 and V5 no longer apply: the count moment is the closed form, so there is nothing realised
to compare against, and no prefix of the draws is ever taken, so there is nothing to be stable.)

Gates needing a fitted `θ̂` or an external reference (V0, V8, V11, V12, V13) are not run
here; see `granular_validation.md`.

**Status.** Purely diagnostic — it changes no estimate, weight matrix, or production file.

## `test_analysis_granular_sections.py`

Gates the reporting sections, now in the four modules — the identification / sensitivity
Jacobian (the variety-count columns and the noise-to-signal mask), the untargeted-moment
PPML (`pyfixest.fepois`, gated against `statsmodels`' Poisson GLM with explicit dummies,
plus the decomposition identity and the fixed-effect-absorbed distance profile), the
within-sector
comparative-advantage comparison, the amplification measures $D_r$ and $L_r(d)$, and the
input-output / Leontief benchmark. No Julia and no real data: it writes
synthetic run trees with the exact file layout the loader expects, then executes **the
notebook's own code cells** against them (located by content, not by index, so inserting
a section above does not change what is tested). The notebook alternates DEFINITION cells
with cells that RUN the reporting; a cell counts as a run cell — and is skipped — when it
hits one of `RUN_CELL_MARKERS` **and** defines nothing at top level, so a new per-section
`for cfg in INDUSTRIES:` cell needs no special handling.

```bash
python test/test_analysis_granular_sections.py
```

Needs `numpy`, `pandas`, `matplotlib`, `statsmodels`, `pyarrow` and `pyfixest` (the
notebook estimates the untargeted moment with it, so it is no longer optional). The reasoning
behind each gate is written out in the notebook's own "Validation of the sections above"
markdown cell.

---

## `test_local_share_dispersion.py` — the local share as a level plus a dispersion

Gate for Test 9 of the amplification section (`granular_lib.py`). The local share is LINEAR in
the realised network, so `E[sum_l h_l omega_lrs] = p_rs` exactly and
`Var = V_rs p_rs (1 - p_rs)` for any indicator `h`: the level carries no granularity and the
whole of it sits in the bar. The fixture is the planted economy of
`test_concentration_identity.py` — a parquet drawn variety by variety FROM the geometry with
equal expenditure across a buyer's varieties, so `V = 1/N_s` exactly — and the notebook's
own cells are executed against it.

```bash
python test/test_local_share_dispersion.py
```

Needs `numpy`, `pandas` and `matplotlib`; no data, no Julia. Ten gates, and four of them
are the ones worth knowing about. The MEASURED standard deviation across 200 replications is
compared against the closed form, two routes sharing no code (median ratio within 10%). The
shape claim `p(1-p)` is gated on the DRAWS rather than on the closed form, where it would be
a tautology: sweeping the radius moves `p` across its range and the empirical dispersion must
track `sqrt(V p(1-p))` and peak in the bin containing one half. And the plan's conjecture —
that cutting a force can lower the point while RAISING the bar — is gated cell by cell as the
equivalence it is: at a fixed `V` the bar grows if and only if `p` moves TOWARDS one half, so
the compensation needs a crossing and is not available wherever `p` sits well below it.
Finally the drawn band is checked to BE the empirical 10-90 range of the draws about the
point, clipped at zero, with the asymmetry gated as an ORDERING (a radius where `p` is small
must give a more right-skewed band than one where `p` is near one half) so it cannot pass by
luck on one configuration, and `p x N_eff = local / V` — the count of effective varieties
landing locally, which says whether a wide band is granular or merely a few events.

---

## `test_alignment_covariance.py` — Test 3 bis, the alignment in kilometres

Gate for `Cov_rho(log T, d)`, the rate the `Distance only` counterfactual integrates. It
exists because the section had none: `alignment_frame` was deleted with the
comparative-advantage section while Test 3 bis kept calling it, so the cell raised
`NameError` at run time with nothing to catch it. Gate 1 is that regression — the cell must
define its own `alignment_frame` and `_buyer_weights`, and the helper must be byte-identical
to Test 8's copy, since two copies are only safe while they agree.

```bash
python test/test_alignment_covariance.py
```

Needs `numpy`, `pandas`, `matplotlib` and `scipy`; no data, no Julia. Six gates. The
covariance is reproduced from a RAW-moment route (`E[xd] - E[x]E[d]`), algebraically
distinct from the centred form the code evaluates, so a sign or weighting slip cannot
cancel out; `Cov(log d, d) > 0` everywhere (two increasing functions of the same variable —
a check, not a finding); a planted ALIGNED geometry comes out negative at the median while
an orthogonal one does not, with the per-buyer sign flip the aggregate hides left visible
rather than asserted away; and the buyer aggregation is the spend-weighted mean of its
sectors, refusing both a missing parquet and a parquet whose region index does not line up.

---

## `test_diversification_identities.py` — the theory note's algebra

Numerical gate for `documentation/diversification.md`, the companion note on customer
diversification. It re-derives every displayed identity of that note — the Herfindahl
representation of variance and the Gram representation of comovement, the
extensive/intensive decomposition, the aggregation gain, the softmax form of the cell
portfolio, the four comparative statics of Lemma 1, the closed-form extensive margin
`E[n] = Σ_r γ`, the FKG bound, the tilting identity behind `δ/γ = −θα κ (Λ̄ + 1 − H̃)`,
and the planner's variance return — by finite differences, Monte Carlo over the
Ricardian assignment, and exact arithmetic.

```bash
python test/test_diversification_identities.py
```

Needs only `numpy`; no data, no Julia. Exit code 1 on any failure. Three tests are there
specifically to catch claims that a first draft of the note got wrong and that are easy
to reintroduce: that a cell's **own** comparative advantage *does* move the composition
of its customer portfolio (through `Φ`, obeying the same elasticity as any rival's `T`);
that the `α → ∞` limit is governed by **relative** log-distance, not by the nearest
destination; and that the note's equation (20) needs a *single* upstream sector, not
merely the absence of cross-sector spillovers.

## `test_extended_economy.py` — the extended parameter set and the forward map

Gates `extended_parameters` / `simulate_economy` / `economy_identities` /
`check_against_julia`, i.e. the design in which ONE implementation produces the estimated
economy and both counterfactuals from `theta+ = (Omega_L, Omega_s, A, alpha, T, N)`.

`theta+` is written down BY HAND in the fixture rather than read from a run tree — that is
the point of the object — over the planted economy of `test_concentration_identity.py`
(S = 3, R = 12, 5 buyers, `N_hat = [4, 10, 30]`).

Twelve gates. (1) the draw-matrix column order is region-outer/sector-inner, as Julia's
column-major `findall` on the `(S, R)` cell mask walks it, with the C-order alternative
shown to differ so the gate is not vacuous. (2) the four identities to machine precision.
(3) `P_r`, `c_r`, `c_tilde_r`, `D_r`, `theta_rs`, `P` and `Y_r` against a hand
recomputation from `theta+` alone — including `D_r = 1 + (1-Omega_L)(P_r/c_r)^(1-lambda)`,
which holds only because the two CES indices collapse, so it fixes the whole expenditure
chain in one number. (4) determinism in `(theta+, draws)`, and the separating check that
the HEAD moves the value block while leaving the winners untouched (the Ricardian argmin
carries no price index). (5) the realised winner frequencies converge to
`sourcing_geometry`'s closed-form `rho`, scored as a z against the binomial standard
error — this is what ties the simulator to the structural half. (6) the exact controls:
`alpha = 0` gives one winner per variety for every buyer, equalising `T` sends the euro
further, `N` is held across regimes. (7) a supplied `(N_max, n_good)` draw matrix
reproduces a hand recomputation exactly, and a wrong shape or count is refused up front
rather than failing as an out-of-range index. (8) zero varieties, a mis-sized `A` and an
empty sector are refused by name. (9) the cross-language comparator, gated against a
parquet written FROM the simulator in Julia's own schema: that leaves its index arithmetic
under test (1-based `replication`/`variety`, the buyer map, `rep*N + variety`, dropped
zero-share rows), and it must catch a single flipped winner and a 5% share perturbation
SEPARATELY and refuse a column map that disagrees with Julia before comparing anything.

(10) `theta` is parsed out of `load_parameters.jl` and wins over a disagreeing
`stats.csv` -- the definitions are sliced out of the loader cell BY TEXT, so the shipped
source is gated without pulling the whole tree reader over the fixture's stubs. (11) the
frame round trip: `economy_frame` emits `suppliers.parquet`'s own schema, one `SIREN` per
`(replication, cell, sector, variety)` and never shared across replications, and the
notebook's own `variety_panel` / `_sector_spend` (also sliced by text) read the economy back
out of it -- which is what makes the reporting stack regime-agnostic. (12) `economy_by_regime`
reads `theta+` once and returns every regime with its own frame, `N` held fixed, the caller's
`data` unmutated, and `D_r` moving across regimes where the two-route arrangement had to hold
it at the baseline.

**What it does not establish**: whether Julia agrees. Only a run with `post_hoc_u.npy` on
disk can say that — `check_against_julia` is the call, and it is the first thing the
section's run cell does.
