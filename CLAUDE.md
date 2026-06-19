# SMM Spatial Comovement — Codebase Guide

## Changelog

| Date | Files | Summary |
|------|-------|---------|
| 2026-06-19 | `run_internal_validity.jl` (new), `CLAUDE.md` | Add a standalone **internal-validity Monte-Carlo driver**. **Additive only** — no existing file is modified, so every production entry point (`main.jl`, `main_gmm.jl`) is byte-identical. It builds a synthetic truth `θ₀ = [agg_labor_share \| agg_industry_share \| A_gravity \| β₀ \| vec(permutedims(T_rs_init))[T_MASK]]` (A_gravity = Stage-0 analytical inversion; β₀ CLI-supplied, asserted monotone), then **overrides only the targets**: after `include("load_parameters.jl")` (so all masks / `T_REF_REGION` / `MOMENT_MASK` / `gamma_threshold` are frozen from the REAL baseline), `set_synthetic_targets!` mutates the const arrays `empirical_moments .= reshape(masked m(θ₀),1,:)` and `reg_coef .= m(θ₀)[BLOCK_RANGES[4]]` **in place** across master+workers (`@everywhere`, no const rebinding → compiled `full_SMM`/Stage-0 see new targets, no staleness). `emp_gamma_ls`/`X_rs`/`emp_pi_r_full` stay REAL. m(θ₀) is generated with a SEPARATE draw seed (`generate_stratified_draws(...; randomise=true, rng=MersenneTwister(999_000))`) so the estimator's production `U_DRAWS` never match the target-generating draws. **Section 1** records `min_g ESS_g` (production draws) and the `H[T,T]` λ_min via `screen_T_identification(θ₀; J=…)`. **Experiment 1** (full-vector point recovery): 10 perturbed starts (×LogUniform[0.5,2] on A,β,T; ×U[0.8,1.2] on Ωᴸ,Ωˢ; β re-sorted), each run through the full three-step (`run_pso_optimization` step1 `skip_initial_beta_search=true,warm_start_params=pert` → `build_step3_weight_matrix` → step3 `gamma_beta_only=true` → step4 `compute_jacobian`+`compute_smm_inference`), reporting `max\|id(θ̂)−id(θ₀)\|/\|id₀\|` and `\|Δ\|/SE` on β+T plus cross-start dispersion. **Experiment 2** (β+T coverage, **M=200**, light-polish PSO `n_particles=40,max_iter_init=60,max_iter_stage=30,max_loop=3`): W/Ω built ONCE at θ₀ (fixed across reps), then per rep draws `m_b = m(θ₀)+ε_b`, `ε_b~N(0,Σ_data)` on the β+γ block (`Σ_data` PSD-sqrt'd), re-estimates β+T warm-started at θ₀, per-rep `compute_jacobian`+inference; reports per-param coverage / bias / `SD(θ̂)/mean(SE)`. **Gauge handling (critical)**: inference SEs are RAW, but raw `T` has an unidentified per-sector reference scale (`unpack_params` divides `T[s,:]` by `T[s,ref]`), so coverage is done in the **T_ref=1 gauge** — point estimates taken straight from `identified(·)` (already normalized), raw SEs rescaled per param by `gauge_factors(θ)` (= 1 for β, = `θ[ref-raw-index(sector)]` for T). Index helpers (`BETA_T_START`, `T_PARAM_OFFSET`, `GB_COLS`/`GB_PARAM_IDX`/`GB_INDICES`) mirror `main.jl`'s β+γ/β+T slicing exactly. Per-experiment `output_folder` is rebound (global) so `run_pso_optimization`/`generate_report` write to per-start/per-rep trees under `internal_validity_<industry>/`. Soft-scope `global` declarations added where the top-level `for` loops rebind globals (`min_ess`, `output_folder`). A `SMOKE_TEST` flag shrinks M/PSO for an end-to-end wiring check. **Untested in this environment** (no Julia/data present); authored to mirror `main.jl` call patterns — run SMOKE_TEST first. |
| 2026-06-18 | `model_CP.jl`, `CLAUDE.md` | Fix `solve_network`/`fast_weighted_regression` to use the **passed draw count** (`N_rho_eff = size(u_draws,1)`), not the global `N_rho` const. **Bug**: every per-variety loop/array used the global `const N_rho = 8000` while the weights were self-normalised over the *passed* number of draws. When a caller passes draws with more rows than `N_rho` (e.g. `test_price_alignment`, which sweeps `N_list=[8000,32000,128000]` but cannot change the `const`), the solver consumed only the first `N_rho` rows yet the per-column weights summed to `1` over all `N`, so the realised CES weight sum was `N_rho/N < 1`. The within-sector price index `P_sr = [Σ_ρ w_ρ p_ρ^{1-ν_s}]^{1/(1-ν_s)}` (`1/(1-ν_s) = -2` at `ν_s=1.5`) was therefore inflated by `(N/N_rho)^2`, blowing up `c_tilde` (hence `labor`, `pi_r`) for **all** samplers — the price-alignment test showed `c_tilde` error 5e-3 → 11.5 → 187 across the three `N` (ratios ≈ `(N/8000)^2`), while normalised-share moments (`gamma_ls`, `industry`) stayed flat. **Fix**: `solve_network` derives `N_rho_eff` from `u_draws` (else `sample_weights`, else global `N_rho`), asserts `u_draws`/`sample_weights` row counts agree, and sizes/loops all per-variety arrays (`z_flat`, `linkages_flat`, `p_rho_s`, `winner_good_idx`, `w_rho_s`, the `exp_val` accumulation) by `N_rho_eff`; `fast_weighted_regression` uses `N_rho_eff = size(sample_weights,1)` for `N_valid` and its `rho` loop. **Production-invariant**: every production caller (`load_parameters`, `main_pso`, `build_step3_weight_matrix`, `compute_jacobian`) generates draws at the global `N_rho`, so `N_rho_eff == N_rho` and behaviour is byte-identical; only off-size callers (the N-sweeping test) are corrected. Pre-existing coupling, not introduced by the IS refactor. |
| 2026-06-18 | `model_CP.jl`, `tools.jl`, `pso_integration.jl`, `main.jl`, `main_gmm.jl`, `CLAUDE.md`, `claude.md` | Replace comonotonic stratified-QMC draws with **per-column independent importance sampling**. `generate_stratified_draws` now returns an `(N_rho × n_good)` weight **matrix** `W` (was a length-`N_rho` vector); each good pair `g` gets an INDEPENDENT proposal stream, so `u[rho, g1] ⊥ u[rho, g2]`. The old shared 25-bin grid made every region's productivity a near-deterministic function of one latent per row, collapsing Ricardian selection (the `min_r c_r` winner became fixed by `T/tau/w`, not by independent Fréchet draws) — symptom: price-alignment `strat` columns ~100× worse than `mc`. **New design**: in uniform space, proposal `q(u)=a·u^{a-1}` (Beta(a,1), inverse-CDF `u=v^{1/a}`) oversamples the selection-relevant `u→0` tail; target Uniform ⇒ IS weight `w ∝ 1/q(u)=u^{1-a}/a`, **bounded on [0,1]** (no degeneracy, unlike a Fréchet/z-space tilt). Proposal uniform `v` is stratified (one point per equal-prob stratum) with an **independent `randperm` per column** → decorrelated AND low-variance. `randomise=false` → `MersenneTwister(g)`+midpoint (PSO-deterministic); `randomise=true` → supplied `rng` perm+jitter (Σ_sim). New kwargs `a::Float64=0.5` (tilt; smaller ⇒ heavier tail, lower ESS) and `verbose::Bool=false` (prints `min_g ESS_g`). Weights self-normalised per column (`Σ_rho W[rho,g]=1`). **Consumer threading** (the gate — every `[rho]` weight access becomes `[rho, winning-g]`): `solve_network` CES `P_sr` now weights each `(rho,s)` variety by `sample_weights[rho, winner_good_idx[rho,s]]` (per-`(rho,s)` matrix `w_rho_s`, replacing the shared `reshape(sample_weights,N_rho,1)`); `exp_val` uses `sample_weights[rho, g_winner]`; backward-compat branch builds `fill(1/N_rho, N_rho, n_good)`; `fast_weighted_regression` uses `sample_weights[rho, g]` (param typed `Matrix{Float64}`); `main.jl` firm-level export uses `[rho, g]`. `sample_weights` kwarg type `Vector{Float64}`→`Matrix{Float64}` across `solve_network`/`SMM`/`SMM_with_network`/`full_SMM`/`parallel_SMM`/`parallel_SMM_safe`/`train_stage_one`/`generate_report`/`compute_scores_modular`/`run_reporting`/`train_stage_pso`. `test_price_alignment` now recomputes the `strat_flip` arm's weights from the flipped quantile `(1-u)^{1-a}` (reusing un-flipped `ws` is an invalid SNIS pairing) and reports `min ESS` + `max|cor(U)−I|` per `N`. **Not byte-compatible** — the estimator now targets the correct (decorrelated) object; the old behaviour was biased and there is **no recovery flag** (intentional). `reg_coef` block error is *unaffected* (separate FKG/quadrature bias). **Caution**: the tilt is keyed to the `−log(1−u)` transform (high-z mass at `u→0`); do not flip the draw without recomputing the weight from the flipped quantile. `using Printf` added to `model_CP.jl` for the ESS line. |
| 2026-06-15 | `tools.jl`, `CLAUDE.md` | Extend `screen_T_identification` with **three print-only mechanism-attribution diagnostics** for a small `H[T,T]` λ_min, all computed inside the existing `if J !== nothing` block from the already-built `H`, `T_cols`, `J`, `W`, `param_labels` (no new call-site wiring — call sites already pass these). **(1) Global v_min support**: the smallest-eigenvalue eigenvector `v_min` of `Symmetric(H[T_cols,T_cols])` is mapped to sectors by parsing each T column's `"T[sname-rname]"` label (strip `"T["`/`"]"`, split on the LAST `-`, sector before — same logic as `test_t_reorder.jl`; `sname_to_s` from `_sector_names`), printing per sector the squared-mass share `Σ v_min[sector]²/Σ v_min²` and dominant sign `sign(Σ v_min[sector])`, and flagging CONCENTRATED (one sector > 0.8 mass → **mechanism 3**, level-weak sector) vs SPREAD with mixed signs (→ **mechanism 1**, inter-sectoral flat direction). **(2) M^s cross-check**: after the per-sector analytical screen builds `out`, prints per sector the v_min support/sign beside that sector's M^s ratio `eval_min/eval_max`, so the worst-M^s sector can be matched to where the global weak direction concentrates. **(3) W=I vs W=W_eff**: recomputes `H_I = J'J` and its `Symmetric(H_I[T_cols,T_cols])` λ_min/conditioning beside the W-weighted values — weak under W_eff but well-conditioned under I ⇒ **mechanism 2** (the W metric de-identifies the T-via-γ_ls params); weak under both ⇒ intrinsic to J. **Returned `global_eigen` gains additive fields** `T_evec_support::Vector{Float64}`, `T_evec_signs::Vector{Int}`, `T_evec_sectors::Vector{Int}`, `T_eval_min_identity::Float64` (initialised at function top so they are always defined, incl. when `J===nothing`); existing fields (`eval_min/eval_max/cond/rank/T_eval_min/T_eval_max`) and all four call sites (which discard the return value) are unchanged. Respects the β-then-T gb column order (existing contiguous-suffix assert untouched); eigen on `Symmetric` wrappers; cost trivial. **Print-only** — no estimate, weight matrix, inference output, or file is affected. |
| 2026-06-15 | `tools.jl`, `claude.md`, `CLAUDE.md` | Make `compute_jacobian` robust to low-magnitude `T_{sr}` columns via a scale-invariant **log-space central step** with explicit raw-unit back-conversion. **Problem**: T columns previously used the same additive central step `h = max(|θ|·step_rel, step_abs)` as every other parameter. Below `|θ|≈1e-5` the absolute floor (`step_abs`, e.g. `1e-9` vs `step_rel=1e-4` relative) broke step-to-magnitude proportionality, and a minus step could straddle the Fréchet `max(T_sr, eps)^(1/θ)` clamp — silently turning the central difference into a biased one-sided difference. Both corrupt the T Jacobian column and hence the T parameter SEs. **Fix** (one function, `compute_jacobian`): columns are partitioned by flat parameter position — **T columns** (strictly-positive, multiplicatively-entering; flat index `≥ 1 + S + R_downstream + N_TAU + 1`, layout `[Ω^L(1) | Ω^s(S) | A(R_downstream) | β(N_TAU) | T]`) now perturb at `θ·exp(±δ)` with `δ = step_rel`, and the log-space central difference `(m₊−m₋)/(2δ)` is converted **back to raw units** by dividing by `θ_j`: `J[:,k] = (m₊−m₋)/(2δθ_j)` (chain rule `∂m/∂θ = (∂m/∂lnθ)·(1/θ)`). The log step is scale-invariant (never crosses zero, immune to the floor, never hits the `eps` clamp). **All other columns** (`Ω^L`, `Ω^s`, `A`, `β`/`α`) keep the additive step unchanged — they are not uniformly multiplicative/sign-definite (e.g. `τ = 1+β_b`), where additive is correct. **The stored column is raw `∂m/∂θ` in both regimes**, so `G = ∂m/∂θ` and `Var = (G'WG)^{-1}` are unaffected in units; the log step is purely a numerical-accuracy device. `J_elast` is still derived from raw `J`. **New kwargs** (backward-compatible defaults): `t_log_step::Bool=true` (set `false` → byte-identical old additive behaviour for every column), `check_symmetry::Bool=false` (opt-in print of per-T-column forward-vs-backward asymmetry, flagging entries diverging by >10× across-replication `J_sd`, i.e. nonlinear/clamped regime), `richardson_check::Bool=false` (opt-in recompute of T columns at `2δ`, reporting the relative gap to the `δ` estimate). Both diagnostics are **print-only** and never alter returned `J`. A `@assert θ_j > 0` guards the log set (T_MASK already excludes zeros → should never fire); the non-finite-Jacobian error now names each offending column's flat param index and step regime. The analytical branch (`analytical=true, K=1`, used by `main_gmm.jl`) routes T columns through the same per-column log logic. **No call site changes** — all callers in `main.jl`, `main_gmm.jl`, and the 2×2 test pass only `theta`/`param_indices`/steps/seeds, all unchanged; the moment vector, `MOMENT_MASK`, and `jacobian_param_indices` are untouched. The `unpack_params` normalization `T_mat[s,:] ./= T_mat[s,ref]` is **deliberately left in place** — the chain rule absorbs it automatically since both perturbed evaluations pass through the same `unpack_params`; no perturbation in normalized space. **Expected diff**: T-block entries of `se_theta.npy` / `se_theta_sandwich.npy` will shift slightly (different truncation error — the intended correction, better-conditioned columns); β/α, Ω, A SEs are invariant. |
| 2026-06-15 | `tools.jl`, `main.jl`, `main_gmm.jl`, `CLAUDE.md` | Promote `screen_T_identification` to a shared `tools.jl` diagnostic and wire it after **every** Jacobian computation in both entry points (4 call sites). **Pre-fix** (required — `main.jl` failed to load): the `compute_jacobian` call at θ̂_2 in `run_step4` passed the `step_rel` kwarg twice (`1e-4` then `1e-9`), a duplicate-keyword parse error; the second occurrence is now `step_abs = 1e-9` (no other change to that call). **Move + extend**: the per-sector analytical eigen-screen (`M^s = Σ_dr ω_dr (diag g − g g')` on free non-ref regions; smallest eigenpair + marginal shares + curvature) was lifted out of `main.jl` into `tools.jl` — its `compute_prices_analytical` dependency is satisfied because both entry points include `model_analytical.jl` before any tools usage (function globals resolve at call time, after `load_parameters.jl`). New keyword signature `screen_T_identification(params; J=nothing, W=nothing, param_labels=nothing, label="")`: when `J` (the β+γ rows × β+T cols `J_gb` slice) is supplied it first prints, prefixed by `label`, (a) the global GMM-information eigen-screen `H = Symmetric(J' W J)` (λ_min, λ_max, cond, rank(J) via `svdvals > sv[1]·1e-8`; `W===nothing` ⇒ identity), then (b) the T-only sub-block `H[T_cols,T_cols]` λ_min/λ_max where T columns are those whose `param_labels` start with `"T["` (fallback `N_TAU+1:end`; asserts β/α-first-then-T column order when labels present; guards empty `T_cols`); then (c) the unchanged per-sector screen. Returns `(out, global_eigen)` — the original per-sector NamedTuple vector plus a `global_eigen` summary NamedTuple (`nothing` when `J===nothing`). **Call sites** (all immediately after Jacobian + gb-slicing, inside the relevant `run_step` guard): `main.jl` step2 `J_gb`/`Weight_matrix_custom[gb,gb]`, step4 `J2_gb`/`W_step3`; `main_gmm.jl` step2 `J1_gb`/`W_step1`, step4 `J2_gb`/`W_eff`. The two old bare calls `screen = screen_T_identification(theta_hat_*)` (which ran *outside* any step guard in `main.jl`) are removed. **Pure diagnostic**: print-only (stdout), no new files, no change to any estimate/weight/inference output; eigen on `Symmetric` wrappers, cost trivial (`H` ≤ (N_TAU+n_T)²). |
| 2026-06-11 | `tools.jl`, `main_gmm.jl`, `CLAUDE.md` | Fix GMM Step-2 Σ_data threshold misalignment by sharing the SMM's reconciliation logic. The SMM (`build_step3_weight_matrix`) reconciles the on-disk `Sigma_beta_gamma[_1].npy` against the (possibly gamma-thresholded) active set — dropping β+γ rows/cols for (s,r) pairs pruned by `gamma_threshold` — but `main_gmm.jl` Step 2 took a **naive positional slice** `Sigma_data_full[1:N_gb, 1:N_gb]` with no `T_MASK` filter. When the file is pre-threshold (size `n_gb_old > N_gb`), that slice silently mixes surviving and pruned γ moments, so `W_eff` is built on a covariance misaligned with the threshold-aware Jacobian/moment vectors (`J2_gb`, `emp_vec_gb`); the existing count-only `@assert size(Omega_gmm,1)==N_REG+length(BLOCK_RANGES[5])` couldn't catch it (right count, wrong rows). With `gamma_threshold=0.01` (nonzero default), this is a live risk, not latent. **Fix**: extracted the SMM's reconcile block (the `X_rs.npy` → `T_mask_moment_old` → reference-region removal → `gamma_old_positions`/`survive`/`keep_idx` → three-way size branch `==n_gb` use-as-is / `==n_gb_old` subset / else error) into a shared `tools.jl` helper `reconcile_sigma_data(Sigma_full, input_folder)`; `build_step3_weight_matrix` now calls it (SMM behavior **bit-identical**), and `main_gmm.jl:150` replaces the naive slice with the same call (the `w_beta`/`w_gamma` block-diagonal fallback flows through it too). GMM thereby gains a guaranteed `X_rs.npy` dependency — already read by `load_parameters.jl:30` (included before Step 2), so safe. Step 4's reload of `step2/Sigma_data_gb.npy` inherits the fix. The subset branch (with its documented `c_s` renormalization caveat → γ SEs ~`c_s` too tight) is **preserved** so GMM matches SMM exactly rather than diverging; the stricter "refuse to subset, force regeneration" policy for the exact GMM path remains an opt-in not taken here. |
| 2026-06-11 | `tools.jl`, `main.jl`, `CLAUDE.md` | Add a **test-only** 2×2 noise-decomposition diagnostic, `run_2x2_inference_test` (tools.jl), gated behind a single `run_2x2_test = false` flag in `main.jl` (default **OFF** → behavior byte-identical to before). It crosses two axes at a fixed estimate θ̂: axis 1 (Jacobian) {simulated FD `J_sim_gb`, analytical FD `J_ana_gb`}, axis 2 (W,Ω) {data-only `(Σ_data, Σ_data⁻¹)`, data+sim `(Ω=Σ_data+Σ_sim, W=Ω⁻¹≈W_step3)`}; the (analytical, data-only) corner is the GMM-style variance at the SMM estimate, to attribute SMM parameter-variance noise to the Jacobian vs the weighting channel. The function is **fully standalone**: it replicates ONLY the linear-algebra core of `compute_smm_inference` inline (GtWG with cholesky→eigenvalue-floor fallback, efficient + sandwich variance, `se_sw`, Hansen J=r'Wr, df, p, rank(G)), with its own symmetrize/PD guards, and does **NOT** call `compute_smm_inference`. Each of the four cells is wrapped in try/catch (failed cell → NaNs, continue). **Hard isolation**: it does not modify `compute_smm_inference`, `model_analytical.jl`, `load_parameters.jl`, or `main_gmm.jl`; it does not touch the production inference calls or their `step{2,3}/inference/` outputs. All test outputs go under a separate tree `<step>/inference_2x2_test/`: a distinct report `report_2x2_<label>.txt` (its own format — 2×2 tables per metric {mean se_sw, median se_sw, Hansen p, rank(G)}, a channel-decomposition block, a per-parameter four-cell se_sw table, plus the verbatim validity note) and `se_2x2_<label>.npy` (Dict of the four se_sw vectors for external plotting; NO plotting inside the function). Call sites: at θ̂_1 (run_step2) and θ̂_2 (run_step4), AFTER the existing inference call, reusing the already-built simulated gb Jacobian (`J_gb`/`J2_gb`) and computing only the analytical gb Jacobian via `compute_jacobian(...; analytical=true, K=1, base_seed=3_000_000`/`4_000_000)` (fresh seeds, never colliding with existing Jacobian/Σ_sim seeds). `gamma_ref_map`/`gb_block_ranges`/`gb_block_names` accepted for signature parity but unused (no per-block/γ-ref plots). New `n_quad=200` flag added near arg-parsing. |
| 2026-06-10 | `tools.jl`, `CLAUDE.md` | Add reference-region γ points to `generate_report`'s dashboard `p1` panel, mirroring `compute_smm_inference` §6b (`gamma_ref_map` block). Previously `p1` plotted only `BLOCK_RANGES[5]` (active, non-reference γ), so each sector's MOMENT_MASK-dropped reference region was missing. `p1` is now built **directly** in `generate_report` (option b) instead of via `bubble_scatter`, because the legend + two-series color/marker distinction are p1-specific; labelling `bubble_scatter`'s series to drive a legend would spill it onto the shared p2/p3 panels. `bubble_scatter` and p2/p3 are unchanged. Per sector in `GAMMA_REF_MAP`, the reference point is reconstructed from the adding-up constraint `y_ref = c_s − Σ sim_gamma[local_positions]`, `x_ref = emp_ref` (skip when `emp_ref ≤ 0`, `local_positions` empty, or `max(local_positions) > length(sim_gamma)`). **Zero offset**: `sim_gamma` is the γ block as a standalone 1-based vector, so `local_positions` index it directly — NO β-offset (unlike §6b's β-then-γ subsystem). Styling matches §6b: non-reference = `RGB(0.247,0.404,0.667)` circles ("Non-reference"), reference = `RGB(0.75,0.30,0.20)` `:diamond` ("Reference (reconstructed)"); axis limits span both series; legend at `:bottomright`. The WLS coef/t-stat annotation is computed on **non-reference points only** — reconstructed reference points do not enter the fit. No error bars (generate_report has no fitted-moment SEs, unlike §6b). |
| 2026-06-10 | `load_parameters.jl`, `model_CP.jl`, `tools.jl`, `pso_integration.jl`, `main_pso.jl`, `test_t_reorder.jl`, `CLAUDE.md` | Unify the T-parameter flattening to **s-major** (`vec(permutedims(X_rs))`), identical to the γ-moment and `T_MASK_MOMENT` convention, so the Jacobian's parameter axis aligns column-for-column with its γ-moment row axis (`T[s,r]` column ↔ `γ[s,r]` row). Previously T parameters used region-major `vec(X_rs)` while γ moments used s-major — the two Jacobian axes disagreed. The parameter LAYOUT `[Ω^L \| Ω^s \| A \| β \| T]` is unchanged; only the internal order of the active-T sub-block changes. Coupled edits (all move together; partial application silently transposes `T_mat`): (1) `load_parameters.jl` `T_MASK = vec(permutedims(X_rs)) .> 0` (now == `T_MASK_MOMENT`); (2) `good_indices = findall(permutedims(reshape(T_MASK, R, S)))`; (3) `model_CP.unpack_params` inverts the new flatten: `T_full = zeros(R*S); T_full[T_MASK]=T_reduced; T_mat = permutedims(reshape(T_full, R, S))` (still returns region-major `vec(T_mat)`, so inter-stage full-T is unchanged); (4) jacobian ref-exclusion `flat_pos = (s-1)*R_full + ref_r`; (5) `PARAM_LABELS` T loop `s=((fp-1)÷R_full)+1, r=((fp-1)%R_full)+1`. Phase-0e surfaced **4 additional active flat-T consumers beyond the documented 5** (the "five edits" premise undercounted): `tools.jl:build_step3_weight_matrix` Σ_data subset (`reshape(collect(T_MASK),S,R)` → `collect(T_MASK)`, used by main.jl/main_gmm.jl Step-3 inference); `tools.jl` & `pso_integration.jl` PSO init/bounds (`vec(T_rs_init)[T_MASK]` → `vec(permutedims(T_rs_init))[T_MASK]`); `pso_integration.jl` warm-start reduce (`params_dict[:T][T_MASK]` → `vec(permutedims(reshape(params_dict[:T],S,R)))[T_MASK]`); and `main_pso.jl` (shares `unpack_params`, so its own region-major `T_MASK`/`good_indices`/`T_init` were flipped too, else it silently breaks). The dormant `second_stage` masking path (`mask_emp_gamma_ls`, `second_stage=false` in every main flow; only `run_amplification_analysis.jl` defines the mask) is intentionally left untouched. `GOOD_S/GOOD_R` remain region-major (findall over (S,R)) — confirmed nothing positionally aligns `GOOD[g]` with the reduced-T block (model reads `T_mat[s,r]` via `SR_TO_GOOD`; all param-block construction uses `T_MASK` order). New `test_t_reorder.jl` GATE asserts T1 (`T_MASK==T_MASK_MOMENT`), T2 (unpack round-trip), T3 (label↔index), T4 (γ-row order == T-col order). |
| 2026-06-09 | `tools.jl` | Fix field-mismatch in `generate_report`: `gamma_emp_result`/`gamma_sim_result` were built with `matrix_report(..., false)` (4-field tuple), but `generate_dashboard_report` accesses `.n_zeros` expecting the 5-field form. Restored to `matrix_report(..., true)` for γ only; π_r remains `false` (zero counts uninformative there, γ sparsity is a meaningful diagnostic). Added a one-line comment to `matrix_report` documenting the asymmetric return type. |
| 2026-06-06 | `main_gmm.jl`, `CLAUDE.md` | Restructure `main_gmm.jl` to mirror `main.jl` control flow. Introduces `run_step1..run_step4` booleans and a `step2/W_eff.npy` resume guard (parity with `main.jl`'s `W_step3.npy` guard). Fixes a guaranteed `UndefVarError`: `Omega_gmm = Sigma_data_gb` was assigned *after* the `@assert size(Omega_gmm,…)` check — assignment now precedes both correctness asserts. Adds θ̂_1 inference in Step 2 (into `step2/inference/`, `K_sim=0`, `Ω=Σ_data`) for comparability with the SMM. Moves Jacobian-at-θ̂_2 and `compute_smm_inference` out of `run_step3` into a new `run_step4` block so inference can rerun without re-optimising; full-column `jacobian_all_step3.npy` is retained for diagnostics. Renames `gb_param_cols`/`gb_param_indices_step3`/`beta_T_start_raw`/`S_`/`R_down_` to `gb_cols`/`gb_param_idx`/`beta_T_start`/`S`/`R_downstream` to match `main.jl`. Adds `param_labels`, `moment_labels`, and `gamma_ref_map` kwargs to all `compute_smm_inference` calls. GMM semantics unchanged: `K=1` Jacobian, `K_sim=0`, `Ω=Σ_data`, `analytical=true`, `gmm_note.txt` preserved. |
| 2026-06-04 | `load_parameters.jl`, `model_CP.jl`, `model_analytical.jl`, `tools.jl`, `pso_integration.jl`, `main.jl`, `main_gmm.jl`, `main_pso.jl`, `CLAUDE.md` | Decouple trade-cost parametrization (`N_TAU`) from reg_coef moment count (`N_REG`). Replaces single `N_beta` with two constants: `N_REG = n_coef` (reg_coef moment count, moment axis) and `N_TAU = n_tau` (β-parameter count, parameter axis). Default `n_tau = n_coef` preserves all existing behavior. Target config `N_TAU=1, N_REG=4`: power-law τ=d^α (one α param) with four binned regression moments (over-identified, df=3 on reg_coef). `build_tau`/`unpack_params`/PSO β-slices key off `N_TAU`; `fast_weighted_regression`/`compute_regression_quadrature`/`distance_bin`/`BLOCK_RANGES[4]`/`Sigma_beta_gamma` file selection key off `N_REG`. Two runtime asserts added in main.jl and main_gmm.jl: (1) Ω size == `N_REG + n_γ`; (2) β/α label count in `gb_cols` == `N_TAU`. New arg: `julia main.jl aero 4 1` → N_REG=4, N_TAU=1. `PARAM_LABELS` β entry is "alpha" when N_TAU==1. |
| 2026-06-02 | `main.jl`, `tools.jl`, `load_parameters.jl`, `CLAUDE.md` | Fix β+γ inference singularity and reconcile gamma_threshold cascade. `main.jl`: inference Jacobian now restricted to β+T COLUMNS (not just β+γ rows) at θ̂_1 and θ̂_2 → G'WG full-rank, exactly identified (df=0); Step-2 Jacobian `base_seed` 0→2_000_000 (decorrelate from Σ_sim draws); deleted duplicate end-of-file Step-2/diagnostic block that overwrote K=50 artifacts with K=2. `tools.jl`: `build_step3_weight_matrix` reconciles `Sigma_beta_gamma` size to the (possibly thresholded) active set with a three-way guard (regenerated / pre-threshold-subset / error), no-op when threshold=0; `compute_jacobian` gains an `all(isfinite)` guard naming the offending (moment,param). `load_parameters.jl`: gamma threshold block moved above `T_mask_local = vec(X_rs_local) .> 0` so pruned pairs are excluded from T_MASK/n_good; `empirical_moments` γ block now sourced from `emp_gamma_ls_local` (thresholded+renormalized, consistent with reporting and Hansen-J); one-upstream-region diagnostic (error on 0 survivors, warn on 1). GMM Σ reordering (γ-first) left as a separate task. |
| 2026-05-28 | `model_analytical.jl` | `compute_regression_quadrature` regressand changed from closest-destination win probability to the whole-industry extensive margin ρ̃_{r's}(z) = 1 − ∏_dr (1 − ρ_{r'dr s}(z)) (Eq. B6), aligning the analytical `reg_coef` with the SMM `linkages_flat`; FWL/WLS design unchanged. |
| 2026-05-28 | `main_gmm.jl` | Fix step-3 GMM inference order-condition failure. `J2_gb` was 251×278 (β+γ rows, all-param cols), making `G'WG` rank-251 against 278 columns → singular, df = −27, meaningless SEs. Fix: restrict Jacobian columns to β+T only (`gb_param_cols = findall(i -> i >= beta_T_start_raw, jacobian_param_indices)`), passing `gb_param_indices_step3` as `param_indices` to `compute_smm_inference`. `jacobian_all_step3.npy` still holds the full-column diagnostic Jacobian. |
| 2026-05-28 | `tools.jl`, `main.jl`, `main_gmm.jl`, `model_CP.jl`, `claude.md` | Make step-3 β+γ-only loss and inference consistent and crash-free. `compute_smm_inference` gains `block_ranges`/`block_names` kwargs (zipped in the residual loop). `loss_function` no longer re-subsets a W already sized to `moment_indices`. `run_pso_optimization` loss block fixed `[3,4]`→`[4,5]` (reg_coef+gamma_ls). `main.jl` step-2/step-3 inference `gb_indices` corrected from γ-then-β to β-then-γ (matching `W_step3`/`Sigma_beta_gamma.npy`); step-3 inference now slices `J2`/emp/sim to the gb rows instead of passing full-length vectors against a gb-sized W. Docs corrected to β-then-γ ordering throughout. |

---

## Architecture

Three-step efficient SMM/GMM estimator with spatial comovement moments.

- **`main.jl`** — SMM entry point (simulation-based, stochastic moments)
- **`main_gmm.jl`** — GMM entry point (analytical EK moments, `model_analytical.jl`)
- **`model_CP.jl`** — Core model: `SMM`, `full_SMM`, `loss_function`, `BLOCK_RANGES`, `BLOCK_NAMES`
- **`model_analytical.jl`** — Closed-form EK moments + Gauss-Legendre quadrature for `reg_coef`
- **`tools.jl`** — Inference, Jacobian, weight matrix, PSO wrapper
- **`pso_integration.jl`** — PSO optimiser (`train_stage_pso`)
- **`load_parameters.jl`** — Data loading, globals (`BLOCK_RANGES`, `MOMENT_MASK`, `U_DRAWS`, …)

---

## Moment blocks

Moments are ordered as five blocks in the masked vector (`MOMENT_MASK` applied):

| Index | Name | Description |
|-------|------|-------------|
| 1 | `agg_labor_share` | Aggregate labor share |
| 2 | `agg_industry_share` | Industry shares π_s |
| 3 | `pi_r` | Regional market shares |
| 4 | `reg_coef` | Distance-bin regression coefficients (`N_REG` entries); GMM (quadrature): regressand is the extensive margin ρ̃_{r's}(z) = 1 − ∏_dr (1 − ρ_{r'dr s}(z)) (Eq. B6, whole-industry), regressors/FE keyed on nearest downstream region. `N_REG` is the moment count (= `n_coef` arg); independent of `N_TAU`. |
| 5 | `gamma_ls` | Location-specific linkage shares γ_ls |

`BLOCK_RANGES` is a 5-tuple of index ranges into the masked vector, one per block.

---

## Invariant: β+γ subsystem ordering

**β (reg_coef) first, then γ (gamma_ls):**

```julia
gb_indices = vcat(collect(BLOCK_RANGES[4]), collect(BLOCK_RANGES[5]))
```

This ordering applies everywhere: `Sigma_beta_gamma.npy`, `W_step3`, `Omega`, `gb_indices` in `main.jl` and `main_gmm.jl`. Do not reverse it.

---

## Invariant: T flat-indexing convention (s-major)

**Both Jacobian axes share one flattening: s-major (region-minor).** The γ-moment
rows and the T-parameter columns enumerate `(s,r)` pairs in the *same* order — all
regions of sector 1, then all regions of sector 2, … — so the `T[s,r]` column aligns
with the `γ[s,r]` row.

```julia
T_MASK = vec(permutedims(X_rs)) .> 0     # == T_MASK_MOMENT (s-major over (S,R))
```

- **Flat position** of active pair `(s,r)` in the reduced-T / moment axis:
  `flat_pos = (s-1)*R_full + r`. Inverse: `s = (flat_pos-1)÷R_full + 1`, `r = (flat_pos-1)%R_full + 1`.
- **`unpack_params`** inverts this flatten: `T_full = zeros(R*S); T_full[T_MASK] = T_reduced;
  T_mat = permutedims(reshape(T_full, R, S))`. It still **returns region-major `vec(T_mat)`**,
  so the inter-stage full-T representation is unchanged — only the *reduced* (active) T
  ordering is s-major.
- **Every reduced-T producer must emit s-major**: PSO init/bounds use
  `vec(permutedims(T_rs_init))[T_MASK]`; the warm-start reduce uses
  `vec(permutedims(reshape(params_dict[:T], S, R)))[T_MASK]`; `build_step3_weight_matrix`'s
  Σ_data subset uses `collect(T_MASK)` directly (T_MASK is already the moment convention).
- **`main_pso.jl`** shares `model_CP.unpack_params`, so its own `T_MASK`/`good_indices`/`T_init`
  are flipped to s-major in lockstep. A partial application silently transposes `T_mat`
  (runs fine, fits wrong T) — see `test_t_reorder.jl`.
- **`GOOD_S/GOOD_R` are region-major** (`findall` over the (S,R) mask) and are *not*
  required to match the reduced-T order: the model reads `T_mat[s,r]` via `SR_TO_GOOD`, and
  no code positionally aligns `GOOD[g]` with the reduced-T block.
- The dormant `second_stage` `mask_emp_gamma_ls` path is unaffected by this convention
  (off in all main flows).

---

## Input data layout

| File | Shape | Description |
|------|-------|-------------|
| `Sigma_beta_gamma.npy` | `(N_REG + n_gamma_kept, N_REG + n_gamma_kept)` | Joint bootstrap covariance of β+γ **moments**, **β block first then γ block** (`BLOCK_RANGES[4]` then `BLOCK_RANGES[5]`). β-block dimension is `N_REG` (moment count), independent of `N_TAU`. |
| `Sigma_beta_gamma_1.npy` | same, for `N_REG==1` case | Same ordering |
| `w_gamma.npy` | `(n_gamma_kept, n_gamma_kept)` | Bootstrap covariance of γ_ls moments (fallback) |
| `w_beta.npy` | `(N_REG, N_REG)` | Bootstrap covariance of β moments (fallback) |

---

## Key functions

### `tools.jl` — `build_step3_weight_matrix`

Assembles the efficient SMM weight matrix `W_step3 = (Σ_data + Σ_sim)^{-1}` over
β and γ moments only.

- File selection keyed on `N_REG` (moment count): `Sigma_beta_gamma_1.npy` for `N_REG==1`, `Sigma_beta_gamma.npy` for `N_REG∈{4,5}`. Independent of `N_TAU`.
- Σ_data is the joint bootstrap covariance of reg_coef and γ_ls moments, **ordering: β block first, then γ block** (`BLOCK_RANGES[4]` followed by `BLOCK_RANGES[5]`). β-block dimension = `N_REG`.
- Estimates Σ_sim from K re-seeded `full_SMM` evaluations at `theta_hat_1`, restricted to `gb_indices` (β then γ).
- Returns `W_step3` of size `(N_REG + n_gamma_kept) × (N_REG + n_gamma_kept)`.
- Assert: `size(Sigma_data,1) == N_REG + count(survive)` (not `N_TAU`).

### `tools.jl` — `compute_smm_inference`

Computes parameter SEs (efficient + sandwich), fitted-moment SEs, residual SEs, and Hansen J-test.

**Signature:**
```julia
compute_smm_inference(theta_hat, J, W, Omega;
    param_indices, empirical_moments_vec, simulated_moments_vec,
    output_folder, industry, K_sim,
    block_ranges = BLOCK_RANGES,   # restricted subsystem or global default
    block_names  = BLOCK_NAMES)
```

The `block_ranges`/`block_names` kwargs control per-block diagnostics in the residual SE loop.
When called for the β+γ subsystem, pass:
```julia
n_reg_loc = length(BLOCK_RANGES[4]); n_gam_loc = length(BLOCK_RANGES[5])
gb_block_ranges = (1:n_reg_loc, (n_reg_loc+1):(n_reg_loc+n_gam_loc))
gb_block_names  = ("reg_coef", "gamma_ls")
```
Defaults (`BLOCK_RANGES`/`BLOCK_NAMES`) leave full-vector callers unaffected.

### `tools.jl` — `run_pso_optimization`

Unified PSO wrapper for Steps 1 and 3.

- `moments_loss_gamma_beta=true` → `moment_blocks=[4,5]` (reg_coef + gamma_ls). **Note:** blocks 4 and 5 are `reg_coef` and `gamma_ls` respectively; blocks 3 and 4 would be `pi_r` + `reg_coef` (incorrect for β+γ).
- `gamma_beta_only=true` → optimises only `beta` and `T` parameters (A_r/labor/industry fixed at warm-start).

### `model_CP.jl` — `loss_function`

Computes `err * W * err'` with optional moment subsetting via `moment_indices`.

W-size collision guard: if `W` is already restricted to `length(moment_indices)` rows/cols
(e.g. a pre-built `W_step3` passed alongside `moment_blocks`), it is used as-is.
A full-size `W` is subsetted as before. This prevents `BoundsError` when step-3 loss runs.

---

## Step-3 inference scope

Inference at θ̂_2 (`main.jl` and `main_gmm.jl`) is computed on the **β+γ moments only**,
ordered β then γ:

```julia
gb_indices = vcat(collect(BLOCK_RANGES[4]), collect(BLOCK_RANGES[5]))
J2_gb      = J2[gb_indices, :]
sim_vec_gb = sim_vec_2[gb_indices]
emp_vec_gb = emp_vec[gb_indices]
```

The full-row Jacobian (`J2`, all moment rows) is still computed and saved to
`jacobian_all_step3.npy` for diagnostics; only the inference slice is gb-restricted.

The step-3 PSO loss is likewise restricted to β+γ via `moment_blocks=[4,5]`.

`compute_smm_inference` takes `block_ranges`/`block_names` kwargs so per-block diagnostics
index the restricted vector correctly. `loss_function` leaves a pre-restricted W
(size == `length(moment_indices)`, e.g. `W_step3`) un-subset.

### T-identification eigen-screen

`screen_T_identification` (in `tools.jl`) runs **after every Jacobian computation**, at
both θ̂_1 (Step 2) and θ̂_2 (Step 4), in `main.jl` and `main_gmm.jl`. It is fed the same
gb slice used for inference (`J_gb`/`J1_gb`/`J2_gb` and the matching gb weight matrix
— `Weight_matrix_custom[gb,gb]`, `W_step3`, `W_step1`, or `W_eff`) plus
`param_labels=PARAM_LABELS[gb_cols]`. It prints (label-prefixed): the global
`H = J'WJ` eigen-screen (λ_min/λ_max/cond/rank), the T-only sub-block `H[T_cols,T_cols]`
eigenvalues, and the per-sector analytical screen. It is **print-only** — no estimate,
weight matrix, inference output, or file is affected. The historical `main.jl`
duplicate-`step_rel`-kwarg bug in the θ̂_2 Jacobian call was fixed at the same time.

**Mechanism attribution** (when the `H[T,T]` sub-block is non-empty): three further
diagnostics attribute a small `H[T,T]` λ_min to one of three mechanisms. (1) The λ_min
eigenvector `v_min` of `H[T,T]` is mapped to sectors by parsing each T column's
`"T[sname-rname]"` label (split on the LAST '-'); per sector the share of squared mass
`Σ v_min[sector]²/Σ v_min²` and dominant sign are printed, flagging CONCENTRATED (one
sector > 0.8 mass → **mechanism 3**, level-weak sector) vs SPREAD with mixed signs
(→ **mechanism 1**, inter-sectoral flat direction). (3) `H_I = J'J` (identity weighting)
and its `H_I[T,T]` λ_min/conditioning are printed beside the W-weighted values: weak
under W_eff but fine under I ⇒ **mechanism 2** (W de-identifies the T-via-γ_ls params);
weak under both ⇒ intrinsic to J. (2) After the per-sector analytical screen, a
cross-check prints each sector's v_min support/sign beside its M^s ratio λ_min/λ_max,
matching the worst-M^s sector to the support of the global weak direction. All three are
**print-only**. `global_eigen` gains additive fields `T_evec_support`, `T_evec_signs`,
`T_evec_sectors`, `T_eval_min_identity` (existing fields and call sites unchanged — call
sites discard the return value).

---

## Two-constant parameter layout

`N_REG` and `N_TAU` are independent constants set in `load_parameters.jl`:

| Constant | Meaning | Axis | Set by | Used in |
|----------|---------|------|--------|---------|
| `N_REG` | reg_coef moment count | moment | `n_coef` arg | `fast_weighted_regression`, `compute_regression_quadrature`, `distance_bin`, `BLOCK_RANGES[4]`, Σ_data file selection |
| `N_TAU` | trade-cost parameter count | parameter | `n_tau` arg (default = `n_coef`) | `build_tau`, `unpack_params`, PSO β-slice, `PARAM_LABELS` β section, `get_param_start_index(:T)` |

Standard runs: `n_tau` unset → `N_TAU = N_REG = n_coef` (no behavior change).
Over-identified run: `n_tau=1, n_coef=4` → `N_TAU=1` (power-law α), `N_REG=4` (four binned moments), df=3 on reg_coef block.

---

## Usage

```bash
# SMM (simulation-based)
julia main.jl auto 1        # N_REG=1, N_TAU=1 (default: n_tau=n_coef)
julia main.jl aero 4        # N_REG=4, N_TAU=4 (bin parametrization, exactly identified on reg_coef)
julia main.jl aero 4 1      # N_REG=4, N_TAU=1 (power-law τ=d^α, over-identified df=3)

# Analytical GMM
julia main_gmm.jl aero 4          # N_REG=4, N_TAU=4, n_quad=200
julia main_gmm.jl aero 4 1        # N_REG=4, N_TAU=1 (over-identified)
julia main_gmm.jl aero 4 1 200 500  # N_REG=4, N_TAU=1, n_quad=500
```
