# SMM Spatial Comovement — Codebase Guide

## Changelog

| Date | Files | Summary |
|------|-------|---------|
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
