# SMM Spatial Comovement — Codebase Guide

## Changelog

| Date | Files | Summary |
|------|-------|---------|
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
| 4 | `reg_coef` | Distance-bin regression coefficients (β, N_beta entries) |
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
| `Sigma_beta_gamma.npy` | `(N_beta + n_gamma_kept, N_beta + n_gamma_kept)` | Joint bootstrap covariance of β+γ moments, **β block first then γ block** (`BLOCK_RANGES[4]` then `BLOCK_RANGES[5]`) |
| `Sigma_beta_gamma_1.npy` | same, for `N_beta==1` case | Same ordering |
| `w_gamma.npy` | `(n_gamma_kept, n_gamma_kept)` | Bootstrap covariance of γ_ls moments (fallback) |
| `w_beta.npy` | `(N_beta, N_beta)` | Bootstrap covariance of β moments (fallback) |

---

## Key functions

### `tools.jl` — `build_step3_weight_matrix`

Assembles the efficient SMM weight matrix `W_step3 = (Σ_data + Σ_sim)^{-1}` over
β and γ moments only.

- Loads `Sigma_beta_gamma.npy` (or `Sigma_beta_gamma_1.npy` for `N_beta==1`). Σ_data is the joint bootstrap covariance of reg_coef and γ_ls moments, **ordering: β block first, then γ block** (`BLOCK_RANGES[4]` followed by `BLOCK_RANGES[5]`).
- Estimates Σ_sim from K re-seeded `full_SMM` evaluations at `theta_hat_1`, restricted to `gb_indices` (β then γ).
- Returns `W_step3` of size `(N_beta + n_gamma_kept) × (N_beta + n_gamma_kept)`.

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

## Usage

```bash
# SMM (simulation-based)
julia main.jl auto 1        # industry=auto, n_coef=1

# Analytical GMM
julia main_gmm.jl aero 4    # industry=aero, n_coef=4, n_quad=200 (default)
julia main_gmm.jl aero 4 resume 200 500   # n_quad=500
```
