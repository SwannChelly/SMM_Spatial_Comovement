# claude.md — SMM Spatial Comovement: Developer & Economist Reference

## Review standard

Every change must pass two filters before being written:
- **Senior economist**: Does it respect the structural model's identification logic, moment targeting, and equilibrium conditions? Does notation stay consistent with the paper?
- **Senior developer**: Is the implementation efficient (time/memory), correct, minimal, and maintainable? No unnecessary abstraction. No breaking of existing interfaces.


---

## Planning and file access protocol

When constructing an execution plan (debugging, extension, or refactor), adhere to the following discipline:

- **Folder-first reasoning**: Always begin by reading the *descriptions of folders and files* (as documented here or inferred from structure) before opening any file.
- **Selective file access**: Only read files if they are expected to be directly relevant for the current task. Avoid exhaustive or systematic file reading.
- **Priority ordering**:
  1. Documentation (this file, comments, high-level descriptions)
  2. Entry points (`main_*.jl`, pipelines)
  3. Core modules (`model_CP.jl`)
  4. Auxiliary tools (only if needed)
- **Avoid unnecessary I/O**: Do not open large or peripheral files unless a clear hypothesis requires it.
- **`.ipynb` restriction**: Never read Jupyter notebook (`.ipynb`) files unless explicitly instructed. Treat them as non-source artifacts unless proven otherwise.
- **Hypothesis-driven inspection**: Each file access must be justified by a specific question (e.g., parameter flow, moment construction, solver behavior).

This protocol is binding: violations are considered both a **modeling failure** (loss of structural clarity) and a **software failure** (inefficient and non-scalable workflow).


---

## Model overview

The code calibrates a **structural spatial production-network model** via Simulated Method of Moments (SMM). The setting is the French economy with one downstream industry (aerospace `aero` or automotive `auto`) sourcing intermediate inputs from upstream suppliers across sectors `s` and regions `r`.

### Economic mechanism (paper: *Spatial Comovements*)

1. **Upstream firms** draw productivity from a Fréchet distribution: `z_{ρsr} ~ Fréchet(θ, T_{sr}^{1/θ})`. `T_{sr}` is the comparative-advantage scale; `θ = 1.768` controls dispersion.
2. **Ricardian selection**: each downstream firm in region `r` buys each variety `ρ` in sector `s` from the cheapest upstream supplier, after iceberg costs `τ_{r'r} = 1 + β_b` (distance-bin `b`).
3. **Nested CES costs**: `P_{sr}` (within-sector, elasticity `ν_s = 2.5`) → `P_r` (across-sector, elasticity `ν = 0.2`) → `c_r` (labor-intermediate, elasticity `λ = 0.5`, labor share `Ω^L`).
4. **Downstream equilibrium**: monopolistic competition with demand elasticity `ε`, giving `Y_r = p_r^ε · P^{-ε} · E · δ_r`.
5. **Amplification**: demand shocks propagate upstream through input-output linkages; the untargeted moment (Table 2) validates this propagation against reduced-form elasticities.

### Fixed parameters (not calibrated)
| Symbol | Value | Meaning |
|--------|-------|---------|
| `θ` | 1.768 | Fréchet shape |
| `λ` | 0.5 | Labor-intermediate substitution |
| `ν` | 0.2 | Cross-sector substitution |
| `ν_s` | 2.5 | Within-sector substitution |
| `ε` | from data | Demand elasticity (industry-specific) |

### Calibrated parameters (parameter vector layout)
```
params = [β_1..β_{N_beta} | Ω^L | Ω^s_1..Ω^s_S | A_1..A_{R_downstream} | T_{nonzero entries}]
```
- `N_beta ∈ {4, 5}`: distance-bin trade cost coefficients (monotone: β_1 ≤ … ≤ β_K)
- `Ω^L`: aggregate labor share in production
- `Ω^s`: sectoral input shares (sum-normalized internally)
- `A_r`: downstream productivity by region
- `T_{sr}`: Fréchet scale for active (sector, region) pairs only (sparse, indexed by `T_MASK`)

### Targeted moments
1. Aggregate labor share
2. Sectoral input shares `Ω^s` (S moments)
3. Sourcing shares `γ_{ls}` = share of sector `s` inputs sourced from region `l` (S×R, sparse)
4. Regression coefficients: P(supplier) on distance-bin dummies + log-productivity (N_beta moments)
5. Regional employment/sales shares `π_r` (R_downstream moments)

Sum-to-1 redundancies are dropped via `MOMENT_MASK` (one per share-constrained block). For the γ_ls block, the dropped moment per sector `s` is the entry at `T_REF_REGION[s]` (the region with the largest empirical sourcing share), which is the same region used to normalize `T_mat[s,:]` in `unpack_params`.

---

## File structure

```
SMM_Spatial_Comovement/
├── model_CP.jl                  # Core: solve_network, compute_moments, full_SMM
├── tools.jl                     # Optimization helpers, reporting, LHS/Halton grids
├── pso_integration.jl           # PSO: parallel_pso_smm, train_stage_pso
├── main.jl                      # Entry point: three-step efficient SMM (Steps 1–3)
├── main_pso.jl                  # Legacy entry point: identity-weighted PSO only (unchanged)
├── main_NM.jl                   # Entry point: Nelder-Mead optimizer (2-step)
├── untargeted_moments.jl        # Validation: 3-model shock simulation + regressions
├── run_untargeted_validation.jl # Interface: validate_table2_all_models
├── compute_amplification_weights.jl  # Section 4.3: w_d^{sr'}, w_{r'}^{sr'd}
├── run_amplification_analysis.jl     # Pipeline: amplification + Python call
├── pkg.jl                       # Package installation
└── run.sh                       # Shell launcher
```

### `model_CP.jl` — core model (read this first)

Key functions:
- `generate_stratified_draws(N_rho, n_good; seed_offset=0)`: CdGM-style stratified uniform draws for Fréchet inverse CDF. Returns `(U, weights)` where `U` is `(N_rho × n_good)`. `seed_offset` shifts the Van der Corput index so that `K` independent draw configurations can be generated without touching the global RNG — required by `build_step3_weight_matrix`.
- `unpack_params(params)`: splits flat vector into `(β, Ω^L, Ω^s, A, T_full)`. `T_full` is `S×R` with zeros for inactive pairs.
- `build_tau(beta)`: builds `(R, R_downstream)` iceberg cost matrix from distance-bin coefficients using precomputed `DistBin`.
- `solve_network(params; return_firm_level, precomputed_tau, u_draws, sample_weights)`: main equilibrium solver. Returns NamedTuple. If `return_firm_level=true`, appends sparse COO firm-level arrays (`firm_exp_rho/s/g/r/val`, `firm_deriv_val`).
- `compute_moments(network, params)`: extracts all 5 moment blocks from solved network.
- `fast_weighted_regression(linkages_flat, z_flat, sample_weights)`: analytical weighted OLS with FWL demeaning by sector×downstream-region FE. Returns `N_beta` distance-bin coefficients.
- `full_SMM(params, simulation=false, second_stage=false, method="original"; ..., W_override=nothing)`: wrapper returning `(loss, moments_tuple)`. When `W_override` is not nothing it is used instead of the global `Weight_matrix_custom`; existing call sites are unaffected.

**Flat indexing convention** (critical): active `(s, r)` pairs are stored in a flat array of length `n_good`. Global constants `GOOD_S[g]`, `GOOD_R[g]` map index `g` back to `(s, r)`. `SR_TO_GOOD[s, r]` is the reverse map (0 for inactive). `SECTOR_GOOD_INDICES[s]` lists all `g` indices for sector `s`.

**Sparse COO convention** (firm-level): `firm_exp_g[i]` is the good-pair index; `GOOD_R[firm_exp_g[i]]` gives the upstream region. Not `firm_exp_r[i]` — that is the *downstream* region.

### `tools.jl` — helpers

Key utilities:
- `generate_lhs_beta(n, n_beta, lb, ub)`: LHS samples with sorted draws → monotone β.
- `generate_stratified_draws` is in `model_CP.jl`, not here.
- `parallel_SMM_safe(...; W_override=nothing)`: try-catch wrapper returning `nothing` on failure. Threads `W_override` to `full_SMM`.
- `distance_bin(d, n_bins)`: maps distance (km) to bin index 1..N_beta (0 = local, excluded from regression).
- `generate_report(...)`: saves `dashboard.png`, `report.txt`, `pi_r.npy`, `productivity.npy`.
- `run_reporting(output_folder, max_loop)`: scans all epoch/stage folders, computes scores, saves `best_simulated_moments.npy`, `best_parameters_list.npy`.
- `find_last_stage_folder(base_folder)`: locates most recent `best_params.npy`.
- `find_resume_state(base_folder)`: determines `(resume_loop, resume_substage)` for `--resume` mode.
- `build_step3_weight_matrix(theta_hat_1, input_folder; K, output_folder)`: assembles `W_step3 = (Σ_data + Σ_sim)^{-1}` restricted to γ_ls and reg_coef moments. `Σ_data` loaded from `Sigma.npy` (joint bootstrap covariance of γ+β, size `n_gamma_kept + n_beta_kept`). `Σ_sim` estimated via K pmap evaluations with MC draws seeded by k, restricted to the same indices. The returned `W_step3` is `N_moments × N_moments` with zeros outside the γ+β sub-block. Saves `Sigma_data.npy`, `Sigma_sim.npy`, `Omega.npy`, `Omega_gb.npy`, `W_step3.npy`, `W_gb.npy`, `diagnostics.txt` to `output_folder/step2/`.
- `compute_smm_inference(theta_hat, J, W, Omega; param_indices, empirical_moments_vec, simulated_moments_vec, output_folder, industry, K_sim)`: computes delta-method standard errors (efficient and sandwich), fitted-moment SEs, moment-residual SEs, and the Hansen J over-identification test. Saves all arrays plus `inference_summary.txt` and `J_stat.txt` to `output_folder/inference/`. Returns a Dict.
- `run_pso_optimization(; weight_matrix, skip_initial_beta_search, warm_start_params, output_subfolder, max_loop, ...)`: unified PSO wrapper for Steps 1 and 3. Runs Stage 0 LHS search (unless `skip_initial_beta_search=true`), Stage 1 full PSO, and 50-loop refinement. Returns `(best_params, best_fitness)`.

### `pso_integration.jl` — PSO optimizer

- `parallel_pso_smm(obj, lb, ub; n_particles, max_iter, warm_start_particle, ...)`: core PSO. Always includes `warm_start_particle` as one particle (guarantees monotone improvement). Restarts stagnant particles every 25 iterations when `fitness > 1.5 × g_best`.
- `train_stage_pso(n_particles, max_iter; variable_list, last_stage_folder, alpha, weight_matrix=nothing, warm_start_override=nothing, ...)`: builds bounds around previous best (multiplicative `alpha`), constructs warm start, defines stage-specific objective, calls `parallel_pso_smm`. `weight_matrix` is forwarded to `parallel_SMM_safe` as `W_override`. `warm_start_override` allows passing a full-parameter warm start when `last_stage_folder=nothing` (used in Step 3 Stage 1).
- `get_param_start_index(param_name)`: maps `:beta/:agg_labor_share_tech/:agg_industry_share_tech/:productivity/:T` to starting index in flat param vector.

### `main.jl` — three-step efficient SMM (primary entry point)

`julia main.jl [industry] [n_coef] [resume] [K_sim]`

Three-step procedure:
1. **Step 1** (`output_subfolder="step1"`): identity-weighted (uses global `Weight_matrix_custom`) PSO → `θ̂_1`. Includes Stage 0 LHS beta search.
2. **Step 2**: `build_step3_weight_matrix(θ̂_1; K=K_sim)` → `W_step3 = (Σ_data + Σ_sim)^{-1}`. Outputs saved to `step2/`.
3. **Step 3** (`output_subfolder="step3"`): efficient-weighted PSO with `W_step3`, warm-started at `θ̂_1`, skips Stage 0. → `θ̂_2`.
4. **Inference at θ̂_1**: Jacobian `J1` over identified parameters via `compute_jacobian(...; param_indices=jacobian_param_indices, output_subdir="step2")`. Then `compute_smm_inference` with `W=W_step3` and `Ω=Omega` produces delta-method SEs, Hansen J-test, and diagnostics. Outputs in `step2/inference/`.
4. **Inference at θ̂_2**: Jacobian `J2` over identified parameters via `compute_jacobian(...; param_indices=jacobian_param_indices, output_subdir="step3", base_seed=1_000_000)`. Then `compute_smm_inference` with same `W` and `Ω`. Outputs in `step3/inference/`.

Resume logic: if `step2/W_step3.npy` exists, skips Steps 1+2; if only `step1/0/` exists, skips Step 1.

### `main_pso.jl` — legacy entry point (unchanged)

Identity-weighted PSO only. Stages:
1. **Stage 0** (LHS beta search): 1500 LHS samples of `β`. Selects best by regression-coefficient distance.
2. **Stage 1** (full PSO, `MAX_ITER_INITIAL=200`): all parameters jointly.
3. **Refinement loops** (`max_loop=50`, 3 sub-stages each): productivity → spatial → technical.
   Alpha schedule: `0.3 → 0.9` over loops.

**Weight matrix**: `Weight_matrix_custom = Diagonal(w_vec)` where `w_vec` has 100× weight on the `reg_coef` block. No longer loaded from `weight_vector.npy`.

### `main_NM.jl` — Nelder-Mead alternative

Two-step bounded NM via `Fminbox(NelderMead())` from `Optim.jl`:
- Step 1: 20 parallel LHS starts, coarse convergence (`f_reltol=1e-6`, 5000 iter).
- Step 2: top-15 solutions refined (`f_reltol=1e-10`, 10000 iter).

### `untargeted_moments.jl` + `run_untargeted_validation.jl` — validation

Three shock models (each uses **its own** parameters estimated on the appropriate data transformation):
- `UNIVARIATE`: pooled AR(1), i.i.d. across regions, params from original data.
- `MULTIVARIATE`: region-specific `ρ_r`, correlated `Σ`, params from original data.
- `MULTIVARIATE_FE`: regional AR(1) on demeaned data + separate time FE `μ_t`.

Parameter files (must be pre-generated by `extract_shock_parameters.py`):
```
baseline_{industry}/
├── rho_univariate.npy
├── sigma_unconditional_univariate.npy
├── rho_r_multivariate.npy / Sigma_innovations_multivariate.npy
├── rho_r_fe.npy / Sigma_innovations_fe.npy / time_fe.npy
└── share_dist.csv   # empirical a_{di}^D distribution
```

---

## Global constants (set once in main, broadcast to all workers)

| Constant | Type | Description |
|----------|------|-------------|
| `S`, `R` | Int | Number of sectors, total regions |
| `R_downstream` | Int | Regions with downstream activity |
| `N_rho` | Int | Varieties per sector (= 1000 in production runs) |
| `n_good` | Int | Number of active (sector, region) pairs |
| `GOOD_S`, `GOOD_R` | Vector{Int} | Sector/region for each good-pair index |
| `SR_TO_GOOD` | Matrix{Int} | (S, R) → good-pair index (0 if inactive) |
| `SECTOR_GOOD_INDICES` | Vector{Vector{Int}} | Good-pair indices per sector |
| `T_MASK` | BitVector | Which entries of `vec(T_rs)` are active |
| `DistBin` | Matrix{Int} | Precomputed distance bins (R × R_downstream) |
| `CLOSEST_PLANT_DIST` | Vector{Float64} | Distance to nearest downstream plant (per upstream region) |
| `CLOSEST_DOWNSTREAM_REGION` | Vector{Int} | Nearest downstream region (per upstream region) |
| `U_DRAWS`, `SAMPLE_WEIGHTS` | Matrix/Vector{Float64} | CdGM stratified draws (N_rho×n_good) + weights |
| `emp_gamma_ls` | Matrix{Float64} | Empirical sourcing shares (R × S, transposed on load) |
| `emp_pi_r` | Vector{Float64} | Empirical downstream sales shares |
| `reg_coef` | Vector{Float64} | Empirical regression coefficients (length N_beta) |
| `empirical_moments` | Matrix{Float64} | Masked empirical moments (1 × N_moments) |
| `MOMENT_MASK` | BitVector | Which moments to keep (drops sum-to-1 redundancies) |
| `Weight_matrix_custom` | Diagonal | Default SMM weight matrix: identity with 100× on reg_coef block |
| `N_beta` | Int | Number of distance bins (4 or 5) |
| `BLOCK_RANGES` | NTuple{5} | Index ranges into masked moment vector for each block |

---

## Input data layout (`baseline_{industry}/`)

| File | Shape | Description |
|------|-------|-------------|
| `stats.csv` | — | `ε` (row 1), labor share (row 2) |
| `distances.npy` | (R, R) | Pairwise distances (km) |
| `w_rs.npy` | (R,) | Regional wage shares (not sector-specific) |
| `regional_wages.npy` | (R,) | Absolute wage level per region |
| `N_downstream_per_region.npy` | (R,) | Downstream firm count per region |
| `filter_N_upstream.npy` | (S, R) | Upstream presence filter |
| `input_share.npy` | (S,) | Sectoral input shares |
| `domestic_share.npy` | (S,) | Domestic sourcing fraction per sector |
| `X_rs.npy` | (S, R) | Initial T_rs values (also defines sparsity mask) |
| `N_rs.npy` | (S, R) | Upstream firm counts |
| `emp_gamma_ls.npy` | (R, S) | Empirical sourcing shares (transposed on load) |
| `X_dr.csv` | col `X_dr` | Downstream sales by region (for π_r) |
| `reg_coef_{4\|5}.npy` | (N_beta,) | Empirical regression coefficients |
| `Sigma.npy` | (n_gamma_kept+N_beta, n_gamma_kept+N_beta) | Joint bootstrap covariance of γ_ls and reg_coef moments (γ block first, then β block, same ordering as BLOCK_RANGES[3] then BLOCK_RANGES[4]) |

**Note**: `w_gamma.npy` and `w_beta.npy` are only required when running `main.jl` (three-step SMM). `main_pso.jl` does not use them. The row/column ordering of `w_gamma.npy` must match the γ_ls entries in the masked moment vector: sector-major (`vec(permutedims(emp_gamma_ls))`), active pairs only, with one entry dropped per sector for sum-to-1 (same as `BLOCK_RANGES[3]`).

---

## Output structure (`reporting_{industry}/`)

`main.jl` (three-step) nests outputs by step:

```
step1/                          # Step 1 outputs (identity-weighted)
  0/                            # Initial PSO stage
  epoch_{k}/
    {1,2,3}/                    # Three sub-stages per loop
      best_params.npy           # Parameter vector (n_params × 1)
      dashboard.png             # γ_ls, π_r, π_s scatter plots
      report.txt                # Moment comparison table
      pi_r.npy
      productivity.npy
  theta_hat_1.npy               # θ̂_1 (best_params from last stage of step1)
  best_simulated_moments.npy
  best_parameters_list.npy
step2/                          # Step 2 outputs (weight matrix construction)
  Sigma_data.npy                # Block-diagonal empirical covariance (N_moments × N_moments)
  Sigma_sim.npy                 # Full simulated covariance from K evaluations
  Omega.npy                     # Sigma_data + Sigma_sim
  W_step3.npy                   # inv(Omega) — efficient weight matrix
  diagnostics.txt               # Condition number log
  jacobian_all.npy              # Jacobian at θ̂_1 (all parameters)
  jacobian_all_elasticity.npy
  jacobian_all_sd.npy
  jacobian_all_elasticity_sd.npy
  jacobian_all_param_indices.npy
  inference/
    var_theta_efficient.npy     # (G₁'WG₁)^{-1} at θ̂_1
    var_theta_sandwich.npy      # sandwich variance at θ̂_1
    se_theta.npy                # √diag(Var_eff)
    se_theta_sandwich.npy       # √diag(Var_sandwich)
    t_stats.npy                 # θ̂_1 / se_theta
    ci_95.npy                   # (p × 2) 95% confidence intervals
    se_moments_fitted.npy       # √diag(J · Var_eff · J')
    se_moment_residuals.npy     # √max(diag(Ω - J · Var_eff · J'), 0)
    J_stat.txt                  # Hansen J statistic, df, p-value
    inference_summary.txt       # Human-readable diagnostics
step3/                          # Step 3 outputs (efficient-weighted)
  0/ epoch_{k}/ {1,2,3}/...    # Same layout as step1/
  theta_hat_2.npy               # θ̂_2 (final efficient estimate)
  jacobian_all_step3.npy        # Jacobian at θ̂_2 (all parameters)
  jacobian_all_step3_elasticity.npy
  jacobian_all_step3_sd.npy
  jacobian_all_step3_elasticity_sd.npy
  jacobian_all_step3_param_indices.npy
  inference/
    var_theta_efficient.npy     # (G₂'WG₂)^{-1} at θ̂_2
    var_theta_sandwich.npy      # sandwich variance at θ̂_2
    se_theta.npy                # √diag(Var_eff)
    se_theta_sandwich.npy       # √diag(Var_sandwich)
    t_stats.npy                 # θ̂_2 / se_theta
    ci_95.npy                   # (p × 2) 95% confidence intervals
    se_moments_fitted.npy       # √diag(J · Var_eff · J')
    se_moment_residuals.npy     # √max(diag(Ω - J · Var_eff · J'), 0)
    J_stat.txt                  # Hansen J statistic, df, p-value
    inference_summary.txt       # Human-readable diagnostics
simulated_panel_unified.parquet
suppliers.parquet
w_srd_r.npy                     # w_{s,r',r}: upstream (s,r') sales share to downstream r
```

`main_pso.jl` (legacy) writes directly to `reporting_{industry}/` without nesting:

```
0/                     # Initial PSO stage
epoch_{k}/
  {1,2,3}/             # Three sub-stages per loop
best_simulated_moments.npy
best_parameters_list.npy
empirical_moments.npy
simulated_panel_unified.parquet
suppliers.parquet
w_srd_r.npy
```

---

## Inference

After Step 3, `compute_smm_inference` computes delta-method standard errors for the active parameters (β and T):

**Efficient variance**: `Var_eff = (G'WG)^{-1}` where `G = ∂m(θ̂_2)/∂θ` (Jacobian at θ̂_2), `W = W_step3`.

**Sandwich variance**: `Var_sw = (G'WG)^{-1} · G'WΩW G · (G'WG)^{-1}` where `Ω = Σ_data + Σ_sim`. When `W = Ω^{-1}` (efficient weighting) the two coincide; deviations flag misspecification or a non-efficient weight matrix.

**Fitted-moment SEs**: `√diag(G · Var_eff · G')` — uncertainty in the model-predicted moments at θ̂_2.

**Residual SEs**: `√max(diag(Ω - G · Var_eff · G'), 0)` — per-moment unexplained variance after controlling for parameter uncertainty.

**Hansen J-test**: `J = r'Wr` where `r = m̂ - m̃(θ̂_2)`, distributed χ²(N_moments − p) under correct specification. A large J (small p-value) indicates moment over-identification failure.

**Identified parameters only**: The Jacobian is computed over identified parameters only, excluding the S+2 directions made flat by internal normalizations in `unpack_params` (first Ω^s, A_1, and T[s, T_REF_REGION[s]] for each s). `jacobian_param_indices` (built once in `main.jl` after constants are distributed) carries these indices and is passed to both `compute_jacobian` and `compute_smm_inference`.

**Caveats**: SEs are delta-method conditional on the draws used for Σ_sim estimation. A Murphy–Topel correction would account for sequential sampling noise across estimation steps. `Σ_data` is non-zero only on the γ_ls and reg_coef blocks, so residual SEs on labor/industry/π_r reflect simulator variance only.

---

## Keeping claude.md current

If a change modifies anything documented in this file — file structure, function signatures, global constants, parameter vector layout, moment construction, input/output conventions, or the optimization pipeline — **claude.md must be updated in the same step as the code change**. Do not defer the update. The changelog entry alone is not sufficient: the relevant section of the reference documentation must also reflect the new state.

---

## Changelog

*Format: date · file(s) changed · description (≤4 sentences)*

<!-- Add entries below in reverse chronological order -->

2026-05-26 · `model_analytical.jl` (new), `main_gmm.jl` (new), `model_CP.jl`, `tools.jl`, `pso_integration.jl`, `load_parameters.jl`, `pkg.jl`, `run.sh`, `claude.md` · Implement closed-form GMM estimation. `model_analytical.jl` provides `compute_moments_analytical`: blocks {Ω^L, Ω^s, π_r, γ_ls} use exact EK closed-form formulas (Φ_{s,dr} → P_sr → P_r → c_r → Y_r); block reg_coef uses 200-node Gauss-Legendre quadrature on the Fréchet CDF with analytical supplier probabilities. `GAMMA_FACTOR[s] = Γ((θ+1-ν_s)/θ)^{1/(1-ν_s)}` precomputed in `load_parameters.jl`. `full_SMM` gains `analytical=false` kwarg (backward-compatible); `parallel_SMM_safe`, `train_stage_pso`, `run_pso_optimization`, and `compute_jacobian` all propagate `analytical` and `n_quad`. `main_gmm.jl` is the new entry point: W_eff = Σ_data^{-1} (Σ_sim=0 by construction), Jacobian uses K=1 (deterministic), SEs are exact delta-method without Murphy-Topel correction. `run.sh` gains `--mode=gmm` and `--n_quad` flags. `build_step3_weight_matrix` marked deprecated for GMM mode. New packages: `SpecialFunctions.jl`, `FastGaussQuadrature.jl`.

---

## GMM analytical mode

`main_gmm.jl` provides closed-form moment evaluation, replacing the SMM simulation
of N_ρ firms by Eaton-Kortum analytical formulas:

| Block | Method | Key formula |
|-------|--------|-------------|
| Ω^L | Exact | Labor share from c̃_r, Y_r |
| Ω^s | Exact | X_s/X from γ_{r'sdr} × expenditure |
| π_r | Exact | Y_r / Σ Y_r |
| γ_ls | Exact | T_{r's}(wτ)^{-θ}/Φ_{s,dr} × exp_sdr |
| reg_coef | Quadrature | Gauss-Legendre n_quad nodes on Fréchet CDF |

**Consequences:**
- Σ_sim = 0 by construction → W_eff = Σ_data^{-1} directly
- Jacobian via finite differences with K=1 (deterministic, no simulation noise)
- No Murphy-Topel correction needed
- ~100× faster per moment evaluation vs SMM at N_ρ=2000

**Entry points:**
- `julia main_gmm.jl aero 4` — GMM (n_quad=200)
- `julia main_gmm.jl aero 4 "" "" 500` — GMM (n_quad=500, high accuracy)
- `./run.sh aero 4 "" "" --mode=gmm` — via shell script

**Key constants added to `load_parameters.jl`:**
- `GAMMA_FACTOR[s] = Γ((θ+1-ν_s)/θ)^{1/(1-ν_s)}` — price index normalization factor

**Validation:**
- Use `test_analytical_vs_simulated(params; N_rho_test=10_000)` to verify analytical
  moments against high-accuracy SMM. Expected: max relative error < 1e-3 for
  {Ω^L, Ω^s, π_r, γ_ls} and < 1e-2 for reg_coef.

---

2026-05-13 · `main.jl`, `main_pso.jl`, `claude.md` · Align MOMENT_MASK γ_ls drop with T_REF_REGION normalization and restrict Jacobian to identified parameters. MOMENT_MASK now drops `γ_{ls}` at `T_REF_REGION[s]` (largest empirical sourcing share) instead of the first active region, matching the `T_mat[s,:] ./= T_mat[s, ref_r]` normalization in `unpack_params`. `main_pso.jl` gains the same `T_REF_REGION` computation. `jacobian_param_indices` excludes the S+2 flat directions (first Ω^s, A_1, T at ref region per sector); `compute_jacobian` and `compute_smm_inference` now receive these indices instead of `nothing`/`collect(1:p)`.

2026-05-04 · `tools.jl` · Fix `TypeError` in `compute_smm_inference`: element-wise `block_omega_sd .> 1e-15` returned a `BitVector` fed into a scalar ternary `?:`; replaced with `ifelse.()` for correct element-wise dispatch.

2026-05-04 · `tools.jl`, `main.jl`, `claude.md`, `README.md` · Extend SMM inference to all parameters and both estimation steps. Jacobian now computed over all parameters (`param_indices=nothing`) at θ̂_1 (→ `step2/jacobian_all.npy`) and θ̂_2 (→ `step3/jacobian_all_step3.npy`). `compute_smm_inference` called after each — at θ̂_1 with W=W_step3/Ω into `step2/inference/`, at θ̂_2 into `step3/inference/`. Hardcoded "Step 3 β and T only" caveat replaced with a generic Murphy–Topel note. Earlier same-day commit also removed the `(1+1/K)` factor and added the inference infrastructure.

2026-04-22 · `main.jl` (new), `model_CP.jl`, `tools.jl`, `pso_integration.jl`, `claude.md` · Implement three-step efficient SMM. `main.jl` orchestrates Step 1 (identity weight), Step 2 (`build_step3_weight_matrix` via K re-seeded pmap evaluations), and Step 3 (efficient weight, warm-started at θ̂_1). `generate_stratified_draws` gains `seed_offset` kwarg; `full_SMM`/`parallel_SMM_safe`/`train_stage_pso` gain `W_override`/`weight_matrix` kwargs for non-destructive weight injection. `main_pso.jl` is unchanged (legacy).

