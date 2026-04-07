# claude.md — SMM Spatial Comovement: Developer & Economist Reference

## Review standard

Every change must pass two filters before being written:
- **Senior economist**: Does it respect the structural model's identification logic, moment targeting, and equilibrium conditions? Does notation stay consistent with the paper?
- **Senior developer**: Is the implementation efficient (time/memory), correct, minimal, and maintainable? No unnecessary abstraction. No breaking of existing interfaces.

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

Sum-to-1 redundancies are dropped via `MOMENT_MASK` (one per share-constrained block).

---

## File structure

```
SMM_Spatial_Comovement/
├── model_CP.jl                  # Core: solve_network, compute_moments, full_SMM
├── tools.jl                     # Optimization helpers, reporting, LHS/Halton grids
├── pso_integration.jl           # PSO: parallel_pso_smm, train_stage_pso
├── main_pso.jl                  # Entry point: PSO optimizer (multi-stage refinement)
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
- `generate_stratified_draws(N_rho)`: CdGM-style stratified uniform draws for Fréchet inverse CDF. Returns `(u_quantiles, weights)`, both length `N_rho`. Deterministic; replaces random draws.
- `unpack_params(params)`: splits flat vector into `(β, Ω^L, Ω^s, A, T_full)`. `T_full` is `S×R` with zeros for inactive pairs.
- `build_tau(beta)`: builds `(R, R)` iceberg cost matrix from distance-bin coefficients using precomputed `DistBin`.
- `solve_network(params; return_firm_level, precomputed_tau, u_draws, sample_weights)`: main equilibrium solver. Returns NamedTuple. If `return_firm_level=true`, appends sparse COO firm-level arrays (`firm_exp_rho/s/g/r/val`, `firm_deriv_val`).
- `compute_moments(network, params)`: extracts all 5 moment blocks from solved network.
- `fast_weighted_regression(linkages_flat, z_flat, sample_weights)`: analytical weighted OLS with FWL demeaning by sector×downstream-region FE. Returns `N_beta` distance-bin coefficients.
- `full_SMM(params, ...)`: wrapper returning `(loss, moments_tuple)`.

**Flat indexing convention** (critical): active `(s, r)` pairs are stored in a flat array of length `n_good`. Global constants `GOOD_S[g]`, `GOOD_R[g]` map index `g` back to `(s, r)`. `SR_TO_GOOD[s, r]` is the reverse map (0 for inactive). `SECTOR_GOOD_INDICES[s]` lists all `g` indices for sector `s`.

**Sparse COO convention** (firm-level): `firm_exp_g[i]` is the good-pair index; `GOOD_R[firm_exp_g[i]]` gives the upstream region. Not `firm_exp_r[i]` — that is the *downstream* region.

### `tools.jl` — helpers

Key utilities:
- `generate_lhs_beta(n, n_beta, lb, ub)`: LHS samples with sorted draws → monotone β.
- `generate_stratified_draws` is in `model_CP.jl`, not here.
- `parallel_SMM_safe(...)`: try-catch wrapper returning `nothing` on failure.
- `distance_bin(d, n_bins)`: maps distance (km) to bin index 1..N_beta (0 = local, excluded from regression).
- `generate_report(...)`: saves `dashboard.png`, `report.txt`, `pi_r.npy`, `productivity.npy`.
- `run_reporting(output_folder, max_loop)`: scans all epoch/stage folders, computes scores, saves `best_simulated_moments.npy`, `best_parameters_list.npy`.
- `find_last_stage_folder(base_folder)`: locates most recent `best_params.npy`.
- `find_resume_state(base_folder)`: determines `(resume_loop, resume_substage)` for `--resume` mode.

### `pso_integration.jl` — PSO optimizer

- `parallel_pso_smm(obj, lb, ub; n_particles, max_iter, warm_start_particle, ...)`: core PSO. Always includes `warm_start_particle` as one particle (guarantees monotone improvement). Restarts stagnant particles every 25 iterations when `fitness > 1.5 × g_best`.
- `train_stage_pso(n_particles, max_iter; variable_list, last_stage_folder, alpha, ...)`: builds bounds around previous best (multiplicative `alpha`), constructs warm start, defines stage-specific objective, calls `parallel_pso_smm`.
- `get_param_start_index(param_name)`: maps `:beta/:agg_labor_share_tech/:agg_industry_share_tech/:productivity/:T` to starting index in flat param vector.

### `main_pso.jl` — optimization pipeline

Stages:
1. **Stage 0** (LHS beta search): 1500 LHS samples of `β`, fixed other params from analytical guess. Selects best by regression-coefficient distance.
2. **Stage 1** (full PSO, `MAX_ITER_INITIAL=200`): all parameters jointly.
3. **Refinement loops** (`max_loop=50`, 3 sub-stages each):
   - Sub-stage 1: productivity `A_r` (tightest alpha, `ε` amplifies errors)
   - Sub-stage 2: spatial structure `β, T` (medium alpha)
   - Sub-stage 3: technical coefficients `Ω^L, Ω^s` (standard alpha)
   - Alpha schedule: `0.3 → 0.9` over loops.

**Weight matrix**: loaded from `weight_vector.npy` (pre-built in Python), applied as `Diagonal(weight_vector)`.

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
| `N_rho` | Int | Varieties per sector (= 100 in production runs) |
| `n_good` | Int | Number of active (sector, region) pairs |
| `GOOD_S`, `GOOD_R` | Vector{Int} | Sector/region for each good-pair index |
| `SR_TO_GOOD` | Matrix{Int} | (S, R) → good-pair index (0 if inactive) |
| `SECTOR_GOOD_INDICES` | Vector{Vector{Int}} | Good-pair indices per sector |
| `T_MASK` | BitVector | Which entries of `vec(T_rs)` are active |
| `DistBin` | Matrix{Int} | Precomputed distance bins (R × R) |
| `CLOSEST_PLANT_DIST` | Vector{Float64} | Distance to nearest downstream plant (per upstream region) |
| `CLOSEST_DOWNSTREAM_REGION` | Vector{Int} | Nearest downstream region (per upstream region) |
| `U_DRAWS`, `SAMPLE_WEIGHTS` | Vector{Float64} | CdGM stratified draws + weights |
| `emp_gamma_ls` | Matrix{Float64} | Empirical sourcing shares (S × R) |
| `emp_pi_r` | Vector{Float64} | Empirical downstream sales shares |
| `reg_coef` | Vector{Float64} | Empirical regression coefficients (length N_beta) |
| `empirical_moments` | Matrix{Float64} | Masked empirical moments (1 × N_moments) |
| `MOMENT_MASK` | BitVector | Which moments to keep (drops sum-to-1 redundancies) |
| `Weight_matrix_custom` | Diagonal | SMM weight matrix |
| `N_beta` | Int | Number of distance bins (4 or 5) |

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
| `weight_vector.npy` | (N_moments,) | SMM diagonal weight vector |

---

## Output structure (`reporting_{industry}/`)

```
0/                     # Initial PSO stage
epoch_{k}/
  {1,2,3}/             # Three sub-stages per loop
    best_params.npy    # Parameter vector (n_params × 1)
    dashboard.png      # γ_ls, π_r, π_s scatter plots
    report.txt         # Moment comparison table
    pi_r.npy
    productivity.npy
best_simulated_moments.npy
best_parameters_list.npy
empirical_moments.npy
simulated_panel_unified.parquet
suppliers.parquet
w_srd_r.npy            # w_{s,r',r}: upstream (s,r') sales share to downstream r
```

---

## Keeping claude.md current

If a change modifies anything documented in this file — file structure, function signatures, global constants, parameter vector layout, moment construction, input/output conventions, or the optimization pipeline — **claude.md must be updated in the same step as the code change**. Do not defer the update. The changelog entry alone is not sufficient: the relevant section of the reference documentation must also reflect the new state.

---

## Changelog

*Format: date · file(s) changed · description (≤4 sentences)*

<!-- Add entries below in reverse chronological order -->

