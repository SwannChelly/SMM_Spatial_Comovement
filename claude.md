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
├── model_CP.jl                  # Core: solve_network, compute_moments, full_SMM, log-T (φ) transforms
├── tools.jl                     # Reporting, LHS/Halton grids, jacobian/inference, weight matrix
├── optimizer.jl                 # Backend-neutral hub: optimize_stage (dispatch), train_stage, run_optimization
├── pso_integration.jl           # PSO backend: parallel_pso_smm + enforce_beta_constraint
├── cmaes_integration.jl         # CMA-ES backend: parallel_cmaes_smm (unit-cube search + incumbent tracker)
├── main.jl                      # Entry point: three-step efficient SMM (Steps 1–3), --optimizer=pso|cmaes
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
- `generate_draws(N_rho, n_good, method; randomise=false, rng, a=0.5, verbose=false)`: unified draw dispatcher. `method ∈ (:qmc, :mc, :is)`. Returns `(U, W)` with `W` **always** an `(N_rho × n_good)` matrix (the invariant that confines the method switch to the generation sites — every `[rho,g]` consumer is untouched). `:qmc` (**default**) = stratified uniform with flat weights and decorrelated columns; `:mc` = i.i.d. uniform, flat weights; `:is` = per-column importance sampling (tilt `a`). See **Draw methods (`:qmc` default) and the IS bias** below.
- `generate_qmc_draws(N_rho, n_good; randomise=false, rng)`: the default sampler. Stratified uniform draws (independent permutation per column → decorrelated, one point per equal-prob stratum → LHS variance reduction), **flat** weight matrix `1/N_rho`. `randomise=false` → `MersenneTwister(g)`+midpoint (PSO-deterministic base); `randomise=true` → supplied `rng` perm+jitter (Σ_sim replications, columns share one stream).
- `generate_is_draws(N_rho, n_good; randomise=false, rng, a=0.5, verbose=false)`: per-column importance-sampling uniform draws for the Fréchet inverse CDF. Returns `(U, W)`, **both** `(N_rho × n_good)` matrices, `W` normalised per column (`sum(W[:,g])=1`). **BIASED for the min-coupled consumer moments** (see below) — not the default; retained for the `reg_coef` quadrature tail.
- `generate_stratified_draws(...)`: backward-compatible alias resolving to `generate_draws(..., :qmc)` (the new default). `a`/`verbose` accepted for signature parity, ignored by `:qmc`.
- `unpack_params(params)`: splits flat vector into `(β, Ω^L, Ω^s, A, T_full)`. `T_full` is `S×R` with zeros for inactive pairs.
- `build_tau(beta)`: builds `(R, R_downstream)` iceberg cost matrix from distance-bin coefficients using precomputed `DistBin`.
- `solve_network(params; return_firm_level, precomputed_tau, u_draws, sample_weights)`: main equilibrium solver. Returns NamedTuple. If `return_firm_level=true`, appends sparse COO firm-level arrays (`firm_exp_rho/s/g/r/val`, `firm_deriv_val`).
- `compute_moments(network, params)`: extracts all 5 moment blocks from solved network.
- `fast_weighted_regression(linkages_flat, z_flat, sample_weights)`: analytical weighted OLS with FWL demeaning by sector×downstream-region FE. Returns `N_beta` distance-bin coefficients. `sample_weights` is the `(N_rho × n_good)` IS matrix; each `(rho, g)` row carries weight `sample_weights[rho, g]`.
- `full_SMM(params, simulation=false, second_stage=false, method="original"; ..., W_override=nothing)`: wrapper returning `(loss, moments_tuple)`. When `W_override` is not nothing it is used instead of the global `Weight_matrix_custom`; existing call sites are unaffected. **`second_stage` is an ignored/defaulted positional arg** — the old masked-moment second-stage branch was dead (referenced `empirical_moments_reduced`/`mask_emp_gamma_ls`, never defined in `load_parameters.jl`) and was removed; the arg is retained only because `parallel_SMM`/`parallel_SMM_safe`/`train_stage_pso` forward it positionally.

#### Draw methods (`:qmc` default) and the IS bias

The Fréchet inverse-CDF transform consumes uniform draws `u ∈ (0,1)`. Three
samplers are available via `generate_draws(N_rho, n_good, method)`, selectable at
the entry point (`draw_method`, CLI `--draws=qmc|mc|is`, default `:qmc`):

- **`:qmc` (default).** Stratified uniform, **flat** weights. Each good pair `g`
  gets an independent permutation (`u[rho,g1] ⊥ u[rho,g2]` — required to keep
  Ricardian `min_r c_r` selection alive; the old shared-bin design collapsed it),
  with one point per equal-probability stratum (LHS variance reduction). Flat
  weights make `solve_network`'s winner-weight shortcut **exact** (see below), so
  it is **unbiased** for the min-coupled consumer moments and weakly beats `:mc`.
- **`:mc`.** Plain i.i.d. uniform, flat weights. Validates the closed form itself.
- **`:is`.** Per-column importance sampling. **Biased** for the min-coupled
  moments — kept only for `reg_coef` tail resolution.

**Why `:is` is biased (the winner-weight shortcut).** `solve_network`'s within-
sector CES price index `P_sr = [Σ_ρ w_ρ p_ρ^{1-ν_s}]^{1/(1-ν_s)}` weights each
row `ρ` by **only the winning column's** weight `W[rho, winner_good_idx[rho,s]]`
— it never multiplies in the importance-density ratios of the *losing* columns.
The realised per-row factor is therefore the correct importance weight for the
joint (min over regions) functional **only when all columns carry the same
weight**, i.e. flat weights. With a non-flat tilt the dropped losing-column
ratios do not cancel, so `c_tilde`, `gamma_ls`, `industry`, `pi_r` are estimated
with a systematic bias that does **not** vanish as `N→∞`. The earlier
"bounded weights ⇒ no degeneracy" claim is only true for **single-column**
functionals (the `reg_coef` quadrature rows, which have no min-coupling) — there
`:is` remains valid and its `u→0` tilt buys real tail resolution. It is false for
every min-coupled block. This is why the default switched to `:qmc`.

**Proposal (`:is` only).** The inverse-CDF `z = T^{1/theta}(-ln(1-u))^{-1/theta}`
puts the selection-relevant large-`z` mass at `u→0`. Proposal `q(u) = a·u^{a-1}`
(`Beta(a,1)`, inverse-CDF `u = v^{1/a}`) oversamples `u→0`; target `Uniform` ⇒ IS
weight `w ∝ 1/q(u) = u^{1-a}/a`, bounded on `[0,1]`. Recommended `a = 0.5`. The
tilt is keyed to the `−log(1−u)` transform (high-z mass at `u→0`); do not flip the
draw without recomputing the weight from the flipped quantile.

**Weights are always a matrix.** `SAMPLE_WEIGHTS` / `U_DRAWS` are both
`(N_rho × n_good)` for every method. Every consumer indexes `[rho, g]`:
`solve_network` weights the CES `P_sr` sum and `exp_val` by the winning pair's
column `winner_good_idx[rho,s]`; `fast_weighted_regression` uses `W[rho,g]`. This
matrix interface is the invariant that lets the method switch stay confined to the
generation sites — consumer indexing, the weight-matrix construction, `MOMENT_MASK`,
the T flat-indexing, and inference are all untouched by the choice of sampler.

**`test_price_alignment`** (in `main_gmm.jl`) compares all three methods against
the closed form. Expected at `θ̂_2`: `:qmc ≤ :mc` on `c_tilde`/`industry`/`pi_r`/
`gamma_ls` (stratification ≥ i.i.d.), `:is` visibly biased on those blocks (it
documents the defect, it does not hide it), and `reg_coef ~1e2` for **all three**
(FKG/quadrature bias in the regression moment, not a sampling artefact).

**Flat indexing convention** (critical): active `(s, r)` pairs are stored in a flat array of length `n_good`. Global constants `GOOD_S[g]`, `GOOD_R[g]` map index `g` back to `(s, r)`. `SR_TO_GOOD[s, r]` is the reverse map (0 for inactive). `SECTOR_GOOD_INDICES[s]` lists all `g` indices for sector `s`.

**Sparse COO convention** (firm-level): `firm_exp_g[i]` is the good-pair index; `GOOD_R[firm_exp_g[i]]` gives the upstream region. Not `firm_exp_r[i]` — that is the *downstream* region.

### `tools.jl` — helpers

Key utilities:
- `generate_lhs_beta(n, n_beta, lb, ub)`: LHS samples with sorted draws → monotone β.
- `generate_draws` / `generate_qmc_draws` / `generate_is_draws` / `generate_mc_draws` (and the `generate_stratified_draws` alias) are in `model_CP.jl`, not here.
- `parallel_SMM_safe(...; W_override=nothing)`: try-catch wrapper returning `nothing` on failure. Threads `W_override` to `full_SMM`.
- `distance_bin(d, n_bins)`: maps distance (km) to bin index 1..N_beta (0 = local, excluded from regression).
- `generate_report(...)`: saves `dashboard.png`, `report.txt`, `pi_r.npy`, `productivity.npy`.
- `run_reporting(output_folder, max_loop)`: scans all epoch/stage folders, computes scores, saves `best_simulated_moments.npy`, `best_parameters_list.npy`.
- `find_last_stage_folder(base_folder)`: locates most recent `best_params.npy`.
- `find_resume_state(base_folder)`: determines `(resume_loop, resume_substage)` for `--resume` mode.
- `build_step3_weight_matrix(theta_hat_1, input_folder; K, output_folder)`: assembles `W_step3 = (Σ_data + Σ_sim)^{-1}` restricted to γ_ls and reg_coef moments. `Σ_data` loaded from `Sigma.npy` (joint bootstrap covariance of γ+β, size `n_gamma_kept + n_beta_kept`). `Σ_sim` estimated via K pmap evaluations with MC draws seeded by k, restricted to the same indices. The returned `W_step3` is `N_moments × N_moments` with zeros outside the γ+β sub-block. Saves `Sigma_data.npy`, `Sigma_sim.npy`, `Omega.npy`, `Omega_gb.npy`, `W_step3.npy`, `W_gb.npy`, `diagnostics.txt` to `output_folder/step2/`.
- `compute_jacobian(theta; K, param_indices, step_rel, step_abs, base_seed, output_subdir, analytical, n_quad, t_log_step, check_symmetry, richardson_check)`: central finite differences of the masked moment vector w.r.t. selected parameters, averaged over `K` re-seeded replications. **Two step regimes** keyed on the flat parameter position: **T columns** (strictly-positive, multiplicatively-entering trade-cost levels; flat index `≥ 1 + S + R_downstream + N_TAU + 1`) take a **log-space central step** `θ·exp(±δ)` with `δ = step_rel`, converted back to raw `∂m/∂θ` by dividing by `θ_j` (chain rule `∂m/∂θ = (∂m/∂lnθ)·(1/θ)`); **all other columns** (`Ω^L`, `Ω^s`, `A`, `β`/`α`) take the additive step `h = max(|θ|·step_rel, step_abs)`. **In both regimes the stored column is raw `∂m/∂θ`**, so `G'WG` inference is unaffected in units — the log step is purely a numerical-accuracy device (scale-invariant, immune to the additive floor, never straddles the Fréchet `eps` clamp). `t_log_step=true` (default) enables this; `t_log_step=false` recovers the byte-identical additive behaviour for every column. `check_symmetry=false` (opt-in) prints a forward-vs-backward asymmetry diagnostic on T columns; `richardson_check=false` (opt-in) recomputes T columns at `2δ` and reports the relative gap. Both diagnostics are print-only and never alter the returned `J`. The analytical branch (`analytical=true`) routes T columns through the same per-column log logic. Saves `J`, `J_elast`, `J_sd`, `J_elast_sd`, and the param-index map under `<output_folder>/<output_subdir>/`.
- `compute_smm_inference(theta_hat, J, W, Omega; param_indices, empirical_moments_vec, simulated_moments_vec, output_folder, industry, K_sim)`: computes delta-method standard errors (efficient and sandwich), fitted-moment SEs, moment-residual SEs, and the Hansen J over-identification test. Saves all arrays plus `inference_summary.txt` and `J_stat.txt` to `output_folder/inference/`. Returns a Dict.
- (`run_optimization` moved out of `tools.jl` → now in `optimizer.jl`; see the Optimizer layer section.)

### Optimizer layer (`--optimizer=pso|cmaes`)

The optimization layer is four files with a clean separation:
- **`optimizer.jl`** — backend-neutral hub: `optimize_stage` (dispatch), `train_stage` (stage builder), `run_optimization` (per-step orchestrator), layout helpers, and the legacy aliases.
- **`pso_integration.jl`** — PSO backend only (`parallel_pso_smm` + shared `enforce_beta_constraint`).
- **`cmaes_integration.jl`** — CMA-ES backend only (`parallel_cmaes_smm`).
- Both backends honor the contract `(objective, lb, ub; x0, n_particles, max_iter, beta_constraint, beta_indices, verbose) -> (best_x, best_f, history)`, so `train_stage`, `run_optimization`, and `main.jl` are optimizer-agnostic.

Include order (all `@everywhere`): `pso_integration.jl`, `cmaes_integration.jl`, `optimizer.jl` — before `load_parameters.jl` (which defines `OPTIMIZER_BACKEND`), which is fine because every cross-file reference (and `optimize_stage`'s `backend=OPTIMIZER_BACKEND` default) resolves at call time. All three live entry points (`main.jl`, `main_gmm.jl`, `run_internal_validity.jl`) include all three optimizer files. **`main_pso.jl` is frozen and NOT wired for this layout** (it includes only `pso_integration.jl` and calls the `train_stage_pso` alias, which now lives in `optimizer.jl`).

The two-step SMM procedure (Step 1 identity-W all params → Step 2 W_step3 → Step 3 efficient-W β+T) is the **same** for both backends; only the intra-step search differs (see `run_optimization`).

**log-T (φ) search space (both backends).** The optimizer never sees raw `T`. The T
block is searched as free log-space `φ_i = log(T_i / T_{s,ref})`, with each sector's
reference entry **dropped** (pinned `T_{s,ref}=1`), so the search dimension is
`N_T_FREE = sum(T_MASK) − #active_sectors` and the S unidentified directions never
enter. Conversion is confined to `model_CP.jl`'s `t_levels_to_free_phi` /
`t_free_phi_to_levels` / `full_to_search` / `search_to_full`; **every disk artifact
and all of `full_SMM`/inference/reporting stay in raw T levels** (reference entries
reconstruct to 1, so `unpack_params`' `./= T_mat[s,ref]` is a no-op). This mirrors
the draw-method invariant. Fresh-start T bounds are `φ ∈ [log 0.1, log 10]` (symmetric,
was `[0.1,10]×T_init`); continuation T bounds are `φ ± |log alpha|`.

### `optimizer.jl` — backend-neutral hub

- `optimize_stage(obj, lb, ub; x0, n_particles, max_iter, beta_constraint, beta_indices, verbose, backend=OPTIMIZER_BACKEND, seed)`: dispatches to `parallel_pso_smm` (x0 → warm-start particle) or `parallel_cmaes_smm` (x0 → initial mean + incumbent floor).
- `train_stage(n_particles, max_iter; variable_list, last_stage_folder, alpha, weight_matrix=nothing, warm_start_override=nothing, ...)`: builds bounds around the previous best (multiplicative `alpha`; additive-in-`φ` for the T block), constructs the warm start, defines the stage objective, calls `optimize_stage`, and reconstructs the full level vector. The T block of every stage vector holds free φ (length `N_T_FREE`), expanded to levels via `t_free_phi_to_levels`. Alias: `train_stage_pso`.
- `run_optimization(; weight_matrix, skip_initial_beta_search, warm_start_params, output_subfolder, max_loop, ...)`: per-step orchestrator (moved out of `tools.jl`). Runs Stage 0 LHS β search (unless `skip_initial_beta_search`) + Stage 1. **PSO backend**: Stage 1 + `max_loop`-epoch × 3-substage refinement. **CMA-ES backend**: refinement loop skipped (`for … in 1:0`) — Stage 1's single joint run is the whole search. Alias: `run_pso_optimization` (used by `main_gmm.jl`, `run_internal_validity.jl`).
- `get_param_start_index` / `get_n_T_params`: flat-layout helpers.

### `pso_integration.jl` — PSO backend

- `parallel_pso_smm(obj, lb, ub; n_particles, max_iter, warm_start_particle, ...)`: core PSO. Always includes `warm_start_particle` as one particle (guarantees monotone improvement). Restarts stagnant particles every 25 iterations when `fitness > 1.5 × g_best`.
- `enforce_beta_constraint(params, beta_indices)`: β-ordering repair (sort), shared with the CMA-ES backend (kept here so the frozen `main_pso.jl`'s `parallel_pso_smm` still resolves it).

### `cmaes_integration.jl` — CMA-ES backend

- `parallel_cmaes_smm(obj, lb, ub; x0, n_particles→λ, max_iter, beta_constraint, beta_indices, sigma0=0.2, seed=1, verbose)`: wraps `CMAEvolutionStrategy.minimize`. Runs on the **unit cube [0,1]^n** (mapped to `[lb,ub]` inside the evaluator) so a single scalar `σ0` is meaningful across heterogeneous coordinates. Population is evaluated via `pmap` (`parallel_evaluation=true`; library passes an `n × popsize` matrix). **Incumbent tracker** runs master-side after `pmap` (Refs inside the pmapped closure would be worker-local and lost) and compares against `f(x0)` before returning, restoring the "loss never worse than warm start" guarantee. β ordering enforced by the same `enforce_beta_constraint` repair as PSO. `history` supersets PSO's keys; σ and axis-ratio print to console at `verbosity=1`.

### `main.jl` — three-step efficient SMM (primary entry point)

`julia main.jl [industry] [n_coef] [n_tau] [K_sim] [draws] [optimizer]`

`optimizer ∈ {pso (default), cmaes}` (ARGS[6]); read into `OPTIMIZER_BACKEND`. With
`--optimizer=cmaes`, each step's staged refinement collapses into one joint CMA-ES run
(see `run_pso_optimization`).

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
| `OPTIMIZER_BACKEND` | Symbol | `:pso` (default) or `:cmaes`, from `--optimizer` (ARGS[6]) |
| `T_REDUCED_S` | Vector{Int} | Sector per reduced-T position (T_MASK order) |
| `SECTOR_REF_REDUCED` | Vector{Int} | Reduced-T index of each sector's reference (pinned T=1) |
| `T_FREE_REDUCED_IDX` | Vector{Int} | Reduced-T positions the optimizer varies as φ |
| `N_T_REDUCED` / `N_T_FREE` | Int | `sum(T_MASK)` / free count (`= N_T_REDUCED − #active sectors`) |

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

2026-07-04 · `optimizer.jl` (new), `pso_integration.jl`, `tools.jl`, `main.jl`, `main_gmm.jl`, `run_internal_validity.jl`, `claude.md` · Restructure the optimization layer for clean separation. New `optimizer.jl` is the backend-neutral hub holding `optimize_stage` (dispatch), `train_stage` (renamed from `train_stage_pso`, moved from `pso_integration.jl`), `run_optimization` (moved from `tools.jl`), and the layout helpers; `pso_integration.jl` is now a pure PSO backend (`parallel_pso_smm` + shared `enforce_beta_constraint`) and `cmaes_integration.jl` a pure CMA-ES backend. `optimizer_api.jl` removed (folded into `optimizer.jl`). Aliases `train_stage_pso`/`run_pso_optimization` retained. Fixed a regression from the CMA-ES commit: `main_gmm.jl` and `run_internal_validity.jl` called `run_pso_optimization`→`train_stage_pso`→`optimize_stage` but never included the dispatcher — both now include `cmaes_integration.jl` + `optimizer.jl`. `main.jl` carries no PSO-specific names. `main_pso.jl` (frozen) is not wired for the new layout.

2026-07-04 · `cmaes_integration.jl` (new), `optimizer_api.jl` (new, later folded into `optimizer.jl`), `model_CP.jl`, `pso_integration.jl`, `tools.jl`, `load_parameters.jl`, `main.jl`, `run.sh`, `pkg.jl`, `claude.md` · Add a CMA-ES optimizer backend behind `--optimizer=pso|cmaes` (ARGS[6] → `OPTIMIZER_BACKEND`), and reparameterize the T block to free log-space φ for both backends. `optimize_stage` (optimizer_api.jl) is the single dispatch seam; `parallel_cmaes_smm` wraps `CMAEvolutionStrategy.minimize` on the unit cube with a master-side incumbent tracker (monotone vs warm start) and `pmap` population evaluation. `train_stage_pso` now calls `optimize_stage` and searches T as `φ_i = log(T_i/T_{s,ref})` with reference entries dropped (`N_T_FREE = sum(T_MASK) − #active sectors`); conversions `t_levels_to_free_phi`/`t_free_phi_to_levels`/`full_to_search`/`search_to_full` live in `model_CP.jl` and keep every disk artifact + `full_SMM`/inference in raw T levels (ref reconstructs to 1, so `unpack_params` normalization is a no-op). CMA-ES collapses each step's refinement loop (`run_optimization` runs `1:0`) into one joint Stage-1 run; the two-step SMM structure (Step 1 identity-W all params, Step 3 efficient-W β+T) is unchanged for both. `run_pso_optimization` renamed to backend-agnostic `run_optimization` (alias kept); `main.jl` now calls `run_optimization` and carries no PSO-specific names. `main_pso.jl` (legacy) left untouched — it would need the two new includes to run under the renamed call path.

2026-06-20 · `model_CP.jl`, `load_parameters.jl`, `CLAUDE.md`, `claude.md` · Remove the dead `second_stage` framework and reorganize `load_parameters.jl`. Verified the `second_stage=true` branch of `full_SMM` is dead — it reads `empirical_moments_reduced`/`Weight_matrix`/`mask_emp_gamma_ls` (defined only in `run_amplification_analysis.jl`, which neither includes `load_parameters.jl` nor calls `full_SMM`), and every live caller passes `second_stage=false`; collapsed it to the `else` body. Kept `second_stage` as an ignored positional arg (removing it would ripple through `parallel_SMM`/`parallel_SMM_safe`/`train_stage_one`/`pso_integration.jl`, which forward it positionally); `main_pso.jl` untouched. In `load_parameters.jl` removed the orphaned `Weight_matrix=nothing` const, the unused `N_rs_local` load, and a duplicate `n_reg` recompute, then grouped the file into 13 banner-delimited sections with one-line role comments — exact execution order, constant values, and all `@everywhere` broadcasts unchanged.

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

