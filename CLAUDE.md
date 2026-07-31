# SMM Spatial Comovement — Codebase Guide

Three-step efficient SMM/GMM estimator of a spatial Eaton–Kortum sourcing model with
comovement moments. This file is the working memory: architecture, invariants, and the
caveats that are expensive to rediscover. History lives in `git log`.

---

## Where things are

Julia `include()` paths resolve against the *including file's* directory: root entry points
reach into `optimizers/`, `test/`, `extras/`; files in those folders reach back with `../`.

**Root — the optimization core**

| File | Role |
|---|---|
| `main.jl` | SMM entry point (simulation-based moments) |
| `main_gmm.jl` | GMM entry point (analytical EK moments) |
| `model_CP.jl` | Model: `solve_network`, `compute_moments`, `full_SMM`, `loss_function`, draw generators, `fast_weighted_regression` / `fast_cloglog_regression` |
| `model_analytical.jl` | Closed-form EK moments + Gauss–Legendre quadrature for `reg_coef`; forward-AD Jacobian |
| `optimizer.jl` | Backend-neutral search hub: `optimize_stage`, `train_stage`, `run_optimization` |
| `profiling.jl` | `invert_T_ge` — Sinkhorn/GE inversion that profiles `T` out of the search |
| `tools.jl` | Inference, Jacobian, weight matrix, reporting, identification screens |
| `load_parameters.jl` | Data loading and every broadcast global (`BLOCK_RANGES`, `MOMENT_MASK`, `U_DRAWS`, masks, labels) |

**Folders** — `optimizers/` (PSO, CMA-ES, TikTak backends behind `--optimizer=`),
`documentation/`, `test/`, `extras/` (post-estimation analysis). Each has a `files.md`.

**Key docs** — `documentation/model.md` (model + math), `inference.md` (SEs, Jacobian,
Hansen J), `optimizer.md` (backends), and for the current direction — granular varieties with
comparative advantage at the attraction-area level — `finite_sample2.tex` (the model),
`plan_granular_aa.md` (what to build) and `granular_validation.md` (gates + open assumptions).
The granular work is behind `--granular` and `--ca_level`; both at their legacy values
(`false`, `ze`) must reproduce today's estimates.

---

## Moment blocks

Five blocks in the masked moment vector; `BLOCK_RANGES` is a 5-tuple of index ranges into it.

| # | Name | Content |
|---|---|---|
| 1 | `agg_labor_share` | aggregate labor share |
| 2 | `agg_industry_share` | industry shares π_s |
| 3 | `pi_r` | regional market shares |
| 4 | `reg_coef` | extensive-margin distance coefficients, `N_REG` entries. Link set by `REG_METHOD` (`:cloglog` default, `:lpm` legacy); outcome is `not_supply` — the exact firm-level form: `b_logd = +θα`, and the `log z` coefficient equals `−θ` |
| 5 | `gamma_ls` | location-specific linkage shares γ_ls |

---

## Invariants — do not break these

**β then γ.** Everywhere: `Sigma_beta_gamma*.npy`, `W_step3`, `Omega`, `gb_indices`.

```julia
gb_indices = vcat(collect(BLOCK_RANGES[4]), collect(BLOCK_RANGES[5]))
```

**T flattening is s-major**, the same enumeration as the γ-moment rows, so a `T[s,r]` Jacobian
column aligns with its `γ[s,r]` row.

```julia
T_MASK = vec(permutedims(X_rs)) .> 0     # == T_MASK_MOMENT
```

* flat position of active `(s,r)`: `(s-1)*R_full + r`; inverse `s = (p-1)÷R_full + 1`, `r = (p-1)%R_full + 1`
* `unpack_params` inverts it and returns **region-major** `vec(T_mat)` — only the *reduced*
  (active) block is s-major
* every reduced-T producer must emit s-major: `vec(permutedims(T_rs_init))[T_MASK]`
* `GOOD_S`/`GOOD_R` are region-major and are *not* required to match; the model reads
  `T_mat[s,r]` through `SR_TO_GOOD`
* a partial application silently transposes `T_mat` — runs fine, fits the wrong T. Gate:
  `test/test_t_reorder.jl`

**`N_REG` ≠ `N_TAU`.** Two independent constants:

| | Meaning | Axis | Used by |
|---|---|---|---|
| `N_REG` | reg_coef moment count | moment | the regressions, `distance_bin`, `BLOCK_RANGES[4]`, Σ file selection |
| `N_TAU` | trade-cost parameter count | parameter | `build_tau`, `unpack_params`, PSO β-slice, `PARAM_LABELS` |

Default `n_tau = n_coef`. The over-identified configuration is `n_tau=1, n_coef=4`: power law
`τ = d^α` with four binned moments, df=3 on the reg_coef block. `PARAM_LABELS` writes `"alpha"`
when `N_TAU==1`.

**Parameter layout** — `[Ω^L(1) | Ω^s(S) | A(R_downstream) | α(N_TAU) | T(reduced)]`.

---

## Critical caveats

**⚠ The analytical/AD Jacobian is FKG-biased on α — never use it for α inference.** The
closed-form `reg_coef` (`compute_regression_quadrature`, hence `analytical_jacobian_ad` and any
`analytical=true` path) builds the extensive margin with the FKG product approximation
`1 − ∏_dr(1 − ρ_dr)` instead of the exact union-of-boxes measure. That biases `∂reg_coef/∂α`.
The AD Jacobian is exact for `γ_ls`/shares but biased on the α column. For α SEs and
identification use the **simulation-based** Jacobian (`main.jl`, `analytical=false`). The
AD-vs-FD agreement to `O(δ²)` does **not** clear this: both share the same FKG regressand, so
the bias is in the estimand, not the differentiation.

**⚠ `H = J'WJ` on T is block-diagonal by sector — read it per sector.** Each sector's T columns
enter only that sector's `γ_ls` moments. The global `H[T,T]` λ_min / condition number mixes
sectors and is dominated by cross-sector coupling that is a weighting/units artifact (expect a
λ_min eigenvector concentrated on one sector, and large cross-sector "coherence" — both normal).
Judge T identification from the per-sector `H[T_s,T_s]` / `M^s` spectra.

**⚠ Σ_data reconciliation renormalizes γ.** `reconcile_sigma_data` subsets a pre-threshold
`Sigma_beta_gamma` to the surviving active set, but the loss uses *renormalized* γ (factor
`c_s ≈ sum_before/sum_after`), so subset γ rows are over-weighted by ~`c_s²` and the resulting
T SEs run ~`c_s` too tight. For exact inference regenerate the Σ file with the threshold applied.

---

## Input data

| File | Shape | Content |
|---|---|---|
| `[Sigma_aa_]Sigma_beta_gamma[_cloglog][_1][_f].npy` | `(N_REG + n_γ)²` | joint bootstrap covariance of β+γ moments, **β block first**. Selected by `sigma_beta_gamma_filename` on `REG_METHOD`, `N_REG==1`, entry point (`_f` = SMM) and γ level (`Sigma_aa_` **prefix** = attraction-area γ) |
| `w_beta.npy` / `w_gamma.npy` | `(N_REG)²` / `(n_γ)²` | block-diagonal fallback |
| `X_rs.npy` | `(S, R)` | drives `T_MASK` and the Σ reconciliation |
| `filter_N_upstream.npy` | `(S, R)` | binary: 1 = cell enters the optimisation, 0 = out. Supplier cell ⟺ `filter==1 & X_rs>0`; control cell ⟺ `filter==1 & X_rs==0` |
| `attraction_area_linkages.npy` | `(R, R_downstream)` | binary ZE → attraction-area incidence; must agree with `CLOSEST_DOWNSTREAM_REGION` |

---

## Key functions

**`build_step3_weight_matrix`** (`tools.jl`) — assembles `W_step3 = (Σ_data + Σ_sim)^{-1}` over
β+γ only. Σ_sim from `K` re-seeded `full_SMM` evaluations at `theta_hat_1`, resampled at
`N_RHO_INFERENCE` (decoupled from the optimization `N_rho`).

**`compute_smm_inference`** (`tools.jl`) — parameter SEs (efficient + sandwich), fitted-moment
SEs, Hansen J. Pass `block_ranges`/`block_names` when calling on the β+γ subsystem:

```julia
n_reg = length(BLOCK_RANGES[4]); n_gam = length(BLOCK_RANGES[5])
gb_block_ranges = (1:n_reg, (n_reg+1):(n_reg+n_gam))
gb_block_names  = ("reg_coef", "gamma_ls")
```

`display_labels`/`display_values` print the non-inferred head parameters (Ω, A) value-only.

**`compute_jacobian`** (`tools.jl`) — fixed-draw central FD, **one log step for every column**
(`θ·exp(±δ)`, `δ = step_rel = 1e-2`, chain-ruled back to raw units by `1/θ`). Stored column is
raw `∂m/∂θ`. `profile_T=true` routes every evaluation through `profiled_theta` so an α
perturbation moves `T*` with it — the returned column is the total derivative along the profiled
manifold, and no `∂m/∂T`-as-free-parameter is formed.

**`screen_T_identification`** (`tools.jl`) — print-only, runs after every Jacobian at θ̂_1 and
θ̂_2 in both entry points. Prints the global `H = J'WJ` screen, the T sub-block, the per-sector
`M^s` spectra, and mechanism attribution (which sector the weak direction sits in; whether it is
weak under `W=I` too). Never affects an estimate.

**`loss_function`** (`model_CP.jl`) — `err·W·err'` with optional `moment_indices`. A `W` already
sized to `length(moment_indices)` is used as-is; a full-size `W` is subset.

---

## Step-3 inference scope

Inference at θ̂_2 is computed on **β+γ only**, β then γ. The full-row Jacobian is still saved to
`jacobian_all_step3.npy` for diagnostics; only the inference slice is gb-restricted. The step-3
loss is likewise restricted via `moment_blocks=[4,5]` (blocks 4 and 5 are `reg_coef` and
`gamma_ls` — *not* 3 and 4).

Under `profile_T`, inference switches to the profiled path: the α CI comes from an α-reduced
**total** Jacobian (`compute_jacobian(...; profile_T=true)`, α columns only), and T gets a
correlated delta-method CI combining `∂T*/∂α`, `∂T*/∂γ`, and the data γ covariance.

---

## Usage

```bash
./run.sh aero --n_coef=4 --n_tau=1                  # power-law τ, over-identified on reg_coef
./run.sh aero --n_coef=4 --n_tau=1 --profile_T=true # Sinkhorn-profiled T
./run.sh aero --optimizer=tiktak --profile_T=true   # pso | cmaes | tiktak

julia main.jl aero 4 1                              # industry n_coef n_tau
julia main_gmm.jl aero 4 1 200                      # + n_quad
```

`run.sh` flags: `--n_coef`, `--n_tau`, `--draws=qmc|mc|is|sobol`, `--n_rho_inf`,
`--optimizer=pso|cmaes|tiktak`, `--profile_T`, `--reg=cloglog|lpm`, `--controls=true|false`.

Draw methods: `:qmc` (default, stratified uniform, flat weights) and `:sobol` (per-sector
digitally-shifted Sobol net) are unbiased for the min-coupled moments; `:is` is **biased** for
them (the CES price index applies only the winning column's weight, so the losing columns'
density ratios do not cancel) and is retained only for `reg_coef` tail resolution.
