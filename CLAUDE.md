# SMM Spatial Comovement — Codebase Guide

The Julia code estimates ONE specification: granular varieties (finite `N_s`),
comparative advantage at the attraction-area level with `T_ls = T_{a(l)s} · n_ls^κ`,
the cell-level cloglog extensive margin (block 4 = `[β_d ; β_κ]`), and T profiled out of
the search by a GE-Sinkhorn inversion. The previous multi-specification code (GMM,
continuum, ZE-level CA, LPM, firm-level regression, CMA-ES/TikTak, joint T search) is
frozen in `old_version/`, with its own `CLAUDE.md` and full changelog. Do not add
backward compatibility to the root code; restore from `old_version/` if needed.

The Python reporting (`utils.py`, `report_lib.py`, `diffusion_lib.py`, `granular_lib.py`,
the notebooks, `test/*.py`) is not part of this rewrite and must not be changed by Julia
work. Two contracts tie them together: `load_parameters.jl` must keep the line
`@everywhere const theta = $(…)` (parsed by `utils._theta_from_julia`), and the artefact
names/locations below must not move.

## Changelog

| Date | Files | Summary |
|------|-------|---------|
| 2026-10-28 | `inference.jl` (rewritten), `main.jl`, `model_CP.jl`, `profiling.jl`, `optimizer.jl`, `load_parameters.jl`, `pkg.jl`, `README.md`, `run.sh`, `CLAUDE.md` | **The data covariance is dropped: W and the inference both come from FAKE WORLDS drawn from the model.** **(1) Step 2, `world_weight_matrix`.** B = `WORLD_B_W` (1000) worlds at θ̂₁: one network of N̂_s varieties per sector with i.i.d. draws (`draw_world`, not the Sobol design), on which β_b (ONE cell-cloglog fit), G_b (realised empty-cell share), γ_b (area sourcing shares, Σ=1) and π_b are computed as on the data; T re-inverted on γ_b with A updated on π_b (`recalibrate_TA`), N̂ calibrated on G_b (`concentrate_N_s(...; target)`), β simulated at μ̂₁ on the world's own simulator (`world_simulator`: Sobol at N_rho randomised by the world seed + its own block-4 permutations, new `draw_cell_perms`); Ω̃ = Cov(e_b), W = Ω̃⁻¹ on block 4. Writes `step2/{Omega_world,W_world}.npy`, `world_residuals.npz`, `world_weight_report.txt` (with the mean residual and its t, i.e. the continuum-simulator bias against a finite world). **(2) Step 3**: `[α ; κ]` under W, loss on block 4 only (new `run_optimization(loss_blocks)`), γ and G being reproduced exactly by the inversion and the bisection. **(3) Step 4, `world_inference`.** B = `WORLD_B_INF` (500) worlds at θ̂₂, the whole chain in each with W fixed: T+A recalibration, N calibration, Nelder–Mead over (α, κ) from the estimate with common random numbers inside the world. Reports estimate/mean/bias/sd/2.5–97.5% quantiles for α, κ, N_s, T (levels and logs), A; the within-sector sd of log T; and closed-form counterfactuals per world (`incidence_stats`: mean and median-buyer sourcing distance, local share at 100/200 km, under estimate / equal T / τ ≡ 1, levels and changes). Writes `step3/inference_world/{world_inference_draws.npz (every world's θ), world_inference.txt, world_inference_labels.txt}`. Fig. 10's 95% share is not computed in Julia; the saved θ's let Python recompute it per world. **Choices stated in `inference.jl`'s header**: A is updated RELATIVE to the estimate, `A_b = Â·(c(T_b)/ĉ)·(π_b/π̂)^(−1/ε)` (analytical c, π), so a world equal to the model returns Â; Ω^L, Ω^s stay at the estimate; an area active in the data that wins nothing in a world gets T = 0 (`EMPTY_T`) and is reported as an empty share; a world with an empty reference area is excluded and counted. **Bug fixed in `invert_T_ge`**: it never converged with a zero-target column (the model γ sums to domestic_share, the target to 1, and the renormalisation removed that common factor from every column but the zero-target one, which drifted by domestic_share^δ per pass); such a column is now held relative to the reference and left out of the residual. Never fires on the data. **Removed**: `load_sigma_data`, `build_weight_matrix` (Σ_sim), `profiled_jacobian`, `run_inference` and its delta methods, `SIGMA_DATA_FILE`. `--n_rho_inf` / `N_RHO_INFERENCE` now have no reader (kept so calls parse). **Python consequence, not handled (Python untouched)**: `step2/{Sigma_data,Sigma_sim,Omega,W_step3,M_sim}.npy`, `jacobian_profiled*.npy` and `step{2,3}/inference/` are no longer written, so the reporting panels and `load_granular_data` parts that read them (SE bands, covariance figures, Jacobian) will not find them. **Verified** (Julia 1.10, synthetic 3-sector × 80-ZE × 6-area tree, n_coef = 1 and 4): a world equal to the model returns Â and T̂ to 1e-9; each world's criterion is deterministic; 30/30 and 40/40 worlds usable for Ω̃, 6/6 and 4/4 re-estimated, all report sections written; the same numbers as a run of the same code on `old_version/`. **Not run**: `main.jl` end to end (Plots/Parquet not installed here) and anything on the real data. |
| 2026-10-28 | all Julia files, `run.sh`, `pkg.jl`, `README.md`, `CLAUDE.md`, `old_version/`, `test/files.md` | **Light rewrite: the main specification only, and Steps 2–4 adapted to it.** The old Julia tree (13k lines) is moved to `old_version/` (same layout, launchers repointed); the root has ~2k lines in `main.jl`, `load_parameters.jl`, `model_CP.jl`, `profiling.jl`, `optimizer.jl` (PSO merged in), `reporting.jl`, `inference.jl`. `run.sh` keeps `--n_coef` (1/4/5), `--n_tau` (1 or `n_coef`), `--n_rho_inf`, `--relax_n_lo`; GMM, `--n_quad`, `--draws`, `--optimizer`, `--reg`, `--controls`, `--profile_T`, `--granular`, `--ca_level` are gone (fixed to the main spec). **Inference rebuilt for κ** (it had been switched off under GRANULAR): the profiled Jacobian now has `[α ; κ]` columns (log step for α, additive for κ) and γ-TARGET columns (T re-inverted, head fixed), which replace the full free-parameter Jacobian; T gets the correlated delta method over `(â, γ̂)`, N_s the implicit-function delta method over `(Ĝ, â, γ̂)`, and the Ḡ_s(K) bands the same substitution. Steps 2–4 run when `SIGMA_DATA_FILE` (`Sigma_aa_cell_cloglog[_1].npy`, β_d → β_κ → γ → G) is present with the right size, else the run stops after Step 1 and says why. **One behaviour change:** a cell-cloglog fit with a singular design (a regressor without variation in one network) is now a FAILED FIT (excluded, counted in `reg_fail`) instead of an error that made the whole evaluation `Inf`; without it Σ_sim crashed on i.i.d. draws. Non-finite losses are rejected (`Inf`). **Dropped diagnostics**: full free-parameter Jacobian (`jacobian_all*.npy`), naive FD block, Richardson/symmetry checks, `screen_T_identification`, 2×2 test, T-init gravity plot, post-hoc calibration-consistency table, untargeted Table 2. **Verified** (Julia 1.10, synthetic tree in the export format, `n_coef/n_tau` = 4/1, 1/1, 4/4): at a fixed θ the profiled θ (T*), moments, loss, N̂_s, q̂, Ḡ_s(K), block-4 diagnostics, draws, masks and T init are BIT-IDENTICAL to `old_version/`; a small-budget Step 1 gives BIT-IDENTICAL stage parameters and `best_simulated_moments.npy` (7 stages); Steps 1–4 + post-hoc run end to end; the K = 0 band equals the bootstrap SE of the target (6e-15) and V_T, V_N are PSD; Python gates pass except `test_local_share_dispersion.py` gate 9 (pre-existing). **Not run on the real data, and the cell-level Σ file does not exist yet.** |

## Files

| File | Role |
|------|------|
| `run.sh` | launcher (run from the repo's parent dir, next to `baseline_<ind>/`) |
| `main.jl` | Steps 1–4 + post-hoc; flags `run_step1..4`, `K_loops`, `K_sim`, `K_jac` at the top |
| `load_parameters.jl` | data → `@everywhere const` globals, moment vector, masks, T init |
| `model_CP.jl` | draws, `unpack_params`, `solve_network`, cell cloglog (block 4), N̂_s bisection, `compute_moments`, `full_SMM` |
| `profiling.jl` | closed-form EK prices, `invert_T_ge` (Sinkhorn), `profiled_theta_full` |
| `optimizer.jl` | PSO, `train_stage`, `run_optimization` (stage 0 + block-coordinate loops) |
| `reporting.jl` | stage dashboard/report, `run_reporting`, `report_granular`, `write_post_hoc` |
| `inference.jl` | fake worlds: `world_weight_matrix` (Step 2), `world_inference` (Step 4) |

## Model objects

Parameter vector (levels): `[Ω^L(1) | Ω^s(S) | A(R_downstream) | α(N_TAU) | κ(1) | T(sum(T_MASK))]`.
Ω^s is normalised to sum to 1, A to A[1] = 1, T per sector to `T[s, ref] = 1`.
T is never searched: the stage objective replaces the T block with `T*(α, κ, Ω, A)`; a
particle whose inversion does not converge gets `Inf` (never scored at a fabricated T).

Moment vector (masked): `[labor | industry (S−1) | π_r (R_d−1) | β_d (N_REG) ; β_κ | γ_{s,a} (active, non-ref) | Ḡ_s(0) (S)]`,
`BLOCK_RANGES` a 6-tuple. Assemble with `moments_to_vec`, never by hand.

Two index spaces: CELLS (`CELL_MASK`, `n_good`, `GOOD_S/R`, draw columns; control cells
included) and T COLUMNS = attraction areas (`T_COL_DIM = n_AA`, `T_MASK`, `SECTOR_T_COLS`,
`EMP_GAMMA_T`). `T[s,l] = T_par[s, AA_OF_ZE[l]] · n_ls^κ` (`gather_T_to_ze`).

### Invariant: s-major T flattening
The reduced T block and the γ moment block enumerate (sector, area) pairs in the same
order — all areas of sector 1, then sector 2 — so the T[s,a] parameter ↔ γ[s,a] moment.
`T_MASK = vec(permutedims(T_ACTIVE))`; flat position `(s-1)·T_COL_DIM + a`.
`T_ID_REDUCED` drops each sector's reference area and aligns with the γ moments.

### Weighting and inference: fake worlds
W lives on block 4 (`[β_d ; β_κ]`, `BLOCK_RANGES[4]` order) only; γ and G are reproduced
exactly by the T inversion and the N bisection. No data covariance is read.

## Steps and artefacts (`reporting_<ind>_profiled_aa_gran_kappa[_nlo1]_pso/`)

- `step1/`: stages `0/`, `epoch_k/<stage>/` (best_params, dashboard, report.txt),
  `theta_hat_1.npy`, `best_simulated_moments.npy`, `granular_diagnostics.npz`, post-hoc files.
- `step2/`: `Omega_world.npy`, `W_world.npy`, `world_residuals.npz`, `world_weight_report.txt`,
  `granular_diagnostics.npz`.
- `step3/`: stages, `theta_hat_2.npy`, `inference_world/` (draws npz, report, labels), post-hoc files.
- post-hoc (`write_post_hoc`): `suppliers.parquet` (50 i.i.d. realisations of the
  N̂_s-variety economy), `suppliers_continuum.parquet`, `post_hoc_{u,good_sr,N_hat}.npy`,
  `w_srd_r.npy`, `suppliers.npy` — read by Python's `check_against_julia`.

## Caveats

- Read T identification per sector: the T block of `J'WJ` is block-diagonal by sector,
  and the global λ_min mixes sectors through a units/weighting artefact.
- Block 4 averages β̂ over `R_CELL_REG = 100` nested networks of N̂_s varieties; failed
  fits are excluded and counted (`reg_fail`); all failed ⇒ NaN ⇒ the particle is rejected.
- `N̂_s` is recalibrated in every fake world, so its integer jumps are part of the
  bootstrap distribution, not a differentiation problem.
- `--relax_n_lo=true` drops the model's lower bound on N_s: a diagnostic, not the spec.
