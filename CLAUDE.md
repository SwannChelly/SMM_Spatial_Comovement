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
| `inference.jl` | Σ_data check, Σ_sim/W, profiled Jacobian, `run_inference` |

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

### Invariant: β → γ → G
Every inference object (`SIGMA_DATA_FILE`, `W_step3`, `Ω`, Jacobian rows) is ordered
`BLOCK_RANGES[4]` (β_d then β_κ), `[5]` (γ), `[6]` (G): `inference_moment_indices()`.

## Steps and artefacts (`reporting_<ind>_profiled_aa_gran_kappa[_nlo1]_pso/`)

- `step1/`: stages `0/`, `epoch_k/<stage>/` (best_params, dashboard, report.txt),
  `theta_hat_1.npy`, `best_simulated_moments.npy`, `granular_diagnostics.npz`, post-hoc files.
- `step2/`: `Sigma_data/Sigma_sim/Omega/W_step3/M_sim.npy`, `jacobian_profiled*.npy`,
  `inference/` at θ̂_1, `granular_diagnostics.npz`.
- `step3/`: stages, `theta_hat_2.npy`, `inference/` at θ̂_2, post-hoc files.
- post-hoc (`write_post_hoc`): `suppliers.parquet` (50 i.i.d. realisations of the
  N̂_s-variety economy), `suppliers_continuum.parquet`, `post_hoc_{u,good_sr,N_hat}.npy`,
  `w_srd_r.npy`, `suppliers.npy` — read by Python's `check_against_julia`.

## Caveats

- Read T identification per sector: the T block of `J'WJ` is block-diagonal by sector,
  and the global λ_min mixes sectors through a units/weighting artefact.
- Block 4 averages β̂ over `R_CELL_REG = 100` nested networks of N̂_s varieties; failed
  fits are excluded and counted (`reg_fail`); all failed ⇒ NaN ⇒ the particle is rejected.
- `N̂_s` is a step function of θ: the inference Jacobian pins it at the median over 5
  draw sets and prints its dispersion (simulation noise in q̂ — raise `--n_rho_inf`).
- `--relax_n_lo=true` drops the model's lower bound on N_s: a diagnostic, not the spec.
