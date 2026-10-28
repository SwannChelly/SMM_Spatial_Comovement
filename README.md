# SMM Spatial Comovement

Structural calibration of a spatial production-network model, from the paper
*"Spatial Comovements."*

The model explains how a demand shock in one region spreads to others through
**supply chains**: downstream firms buy intermediate inputs from upstream suppliers
in different regions, and shared suppliers create *spatial comovement*. This
repository calibrates the model's parameters to sourcing and comovement data and
delivers standard errors and over-identification tests.

The estimator is a three-step efficient SMM (`main.jl`). The previous
multi-specification code, including the analytical GMM, is kept in `old_version/`.

> **New here?** Read [`documentation/model.md`](documentation/model.md) for the
> economics and the equations, and [`documentation/optimizer.md`](documentation/optimizer.md)
> for how the parameter search works. This README covers how to *run* the code.

---

## Project structure

```
SMM_Spatial_Comovement/
├── run.sh               # launcher (start here)
├── pkg.jl               # one-off: installs the Julia packages
├── main.jl              # Steps 1–4 + post-hoc
├── load_parameters.jl   # data → constants, moment vector, masks, T starting value
├── model_CP.jl          # simulated model, cell-level cloglog (block 4), N_s bisection, moments
├── profiling.jl         # closed-form prices, GE-Sinkhorn inversion T*(α, κ, Ω, A)
├── optimizer.jl         # PSO and the staged block-coordinate search
├── reporting.jl         # stage reports, run summaries, granular diagnostics, post-hoc economy
├── inference.jl         # fake worlds: W from Cov(β_world − β_sim), parametric-bootstrap inference
│
├── utils.py, report_lib.py, diffusion_lib.py, granular_lib.py   # Python reporting
├── model_report.ipynb, tests_counterfactuals.ipynb, gates.ipynb
├── analysis.ipynb       # LEGACY notebook (continuum ZE model)
│
├── test/                # Python gates for the reporting (Julia tests are in old_version/test/)
├── documentation/       # conceptual docs (written for the multi-specification code)
├── old_version/         # the previous multi-specification Julia code, frozen
└── CLAUDE.md            # implementation reference, invariants, changelog
```

The Julia code estimates one specification: finite varieties `N_s` per sector,
comparative advantage `T_ls = T_{a(l)s} · n_ls^κ` at the attraction-area level, the
cell-level cloglog extensive margin, and T profiled out of the search. GMM, the continuum
and ZE-level models, other optimizers and the diagnostic scripts live in `old_version/`.

---

## Requirements

**Julia packages** (one-off):

```bash
julia pkg.jl
```

**Python** (for the reporting notebooks): each notebook installs its own dependencies
in its first cell — just open and run it, no separate setup needed.

Two notebooks report a granular / attraction-area run, and they split by what they
answer. `model_report.ipynb` displays the FIT of the model to its targets — the moment
table, the extensive margin and the zero-supplier share with their standard errors, the
count curve, the Jacobian, identification, and the untargeted distance elasticity.
`tests_counterfactuals.ipynb` interrogates the model one test at a time and runs the
counterfactuals. Set `MU = 1` or `MU = 2` in the Constants cell to read the step-1 or
the step-3 estimate.

Neither notebook holds a function definition. Everything they call is in `utils.py`,
`report_lib.py`, `diffusion_lib.py` or `granular_lib.py`, so a section is edited in a
file and the gates under `test/` import the same code rather than executing the
notebook's JSON.

**Every economy is simulated in Python** from the extended parameter set
`theta+ = (Omega_L, Omega_s, A, alpha, T, N)` — the estimated one and both
counterfactuals, by one forward map (`utils.simulate_economy`). `suppliers.parquet` is
not read by the reporting: it is the input to `utils.check_against_julia`, which compares
the Python port against `solve_network` to the bit given Julia's own draws. That is the
only exact check on the port, so run it after any change to either implementation:

```python
import utils
d = utils.load_granular_data("aero", mu=2, base="..", profile_T=True, ca_level="aa",
                             granular=True, relax_n_lo=True, optimizer="pso",
                             parts=("core", "firm"))
utils.check_against_julia(d)          # needs suppliers.parquet AND post_hoc_u.npy
```

**The gates.** `gates.ipynb` runs the whole suite one file at a time with its verdict in
view, and then renders the objects the gates reason about — the planted economy, the
identities the forward map closes, the three regimes and the channel decomposition — so a
number in an assertion can be read against the thing it constrains. The `.py` files under
`test/` remain the source of truth; the notebook runs them in a subprocess rather than
restating them, so a gate lives in exactly one place. 259 numbered gates across 10 files,
all against synthetic fixtures — no run tree, no Julia, no real data.

**Input data.** Each industry needs `baseline_<industry>/` (next to the repository)
with `stats.csv` (`epsilon`, `labor_share`), `distances.npy`, `filter_N_upstream.{npy,csv}`,
`w_rs.npy`, `regional_wages.npy`, `N_downstream_per_region.npy`, `input_share.npy`,
`domestic_share.npy`, `emp_gamma_ls.npy`, `X_dr.csv`, `attraction_area_linkages.npy`,
`G_K.csv`, `cells_ls.csv`, `beta_d_<n>.npy`, `beta_kappa_<n>.npy` (and `beta_d_1.npy`,
the α prior). No data covariance is needed: Steps 2–4 draw their uncertainty from fake
worlds simulated by the model.

---

## How to run

From the directory containing the repository and the `baseline_<industry>/` folders:

```bash
./SMM_Spatial_Comovement/run.sh aero                                   # n_coef=4, n_tau=1
./SMM_Spatial_Comovement/run.sh aero --n_coef=1 --n_rho_inf=20000
./SMM_Spatial_Comovement/run.sh aero --relax_n_lo=true                 # diagnostic
./SMM_Spatial_Comovement/run.sh both
```

| Option | Values | Default | Meaning |
|--------|--------|---------|---------|
| `<industry>` | `aero`, `auto`, `car`, `both` | — | economy (`both` runs aero then auto) |
| `--n_coef` | `1`, `4`, `5` | `4` | distance coefficients of the cell cloglog (`1` = ln d, else bins) |
| `--n_tau` | `1` or `n_coef` | `1` | trade-cost parameters (`1` = τ = d^α; else one α per bin) |
| `--n_rho_inf` | integer | `10000` | unused (inference now from fake worlds, at the estimation's `N_rho`); kept so old calls parse |
| `--relax_n_lo` | `true`, `false` | `false` | DIAGNOSTIC: N_LO forced to 1 (drops the model's bound) |

Outputs go to `reporting_<industry>_profiled_aa_gran_kappa[_nlo1]_pso/` (log in `logs.log`;
stop with `pkill -f 'julia.*main'`). Worker count: `N_WORKERS` (default 50), heap per
worker: `WORKER_HEAP_HINT` (default 2G).

## What it computes

1. **Step 1** — identity-weighted PSO over `[Ω^L, Ω^s, A, α, κ]`, T = T* by Sinkhorn
   inversion of the observed area-level sourcing shares, N_s by integer bisection on the
   zero-supplier share; a joint fit then block-coordinate loops → θ̂₁ (`step1/`).
2. **Step 2** — `WORLD_B_W` (1000) fake worlds at θ̂₁: one network of N̂_s varieties with
   i.i.d. draws, its own β, γ, G, π; T re-inverted with A updated, N calibrated, β
   simulated with the world's seed; `W = Cov(β_world − β_sim)^{-1}` on block 4 (`step2/`).
   Then the Step-4 inference at θ̂₁ with Step 1's weighting W = I (`step2/inference_world/`).
3. **Step 3** — `[α ; κ]` re-estimated under W on block 4, the rest fixed at θ̂₁ → θ̂₂ (`step3/`).
4. **Step 4** — `WORLD_B_INF` (500) fake worlds at θ̂₂, the whole chain (T, A, N, local
   search over α, κ) re-run in each (`step3/inference_world/`): sd, 2.5/97.5% quantiles and
   bias for every parameter, the dispersion of log T, and closed-form counterfactuals (mean sourcing distance, local share at 100/200 km
   under the estimate, equal T, no trade cost).

Post-hoc: the firm-level economy at θ̂₁ and θ̂₂ (`suppliers.parquet` etc.), the input of
the Python `check_against_julia`.

---

## Further reading

- [`documentation/model.md`](documentation/model.md), [`optimizer.md`](documentation/optimizer.md), [`inference.md`](documentation/inference.md) — written for the multi-specification code (`old_version/`); the model and the PSO are unchanged, the inference is described in `CLAUDE.md`.
- [`CLAUDE.md`](CLAUDE.md) — implementation reference, invariants, and changelog.
