# SMM Spatial Comovement

Structural calibration of a spatial production-network model, from the paper
*"Spatial Comovements."*

The model explains how a demand shock in one region spreads to others through
**supply chains**: downstream firms buy intermediate inputs from upstream suppliers
in different regions, and shared suppliers create *spatial comovement*. This
repository calibrates the model's parameters to sourcing and comovement data and
delivers standard errors and over-identification tests.

Two estimators are provided and share the same model, data, and moments:

- **SMM** (`main.jl`) — simulation-based, robust, the default path.
- **GMM** (`main_gmm.jl`) — closed-form analytical moments; faster, with exact
  standard errors.

> **New here?** Read [`documentation/model.md`](documentation/model.md) for the
> economics and the equations, and [`documentation/optimizer.md`](documentation/optimizer.md)
> for how the parameter search works. This README covers how to *run* the code.

---

## Project structure

```
SMM_Spatial_Comovement/
│
├── run.sh                  # ← launcher for a calibration (start here)
├── pkg.jl                  # one-off: installs all Julia packages
│
├── main.jl                 # SMM entry point (simulation-based)
├── main_gmm.jl             # GMM entry point (analytical, exact SEs)
│
├── model_CP.jl             # core model — simulated moments
├── model_analytical.jl     # core model — closed-form moments (GMM)
├── load_parameters.jl      # loads data, builds moments, masks, globals
├── optimizer.jl            # backend-neutral search: staging, bounds, loops
├── tools.jl                # inference, Jacobian, weight matrix, reporting
│
├── analysis.ipynb          # LEGACY post-run analysis (continuum ZE model)
│
│   # Python reporting for a --granular=true --ca_level=aa run. The notebooks are
│   # narrative plus run cells; every function lives in one of the four modules.
├── utils.py                # loader, theta+, Ricardian geometry, THE ECONOMY
├── report_lib.py           # moment fit, count curve, Jacobian, identification, untargeted
├── diffusion_lib.py        # amplification, local share, incidence, IO benchmark
├── granular_lib.py         # comparative advantage, concentration, portfolios, dispersion
├── model_report.ipynb          # the FIT of the model to its targets
├── tests_counterfactuals.ipynb # the tests and the counterfactuals
├── gates.ipynb                 # the test suite, run with its output in view
│
├── optimizers/             # interchangeable search engines (--optimizer=…)
│   ├── pso_integration.jl      #   Particle Swarm Optimization (default)
│   └── cmaes_integration.jl    #   CMA-ES
│
├── documentation/          # conceptual docs (start with model.md)
│   ├── model.md
│   └── optimizer.md
│
├── test/                   # validation & diagnostics (run by hand)
│   ├── run_test.sh
│   ├── run_internal_validity.jl
│   ├── run_untargeted_validation.jl
│   ├── test_extensive_margin.jl
│   └── test_t_reorder.jl
│
├── extras/                 # post-estimation analysis (amplification, Table 2)
│   ├── untargeted_moments.jl
│   ├── compute_amplification_weights.jl
│   └── run_amplification_analysis.jl
│
├── CLAUDE.md               # detailed developer/implementation reference + changelog
└── README.md               # this file
```

Each subfolder contains a short `files.md` describing its files.

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

**Input data.** Each industry needs a `baseline_<industry>/` folder (e.g.
`baseline_aero/`) holding the empirical inputs — distances, wages, sectoral shares,
empirical sourcing shares `emp_gamma_ls`, regional shares `emp_pi_r`, the bootstrap
covariance `Sigma_beta_gamma.npy`, and `stats.csv` (fixed parameters and the α
prior). `run.sh` checks that this folder exists before launching.

---

## How to run

Everything goes through **`run.sh`**, launched from the directory that contains both
this repo folder and the `baseline_<industry>/` data folders.

```bash
./run.sh <industry> [options]
```

### Simple example

```bash
./run.sh auto --n_tau=1 --n_coef=4 --mode=smm --optimizer=pso
```

This calibrates the **auto** industry by **SMM** using **PSO**, with a single
power-law trade-cost elasticity α (`--n_tau=1`) matched against four binned distance
moments (`--n_coef=4`, so the trade-cost block is over-identified).

### Options

| Option | Values | Default | Meaning |
|--------|--------|---------|---------|
| `<industry>` | `aero`, `auto`, `car`, `both` | — | which economy (`both` runs aero then auto) |
| `--n_coef` | `1`, `4`, `5` | `4` | number of distance-regression moments (`reg_coef`) |
| `--n_tau` | `1`, `4`, `5` | `= n_coef` | number of trade-cost parameters (`1` = power-law α; `4`/`5` = per-bin) |
| `--mode` | `smm`, `gmm` | `smm` | simulation-based vs. analytical estimator |
| `--optimizer` | `pso`, `cmaes` | `pso` | search engine (SMM path) |
| `--draws` | `sobol`, `qmc`, `mc`, `is` | `sobol` | Fréchet draw method (SMM only) |
| `--n_quad` | integer | `200` | Gauss–Legendre nodes for `reg_coef` (GMM only) |

Setting `--n_tau` below `--n_coef` (e.g. `--n_tau=1 --n_coef=4`) over-identifies the
trade-cost block and lets the Hansen J-test speak to model fit.

### More examples

```bash
./run.sh aero                                  # defaults: SMM, PSO, n_coef=4
./run.sh aero --n_coef=4 --optimizer=cmaes     # SMM with CMA-ES
./run.sh aero --n_coef=4 --n_tau=1 --mode=gmm  # analytical GMM, over-identified
./run.sh both --n_coef=4 --mode=smm            # aero then auto, sequentially
```

### Monitoring and stopping

`run.sh` launches Julia in the background and prints where the log is:

```bash
tail -f reporting_aero/logs.log     # SMM   → reporting_<industry>/
tail -f reporting_gmm_aero/logs.log # GMM   → reporting_gmm_<industry>/
pkill -f 'julia.*main'              # stop
```

---

## What it computes

Both entry points run a **three-step efficient estimator**:

1. **Step 1 — first fit.** Minimize the identity-weighted distance between model and
   data moments → θ̂₁. The trade-cost elasticity α is **not** searched: it is read from
   the extensive-margin distance regression (the `prior_alpha` in `stats.csv`). Given
   that α, the comparative-advantage scales T are initialized by inverting the observed
   sourcing shares (a closed-form fixed point). From this theory-based starting point a
   **global joint search** over all parameters runs — under PSO a joint fit plus
   block-coordinate refinement loops; under CMA-ES a single adaptive run. See
   [`documentation/optimizer.md`](documentation/optimizer.md).
2. **Step 2 — efficient weight matrix.** Build `W = (Σ_data + Σ_sim)^{-1}`, where
   `Σ_data` is the bootstrap moment covariance and `Σ_sim` the simulation covariance
   at θ̂₁ (GMM uses `Σ_data` only).
3. **Step 3 — efficient fit.** Re-optimize the trade-cost and comparative-advantage
   parameters (α, T) under `W`, warm-started at θ̂₁ → θ̂₂.
4. **Step 4 — inference.** Jacobian at θ̂₂, delta-method standard errors (efficient
   and sandwich), fitted-moment SEs, and the Hansen J over-identification test. The
   derivation and the reports are in
   [`documentation/inference.md`](documentation/inference.md).

Results are written under `reporting_<industry>/` (or `reporting_gmm_<industry>/`),
one folder per step, with parameter estimates, reports, dashboards, and the
`inference/` outputs. The exact output layout and file meanings are documented in
[`CLAUDE.md`](CLAUDE.md).

---

## The parameters

**Estimated** (full vector laid out as `[Ω^L | Ω^s | A | α | T]`):

| Symbol | Code name | Meaning |
|--------|-----------|---------|
| Ω^L | `agg_labor_share_tech` | labor share in production |
| Ω^s | `agg_industry_share_tech` | sectoral input shares |
| A_r | `productivity` | downstream productivity by region |
| α | `alpha` | trade-cost elasticity — one power-law coefficient (`n_tau=1`) or one per distance bin (`n_tau=4/5`) |
| T_{sr} | `T` | Fréchet scale — comparative advantage by sector-region |

**Fixed** (calibrated outside the loop, read from `stats.csv`): demand elasticity ε,
substitution elasticities λ, ν, ν_s, and the Fréchet shape θ.

> **Note on notation.** The only trade-cost parameter is α. The `reg_coef` distance
> coefficients (block 4 of the moments, sometimes written β in the extensive-margin
> regression) are **moments the model matches, not parameters it estimates**. The
> `--n_coef` moments and the `--n_tau` α-parameters are independent counts.

---

## Validation and analysis

- **Did the estimator recover the truth?** — `test/run_test.sh` (internal-validity
  Monte-Carlo).
- **Out-of-sample fit (Table 2 comovement)** — `test/run_untargeted_validation.jl`.
- **Amplification (Section 4.3)** — `extras/run_amplification_analysis.jl`.

See the `files.md` in `test/` and `extras/` for details.

---

## Further reading

- [`documentation/model.md`](documentation/model.md) — the model, in equations (incl. the per-sector Sobol draw scheme).
- [`documentation/optimizer.md`](documentation/optimizer.md) — PSO and CMA-ES.
- [`documentation/inference.md`](documentation/inference.md) — standard errors, the Jacobian, and the Hansen J-test.
- [`CLAUDE.md`](CLAUDE.md) — implementation reference, invariants, and changelog.
