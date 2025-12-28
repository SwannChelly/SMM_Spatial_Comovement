# SMM Spatial Comovement

Simulated Method of Moments (SMM) calibration of a spatial production network model, as described in the paper *"Spatial Comovements"*.

This code calibrates a structural model of supply chain linkages to study how demand shocks propagate across regions through input-output networks.

---

## Overview

The model features:
- **Nested CES production**: Downstream firms source intermediate inputs from upstream suppliers across sectors and regions
- **Ricardian supplier selection**: Firms choose lowest-cost suppliers given productivities and trade costs
- **Fréchet productivity draws**: Upstream firm productivities follow a Fréchet distribution
- **Distance-based trade costs**: Iceberg costs increase with geographic distance between regions

The calibration matches:
1. Aggregate labor share
2. Sectoral input shares
3. Sourcing shares by region-sector (γ_ls)
4. Regression coefficients (supplier probability vs. distance)
5. Regional employment shares (π_r)

---

## Project Structure

```
SMM_Spatial_Comovement/
├── main_pso.jl                    # Main entry point (PSO optimization)
├── model_CP.jl                    # Core structural model
├── pso_integration.jl             # Particle Swarm Optimization implementation
├── tools.jl                       # Helper functions and utilities
├── untargeted_moments.jl          # Table 2 regression validation
├── run_untargeted_validation.jl   # Interface for validation
├── compute_amplification_weights.jl  # Section 4.3 amplification analysis
├── run_amplification_analysis.jl  # Full amplification pipeline
├── pkg.jl                         # Package installation script
└── run.sh                         # Shell script to launch calibration
```

---

## Requirements

### Julia Packages

Run `pkg.jl` to install all required packages:

```julia
julia pkg.jl
```

Or install manually:
```julia
using Pkg
Pkg.add(["NPZ", "Distributions", "DataFrames", "CSV", "Plots", "StatsPlots",
         "FixedEffectModels", "RDatasets", "CategoricalArrays", "Optim",
         "QuasiMonteCarlo", "HaltonSequences", "ProgressMeter", 
         "SharedArrays", "StatsBase"])
```

### Python (for plotting)
```bash
pip install -r requirements_py.txt
```

---

## Input Data

The model requires a `baseline_{industry}/` folder containing:

| File | Description |
|------|-------------|
| `stats.csv` | Model parameters (ε, labor share) |
| `distances.npy` | R×R matrix of distances between regions |
| `w_rs.npy` | S×R matrix of sectoral wage shares |
| `regional_wages.npy` | R-vector of regional wages |
| `N_downstream_per_region.npy` | Number of downstream firms per region |
| `filter_N_upstream.npy` | S×R filter for upstream presence |
| `input_share.npy` | Sectoral input shares |
| `domestic_share.npy` | Domestic sourcing shares by sector |
| `N_rs.npy` | Number of upstream firms per region-sector |
| `emp_gamma_ls.npy` | Empirical sourcing shares |
| `emp_pi_r.npy` | Empirical regional employment shares |
| `reg_coef.npy` | Empirical regression coefficients |

---

## Usage

### Quick Start

```bash
# Run calibration for aerospace industry
./run.sh aero

# Or for automotive
./run.sh car
```

### Manual Execution

```bash
# Launch with logging
nohup julia main_pso.jl aero > reporting_aero/logs.log 2>&1 &

# Monitor progress
tail -f reporting_aero/logs.log
```

### From Julia REPL

```julia
include("main_pso.jl")
```

---

## Calibration Process

The optimization uses a multi-stage Particle Swarm Optimization (PSO) approach:

### Stage 0: Initial Search
- Grid search over β (trade cost) parameters
- Find reasonable starting values matching regression coefficients

### Stage 1: Full PSO
- Optimize all parameters jointly
- 200 iterations with all available CPU cores

### Refinement Loops (×10)
Each loop contains 3 stages targeting parameters by sensitivity:

1. **Productivity (A_r)** — High sensitivity (ε ≈ -16 amplifies errors)
2. **Spatial Structure (β, T)** — Medium sensitivity
3. **Technical Coefficients (Ω^L, Ω^s)** — Low sensitivity (damped by λ, ν)

---

## Outputs

Results are saved to `reporting_{industry}/`:

```
reporting_aero/
├── 0/                          # Initial stage results
│   ├── best_params.npy
│   ├── pso_history.npy
│   ├── dashboard.png
│   └── report.txt
├── epoch_1/                    # Refinement loop 1
│   ├── 1/, 2/, 3/             # Stages within loop
├── epoch_2/
│   └── ...
├── best_simulated_moments.npy
├── best_parameters_list.npy
├── empirical_moments.npy
├── pso_convergence.png
└── logs.log
```

### Key Output Files

| File | Description |
|------|-------------|
| `best_params.npy` | Calibrated parameter vector |
| `report.txt` | Comparison of empirical vs simulated moments |
| `dashboard.png` | Distribution plots (γ_ls, π_r, etc.) |
| `pi_r.npy` | Simulated regional employment shares |
| `productivity.npy` | Calibrated productivity parameters |
| `w_srd_r.npy` | Trade flow weights for amplification |

---

## Model Validation

### Untargeted Moments (Table 2)

After calibration, validate against Table 2 regression:

```julia
include("run_untargeted_validation.jl")

# Load best parameters
best_params = NPZ.npzread("reporting_aero/epoch_100/300/best_params.npy")

# Run validation
results = validate_table2(best_params, "aero", T_periods=36)
```

This simulates demand shocks through the calibrated network and estimates:
```
d ln x_{i,t} = α_i + β × d ln x_{s,t} + ε_{i,t}
```

**Empirical targets**: Aerospace β = 0.112, Automotive β = 0.161

### Amplification Analysis (Section 4.3)

```julia
include("run_amplification_analysis.jl")
run_amplification_analysis("aero")
```

Computes:
- w_d^{sr'}: Share of sales to downstream
- w_{r'}^{sr'd}: Share of downstream sales to local region
- Amplification coefficients A_{r'}

---

## Key Parameters

| Symbol | Variable | Description |
|--------|----------|-------------|
| β | `beta` | Distance bin trade costs |
| Ω^L | `agg_labor_share_tech` | Labor share in production |
| Ω^s | `agg_industry_share_tech` | Sectoral input shares |
| A_r | `productivity` | Downstream productivity by region |
| T_{sr} | `T` | Sector x Region comparative advantage |

### Fixed Parameters

| Symbol | Value | Description |
|--------|-------|-------------|
| λ | 0.5 | Labor-intermediate substitution |
| ν | 0.2 | Cross-sector substitution |
| ν_s | 2.5 | Within-sector substitution |
| θ | 1.768 | Fréchet shape (productivity dispersion) |

---

## Parallelization

The code automatically uses all available CPU cores:

```julia
available = Sys.CPU_THREADS - nprocs()
addprocs(max(available-1, 0))
```

PSO particles are evaluated in parallel using `pmap()`.

---

## Troubleshooting

### Kill Stuck Processes
```bash
ps aux | grep '[j]ulia' | awk '{print $2}' | xargs kill -9
```

### Memory Issues
Reduce `N_rho` (number of varieties per sector) or `N_PARTICLES` in `main_pso.jl`.

### Convergence Issues
- Increase `MAX_ITER_INITIAL` or `MAX_ITER_STAGE`
- Adjust `alpha` bounds (tighter = more focused search)
- Check that empirical moments are correctly loaded

---

## Citation


---

## License

[Add license information]

---

## Contact

[Add contact information]