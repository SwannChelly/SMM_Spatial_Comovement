##### Untargeted Moments: Reproduce Table 2 Regression #####
# Author: Swann Chelly
# Purpose: Validate calibrated model by reproducing Table 2 regression
#
# This file uses solve_network() from model_CP.jl and simulates
# productivity shocks to generate time series variation for regression analysis.
#
# ═══════════════════════════════════════════════════════════════════════════════
# NEW SIMULATION SETUP
# ═══════════════════════════════════════════════════════════════════════════════
#
# Downstream productivity shocks:
#   A_{drt} = A_{dr} × exp(z_{dr,t})
#   z_{dr,t} follows AR(1) with parameters (ρ_d, σ_d)
#   Shocks are i.i.d. across regions but AR(1) over time
#
# Under sticky prices:
#   d ln x_{d,t} = Σ_r w_r^d × d ln x_{dr,t}  (aggregate downstream)
#   d ln x_{dr,t} = z_{dr,t}                   (regional downstream)
#   where w_r^d = share of downstream firm in r in total downstream sales
#
# Supplier sales evolution:
#   d ln x_{i,t} = a_{di}^D × Σ_r a_{rdi}^D × d ln x_{drt} + (1 - a_{di}^D) × d ln x_{oi,t}
#
# Where:
#   a_{rdi}^D = share of supplier i's sales to downstream r in total sales to downstream
#             (computed from calibrated model: firm_expenditure_shares)
#   a_{di}^D  = share of supplier i's total sales going to downstream industry
#             (drawn from empirical distribution by sector from CSV file)
#   d ln x_{oi,t} = sales growth of "other customer" (i.i.d. with variance σ_{sr'})
#
# Each supplier (ρ, s, l) has a different "other customer" with i.i.d. shocks.
#
# ═══════════════════════════════════════════════════════════════════════════════

using Distributed
using Distributions
using Random
using NPZ
using LinearAlgebra
using DataFrames
using CSV
using FixedEffectModels
using CategoricalArrays
using Statistics


"""
    SimulationConfig

Configuration for the untargeted moments simulation.
"""
struct SimulationConfig
    T_periods::Int          # Number of time periods (quarters)
    sigma_d::Float64        # Std of downstream AR(1) shocks
    rho_d::Float64          # AR(1) persistence for downstream
    seed::Int               # Random seed
end

# Default configuration: 36 quarters (9 years)
const DEFAULT_CONFIG = SimulationConfig(36, 0.05, -0.15, 42)


"""
    load_exposure_distribution(input_folder)

Load the empirical distribution of a_{di}^D (share of sales to downstream industry)
from CSV file share_dist.csv with columns: A129 (sector as string), PartCa (exposure share).

The sectors (A129) are sorted alphabetically and assigned integer indices (1, 2, 3, ...).

Returns a Dict: sector_index (Int) => Vector of exposure values
"""
function load_exposure_distribution(input_folder)
    csv_path = joinpath(input_folder, "share_dist.csv")
    
    if !isfile(csv_path)
        @warn "Exposure distribution file not found: $csv_path. Using uniform [0.1, 0.9]."
        return nothing
    end
    
    df = CSV.read(csv_path, DataFrame)
    
    # Ensure columns exist
    if !("A129" in names(df)) || !("PartCa" in names(df))
        error("CSV must have columns 'A129' (sector) and 'PartCa' (exposure)")
    end
    
    # Convert A129 to string if not already
    df.A129 = string.(df.A129)
    
    # Sort by A129 (alphabetically)
    sort!(df, :A129)
    
    # Get unique sectors and create mapping to integers
    unique_sectors = sort(unique(df.A129))
    sector_to_int = Dict(s => i for (i, s) in enumerate(unique_sectors))
    
    println("  Sector mapping (A129 -> integer index):")
    for (s, i) in sort(collect(sector_to_int), by=x->x[2])
        println("    $s -> $i")
    end
    
    # Build dictionary: sector_index => vector of exposures
    exposure_by_sector = Dict{Int, Vector{Float64}}()
    
    for row in eachrow(df)
        sector_idx = sector_to_int[row.A129]
        exposure = Float64(row.PartCa)
        
        if !haskey(exposure_by_sector, sector_idx)
            exposure_by_sector[sector_idx] = Float64[]
        end
        push!(exposure_by_sector[sector_idx], exposure)
    end
    
    println("  Loaded exposure distribution for $(length(exposure_by_sector)) sectors")
    for (s, vals) in sort(collect(exposure_by_sector))
        println("    Sector $s: n=$(length(vals)), mean=$(round(mean(vals), digits=3)), " *
                "std=$(round(std(vals), digits=3))")
    end
    
    return exposure_by_sector
end


"""
    draw_exposures(exposure_by_sector, S_local, R_local, N_rho_local; seed=42)

Draw a_{di}^D for each supplier from the empirical distribution of their sector.

Returns: Matrix (N_rho × S × R) of exposure values a_{di}^D ∈ [0, 1]
"""
function draw_exposures(exposure_by_sector, S_local, R_local, N_rho_local; seed=42)
    
    Random.seed!(seed)
    
    a_d_D = zeros(N_rho_local, S_local, R_local)
    
    for s in 1:S_local
        if exposure_by_sector !== nothing && haskey(exposure_by_sector, s)
            # Draw from empirical distribution
            emp_dist = exposure_by_sector[s]
            for l in 1:R_local
                for rho in 1:N_rho_local
                    a_d_D[rho, s, l] = rand(emp_dist)
                end
            end
        else
            # Fallback: uniform distribution
            for l in 1:R_local
                for rho in 1:N_rho_local
                    a_d_D[rho, s, l] = rand() * 0.8 + 0.1  # Uniform [0.1, 0.9]
                end
            end
        end
    end
    
    return a_d_D
end


"""
    compute_a_rdi_D(network)

Compute a_{rdi}^D: share of supplier i's sales to downstream region r 
in total sales to downstream industry.

From model_CP.jl:
  firm_expenditure_shares[ρ, s, l, r] = share of downstream r's total cost going to (ρ,s,l)
  Total sales of (ρ,s,l) to r = firm_expenditure_shares[ρ,s,l,r] × total_cost_r
  
  a_{rdi}^D = sales_to_r / Σ_{r'} sales_to_{r'}

Returns: Array (N_rho × S × R_upstream × R_downstream) of shares
         Sum over r_downstream = 1 for each supplier
"""
function compute_a_rdi_D(network)
    
    exp_shares = network.firm_expenditure_shares  # (N_rho, S, l, r)
    Y_r = network.Y_r                              # Baseline downstream sales
    mu = network.mu                                # Inverse markup
    
    N_rho_local, S_local, R_local, _ = size(exp_shares)
    
    # Compute total cost by region: total_cost_r = μ × Y_r
    total_cost = mu .* Y_r
    
    # Sales from supplier (ρ,s,l) to downstream r
    # sales[ρ,s,l,r] = exp_shares[ρ,s,l,r] × total_cost[r]
    sales_to_downstream = zeros(N_rho_local, S_local, R_local, R_local)
    
    for r in 1:R_local
        if total_cost[r] > 1e-10
            sales_to_downstream[:, :, :, r] = exp_shares[:, :, :, r] .* total_cost[r]
        end
    end
    
    # Total sales of supplier (ρ,s,l) to entire downstream industry
    total_sales_to_downstream = sum(sales_to_downstream, dims=4)  # (N_rho, S, R, 1)
    
    # Compute a_{rdi}^D = sales_to_r / total_sales_to_downstream
    a_rdi_D = zeros(N_rho_local, S_local, R_local, R_local)
    
    for l in 1:R_local      # Upstream region
        for s in 1:S_local
            for rho in 1:N_rho_local
                total = total_sales_to_downstream[rho, s, l, 1]
                if total > 1e-12
                    for r in 1:R_local  # Downstream region
                        a_rdi_D[rho, s, l, r] = sales_to_downstream[rho, s, l, r] / total
                    end
                end
            end
        end
    end
    
    return a_rdi_D, sales_to_downstream, total_sales_to_downstream[:,:,:,1]
end


"""
    generate_downstream_shocks(R_local, T, config)

Generate regional downstream productivity shocks z_{dr,t}.

Shock structure:
- ACROSS REGIONS: i.i.d. at each time t from same distribution
- OVER TIME: AR(1) process per region: z_{dr,t} = ρ × z_{dr,t-1} + ε_{r,t}

Returns: Matrix (R × T) of log shocks z_{dr,t}
         d ln x_{dr,t} = z_{dr,t} (under sticky prices)
"""
function generate_downstream_shocks(R_local, T, config::SimulationConfig)
    
    Random.seed!(config.seed)
    
    # Innovation std (ensures unconditional std of z equals sigma_d)
    innovation_std = config.sigma_d * sqrt(1 - config.rho_d^2)
    
    # Draw ALL innovations at once: i.i.d. across regions and time
    innovations = randn(R_local, T) * innovation_std
    
    # AR(1) process in logs
    z = zeros(R_local, T)
    z[:, 1] = randn(R_local) * config.sigma_d  # Initial from unconditional distribution
    
    for t in 2:T
        z[:, t] = config.rho_d * z[:, t-1] + innovations[:, t]
    end
    
    return z
end


"""
    generate_other_customer_shocks(N_rho_local, S_local, R_local, T, sigma_sr; seed=42)

Generate i.i.d. shocks for "other customers" of each supplier.

Each supplier (ρ, s, l) has a different other customer whose sales are i.i.d.
d ln x_{oi,t} ~ N(0, σ_{sl}²)

where σ_{sl} is the sector-region specific standard deviation loaded from sigma_sr.csv

# Arguments
- `N_rho_local`: Number of varieties per sector-region
- `S_local`: Number of sectors
- `R_local`: Number of regions
- `T`: Number of time periods
- `sigma_sr`: Matrix (S × R) of standard deviations, or nothing for default
- `seed`: Random seed (default 42)

Returns: Array (N_rho × S × R × T) of i.i.d. shocks
"""
function generate_other_customer_shocks(N_rho_local, S_local, R_local, T, sigma_sr; seed=42)
    
    Random.seed!(seed + 1000)  # Different seed from downstream shocks
    
    shocks = zeros(N_rho_local, S_local, R_local, T)
    
    if sigma_sr === nothing
        # Use default standard deviation
        default_sigma = 0.17
        shocks = randn(N_rho_local, S_local, R_local, T) * default_sigma
    else
        # Use sector-region specific standard deviations
        for s in 1:S_local
            for l in 1:R_local
                sigma = sigma_sr[s, l]
                shocks[:, s, l, :] = randn(N_rho_local, T) * sigma
            end
        end
    end
    
    return shocks
end


"""
    simulate_supplier_sales(network, a_d_D, downstream_shocks, other_shocks, config)

Simulate supplier sales growth using the new model:

d ln x_{i,t} = a_{di}^D × Σ_r a_{rdi}^D × d ln x_{dr,t} + (1 - a_{di}^D) × d ln x_{oi,t}

# Arguments
- `network`: Output from solve_network(params, return_firm_level=true)
- `a_d_D`: Exposure to downstream industry (N_rho × S × R)
- `downstream_shocks`: Regional downstream shocks z_{dr,t} (R × T)
- `other_shocks`: Other customer shocks (N_rho × S × R × T)
- `config`: Simulation configuration

# Returns
- `d_ln_x_it`: Supplier sales growth (N_rho × S × R × T-1)
- `d_ln_x_dt`: Aggregate downstream growth (T-1,)
- `d_ln_x_drt`: Regional downstream growth (R × T-1)
- `weighted_exposure_it`: Σ_r a_{rdi}^D × d ln x_{dr,t} for each supplier (N_rho × S × R × T-1)
"""
function simulate_supplier_sales(network, a_d_D, downstream_shocks, other_shocks, config)
    
    R_local, T = size(downstream_shocks)
    N_rho_local, S_local, _ = size(a_d_D)
    
    # Get baseline downstream sales and compute w_r^d (share of downstream sales)
    Y_r = network.Y_r
    w_r_d = Y_r ./ sum(Y_r)  # Share of downstream in each region
    
    # Compute a_{rdi}^D from network
    a_rdi_D, _, _ = compute_a_rdi_D(network)
    
    # d ln x_{dr,t} = z_{dr,t} (regional downstream growth)
    # Under sticky prices, sales growth equals productivity shock
    d_ln_x_drt = downstream_shocks  # (R × T)
    
    # d ln x_{d,t} = Σ_r w_r^d × d ln x_{dr,t} (aggregate downstream growth)
    d_ln_x_dt = sum(w_r_d .* d_ln_x_drt, dims=1)[1, :]  # (T,)
    
    # Storage for supplier growth
    d_ln_x_it = zeros(N_rho_local, S_local, R_local, T)
    weighted_exposure_it = zeros(N_rho_local, S_local, R_local, T)
    
    # Compute supplier sales growth
    for t in 1:T
        for l in 1:R_local      # Upstream region
            for s in 1:S_local
                for rho in 1:N_rho_local
                    # Weighted exposure to downstream: Σ_r a_{rdi}^D × d ln x_{dr,t}
                    exposure_term = 0.0
                    for r in 1:R_local
                        exposure_term += a_rdi_D[rho, s, l, r] * d_ln_x_drt[r, t]
                    end
                    weighted_exposure_it[rho, s, l, t] = exposure_term
                    
                    # Supplier sales growth
                    a_d = a_d_D[rho, s, l]
                    d_ln_x_it[rho, s, l, t] = a_d * exposure_term + 
                                               (1 - a_d) * other_shocks[rho, s, l, t]
                end
            end
        end
    end
    
    return d_ln_x_it, d_ln_x_dt, d_ln_x_drt, weighted_exposure_it
end


"""
    build_panel_and_regress(d_ln_x_it, d_ln_x_dt, weighted_exposure_it, network, a_d_D)

Build panel dataset and run Table 2 regressions.

# Regression specifications:
  reg1: d ln x_{i,t} = α_i + β × downstream_growth + ε
  reg2: d ln x_{i,t} = α_i + γ_{st} + β × downstream_growth + ε

Empirical target (Aerospace): 0.112
"""
function build_panel_and_regress(d_ln_x_it, d_ln_x_dt, weighted_exposure_it, network, a_d_D)
    
    N_rho_local, S_local, R_local, T = size(d_ln_x_it)
    linkages = network.linkages
    
    println("\nBuilding panel dataset (suppliers only)...")
    
    # Build panel - ONLY suppliers (firms with linkages > 0)
    firm_ids = Int[]
    sector_ids = Int[]
    region_ids = Int[]
    period_ids = Int[]
    sales_growth = Float64[]
    downstream_growth = Float64[]
    weighted_exposure = Float64[]
    exposure_to_downstream = Float64[]
    
    firm_id = 0
    n_suppliers = 0
    
    for l in 1:R_local
        for s in 1:S_local
            for rho in 1:N_rho_local
                firm_id += 1
                
                # Only include suppliers
                if linkages[rho, s, l] == 0
                    continue
                end
                
                n_suppliers += 1
                
                for t in 1:T
                    push!(firm_ids, firm_id)
                    push!(sector_ids, s)
                    push!(region_ids, l)
                    push!(period_ids, t)
                    push!(sales_growth, d_ln_x_it[rho, s, l, t])
                    push!(downstream_growth, d_ln_x_dt[t])
                    push!(weighted_exposure, weighted_exposure_it[rho, s, l, t])
                    push!(exposure_to_downstream, a_d_D[rho, s, l])
                end
            end
        end
    end
    
    panel_df = DataFrame(
        firm_id = firm_ids,
        sector = sector_ids,
        region = region_ids,
        period = period_ids,
        d_ln_x = sales_growth,
        downstream_growth = downstream_growth,
        weighted_exposure = weighted_exposure,
        a_d_D = exposure_to_downstream
    )
    
    # Add sales growth assuming no other customer (a_d_D = 1)
    # d ln x_{i,t} = 1 × weighted_exposure + 0 × other_shock = weighted_exposure
    panel_df.d_ln_x_no_other = panel_df.weighted_exposure
    
    println("  Observations: $(nrow(panel_df))")
    println("  Unique suppliers: $n_suppliers")
    println("  Periods: $T")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Regressions
    # ─────────────────────────────────────────────────────────────────────────
    println("\n" * "="^60)
    println("Table 2 Regression Results")
    println("="^60)
    
    results = Dict()
    
    # Specification 1: Firm FE, downstream_growth
    println("\nSpecification 1: Firm FE")
    println("  d ln x_{i,t} = α_i + β × downstream_growth + ε")
    try
        reg1 = reg(panel_df, @formula(d_ln_x ~ downstream_growth + fe(firm_id)))
        
        beta = coef(reg1)[1]
        beta_se = stderror(reg1)[1]
        ci_lower = beta - 1.96 * beta_se
        ci_upper = beta + 1.96 * beta_se
        
        results["reg1"] = Dict(
            "beta" => beta,
            "beta_se" => beta_se,
            "ci_lower" => ci_lower,
            "ci_upper" => ci_upper,
            "R2" => r2(reg1),
            "N" => nobs(reg1)
        )
        
        println("  β (downstream_growth): $(round(beta, digits=4)) " *
                "(se: $(round(beta_se, digits=4)))")
        println("  95% CI: [$(round(ci_lower, digits=4)), $(round(ci_upper, digits=4))]")
        println("  R²: $(round(results["reg1"]["R2"], digits=4))")
        println("  N:  $(results["reg1"]["N"])")
    catch e
        println("  ERROR: $e")
        results["reg1"] = nothing
    end
    
    # Specification 2: Firm FE, weighted_exposure
    println("\nSpecification 2: Firm FE (weighted exposure)")
    println("  d ln x_{i,t} = α_i + β × weighted_exposure + ε")
    try
        reg2 = reg(panel_df, @formula(d_ln_x ~ a_d*weighted_exposure + fe(firm_id)))
        
        beta = coef(reg2)[1]
        beta_se = stderror(reg2)[1]
        ci_lower = beta - 1.96 * beta_se
        ci_upper = beta + 1.96 * beta_se
        
        results["reg2"] = Dict(
            "beta" => beta,
            "beta_se" => beta_se,
            "ci_lower" => ci_lower,
            "ci_upper" => ci_upper,
            "R2" => r2(reg2),
            "N" => nobs(reg2)
        )
        
        println("  β (weighted_exposure): $(round(beta, digits=4)) " *
                "(se: $(round(beta_se, digits=4)))")
        println("  95% CI: [$(round(ci_lower, digits=4)), $(round(ci_upper, digits=4))]")
        println("  R²: $(round(results["reg2"]["R2"], digits=4))")
        println("  N:  $(results["reg2"]["N"])")
    catch e
        println("  ERROR: $e")
        results["reg2"] = nothing
    end
    
    # Specification 3: No other customer (a_d_D = 1), downstream_growth
    # Sales growth = weighted_exposure when firms have no other customer
    println("\nSpecification 3: No other customer (a_d_D = 1)")
    println("  d ln x_{i,t}^{no other} = α_i + β × downstream_growth + ε")
    try
        reg3 = reg(panel_df, @formula(d_ln_x_no_other ~ downstream_growth + fe(firm_id)))
        
        beta = coef(reg3)[1]
        beta_se = stderror(reg3)[1]
        ci_lower = beta - 1.96 * beta_se
        ci_upper = beta + 1.96 * beta_se
        
        results["reg3"] = Dict(
            "beta" => beta,
            "beta_se" => beta_se,
            "ci_lower" => ci_lower,
            "ci_upper" => ci_upper,
            "R2" => r2(reg3),
            "N" => nobs(reg3)
        )
        
        println("  β (downstream_growth): $(round(beta, digits=4)) " *
                "(se: $(round(beta_se, digits=4)))")
        println("  95% CI: [$(round(ci_lower, digits=4)), $(round(ci_upper, digits=4))]")
        println("  R²: $(round(results["reg3"]["R2"], digits=4))")
        println("  N:  $(results["reg3"]["N"])")
    catch e
        println("  ERROR: $e")
        results["reg3"] = nothing
    end
    
    return panel_df, results
end


"""
    run_untargeted_validation(params; config=DEFAULT_CONFIG, empirical=nothing, input_folder=nothing)

Main function: validate calibrated model against Table 2.

Uses globals from main_pso.jl: S, R, N_downstream_per_region, regional_wages,
w_rs, distances, DistBin, N_rho, theta, lambda, nu, nu_s, epsilon, sigma_sr

# Arguments
- `params`: Calibrated parameter vector
- `config`: SimulationConfig
- `empirical`: Dict with empirical coefficients for comparison
- `input_folder`: Path to folder containing share_dist.csv

# Returns
Dict with network, panel_df, regression_results, config, shocks, exposures
"""
function run_untargeted_validation(
    params;
    config::SimulationConfig = DEFAULT_CONFIG,
    empirical = nothing,
    input_folder = nothing
)
    
    println("\n" * "="^70)
    println("UNTARGETED MOMENT VALIDATION: Table 2 Regression")
    println("="^70)
    println("Config: T=$(config.T_periods), σ_d=$(config.sigma_d), ρ_d=$(config.rho_d)")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 1: Solve network using model_CP.jl
    # ─────────────────────────────────────────────────────────────────────────
    println("\n[Step 1] Solving baseline network (fixed structure)...")
    network = solve_network(params, return_firm_level=true)
    
    n_suppliers = sum(network.linkages .> 0)
    println("  Total suppliers: $n_suppliers")
    println("  Total downstream sales: $(round(sum(network.Y_r), digits=4))")
    
    N_rho_local = size(network.firm_expenditure_shares, 1)
    S_local = size(network.firm_expenditure_shares, 2)
    R_local = size(network.firm_expenditure_shares, 3)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 2: Load exposure distribution and draw a_{di}^D
    # ─────────────────────────────────────────────────────────────────────────
    println("\n[Step 2] Drawing exposure to downstream (a_{di}^D)...")
    
    exposure_by_sector = nothing
    if input_folder !== nothing
        exposure_by_sector = load_exposure_distribution(input_folder)
    end
    
    a_d_D = draw_exposures(exposure_by_sector, S_local, R_local, N_rho_local; seed=config.seed)
    
    # Compute a_{rdi}^D from network
    a_rdi_D, sales_to_downstream, total_sales = compute_a_rdi_D(network)
    
    println("  a_{di}^D stats: mean=$(round(mean(a_d_D), digits=3)), " *
            "std=$(round(std(a_d_D), digits=3))")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 3: Generate shocks
    # ─────────────────────────────────────────────────────────────────────────
    println("\n[Step 3] Generating shocks...")
    println("  Using global σ_{sr} matrix of size $(size(sigma_sr))")
    
    # Regional downstream productivity shocks (AR(1))
    downstream_shocks = generate_downstream_shocks(R_local, config.T_periods, config)
    println("  Downstream shock std (realized): $(round(std(downstream_shocks), digits=4))")
    
    # Other customer shocks (i.i.d. with sector-region specific σ_{sr} from global)
    other_shocks = generate_other_customer_shocks(
        N_rho_local, S_local, R_local, config.T_periods, sigma_sr; seed=config.seed
    )
    println("  Other customer shock std (overall): $(round(std(other_shocks), digits=4))")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 4: Simulate supplier sales
    # ─────────────────────────────────────────────────────────────────────────
    println("\n[Step 4] Simulating supplier sales...")
    
    d_ln_x_it, d_ln_x_dt, d_ln_x_drt, weighted_exposure_it = simulate_supplier_sales(
        network, a_d_D, downstream_shocks, other_shocks, config
    )
    
    println("  Supplier sales growth std: $(round(std(d_ln_x_it), digits=4))")
    println("  Aggregate downstream growth std: $(round(std(d_ln_x_dt), digits=4))")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 5: Build panel and regress
    # ─────────────────────────────────────────────────────────────────────────
    println("\n[Step 5] Running regressions...")
    
    panel_df, reg_results = build_panel_and_regress(
        d_ln_x_it, d_ln_x_dt, weighted_exposure_it, network, a_d_D
    )

    
    return Dict(
        "network" => network,
        "panel_df" => panel_df,
        "regression_results" => reg_results,
        "config" => config,
        "downstream_shocks" => downstream_shocks,
        "other_shocks" => other_shocks,
        "a_d_D" => a_d_D,
        "a_rdi_D" => a_rdi_D
    )
end