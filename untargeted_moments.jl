##### Untargeted Moments: Reproduce Table 2 Regression #####
# Author: Swann Chelly
# Purpose: Validate calibrated model by reproducing Table 2 regression
#
# This file uses solve_network() from model_CP.jl and simulates
# demand shocks to generate time series variation for regression analysis.
#
# Key design:
#   1. Use solve_network(params, return_firm_level=true) to get fixed network
#   2. Simulate i.i.d. demand shocks δ_r(t) across regions
#   3. Propagate shocks through fixed linkages
#   4. Run regression to compare with Table 2 coefficients
#
# ═══════════════════════════════════════════════════════════════════════════════
# THEORY (from paper)
# ═══════════════════════════════════════════════════════════════════════════════
#
# At baseline:
#   Y_r = p_r^ε · P^{-ε} · E · δ_r  (downstream sales)
#   Upstream sales from (ρ,s,l) to r = share_{ρsl→r} × μ × Y_r
#
# At time t with shock δ_r(t):
#   Y_r(t) = Y_r_baseline × δ_r(t)  (proportional scaling)
#   Upstream sales(t) = share × μ × Y_r(t) = share × μ × Y_r_baseline × δ_r(t)
#
# Growth rate (markup μ cancels):
#   d ln(upstream sales) = d ln(Y_r) = d ln(δ_r)
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
    T_periods::Int          # Number of time periods
    sigma_s::Float64        # Standard deviation of log-normal demand shocks
    rho_s::Float64           # AR(1) persistence
    seed::Int               # Random seed
end

# Default configuration
const DEFAULT_CONFIG = SimulationConfig(36, 0.17, 0.92, 42) # Aerospace config


"""
    generate_shock_process(R_local, T, config)

Generate demand shocks δ_r(t) for downstream regions.

Shock structure:
- ACROSS REGIONS: i.i.d. at each time t from same distribution
- OVER TIME: AR(1) process per region: z_r(t) = ρ × z_r(t-1) + ε_r(t)

The multiplicative shock is: δ_r(t) = exp(z_r(t))
"""
function generate_shock_process(R_local, T, config::SimulationConfig)
    
    Random.seed!(config.seed)
    
    # Innovation std (ensures unconditional std of z equals sigma_s)
    innovation_std = config.sigma_s * sqrt(1 - config.rho_s^2)
    
    # Draw ALL innovations at once: i.i.d. across regions and time
    innovations = randn(R_local, T) * innovation_std
    
    # AR(1) process in logs
    z = zeros(R_local, T)
    z[:, 1] = randn(R_local) * config.sigma_s  # Initial from unconditional
    
    for t in 2:T
        z[:, t] = config.rho_s * z[:, t-1] + innovations[:, t]
    end
    
    # Convert to multiplicative shocks
    delta = exp.(z)
    
    return delta
end


"""
    propagate_shocks(network, shocks_matrix)

Propagate demand shocks through the FIXED production network.

# Arguments
- `network`: Output from solve_network(params, return_firm_level=true)
- `shocks_matrix`: Matrix of shocks δ_r(t) (R × T)

# Returns
- `upstream_sales_ts`: Firm sales time series (N_rho × S × R × T)
- `downstream_sales_ts`: Total downstream sales (T,)
- `downstream_sales_by_region_ts`: Downstream sales by region (R × T)
"""
function propagate_shocks(network, shocks_matrix)
    
    R_local, T = size(shocks_matrix)
    N_rho_local = size(network.firm_expenditure_shares, 1)
    S_local = size(network.firm_expenditure_shares, 2)
    
    mu = network.mu                          # Inverse markup
    Y_r_baseline = network.Y_r               # Baseline downstream sales
    exp_shares = network.firm_expenditure_shares  # Expenditure shares
    
    # Storage
    upstream_sales_ts = zeros(N_rho_local, S_local, R_local, T)
    downstream_sales_ts = zeros(T)
    downstream_sales_by_region_ts = zeros(R_local, T)
    
    for t in 1:T
        delta_t = shocks_matrix[:, t]
        
        # Downstream sales scale with shock: Y_r(t) = Y_r_baseline × δ_r(t)
        Y_r_t = Y_r_baseline .* delta_t
        
        downstream_sales_ts[t] = sum(Y_r_t)
        downstream_sales_by_region_ts[:, t] = Y_r_t
        
        # Upstream sales = expenditure_share × total_cost
        # Total cost_r = μ × Y_r (inverse markup × revenue)
        for l in 1:R_local      # Upstream region
            for r in 1:R_local  # Downstream region
                if Y_r_baseline[r] > 0
                    total_cost_r_t = mu * Y_r_t[r]
                    upstream_sales_ts[:, :, l, t] .+= 
                        exp_shares[:, :, l, r] .* total_cost_r_t
                end
            end
        end
    end
    
    return upstream_sales_ts, downstream_sales_ts, downstream_sales_by_region_ts
end


"""
    build_panel_and_regress(upstream_sales_ts, downstream_sales_ts, network)

Build panel dataset and run Table 2 regression.

# Regression specification
d ln x_{i,t} = α_i + β × d ln x_{s,t} + ε_{i,t}

This corresponds to Table 2 column (5): Sup × a^D × d ln x_{s,t}
Since all simulated firms are suppliers with a^D = 1, β captures
the comovement between supplier sales and downstream industry sales.

Empirical targets: Aerospace = 0.112, Car = 0.161
"""
function build_panel_and_regress(upstream_sales_ts, downstream_sales_ts, network)
    
    N_rho_local, S_local, R_local, T = size(upstream_sales_ts)
    linkages = network.linkages
    
    println("\nBuilding panel dataset (suppliers only)...")
    
    # Compute downstream growth rates
    d_ln_downstream = zeros(T - 1)
    for t in 2:T
        if downstream_sales_ts[t] > 1e-10 && downstream_sales_ts[t-1] > 1e-10
            d_ln_downstream[t-1] = log(downstream_sales_ts[t]) - log(downstream_sales_ts[t-1])
        end
    end
    
    # Build panel - ONLY suppliers
    firm_ids = Int[]
    sector_ids = Int[]
    region_ids = Int[]
    period_ids = Int[]
    sales_growth = Float64[]
    downstream_growth = Float64[]
    
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
                firm_sales = upstream_sales_ts[rho, s, l, :]
                
                for t in 2:T
                    if firm_sales[t] > 1e-12 && firm_sales[t-1] > 1e-12
                        d_ln_x = log(firm_sales[t]) - log(firm_sales[t-1])
                        
                        push!(firm_ids, firm_id)
                        push!(sector_ids, s)
                        push!(region_ids, l)
                        push!(period_ids, t)
                        push!(sales_growth, d_ln_x)
                        push!(downstream_growth, d_ln_downstream[t-1])
                    end
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
        d_ln_downstream = downstream_growth
    )
    
    println("  Observations: $(nrow(panel_df))")
    println("  Unique suppliers: $n_suppliers")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Regressions
    # ─────────────────────────────────────────────────────────────────────────
    println("\n" * "="^60)
    println("Table 2 Column (5) Regression")
    println("="^60)
    
    results = Dict()
    
    # Specification 1: Firm FE only
    println("\nSpecification 1: Firm FE")
    try
        reg1 = reg(panel_df, @formula(d_ln_x ~ d_ln_downstream + fe(firm_id)))
        
        results["firm_fe"] = Dict(
            "beta" => coef(reg1)[1],
            "beta_se" => stderror(reg1)[1],
            "R2" => r2(reg1),
            "N" => nobs(reg1)
        )
        
        println("  β (d_ln_downstream): $(round(results["firm_fe"]["beta"], digits=4)) " *
                "(se: $(round(results["firm_fe"]["beta_se"], digits=4)))")
        println("  R²: $(round(results["firm_fe"]["R2"], digits=4))")
        println("  N:  $(results["firm_fe"]["N"])")
    catch e
        println("  ERROR: $e")
        results["firm_fe"] = nothing
    end
    
    # Specification 2: Firm FE + Sector × Period FE
    println("\nSpecification 2: Firm FE + Sector × Period FE")
    try
        panel_df.sector_period = string.(panel_df.sector) .* "_" .* string.(panel_df.period)
        reg2 = reg(panel_df, @formula(d_ln_x ~ fe(firm_id) + fe(sector_period)))
        
        results["sector_period_fe"] = Dict(
            "beta" => coef(reg2)[1],
            "beta_se" => stderror(reg2)[1],
            "R2" => r2(reg2),
            "N" => nobs(reg2)
        )
        println("With sectorxpreiod FE")
        println("  β (d_ln_downstream): $(round(results["sector_period_fe"]["beta"], digits=4)) " *
                "(se: $(round(results["sector_period_fe"]["beta_se"], digits=4)))")
        println("  R²: $(round(results["sector_period_fe"]["R2"], digits=4))")
        println("  N:  $(results["sector_period_fe"]["N"])")
    catch e
        println("  ERROR: $e")
        results["sector_period_fe"] = nothing
    end
    
    return panel_df, results
end


"""
    run_untargeted_validation(params; config=DEFAULT_CONFIG, empirical=nothing)

Main function: validate calibrated model against Table 2.

Uses globals from main_pso.jl: S, R, N_downstream_per_region, regional_wages,
w_rs, distances, DistBin, N_rho, theta, lambda, nu, nu_s, epsilon

# Arguments
- `params`: Calibrated parameter vector
- `config`: SimulationConfig
- `empirical`: Dict with empirical coefficients for comparison

# Returns
Dict with network, panel_df, regression_results, config, shocks
"""
function run_untargeted_validation(
    params;
    config::SimulationConfig = DEFAULT_CONFIG,
    empirical = nothing
)
    
    println("\n" * "="^70)
    println("UNTARGETED MOMENT VALIDATION: Table 2 Regression")
    println("="^70)
    println("Config: T=$(config.T_periods), σ=$(config.sigma_s), ρ=$(config.rho_s)")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 1: Solve network using model_CP.jl
    # ─────────────────────────────────────────────────────────────────────────
    println("\n[Step 1] Solving baseline network (fixed structure)...")
    network = solve_network(params, return_firm_level=true)
    
    n_suppliers = sum(network.linkages .> 0)
    supplier_rate = 100 * mean(network.linkages)
    println("  Total suppliers: $n_suppliers")
    println("  Supplier rate: $(round(supplier_rate, digits=2))%")
    println("  Total downstream sales: $(round(sum(network.Y_r), digits=4))")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 2: Generate demand shocks
    # ─────────────────────────────────────────────────────────────────────────
    println("\n[Step 2] Generating demand shocks...")
    shocks = generate_shock_process(R, config.T_periods, config)
    println("  Shock std (realized): $(round(std(log.(shocks)), digits=4))")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 3: Propagate shocks through fixed network
    # ─────────────────────────────────────────────────────────────────────────
    println("\n[Step 3] Propagating shocks through network...")
    upstream_sales_ts, downstream_sales_ts, _ = propagate_shocks(network, shocks)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 4: Build panel and regress
    # ─────────────────────────────────────────────────────────────────────────
    println("\n[Step 4] Running regressions...")
    panel_df, reg_results = build_panel_and_regress(
        upstream_sales_ts, downstream_sales_ts, network
    )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 5: Compare to empirical
    # ─────────────────────────────────────────────────────────────────────────
    if empirical !== nothing
        println("\n" * "="^60)
        println("Comparison to Empirical (Table 2, Column 5)")
        println("="^60)
        
        println("\n                      Empirical    Simulated")
        println("-"^50)
        
        if reg_results["firm_fe"] !== nothing && haskey(empirical, "gamma_col5")
            emp_val = empirical["gamma_col5"]
            sim_val = reg_results["firm_fe"]["beta"]
            println("β (Sup × a^D × d ln x): $(round(emp_val, digits=4))        $(round(sim_val, digits=4))")
            
            rel_diff = abs(sim_val - emp_val) / abs(emp_val)
            println("\nRelative difference: $(round(100*rel_diff, digits=1))%")
        end
    end
    
    return Dict(
        "network" => network,
        "panel_df" => panel_df,
        "regression_results" => reg_results,
        "config" => config,
        "shocks" => shocks
    )
end