##### Untargeted Moments: Reproduce Table 2 Regression #####
# Author: Based on model_CP.jl structure
# Purpose: Simulate productivity shocks with FIXED network and reproduce supplier correlation regression
#
# Key design: 
#   1. Solve model ONCE to determine network structure (who supplies whom)
#   2. Shocks to downstream firms propagate through fixed linkages
#   3. Upstream sales scale with downstream demand shocks
#
# NOTE: This file expects the following global constants to be defined via @everywhere in main_pso.jl:
#   S, R, N_downstream_per_region, regional_wages, w_rs, distances, DistBin,
#   N_rho, theta, lambda, nu, nu_s, epsilon

using Distributed
using SparseArrays
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
    T_periods::Int          # Number of time periods to simulate
    σ_shock::Float64        # Standard deviation of log-normal productivity shocks
    ρ_ar::Float64           # AR(1) persistence of shocks
    seed::Int               # Random seed for reproducibility
end

# Default configuration
default_config = SimulationConfig(120, 0.05, 0.8, 42)


"""
    solve_baseline_network(params)

Solve the model ONCE to determine the fixed network structure.

Uses global constants: S, R, N_downstream_per_region, regional_wages, w_rs, 
                       distances, DistBin, N_rho, theta, lambda, nu, nu_s, epsilon

Returns:
- `firm_sales_baseline`: Baseline sales for each upstream firm (N_rho × S × R)
- `firm_linkages`: Binary matrix indicating supplier status (N_rho × S × R)
- `firm_downstream_exposure`: Share of each firm's sales going to downstream industry
- `downstream_sales_baseline`: Baseline sales of downstream industry by region
"""
function solve_baseline_network(params)
    
    # Unpack parameters
    R_downstream = sum(N_downstream_per_region .!= 0)
    beta = params[1:5]
    labor_share_tech = params[6]
    input_share_tech = params[7:6+S] / sum(params[7:6+S])
    productivity_ = params[(S+7):(R_downstream+S+6)]
    T_ = params[(R_downstream+S+7):end]
    
    # Build trade costs
    tau = ones(R, R, S)
    for i in 1:R, j in 1:R
        b = DistBin[i, j]
        if b > 0
            for s in 1:S
                tau[i, j, s] += beta[b]
            end
        end
    end
    
    # Set up productivity
    productivity = ones(R)
    productivity[N_downstream_per_region .!= 0] = productivity_
    input_share_tech_mat = reshape(input_share_tech, 1, S)
    T = reshape(T_, S, R)
    
    # Draw upstream firm productivities (FIXED for all time)
    Random.seed!(50)
    frechet_rand = Frechet.(theta, T.^(1/theta))
    upstream_variety_productivity = zeros(S, R, N_rho)
    for s in 1:S, r in 1:R
        upstream_variety_productivity[s, r, :] = rand(frechet_rand[s, r], N_rho)
    end
    upstream_variety_productivity = permutedims(upstream_variety_productivity, (3, 2, 1))  # N_rho × R × S
    inv_upstream_variety_productivity = upstream_variety_productivity.^(-1)
    tau_reshaped = permutedims(tau, (3, 1, 2))  # S × R × R
    
    # Initialize storage
    firm_sales_baseline = zeros(N_rho, S, R)  # Sales of firm (rho, s, l) 
    firm_linkages = zeros(N_rho, S, R)        # 1 if firm supplies downstream industry
    firm_sales_to_downstream = zeros(N_rho, S, R, R)  # Sales to each downstream region
    c_r = zeros(R)
    
    # Solve for each downstream region
    for r in 1:R
        if N_downstream_per_region[r] >= 1
            # Prices faced by downstream firm in region r
            tau_ = reshape(tau_reshaped[:, :, r]', 1, R, S)
            prices_ = inv_upstream_variety_productivity .* tau_ .* reshape(w_rs, (1, R, 1))
            
            # Select lowest price supplier for each variety
            min_coord_rho = reshape(argmin(prices_, dims=2), N_rho, S)
            p_rs_rho = prices_[min_coord_rho]
            
            # Price indices
            p_rs = sum(1/N_rho .* p_rs_rho.^(1 .- reshape(nu_s, 1, S)), dims=1).^(1 ./ (1 .- reshape(nu_s, 1, S)))
            p_r = sum((p_rs) .^ (1 - nu) .* input_share_tech_mat).^(1 ./ (1 - nu))
            c_r_ = (labor_share_tech * regional_wages[r]^(1-lambda) + (1-labor_share_tech) * p_r^(1-lambda))^(1/(1-lambda))
            c_r[r] = c_r_ * (productivity[r]^(-1))
            
            # Compute sales from each upstream firm to this downstream region
            for l in 1:R
                # Which varieties in region l won the competition for region r?
                winner_mask = map(x -> x[2] == l ? 1.0 : 0.0, min_coord_rho)  # N_rho × S
                
                # Mark these firms as suppliers
                firm_linkages[:, :, l] .= max.(firm_linkages[:, :, l], winner_mask)
                
                # Compute their sales contribution
                # Sales = (input share) × (1-labor share) × (price index terms) × (downstream output)
                sales_contribution = winner_mask .* (1/N_rho) .* input_share_tech_mat .* (1-labor_share_tech) .* 
                    (p_rs_rho ./ p_rs).^(1 .- reshape(nu_s, 1, S)) .* 
                    (p_rs ./ p_r).^(1-nu) .* (p_r / c_r_).^(1-lambda)
                
                # Store (will multiply by downstream output later)
                firm_sales_to_downstream[:, :, l, r] = sales_contribution
            end
        end
    end
    
    # Compute downstream output and final upstream sales
    # From the paper (p.24): q_r = (p_r/P)^{ε-1} * (E/P) * δ_r
    # With monopolistic competition: p_r = c_r / μ where μ = (ε-1)/ε (inverse markup)
    # 
    # Price index: P = [Σ_r (p_r)^ε δ_r]^{1/ε} = [Σ_r (c_r/μ)^ε δ_r]^{1/ε}
    # 
    # Sales (revenue): Y_r = p_r * q_r = (c_r/μ)^ε * P^{-ε} * E * δ_r
    
    inv_markup = (epsilon - 1) / epsilon  # μ = (ε-1)/ε, this is 1/markup
    delta_r_baseline = ones(R)
    
    # Prices: p_r = c_r / μ = c_r * ε/(ε-1)
    p_r = c_r ./ inv_markup
    
    # Price index: P = [Σ_r p_r^ε δ_r]^{1/ε}
    price_index = sum((p_r[N_downstream_per_region .!= 0]).^(epsilon) .* 
                      delta_r_baseline[N_downstream_per_region .!= 0])^(1/epsilon)
    
    E = 1  # Total expenditure (normalized)
    
    # Downstream sales by region (one downstream firm per region)
    # Y_r = p_r^ε * P^{-ε} * E * δ_r
    downstream_sales_baseline = zeros(R)
    downstream_sales_baseline[N_downstream_per_region .!= 0] = 
        (p_r[N_downstream_per_region .!= 0]).^(epsilon) .* 
        price_index^(-epsilon) .* E .* 
        delta_r_baseline[N_downstream_per_region .!= 0]
    
    # Upstream firm baseline sales (summed over all downstream customers)
    for l in 1:R
        for r in 1:R
            if N_downstream_per_region[r] >= 1
                firm_sales_baseline[:, :, l] .+= firm_sales_to_downstream[:, :, l, r] .* 
                    downstream_sales_baseline[r]
            end
        end
    end
    
    # Compute exposure: share of each firm's sales coming from downstream industry
    # (In this model, ALL sales are to downstream, so exposure = 1 for suppliers)
    # But we track the regional composition for shock propagation
    firm_downstream_exposure = firm_sales_to_downstream  # Keep the full tensor
    
    println("Network solved:")
    println("  Total suppliers: $(sum(firm_linkages .> 0))")
    println("  Supplier rate: $(round(100*mean(firm_linkages), digits=2))%")
    println("  Total downstream sales: $(round(sum(downstream_sales_baseline), digits=4))")
    
    return (
        firm_sales_baseline = firm_sales_baseline,
        firm_linkages = firm_linkages,
        firm_sales_to_downstream = firm_sales_to_downstream,
        downstream_sales_baseline = downstream_sales_baseline,
        c_r = c_r,
        N_downstream_per_region = N_downstream_per_region
    )
end


"""
    propagate_shocks(baseline, shocks_matrix)

Propagate downstream demand shocks through the FIXED network.

Uses global constants: epsilon

# Arguments
- `baseline`: Output from solve_baseline_network
- `shocks_matrix`: Matrix of shocks δ_r(t) for each downstream region and time period (R × T)

# Returns
- `upstream_sales_ts`: Time series of upstream firm sales (N_rho × S × R × T)
- `downstream_sales_ts`: Time series of total downstream sales (T,)
"""
function propagate_shocks(baseline, shocks_matrix)
    
    N_downstream_per_region_local = baseline.N_downstream_per_region
    c_r = baseline.c_r
    
    R_local, T = size(shocks_matrix)
    N_rho_local = size(baseline.firm_sales_baseline, 1)
    S_local = size(baseline.firm_sales_baseline, 2)
    
    # Storage
    upstream_sales_ts = zeros(N_rho_local, S_local, R_local, T)
    downstream_sales_ts = zeros(T)
    downstream_sales_by_region_ts = zeros(R_local, T)
    
    for t in 1:T
        delta_r_t = shocks_matrix[:, t]
        
        # Downstream sales scale with demand shock
        # y_r(t) ∝ δ_r(t) × c_r^(ε-1)
        # We use proportional scaling from baseline
        downstream_sales_t = zeros(R_local)
        downstream_sales_t[N_downstream_per_region_local .!= 0] = 
            baseline.downstream_sales_baseline[N_downstream_per_region_local .!= 0] .* 
            delta_r_t[N_downstream_per_region_local .!= 0]
        
        downstream_sales_ts[t] = sum(downstream_sales_t)
        downstream_sales_by_region_ts[:, t] = downstream_sales_t
        
        # Upstream sales: scale by their customers' demand shocks
        # Each firm's sales to region r scale with δ_r(t)
        for l in 1:R_local
            for r in 1:R_local
                if N_downstream_per_region_local[r] >= 1
                    # Sales from firm in l to downstream in r, scaled by shock
                    upstream_sales_ts[:, :, l, t] .+= 
                        baseline.firm_sales_to_downstream[:, :, l, r] .* downstream_sales_t[r]
                end
            end
        end
    end
    
    return upstream_sales_ts, downstream_sales_ts, downstream_sales_by_region_ts
end


"""
    generate_shock_process(R_local, T, config)

Generate demand shocks for downstream regions.

Shock structure:
- ACROSS REGIONS: i.i.d. at each time t, all drawn from the same distribution
- OVER TIME: AR(1) process for each region independently

Each region r follows: z_r(t) = ρ × z_r(t-1) + ε_r(t)
where ε_r(t) ~ N(0, σ²_innovation) are i.i.d. across both r and t

The multiplicative shock is: δ_r(t) = exp(z_r(t))

Parameters:
- σ_shock: unconditional standard deviation of z_r(t)
- ρ_ar: AR(1) persistence parameter
"""
function generate_shock_process(R_local, T, config::SimulationConfig)
    
    Random.seed!(config.seed)
    
    # Innovation standard deviation (ensures unconditional std of z equals σ_shock)
    innovation_std = config.σ_shock * sqrt(1 - config.ρ_ar^2)
    
    # Draw ALL innovations at once: i.i.d. across regions and time from N(0, innovation_std²)
    # This ensures shocks are i.i.d. across regions from the same distribution
    innovations = randn(R_local, T) * innovation_std
    
    # AR(1) process in logs for each region independently
    z = zeros(R_local, T)
    
    # Initial period: draw from unconditional distribution N(0, σ_shock²)
    # These are i.i.d. across regions from the same law
    z[:, 1] = randn(R_local) * config.σ_shock
    
    # Subsequent periods: AR(1) evolution
    # At each t, the innovations are i.i.d. across regions from N(0, innovation_std²)
    for t in 2:T
        z[:, t] = config.ρ_ar * z[:, t-1] + innovations[:, t]
    end
    
    # Convert to multiplicative shocks
    delta = exp.(z)
    
    return delta
end


"""
    build_panel_and_regress(upstream_sales_ts, downstream_sales_ts, baseline)

Build panel dataset and run Table 2 regression.

Since there's no outside sector, ALL firms with positive sales are suppliers (a^D_is = 1).
We regress supplier sales growth on downstream industry growth.
This corresponds to column (5) of Table 2: Sup_is × a^D_is × d ln x_{s,t} with coefficient 0.112.
"""
function build_panel_and_regress(upstream_sales_ts, downstream_sales_ts, baseline)
    
    N_rho_local, S_local, R_local, T = size(upstream_sales_ts)
    
    println("\nBuilding panel dataset (suppliers only)...")
    
    # Compute downstream industry growth rates
    d_ln_downstream = zeros(T-1)
    for t in 2:T
        if downstream_sales_ts[t] > 1e-10 && downstream_sales_ts[t-1] > 1e-10
            d_ln_downstream[t-1] = log(downstream_sales_ts[t]) - log(downstream_sales_ts[t-1])
        end
    end
    
    # Build panel - ONLY suppliers (firms with positive sales)
    firm_ids = Int[]
    sector_ids = Int[]
    region_ids = Int[]
    period_ids = Int[]
    sales_growth = Float64[]
    downstream_growth = Float64[]
    
    firm_id = 0
    n_suppliers = 0
    
    for l in 1:R_local  # Upstream region
        for s in 1:S_local  # Sector
            for rho in 1:N_rho_local  # Variety
                firm_id += 1
                
                # Only include suppliers (firms with positive baseline sales)
                if baseline.firm_linkages[rho, s, l] == 0
                    continue
                end
                
                n_suppliers += 1
                
                # Get sales time series
                firm_sales = upstream_sales_ts[rho, s, l, :]
                
                # Add observations for each period
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
    println("  Unique suppliers: $(length(unique(panel_df.firm_id)))")
    
    # ========================================
    # Run regression: d ln x_{i,t} = α_i + β × d ln x_{s,t} + FE + ε
    # 
    # This corresponds to Table 2 column (5): Sup × a^D × d ln x_{s,t}
    # Since all firms here are suppliers with a^D = 1, β captures the 
    # comovement between suppliers and downstream industry.
    # Empirical target: 0.112 (aerospace)
    # ========================================
    
    println("\n" * "="^60)
    println("Table 2 Column (5) Regression")
    println("(Suppliers only, comparing to Sup × a^D × d ln x coefficient)")
    println("="^60)
    
    results = Dict()
    
    # Basic specification: firm FE only
    println("\nSpecification 1: Firm FE")
    try
        reg1 = reg(panel_df, @formula(d_ln_x ~ d_ln_downstream + fe(firm_id)))
        
        results["firm_fe"] = Dict(
            "beta" => coef(reg1)[1],
            "beta_se" => stderror(reg1)[1],
            "R2" => r2(reg1),
            "N" => nobs(reg1)
        )
        
        println("  β (d_ln_downstream): $(round(results["firm_fe"]["beta"], digits=4)) (se: $(round(results["firm_fe"]["beta_se"], digits=4)))")
        println("  R²: $(round(results["firm_fe"]["R2"], digits=4))")
        println("  N:  $(results["firm_fe"]["N"])")
    catch e
        println("  ERROR: $e")
        results["firm_fe"] = nothing
    end
    
    # With sector × period FE (absorbs d_ln_downstream, so we look at within-sector variation)
    println("\nSpecification 2: Firm FE + Sector × Period FE")
    try
        panel_df.sector_period = string.(panel_df.sector) .* "_" .* string.(panel_df.period)
        
        # With sector×period FE, d_ln_downstream is absorbed
        # We can still estimate firm-level exposure effects if we had variation in a^D
        # For now, just report the R² to show how much variation is explained
        reg2 = reg(panel_df, @formula(d_ln_x ~ fe(firm_id) + fe(sector_period)))
        
        results["sector_period_fe"] = Dict(
            "R2" => r2(reg2),
            "N" => nobs(reg2)
        )
        
        println("  R² (with sector×period FE): $(round(results["sector_period_fe"]["R2"], digits=4))")
        println("  N:  $(results["sector_period_fe"]["N"])")
    catch e
        println("  ERROR: $e")
        results["sector_period_fe"] = nothing
    end
    
    return panel_df, results
end


"""
    run_untargeted_validation(params; config=default_config, empirical=nothing)

Main function: validate calibrated model against Table 2.

Uses global constants: S, R, N_downstream_per_region, regional_wages, w_rs, 
                       distances, DistBin, N_rho, theta, lambda, nu, nu_s, epsilon
"""
function run_untargeted_validation(
    params; 
    config::SimulationConfig = default_config,
    empirical = nothing
)
    
    println("\n" * "="^70)
    println("UNTARGETED MOMENT VALIDATION: Table 2 Regression")
    println("="^70)
    println("Config: T=$(config.T_periods), σ=$(config.σ_shock), ρ=$(config.ρ_ar)")
    
    # Step 1: Solve network ONCE (uses global constants)
    println("\n[Step 1] Solving baseline network (fixed structure)...")
    baseline = solve_baseline_network(params)
    
    # Step 2: Generate shocks
    println("\n[Step 2] Generating demand shocks...")
    shocks = generate_shock_process(R, config.T_periods, config)
    println("  Shock std (realized): $(round(std(log.(shocks)), digits=4))")
    
    # Step 3: Propagate through fixed network
    println("\n[Step 3] Propagating shocks through network...")
    upstream_sales_ts, downstream_sales_ts, _ = propagate_shocks(baseline, shocks)
    
    # Step 4: Build panel and regress
    println("\n[Step 4] Running regressions...")
    panel_df, reg_results = build_panel_and_regress(
        upstream_sales_ts, downstream_sales_ts, baseline
    )
    
    # Step 5: Compare to empirical
    if empirical !== nothing
        println("\n" * "="^60)
        println("Comparison to Empirical (Table 2, Column 5)")
        println("="^60)
        
        println("\n                      Empirical    Simulated")
        println("-"^50)
        
        if reg_results["firm_fe"] !== nothing && haskey(empirical, "gamma_col5")
            println("β (Sup × a^D × d ln x): $(round(empirical["gamma_col5"], digits=4))        $(round(reg_results["firm_fe"]["beta"], digits=4))")
            
            rel_diff = abs(reg_results["firm_fe"]["beta"] - empirical["gamma_col5"]) / abs(empirical["gamma_col5"])
            println("\nRelative difference: $(round(100*rel_diff, digits=1))%")
        end
    end
    
    return Dict(
        "baseline" => baseline,
        "panel_df" => panel_df,
        "regression_results" => reg_results,
        "config" => config,
        "shocks" => shocks
    )
end


# ============================================================================
# STANDALONE EXECUTION (for testing without main_pso.jl)
# ============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    
    println("="^70)
    println("Running Untargeted Moments Validation (Standalone Mode)")
    println("="^70)
    println("\nWARNING: Standalone mode requires loading data and defining globals.")
    println("For normal usage, run from main_pso.jl which defines all @everywhere constants.\n")
    
    industry = "aero"
    input_folder = "./baseline_" * industry
    output_folder = "./reporting_" * industry
    
    # Load data and define globals (mimics main_pso.jl)
    println("Loading data and defining globals...")
    coefs = CSV.read(joinpath(input_folder, "stats.csv"), DataFrame)
    
    global const distances = NPZ.npzread(joinpath(input_folder, "distances.npy"))
    global const w_rs = NPZ.npzread(joinpath(input_folder, "w_rs.npy"))
    global const regional_wages = NPZ.npzread(joinpath(input_folder, "regional_wages.npy"))
    global const N_downstream_per_region = NPZ.npzread(joinpath(input_folder, "N_downstream_per_region.npy"))
    filter_N_upstream_local = NPZ.npzread(joinpath(input_folder, "filter_N_upstream.npy"))
    
    global const S = size(filter_N_upstream_local, 1)
    global const R = size(filter_N_upstream_local, 2)
    
    # Build DistBin
    DistBin_local = zeros(Int, R, R)
    for i in 1:R, j in 1:R
        d = distances[i, j]
        DistBin_local[i, j] = d <= 20 ? 0 : d <= 50 ? 1 : d <= 100 ? 2 : d <= 150 ? 3 : d <= 200 ? 4 : 5
    end
    global const DistBin = DistBin_local
    
    global const N_rho = 50
    global const theta = 1.768
    global const lambda = 0.5
    global const nu = 0.2
    global const nu_s = ones(S) .* 2.5
    global const epsilon = coefs[1, "value"]
    
    # Load calibrated parameters
    println("Loading calibrated parameters...")
    best_params = nothing
    for loop in 50:-1:1
        for stage in 10:-1:1
            f = joinpath(output_folder, "epoch_$loop", "$stage", "best_params.npy")
            if isfile(f)
                best_params = NPZ.npzread(f)[:, 1]
                println("  Loaded from: $f")
                @goto found
            end
        end
    end
    @label found
    
    if best_params === nothing
        f = joinpath(output_folder, "0", "best_params.npy")
        best_params = NPZ.npzread(f)[:, 1]
        println("  Loaded from: $f")
    end
    
    # Empirical coefficient from Table 2, Column (5)
    empirical = Dict(
        "gamma_col5" => 0.112   # Aerospace, col (5): Sup × a^D × d ln x
    )
    
    # Run validation
    config = SimulationConfig(120, 0.05, 0.8, 42)
    results = run_untargeted_validation(best_params, config=config, empirical=empirical)
    
    # Save results
    if results["regression_results"]["firm_fe"] !== nothing
        NPZ.npzwrite(joinpath(output_folder, "untargeted_beta.npy"), 
                     [results["regression_results"]["firm_fe"]["beta"]])
    end
    
    println("\n" * "="^70)
    println("Done!")
    println("="^70)
end