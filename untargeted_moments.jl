##### Untargeted Moments: Reproduce Table 2 Regression #####
# Author: Swann Chelly
# Purpose: Validate calibrated model by reproducing Table 2 regression
#
# This file uses solve_network() from model_CP.jl and simulates
# productivity shocks to generate time series variation for regression analysis.
#
# ═══════════════════════════════════════════════════════════════════════════════
# SHOCK MODELS
# ═══════════════════════════════════════════════════════════════════════════════
#
# MODEL 1: UNIVARIATE (Original)
#   z_{r,t} = ρ × z_{r,t-1} + u_{r,t}
#   u_{r,t} ~ N(0, σ_d²) i.i.d. across regions
#   - Same persistence ρ for all regions
#   - Independent shocks across regions
#
# MODEL 2: MULTIVARIATE (New)
#   z_{r,t} = ρ_r × z_{r,t-1} + u_{r,t}
#   u_t ~ N(0, Σ)
#   - Region-specific persistence ρ_r
#   - Correlated innovations across regions via Σ
#
# ═══════════════════════════════════════════════════════════════════════════════
#
# Under sticky prices:
#   d ln x_{d,t} = Σ_r w_r^d × d ln x_{dr,t}  (aggregate downstream)
#   d ln x_{dr,t} = z_{dr,t}                   (regional downstream)
#
# Supplier sales evolution:
#   d ln x_{i,t} = a_{di}^D × Σ_r a_{rdi}^D × d ln x_{drt} + (1 - a_{di}^D) × d ln x_{oi,t}
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


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION STRUCTS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ShockModel

Enum-like type to specify which shock model to use.
"""
@enum ShockModel begin
    UNIVARIATE   # Original: same ρ, i.i.d. across regions
    MULTIVARIATE # New: region-specific ρ_r, correlated Σ
end


"""
    SimulationConfig

Configuration for the untargeted moments simulation.

# Fields
- `T_periods`: Number of time periods (quarters)
- `sigma_d`: Std of downstream AR(1) shocks (univariate only)
- `rho_d`: AR(1) persistence for downstream (univariate only)
- `seed`: Random seed
- `shock_model`: Which shock model to use (UNIVARIATE or MULTIVARIATE)
"""
struct SimulationConfig
    T_periods::Int
    sigma_d::Float64        # Used only for UNIVARIATE
    rho_d::Float64          # Used only for UNIVARIATE
    seed::Int
    shock_model::ShockModel
end

# Constructor with default shock model (backward compatible)
SimulationConfig(T_periods, sigma_d, rho_d, seed) = 
    SimulationConfig(T_periods, sigma_d, rho_d, seed, UNIVARIATE)

# Default configuration
const DEFAULT_CONFIG = SimulationConfig(36, 0.05, -0.15, 42, UNIVARIATE)


"""
    MultivariateShockParams

Parameters for the multivariate shock model.

# Fields
- `rho_r`: Vector of region-specific persistence parameters (R,)
- `Sigma`: Cross-regional covariance matrix of innovations (R, R)
"""
struct MultivariateShockParams
    rho_r::Vector{Float64}
    Sigma::Matrix{Float64}
end


"""
    load_multivariate_params(input_folder)

Load multivariate shock parameters from .npy files.

Expected files:
- rho_r.npy: Region-specific persistence (R,)
- Sigma_innovations.npy: Covariance matrix (R, R)
"""
function load_multivariate_params(input_folder::String)
    rho_path = joinpath(input_folder, "rho_r.npy")
    sigma_path = joinpath(input_folder, "Sigma_innovations.npy")
    
    if !isfile(rho_path)
        error("Multivariate params file not found: $rho_path")
    end
    if !isfile(sigma_path)
        error("Multivariate params file not found: $sigma_path")
    end
    
    rho_r = NPZ.npzread(rho_path)
    Sigma = NPZ.npzread(sigma_path)
    
    # Ensure Sigma is positive definite
    eigenvalues = eigvals(Symmetric(Sigma))
    if minimum(eigenvalues) < 0
        @warn "Sigma not positive definite, adding ridge"
        Sigma = Sigma + abs(minimum(eigenvalues)) * 1.01 * I
    end
    
    println("  Loaded multivariate shock parameters:")
    println("    R = $(length(rho_r))")
    println("    ρ_r range: [$(round(minimum(rho_r), digits=3)), $(round(maximum(rho_r), digits=3))]")
    println("    ρ_r mean: $(round(mean(rho_r), digits=3))")
    println("    Σ diagonal mean: $(round(mean(diag(Sigma)), digits=6))")
    println("    Σ off-diagonal mean: $(round(mean(Sigma[.!I(size(Sigma,1))]), digits=6))")
    
    return MultivariateShockParams(vec(rho_r), Sigma)
end


# ═══════════════════════════════════════════════════════════════════════════════
# EXPOSURE DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════

"""
    load_exposure_distribution(input_folder)

Load the empirical distribution of a_{di}^D (share of sales to downstream industry)
from CSV file share_dist.csv with columns: A129 (sector as string), PartCa (exposure share).
"""
function load_exposure_distribution(input_folder)
    csv_path = joinpath(input_folder, "share_dist.csv")
    
    if !isfile(csv_path)
        @warn "Exposure distribution file not found: $csv_path. Using uniform [0.1, 0.9]."
        return nothing
    end
    
    df = CSV.read(csv_path, DataFrame)
    
    if !("A129" in names(df)) || !("PartCa" in names(df))
        error("CSV must have columns 'A129' (sector) and 'PartCa' (exposure)")
    end
    
    df.A129 = string.(df.A129)
    sort!(df, :A129)
    
    unique_sectors = sort(unique(df.A129))
    sector_to_int = Dict(s => i for (i, s) in enumerate(unique_sectors))
    
    println("  Sector mapping (A129 -> integer index):")
    for (s, i) in sort(collect(sector_to_int), by=x->x[2])
        println("    $s -> $i")
    end
    
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
"""
function draw_exposures(exposure_by_sector, S_local, R_local, N_rho_local; seed=42)
    
    Random.seed!(seed)
    
    a_d_D = zeros(N_rho_local, S_local, R_local)
    
    for s in 1:S_local
        if exposure_by_sector !== nothing && haskey(exposure_by_sector, s)
            emp_dist = exposure_by_sector[s]
            for l in 1:R_local
                for rho in 1:N_rho_local
                    a_d_D[rho, s, l] = rand(emp_dist)
                end
            end
        else
            for l in 1:R_local
                for rho in 1:N_rho_local
                    a_d_D[rho, s, l] = rand() * 0.8 + 0.1
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
"""
function compute_a_rdi_D(network)
    
    exp_shares = network.firm_expenditure_shares
    Y_r = network.Y_r
    mu = network.mu
    
    N_rho_local, S_local, R_local, _ = size(exp_shares)
    
    total_cost = mu .* Y_r
    
    sales_to_downstream = zeros(N_rho_local, S_local, R_local, R_local)
    
    for r in 1:R_local
        if total_cost[r] > 1e-10
            sales_to_downstream[:, :, :, r] = exp_shares[:, :, :, r] .* total_cost[r]
        end
    end
    
    total_sales_to_downstream = sum(sales_to_downstream, dims=4)
    
    a_rdi_D = zeros(N_rho_local, S_local, R_local, R_local)
    
    for l in 1:R_local
        for s in 1:S_local
            for rho in 1:N_rho_local
                total = total_sales_to_downstream[rho, s, l, 1]
                if total > 1e-12
                    for r in 1:R_local
                        a_rdi_D[rho, s, l, r] = sales_to_downstream[rho, s, l, r] / total
                    end
                end
            end
        end
    end
    
    return a_rdi_D, sales_to_downstream, total_sales_to_downstream[:,:,:,1]
end


# ═══════════════════════════════════════════════════════════════════════════════
# SHOCK GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

"""
    generate_downstream_shocks_univariate(R_local, T, config)

Generate regional downstream shocks using UNIVARIATE model.

Model:
    z_{r,t} = ρ × z_{r,t-1} + u_{r,t}
    u_{r,t} ~ N(0, σ²) i.i.d. across regions

Returns: Matrix (R × T) of log shocks z_{r,t}
"""
function generate_downstream_shocks_univariate(R_local, T, config::SimulationConfig)
    
    Random.seed!(config.seed)
    
    innovation_std = config.sigma_d * sqrt(1 - config.rho_d^2)
    
    # i.i.d. innovations across regions and time
    innovations = randn(R_local, T) * innovation_std
    
    z = zeros(R_local, T)
    z[:, 1] = randn(R_local) * config.sigma_d
    
    for t in 2:T
        z[:, t] = config.rho_d * z[:, t-1] + innovations[:, t]
    end
    
    return z
end


"""
    generate_downstream_shocks_multivariate(R_local, T, params, seed)

Generate regional downstream shocks using MULTIVARIATE model.

Model:
    z_{r,t} = ρ_r × z_{r,t-1} + u_{r,t}
    u_t ~ N(0, Σ)

Where:
- ρ_r: region-specific persistence
- Σ: cross-regional covariance matrix

Returns: Matrix (R × T) of log shocks z_{r,t}
"""
function generate_downstream_shocks_multivariate(
    R_local::Int, 
    T::Int, 
    params::MultivariateShockParams,
    seed::Int
)
    Random.seed!(seed)
    
    rho_r = params.rho_r
    Sigma = params.Sigma
    
    # Validate dimensions
    if length(rho_r) != R_local
        error("rho_r length ($(length(rho_r))) != R ($R_local)")
    end
    if size(Sigma, 1) != R_local || size(Sigma, 2) != R_local
        error("Sigma size ($(size(Sigma))) != ($R_local, $R_local)")
    end
    
    # Cholesky decomposition for correlated draws
    L = cholesky(Symmetric(Sigma)).L
    
    # Draw correlated innovations: u_t = L × ε_t where ε ~ N(0, I)
    standard_normals = randn(R_local, T)
    innovations = L * standard_normals  # (R × T)
    
    # Compute unconditional variance for initialization
    # For AR(1): Var(z_r) = σ²_r / (1 - ρ_r²)
    # where σ²_r = Σ_{rr}
    unconditional_std = sqrt.(diag(Sigma) ./ (1 .- rho_r.^2))
    
    # Initialize and simulate
    z = zeros(R_local, T)
    z[:, 1] = unconditional_std .* randn(R_local)
    
    for t in 2:T
        # z_{r,t} = ρ_r × z_{r,t-1} + u_{r,t}
        z[:, t] = rho_r .* z[:, t-1] + innovations[:, t]
    end
    
    return z
end


"""
    generate_downstream_shocks(R_local, T, config; multivar_params=nothing)

Main dispatch function for generating downstream shocks.

Uses config.shock_model to determine which model to use.
"""
function generate_downstream_shocks(
    R_local::Int, 
    T::Int, 
    config::SimulationConfig;
    multivar_params::Union{Nothing, MultivariateShockParams} = nothing
)
    if config.shock_model == UNIVARIATE
        return generate_downstream_shocks_univariate(R_local, T, config)
    elseif config.shock_model == MULTIVARIATE
        if multivar_params === nothing
            error("multivar_params required for MULTIVARIATE shock model")
        end
        return generate_downstream_shocks_multivariate(R_local, T, multivar_params, config.seed)
    else
        error("Unknown shock model: $(config.shock_model)")
    end
end


"""
    generate_other_customer_shocks(N_rho_local, S_local, R_local, T, sigma_sr; seed=42)

Generate i.i.d. shocks for "other customers" of each supplier.
"""
function generate_other_customer_shocks(N_rho_local, S_local, R_local, T, sigma_sr; seed=42)
    
    Random.seed!(seed + 1000)
    
    shocks = zeros(N_rho_local, S_local, R_local, T)
    
    if sigma_sr === nothing
        default_sigma = 0.17
        shocks = randn(N_rho_local, S_local, R_local, T) * default_sigma
    else
        for s in 1:S_local
            for l in 1:R_local
                sigma = sigma_sr[s, l]
                shocks[:, s, l, :] = randn(N_rho_local, T) * sigma
            end
        end
    end
    
    return shocks
end


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

"""
    simulate_supplier_sales(network, a_d_D, downstream_shocks, other_shocks, config)

Simulate supplier sales growth.

Model:
    d ln x_{i,t} = a_{di}^D × Σ_r a_{rdi}^D × d ln x_{dr,t} + (1 - a_{di}^D) × d ln x_{oi,t}
"""
function simulate_supplier_sales(network, a_d_D, downstream_shocks, other_shocks, config)
    
    R_local, T = size(downstream_shocks)
    N_rho_local, S_local, _ = size(a_d_D)
    
    Y_r = network.Y_r
    w_r_d = Y_r ./ sum(Y_r)
    
    a_rdi_D, _, _ = compute_a_rdi_D(network)
    
    d_ln_x_drt = downstream_shocks
    d_ln_x_dt = sum(w_r_d .* d_ln_x_drt, dims=1)[1, :]
    
    d_ln_x_it = zeros(N_rho_local, S_local, R_local, T)
    weighted_exposure_it = zeros(N_rho_local, S_local, R_local, T)
    
    for t in 1:T
        for l in 1:R_local
            for s in 1:S_local
                for rho in 1:N_rho_local
                    exposure_term = 0.0
                    for r in 1:R_local
                        exposure_term += a_rdi_D[rho, s, l, r] * d_ln_x_drt[r, t]
                    end
                    weighted_exposure_it[rho, s, l, t] = exposure_term
                    
                    a_d = a_d_D[rho, s, l]
                    d_ln_x_it[rho, s, l, t] = a_d * exposure_term + 
                                               (1 - a_d) * other_shocks[rho, s, l, t]
                end
            end
        end
    end
    
    return d_ln_x_it, d_ln_x_dt, d_ln_x_drt, weighted_exposure_it
end


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL CONSTRUCTION AND REGRESSION
# ═══════════════════════════════════════════════════════════════════════════════

"""
    build_panel_and_regress(d_ln_x_it, d_ln_x_dt, weighted_exposure_it, network, a_d_D)

Build panel dataset and run Table 2 regressions.
"""
function build_panel_and_regress(d_ln_x_it, d_ln_x_dt, weighted_exposure_it, network, a_d_D)
    
    N_rho_local, S_local, R_local, T = size(d_ln_x_it)
    linkages = network.linkages
    
    println("\nBuilding panel dataset (suppliers only)...")
    
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
    
    panel_df.d_ln_x_no_other = panel_df.weighted_exposure
    
    println("  Observations: $(nrow(panel_df))")
    println("  Unique suppliers: $n_suppliers")
    println("  Periods: $T")
    
    println("\n" * "="^60)
    println("Table 2 Regression Results")
    println("="^60)
    
    results = Dict()
    
    # Specification 1
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
    
    # Specification 2
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
    
    # Specification 3
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


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN VALIDATION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

"""
    run_untargeted_validation(params; config, empirical, input_folder, multivar_params)

Main function: validate calibrated model against Table 2.

# Arguments
- `params`: Calibrated parameter vector
- `config`: SimulationConfig (includes shock_model field)
- `empirical`: Dict with empirical coefficients for comparison
- `input_folder`: Path to folder containing share_dist.csv
- `multivar_params`: MultivariateShockParams (required if config.shock_model == MULTIVARIATE)

# Returns
Dict with network, panel_df, regression_results, config, shocks, exposures
"""
function run_untargeted_validation(
    params;
    config::SimulationConfig = DEFAULT_CONFIG,
    empirical = nothing,
    input_folder = nothing,
    multivar_params::Union{Nothing, MultivariateShockParams} = nothing
)
    
    model_name = config.shock_model == UNIVARIATE ? "UNIVARIATE" : "MULTIVARIATE"
    
    println("\n" * "="^70)
    println("UNTARGETED MOMENT VALIDATION: Table 2 Regression")
    println("Shock Model: $model_name")
    println("="^70)
    println("Config: T=$(config.T_periods), seed=$(config.seed)")
    
    if config.shock_model == UNIVARIATE
        println("  σ_d=$(config.sigma_d), ρ_d=$(config.rho_d)")
    else
        if multivar_params !== nothing
            println("  ρ_r mean=$(round(mean(multivar_params.rho_r), digits=3))")
            println("  Σ diagonal mean=$(round(mean(diag(multivar_params.Sigma)), digits=6))")
        end
    end
    
    # Step 1: Solve network
    println("\n[Step 1] Solving baseline network (fixed structure)...")
    network = solve_network(params, return_firm_level=true)
    
    n_suppliers = sum(network.linkages .> 0)
    println("  Total suppliers: $n_suppliers")
    println("  Total downstream sales: $(round(sum(network.Y_r), digits=4))")
    
    N_rho_local = size(network.firm_expenditure_shares, 1)
    S_local = size(network.firm_expenditure_shares, 2)
    R_local = size(network.firm_expenditure_shares, 3)
    
    # Step 2: Draw exposures
    println("\n[Step 2] Drawing exposure to downstream (a_{di}^D)...")
    
    exposure_by_sector = nothing
    if input_folder !== nothing
        exposure_by_sector = load_exposure_distribution(input_folder)
    end
    
    a_d_D = draw_exposures(exposure_by_sector, S_local, R_local, N_rho_local; seed=config.seed)
    a_rdi_D, sales_to_downstream, total_sales = compute_a_rdi_D(network)
    
    println("  a_{di}^D stats: mean=$(round(mean(a_d_D), digits=3)), " *
            "std=$(round(std(a_d_D), digits=3))")
    
    # Step 3: Generate shocks
    println("\n[Step 3] Generating shocks ($model_name)...")
    
    if config.shock_model == MULTIVARIATE && multivar_params === nothing
        # Try to load from input_folder
        if input_folder !== nothing
            println("  Loading multivariate params from: $input_folder")
            multivar_params = load_multivariate_params(input_folder)
        else
            error("multivar_params required for MULTIVARIATE model")
        end
    end
    
    # Generate downstream shocks
    downstream_shocks = generate_downstream_shocks(
        R_local, config.T_periods, config; 
        multivar_params=multivar_params
    )
    println("  Downstream shock std (realized): $(round(std(downstream_shocks), digits=4))")
    
    # Other customer shocks
    other_shocks = generate_other_customer_shocks(
        N_rho_local, S_local, R_local, config.T_periods, sigma_sr; seed=config.seed
    )
    println("  Other customer shock std (overall): $(round(std(other_shocks), digits=4))")
    
    # Step 4: Simulate
    println("\n[Step 4] Simulating supplier sales...")
    
    d_ln_x_it, d_ln_x_dt, d_ln_x_drt, weighted_exposure_it = simulate_supplier_sales(
        network, a_d_D, downstream_shocks, other_shocks, config
    )
    
    println("  Supplier sales growth std: $(round(std(d_ln_x_it), digits=4))")
    println("  Aggregate downstream growth std: $(round(std(d_ln_x_dt), digits=4))")
    
    # Step 5: Regress
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
        "a_rdi_D" => a_rdi_D,
        "shock_model" => model_name
    )
end