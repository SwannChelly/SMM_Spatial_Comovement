##### Untargeted Moments: Reproduce Table 2 Regression (CORRECTED) #####
# Author: Swann Chelly (corrected by Claude)
# Purpose: Validate calibrated model by reproducing Table 2 regression
#
# CORRECTIONS MADE:
# 1. Fixed undefined `sigma_sr` variable
# 2. Improved multivariate initialization to use FULL unconditional covariance
# 3. Added univariate parameter loading from Python exports
# 4. Clarified conditional vs unconditional variance terminology
#
# ═══════════════════════════════════════════════════════════════════════════════
# VARIANCE TERMINOLOGY
# ═══════════════════════════════════════════════════════════════════════════════
#
# CONDITIONAL (Innovation) Variance:
#   Var(u_t) = Sigma                    (what we estimate from residuals)
#
# UNCONDITIONAL (Stationary) Variance:
#   Var(z_t) = Gamma_0                  (long-run variance when process is stationary)
#   [Gamma_0]_{rs} = Sigma_{rs} / (1 - rho_r * rho_s)
#
# For simulation:
#   - Use INNOVATION variance (Sigma) for the noise term
#   - Use UNCONDITIONAL variance (Gamma_0) for initialization
#
# ═══════════════════════════════════════════════════════════════════════════════
# SHOCK MODELS
# ═══════════════════════════════════════════════════════════════════════════════
#
# MODEL 1: UNIVARIATE (Original)
#   z_{r,t} = rho × z_{r,t-1} + u_{r,t}
#   u_{r,t} ~ N(0, sigma_innovation^2) i.i.d. across regions
#   - Same persistence rho for all regions
#   - Independent shocks across regions
#
# MODEL 2: MULTIVARIATE (New)
#   z_{r,t} = rho_r × z_{r,t-1} + u_{r,t}
#   u_t ~ N(0, Sigma_innovation)
#   - Region-specific persistence rho_r
#   - Correlated innovations across regions via Sigma
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
    UNIVARIATE   # Original: same rho, i.i.d. across regions
    MULTIVARIATE # New: region-specific rho_r, correlated Sigma
end


"""
    SimulationConfig

Configuration for the untargeted moments simulation.

# Fields
- `T_periods`: Number of time periods (quarters)
- `sigma_d`: Unconditional std of downstream AR(1) (univariate only)
- `rho_d`: AR(1) persistence for downstream (univariate only)
- `seed`: Random seed
- `shock_model`: Which shock model to use (UNIVARIATE or MULTIVARIATE)

Note: sigma_sr (other customer shocks) is loaded as a global constant in main_pso.jl
"""
struct SimulationConfig
    T_periods::Int
    sigma_d::Float64        # UNCONDITIONAL std (used only for UNIVARIATE)
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
- `Sigma_innovation`: Innovation (conditional) covariance matrix (R, R)
- `Gamma_unconditional`: Unconditional covariance matrix (R, R) - optional, computed if not provided
"""
struct MultivariateShockParams
    rho_r::Vector{Float64}
    Sigma_innovation::Matrix{Float64}
    Gamma_unconditional::Matrix{Float64}
end

# Constructor that computes Gamma_unconditional from Sigma and rho_r
function MultivariateShockParams(rho_r::Vector{Float64}, Sigma::Matrix{Float64})
    R = length(rho_r)
    Gamma = zeros(R, R)
    for i in 1:R
        for j in 1:R
            denom = 1 - rho_r[i] * rho_r[j]
            if abs(denom) > 1e-10 && abs(rho_r[i]) < 1 && abs(rho_r[j]) < 1
                Gamma[i, j] = Sigma[i, j] / denom
            else
                Gamma[i, j] = abs(Sigma[i, j]) > 1e-10 ? 1e6 : 0.0
            end
        end
    end
    return MultivariateShockParams(rho_r, Sigma, Gamma)
end


"""
    UnivariateShockParams

Parameters for the univariate shock model (loaded from Python).

# Fields
- `rho`: Pooled AR(1) coefficient
- `sigma_innovation`: Innovation std (conditional)
- `sigma_unconditional`: Unconditional std (stationary)
"""
struct UnivariateShockParams
    rho::Float64
    sigma_innovation::Float64
    sigma_unconditional::Float64
end

# Constructor that computes innovation std from unconditional std
function UnivariateShockParams(rho::Float64, sigma_unconditional::Float64)
    sigma_innovation = sigma_unconditional * sqrt(1 - rho^2)
    return UnivariateShockParams(rho, sigma_innovation, sigma_unconditional)
end


"""
    load_multivariate_params(input_folder)

Load multivariate shock parameters from .npy files exported by Python.

Expected files (from Python export_params_for_julia):
- rho_r.npy: Region-specific persistence (R,)
- Sigma_innovations.npy: Innovation covariance matrix (R, R)
- Gamma_unconditional.npy: Unconditional covariance matrix (R, R) [optional]
"""
function load_multivariate_params(input_folder::String)
    rho_path = joinpath(input_folder, "rho_r.npy")
    sigma_path = joinpath(input_folder, "Sigma_innovations.npy")
    gamma_path = joinpath(input_folder, "Gamma_unconditional.npy")
    
    if !isfile(rho_path)
        error("Multivariate params file not found: $rho_path")
    end
    if !isfile(sigma_path)
        error("Multivariate params file not found: $sigma_path")
    end
    
    rho_r = vec(NPZ.npzread(rho_path))
    Sigma = NPZ.npzread(sigma_path)
    
    # Ensure Sigma is positive definite
    Sigma = ensure_psd(Sigma)
    
    # Load or compute Gamma_unconditional
    if isfile(gamma_path)
        Gamma = NPZ.npzread(gamma_path)
        # Replace inf/nan with large values
        Gamma = replace(Gamma, Inf => 1e6, -Inf => -1e6, NaN => 0.0)
        Gamma = ensure_psd(Gamma)
        params = MultivariateShockParams(rho_r, Sigma, Gamma)
    else
        # Compute from Sigma and rho_r
        params = MultivariateShockParams(rho_r, Sigma)
    end
    
    println("  Loaded multivariate shock parameters:")
    println("    R = $(length(rho_r))")
    println("    rho_r range: [$(round(minimum(rho_r), digits=3)), $(round(maximum(rho_r), digits=3))]")
    println("    rho_r mean: $(round(mean(rho_r), digits=3))")
    println("    Sigma (innovation) diagonal mean: $(round(mean(diag(Sigma)), digits=6))")
    println("    Gamma (unconditional) diagonal mean: $(round(mean(diag(params.Gamma_unconditional)), digits=6))")
    
    return params
end


"""
    load_univariate_params(input_folder)

Load univariate shock parameters from .npy files exported by Python.

Expected files:
- rho_univariate.npy: Pooled AR(1) coefficient
- sigma_unconditional_univariate.npy: Unconditional std
- sigma_innovation_univariate.npy: Innovation std [optional]
"""
function load_univariate_params(input_folder::String)
    rho_path = joinpath(input_folder, "rho_univariate.npy")
    sigma_uncond_path = joinpath(input_folder, "sigma_unconditional_univariate.npy")
    sigma_innov_path = joinpath(input_folder, "sigma_innovation_univariate.npy")
    
    if !isfile(rho_path)
        return nothing  # Univariate params not exported
    end
    
    rho = NPZ.npzread(rho_path)[1]
    sigma_unconditional = NPZ.npzread(sigma_uncond_path)[1]
    
    if isfile(sigma_innov_path)
        sigma_innovation = NPZ.npzread(sigma_innov_path)[1]
        params = UnivariateShockParams(rho, sigma_innovation, sigma_unconditional)
    else
        params = UnivariateShockParams(rho, sigma_unconditional)
    end
    
    println("  Loaded univariate shock parameters:")
    println("    rho: $(round(rho, digits=4))")
    println("    sigma_innovation: $(round(params.sigma_innovation, digits=6))")
    println("    sigma_unconditional: $(round(sigma_unconditional, digits=6))")
    
    return params
end


"""
    ensure_psd(A)

Ensure matrix is positive semi-definite by adjusting eigenvalues if needed.
"""
function ensure_psd(A::Matrix{Float64})::Matrix{Float64}
    # Make symmetric
    A_sym = (A + A') / 2
    
    # Eigendecomposition
    eigen_decomp = eigen(Symmetric(A_sym))
    eigenvalues = eigen_decomp.values
    eigenvectors = eigen_decomp.vectors
    
    # Check if already PSD
    min_eig = minimum(eigenvalues)
    if min_eig >= 0
        return A_sym
    end
    
    # Fix negative eigenvalues
    @warn "Matrix not positive definite (min eigenvalue: $min_eig), adjusting"
    eigenvalues_fixed = max.(eigenvalues, 1e-10)
    
    # Reconstruct matrix
    A_psd = eigenvectors * Diagonal(eigenvalues_fixed) * eigenvectors'
    
    return (A_psd + A_psd') / 2  # Ensure symmetry
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
    generate_downstream_shocks_univariate(R_local, T, config; uni_params=nothing)

Generate regional downstream shocks using UNIVARIATE model.

Model:
    z_{r,t} = rho × z_{r,t-1} + u_{r,t}
    u_{r,t} ~ N(0, sigma_innovation^2) i.i.d. across regions

Initialization:
    z_{r,0} ~ N(0, sigma_unconditional^2)  (stationary distribution)

If `uni_params` is provided (from Python), uses those values.
Otherwise uses config.sigma_d as UNCONDITIONAL std (backward compatible).

Returns: Matrix (R × T) of log shocks z_{r,t}
"""
function generate_downstream_shocks_univariate(
    R_local::Int, 
    T::Int, 
    config::SimulationConfig;
    uni_params::Union{Nothing, UnivariateShockParams} = nothing
)
    Random.seed!(config.seed)
    
    if uni_params !== nothing
        # Use params loaded from Python
        rho = uni_params.rho
        sigma_innovation = uni_params.sigma_innovation
        sigma_unconditional = uni_params.sigma_unconditional
    else
        # Backward compatible: config.sigma_d is UNCONDITIONAL std
        rho = config.rho_d
        sigma_unconditional = config.sigma_d
        sigma_innovation = sigma_unconditional * sqrt(1 - rho^2)
    end
    
    # i.i.d. innovations with INNOVATION std
    innovations = randn(R_local, T) * sigma_innovation
    
    # Initialize from UNCONDITIONAL (stationary) distribution
    z = zeros(R_local, T)
    z[:, 1] = randn(R_local) * sigma_unconditional
    
    for t in 2:T
        z[:, t] = rho * z[:, t-1] + innovations[:, t]
    end
    
    return z
end


"""
    generate_downstream_shocks_multivariate(R_local, T, params, seed)

Generate regional downstream shocks using MULTIVARIATE model.

Model:
    z_{r,t} = rho_r × z_{r,t-1} + u_{r,t}
    u_t ~ N(0, Sigma_innovation)

Initialization (CORRECTED):
    z_0 ~ N(0, Gamma_unconditional)
    
Where Gamma_unconditional is the FULL unconditional covariance matrix,
not just the diagonal. This properly captures cross-regional correlation
at initialization.

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
    Sigma = params.Sigma_innovation
    Gamma = params.Gamma_unconditional
    
    # Validate dimensions
    if length(rho_r) != R_local
        error("rho_r length ($(length(rho_r))) != R ($R_local)")
    end
    if size(Sigma, 1) != R_local || size(Sigma, 2) != R_local
        error("Sigma size ($(size(Sigma))) != ($R_local, $R_local)")
    end
    
    # Cholesky decomposition of INNOVATION covariance for simulation
    L_innovation = cholesky(Symmetric(Sigma)).L
    
    # Cholesky decomposition of UNCONDITIONAL covariance for initialization
    L_unconditional = cholesky(Symmetric(Gamma)).L
    
    # Draw correlated innovations: u_t = L_innovation × eps_t where eps ~ N(0, I)
    standard_normals = randn(R_local, T)
    innovations = L_innovation * standard_normals  # (R × T)
    
    # Initialize from FULL unconditional distribution N(0, Gamma)
    # This is the CORRECTED version - uses full covariance, not just diagonal
    z = zeros(R_local, T)
    z[:, 1] = L_unconditional * randn(R_local)
    
    for t in 2:T
        # z_{r,t} = rho_r × z_{r,t-1} + u_{r,t}
        z[:, t] = rho_r .* z[:, t-1] + innovations[:, t]
    end
    
    return z
end


"""
    generate_downstream_shocks(R_local, T, config; multivar_params=nothing, univar_params=nothing)

Main dispatch function for generating downstream shocks.

Uses config.shock_model to determine which model to use.
"""
function generate_downstream_shocks(
    R_local::Int, 
    T::Int, 
    config::SimulationConfig;
    multivar_params::Union{Nothing, MultivariateShockParams} = nothing,
    univar_params::Union{Nothing, UnivariateShockParams} = nothing
)
    if config.shock_model == UNIVARIATE
        return generate_downstream_shocks_univariate(R_local, T, config; uni_params=univar_params)
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
    generate_other_customer_shocks(N_rho_local, S_local, R_local, T, config)

Generate i.i.d. shocks for "other customers" of each supplier.

Uses the global `sigma_sr` matrix (S × R) loaded in main_pso.jl from sigma_sr.npy.
If sigma_sr is not defined or is nothing, falls back to default σ = 0.17.

The shocks are sector-region specific: each (s, l) pair has its own std from sigma_sr[s, l].
"""
function generate_other_customer_shocks(
    N_rho_local::Int, 
    S_local::Int, 
    R_local::Int, 
    T::Int, 
    config::SimulationConfig
)
    Random.seed!(config.seed + 1000)
    
    shocks = zeros(N_rho_local, S_local, R_local, T)
    
    # Check if global sigma_sr is defined and available
    use_sigma_sr = false
    try
        if @isdefined(sigma_sr) && sigma_sr !== nothing
            use_sigma_sr = true
        end
    catch
        use_sigma_sr = false
    end
    
    if use_sigma_sr
        # Use sector-region specific std from global sigma_sr
        for s in 1:S_local
            for l in 1:R_local
                sigma = sigma_sr[s, l]
                shocks[:, s, l, :] = randn(N_rho_local, T) * sigma
            end
        end
        println("  Using global sigma_sr matrix (mean=$(round(mean(sigma_sr), digits=4)))")
    else
        # Fallback to default
        default_sigma = 0.17
        shocks = randn(N_rho_local, S_local, R_local, T) * default_sigma
        println("  Using default sigma_other = $default_sigma (sigma_sr not found)")
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
    println("  d ln x_{i,t} = alpha_i + beta × downstream_growth + eps")
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
        
        println("  beta (downstream_growth): $(round(beta, digits=4)) " *
                "(se: $(round(beta_se, digits=4)))")
        println("  95% CI: [$(round(ci_lower, digits=4)), $(round(ci_upper, digits=4))]")
        println("  R2: $(round(results["reg1"]["R2"], digits=4))")
        println("  N:  $(results["reg1"]["N"])")
    catch e
        println("  ERROR: $e")
        results["reg1"] = nothing
    end
    
    # Specification 2
    println("\nSpecification 2: Firm FE (weighted exposure)")
    println("  d ln x_{i,t} = alpha_i + beta × weighted_exposure + eps")
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
        
        println("  beta (weighted_exposure): $(round(beta, digits=4)) " *
                "(se: $(round(beta_se, digits=4)))")
        println("  95% CI: [$(round(ci_lower, digits=4)), $(round(ci_upper, digits=4))]")
        println("  R2: $(round(results["reg2"]["R2"], digits=4))")
        println("  N:  $(results["reg2"]["N"])")
    catch e
        println("  ERROR: $e")
        results["reg2"] = nothing
    end
    
    # Specification 3
    println("\nSpecification 3: No other customer (a_d_D = 1)")
    println("  d ln x_{i,t}^{no other} = alpha_i + beta × downstream_growth + eps")
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
        
        println("  beta (downstream_growth): $(round(beta, digits=4)) " *
                "(se: $(round(beta_se, digits=4)))")
        println("  95% CI: [$(round(ci_lower, digits=4)), $(round(ci_upper, digits=4))]")
        println("  R2: $(round(results["reg3"]["R2"], digits=4))")
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
- `input_folder`: Path to folder containing share_dist.csv and shock params
- `multivar_params`: MultivariateShockParams (loaded from input_folder if not provided)
- `univar_params`: UnivariateShockParams (loaded from input_folder if not provided)

# Returns
Dict with network, panel_df, regression_results, config, shocks, exposures
"""
function run_untargeted_validation(
    params;
    config::SimulationConfig = DEFAULT_CONFIG,
    empirical = nothing,
    input_folder = nothing,
    multivar_params::Union{Nothing, MultivariateShockParams} = nothing,
    univar_params::Union{Nothing, UnivariateShockParams} = nothing
)
    
    model_name = config.shock_model == UNIVARIATE ? "UNIVARIATE" : "MULTIVARIATE"
    
    println("\n" * "="^70)
    println("UNTARGETED MOMENT VALIDATION: Table 2 Regression")
    println("Shock Model: $model_name")
    println("="^70)
    println("Config: T=$(config.T_periods), seed=$(config.seed)")
    
    if config.shock_model == UNIVARIATE
        println("  sigma_d (unconditional)=$(config.sigma_d), rho_d=$(config.rho_d)")
    else
        if multivar_params !== nothing
            println("  rho_r mean=$(round(mean(multivar_params.rho_r), digits=3))")
            println("  Sigma (innovation) diagonal mean=$(round(mean(diag(multivar_params.Sigma_innovation)), digits=6))")
            println("  Gamma (unconditional) diagonal mean=$(round(mean(diag(multivar_params.Gamma_unconditional)), digits=6))")
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
    
    # Step 3: Load shock parameters if needed
    println("\n[Step 3] Loading/preparing shock parameters...")
    
    if config.shock_model == MULTIVARIATE && multivar_params === nothing
        if input_folder !== nothing
            println("  Loading multivariate params from: $input_folder")
            multivar_params = load_multivariate_params(input_folder)
        else
            error("multivar_params required for MULTIVARIATE model")
        end
    end
    
    if config.shock_model == UNIVARIATE && univar_params === nothing && input_folder !== nothing
        println("  Attempting to load univariate params from: $input_folder")
        univar_params = load_univariate_params(input_folder)
        if univar_params === nothing
            println("  (Not found, using config values)")
        end
    end
    
    # Step 4: Generate shocks
    println("\n[Step 4] Generating shocks ($model_name)...")
    
    downstream_shocks = generate_downstream_shocks(
        R_local, config.T_periods, config; 
        multivar_params=multivar_params,
        univar_params=univar_params
    )
    println("  Downstream shock std (realized): $(round(std(downstream_shocks), digits=4))")
    
    # Other customer shocks - uses global sigma_sr matrix from main_pso.jl
    other_shocks = generate_other_customer_shocks(
        N_rho_local, S_local, R_local, config.T_periods, config
    )
    println("  Other customer shock std (realized): $(round(std(other_shocks), digits=4))")
    
    # Step 5: Simulate
    println("\n[Step 5] Simulating supplier sales...")
    
    d_ln_x_it, d_ln_x_dt, d_ln_x_drt, weighted_exposure_it = simulate_supplier_sales(
        network, a_d_D, downstream_shocks, other_shocks, config
    )
    
    println("  Supplier sales growth std: $(round(std(d_ln_x_it), digits=4))")
    println("  Aggregate downstream growth std: $(round(std(d_ln_x_dt), digits=4))")
    
    # Step 6: Regress
    println("\n[Step 6] Running regressions...")
    
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
        "shock_model" => model_name,
        "multivar_params" => multivar_params,
        "univar_params" => univar_params
    )
end