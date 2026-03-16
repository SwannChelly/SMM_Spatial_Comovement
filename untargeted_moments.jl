##### Untargeted Moments: Three Shock Models with Correct Parameters #####
# Author: Swann Chelly (corrected by Claude)
# Purpose: Validate calibrated model with UNIVARIATE, MULTIVARIATE, and MULTIVARIATE_FE
#          Each model uses its OWN parameters estimated appropriately
#
# KEY CORRECTION: Different parameters for models with/without time FE
# ═══════════════════════════════════════════════════════════════════════════════
#
# MODEL 1: UNIVARIATE (no FE)
#   dln x_{rt} = ρ × dln x_{rt-1} + u_{rt}
#   Parameters from ORIGINAL data
#
# MODEL 2: MULTIVARIATE (no FE)
#   dln x_{rt} = ρ_r × dln x_{rt-1} + u_{rt}, u_t ~ N(0, Σ)
#   Parameters from ORIGINAL data (Σ includes aggregate variation)
#
# MODEL 3: MULTIVARIATE_FE
#   dln x_{rt} = μ_t + z_{rt}, z_{rt} = ρ_r × z_{rt-1} + ε_{rt}
#   Parameters from DEMEANED data (Σ only regional spillovers) + time FE
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

@enum ShockModel begin
    UNIVARIATE        # Same rho, i.i.d. across regions (NO time FE)
    MULTIVARIATE      # Region-specific rho_r, correlated Sigma (NO time FE)
    MULTIVARIATE_FE   # Region-specific rho_r, correlated Sigma + time FE
end

struct SimulationConfig
    T_periods::Int
    sigma_d::Float64
    rho_d::Float64
    seed::Int
    shock_model::ShockModel
    time_fe_mode::String  # "resample", "new", "empirical" (for MULTIVARIATE_FE only)
end

# Constructors
SimulationConfig(T_periods, sigma_d, rho_d, seed) = 
    SimulationConfig(T_periods, sigma_d, rho_d, seed, UNIVARIATE, "resample")

SimulationConfig(T_periods, sigma_d, rho_d, seed, shock_model::ShockModel) = 
    SimulationConfig(T_periods, sigma_d, rho_d, seed, shock_model, "resample")

const DEFAULT_CONFIG = SimulationConfig(36, 0.05, -0.15, 42, UNIVARIATE, "resample")

"""
    MultivariateShockParams

Parameters for multivariate shock models WITH dimension handling.

# Fields
- `rho_r`: Vector of region-specific persistence parameters (R_downstream,)
- `Sigma_innovation`: Innovation (conditional) covariance matrix (R_downstream, R_downstream)
- `Gamma_unconditional`: Unconditional covariance matrix (R_downstream, R_downstream)
- `active_regions`: Indices of regions with downstream activity (R_downstream,)
- `time_fe`: Empirical time FE μ_t (optional, only for MULTIVARIATE_FE)
- `time_fe_stats`: [mean, std] of time FE (optional)
- `model_type`: "multivariate" or "multivariate_fe"
"""
struct MultivariateShockParams
    rho_r::Vector{Float64}
    Sigma_innovation::Matrix{Float64}
    Gamma_unconditional::Matrix{Float64}
    active_regions::Vector{Int}
    time_fe::Union{Nothing, Vector{Float64}}
    time_fe_stats::Union{Nothing, Vector{Float64}}
    model_type::String  # "multivariate" or "multivariate_fe"
end

# Constructor without time FE (for MULTIVARIATE)
function MultivariateShockParams(
    rho_r::Vector{Float64}, 
    Sigma::Matrix{Float64}, 
    active_regions::Vector{Int},
    model_type::String = "multivariate"
)
    R_downstream = length(rho_r)
    Gamma = zeros(R_downstream, R_downstream)
    for i in 1:R_downstream, j in 1:R_downstream
        denom = 1 - rho_r[i] * rho_r[j]
        if abs(denom) > 1e-10 && abs(rho_r[i]) < 1 && abs(rho_r[j]) < 1
            Gamma[i, j] = Sigma[i, j] / denom
        else
            Gamma[i, j] = abs(Sigma[i, j]) > 1e-10 ? 1e6 : 0.0
        end
    end
    return MultivariateShockParams(rho_r, Sigma, Gamma, active_regions, nothing, nothing, model_type)
end

struct UnivariateShockParams
    rho::Float64
    sigma_innovation::Float64
    sigma_unconditional::Float64
end

function UnivariateShockParams(rho::Float64, sigma_unconditional::Float64)
    sigma_innovation = sigma_unconditional * sqrt(1 - rho^2)
    return UnivariateShockParams(rho, sigma_innovation, sigma_unconditional)
end

# ═══════════════════════════════════════════════════════════════════════════════
# DIMENSION HANDLING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    identify_active_downstream_regions(N_downstream_per_region)

Identify which regions have downstream activity.

Returns vector of indices (1-based) of regions with N_downstream_per_region > 0.
"""
function identify_active_downstream_regions(N_downstream_per_region::Vector{Float64})
    # All downstream regions are active in the new index space
    return collect(1:length(N_downstream_per_region))
end

# ═══════════════════════════════════════════════════════════════════════════════
# LOADING FUNCTIONS (MODEL-SPECIFIC)
# ═══════════════════════════════════════════════════════════════════════════════

function ensure_psd(A::Matrix{Float64})::Matrix{Float64}
    A_sym = (A + A') / 2
    eigen_decomp = eigen(Symmetric(A_sym))
    eigenvalues = eigen_decomp.values
    eigenvectors = eigen_decomp.vectors
    
    min_eig = minimum(eigenvalues)
    if min_eig >= 0
        return A_sym
    end
    
    @warn "Matrix not positive definite (min eigenvalue: $min_eig), adjusting"
    eigenvalues_fixed = max.(eigenvalues, 1e-10)
    A_psd = eigenvectors * Diagonal(eigenvalues_fixed) * eigenvectors'
    return (A_psd + A_psd') / 2
end

"""
    load_multivariate_params(input_folder, model_type)

Load multivariate shock parameters for specific model type.

# Arguments
- `input_folder`: Directory containing parameter files
- `model_type`: "multivariate" or "multivariate_fe"

# Files loaded depend on model_type:

MULTIVARIATE (no FE):
- rho_r_multivariate.npy: Region-specific persistence (R_downstream,)
- Sigma_innovations_multivariate.npy: Innovation covariance (R×R)
- Gamma_unconditional_multivariate.npy: Unconditional covariance (R×R) [optional]

MULTIVARIATE_FE:
- rho_r_fe.npy: Region-specific persistence (R_downstream,)
- Sigma_innovations_fe.npy: Innovation covariance (R×R) [CLEAN]
- Gamma_unconditional_fe.npy: Unconditional covariance (R×R) [optional]
- time_fe.npy: Time fixed effects μ_t (T,)
- time_fe_stats.npy: [mean, std] of μ_t

Requires global constant N_downstream_per_region to identify active regions.
"""
function load_multivariate_params(input_folder::String, model_type::String = "multivariate")
    # Determine file names based on model type
    if model_type == "multivariate"
        rho_file = "rho_r_multivariate.npy"
        sigma_file = "Sigma_innovations_multivariate.npy"
        gamma_file = "Gamma_unconditional_multivariate.npy"
        model_desc = "MULTIVARIATE (no FE)"
    elseif model_type == "multivariate_fe"
        rho_file = "rho_r_fe.npy"
        sigma_file = "Sigma_innovations_fe.npy"
        gamma_file = "Gamma_unconditional_fe.npy"
        model_desc = "MULTIVARIATE_FE"
    else
        error("Unknown model_type: $model_type. Use 'multivariate' or 'multivariate_fe'")
    end
    
    rho_path = joinpath(input_folder, rho_file)
    sigma_path = joinpath(input_folder, sigma_file)
    gamma_path = joinpath(input_folder, gamma_file)
    time_fe_path = joinpath(input_folder, "time_fe.npy")
    time_fe_stats_path = joinpath(input_folder, "time_fe_stats.npy")
    
    if !isfile(rho_path) || !isfile(sigma_path)
        error("$model_desc params not found in $input_folder")
    end
    
    rho_r = vec(NPZ.npzread(rho_path))
    Sigma = ensure_psd(NPZ.npzread(sigma_path))
    
    # Identify active downstream regions from global constant
    if !@isdefined(N_downstream_per_region)
        error("N_downstream_per_region not defined in global scope")
    end
    active_regions = identify_active_downstream_regions(N_downstream_per_region)
    
    # Validate dimensions
    R_downstream = length(active_regions)
    if length(rho_r) != R_downstream
        error("$model_desc: rho_r length ($(length(rho_r))) != R_downstream ($R_downstream)")
    end
    if size(Sigma, 1) != R_downstream || size(Sigma, 2) != R_downstream
        error("$model_desc: Sigma dimensions ($(size(Sigma))) != ($R_downstream, $R_downstream)")
    end
    
    # Load or compute Gamma
    if isfile(gamma_path)
        Gamma = NPZ.npzread(gamma_path)
        Gamma = replace(Gamma, Inf => 1e6, -Inf => -1e6, NaN => 0.0)
        Gamma = ensure_psd(Gamma)
    else
        R = length(rho_r)
        Gamma = zeros(R, R)
        for i in 1:R, j in 1:R
            denom = 1 - rho_r[i] * rho_r[j]
            if abs(denom) > 1e-10 && abs(rho_r[i]) < 1 && abs(rho_r[j]) < 1
                Gamma[i, j] = Sigma[i, j] / denom
            else
                Gamma[i, j] = abs(Sigma[i, j]) > 1e-10 ? 1e6 : 0.0
            end
        end
    end
    
    # Load time FE only for MULTIVARIATE_FE
    time_fe = nothing
    time_fe_stats = nothing
    if model_type == "multivariate_fe"
        if isfile(time_fe_path)
            time_fe = vec(NPZ.npzread(time_fe_path))
        else
            @warn "$model_desc: time_fe.npy not found"
        end
        if isfile(time_fe_stats_path)
            time_fe_stats = vec(NPZ.npzread(time_fe_stats_path))
        end
    end
    
    params = MultivariateShockParams(rho_r, Sigma, Gamma, active_regions, time_fe, time_fe_stats, model_type)
    
    println("  Loaded $model_desc parameters:")
    println("    R_local (total regions): $(length(N_downstream_per_region))")
    println("    R_downstream (active): $R_downstream")
    println("    Active region indices: $(active_regions[1:min(5, end)])$(length(active_regions) > 5 ? "..." : "")")
    println("    rho_r: mean=$(round(mean(rho_r), digits=3)), range=[$(round(minimum(rho_r), digits=3)), $(round(maximum(rho_r), digits=3))]")
    println("    Sigma diagonal: mean=$(round(mean(diag(Sigma)), digits=6))")
    if time_fe !== nothing
        println("    Time FE: T=$(length(time_fe)), mean=$(round(mean(time_fe), digits=4)), std=$(round(std(time_fe), digits=4))")
    end
    
    return params
end

"""
    load_univariate_params(input_folder)

Load univariate shock parameters (estimated on ORIGINAL data).

Files loaded:
- rho_univariate.npy: Pooled AR(1) coefficient
- sigma_unconditional_univariate.npy: Unconditional std
- sigma_innovation_univariate.npy: Innovation std [optional]
"""
function load_univariate_params(input_folder::String)
    rho_path = joinpath(input_folder, "rho_univariate.npy")
    sigma_uncond_path = joinpath(input_folder, "sigma_unconditional_univariate.npy")
    sigma_innov_path = joinpath(input_folder, "sigma_innovation_univariate.npy")
    
    if !isfile(rho_path)
        return nothing
    end
    
    rho = NPZ.npzread(rho_path)[1]
    sigma_unconditional = NPZ.npzread(sigma_uncond_path)[1]
    
    if isfile(sigma_innov_path)
        sigma_innovation = NPZ.npzread(sigma_innov_path)[1]
        params = UnivariateShockParams(rho, sigma_innovation, sigma_unconditional)
    else
        params = UnivariateShockParams(rho, sigma_unconditional)
    end
    
    println("  Loaded UNIVARIATE (no FE) parameters:")
    println("    rho: $(round(rho, digits=4))")
    println("    sigma_innovation: $(round(params.sigma_innovation, digits=6))")
    println("    sigma_unconditional: $(round(sigma_unconditional, digits=6))")
    
    return params
end

# ═══════════════════════════════════════════════════════════════════════════════
# EXPOSURE DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════

"""
Load empirical distribution of a_{di}^D from share_dist.csv
"""
function load_exposure_distribution(input_folder)
    csv_path = joinpath(input_folder, "share_dist.csv")
    
    if !isfile(csv_path)
        @warn "Exposure distribution file not found: $csv_path. Using uniform [0.1, 0.9]."
        return nothing
    end
    
    df = CSV.read(csv_path, DataFrame)
    if !("A129" in names(df)) || !("PartCa" in names(df))
        error("CSV must have columns 'A129' (sector) and 'PartCa' (exposure share)")
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
        println("    Sector $s: n=$(length(vals)), mean=$(round(mean(vals), digits=3)), std=$(round(std(vals), digits=3))")
    end
    
    return exposure_by_sector
end

"""
Draw a_{di}^D for each supplier from empirical distribution
"""
function draw_exposures(exposure_by_sector, S_local, R_local, N_rho_local; seed=42)
    Random.seed!(seed)
    a_d_D = zeros(N_rho_local, S_local, R_local)
    
    for s in 1:S_local, l in 1:R_local, rho in 1:N_rho_local
        if exposure_by_sector !== nothing && haskey(exposure_by_sector, s)
            a_d_D[rho, s, l] = rand(exposure_by_sector[s])
        else
            a_d_D[rho, s, l] = rand() * 0.8 + 0.1
        end
    end
    
    return a_d_D
end

"""
Compute a_{rdi}^D: share of supplier i's sales to downstream region r
"""
function compute_a_rdi_D(network)
    exp_shares = network.firm_expenditure_shares
    Y_r = network.Y_r
    mu = network.mu

    N_rho_local, S_local, R_upstream, R_down = size(exp_shares)
    total_cost = mu .* Y_r

    sales_to_downstream = zeros(N_rho_local, S_local, R_upstream, R_down)
    for r in 1:R_down
        if total_cost[r] > 1e-10
            sales_to_downstream[:, :, :, r] = exp_shares[:, :, :, r] .* total_cost[r]
        end
    end

    total_sales_to_downstream = sum(sales_to_downstream, dims=4)
    a_rdi_D = zeros(N_rho_local, S_local, R_upstream, R_down)

    for l in 1:R_upstream, s in 1:S_local, rho in 1:N_rho_local
        total = total_sales_to_downstream[rho, s, l, 1]
        if total > 1e-12
            for r in 1:R_down
                a_rdi_D[rho, s, l, r] = sales_to_downstream[rho, s, l, r] / total
            end
        end
    end

    return a_rdi_D, sales_to_downstream, total_sales_to_downstream[:,:,:,1]
end

# ═══════════════════════════════════════════════════════════════════════════════
# SHOCK GENERATION (THREE MODELS WITH DIMENSION HANDLING)
# ═══════════════════════════════════════════════════════════════════════════════

function generate_time_fe(T::Int, time_fe_empirical, time_fe_stats; mode::String = "resample")
    if mode == "empirical"
        if time_fe_empirical === nothing || length(time_fe_empirical) != T
            error("Cannot use mode='empirical': time_fe length mismatch or missing")
        end
        return copy(time_fe_empirical)
    elseif mode == "resample"
        if time_fe_empirical === nothing
            error("Cannot resample: time_fe_empirical not provided")
        end
        return rand(time_fe_empirical, T)
    elseif mode == "new"
        if time_fe_stats === nothing
            @warn "time_fe_stats not provided, using N(0, 0.01)"
            return randn(T) * 0.01
        end
        mu_mean, mu_std = time_fe_stats
        return randn(T) * mu_std .+ mu_mean
    else
        error("Unknown mode: $mode. Use 'empirical', 'resample', or 'new'")
    end
end

"""
    generate_downstream_shocks_univariate(R_local, T, config; uni_params, active_regions)

Generate regional downstream shocks using UNIVARIATE model (NO time FE).

Model:
    dln x_{r,t} = rho × dln x_{r,t-1} + u_{r,t}
    u_{r,t} ~ N(0, sigma_innovation^2) i.i.d. across regions

Returns: Matrix (R_local × T) of log shocks
"""
function generate_downstream_shocks_univariate(
    R_local::Int, 
    T::Int, 
    config::SimulationConfig; 
    uni_params=nothing,
    active_regions::Union{Nothing, Vector{Int}} = nothing
)
    Random.seed!(config.seed)
    
    if uni_params !== nothing
        rho = uni_params.rho
        sigma_innovation = uni_params.sigma_innovation
        sigma_unconditional = uni_params.sigma_unconditional
    else
        rho = config.rho_d
        sigma_unconditional = config.sigma_d
        sigma_innovation = sigma_unconditional * sqrt(1 - rho^2)
    end
    
    # Generate shocks directly for R_local (= R_downstream) regions
    # active_regions is now 1:R_downstream, so no embedding needed
    R_gen = active_regions !== nothing ? length(active_regions) : R_local

    innovations = randn(R_gen, T) * sigma_innovation
    z = zeros(R_gen, T)
    z[:, 1] = randn(R_gen) * sigma_unconditional

    for t in 2:T
        z[:, t] = rho * z[:, t-1] + innovations[:, t]
    end

    return z
end

"""
    generate_downstream_shocks_multivariate(R_local, T, params, seed)

Generate regional downstream shocks using MULTIVARIATE model (NO time FE).

Model:
    dln x_{r,t} = ρ_r × dln x_{r,t-1} + u_{r,t}
    u_t ~ N(0, Σ)

Returns: Matrix (R_local × T) of shocks
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
    active_regions = params.active_regions
    
    R_downstream = length(rho_r)
    
    # Validate dimensions
    if R_downstream != length(active_regions)
        error("rho_r length ($R_downstream) != active_regions length ($(length(active_regions)))")
    end
    if size(Sigma, 1) != R_downstream || size(Sigma, 2) != R_downstream
        error("Sigma size ($(size(Sigma))) != ($R_downstream, $R_downstream)")
    end
    
    # Cholesky decomposition
    L_innovation = cholesky(Symmetric(Sigma)).L
    L_unconditional = cholesky(Symmetric(Gamma)).L
    
    # Generate shocks directly for R_downstream regions (no embedding needed)
    standard_normals = randn(R_downstream, T)
    innovations = L_innovation * standard_normals

    z = zeros(R_downstream, T)
    z[:, 1] = L_unconditional * randn(R_downstream)

    for t in 2:T
        z[:, t] = rho_r .* z[:, t-1] + innovations[:, t]
    end

    return z
end

"""
    generate_downstream_shocks_multivariate_fe(R_local, T, params, seed, time_fe_mode)

Generate regional downstream shocks using MULTIVARIATE_FE model.

Full model:
    dln x_{r,t} = μ_t + z_{r,t}
    z_{r,t} = ρ_r × z_{r,t-1} + ε_{r,t}
    ε_t ~ N(0, Σ_clean)

Returns: Matrix (R_local × T) of TOTAL downstream growth
"""
function generate_downstream_shocks_multivariate_fe(
    R_local::Int, 
    T::Int, 
    params::MultivariateShockParams, 
    seed::Int,
    time_fe_mode::String
)
    # Step 1: Generate regional component (same as multivariate)
    z = generate_downstream_shocks_multivariate(R_local, T, params, seed)
    
    # Step 2: Generate time FE
    if params.time_fe === nothing
        @warn "No time FE available, using zero time FE"
        mu_t = zeros(T)
    else
        mu_t = generate_time_fe(T, params.time_fe, params.time_fe_stats; mode=time_fe_mode)
    end
    
    # Step 3: Combine: dln x_{r,t} = μ_t + z_{r,t}
    dln_x = z .+ mu_t'
    
    println("    Regional component std: $(round(std(z), digits=4))")
    println("    Time FE std: $(round(std(mu_t), digits=4))")
    println("    Total std: $(round(std(dln_x), digits=4))")
    
    return dln_x
end

"""
Generate other customer shocks using global sigma_sr matrix
"""
function generate_other_customer_shocks(N_rho_local::Int, S_local::Int, R_local::Int, T::Int, config::SimulationConfig)
    Random.seed!(config.seed + 1000)
    shocks = zeros(N_rho_local, S_local, R_local, T)
    
    use_sigma_sr = false
    try
        if @isdefined(sigma_sr) && sigma_sr !== nothing
            use_sigma_sr = true
        end
    catch
        use_sigma_sr = false
    end
    
    if use_sigma_sr
        for s in 1:S_local, l in 1:R_local
            sigma = sigma_sr[s, l]
            shocks[:, s, l, :] = randn(N_rho_local, T) * sigma
        end
        println("  Using global sigma_sr matrix (mean=$(round(mean(sigma_sr), digits=4)))")
    else
        default_sigma = 0.17
        shocks = randn(N_rho_local, S_local, R_local, T) * default_sigma
        println("  Using default sigma_other = $default_sigma (sigma_sr not found)")
    end
    
    return shocks
end

# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION (CORRECTED FOR R_downstream)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    simulate_supplier_sales(network, a_d_D, downstream_shocks, other_shocks, config, active_regions)

Simulate supplier sales growth.

Model:
    d ln x_{i,t} = a_{di}^D × Σ_r a_{rdi}^D × d ln x_{dr,t} + (1 - a_{di}^D) × d ln x_{oi,t}

Aggregate downstream growth computed using ONLY active downstream regions.
"""
function simulate_supplier_sales(network, a_d_D, downstream_shocks, other_shocks, config, active_regions)
    R_down, T = size(downstream_shocks)
    N_rho_local, S_local, R_upstream = size(a_d_D)

    Y_r = network.Y_r

    # Compute weights — Y_r is already R_downstream, all active
    w_r_d = Y_r ./ sum(Y_r)

    a_rdi_D, _, _ = compute_a_rdi_D(network)

    d_ln_x_drt = downstream_shocks

    # Aggregate downstream growth
    d_ln_x_dt = sum(w_r_d .* d_ln_x_drt, dims=1)[1, :]

    d_ln_x_it = zeros(N_rho_local, S_local, R_upstream, T)
    weighted_exposure_it = zeros(N_rho_local, S_local, R_upstream, T)

    for t in 1:T, l in 1:R_upstream, s in 1:S_local, rho in 1:N_rho_local
        exposure_term = sum(a_rdi_D[rho, s, l, r] * d_ln_x_drt[r, t] for r in 1:R_down)
        weighted_exposure_it[rho, s, l, t] = exposure_term

        a_d = a_d_D[rho, s, l]
        d_ln_x_it[rho, s, l, t] = a_d * exposure_term + (1 - a_d) * other_shocks[rho, s, l, t]
    end

    return d_ln_x_it, d_ln_x_dt, d_ln_x_drt, weighted_exposure_it
end

# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED PANEL CONSTRUCTION (ALL THREE MODELS)
# ═══════════════════════════════════════════════════════════════════════════════

"""
Build unified panel with all three shock models and run regressions on each.
"""
function build_unified_panel_and_regress(
    network, a_d_D, other_shocks, config,
    uni_shocks, multi_shocks, multi_fe_shocks, active_regions
)
    N_rho_local, S_local, R_local, T = size(other_shocks)
    linkages = network.linkages
    
    println("\n[Building unified panel with all three shock models]...")
    
    # Simulate for each model
    println("  Simulating UNIVARIATE...")
    d_ln_x_it_uni, d_ln_x_dt_uni, d_ln_x_drt_uni, weighted_exposure_it_uni = 
        simulate_supplier_sales(network, a_d_D, uni_shocks, other_shocks, config, active_regions)
    
    println("  Simulating MULTIVARIATE...")
    d_ln_x_it_multi, d_ln_x_dt_multi, d_ln_x_drt_multi, weighted_exposure_it_multi = 
        simulate_supplier_sales(network, a_d_D, multi_shocks, other_shocks, config, active_regions)
    
    println("  Simulating MULTIVARIATE_FE...")
    d_ln_x_it_multi_fe, d_ln_x_dt_multi_fe, d_ln_x_drt_multi_fe, weighted_exposure_it_multi_fe = 
        simulate_supplier_sales(network, a_d_D, multi_fe_shocks, other_shocks, config, active_regions)
    
    # Build unified panel
    println("  Building unified panel...")
    
    firm_ids = Int[]
    sector_ids = Int[]
    region_ids = Int[]
    period_ids = Int[]
    
    # Univariate columns
    d_ln_x_uni_vec = Float64[]
    downstream_growth_uni_vec = Float64[]
    weighted_exposure_uni_vec = Float64[]
    d_ln_x_no_other_uni_vec = Float64[]
    
    # Multivariate columns
    d_ln_x_multi_vec = Float64[]
    downstream_growth_multi_vec = Float64[]
    weighted_exposure_multi_vec = Float64[]
    d_ln_x_no_other_multi_vec = Float64[]
    
    # Multivariate FE columns
    d_ln_x_multi_fe_vec = Float64[]
    downstream_growth_multi_fe_vec = Float64[]
    weighted_exposure_multi_fe_vec = Float64[]
    d_ln_x_no_other_multi_fe_vec = Float64[]
    
    # Shared columns
    exposure_to_downstream = Float64[]
    
    firm_id = 0
    n_suppliers = 0
    
    for l in 1:R_local, s in 1:S_local, rho in 1:N_rho_local
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
            push!(exposure_to_downstream, a_d_D[rho, s, l])
            
            # Univariate
            push!(d_ln_x_uni_vec, d_ln_x_it_uni[rho, s, l, t])
            push!(downstream_growth_uni_vec, d_ln_x_dt_uni[t])
            push!(weighted_exposure_uni_vec, weighted_exposure_it_uni[rho, s, l, t])
            push!(d_ln_x_no_other_uni_vec, weighted_exposure_it_uni[rho, s, l, t])
            
            # Multivariate
            push!(d_ln_x_multi_vec, d_ln_x_it_multi[rho, s, l, t])
            push!(downstream_growth_multi_vec, d_ln_x_dt_multi[t])
            push!(weighted_exposure_multi_vec, weighted_exposure_it_multi[rho, s, l, t])
            push!(d_ln_x_no_other_multi_vec, weighted_exposure_it_multi[rho, s, l, t])
            
            # Multivariate FE
            push!(d_ln_x_multi_fe_vec, d_ln_x_it_multi_fe[rho, s, l, t])
            push!(downstream_growth_multi_fe_vec, d_ln_x_dt_multi_fe[t])
            push!(weighted_exposure_multi_fe_vec, weighted_exposure_it_multi_fe[rho, s, l, t])
            push!(d_ln_x_no_other_multi_fe_vec, weighted_exposure_it_multi_fe[rho, s, l, t])
        end
    end
    
    panel_df = DataFrame(
        firm_id = firm_ids,
        sector = sector_ids,
        region = region_ids,
        period = period_ids,
        a_d_D = exposure_to_downstream,
        
        # Univariate
        d_ln_x_uni = d_ln_x_uni_vec,
        downstream_growth_uni = downstream_growth_uni_vec,
        weighted_exposure_uni = weighted_exposure_uni_vec,
        d_ln_x_no_other_uni = d_ln_x_no_other_uni_vec,
        
        # Multivariate
        d_ln_x_multi = d_ln_x_multi_vec,
        downstream_growth_multi = downstream_growth_multi_vec,
        weighted_exposure_multi = weighted_exposure_multi_vec,
        d_ln_x_no_other_multi = d_ln_x_no_other_multi_vec,
        
        # Multivariate FE
        d_ln_x_multi_fe = d_ln_x_multi_fe_vec,
        downstream_growth_multi_fe = downstream_growth_multi_fe_vec,
        weighted_exposure_multi_fe = weighted_exposure_multi_fe_vec,
        d_ln_x_no_other_multi_fe = d_ln_x_no_other_multi_fe_vec
    )
    
    println("  Observations: $(nrow(panel_df))")
    println("  Unique suppliers: $n_suppliers")
    println("  Periods: $T")
    
    # Run regressions for each model
    println("\n" * "="^70)
    println("REGRESSION RESULTS (ALL THREE MODELS)")
    println("="^70)
    
    results = Dict()
    
    for (model_name, suffix) in [("UNIVARIATE", "uni"), ("MULTIVARIATE", "multi"), ("MULTIVARIATE_FE", "multi_fe")]
        println("\n" * "-"^60)
        println("MODEL: $model_name")
        println("-"^60)
        
        results[model_name] = Dict()
        
        # Reg 1
        try
            formula = @eval @formula($(Symbol("d_ln_x_$suffix")) ~ $(Symbol("downstream_growth_$suffix")) + fe(firm_id))
            reg1 = reg(panel_df, formula)
            
            beta = coef(reg1)[1]
            beta_se = stderror(reg1)[1]
            
            results[model_name]["reg1"] = Dict(
                "beta" => beta,
                "beta_se" => beta_se,
                "ci_lower" => beta - 1.96 * beta_se,
                "ci_upper" => beta + 1.96 * beta_se,
                "R2" => r2(reg1),
                "N" => nobs(reg1)
            )
            
            println("  Reg 1: β = $(round(beta, digits=4)) (se: $(round(beta_se, digits=4)))")
        catch e
            println("  Reg 1: ERROR - $e")
            results[model_name]["reg1"] = nothing
        end
        
        # Reg 2
        try
            formula = @eval @formula($(Symbol("d_ln_x_$suffix")) ~ a_d_D * $(Symbol("weighted_exposure_$suffix")) + fe(firm_id))
            reg2 = reg(panel_df, formula)
            
            beta = coef(reg2)[1]
            beta_se = stderror(reg2)[1]
            
            results[model_name]["reg2"] = Dict(
                "beta" => beta,
                "beta_se" => beta_se,
                "ci_lower" => beta - 1.96 * beta_se,
                "ci_upper" => beta + 1.96 * beta_se,
                "R2" => r2(reg2),
                "N" => nobs(reg2)
            )
            
            println("  Reg 2: β = $(round(beta, digits=4)) (se: $(round(beta_se, digits=4)))")
        catch e
            println("  Reg 2: ERROR - $e")
            results[model_name]["reg2"] = nothing
        end
        
        # Reg 3
        try
            formula = @eval @formula($(Symbol("d_ln_x_no_other_$suffix")) ~ $(Symbol("downstream_growth_$suffix")) + fe(firm_id))
            reg3 = reg(panel_df, formula)
            
            beta = coef(reg3)[1]
            beta_se = stderror(reg3)[1]
            
            results[model_name]["reg3"] = Dict(
                "beta" => beta,
                "beta_se" => beta_se,
                "ci_lower" => beta - 1.96 * beta_se,
                "ci_upper" => beta + 1.96 * beta_se,
                "R2" => r2(reg3),
                "N" => nobs(reg3)
            )
            
            println("  Reg 3: β = $(round(beta, digits=4)) (se: $(round(beta_se, digits=4)))")
        catch e
            println("  Reg 3: ERROR - $e")
            results[model_name]["reg3"] = nothing
        end
    end
    
    # Build regional sales DataFrame
    regional_sales_df = build_regional_sales_df(
        d_ln_x_drt_uni, d_ln_x_drt_multi, d_ln_x_drt_multi_fe
    )
    
    return panel_df, results, regional_sales_df
end



"""
Build regional sales DataFrame for all three shock models.

Returns DataFrame with columns:
- ze2010: Region identifier (1 to R_local)
- period: Time period (1 to T)
- d_ln_x_drt_uni: Regional sales change (UNIVARIATE model)
- d_ln_x_drt_multi: Regional sales change (MULTIVARIATE model)
- d_ln_x_drt_multi_fe: Regional sales change (MULTIVARIATE_FE model)
"""
function build_regional_sales_df(
    d_ln_x_drt_uni, d_ln_x_drt_multi, d_ln_x_drt_multi_fe
)
    R_local, T = size(d_ln_x_drt_uni)
    
    ze2010_vec = Int[]
    period_vec = Int[]
    d_ln_x_drt_uni_vec = Float64[]
    d_ln_x_drt_multi_vec = Float64[]
    d_ln_x_drt_multi_fe_vec = Float64[]
    
    for r in 1:R_local, t in 1:T
        push!(ze2010_vec, r)
        push!(period_vec, t)
        push!(d_ln_x_drt_uni_vec, d_ln_x_drt_uni[r, t])
        push!(d_ln_x_drt_multi_vec, d_ln_x_drt_multi[r, t])
        push!(d_ln_x_drt_multi_fe_vec, d_ln_x_drt_multi_fe[r, t])
    end
    
    regional_df = DataFrame(
        ze2010 = ze2010_vec,
        period = period_vec,
        d_ln_x_drt_uni = d_ln_x_drt_uni_vec,
        d_ln_x_drt_multi = d_ln_x_drt_multi_vec,
        d_ln_x_drt_multi_fe = d_ln_x_drt_multi_fe_vec
    )
    
    println("  Regional sales DataFrame: $(nrow(regional_df)) observations ($(R_local) regions × $T periods)")
    
    return regional_df
end

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN VALIDATION FUNCTION (ALL THREE MODELS WITH CORRECT PARAMETERS)
# ═══════════════════════════════════════════════════════════════════════════════

"""
Run untargeted validation with ALL THREE shock models using CORRECT parameters.

Each model uses its own parameters estimated appropriately:
- UNIVARIATE: From original data
- MULTIVARIATE: From original data
- MULTIVARIATE_FE: From demeaned data + time FE
"""
function run_untargeted_validation(
    params;
    config::SimulationConfig = DEFAULT_CONFIG,
    empirical = nothing,
    input_folder = nothing
)
    println("\n" * "="^70)
    println("UNTARGETED VALIDATION: ALL THREE SHOCK MODELS")
    println("WITH MODEL-SPECIFIC PARAMETERS")
    println("="^70)
    println("Config: T=$(config.T_periods), seed=$(config.seed)")
    println("Time FE mode: $(config.time_fe_mode)")
    
    # Step 1: Solve network
    println("\n[Step 1] Solving network...")
    network = solve_network(params, return_firm_level=true)
    
    N_rho_local = size(network.firm_expenditure_shares, 1)
    S_local = size(network.firm_expenditure_shares, 2)
    R_upstream = size(network.firm_expenditure_shares, 3)
    R_down = size(network.firm_expenditure_shares, 4)
    R_local = R_upstream  # For backward compatibility in exposure drawing
    
    println("  Suppliers: $(sum(network.linkages .> 0))")
    
    # Step 2: Draw exposures
    println("\n[Step 2] Drawing exposures (a_{di}^D)...")
    exposure_by_sector = input_folder !== nothing ? load_exposure_distribution(input_folder) : nothing
    a_d_D = draw_exposures(exposure_by_sector, S_local, R_local, N_rho_local; seed=config.seed)
    
    println("  a_{di}^D stats: mean=$(round(mean(a_d_D), digits=3)), std=$(round(std(a_d_D), digits=3))")
    
    # Step 3: Load MODEL-SPECIFIC parameters
    println("\n[Step 3] Loading MODEL-SPECIFIC shock parameters...")
    
    if input_folder === nothing
        error("input_folder required for loading parameters")
    end
    
    # Load UNIVARIATE params (from original data)
    println("\n  Loading UNIVARIATE parameters...")
    uni_params = load_univariate_params(input_folder)
    if uni_params === nothing
        @warn "UNIVARIATE parameters not found"
    end
    
    # Load MULTIVARIATE params (from original data)
    println("\n  Loading MULTIVARIATE parameters...")
    multi_params = load_multivariate_params(input_folder, "multivariate")
    
    # Load MULTIVARIATE_FE params (from demeaned data + time FE)
    println("\n  Loading MULTIVARIATE_FE parameters...")
    multi_fe_params = load_multivariate_params(input_folder, "multivariate_fe")
    
    # Get active regions (same for all multivariate models)
    active_regions = multi_params.active_regions
    println("\n  Active downstream regions: $(length(active_regions)) / $R_down")
    
    # Step 4: Generate shocks for ALL THREE models with THEIR OWN parameters
    println("\n[Step 4] Generating shocks (each model uses its own parameters)...")
    
    println("  Generating UNIVARIATE shocks (using univariate params)...")
    uni_shocks = generate_downstream_shocks_univariate(
        R_down, config.T_periods, config;
        uni_params=uni_params,
        active_regions=active_regions
    )
    println("    Std: $(round(std(uni_shocks), digits=4))")

    println("  Generating MULTIVARIATE shocks (using multivariate params)...")
    multi_shocks = generate_downstream_shocks_multivariate(R_down, config.T_periods, multi_params, config.seed)
    println("    Std: $(round(std(multi_shocks), digits=4))")

    println("  Generating MULTIVARIATE_FE shocks (using multivariate_fe params + time FE)...")
    multi_fe_shocks = generate_downstream_shocks_multivariate_fe(R_down, config.T_periods, multi_fe_params, config.seed, config.time_fe_mode)
    
    # Generate other customer shocks (same for all models)
    println("  Generating other customer shocks...")
    other_shocks = generate_other_customer_shocks(N_rho_local, S_local, R_upstream, config.T_periods, config)
    println("    Std: $(round(std(other_shocks), digits=4))")
    
    # Step 5: Build unified panel and run regressions
    println("\n[Step 5] Building unified panel and running regressions...")
    
    panel_df, reg_results, regional_sales_df = build_unified_panel_and_regress(
        network, a_d_D, other_shocks, config,
        uni_shocks, multi_shocks, multi_fe_shocks, active_regions
    )
    
    # Comparison table
    println("\n" * "="^70)
    println("COMPARISON TABLE")
    println("="^70)
    
    for reg_type in ["reg1", "reg2", "reg3"]
        reg_name = Dict("reg1" => "Reg 1 (downstream_growth)", 
                       "reg2" => "Reg 2 (weighted_exposure)",
                       "reg3" => "Reg 3 (no other customer)")[reg_type]
        
        println("\n$reg_name:")
        for model in ["UNIVARIATE", "MULTIVARIATE", "MULTIVARIATE_FE"]
            if reg_results[model][reg_type] !== nothing
                r = reg_results[model][reg_type]
                println("  $model: β = $(round(r["beta"], digits=4)) (se: $(round(r["beta_se"], digits=4)))")
            end
        end
    end
    
    if empirical !== nothing && haskey(empirical, "gamma_col5")
        println("\n" * "="^60)
        println("Empirical target: $(empirical["gamma_col5"])")
        for model in ["UNIVARIATE", "MULTIVARIATE", "MULTIVARIATE_FE"]
            if reg_results[model]["reg1"] !== nothing
                beta = reg_results[model]["reg1"]["beta"]
                rel_diff = abs(beta - empirical["gamma_col5"]) / abs(empirical["gamma_col5"]) * 100
                println("  $model: $(round(rel_diff, digits=1))% difference")
            end
        end
    end
    
    println("\n" * "="^70)
    println("KEY: Each model used its own parameters")
    println("  UNIVARIATE: ρ, σ from ORIGINAL data")
    println("  MULTIVARIATE: ρ_r, Σ from ORIGINAL data (includes aggregate)")
    println("  MULTIVARIATE_FE: ρ_r, Σ_clean from DEMEANED data + μ_t")
    println("="^70)
    
    return Dict(
        "network" => network,
        "panel_df" => panel_df,
        "regional_sales_df" => regional_sales_df,  # NEW
        "regression_results" => reg_results,
        "config" => config,
        "shocks" => Dict(
            "univariate" => uni_shocks,
            "multivariate" => multi_shocks,
            "multivariate_fe" => multi_fe_shocks,
            "other" => other_shocks
        ),
        "a_d_D" => a_d_D,
        "params" => Dict(
            "univariate" => uni_params,
            "multivariate" => multi_params,
            "multivariate_fe" => multi_fe_params
        ),
        "active_regions" => active_regions
    )
end