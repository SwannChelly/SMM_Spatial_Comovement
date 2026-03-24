##### SMM for Spatial Comovement #####
# Author: Swann Chelly 
# 
# This code implements the structural model from the paper "Spatial Comovements"
# 
# ═══════════════════════════════════════════════════════════════════════════════
# NOTATION (following the paper, Appendix B)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Indices:
#   r, r'    : regions (r for downstream buyer, r' or l for upstream seller)
#   s        : upstream sector
#   ρ (rho)  : variety within a sector
#
# Parameters:
#   ε (epsilon)   : demand elasticity for final goods (negative, e.g., -3.67)
#   Ω^L (Omega_L) : labor share parameter in production
#   Ω^s (Omega_s) : sectoral input share for sector s
#   λ (lambda)    : elasticity of substitution between labor and intermediates
#   ν (nu)        : elasticity of substitution across sectors
#   ν_s (nu_s)    : elasticity of substitution across varieties within sector s
#   θ (theta)     : Fréchet shape parameter (productivity dispersion)
#   T_{sr}        : Fréchet scale (average productivity in sector s, region r)
#   A_r           : downstream firm productivity in region r
#   τ_{r'rs}      : iceberg trade cost from r' to r for sector s
#   δ_r           : demand shifter for downstream region r
#
# Prices:
#   w_r           : wage in region r
#   w_{rs}        : wage in sector s, region r
#   p_{ρsr'}      : price of variety ρ from sector s, region r'
#   P_{sr}        : price index for sector s inputs in region r
#   P_r           : aggregate intermediate input price index in region r
#   c_r           : unit cost of downstream firm in region r (before productivity)
#   c̃_r = c_r/A_r : unit cost after productivity adjustment
#
# Key relationships (from paper p.24-26):
#   Price:     p_r = c̃_r / μ  where μ = (ε-1)/ε (inverse markup)
#   Demand:    q_r = (p_r/P)^{ε-1} · (E/P) · δ_r
#   Sales:     Y_r = p_r · q_r = p_r^ε · P^{-ε} · E · δ_r
#   Cost:      C_r = c̃_r · q_r = μ · Y_r  (total cost = inverse markup × revenue)
#   Inputs:    Input expenditure_r = (1 - Ω^L) · C_r = (1 - Ω^L) · μ · Y_r
#
# Trade flows:
#   X_{r'rs} = γ_{r'rs} · (input expenditure on sector s in region r)
#   where γ_{r'rs} is the sourcing share from r' for inputs s used in r
#
# ═══════════════════════════════════════════════════════════════════════════════

##################### Packages ###################

using Distributed
using SparseArrays
using Distributions
using Random
using NPZ
using LinearAlgebra
using QuasiMonteCarlo
using DataFrames
using Optim
using CSV
using FixedEffectModels, RDatasets, CategoricalArrays


###### Testing environment #####
test = false 
if test
    industry = "aero"
    input_folder = "./baseline_"*industry
    
    coefs = CSV.read(joinpath(input_folder,"stats.csv"), DataFrame)
    distances = NPZ.npzread(joinpath(input_folder, "distances.npy"))
    w_rs = NPZ.npzread(joinpath(input_folder, "w_rs.npy"))
    N_downstream_per_region = NPZ.npzread(joinpath(input_folder,"N_downstream_per_region.npy"))
    filter_N_upstream = NPZ.npzread(joinpath(input_folder,"filter_N_upstream.npy"))
    S, R = size(filter_N_upstream)
    R_downstream = size(N_downstream_per_region[N_downstream_per_region.!=0])[1]
    delta_r = ones(R)

    N_rho = 50
    agg_labor_share = coefs[2,"value"]
    agg_industry_share = NPZ.npzread(joinpath(input_folder,"input_share.npy"))
    epsilon = coefs[1,"value"]
    lambda = 0.5
    nu = 0.2
    nu_s = ones(S).*2.5
    theta = 1.768

    w_rs = NPZ.npzread(joinpath(input_folder, "w_rs.npy"))
    regional_wages = NPZ.npzread(joinpath(input_folder, "regional_wages.npy"))

    domestic_share = NPZ.npzread(joinpath(input_folder,"domestic_share.npy"))
    emp_gamma_ls = (NPZ.npzread(joinpath(input_folder,"emp_gamma_ls.npy"))')
    emp_pi_r = NPZ.npzread(joinpath(input_folder,"emp_pi_r.npy"))[2:end]
    reg_coef = NPZ.npzread(joinpath(input_folder,"reg_coef.npy"))
    empirical_moments = [[agg_labor_share],agg_industry_share[2:end],emp_gamma_ls,reg_coef,emp_pi_r]
    empirical_moments = vcat([vec(empirical_moments[i]) for i in 1:(length(empirical_moments)-1)]...)   
    empirical_moments = reshape(empirical_moments,1,length(empirical_moments))

    function distance_bin(d)
        if 20 < d <= 50
            return 1
        elseif 50 < d <= 100
            return 2
        elseif 100 < d <= 150
            return 3
        elseif 150 < d <= 200
            return 4
        elseif d > 200
            return 5
        else
            return 0
        end
    end

    DistBin = Array{Int}(undef, R, R)
    for i in 1:R, j in 1:R
        DistBin[i,j] = distance_bin(distances[i,j])
    end
end


##################### Helper Functions ###################

"""
    unpack_params(params) -> (β, Ω^L, Ω^s, A, T)

Unpack parameter vector into model components (paper notation).

Returns:
- β (beta): Trade cost parameters for distance bins [5 elements]
- Ω^L (Omega_L): Labor share in production [scalar]
- Ω^s (Omega_s): Sectoral input shares [S elements, normalized to sum to 1]
- A: Downstream firm productivity by region [R_downstream elements]
- T: Fréchet scale parameters [S × R elements]
"""
function unpack_params(params)
    beta = params[1:N_beta]
    Omega_L = params[N_beta + 1]
    Omega_s = params[(N_beta + 2):(N_beta + 1 + S)] / sum(params[(N_beta + 2):(N_beta + 1 + S)])
    A = params[(N_beta + S + 2):(N_beta + R_downstream + S + 1)]
    T = params[(N_beta + R_downstream + S + 2):end]
    
    return beta, Omega_L, Omega_s, A, T
end


"""
    build_tau(beta) -> τ[r', r, s]

Build iceberg trade cost matrix from distance bin coefficients.

τ_{r'rs} = 1 + β_b  where b = DistBin[r', r]

Returns array of size (R, R, S).
"""
function build_tau(beta)
    tau = ones(R, R, S)
    for r_prime in 1:R, r in 1:R
        b = DistBin[r_prime, r]
        if b > 0
            for s in 1:S
                tau[r_prime, r, s] += beta[b]
            end
        end
    end
    return tau
end


##################### Core Model: Network Solution ###################

"""
    solve_network(params; return_firm_level=false)

Solve the production network equilibrium for given parameters.

This function:
1. Draws upstream firm productivities from Fréchet distribution
2. For each downstream region, finds lowest-cost suppliers (Ricardian selection)
3. Computes nested CES price indices
4. Calculates downstream sales and trade flows

# Arguments
- `params`: Parameter vector [β, Ω^L, Ω^s, A, T]
- `return_firm_level`: If true, return firm-level data for untargeted validation

# Returns (NamedTuple):
- `X_lrs`: Trade flows X_{r'rs} from upstream (r',s) to downstream r [R × R × S]
- `c_tilde_r`: Unit costs c̃_r = c_r/A_r of downstream firms [R]
- `Y_r`: Downstream sales (revenue) by region [R]
- `linkages`: Firm-level supplier indicator [N_rho × S × R]
- `z`: Firm productivities [N_rho × R × S]
- `closest_plant_dist`: Distance to nearest downstream plant [R]

If return_firm_level=true, additionally returns:
- `firm_expenditure_shares`: Expenditure share from (ρ,s,l) to r [N_rho × S × R × R]
- `mu`: Inverse markup μ = (ε-1)/ε
- `P`: Aggregate downstream price index
"""
function solve_network(params; return_firm_level=false)
    
    Random.seed!(50)  # For reproducibility
    
    # ─────────────────────────────────────────────────────────────────────────
    # Unpack parameters (paper notation)
    # ─────────────────────────────────────────────────────────────────────────
    beta, Omega_L, Omega_s_vec, A_vec, T_vec = unpack_params(params)
    
    # Build trade cost matrix τ_{r'rs}
    tau = build_tau(beta)
    
    # Expand productivity A_r to full R vector
    A_r = ones(R)
    A_r[N_downstream_per_region .!= 0] = A_vec
    
    # Reshape for broadcasting
    Omega_s = reshape(Omega_s_vec, 1, S)  # (1, S)
    nu_s_mat = reshape(nu_s, 1, S)        # (1, S)
    T = reshape(T_vec, S, R)              # (S, R)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Draw upstream firm productivities: z_{ρsr} ~ Fréchet(θ, T_{sr}^{1/θ})
    # ─────────────────────────────────────────────────────────────────────────
    frechet_dist = Frechet.(theta, T.^(1/theta))
    z = zeros(S, R, N_rho)
    for s in 1:S, r in 1:R
        z[s, r, :] = rand(frechet_dist[s, r], N_rho)
    end
    # Reshape to (N_rho, R, S) for indexing as z[ρ, r', s]
    z = permutedims(z, (3, 2, 1))
    z_inv = z.^(-1)
    
    # Reshape tau for broadcasting: (S, R_origin, R_dest)
    tau_perm = permutedims(tau, (3, 1, 2))
    
    # ─────────────────────────────────────────────────────────────────────────
    # Distance to closest downstream plant (for regression)
    # ─────────────────────────────────────────────────────────────────────────
    closest_plant_dist = map(x -> distances[x[1], x[2]], 
        argmin(1 ./ (1 ./ distances .* (N_downstream_per_region .> 0)'), dims=2))
    closest_downstream_region = closest_plant_region = vec(getindex.(argmin(1 ./ (1 ./ distances .* (N_downstream_per_region .> 0)'), dims=2), 2))
    
    # ─────────────────────────────────────────────────────────────────────────
    # Initialize storage
    # ─────────────────────────────────────────────────────────────────────────
    X_lrs = zeros(R, R, S)              # Trade flows (as expenditure shares initially)
    c_tilde_r = zeros(R)                # Unit costs c̃_r = c_r / A_r
    linkages = zeros(N_rho, S, R)       # Firm-level supplier indicator
    
    if return_firm_level
        firm_expenditure_shares = zeros(N_rho, S, R, R)  # Share from (ρ,s,l) to r
        firm_intermediate_derivative = zeros(N_rho, S, R, R)
    end
    
    # ─────────────────────────────────────────────────────────────────────────
    # Solve for each downstream region r
    # ─────────────────────────────────────────────────────────────────────────
    for r in 1:R
        if N_downstream_per_region[r] < 1
            continue
        end
        
        # ─────────────────────────────────────────────────────────────────────
        # Compute prices faced by downstream firm r from all potential suppliers
        # p_{ρsr'→r} = w_{r's} · τ_{r'rs} / z_{ρsr'}
        # ─────────────────────────────────────────────────────────────────────
        tau_to_r = reshape(tau_perm[:, :, r]', 1, R, S)  # (1, R, S)
        prices = z_inv .* tau_to_r .* reshape(w_rs, (1, R, 1))  # (N_rho, R, S)
        
        # ─────────────────────────────────────────────────────────────────────
        # Ricardian selection: lowest-cost supplier wins each variety
        # ─────────────────────────────────────────────────────────────────────
        min_coord = reshape(argmin(prices, dims=2), N_rho, S)  # Winner indices
        p_rho_s = prices[min_coord]  # (N_rho, S) - winning prices
        
        # ─────────────────────────────────────────────────────────────────────
        # Nested CES price indices (paper eq. in Appendix B)
        # 
        # P_{sr} = [Σ_ρ (1/N_ρ) · p_{ρs}^{1-ν_s}]^{1/(1-ν_s)}  (within-sector)
        # P_r   = [Σ_s Ω^s · P_{sr}^{1-ν}]^{1/(1-ν)}           (across-sector)
        # c_r   = [Ω^L w_r^{1-λ} + (1-Ω^L) P_r^{1-λ}]^{1/(1-λ)} (unit cost)
        # ─────────────────────────────────────────────────────────────────────
        P_sr = sum(1/N_rho .* p_rho_s.^(1 .- nu_s_mat), dims=1).^(1 ./ (1 .- nu_s_mat))
        P_r = sum(P_sr.^(1 - nu) .* Omega_s)^(1 / (1 - nu))
        c_r = (Omega_L * regional_wages[r]^(1-lambda) + 
               (1-Omega_L) * P_r^(1-lambda))^(1/(1-lambda))
        
        # Apply downstream productivity: c̃_r = c_r / A_r
        c_tilde_r[r] = c_r / A_r[r]
        
        # ─────────────────────────────────────────────────────────────────────
        # Compute expenditure shares from each upstream region l
        # 
        # From nested CES demand (paper Appendix B):
        # share_{ρsl→r} = (1/N_ρ) · Ω^s · (1-Ω^L) · 
        #                 (p_{ρs}/P_{sr})^{1-ν_s} · (P_{sr}/P_r)^{1-ν} · (P_r/c_r)^{1-λ}
        #
        # This is the share of TOTAL COST going to input (ρ,s) from region l
        # To get actual expenditure, multiply by total cost = μ · Y_r
        # ─────────────────────────────────────────────────────────────────────
        labor_substitution_factor = (1 - Omega_L) * (P_r / c_r)^(1 - lambda)
        for l in 1:R
            # Which varieties from region l won?
            winner_mask = map(x -> x[2] == l ? 1.0 : 0.0, min_coord)  # (N_rho, S)
            
            # Update firm-level linkages (binary supplier status)
            linkages[:, :, l] .= max.(linkages[:, :, l], winner_mask)
            
            # Expenditure share for winning firms
            exp_share = winner_mask .* (1/N_rho) .* Omega_s .* (1-Omega_L) .* 
                (p_rho_s ./ P_sr).^(1 .- nu_s_mat) .* 
                (P_sr ./ P_r).^(1-nu) .* 
                (P_r / c_r).^(1-lambda)
            
            # Aggregate to region level
            X_lrs[l, r, :] = sum(exp_share, dims=1)
            
            # Store firm-level for untargeted validation
            if return_firm_level
                firm_expenditure_shares[:, :, l, r] = exp_share
                firm_intermediate_derivative[:, :, l, r] = exp_share ./ labor_substitution_factor
            end
        end
    end
    
    # ─────────────────────────────────────────────────────────────────────────
    # Compute downstream sales Y_r (paper eq. on p.24)
    # 
    # With monopolistic competition:
    #   p_r = c̃_r / μ  where μ = (ε-1)/ε (inverse markup)
    #   P = [Σ_r p_r^ε · δ_r]^{1/ε}  (price index, ε < 0)
    #   Y_r = p_r^ε · P^{-ε} · E · δ_r  (sales = revenue)
    # ─────────────────────────────────────────────────────────────────────────
    mu = (epsilon - 1) / epsilon  # Inverse markup μ = (ε-1)/ε
    
    # Prices: p_r = c̃_r / μ
    p_r = c_tilde_r ./ mu
    
    # Price index: P = [Σ_r p_r^ε · δ_r]^{1/ε}
    # Note: ε < 0, so higher price → lower p_r^ε contribution
    active = N_downstream_per_region .!= 0
    P = sum((p_r[active]).^epsilon .* delta_r[active])^(1/epsilon)
    
    E = 1.0  # Normalize total expenditure
    
    # Downstream sales: Y_r = p_r^ε · P^{-ε} · E · δ_r
    Y_r = zeros(R)
    Y_r[active] = (p_r[active]).^epsilon .* P^(-epsilon) .* E .* delta_r[active]
    
    # ─────────────────────────────────────────────────────────────────────────
    # Scale expenditure shares to actual trade flows
    # 
    # X_lrs currently contains expenditure SHARES of total cost
    # Total cost = μ · Y_r (since price = cost/μ → cost = μ × revenue)
    # Actual trade flow: X_lrs = share × μ × Y_r
    # ─────────────────────────────────────────────────────────────────────────
    for r in 1:R
        if N_downstream_per_region[r] >= 1
            total_cost_r = mu * Y_r[r]
            X_lrs[:, r, :] .*= total_cost_r
        end
    end
    
    # ─────────────────────────────────────────────────────────────────────────
    # Return results
    # ─────────────────────────────────────────────────────────────────────────
    if return_firm_level
        return (
            X_lrs = X_lrs,
            c_tilde_r = c_tilde_r,
            Y_r = Y_r,
            linkages = linkages,
            z = z,
            closest_plant_dist = closest_plant_dist,
            firm_expenditure_shares = firm_expenditure_shares,
            firm_intermediate_derivative = firm_intermediate_derivative,
            mu = mu,
            P = P,
            closest_downstream_region = closest_downstream_region
        )
    else
        return (
            X_lrs = X_lrs,
            c_tilde_r = c_tilde_r,
            Y_r = Y_r,
            linkages = linkages,
            z = z,
            closest_plant_dist = closest_plant_dist,
            closest_downstream_region = closest_downstream_region
        )
    end
end


##################### Moment Computation ###################

"""
    compute_moments(network, params)

Compute targeted moments from solved network for SMM estimation.

# Moments (matching empirical_moments structure):
1. Aggregate labor share: Σ_r w_r·L_r / Σ_r C_r
2. Sectoral input shares: X_s / X
3. Sourcing shares γ_{ls}: Share of sector s inputs from region l
4. Regression coefficients: Elasticity of supplier probability to distance
5. Regional employment shares π_r
"""
function compute_moments(network, params)
    
    beta, Omega_L, Omega_s_vec, A_vec, T_vec = unpack_params(params)
    
    X_lrs = network.X_lrs
    c_tilde_r = network.c_tilde_r  
    Y_r = network.Y_r
    linkages = network.linkages
    z = network.z
    closest_plant_dist = network.closest_plant_dist
    closest_downstream_region = network.closest_downstream_region
    
    # ─────────────────────────────────────────────────────────────────────────
    # 1. Aggregate labor share (matching model_CP.jl exactly)
    # ─────────────────────────────────────────────────────────────────────────
    
    active = N_downstream_per_region .!= 0
    
    # Compute price index and B exactly as in model_CP.jl
    markup = (epsilon - 1) / epsilon
    price_index = sum((c_tilde_r[active] .* markup).^epsilon .* delta_r[active])^(1/epsilon)
    E = 1.0
    B = (markup / price_index)^(epsilon - 1) * E / price_index
    
    # Compute y_r (output-related measure) as in model_CP.jl
    y_r = zeros(R)
    y_r[active] = c_tilde_r[active].^(epsilon - 1) .* delta_r[active] .* B
    
    # Compute labor_r exactly as in model_CP.jl:
    # labor_r = labor_share_tech * y_r * (regional_wages / c_r)^(-lambda)
    labor_r = zeros(R)
    labor_r[active] = Omega_L .* y_r[active] .* (regional_wages[active] ./ c_tilde_r[active]).^(-lambda)
    
    # Aggregate labor share exactly as in model_CP.jl:
    # agg_labor_share = sum(regional_wages * labor_r) / sum(c_r * y_r)
    agg_labor_share = sum(regional_wages .* labor_r) / sum(c_tilde_r .* y_r)
    
    # ─────────────────────────────────────────────────────────────────────────
    # 2. Sectoral input shares: X_s / X
    # ─────────────────────────────────────────────────────────────────────────
    X_ls = reshape(sum(X_lrs, dims=2), R, S)  # Sum over downstream regions
    X_s = sum(X_ls, dims=1)                    # Sum over upstream regions
    X = sum(X_s)                               # Total input purchases
    agg_industry_share = X_s ./ X              # (1, S)
    
    # ─────────────────────────────────────────────────────────────────────────
    # 3. Sourcing shares γ_{ls}: Share of sector s sourced from region l
    # γ_{ls} = (X_{ls} / X_s) × domestic_share_s
    # ─────────────────────────────────────────────────────────────────────────
    gamma_ls = X_ls ./ X_s .* reshape(domestic_share, 1, S)
    
    # ─────────────────────────────────────────────────────────────────────────
    # 4. Regression: P(supplier) vs distance bins
    # ─────────────────────────────────────────────────────────────────────────
    sirens = collect(1:(S*R*N_rho))
    sectors = Int[]
    regions = Int[]
    downstream_regions = Int[]  
    suppliers = Float64[]
    distance_vec = Float64[]
    size_vec = Float64[]

    for r in 1:R
        for s in 1:S
            for rho in 1:N_rho
                push!(sectors, s)
                push!(regions, r)
                push!(downstream_regions, closest_downstream_region[r])              
                push!(suppliers, linkages[rho, s, r] > 0 ? 1.0 : 0.0)
                push!(size_vec, log(z[rho, r, s]))
                push!(distance_vec, closest_plant_dist[r])
            end
        end
    end

    df = DataFrame(
        SIREN = sirens,
        A129 = sectors,
        ze2010 = regions,
        supplier = suppliers,
        log_productivity = size_vec,
        distance = distance_vec,
        downstream_regions = downstream_regions
    )

    # Define bins based on N_beta (must match distance_bin function in tools.jl)
    if N_beta == 5
        bins = [20, 50, 100, 150, 200, Inf]
    elseif N_beta == 4
        bins = [50, 100, 150, 200, Inf]
    else
        error("Unsupported N_beta: $N_beta. Add bin definition for this case.")
    end
    df.A129_r = string.(df.A129) .* "_" .* string.(df.downstream_regions)
    df.distance_bin = cut(df.distance, bins, extend=true)

    fixest = reg(df, @formula(supplier ~ distance_bin + log_productivity + fe(A129_r)))
    reg_coef = fixest.coef[1:N_beta]  # Changed from hardcoded 1:5
    
    # ─────────────────────────────────────────────────────────────────────────
    # 5. Regional employment shares π_r (matching model_CP.jl)
    # 
    # From model_CP.jl:
    #   pi_r = labor_r[active] / sum(labor_r[active])
    # ─────────────────────────────────────────────────────────────────────────
    #pi_r = labor_r[active] ./ sum(labor_r[active])

    # 6. Downstream sales share
    pi_r = Y_r[active]/sum(Y_r[active])
    
    return (
        agg_labor_share = [agg_labor_share],
        agg_industry_share = agg_industry_share[2:end],  # Drop first (normalized)
        gamma_ls = gamma_ls,
        reg_coef = reg_coef,
        pi_r = pi_r[2:end]  # Drop first
    )
end


##################### Main SMM Function ###################

"""
    SMM(params, simulation=false)

Main SMM function for optimization.

# Arguments
- `params`: Parameter vector
- `simulation`: If true, return trade flow matrix only

# Returns
- If simulation=true: Trade flow matrix (R × R)
- If simulation=false: Tuple of moments for SMM estimation
"""
function SMM(params, simulation=false)
    
    # Solve network
    network = solve_network(params, return_firm_level=false)
    
    if simulation
        # Return aggregated trade flows (sum over sectors)
        return reshape(sum(network.X_lrs, dims=3), R, R)
    end
    
    # Compute and return moments
    moments = compute_moments(network, params)
    
    return (
        moments.agg_labor_share,
        moments.agg_industry_share,
        moments.gamma_ls,
        moments.reg_coef,
        moments.pi_r
    )
end


"""
    SMM_with_network(params)

Solve network and return both moments and firm-level data.
Used for untargeted moment validation (Table 2 regression).

# Returns
- `moments`: Tuple of targeted moments  
- `network`: Full network solution including firm-level data
"""
function SMM_with_network(params)
    
    # Solve network with firm-level data
    network = solve_network(params, return_firm_level=true)
    
    # Compute moments
    moments = compute_moments(network, params)
    
    return moments, network
end


##################### Loss Function ###################

"""
    loss_function(simulated_moments, emp, W, method="original")

Compute loss between empirical and simulated moments.

# Methods
- "original": Raw squared difference
- "normalize": Difference scaled by sqrt of moment group size
- "hybrid": Percentage deviation for non-zero, absolute for zeros
"""
function loss_function(simulated_moments, emp, W, method="original")
    
    # Backward compatibility with Bool
    if method isa Bool
        method = method ? "normalize" : "original"
    end
    
    # Compute normalization factors
    square_size = sqrt.(vcat([fill(length(vec(m)), length(vec(m))) for m in simulated_moments]...))'
    
    # Flatten moments
    simulated_moments = vcat([vec(simulated_moments[i]) for i in 1:length(simulated_moments)]...)
    N = length(simulated_moments)
    simulated_moments = reshape(simulated_moments, (1, N))
    
    if method == "original"
        err = (emp - simulated_moments)
    elseif method == "normalize"
        err = (emp - simulated_moments) ./ square_size
    elseif method == "hybrid"
        emp_vec = vec(emp)
        sim_vec = vec(simulated_moments)
        err_vec = similar(emp_vec)
        for i in 1:length(emp_vec)
            if abs(emp_vec[i]) > 1e-10
                err_vec[i] = (sim_vec[i] - emp_vec[i]) / emp_vec[i]
            else
                err_vec[i] = sim_vec[i]
            end
        end
        err = reshape(err_vec, (1, N))
    else
        error("Unknown method: $method. Use 'original', 'normalize', or 'hybrid'.")
    end
    
    W = isnothing(W) ? I(N) : W
    return err * W * err'
end


"""
    full_SMM(params, simulation=false, second_stage=false, method="original")

Full SMM evaluation: compute loss and return moments.
"""
function full_SMM(params, simulation=false, second_stage=false, method="original")
    
    simulated_moments = SMM(params, simulation)
    
    if second_stage
        emp = empirical_moments_reduced
        W = Weight_matrix
        moments = [simulated_moments[3][mask_emp_gamma_ls .!= 0]]
    else
        emp = empirical_moments
        W = Weight_matrix_custom
        moments = simulated_moments
    end
    
    if simulation
        return simulated_moments
    else
        return loss_function(moments, emp, W, method), simulated_moments
    end
end