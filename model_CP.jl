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

    CLOSEST_PLANT_DIST = vec(map(x -> distances[x[1], x[2]], argmin(1 ./ (1 ./ distances .* (N_downstream_per_region .> 0)'), dims=2)))
    CLOSEST_DOWNSTREAM_REGION = vec(getindex.(argmin(1 ./ (1 ./ distances .* (N_downstream_per_region .> 0)'), dims=2), 2))
end


##################### Stratified Draws (CdGM-style) ###################

"""
    generate_stratified_draws(N_rho; seed=50) -> (u_quantiles, weights)

Generate CdGM-style stratified uniform draws for Fréchet inverse CDF.

Uses 25 non-uniform bins on [0,1] that oversample the upper tail.
Within each bin, draws are evenly spaced (deterministic).

Returns:
- `u_quantiles`: Vector of length N_rho with uniform quantiles in (0,1)
- `weights`: Vector of length N_rho, weights sum to 1.0
"""
function generate_stratified_draws(N_rho; seed=50)

    bin_edges = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50,
                 0.55, 0.60, 0.65, 0.70, 0.75,
                 0.80, 0.85, 0.90, 0.925, 0.95,
                 0.96, 0.97, 0.98, 0.99, 0.995,
                 0.996, 0.997, 0.998, 0.999, 1.0]

    N_bins = length(bin_edges) - 1  # 25

    # Distribute N_rho across bins; extras go to tail bins
    base_per_bin = div(N_rho, N_bins)
    remainder = mod(N_rho, N_bins)

    n_per_bin = fill(base_per_bin, N_bins)
    for i in 1:remainder
        n_per_bin[N_bins - i + 1] += 1
    end

    @assert sum(n_per_bin) == N_rho

    # Generate uniform quantiles within each bin (evenly spaced midpoints)
    u_quantiles = Float64[]
    sample_weights = Float64[]

    for b in 1:N_bins
        lo = bin_edges[b]
        hi = bin_edges[b + 1]
        width = hi - lo
        n = n_per_bin[b]

        for k in 1:n
            u = lo + (k - 0.5) / n * width  # midpoints within bin
            push!(u_quantiles, u)
            push!(sample_weights, width / n)  # weight = bin_width / n_firms_in_bin
        end
    end

    # Normalize weights to sum to 1
    sample_weights ./= sum(sample_weights)

    return u_quantiles, sample_weights
end


##################### Helper Functions ###################

"""
    unpack_params(params) -> (β, Ω^L, Ω^s, A, T)

Unpack parameter vector into model components (paper notation).

Returns:
- β (beta): Trade cost parameters for distance bins [N_beta elements]
- Ω^L (Omega_L): Labor share in production [scalar]
- Ω^s (Omega_s): Sectoral input shares [S elements, normalized to sum to 1]
- A: Downstream firm productivity by region [R_downstream elements]
- T: Fréchet scale parameters [S × R elements, full vector with zeros for masked entries]
"""
function unpack_params(params)
    beta = params[1:N_beta]
    Omega_L = params[N_beta + 1]
    Omega_s = params[(N_beta + 2):(N_beta + 1 + S)] / sum(params[(N_beta + 2):(N_beta + 1 + S)])
    A = params[(N_beta + S + 2):(N_beta + R_downstream + S + 1)]
    T_reduced = params[(N_beta + R_downstream + S + 2):end]

    # Expand reduced T back to full S*R vector using T_MASK
    T_full = zeros(S * R)
    T_full[T_MASK] = T_reduced

    return beta, Omega_L, Omega_s, A, T_full
end


"""
    build_tau(beta) -> τ[r', r]

Build iceberg trade cost matrix from distance bin coefficients.

τ_{r'r} = 1 + β_b  where b = DistBin[r', r]

Returns matrix of size (R, R). Trade costs are identical across sectors.
"""
function build_tau(beta)
    tau = ones(R, R_downstream)
    for r_prime in 1:R, r_d in 1:R_downstream
        b = DistBin[r_prime, r_d]
        if b > 0
            tau[r_prime, r_d] += beta[b]
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
- `X_ls_flat`: Trade flows by good pair index [n_good], scaled by μ·Y_r
- `c_tilde_r`: Unit costs c̃_r = c_r/A_r of downstream firms [R]
- `Y_r`: Downstream sales (revenue) by region [R]
- `linkages_flat`: Firm-level supplier indicator [N_rho × n_good]
- `z_flat`: Firm productivities [N_rho × n_good]
- `closest_plant_dist`: Distance to nearest downstream plant [R]

If return_firm_level=true, additionally returns sparse COO vectors:
- `firm_exp_rho`, `firm_exp_s`, `firm_exp_g`, `firm_exp_r`: Indices
- `firm_exp_val`: Expenditure share values
- `firm_deriv_val`: Intermediate derivative values
- `mu`: Inverse markup μ = (ε-1)/ε
- `P`: Aggregate downstream price index
"""
function solve_network(params; return_firm_level=false,
                       precomputed_tau::Union{Nothing, Matrix{Float64}}=nothing,
                       u_draws::Union{Nothing, Vector{Float64}}=nothing,
                       sample_weights::Union{Nothing, Vector{Float64}}=nothing)

    # ─────────────────────────────────────────────────────────────────────────
    # Unpack parameters (paper notation)
    # ─────────────────────────────────────────────────────────────────────────
    beta, Omega_L, Omega_s_vec, A_vec, T_vec = unpack_params(params)

    # Build trade cost matrix τ_{r'r} — identical across sectors
    tau = precomputed_tau === nothing ? build_tau(beta) : precomputed_tau

    # A_vec is already R_downstream length
    A_r = A_vec

    # Reshape for broadcasting
    Omega_s = reshape(Omega_s_vec, 1, S)  # (1, S)
    nu_s_mat = reshape(nu_s, 1, S)        # (1, S)
    T = reshape(T_vec, S, R)              # (S, R)

    # ─────────────────────────────────────────────────────────────────────────
    # Draw upstream firm productivities — flat (N_rho, n_good) layout
    # Only good (s,r) pairs (where T_MASK is true) are computed.
    # ─────────────────────────────────────────────────────────────────────────
    if u_draws === nothing
        # Backward compatibility: random Fréchet draws with uniform weights
        Random.seed!(50)
        z_flat = zeros(N_rho, n_good)
        for g in 1:n_good
            T_sr = T[GOOD_S[g], GOOD_R[g]]
            if T_sr > 0
                d = Frechet(theta, T_sr^(1/theta))
                z_flat[:, g] = rand(d, N_rho)
            end
        end
        if sample_weights === nothing
            sample_weights = fill(1.0/N_rho, N_rho)
        end
    else
        # CdGM-style: Fréchet inverse CDF from stratified uniform draws
        # u_draws is now a Vector{Float64} of length N_rho (same quantiles for all pairs)
        # F⁻¹(u) = σ · (-ln(1-u))^(-1/θ) where σ = T_{sr}^{1/θ}
        z_flat = zeros(N_rho, n_good)
        for g in 1:n_good
            T_sr = T[GOOD_S[g], GOOD_R[g]]
            scale = T_sr^(1/theta)
            for rho in 1:N_rho
                z_flat[rho, g] = scale * (-log(1.0 - u_draws[rho]))^(-1.0/theta)
            end
        end
    end
    z_inv_flat = z_flat .^ (-1)

    # Reshape sample_weights for broadcasting in CES aggregation
    w_rho = reshape(sample_weights, N_rho, 1)  # (N_rho, 1)

    # ─────────────────────────────────────────────────────────────────────────
    # Distance to closest downstream plant (precomputed constants)
    # ─────────────────────────────────────────────────────────────────────────
    closest_plant_dist = CLOSEST_PLANT_DIST
    closest_downstream_region = CLOSEST_DOWNSTREAM_REGION

    # ─────────────────────────────────────────────────────────────────────────
    # Initialize storage — flat arrays indexed by good pair index
    # ─────────────────────────────────────────────────────────────────────────
    X_shares_by_r = zeros(n_good, R_downstream)  # Raw expenditure shares per good pair per downstream r
    c_tilde_r = zeros(R_downstream)              # Unit costs c̃_r = c_r / A_r
    linkages_flat = zeros(N_rho, n_good) # Firm-level supplier indicator

    if return_firm_level
        # Sparse COO storage: one entry per (rho, downstream_r) winning pair
        firm_exp_rho = Int[]
        firm_exp_s = Int[]
        firm_exp_g = Int[]        # good-pair index (encodes upstream region via GOOD_R[g])
        firm_exp_r = Int[]        # downstream region
        firm_exp_val = Float64[]  # expenditure share value
        firm_deriv_val = Float64[] # intermediate derivative value
    end

    # ─────────────────────────────────────────────────────────────────────────
    # Solve for each downstream region r
    # Prices and argmin computed only over active (s,r') pairs per sector.
    # The `for l in 1:R` inner loop is eliminated — winners are directly indexed.
    # ─────────────────────────────────────────────────────────────────────────
    for r_d in 1:R_downstream
        r = DOWNSTREAM_REGIONS[r_d]  # R-index for wages/regional lookups

        # ─────────────────────────────────────────────────────────────────────
        # For each sector, find cheapest supplier among active upstream regions
        # p_{ρsr'→r} = w_{r's} · τ_{r'r} / z_{ρsr'}
        # ─────────────────────────────────────────────────────────────────────
        p_rho_s = zeros(N_rho, S)
        winner_good_idx = zeros(Int, N_rho, S)

        for s in 1:S
            g_indices = SECTOR_GOOD_INDICES[s]
            if isempty(g_indices); continue; end
            regions_s = SECTOR_GOOD_REGIONS[s]

            # Prices only for active upstream (s,r') pairs
            tau_sr = reshape(tau[regions_s, r_d], 1, :)      # (1, n_active_in_s); tau is R × R_downstream
            w_sr = reshape(W_RS_FLAT[g_indices], 1, :)       # (1, n_active_in_s)
            prices_s = z_inv_flat[:, g_indices] .* tau_sr .* w_sr  # (N_rho, n_active_in_s)

            # Ricardian selection: lowest-cost supplier wins each variety
            min_local = argmin(prices_s, dims=2)  # (N_rho, 1)
            for rho in 1:N_rho
                local_idx = min_local[rho][2]
                winner_good_idx[rho, s] = g_indices[local_idx]
                p_rho_s[rho, s] = prices_s[rho, local_idx]
            end
        end

        # ─────────────────────────────────────────────────────────────────────
        # Nested CES price indices (paper eq. in Appendix B)
        #
        # P_{sr} = [Σ_ρ (1/N_ρ) · p_{ρs}^{1-ν_s}]^{1/(1-ν_s)}  (within-sector)
        # P_r   = [Σ_s Ω^s · P_{sr}^{1-ν}]^{1/(1-ν)}           (across-sector)
        # c_r   = [Ω^L w_r^{1-λ} + (1-Ω^L) P_r^{1-λ}]^{1/(1-λ)} (unit cost)
        # ─────────────────────────────────────────────────────────────────────
        P_sr = sum(w_rho .* p_rho_s.^(1 .- nu_s_mat), dims=1).^(1 ./ (1 .- nu_s_mat))
        P_r = sum(P_sr.^(1 - nu) .* Omega_s)^(1 / (1 - nu))
        c_r = (Omega_L * regional_wages[r]^(1-lambda) +
               (1-Omega_L) * P_r^(1-lambda))^(1/(1-lambda))

        # Apply downstream productivity: c̃_r = c_r / A_r
        c_tilde_r[r_d] = c_r / A_r[r_d]

        # ─────────────────────────────────────────────────────────────────────
        # Accumulate expenditure shares directly by winner good pair
        #
        # share_{ρs} = w_ρ · Ω^s · (1-Ω^L) ·
        #              (p_{ρs}/P_{sr})^{1-ν_s} · (P_{sr}/P_r)^{1-ν} · (P_r/c_r)^{1-λ}
        # ─────────────────────────────────────────────────────────────────────
        labor_substitution_factor = (1 - Omega_L) * (P_r / c_r)^(1 - lambda)
        for s in 1:S
            P_sr_s = P_sr[s]
            nu_s_s = nu_s[s]
            for rho in 1:N_rho
                g_winner = winner_good_idx[rho, s]
                if g_winner == 0; continue; end

                # Mark linkage
                linkages_flat[rho, g_winner] = 1.0

                # Expenditure share for this variety
                exp_val = sample_weights[rho] * Omega_s_vec[s] * (1-Omega_L) *
                    (p_rho_s[rho, s] / P_sr_s)^(1 - nu_s_s) *
                    (P_sr_s / P_r)^(1 - nu) *
                    (P_r / c_r)^(1 - lambda)

                X_shares_by_r[g_winner, r_d] += exp_val

                if return_firm_level
                    push!(firm_exp_rho, rho)
                    push!(firm_exp_s, s)
                    push!(firm_exp_g, g_winner)
                    push!(firm_exp_r, r)  # keep R-indexed for backward compat
                    push!(firm_exp_val, exp_val)
                    push!(firm_deriv_val, exp_val / labor_substitution_factor)
                end
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

    # Prices: p_r = c̃_r / μ  (R_downstream length)
    p_r = c_tilde_r ./ mu

    # Price index: P = [Σ_r p_r^ε · δ_r]^{1/ε}
    # Note: ε < 0, so higher price → lower p_r^ε contribution
    P = sum(p_r.^epsilon .* delta_r[DOWNSTREAM_REGIONS])^(1/epsilon)

    E = 1.0  # Normalize total expenditure

    # Downstream sales: Y_r = p_r^ε · P^{-ε} · E · δ_r  (R_downstream length)
    Y_r = p_r.^epsilon .* P^(-epsilon) .* E .* delta_r[DOWNSTREAM_REGIONS]

    # ─────────────────────────────────────────────────────────────────────────
    # Scale expenditure shares to actual trade flows
    #
    # X_shares_by_r contains raw expenditure SHARES per good pair per downstream r.
    # Total cost for downstream r = μ · Y_r.
    # Actual flow: X_ls_flat[g] = Σ_r X_shares_by_r[g, r_d] × μ × Y_r[r_d]
    # ─────────────────────────────────────────────────────────────────────────
    X_ls_flat = zeros(n_good)
    for r_d in 1:R_downstream
        total_cost_r = mu * Y_r[r_d]
        for g in 1:n_good
            X_ls_flat[g] += X_shares_by_r[g, r_d] * total_cost_r
        end
    end

    # Pad c_tilde_r and Y_r back to R-length for backward compatibility
    # (compute_moments uses c_tilde_r[active] with R-index; compute_amplification_weights uses Y_r[r])
    c_tilde_r_full = zeros(R); c_tilde_r_full[DOWNSTREAM_REGIONS] = c_tilde_r
    Y_r_full       = zeros(R); Y_r_full[DOWNSTREAM_REGIONS]       = Y_r

    # ─────────────────────────────────────────────────────────────────────────
    # Return results
    # ─────────────────────────────────────────────────────────────────────────
    if return_firm_level
        return (
            X_ls_flat = X_ls_flat,
            c_tilde_r = c_tilde_r_full,
            Y_r = Y_r_full,
            linkages_flat = linkages_flat,
            z_flat = z_flat,
            closest_plant_dist = closest_plant_dist,
            firm_exp_rho = firm_exp_rho,
            firm_exp_s = firm_exp_s,
            firm_exp_g = firm_exp_g,
            firm_exp_r = firm_exp_r,
            firm_exp_val = firm_exp_val,
            firm_deriv_val = firm_deriv_val,
            mu = mu,
            P = P,
            closest_downstream_region = closest_downstream_region,
            sample_weights = sample_weights
        )
    else
        return (
            X_ls_flat = X_ls_flat,
            c_tilde_r = c_tilde_r_full,
            Y_r = Y_r_full,
            linkages_flat = linkages_flat,
            z_flat = z_flat,
            closest_plant_dist = closest_plant_dist,
            closest_downstream_region = closest_downstream_region,
            sample_weights = sample_weights
        )
    end
end


##################### Analytical Weighted OLS ###################

"""
    fast_weighted_regression(linkages, z, sample_weights)

Compute regression: supplier ~ distance_bin_dummies + log_productivity + fe(A129_r)
using analytical weighted OLS with group demeaning (Frisch-Waugh-Lovell).

Replaces FixedEffectModels.reg() for major speedup per evaluation.
"""
function fast_weighted_regression(linkages_flat, z_flat, sample_weights)

    n_regressors = N_beta + 1  # distance bins + log_productivity

    # All good pairs are valid (n_good entries, each with N_rho varieties)
    N_valid = n_good * N_rho

    y = Vector{Float64}(undef, N_valid)
    X = zeros(N_valid, n_regressors)
    w = Vector{Float64}(undef, N_valid)
    fe_group = Vector{Int}(undef, N_valid)

    idx = 0
    for g in 1:n_good
        s = GOOD_S[g]
        r = GOOD_R[g]
        dr = CLOSEST_DOWNSTREAM_REGION[r]
        b = DistBin[r, dr]
        group_id = (s - 1) * R_downstream + dr

        for rho in 1:N_rho
            idx += 1
            y[idx] = linkages_flat[rho, g] > 0 ? 1.0 : 0.0
            w[idx] = sample_weights[rho]
            fe_group[idx] = group_id

            if b > 0 && b <= N_beta
                X[idx, b] = 1.0
            end
            X[idx, n_regressors] = log(z_flat[rho, g])
        end
    end

    # Weighted FWL demeaning by fixed-effect groups
    unique_groups = unique(fe_group)
    for g in unique_groups
        mask = fe_group .== g
        w_g = w[mask]
        total_w = sum(w_g)
        if total_w < 1e-15; continue; end

        y[mask] .-= sum(w_g .* y[mask]) / total_w
        for j in 1:n_regressors
            X[mask, j] .-= sum(w_g .* X[mask, j]) / total_w
        end
    end

    # Weighted OLS: transform by sqrt(w), then ordinary least squares
    sqrt_w = sqrt.(w)
    Xw = sqrt_w .* X
    yw = sqrt_w .* y
    coefs = Xw \ yw

    return coefs[1:N_beta]
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

    X_ls_flat = network.X_ls_flat
    c_tilde_r = network.c_tilde_r
    Y_r = network.Y_r
    linkages_flat = network.linkages_flat
    z_flat = network.z_flat
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
    # Reconstruct X_ls (R, S) from flat representation
    X_ls = zeros(R, S)
    for g in 1:n_good
        X_ls[GOOD_R[g], GOOD_S[g]] = X_ls_flat[g]
    end
    X_s = sum(X_ls, dims=1)                    # Sum over upstream regions
    X = sum(X_s)                               # Total input purchases
    agg_industry_share = X_s ./ X              # (1, S)

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Sourcing shares γ_{ls}: Share of sector s sourced from region l
    # γ_{ls} = (X_{ls} / X_s) × domestic_share_s
    # ─────────────────────────────────────────────────────────────────────────
    gamma_ls = X_ls ./ X_s .* reshape(domestic_share, 1, S)

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Regression: P(supplier) vs distance bins (analytical weighted OLS)
    # ─────────────────────────────────────────────────────────────────────────
    sw = network.sample_weights
    reg_coef = fast_weighted_regression(linkages_flat, z_flat, sw)

    
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
        agg_industry_share = vec(agg_industry_share),  # Full S elements (mask handles [2:end])
        gamma_ls = gamma_ls,
        reg_coef = reg_coef,
        pi_r = pi_r                                     # Full R_downstream (mask handles [2:end])
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
function SMM(params, simulation=false; precomputed_tau::Union{Nothing, Matrix{Float64}}=nothing,
             u_draws::Union{Nothing, Vector{Float64}}=nothing,
             sample_weights::Union{Nothing, Vector{Float64}}=nothing)

    # Solve network
    network = solve_network(params, return_firm_level=false, precomputed_tau=precomputed_tau,
                            u_draws=u_draws, sample_weights=sample_weights)
    
    if simulation
        # Return aggregated trade flows (sum over sectors) as (R, R)
        # Reconstruct from flat X_ls: X_lr = Σ_s X_lrs, but X_ls_flat only has (l,s) pairs.
        # This path is unused in optimization; return (R, S) sourcing instead.
        X_ls = zeros(R, S)
        for g in 1:n_good
            X_ls[GOOD_R[g], GOOD_S[g]] = network.X_ls_flat[g]
        end
        return X_ls
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
function SMM_with_network(params; precomputed_tau::Union{Nothing, Matrix{Float64}}=nothing,
                          u_draws::Union{Nothing, Vector{Float64}}=nothing,
                          sample_weights::Union{Nothing, Vector{Float64}}=nothing)

    # Solve network with firm-level data
    network = solve_network(params, return_firm_level=true, precomputed_tau=precomputed_tau,
                            u_draws=u_draws, sample_weights=sample_weights)
    
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

    if method isa Bool
        method = method ? "normalize" : "original"
    end

    square_size = sqrt.(vcat([fill(length(vec(m)), length(vec(m))) for m in simulated_moments]...))
    sim_flat = vcat([vec(simulated_moments[i]) for i in 1:length(simulated_moments)]...)

    sim_flat    = sim_flat[MOMENT_MASK]
    square_size = square_size[MOMENT_MASK]
    emp_flat    = vec(emp)

    N = length(sim_flat)

    if method == "original"
        err = reshape(emp_flat - sim_flat, (1, N))

    elseif method == "normalize"
        err = reshape((emp_flat - sim_flat) ./ square_size, (1, N))

    elseif method == "log"
        # Block boundaries in the masked moment vector:
        #   [1 : n_good]              labor share + industry shares + gamma_ls → log
        #   [n_good+1 : n_good+N_beta] reg_coef (negative, level deviation)   → level
        #   [n_good+N_beta+1 : end]   pi_r                                     → log
        eps = 1e-12
        err = zeros(N)

        # Log blocks: all strictly positive moments
        log_end   = n_good
        reg_start = n_good + 1
        reg_end   = n_good + N_beta
        pi_start  = n_good + N_beta + 1

        err[1:log_end] = log.(max.(emp_flat[1:log_end], eps)) .-
                         log.(max.(sim_flat[1:log_end], eps))

        # Level block: reg_coef (negative coefficients, log undefined)
        err[reg_start:reg_end] = emp_flat[reg_start:reg_end] .-
                                 sim_flat[reg_start:reg_end]

        # Log block: pi_r
        err[pi_start:end] = log.(max.(emp_flat[pi_start:end], eps)) .-
                            log.(max.(sim_flat[pi_start:end], eps))

        err = reshape(err, (1, N))

    else
        error("Unknown method: $method. Use 'original', 'normalize', or 'log'.")
    end

    W = isnothing(W) ? I(N) : W
    return err * W * err'
end


"""
    full_SMM(params, simulation=false, second_stage=false, method="original")

Full SMM evaluation: compute loss and return moments.
"""
function full_SMM(params, simulation=false, second_stage=false, method="original";
                  precomputed_tau::Union{Nothing, Matrix{Float64}}=nothing,
                  u_draws::Union{Nothing, Vector{Float64}}=nothing,
                  sample_weights::Union{Nothing, Vector{Float64}}=nothing)

    simulated_moments = SMM(params, simulation; precomputed_tau=precomputed_tau,
                            u_draws=u_draws, sample_weights=sample_weights)
    
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