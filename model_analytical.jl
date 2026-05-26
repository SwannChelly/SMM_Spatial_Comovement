##### Analytical GMM moments for Spatial Comovement #####
# Closed-form Eaton-Kortum implementation for GMM estimation.
# Replaces solve_network/compute_moments for moment blocks
# {Ω^L, Ω^s, π_r, γ_ls} using exact EK formulas.
# The reg_coef block uses Gauss-Legendre quadrature on Fréchet z (deterministic).
#
# Requires: GAMMA_FACTOR const (added to load_parameters.jl after loading nu_s/theta)

using SpecialFunctions
using FastGaussQuadrature
using Printf


"""
    compute_Phi(T_mat, tau)  ->  Matrix{T}  (S × R_downstream)

Φ_{s,dr} = Σ_{r'} T_{r's} (w_{r's} τ_{r'dr})^{-θ}

Summed over active (s, r') pairs using SECTOR_GOOD_INDICES.
"""
function compute_Phi(T_mat::AbstractMatrix{FT}, tau::AbstractMatrix{FT}) where {FT}
    Phi = zeros(FT, S, R_downstream)
    for g in 1:n_good
        s      = GOOD_S[g]
        r_p    = GOOD_R[g]
        T_val  = T_mat[s, r_p]
        w_val  = W_RS_FLAT[g]
        for dr in 1:R_downstream
            Phi[s, dr] += T_val * (w_val * tau[r_p, dr])^(-theta)
        end
    end
    return Phi
end


"""
    compute_prices_analytical(Omega_L, Omega_s_vec, A_r, T_mat, tau)
        -> NamedTuple with P_sr, P_r, c_r, c_tilde_r, p_r, P_agg, Y_r, mu, Phi

Closed-form CES price indices and downstream outputs using EK analytics.

P_{s,dr} = GAMMA_FACTOR[s] × Φ_{s,dr}^{-1/θ}

where GAMMA_FACTOR[s] = Γ((θ+1-ν_s)/θ)^{1/(1-ν_s)}.
"""
function compute_prices_analytical(
    Omega_L::FT, Omega_s_vec, A_r, T_mat::AbstractMatrix, tau::AbstractMatrix
) where {FT}
    Phi = compute_Phi(T_mat, tau)

    # Sector price indices per downstream region: P_sr[s, dr]
    P_sr = similar(Phi)
    for s in 1:S, dr in 1:R_downstream
        P_sr[s, dr] = GAMMA_FACTOR[s] * Phi[s, dr]^(-1/theta)
    end

    # Aggregate intermediate price index per downstream region: P_r[dr]
    # reshape Omega_s to (S,1) for broadcasting against P_sr (S × R_downstream)
    Os_col = reshape(collect(Omega_s_vec), S, 1)
    P_r = vec(sum(Os_col .* P_sr .^ (1 - nu), dims=1)) .^ (1 / (1 - nu))

    # Unit cost c_r[dr] = (Ω^L w_r^{1-λ} + (1-Ω^L) P_r^{1-λ})^{1/(1-λ)}
    c_r = similar(P_r)
    for dr in 1:R_downstream
        r = DOWNSTREAM_REGIONS[dr]
        c_r[dr] = (Omega_L * regional_wages[r]^(1-lambda) +
                   (1-Omega_L) * P_r[dr]^(1-lambda))^(1/(1-lambda))
    end

    # Downstream productivity: c̃_r = c_r / A_r
    c_tilde_r = c_r ./ A_r

    # Downstream prices: p_r = c̃_r / μ
    mu = FT(epsilon / (epsilon - 1))
    p_r = c_tilde_r ./ mu

    # Aggregate price index: P = [Σ_dr p_r^ε δ_r]^{1/ε}
    delta_down = delta_r[DOWNSTREAM_REGIONS]
    P_agg = sum(p_r .^ epsilon .* delta_down)^(1/epsilon)

    # Downstream sales: Y_r = p_r^ε × P^{-ε} × δ_r
    Y_r = p_r .^ epsilon .* P_agg^(-epsilon) .* delta_down

    return (P_sr=P_sr, P_r=P_r, c_r=c_r, c_tilde_r=c_tilde_r,
            p_r=p_r, P_agg=P_agg, Y_r=Y_r, mu=mu, Phi=Phi)
end



"""
    compute_regression_quadrature(T_mat, tau, Phi; n_quad=200) -> Vector (N_beta,)

Distance-bin regression coefficients via deterministic Gauss-Legendre quadrature.

For each active (r', s) pair at n_quad nodes in the Fréchet CDF:
  z_k = T_{r's}^{1/θ} × (-log(1-u_k))^{-1/θ}
  y_k = exp(z_k^{-θ} × (A_{r's,dr} - Φ_{s,dr}) × (w_{r's}τ_{r'dr})^θ)
      = P(firm at z_k beats all other competitors for closest downstream dr | 1 draw/region)

After weighted FWL demeaning by (sector × closest downstream), WLS gives β_{1..N_beta}.

The z-variation in y_k captures the productivity gradient; the distance-bin variation
captures the τ effect on supplier probability, consistent with LPM on 0/1 linkages.
"""
function compute_regression_quadrature(T_mat, tau, Phi; n_quad::Int=200)
    FT = eltype(Phi)

    # Gauss-Legendre nodes/weights on (-1,1), mapped to (0,1)
    nodes_raw, weights_raw = gausslegendre(n_quad)
    u_nodes  = (nodes_raw  .+ 1) ./ 2
    gl_wts   = weights_raw ./ 2   # Jacobian of (-1,1)→(0,1)

    N_total = n_good * n_quad
    n_reg   = N_beta + 1  # distance bins + log(z)

    y        = Vector{FT}(undef, N_total)
    X        = zeros(FT, N_total, n_reg)
    w_arr    = Vector{Float64}(undef, N_total)
    fe_group = Vector{Int}(undef, N_total)

    idx = 0
    for g in 1:n_good
        s    = GOOD_S[g]
        r_p  = GOOD_R[g]
        dr   = CLOSEST_DOWNSTREAM_REGION[r_p]
        gid  = (s - 1) * R_downstream + dr

        T_val  = T_mat[s, r_p]
        w_val  = W_RS_FLAT[g]
        tau_val = tau[r_p, dr]

        # EK term for (r', s) supplying dr
        A_rsd  = T_val * (w_val * tau_val)^(-theta)
        phi_dr = Phi[s, dr]

        # Exponent coefficient (negative, so y_k = exp(negative × z_k^{-θ}) ∈ (0,1))
        exp_coef = (A_rsd - phi_dr) * (w_val * tau_val)^theta

        scale = T_val^(1/theta)

        if N_beta == 1
            log_dist = LOG_CLOSEST_DIST[r_p]
        else
            b = DistBin[r_p, dr]
        end

        for k in 1:n_quad
            idx += 1
            u_k = u_nodes[k]
            # Guard against u_k == 1 (→ z = ∞)
            u_k = min(u_k, 1.0 - 1e-14)
            z_k = scale * (-log(1.0 - u_k))^(-1.0/theta)

            y[idx]        = exp(z_k^(-theta) * exp_coef)
            w_arr[idx]    = gl_wts[k]
            fe_group[idx] = gid

            if N_beta == 1
                X[idx, 1] = log_dist
            else
                if b > 0 && b <= N_beta
                    X[idx, b] = FT(1.0)
                end
            end
            X[idx, n_reg] = log(z_k)
        end
    end

    # Weighted FWL demeaning by fixed-effect groups
    for gid in unique(fe_group)
        mask    = fe_group .== gid
        w_g     = w_arr[mask]
        total_w = sum(w_g)
        total_w < 1e-15 && continue

        y[mask] .-= sum(w_g .* y[mask]) / total_w
        for j in 1:n_reg
            X[mask, j] .-= sum(w_g .* X[mask, j]) / total_w
        end
    end

    # Weighted OLS: sqrt(w) transform → ordinary LS
    sqrt_w = sqrt.(w_arr)
    Xw  = sqrt_w .* X
    yw  = sqrt_w .* y
    coefs = Xw \ yw

    return coefs[1:N_beta]
end


"""
    compute_moments_analytical(params; n_quad=200) -> NamedTuple

Drop-in replacement for compute_moments(network, params).
Returns the same 5 moment blocks using closed-form EK formulas.

Blocks {Ω^L, Ω^s, π_r, γ_ls} are exact; reg_coef uses Gauss-Legendre quadrature.
"""
function compute_moments_analytical(params; n_quad::Int=200)
    Omega_L, Omega_s_vec, A_vec, beta, T_vec = unpack_params(params)
    T_mat = reshape(T_vec, S, R)

    tau = build_tau(beta)

    prices = compute_prices_analytical(Omega_L, Omega_s_vec, A_vec, T_mat, tau)
    (; P_sr, P_r, c_r, c_tilde_r, p_r, P_agg, Y_r, mu, Phi) = prices

    FT = eltype(Y_r)

    # ── Block 1: Aggregate labor share ──────────────────────────────────────────
    # note: c_tilde_r × markup = p_r and P_agg = price_index (from solve_network)
    markup  = FT((epsilon - 1) / epsilon)
    B       = (markup / P_agg)^(epsilon - 1) / P_agg
    delta_d = delta_r[DOWNSTREAM_REGIONS]

    y_r_vec = c_tilde_r .^ (epsilon - 1) .* delta_d .* B
    labor_r  = Omega_L .* y_r_vec .* (regional_wages[DOWNSTREAM_REGIONS] ./ c_tilde_r) .^ (-lambda)

    agg_ls = sum(regional_wages[DOWNSTREAM_REGIONS] .* labor_r) /
             sum(c_tilde_r .* y_r_vec)

    # ── Blocks 2 & 5: Build X_ls (R × S) for sectoral shares and γ_ls ──────────
    # X_ls[r', s] = Σ_{dr} γ_{r',s,dr} × expenditure_{s,dr}
    X_ls = zeros(FT, R, S)
    for g in 1:n_good
        s   = GOOD_S[g]
        r_p = GOOD_R[g]
        T_val = T_mat[s, r_p]
        w_val = W_RS_FLAT[g]
        Os    = Omega_s_vec[s]

        for dr in 1:R_downstream
            phi_sdr = Phi[s, dr]
            phi_sdr < 1e-300 && continue
            gamma_r_sdr = T_val * (w_val * tau[r_p, dr])^(-theta) / phi_sdr
            exp_sdr = Os * (P_sr[s, dr] / P_r[dr])^(1-nu) *
                      (P_r[dr] / c_r[dr])^(1-lambda) * (1-Omega_L) * mu * Y_r[dr]
            X_ls[r_p, s] += gamma_r_sdr * exp_sdr
        end
    end
    X_s   = vec(sum(X_ls, dims=1))
    X_tot = sum(X_s)
    agg_industry_share = X_s ./ X_tot

    # ── Block 3: Downstream sales shares π_r ────────────────────────────────────
    pi_r = Y_r ./ sum(Y_r)

    # ── Block 4: Regression coefficients via quadrature ─────────────────────────
    reg_coef_sim = compute_regression_quadrature(T_mat, tau, Phi; n_quad=n_quad)

    # ── Block 5: Sourcing shares γ_ls ───────────────────────────────────────────
    gamma_ls = zeros(FT, R, S)
    for s in 1:S
        xs = X_s[s]
        xs < 1e-300 && continue
        for r in 1:R
            gamma_ls[r, s] = X_ls[r, s] / xs * domestic_share[s]
        end
    end

    return (
        agg_labor_share    = [agg_ls],
        agg_industry_share = vec(agg_industry_share),
        pi_r               = pi_r,
        reg_coef           = reg_coef_sim,
        gamma_ls           = gamma_ls,
    )
end


"""
    test_analytical_vs_simulated(params; N_rho_test=10_000)

Numerical equivalence check: compares analytical moments against high-accuracy
SMM at N_rho_test. Prints per-block max relative error.

Acceptance criterion:
  < 1e-3 for blocks {Ω^L, Ω^s, π_r, γ_ls}
  < 1e-2 for reg_coef (quadrature introduces controlled approximation)
"""
function test_analytical_vs_simulated(params; N_rho_test::Int=10_000, n_quad::Int=200)
    println("Generating high-accuracy stratified draws (N_rho=$N_rho_test)...")
    u_draws, sw = generate_stratified_draws(N_rho_test, n_good; randomise=false)

    println("Solving SMM network...")
    network = solve_network(params; u_draws=u_draws, sample_weights=sw)
    moms_sim = compute_moments(network, params)

    println("Computing analytical moments...")
    moms_ana = compute_moments_analytical(params; n_quad=n_quad)

    block_names = ("agg_labor_share", "agg_industry_share", "pi_r", "reg_coef", "gamma_ls")
    eps_tol = [1e-3, 1e-3, 1e-3, 1e-2, 1e-3]

    for (k, name) in enumerate(block_names)
        sim_b = vec(moms_sim[k])
        ana_b = vec(moms_ana[k])
        rel_err = maximum(abs.(sim_b .- ana_b) ./ (abs.(sim_b) .+ 1e-10))
        status = rel_err < eps_tol[k] ? "✓ OK" : "✗ FAIL"
        @printf("  %-20s  max_rel_err = %.2e  tol=%.0e  %s\n",
                name, rel_err, eps_tol[k], status)
    end
end
