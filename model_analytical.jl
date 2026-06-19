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
    compute_regression_quadrature(T_mat, tau, Phi; n_quad=200) -> Vector (N_REG,)

Distance-bin regression coefficients via deterministic Gauss-Legendre quadrature.

Regressand (Eq. B6): the whole-industry extensive margin
    ρ̃_{r's}(z) = 1 − ∏_{dr=1}^{R_downstream} (1 − ρ_{r'dr s}(z))
where ρ_{r'dr s}(z) = exp(z^{-θ} · coef_dr) is the per-destination win probability
and coef_dr = T_{r's} − Φ_{s,dr}·(w_{r's}τ_{r'dr})^θ ≤ 0.

Regressors (distance-bin dummies or log distance) and FE group are keyed on the
nearest downstream region CLOSEST_DOWNSTREAM_REGION[r_p], matching Eq. (1) and
fast_weighted_regression. Only the dependent variable changes: from the single
closest-destination win probability to the product across all downstream regions.

After weighted FWL demeaning by (sector × closest downstream), WLS gives β_{1..N_REG}.

Caveat: ρ̃_{r's}(z) assumes conditional independence across destinations. The SMM
linkages_flat realises the true win-anywhere event with shared competitor draws
(positively correlated), so the product over-states the extensive margin (FKG/Harris).
Expect max_rel_err ≈ 1e-2 vs. simulated moments, not machine zero.
"""
function compute_regression_quadrature(T_mat, tau, Phi; n_quad::Int=200)
    FT = eltype(Phi)

    nodes_raw, weights_raw = gausslegendre(n_quad)
    u_nodes = (nodes_raw .+ 1) ./ 2
    gl_wts  = weights_raw ./ 2

    N_total = n_good * n_quad
    n_reg   = N_REG + 1
    y        = Vector{FT}(undef, N_total)
    X        = zeros(FT, N_total, n_reg)
    w_arr    = Vector{Float64}(undef, N_total)
    fe_group = Vector{Int}(undef, N_total)

    coef = Vector{FT}(undef, R_downstream)   # reused buffer, no per-g allocation

    idx = 0
    for g in 1:n_good
        s     = GOOD_S[g]
        r_p   = GOOD_R[g]
        T_val = T_mat[s, r_p]
        w_val = W_RS_FLAT[g]

        # Regressors / FE keyed on the nearest downstream region (matches Eq. (1)
        # and fast_weighted_regression). Unchanged.
        dr0 = CLOSEST_DOWNSTREAM_REGION[r_p]
        gid = (s - 1) * R_downstream + dr0
        if N_REG == 1
            log_dist = LOG_CLOSEST_DIST[r_p]
        else
            b = DistBin[r_p, dr0]
        end

        # Per-destination exponent coefficient; coef_dr ≤ 0 (Φ_{s,dr}(wτ)^θ ≥ T_{r's}).
        @inbounds for dr in 1:R_downstream
            coef[dr] = T_val - Phi[s, dr] * (w_val * tau[r_p, dr])^theta
        end

        scale = T_val^(1/theta)

        for k in 1:n_quad
            idx += 1
            u_k  = min(u_nodes[k], 1.0 - 1e-14)
            z_k  = scale * (-log(1.0 - u_k))^(-1.0/theta)
            zinv = z_k^(-theta)

            # Extensive margin (Eq. B6): 1 − ∏_dr (1 − ρ_{r'dr s}(z)),
            # ρ_dr = exp(zinv·coef_dr), coef_dr ≤ 0 ⇒ ρ_dr ∈ (0,1].
            logprod = zero(FT)
            @inbounds for dr in 1:R_downstream
                logprod += log1p(max(-exp(zinv * coef[dr]), -one(FT)))  # log(1 − ρ_dr); clamp guards FP coef>0
            end
            y[idx]        = -expm1(logprod)   # 1 − ∏(1 − ρ_dr)
            w_arr[idx]    = gl_wts[k]
            fe_group[idx] = gid

            if N_REG == 1
                X[idx, 1] = log_dist
            else
                (b > 0 && b <= N_REG) && (X[idx, b] = FT(1.0))
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

    return coefs[1:N_REG]
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
function test_analytical_vs_simulated(params; N_rho_test::Int=10_000, n_quad::Int=200,
                                      draw_method::Symbol=DRAW_METHOD)
    println("Generating high-accuracy draws (method=:$draw_method, N_rho=$N_rho_test)...")
    u_draws, sw = generate_draws(N_rho_test, n_good, draw_method; randomise=false)

    println("Solving SMM network...")
    network  = solve_network(params; u_draws=u_draws, sample_weights=sw)
    moms_sim = compute_moments(network, params)

    println("Computing analytical moments...")
    moms_ana = compute_moments_analytical(params; n_quad=n_quad)

    block_names = ("agg_labor_share", "agg_industry_share", "pi_r", "reg_coef", "gamma_ls")
    eps_tol     = [1e-3, 1e-3, 1e-3, 1e-2, 1e-3]

    # ── (a) per-block scalar (unchanged output) ─────────────────────────────
    for (k, name) in enumerate(block_names)
        sim_b = vec(moms_sim[k]); ana_b = vec(moms_ana[k])
        rel_err = maximum(abs.(sim_b .- ana_b) ./ (abs.(sim_b) .+ 1e-10))
        status  = rel_err < eps_tol[k] ? "✓ OK" : "✗ FAIL"
        @printf("  %-20s  max_rel_err = %.2e  tol=%.0e  %s\n", name, rel_err, eps_tol[k], status)
    end

    # ─────────────────────────────────────────────────────────────────────────
    # DIAGNOSTIC INSTRUMENTATION (print-only; no effect on return/criteria)
    # ─────────────────────────────────────────────────────────────────────────
    println("\n" * "="^72)
    println("DIAGNOSTIC: localising the analytical/simulated gap")
    println("="^72)

    # ── (c) shared price-solver intermediates, region by region ─────────────
    # Both moment functions descend from the same prices; if these diverge the
    # gap is upstream of any single block. Recompute the analytical prices and
    # compare to the simulated network on DOWNSTREAM regions.
    Ω_L, Ω_s, A, β, T_vec = unpack_params(params)
    T_mat = reshape(T_vec, S, R)
    τ     = build_tau(β)
    pr    = compute_prices_analytical(Ω_L, Ω_s, A, T_mat, τ)

    c_sim = network.c_tilde_r[DOWNSTREAM_REGIONS]   # padded → restrict to downstream
    c_ana = pr.c_tilde_r
    Y_sim = network.Y_r[DOWNSTREAM_REGIONS]
    Y_ana = pr.Y_r
    relv(a, b) = abs(a - b) / (abs(a) + 1e-12)

    @printf("  c_tilde_r : max_rel=%.3e  mean_rel=%.3e\n",
            maximum(relv.(c_sim, c_ana)), mean(relv.(c_sim, c_ana)))
    @printf("  Y_r       : max_rel=%.3e  mean_rel=%.3e\n",
            maximum(relv.(Y_sim, Y_ana)), mean(relv.(Y_sim, Y_ana)))
    # worst downstream region for c_tilde — the entry to chase if prices diverge
    wd = argmax(relv.(c_sim, c_ana))
    @printf("    worst c̃_r at downstream idx %d: sim=%.6e ana=%.6e\n",
            wd, c_sim[wd], c_ana[wd])

    # ── (b) γ_ls error per ACTIVE (s,r) pair, joined to structure ────────────
    # γ is closed-form on both sides (no quadrature) so any gap is a construction
    # mismatch. We tabulate per-sector and test the renormalisation hypothesis:
    # c_s = sum_before/sum_after (threshold renorm). emp_gamma_ls is post-thresh.
    γ_sim = moms_sim[5]   # (R, S)
    γ_ana = moms_ana[5]   # (R, S)

    # reconstruct c_s = sum_before / sum_after per sector from the RAW file
    X_rs_raw = NPZ.npzread(joinpath(input_folder, "X_rs.npy"))   # (S,R), pre-threshold mask
    emp_raw  = permutedims(NPZ.npzread(joinpath(input_folder, "emp_gamma_ls.npy")))  # (R,S) raw
    println("\n  Per-sector γ_ls gap vs structure:")
    @printf("  %-8s %7s %10s %10s %10s %10s\n",
            "sector", "n_act", "max_rel", "mean_rel", "c_s", "mean|γ|")
    sector_gap = Float64[]
    sector_cs  = Float64[]
    for s in 1:S
        regs = SECTOR_GOOD_REGIONS[s]
        isempty(regs) && continue
        e = [relv(γ_sim[r, s], γ_ana[r, s]) for r in regs]
        # c_s: ratio of pre-threshold sector mass to post-threshold survivor mass
        col_raw   = @view emp_raw[:, s]
        sum_before = sum(col_raw)
        sum_after  = sum(col_raw[r] for r in 1:R if emp_gamma_ls[r, s] > 0; init=0.0)
        c_s = sum_after > 1e-15 ? sum_before / sum_after : NaN
        mean_g = mean(abs.(γ_sim[regs, s]))
        push!(sector_gap, mean(e)); push!(sector_cs, c_s)
        @printf("  %-8d %7d %10.3e %10.3e %10.4f %10.4e\n",
                s, length(regs), maximum(e), mean(e), c_s, mean_g)
    end

    # hypothesis 3a: does per-sector γ gap correlate with c_s (renorm) or with
    # coverage (n active)? A high corr with c_s ⇒ threshold-renorm mismatch.
    if length(sector_gap) > 2
        cc = cor(sector_gap, sector_cs)
        @printf("\n  corr( per-sector γ gap , c_s )      = %+.3f\n", cc)
        @printf("  (high positive ⇒ threshold-renormalisation mismatch is the driver)\n")
    end

    # ── per-entry γ gap vs |γ| magnitude (hypothesis: small shares diverge) ──
    big = Tuple{Float64,Int,Int,Float64,Float64}[]   # (rel, s, r, γ_sim, γ_ana)
    for s in 1:S, r in SECTOR_GOOD_REGIONS[s]
        push!(big, (relv(γ_sim[r, s], γ_ana[r, s]), s, r, γ_sim[r, s], γ_ana[r, s]))
    end
    sort!(big, by = x -> -x[1])
    println("\n  Top-8 worst γ entries (rel | s | r | γ_sim | γ_ana):")
    for i in 1:min(8, length(big))
        rel, s, r, gs, ga = big[i]
        @printf("    %.3e  s=%-3d r=%-4d  sim=%.5e  ana=%.5e  ratio=%.3f\n",
                rel, s, r, gs, ga, ga / (gs + 1e-12))
    end

    # ── (b') reg_coef: raw coefficients, not rel_err — sign/scale tells bug ──
    # vs documented FKG bias. If signs match and ana ≈ k·sim (k≈const>1), it's
    # the conditional-independence over-statement; if signs flip, it's a bug.
    rc_sim = vec(moms_sim[4]); rc_ana = vec(moms_ana[4])
    println("\n  reg_coef raw (bin | sim | ana | ana/sim):")
    for b in 1:length(rc_sim)
        @printf("    bin %d:  sim=%+.5e  ana=%+.5e  ratio=%+.3f\n",
                b, rc_sim[b], rc_ana[b], rc_ana[b] / (rc_sim[b] + 1e-12))
    end

    # ── (b'') labor share: the exact block that should NOT diverge ───────────
    @printf("\n  labor share:  sim=%.6e  ana=%.6e  rel=%.3e  (exact block — gap ⇒ shared cause)\n",
            moms_sim[1][1], moms_ana[1][1], relv(moms_sim[1][1], moms_ana[1][1]))
    println("="^72)
end