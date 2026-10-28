##### profiling.jl — T profiled out of the search by a GE-Sinkhorn inversion #####
#
# For a head particle (Ω^L, Ω^s, A, α, κ), T*(·) is the area-level comparative
# advantage that reproduces the observed sourcing shares γ_{s,a} exactly, with the
# destination expenditure made endogenous through the closed-form Eaton–Kortum price
# solve. The γ block is therefore juste-identified; [β_d ; β_κ], π_r, the shares and
# Ḡ_s(0) are fitted by the head. Only the closed-form PRICES are used here — the
# extensive-margin moments always come from the simulation.

using LinearAlgebra
using Printf

"""
    compute_Phi(T_mat, tau) -> Φ (S × R_downstream),  Φ_{s,r} = Σ_l T_ls (w_l τ_lr)^{−θ}
"""
function compute_Phi(T_mat::AbstractMatrix{FT}, tau::AbstractMatrix{FT}) where {FT}
    Phi = zeros(FT, S, R_downstream)
    for g in 1:n_good
        s = GOOD_S[g]; r_p = GOOD_R[g]
        T_val = T_mat[s, r_p]; w_val = W_RS_FLAT[g]
        for dr in 1:R_downstream
            Phi[s, dr] += T_val * (w_val * tau[r_p, dr])^(-theta)
        end
    end
    return Phi
end

"""
    compute_prices_analytical(Ω^L, Ω^s, A, T_mat, tau) -> NamedTuple

Closed-form EK prices: `P_sr = Γ_s Φ_{s,r}^{−1/θ}`, the CES aggregates `P_r`, `c_r`,
downstream prices and sales `Y_r`.
"""
function compute_prices_analytical(Omega_L::FT, Omega_s_vec, A_r, T_mat::AbstractMatrix,
                                   tau::AbstractMatrix) where {FT}
    Phi = compute_Phi(T_mat, tau)
    P_sr = similar(Phi)
    for s in 1:S, dr in 1:R_downstream
        P_sr[s, dr] = GAMMA_FACTOR[s] * Phi[s, dr]^(-1/theta)
    end
    Os_col = reshape(collect(Omega_s_vec), S, 1)
    P_r = vec(sum(Os_col .* P_sr .^ (1 - nu), dims=1)) .^ (1 / (1 - nu))
    c_r = similar(P_r)
    for dr in 1:R_downstream
        r = DOWNSTREAM_REGIONS[dr]
        c_r[dr] = (Omega_L * regional_wages[r]^(1-lambda) +
                   (1-Omega_L) * P_r[dr]^(1-lambda))^(1/(1-lambda))
    end
    c_tilde_r  = c_r ./ A_r
    mu         = FT(epsilon / (epsilon - 1))
    p_r        = c_tilde_r ./ mu
    delta_down = delta_r[DOWNSTREAM_REGIONS]
    P_agg      = sum(p_r .^ epsilon .* delta_down)^(1/epsilon)
    Y_r        = p_r .^ epsilon .* P_agg^(-epsilon) .* delta_down
    return (P_sr=P_sr, P_r=P_r, c_r=c_r, c_tilde_r=c_tilde_r,
            p_r=p_r, P_agg=P_agg, Y_r=Y_r, mu=mu, Phi=Phi)
end

"""
    gamma_ls_analytical(Ω^L, Ω^s, A, T_mat, tau) -> γ (R × S) at the ZE level
"""
function gamma_ls_analytical(Omega_L::Real, Omega_s, A, T_mat::AbstractMatrix, tau::AbstractMatrix)
    (; P_sr, P_r, c_r, Y_r, mu, Phi) = compute_prices_analytical(Omega_L, Omega_s, A, T_mat, tau)
    FT = eltype(Y_r)
    X_ls = zeros(FT, R, S)
    for g in 1:n_good
        s = GOOD_S[g]; r_p = GOOD_R[g]
        T_val = T_mat[s, r_p]; w_val = W_RS_FLAT[g]; Os = Omega_s[s]
        for dr in 1:R_downstream
            phi_sdr = Phi[s, dr]
            phi_sdr < 1e-300 && continue
            exp_sdr = Os * (P_sr[s, dr] / P_r[dr])^(1 - nu) *
                      (P_r[dr] / c_r[dr])^(1 - lambda) * (1 - Omega_L) * mu * Y_r[dr]
            X_ls[r_p, s] += T_val * (w_val * tau[r_p, dr])^(-theta) / phi_sdr * exp_sdr
        end
    end
    X_s = vec(sum(X_ls, dims=1))
    gamma_ls = zeros(FT, R, S)
    for s in 1:S
        X_s[s] < 1e-300 && continue
        for r in 1:R
            gamma_ls[r, s] = X_ls[r, s] / X_s[s] * domestic_share[s]
        end
    end
    return gamma_ls
end

"""
    invert_T_ge(alpha, Ω^L, Ω^s, A; kappa, target=EMP_GAMMA_T_TILDE, T_init=copy(T_rs_init),
                max_iter=600, tol=1e-9, damping=0.3) -> (T, iters, converged, resid)

Damped multiplicative Sinkhorn update in the area-level T space,
`T⁺[s,a] = T[s,a] (target[a,s] / γ_model[a,s])^damping`, renormalised to `T[s,ref] = 1`
each pass. Only the per-sector PROFILE of `target` matters (the reference gauge absorbs
the scale). `damping = 0.3` keeps the ε-driven demand loop stable over the α box (see
old_version/profiling.jl for the measured stability map).
"""
function invert_T_ge(alpha, Omega_L::Real, Omega_s, A;
                     kappa::Real,
                     target::AbstractMatrix = EMP_GAMMA_T_TILDE,
                     T_init::AbstractMatrix = copy(T_rs_init),
                     max_iter::Int = 600, tol::Float64 = 1e-9, damping::Float64 = 0.3)
    tau   = build_tau(alpha)
    T_par = Matrix{Float64}(undef, S, T_COL_DIM)
    T_par .= T_init
    for s in 1:S
        ref = T_REF_REGION[s]
        (ref > 0 && T_par[s, ref] > 0) && (T_par[s, :] ./= T_par[s, ref])
    end
    resid = Inf
    iters = 0
    T_new = similar(T_par)
    for it in 1:max_iter
        iters = it
        gamma_model = aggregate_gamma_to_T(
            gamma_ls_analytical(Omega_L, Omega_s, A, gather_T_to_ze(T_par, kappa), tau))
        T_new .= T_par
        for s in 1:S
            cols = SECTOR_T_COLS[s]
            isempty(cols) && continue
            for c in cols
                gm = gamma_model[c, s]; ge = target[c, s]
                ratio = (gm > 1e-300 && ge > 0) ? ge / gm : 1.0
                T_new[s, c] = exp(log(T_par[s, c]) + damping * log(ratio))
            end
            rv = T_new[s, T_REF_REGION[s]]
            rv > 0 && (T_new[s, cols] ./= rv)
        end
        max_step = 0.0
        for s in 1:S, c in SECTOR_T_COLS[s]
            d = abs(log(T_new[s, c]) - log(T_par[s, c]))
            d > max_step && (max_step = d)
        end
        T_par .= T_new
        resid = max_step
        resid < tol && break
    end
    return (T = T_par, iters = iters, converged = resid < tol, resid = resid)
end

# Per-worker warm start: the last converged T*. The fixed point is unique, so a
# CONVERGED result does not depend on the start; a failed warm attempt is retried from
# T_rs_init, so the value returned never depends on the cache.
const T_WARM_CACHE = Ref{Union{Nothing, Matrix{Float64}}}(nothing)

function invert_T_ge_warm(alpha, Omega_L::Real, Omega_s, A; kwargs...)
    cached = T_WARM_CACHE[]
    if cached !== nothing && all(isfinite, cached)
        res = invert_T_ge(alpha, Omega_L, Omega_s, A; T_init = copy(cached), kwargs...)
        res.converged && (T_WARM_CACHE[] = copy(res.T); return res)
    end
    res = invert_T_ge(alpha, Omega_L, Omega_s, A; kwargs...)
    res.converged && (T_WARM_CACHE[] = copy(res.T))
    return res
end

"""
    normalized_head(x) -> (Ω^L, Ω^s, A, α, κ) as `unpack_params` reads them
"""
function normalized_head(x::AbstractVector)
    Ωs = x[2:(1 + S)]; A = x[(S + 2):(S + R_downstream + 1)]
    return x[1], Ωs ./ sum(Ωs), A ./ A[1], collect(x[alpha_range()]), unpack_kappa(x)
end

"""
    with_T(x, T_par) -> copy of the level vector `x` with its T block set to `T_par`
"""
function with_T(x::AbstractVector, T_par::AbstractMatrix)
    xf = copy(x)
    xf[t_param_start():end] = vec(permutedims(T_par))[T_MASK]
    return xf
end

"""
    profiled_theta_full(x) -> (theta, converged, resid, iters, alpha)

`x` with its T block replaced by `T*(α, κ, Ω, A)`. On `converged = false` the T block
is the last iterate: the caller must reject it, not score it.
"""
function profiled_theta_full(x::AbstractVector)
    ΩL, Ωs, A, α, κ = normalized_head(x)
    res = invert_T_ge_warm(α, ΩL, Ωs, A; kappa = κ)
    return (theta = with_T(x, res.T), converged = res.converged, resid = res.resid,
            iters = res.iters, alpha = α)
end

profiled_theta(x::AbstractVector) = profiled_theta_full(x).theta
