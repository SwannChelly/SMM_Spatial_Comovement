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
            # A ZERO-target column (a fake world in which an area active in the data wins
            # nothing — never on the data) is held fixed relative to the reference: the
            # ratio-1 update would let the gauge drift it, since the model γ sums to
            # domestic_share and the target to 1, a common factor the renormalisation
            # removes from every positive column but not from this one.
            for c in cols
                target[c, s] > 0 || (T_new[s, c] = T_par[s, c])
            end
        end
        max_step = 0.0
        for s in 1:S, c in SECTOR_T_COLS[s]
            target[c, s] > 0 || continue
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

# ══════════════════════════════════════════════════════════════════════════════
# HEAD PROFILING (PROFILE_HEAD): (Ω^L, Ω^s, A) from blocks 1–3, jointly with T
# ══════════════════════════════════════════════════════════════════════════════
#
# Given (α, κ, T) the head is just-identified by blocks 1–3: Ω^L by the aggregate
# labour share (1 equation), Ω^s by the sector shares of intermediate spending (S − 1
# free equations, ΣΩ^s = 1), A by the downstream sales shares (R_d − 1, A_1 = 1). All
# of it runs through the EXPECTED price indices P_sr = Γ_s Φ_sr^{−1/θ}, which do not
# depend on the head, so the profile is deterministic:
#
#   ℓ_r  = Ω^L (w_r/c_r)^{1−λ}                         regional labour share (no A)
#   s_rs = Ω_s P_sr^{1−ν} / Σ_k Ω_k P_kr^{1−ν}         sector split of intermediates
#   L    = Σ_r π_r ℓ_r                                 block 1 (cost ∝ sales ∝ π_r)
#   Õ_s  = Σ_r π_r (1−ℓ_r) s_rs / Σ_r π_r (1−ℓ_r)      block 2
#   A_r ∝ c_r (π_r/δ_r)^{−1/ε}                         block 3, closed form, A_1 = 1
#
# π_r is the TARGET sales share throughout: at the profile A reproduces it exactly, so
# the Sinkhorn expenditure weights ω_sr = π_r (1−ℓ_r) s_rs carry no A and A is computed
# once, at the end. Pinning sales also removes the ε channel of the fixed-head T map
# (raise T ⇒ cheaper inputs ⇒ with |ε| large sales jump ⇒ demand returns).
#
# (Ω^L, Ω^s) are solved JOINTLY by Newton on x = (logit Ω^L, log Ω_s[2:S] − log Ω_s[1]):
# the weights π_r (1−ℓ_r) move with Ω^s through P_r, so the two blocks are coupled. The
# system is S × S and costs O(S R_d) per evaluation, negligible beside one Φ pass, so it
# is solved to convergence at every Sinkhorn pass and the T map stays a map of T alone.

"""
    expected_sector_prices(T_ze, tau) -> (P_sr (S × R_d), Φ)
"""
function expected_sector_prices(T_ze::AbstractMatrix, tau::AbstractMatrix)
    Phi  = compute_Phi(T_ze, tau)
    P_sr = similar(Phi)
    @inbounds for s in 1:S, dr in 1:R_downstream
        P_sr[s, dr] = GAMMA_FACTOR[s] * Phi[s, dr]^(-1/theta)
    end
    return P_sr, Phi
end

"""
    head_targets(; L, Os, pi) -> NamedTuple

The blocks the head is profiled on: the labour share, the sector shares renormalised to
the simplex (the model's shares sum to one; only their profile is a target) and the
downstream sales shares (`DOWNSTREAM_REGIONS` order). Defaults: the data.
"""
head_targets(; L = agg_labor_share, Os = agg_industry_share, pi = emp_pi_r) =
    (L = Float64(L), Os = collect(Float64, vec(Os)) ./ sum(Os), pi = collect(Float64, vec(pi)) ./ sum(pi))

"""
    head_shares(x, P_sr) -> (Ω^L, Ω^s, ℓ, s_rs, P_r, c_r) at x = (logit Ω^L, z_2..z_S)
"""
function head_shares(x::AbstractVector, P_sr::AbstractMatrix)
    ΩL = 1 / (1 + exp(-x[1]))
    z  = vcat(0.0, x[2:end])
    Ωs = exp.(z .- maximum(z)); Ωs ./= sum(Ωs)
    num  = Ωs .* P_sr .^ (1 - nu)
    den  = vec(sum(num; dims = 1))
    s_rs = num ./ reshape(den, 1, :)
    P_r  = den .^ (1 / (1 - nu))
    c_r  = similar(P_r); ell = similar(P_r)
    @inbounds for dr in eachindex(P_r)
        w = regional_wages[DOWNSTREAM_REGIONS[dr]]
        c_r[dr] = (ΩL * w^(1 - lambda) + (1 - ΩL) * P_r[dr]^(1 - lambda))^(1 / (1 - lambda))
        ell[dr] = ΩL * (w / c_r[dr])^(1 - lambda)
    end
    return ΩL, Ωs, ell, s_rs, P_r, c_r
end

# Blocks 1–2 at x: S equations (labour share, then sector shares 2..S).
function head_residual(x::AbstractVector, P_sr::AbstractMatrix, tg)
    _, _, ell, s_rs, _, _ = head_shares(x, P_sr)
    wI = tg.pi .* (1 .- ell)
    Ot = (s_rs * wI) ./ sum(wI)
    return vcat(sum(tg.pi .* ell) - tg.L, Ot[2:end] .- tg.Os[2:end])
end

# Newton start: the one-region closed form at the π-weighted mean log prices,
# Ω_s ∝ Õ_s P̄_s^{ν−1}, then logit Ω^L = logit L + (1−λ) log(P̄/w̄).
function head_init(P_sr::AbstractMatrix, tg)
    lP   = log.(P_sr) * tg.pi
    Os   = max.(tg.Os, 1e-12) .* exp.((nu - 1) .* lP); Os ./= sum(Os)
    Pbar = sum(Os .* exp.((1 - nu) .* lP))^(1 / (1 - nu))
    wbar = exp(sum(tg.pi .* log.(regional_wages[DOWNSTREAM_REGIONS])))
    return vcat(log(tg.L / (1 - tg.L)) + (1 - lambda) * log(Pbar / wbar), log.(Os[2:end]) .- log(Os[1]))
end

"""
    solve_head(P_sr; x0, tg, tol=1e-12, max_iter=50) -> (x, converged, resid, iters)

Newton with backtracking on ‖F‖², central-difference Jacobian (S × S; the residual is
exact, so an approximate Jacobian costs speed, not precision). A singular Newton step
falls back to a damped gradient step.
"""
function solve_head(P_sr::AbstractMatrix; x0 = nothing, tg = head_targets(),
                    tol::Float64 = 1e-12, max_iter::Int = 50)
    x = x0 === nothing ? head_init(P_sr, tg) : collect(Float64, x0)
    F = head_residual(x, P_sr, tg)
    n = length(x)
    J = zeros(n, n)
    iters = 0
    for it in 1:max_iter
        iters = it
        maximum(abs, F) < tol && break
        for j in 1:n
            h = 1e-6 * max(1.0, abs(x[j]))
            xp = copy(x); xp[j] += h
            xm = copy(x); xm[j] -= h
            J[:, j] = (head_residual(xp, P_sr, tg) .- head_residual(xm, P_sr, tg)) ./ (2h)
        end
        dx = try -(J \ F) catch; -0.5 .* (J' * F) end
        all(isfinite, dx) || (dx = -0.5 .* (J' * F))
        step = 1.0; improved = false
        for _ in 1:30
            Fn = head_residual(x .+ step .* dx, P_sr, tg)
            if all(isfinite, Fn) && sum(abs2, Fn) < sum(abs2, F)
                x = x .+ step .* dx; F = Fn; improved = true
                break
            end
            step /= 2
        end
        improved || break
    end
    r = maximum(abs, F)
    return (x = x, converged = r < 10 * tol, resid = r, iters = iters)
end

"""
    profile_A(c_r, pi) -> A with A_1 = 1:  π_r ∝ δ_r (c_r/A_r)^ε ⇒ A_r ∝ c_r (π_r/δ_r)^{−1/ε}
"""
function profile_A(c_r::AbstractVector, pi::AbstractVector)
    A = c_r .* (pi ./ delta_r[DOWNSTREAM_REGIONS]) .^ (-1 / epsilon)
    return A ./ A[1]
end

# ZE-level γ (R × S) with the expenditure weights ω (S × R_d) supplied: the arithmetic
# of `gamma_ls_analytical`, whose weights come from the full price solve instead.
function gamma_ze_from_weights(T_ze::AbstractMatrix, tau::AbstractMatrix,
                               Phi::AbstractMatrix, ω::AbstractMatrix)
    X_ls = zeros(R, S)
    @inbounds for g in 1:n_good
        s = GOOD_S[g]; r_p = GOOD_R[g]
        T_val = T_ze[s, r_p]; w_val = W_RS_FLAT[g]
        for dr in 1:R_downstream
            Phi[s, dr] < 1e-300 && continue
            X_ls[r_p, s] += T_val * (w_val * tau[r_p, dr])^(-theta) / Phi[s, dr] * ω[s, dr]
        end
    end
    X_s = vec(sum(X_ls; dims = 1))
    gamma = zeros(R, S)
    @inbounds for s in 1:S, r in 1:R
        X_s[s] < 1e-300 && continue
        gamma[r, s] = X_ls[r, s] / X_s[s] * domestic_share[s]
    end
    return gamma
end

"""
    invert_T_head_ge(alpha; kappa, target=EMP_GAMMA_T_TILDE, tg=head_targets(),
                     T_init=copy(T_rs_init), x_init=nothing, max_iter=600, tol=1e-9,
                     damping=0.3)
        -> (T, Omega_L, Omega_s, A, x, iters, converged, resid, head_resid)

Joint profile of T and the head at (α, κ). Each pass: P_sr from the current T → (Ω^L,
Ω^s) solved at those prices (warm-started from the previous pass) → ω_sr = π_r (1−ℓ_r)
s_rs → closed-form γ → the damped multiplicative T update of `invert_T_ge`, with the same
reference gauge and the same rule for a zero-target column (held fixed relative to the
reference). A is computed once at the end. `converged` requires the T iteration and the
final head solve to converge.
"""
function invert_T_head_ge(alpha; kappa::Real,
                          target::AbstractMatrix = EMP_GAMMA_T_TILDE,
                          tg = head_targets(),
                          T_init::AbstractMatrix = copy(T_rs_init),
                          x_init = nothing,
                          max_iter::Int = 600, tol::Float64 = 1e-9, damping::Float64 = 0.3)
    tau   = build_tau(alpha)
    T_par = Matrix{Float64}(undef, S, T_COL_DIM)
    T_par .= T_init
    for s in 1:S
        ref = T_REF_REGION[s]
        (ref > 0 && T_par[s, ref] > 0) && (T_par[s, :] ./= T_par[s, ref])
    end
    x = x_init
    resid = Inf; iters = 0
    T_new = similar(T_par)
    ω = zeros(S, R_downstream)
    for it in 1:max_iter
        iters = it
        T_ze = gather_T_to_ze(T_par, kappa)
        P_sr, Phi = expected_sector_prices(T_ze, tau)
        x = solve_head(P_sr; x0 = x, tg = tg).x
        _, _, ell, s_rs, _, _ = head_shares(x, P_sr)
        @inbounds for dr in 1:R_downstream, s in 1:S
            ω[s, dr] = tg.pi[dr] * (1 - ell[dr]) * s_rs[s, dr]
        end
        gamma_model = aggregate_gamma_to_T(gamma_ze_from_weights(T_ze, tau, Phi, ω))
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
            for c in cols
                target[c, s] > 0 || (T_new[s, c] = T_par[s, c])
            end
        end
        max_step = 0.0
        for s in 1:S, c in SECTOR_T_COLS[s]
            target[c, s] > 0 || continue
            d = abs(log(T_new[s, c]) - log(T_par[s, c]))
            d > max_step && (max_step = d)
        end
        T_par .= T_new
        resid = max_step
        resid < tol && break
    end
    P_sr, _ = expected_sector_prices(gather_T_to_ze(T_par, kappa), tau)
    h = solve_head(P_sr; x0 = x, tg = tg)
    ΩL, Ωs, _, _, _, c_r = head_shares(h.x, P_sr)
    A = profile_A(c_r, tg.pi)
    return (T = T_par, Omega_L = ΩL, Omega_s = Ωs, A = A, x = h.x, iters = iters,
            converged = resid < tol && h.converged && all(isfinite, A),
            resid = resid, head_resid = h.resid)
end

# Per-worker warm start of the joint profile, (T*, x*) of the last converged call; same
# rule as T_WARM_CACHE (written on convergence only, a failed warm start retried cold).
const HEAD_WARM_CACHE = Ref{Union{Nothing, Tuple{Matrix{Float64}, Vector{Float64}}}}(nothing)

function invert_T_head_ge_warm(alpha; kwargs...)
    cached = HEAD_WARM_CACHE[]
    if cached !== nothing && all(isfinite, cached[1])
        res = invert_T_head_ge(alpha; T_init = copy(cached[1]), x_init = copy(cached[2]), kwargs...)
        res.converged && (HEAD_WARM_CACHE[] = (copy(res.T), copy(res.x)); return res)
    end
    res = invert_T_head_ge(alpha; kwargs...)
    res.converged && (HEAD_WARM_CACHE[] = (copy(res.T), copy(res.x)))
    return res
end

"""
    with_head(x, Ω^L, Ω^s, A) -> copy of the level vector `x` with its head set
"""
function with_head(x::AbstractVector, ΩL::Real, Ωs::AbstractVector, A::AbstractVector)
    xf = collect(Float64, x)
    xf[1] = ΩL
    xf[2:(1 + S)] = Ωs
    xf[(S + 2):(S + R_downstream + 1)] = A
    return xf
end

"""
    head_moment_residuals(Ω^L, Ω^s, A, T_ze, tau; tg) -> (labor, industry, pi_r)

Analytical blocks 1–3 at a θ against their targets (max |model − target| per block),
from the full closed-form price solve. ≈ 0 at a head profile; the gap to the SIMULATED
blocks is the Monte-Carlo error of the realised price indices.
"""
function head_moment_residuals(ΩL, Ωs, A, T_ze::AbstractMatrix, tau::AbstractMatrix;
                               tg = head_targets())
    pr   = compute_prices_analytical(ΩL, Ωs, A, T_ze, tau)
    ell  = ΩL .* (regional_wages[DOWNSTREAM_REGIONS] ./ pr.c_r) .^ (1 - lambda)
    pi_m = pr.Y_r ./ sum(pr.Y_r)
    s_rs = Ωs .* (pr.P_sr ./ reshape(pr.P_r, 1, :)) .^ (1 - nu)
    wI   = pi_m .* (1 .- ell)
    return (labor = abs(sum(pi_m .* ell) - tg.L),
            industry = maximum(abs, (s_rs * wI) ./ sum(wI) .- tg.Os),
            pi_r = maximum(abs, pi_m .- tg.pi))
end

"""
    profiled_theta_full(x) -> (theta, converged, resid, iters, alpha)

`x` with its T block replaced by `T*(α, κ, Ω, A)` — and, under PROFILE_HEAD, its head
too: only [α ; κ] are read from `x`, and (Ω^L, Ω^s, A, T) come from `invert_T_head_ge`
(Ω^s on the simplex, A_1 = 1, as `unpack_params` reads them). On `converged = false` the
blocks are the last iterate: the caller must reject it, not score it.
"""
function profiled_theta_full(x::AbstractVector)
    if PROFILE_HEAD
        α = collect(x[alpha_range()])
        res = invert_T_head_ge_warm(α; kappa = unpack_kappa(x))
        return (theta = with_T(with_head(x, res.Omega_L, res.Omega_s, res.A), res.T),
                converged = res.converged, resid = max(res.resid, res.head_resid),
                iters = res.iters, alpha = α)
    end
    ΩL, Ωs, A, α, κ = normalized_head(x)
    res = invert_T_ge_warm(α, ΩL, Ωs, A; kappa = κ)
    return (theta = with_T(x, res.T), converged = res.converged, resid = res.resid,
            iters = res.iters, alpha = α)
end

profiled_theta(x::AbstractVector) = profiled_theta_full(x).theta
