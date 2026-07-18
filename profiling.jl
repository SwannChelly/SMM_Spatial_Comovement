##### profiling.jl — internal T-profiling via GE-Sinkhorn inversion #####
#
# Profiles the comparative-advantage block T OUT of the outer search. Given a
# particle (α, Ω^L, Ω^s, A) the outer optimizer no longer searches T; instead this
# file computes the T*(α, Ω, A) that reproduces the observed sourcing shares γ_ls,
# then the loss is evaluated on the remaining moments. The theory (unique matrix
# scaling; see documentation/initialisation.md) guarantees a unique T* once T is
# normalized to its per-sector reference region.
#
# DESIGN A (exact γ_ls inversion — the plan's recommendation). T*(α,Ω,A) is the
# Sinkhorn fixed point of the closed-form sourcing map, with the destination
# expenditure weights ω = exp_{s,dr}(T) made ENDOGENOUS through the exact EK price
# solve compute_prices_analytical. The γ_ls block is thereby juste-identified
# (loss ≈ 0); reg_coef is fit by (α,Ω,A) alone in the outer search.
#
# ── STRICT SEPARATION (plan §7.1) ────────────────────────────────────────────
# The inversion uses ONLY compute_prices_analytical (the closed-form EK prices —
# exact) to drive endogenous expenditure. It NEVER calls compute_regression_quadrature
# (analytically biased by FKG, see model_analytical.jl:108-111). The reg_coef of
# the production loss keeps coming from the SMM simulation (solve_network /
# linkages_flat, correct). This file touches no production path: it defines new
# functions only, gated behind the profile_T flag at the call sites.
#
# ── CONVERGENCE (plan §4) ─────────────────────────────────────────────────────
# With endogenous ω this is no longer a pure matrix scaling. The one-step map Ψ
# linearizes as  ∂Ψ/∂logT = J_Sink (Birkhoff contraction κ_S<1) + J_GE, with
# ‖J_GE‖ ∝ |1−ν|, |1−λ|. At this calibration (ν=0.2, λ=0.5) the GE channel is NOT
# a small perturbation, so convergence/uniqueness must be established empirically —
# see test/test_ge_inversion.jl (Phase-0 gate). invert_T_ge is damped and returns
# a `converged` flag so callers can fall back on the warm start if the gate fails.
#
# ── GAUGE (plan §7.5) ─────────────────────────────────────────────────────────
# The returned T is per-sector reference-normalized (T[s,ref]=1), exactly like
# unpack_params' `T[s,:] /= T[s,ref]`. The reduced (T_MASK, s-major) block for the
# parameter vector is `vec(permutedims(T))[T_MASK]` (matches optimizer.jl:230 and
# the CLAUDE.md s-major invariant).

using LinearAlgebra
using Printf


"""
    gamma_ls_analytical(Omega_L, Omega_s, A, T_mat, tau) -> (gamma_ls::(R,S), Phi, X_ls)

The exact closed-form sourcing shares γ_ls of the analytical model, replicating
ONLY blocks 2 & 5 of `compute_moments_analytical` (model_analytical.jl:243-280) —
no reg_coef quadrature is touched. Endogenous destination expenditure comes from
`compute_prices_analytical`. `Omega_s`/`A` are expected already normalized (as
`unpack_params` produces). Generic in `eltype`.
"""
function gamma_ls_analytical(Omega_L::Real, Omega_s, A, T_mat::AbstractMatrix, tau::AbstractMatrix)
    prices = compute_prices_analytical(Omega_L, Omega_s, A, T_mat, tau)
    (; P_sr, P_r, c_r, Y_r, mu, Phi) = prices
    FT = eltype(Y_r)

    X_ls = zeros(FT, R, S)
    for g in 1:n_good
        s     = GOOD_S[g]
        r_p   = GOOD_R[g]
        T_val = T_mat[s, r_p]
        w_val = W_RS_FLAT[g]
        Os    = Omega_s[s]
        for dr in 1:R_downstream
            phi_sdr = Phi[s, dr]
            phi_sdr < 1e-300 && continue
            gamma_r_sdr = T_val * (w_val * tau[r_p, dr])^(-theta) / phi_sdr
            exp_sdr = Os * (P_sr[s, dr] / P_r[dr])^(1 - nu) *
                      (P_r[dr] / c_r[dr])^(1 - lambda) * (1 - Omega_L) * mu * Y_r[dr]
            X_ls[r_p, s] += gamma_r_sdr * exp_sdr
        end
    end

    X_s = vec(sum(X_ls, dims=1))
    gamma_ls = zeros(FT, R, S)
    for s in 1:S
        xs = X_s[s]
        xs < 1e-300 && continue
        for r in 1:R
            gamma_ls[r, s] = X_ls[r, s] / xs * domestic_share[s]
        end
    end
    return gamma_ls, Phi, X_ls
end


"""
    invert_T_ge(alpha, Omega_L, Omega_s, A;
                target = emp_gamma_ls, T_init = copy(T_rs_init),
                max_iter = 500, tol = 1e-9, damping = 0.9, verbose = false)
        -> (T::(S,R), iters, converged::Bool, resid, resid_hist)

`damping` is the log-space relaxation δ (δ=1 is pure Sinkhorn). Phase-0 measured
the undamped map as a strong contraction (ρ_full≈0.01 at the aero calibration),
so the default is 0.9 (fast) rather than the very conservative 0.5; drop it toward
0.5 if a far-from-init particle ever oscillates.

Profile T by the GE-Sinkhorn inversion of `target` (default: observed
`emp_gamma_ls`). The damped multiplicative update, per active (s,r), is

    T⁺[s,r] = T[s,r] · ( target[r,s] / γ_model[r,s] )^damping ,   then T[s,·] /= T[s,ref]

where γ_model = `gamma_ls_analytical(..., T)` carries the endogenous expenditure.
Only the profile of `target` is matched (its total is not identified — the
per-sector reference renormalization absorbs the neutral scale, exactly as in
`invert_T_from_gamma` and initialisation.md §2.1). Deterministic ⇒ no noise added
to the outer PSO. Returns the reference-normalized (S,R) T and convergence info;
`converged=false` (resid ≥ tol at max_iter) signals the caller to fall back on
`T_init`.
"""
function invert_T_ge(alpha, Omega_L::Real, Omega_s, A;
                     target::AbstractMatrix = emp_gamma_ls,
                     T_init::AbstractMatrix = copy(T_rs_init),
                     max_iter::Int = 500, tol::Float64 = 1e-9,
                     damping::Float64 = 0.9, verbose::Bool = false)
    tau   = build_tau(alpha)
    T_mat = Matrix{Float64}(undef, S, R)
    T_mat .= T_init

    # Reference-normalize the warm start per sector (gauge: T[s,ref] = 1).
    for s in 1:S
        ref = T_REF_REGION[s]
        (ref > 0 && T_mat[s, ref] > 0) && (T_mat[s, :] ./= T_mat[s, ref])
    end

    resid      = Inf
    iters      = 0
    resid_hist = Float64[]
    T_new      = similar(T_mat)

    for it in 1:max_iter
        iters = it
        gamma_model, _, _ = gamma_ls_analytical(Omega_L, Omega_s, A, T_mat, tau)

        T_new .= T_mat
        for s in 1:S
            regs = SECTOR_GOOD_REGIONS[s]
            isempty(regs) && continue
            ref = T_REF_REGION[s] > 0 ? T_REF_REGION[s] : regs[1]
            for r in regs
                gm    = gamma_model[r, s]
                ge    = target[r, s]
                ratio = (gm > 1e-300 && ge > 0) ? ge / gm : 1.0
                T_new[s, r] = exp(log(T_mat[s, r]) + damping * log(ratio))
            end
            rv = T_new[s, ref]
            rv > 0 && (T_new[s, regs] ./= rv)
        end

        max_step = 0.0
        for s in 1:S, r in SECTOR_GOOD_REGIONS[s]
            d = abs(log(T_new[s, r]) - log(T_mat[s, r]))
            d > max_step && (max_step = d)
        end
        T_mat .= T_new
        resid = max_step
        push!(resid_hist, resid)
        verbose && @printf("    [invert_T_ge] it=%3d  max|Δlog T|=%.3e\n", it, resid)
        resid < tol && break
    end

    converged = resid < tol
    return (T = T_mat, iters = iters, converged = converged,
            resid = resid, resid_hist = resid_hist)
end


"""
    profiled_theta(x_levels) -> Vector{Float64}

Given a full LEVEL parameter vector `x_levels = [Ω^L | Ω^s | A | α | T]`, return a
copy whose T block is replaced by the profiled `T*(α,Ω,A) = invert_T_ge(...)`. The
head is normalized exactly as `unpack_params` reads it (Ω^s ÷ sum, A ÷ A[1]) before
the inversion, and the reduced T* block is written in s-major `T_MASK` order. This
is the single interception point the optimizer uses when `profile_T=true`; kept a
top-level (`@everywhere`) function so it serializes cleanly across the PSO's `pmap`
workers (a nested closure calling `invert_T_ge` would not).
"""
function profiled_theta(x_levels::AbstractVector)
    return profiled_theta_full(x_levels).theta
end


"""
    profiled_theta_full(x_levels) -> (theta, converged::Bool, resid, iters)

Same as `profiled_theta` but also surfaces the `invert_T_ge` convergence status of
the T inversion for this particle. Used both by `profiled_theta` (which keeps only
`.theta`) and by the PSO per-report T-convergence diagnostic (which keeps only the
`.converged` flag). Kept top-level so it serializes across the `pmap` workers.
"""
function profiled_theta_full(x_levels::AbstractVector)
    ΩL = x_levels[1]
    Ωs = x_levels[2:(1 + S)];                      Ωs = Ωs ./ sum(Ωs)
    A  = x_levels[(S + 2):(S + R_downstream + 1)]; A  = A ./ A[1]
    α  = x_levels[(S + R_downstream + 2):(1 + S + R_downstream + N_TAU)]
    res = invert_T_ge(α, ΩL, Ωs, A)
    xf = copy(x_levels)
    xf[(2 + S + R_downstream + N_TAU):end] = vec(permutedims(res.T))[T_MASK]
    return (theta = xf, converged = res.converged, resid = res.resid, iters = res.iters)
end


"""
    profiled_theta_converged(x_levels) -> Bool

Whether the GE-Sinkhorn T inversion converged for a given head particle. Skips the
T-block scatter that `profiled_theta_full` does (only the flag is needed), so it is
the cheap probe the PSO report runs over the whole swarm. Any error in the inversion
(non-finite head, etc.) is treated as non-converged.
"""
function profiled_theta_converged(x_levels::AbstractVector)::Bool
    try
        ΩL = x_levels[1]
        Ωs = x_levels[2:(1 + S)];                      Ωs = Ωs ./ sum(Ωs)
        A  = x_levels[(S + 2):(S + R_downstream + 1)]; A  = A ./ A[1]
        α  = x_levels[(S + R_downstream + 2):(1 + S + R_downstream + N_TAU)]
        return invert_T_ge(α, ΩL, Ωs, A).converged
    catch
        return false
    end
end


"""
    assemble_theta(Omega_L, Omega_s, A, alpha, T_mat) -> Vector{Float64}

Assemble the full LEVEL parameter vector `[Ω^L | Ω^s | A | α | T(reduced, s-major)]`
from a profiled T (S,R). The reduced T block is `vec(permutedims(T_mat))[T_MASK]`
(s-major, T_MASK order — matches optimizer.jl:230). Head blocks are inserted as
given (the outer search supplies them raw; unpack_params normalizes Ω^s/A on read,
so pass them exactly as the particle carries them).
"""
function assemble_theta(Omega_L::Real, Omega_s, A, alpha, T_mat::AbstractMatrix)
    T_red = vec(permutedims(T_mat))[T_MASK]
    return vcat(Float64(Omega_L), collect(Float64, Omega_s), collect(Float64, A),
                collect(Float64, alpha), collect(Float64, T_red))
end
