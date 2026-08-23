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
                target = EMP_GAMMA_T_TILDE, T_init = copy(T_rs_init),
                max_iter = 2000, tol = 1e-9, damping = 0.5, verbose = false)
        -> (T::(S,T_COL_DIM), iters, converged::Bool, resid, resid_hist)

`damping` is the log-space relaxation δ (δ=1 is pure Sinkhorn). The update is
Richardson iteration on log T against M = ∂log γ/∂log T, so it is stable iff
δ < 2/λ_max(M) and fastest at δ = 2/(λ_min+λ_max).

**Why the default is 0.5 and not 0.9.** The original 0.9 was set from a Phase-0
measurement taken at α ≈ 0.1 with a default head, where the map is nearly the
identity (λ_max ≈ 1.0, i.e. the multiplicative update is an exact Newton step) and
ρ_full ≈ 0.01. That is not representative: λ_max grows with α through the
ε-driven demand loop — raising T in an area cuts input prices in the destinations
it serves, and with ε = −16 their sales jump, sending demand back to that area.
`test/test_ge_inversion.jl` measures the whole map on aero: the pure matrix-scaling
channel contributes λ_max ≡ 1.000 at EVERY α and the ν/λ price channels add
nothing, while the ε channel takes λ_max from 1.006 at α = 0.1 to 2.31 at α = 0.8.
Empirically (its section 5b/c, from the production warm start) δ = 0.9 converges
only up to α ≈ 0.50, δ = 0.5 up to α ≈ 0.70, δ = 0.3 up to α ≈ 1.10. Under
`profile_T` a non-converging particle was still SCORED (see `profiled_theta_full`),
so the outer PSO read the solver's ceiling as a wall in the criterion and α could
not travel past it. At α̂ = 0.43 the cost of the change is 34 iterations instead of
32 — the inversion is ~13% of one loss — so 0.5 is close to free.

`max_iter` is 2000 rather than 500 for the same reason: at δ = 0.5 the iteration
count rises with α (34 at α̂, 60 at α = 0.7), and a budget that truncates a
converging run is indistinguishable downstream from a divergent one.

Profile T by the GE-Sinkhorn inversion of `target`. Both `target` and the returned
`T` live in the T-COLUMN space (`T_COL_DIM` wide): upstream ZE under `CA_LEVEL == :ze`,
attraction areas under `:aa`. The default target is the
DOMESTIC-SHARE-BALANCED γ̃ = `EMP_GAMMA_T_TILDE` = EMP_GAMMA_T / domestic_share,
whose per-sector total is 1 — matching the expenditure (column) margin ω, so the
Sinkhorn margins are COMPATIBLE (Σ row = Σ column) as the theorem requires. Passing
the raw `emp_gamma_ls` (Σ_r = domestic_share < 1) would leave the transport problem
unbalanced; the per-sector reference gauge absorbs the neutral scale so the returned
T is identical either way, but the balanced target makes the precondition explicit
(initialisation.md §2.1). The damped multiplicative update, per active (s,r), is

    T⁺[s,r] = T[s,r] · ( target[r,s] / γ_model[r,s] )^damping ,   then T[s,·] /= T[s,ref]

where γ_model = `gamma_ls_analytical(..., T)` carries the endogenous expenditure.
Only the profile of `target` is matched (its total is not identified — the
per-sector reference renormalization absorbs the neutral scale). Deterministic ⇒ no
noise added to the outer PSO. Returns the reference-normalized (S,R) T and
convergence info; `converged=false` (resid ≥ tol at max_iter) signals the caller to
fall back on `T_init`.
"""
function invert_T_ge(alpha, Omega_L::Real, Omega_s, A;
                     target::AbstractMatrix = EMP_GAMMA_T_TILDE,
                     T_init::AbstractMatrix = copy(T_rs_init),
                     max_iter::Int = 2000, tol::Float64 = 1e-9,
                     damping::Float64 = 0.5, verbose::Bool = false)
    tau   = build_tau(alpha)
    # The inversion iterates in the T-COLUMN space: ZE under :ze, attraction areas
    # under :aa. Each pass gathers T onto the ZE the model simulates, evaluates the
    # closed-form ZE-level γ, and aggregates it back to the T columns — so the system
    # stays square (one multiplicative update per active (s, column)) and the
    # contraction is inherited from the ZE-level version.
    T_par = Matrix{Float64}(undef, S, T_COL_DIM)
    T_par .= T_init

    # Reference-normalize the warm start per sector (gauge: T[s,ref] = 1).
    for s in 1:S
        ref = T_REF_REGION[s]
        (ref > 0 && T_par[s, ref] > 0) && (T_par[s, :] ./= T_par[s, ref])
    end

    resid      = Inf
    iters      = 0
    resid_hist = Float64[]
    T_new      = similar(T_par)

    for it in 1:max_iter
        iters = it
        gamma_ze, _, _ = gamma_ls_analytical(Omega_L, Omega_s, A, gather_T_to_ze(T_par), tau)
        gamma_model    = aggregate_gamma_to_T(gamma_ze)

        T_new .= T_par
        for s in 1:S
            cols = SECTOR_T_COLS[s]
            isempty(cols) && continue
            ref = T_REF_REGION[s] > 0 ? T_REF_REGION[s] : cols[1]
            for c in cols
                gm    = gamma_model[c, s]
                ge    = target[c, s]
                ratio = (gm > 1e-300 && ge > 0) ? ge / gm : 1.0
                T_new[s, c] = exp(log(T_par[s, c]) + damping * log(ratio))
            end
            rv = T_new[s, ref]
            rv > 0 && (T_new[s, cols] ./= rv)
        end

        max_step = 0.0
        for s in 1:S, c in SECTOR_T_COLS[s]
            d = abs(log(T_new[s, c]) - log(T_par[s, c]))
            d > max_step && (max_step = d)
        end
        T_par .= T_new
        resid = max_step
        push!(resid_hist, resid)
        verbose && @printf("    [invert_T_ge] it=%3d  max|Δlog T|=%.3e\n", it, resid)
        resid < tol && break
    end

    converged = resid < tol
    return (T = T_par, iters = iters, converged = converged,
            resid = resid, resid_hist = resid_hist)
end


# ── Warm-start cache for the profiled inversion ────────────────────────────────
# `invert_T_ge`'s default start is `T_rs_init`, the γ-inversion at the PRIOR α with a
# default head. Under `profile_T` the PSO evaluates hundreds of nearby particles per
# iteration, so that start is the SAME distant point every time even once the swarm
# has concentrated somewhere else entirely — and at higher α the failures measured by
# `test/test_ge_inversion.jl` are BASIN failures (ρ(I−δM) < 1, so the fixed point is
# locally attracting, yet the run started from `T_rs_init` still oscillates). Starting
# from the last T* this worker actually solved attacks that directly.
#
# One slot per worker (`profiling.jl` is `@everywhere include`d, and PSO's `pmap`
# hands each worker a stream of particles from the same swarm, hence from a similar
# region of the head box). `Ref` rather than a Dict: nothing here is a memo — a cached
# T* is only ever a STARTING POINT, never a returned value.
#
# DETERMINISM, which is the reason for the two-stage structure below. The fixed point
# is unique (test section 3: 10/10 random starts agree to 2.5e-11), so a CONVERGED
# result does not depend on where the iteration began. A NON-converged result does.
# So the cache is written only on convergence, and a warm-started run that fails is
# retried from `T_rs_init` — which means the value returned to the optimizer is
# exactly what it would have been without the cache, converged or not. The cache can
# change the iteration count and whether a particle converges; it can never change the
# T* of a particle that converges, nor the fallback T of one that does not.
const T_WARM_CACHE = Ref{Union{Nothing, Matrix{Float64}}}(nothing)

"""
    invert_T_ge_warm(alpha, Omega_L, Omega_s, A; kwargs...) -> same as invert_T_ge

`invert_T_ge` started from this worker's last converged T* when there is one, falling
back to the ordinary `T_rs_init` start if the warm attempt fails. See the note above
for why that fallback is what keeps the profiled objective deterministic. Only the
`profile_T` optimisation path uses this; the inference Jacobians in `tools.jl` keep
calling `invert_T_ge` directly with their own explicit `T_init`, since each of their
perturbations must start from the same pinned point to difference cleanly.
"""
function invert_T_ge_warm(alpha, Omega_L::Real, Omega_s, A; kwargs...)
    cached = T_WARM_CACHE[]
    if cached !== nothing && size(cached) == (S, T_COL_DIM) && all(isfinite, cached)
        res = invert_T_ge(alpha, Omega_L, Omega_s, A; T_init = copy(cached), kwargs...)
        if res.converged
            T_WARM_CACHE[] = copy(res.T)
            return res
        end
    end
    res = invert_T_ge(alpha, Omega_L, Omega_s, A; kwargs...)   # T_init = T_rs_init
    res.converged && (T_WARM_CACHE[] = copy(res.T))
    return res
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
the T inversion for this particle. Routed through `invert_T_ge_warm`, so it starts
from this worker's last converged T*.

⚠ On `converged = false` the T block is still filled with the LAST ITERATE — this
function does not fall back and does not penalise. The caller must act on the flag
(`optimizer.jl`'s objective should return `Inf` rather than score a fabricated T);
otherwise the outer search reads the solver's failure region as a region of bad fit. Used both by `profiled_theta` (which keeps only
`.theta`) and by the PSO per-report T-convergence diagnostic (which keeps only the
`.converged` flag). Kept top-level so it serializes across the `pmap` workers.
"""
function profiled_theta_full(x_levels::AbstractVector)
    ΩL = x_levels[1]
    Ωs = x_levels[2:(1 + S)];                      Ωs = Ωs ./ sum(Ωs)
    A  = x_levels[(S + 2):(S + R_downstream + 1)]; A  = A ./ A[1]
    α  = x_levels[(S + R_downstream + 2):(1 + S + R_downstream + N_TAU)]
    res = invert_T_ge_warm(α, ΩL, Ωs, A)
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
    return profiled_theta_probe(x_levels).converged
end


"""
    profiled_theta_probe(x_levels) -> (converged::Bool, alpha, iters::Int, resid::Float64)

The T-inversion health of one head particle, with the covariates needed to attribute
a failure. Same computation as `profiled_theta_converged` (the T-block scatter is
skipped — only the diagnostic is wanted), but it also returns the particle's α, the
iteration count and the terminal residual, so the caller can report the non-convergence
rate BY α rather than as one pooled number.

That distinction is the whole point: a rate that rises monotonically in α says the α
box is too wide; a rate that is flat in α but nonzero says the failures live somewhere
else in the head (Ω^L, Ω^s, A) and narrowing α will not fix them. `resid` separates the
two failure modes further — a residual just above `tol` is a particle that needed more
iterations, one many orders above it is a particle the map is not contracting for.

Any error in the inversion (non-finite head, an underflowed T column) counts as
non-converged, with `resid = Inf`; α is still reported when it can be read.
"""
function profiled_theta_probe(x_levels::AbstractVector)
    α_fallback = fill(NaN, N_TAU)
    try
        ΩL = x_levels[1]
        Ωs = x_levels[2:(1 + S)];                      Ωs = Ωs ./ sum(Ωs)
        A  = x_levels[(S + 2):(S + R_downstream + 1)]; A  = A ./ A[1]
        α  = collect(Float64, x_levels[(S + R_downstream + 2):(1 + S + R_downstream + N_TAU)])
        α_fallback = α
        res = invert_T_ge_warm(α, ΩL, Ωs, A)
        return (converged = res.converged, alpha = α, iters = res.iters, resid = res.resid)
    catch
        return (converged = false, alpha = α_fallback, iters = -1, resid = Inf)
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
