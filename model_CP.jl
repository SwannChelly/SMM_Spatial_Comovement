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
using Printf
using QuasiMonteCarlo
using DataFrames
using Optim
using CSV
using FixedEffectModels, RDatasets, CategoricalArrays

##################### Stratified Draws (CdGM-style) ###################

"""
Van der Corput sequence (base 2) — deterministic quasi-random in [0,1).
"""
function vdc(n::Int)::Float64
    r, den = 0.0, 1.0
    while n > 0
        den *= 2
        r += (n % 2) / den
        n = div(n, 2)
    end
    return r
end

"""
    generate_is_draws(N_rho, n_good; randomise=false, rng=Random.GLOBAL_RNG,
                      a=0.5, verbose=false)
        -> (U, W)

Per-column importance-sampling draws on (0, 1) for the Fréchet inverse-CDF
transform in `solve_network`. Returns:
  - `U::Matrix{Float64}` of shape (N_rho, n_good) — quantiles in (0, 1)
  - `W::Matrix{Float64}` of shape (N_rho, n_good) — IS weights, normalised
    PER COLUMN (`sum(W[:, g]) = 1` for every good pair g)

NOTE (defect): the IS tilt is BIASED for the min-coupled consumer moments
(`c_tilde`, `gamma_ls`, `industry`, `pi_r`). `solve_network`'s CES price index
applies only the WINNING column's weight `W[rho, winner_good_idx[rho,s]]` to each
row, dropping the density ratios of the *losing* columns; with a non-flat tilt
those ratios do not cancel, so the realised per-row weight is not the correct
importance weight for the joint (min over regions) functional. The "bounded
weights ⇒ no degeneracy" property only makes this estimator UNBIASED for
single-column functionals (the `reg_coef` quadrature rows), where there is no
min-coupling. For the default production path use `:qmc` (flat weights → the
winner-weight shortcut is exact). `:is` is retained only for the `reg_coef`
tail-resolution use case; it is NOT the default. See `generate_qmc_draws`.

This replaces the previous comonotonic stratified-QMC design (which returned a
length-N_rho weight VECTOR shared across columns). The shared bin grid made
every region's productivity a near-deterministic function of one latent per
row, collapsing Ricardian selection — the winner of `min_r c_r` became fixed by
T/tau/w rather than by independent Fréchet draws. Here each good pair g gets an
INDEPENDENT proposal stream, so `u[rho, g1]` and `u[rho, g2]` are independent.

# Importance sampling in uniform space

The downstream transform (unchanged) is `z = scale·(-log(1-u))^(-1/theta)`, so
the selection-relevant large-z mass sits at `u → 0`. The proposal must therefore
oversample `u` near 0. Proposal `q(u) = a·u^(a-1)`, a Beta(a, 1) with a ∈ (0, 1):
mass piles up at 0, inverse-CDF is `u = v^(1/a)`. The target is Uniform (p ≡ 1),
so the IS weight is `w ∝ 1/q(u) = u^(1-a)/a` — BOUNDED on [0, 1] (→ 0 at u → 0,
→ 1/a at u → 1): no weight degeneracy (unlike a Fréchet-scale tilt, whose weights
blow up in the body).

The proposal uniform `v` is stratified (one point per equal-prob stratum) with an
INDEPENDENT permutation per column → columns decorrelated AND low-variance.

# Two modes

`randomise=false` (default — used during optimisation):
    `MersenneTwister(g)` + within-stratum midpoint (0.5) → deterministic
    (PSO-safe; freezes Monte-Carlo noise so the SMM criterion is a
    deterministic function of theta).

`randomise=true` (used for Σ_sim estimation):
    permutation + jitter drawn from the supplied `rng` → independent per
    replication. Pass `MersenneTwister(k)` for replication k.

# Arguments

- `N_rho::Int`            : draws per good pair (production: 1000).
- `n_good::Int`           : number of active (sector, region) pairs.
- `randomise::Bool=false` : false → MersenneTwister(g) + midpoint (base);
                            true  → rng permutation + jitter (Σ_sim).
- `rng::AbstractRNG`      : source of randomness when `randomise=true`.
- `a::Float64=0.5`        : IS tilt, a ∈ (0, 1); smaller a ⇒ heavier tail
                            oversampling, lower ESS.
- `verbose::Bool=false`   : print `min_g ESS_g` (effective sample size of the
                            weakest column) as a degeneracy health check.
"""
function generate_is_draws(N_rho::Int, n_good::Int;
                           randomise::Bool=false,
                           rng::AbstractRNG=Random.GLOBAL_RNG,
                           a::Float64=0.5,
                           verbose::Bool=false)
    @assert 0.0 < a < 1.0 "IS tilt a ∈ (0,1); smaller a ⇒ heavier tail oversampling, lower ESS."

    U     = Matrix{Float64}(undef, N_rho, n_good)
    W     = Matrix{Float64}(undef, N_rho, n_good)
    inv_N = 1.0 / N_rho
    lo, hi = eps(), 1.0 - eps()

    @inbounds for g in 1:n_good
        col_rng = randomise ? rng : MersenneTwister(g)
        perm    = randperm(col_rng, N_rho)
        wsum    = 0.0
        for rho in 1:N_rho
            xi = randomise ? rand(rng) : 0.5         # within-stratum position
            v  = (perm[rho] - xi) * inv_N            # stratified proposal uniform
            u  = clamp(v^(1.0 / a), lo, hi)          # power-law tilt toward u→0
            w  = u^(1.0 - a) / a                     # ∝ 1/q(u), bounded
            U[rho, g] = u
            W[rho, g] = w
            wsum     += w
        end
        @views W[:, g] ./= wsum                      # per-column SNIS: Σ_rho W = 1
    end

    if verbose
        # ESS_g = 1 / Σ_rho W[rho,g]^2 (columns sum to 1) → count in [1, N_rho].
        min_ess = Inf
        @inbounds for g in 1:n_good
            s2 = 0.0
            for rho in 1:N_rho
                s2 += W[rho, g]^2
            end
            min_ess = min(min_ess, 1.0 / s2)
        end
        @printf("  generate_is_draws: min_g ESS_g = %.1f / %d  (frac %.3f, a=%.2f)\n",
                min_ess, N_rho, min_ess / N_rho, a)
    end

    return U, W
end


"""
    generate_qmc_draws(N_rho, n_good; randomise=false, rng=Random.GLOBAL_RNG)
        -> (U, W)

**Default production sampler.** Per-column STRATIFIED uniform draws on (0, 1) for
the Fréchet inverse-CDF transform in `solve_network`, with FLAT weights. Returns:
  - `U::Matrix{Float64}` (N_rho × n_good) — stratified quantiles in (0, 1)
  - `W::Matrix{Float64}` (N_rho × n_good) — uniform weights `1/N_rho` (matrix, so
    `[rho, g]` consumers are untouched; columns sum to 1)

Each good pair `g` gets an INDEPENDENT permutation, so `u[rho, g1] ⊥ u[rho, g2]`
(decorrelated — required to keep Ricardian `min_r c_r` selection alive) while one
point per equal-probability stratum gives the variance reduction of LHS. Unlike
`generate_is_draws`, there is NO importance tilt: the weight is uniform, which
makes `solve_network`'s winner-weight shortcut (it applies only the winning
column's weight per row) EXACT for the min-coupled moments — the source of the IS
sampler's bias. Stratification weakly beats plain i.i.d. MC on every block.

Two modes mirror `generate_is_draws`:
  - `randomise=false` (default; optimisation): `MersenneTwister(g)` + midpoint
    (0.5) within each stratum → deterministic per column (PSO-safe).
  - `randomise=true` (Σ_sim estimation): permutation + jitter from the supplied
    `rng`. NOTE: in this mode ALL columns of one call share the single `rng`
    stream (they are drawn sequentially, not from independent `MersenneTwister(g)`
    streams as in the base mode). This is intentional for Σ_sim, which wants
    independent DESIGNS across replications k (pass `MersenneTwister(k)`); within a
    replication the columns remain decorrelated via distinct random permutations.
"""
function generate_qmc_draws(N_rho::Int, n_good::Int;
                            randomise::Bool=false,
                            rng::AbstractRNG=Random.GLOBAL_RNG)
    U     = Matrix{Float64}(undef, N_rho, n_good)
    inv_N = 1.0 / N_rho
    lo, hi = eps(), 1.0 - eps()

    @inbounds for g in 1:n_good
        col_rng = randomise ? rng : MersenneTwister(g)
        perm    = randperm(col_rng, N_rho)
        for rho in 1:N_rho
            xi = randomise ? rand(rng) : 0.5      # within-stratum position
            u  = clamp((perm[rho] - xi) * inv_N, lo, hi)   # stratified uniform, NO tilt
            U[rho, g] = u
        end
    end

    # Flat weight MATRIX (N_rho × n_good): the winner-weight shortcut in
    # solve_network is exact iff weights are flat, so this is unbiased for the
    # min-coupled moments.
    W = fill(inv_N, N_rho, n_good)
    return U, W
end


function generate_mc_draws(N_rho::Int, n_good::Int, rng::AbstractRNG)
    U = rand(rng, N_rho, n_good)
    # Uniform weight MATRIX (N_rho × n_good) so consumers can index [rho, g],
    # matching the per-column weight convention of the other samplers.
    w = fill(1.0 / N_rho, N_rho, n_good)
    return U, w
end


"""
    sobol_scrambled_net(N, d, rng) -> Matrix{Float64} (N × d)

A `d`-dimensional Sobol net of `N` points in (0, 1), randomised by a base-2
DIGITAL SHIFT drawn from the explicit `rng`. Returns an `N × d` matrix (points in
rows) so callers index `[point, dim]`.

Why a digital shift rather than QuasiMonteCarlo's `OwenScramble`: the goal is a
randomisation that (a) consumes an EXPLICIT `rng` so the design is reproducible
under a frozen seed (PSO determinism) and independent across Σ_sim replications,
and (b) NEVER calls `Random.seed!` globally. QuasiMonteCarlo's scramble draws
from the global RNG and cannot be seeded per call without a global reseed, so we
keep the deterministic Sobol net from QuasiMonteCarlo and apply our own
rng-consuming randomisation. A base-2 digital shift (XOR of the fixed-point bits
with a random per-dimension mask) PRESERVES the (t, m, s)-net equidistribution
and dissolves the Sobol origin (the all-zeros first point), so no coordinate
collapses to u = 0.
"""
function sobol_scrambled_net(N::Int, d::Int, rng::AbstractRNG)
    pts   = QuasiMonteCarlo.sample(N, zeros(d), ones(d), SobolSample())  # d × N in [0,1]
    U     = Matrix{Float64}(undef, N, d)
    nbits = 52
    scale = Float64(2)^nbits
    mask  = (UInt64(1) << nbits) - UInt64(1)
    @inbounds for j in 1:d
        shift = rand(rng, UInt64) & mask            # per-dim digital shift, from rng
        for i in 1:N
            m       = floor(UInt64, pts[j, i] * scale) & mask
            U[i, j] = (m ⊻ shift) / scale
        end
    end
    return U
end


"""
    generate_sobol_draws(N_rho, n_good; randomise=false, rng=Random.GLOBAL_RNG,
                         seed=42) -> (U, W)

Sobol-based draws on (0, 1) for the Fréchet inverse-CDF transform, with FLAT
weights (like `:qmc`). Each SECTOR's active (sector, region) columns form one
Sobol net of dimension `d = length(SECTOR_GOOD_INDICES[s])`, scrambled by a
per-sector digital shift consumed from the master rng. Building one net per
sector (rather than a single global `n_good`-dim net) keeps the dimension low
where the Ricardian `min_r c_r` interaction actually lives — within a sector,
across its regions — which is where the net's superior equidistribution pays off;
distinct sectors draw distinct shifts and are therefore decorrelated.

Returns `U::Matrix (N_rho × n_good)` and the flat weight `W = fill(1/N_rho, …)`.
The flat weight makes `solve_network`'s winner-weight shortcut EXACT, so this is
unbiased for the min-coupled moments (same property as `:qmc`).

  - `randomise=false` (optimisation): master = `MersenneTwister(seed)` (frozen) →
    deterministic per call (PSO-safe). NOTE: this is a Sobol net at a FROZEN
    scramble seed, not a midpoint rule.
  - `randomise=true` (Σ_sim): master = supplied `rng` → independent scrambles per
    replication (pass `MersenneTwister(k)` for replication k).
"""
function generate_sobol_draws(N_rho::Int, n_good::Int;
                              randomise::Bool=false,
                              rng::AbstractRNG=Random.GLOBAL_RNG,
                              seed::Int=42)
    U      = Matrix{Float64}(undef, N_rho, n_good)
    lo, hi = eps(), 1.0 - eps()
    master = randomise ? rng : MersenneTwister(seed)
    covered = 0
    @inbounds for s in 1:length(SECTOR_GOOD_INDICES)
        g_indices = SECTOR_GOOD_INDICES[s]
        d = length(g_indices)
        d == 0 && continue
        net = sobol_scrambled_net(N_rho, d, master)   # N_rho × d, scrambled from master
        for (k, g) in enumerate(g_indices)
            for rho in 1:N_rho
                U[rho, g] = clamp(net[rho, k], lo, hi)
            end
        end
        covered += d
    end
    @assert covered == n_good "generate_sobol_draws: covered $covered active goods != n_good $n_good"
    # Flat weight MATRIX (N_rho × n_good): winner-weight shortcut exact ⇒ unbiased.
    W = fill(1.0 / N_rho, N_rho, n_good)
    return U, W
end


"""
    generate_draws(N_rho, n_good, method::Symbol; randomise=false,
                   rng=Random.GLOBAL_RNG, a=0.5, verbose=false) -> (U, W::Matrix)

Unified draw dispatcher. `method ∈ (:qmc, :mc, :is, :sobol)`:
  - `:qmc` (default everywhere) — stratified uniform, flat weights, decorrelated
    columns. Unbiased for the min-coupled moments; weakly beats `:mc`.
  - `:mc`  — i.i.d. uniform, flat weights. `randomise=false` ⇒ deterministic
    (`MersenneTwister(0)`) for PSO; `randomise=true` ⇒ supplied `rng`.
  - `:is`  — per-column importance sampling (tilt `a`). BIASED for min-coupled
    moments; retain only for `reg_coef` tail resolution. `a`/`verbose` apply only
    here (ignored for the other methods).
  - `:sobol` — per-sector digitally-shifted Sobol net, flat weights. Same
    unbiasedness as `:qmc` (flat weights ⇒ winner-weight shortcut exact); aims for
    lower Σ_sim variance on thick (multi-region) sectors. `randomise=false` ⇒
    frozen scramble seed (PSO-safe); `randomise=true` ⇒ supplied `rng`.

W is always an `(N_rho × n_good)` matrix, so every `[rho, g]` consumer is
untouched regardless of method — the invariant that confines the method switch to
the generation sites.
"""
function generate_draws(N_rho::Int, n_good::Int, method::Symbol;
                        randomise::Bool=false,
                        rng::AbstractRNG=Random.GLOBAL_RNG,
                        a::Float64=0.5,
                        verbose::Bool=false)
    if method === :qmc
        return generate_qmc_draws(N_rho, n_good; randomise=randomise, rng=rng)
    elseif method === :mc
        mc_rng = randomise ? rng : MersenneTwister(0)
        return generate_mc_draws(N_rho, n_good, mc_rng)
    elseif method === :is
        return generate_is_draws(N_rho, n_good; randomise=randomise, rng=rng,
                                 a=a, verbose=verbose)
    elseif method === :sobol
        return generate_sobol_draws(N_rho, n_good; randomise=randomise, rng=rng,
                                    seed=42)
    else
        error("Unknown draw method :$method (choose :qmc, :mc, :is, or :sobol)")
    end
end


# Backward-compatible alias: existing call sites that have not been threaded with
# an explicit method keep compiling and resolve to the new DEFAULT (:qmc). The
# `a`/`verbose` kwargs are accepted for signature parity but ignored by :qmc.
generate_stratified_draws(N_rho::Int, n_good::Int; randomise::Bool=false,
                          rng::AbstractRNG=Random.GLOBAL_RNG,
                          a::Float64=0.5, verbose::Bool=false) =
    generate_draws(N_rho, n_good, :qmc; randomise=randomise, rng=rng,
                   a=a, verbose=verbose)


##################### Helper Functions ###################

"""
    unpack_params(params) -> (Ω^L, Ω^s, A, α, T)

Unpack parameter vector into model components (paper notation).

Parameter vector layout: [Ω^L(1) | Ω^s(S) | A(R_downstream) | α(N_TAU) | T(sum(T_MASK))]

The T block lives in the **T-COLUMN space** (`T_COL_DIM` wide): upstream ZE under
`CA_LEVEL == :ze`, attraction areas under `:aa`. It is scattered s-major into an
`(S, T_COL_DIM)` matrix, per-sector reference-normalised (`T[s, ref] = 1`), and then
**gathered** to the (S, R) ZE-level matrix the model reads:

    T[s, l] = T_par[s, T_GATHER[l]]

`T_GATHER` is the identity under `:ze` (so this is byte-identical to the historical
scatter), and `AA_OF_ZE` under `:aa` — Assumption 2 of `finite_sample2.tex`: every ZE
of an area shares the Fréchet SCALE, not the realisation of its champions, so each ZE
still runs its own Ricardian competition and two ZE at different distances still have
different win probabilities.

Returns:
- Ω^L (Omega_L): Labor share in production [scalar]
- Ω^s (Omega_s): Sectoral input shares [S elements, normalized to sum to 1]
- A: Downstream firm productivity by region [R_downstream elements]
- α (alpha): Trade cost parameters [N_TAU elements; 1 = power-law α, >1 = bin coefficients]
- T: Fréchet scale parameters [S × R elements (ZE level), flattened s-fastest]
"""
function unpack_params(params)
    Omega_L = params[1]
    Omega_s = params[2:(1 + S)] / sum(params[2:(1 + S)])
    A = params[(S + 2):(S + R_downstream + 1)]
    A = A ./ A[1]
    alpha = params[(S + R_downstream + 2):(S + R_downstream + 1 + N_TAU)]

    T_reduced = params[(S + R_downstream + 2 + N_TAU):end]
    # eltype(params) (not a hard-coded Float64) so ForwardDiff Duals survive the
    # scatter into T_full — byte-identical for Float64 params.
    T_full = zeros(eltype(params), T_COL_DIM * S)
    T_full[T_MASK] = T_reduced
    T_par = permutedims(reshape(T_full, T_COL_DIM, S))   # s-major flat → (S, T_COL_DIM)

    for s in 1:S
        ref_c = T_REF_REGION[s]
        if ref_c > 0 && T_par[s, ref_c] > 0
            T_par[s, :] ./= T_par[s, ref_c]
        end
    end

    # Gather T columns onto ZE. Identity (and a plain copy) under :ze.
    if CA_LEVEL === :aa
        T_mat = Matrix{eltype(T_par)}(undef, S, R)
        @inbounds for l in 1:R
            c = T_GATHER[l]
            for s in 1:S
                T_mat[s, l] = T_par[s, c]
            end
        end
    else
        T_mat = T_par
    end

    return Omega_L, Omega_s, A, alpha, vec(T_mat)
end


"""
    unpack_T_par(params) -> Matrix (S × T_COL_DIM)

The comparative-advantage matrix in the **T-COLUMN space** (before the gather onto
ZE): reference-normalised per sector, zeros outside `T_MASK`. `unpack_params` returns
the gathered ZE-level version; this is what the Sinkhorn inversion and the T
delta-method iterate on, since those live in parameter space. Identical to
`reshape(unpack_params(params)[5], S, R)` under `CA_LEVEL == :ze`.
"""
function unpack_T_par(params)
    T_reduced = params[(S + R_downstream + 2 + N_TAU):end]
    T_full = zeros(eltype(params), T_COL_DIM * S)
    T_full[T_MASK] = T_reduced
    T_par = permutedims(reshape(T_full, T_COL_DIM, S))
    for s in 1:S
        ref_c = T_REF_REGION[s]
        if ref_c > 0 && T_par[s, ref_c] > 0
            T_par[s, :] ./= T_par[s, ref_c]
        end
    end
    return T_par
end


"""
    gather_T_to_ze(T_par) -> Matrix (S × R)

Broadcast the T-column parameters onto the upstream ZE the model actually simulates:
`T[s, l] = T_par[s, T_GATHER[l]]`. The identity under `:ze`.
"""
function gather_T_to_ze(T_par::AbstractMatrix)
    CA_LEVEL === :ze && return T_par
    T_mat = Matrix{eltype(T_par)}(undef, S, R)
    @inbounds for l in 1:R
        c = T_GATHER[l]
        for s in 1:S
            T_mat[s, l] = T_par[s, c]
        end
    end
    return T_mat
end


"""
    aggregate_gamma_to_T(gamma_ze) -> Matrix (T_COL_DIM × S)

Aggregate ZE-level sourcing shares to the T-column space — under `:aa` the
attraction-area aggregate `γ_{s,a} = Σ_{l ∈ a} γ_{ls}` summed over EVERY cell of the
area (control cells included, which is what makes the match to `EMP_GAMMA_T`
unbiased). The identity under `:ze`.
"""
function aggregate_gamma_to_T(gamma_ze::AbstractMatrix)
    CA_LEVEL === :ze && return gamma_ze
    out = zeros(eltype(gamma_ze), T_COL_DIM, S)
    @inbounds for g in 1:n_good
        l = GOOD_R[g]; s = GOOD_S[g]
        out[T_GATHER[l], s] += gamma_ze[l, s]
    end
    return out
end


"""
    concentrate_N_s(q_hat) -> (N_hat::Vector{Int}, clamped::Vector{Symbol})

Profile the variety count `N_s` out of the loss by a **monotone integer bisection**
on the count moment (plan D5; `finite_sample2.tex` §4.2).

`q_hat[g]` is the probability that cell `g` wins a given variety SOMEWHERE in the
downstream industry, estimated as the column mean of `linkages_flat` over the whole
draw pool. By Lemma 2 (`q ⊥ N_s`) it does not depend on the variety count, so

    Ḡ_s(n) = mean over cells l of sector s of (1 − q̂_ls)^n

is closed form and strictly decreasing in `n`; matching it to `G_TARGET[s]` needs no
re-simulation. The search is over the integers of `[N_LO[s], N_HI[s]]`, the bounds
implied by the observed distinct-supplier count, so `N̂_s` is an integer at every
evaluation — never relaxed, never rounded — and the outer optimiser never sees it.

`clamped[s] ∈ (:none, :lo, :hi)` records a bound that bound. Clamping is
INFORMATIVE, not benign: `:hi` means the model cannot generate enough sparsity even
when every variety is sourced from a single origin — a rejection signal for the
mechanism, which is why it is reported per sector rather than silently absorbed.
"""
function concentrate_N_s(q_hat::AbstractVector{<:Real})
    N_hat   = zeros(Int, S)
    clamped = fill(:none, S)
    @inbounds for s in 1:S
        cells = CELLS_OF_SECTOR[s]
        lo0, hi0 = N_LO[s], N_HI[s]
        if isempty(cells)
            N_hat[s] = lo0
            continue
        end
        Gs = n -> begin
            acc = 0.0
            for g in cells
                acc += (1.0 - q_hat[g])^n
            end
            acc / length(cells)
        end
        tgt = G_TARGET[s]
        if Gs(lo0) <= tgt
            N_hat[s], clamped[s] = lo0, :lo
        elseif Gs(hi0) >= tgt
            N_hat[s], clamped[s] = hi0, :hi
        else
            lo, hi = lo0, hi0
            while hi - lo > 1
                mid = (lo + hi) ÷ 2
                Gs(mid) > tgt ? (lo = mid) : (hi = mid)
            end
            N_hat[s] = abs(Gs(lo) - tgt) <= abs(Gs(hi) - tgt) ? lo : hi
        end
        @assert N_hat[s] <= N_rho "N̂[$s] = $(N_hat[s]) exceeds the per-replication variety " *
            "block width N_rho = $N_rho — it must be ≥ maximum(N_HI); restart."
    end
    return N_hat, clamped
end



##################### log-T = (φ) reparameterization #####################
# The optimizer searches φ (free, log-space, per-sector reference dropped); the
# model and every disk artifact stay in raw T *levels*. These two helpers are the
# only bridge — confined here so the choice of search space never leaks into
# solve_network / compute_moments / inference / reporting (the reference entries
# reconstruct to T=1, so unpack_params' `./= T_mat[s,ref]` stays a no-op).

"""
    t_levels_to_free_phi(T_red) -> φ   (length N_T_FREE)

Map a full reduced-T LEVEL block (length `N_T_REDUCED`, sector-major, `T_MASK`
order) to the free search vector `φ_k = log(T_i / T_{s(i),ref})`. Reference
entries are dropped. Inverse of [`t_free_phi_to_levels`](@ref).
"""
function t_levels_to_free_phi(T_red::AbstractVector)
    φ = Vector{Float64}(undef, N_T_FREE)
    @inbounds for (k, i) in enumerate(T_FREE_REDUCED_IDX)
        ref_i = SECTOR_REF_REDUCED[T_REDUCED_S[i]]
        φ[k] = log(T_red[i]) - log(T_red[ref_i])
    end
    return φ
end

"""
    t_free_phi_to_levels(φ) -> T_red   (length N_T_REDUCED)

Inverse of [`t_levels_to_free_phi`](@ref): scatter `exp.(φ)` into the free
reduced-T positions; reference entries are set to 1. `eltype(φ)` is preserved so
ForwardDiff Duals survive.
"""
function t_free_phi_to_levels(φ::AbstractVector)
    T_red = ones(eltype(φ), N_T_REDUCED)
    @inbounds for (k, i) in enumerate(T_FREE_REDUCED_IDX)
        T_red[i] = exp(φ[k])
    end
    return T_red
end

"""
    full_to_search(p) / search_to_full(x)

Convert a full LEVEL parameter vector `[Ω^L | Ω^s | A | α | T(N_T_REDUCED)]` to/from
the full SEARCH vector `[Ω^L | Ω^s | A | α | φ(N_T_FREE)]`. Only the trailing T
block is transformed; the head is untouched. Search vector is shorter by the
number of active sectors.
"""
function full_to_search(p::AbstractVector)
    nhead = 1 + S + R_downstream + N_TAU
    return vcat(p[1:nhead], t_levels_to_free_phi(p[(nhead + 1):end]))
end
function search_to_full(x::AbstractVector)
    nhead = 1 + S + R_downstream + N_TAU
    return vcat(x[1:nhead], t_free_phi_to_levels(x[(nhead + 1):end]))
end


"""
    build_tau(alpha) -> τ[r', r]

Build iceberg trade cost matrix from distance bin coefficients.

τ_{r'r} = 1 + α_b  where b = DistBin[r', r]

Returns matrix of size (R, R). Trade costs are identical across sectors.
"""
function build_tau(alpha)
    # eltype(alpha) keeps ForwardDiff Duals through the τ construction; identical
    # to ones(Float64, …) for the Float64 production path.
    tau = ones(eltype(alpha), R, R_downstream)
    if N_TAU == 1
        # Power-law: τ_{r',r} = max(d, 1)^α = exp(α · log(max(d, 1)))
        for r_prime in 1:R, r_d in 1:R_downstream
            tau[r_prime, r_d] = exp(alpha[1] * LOG_DIST_DOWNSTREAM[r_prime, r_d])
        end
    else
        for r_prime in 1:R, r_d in 1:R_downstream
            b = DistBin[r_prime, r_d]
            if b > 0 && b <= N_TAU
                tau[r_prime, r_d] += alpha[b]
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
- `params`: Parameter vector [α, Ω^L, Ω^s, A, T]
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
                       u_draws::Union{Nothing, Matrix{Float64}}=nothing,
                       sample_weights::Union{Nothing, Matrix{Float64}}=nothing)

    # ─────────────────────────────────────────────────────────────────────────
    # Unpack parameters (paper notation)
    # ─────────────────────────────────────────────────────────────────────────
    Omega_L, Omega_s_vec, A_vec, alpha, T_vec = unpack_params(params)

    # Build trade cost matrix τ_{r'r} — identical across sectors
    tau = precomputed_tau === nothing ? build_tau(alpha) : precomputed_tau

    # A_vec is already R_downstream length
    A_r = A_vec

    # Reshape for broadcasting
    Omega_s = reshape(Omega_s_vec, 1, S)  # (1, S)
    nu_s_mat = reshape(nu_s, 1, S)        # (1, S)
    T = reshape(T_vec, S, R)              # (S, R)

    # ─────────────────────────────────────────────────────────────────────────
    # Draw upstream firm productivities — flat (N_rho_eff, n_good) layout
    # Only good (s,r) pairs (where T_MASK is true) are computed.
    #
    # Row count is taken from the PASSED draws/weights (`N_rho_eff`), not the
    # global `N_rho` const. The two coincide in production (all draws are
    # generated at `N_rho`), but a caller may pass a different number of draws
    # (e.g. the price-alignment test sweeps N). Using the global const while the
    # weights are normalised over the passed N would consume only the first
    # `N_rho` rows and break the per-column Σw=1 invariant, inflating the CES
    # price index by (N/N_rho)^2.
    # ─────────────────────────────────────────────────────────────────────────
    N_rho_eff = u_draws !== nothing ? size(u_draws, 1) :
                sample_weights !== nothing ? size(sample_weights, 1) : N_rho
    if u_draws !== nothing && sample_weights !== nothing
        @assert size(sample_weights, 1) == N_rho_eff "sample_weights rows ($(size(sample_weights,1))) must match u_draws rows ($N_rho_eff)"
    end

    if u_draws === nothing
        # Backward compatibility: random Fréchet draws with uniform weights
        Random.seed!(50)
        z_flat = zeros(N_rho_eff, n_good)
        for g in 1:n_good
            T_sr = T[GOOD_S[g], GOOD_R[g]]
            if T_sr > 0
                d = Frechet(theta, T_sr^(1/theta))
                z_flat[:, g] = rand(d, N_rho_eff)
            end
        end
        if sample_weights === nothing
            # Uniform weight matrix so downstream [rho, g] indexing is consistent.
            sample_weights = fill(1.0/N_rho_eff, N_rho_eff, n_good)
        end
    else
        # CdGM-style: Fréchet inverse CDF from stratified uniform draws
        # u_draws is a Matrix{Float64} (N_rho_eff × n_good) with per-pair quantiles
        # F⁻¹(u) = σ · (-ln(1-u))^(-1/θ) where σ = T_{sr}^{1/θ}
        z_flat = zeros(N_rho_eff, n_good)
        for g in 1:n_good
            T_sr = T[GOOD_S[g], GOOD_R[g]]
            scale = max(T_sr, eps(Float64))^(1/theta)
            for rho in 1:N_rho_eff
                z_flat[rho, g] = scale * (-log(1.0 - u_draws[rho, g]))^(-1.0/theta)
            end
        end
    end
    z_inv_flat = z_flat .^ (-1)

    # NOTE: weights are now per-(rho, good-pair) — `sample_weights[rho, g]`.
    # The CES price index weights each (rho, s) variety by the weight of the
    # good pair that WON that variety (built per downstream region below from
    # `winner_good_idx`); there is no single shared length-N_rho weight vector.

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
    linkages_flat = zeros(N_rho_eff, n_good) # Firm-level supplier indicator

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
        p_rho_s = zeros(N_rho_eff, S)
        winner_good_idx = zeros(Int, N_rho_eff, S)

        for s in 1:S
            g_indices = SECTOR_GOOD_INDICES[s]
            if isempty(g_indices); continue; end
            regions_s = SECTOR_GOOD_REGIONS[s]

            # Prices only for active upstream (s,r') pairs
            tau_sr = reshape(tau[regions_s, r_d], 1, :)      # (1, n_active_in_s); tau is R × R_downstream
            w_sr = reshape(W_RS_FLAT[g_indices], 1, :)       # (1, n_active_in_s)
            prices_s = z_inv_flat[:, g_indices] .* tau_sr .* w_sr  # (N_rho_eff, n_active_in_s)

            # Ricardian selection: lowest-cost supplier wins each variety
            min_local = argmin(prices_s, dims=2)  # (N_rho_eff, 1)
            for rho in 1:N_rho_eff
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
        # Per-(rho, s) weight = weight of the good pair that won this variety.
        w_rho_s = Matrix{Float64}(undef, N_rho_eff, S)
        for s in 1:S
            for rho in 1:N_rho_eff
                g_w = winner_good_idx[rho, s]
                w_rho_s[rho, s] = g_w == 0 ? 0.0 : sample_weights[rho, g_w]
            end
        end
        P_sr = sum(w_rho_s .* p_rho_s.^(1 .- nu_s_mat), dims=1).^(1 ./ (1 .- nu_s_mat))
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
            for rho in 1:N_rho_eff
                g_winner = winner_good_idx[rho, s]
                if g_winner == 0; continue; end

                # Mark linkage
                linkages_flat[rho, g_winner] = 1.0

                # Expenditure share for this variety (weight = winning pair's column)
                exp_val = sample_weights[rho, g_winner] * Omega_s_vec[s] * (1-Omega_L) *
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
    mu = epsilon/(epsilon - 1)    # Inverse markup μ = ε/(ε-1)

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
function fast_weighted_regression(linkages_flat, z_flat, sample_weights::Matrix{Float64};
                                  include_control::Bool=true,
                                  include_size_control::Bool=!include_control,
                                  rho_range::Union{Nothing, Vector{UnitRange{Int}}}=nothing,
                                  obs_weight::Union{Nothing, Vector{Float64}}=nothing,
                                  return_size_coef::Bool=false)

    # Regressors: N_REG distance-bin dummies, plus (optionally) a log-z size control.
    # The size control and the control group are MUTUALLY EXCLUSIVE by construction:
    # the control-only (filter==2) firms have no productivity (z ≡ −∞), so they cannot
    # carry a log-z regressor. Hence the default couples them — size control ON only
    # when the control group is OFF (include_size_control = !include_control):
    #   • include_control=true  (production): control y=0 rows appended, NO log-z column.
    #   • include_control=false (diagnostic): supplier pairs only, log-z size control ADDED
    #     — conditioning on productivity purges the T-through-z omitted-variable confound,
    #       so the distance slope loads on the trade-cost/α channel (see identification note).
    # z_flat is always built by solve_network for the Ricardian selection; it is regressed
    # on only when include_size_control is true.
    @assert CA_LEVEL === :aa || !(include_control && include_size_control) (
        "size control needs firm productivity; the control group has z ≡ −∞ under " *
        ":ze, so include_control and include_size_control cannot both be true")
    n_size       = include_size_control ? 1 : 0
    n_regressors = N_REG + n_size          # distance bins (+ log-z size control)
    size_col     = N_REG + 1               # column index of the log-z control (if present)

    # Row count from the passed draws (not the global N_rho const), so callers
    # may pass a different number of varieties (e.g. the price-alignment test).
    N_rho_eff = size(sample_weights, 1)

    # Per-good variety rows. `rho_range === nothing` ⇒ every drawn variety (the
    # continuum/legacy design). Under granularity the caller passes the PREFIX
    # varieties of a realised economy, with a flat `obs_weight` so each cell carries
    # total weight 1. Unused on the production path, kept as a hook.
    rows_of = g -> rho_range === nothing ? (1:N_rho_eff) : rho_range[g]
    n_rows_goods = rho_range === nothing ? n_good * N_rho_eff : sum(length, rho_range)

    # Rows: the good rows (+ N_CONTROL control-only pairs × N_rho_eff when
    # include_control). The `include_control=false` path is used by the diagnostic that
    # contrasts the regression with vs without the no-supplier control group, and is
    # the production path under :aa (control cells are ordinary goods there).
    n_ctrl_eff = include_control ? N_CONTROL : 0
    N_valid = n_rows_goods + n_ctrl_eff * N_rho_eff

    y = Vector{Float64}(undef, N_valid)
    X = zeros(N_valid, n_regressors)
    w = Vector{Float64}(undef, N_valid)
    fe_group = Vector{Int}(undef, N_valid)

    idx = 0
    for g in 1:n_good
        s = GOOD_S[g]
        r = GOOD_R[g]
        dr = CLOSEST_DOWNSTREAM_REGION[r]
        group_id = (s - 1) * R_downstream + dr
        if N_REG == 1
            log_dist = LOG_CLOSEST_DIST[r]
        else
            b = DistBin[r, dr]
        end

        for rho in rows_of(g)
            idx += 1
            y[idx] = linkages_flat[rho, g] > 0 ? 1.0 : 0.0
            w[idx] = obs_weight === nothing ? sample_weights[rho, g] : obs_weight[g]
            fe_group[idx] = group_id

            if N_REG == 1
                X[idx, 1] = log_dist
            else
                if b > 0 && b <= N_REG
                    X[idx, b] = 1.0
                end
            end

            if include_size_control
                X[idx, size_col] = log(z_flat[rho, g])   # size/productivity control
            end
        end
    end

    # ── Control-only (filter==2) pairs: N_rho_eff rows of y=0 per pair ───────────
    # Exogenous zeros of the extensive margin: sector-regions with only control-group
    # firms. They contribute additional distance observations (no supplier, y=0) at
    # their distance-to-nearest-downstream bin, sharing the (sector × nearest-down)
    # fixed effect. Flat weight 1/N_rho_eff ⇒ each control pair carries total weight 1,
    # matching a supplier pair (whose sample_weights column sums to 1). No log-z term.
    if include_control
        w_ctrl = 1.0 / N_rho_eff
        for c in 1:N_CONTROL
            s = CONTROL_S[c]
            r = CONTROL_R[c]
            dr = CLOSEST_DOWNSTREAM_REGION[r]
            group_id = (s - 1) * R_downstream + dr
            if N_REG == 1
                log_dist = LOG_CLOSEST_DIST[r]
            else
                b = DistBin[r, dr]
            end

            for rho in 1:N_rho_eff
                idx += 1
                y[idx] = 0.0
                w[idx] = w_ctrl
                fe_group[idx] = group_id

                if N_REG == 1
                    X[idx, 1] = log_dist
                else
                    if b > 0 && b <= N_REG
                        X[idx, b] = 1.0
                    end
                end
            end
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

    # The log-z coefficient is a free over-identifying diagnostic (it should equal
    # −θ under the cloglog link; see fast_cloglog_regression). Returned only on
    # request, so the historical N_REG-long return is untouched.
    return return_size_coef ?
        vcat(coefs[1:N_REG], include_size_control ? coefs[size_col] : NaN) :
        coefs[1:N_REG]
end


##################### Fast cloglog (IRLS over the FWL kernel) ###################

"""
    _cloglog_irls(y, X, w, fe_group; max_iter=50, tol=1e-9, eta_clamp=30.0) -> β

Fit a complementary-log-log GLM  `P(y=1) = 1 − exp(−exp(η))`, `η = Xβ + a_{fe}`,
by IRLS. The single categorical fixed effect `fe_group` is absorbed via **weighted
within-group demeaning** each iteration (Frisch–Waugh–Lovell — exact for ONE FE
dimension), so no dummy matrix is built. This is the nonlinear analogue of the
`fast_weighted_regression` kernel: each IRLS step is a weighted LS on the demeaned
design. `w` are analytic (observation) weights. Returns the coefficient vector `β`
(length `size(X,2)`); the fixed effects are profiled out.

The IRLS quantities for cloglog: with `μ = 1 − exp(−exp(η))`,
`dμ/dη = e^{η}(1−μ)`, working response `z = η + (y−μ)/(dμ/dη)`, and IRLS weight
`W = w·(dμ/dη)² / (μ(1−μ))`.
"""
function _cloglog_irls(y::AbstractVector{<:Real}, X::AbstractMatrix{<:Real},
                       w::AbstractVector{<:Real}, fe_group::AbstractVector{<:Integer};
                       max_iter::Int=50, tol::Float64=1e-9, eta_clamp::Float64=30.0)
    n, k = size(X)
    yf = Float64.(y); Xf = Float64.(X); wf = Float64.(w)

    # Precompute group → row indices once (weighted within-transform is one pass/FE).
    groups  = unique(fe_group)
    rows_of = Dict(g => findall(==(g), fe_group) for g in groups)

    # Init: μ shrunk toward 0.5 (avoids η = ±∞ at all-0 / all-1 starts), η = cloglog link.
    μ = (yf .+ 0.5) ./ 2
    η = clamp.(log.(.-log.(1 .- μ)), -eta_clamp, eta_clamp)

    β = zeros(k)
    for _ in 1:max_iter
        μ    = clamp.(1 .- exp.(.-exp.(η)), 1e-12, 1 - 1e-12)
        dμdη = exp.(η) .* (1 .- μ)                       # e^{η}·e^{−e^{η}}
        zwork = η .+ (yf .- μ) ./ dμdη                   # working response
        W    = wf .* (dμdη .^ 2) ./ (μ .* (1 .- μ))      # IRLS × analytic weights

        # Weighted within-group demeaning (FWL) of zwork and each X column.
        zt = copy(zwork); Xt = copy(Xf)
        for g in groups
            r  = rows_of[g]
            sw = sum(@view W[r])
            sw < 1e-300 && continue
            zt[r] .-= sum(W[r] .* zwork[r]) / sw
            @inbounds for j in 1:k
                Xt[r, j] .-= sum(W[r] .* Xf[r, j]) / sw
            end
        end

        # Weighted LS on the demeaned system: (Xt' W Xt) β = Xt' W zt.
        WXt   = W .* Xt
        β_new = (Xt' * WXt) \ (Xt' * (W .* zt))

        # Rebuild η = Xβ + a_fe, with a_g = weighted group mean of (zwork − Xβ).
        resid = zwork .- Xf * β_new
        η_new = Xf * β_new
        for g in groups
            r  = rows_of[g]
            sw = sum(@view W[r])
            sw < 1e-300 && continue
            η_new[r] .+= sum(W[r] .* resid[r]) / sw
        end
        η_new = clamp.(η_new, -eta_clamp, eta_clamp)

        Δ = maximum(abs.(β_new .- β))
        β = β_new; η = η_new
        Δ < tol && break
    end
    return β
end

"""
    fast_cloglog_regression(linkages_flat, z_flat, sample_weights; kwargs...) -> Vector (N_REG)

Complementary-log-log extensive-margin regression, fit by IRLS over the weighted-FWL
kernel (`_cloglog_irls`). Drop-in sibling of `fast_weighted_regression` (same design:
distance-bin regressors + optional log-z size control + optional control-group rows,
same `(sector × nearest-downstream)` FE), but the correct nonlinear link instead of a
linear probability model.

The outcome is `not_supply = 1 − supplier`, so with `P(not_supply)=1−exp(−exp(η))` the
distance coefficient equals **αθ** (the empirical `reg_coef_cloglog` convention — note the
SIGN/OUTCOME differ from `fast_weighted_regression`, which returns the LPM slope of
`P(supplier)`). Under this specification the coefficients are, in the single-destination
limit, `β_distance = θα` and `β_logz = −θ`, so conditioning on size (`include_size_control`)
purges the T-through-productivity confound and loads the distance slope on α.
Returns the `N_REG` distance coefficients.
"""
function fast_cloglog_regression(linkages_flat, z_flat, sample_weights::Matrix{Float64};
                                 include_control::Bool=true,
                                 include_size_control::Bool=!include_control,
                                 rho_range::Union{Nothing, Vector{UnitRange{Int}}}=nothing,
                                 obs_weight::Union{Nothing, Vector{Float64}}=nothing,
                                 return_size_coef::Bool=false,
                                 max_iter::Int=50, tol::Float64=1e-9)
    @assert CA_LEVEL === :aa || !(include_control && include_size_control) (
        "size control needs firm productivity; control firms have z ≡ −∞ under :ze")
    n_size       = include_size_control ? 1 : 0
    n_regressors = N_REG + n_size
    size_col     = N_REG + 1
    N_rho_eff    = size(sample_weights, 1)
    rows_of      = g -> rho_range === nothing ? (1:N_rho_eff) : rho_range[g]
    n_rows_goods = rho_range === nothing ? n_good * N_rho_eff : sum(length, rho_range)
    n_ctrl_eff   = include_control ? N_CONTROL : 0
    N_valid      = n_rows_goods + n_ctrl_eff * N_rho_eff

    y        = Vector{Float64}(undef, N_valid)
    X        = zeros(N_valid, n_regressors)
    w        = Vector{Float64}(undef, N_valid)
    fe_group = Vector{Int}(undef, N_valid)

    idx = 0
    for g in 1:n_good
        s = GOOD_S[g]; r = GOOD_R[g]
        dr = CLOSEST_DOWNSTREAM_REGION[r]
        group_id = (s - 1) * R_downstream + dr
        local b, log_dist
        if N_REG == 1
            log_dist = LOG_CLOSEST_DIST[r]
        else
            b = DistBin[r, dr]
        end
        for rho in rows_of(g)
            idx += 1
            y[idx] = linkages_flat[rho, g] > 0 ? 0.0 : 1.0   # not_supply
            w[idx] = obs_weight === nothing ? sample_weights[rho, g] : obs_weight[g]
            fe_group[idx] = group_id
            if N_REG == 1
                X[idx, 1] = log_dist
            else
                (b > 0 && b <= N_REG) && (X[idx, b] = 1.0)
            end
            if include_size_control
                X[idx, size_col] = log(z_flat[rho, g])
            end
        end
    end

    if include_control
        w_ctrl = 1.0 / N_rho_eff
        for c in 1:N_CONTROL
            s = CONTROL_S[c]; r = CONTROL_R[c]
            dr = CLOSEST_DOWNSTREAM_REGION[r]
            group_id = (s - 1) * R_downstream + dr
            local b, log_dist
            if N_REG == 1
                log_dist = LOG_CLOSEST_DIST[r]
            else
                b = DistBin[r, dr]
            end
            for rho in 1:N_rho_eff
                idx += 1
                y[idx] = 1.0                 # control firms never supply → not_supply=1
                w[idx] = w_ctrl
                fe_group[idx] = group_id
                if N_REG == 1
                    X[idx, 1] = log_dist
                else
                    (b > 0 && b <= N_REG) && (X[idx, b] = 1.0)
                end
            end
        end
    end

    β = _cloglog_irls(y, X, w, fe_group; max_iter=max_iter, tol=tol)
    # Under the firm-level reduced form (finite_sample2.tex Prop. 1) the log-z
    # coefficient equals −θ exactly: a free over-identifying test, returned on
    # request. The historical N_REG-long return is unchanged by default.
    return return_size_coef ?
        vcat(β[1:N_REG], include_size_control ? β[size_col] : NaN) :
        β[1:N_REG]
end


##################### Moment Computation ###################

"""
    compute_moments(network, params)

Compute targeted moments from solved network for SMM estimation.

# Moments (matching empirical_moments structure):
1. Aggregate labor share: Σ_r w_r·L_r / Σ_r C_r
2. Sectoral input shares: X_s / X
3. Regional employment shares π_r
4. Regression coefficients: Elasticity of supplier probability to distance
5. Sourcing shares γ_{ls}: Share of sector s inputs from region l
"""
function compute_moments(network, params; N_fixed::Union{Nothing, Vector{Int}}=nothing)

    Omega_L, Omega_s_vec, A_vec, alpha, T_vec = unpack_params(params)

    X_ls_flat = network.X_ls_flat
    c_tilde_r = network.c_tilde_r
    Y_r = network.Y_r
    linkages_flat = network.linkages_flat
    z_flat = network.z_flat
    closest_plant_dist = network.closest_plant_dist
    closest_downstream_region = network.closest_downstream_region
    active = N_downstream_per_region .!= 0
    
    # ─────────────────────────────────────────────────────────────────────────
    # We can use the output of the network to build it. No need for those additional calculs to build y_r. 
    # 1. Aggregate labor share (matching model_CP.jl exactly)
    # ─────────────────────────────────────────────────────────────────────────
    
    
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
    #
    # Under CA_LEVEL == :aa the block-5 moment is the ATTRACTION-AREA aggregate
    # γ_{s,a} = Σ_{l ∈ a} γ_{ls}, summed over EVERY cell of the area, control cells
    # included — the same rule the empirical target `EMP_GAMMA_T` uses. A control
    # cell contributes 0 in the data and a strictly positive γ in the model; that is
    # the correct unbiased match (E[γ̂_ls] = γ_ls holds unconditionally), not a
    # mismatch — see load_parameters.jl SECTION 3b.
    # ─────────────────────────────────────────────────────────────────────────
    gamma_ls = X_ls ./ X_s .* reshape(domestic_share, 1, S)
    gamma_out = gamma_ls
    if CA_LEVEL === :aa
        gamma_out = zeros(eltype(gamma_ls), T_COL_DIM, S)
        for g in 1:n_good
            l = GOOD_R[g]; s = GOOD_S[g]
            gamma_out[T_GATHER[l], s] += gamma_ls[l, s]
        end
    end

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Regression: extensive margin vs distance.
    #    REG_METHOD selects the link (:lpm linear-probability, or :cloglog — the correct
    #    complementary-log-log with distance coef = αθ). REG_INCLUDE_CONTROL /
    #    REG_INCLUDE_SIZE are the effective design flags (load_parameters.jl SECTION 6);
    #    the empirical target loaded there matches REG_METHOD.
    #
    #    IDENTICAL under GRANULAR: by Prop. 1 the firm-level index carries NO N_s term
    #    (conditioning on the firm removes it), so the regression is run ONCE on the
    #    ordinary draws in both modes. Block 4 is therefore the continuum-limit
    #    coefficient β(E[y]); the finite-sample bias of the auxiliary cloglog is NOT
    #    cancelled by a binding-function average — it is measured by validation gate
    #    V12 and reported alongside α̂ (granular_validation.md Part III).
    # ─────────────────────────────────────────────────────────────────────────
    sw = network.sample_weights
    reg_fun = REG_METHOD == :cloglog ? fast_cloglog_regression : fast_weighted_regression

    reg_all  = reg_fun(linkages_flat, z_flat, sw;
                       include_control      = REG_INCLUDE_CONTROL,
                       include_size_control = REG_INCLUDE_SIZE,
                       return_size_coef     = REG_INCLUDE_SIZE)
    reg_coef = reg_all[1:N_REG]
    # The log-z coefficient is a free over-identifying test (Prop. 1(c): it equals −θ).
    b_logz   = REG_INCLUDE_SIZE ? reg_all[N_REG + 1] : NaN

    # ─────────────────────────────────────────────────────────────────────────
    # 4b. GRANULAR: profile the variety count, then the count moment — both from q̂
    #     alone, in CLOSED FORM. No prefix of the draws is taken and no replicated
    #     economy is simulated (see load_parameters.jl SECTION 9b).
    #
    #       q̂_ls   = share of varieties cell l wins SOMEWHERE downstream (Lemma 2:
    #                free of N_s), i.e. a column mean of linkages_flat;
    #       N̂_s    = monotone integer bisection on Ḡ_s(n) = mean_l (1 − q̂_ls)^n;
    #       Ḡ_s(0) = that same closed form evaluated at N̂_s.
    #
    #     The closed form IS the expectation of the realised empty-cell share — by
    #     linearity over cells, the dependence between cells (they share variety
    #     draws) does not affect the mean — so it is unbiased AND noise-free.
    # ─────────────────────────────────────────────────────────────────────────
    N_hat      = Int[]
    N_hat_free = Int[]
    clamped    = Symbol[]
    q_hat      = Float64[]
    G0         = Float64[]

    if GRANULAR
        n_draw = size(linkages_flat, 1)
        q_hat  = Vector{Float64}(undef, n_good)
        @inbounds for g in 1:n_good
            acc = 0.0
            for rho in 1:n_draw
                acc += linkages_flat[rho, g]
            end
            q_hat[g] = clamp(acc / n_draw, 0.5 / n_draw, 1.0 - 1e-12)
        end
        N_hat_free, clamped = concentrate_N_s(q_hat)
        # Finite-difference / counterfactual mode: the caller pins N̂_s (it is a STEP
        # function of θ, so a central FD could otherwise straddle a jump). The free
        # bisection value is returned as a diagnostic (`N_hat_free`) rather than warned
        # about here — this runs on every worker for every FD evaluation, and the free
        # value differs from the pinned one for a SECOND, benign reason: an evaluation on
        # a different draw set has a different q̂ hence a different N̂. The caller
        # (`compute_jacobian`) is the only place that can tell the two apart, and it
        # reports the dispersion once.
        N_hat = N_fixed === nothing ? N_hat_free : copy(N_fixed)

        # Block 6: Ḡ_s(0) over the cells inside active attraction areas — empty cells
        # included, since those ARE the K = 0 mass (finite_sample2.tex §3.2).
        G0 = zeros(S)
        @inbounds for s in 1:S
            cells = CELLS_OF_SECTOR[s]
            isempty(cells) && continue
            acc = 0.0
            for g in cells
                acc += (1.0 - q_hat[g])^N_hat[s]
            end
            G0[s] = acc / length(cells)
        end
    end


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
        agg_labor_share   = [agg_labor_share],
        agg_industry_share = vec(agg_industry_share),  # Full S elements (mask handles [2:end])
        pi_r              = pi_r,                       # block 3 — Full R_downstream (mask handles [2:end])
        reg_coef          = reg_coef,                   # block 4
        gamma_ls          = gamma_out,                  # block 5 — ZE or AA level (mask handles inactive/ref)
        G0                = G0,                         # block 6 — empty (Float64[]) unless GRANULAR
        # ── Non-moment diagnostics (never enter the moment vector) ───────────
        N_hat             = N_hat,                      # profiled variety count per sector
        N_hat_free        = N_hat_free,                 # the bisection value BEFORE any pin
        clamped           = clamped,                    # :none / :lo / :hi per sector
        q_hat             = q_hat,                      # win-somewhere probability per cell
        b_logz            = b_logz,                     # log-z coefficient (should equal −θ)
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
             u_draws::Union{Nothing, Matrix{Float64}}=nothing,
             sample_weights::Union{Nothing, Matrix{Float64}}=nothing,
             N_fixed::Union{Nothing, Vector{Int}}=nothing)

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
    moments = compute_moments(network, params; N_fixed=N_fixed)

    return moment_blocks_tuple(moments)
end


"""
    granular_report(params; u_draws, sample_weights, N_fixed=nothing) -> NamedTuple

Non-moment granular diagnostics at a parameter vector: the profiled variety count
`N̂_s`, which sectors clamped at a bound, the win-somewhere probability `q̂` per cell,
the count moment `Ḡ_s(0)`, the log-z coefficient (should equal `−θ`), and the two
independent routes to `N_s` (the bisection on `Ḡ_s(0)` versus the closed-form
`N^count_s = N_supplier_s / Σ_l q̂_ls` of `finite_sample2.tex` §3.3 — a free
over-identifying check, gate V7).

Also returns the expected realised supplier counts `E[K_ls] = N̂_s · q̂_ls` and the
implied dispersion. Nothing here is simulated on a realised economy: every quantity
is a closed-form function of `q̂` and `N̂_s` (see load_parameters.jl SECTION 9b).

Recomputed on demand rather than smuggled out of the loss, so nothing about the
optimiser's return contract changes and no mutable state crosses the `pmap` workers.
"""
function granular_report(params; u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS,
                         N_fixed::Union{Nothing, Vector{Int}}=nothing)
    @assert GRANULAR "granular_report is only defined under GRANULAR=true"
    network = solve_network(params; u_draws=u_draws, sample_weights=sample_weights)
    m = compute_moments(network, params; N_fixed=N_fixed)

    # Second route to N_s (over-identifying check): N^count_s = N_supplier_s / Σ_l q̂_ls.
    N_count = [begin
        cells = CELLS_OF_SECTOR[s]
        sq = isempty(cells) ? 0.0 : sum(m.q_hat[g] for g in cells)
        sq > 0 ? N_HI[s] / sq : NaN
    end for s in 1:S]

    # Expected supplier count per cell, K_ls ~ Bin(N̂_s, q̂_ls) ⇒ E[K] = N̂_s q̂.
    EK = [m.N_hat[GOOD_S[g]] * m.q_hat[g] for g in 1:n_good]

    return (N_hat = m.N_hat, clamped = m.clamped, q_hat = m.q_hat, b_logz = m.b_logz,
            G0 = m.G0, G_target = collect(G_TARGET), N_count = N_count, EK = EK)
end


"""
    moment_blocks_tuple(m) -> Tuple

The moment blocks of a `compute_moments` / `compute_moments_analytical` result, in
moment-vector order and WITHOUT the non-moment diagnostic fields. Five blocks in
legacy mode; six under `GRANULAR` — block 6 (`G0`) is APPENDED, never inserted, so
every existing index into blocks 1–5 is untouched.
"""
function moment_blocks_tuple(m)
    return GRANULAR ?
        (m.agg_labor_share, m.agg_industry_share, m.pi_r, m.reg_coef, m.gamma_ls, m.G0) :
        (m.agg_labor_share, m.agg_industry_share, m.pi_r, m.reg_coef, m.gamma_ls)
end


"""
    moments_to_vec(sim) -> Vector

Stack a moment-block tuple (or NamedTuple) into the masked moment vector — the
single assembly used by the loss, the Jacobian and every inference path. Replaces
the hard-coded `vcat([vec(m[i]) for i in 1:5]...)[MOMENT_MASK]`, which would silently
drop block 6 under `GRANULAR`.
"""
function moments_to_vec(sim)
    blocks = sim isa Tuple ? sim : moment_blocks_tuple(sim)
    return vcat([vec(blocks[i]) for i in 1:length(blocks)]...)[MOMENT_MASK]
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
                          u_draws::Union{Nothing, Matrix{Float64}}=nothing,
                          sample_weights::Union{Nothing, Matrix{Float64}}=nothing)

    # Solve network with firm-level data
    network = solve_network(params, return_firm_level=true, precomputed_tau=precomputed_tau,
                            u_draws=u_draws, sample_weights=sample_weights)
    
    # Compute moments
    moments = compute_moments(network, params)
    
    return moments, network
end


##################### Loss Function ###################

"""
    loss_function(simulated_moments, emp, W; moment_indices=nothing)

Compute the GMM/SMM loss `err * W * err'` between empirical and simulated moments,
where `err = emp − sim` (raw difference) over the masked moment vector. When
`moment_indices` is given, both the residual and `W` are restricted to those moments
(a pre-restricted `W` of matching size is used as-is).
"""
function loss_function(simulated_moments, emp, W;
                       moment_indices::Union{Nothing, Vector{Int}} = nothing)

    sim_flat = moments_to_vec(simulated_moments)
    emp_flat = vec(emp)

    if moment_indices !== nothing
        sim_flat = sim_flat[moment_indices]
        emp_flat = emp_flat[moment_indices]
    end

    N   = length(sim_flat)
    err = reshape(emp_flat - sim_flat, (1, N))

    W = isnothing(W) ? I(N) : W
    if moment_indices !== nothing && !isa(W, UniformScaling)
        # If W is already restricted to the selected moments (e.g. W_step3 over β+γ,
        # size == length(moment_indices)), use it as-is — err is subset to the same
        # moments in the same (β-then-γ) order. Otherwise subset a full-size W.
        if size(W, 1) != length(moment_indices)
            W = W[moment_indices, moment_indices]
        end
    end
    return err * W * err'
end


"""
    full_SMM(params, simulation=false, second_stage=false;
             analytical=false, n_quad=200, ...)

Full SMM evaluation: compute loss and return moments.

When `analytical=true`, uses compute_moments_analytical (closed-form EK formulas
+ Gauss-Legendre quadrature for reg_coef). The simulation/u_draws/sample_weights
kwargs are ignored in analytical mode.
"""
function full_SMM(params, simulation=false, second_stage=false;
                  precomputed_tau::Union{Nothing, Matrix{Float64}}=nothing,
                  u_draws::Union{Nothing, Matrix{Float64}}=nothing,
                  sample_weights::Union{Nothing, Matrix{Float64}}=nothing,
                  W_override::Union{Nothing, AbstractMatrix}=nothing,
                  moment_blocks::Union{Nothing, Vector{Int}}=nothing,
                  analytical::Bool=false,
                  n_quad::Int=200,
                  N_fixed::Union{Nothing, Vector{Int}}=nothing)

    if analytical
        @assert !GRANULAR "the analytical/GMM path does not implement the granular count " *
            "moment (block 6): the extensive margin there is the FKG-approximated " *
            "continuum object. Run granular estimation through the SMM path (main.jl)."
        moms_nt = compute_moments_analytical(params; n_quad=n_quad)
        simulated_moments = moment_blocks_tuple(moms_nt)
    else
        simulated_moments = SMM(params, simulation; precomputed_tau=precomputed_tau,
                                u_draws=u_draws, sample_weights=sample_weights,
                                N_fixed=N_fixed)
    end

    # `second_stage` is retained as an ignored positional arg for call-site
    # compatibility (parallel_SMM / parallel_SMM_safe / train_stage_pso pass it
    # positionally); the second-stage masked-moment branch was dead and removed.
    emp = empirical_moments
    W = W_override !== nothing ? W_override : Weight_matrix_custom
    moments = simulated_moments

    moment_indices = moment_blocks === nothing ? nothing :
        vcat([collect(BLOCK_RANGES[b]) for b in moment_blocks]...)

    if simulation
        return simulated_moments
    else
        return loss_function(moments, emp, W; moment_indices=moment_indices), simulated_moments
    end
end