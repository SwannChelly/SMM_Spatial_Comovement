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
    FlatWeights(w, n, m) <: AbstractMatrix{Float64}

An `n × m` matrix every entry of which is `w`, stored in three words.

Both surviving samplers (`:sobol`, `:mc`) return FLAT weights `1/N_rho` — that is
precisely what makes `solve_network`'s winner-weight shortcut exact, and hence what
makes both unbiased for the min-coupled moments. Materialising that as a dense
`fill(1/N_rho, N_rho, n_good)` cost one full-size matrix (6.7 MB at N_rho = 723,
n_good = 1161; 37 MB at `N_RHO_INFERENCE = 4000`) **per worker**, allocated again for
every Jacobian and every Σ_sim replication, to hold one repeated number.

This type keeps the `[rho, g]` indexing contract the whole codebase is built on —
that invariant is what confines the draw method to the generation sites — while
storing nothing. Any consumer that genuinely needs a dense copy can `collect` it.
"""
struct FlatWeights <: AbstractMatrix{Float64}
    w :: Float64
    n :: Int
    m :: Int
end

Base.size(W::FlatWeights) = (W.n, W.m)
Base.IndexStyle(::Type{FlatWeights}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(W::FlatWeights, ::Int) = W.w

"""
    flat_weight(W) -> Float64 or nothing

The common value of `W` if it is constant (a `FlatWeights`), else `nothing`. Lets a
hot loop hoist the weight out entirely instead of indexing per row, without giving up
support for a genuinely non-flat weight matrix.
"""
flat_weight(W::FlatWeights) = W.w
flat_weight(::AbstractMatrix{Float64}) = nothing


function generate_mc_draws(N_rho::Int, n_good::Int, rng::AbstractRNG)
    U = rand(rng, N_rho, n_good)
    # Uniform weight matrix (N_rho × n_good) so consumers can index [rho, g],
    # matching the per-column weight convention of the other samplers. Stored as a
    # FlatWeights — constant value, no per-element storage.
    w = FlatWeights(1.0 / N_rho, N_rho, n_good)
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
FLAT weights. Each SECTOR's active (sector, region) columns form one
Sobol net of dimension `d = length(SECTOR_GOOD_INDICES[s])`, scrambled by a
per-sector digital shift consumed from the master rng. Building one net per
sector (rather than a single global `n_good`-dim net) keeps the dimension low
where the Ricardian `min_r c_r` interaction actually lives — within a sector,
across its regions — which is where the net's superior equidistribution pays off;
distinct sectors draw distinct shifts and are therefore decorrelated.

Returns `U::Matrix (N_rho × n_good)` and the flat weight `W = FlatWeights(1/N_rho, …)`.
The flat weight makes `solve_network`'s winner-weight shortcut EXACT, so this is
unbiased for the min-coupled moments (the same property `:mc` has).

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
    # `FlatWeights` keeps the [rho, g] contract with no per-element storage.
    W = FlatWeights(1.0 / N_rho, N_rho, n_good)
    return U, W
end


"""
    generate_draws(N_rho, n_good, method::Symbol; randomise=false,
                   rng=Random.GLOBAL_RNG, verbose=false) -> (U, W::AbstractMatrix)

Unified draw dispatcher. `method ∈ (:sobol, :mc)` — both carry FLAT weights, which
is what makes `solve_network`'s winner-weight shortcut exact and hence both
unbiased for the min-coupled moments (`c_tilde`, `gamma_ls`, `industry`, `pi_r`):

  - `:sobol` — per-sector digitally-shifted Sobol net. The OPTIMISATION default:
    `randomise=false` gives a frozen scramble seed, so the SMM criterion is a
    deterministic function of θ (PSO-safe), and the net's equidistribution buys
    variance reduction on thick (multi-region) sectors.
  - `:mc` — i.i.d. uniform. The INFERENCE default (`INFERENCE_DRAW_METHOD`):
    Σ_sim and the Jacobian want genuinely independent designs per replication,
    and i.i.d. draws avoid the cross-sector coupling of the per-sector Sobol nets
    (two sectors sharing a Sobol dimension index differ only by a digital shift,
    so their columns are a deterministic XOR of one another — invisible to a
    correlation gate, but it distorts the cross-sector CES price aggregation).
    `randomise=false` ⇒ deterministic (`MersenneTwister(0)`); `randomise=true` ⇒
    supplied `rng`.

The `:is` (importance-sampling) and `:qmc` (stratified-uniform) samplers were
REMOVED. `:is` was biased for every min-coupled moment — `solve_network` applies
only the winning column's weight per row and drops the losing columns' density
ratios, which do not cancel under a non-flat tilt — and `:qmc` was strictly
dominated by `:sobol` at the same cost.

W is always an `(N_rho × n_good)` AbstractMatrix — a `FlatWeights`, since both
methods are flat — so every `[rho, g]` consumer is
untouched regardless of method — the invariant that confines the method switch to
the generation sites.
"""
function generate_draws(N_rho::Int, n_good::Int, method::Symbol;
                        randomise::Bool=false,
                        rng::AbstractRNG=Random.GLOBAL_RNG,
                        verbose::Bool=false)
    if method === :sobol
        return generate_sobol_draws(N_rho, n_good; randomise=randomise, rng=rng,
                                    seed=42)
    elseif method === :mc
        mc_rng = randomise ? rng : MersenneTwister(0)
        return generate_mc_draws(N_rho, n_good, mc_rng)
    else
        error("Unknown draw method :$method (choose :sobol or :mc)")
    end
end


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
    gbar_logfact_table(m) -> Vector{Float64}

`lg[i+1] = log(i!)` for `i = 0:m`. Built by cumulative sum (no SpecialFunctions
dependency); `m + 1` entries, rebuilt per call — negligible beside one moment
evaluation.
"""
function gbar_logfact_table(m::Integer)
    lg = Vector{Float64}(undef, m + 1)
    lg[1] = 0.0
    @inbounds for i in 1:m
        lg[i+1] = lg[i] + log(i)
    end
    return lg
end

"""
    gbar_cell(k, m, n, lg) -> Float64

UNBIASED estimator of `(1 − q_l)^n`, the probability that cell `l` hosts no
supplier among `n` varieties, from `k` observed wins out of `m` draws.

    E[ C(m−k, n) / C(m, n) ] = (1 − q)^n      for every n ≤ m

Read it as: draw `n` of the `m` simulated varieties without replacement and ask
whether all of them lost. Marginalising over the draws, each is a win with
probability `q` independently, so the expectation is exactly `(1 − q)^n` — no
approximation, no tuning constant.

This REPLACES the plug-in `(1 − q̂)^n` with its `q̂` floored at `0.5/m`. Two
defects of that construction are removed at once:

  * **The floor capped identification.** `q̂ ≥ 0.5/m` forced
    `Ḡ_s(0) ≤ (1 − 0.5/m)^n` for every θ, so no `N̂_s` above
    `ln(Ḡ_target)/ln(1 − 0.5/m)` was reachable — roughly 11–53 at `m = 100`
    against variety-count bounds `N_HI` of 24–291. The upper half of the
    bisection range was unreachable, so a `:hi` clamp could never occur and the
    over-identification check was silently one-sided. The estimator below has no
    such ceiling: it spans the whole `[N_LO, N_HI]` range.
  * **Jensen bias.** `(1 − q)^n` is convex in `q`, so the plug-in is biased
    upward by roughly `n(n−1)q / (2m(1−q))` — several percent of `Ḡ` at
    production draw counts, enough to move `N̂_s` by tens of percent.

Evaluated in logs off a shared log-factorial table, so it costs `O(1)` per
(cell, n) rather than `O(n)`:

    log Ḡ = log(m−k)! − log(m−k−n)! − log m! + log(m−n)!

`n > m − k` ⇒ 0 (a cell winning `k` of `m` draws cannot leave `n` varieties
unserved once `n` exceeds the losing draws). `n ≤ 0` ⇒ 1.

**Caveat.** Unbiasedness assumes the `m` draws are i.i.d. within a cell. The
optimisation draws are `:sobol`, which is equidistributed rather than
independent — that makes the win count LESS dispersed than Binomial, so the
residual bias is smaller than the plug-in's, not larger, but it is not exactly
zero. Inference draws (`:mc`) do satisfy the assumption.
"""
@inline function gbar_cell(k::Integer, m::Integer, n::Integer, lg::Vector{Float64})
    n <= 0 && return 1.0
    n > m - k && return 0.0
    return exp(lg[m-k+1] - lg[m-k-n+1] - lg[m+1] + lg[m-n+1])
end

"""
    gbar_sector(cells, k_counts, m, n, lg) -> Float64

`Ḡ_s(n) = mean over cells l of (1 − q_l)^n`, via the unbiased `gbar_cell`. This
is the ONE definition of the count moment: `concentrate_N_s` bisects on it and
block 6 reports it at `N̂_s`, so the moment being matched and the moment being
reported cannot drift apart.
"""
function gbar_sector(cells, k_counts::AbstractVector{<:Integer}, m::Integer,
                     n::Integer, lg::Vector{Float64})
    isempty(cells) && return 1.0
    acc = 0.0
    @inbounds for g in cells
        acc += gbar_cell(k_counts[g], m, n, lg)
    end
    return acc / length(cells)
end


"""
    concentrate_N_s(k_counts, m) -> (N_hat::Vector{Int}, clamped::Vector{Symbol})

Profile the variety count `N_s` out of the loss by a **monotone integer bisection**
on the count moment (plan D5; `finite_sample2.tex` §4.2).

`k_counts[g]` is the number of the `m` simulated varieties that cell `g` wins
SOMEWHERE in the downstream industry (the column sum of `linkages_flat`). By Lemma 2
(`q ⊥ N_s`) the underlying win probability does not depend on the variety count, so

    Ḡ_s(n) = mean over cells l of sector s of (1 − q_ls)^n

is closed form and decreasing in `n`; matching it to `G_TARGET[s]` needs no
re-simulation. It is evaluated by `gbar_sector`/`gbar_cell`, the UNBIASED
combinatorial estimator — see `gbar_cell` for why the earlier plug-in
`(1 − q̂)^n` with `q̂` floored at `0.5/m` capped the reachable `N̂_s` and biased
`Ḡ` upward. The search is over the integers of `[N_LO[s], N_HI[s]]`, the bounds
implied by the observed distinct-supplier count, so `N̂_s` is an integer at every
evaluation — never relaxed, never rounded — and the outer optimiser never sees it.

`clamped[s] ∈ (:none, :lo, :hi)` records a bound that bound. Clamping is
INFORMATIVE, not benign: `:hi` means the model cannot generate enough sparsity even
when every variety is sourced from a single origin — a rejection signal for the
mechanism, which is why it is reported per sector rather than silently absorbed.
"""
function concentrate_N_s(k_counts::AbstractVector{<:Integer}, m::Integer)
    N_hat   = zeros(Int, S)
    clamped = fill(:none, S)
    lg      = gbar_logfact_table(m)
    @inbounds for s in 1:S
        cells = CELLS_OF_SECTOR[s]
        lo0, hi0 = N_LO[s], N_HI[s]
        if isempty(cells)
            N_hat[s] = lo0
            continue
        end
        Gs = n -> gbar_sector(cells, k_counts, m, n, lg)
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
- `z_flat`: Firm productivities [N_rho × n_good], or `nothing` when `return_z=false`
- `logz_const`: per-good `log(scale_g)`, the θ-dependent half of `log z` (see below)
- `u_draws`: the draws the network was solved on (a reference, not a copy)
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
                       sample_weights::Union{Nothing, AbstractMatrix{Float64}}=nothing,
                       return_z::Bool=true)

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

    # Per-good constant part of log z, filled alongside z below.
    #
    #   log z[ρ,g] = log(scale_g)  +  (−1/θ)·log(−log(1 − u[ρ,g]))
    #              = logz_const[g] +  logz_resid[ρ,g]
    #
    # The SECOND term does not depend on θ — it is a function of the draws alone. That
    # is what lets the streaming cloglog regression reconstruct the size regressor from
    # `u_draws` (cached once per draw set) instead of storing an 839k-row column, and
    # what lets it represent the whole IRLS linear predictor as a per-CELL vector; see
    # `logz_resid` and `_cloglog_irls_cells`. Left as NaN on the legacy random-Fréchet
    # branch, where there is no `u` to decompose and the dense regression path is used.
    logz_const = fill(NaN, n_good)

    z_flat = nothing
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
            sample_weights = FlatWeights(1.0/N_rho_eff, N_rho_eff, n_good)
        end
        z_inv_flat = z_flat .^ (-1)
    else
        # CdGM-style: Fréchet inverse CDF from stratified uniform draws
        # u_draws is a Matrix{Float64} (N_rho_eff × n_good) with per-pair quantiles
        # F⁻¹(u) = σ · (-ln(1-u))^(-1/θ) where σ = T_{sr}^{1/θ}
        #
        # Only `z_inv_flat` is needed inside this function (the Ricardian argmin). When
        # the caller does not want `z_flat` back either — the streaming regression
        # rebuilds log z from `u_draws` + `logz_const`, so nothing downstream reads it —
        # z is formed one COLUMN at a time into a length-N_rho scratch buffer instead of
        # a second full-size matrix. Same operations on the same values, in the same
        # order, so `z_inv_flat` is BIT-IDENTICAL either way; that matters because it
        # drives the argmin, where a last-ulp difference could flip a winner.
        # The two branches are spelled out rather than sharing a `zcol` that is either a
        # column view or the scratch buffer: that would make the destination a Union
        # inside the hot per-variety loop, costing a dynamic dispatch per element.
        if return_z
            # Unchanged from before the memory work: build z in full, invert in one
            # whole-matrix broadcast.
            z_flat = zeros(N_rho_eff, n_good)
            for g in 1:n_good
                T_sr = T[GOOD_S[g], GOOD_R[g]]
                scale = max(T_sr, eps(Float64))^(1/theta)
                logz_const[g] = log(scale)
                for rho in 1:N_rho_eff
                    z_flat[rho, g] = scale * (-log(1.0 - u_draws[rho, g]))^(-1.0/theta)
                end
            end
            z_inv_flat = z_flat .^ (-1)
        else
            z_inv_flat = Matrix{Float64}(undef, N_rho_eff, n_good)
            zbuf = Vector{Float64}(undef, N_rho_eff)
            for g in 1:n_good
                T_sr = T[GOOD_S[g], GOOD_R[g]]
                scale = max(T_sr, eps(Float64))^(1/theta)
                logz_const[g] = log(scale)
                for rho in 1:N_rho_eff
                    zbuf[rho] = scale * (-log(1.0 - u_draws[rho, g]))^(-1.0/theta)
                end
                @views z_inv_flat[:, g] .= zbuf .^ (-1)
            end
        end
    end

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
    # Firm-level supplier indicator. A BitMatrix, not a Float64 matrix: it is a pure
    # 0/1 flag, so this is 1/64 of the storage (6.7 MB → 0.1 MB at N_rho = 723,
    # n_good = 1161; 37 MB → 0.6 MB at N_RHO_INFERENCE). Every consumer reads it
    # through `> 0` / `!= 0` / `sum` / `count`, which are unchanged on a BitMatrix.
    linkages_flat = falses(N_rho_eff, n_good)

    # Per-downstream-region scratch, hoisted OUT of the `for r_d` loop below.
    # These four are (N_rho_eff × S) and used to be allocated afresh on every one of
    # the R_downstream passes — of the order of R_downstream × 4 × N_rho × S × 8 bytes
    # of pure churn per solve_network call, allocated only to be overwritten on the
    # next pass. Nothing survives an iteration, so one buffer set serves all of them.
    #
    # Reuse is BIT-IDENTICAL, not merely equivalent, on one condition each:
    #   * `winner_good_idx` / `p_rho_s` — the sector loop `continue`s on an EMPTY
    #     sector, leaving those columns untouched. Sector emptiness does not depend on
    #     r_d, so a column skipped on one pass is skipped on every pass and keeps the
    #     zeros it was allocated with — exactly what `zeros(...)` per pass gave.
    #   * `w_rho_s` / `ces_buf` — written for every (rho, s) on every pass, so there is
    #     no stale state to inherit.
    p_rho_s         = zeros(N_rho_eff, S)
    winner_good_idx = zeros(Int, N_rho_eff, S)
    w_rho_s         = Matrix{Float64}(undef, N_rho_eff, S)
    ces_buf         = Matrix{Float64}(undef, N_rho_eff, S)

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
        # p_rho_s / winner_good_idx are the hoisted buffers (see above): every
        # non-empty sector overwrites its column in full below, and empty sectors keep
        # the zeros they were allocated with.
        for s in 1:S
            g_indices = SECTOR_GOOD_INDICES[s]
            if isempty(g_indices); continue; end
            regions_s = SECTOR_GOOD_REGIONS[s]

            # Ricardian selection: lowest-cost supplier wins each variety.
            #
            # Fused min-and-argmin. The previous form materialised
            #   prices_s = z_inv_flat[:, g_indices] .* tau_sr .* w_sr
            # and then reduced it with argmin(...; dims=2), allocating an
            # (N_rho × n_active_in_s) matrix plus a CartesianIndex array for EVERY
            # (downstream region, sector) pair — of the order of 60 MB per
            # solve_network call at production sizes, purely to be reduced away.
            #
            # The arithmetic is unchanged: `z * t * w` associates left-to-right
            # exactly as the broadcast `z .* tau .* w` did, so the products are
            # bit-identical. Ties keep the FIRST minimiser and a NaN wins outright,
            # both matching `argmin`.
            tv = tau[regions_s, r_d]           # (n_active_in_s,) — tau is R × R_downstream
            wv = W_RS_FLAT[g_indices]          # (n_active_in_s,)
            n_act = length(g_indices)
            @inbounds for rho in 1:N_rho_eff
                g1   = g_indices[1]
                best = z_inv_flat[rho, g1] * tv[1] * wv[1]
                bidx = 1
                if !isnan(best)
                    for li in 2:n_act
                        gl = g_indices[li]
                        p  = z_inv_flat[rho, gl] * tv[li] * wv[li]
                        if isnan(p)
                            best = p; bidx = li; break
                        elseif p < best
                            best = p; bidx = li
                        end
                    end
                end
                winner_good_idx[rho, s] = g_indices[bidx]
                p_rho_s[rho, s] = best
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
        for s in 1:S
            nu_s_s = nu_s[s]
            for rho in 1:N_rho_eff
                g_w = winner_good_idx[rho, s]
                w_rho_s[rho, s] = g_w == 0 ? 0.0 : sample_weights[rho, g_w]
                # Fused into the same pass, into a hoisted buffer. The broadcast
                # `w_rho_s .* p_rho_s.^(1 .- nu_s_mat)` materialised TWO further
                # (N_rho × S) temporaries per downstream region — the power and the
                # product — purely to be reduced away by the `sum` on the next line.
                # Same operands in the same order, and the reduction is still the same
                # `sum(...; dims=1)` (pairwise), so P_sr is bit-identical.
                ces_buf[rho, s] = w_rho_s[rho, s] * p_rho_s[rho, s]^(1 - nu_s_s)
            end
        end
        P_sr = sum(ces_buf, dims=1).^(1 ./ (1 .- nu_s_mat))
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
                linkages_flat[rho, g_winner] = true

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
    # `z_flat` is a full-size matrix (6.7 MB / 37 MB) that the production moment path no
    # longer reads: the streaming cloglog rebuilds log z from `u_draws` + `logz_const`.
    # Dropping the reference here means it is collectable BEFORE the regression runs —
    # i.e. at the peak of the evaluation — rather than being pinned until the moments
    # return. External callers (the LPM path, `test/`, `extras/`) keep it via the
    # default `return_z = true`.
    z_out = return_z ? z_flat : nothing

    if return_firm_level
        return (
            X_ls_flat = X_ls_flat,
            c_tilde_r = c_tilde_r_full,
            Y_r = Y_r_full,
            linkages_flat = linkages_flat,
            z_flat = z_out,
            logz_const = logz_const,
            u_draws = u_draws,
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
            z_flat = z_out,
            logz_const = logz_const,
            u_draws = u_draws,
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
function fast_weighted_regression(linkages_flat, z_flat, sample_weights::AbstractMatrix{Float64};
                                  include_control::Bool=true,
                                  include_size_control::Bool=!include_control,
                                  rho_range::Union{Nothing, Vector{UnitRange{Int}}}=nothing,
                                  obs_weight::Union{Nothing, Vector{Float64}}=nothing,
                                  return_size_coef::Bool=false,
                                  u_draws::Union{Nothing, Matrix{Float64}}=nothing,
                                  logz_const::Union{Nothing, Vector{Float64}}=nothing,
                                  inv_theta::Union{Nothing, Float64}=nothing)

    # Regressors: N_REG distance-bin dummies, plus (optionally) a log-z size control.
    # The size control and the control group are MUTUALLY EXCLUSIVE by construction:
    # the control-only (filter==2) firms have no productivity (z ≡ −∞), so they cannot
    # carry a log-z regressor. Hence the default couples them — size control ON only
    # when the control group is OFF (include_size_control = !include_control):
    #   • include_control=true  (production): control y=0 rows appended, NO log-z column.
    #   • include_control=false (diagnostic): supplier pairs only, log-z size control ADDED
    #     — conditioning on productivity purges the T-through-z omitted-variable confound,
    #       so the distance slope loads on the trade-cost/α channel (see identification note).
    # The log-z regressor is taken from `z_flat` on the dense path and rebuilt from
    # `u_draws` + `logz_const` on the streaming one; `z_flat` is only materialised by
    # `solve_network` when some caller actually needs it (see `SMM`'s `need_z`).
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

    # ── Streaming (cell-level) path — the production path ─────────────────────
    # Same fit, one pass, nothing of order N_valid allocated. Gated by exactly the
    # predicate the cloglog sibling uses, so the two links agree on when `z_flat` is
    # needed at all (see `SMM`).
    if reg_streaming_ok(include_size_control, rho_range, obs_weight, u_draws, logz_const)
        cells = _build_reg_cells(include_control, include_size_control,
                                 N_rho_eff, logz_const)
        lzr = include_size_control ?
              logz_resid(u_draws, inv_theta === nothing ? 1.0 / theta : inv_theta) :
              nothing
        coefs = _wls_cells(cells, linkages_flat, sample_weights, lzr,
                           n_regressors, size_col)
        return return_size_coef ?
            vcat(coefs[1:N_REG], include_size_control ? coefs[size_col] : NaN) :
            coefs[1:N_REG]
    end

    @assert z_flat !== nothing (
        "the dense LPM path needs z_flat, but solve_network was called with " *
        "return_z=false. Either pass u_draws/logz_const so the streaming path applies, " *
        "or solve with return_z=true.")

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
    CloglogDesign

Per-process cache of the part of the extensive-margin regression design that does
NOT move with θ: the distance columns of `X`, the observation weights `w`, the
fixed-effect labels `fe_group`, and the group → row index derived from them. Only
the not-supply indicator `y` and the log-z size column are refreshed per call, via
the `row_g`/`row_rho` maps.

Validity is checked on two things: the design SHAPE (`key`) and the IDENTITY of the
weight matrix the design was built from (`weights_ref`, compared with `===`). The
second is what makes it safe under the Jacobian and Σ_sim replications, which pass
their own draws: a different weight matrix object is a cache miss, not a silent
reuse. Holding the reference also pins the matrix, which is intended.

Julia's `Distributed` gives every worker its own copy of module state and no
threads are used anywhere in this codebase, so this global is not shared.
"""
mutable struct CloglogDesign
    key          :: NTuple{6, Int}
    weights_ref  :: AbstractMatrix{Float64}
    X            :: Matrix{Float64}
    w            :: Vector{Float64}
    fe_group     :: Vector{Int}
    group_rows   :: Vector{Vector{Int}}
    y            :: Vector{Float64}
    row_g        :: Vector{Int}
    row_rho      :: Vector{Int}
    n_rows_goods :: Int
end

const _CLOGLOG_DESIGN = Ref{Union{Nothing, CloglogDesign}}(nothing)

"""
    reset_cloglog_design!()

Drop the cached design. Only needed by tests that want to force a rebuild; the
identity check on the weight matrix handles every production path.
"""
reset_cloglog_design!() = (_CLOGLOG_DESIGN[] = nothing)

function _build_cloglog_design(sample_weights::AbstractMatrix{Float64}, key, include_control::Bool,
                               include_size_control::Bool, rows_of, obs_weight,
                               N_rho_eff::Int, n_ctrl_eff::Int, N_valid::Int,
                               n_regressors::Int, n_rows_goods::Int, size_col::Int)
    y        = Vector{Float64}(undef, N_valid)
    X        = zeros(N_valid, n_regressors)
    w        = Vector{Float64}(undef, N_valid)
    fe_group = Vector{Int}(undef, N_valid)
    row_g    = Vector{Int}(undef, n_rows_goods)
    row_rho  = Vector{Int}(undef, n_rows_goods)

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
            y[idx] = 0.0                                  # overwritten per call
            w[idx] = obs_weight === nothing ? sample_weights[rho, g] : obs_weight[g]
            fe_group[idx] = group_id
            row_g[idx] = g; row_rho[idx] = rho
            if N_REG == 1
                X[idx, 1] = log_dist
            else
                (b > 0 && b <= N_REG) && (X[idx, b] = 1.0)
            end
            # X[idx, size_col] is written per call when include_size_control.
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
    @assert idx == N_valid "cloglog design built $idx rows, expected $N_valid"

    return CloglogDesign(key, sample_weights, X, w, fe_group,
                         build_fe_group_rows(fe_group), y, row_g, row_rho, n_rows_goods)
end

"""
    build_fe_group_rows(fe_group) -> Vector{Vector{Int}}

Row indices of each non-empty fixed-effect group, in one O(n) counting pass.

Replaces `Dict(g => findall(==(g), fe_group) for g in unique(fe_group))`, which
costs one full scan of `fe_group` PER GROUP — with `S × R_downstream` groups over
`n_good × N_rho` rows that is tens of millions of comparisons on every call, for a
result that is identical every time (`fe_group` is pure geography). The FE loops that
consume this are per-group and operate on disjoint row sets, so group ORDER does not
affect any result; this returns them in group-id order rather than first-appearance
order.
"""
function build_fe_group_rows(fe_group::AbstractVector{<:Integer})
    isempty(fe_group) && return Vector{Int}[]
    gmax   = maximum(fe_group)
    counts = zeros(Int, gmax)
    @inbounds for g in fe_group
        counts[g] += 1
    end
    out = Vector{Vector{Int}}()
    slot = zeros(Int, gmax)          # slot[g] = position of group g in `out`, 0 if empty
    for g in 1:gmax
        if counts[g] > 0
            push!(out, Vector{Int}(undef, counts[g]))
            slot[g] = length(out)
        end
    end
    fill_at = zeros(Int, length(out))
    @inbounds for i in eachindex(fe_group)
        p = slot[fe_group[i]]
        fill_at[p] += 1
        out[p][fill_at[p]] = i
    end
    return out
end

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

Pass `group_rows` (from `build_fe_group_rows`) to skip rebuilding the FE index, and
`iters_out` to read back how many IRLS iterations were actually used.
"""
function _cloglog_irls(y::AbstractVector{<:Real}, X::AbstractMatrix{<:Real},
                       w::AbstractVector{<:Real}, fe_group::AbstractVector{<:Integer};
                       group_rows::Union{Nothing,Vector{Vector{Int}}}=nothing,
                       max_iter::Int=50, tol::Float64=1e-9, eta_clamp::Float64=30.0,
                       iters_out::Union{Nothing,Base.RefValue{Int}}=nothing)
    n, k = size(X)
    yf = y isa Vector{Float64} ? y : Float64.(y)
    Xf = X isa Matrix{Float64} ? X : Float64.(X)
    wf = w isa Vector{Float64} ? w : Float64.(w)

    # Group → row indices. Supplied by the caller when the design is cached (the
    # common case); otherwise built in one pass.
    grows = group_rows === nothing ? build_fe_group_rows(fe_group) : group_rows

    # Init: μ shrunk toward 0.5 (avoids η = ±∞ at all-0 / all-1 starts), η = cloglog link.
    μ = (yf .+ 0.5) ./ 2
    η = clamp.(log.(.-log.(1 .- μ)), -eta_clamp, eta_clamp)

    # ── Buffers, allocated ONCE rather than once per IRLS iteration ────────────
    # The loop below used to `copy(Xf)` every pass; at production sizes that is
    # ~17 MB of churn per iteration and up to 50 iterations per evaluation.
    dμdη  = Vector{Float64}(undef, n)
    zwork = Vector{Float64}(undef, n)
    W     = Vector{Float64}(undef, n)
    zt    = Vector{Float64}(undef, n)
    Xt    = Matrix{Float64}(undef, n, k)
    Xb    = Vector{Float64}(undef, n)
    η_new = Vector{Float64}(undef, n)
    sw_g  = Vector{Float64}(undef, length(grows))

    β = zeros(k)
    n_it = 0
    for _ in 1:max_iter
        n_it += 1
        @inbounds for i in 1:n
            μi      = clamp(1 - exp(-exp(η[i])), 1e-12, 1 - 1e-12)
            di      = exp(η[i]) * (1 - μi)               # e^{η}·e^{−e^{η}}
            dμdη[i] = di
            zwork[i] = η[i] + (yf[i] - μi) / di          # working response
            W[i]     = wf[i] * (di^2) / (μi * (1 - μi))  # IRLS × analytic weights
            zt[i]    = zwork[i]
        end
        copyto!(Xt, Xf)

        # Weighted within-group demeaning (FWL) of zwork and each X column.
        @inbounds for (p, r) in enumerate(grows)
            sw = sum(@view W[r])
            sw_g[p] = sw
            sw < 1e-300 && continue
            cz = sum(i -> W[i] * zwork[i], r) / sw
            for i in r
                zt[i] -= cz
            end
            for j in 1:k
                cx = sum(i -> W[i] * Xf[i, j], r) / sw
                for i in r
                    Xt[i, j] -= cx
                end
            end
        end

        # Weighted LS on the demeaned system: (Xt' W Xt) β = Xt' W zt.
        WXt   = W .* Xt
        β_new = (Xt' * WXt) \ (Xt' * (W .* zt))

        # Rebuild η = Xβ + a_fe, with a_g = weighted group mean of (zwork − Xβ).
        # Xf*β is formed ONCE (it was computed twice: for `resid` and for `η_new`).
        mul!(Xb, Xf, β_new)
        @inbounds for i in 1:n
            η_new[i] = Xb[i]
        end
        @inbounds for (p, r) in enumerate(grows)
            sw = sw_g[p]
            sw < 1e-300 && continue
            cr = sum(i -> W[i] * (zwork[i] - Xb[i]), r) / sw
            for i in r
                η_new[i] += cr
            end
        end
        @inbounds for i in 1:n
            η_new[i] = clamp(η_new[i], -eta_clamp, eta_clamp)
        end

        Δ = maximum(abs.(β_new .- β))
        β = β_new
        copyto!(η, η_new)
        Δ < tol && break
    end
    iters_out === nothing || (iters_out[] = n_it)
    return β
end

##################### Streaming (cell-level) regression design ###################

"""
    logz_resid(u_draws) -> Matrix (N_rho × n_good)

The θ-INVARIANT half of the log-productivity regressor,

    logz_resid[ρ,g] = (−1/θ)·log(−log(1 − u[ρ,g])),

so that `log z[ρ,g] = logz_const[g] + logz_resid[ρ,g]` with
`logz_const[g] = log(max(T_{s(g)r(g)}, eps)^{1/θ})` (filled by `solve_network`).

Only `logz_const` moves with θ. Caching this matrix on the IDENTITY of the draw set
means it is built ONCE per draw set rather than once per evaluation — under the
Jacobian one replication runs `2·n_perturb+1` evaluations on the same `u_k`, so this
removes that many rebuilds, along with the `n_good × N_rho` `log()` calls each of them
used to spend materialising the size column.

The previous cache entry is dropped BEFORE the new matrix is allocated, so the Σ_sim
loop (a fresh draw set per replication) never holds two of these at once.
"""
mutable struct LogzResidCache
    u_ref :: Matrix{Float64}
    M     :: Matrix{Float64}
end

const _LOGZ_RESID = Ref{Union{Nothing, LogzResidCache}}(nothing)

reset_logz_resid!() = (_LOGZ_RESID[] = nothing)

function logz_resid(u_draws::Matrix{Float64}, inv_theta::Float64)
    C = _LOGZ_RESID[]
    (C !== nothing && C.u_ref === u_draws) && return C.M

    _LOGZ_RESID[] = nothing            # let the old one go before allocating the new
    n, m = size(u_draws)
    M = Matrix{Float64}(undef, n, m)
    @inbounds for j in 1:m, i in 1:n
        M[i, j] = -inv_theta * log(-log(1.0 - u_draws[i, j]))
    end
    _LOGZ_RESID[] = LogzResidCache(u_draws, M)
    return M
end

"""
    REG_STREAMING

Master switch for the streaming (cell-level) extensive-margin regression, ON by
default — it is the production path for BOTH links, `:cloglog` and `:lpm`.

Set `REG_STREAMING[] = false` to route `fast_cloglog_regression` and
`fast_weighted_regression` back through their dense kernels, which materialise the
`N_valid × k` design and are retained verbatim as the reference implementations. Kept
so the two can be diffed at any point — see `test/test_cloglog_streaming.jl`.

A `Ref`, not a `const Bool`, so it can be flipped at runtime; under `Distributed` each
worker has its own copy, so flipping it on the master does NOT propagate (use
`@everywhere` if you mean to change every worker).
"""
const REG_STREAMING = Ref(true)

"""
    reg_streaming_ok(include_size_control, rho_range, obs_weight, u_draws, logz_const)

Whether the extensive-margin regression can take the streaming path (both links). It
cannot when the switch is off, when a non-production row layout is requested (`rho_range` /
`obs_weight`), or when the size control is on but the `log z` decomposition is
unavailable — which happens exactly on the legacy random-Fréchet branch of
`solve_network`, where there is no `u` and `logz_const` is left NaN.

This is also what decides whether `z_flat` must be materialised at all: see `SMM`.
"""
function reg_streaming_ok(include_size_control::Bool, rho_range, obs_weight,
                          u_draws, logz_const)
    REG_STREAMING[] || return false
    (rho_range === nothing && obs_weight === nothing) || return false
    include_size_control || return true
    return u_draws !== nothing && logz_const !== nothing && all(isfinite, logz_const)
end

"""
    _solve_within_normal_eq(A, rhs, ctx) -> β

`A \\ rhs` on the `k × k` within-group normal equations, replacing the bare
`SingularException(k)` with the diagnosis.

`A = X̃'WX̃` is singular exactly when the demeaned design is rank-deficient, and there is
one overwhelmingly likely cause: if EVERY row carries a distance dummy, the `N_REG`
dummy columns sum to the constant — which the fixed effect has already absorbed — so
`(1,1,…,1,0)` is an exact null vector no matter how much within-group variation the bins
have. `distance_bin` returns 0 below its first cutoff precisely to leave that base
category empty; a bin scheme with no such cells is not estimable.

The dense cloglog kernel fails the same way (it also solves a `k × k` system), but the
dense LPM kernel does a TALL QR, which can quietly return a solution for a
rank-deficient design. Erroring here is deliberate: an unidentified coefficient vector
that depends on the solver is worse than a stop.
"""
function _solve_within_normal_eq(A::Matrix{Float64}, rhs::Vector{Float64}, ctx::String)
    try
        return A \ rhs
    catch e
        e isa LinearAlgebra.SingularException || rethrow()
        error("""
        $ctx: the within-group normal equations are singular (rank-deficient design).

        The usual cause is that no cell falls in the BASE distance category, so the
        $(size(A,1) > 1 ? "N_REG" : "single") dummy columns sum to the constant that the
        (sector × nearest-downstream) fixed effect already absorbs. Check that
        `distance_bin` returns 0 for at least some (r, dr) pairs — it does so below its
        first cutoff (d ≤ 20 for N_REG = 5, d ≤ 50 for N_REG = 4).

        Other possibilities: a bin that no cell occupies, or a log-z column that is
        constant within every fixed-effect group.""")
    end
end

"""
    RegCells

The extensive-margin design at CELL resolution — one entry per (sector, region) cell
rather than one per (cell, variety) row.

The dense design has `n_good × N_rho` rows (839,403 at n_good = 1161, N_rho = 723), but
of its five columns only the log-z one varies down the rows of a cell: the distance
regressors, the fixed effect and the weight are all cell-level, stored `N_rho` times
over. This holds each of them ONCE:

  - `gcol[c]`, `gval[c]` — the geography regressor as a (column, value) pair.
    With `N_REG > 1` the design is bin DUMMIES, so `gval = 1` and `gcol` is the bin
    (0 = base category, an all-zero row); with `N_REG == 1` it is the CONTINUOUS
    log-distance, so `gcol = 1` and `gval = LOG_CLOSEST_DIST[r]`. Because a row carries
    at most one dummy, the dummy block of `X'WX` is diagonal by construction.
  - `grp[c]` — the (sector × nearest-downstream) fixed effect, compacted to `1:n_grp`.
  - `has_z[c]` — whether the row carries the log-z control. False for control-only
    cells, which have no productivity draw and take a size column of exactly 0.
  - `lzc[c]` — `logz_const[g]`, the θ-dependent part of `log z`.

Cells `1:n_good` are the goods; `n_good+1 : n_good+n_ctrl` are the control-only pairs
(no supplier, flat weight `1/N_rho`). Total storage is `O(n_cell)`, i.e. kilobytes
against the 74 MB (408 MB at `N_RHO_INFERENCE`) of `CloglogDesign`.

Shared by BOTH links: `_cloglog_irls_cells` fits the cloglog GLM from it and
`_wls_cells` the linear-probability model. The design is identical between them — only
the outcome convention (`not_supply` vs `supplier`) and the fitting differ.
"""
struct RegCells
    n_good_c :: Int
    n_ctrl_c :: Int
    n_rho    :: Int
    gcol     :: Vector{Int}
    gval     :: Vector{Float64}
    grp      :: Vector{Int}
    n_grp    :: Int
    has_z    :: Vector{Bool}
    lzc      :: Vector{Float64}
end

function _build_reg_cells(include_control::Bool, include_size_control::Bool,
                              N_rho_eff::Int, logz_const)
    n_ctrl_c = include_control ? N_CONTROL : 0
    n_cell   = n_good + n_ctrl_c

    gcol  = zeros(Int, n_cell)
    gval  = zeros(Float64, n_cell)
    raw_g = Vector{Int}(undef, n_cell)
    has_z = fill(false, n_cell)          # Vector{Bool}, matching the field type
    lzc   = zeros(Float64, n_cell)

    @inbounds for c in 1:n_cell
        is_good = c <= n_good
        s = is_good ? GOOD_S[c] : CONTROL_S[c - n_good]
        r = is_good ? GOOD_R[c] : CONTROL_R[c - n_good]
        dr = CLOSEST_DOWNSTREAM_REGION[r]
        raw_g[c] = (s - 1) * R_downstream + dr

        if N_REG == 1
            gcol[c] = 1
            gval[c] = LOG_CLOSEST_DIST[r]
        else
            b = DistBin[r, dr]
            if b > 0 && b <= N_REG
                gcol[c] = b
                gval[c] = 1.0
            end                       # else: base category, all-zero geography row
        end

        # Control cells have z ≡ −∞: they carry a size column of exactly 0, matching
        # the dense build (which leaves those entries at their zero initialisation).
        if include_size_control && is_good
            has_z[c] = true
            lzc[c]   = logz_const[c]
        end
    end

    # Compact the (sector × nearest-downstream) labels to 1:n_grp — most of the
    # S · R_downstream label space is empty, and the accumulators are dense in n_grp.
    gmax = maximum(raw_g)
    slot = zeros(Int, gmax)
    n_grp = 0
    grp = Vector{Int}(undef, n_cell)
    @inbounds for c in 1:n_cell
        rg = raw_g[c]
        if slot[rg] == 0
            n_grp += 1
            slot[rg] = n_grp
        end
        grp[c] = slot[rg]
    end

    return RegCells(n_good, n_ctrl_c, N_rho_eff, gcol, gval, grp, n_grp, has_z, lzc)
end

"""
    _cloglog_irls_cells(D, linkages, sample_weights, lzr, k, size_col; …) -> β

Cloglog IRLS over a `RegCells` design, allocating NOTHING of order
`n_cell × N_rho`. Mathematically identical to `_cloglog_irls` on the equivalent dense
design; it just never forms that design.

Two facts make this possible.

**(1) The weighted-within (FWL) transform has a closed form in accumulators.** With
`S^W_p = Σ_{i∈p} W_i`, `S^X_p = Σ_{i∈p} W_i X_i`, `S^z_p = Σ_{i∈p} W_i z_i`,

    X̃' W X̃ = Σ_i W_i X_i X_i'  −  Σ_p S^X_p (S^X_p)' / S^W_p
    X̃' W z̃ = Σ_i W_i X_i z_i   −  Σ_p S^X_p  S^z_p    / S^W_p

(expand `Σ_i W_i (X_i − m_p)(X_i − m_p)'` with `m_p = S^X_p / S^W_p`). So the `k × k`
normal equations are built in ONE streaming pass — the dense kernel's `copyto!(Xt, Xf)`
and `W .* Xt`, ~67 MB of churn PER IRLS ITERATION at production sizes (372 MB at
`N_RHO_INFERENCE`), are gone. The fixed effects come back from the same accumulators:
`a_p = (S^z_p − S^X_p·β) / S^W_p`, which is exactly the dense kernel's weighted group
mean of `z − Xβ`.

**(2) The linear predictor is a per-CELL vector.** Since `log z = lzc[c] + lzr[ρ,c]`
and only `lzc` moves with β,

    η[c,ρ] = ( β_geo·gval[c] + β_size·lzc[c] + a_{grp[c]} )  +  β_size·lzr[ρ,c]
           =            Acell[c]                             +  β_size·lzr[ρ,c]

so the IRLS state is a length-`n_cell` vector, not a length-`n_cell·N_rho` one. Every
per-row quantity (`η`, `μ`, `dμ/dη`, the working response, the IRLS weight) is formed
in registers inside the loop and never stored — eight full-length vectors in the dense
kernel, zero here. `η` is clamped where it is USED, which is what the dense kernel's
clamp-then-store does.

The first iteration is the exception: the initialiser `μ = (y + 0.5)/2` depends on the
outcome, so `η` there is one of two values keyed on `y` — computed on the fly from
`linkages`, again with nothing stored.

`lzr` may be `nothing` when no size control is requested. `sample_weights` is indexed
per row only when it is not flat; a `FlatWeights` is hoisted to a scalar.
"""
function _cloglog_irls_cells(D::RegCells, linkages, sample_weights::AbstractMatrix{Float64},
                             lzr::Union{Nothing, AbstractMatrix{Float64}},
                             k::Int, size_col::Int;
                             max_iter::Int=50, tol::Float64=1e-9, eta_clamp::Float64=30.0,
                             iters_out::Union{Nothing, Base.RefValue{Int}}=nothing)
    n_cell   = D.n_good_c + D.n_ctrl_c
    n_rho    = D.n_rho
    n_grp    = D.n_grp
    has_size = lzr !== nothing
    wflat    = flat_weight(sample_weights)
    w_ctrl   = 1.0 / n_rho

    # η at the IRLS initialiser μ = (y + 0.5)/2, for y = 0 and y = 1 respectively.
    eta0_0 = clamp(log(-log(1.0 - 0.25)), -eta_clamp, eta_clamp)
    eta0_1 = clamp(log(-log(1.0 - 0.75)), -eta_clamp, eta_clamp)

    Axx = zeros(k, k)          # Σ W X X'
    bxz = zeros(k)             # Σ W X z
    SX  = zeros(k, n_grp)      # per-group Σ W X
    SW  = zeros(n_grp)         # per-group Σ W
    Sz  = zeros(n_grp)         # per-group Σ W z
    a   = zeros(n_grp)         # fixed effects
    Acell = zeros(n_cell)      # cell-level part of η
    A   = zeros(k, k)
    rhs = zeros(k)

    β = zeros(k)
    first_pass = true
    n_it = 0

    for _ in 1:max_iter
        n_it += 1
        fill!(Axx, 0.0); fill!(bxz, 0.0); fill!(SX, 0.0)
        fill!(SW, 0.0);  fill!(Sz, 0.0)
        b_size = has_size ? β[size_col] : 0.0

        @inbounds for c in 1:n_cell
            is_good = c <= D.n_good_c
            hz  = has_size && D.has_z[c]
            p   = D.grp[c]
            gc  = D.gcol[c]
            gv  = D.gval[c]
            ac  = Acell[c]
            lc  = D.lzc[c]

            sw = 0.0; swz = 0.0; swl = 0.0; swll = 0.0; swlz = 0.0

            for rho in 1:n_rho
                # Outcome: not_supply. Control cells never supply.
                yi = is_good ? (linkages[rho, c] > 0 ? 0.0 : 1.0) : 1.0

                lzr_i = hz ? lzr[rho, c] : 0.0
                lz    = hz ? lc + lzr_i : 0.0

                η = first_pass ? (yi > 0.5 ? eta0_1 : eta0_0) :
                                 clamp(ac + b_size * lzr_i, -eta_clamp, eta_clamp)

                eη = exp(η)
                μi = clamp(1.0 - exp(-eη), 1e-12, 1.0 - 1e-12)
                di = eη * (1.0 - μi)                       # dμ/dη = e^η·e^{−e^η}
                zw = η + (yi - μi) / di                    # working response
                wi = (is_good ? (wflat === nothing ? sample_weights[rho, c] : wflat) : w_ctrl)
                Wi = wi * (di * di) / (μi * (1.0 - μi))    # IRLS × analytic weight

                sw  += Wi
                swz += Wi * zw
                if hz
                    swl  += Wi * lz
                    swll += Wi * lz * lz
                    swlz += Wi * lz * zw
                end
            end

            # Assemble this cell's contribution. A row carries at most ONE geography
            # column, so the dummy block picks up no off-diagonal terms.
            if gc > 0
                Axx[gc, gc] += sw * gv * gv
                bxz[gc]     += gv * swz
                SX[gc, p]   += gv * sw
                hz && (Axx[gc, size_col] += gv * swl)
            end
            if hz
                Axx[size_col, size_col] += swll
                bxz[size_col]           += swlz
                SX[size_col, p]         += swl
            end
            SW[p] += sw
            Sz[p] += swz
        end

        # Symmetrise, then apply the within-group correction.
        @inbounds for i in 1:k, j in 1:k
            A[i, j] = j >= i ? Axx[i, j] : Axx[j, i]
        end
        copyto!(rhs, bxz)
        @inbounds for p in 1:n_grp
            SW[p] < 1e-300 && continue          # matches the dense kernel: no demeaning
            inv_sw = 1.0 / SW[p]
            for i in 1:k
                sxi = SX[i, p]
                sxi == 0.0 && continue          # row i of the correction is all zeros
                for j in 1:k
                    A[i, j] -= sxi * SX[j, p] * inv_sw
                end
                rhs[i] -= sxi * Sz[p] * inv_sw
            end
        end

        β_new = _solve_within_normal_eq(A, rhs, "cloglog IRLS (streaming)")

        # Fixed effects: weighted group mean of (z − Xβ) — the dense kernel's `cr`.
        @inbounds for p in 1:n_grp
            if SW[p] < 1e-300
                a[p] = 0.0
            else
                acc = Sz[p]
                for i in 1:k
                    acc -= SX[i, p] * β_new[i]
                end
                a[p] = acc / SW[p]
            end
        end

        b_size_new = has_size ? β_new[size_col] : 0.0
        @inbounds for c in 1:n_cell
            gc = D.gcol[c]
            Acell[c] = (gc > 0 ? β_new[gc] * D.gval[c] : 0.0) +
                       ((has_size && D.has_z[c]) ? b_size_new * D.lzc[c] : 0.0) +
                       a[D.grp[c]]
        end

        Δ = maximum(abs.(β_new .- β))
        β = β_new
        first_pass = false
        Δ < tol && break
    end

    iters_out === nothing || (iters_out[] = n_it)
    return β
end


"""
    _wls_cells(D, linkages, sample_weights, lzr, k, size_col) -> β

Weighted least squares with one absorbed fixed effect, over a `RegCells` design — the
LPM (`:lpm`) counterpart of `_cloglog_irls_cells`, and the streaming replacement for
`fast_weighted_regression`'s materialised design.

Same closed form for the weighted-within (FWL) transform:

    X̃' W X̃ = Σ_i W_i X_i X_i'  −  Σ_p S^X_p (S^X_p)' / S^W_p
    X̃' W ỹ = Σ_i W_i X_i y_i   −  Σ_p S^X_p  S^y_p    / S^W_p

so the `k × k` normal equations are accumulated in ONE pass and nothing of order
`n_cell × N_rho` is ever allocated. There is no iteration here (the LPM is linear), so
this is a single sweep — strictly cheaper than the cloglog path.

The outcome is `supplier = 1{linkage}` — the LPM convention, and the OPPOSITE of the
cloglog kernel's `not_supply`; control-only cells contribute `y = 0`. Kept exactly as
`fast_weighted_regression` builds it so the two are comparable coefficient by
coefficient.

Note this solves the NORMAL equations where the dense sibling forms `sqrt(w) .* X` and
takes a pivoted-QR least squares. The two are algebraically identical and differ only
in conditioning: squaring the design squares its condition number, so on a
badly-scaled design the streaming path loses roughly twice as many digits. With 0/1
bin dummies and a log-z column that is not a live concern (the design is well scaled by
construction), but it IS the reason to keep the dense kernel as the reference and the
reason `REG_STREAMING[] = false` exists.
"""
function _wls_cells(D::RegCells, linkages, sample_weights::AbstractMatrix{Float64},
                    lzr::Union{Nothing, AbstractMatrix{Float64}},
                    k::Int, size_col::Int)
    n_cell   = D.n_good_c + D.n_ctrl_c
    n_rho    = D.n_rho
    n_grp    = D.n_grp
    has_size = lzr !== nothing
    wflat    = flat_weight(sample_weights)
    w_ctrl   = 1.0 / n_rho

    Axx = zeros(k, k)
    bxy = zeros(k)
    SX  = zeros(k, n_grp)
    SW  = zeros(n_grp)
    Sy  = zeros(n_grp)

    @inbounds for c in 1:n_cell
        is_good = c <= D.n_good_c
        hz  = has_size && D.has_z[c]
        p   = D.grp[c]
        gc  = D.gcol[c]
        gv  = D.gval[c]
        lc  = D.lzc[c]

        sw = 0.0; swy = 0.0; swl = 0.0; swll = 0.0; swly = 0.0

        for rho in 1:n_rho
            # Outcome: supplier (the LPM convention). Control cells never supply.
            yi = is_good ? (linkages[rho, c] > 0 ? 1.0 : 0.0) : 0.0
            lz = hz ? lc + lzr[rho, c] : 0.0
            Wi = is_good ? (wflat === nothing ? sample_weights[rho, c] : wflat) : w_ctrl

            sw  += Wi
            swy += Wi * yi
            if hz
                swl  += Wi * lz
                swll += Wi * lz * lz
                swly += Wi * lz * yi
            end
        end

        if gc > 0
            Axx[gc, gc] += sw * gv * gv
            bxy[gc]     += gv * swy
            SX[gc, p]   += gv * sw
            hz && (Axx[gc, size_col] += gv * swl)
        end
        if hz
            Axx[size_col, size_col] += swll
            bxy[size_col]           += swly
            SX[size_col, p]         += swl
        end
        SW[p] += sw
        Sy[p] += swy
    end

    A = zeros(k, k)
    @inbounds for i in 1:k, j in 1:k
        A[i, j] = j >= i ? Axx[i, j] : Axx[j, i]
    end
    rhs = copy(bxy)
    @inbounds for p in 1:n_grp
        SW[p] < 1e-300 && continue          # matches the dense kernel: no demeaning
        inv_sw = 1.0 / SW[p]
        for i in 1:k
            sxi = SX[i, p]
            sxi == 0.0 && continue
            for j in 1:k
                A[i, j] -= sxi * SX[j, p] * inv_sw
            end
            rhs[i] -= sxi * Sy[p] * inv_sw
        end
    end

    return _solve_within_normal_eq(A, rhs, "LPM weighted LS (streaming)")
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
function fast_cloglog_regression(linkages_flat, z_flat, sample_weights::AbstractMatrix{Float64};
                                 include_control::Bool=true,
                                 include_size_control::Bool=!include_control,
                                 rho_range::Union{Nothing, Vector{UnitRange{Int}}}=nothing,
                                 obs_weight::Union{Nothing, Vector{Float64}}=nothing,
                                 return_size_coef::Bool=false,
                                 max_iter::Int=50, tol::Float64=1e-9,
                                 iters_out::Union{Nothing,Base.RefValue{Int}}=nothing,
                                 u_draws::Union{Nothing, Matrix{Float64}}=nothing,
                                 logz_const::Union{Nothing, Vector{Float64}}=nothing,
                                 inv_theta::Union{Nothing, Float64}=nothing)
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

    # ── Streaming (cell-level) path — the production path ─────────────────────
    # Same fit, without ever materialising the N_valid × k design. Requires the
    # log-z decomposition (`u_draws` + `logz_const` from solve_network) when the size
    # control is on, and the plain per-cell row layout — the non-production
    # `rho_range`/`obs_weight` variants keep the dense kernel.
    if reg_streaming_ok(include_size_control, rho_range, obs_weight,
                            u_draws, logz_const)
        cells = _build_reg_cells(include_control, include_size_control,
                                     N_rho_eff, logz_const)
        lzr = include_size_control ?
              logz_resid(u_draws, inv_theta === nothing ? 1.0 / theta : inv_theta) :
              nothing
        β = _cloglog_irls_cells(cells, linkages_flat, sample_weights, lzr,
                                n_regressors, size_col;
                                max_iter=max_iter, tol=tol, iters_out=iters_out)
        return return_size_coef ?
            vcat(β[1:N_REG], include_size_control ? β[size_col] : NaN) :
            β[1:N_REG]
    end

    @assert z_flat !== nothing (
        "the dense cloglog path needs z_flat, but solve_network was called with " *
        "return_z=false. Either pass u_draws/logz_const so the streaming path applies, " *
        "or solve with return_z=true.")

    # ── Reuse the θ-invariant design ──────────────────────────────────────────
    # Only `y` (the not-supply indicator) and the log-z size column move with θ.
    # The distance columns are pure geography, `w` is a function of the frozen
    # draws, `fe_group` is geography, and the group → rows index derived from it is
    # the same on every call. Rebuilding all of that per evaluation dominated the
    # cost of the moment. The cache is keyed on the design SHAPE and on the IDENTITY
    # of the weight matrix (`===`), so a different draw set — the Jacobian and Σ_sim
    # replications pass their own — forces a rebuild rather than silently reusing a
    # stale design. Bypassed entirely for the non-production `rho_range`/`obs_weight`
    # variants.
    use_cache = rho_range === nothing && obs_weight === nothing
    key = (N_rho_eff, n_good, n_ctrl_eff, N_REG, n_size, N_valid)
    D   = use_cache ? _CLOGLOG_DESIGN[] : nothing
    if D === nothing || D.key != key || D.weights_ref !== sample_weights
        D = _build_cloglog_design(sample_weights, key, include_control,
                                  include_size_control, rows_of, obs_weight,
                                  N_rho_eff, n_ctrl_eff, N_valid, n_regressors,
                                  n_rows_goods, size_col)
        use_cache && (_CLOGLOG_DESIGN[] = D)
    end

    y = D.y; X = D.X
    @inbounds for i in 1:D.n_rows_goods
        g = D.row_g[i]; rho = D.row_rho[i]
        y[i] = linkages_flat[rho, g] > 0 ? 0.0 : 1.0     # not_supply
    end
    if include_size_control
        @inbounds for i in 1:D.n_rows_goods
            X[i, size_col] = log(z_flat[D.row_rho[i], D.row_g[i]])
        end
    end
    # Control rows keep y = 1 and a zero size column from the build: control firms
    # never supply and have no productivity draw.

    β = _cloglog_irls(y, X, D.w, D.fe_group; group_rows=D.group_rows,
                      max_iter=max_iter, tol=tol, iters_out=iters_out)
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

    # `u_draws`/`logz_const` let the fit rebuild the log-z regressor from the θ-invariant
    # decomposition instead of reading a materialised z (see `logz_resid`), which is what
    # allows the streaming kernel and lets `z_flat` be dropped upstream. Both links take
    # them, so both get the streaming path.
    reg_fun = REG_METHOD == :cloglog ? fast_cloglog_regression : fast_weighted_regression

    reg_all = reg_fun(linkages_flat, z_flat, sw;
                      include_control      = REG_INCLUDE_CONTROL,
                      include_size_control = REG_INCLUDE_SIZE,
                      return_size_coef     = REG_INCLUDE_SIZE,
                      u_draws              = network.u_draws,
                      logz_const           = network.logz_const,
                      inv_theta            = 1.0 / theta)
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
        # Win COUNTS, not a floored share: the count moment is now evaluated with the
        # unbiased `gbar_cell` (see its docstring), which needs k and m rather than q̂.
        # `q_hat` is kept as the raw diagnostic — unclamped, since nothing downstream
        # raises it to a power any more.
        k_counts = Vector{Int}(undef, n_good)
        q_hat    = Vector{Float64}(undef, n_good)
        @inbounds for g in 1:n_good
            acc = 0
            for rho in 1:n_draw
                acc += linkages_flat[rho, g] != 0
            end
            k_counts[g] = acc
            q_hat[g]    = acc / n_draw
        end
        N_hat_free, clamped = concentrate_N_s(k_counts, n_draw)
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
        # Same estimator the bisection matched on, so the moment reported to the loss
        # and the moment N̂_s was chosen against cannot drift apart.
        lg_G0 = gbar_logfact_table(n_draw)
        G0 = zeros(S)
        @inbounds for s in 1:S
            cells = CELLS_OF_SECTOR[s]
            isempty(cells) && continue
            G0[s] = gbar_sector(cells, k_counts, n_draw, N_hat[s], lg_G0)
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
             sample_weights::Union{Nothing, AbstractMatrix{Float64}}=nothing,
             N_fixed::Union{Nothing, Vector{Int}}=nothing)

    # The full-size `z_flat` (6.7 MB at N_rho = 723, n_good = 1161; 37 MB at
    # N_RHO_INFERENCE) is only needed by the DENSE regression path. Under the streaming
    # path — either link — it is never read, so it is neither returned NOR materialised;
    # `solve_network` then fills `z_inv_flat` column by column through a length-N_rho
    # scratch buffer, which is the same arithmetic on the same values. That removes two
    # full-size matrices from the peak of the evaluation, which is the regression.
    #
    # The predicate is the one the regressions themselves use, so the two cannot
    # disagree; the `u_draws === nothing` case is the legacy random-Fréchet branch, which
    # has no log-z decomposition and always needs the dense path.
    need_z = !REG_STREAMING[] || (REG_INCLUDE_SIZE && u_draws === nothing)

    # Solve network
    network = solve_network(params, return_firm_level=false, precomputed_tau=precomputed_tau,
                            u_draws=u_draws, sample_weights=sample_weights,
                            return_z=need_z)
    
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
                          sample_weights::Union{Nothing, AbstractMatrix{Float64}}=nothing)

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
            # A W that is neither already-restricted nor full-size means the caller's
            # moment_blocks and the moment set W was built over have drifted apart;
            # subsetting here would index a restricted W with global positions.
            size(W, 1) >= maximum(moment_indices) || error(
                "loss_function: W is $(size(W,1))×$(size(W,1)) but moment_indices has " *
                "$(length(moment_indices)) entries reaching index $(maximum(moment_indices)). " *
                "W is restricted to a DIFFERENT moment set than the one requested — check " *
                "that moment_blocks matches inference_moment_indices() (β → γ → G).")
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
                  sample_weights::Union{Nothing, AbstractMatrix{Float64}}=nothing,
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