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
    empirical_moments = [[agg_labor_share],agg_industry_share[2:end],emp_pi_r,reg_coef,emp_gamma_ls]
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
    unpack_params(params) -> (Ω^L, Ω^s, A, β, T)

Unpack parameter vector into model components (paper notation).

Parameter vector layout: [Ω^L(1) | Ω^s(S) | A(R_downstream) | β(N_TAU) | T(sum(T_MASK))]

Returns:
- Ω^L (Omega_L): Labor share in production [scalar]
- Ω^s (Omega_s): Sectoral input shares [S elements, normalized to sum to 1]
- A: Downstream firm productivity by region [R_downstream elements]
- β (beta): Trade cost parameters [N_TAU elements; 1 = power-law α, >1 = bin coefficients]
- T: Fréchet scale parameters [S × R elements, full vector with zeros for masked entries]
"""
function unpack_params(params)
    Omega_L = params[1]
    Omega_s = params[2:(1 + S)] / sum(params[2:(1 + S)])
    A = params[(S + 2):(S + R_downstream + 1)]
    A = A ./ A[1]
    beta = params[(S + R_downstream + 2):(S + R_downstream + 1 + N_TAU)]

    T_reduced = params[(S + R_downstream + 2 + N_TAU):end]
    # eltype(params) (not a hard-coded Float64) so ForwardDiff Duals survive the
    # scatter into T_full — byte-identical for Float64 params.
    T_full = zeros(eltype(params), R * S)
    T_full[T_MASK] = T_reduced
    T_mat = permutedims(reshape(T_full, R, S))   # s-major flat (R,S) → (S,R)

    for s in 1:S
        ref_r = T_REF_REGION[s]
        if ref_r > 0 && T_mat[s, ref_r] > 0
            T_mat[s, :] ./= T_mat[s, ref_r]
        end
    end

    return Omega_L, Omega_s, A, beta, vec(T_mat)
end


##################### log-T (φ) reparameterization #####################
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

Convert a full LEVEL parameter vector `[Ω^L | Ω^s | A | β | T(N_T_REDUCED)]` to/from
the full SEARCH vector `[Ω^L | Ω^s | A | β | φ(N_T_FREE)]`. Only the trailing T
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
    build_tau(beta) -> τ[r', r]

Build iceberg trade cost matrix from distance bin coefficients.

τ_{r'r} = 1 + β_b  where b = DistBin[r', r]

Returns matrix of size (R, R). Trade costs are identical across sectors.
"""
function build_tau(beta)
    # eltype(beta) keeps ForwardDiff Duals through the τ construction; identical
    # to ones(Float64, …) for the Float64 production path.
    tau = ones(eltype(beta), R, R_downstream)
    if N_TAU == 1
        # Power-law: τ_{r',r} = max(d, 1)^α = exp(α · log(max(d, 1)))
        for r_prime in 1:R, r_d in 1:R_downstream
            tau[r_prime, r_d] = exp(beta[1] * LOG_DIST_DOWNSTREAM[r_prime, r_d])
        end
    else
        for r_prime in 1:R, r_d in 1:R_downstream
            b = DistBin[r_prime, r_d]
            if b > 0 && b <= N_TAU
                tau[r_prime, r_d] += beta[b]
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
    Omega_L, Omega_s_vec, A_vec, beta, T_vec = unpack_params(params)

    # Build trade cost matrix τ_{r'r} — identical across sectors
    tau = precomputed_tau === nothing ? build_tau(beta) : precomputed_tau

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
function fast_weighted_regression(linkages_flat, z_flat, sample_weights::Matrix{Float64})

    n_regressors = N_REG + 1  # distance bins + log_productivity

    # Row count from the passed draws (not the global N_rho const), so callers
    # may pass a different number of varieties (e.g. the price-alignment test).
    N_rho_eff = size(sample_weights, 1)

    # All good pairs are valid (n_good entries, each with N_rho_eff varieties)
    N_valid = n_good * N_rho_eff

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

        for rho in 1:N_rho_eff
            idx += 1
            y[idx] = linkages_flat[rho, g] > 0 ? 1.0 : 0.0
            w[idx] = sample_weights[rho, g]
            fe_group[idx] = group_id

            if N_REG == 1
                X[idx, 1] = log_dist
            else
                if b > 0 && b <= N_REG
                    X[idx, b] = 1.0
                end
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

    return coefs[1:N_REG]
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
function compute_moments(network, params)

    Omega_L, Omega_s_vec, A_vec, beta, T_vec = unpack_params(params)

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
        agg_labor_share   = [agg_labor_share],
        agg_industry_share = vec(agg_industry_share),  # Full S elements (mask handles [2:end])
        pi_r              = pi_r,                       # block 3 — Full R_downstream (mask handles [2:end])
        reg_coef          = reg_coef,                   # block 4
        gamma_ls          = gamma_ls,                   # block 5 — Full (mask handles inactive/ref entries)
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
             sample_weights::Union{Nothing, Matrix{Float64}}=nothing)

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
        moments.pi_r,
        moments.reg_coef,
        moments.gamma_ls
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
    loss_function(simulated_moments, emp, W, method="original")

Compute loss between empirical and simulated moments.

# Methods
- "original": Raw squared difference
- "normalize": Difference scaled by sqrt of moment group size
- "hybrid": Percentage deviation for non-zero, absolute for zeros
"""
function loss_function(simulated_moments, emp, W, method="original";
                       moment_indices::Union{Nothing, Vector{Int}} = nothing)

    if method isa Bool
        method = method ? "normalize" : "original"
    end

    if moment_indices !== nothing && method == "log"
        error("moment_indices subsetting is not compatible with method=\"log\"")
    end

    square_size = sqrt.(vcat([fill(length(vec(m)), length(vec(m))) for m in simulated_moments]...))
    sim_flat = vcat([vec(simulated_moments[i]) for i in 1:length(simulated_moments)]...)

    sim_flat    = sim_flat[MOMENT_MASK]
    square_size = square_size[MOMENT_MASK]
    emp_flat    = vec(emp)

    if moment_indices !== nothing
        sim_flat    = sim_flat[moment_indices]
        square_size = square_size[moment_indices]
        emp_flat    = emp_flat[moment_indices]
    end

    N = length(sim_flat)

    if method == "original"
        err = reshape(emp_flat - sim_flat, (1, N))

    elseif method == "normalize"
        err = reshape((emp_flat - sim_flat) ./ square_size, (1, N))

    elseif method == "log"
        # Block boundaries in the masked moment vector:
        #   [1 : n_good]              labor share + industry shares + pi_r → log
        #   [n_good+1 : n_good+N_REG] reg_coef (negative, level deviation) → level
        #   [n_good+N_REG+1 : end]   gamma_ls                              → log
        eps = 1e-12
        err = zeros(N)

        # Log blocks: all strictly positive moments
        log_end   = n_good
        reg_start = n_good + 1
        reg_end   = n_good + N_REG
        pi_start  = n_good + N_REG + 1

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
    full_SMM(params, simulation=false, second_stage=false, method="original";
             analytical=false, n_quad=200, ...)

Full SMM evaluation: compute loss and return moments.

When `analytical=true`, uses compute_moments_analytical (closed-form EK formulas
+ Gauss-Legendre quadrature for reg_coef). The simulation/u_draws/sample_weights
kwargs are ignored in analytical mode.
"""
function full_SMM(params, simulation=false, second_stage=false, method="original";
                  precomputed_tau::Union{Nothing, Matrix{Float64}}=nothing,
                  u_draws::Union{Nothing, Matrix{Float64}}=nothing,
                  sample_weights::Union{Nothing, Matrix{Float64}}=nothing,
                  W_override::Union{Nothing, AbstractMatrix}=nothing,
                  moment_blocks::Union{Nothing, Vector{Int}}=nothing,
                  analytical::Bool=false,
                  n_quad::Int=200)

    if analytical
        moms_nt = compute_moments_analytical(params; n_quad=n_quad)
        simulated_moments = (moms_nt.agg_labor_share,
                             moms_nt.agg_industry_share,
                             moms_nt.pi_r,
                             moms_nt.reg_coef,
                             moms_nt.gamma_ls)
    else
        simulated_moments = SMM(params, simulation; precomputed_tau=precomputed_tau,
                                u_draws=u_draws, sample_weights=sample_weights)
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
        return loss_function(moments, emp, W, method; moment_indices=moment_indices), simulated_moments
    end
end