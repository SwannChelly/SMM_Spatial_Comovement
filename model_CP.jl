##### Spatial Comovement — simulated model (main specification) #####
#
# Notation (paper, Appendix B): r downstream buyer, l (or r') upstream cell, s sector,
# ρ variety. ε demand elasticity, χ^L labour share, χ^s sector shares, λ labour/inputs
# elasticity, ν across sectors, ν_s across varieties, θ Fréchet shape, T_ls Fréchet
# scale, A_r downstream productivity, τ = d^α (or 1 + α_b by distance bin).
#
# T_ls = T_{a(l)s} · n_ls^κ: comparative advantage at the attraction-area level a(l),
# scaled by the cell's firm count n_ls. N_s varieties per sector, profiled out of the
# loss by an integer bisection on the count moment Ḡ_s(0).

using Distributed
using Distributions
using Random
using NPZ
using LinearAlgebra
using Printf
using QuasiMonteCarlo

##################### Draws ###################

"""
    FlatWeights(w, n, m) <: AbstractMatrix{Float64}

An `n × m` matrix every entry of which is `w`, stored in three words. Both samplers
return flat weights `1/N_rho`; flat weights make `solve_network`'s winner-weight
shortcut exact, hence the moments unbiased.
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
    sobol_scrambled_net(N, d, rng) -> Matrix (N × d)

A `d`-dimensional Sobol net randomised by a base-2 digital shift drawn from `rng`
(reproducible under a frozen seed, no global reseed; the shift preserves the net and
removes the all-zeros origin).
"""
function sobol_scrambled_net(N::Int, d::Int, rng::AbstractRNG)
    pts   = QuasiMonteCarlo.sample(N, zeros(d), ones(d), SobolSample())
    U     = Matrix{Float64}(undef, N, d)
    nbits = 52
    scale = Float64(2)^nbits
    mask  = (UInt64(1) << nbits) - UInt64(1)
    @inbounds for j in 1:d
        shift = rand(rng, UInt64) & mask
        for i in 1:N
            m       = floor(UInt64, pts[j, i] * scale) & mask
            U[i, j] = (m ⊻ shift) / scale
        end
    end
    return U
end

"""
    sobol_dim_shifted(N, j, rng) -> Vector (N)

Dimension `j` of an `N`-point Sobol net, randomised by its own digital shift from `rng`.
"""
function sobol_dim_shifted(N::Int, j::Int, rng::AbstractRNG)
    pts   = QuasiMonteCarlo.sample(N, zeros(j), ones(j), SobolSample())
    nbits = 52
    scale = Float64(2)^nbits
    mask  = (UInt64(1) << nbits) - UInt64(1)
    shift = rand(rng, UInt64) & mask
    return [((floor(UInt64, pts[j, i] * scale) & mask) ⊻ shift) / scale for i in 1:N]
end

"""
    generate_draws(N_rho, n_good, method; randomise=false, rng=Random.GLOBAL_RNG) -> (U, W)

`U` is `N_rho × (n_good + S)`: columns `1:n_good` are the cells, column `n_good + s` the
foreign competitor of sector s (one cost per variety, common to every buyer; 0.5 where
the sector does not import). `W` covers the cells only (`N_rho × n_good`).

  * `:sobol` — one scrambled Sobol net per SECTOR (the Ricardian min lives within a
    sector, across its cells). `randomise=false` freezes the scramble (seed 42), so the
    criterion is a deterministic function of θ. Used for the optimisation draws. The
    foreign draw is the next Sobol dimension of the sector's net (d + 1 for d cells), so
    it is stratified jointly with the cells; its shift comes from a stream derived from
    the master without consuming it, so the cell columns — and whatever the caller draws
    from `rng` afterwards — are those of the domestic-only design.
  * `:mc` — i.i.d. uniforms (`randomise=false` ⇒ `MersenneTwister(0)`), the foreign
    columns drawn after the cells. Used for inference and the post-hoc replications.
"""
function generate_draws(N_rho::Int, n_good::Int, method::Symbol;
                        randomise::Bool=false, rng::AbstractRNG=Random.GLOBAL_RNG)
    W = FlatWeights(1.0 / N_rho, N_rho, n_good)
    lo, hi = eps(), 1.0 - eps()
    if method === :sobol
        U      = fill(0.5, N_rho, n_good + S)
        master = randomise ? rng : MersenneTwister(42)
        # One net per sector, because a variety's supplier is the min over the cells of its own
        # sector only: competition never crosses sectors, so the sectors are separate integrals.
        # The nets are a numerical integration device, not an assumption about links between
        # sectors. `master` is drawn from sector after sector, so each sector gets its own shift.
        @inbounds for g_indices in SECTOR_GOOD_INDICES
            isempty(g_indices) && continue
            net = sobol_scrambled_net(N_rho, length(g_indices), master)
            for (k, g) in enumerate(g_indices), rho in 1:N_rho
                U[rho, g] = clamp(net[rho, k], lo, hi)
            end
        end
        frng = MersenneTwister(hash((:foreign_draws, rand(copy(master), UInt64))))
        for s in 1:S
            (FOREIGN_ACTIVE[s] && !isempty(SECTOR_GOOD_INDICES[s])) || continue
            uF = sobol_dim_shifted(N_rho, length(SECTOR_GOOD_INDICES[s]) + 1, frng)
            @inbounds for rho in 1:N_rho
                U[rho, n_good + s] = clamp(uF[rho], lo, hi)
            end
        end
        return U, W
    elseif method === :mc
        r = randomise ? rng : MersenneTwister(0)
        U = rand(r, N_rho, n_good)
        return hcat(U, clamp.(rand(r, N_rho, S), lo, hi)), W
    end
    error("Unknown draw method :$method (choose :sobol or :mc)")
end

##################### Parameters ###################

t_param_start() = S + R_downstream + 3 + N_TAU          # first T index
alpha_range()   = (S + R_downstream + 2):(S + R_downstream + 1 + N_TAU)
kappa_index()   = S + R_downstream + 2 + N_TAU
unpack_kappa(params) = params[kappa_index()]

"""
    unpack_T_par(params) -> Matrix (S × T_COL_DIM)

The area-level comparative-advantage matrix: the reduced T block scattered s-major
over `T_MASK`, reference-normalised per sector (`T[s, ref] = 1`).
"""
function unpack_T_par(params)
    T_full = zeros(eltype(params), T_COL_DIM * S)
    T_full[T_MASK] = params[t_param_start():end]
    T_par = permutedims(reshape(T_full, T_COL_DIM, S))
    for s in 1:S
        ref_c = T_REF_REGION[s]
        (ref_c > 0 && T_par[s, ref_c] > 0) && (T_par[s, :] ./= T_par[s, ref_c])
    end
    return T_par
end

"""
    gather_T_to_ze(T_par, kappa) -> Matrix (S × R)

`T[s, l] = T_par[s, a(l)] · n_ls^κ`: every ZE of an area shares the Fréchet scale, so
each ZE still runs its own Ricardian competition.
"""
function gather_T_to_ze(T_par::AbstractMatrix, kappa)
    T_mat = Matrix{promote_type(eltype(T_par), typeof(kappa))}(undef, S, R)
    @inbounds for l in 1:R, s in 1:S
        T_mat[s, l] = T_par[s, T_GATHER[l]] * exp(kappa * LOG_N_LS[s, l])
    end
    return T_mat
end

"""
    foreign_phi(T_par) -> Vector (S):  Φ^F_s = T_par[s, FOREIGN_COL], zero where s does not import
"""
foreign_phi(T_par::AbstractMatrix) =
    [FOREIGN_ACTIVE[s] ? T_par[s, FOREIGN_COL] : zero(eltype(T_par)) for s in 1:S]

"""
    unpack_params(params) -> (χ^L, χ^s, A, α, vec(T), Φ^F)

χ^s is normalised to sum to one, A to A[1] = 1, T is gathered to the (S, R) ZE level and
Φ^F (S) is the foreign Fréchet scale, in the same reference-normalised units as T.
"""
function unpack_params(params)
    chi_L = params[1]
    chi_s = params[2:(1 + S)] / sum(params[2:(1 + S)])
    A = params[(S + 2):(S + R_downstream + 1)]
    A = A ./ A[1]
    alpha = params[alpha_range()]
    T_par = unpack_T_par(params)
    T_mat = gather_T_to_ze(T_par, unpack_kappa(params))
    return chi_L, chi_s, A, alpha, vec(T_mat), foreign_phi(T_par)
end

"""
    aggregate_gamma_to_T(gamma_ze; foreign = nothing) -> Matrix (T_COL_DIM × S)

Area aggregate `γ_{s,a} = Σ_{l ∈ a} γ_{ls}` over every simulated cell of the area; the
foreign row `FOREIGN_COL` is `foreign` (S) when given, zero otherwise.
"""
function aggregate_gamma_to_T(gamma_ze::AbstractMatrix; foreign = nothing)
    out = zeros(eltype(gamma_ze), T_COL_DIM, S)
    @inbounds for g in 1:n_good
        out[T_GATHER[GOOD_R[g]], GOOD_S[g]] += gamma_ze[GOOD_R[g], GOOD_S[g]]
    end
    foreign === nothing || (out[FOREIGN_COL, :] .= foreign)
    return out
end

"""
    gamma_T_from_flows(X_ls, X_F) -> Matrix (T_COL_DIM × S)

Shares of each sector's TOTAL purchases: area aggregates of the domestic flows `X_ls`
(R × S) and the foreign flow `X_F` (S) on the foreign row, divided by `Σ_l X_ls + X_F`.
They sum to one per sector, as the data γ̂ with f̂ on the foreign row.
"""
function gamma_T_from_flows(X_ls::AbstractMatrix, X_F::AbstractVector)
    X_tot = vec(sum(X_ls; dims = 1)) .+ X_F
    g = zeros(promote_type(eltype(X_ls), eltype(X_F)), R, S)
    f = zeros(eltype(g), S)
    @inbounds for s in 1:S
        X_tot[s] > 1e-300 || continue
        for r in 1:R
            g[r, s] = X_ls[r, s] / X_tot[s]
        end
        f[s] = X_F[s] / X_tot[s]
    end
    return aggregate_gamma_to_T(g; foreign = f)
end

"""
    build_tau(alpha) -> τ (R × R_downstream)

`τ = max(d,1)^α` with one α. With `N_TAU = K` bins, `ln τ` is continuous and piecewise
linear in `ln d`, with knots at the bin centres `TAU_LOG_KNOTS = ln d̄_0 … ln d̄_K`:
`α_k` is the distance elasticity of the trade cost on `[d̄_{k-1}, d̄_k]`, `ln τ(d̄_0) = 0`,
and the end segments extend below `d̄_0` (slope α_1) and above `d̄_K` (slope α_K). With
every `α_k = α`, `τ = (d/d̄_0)^α`, i.e. `d^α` up to a factor common to all origins of a
buyer, which cancels in the shares: the power law is nested exactly.
"""
function build_tau(alpha)
    tau = ones(eltype(alpha), R, R_downstream)
    x = TAU_LOG_KNOTS
    for r_prime in 1:R, r_d in 1:R_downstream
        ld = LOG_DIST_DOWNSTREAM[r_prime, r_d]
        if N_TAU == 1
            tau[r_prime, r_d] = exp(alpha[1] * ld)
        else
            lnt = alpha[1] * (min(ld, x[2]) - x[1])
            for k in 2:N_TAU
                ld > x[k] || break
                lnt += alpha[k] * ((k == N_TAU ? ld : min(ld, x[k + 1])) - x[k])
            end
            tau[r_prime, r_d] = exp(lnt)
        end
    end
    return tau
end

##################### Count moment and the variety count ###################

"""
    gbar_logfact_table(m) -> Vector  (lg[i+1] = log i!, i = 0..m)
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

UNBIASED estimator of `(1 − q)^n` from `k` wins out of `m` i.i.d. draws:
`E[C(m−k, n)/C(m, n)] = (1 − q)^n` — draw `n` of the `m` varieties without replacement
and ask whether all of them lost.
"""
@inline function gbar_cell(k::Integer, m::Integer, n::Integer, lg::Vector{Float64})
    n <= 0 && return 1.0
    n > m - k && return 0.0
    return exp(lg[m-k+1] - lg[m-k-n+1] - lg[m+1] + lg[m-n+1])
end

"""
    gbar_sector(cells, k_counts, m, n, lg) -> Ḡ_s(n) = mean_l (1 − q_l)^n
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

"""K values at which the supplier-count distribution is tabulated (K = 0 is targeted)."""
const COUNT_CURVE_K = 0:3

"""
    gbar_cell_K(k, m, n, j, lg) -> Float64

UNBIASED estimator of `Pr[K_ls = j] = C(n,j) q^j (1−q)^(n−j)` (hypergeometric argument
of `gbar_cell`; at `j = 0` it is `gbar_cell`).
"""
@inline function gbar_cell_K(k::Integer, m::Integer, n::Integer, j::Integer,
                             lg::Vector{Float64})
    n <= 0 && return j == 0 ? 1.0 : 0.0
    (j < 0 || j > n || j > k) && return 0.0
    n - j > m - k && return 0.0
    return exp((lg[k+1]     - lg[j+1]   - lg[k-j+1]) +
               (lg[m-k+1]   - lg[n-j+1] - lg[m-k-n+j+1]) -
               (lg[m+1]     - lg[n+1]   - lg[m-n+1]))
end

function gbar_sector_K(cells, k_counts::AbstractVector{<:Integer}, m::Integer,
                       n::Integer, j::Integer, lg::Vector{Float64})
    isempty(cells) && return j == 0 ? 1.0 : 0.0
    acc = 0.0
    @inbounds for g in cells
        acc += gbar_cell_K(k_counts[g], m, n, j, lg)
    end
    return acc / length(cells)
end

"""
    french_variety_share(linkages) -> q (S):  share of the draws won by a French cell for
    at least one buyer (1 in a sector without foreign competitor)
"""
function french_variety_share(linkages::AbstractMatrix)
    m = size(linkages, 1)
    q = ones(S)
    for s in 1:S
        (FOREIGN_ACTIVE[s] && !isempty(CELLS_OF_SECTOR[s])) || continue
        n = 0
        @inbounds for rho in 1:m
            any(g -> linkages[rho, g] != 0, CELLS_OF_SECTOR[s]) && (n += 1)
        end
        q[s] = n / m
    end
    return q
end

"""
    n_hi_bound(s, m, q) -> the upper bound of the bisection for sector s

`N^obs_s / q_s` (the varieties won by France are at most the observed suppliers),
capped by `N_HI_CAP[s]` and the draw count `m`; `N_HI[s]` when `q === nothing` or q_s = 1.
"""
function n_hi_bound(s::Int, m::Integer, q)
    (q === nothing || q[s] >= 1) && return min(N_HI[s], m)
    hi = q[s] > 0 ? ceil(Int, N_HI[s] / q[s] - 1e-9) : N_HI_CAP[s]
    return max(min(hi, N_HI_CAP[s], m), min(N_HI[s], m))
end

"""
    n_lo_bound(s, q) -> the lower bound of the bisection for sector s

Each variety won by France has at most one supplier per buyer region, so the observed
suppliers satisfy `N^obs_s ≤ R_d · N_s q_s`, i.e. `N_s ≥ N^obs_s / (R_d q_s)`. Without
imports (q_s = 1) this is `N_LO[s] = ⌈N^obs_s / R_d⌉`; never below `N_LO[s]`, and
`N_LO[s] = 1` under `RELAX_N_LO`.
"""
function n_lo_bound(s::Int, q)
    (RELAX_N_LO || q === nothing || q[s] >= 1 || q[s] <= 0) && return N_LO[s]
    return max(N_LO[s], ceil(Int, N_HI[s] / (R_downstream * q[s]) - 1e-9))
end

"""
    concentrate_N_s(k_counts, m; target = G_TARGET, q = nothing) -> (N_hat, clamped)

`target` is the Ḡ_s(0) matched: the data by default, a fake world's own realised
empty-cell share in inference.jl. `q` is `french_variety_share` of the same draws: with
imports the bounds are `N^obs_s / (R_d q_s)` (`n_lo_bound`) and `N^obs_s / q_s`
(`n_hi_bound`), `⌈N^obs_s / R_d⌉` and `N^obs_s` without.

Monotone integer bisection on `Ḡ_s(n)` over `[n_lo_bound, n_hi_bound]` matching `G_TARGET[s]`.
`k_counts[g]` is the number of the `m` draws cell `g` wins somewhere downstream (free of
`N_s`, Lemma 2). `clamped[s] ∈ (:none, :lo, :hi)` records a binding bound — a rejection
signal for the mechanism, reported per sector.
"""
function concentrate_N_s(k_counts::AbstractVector{<:Integer}, m::Integer;
                         target::AbstractVector{<:Real} = G_TARGET, q = nothing)
    N_hat   = zeros(Int, S)
    clamped = fill(:none, S)
    lg      = gbar_logfact_table(m)
    @inbounds for s in 1:S
        cells = CELLS_OF_SECTOR[s]
        hi0 = n_hi_bound(s, m, q)
        lo0 = min(n_lo_bound(s, q), hi0)
        if isempty(cells)
            N_hat[s] = lo0
            continue
        end
        Gs = n -> gbar_sector(cells, k_counts, m, n, lg)
        tgt = target[s]
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
        @assert N_hat[s] <= N_rho "N̂[$s] = $(N_hat[s]) exceeds N_rho = $N_rho"
    end
    return N_hat, clamped
end

##################### Network solution ###################

"""
    solve_network(params; u_draws, sample_weights, precomputed_tau=nothing,
                  return_firm_level=false)

Ricardian selection (lowest-cost cell wins each variety for each buyer), nested CES
price indices, downstream sales and trade flows, on the passed draws (`N_rho_eff` =
their row count). `return_firm_level` adds the per-(variety, buyer) expenditure shares
(COO vectors) and the productivity matrix `z_flat`.

The foreign competitor of sector s bids `p^F_ρs = (−log(1 − u^F_ρs)/Φ^F_s)^{1/θ}` for
variety ρ, the same for every buyer (`u_draws[:, n_good + s]`); it wins where it is
strictly cheaper than the best domestic cell. Its varieties enter the price indices and
the spending (`X_F_flat`, `X_F_by_r`) but create no domestic linkage, so the extensive
margin, the win counts and the firm-level rows are French only.
"""
function solve_network(params; u_draws::Matrix{Float64},
                       sample_weights::AbstractMatrix{Float64},
                       precomputed_tau::Union{Nothing, Matrix{Float64}}=nothing,
                       return_firm_level::Bool=false)
    chi_L, chi_s_vec, A_r, alpha, T_vec, PhiF = unpack_params(params)
    tau = precomputed_tau === nothing ? build_tau(alpha) : precomputed_tau
    T   = reshape(T_vec, S, R)
    nu_s_mat = reshape(nu_s, 1, S)
    N_rho_eff = size(u_draws, 1)
    @assert size(sample_weights, 1) == N_rho_eff "sample_weights rows must match u_draws rows"
    any(>(0), PhiF) && @assert(size(u_draws, 2) >= n_good + S,
                               "u_draws needs n_good + S columns (foreign draws); use generate_draws")

    # Foreign bid per (variety, sector); Inf where the sector does not import.
    p_F = fill(Inf, N_rho_eff, S)
    for s in 1:S
        PhiF[s] > 0 || continue
        sc = PhiF[s]^(-1 / theta)
        @inbounds for rho in 1:N_rho_eff
            p_F[rho, s] = sc * (-log(1.0 - u_draws[rho, n_good + s]))^(1 / theta)
        end
    end
    X_F_by_r = zeros(S, R_downstream)

    # z = T^{1/θ} (−log(1−u))^{−1/θ}, built one column at a time; only 1/z drives the argmin.
    z_inv_flat = Matrix{Float64}(undef, N_rho_eff, n_good)
    z_flat     = return_firm_level ? Matrix{Float64}(undef, N_rho_eff, n_good) : nothing
    zbuf = Vector{Float64}(undef, N_rho_eff)
    for g in 1:n_good
        scale = max(T[GOOD_S[g], GOOD_R[g]], eps(Float64))^(1/theta)
        for rho in 1:N_rho_eff
            zbuf[rho] = scale * (-log(1.0 - u_draws[rho, g]))^(-1.0/theta)
        end
        @views z_inv_flat[:, g] .= zbuf .^ (-1)
        return_firm_level && (@views z_flat[:, g] .= zbuf)
    end

    X_shares_by_r   = zeros(n_good, R_downstream)
    c_tilde_r       = zeros(R_downstream)
    linkages_flat   = falses(N_rho_eff, n_good)
    p_rho_s         = zeros(N_rho_eff, S)
    winner_good_idx = zeros(Int, N_rho_eff, S)
    w_rho_s         = Matrix{Float64}(undef, N_rho_eff, S)
    ces_buf         = Matrix{Float64}(undef, N_rho_eff, S)
    tau_c::Matrix{Float64}   = tau
    z_inv_c::Matrix{Float64} = z_inv_flat

    if return_firm_level
        firm_exp_rho = Int[]; firm_exp_s = Int[]; firm_exp_g = Int[]; firm_exp_r = Int[]
        firm_exp_val = Float64[]; firm_deriv_val = Float64[]
    end

    for r_d in 1:R_downstream
        r = DOWNSTREAM_REGIONS[r_d]

        # Lowest delivered cost p = w τ / z per (variety, sector); ties keep the first
        # minimiser and a NaN wins outright (as `argmin`).
        for s in 1:S
            g_indices = SECTOR_GOOD_INDICES[s]
            isempty(g_indices) && continue
            tv::Vector{Float64} = tau_c[SECTOR_GOOD_REGIONS[s], r_d]
            wv::Vector{Float64} = W_RS_FLAT[g_indices]
            n_act = length(g_indices)
            @inbounds for rho in 1:N_rho_eff
                best = z_inv_c[rho, g_indices[1]] * tv[1] * wv[1]
                bidx = 1
                if !isnan(best)
                    for li in 2:n_act
                        p = z_inv_c[rho, g_indices[li]] * tv[li] * wv[li]
                        if isnan(p)
                            best = p; bidx = li; break
                        elseif p < best
                            best = p; bidx = li
                        end
                    end
                end
                # Foreign bid strictly below the best domestic cell (a NaN domestic price
                # keeps winning, as above): the variety is imported (winner = −1).
                if p_F[rho, s] < best
                    winner_good_idx[rho, s] = -1
                    p_rho_s[rho, s] = p_F[rho, s]
                else
                    winner_good_idx[rho, s] = g_indices[bidx]
                    p_rho_s[rho, s] = best
                end
            end
        end

        # P_sr = [Σ_ρ w_ρ p_ρ^{1−ν_s}]^{1/(1−ν_s)}, P_r = [Σ_s χ^s P_sr^{1−ν}]^{1/(1−ν)},
        # c_r = [χ^L w_r^{1−λ} + (1−χ^L) P_r^{1−λ}]^{1/(1−λ)}.
        for s in 1:S
            nu_s_s = nu_s[s]
            for rho in 1:N_rho_eff
                g_w = winner_good_idx[rho, s]
                # A variety's weight is the sector's (constant across its cells), so an
                # imported variety reads it on the sector's first cell.
                w_rho_s[rho, s] = g_w == 0 ? 0.0 :
                                  sample_weights[rho, g_w > 0 ? g_w : SECTOR_GOOD_INDICES[s][1]]
                ces_buf[rho, s] = w_rho_s[rho, s] * p_rho_s[rho, s]^(1 - nu_s_s)
            end
        end
        P_sr = sum(ces_buf, dims=1).^(1 ./ (1 .- nu_s_mat))
        P_r = sum(P_sr.^(1 - nu) .* reshape(chi_s_vec, 1, S))^(1 / (1 - nu))
        c_r = (chi_L * regional_wages[r]^(1-lambda) +
               (1-chi_L) * P_r^(1-lambda))^(1/(1-lambda))
        c_tilde_r[r_d] = c_r / A_r[r_d]

        # Expenditure share of each won variety in the buyer's unit cost.
        labor_substitution_factor = (1 - chi_L) * (P_r / c_r)^(1 - lambda)
        for s in 1:S
            P_sr_s = P_sr[s]
            nu_s_s = nu_s[s]
            for rho in 1:N_rho_eff
                g_winner = winner_good_idx[rho, s]
                g_winner == 0 && continue
                if g_winner < 0
                    X_F_by_r[s, r_d] += w_rho_s[rho, s] * chi_s_vec[s] * (1-chi_L) *
                        (p_rho_s[rho, s] / P_sr_s)^(1 - nu_s_s) *
                        (P_sr_s / P_r)^(1 - nu) *
                        (P_r / c_r)^(1 - lambda)
                    continue
                end
                linkages_flat[rho, g_winner] = true
                exp_val = sample_weights[rho, g_winner] * chi_s_vec[s] * (1-chi_L) *
                    (p_rho_s[rho, s] / P_sr_s)^(1 - nu_s_s) *
                    (P_sr_s / P_r)^(1 - nu) *
                    (P_r / c_r)^(1 - lambda)
                X_shares_by_r[g_winner, r_d] += exp_val
                if return_firm_level
                    push!(firm_exp_rho, rho); push!(firm_exp_s, s); push!(firm_exp_g, g_winner)
                    push!(firm_exp_r, r); push!(firm_exp_val, exp_val)
                    push!(firm_deriv_val, exp_val / labor_substitution_factor)
                end
            end
        end
    end

    # Downstream: p_r = c̃_r/μ, P = [Σ_r p_r^ε δ_r]^{1/ε}, Y_r = p_r^ε P^{−ε} δ_r (E = 1).
    mu  = epsilon/(epsilon - 1)
    p_r = c_tilde_r ./ mu
    P   = sum(p_r.^epsilon .* delta_r[DOWNSTREAM_REGIONS])^(1/epsilon)
    Y_r = p_r.^epsilon .* P^(-epsilon) .* 1.0 .* delta_r[DOWNSTREAM_REGIONS]

    X_ls_flat = zeros(n_good)
    X_F_flat  = zeros(S)
    for r_d in 1:R_downstream
        total_cost_r = mu * Y_r[r_d]
        for g in 1:n_good
            X_ls_flat[g] += X_shares_by_r[g, r_d] * total_cost_r
        end
        for s in 1:S
            X_F_flat[s] += X_F_by_r[s, r_d] * total_cost_r
        end
    end
    c_tilde_r_full = zeros(R); c_tilde_r_full[DOWNSTREAM_REGIONS] = c_tilde_r
    Y_r_full       = zeros(R); Y_r_full[DOWNSTREAM_REGIONS]       = Y_r

    # X_F_by_r: foreign spending per unit of the buyer's cost (S × R_downstream), the
    # imported part of the intermediate share; X_F_flat: in the units of X_ls_flat.
    base = (X_ls_flat = X_ls_flat, X_F_flat = X_F_flat, X_F_by_r = X_F_by_r,
            c_tilde_r = c_tilde_r_full, Y_r = Y_r_full,
            linkages_flat = linkages_flat, sample_weights = sample_weights)
    return_firm_level || return base
    return merge(base, (z_flat = z_flat, firm_exp_rho = firm_exp_rho, firm_exp_s = firm_exp_s,
                        firm_exp_g = firm_exp_g, firm_exp_r = firm_exp_r,
                        firm_exp_val = firm_exp_val, firm_deriv_val = firm_deriv_val,
                        mu = mu, P = P))
end

##################### Block 4: the cell-level cloglog ###################
#
# Data: one cloglog per (sector, ZE) cell of the regression set 𝒜, has_sup on the
# distance regressor(s) and ln n_ls, sector × AA fixed effect, groups without variation
# dropped. Model: the same statistic averaged over R_CELL_REG networks, network j being
# the first N̂_s entries of a fixed random permutation of the m simulated varieties
# (nested in N), y_ls = 1{cell l wins one of them for at least one buyer} — exactly
# `linkages_flat`. Averaging the statistic (not the statistic of the average) matches a
# data moment computed on ONE network.

"""
    build_fe_group_rows(fe_group) -> Vector{Vector{Int}}  (row indices per group)
"""
function build_fe_group_rows(fe_group::AbstractVector{<:Integer})
    isempty(fe_group) && return Vector{Int}[]
    gmax   = maximum(fe_group)
    counts = zeros(Int, gmax)
    @inbounds for g in fe_group
        counts[g] += 1
    end
    out  = Vector{Vector{Int}}()
    slot = zeros(Int, gmax)
    for g in 1:gmax
        counts[g] > 0 || continue
        push!(out, Vector{Int}(undef, counts[g])); slot[g] = length(out)
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
    _cloglog_irls(y, X, w, fe_group; max_iter=50, tol=1e-9, eta_clamp=30.0, iters_out) -> β

Complementary-log-log GLM `P(y=1) = 1 − exp(−exp(η))`, `η = Xβ + a_fe`, by IRLS with the
single fixed effect absorbed by weighted within-group demeaning (exact for one FE).
"""
function _cloglog_irls(y::AbstractVector{<:Real}, X::AbstractMatrix{<:Real},
                       w::AbstractVector{<:Real}, fe_group::AbstractVector{<:Integer};
                       max_iter::Int=50, tol::Float64=1e-9, eta_clamp::Float64=30.0,
                       iters_out::Union{Nothing,Base.RefValue{Int}}=nothing)
    n, k = size(X)
    yf = Float64.(y); Xf = Matrix{Float64}(X); wf = Float64.(w)
    grows = build_fe_group_rows(fe_group)

    mu = (yf .+ 0.5) ./ 2
    eta = clamp.(log.(.-log.(1 .- mu)), -eta_clamp, eta_clamp)
    zwork = Vector{Float64}(undef, n); W = Vector{Float64}(undef, n)
    zt = Vector{Float64}(undef, n);    Xt = Matrix{Float64}(undef, n, k)
    Xb = Vector{Float64}(undef, n);    eta_new = Vector{Float64}(undef, n)
    sw_g = Vector{Float64}(undef, length(grows))

    beta = zeros(k)
    n_it = 0
    for _ in 1:max_iter
        n_it += 1
        @inbounds for i in 1:n
            mu_i       = clamp(1 - exp(-exp(eta[i])), 1e-12, 1 - 1e-12)
            di       = exp(eta[i]) * (1 - mu_i)
            zwork[i] = eta[i] + (yf[i] - mu_i) / di
            W[i]     = wf[i] * (di^2) / (mu_i * (1 - mu_i))
            zt[i]    = zwork[i]
        end
        copyto!(Xt, Xf)
        @inbounds for (p, r) in enumerate(grows)
            sw = sum(@view W[r]); sw_g[p] = sw
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
        WXt   = W .* Xt
        beta_new = (Xt' * WXt) \ (Xt' * (W .* zt))
        mul!(Xb, Xf, beta_new)
        @inbounds for i in 1:n
            eta_new[i] = Xb[i]
        end
        @inbounds for (p, r) in enumerate(grows)
            sw = sw_g[p]
            sw < 1e-300 && continue
            cr = sum(i -> W[i] * (zwork[i] - Xb[i]), r) / sw
            for i in r
                eta_new[i] += cr
            end
        end
        @inbounds for i in 1:n
            eta_new[i] = clamp(eta_new[i], -eta_clamp, eta_clamp)
        end
        max_change = maximum(abs.(beta_new .- beta))
        beta = beta_new
        copyto!(eta, eta_new)
        max_change < tol && break
    end
    iters_out === nothing || (iters_out[] = n_it)
    return beta
end

"""
    cell_perms(m) -> Vector{Matrix{Int32}}

Per sector, `(k_max × R_CELL_REG)`: column j is the first `k_max = min(m, max N_HI_CAP)`
entries of a random permutation of `1:m`. Deterministic in `(m, CELL_PERM_SEED)`,
cached per worker on `m`.
"""
const _CELL_PERMS = Ref{Union{Nothing, Tuple{Int, Vector{Matrix{Int32}}}}}(nothing)

function cell_perms(m::Int)
    C = _CELL_PERMS[]
    (C !== nothing && C[1] == m) && return C[2]
    P = draw_cell_perms(m, MersenneTwister(CELL_PERM_SEED + m))
    _CELL_PERMS[] = (m, P)
    return P
end

"""
    draw_cell_perms(m, rng) -> Vector{Matrix{Int32}}

The permutation set of `cell_perms` from an explicit `rng`, not cached: each fake world
(inference.jl) draws its own, so its simulated β carries its own simulation noise.
"""
function draw_cell_perms(m::Int, rng::AbstractRNG)
    k_max = min(m, maximum(N_HI_CAP))
    P = [Matrix{Int32}(undef, k_max, R_CELL_REG) for _ in 1:S]
    for s in 1:S, j in 1:R_CELL_REG
        P[s][:, j] .= Int32.(randperm(rng, m)[1:k_max])
    end
    return P
end

"""
    cell_cloglog_beta(y; max_iter=100) -> (β, ok)

The data statistic on 𝒜 (`REG_CELL_*`): cloglog of `y` with groups without variation in
`y` dropped. `ok = false` on non-convergence or a non-finite coefficient.
"""
function cell_cloglog_beta(y::AbstractVector{<:Real}; max_iter::Int = 100)
    n  = length(y)
    ng = REG_CELL_N_FE
    sy = zeros(ng); sy2 = zeros(ng); cnt = zeros(Int, ng)
    @inbounds for i in 1:n
        f = REG_CELL_FE[i]
        sy[f] += y[i]; cnt[f] += 1
    end
    keep = falses(n)
    @inbounds for i in 1:n
        f = REG_CELL_FE[i]
        keep[i] = 0 < sy[f] < cnt[f]
    end
    if !all(v -> v == 0 || v == 1, y)          # fractional y: drop constant groups only
        @inbounds for i in 1:n
            f = REG_CELL_FE[i]
            sy2[f] += (y[i] - sy[f] / cnt[f])^2
        end
        @inbounds for i in 1:n
            keep[i] = sy2[REG_CELL_FE[i]] > 0
        end
    end
    idx = findall(keep)
    k = size(REG_CELL_X, 2)
    length(idx) <= k && return (fill(NaN, k), false)
    it = Ref(0)
    beta = try
        _cloglog_irls(view(y, idx), REG_CELL_X[idx, :], ones(length(idx)),
                      REG_CELL_FE[idx]; max_iter = max_iter, iters_out = it)
    catch e
        # A regressor without variation on the kept rows (e.g. an empty distance bin in
        # one network) makes the design singular: a failed fit, like quasi-separation.
        e isa LinearAlgebra.SingularException || rethrow()
        return (fill(NaN, k), false)
    end
    return (beta, all(isfinite, beta) && it[] < max_iter)
end

"""
    cell_cloglog_moment(linkages, N_hat; perms = cell_perms(m)) -> (β̄, n_fail, β_reps)

Block 4: the mean of `cell_cloglog_beta` over the `R_CELL_REG` nested networks. Failed
fits are excluded from the mean and counted.
"""
function cell_cloglog_moment(linkages, N_hat::AbstractVector{<:Integer};
                             perms::Union{Nothing, Vector{Matrix{Int32}}} = nothing)
    P  = perms === nothing ? cell_perms(size(linkages, 1)) : perms
    nA = length(REG_CELL_G)
    k  = size(REG_CELL_X, 2)
    y  = Vector{Float64}(undef, nA)
    reps = fill(NaN, k, R_CELL_REG)
    acc = zeros(k); n_ok = 0
    @inbounds for j in 1:R_CELL_REG
        for i in 1:nA
            g = REG_CELL_G[i]; Ps = P[GOOD_S[g]]
            hit = false
            for t in 1:min(N_hat[GOOD_S[g]], size(Ps, 1))
                if linkages[Ps[t, j], g] != 0
                    hit = true
                    break
                end
            end
            y[i] = hit ? 1.0 : 0.0
        end
        beta, ok = cell_cloglog_beta(y)
        if ok
            reps[:, j] .= beta; acc .+= beta; n_ok += 1
        end
    end
    return (n_ok > 0 ? acc ./ n_ok : fill(NaN, k), R_CELL_REG - n_ok, reps)
end

##################### Moments ###################

"""
    labor_share(c_tilde_r, params) -> aggregate labour cost share (block 1)

`c_tilde_r` is the full-length (R) A-adjusted unit cost of a solved network; also used
by the fake worlds (inference.jl), which measure their own labour share.
"""
function labor_share(c_tilde_r::AbstractVector, params::AbstractVector)
    chi_L = params[1]
    active = N_downstream_per_region .!= 0
    markup = (epsilon - 1) / epsilon
    price_index = sum((c_tilde_r[active] .* markup).^epsilon .* delta_r[active])^(1/epsilon)
    B = (markup / price_index)^(epsilon - 1) * 1.0 / price_index
    y_r = zeros(R)
    y_r[active] = c_tilde_r[active].^(epsilon - 1) .* delta_r[active] .* B
    # A_r is Hicks-neutral (y = A·F(ℓ, q), eq. B1): labour demand is
    # ℓ = χ_L (w/c)^{-λ} y/A with c the A-free unit cost of the CES bundle, so the
    # cost share w ℓ/(c̃ y) = χ_L (w/c)^{1-λ} carries no A — the complement of the
    # intermediate share (1-χ_L)(P/c)^{1-λ} solve_network uses. Evaluating it at c̃ = c/A
    # instead gave χ_L (w A/c)^{1-λ}, so the two shares did not sum to one and the
    # A[1] = 1 normalisation leaked into the moment.
    A_full = zeros(R)
    A_full[DOWNSTREAM_REGIONS] = params[(S + 2):(S + R_downstream + 1)] ./ params[S + 2]
    c_r = c_tilde_r .* A_full
    labor_r = zeros(R)
    labor_r[active] = chi_L .* (y_r[active] ./ A_full[active]) .*
                      (regional_wages[active] ./ c_r[active]).^(-lambda)
    return sum(regional_wages .* labor_r) / sum(c_tilde_r .* y_r)
end

"""
    compute_moments(network, params; N_fixed=nothing) -> NamedTuple

Blocks 1–6 (labour share, industry shares, π_r, [β_d ; β_κ], γ_{s,a}, Ḡ_s(0)) and the
non-moment diagnostics. `N_fixed` pins the variety count (finite differences: N̂_s is a
step function of θ). `perms` replaces the block-4 network permutations (default: the
estimation's `cell_perms`), so a draw set other than the estimation's carries its own.
"""
function compute_moments(network, params; N_fixed::Union{Nothing, Vector{Int}}=nothing,
                         perms::Union{Nothing, Vector{Matrix{Int32}}}=nothing)
    X_ls_flat = network.X_ls_flat
    c_tilde_r = network.c_tilde_r
    Y_r = network.Y_r
    linkages_flat = network.linkages_flat
    active = N_downstream_per_region .!= 0

    # 1. Labour share
    agg_labor_share = labor_share(c_tilde_r, params)

    # 2. Sector shares X_s / X of TOTAL intermediate spending (domestic + imported), the
    # base of the data's input shares (TES dom + imp).
    X_ls = zeros(R, S)
    for g in 1:n_good
        X_ls[GOOD_R[g], GOOD_S[g]] = X_ls_flat[g]
    end
    X_s = sum(X_ls, dims=1) .+ reshape(network.X_F_flat, 1, S)
    agg_industry_share = X_s ./ sum(X_s)

    # 5. Sourcing shares of total purchases: areas, and the import share on the foreign row
    gamma_out = gamma_T_from_flows(X_ls, network.X_F_flat)

    # 4b/6. Win counts → N̂_s (bisection) → Ḡ_s(0), Ḡ_s(K); all closed form in k.
    n_draw = size(linkages_flat, 1)
    k_counts = Vector{Int}(undef, n_good)
    @inbounds for g in 1:n_good
        acc = 0
        for rho in 1:n_draw
            acc += linkages_flat[rho, g] != 0
        end
        k_counts[g] = acc
    end
    q_hat = k_counts ./ n_draw
    q_french = french_variety_share(linkages_flat)
    N_hat_free, clamped = concentrate_N_s(k_counts, n_draw; q = q_french)
    N_hat = N_fixed === nothing ? N_hat_free : copy(N_fixed)

    reg_coef, reg_fail, _ = cell_cloglog_moment(linkages_flat, N_hat; perms = perms)

    lg = gbar_logfact_table(n_draw)
    G0 = zeros(S)
    G_curve = zeros(S, length(COUNT_CURVE_K))
    @inbounds for s in 1:S
        cells = CELLS_OF_SECTOR[s]
        isempty(cells) && continue
        G0[s] = gbar_sector(cells, k_counts, n_draw, N_hat[s], lg)
        for (i, KK) in enumerate(COUNT_CURVE_K)
            G_curve[s, i] = gbar_sector_K(cells, k_counts, n_draw, N_hat[s], KK, lg)
        end
    end

    # 3. Downstream sales shares
    pi_r = Y_r[active] / sum(Y_r[active])

    return (agg_labor_share = [agg_labor_share], agg_industry_share = vec(agg_industry_share),
            pi_r = pi_r, reg_coef = reg_coef, gamma_ls = gamma_out, G0 = G0,
            N_hat = N_hat, N_hat_free = N_hat_free, clamped = clamped, q_hat = q_hat,
            q_french = q_french, foreign_share = gamma_out[FOREIGN_COL, :],
            G_curve = G_curve, k_counts = k_counts, n_draw = n_draw, reg_fail = reg_fail)
end

moment_blocks_tuple(m) = (m.agg_labor_share, m.agg_industry_share, m.pi_r, m.reg_coef, m.gamma_ls, m.G0)

"""
    moments_to_vec(sim) -> masked moment vector (from the 6-block tuple or the NamedTuple)
"""
function moments_to_vec(sim)
    blocks = sim isa Tuple ? sim : moment_blocks_tuple(sim)
    return vcat([vec(b) for b in blocks]...)[MOMENT_MASK]
end

"""
    loss_function(sim, emp, W; moment_indices=nothing) -> 1×1 matrix  err W err'

`err = emp − sim` over the masked vector, restricted to `moment_indices` when given; a
`W` already of that size is used as is.
"""
function loss_function(simulated_moments, emp, W;
                       moment_indices::Union{Nothing, Vector{Int}} = nothing)
    sim_flat = moments_to_vec(simulated_moments)
    emp_flat = vec(emp)
    if moment_indices !== nothing
        sim_flat = sim_flat[moment_indices]; emp_flat = emp_flat[moment_indices]
        size(W, 1) == length(moment_indices) || (W = W[moment_indices, moment_indices])
    end
    err = reshape(emp_flat - sim_flat, (1, length(sim_flat)))
    return err * W * err'
end

"""
    full_SMM(params; u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS, precomputed_tau,
             W_override, moment_blocks, N_fixed, return_full=false) -> (loss, moments)

`moments` is the 6-block tuple, or the whole `compute_moments` NamedTuple when
`return_full`.
"""
function full_SMM(params; u_draws::Matrix{Float64}=U_DRAWS,
                  sample_weights::AbstractMatrix{Float64}=SAMPLE_WEIGHTS,
                  precomputed_tau::Union{Nothing, Matrix{Float64}}=nothing,
                  W_override::Union{Nothing, AbstractMatrix}=nothing,
                  moment_blocks::Union{Nothing, Vector{Int}}=nothing,
                  N_fixed::Union{Nothing, Vector{Int}}=nothing,
                  return_full::Bool=false)
    network = solve_network(params; u_draws=u_draws, sample_weights=sample_weights,
                            precomputed_tau=precomputed_tau)
    m = compute_moments(network, params; N_fixed=N_fixed)
    moments = return_full ? m : moment_blocks_tuple(m)
    W = W_override !== nothing ? W_override : Weight_matrix_custom
    idx = moment_blocks === nothing ? nothing : vcat([collect(BLOCK_RANGES[b]) for b in moment_blocks]...)
    return loss_function(moments, empirical_moments, W; moment_indices=idx), moments
end

"""
    safe_SMM(params; kwargs...) -> full_SMM result, or `nothing` on error
"""
function safe_SMM(params; kwargs...)
    try
        return full_SMM(params; kwargs...)
    catch e
        println("ERROR in full_SMM: ", e)
        return nothing
    end
end

"""
    granular_report(params; u_draws, sample_weights, N_fixed=nothing) -> NamedTuple

Non-moment diagnostics: N̂_s and clamps, the bisection's bounds `N_lo` (N^obs_s/(R_d q_s))
and `N_hi` (N^obs_s/q_s with imports) and the French variety share `q_french`, the simulated import share,
q̂ per cell, Ḡ_s(0), the second route to N_s
`N^count_s = N_supplier_s / Σ_l q̂_ls`, expected supplier counts `N̂_s q̂`, and block 4 as
model `E[β̂(y)]`, its sd across networks and `β̂(E[y])` (the cloglog on the expected
network — the gap to `E[β̂(y)]` is what one network's granularity adds).
"""
function granular_report(params; u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS,
                         N_fixed::Union{Nothing, Vector{Int}}=nothing)
    network = solve_network(params; u_draws=u_draws, sample_weights=sample_weights)
    m = compute_moments(network, params; N_fixed=N_fixed)
    N_count = [begin
        sq = sum(m.q_hat[g] for g in CELLS_OF_SECTOR[s]; init = 0.0)
        sq > 0 ? N_HI[s] / sq : NaN
    end for s in 1:S]
    EK = [m.N_hat[GOOD_S[g]] * m.q_hat[g] for g in 1:n_good]
    lg = gbar_logfact_table(m.n_draw)
    Ey = [1.0 - gbar_cell(m.k_counts[g], m.n_draw, m.N_hat[GOOD_S[g]], lg) for g in REG_CELL_G]
    beta_expected, _ = cell_cloglog_beta(Ey)
    _, _, reps = cell_cloglog_moment(network.linkages_flat, m.N_hat)
    reg_sd = [std(filter(isfinite, reps[i, :])) for i in 1:size(reps, 1)]
    N_hi = [n_hi_bound(s, m.n_draw, m.q_french) for s in 1:S]
    N_lo = [min(n_lo_bound(s, m.q_french), N_hi[s]) for s in 1:S]
    return (N_hat = m.N_hat, clamped = m.clamped, q_hat = m.q_hat, reg_coef = m.reg_coef,
            reg_fail = m.reg_fail, reg_sd = reg_sd, beta_expected = beta_expected,
            G0 = m.G0, N_count = N_count, EK = EK, G_curve = m.G_curve,
            k_counts = m.k_counts, n_draw = m.n_draw,
            q_french = m.q_french, N_lo = N_lo, N_hi = N_hi, foreign_share = m.foreign_share)
end
