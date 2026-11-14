##### Spatial Comovement — simulated model (main specification) #####
#
# Notation (paper, Appendix B): r downstream buyer, l (or r') upstream cell, s sector,
# ρ variety. ε demand elasticity, χ^L labour share, χ^s sector shares, λ labour/inputs
# elasticity, ν across sectors, ν_s across varieties, θ Fréchet shape, T_ls Fréchet
# scale, A_r downstream productivity, τ = d^α (or piecewise in ln d by distance bin).
#
# T_ls = T_{a(l)s} · n_ls^κ: comparative advantage at the attraction-area level a(l),
# scaled by the cell's firm count n_ls. N_s varieties per sector, profiled out of the
# loss by an integer bisection on the count moment Ḡ_s(0).
#
# Index conventions in the code: `g` runs over the active cells (`GOOD_S[g]`, `GOOD_R[g]`
# give its sector and zone), `l` over zones, `dr` over downstream buyers in
# `DOWNSTREAM_REGIONS` order (`region = DOWNSTREAM_REGIONS[dr]` is the zone), `rho` over
# simulated varieties. Matrix suffixes give the order of the indices (`T_ls` is S × R,
# `X_ls` is R × S, as in the paper's subscripts or the array layout stated in each docstring).

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
removes the all-zeros origin). Each coordinate's 52-bit mantissa is XOR-ed with its own
random shift.
"""
function sobol_scrambled_net(N::Int, d::Int, rng::AbstractRNG)
    sobol_points = QuasiMonteCarlo.sample(N, zeros(d), ones(d), SobolSample())
    U     = Matrix{Float64}(undef, N, d)
    nbits = 52
    scale = Float64(2)^nbits
    mask  = (UInt64(1) << nbits) - UInt64(1)
    @inbounds for j in 1:d
        shift = rand(rng, UInt64) & mask
        for i in 1:N
            mantissa = floor(UInt64, sobol_points[j, i] * scale) & mask
            U[i, j]  = (mantissa ⊻ shift) / scale
        end
    end
    return U
end

"""
    sobol_dim_shifted(N, j, rng) -> Vector (N)

Dimension `j` of an `N`-point Sobol net, randomised by its own digital shift from `rng`.
"""
function sobol_dim_shifted(N::Int, j::Int, rng::AbstractRNG)
    sobol_points = QuasiMonteCarlo.sample(N, zeros(j), ones(j), SobolSample())
    nbits = 52
    scale = Float64(2)^nbits
    mask  = (UInt64(1) << nbits) - UInt64(1)
    shift = rand(rng, UInt64) & mask
    return [((floor(UInt64, sobol_points[j, i] * scale) & mask) ⊻ shift) / scale for i in 1:N]
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
        @inbounds for sector_cells in SECTOR_GOOD_INDICES
            isempty(sector_cells) && continue
            net = sobol_scrambled_net(N_rho, length(sector_cells), master)
            for (k, g) in enumerate(sector_cells), rho in 1:N_rho
                U[rho, g] = clamp(net[rho, k], lo, hi)
            end
        end
        # The foreign competitor's cost draw: one extra Sobol dimension per importing sector.
        foreign_rng = MersenneTwister(hash((:foreign_draws, rand(copy(master), UInt64))))
        for s in 1:S
            (FOREIGN_ACTIVE[s] && !isempty(SECTOR_GOOD_INDICES[s])) || continue
            u_foreign = sobol_dim_shifted(N_rho, length(SECTOR_GOOD_INDICES[s]) + 1, foreign_rng)
            @inbounds for rho in 1:N_rho
                U[rho, n_good + s] = clamp(u_foreign[rho], lo, hi)
            end
        end
        return U, W
    elseif method === :mc
        draw_rng = randomise ? rng : MersenneTwister(0)
        U = rand(draw_rng, N_rho, n_good)
        return hcat(U, clamp.(rand(draw_rng, N_rho, S), lo, hi)), W
    end
    error("Unknown draw method :$method (choose :sobol or :mc)")
end

##################### Parameters ###################

# Positions in the level vector [χ^L | χ^s (S) | A (R_d) | α (N_TAU) | κ | T block].
t_param_start() = S + R_downstream + 3 + N_TAU          # first T index
alpha_range()   = (S + R_downstream + 2):(S + R_downstream + 1 + N_TAU)
kappa_index()   = S + R_downstream + 2 + N_TAU
unpack_kappa(params) = params[kappa_index()]

"""
    unpack_T_area(params) -> Matrix (S × T_COL_DIM)

The area-level comparative-advantage matrix: the reduced T block scattered s-major
over `T_MASK`, reference-normalised per sector (`T[s, ref] = 1`). Its last column
(`FOREIGN_COL`) holds the foreign scale Φ^F_s.
"""
function unpack_T_area(params)
    T_full = zeros(eltype(params), T_COL_DIM * S)
    T_full[T_MASK] = params[t_param_start():end]
    T_area = permutedims(reshape(T_full, T_COL_DIM, S))
    for s in 1:S
        reference_area = T_REF_REGION[s]
        (reference_area > 0 && T_area[s, reference_area] > 0) && (T_area[s, :] ./= T_area[s, reference_area])
    end
    return T_area
end

"""
    gather_T_to_ze(T_area, kappa) -> T_ls (S × R)

`T[s, l] = T_area[s, a(l)] · n_ls^κ`: every ZE of an area shares the Fréchet scale, so
each ZE still runs its own Ricardian competition; the density term n_ls^κ makes a cell
with more firms a stronger competitor.
"""
function gather_T_to_ze(T_area::AbstractMatrix, kappa)
    T_ls = Matrix{promote_type(eltype(T_area), typeof(kappa))}(undef, S, R)
    @inbounds for l in 1:R, s in 1:S
        T_ls[s, l] = T_area[s, T_GATHER[l]] * exp(kappa * LOG_N_LS[s, l])
    end
    return T_ls
end

"""
    foreign_phi(T_area) -> Vector (S):  Φ^F_s = T_area[s, FOREIGN_COL], zero where s does not import
"""
foreign_phi(T_area::AbstractMatrix) =
    [FOREIGN_ACTIVE[s] ? T_area[s, FOREIGN_COL] : zero(eltype(T_area)) for s in 1:S]

"""
    unpack_params(params) -> (χ^L, χ^s, A, α, vec(T_ls), Φ^F)

χ^s (S) is normalised to sum to one, A (R_d) to A[1] = 1, T is gathered to the (S × R)
ZE level and flattened, and Φ^F (S) is the foreign Fréchet scale, in the same
reference-normalised units as T.
"""
function unpack_params(params)
    chi_L = params[1]
    chi_s = params[2:(1 + S)] / sum(params[2:(1 + S)])
    A = params[(S + 2):(S + R_downstream + 1)]
    A = A ./ A[1]
    alpha = params[alpha_range()]
    T_area = unpack_T_area(params)
    T_ls = gather_T_to_ze(T_area, unpack_kappa(params))
    return chi_L, chi_s, A, alpha, vec(T_ls), foreign_phi(T_area)
end

"""
    aggregate_gamma_to_T(gamma_cell; foreign = nothing) -> γ (T_COL_DIM × S)

Area aggregate `γ_{s,a} = Σ_{l ∈ a} γ_{ls}` of the cell-level shares `gamma_cell` (R × S)
over every simulated cell of the area; the foreign row `FOREIGN_COL` is `foreign` (S)
when given, zero otherwise.
"""
function aggregate_gamma_to_T(gamma_cell::AbstractMatrix; foreign = nothing)
    gamma_area = zeros(eltype(gamma_cell), T_COL_DIM, S)
    @inbounds for g in 1:n_good
        gamma_area[T_GATHER[GOOD_R[g]], GOOD_S[g]] += gamma_cell[GOOD_R[g], GOOD_S[g]]
    end
    foreign === nothing || (gamma_area[FOREIGN_COL, :] .= foreign)
    return gamma_area
end

"""
    gamma_T_from_flows(X_ls, X_F) -> γ (T_COL_DIM × S)

Shares of each sector's TOTAL purchases: area aggregates of the domestic flows `X_ls`
(R × S) and the foreign flow `X_F` (S) on the foreign row, divided by `Σ_l X_ls + X_F`.
They sum to one per sector, as the data γ̂ with f̂ on the foreign row.
"""
function gamma_T_from_flows(X_ls::AbstractMatrix, X_F::AbstractVector)
    total_purchases_s = vec(sum(X_ls; dims = 1)) .+ X_F
    gamma_cell   = zeros(promote_type(eltype(X_ls), eltype(X_F)), R, S)
    import_share = zeros(eltype(gamma_cell), S)
    @inbounds for s in 1:S
        total_purchases_s[s] > 1e-300 || continue
        for l in 1:R
            gamma_cell[l, s] = X_ls[l, s] / total_purchases_s[s]
        end
        import_share[s] = X_F[s] / total_purchases_s[s]
    end
    return aggregate_gamma_to_T(gamma_cell; foreign = import_share)
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
    log_knots = TAU_LOG_KNOTS
    for l in 1:R, dr in 1:R_downstream
        log_distance = LOG_DIST_DOWNSTREAM[l, dr]
        if N_TAU == 1
            tau[l, dr] = exp(alpha[1] * log_distance)
        else
            # Add up the segments of ln τ crossed between the first knot and ln d.
            log_tau = alpha[1] * (min(log_distance, log_knots[2]) - log_knots[1])
            for k in 2:N_TAU
                log_distance > log_knots[k] || break
                log_tau += alpha[k] * ((k == N_TAU ? log_distance : min(log_distance, log_knots[k + 1])) - log_knots[k])
            end
            tau[l, dr] = exp(log_tau)
        end
    end
    return tau
end

##################### Count moment and the variety count ###################
#
# A cell wins k of the m simulated varieties for at least one buyer. With N_s varieties
# drawn from the same distribution, the probability that it supplies nothing is
# (1 − q_l)^N_s, q_l = k/m; the estimators below are the unbiased versions from k and m.

"""
    gbar_logfact_table(m) -> Vector  (log_factorial[i+1] = log i!, i = 0..m)
"""
function gbar_logfact_table(m::Integer)
    log_factorial = Vector{Float64}(undef, m + 1)
    log_factorial[1] = 0.0
    @inbounds for i in 1:m
        log_factorial[i+1] = log_factorial[i] + log(i)
    end
    return log_factorial
end

"""
    gbar_cell(k, m, n, log_factorial) -> Float64

UNBIASED estimator of `(1 − q)^n` from `k` wins out of `m` i.i.d. draws:
`E[C(m−k, n)/C(m, n)] = (1 − q)^n` — draw `n` of the `m` varieties without replacement
and ask whether all of them lost.
"""
@inline function gbar_cell(k::Integer, m::Integer, n::Integer, log_factorial::Vector{Float64})
    n <= 0 && return 1.0
    n > m - k && return 0.0
    return exp(log_factorial[m-k+1] - log_factorial[m-k-n+1] - log_factorial[m+1] + log_factorial[m-n+1])
end

"""
    gbar_sector(cells, k_counts, m, n, log_factorial) -> Ḡ_s(n) = mean_l (1 − q_l)^n

The share of a sector's cells that supply no buyer when the sector has `n` varieties.
"""
function gbar_sector(cells, k_counts::AbstractVector{<:Integer}, m::Integer,
                     n::Integer, log_factorial::Vector{Float64})
    isempty(cells) && return 1.0
    sum_over_cells = 0.0
    @inbounds for g in cells
        sum_over_cells += gbar_cell(k_counts[g], m, n, log_factorial)
    end
    return sum_over_cells / length(cells)
end

"""K values at which the supplier-count distribution is tabulated (K = 0 is targeted)."""
const COUNT_CURVE_K = 0:3

"""
    gbar_cell_K(k, m, n, j, log_factorial) -> Float64

UNBIASED estimator of `Pr[K_ls = j] = C(n,j) q^j (1−q)^(n−j)` (hypergeometric argument
of `gbar_cell`; at `j = 0` it is `gbar_cell`).
"""
@inline function gbar_cell_K(k::Integer, m::Integer, n::Integer, j::Integer,
                             log_factorial::Vector{Float64})
    n <= 0 && return j == 0 ? 1.0 : 0.0
    (j < 0 || j > n || j > k) && return 0.0
    n - j > m - k && return 0.0
    return exp((log_factorial[k+1]     - log_factorial[j+1]   - log_factorial[k-j+1]) +
               (log_factorial[m-k+1]   - log_factorial[n-j+1] - log_factorial[m-k-n+j+1]) -
               (log_factorial[m+1]     - log_factorial[n+1]   - log_factorial[m-n+1]))
end

"""
    gbar_sector_K(cells, k_counts, m, n, j, log_factorial) -> mean over the sector's cells
    of Pr[K_ls = j], the share of cells with exactly j varieties won
"""
function gbar_sector_K(cells, k_counts::AbstractVector{<:Integer}, m::Integer,
                       n::Integer, j::Integer, log_factorial::Vector{Float64})
    isempty(cells) && return j == 0 ? 1.0 : 0.0
    sum_over_cells = 0.0
    @inbounds for g in cells
        sum_over_cells += gbar_cell_K(k_counts[g], m, n, j, log_factorial)
    end
    return sum_over_cells / length(cells)
end

"""
    french_variety_share(linkages) -> q (S):  share of the draws won by a French cell for
    at least one buyer (1 in a sector without foreign competitor)

`linkages` is `N_rho × n_good`, nonzero where the cell supplies the variety to some buyer.
"""
function french_variety_share(linkages::AbstractMatrix)
    n_draws = size(linkages, 1)
    q = ones(S)
    for s in 1:S
        (FOREIGN_ACTIVE[s] && !isempty(CELLS_OF_SECTOR[s])) || continue
        n_won_in_france = 0
        @inbounds for rho in 1:n_draws
            any(g -> linkages[rho, g] != 0, CELLS_OF_SECTOR[s]) && (n_won_in_france += 1)
        end
        q[s] = n_won_in_france / n_draws
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
    upper = q[s] > 0 ? ceil(Int, N_HI[s] / q[s] - 1e-9) : N_HI_CAP[s]
    return max(min(upper, N_HI_CAP[s], m), min(N_HI[s], m))
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
    concentrate_N_s(k_counts, m; target = G_TARGET, q = nothing) -> (N_hat (S), clamped (S))

`target` is the Ḡ_s(0) matched: the data by default, a fake world's own realised
empty-cell share in inference.jl. `q` is `french_variety_share` of the same draws: with
imports the bounds are `N^obs_s / (R_d q_s)` (`n_lo_bound`) and `N^obs_s / q_s`
(`n_hi_bound`), `⌈N^obs_s / R_d⌉` and `N^obs_s` without.

Monotone integer bisection on `Ḡ_s(n)` over `[n_lo_bound, n_hi_bound]` matching `target[s]`:
more varieties leave fewer cells without a buyer, so Ḡ_s is decreasing in n.
`k_counts[g]` is the number of the `m` draws cell `g` wins somewhere downstream (free of
`N_s`, Lemma 2). `clamped[s] ∈ (:none, :lo, :hi)` records a binding bound — a rejection
signal for the mechanism, reported per sector.
"""
function concentrate_N_s(k_counts::AbstractVector{<:Integer}, m::Integer;
                         target::AbstractVector{<:Real} = G_TARGET, q = nothing)
    N_hat   = zeros(Int, S)
    clamped = fill(:none, S)
    log_factorial = gbar_logfact_table(m)
    @inbounds for s in 1:S
        cells = CELLS_OF_SECTOR[s]
        upper_bound = n_hi_bound(s, m, q)
        lower_bound = min(n_lo_bound(s, q), upper_bound)
        if isempty(cells)
            N_hat[s] = lower_bound
            continue
        end
        G_bar = n -> gbar_sector(cells, k_counts, m, n, log_factorial)
        target_s = target[s]
        if G_bar(lower_bound) <= target_s
            N_hat[s], clamped[s] = lower_bound, :lo
        elseif G_bar(upper_bound) >= target_s
            N_hat[s], clamped[s] = upper_bound, :hi
        else
            lo, hi = lower_bound, upper_bound
            while hi - lo > 1
                mid = (lo + hi) ÷ 2
                G_bar(mid) > target_s ? (lo = mid) : (hi = mid)
            end
            N_hat[s] = abs(G_bar(lo) - target_s) <= abs(G_bar(hi) - target_s) ? lo : hi
        end
        @assert N_hat[s] <= N_rho "N̂[$s] = $(N_hat[s]) exceeds N_rho = $N_rho"
    end
    return N_hat, clamped
end

##################### Network solution ###################

"""
    solve_network(params; u_draws, sample_weights, precomputed_tau=nothing,
                  return_firm_level=false) -> NamedTuple

Ricardian selection (lowest-cost cell wins each variety for each buyer), nested CES
price indices, downstream sales and trade flows, on the passed draws (`N_rho_eff` =
their row count). `u_draws` is `N_rho_eff × (n_good + S)` (cells, then one foreign
column per sector) and `sample_weights` `N_rho_eff × n_good`.

Returns `X_ls_flat` (n_good, domestic sales of each cell), `X_F_flat` (S, imports),
`X_F_by_r` (S × R_d, imports per unit of the buyer's cost), `c_tilde_r` and `Y_r` (R,
A-adjusted unit cost and sales, zero outside the downstream zones), `linkages_flat`
(N_rho_eff × n_good, true where the cell supplies the variety to some buyer) and the
weights. `return_firm_level` adds the per-(variety, buyer) expenditure shares (COO
vectors), the productivity matrix `z_flat`, the markup `mu` and the price index `P`.

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
    chi_L, chi_s, A_r, alpha, T_vec, PhiF = unpack_params(params)
    tau = precomputed_tau === nothing ? build_tau(alpha) : precomputed_tau
    T_ls = reshape(T_vec, S, R)
    nu_s_mat = reshape(nu_s, 1, S)
    N_rho_eff = size(u_draws, 1)
    @assert size(sample_weights, 1) == N_rho_eff "sample_weights rows must match u_draws rows"
    any(>(0), PhiF) && @assert(size(u_draws, 2) >= n_good + S,
                               "u_draws needs n_good + S columns (foreign draws); use generate_draws")

    # 1) The foreign competitor's price for each variety; Inf where the sector does not import.
    p_F = fill(Inf, N_rho_eff, S)
    for s in 1:S
        PhiF[s] > 0 || continue
        foreign_scale = PhiF[s]^(-1 / theta)
        @inbounds for rho in 1:N_rho_eff
            p_F[rho, s] = foreign_scale * (-log(1.0 - u_draws[rho, n_good + s]))^(1 / theta)
        end
    end
    X_F_by_r = zeros(S, R_downstream)

    # 2) Each cell's productivity draw for each variety, z = T^{1/θ} (−log(1−u))^{−1/θ},
    #    built one column at a time; only 1/z enters the cost comparison.
    z_inv_flat = Matrix{Float64}(undef, N_rho_eff, n_good)
    z_flat     = return_firm_level ? Matrix{Float64}(undef, N_rho_eff, n_good) : nothing
    z_column = Vector{Float64}(undef, N_rho_eff)
    for g in 1:n_good
        T_root = max(T_ls[GOOD_S[g], GOOD_R[g]], eps(Float64))^(1/theta)
        for rho in 1:N_rho_eff
            z_column[rho] = T_root * (-log(1.0 - u_draws[rho, g]))^(-1.0/theta)
        end
        @views z_inv_flat[:, g] .= z_column .^ (-1)
        return_firm_level && (@views z_flat[:, g] .= z_column)
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

    for dr in 1:R_downstream
        region = DOWNSTREAM_REGIONS[dr]

        # 3) Ricardian selection: for each variety the buyer picks the lowest delivered cost
        #    p = w τ / z among the sector's cells; ties keep the first minimiser and a NaN
        #    wins outright (as `argmin`).
        for s in 1:S
            sector_cells = SECTOR_GOOD_INDICES[s]
            isempty(sector_cells) && continue
            tau_to_buyer::Vector{Float64} = tau_c[SECTOR_GOOD_REGIONS[s], dr]
            cell_wages::Vector{Float64}   = W_RS_FLAT[sector_cells]
            n_sector_cells = length(sector_cells)
            @inbounds for rho in 1:N_rho_eff
                best_price = z_inv_c[rho, sector_cells[1]] * tau_to_buyer[1] * cell_wages[1]
                best_position = 1
                if !isnan(best_price)
                    for position in 2:n_sector_cells
                        price = z_inv_c[rho, sector_cells[position]] * tau_to_buyer[position] * cell_wages[position]
                        if isnan(price)
                            best_price = price; best_position = position; break
                        elseif price < best_price
                            best_price = price; best_position = position
                        end
                    end
                end
                # Foreign bid strictly below the best domestic cell (a NaN domestic price
                # keeps winning, as above): the variety is imported (winner = −1).
                if p_F[rho, s] < best_price
                    winner_good_idx[rho, s] = -1
                    p_rho_s[rho, s] = p_F[rho, s]
                else
                    winner_good_idx[rho, s] = sector_cells[best_position]
                    p_rho_s[rho, s] = best_price
                end
            end
        end

        # 4) Nested CES price indices at the realised prices:
        #    P_sr = [Σ_ρ w_ρ p_ρ^{1−ν_s}]^{1/(1−ν_s)}, P_r = [Σ_s χ^s P_sr^{1−ν}]^{1/(1−ν)},
        #    c_r = [χ^L w_r^{1−λ} + (1−χ^L) P_r^{1−λ}]^{1/(1−λ)}.
        for s in 1:S
            nu_within = nu_s[s]
            for rho in 1:N_rho_eff
                winner = winner_good_idx[rho, s]
                # A variety's weight is the sector's (constant across its cells), so an
                # imported variety reads it on the sector's first cell.
                w_rho_s[rho, s] = winner == 0 ? 0.0 :
                                  sample_weights[rho, winner > 0 ? winner : SECTOR_GOOD_INDICES[s][1]]
                ces_buf[rho, s] = w_rho_s[rho, s] * p_rho_s[rho, s]^(1 - nu_within)
            end
        end
        P_sr = sum(ces_buf, dims=1).^(1 ./ (1 .- nu_s_mat))
        P_r = sum(P_sr.^(1 - nu) .* reshape(chi_s, 1, S))^(1 / (1 - nu))
        c_r = (chi_L * regional_wages[region]^(1-lambda) +
               (1-chi_L) * P_r^(1-lambda))^(1/(1-lambda))
        c_tilde_r[dr] = c_r / A_r[dr]

        # 5) Expenditure share of each won variety in the buyer's unit cost; imported
        #    varieties go to the foreign bill and create no domestic linkage.
        labor_substitution_factor = (1 - chi_L) * (P_r / c_r)^(1 - lambda)
        for s in 1:S
            P_sr_value = P_sr[s]
            nu_within = nu_s[s]
            for rho in 1:N_rho_eff
                winner = winner_good_idx[rho, s]
                winner == 0 && continue
                if winner < 0
                    X_F_by_r[s, dr] += w_rho_s[rho, s] * chi_s[s] * (1-chi_L) *
                        (p_rho_s[rho, s] / P_sr_value)^(1 - nu_within) *
                        (P_sr_value / P_r)^(1 - nu) *
                        (P_r / c_r)^(1 - lambda)
                    continue
                end
                linkages_flat[rho, winner] = true
                exp_val = sample_weights[rho, winner] * chi_s[s] * (1-chi_L) *
                    (p_rho_s[rho, s] / P_sr_value)^(1 - nu_within) *
                    (P_sr_value / P_r)^(1 - nu) *
                    (P_r / c_r)^(1 - lambda)
                X_shares_by_r[winner, dr] += exp_val
                if return_firm_level
                    push!(firm_exp_rho, rho); push!(firm_exp_s, s); push!(firm_exp_g, winner)
                    push!(firm_exp_r, region); push!(firm_exp_val, exp_val)
                    push!(firm_deriv_val, exp_val / labor_substitution_factor)
                end
            end
        end
    end

    # 6) Downstream market: p_r = c̃_r/μ, P = [Σ_r p_r^ε δ_r]^{1/ε}, Y_r = p_r^ε P^{−ε} δ_r (E = 1).
    mu  = epsilon/(epsilon - 1)
    p_r = c_tilde_r ./ mu
    P_agg = sum(p_r.^epsilon .* delta_r[DOWNSTREAM_REGIONS])^(1/epsilon)
    Y_r = p_r.^epsilon .* P_agg^(-epsilon) .* 1.0 .* delta_r[DOWNSTREAM_REGIONS]

    # 7) Trade flows: each buyer's spending shares times its total cost μ Y_r.
    X_ls_flat = zeros(n_good)
    X_F_flat  = zeros(S)
    for dr in 1:R_downstream
        total_cost_r = mu * Y_r[dr]
        for g in 1:n_good
            X_ls_flat[g] += X_shares_by_r[g, dr] * total_cost_r
        end
        for s in 1:S
            X_F_flat[s] += X_F_by_r[s, dr] * total_cost_r
        end
    end
    c_tilde_r_full = zeros(R); c_tilde_r_full[DOWNSTREAM_REGIONS] = c_tilde_r
    Y_r_full       = zeros(R); Y_r_full[DOWNSTREAM_REGIONS]       = Y_r

    # X_F_by_r: foreign spending per unit of the buyer's cost (S × R_downstream), the
    # imported part of the intermediate share; X_F_flat: in the units of X_ls_flat.
    network = (X_ls_flat = X_ls_flat, X_F_flat = X_F_flat, X_F_by_r = X_F_by_r,
               c_tilde_r = c_tilde_r_full, Y_r = Y_r_full,
               linkages_flat = linkages_flat, sample_weights = sample_weights)
    return_firm_level || return network
    return merge(network, (z_flat = z_flat, firm_exp_rho = firm_exp_rho, firm_exp_s = firm_exp_s,
                           firm_exp_g = firm_exp_g, firm_exp_r = firm_exp_r,
                           firm_exp_val = firm_exp_val, firm_deriv_val = firm_deriv_val,
                           mu = mu, P = P_agg))
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

The rows of each nonempty fixed-effect group, groups in increasing label order.
"""
function build_fe_group_rows(fe_group::AbstractVector{<:Integer})
    isempty(fe_group) && return Vector{Int}[]
    max_group = maximum(fe_group)
    group_size = zeros(Int, max_group)
    @inbounds for group in fe_group
        group_size[group] += 1
    end
    group_rows = Vector{Vector{Int}}()
    group_slot = zeros(Int, max_group)
    for group in 1:max_group
        group_size[group] > 0 || continue
        push!(group_rows, Vector{Int}(undef, group_size[group])); group_slot[group] = length(group_rows)
    end
    n_filled = zeros(Int, length(group_rows))
    @inbounds for i in eachindex(fe_group)
        slot = group_slot[fe_group[i]]
        n_filled[slot] += 1
        group_rows[slot][n_filled[slot]] = i
    end
    return group_rows
end

"""
    _cloglog_irls(y, X, w, fe_group; max_iter=50, tol=1e-9, eta_clamp=30.0, iters_out) -> β

Complementary-log-log GLM `P(y=1) = 1 − exp(−exp(η))`, `η = Xβ + a_fe`, by IRLS with the
single fixed effect absorbed by weighted within-group demeaning (exact for one FE).
`y` (n), `X` (n × k), weights `w` (n), `fe_group` (n) integer labels. Each iteration
forms the working response z = η + (y − μ)/(dμ/dη) and weights w (dμ/dη)²/(μ(1−μ)),
demeans z and X within groups (Frisch–Waugh–Lovell), solves the weighted least squares
for β and recovers the fixed effects from the group means. `iters_out` receives the
iteration count.
"""
function _cloglog_irls(y::AbstractVector{<:Real}, X::AbstractMatrix{<:Real},
                       w::AbstractVector{<:Real}, fe_group::AbstractVector{<:Integer};
                       max_iter::Int=50, tol::Float64=1e-9, eta_clamp::Float64=30.0,
                       iters_out::Union{Nothing,Base.RefValue{Int}}=nothing)
    n, k = size(X)
    yf = Float64.(y); Xf = Matrix{Float64}(X); wf = Float64.(w)
    group_rows = build_fe_group_rows(fe_group)

    mu = (yf .+ 0.5) ./ 2
    eta = clamp.(log.(.-log.(1 .- mu)), -eta_clamp, eta_clamp)
    z_working = Vector{Float64}(undef, n); W = Vector{Float64}(undef, n)
    z_demeaned = Vector{Float64}(undef, n); X_demeaned = Matrix{Float64}(undef, n, k)
    X_beta = Vector{Float64}(undef, n);    eta_new = Vector{Float64}(undef, n)
    group_weight = Vector{Float64}(undef, length(group_rows))

    beta = zeros(k)
    iterations = 0
    for _ in 1:max_iter
        iterations += 1
        # 1) Working response and IRLS weights at the current linear index.
        @inbounds for i in 1:n
            mu_i         = clamp(1 - exp(-exp(eta[i])), 1e-12, 1 - 1e-12)
            dmu_deta     = exp(eta[i]) * (1 - mu_i)
            z_working[i] = eta[i] + (yf[i] - mu_i) / dmu_deta
            W[i]         = wf[i] * (dmu_deta^2) / (mu_i * (1 - mu_i))
            z_demeaned[i] = z_working[i]
        end
        # 2) Absorb the fixed effect: weighted within-group demeaning of z and X.
        copyto!(X_demeaned, Xf)
        @inbounds for (group, rows) in enumerate(group_rows)
            weight_sum = sum(@view W[rows]); group_weight[group] = weight_sum
            weight_sum < 1e-300 && continue
            z_mean = sum(i -> W[i] * z_working[i], rows) / weight_sum
            for i in rows
                z_demeaned[i] -= z_mean
            end
            for j in 1:k
                x_mean = sum(i -> W[i] * Xf[i, j], rows) / weight_sum
                for i in rows
                    X_demeaned[i, j] -= x_mean
                end
            end
        end
        # 3) Weighted least squares on the demeaned data, then the fixed effects back.
        W_X_demeaned = W .* X_demeaned
        beta_new = (X_demeaned' * W_X_demeaned) \ (X_demeaned' * (W .* z_demeaned))
        mul!(X_beta, Xf, beta_new)
        @inbounds for i in 1:n
            eta_new[i] = X_beta[i]
        end
        @inbounds for (group, rows) in enumerate(group_rows)
            weight_sum = group_weight[group]
            weight_sum < 1e-300 && continue
            fixed_effect = sum(i -> W[i] * (z_working[i] - X_beta[i]), rows) / weight_sum
            for i in rows
                eta_new[i] += fixed_effect
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
    iters_out === nothing || (iters_out[] = iterations)
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
    cached = _CELL_PERMS[]
    (cached !== nothing && cached[1] == m) && return cached[2]
    perms = draw_cell_perms(m, MersenneTwister(CELL_PERM_SEED + m))
    _CELL_PERMS[] = (m, perms)
    return perms
end

"""
    draw_cell_perms(m, rng) -> Vector{Matrix{Int32}}

The permutation set of `cell_perms` from an explicit `rng`, not cached: each fake world
(inference.jl) draws its own, so its simulated β carries its own simulation noise.
"""
function draw_cell_perms(m::Int, rng::AbstractRNG)
    k_max = min(m, maximum(N_HI_CAP))
    perms = [Matrix{Int32}(undef, k_max, R_CELL_REG) for _ in 1:S]
    for s in 1:S, j in 1:R_CELL_REG
        perms[s][:, j] .= Int32.(randperm(rng, m)[1:k_max])
    end
    return perms
end

"""
    cell_cloglog_beta(y; max_iter=100) -> (β, ok)

The data statistic on 𝒜 (`REG_CELL_*`): cloglog of `y` (one entry per cell of 𝒜) with
groups without variation in `y` dropped, since a group that always or never supplies
carries no within-group information. `ok = false` on non-convergence or a non-finite
coefficient.
"""
function cell_cloglog_beta(y::AbstractVector{<:Real}; max_iter::Int = 100)
    n_cells  = length(y)
    n_groups = REG_CELL_N_FE
    group_y_sum = zeros(n_groups); group_y_sq_dev = zeros(n_groups); group_size = zeros(Int, n_groups)
    @inbounds for i in 1:n_cells
        group = REG_CELL_FE[i]
        group_y_sum[group] += y[i]; group_size[group] += 1
    end
    keep = falses(n_cells)
    @inbounds for i in 1:n_cells
        group = REG_CELL_FE[i]
        keep[i] = 0 < group_y_sum[group] < group_size[group]
    end
    if !all(v -> v == 0 || v == 1, y)          # fractional y: drop constant groups only
        @inbounds for i in 1:n_cells
            group = REG_CELL_FE[i]
            group_y_sq_dev[group] += (y[i] - group_y_sum[group] / group_size[group])^2
        end
        @inbounds for i in 1:n_cells
            keep[i] = group_y_sq_dev[REG_CELL_FE[i]] > 0
        end
    end
    kept_rows = findall(keep)
    n_regressors = size(REG_CELL_X, 2)
    length(kept_rows) <= n_regressors && return (fill(NaN, n_regressors), false)
    iterations = Ref(0)
    beta = try
        _cloglog_irls(view(y, kept_rows), REG_CELL_X[kept_rows, :], ones(length(kept_rows)),
                      REG_CELL_FE[kept_rows]; max_iter = max_iter, iters_out = iterations)
    catch e
        # A regressor without variation on the kept rows (e.g. an empty distance bin in
        # one network) makes the design singular: a failed fit, like quasi-separation.
        e isa LinearAlgebra.SingularException || rethrow()
        return (fill(NaN, n_regressors), false)
    end
    return (beta, all(isfinite, beta) && iterations[] < max_iter)
end

"""
    cell_cloglog_moment(linkages, N_hat; perms = cell_perms(m)) -> (β̄, n_fail, β_reps)

Block 4: the mean of `cell_cloglog_beta` over the `R_CELL_REG` nested networks. Network j
gives each sector its first N̂_s varieties of permutation j, and a cell of 𝒜 is a
supplier if it wins at least one of them. Failed fits are excluded from the mean and
counted. `β_reps` is `n_regressors × R_CELL_REG` (NaN for a failed network).
"""
function cell_cloglog_moment(linkages, N_hat::AbstractVector{<:Integer};
                             perms::Union{Nothing, Vector{Matrix{Int32}}} = nothing)
    network_perms = perms === nothing ? cell_perms(size(linkages, 1)) : perms
    n_cells      = length(REG_CELL_G)
    n_regressors = size(REG_CELL_X, 2)
    y  = Vector{Float64}(undef, n_cells)
    beta_by_network = fill(NaN, n_regressors, R_CELL_REG)
    beta_sum = zeros(n_regressors); n_ok = 0
    @inbounds for j in 1:R_CELL_REG
        for i in 1:n_cells
            g = REG_CELL_G[i]; sector_perm = network_perms[GOOD_S[g]]
            supplies = false
            for t in 1:min(N_hat[GOOD_S[g]], size(sector_perm, 1))
                if linkages[sector_perm[t, j], g] != 0
                    supplies = true
                    break
                end
            end
            y[i] = supplies ? 1.0 : 0.0
        end
        beta, ok = cell_cloglog_beta(y)
        if ok
            beta_by_network[:, j] .= beta; beta_sum .+= beta; n_ok += 1
        end
    end
    return (n_ok > 0 ? beta_sum ./ n_ok : fill(NaN, n_regressors), R_CELL_REG - n_ok, beta_by_network)
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
    # Downstream sales from the CES demand system, at prices c̃_r times the inverse markup.
    inverse_markup = (epsilon - 1) / epsilon
    price_index = sum((c_tilde_r[active] .* inverse_markup).^epsilon .* delta_r[active])^(1/epsilon)
    demand_scale = (inverse_markup / price_index)^(epsilon - 1) * 1.0 / price_index
    y_r = zeros(R)
    y_r[active] = c_tilde_r[active].^(epsilon - 1) .* delta_r[active] .* demand_scale
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
    compute_moments(network, params; N_fixed=nothing, perms=nothing) -> NamedTuple

Blocks 1–6 (labour share, industry shares, π_r, [β_d ; β_κ], γ_{s,a}, Ḡ_s(0)) and the
non-moment diagnostics. `N_fixed` pins the variety count (finite differences: N̂_s is a
step function of θ). `perms` replaces the block-4 network permutations (default: the
estimation's `cell_perms`), so a draw set other than the estimation's carries its own.
The field `gamma_ls` holds the AREA-level shares γ (T_COL_DIM × S), foreign row included.
"""
function compute_moments(network, params; N_fixed::Union{Nothing, Vector{Int}}=nothing,
                         perms::Union{Nothing, Vector{Matrix{Int32}}}=nothing)
    X_ls_flat = network.X_ls_flat
    c_tilde_r = network.c_tilde_r
    Y_r = network.Y_r
    linkages_flat = network.linkages_flat
    active = N_downstream_per_region .!= 0

    # 1. Labour share
    model_labor_share = labor_share(c_tilde_r, params)

    # 2. Sector shares X_s / X of TOTAL intermediate spending (domestic + imported), the
    # base of the data's input shares (TES dom + imp).
    X_ls = zeros(R, S)
    for g in 1:n_good
        X_ls[GOOD_R[g], GOOD_S[g]] = X_ls_flat[g]
    end
    X_s = sum(X_ls, dims=1) .+ reshape(network.X_F_flat, 1, S)
    model_industry_share = X_s ./ sum(X_s)

    # 5. Sourcing shares of total purchases: areas, and the import share on the foreign row
    gamma_area = gamma_T_from_flows(X_ls, network.X_F_flat)

    # 4b/6. Win counts → N̂_s (bisection) → Ḡ_s(0), Ḡ_s(K); all closed form in k.
    n_draw = size(linkages_flat, 1)
    k_counts = Vector{Int}(undef, n_good)
    @inbounds for g in 1:n_good
        n_won = 0
        for rho in 1:n_draw
            n_won += linkages_flat[rho, g] != 0
        end
        k_counts[g] = n_won
    end
    q_hat = k_counts ./ n_draw
    q_french = french_variety_share(linkages_flat)
    N_hat_free, clamped = concentrate_N_s(k_counts, n_draw; q = q_french)
    N_hat = N_fixed === nothing ? N_hat_free : copy(N_fixed)

    # 4. The cell-level cloglog on networks of N̂_s varieties.
    reg_coef, reg_fail, _ = cell_cloglog_moment(linkages_flat, N_hat; perms = perms)

    # 6. The share of cells without a buyer, and the supplier-count curve.
    log_factorial = gbar_logfact_table(n_draw)
    G0 = zeros(S)
    G_curve = zeros(S, length(COUNT_CURVE_K))
    @inbounds for s in 1:S
        cells = CELLS_OF_SECTOR[s]
        isempty(cells) && continue
        G0[s] = gbar_sector(cells, k_counts, n_draw, N_hat[s], log_factorial)
        for (i, K) in enumerate(COUNT_CURVE_K)
            G_curve[s, i] = gbar_sector_K(cells, k_counts, n_draw, N_hat[s], K, log_factorial)
        end
    end

    # 3. Downstream sales shares
    pi_r = Y_r[active] / sum(Y_r[active])

    return (agg_labor_share = [model_labor_share], agg_industry_share = vec(model_industry_share),
            pi_r = pi_r, reg_coef = reg_coef, gamma_ls = gamma_area, G0 = G0,
            N_hat = N_hat, N_hat_free = N_hat_free, clamped = clamped, q_hat = q_hat,
            q_french = q_french, foreign_share = gamma_area[FOREIGN_COL, :],
            G_curve = G_curve, k_counts = k_counts, n_draw = n_draw, reg_fail = reg_fail)
end

"""
    moment_blocks_tuple(moments) -> the six moment blocks of a `compute_moments` result, in order
"""
moment_blocks_tuple(moments) = (moments.agg_labor_share, moments.agg_industry_share, moments.pi_r,
                                moments.reg_coef, moments.gamma_ls, moments.G0)

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

The SMM criterion at `params`: solve the network on the estimation draws, compute the
moments, and weight their gap to the data by `W_override` (default the custom weight
matrix) on the blocks `moment_blocks` (default all). `moments` is the 6-block tuple, or
the whole `compute_moments` NamedTuple when `return_full`.
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
    all_moments = compute_moments(network, params; N_fixed=N_fixed)
    moments = return_full ? all_moments : moment_blocks_tuple(all_moments)
    W = W_override !== nothing ? W_override : Weight_matrix_custom
    moment_indices = moment_blocks === nothing ? nothing : vcat([collect(BLOCK_RANGES[b]) for b in moment_blocks]...)
    return loss_function(moments, empirical_moments, W; moment_indices=moment_indices), moments
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
`N^count_s = N_supplier_s / Σ_l q̂_ls`, expected supplier counts `N̂_s q̂` (`EK`, per cell),
and block 4 as model `E[β̂(y)]`, its sd across networks and `β̂(E[y])` (the cloglog on the
expected network — the gap to `E[β̂(y)]` is what one network's granularity adds).
"""
function granular_report(params; u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS,
                         N_fixed::Union{Nothing, Vector{Int}}=nothing)
    network = solve_network(params; u_draws=u_draws, sample_weights=sample_weights)
    moments = compute_moments(network, params; N_fixed=N_fixed)
    N_count = [begin
        sum_q = sum(moments.q_hat[g] for g in CELLS_OF_SECTOR[s]; init = 0.0)
        sum_q > 0 ? N_HI[s] / sum_q : NaN
    end for s in 1:S]
    expected_suppliers = [moments.N_hat[GOOD_S[g]] * moments.q_hat[g] for g in 1:n_good]
    log_factorial = gbar_logfact_table(moments.n_draw)
    expected_y = [1.0 - gbar_cell(moments.k_counts[g], moments.n_draw, moments.N_hat[GOOD_S[g]], log_factorial)
                  for g in REG_CELL_G]
    beta_expected, _ = cell_cloglog_beta(expected_y)
    _, _, beta_by_network = cell_cloglog_moment(network.linkages_flat, moments.N_hat)
    reg_sd = [std(filter(isfinite, beta_by_network[i, :])) for i in 1:size(beta_by_network, 1)]
    N_hi = [n_hi_bound(s, moments.n_draw, moments.q_french) for s in 1:S]
    N_lo = [min(n_lo_bound(s, moments.q_french), N_hi[s]) for s in 1:S]
    return (N_hat = moments.N_hat, clamped = moments.clamped, q_hat = moments.q_hat, reg_coef = moments.reg_coef,
            reg_fail = moments.reg_fail, reg_sd = reg_sd, beta_expected = beta_expected,
            G0 = moments.G0, N_count = N_count, EK = expected_suppliers, G_curve = moments.G_curve,
            k_counts = moments.k_counts, n_draw = moments.n_draw,
            q_french = moments.q_french, N_lo = N_lo, N_hi = N_hi, foreign_share = moments.foreign_share)
end
