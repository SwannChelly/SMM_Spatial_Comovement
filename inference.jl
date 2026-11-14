##### inference.jl — Steps 2 and 4: weighting and inference from fake worlds #####
#
# The data covariance is not used. Both the weight matrix and the inference come from
# FAKE WORLDS drawn from the model at an estimate:
#
#   Step 2 (`world_weight_matrix`). At θ̂₁ (W = I), draw B worlds: one network of N̂_s
#   varieties per sector with i.i.d. uniform draws (NOT the estimation's Sobol design).
#   Each world yields its own "data" (β_b, γ_b, G_b, π_b, L_b, Õ_b), computed as on the
#   real data. T and the aggregates are re-profiled on them (`profile_world`), N is
#   calibrated on G_b, β is simulated
#   at μ̂₁ with a seed proper to the world and at the estimation's m, and
#   e_b = β_b − β_sim is kept. Ω̃ = Cov(e) and W = Ω̃⁻¹, on block 4 only: γ and G are
#   reproduced exactly by the T inversion and the N bisection, so their weight does not
#   matter. No derivative is computed. When block 4 has as many entries as (α, κ) the
#   system is exactly identified and W does not move the point estimate.
#
#   Step 4 (`world_inference`). At θ̂₂, draw B worlds and re-run the WHOLE chain in each,
#   W fixed: the profile of T and the aggregates, N calibration, and a local search over (α, κ)
#   started at the estimate. The B parameter sets give, for every parameter, sd = their
#   dispersion, a 95% interval = their 2.5/97.5% quantiles, and bias = mean − estimate;
#   plus the within-sector dispersion of log T and closed-form counterfactuals
#   recomputed world by world.
#
# ── The profile in a world ───────────────────────────────────────────────────
# The world is treated as the data are: (T, χ^L, χ^s, A) are profiled jointly
# (`profile_ge`) on the world's own γ_b, labour share L_b, sector shares Õ_b and sales
# shares π_b, so the sampling noise of these moments enters the parameter draws, and
# χ^L, χ^s, A are reported with α, κ, N_s and T.

# ── The foreign competitor ───────────────────────────────────────────────────
# A world's network draws the foreign bids too (`generate_draws`' foreign columns), so
# its import share f_b is the realised share of its N̂_s varieties won abroad, noisy
# with N̂_s. It sits on the foreign row of γ_b, and the profile re-solves Φ^F_s on it
# with the areas' T: the sampling noise of the import share enters the draws of T and
# Φ^F. The counterfactuals hold Φ^F_s at the world's estimate.

# ── Areas that come out empty ────────────────────────────────────────────────
# A world is ONE network, so an area active in the data can win nothing in it (γ_b = 0).
# On the data such an area would not be active and its T would be zero; the world does
# the same: its T is set to `EMPTY_T` and held there by `profile_ge`. Such draws are
# reported as EMPTY rather than as a log T of −69. A world whose REFERENCE area is empty
# cannot be normalised and is excluded (`:empty_ref`).
#
# ── Seeds ────────────────────────────────────────────────────────────────────
# World b uses seed_world + b for its network and seed_sim + b for its simulator (Sobol
# draws at N_rho randomised by that seed, and its own R_CELL_REG block-4 permutations).
# Within a world the simulator is held fixed across the local search (common random
# numbers), so the world's criterion is deterministic in (α, κ).

using Printf
using Dates
using Optim

const EMPTY_T     = 1e-30
const EMPTY_T_CUT = 1e-20   # the per-sector renormalisation moves EMPTY_T slightly
const WORLD_PENALTY = 1e12

"""
    _sym_inv(A; floor_rel=1e-10) -> inv(A)

Inverse of a symmetric matrix by Cholesky, with an eigenvalue floor at
`floor_rel × λ_max` when it is not positive definite.
"""
function _sym_inv(A::AbstractMatrix; floor_rel::Float64 = 1e-10)
    A_symmetric = Symmetric((Matrix(A) .+ Matrix(A)') ./ 2)
    A_inverse = try
        Matrix(inv(cholesky(A_symmetric)))
    catch
        eigen_decomposition = eigen(A_symmetric)
        eigen_decomposition.vectors * Diagonal(1.0 ./ max.(eigen_decomposition.values, eigen_decomposition.values[end] * floor_rel)) * eigen_decomposition.vectors'
    end
    return (A_inverse .+ A_inverse') ./ 2
end

##################### One fake world ###################

"""
    draw_world(theta, N_hat, seed) -> NamedTuple

One fake world at `theta`: a single network with `N_hat[s]` varieties per sector
(i.i.d. draws, weight `1/N̂_s`, rows past `N̂_s` padded at u = 0.5 with weight zero) and
the statistics computed on it as on the real data:

  * `beta`  — the cell cloglog of `has_sup_ls = 1{cell l wins one of the N̂_s varieties
              for at least one buyer}`: ONE network, ONE fit (`ok = false` if it fails);
  * `G`     — per sector, the share of cells hosting no supplier;
  * `gamma` — the shares of total purchases by area, and the import share on the foreign
              row, summing to one per sector (the form `profile_ge` takes as `gamma_target`;
              `EMP_GAMMA_T_TILDE` on the data);
  * `sales_shares` — downstream sales shares π_r (R_d), `DOWNSTREAM_REGIONS` order;
  * `labor_share`, `sector_shares` — the labour share and the sector shares (S) of
              intermediate spending (blocks 1–2), targets of the profile with `sales_shares`;
  * `ref_empty` — true when some sector's reference area wins nothing (the world cannot
              be normalised and is excluded).
"""
function draw_world(theta::AbstractVector, N_hat::Vector{Int}, seed::Integer)
    N_max = maximum(N_hat)
    u_draws_world = generate_draws(N_max, n_good, :mc; randomise = true, rng = MersenneTwister(seed))[1]
    weights_world = zeros(N_max, n_good)
    for g in 1:n_good
        n_varieties = N_hat[GOOD_S[g]]
        weights_world[1:n_varieties, g] .= 1.0 / n_varieties
        u_draws_world[(n_varieties + 1):N_max, g] .= 0.5
    end
    network = solve_network(theta; u_draws = u_draws_world, sample_weights = weights_world)
    supplies = falses(n_good)
    @inbounds for g in 1:n_good
        supplies[g] = any(view(network.linkages_flat, 1:N_hat[GOOD_S[g]], g))
    end
    beta, ok = cell_cloglog_beta(Float64[supplies[g] for g in REG_CELL_G])
    G = [isempty(cells) ? 0.0 : count(g -> !supplies[g], cells) / length(cells) for cells in CELLS_OF_SECTOR]
    X_s = copy(network.X_F_flat)                         # total purchases: imports + areas
    for g in 1:n_good
        X_s[GOOD_S[g]] += network.X_ls_flat[g]
    end
    gamma_area = zeros(T_COL_DIM, S)
    for g in 1:n_good
        s = GOOD_S[g]
        X_s[s] > 0 && (gamma_area[T_GATHER[GOOD_R[g]], s] += network.X_ls_flat[g] / X_s[s])
    end
    for s in 1:S
        (FOREIGN_ACTIVE[s] && X_s[s] > 0) && (gamma_area[FOREIGN_COL, s] = network.X_F_flat[s] / X_s[s])
    end
    Y_downstream = network.Y_r[DOWNSTREAM_REGIONS]
    ref_empty = any(s -> T_REF_REGION[s] > 0 && gamma_area[T_REF_REGION[s], s] <= 0, 1:S)
    return (beta = beta, ok = ok, G = G, gamma = gamma_area, sales_shares = Y_downstream ./ sum(Y_downstream), ref_empty = ref_empty,
            labor_share = labor_share(network.c_tilde_r, theta), sector_shares = X_s ./ sum(X_s))
end

"""
    world_reference(theta_hat) -> (theta_vec, chi_L, chi_s, A, alpha, kappa, T)

What the worlds read off the estimate: the level vector itself (`theta_vec`), χ^L, χ^s, A
as `unpack_params` normalises them, α̂, κ̂ and T̂ by area (S × T_COL_DIM).
"""
function world_reference(theta_hat::AbstractVector)
    chi_L, chi_s, A, alpha, _ = unpack_params(theta_hat)
    return (theta_vec = collect(Float64, theta_hat), chi_L = chi_L, chi_s = collect(chi_s), A = collect(A),
            alpha = collect(alpha), kappa = Float64(unpack_kappa(theta_hat)),
            T = Matrix(unpack_T_area(theta_hat)))
end

"""
    profile_world(ref, alpha, kappa, world; T_init, aggregate_coordinates_init)
        -> (T, A, chi_L, chi_s, aggregate_coordinates, converged)

Profile T and the aggregates at (α, κ) on a world's own moments: γ (`world.gamma`) and
blocks 1–3 (`world.labor_share`, `world.sector_shares`, `world.sales_shares`), i.e. the
estimator applied to the fake data. An area that wins nothing in the world starts at
`EMPTY_T` and stays there. `aggregate_coordinates_init` warm-starts the Newton solve of
(χ^L, χ^s) (`solve_aggregates`).
"""
function profile_world(ref, alpha, kappa, world;
                       T_init::AbstractMatrix = ref.T, aggregate_coordinates_init = nothing)
    T = Matrix{Float64}(T_init)
    for s in 1:S, a in SECTOR_T_COLS[s]
        (world.gamma[a, s] <= 0 && a != T_REF_REGION[s]) && (T[s, a] = EMPTY_T)
    end
    profile = profile_ge(alpha; kappa = kappa, gamma_target = world.gamma, T_init = T,
                     aggregate_coordinates_init = aggregate_coordinates_init,
                     targets = profile_targets(labor_share = world.labor_share, sector_shares = world.sector_shares,
                                               sales_shares = world.sales_shares))
    return (T = profile.T, A = profile.A, chi_L = profile.chi_L, chi_s = profile.chi_s, aggregate_coordinates = profile.aggregate_coordinates,
            converged = profile.converged)
end

"""
    world_theta(ref, alpha, kappa, cal) -> the level vector at (α, κ) with the profiled
    aggregates and T of `cal` (`profile_world`)
"""
function world_theta(ref, alpha, kappa, world_profile)
    theta_vec = with_aggregates(ref.theta_vec, world_profile.chi_L, world_profile.chi_s, world_profile.A)
    theta_vec[alpha_range()] = alpha
    theta_vec[kappa_index()] = kappa
    return with_T(theta_vec, world_profile.T)
end

"""
    world_simulator(seed) -> (u, w, perms)

The simulator of one world: `N_rho` Sobol draws (the estimation's m and design)
randomised by the world's seed, and the world's own block-4 network permutations.
"""
function world_simulator(seed::Integer)
    rng = MersenneTwister(seed)
    u, w = generate_draws(N_rho, n_good, :sobol; randomise = true, rng = rng)
    return (u = u, w = w, perms = draw_cell_perms(N_rho, rng))
end

"""
    simulate_beta(params, G_target, sim) -> (beta, N_hat, n_fail)

Block 4 at `params` on a world's simulator: one Ricardian solve, `N̂_s` calibrated on
the world's empty-cell shares `G_target`, then the cell cloglog averaged over the
simulator's networks.
"""
function simulate_beta(params::AbstractVector, G_target::AbstractVector, sim)
    network = solve_network(params; u_draws = sim.u, sample_weights = sim.w)
    linkages = network.linkages_flat
    n_draws = size(linkages, 1)
    N_hat, _ = concentrate_N_s(vec(sum(linkages; dims = 1)), n_draws; target = G_target,
                               q = french_variety_share(linkages))
    beta, n_fail, _ = cell_cloglog_moment(linkages, N_hat; perms = sim.perms)
    return beta, N_hat, n_fail
end

# A world that throws is a failed world, its reason the first 80 characters of the error.
_failure(err) = (ok = false, reason = Symbol("error: ", first(sprint(showerror, err), 80)))
function _count_failures(results)
    failures = Dict{Symbol, Int}()
    for result in results
        result.ok || (failures[result.reason] = get(failures, result.reason, 0) + 1)
    end
    return failures
end

##################### Step 2: the weight matrix ###################

# One world of Step 2: its own β, the β the model simulates at the estimate once T, the
# aggregates and N are re-calibrated on the world, and their gap e_b.
function _world_residual(theta_hat, N_hat, ref, seed_w::Int, seed_s::Int)
    try
        world = draw_world(theta_hat, N_hat, seed_w)
        world.ok        || return (ok = false, reason = :world_fit)
        world.ref_empty && return (ok = false, reason = :empty_ref)
        world_profile = profile_world(ref, ref.alpha, ref.kappa, world)
        world_profile.converged   || return (ok = false, reason = :T_inversion)
        beta_sim, N_world, n_fail_sim = simulate_beta(world_theta(ref, ref.alpha, ref.kappa, world_profile),
                                   world.G, world_simulator(seed_s))
        all(isfinite, beta_sim) || return (ok = false, reason = :sim_fit)
        return (ok = true, reason = :none, e = world.beta .- beta_sim, beta_world = world.beta,
                beta_sim = beta_sim, N = N_world, n_fail = n_fail_sim, G = world.G)
    catch err
        return _failure(err)
    end
end

"""
    world_weight_matrix(theta_hat; B = 1000, dest, seed_world, seed_sim) -> (W, Omega, info)

Ω̃ = Cov(e_b) over B fake worlds at `theta_hat` and W = Ω̃⁻¹, on block 4 only (see the
header). Writes `Omega_world.npy`, `W_world.npy`, `world_residuals.npz` and
`world_weight_report.txt` under `dest`. Failed worlds are excluded and counted by reason.
"""
function world_weight_matrix(theta_hat::Vector{Float64}; B::Int = 1000, dest::AbstractString,
                             seed_world::Int = 11_000_000, seed_sim::Int = 12_000_000)
    mkpath(dest)
    N_hat = granular_report(theta_hat).N_hat
    ref   = world_reference(theta_hat)
    println("Fake worlds for the weight matrix: B = $B at θ̂ (N̂_s = $N_hat)")
    results  = pmap(b -> _world_residual(theta_hat, N_hat, ref, seed_world + b, seed_sim + b), 1:B)
    usable = [r for r in results if r.ok]
    failures = _count_failures(results)
    n_block4 = length(BLOCK_RANGES[4])
    length(usable) > n_block4 + 1 || error("world_weight_matrix: only $(length(usable)) of $B worlds usable ($failures)")

    residuals     = permutedims(reduce(hcat, (r.e for r in usable)))           # n_ok × n_block4
    Omega = cov(residuals; dims = 1)
    W     = _sym_inv(Omega)
    e_bar = vec(mean(residuals; dims = 1))
    _, simulated = full_SMM(theta_hat)
    e_data = vec(empirical_moments)[BLOCK_RANGES[4]] .- moments_to_vec(simulated)[BLOCK_RANGES[4]]

    NPZ.npzwrite(joinpath(dest, "Omega_world.npy"), Omega)
    NPZ.npzwrite(joinpath(dest, "W_world.npy"), W)
    rows_of(f) = permutedims(reduce(hcat, (f(r) for r in usable)))
    NPZ.npzwrite(joinpath(dest, "world_residuals.npz"),
                 Dict("e" => residuals, "beta_world" => rows_of(r -> r.beta_world),
                      "beta_sim" => rows_of(r -> r.beta_sim), "N_hat_world" => rows_of(r -> r.N),
                      "G_world" => rows_of(r -> r.G),
                      "N_hat" => N_hat, "e_data" => e_data))

    labels = MOMENT_LABELS[collect(BLOCK_RANGES[4])]
    open(joinpath(dest, "world_weight_report.txt"), "w") do io
        for out in (stdout, io)
            println(out, "="^72, "\nWEIGHT MATRIX FROM FAKE WORLDS (block 4 only)  ", now(), "\n", "="^72)
            @printf(out, "  worlds: %d drawn, %d usable\n", B, length(usable))
            for (r, v) in failures
                @printf(out, "    excluded (%s): %d\n", string(r), v)
            end
            @printf(out, "\n  %-24s %12s %12s %10s %12s\n", "moment", "mean e_b", "sd e_b", "t(mean)", "e_data")
            for i in 1:n_block4
                sd = sqrt(Omega[i, i])
                @printf(out, "  %-24s %12.5f %12.5f %10.2f %12.5f\n", labels[i], e_bar[i], sd,
                        e_bar[i] / (sd / sqrt(length(usable))), e_data[i])
            end
            println(out, "\n  Ω̃ = Cov(e_b):")
            for i in 1:n_block4
                println(out, "    ", join([@sprintf("%12.4e", Omega[i, j]) for j in 1:n_block4], " "))
            end
            println(out, "  correlation:")
            for i in 1:n_block4
                println(out, "    ", join([@sprintf("%8.3f", Omega[i, j] / sqrt(Omega[i, i] * Omega[j, j]))
                                           for j in 1:n_block4], " "))
            end
            n_block4 == N_TAU + 1 && println(out, "\n  Block 4 has $n_block4 entries for $n_block4 parameters (α, κ): exactly " *
                                           "identified, so W does not move the point estimate.")
            println(out, "\n  A mean residual far from zero (|t| ≫ 2) means the continuum simulator does not\n" *
                         "  reproduce, on average, the β of a finite world at the same parameters: the\n" *
                         "  estimator is biased there, and Step 4's bias column measures what it does to θ̂.")
        end
    end
    return W, Omega, (n_ok = length(usable), failures = failures, e_bar = e_bar, N_hat = N_hat)
end

##################### Step 4: the whole chain in each world ###################

# Nelder–Mead start: one vertex per coordinate, displaced by `steps`.
struct WorldSimplexer <: Optim.Simplexer
    steps::Vector{Float64}
end
function Optim.simplexer(world_simplexer::WorldSimplexer, x0::AbstractArray)
    vertices = [copy(x0) for _ in 1:(length(x0) + 1)]
    for i in eachindex(x0)
        vertices[i + 1][i] += world_simplexer.steps[i]
    end
    return vertices
end

"""
    estimate_world(world, ref, W, sim; max_iter)
        -> (ok, alpha, kappa, T, A, chi_L, chi_s, N, loss, iterations, theta_vec)

The estimator applied to one fake world. For each candidate (α, κ): T re-inverted on the
world's γ and blocks 1–3 (T and the aggregates profiled), N calibrated on its G, β simulated on the world's
simulator; criterion `e' W e`, `e = β_world − β_sim`. Nelder–Mead over (α, κ) from the
estimate, inside the estimation's κ box and every α_k ∈ [1e-4, ALPHA_MAX]; the α block
is read sorted, as in the estimation (α_1 ≤ … ≤ α_K). T and the Newton start of (χ^L, χ^s) are warm-started from the last converged
evaluation of the world; the fixed point is unique, so this changes cost, not values.
"""
function estimate_world(world, ref, W::AbstractMatrix, sim; max_iter::Int = 150)
    n_alpha = N_TAU
    T_warm = Ref(ref.T); coordinates_warm = Ref{Any}(nothing)
    alpha_kappa_of(x) = (sort(x[1:n_alpha]), x[n_alpha + 1])
    in_box(alpha, kappa) = KAPPA_LO <= kappa <= KAPPA_HI && all(a -> 1e-4 <= a <= ALPHA_MAX, alpha)

    function criterion(x)
        alpha, kappa = alpha_kappa_of(x)
        in_box(alpha, kappa) || return WORLD_PENALTY
        world_profile = try
            profile_world(ref, alpha, kappa, world; T_init = T_warm[], aggregate_coordinates_init = coordinates_warm[])
        catch
            return WORLD_PENALTY
        end
        world_profile.converged || return WORLD_PENALTY
        T_warm[] = world_profile.T; coordinates_warm[] = world_profile.aggregate_coordinates
        beta, _, _ = simulate_beta(world_theta(ref, alpha, kappa, world_profile), world.G, sim)
        all(isfinite, beta) || return WORLD_PENALTY
        residual = world.beta .- beta
        return dot(residual, W * residual)
    end

    start_point    = vcat(ref.alpha, ref.kappa)
    steps = vcat(0.1 .* max.(abs.(ref.alpha), 0.05), 0.1)
    optimum = Optim.optimize(criterion, start_point, NelderMead(initial_simplex = WorldSimplexer(steps)),
                         Optim.Options(iterations = max_iter))
    alpha, kappa = alpha_kappa_of(Optim.minimizer(optimum))
    min_loss = Optim.minimum(optimum)
    world_profile = profile_world(ref, alpha, kappa, world; T_init = T_warm[], aggregate_coordinates_init = coordinates_warm[])
    theta_vec = world_theta(ref, alpha, kappa, world_profile)
    _, N, _ = simulate_beta(theta_vec, world.G, sim)
    return (ok = world_profile.converged && min_loss < WORLD_PENALTY, alpha = alpha, kappa = kappa, T = world_profile.T,
            A = world_profile.A, chi_L = world_profile.chi_L, chi_s = world_profile.chi_s, N = N, loss = min_loss,
            iterations = Optim.iterations(optimum), theta_vec = theta_vec)
end

"""
    incidence_stats(params; regime = :estimate, radii = (100.0, 200.0)) -> NamedTuple

Closed-form geography of the upstream response at `params` (analytical prices, the
expected network): for every buyer the flows `X_lr` (R × R_d) to each upstream ZE, then the
purchase-weighted mean sourcing distance (km) and the local shares within each radius,
aggregated over buyers with their upstream purchases (`mean_*`) and at the median buyer
(`med_*`). `:equal_T` sets every non-empty cell's T to one (`Equal comparative
advantage`), `:no_trade_cost` sets τ ≡ 1; prices are re-solved in both. The flows are
DOMESTIC (the shares are of French purchases).

The counterfactuals are about the reorganisation of the DOMESTIC network, so the foreign
margin is neutralised: each buyer's import share f_rs = Φ^F_s/Φ_rs is held at its
estimate. The regime's foreign scale is buyer-specific,
`Φ^F_rs = Φ^F_s · Φ^dom_rs(regime) / Φ^dom_rs(estimate)`, with Φ^dom the French part of
Φ, so f_rs, the leakage abroad and the sector mix of the domestic bill do not move for
foreign reasons; the within-sector geography never depends on Φ^F. This also makes
`:equal_T` independent of the reference area that fixes the units of T and Φ^F.
"""
function incidence_stats(params::AbstractVector; regime::Symbol = :estimate,
                         radii = (100.0, 200.0))
    # (not called `theta`: that is the Fréchet shape used below)
    chi_L, chi_s, A, alpha, T_vec, PhiF = unpack_params(params)
    T_ls = reshape(collect(T_vec), S, R)
    tau_est = build_tau(alpha)
    if regime !== :estimate && any(>(0), PhiF)
        # hold f_rs at the estimate: scale Φ^F with the French term of each buyer
        Phi_domestic_estimate = compute_Phi(T_ls, tau_est)
    end
    regime === :equal_T && (T_ls = Float64.(T_ls .> EMPTY_T_CUT))
    tau = regime === :no_trade_cost ? ones(R, R_downstream) : tau_est
    if regime !== :estimate && any(>(0), PhiF)
        Phi_domestic_regime = compute_Phi(T_ls, tau)
        PhiF = [PhiF[s] > 0 && Phi_domestic_estimate[s, dr] > 0 ? PhiF[s] * Phi_domestic_regime[s, dr] / Phi_domestic_estimate[s, dr] : 0.0
                for s in 1:S, dr in 1:R_downstream]
    end
    prices  = compute_prices_analytical(chi_L, chi_s, A, T_ls, tau; PhiF = PhiF)
    X_lr = zeros(R, R_downstream)
    for g in 1:n_good
        s = GOOD_S[g]; l = GOOD_R[g]
        T_ls[s, l] > 0 || continue
        for dr in 1:R_downstream
            phi_sd = prices.Phi[s, dr]
            phi_sd < 1e-300 && continue
            omega_sr = chi_s[s] * (prices.P_sr[s, dr] / prices.P_r[dr])^(1 - nu) *
                 (prices.P_r[dr] / prices.c_r[dr])^(1 - lambda) * (1 - chi_L) * prices.mu * prices.Y_r[dr]
            X_lr[l, dr] += T_ls[s, l] * (W_RS_FLAT[g] * tau[l, dr])^(-theta) / phi_sd * omega_sr
        end
    end
    distance_km  = exp.(LOG_DIST_DOWNSTREAM)
    purchases_r  = vec(sum(X_lr; dims = 1))
    mean_distance_r = [sum(X_lr[:, dr] .* distance_km[:, dr]) / purchases_r[dr] for dr in 1:R_downstream]
    local_share_r  = [[sum(X_lr[distance_km[:, dr] .<= radius, dr]) / purchases_r[dr] for dr in 1:R_downstream] for radius in radii]
    buyer_weight = purchases_r ./ sum(purchases_r)
    PhiF_sr = PhiF isa AbstractMatrix ? PhiF : [PhiF[s] for s in 1:S, dr in 1:R_downstream]
    return (mean_dist = dot(buyer_weight, mean_distance_r), med_dist = median(mean_distance_r),
            mean_local = [dot(buyer_weight, shares) for shares in local_share_r], med_local = [median(shares) for shares in local_share_r],
            import_share = PhiF_sr ./ prices.Phi)
end

const CF_REGIMES = (:estimate, :equal_T, :no_trade_cost)
const CF_STATS   = ("mean_dist_km", "med_dist_km", "mean_local_100km", "mean_local_200km",
                    "med_local_100km", "med_local_200km")
counterfactual_labels() = [string(rg, ":", nm) for rg in CF_REGIMES for nm in CF_STATS]

"""
    counterfactual_row(params) -> `incidence_stats` under the three regimes, flattened
"""
function counterfactual_row(params::AbstractVector)
    row_values = Float64[]
    for regime in CF_REGIMES
        stats = incidence_stats(params; regime = regime)
        append!(row_values, [stats.mean_dist, stats.med_dist, stats.mean_local..., stats.med_local...])
    end
    return row_values
end

# One world of Step 4: the whole estimation chain on the world, and the counterfactuals at
# the world's estimate.
function _world_estimate(theta_hat, N_hat, ref, W, seed_w::Int, seed_s::Int, max_iter::Int)
    try
        world = draw_world(theta_hat, N_hat, seed_w)
        world.ok        || return (ok = false, reason = :world_fit)
        world.ref_empty && return (ok = false, reason = :empty_ref)
        estimate = estimate_world(world, ref, W, world_simulator(seed_s); max_iter = max_iter)
        estimate.ok || return (ok = false, reason = :estimation)
        counterfactuals = try counterfactual_row(estimate.theta_vec) catch; fill(NaN, length(counterfactual_labels())) end
        return (ok = true, reason = :none, estimate = estimate, counterfactuals = counterfactuals)
    catch err
        return _failure(err)
    end
end

"""
    world_inference(theta_hat, W; B = 500, dest, tcol_names, ...) -> NamedTuple

Parametric bootstrap at `theta_hat` (see the header). For α, κ, N_s, T (levels and
logs) and A: estimate, mean over worlds, bias = mean − estimate, sd, 2.5/97.5%
quantiles; the same for the within-sector sd of log T and the counterfactuals of `counterfactual_row` (levels
and changes against the estimated regime). Writes `world_inference_draws.npz` and
`world_inference.txt` under `dest`.
"""
function world_inference(theta_hat::Vector{Float64}, W::AbstractMatrix; B::Int = 500,
                         dest::AbstractString,
                         tcol_names::AbstractVector{<:AbstractString},
                         sector_names::AbstractVector{<:AbstractString} = string.(1:S),
                         seed_world::Int = 21_000_000, seed_sim::Int = 22_000_000,
                         max_iter::Int = 150, label::AbstractString = "")
    mkpath(dest)
    N_hat = granular_report(theta_hat).N_hat
    ref   = world_reference(theta_hat)
    W     = Matrix{Float64}(W)
    println("Fake worlds for inference: B = $B at θ̂ (Nelder–Mead over (α, κ), ≤ $max_iter iterations)")
    results  = pmap(b -> _world_estimate(theta_hat, N_hat, ref, W, seed_world + b, seed_sim + b, max_iter), 1:B)
    usable = [r for r in results if r.ok]
    failures = _count_failures(results)
    n_ok = length(usable)
    n_ok >= 2 || error("world_inference: only $n_ok of $B worlds usable ($failures)")

    # ── parameter draws ──────────────────────────────────────────────────────
    T_labels = [((flat_position - 1) % T_COL_DIM + 1 == FOREIGN_COL ? "PhiF[$(sector_names[(flat_position - 1) ÷ T_COL_DIM + 1])]" :
              "T[$(sector_names[(flat_position - 1) ÷ T_COL_DIM + 1])-$(tcol_names[(flat_position - 1) % T_COL_DIM + 1])]")
             for flat_position in findall(T_MASK)]
    T_reduced(T) = (v = vec(permutedims(T))[T_MASK]; v[v .<= EMPTY_T_CUT] .= NaN; v)
    labels = vcat(["alpha[$i]" for i in 1:N_TAU], ["kappa"], ["N_s[$s]" for s in sector_names],
               T_labels, ["A[$i]" for i in 1:R_downstream],
               ["chi_L"], ["chi_s[$s]" for s in sector_names])
    draw_row(alpha, kappa, N, T, A, chi_L, chi_s) = vcat(alpha, kappa, Float64.(N), T_reduced(T), A, chi_L, chi_s)
    param_draws   = permutedims(reduce(hcat, (draw_row(r.estimate.alpha, r.estimate.kappa, r.estimate.N, r.estimate.T, r.estimate.A,
                                       r.estimate.chi_L, r.estimate.chi_s) for r in usable)))
    estimate_row = draw_row(ref.alpha, ref.kappa, N_hat, ref.T, ref.A, ref.chi_L, ref.chi_s)
    T_columns  = findall(l -> startswith(l, "T["), labels)

    # ── the dispersion of log T ──────────────────────────────────────────────
    log_or_nan(values) = values > EMPTY_T_CUT ? log(values) : NaN
    log_T_dispersion(T) = [(v = filter(isfinite, log_or_nan.(T[s, SECTOR_DOM_COLS[s]])); length(v) > 1 ? std(v) : NaN) for s in 1:S]
    dispersion_draws = permutedims(reduce(hcat, (log_T_dispersion(r.estimate.T) for r in usable)))

    # ── counterfactuals: levels and changes against the estimated regime ─────
    cf_lbl = counterfactual_labels()
    counterfactual_draws     = permutedims(reduce(hcat, (r.counterfactuals for r in usable)))
    cf_est = counterfactual_row(theta_hat)
    n_stats     = length(CF_STATS)
    change(draws)   = hcat([draws[:, ((k - 1) * n_stats + 1):(k * n_stats)] .- draws[:, 1:n_stats] for k in 2:length(CF_REGIMES)]...)
    change_lbl   = ["Δ " * l for l in cf_lbl[(n_stats + 1):end]]
    change_draws    = change(counterfactual_draws); change_est = vec(change(permutedims(cf_est)))

    NPZ.npzwrite(joinpath(dest, "world_inference_draws.npz"),
                 Dict("params" => param_draws, "estimate" => estimate_row,
                      "disp_logT" => dispersion_draws, "counterfactuals" => counterfactual_draws, "counterfactuals_est" => cf_est,
                      "loss" => [r.estimate.loss for r in usable], "iters" => [r.estimate.iterations for r in usable],
                      "thetas" => permutedims(reduce(hcat, (r.estimate.theta_vec for r in usable)))))
    write(joinpath(dest, "world_inference_labels.txt"), join(labels, "\n") * "\n")

    function print_block(io, title, names, draws, estimate_values)
        println(io, "\n--- $title ---")
        @printf(io, "  %-30s %11s %11s %11s %11s %11s %11s\n", "", "estimate", "mean", "bias", "sd", "q2.5", "q97.5")
        for j in eachindex(names)
            values = filter(isfinite, draws[:, j])
            isempty(values) && continue
            quantiles = quantile(values, [0.025, 0.975])
            @printf(io, "  %-30s %11.4g %11.4g %11.4g %11.4g %11.4g %11.4g\n",
                    names[j], estimate_values[j], mean(values), mean(values) - estimate_values[j], std(values), quantiles[1], quantiles[2])
        end
    end

    open(joinpath(dest, "world_inference.txt"), "w") do io
        for out in (stdout, io)
            println(out, "="^96, "\nINFERENCE FROM FAKE WORLDS (whole chain re-estimated in each)",
                    isempty(label) ? "" : "  [$label]", "  ", now(), "\n", "="^96)
            @printf(out, "  worlds: %d drawn, %d usable; median Nelder–Mead iterations %d\n", B, n_ok,
                    round(Int, median([r.estimate.iterations for r in usable])))
            for (r, v) in failures
                @printf(out, "    excluded (%s): %d\n", string(r), v)
            end
            println(out, "  sd = dispersion across worlds; CI = 2.5/97.5% quantiles; bias = mean − estimate.")
            other_columns = setdiff(eachindex(labels), T_columns)
            print_block(out, "α, κ, N_s, A, χ^L, χ^s", labels[other_columns], param_draws[:, other_columns], estimate_row[other_columns])
            print_block(out, "T (levels, per-sector reference = 1)", labels[T_columns], param_draws[:, T_columns], estimate_row[T_columns])
            print_block(out, "log T", labels[T_columns], log.(param_draws[:, T_columns]), log.(estimate_row[T_columns]))
            share_empty = vec(mean(isnan.(param_draws[:, T_columns]); dims = 1))
            if any(>(0), share_empty)
                println(out, "\n--- share of worlds in which the area wins nothing (T set to 0; excluded above) ---")
                for j in findall(>(0), share_empty)
                    @printf(out, "  %-30s %8.3f\n", labels[T_columns][j], share_empty[j])
                end
            end
            print_block(out, "within-sector sd of log T", ["sector $s" for s in sector_names], dispersion_draws, log_T_dispersion(ref.T))
            print_block(out, "counterfactuals (closed form, levels)", cf_lbl, counterfactual_draws, cf_est)
            print_block(out, "counterfactuals (change against the estimated regime)", change_lbl, change_draws, change_est)
        end
    end
    return (draws = param_draws, labels = labels, estimate = estimate_row, n_ok = n_ok, failures = failures)
end
