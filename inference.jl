##### inference.jl — Steps 2 and 4: weighting and inference from fake worlds #####
#
# The data covariance is not used. Both the weight matrix and the inference come from
# FAKE WORLDS drawn from the model at an estimate:
#
#   Step 2 (`world_weight_matrix`). At θ̂₁ (W = I), draw B worlds: one network of N̂_s
#   varieties per sector with i.i.d. uniform draws (NOT the estimation's Sobol design).
#   Each world yields its own "data" (β_b, γ_b, G_b, π_b), computed as on the real data.
#   T is re-inverted on γ_b with A updated on π_b, N is calibrated on G_b, β is simulated
#   at μ̂₁ with a seed proper to the world and at the estimation's m, and
#   e_b = β_b − β_sim is kept. Ω̃ = Cov(e) and W = Ω̃⁻¹, on block 4 only: γ and G are
#   reproduced exactly by the T inversion and the N bisection, so their weight does not
#   matter. No derivative is computed. When block 4 has as many entries as (α, κ) the
#   system is exactly identified and W does not move the point estimate.
#
#   Step 4 (`world_inference`). At θ̂₂, draw B worlds and re-run the WHOLE chain in each,
#   W fixed: T inversion + A update, N calibration, and a local search over (α, κ)
#   started at the estimate. The B parameter sets give, for every parameter, sd = their
#   dispersion, a 95% interval = their 2.5/97.5% quantiles, and bias = mean − estimate;
#   plus the within-sector dispersion of log T and closed-form counterfactuals
#   recomputed world by world.
#
# ── The A update ─────────────────────────────────────────────────────────────
# Downstream sales are Y_r ∝ (c_r/A_r)^ε δ_r and c_r depends on (α, κ, T) only (upstream
# wages ≡ 1), so π_r is matched by A in closed form. The update is written RELATIVE to
# the estimate,
#
#     A_b = Â · (c_r(α, κ, T_b)/c_r(α̂, κ̂, T̂)) · (π_b/π̂)^(−1/ε),      A_b ./= A_b[1],
#
# with c_r and π̂ from the analytical price solve, so a world identical to the model
# returns Â exactly. γ depends on A through Y_r, so the T inversion and the A update
# alternate until A stops moving. Ω^L and Ω^s stay at the estimate.
#
# ── Areas that come out empty ────────────────────────────────────────────────
# A world is ONE network, so an area active in the data can win nothing in it (γ_b = 0).
# On the data such an area would not be active and its T would be zero; the world does
# the same: its T is set to `EMPTY_T` and held there by `invert_T_ge`. Such draws are
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
    As = Symmetric((Matrix(A) .+ Matrix(A)') ./ 2)
    M = try
        Matrix(inv(cholesky(As)))
    catch
        F = eigen(As)
        F.vectors * Diagonal(1.0 ./ max.(F.values, F.values[end] * floor_rel)) * F.vectors'
    end
    return (M .+ M') ./ 2
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
  * `gamma` — the sourcing shares by area, summing to one per sector (the form
              `invert_T_ge` takes as `target`; `EMP_GAMMA_T_TILDE` on the data);
  * `pi`    — downstream sales shares, `DOWNSTREAM_REGIONS` order.
"""
function draw_world(theta::AbstractVector, N_hat::Vector{Int}, seed::Integer)
    N_max = maximum(N_hat)
    u = generate_draws(N_max, n_good, :mc; randomise = true, rng = MersenneTwister(seed))[1]
    W = zeros(N_max, n_good)
    for g in 1:n_good
        n = N_hat[GOOD_S[g]]
        W[1:n, g] .= 1.0 / n
        u[(n + 1):N_max, g] .= 0.5
    end
    net = solve_network(theta; u_draws = u, sample_weights = W)
    has = falses(n_good)
    @inbounds for g in 1:n_good
        has[g] = any(view(net.linkages_flat, 1:N_hat[GOOD_S[g]], g))
    end
    beta, ok = cell_cloglog_beta(Float64[has[g] for g in REG_CELL_G])
    G = [isempty(c) ? 0.0 : count(g -> !has[g], c) / length(c) for c in CELLS_OF_SECTOR]
    X_s = zeros(S)
    for g in 1:n_good
        X_s[GOOD_S[g]] += net.X_ls_flat[g]
    end
    gam = zeros(T_COL_DIM, S)
    for g in 1:n_good
        s = GOOD_S[g]
        X_s[s] > 0 && (gam[T_GATHER[GOOD_R[g]], s] += net.X_ls_flat[g] / X_s[s])
    end
    Yd = net.Y_r[DOWNSTREAM_REGIONS]
    ref_empty = any(s -> T_REF_REGION[s] > 0 && gam[T_REF_REGION[s], s] <= 0, 1:S)
    return (beta = beta, ok = ok, G = G, gamma = gam, pi = Yd ./ sum(Yd), ref_empty = ref_empty)
end

"""
    world_reference(theta_hat) -> NamedTuple

What the A update and the world parameter vectors read off the estimate: the level
vector itself (its Ω^L, Ω^s are kept), Â normalised, α̂, κ̂, T̂ by area, and the
analytical c_r and π_r at θ̂.
"""
function world_reference(theta_hat::AbstractVector)
    ΩL, Ωs, A, α, κ = normalized_head(theta_hat)
    Tp = unpack_T_par(theta_hat)
    pr = compute_prices_analytical(ΩL, Ωs, A, gather_T_to_ze(Tp, κ), build_tau(α))
    return (theta = collect(Float64, theta_hat), ΩL = ΩL, Ωs = collect(Ωs), A = collect(A),
            alpha = collect(α), kappa = Float64(κ), T = Matrix(Tp),
            c = collect(pr.c_r), pi = pr.Y_r ./ sum(pr.Y_r))
end

"""
    recalibrate_TA(ref, alpha, kappa, gamma_b, pi_b; T_init, A_init) -> (T, A, converged)

Re-invert T on a world's sourcing shares `gamma_b` and update A on its sales shares
`pi_b` (see the header), alternating until `max |Δ log A| < tol`. `converged` requires
every inversion to converge and the alternation to settle.
"""
function recalibrate_TA(ref, alpha, kappa, gamma_b::AbstractMatrix, pi_b::AbstractVector;
                        T_init::AbstractMatrix = ref.T, A_init::AbstractVector = ref.A,
                        max_outer::Int = 20, tol::Float64 = 1e-8)
    T = Matrix{Float64}(T_init)
    A = Vector{Float64}(A_init)
    for s in 1:S, c in SECTOR_T_COLS[s]
        (gamma_b[c, s] <= 0 && c != T_REF_REGION[s]) && (T[s, c] = EMPTY_T)
    end
    tau  = build_tau(alpha)
    pfac = (pi_b ./ ref.pi) .^ (-1 / epsilon)
    conv = true
    for _ in 1:max_outer
        res = invert_T_ge(alpha, ref.ΩL, ref.Ωs, A; kappa = kappa, target = gamma_b, T_init = T)
        conv &= res.converged
        T = res.T
        c = compute_prices_analytical(ref.ΩL, ref.Ωs, A, gather_T_to_ze(T, kappa), tau).c_r
        A_new = ref.A .* (c ./ ref.c) .* pfac
        A_new ./= A_new[1]
        step = maximum(abs.(log.(A_new) .- log.(A)))
        A = A_new
        step < tol && return (T = T, A = A, converged = conv)
    end
    return (T = T, A = A, converged = false)
end

"""
    world_theta(ref, alpha, kappa, T, A) -> level vector: the estimate's Ω, new α, κ, A, T
"""
function world_theta(ref, alpha, kappa, T, A)
    x = copy(ref.theta)
    x[(S + 2):(S + R_downstream + 1)] = A
    x[alpha_range()] = alpha
    x[kappa_index()] = kappa
    return with_T(x, T)
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
    net = solve_network(params; u_draws = sim.u, sample_weights = sim.w)
    L = net.linkages_flat
    m = size(L, 1)
    N_hat, _ = concentrate_N_s(vec(sum(L; dims = 1)), m; target = G_target)
    beta, n_fail, _ = cell_cloglog_moment(L, N_hat; perms = sim.perms)
    return beta, N_hat, n_fail
end

_failure(err) = (ok = false, reason = Symbol("error: ", first(sprint(showerror, err), 80)))
function _count_failures(res)
    fails = Dict{Symbol, Int}()
    for r in res
        r.ok || (fails[r.reason] = get(fails, r.reason, 0) + 1)
    end
    return fails
end

##################### Step 2: the weight matrix ###################

function _world_residual(theta_hat, N_hat, ref, seed_w::Int, seed_s::Int)
    try
        world = draw_world(theta_hat, N_hat, seed_w)
        world.ok        || return (ok = false, reason = :world_fit)
        world.ref_empty && return (ok = false, reason = :empty_ref)
        cal = recalibrate_TA(ref, ref.alpha, ref.kappa, world.gamma, world.pi)
        cal.converged   || return (ok = false, reason = :T_inversion)
        βs, Nb, nf = simulate_beta(world_theta(ref, ref.alpha, ref.kappa, cal.T, cal.A),
                                   world.G, world_simulator(seed_s))
        all(isfinite, βs) || return (ok = false, reason = :sim_fit)
        return (ok = true, reason = :none, e = world.beta .- βs, beta_world = world.beta,
                beta_sim = βs, N = Nb, n_fail = nf)
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
    res  = pmap(b -> _world_residual(theta_hat, N_hat, ref, seed_world + b, seed_sim + b), 1:B)
    good = [r for r in res if r.ok]
    fails = _count_failures(res)
    k = length(BLOCK_RANGES[4])
    length(good) > k + 1 || error("world_weight_matrix: only $(length(good)) of $B worlds usable ($fails)")

    E     = permutedims(reduce(hcat, (r.e for r in good)))           # n_ok × k
    Omega = cov(E; dims = 1)
    W     = _sym_inv(Omega)
    e_bar = vec(mean(E; dims = 1))
    _, sim = full_SMM(theta_hat)
    e_data = vec(empirical_moments)[BLOCK_RANGES[4]] .- moments_to_vec(sim)[BLOCK_RANGES[4]]

    NPZ.npzwrite(joinpath(dest, "Omega_world.npy"), Omega)
    NPZ.npzwrite(joinpath(dest, "W_world.npy"), W)
    rows_of(f) = permutedims(reduce(hcat, (f(r) for r in good)))
    NPZ.npzwrite(joinpath(dest, "world_residuals.npz"),
                 Dict("e" => E, "beta_world" => rows_of(r -> r.beta_world),
                      "beta_sim" => rows_of(r -> r.beta_sim), "N_hat_world" => rows_of(r -> r.N),
                      "N_hat" => N_hat, "e_data" => e_data))

    labels = MOMENT_LABELS[collect(BLOCK_RANGES[4])]
    open(joinpath(dest, "world_weight_report.txt"), "w") do io
        for out in (stdout, io)
            println(out, "="^72, "\nWEIGHT MATRIX FROM FAKE WORLDS (block 4 only)  ", now(), "\n", "="^72)
            @printf(out, "  worlds: %d drawn, %d usable\n", B, length(good))
            for (r, v) in fails
                @printf(out, "    excluded (%s): %d\n", string(r), v)
            end
            @printf(out, "\n  %-24s %12s %12s %10s %12s\n", "moment", "mean e_b", "sd e_b", "t(mean)", "e_data")
            for i in 1:k
                sd = sqrt(Omega[i, i])
                @printf(out, "  %-24s %12.5f %12.5f %10.2f %12.5f\n", labels[i], e_bar[i], sd,
                        e_bar[i] / (sd / sqrt(length(good))), e_data[i])
            end
            println(out, "\n  Ω̃ = Cov(e_b):")
            for i in 1:k
                println(out, "    ", join([@sprintf("%12.4e", Omega[i, j]) for j in 1:k], " "))
            end
            println(out, "  correlation:")
            for i in 1:k
                println(out, "    ", join([@sprintf("%8.3f", Omega[i, j] / sqrt(Omega[i, i] * Omega[j, j]))
                                           for j in 1:k], " "))
            end
            k == N_TAU + 1 && println(out, "\n  Block 4 has $k entries for $k parameters (α, κ): exactly " *
                                           "identified, so W does not move the point estimate.")
            println(out, "\n  A mean residual far from zero (|t| ≫ 2) means the continuum simulator does not\n" *
                         "  reproduce, on average, the β of a finite world at the same parameters: the\n" *
                         "  estimator is biased there, and Step 4's bias column measures what it does to θ̂.")
        end
    end
    return W, Omega, (n_ok = length(good), fails = fails, e_bar = e_bar, N_hat = N_hat)
end

##################### Step 4: the whole chain in each world ###################

# Nelder–Mead start: one vertex per coordinate, displaced by `steps`.
struct WorldSimplexer <: Optim.Simplexer
    steps::Vector{Float64}
end
function Optim.simplexer(S_::WorldSimplexer, x0::AbstractArray)
    v = [copy(x0) for _ in 1:(length(x0) + 1)]
    for i in eachindex(x0)
        v[i + 1][i] += S_.steps[i]
    end
    return v
end

"""
    estimate_world(world, ref, W, sim; max_iter) -> NamedTuple

The estimator applied to one fake world. For each candidate (α, κ): T re-inverted on the
world's γ with A updated on its π, N calibrated on its G, β simulated on the world's
simulator; criterion `e' W e`, `e = β_world − β_sim`. Nelder–Mead over (α, κ) from the
estimate, inside the estimation's κ box and every α_k ∈ [1e-4, ALPHA_MAX]. T and A are warm-started from the last converged evaluation of the
world; the fixed point is unique, so this changes cost, not values.
"""
function estimate_world(world, ref, W::AbstractMatrix, sim; max_iter::Int = 150)
    nα = N_TAU
    T_warm = Ref(ref.T); A_warm = Ref(ref.A)
    unpack_x(x) = (x[1:nα], x[nα + 1])
    inbox(α, κ) = KAPPA_LO <= κ <= KAPPA_HI && all(a -> 1e-4 <= a <= ALPHA_MAX, α)

    function crit(x)
        α, κ = unpack_x(x)
        inbox(α, κ) || return WORLD_PENALTY
        cal = try
            recalibrate_TA(ref, α, κ, world.gamma, world.pi; T_init = T_warm[], A_init = A_warm[])
        catch
            return WORLD_PENALTY
        end
        cal.converged || return WORLD_PENALTY
        T_warm[] = cal.T; A_warm[] = cal.A
        β, _, _ = simulate_beta(world_theta(ref, α, κ, cal.T, cal.A), world.G, sim)
        all(isfinite, β) || return WORLD_PENALTY
        e = world.beta .- β
        return dot(e, W * e)
    end

    x0    = vcat(ref.alpha, ref.kappa)
    steps = vcat(0.1 .* max.(abs.(ref.alpha), 0.05), 0.1)
    opt = Optim.optimize(crit, x0, NelderMead(initial_simplex = WorldSimplexer(steps)),
                         Optim.Options(iterations = max_iter))
    α, κ = unpack_x(Optim.minimizer(opt))
    f̂ = Optim.minimum(opt)
    cal = recalibrate_TA(ref, α, κ, world.gamma, world.pi; T_init = T_warm[], A_init = A_warm[])
    θ = world_theta(ref, α, κ, cal.T, cal.A)
    _, N, _ = simulate_beta(θ, world.G, sim)
    return (ok = cal.converged && f̂ < WORLD_PENALTY, alpha = α, kappa = κ, T = cal.T,
            A = cal.A, N = N, loss = f̂, iters = Optim.iterations(opt), theta = θ)
end

"""
    incidence_stats(params; regime = :estimate, radii = (100.0, 200.0)) -> NamedTuple

Closed-form geography of the upstream response at `params` (analytical prices, the
expected network): for every buyer the flows `X[l, r]` to each upstream ZE, then the
purchase-weighted mean sourcing distance (km) and the local shares within each radius,
aggregated over buyers with their upstream purchases (`mean_*`) and at the median buyer
(`med_*`). `:equal_T` sets every non-empty cell's T to one (`Equal comparative
advantage`), `:no_trade_cost` sets τ ≡ 1; prices are re-solved in both.
"""
function incidence_stats(params::AbstractVector; regime::Symbol = :estimate,
                         radii = (100.0, 200.0))
    # (not called `theta`: that is the Fréchet shape used below)
    ΩL, Ωs, A, α, Tvec = unpack_params(params)
    T_mat = reshape(collect(Tvec), S, R)
    regime === :equal_T && (T_mat = Float64.(T_mat .> EMPTY_T_CUT))
    tau = regime === :no_trade_cost ? ones(R, R_downstream) : build_tau(α)
    pr  = compute_prices_analytical(ΩL, Ωs, A, T_mat, tau)
    X = zeros(R, R_downstream)
    for g in 1:n_good
        s = GOOD_S[g]; l = GOOD_R[g]
        T_mat[s, l] > 0 || continue
        for dr in 1:R_downstream
            Φ = pr.Phi[s, dr]
            Φ < 1e-300 && continue
            ex = Ωs[s] * (pr.P_sr[s, dr] / pr.P_r[dr])^(1 - nu) *
                 (pr.P_r[dr] / pr.c_r[dr])^(1 - lambda) * (1 - ΩL) * pr.mu * pr.Y_r[dr]
            X[l, dr] += T_mat[s, l] * (W_RS_FLAT[g] * tau[l, dr])^(-theta) / Φ * ex
        end
    end
    dkm  = exp.(LOG_DIST_DOWNSTREAM)
    tot  = vec(sum(X; dims = 1))
    dbar = [sum(X[:, dr] .* dkm[:, dr]) / tot[dr] for dr in 1:R_downstream]
    loc  = [[sum(X[dkm[:, dr] .<= d, dr]) / tot[dr] for dr in 1:R_downstream] for d in radii]
    w = tot ./ sum(tot)
    return (mean_dist = dot(w, dbar), med_dist = median(dbar),
            mean_local = [dot(w, L) for L in loc], med_local = [median(L) for L in loc])
end

const CF_REGIMES = (:estimate, :equal_T, :no_trade_cost)
const CF_STATS   = ("mean_dist_km", "med_dist_km", "mean_local_100km", "mean_local_200km",
                    "med_local_100km", "med_local_200km")
counterfactual_labels() = [string(rg, ":", nm) for rg in CF_REGIMES for nm in CF_STATS]

"""
    counterfactual_row(params) -> `incidence_stats` under the three regimes, flattened
"""
function counterfactual_row(params::AbstractVector)
    v = Float64[]
    for rg in CF_REGIMES
        st = incidence_stats(params; regime = rg)
        append!(v, [st.mean_dist, st.med_dist, st.mean_local..., st.med_local...])
    end
    return v
end

function _world_estimate(theta_hat, N_hat, ref, W, seed_w::Int, seed_s::Int, max_iter::Int)
    try
        world = draw_world(theta_hat, N_hat, seed_w)
        world.ok        || return (ok = false, reason = :world_fit)
        world.ref_empty && return (ok = false, reason = :empty_ref)
        est = estimate_world(world, ref, W, world_simulator(seed_s); max_iter = max_iter)
        est.ok || return (ok = false, reason = :estimation)
        cf = try counterfactual_row(est.theta) catch; fill(NaN, length(counterfactual_labels())) end
        return (ok = true, reason = :none, est = est, cf = cf)
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
    res  = pmap(b -> _world_estimate(theta_hat, N_hat, ref, W, seed_world + b, seed_sim + b, max_iter), 1:B)
    good = [r for r in res if r.ok]
    fails = _count_failures(res)
    n_ok = length(good)
    n_ok >= 2 || error("world_inference: only $n_ok of $B worlds usable ($fails)")

    # ── parameter draws ──────────────────────────────────────────────────────
    t_lbl = ["T[$(sector_names[(fp - 1) ÷ T_COL_DIM + 1])-$(tcol_names[(fp - 1) % T_COL_DIM + 1])]"
             for fp in findall(T_MASK)]
    t_red(T) = (v = vec(permutedims(T))[T_MASK]; v[v .<= EMPTY_T_CUT] .= NaN; v)
    lbl = vcat(["alpha[$i]" for i in 1:N_TAU], ["kappa"], ["N_s[$s]" for s in sector_names],
               t_lbl, ["A[$i]" for i in 1:R_downstream])
    row(α, κ, N, T, A) = vcat(α, κ, Float64.(N), t_red(T), A)
    P   = permutedims(reduce(hcat, (row(r.est.alpha, r.est.kappa, r.est.N, r.est.T, r.est.A) for r in good)))
    est = row(ref.alpha, ref.kappa, N_hat, ref.T, ref.A)
    iT  = findall(l -> startswith(l, "T["), lbl)

    # ── the dispersion of log T ──────────────────────────────────────────────
    lg(x) = x > EMPTY_T_CUT ? log(x) : NaN
    disp(T) = [(v = filter(isfinite, lg.(T[s, SECTOR_T_COLS[s]])); length(v) > 1 ? std(v) : NaN) for s in 1:S]
    Dp = permutedims(reduce(hcat, (disp(r.est.T) for r in good)))

    # ── counterfactuals: levels and changes against the estimated regime ─────
    cf_lbl = counterfactual_labels()
    Cp     = permutedims(reduce(hcat, (r.cf for r in good)))
    cf_est = counterfactual_row(theta_hat)
    ns     = length(CF_STATS)
    Δ(M)   = hcat([M[:, ((k - 1) * ns + 1):(k * ns)] .- M[:, 1:ns] for k in 2:length(CF_REGIMES)]...)
    Δlbl   = ["Δ " * l for l in cf_lbl[(ns + 1):end]]
    ΔCp    = Δ(Cp); Δest = vec(Δ(permutedims(cf_est)))

    NPZ.npzwrite(joinpath(dest, "world_inference_draws.npz"),
                 Dict("params" => P, "estimate" => est,
                      "disp_logT" => Dp, "counterfactuals" => Cp, "counterfactuals_est" => cf_est,
                      "loss" => [r.est.loss for r in good], "iters" => [r.est.iters for r in good],
                      "thetas" => permutedims(reduce(hcat, (r.est.theta for r in good)))))
    write(joinpath(dest, "world_inference_labels.txt"), join(lbl, "\n") * "\n")

    function block(io, title, names, M, e)
        println(io, "\n--- $title ---")
        @printf(io, "  %-30s %11s %11s %11s %11s %11s %11s\n", "", "estimate", "mean", "bias", "sd", "q2.5", "q97.5")
        for j in eachindex(names)
            x = filter(isfinite, M[:, j])
            isempty(x) && continue
            q = quantile(x, [0.025, 0.975])
            @printf(io, "  %-30s %11.4g %11.4g %11.4g %11.4g %11.4g %11.4g\n",
                    names[j], e[j], mean(x), mean(x) - e[j], std(x), q[1], q[2])
        end
    end

    open(joinpath(dest, "world_inference.txt"), "w") do io
        for out in (stdout, io)
            println(out, "="^96, "\nINFERENCE FROM FAKE WORLDS (whole chain re-estimated in each)",
                    isempty(label) ? "" : "  [$label]", "  ", now(), "\n", "="^96)
            @printf(out, "  worlds: %d drawn, %d usable; median Nelder–Mead iterations %d\n", B, n_ok,
                    round(Int, median([r.est.iters for r in good])))
            for (r, v) in fails
                @printf(out, "    excluded (%s): %d\n", string(r), v)
            end
            println(out, "  sd = dispersion across worlds; CI = 2.5/97.5% quantiles; bias = mean − estimate.")
            head = setdiff(eachindex(lbl), iT)
            block(out, "α, κ, N_s, A", lbl[head], P[:, head], est[head])
            block(out, "T (levels, per-sector reference = 1)", lbl[iT], P[:, iT], est[iT])
            block(out, "log T", lbl[iT], log.(P[:, iT]), log.(est[iT]))
            emp = vec(mean(isnan.(P[:, iT]); dims = 1))
            if any(>(0), emp)
                println(out, "\n--- share of worlds in which the area wins nothing (T set to 0; excluded above) ---")
                for j in findall(>(0), emp)
                    @printf(out, "  %-30s %8.3f\n", lbl[iT][j], emp[j])
                end
            end
            block(out, "within-sector sd of log T", ["sector $s" for s in sector_names], Dp, disp(ref.T))
            block(out, "counterfactuals (closed form, levels)", cf_lbl, Cp, cf_est)
            block(out, "counterfactuals (change against the estimated regime)", Δlbl, ΔCp, Δest)
        end
    end
    return (draws = P, labels = lbl, estimate = est, n_ok = n_ok, fails = fails)
end
