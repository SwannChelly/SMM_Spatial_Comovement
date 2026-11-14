##### sensitivity.jl — moment sensitivity at an estimate: Jacobian and one-parameter profiles #####
#
# Two objects, both written under `<step>/sensitivity/` and plotted by the Python
# reporting (report_lib):
#
#   1. JACOBIAN `sensitivity_jacobian`. Central finite differences of the masked moment
#      vector with COMMON RANDOM NUMBERS: the + and − evaluations of a column share the
#      productivity draws, the block-4 network permutations and the variety counts
#      (N̂_s pinned at the base value of that draw set). It is computed on the
#      estimation draws (`J_est`, the derivative of the surface the optimiser actually
#      minimised) and on K independent draw sets (randomised Sobol nets + their own
#      permutations, `world_simulator`), whose mean is `J` and whose dispersion `J_sd`
#      measures the Monte-Carlo noise of each entry.
#
#      Why the noise matters: at fixed draws every indicator-based moment (q̂ → Ḡ_s(0),
#      has_sup → β_d, β_κ) is a STEP function of θ. A central difference of width 2h
#      counts the winners that switch inside the window and divides by h, so its
#      expectation is the derivative of the smooth averaged moment but its variance
#      grows like 1/h. Hence relative steps of order 1e-2 rather than small ones, and a
#      noise-to-signal reading (`J_sd/√K` against `|J|`) before an entry is interpreted.
#
#      Columns and the derivative each one carries:
#        α_k, κ    TOTAL derivative along the estimator's manifold: T and the
#                  aggregates re-profiled at θ ± h (`profile_theta`).
#                  Holding T fixed would let α move γ, which the inversion pins in
#                  estimation, and suggest a spurious identification of α by γ.
#        T[s,a]    partial derivative, T free (every identified, non-reference area).
#        aggregate partial derivative of χ^L, χ^s[2:S], A[2:R_d] (the normalised-out
#                  χ^s[1] and A[1] are gauge directions and are skipped).
#        N_s       unit difference in varieties, (m(N+1) − m(N−1))/2 (forward at N = 1),
#                  other sectors pinned. Only blocks 4 and 6 move: the value blocks
#                  are computed on the full draw set and carry no N_s.
#      Log steps (θ·e^{±h}) for the positive parameters, an additive step for κ; the
#      stored derivative is in LEVELS (∂m/∂θ), N_s in varieties.
#
#   2. PROFILES `sensitivity_profiles` — Catherine et al.'s Figure D.1. One parameter at
#      a time is swept over a grid around the estimate, everything else held at θ̂, and
#      the whole moment vector is recorded at each point, evaluated as the estimator
#      evaluates it: for α_k and κ, T and the aggregates re-profiled and N̂_s re-calibrated by the bisection
#      (so its integer jumps are part of the curve); for N_s, the other sectors' counts
#      pinned. Each point is computed on the estimation draws (the curve the optimiser
#      saw) and on the same K independent draw sets (the band). The global sweep shows
#      what a derivative cannot: flat stretches, non-monotonicity, and whether the curve
#      crosses the data moment inside the grid.
#
# Seeds: draw set k ≥ 1 is `world_simulator(seed + k)`, disjoint from the fake worlds'.

using Printf
using Dates

const SENS_SEED = 41_000_000

"""
    sens_drawset(k; seed) -> (u, w, perms)

k = 0: the estimation draws (`U_DRAWS`, flat weights, the cached `cell_perms`).
k ≥ 1: an independent randomised Sobol set with its own block-4 permutations.
"""
sens_drawset(k::Int; seed::Int = SENS_SEED) =
    k == 0 ? (u = U_DRAWS, w = SAMPLE_WEIGHTS, perms = nothing) : world_simulator(seed + k)

"""
    sens_moments(theta, ds; N_fixed=nothing) -> (m, N_hat)

The masked moment vector at `theta` on draw set `ds`; N̂_s re-calibrated unless pinned.
"""
function sens_moments(theta::AbstractVector, draw_set; N_fixed::Union{Nothing, Vector{Int}} = nothing)
    network = solve_network(theta; u_draws = draw_set.u, sample_weights = draw_set.w)
    moments = compute_moments(network, theta; N_fixed = N_fixed, perms = draw_set.perms)
    return moments_to_vec(moments), moments.N_hat
end

"""
    sens_columns(theta; include_T=true, include_aggregates=true) -> Vector of NamedTuples

The Jacobian's parameter axis: (label, kind, index) with `kind ∈ (:aggregate, :alpha,
:kappa, :T, :N)`; `index` is the position in θ, or the sector for `:N`.
"""
function sens_columns(theta::AbstractVector; include_T::Bool = true, include_aggregates::Bool = true)
    columns = NamedTuple{(:label, :kind, :index), Tuple{String, Symbol, Int}}[]
    if include_aggregates
        push!(columns, (label = PARAM_LABELS[1], kind = :aggregate, index = 1))
        for i in 3:(S + 1)
            push!(columns, (label = PARAM_LABELS[i], kind = :aggregate, index = i))
        end
        for i in (S + 3):(S + R_downstream + 1)
            push!(columns, (label = PARAM_LABELS[i], kind = :aggregate, index = i))
        end
    end
    for i in alpha_range()
        push!(columns, (label = PARAM_LABELS[i], kind = :alpha, index = i))
    end
    push!(columns, (label = "kappa", kind = :kappa, index = kappa_index()))
    if include_T
        for reduced_position in T_ID_REDUCED
            i = t_param_start() - 1 + reduced_position
            push!(columns, (label = PARAM_LABELS[i], kind = :T, index = i))
        end
    end
    for s in 1:S
        push!(columns, (label = "N_s[$(SECTOR_NAMES[s])]", kind = :N, index = s))
    end
    return columns
end

"""
    sens_profile(x) -> `profile_theta(x)` from a COLD start

The warm cache of the profile makes the converged T* depend, at the
inversion tolerance, on whatever the worker solved last; a finite difference divides
that by the step. Clearing them makes every sensitivity output a function of θ alone.
"""
function sens_profile(theta_vec::AbstractVector)
    PROFILE_WARM_CACHE[] = nothing
    return profile_theta(theta_vec)
end

# θ at a continuous perturbation of column `c`; `nothing` when the profile fails.
function _sens_theta(theta::AbstractVector, column, sign::Int, h::Float64)
    theta_shifted = collect(Float64, theta)
    if column.kind === :kappa
        theta_shifted[column.index] += sign * h
    else
        theta_shifted[column.index] *= exp(sign * h)
    end
    if column.kind in (:alpha, :kappa)
        profile = sens_profile(theta_shifted)
        profile.converged || return nothing
        return profile.theta_vec
    end
    return theta_shifted
end

function _sens_column_task(theta, column, k::Int, N_base::Vector{Int}, h::Float64, seed::Int)
    draw_set = sens_drawset(k; seed = seed)
    n_moments = sum(MOMENT_MASK)
    try
        if column.kind === :N
            s = column.index
            N_up = copy(N_base); N_up[s] += 1
            N_down = copy(N_base); N_down[s] = max(N_base[s] - 1, 1)
            m_up, _ = sens_moments(theta, draw_set; N_fixed = N_up)
            m_down, _ = sens_moments(theta, draw_set; N_fixed = N_down)
            return (m_up .- m_down) ./ (N_up[s] - N_down[s])
        end
        theta_plus = _sens_theta(theta, column, +1, h)
        theta_minus = _sens_theta(theta, column, -1, h)
        (theta_plus === nothing || theta_minus === nothing) && return fill(NaN, n_moments)
        m_plus, _ = sens_moments(theta_plus, draw_set; N_fixed = N_base)
        m_minus, _ = sens_moments(theta_minus, draw_set; N_fixed = N_base)
        dtheta = column.kind === :kappa ? 2h : theta[column.index] * (exp(h) - exp(-h))
        return (m_plus .- m_minus) ./ dtheta
    catch err
        @warn "sensitivity column $(column.label), draw set $k failed: $(sprint(showerror, err))"
        return fill(NaN, n_moments)
    end
end

_nanmean(x) = (finite = filter(isfinite, x); isempty(finite) ? NaN : mean(finite))
_nanstd(x)  = (finite = filter(isfinite, x); length(finite) < 2 ? NaN : std(finite))

"""
    sensitivity_jacobian(theta; K=10, h=0.02, include_T=true, include_aggregates=true,
                         seed=SENS_SEED, dest) -> NamedTuple

See the header. Writes `jacobian.npz` (`J`, `J_sd`, `J_est`, `J_draws` (K × moments ×
parameters), `m_base`, `m_base_draws`, `theta`, `N_base`, `kind`, `h`, `K`) and
`jacobian_param_labels.txt` / `jacobian_moment_labels.txt` under `dest`.
"""
function sensitivity_jacobian(theta::AbstractVector; K::Int = 10, h::Float64 = 0.02,
                              include_T::Bool = true, include_aggregates::Bool = true,
                              seed::Int = SENS_SEED, dest::AbstractString)
    mkpath(dest)
    theta_vec = collect(Float64, theta)
    columns = sens_columns(theta_vec; include_T = include_T, include_aggregates = include_aggregates)
    n_params, n_moments = length(columns), sum(MOMENT_MASK)
    println("Sensitivity Jacobian: $n_params columns × $(K + 1) draw sets (estimation + $K), step $h")

    base_moments = pmap(k -> sens_moments(theta_vec, sens_drawset(k; seed = seed)), 0:K)
    N_base = [draw_result[2] for draw_result in base_moments]
    tasks = [(j, k) for k in 0:K for j in 1:n_params]
    column_derivatives = pmap(t -> _sens_column_task(theta_vec, columns[t[1]], t[2], N_base[t[2] + 1], h, seed), tasks)

    J_all = fill(NaN, K + 1, n_moments, n_params)
    for ((j, k), derivative) in zip(tasks, column_derivatives)
        J_all[k + 1, :, j] .= derivative
    end
    J_est   = J_all[1, :, :]
    J_draws = J_all[2:end, :, :]
    J    = [_nanmean(J_draws[:, i, j]) for i in 1:n_moments, j in 1:n_params]
    J_sd = [_nanstd(J_draws[:, i, j])  for i in 1:n_moments, j in 1:n_params]

    NPZ.npzwrite(joinpath(dest, "jacobian.npz"),
                 Dict("J" => J, "J_sd" => J_sd, "J_est" => J_est, "J_draws" => J_draws,
                      "m_base" => base_moments[1][1], "m_base_draws" => permutedims(reduce(hcat, (draw_result[1] for draw_result in base_moments[2:end]))),
                      "theta" => [column.kind === :N ? Float64(N_base[1][column.index]) : theta_vec[column.index] for column in columns],
                      "N_base" => Float64.(N_base[1]), "h" => [h], "K" => [Float64(K)],
                      "kind" => [Float64(findfirst(==(column.kind), (:aggregate, :alpha, :kappa, :T, :N))) for column in columns]))
    write(joinpath(dest, "jacobian_param_labels.txt"), join([column.label for column in columns], "\n") * "\n")
    write(joinpath(dest, "jacobian_moment_labels.txt"), join(MOMENT_LABELS, "\n") * "\n")
    n_nan = count(!isfinite, J)
    println("  written to $dest ($(n_nan) non-finite entries of $(length(J)))")
    return (J = J, J_sd = J_sd, J_est = J_est, labels = [column.label for column in columns])
end

# ── profiles ────────────────────────────────────────────────────────────────

"""
    profile_grid(theta, c; G=15, span=2.0, kappa_span=0.5) -> Vector

α_k: `α̂ · span^{range(−1, 1)}`, inside `[1e-4, ALPHA_MAX]`. κ: `κ̂ ± kappa_span`, inside
the κ box. N_s: the integers `round(N̂ · span^{range(−1, 1)})` in `[1, N_HI_CAP[s]]`. The
estimate itself is always on the grid.
"""
function profile_grid(theta::AbstractVector, column, N_hat::Vector{Int}; G::Int = 15,
                      span::Float64 = 2.0, kappa_span::Float64 = 0.5)
    grid_steps = range(-1, 1; length = G)
    if column.kind === :alpha
        alpha_hat = theta[column.index]
        return sort(unique(vcat(alpha_hat, clamp.(alpha_hat .* span .^ grid_steps, 1e-4, ALPHA_MAX))))
    elseif column.kind === :kappa
        kappa = theta[column.index]
        return sort(unique(vcat(kappa, clamp.(kappa .+ kappa_span .* grid_steps, KAPPA_LO, KAPPA_HI))))
    end
    N_hat_s = N_hat[column.index]
    return sort(unique(vcat(N_hat_s, clamp.(round.(Int, N_hat_s .* span .^ grid_steps), 1, N_HI_CAP[column.index]))))
end

function _profile_task(theta, column, grid, k::Int, N_base::Vector{Int}, seed::Int)
    draw_set = sens_drawset(k; seed = seed)
    n_moments = sum(MOMENT_MASK)
    moments_on_grid = fill(NaN, length(grid), n_moments)
    N_on_grid = fill(NaN, length(grid), S)
    converged = falses(length(grid))
    if column.kind === :N
        # one Ricardian solve per draw set: the network does not depend on N_s
        network = solve_network(theta; u_draws = draw_set.u, sample_weights = draw_set.w)
        for (i, N_value) in enumerate(grid)
            N_pinned = copy(N_base); N_pinned[column.index] = Int(N_value)
            moments = compute_moments(network, theta; N_fixed = N_pinned, perms = draw_set.perms)
            moments_on_grid[i, :] .= moments_to_vec(moments); N_on_grid[i, :] .= N_pinned; converged[i] = true
        end
        return (moments_on_grid = moments_on_grid, N = N_on_grid, converged = converged)
    end
    for (i, value) in enumerate(grid)
        theta_point = collect(Float64, theta); theta_point[column.index] = value
        try
            profile = sens_profile(theta_point)
            profile.converged || continue
            moments, N = sens_moments(profile.theta_vec, draw_set)       # N̂_s re-calibrated, as in estimation
            moments_on_grid[i, :] .= moments; N_on_grid[i, :] .= N; converged[i] = true
        catch err
            @warn "profile $(column.label) = $value, draw set $k failed: $(sprint(showerror, err))"
        end
    end
    return (moments_on_grid = moments_on_grid, N = N_on_grid, converged = converged)
end

# The moments a parameter is meant to move (Figure D.1 pairs one moment with one
# parameter): α → the distance coefficients, κ → β_κ, N_s → Ḡ_s(0).
function profile_targets(column)
    block4 = collect(BLOCK_RANGES[4])
    column.kind === :alpha && return block4[1:(end - 1)]
    column.kind === :kappa && return [block4[end]]
    return [collect(BLOCK_RANGES[6])[column.index]]
end

"""
    sensitivity_profiles(theta; K=10, G=15, span=2.0, kappa_span=0.5, seed, dest)

See the header. One file per parameter, `profile_<label>.npz`: `grid`, `m_est`
(grid × moments, estimation draws), `m_mean` / `m_sd` (across the K independent sets),
`N_est` (N̂_s along the grid), `converged`, `estimate`, `targets` (moment indices,
1-based into the masked vector), `data` (the empirical moments). Plus
`profiles_index.txt` and `profiles_report.txt`.
"""
function sensitivity_profiles(theta::AbstractVector; K::Int = 10, G::Int = 15,
                              span::Float64 = 2.0, kappa_span::Float64 = 0.5,
                              seed::Int = SENS_SEED, dest::AbstractString)
    mkpath(dest)
    theta_vec = collect(Float64, theta)
    columns = filter(column -> column.kind in (:alpha, :kappa, :N), sens_columns(theta_vec; include_T = false, include_aggregates = false))
    base_moments = pmap(k -> sens_moments(theta_vec, sens_drawset(k; seed = seed)), 0:K)
    N_base = [draw_result[2] for draw_result in base_moments]
    grids = [profile_grid(theta_vec, column, N_base[1]; G = G, span = span, kappa_span = kappa_span) for column in columns]
    println("Sensitivity profiles: $(length(columns)) parameters × ≤ $(maximum(length.(grids))) points × $(K + 1) draw sets")
    tasks = [(j, k) for j in eachindex(columns) for k in 0:K]
    profile_results = pmap(t -> _profile_task(theta_vec, columns[t[1]], grids[t[1]], t[2], N_base[t[2] + 1], seed), tasks)

    data_moments = vec(empirical_moments)
    profile_file(l) = "profile_" * replace(l, r"[^A-Za-z0-9_]+" => "_") * ".npz"
    lines = String["MOMENT SENSITIVITY PROFILES  $(now())",
                   "  one parameter swept, the rest at the estimate; α, κ: T re-inverted and N̂ re-calibrated",
                   "  slope = central difference at the estimate (estimation draws); crosses = the model",
                   "  curve crosses the data moment inside the grid; monotone = share of grid steps with",
                   "  the slope's sign.", ""]
    push!(lines, @sprintf("  %-22s %-22s %11s %11s %11s %8s %8s", "parameter", "moment", "estimate",
                          "model", "data", "monot.", "crosses"))
    for (j, column) in enumerate(columns)
        set_results = [profile_results[i] for (i, t) in enumerate(tasks) if t[1] == j]       # k = 0..K in order
        m_est = set_results[1].moments_on_grid
        m_draws = cat((r.moments_on_grid for r in set_results[2:end])...; dims = 3)                 # grid × m × K
        m_mean = [_nanmean(m_draws[i, l, :]) for i in axes(m_draws, 1), l in axes(m_draws, 2)]
        m_sd   = [_nanstd(m_draws[i, l, :])  for i in axes(m_draws, 1), l in axes(m_draws, 2)]
        target_moments = profile_targets(column)
        estimate = column.kind === :N ? Float64(N_base[1][column.index]) : theta_vec[column.index]
        NPZ.npzwrite(joinpath(dest, profile_file(column.label)),
                     Dict("grid" => Float64.(grids[j]), "m_est" => m_est, "m_mean" => m_mean,
                          "m_sd" => m_sd, "N_est" => set_results[1].N, "converged" => Float64.(set_results[1].converged),
                          "estimate" => [estimate], "targets" => Float64.(target_moments), "data" => data_moments,
                          "kind" => [Float64(findfirst(==(column.kind), (:aggregate, :alpha, :kappa, :T, :N)))]))
        grid = Float64.(grids[j]); estimate_position = findfirst(==(estimate), grid)
        for target in target_moments
            model_curve = m_est[:, target]
            finite = isfinite.(model_curve)
            steps = diff(model_curve[finite])
            slope = (estimate_position === nothing || estimate_position == 1 || estimate_position == length(grid)) ? NaN :
                    (model_curve[estimate_position + 1] - model_curve[estimate_position - 1]) / (grid[estimate_position + 1] - grid[estimate_position - 1])
            monotone_share = isempty(steps) || !isfinite(slope) ? NaN : count(x -> sign(x) == sign(slope), steps) / length(steps)
            gap_to_data = model_curve[finite] .- data_moments[target]
            crosses = length(gap_to_data) > 1 && any(gap_to_data[1:(end - 1)] .* gap_to_data[2:end] .<= 0)
            push!(lines, @sprintf("  %-22s %-22s %11.4g %11.4g %11.4g %8.2f %8s", column.label,
                                  first(MOMENT_LABELS[target], 22), estimate,
                                  estimate_position === nothing ? NaN : model_curve[estimate_position], data_moments[target], monotone_share, string(crosses)))
        end
    end
    write(joinpath(dest, "profiles_index.txt"), join([column.label * "\t" * profile_file(column.label) for column in columns], "\n") * "\n")
    write(joinpath(dest, "profiles_report.txt"), join(lines, "\n") * "\n")
    println(join(lines, "\n"))
    return nothing
end

"""
    run_sensitivity(theta, dest; K, G, h, include_T) — both objects, never throws
"""
function run_sensitivity(theta::AbstractVector, dest::AbstractString; K::Int = 10, G::Int = 15,
                         h::Float64 = 0.02, include_T::Bool = true)
    try
        sensitivity_profiles(theta; K = K, G = G, dest = dest)
    catch err
        @warn "sensitivity profiles failed: $(sprint(showerror, err))"
    end
    try
        sensitivity_jacobian(theta; K = K, h = h, include_T = include_T, dest = dest)
    catch err
        @warn "sensitivity Jacobian failed: $(sprint(showerror, err))"
    end
    return nothing
end
