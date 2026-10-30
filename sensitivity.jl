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
#        α_k, κ    TOTAL derivative along the estimator's manifold: T (and the head
#                  under PROFILE_HEAD) re-profiled at θ ± h (`profiled_theta_full`).
#                  Holding T fixed would let α move γ, which the inversion pins in
#                  estimation, and suggest a spurious identification of α by γ.
#        T[s,a]    partial derivative, T free (every identified, non-reference area).
#        head      partial derivative of Ω^L, Ω^s[2:S], A[2:R_d] (the normalised-out
#                  Ω^s[1] and A[1] are gauge directions and are skipped).
#        N_s       unit difference in varieties, (m(N+1) − m(N−1))/2 (forward at N = 1),
#                  other sectors pinned. Only blocks 4 and 6 move: the value blocks
#                  are computed on the full draw set and carry no N_s.
#      Log steps (θ·e^{±h}) for the positive parameters, an additive step for κ; the
#      stored derivative is in LEVELS (∂m/∂θ), N_s in varieties.
#
#   2. PROFILES `sensitivity_profiles` — Catherine et al.'s Figure D.1. One parameter at
#      a time is swept over a grid around the estimate, everything else held at θ̂, and
#      the whole moment vector is recorded at each point, evaluated as the estimator
#      evaluates it: for α_k and κ, T re-inverted and N̂_s re-calibrated by the bisection
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
function sens_moments(theta::AbstractVector, ds; N_fixed::Union{Nothing, Vector{Int}} = nothing)
    net = solve_network(theta; u_draws = ds.u, sample_weights = ds.w)
    m = compute_moments(net, theta; N_fixed = N_fixed, perms = ds.perms)
    return moments_to_vec(m), m.N_hat
end

"""
    sens_columns(theta; include_T=true, include_head=true) -> Vector of NamedTuples

The Jacobian's parameter axis: (label, kind, index) with `kind ∈ (:head, :alpha,
:kappa, :T, :N)`; `index` is the position in θ, or the sector for `:N`.
"""
function sens_columns(theta::AbstractVector; include_T::Bool = true, include_head::Bool = true)
    cols = NamedTuple{(:label, :kind, :index), Tuple{String, Symbol, Int}}[]
    if include_head
        push!(cols, (label = PARAM_LABELS[1], kind = :head, index = 1))
        for i in 3:(S + 1)
            push!(cols, (label = PARAM_LABELS[i], kind = :head, index = i))
        end
        for i in (S + 3):(S + R_downstream + 1)
            push!(cols, (label = PARAM_LABELS[i], kind = :head, index = i))
        end
    end
    for i in alpha_range()
        push!(cols, (label = PARAM_LABELS[i], kind = :alpha, index = i))
    end
    push!(cols, (label = "kappa", kind = :kappa, index = kappa_index()))
    if include_T
        for r in T_ID_REDUCED
            i = t_param_start() - 1 + r
            push!(cols, (label = PARAM_LABELS[i], kind = :T, index = i))
        end
    end
    for s in 1:S
        push!(cols, (label = "N_s[$(SECTOR_NAMES[s])]", kind = :N, index = s))
    end
    return cols
end

"""
    sens_profile(x) -> `profiled_theta_full(x)` from a COLD start

The warm caches of the T (and head) inversion make the converged T* depend, at the
inversion tolerance, on whatever the worker solved last; a finite difference divides
that by the step. Clearing them makes every sensitivity output a function of θ alone.
"""
function sens_profile(x::AbstractVector)
    T_WARM_CACHE[] = nothing
    HEAD_WARM_CACHE[] = nothing
    return profiled_theta_full(x)
end

# θ at a continuous perturbation of column `c`; `nothing` when the profile fails.
function _sens_theta(theta::AbstractVector, c, sign::Int, h::Float64)
    x = collect(Float64, theta)
    if c.kind === :kappa
        x[c.index] += sign * h
    else
        x[c.index] *= exp(sign * h)
    end
    if c.kind in (:alpha, :kappa)
        r = sens_profile(x)
        r.converged || return nothing
        return r.theta
    end
    return x
end

function _sens_column_task(theta, c, k::Int, N_base::Vector{Int}, h::Float64, seed::Int)
    ds = sens_drawset(k; seed = seed)
    n_m = sum(MOMENT_MASK)
    try
        if c.kind === :N
            s = c.index
            Nu = copy(N_base); Nu[s] += 1
            Nd = copy(N_base); Nd[s] = max(N_base[s] - 1, 1)
            mu, _ = sens_moments(theta, ds; N_fixed = Nu)
            md, _ = sens_moments(theta, ds; N_fixed = Nd)
            return (mu .- md) ./ (Nu[s] - Nd[s])
        end
        tp = _sens_theta(theta, c, +1, h)
        tm = _sens_theta(theta, c, -1, h)
        (tp === nothing || tm === nothing) && return fill(NaN, n_m)
        mp, _ = sens_moments(tp, ds; N_fixed = N_base)
        mm, _ = sens_moments(tm, ds; N_fixed = N_base)
        dθ = c.kind === :kappa ? 2h : theta[c.index] * (exp(h) - exp(-h))
        return (mp .- mm) ./ dθ
    catch err
        @warn "sensitivity column $(c.label), draw set $k failed: $(sprint(showerror, err))"
        return fill(NaN, n_m)
    end
end

_nanmean(x) = (f = filter(isfinite, x); isempty(f) ? NaN : mean(f))
_nanstd(x)  = (f = filter(isfinite, x); length(f) < 2 ? NaN : std(f))

"""
    sensitivity_jacobian(theta; K=10, h=0.02, include_T=true, include_head=true,
                         seed=SENS_SEED, dest) -> NamedTuple

See the header. Writes `jacobian.npz` (`J`, `J_sd`, `J_est`, `J_draws` (K × moments ×
parameters), `m_base`, `m_base_draws`, `theta`, `N_base`, `kind`, `h`, `K`) and
`jacobian_param_labels.txt` / `jacobian_moment_labels.txt` under `dest`.
"""
function sensitivity_jacobian(theta::AbstractVector; K::Int = 10, h::Float64 = 0.02,
                              include_T::Bool = true, include_head::Bool = true,
                              seed::Int = SENS_SEED, dest::AbstractString)
    mkpath(dest)
    θ = collect(Float64, theta)
    cols = sens_columns(θ; include_T = include_T, include_head = include_head)
    n_p, n_m = length(cols), sum(MOMENT_MASK)
    println("Sensitivity Jacobian: $n_p columns × $(K + 1) draw sets (estimation + $K), step $h")

    base = pmap(k -> sens_moments(θ, sens_drawset(k; seed = seed)), 0:K)
    N_base = [b[2] for b in base]
    tasks = [(j, k) for k in 0:K for j in 1:n_p]
    res = pmap(t -> _sens_column_task(θ, cols[t[1]], t[2], N_base[t[2] + 1], h, seed), tasks)

    J_all = fill(NaN, K + 1, n_m, n_p)
    for ((j, k), v) in zip(tasks, res)
        J_all[k + 1, :, j] .= v
    end
    J_est   = J_all[1, :, :]
    J_draws = J_all[2:end, :, :]
    J    = [_nanmean(J_draws[:, i, j]) for i in 1:n_m, j in 1:n_p]
    J_sd = [_nanstd(J_draws[:, i, j])  for i in 1:n_m, j in 1:n_p]

    NPZ.npzwrite(joinpath(dest, "jacobian.npz"),
                 Dict("J" => J, "J_sd" => J_sd, "J_est" => J_est, "J_draws" => J_draws,
                      "m_base" => base[1][1], "m_base_draws" => permutedims(reduce(hcat, (b[1] for b in base[2:end]))),
                      "theta" => [c.kind === :N ? Float64(N_base[1][c.index]) : θ[c.index] for c in cols],
                      "N_base" => Float64.(N_base[1]), "h" => [h], "K" => [Float64(K)],
                      "kind" => [Float64(findfirst(==(c.kind), (:head, :alpha, :kappa, :T, :N))) for c in cols]))
    write(joinpath(dest, "jacobian_param_labels.txt"), join([c.label for c in cols], "\n") * "\n")
    write(joinpath(dest, "jacobian_moment_labels.txt"), join(MOMENT_LABELS, "\n") * "\n")
    n_nan = count(!isfinite, J)
    println("  written to $dest ($(n_nan) non-finite entries of $(length(J)))")
    return (J = J, J_sd = J_sd, J_est = J_est, labels = [c.label for c in cols])
end

# ── profiles ────────────────────────────────────────────────────────────────

"""
    profile_grid(theta, c; G=15, span=2.0, kappa_span=0.5) -> Vector

α_k: `α̂ · span^{range(−1, 1)}`, inside `[1e-4, ALPHA_MAX]`. κ: `κ̂ ± kappa_span`, inside
the κ box. N_s: the integers `round(N̂ · span^{range(−1, 1)})` in `[1, N_HI[s]]`. The
estimate itself is always on the grid.
"""
function profile_grid(theta::AbstractVector, c, N_hat::Vector{Int}; G::Int = 15,
                      span::Float64 = 2.0, kappa_span::Float64 = 0.5)
    t = range(-1, 1; length = G)
    if c.kind === :alpha
        a = theta[c.index]
        return sort(unique(vcat(a, clamp.(a .* span .^ t, 1e-4, ALPHA_MAX))))
    elseif c.kind === :kappa
        κ = theta[c.index]
        return sort(unique(vcat(κ, clamp.(κ .+ kappa_span .* t, KAPPA_LO, KAPPA_HI))))
    end
    n = N_hat[c.index]
    return sort(unique(vcat(n, clamp.(round.(Int, n .* span .^ t), 1, N_HI[c.index]))))
end

function _profile_task(theta, c, grid, k::Int, N_base::Vector{Int}, seed::Int)
    ds = sens_drawset(k; seed = seed)
    n_m = sum(MOMENT_MASK)
    M = fill(NaN, length(grid), n_m)
    Nh = fill(NaN, length(grid), S)
    conv = falses(length(grid))
    if c.kind === :N
        # one Ricardian solve per draw set: the network does not depend on N_s
        net = solve_network(theta; u_draws = ds.u, sample_weights = ds.w)
        for (i, n) in enumerate(grid)
            Nf = copy(N_base); Nf[c.index] = Int(n)
            m = compute_moments(net, theta; N_fixed = Nf, perms = ds.perms)
            M[i, :] .= moments_to_vec(m); Nh[i, :] .= Nf; conv[i] = true
        end
        return (M = M, N = Nh, conv = conv)
    end
    for (i, v) in enumerate(grid)
        x = collect(Float64, theta); x[c.index] = v
        try
            r = sens_profile(x)
            r.converged || continue
            m, N = sens_moments(r.theta, ds)       # N̂_s re-calibrated, as in estimation
            M[i, :] .= m; Nh[i, :] .= N; conv[i] = true
        catch err
            @warn "profile $(c.label) = $v, draw set $k failed: $(sprint(showerror, err))"
        end
    end
    return (M = M, N = Nh, conv = conv)
end

# The moments a parameter is meant to move (Figure D.1 pairs one moment with one
# parameter): α → the distance coefficients, κ → β_κ, N_s → Ḡ_s(0).
function profile_targets(c)
    b4 = collect(BLOCK_RANGES[4])
    c.kind === :alpha && return b4[1:(end - 1)]
    c.kind === :kappa && return [b4[end]]
    return [collect(BLOCK_RANGES[6])[c.index]]
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
    θ = collect(Float64, theta)
    cols = filter(c -> c.kind in (:alpha, :kappa, :N), sens_columns(θ; include_T = false, include_head = false))
    base = pmap(k -> sens_moments(θ, sens_drawset(k; seed = seed)), 0:K)
    N_base = [b[2] for b in base]
    grids = [profile_grid(θ, c, N_base[1]; G = G, span = span, kappa_span = kappa_span) for c in cols]
    println("Sensitivity profiles: $(length(cols)) parameters × ≤ $(maximum(length.(grids))) points × $(K + 1) draw sets")
    tasks = [(j, k) for j in eachindex(cols) for k in 0:K]
    res = pmap(t -> _profile_task(θ, cols[t[1]], grids[t[1]], t[2], N_base[t[2] + 1], seed), tasks)

    emp = vec(empirical_moments)
    fname(l) = "profile_" * replace(l, r"[^A-Za-z0-9_]+" => "_") * ".npz"
    lines = String["MOMENT SENSITIVITY PROFILES  $(now())",
                   "  one parameter swept, the rest at the estimate; α, κ: T re-inverted and N̂ re-calibrated",
                   "  slope = central difference at the estimate (estimation draws); crosses = the model",
                   "  curve crosses the data moment inside the grid; monotone = share of grid steps with",
                   "  the slope's sign.", ""]
    push!(lines, @sprintf("  %-22s %-22s %11s %11s %11s %8s %8s", "parameter", "moment", "estimate",
                          "model", "data", "monot.", "crosses"))
    for (j, c) in enumerate(cols)
        rs = [res[i] for (i, t) in enumerate(tasks) if t[1] == j]       # k = 0..K in order
        Mest = rs[1].M
        Md = cat((r.M for r in rs[2:end])...; dims = 3)                 # grid × m × K
        m_mean = [_nanmean(Md[i, l, :]) for i in axes(Md, 1), l in axes(Md, 2)]
        m_sd   = [_nanstd(Md[i, l, :])  for i in axes(Md, 1), l in axes(Md, 2)]
        tg = profile_targets(c)
        est = c.kind === :N ? Float64(N_base[1][c.index]) : θ[c.index]
        NPZ.npzwrite(joinpath(dest, fname(c.label)),
                     Dict("grid" => Float64.(grids[j]), "m_est" => Mest, "m_mean" => m_mean,
                          "m_sd" => m_sd, "N_est" => rs[1].N, "converged" => Float64.(rs[1].conv),
                          "estimate" => [est], "targets" => Float64.(tg), "data" => emp,
                          "kind" => [Float64(findfirst(==(c.kind), (:head, :alpha, :kappa, :T, :N)))]))
        g = Float64.(grids[j]); i0 = findfirst(==(est), g)
        for t in tg
            y = Mest[:, t]
            ok = isfinite.(y)
            d = diff(y[ok])
            slope = (i0 === nothing || i0 == 1 || i0 == length(g)) ? NaN :
                    (y[i0 + 1] - y[i0 - 1]) / (g[i0 + 1] - g[i0 - 1])
            mono = isempty(d) || !isfinite(slope) ? NaN : count(x -> sign(x) == sign(slope), d) / length(d)
            r = y[ok] .- emp[t]
            crosses = length(r) > 1 && any(r[1:(end - 1)] .* r[2:end] .<= 0)
            push!(lines, @sprintf("  %-22s %-22s %11.4g %11.4g %11.4g %8.2f %8s", c.label,
                                  first(MOMENT_LABELS[t], 22), est,
                                  i0 === nothing ? NaN : y[i0], emp[t], mono, string(crosses)))
        end
    end
    write(joinpath(dest, "profiles_index.txt"), join([c.label * "\t" * fname(c.label) for c in cols], "\n") * "\n")
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
    catch e
        @warn "sensitivity profiles failed: $(sprint(showerror, e))"
    end
    try
        sensitivity_jacobian(theta; K = K, h = h, include_T = include_T, dest = dest)
    catch e
        @warn "sensitivity Jacobian failed: $(sprint(showerror, e))"
    end
    return nothing
end
