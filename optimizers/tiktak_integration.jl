"""
TikTak global multistart optimizer backend for the SMM optimizer layer.

`parallel_tiktak_smm` obeys the same optimizer contract as the PSO and CMA-ES
backends —

    (objective, lb, ub; x0, ...) -> (best_x::Vector, best_f::Float64, history::Dict)

— so it drops into `optimize_stage` behind the `--optimizer=tiktak` switch with no
change to the callers in `train_stage` / `run_optimization`.

TikTak (Guvenen; Arnoud, Guvenen & Kleineberg 2019 — https://github.com/serdarozkan/TikTak)
is a **multistart global** method with two stages:

1. **Pre-testing.** Evaluate the objective at a large low-discrepancy (Sobol) set
   of points and keep the best `n_restarts` *legitimate* (finite) ones, sorted by
   loss. This is embarrassingly parallel (`pmap`).

2. **Sequential local search.** Walk the kept points from best to worst. For the
   `i`-th kept point `s_i`, start a local optimizer not from `s_i` itself but from a
   convex combination pulled toward the best local optimum found so far `z*`:

        z_start_i = (1 - θ_i)·s_i + θ_i·z*,     θ_i = clamp((i/K)^{1/2}, θ_min, θ_max)

   θ_i grows with `i`, so early (best-pretest) points explore around their own
   basin while later (worse-pretest) points are used to *refine* the incumbent.
   Whenever a local search improves `z*`, it is updated.

Design choices specific to this codebase (mirroring `cmaes_integration.jl`):

- **Normalized search cube.** Nelder-Mead uses a single-scale simplex, so it wants
  comparably-scaled coordinates. Our raw box mixes shares in [0,1], α, and log-T (φ)
  of very different widths. We therefore run everything on the unit cube [0,1]^n and
  map back to [lb,ub] inside the evaluator (identical to the CMA-ES backend); the
  blend, the simplex, and every clamp live in cube space.
- **Local solver = Nelder-Mead (`Optim.jl`).** Derivative-free, already a project
  dependency (no new package). `local_search` is factored out so a DFNLS/BOBYQA
  engine (e.g. `NLopt.:LN_BOBYQA`) could replace it without touching the driver.
- **Parallel local searches.** The sequential stage is parallelized in batches of
  `nworkers()`: within a batch, all starts blend toward the incumbent as it stood at
  batch start; the incumbent is updated after each batch (the standard parallel
  TikTak trade-off). Each local search runs serially on one worker, so the objective
  need not itself be parallel.
- **α ordering.** Enforced by the same `enforce_alpha_constraint` repair PSO/CMA-ES
  use, applied in real space inside `to_real` before every evaluation.
- **Monotonicity vs warm start.** `x0` is injected as a pre-test candidate and the
  returned point is compared against `f(x0)` at the end, so — as in the other
  backends — the result is never worse than the warm start the staged pipeline relies
  on.

Cross-file references (`enforce_alpha_constraint`, `sobol_scrambled_net`, `N_TAU`)
resolve at call time, so this file may be included in any order as long as every
backend + `model_CP.jl` is `@everywhere include`d before any optimizer runs.
"""

using Distributed
using Random
using Statistics
using Optim

"""
    tiktak_blend_weight(i, K; theta_min=0.1, theta_max=0.995) -> θ_i

TikTak convex-combination weight for the `i`-th kept point out of `K`
(1-indexed, best first): `θ_i = clamp((i/K)^{1/2}, θ_min, θ_max)`. Small for the
best pretest points (explore their own basin), near `θ_max` for the last ones
(pull hard toward the incumbent best to refine it).
"""
function tiktak_blend_weight(i::Int, K::Int; theta_min::Float64 = 0.1, theta_max::Float64 = 0.995)
    K <= 1 && return theta_min
    return clamp(sqrt(i / K), theta_min, theta_max)
end

"""
    parallel_tiktak_smm(objective, lb, ub; x0, ...) -> (best_x, best_f, history)

TikTak counterpart to `parallel_pso_smm` / `parallel_cmaes_smm`. `objective` maps a
full/real parameter vector to a scalar loss (or `nothing`/`Inf` on failure). `x0` is
the warm start (previous best); may be `nothing`.

Contract-mapped budget knobs:
- `n_particles` → `n_restarts`, the number of local searches (kept best Sobol points).
- `max_iter`    → Nelder-Mead iteration budget per local search.
Pre-test draws = `sobol_multiplier · n_restarts` (TikTak keeps the best `n_restarts`).
"""
function parallel_tiktak_smm(
    objective::Function,
    lb::Vector{Float64},
    ub::Vector{Float64};
    x0::Union{Vector{Float64}, Nothing} = nothing,
    n_particles::Union{Int, Nothing} = nothing,   # → number of local restarts K
    max_iter::Union{Int, Nothing} = nothing,      # → Nelder-Mead iterations / search
    alpha_constraint::Bool = true,
    alpha_indices::UnitRange = 1:0,
    sobol_multiplier::Int = 100,                   # pretest draws = multiplier · K
    theta_min::Float64 = 0.1,                      # blend-weight floor
    theta_max::Float64 = 0.995,                    # blend-weight ceiling
    seed::Int = 1,
    verbose::Bool = false,
)
    d = length(lb)
    span = ub .- lb
    # Guard against zero-width coordinates (fixed params): map through unchanged.
    span_safe = map(s -> s == 0.0 ? 1.0 : s, span)

    # unit-cube z ∈ [0,1]^d  →  real parameter vector (+ α repair)
    function to_real(z)
        x = lb .+ clamp.(z, 0.0, 1.0) .* span
        return alpha_constraint ? enforce_alpha_constraint(x, alpha_indices) : x
    end

    # scalar objective on the cube, sanitized (nothing / non-finite → Inf)
    function eval_real(z)
        v = objective(to_real(z))
        return (v === nothing || !isfinite(v)) ? Inf : Float64(v)
    end

    K         = n_particles === nothing ? 70 : max(1, n_particles)
    n_iter    = max_iter === nothing ? 200 : max_iter
    n_sobol   = max(K, sobol_multiplier * K)
    rng       = MersenneTwister(seed)

    history = Dict{String, Any}(
        "best_fitness" => Float64[],
        "mean_fitness" => Float64[],
        "best_params"  => Vector{Float64}[],
    )

    # ── Stage 1: pre-testing (embarrassingly parallel) ───────────────────────
    net       = sobol_scrambled_net(n_sobol, d, rng)          # n_sobol × d in [0,1)
    cube_pts  = Vector{Float64}[net[i, :] for i in 1:n_sobol]
    # Inject the warm start as a candidate (in cube coords), so the pretest set is
    # never worse than x0 and the incumbent can start there.
    if x0 !== nothing
        push!(cube_pts, clamp.((x0 .- lb) ./ span_safe, 0.0, 1.0))
    end

    if verbose
        println("\n" * "="^60)
        println("Starting TikTak Optimization")
        println("="^60)
        println("Dimension: $d | restarts K: $K | pretest draws: $(length(cube_pts)) | NM iters/search: $n_iter")
        println("workers: $(nworkers()) | seed: $seed | warm start: $(x0 === nothing ? "none" : "yes")")
        println("="^60); flush(stdout)
        println("[TikTak] Pre-testing $(length(cube_pts)) Sobol points in parallel...")
        flush(stdout)
    end

    f_pre  = pmap(eval_real, cube_pts)
    f_pre  = Float64[v === nothing ? Inf : Float64(v) for v in f_pre]
    order  = sortperm(f_pre)                                  # ascending loss (best first)
    finite = filter(i -> isfinite(f_pre[i]), order)
    use    = isempty(finite) ? order : finite                # keep legitimate points if any
    n_keep = min(K, length(use))
    kept   = Vector{Float64}[cube_pts[use[i]] for i in 1:n_keep]
    kept_f = Float64[f_pre[use[i]] for i in 1:n_keep]

    if verbose
        n_ok = count(isfinite, f_pre)
        println("[TikTak] Pretest done: $n_ok/$(length(f_pre)) finite; best pretest loss = $(round(kept_f[1], digits=6)); keeping $n_keep starts.")
        flush(stdout)
    end

    # ── incumbent global best (starts at the best pretest point) ─────────────
    gbest_z = copy(kept[1]); gbest_f = kept_f[1]

    # ── local search on the cube; pluggable derivative-free engine ───────────
    # Nelder-Mead (Optim.jl) for d ≥ 2; a Brent line-search over the unit interval
    # for the 1-D case (Optim's simplex needs ≥ 2 dimensions), which arises only in
    # the single-α gamma_beta_only sub-stage. A DFNLS/BOBYQA engine (e.g. NLopt's
    # LN_BOBYQA) could replace this block without touching the driver.
    function local_search(z_start::Vector{Float64})
        try
            if d == 1
                # Univariate Brent over the unit interval (Brent is Optim's default
                # for the (f, lower, upper) signature).
                res = Optim.optimize(t -> eval_real([t]), 0.0, 1.0)
                return ([Optim.minimizer(res)], Optim.minimum(res))
            else
                res = Optim.optimize(eval_real, z_start, NelderMead(),
                                     Optim.Options(iterations = n_iter))
                return (Optim.minimizer(res), Optim.minimum(res))
            end
        catch
            # A failed local search must not sink the whole batch: fall back to the
            # start point's own objective value.
            return (z_start, eval_real(z_start))
        end
    end

    # ── Stage 2: sequential local searches, batched over workers ─────────────
    W = max(1, nworkers())
    i = 1
    while i <= n_keep
        j = min(i + W - 1, n_keep)
        # Blend each start toward the incumbent as it stands at the batch start.
        starts = Vector{Float64}[
            let θ = tiktak_blend_weight(k, n_keep; theta_min = theta_min, theta_max = theta_max)
                (1 - θ) .* kept[k] .+ θ .* gbest_z
            end
            for k in i:j
        ]
        results = pmap(local_search, starts)
        for (zmin, fmin) in results
            fmin = (fmin === nothing || !isfinite(fmin)) ? Inf : Float64(fmin)
            if fmin < gbest_f
                gbest_f = fmin
                gbest_z = copy(zmin)
            end
        end

        push!(history["best_fitness"], gbest_f)
        finite_batch = Float64[r[2] for r in results if isfinite(r[2])]
        push!(history["mean_fitness"], isempty(finite_batch) ? Inf : mean(finite_batch))
        push!(history["best_params"], to_real(gbest_z))

        if verbose
            println("[TikTak] local searches $i–$j / $n_keep : best loss = $(round(gbest_f, digits=6))")
            flush(stdout)
        end
        i = j + 1
    end

    # ── restore monotonicity vs the warm start ───────────────────────────────
    if x0 !== nothing
        x0r = alpha_constraint ? enforce_alpha_constraint(copy(x0), alpha_indices) : copy(x0)
        f0  = objective(x0r)
        f0  = (f0 === nothing || !isfinite(f0)) ? Inf : Float64(f0)
        if f0 <= gbest_f
            gbest_f = f0
            # x0r is already in real space; recover its cube coords for consistency.
            gbest_z = clamp.((x0r .- lb) ./ span_safe, 0.0, 1.0)
        end
    end

    if verbose
        println("[TikTak] Done. Best fitness: $(round(gbest_f, digits=6))")
        flush(stdout)
    end

    return to_real(gbest_z), gbest_f, history
end
