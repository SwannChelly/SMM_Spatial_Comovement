"""
CMA-ES integration for SMM calibration.

`parallel_cmaes_smm` mirrors the `parallel_pso_smm` contract exactly —
`(objective, lb, ub; ...) -> (best_x, best_f, history)` — so it drops into
`optimize_stage` behind the `--optimizer=cmaes` switch with no change to the
callers in `train_stage` / `run_pso_optimization`.

Design choices specific to this codebase:
- **Normalized search cube.** CMA-ES assumes comparably scaled coordinates and a
  single scalar σ. Our raw box mixes shares in [0,1], α, and log-T (φ) of very
  different widths, so we run CMA-ES on the unit cube [0,1]^n and map back to
  [lb,ub] inside the evaluator. One `σ0` is then meaningful for every coordinate.
- **Parallel evaluation.** The library hands us the whole population each
  generation; we `pmap` it, reusing the same worker pool PSO uses.
- **Incumbent tracking (monotonicity).** Standard CMA-ES may let the mean drift
  to a worse point, but the staged pipeline relies on "loss never increases from
  the previous stage" (PSO guaranteed this via the warm-start particle). We track
  the best-ever evaluated point and compare it against f(x0) before returning, so
  the returned solution is never worse than the warm start.
- **α ordering.** Enforced by the same `enforce_alpha_constraint` repair PSO uses,
  applied in real space before evaluation.
"""

using Distributed
using Statistics
using CMAEvolutionStrategy

"""
    parallel_cmaes_smm(objective, lb, ub; x0, ...) -> (best_x, best_f, history)

CMA-ES counterpart to `parallel_pso_smm`. `objective` maps a full/real parameter
vector to a scalar loss (or `nothing`/`Inf` on failure). `x0` is the warm start
(previous best); may be `nothing` (→ cube center).
"""
function parallel_cmaes_smm(
    objective::Function,
    lb::Vector{Float64},
    ub::Vector{Float64};
    x0::Union{Vector{Float64}, Nothing} = nothing,
    n_particles::Union{Int, Nothing} = nothing,   # → CMA-ES popsize λ (nothing = default)
    max_iter::Union{Int, Nothing} = nothing,      # generations budget (→ maxfevals = λ·max_iter)
    alpha_constraint::Bool = true,
    alpha_indices::UnitRange = 1:0,
    sigma0::Float64 = 0.2,                         # initial step on the unit cube
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

    z0 = x0 === nothing ? fill(0.5, d) : clamp.((x0 .- lb) ./ span_safe, 0.0, 1.0)

    # ── incumbent tracker (updated MASTER-side; see f_par) ───────────────────
    # NB: the tracker must live on the master. `eval_one` runs on workers via
    # pmap, so any Ref update inside it would touch a worker-local copy and be
    # lost. `eval_one` is therefore pure; the incumbent is updated in f_par after
    # pmap returns the population's losses to the master.
    best_f = Ref(Inf)
    best_x = Ref(to_real(z0))

    eval_one(z) = begin
        v = objective(to_real(z))
        (v === nothing || !isfinite(v)) ? Inf : Float64(v)
    end

    # ── history (superset of PSO's keys) ─────────────────────────────────────
    history = Dict{String, Any}(
        "best_fitness" => Float64[],
        "mean_fitness" => Float64[],
        "best_params"  => Vector{Float64}[],
    )

    # Library calls this once per generation with the whole population as an
    # (n × popsize) matrix (individuals in columns); also accept a Vector of
    # vectors defensively. Returns a vector of popsize losses.
    function f_par(pop)
        cols = pop isa AbstractMatrix ? [pop[:, j] for j in 1:size(pop, 2)] : collect(pop)
        vals = pmap(eval_one, cols)
        vals = Float64[v === nothing ? Inf : Float64(v) for v in vals]
        gi = argmin(vals)
        if vals[gi] < best_f[]
            best_f[] = vals[gi]
            best_x[] = to_real(cols[gi])   # re-map (incl. α repair) the winner
        end
        finite = filter(isfinite, vals)
        push!(history["best_fitness"], best_f[])
        push!(history["mean_fitness"], isempty(finite) ? Inf : mean(finite))
        push!(history["best_params"], copy(best_x[]))
        return vals
    end

    popsize  = n_particles === nothing ? 4 + floor(Int, 3 * log(d)) : n_particles
    maxfevals = max_iter === nothing ? 300 * popsize : max_iter * popsize

    if verbose
        println("\n" * "="^60)
        println("Starting CMA-ES Optimization")
        println("="^60)
        println("Dimension: $d | popsize λ: $popsize | σ0: $sigma0 | seed: $seed")
        println("maxfevals: $maxfevals | warm start: $(x0 === nothing ? "none (cube center)" : "yes")")
        println("="^60); flush(stdout)
    end

    # `lower`/`upper` on the unit cube keep CMA-ES's own boundary handling active
    # (a light penalty near the walls) on top of the clamp in `to_real`.
    result = CMAEvolutionStrategy.minimize(
        f_par, z0, sigma0;
        lower = zeros(d),
        upper = ones(d),
        popsize = popsize,
        parallel_evaluation = true,
        maxfevals = maxfevals,
        ftol = 1e-11,
        xtol = 1e-12,
        seed = seed,
        verbosity = verbose ? 1 : 0,
    )

    # ── restore monotonicity vs the warm start ───────────────────────────────
    if x0 !== nothing
        x0r = alpha_constraint ? enforce_alpha_constraint(copy(x0), alpha_indices) : copy(x0)
        f0 = objective(x0r)
        f0 = (f0 === nothing || !isfinite(f0)) ? Inf : Float64(f0)
        if f0 <= best_f[]
            best_f[] = f0
            best_x[] = x0r
        end
    end

    if verbose
        println("[CMA-ES] Done. Best fitness: $(round(best_f[], digits=6))")
        flush(stdout)
    end

    return best_x[], best_f[], history
end
