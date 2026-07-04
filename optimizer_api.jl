"""
Optimizer dispatch layer.

`optimize_stage` is the single seam between the SMM pipeline and the concrete
optimizer. Both backends honor the same contract:

    (objective, lb, ub; x0, ...) -> (best_x::Vector, best_f::Float64, history::Dict)

so `train_stage` calls `optimize_stage` once and neither it nor `main.jl` needs to
know which optimizer runs. Selection is by `OPTIMIZER_BACKEND` (set in
`load_parameters.jl` from the `--optimizer` entry-point flag) and overridable
per-call via the `backend` kwarg.

`x0` is the warm start / incumbent (previous best), in the stage's search space.
For PSO it becomes the guaranteed warm-start particle; for CMA-ES the initial
mean plus incumbent floor.
"""

"""
    optimize_stage(objective, lb, ub; x0, n_particles, max_iter,
                   beta_constraint, beta_indices, verbose, backend, seed)

Dispatch to the selected optimizer backend. Returns `(best_x, best_f, history)`.
"""
function optimize_stage(
    objective::Function,
    lb::Vector{Float64},
    ub::Vector{Float64};
    x0::Union{Vector{Float64}, Nothing} = nothing,
    n_particles::Int = 70,
    max_iter::Int = 100,
    beta_constraint::Bool = true,
    beta_indices::UnitRange = 1:0,
    verbose::Bool = false,
    backend::Symbol = OPTIMIZER_BACKEND,
    seed::Int = 1,
)
    if backend == :cmaes
        return parallel_cmaes_smm(
            objective, lb, ub;
            x0 = x0,
            n_particles = n_particles,
            max_iter = max_iter,
            beta_constraint = beta_constraint,
            beta_indices = beta_indices,
            seed = seed,
            verbose = verbose,
        )
    elseif backend == :pso
        return parallel_pso_smm(
            objective, lb, ub;
            n_particles = n_particles,
            max_iter = max_iter,
            warm_start_particle = x0,
            beta_constraint = beta_constraint,
            beta_indices = beta_indices,
            verbose = verbose,
        )
    else
        error("Unknown optimizer backend :$backend (expected :pso or :cmaes)")
    end
end
