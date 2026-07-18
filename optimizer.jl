"""
Optimization layer — backend-neutral hub.

This file owns everything that is independent of *which* optimizer runs:

- `optimize_stage`  : the single dispatch seam (PSO / CMA-ES / TikTak on `OPTIMIZER_BACKEND`).
- `train_stage`     : the staged optimization builder — constructs bounds, warm
                      start, and the objective closure (in the log-space φ search
                      space for T), then calls `optimize_stage`.
- `run_optimization`: the per-step orchestrator — Stage-0 α search, Stage-1 joint
                      fit, and (PSO only) the block-coordinate refinement loop, with
                      folder/report bookkeeping.
- `get_param_start_index` / `get_n_T_params` : flat-layout helpers.

The concrete optimizers live in the backend files and honor one contract:

    (objective, lb, ub; x0, ...) -> (best_x::Vector, best_f::Float64, history::Dict)

  - PSO   : `parallel_pso_smm`    (pso_integration.jl)
  - CMA-ES: `parallel_cmaes_smm`  (cmaes_integration.jl)
  - TikTak: `parallel_tiktak_smm` (tiktak_integration.jl)

so `main.jl` / `run_optimization` never mention a specific optimizer. Legacy names
`train_stage_pso` / `run_pso_optimization` are retained as aliases at the bottom.

Cross-file references (`parallel_pso_smm`, `parallel_cmaes_smm`,
`parallel_SMM_safe`, `generate_initial_alphas`, `generate_report`, `run_reporting`,
`unpack_params`, `build_tau`, the φ transforms, and `OPTIMIZER_BACKEND`) all resolve
at call time, so this file may be included before or after them as long as every
file is `@everywhere include`d before any optimizer runs.
"""

using Distributed
using Random
using Statistics
using NPZ
using Printf


# ═══════════════════════════════════════════════════════════════════════════
# Dispatch seam
# ═══════════════════════════════════════════════════════════════════════════

"""
    optimize_stage(objective, lb, ub; x0, n_particles, max_iter,
                   alpha_constraint, alpha_indices, verbose, backend, seed)

Dispatch to the selected optimizer backend. Returns `(best_x, best_f, history)`.
`x0` is the warm start / incumbent (previous best), in the stage's search space:
for PSO it becomes the guaranteed warm-start particle; for CMA-ES the initial mean
plus incumbent floor; for TikTak a pre-test candidate plus incumbent floor.
"""
function optimize_stage(
    objective::Function,
    lb::Vector{Float64},
    ub::Vector{Float64};
    x0::Union{Vector{Float64}, Nothing} = nothing,
    n_particles::Int = 70,
    max_iter::Int = 100,
    alpha_constraint::Bool = true,
    alpha_indices::UnitRange = 1:0,
    verbose::Bool = false,
    backend::Symbol = OPTIMIZER_BACKEND,
    seed::Int = 1,
    t_conv_probe::Union{Function, Nothing} = nothing,
)
    if backend == :cmaes
        return parallel_cmaes_smm(
            objective, lb, ub;
            x0 = x0,
            n_particles = n_particles,
            max_iter = max_iter,
            alpha_constraint = alpha_constraint,
            alpha_indices = alpha_indices,
            seed = seed,
            verbose = verbose,
        )
    elseif backend == :tiktak
        return parallel_tiktak_smm(
            objective, lb, ub;
            x0 = x0,
            n_particles = n_particles,
            max_iter = max_iter,
            alpha_constraint = alpha_constraint,
            alpha_indices = alpha_indices,
            seed = seed,
            verbose = verbose,
        )
    elseif backend == :pso
        return parallel_pso_smm(
            objective, lb, ub;
            n_particles = n_particles,
            max_iter = max_iter,
            warm_start_particle = x0,
            alpha_constraint = alpha_constraint,
            alpha_indices = alpha_indices,
            verbose = verbose,
            t_conv_probe = t_conv_probe,
        )
    else
        error("Unknown optimizer backend :$backend (expected :pso, :cmaes or :tiktak)")
    end
end


# ═══════════════════════════════════════════════════════════════════════════
# Flat-layout helpers
# Layout: [Ω^L(1), Ω^s(S), A(R_downstream), alpha(N_TAU), T(sum(T_MASK))]
# ═══════════════════════════════════════════════════════════════════════════

"""
    get_param_start_index(param_name)

Starting index of a parameter block in the full (level) parameter vector.
"""
function get_param_start_index(param_name::Symbol)
    if param_name == :agg_labor_share_tech
        return 1
    elseif param_name == :agg_industry_share_tech
        return 2
    elseif param_name == :productivity
        return 2 + S
    elseif param_name == :alpha
        return 2 + S + R_downstream
    elseif param_name == :T
        return 2 + S + R_downstream + N_TAU
    else
        error("Unknown parameter name: $param_name")
    end
end

# Number of T parameters (only non-zero gamma_ls entries)
function get_n_T_params()
    return sum(T_MASK)
end


# ═══════════════════════════════════════════════════════════════════════════
# Staged optimization builder (backend-agnostic)
# ═══════════════════════════════════════════════════════════════════════════

"""
    train_stage(n_particles, max_iter; kwargs...)

Backend-agnostic staged training. Builds bounds and the warm start (in the
log-space φ search space for the T block), defines the stage objective, and hands
off to `optimize_stage` (PSO or CMA-ES per `OPTIMIZER_BACKEND`). Automatically
extracts the previous best as the warm start / incumbent, preserving monotone
improvement across stages.

# Arguments
- `n_particles`: population size (PSO particles / CMA-ES λ)
- `max_iter`: iteration/generation budget
- `init_alpha`: initial alpha values (if starting fresh)
- `variable_list`: which parameters to optimize (e.g., ["alpha", "T"]); `nothing` = all
- `last_stage_folder`: folder with previous stage results (`nothing` = fresh start)
- `K`: which parameter set (column) to use from previous stage
- `radius`: search-radius multiplier
- `second_stage`: legacy flag (unused live path)

# Returns
- `best_params`: best parameter vector found (full, level space)
- `best_fitness`: best loss value achieved
- `history`: optimization history
"""
function train_stage(
    n_particles::Int,
    max_iter::Int;
    init_alpha = nothing,
    variable_list = nothing,
    last_stage_folder = nothing,
    K = 1,
    radius = 0.1,
    second_stage = false,
    method = false,
    u_draws::Union{Nothing, Matrix{Float64}} = nothing,
    sample_weights::Union{Nothing, Matrix{Float64}} = nothing,
    weight_matrix::Union{Nothing, AbstractMatrix} = nothing,
    warm_start_override::Union{Nothing, Vector{Float64}} = nothing,
    moment_blocks::Union{Nothing, Vector{Int}} = nothing,
    analytical::Bool = false,
    n_quad::Int = 200,
    profile_T::Bool = false
)

    # ── T-profiling (Design A, profiling.jl) ─────────────────────────────────
    # When profile_T, the T block is NOT searched: to each particle's head
    # (Ω^L, Ω^s, A, α) we assign T*(α,Ω,A) = invert_T_ge(...), collapsing the
    # search by ~N_T_REDUCED dimensions. `profiled_theta` (a top-level function in
    # profiling.jl, so it serializes across the PSO pmap) overwrites the T block of
    # a full LEVEL vector with that inversion; it is applied in the objective and
    # the final reconstruction. profile_T=false ⇒ every branch below is the
    # historical path, byte-identical.
    nhead_lvl     = 1 + S + R_downstream + N_TAU         # [Ω^L|Ω^s|A|α] length
    T_PLACEHOLDER = vec(permutedims(T_rs_init))[T_MASK]  # inert filler (overwritten)

    # ── Init-anchored search box for α and T ─────────────────────────────────
    # α and T are constrained to [BOUND_LO, BOUND_HI] × their INITIAL value in every
    # stage: T to T_rs_init (the γ-inversion), α to TAU_PRIOR (the α prior). This
    # aligns the trust region with the theory-based warm start so PSO never drifts
    # beyond that factor of it. The per-stage radius `radius` still anneals INSIDE
    # this box in continue stages. Falls back to the stage's starting value when no
    # prior anchor is available (TAU_PRIOR === nothing ⇒ box around init_alpha).
    # BOUND_LO/BOUND_HI are the shared globals from load_parameters.jl.
    phi_anchor = t_levels_to_free_phi(vec(permutedims(T_rs_init))[T_MASK])  # φ of T_rs_init (N_T_FREE)
    T_phi_lo   = phi_anchor .+ log(BOUND_LO)
    T_phi_hi   = phi_anchor .+ log(BOUND_HI)

    # Build bounds based on previous stage or initialization
    if last_stage_folder === nothing
        # Fresh start - use init_alpha


        A = copy(emp_pi_r_full).^(1/abs(epsilon)).*regional_wages[N_downstream_per_region .!= 0]  # analytical inversion
        A ./= sum(A)

        # T block is searched as free log-space φ (ref entries dropped). The box is
        # φ_init + [log(BOUND_LO), log(BOUND_HI)] ⇒ T ∈ [×BOUND_LO, ×BOUND_HI] × T_rs_init,
        # centred on the γ-inversion init (T_init is no longer ≡1, so the box must track it).
        # α is boxed to [×BOUND_LO, ×BOUND_HI] × the α prior (TAU_PRIOR), else × init_alpha.
        alpha_anchor = TAU_PRIOR !== nothing ? TAU_PRIOR : init_alpha
        # profile_T ⇒ the T (φ) block is dropped from the search space entirely.
        T_lb_block = profile_T ? Float64[] : T_phi_lo
        T_ub_block = profile_T ? Float64[] : T_phi_hi
        lb = vcat(
            0.8*agg_labor_share,
            0.8 .* agg_industry_share,
            0.8.* A,
            alpha_anchor .* BOUND_LO,
            T_lb_block
        )

        ub = vcat(
            1.2*agg_labor_share,
            1.2 .* agg_industry_share,
            A .* 1.2,
            alpha_anchor .* BOUND_HI,
            T_ub_block
        )

        alpha_constraint = true
        alpha_start = 1 + S + R_downstream + 1   # alpha follows Ω^L, Ω^s, A in new layout
        alpha_indices = alpha_start:(alpha_start + N_TAU - 1)
        best_params_prev = nothing
        # Use warm_start_override if provided (Step 3 warm-start from θ̂_1); convert
        # its level T block to the free-φ search space. profile_T ⇒ head only (the
        # search vector carries no T block); the level and search heads coincide.
        warm_start = warm_start_override === nothing ? nothing :
            (profile_T ? warm_start_override[1:nhead_lvl] : full_to_search(warm_start_override))
        cached_tau = nothing  # Alpha is being optimized in initial stage

    else
        # Continue from previous stage - load best params
        best_params_prev = NPZ.npzread(joinpath(last_stage_folder, "best_params.npy"))[:,K]
        names = [:agg_labor_share_tech, :agg_industry_share_tech, :productivity, :alpha, :T]
        vals = unpack_params(best_params_prev)
        params_dict = Dict(names .=> vals)
        # T from unpack_params is full S*R; reduce to non-zero entries, then map to
        # the free log-space search vector φ (ref entries dropped). All downstream
        # length/slice arithmetic then uses N_T_FREE automatically.
        T_red_levels = vec(permutedims(reshape(params_dict[:T], S, R)))[T_MASK]   # region-major full → s-major, then mask
        params_dict[:T] = t_levels_to_free_phi(T_red_levels)

        # Handle single variable or list
        var_list = isa(variable_list, String) ? [variable_list] : variable_list

        # Check if alpha is fixed — if so, precompute and cache tau
        alpha_is_fixed = !("alpha" in var_list)
        cached_tau = alpha_is_fixed ? build_tau(params_dict[:alpha]) : nothing
        alpha_is_fixed && println("[stage] Alpha is fixed — tau precomputed and cached")

        # Build bounds for selected variables
        # For most parameters: [value * radius, value / radius]
        # For agg_labor_share_tech: [value * (1-radius), value * (1+radius)] to stay in [0,1]
        lb_parts = Vector{Float64}[]
        ub_parts = Vector{Float64}[]

        for v in var_list
            val = params_dict[Symbol(v)]
            if v == "agg_labor_share_tech"
                # Symmetric percentage bounds: ±radius around current value
                lb_v = val .* (1 - radius)
                ub_v = val .* (1 + radius)
                # Clamp to valid range [0.001, 1.0]
                lb_v = max.(lb_v, 0.001)
                ub_v = min.(ub_v, 1.0)
            elseif v == "T"
                # val is φ (log space): the per-stage radius gives the additive box
                # φ ± |log radius|, then clamped to the init-anchored [×BOUND_LO, ×BOUND_HI]
                # × T_rs_init box (T_phi_lo/T_phi_hi) so the search never leaves it in any stage.
                lb_v = max.(val .+ log(radius), T_phi_lo)
                ub_v = min.(val .- log(radius), T_phi_hi)
            elseif v == "alpha"
                # Per-stage radius around the incumbent, clamped to [×BOUND_LO, ×BOUND_HI] ×
                # the α prior (TAU_PRIOR); falls back to × the incumbent when no prior.
                anchor = TAU_PRIOR !== nothing ? TAU_PRIOR : val
                lb_v = max.(val .* (1 - radius), anchor .* BOUND_LO)
                ub_v = min.(val .* (1 + radius), anchor .* BOUND_HI)
            else
                # Standard multiplicative bounds
                lb_v = val .* radius
                ub_v = val ./ radius
            end
            push!(lb_parts, isa(lb_v, Number) ? [lb_v] : lb_v)
            push!(ub_parts, isa(ub_v, Number) ? [ub_v] : ub_v)
        end

        lb = vcat(lb_parts...)
        ub = vcat(ub_parts...)

        # Check if alpha is being optimized
        alpha_constraint = "alpha" in var_list
        if alpha_constraint
            alpha_idx_in_var = findfirst(==("alpha"), var_list)
            alpha_start = alpha_idx_in_var == 1 ? 1 : sum([length(params_dict[Symbol(var_list[i])]) for i in 1:(alpha_idx_in_var-1)]) + 1
            alpha_indices = alpha_start:(alpha_start + N_TAU - 1)
        else
            alpha_indices = 1:0
        end

        # Handle second stage T masking
        if "T" in var_list && second_stage
            t_idx = findfirst(==("T"), var_list)
            t_start = t_idx == 1 ? 0 : sum([length(params_dict[Symbol(var_list[i])]) for i in 1:(t_idx-1)])

            mask = vec(mask_emp_gamma_ls)
            t_length = length(params_dict[:T])
            t_indices = (t_start + 1):(t_start + t_length)

            lb_T = lb[t_indices][mask .== 1]
            ub_T = ub[t_indices][mask .== 1]

            lb = vcat(lb[1:t_start], lb_T, (t_start + t_length < length(lb)) ? lb[t_start + t_length + 1:end] : Float64[])
            ub = vcat(ub[1:t_start], ub_T, (t_start + t_length < length(ub)) ? ub[t_start + t_length + 1:end] : Float64[])
        end

        # CRITICAL: Extract warm start particle (previous best in reduced parameter space)
        warm_start = Float64[]
        for var in var_list
            var_symbol = Symbol(var)
            append!(warm_start, params_dict[var_symbol])
        end

        # Handle second stage T masking for warm start
        if "T" in var_list && second_stage
            t_idx = findfirst(==("T"), var_list)
            t_start = t_idx == 1 ? 0 : sum([length(params_dict[Symbol(var_list[i])]) for i in 1:(t_idx-1)])
            mask = vec(mask_emp_gamma_ls)
            t_length = length(params_dict[:T])
            t_indices = (t_start + 1):(t_start + t_length)

            # Extract only non-masked T values
            warm_start_T = warm_start[t_indices][mask .== 1]
            warm_start = vcat(
                warm_start[1:t_start],
                warm_start_T,
                (t_start + t_length < length(warm_start)) ? warm_start[t_start + t_length + 1:end] : Float64[]
            )
        end

        println("\n[WARM START]")
        println("  Previous best extracted from: $last_stage_folder")
        println("  Warm start dimension: $(length(warm_start))")
        println("  Bounds dimension: $(length(lb))")
        @assert length(warm_start) == length(lb) "Warm start dimension mismatch!"
    end

    # Reconstruct the full (level) parameter vector from a stage particle, BEFORE
    # T-profiling. Factored out so both `objective` and the T-convergence probe
    # (below) build x_full through the exact same path.
    function build_x_full(x_stage)
        if last_stage_folder !== nothing
            # Reconstruct full parameter vector
            x_full = copy(best_params_prev)
            var_list = isa(variable_list, String) ? [variable_list] : variable_list

            for (var_idx, var) in enumerate(var_list)
                var_symbol = Symbol(var)
                param_start = get_param_start_index(var_symbol)
                var_len = length(params_dict[var_symbol])

                stage_start = var_idx == 1 ? 1 : sum([length(params_dict[Symbol(var_list[i])]) for i in 1:(var_idx-1)]) + 1
                stage_end = stage_start + var_len - 1

                if var == "T"
                    # stage holds free φ (length N_T_FREE); expand to level T block
                    # (length N_T_REDUCED, ref entries = 1) before writing to x_full.
                    x_full[param_start:(param_start + N_T_REDUCED - 1)] = t_free_phi_to_levels(x_stage[stage_start:stage_end])
                else
                    x_full[param_start:(param_start + var_len - 1)] = x_stage[stage_start:stage_end]
                end
            end
            return x_full
        else
            # profile_T fresh start: x_stage is head-only ⇒ append an inert T block
            # (overwritten by profiled_theta below). Otherwise expand φ → levels.
            return profile_T ? vcat(x_stage, T_PLACEHOLDER) : search_to_full(x_stage)
        end
    end

    # Define objective function
    function objective(x_stage)
        x_full = build_x_full(x_stage)

        # T-profiling: replace the (searched-or-stale) T block with T*(α,Ω,A).
        profile_T && (x_full = profiled_theta(x_full))

        # Evaluate SMM (or analytical GMM)
        result = parallel_SMM_safe(x_full, false, second_stage, method, false;
                                   precomputed_tau=cached_tau, u_draws=u_draws, sample_weights=sample_weights,
                                   W_override=weight_matrix, moment_blocks=moment_blocks,
                                   analytical=analytical, n_quad=n_quad)

        if isnothing(result)
            return Inf
        else
            return result[1][1] # Score
        end
    end

    # Run the selected optimizer (PSO or CMA-ES) with warm start
    println("\n" * "="^60)
    println("Starting Optimization (backend = :$OPTIMIZER_BACKEND)")
    println("="^60)
    if variable_list !== nothing
        var_str = isa(variable_list, String) ? variable_list : join(variable_list, ", ")
        println("Optimizing variables: $var_str")
    end
    println("Particles/λ: $n_particles")
    println("Max iterations/generations: $max_iter")
    println("Dimension: $(length(lb))")
    println("="^60)

    # Under profiling, expose a swarm-level T-convergence probe: maps a stage
    # particle to whether its GE-Sinkhorn T inversion converged. The backend runs
    # it over the whole swarm at each periodic report so the log shows how many
    # particles carry a non-converged (fallback) T. `nothing` when not profiling.
    t_conv_probe = profile_T ? (x_stage -> profiled_theta_converged(build_x_full(x_stage))) : nothing

    best_params, best_fitness, history = optimize_stage(
        objective,
        lb, ub;
        x0 = warm_start,                   # CRITICAL: previous best (warm start / incumbent)
        n_particles = n_particles,
        max_iter = max_iter,
        alpha_constraint = alpha_constraint,
        alpha_indices = alpha_indices,
        verbose = true,
        t_conv_probe = t_conv_probe,
    )

    # If optimizing subset, reconstruct full vector (T: φ → level, ref = 1)
    if last_stage_folder !== nothing
        final_params = copy(best_params_prev)
        var_list = isa(variable_list, String) ? [variable_list] : variable_list

        for (var_idx, var) in enumerate(var_list)
            var_symbol = Symbol(var)
            param_start = get_param_start_index(var_symbol)
            var_len = length(params_dict[var_symbol])

            stage_start = var_idx == 1 ? 1 : sum([length(params_dict[Symbol(var_list[i])]) for i in 1:(var_idx-1)]) + 1
            stage_end = stage_start + var_len - 1

            if var == "T"
                final_params[param_start:(param_start + N_T_REDUCED - 1)] = t_free_phi_to_levels(best_params[stage_start:stage_end])
            else
                final_params[param_start:(param_start + var_len - 1)] = best_params[stage_start:stage_end]
            end
        end
    else
        final_params = profile_T ? vcat(best_params, T_PLACEHOLDER) : search_to_full(best_params)
    end

    # T-profiling: the returned θ carries T*(α̂,Ω̂,Â) so downstream reporting and
    # Step-4 inference (which keeps the full T columns) see the profiled T.
    profile_T && (final_params = profiled_theta(final_params))

    return final_params, best_fitness, history
end


# ═══════════════════════════════════════════════════════════════════════════
# Per-step orchestrator (backend-agnostic)
# ═══════════════════════════════════════════════════════════════════════════

"""
    run_optimization(; kwargs...) -> (best_params, best_fitness)

Unified optimizer wrapper for Steps 1 and 3 of three-step SMM. Backend-agnostic:
the concrete optimizer (PSO or CMA-ES) is selected by `OPTIMIZER_BACKEND` inside
`train_stage` → `optimize_stage`. Under `:pso` it runs the staged refinement loop;
under `:cmaes` the loop collapses to the single joint Stage-1 run.

`run_pso_optimization` is retained as a backward-compatible alias.

# Keyword arguments
- `weight_matrix`: SMM weight matrix passed to full_SMM (default: uses global Weight_matrix_custom)
- `skip_initial_alpha_search`: if true, skip Stage 0 LHS search (use warm_start_params alpha)
- `warm_start_params`: full parameter vector to warm-start Stage 1 (nothing = fresh start)
- `output_subfolder`: subfolder under output_folder for all stage outputs
- `max_loop`: number of refinement loops (PSO backend only; default 50)
- `n_particles`, `max_iter_initial`, `max_iter_stage`: optimizer configuration
- `alpha_search_method`: "log_grid" or "lhs"
- `alpha_selection_criterion`: "reg_coef" or "score"
"""
function run_optimization(;
    weight_matrix::Union{Nothing, AbstractMatrix} = nothing,
    skip_initial_alpha_search::Bool = false,
    warm_start_params::Union{Nothing, Vector{Float64}} = nothing,
    output_subfolder::String = "step1",
    max_loop::Int = 50,
    n_particles::Int = 100,
    max_iter_initial::Int = 200,
    max_iter_stage::Int = 50,
    alpha_search_method::String = "lhs",
    alpha_selection_criterion::String = "reg_coef",
    length_range_alpha::Int = 40,
    method::String = "original",
    gamma_beta_only::Bool = false,          # step 3: fix structural params, optimize only alpha+T
    moments_loss_gamma_beta::Bool = false,  # step 3: compute loss on gamma_ls + reg_coef moments only
    analytical::Bool = false,              # GMM mode: closed-form moments (no simulation)
    n_quad::Int = 200,                     # quadrature nodes for reg_coef block in analytical mode
    profile_T::Bool = false                # Design A: profile T out of the search (profiling.jl)
)
    loop_base = joinpath(output_folder, output_subfolder)
    mkpath(loop_base)

    moment_blocks = moments_loss_gamma_beta ? [4, 5] : nothing   # reg_coef + gamma_ls (β+γ)

    best_params = nothing
    best_fitness = Inf
    stage = 0

    # ── Stage 0: LHS alpha search ─────────────────────────────────────────────
    if !skip_initial_alpha_search
        println("\n" * "="^70)
        println("[$output_subfolder] STAGE 0: Finding good initial alpha values")
        println("="^70)

        # Anchor the coarse α search to [×BOUND_LO, ×BOUND_HI] × the α prior (N_TAU==1)
        # so the selected init_alpha lands inside the prior-anchored Stage-1 box; else the
        # historical [0.5, 1.5] range.
        if N_TAU == 1 && TAU_PRIOR !== nothing
            alpha_min = TAU_PRIOR[1] * BOUND_LO
            alpha_max = TAU_PRIOR[1] * BOUND_HI
        else
            alpha_min = 0.5
            alpha_max = 1.5
        end
        println("Search is done $alpha_search_method with min $alpha_min and max $alpha_max")
        if N_TAU == 1
            length_range_alpha = 10000
        end
        if alpha_search_method == "log_grid"
            alpha_candidates = generate_initial_alphas("log_grid", N_TAU, alpha_min, alpha_max;
                                                     log_grid_length=length_range_alpha)
        else
            alpha_candidates = generate_initial_alphas("lhs", N_TAU, alpha_min, alpha_max;
                                                     lhs_n_samples=10000)
        end
        println("  Generated $(length(alpha_candidates)) alpha candidates")

        A_init = copy(emp_pi_r_full).^(1/abs(epsilon)) .* regional_wages[N_downstream_per_region .!= 0]
        A_init ./= sum(A_init)
        T_init_nz = vec(permutedims(T_rs_init))[T_MASK]   # s-major to match T_MASK
        # New layout: [Ω^L | Ω^s | A | α(N_TAU) | T] — alpha is inserted between A and T
        init_other_prefix = vcat([agg_labor_share], agg_industry_share, A_init)
        expanding_alpha = [vcat(init_other_prefix, alpha, T_init_nz) for alpha in alpha_candidates]

        results_ = pmap(p -> parallel_SMM_safe(p; u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS,
                                               W_override=weight_matrix,
                                               analytical=analytical, n_quad=n_quad), expanding_alpha)

        if alpha_selection_criterion == "reg_coef"
            reg_coefs_sim = [r !== nothing ? r[2][4] : fill(NaN, N_REG) for r in results_]
            reg_distances  = [sum((reg_coef .- rc).^2) for rc in reg_coefs_sim]
            best_idx = argmin(reg_distances)
        else
            scores = [r !== nothing ? r[1][1] : Inf for r in results_]
            best_idx = argmin(scores)
        end
        init_alpha  = alpha_candidates[best_idx]
        println("  Best initial alpha (trade-cost params, length $N_TAU): ",
                round.(init_alpha, digits=6))
        if alpha_selection_criterion == "reg_coef"
            println("  Simulated reg_coef at best alpha: ",
                    round.(reg_coefs_sim[best_idx], digits=6))
            println("  Empirical  reg_coef:                  ",
                    round.(reg_coef, digits=6))
        end
    else
        @assert warm_start_params !== nothing "skip_initial_alpha_search=true requires warm_start_params"
        alpha_start_idx = S + R_downstream + 2   # new layout: [Ω^L | Ω^s(S) | A(Rd) | α(N_TAU) | T]
        init_alpha = warm_start_params[alpha_start_idx:(alpha_start_idx + N_TAU - 1)]
        println("\n[$output_subfolder] Skipping Stage 0: using warm_start alpha $(round.(init_alpha, digits=6))")
    end

    # ── Stage 1: initial joint fit ───────────────────────────────────────────
    println("\n" * "="^70)
    println("[$output_subfolder] STAGE 1: initial joint fit (backend = :$OPTIMIZER_BACKEND)")
    println("="^70)

    stage = 0

    if gamma_beta_only
        # Save warm_start_params as a seed folder so train_stage can treat it as a
        # "previous stage" and restrict optimisation to alpha+T only.
        @assert warm_start_params !== nothing "gamma_beta_only=true requires warm_start_params"
        seed_folder = joinpath(loop_base, "seed")
        mkpath(seed_folder)
        NPZ.npzwrite(joinpath(seed_folder, "best_params.npy"), reshape(warm_start_params, :, 1))

        println("[$output_subfolder] gamma_beta_only: optimising α then T (decoupled; A_r/labor/industry fixed at θ̂_1)")
        # Decoupled initial fit: α first (T held at the warm-start init), then T
        # (α held at its just-fitted value). Each sub-stage is a single-variable
        # train_stage; the α result seeds the T fit via a scratch folder.
        best_params, best_fitness, history = train_stage(
            n_particles, max_iter_initial;
            variable_list     = ["alpha"],
            last_stage_folder = seed_folder,
            K=1, radius=0.5, second_stage=false, method=method,
            u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS, weight_matrix=weight_matrix,
            moment_blocks=moment_blocks, analytical=analytical, n_quad=n_quad,
            profile_T=profile_T
        )
        # profile_T ⇒ T is profiled inside the α stage (its objective assigns
        # T*(α,Ω,A)); the separate T sub-stage is not needed.
        if !profile_T
            seed_alpha_folder = joinpath(loop_base, "seed_alpha")
            mkpath(seed_alpha_folder)
            NPZ.npzwrite(joinpath(seed_alpha_folder, "best_params.npy"), reshape(best_params, :, 1))
            best_params, best_fitness, history = train_stage(
                n_particles, max_iter_initial;
                variable_list     = ["T"],
                last_stage_folder = seed_alpha_folder,
                K=1, radius=0.5, second_stage=false, method=method,
                u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS, weight_matrix=weight_matrix,
                moment_blocks=moment_blocks, analytical=analytical, n_quad=n_quad
            )
        end
    else
        best_params, best_fitness, history = train_stage(
            n_particles, max_iter_initial;
            init_alpha      = init_alpha,
            variable_list  = nothing,
            last_stage_folder = nothing,
            radius          = 0.5,
            second_stage   = false,
            method         = method,
            u_draws        = U_DRAWS,
            sample_weights = SAMPLE_WEIGHTS,
            weight_matrix  = weight_matrix,
            warm_start_override = warm_start_params,
            moment_blocks  = moment_blocks,
            analytical     = analytical,
            n_quad         = n_quad,
            profile_T      = profile_T
        )
    end

    stage0_folder = joinpath(loop_base, string(stage))
    mkpath(stage0_folder)
    NPZ.npzwrite(joinpath(stage0_folder, "best_params.npy"), reshape(best_params, :, 1))
    generate_report(loop_base, string(stage), 1, nothing, best_params, "";
                    u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS,
                    analytical=analytical, n_quad=n_quad)

    # ── Refinement loops ─────────────────────────────────────────────────────
    # α and T are DECOUPLED: every loop refines α alone, then T alone (α first),
    # so each sub-stage lets one block adapt to the other's latest update instead
    # of moving them together in one joint step. PSO AND TikTak run the staged
    # block-coordinate refinement (productivity → α → T → technical, or the
    # α-then-T alternation for gamma_beta_only): each sub-stage re-runs the backend
    # on one parameter block (warm-started at the incumbent, monotone-floored), so
    # TikTak gets the same successive-block descent PSO does. Only CMA-ES collapses
    # the general (non-gamma_beta_only) case into the single joint Stage 1 run above
    # (it learns cross-block covariance directly and stops on its own ftol/xtol);
    # the gamma_beta_only α-then-T alternation still runs for every backend.
    radius_start, radius_end = 0.3, 0.9
    # gamma_beta_only: 2 sub-stages/loop (α, then T — all backends).
    # else: four sub-stages/loop (productivity, α, T, technical — PSO & TikTak;
    # CMA-ES skips this case).
    # profile_T drops the standalone T sub-stage from each loop: gamma_beta_only
    # goes 2→1 (α only), the joint path 4→3 (productivity, α, technical).
    substages_per_loop = gamma_beta_only ? (profile_T ? 1 : 2) : (profile_T ? 3 : 4)
    # Only CMA-ES skips the joint-path block-coordinate refinement; PSO and TikTak run it.
    run_refinement = gamma_beta_only || OPTIMIZER_BACKEND != :cmaes

    for loop in (run_refinement ? (1:max_loop) : (1:0))
        radius = radius_start + (loop - 1) * (radius_end - radius_start) / (max_loop - 1)
        past_loop_folder = loop == 1 ? loop_base : joinpath(loop_base, "epoch_$(loop-1)")
        loop_folder = joinpath(loop_base, "epoch_$loop")
        mkpath(loop_folder)

        println("\n[$output_subfolder] LOOP $loop/$max_loop  radius=$radius")

        if gamma_beta_only
            # Decoupled α/T refinement (both backends): α alone, then T alone.
            # A_r / labor / industry shares stay fixed at warm start throughout.
            # Each sub-stage lets α adapt to the previous T update and vice versa.
            OPTIMIZER_BACKEND == :cmaes && (max_iter_stage = 100)
            substage_folder = past_loop_folder
            # profile_T ⇒ refine α alone (T is profiled inside its objective); the
            # standalone T sub-stage is dropped.
            for var in (profile_T ? ("alpha",) : ("alpha", "T"))
                best_params, best_fitness, history = train_stage(
                    n_particles, max_iter_stage;
                    variable_list     = [var],
                    last_stage_folder = joinpath(substage_folder, string(stage)),
                    K=1, radius=radius, second_stage=false, method=method,
                    u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS, weight_matrix=weight_matrix,
                    moment_blocks=moment_blocks, analytical=analytical, n_quad=n_quad,
                    profile_T=profile_T
                )
                stage += 1
                folder = joinpath(loop_folder, string(stage)); mkpath(folder)
                NPZ.npzwrite(joinpath(folder, "best_params.npy"), reshape(best_params, :, 1))
                generate_report(loop_folder, string(stage), 1, [var], best_params, string(radius);
                                u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS,
                                analytical=analytical, n_quad=n_quad)
                substage_folder = loop_folder
            end
        else
            # Sub-stage 1: Productivity
            radius_prod = 0.7 + 0.2 * radius
            best_params, best_fitness, history = train_stage(
                n_particles, max_iter_stage;
                variable_list     = ["productivity"],
                last_stage_folder = joinpath(past_loop_folder, string(stage)),
                K=1, radius=radius_prod, second_stage=false, method=method,
                u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS, weight_matrix=weight_matrix,
                moment_blocks=moment_blocks, analytical=analytical, n_quad=n_quad,
                profile_T=profile_T
            )
            stage += 1
            folder = joinpath(loop_folder, string(stage)); mkpath(folder)
            NPZ.npzwrite(joinpath(folder, "best_params.npy"), reshape(best_params, :, 1))
            generate_report(loop_folder, string(stage), 1, ["productivity"], best_params, string(radius_prod);
                            u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS,
                            analytical=analytical, n_quad=n_quad)

            # Sub-stage 2a: Spatial structure — α alone (T held), decoupled from T
            best_params, best_fitness, history = train_stage(
                n_particles, max_iter_stage;
                variable_list     = ["alpha"],
                last_stage_folder = joinpath(loop_folder, string(stage)),
                K=1, radius=radius, second_stage=false, method=method,
                u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS, weight_matrix=weight_matrix,
                moment_blocks=moment_blocks, analytical=analytical, n_quad=n_quad,
                profile_T=profile_T
            )
            stage += 1
            folder = joinpath(loop_folder, string(stage)); mkpath(folder)
            NPZ.npzwrite(joinpath(folder, "best_params.npy"), reshape(best_params, :, 1))
            generate_report(loop_folder, string(stage), 1, ["alpha"], best_params, string(radius);
                            u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS,
                            analytical=analytical, n_quad=n_quad)

            # Sub-stage 2b: Spatial structure — T alone (α held at its 2a value).
            # profile_T ⇒ skipped: T is profiled inside every objective, so the
            # α (2a) output already carries the up-to-date T*; technical (3) below
            # continues directly from it (its stage counter is not advanced here).
            if !profile_T
                best_params, best_fitness, history = train_stage(
                    n_particles, max_iter_stage;
                    variable_list     = ["T"],
                    last_stage_folder = joinpath(loop_folder, string(stage)),
                    K=1, radius=radius, second_stage=false, method=method,
                    u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS, weight_matrix=weight_matrix,
                    moment_blocks=moment_blocks, analytical=analytical, n_quad=n_quad
                )
                stage += 1
                folder = joinpath(loop_folder, string(stage)); mkpath(folder)
                NPZ.npzwrite(joinpath(folder, "best_params.npy"), reshape(best_params, :, 1))
                generate_report(loop_folder, string(stage), 1, ["T"], best_params, string(radius);
                                u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS,
                                analytical=analytical, n_quad=n_quad)
            end

            # Sub-stage 3: Technical coefficients
            best_params, best_fitness, history = train_stage(
                n_particles, max_iter_stage;
                variable_list     = ["agg_labor_share_tech", "agg_industry_share_tech"],
                last_stage_folder = joinpath(loop_folder, string(stage)),
                K=1, radius=radius, second_stage=false, method=method,
                u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS, weight_matrix=weight_matrix,
                moment_blocks=moment_blocks, analytical=analytical, n_quad=n_quad,
                profile_T=profile_T
            )
            stage += 1
            folder = joinpath(loop_folder, string(stage)); mkpath(folder)
            NPZ.npzwrite(joinpath(folder, "best_params.npy"), reshape(best_params, :, 1))
            generate_report(loop_folder, string(stage), 1,
                            ["agg_labor_share_tech", "agg_industry_share_tech"], best_params, string(radius);
                            u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS,
                            analytical=analytical, n_quad=n_quad)
        end

        println("  ✓ Loop $loop done. Fitness: $(round(best_fitness, digits=6))")

        # Convergence check
        if loop > 2
            prev_folder = joinpath(loop_base, "epoch_$(loop-1)", string(stage - substages_per_loop))
            prev_params = NPZ.npzread(joinpath(prev_folder, "best_params.npy"))[:, 1]
            param_change = maximum(abs.(best_params .- prev_params) ./ (abs.(prev_params) .+ 1e-10))
            if param_change < 1e-6
                println("  Convergence: Δparams < 1e-6. Stopping.")
                break
            end
        end
    end

    run_reporting(loop_base, max_loop; u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS,
                  analytical=analytical, n_quad=n_quad)
    return best_params, best_fitness
end


# ═══════════════════════════════════════════════════════════════════════════
# Backward-compatible aliases (legacy names)
# ═══════════════════════════════════════════════════════════════════════════
const train_stage_pso      = train_stage        # legacy name (main_pso.jl)
const run_pso_optimization = run_optimization   # legacy name (main_gmm.jl, run_internal_validity.jl)
