"""
Particle Swarm Optimization integration for SMM calibration
This file provides functions to replace Halton grid search with PSO
while maintaining compatibility with your existing workflow.
"""

using Distributed
using Random
using Statistics
using NPZ

"""
    parallel_pso_smm(objective_func, lb, ub; n_particles=70, max_iter=100, kwargs...)

PSO optimized for your SMM calibration with parallel evaluation.

# Arguments
- `objective_func`: Function that takes params vector and returns loss
- `lb`: Lower bounds for parameters
- `ub`: Upper bounds for parameters
- `n_particles`: Number of particles (recommend = number of cores)
- `max_iter`: Maximum iterations
- `w_start`: Initial inertia (0.9 = more exploration)
- `w_end`: Final inertia (0.4 = more exploitation)
- `c1`: Cognitive parameter (personal best attraction)
- `c2`: Social parameter (global best attraction)
- `beta_constraint`: Enforce beta ordering if true
- `verbose`: Print progress

# Returns
- `best_params`: Best parameter vector found
- `best_fitness`: Best loss value
- `history`: Optimization history (for plotting)
"""
function parallel_pso_smm(
    objective_func::Function,
    lb::Vector{Float64},
    ub::Vector{Float64};
    n_particles::Int = 70,
    max_iter::Int = 100,
    w_start::Float64 = 0.9,
    w_end::Float64 = 0.4,
    c1::Float64 = 2.0,
    c2::Float64 = 2.0,
    beta_constraint::Bool = true,
    beta_indices::UnitRange = 1:5,
    verbose::Bool = true
)
    
    d = length(lb)
    
    # Initialize particles randomly in bounds
    particles = [lb .+ rand(d) .* (ub .- lb) for _ in 1:n_particles]
    
    # Apply beta constraint
    if beta_constraint
        for i in 1:n_particles
            particles[i] = enforce_beta_constraint(particles[i], beta_indices)
        end
    end
    
    # Initialize velocities
    velocities = [0.1 * (ub .- lb) .* randn(d) for _ in 1:n_particles]
    
    # PARALLEL EVALUATION - This is where we leverage your many cores
    if verbose
        println("\n[PSO] Evaluating $n_particles initial particles in parallel...")
        flush(stdout)
    end
    
    fitness = pmap(objective_func, particles)
    
    # Handle any failed evaluations (returning nothing)
    for i in 1:n_particles
        if isnothing(fitness[i])
            fitness[i] = Inf
        end
    end
    
    # Personal best
    p_best = copy(particles) # Param
    p_best_fitness = copy(fitness) # Score : Loss output
    
    # Global best
    g_best_idx = argmin(p_best_fitness)
    g_best = copy(p_best[g_best_idx])
    g_best_fitness = p_best_fitness[g_best_idx]
    
    # History tracking
    history = Dict(
        "best_fitness" => Float64[g_best_fitness],
        "mean_fitness" => Float64[mean(filter(isfinite, fitness))],
        "best_params" => [copy(g_best)]
    )
    
    if verbose
        println("[PSO] Initialization complete:")
        println("  Workers: $(nworkers())")
        println("  Particles: $n_particles")
        println("  Dimension: $d")
        println("  Initial best fitness: $(round(g_best_fitness, digits=6))")
        println()
    end
    
    # Main PSO loop
    for iter in 1:max_iter
        t_start = time()
        
        # Adaptive inertia weight (decreases over time)
        w = w_start - (w_start - w_end) * (iter / max_iter)
        
        # Update all particles
        for i in 1:n_particles
            r1, r2 = rand(d), rand(d)
            
            # Update velocity: inertia + cognitive + social components
            velocities[i] = w * velocities[i] .+ 
                           c1 * r1 .* (p_best[i] .- particles[i]) .+
                           c2 * r2 .* (g_best .- particles[i])
            
            # Velocity clamping (prevent particles from moving too fast)
            v_max = 0.2 * (ub .- lb)
            velocities[i] = clamp.(velocities[i], -v_max, v_max)
            
            # Update position
            particles[i] = particles[i] .+ velocities[i]
            
            # Enforce bounds
            particles[i] = clamp.(particles[i], lb, ub)
            
            # Enforce beta constraint
            if beta_constraint
                particles[i] = enforce_beta_constraint(particles[i], beta_indices)
            end
        end
        
        # PARALLEL EVALUATION - All particles evaluated at once
        fitness = pmap(objective_func, particles)
        
        # Handle failures
        for i in 1:n_particles
            if isnothing(fitness[i])
                fitness[i] = Inf
            end
        end
        
        # Update personal bests
        for i in 1:n_particles
            if fitness[i] < p_best_fitness[i]
                p_best[i] = copy(particles[i])
                p_best_fitness[i] = fitness[i]
            end
        end
        
        # Update global best
        current_best_idx = argmin(p_best_fitness)
        if p_best_fitness[current_best_idx] < g_best_fitness
            g_best = copy(p_best[current_best_idx])
            g_best_fitness = p_best_fitness[current_best_idx]
        end
        
        # Store history
        push!(history["best_fitness"], g_best_fitness)
        push!(history["mean_fitness"], mean(filter(isfinite, fitness)))
        push!(history["best_params"], copy(g_best))
        
        t_elapsed = time() - t_start
        
        if verbose && (iter % 5 == 0 || iter == 1 || iter == max_iter)
            println("[PSO] Iteration $iter/$max_iter ($(round(t_elapsed, digits=2))s):")
            println("  Best fitness:     $(round(g_best_fitness, digits=6))")
            println("  Mean fitness:     $(round(mean(filter(isfinite, fitness)), digits=6))")
            if iter > 1
                improvement = history["best_fitness"][end-1] - g_best_fitness
                pct_improvement = 100 * improvement / history["best_fitness"][1]
                println("  Improvement:      $(round(improvement, digits=6)) ($(round(pct_improvement, digits=2))%)")
            end
            println()
            flush(stdout)
        end
    end
    
    return g_best, g_best_fitness, history
end


"""
    enforce_beta_constraint(params, beta_indices)

Ensure beta parameters are ordered: β₁ < β₂ < β₃ < β₄ < β₅
"""
function enforce_beta_constraint(params::Vector{Float64}, beta_indices::UnitRange)
    params_new = copy(params)
    betas = params[beta_indices]
    betas_sorted = sort(betas)
    params_new[beta_indices] = betas_sorted
    return params_new
end


"""
    train_stage_pso(n_particles, max_iter, init_beta, variable_list, last_stage_folder, K, alpha, second_stage)

Replace train_stage_one with PSO-based training.
This integrates directly with your existing workflow.

# Arguments
- `n_particles`: Number of PSO particles (use your number of cores)
- `max_iter`: Number of PSO iterations (50-100 typically sufficient)
- `init_beta`: Initial beta values (if starting fresh)
- `variable_list`: Which parameters to optimize (e.g., ["beta", "T"])
- `last_stage_folder`: Folder with previous stage results
- `K`: Which parameter set to use from previous stage
- `alpha`: Search radius multiplier
- `second_stage`: Whether this is second-stage optimization

# Returns
- `best_params`: Best parameter vector found
- `best_fitness`: Best loss value achieved
- `history`: Optimization history
"""
function train_stage_pso(
    n_particles::Int,
    max_iter::Int;
    init_beta = nothing,
    variable_list = nothing,
    last_stage_folder = nothing,
    K = 1,
    alpha = 0.1,
    second_stage = false
)
    
    # Build bounds based on previous stage or initialization
    if last_stage_folder === nothing
        # Fresh start - use init_beta
        A = copy(N_downstream_per_region[N_downstream_per_region .!= 0])
        A ./= sum(A)
        
        lb = vcat(
            init_beta .* 0.5,
            0.8 * agg_labor_share,
            0.8 .* agg_industry_share,
            0.01 .* A,
            0.1 * ones(S*R)
        )
        
        ub = vcat(
            init_beta .* 2,
            1.2 * agg_labor_share,
            1.2 .* agg_industry_share,
            A .* 10,
            100 * ones(S*R)
        )
        
        beta_constraint = true
        beta_indices = 1:5
        
    else
        # Continue from previous stage - load best params
        best_params_prev = NPZ.npzread(joinpath(last_stage_folder, "best_params.npy"))[:,K]
        names = [:beta, :agg_labor_share_tech, :agg_industry_share_tech, :productivity, :T]
        vals = unpack_params(best_params_prev)
        params_dict = Dict(names .=> vals)
        
        # Handle single variable or list
        var_list = isa(variable_list, String) ? [variable_list] : variable_list
        
        # Build bounds for selected variables
        lb = vcat([params_dict[Symbol(v)] ./ alpha for v in var_list]...)
        ub = vcat([params_dict[Symbol(v)] .* alpha for v in var_list]...)
        
        # Check if beta is being optimized
        beta_constraint = "beta" in var_list
        if beta_constraint
            beta_idx_in_var = findfirst(==("beta"), var_list)
            beta_start = beta_idx_in_var == 1 ? 1 : sum([length(params_dict[Symbol(var_list[i])]) for i in 1:(beta_idx_in_var-1)]) + 1
            beta_indices = beta_start:(beta_start + 4)
        else
            beta_indices = 1:0
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
    end
    
    # Define objective function
    function objective(x_stage)
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
                
                x_full[param_start:(param_start + var_len - 1)] = x_stage[stage_start:stage_end]
            end
        else
            x_full = x_stage
        end
        
        # Evaluate SMM
        result = parallel_SMM_safe(x_full, false, second_stage, false)
        
        if isnothing(result)
            return Inf
        else
            return result[1][1] # Score
        end
    end
    
    # Run PSO
    println("\n" * "="^60)
    println("Starting PSO Optimization")
    println("="^60)
    if variable_list !== nothing
        var_str = isa(variable_list, String) ? variable_list : join(variable_list, ", ")
        println("Optimizing variables: $var_str")
    end
    println("Particles: $n_particles")
    println("Max iterations: $max_iter")
    println("Dimension: $(length(lb))")
    println("="^60)
    
    best_params, best_fitness, history = parallel_pso_smm(
        objective,
        lb, ub,
        n_particles = n_particles,
        max_iter = max_iter,
        beta_constraint = beta_constraint,
        beta_indices = beta_indices,
        verbose = true
    )
    
    # If optimizing subset, reconstruct full vector
    if last_stage_folder !== nothing
        final_params = copy(best_params_prev)
        var_list = isa(variable_list, String) ? [variable_list] : variable_list
        
        for (var_idx, var) in enumerate(var_list)
            var_symbol = Symbol(var)
            param_start = get_param_start_index(var_symbol)
            var_len = length(params_dict[var_symbol])
            
            stage_start = var_idx == 1 ? 1 : sum([length(params_dict[Symbol(var_list[i])]) for i in 1:(var_idx-1)]) + 1
            stage_end = stage_start + var_len - 1
            
            final_params[param_start:(param_start + var_len - 1)] = best_params[stage_start:stage_end]
        end
    else
        final_params = best_params
    end
    
    return final_params, best_fitness, history
end


"""
    get_param_start_index(param_name)

Get starting index of a parameter in the full parameter vector.
Structure: [beta(5), labor_share(1), industry_share(S), productivity(R_downstream), T(S*R)]
"""
function get_param_start_index(param_name::Symbol)
    if param_name == :beta
        return 1
    elseif param_name == :agg_labor_share_tech
        return 6
    elseif param_name == :agg_industry_share_tech
        return 7
    elseif param_name == :productivity
        return 7 + S
    elseif param_name == :T
        return 7 + S + R_downstream
    else
        error("Unknown parameter name: $param_name")
    end
end