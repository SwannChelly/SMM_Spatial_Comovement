"""
Particle Swarm Optimization integration for SMM calibration - FIXED VERSION
Key fix: Always includes previous best as a warm start particle for monotonic improvement
Additional fix: Enforce [0, 1] bounds for agg_labor_share_tech
"""

using Distributed
using Random
using Statistics
using NPZ
using Printf

"""
    parallel_pso_smm(objective_func, lb, ub; warm_start_particle=nothing, kwargs...)

PSO optimized for your SMM calibration with parallel evaluation.

**KEY FEATURE**: warm_start_particle ensures previous best is always included,
guaranteeing monotonic improvement across stages.

# Arguments
- `objective_func`: Function that takes params vector and returns loss
- `lb`: Lower bounds for parameters
- `ub`: Upper bounds for parameters
- `n_particles`: Number of particles (recommend = number of cores)
- `max_iter`: Maximum iterations
- `warm_start_particle`: Previous best solution (CRITICAL for monotonic improvement!)
- `w_start`: Initial inertia (1.5 = more exploration)
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
    warm_start_particle::Union{Vector{Float64}, Nothing} = nothing,
    w_start::Float64 = 0.9,
    w_end::Float64 = 0.4,
    c1::Float64 = 1.5,
    c2::Float64 = 2.5,
    beta_constraint::Bool = true,
    beta_indices::UnitRange = 1:N_beta,  # Changed from 1:5
    verbose::Bool = true
)
    
    d = length(lb)
    
    # CRITICAL: Reserve one slot for warm start if provided
    n_random = warm_start_particle === nothing ? n_particles : n_particles - 1
    particles = [lb .+ rand(d) .* (ub .- lb) for _ in 1:n_random]
    
    # Apply beta constraint to random particles
    if beta_constraint
        for i in 1:n_random
            particles[i] = enforce_beta_constraint(particles[i], beta_indices)
        end
    end
    
    # CRITICAL FIX: Add previous best as a particle (guarantees monotonic improvement)
    if warm_start_particle !== nothing
        # Ensure warm start is within current bounds (bounds may have changed)
        warm_start_clamped = clamp.(warm_start_particle, lb, ub)
        
        # Apply beta constraint
        if beta_constraint
            warm_start_clamped = enforce_beta_constraint(warm_start_clamped, beta_indices)
        end
        
        push!(particles, warm_start_clamped)
        
        if verbose
            println("[PSO] ✓ Including previous best as warm start particle")
            println("  This guarantees fitness will not increase from previous stage")
        end
    end
    
    # Initialize velocities (including for warm start particle)
    velocities = [0.1 * (ub .- lb) .* randn(d) for _ in 1:length(particles)]
    
    # PARALLEL EVALUATION - This is where we leverage your many cores
    if verbose
        println("\n[PSO] Evaluating $(length(particles)) initial particles in parallel...")
        flush(stdout)
    end
    
    fitness = pmap(objective_func, particles)
    
    # Handle any failed evaluations (returning nothing)
    for i in 1:length(particles)
        if isnothing(fitness[i])
            fitness[i] = Inf
        end
    end
    
    # Personal best
    p_best = copy(particles)
    p_best_fitness = copy(fitness)
    
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
    
    # Track last printed fitness for improvement display
    last_printed_fitness = g_best_fitness
    last_printed_iter = 0

    # Track previous loss decomposition for Δ reporting
    prev_c = nothing
    
    if verbose
        println("[PSO] Initialization complete:")
        println("  Workers: $(nworkers())")
        println("  Particles: $(length(particles))")
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
        for i in 1:length(particles)
            r1, r2 = rand(d), rand(d)
            
            # Update velocity: inertia + cognitive + social components
            velocities[i] = w * velocities[i] .+ 
                           c1 * r1 .* (p_best[i] .- particles[i]) .+
                           c2 * r2 .* (g_best .- particles[i])
            
            # Velocity clamping (prevent particles from moving too fast)
            v_max = 0.1 * (ub .- lb)
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
        for i in 1:length(particles)
            if isnothing(fitness[i])
                fitness[i] = Inf
            end
        end
        
        # Update personal bests
        for i in 1:length(particles)
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

        # ============== ADD PARTICLE RESTART HERE ==============
        # Reinitialize stagnant particles to escape local minima
        if iter > 30 && iter % 25 == 0
            n_restarted = 0
            for i in 1:length(particles)
                # Check if particle hasn't improved recently
                if fitness[i] > g_best_fitness * 1.5  # Particle is far from best
                    particles[i] = lb .+ rand(d) .* (ub .- lb)
                    if beta_constraint
                        particles[i] = enforce_beta_constraint(particles[i], beta_indices)
                    end
                    velocities[i] = 0.1 * (ub .- lb) .* randn(d)
                    p_best[i] = copy(particles[i])
                    p_best_fitness[i] = Inf  # Reset personal best
                    n_restarted += 1
                end
            end
            if verbose && n_restarted > 0
                println("[PSO] Restarted $n_restarted stagnant particles")
            end
        end
        # ========================================================
        
        # Store history
        push!(history["best_fitness"], g_best_fitness)
        push!(history["mean_fitness"], mean(filter(isfinite, fitness)))
        push!(history["best_params"], copy(g_best))
        
        t_elapsed = time() - t_start
        
        if verbose && (iter % 5 == 0 || iter == 1 || iter == max_iter)
            println("[PSO] Iteration $iter/$max_iter ($(round(t_elapsed, digits=2))s):")
            println("  Best fitness:     $(round(g_best_fitness, digits=6))")
            println("  Mean fitness:     $(round(mean(filter(isfinite, fitness)), digits=6))")

            # Loss decomposition by block
            #c, blocks, total = loss_decomposition(g_best)
            #if prev_c !== nothing
            #    ΔL = total - sum(prev_c)
            #    if abs(ΔL) > 1e-12
            #        prev_blocks = [sum(prev_c[r]) for r in BLOCK_RANGES]
            #        Δblocks = blocks .- prev_blocks
            #        parts = join([
            #            @sprintf("%s:%+.0f%%", BLOCK_NAMES[k], 100*Δblocks[k]/abs(ΔL))
            #            for k in 1:5
            #        ], " | ")
            #        println("  Δ by block:       $parts")
            #    end
            #end
            #prev_c = c

            # Keep existing improvement logging below...
            if last_printed_iter > 0
                improvement = last_printed_fitness - g_best_fitness
                pct_improvement = 100 * improvement / last_printed_fitness
                println("  Improvement:      $(round(improvement, digits=6)) ($(round(pct_improvement, digits=2))% since iter $last_printed_iter)")
            end
            println()
            flush(stdout)

            last_printed_fitness = g_best_fitness
            last_printed_iter = iter
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
    train_stage_pso(n_particles, max_iter; kwargs...)

PSO-based training with automatic warm start from previous stage.

**KEY FEATURE**: Automatically extracts previous best and passes it as warm start,
ensuring monotonic improvement across optimization stages.

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
- `best_params`: Best parameter vector found (full parameter space)
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
    second_stage = false,
    method = false,
    u_draws::Union{Nothing, Matrix{Float64}} = nothing,
    sample_weights::Union{Nothing, Vector{Float64}} = nothing,
    weight_matrix::Union{Nothing, AbstractMatrix} = nothing,
    warm_start_override::Union{Nothing, Vector{Float64}} = nothing
)
    
    # Build bounds based on previous stage or initialization
    if last_stage_folder === nothing
        # Fresh start - use init_beta


        A = copy(emp_pi_r_full).^(1/abs(epsilon)).*regional_wages[N_downstream_per_region .!= 0]  # analytical inversion
        A ./= sum(A) 
        
        T_init_nz = vec(T_rs_init)[T_MASK]   # Only non-zero T values
        lb = vcat(
            init_beta .* 0.5,
            0.8*agg_labor_share,
            0.8 .* agg_industry_share,
            0.8.* A,
            0.1 * T_init_nz
        )

        ub = vcat(
            init_beta .* 1.5,
            1.2*agg_labor_share,
            1.2 .* agg_industry_share,
            A .* 1.2,
            10 * T_init_nz
        )
        
        beta_constraint = true
        beta_indices = 1:N_beta
        best_params_prev = nothing
        # Use warm_start_override if provided (Step 3 warm-start from θ̂_1)
        warm_start = warm_start_override
        cached_tau = nothing  # Beta is being optimized in initial stage
        
    else
        # Continue from previous stage - load best params
        best_params_prev = NPZ.npzread(joinpath(last_stage_folder, "best_params.npy"))[:,K]
        names = [:beta, :agg_labor_share_tech, :agg_industry_share_tech, :productivity, :T]
        vals = unpack_params(best_params_prev)
        params_dict = Dict(names .=> vals)
        # T from unpack_params is full S*R; reduce to only non-zero entries for optimization
        params_dict[:T] = params_dict[:T][T_MASK]
        
        # Handle single variable or list
        var_list = isa(variable_list, String) ? [variable_list] : variable_list

        # Check if beta is fixed — if so, precompute and cache tau
        beta_is_fixed = !("beta" in var_list)
        cached_tau = beta_is_fixed ? build_tau(params_dict[:beta]) : nothing
        beta_is_fixed && println("[PSO] Beta is fixed — tau precomputed and cached")
        
        # Build bounds for selected variables
        # For most parameters: [value * alpha, value / alpha]
        # For agg_labor_share_tech: [value * (1-alpha), value * (1+alpha)] to stay in [0,1]
        lb_parts = Vector{Float64}[]
        ub_parts = Vector{Float64}[]
        
        for v in var_list
            val = params_dict[Symbol(v)]
            if v == "agg_labor_share_tech"
                # Symmetric percentage bounds: ±alpha around current value
                lb_v = val .* (1 - alpha)
                ub_v = val .* (1 + alpha)
                # Clamp to valid range [0.001, 1.0]
                lb_v = max.(lb_v, 0.001)
                ub_v = min.(ub_v, 1.0)
            else
                # Standard multiplicative bounds
                lb_v = val .* alpha
                ub_v = val ./ alpha
            end
            push!(lb_parts, isa(lb_v, Number) ? [lb_v] : lb_v)
            push!(ub_parts, isa(ub_v, Number) ? [ub_v] : ub_v)
        end
        
        lb = vcat(lb_parts...)
        ub = vcat(ub_parts...)
        
        # Check if beta is being optimized
        beta_constraint = "beta" in var_list
        if beta_constraint
            beta_idx_in_var = findfirst(==("beta"), var_list)
            beta_start = beta_idx_in_var == 1 ? 1 : sum([length(params_dict[Symbol(var_list[i])]) for i in 1:(beta_idx_in_var-1)]) + 1
            beta_indices = beta_start:(beta_start + N_beta - 1)  # Changed from + 4
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
        result = parallel_SMM_safe(x_full, false, second_stage, method, false;
                                   precomputed_tau=cached_tau, u_draws=u_draws, sample_weights=sample_weights,
                                   W_override=weight_matrix)
        
        if isnothing(result)
            return Inf
        else
            return result[1][1] # Score
        end
    end
    
    # Run PSO with warm start
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
        warm_start_particle = warm_start,  # CRITICAL: Pass previous best
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
        return N_beta + 1
    elseif param_name == :agg_industry_share_tech
        return N_beta + 2
    elseif param_name == :productivity
        return N_beta + 2 + S
    elseif param_name == :T
        return N_beta + 2 + S + R_downstream
    else
        error("Unknown parameter name: $param_name")
    end
end

# Number of T parameters (only non-zero gamma_ls entries)
function get_n_T_params()
    return sum(T_MASK)
end