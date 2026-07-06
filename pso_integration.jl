"""
PSO backend for the SMM optimizer layer.

Exposes the PSO half of the optimizer contract:
- `parallel_pso_smm(objective, lb, ub; warm_start_particle, ...) -> (best_x, best_f, history)`
- `enforce_beta_constraint` — the β-ordering repair, shared with the CMA-ES backend.

The backend-neutral orchestration (`optimize_stage`, `train_stage`,
`run_optimization`) lives in `optimizer.jl`; this file knows nothing about which
backend is selected. Key feature: `warm_start_particle` is always included as a
particle, guaranteeing monotone improvement across stages.
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
    beta_indices::UnitRange = 1:N_TAU,
    log_mask::Union{Nothing, Vector{Bool}} = nothing,
    verbose::Bool = false
)

    d = length(lb)

    # ── Search-space transform: flagged dims (T) optimised in log ──────────────
    # Confined to this function: bounds, particles, velocities, clamps and restarts
    # operate in log coordinate; the objective is exponentiated to levels before
    # evaluation, and the returned optimum is in levels. The Jacobian/inference
    # never see this (they consume best_params in levels).
    lmask = log_mask === nothing ? falses(d) : log_mask
    @assert length(lmask) == d "log_mask length $(length(lmask)) != dim $d"
    any(lmask) && @assert all(>(0), lb[lmask]) && all(>(0), ub[lmask]) "log dims need positive bounds"
    _fwd(y) = Float64[lmask[i] ? exp(y[i]) : y[i] for i in 1:d]   # search → level
    _inv(x) = Float64[lmask[i] ? log(x[i]) : x[i] for i in 1:d]   # level  → search
    lb = _inv(lb); ub = _inv(ub)
    warm_start_particle = warm_start_particle === nothing ? nothing : _inv(warm_start_particle)
    obj = y -> objective_func(_fwd(y))

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
    
    fitness = pmap(obj, particles)

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
        fitness = pmap(obj, particles)
        
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
    
    return _fwd(g_best), g_best_fitness, history
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
