using Distributed
using SharedArrays
using LinearAlgebra
using Random

"""
Parallel Particle Swarm Optimization optimized for multi-core systems.
Uses Julia's Distributed computing for particle evaluation.

Setup (add at top of your main script):
    using Distributed
    addprocs(70)  # Add 70 worker processes
    @everywhere using YourPackage  # Load your code on all workers
    @everywhere include("tools.jl")  # Include necessary files
"""

"""
    parallel_pso(objective_func, lb, ub; kwargs...)

Parallel Particle Swarm Optimization for multi-core optimization.
Evaluates all particles in parallel each iteration using Julia's pmap.

# Purpose
Minimizes objective_func by evolving a swarm of particles through parameter space.
Each particle represents a candidate solution. Particles move based on:
- Their own best position (cognitive component)
- The swarm's global best position (social component)
- Their current velocity (inertia)

# Arguments
- `objective_func::Function`: Function to minimize, takes Vector{Float64} and returns scalar
- `lb::Vector{Float64}`: Lower bounds for each parameter
- `ub::Vector{Float64}`: Upper bounds for each parameter

# Keyword Arguments
- `n_particles::Int=70`: Number of particles in swarm (recommend = number of cores)
- `max_iter::Int=100`: Maximum number of iterations
- `w_start::Float64=0.9`: Initial inertia weight (controls exploration vs exploitation)
                          Higher values = more exploration early on
- `w_end::Float64=0.4`: Final inertia weight (decreases linearly over iterations)
                        Lower values = more exploitation later
- `c1::Float64=2.0`: Cognitive parameter (attraction to personal best)
                     Higher = particles trust their own experience more
- `c2::Float64=2.0`: Social parameter (attraction to global best)
                     Higher = particles follow swarm more
- `beta_constraint::Bool=true`: Whether to enforce ordering constraint on beta parameters
                                (beta[1] < beta[2] < ... < beta[5])
- `beta_indices::UnitRange=1:5`: Indices of beta parameters in solution vector
- `verbose::Bool=true`: Print progress information during optimization
- `checkpoint_every::Int=10`: Save checkpoint every N iterations (for long runs)
- `checkpoint_file::String="pso_checkpoint.jld2"`: Filename for checkpoint saves

# Returns
- `best_params::Vector{Float64}`: Best parameter set found
- `best_fitness::Float64`: Objective function value at best parameters
- `history::Dict`: Dictionary containing optimization history:
    - "best_fitness": Best fitness at each iteration
    - "mean_fitness": Mean fitness of swarm at each iteration
    - "std_fitness": Standard deviation of fitness at each iteration
    - "best_params": Best parameter set at each iteration
    - "diversity": Swarm diversity metric at each iteration

# Example
```julia
function my_objective(x)
    return sum(x.^2)  # Minimize sum of squares
end

lb = zeros(10)
ub = ones(10)

best, fitness, hist = parallel_pso(
    my_objective, lb, ub,
    n_particles=50, max_iter=100
)
```

# Notes
- Requires Distributed.jl and worker processes (use addprocs() before calling)
- All particles are evaluated in parallel using pmap
- With 70 cores and 70 particles, each iteration evaluates all particles simultaneously
- Inertia weight decreases linearly from w_start to w_end (exploration → exploitation)
"""
function parallel_pso(
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
    verbose::Bool = true,
    checkpoint_every::Int = 10,
    checkpoint_file::String = "pso_checkpoint.jld2"
)
    
    d = length(lb)
    
    # Initialize particles
    particles = [lb .+ rand(d) .* (ub .- lb) for _ in 1:n_particles]
    
    # Apply beta constraint
    if beta_constraint
        for i in 1:n_particles
            particles[i] = enforce_beta_constraint(particles[i], beta_indices)
        end
    end
    
    # Initialize velocities
    velocities = [0.1 * (ub .- lb) .* randn(d) for _ in 1:n_particles]
    
    # Parallel evaluation of initial particles
    if verbose
        println("Evaluating initial $n_particles particles in parallel...")
        flush(stdout)
    end
    
    fitness = pmap(objective_func, particles)
    
    # Personal best
    p_best = copy(particles)
    p_best_fitness = copy(fitness)
    
    # Global best
    g_best_idx = argmin(p_best_fitness)
    g_best = copy(p_best[g_best_idx])
    g_best_fitness = p_best_fitness[g_best_idx]
    
    # History
    history = Dict(
        "best_fitness" => Float64[g_best_fitness],
        "mean_fitness" => Float64[mean(fitness)],
        "std_fitness" => Float64[std(fitness)],
        "best_params" => [copy(g_best)],
        "diversity" => Float64[]
    )
    
    if verbose
        println("\nPSO Initialization:")
        println("  Workers: $(nworkers())")
        println("  Particles: $n_particles")
        println("  Dimension: $d")
        println("  Initial best fitness: $g_best_fitness")
        println("  Initial mean fitness: $(mean(fitness))")
        println()
    end
    
    # Main PSO loop
    for iter in 1:max_iter
        t_start = time()
        
        # Adaptive inertia weight
        w = w_start - (w_start - w_end) * (iter / max_iter)
        
        # Update all particles
        for i in 1:n_particles
            r1, r2 = rand(d), rand(d)
            
            # Update velocity
            velocities[i] = w * velocities[i] .+ 
                           c1 * r1 .* (p_best[i] .- particles[i]) .+
                           c2 * r2 .* (g_best .- particles[i])
            
            # Velocity clamping
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
        
        # Parallel evaluation - THIS IS WHERE PARALLELIZATION HAPPENS
        fitness = pmap(objective_func, particles)
        
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
        
        # Calculate diversity (average distance to centroid)
        centroid = mean(particles)
        diversity = mean([norm(p .- centroid) for p in particles])
        
        # Store history
        push!(history["best_fitness"], g_best_fitness)
        push!(history["mean_fitness"], mean(fitness))
        push!(history["std_fitness"], std(fitness))
        push!(history["best_params"], copy(g_best))
        push!(history["diversity"], diversity)
        
        t_elapsed = time() - t_start
        
        # Checkpointing
        if iter % checkpoint_every == 0
            save_checkpoint(checkpoint_file, g_best, g_best_fitness, history, iter)
        end
        
        if verbose && (iter % 5 == 0 || iter == 1 || iter == max_iter)
            println("Iteration $iter/$max_iter ($(round(t_elapsed, digits=2))s):")
            println("  Best fitness:     $(round(g_best_fitness, digits=6))")
            println("  Mean fitness:     $(round(mean(fitness), digits=6))")
            println("  Std fitness:      $(round(std(fitness), digits=6))")
            println("  Diversity:        $(round(diversity, digits=6))")
            println("  Inertia (w):      $(round(w, digits=3))")
            if iter > 1
                improvement = history["best_fitness"][end-1] - g_best_fitness
                println("  Improvement:      $(round(improvement, digits=6))")
            end
            println()
            flush(stdout)
        end
    end
    
    return g_best, g_best_fitness, history
end


"""
    multi_swarm_pso(objective_func, lb, ub; kwargs...)

Multi-Swarm Particle Swarm Optimization with periodic migration between swarms.
Runs multiple independent swarms in parallel with occasional information exchange.

# Purpose
Better exploration of parameter space than single swarm PSO by maintaining multiple
independent search populations. Swarms periodically exchange their best solutions
(migration) to share information while maintaining diversity. Ideal when you have
many cores (e.g., 70 cores = 7 swarms × 10 particles each).

# Key Advantage
Multiple swarms explore different regions simultaneously, reducing risk of getting
stuck in local optima. Migration allows successful swarms to influence others while
maintaining independent exploration.

# Arguments
- `objective_func::Function`: Function to minimize, takes Vector{Float64} and returns scalar
- `lb::Vector{Float64}`: Lower bounds for each parameter
- `ub::Vector{Float64}`: Upper bounds for each parameter

# Keyword Arguments
- `n_swarms::Int=7`: Number of independent swarms to run in parallel
                     Recommend: sqrt(total_cores) to n_cores/5
- `particles_per_swarm::Int=10`: Number of particles in each swarm
                                  Total particles = n_swarms × particles_per_swarm
- `max_iter::Int=100`: Maximum number of iterations for each swarm
- `migration_interval::Int=20`: How often to exchange solutions between swarms (in iterations)
                                Smaller = more information sharing, larger = more independence
- `migration_rate::Float64=0.2`: Fraction of particles to migrate between swarms
                                  e.g., 0.2 = migrate top 20% of particles to neighboring swarm
- `beta_constraint::Bool=true`: Whether to enforce ordering constraint on beta parameters
- `beta_indices::UnitRange=1:5`: Indices of beta parameters in solution vector
- `verbose::Bool=true`: Print progress information during optimization

# Returns
- `best_params::Vector{Float64}`: Best parameter set found across all swarms
- `best_fitness::Float64`: Objective function value at best parameters
- `history::Dict`: Dictionary containing optimization history:
    - "best_fitness": Global best fitness at each iteration
    - "mean_fitness": Mean fitness across all swarms at each iteration
    - "best_params": Global best parameter set at each iteration
    - "swarm_bests": Best fitness of each swarm at each iteration (track diversity)

# Migration Strategy
At each migration interval:
1. Each swarm sends its N best particles to neighboring swarm
2. Receiving swarm replaces its N worst particles with immigrants
3. Forms a ring topology: Swarm 1 → 2 → 3 → ... → N → 1

# Example
```julia
# With 70 cores: 7 swarms × 10 particles = 70 parallel evaluations
best, fitness, hist = multi_swarm_pso(
    my_objective, lb, ub,
    n_swarms=7, 
    particles_per_swarm=10,
    max_iter=100,
    migration_interval=20
)
```

# When to Use Multi-Swarm vs Single Swarm
- Use multi-swarm when:
  * Parameter space is highly multimodal (many local optima)
  * You want robust global search
  * You have many cores (>30)
- Use single swarm when:
  * You need fast convergence to local optimum
  * Parameter space is relatively smooth
  * You have fewer cores
"""
function multi_swarm_pso(
    objective_func::Function,
    lb::Vector{Float64},
    ub::Vector{Float64};
    n_swarms::Int = 7,  # 7 swarms × 10 particles = 70 cores
    particles_per_swarm::Int = 10,
    max_iter::Int = 100,
    migration_interval::Int = 20,
    migration_rate::Float64 = 0.2,
    beta_constraint::Bool = true,
    beta_indices::UnitRange = 1:5,
    verbose::Bool = true
)
    
    d = length(lb)
    
    # Initialize swarms
    swarms = []
    for s in 1:n_swarms
        particles = [lb .+ rand(d) .* (ub .- lb) for _ in 1:particles_per_swarm]
        if beta_constraint
            particles = [enforce_beta_constraint(p, beta_indices) for p in particles]
        end
        velocities = [0.1 * (ub .- lb) .* randn(d) for _ in 1:particles_per_swarm]
        push!(swarms, (particles=particles, velocities=velocities))
    end
    
    # Evaluate all swarms in parallel
    all_particles = vcat([s.particles for s in swarms]...)
    all_fitness = pmap(objective_func, all_particles)
    
    # Split fitness back to swarms
    swarm_data = []
    idx = 1
    for s in 1:n_swarms
        n = particles_per_swarm
        fitness = all_fitness[idx:idx+n-1]
        p_best = copy(swarms[s].particles)
        p_best_fitness = copy(fitness)
        g_best_idx = argmin(fitness)
        
        push!(swarm_data, (
            particles = swarms[s].particles,
            velocities = swarms[s].velocities,
            fitness = fitness,
            p_best = p_best,
            p_best_fitness = p_best_fitness,
            g_best = copy(p_best[g_best_idx]),
            g_best_fitness = fitness[g_best_idx]
        ))
        idx += n
    end
    
    # Global best across all swarms
    global_best_swarm = argmin([sd.g_best_fitness for sd in swarm_data])
    g_best = copy(swarm_data[global_best_swarm].g_best)
    g_best_fitness = swarm_data[global_best_swarm].g_best_fitness
    
    history = Dict(
        "best_fitness" => Float64[g_best_fitness],
        "mean_fitness" => Float64[],
        "best_params" => [copy(g_best)],
        "swarm_bests" => [[sd.g_best_fitness for sd in swarm_data]]
    )
    
    if verbose
        println("Multi-Swarm PSO Initialization:")
        println("  Workers: $(nworkers())")
        println("  Swarms: $n_swarms")
        println("  Particles per swarm: $particles_per_swarm")
        println("  Total particles: $(n_swarms * particles_per_swarm)")
        println("  Initial best fitness: $g_best_fitness")
        println()
    end
    
    # Main loop
    for iter in 1:max_iter
        t_start = time()
        w = 0.9 - 0.5 * (iter / max_iter)
        
        # Update each swarm
        for s in 1:n_swarms
            sd = swarm_data[s]
            for i in 1:particles_per_swarm
                r1, r2 = rand(d), rand(d)
                sd.velocities[i] = w * sd.velocities[i] .+ 
                                   2.0 * r1 .* (sd.p_best[i] .- sd.particles[i]) .+
                                   2.0 * r2 .* (sd.g_best .- sd.particles[i])
                
                v_max = 0.2 * (ub .- lb)
                sd.velocities[i] = clamp.(sd.velocities[i], -v_max, v_max)
                sd.particles[i] = clamp.(sd.particles[i] .+ sd.velocities[i], lb, ub)
                
                if beta_constraint
                    sd.particles[i] = enforce_beta_constraint(sd.particles[i], beta_indices)
                end
            end
        end
        
        # Parallel evaluation of ALL particles
        all_particles = vcat([sd.particles for sd in swarm_data]...)
        all_fitness = pmap(objective_func, all_particles)
        
        # Update swarm data
        idx = 1
        for s in 1:n_swarms
            n = particles_per_swarm
            fitness = all_fitness[idx:idx+n-1]
            swarm_data[s] = (
                particles = swarm_data[s].particles,
                velocities = swarm_data[s].velocities,
                fitness = fitness,
                p_best = [fitness[i] < swarm_data[s].p_best_fitness[i] ? 
                         copy(swarm_data[s].particles[i]) : swarm_data[s].p_best[i] 
                         for i in 1:n],
                p_best_fitness = [min(fitness[i], swarm_data[s].p_best_fitness[i]) 
                                 for i in 1:n],
                g_best = swarm_data[s].g_best,
                g_best_fitness = swarm_data[s].g_best_fitness
            )
            
            # Update swarm's global best
            best_idx = argmin(swarm_data[s].p_best_fitness)
            if swarm_data[s].p_best_fitness[best_idx] < swarm_data[s].g_best_fitness
                swarm_data[s] = merge(swarm_data[s], (
                    g_best = copy(swarm_data[s].p_best[best_idx]),
                    g_best_fitness = swarm_data[s].p_best_fitness[best_idx]
                ))
            end
            idx += n
        end
        
        # Migration between swarms
        if iter % migration_interval == 0
            n_migrants = max(1, Int(floor(migration_rate * particles_per_swarm)))
            for s in 1:n_swarms
                target_swarm = mod(s, n_swarms) + 1
                # Send best particles to neighboring swarm
                best_indices = sortperm(swarm_data[s].p_best_fitness)[1:n_migrants]
                worst_indices = sortperm(swarm_data[target_swarm].p_best_fitness)[end-n_migrants+1:end]
                
                for (bi, wi) in zip(best_indices, worst_indices)
                    swarm_data[target_swarm].particles[wi] = copy(swarm_data[s].p_best[bi])
                end
            end
        end
        
        # Update global best
        for sd in swarm_data
            if sd.g_best_fitness < g_best_fitness
                g_best = copy(sd.g_best)
                g_best_fitness = sd.g_best_fitness
            end
        end
        
        push!(history["best_fitness"], g_best_fitness)
        push!(history["mean_fitness"], mean(all_fitness))
        push!(history["best_params"], copy(g_best))
        push!(history["swarm_bests"], [sd.g_best_fitness for sd in swarm_data])
        
        if verbose && (iter % 5 == 0 || iter == 1)
            println("Iteration $iter/$max_iter ($(round(time()-t_start, digits=2))s):")
            println("  Global best: $(round(g_best_fitness, digits=6))")
            println("  Mean fitness: $(round(mean(all_fitness), digits=6))")
            println("  Swarm bests: ", [round(sd.g_best_fitness, digits=4) for sd in swarm_data])
            println()
        end
    end
    
    return g_best, g_best_fitness, history
end


"""
    enforce_beta_constraint(params, beta_indices)

Enforce ordering constraint on beta parameters: β₁ < β₂ < β₃ < β₄ < β₅

# Purpose
In your SMM model, beta parameters must be ordered (e.g., representing increasing
substitution elasticities or other monotonic relationship). This function ensures
any proposed parameter set satisfies this constraint by sorting the beta values.

# Arguments
- `params::Vector{Float64}`: Full parameter vector
- `beta_indices::UnitRange`: Indices where beta parameters are located in params
                              e.g., 1:5 if betas are first 5 elements

# Returns
- `Vector{Float64}`: New parameter vector with sorted beta values, other params unchanged

# Example
```julia
params = [3.0, 1.0, 2.0, 5.0, 4.0, 0.8, 0.5, ...]  # betas are out of order
                                                     # (indices 1:5)
corrected = enforce_beta_constraint(params, 1:5)
# corrected = [1.0, 2.0, 3.0, 4.0, 5.0, 0.8, 0.5, ...]  # betas now sorted
```

# Note
This is called automatically within PSO after each position update to ensure
all candidate solutions remain feasible.
"""
function enforce_beta_constraint(params::Vector{Float64}, beta_indices::UnitRange)
    params_new = copy(params)
    betas = params[beta_indices]
    betas_sorted = sort(betas)
    params_new[beta_indices] = betas_sorted
    return params_new
end


"""
    save_checkpoint(filename, best_params, best_fitness, history, iteration)

Save optimization state to disk for crash recovery or resumption.

# Purpose
Long-running optimizations (hours or days) can be interrupted. This function
saves the current state so you can resume from the last checkpoint rather than
starting over. Essential for production runs on clusters.

# Arguments
- `filename::String`: Path to save checkpoint file (uses JLD2 format)
- `best_params::Vector{Float64}`: Current best parameter set found
- `best_fitness::Float64`: Objective value at best_params
- `history::Dict`: Full optimization history (all tracked metrics)
- `iteration::Int`: Current iteration number

# Creates
A .jld2 file containing all optimization state that can be loaded with
load_checkpoint() to resume optimization.

# Example
```julia
# In PSO loop, every 10 iterations:
if iter % 10 == 0
    save_checkpoint("pso_backup.jld2", g_best, g_best_fitness, history, iter)
end

# Later, if run crashes, resume with:
prev_best, prev_fitness, prev_hist, last_iter = load_checkpoint("pso_backup.jld2")
```

# Note
Requires JLD2.jl package. File size is typically small (< 1 MB) since it only
stores vectors and scalars, not the full swarm state.
"""
function save_checkpoint(filename, best_params, best_fitness, history, iteration)
    using JLD2
    @save filename best_params best_fitness history iteration
    println("  [Checkpoint saved at iteration $iteration]")
end


"""
    load_checkpoint(filename)

Load previously saved optimization state from disk.

# Purpose
Resume an interrupted optimization from a checkpoint file. Returns all saved
state so you can continue PSO from where it left off.

# Arguments
- `filename::String`: Path to checkpoint file (created by save_checkpoint)

# Returns
- `best_params::Vector{Float64}`: Best parameters found before interruption
- `best_fitness::Float64`: Objective value at best_params
- `history::Dict`: Full optimization history up to checkpoint
- `iteration::Int`: Last completed iteration number

# Example
```julia
# Load checkpoint
prev_best, prev_fitness, prev_hist, last_iter = load_checkpoint("pso_backup.jld2")

# Resume optimization from last_iter+1
# Use prev_best to initialize new swarm around previous best solution
```

# Note
Must have been saved with save_checkpoint(). Returns exactly what was saved.
"""
function load_checkpoint(filename)
    using JLD2
    @load filename best_params best_fitness history iteration
    return best_params, best_fitness, history, iteration
end


# ============================================================================
# USAGE EXAMPLE FOR YOUR SMM
# ============================================================================

"""
Setup at the start of your main script:
"""
# using Distributed
# addprocs(70)  # Add 70 worker processes
# 
# @everywhere using NPZ, Statistics, Random
# @everywhere include("tools.jl")  # Your SMM code
# @everywhere include("pso_parallel.jl")  # This file


"""
    optimize_smm_with_parallel_pso(initial_params_file, K, variable_list, n_particles, max_iter)

High-level wrapper to optimize SMM parameters using parallel PSO.
Integrates with your existing SMM training infrastructure.

# Purpose
Simplifies PSO usage for SMM estimation. Handles:
- Loading initial parameters from your .npy files
- Building bounds for specified variables
- Creating objective function that calls your train_stage_one
- Running parallel PSO
- Reconstructing full parameter vector from optimized subset

# Arguments
- `initial_params_file::String`: Path to .npy file with starting parameters
                                  (e.g., "./reporting_aero/0/best_params.npy")
- `K::Int=1`: Column index to use from params file (for ensemble of parameter sets)
- `variable_list::Vector{String}=["beta"]`: Which parameters to optimize
                                             Options: "beta", "T", "productivity", 
                                             "agg_labor_share_tech", "agg_industry_share_tech"
                                             Can optimize multiple: ["beta", "T"]
- `n_particles::Int=70`: Number of PSO particles (recommend = number of CPU cores)
- `max_iter::Int=50`: Number of PSO iterations to run

# Returns
- `final_params::Vector{Float64}`: Full parameter vector with optimized values
                                    (non-optimized params unchanged from initial)
- `best_fitness::Float64`: SMM objective function value at final_params
- `history::Dict`: PSO optimization history (best fitness, mean fitness, etc.)

# How It Works
1. Loads initial parameters from file
2. Extracts only the variables in variable_list to optimize
3. Sets bounds: lb = initial / (1+alpha), ub = initial * (1+alpha)
4. Creates objective function that:
   - Takes optimized variables
   - Reconstructs full parameter vector
   - Calls train_stage_one to evaluate SMM
5. Runs parallel PSO on objective function
6. Returns full parameter vector with optimized variables updated

# Example
```julia
# Optimize only beta parameters
best_params, fitness, hist = optimize_smm_with_parallel_pso(
    "./reporting_aero/epoch_1/1/best_params.npy",
    1,                    # Use first parameter set
    ["beta"],             # Optimize only betas
    70,                   # Use 70 particles (one per core)
    50                    # Run for 50 iterations
)

# Optimize beta and T together
best_params, fitness, hist = optimize_smm_with_parallel_pso(
    "./reporting_aero/epoch_1/1/best_params.npy",
    1,
    ["beta", "T"],        # Optimize both betas and T parameters
    70,
    100
)
```

# Notes
- Requires train_stage_one function from your tools.jl
- Automatically handles beta ordering constraint if "beta" in variable_list
- Alpha parameter (0.1) controls search radius: larger = wider search
- Initial params file should be from previous stage or initialization
"""
function optimize_smm_with_parallel_pso(
    initial_params_file::String,
    K::Int = 1,
    variable_list::Vector{String} = ["beta"],
    n_particles::Int = 70,
    max_iter::Int = 50
)
    # Load initial parameters
    best_params = NPZ.npzread(initial_params_file)[:,K]
    names = [:beta, :agg_labor_share_tech, :agg_industry_share_tech, :productivity, :T]
    vals = unpack_params(best_params)
    params_dict = Dict(names .=> vals)
    
    # Build bounds for variables to optimize
    alpha = 0.1
    lb_list = []
    ub_list = []
    var_indices = Dict()
    current_idx = 1
    
    for var in variable_list
        var_symbol = Symbol(var)
        var_values = params_dict[var_symbol]
        var_len = length(var_values)
        
        push!(lb_list, var_values ./ alpha)
        push!(ub_list, var_values .* alpha)
        
        var_indices[var] = current_idx:(current_idx + var_len - 1)
        current_idx += var_len
    end
    
    lb = vcat(lb_list...)
    ub = vcat(ub_list...)
    
    # Objective function (evaluated in parallel on workers)
    function smm_objective(x_stage)
        # Reconstruct full parameter vector
        x_full = copy(best_params)
        for var in variable_list
            var_symbol = Symbol(var)
            # Map stage variables back to full parameter positions
            param_start = get_param_start_index(names, var_symbol)
            var_len = length(params_dict[var_symbol])
            stage_idx = var_indices[var]
            x_full[param_start:param_start+var_len-1] = x_stage[stage_idx]
        end
        
        # Evaluate SMM
        _, results = train_stage_one(1, nothing, [x_full], false)
        return results[1][1][1]  # Extract loss
    end
    
    # Run parallel PSO
    println("Starting Parallel PSO optimization...")
    println("Variables: $(join(variable_list, ", "))")
    println("Particles: $n_particles (using $(nworkers()) workers)")
    println()
    
    beta_constraint = "beta" in variable_list
    beta_idx = beta_constraint ? var_indices["beta"] : 1:0
    
    best_params_opt, best_fitness, history = parallel_pso(
        smm_objective,
        lb, ub,
        n_particles = n_particles,
        max_iter = max_iter,
        beta_constraint = beta_constraint,
        beta_indices = beta_idx,
        verbose = true,
        checkpoint_every = 10,
        checkpoint_file = "smm_pso_checkpoint.jld2"
    )
    
    # Reconstruct full parameter vector
    final_params = copy(best_params)
    for var in variable_list
        var_symbol = Symbol(var)
        param_start = get_param_start_index(names, var_symbol)
        var_len = length(params_dict[var_symbol])
        stage_idx = var_indices[var]
        final_params[param_start:param_start+var_len-1] = best_params_opt[stage_idx]
    end
    
    return final_params, best_fitness, history
end


"""
    get_param_start_index(names, target)

Find the starting index of a specific parameter in the full parameter vector.

# Purpose
Your parameter vector is structured as: [beta(5), labor_share(1), industry_share(1), 
prod(N), T(S*R)]. This function calculates where each parameter block begins so
we can extract or update specific parameters.

# Arguments
- `names::Vector{Symbol}`: Ordered list of parameter names 
                           (e.g., [:beta, :agg_labor_share_tech, ...])
- `target::Symbol`: Which parameter's start index to find

# Returns
- `Int`: Index where target parameter begins in full vector

# Example
```julia
names = [:beta, :agg_labor_share_tech, :productivity, :T]
idx = get_param_start_index(names, :productivity)
# Returns 7 (after 5 betas + 1 labor_share + 1 industry_share)

# Use to extract productivity values:
prod_length = get_param_length(:productivity)
productivity = full_params[idx:idx+prod_length-1]
```

# Note
Must match the structure used by your unpack_params function. Throws error
if target parameter not found in names.
"""
function get_param_start_index(names, target)
    # Implement based on your parameter structure
    # Return starting index of target parameter in full vector
    idx = 1
    for name in names
        if name == target
            return idx
        end
        idx += get_param_length(name)
    end
    error("Parameter $target not found")
end

"""
    get_param_length(name)

Get the number of elements for a given parameter type.

# Purpose
Different parameters have different dimensions (betas: 5, labor share: 1, T: S×R).
This function returns how many elements each parameter occupies in the full vector.

# Arguments
- `name::Symbol`: Parameter name (e.g., :beta, :T, :productivity)

# Returns
- `Int`: Number of elements for this parameter

# Example
```julia
get_param_length(:beta)                    # Returns 5
get_param_length(:agg_labor_share_tech)    # Returns 1
get_param_length(:T)                       # Returns S*R (e.g., 160)
```

# Note
You must customize this function to match your actual data dimensions:
- S = number of sectors
- R = number of regions
- Productivity length depends on your network structure
Adjust the values in the Dict to match your model.
"""
function get_param_length(name)
    # Return length of each parameter type
    # Adjust to your actual dimensions
    if name == :beta
        return 5
    elseif name == :agg_labor_share_tech
        return 1
    # Add others...
    else
        return 1
    end
end