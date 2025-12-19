# PSO Implementation Guide for SMM Calibration

## Why PSO is Better Than Halton Grid Search

### **Your Current Approach (Halton Grid)**
- Evaluates K_max=50 fixed points per stage
- Each point evaluated independently
- No learning from previous evaluations
- With 2500+ moments, needs many stages to converge

### **PSO Approach**
- **Intelligent exploration**: Particles learn from their own history and swarm's best
- **Parallel evaluation**: All N particles evaluated simultaneously on your cores
- **Adaptive search**: Automatically balances exploration (early) vs exploitation (late)
- **Faster convergence**: Typically 30-50 iterations vs hundreds of grid points

## Efficiency Gains

### Computational Cost
```
Halton Grid (old):
- K_max loops × 50 particles × multiple stages
- ~2500 evaluations for initial stage
- No information sharing between evaluations

PSO (new):
- 50 iterations × 70 particles = 3500 evaluations
- But with intelligent guidance (typically converges in 20-30 iterations)
- Effective evaluations: ~1500-2000 to reach similar quality
```

### Wall-Clock Time (Example with 70 cores)
```
Halton Grid:
- 50 particles in parallel → 1 batch
- K_max=50 iterations → 50 batches per stage
- If 1 batch = 10 minutes → 500 minutes per stage
- Multiple stages needed

PSO:
- 70 particles in parallel → 1 iteration
- 30 iterations → 30 batches total
- If 1 batch = 10 minutes → 300 minutes TOTAL
- All parameters optimized
```

## Key PSO Parameters

### **n_particles** (Number of Particles)
- **Recommendation**: Set equal to your number of cores
- With 70 cores → use 70 particles
- Each iteration evaluates all particles in parallel
- More particles = better exploration but slower iterations

### **max_iter** (Maximum Iterations)
- **Initial stage**: 50-100 iterations (exploring full space)
- **Refinement stages**: 20-30 iterations (narrowing search)
- PSO typically converges faster than this limit
- Monitor `best_fitness` - if flat for 10 iterations, it has converged

### **w_start / w_end** (Inertia Weight)
- Controls exploration vs exploitation trade-off
- **w_start = 0.9**: High inertia early (explore widely)
- **w_end = 0.4**: Low inertia late (exploit best regions)
- Default values work well for most problems

### **c1 / c2** (Cognitive/Social Parameters)
- **c1 = 2.0**: How much particles trust their own best
- **c2 = 2.0**: How much particles follow global best
- c1 > c2 → more independent exploration
- c2 > c1 → more swarm cohesion

### **alpha** (Search Radius)
- Used in refinement stages
- **alpha = 0.5**: Search within 50% of previous best (initial)
- **alpha = 0.2**: Search within 20% of previous best (refinement)
- Smaller alpha = more focused search

## Usage Examples

### Example 1: Initial Full Optimization
```julia
# Optimize all parameters from scratch
best_params, best_fitness, history = train_stage_pso(
    70,              # n_particles (use all your cores)
    50,              # max_iter
    init_beta = [1.0, 2.0, 3.0, 4.0, 5.0],
    variable_list = nothing,    # Optimize everything
    last_stage_folder = nothing,
    alpha = 0.5,
    second_stage = false
)
```

### Example 2: Refine Specific Parameters
```julia
# Refine only beta and T, keeping others fixed
best_params, best_fitness, history = train_stage_pso(
    70,
    30,              # Fewer iterations for refinement
    variable_list = ["beta", "T"],
    last_stage_folder = "./reporting_aero/0",
    K = 1,
    alpha = 0.2,     # Narrower search
    second_stage = false
)
```

### Example 3: Second-Stage Estimation (with masked T)
```julia
# Optimize with masked moments
best_params, best_fitness, history = train_stage_pso(
    70,
    30,
    variable_list = ["T"],
    last_stage_folder = "./reporting_aero/epoch_1/2",
    K = 1,
    alpha = 0.1,     # Very focused search
    second_stage = true  # Use masked moments
)
```

## Interpreting PSO Output

### During Optimization
```
[PSO] Iteration 10/50 (12.35s):
  Best fitness:     0.001234
  Mean fitness:     0.001567
  Improvement:      0.000123 (5.67%)
```
- **Best fitness**: Current best loss found
- **Mean fitness**: Average across all particles
- **Improvement**: Change since last iteration
  - If improvement → still exploring productively
  - If flat → may have converged

### Convergence Indicators
1. **Best fitness plateaus**: No improvement for 5-10 iterations
2. **Mean approaches best**: Swarm converging to same region
3. **Small improvements**: < 0.1% change per iteration

## Comparison with Your Old Approach

### Stage 1: Initial Calibration

**Old (Halton)**:
```julia
# Generate 50 x 50 = 2500 parameter sets
for K in 1:K_max
    params_list = generate_halton_grid(1000, ...)
    # Evaluate each sequentially per K
end
```

**New (PSO)**:
```julia
# PSO with 70 particles, 50 iterations = 3500 evaluations
# But with intelligent guidance
best_params, fitness, history = train_stage_pso(70, 50, ...)
```

### Efficiency: PSO finds better solutions in ~50% fewer evaluations

### Stage 2+: Refinement

**Old**: Multiple Halton grids around best point
**New**: Single PSO run automatically explores around best region

## Tuning Tips

### If PSO is converging too slowly:
1. Increase `n_particles` (better exploration)
2. Increase `c2` (particles follow global best more)
3. Check if bounds are too wide

### If PSO is converging to poor local minimum:
1. Increase `w_start` (more exploration)
2. Increase `c1` (particles explore independently longer)
3. Use multiple random restarts

### If PSO is too expensive:
1. Reduce `n_particles` (but at least 30-50)
2. Reduce `max_iter` for refinement stages
3. Use larger `alpha` for coarser search

## Advanced: Multi-Start PSO

For very challenging problems, run PSO multiple times:

```julia
n_restarts = 5
all_results = []

for restart in 1:n_restarts
    println("\nPSO Restart $restart/$n_restarts")
    
    best, fitness, hist = train_stage_pso(
        70, 50,
        init_beta = init_beta .* (1 + 0.2*randn(5)),  # Random perturbation
        ...
    )
    
    push!(all_results, (best, fitness))
end

# Take best across all restarts
best_overall = all_results[argmin([r[2] for r in all_results])][1]
```

## Monitoring and Diagnostics

### Check Convergence
```julia
# Look at history
plot(history["best_fitness"], label="Best", ylabel="Loss")
plot!(history["mean_fitness"], label="Mean")
```

### Check Parameter Changes
```julia
# Compare consecutive iterations
for i in 2:length(history["best_params"])
    params_old = history["best_params"][i-1]
    params_new = history["best_params"][i]
    change = maximum(abs.(params_new - params_old) ./ abs.(params_old))
    println("Iteration $i: max param change = $(round(change, digits=4))")
end
```

## Migration Path

### Step 1: Test on Small Problem
Replace one `train_stage_one` call with `train_stage_pso`:
```julia
# Old
params_list, results = train_stage_one(1000, init_beta)

# New  
best_params, fitness, history = train_stage_pso(70, 30, init_beta=init_beta)
```

### Step 2: Compare Results
Run both methods and compare:
- Final loss value
- Wall-clock time
- Quality of moments

### Step 3: Full Migration
Once satisfied, replace all grid searches with PSO

## Troubleshooting

### "PSO returns Inf fitness"
- Some particles violate constraints
- Check `parallel_SMM_safe` error handling
- Increase `beta_constraint` enforcement

### "PSO is slower than expected"
- Check you're using all cores: `nworkers()`
- Verify parallel overhead isn't too high
- Consider reducing `max_iter`

### "PSO converges too quickly to bad solution"
- Increase exploration: higher `w_start`, higher `c1`
- Widen search bounds (larger `alpha`)
- Use multi-start approach