# PSO Quick Start Checklist

## Implementation Checklist

### ✅ Files to Add
- [ ] `pso_integration.jl` - Core PSO functions (provided)
- [ ] `main_pso.jl` - Modified main file using PSO (provided)

### ✅ Code Changes

1. **Add PSO include**
```julia
@everywhere include("pso_integration.jl")
```

2. **Replace Halton grid calls**
```julia
# OLD
params_list = generate_halton_grid(n, 2000, ...)
params_list, results = train_stage_one(n, init_beta, params_list)

# NEW
best_params, fitness, history = train_stage_pso(
    n_particles=70,
    max_iter=50,
    init_beta=init_beta
)
```

## Expected Performance

### Your Current Setup
- ~2500+ moments to match
- K_max = 50 parameter sets per stage
- Multiple refinement stages

### Estimated Speedup

| Metric | Old (Halton) | New (PSO) | Improvement |
|--------|--------------|-----------|-------------|
| Evaluations to convergence | ~5000-10000 | ~2000-3000 | **2-3x fewer** |
| Parallelization efficiency | Medium (50 cores) | High (70 cores) | **40% more cores used** |
| Time per stage | ~8-10 hours | ~3-5 hours | **50-60% faster** |
| Quality at convergence | Good | Better | **Better exploration** |

### Real Example (estimated for your problem)

```
Problem size: 2500 moments, ~200 parameters

OLD APPROACH (Halton Grid):
├─ Stage 0: K_max iterations × 50 evals = 2,500 evals (8 hours)
├─ Loop 1: 3 stages × 1,000 evals each = 3,000 evals (9 hours)
├─ Loop 2-5: Similar = ~36 hours
└─ TOTAL: ~50-60 hours

NEW APPROACH (PSO):
├─ Stage 0: 50 iter × 70 particles = 3,500 evals (5 hours)
├─ Loop 1-3: 30 iter × 70 × 2 stages = 4,200 evals (6 hours)
└─ TOTAL: ~15-20 hours (converges in 3 loops vs 5+)

SAVINGS: 30-40 hours (~60% reduction)
```

## Configuration Recommendations

### For Your Problem Size (2500 moments)

```julia
# Initial broad search
STAGE_0_PARTICLES = 70      # Use all cores
STAGE_0_ITERATIONS = 50     # Thorough exploration

# Refinement searches
REFINE_PARTICLES = 70       # Still use all cores
REFINE_ITERATIONS = 30      # Faster convergence in focused region
REFINE_ALPHA = 0.2          # Search within ±20% of best
```

### Convergence Criteria
```julia
# Stop refinement loop when:
max_param_change < 0.01     # Parameters changing < 1%
# OR
improvement < 0.001         # Fitness improving < 0.1%
# OR
loop > 10                   # Maximum 10 refinement loops
```

## Testing Protocol

### Phase 1: Validation (2-3 hours)
```julia
# Run both methods on SMALL problem
# - Reduce S and R for faster evaluation
# - Compare final loss values
# - Verify PSO finds similar/better solution

@time begin
    # Old method
    params_old = train_stage_one(100, init_beta)
end

@time begin
    # New method
    params_new, _, _ = train_stage_pso(20, 10, init_beta=init_beta)
end

println("Old loss: ", full_SMM(params_old)[1])
println("New loss: ", full_SMM(params_new)[1])
```

### Phase 2: Single Stage (4-6 hours)
```julia
# Run PSO on one full stage
best_params, fitness, history = train_stage_pso(
    70, 50,
    init_beta = init_beta
)

# Check convergence plot
plot(history["best_fitness"])
savefig("pso_test_stage.png")

# Compare with your best Halton result
halton_best_loss = ...  # Your previous best
pso_loss = fitness
println("Improvement: ", halton_best_loss - pso_loss)
```

### Phase 3: Full Run (15-20 hours)
```julia
# Run complete multi-stage optimization
# Use main_pso.jl
```

## Monitoring During Run

### Terminal Output to Watch
```
[PSO] Iteration 10/50:
  Best fitness:     0.001234  ← Should decrease steadily
  Mean fitness:     0.001567  ← Should approach best fitness
  Improvement:      0.000123  ← Should be positive
```

### Red Flags
- ⚠️ Best fitness = Inf → Particles violating constraints
- ⚠️ Improvement = 0 for 10+ iterations → May have converged (or stuck)
- ⚠️ Mean >> Best for many iterations → Poor convergence
- ⚠️ Evaluation time increasing → Memory leak?

### Good Signs
- ✅ Steady decrease in best fitness
- ✅ Mean approaching best (swarm converging)
- ✅ Positive improvements each iteration
- ✅ Parameter changes becoming smaller

## Post-Run Analysis

### Compare Solutions
```julia
# Load old best
old_params = NPZ.npzread("./baseline_aero/old_best.npy")

# Load PSO best
new_params = NPZ.npzread("./reporting_aero/0/best_params.npy")

# Compare moments
old_moments = full_SMM(old_params)
new_moments = full_SMM(new_params)

# Which moments improved?
diff = abs.(old_moments - empirical_moments) - abs.(new_moments - empirical_moments)
improved = sum(diff .> 0)
worse = sum(diff .< 0)

println("Improved moments: $improved")
println("Worse moments: $worse")
```

### Visualize Convergence
```julia
# Plot PSO convergence
history = NPZ.npzread("./reporting_aero/0/pso_history.npy")
plot(history["best_fitness"], label="PSO Best", lw=2)
plot!(history["mean_fitness"], label="PSO Mean", lw=2, style=:dash)
xlabel!("Iteration")
ylabel!("Loss")
title!("PSO Convergence")
savefig("pso_convergence.png")
```

## Troubleshooting Guide

### Issue: PSO slower than expected
**Check:**
```julia
println("Workers: ", nworkers())  # Should be ~70
println("Particles: ", N_PARTICLES)  # Should equal workers
```
**Fix:** Ensure particles = workers for full parallelization

### Issue: Poor final solution
**Check:** Are you trapped in local minimum?
**Fix:** 
- Increase `w_start` to 0.95 (more exploration)
- Run multi-start PSO (5 random initializations)
- Widen bounds (increase `alpha`)

### Issue: Particles violating constraints
**Check:** Beta ordering constraint
**Fix:** Verify `enforce_beta_constraint` is being called

### Issue: Evaluation errors
**Check:** `parallel_SMM_safe` returning `nothing`
**Fix:** Enable error printing:
```julia
result = parallel_SMM_safe(params, false, false, true)  # Last arg = show errors
```

## Next Steps After Validation

Once PSO is working well:

1. **Fine-tune parameters**
   - Adjust `max_iter` based on convergence speed
   - Tune `alpha` for each refinement stage
   - Experiment with `w_start/w_end`

2. **Add early stopping**
   ```julia
   if iteration > 10 && improvement < 0.0001
       println("Early stop: converged")
       break
   end
   ```

3. **Save intermediate results**
   ```julia
   if iter % 10 == 0
       NPZ.npzwrite("checkpoint_iter_$iter.npy", g_best)
   end
   ```

4. **Add adaptive population**
   - Start with 100 particles for exploration
   - Reduce to 50 particles after iteration 20
   - Further reduce to 30 for final refinement

## Support & Debugging

If you encounter issues:

1. **Start small**: Test on reduced problem (fewer regions/sectors)
2. **Check one stage**: Don't run full multi-stage until one stage works
3. **Monitor resources**: Check CPU usage is ~100% during evaluation
4. **Save everything**: Keep all PSO histories for debugging

## Success Criteria

Your PSO implementation is successful when:

- ✅ Converges to loss < your current Halton best
- ✅ Takes < 20 hours for full multi-stage optimization
- ✅ Uses all available cores efficiently (check `top` or `htop`)
- ✅ Produces better moment matches than grid search
- ✅ Converges smoothly without getting stuck