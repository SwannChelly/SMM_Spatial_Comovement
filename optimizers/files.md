# `optimizers/` — optimizer backends

These two files are the *interchangeable engines* that search the parameter space.
They are selected at run time with `--optimizer=pso` (default) or `--optimizer=cmaes`.

Both obey **one contract** so the rest of the code never needs to know which one is
running:

```
(objective, lower_bounds, upper_bounds; warm_start, …)  →  (best_x, best_loss, history)
```

The backend-neutral orchestration that *calls* these engines (staging, bounds
construction, refinement loops) lives one level up in `../optimizer.jl`. For the
economics of what each optimizer does, see `../documentation/optimizer.md`.

| File | Role |
|------|------|
| `pso_integration.jl` | **Particle Swarm Optimization** (default). A population of candidate parameter vectors ("particles") that move through the search space, pulled toward each particle's own best point and the swarm's global best. Includes the *warm-start particle* trick that guarantees the loss never worsens from one stage to the next. |
| `cmaes_integration.jl` | **CMA-ES** (Covariance Matrix Adaptation Evolution Strategy). An adaptive Gaussian search that learns the shape (covariance) of the good region and rescales its steps accordingly. Wraps the `CMAEvolutionStrategy` library, runs on a normalized unit cube, and tracks an incumbent so it too never returns a point worse than its warm start. |

Both share the small helper `enforce_alpha_constraint`, which keeps the trade-cost
parameters α ordered (α₁ ≤ α₂ ≤ … ) after every move.
