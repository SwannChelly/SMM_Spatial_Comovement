# `optimizers/` — optimizer backends

These files are the *interchangeable engines* that search the parameter space.
They are selected at run time with `--optimizer=pso` (default), `--optimizer=cmaes`,
or `--optimizer=tiktak`.

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
| `tiktak_integration.jl` | **TikTak** (Guvenen multistart; [serdarozkan/TikTak](https://github.com/serdarozkan/TikTak)). A global multistart method: (1) a **pre-test** evaluates a large scrambled-Sobol net and keeps the best `n_restarts` legitimate points, then (2) runs a **local search** from each — but started at a convex combination `(1−θᵢ)·sᵢ + θᵢ·z*` pulled toward the best optimum `z*` found so far, with `θᵢ = clamp((i/K)^{1/2}, 0.1, 0.995)` growing with the restart index, so late starts *refine* the incumbent. Local solver is **Nelder-Mead** (`Optim.jl`, no new dependency; Brent for the 1-D single-α sub-stage), pluggable for a DFNLS/BOBYQA engine. Like CMA-ES it runs on the normalized unit cube and floors the result at the warm start. Well-suited to the Sinkhorn-reduced (`profile_T`) parameter space. |

TikTak's parallelism: the pre-test is `pmap`ped across all workers; the local
searches run in batches of `nworkers()` (each search serial on one worker, blending
toward the incumbent as it stood at batch start — the parallel-TikTak trade-off). The
pre-test size is dimension-aware — `clamp(points_per_dim·d, K, sobol_multiplier·K)` —
so a low-dim block-coordinate sub-stage doesn't re-run a saturated high-dim pre-test.

Like PSO, TikTak runs `run_optimization`'s **staged block-coordinate refinement**
(productivity → α → T → technical, and the `gamma_beta_only` α-then-T alternation):
each sub-stage re-runs TikTak on one block, warm-started at the incumbent and
monotone-floored. Only CMA-ES collapses the joint path into a single Stage-1 run.

All three share the small helper `enforce_alpha_constraint`, which keeps the
trade-cost parameters α ordered (α₁ ≤ α₂ ≤ … ) after every move.
