# The optimizer

This note explains **how the parameters are searched for**. The estimator minimizes
a weighted distance between model and data moments,

```
   loss(θ) = (m_sim(θ) − m_emp)' · W · (m_sim(θ) − m_emp).
```

`loss(θ)` is not smooth or convex — it comes from a simulated model with Ricardian
(min-cost) selection, so it has flat regions, kinks and (in SMM) a little simulation
noise. Gradient methods are unreliable here, so we use **derivative-free population
search**. Two engines are available and interchangeable:

- **PSO** (Particle Swarm Optimization) — the default.
- **CMA-ES** (Covariance Matrix Adaptation Evolution Strategy).

Select one with `--optimizer=pso` or `--optimizer=cmaes` in `run.sh`.

---

## 1. How the two engines plug in

Everything that is *independent* of which engine runs lives in **`optimizer.jl`**:
building the search box, warm-starting from the previous stage, the staged
refinement schedule, and all the file/report bookkeeping. It calls the engine
through a single seam, `optimize_stage`, and every engine honours the same contract:

```
(objective, lower_bounds, upper_bounds; warm_start, n_particles, max_iter, …)
        →  (best_x, best_loss, history)
```

So `main.jl` never names a specific optimizer. The two concrete engines are
`optimizers/pso_integration.jl` and `optimizers/cmaes_integration.jl`.

A few conventions shared by both:

- **Warm start / monotonicity.** Each stage starts from the previous stage's best
  point and both engines guarantee the returned loss is *never worse* than that
  starting point (PSO by injecting it as a particle; CMA-ES by tracking an
  incumbent). This makes the staged pipeline safe to chain.
- **T searched in log space.** The Fréchet scales `T` are strictly positive and
  enter multiplicatively, so they are searched as `φ = log T`. This keeps steps
  scale-invariant and the box symmetric.
- **α ordering.** The trade-cost coefficients are kept ordered (`α₁ ≤ α₂ ≤ …`) after
  every move by `enforce_alpha_constraint`.
- **Init-anchored box.** In every stage α and T are confined to `[×0.5, ×2]` of their
  theory-based starting values (the α prior and the γ-inversion for T). The per-stage
  search radius anneals *inside* this box.

---

## 2. PSO — Particle Swarm Optimization

### Mechanism

A **swarm** of candidate parameter vectors ("particles") flies through the search
box. Each particle `i` remembers its own best point `p_best_i`, and the swarm shares
the global best `g_best`. Every iteration each particle updates its velocity as a
blend of three pulls,

```
   v_i ← w · v_i  +  c₁ · r₁ · (p_best_i − x_i)  +  c₂ · r₂ · (g_best − x_i)
   x_i ← x_i + v_i
```

- `w` (**inertia**) — keep going in the current direction; decays from `w_start` to
  `w_end` over the run, shifting the swarm from exploration to exploitation.
- `c₁` (**cognitive**) — pull back toward the particle's own best.
- `c₂` (**social**) — pull toward the swarm's best.
- `r₁, r₂` — fresh uniform random numbers, the source of stochastic search.

Velocities are clamped, positions are clamped to the box, and **stagnant particles
are restarted** periodically (any particle whose loss is far above `g_best` is thrown
to a new random point) to escape local minima. All particles in a generation are
evaluated **in parallel** across CPU workers.

### Main parameters (`parallel_pso_smm`)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `n_particles` | ~100 | swarm size (set to roughly the number of cores) |
| `max_iter` | 200 / 50 | iterations (initial fit / refinement stages) |
| `w_start`, `w_end` | 0.9, 0.4 | inertia decay (exploration → exploitation) |
| `c1` | 1.5 | cognitive pull (own best) |
| `c2` | 2.5 | social pull (global best) |
| `warm_start_particle` | previous best | guarantees monotone improvement |

### How it is used in the code

`optimizer.jl` runs PSO in **stages** (`run_optimization`):

1. **Stage 0** — a coarse search over the trade-cost `α` alone to find a good
   starting value that matches the distance regression.
2. **Stage 1** — one joint PSO fit over *all* parameters.
3. **Refinement loops** — repeated **block-coordinate** passes: each loop refines one
   group of parameters at a time (productivity `A`, then `α`, then `T`, then the
   technical shares), with the search radius annealing from wide to narrow. This
   staged decoupling is what PSO is good at here.

---

## 3. CMA-ES — Covariance Matrix Adaptation Evolution Strategy

### Mechanism

CMA-ES samples each generation's population from a **multivariate Gaussian** and then
*adapts that Gaussian* using the best points it saw:

```
   x_i  ~  N(mean, σ² · C),     i = 1 … λ
```

- The **mean** moves toward the weighted average of the best individuals.
- The **step size** `σ` grows or shrinks depending on whether recent steps were
  consistently in the same direction.
- The **covariance matrix** `C` learns the *shape* of the good region — it stretches
  along directions where the loss falls slowly and shrinks along steep ones, so the
  search automatically rescales itself. This is the key advantage over PSO: it learns
  cross-parameter correlations instead of moving each coordinate independently.

Because a single scalar `σ` has to be meaningful for every coordinate, the code runs
CMA-ES on a **normalized unit cube** `[0,1]^n` and maps back to `[lb, ub]` inside the
evaluator. It tracks the best-ever point (incumbent) so the returned solution is
never worse than the warm start, and evaluates each generation's population in
parallel.

### Main parameters (`parallel_cmaes_smm`)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `n_particles` (λ) | `4 + ⌊3 ln n⌋` | population size per generation |
| `max_iter` | 300 | generations (`maxfevals = λ · max_iter`) |
| `sigma0` | 0.2 | initial step size on the unit cube |
| `ftol`, `xtol` | 1e-11, 1e-12 | convergence tolerances (it stops on its own) |
| `x0` | previous best | warm start / incumbent floor |
| `seed` | 1 | reproducibility |

Wraps the `CMAEvolutionStrategy.jl` library.

### How it is used in the code

CMA-ES enters through the *same* `optimize_stage` seam. Because it already learns
cross-block covariance and stops on its own tolerances, the block-coordinate
refinement loop is **collapsed to the single joint Stage-1 run** for the general
case — CMA-ES does internally what PSO does through staging. (The `gamma_beta_only`
Step-3 case still runs the α-then-T alternation under both engines.) CMA-ES is
available for the SMM path.

---

## 4. Which to use

- **PSO** is the default and the most-tested path; its staged refinement is well
  suited to the flat/kinked SMM loss and it parallelizes trivially.
- **CMA-ES** is a good second opinion and can be more sample-efficient once near the
  optimum, thanks to covariance adaptation. Use it to confirm a PSO solution or when
  the loss surface is strongly correlated across parameters.

Both produce the same downstream objects (best parameters, reports), so the choice
does not affect the inference step that follows.
