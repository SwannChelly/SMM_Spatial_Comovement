# The model

This note describes the structural model that the code calibrates. It is meant for
an economist: it states the ideas and the equations, not the Julia. It corresponds
to Appendix B of *"Spatial Comovements"*.

---

## 1. What the model is for

We want to explain how a demand shock in one place propagates to other places
through **supply chains**. Downstream firms buy intermediate inputs from upstream
suppliers located in different regions. When two downstream buyers happen to source
from the same region-sector, a shock to one shows up in the other — this is the
*spatial comovement* the paper measures. The model gives us a structural mapping
from a handful of parameters to the observable sourcing and comovement patterns,
which we then match to the data by the Simulated Method of Moments (SMM), or in
closed form by GMM.

---

## 2. Environment

- **Regions** indexed by `r` (downstream buyer) and `r'` or `l` (upstream seller).
- **Upstream sectors** indexed by `s`.
- **Varieties** within a sector indexed by `ρ`.

Each downstream region `r` has a representative producer with productivity `A_r`.
Production is **nested CES**:

1. At the top, labor and a bundle of intermediates combine with elasticity `λ`.
2. The intermediate bundle combines sectors with elasticity `ν`.
3. Within a sector, varieties combine with elasticity `ν_s`.

The technology shares are the parameters `Ω^L` (labor) and `Ω^s` (sector `s`).

---

## 3. Supplier productivities and trade costs

Upstream productivities are **Fréchet**: variety `ρ` in sector `s`, region `r'`,
has productivity drawn from a Fréchet distribution with

- **shape** `θ` — governs dispersion (how different suppliers are); fixed/calibrated
  outside the estimation, and
- **scale** `T_{sr'}` — the average productivity ("comparative advantage") of sector
  `s` in region `r'`. This is one of the parameters we estimate.

Shipping is subject to **iceberg trade costs** `τ_{r'rs} ≥ 1`: delivering one unit
from `r'` to `r` requires producing `τ_{r'rs}` units. Trade costs rise with distance
`d_{r'r}`. In the power-law parametrization used for the headline runs,

```
τ_{r'r} = d_{r'r}^{α}
```

with a single elasticity `α` (this is the `N_TAU = 1` case). A more flexible version
replaces the single `α` by a step function over distance bins (`N_TAU = 4` or `5`),
one coefficient per bin.

---

## 4. Prices and sourcing

Because productivities are Fréchet and firms buy from the cheapest supplier
(Ricardian selection), the standard Eaton–Kortum algebra applies **within each
region's sourcing problem**. The probability that region `r` sources a given variety
of sector `s` from region `r'` — the **sourcing share** — is

```
                 T_{sr'} · (w_{r'} τ_{r'rs})^{-θ}
   γ_{r'rs}  =  ─────────────────────────────────────
                Σ_l  T_{sl} · (w_{l}  τ_{lrs})^{-θ}
```

where `w_{r'}` is the upstream wage. The denominator `Φ_{sr} = Σ_l T_{sl}(w_l τ_{lrs})^{-θ}`
is the sector-`s` "price access" of region `r`; the sector price index is
`P_{sr} ∝ Φ_{sr}^{-1/θ}`. These `γ_{r'rs}` — how much of region `r`'s sector-`s`
input spending goes to each origin `r'` — are the workhorse moment of the model.

Downstream, prices, quantities and sales follow from CES demand with elasticity `ε`
and a demand shifter `δ_r`:

```
   price       p_r  = c̃_r / μ,        μ = (ε−1)/ε   (inverse markup)
   unit cost   c̃_r  = c_r / A_r
   sales       Y_r  = p_r^{ε} · P^{−ε} · E · δ_r
   input spend = (1 − Ω^L) · μ · Y_r
```

Trade flows are then `X_{r'rs} = γ_{r'rs} ×` (region `r`'s spending on sector-`s`
inputs).

---

## 5. How the model is solved (two implementations)

The same economics is implemented two ways; they agree in the limit and are used for
different estimators.

- **`model_CP.jl` — simulation (SMM).** Draws a large number of Fréchet varieties,
  lets each downstream region pick its cheapest supplier variety by variety, and
  averages. The draws use variance-reduction (quasi-Monte-Carlo / Sobol) so the
  simulated moments are smooth enough for the optimizer. This is the engine behind
  `main.jl`.

- **`model_analytical.jl` — closed form (GMM).** With wages normalized, the sourcing
  map factorizes and the moments can be written in closed form; the distance
  regression coefficient is computed by Gauss–Legendre quadrature over the
  productivity distribution. No simulation noise, and standard errors are exact. This
  is the engine behind `main_gmm.jl`.

---

## 5b. Simulation draws — per-sector Sobol nets (non-standard, deliberate)

This section documents a choice in the SMM engine that is *not* the textbook default,
because it matters for correctness.

**The problem.** To simulate the model we need, for each variety `ρ`, a productivity
draw for every region `r'` that could supply it. Those draws should be **independent
across regions** — that independence is exactly what makes the Ricardian "pick the
cheapest region" a genuine random selection. If we draw them by plain Monte Carlo
(`--draws=mc`), independence holds but the moments are noisy. If instead we draw a
single high-dimensional **quasi-Monte-Carlo (Sobol) net** over all regions at once,
we get low variance — but Sobol coordinates are *jointly* low-discrepancy, i.e. they
are **correlated by construction across dimensions**. Feeding correlated "productivity"
coordinates into a min-over-regions operator distorts *which region wins*, biasing the
sourcing shares `γ_{r'rs}`. So the naive fix (a global Sobol net) trades noise for bias.

**The choice (`--draws=sobol`, the default).** We keep the variance reduction of Sobol
but restore cross-region independence:

- **One low-dimensional Sobol net per sector.** For each sector `s` we build a Sobol
  net whose dimension is only the number of active regions in that sector — because the
  Ricardian coupling (`min` over suppliers) lives *within a sector, across its regions*,
  not across sectors. Keeping the net low-dimensional is where equidistribution actually
  pays off; different sectors get independent nets.
- **A digital shift (scramble) that decorrelates the coordinates.** Each net is
  randomized by a base-2 *digital shift* (an XOR of the fixed-point bits with a
  per-dimension random mask drawn from an explicit RNG). This preserves the Sobol net's
  equidistribution, dissolves the degenerate all-zeros Sobol origin, and — crucially —
  drives the pairwise correlation between region-coordinates down to ≈ 1/√N, so within a
  sector the draws **behave like independent draws** across regions. That is the sense in
  which the scheme "mimics independence across regions."
- **Flat weights.** Every draw carries weight `1/N`, so the simulator's winner-weight
  computation is exact and the estimator stays **unbiased** for the min-coupled moments
  (the same guarantee as plain `--draws=qmc`), while achieving lower simulation variance
  on thick, multi-region sectors.

The scramble is seeded explicitly: a **frozen seed** during optimization (so the loss is
deterministic and the search reproducible) and an **independent seed per replication**
when estimating simulation variance (Step 2 / the Jacobian). Alternative draw methods
remain available — `qmc` (stratified uniform), `mc` (plain Monte Carlo), and `is`
(importance sampling, biased for the min-coupled moments and kept only for the distance
regression's tail) — but `sobol` is the default for the reasons above.

---

## 6. The moments we match

Estimation matches five blocks of moments (empirical vs. model):

| Block | Symbol | What it is |
|-------|--------|-----------|
| 1 | aggregate labor share | economy-wide labor share |
| 2 | industry shares `π_s` | share of each upstream sector |
| 3 | regional shares `π_r` | share of activity by region |
| 4 | `reg_coef` | coefficients of a regression of supplier probability on distance (the **extensive margin** of sourcing) |
| 5 | `γ_ls` | the sourcing shares of Section 4 — who buys from whom |

Block 4 deserves a word. Empirically it is the regression of "does region `r`
source anything from region `r'`" on distance. In the analytical model the
regressand is the whole-industry extensive margin
`ρ̃_{r's}(z) = 1 − ∏_dr (1 − ρ_{r'drs}(z))` (probability of winning *somewhere*),
integrated over the productivity range. Matching it identifies the trade-cost
elasticity `α`.

---

## 7. Parameters

**Estimated:**

| Symbol | Name in code | Meaning |
|--------|--------------|---------|
| `Ω^L` | `agg_labor_share_tech` | labor share in production |
| `Ω^s` | `agg_industry_share_tech` | sectoral input shares |
| `A_r` | `productivity` | downstream productivity by region |
| `α` | `alpha` | trade-cost elasticity (`N_TAU` of them: one power-law coefficient, or one per distance bin) |
| `T_{sr}` | `T` | Fréchet scale / comparative advantage by sector-region |

The only trade-cost parameter is `α`. The distance regression coefficients of block 4
(`reg_coef`) — sometimes written `β` in the extensive-margin regression of Section 6 —
are **moments the model matches, not parameters.** The moment count `N_REG` and the
parameter count `N_TAU` are independent.

The full parameter vector is laid out as `[Ω^L | Ω^s | A | α | T]`. Only the T
entries with a non-zero empirical sourcing share are free (`T_MASK`), and within each
sector T is identified only up to a scale, so one reference region per sector is
normalized. See the "T flat-indexing convention" section of `../CLAUDE.md` for the
exact indexing.

**Fixed (calibrated outside the loop):** demand elasticity `ε`, the substitution
elasticities `λ`, `ν`, `ν_s`, and the Fréchet shape `θ`.

---

## 8. From parameters to estimates

The mapping "parameters → moments" of Sections 4–6 is the object the optimizer
inverts. Given a candidate parameter vector, the model produces simulated moments;
the estimator forms the weighted distance to the empirical moments,

```
   loss(θ) = (m_sim(θ) − m_emp)' · W · (m_sim(θ) − m_emp),
```

and searches for the θ that minimizes it. The search itself is described in
`optimizer.md`; the three-step efficient weighting scheme, the standard errors, and the
Hansen J-test are in `inference.md`. The README ties both to how you run the code.
