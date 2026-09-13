# Firm counts inside comparative advantage: `T_ls = T_{a(l)s} · N_sl^η`

**Status: plan only. No code changed, no regression run** (this environment has no
`baseline_*` tree, so every empirical statement below is a prediction to be checked,
not a measurement).

---

## 0. What this is

Comparative advantage is currently a property of the attraction area alone
(`eq:T-area`): `T_{r's} = T_{a(r')s}` for every `r'` in the area. Every commuting zone
of an area is given the *same* Fréchet scale, so the model must explain all
within-area dispersion of sourcing — which zone of Toulouse's area actually supplies —
by distance and chance alone.

The proposal adds the observed firm count of the cell:

```
T_ls = T_{a(l)s} · N_sl^η
```

with `N_sl` the number of firms in sector `s`, commuting zone `l`, taken from the data
and treated as exogenous; `η` a single scalar for the whole industry.

Three claims organise the rest of this document, in decreasing order of confidence:

1. **The specification is not ad hoc.** It is the exact max-stable form of a cell
   whose `N_sl` establishments draw *correlated* Fréchet productivities, with `η`
   indexing that correlation. `η = 1` (independence) is a sharp null; `η ∈ (0,1]` is
   the predicted range.
2. **It is a one-line change to the model.** The cell's draw stays Fréchet with the
   same shape `θ`, so every closed form in the paper survives verbatim. The entire
   Julia edit is a multiplicative factor at the point where `T` is gathered onto
   commuting zones.
3. **It is not free.** `η` adds a parameter without adding a moment, so the current
   estimator cannot identify it — every block is already exactly absorbed by its own
   parameter. `η` must be calibrated outside (the user's route) or given its own
   moment. And the regression that calibrates it needs **area × sector fixed
   effects**, without which the estimate is not the model's `η`.

---

## 1. The theory

### 1.1 Where `N^η` comes from, and why `η ≤ 1`

Let cell `(l,s)` host `N_sl` establishments. For variety `ρ`, each establishment
`j` draws a productivity, and only the cheapest one in the cell can be the cell's
offer. If the draws were independent Fréchet(`T_a`, `θ`), the cell's offer is the
maximum of `N_sl` independent draws, and the maximum of `N` iid Fréchet(`T`,`θ`) is
Fréchet(`N·T`, `θ`) exactly. That is `η = 1`.

Independence across establishments of the same commuting zone is implausible — they
share a labour pool, infrastructure and often a technology. The standard extreme-value
device for exchangeable dependence is the logistic (Gumbel-copula) multivariate Fréchet
with dependence parameter `η ∈ (0,1]`:

```
Pr(z_1 ≤ x, …, z_N ≤ x) = exp( −[ N · (T x^{−θ})^{1/η} ]^{η} ) = exp( −N^η T x^{−θ} )
```

so the cell's best draw is **Fréchet(`N^η T_a`, `θ`)** — exactly the proposed
specification, with no approximation. The interpretation is clean:

| `η` | within-cell dependence | reading |
|---|---|---|
| `1` | independent | each firm is a genuinely separate shot at the market |
| `(0,1)` | positively dependent | `N^η` is the *effective* number of independent tries |
| `→ 0` | perfectly dependent | firm counts carry no productivity information; the current model |
| `> 1` | — | outside the max-stable family: agglomeration externalities, or reverse causality |

So the baseline model is the `η = 0` corner, and `η > 1` is a rejection of the
micro-foundation rather than a large effect. This gives the estimate a prior to be
read against, which a free elasticity would not have.

**Note the shape `θ` is untouched.** The cell's draw is still Fréchet with the same
dispersion; only its scale moves. This is why nothing else in the model changes.

### 1.2 What the change leaves invariant

Because only the scale moves, every object the paper derives is unchanged in form:

- sourcing shares `γ_{r'rs} = T_{r's}(w τ)^{−θ} / Φ_{rs}` (`eq:gamma_rrs`) — with
  `T_{r's}` now carrying the firm count;
- the price index, the CES nests, the GE solve;
- the extensive-margin union `p_{r's}` and the binomial count `K_{r's} ~ Bin(N_s, p_{r's})`;
- the Sinkhorn inversion of `T` from `γ` — it becomes a matrix-scaling problem with
  the *reweighted* kernel `K'_{a,r} = Σ_{l∈a} N_sl^η τ_{lr}^{−θ}`, which is still
  strictly positive, so existence, uniqueness and the contraction all carry over.

That last point matters operationally: the whole profiling apparatus (`invert_T_ge`,
the warm start, the T delta method) needs no rethinking, only the factor.

### 1.3 `η` is identified from within-area variation, and nothing else

`T_{a s}` is a free parameter for every active (sector, area) pair, and it is profiled
out so that the area-level sourcing shares `γ_{as}` are matched **exactly**. Therefore
any part of `N_sl^η` that is constant within an area is absorbed by `T_{as}` and cannot
identify `η`. Only the within-area, within-sector dispersion of `log N_sl` carries
information.

Two consequences, and the first is the single most common way this exercise goes wrong:

- **The calibrating regression must carry area × sector fixed effects.** A pooled
  regression of `γ_sl` on `log N_sl` estimates a mixture of the within-area elasticity
  (which is `η`) and the between-area correlation of firm density with comparative
  advantage (which is absorbed by `T_as` and is *not* `η`). Since dense areas are
  presumably also technologically favoured, a pooled regression overstates `η`,
  probably a lot.
- The fixed effect in the regression and the parameter profiled out by Sinkhorn are
  **the same object**. This is a genuine correspondence, not an analogy: the PPML
  fixed effect removes exactly the degrees of freedom the inversion consumes, so the
  regression uses precisely the variation the estimator leaves free.

### 1.4 Where `η` enters the moments — and where it does not

| Block | Moment | Parameter absorbing it | Does `η` enter? |
|---|---|---|---|
| 1–3 | labour share, sector shares, `π_r` | `Ω_L`, `Ω_s`, `A_r` | no (cost-share / demand objects) |
| 4 | `reg_coef` (firm-level cloglog) | `α` | **no, to first order** — see below |
| 5 | `γ_as` (area aggregates) | `T_as` (Sinkhorn, exact) | yes, but exactly absorbed |
| 6 | `Ḡ_s(0)` | `N_s` (bisection, exact) | yes, but exactly absorbed |

**Block 4 is insulated, and this is the important structural result.** The firm-level
cloglog conditions on the firm's own productivity `z`. Conditional on `z`, the
probability of winning is `exp(−Φ_{rs}(wτ_{lr})^{θ} z^{−θ})`: the cell's own `T` does
*not* appear, because the competition `Φ_{rs}` is a buyer-level index that the
area × sector fixed effect absorbs. So `α` is read off the same clean within-area
distance gradient as before.

This also *repairs* the paper's identification paragraph rather than breaking it. The
paper currently argues that `α` is clean because "comparative advantage is constant
within the area". That is no longer true. But it was never the real reason: `α` is
clean because the size control makes `T` drop out of the win probability. Under the
new specification the size control becomes load-bearing rather than merely useful, and
the identification paragraph should say so.

**Consequence for the moment accounting.** With every block already exactly absorbed,
`η` is a free parameter with no moment left to match. Inside the current estimator it
is therefore *not identified* — it must be calibrated from outside (§2) or given its
own moment (§3.4). This is not a limitation to work around; it is the reason the
user's route is the right one.

**A free specification test.** The same logic says the coefficient on `log N_sl` in
the *firm-level* cloglog, conditional on firm size, should be zero. Running the
empirical cloglog with `log N_sl` added is therefore a test of the micro-foundation
and of the size proxy at once, at essentially no cost. A significantly negative
coefficient would say either that the sales proxy for `z` is failing or that
within-cell competition is not being handled by `Φ` the way the model assumes.

---

## 2. Calibrating `η` in the data

### 2.1 The regression as proposed, and three corrections

The proposed regression is `γ_sl = α log d_l + β log N_sl`. Mapping it onto
`eq:gamma_rrs`:

```
log γ_{l s} = log T_{a(l)s} + η log N_sl − θ α log d_l + log(market access) − log Φ
```

so `β ↦ η` and the distance coefficient `↦ −θα`. Three things need fixing.

1. **Fixed effects (§1.3).** Without area × sector effects, `β` is not `η`.

2. **`η` is `β` itself, not `β/θ`.** Comparative advantage enters `γ` linearly while
   distance enters as `τ^{−θ}`. So the distance coefficient identifies only the
   product `θα` — but `η` is read off directly, free of `θ`. Given that
   `documentation/open_issues.md` §1.1 flags `θ = 1` as an unresolved placeholder on
   which every `θα` statement depends, **`η` is the one elasticity here that is
   immune to that decision.** Worth stating in the paper.

3. **Zeros and functional form.** `γ_ls` has exact zeros (cells with no observed
   sourcing) and those zeros are the informative part — they are what `G_s(0)` is
   about. `log γ` drops them and OLS on `log γ` would select on the outcome. Use
   **PPML** (`pyfixest.fepois`, already a dependency and already used for the
   untargeted moment), which is consistent for the conditional mean under the
   exponential form and keeps the zeros. A secondary reason: the exponential form is
   what the model implies, so PPML estimates the model's own object rather than a
   log-linear approximation of it.

A fourth, smaller point: because the own cell sits inside `Φ_{rs}`, the exact
elasticity is `η(1 − γ)` rather than `η`, so a direct regression *understates* `η`.
At the commuting-zone level within an area `γ` is small and the correction is second
order, but it is one more reason to prefer estimator (iii) below.

### 2.2 A ladder of three estimators

Run all three: they cost almost nothing and their disagreement is itself the
diagnostic.

**(i) Direct PPML — the user's regression, done properly.**

```
γ_ls  ~  Poisson( exp( FE_{a(l)×s} + η·log N_sl + δ·log d_l ) )
```
weights: none (or purchase weights); cluster on the area. `δ` recovers `−θα` and is a
free cross-check against the cloglog estimate of `α` — an over-identification test the
paper does not currently have. Robust, needs no model input. Its weakness is that
`log d_l` is a proxy for the full market-access index; whatever of that index is
correlated with firm density leaks into `η̂` (upward, since central cells have both
better access and more firms).

**(ii) PPML with the model's market access as an offset.**

The model gives `γ_{ls} ∝ T_{ls} · M_{ls}(α)` in closed form, where
`M_{ls} = Σ_r ω_r τ_{lr}^{−θ}/Φ_{rs}` — this is exactly the object
`invert_T_from_gamma` already builds. Put `log M_ls(α̂)` in as an offset and drop
`log d_l`. This removes the proxy error entirely; the price is that `η̂` now depends on
`α̂` and `θ`, so it must be re-run if either moves.

**(iii) Regress the inverted `T` on the firm count. *Recommended as the headline.***

Run the existing zone-level Sinkhorn inversion (`--ca_level=ze`, which is precisely
`invert_T_from_gamma` at the commuting-zone level) to get `T̂_ls` reproducing the
observed `γ_ls`. Then

```
T̂_ls  ~  Poisson( exp( FE_{a(l)×s} + η·log N_sl ) )
```

No distance term is needed: market access, the `Φ` denominator and the multi-buyer
aggregation are all inverted out, so `η` is the only thing left. This is the
model-consistent estimator and it reuses machinery that already exists and is already
gated. Its weakness is that the inversion returns `T̂ = 0` wherever `γ = 0`, which is
a degenerate value rather than a measurement; PPML tolerates the zeros arithmetically
but they should be checked (how many cells, and whether dropping them moves `η̂`).

If (i), (ii) and (iii) agree to within a standard error, `η` is well measured and the
choice is cosmetic. If (i) is much larger than (iii), the market-access proxy was
doing the work and (iii) is right.

### 2.3 Gates on the data, before anything is estimated

These are cheap and two of them are decisive.

- **G1 — `N_sr` must be a population count, not a supplier count.** If `N_sl` counts
  the firms observed *supplying the downstream industry*, regressing `γ` on it is
  circular: `η` would come out near 1 mechanically, block 5 and block 6 would carry
  the same information twice, and the exercise would be void. The test is immediate:
  **control cells (`filter == 1` and `X_rs == 0`) must have `N_sl ≥ 1`.** If those
  cells are all zero, `N_sr` is a supplier count — stop.
- **G2 — within-area variation must exist.** Report, per sector, the share of the
  variance of `log N_sl` that survives area fixed effects, and the partial
  correlation between `log N_sl` and `log d_l` within area. If the within-area
  variance share is small, `η` is weakly identified whatever the point estimate says.
  If the partial correlation with distance is strongly negative, the current `α` has
  been absorbing a firm-density gradient — which is the substantive motivation for
  the whole change, and should be reported as such.
- **G3 — `N_sl ≥ 1` on every simulated cell.** The old three-status encoding of
  `filter_N_upstream` had status 0 = "no firms", so the binary `filter == 1` should
  already imply at least one firm. Assert it at load rather than assume it: with
  `η > 0`, an `N_sl = 0` cell gets `T = 0` and can *never* supply, which would make
  its contribution to `Ḡ_s(0)` mechanical and would quietly corrupt `N̂_s`. If the
  assertion fires, either drop those cells from `CELL_MASK` *and* from the empirical
  `Ḡ_s(0)` denominator (consistently), or floor `N` at 1 — but decide, do not let it
  pass silently.
- **G4 — how endogenous is `N_sl`?** The defence for treating firm counts as
  exogenous is that the modelled downstream industry is a small share of each
  upstream sector's total demand. Report that share sector by sector. It will be
  weakest exactly where it matters most: `C30C` carries 56% of modelled aerospace
  purchases, and if aerospace also buys a large share of `C30C`'s output then the
  firm count in that sector is plainly a response to the industry, not a primitive.
  This is a caveat to state, not a problem to solve.

### 2.4 Where the calibrated value lives

Write `η` into `stats.csv` as a named row (`eta_firm_count`), read by the same
`_read_named_value` helper that already reads `prior_alpha`, so a missing value is an
explicit warning rather than a silent default. Store its standard error beside it —
§3.4 needs it.

---

## 3. Implementation

### 3.1 Julia — four sites, none of them deep

The specification enters at exactly one conceptual point (the gather of `T` from the
parameter column space onto commuting zones), which appears in four places:

1. **`load_parameters.jl` SECTION 1 / 2b.** Load `N_sr.npy` `(S, R)`; read `η` from
   `stats.csv` (or a `--eta=` flag); run gates G1/G3; build and broadcast
   ```julia
   @everywhere const N_FAC = $(N_sr_local .^ eta_local)   # (S, R), ≡ 1 when η = 0
   ```
   Build it once, as a matrix, so no hot loop ever calls `^`.

2. **`model_CP.jl` `gather_T_to_ze`** — `T_mat[s,l] = T_par[s, T_GATHER[l]] * N_FAC[s,l]`.

3. **`model_CP.jl` `unpack_params`** — the same gather is written out inline there.
   **Refactor it to call `gather_T_to_ze`** rather than patching the duplicate; two
   copies of this loop is exactly how a `T` matrix silently transposes (the codebase
   has a documented history of that failure mode).

4. **`load_parameters.jl` `invert_T_from_gamma`** (the warm start) — two lines: the
   `Φ` accumulation gains the factor, and the market-access aggregation back to the
   column space becomes `M_col[T_GATHER[l]] += M * N_FAC[s,l]`, because
   `γ_as = T_as · Σ_{l∈a} N_sl^η M_ls`.

**`profiling.jl` needs no change at all.** `invert_T_ge` already routes through
`gather_T_to_ze` and `aggregate_gamma_to_T`, so it inherits the factor. That is worth
verifying explicitly rather than trusting, but it is the design's main dividend.

Everything downstream — `solve_network`'s `scale = max(T_sr, eps)^(1/θ)`, the
regression cells, `q̂`, `concentrate_N_s`, the Jacobian, the weight matrix — reads
`T_mat` and needs nothing.

### 3.2 Python — two sites

5. **`diffusion.ipynb`** — the loader gains `N_sr` and `N_FAC`; `sourcing_geometry`'s
   one line
   `T_cell = np.ones(...) if equalise_T else est["T"][s, aa_of_ze[cells]]`
   becomes `... * N_fac[s, cells]`, and `equalise_T` becomes a **three-way switch**
   (§4.3).
6. **`analysis_granular.ipynb`** — same `T` reconstruction; the §4 comparative-
   advantage section (the variance decomposition, the distance equivalence, the
   alignment covariance) all read `T` at the cell level and will change numerically.

### 3.3 Gates

- **V0 — `η = 0` must reproduce the current estimator bit-identically.** `N_FAC ≡ 1`
  makes this structural rather than approximate, which is the point of implementing it
  as a multiplicative factor. This is the analogue of the `--granular=false --ca_level=ze`
  reference run and should be the first thing checked.
- **V1 — the two gathers agree.** After the refactor, assert `unpack_params`'s `T`
  block equals `gather_T_to_ze(unpack_T_par(θ))` for a random `θ`.
- **V2 — Sinkhorn still converges and still reproduces `γ_as` exactly** at the
  calibrated `η`, with the residual and iteration count reported. The kernel changed;
  the contraction constant may have moved.
- **V3 — `α̂` should barely move** (§1.4). If it moves materially, the block-4
  insulation argument is wrong and must be diagnosed before anything is quoted.

### 3.4 If `η` is later wanted as an estimated parameter (optional, Phase 3)

Two changes: add a seventh moment block — the within-area PPML coefficient of §2.2(i),
one scalar per industry (or per sector, if the sector-level estimates are worth
matching) — and add `η` to the free head beside `α`, with `T_as` still profiled. The
parameter layout gains one entry after `α`, which ripples into `PARAM_LABELS`,
`jacobian_param_indices`, the PSO bounds and the T delta method. Worth it only if the
paper wants a standard error on `η` and an over-identification check.

**In the interim, propagate the calibration uncertainty.** `T* = T*(α, γ, η)`, and the
existing delta method (`compute_profiled_T_inference`) already propagates `α` and `γ`
noise into `T`. Adding `∂T*/∂η · Var(η̂)` is one more finite-difference column and is
structurally identical to the `∂T*/∂α` column that already exists. Without it, the
reported `T` standard errors treat a calibrated number as known exactly.

---

## 4. Consequences

### 4.1 What does not move

- `α̂`, hence `θα`, hence the distance-equivalence figure, the `(αθ)²` law for
  buyer-specificity, and `§4`'s strength comparison. (Prediction, gated by V3.)
- `Ω_L`, `Ω_s`, `A_r` and blocks 1–3.
- `γ_as` — matched exactly by construction, before and after.
- The estimator's cost: `N_FAC` is precomputed; the loss is unchanged in complexity.

### 4.2 What moves

- **`T̂_as` becomes productivity *per firm*, not per cell.** Its cross-area dispersion
  will fall, since the density component is now explicit. `§4` reports
  `sd(log T̂) ≈ 0.91 / 0.96` against `θα`; that number will fall and must be
  re-interpreted — it is no longer the whole of comparative advantage.
- **The variance decomposition of `log ψ = log T − θα log d` gains three terms.**
  With `log T = log T_a + η log N`:
  ```
  Var(log ψ) = Var(log T_a) + η² Var(log N) + (θα)² Var(log d)
             + 2η Cov(log T_a, log N) − 2θα Cov(log T_a, log d) − 2θα η Cov(log N, log d)
  ```
  Six terms instead of three, and the new co-location term `Cov(log N, log d)` is the
  one that says whether firm density and proximity to the buyer reinforce each other.
  This is a richer table, not a broken one.
- **`N̂_s` will most likely rise.** `Ḡ_s(0) = mean_l (1 − q̂_ls)^{N_s}` is convex in
  `q̂`, so adding within-area dispersion to `q̂` raises `Ḡ_s(0)` at a given `N_s`; the
  bisection compensates by raising `N̂_s`. **Watch the `N_HI = N^obs_s` ceiling**: a
  sector clamped at `:hi` is a rejection signal, not a fitted value.
- **Concentration rises across the board.** More dispersed `T` within areas means the
  same few zones win for everybody: `H` up, `n_eff` down, `Q` up, `C^gran` up. Every
  number in `§5.1` moves, and the four-cell decomposition shifts toward the common
  cells. The direction is predictable; the magnitude is not.

### 4.3 What this buys — three things the current model cannot do

**(a) Within-area sourcing stops being explained by distance alone.** This is the
substantive point. Today every commuting zone of an area carries the *same* `T`, so
the model must attribute all within-area variation in `γ_ls` to distance and
granularity. That is a strong restriction and it is almost certainly rejected. **Check
it before implementing anything**: compare the model's within-area dispersion of
`γ_ls` against the data's. If the model already fits it, `η` is not needed; if it
badly under-predicts it, that gap is the size of the prize and should be the first
figure of the section.

**(b) The zeros get an economic driver.** Under `:aa`, control cells are simulated
goods whose zeros are model predictions — the paper's central claim for the granular
extension. Today a control cell's zero can only be explained by its distance. With
`η > 0` a thin cell gets a low `T` and its zero is explained by the thing that
plausibly explains it. Symmetrically, a *firm-dense* cell with no supplier becomes a
sharper puzzle — the model is being made more falsifiable, not more flexible.

The discipline that makes this a result rather than a tautology: **calibrate `η` on
the intensive margin (`γ`, §2.2) and validate it on the extensive margin.** The
untargeted count curve `G_s(K)` for `K = 1,2,3` is the natural test, and the paper
already records the current failure there ("the model overstates those with exactly
one supplier"). More dispersion in `q̂` moves mass out of `K = 1` into both tails,
which is the right direction. If `η` calibrated on `γ` fixes the `K = 1` overshoot,
that is genuine out-of-sample evidence. Calibrating `η` on `G_s(K)` instead would be
circular and must not be done.

**(c) Comparative advantage splits into two economically distinct forces.** The
counterfactual `equalise_T` currently means one thing. It now means three, and the
decomposition is interesting in its own right:

| regime | what is switched off | reads as |
|---|---|---|
| `equalise_T_area` | `T_as` equalised, density kept | pure technological/Ricardian advantage |
| `eta = 0` | density kept out of `T`, area advantage kept | agglomeration / extensive capacity |
| both | the current `equalise_T` | all of comparative advantage |

Asking which of the two localises shocks — is aerospace concentrated because Toulouse
is *good*, or because Toulouse is *thick*? — is a question the current model cannot
pose. It is arguably the best reason to make this change.

---

## 5. Risks, honestly stated

1. **Circularity if `N_sr` is a supplier count** (gate G1). Fatal if unchecked.
2. **Endogeneity of firm location** (gate G4). Firms locate where the industry is.
   The defence is that the modelled industry is a small share of each upstream
   sector's demand, and it is weakest in the sector that matters most (`C30C`). This
   biases `η̂` up and, through it, overstates how much of the concentration in `§5.1`
   is comparative advantage.
3. **Weak identification** (gate G2). If `log N` has little within-area variance,
   `η̂` will be noisy and the whole exercise is cosmetic. Report the within-area
   variance share, not just the t-statistic.
4. **The model fit improves for a partly mechanical reason.** Firm counts predict
   supplier presence nearly by construction in the data. The §4.3(b) discipline
   (calibrate on `γ`, validate on `G_s(K)`) is what separates a result from a
   tautology, and it should be stated in the paper, not just observed in the code.
5. **`§5.1` has to be re-run in full.** Concentration, the four cells, `V`, `Q`,
   `C^gran`, the counterfactual table and `fig:concentration_decomposition` all move.
   This is a re-reading of a written section, not a code change, and it is the largest
   time cost of the whole proposal.
6. **`θ = 1` is still a placeholder** (`open_issues.md` §1.1). `η` is immune to it
   (§2.1), but estimator (ii) and every `θα` statement are not. Doing this change and
   the `θ` decision in the same run would confound the two; do them separately.

---

## 6. Plan

**Phase 0 — decide whether it is worth doing (no code).**
- G1, G2, G4 on the data. If G1 fails, stop.
- Measure the current model's within-area `γ_ls` dispersion against the data's
  (§4.3(a)). This is the size of the prize and it is one figure off artefacts that
  already exist.

**Phase 1 — calibrate `η` (data side, no model change).**
- Estimators (i), (ii), (iii) of §2.2, per industry and per sector; report the three
  side by side and reconcile the disagreements.
- The free specification test of §1.4: `log N` in the firm-level cloglog should carry
  a zero coefficient.
- Write `η̂` and its standard error into `stats.csv`.

**Phase 2 — implement (four Julia sites, two Python).**
- §3.1 in order, with the `unpack_params` refactor rather than a second copy of the
  gather.
- Gates V0–V3. V0 first: `η = 0` bit-identical to the current run.
- Re-run both industries; report `α̂` (should not move), `N̂_s` (should rise, watch
  the `:hi` clamp), and the Sinkhorn residual.

**Phase 3 — validate and re-read.**
- The untargeted `G_s(K)` curves — the out-of-sample test of §4.3(b).
- The three-way counterfactual of §4.3(c).
- Re-read `§4` (the six-term decomposition) and `§5.1` (everything) on the new run.
- Optionally promote `η` to an estimated parameter (§3.4); at minimum, propagate its
  calibration variance into the `T` standard errors.
