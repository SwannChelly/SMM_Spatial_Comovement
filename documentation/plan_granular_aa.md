# Implementation plan — granular varieties + comparative advantage at the attraction-area level

Target: the **simple version**. Comparative advantage lives at the attraction-area (AA) × sector
level only (`σ²_ε = 0`, no ZE-level deviation); the number of varieties `N_s` is finite, per
sector, and estimated.

Companion theory: `documentation/finite_sample2.tex` — §3 (two-tier geography), §5 (count moment),
§6 (what must change), §3.9 (the empirical within-area diagnostic), §7 (computing the granular
model).

---

## 0. The five design decisions

### D1 — The economy is simulated with exactly `N_s` varieties, `R` replications

Each replication is one realised granular economy: `N_s` varieties per sector, one Fréchet
champion drawn per (cell, variety), Ricardian competition across cells for each variety and each
downstream buyer. Moments are computed **on the realised economy** and averaged over `R`
replications.

Three reasons this beats the closed-form alternative (estimate `q_ls` from a large pool, apply
`Pr(K = 0) = (1−q)^{N_s}` analytically); the full comparison with numbers is §7.

1. **The finite-variety economy has to be solvable anyway.** Spatial comovement is propagation
   through the *realised* network; the granular extensive margin is what makes it interesting. A
   closed-form estimation would leave "how do I now solve the economy at `N̂_s`?" unanswered, and
   would force a re-mapping step to avoid reintroducing the love-of-variety factor.
2. **It matches the binding function.** The empirical `reg_coef` is an auxiliary statistic on one
   realised economy of fixed size; simulating economies cancels the auxiliary estimator's
   finite-sample bias (measured at 2–4% of `β`, §7.2).
3. **There is no `q̂` to resolve.** A cell with a tiny win probability simply wins zero varieties
   — the right answer with the right probability.

**`R` replications, never `N_rho = N_s`.** The value block is a numerical integral; the variety
count is a structural parameter. At `N_s ≈ 10²` the labor share, `π_s`, `π_r` and the shares
computed from `N_s` draws would be hopeless. Granularity stays exactly `N_s` *within* a
replication; the value block averages over all `R × N_s` draws.

**Fixed draw pool + prefixes.** Pre-draw `N_max` varieties per sector per replication and take the
first `N_s`. Common random numbers across `N_s` candidates and across `θ`, which is what keeps the
optimiser usable.

### D2 — The value block keeps the self-normalised weights

`sample_weights[ρ,g] = 1/N_ρ` are self-normalised per column, so the within-sector CES index is an
**average**, not a sum — the certainty-equivalent price index, invariant to the variety count.
That is Option 1 of the note's §6(a) implemented implicitly. Keep it, and keep it in the finite
solve too (weights `1/N_s`), so the estimated economy and any counterfactual are the same object.

> **Carry into counterfactuals.** The love-of-variety channel is normalised out at the calibration
> point, not removed from the economics. A counterfactual that moves the variety set must undo the
> normalisation first (note App. D.4) or the propagation channel is silenced.

### D3 — Comparative advantage, and the γ moments, at the AA level

`T_{sl} = T_{s,a(l)}` for every ZE `l` in attraction area `a` — one parameter per active
(sector, AA). Draws stay independent across ZE: two ZE in an area share the Fréchet *scale*, not
the realisation.

Block 5 becomes the AA aggregate `γ_{s,a} = Σ_{l∈a} γ_{ls}`. A ZE inside an active AA with no
supplier has `γ̂_ls = 0` and a bootstrap variance of zero — it cannot enter the γ block without
infinite weight. Its information goes to the extensive margin and the counts. Use the
`..._aa.npy` Σ files. The γ block is then just-identified against `T_aa` (one moment per active
`(s,a)`, minus one reference per sector). The ZE-level variant stays behind a flag as the
over-identifying configuration of the note's §6(b) — not part of this version.

### D4 — The extensive-margin regression is at the **firm** level, and `not_supply` is its exact form

**This is the point that determines the whole design of block 4.** The empirical regression is

```
y_i = 1{firm i is a supplier}   on   distance + log(z_i) + FE(sector × AA)
```

one row per **firm**, with the firm's own productivity as control. In the granular model a firm
**is** a (cell, variety) champion:

| model object | code | data object |
|---|---|---|
| champion productivity of cell `g` for variety `ρ` | `z_flat[ρ, g]` (`model_CP.jl:609–616`) | `z_i` |
| that champion wins somewhere | `linkages_flat[ρ, g]` (`model_CP.jl:720`) | `supplier_i` |
| cell's distance to its downstream anchor | `LOG_CLOSEST_DIST[r]` / `DistBin[r,dr]` | `log_d` |
| FE group | `(s−1)*R_downstream + CLOSEST_DOWNSTREAM_REGION[r]` | sector × AA |

So the **existing `(ρ, g)` row structure is the right one**, and it is right for a reason rather
than by accident: under granularity `ρ` runs `1..N_s` instead of `1..N_rho`, and each row is a
firm. The design is `n_cells × N_s` rows per replication.

**And the existing `not_supply` convention is also right — I was wrong to push the supplier
indicator.** Condition on the firm's own productivity `z`. Its delivered cost to buyer `a` is
`w_l τ_{la}/z`, and it supplies iff every competitor's cost is higher. The competitors' minimum is
Weibull, so

```
Pr(firm supplies | z) = exp( − Φ⁻ˡ_{sa} (w_l τ_{la})^θ z^{−θ} )
⟹ Pr(NOT supply | z)  = 1 − exp(−exp(η)),   η = ln Φ⁻ˡ_{sa} + θ ln w_l + θα ln d_{la} − θ ln z
```

which is **exactly a cloglog on `not_supply`**, with `+θα` on log distance and `−θ` on log
productivity, the fixed effect absorbing `ln Φ⁻ˡ_{sa}`. Verified by simulating the exact object
(`θ = 1`, `α = 0.30`, firms = (cell × variety) champions):

| convention | `b_logd` (truth `+θα = 0.30`) | `b_logz` (truth `−θ = −1.0`) |
|---|---|---|
| **`y = not_supply`** | **+0.327** | **−1.051** |
| `y = supplier` | −0.426 | +1.452 |

The supplier indicator recovers nothing. Proposition 1 of the note (`y = 1{supplier}`, slope
`−θα`) is a **cell-level** statement about `Pr(K_ls ≥ 1)`; it does not transfer to the firm-level
regression, and the code's original comment at `model_CP.jl:1042` was correct.

Two consequences worth having.

* **`N_s` does not appear in `η` at all.** The firm-level regression is *exactly* invariant to the
  variety count — cleaner than the cell-level case, where the invariance was only up to
  specification error. Block 4 carries `α`, block 6 carries `N_s`, with no residual coupling.
* **The `log z` coefficient is a free over-identifying test: it should equal `−θ`.** With `θ`
  calibrated at 1.0, the empirical coefficient on `log z` should come out near `−1`. A large
  departure means either `θ ≠ 1` or the firm↔champion mapping of §1 is broken — exactly the failure
  mode that the open modelling decision below is about. It costs nothing to report. (If the β block
  of Σ were extended to include it, `θ` could be estimated rather than calibrated; that is an
  extension, not part of this version.)

Two further consequences for the code.

* **The "collapse to one row per cell" idea is dead**, and it was wrong: `log(z_flat[ρ,g])` varies
  across varieties within a cell, exactly as objected. `Pr(K_ls ≥ 1) = 1 − (1−q_ls)^{N_s}` is a
  *different* moment — the cell-level extensive margin — and it is not what the empirical
  regression estimates.
* **The `include_control` / `include_size_control` exclusion can be dropped.** The assertion at
  `model_CP.jl:846` and `:1053` exists because status-2 (control) cells had `T ≡ 0` and hence no
  productivity draw. Under D3 they inherit `T_{s,a} > 0` and become ordinary goods with real
  draws, so both the control rows and the `log z` control can be on at once. That removes the
  awkward coupling in the current code.

### D5 — `N_s` is profiled over the integers, with a closed-form inner loop

`N_s` is estimated by the same weighted objective as everything else, concentrated rather than
searched jointly:

```
N̂(θ_c) = argmin over N ∈ ∏_s [N_LO_s, N_HI_s] ∩ ℕ  of  r(θ_c, N)' W r(θ_c, N)
```

— the textbook profiled extremum estimator, identical to joint minimisation over `(θ_c, N)`.
`N̂_s` is an integer at every evaluation: integrality is never relaxed, never rounded, and the
outer optimiser never sees `N_s`.

Profiling on realised economies would need `R ×` (number of candidates) regressions per loss
evaluation. So **locate `N̂_s` with the closed form and evaluate the loss with the simulated
economies at that value**:

```
q̂[s,l] = mean over the pooled draws of linkages_flat[·, g(s,l)]     # win-somewhere probability
G(s,n) = mean over l ∈ cells_s of (1 − q̂[s,l])^n                    # closed form, strictly ↓ in n
N̂[s]   = integer bisection of G(s,·) onto G_TARGET[s], clamped to [N_LO_s, N_HI_s]
```

This is legitimate because `q_ls` is free of `N_s` — proved in `finite_sample2.tex` §7.3: winning
a variety is a comparison of relative delivered costs, so `q_ls` depends only on `(T, w, τ, θ)`,
while `N_s`, the price index, downstream costs and expenditures are aggregators built *after* the
winners are known. (The one condition that would break it is endogenous upstream wages; wages are
data here.) Accuracy requirements on `q̂` are mild since the bisection is on a monotone function.

**Why the bisection is nearly exact.** With `y = 1{supplier}` the cloglog index separates
*exactly*:

```
η_ls = ln(−ln Pr(K_ls = 0)) = ln N_s + ln(−ln(1 − q_ls))
```

and `ln N_s` is constant within sector, hence absorbed by the sector × AA fixed effect. Verified
over a factor of 667 in `N_s`: the coefficient holds at `−0.30013` with 3.8e−4 relative drift and
equals `−θα`. So the count block carries `N_s` and the regression block carries `α`. Two riders:
the invariance degrades to ~9% under misspecification (fit `log d` when the truth is binned),
which argues for `N_REG = 4`; and it is derived for the cell-level extensive margin, so with the
firm-level regression of D4 the drift must be **measured** at `θ̂`, not assumed (gate G5).

---

## 1. Data inputs

### Supplied, conventions settled

| File | Convention |
|---|---|
| `G_K.csv` — `A129`, `K`, `G(K)`, `N_supplier_s` | `G(K) = Pr(K_ls ≤ K)`; `G(0)` = share of ZE with **0** suppliers. Denominator: **ZE inside active attraction areas only**. `N_supplier_s` = total suppliers in the sector, repeated on every row (assert constant within `A129`). |
| `Sigma_beta_gamma_cloglog[_1]_f_aa.npy` | Ordering **β → γ(AA) → G**, exactly `S` rows of `G_s(0)`. **This version uses this file.** |
| `Sigma_beta_gamma_cloglog[_1]_f.npy` | Same with γ at ZE level — over-identifying variant, behind a flag. |
| `closest_downstream_region.npy` | `R × R_downstream` binary incidence: which ZE belongs to which AA. |

Derived immediately:

```
N_HI[s]     = N_supplier_s                        # every variety won by a single origin
N_LO[s]     = ceil(N_supplier_s / R_downstream)   # a different origin per buyer
G_TARGET[s] = G_K.csv at K = 0
```

### The AA map must be verified at load time, not assumed

The code already computes the same partition internally as `CLOSEST_DOWNSTREAM_REGION`
(`load_parameters.jl:507`), which builds today's FE group. Assert equality at startup (gate G7):

```
argmax_col(closest_downstream_region.npy)[l] == CLOSEST_DOWNSTREAM_REGION[l]   ∀ l
rows sum to exactly 1 ; shape is (R, R_downstream) ; all AA indices in range
```

A mismatch means the model's fixed effect and the empirical one are different partitions, which
silently invalidates the alignment argument of `finite_sample2.tex` §2.2.

### Distance: two mismatches to fix on the Python side

* **Floor.** Julia uses `log(max(d, 1.0))` (`load_parameters.jl:510`). Mirror it —
  `np.log(np.maximum(d, 1.0))` — or a ZE hosting its own downstream plant gives `−inf`.
* **Which distance.** Julia's regressor is `LOG_CLOSEST_DIST[r]`, distance to the **nearest**
  downstream region, and bins are `DistBin[r, dr]` at that same region. Confirm
  `dist_com_w_arithmetic` is that and not an arithmetic mean over all downstream plants.

The **bins already match**: `distance_bin(d, 4)` gives `(50,100] → 1`, `(100,150] → 2`,
`(150,200] → 3`, `d > 200 → 4`, and `d ≤ 50 → 0` = no dummy = the reference. Make the Python side
explicit rather than lucky:

```python
df["dist_bin"] = pd.cut(d, [0, 50, 100, 150, 200, np.inf],
                        labels=["0-50","50-100","100-150","150-200","200+"], right=True)
# formula: "... + C(dist_bin, Treatment(reference='0-50')) + ..."
```

`right=True` reproduces Julia's `50 < d <= 100`.

### The open modelling decision: firm counts per cell

D4 makes the firm-level regression a model object, but exposes a mapping question that must be
decided before the first run.

**In the model, cell `(s,l)` contains exactly `N_s` champion-firms — one per variety.** In the
data the firm count per cell varies by orders of magnitude. So the model, read literally, predicts
the same number of firms everywhere, which is false. Three ways out:

1. **Accept the sample mismatch for v1.** Run the model regression over the `n_cells × N_s`
   champion rows and the empirical one over the observed firms. Same regression, different
   sample. The weakness is specific and measurable: the model's `z` are *maxima* over a continuum,
   so their distribution is shifted right relative to the data's firms, and the two `log z`
   coefficients are estimated over different supports. **Measure the overlap before trusting it**
   (gate G6).
2. **(Recommended) Give the model a firm count via a productivity floor.** With a floor `z̄_s`,
   the number of firms in `(l,s,ρ)` above it is `Poisson(T_sl z̄_s^{−θ})`, so the cell's firm count
   is `Poisson(N_s T_sl z̄_s^{−θ})` — varying with `T_sl` as in the data. One extra parameter per
   sector, and the observed firm count per cell becomes a **new moment**, a strong one for `T`.
   The regression then runs over exactly the model's firms with the same `z` support as the data.
   Note this reinstates the ingredient the note's Question 1 explicitly set aside ("no need for a
   Pareto lower bound or a Poisson firm count") — **the choice of a firm-level empirical moment is
   what forces it back in**, and that is worth recording as a modelling consequence rather than an
   implementation detail.
3. **Take the observed firm count as given.** Cell `(s,l)` has `M_ls` firms from the data, each
   assigned to a variety; a firm supplies iff it is its cell's best for that variety and the cell
   wins somewhere. No new parameter and the firm count is right by construction, but `M_ls`
   becomes exogenous data rather than a model outcome, and the firm→variety assignment needs a
   rule.

Recommendation: build v1 on option 1 with gate G6 measuring the damage, and treat option 2 as the
first extension.

### Deferred, recorded so it is not lost

**`SIREN` (cell-level firm count) is not in this version.** When it arrives it is a cell-level
covariate, so it joins the design as one extra column and must simultaneously enter the empirical
target. Note it is *not* a substitute for `log z`: `log z` is the firm's own productivity
(firm-level), `log SIREN` is the cell's firm count (cell-level). §3.9 showed the latter is the
dominant *within-area* force, so until it is included `α̂` is expected to be biased upward in
magnitude.

---

## 2. Moment vector layout

| # | Block | Before | After |
|---|---|---|---|
| 1 | `agg_labor_share` | 1 | unchanged |
| 2 | `agg_industry_share` | S−1 | unchanged |
| 3 | `pi_r` | R_d−1 | unchanged |
| 4 | `reg_coef` | `N_REG` | same size; **firm-level rows over `N_s` varieties**, outcome stays `not_supply`, `log z` control on |
| 5 | `gamma_ls` | active (s,ZE) − ref | **active (s,AA) − ref** |
| 6 | `G0` | — | **new**: `G_s(0)`, one per sector |

Σ ordering **β → γ → G**, extending the existing `gb_indices` invariant.

---

## 3. The estimation algorithm

### 3.1 Pseudo-code

`R` = ZE, `R_d` = downstream regions = number of AAs, `S` = sectors, `n_cells` = |CELL_MASK| ≈ 1161.

```text
════════════════════════════════════════════════════════════════════════════════
PRECOMPUTED ONCE   (load_parameters.jl)
════════════════════════════════════════════════════════════════════════════════
AA_OF_ZE        :: Vector{Int}(R)        from closest_downstream_region.npy
                                         @assert == CLOSEST_DOWNSTREAM_REGION      (G7)
AA_ACTIVE       :: BitMatrix(S, R_d)     a hosts ≥1 observed supplier in s   → 𝒜⁺_s
CELL_MASK       :: BitMatrix(S, R)       has_firms[s,l] && AA_ACTIVE[s, AA_OF_ZE[l]]
                                         has_firms = filter_N_upstream ∈ {1,2}
CELLS_OF_SECTOR :: Vector{Vector{Int}}(S)
GOOD_S, GOOD_R, SR_TO_GOOD               rebuilt from CELL_MASK  (n_good = n_cells)
T_MASK_AA       :: BitMatrix(S, R_d)     = AA_ACTIVE                → estimated T block
T_REF_AA        :: Vector{Int}(S)        one reference AA per sector, T[s,ref] ≡ 1
emp_gamma_aa    :: Matrix(R_d, S)        Σ_{l∈a} emp_gamma_ls[l,s]  → block-5 target
G_TARGET        :: Vector{Float64}(S)    G_K.csv at K = 0           → block-6 target
G_CURVE         :: Vector{Vector}(S)     full curve                 → gate G4 only
N_LO, N_HI      :: Vector{Int}(S)
N_MAX           :: Int                   = maximum(N_HI); size of the draw pool
LOG_DIST        :: Vector(R)             log(max(closest_plant_dist, 1.0))
DIST_BIN        :: Vector{Int}(R)        ∈ 0:N_REG, 0 = reference (d ≤ 50)
FE_GROUP        :: Vector{Int}(n_cells)  (s−1)*R_d + AA_OF_ZE[l]
U_POOL          :: Array(R_rep, N_MAX, n_cells)   fixed uniform draws — common random numbers

════════════════════════════════════════════════════════════════════════════════
PARAMETER VECTOR   (N_s is NOT in it)
════════════════════════════════════════════════════════════════════════════════
θ_c = [ Ω^L(1) | Ω^s(S) | A(R_d) | α(N_TAU) | T_aa(n_T_aa) ],  n_T_aa = count(T_MASK_AA)

════════════════════════════════════════════════════════════════════════════════
ONE LOSS EVALUATION
════════════════════════════════════════════════════════════════════════════════
function loss(θ_c, W)

  ── 0. unpack, gather AA → ZE ────────────────────────────────────────────────
  Ω^L, Ω^s, A, α, T_aa = unpack_params(θ_c)          # T_aa :: (S, R_d), ref-normalised
  T[s,l] = T_aa[s, AA_OF_ZE[l]]                      # (S,R) gather — the one new structural line

  ── 1. locate N̂_s : closed form on the POOLED draws, no economy needed ───────
  #   solve once at full pool width to get the win-somewhere probability
  net_pool = solve_network(θ_c; u_draws = flatten(U_POOL))       # R_rep*N_MAX rows
  q̂[s,l]   = mean over pooled ρ of net_pool.linkages_flat[ρ, SR_TO_GOOD[s,l]]
  q̂[s,l]   = clamp(q̂[s,l], 0.5/(R_rep*N_MAX), 1 − 1e-12)
  for s in 1:S
      G(s,n) = mean over l ∈ CELLS_OF_SECTOR[s] of (1 − q̂[s,l])^n      # ↓ in n
      if     G(s,N_LO[s]) ≤ G_TARGET[s]   N̂[s], clamped[s] = N_LO[s], :lo
      elseif G(s,N_HI[s]) ≥ G_TARGET[s]   N̂[s], clamped[s] = N_HI[s], :hi
      else
          lo, hi = N_LO[s], N_HI[s]                              # G(lo) > target > G(hi)
          while hi − lo > 1
              mid = (lo + hi) ÷ 2
              G(s,mid) > G_TARGET[s] ? (lo = mid) : (hi = mid)
          end
          N̂[s], clamped[s] = argmin_{n ∈ {lo,hi}} |G(s,n) − G_TARGET[s]|, :none
      end
  end

  ── 2. R realised granular economies, prefix N̂_s of the pool ─────────────────
  for rep in 1:R_rep
      net[rep] = solve_network(θ_c; u_draws = U_POOL[rep, 1:N̂[sector], :])
      #   → linkages_flat :: (N̂_s, n_cells), z_flat :: (N̂_s, n_cells)
      #   value block uses weights 1/N̂_s  (average, not sum — D2)
  end

  ── 3. block 4 — FIRM-level cloglog, y = 1{supplier}, control log z ──────────
  #   rows = (variety ρ, cell g) = firms.  n_cells × N̂_s per replication.
  for rep in 1:R_rep
      y[ρ,g]   = 1 − net[rep].linkages_flat[ρ, g]        # not_supply — the EXACT firm-level form
      X[ρ,g,:] = [ DIST_BIN onehot (or LOG_DIST) | log(net[rep].z_flat[ρ,g]) ]
      w[ρ,g]   = 1 / N̂_s                                 # each cell carries total weight 1
      reg[rep] = cloglog_irls(y, X, fe = FE_GROUP[g])
  end
  reg_coef = mean over rep of reg[rep][1:N_REG]          # binding function estimate
  b_logz   = mean over rep of reg[rep][N_REG+1]          # DIAGNOSTIC: should equal −θ (= −1.0)

  ── 4. block 5 — AA-level shares, averaged over replications ─────────────────
  γ_aa[a,s] = mean over rep of Σ_{l : AA_OF_ZE[l] == a} γ[rep][l,s]

  ── 5. block 6 — realised empty-ZE share, averaged ───────────────────────────
  G0[s] = mean over rep of ( share of l ∈ CELLS_OF_SECTOR[s] with K_ls[rep] == 0 )
  #   K_ls[rep] = Σ_ρ linkages_flat[ρ, g(s,l)]  — the realised supplier count

  ── 6. blocks 1–3, then assemble ─────────────────────────────────────────────
  labor, industry, π_r = mean over rep of compute_moments(net[rep], θ_c)
  m = vcat(labor, industry, π_r, reg_coef, γ_aa[mask], G0)
  return (m .− m̂)' * W * (m .− m̂),  (N̂, clamped, K_dist)
end

════════════════════════════════════════════════════════════════════════════════
ONCE AT θ̂ — verification, not an inner loop
════════════════════════════════════════════════════════════════════════════════
for s in 1:S, n in {N̂[s]−δ … N̂[s]+δ} ∩ [N_LO[s], N_HI[s]]      # δ ≈ 10% of N̂[s]
    record the FULL weighted loss with N̂[s] := n               # re-uses U_POOL prefixes
report argmin vs N̂  ⇒  if it differs, step 1's closed-form bisection is not locating the
                        profiled optimum and must be promoted to a full-loss line search
```

### 3.2 Sizing `R_rep`

Set by the dispersion of `β̂` across realised economies: measured at ~0.16–0.18 in configurations
matching the data, so `R_rep ≈ 300` gives an MC standard error of ~0.010 on block 4. At
`N̂_s ≈ 10²` that is ~3×10⁴ variety-draws against today's `N_rho = 1000` — roughly 30× the current
simulation cost. Common random numbers absorb most of that across `θ`, which is what keeps the
optimiser workable; measure it in Phase 0 before committing.

### 3.3 Integrality and the bounds — read the clamp as a diagnostic

`N̂_s` is an integer at every evaluation. Clamping is informative rather than benign:

* `clamped = :hi` — even the maximum admissible count (`N_supplier_s`) leaves too many empty ZE;
  the model cannot generate enough sparsity.
* `clamped = :lo` — even the minimum admissible count produces too few empty ZE.

Either is a **rejection signal for the mechanism**. Log it per sector per report.

`N̂_s(θ)` is a step function of the continuous parameters, so the profiled loss is piecewise smooth
with jumps. PSO / TikTak are derivative-free and unaffected. The FD Jacobian can straddle a jump:
at `θ̂` recompute it **holding `N̂_s` fixed** — the correct object anyway, since `N̂_s` is locally
constant with probability one — and assert that no perturbation moved it.

### 3.4 A second, free route to `N_s`

`E[Σ_l K_ls] = N_s Σ_l q_ls`, so `N_s^count = N_supplier_s / Σ_l q̂[s,l]` is a closed-form second
estimate. This identity is what generates the bounds (`Σ_l q_ls ∈ [1, min(N_d, R)]`), so it is
admissible by construction. Print both every report; a large gap is a mechanism finding.

---

## 4. File-by-file change list

### `load_parameters.jl`

* **Active set — the biggest change.** Replace `active_mat = (filter_N_upstream .== 1) .& (X_rs .> 0)` by
  * `has_firms[s,l] = filter_N_upstream[s,l] ∈ {1,2}` (status 0 = no firms, stays out);
  * `AA_ACTIVE[s,a] = any(l ∈ a : filter==1 && X_rs>0)` → `𝒜⁺_s`;
  * `CELL_MASK[s,l] = has_firms[s,l] && AA_ACTIVE[s, AA_OF_ZE[l]]`.

  Then `n_good`, `GOOD_S`, `GOOD_R`, `SR_TO_GOOD`, `SECTOR_GOOD_INDICES` are built from
  `CELL_MASK`. Status-2 (control) cells become ordinary goods with `T = T_{s,a} > 0` and real
  productivity draws — **this is the endogenisation**. `CONTROL_S/CONTROL_R/N_CONTROL` become
  vestigial and `INCLUDE_CONTROL` loses its meaning.
* New: `AA_OF_ZE`, `n_AA`, `AA_CELLS`, `T_MASK_AA`, `T_REF_AA`, `CELL_MASK`, `CELLS_OF_SECTOR`,
  `N_LO`, `N_HI`, `N_MAX`, `G_TARGET`, `G_CURVE`, `R_REP`, `U_POOL`.
* γ target → `emp_gamma_aa`. Keep `emp_gamma_ls` for the ZE-level variant and the §3.9 diagnostic.
* Six-block `compute_block_ranges`; `BLOCK_NAMES += ("G0",)`; block 5 over `(s,AA)` minus one
  reference AA per sector.
* Labels: `T[sname-aaname]`, `gamma[sname-aaname]`, `G0[sname]`.

### `model_CP.jl`

* `unpack_params`: `T_reduced` scatters into `(S, n_AA)` s-major, ref-normalise on `T_REF_AA`,
  then gather `T[s,l] = T_aa[s, AA_OF_ZE[l]]`. Keep the `eltype(params)` generic array so the AD
  path survives.
* `solve_network`: no logic change; it receives `N̂_s` rows instead of `N_rho`, and ~8× more goods.
* New `simulate_replications(θ_c, N̂, U_POOL) -> Vector{net}` — the `R_rep` loop.
* New `concentrate_N_s(q̂, G_TARGET, N_LO, N_HI) -> (N̂, clamped)` — §3.1 step 1.
* `fast_cloglog_regression`: **keep the `(ρ, g)` row structure and the `not_supply` outcome**,
  keep `log(z_flat[ρ,g])`, return the `log z` coefficient alongside the `N_REG` distance ones as a
  diagnostic, and **delete the
  `include_control`/`include_size_control` exclusion assertion** (`:846`, `:1053`) — control cells
  now have real draws. Rows become `n_cells × N̂_s`.
* `compute_moments`: γ → AA aggregate; new `G0` from realised counts; average over replications.
* `moments_to_vec` / `full_SMM`: append block 6.

### `profiling.jl`

* `invert_T_ge`: target `emp_gamma_aa`; one multiplicative update per active `(s,a)` against the
  **aggregated** model share `Σ_{l∈a} γ_model[l,s]`; ref-normalise on `T_REF_AA`. Square system
  (`|𝒜⁺_s|` equations, `|𝒜⁺_s|` unknowns); contraction inherited from the ZE-level version
  (note App. C).

### `tools.jl`

* `sigma_beta_gamma_filename(; smm, aa)` → `_aa` suffix, default `aa=true`.
* `reconcile_sigma_data`: reconcile at the AA level; `G0` rows never pruned.
* `build_step3_weight_matrix`: `gb_indices = vcat(BLOCK_RANGES[4], BLOCK_RANGES[5], BLOCK_RANGES[6])`.
* `compute_jacobian`: add `hold_N_s` and the "no perturbation moved `N̂_s`" assertion (§3.3).
* `compute_smm_inference`: report `N̂_s` value-only; delta-method SE from the `G_s(0)` rows of Σ is
  a follow-up.
* `generate_report`: the G4 curve plot, both `N_s` routes (§3.4), clamp flags, and the realised
  count distribution.

### `optimizer.jl` / `main.jl` / `run.sh`

* T block sized `n_T_aa`; block-coordinate stages unchanged in structure.
* New flags: `--n_rep` (`R_rep`), `--gamma_level=aa|ze`, `--granular=true|false` (false ⟹ the
  current continuum model, for gate G0).

---

## 5. Validation gates, in implementation order

| Gate | What it checks | Pass criterion |
|---|---|---|
| **G0** | With `--granular=false` (pool width instead of `N̂_s`, outcome reverted) the pipeline reproduces today's `reg_coef` | bitwise-ish; isolates the refactor from the modelling change |
| **G7** | AA map: `argmax_col(closest_downstream_region.npy) == CLOSEST_DOWNSTREAM_REGION`, rows sum to 1, shape `(R, R_d)` | exact, **at load time** — a mismatch invalidates the alignment argument |
| **G1** | `concentrate_N_s` on synthetic `q̂`: `G(s,·)` monotone; bisection recovers a planted integer; clamps fire at the bounds | exact integer recovery |
| **G8** | Closed form vs simulation at `θ̂`: `G(s, N̂_s)` from `q̂` against the realised empty share | agree to MC error; a gap points at a bug in the winner accounting |
| **G6** | **Firm-count mapping (§1)**: distribution of model `log z` (champions) vs the empirical firms' `log z` | report the overlap; poor overlap ⇒ option 2 of §1 is needed before the coefficient can be trusted |
| **G2** | Two routes to `N_s` (§3.4) | same order of magnitude; a large gap is a finding |
| **G4** | Untargeted fit of the whole curve `G_s(K)`, `K ≥ 1` (only `K=0` is targeted) | reported max deviation |
| **G3** | No sector clamped at `N_LO` / `N_HI` at the optimum | `clamped == :none` ∀ s |
| **G5** | Drift of `reg_coef` in `N_s` at `θ̂` | should be ~0: `η` has no `N_s` term at the firm level (D4). A visible drift means the firm↔champion mapping or the union-over-buyers approximation is doing something unintended |
| **G11** | `b_logz ≈ −θ = −1.0`, on **both** simulated and empirical data | the cheapest check on the §1 firm-count mapping; a large gap invalidates the coefficient comparison |
| **G9** | AA-level Sinkhorn: round-trip recovery of a planted `T_aa` from its own AA aggregates | ~1e−8, mirroring `test/test_ge_inversion.jl` |
| **G10** | `α̂` against the §3.9 within-area evidence (`α/η_size ≈ 1.1`, θ=1) and the joint free-`T` search (`α ≈ 0.30`) | the evidence predicts `α̂` moves up; a large gap the other way is a finding, not a bug |

**Phase 0, before anything.** Time one loss evaluation: `n_good` goes ~142 → ~1161 and the
regression goes from `n_good × N_rho` to `n_cells × N̂_s × R_rep` rows. Also compute `N̂_s`'s
plausible range straight from `G_K.csv` and the bounds — it decides the cost and it costs nothing.

Sequence: **Phase 0** → **G0** → AA-level `T` + AA γ + G7/G9 → granularity + `N_s` + G1/G8 → full
estimation + G2–G6, G10.

---

## 6. Data-side actions

**Retracted: do NOT regenerate the β target with the supplier outcome.** I previously asked for
that on the strength of Proposition 1, which is a *cell-level* result. The moment regression is
firm-level, where `not_supply` is the exact form (D4). **The existing β target and the β block of
Σ are correct as they stand** — nothing to regenerate.

Still to fix in the empirical script:

* **`nunique() == 2` is the wrong filter.** It conditions on a *realised* outcome, which is not
  reproducible in the model. Use `groupby("A129_AA")["supplier"].transform("max") == 1` — the same
  `𝒜⁺_s` rule used for `AA_ACTIVE` and for the `G(K)` denominator. Define `𝒜⁺_s` **once**, export
  it, use it everywhere.
* **The outcome must be `not_supply`.** The draft's comment says so but the code regresses
  `supplier_num`. Use `1 − supplier`, and read the distance coefficient as `+αθ`, so `α = β̂/θ`.
* **Report the `log z` coefficient** — it should be near `−θ = −1` (D4). This is the cheapest
  available check on the firm↔champion mapping.
* `cov_type="HC1"` is fine for eyeballing but must not be the source of the β block of Σ — that
  comes from the joint bootstrap that also produces the γ and G blocks and their cross-covariances.
  Cluster at `A129_AA` when reporting.
* Confirm `A129_AA` concatenates **sector × AA**, matching
  `(s−1)*R_downstream + CLOSEST_DOWNSTREAM_REGION[r]`.
* The distance floor and the distance variable — see §1.

---

## 7. Why simulate rather than use the closed form — the comparison

**(a)** simulate `N_s` varieties on realised outcomes; **(b)** estimate `q_ls` from a large pool
and apply `Pr(K = 0) = (1−q)^{N_s}` analytically. Measured on synthetics calibrated to the observed
0.915 empty share, `θα = 0.30`.

### 7.1 The settling argument

Under (b) the estimation **never solves a finite-variety network**: the value block sits at the
certainty-equivalent price index while the extensive margin is granular. That leaves "how do you
solve the economy at `N̂_s`?" unanswered — and that economy is the paper's object, since comovement
is propagation through the *realised* network. The solver must exist regardless, so estimating on a
different economy than the counterfactuals is a consistency liability for no gain, and it needs a
re-mapping step (keep weights `1/N̂_s`, an average not a sum) to avoid reintroducing the
love-of-variety factor.

### 7.2 The bias advantage of (a): 2–4%

(a) matches the binding function `E[β̂(sim)]`, cancelling the auxiliary FE cloglog's finite-sample
bias; (b) matches `β(E[y])`, its `T → ∞` limit. At 1161 cells and data-consistent group sizes:

| config | cells/group | suppliers/group | `β(E[y])` | `E[β̂(realised)]` | gap | MC se |
|---|---|---|---|---|---|---|
| `G = 20` | 58 | 5.0 | −0.3004 | −0.3053 | −0.005 (2%) | 0.008 |
| `G = 30` | 39 | 3.3 | −0.3005 | −0.3139 | −0.013 (4%) | 0.009 |
| `G = 50` | 23 | 2.0 | −0.3002 | −0.3122 | −0.012 (4%) | 0.009 |

Real in direction, modest in size — a reason to prefer (a), not the decisive one. *(An earlier
"15%" came from a synthetic with 0.82 suppliers per group, which `𝒜⁺` cannot produce. An earlier
"100% non-convergence" was an IRLS warm-start artefact, not separation.)*

### 7.3 Variance is a wash

At 20,000 variety-draws for both, estimating `E[G(0)]`: (a) bias −0.00000, sd 0.00029; (b) bias
+0.00007, sd 0.00023. `G(0)` is a mean over hundreds of cells, so averaging over cells already
does most of the variance reduction.

### 7.4 A diagnostic that survives, not an objection

A realised economy can empty an entire retained AA, which the data never does by construction.
This is **not** an argument against (a): `𝒜⁺` is a choice of which areas to study, not a model
prediction to reproduce; within a retained area (a) draws firms in all ZE and lets Ricardian
competition decide; and a degenerate group contributes nothing to `β̂`, which is what the FE
cloglog does on either side. What survives is a diagnostic — the rate is governed by suppliers per
group, directly countable:

| `G` (active AA×sector groups) | cells/group | suppliers/group | P(group comes out empty) |
|---|---|---|---|
| 20 | 58 | 5.0 | 0.7% |
| 30 | 39 | 3.3 | 3.7% |
| 50 | 23 | 2.0 | 14% |
| 99 (extreme: one each) | 12 | 1.0 | 37% |

Count `G` before starting. ≥3 suppliers per group ⇒ a few percent, ignorable. Near 1 ⇒ the model
says the observed configuration is unlikely, which is itself worth knowing.

### 7.5 What (b) keeps

The `N_s` inner loop (D5) and the cross-check at `θ̂` (G8). Profiling on realised economies would
need `R ×` (candidates) regressions per loss evaluation.

---

## 8. What this version deliberately does not do

* No ZE-level deviation `ν_sl` (`σ²_ε = 0`) — the note's §4 relaxation, the negative binomial and
  the frailty attenuation are out of scope.
* No ZE-level γ moments in the loss (the `_aa` Σ is the default).
* No `K ≥ 1` count thresholds in the loss — only `G_s(0)`, since that is what Σ covers. The rest of
  the curve is an untargeted check (G4).
* No love-of-variety channel in the value block (D2) — correct for identification, must be undone
  for counterfactuals.
* No firm count per cell as a model outcome (§1, option 1) — the first extension.
* `𝒜⁺_s` is taken as given, never explained (note §3.2).

---

## 9. Conventions settled

| Item | Answer |
|---|---|
| Σ ordering | β → γ → G, exactly `S` rows of `G_s(0)` |
| `G(K)` | `Pr(K_ls ≤ K)`; `G(0)` = share with 0 suppliers |
| `G(K)` denominator | ZE inside active attraction areas only |
| `N_s^obs` | `N_supplier_s` in `G_K.csv`, repeated per row |
| AA map | `closest_downstream_region.npy`, asserted equal to `CLOSEST_DOWNSTREAM_REGION` (G7) |
| cloglog outcome | **`y = not_supply`** — the exact firm-level form; `b_logd = +θα`, `b_logz = −θ` |
| regression unit | **firm = (cell, variety) champion**, control `log z` |
| `SIREN` | deferred; cell-level, not a substitute for `log z` |
| `θ` | 1.0 |
