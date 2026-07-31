# Implementation plan — granular varieties + comparative advantage at the attraction-area level

Target: the **simple version**. Comparative advantage lives at the attraction-area (AA) ×
sector level only (`σ²_ε = 0`, no ZE-level deviation); the number of varieties `N_s` is finite
and estimated from the supplier-count distribution.

Companion theory: `documentation/finite_sample2.tex` (§3 for the two-tier geography, §5 for the
count moment, §6 for what must change, §3.9 for the empirical diagnostic).

---

## 0. Four design decisions that make this cheap

These are the load-bearing choices. Everything downstream follows from them.

### D1 — ~~Granularity is applied analytically~~ **superseded by §A: draw `N_s` varieties, `R` replications**

> **Superseded.** See §A. The closed-form Binomial is retained for the `N_s` inner loop and as a
> cross-check, but the moments are computed on simulated economies. The text below documents the
> closed-form machinery, which is still used in that reduced role.


`solve_network` already computes exactly the object granularity needs. At `model_CP.jl:720`,
inside the loop over downstream buyers,

```julia
linkages_flat[rho, g_winner] = 1.0
```

so `linkages_flat[ρ, g] ∈ {0,1}` is the indicator *"pair `g` wins variety `ρ` for **at least
one** downstream buyer"* — the win-somewhere event. With the flat weights `1/N_rho` the column
mean is exactly the win-somewhere probability of the note's §5:

```
q[s,l] = mean over ρ of linkages_flat[ρ, g(s,l)]        # = q_ls
```

Granularity then enters in closed form, by Poisson thinning / the Binomial of eq. (14):

```
Pr(K_ls = 0) = (1 - q_ls)^{N_s}                (or exp(-N_s q_ls) under Poissonization)
```

**Do not draw `N_s` varieties.** Drawing ~10² varieties would make every moment carry
granular sampling noise of the same order as the granularity itself, and the loss would be
unusable. Keep `N_rho` large: it stays a pure numerical-integration device for `q_ls`, exactly
as today. Simulation noise → 0 as `N_rho` → ∞ while granularity stays exact. This also means
`N_s` never enters the draw generation, so nothing in `generate_draws` / `U_DRAWS` changes.

### D2 — The value block is untouched, and the love-of-variety term is normalized out for free

`sample_weights[ρ,g] = 1/N_rho` are self-normalized per column, so the within-sector CES index
computed in `solve_network` is an **average**, not a sum over varieties — hence invariant to the
variety count. That is precisely Option 1 of the note's §6(a) (`T_{s,a} ↦ T_{s,a} N_s^{θ/(1-ν_s)}`),
implemented implicitly and at no cost. Consequence: `N_s` affects **only** the extensive-margin
and count moments; every value moment (labor share, `π_s`, `π_r`, the shares themselves) is
numerically unchanged from the baseline at the same `(Ω, A, α, T)`.

> **Caveat to carry into counterfactuals.** The love-of-variety channel is normalized out at the
> calibration point, not removed from the economics. Any counterfactual that moves the set of
> active varieties must undo the normalization first (note §D.4), or the propagation channel is
> silenced.

### D3 — The γ moments move to the **AA level**

A ZE inside an active AA with no supplier has observed `γ̂_ls = 0` and a **bootstrap variance of
zero** — it cannot enter the γ block without giving it infinite weight. Its information belongs
to the extensive margin and the counts, not to the γ block. So block 5 becomes the AA aggregate
`γ_{s,a} = Σ_{l∈a} γ_{ls}`, which is strictly positive on every active AA by construction. This
is what the `..._aa.npy` Σ files are for, and it makes the γ block exactly just-identified
against the AA-level `T` (one moment per active `(s,a)`, minus one reference per sector).

The ZE-level variant (non-`_aa` files) stays reachable behind a flag as the over-identifying
configuration of the note's §6(b) — **not** part of the simple version.

### D4 — `N_s` is **profiled** out over the integers, and the supplier indicator makes it a root-find

`N_s` is estimated by the same weighted objective as everything else, concentrated rather than
searched jointly:

```
N̂(θ_c) = argmin over N ∈ ∏_s [N_LO_s, N_HI_s] ∩ ℕ  of  r(θ_c,N)' W r(θ_c,N)
```

Textbook profiled extremum estimator — identical to joint minimisation over `(θ_c, N)`. The inner
minimisation needs **no re-simulation** (`q̂` is `N_s`-free, D1 and `finite_sample2.tex` §7.3), and
`N̂_s` is an integer at every evaluation, so integrality is never relaxed and never rounded.

**With `y = 1{supplier}` the inner problem collapses to a one-dimensional root-find.** The
cloglog index of the supplier indicator separates *exactly*:

```
η_ls = ln(−ln Pr(K_ls = 0)) = ln N_s + ln(−ln(1 − q_ls))
```

and `ln N_s` is constant within sector, hence **absorbed by the AA×sector fixed effect**. Block 4
therefore carries essentially no information about `N_s`, and the profiled optimum is the value
that zeroes the count residual. Simulation of the exact object (1161 cells, 40 groups, true
`θα = 0.30`):

| `N_s` | empty share | `b_dist` supplier (well spec.) | supplier (misspec.) | `not_supply` |
|---|---|---|---|---|
| 30 | 0.992 | **−0.300127** | −0.3238 | +0.067 |
| 250 | 0.941 | **−0.300119** | −0.3275 | +0.109 |
| 1000 | 0.812 | **−0.300092** | −0.3391 | +0.157 |
| 20000 | 0.170 | **−0.300014** | — | — |
| *drift over range* | | **3.8e−4** | 8.8e−2 | ×2.3 |

So the inner loop is:

```
G(s,n) := mean over l ∈ cells_s of (1 − q̂[s,l])^n          # closed form, ↓ in n
N̂[s]  := integer bisection of G(s,·) onto G_TARGET[s], clamped to [N_LO_s, N_HI_s]
```

with the **full-loss integer line search demoted to a verification step run once at `θ̂`**, not an
inner loop. Two riders:

* the invariance is exact only if the distance term is flexible enough to fit the true trade-cost
  profile — under misspecification ~9% of `N_s`-dependence returns. That argues for the binned
  specification (`N_REG = 4`) over the single power law, and for **measuring** the drift at `θ̂`
  rather than assuming it away;
* `α` becomes **directly readable**: `b_dist = −θα` to four decimals, so `α = −b_dist/θ` with no
  attenuation correction — unlike every other configuration in the note (the LPM factor `λ`, the
  frailty factor, the censoring factor).

---

## 1. The regression is natively one row per cell

The baseline builds the design with one row per (draw, pair) — `(n_good + N_CONTROL) × N_rho ≈
1.1M` rows (`model_CP.jl:1049–1130`). Under granularity that is not merely wasteful, it is the
**wrong unit of observation**.

In the data one observation is a (sector, ZE) cell with `y_ls = 1{l hosts a supplier in s}` —
1161 rows. In the granular model the counterpart of that event is

```
Pr(K_ls ≥ 1) = 1 − (1 − q_ls)^{N_s}
```

**one number per cell.** There is no per-draw outcome to regress: the million-row design was an
artefact of the continuum model, where the aggregated object was a per-*variety* win. The
simulated design now matches the empirical one row for row, which is what makes the moment a
comparison of like with like. A draw-level regressor would have no empirical counterpart and would
make the two regressions incommensurable.

### On "the draws within a cell don't share the same productivity draw"

Correct, and it is exactly why the draw dimension must not be in the regression. Two things follow.

* **Under granularity the objection is moot**: `Pr(K_ls ≥ 1)` has no draw index. The draws serve
  only to estimate `q_ls`.
* **For the `N_s = 1` backward-compatibility gate** (G0), where the design *is* collapsed from
  draws, the collapse is exact iff nothing varies within a cell. Checking what actually varies in
  `fast_cloglog_regression`:

  | Column | Varies within cell? |
  |---|---|
  | distance bin / `log_dist` (from `r`, `dr`) | no — cell-level by construction |
  | FE group `(s-1)*R_downstream + dr` | no |
  | `sample_weights[rho,g]` | no under `:qmc` / `:sobol` / `:mc`; **yes under `:is`** |
  | `log(z_flat[rho,g])` (size control) | **yes** — but only when `include_size_control = !include_control`, i.e. off on the production path |

  So on the production path (`--controls=true`, `--draws=qmc`) the collapse is exact — verified to
  `1.3e-15` on a synthetic design (30,000 rows vs 60). Under `:is`, or with the firm-level `log z`
  control on, it is not.

The `log z` control should be dropped regardless of the collapse: it is the *winning draw's
productivity*, whereas the data condition on a **cell-level firm count** (`log SIREN`). They are
different variables, so matching a coefficient on one against the other was never comparing like
with like. The control the specification actually needs (§3.9 of the note) is cell-level — so the
two requirements coincide.

## 2. Data inputs

### Supplied, with the conventions now settled

| File | Convention (confirmed) |
|---|---|
| `G_K.csv` — `A129`, `K`, `G(K)`, `N_supplier_s` | `G(0)` = share of ZE with **0** suppliers; `G(1)` = share with **at most 1**; i.e. `G(K) = Pr(K_ls ≤ K)`. Denominator = **ZE inside active attraction areas only** (the 1161 cells of §3.9). `N_supplier_s` = total suppliers in the sector, **repeated on every row** — read once per `A129` and assert constant within sector. |
| `Sigma_beta_gamma_cloglog[_1]_f_aa.npy` | Ordering **β → γ(AA) → G**, with exactly **`S`** `G_s(0)` rows. **The simple version uses this file.** |
| `Sigma_beta_gamma_cloglog[_1]_f.npy` | Same, γ at **ZE** level — the over-identifying variant, behind a flag, not part of this version. |
| `closest_downstream_region.npy` | `R × R_downstream` **binary** incidence matrix: which upstream ZE belongs to which attraction area. |

Derived immediately:

```
N_HI[s] = N_supplier_s                              # every variety won by one origin
N_LO[s] = ceil(N_supplier_s / R_downstream)         # every variety won by a different origin per buyer
G_TARGET[s] = G_K.csv value at K = 0
```

### The AA map must be verified, not assumed

`closest_downstream_region.npy` is an incidence matrix; the code already computes the same
partition internally as `CLOSEST_DOWNSTREAM_REGION` (`load_parameters.jl:507`), which is what
builds today's fixed-effect group. **Add an explicit test** (new `test/test_aa_map.jl`, plus a
startup `@assert` in `load_parameters.jl`) checking that

```
argmax over columns of closest_downstream_region.npy[l, :]  ==  CLOSEST_DOWNSTREAM_REGION[l]   ∀ l
```

together with: each row sums to exactly 1 (a ZE belongs to exactly one AA), the matrix is
`R × R_downstream`, and every AA index is in range. A mismatch means the model's fixed effect and
the empirical one are different partitions, which silently invalidates the whole alignment
argument of `finite_sample2.tex` §2.2 — so it must fail loudly at load time, not be discovered
later.

### Deferred to a later version (recorded here so it is not lost)

**`SIREN` per (sector, ZE) — the firm count — is NOT in this version.** Consequence to keep in
mind: §3.9 showed the size control is the dominant within-area force and that omitting it inflates
the distance coefficient in magnitude (−0.284 → −0.130 in logs, −0.017 → −0.011 in levels once it
is added). So **`α̂` from this version is expected to be biased upward in magnitude**, and the
`α/η_size ≈ 1.1` reading of §3.9 is the target to compare against once `SIREN` arrives. When it
does, the changes are small and localized:

* a new `SIREN_LOG[s,l]` array loaded alongside `filter_N_upstream`;
* one extra column in the cell-level design of `cloglog_cell_regression`;
* the loading `η_size` becomes a scalar parameter appended to `θ` (or fixed at 1, but §3.9 argues
  against assuming it);
* the collapse of §1 stays exact, because `log SIREN` is a **cell-level** covariate — unlike the
  draw-level `log z` it replaces.

## 3. Moment vector layout

Blocks 1–4 keep their roles. Block 5 changes level; block 6 is new.

| # | Block | Before | After |
|---|---|---|---|
| 1 | `agg_labor_share` | 1 | unchanged |
| 2 | `agg_industry_share` | S−1 | unchanged |
| 3 | `pi_r` | R_d−1 | unchanged |
| 4 | `reg_coef` | `N_REG` | unchanged size; **granular** outcome, **cell-level** design, outcome flipped to `1{supplier}` |
| 5 | `gamma_ls` | active (s,ZE) − ref | **active (s,AA) − ref** |
| 6 | `G0` | — | **new**: `G_s(0)`, one per sector with an active AA |

**Assumed Σ ordering: β → γ → G**, i.e. block 6 appended last, matching the existing
`gb_indices = vcat(BLOCK_RANGES[4], BLOCK_RANGES[5])` invariant extended by `BLOCK_RANGES[6]`.
*If the supplied Σ files order them differently, this is a one-line change in `gb_indices` and in
`reconcile_sigma_data` — see the questions in §8.*

The invariant "β first, then γ" documented in `CLAUDE.md` is preserved and extended, not
reversed.

---

## 4. The estimation algorithm

### 4.1 Pseudo-code

Shapes are given for every object. `R` = upstream ZE, `R_d` = downstream regions (= number of
attraction areas), `S` = sectors, `n_cells` = |CELL_MASK| ≈ 1161.

```text
════════════════════════════════════════════════════════════════════════════════
PRECOMPUTED ONCE   (load_parameters.jl)
════════════════════════════════════════════════════════════════════════════════
AA_OF_ZE          :: Vector{Int}(R)          l ↦ a(l);  from closest_downstream_region.npy
                                             @assert AA_OF_ZE == CLOSEST_DOWNSTREAM_REGION  (G7)
AA_ACTIVE         :: BitMatrix(S, R_d)       a hosts ≥1 observed supplier in s   → 𝒜⁺_s
CELL_MASK         :: BitMatrix(S, R)         has_firms[s,l] && AA_ACTIVE[s, AA_OF_ZE[l]]
                                             has_firms = filter_N_upstream ∈ {1,2}
CELLS_OF_SECTOR   :: Vector{Vector{Int}}(S)  l-indices of CELL_MASK[s,:]
GOOD_S,GOOD_R,SR_TO_GOOD                     rebuilt from CELL_MASK  (n_good ≈ n_cells)
T_MASK_AA         :: BitMatrix(S, R_d)       = AA_ACTIVE          → the estimated T block
T_REF_AA          :: Vector{Int}(S)          one reference AA per sector, T[s,ref] ≡ 1
emp_gamma_aa      :: Matrix(R_d, S)          Σ_{l∈a} emp_gamma_ls[l,s]        → block-5 target
G_TARGET          :: Vector{Float64}(S)      G_K.csv at K=0                   → block-6 target
G_CURVE           :: Vector{Vector}(S)       full G_K.csv curve               → gate G3 only
N_LO, N_HI        :: Vector{Int}(S)          ceil(N_supplier_s / R_d),  N_supplier_s
LOG_DIST          :: Vector(R)               log(max(closest_plant_dist, 1.0))   ← the 1.0 floor
DIST_BIN          :: Vector{Int}(R)          distance_bin(...) ∈ 0:N_REG, 0 = reference (d ≤ 50)
FE_GROUP          :: Vector{Int}(n_cells)    (s-1)*R_d + AA_OF_ZE[l]

════════════════════════════════════════════════════════════════════════════════
PARAMETER VECTOR      (N_s is NOT in it)
════════════════════════════════════════════════════════════════════════════════
θ_c = [ Ω^L(1) | Ω^s(S) | A(R_d) | α(N_TAU) | T_aa(n_T_aa) ],   n_T_aa = count(T_MASK_AA)

════════════════════════════════════════════════════════════════════════════════
ONE LOSS EVALUATION
════════════════════════════════════════════════════════════════════════════════
function loss(θ_c, W; N_prev = nothing)

  ── 0. unpack, gather AA → ZE ────────────────────────────────────────────────
  Ω^L, Ω^s, A, α, T_aa = unpack_params(θ_c)        # T_aa :: Matrix(S, R_d), ref-normalised
  T[s,l] = T_aa[s, AA_OF_ZE[l]]                    # (S,R) gather — the one structural new line

  ── 1. solve the network (UNCHANGED) ─────────────────────────────────────────
  net = solve_network(Ω^L, Ω^s, A, α, T; u_draws = U_DRAWS)
  #   net.linkages_flat :: (N_rho, n_good), 1 if the pair wins variety ρ for ≥1 buyer

  ── 2. win-somewhere probability, per cell ───────────────────────────────────
  for (s,l) in CELL_MASK
      g       = SR_TO_GOOD[s,l]
      q̂[s,l]  = mean_ρ net.linkages_flat[ρ, g]                    # ∈ [0,1]
      n_win[s,l] = q̂[s,l] * N_rho                                  # ← diagnostic, see §4.5
      q̂[s,l]  = clamp(q̂[s,l], 0.5/N_rho, 1 - 1e-12)                # floor: unresolved ⇒ MC limit
  end
  n_floored = count(cells at the floor)                            # ← report every evaluation

  ── 3. profile out N_s, per sector — closed form, NO re-simulation ───────────
  #   G(s,n) = mean over l ∈ CELLS_OF_SECTOR[s] of (1 - q̂[s,l])^n     strictly ↓ in n
  for s in 1:S
      lo, hi = N_LO[s], N_HI[s]
      if     G(s,lo) <= G_TARGET[s]   N̂[s], clamped[s] = lo, :lo    # too few empties even at lo
      elseif G(s,hi) >= G_TARGET[s]   N̂[s], clamped[s] = hi, :hi    # too many empties even at hi
      else
          while hi - lo > 1                                        # G(lo) > target > G(hi)
              mid = (lo + hi) ÷ 2
              G(s,mid) > G_TARGET[s] ? (lo = mid) : (hi = mid)
          end
          N̂[s]      = argmin_{n ∈ {lo,hi}} |G(s,n) - G_TARGET[s]|
          clamped[s]= :none
      end
  end
  #   cost: ⌈log2(N_HI-N_LO)⌉ ≈ 12–15 evaluations of a length-|CELLS_OF_SECTOR[s]| mean

  ── 4. block 4 — cell-level cloglog on the SUPPLIER indicator ────────────────
  p1[s,l] = 1 - (1 - q̂[s,l])^N̂[s]                  # Pr(K_ls ≥ 1) ∈ (0,1), fractional outcome
  X       = N_REG == 1 ? [LOG_DIST[l]]                                  # (n_cells, 1)
                       : onehot(DIST_BIN[l], 1:N_REG)                   # (n_cells, N_REG), base = bin 0
  #   NO draw-level column.  log(SIREN) joins X here when it arrives — and must then
  #   be in the empirical target too (§2), or the two coefficients are different estimands.
  reg_coef = cloglog_irls(y = p1, X = X, fe = FE_GROUP, w = ones(n_cells))   # (N_REG,)

  ── 5. block 5 — AA-level shares ─────────────────────────────────────────────
  γ[s,l]     from compute_moments(net, θ_c)          # (R,S) as today
  γ_aa[a,s]  = Σ_{l : AA_OF_ZE[l] == a} γ[l,s]       # (R_d, S), masked by AA_ACTIVE minus ref

  ── 6. block 6 ───────────────────────────────────────────────────────────────
  G0[s] = G(s, N̂[s])                                 # residual ≈ 0 by step 3

  ── 7. assemble ──────────────────────────────────────────────────────────────
  m = vcat(labor, industry, π_r, reg_coef, γ_aa[mask], G0)
  return (m .- m̂)' * W * (m .- m̂),  (N̂, clamped, n_floored, n_win)
end

════════════════════════════════════════════════════════════════════════════════
ONCE AT θ̂ — verification, not an inner loop
════════════════════════════════════════════════════════════════════════════════
#   Step 3 zeroes the block-6 residual. That is the profiled optimum only to the extent
#   that block 4 is N_s-invariant (D4: 4e-4 well specified, ~9% misspecified). Confirm:
for s in 1:S
    for n in {N̂[s]-δ … N̂[s]+δ} ∩ [N_LO[s], N_HI[s]]      # δ ≈ 10% of N̂[s]
        record full weighted loss with N̂[s] := n          # re-uses q̂; no re-simulation
    end
end
report argmin vs N̂;  a gap ⇒ the block-4 coupling is material ⇒ promote step 3 to a
                      coordinate-descent line search on the full loss
```

### 4.2 Where `N_s` enters the loss

With the supplier indicator, essentially only **block 6**. `ln N_s` sits inside the AA×sector
fixed effect of block 4 (D4), so `reg_coef` is invariant to it to `4e−4` under correct
specification. That is what makes the inner problem a root-find rather than a search, and what
gives the clean division of labor: `α` from the distance gradient in block 4, `N_s` from the
level in block 6. The residual coupling under misspecification (~9%) is why the full-loss line
search is still run once at `θ̂` as a verification.

Because the profiling runs *inside* `loss`, the FD Jacobian automatically picks up the total
derivative `dm/dα` along the `N̂_s(α)` manifold — the same profiled-Jacobian structure the
existing `profile_T` machinery already handles.

### 4.3 Two independent closed-form routes to `N_s` (a free over-identifying test)

The model's expected total number of distinct supplier firms is `E[Σ_l K_ls] = N_s · Σ_l q_ls`.
Matching the observed count gives a second, non-iterative estimate:

```
N_s^count = N_s^obs / Σ_{l ∈ cells_s} q̂[s,l]
```

This identity is exactly what generates the bounds of note §5 (`Σ_l q_ls ∈ [1, min(N_d, R)]` ⟹
`N_s^count ∈ [N^obs/N_d, N^obs]`), so it is admissible by construction. Comparing `N̂_s` (from
`G_s(0)`) with `N_s^count` is a genuine test of the granular mechanism, free of any extra
simulation. Print both every report.

### 4.4 Integrality and the bounds — read the clamp as a diagnostic

`N̂_s` is an integer at every evaluation, so integrality is never relaxed and never rounded.
Clamping, however, is **informative rather than benign**:

* `clamped = :hi` — even the maximum admissible count (`N_supplier_s`, every variety won by a
  single origin) leaves too many empty ZE: the model cannot generate enough sparsity.
* `clamped = :lo` — even the minimum admissible count (`N_supplier_s / R_downstream`) produces
  too few empty ZE.

Either is a **rejection signal for the mechanism**, not a numerical nuisance. Log it per sector
per report and treat a persistently clamped sector as a finding.

One caution on the discreteness: `N̂_s(θ)` is a step function of the continuous parameters, so the
profiled loss is **piecewise smooth with jumps at the switch points**. Two consequences. (i) PSO /
TikTak are unaffected (both derivative-free). (ii) The FD Jacobian can straddle a jump.
Mitigation: at the reported estimate recompute the Jacobian **holding `N̂_s` fixed** at its value
at `θ̂` — the correct object anyway, since `N̂_s` is integer-valued and locally constant with
probability one — and assert that no FD perturbation changed `N̂_s`, reporting it if one did.

### 4.5 Resolving `q_ls` in the tail — now only for the `N_s` inner loop

> **Scope reduced by §A.** With the moments computed on simulated economies there is no `q̂` in the
> moment path, so the precision requirement below applies only to the closed-form `N_s` bisection,
> where a monotone root-find tolerates far more noise. Kept because the analytic control variate is
> still the right way to compute `q̂` cheaply.


This is the largest practical risk in the whole design, and it should be measured in Phase 0
before anything is built.

**The problem.** The cloglog index of the supplier indicator is `η_ls = ln N_s + ln(−ln(1−q_ls))
≈ ln(N_s q_ls)`, so

```
∂η / ∂ln q = 1
```

— a relative error in `q̂` passes into the index **one for one**. And the granular regime forces
`q_ls` to be small: to reproduce an empty share of 0.915 the model needs `N_s q_ls ≈ −ln(0.915)
= 0.089` for a typical cell, and `N_s q_ls ≳ 1` for the active ones. So the model must resolve
`q` down to order `1/N_s`, while `N_rho` draws give roughly `N_rho · q ≈ N_rho / N_s` wins per
active cell.

**With `N_rho = 1000` (the current value, `load_parameters.jl:58`) this is nowhere near enough:**

| `N_s` | typical `q` | wins per cell @ `N_rho`=1000 | rel. SE of `q̂` | @ 10⁵ | rel. SE |
|---|---|---|---|---|---|
| 50 | 1.8e−3 | 1.8 | 75% | 178 | 7.5% |
| 250 | 3.6e−4 | 0.36 | 168% | 36 | 17% |
| 500 | 1.8e−4 | 0.18 | 236% | 18 | 24% |
| 1000 | 8.9e−5 | 0.09 | 334% | 8.9 | 34% |

A 75% relative error in `q̂` is a 0.75 error in `η`, against a distance coefficient of ≈ −0.3
spread over ~1.2 of `ln d`. The moment would be pure noise. Worse, the marginal cells — the ones
that decide the extensive margin — sit *below* the typical `q`, so they are resolved worst, and
many will return `q̂ = 0` exactly.

Note this is not a wholly new problem: the reg_coef Jacobian was already the noisiest column, and
`:is` exists in the codebase precisely "for reg_coef tail resolution". Granularity sharpens it,
because `(1−q)^{N_s}` amplifies small-`q` error.

**Mitigation, in order of value.**

1. **Compute the dominant term of `q_ls` analytically (recommended).** For a *single* downstream
   buyer the win probability is the closed-form EK share `γ_lrs = T_sl (w_l τ_lrs)^{−θ} / Φ_sr` —
   **zero simulation noise**, and `gamma_ls_analytical` in `model_analytical.jl` already computes
   it. Only the *union* over buyers needs simulating. Since each ZE's nearest downstream buyer
   `r* = a(l)` dominates, decompose
   ```
   q_ls  =  γ_{l,r*,s}                      ← analytic, exact
          + Δ_ls,   Δ_ls = Pr(wins somewhere but NOT at r*)   ← simulated
   ```
   `Δ_ls ≪ γ_{l,r*,s}`, and what matters for `q̂` is `Δ`'s *absolute* error, not its relative
   error — so a modest `N_rho` suffices for it. This is a control variate, and it removes
   essentially all of the noise in the table above. **Do this before raising `N_rho`.**
2. **Raise `N_rho`.** Linear cost, and it must rise roughly proportionally to `N_s` to hold
   precision fixed. Useful as a check on (1), expensive as the primary fix.
3. **Importance sampling for `q̂` specifically.** The bias documented for `:is` concerns the CES
   price index, where only the winning column's weight is applied so the losing columns' density
   ratios fail to cancel. The win *indicator* is not that kind of functional: `q = E[w_ρ · 1{l
   wins}]` is a properly weighted estimator and `:is` is unbiased for it. So `:is` may legitimately
   be used for the `q̂` estimate even where it must not be used for the value block — but this
   requires the two to be computed from different draw sets, which is a real complication.

**Phase-0 measurement, before any implementation.** Run the current model once and report the
distribution across cells of `n_win = q̂ · N_rho`: the median, the 10th percentile, and the count
of cells with `q̂ = 0`. If the 10th percentile is below ~30 wins, mitigation (1) is not optional.
Report `n_floored` at every evaluation thereafter.

---

## 5. File-by-file change list

### `load_parameters.jl`

* **SECTION 4 (active set) — the biggest change.**
  `active_mat` currently `(filter_N_upstream .== 1) .& (X_rs .> 0)`. Now build three objects:
  * `has_firms[s,l] = filter_N_upstream[s,l] ∈ {1,2}` (status 0 = no firms, stays out — consistent
    with the diagnostic, where `log(SIREN)` was defined on all 1161 cells);
  * `AA_ACTIVE[s,a] = any(l ∈ a : filter==1 && X_rs>0)` → `𝒜⁺_s`;
  * `CELL_MASK[s,l] = has_firms[s,l] && AA_ACTIVE[s, AA_OF_ZE[l]]`.
  Then **`n_good`, `GOOD_S`, `GOOD_R`, `SR_TO_GOOD`, `SECTOR_GOOD_INDICES` are built from
  `CELL_MASK`, not from `active_mat`** — the former control (status-2) cells become goods, i.e.
  candidate origins in `solve_network` with `T = T_{s,a} > 0`. This *is* the endogenization.
  `CONTROL_S/CONTROL_R/N_CONTROL` become vestigial (their rows are now regular cells);
  `INCLUDE_CONTROL` loses its meaning and should be removed from the cloglog path.
* New: `AA_OF_ZE`, `n_AA`, `AA_CELLS[a]`, `T_MASK_AA`, `T_REF_AA`, `CELL_MASK`, `CELLS_OF_SECTOR[s]`,
  `SIREN_LOG`, `N_LO`, `N_HI`, `G_TARGET`, `G_CURVE`.
* SECTION 6: γ target → `emp_gamma_aa`. Keep `emp_gamma_ls` (needed for the ZE-level variant and
  for the §3.9 diagnostic).
* SECTION 8: extend `compute_block_ranges` to six blocks; `BLOCK_NAMES += ("G0",)`; MOMENT_MASK
  gains the block-6 entries and block 5 is now over `(s,AA)` minus one reference AA per sector.
* SECTION 12: labels — `T[sname-aaname]`, `gamma[sname-aaname]`, `G0[sname]`.
* T init: `invert_T_from_gamma` now inverts to AA level (aggregate the target first).

### `model_CP.jl`

* `unpack_params`: `T_reduced` scatters into `(S, n_AA)` s-major, reference-normalize per sector
  with `T_REF_AA`, then **gather** `T[s,l] = T_aa[s, AA_OF_ZE[l]]` before returning the `(S,R)`
  flat vector. Keep the `eltype(params)` generic array so the AD path survives.
* `solve_network`: **no logic change.** It simply receives ~8× more goods.
* New `winsomewhere_prob(net) -> q` (column means of `linkages_flat` over `CELL_MASK`).
* New `concentrate_N_s(q, G_TARGET, N_LO, N_HI) -> (N̂, clamped)` — §4.1 step 2.
* New `cloglog_cell_regression(p0, siren_log; ...)` — one row per cell, `y = p0`, weight 1, FE
  `(s, AA_OF_ZE[l])`, regressors distance + `log SIREN`. **Reuses `_cloglog_irls` unchanged** (it
  already takes a raw design). Delete the draw-level builder and the control-row append block.
* `compute_moments`: γ → AA aggregate; `reg_coef` via the new cell regression; new sixth return
  field `G0`. Signature gains the concentrated `N̂` (or computes it internally).
* `moments_to_vec` / `full_SMM`: append block 6 to both the tuple and the `vcat`.

### `profiling.jl`

* `invert_T_ge`: target becomes `emp_gamma_aa`; the multiplicative update runs once per active
  `(s,a)` against the **aggregated** model share `Σ_{l∈a} γ_model[l,s]`; ref-normalize per sector
  on `T_REF_AA`. The system is square (`|𝒜⁺_s|` equations, `|𝒜⁺_s|` unknowns) and its contraction
  is inherited from the ZE-level version (note App. C).
* `gamma_ls_analytical`: unchanged, but must receive the gathered ZE-level `T`.

### `tools.jl`

* `sigma_beta_gamma_filename(; smm, aa)` → append `_aa` when `aa=true`. Default `aa=true` for the
  simple version.
* `reconcile_sigma_data`: the active-set reconciliation moves to the AA level (`AA_ACTIVE` replaces
  the `T_MASK_MOMENT` logic), plus the new `G0` rows are never pruned.
* `build_step3_weight_matrix`: `gb_indices = vcat(BLOCK_RANGES[4], BLOCK_RANGES[5], BLOCK_RANGES[6])`.
* `compute_jacobian`: mechanically unchanged; add the "`N̂_s` unchanged under perturbation" check of
  §4.4 and a `hold_N_s` kwarg.
* `compute_smm_inference`: report `N̂_s` in the value-only block (the machinery for non-inferred
  parameters already exists). Delta-method SE for `N̂_s` from the `G_s(0)` rows of Σ is a follow-up.
* `generate_report`: add the G3 plot (simulated `G_s(K)` curve vs `G_K.csv`), the two `N_s` routes
  of §4.3, and the clamp flags.

### `optimizer.jl`

* Bounds/warm start for the T block now sized `n_T_aa` instead of `n_T`. The block-coordinate
  sub-stages are unchanged in structure (`"T"` now means the AA block).
* Nothing about `N_s` — it is concentrated inside the objective.

### `main.jl` / `run.sh`

* New flags: `--gamma_level=aa|ze` (selects block 5 and the Σ file, default `aa`),
  `--granular=true|false` (default true; false ⟹ `N_s ≡ 1`, i.e. the current continuum model —
  the G0 regression gate).

---

## 6. Validation gates, in implementation order

| Gate | What it checks | Pass criterion |
|---|---|---|
| **G0** | The regression refactor alone, before granularity. Cell-level cloglog with `y = 1-q` (i.e. `N_s = 1`) vs today's draw-level `reg_coef`, production path (no draw-level size control). | agree to ~1e-10; §1 proves they are algebraically identical |
| **G1** | `concentrate_N_s` on synthetic `q`: `G(n)` monotone decreasing; bisection recovers a planted integer `N_s`; clamps fire correctly at the bounds | exact integer recovery |
| **G2** | The two routes of §4.3 at `θ̂`: `N̂_s` from `G_s(0)` vs `N_s^count` from the total firm count | same order of magnitude; a large gap is a mechanism finding, report it |
| **G3** | Untargeted fit of the whole curve: simulated `G_s(K)` vs `G_K.csv` for `K ≥ 1` (only `K=0` is targeted) | visual + reported max deviation |
| **G4** | No sector clamped at `N_LO` / `N_HI` at the optimum | `clamped == :none` ∀ s |
| **G5** | `α̂` vs the §3.9 within-area evidence (`α/η_size ≈ 1.1`, θ=1) and vs the joint free-`T` search (`α ≈ 0.30`) | the §3.9 evidence predicts `α̂` moves **up** relative to the joint free-`T` search; a large gap the other way is a finding to report, not a bug to chase |
| **G6** | AA-level Sinkhorn: round-trip recovery of a planted `T_aa` from its own AA aggregates | ~1e-8, mirroring `test/test_ge_inversion.jl` |
| **G7** | The AA map: `argmax_col(closest_downstream_region.npy) == CLOSEST_DOWNSTREAM_REGION`, rows sum to 1, shape `R × R_downstream` | exact; **run at load time**, not only in tests — a mismatch silently invalidates the alignment argument |
| **G10** | `Δ = E[β̂(sim)] − β(E[y])` at `θ̂`, design (a) with `R ≈ 1000` (§A.5) | report it; 15% in the calibrated synthetic, so do not assume it away |
| **G9** | Phase 0, before anything: distribution of `n_win = q̂·N_rho` across cells — median, p10, count of `q̂ = 0` | p10 ≥ ~30 wins; below that, the analytic-dominant-term mitigation of §4.5 is mandatory, not optional |
| **G8** | D1 itself: simulate `R` economies with exactly `N̂_s` varieties, average their moments, compare against the closed-form values | agree to `O(1/√R)`; turns the Rao–Blackwellization argument into a test |

**Phase 0, before any of this: measure the cost.** `n_good` goes from ~142 to ~1161, so
`solve_network`'s per-sector price loops grow ~8×, while the regression design shrinks from ~1.1M
rows to ~1161. The net effect on one loss evaluation is not predictable a priori — time it, and if
`solve_network` dominates, `N_rho` can be cut substantially now that the noisiest moment channel
(`reg_coef`) has become deterministic given the draws.

Suggested sequence: **Phase 0** (timing) → **G0** (refactor, zero behavioural change) → AA-level
`T` + AA γ + G6 → granularity + `N_s` concentration + G1 → full estimation + G2–G5.

---

## 7. What this version deliberately does **not** do

* No ZE-level deviation `ν_sl` (`σ²_ε = 0`) — the note's §4 relaxation, and with it the negative
  binomial and the frailty attenuation, are out of scope here.
* No ZE-level γ moments in the loss (the `_aa` Σ is the default) — over-identification is a later
  switch.
* No `K ≥ 1` count thresholds in the loss — only `G_s(0)`, since that is what Σ covers. The rest
  of the curve is an untargeted check (G3).
* No love-of-variety channel in the value block (D2) — correct for identification, must be undone
  for counterfactuals.
* `𝒜⁺_s` is taken as given, never explained (note §3.2).

---

## A. Design (a) "simulate `N_s` varieties" vs design (b) "`q̂` + closed form"

**(a)** — put `N_s` in the parameter vector, draw firms in **every** ZE of every retained
attraction area, run the Ricardian competition over `N_s` varieties, compute `G` and the
regression on the **realised 0/1** outcomes.
**(b)** — estimate `q_ls` from a large draw pool and apply `Pr(K_ls = 0) = (1−q_ls)^{N_s}` in
closed form.

**Conclusion: build (a).** The reasoning below corrects two numbers I previously reported from a
mis-calibrated synthetic, and adds the argument that settles it.

### A.1 The settling argument: you need the finite-variety solver anyway

Under (b) the estimation **never solves a finite-variety network**. It solves the continuum
network (`N_rho` draws, self-normalised weights) to get `q_ls`, then layers the Binomial on top.
The value block is therefore evaluated at the certainty-equivalent price index while the extensive
margin is granular — a coherent hybrid (the note's expected-price-index GE), but a hybrid.

That leaves the question unanswered: **once `N̂_s` is estimated, how do you solve the economy with
a finite number of varieties?** You cannot avoid it, because that economy is the paper's actual
object. Spatial comovement is about how a shock propagates through the *realised* network, and it
is the granular extensive margin — suppliers appearing and disappearing — that makes the
propagation interesting. A continuum network has a degenerate extensive margin, which is the
problem the whole note exists to fix.

So (a)'s machinery must be built regardless. Given that, running the estimation on a *different*
economy than the counterfactuals is a consistency liability for no gain. Two economies also mean a
re-mapping step: `T̂` is identified under the self-normalised (`1/N_rho`) price index, so a finite
solve must keep weights `1/N̂_s` — an average, not a sum — or it silently reintroduces the
love-of-variety factor `N_s^{1/(1−ν_s)}` and moves the calibration point (note App. D).

### A.2 Correction: the bias advantage of (a) is 2–4%, not 15%

The incidental-parameters argument is right — (a) matches the binding function `E[β̂(sim)]`, which
cancels the finite-sample bias of the auxiliary FE cloglog, while (b) matches `β(E[y])`, its
`T → ∞` limit. But my earlier "15%" came from a synthetic with **0.82 suppliers per fixed-effect
group**, which the data cannot produce: `𝒜⁺` guarantees ≥1 supplier per group, and with 99
suppliers over `G` groups the ratio is `99/G ≥ 1`. Re-measured at 1161 cells and data-consistent
group sizes, `θα = 0.30`:

| config | cells/group | suppliers/group | `β(E[y])` | `E[β̂(realised)]` | gap | MC se |
|---|---|---|---|---|---|---|
| `G = 20` | 58 | 5.0 | −0.3004 | −0.3053 | −0.005 (2%) | 0.008 |
| `G = 30` | 39 | 3.3 | −0.3005 | −0.3139 | −0.013 (4%) | 0.009 |
| `G = 50` | 23 | 2.0 | −0.3002 | −0.3122 | −0.012 (4%) | 0.009 |

So the bias is **2–4% of `β`, around 1.5 MC standard errors** — real in direction, modest in size.
It is a reason to prefer (a), not the decisive one.

*(An intermediate run reporting 100% non-convergence of the realised-economy regression was an
artefact of an IRLS warm start `η = log(−log(1−y))` applied to 0/1 data. With a constant start and
deviance step halving, 0% fail and there is no separation problem.)*

### A.3 Retraction: the "support mismatch" objection does not stand

I previously argued that (a) is inconsistent with conditioning on `𝒜⁺` because a realised economy
empties 59% of the retained groups. That figure came from the same mis-calibrated synthetic. The
objection was also wrong in substance:

* `𝒜⁺` is a **choice of which attraction areas to study**, not a model prediction to be
  reproduced. Within a retained area, (a) draws firms in **all** ZE — with and without observed
  suppliers — and lets the Ricardian competition decide which host one. That *is* the
  endogenisation the two-tier design is for.
* When a simulated group does come out empty it contributes nothing to `β̂`, which is exactly what
  the FE cloglog does to any degenerate group. Indirect inference requires the **same function**
  applied to data and simulation, not the same realised support — so this is an efficiency
  question, not a bias.

What survives is a **diagnostic**, not an argument. The rate at which retained groups come out
empty is governed by suppliers per group, which is directly countable in the data:

| `G` (active AA×sector groups) | cells/group | suppliers/group | P(group comes out empty) |
|---|---|---|---|
| 20 | 58 | 5.0 | 0.7% |
| 30 | 39 | 3.3 | 3.7% |
| 50 | 23 | 2.0 | 14% |
| 99 (the extreme: one supplier each) | 12 | 1.0 | 37% |

**Count `G` before starting.** If suppliers per group is ≥3 the effect is a few percent and can be
ignored. If it is near 1, the model is saying the observed configuration is unlikely, which is
itself worth knowing.

### A.4 What (b) keeps: the cheap inner loop

`G(s,n) = mean_l (1−q̂_l)^n` is closed form, monotone in `n`, and instant for any candidate.
Profiling `N_s` on realised economies would need `R ×` (number of candidates) IRLS fits per loss
evaluation. So keep (b) **for the `N_s` root-find**: locate `N̂_s` with the closed form, then
evaluate the loss with the simulated economies at that `N̂_s`. Also keep the closed form as a
cross-check at `θ̂` (gate G8): the simulated `G(s, N̂_s)` and the closed-form one must agree to
Monte-Carlo error, and a gap points at a bug in the winner accounting.

### A.5 What (a) must not do

**Never set `N_rho = N_s` for the whole simulation.** The value block is a numerical integral; the
variety count is a structural parameter. At `N_s ≈ 110` the labor share, `π_s`, `π_r` and the
shares would be computed from 110 draws. Use **`R` replications of `N_s` varieties**: granularity
stays exactly `N_s` within a replication, while the value block averages over all `R × N_s` draws.

Sizing `R`: the dispersion of `β̂` across economies is ~0.16–0.18 in the configurations above, so
`R ≈ 300` gives an MC standard error of ~0.010 on the block-4 moment. At `N̂_s ≈ 110` that is
~33,000 variety-draws against today's `N_rho = 1000` — roughly 33× the current simulation cost.
Common random numbers (a fixed draw pool, with `N_s` taken as a prefix) make most of that noise
difference out across `θ`, which is what keeps the optimiser usable.

### A.6 Consequences for the plan

* **D1 flips.** Draw `N̂_s` varieties per replication, `R` replications; the closed-form Binomial
  is demoted from the moment computation to the `N_s` inner loop and the cross-check.
* **§4.5 (tail resolution of `q̂`) largely evaporates.** It was an artefact of (b)'s two-step: in
  (a) there is no `q̂`, and a cell with a tiny win probability simply wins zero varieties — the
  right answer with the right probability. `q̂` still appears in the `N_s` root-find, where only
  moderate accuracy is needed since the bisection is on a monotone function.
* **D4 stands**, with the loss evaluated on simulated economies: profile `N_s` by the closed-form
  bisection, verify once at `θ̂` with the full loss.

---

## 8bis. Review of the empirical-target script

The script that produces the β target must run **the same regression, on the same unit, with the
same support and the same regressors** as the model, or the moment compares different estimands.
Findings on the draft, ordered by severity.

### Blocking

**(1) Unit of observation: firm-level has no model counterpart.** The draft is at firm level
(`df : niveau firme`, one row per firm, `log_X = log(eff_3112)`). The model's object is
`Pr(K_ls ≥ 1)` — a **(sector, ZE) cell**. A firm-level regression asks "does firm *i* supply",
whose model probability is *zero*: the model has a continuum of firms per (ZE, variety) and only
champions are observable, so non-supplier firms are not model objects at all.

The inconsistency is provable rather than a matter of taste. Reading the model at firm level
forces the fraction of supplying firms in a cell to be `q_ls`, while the granular model says the
supplier count is `E[K_ls] = N_s q_ls` out of `N_ls` observed firms — so `N_ls = N_s` for every
cell, i.e. every ZE would have to contain exactly the sector's variety count. Firm counts vary by
orders of magnitude across ZE, so that is false.

**Use the cell-level dataset** — the same 1161-row frame as the §3.9 diagnostic: one row per
(sector, ZE) inside an active AA, `y = 1{cell hosts ≥ 1 supplier}`. This is also the unit of
`G_K.csv`, so all three moments (β, γ, G) then share one support.

**(2) `nunique() == 2` is the wrong filter, and it is not reproducible in the model.** It keeps
only groups where the outcome takes *both* values — a filter on a **realised random outcome**.
The model's outcome is a probability in (0,1) and is never exactly 0 or 1, so `nunique()` is
undefined on simulated data and the two samples cannot be made to coincide.

The support must be defined by a rule that does not depend on the realisation. Use

```python
keep = df.groupby("A129_AA")["supplier"].transform("max") == 1     # 𝒜⁺_s
```

i.e. attraction areas hosting at least one supplier — the definition already used for `AA_ACTIVE`
and for the `G_K.csv` denominator. Note `nunique()==2` additionally drops all-ones groups; under
cloglog with FE those contribute nothing to `β̂` anyway, so dropping them does not change the
estimate — but it *does* change the sample, and hence would silently desynchronise the β support
from the `G_K.csv` denominator. Define `𝒜⁺_s` **once**, export it, and use it for the regression,
for the `G(K)` denominator and for the model's `CELL_MASK`.

**(3) `log_X` is in the target but not in the model.** `SIREN`/size is deferred (§2). If the
empirical β is estimated *with* a size control and the model computes β *without* one, the two
coefficients are different estimands and the moment is meaningless. For v1: **include size on both
sides or on neither.** Given §3.9 showed size is the dominant within-area force, the honest options
are to bring `SIREN` forward, or to run v1 without size and record that `α̂` is biased upward in
magnitude.

### To fix, cheap

**(4) The distance floor.** Julia uses `log(max(d, 1.0))` (`load_parameters.jl:510`). The draft
uses `np.log(df.dist_com_w_arithmetic)` with no floor — any `d = 0` (a ZE hosting its own
downstream plant) becomes `−inf`. Mirror the floor: `np.log(np.maximum(d, 1.0))`.

**(5) Which distance.** Julia's regressor is `LOG_CLOSEST_DIST[r]` = distance from ZE `r` to its
**nearest downstream region**, and the bins are `DistBin[r, dr]` at that same nearest region.
Confirm `dist_com_w_arithmetic` is that distance and not a weighted average over all downstream
plants — the name suggests an arithmetic mean, which would be a different regressor.

**(6) The stale `not_supply` comment.** The header says `Y = "ne fournit pas"` so that
`coef(log_d) = αθ`, and there is a commented-out `not_supply` line — but the code regresses
`supplier_num`. The code is now the right one; the comment must read **`coef(log_d) = −αθ`, hence
`α = −β̂/θ`**.

### Confirmed correct

**(7) The distance bins match.** Julia `distance_bin(d, 4)` gives `(50,100] → 1`, `(100,150] → 2`,
`(150,200] → 3`, `d > 200 → 4`, and `d ≤ 50 → 0` = **no dummy**, i.e. the reference. The draft's
extraction order

```python
["C(dist_bin)[T.50-100]", "C(dist_bin)[T.100-150]", "C(dist_bin)[T.150-200]", "C(dist_bin)[T.200+]"]
```

matches, with `0-50` as base. Make it robust rather than lucky: build `dist_bin` as an explicit
ordered categorical and set the reference by name.

```python
df["dist_bin"] = pd.cut(d, [0, 50, 100, 150, 200, np.inf],
                        labels=["0-50","50-100","100-150","150-200","200+"], right=True)
# formula: "... + C(dist_bin, Treatment(reference='0-50')) + ..."
```

Left-open/right-closed (`right=True`) matches Julia's `50 < d <= 100`. Also assert that no cell
falls outside the bins.

### Notes, not errors

**(8) `cov_type="HC1"` is not what enters `W`.** The weight matrix uses `Σ_data` from the joint
bootstrap over suppliers that also produces the γ and G blocks and their cross-covariances. HC1
here is fine for eyeballing; it must not be the source of the β block of Σ. Cluster at
`A129_AA` when reporting, matching the §3.9 diagnostic.

**(9) `C(A129_AA)` must be sector × AA, not AA alone.** The Julia group is
`(s-1)*R_downstream + CLOSEST_DOWNSTREAM_REGION[r]`. Confirm `A129_AA` concatenates both.

**(10) Explicit dummies scale badly.** `smf.glm` with `C(A129_AA)` materialises one column per
group. At 1161 rows and a few dozen groups this is fine; if the group count grows, use
`pyfixest`'s `feglm` (already a dependency) to absorb instead.

---

## 9. Conventions settled, and what remains

### Settled

| Item | Answer |
|---|---|
| Σ block ordering | **β → γ → G**, exactly `S` rows of `G_s(0)` |
| `G(K)` convention | `G(K) = Pr(K_ls ≤ K)`; `G(0)` = share of ZE with **0** suppliers |
| `G(K)` denominator | ZE inside **active** attraction areas only |
| `N_s^obs` | column `N_supplier_s` of `G_K.csv`, repeated on every row (assert constant within `A129`) |
| AA map | `closest_downstream_region.npy`, `R × R_downstream` binary — **asserted equal** to the internally computed `CLOSEST_DOWNSTREAM_REGION` (gate G7) |
| `SIREN` size control | **deferred**; consequences recorded in §2 |
| cloglog outcome | **`y = 1{supplier}`** — the model's exact reduced form; requires the empirical β target and the β block of Σ to be regenerated (see below) |
| `θ` | `1.0`, confirmed |

### Required before the first run — a data-side action

**Switching the outcome to `y = 1{supplier}` is not a sign flip.** The cloglog link is asymmetric
under outcome reversal, so the empirical target and its bootstrap covariance must be regenerated
with the same outcome:

* the empirical `reg_coef` target (the β block of the moment vector);
* the **β rows and columns** of all four `Sigma_beta_gamma_cloglog[_1]_f[_aa].npy` files.

The γ and G blocks are unaffected — only the β block and its cross-covariances with γ and G.
This cannot be applied post hoc to the existing files.

The payoff is large and is the reason to do it: `b_dist = −θα` exactly, invariant to `N_s`
(D4), so `α` is read straight off the coefficient with no attenuation correction, and the
`(α, N_s)` identification separates cleanly — `α` from the distance gradient in block 4, `N_s`
from the level in block 6.

### Remaining

1. **`α̂` from this version is expected to be biased upward in magnitude** because `SIREN` is
   deferred (§2). Worth deciding in advance whether that is acceptable for a first run, or whether
   `SIREN` should arrive before the estimation is launched.
