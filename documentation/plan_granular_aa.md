# Implementation plan — granular varieties + comparative advantage at the attraction-area level

Target: the **simple version**. Comparative advantage lives at the attraction-area (AA) ×
sector level only (`σ²_ε = 0`, no ZE-level deviation); the number of varieties `N_s` is finite
and estimated from the supplier-count distribution.

Companion theory: `documentation/finite_sample2.tex` (§3 for the two-tier geography, §5 for the
count moment, §6 for what must change, §3.9 for the empirical diagnostic).

---

## 0. Four design decisions that make this cheap

These are the load-bearing choices. Everything downstream follows from them.

### D1 — Granularity is applied **analytically**, never by drawing `N_s` varieties

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

```
# ── Precomputed once (load_parameters.jl) ─────────────────────────────────────
AA_OF_ZE[l]            : ZE → attraction area                     (length R)
AA_ACTIVE[s,a]         : a hosts ≥1 observed supplier in s          (S × n_AA)
CELL_MASK[s,l]         : has_firms[s,l] AND AA_ACTIVE[s, AA_OF_ZE[l]]
                         → the ~1161 cells: regression + count support
T_MASK_AA[s,a]         : = AA_ACTIVE                  → the estimated T block
T_REF_AA[s]            : one reference AA per sector, T[s,ref] = 1
emp_gamma_aa[a,s]      : Σ_{l∈a} emp_gamma_ls[l,s]                 → block-5 target
G_TARGET[s]            : G_K.csv value at K = 0                    → block-6 target
N_LO[s], N_HI[s]       : ceil(N_s^obs / R_downstream), N_s^obs
SIREN_LOG[s,l]         : log firm count                            → size control

# ── Parameter vector (N_s is NOT in it) ───────────────────────────────────────
θ = [ Ω^L(1) | Ω^s(S) | A(R_d) | α(N_TAU) | T_aa(n_T_aa) ]

# ── One loss evaluation ───────────────────────────────────────────────────────
function loss(θ, W):

    Ω^L, Ω^s, A, α, T_aa = unpack_params(θ)          # T block now (S × n_AA)
    T[s,l] = T_aa[s, AA_OF_ZE[l]]                    # gather: NEW, the only
                                                     # structural line added

    net = solve_network(θ; u_draws = U_DRAWS)         # UNCHANGED

    # ---- (1) win-somewhere probability, per cell -----------------------------
    for (s,l) in CELL_MASK:
        q[s,l] = mean_ρ net.linkages_flat[ρ, SR_TO_GOOD[s,l]]
        q[s,l] = clamp(q[s,l], eps, 1 - eps)         # guard the ^N_s and the log

    # ---- (2) profile out N_s over the integers (NO re-simulation) -----------
    #  G(s,n) := mean over l with CELL_MASK[s,l] of (1 - q[s,l])^n   # closed form, ↓ in n
    #  2a: warm start = exact-match root of G(s,n) = G_TARGET[s]
    for s in 1:S:
        if   G(s,N_LO[s]) <= G_TARGET[s]: n0[s] = N_LO[s]
        elif G(s,N_HI[s]) >= G_TARGET[s]: n0[s] = N_HI[s]
        else:
            lo, hi = N_LO[s], N_HI[s]                 # G(lo) > target > G(hi)
            while hi - lo > 1:
                mid = (lo + hi) ÷ 2
                if G(s,mid) > G_TARGET[s]: lo = mid else: hi = mid
            n0[s] = argmin over n ∈ {lo,hi} of |G(s,n) - G_TARGET[s]|

    #  2b: integer coordinate descent on the FULL weighted loss (blocks 4+6).
    #      Sectors couple only through the shared distance coefficients of the
    #      pooled regression, so one pass usually suffices; a 2nd pass is the check.
    N̂ = n0
    repeat until no sector moves (max 3 passes):
        for s in 1:S:
            N̂[s] = argmin over n ∈ window(N̂[s]) ∩ [N_LO[s],N_HI[s]] ∩ ℕ
                     of loss_blocks_4_6(N̂ with entry s replaced by n)
            #  window = geometric bracket around the incumbent, then integer
            #  bisection on the loss difference; each candidate = ONE collapsed
            #  cloglog fit on ~1161 rows — cheap, and no re-simulation
    clamped[s] = (N̂[s]==N_LO[s]) ? :lo : (N̂[s]==N_HI[s]) ? :hi : :none

    # ---- (3) granular extensive margin --------------------------------------
    p0[s,l] = (1 - q[s,l])^N̂[s]                      # Pr(no supplier), exact

    # ---- (4) block 4: cell-level cloglog, fractional outcome ----------------
    #   one row per cell; y = 1 - p0  (SUPPLIER indicator)
    #   FE group = (s, AA_OF_ZE[l])  — identical to today's group id
    #   regressors = distance bins (N_REG>1) or log d (N_REG==1); NO draw-level
    #                control (SIREN_LOG joins here when it arrives)
    #   weight = 1
    reg_coef = cloglog_irls(y = 1 - p0, X = [dist], fe = (s, AA_OF_ZE[l]))

    # ---- (5) block 5: AA-level shares ---------------------------------------
    γ_aa[s,a] = Σ_{l ∈ a} γ[s,l]                      # γ from compute_moments

    # ---- (6) block 6: near-zero residual, not exactly zero ------------------
    G0[s] = G(s, N̂[s])          # ≈ target; step 2b may trade a little G0 fit
                                # against block 4 — that is the profiled optimum

    m = [labor | industry | π_r | reg_coef | γ_aa | G0]
    return (m - m̂)' W (m - m̂),  N̂,  clamped
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

## 8. Conventions settled, and what remains

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
