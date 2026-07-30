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

### D4 — `N_s` is concentrated out by integer bisection, at zero simulation cost

Given `q̂`, the model's empty-ZE share is a **closed-form, strictly decreasing** function of an
integer `n`:

```
G_s(0; n) = mean over l ∈ cells_s of (1 - q̂[s,l])^n
```

Each term is decreasing in `n`, so the mean is. Matching the `K=0` target from `G_K.csv` is
therefore a one-dimensional monotone root-find over integers, per sector, costing ~15 evaluations
of a length-`|cells_s|` mean. **The outer optimizer never sees `N_s`: no mixed-integer search, no
smoothing, no rounding heuristic.** `N_s` remains an exact natural number by construction.

Two equivalent formulations of the same estimator:

| | `N_s` in θ | `G_s(0)` in the loss | df on block 6 |
|---|---|---|---|
| (a) joint | yes | yes | 0 |
| (b) **concentrated (recommended)** | no | no — it *defines* `N̂_s` | — |

They give the same point estimate. (b) is cheaper and never leaves ℕ. Note that `N_s` still
couples to `α` through **block 4**: the regression outcome is `(1-q_ls)^{N̂_s}`, so `G_s(0)` fixes
the *level* of the extensive margin and `reg_coef` its *distance gradient* — exactly the joint
identification of the note's §6(d). The `G_s(0)` rows of Σ are needed for **inference on `N̂_s`**
(delta method), not for the loss.

---

## 1. A fifth observation that shrinks the code

The current `reg_coef` regression builds `(n_good + N_CONTROL) × N_rho ≈ 1.1M` rows
(`model_CP.jl:1049–1130`), one per (draw, pair). But the cloglog IRLS estimating equation is
**linear in `y`**:

```
Σ  w · x · (dμ/dη) · (y - μ) / (μ(1-μ))  =  0
```

Within one cell `(s,l)` every draw shares the same regressors, hence the same `η, μ, dμ/dη`,
hence the same IRLS weight — so `N_rho` rows of weight `1/N_rho` with outcomes `y_ρ` are
**algebraically identical** to *one* row of weight `1` with outcome `mean_ρ y_ρ`. The FE
absorption (weighted group means, `model_CP.jl` `_cloglog_irls`) collapses identically for the
same reason.

Therefore:

* the design collapses from ~1.1M rows to **one row per cell** (~1161), a large speedup;
* the current moment is *already* the fractional-outcome cell regression with `y = 1 - q_ls`;
* granularity is the single substitution **`y = (1 - q_ls)^{N_s}`**;
* **`N_s = 1` reproduces the current continuum moment exactly** — a perfect regression gate
  (see G0 in §6).

**One condition:** the collapse is exact only if no regressor varies *within* a cell. The
draw-level `log(z_flat[ρ,g])` size control does vary. On the production path it is off
(`include_size_control = !include_control`, and `include_control=true`), so the collapse is exact
there. And per the §3.9 diagnostic the size control we actually want is the **cell-level
`log(SIREN)`** anyway — the same variable the empirical target conditions on. So: drop the
draw-level `log z` control, add cell-level `log(SIREN)`.

---

## 2. Data inputs

### Supplied

| File | Use |
|---|---|
| `G_K.csv` (`A129`, `K`, `G(K)`) | `K=0` row → block-6 target per sector; full curve → untargeted fit check (G3) |
| `Sigma_beta_gamma_cloglog[_1]_f.npy` | β + γ(**ZE**) + `G_s(0)` covariance — over-identifying variant |
| `Sigma_beta_gamma_cloglog[_1]_f_aa.npy` | β + γ(**AA**) + `G_s(0)` covariance — **the simple version uses this** |

### Still needed

1. **`N_suppliers_obs` per sector** — the observed number of distinct upstream supplier firms
   `N_s^obs`. Gives the admissible range directly (note §5):
   ```
   N_LO[s] = ceil(N_s^obs / R_downstream)      N_HI[s] = N_s^obs
   ```
   It is in principle recoverable from `G_K.csv` (`Σ_l K_ls = R_cells,s · Σ_K K·f_s(K)`), but only
   if the ZE denominator `R_cells,s` is known and the `K` support is complete. **Cleanest: add a
   column** (`N_suppliers`, and ideally `n_ZE` — the denominator of `G(K)`) to `G_K.csv`. It also
   supplies the independent cross-check of §4.3.
2. **The ZE → attraction-area map**, as an explicit input (`attraction_area.csv`, `ze2010` →
   `ze2010_downstream`). The code already has `CLOSEST_DOWNSTREAM_REGION` (`load_parameters.jl:507`),
   which is what builds today's FE group `(s-1)*R_downstream + CLOSEST_DOWNSTREAM_REGION[r]`. But
   the Python diagnostic merges a separate `attraction_area` table. **These must be the same
   partition or the whole alignment argument of §2.2 fails.** Load the file and `@assert` it equals
   `CLOSEST_DOWNSTREAM_REGION`, so a divergence fails loudly instead of silently changing the
   estimand.
3. **`SIREN` per (sector, ZE)** — the firm count, for the cell-level size control (§3.9 showed it
   is the dominant within-area force, and the empirical target conditions on it). Same shape as
   `filter_N_upstream`.

---

## 3. Moment vector layout

Blocks 1–4 keep their roles. Block 5 changes level; block 6 is new.

| # | Block | Before | After |
|---|---|---|---|
| 1 | `agg_labor_share` | 1 | unchanged |
| 2 | `agg_industry_share` | S−1 | unchanged |
| 3 | `pi_r` | R_d−1 | unchanged |
| 4 | `reg_coef` | `N_REG` | unchanged size; **granular** outcome, **cell-level** design |
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

    # ---- (2) concentrate out N_s: integer bisection, per sector --------------
    for s in 1:S:
        G(n) := mean over l with CELL_MASK[s,l] of (1 - q[s,l])^n   # ↓ in n
        if G(N_LO[s]) <= G_TARGET[s]:  N̂[s] = N_LO[s]; clamped[s] = :lo
        elif G(N_HI[s]) >= G_TARGET[s]: N̂[s] = N_HI[s]; clamped[s] = :hi
        else:
            lo, hi = N_LO[s], N_HI[s]                 # G(lo) > target > G(hi)
            while hi - lo > 1:
                mid = (lo + hi) ÷ 2
                if G(mid) > G_TARGET[s]: lo = mid else: hi = mid
            N̂[s] = argmin over n ∈ {lo, hi} of |G(n) - G_TARGET[s]|
            clamped[s] = :none

    # ---- (3) granular extensive margin --------------------------------------
    p0[s,l] = (1 - q[s,l])^N̂[s]                      # Pr(no supplier), exact

    # ---- (4) block 4: cell-level cloglog, fractional outcome ----------------
    #   one row per cell; y = p0 (the `not_supply` convention already in the code)
    #   FE group = (s, AA_OF_ZE[l])  — identical to today's group id
    #   regressors = distance bins (N_REG>1) or log d (N_REG==1), + SIREN_LOG
    #   weight = 1
    reg_coef = cloglog_irls(y = p0, X = [dist | SIREN_LOG], fe = (s, AA_OF_ZE[l]))

    # ---- (5) block 5: AA-level shares ---------------------------------------
    γ_aa[s,a] = Σ_{l ∈ a} γ[s,l]                      # γ from compute_moments

    # ---- (6) block 6: fitted exactly by construction ------------------------
    G0[s] = G(N̂[s])                                   # residual ≈ 0

    m = [labor | industry | π_r | reg_coef | γ_aa | G0]
    return (m - m̂)' W (m - m̂),  N̂,  clamped
```

### 4.2 Where `N_s` enters the loss

Block 6 is fitted by construction, so `N̂_s` influences the objective **only through block 4**:
`p0` shifts the level of the simulated extensive margin, and the cloglog FE absorbs the level
within each AA × sector, leaving the *distance profile* to be fit by `α`. This is the intended
division of labor. It also means the loss is a **profiled** objective in exactly the sense the
existing `profile_T` machinery already handles — the FD Jacobian will pick up the total
derivative `dm/dα` along the `N̂_s(α)` manifold automatically, provided the concentration runs
inside the perturbed evaluation (it does, since it lives inside `loss`).

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

* `clamped = :hi` — even the maximum admissible variety count leaves too many empty ZE: the model
  cannot generate enough sparsity. Suspect `α` too small or `T` too concentrated.
* `clamped = :lo` — even the minimum admissible count produces too few empty ZE.

Either is a **rejection signal for the mechanism**, not a numerical nuisance. Log it per sector
per report, and treat a persistently clamped sector as a finding.

One caution on the discreteness: `N̂_s(θ)` is a step function of the continuous parameters, so
the profiled loss is **piecewise smooth with jumps at the switch points**. Two consequences.
(i) PSO / TikTak are unaffected (both are derivative-free). (ii) The FD Jacobian can straddle a
jump. Mitigation: at the reported estimate, recompute the Jacobian **holding `N̂_s` fixed** at its
value at `θ̂` — the correct object anyway, since `N̂_s` is integer-valued and locally constant with
probability one. Add a check that no FD perturbation changed `N̂_s`; if one did, report it.

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
| **G5** | `α̂` vs the §3.9 within-area evidence (`α/η_size ≈ 1.1`, θ=1) and vs the joint free-`T` search (`α ≈ 0.30`) | the note predicts the new `α̂` moves **up**; if it lands near 0.30 the profiling repair did not bite and §3.6 needs revisiting |
| **G6** | AA-level Sinkhorn: round-trip recovery of a planted `T_aa` from its own AA aggregates | ~1e-8, mirroring `test/test_ge_inversion.jl` |

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

## 8. Open questions

Ordered by how much they change the implementation.

1. **Σ block ordering.** Where do the `G_s(0)` rows sit in the four new Σ files — appended after γ
   (my assumption: β → γ → G), or interleaved/first? And are there exactly `S` of them, or only for
   sectors with an active AA?
2. **`G(K)` convention.** The column is described as *"the share of ZE with **less than** K
   suppliers"*. Read literally, `G(0) = 0` and the empty-ZE share is `G(1)`. I have assumed `≤ K`
   (so `G(0)` = empty-ZE share). Which is it? This shifts every index by one.
3. **`G(K)` denominator.** Is the share taken over *all* ZE, or only ZE inside active attraction
   areas (`𝒜⁺_s`, the 1161 cells of §3.9)? The model counterpart must use the same support, and the
   difference is large (91.5% vs something else). My assumption: active AAs only.
4. **`N_s^obs`** — can you add the observed distinct-supplier count per sector (and the `G(K)`
   denominator `n_ZE`) as columns of `G_K.csv`? Needed for the bounds and for the free cross-check
   of §4.3. Otherwise I derive it from the curve, which requires the complete `K` support.
5. **The AA map.** Should the Julia side use the existing `CLOSEST_DOWNSTREAM_REGION`, or do you
   have a separate `attraction_area` table (the one the Python diagnostic merges)? If the latter,
   please supply it — I will assert equality against `CLOSEST_DOWNSTREAM_REGION` so any divergence
   fails loudly.
6. **`SIREN` per (sector, ZE)** — available as an array on the Julia side? §3.9 showed the size
   control is not optional, and the empirical target conditions on it, so the model must too.
7. **Outcome convention for the cloglog.** The code uses `y = not_supply` with distance coefficient
   `+αθ`; the note's Proposition 1 shows the *exact* reduced form uses the supplier indicator with
   `-θα`. Keep the current convention (fine for the SMM, since the same regression runs on both
   sides) or switch to the exact one? I would keep it for continuity and note the wedge.
8. **`θ = 1`** in `load_parameters.jl:64` (with `1.768` commented out). Confirm this is the
   intended calibration, since every reg_coef ↔ `α` mapping scales by `1/θ`.
