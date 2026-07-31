# Implementation plan — granular varieties + comparative advantage at the attraction-area level

What to build. The model is in `documentation/finite_sample2.tex`; the validation gates and the
open assumptions are in `documentation/granular_validation.md` and are **not** part of this
implementation.

---

## 0. The overriding constraint: the previous version must stay reachable

Every change below is behind a flag, and with all flags at their legacy values the code must
reproduce the current estimates. Two independent switches, because they are two independent
modelling changes:

| Flag | Legacy value | New value | Governs |
|---|---|---|---|
| `--granular` | `false` | `true` | finite `N_s`, the count moment (block 6), the `N_s` profiling. `false` ⟹ the continuum extensive margin exactly as today, **no block 6**, moment vector of length 5 |
| `--ca_level` | `ze` | `aa` | comparative advantage at ZE or attraction-area level, and correspondingly the γ block and the Σ file (`_aa` suffix) |

**`--granular=false --ca_level=ze` is the full EK model as it stands today.** It must be a
supported configuration, not a historical curiosity: no dead code paths, no silent behaviour
change, and it is the first thing to check after every commit.

Design rules that follow:

* **Block 6 is appended, never inserted.** `BLOCK_RANGES` becomes a 6-tuple only when
  `--granular=true`; otherwise it stays a 5-tuple and every existing index into it is untouched.
  The Σ-file selector gains the `G` rows only in granular mode.
* **The AA layer is a gather, not a rewrite.** `T[s,l] = T_aa[s, AA_OF_ZE[l]]` under
  `--ca_level=aa`; under `ze` the existing reduced-T scatter runs unchanged. `unpack_params`
  branches once, at the top.
* **The active set is the one genuinely shared change.** Under `--ca_level=aa` the status-2
  ("control") cells become ordinary goods; under `ze` they stay as they are. Keep both paths in
  `load_parameters.jl` behind the same flag.
* **The regression keeps its `(ρ, g)` row structure in both modes.** Only the number of rows
  changes: `N_rho` under `false`, `N̂_s` under `true`. The outcome (`not_supply`) and the `log z`
  control are the same in both.

---

## 1. The five design decisions

### D1 — The economy is simulated with exactly `N_s` varieties, `R_rep` replications

Each replication is one realised granular economy: `N_s` varieties per sector, one Fréchet
champion per (cell, variety), Ricardian competition across cells for each variety and buyer.
Moments are computed on the realised economies and averaged.

**One solve, prefixes for everything.** The winner of a variety is the argmin over cells of *that
variety's own draws*, so it does not depend on how many varieties exist. A single Ricardian solve
at pool width therefore serves every candidate `N_s`: `q̂` is a column mean over all pooled
varieties, the realised counts are prefix row-sums, and the `R_rep` economies are prefixes of the
same pool. No re-solve, no new draws.

**`R_rep` replications, never `N_rho = N_s`.** The value block is a numerical integral; the variety
count is a structural parameter. At `N_s ≈ 10²` the labour share, `π_s`, `π_r` and the shares
computed from `N_s` draws would be hopeless. Granularity stays exactly `N_s` *within* a
replication; the value block averages over all `R_rep × N_s` draws.

**Fixed draw pool, common random numbers.** `U_POOL :: (R_rep, N_BLOCK, n_cells)`, drawn once.
`N_BLOCK ≥ max_s N_HI[s]`. Prefixes give common random numbers across `N_s` candidates and across
`θ`, which is what keeps the optimiser usable.

Sizing `R_rep`: the dispersion of `β̂` across realised economies is ~0.16–0.18 in configurations
matching the data, so `R_rep ≈ 300` gives an MC standard error of ~0.010 on block 4.

### D2 — The value block is computed at the certainty-equivalent price index

Compute the value block from the **pooled** draws, not the `N̂_s` prefix: downstream firms optimise
against `E[P^{1−ν_s}]^{1/(1−ν_s)}`. The self-normalised weights (`1/N`, an average not a sum)
already implement the love-of-variety normalisation. This is what makes the `N_s` inner loop of D5
exact.

It is an assumption with a measured cost, discussed in `granular_validation.md` §A.1 and tested by
gate V11. Do not re-derive it here; implement the pooled-draw computation and move on.

### D3 — Comparative advantage, and the γ moments, at the AA level (`--ca_level=aa`)

`T_{sl} = T_{s,a(l)}` — one parameter per active (sector, AA). Draws stay independent across ZE:
two ZE in an area share the Fréchet *scale*, not the realisation.

Block 5 becomes the AA aggregate `γ_{s,a} = Σ_{l∈a} γ_{ls}`. A ZE inside an active AA with no
supplier has `γ̂_ls = 0` and a bootstrap variance of zero — it cannot enter the γ block without
infinite weight; its information goes to the extensive margin and the counts. Use the `..._aa.npy`
Σ files. The γ block is then just-identified against `T_aa` (one moment per active `(s,a)`, minus
one reference per sector).

### D4 — The extensive-margin regression is at the firm level, outcome `not_supply`

A firm **is** a (cell, variety) champion:

| model | code | data |
|---|---|---|
| champion productivity of cell `g` for variety `ρ` | `z_flat[ρ, g]` (`model_CP.jl:609–616`) | `z_i` |
| that champion wins somewhere | `linkages_flat[ρ, g]` (`model_CP.jl:720`) | `supplier_i` |
| distance to the downstream anchor | `LOG_CLOSEST_DIST[r]` / `DistBin[r,dr]` | `log_d` |
| FE group | `(s−1)*R_downstream + CLOSEST_DOWNSTREAM_REGION[r]` | sector × AA |

So the existing `(ρ, g)` row structure and the existing `not_supply` outcome are both correct:
conditioning on the firm's productivity, `Pr(no supply | z)` is exactly a cloglog with `+θα` on
log distance and `−θ` on `log z`, and **`N_s` does not appear** (`finite_sample2.tex`,
Proposition 1). Under granularity `ρ` simply runs `1..N̂_s` instead of `1..N_rho`.

Two code consequences:

* **Return the `log z` coefficient** alongside the `N_REG` distance coefficients. It should equal
  `−θ`; it is a free diagnostic and costs one array slot.
* **Drop the `include_control` / `include_size_control` exclusion** (`model_CP.jl:846`, `:1053`)
  under `--ca_level=aa`. It exists because status-2 cells had `T ≡ 0` and hence no productivity
  draw; under D3 they inherit `T_{s,a} > 0` and have real draws, so both can be on at once. Keep
  the assertion under `--ca_level=ze`.

### D5 — `N_s` profiled over the integers, closed-form inner loop

```
N̂(θ_c) = argmin over N ∈ ∏_s [N_LO_s, N_HI_s] ∩ ℕ  of  r(θ_c, N)' W r(θ_c, N)
```

the profiled extremum estimator — identical to joint minimisation over `(θ_c, N)`, integer at
every evaluation, never relaxed and never rounded, and the outer optimiser never sees `N_s`.

Because `q_ls` is free of `N_s` (`finite_sample2.tex`, Lemma 2) and `G(s,n)` is closed form and
strictly decreasing in `n`, the inner problem is a monotone integer bisection on `q̂` — no
re-simulation. Bounds from the observed firm count:

```
N_HI[s] = N_supplier_s                        N_LO[s] = ceil(N_supplier_s / R_downstream)
```

Clamping at either bound is recorded per sector per report: it is a rejection signal for the
mechanism, not a numerical nuisance.

---

## 2. Data inputs

| File | Convention |
|---|---|
| `G_K.csv` — `A129`, `K`, `G(K)`, `N_supplier_s` | `G(K) = Pr(K_ls ≤ K)`; `G(0)` = share of ZE with **0** suppliers. Denominator: **ZE inside active attraction areas only**. `N_supplier_s` repeated on every row — assert constant within `A129`. |
| `Sigma_beta_gamma_cloglog[_1]_f_aa.npy` | Ordering **β → γ(AA) → G**, exactly `S` rows of `G_s(0)`. Used when `--ca_level=aa`. |
| `Sigma_beta_gamma_cloglog[_1]_f.npy` | γ at ZE level. Used when `--ca_level=ze`; the `G` rows are read only in granular mode. |
| `closest_downstream_region.npy` | `R × R_downstream` binary incidence, ZE → AA. Assert `argmax_col(·) == CLOSEST_DOWNSTREAM_REGION` at load time. |

**Deferred: `SIREN`** (cell-level firm count). When it arrives it is one extra column in the
design and must simultaneously enter the empirical target. It is *not* a substitute for `log z`:
`log z` is the firm's own productivity, `log SIREN` the cell's firm count. Until it is in, `α̂` is
expected to be biased upward in magnitude.

**Open modelling decision, recorded not resolved.** The firm-level unit gives every cell exactly
`N_s` firms while observed counts vary by orders of magnitude. v1 accepts the mismatch; the
alternatives (a productivity floor making the count Poisson in `T_sl`, or taking the observed
count as given) are in `granular_validation.md` §A.2, with gate V6 measuring the damage.

---

## 3. Moment vector layout

| # | Block | `--granular=false` | `--granular=true` |
|---|---|---|---|
| 1 | `agg_labor_share` | 1 | unchanged |
| 2 | `agg_industry_share` | S−1 | unchanged |
| 3 | `pi_r` | R_d−1 | unchanged |
| 4 | `reg_coef` | `N_REG`, rows `(ρ ≤ N_rho, g)` | `N_REG`, rows `(ρ ≤ N̂_s, g)` |
| 5 | `gamma_ls` | active (s,ZE) − ref, or (s,AA) − ref under `--ca_level=aa` | same |
| 6 | `G0` | **absent** | `S` entries |

Σ ordering **β → γ → G**, extending the existing `gb_indices` invariant.

---

## 4. The estimation algorithm

`R` = ZE, `R_d` = downstream regions = number of AAs, `S` = sectors, `n_cells` = |CELL_MASK|.

```text
════════════════════════════════════════════════════════════════════════════════
PRECOMPUTED ONCE   (load_parameters.jl)
════════════════════════════════════════════════════════════════════════════════
AA_OF_ZE        :: Vector{Int}(R)        from closest_downstream_region.npy
                                         @assert == CLOSEST_DOWNSTREAM_REGION
AA_ACTIVE       :: BitMatrix(S, R_d)     a hosts ≥1 observed supplier in s   → 𝒜⁺_s
CELL_MASK       :: BitMatrix(S, R)       ca_level == :aa ?
                                           has_firms[s,l] && AA_ACTIVE[s, AA_OF_ZE[l]]
                                         : (filter_N_upstream .== 1) .& (X_rs .> 0)   # legacy
                                         has_firms = filter_N_upstream ∈ {1,2}
CELLS_OF_SECTOR :: Vector{Vector{Int}}(S)
GOOD_S, GOOD_R, SR_TO_GOOD               rebuilt from CELL_MASK
T_MASK_AA       :: BitMatrix(S, R_d)     = AA_ACTIVE                (aa mode only)
T_REF_AA        :: Vector{Int}(S)        one reference AA per sector, T[s,ref] ≡ 1
emp_gamma_aa    :: Matrix(R_d, S)        Σ_{l∈a} emp_gamma_ls[l,s]  → block-5 target (aa)
G_TARGET        :: Vector{Float64}(S)    G_K.csv at K = 0           → block-6 target
N_LO, N_HI      :: Vector{Int}(S)        from N_supplier_s
N_BLOCK         :: Int                   ≥ maximum(N_HI)
LOG_DIST        :: Vector(R)             log(max(closest_plant_dist, 1.0))
DIST_BIN        :: Vector{Int}(R)        ∈ 0:N_REG, 0 = reference (d ≤ 50)
FE_GROUP        :: Vector{Int}(n_cells)  (s−1)*R_d + AA_OF_ZE[l]
U_POOL          :: Array(R_rep, N_BLOCK, n_cells)   drawn once; prefixes = common random numbers

════════════════════════════════════════════════════════════════════════════════
PARAMETER VECTOR   (N_s is NOT in it)
════════════════════════════════════════════════════════════════════════════════
θ_c = [ Ω^L(1) | Ω^s(S) | A(R_d) | α(N_TAU) | T_block ]
      T_block = T_aa(n_T_aa)  under --ca_level=aa ;  T_reduced(n_T)  under ze

════════════════════════════════════════════════════════════════════════════════
ONE LOSS EVALUATION
════════════════════════════════════════════════════════════════════════════════
function loss(θ_c, W)

  ── 0. unpack ───────────────────────────────────────────────────────────────
  Ω^L, Ω^s, A, α, T_block = unpack_params(θ_c)
  T[s,l] = (ca_level == :aa) ? T_aa[s, AA_OF_ZE[l]]      # gather — the one new line
                             : T_reduced scattered as today

  ── 1. ONE Ricardian solve at pool width ────────────────────────────────────
  pool = solve_network(θ_c; u_draws = U_POOL)
  #   pool.linkages_flat :: (R_rep, N_BLOCK, n_cells)   1 = wins somewhere
  #   pool.z_flat        :: (R_rep, N_BLOCK, n_cells)   champion productivity
  #   value block computed here, over ALL pooled draws              ← D2

  if !granular
      # legacy path: no prefix, no block 6
      reg_coef, b_logz = cloglog(y = 1 − pool.linkages_flat, X = [dist | log z], fe = FE_GROUP)
      γ, labor, industry, π_r = compute_moments(pool, θ_c)
      m = vcat(labor, industry, π_r, reg_coef, γ[mask])
      return (m − m̂)' W (m − m̂)
  end

  ── 2. q̂ from ALL pooled varieties → N̂_s by bisection ──────────────────────
  q̂[s,l] = mean over (rep, ρ) of pool.linkages_flat[rep, ρ, g(s,l)]
  q̂[s,l] = clamp(q̂[s,l], 0.5/(R_rep*N_BLOCK), 1 − 1e-12)
  for s in 1:S
      G(s,n) = mean over l ∈ CELLS_OF_SECTOR[s] of (1 − q̂[s,l])^n        # ↓ in n
      if     G(s,N_LO[s]) ≤ G_TARGET[s]   N̂[s], clamped[s] = N_LO[s], :lo
      elseif G(s,N_HI[s]) ≥ G_TARGET[s]   N̂[s], clamped[s] = N_HI[s], :hi
      else
          lo, hi = N_LO[s], N_HI[s]
          while hi − lo > 1
              mid = (lo + hi) ÷ 2
              G(s,mid) > G_TARGET[s] ? (lo = mid) : (hi = mid)
          end
          N̂[s], clamped[s] = argmin_{n ∈ {lo,hi}} |G(s,n) − G_TARGET[s]|, :none
      end
      @assert N̂[s] ≤ N_BLOCK          # else widen N_BLOCK and restart
  end

  ── 3. the R_rep economies are PREFIXES of the pool — no new draws ──────────
  K[rep,s,l] = Σ_{ρ ≤ N̂[s]} pool.linkages_flat[rep, ρ, g(s,l)]

  ── 4. block 4 — firm-level cloglog on the prefix ───────────────────────────
  for rep in 1:R_rep
      y[ρ,g] = 1 − pool.linkages_flat[rep, ρ, g]        for ρ ≤ N̂[sector(g)]
      X[ρ,g] = [ DIST_BIN onehot (or LOG_DIST) | log(pool.z_flat[rep, ρ, g]) ]
      w[ρ,g] = 1 / N̂[sector(g)]                         # each cell carries total weight 1
      reg[rep] = cloglog_irls(y, X, fe = FE_GROUP[g])
  end
  reg_coef = mean over rep of reg[rep][1:N_REG]
  b_logz   = mean over rep of reg[rep][N_REG+1]         # diagnostic: should equal −θ

  ── 5. block 5 — shares (AA-aggregated under --ca_level=aa) ─────────────────
  γ_aa[a,s] = Σ_{l : AA_OF_ZE[l] == a} γ[l,s]

  ── 6. block 6 — realised empty-ZE share, averaged over replications ────────
  G0[s] = mean over rep of ( share of l ∈ CELLS_OF_SECTOR[s] with K[rep,s,l] == 0 )

  ── 7. assemble ────────────────────────────────────────────────────────────
  m = vcat(labor, industry, π_r, reg_coef, γ_aa[mask], G0)
  return (m − m̂)' W (m − m̂),  (N̂, clamped, b_logz, K)
end
```

`N̂_s(θ)` is a step function of the continuous parameters, so the profiled loss is piecewise smooth
with jumps. PSO / TikTak are derivative-free and unaffected. The FD Jacobian can straddle a jump:
at `θ̂` recompute it **holding `N̂_s` fixed** — the correct object anyway, since `N̂_s` is locally
constant with probability one — and assert that no perturbation moved it.

---

## 5. File-by-file change list

### `load_parameters.jl`

* **Flags.** Parse `--granular` and `--ca_level`; broadcast as `GRANULAR::Bool`, `CA_LEVEL::Symbol`.
* **Active set.** Under `:aa`, replace `active_mat` by
  * `has_firms[s,l] = filter_N_upstream[s,l] ∈ {1,2}` (status 0 = no firms, stays out);
  * `AA_ACTIVE[s,a] = any(l ∈ a : filter==1 && X_rs>0)` → `𝒜⁺_s`;
  * `CELL_MASK[s,l] = has_firms[s,l] && AA_ACTIVE[s, AA_OF_ZE[l]]`;

  then build `n_good`, `GOOD_S`, `GOOD_R`, `SR_TO_GOOD`, `SECTOR_GOOD_INDICES` from `CELL_MASK`.
  Status-2 cells become ordinary goods with `T = T_{s,a} > 0` — **this is the endogenisation**.
  Under `:ze` the current code path runs unchanged.
* New constants: `AA_OF_ZE`, `n_AA`, `AA_CELLS`, `T_MASK_AA`, `T_REF_AA`, `CELL_MASK`,
  `CELLS_OF_SECTOR`, `N_LO`, `N_HI`, `N_BLOCK`, `G_TARGET`, `R_REP`, `U_POOL`, `GRANULAR`,
  `CA_LEVEL`.
* γ target → `emp_gamma_aa` under `:aa`; keep `emp_gamma_ls`.
* `compute_block_ranges` returns 5 or 6 blocks depending on `GRANULAR`; `BLOCK_NAMES` likewise.
* Labels: `T[sname-aaname]`, `gamma[sname-aaname]`, `G0[sname]` in the new modes.

### `model_CP.jl`

* `unpack_params`: branch once on `CA_LEVEL`. Under `:aa`, scatter into `(S, n_AA)` s-major,
  ref-normalise on `T_REF_AA`, gather `T[s,l] = T_aa[s, AA_OF_ZE[l]]`. Keep the `eltype(params)`
  generic array so the AD path survives.
* `solve_network`: no logic change; it receives more goods and a 3-D draw array.
* New `concentrate_N_s(q̂, G_TARGET, N_LO, N_HI) -> (N̂, clamped)`.
* `fast_cloglog_regression`: keep the `(ρ, g)` rows and the `not_supply` outcome; add the prefix
  bound `ρ ≤ N̂[sector]`; **return the `log z` coefficient** alongside the `N_REG` distance ones;
  drop the `include_control` / `include_size_control` exclusion under `:aa`.
* `compute_moments`: γ → AA aggregate under `:aa`; new `G0` from realised counts under `GRANULAR`;
  average over replications.
* `moments_to_vec` / `full_SMM`: append block 6 only when `GRANULAR`.

### `profiling.jl`

* `invert_T_ge` under `:aa`: target `emp_gamma_aa`; one multiplicative update per active `(s,a)`
  against the **aggregated** model share `Σ_{l∈a} γ_model[l,s]`; ref-normalise on `T_REF_AA`.
  Square system, contraction inherited from the ZE-level version.

### `tools.jl`

* `sigma_beta_gamma_filename(; smm, aa)` → `_aa` suffix when `CA_LEVEL == :aa`.
* `reconcile_sigma_data`: reconcile at the AA level under `:aa`; `G0` rows never pruned.
* `build_step3_weight_matrix`: `gb_indices` gains `BLOCK_RANGES[6]` when `GRANULAR`.
* `compute_jacobian`: `hold_N_s` kwarg and the "no perturbation moved `N̂_s`" assertion.
* `compute_smm_inference`: report `N̂_s` value-only.
* `generate_report`: `N̂_s` and clamp flags per sector, `b_logz` against `−θ`, and the realised
  count distribution.

### `optimizer.jl` / `main.jl` / `run.sh`

* T block sized `n_T_aa` under `:aa`; block-coordinate stages unchanged in structure.
* New flags: `--granular=true|false` (default `false` until V0 passes), `--ca_level=ze|aa`
  (default `ze`), `--n_rep` (`R_rep`).

---

## 6. Suggested build order

1. **Flags and the legacy path.** Add `--granular` / `--ca_level`, wire them through, change
   nothing else. Confirm the legacy configuration is bit-identical.
2. **AA layer** (`--ca_level=aa`): `AA_OF_ZE` + assertion, `CELL_MASK`, `T_aa` gather,
   AA-aggregated γ, `_aa` Σ selector, AA-level Sinkhorn.
3. **Draw pool restructure**: `U_POOL` as `(R_rep, N_BLOCK, n_cells)`, one solve, prefix
   arithmetic. Still `--granular=false`.
4. **Granularity** (`--granular=true`): `concentrate_N_s`, the prefix regression, block 6, the
   report additions.
5. Then the gates in `granular_validation.md`, one at a time.

---

## 7. Data-side actions

The empirical β target and the β block of Σ are **correct as they stand** — the `not_supply`
outcome and the firm-level unit are the exact reduced form, so nothing needs regenerating. The
remaining fixes to the empirical script are listed in `granular_validation.md` Part IV; the one
that matters for consistency with the model is replacing `nunique() == 2` by
`transform("max") == 1`, so that `𝒜⁺_s` is defined once and shared by the regression, the `G(K)`
denominator and `CELL_MASK`.
