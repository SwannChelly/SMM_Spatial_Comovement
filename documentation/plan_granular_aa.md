# Granular varieties + comparative advantage at the attraction-area level

The design, as implemented. The model is in `documentation/finite_sample2.tex`; the
validation gates and the open assumptions are in `documentation/granular_validation.md`;
the record of everything that changed, and why, is in
`documentation/granular_aa_changes.md`.

---

## 0. The overriding constraint: the previous version stays reachable

Every change is behind a flag, and with all flags at their legacy values the code
reproduces the previous estimates bit for bit. Two switches, because they are two
independent modelling changes:

| Flag | Legacy value | New value | Governs |
|---|---|---|---|
| `--granular` | `false` | `true` | finite `N_s`, the count moment (block 6), the `N_s` profiling. `false` ⟹ the continuum extensive margin exactly as before, **no block 6**, moment vector of length 5 |
| `--ca_level` | `ze` | `aa` | comparative advantage at ZE or attraction-area level, and correspondingly the γ block and the Σ file (`Sigma_aa_` prefix) |

**`--granular=false --ca_level=ze` is the full EK model as it stood.** It is a supported
configuration, not a historical curiosity: no dead code paths, no silent behaviour
change, and it is the first thing to check after every commit (gate **V0**).

**`--granular=true` requires `--ca_level=aa`** — a hard error otherwise, in both
`load_parameters.jl` and `run.sh`. Under ZE-level comparative advantage only the supplier
cells are estimated, so no cell can come out empty: the count moment has no content, and
the estimator would be fitting `T > 0` for regions whose observed `γ_ls` is 0. Granularity
only means something once cells inherit their area's comparative advantage. The reverse
combination (`--granular=false --ca_level=aa`, the AA-level continuum) is legitimate and
only warns.

Design rules that follow:

* **Block 6 is appended, never inserted.** `BLOCK_RANGES` becomes a 6-tuple only when
  `--granular=true`; otherwise it stays a 5-tuple and every existing index into it is
  untouched. The Σ-file selector keeps the `G` rows only in granular mode.
* **The AA layer is a gather, not a rewrite.** `T[s,l] = T_par[s, AA_OF_ZE[l]]` under
  `--ca_level=aa`; under `ze` the existing reduced-T scatter runs unchanged.
  `unpack_params` branches once, at the top.
* **The active set is the one genuinely shared change.** Under `--ca_level=aa` the control
  cells (`filter == 1` and `X_rs == 0`) become ordinary goods; under `ze` they stay as
  they are. The `filter_N_upstream` re-encoding of §2 is *not* behind a flag — it changes
  how the control set is built in both modes.
* **The regression keeps its `(ρ, g)` row structure and its draws in both modes.** The
  outcome (`not_supply`) and the `log z` control are the same, and so is the row set:
  `N_s` never enters block 4 (Proposition 1).

---

## 1. The five design decisions

### D1 — The variety count never enters the simulation

`N_s` touches exactly two objects, and both are **closed form** in the win probability
`q̂`:

* the count moment `Ḡ_s(0) = mean_l (1 − q̂_ls)^{N_s}`, and
* `N̂_s` itself, located by a monotone integer bisection on that same expression (D5).

`q̂_ls` — the probability that cell `l` wins a given variety *somewhere* in the downstream
industry — is free of `N_s` (`finite_sample2.tex`, Lemma 2) and is nothing but a column
mean of `linkages_flat`. And by Proposition 1 the firm-level extensive margin carries no
`N_s` term either. Therefore:

* **no prefix of the draws is ever taken**,
* **no realised economy is ever simulated or replicated**,
* every moment is computed **once**, on **one** set of draws, exactly as in the continuum
  model.

There is no draw pool, no replication count, and no separate granular sampler. Simulation
noise is handled **uniformly across all six blocks** by the draw design (`DRAW_METHOD`,
`:sobol` by default) and by `N_rho` — not by a replication device special-cased to one
block. `N_rho` keeps its usual meaning and value; note only that under `--granular` it
also sets the precision of `q̂`, hence of `N̂_s` (`s.e.(q̂) ≈ √(q(1−q)/N_rho)` before the
QMC gain), so raise it if the count moment looks noisy.

**What this costs, stated once.** Block 4 is the continuum-limit coefficient `β(E[y])`
rather than `E[β̂(N_s)]`, so the auxiliary cloglog's finite-sample bias is *not* cancelled
by matching a binding function. It is measured by gate **V12** (2–4% of `β` in
data-consistent configurations) and should be reported alongside `α̂`, or added as a local
offset. In exchange, granular estimation costs exactly what the continuum model costs, and
the `log z` coefficient recovers `−θ` cleanly rather than carrying the finite-sample bias
of a ten-variety regression.

### D2 — The value block is computed at the certainty-equivalent price index

The labour share, `π_s`, `π_r` and the sourcing shares are computed from the draws at
`E[P^{1−ν_s}]^{1/(1−ν_s)}`. The self-normalised weights (`1/N_rho`, an average not a sum)
already implement the love-of-variety normalisation. Under D1 this is automatic — there is
no other economy to compute them on.

It is an assumption with a measured cost, discussed in `granular_validation.md` §A.1 and
tested by gate V11.

### D3 — Comparative advantage, and the γ moments, at the AA level (`--ca_level=aa`)

`T_{sl} = T_{s,a(l)}` — one parameter per active (sector, AA). Draws stay independent
across ZE: two ZE in an area share the Fréchet *scale*, not the realisation.

Block 5 becomes the AA aggregate `γ_{s,a} = Σ_{l∈a} γ_{ls}`. A ZE inside an active AA with
no supplier has `γ̂_ls = 0` and a bootstrap variance of zero — it cannot enter the γ block
without infinite weight; its information goes to the extensive margin and the counts. Use
the `Sigma_aa_beta_gamma*` files (§2), and note the aggregation runs over **(sector, AA)**
pairs — an AA active in one sector may be empty in another. The γ block is then
just-identified against `T_aa` (one moment per active `(s,a)`, minus one reference per
sector).

### D4 — The extensive-margin regression is at the firm level, outcome `not_supply`

A firm **is** a (cell, variety) champion:

| model | code | data |
|---|---|---|
| champion productivity of cell `g` for variety `ρ` | `z_flat[ρ, g]` | `z_i` |
| that champion wins somewhere | `linkages_flat[ρ, g]` | `supplier_i` |
| distance to the downstream anchor | `LOG_CLOSEST_DIST[r]` / `DistBin[r,dr]` | `log_d` |
| FE group | `(s−1)*R_downstream + CLOSEST_DOWNSTREAM_REGION[r]` | sector × AA |

So the existing `(ρ, g)` row structure and the existing `not_supply` outcome are both
correct: conditioning on the firm's productivity, `Pr(no supply | z)` is exactly a cloglog
with `+θα` on log distance and `−θ` on `log z`, and **`N_s` does not appear**
(`finite_sample2.tex`, Proposition 1). The regression is therefore run **once**, on the
ordinary draws, identically in both modes.

Two code consequences:

* **Return the `log z` coefficient** alongside the `N_REG` distance coefficients. It
  should equal `−θ`; it is a free over-identifying test and costs one array slot.
* **Drop the `include_control` / `include_size_control` exclusion** under `--ca_level=aa`.
  It exists because control cells had `T ≡ 0` and hence no productivity draw; under D3
  they inherit `T_{s,a} > 0` and have real draws. Under `:aa` the effective design is
  therefore: control-row path **off** (those cells are already ordinary goods — appending
  them again would double-count) and log-z size control **on**. Keep the assertion under
  `--ca_level=ze`.

### D5 — `N_s` profiled over the integers, closed-form inner loop

```
N̂(θ_c) = argmin over N ∈ ∏_s [N_LO_s, N_HI_s] ∩ ℕ  of  r(θ_c, N)' W r(θ_c, N)
```

the profiled extremum estimator — identical to joint minimisation over `(θ_c, N)`, integer
at every evaluation, never relaxed and never rounded, and the outer optimiser never sees
`N_s`.

Because `q_ls` is free of `N_s` (Lemma 2) and `Ḡ_s(n)` is closed form and strictly
decreasing in `n`, the inner problem is a monotone integer bisection on `q̂` — no
re-simulation, and no dependence on any realised draw beyond `q̂`. Bounds from the observed
firm count:

```
N_HI[s] = N_supplier_s                        N_LO[s] = ceil(N_supplier_s / R_downstream)
```

Clamping at either bound is recorded per sector per report: it is a rejection signal for
the mechanism, not a numerical nuisance.

---

## 2. Data inputs

### `filter_N_upstream.npy` — binary encoding

`(S, R)`, values in `{0, 1}` only. The three-status encoding (0 no firms / 1 has suppliers
/ 2 firms but no supplier) is gone.

```
filter_N_upstream[s,l] == 1   ⟺   the (sector, ZE) cell enters the optimisation
supplier cell                 ⟺   filter == 1  AND  X_rs[s,l] >  0
control cell (no supplier)    ⟺   filter == 1  AND  X_rs[s,l] == 0
```

Three consequences for the code:

* **`supplier_cells` is unchanged.** `(filter_N_upstream .== 1) .& (X_rs .> 0)` still
  selects exactly the supplier cells under the new encoding. It plays two roles that must
  be kept apart: under `--ca_level=ze` it IS the estimated set; under `:aa` it only
  *defines* `𝒜⁺`, i.e. which areas host at least one supplier in a sector.
* **The control set is `findall((filter_N_upstream .== 1) .& (X_rs .== 0))`.** Same object
  as the old status 2, new encoding.
* **`CELL_MASK` is given, not derived.** Under `--ca_level=aa` the cell set is simply
  `filter_N_upstream .== 1`. The old `has_firms` construction disappears.

Because the filter now encodes the modelling decision directly, **assert rather than
assume** that it already excludes cells sitting in attraction areas with no supplier:

```julia
@assert all(AA_ACTIVE[s, AA_OF_ZE[l]] for (s,l) in findall(CELL_MASK))
```

If it fires, the filter is broader than `𝒜⁺`; intersect and warn rather than silently
proceed.

### `attraction_area_linkages.npy` — the ZE → attraction-area map

`(R, R_downstream)` binary incidence. Each attraction area is anchored on a downstream
region (its closest). `AA_OF_ZE[l]` is the column where row `l` is 1. Asserted at load:

```julia
@assert size(A) == (R, R_downstream)
@assert all(sum(A, dims=2) .== 1)                       # exactly one AA per ZE
@assert vec(argmax(A, dims=2) .|> last) == CLOSEST_DOWNSTREAM_REGION
```

The last one is the important one: the model's fixed effect and the empirical one must be
the same partition, or the whole alignment argument fails.

### Aggregating γ to the attraction-area level

Two rules, both easy to get wrong.

1. **The moment set is (sector, AA) pairs with a supplier in the data**, not AAs:
   ```
   AA_ACTIVE[s,a] = any( l ∈ a : filter[s,l] == 1 && X_rs[s,l] > 0 )
   ```
   A given AA can be active in one sector and empty in another; the sectoral dimension
   must be carried through. Block 5 is the active `(s,a)` set minus one reference AA per
   sector.
2. **The sum runs over every cell in `CELL_MASK`, control cells included**, on *both*
   sides:
   ```
   model :  γ_aa[a,s]         = Σ_{l : AA_OF_ZE[l]==a, CELL_MASK[s,l]} γ_model[l,s]
   data  :  emp_gamma_aa[a,s] = Σ_{l : AA_OF_ZE[l]==a, CELL_MASK[s,l]} emp_gamma_ls[l,s]
   ```
   Control cells contribute `0` on the data side and a strictly positive `γ` on the model
   side. That is the correct treatment, not a mismatch: **the data are one realisation of
   the model**, so a control cell has `E[γ_ls] > 0` while its realised draw happens to be
   `0`, and `E[γ̂_ls] = γ_ls` holds unconditionally. Excluding those cells from the model
   sum would bias the aggregate downward — it would compare the model's
   conditional-on-activity aggregate with the data's unconditional one. Assert
   `emp_gamma_ls == 0` wherever `!CELL_MASK`.

### Σ files

Eight files, all of them carrying **three blocks in the order β → γ → G**, with `S` rows of
`G_s(0)`. On disk the dimension is therefore always

```
size(Σ) = (N_REG + n_γ + S)²          n_γ = ZE-level or AA-level depending on the file
```

The AA variant is a **prefix**, not a suffix:

| level | link | `N_REG` | file |
|---|---|---|---|
| ZE × S | lpm | >1 | `Sigma_beta_gamma_f.npy` |
| ZE × S | lpm | 1 | `Sigma_beta_gamma_1_f.npy` |
| ZE × S | cloglog | >1 | `Sigma_beta_gamma_cloglog_f.npy` |
| ZE × S | cloglog | 1 | `Sigma_beta_gamma_cloglog_1_f.npy` |
| AA × S | lpm | >1 | `Sigma_aa_beta_gamma_f.npy` |
| AA × S | lpm | 1 | `Sigma_aa_beta_gamma_1_f.npy` |
| AA × S | cloglog | >1 | `Sigma_aa_beta_gamma_cloglog_f.npy` |
| AA × S | cloglog | 1 | `Sigma_aa_beta_gamma_cloglog_1_f.npy` |

so

```julia
function sigma_beta_gamma_filename(; smm::Bool, aa::Bool = (CA_LEVEL == :aa))
    prefix = aa ? "Sigma_aa_beta_gamma" : "Sigma_beta_gamma"
    link   = REG_METHOD == :cloglog ? "_cloglog" : ""
    n1     = N_REG == 1 ? "_1" : ""
    fsuf   = smm ? "_f" : ""
    return prefix * link * n1 * fsuf * ".npy"
end
```

**Selection and reduction — the operative rule.**

| mode | file | blocks kept |
|---|---|---|
| `--granular=false` | **ZE**-level (`Sigma_beta_gamma*`) | `reg_coef` + `γ` — the `G` rows and columns are **dropped** |
| `--granular=true` | **AA**-level (`Sigma_aa_beta_gamma*`) | `reg_coef` + `γ` + `G_s(0)` — all three |

Since the file layout is fixed, the reduction is a leading-block slice, not a search:

```julia
n_G          = S
n_gamma_file = size(Σ, 1) - N_REG - n_G
@assert n_gamma_file > 0  "Σ must carry N_REG + n_γ + S rows (β → γ → G)"
keep = GRANULAR ? (1:size(Σ,1)) : (1:(N_REG + n_gamma_file))
Σ    = Σ[keep, keep]
```

Order of operations in `reconcile_sigma_data`: load the file for the current level, split
into β / γ / G, reconcile **the γ block only** against the active set, then reassemble —
keeping `G` only under `--granular`. The `G` rows are never pruned.

### `G_K.csv`

`A129`, `K`, `G(K)`, `N_supplier_s`. `G(K) = Pr(K_ls ≤ K)`; `G(0)` = share of ZE with **0**
suppliers; denominator = ZE inside active attraction areas. `N_supplier_s` repeated on
every row — assert constant within `A129`. Bounds: `N_HI[s] = N_supplier_s`,
`N_LO[s] = ceil(N_supplier_s / R_downstream)`.

### Deferred and open

**`SIREN`** (cell-level firm count) is not in this version. When it arrives it is one extra
column in the design and must simultaneously enter the empirical target; it is *not* a
substitute for `log z`. Until then `α̂` is expected to be biased upward in magnitude.

**Firm counts per cell**: the firm-level unit gives every cell exactly `N_s` firms while
observed counts vary by orders of magnitude. v1 accepts the mismatch; see
`granular_validation.md` §A.2.

---

## 3. Moment vector layout

| # | Block | `--granular=false` | `--granular=true` |
|---|---|---|---|
| 1 | `agg_labor_share` | 1 | unchanged |
| 2 | `agg_industry_share` | S−1 | unchanged |
| 3 | `pi_r` | R_d−1 | unchanged |
| 4 | `reg_coef` | `N_REG`, rows `(ρ ≤ N_rho, g)` | **identical** — `N_s` does not enter |
| 5 | `gamma_ls` | active (s,ZE) − ref, or (s,AA) − ref under `--ca_level=aa` | same |
| 6 | `G0` | **absent** | `S` entries, closed form |

Σ ordering **β → γ → G**, extending the existing `gb_indices` invariant. The file always
carries all three blocks; the legacy mode slices off the `G` rows and columns (§2).

---

## 4. The estimation algorithm

`R` = ZE, `R_d` = downstream regions = number of AAs, `S` = sectors,
`n_cells` = |CELL_MASK|.

```text
════════════════════════════════════════════════════════════════════════════════
PRECOMPUTED ONCE   (load_parameters.jl)
════════════════════════════════════════════════════════════════════════════════
AA_OF_ZE        :: Vector{Int}(R)        from attraction_area_linkages.npy (R × R_d binary)
                                         @assert rows sum to 1, and == CLOSEST_DOWNSTREAM_REGION
AA_ACTIVE       :: BitMatrix(S, R_d)     any(l ∈ a : filter[s,l]==1 && X_rs[s,l] > 0)  → 𝒜⁺_s
                                         NB sectoral: an AA can be active in s and empty in s'
CELL_MASK       :: BitMatrix(S, R)       ca_level == :aa ? (filter_N_upstream .== 1)
                                                         : supplier_cells            # legacy
                                         @assert CELL_MASK ⊆ AA_ACTIVE[s, AA_OF_ZE[l]]
CELLS_OF_SECTOR :: Vector{Vector{Int}}(S)
GOOD_S, GOOD_R, SR_TO_GOOD               rebuilt from CELL_MASK

── the T-COLUMN space (T parameters AND γ moments share it) ────────────────────
T_COL_DIM       :: Int                   ca_level == :aa ? n_AA : R
T_ACTIVE        :: BitMatrix(S,T_COL_DIM) ca_level == :aa ? AA_ACTIVE : supplier_cells
T_GATHER        :: Vector{Int}(R)        ZE → T column; identity under :ze, AA_OF_ZE under :aa
T_MASK          :: BitVector             vec(permutedims(T_ACTIVE))          # s-major
T_REF_REGION    :: Vector{Int}(S)        one reference column per sector, T[s,ref] ≡ 1
SECTOR_T_COLS   :: Vector{Vector{Int}}(S)
EMP_GAMMA_T     :: Matrix(T_COL_DIM, S)  block-5 target; Σ_{l∈a} emp_gamma_ls[l,s] under :aa

── granular targets and bounds ─────────────────────────────────────────────────
G_TARGET        :: Vector{Float64}(S)    G_K.csv at K = 0            → block-6 target
N_LO, N_HI      :: Vector{Int}(S)        from N_supplier_s

── geography and draws ─────────────────────────────────────────────────────────
LOG_DIST        :: Vector(R)             log(max(closest_plant_dist, 1.0))
DIST_BIN        :: Vector{Int}(R)        ∈ 0:N_REG, 0 = reference (d ≤ 50)
FE_GROUP        :: Vector{Int}(n_cells)  (s−1)*R_d + AA_OF_ZE[l]
U_DRAWS         :: Matrix(N_rho, n_cells)  ONE draw set, DRAW_METHOD (:sobol), drawn once
                                         — no pool, no replication, both modes identical

════════════════════════════════════════════════════════════════════════════════
PARAMETER VECTOR   (N_s is NOT in it)
════════════════════════════════════════════════════════════════════════════════
θ_c = [ Ω^L(1) | Ω^s(S) | A(R_d) | α(N_TAU) | T_block ]
      T_block = the active entries of an (S, T_COL_DIM) matrix, s-major, ref dropped

════════════════════════════════════════════════════════════════════════════════
ONE LOSS EVALUATION
════════════════════════════════════════════════════════════════════════════════
function loss(θ_c, W)

  ── 0. unpack ───────────────────────────────────────────────────────────────
  Ω^L, Ω^s, A, α, T_par = unpack_params(θ_c)     # T_par :: (S,T_COL_DIM), ref-normalised
  T[s,l] = T_par[s, T_GATHER[l]]                 # gather — the one new line; id. under :ze

  ── 1. ONE Ricardian solve, the SAME draws in both modes ────────────────────
  net = solve_network(θ_c; u_draws = U_DRAWS)
  #   net.linkages_flat :: (N_rho, n_cells)   1 = cell wins that variety somewhere
  #   net.z_flat        :: (N_rho, n_cells)   champion productivity
  #   value block computed here, over all draws                          ← D2

  ── 2. blocks 1–3 — value block, unchanged ──────────────────────────────────
  labor, industry, π_r = value_moments(net, θ_c)

  ── 3. block 4 — firm-level cloglog, run ONCE; N_s appears nowhere ──────────
  y[ρ,g] = 1 − net.linkages_flat[ρ,g]                    # not_supply
  X[ρ,g] = [ DIST_BIN onehot (or LOG_DIST) | log net.z_flat[ρ,g] ]
  w[ρ,g] = 1 / N_rho                                     # each cell carries total weight 1
  reg_coef, b_logz = cloglog_irls(y, X, w, fe = FE_GROUP[g])
  #   Prop. 1: conditioning on the firm removes N_s ⇒ block 4 is EXACTLY invariant to it
  #   b_logz should equal −θ — a free over-identifying test

  ── 4. block 5 — shares (AA-aggregated under --ca_level=aa) ─────────────────
  γ_aa[a,s] = Σ_{l : AA_OF_ZE[l]==a AND CELL_MASK[s,l]} γ[l,s]     for (s,a) ∈ AA_ACTIVE
  #   control cells ARE in the sum on both sides: they contribute 0 in the data and a
  #   positive γ in the model, and E[γ̂]=γ makes that an unbiased match, not a mismatch

  if !granular
      m = vcat(labor, industry, π_r, reg_coef, γ[mask])
      return (m − m̂)' W (m − m̂)
  end

  ── 5. q̂ from ALL draws — free of N_s (Lemma 2) ────────────────────────────
  q̂[g] = mean over ρ of net.linkages_flat[ρ, g]
  q̂[g] = clamp(q̂[g], 0.5/N_rho, 1 − 1e-12)

  ── 6. N̂_s — monotone integer bisection on the CLOSED FORM, no re-simulation ─
  for s in 1:S
      Ḡ(s,n) = mean over g ∈ CELLS_OF_SECTOR[s] of (1 − q̂[g])^n          # ↓ in n
      if     Ḡ(s,N_LO[s]) ≤ G_TARGET[s]   N̂[s], clamped[s] = N_LO[s], :lo
      elseif Ḡ(s,N_HI[s]) ≥ G_TARGET[s]   N̂[s], clamped[s] = N_HI[s], :hi
      else
          lo, hi = N_LO[s], N_HI[s]
          while hi − lo > 1
              mid = (lo + hi) ÷ 2
              Ḡ(s,mid) > G_TARGET[s] ? (lo = mid) : (hi = mid)
          end
          N̂[s], clamped[s] = argmin_{n ∈ {lo,hi}} |Ḡ(s,n) − G_TARGET[s]|, :none
      end
  end

  ── 7. block 6 — the count moment, the SAME closed form at N̂_s ─────────────
  G0[s] = Ḡ(s, N̂[s]) = mean over g ∈ CELLS_OF_SECTOR[s] of (1 − q̂[g])^{N̂[s]}
  #   This IS E[share of cells with K_ls = 0]: by linearity over cells the dependence
  #   between cells (they share variety draws) does not affect the mean. Unbiased AND
  #   noise-free — there is nothing to simulate and nothing to average.

  ── 8. assemble ────────────────────────────────────────────────────────────
  m = vcat(labor, industry, π_r, reg_coef, γ_aa[mask], G0)
  return (m − m̂)' W (m − m̂),  (N̂, clamped, b_logz, q̂)
end
```

`N̂_s(θ)` is a step function of the continuous parameters, so the profiled loss is
piecewise smooth with jumps. PSO / TikTak are derivative-free and unaffected. The FD
Jacobian can straddle a jump: at `θ̂` it is recomputed **holding `N̂_s` fixed**
(`compute_jacobian(...; hold_N_s=true)`, the default under `GRANULAR`) — the correct object
anyway, since `N̂_s` is locally constant with probability one — and `compute_moments` warns
if any perturbation would have moved it.

---

## 5. Where things live

### `load_parameters.jl`

* **Flags** (SECTION 2b): `GRANULAR::Bool`, `CA_LEVEL::Symbol`, with the
  `granular ⇒ ca_level == :aa` hard error.
* **Geography locals** (SECTION 2c), hoisted so the AA assertion can run early.
* **`G_K.csv`, `N_LO`, `N_HI`, `G_TARGET`** (SECTION 2d).
* **AA layer and the two index spaces** (SECTION 3b): `AA_OF_ZE`, `n_AA`, `AA_ACTIVE`,
  `CELL_MASK`, `T_COL_DIM`, `T_ACTIVE`, `T_GATHER`, `T_MASK`, `SECTOR_T_COLS`,
  `EMP_GAMMA_T`, `EMP_GAMMA_T_TILDE`, plus the per-sector composition table (ZE active / ZE
  control / ZE total / AA active / AA total).
* **Effective regression flags** `REG_INCLUDE_CONTROL`, `REG_INCLUDE_SIZE` (SECTION 6).
* **Moment vector** (SECTION 8): `compute_block_ranges` returns 5 or 6 ranges;
  `BLOCK_NAMES` and the labels (`T[sname-aaname]`, `gamma[sname-aaname]`, `G0[sname]`)
  follow.
* **Draws** (SECTION 9 / 9b): one `generate_draws(N_rho, n_good, DRAW_METHOD)` in both
  modes.
* **T starting values** (SECTION 10b): the Sinkhorn inversion `invert_T_from_gamma`, in the
  T-column space, in both modes. There is no gravity fallback — the old gravity guess is
  that inversion at `α = 0`.

### `model_CP.jl`

* `unpack_params` / `unpack_T_par` / `gather_T_to_ze` / `aggregate_gamma_to_T`.
* `concentrate_N_s(q̂) -> (N̂, clamped)` — the bisection.
* `fast_cloglog_regression` / `fast_weighted_regression`: `return_size_coef` returns the
  `log z` coefficient; the control/size exclusion is relaxed under `:aa`.
* `compute_moments`: AA γ aggregate, block 4 once, then `q̂` → `N̂_s` → block 6, all closed
  form. `N_fixed` pins `N̂_s` for the Jacobian.
* `granular_report`: `N̂_s`, clamps, `q̂`, `Ḡ_s(0)`, `b_logz`, `E[K] = N̂_s q̂`, `N^count_s`.
* `moments_to_vec` / `moment_blocks_tuple`: 5 or 6 blocks, never a hard-coded `1:5`.

### `profiling.jl`

* `invert_T_ge` iterates in the T-column space: gather → closed-form ZE γ → aggregate back.
  Square system, contraction inherited from the ZE-level version.

### `tools.jl`

* `sigma_beta_gamma_filename(; smm, aa)`, `reconcile_sigma_data` (β / γ / G split),
  `inference_moment_indices` / `inference_block_layout`.
* `compute_jacobian`: `hold_N_s`.
* `report_granular` and the `report.txt` count-moment section
  (`generate_dashboard_report(...; G0_, granular_info)`).

### `optimizer.jl` / `main.jl` / `run.sh`

* T block sized in the T-column space; block-coordinate stages unchanged in structure.
* Flags: `--granular=true|false` (default `false`), `--ca_level=ze|aa` (default `ze`).

---

## 6. Verification order

1. **V0 first.** `--granular=false --ca_level=ze` must reproduce the previous estimates.
   Note the code is not bitwise reproducible across runs (multithreaded BLAS in the
   least-squares solves, ~4e-13 on the loss), so compare against a *contemporaneous*
   baseline run, not a stored fingerprint.
2. The input-file gates V1 / V1a / V1b — they fail loudly at load time.
3. The structural gates V2 (bisection), V3 (AA Sinkhorn), V10 (block 4 exactly `N_s`-free),
   and `Ḡ_s(0)` monotone in `N_s`.
4. Then the reported diagnostics V6, V7, V9, and the measurements V8, V11, V12, V13.

`julia test/test_granular_aa.jl <industry> <n_coef> <n_tau> <granular> <ca_level>` runs
everything in groups 2–3 and reports the rest.

---

## 7. Data-side actions

The empirical β target and the β block of Σ are **correct as they stand** — the
`not_supply` outcome and the firm-level unit are the exact reduced form, so nothing needs
regenerating. The remaining fixes to the empirical script are listed in
`granular_validation.md` Part IV; the one that matters for consistency with the model is
replacing `nunique() == 2` by `transform("max") == 1`, so that `𝒜⁺_s` is defined once and
shared by the regression, the `G(K)` denominator and `CELL_MASK`.
