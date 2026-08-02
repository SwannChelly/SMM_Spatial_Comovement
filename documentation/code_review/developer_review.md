# Developer review — granular / attraction-area branch

Static review (no Julia). Paths relative to branch `claude/granular-aa-plan-98cmrn`;
baseline `main`.

> Produced by a review agent, 2026-08-02. Static review — no finding was confirmed by
> running the code. Findings are ordered most-severe-first. Items fixed since are marked
> **FIXED** inline with the commit.

**Headline.** The index-alignment work (T ↔ γ in the AA column space, block 6 *appended*,
`inference_moment_indices()` / `inference_block_layout()` / `moments_to_vec` as single choke
points) is done carefully — **no** transposed reshape or off-by-one was found in the
moment/parameter axes. The failures are elsewhere: one place where the Step-3 **loss** was
not migrated to the 6-block moment set while the **weight matrix** was, a stale
`N_rho`-vs-`N_s` assertion, two fresh-tree resume/filesystem bugs, and a conceptual gap —
`N_s` is profiled out of the loss but not out of the inference.

---

## 1. Step-3 loss uses β+γ while `W_step3` is built on β+γ+G — `BoundsError`

**CRITICAL** · `optimizer.jl:529` (with `tools.jl:1344`, `model_CP.jl:1652-1658`)

```julia
moment_blocks = moments_loss_gamma_beta ? [4, 5] : nothing   # optimizer.jl:529
```

`build_step3_weight_matrix` now sizes `W_step3` on `inference_moment_indices()` = blocks
4∪5∪6 (`N_REG+n_γ+S`). The Step-3 loss still asks for `[4,5]` (`N_REG+n_γ`). In
`loss_function` the guard that made this work on `main` (`size(W,1)==length(moment_indices)`
⇒ use as-is) now **mis-fires**, so `W_step3` (`n_gb×n_gb`) is indexed with **global masked
moment indices**. Block 4 starts at global index `S+R_downstream`; block 5 ends
`R_downstream−1` positions past `n_gb`.

*Failure:* `--granular=true` Step 3, first worker loss evaluation → `BoundsError` on
`W_step3`. If `parallel_SMM_safe`'s try/catch absorbs it into a sentinel score, PSO instead
runs to completion on a constant objective and `θ̂_2` = warm start (worse — silent).

*Fix:* `moment_blocks = moments_loss_gamma_beta ? (GRANULAR ? [4,5,6] : [4,5]) : nothing`,
ideally derived from the same helper as `inference_moment_indices()`.

> **FIXED** (commit `9bec188`). `moment_blocks` follows `GRANULAR`; `loss_function` now
> raises a descriptive error instead of subsetting a mismatched `W`. The economist review
> found this independently (its finding 2).

## 2. `concentrate_N_s` asserts `N̂_s ≤ N_rho`, and `N_rho` is 100 under GRANULAR

**CRITICAL** · `model_CP.jl:573-574`, `load_parameters.jl:208`

`N_HI[s] = N_supplier_s` (observed distinct supplier count, `load_parameters.jl:191`); the
bisection returns anything in `[N_LO, N_HI]`. The assert is a leftover from a "prefix of
draws = varieties" design that `load_parameters.jl` SECTION 9b (lines 752-771) explicitly
abandons ("`N_rho` … is NOT tied to `N_HI`, because no prefix of the draws is ever taken") —
and the same branch *lowered* `N_rho` to 100.

*Failure:* any sector with >100 distinct suppliers → `AssertionError` inside `compute_moments`
on a worker at the first `full_SMM`, surfacing as a `pmap` `CompositeException`.

*Fix:* delete the assertion (or check against `N_HI[s]`).

> **FIXED, by the opposite route** (commit `b99876d`, revised `59327ce`). At the author's
> direction `N_rho` is now `max(100, maximum(N_HI))`, making `N̂_s ≤ N_rho` a true invariant;
> the assert is kept as a live guard.

## 3. `run_step1 = false` on a branch whose output tree is brand new

**CRITICAL** · `main.jl:67` (was `true` on `main`), with `main.jl:204-205`, `tools.jl:910-913`

Granular runs write to a *new* folder `./reporting_<ind>_profiled_aa_gran_pso`
(`main.jl:153-155`), so `step1/` does not exist; line 204 calls `find_last_stage_folder`,
whose first statement is `readdir(base_folder)`.

*Failure:* first `--granular=true` run dies right after `load_parameters.jl` with
`SystemError: opening file ".../step1"`.

> **FIXED** (commit `9bec188`) — now a descriptive `error()` naming the missing directory and
> the switch to flip. `run_step1` itself is left as the author's knob.

Same class: `load_jacobian` flipped `false→true` (`main.jl:76`).
`compute_jacobian(load_existing=true)` only checks the **column count** (`tools.jl:1588`) and
merely **warns** when the saved `param_indices` differ (`:1594`) — a stale Jacobian of the
right width is accepted. Make that an error. **Still open.**

## 4. `run_profiled_inference` writes into `<step>/inference/` before anything creates it

**CRITICAL (GRANULAR only)** · `tools.jl:3412-3418`

The only `mkpath` for that directory is inside `compute_smm_inference` (`tools.jl:2057`),
called afterwards at `tools.jl:3435`; `grep -n mkpath` confirms nothing else creates
`step2/inference` or `step3/inference`.

*Failure:* first granular run, Step 2 — after paying for `build_step3_weight_matrix`
(K=10000) and the Jacobian, `NPZ.npzwrite` throws `SystemError`.

*Fix:* `mkpath(joinpath(output_folder,"inference"))` at the top of `run_profiled_inference`.

> **FIXED** (commit `9bec188`).

## 5. `N_s` is profiled out of the loss but not out of the inference

**MAJOR** · `tools.jl:1502,1527-1556`, `tools.jl:2135`, `tools.jl:1367`

(a) **Jacobian vs estimator.** `compute_jacobian` defaults `hold_N_s = GRANULAR` and pins
`N_fixed`, so the α column contains `∂Ḡ_s(0)/∂α` **at fixed N**. The estimator re-runs
`concentrate_N_s` at every evaluation (`model_CP.jl:1439`), so along the estimated manifold
`dḠ_s/dα = ∂Ḡ_s/∂α + (∂Ḡ_s/∂N)(dN*/dα) ≈ 0` — `N̂_s` is *chosen* to hit the target. This is
exactly the profiling issue the branch solved for `T` (`profile_T=true`) and not for `N_s`.
`G_α` then carries `S` rows of spurious information weighted by `1/Var(Ĝ_s)`, biasing
`se(α̂)` down.

(b) **`Σ_sim[G,G] ≈ 0` by construction**: `build_step3_weight_matrix` (`tools.jl:1367`) calls
`full_SMM` **without** `N_fixed`, so each replication re-profiles `N̂_s` onto the target.
Compounds (a).

(c) **Hansen df**: `df = N_mom − p` (`tools.jl:2135`) with `p = N_TAU`; the `S` count moments
are consumed by the `S` profiled `N̂_s`, so `df` is overstated by `S` and the test
under-rejects.

*Fix:* either drop block 6 from `W`/`G`/residual (then finding 1 disappears too), or use the
total derivative (`hold_N_s=false`) and set `df = N_mom − p − S`.

> **STILL OPEN — and the largest remaining item.** Note the economist review reaches the same
> diagnosis but prefers the first remedy (drop block 6 from the inference subsystem) where
> this review offers both. The two remedies are not equivalent: dropping block 6 also reverts
> the finding-1 fix.

## 6. `reconcile_sigma_data` assumes every Σ file has `S` trailing G rows; the compat branch is dead

**MAJOR** · `tools.jl:1257-1332`

`n_gamma_file = n_total − N_REG − S`; the "pre-granular file with no G block" branch fires
only when `n_gamma_file ≤ 0`, i.e. `n_γ ≤ S` — **unreachable** for real data. A legacy
`Sigma_beta_gamma_cloglog_f.npy` is split as γ = first `n_γ−S` rows, G = last `S` γ rows,
then fails with a misleading "matching neither the active count … nor the pre-threshold
count".

*Silent case:* a file whose γ block is exactly `n_gam + S` rows passes
`n_gamma_file == n_gam` and is used with the **wrong** γ rows. Nothing checks content, only
counts.

*Failure:* `--granular=false --ca_level=ze` — advertised at `run.sh:20,42` as "the legacy
model (V0 reference)" — hard-errors in Step 2 against the existing on-disk Σ.

*Fix:* accept `n_total ∈ {N_REG+n_γ, N_REG+n_γ+S}` and branch on which matched. **Still open.**

## 7. `N_rho=100` makes block 6 exponentially noisy; the `q̂` clamp is draw-count dependent

**MAJOR** · `load_parameters.jl:208`, `model_CP.jl:1437,1458`

`∂log Ḡ_s/∂q̂ = −N/(1−q̂)`. With `N̂_s` in the hundreds and
`s.e.(q̂) ≈ √(q(1−q)/100) ≈ 0.05`, one s.e. in `q̂` moves `Ḡ_s` by orders of magnitude.
`load_parameters.jl:770` says to *raise* `N_rho` under GRANULAR; the branch lowered it
1000→100.

Worse, the clamp floor is `0.5/n_draw`: the loss uses `U_DRAWS` (`n_draw=100` ⇒ `5e-3`)
while `Σ_sim`, the Jacobian and `compute_N_s_jacobian` use `N_RHO_INFERENCE=4000`
(⇒ `1.25e-4`). For a zero-win cell the two differ by `e^{−N/200}` — the block-6 moment being
*fitted* is not the one whose covariance is in `Ω` nor the one being differentiated.

> **PARTIALLY ADDRESSED.** `N_rho` is now `max(100, maximum(N_HI))`. The loss-vs-inference
> floor mismatch **remains** and is widened by the sampler split (optimisation `:sobol`,
> inference `:mc`); setting `--n_rho_inf` equal to `N_rho` closes it.

## 8. Per-sector `M^s` identification screen is wrong under `:aa`

**MAJOR (print-only)** · `tools.jl:4029-4038`

`regs = SECTOR_GOOD_REGIONS[s]` are **ZE** indices; `ref = T_REF_REGION[s]` is an
**attraction-area** index under `:aa`. `findall(!=(ref), regs)` therefore drops (usually)
nothing, and `M^s` is built in ZE space while the parameters are AA-level. The printed
"sector s: M λ_min/λ_max" is not the identification of the estimated T block — and it is
precisely the diagnostic a reader will use to judge AA-level T.

> **FIXED.** `M^s` is now built in the T-PARAMETER column space: the cell-level
> bilateral shares are collapsed onto `SECTOR_T_COLS[s]` via `T_GATHER` before forming
> `M = Σ_dr ω_dr (diag g − g g')`, and the reference row is dropped by position within
> that column set. The multinomial structure survives the grouping precisely because
> `T` is constant within an attraction area, so the aggregated shares give the right
> block. Under `:ze`, `T_GATHER` is the identity and the result is the previous
> cell-level matrix exactly. The printed line now names the space and the free count.

## 9–13. Smaller items

- **`compute_N_s_jacobian` exact-zero check** (`tools.jl:818-823`): `D[i,:] != zeros(K)` is
  `true` for any `NaN`, turning one non-finite moment into an `error()` that blames the wrong
  thing and aborts Step 2/4. **MINOR**
- **Seed collision** (`main.jl:309` vs `:518`): Step-2 `N_s` Jacobian uses
  `MersenneTwister(7_000_001…010)`; Step-4 profiled α Jacobian uses `7_000_001…050`. The
  comment at `:518` claims disjointness from "step2 (6e6)" and overlooks the 7e6 block it
  introduced. **MINOR**
- **Wasted probe** (`tools.jl:1527-1556` sits *before* the `load_existing` early return at
  `:1576`): 5 extra `pmap` `granular_report` calls per invocation, discarded. **MINOR**
- **Stale defaults**: `profile_T` defaults to `true` in `main.jl:87-88` vs `false` in
  `run.sh:57`; `n_rho_inference` is 1000 (`main.jl:95`) / 4000 (`run.sh:58`) / 10000 (help,
  `run.sh:36`); `include_control` flipped `true→false` and is inert under `:aa` anyway
  (`load_parameters.jl:595-596`) though `run.sh --controls` doesn't say so. **MINOR**
- **Missing assert**: Step 2 lacks the `size(Omega_step2,1)==length(gb_indices)` check Step 4
  has (`main.jl:472`). **MINOR**
- `tools.jl:1849-1854` `param_sector` uses `T_first = 1+S+R_downstream` (no `+N_TAU`) ⇒ α
  column mis-attributed in the Richardson print (pre-existing, print-only).
- `tools.jl:2958-2963`/`3095-3115`: `V_T`/`V_N` mix `Var_alpha_sw` with `Cov` terms built
  from `P = Var_alpha_eff·G'W`; the "PSD by construction" claim holds for a consistent joint
  covariance, not this mixture.
- `model_analytical.jl:416-436` indexes `γ_sim[r,s]` with ZE `r` while under `:aa` the block
  is `(T_COL_DIM,S)` ⇒ `BoundsError`. Inert (GMM-only, `run.sh:148-152` forces `ze`); add
  `@assert CA_LEVEL === :ze`.

---

## Index alignment — verified clean

`T_MASK == T_MASK_MOMENT == vec(permutedims(T_ACTIVE))` over `(S,T_COL_DIM)`
(`load_parameters.jl:362-365`), with flat position `(s-1)*T_COL_DIM + c` used consistently in
the MOMENT_MASK γ loop (`:680,:689`), the excluded-reference computation (`:1071`),
`PARAM_LABELS` (`:1231-1234`), `_gamma_block_rs` (`tools.jl:2501-2508`) and the Richardson
moment→sector map (`tools.jl:1842`). `unpack_params`/`unpack_T_par`/`gather_T_to_ze`
(`model_CP.jl:421-499`) scatter → `permutedims` → ref-normalise → gather, and every consumer
reshapes `vec(T_mat)` as `(S,R)`. All reduced-T producers emit s-major in the *parameter*
space (`optimizer.jl:194,204,262,566`, `main.jl:175`, `profiling.jl:211,247`) — notably
`optimizer.jl:262` was correctly switched from `reshape(params_dict[:T],S,R)` (ZE-gathered)
to `unpack_T_par`, the one spot where a transposed reshape would have been invisible.
`n_T == n_γ` still holds under `:aa`. β→γ→G ordering is respected in
`inference_moment_indices()`, `inference_block_layout()`, `build_step3_weight_matrix`,
`reconcile_sigma_data`, both `main.jl` sites and `main_gmm.jl`. The `V_N` assembly dimensions
check out (`L` is `S×(S+N_TAU+n_γ)`, `B_mat = S×n_T · n_T×n_γ`).
`compute_block_ranges(...; n_G0)` appends rather than inserts.

---

## Sobol independence — verdict

`generate_sobol_draws` (`model_CP.jl:312-336`) builds **one Sobol net of dimension
`d = |SECTOR_GOOD_INDICES[s]|` per sector**. Byte-identical to `main` — pre-existing, but
`:sobol` is the default in `main.jl:57` and `run.sh:55`.

**(i) Within a sector — not independent, but the right kind of dependence, with a caveat the
granular default breaks.** A `d`-dim Sobol net is deliberately not i.i.d.: it is a
`(t,m,s)`-net whose coordinates are jointly equidistributed; pairwise correlation is `O(1/N)`
and for the 2-D projections the net resolves, it is *better* than i.i.d. for integrating
`min_r c_r`. But it is a well-known pathology that for small `N` the 2-D projections of
high-index dimension pairs degenerate into near-linear stripes. This branch sets
`N_rho = 100` while `d` grew (under `:aa`, `CELL_MASK` is `filter==1`, so control cells are
now simulated — `load_parameters.jl:334`). `N=100` in a `d`-of-order-hundreds net is exactly
the bad regime, so spurious cross-region dependence in the Ricardian `min` is a **live
risk**. Do not run granular at `N_rho=100` with `:sobol` without check 7 below.

> Independently measured on a reimplementation of this design: cross-cell correlation of
> 0.24 at `d=22, N=100` versus 0.005 at `N=1024`, and the argmin-share error versus the
> closed form improves only 0.83× over plain MC at `N=100` (versus 0.50× at `N=256`).
> Confirms the verdict. `N_rho` has since been raised to `max(100, maximum(N_HI))`.

**(ii) Across sectors — genuinely, deterministically dependent. A real defect.**
`sobol_scrambled_net` regenerates `QuasiMonteCarlo.sample(N, zeros(d), ones(d),
SobolSample())` on every call, so sector `s` dim `k` and sector `s'` dim `k` are the **same
base column** `m_i`; only the digital shift differs:
`u_i = m_i ⊻ a`, `v_i = m_i ⊻ b` ⇒ `u_i ⊻ v_i = a ⊻ b`, a **constant for every i**. In the
realised design `v` is a deterministic bijection of `u`; Pearson correlation ≈ 0 (so a
`max|cor|` gate passes), but the 2-D point set lies on a 1-D XOR curve. This does **not**
touch the within-sector `min` (blocks 4, 5, 6 are safe) but it does touch anything nonlinear
across sectors: `P_r = (Σ_s Ω^s P_sr^{1-ν})^{1/(1-ν)}` (`model_CP.jl:846`) and downstream
`c_r`, `Y_r`, `π_r`, `agg_labor_share`, `agg_industry_share` — i.e. the **Step-1** fit that
pins `Ω̂^L, Ω̂^s, Â` for the rest of the pipeline.

*Fix:* draw one net of dimension `n_good` and slice per sector, or give each sector its own
disjoint Sobol dimension range.

> **PARTIALLY ADDRESSED** (commit `59327ce`): inference draws (Σ_sim, all Jacobians) now use
> `:mc` via a new `INFERENCE_DRAW_METHOD`, so Σ_sim no longer inherits the coupling. The
> **optimisation** draws still use `:sobol` and still carry it — the underlying defect in
> `sobol_scrambled_net` is unfixed.

**(b) Digital shift — correct.** `m ⊻ shift` is measure-preserving on the dyadic grid,
preserves `(t,m,s)`-net equidistribution, makes each point marginally `U[0,1]^d` (so RQMC is
unbiased), and maps Sobol's all-zero first point to `shift/2^52` — the origin is dissolved.
`u=0` needs `m == shift` (prob `~N/2^52`) and `clamp(·, eps(), 1-eps())` (`:327`) caps it.
Caveat: a digital shift is **not** an Owen scramble — it randomises the net without breaking
higher-order structure, which is why (ii) survives it.

**(c) Flat weights / `[rho, good]` — verified.** `W = fill(1/N_rho, N_rho, n_good)` (`:334`)
⇒ winner-weight shortcut exact. All consumers index `[rho,g]`:
`model_CP.jl:842,871,1042,1250`, `main.jl:690`. No `[rho]`-only access remains.

**(d) Granularity — no draw-semantics change.** `q̂` is a column mean of `linkages_flat` and
`Ḡ_s(0)` is closed form (`model_CP.jl:1429-1461`); no prefix, no replicated economy.
`N_rho_eff = size(u_draws,1)` is honoured (`:732-736,:1006,:1225`) and
`n_draw = size(linkages_flat,1)` (`:1430`) — correct. What breaks is findings 2 and 7.

**(e) Determinism / replication independence — correct.** `randomise=false` ⇒
`MersenneTwister(42)` master (`:318`), identical across workers ⇒ PSO objective
deterministic. `randomise=true` ⇒ shifts from the caller's `MersenneTwister(k)` ⇒ independent
designs per `Σ_sim`/Jacobian replication.

---

## Inference, Steps 3–4

**Correct:** `W = (Σ_data+Σ_sim)^{-1}` is built on `inference_moment_indices()` and every
consumer uses the same index vector and ordering; `main.jl:472` asserts the Ω size; the
Jacobian column restriction matches `param_indices` and the α-label count is asserted
(`:255`, `:475`); the profiled path routes only α through `compute_jacobian(profile_T=true)`
with `T` following via `profiled_theta`, and `run_profiled_inference` asserts the α column
count (`tools.jl:3380`); `_dTstar_dalpha`'s
`T_red_pos = gb_param_idx[T_pos] .- (1+S+R_downstream+N_TAU)` is right and range-asserted; no
stale intra-run reload.

**Broken by the granular flags:** findings 1, 5, 6, 7, 10. One further pre-existing mismatch
amplified here: the Hansen residual uses `sim_vec` at `U_DRAWS` (`N_rho=100`) while `Ω` is
estimated at `N_RHO_INFERENCE=4000` — second-order for blocks 4–5, exponential for block 6.

---

## `main.jl` line by line

**Args (50-136).** Twelve slots; `run.sh:209` passes exactly
`ind N_COEF N_TAU 10000 DRAWS OPTIMIZER PROFILE_T N_RHO_INF REG CONTROLS GRANULAR CA_LEVEL`,
lining up one-for-one with
`main.jl:50,51,52,53,57,62,87,95,102,108,115,118`. ✓ Validation present for
draws/optimizer/reg/ca_level/n_rho_inference/n_coef/n_tau; `granular⇒ca_level==:aa` enforced
twice (`run.sh:133`, `load_parameters.jl:111`). Default disagreements listed above.

**`run_2x2_test` auto-disable (131-136).** Correct and correctly placed before Step 2; one of
three layers of the same guard (`model_analytical.jl:257`, `model_CP.jl:1685`,
`tools.jl:1509`) — good defence in depth.

**Output folder (148-156).** `./reporting_<ind>[_profiled][_aa][_gran]_<optimizer>`,
reproduced exactly by `run.sh:185-190,239-244` ⇒ granular cannot overwrite legacy artefacts ✓
(and this is why finding 3 bites).

**`@everywhere` broadcasts.** All model/tool/optimizer files are `@everywhere include`d at
38-46 *before* `include("load_parameters.jl")` at 159, so `compute_block_ranges`,
`distance_bin`, `generate_draws`, `moments_to_vec` resolve on the master and every new
`@everywhere const` (`GRANULAR`, `CA_LEVEL`, `T_COL_DIM`, `T_ACTIVE`, `T_GATHER`,
`CELL_MASK`, `CELLS_OF_SECTOR`, `G_TARGET`, `N_LO/N_HI`, `EMP_GAMMA_T{,_TILDE}`,
`SECTOR_T_COLS`) reaches workers before the first `pmap`. ✓ Two globals used inside functions
are **master-only locals**, never broadcast: `_sector_names` (read by
`screen_T_identification`, `tools.jl:3866`) and `input_folder` (read by
`generate_dashboard_report` `:262` and `reconcile_sigma_data`) — fine today, `UndefVarError`
if either is ever `pmap`ed. Pre-existing.

**Resume guards.** `run_step2=false` loads only `step2/W_step3.npy` (370); Step 4 re-reads
`Omega`/`Sigma_data` itself (455-456, 520) — consistent. `run_step3=false` + `run_step4=true`
reads `step3/theta_hat_2.npy` (408) with no `isfile` guard.

**Dead code.** 410-412 re-flattens `theta_hat_1`, already done at 207-209. The `else` branches
at 310-334 and 536-559 (joint α+T inference + `compute_T_delta_inference`) are unreachable as
shipped, since `profile_T` is `true` for every `run.sh` invocation and defaults to `true` when
absent — the non-profiled path is untested on this branch.

**Post-hoc (606-699).** Runs unconditionally *after* Step 4 and calls
`run_reporting(step1,…)`, which needs `step1/` — same fresh-tree failure as finding 3, and it
is outside every `run_step*` guard. `N_rho_out = size(network.linkages_flat,1)` (653) is
correct; the "pool width under GRANULAR" comment is stale (there is no pool). `siren_map`
build (670-679) and read (684) use the same `(l,s,rho)` key ✓.

---

## Consistency checks I recommend running

1. In `loss_function`, assert `size(W,1) == length(moment_indices) || size(W,1) ==
   length(vec(emp))` with sizes in the message — turns finding 1 into a one-line diagnosis.
   *(Done, commit `9bec188`.)*
2. Startup gate: `@assert sort(vcat([collect(BLOCK_RANGES[b]) for b in moment_blocks]...)) ==
   sort(inference_moment_indices())`.
3. Before `reconcile_sigma_data`, assert
   `size(Σ,1) ∈ (N_REG+n_γ_active, N_REG+n_γ_active+S, N_REG+n_γ_pre+S)` and print which
   branch matched — catches finding 6 and any silent mis-split.
4. Print `maximum(N_HI)` next to `N_rho` at load time and assert `all(N_LO .<= N_HI)`.
   Confirms/refutes finding 2 in one run.
5. **`N_s` envelope check:** at `θ̂`, compute `Ḡ_s(0)` with `N_fixed = N̂` and with `N` free
   at `α·exp(±δ)`. Free-`N` difference ≈ 0 while pinned ≠ 0 ⇒ finding 5 is real; report the
   ratio.
6. Print `N_mom`, `p`, and `p + (GRANULAR ? S : 0)` side by side in `inference_summary.txt`.
7. **Intra-sector Sobol projection gate:** for the 3 largest sectors at production `N_rho`,
   over all intra-sector column pairs compute (a) `max|cor|` (expect `≲3/√N`) and, more
   informative, (b) a 2-D equidistribution statistic — max over a `⌈√N⌉×⌈√N⌉` grid of
   `|#points in cell − N/⌈√N⌉²|`. Run at `N_rho ∈ {100, 1000, 8192}`.
8. **Cross-sector XOR gate:** for `g` in sector 1 and `g'` in sector 2 sharing a Sobol
   dimension index, check
   `length(unique(floor.(UInt64,U[:,g].*2^52) .⊻ floor.(UInt64,U[:,g'].*2^52))) == 1`.
   Returns `1` ⇒ coupling confirmed exactly. The existing `test_price_alignment` `max|cor|`
   gate (`main_gmm.jl:430`) will **not** catch it.
9. Evaluate `Ḡ_s(0)` at the same `θ̂` with `N_rho` vs `N_RHO_INFERENCE` draws and print the
   per-sector ratio — confirms finding 7.
10. `load_parameters.jl:338-339` and `test/test_granular_aa.jl:147` use
    `[(s,l) for s in 1:S, l in 1:R if cond]` — the only two occurrences of that form in the
    repo, in code the changelog marks untested. Run
    `julia -e '@assert [(i,j) for i in 1:2, j in 1:2 if i==j] == [(1,1),(2,2)]'`; if it
    errors, rewrite as `for s in 1:S for l in 1:R if cond`.
11. Inside `screen_T_identification`, assert `T_REF_REGION[s] ∈ SECTOR_T_COLS[s]` and that
    `regs`' index space matches — surfaces finding 8 immediately. *(Fixed; the screen now
    builds `M^s` in the T-column space directly.)*
12. **Fresh-tree smoke test:** `rm -rf reporting_<ind>_profiled_aa_gran_pso && ./run.sh <ind>
    --n_coef=4 --n_tau=1 --profile_T=true --granular=true --ca_level=aa` with tiny
    `n_particles`/`max_iter`. That single run exercises findings 1, 2, 3, 4 and 6 in order.
