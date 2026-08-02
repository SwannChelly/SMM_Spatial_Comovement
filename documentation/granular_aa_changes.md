# Granular / attraction-area version — the complete set of modifications

What changed in the codebase to implement `documentation/plan_granular_aa.md`, why, and how it
was verified. Organised by theme rather than by commit; `CLAUDE.md` carries the same material
in chronological changelog form.

The single constraint everything is built around: **`--granular=false --ca_level=ze`
reproduces the previous estimator exactly.** That is gate V0, and it is checked after every
change.

---

## 1. The two flags

| Flag | Default | Const | Entry points |
|---|---|---|---|
| `--granular=true\|false` | `false` | `GRANULAR::Bool` | `main.jl` 11th positional arg, `run.sh` |
| `--ca_level=ze\|aa` | `ze` | `CA_LEVEL::Symbol` | `main.jl` 12th positional arg, `run.sh` |

`--granular=true` **requires** `--ca_level=aa` — a hard error in both `load_parameters.jl`
(SECTION 2b) and `run.sh`. Under ZE-level comparative advantage only the supplier cells are
estimated, so no cell can come out empty: the count moment has no content, and the estimator
would be fitting `T > 0` for regions whose observed `γ_ls` is 0. The reverse combination
(`false`, `:aa`) is the AA-level continuum — legitimate, and it only warns.

Output trees are separated: `reporting_<ind>[_profiled][_aa][_gran]_<optimizer>`, so the
legacy fit stays reproducible side by side.

---

## 2. The central abstraction: two index spaces

This is the change that touches the most code, and conflating the two spaces is the easiest
way to break the file.

**CELL space** — `CELL_MASK :: BitMatrix(S, R)`, the (sector, ZE) cells actually simulated.
Drives `n_good`, `GOOD_S/GOOD_R`, `SR_TO_GOOD`, `SECTOR_GOOD_INDICES/REGIONS`, `W_RS_FLAT`,
`CELLS_OF_SECTOR`, and the columns of `U_DRAWS`.

**T-COLUMN space** — width `T_COL_DIM`, the columns of the comparative-advantage matrix
**and** of the γ moment block. The CLAUDE.md invariant "the `T[s,c]` parameter column aligns
with the `γ[s,c]` moment row" is preserved verbatim, now over T columns rather than ZE.

| | `CA_LEVEL = :ze` | `CA_LEVEL = :aa` |
|---|---|---|
| `CELL_MASK` | `supplier_cells` | `filter_N_upstream .== 1` (control cells included) |
| `T_COL_DIM` | `R` | `n_AA` (= `R_downstream`) |
| `T_ACTIVE` | `supplier_cells` | `AA_ACTIVE` |
| `T_GATHER[l]` | `l` (identity) | `AA_OF_ZE[l]` |
| block 5 | `γ_ls` | `γ_{s,a} = Σ_{l∈a} γ_ls` |

Under `:ze` the two spaces coincide and every line reads exactly as before — which is why V0
holds. Under `:aa`, `unpack_params` scatters s-major into `(S, T_COL_DIM)`, ref-normalises,
then **gathers** `T[s,l] = T_par[s, T_GATHER[l]]`. Control cells thereby inherit
`T_{s,a} > 0` and become ordinary goods: **that is the endogenisation.**

New helpers in `model_CP.jl`: `unpack_T_par` (the parameter matrix before the gather — what
the Sinkhorn inversion, the optimiser's warm-start reduction and the T delta method iterate
on), `gather_T_to_ze`, `aggregate_gamma_to_T`.

Every place that previously indexed the γ/T flat layout with `R` now uses `T_COL_DIM`:
`load_parameters.jl` (T mask, moment mask, reference columns, `jacobian_param_indices`,
`GAMMA_REF_MAP`, labels), `tools.jl` (`_gamma_block_rs`, the Richardson sector map,
`reconcile_sigma_data`), `model_analytical.jl` (the γ adding-up AD test).

---

## 3. Granularity: what `N_s` actually touches

Only two objects, and both are **closed form** in `q̂`:

* `N̂_s`, by monotone integer bisection on `Ḡ_s(n) = mean_l (1−q̂_ls)^n` — `concentrate_N_s`;
* block 6, `Ḡ_s(0)` evaluated at `N̂_s`.

`q̂_ls` is a column mean of `linkages_flat` and is free of `N_s` (Lemma 2); block 4 carries no
`N_s` term (Proposition 1). So **no prefix of the draws is taken and no realised economy is
simulated**: every moment is computed once, on one draw set, exactly as in the continuum.

Consequences in code:

* one `generate_draws(N_rho, n_good, DRAW_METHOD)` in both modes — no pool, no replication
  count, no separate granular sampler;
* `N_rho` keeps its usual meaning and value (it additionally sets the precision of `q̂`, hence
  of `N̂_s`);
* simulation noise is handled uniformly across all six blocks by the draw design (`:sobol`).

`compute_moments` gains an `N_fixed` kwarg (threaded through `SMM` and `full_SMM`) that pins
`N̂_s`; `compute_jacobian` uses it via `hold_N_s` (default `true` under `GRANULAR`) because
`N̂_s(θ)` is a step function and a central FD could straddle a jump. `compute_moments` returns
the free bisection value as `N_hat_free` beside the pinned `N_hat`, but does **not** warn on a
difference: it runs on every worker for every FD evaluation, and the two differ for a second,
benign reason — an evaluation on a different draw set has a different `q̂` hence a different
`N̂_s`. `compute_jacobian` is the only place that can separate the two, and it does so once:
before the replications it evaluates `N̂_s` freely on `min(K, 5)` independent draw sets, pins
the per-sector median, and prints the observed spread. A spread above 10% of the pinned value
warns once — that is Monte-Carlo noise in `q̂`, not a jump, and the fix is a larger
`N_RHO_INFERENCE` (and `N_rho`, which sets `q̂` in the loss).

### Why not replicate

An earlier iteration simulated `R_REP` realised economies and averaged block 4 over them, to
match the binding function. It was dropped: it treated simulation noise on **one** block with
a bespoke device while every other block relied on the draw design, and it was inconsistent
with D2, which already puts the value block at the certainty-equivalent index computed on the
full draw set. The full argument is in `granular_validation.md` Part III. The cost of dropping
it — block 4 is `β(E[y])` rather than `E[β̂(N_s)]`, so the auxiliary cloglog's finite-sample
bias is not differenced out — is gate V12, to be reported with `α̂`.

Measured benefit beyond cost: with block 4 at the continuum limit the `log z` coefficient
recovers `−θ` to 0.4% on the synthetic set, against a 43% gap under the replication design
(which was that same finite-sample bias, on ten-variety prefixes).

---

## 4. Moment vector and inference plumbing

* `compute_block_ranges(...; n_G0)` returns **5 or 6** ranges. Block 6 is **appended, never
  inserted**, so every existing index into blocks 1–5 is untouched.
* `moments_to_vec(sim)` / `moment_blocks_tuple(m)` replace every hard-coded
  `vcat([vec(m[i]) for i in 1:5]...)[MOMENT_MASK]`, which would silently drop block 6. All
  call sites updated (`main.jl`, `main_gmm.jl`, `tools.jl`, `test/`).
* `inference_moment_indices()` / `inference_block_layout()` replace the open-coded
  `vcat(BLOCK_RANGES[4], BLOCK_RANGES[5])` and extend the β → γ ordering invariant to
  β → γ → G.
* `Weight_matrix_custom`, `build_step3_weight_matrix`, `compute_smm_inference` and the Hansen
  df follow automatically.

---

## 5. Σ files

`sigma_beta_gamma_filename(; smm, aa)` gains the `Sigma_aa_` **prefix**, defaulted from
`CA_LEVEL` (the γ level is what drives the file: its γ rows must match the γ moments). Eight
variants: `{ZE, AA} × {lpm, cloglog} × {N_REG>1, N_REG==1}`, with a trailing `_f` on the SMM
path.

Every file carries **three blocks, β → γ → G**, at a fixed `N_REG + n_γ + S`. So
`reconcile_sigma_data` splits on that fixed layout rather than searching: it reconciles the **γ
block only** against the active set, then reassembles, keeping the `G` rows only under
`GRANULAR`. `G` rows are never pruned. The pre-threshold γ-subset branch is retained for old
ZE-level files and errors out under `:aa`, where the file must already match.

---

## 6. New and re-encoded inputs

| File | Change |
|---|---|
| `filter_N_upstream.npy` | now **binary**. Control cell ⟺ `filter == 1 & X_rs == 0` (was status 2). `supplier_cells = (filter .== 1) .& (X_rs .> 0)` is unchanged, so the legacy path needed no edit |
| `attraction_area_linkages.npy` | new, `(R, R_downstream)` binary. Three load-time assertions; the decisive one is `argmax_col == CLOSEST_DOWNSTREAM_REGION` — the model's fixed effect and the empirical `A129_AA` grouping must be the SAME partition |
| `G_K.csv` | new. `G(0)` → `G_TARGET`; `N_supplier_s` → `N_HI`, and `N_LO = ⌈N_supplier_s / R_downstream⌉` |
| `Sigma_*` | β → γ → G layout, `Sigma_aa_` prefix under `:aa` (§5) |

`emp_gamma_ls` stays ZE-level in both modes; `EMP_GAMMA_T` (and `EMP_GAMMA_T_TILDE`, the
Sinkhorn row margin) is the derived moment/T-space target — the same matrix under `:ze`, the AA
aggregate under `:aa`. The AA sum runs over **every cell in `CELL_MASK`, control cells
included**, on both sides; the data contribute 0 there and the model a positive `γ`, which is
the unbiased match, not a mismatch.

---

## 7. The extensive-margin regression

`fast_cloglog_regression` and `fast_weighted_regression` gain `return_size_coef`, which appends
the `log z` coefficient to the returned vector — a free over-identifying test (it should equal
`−θ`, Proposition 1(c)).

The effective design flags are now consts:

| | `REG_INCLUDE_CONTROL` | `REG_INCLUDE_SIZE` |
|---|---|---|
| `:ze` | `INCLUDE_CONTROL` | `!INCLUDE_CONTROL` (the legacy coupling) |
| `:aa` | **`false`** | **`true`** |

Under `:aa` the control cells are already simulated goods, so appending them again as `y = 0`
rows would double-count them; and they carry a real `z`, so the size control is unconditional.
That combination is the firm-level reduced form of Proposition 1 exactly: cloglog on
`not_supply`, log distance and log own productivity, area × sector fixed effect. The
mutual-exclusion assertion is kept under `:ze`. `compute_regression_quadrature` (the analytical
path) mirrors both flags.

---

## 8. Simplifications made along the way

* **`T_gravity` removed.** The Sinkhorn inversion `invert_T_from_gamma` is the T starting value
  in both modes. The old gravity guess `T ∝ γ·w^θ` *is* that inversion at `α = 0` (τ ≡ 1 ⇒
  market access is common within a sector and cancels), so a missing `prior_alpha` falls back
  to `α = 0` rather than to a second code path. The init diagnostic keeps its comparison, with
  the `α = 0` inversion as baseline (`T_init_alpha0_vs_inversion_a*.png`).
* **`active_mat` renamed `supplier_cells`**, with its two roles documented: under `:ze` it IS
  the estimated set; under `:aa` it only defines `𝒜⁺`.
* **`gamma_threshold` block removed** — the pruning loop, its pre-threshold diagnostic and the
  `gamma_threshold.npy` write. Replaced by a per-sector composition table (ZE active / ZE
  control / ZE total / AA active / AA total) with warnings for sectors having zero or a single
  supplier cell.
* **`N_BLOCK`, `R_REP`, `N_POOL`, `U_POOL`, `POOL_WEIGHTS`, `GRANULAR_DRAW_METHOD`,
  `granular_prefix_rows`, `inference_draws`, `--n_rep` all deleted** — see §3.
* **`invert_T_ge` moved to the T-column space**: gather → closed-form ZE γ → aggregate back, so
  the system stays square and the contraction is inherited from the ZE-level version.

---

## 9. Reporting

* `report.txt` (`generate_dashboard_report(...; G0_, granular_info)`) gains a count-moment
  section: empirical vs fitted `Ḡ_s(0)` per sector, then the profiled `N̂_s` table with
  `N_LO`/`N_HI`, the clamp flag, `N^count_s`, and `b_logz` against `−θ`.
* `report_granular` (`tools.jl`) prints the same table to stdout and writes
  `granular_diagnostics.{txt,npz}`, wired into `main.jl` after Step 2 and Step 4.
* `granular_report` (`model_CP.jl`) is the underlying recomputation — nothing is smuggled out
  of the loss, so the optimiser's return contract is unchanged and no mutable state crosses the
  `pmap` workers.

---

## 10. Verification status

Run on Julia 1.10 against a synthetic dataset built for the purpose (the real data are not in
this environment).

| Check | Result |
|---|---|
| **V0** — legacy config | **bit-identical** on 8/8 artefacts (simulated + analytical moment vectors, loss, `invert_T_ge` T\*, draws, `MOMENT_MASK`, `T_MASK`, labels) |
| V1 / V1a / V1b — inputs | PASS (and a corrupted AA partition is correctly rejected) |
| V2 — `N_s` root-find | PASS: monotone, exact planted-integer recovery, clamps fire at both bounds |
| V3 — AA Sinkhorn round-trip | PASS, `max\|Δlog T\| ≈ 6e−15` |
| V10 — block 4 `N_s`-free | PASS, **exactly** 0 |
| `Ḡ_s(0)` monotone in `N_s` | PASS |
| V6 — `b_logz` vs `−θ` | −0.9958 against −1.0 (0.4%) |
| V7 — two routes to `N_s` | reported |
| V9 — bounds not binding | FAILS on the synthetic set only because its `G` target (0.6) is unreachable — the gate doing its job |
| End to end | both configurations load, evaluate the loss, build `W`, compute the Jacobian, run inference, and write `report.txt` |

**A caveat on V0.** The pre-existing code is *not* bitwise reproducible across runs: two runs
of unmodified `HEAD` differ by ~4e−13 relative on the loss, from multithreaded BLAS in the
least-squares solves. V0 must therefore be read against a **contemporaneous** baseline run, not
a stored fingerprint. Against a same-session `HEAD` run the new code is bit-identical.

**Not verified on the real data.** Run V0 first, then `test/test_granular_aa.jl`.

---

## 11. Open items

* **V12** — the binding-function bias, the acknowledged cost of the design (§3). Measure it at
  `θ̂` and report it with `α̂`, or add it as a local offset.
* **V8, V11, V13** — the untargeted count-curve fit, the value-block sensitivity to the
  realised price index, and the group-emptying rate. All need a fitted `θ̂`.
* **A.1** — the certainty-equivalent price index. `θ = 1` sits exactly on the boundary of finite
  variance for `(c*)^{1−ν_s}`; the value commented out at `load_parameters.jl:64` is `1.768`,
  which would make the whole issue second order. Worth revisiting independently.
* **A.2** — firm counts per cell. The firm-level unit gives every cell exactly `N_s` firms while
  observed counts vary by orders of magnitude; `SIREN` is deferred.
* **GMM** — `full_SMM(analytical=true)` asserts `!GRANULAR`: the analytical extensive margin is
  the FKG-approximated continuum object and has no count moment. The `:aa` γ aggregate *is*
  supported there.
