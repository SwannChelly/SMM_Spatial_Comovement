# Economist review — granular / attraction-area estimator

Theory: `documentation/finite_sample2.tex`. Code: branch `claude/granular-aa-plan-98cmrn`.
Focus: what `--granular=true --ca_level=aa` switches on.

> Produced by a review agent, 2026-08-02. Static review — Julia was not available in the
> review environment, so no finding here has been confirmed by running the code. Findings
> are ordered most-severe-first. Where a later check contradicted a finding, a **VERIFIED**
> note has been added inline.

**Overall.** The structural translation is largely faithful. The Fréchet draw is exactly
eq. (2) with `T` as scale; `T_{sl}=T_{s,a(l)}` is a genuine *gather* with independent
per-ZE draws (Assumption 2, not a collapse of ZE); `q̂_ls` is the win-somewhere column mean
and is genuinely `N_s`-free (Lemma 2); block 4 is exactly `N_s`-invariant (Prop. 1(b)) and
*asserted* to be so; the cloglog runs on `not_supply` giving `+θα` / `−θ` per Table 2; the
price index is the CES **average**, i.e. the Appendix-B gauge; and the γ normalisation is
consistent model-side and target-side under `:aa`. The problems concentrate in the
numerical implementation of the count moment and in the draw count that was cut to pay for
the ten-fold increase in cells. Two are arithmetic certainties. One aborts a granular run
at Step 3.

---

## 1. CRITICAL — the `q̂` floor makes the count moment mechanically unable to identify `N_s` above ≈18

**Theory.** Eqs. (25)/(28): `Ḡ_s(0)=|L⁺_s|⁻¹Σ_l(1−q_ls)^{N_s}` evaluated at `q̂`;
`granular_validation.md:130-133` claims it is "unbiased **and** noise-free".

**Code.** `model_CP.jl:1437`

```julia
q_hat[g] = clamp(acc / n_draw, 0.5 / n_draw, 1.0 - 1e-12)
```

with `n_draw = N_rho = 100` (`load_parameters.jl:208`); used at `model_CP.jl:1458-1460` and
in `concentrate_N_s` (`model_CP.jl:556`).

**What differs.** Every cell's `q̂` is floored at `0.5/N_rho = 0.005`, so **identically, for
every parameter vector**, `Ḡ_s(0) ≤ (1−0.5/N_rho)^{N_s} = 0.995^{N_s}`. With the paper's own
target `Ḡ_s(0) ≈ 0.915` (§5.1: 1062/1161 cells empty), the moment is attainable only for
`N_s ≤ ln(0.915)/ln(0.995) = 17.7`. `concentrate_N_s` can never return `N̂_s > 18`, whatever
the model says. Since `N_LO = ⌈N^obs_s/N_d⌉` (eq. 22) will typically exceed that, the
bisection takes the `Gs(lo0) <= tgt` branch (`model_CP.jl:565-566`) and returns
`N̂_s = N_LO` with `clamped = :lo` in every sector at every θ. V9 then fails for a purely
numerical reason that will be misread as the "rejection signal for the mechanism" the doc
says a clamp means.

> **VERIFIED — the stated consequence does NOT hold for this dataset.** Checked against the
> actual run: `G_TARGET = [0.8929, 0.9266, 0.9231, 0.865, 0.9223, 0.8225, 0.7681, 0.9444,
> 0.9125, 0.9444]` and `N_LO = [2, 3, 1, 14, 3, 13, 11, 2, 12, 4]`. The per-sector ceiling
> `ln(Ḡ_target)/ln(1-0.5/N_rho)` is 11–53 at `N_rho=100`, which is *above* every `N_LO`, so
> **no sector is pinned at `N_LO` by the floor**. The mechanism is real but its consequence
> here is different: **truncation from above**. At `N_rho=100`, sector 4 could never exceed
> `N̂≈29` against `N_HI=291`, and sector 6 `N̂≈39` against `280` — the upper half of the
> bisection range was unreachable, so a `:hi` clamp could not occur and the
> over-identification check was silently one-sided. `N_rho` has since been raised to
> `max(100, maximum(N_HI))`, which lifts the ceilings to ~34–154 but does not fully close
> the gap for sectors 4 and 6.

Even without the floor the plug-in is biased: `f(q)=(1−q)^N` is convex, so
`E[(1−q̂)^N] ≈ (1−q)^N[1+N(N−1)q/(2m(1−q))]`, `m=N_rho`. At the calibration (`Nq≈0.089` to
hit 0.915) the relative bias is `≈N(Nq)/(2m) ≈ N/2250` at `m=100` — ~4.5% of `Ḡ` at
`N=100`, which alone moves `N̂` by tens of percent. The doc's "noise-free" claim was
verified at **20,000** draws (`granular_validation.md:137`); production runs at 100.

**Consequence.** `N̂_s` is determined by `N_rho` and the smoothing floor, not by the data.
Everything resting on it (V7, V9, the counterfactual love-of-variety channel) is affected.

**Fix direction.** Use the exactly unbiased U-statistic: with `k_g` wins out of `m`,
`E[C(m−k_g,N)/C(m,N)] = (1−q)^N` for `N ≤ m−k_g`; drop the floor; raise `N_rho` so
`1/N_rho ≪ min_l q_ls`.

---

## 2. CRITICAL — Step 3 crashes under `GRANULAR`, and block 6 is silently excluded from the efficient objective

`optimizer.jl:529`: `moment_blocks = moments_loss_gamma_beta ? [4, 5] : nothing`.
`main.jl:384-391` calls Step 3 with `weight_matrix = W_step3`, built over
`inference_moment_indices()` = β→γ→**G** (`tools.jl:1196-1198`, `1344`), so
`size(W_step3,1)=N_REG+n_γ+S` while `moment_indices` has length `N_REG+n_γ`. In
`loss_function` (`model_CP.jl:1656-1657`):

```julia
if size(W, 1) != length(moment_indices)
    W = W[moment_indices, moment_indices]
```

Under `!GRANULAR` the sizes coincide and `W` is used as-is. Under `GRANULAR` they differ by
`S`, so the **restricted** `W_step3` is indexed with **global masked-vector** positions. The
last index of `BLOCK_RANGES[5]` is `S+R_down−1+N_REG+n_γ`, exceeding `n_gb` by
`R_downstream−1 > 0` ⇒ `BoundsError` on the first Step-3 loss evaluation.

> **VERIFIED and FIXED** (commit `9bec188`). The developer agent found this independently.
> `moment_blocks` now follows `GRANULAR`, and `loss_function` raises a descriptive error
> rather than subsetting a mismatched `W`.

**Beyond the crash.** (a) Even fixed, dropping G while keeping the *full* `Ω^{-1}` is not
the efficient weight for the subset — that is `(Ω_{45,45})^{-1}`, not the (4,5) block of
`Ω^{-1}`. (b) Eq. (29) defines `N̂ = argmin_N r'Wr` over the full weighted criterion; the
code instead sets `N̂` by per-sector *exact matching* of `Ḡ_s(0)`. These coincide only if
`W` is block-diagonal between G and (β,γ) — which Appendix C explicitly says it is not ("a
single joint bootstrap is needed to estimate the cross-covariance blocks"). So the
implemented estimator is not the profiled extremum estimator of eq. (29). **Still open.**

---

## 3. MAJOR — `N_rho` cut to 100 reintroduces the granular price-index bias the paper assumes away

**Theory.** §4.3 + Table 3: the estimator adopts the **certainty-equivalent** index; the
realised finite-variety index exceeds it by **+11.1% at 25, +6.5% at 100, +1.0% at 1000**,
with `θ=1, ν_s=1.5` exactly on the infinite-variance boundary.

**Code.** `load_parameters.jl:208`: `n_rho_local = granular_local ? 100 : 1000`.
`solve_network` computes `P_sr = [Σ_ρ w_ρ p_ρ^{1−ν_s}]^{1/(1−ν_s)}` as a **sample** average
over those draws (`model_CP.jl:838-843`) — a convex transform (`1/(1−ν_s) = −2`) of an MC
mean. Table 3 *is* the bias of this estimator as a function of the draw count. At
`N_rho=100` the simulated `P_sr` sits ≈6.5% above the certainty-equivalent index the model
is defined on — exactly the Jensen term §4.3 assumes away — and with infinite variance it is
also very noisy. The model now has two inconsistent `N`s: `N_rho=100` in the value block,
`N̂_s` in the count block.

> **PARTIALLY ADDRESSED.** `N_rho` is now `max(100, maximum(N_HI))` = 291 on the `auto`
> sample, which by Table 3 puts the Jensen gap between the +6.5% (at 100) and +1.0% (at
> 1000) rows. The two-inconsistent-`N`s point stands.

Three riders on the same line:

- `granular_validation.md` and `load_parameters.jl:768` both assert "`N_rho` keeps its usual
  meaning and value". It does not.
- `N_RHO_INFERENCE` defaults to **1000** (`main.jl:95`), so `Σ_sim` and the Jacobian evaluate
  a *different* `Ḡ_s(0)` and `P_sr` from the ones the optimiser minimised. Under `!GRANULAR`
  that only changed noise; under `GRANULAR` it changes the moment's expectation (finding 1).
- Under profiling, `invert_T_ge` matches the **analytical continuum** γ
  (`profiling.jl:144-145`) while the loss compares the **simulated** γ at 100 draws — the "γ
  juste-identified along `T*(α)`" basis of the profiled-α inference degrades by √10 in noise.

---

## 4. MAJOR — under `:aa`, `:sobol` runs at `N_rho=100` in a per-sector dimension of ~116

`model_CP.jl:312-336`: one Sobol net per sector, `d = length(SECTOR_GOOD_INDICES[s])`,
`N = N_rho` points, digital shift only. Default `:sobol` (`main.jl:57`, `run.sh:55`). Under
`:ze`, `d` was the number of *supplier* cells (~14). Under `:aa`, `CELL_MASK` is every
`filter==1` cell, `n_good` goes ~142 → ~1161 (`granular_validation.md:100`), so `d` per
sector is order 100+ with `N=100` — fewer points than dimensions, and not a power of two
(CLAUDE.md notes the `(t,m,s)`-net property is sharp only at `N=2^m`). High Sobol dimensions
are strongly correlated at small `N`. The entire justification for the per-sector net was to
keep dimension low *because the Ricardian `min_r c_r` lives within a sector across its
regions*, and the 2026-06-18 changelog documents that correlated columns collapse Ricardian
selection and bias every min-coupled moment — now including `q̂`, hence blocks 5 and 6.

> **PARTIALLY ADDRESSED.** `N_rho` raised to 291. An independent measurement of the design
> (one net per sector, per-dimension digital shift) found cross-cell correlation of 0.24 at
> `d=22, N=100` versus 0.005 at `N=1024`, so the concern is real and 291 only partly
> relieves it. Separately, the developer review found a *cross-sector* coupling defect in the
> same routine (see that report, §2a-ii); inference draws have since been switched to `:mc`.

---

## 5. MAJOR — the `reg_coef` target file is not keyed on the regression design, but the design flips with `--ca_level`

Prop. 1(a): cloglog on "does not supply", on log distance **and log own productivity**,
area×sector FE. The effective flags flip with `CA_LEVEL` (`load_parameters.jl:595-596`):

| | `REG_INCLUDE_CONTROL` | `REG_INCLUDE_SIZE` |
|---|---|---|
| `:ze` (default) | **true** (control rows appended) | **false** (no log-z) |
| `:aa` | **false** | **true** (log-z on) |

but both read the *same* target, `reg_coef_cloglog_$(n_coef).npy`
(`load_parameters.jl:623-628`), with no `:aa`/size-control variant — in contrast to
`sigma_beta_gamma_filename` (`tools.jl:1112-1118`), which *is* keyed on `aa`. A cloglog
slope estimated without a size control and with appended control-group zeros is a different
estimand from one with log own productivity and no control rows. The paper's own Table 5
quantifies the wedge: adding size moves the distance coefficient `−0.017 → −0.011` in levels
and `−0.284 → −0.130` in logs (35–55%). At most one of the two configurations matches its
target. **Still open — the fix is data-side, outside this repo.**

**Related, SUSPECTED.** `raph.md:44` writes the empirical control as `ν log X_i`, "`X_i` un
proxy **pour la taille** de l'entreprise", while the model regresses on `log z`, the firm's
own Fréchet draw (`model_CP.jl:1258`). In this model sales are a nonlinear function of `z`
and of the destination set, so these are not the same regressor; if the data conditions on
size and the model on productivity, the conditional distance slope — hence `α̂ = β̂/θ` —
carries a non-differenced bias, and `b_logz = −θ` is not comparable to anything empirical.

---

## 6. MAJOR — Hansen J ignores the `S` profiled `N̂_s`

`tools.jl:2132-2136`: `J = r'Wr`, `df = N_mom − p`, with `p = N_TAU` under profiling and
`N_mom` **including** the `S` block-6 rows. `N̂_s` is chosen to set those residuals to ≈0, so
they contribute ≈0 to `J` while `df` counts them as free over-identifying restrictions — `J`
deflated and `df` inflated simultaneously. Under the eq. (29) reading the correct `df` is
`N_mom − p − S`. Relatedly, block-6 rows of `Σ_sim` *are* estimated by re-simulation
(`tools.jl:1359-1367`), which is right, but that cannot coexist with the doc's "noise-free"
claim (finding 1). **Still open.**

---

## 7. MAJOR — leftover assertion `N̂_s ≤ N_rho` from the deleted prefix design

`model_CP.jl:573`: `@assert N_hat[s] <= N_rho "... exceeds the per-replication variety block
width ..."`. `granular_aa_changes.md` §3 and `load_parameters.jl:205-207` state explicitly
that `N_rho` is *not* tied to `N_HI` because no prefix is ever taken. The assertion is
residue of the abandoned realised-economy design and now imposes a hard cap `N̂_s ≤ 100` —
either never binding (finding 1 already caps at ~18) or aborting the run if `N_LO[s] > 100`.
Either way it constrains a parameter the theory bounds only by `[N^obs/N_d, N^obs]`.

> **RESOLVED, by the opposite route.** Rather than deleting the assert, `N_rho` is now
> `max(100, maximum(N_HI))`, which makes `N̂_s ≤ N_rho` a true invariant. The assert is kept
> as a live guard.

---

## 8. MINOR/SUSPECTED — the regressand is the union over all buyers; Prop. 1 is the anchor-market event

Prop. 1 conditions on the area anchor `a` (`Φ^{-l}_{sa}`, `τ_{las}`, `d_{la}`); the cloglog
is exact for a single destination, and the doc says the union `q_ls` is "evaluated by
simulation" (Remark 1). But `linkages_flat[rho,g]=1` is set inside the loop over **every**
downstream region (`model_CP.jl:919`), so `not_supply` (`model_CP.jl:1249`) is the union
event. The code's own docstring is honest ("in the single-destination limit
`β_distance = θα`", `model_CP.jl:1206-1210`).

The code is arguably more defensible than the theory: as an indirect-inference auxiliary
model, matching the same regression on both sides is consistent for `α` even with an inexact
link — *provided the empirical outcome is also "supplies any downstream buyer"*. But the
paper's stronger claims ("the empirical specification is the model, not an approximation to
it"; "`α = β̂/θ` with no attenuation correction") do not hold for the object the code
computes. Fix the text, and confirm the empirical `Sup_i` is union-over-buyers.

---

## 9. MINOR — cross-sector weighting of block 4

Each simulated cell carries total regression weight 1 (`sample_weights[ρ,g]=1/N_rho`,
`model_CP.jl:1254`) in every sector. Under Prop. 1 each cell has exactly `N_s` firms, so
*within* a sector equal weight is right; *across* sectors it is right only if `N_s` is
common. The paper's own "Firm counts" extension flags that observed counts vary by orders of
magnitude. If the empirical regression is unweighted at the firm level, model and data pool
sectors with different implicit weights and the pooled `β̂` is not the same functional on
both sides.

---

## 10. MINOR — notation collision in eq. (28) (documentation only)

Eq. (25) defines `Ḡ_s(n)` by the **threshold** `K_ls ≤ n`; eq. (28) writes
`Ḡ_s(n)=|L⁺|⁻¹Σ_l(1−q̂_ls)^n` with `n` the **variety count**. They agree only at
`n=0`/`n=N_s`. The code implements the §4.2 reading correctly (`model_CP.jl:551-571`). Fix
the `.tex`, not the code.

---

## Checked and CORRECT (so it is not relitigated)

- **Fréchet:** `z = T^{1/θ}(−ln(1−u))^{−1/θ}` ⇒ `P(z≤x)=exp(−T x^{−θ})` — eq. (2) exactly,
  `T` as scale inside the exponent (`model_CP.jl:754-759`).
- **Assumption 2 is a gather, not a collapse:** `T[s,l]=T_par[s,T_GATHER[l]]`
  (`model_CP.jl:443-455`); each ZE keeps its own draw column and its own `τ_{lr}`; `Φ` sums
  over **cells** (`profiling.jl:57-70`, `invert_T_from_gamma`).
- **AA partition = the empirical one:** hard assert `aa_of_ze == CLOSEST_DOWNSTREAM_REGION`
  (`load_parameters.jl:307-312`).
- **γ normalisation end-to-end:** `Σ_l γ = domestic_share` (`model_CP.jl:1378`); AA aggregate
  over every `CELL_MASK` cell, control cells included, on **both** sides
  (`model_CP.jl:1380-1386`, `load_parameters.jl:470-489`), with a hard error if observed γ
  mass sits outside `CELL_MASK`; Sinkhorn target renormalised to `Σ_c=1`. No mismatch found.
- **Cloglog sign/FE:** `y=not_supply`, `X=log d ⇒ +θα`, `X=log z ⇒ −θ`; FE
  `= (s−1)R_down + dr` with `dr` the ZE's nearest downstream region = its area
  (`model_CP.jl:1247`). Matches eq. (13), Table 2, `raph.md:40`.
- **`N_s` confinement enforced:** `compute_N_s_jacobian` *asserts* that bumping `N̂_s` moves
  nothing outside block 6 (`tools.jl:2786-2830`).
- **Price gauge:** CES as an **average** (`model_CP.jl:842`) = Appendix-B `T̃=T N^κ`; no
  stray `N_s`.
- **No silent continuum/granular mixture:** `compute_moments_analytical` hard-asserts
  `!GRANULAR` (`model_analytical.jl:260`); `:aa` γ aggregation *is* mirrored there
  (`:326-333`) and in the quadrature design flags (`:113-115`). The LPM-vs-cloglog asymmetry
  is confined to GMM, which `run.sh:148-151` forces to `(false, ze)`.
- **Bounds:** `N_LO=⌈N^obs/N_d⌉`, `N_HI=N^obs` (`load_parameters.jl:186-188`);
  `N^count=N^obs/Σ_l q̂` (`model_CP.jl:1559-1563`) — eq. (22) and its free check.
- **Wages:** `w_rs` collapsed to 1 where observed; wage-less control cells set to 1 with a
  warning (`load_parameters.jl:29`, `:382-389`) — the `w≡1` normalisation, with no zero-wage
  cell able to win everything.

---

## Consistency checks I recommend running

1. **The floor ceiling (settles finding 1 in one line).** Print `(1−0.5/N_rho)^{N_LO[s]}` vs
   `G_TARGET[s]` per sector. **If below the target, that sector's `N̂_s` is pinned at `N_LO`
   by arithmetic.** *(Run: no sector is pinned; see the VERIFIED note in finding 1. The
   variant worth keeping is the upper ceiling `ln(Ḡ_target)/ln(1−0.5/N_rho)` against
   `N_HI[s]`.)*
2. **Mass at the floor.** `count(q_hat .<= 0.5/N_rho + 1e-15)/n_good` and the quantiles of
   `q_hat`. **Expected ≈0.** Anywhere near 0.915 ⇒ `Ḡ_s(0)` is a function of the floor.
3. **Draw-count invariance.** `granular_report(θ)` at `N_rho ∈ {100, 1000, 10000, 40000}`,
   same θ. **`N̂_s` must be flat in `N_rho`;** monotone drift ⇒ draw-count artefact. Same for
   `b_logz` (target `−θ = −1`).
4. **Unbiased vs plug-in `Ḡ`.** At fixed draws compare `mean_l (1−q̂_l)^N` to the U-statistic
   `mean_l C(m−k_l,N)/C(m,N)` (`m=N_rho`, `k_l=m q̂_l` pre-floor). **Must agree to O(1/m);**
   the gap is finding 1 in moment units.
5. **Price Jensen gap.** `test_analytical_vs_simulated(θ; N_rho_test=N_rho)` under
   `--granular=false --ca_level=aa`, at `N_rho ∈ {100, 1000, 10000}`; read the
   `c_tilde`/`pi_r` block errors. **Expect ≈+6.5% at 100 and +1% at 1000 — Table 3
   reproduced**, confirming `N_rho` is acting as `N_s` in the value block.
6. **Sobol degeneracy.** For the three thickest sectors print `d`, `N_rho`, and
   `maximum(abs, cor(U_DRAWS[:,cols]) − I)`. **Expect ≈`1/√N_rho`; ≥0.3 means correlated
   columns and a biased Ricardian min.**
7. **Step-3 smoke test.** Run past Step 2. **Expected `BoundsError` in `loss_function`** —
   *now fixed; the run should proceed.*
8. **Target-spec match.** Re-estimate the empirical cloglog with *exactly* the `:aa` design
   (not_supply, distance, **log own productivity**, area×sector FE, no control rows, support
   = ZE in active areas) and compare to `reg_coef_cloglog_$(n_coef).npy`. **They must be
   identical numbers.**
9. **Hansen df.** Recompute `p` with `df = N_mom − p − S`, and print `‖r_G‖` — **it should be
   ≈0 by construction**, confirming those `S` moments are not over-identifying restrictions.
10. **`N̂_s` route agreement (V7, already coded).** `N̂_s` vs
    `N^count_s = N^obs_s/Σ_l q̂_ls`. **Cheapest single diagnostic of whether the granular
    block works at all.**
