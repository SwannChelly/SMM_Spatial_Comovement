*Independent referee report on the first draft of `diversification.md`, produced by a
separate agent that re-derived every proposition from scratch (its own scripts, its own
seeds and parameterisations) and checked every cross-reference against the paper. It is
kept here as the provenance of the current version.*

**Disposition of the findings.** All of E1–E8 and O1–O10 were accepted and applied;
so were I1–I12, with two qualifications recorded here rather than silently:

- **O8, variety counts.** The referee could not source `N̂_s ∈ [2,23]` / `[3,14]` to a
  repository artefact. They are in the paper (§"Estimation and fit"), and the note now
  cites it.
- **E2, the two `ρ` conventions.** The referee is right that the draft's formula was
  self-contradicting and that the code uses the exclusion (champion) version. The
  paper's `eq:app-rho`, with the full `Φ_{rs}`, is *not* an error: it is the
  probability for an arbitrary firm inside a region's continuum, a different and
  internally coherent object. §4.3 of the note now states both and says which one it
  uses and why.

Two findings concern the **paper**, not the note, and are worth acting on there: the
main text's inequality `p_{r's} > 1 - ∏_r(1-γ_{r'rs})` is backwards (positive
dependence makes a union *smaller*), and the companion material this note builds on is
currently inside a `\begin{comment}` block.

---

# Referee report — `documentation/diversification.md`

Independent adversarial verification. Every proposition was re-derived by hand and
re-checked numerically with my own scripts (own parameterisation, own seeds), in
`/tmp/.../scratchpad/ref/{A_calculus,B_limits,C_prob,D2_monotone,E_prop10,F_prop11}.py`.
The pre-existing `check.py` was read only after I had formed my own view; it is assessed
in a dedicated section at the end.

---

## VERDICT

The core calculus of the note is correct and I could reproduce every displayed derivative
to 1e-10 or better: Lemma 1, equations (3)–(5), (7)–(12), the A.7 counterexample to the
digit, the A.11 tilting identity (symbolically), the multi-sector form of (19), and the
FKG direction in (15) (by Monte Carlo). That is a real, non-trivial body of correct work,
and the FKG statement in (15) is in fact *more* correct than the corresponding sentence in
the paper's main text, which has the inequality the wrong way round. **But three
substantive claims are false as written and one numerical table is not commensurable with
the object it is compared to.** (i) Proposition 5 / Corollary 5.1 assert that a cell's own
`T_{r's}` (and own wage) is irrelevant for the *composition* of its portfolio; in general
equilibrium this is false, because `T_{r's}` sits inside `Φ_{rs}`, and the own-`T`
derivative is exactly the note's own equation (10) evaluated at `k = r'`, which is not
zero — I measure it at 0.23 in log points in a five-region example. Corollary 5.1's
parenthetical "the `T` of *other* origins enter `Φ_{rs}`" is flatly wrong, and §8 builds
an economic reading on it. (ii) The displayed win probability in §4.3 uses the full
`Φ_{rs}`, which *contains* `T_{r's}`, and is therefore self-contradicting with the very
sentence it is supposed to support; it also disagrees with the note's own Appendix A.10
and with the estimation code, and Monte Carlo picks the A.10/code version (error 3e-4)
over the §4.3 version (error 0.16). (iii) Proposition 11's stated hypothesis
("abstracting from cross-sector spillovers") is insufficient for (19), which needs a
*single* upstream sector; in a three-sector economy (19) is off by 68–95%. On top of that,
the §5 table feeds a PPML constant elasticity `η̂` into a formula the note itself derived
for a *linear-in-log-d* ratio normalised at `d = 1`; on the repo's own conversion the
table's `TF` changes from 0.61/0.15 to 0.31/0.11 and the "implied `Ñ`" from 2.6/1.2 to
1.46/1.12, i.e. the headline contrast largely evaporates. Several cross-references are
wrong (two "Proposition 8" that should be 9, a "Proposition 1" that is the paper's cluster
proposition, "first-order condition (14)", "§5.3"), and §8 contains at least one sign
inversion and a series of empirical conjectures presented as results. None of these is
fatal to the project; all are fixable, and most are one-sentence fixes.

---

## ERRORS

### E1. Proposition 5 and Corollary 5.1: own comparative advantage is *not* irrelevant for composition in general equilibrium
**Location.** Lines 223–224 ("In particular `w_r^{sr'd}` does not depend on `T_{r's}` nor
on `w_{r'}`"), lines 228–232 (Corollary 5.1, especially "the `T` of *other* origins enter
`Φ_{rs}`, hence `m_{rs}`"), line 549 ("At the *cell* level, own `T` is irrelevant for
composition"), lines 779–780 (A.8 remark: "by Proposition 5 the own-`T` channel is absent
from the composition to begin with").

**What is wrong.** `Φ_{rs} = Σ_l T_{ls}(w_lτ_{lrs})^{-θ}` — the note's own notation table,
line 66 — sums over **all** `l`, own origin included. So `m_{rs} = X_{rs}/Φ_{rs}` moves
with `T_{r's}`, and it moves *by a different amount at each destination* (`∂lnΦ_{rs}/∂ln
T_{r's} = γ_{r'rs}`, which varies over `r`). The identity (6) is correct — the
destination-independent factor `T_{r's}w_{r'}^{-θ}` does cancel out of the softmax — but
what survives inside `m` is not zero. The own-`T` composition derivative is precisely the
note's (10) at `k = r'`, and the paper's own text stresses this: "the result is
independent of whether `r' = k`", i.e. the own-`T` effect exists and has the same form as
anyone else's. Corollary 5.1's "the `T` of *other* origins" is therefore wrong, and A.8's
rationalisation for why the indicator cancels ("by Proposition 5 the own-`T` channel is
absent") is the wrong explanation: the indicator cancels because the *direct* effect on
`γ_{r'rs}` is destination-independent, while the `Φ` channel from own `T` is
destination-*dependent* and survives.

**Measurement** (`ref/A_calculus.py`, 5 upstream × 4 downstream, θ=1.768, α=0.35, wages
not all 1): finite-difference `∂ln w_r^{sr'd}/∂ln T_{r's}` = `[0.0972, −0.0446, 0.0359,
0.2320]`, matching (10) at `k=r'` to 1.8e-10. Own wage likewise moves composition:
`∂ln w_r/∂ln w_{r'}` = `[−0.1719, 0.0788, −0.0635, −0.4101]`.

**Correction.** State (6) as an identity *given* `m` and say explicitly that `T_{r's}` and
`w_{r'}` enter only through `Φ_{rs}` — exactly like every other origin's — so the own
channel is `∂H/∂ln T_{r's} = −2 Cov_w(w, γ_{r'·s})`, which is generically **negative** (own
`T` *diversifies* the cell, because `dlnγ/dlnT = 1−γ` raises the share proportionally more
where the origin is currently small, i.e. at the far destinations). Rewrite Corollary 5.1
as: *own comparative advantage moves the composition only through market access, and it
does so in the diversifying direction whenever the origin's portfolio weight and its
sourcing share are positively associated.* §8's first comparative-advantage bullet
(line 549–551) then needs rewriting, since it currently rests on a false irrelevance.

### E2. §4.3 implication 2: the displayed win probability is wrong (and contradicts A.10 and the code)
**Location.** Lines 355–356: "`ρ_{r'rs}(z) = exp(-Φ_{rs}(w_{r'}τ_{r'rs})^θ z^{-θ})` does
not involve `T_{r's}`".

**What is wrong.** With the note's own `Φ_{rs}` (line 66), that expression *does* involve
`T_{r's}`. The sentence contradicts itself. The object Proposition 9 needs — the
probability that the *champion draw of cell `(r',s)` for variety `ρ`* wins at `r` — is
```
   ρ_{r'rs}(z) = exp( − κ_{r'rs} z^{−θ} ),
   κ_{r'rs} = Σ_{l≠r'} T_{ls}(w_lτ_{lrs})^{−θ} (w_{r'}τ_{r'rs})^{θ}
            = (Φ_{rs} − T_{r's}(w_{r'}τ_{r'rs})^{−θ})(w_{r'}τ_{r'rs})^{θ},
```
which is exactly what the note itself writes in A.10 (lines 819, 835) and exactly what the
code computes (`model_analytical.jl:159`, `coef[dr] = T_val − Phi[s,dr]*(w_val*tau)^theta`).
Only the exclusion version integrates back to `γ`: `E_u[e^{−κu}] = T/(T+κ) = γ`, which is
the check the note performs at line 837 — that check fails for the full-`Φ` version.

**Measurement** (`ref/C_prob.py`, 6×5 EK economy, 2M competitor draws, own draw at the 0.6
quantile): MC win probabilities `[0.1407, 0.0574, 0.4059, 0.0051, 0.3497]`; excluding own
`T` gives max error **2.7e-4** (Monte-Carlo noise); the full-`Φ` formula gives
`[0.0845, 0.0344, 0.2437, 0.0031, 0.2099]`, max error **0.162**.

**Correction.** Replace the displayed formula by the `κ_{r'rs}` version and note that it
is the champion-draw convention, i.e. the one the estimator simulates. (The full-`Φ` form
does appear in the paper — main text line 100, Appendix eq. `app-rho` — because the paper
there conditions on an *arbitrary* firm of productivity `z` inside a within-region
continuum rather than on the region's champion. If the note wants to keep the paper's
notation it must say which of the two conventions it is in; it cannot use one formula and
the other's property.)

### E3. Proposition 11: the stated hypothesis is insufficient for equation (19)
**Location.** Lines 443–447 ("Abstracting from cross-sector spillovers, so that
`M_{r'r} = ζ_{r's} w_r^{sr'd}` for the sector under policy") against A.12 line 889 ("If the
economy has a single upstream sector").

**What is wrong.** `M_{r'r} = Σ_{s'} ζ_{r's'}w_r^{s'r'd}` already *is* the "no cross-sector
spillover" object (the note's own A.12, line 879). Collapsing it to `ζ_{r's}w_r^{sr'd}`
additionally requires that region `r'` has no upstream sector other than `s`. The appendix
gets this right; the proposition statement does not, and (19) is what the note then
recommends as statistic 9 and as the policy sign test.

**Measurement** (`ref/F_prop11.py`, 3 sectors, 4 upstream, 3 downstream): the multi-sector
formula on line 453 reproduces the true derivative to 5e-11 for every `(k,s)`; equation
(19) applied to the same multi-sector economy is off by **86.7%, 74.6%, 68.0%, 95.4%** for
`k = 0,1,2,3`. In a genuinely single-sector economy (19) is exact (1e-11).

**Correction.** Either restate (19) under "a single upstream sector (or `ζ_{r's'} = 0` for
`s' ≠ s`)", or present the multi-sector expression as the proposition and (19) as the
one-sector special case. Statistic 9 in §7.1 should point at the multi-sector formula.

### E4. §5's table plugs a PPML constant elasticity into a formula derived for a linear ratio
**Location.** Lines 415–424 (the table and the sentence introducing it), and (18) at
line 410.

**What is wrong.** (16)–(17) are derived for the paper's Appendix-B.5 object
`δ/γ = (dE[a]/d ln d) / E[a]|_{d=1}` — a *linear-in-log-d* regression coefficient ratio,
normalised at `d = 1`; the note is careful about this and gives it a name (`κ`). But the
table then uses `η̂`, which the paper obtains from `E[a_{ir}|s,r',r] = exp(α_{rs} + η log
Dist)` — a PPML *constant elasticity*. These are not the same number. The repo's own
`CLAUDE.md` records the conversion `δ/γ ≈ η/(1 + |η| · mean log d)` with
`EMPIRICAL_MEAN_LOG_D = 5.8`, and states in terms that "PPML fits a constant elasticity …
the spatial-comovement regression is LINEAR IN THE LEVEL". The note nowhere flags this,
and its own `κ` machinery is precisely the wedge it silently ignores.

**Measurement** (`ref/E_prop10.py`). Table arithmetic *as given* is internally correct:
`0.166/0.27 = 0.6148 → 0.61`, `1 − 0.6148 = 0.3852 → 0.39`, `1/0.3852 = 2.596 → 2.6`;
`0.063/0.43 = 0.1465 → 0.15`, `0.8535 → 0.85`, `1.172 → 1.2`. But converting `η̂` at
mean log d = 5.8:

| | `θα` | `δ/γ` implied by `η̂` | `TF` | implied `H̃` | implied `Ñ` |
|---|---|---|---|---|---|
| Motor vehicles | 0.27 | −0.0846 | 0.313 | 0.687 | 1.46 |
| Aerospace | 0.43 | −0.0461 | 0.107 | 0.893 | 1.12 |

The `Ñ` contrast collapses from 2.6-vs-1.2 to 1.46-vs-1.12.

**Correction.** Either convert `η̂` before entering (18) and say so, or state that the
model counterpart of `δ/γ` in (17) must be re-estimated in the *linear* specification
(which is what the paper's `δ` and `γ` are), not read off the PPML `η̂`. Flag the
non-commensurability explicitly; the note currently reads as if `η̂` and `δ/γ` were the
same object, which is the paper's assertion but not a true one.

### E5. Corollary 6.1's main-text gloss: "spatially uniform demand" ≠ "`m` constant"
**Location.** Lines 280–282: "`Cov_w(w, x) > 0` … is guaranteed when demand is spatially
uniform (`m_{rs}` constant, Corollary 6.1 in the appendix)".

**What is wrong.** `m_{rs} = X_{rs}/Φ_{rs}`. Uniform `X` does *not* give constant `m`,
because `Φ_{rs}` varies across buyers with geography — and the sign can flip.

**Measurement** (`ref/B_limits.py`, 3 upstream × 2 downstream, `X = (1,1)`, α = 0.6):
`Φ = (0.3225, 0.0490)`, so `m = (3.10, 20.39)` — not constant — and
`Cov_w(w, x) = −0.105 < 0`. Trade costs *diversify* under uniform demand in that geometry.

**Correction.** The correct sufficient condition is `X_{rs} ∝ Φ_{rs}` (demand proportional
to local price access), not uniform demand. Say that, or drop the gloss.

### E6. §8 "smaller pairwise angles" is the wrong sign for the argument being made
**Location.** Lines 563–565: "Aerospace is the extreme case on the *angle*: portfolios are
short *and* nearly collinear … Motor vehicles has longer portfolios *and* smaller pairwise
angles between suppliers attached to different hubs."

**What is wrong.** By Proposition 2, `Corr = cos ∠`; *smaller* angle = *more* correlated.
The intended contrast (aerospace comoves more, motor vehicles less) requires motor vehicles
to have **larger** angles. As written the two sentences say the opposite of each other.

**Correction.** "larger pairwise angles".

### E7. Wrong cross-references
- **Line 183**: "`n_i` is the extensive margin (**Proposition 8** gives it in closed
  form)". Proposition 8 is the demand-geography derivative; the extensive margin is
  **Proposition 9**. (Also: Prop. 9 gives `E[n_ρ]`, not `n_i`, so "in closed form" should
  be "in expectation in closed form".)
- **Line 232**: "(In the granular model it re-enters at the firm level through selection:
  **Proposition 8**.)" → **Proposition 9**.
- **Line 462–463**: "This is precisely **Proposition 1's** 'moderately central but
  currently under-supplied' statement". That phrase is in the *paper's* cluster-policy
  proposition (`\label{prop:cluster_policy}`), not in the note's Proposition 1 (variance
  representation). Cite the paper's proposition.
- **Line 464**: "it slots directly into the paper's **first-order condition (14)**". The
  paper's FOC is `\label{eq:foc}`. Counting the source, it is equation 18 (author's own
  informal count, where `eq:w_elast_T` is 13) or 20 (strict LaTeX count, where two-row
  `align`s number twice). Under *neither* counting is it 14; under the author's own count
  (14) is `eq:V_elast_tau`, the comovement derivative.
- **Line 117**: "the paper's own assumption in **§5.3**". There is no §5.3; the equation
  `d ln x_{it} = Σ_r a^D_{ri} δ^D_{r,t}` appears only inside the commented-out
  "Reorganizing production network" block (main text line 267), and Appendix B.5 has it
  with an extra idiosyncratic term `δ_i^D` (line 595), which Assumption 1 silently drops.

### E8. The FKG product formula is misattributed to the paper's Appendix B, and the paper's own inequality points the wrong way
**Location.** Lines 338–342 and 360–363.

**What is wrong.** The current paper does **not** use `ρ̃ = 1 − ∏(1−ρ)`. Main text line 67
says the opposite: "there is no tractable closed form for `p_{r's}`, and we evaluate it by
simulation", with the inclusion–exclusion sum relegated to a footnote. The product formula
survives only in the *analytical/GMM* code path (`compute_regression_quadrature`,
`model_analytical.jl:170–174`), i.e. exactly the path `CLAUDE.md` already flags as
FKG-biased. Two consequences:
1. "the FKG product formula **used in the paper's Appendix B**" is a misattribution.
2. Implication 3 ("both go through `q̂`, so the direction of the bias in any
   diversification statistic built on `q̂` is known") is wrong for the artefact the note's
   own Appendix B points at: `q_hat` in `granular_diagnostics.npz` is the **exact simulated
   union** (`model_CP.jl:2244–2251` counts `linkages_flat`, set at the Ricardian argmin,
   `model_CP.jl:935`). There is no FKG bias there at all.

**Bonus finding for the authors** (not the note's fault, but the note should say it,
because (15) contradicts the paper): the paper's main text line 67 asserts
`p_{r's} > 1 − ∏_r(1 − γ_{r'rs})` while giving the correct reason (positive dependence).
Positive dependence makes the union *smaller*, not larger, so the inequality is backwards.
My MC (`ref/C_prob.py`): `γ_{r'·} = (0.207, 0.152, 0.362, 0.088, 0.327)`, product formula
`1 − ∏(1−γ) = 0.7365`, simulated `q̂ = 0.4163`. The note's (15) is right; the paper's
sentence is wrong; the note should flag the disagreement rather than attribute the product
formula to the paper.

---

## OVERREACH

### O1. Proposition 6's limits are proved only in the partial (`m` fixed) regime
**Location.** Lines 269–271 ("Moreover `lim_{α→∞} H = 1` whenever the nearest destination
is unique, and `lim_{α→0} H = Σ m̄²`"); proof at lines 742–745.

The "Moreover" sits after the sentence that introduces the *total* (Φ-adjusting) regime, so
it reads as a claim about both. The proof holds `m` fixed:
`w_r/w_{r*} = (m_r/m_{r*}) exp(−θα(ln d_{r'r} − ln d_{r'r*}))`. In the total regime
`Φ_{rs} ~ T_{l*(r)s}d_{l*(r)r}^{−θα}` blows up, so `w_r ∝ (X_{rs}/T_{l*(r)s})
(d_{r'r}/d_{l*(r)r})^{−θα}`: the surviving destination is the argmin of **relative**
distance `ln d_{r'r} − ln d_{l*(r)r}`, not the nearest destination.

**Measurement** (`ref/B_limits.py`): a geometry where `r'` is at distance 100 from dest 0
and 300 from dest 1, but dest 0 is glued to a rival at distance 10. Nearest destination is
0; as α rises the whole portfolio goes to **destination 1** (`w = (0, 1)`, `H → 1`).

*The limit value `H → 1` survives*; the stated uniqueness condition does not. Correction:
say the limits are for the partial regime, or restate the α→∞ condition as uniqueness of
`argmin_r (ln d_{r'r} − ln d_{l*(r)r})`.

### O2. The illustrative table's `κ = 1, Λ̄ = 0` is not merely illustrative — `Λ̄ = 0` is on the boundary and forces the answer
**Location.** Lines 418–424.

`Λ = −ln ρ ≥ 0` with equality iff `ρ = 1`, so `Λ̄ = E_ν[Λ] > 0` strictly. From (17),
`H̃ = 1 + Λ̄ − TF/κ`, so setting `Λ̄ = 0` yields the *lower* bound on `H̃` and the *upper*
bound on `Ñ` — i.e. it maximises the reported diversification contrast. Worse, since
`H̃ = E_ν[a] ≤ 1`, admissibility with `κ = 1` requires `Λ̄ ≤ TF`, i.e. a tilted-average win
probability `ρ ≥ e^{−TF}` = **0.54 (auto) and 0.86 (aero)** — implausibly high for a
supplier's per-destination win probability. So either `κ` is well below 1 (which the note
says is likely) or the decomposition does not admit these `TF`s at all; the "implied `H̃`"
column is therefore not identified by the numbers shown. Label the column as a bound, or
drop it until `Λ̄` and `κ` are measured.

### O3. `TF ∈ [0, ~1]` is asserted without justification
**Location.** Line 410. `TF = κ(Λ̄ + 1 − H̃)` and `Λ̄` is unbounded above (`ρ → 0` gives
`Λ → ∞`), so nothing in the note bounds `TF` by 1. Either prove a bound or drop the range.

### O4. §8 "The two forces therefore compound rather than offset in aerospace" (bolded)
**Location.** Lines 541–544. This asserts that a larger `θα` concentrates portfolios in
aerospace, i.e. `Cov_w(w, ξ) > 0` there. The note's own Proposition 6 reading (ii)
(lines 279–285) says the sign is "an empirical matter, not a theorem", and A.7 gives an
explicit counterexample. A bolded signed claim about a specific industry is not supported
by anything in the note. Present it as a hypothesis to be measured.

### O5. §8's arithmetic reading of the reversal is circular
**Location.** Lines 543–544: "Equation (18) states the reversal as arithmetic: a larger
`θα` multiplied by a much smaller `(1 − H̃)` yields a smaller `|δ/γ|`." The "much smaller
`(1 − H̃)`" for aerospace was itself *backed out of* `|δ/γ|` in the table two paragraphs
earlier. Nothing has been explained; an identity has been read in both directions. Say so,
or drop the sentence until `H̃` is computed independently (which is exactly the consistency
check the note proposes at lines 426–431 — that check is the right instrument, and until it
is run §8 has no content here).

### O6. Abstract overstates (17) and the bridge
**Location.** Lines 27–30: "the untargeted moment `δ/γ` equals `-θα` times
`(Λ̄ + 1 - H̃)`" — the κ factor is silently dropped, though §5 keeps it and warns it is not
1. And "so the cross-industry contrast in `δ/γ` *is* a cross-industry contrast in customer
diversification": `δ/γ` is a product of *four* factors (`θ`, `α`, `κ`, `Λ̄ + 1 − H̃`), only
one of which is diversification. Line 378 ("It is exactly a Herfindahl") is the same
overstatement: `1 − a` is not a Herfindahl; `E_ν[a]` is Herfindahl-*like*, and only equals
`H_i` under a condition (constant `ν` within supplier) that the note itself says never
holds (line 871).

### O7. Prop. 9's monotonicity of `q̂` is asserted but not proved in A.10
**Location.** Line 332 ("so is `q̂_{r's}`"). A.10 proves `E[n]` monotone (part i) and the
FKG bound (part ii); the `q̂` claim needs `ρ̃(u)` decreasing in `u` plus the likelihood-ratio
order, which only appears in the *Remark*. It is a two-line argument — write it, or point
to the Remark. (I verified it numerically: `q̂` = 0.147, 0.259, 0.416, 0.595, 0.754, 0.864
as `T_{r's}` scales ×0.25 … ×8.)

### O8. §8 is largely empirical conjecture presented as result — flag list
Each of the following is stated in the indicative and is *not* computed anywhere in the
note or the repo artefacts it cites:
- L531–535: "`N^0_s` … is a small number in aerospace … and a substantially larger one in
  motor vehicles." Not computed. It *is* computable from `X_rs.npy` alone (see I4) and
  should simply be computed.
- L553–554: "These suppliers are small." No support.
- L555–559: "The large aerospace suppliers should therefore be *more* multi-homing than the
  average, and the sales-weighted and unweighted `N` should diverge more in aerospace than
  in motor vehicles." The first half follows from Prop. 9; the *cross-industry* comparison
  additionally needs the within-sector dispersion of `T̂` to be larger in aerospace, which
  is not established. The note does label it "a sharp, falsifiable prediction", which is
  the right register — but only for the first half.
- L563–567: "portfolios are short *and* nearly collinear", "Motor vehicles has longer
  portfolios". Not computed.
- L568–570: "the paper's current narrative — 'hub regions act as aggregators of
  network-driven fluctuations' — is a statement about statistic 6." It is not: the paper's
  own gloss is "their activity loads on shocks affecting a broad set of downstream
  locations", i.e. portfolio *breadth* (statistics 1/4/8 — the *length*), not pairwise
  similarity (statistic 6 — the *angle*). This misreads the very distinction §3.2 sets up.
- L201–203: "`N̂_s ∈ [2, 23]` in motor vehicles and `[3, 14]` in aerospace." I could not
  verify these against any artefact in the repo; they should be sourced or dropped.

### O9. Proposition 11's `𝒱` is not the paper's `𝒱`
**Location.** Lines 443–444 and 463–464. The paper's variance constraint is `ΩΣΩᵀ ≤ V̄`
(`eq:planner_problem`), a full quadratic form that **includes cross-region covariances**;
the note's `𝒱 = Σ_{r'} ω_{r'} Var(r')` is the weighted sum of diagonal terms only. The two
have different derivatives (`ref/F_prop11.py`: −0.001651 vs −0.001584 for `k=0`;
−0.006502 vs −0.006091 for `k=1`). The claim that (19) "slots directly into the paper's
first-order condition" is therefore not right, and it sits badly with the note's own §8
conclusion that "the aggregate comovement contrast is the *similarity* contrast" — the
object Prop. 11 differentiates is the one that throws similarity away.

### O10. Equation (2) is stated as an identity but is not one for the object (1) defines
**Location.** Line 90, and lines 93–95 which concede the point ("with finitely many
varieties they do not [coincide]"). (1) defines `w_r^{sr'd}` from `γ × X` (continuum);
(2) equates it to `Σ_i σ_i a_{ir}` (granular realisation). Both cannot hold. Proposition 4
is a correct Jensen inequality *for the granular definition*, so the fix is to define
`w_{r's}` in Prop. 4 as the realised cell portfolio and mark the continuum/granular
distinction at (2) with `≈` plus a pointer.

Relatedly, Appendix B's implementation warning (lines 920–923) says `w_srd_r.npy` and
`γ̂ × X` "differ by exactly the granular wedge of Proposition 4". They do not: Prop. 4's
wedge is a gap between two *Herfindahls* (`Σσ_iH_i − H_{r's}`), not a gap between two
portfolio vectors. Rephrase.

---

## IMPRECISIONS / presentation issues

**I1. Every principal cross-reference to the paper points at commented-out material.**
`\begin{comment}` at `structural_2026_endogeneous.tex:245` closes at `:421` and swallows
the whole "Reorganizing production network" subsubsection *and* the optimal-cluster-policy
subsection, including `eq:w_elast_T` (the note's (13)), `eq:V_elast_tau`, `eq:foc`,
`prop:cluster_policy`, and the "hub regions act as aggregators" paragraph quoted at
line 569. None of those equations is numbered in the compiled paper. The note should say
that its companion material is currently commented out, or the two should be
re-synchronised.

**I2. Symbol collisions not flagged.** The note flags the `w` clash (lines 73–75) but not:
- `ρ` = a variety (line 62) **and** `ρ_{r'rs}(z)` = win probability (line 70), both in the
  same table, with `n_ρ` in Prop. 9 using the first sense;
- `γ` = sourcing share (line 67) **and** `γ` = the regression coefficient in `δ/γ`
  (line 372) — the single most confusing clash in the note, and the one that matters most
  for §5;
- `σ_i` = firm share of cell sales (line 90) **and** `σ²` = shock variance (line 127);
- `κ` = the `d=1` normalisation constant (line 384) **and** `κ_{r'rs}` = the competition
  index (line 835);
- `x_r` = proximity (line 254) **and** `x_i` = supplier sales (line 625);
- `N` = variety count `N_s`, effective customers `N_i`, `N^0_s`, and `N(u)` in A.10.
Also, the paper's `p_{r's}` and the note's `q̂_{r's}` are the same object; the note never
says so.

**I3. Prop. 7's "the effect is the same for every origin `r'`" is literally false**
(lines 306–308). In (10) only `−γ_{krs}` is origin-free; `Σ_{r''}w_{r''}^{sr'd}γ_{kr''s}`
depends on `r'`. The note immediately says the sign is origin-specific, so it contradicts
itself in one sentence. The paper's "uniform within-portfolio reallocation" means the
*ranking* of destinations is common, which is what should be said.

**I4. `N^0_s = 1/Σ_r m̄²` needs "`m̄` evaluated at `α = 0`".** At `α = 0`, `τ ≡ 1`, so
`Φ_{rs}` is buyer-invariant and `m̄ = X̄` — which is exactly why §9 can call `N^0_s`
observable. I verified this (`ref/A_calculus.py`: `H(α=10⁻⁹) = 0.40563` equals
`Σ m̄²|_{α=0} = 0.40563`, whereas `Σ m̄²` evaluated at `α = 0.35` is 0.4413 — a 9%
difference). As written, statistic 5 (line 481) and §8 (line 532) leave `m̄` at the
estimated `α`, where `Φ` is model output and the "directly observed" claim (line 530) fails.

**I5. `H_i` as "exactly the variance" needs `σ² = 1` and a common `σ²` across industries.**
Line 129 and the ratio claim at lines 139–140 ("more volatile by a factor `H_aero/H_auto`")
are only right if the two industries share the same regional-shock variance. Also, the
paper's own §B.5 transmission equation carries an idiosyncratic `δ_i^D` that Assumption 1
drops without comment; with it, `Var = σ²H_i + Var(δ_i^D)` and the ratio statement fails.

**I6. Proposition 10's last claim needs `Σ_r a_{ir} = 1`, which the paper's `a_r^D` does
not satisfy.** A.11 line 866 divides by `Σ_r a_{ir} = 1`. The paper's displayed
`a_r^D(z,d) = γ_{r'rs}x_{rs}/(γ_{r'rs}x_{rs} + Σ_{r''≠r}γ_{r'r''s}x_{r''s}ρ_{r'rs}(z))`
(appendix line ~626) has an `r`-dependent denominator and does **not** sum to one over `r`
(it also carries an evident typo, `ρ_{r'rs}` where `ρ_{r'r''s}` is meant — with the typo
taken literally the paper's own `d ln a/d ln d = (1−a) d ln γ/d ln d` step is wrong, since
`ρ_{r'rs}` also depends on `d_{r'r}`). The note silently reinterprets `a` as the realised
portfolio share. That reinterpretation is the right one, but it should be stated, since the
whole bridge to `H_i` hangs on it.

**I7. `H̃ ≥ H` "in the typical configuration" (line 872–873) is exactly `Cov_p(ν, a) ≥ 0`
with `p_r = a_r`.** Since `H̃ = H + Cov_p(ν,a)/E_p[ν]`, this is a clean one-line condition
and holds iff `ν` and `a` are positively associated. I found counterexamples with random
`(ν, a)` (`ref/F_prop11.py`: `H = 0.4031`, `H̃ = 0.3563`). Give the condition rather than
"typical".

**I8. `E[n|n≥1]` as an "upper bound on the effective number of customers" (line 351)** is a
bound on the conditional *mean* of `N_i`, not on `N_i`; Prop. 3's `N_i ≤ n_i` is pathwise.
One word.

**I9. A.10's "tends to 1 as `u → ∞`"** (line 842) needs uniqueness of `argmin_r κ_{r'rs}`;
with ties the limit is the number of tied destinations.

**I10. Truncated quote.** Lines 40–43 end the paper's sentence at "…toward a single
downstream region." The paper continues "…when moving away from the other." No ellipsis.

**I11. `sourcing_geometry`'s `tot` (Appendix B, line 913) is `Φ` summed over that sector's
*domestic active cells only*.** If foreign competition is part of `Φ` in the model, `m =
X/tot` is not `X/Φ`, and the discrepancy is buyer-specific (so it does not cancel in the
softmax). Worth a line in Appendix B.

**I12. The `Ḡ_s(0) = mean_l (1−q̂_{ls})^{N̂_s}` closed form quoted at line 349** has been
superseded in the code: `model_CP.jl:2237–2245` says the count moment "is now evaluated
with the unbiased `gbar_cell` … which needs `k` and `m` rather than `q̂`", and `q_hat` is
retained as a raw diagnostic. The statistic the note wants is still on disk, but the
sentence describing where it comes from is out of date.

---

## VERIFIED — what I checked and confirmed

All numerical checks are mine, in `/tmp/.../scratchpad/ref/`, with my own seeds and an
asymmetric 5-upstream × 4-downstream geometry (so a transposed index would show up).

| Item | Method | Result |
|---|---|---|
| **Prop. 1** (A.1) | Hand derivation | Correct given Assumption 1. `Var = a'Σa`; `= σ²H_i = σ²/N_i` under `Σ = σ²I`. See I5 for the two caveats. |
| **Prop. 2** (A.2) | Numeric, 5-sector `M` (`A_calculus.py`) | `max\|Cov − σ²ζζ'⟨M̃,M̃⟩\| = 1.4e-17`; `max\|Corr − cos\| = 2.2e-16`; `ζ ∈ [0.28, 0.60] ⊂ [0,1]`. Statement and proof correct. |
| **Prop. 3** (A.3) | Algebra + numeric with a zero-share firm | `max\|H − (1+CV²)/n\| = 2.2e-16`. `CV` must be the *population* CV (ddof = 0) — the note's A.3 uses that, correctly. |
| **Prop. 4** (A.4) | 9 firms × 6 destinations | LHS `0.17895668995107 72` = RHS `…722`. Identity and non-negativity correct. |
| **Prop. 5, eq. (6)** | Numeric identity | `max\|w − m τ^{-θ}/Σ\| = 1.1e-16`. The *identity* is correct; the *interpretation* is not — see E1. |
| **Lemma 1** (A.6) | Independent 7-point softmax, FD | `Ḣ_FD = 0.07827471274`, `2Cov_w(w,v̇) = 0.07827471277`. Both parts correct, including `E_w[w] = H`. |
| **Prop. 6 eq. (8)**, partial | FD with `m` pinned | `−0.02846621` both ways, diff 1.3e-10. Correct. |
| **Prop. 6 eq. (9)**, total | Full-GE FD in α | `0.17843088` both ways, diff 6.0e-11. `∂lnΦ/∂α = Σ_lγ_{lr}x_{lr}` verified. Correct. |
| **Prop. 6 limits** | α = 1e-9 and α = 5…150 | `α→0` limit correct *provided* `m̄` is taken at `α=0` (I4). `α→∞` limit `H→1` correct; the stated uniqueness condition is wrong in the total regime (O1). |
| **Cor. 6.1** | Chebyshev/association argument | Correct as stated in the appendix (`w` is an increasing function of `x` when `m` is constant and `α>0`, so `Cov_w(w,x) ≥ 0`, strict unless `x` constant). The main-text gloss is wrong (E5). |
| **Prop. 7 eq. (10)** | FD for `k = r'`, `k ≠ r'` | Matches to 1.4–2.6e-10 for all `k`. **It does coincide with the paper's `eq:w_elast_T`** — I compared term by term: `Σ_{r''}w_{r''}^{sr'd}γ_{kr''s} − γ_{krs}`, identical, and the paper's intermediate step `d lnγ_{r'rs}/d lnT_{ks} = 1{r'=k} − γ_{krs}` is itself correct. The equation *number* is doubtful (E7). |
| **Prop. 7 eq. (11)** | FD for all five `k` | Max diff 5.3e-11. Correct. |
| **Prop. 8 eq. (12)** | FD for all four `k` | Max diff 1.0e-10; `Σ_k = −1.0e-17`. Correct, and the homogeneity remark is right. |
| **Prop. 9 eq. (13)** | 4M-draw EK simulation | `E[n]_MC = 1.13745` vs `Σ_rγ = 1.13570`. Marginal win probs match `γ` to 4.8e-4. The "linearity of expectation needs no independence" remark is correct, and the matched-varieties setup is the model's. |
| **Prop. 9 eq. (14)** | Same | MC `2.7327` vs `Σγ/q̂ = 2.7286`. Correct. |
| **Prop. 9 eq. (15) / A.10(ii)** | MC at four own-draw quantiles | Product formula ≥ `ρ̃` at **every** quantile: gaps `+0.0395, +0.1899, +0.1168, +0.0115`. The association argument is right: `W_r^c` is an increasing event in the independent competitor vector, EPW association gives `P(∩W_r^c) ≥ ∏P(W_r^c)`, hence `ρ̃ ≤ 1 − ∏(1−ρ)`. **The direction in the note is correct**, `q̂ ≤ q̂^{FKG}`, `E[n\|n≥1] ≥ E[n]/q̂^{FKG}`. (Attribution is not — E8.) |
| **Prop. 9, monotonicity in `T`** | Re-simulated at `T×{0.25,…,8}` | `E[n]`, `q̂`, `E[n\|n≥1]` all strictly increasing. The `E[n]` proof (`γ = A/(A+B_r)`) is correct; `q̂` is asserted not proved (O7). |
| **A.10 Remark** | mpmath (250 dps) inclusion–exclusion, validated against MC to 2.5e-4; 60 random geometries | `u ~ Exp(T)`, `E_u[e^{−κu}] = T/(T+κ) = γ` ✓ (this only works with `κ` **excluding** own `T`, confirming E2). LR ordering of the exponential family ✓; the tilted-measure step `E[n\|n≥1] = E_{P̃_T}[N/ρ̃] = E[N]/E[ρ̃]` ✓. `g(u) = N/ρ̃` equals `R_d` at 0 and 1 at ∞ ✓. **No violation of the conjectured monotonicity in 60 random geometries** (`Ru ∈ [3,7)`, `Rd ∈ [2,6)`, `θ ∈ [0.8,3]`, `α ∈ [0.05,1.5]`). The note's refusal to claim it as proved is exactly the right call. |
| **Prop. 10 / A.11** | sympy, 4 states | `E[νa(Λ+1−a)] − E[νa](Λ̄+1−H̃) ≡ 0` identically. The `τ = d^α ⇒ d lnτ/d ln d = α` step is exact. **(16) faithfully restates the paper's Appendix-B.5 display**, including the `d = 1` normalisation of the denominator and the `ν = ρ/ρ̃` selection term — I compared character by character against `ap_structural_2026_endogeneous.tex:637`. The `H̃ = H_i` corollary is algebraically right *given* `Σ_r a_{ir} = 1` (I6). |
| **§5 table arithmetic** | Exact recomputation | `TF`, `H̃`, `Ñ` are internally consistent to the digit as printed (0.6148→0.61, 0.3852→0.39, 2.596→2.6; 0.1465→0.15, 0.8535→0.85, 1.172→1.2). The *inputs* `θα = 0.27/0.43` and `η̂ = −0.166/−0.063` match `CLAUDE.md` and the paper's untargeted-moment paragraph respectively. The commensurability is the problem, not the arithmetic (E4, O2). |
| **Prop. 11, multi-sector form** (line 453) | FD, 3 sectors | Exact to 5e-11 for every `(k,s)`. Derivation correct. |
| **Prop. 11 eq. (19)** | FD, single-sector | Exact to 1e-11. `Cov_w(w,γ) = Σw²γ − Hγ̄_k` verified; consistent with (11) via `𝒱 = σ²Σωζ²H`. Hypothesis mis-stated (E3). |
| **A.7 counterexample** | mpmath, 40 dps | `H(0) = 0.9802` ✓; `Cov_w(w,x) = −0.02233968057` → `−0.02234` ✓; `∂H/∂α = −0.04467936114` → `−0.04468` ✓ (matches a symmetric FD to 15 digits); `α* = ln99/ln10 = 1.99563519` → `1.9956` ✓; `w(α*) = (0.5, 0.5)`, `H = 0.5` exactly ✓; `H(α)` traced: 0.9802, 0.9400, 0.8333, 0.6330, **0.5000**, 0.8361, 0.9998, 1.0000 — strictly falling then strictly rising, as claimed, and I verified analytically that `w_1 = t/(1+t)` with `t = 99·10^{−α}` strictly decreasing, so the minimum at `w_1 = ½` is unique. **Every number in A.7 is exact.** |
| **§7.1 `𝒦 > 1` claim** (line 494) | Same example | `N⁰ = 1/0.9802 = 1.0202`, `N(α*) = 2`, `𝒦 = 1.9604 > 1` ✓. |
| **§7.2(a)** | Combinatorics | `2³` grid = 8 cells, 6 orderings for 3 factors ✓. |
| **Definition 1** | | Hill number of order 2 is `(Σp²)^{1/(1−2)} = 1/H` ✓; order 1 is `exp(−Σ a ln a)` ✓; `1 ≤ N ≤ R_d` ✓. |
| **Appendix B code map** | Direct inspection | `w_srd_r.npy` indexed `[s, r', r]` ✓ (`main.jl:695,710`); written by `write_post_hoc` at both `step1` and `step3` ✓; `sourcing_geometry(...)["by_sector"][s]["rho"] = psi/tot` **is** `γ_{r'rs}` and `tot` **is** `Φ` up to `w ≡ 1` ✓ (see I11); `q_hat` is in `granular_diagnostics.npz` ✓ (`tools.jl:1162`); `sourcing_geometry(alpha=0, equalise_T=True)` exists ✓; `N_TAU = 1` is enforced by the function ✓; `build_a_ir_panel` and the `share × downstream_purchase` warning ✓ (`CLAUDE.md`). |
| **§1's quotation of the paper** | String match | Verbatim, except silently truncated (I10). |
| **Equation numbering (1)–(19)** | Enumeration | No gaps, no duplicates, and every one of (1)–(19) is referenced at least once elsewhere in the note. Internally clean. |
| **Proposition numbering 1–11, Lemma 1, A.1–A.12** | Enumeration | The appendix map is right: A.1–A.5 = P1–P5, A.6 = Lemma 1, A.7–A.12 = P6–P11. Corollaries 5.1 and 6.1 are attached to the right propositions. Only the *content* cross-references are wrong (E7). |

**Corollary 5.1** — the statement "comparative advantage determines the *level* of sales,
not the *composition*" is correct **only** for the direct channel; see E1 for the GE channel.
**Corollary 6.1** — correct as stated in the appendix; see E5 for the main-text gloss.

---

## Assessment of the pre-existing `check.py`

It runs clean (21/21 PASS) but it is not a check of the note's contested claims, and two of
its assertions are mislabelled or vacuous.

1. **Mislabelled, and it hides E1.** Line:
   `ok("P5 own-T irrelevant for own composition (row 0 unchanged when only Phi held)", not np.allclose(w2[0],w[0]))`.
   The condition passes when row 0 **changes**. The printed line reads "PASS P5 own-T
   irrelevant …" while the code has just demonstrated the opposite — precisely the error in
   E1. This is the single most misleading line in the script.
2. **Vacuous.** `ok("P5 composition depends on origin only through d", np.allclose(sm_fix,w))`.
   `sm_fix` is recomputed with the identical expression as `sm`, which was already asserted
   equal to `w` two lines above. It re-asserts the previous test and tests nothing about
   origin-dependence (that would require varying the origin's `T`/`w` at fixed `d`).
3. **Correctly labelled but narrow.** The P11 test is honestly labelled "single-sector" —
   but that means the script never exercises the note's *stated* hypothesis, so it cannot
   catch E3.
4. **Not covered at all**: the §5 table and its `η̂`-vs-`δ/γ` commensurability (E4); the
   §4.3 `ρ` formula (E2); the α→∞ uniqueness condition in the total regime (O1); Cor. 6.1's
   "spatially uniform demand" gloss (E5); the `Λ̄ ≥ 0` admissibility of the implied `H̃`
   (O2); every cross-reference (E7); the attribution of the FKG formula (E8); §8 entirely.
5. **Minor hygiene**: `Z=rng.weibull(...)` is dead code immediately overwritten (it does
   consume RNG state); `rng.random(0).sum()` inside `V` is a dead `0.0`; the FKG test
   carries a `+1e-3` slack and is a `bad==0` counter that would also pass on an empty loop;
   the `P9` tolerances (2e-2, 3e-2) are loose. None of these produces a false PASS here,
   but the first two make the script harder to trust on inspection.

**Net.** `check.py` correctly verifies the calculus (P2, P3, P4, P6 partial and total,
P7, P8, P9's mean and FKG direction, P10's tilt, P11 single-sector, A.7) — and my
independent scripts agree with it on all of those. It does **not** verify anything that is
actually wrong in the note, and one of its PASS lines actively asserts the opposite of what
it tests.
