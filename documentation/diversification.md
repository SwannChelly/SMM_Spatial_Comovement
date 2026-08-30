# Customer diversification in a spatial production network

*Theory, measurement, and the comparison of motor vehicles and aerospace*

**Companion note to §5 of "Spatial Comovements". All notation follows the paper
(main text and Appendix B); Appendix A of this note collects the proofs.**

---

## Abstract

We ask how diversified the customer base of an upstream supplier — a firm, or a
sector × region cell — is in the estimated model, and what that diversification is
worth. Under the paper's maintained assumption that downstream demand shocks are the
only source of fluctuations, and when those shocks are independent across downstream
regions, the variance of a supplier's sales growth is *exactly* the Herfindahl index
of its customer-sales portfolio, and the covariance between two upstream regions is
*exactly* the inner product of their portfolios. Diversification is therefore not a
descriptive statistic bolted onto the model: it is the model's own volatility object.
We characterise how the portfolio responds to the three primitives the paper
estimates — trade costs `α`, comparative advantage `T_{r's}`, and the geography of
downstream demand `X_{rs}` — in closed form, through a single softmax lemma. Two
results organise the discussion. First, a cell's *own* comparative advantage
determines the *size* of its sales but is irrelevant for the *composition* of its
customer portfolio; comparative advantage acts on diversification only through the
competition it creates at destinations (`Φ_{rs}`) and, in the granular model, through
selection on productivity. Second, the Herfindahl is the term the paper's own
Appendix B.5 decomposition calls `1 - a`: the untargeted moment `δ/γ` equals
`-θα` times `(Λ̄ + 1 - H̃)`, so the cross-industry contrast in `δ/γ` *is* a
cross-industry contrast in customer diversification. We propose a small set of
statistics, all computable from artefacts the estimation already writes, and state
the identification and measurement caveats.

---

## 1. Motivation

The paper already advances a diversification argument in words. Discussing the
untargeted moment it writes: *"In the motor vehicle industry, production is
distributed around many hubs which implies that customer portfolio of suppliers is
very diversified. Moving away from one hub can largely redirect sales toward other
hubs. By contrast, the aerospace industry is much more concentrated around two hubs,
leaving less scope for reallocating sales toward a single downstream region."* The
same idea reappears in the cluster-policy section, where the planner's variance
return `𝒱'_{ks}` is left sign-ambiguous.

Both statements are quantitative claims about an object the model produces but the
paper never reports. This note supplies it, proves the properties that make it the
right object, and shows that it is pinned down by a moment the estimation does not
target.

---

## 2. Environment, notation, and the two portfolios

We keep the paper's notation throughout.

| Symbol | Meaning |
|---|---|
| `r`, `k` | downstream (buyer) region; `R_d` of them |
| `r'`, `l` | upstream (supplier) region |
| `s` | upstream sector; `ρ` a variety; `i` a supplier firm (one variety in one cell) |
| `θ` | Fréchet shape; `T_{r's}` Fréchet scale (comparative advantage) |
| `τ_{r'rs} = d_{r'r}^{α}` | iceberg trade cost; `α` the estimated elasticity |
| `w_{r'}` | upstream wage (normalised to 1 in the estimated model) |
| `Φ_{rs} = Σ_l T_{ls}(w_l τ_{lrs})^{-θ}` | sector-`s` price access of buyer `r` |
| `γ_{r'rs} = T_{r's}(w_{r'}τ_{r'rs})^{-θ}/Φ_{rs}` | sourcing share of `r` on origin `r'` |
| `X_{rs}` | purchases of sector-`s` inputs by the downstream firm in `r` |
| `δ_r^D`, `Σ_{dd}` | downstream demand shock and its covariance matrix |
| `ρ_{r'rs}(z)`, `ρ̃_{r's}(z)`, `Λ_{r'rs} = -ln ρ_{r'rs}` | win probability at `r`, win-somewhere probability, and its log |
| `N_s`, `q̂_{r's}` | varieties per cell; unconditional probability that a variety of `(r',s)` supplies the industry |

*(As in the paper, `w` carries two meanings — the upstream wage `w_{r'}` and the
portfolio weights `w_r^{sr'd}` below. We keep the paper's convention; the superscript
disambiguates.)*

**The two portfolios.** The paper defines the cell-level customer portfolio

```
                    γ_{r'rs} X_{rs}
   w_r^{sr'd}  =  ────────────────────── ,          Σ_r w_r^{sr'd} = 1,          (1)
                  Σ_{r''} γ_{r'r''s} X_{r''s}
```

the share of the sales of cell `(r',s)` to the downstream industry that go to buyer
region `r`, and its firm-level analogue `a_{ir}` (Appendix B.5's `a^D_{ri}`), the
share of supplier `i`'s sales going to `r`. The two are linked by

```
   w_r^{sr'd} = Σ_{i ∈ (r',s)} σ_i a_{ir},          σ_i ≡ x_i / Σ_{j∈(r',s)} x_j,   (2)
```

with `σ_i` the supplier's share of its cell's sales. In the continuum limit
(`N_s → ∞`) the two coincide in expectation; with finitely many varieties they do
not, and Proposition 4 measures the wedge.

**Definition 1 (customer Herfindahl and effective number of customers).**

```
   H_i    ≡ Σ_r a_{ir}²          N_i    ≡ 1/H_i        (firm level)
   H_{r's} ≡ Σ_r (w_r^{sr'd})²    N_{r's} ≡ 1/H_{r's}   (cell level)
```

`N` is the Hill number of order 2 of the portfolio: the number of *equally sized*
customers that would deliver the same concentration. `1 ≤ N ≤ R_d`.

---

## 3. Why the Herfindahl, and not some other index of concentration

### 3.1 Diversification is variance

**Assumption 1 (shock transmission).** Downstream demand shocks `δ^D` are the only
source of fluctuations; they are mean zero with covariance `Σ_{dd}`; and the network
— the sourcing shares `γ`, the portfolio weights, and all prices — is held at its
estimated equilibrium value, so that transmission is evaluated to first order. This
is the paper's own assumption in §5.3, where `d ln x_{it} = Σ_r a^D_{ri} δ^D_{r,t}`.

**Proposition 1 (variance representation).** Under Assumption 1,

```
   Var( d ln x_i ) = a_i' Σ_{dd} a_i .
```

If in addition the shocks are independent across downstream regions with common
variance `σ²` (`Σ_{dd} = σ² I`), then

```
   Var( d ln x_i ) = σ² Σ_r a_{ir}² = σ² H_i = σ² / N_i .                        (3)
```

*Proof: Appendix A.1.*

Equation (3) is the reason to prefer the Herfindahl over any other index of
concentration. Entropy-based measures (the Hill number of order 1,
`exp(-Σ_r a ln a)`) and simple customer counts are legitimate descriptions of a
portfolio, but only the order-2 index is the variance the model implies. Under
Assumption 1 the statement *"aerospace suppliers are less diversified than motor
vehicle suppliers"* and the statement *"aerospace suppliers' sales are more volatile
by a factor `H_aero/H_auto`"* are the same statement.

### 3.2 Comovement is an angle, diversification is a length

Let `M_{r'r} = Σ_s w_s^{r'} w_d^{sr'} w_r^{sr'd}` be the paper's network matrix and
`ζ_{r'} ≡ Σ_r M_{r'r} = Σ_s w_s^{r'} w_d^{sr'} ∈ [0,1]` region `r'`'s exposure to the
downstream industry. Write `M̃_{r'·} ≡ M_{r'·}/ζ_{r'}` for the normalised row, which
is a portfolio (it sums to one).

**Proposition 2 (Gram representation of spatial comovement).** Under Assumption 1
with `Σ_{dd} = σ² I`,

```
   Cov(r', l)  = σ² ζ_{r'} ζ_l ⟨ M̃_{r'·} , M̃_{l·} ⟩
   Var(r')     = σ² ζ_{r'}² H^M_{r'} ,        H^M_{r'} ≡ ‖M̃_{r'·}‖²
   Corr(r', l) = ⟨M̃_{r'·}, M̃_{l·}⟩ / (‖M̃_{r'·}‖ ‖M̃_{l·}‖) = cos ∠(M̃_{r'·}, M̃_{l·}).
```

The matrix of spatial comovements is the Gram matrix of the customer portfolios.

*Proof: Appendix A.2.*

Proposition 2 separates two things the paper's narrative currently bundles. The
*length* of a portfolio vector is its owner's own volatility — diversification. The
*angle* between two portfolios is their comovement — similarity. A region can be
poorly diversified without generating aggregate comovement (if it is alone in its
exposure), and an industry can be made of well-diversified suppliers that nonetheless
all point in the same direction. Section 8 argues that aerospace is extreme on
*both* margins, and that the aggregate consequence runs mostly through the angle.

### 3.3 Two margins inside the Herfindahl

**Proposition 3 (extensive/intensive decomposition).** Let `n_i ≡ #{r : a_{ir} > 0}`
and let `CV_i` be the coefficient of variation of `{a_{ir}}` over the `n_i` served
destinations. Then

```
   H_i = (1 + CV_i²) / n_i ,        equivalently        N_i = n_i / (1 + CV_i²) ≤ n_i .   (4)
```

*Proof: Appendix A.3.*

The two factors are the model's two margins: `n_i` is the extensive margin
(Proposition 8 gives it in closed form), `CV_i` the intensive margin. `N_i ≤ n_i`
with equality iff the supplier splits its sales equally, so the effective number of
customers is always a lower bound on the raw count — which is why the raw count,
reported alone, overstates diversification.

### 3.4 Aggregation: firms, cells, regions

**Proposition 4 (aggregation gain).** With `σ_i` as in (2),

```
   Σ_i σ_i H_i  -  H_{r's}  =  Σ_i σ_i Σ_r (a_{ir} - w_r^{sr'd})²  ≥  0 .          (5)
```

*Proof: Appendix A.4.*

A cell is never more concentrated than the average of the firms it contains, and the
gap is exactly the between-firm dispersion of portfolios inside the cell. This is the
granular content of the model: with few varieties per sector the cell inherits its
firms' concentration; with many, cross-firm heterogeneity diversifies it away. In the
estimated fits the variety counts are `N̂_s ∈ [2, 23]` in motor vehicles and
`[3, 14]` in aerospace, so the aggregation gain is bounded and industry-specific,
and (5) says how to measure it rather than assume it.

---

## 4. What the model implies for the portfolio

Everything in this section holds `X_{rs}` fixed, as the paper does when it derives
its equations (12)–(13).

### 4.1 The portfolio is a softmax, and own comparative advantage drops out

**Proposition 5 (portfolio composition).** Define the *destination attractiveness*
`m_{rs} ≡ X_{rs} / Φ_{rs}`. Then

```
   w_r^{sr'd}  =  m_{rs} τ_{r'rs}^{-θ} / Σ_{r''} m_{r''s} τ_{r'r''s}^{-θ}
               =  softmax_r ( ln m_{rs} - θα ln d_{r'r} )                        (6)
```

under the power law `τ_{r'r} = d_{r'r}^{α}`. In particular `w_r^{sr'd}` does not
depend on `T_{r's}` nor on `w_{r'}`.

*Proof: Appendix A.5.*

**Corollary 5.1.** A cell's own comparative advantage determines the *level* of its
sales to the industry, not the *composition* of its customer portfolio. At the cell
level, comparative advantage affects diversification only through general
equilibrium: the `T` of *other* origins enter `Φ_{rs}`, hence `m_{rs}`. (In the
granular model it re-enters at the firm level through selection: Proposition 8.)

Corollary 5.1 matters for interpretation. It says that "Toulouse is a hub" cannot, by
itself, make Toulouse's own suppliers concentrated; what makes them concentrated is
that Toulouse is *where the buyer is*, i.e. the geography of `m`, plus the trade cost
that ties them to it.

### 4.2 One lemma, four comparative statics

**Lemma 1 (softmax derivative of a Herfindahl).** Let `w_r(t) = e^{v_r(t)}/Σ_{r''}
e^{v_{r''}(t)}` and `H(t) = Σ_r w_r(t)²`. Then

```
   ẇ_r = w_r ( v̇_r - E_w[v̇] )              and        Ḣ = 2 Cov_w( w , v̇ ) ,     (7)
```

where `E_w` and `Cov_w` are taken under the probability `w` over destinations, so
that `Cov_w(w, v̇) = Σ_r w_r² v̇_r - H · Σ_r w_r v̇_r`.

*Proof: Appendix A.6.*

Applying Lemma 1 to (6) with `v_r = ln m_{rs} - θα ln d_{r'r}` gives all the
comparative statics of interest. Write `x_r ≡ -θ ln d_{r'r}` for *proximity*.

**Proposition 6 (trade costs).** Holding `m` fixed,

```
   ∂H_{r's}/∂α  =  2 Cov_w( w , x ) .                                            (8)
```

Allowing `Φ` to adjust (holding `T` and `X` fixed), the same expression holds with
`x_r` replaced by *relative* proximity

```
   ξ_r ≡ x_r - Σ_l γ_{lrs} x_{lr}  =  -θ ( ln d_{r'r} - Σ_l γ_{lrs} ln d_{lr} ) ,  (9)
```

i.e. how much closer `r'` is to `r` than `r`'s average incumbent supplier. Moreover
`lim_{α→∞} H_{r's} = 1` whenever the nearest destination is unique, and
`lim_{α→0} H_{r's} = Σ_r m̄_{rs}²` with `m̄ ≡ m/Σm`, the same for every origin.

*Proof: Appendix A.7.*

Two readings. (i) The limits are unambiguous: with no trade cost every supplier holds
the *market portfolio* and diversification is entirely a property of the demand
geography; with prohibitive trade costs every supplier is a single-customer supplier.
(ii) The local effect is *not* signed in general. `Cov_w(w, x) > 0` — trade costs
concentrate — whenever weight and proximity are positively associated in the
portfolio, which is the typical case and is guaranteed when demand is spatially
uniform (`m_{rs}` constant, Corollary 6.1 in the appendix). But a supplier whose
dominant customer is *far* (a small remote cell selling into a distant hub) is
*diversified* by an increase in `α` before being concentrated by it. Reporting (8)
alongside the level is therefore informative, and the sign is an empirical matter,
not a theorem. Appendix A.7 gives an explicit two-destination counterexample.

**Proposition 7 (comparative advantage).** For any origin `k`,

```
   ∂ ln w_r^{sr'd} / ∂ ln T_{ks}  =  Σ_{r''} w_{r''}^{sr'd} γ_{kr''s} - γ_{krs}   (10)
   ∂ H_{r's}      / ∂ ln T_{ks}  =  - 2 Cov_w( w , γ_{k·s} ) .                    (11)
```

*Proof: Appendix A.8.*

Equation (10) *is* the paper's equation (13), obtained here as a corollary of
Proposition 5 — which is a useful cross-check, since the two derivations are
independent (the paper's goes through the indicator cancellation, ours through the
`T`-independence of the composition). Equation (11) is new and is the answer to
"what does comparative advantage do to diversification":

> Raising the comparative advantage of region `k` **diversifies** every supplier's
> portfolio if and only if `k` competes at the destinations where that supplier is
> already concentrated (`Cov_w(w, γ_{k·s}) > 0`), and **concentrates** it otherwise.

The effect is the same for every origin `r'` (the paper's "uniform within-portfolio
reallocation"), but its *sign* is origin-specific because it depends on where that
origin's weight sits.

**Proposition 8 (downstream demand geography).** For any buyer `k`,

```
   ∂ H_{r's} / ∂ ln X_{ks}  =  2 w_k^{sr'd} ( w_k^{sr'd} - H_{r's} ) .            (12)
```

A demand expansion at `k` concentrates the portfolio iff `k`'s weight exceeds the
sales-weighted average weight `H`. Summing (12) over `k` gives zero, as homogeneity
of degree zero requires.

*Proof: Appendix A.9.*

### 4.3 The extensive margin in closed form, and how comparative advantage re-enters

**Proposition 9 (multi-homing).** Let `n_ρ` be the number of downstream regions
served by variety `ρ` of cell `(r',s)`. Then

```
   E[ n_ρ ]        =  Σ_r γ_{r'rs}                                                (13)
   E[ n_ρ | n_ρ ≥ 1 ]  =  Σ_r γ_{r'rs} / q̂_{r's} ,                                (14)
```

where `q̂_{r's} = E_z[ ρ̃_{r's}(z) ]`. `E[n_ρ]` is strictly increasing in `T_{r's}`;
so is `q̂_{r's}`, and Appendix A.10 gives the likelihood-ratio argument under which
the conditional rate (14) inherits that monotonicity. Moreover the win events `{W_r}`
are positively associated conditional on `z`, so that

```
   ρ̃_{r's}(z)  ≤  1 - ∏_r ( 1 - ρ_{r'rs}(z) ) ,                                   (15)
```

i.e. the FKG product formula used in the paper's Appendix B *overstates* the
win-somewhere probability, and therefore *understates* the multi-homing rate (14).

*Proof: Appendix A.10.*

Three implications.

1. **The extensive margin of diversification is free.** `Σ_r γ_{r'rs}` needs no
   simulation, and `q̂_{r's}` is already computed by the estimator (it is the object
   the count moment `Ḡ_s(0) = mean_l (1-q̂_{ls})^{N̂_s}` is built from). The ratio
   (14) is the *multi-homing intensity* and, by Proposition 3, an upper bound on the
   effective number of customers.
2. **Comparative advantage does raise firm-level diversification** — through
   selection, not composition. Conditional on productivity `z`, the win probability
   `ρ_{r'rs}(z) = exp(-Φ_{rs}(w_{r'}τ_{r'rs})^θ z^{-θ})` does not involve `T_{r's}`;
   but `z` is Fréchet with scale `T_{r's}`, so a higher `T` first-order stochastically
   dominates, and every `γ_{r'rs}` rises. A strong cell hosts productive firms, and
   productive firms win more markets. This is the firm-level counterpart of
   Corollary 5.1's cell-level irrelevance, and the two must not be conflated.
3. **The FKG bound has a sign.** (15) says the true reach of a supplier is *smaller*
   than the product formula suggests but its conditional multi-homing is *larger*;
   both go through `q̂`, so the direction of the bias in any diversification statistic
   built on `q̂` is known rather than guessed.

---

## 5. The bridge: diversification *is* the untargeted moment

Appendix B.5 of the paper derives, for the reduced-form spatial-comovement regression,

```
   δ/γ  ≃  -θ E(d ln τ / d ln d) ·  E[ ν a (Λ + 1 - a) ] / E[ ν a ]|_{d=1} ,       (16)
```

with `ν ≡ ρ_{r'rs}(z)/ρ̃_{r's}(z)` the selection correction induced by conditioning on
`Sup_i = 1`, and `a ≡ a^D_r(z,d)`. The `(1 - a)` term is described in the paper as
the redirection of sales toward the rest of the portfolio. It is exactly a
Herfindahl.

**Proposition 10 (the transmission factor).** Define the tilted measure
`dP_ν ∝ ν a dP` and

```
   H̃ ≡ E_ν[a] = E[ν a²]/E[ν a] ,    Λ̄ ≡ E_ν[Λ] ,    κ ≡ E[ν a] / E[ν a]|_{d=1} .
```

Then, with `τ = d^α`, (16) is exactly

```
   δ/γ  =  - θα · κ · ( Λ̄ + 1 - H̃ ) .                                            (17)
```

If, within a supplier, `ν` does not vary across destinations, then `H̃` computed over
that supplier's destinations equals its customer Herfindahl `H_i` of Definition 1.

*Proof: Appendix A.11.*

Two comments on what (17) does and does not deliver. It is an exact restatement of
the paper's own decomposition, so it inherits its approximations — in particular the
continuous-distance derivation and the treatment of `κ`, which equals one only if
the tilted mean share is distance-invariant. And `H̃` is a *tilted* Herfindahl: `ν`
is larger for near destinations, so `H̃` overweights the nearby (large) shares and
`H̃ ≥ H` in the typical case. Both wedges are measurable in the simulated economy and
should be reported rather than assumed away.

Subject to that, (17) turns the paper's verbal argument into an accounting identity.
Define the **transmission factor**

```
   TF  ≡  |δ/γ| / (θα)  =  κ ( Λ̄ + 1 - H̃ )  ∈  [0, ~1] .                          (18)
```

`θα` is the elasticity the *estimation targets* (the extensive margin); `δ/γ` is the
elasticity it does *not* target. Their ratio is a diversification statistic. With the
current fits — `θα = 0.27` (motor vehicles) and `0.43` (aerospace), model
`η̂ = -0.166` and `-0.063` — the implied factors are

| | `θα` | `η̂` (model) | `TF` | implied `H̃` (κ=1, Λ̄=0) | implied `Ñ = 1/H̃` |
|---|---|---|---|---|---|
| Motor vehicles | 0.27 | −0.166 | 0.61 | 0.39 | ≈ 2.6 |
| Aerospace | 0.43 | −0.063 | 0.15 | 0.85 | ≈ 1.2 |

*(These are illustrative: `θα`, `η̂`, `Λ̄` and `κ` must be recomputed from one and the
same run before the table enters the paper.)*

The exercise the paper should run is the **consistency check**: compute `H̃`, `Λ̄` and
`κ` directly on the simulated firm-level economy and verify that (17) reproduces the
`η̂` obtained from the PPML regression. If it does, the diversification statistic is
not a new object requiring its own validation — it is the moment the model already
matches, read in a different unit. That is the strongest possible warrant for putting
it in the paper.

---

## 6. The planner's variance return, in closed form

The paper's cluster-policy proposition rests on `𝒱'_{ks} ≡ ∂𝒱/∂ ln T_{ks}`, described
as sign-ambiguous — "negative when `k` is a stabilizing hub, positive when `k` is a
concentrating hub" — but never computed. Under the independence assumption of this
note it has a closed form, and the "stabilizing hub" condition becomes a covariance.

**Proposition 11 (variance return).** Let `𝒱 = Σ_{r'} ω_{r'} Var(r')` for planner
weights `ω`, with `Var(r')` as in Proposition 2. Abstracting from cross-sector
spillovers, so that `M_{r'r} = ζ_{r's} w_r^{sr'd}` for the sector under policy,

```
   ∂𝒱/∂ ln T_{ks}  =  - 2σ² Σ_{r'} ω_{r'} ζ_{r's}² · Cov_{w^{r'}}( w^{r'} , γ_{k·s} ) .  (19)
```

In the general multi-sector case,

```
   ∂𝒱/∂ ln T_{ks} = 2σ² Σ_{r'} ω_{r'} ζ_{r's} Σ_r M_{r'r} w_r^{sr'd} ( γ̄_k^{(r')} - γ_{krs} ),
   γ̄_k^{(r')} ≡ Σ_{r''} w_{r''}^{sr'd} γ_{kr''s} .
```

*Proof: Appendix A.12.*

Region `k` is a **stabilizing** target for cluster policy (`𝒱' < 0`) iff, averaged
over origins with weights `ω ζ²`, it competes at the destinations where existing
suppliers are concentrated. This is precisely Proposition 1's "moderately central but
currently under-supplied" statement, now with a sign test that can be computed for
every commuting zone from `γ̂` and the estimated portfolios — and it slots directly
into the paper's first-order condition (14) without any new machinery.

---

## 7. Proposed measurement

### 7.1 Statistics

All of the following are computable from artefacts the estimation already writes;
none requires re-simulating the model.

| # | Statistic | Definition | Level | Source |
|---|---|---|---|---|
| 1 | Effective number of customers | `N_i = 1/Σ_r a_{ir}²` | firm | `suppliers.parquet` |
| 2 | Customers served | `n_i`; and `Σ_r γ̂_{r'rs}`, `Σ_r γ̂/q̂` in closed form | firm | parquet; `γ̂`, `q̂` |
| 3 | Intensive margin | `CV_i` from (4) | firm | parquet |
| 4 | Cell portfolio | `N_{r's} = 1/Σ_r (w_r^{sr'd})²` | cell | `w_srd_r.npy`, or `γ̂ × X` |
| 5 | **Realised diversification** `𝒦` | `N_{r's} / N^0_s`, `N^0_s ≡ 1/Σ_r m̄_{rs}²` | cell | closed form |
| 6 | Portfolio similarity | `cos ∠(w^{r'}, w^{l})`, and its distance profile | cell pair | closed form |
| 7 | Aggregation gain | equation (5) | cell | parquet |
| 8 | Local volatility | `ζ_{r'}² H^M_{r'}` | region | `M` |
| 9 | Variance return | equation (19) | policy | closed form |
| 10 | Dual: buyer's supplier base | `1/Σ_{r'} γ̂_{r'rs}²` | buyer | `γ̂` |

Statistic 5 deserves emphasis. Raw effective numbers are **not comparable across
industries**: the two samples do not have the same number of downstream regions, nor
the same size distribution of buyers, so a difference in `N` conflates the portfolio
choice with the menu. `𝒦 = N_{r's}/N^0_s` divides by the diversification the demand
geography *offers* — the `α = 0` benchmark of Proposition 6, in which every origin
holds the market portfolio — and is therefore the scale-free statistic for the
cross-industry comparison. Note that `𝒦 > 1` is possible: see the counterexample in
Appendix A.7. It is not a bounded index, and should not be presented as one.

### 7.2 Two decompositions to report

**(a) Counterfactual decomposition.** Recompute `N` under `α = 0`, under `T`
equalised within sector, and under `X_{rs}` uniform, using the closed form (6). This
isolates the three forces and reuses the machinery already written for the
comparative-advantage section of the reporting notebook. Because the three factors
interact, report either the full `2³` grid or the Shapley average over the six
orderings; a single sequential decomposition is order-dependent and should be
labelled as such.

**(b) Firm vs cell.** Report `Σ_i σ_i N_i` and `N_{r's}` side by side; equation (5)
attributes the gap to between-firm portfolio dispersion, which is the granular
margin and is industry-specific through `N̂_s`.

### 7.3 Figures

1. The CDF of `N_i` by industry (model), with the `α=0` market-portfolio benchmark
   `N^0_s` drawn as a vertical line. One picture carries the whole argument.
2. The counterfactual decomposition of 7.2(a) as a bar panel, per industry.
3. A map of `N_{r's}` (or of `𝒦`) over commuting zones. The two-hub versus many-hub
   contrast is a spatial statement and reads best as a map.
4. Optional: portfolio similarity (statistic 6) against distance, by industry — the
   *angle* counterpart of the *length* in figures 1–3, and the object that Proposition
   2 says drives aggregate comovement.

---

## 8. Reading the two industries

The three forces are separately identified in the estimation and can be signed
individually.

**Downstream concentration.** This is the *only* input to the comparison that is
directly observed rather than estimated: `X_{rs}` and the location of downstream
plants are data. `N^0_s = 1/Σ_r m̄_{rs}²` is a small number in aerospace (Toulouse and
Île-de-France, with Marignane, Nantes and Bordeaux behind) and a substantially larger
one in motor vehicles. Proposition 6 says that at `α = 0` this number *is* every
supplier's diversification, so it caps what any supplier in aerospace can achieve. We
recommend reporting `N^0_s` as a descriptive statistic in the data section, before
any model output: it carries a large part of the argument on its own.

**Trade costs.** `θα` is 0.43 in aerospace against 0.27 in motor vehicles. By
Proposition 6, aerospace portfolios sit at a "colder" softmax temperature: at equal
demand geography they would already be more concentrated. **The two forces therefore
compound rather than offset in aerospace**, which is the central mechanical reason the
industry ranking of `δ/γ` reverses the industry ranking of the extensive-margin
gradient — the paper's puzzle. Equation (18) states the reversal as arithmetic: a
larger `θα` multiplied by a much smaller `(1 - H̃)` yields a smaller `|δ/γ|`.

**Comparative advantage.** Here the three results of §4 must be kept apart, because
they do not point the same way.

- At the *cell* level, own `T` is irrelevant for composition (Corollary 5.1). The
  concentration of `T̂` in aerospace does not, by itself, concentrate portfolios.
- Through *competition*, a strong Toulouse raises `Φ` at the destinations near it,
  lowers `m` there, and therefore pushes *distant* suppliers out of the Toulouse
  market — mechanically diversifying them (Proposition 7, `Cov_w(w, γ_{k·s}) > 0`
  for suppliers whose weight sits on Toulouse). These suppliers are small.
- At the *firm* level, high `T` around the hub produces productive firms that win
  many markets (Proposition 9). The large aerospace suppliers should therefore be
  *more* multi-homing than the average, and the sales-weighted and unweighted `N`
  should diverge more in aerospace than in motor vehicles. This is a sharp,
  falsifiable prediction of the granular model and worth reporting as such.

**The aggregate.** Propositions 1 and 2 say the industry-level consequence is not
read off `N` alone. Aerospace is the extreme case on the *angle*: portfolios are
short *and* nearly collinear, because almost every one of them points at the same two
destinations. Motor vehicles has longer portfolios *and* smaller pairwise angles
between suppliers attached to different hubs. Since `Corr` is the cosine and is
invariant to the length, the aggregate comovement contrast is the *similarity*
contrast; the diversification contrast governs the *level* of individual volatility.
Reporting statistics 1 and 6 together is what separates the two, and the paper's
current narrative — "hub regions act as aggregators of network-driven fluctuations" —
is a statement about statistic 6.

---

## 9. Caveats

**Independence of downstream shocks.** Proposition 1's identification of variance with
the Herfindahl requires `Σ_{dd} = σ² I`. Motor vehicles and aerospace are subject to
large industry-wide shocks; with a common factor, every portfolio loads on it and all
effective numbers collapse toward 1. The statistics should therefore be presented as
applying to the *idiosyncratic regional* component of downstream demand — which is
also the component the paper's counterfactual shocks (EV eco-score, military
procurement) are designed to isolate. As a robustness exercise, report `a_i' Σ̂ a_i`
for an estimated `Σ̂` beside `σ² H_i`; Proposition 1's first line is stated for a
general `Σ_{dd}` precisely so that this is available.

**`a_{ir}` is a model object.** The estimation observes the extensive margin at the
industry level (`Sup_i`), the count distribution `G_s(K)`, and the regional sourcing
shares `γ_{ls}` aggregated over buyers — not the bilateral supplier × buyer matrix.
The firm-level portfolio is therefore *not* directly measurable in the data, and the
diversification statistics are model output. This is exactly why §5 matters: the
untargeted moment `δ/γ` is the observable that disciplines them, and (17) is the map.
Statistics that *are* observable — `N^0_s`, the buyer dual (statistic 10, from
`γ̂`) — should be flagged as such and separated from the rest in the table.

**Sales outside the industry.** `a_{ir}` is a portfolio share *within* the modelled
downstream industry. A real supplier also sells elsewhere, which is a diversification
channel the model does not represent. The reported `N` therefore *understate* true
diversification, in the same direction as the bias the paper already notes on
`δ/γ`. If a firm's industry share `φ_i` is available, `H_i^{total} = φ_i² H_i + …`
bounds the correction; absent that, the statistics are conditional on industry
exposure and should be labelled so.

**Fixed-network first order.** Assumption 1 evaluates transmission at the estimated
network. Over the horizon at which shocks propagate this is the paper's own
convention, but the comparative statics of §4 are precisely about the network moving,
so the two must not be applied simultaneously: (8), (11) and (12) describe how a
*policy* or a *parameter* moves the portfolio, not how a *shock* does.

**Simulation noise.** Firm-level statistics computed on `suppliers.parquet` inherit
the simulation design. `N_i` is a ratio of sums of squares of simulated shares; its
sampling variability across draw sets should be reported (the estimation already
carries the machinery for re-simulating at independent seeds), especially for the
distributional statement of Figure 1.

---

## Appendix A. Proofs

Throughout, `Σ_r` runs over the `R_d` downstream regions, portfolios are elements of
the simplex, and `E_w`, `Cov_w` denote moments taken under the probability `w` on
that index set: `E_w[f] = Σ_r w_r f_r`, `Cov_w(f,g) = Σ_r w_r f_r g_r - E_w[f]E_w[g]`.

### A.1 Proposition 1

Supplier `i`'s sales are `x_i = Σ_r X_{ir}`, where `X_{ir}` is its sales to the
downstream firm of region `r`. Under Assumption 1 a demand shock shifts that firm's
scale, `d ln X_{ir} = δ^D_r`, with the cost shares held fixed. Hence
`d x_i = Σ_r X_{ir} δ^D_r` and, dividing by `x_i`,

```
   d ln x_i = Σ_r a_{ir} δ^D_r = a_i' δ^D ,        a_{ir} = X_{ir}/x_i ,
```

which is the paper's transmission equation. Therefore
`Var(d ln x_i) = a_i' Σ_{dd} a_i`. With `Σ_{dd} = σ² I` this is `σ² Σ_r a_{ir}²
= σ² H_i`. ∎

### A.2 Proposition 2

Regional upstream activity growth is `d ln y_{r'} = Σ_r M_{r'r} δ^D_r`, so
`Cov(r',l) = M_{r'·} Σ_{dd} M_{l·}'`. With `Σ_{dd} = σ² I`,
`Cov(r',l) = σ² Σ_r M_{r'r} M_{lr} = σ² ζ_{r'} ζ_l ⟨M̃_{r'·}, M̃_{l·}⟩` by definition
of `ζ` and `M̃`. Setting `l = r'` gives `Var(r') = σ² ζ_{r'}² ‖M̃_{r'·}‖²`, and
`H^M_{r'} = ‖M̃_{r'·}‖²` because `M̃_{r'·}` sums to one. Finally

```
   Corr(r',l) = Cov(r',l)/√(Var(r')Var(l))
              = σ²ζ_{r'}ζ_l⟨M̃_{r'·},M̃_{l·}⟩ / (σ²ζ_{r'}ζ_l‖M̃_{r'·}‖‖M̃_{l·}‖) ,
```

the `ζ`'s cancelling, which is the cosine of the angle between the two portfolios. ∎

### A.3 Proposition 3

Let `S_i = {r : a_{ir}>0}`, `|S_i| = n_i`. Since `Σ_{r∈S_i} a_{ir} = 1`, the mean of
the positive shares is `ā = 1/n_i`. Write `V` for their variance, so
`CV_i² = V/ā² = n_i² V`. Then

```
   H_i = Σ_{r∈S_i} a_{ir}² = n_i ( V + ā² ) = n_i ( V + 1/n_i² )
       = n_i V + 1/n_i = (1/n_i)( n_i² V + 1 ) = (1 + CV_i²)/n_i .
```

`N_i = n_i/(1+CV_i²) ≤ n_i`, with equality iff `V = 0`. ∎

### A.4 Proposition 4

Fix `r`. With `Σ_i σ_i = 1` and `w_r = Σ_i σ_i a_{ir}` from (2),

```
   Σ_i σ_i (a_{ir} - w_r)² = Σ_i σ_i a_{ir}² - 2 w_r Σ_i σ_i a_{ir} + w_r²
                           = Σ_i σ_i a_{ir}² - w_r² .
```

Summing over `r` gives `Σ_i σ_i H_i - H_{r's} = Σ_i σ_i Σ_r (a_{ir}-w_r)² ≥ 0`. ∎
*(This is the conditional-variance decomposition; equivalently, Jensen's inequality
applied to the convex map `a ↦ a²`.)*

### A.5 Proposition 5

Substituting `γ_{r'rs} = T_{r's}(w_{r'}τ_{r'rs})^{-θ}/Φ_{rs}` into (1),

```
   γ_{r'rs} X_{rs} = [ T_{r's} w_{r'}^{-θ} ] · τ_{r'rs}^{-θ} · ( X_{rs}/Φ_{rs} )
                   = [ T_{r's} w_{r'}^{-θ} ] · τ_{r'rs}^{-θ} m_{rs} .
```

The bracketed factor does not depend on the destination `r`, so it is common to the
numerator and to every term of the denominator of (1) and cancels:

```
   w_r^{sr'd} = m_{rs} τ_{r'rs}^{-θ} / Σ_{r''} m_{r''s} τ_{r'r''s}^{-θ} .
```

Under `τ_{r'r} = d_{r'r}^α`, `τ_{r'rs}^{-θ} = exp(-θα ln d_{r'r})`, giving the softmax
form (6). Neither `T_{r's}` nor `w_{r'}` appears. ∎

*Remark.* The cancellation needs only that the origin-specific components of `γ`
factor out of the destination index; it does not require `τ` to be sector-neutral,
and it survives any origin-level shifter (a productivity subsidy, a wage change).

### A.6 Lemma 1

Write `Z(t) = Σ_{r''} e^{v_{r''}(t)}`, so `w_r = e^{v_r}/Z` and
`d ln Z/dt = Σ_{r''}(e^{v_{r''}}/Z) v̇_{r''} = E_w[v̇]`. Then

```
   ẇ_r = w_r ( v̇_r - d ln Z/dt ) = w_r ( v̇_r - E_w[v̇] ) ,
```

which is (7)(i) and implies `Σ_r ẇ_r = 0`. For (7)(ii),

```
   Ḣ = 2 Σ_r w_r ẇ_r = 2 Σ_r w_r² ( v̇_r - E_w[v̇] ) = 2 ( Σ_r w_r² v̇_r - H E_w[v̇] ) .
```

Taking `f_r = w_r` and `g_r = v̇_r` in the definition of `Cov_w`,

```
   Cov_w(w, v̇) = Σ_r w_r · w_r · v̇_r - ( Σ_r w_r · w_r )( Σ_r w_r v̇_r )
               = Σ_r w_r² v̇_r - H E_w[v̇] ,
```

since `E_w[w] = Σ_r w_r² = H`. Hence `Ḣ = 2 Cov_w(w, v̇)`. ∎

### A.7 Proposition 6

*Partial effect.* In (6), `v_r = ln m_{rs} + α x_r` with `x_r = -θ ln d_{r'r}`.
Holding `m` fixed, `v̇_r = x_r`, and Lemma 1 gives `∂H/∂α = 2 Cov_w(w, x)`.

*Total effect.* `Φ_{rs} = Σ_l T_{ls}(w_l d_{lr}^α)^{-θ}`, so

```
   ∂Φ_{rs}/∂α = Σ_l T_{ls} w_l^{-θ} d_{lr}^{-θα} ( -θ ln d_{lr} )
              = Φ_{rs} Σ_l γ_{lrs} x_{lr} ,
```

i.e. `∂ ln Φ_{rs}/∂α = Σ_l γ_{lrs} x_{lr}`. Since `ln m_{rs} = ln X_{rs} - ln Φ_{rs}`
and `X` is held fixed, `v̇_r = x_{r'r} - Σ_l γ_{lrs} x_{lr} = ξ_r`, which is (9);
Lemma 1 then gives `dH/dα = 2 Cov_w(w, ξ)`.

*Limits.* As `α → 0`, `w_r → m_{rs}/Σ_{r''} m_{r''s} = m̄_{rs}`, independent of `r'`,
so `H → Σ_r m̄_{rs}²`. As `α → ∞`, let `r* = argmin_r d_{r'r}` be unique. For `r ≠ r*`,
`w_r/w_{r*} = (m_{rs}/m_{r*s}) exp(-θα(ln d_{r'r} - ln d_{r'r*})) → 0`, so
`w → e_{r*}` and `H → 1`.

**Corollary 6.1 (sufficient condition for the sign).** If `m_{rs}` is constant across
`r`, then `w_r = e^{αx_r}/Σ e^{αx_{r''}}` is a strictly increasing function of `x_r`
(for `α>0`), so `w` and `x` are comonotone and `Cov_w(w,x) ≥ 0` by Chebyshev's sum
inequality, strictly unless all `x_r` are equal. Trade costs then unambiguously
concentrate the portfolio.

**Counterexample (the sign is not general).** Two destinations; set `θ = 1`,
`m = (0.99, 0.01)`, `d_{r'1} = 10`, `d_{r'2} = 1`, so `x = (-ln 10, 0)`. At `α = 0`,
`w = (0.99, 0.01)` and `H = 0.9802`. At `α = ln 99/ln 10 ≈ 1.9956`,
`w ∝ (0.99·10^{-1.9956}, 0.01) = (0.01, 0.01)`, so `w = (0.5,0.5)` and `H = 0.5`.
Hence `H` *falls* over that range and `∂H/∂α < 0` somewhere in it; direct evaluation
at `α = 0` gives `Cov_w(w,x) = -0.02234`, so `∂H/∂α = -0.04468 < 0`. As `α → ∞`,
`H → 1`: the map `α ↦ H` is non-monotone, falling then rising. The economics: a
supplier whose dominant customer is remote is *diversified* by higher trade costs,
until they are high enough to strand it with its nearest customer. ∎

### A.8 Proposition 7

In (6), `T_{ks}` enters only through `Φ_{rs}`, and

```
   ∂ ln Φ_{rs} / ∂ ln T_{ks} = T_{ks}(w_k τ_{krs})^{-θ} / Φ_{rs} = γ_{krs} ,
```

so `v̇_r = ∂ ln m_{rs}/∂ ln T_{ks} = -γ_{krs}`. Lemma 1(i) gives

```
   ∂ ln w_r^{sr'd}/∂ ln T_{ks} = -γ_{krs} + Σ_{r''} w_{r''}^{sr'd} γ_{kr''s} ,
```

which is (10) — and coincides with the paper's equation (13), including the fact
that the answer does not depend on whether `r' = k` (in the paper's derivation the
indicator cancels in the third line; here it never appears, because by Proposition 5
the own-`T` channel is absent from the composition to begin with). Lemma 1(ii) gives
`∂H/∂ ln T_{ks} = 2 Cov_w(w, -γ_{k·s}) = -2 Cov_w(w, γ_{k·s})`. ∎

*Remark.* Both statements hold `X_{rs}` fixed, as the paper does. Allowing
`X_{rs}` to respond adds the terms in the paper's general chain rule.

### A.9 Proposition 8

`X_{ks}` enters `v_r = ln X_{rs} - ln Φ_{rs} + αx_r` only at `r = k`, so
`v̇_r = 1{r=k}` and Lemma 1(ii) gives

```
   ∂H/∂ ln X_{ks} = 2 Cov_w(w, 1_k) = 2 ( Σ_r w_r² 1{r=k} - H Σ_r w_r 1{r=k} )
                  = 2 w_k ( w_k - H ) .
```

Summing over `k`: `2(Σ_k w_k² - H Σ_k w_k) = 2(H - H) = 0`, consistent with `w`
being homogeneous of degree zero in `X_{·s}`. ∎

### A.10 Proposition 9

*(i) The mean.* Varieties are matched across origins: variety `ρ` of sector `s` is
produced in every cell, with independent draws `z_{lρ} ~ Fréchet(θ, T_{ls})`, and
buyer `r` awards variety `ρ` to `argmin_l w_l τ_{lrs}/z_{lρ}`. For independent
Fréchet draws the standard Eaton–Kortum computation gives

```
   P( r' wins variety ρ at r ) = T_{r's}(w_{r'}τ_{r'rs})^{-θ} / Φ_{rs} = γ_{r'rs} .
```

Since `n_ρ = Σ_r 1{r' wins ρ at r}`, linearity of expectation — which requires no
independence across destinations — gives `E[n_ρ] = Σ_r γ_{r'rs}`. Writing
`γ_{r'rs} = A/(A + B_r)` with `A = T_{r's}(w_{r'}τ_{r'rs})^{-θ}` strictly increasing
in `T_{r's}` and `B_r = Σ_{l≠r'} T_{ls}(w_l τ_{lrs})^{-θ}` free of it, each `γ` and
hence `E[n_ρ]` is strictly increasing in `T_{r's}`. Equation (14) follows from
`E[n|n≥1] = E[n]/P(n≥1)` and `P(n≥1) = q̂_{r's}`.

*(ii) Positive association and the FKG bound.* Condition on the origin's own draw
`z ≡ z_{r'ρ}`. The event `W_r = {r' wins at r}` is
`{ z_{lρ} < z · Q_{lr} ∀ l ≠ r' }` for constants `Q_{lr} = (w_l τ_{lrs})/(w_{r'}
τ_{r'rs})`, i.e. a lower orthant in the vector of competitor draws
`(z_{lρ})_{l≠r'}`, whose components are independent. Independent random variables are
associated (Esary–Proschan–Walkup; Harris/FKG), and association is preserved by
monotone functions, so the increasing events `W_r^c` satisfy

```
   P( ∩_r W_r^c | z ) ≥ ∏_r P( W_r^c | z ) = ∏_r ( 1 - ρ_{r'rs}(z) ) .
```

Taking complements, `ρ̃_{r's}(z) = 1 - P(∩_r W_r^c|z) ≤ 1 - ∏_r(1-ρ_{r'rs}(z))`,
which is (15). Consequently `q̂ ≤ q̂^{FKG}` and, `E[n]` being unaffected,
`E[n|n≥1] = E[n]/q̂ ≥ E[n]/q̂^{FKG}`. ∎

*Remark (monotonicity of (14)).* Substituting `u ≡ z^{-θ}`, which is exponential with
rate `T_{r's}`, one has `E[n] = E_u[N(u)]` with `N(u) = Σ_r e^{-κ_{r'rs}u}`,
`κ_{r'rs} = Σ_{l≠r'}T_{ls}(w_lτ_{lrs})^{-θ}(w_{r'}τ_{r'rs})^{θ}`, and
`q̂ = E_u[ρ̃(u)]`; neither `N` nor `ρ̃` depends on `T_{r's}`, which enters only the
distribution of `u`. (One checks `E_u[e^{-κ u}] = T/(T+κ) = γ`, recovering (13).)
The exponential family is ordered by likelihood ratio, `f_{T_2}/f_{T_1}` decreasing in
`u` for `T_2 > T_1`, and this ordering survives tilting by `ρ̃(u)`; hence
`E[n|n≥1] = E_{P̃_T}[N/ρ̃]` is increasing in `T_{r's}` whenever the conditional
multi-homing rate `N(u)/ρ̃(u)` is non-increasing in `u`. That rate equals `R_d` at
`u = 0` and tends to `1` as `u → ∞`; monotonicity in between is intuitive but we do
not prove it, and it should be checked numerically on the estimated economy before
(14) is presented as a monotone comparative static.

### A.11 Proposition 10

Write `E` for the expectation over `(z, r', d)` in (16) and define the tilted
probability `dP_ν = (ν a / E[ν a]) dP`, legitimate because `ν ≥ 0` and `a ≥ 0`. For
any `f`, `E[ν a f] = E[ν a] · E_ν[f]`. Applying this to `f = Λ + 1 - a`,

```
   E[ν a (Λ + 1 - a)] = E[ν a] ( E_ν[Λ] + 1 - E_ν[a] ) = E[ν a]( Λ̄ + 1 - H̃ ) .
```

With `τ = d^α` we have `d ln τ/d ln d = α`, so (16) becomes

```
   δ/γ = -θα · ( E[ν a] / E[ν a]|_{d=1} ) · ( Λ̄ + 1 - H̃ ) = -θα κ ( Λ̄ + 1 - H̃ ) ,
```

which is (17). For the last claim, fix a supplier `i` and take the inner expectation
over destinations only. If `ν_{ir} = ν̄_i` for all `r`, then

```
   H̃_i = E[ν a²]/E[ν a] = ν̄_i Σ_r a_{ir}² / ( ν̄_i Σ_r a_{ir} ) = Σ_r a_{ir}² = H_i ,
```

using `Σ_r a_{ir} = 1`. ∎

*Remark.* `ν_{ir} = ρ_{r'rs}(z)/ρ̃_{r's}(z)` is decreasing in `d_{r'r}` and therefore
larger at the destinations that carry the larger shares, so the tilt overweights the
big positions and `H̃ ≥ H` in the typical configuration. The gap, like `κ`, is
measurable on the simulated economy.

### A.12 Proposition 11

Let `ζ_{r's} ≡ w_s^{r'} w_d^{sr'}` and, abstracting from cross-sector spillovers,
`M_{r'r} = Σ_{s'} ζ_{r's'} w_r^{s'r'd}`. With `𝒱 = Σ_{r'} ω_{r'} Var(r')` and
Proposition 2, `𝒱 = σ² Σ_{r'} ω_{r'} Σ_r M_{r'r}²`, so

```
   ∂𝒱/∂ ln T_{ks} = 2σ² Σ_{r'} ω_{r'} Σ_r M_{r'r} ∂M_{r'r}/∂ ln T_{ks} .
```

Only sector `s`'s portfolio moves, and by (10)
`∂M_{r'r}/∂ ln T_{ks} = ζ_{r's} w_r^{sr'd} ( γ̄_k^{(r')} - γ_{krs} )` with
`γ̄_k^{(r')} = Σ_{r''} w_{r''}^{sr'd} γ_{kr''s}`, which is the general expression in
the text. If the economy has a single upstream sector, `M_{r'r} = ζ_{r's} w_r^{sr'd}`
and

```
   ∂𝒱/∂ ln T_{ks} = 2σ² Σ_{r'} ω_{r'} ζ_{r's}² Σ_r w_r ( w_r ) ( γ̄_k^{(r')} - γ_{krs} )
                  = 2σ² Σ_{r'} ω_{r'} ζ_{r's}² ( γ̄_k^{(r')} H_{r's} - Σ_r w_r² γ_{krs} )
                  = - 2σ² Σ_{r'} ω_{r'} ζ_{r's}² Cov_{w^{r'}}( w^{r'} , γ_{k·s} ) ,
```

using `Cov_w(w,γ) = Σ_r w_r²γ_{krs} - H E_w[γ] = Σ_r w_r²γ_{krs} - H γ̄_k`. ∎

*Remark.* This holds `X_{rs}` and the exposures `ζ` fixed, exactly as the paper's
equations (12)–(13) do. Letting `ζ_{r's}` respond to `T_{ks}` adds a level term that
is first-order in the efficiency channel already priced in the planner's objective.

---

## Appendix B. Correspondence with the code

| Object | Where it is | Note |
|---|---|---|
| `a_{ir}` panel, zero-filled | `analysis_granular.ipynb`, `build_a_ir_panel` | one row per (supplier, downstream region); `a_ir` already computed |
| `w_r^{sr'd}` | `<run>/<step>/w_srd_r.npy`, `[s, r', r]` | written by `write_post_hoc` in `main.jl` |
| `γ_{r'rs}` (closed form) | `sourcing_geometry(data)["by_sector"][s]["rho"]` | `psi/tot`, cells × downstream |
| `Φ_{rs}` | the `tot` array inside `sourcing_geometry` | so `m_{rs} = X_{rs}/Φ_{rs}` is one line |
| `X_{rs}` | `X_rs.npy` / the downstream purchase column of `suppliers.parquet` | |
| `q̂_{r's}` | the count-moment machinery (`granular_diagnostics.npz`, `Ḡ_s(0)`) | needed for (14) |
| `α̂`, `T̂`, `θ` | `unpack_estimated_T`, `model_theta` | `N_TAU = 1` required for a single `α` |
| `α = 0` / `T` equalised counterfactuals | `sourcing_geometry(..., alpha=0, equalise_T=True)` | already implemented |
| distances `d_{r'r}` | `distances.npy`, model's own 1..R indices | diagonal is the positive internal distance |

Two implementation warnings. First, `w_r^{sr'd}` in `w_srd_r.npy` is built from the
*simulated* firm-level economy, whereas `γ̂ × X` gives its continuum counterpart; they
differ by exactly the granular wedge of Proposition 4, so the two should be reported
together rather than substituted for one another. Second, `share` in
`suppliers.parquet` is a share of the *buyer's unit cost*, so `X_{ir}` must be formed
as `share × downstream_purchase` before any portfolio share is taken — which is what
`build_a_ir_panel` does.

---

## References

Acemoglu, D., V. M. Carvalho, A. Ozdaglar and A. Tahbaz-Salehi (2012), "The Network
Origins of Aggregate Fluctuations," *Econometrica* 80(5), 1977–2016.

Barrot, J.-N. and J. Sauvagnat (2016), "Input Specificity and the Propagation of
Idiosyncratic Shocks in Production Networks," *Quarterly Journal of Economics*
131(3), 1543–1592.

Carvalho, V. M., M. Nirei, Y. U. Saito and A. Tahbaz-Salehi (2021), "Supply Chain
Disruptions: Evidence from the Great East Japan Earthquake," *Quarterly Journal of
Economics* 136(2), 1255–1321.

di Giovanni, J., A. A. Levchenko and I. Méjean (2014), "Firms, Destinations, and
Aggregate Fluctuations," *Econometrica* 82(4), 1303–1340.

Eaton, J. and S. Kortum (2002), "Technology, Geography, and Trade," *Econometrica*
70(5), 1741–1779.

Esary, J. D., F. Proschan and D. W. Walkup (1967), "Association of Random Variables,
with Applications," *Annals of Mathematical Statistics* 38(5), 1466–1474.

Fortuin, C. M., P. W. Kasteleyn and J. Ginibre (1971), "Correlation Inequalities on
Some Partially Ordered Sets," *Communications in Mathematical Physics* 22(2), 89–103.

Gabaix, X. (2011), "The Granular Origins of Aggregate Fluctuations," *Econometrica*
79(3), 733–772.

Hill, M. O. (1973), "Diversity and Evenness: A Unifying Notation and Its
Consequences," *Ecology* 54(2), 427–432.

Kramarz, F., J. Martin and I. Méjean (2020), "Volatility in the Small and in the
Large: The Lack of Diversification in International Trade," *Journal of International
Economics* 122, 103276.
