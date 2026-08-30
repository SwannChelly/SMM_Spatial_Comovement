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
results organise the discussion. First, comparative advantage reaches the
*composition* of a portfolio through one channel only — the market access `Φ_{rs}` it
creates at destinations — and it does so identically whether the origin whose `T` moves
is the supplier's own or a rival's, because the direct, destination-invariant part of
own `T` cancels: `T` sets the *size* of a cell's sales and touches its *composition*
only through competition. Second, the Herfindahl is what the paper's own Appendix B.5
decomposition carries as `1 - a`: the untargeted moment satisfies
`δ/γ = -θα · κ · (Λ̄ + 1 - H̃)`, so a cross-industry contrast in `δ/γ` is, holding the
other three factors fixed, a contrast in customer diversification. We propose a small set of
statistics, all computable from artefacts the estimation already writes, and state
the identification and measurement caveats.

---

## 1. Motivation

The paper already advances a diversification argument in words. Discussing the
untargeted moment it writes: *"In the motor vehicle industry, production is
distributed around many hubs which implies that customer portfolio of suppliers is
very diversified. Moving away from one hub can largely redirect sales toward other
hubs. By contrast, the aerospace industry is much more concentrated around two hubs,
leaving less scope for reallocating sales toward a single downstream region when
moving away from the other."* The
same idea reappears in the cluster-policy section, where the planner's variance
return `𝒱'_{ks}` is left sign-ambiguous.

Both statements are quantitative claims about an object the model produces but the
paper never reports. (Both also sit inside the `\begin{comment}` block that currently
spans lines 245–421 of `structural_2026_endogeneous.tex`, so the equations this note
cross-references — the `d ln w / d ln T` elasticity, the comovement derivative, the
planner's first-order condition and the cluster proposition — carry no number in the
compiled paper. We therefore refer to them by LaTeX label.) This note supplies it, proves the properties that make it the
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

*Symbol collisions inherited from the paper, kept so the notation matches it and
flagged once here: `w` is both the upstream wage `w_{r'}` and the portfolio weight
`w_r^{sr'd}` (the superscript disambiguates); `ρ` is both a variety index and the win
probability `ρ_{r'rs}(z)`; `γ` is both the sourcing share `γ_{r'rs}` and the level
coefficient of the reduced-form regression in the ratio `δ/γ` of §5; `N` serves for the
variety count `N_s` and for effective numbers of customers. Introduced here and not in
the paper: `σ_i` (a supplier's share of its cell's sales, distinct from the shock
variance `σ²`), `κ` (the normalisation constant of §5, distinct from the competition
index `κ_{r'rs}` of §4.3) and `x_r` (proximity, distinct from sales `x_i`). The paper's
`p_{r's}` (its equation `eq:p`) is this note's `q̂_{r's}`.*

**The two portfolios.** The paper defines the cell-level customer portfolio

```
                    γ_{r'rs} X_{rs}
   w_r^{sr'd}  =  ────────────────────── ,          Σ_r w_r^{sr'd} = 1,          (1)
                  Σ_{r''} γ_{r'r''s} X_{r''s}
```

the share of the sales of cell `(r',s)` to the downstream industry that go to buyer
region `r`, and its firm-level analogue `a_{ir}` (Appendix B.5's `a^D_{ri}`), the
share of supplier `i`'s sales going to `r`. Definition (1) is the *continuum*
portfolio, built from `γ × X`. Its granular counterpart is the realised

```
   ŵ_r^{sr'd} ≡ Σ_{i ∈ (r',s)} σ_i a_{ir},          σ_i ≡ x_i / Σ_{j∈(r',s)} x_j,   (2)
```

with `σ_i` the supplier's share of its cell's sales. The two coincide in expectation as
`N_s → ∞` and differ in any finite economy: §4 works with (1), Proposition 4 with (2),
and Appendix B records which artefact holds which.

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
estimated equilibrium value, so that transmission is evaluated to first order. This is
the paper's own transmission equation `d ln x_{it} = Σ_r a^D_{ri} δ^D_{r,t}` (opening
paragraph of "Reorganizing production network"). Appendix B.5 carries in addition a
supplier-specific term `δ_i^D`, which Assumption 1 sets aside; restoring it adds
`Var(δ_i^D)` to every variance below and leaves the rankings of §8 unchanged.

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
by a factor `H_aero/H_auto`"* are the same statement — the second one additionally
requiring the two industries to face the same regional shock variance `σ²`, which is an
assumption and not a result.

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
(Proposition 9 gives its mean in closed form), `CV_i` the intensive margin. `N_i ≤ n_i`
with equality iff the supplier splits its sales equally, so the effective number of
customers is always a lower bound on the raw count — which is why the raw count,
reported alone, overstates diversification.

### 3.4 Aggregation: firms, cells, regions

**Proposition 4 (aggregation gain).** With `σ_i` and the realised cell portfolio
`ŵ^{sr'd}` of (2), and `Ĥ_{r's} ≡ Σ_r (ŵ_r^{sr'd})²`,

```
   Σ_i σ_i H_i  -  Ĥ_{r's}  =  Σ_i σ_i Σ_r (a_{ir} - ŵ_r^{sr'd})²  ≥  0 .          (5)
```

*Proof: Appendix A.4.*

A cell is never more concentrated than the average of the firms it contains, and the
gap is exactly the between-firm dispersion of portfolios inside the cell. This is the
granular content of the model: with few varieties per sector the cell inherits its
firms' concentration; with many, cross-firm heterogeneity diversifies it away. In the
paper's estimated variety counts range over `[2, 23]` in motor vehicles and `[3, 14]`
in aerospace (§"Estimation and fit"), so the aggregation gain is bounded and
industry-specific, and (5) says how to measure it rather than assume it.

---

## 4. What the model implies for the portfolio

Everything in this section holds `X_{rs}` fixed, as the paper does when it derives its
`eq:w_elast_tau` and `eq:w_elast_T`.

### 4.1 The portfolio is a softmax, and the direct own-`T` channel cancels

**Proposition 5 (portfolio composition).** Define the *destination attractiveness*
`m_{rs} ≡ X_{rs} / Φ_{rs}`. Then

```
   w_r^{sr'd}  =  m_{rs} τ_{r'rs}^{-θ} / Σ_{r''} m_{r''s} τ_{r'r''s}^{-θ}
               =  softmax_r ( ln m_{rs} - θα ln d_{r'r} )                        (6)
```

under the power law `τ_{r'r} = d_{r'r}^{α}`. Given `m`, the composition therefore
depends on the origin `r'` only through the vector of bilateral trade costs: the
origin-specific factor `T_{r's} w_{r'}^{-θ}`, being common to every destination,
cancels out of the share.

*Proof: Appendix A.5.*

**Corollary 5.1 (how comparative advantage reaches the composition).** `T_{r's}` is
*not* absent from (6): it sits inside `Φ_{rs}`, and therefore inside `m_{rs}`, exactly
as every other origin's `T` does. What cancels is only the *direct*,
destination-invariant part. Consequently own `T` moves the composition through market
access alone, by the same formula (10) that governs any rival's `T` evaluated at
`k = r'` — which is precisely why the paper's equation `eq:w_elast_T` comes out
independent of whether `r' = k`. Two readings follow.

- **Levels versus composition.** A cell's own comparative advantage is the dominant
  determinant of *how much* it sells (the direct channel), and only a
  market-access-sized determinant of *to whom* (the `Φ` channel). "Toulouse is a hub"
  therefore cannot by itself make Toulouse's suppliers concentrated: what concentrates
  them is that Toulouse is *where the buyer is*, i.e. the geography of `m`, together
  with the trade cost that ties them to it.
- **Sign of the own-`T` channel.** By (11) at `k = r'`,
  `∂H_{r's}/∂ ln T_{r's} = -2 Cov_w(w, γ_{r'·s})`. Since `w_r ∝ γ_{r'rs} X_{rs}`,
  weight and own sourcing share are positively associated whenever demand is not too
  unevenly distributed — and *exactly* so when `X_{·s}` is flat, in which case
  `w` is an increasing function of `γ_{r'·s}` and the covariance is non-negative by
  Chebyshev. Own comparative advantage then **diversifies** the cell, because
  `d ln γ/d ln T = 1 - γ` raises the share proportionally more where the origin is
  currently small, i.e. at the distant destinations. In 2 007 randomly drawn
  geometries the sign is negative in 74% of cells, so this is the typical but not the
  universal case, and the covariance should be reported rather than signed a priori.

(In the granular model comparative advantage re-enters at the firm level through a
third, quite different channel — selection on productivity: Proposition 9.)

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

i.e. how much closer `r'` is to `r` than `r`'s average incumbent supplier. In the
*partial* regime, `lim_{α→0} H_{r's} = Σ_r m̄_{rs}²` with `m̄ ≡ m/Σm`, the same for every
origin, and `lim_{α→∞} H_{r's} = 1` whenever the nearest destination is unique. Both
limits also hold in the total regime, but there the surviving destination is the
`argmin_r ( ln d_{r'r} - ln d_{l*(r)r} )` — *relative* distance, `l*(r)` being `r`'s
cheapest origin — not the nearest destination, and `m̄` must be evaluated at `α = 0`,
where `Φ_{rs}` is buyer-invariant and `m̄ = X̄_{·s}`.

*Proof: Appendix A.7.*

Two readings. (i) The limits are unambiguous: with no trade cost every supplier holds
the *market portfolio* and diversification is entirely a property of the demand
geography; with prohibitive trade costs every supplier is a single-customer supplier.
(ii) The local effect is *not* signed in general. `Cov_w(w, x) > 0` — trade costs
concentrate — whenever weight and proximity are positively associated in the
portfolio, which is the typical case and is guaranteed when `m_{rs}` is constant across
buyers, i.e. when `X_{rs} ∝ Φ_{rs}` (Corollary 6.1 in the appendix; note that *uniform
demand* is not enough, since `Φ_{rs}` varies across buyers with geography). But a supplier whose
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

The *ranking* of destinations by the reallocation they receive is common to every
origin `r'` — destinations are ordered by `-γ_{krs}`, which is origin-free; this is the
paper's "uniform within-portfolio reallocation". The *level* (through the portfolio
average `Σ_{r''} w_{r''}^{sr'd} γ_{kr''s}`) and hence the *sign* of the effect on any
one destination, and on `H`, are origin-specific, because they depend on where that
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

Two conventions must be separated before the extensive margin can be written down. In
the paper's setup each (region, variety) pair hosts a continuum of firms whose
productivities form a Poisson process of mean `T_{r's}z^{-θ}`, so the region's
*champion* is `Fréchet(T_{r's}, θ)` (its equation `eq:app-champion`). The probability
that an *arbitrary* firm of productivity `z` inside that continuum is the cheapest
supplier at `r` must also beat its own region's rivals and is
`exp(-Φ_{rs}(w_{r's}τ_{r'rs})^{θ}z^{-θ})`, which is the paper's `eq:app-rho`. The
probability that the region's *champion* wins is instead

```
   ρ_{r'rs}(z) = exp( -κ_{r'rs} z^{-θ} ),
   κ_{r'rs} = Σ_{l≠r'} T_{ls}(w_l τ_{lrs})^{-θ} · (w_{r'} τ_{r'rs})^{θ}
            = T_{r's} (1 - γ_{r'rs}) / γ_{r'rs} ,                                (13)
```

with the own term *excluded*. It is the champion object that the estimator simulates —
one draw per (cell, variety), the winner taken as the Ricardian argmin across regions —
and only the champion object integrates back to the sourcing share,
`E_z[ρ_{r'rs}(z)] = γ_{r'rs}` (Appendix A.10). Everything below is the champion
convention, and `κ_{r'rs}` rather than `Φ_{rs}(w τ)^θ` is what appears.

**Proposition 9 (multi-homing).** Let `n_ρ` be the number of downstream regions served
by variety `ρ` of cell `(r',s)`. Then

```
   E[ n_ρ ]        =  Σ_r γ_{r'rs}                                                (14)
   E[ n_ρ | n_ρ ≥ 1 ]  =  Σ_r γ_{r'rs} / q̂_{r's} ,                                (15)
```

where `q̂_{r's} = E_z[ ρ̃_{r's}(z) ]` is the paper's `p_{r's}`. `E[n_ρ]` and `q̂_{r's}`
are both strictly increasing in `T_{r's}`; Appendix A.10 gives the likelihood-ratio
argument under which the conditional rate (15) inherits that monotonicity, and states
what is *not* proved there. Moreover the win events `{W_r}` are positively associated,
conditionally on `z` and unconditionally, so that

```
   ρ̃_{r's}(z) ≤ 1 - ∏_r (1 - ρ_{r'rs}(z))   and   q̂_{r's} ≤ 1 - ∏_r (1 - γ_{r'rs}) .  (16)
```

*Proof: Appendix A.10.*

Three implications.

1. **The extensive margin of diversification is free.** `Σ_r γ_{r'rs}` needs no
   simulation, and `q̂_{r's}` is already produced by the estimator as a diagnostic. The
   ratio (15) is the *multi-homing intensity* and, by Proposition 3, an upper bound on
   the mean effective number of customers of the cell's varieties.
2. **Comparative advantage does raise firm-level diversification** — through
   selection, not composition. By (13), `κ_{r'rs}` involves the rivals' `T` and not the
   origin's own, so conditionally on `z` the win probability is free of `T_{r's}`; but
   `z` is Fréchet with scale `T_{r's}`, so a higher `T` first-order stochastically
   dominates and every `γ_{r'rs}` rises. A strong cell hosts productive firms, and
   productive firms win more markets. This channel is distinct from the market-access
   channel of Corollary 5.1 and the two must not be conflated: they can point in
   opposite directions.
3. **The product formula is an upper bound, and it is not what the estimator uses.**
   (16) is the FKG/Harris inequality: positive dependence makes the union *smaller*
   than under independence. Two consequences for the paper. (a) The main text currently
   asserts the reverse inequality, `p_{r's} > 1 - ∏_r(1 - γ_{r'rs})`, while giving
   positive dependence as the reason; the reason is right and the inequality is
   backwards. A Monte-Carlo check on a five-destination economy gives `q̂ = 0.416`
   against a product formula of `0.737`. (b) The product form survives only on the
   *analytical/GMM* path (`compute_regression_quadrature`), which the repository
   already flags as FKG-biased; the `q̂` written to disk by the simulated estimator is
   the exact realised union, so a statistic built on it carries no FKG bias at all.

## 5. The bridge: diversification *is* the untargeted moment

Appendix B.5 of the paper derives, for the reduced-form spatial-comovement regression,

```
   δ/γ  ≃  -θ E(d ln τ / d ln d) ·  E[ ν a (Λ + 1 - a) ] / E[ ν a ]|_{d=1} ,       (17)
```

with `ν ≡ ρ_{r'rs}(z)/ρ̃_{r's}(z)` the selection correction induced by conditioning on
`Sup_i = 1`, and `a ≡ a^D_r(z,d)`. The `(1 - a)` term is described in the paper as
the redirection of sales toward the rest of the portfolio. Averaged with the right
weights it is a Herfindahl.

**Proposition 10 (the transmission factor).** Define the tilted measure
`dP_ν ∝ ν a dP` and

```
   H̃ ≡ E_ν[a] = E[ν a²]/E[ν a] ,    Λ̄ ≡ E_ν[Λ] ,    κ ≡ E[ν a] / E[ν a]|_{d=1} .
```

Then, with `τ = d^α`, (17) is exactly

```
   δ/γ  =  - θα · κ · ( Λ̄ + 1 - H̃ ) .                                            (18)
```

If, within a supplier, `ν` does not vary across destinations, then `H̃` computed over
that supplier's destinations equals its customer Herfindahl `H_i` of Definition 1.

*Proof: Appendix A.11.*

Two comments on what (18) does and does not deliver. It is an exact restatement of
the paper's own decomposition, so it inherits its approximations — in particular the
continuous-distance derivation and the treatment of `κ`, which equals one only if
the tilted mean share is distance-invariant. And `H̃` is a *tilted* Herfindahl: `ν`
is larger for near destinations, so `H̃` overweights the nearby (large) shares and
`H̃ ≥ H` in the typical case. Both wedges are measurable in the simulated economy and
should be reported rather than assumed away.

Subject to that, (18) turns the paper's verbal argument into an accounting identity.
Define the **transmission factor**

```
   TF  ≡  |δ/γ| / (θα)  =  κ ( Λ̄ + 1 - H̃ ) .                                     (19)
```

`θα` is the elasticity the *estimation targets* (the extensive margin); `δ/γ` is the
elasticity it does *not* target. Their ratio is a diversification statistic.

**A scale warning before any number is read.** (17)–(19) are derived for the paper's
`δ/γ`, the ratio of two coefficients of a regression that is *linear in the level* of
`a_{ir}` on `log d`, with the denominator normalised at `d = 1`. The model counterpart
the paper reports, `η̂`, comes instead from a PPML *constant-elasticity* fit
`E[a_{ir}|s,r',r] = exp(α_{rs} + η log Dist)`. These are not the same number. Fitting a
constant elasticity in the linear form puts the intercept at `log d = 0`, so that
`δ/γ ≈ η / (1 + |η| · E[log d])`, with `E[log d] ≈ 5.8` in the estimation sample; the
ratio is mechanically compressed, and `|δ/γ|` is bounded above by `1/E[log d] ≈ 0.17`
whatever the true elasticity. Either `η̂` is converted before entering (19), or the
model's `δ/γ` is re-estimated in the linear form. The table below does the conversion
and reports both.

**A second warning about the implied column.** From (18), `H̃ = 1 + Λ̄ - TF/κ`. Since
`Λ = -ln ρ ≥ 0`, `Λ̄ > 0` strictly, so setting `Λ̄ = 0` and `κ = 1` returns a *lower*
bound on `H̃` and hence an *upper* bound on the effective number of customers. The
column is therefore a bound, not an estimate; and since `H̃ ≤ 1`, admissibility at
`κ = 1` requires `Λ̄ ≤ TF`, which is itself informative about how far `κ` must be from
one.

|  | `θα` | `δ/γ` | `TF` | `H̃` lower bd. | `Ñ = 1/H̃` upper bd. |
|---|---|---|---|---|---|
| Motor vehicles — data (Table 3) | 0.27 | −0.109 | 0.404 | 0.596 | 1.68 |
| Aerospace — data (Table 3) | 0.43 | −0.098 | 0.228 | 0.772 | 1.30 |
| Motor vehicles — model, `η̂ = −0.166` converted | 0.27 | −0.085 | 0.313 | 0.687 | 1.46 |
| Aerospace — model, `η̂ = −0.063` converted | 0.43 | −0.046 | 0.107 | 0.893 | 1.12 |

*(Illustrative: `θα` is read off the current fits, `η̂` from the paper's untargeted-moment
paragraph, and the conversion uses `E[log d] = 5.8`. All four must be recomputed from
one and the same run, with `Λ̄` and `κ` measured rather than set, before the table
enters the paper. Note what the conversion costs: on the raw `η̂` the implied effective
numbers would read 2.6 against 1.2; converted, 1.46 against 1.12. The **ranking** is
robust across all four rows — motor-vehicle portfolios are the more diversified — the
**magnitude** is not.)*

The exercise the paper should run is the **consistency check**: compute `H̃`, `Λ̄` and
`κ` directly on the simulated firm-level economy and verify that (18) reproduces the
model's own `δ/γ`, estimated in the *linear* form so that the two sides are
commensurable. If it does, the diversification statistic is
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
weights `ω`, with `Var(r')` as in Proposition 2, and let
`ζ_{r's} ≡ w_s^{r'} w_d^{sr'}`. In the general case, abstracting from cross-sector
spillovers so that only sector `s`'s portfolio responds to `T_{ks}`,

```
   ∂𝒱/∂ ln T_{ks} = 2σ² Σ_{r'} ω_{r'} ζ_{r's} Σ_r M_{r'r} w_r^{sr'd} ( γ̄_k^{(r')} - γ_{krs} ),
   γ̄_k^{(r')} ≡ Σ_{r''} w_{r''}^{sr'd} γ_{kr''s} .
```

If in addition the upstream economy has a **single** sector, so that
`M_{r'r} = ζ_{r's} w_r^{sr'd}`, this collapses to

```
   ∂𝒱/∂ ln T_{ks}  =  - 2σ² Σ_{r'} ω_{r'} ζ_{r's}² · Cov_{w^{r'}}( w^{r'} , γ_{k·s} ) .  (20)
```

*Proof: Appendix A.12.*

The distinction matters: `M_{r'r} = Σ_{s'} ζ_{r's'} w_r^{s'r'd}` is *already* the
no-cross-sector-spillover object, and (20) additionally requires region `r'` to host no
upstream sector other than `s`. In a three-sector numerical economy the general
expression is exact while (20) is off by 68–95%, so it is the general expression that
should be computed and (20) that should be read as intuition.

Region `k` is a **stabilizing** target for cluster policy (`𝒱' < 0`) iff, averaged
over origins with weights `ω ζ²`, it competes at the destinations where existing
suppliers are concentrated. This is precisely the "moderately central but currently
under-supplied" statement of the paper's cluster proposition (`prop:cluster_policy`),
now with a sign test computable for every commuting zone from `γ̂` and the estimated
portfolios, and it enters the planner's first-order condition (`eq:foc`) with no new
machinery.

One caveat on the object differentiated. The paper's constraint is the full quadratic
form `Ω Σ Ω'` (`eq:planner_problem`), which *includes* the cross-region covariances,
whereas `𝒱 = Σ_{r'} ω_{r'} Var(r')` keeps only the diagonal. By Proposition 2 the
off-diagonal terms are the portfolio *angles*, and §8 argues that they carry the
aggregate contrast — so the diagonal aggregate is the wrong object if the planner cares
about comovement rather than about the sum of local variances. Extending (20) is
mechanical: replace `M_{r'r} M_{r'r}` by `Σ_l Ω_{r'l} M_{lr}` in the proof.

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
| 5 | **Realised diversification** `𝒦` | `N_{r's} / N^0_s`, `N^0_s ≡ 1/Σ_r m̄_{rs}²` **at `α = 0`** (so `m̄ = X̄_{·s}`) | cell | `X_rs.npy` |
| 6 | Portfolio similarity | `cos ∠(w^{r'}, w^{l})`, and its distance profile | cell pair | closed form |
| 7 | Aggregation gain | equation (5) | cell | parquet |
| 8 | Local volatility | `ζ_{r'}² H^M_{r'}` | region | `M` |
| 9 | Variance return | the general expression of Prop. 11 (not (20) unless single-sector) | policy | closed form |
| 10 | Dual: buyer's supplier base | `1/Σ_{r'} γ̂_{r'rs}²` | buyer | `γ̂` |

Statistic 5 deserves emphasis. Raw effective numbers are **not comparable across
industries**: the two samples do not have the same number of downstream regions, nor
the same size distribution of buyers, so a difference in `N` conflates the portfolio
choice with the menu. `𝒦 = N_{r's}/N^0_s` divides by the diversification the demand
geography *offers* — the `α = 0` benchmark of Proposition 6, in which every origin
holds the market portfolio — and is therefore the scale-free statistic for the
cross-industry comparison. `N^0_s` must be evaluated **at `α = 0`**, where `τ ≡ 1`
makes `Φ_{rs}` buyer-invariant and `m̄` collapses to the observed expenditure shares
`X̄_{·s}`; evaluated at the estimated `α` it is a different number (9% away in a
numerical example) and forfeits the observability that makes the benchmark useful. Note that `𝒦 > 1` is possible: see the counterexample in
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
demand geography they would be more concentrated *provided* `Cov_w(w, ξ) > 0` there —
which Proposition 6 shows is typical but not universal, and which must therefore be
measured rather than assumed. If it holds, the two forces compound rather than offset
in aerospace: a candidate mechanical account of why the industry ranking of `δ/γ`
reverses the ranking of the extensive-margin gradient, the paper's puzzle. Note that
(19) does not by itself *explain* that reversal — reading a small `(1 - H̃)` off a small
`|δ/γ|` and then invoking it to account for the same `|δ/γ|` is circular. The identity
becomes an explanation only once `H̃` is computed independently on the simulated
economy, which is exactly the consistency check proposed at the end of §5.

**Comparative advantage.** Here the three channels of §4 must be kept apart, because
they do not point the same way.

- Through *market access*, own and rival `T` act identically (Corollary 5.1): the
  composition responds to `Φ` alone. Own `T` typically **diversifies** the cell, because
  it raises the origin's share proportionally more where that share is currently small,
  i.e. at the distant destinations. A concentrated `T̂` in aerospace therefore does not
  concentrate portfolios through this channel.
- Through *competition*, a strong Toulouse raises `Φ` at the destinations near it,
  lowers `m` there, and pushes *other* suppliers out of the Toulouse market —
  diversifying them, by (11), whenever their weight sits on Toulouse. Whether those
  suppliers are large enough to matter for the industry aggregate is an empirical
  question the statistics of §7 answer directly.
- At the *firm* level, high `T` around the hub produces productive firms that win many
  markets (Proposition 9, implication 2). The prediction is that aerospace's large
  suppliers are *more* multi-homing than its average one. Whether the gap between the
  sales-weighted and the unweighted `N` is *larger* in aerospace than in motor vehicles
  additionally requires the within-sector dispersion of `T̂` to be larger there, which
  is measurable and is not established here.

**The aggregate.** Propositions 1 and 2 say the industry-level consequence is not read
off `N` alone. Two distinct hypotheses follow, and this note offers evidence for
neither — they are what statistics 1 and 6 are for.

1. *Length.* Aerospace portfolios are shorter (`N` smaller), for the demand-geography
   and trade-cost reasons above. This is the individual-volatility statement.
2. *Angle.* Aerospace portfolios are more nearly collinear — motor vehicles should show
   **larger** pairwise angles between suppliers attached to different hubs — because
   almost every aerospace portfolio points at the same two destinations. Since `Corr` is
   the cosine and is invariant to length, this is the aggregate-comovement statement,
   and it does not follow from (1).

The paper's existing sentence, that hub regions "act as aggregators of network-driven
fluctuations" because "their activity loads on shocks affecting a broad set of
downstream locations", is a statement about portfolio *breadth* — statistics 1, 4 and 8,
the length. The collinearity claim is a different one and needs statistic 6. Reporting
both is what keeps them apart.

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
untargeted moment `δ/γ` is the observable that disciplines them, and (18) is the map.
Statistics that *are* observable should be flagged as such and separated from the rest
in the table: `N^0_s` at `α = 0` needs only `X_rs.npy`, whereas the buyer dual
(statistic 10) needs the estimated `γ̂` and is model output.

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
form (6). ∎

*Remark 1.* The cancellation needs only that the origin-specific components of `γ`
factor out of the destination index; it does not require `τ` to be sector-neutral, and
it holds for any origin-level shifter (a productivity subsidy, a wage change).

*Remark 2 (what does **not** follow).* It does not follow that `w_r^{sr'd}` is free of
`T_{r's}` or of `w_{r'}`. Both appear inside `Φ_{rs}`, hence inside `m_{rs}`, and they
do so with a destination-*varying* weight, since `∂ ln Φ_{rs}/∂ ln T_{r's} = γ_{r'rs}`
depends on `r`. What (6) establishes is that only this market-access channel survives:
the direct, destination-invariant part cancels. Numerically, in a five-origin ×
four-destination economy (`θ = 1.768`, `α = 0.35`, wages not all equal) the own-`T`
composition derivative is `(-0.271, 0.070, 0.175, 0.152)` — plainly not zero — and it
reproduces (10) at `k = r'` to `1.8·10⁻¹⁰`.

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

*Limits, partial regime (`m` fixed).* As `α → 0`, `w_r → m_{rs}/Σ_{r''} m_{r''s}
= m̄_{rs}`, independent of `r'`, so `H → Σ_r m̄_{rs}²`. As `α → ∞`, let
`r* = argmin_r d_{r'r}` be unique; for `r ≠ r*`,
`w_r/w_{r*} = (m_{rs}/m_{r*s}) exp(-θα(ln d_{r'r} - ln d_{r'r*})) → 0`, so `w → e_{r*}`
and `H → 1`.

*Limits, total regime.* The limiting values are unchanged but the arguments are not.
At `α → 0`, `τ ≡ 1` makes `Φ_{rs}` buyer-invariant, so `m̄` collapses to the
expenditure shares `X̄_{·s}` — which is what makes the `α = 0` benchmark observable
(§7.1). At `α → ∞`, `Φ_{rs} ~ T_{l*(r)s} w_{l*(r)}^{-θ} d_{l*(r)r}^{-θα}` where `l*(r)`
is `r`'s cheapest origin, so
`w_r ∝ (X_{rs}/T_{l*(r)s}) (d_{r'r}/d_{l*(r)r})^{-θα}` and the surviving destination is
the `argmin_r (ln d_{r'r} - ln d_{l*(r)r})`: **relative**, not absolute, distance. A
two-destination example makes the difference bite: with `r'` at distance 100 from
destination 1 and 300 from destination 2, but destination 1 glued to a rival origin at
distance 10, the nearest destination is 1 while at `α = 3` the portfolio is
`(5·10⁻⁶, 1 - 5·10⁻⁶)` — entirely on destination 2. `H → 1` either way, but the *rate*
is governed by the gap in relative log-distance, which can be small even when the
absolute distances are well separated: in a five-origin economy one origin is still at
`H = 0.67` at `α = 20`, because its two best destinations differ by 0.014 in relative
log-distance.

**Corollary 6.1 (sufficient condition for the sign).** If `m_{rs}` is constant across
`r` — equivalently `X_{rs} ∝ Φ_{rs}`, *not* `X_{rs}` uniform, since `Φ_{rs}` varies
across buyers with geography — then `w_r = e^{αx_r}/Σ e^{αx_{r''}}` is a strictly
increasing function of `x_r` (for `α>0`), so `w` and `x` are comonotone and
`Cov_w(w,x) ≥ 0` by Chebyshev's sum inequality, strictly unless all `x_r` are equal.
Trade costs then unambiguously concentrate the portfolio. Under uniform `X` the sign
can go either way: in a three-origin × two-destination example with `X = (1,1)` and
`α = 0.6`, `Φ = (0.3225, 0.0490)` so `m = (3.10, 20.39)` is far from constant and
`Cov_w(w,x) = -0.105 < 0`.

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

which is (10) — and coincides with the paper's `eq:w_elast_T`, including the fact
that the answer does not depend on whether `r' = k`. The two derivations explain that
independence differently and both are right: in the paper the direct term `1{r'=k}`
appears and cancels against the portfolio average, while here it never appears because
Proposition 5 has already removed the destination-invariant part of own `T` from the
composition. What is *not* removed, in either derivation, is the market-access part —
which is exactly why (10) applies verbatim at `k = r'`. Lemma 1(ii) gives
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
hence `E[n_ρ]` is strictly increasing in `T_{r's}`. Equation (15) follows from
`E[n|n≥1] = E[n]/P(n≥1)` and `P(n≥1) = q̂_{r's}`. `q̂` is increasing in `T_{r's}` for
the same reason: writing `u ≡ z^{-θ}`, which is exponential with rate `T_{r's}` (see
the Remark), `q̂ = E_u[ρ̃(u)]` with `ρ̃` decreasing in `u` and free of `T_{r's}`, and the
exponential family is stochastically decreasing in its rate.

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

Taking complements, `ρ̃_{r's}(z) = 1 - P(∩_r W_r^c|z) ≤ 1 - ∏_r(1-ρ_{r'rs}(z))`, the
first half of (16). The unconditional half follows by mixing over `z`: each
`P(W_r^c|z)` is increasing in `z^{-θ}`, hence a monotone function of the single random
variable `z`, so those functions are associated and

```
   q̂^c ≡ P(∩_r W_r^c) = E_z[ P(∩_r W_r^c|z) ] ≥ E_z[ ∏_r P(W_r^c|z) ]
                      ≥ ∏_r E_z[ P(W_r^c|z) ] = ∏_r (1 - γ_{r'rs}) ,
```

the first inequality by the conditional result and the second by association in `z`.
Consequently `q̂ ≤ q̂^{FKG}` and, `E[n]` being unaffected,
`E[n|n≥1] = E[n]/q̂ ≥ E[n]/q̂^{FKG}`. ∎

*Remark (monotonicity of (15)).* Substituting `u ≡ z^{-θ}`, which is exponential with
rate `T_{r's}`, one has `E[n] = E_u[N(u)]` with `N(u) = Σ_r e^{-κ_{r'rs}u}`,
`κ_{r'rs} = Σ_{l≠r'}T_{ls}(w_lτ_{lrs})^{-θ}(w_{r'}τ_{r'rs})^{θ}`, and
`q̂ = E_u[ρ̃(u)]`; neither `N` nor `ρ̃` depends on `T_{r's}`, which enters only the
distribution of `u`. (One checks `E_u[e^{-κ u}] = T/(T+κ) = γ`, recovering (14).)
The exponential family is ordered by likelihood ratio, `f_{T_2}/f_{T_1}` decreasing in
`u` for `T_2 > T_1`, and this ordering survives tilting by `ρ̃(u)`; hence
`E[n|n≥1] = E_{P̃_T}[N/ρ̃]` is increasing in `T_{r's}` whenever the conditional
multi-homing rate `N(u)/ρ̃(u)` is non-increasing in `u`. That rate equals `R_d` at
`u = 0` and tends to `1` as `u → ∞` (the latter provided `argmin_r κ_{r'rs}` is unique;
with `t` tied destinations the limit is `t`); monotonicity in between is intuitive but
we do not prove it, and it should be checked numerically on the estimated economy
before (15) is presented as a monotone comparative static.

### A.11 Proposition 10

Write `E` for the expectation over `(z, r', d)` in (17) and define the tilted
probability `dP_ν = (ν a / E[ν a]) dP`, legitimate because `ν ≥ 0` and `a ≥ 0`. For
any `f`, `E[ν a f] = E[ν a] · E_ν[f]`. Applying this to `f = Λ + 1 - a`,

```
   E[ν a (Λ + 1 - a)] = E[ν a] ( E_ν[Λ] + 1 - E_ν[a] ) = E[ν a]( Λ̄ + 1 - H̃ ) .
```

With `τ = d^α` we have `d ln τ/d ln d = α`, so (17) becomes

```
   δ/γ = -θα · ( E[ν a] / E[ν a]|_{d=1} ) · ( Λ̄ + 1 - H̃ ) = -θα κ ( Λ̄ + 1 - H̃ ) ,
```

which is (18). For the last claim, fix a supplier `i` and take the inner expectation
over destinations only. If `ν_{ir} = ν̄_i` for all `r`, then

```
   H̃_i = E[ν a²]/E[ν a] = ν̄_i Σ_r a_{ir}² / ( ν̄_i Σ_r a_{ir} ) = Σ_r a_{ir}² = H_i ,
```

using `Σ_r a_{ir} = 1`. ∎

*Remark 1 (which `a` is being used).* The last step divides by `Σ_r a_{ir} = 1`, i.e.
it reads `a` as the *realised portfolio share*. The paper's displayed
`a_r^D(z,d) = γ_{r'rs}x_{rs}/(γ_{r'rs}x_{rs} + Σ_{r''≠r}γ_{r'r''s}x_{r''s}ρ(z))` has an
`r`-dependent denominator and does not sum to one over `r` (it also carries an evident
typo, `ρ_{r'rs}` where `ρ_{r'r''s}` is meant — taken literally, the paper's own step
`d ln a/d ln d = (1-a)·d ln γ/d ln d` would fail, since `ρ_{r'rs}` also depends on
`d_{r'r}`). We read `a` throughout as the realised share, which is both the object
Definition 1 uses and the one the simulated panel measures; the whole bridge to `H_i`
rests on that reading, so it is stated rather than left implicit.

*Remark 2 (the sign of the tilt).* Writing the inner expectation under the probability
`p_r = a_{ir}`, `H̃_i = H_i + Cov_p(ν, a)/E_p[ν]`, so `H̃ ≥ H` **iff** `ν` and `a` are
positively associated across the supplier's destinations. Since
`ν_{ir} = ρ_{r'rs}(z)/ρ̃_{r's}(z)` is decreasing in `d_{r'r}` and shares are larger
nearby, that is the expected configuration — but it is a condition, not a theorem, and
random `(ν, a)` violates it. The gap, like `κ`, is measurable on the simulated
economy.

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
the text. Note that `M_{r'r} = Σ_{s'} ζ_{r's'} w_r^{s'r'd}` *is* the
no-cross-sector-spillover object; collapsing it to a single term additionally requires
region `r'` to host no upstream sector other than `s`. Under that stronger hypothesis,
`M_{r'r} = ζ_{r's} w_r^{sr'd}` and

```
   ∂𝒱/∂ ln T_{ks} = 2σ² Σ_{r'} ω_{r'} ζ_{r's}² Σ_r w_r ( w_r ) ( γ̄_k^{(r')} - γ_{krs} )
                  = 2σ² Σ_{r'} ω_{r'} ζ_{r's}² ( γ̄_k^{(r')} H_{r's} - Σ_r w_r² γ_{krs} )
                  = - 2σ² Σ_{r'} ω_{r'} ζ_{r's}² Cov_{w^{r'}}( w^{r'} , γ_{k·s} ) ,
```

using `Cov_w(w,γ) = Σ_r w_r²γ_{krs} - H E_w[γ] = Σ_r w_r²γ_{krs} - H γ̄_k`. ∎

*Remark.* This holds `X_{rs}` and the exposures `ζ` fixed, exactly as the paper's
`eq:w_elast_tau`–`eq:w_elast_T` do. Letting `ζ_{r's}` respond to `T_{ks}` adds a level
term that is first-order in the efficiency channel already priced in the planner's
objective. For the paper's full quadratic constraint `ΩΣΩ'`, replace `M_{r'r}·M_{r'r}`
in the first display by `Σ_l Ω_{r'l} M_{lr}`; the covariance form of (20) is then a
cross-covariance and no longer collapses to a single `Cov_w`.

---

## Appendix B. Correspondence with the code

| Object | Where it is | Note |
|---|---|---|
| `a_{ir}` panel, zero-filled | `analysis_granular.ipynb`, `build_a_ir_panel` | one row per (supplier, downstream region); `a_ir` already computed |
| `w_r^{sr'd}` | `<run>/<step>/w_srd_r.npy`, `[s, r', r]` | written by `write_post_hoc` in `main.jl` |
| `γ_{r'rs}` (closed form) | `sourcing_geometry(data)["by_sector"][s]["rho"]` | `psi/tot`, cells × downstream |
| `Φ_{rs}` | the `tot` array inside `sourcing_geometry` | `m_{rs} = X_{rs}/tot` is one line — but `tot` sums over that sector's *domestic active cells only*, so if foreign competition belongs in `Φ` the discrepancy is buyer-specific and does not cancel in the softmax |
| `X_{rs}` | `X_rs.npy` / the downstream purchase column of `suppliers.parquet` | |
| `q̂_{r's}` | `granular_diagnostics.npz` (raw diagnostic, alongside the count moment) | needed for (15); it is the **exact** simulated union, not an FKG approximation |
| `α̂`, `T̂`, `θ` | `unpack_estimated_T`, `model_theta` | `N_TAU = 1` required for a single `α` |
| `α = 0` / `T` equalised counterfactuals | `sourcing_geometry(..., alpha=0, equalise_T=True)` | already implemented |
| distances `d_{r'r}` | `distances.npy`, model's own 1..R indices | diagonal is the positive internal distance |

Two implementation warnings. First, `w_srd_r.npy` holds the *realised* portfolio
`ŵ^{sr'd}` of (2), built from the simulated firm-level economy, whereas `γ̂ × X` gives
the continuum object `w^{sr'd}` of (1). They are different vectors — Proposition 4
compares their *Herfindahls*, not the vectors themselves — so the two should be
reported side by side rather than substituted for one another. Second, `share` in
`suppliers.parquet` is a share of the *buyer's unit cost*, so `X_{ir}` must be formed
as `share × downstream_purchase` before any portfolio share is taken — which is what
`build_a_ir_panel` does.

---

## Appendix C. Verification

Every displayed identity in this note was checked numerically before it was written
down, and then re-checked independently. The first pass used finite differences on
asymmetric random economies (5 upstream origins × 4 downstream regions, unequal wages,
`θ = 1.768`) for Lemma 1 and equations (3)–(5), (7)–(12) and (20); Monte-Carlo
simulation of the Ricardian assignment (4·10⁶ draws) for (14)–(16); symbolic algebra
for (18); and exact arithmetic for the counterexample of A.7. That pass is preserved as
`test/test_diversification_identities.py`, which reproduces every identity below and
fails loudly if one of them is edited into something false. A second, independent
pass — different parameterisations, different seeds, `mpmath` at 250 digits for the
inclusion–exclusion sum, and 60 random geometries for the monotonicity conjecture of
A.10 — reproduced all of them and, in addition, produced the three corrections now
incorporated: the own-`T` channel of Corollary 5.1 (which an earlier draft wrongly
declared absent), the champion-versus-arbitrary-firm convention of §4.3, and the
single-sector hypothesis of Proposition 11. The `η̂`-to-`δ/γ` conversion of §5 also
comes from that pass.

Three claims in this note are **not** proved and are flagged where they appear: the
monotonicity of (15) in `T_{r's}` (A.10, Remark — verified in 60 random geometries, no
violation); the sign of `Cov_w(w, ξ)` in any particular industry (§8); and every
statement in §8 about which industry has the shorter or the more collinear portfolios,
which are the hypotheses the statistics of §7 are meant to test.

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
