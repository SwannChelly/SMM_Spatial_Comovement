# Customer diversification in a spatial production network

*Theory, measurement, and the comparison of motor vehicles and aerospace*

**Companion note to §5 of "Spatial Comovements". All notation follows the paper (main text and Appendix B); Appendix A of this note collects the proofs.**

---

## Abstract

We ask how diversified the customer base of an upstream supplier — a firm, or a sector $\times$ region cell — is in the estimated model, and what that diversification is worth. Under the paper's maintained assumption that downstream demand shocks are the only source of fluctuations, and when those shocks are independent across downstream regions, the variance of a supplier's sales growth is *exactly* the Herfindahl index of its customer-sales portfolio, and the covariance between two upstream regions is *exactly* the inner product of their portfolios. Diversification is therefore not a descriptive statistic bolted onto the model: it is the model's own volatility object. We characterise how the portfolio responds to the three primitives the paper estimates — trade costs $\alpha$, comparative advantage $T_{r's}$, and the geography of downstream demand $X_{rs}$ — in closed form, through a single softmax lemma. Two results organise the discussion. First, comparative advantage reaches the *composition* of a portfolio through one channel only — the market access $\Phi_{rs}$ it creates at destinations — and it does so identically whether the origin whose $T$ moves is the supplier's own or a rival's, because the direct, destination-invariant part of own $T$ cancels: $T$ sets the *size* of a cell's sales and touches its *composition* only through competition. Second, the Herfindahl is what the paper's own Appendix B.5 decomposition carries as $1-a$: the untargeted moment satisfies
$$\frac{\delta}{\gamma} \;=\; -\theta\alpha\,\kappa\,\bigl(\bar\Lambda + 1 - \tilde H\bigr),$$
so a cross-industry contrast in $\delta/\gamma$ is, holding the other three factors fixed, a contrast in customer diversification. Third, the Herfindahl is tied to the *distribution of downstream sales* by two exact statements: a supplier's concentration factors into the market portfolio's own Herfindahl $H^X_s$ times a dispersion-of-access term, plus a term measuring whether that access is aligned with buyer size; and because the market portfolio is the sales-weighted average of the suppliers' portfolios, $H^X_s$ is a floor on average supplier concentration at any parameter values — the effective number of downstream buyers caps, on average and not supplier by supplier, the effective number of customers any supplier can have. We propose a small set of statistics, all computable from artefacts the estimation already writes, and state the identification and measurement caveats.

---

## 1. Motivation

The paper already advances a diversification argument in words. Discussing the untargeted moment it writes: *"In the motor vehicle industry, production is distributed around many hubs which implies that customer portfolio of suppliers is very diversified. Moving away from one hub can largely redirect sales toward other hubs. By contrast, the aerospace industry is much more concentrated around two hubs, leaving less scope for reallocating sales toward a single downstream region when moving away from the other."* The same idea reappears in the cluster-policy section, where the planner's variance return $\mathcal V'_{ks}$ is left sign-ambiguous.

Both statements are quantitative claims about an object the model produces but the paper never reports. (Both also sit inside the `\begin{comment}` block that currently spans lines 245–421 of `structural_2026_endogeneous.tex`, so the equations this note cross-references — the $d\ln w/d\ln T$ elasticity, the comovement derivative, the planner's first-order condition and the cluster proposition — carry no number in the compiled paper. We therefore refer to them by LaTeX label.) This note supplies it, proves the properties that make it the right object, and shows that it is pinned down by a moment the estimation does not target.

---

## 2. Environment, notation, and the two portfolios

We keep the paper's notation throughout.

| Symbol | Meaning |
|---|---|
| $r$, $k$ | downstream (buyer) region; $R_d$ of them |
| $r'$, $l$ | upstream (supplier) region |
| $s$ | upstream sector; $\rho$ a variety; $i$ a supplier firm (one variety in one cell) |
| $\theta$ | Fréchet shape; $T_{r's}$ Fréchet scale (comparative advantage) |
| $\tau_{r'rs} = d_{r'r}^{\alpha}$ | iceberg trade cost; $\alpha$ the estimated elasticity |
| $w_{r'}$ | upstream wage (normalised to 1 in the estimated model) |
| $\Phi_{rs} = \sum_l T_{ls}(w_l \tau_{lrs})^{-\theta}$ | sector-$s$ price access of buyer $r$ |
| $\gamma_{r'rs} = T_{r's}(w_{r'}\tau_{r'rs})^{-\theta}/\Phi_{rs}$ | sourcing share of $r$ on origin $r'$ |
| $X_{rs}$ | purchases of sector-$s$ inputs by the downstream firm in $r$ |
| $\delta_r^D$, $\Sigma_{dd}$ | downstream demand shock and its covariance matrix |
| $\rho_{r'rs}(z)$, $\tilde\rho_{r's}(z)$, $\Lambda_{r'rs} = -\ln \rho_{r'rs}$ | win probability at $r$, win-somewhere probability, and its log |
| $N_s$, $\hat q_{r's}$ | varieties per cell; unconditional probability that a variety of $(r',s)$ supplies the industry |

*Symbol collisions inherited from the paper, kept so the notation matches it and flagged once here: $w$ is both the upstream wage $w_{r'}$ and the portfolio weight $w_r^{sr'd}$ (the superscript disambiguates); $\rho$ is both a variety index and the win probability $\rho_{r'rs}(z)$; $\gamma$ is both the sourcing share $\gamma_{r'rs}$ and the level coefficient of the reduced-form regression in the ratio $\delta/\gamma$ of §5; $N$ serves for the variety count $N_s$ and for effective numbers of customers. Introduced here and not in the paper: $\sigma_i$ (a supplier's share of its cell's sales, distinct from the shock variance $\sigma^2$), $\kappa$ (the normalisation constant of §5, distinct from the competition index $\kappa_{r'rs}$ of §4.3) and $x_r$ (proximity, distinct from sales $x_i$). The paper's $p_{r's}$ (its equation `eq:p`) is this note's $\hat q_{r's}$.*

**The two portfolios.** The paper defines the cell-level customer portfolio

$$
w_r^{sr'd} \;=\; \frac{\gamma_{r'rs}\,X_{rs}}{\sum_{r''}\gamma_{r'r''s}X_{r''s}},
\qquad \sum_r w_r^{sr'd} = 1,
\tag{1}
$$

the share of the sales of cell $(r',s)$ to the downstream industry that go to buyer region $r$, and its firm-level analogue $a_{ir}$ (Appendix B.5's $a^D_{ri}$), the share of supplier $i$'s sales going to $r$. Definition (1) is the *continuum* portfolio, built from $\gamma\times X$. Its granular counterpart is the realised

$$
\hat w_r^{sr'd} \;\equiv\; \sum_{i\in(r',s)} \sigma_i\, a_{ir},
\qquad
\sigma_i \equiv \frac{x_i}{\sum_{j\in(r',s)} x_j},
\tag{2}
$$

with $\sigma_i$ the supplier's share of its cell's sales. The two coincide in expectation as $N_s\to\infty$ and differ in any finite economy: §4 works with (1), Proposition 4 with (2), and Appendix B records which artefact holds which.

**Definition 1 (customer Herfindahl and effective number of customers).**

$$
H_i \equiv \sum_r a_{ir}^2, \qquad N_i \equiv \frac{1}{H_i} \quad\text{(firm level)}
$$
$$
H_{r's} \equiv \sum_r \bigl(w_r^{sr'd}\bigr)^2, \qquad N_{r's} \equiv \frac{1}{H_{r's}} \quad\text{(cell level)}
$$

$N$ is the Hill number of order 2 of the portfolio: the number of *equally sized* customers that would deliver the same concentration. $1 \le N \le R_d$.

---

## 3. Why the Herfindahl, and not some other index of concentration

### 3.1 Diversification is variance

**Assumption 1 (shock transmission).** Downstream demand shocks $\delta^D$ are the only source of fluctuations; they are mean zero with covariance $\Sigma_{dd}$; and the network — the sourcing shares $\gamma$, the portfolio weights, and all prices — is held at its estimated equilibrium value, so that transmission is evaluated to first order. This is the paper's own transmission equation $d\ln x_{it} = \sum_r a^D_{ri}\,\delta^D_{r,t}$ (opening paragraph of "Reorganizing production network"). Appendix B.5 carries in addition a supplier-specific term $\delta_i^D$, which Assumption 1 sets aside; restoring it adds $\operatorname{Var}(\delta_i^D)$ to every variance below and leaves the rankings of §8 unchanged.

**Proposition 1 (variance representation).** Under Assumption 1,

$$
\operatorname{Var}\bigl(d\ln x_i\bigr) \;=\; a_i'\,\Sigma_{dd}\,a_i .
$$

If in addition the shocks are independent across downstream regions with common variance $\sigma^2$ ($\Sigma_{dd} = \sigma^2 I$), then

$$
\operatorname{Var}\bigl(d\ln x_i\bigr) \;=\; \sigma^2 \sum_r a_{ir}^2 \;=\; \sigma^2 H_i \;=\; \frac{\sigma^2}{N_i}.
\tag{3}
$$

*Proof: Appendix A.1.*

Equation (3) is the reason to prefer the Herfindahl over any other index of concentration. Entropy-based measures (the Hill number of order 1, $\exp(-\sum_r a\ln a)$) and simple customer counts are legitimate descriptions of a portfolio, but only the order-2 index is the variance the model implies. Under Assumption 1 the statement *"aerospace suppliers are less diversified than motor vehicle suppliers"* and the statement *"aerospace suppliers' sales are more volatile by a factor $H_{\text{aero}}/H_{\text{auto}}$"* are the same statement — the second one additionally requiring the two industries to face the same regional shock variance $\sigma^2$, which is an assumption and not a result.

### 3.2 Comovement is an angle, diversification is a length

Let $M_{r'r} = \sum_s w_s^{r'} w_d^{sr'} w_r^{sr'd}$ be the paper's network matrix and $\zeta_{r'} \equiv \sum_r M_{r'r} = \sum_s w_s^{r'} w_d^{sr'} \in [0,1]$ region $r'$'s exposure to the downstream industry. Write $\tilde M_{r'\cdot} \equiv M_{r'\cdot}/\zeta_{r'}$ for the normalised row, which is a portfolio (it sums to one).

**Proposition 2 (Gram representation of spatial comovement).** Under Assumption 1 with $\Sigma_{dd} = \sigma^2 I$,

$$
\begin{aligned}
\operatorname{Cov}(r',l) &= \sigma^2 \zeta_{r'}\zeta_l \,\bigl\langle \tilde M_{r'\cdot},\, \tilde M_{l\cdot}\bigr\rangle,\\
\operatorname{Var}(r') &= \sigma^2 \zeta_{r'}^2 H^M_{r'},
\qquad H^M_{r'} \equiv \bigl\|\tilde M_{r'\cdot}\bigr\|^2,\\
\operatorname{Corr}(r',l) &= \frac{\langle \tilde M_{r'\cdot}, \tilde M_{l\cdot}\rangle}{\|\tilde M_{r'\cdot}\|\,\|\tilde M_{l\cdot}\|}
= \cos \angle\bigl(\tilde M_{r'\cdot}, \tilde M_{l\cdot}\bigr).
\end{aligned}
$$

The matrix of spatial comovements is the Gram matrix of the customer portfolios.

*Proof: Appendix A.2.*

Proposition 2 separates two things the paper's narrative currently bundles. The *length* of a portfolio vector is its owner's own volatility — diversification. The *angle* between two portfolios is their comovement — similarity. A region can be poorly diversified without generating aggregate comovement (if it is alone in its exposure), and an industry can be made of well-diversified suppliers that nonetheless all point in the same direction. Section 8 argues that aerospace is extreme on *both* margins, and that the aggregate consequence runs mostly through the angle.

### 3.3 Two margins inside the Herfindahl

**Proposition 3 (extensive/intensive decomposition).** Let $n_i \equiv \#\{r : a_{ir} > 0\}$ and let $CV_i$ be the coefficient of variation of $\{a_{ir}\}$ over the $n_i$ served destinations. Then

$$
H_i = \frac{1 + CV_i^2}{n_i},
\qquad\text{equivalently}\qquad
N_i = \frac{n_i}{1 + CV_i^2} \;\le\; n_i .
\tag{4}
$$

*Proof: Appendix A.3.*

The two factors are the model's two margins: $n_i$ is the extensive margin (Proposition 9 gives its mean in closed form), $CV_i$ the intensive margin. $N_i \le n_i$ with equality iff the supplier splits its sales equally, so the effective number of customers is always a lower bound on the raw count — which is why the raw count, reported alone, overstates diversification.

### 3.4 Aggregation: firms, cells, regions

**Proposition 4 (aggregation gain).** With $\sigma_i$ and the realised cell portfolio $\hat w^{sr'd}$ of (2), and $\hat H_{r's} \equiv \sum_r (\hat w_r^{sr'd})^2$,

$$
\sum_i \sigma_i H_i \;-\; \hat H_{r's}
\;=\; \sum_i \sigma_i \sum_r \bigl(a_{ir} - \hat w_r^{sr'd}\bigr)^2 \;\ge\; 0 .
\tag{5}
$$

*Proof: Appendix A.4.*

A cell is never more concentrated than the average of the firms it contains, and the gap is exactly the between-firm dispersion of portfolios inside the cell. This is the granular content of the model: with few varieties per sector the cell inherits its firms' concentration; with many, cross-firm heterogeneity diversifies it away. In the paper's estimated variety counts range over $[2,23]$ in motor vehicles and $[3,14]$ in aerospace (§"Estimation and fit"), so the aggregation gain is bounded and industry-specific, and (5) says how to measure it rather than assume it.

---

## 4. What the model implies for the portfolio

Everything in this section holds $X_{rs}$ fixed, as the paper does when it derives its `eq:w_elast_tau` and `eq:w_elast_T`.

### 4.1 The portfolio is a softmax, and the direct own-$T$ channel cancels

**Proposition 5 (portfolio composition).** Define the *destination attractiveness* $m_{rs} \equiv X_{rs}/\Phi_{rs}$. Then

$$
w_r^{sr'd}
\;=\; \frac{m_{rs}\,\tau_{r'rs}^{-\theta}}{\sum_{r''} m_{r''s}\,\tau_{r'r''s}^{-\theta}}
\;=\; \operatorname*{softmax}_r\bigl(\ln m_{rs} - \theta\alpha \ln d_{r'r}\bigr)
\tag{6}
$$

under the power law $\tau_{r'r} = d_{r'r}^{\alpha}$. Given $m$, the composition therefore depends on the origin $r'$ only through the vector of bilateral trade costs: the origin-specific factor $T_{r's} w_{r'}^{-\theta}$, being common to every destination, cancels out of the share.

*Proof: Appendix A.5.*

**Corollary 5.1 (how comparative advantage reaches the composition).** $T_{r's}$ is *not* absent from (6): it sits inside $\Phi_{rs}$, and therefore inside $m_{rs}$, exactly as every other origin's $T$ does. What cancels is only the *direct*, destination-invariant part. Consequently own $T$ moves the composition through market access alone, by the same formula (10) that governs any rival's $T$ evaluated at $k = r'$ — which is precisely why the paper's equation `eq:w_elast_T` comes out independent of whether $r' = k$. Two readings follow.

- **Levels versus composition.** A cell's own comparative advantage is the dominant determinant of *how much* it sells (the direct channel), and only a market-access-sized determinant of *to whom* (the $\Phi$ channel). "Toulouse is a hub" therefore cannot by itself make Toulouse's suppliers concentrated: what concentrates them is that Toulouse is *where the buyer is*, i.e. the geography of $m$, together with the trade cost that ties them to it.
- **Sign of the own-$T$ channel.** By (11) at $k = r'$, $\partial H_{r's}/\partial \ln T_{r's} = -2\operatorname{Cov}_w(w, \gamma_{r'\cdot s})$. Since $w_r \propto \gamma_{r'rs} X_{rs}$, weight and own sourcing share are positively associated whenever demand is not too unevenly distributed — and *exactly* so when $X_{\cdot s}$ is flat, in which case $w$ is an increasing function of $\gamma_{r'\cdot s}$ and the covariance is non-negative by Chebyshev. Own comparative advantage then **diversifies** the cell, because $d\ln\gamma/d\ln T = 1-\gamma$ raises the share proportionally more where the origin is currently small, i.e. at the distant destinations. In 2 007 randomly drawn geometries the sign is negative in 74% of cells, so this is the typical but not the universal case, and the covariance should be reported rather than signed a priori.

(In the granular model comparative advantage re-enters at the firm level through a third, quite different channel — selection on productivity: Proposition 9.)

### 4.2 One lemma, four comparative statics

**Lemma 1 (softmax derivative of a Herfindahl).** Let $w_r(t) = e^{v_r(t)}/\sum_{r''} e^{v_{r''}(t)}$ and $H(t) = \sum_r w_r(t)^2$. Then

$$
\dot w_r = w_r\bigl(\dot v_r - \mathbb E_w[\dot v]\bigr)
\qquad\text{and}\qquad
\dot H = 2\operatorname{Cov}_w\bigl(w, \dot v\bigr),
\tag{7}
$$

where $\mathbb E_w$ and $\operatorname{Cov}_w$ are taken under the probability $w$ over destinations, so that $\operatorname{Cov}_w(w,\dot v) = \sum_r w_r^2 \dot v_r - H\sum_r w_r \dot v_r$.

*Proof: Appendix A.6.*

Applying Lemma 1 to (6) with $v_r = \ln m_{rs} - \theta\alpha\ln d_{r'r}$ gives all the comparative statics of interest. Write $x_r \equiv -\theta \ln d_{r'r}$ for *proximity*.

**Proposition 6 (trade costs).** Holding $m$ fixed,

$$
\frac{\partial H_{r's}}{\partial \alpha} = 2 \operatorname{Cov}_w(w, x).
\tag{8}
$$

Allowing $\Phi$ to adjust (holding $T$ and $X$ fixed), the same expression holds with $x_r$ replaced by *relative* proximity

$$
\xi_r \equiv x_r - \sum_l \gamma_{lrs} x_{lr}
= -\theta\Bigl(\ln d_{r'r} - \sum_l \gamma_{lrs}\ln d_{lr}\Bigr),
\tag{9}
$$

i.e. how much closer $r'$ is to $r$ than $r$'s average incumbent supplier. In the *partial* regime, $\lim_{\alpha\to 0} H_{r's} = \sum_r \bar m_{rs}^2$ with $\bar m \equiv m/\sum m$, the same for every origin, and $\lim_{\alpha\to\infty} H_{r's} = 1$ whenever the nearest destination is unique. Both limits also hold in the total regime, but there the surviving destination is the $\operatorname{argmin}_r(\ln d_{r'r} - \ln d_{l^*(r)r})$ — *relative* distance, $l^*(r)$ being $r$'s cheapest origin — not the nearest destination, and $\bar m$ must be evaluated at $\alpha = 0$, where $\Phi_{rs}$ is buyer-invariant and $\bar m = \bar X_{\cdot s}$.

*Proof: Appendix A.7.*

Two readings. (i) The limits are unambiguous: with no trade cost every supplier holds the *market portfolio* and diversification is entirely a property of the demand geography; with prohibitive trade costs every supplier is a single-customer supplier. (ii) The local effect is *not* signed in general. $\operatorname{Cov}_w(w,x) > 0$ — trade costs concentrate — whenever weight and proximity are positively associated in the portfolio, which is the typical case and is guaranteed when $m_{rs}$ is constant across buyers, i.e. when $X_{rs} \propto \Phi_{rs}$ (Corollary 6.1 in the appendix; note that *uniform demand* is not enough, since $\Phi_{rs}$ varies across buyers with geography). But a supplier whose dominant customer is *far* (a small remote cell selling into a distant hub) is *diversified* by an increase in $\alpha$ before being concentrated by it. Reporting (8) alongside the level is therefore informative, and the sign is an empirical matter, not a theorem. Appendix A.7 gives an explicit two-destination counterexample.

**Proposition 7 (comparative advantage).** For any origin $k$,

$$
\frac{\partial \ln w_r^{sr'd}}{\partial \ln T_{ks}}
= \sum_{r''} w_{r''}^{sr'd}\gamma_{kr''s} - \gamma_{krs}
\tag{10}
$$
$$
\frac{\partial H_{r's}}{\partial \ln T_{ks}}
= -2 \operatorname{Cov}_w\bigl(w, \gamma_{k\cdot s}\bigr).
\tag{11}
$$

*Proof: Appendix A.8.*

Equation (10) *is* the paper's equation (13), obtained here as a corollary of Proposition 5 — which is a useful cross-check, since the two derivations are independent (the paper's goes through the indicator cancellation, ours through the $T$-independence of the composition). Equation (11) is new and is the answer to "what does comparative advantage do to diversification":

> Raising the comparative advantage of region $k$ **diversifies** every supplier's portfolio if and only if $k$ competes at the destinations where that supplier is already concentrated ($\operatorname{Cov}_w(w,\gamma_{k\cdot s}) > 0$), and **concentrates** it otherwise.

The *ranking* of destinations by the reallocation they receive is common to every origin $r'$ — destinations are ordered by $-\gamma_{krs}$, which is origin-free; this is the paper's "uniform within-portfolio reallocation". The *level* (through the portfolio average $\sum_{r''} w_{r''}^{sr'd}\gamma_{kr''s}$) and hence the *sign* of the effect on any one destination, and on $H$, are origin-specific, because they depend on where that origin's weight sits.

**Proposition 8 (downstream demand geography).** For any buyer $k$,

$$
\frac{\partial H_{r's}}{\partial \ln X_{ks}}
= 2\, w_k^{sr'd}\bigl(w_k^{sr'd} - H_{r's}\bigr).
\tag{12}
$$

A demand expansion at $k$ concentrates the portfolio iff $k$'s weight exceeds the sales-weighted average weight $H$. Summing (12) over $k$ gives zero, as homogeneity of degree zero requires.

*Proof: Appendix A.9.*

### 4.3 The extensive margin in closed form, and how comparative advantage re-enters

Two conventions must be separated before the extensive margin can be written down. In the paper's setup each (region, variety) pair hosts a continuum of firms whose productivities form a Poisson process of mean $T_{r's}z^{-\theta}$, so the region's *champion* is $\mathrm{Fr\acute echet}(T_{r's},\theta)$ (its equation `eq:app-champion`). The probability that an *arbitrary* firm of productivity $z$ inside that continuum is the cheapest supplier at $r$ must also beat its own region's rivals and is $\exp\bigl(-\Phi_{rs}(w_{r's}\tau_{r'rs})^{\theta}z^{-\theta}\bigr)$, which is the paper's `eq:app-rho`. The probability that the region's *champion* wins is instead

$$
\rho_{r'rs}(z) = \exp\bigl(-\kappa_{r'rs} z^{-\theta}\bigr),
$$
$$
\kappa_{r'rs} = \sum_{l\ne r'} T_{ls}\bigl(w_l\tau_{lrs}\bigr)^{-\theta}\cdot\bigl(w_{r'}\tau_{r'rs}\bigr)^{\theta}
= T_{r's}\,\frac{1-\gamma_{r'rs}}{\gamma_{r'rs}},
\tag{13}
$$

with the own term *excluded*. It is the champion object that the estimator simulates — one draw per (cell, variety), the winner taken as the Ricardian argmin across regions — and only the champion object integrates back to the sourcing share, $\mathbb E_z[\rho_{r'rs}(z)] = \gamma_{r'rs}$ (Appendix A.10). Everything below is the champion convention, and $\kappa_{r'rs}$ rather than $\Phi_{rs}(w\tau)^{\theta}$ is what appears.

**Proposition 9 (multi-homing).** Let $n_\rho$ be the number of downstream regions served by variety $\rho$ of cell $(r',s)$. Then

$$
\mathbb E[n_\rho] = \sum_r \gamma_{r'rs}
\tag{14}
$$
$$
\mathbb E[n_\rho \mid n_\rho \ge 1] = \frac{\sum_r \gamma_{r'rs}}{\hat q_{r's}},
\tag{15}
$$

where $\hat q_{r's} = \mathbb E_z[\tilde\rho_{r's}(z)]$ is the paper's $p_{r's}$. $\mathbb E[n_\rho]$ and $\hat q_{r's}$ are both strictly increasing in $T_{r's}$; Appendix A.10 gives the likelihood-ratio argument under which the conditional rate (15) inherits that monotonicity, and states what is *not* proved there. Moreover the win events $\{W_r\}$ are positively associated, conditionally on $z$ and unconditionally, so that

$$
\tilde\rho_{r's}(z) \le 1 - \prod_r \bigl(1-\rho_{r'rs}(z)\bigr)
\qquad\text{and}\qquad
\hat q_{r's} \le 1 - \prod_r \bigl(1-\gamma_{r'rs}\bigr).
\tag{16}
$$

*Proof: Appendix A.10.*

Three implications.

1. **The extensive margin of diversification is free.** $\sum_r \gamma_{r'rs}$ needs no simulation, and $\hat q_{r's}$ is already produced by the estimator as a diagnostic. The ratio (15) is the *multi-homing intensity* and, by Proposition 3, an upper bound on the mean effective number of customers of the cell's varieties.
2. **Comparative advantage does raise firm-level diversification** — through selection, not composition. By (13), $\kappa_{r'rs}$ involves the rivals' $T$ and not the origin's own, so conditionally on $z$ the win probability is free of $T_{r's}$; but $z$ is Fréchet with scale $T_{r's}$, so a higher $T$ first-order stochastically dominates and every $\gamma_{r'rs}$ rises. A strong cell hosts productive firms, and productive firms win more markets. This channel is distinct from the market-access channel of Corollary 5.1 and the two must not be conflated: they can point in opposite directions.
3. **The product formula is an upper bound, and it is not what the estimator uses.** (16) is the FKG/Harris inequality: positive dependence makes the union *smaller* than under independence. Two consequences for the paper. (a) The main text currently asserts the reverse inequality, $p_{r's} > 1 - \prod_r(1-\gamma_{r'rs})$, while giving positive dependence as the reason; the reason is right and the inequality is backwards. A Monte-Carlo check on a five-destination economy gives $\hat q = 0.416$ against a product formula of $0.737$. (b) The product form survives only on the *analytical/GMM* path (`compute_regression_quadrature`), which the repository already flags as FKG-biased; the $\hat q$ written to disk by the simulated estimator is the exact realised union, so a statistic built on it carries no FKG bias at all.


### 4.4 The floor: what the distribution of downstream sales alone implies

Sections 4.1–4.3 hold $X_{rs}$ fixed and vary the model's parameters. This section does the opposite: it fixes nothing and asks how much of a supplier's Herfindahl is already contained in the *distribution of downstream sales* — the vector $X_{\cdot s}$ of purchases of sector-$s$ inputs across buyer regions. The question matters because that vector is data, not model output (§9), and because the paper's verbal argument — motor vehicles has "many hubs", aerospace "two" — is a statement about it and about nothing else.

Write $\bar X_{rs} \equiv X_{rs}/\sum_{r''}X_{r''s}$ for the **market portfolio**, i.e. the portfolio a supplier would hold if it sold to every buyer in proportion to that buyer's size, and

$$
H^X_s \;\equiv\; \sum_r \bar X_{rs}^2, \qquad N^X_s \;\equiv\; 1/H^X_s ,
$$

for its Herfindahl and effective number of buyers. In the model each downstream region hosts one representative buyer, so the distribution of $X_{\cdot s}$ across regions *is* the size distribution of downstream customers, and $N^X_s$ is the effective number of customers the industry has to offer. This is the same number as the $\alpha = 0$ benchmark $N^0_s$ of Proposition 6, but it is about to be given a role that does not require the counterfactual: it binds at the *estimated* $\alpha$.

**Proposition 10 (access tilt).** Define the **access tilt** of origin $r'$ at buyer $r$ as its over- or under-weight relative to the market portfolio,

$$
u_r^{r'} \;\equiv\; \frac{w_r^{sr'd}}{\bar X_{rs}}
\;=\; \frac{\tau_{r'rs}^{-\theta}/\Phi_{rs}}{\sum_{r''}\bar X_{r''s}\,\tau_{r'r''s}^{-\theta}/\Phi_{r''s}},
\qquad \mathbb E_{\bar X}\bigl[u^{r'}\bigr] = 1 ,
\tag{17}
$$

where $\mathbb E_{\bar X}$, $\operatorname{Var}_{\bar X}$ and $\operatorname{Cov}_{\bar X}$ are taken under the probability $\bar X_{\cdot s}$ over buyer regions. Then, exactly,

$$
H_{r's}
\;=\; \underbrace{H^X_s}_{\text{demand geography}}\;\cdot\;
\underbrace{\bigl(1 + \operatorname{Var}_{\bar X}(u^{r'})\bigr)}_{\text{dispersion of access}}
\;+\;
\underbrace{\operatorname{Cov}_{\bar X}\bigl(\bar X_{\cdot s},\, (u^{r'})^2\bigr)}_{\text{alignment of access with size}} .
\tag{18}
$$

*Proof: Appendix A.11.*

Equation (18) is the exact decomposition of a supplier's concentration into what the market offers and what its own location does with the offer. The tilt (17) carries both structural forces at once — the trade cost $\tau_{r'rs}^{-\theta}$ and the competition it faces at each destination, $1/\Phi_{rs}$ — and it is the *only* channel through which they reach $H$: given $\bar X$, two origins with the same tilt vector have the same Herfindahl whatever their $T$, their wage, or their sales. Three readings.

- **Dispersion always concentrates.** The second factor is $1 + \operatorname{Var}_{\bar X}(u) \ge 1$ with equality iff $u \equiv 1$. Any access tilt whatsoever — even one uncorrelated with buyer size — raises the Herfindahl above $H^X_s$, because a mean-one multiplicative distortion of a portfolio can only spread its weights further apart. Trade costs never diversify a supplier *relative to the market portfolio* through this term; they can only do so through the second.
- **The sign lives entirely in the alignment term.** $\operatorname{Cov}_{\bar X}(\bar X, u^2) > 0$ says the supplier's access is best precisely at the large buyers — a Toulouse-based aerospace supplier — and its portfolio is then more concentrated than the market's on both counts. $\operatorname{Cov}_{\bar X}(\bar X, u^2) < 0$ says the supplier is well placed at the *small* buyers, and its tilt then pulls weight away from the hubs: such a supplier can be strictly **more diversified than the market portfolio**. This is not a curiosity: in 200 random economies (7 origins, 6 destinations, $\alpha = 0.35$) 36% of cells had $H_{r's} < H^X_s$, with $N_{r's}/N^X_s$ reaching 1.73. A cap stated supplier by supplier would therefore be false — see Proposition 11 for the statement that is true.
- **It nests Proposition 6.** At $\alpha = 0$, $\tau \equiv 1$ makes $\Phi_{rs}$ buyer-invariant, so $u \equiv 1$, both correction terms vanish and $H_{r's} = H^X_s$ for every origin. Proposition 6's $\alpha \to 0$ limit is the special case of (18) in which the tilt is switched off.

The individual bound fails, but the *aggregate* one holds without any condition, because the market portfolio is not an arbitrary benchmark: it is the sales-weighted average of the suppliers' own portfolios.

**Proposition 11 (the market portfolio is the sales-weighted average, and hence a floor).** Let $\sigma_{r'} \equiv \bigl(\sum_r \gamma_{r'rs}X_{rs}\bigr)/\sum_l\bigl(\sum_r \gamma_{lrs}X_{rs}\bigr)$ be cell $(r',s)$'s share of the sector's sales to the industry. Then

$$
\sum_{r'} \sigma_{r'}\, w_r^{sr'd} \;=\; \bar X_{rs} \quad \text{for every } r,
\tag{19}
$$

and consequently

$$
\sum_{r'} \sigma_{r'} H_{r's} \;-\; H^X_s
\;=\; \sum_{r'}\sigma_{r'}\sum_r \bigl(w_r^{sr'd} - \bar X_{rs}\bigr)^2 \;\ge\; 0 .
\tag{20}
$$

The same identity holds one level down, with $\sigma_i$ over firms and the realised portfolios $a_{ir}$ of (2), so that

$$
\sum_i \sigma_i H_i \;\ge\; \sum_{r'}\sigma_{r'} \hat H_{r's} \;\ge\; H^X_s ,
\qquad\text{equivalently}\qquad
\Bigl(\sum_i \sigma_i H_i\Bigr)^{-1} \;\le\; \hat N_s^{\,\text{cell}} \;\le\; N^X_s .
\tag{21}
$$

*Proof: Appendix A.12.*

This is Proposition 4's algebra applied one level up, and it is the link the paper's argument needs. Three consequences.

1. **The distribution of downstream sales is a hard cap on average diversification.** The sales-weighted effective number of customers of a sector's suppliers can never exceed the effective number of downstream buyers, at *any* $\alpha$, $T$ or geometry. If aerospace's downstream purchases are effectively spread over two regions, then no configuration of trade costs or comparative advantage can make its suppliers diversified on average — the concentration is in the demand, and the model can only add to it. That is the precise version of the paper's "two hubs versus many hubs" sentence, and (21) says it is a theorem about the *average*, not about each supplier: individual suppliers can and do beat the market portfolio (Proposition 10), and quoting the cap supplier by supplier would overstate it.
2. **The gap is the angle again.** By (20) the excess of average supplier concentration over demand concentration is exactly the sales-weighted dispersion of portfolios around the market portfolio. A sector in which every supplier holds nearly the market portfolio has a small gap *and*, by Proposition 2, nearly collinear portfolios — so this single number carries both the length and the angle content of §3.2. It is worth reporting for that reason alone.
3. **It splits the statistic into an observed part and an estimated part.** Writing $\bar H_s \equiv \sum_{r'}\sigma_{r'}H_{r's}$, the decomposition $\bar H_s = H^X_s + \text{gap}_s$ separates a term that needs only `X_rs.npy` from a term that is model output. A cross-industry contrast in $\bar H_s$ should always be reported as a contrast in these two components: if aerospace's higher concentration is entirely $H^X_s$, the model is not doing the work, the data are.

Two caveats on (19), both minor and both worth stating. First, $\sum_{r'}\gamma_{r'rs} = 1$ requires summing over *all* origins; in the estimated model the domestic share is a sector-level constant, so it cancels in the normalisation and (19) holds as written, but if the domestic share were buyer-specific, $\bar X_{rs}$ would have to be replaced by the distribution of *domestically sourced* purchases. Second, (19) is an accounting identity of the realised economy and therefore holds in the granular simulation exactly, not only in expectation — which is why (21) can chain the realised firm-level and cell-level statements.

---

## 5. The bridge: diversification *is* the untargeted moment

Appendix B.5 of the paper derives, for the reduced-form spatial-comovement regression,

$$
\frac{\delta}{\gamma} \;\simeq\; -\theta\,\mathbb E\!\left[\frac{d\ln\tau}{d\ln d}\right]\cdot
\frac{\mathbb E\bigl[\nu\, a\,(\Lambda + 1 - a)\bigr]}{\mathbb E[\nu a]\big|_{d=1}},
\tag{22}
$$

with $\nu \equiv \rho_{r'rs}(z)/\tilde\rho_{r's}(z)$ the selection correction induced by conditioning on $\mathrm{Sup}_i = 1$, and $a \equiv a^D_r(z,d)$. The $(1-a)$ term is described in the paper as the redirection of sales toward the rest of the portfolio. Averaged with the right weights it is a Herfindahl.

**Proposition 12 (the transmission factor).** Define the tilted measure $dP_\nu \propto \nu a\, dP$ and

$$
\tilde H \equiv \mathbb E_\nu[a] = \frac{\mathbb E[\nu a^2]}{\mathbb E[\nu a]},
\qquad
\bar\Lambda \equiv \mathbb E_\nu[\Lambda],
\qquad
\kappa \equiv \frac{\mathbb E[\nu a]}{\mathbb E[\nu a]\big|_{d=1}} .
$$

Then, with $\tau = d^{\alpha}$, (22) is exactly

$$
\frac{\delta}{\gamma} = -\theta\alpha\,\kappa\,\bigl(\bar\Lambda + 1 - \tilde H\bigr).
\tag{23}
$$

If, within a supplier, $\nu$ does not vary across destinations, then $\tilde H$ computed over that supplier's destinations equals its customer Herfindahl $H_i$ of Definition 1.

*Proof: Appendix A.13.*

Two comments on what (23) does and does not deliver. It is an exact restatement of the paper's own decomposition, so it inherits its approximations — in particular the continuous-distance derivation and the treatment of $\kappa$, which equals one only if the tilted mean share is distance-invariant. And $\tilde H$ is a *tilted* Herfindahl: $\nu$ is larger for near destinations, so $\tilde H$ overweights the nearby (large) shares and $\tilde H \ge H$ in the typical case. Both wedges are measurable in the simulated economy and should be reported rather than assumed away.

Subject to that, (23) turns the paper's verbal argument into an accounting identity. Define the **transmission factor**

$$
TF \;\equiv\; \frac{|\delta/\gamma|}{\theta\alpha} \;=\; \kappa\bigl(\bar\Lambda + 1 - \tilde H\bigr).
\tag{24}
$$

$\theta\alpha$ is the elasticity the *estimation targets* (the extensive margin); $\delta/\gamma$ is the elasticity it does *not* target. Their ratio is a diversification statistic.

**A scale warning before any number is read.** (22)–(24) are derived for the paper's $\delta/\gamma$, the ratio of two coefficients of a regression that is *linear in the level* of $a_{ir}$ on $\log d$, with the denominator normalised at $d = 1$. The model counterpart the paper reports, $\hat\eta$, comes instead from a PPML *constant-elasticity* fit $\mathbb E[a_{ir}\mid s,r',r] = \exp(\alpha_{rs} + \eta \log \mathrm{Dist})$. These are not the same number. Fitting a constant elasticity in the linear form puts the intercept at $\log d = 0$, so that

$$
\frac{\delta}{\gamma} \approx \frac{\eta}{1 + |\eta|\,\mathbb E[\log d]},
$$

with $\mathbb E[\log d]\approx 5.8$ in the estimation sample; the ratio is mechanically compressed, and $|\delta/\gamma|$ is bounded above by $1/\mathbb E[\log d] \approx 0.17$ whatever the true elasticity. Either $\hat\eta$ is converted before entering (24), or the model's $\delta/\gamma$ is re-estimated in the linear form. The table below does the conversion and reports both.

**A second warning about the implied column.** From (23), $\tilde H = 1 + \bar\Lambda - TF/\kappa$. Since $\Lambda = -\ln\rho \ge 0$, $\bar\Lambda > 0$ strictly, so setting $\bar\Lambda = 0$ and $\kappa = 1$ returns a *lower* bound on $\tilde H$ and hence an *upper* bound on the effective number of customers. The column is therefore a bound, not an estimate; and since $\tilde H \le 1$, admissibility at $\kappa = 1$ requires $\bar\Lambda \le TF$, which is itself informative about how far $\kappa$ must be from one.

|  | $\theta\alpha$ | $\delta/\gamma$ | $TF$ | $\tilde H$ lower bd. | $\tilde N = 1/\tilde H$ upper bd. |
|---|---|---|---|---|---|
| Motor vehicles — data (Table 3) | 0.27 | −0.109 | 0.404 | 0.596 | 1.68 |
| Aerospace — data (Table 3) | 0.43 | −0.098 | 0.228 | 0.772 | 1.30 |
| Motor vehicles — model, $\hat\eta = -0.166$ converted | 0.27 | −0.085 | 0.313 | 0.687 | 1.46 |
| Aerospace — model, $\hat\eta = -0.063$ converted | 0.43 | −0.046 | 0.107 | 0.893 | 1.12 |

*(Illustrative: $\theta\alpha$ is read off the current fits, $\hat\eta$ from the paper's untargeted-moment paragraph, and the conversion uses $\mathbb E[\log d] = 5.8$. All four must be recomputed from one and the same run, with $\bar\Lambda$ and $\kappa$ measured rather than set, before the table enters the paper. Note what the conversion costs: on the raw $\hat\eta$ the implied effective numbers would read 2.6 against 1.2; converted, 1.46 against 1.12. The **ranking** is robust across all four rows — motor-vehicle portfolios are the more diversified — the **magnitude** is not.)*

The exercise the paper should run is the **consistency check**: compute $\tilde H$, $\bar\Lambda$ and $\kappa$ directly on the simulated firm-level economy and verify that (23) reproduces the model's own $\delta/\gamma$, estimated in the *linear* form so that the two sides are commensurable. If it does, the diversification statistic is not a new object requiring its own validation — it is the moment the model already matches, read in a different unit. That is the strongest possible warrant for putting it in the paper.

---

## 6. The planner's variance return, in closed form

The paper's cluster-policy proposition rests on $\mathcal V'_{ks} \equiv \partial \mathcal V/\partial\ln T_{ks}$, described as sign-ambiguous — "negative when $k$ is a stabilizing hub, positive when $k$ is a concentrating hub" — but never computed. Under the independence assumption of this note it has a closed form, and the "stabilizing hub" condition becomes a covariance.

**Proposition 13 (variance return).** Let $\mathcal V = \sum_{r'}\omega_{r'}\operatorname{Var}(r')$ for planner weights $\omega$, with $\operatorname{Var}(r')$ as in Proposition 2, and let $\zeta_{r's} \equiv w_s^{r'} w_d^{sr'}$. In the general case, abstracting from cross-sector spillovers so that only sector $s$'s portfolio responds to $T_{ks}$,

$$
\frac{\partial \mathcal V}{\partial \ln T_{ks}}
= 2\sigma^2 \sum_{r'} \omega_{r'}\,\zeta_{r's} \sum_r M_{r'r}\, w_r^{sr'd}
\bigl(\bar\gamma_k^{(r')} - \gamma_{krs}\bigr),
\qquad
\bar\gamma_k^{(r')} \equiv \sum_{r''} w_{r''}^{sr'd}\gamma_{kr''s}.
$$

If in addition the upstream economy has a **single** sector, so that $M_{r'r} = \zeta_{r's} w_r^{sr'd}$, this collapses to

$$
\frac{\partial \mathcal V}{\partial \ln T_{ks}}
= -2\sigma^2 \sum_{r'} \omega_{r'}\,\zeta_{r's}^2\,
\operatorname{Cov}_{w^{r'}}\bigl(w^{r'}, \gamma_{k\cdot s}\bigr).
\tag{25}
$$

*Proof: Appendix A.14.*

The distinction matters: $M_{r'r} = \sum_{s'}\zeta_{r's'} w_r^{s'r'd}$ is *already* the no-cross-sector-spillover object, and (25) additionally requires region $r'$ to host no upstream sector other than $s$. In a three-sector numerical economy the general expression is exact while (25) is off by 68–95%, so it is the general expression that should be computed and (25) that should be read as intuition.

Region $k$ is a **stabilizing** target for cluster policy ($\mathcal V' < 0$) iff, averaged over origins with weights $\omega\zeta^2$, it competes at the destinations where existing suppliers are concentrated. This is precisely the "moderately central but currently under-supplied" statement of the paper's cluster proposition (`prop:cluster_policy`), now with a sign test computable for every commuting zone from $\hat\gamma$ and the estimated portfolios, and it enters the planner's first-order condition (`eq:foc`) with no new machinery.

One caveat on the object differentiated. The paper's constraint is the full quadratic form $\Omega\Sigma\Omega'$ (`eq:planner_problem`), which *includes* the cross-region covariances, whereas $\mathcal V = \sum_{r'}\omega_{r'}\operatorname{Var}(r')$ keeps only the diagonal. By Proposition 2 the off-diagonal terms are the portfolio *angles*, and §8 argues that they carry the aggregate contrast — so the diagonal aggregate is the wrong object if the planner cares about comovement rather than about the sum of local variances. Extending (25) is mechanical: replace $M_{r'r}M_{r'r}$ by $\sum_l \Omega_{r'l} M_{lr}$ in the proof.

---

## 7. Proposed measurement

### 7.1 Statistics

All of the following are computable from artefacts the estimation already writes; none requires re-simulating the model.

| # | Statistic | Definition | Level | Source |
|---|---|---|---|---|
| 1 | Effective number of customers | $N_i = 1/\sum_r a_{ir}^2$ | firm | `suppliers.parquet` |
| 2 | Customers served | $n_i$; and $\sum_r \hat\gamma_{r'rs}$, $\sum_r \hat\gamma/\hat q$ in closed form | firm | parquet; $\hat\gamma$, $\hat q$ |
| 3 | Intensive margin | $CV_i$ from (4) | firm | parquet |
| 4 | Cell portfolio | $N_{r's} = 1/\sum_r (w_r^{sr'd})^2$ | cell | `w_srd_r.npy`, or $\hat\gamma\times X$ |
| 5 | **Realised diversification** $\mathcal K$ | $N_{r's}/N^X_s$, $N^X_s \equiv 1/\sum_r \bar X_{rs}^2$ (= $N^0_s$, the $\alpha=0$ benchmark) | cell | `X_rs.npy` |
| 5a | **Effective number of buyers** | $N^X_s$ alone — *observed*, no estimation | sector | `X_rs.npy` |
| 5b | **Excess concentration** | $\bar H_s - H^X_s = \sum_{r'}\sigma_{r'}\|w^{r'}-\bar X\|^2$, equation (22) | sector | $\hat\gamma$, `X_rs.npy` |
| 5c | **Access tilt** | $\operatorname{Var}_{\bar X}(u^{r'})$ and $\operatorname{Cov}_{\bar X}(\bar X,(u^{r'})^2)$, the two terms of (18) | cell | $\hat\gamma$, `X_rs.npy` |
| 6 | Portfolio similarity | $\cos\angle(w^{r'}, w^{l})$, and its distance profile | cell pair | closed form |
| 7 | Aggregation gain | equation (5) | cell | parquet |
| 8 | Local volatility | $\zeta_{r'}^2 H^M_{r'}$ | region | $M$ |
| 9 | Variance return | the general expression of Prop. 13 (not (25) unless single-sector) | policy | closed form |
| 10 | Dual: buyer's supplier base | $1/\sum_{r'} \hat\gamma_{r'rs}^2$ | buyer | $\hat\gamma$ |

Statistic 5 deserves emphasis. Raw effective numbers are **not comparable across industries**: the two samples do not have the same number of downstream regions, nor the same size distribution of buyers, so a difference in $N$ conflates the portfolio choice with the menu. $\mathcal K = N_{r's}/N^X_s$ divides by the diversification the demand geography *offers* — the market portfolio of Proposition 10, which is also the $\alpha = 0$ benchmark of Proposition 6 — and is therefore the scale-free statistic for the cross-industry comparison. $N^X_s$ is read straight off the observed expenditure shares $\bar X_{\cdot s}$; evaluating the benchmark at the estimated $\alpha$ instead gives a different number (9% away in a numerical example) and forfeits the observability that makes it useful. Note that $\mathcal K > 1$ is possible cell by cell — Proposition 10 says exactly when, and Appendix A.7 gives a counterexample — so $\mathcal K$ is not a bounded index and should not be presented as one. What *is* bounded is its sales-weighted harmonic average: by Proposition 11, $\sum_{r'}\sigma_{r'}/\mathcal K_{r'} \ge 1$. Statistics 5a–5c decompose the level accordingly: 5a is the observed floor, 5b the model's contribution to concentration on top of it, and 5c the two channels of (18) through which a single cell departs from the market portfolio.

### 7.2 Two decompositions to report

**(a) Counterfactual decomposition.** Recompute $N$ under $\alpha = 0$, under $T$ equalised within sector, and under $X_{rs}$ uniform, using the closed form (6). This isolates the three forces and reuses the machinery already written for the comparative-advantage section of the reporting notebook. Because the three factors interact, report either the full $2^3$ grid or the Shapley average over the six orderings; a single sequential decomposition is order-dependent and should be labelled as such.

**(b) Firm vs cell.** Report $\sum_i \sigma_i N_i$ and $N_{r's}$ side by side; equation (5) attributes the gap to between-firm portfolio dispersion, which is the granular margin and is industry-specific through $\hat N_s$.

**(c) Model vs demand geography.** Report $\bar H_s = H^X_s + \text{gap}_s$ of equation (22): the first term is observed, the second is what the estimated network adds. Since (22) also equals the sales-weighted dispersion of portfolios around the market portfolio, it is the length-side reading of the collinearity claim of §8 and should be reported beside statistic 6 rather than instead of it.

### 7.3 Figures

1. The CDF of $N_i$ by industry (model), with the market-portfolio benchmark $N^X_s$ drawn as a vertical line and the sales-weighted mean $1/\sum_i\sigma_i H_i$ marked on the axis. One picture carries the whole argument, and Proposition 11 says exactly what to look at in it: the *weighted mean* must sit at or below the vertical line, while individual suppliers may sit above it.
2. The counterfactual decomposition of 7.2(a) as a bar panel, per industry.
3. A map of $N_{r's}$ (or of $\mathcal K$) over commuting zones. The two-hub versus many-hub contrast is a spatial statement and reads best as a map.
4. Optional: portfolio similarity (statistic 6) against distance, by industry — the *angle* counterpart of the *length* in figures 1–3, and the object that Proposition 2 says drives aggregate comovement.

---

## 8. Reading the two industries

The three forces are separately identified in the estimation and can be signed individually.

**Downstream concentration.** This is the *only* input to the comparison that is directly observed rather than estimated: $X_{rs}$ and the location of downstream plants are data. $N^X_s = 1/\sum_r \bar X_{rs}^2$ — the effective number of downstream buyers, equal to the $\alpha = 0$ benchmark $N^0_s$ of Proposition 6 — is a small number in aerospace (Toulouse and Île-de-France, with Marignane, Nantes and Bordeaux behind) and a substantially larger one in motor vehicles. By Proposition 11 it is a **cap on the sales-weighted average** effective number of customers, at the estimated $\alpha$ and not merely at $\alpha = 0$: no configuration of trade costs or comparative advantage can make a sector's suppliers diversified on average beyond the diversification its buyers offer. The cap is on the average and *not* supplier by supplier — Proposition 10 shows that a supplier well placed at the small buyers is tilted away from the hubs and can beat the market portfolio (36% of cells do, in random economies) — so the claim must be stated in the weighted form or it is false. We recommend reporting $N^X_s$ as a descriptive statistic in the data section, before any model output, together with the gap $\bar H_s - H^X_s$ of (22): the first carries a large part of the argument on its own, and the second is exactly the part the model adds.

**Trade costs.** $\theta\alpha$ is 0.43 in aerospace against 0.27 in motor vehicles. By Proposition 6, aerospace portfolios sit at a "colder" softmax temperature: at equal demand geography they would be more concentrated *provided* $\operatorname{Cov}_w(w,\xi) > 0$ there — which Proposition 6 shows is typical but not universal, and which must therefore be measured rather than assumed. If it holds, the two forces compound rather than offset in aerospace: a candidate mechanical account of why the industry ranking of $\delta/\gamma$ reverses the ranking of the extensive-margin gradient, the paper's puzzle. Note that (24) does not by itself *explain* that reversal — reading a small $(1-\tilde H)$ off a small $|\delta/\gamma|$ and then invoking it to account for the same $|\delta/\gamma|$ is circular. The identity becomes an explanation only once $\tilde H$ is computed independently on the simulated economy, which is exactly the consistency check proposed at the end of §5.

**Comparative advantage.** Here the three channels of §4 must be kept apart, because they do not point the same way.

- Through *market access*, own and rival $T$ act identically (Corollary 5.1): the composition responds to $\Phi$ alone. Own $T$ typically **diversifies** the cell, because it raises the origin's share proportionally more where that share is currently small, i.e. at the distant destinations. A concentrated $\hat T$ in aerospace therefore does not concentrate portfolios through this channel.
- Through *competition*, a strong Toulouse raises $\Phi$ at the destinations near it, lowers $m$ there, and pushes *other* suppliers out of the Toulouse market — diversifying them, by (11), whenever their weight sits on Toulouse. Whether those suppliers are large enough to matter for the industry aggregate is an empirical question the statistics of §7 answer directly.
- At the *firm* level, high $T$ around the hub produces productive firms that win many markets (Proposition 9, implication 2). The prediction is that aerospace's large suppliers are *more* multi-homing than its average one. Whether the gap between the sales-weighted and the unweighted $N$ is *larger* in aerospace than in motor vehicles additionally requires the within-sector dispersion of $\hat T$ to be larger there, which is measurable and is not established here.

**The aggregate.** Propositions 1 and 2 say the industry-level consequence is not read off $N$ alone. Two distinct hypotheses follow, and this note offers evidence for neither — they are what statistics 1 and 6 are for.

1. *Length.* Aerospace portfolios are shorter ($N$ smaller), for the demand-geography and trade-cost reasons above. This is the individual-volatility statement.
2. *Angle.* Aerospace portfolios are more nearly collinear — motor vehicles should show **larger** pairwise angles between suppliers attached to different hubs — because almost every aerospace portfolio points at the same two destinations. Since $\operatorname{Corr}$ is the cosine and is invariant to length, this is the aggregate-comovement statement, and it does not follow from (1).

The paper's existing sentence, that hub regions "act as aggregators of network-driven fluctuations" because "their activity loads on shocks affecting a broad set of downstream locations", is a statement about portfolio *breadth* — statistics 1, 4 and 8, the length. The collinearity claim is a different one and needs statistic 6. Reporting both is what keeps them apart.

---

## 9. Caveats

**Independence of downstream shocks.** Proposition 1's identification of variance with the Herfindahl requires $\Sigma_{dd} = \sigma^2 I$. Motor vehicles and aerospace are subject to large industry-wide shocks; with a common factor, every portfolio loads on it and all effective numbers collapse toward 1. The statistics should therefore be presented as applying to the *idiosyncratic regional* component of downstream demand — which is also the component the paper's counterfactual shocks (EV eco-score, military procurement) are designed to isolate. As a robustness exercise, report $a_i'\hat\Sigma a_i$ for an estimated $\hat\Sigma$ beside $\sigma^2 H_i$; Proposition 1's first line is stated for a general $\Sigma_{dd}$ precisely so that this is available.

**$a_{ir}$ is a model object.** The estimation observes the extensive margin at the industry level ($\mathrm{Sup}_i$), the count distribution $G_s(K)$, and the regional sourcing shares $\gamma_{ls}$ aggregated over buyers — not the bilateral supplier $\times$ buyer matrix. The firm-level portfolio is therefore *not* directly measurable in the data, and the diversification statistics are model output. This is exactly why §5 matters: the untargeted moment $\delta/\gamma$ is the observable that disciplines them, and (23) is the map. Statistics that *are* observable should be flagged as such and separated from the rest in the table: $N^X_s$ needs only `X_rs.npy`, whereas the buyer dual (statistic 10) needs the estimated $\hat\gamma$ and is model output. Propositions 10 and 11 are what make that separation useful rather than cosmetic — they express the model output as the observed floor $H^X_s$ plus a residual, so the reader can see how much of a cross-industry contrast is data and how much is estimate.

**Sales outside the industry.** $a_{ir}$ is a portfolio share *within* the modelled downstream industry. A real supplier also sells elsewhere, which is a diversification channel the model does not represent. The reported $N$ therefore *understate* true diversification, in the same direction as the bias the paper already notes on $\delta/\gamma$. If a firm's industry share $\varphi_i$ is available, $H_i^{\text{total}} = \varphi_i^2 H_i + \dots$ bounds the correction; absent that, the statistics are conditional on industry exposure and should be labelled so.

**Fixed-network first order.** Assumption 1 evaluates transmission at the estimated network. Over the horizon at which shocks propagate this is the paper's own convention, but the comparative statics of §4 are precisely about the network moving, so the two must not be applied simultaneously: (8), (11) and (12) describe how a *policy* or a *parameter* moves the portfolio, not how a *shock* does.

**Simulation noise.** Firm-level statistics computed on `suppliers.parquet` inherit the simulation design. $N_i$ is a ratio of sums of squares of simulated shares; its sampling variability across draw sets should be reported (the estimation already carries the machinery for re-simulating at independent seeds), especially for the distributional statement of Figure 1.

---

## Appendix A. Proofs

Throughout, $\sum_r$ runs over the $R_d$ downstream regions, portfolios are elements of the simplex, and $\mathbb E_w$, $\operatorname{Cov}_w$ denote moments taken under the probability $w$ on that index set: $\mathbb E_w[f] = \sum_r w_r f_r$, $\operatorname{Cov}_w(f,g) = \sum_r w_r f_r g_r - \mathbb E_w[f]\mathbb E_w[g]$.

### A.1 Proposition 1

Supplier $i$'s sales are $x_i = \sum_r X_{ir}$, where $X_{ir}$ is its sales to the downstream firm of region $r$. Under Assumption 1 a demand shock shifts that firm's scale, $d\ln X_{ir} = \delta^D_r$, with the cost shares held fixed. Hence $dx_i = \sum_r X_{ir}\delta^D_r$ and, dividing by $x_i$,

$$
d\ln x_i = \sum_r a_{ir}\delta^D_r = a_i'\delta^D,
\qquad a_{ir} = \frac{X_{ir}}{x_i},
$$

which is the paper's transmission equation. Therefore $\operatorname{Var}(d\ln x_i) = a_i'\Sigma_{dd}a_i$. With $\Sigma_{dd} = \sigma^2 I$ this is $\sigma^2\sum_r a_{ir}^2 = \sigma^2 H_i$. $\blacksquare$

### A.2 Proposition 2

Regional upstream activity growth is $d\ln y_{r'} = \sum_r M_{r'r}\delta^D_r$, so $\operatorname{Cov}(r',l) = M_{r'\cdot}\Sigma_{dd}M_{l\cdot}'$. With $\Sigma_{dd} = \sigma^2 I$, $\operatorname{Cov}(r',l) = \sigma^2\sum_r M_{r'r}M_{lr} = \sigma^2\zeta_{r'}\zeta_l\langle \tilde M_{r'\cdot}, \tilde M_{l\cdot}\rangle$ by definition of $\zeta$ and $\tilde M$. Setting $l = r'$ gives $\operatorname{Var}(r') = \sigma^2\zeta_{r'}^2\|\tilde M_{r'\cdot}\|^2$, and $H^M_{r'} = \|\tilde M_{r'\cdot}\|^2$ because $\tilde M_{r'\cdot}$ sums to one. Finally

$$
\operatorname{Corr}(r',l)
= \frac{\operatorname{Cov}(r',l)}{\sqrt{\operatorname{Var}(r')\operatorname{Var}(l)}}
= \frac{\sigma^2\zeta_{r'}\zeta_l\langle \tilde M_{r'\cdot},\tilde M_{l\cdot}\rangle}
{\sigma^2\zeta_{r'}\zeta_l\|\tilde M_{r'\cdot}\|\|\tilde M_{l\cdot}\|},
$$

the $\zeta$'s cancelling, which is the cosine of the angle between the two portfolios. $\blacksquare$

### A.3 Proposition 3

Let $S_i = \{r : a_{ir} > 0\}$, $|S_i| = n_i$. Since $\sum_{r\in S_i} a_{ir} = 1$, the mean of the positive shares is $\bar a = 1/n_i$. Write $V$ for their variance, so $CV_i^2 = V/\bar a^2 = n_i^2 V$. Then

$$
H_i = \sum_{r\in S_i} a_{ir}^2 = n_i\bigl(V + \bar a^2\bigr) = n_i\Bigl(V + \frac{1}{n_i^2}\Bigr)
= n_i V + \frac{1}{n_i} = \frac{1}{n_i}\bigl(n_i^2 V + 1\bigr) = \frac{1 + CV_i^2}{n_i}.
$$

$N_i = n_i/(1+CV_i^2) \le n_i$, with equality iff $V = 0$. $\blacksquare$

### A.4 Proposition 4

Fix $r$. With $\sum_i \sigma_i = 1$ and $w_r = \sum_i \sigma_i a_{ir}$ from (2),

$$
\sum_i \sigma_i (a_{ir}-w_r)^2
= \sum_i \sigma_i a_{ir}^2 - 2w_r\sum_i \sigma_i a_{ir} + w_r^2
= \sum_i \sigma_i a_{ir}^2 - w_r^2 .
$$

Summing over $r$ gives $\sum_i \sigma_i H_i - H_{r's} = \sum_i \sigma_i\sum_r (a_{ir}-w_r)^2 \ge 0$. $\blacksquare$

*(This is the conditional-variance decomposition; equivalently, Jensen's inequality applied to the convex map $a\mapsto a^2$.)*

### A.5 Proposition 5

Substituting $\gamma_{r'rs} = T_{r's}(w_{r'}\tau_{r'rs})^{-\theta}/\Phi_{rs}$ into (1),

$$
\gamma_{r'rs}X_{rs}
= \bigl[T_{r's}w_{r'}^{-\theta}\bigr]\cdot \tau_{r'rs}^{-\theta}\cdot \frac{X_{rs}}{\Phi_{rs}}
= \bigl[T_{r's}w_{r'}^{-\theta}\bigr]\cdot \tau_{r'rs}^{-\theta} m_{rs}.
$$

The bracketed factor does not depend on the destination $r$, so it is common to the numerator and to every term of the denominator of (1) and cancels:

$$
w_r^{sr'd} = \frac{m_{rs}\tau_{r'rs}^{-\theta}}{\sum_{r''} m_{r''s}\tau_{r'r''s}^{-\theta}} .
$$

Under $\tau_{r'r} = d_{r'r}^{\alpha}$, $\tau_{r'rs}^{-\theta} = \exp(-\theta\alpha\ln d_{r'r})$, giving the softmax form (6). $\blacksquare$

*Remark 1.* The cancellation needs only that the origin-specific components of $\gamma$ factor out of the destination index; it does not require $\tau$ to be sector-neutral, and it holds for any origin-level shifter (a productivity subsidy, a wage change).

*Remark 2 (what does **not** follow).* It does not follow that $w_r^{sr'd}$ is free of $T_{r's}$ or of $w_{r'}$. Both appear inside $\Phi_{rs}$, hence inside $m_{rs}$, and they do so with a destination-*varying* weight, since $\partial\ln\Phi_{rs}/\partial\ln T_{r's} = \gamma_{r'rs}$ depends on $r$. What (6) establishes is that only this market-access channel survives: the direct, destination-invariant part cancels. Numerically, in a five-origin $\times$ four-destination economy ($\theta = 1.768$, $\alpha = 0.35$, wages not all equal) the own-$T$ composition derivative is $(-0.271, 0.070, 0.175, 0.152)$ — plainly not zero — and it reproduces (10) at $k = r'$ to $1.8\cdot 10^{-10}$.

### A.6 Lemma 1

Write $Z(t) = \sum_{r''} e^{v_{r''}(t)}$, so $w_r = e^{v_r}/Z$ and $d\ln Z/dt = \sum_{r''}(e^{v_{r''}}/Z)\dot v_{r''} = \mathbb E_w[\dot v]$. Then

$$
\dot w_r = w_r\Bigl(\dot v_r - \frac{d\ln Z}{dt}\Bigr) = w_r\bigl(\dot v_r - \mathbb E_w[\dot v]\bigr),
$$

which is (7)(i) and implies $\sum_r \dot w_r = 0$. For (7)(ii),

$$
\dot H = 2\sum_r w_r \dot w_r = 2\sum_r w_r^2\bigl(\dot v_r - \mathbb E_w[\dot v]\bigr)
= 2\Bigl(\sum_r w_r^2 \dot v_r - H\,\mathbb E_w[\dot v]\Bigr).
$$

Taking $f_r = w_r$ and $g_r = \dot v_r$ in the definition of $\operatorname{Cov}_w$,

$$
\operatorname{Cov}_w(w,\dot v)
= \sum_r w_r\cdot w_r\cdot \dot v_r - \Bigl(\sum_r w_r\cdot w_r\Bigr)\Bigl(\sum_r w_r \dot v_r\Bigr)
= \sum_r w_r^2 \dot v_r - H\,\mathbb E_w[\dot v],
$$

since $\mathbb E_w[w] = \sum_r w_r^2 = H$. Hence $\dot H = 2\operatorname{Cov}_w(w,\dot v)$. $\blacksquare$

### A.7 Proposition 6

*Partial effect.* In (6), $v_r = \ln m_{rs} + \alpha x_r$ with $x_r = -\theta\ln d_{r'r}$. Holding $m$ fixed, $\dot v_r = x_r$, and Lemma 1 gives $\partial H/\partial\alpha = 2\operatorname{Cov}_w(w,x)$.

*Total effect.* $\Phi_{rs} = \sum_l T_{ls}(w_l d_{lr}^{\alpha})^{-\theta}$, so

$$
\frac{\partial \Phi_{rs}}{\partial \alpha}
= \sum_l T_{ls}w_l^{-\theta}d_{lr}^{-\theta\alpha}\bigl(-\theta\ln d_{lr}\bigr)
= \Phi_{rs}\sum_l \gamma_{lrs}x_{lr},
$$

i.e. $\partial\ln\Phi_{rs}/\partial\alpha = \sum_l \gamma_{lrs}x_{lr}$. Since $\ln m_{rs} = \ln X_{rs} - \ln\Phi_{rs}$ and $X$ is held fixed, $\dot v_r = x_{r'r} - \sum_l \gamma_{lrs}x_{lr} = \xi_r$, which is (9); Lemma 1 then gives $dH/d\alpha = 2\operatorname{Cov}_w(w,\xi)$.

*Limits, partial regime ($m$ fixed).* As $\alpha\to 0$, $w_r \to m_{rs}/\sum_{r''}m_{r''s} = \bar m_{rs}$, independent of $r'$, so $H\to\sum_r \bar m_{rs}^2$. As $\alpha\to\infty$, let $r^* = \operatorname{argmin}_r d_{r'r}$ be unique; for $r\ne r^*$,
$$
\frac{w_r}{w_{r^*}} = \frac{m_{rs}}{m_{r^*s}}\exp\bigl(-\theta\alpha(\ln d_{r'r} - \ln d_{r'r^*})\bigr)\to 0,
$$
so $w\to e_{r^*}$ and $H\to 1$.

*Limits, total regime.* The limiting values are unchanged but the arguments are not. At $\alpha\to 0$, $\tau\equiv 1$ makes $\Phi_{rs}$ buyer-invariant, so $\bar m$ collapses to the expenditure shares $\bar X_{\cdot s}$ — which is what makes the $\alpha = 0$ benchmark observable (§7.1). At $\alpha\to\infty$, $\Phi_{rs}\sim T_{l^*(r)s}w_{l^*(r)}^{-\theta}d_{l^*(r)r}^{-\theta\alpha}$ where $l^*(r)$ is $r$'s cheapest origin, so
$$
w_r \propto \frac{X_{rs}}{T_{l^*(r)s}}\Bigl(\frac{d_{r'r}}{d_{l^*(r)r}}\Bigr)^{-\theta\alpha}
$$
and the surviving destination is the $\operatorname{argmin}_r(\ln d_{r'r} - \ln d_{l^*(r)r})$: **relative**, not absolute, distance. A two-destination example makes the difference bite: with $r'$ at distance 100 from destination 1 and 300 from destination 2, but destination 1 glued to a rival origin at distance 10, the nearest destination is 1 while at $\alpha = 3$ the portfolio is $(5\cdot 10^{-6},\, 1 - 5\cdot 10^{-6})$ — entirely on destination 2. $H\to 1$ either way, but the *rate* is governed by the gap in relative log-distance, which can be small even when the absolute distances are well separated: in a five-origin economy one origin is still at $H = 0.67$ at $\alpha = 20$, because its two best destinations differ by 0.014 in relative log-distance.

**Corollary 6.1 (sufficient condition for the sign).** If $m_{rs}$ is constant across $r$ — equivalently $X_{rs}\propto\Phi_{rs}$, *not* $X_{rs}$ uniform, since $\Phi_{rs}$ varies across buyers with geography — then $w_r = e^{\alpha x_r}/\sum e^{\alpha x_{r''}}$ is a strictly increasing function of $x_r$ (for $\alpha > 0$), so $w$ and $x$ are comonotone and $\operatorname{Cov}_w(w,x)\ge 0$ by Chebyshev's sum inequality, strictly unless all $x_r$ are equal. Trade costs then unambiguously concentrate the portfolio. Under uniform $X$ the sign can go either way: in a three-origin $\times$ two-destination example with $X = (1,1)$ and $\alpha = 0.6$, $\Phi = (0.3225, 0.0490)$ so $m = (3.10, 20.39)$ is far from constant and $\operatorname{Cov}_w(w,x) = -0.105 < 0$.

**Counterexample (the sign is not general).** Two destinations; set $\theta = 1$, $m = (0.99, 0.01)$, $d_{r'1} = 10$, $d_{r'2} = 1$, so $x = (-\ln 10, 0)$. At $\alpha = 0$, $w = (0.99, 0.01)$ and $H = 0.9802$. At $\alpha = \ln 99/\ln 10 \approx 1.9956$, $w\propto(0.99\cdot 10^{-1.9956}, 0.01) = (0.01, 0.01)$, so $w = (0.5,0.5)$ and $H = 0.5$. Hence $H$ *falls* over that range and $\partial H/\partial\alpha < 0$ somewhere in it; direct evaluation at $\alpha = 0$ gives $\operatorname{Cov}_w(w,x) = -0.02234$, so $\partial H/\partial\alpha = -0.04468 < 0$. As $\alpha\to\infty$, $H\to 1$: the map $\alpha\mapsto H$ is non-monotone, falling then rising. The economics: a supplier whose dominant customer is remote is *diversified* by higher trade costs, until they are high enough to strand it with its nearest customer. $\blacksquare$

### A.8 Proposition 7

In (6), $T_{ks}$ enters only through $\Phi_{rs}$, and

$$
\frac{\partial\ln\Phi_{rs}}{\partial\ln T_{ks}}
= \frac{T_{ks}(w_k\tau_{krs})^{-\theta}}{\Phi_{rs}} = \gamma_{krs},
$$

so $\dot v_r = \partial\ln m_{rs}/\partial\ln T_{ks} = -\gamma_{krs}$. Lemma 1(i) gives

$$
\frac{\partial\ln w_r^{sr'd}}{\partial\ln T_{ks}}
= -\gamma_{krs} + \sum_{r''} w_{r''}^{sr'd}\gamma_{kr''s},
$$

which is (10) — and coincides with the paper's `eq:w_elast_T`, including the fact that the answer does not depend on whether $r' = k$. The two derivations explain that independence differently and both are right: in the paper the direct term $\mathbf 1\{r'=k\}$ appears and cancels against the portfolio average, while here it never appears because Proposition 5 has already removed the destination-invariant part of own $T$ from the composition. What is *not* removed, in either derivation, is the market-access part — which is exactly why (10) applies verbatim at $k = r'$. Lemma 1(ii) gives $\partial H/\partial\ln T_{ks} = 2\operatorname{Cov}_w(w,-\gamma_{k\cdot s}) = -2\operatorname{Cov}_w(w,\gamma_{k\cdot s})$. $\blacksquare$

*Remark.* Both statements hold $X_{rs}$ fixed, as the paper does. Allowing $X_{rs}$ to respond adds the terms in the paper's general chain rule.

### A.9 Proposition 8

$X_{ks}$ enters $v_r = \ln X_{rs} - \ln\Phi_{rs} + \alpha x_r$ only at $r = k$, so $\dot v_r = \mathbf 1\{r=k\}$ and Lemma 1(ii) gives

$$
\frac{\partial H}{\partial\ln X_{ks}}
= 2\operatorname{Cov}_w(w,\mathbf 1_k)
= 2\Bigl(\sum_r w_r^2\mathbf 1\{r=k\} - H\sum_r w_r\mathbf 1\{r=k\}\Bigr)
= 2 w_k (w_k - H).
$$

Summing over $k$: $2\bigl(\sum_k w_k^2 - H\sum_k w_k\bigr) = 2(H - H) = 0$, consistent with $w$ being homogeneous of degree zero in $X_{\cdot s}$. $\blacksquare$

### A.10 Proposition 9

*(i) The mean.* Varieties are matched across origins: variety $\rho$ of sector $s$ is produced in every cell, with independent draws $z_{l\rho}\sim\mathrm{Fr\acute echet}(\theta, T_{ls})$, and buyer $r$ awards variety $\rho$ to $\operatorname{argmin}_l w_l\tau_{lrs}/z_{l\rho}$. For independent Fréchet draws the standard Eaton–Kortum computation gives

$$
\mathbb P\bigl(r'\text{ wins variety }\rho\text{ at }r\bigr)
= \frac{T_{r's}(w_{r'}\tau_{r'rs})^{-\theta}}{\Phi_{rs}} = \gamma_{r'rs}.
$$

Since $n_\rho = \sum_r \mathbf 1\{r'\text{ wins }\rho\text{ at }r\}$, linearity of expectation — which requires no independence across destinations — gives $\mathbb E[n_\rho] = \sum_r \gamma_{r'rs}$. Writing $\gamma_{r'rs} = A/(A+B_r)$ with $A = T_{r's}(w_{r'}\tau_{r'rs})^{-\theta}$ strictly increasing in $T_{r's}$ and $B_r = \sum_{l\ne r'}T_{ls}(w_l\tau_{lrs})^{-\theta}$ free of it, each $\gamma$ and hence $\mathbb E[n_\rho]$ is strictly increasing in $T_{r's}$. Equation (15) follows from $\mathbb E[n\mid n\ge 1] = \mathbb E[n]/\mathbb P(n\ge 1)$ and $\mathbb P(n\ge 1) = \hat q_{r's}$. $\hat q$ is increasing in $T_{r's}$ for the same reason: writing $u \equiv z^{-\theta}$, which is exponential with rate $T_{r's}$ (see the Remark), $\hat q = \mathbb E_u[\tilde\rho(u)]$ with $\tilde\rho$ decreasing in $u$ and free of $T_{r's}$, and the exponential family is stochastically decreasing in its rate.

*(ii) Positive association and the FKG bound.* Condition on the origin's own draw $z\equiv z_{r'\rho}$. The event $W_r = \{r'\text{ wins at }r\}$ is $\{z_{l\rho} < z\,Q_{lr}\ \forall l\ne r'\}$ for constants $Q_{lr} = (w_l\tau_{lrs})/(w_{r'}\tau_{r'rs})$, i.e. a lower orthant in the vector of competitor draws $(z_{l\rho})_{l\ne r'}$, whose components are independent. Independent random variables are associated (Esary–Proschan–Walkup; Harris/FKG), and association is preserved by monotone functions, so the increasing events $W_r^c$ satisfy

$$
\mathbb P\Bigl(\bigcap_r W_r^c \,\Big|\, z\Bigr) \;\ge\; \prod_r \mathbb P\bigl(W_r^c\mid z\bigr)
= \prod_r \bigl(1-\rho_{r'rs}(z)\bigr).
$$

Taking complements, $\tilde\rho_{r's}(z) = 1 - \mathbb P(\bigcap_r W_r^c\mid z) \le 1 - \prod_r(1-\rho_{r'rs}(z))$, the first half of (16). The unconditional half follows by mixing over $z$: each $\mathbb P(W_r^c\mid z)$ is increasing in $z^{-\theta}$, hence a monotone function of the single random variable $z$, so those functions are associated and

$$
\hat q^{\,c} \equiv \mathbb P\Bigl(\bigcap_r W_r^c\Bigr)
= \mathbb E_z\Bigl[\mathbb P\bigl(\textstyle\bigcap_r W_r^c\mid z\bigr)\Bigr]
\ge \mathbb E_z\Bigl[\prod_r \mathbb P(W_r^c\mid z)\Bigr]
\ge \prod_r \mathbb E_z\bigl[\mathbb P(W_r^c\mid z)\bigr]
= \prod_r (1-\gamma_{r'rs}),
$$

the first inequality by the conditional result and the second by association in $z$. Consequently $\hat q \le \hat q^{\mathrm{FKG}}$ and, $\mathbb E[n]$ being unaffected, $\mathbb E[n\mid n\ge 1] = \mathbb E[n]/\hat q \ge \mathbb E[n]/\hat q^{\mathrm{FKG}}$. $\blacksquare$

*Remark (monotonicity of (15)).* Substituting $u\equiv z^{-\theta}$, which is exponential with rate $T_{r's}$, one has $\mathbb E[n] = \mathbb E_u[N(u)]$ with $N(u) = \sum_r e^{-\kappa_{r'rs}u}$, $\kappa_{r'rs} = \sum_{l\ne r'}T_{ls}(w_l\tau_{lrs})^{-\theta}(w_{r'}\tau_{r'rs})^{\theta}$, and $\hat q = \mathbb E_u[\tilde\rho(u)]$; neither $N$ nor $\tilde\rho$ depends on $T_{r's}$, which enters only the distribution of $u$. (One checks $\mathbb E_u[e^{-\kappa u}] = T/(T+\kappa) = \gamma$, recovering (14).) The exponential family is ordered by likelihood ratio, $f_{T_2}/f_{T_1}$ decreasing in $u$ for $T_2 > T_1$, and this ordering survives tilting by $\tilde\rho(u)$; hence $\mathbb E[n\mid n\ge 1] = \mathbb E_{\tilde P_T}[N/\tilde\rho]$ is increasing in $T_{r's}$ whenever the conditional multi-homing rate $N(u)/\tilde\rho(u)$ is non-increasing in $u$. That rate equals $R_d$ at $u = 0$ and tends to 1 as $u\to\infty$ (the latter provided $\operatorname{argmin}_r\kappa_{r'rs}$ is unique; with $t$ tied destinations the limit is $t$); monotonicity in between is intuitive but we do not prove it, and it should be checked numerically on the estimated economy before (15) is presented as a monotone comparative static.

### A.11 Proposition 10

Fix the origin $r'$ and the sector $s$ and drop both indices. By Proposition 5, $w_r \propto m_r\tau_r^{-\theta} = \bar X_r\cdot(\tau_r^{-\theta}/\Phi_r)$ up to a constant, so $u_r \equiv w_r/\bar X_r$ is proportional to $\tau_r^{-\theta}/\Phi_r$; the normalisation in (17) is the one that makes $\sum_r \bar X_r u_r = \sum_r w_r = 1$, i.e. $\mathbb E_{\bar X}[u]=1$. Note that $u$ is defined without reference to the model — it is the portfolio's over-weight relative to the market portfolio — and that (17) merely says what the model makes it equal to.

Now treat the destination index as a random variable drawn under the probability $\bar X$, so that $\bar X_r$ itself and $u_r$ are two random variables on that space. Then

$$
H = \sum_r w_r^2 = \sum_r \bar X_r\cdot\bigl(\bar X_r u_r^2\bigr) = \mathbb E_{\bar X}\bigl[\bar X u^2\bigr]
= \mathbb E_{\bar X}[\bar X]\,\mathbb E_{\bar X}[u^2] + \operatorname{Cov}_{\bar X}\bigl(\bar X, u^2\bigr).
$$

Since $\mathbb E_{\bar X}[\bar X] = \sum_r \bar X_r^2 = H^X$ and $\mathbb E_{\bar X}[u^2] = \operatorname{Var}_{\bar X}(u) + \mathbb E_{\bar X}[u]^2 = 1 + \operatorname{Var}_{\bar X}(u)$, this is (18). The factor $1+\operatorname{Var}_{\bar X}(u)\ge 1$ with equality iff $u$ is $\bar X$-a.s. constant, hence (by $\mathbb E_{\bar X}[u]=1$) iff $u\equiv 1$, i.e. iff $w = \bar X$. At $\alpha = 0$, $\tau\equiv 1$ and $\Phi_r = \sum_l T_l w_l^{-\theta}$ is buyer-invariant, so $u\equiv 1$ and $H = H^X$, which is Proposition 6's partial-regime limit. $\blacksquare$

### A.12 Proposition 11

Cell $(r',s)$'s sales to the downstream industry are $S_{r'} = \sum_r \gamma_{r'rs}X_{rs}$ and its portfolio is $w_r^{sr'd} = \gamma_{r'rs}X_{rs}/S_{r'}$, so $\sigma_{r'}w_r^{sr'd} = \gamma_{r'rs}X_{rs}/\sum_l S_l$. Summing over origins and using $\sum_{r'}\gamma_{r'rs} = 1$,

$$
\sum_{r'}\sigma_{r'}w_r^{sr'd} = \frac{X_{rs}}{\sum_l S_l} = \frac{X_{rs}}{\sum_l\sum_{r''}\gamma_{lr''s}X_{r''s}} = \frac{X_{rs}}{\sum_{r''}X_{r''s}} = \bar X_{rs},
$$

which is (19): the market portfolio is the sales-weighted average of the cells' portfolios, i.e. total sales to buyer $r$ are $r$'s purchases. Equation (20) is then the bias–variance identity for a weighted mean: for any weights $\sigma$ summing to one, vectors $w^{r'}$ and their weighted mean $\bar w$,

$$
\sum_{r'}\sigma_{r'}\|w^{r'}\|^2 - \|\bar w\|^2 = \sum_{r'}\sigma_{r'}\|w^{r'} - \bar w\|^2 \ge 0 ,
$$

which is A.4's computation with the cell in place of the firm; substituting $\bar w = \bar X_{\cdot s}$ from (19) gives (20). For (21), apply the same identity to the realised firm portfolios within each cell (Proposition 4) and then across cells, the outer weights being each cell's share of sector sales; the two applications compose because the firm weights $\sigma_i$ multiply out to the firms' shares of *sector* sales. The chain requires only that realised sales to buyer $r$ sum to $r$'s realised purchases, which holds by construction in the simulated economy, so (21) is exact there and not merely an expectation. $\blacksquare$

*Remark.* Nothing in the argument uses the Fréchet structure, the softmax form, or the fixed network: (19)–(21) are accounting. They therefore survive every extension of the model that preserves market clearing at the buyer level.

### A.13 Proposition 12

Write $\mathbb E$ for the expectation over $(z, r', d)$ in (22) and define the tilted probability $dP_\nu = (\nu a/\mathbb E[\nu a])\,dP$, legitimate because $\nu\ge 0$ and $a\ge 0$. For any $f$, $\mathbb E[\nu a f] = \mathbb E[\nu a]\cdot\mathbb E_\nu[f]$. Applying this to $f = \Lambda + 1 - a$,

$$
\mathbb E\bigl[\nu a(\Lambda + 1 - a)\bigr]
= \mathbb E[\nu a]\bigl(\mathbb E_\nu[\Lambda] + 1 - \mathbb E_\nu[a]\bigr)
= \mathbb E[\nu a]\bigl(\bar\Lambda + 1 - \tilde H\bigr).
$$

With $\tau = d^{\alpha}$ we have $d\ln\tau/d\ln d = \alpha$, so (22) becomes

$$
\frac{\delta}{\gamma}
= -\theta\alpha\cdot\frac{\mathbb E[\nu a]}{\mathbb E[\nu a]|_{d=1}}\cdot\bigl(\bar\Lambda + 1 - \tilde H\bigr)
= -\theta\alpha\,\kappa\bigl(\bar\Lambda + 1 - \tilde H\bigr),
$$

which is (23). For the last claim, fix a supplier $i$ and take the inner expectation over destinations only. If $\nu_{ir} = \bar\nu_i$ for all $r$, then

$$
\tilde H_i = \frac{\mathbb E[\nu a^2]}{\mathbb E[\nu a]}
= \frac{\bar\nu_i\sum_r a_{ir}^2}{\bar\nu_i\sum_r a_{ir}}
= \sum_r a_{ir}^2 = H_i,
$$

using $\sum_r a_{ir} = 1$. $\blacksquare$

*Remark 1 (which $a$ is being used).* The last step divides by $\sum_r a_{ir} = 1$, i.e. it reads $a$ as the *realised portfolio share*. The paper's displayed
$$
a_r^D(z,d) = \frac{\gamma_{r'rs}x_{rs}}{\gamma_{r'rs}x_{rs} + \sum_{r''\ne r}\gamma_{r'r''s}x_{r''s}\rho(z)}
$$
has an $r$-dependent denominator and does not sum to one over $r$ (it also carries an evident typo, $\rho_{r'rs}$ where $\rho_{r'r''s}$ is meant — taken literally, the paper's own step $d\ln a/d\ln d = (1-a)\cdot d\ln\gamma/d\ln d$ would fail, since $\rho_{r'rs}$ also depends on $d_{r'r}$). We read $a$ throughout as the realised share, which is both the object Definition 1 uses and the one the simulated panel measures; the whole bridge to $H_i$ rests on that reading, so it is stated rather than left implicit.

*Remark 2 (the sign of the tilt).* Writing the inner expectation under the probability $p_r = a_{ir}$, $\tilde H_i = H_i + \operatorname{Cov}_p(\nu,a)/\mathbb E_p[\nu]$, so $\tilde H \ge H$ **iff** $\nu$ and $a$ are positively associated across the supplier's destinations. Since $\nu_{ir} = \rho_{r'rs}(z)/\tilde\rho_{r's}(z)$ is decreasing in $d_{r'r}$ and shares are larger nearby, that is the expected configuration — but it is a condition, not a theorem, and random $(\nu, a)$ violates it. The gap, like $\kappa$, is measurable on the simulated economy.

### A.14 Proposition 13

Let $\zeta_{r's} \equiv w_s^{r'}w_d^{sr'}$ and, abstracting from cross-sector spillovers, $M_{r'r} = \sum_{s'}\zeta_{r's'}w_r^{s'r'd}$. With $\mathcal V = \sum_{r'}\omega_{r'}\operatorname{Var}(r')$ and Proposition 2, $\mathcal V = \sigma^2\sum_{r'}\omega_{r'}\sum_r M_{r'r}^2$, so

$$
\frac{\partial\mathcal V}{\partial\ln T_{ks}}
= 2\sigma^2\sum_{r'}\omega_{r'}\sum_r M_{r'r}\frac{\partial M_{r'r}}{\partial\ln T_{ks}} .
$$

Only sector $s$'s portfolio moves, and by (10) $\partial M_{r'r}/\partial\ln T_{ks} = \zeta_{r's}w_r^{sr'd}\bigl(\bar\gamma_k^{(r')}-\gamma_{krs}\bigr)$ with $\bar\gamma_k^{(r')} = \sum_{r''}w_{r''}^{sr'd}\gamma_{kr''s}$, which is the general expression in the text. Note that $M_{r'r} = \sum_{s'}\zeta_{r's'}w_r^{s'r'd}$ *is* the no-cross-sector-spillover object; collapsing it to a single term additionally requires region $r'$ to host no upstream sector other than $s$. Under that stronger hypothesis, $M_{r'r} = \zeta_{r's}w_r^{sr'd}$ and

$$
\begin{aligned}
\frac{\partial\mathcal V}{\partial\ln T_{ks}}
&= 2\sigma^2\sum_{r'}\omega_{r'}\zeta_{r's}^2\sum_r w_r\,(w_r)\bigl(\bar\gamma_k^{(r')}-\gamma_{krs}\bigr)\\
&= 2\sigma^2\sum_{r'}\omega_{r'}\zeta_{r's}^2\Bigl(\bar\gamma_k^{(r')}H_{r's} - \sum_r w_r^2\gamma_{krs}\Bigr)\\
&= -2\sigma^2\sum_{r'}\omega_{r'}\zeta_{r's}^2\operatorname{Cov}_{w^{r'}}\bigl(w^{r'},\gamma_{k\cdot s}\bigr),
\end{aligned}
$$

using $\operatorname{Cov}_w(w,\gamma) = \sum_r w_r^2\gamma_{krs} - H\,\mathbb E_w[\gamma] = \sum_r w_r^2\gamma_{krs} - H\bar\gamma_k$. $\blacksquare$

*Remark.* This holds $X_{rs}$ and the exposures $\zeta$ fixed, exactly as the paper's `eq:w_elast_tau`–`eq:w_elast_T` do. Letting $\zeta_{r's}$ respond to $T_{ks}$ adds a level term that is first-order in the efficiency channel already priced in the planner's objective. For the paper's full quadratic constraint $\Omega\Sigma\Omega'$, replace $M_{r'r}\cdot M_{r'r}$ in the first display by $\sum_l \Omega_{r'l}M_{lr}$; the covariance form of (25) is then a cross-covariance and no longer collapses to a single $\operatorname{Cov}_w$.

---

## Appendix B. Correspondence with the code

| Object | Where it is | Note |
|---|---|---|
| $a_{ir}$ panel, zero-filled | `analysis_granular.ipynb`, `build_a_ir_panel` | one row per (supplier, downstream region); `a_ir` already computed |
| $w_r^{sr'd}$ | `<run>/<step>/w_srd_r.npy`, `[s, r', r]` | written by `write_post_hoc` in `main.jl` |
| $\gamma_{r'rs}$ (closed form) | `sourcing_geometry(data)["by_sector"][s]["rho"]` | `psi/tot`, cells $\times$ downstream |
| $\Phi_{rs}$ | the `tot` array inside `sourcing_geometry` | $m_{rs} = X_{rs}/\texttt{tot}$ is one line — but `tot` sums over that sector's *domestic active cells only*, so if foreign competition belongs in $\Phi$ the discrepancy is buyer-specific and does not cancel in the softmax |
| $X_{rs}$ | `X_rs.npy` / the downstream purchase column of `suppliers.parquet` | |
| $\bar X_{\cdot s}$, $H^X_s$, $N^X_s$ | one line from `X_rs.npy` | column-normalise and square; **no estimate enters**, so it can be reported in the data section |
| tilt $u^{r'}$, equation (17) | `w_srd_r.npy` (or $\hat\gamma\times X$) divided by $\bar X_{\cdot s}$ | the two terms of (18) are then a variance and a covariance under the weights $\bar X_{\cdot s}$ |
| cell sales weights $\sigma_{r'}$ | row sums of $\hat\gamma\times X$, normalised | needed for (20)–(21); use the *realised* sales if the realised portfolios are used, so that (19) holds exactly |
| $\hat q_{r's}$ | `granular_diagnostics.npz` (raw diagnostic, alongside the count moment) | needed for (15); it is the **exact** simulated union, not an FKG approximation |
| $\hat\alpha$, $\hat T$, $\theta$ | `unpack_estimated_T`, `model_theta` | `N_TAU = 1` required for a single $\alpha$ |
| $\alpha = 0$ / $T$ equalised counterfactuals | `sourcing_geometry(..., alpha=0, equalise_T=True)` | already implemented |
| distances $d_{r'r}$ | `distances.npy`, model's own $1..R$ indices | diagonal is the positive internal distance |

Two implementation warnings. First, `w_srd_r.npy` holds the *realised* portfolio $\hat w^{sr'd}$ of (2), built from the simulated firm-level economy, whereas $\hat\gamma\times X$ gives the continuum object $w^{sr'd}$ of (1). They are different vectors — Proposition 4 compares their *Herfindahls*, not the vectors themselves — so the two should be reported side by side rather than substituted for one another. Second, `share` in `suppliers.parquet` is a share of the *buyer's unit cost*, so $X_{ir}$ must be formed as `share` $\times$ `downstream_purchase` before any portfolio share is taken — which is what `build_a_ir_panel` does. A third, specific to §4.4: equation (19) is an adding-up identity, so it is also a **check on the artefacts** — if $\sum_{r'}\sigma_{r'}w_r^{sr'd}$ does not reproduce $\bar X_{rs}$ to numerical precision, the sales weights and the portfolios have been taken from inconsistent objects (typically continuum $\hat\gamma\times X$ weights against realised `w_srd_r.npy` portfolios), and the floor of Proposition 11 will appear to fail for a purely bookkeeping reason.

---

## Appendix C. Verification

Every displayed identity in this note was checked numerically before it was written down, and then re-checked independently. The first pass used finite differences on asymmetric random economies (5 upstream origins $\times$ 4 downstream regions, unequal wages, $\theta = 1.768$) for Lemma 1 and equations (3)–(5), (7)–(12), (17)–(21) and (25); Monte-Carlo simulation of the Ricardian assignment ($4\cdot 10^6$ draws) for (14)–(16); symbolic algebra for (23); and exact arithmetic for the counterexample of A.7. That pass is preserved as `test/test_diversification_identities.py`, which reproduces every identity below and fails loudly if one of them is edited into something false. A second, independent pass — different parameterisations, different seeds, `mpmath` at 250 digits for the inclusion–exclusion sum, and 60 random geometries for the monotonicity conjecture of A.10 — reproduced all of them and, in addition, produced the three corrections now incorporated: the own-$T$ channel of Corollary 5.1 (which an earlier draft wrongly declared absent), the champion-versus-arbitrary-firm convention of §4.3, and the single-sector hypothesis of Proposition 13. The $\hat\eta$-to-$\delta/\gamma$ conversion of §5 also comes from that pass.

Three claims in this note are **not** proved and are flagged where they appear: the monotonicity of (15) in $T_{r's}$ (A.10, Remark — verified in 60 random geometries, no violation); the sign of $\operatorname{Cov}_w(w,\xi)$ in any particular industry (§8); and every statement in §8 about which industry has the shorter or the more collinear portfolios, which are the hypotheses the statistics of §7 are meant to test.

---

## References

Acemoglu, D., V. M. Carvalho, A. Ozdaglar and A. Tahbaz-Salehi (2012), "The Network Origins of Aggregate Fluctuations," *Econometrica* 80(5), 1977–2016.

Barrot, J.-N. and J. Sauvagnat (2016), "Input Specificity and the Propagation of Idiosyncratic Shocks in Production Networks," *Quarterly Journal of Economics* 131(3), 1543–1592.

Carvalho, V. M., M. Nirei, Y. U. Saito and A. Tahbaz-Salehi (2021), "Supply Chain Disruptions: Evidence from the Great East Japan Earthquake," *Quarterly Journal of Economics* 136(2), 1255–1321.

di Giovanni, J., A. A. Levchenko and I. Méjean (2014), "Firms, Destinations, and Aggregate Fluctuations," *Econometrica* 82(4), 1303–1340.

Eaton, J. and S. Kortum (2002), "Technology, Geography, and Trade," *Econometrica* 70(5), 1741–1779.

Esary, J. D., F. Proschan and D. W. Walkup (1967), "Association of Random Variables, with Applications," *Annals of Mathematical Statistics* 38(5), 1466–1474.

Fortuin, C. M., P. W. Kasteleyn and J. Ginibre (1971), "Correlation Inequalities on Some Partially Ordered Sets," *Communications in Mathematical Physics* 22(2), 89–103.

Gabaix, X. (2011), "The Granular Origins of Aggregate Fluctuations," *Econometrica* 79(3), 733–772.

Hill, M. O. (1973), "Diversity and Evenness: A Unifying Notation and Its Consequences," *Ecology* 54(2), 427–432.

Kramarz, F., J. Martin and I. Méjean (2020), "Volatility in the Small and in the Large: The Lack of Diversification in International Trade," *Journal of International Economics* 122, 103276.