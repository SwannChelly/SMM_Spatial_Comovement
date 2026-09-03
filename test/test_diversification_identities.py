"""
Numerical gate for `documentation/diversification.md`.

Every displayed identity of that note is re-derived here by finite differences, Monte
Carlo, or exact arithmetic, on asymmetric random economies (unequal wages, unequal
region counts) so that a transposed index shows up.  Run it after any edit to the
note's propositions:

    python test/test_diversification_identities.py

No data and no Julia are needed; only numpy.  The economy is the paper's: origins r'
(upstream), destinations r (downstream), one sector, Frechet(theta, T_{r's}) champions
matched across origins, iceberg trade cost tau = d^alpha.
"""
import numpy as np

THETA = 1.768
FAILURES = []


def ok(name, cond):
    print(("PASS  " if cond else "FAIL  ") + name)
    if not cond:
        FAILURES.append(name)


# --------------------------------------------------------------------------- model
def gamma_phi(T, alpha, wage, d):
    """Sourcing shares gamma_{r'rs} and price access Phi_{rs}."""
    psi = T[:, None] * (wage[:, None] * d ** alpha) ** (-THETA)
    return psi / psi.sum(0, keepdims=True), psi.sum(0)


def portfolio(T, alpha, X, wage, d):
    """w_r^{sr'd} of equation (1), one row per origin."""
    g, _ = gamma_phi(T, alpha, wage, d)
    num = g * X[None, :]
    return num / num.sum(1, keepdims=True)


def H(w):
    return (w ** 2).sum(-1)


def draw(seed, n_up=5, n_down=4):
    r = np.random.default_rng(seed)
    return dict(T=r.lognormal(0, .7, n_up), X=r.lognormal(0, .9, n_down),
                wage=r.lognormal(0, .15, n_up), d=r.uniform(20, 600, (n_up, n_down)),
                alpha=0.35)


E = draw(2024)
T, X, wage, d, alpha = E["T"], E["X"], E["wage"], E["d"], E["alpha"]
w = portfolio(T, alpha, X, wage, d)
g, Phi = gamma_phi(T, alpha, wage, d)
x = -THETA * np.log(d)                      # proximity, section 4.2
h = 1e-6

print("=" * 72)
print("Section 3 -- the Herfindahl representation")
print("=" * 72)

# Proposition 2 -- Gram representation
M = np.random.default_rng(5).random((4, len(X))) * np.random.default_rng(6).random((4, 1))
sig2 = 2.3
C = M @ (np.eye(len(X)) * sig2) @ M.T
zeta = M.sum(1)
Mt = M / zeta[:, None]
ok("P2  Var = sigma^2 zeta^2 ||Mtilde||^2",
   np.allclose(np.diag(C), sig2 * zeta ** 2 * (Mt ** 2).sum(1)))
corr = C / np.sqrt(np.outer(np.diag(C), np.diag(C)))
cos = (Mt @ Mt.T) / np.sqrt(np.outer((Mt ** 2).sum(1), (Mt ** 2).sum(1)))
ok("P2  Corr = cosine of the portfolio angle", np.allclose(corr, cos))

# Proposition 3 -- extensive / intensive, with a genuinely unserved destination
a_i = np.array([.5, .3, .2, 0., 0.])
pos = a_i[a_i > 0]
n_i = pos.size
cv2 = pos.var() / pos.mean() ** 2           # population CV, as in A.3
ok("P3  H = (1 + CV^2) / n", np.isclose((a_i ** 2).sum(), (1 + cv2) / n_i))
ok("P3  N <= n", 1 / (a_i ** 2).sum() <= n_i)

# Proposition 4 -- aggregation gain (stated for the REALISED cell portfolio)
rr = np.random.default_rng(11)
A = rr.dirichlet(np.ones(len(X)), size=9)
s_i = rr.dirichlet(np.ones(9))
w_hat = (s_i[:, None] * A).sum(0)
lhs = (s_i * (A ** 2).sum(1)).sum() - (w_hat ** 2).sum()
rhs = (s_i[:, None] * (A - w_hat) ** 2).sum()
ok("P4  aggregation gap identity, and non-negative", np.isclose(lhs, rhs) and lhs >= 0)

print()
print("=" * 72)
print("Section 4 -- the portfolio and its comparative statics")
print("=" * 72)

# Proposition 5 -- the softmax form
m = X / Phi
sm = m[None, :] * d ** (-THETA * alpha)
sm = sm / sm.sum(1, keepdims=True)
ok("P5  w = softmax(ln m - theta*alpha*ln d), eq (6)", np.allclose(w, sm))
ok("P5  given m, every origin's composition differs only through d",
   np.allclose(sm, w) and not np.allclose(w[0], w[1]))

# Corollary 5.1 -- own T DOES move the composition, through Phi, and obeys eq (10).
# (An earlier draft claimed the opposite; this is the test that catches it.)
rp = 0
Tp, Tm = T.copy(), T.copy()
Tp[rp] *= np.exp(h)
Tm[rp] *= np.exp(-h)
fd_lw = (np.log(portfolio(Tp, alpha, X, wage, d))
         - np.log(portfolio(Tm, alpha, X, wage, d))) / (2 * h)
ok("C5.1 own T moves the composition (derivative is not zero)",
   np.abs(fd_lw[rp]).max() > 1e-2)
pred_own = (w * g[rp][None, :]).sum(1, keepdims=True) - g[rp][None, :]
ok("C5.1 own-T derivative obeys eq (10) at k = r'", np.allclose(fd_lw, pred_own, atol=1e-6))

# Corollary 5.1, second bullet: with flat X, own T always diversifies.
viol = 0
for t in range(300):
    r = np.random.default_rng(1000 + t)
    n_up, n_down = int(r.integers(3, 8)), int(r.integers(2, 7))
    Tt, wt = r.lognormal(0, 1, n_up), r.lognormal(0, .3, n_up)
    dt, at = r.uniform(10, 800, (n_up, n_down)), r.uniform(.05, 1.2)
    gt, _ = gamma_phi(Tt, at, wt, dt)
    Wt = portfolio(Tt, at, np.ones(n_down), wt, dt)
    for o in range(n_up):
        cov = (Wt[o] ** 2 * gt[o]).sum() - (Wt[o] ** 2).sum() * (Wt[o] * gt[o]).sum()
        viol += (-2 * cov) > 1e-14
ok("C5.1 flat X => dH/dlnT_own <= 0 in all 300 random economies", viol == 0)

# Lemma 1 / Proposition 6 -- trade costs, partial and total
Hpart = lambda al: H(((m[None, :] * d ** (-THETA * al))
                      / (m[None, :] * d ** (-THETA * al)).sum(1, keepdims=True)))
fd = (Hpart(alpha + h) - Hpart(alpha - h)) / (2 * h)
cov = (w ** 2 * x).sum(1) - H(w) * (w * x).sum(1)
ok("P6  partial dH/dalpha = 2 Cov_w(w, x), eq (8)", np.allclose(fd, 2 * cov, rtol=1e-5))

Htot = lambda al: H(portfolio(T, al, X, wage, d))
fd = (Htot(alpha + h) - Htot(alpha - h)) / (2 * h)
xi = x - (g * x).sum(0)[None, :]                       # relative proximity, eq (9)
cov = (w ** 2 * xi).sum(1) - H(w) * (w * xi).sum(1)
ok("P6  total dH/dalpha = 2 Cov_w(w, xi), eq (9)", np.allclose(fd, 2 * cov, rtol=1e-4))

# Proposition 6, limits, and the A.7 counterexample
def portfolio_log(T_, al, X_, wage_, d_):
    """Same as `portfolio`, in logs, so that very large alpha does not underflow."""
    lp = np.log(T_)[:, None] - THETA * (np.log(wage_)[:, None] + al * np.log(d_))
    ln = lp - np.logaddexp.reduce(lp, axis=0)[None, :] + np.log(X_)[None, :]
    return np.exp(ln - np.logaddexp.reduce(ln, axis=1)[:, None])


# The limit is governed by RELATIVE log-distance (A.7), so convergence is slow when two
# destinations are nearly tied on it -- here one origin is still at H = 0.67 at alpha = 20.
ok("P6  large alpha drives H to 1 in the total regime",
   bool((H(portfolio_log(T, 1000., X, wage, d)) > 0.999).all())
   and H(portfolio_log(T, 20., X, wage, d)).min() < 0.7)
mm, xx = np.array([.99, .01]), np.array([-np.log(10), 0.])
hh = lambda al: (lambda v: ((v / v.sum()) ** 2).sum())(mm * np.exp(al * xx))
w0 = mm / mm.sum()
cov0 = (w0 ** 2 * xx).sum() - hh(0.) * (w0 * xx).sum()
ok("A.7 counterexample: H(0)=0.9802, H(alpha*)=0.5, dH/dalpha<0 at 0, H(50)->1",
   np.isclose(hh(0.), 0.9802) and np.isclose(hh(np.log(99) / np.log(10)), .5)
   and np.isclose(2 * cov0, -0.044679, atol=1e-6) and hh(50.) > 0.999)

# Corollary 6.1 -- the sufficient condition is X proportional to Phi, not X uniform
w_prop = portfolio(T, alpha, Phi.copy(), wage, d)
cov_prop = (w_prop ** 2 * x).sum(1) - H(w_prop) * (w_prop * x).sum(1)
ok("C6.1 X proportional to Phi => Cov_w(w, x) >= 0", bool((cov_prop >= -1e-12).all()))

# Proposition 7 -- comparative advantage
k = 2
Tp, Tm = T.copy(), T.copy()
Tp[k] *= np.exp(h)
Tm[k] *= np.exp(-h)
wp, wm = portfolio(Tp, alpha, X, wage, d), portfolio(Tm, alpha, X, wage, d)
pred = (w * g[k][None, :]).sum(1, keepdims=True) - g[k][None, :]
ok("P7  eq (10) dln w / dln T_k", np.allclose((np.log(wp) - np.log(wm)) / (2 * h), pred, atol=1e-6))
covg = (w ** 2 * g[k][None, :]).sum(1) - H(w) * (w * g[k][None, :]).sum(1)
ok("P7  eq (11) dH / dln T_k = -2 Cov_w(w, gamma_k)",
   np.allclose((H(wp) - H(wm)) / (2 * h), -2 * covg, atol=1e-6))

# Proposition 8 -- downstream demand
kk = 3
Xp, Xm = X.copy(), X.copy()
Xp[kk] *= np.exp(h)
Xm[kk] *= np.exp(-h)
fd = (H(portfolio(T, alpha, Xp, wage, d)) - H(portfolio(T, alpha, Xm, wage, d))) / (2 * h)
ok("P8  eq (12) dH / dln X_k = 2 w_k (w_k - H)",
   np.allclose(fd, 2 * w[:, kk] * (w[:, kk] - H(w)), atol=1e-6))
ok("P8  the derivatives sum to zero over buyers",
   np.allclose(sum(2 * w[:, j] * (w[:, j] - H(w)) for j in range(len(X))), 0, atol=1e-12))

# Statistic 5 -- N^0 must be taken at alpha = 0, where mbar collapses to Xbar
_, Phi0 = gamma_phi(T, 0.0, wage, d)
m0 = X / Phi0
ok("S5  at alpha = 0 the market portfolio is the observed expenditure share",
   np.allclose(m0 / m0.sum(), X / X.sum()))

print()
print("=" * 72)
print("Section 4.4 -- the link to the distribution of downstream sales")
print("=" * 72)

Xbar = X / X.sum()
HX = (Xbar ** 2).sum()

# Proposition 10 -- the access tilt decomposition, eq (17)-(18)
u = w / Xbar[None, :]                                   # over-weight vs the market portfolio
ok("P10 the tilt has mean one under Xbar, eq (17)",
   np.allclose((Xbar[None, :] * u).sum(1), 1))
_, Phi_ = gamma_phi(T, alpha, wage, d)
u_struct = (d ** (-THETA * alpha)) / Phi_[None, :]
u_struct = u_struct / (Xbar[None, :] * u_struct).sum(1, keepdims=True)
ok("P10 tilt = normalised tau^-theta / Phi, eq (17)", np.allclose(u, u_struct))
Eu2 = (Xbar[None, :] * u ** 2).sum(1)
Var_u = Eu2 - 1
Cov_Xu2 = (Xbar[None, :] ** 2 * u ** 2).sum(1) - HX * Eu2
ok("P10 H = H^X (1 + Var(u)) + Cov(Xbar, u^2), eq (18)",
   np.allclose(H(w), HX * (1 + Var_u) + Cov_Xu2))
ok("P10 the dispersion factor is >= 1, with equality only at u == 1", (Eu2 >= 1 - 1e-12).all())
w0 = portfolio(T, 0.0, X, wage, d)                      # alpha = 0 nests Proposition 6
ok("P10 at alpha = 0 the tilt is one and H = H^X",
   np.allclose(w0, Xbar[None, :]) and np.allclose(H(w0), HX))

# Proposition 10, third bullet -- the individual bound genuinely FAILS
below, tot, worst = 0, 0, 0.0
for sd in range(200):
    Ez = draw(sd, 7, 6)
    wz = portfolio(Ez["T"], Ez["alpha"], Ez["X"], Ez["wage"], Ez["d"])
    hxz = ((Ez["X"] / Ez["X"].sum()) ** 2).sum()
    below += int((H(wz) < hxz).sum()); tot += wz.shape[0]
    worst = max(worst, (hxz / H(wz)).max())
print(f"      cells strictly more diversified than the market portfolio: "
      f"{below}/{tot} ({100 * below / tot:.0f}%), max N/N^X = {worst:.2f}")
ok("P10 a cell CAN beat the market portfolio (so the cap is not supplier-by-supplier)",
   below > 0 and worst > 1)

# Proposition 11 -- the market portfolio is the sales-weighted average, and a floor
S_cell = (g * X[None, :]).sum(1)
sig_cell = S_cell / S_cell.sum()
ok("P11 sales-weighted mean portfolio = observed expenditure shares, eq (19)",
   np.allclose((sig_cell[:, None] * w).sum(0), Xbar))
Hbar = (sig_cell * H(w)).sum()
gap = (sig_cell[:, None] * (w - Xbar[None, :]) ** 2).sum()
ok("P11 Hbar - H^X = weighted dispersion around the market portfolio, eq (20)",
   np.isclose(Hbar - HX, gap))
ok("P11 the floor binds on the average, at the ESTIMATED alpha", Hbar >= HX - 1e-12)
viol = 0
for sd in range(200):
    Ez = draw(sd, 7, 6)
    gz, _ = gamma_phi(Ez["T"], Ez["alpha"], Ez["wage"], Ez["d"])
    wz = portfolio(Ez["T"], Ez["alpha"], Ez["X"], Ez["wage"], Ez["d"])
    sz = (gz * Ez["X"][None, :]).sum(1); sz = sz / sz.sum()
    hxz = ((Ez["X"] / Ez["X"].sum()) ** 2).sum()
    viol += int((sz * H(wz)).sum() < hxz - 1e-12)
ok("P11 no violation of the floor in 200 random economies", viol == 0)

# Proposition 11, firm level -- the chain firm >= cell >= market, eq (21)
rr2 = np.random.default_rng(23)
A_f = rr2.dirichlet(np.ones(len(X)) * .6, size=12)      # firm portfolios
s_f = rr2.dirichlet(np.ones(12))                        # firm shares of SECTOR sales
w_cell = (s_f[:, None] * A_f).sum(0)                    # realised cell/market portfolio
ok("P11 firm >= cell, eq (21) first inequality",
   (s_f * H(A_f)).sum() >= H(w_cell) - 1e-12)
ok("P11 cell >= market when the cells average to the market portfolio",
   H(w_cell) >= ((w_cell / w_cell.sum()) ** 2).sum() - 1e-12)

print()
print("=" * 72)
print("Section 4.3 -- the extensive margin (Monte Carlo)")
print("=" * 72)

rng = np.random.default_rng(7)
n_draw = 400_000
U = rng.random((n_draw, len(T)))
Z = (-np.log(U) / T[None, :]) ** (-1 / THETA)          # Frechet champions
cost = (wage[:, None] * d ** alpha)[None, :, :] / Z[:, :, None]
win = cost == cost.min(1, keepdims=True)
ok("P9  marginal win probability equals gamma", np.allclose(win[:, 0, :].mean(0), g[0], rtol=4e-2))
ok("P9  eq (14) E[n] = sum_r gamma", np.allclose(win[:, 0, :].sum(1).mean(), g[0].sum(), rtol=3e-2))
bad = 0
for o in range(len(T)):
    q = (win[:, o, :].sum(1) > 0).mean()
    bad += q > 1 - np.prod(1 - g[o]) + 3e-3
ok("P9  eq (16) unconditional FKG bound holds for every origin", bad == 0)
kap = T[:, None] * (1 - g) / g                         # eq (13), own term excluded
ok("P9  eq (13) E_u[exp(-kappa u)] = gamma (champion convention)",
   np.allclose(T[:, None] / (T[:, None] + kap), g))

print()
print("=" * 72)
print("Sections 5 and 6")
print("=" * 72)

# Proposition 12 -- the tilting identity, and the table of section 5
rr = np.random.default_rng(3)
nu, av, Lam = rr.random(200) + .2, rr.random(200), rr.random(200)
den = (nu * av).mean()
ok("P12 eq (23) tilting identity",
   np.isclose((nu * av * (Lam + 1 - av)).mean() / den,
              (nu * av * Lam).mean() / den + 1 - (nu * av ** 2).mean() / den))
a_p = rr.dirichlet(np.ones(6))
nu_p = rr.random(6) + .2
ok("A.13 Htilde = H + Cov_p(nu, a) / E_p[nu]",
   np.isclose((nu_p * a_p ** 2).sum() / (nu_p * a_p).sum(),
              (a_p ** 2).sum()
              + ((a_p * nu_p * a_p).sum() - (a_p * nu_p).sum() * (a_p * a_p).sum())
              / (a_p * nu_p).sum()))

conv = lambda e, L=5.8: e / (1 + abs(e) * L)           # PPML eta -> linear delta/gamma
rows = [("motor vehicles, data", .27, -.109), ("aerospace, data", .43, -.098),
        ("motor vehicles, model", .27, conv(-.166)), ("aerospace, model", .43, conv(-.063))]
for lab, ta, dg in rows:
    tf = abs(dg) / ta
    print(f"      {lab:24s} delta/gamma={dg:+.4f}  TF={tf:.3f}  H={1-tf:.3f}  N={1/(1-tf):.2f}")
ok("S5  table reproduces (conversion at mean log d = 5.8)",
   np.isclose(conv(-.166), -0.08457, atol=1e-4) and np.isclose(conv(-.063), -0.04614, atol=1e-4))

# Proposition 13 -- the variance return.  Equation (25) needs a SINGLE upstream sector;
# the general expression is what holds with several.
om = np.random.default_rng(13).dirichlet(np.ones(len(T)))
zeta = np.random.default_rng(14).random(len(T)) * .4
V = lambda T_: (om * (( zeta[:, None] * portfolio(T_, alpha, X, wage, d)) ** 2).sum(1)).sum()
ok("P13 eq (25) exact when the upstream economy has one sector",
   np.allclose((V(Tp) - V(Tm)) / (2 * h), -2 * (om * zeta ** 2 * covg).sum(), rtol=1e-5))

print()
print("=" * 72)
if FAILURES:
    print(f"{len(FAILURES)} FAILED: " + "; ".join(FAILURES))
    raise SystemExit(1)
print("all identities of documentation/diversification.md reproduce")
