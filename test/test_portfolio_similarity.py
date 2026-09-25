"""Gate for Test 8 of the comparative-advantage section: the cosine similarity between
supplier customer portfolios, the groups cut out of it, and the within/between split.

The fixture plants TWO segmented blocks of suppliers -- one selling only to the first half
of the buyers, one only to the second -- plus a handful of cells that straddle them, so the
partition has a known answer and the rewiring null has something to fail against."""
import os, sys, numpy as np, pandas as pd, matplotlib, warnings
matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _nbmod import install

S, R, BUY = 2, 24, 10
CELLS = {0: np.arange(12), 1: np.arange(12, 24)}
buyers = np.arange(1, BUY + 1)
rng = np.random.default_rng(11)

# sector 0: two blocks (cells 0-4 sell to buyers 1-5, cells 5-9 to buyers 6-10), cells 10-11
# straddle. sector 1: one common clientele, every cell selling in the same proportions.
RHO = {}
p0 = np.zeros((12, BUY))
p0[:5, :5] = rng.uniform(.8, 1.2, (5, 5)); p0[:5, 5:] = 1e-4
p0[5:10, 5:] = rng.uniform(.8, 1.2, (5, 5)); p0[5:10, :5] = 1e-4
p0[10:] = rng.uniform(.8, 1.2, (2, BUY))
common = rng.uniform(.5, 1.5, BUY)
p1 = np.tile(common, (12, 1)) * rng.uniform(.98, 1.02, (12, BUY))
for s, p in ((0, p0), (1, p1)):
    RHO[s] = p / p.sum(axis=0, keepdims=True)          # columns sum to one, as the model's

PI_R = rng.uniform(0.5, 2.0, BUY); PI_R /= PI_R.sum()

def sourcing_geometry(data, alpha=None, equalise_T=False):
    return {"by_sector": {s: {"cells": CELLS[s], "rho": RHO[s]} for s in range(S)},
            "alpha": 0.4, "theta": 1.0, "downstream": buyers}

def _buyer_weights(data):
    return PI_R

def _by_sector_code(df, data):
    return df

data = {"S": S, "R": R, "sector_names": ["A", "B"], "emp_pi_r": PI_R}
PORTFOLIO_TAUS = (0.6, 0.7, 0.8)
sim_color = (.2, .4, .7); toulouse_color = (.5, .2, .1)
# The library, with the fixture installed where its functions resolve their globals.
import utils, granular_lib
install(globals(), [utils, granular_lib])

# --- 1. the cosine, and its scale invariance ---------------------------------
X, cells = portfolio_matrix(data, 0)
C, ok = portfolio_similarity(X)
assert ok.all() and C.shape == (12, 12)
assert np.allclose(np.diag(C), 1.0) and np.allclose(C, C.T)
assert C.min() >= -1 - 1e-12 and C.max() <= 1 + 1e-12
# SCALE INVARIANCE, the correction the plan asks to state: shares and raw sales, and any
# per-supplier rescaling, give the SAME C. If they did not, the normalisation would be a
# modelling choice rather than a convention.
sc = rng.uniform(0.1, 10.0, X.shape[0])[:, None]
assert np.allclose(portfolio_similarity(X * sc)[0], C, atol=1e-12)
assert np.allclose(portfolio_similarity(X / X.sum(axis=1, keepdims=True))[0], C, atol=1e-12)
# but the BUYER weighting is NOT a convention -- it must move the angles
Xu, _ = portfolio_matrix(data, 0, weights=None)
assert not np.allclose(portfolio_similarity(Xu)[0], C, atol=1e-6)
# a cell with no sales has no angle and is dropped, not counted as similar to everything
X0 = X.copy(); X0[3] = 0.0
C0, ok0 = portfolio_similarity(X0)
assert ok0.sum() == 11 and C0.shape == (11, 11) and np.isfinite(C0).all()
try:
    portfolio_matrix(data, 0, weights=np.ones(BUY + 1)); raise AssertionError("no raise")
except ValueError:
    pass
print("1 ok  cosine is symmetric, in [-1,1], invariant to portfolio scale and to the "
      "share normalisation, but NOT to the buyer weighting; empty portfolios are dropped")

# --- 2. the groups -----------------------------------------------------------
lab, A = portfolio_groups(C, 0.8)
assert not A.diagonal().any() and (A == A.T).all()
assert (A == ((C > 0.8) & ~np.eye(12, dtype=bool))).all()
# the two planted blocks must come out as blocks: cells 0-4 together, 5-9 together, apart
assert len(set(lab[:5])) == 1 and len(set(lab[5:10])) == 1
assert lab[0] != lab[5]
# CHAINING, named in the text: lowering tau merges everything into one giant component
assert portfolio_groups(C, 0.0)[0].max() == 0
# and the group count must be monotone in tau (a higher bar cuts, never joins)
ng = [portfolio_groups(C, t)[0].max() + 1 for t in (0.3, 0.6, 0.9, 0.99)]
assert all(x <= y for x, y in zip(ng, ng[1:])), ng
print(f"2 ok  the planted blocks are recovered at tau = 0.8; the group count rises with "
      f"tau {ng} and collapses to one component at tau = 0 (the chaining the text names)")

# --- 3. the decomposition ----------------------------------------------------
cw, cb, dl = portfolio_decomposition(C, lab)
# ORDERED pairs: the denominators must be sum_g N_g(N_g-1) and sum_{g!=h} N_g N_h, which is
# what makes the two averages comparable. Recomputed BY HAND, sharing no code.
num_w = den_w = num_b = den_b = 0.0
for g in np.unique(lab):
    ig = np.flatnonzero(lab == g)
    for j in ig:
        for k in ig:
            if j != k:
                num_w += C[j, k]; den_w += 1
    for h in np.unique(lab):
        if h == g:
            continue
        ih = np.flatnonzero(lab == h)
        for j in ig:
            for k in ih:
                num_b += C[j, k]; den_b += 1
assert abs(cw - num_w / den_w) < 1e-12 and abs(cb - num_b / den_b) < 1e-12
assert abs(dl - (cw - cb)) < 1e-15 and dl > 0
# SINGLETONS: nothing in the within count, their whole row in the between one
lab_one = np.arange(12)                                    # every cell its own group
assert np.isnan(portfolio_decomposition(C, lab_one)[0])
off = ~np.eye(12, dtype=bool)
assert abs(portfolio_decomposition(C, lab_one)[1] - C[off].mean()) < 1e-12
# a lone singleton beside one real group must not enter the within denominator
lab_mix = np.where(np.arange(12) < 6, 0, np.arange(12))
cw_m = portfolio_decomposition(C, lab_mix)[0]
ig = np.arange(6)
assert abs(cw_m - C[np.ix_(ig, ig)][~np.eye(6, dtype=bool)].mean()) < 1e-12
print(f"3 ok  ordered-pair denominators reproduce a hand recomputation; singletons are "
      f"absent from the within average and present in the between one "
      f"(within {cw:.3f}, between {cb:.3f}, delta {dl:.3f})")

# --- 4. the rewiring null ----------------------------------------------------
B = _rewire(A, np.random.default_rng(3))
assert (B.sum(axis=1) == A.sum(axis=1)).all(), "the rewiring must preserve every degree"
assert (B == B.T).all() and not B.diagonal().any()
rew, per = portfolio_null(C, A, lab, n_draws=60, seed=1)
assert rew.size == per.size == 60
# the PERMUTATION null never degenerates -- the sizes are fixed, so between pairs survive
assert np.isfinite(per).all()
# and it is NOT centred on zero for a lopsided size profile, which is exactly why the raw
# delta cannot be read against zero
assert dl > np.nanmean(per) + 3 * np.nanstd(per), (dl, np.nanmean(per), np.nanstd(per))
# the REWIRING null is reported with its degeneracy rather than silently averaged: on a
# graph this dense the rewired version is usually connected, leaving one group and no gap
print(f"4 ok  rewiring preserves every degree; the permutation null is finite on every "
      f"draw (mean {np.nanmean(per):+.3f}) and the planted delta {dl:+.3f} clears it, "
      f"while the rewiring null degenerates on {np.mean(~np.isfinite(rew)):.0%} of draws")

# --- 5. the report, and the segmented/common contrast ------------------------
rep = customer_portfolio_report(data, n_draws=40, seed=2, verbose=False)
assert len(rep) == S * len(PORTFOLIO_TAUS)
assert ((rep["null_degenerate"] >= 0) & (rep["null_degenerate"] <= 1)).all()
assert set(rep["tau"]) == set(PORTFOLIO_TAUS)
# the permutation null scores wherever the gap itself is defined -- both are NaN together,
# and only when the whole sector collapses into ONE group and there are no between pairs
fin = np.isfinite(rep["delta"].to_numpy(dtype=float))
assert (fin == np.isfinite(rep["delta_null_mean"].to_numpy(dtype=float))).all()
assert (rep.loc[~fin, "n_groups"] == 1).all()
# The segmented sector separates at the two higher thresholds and CHAINS into one group at
# the lowest -- the limitation the text names, visible in the table rather than argued.
A8 = rep[(rep["sector_name"] == "A") & (rep["tau"] == 0.8)].iloc[0]
A7 = rep[(rep["sector_name"] == "A") & (rep["tau"] == 0.7)].iloc[0]
A6 = rep[(rep["sector_name"] == "A") & (rep["tau"] == 0.6)].iloc[0]
assert A8["delta_excess"] > 0.4 and A7["delta_excess"] > 0.4
assert A6["n_groups"] == 1 and not np.isfinite(A6["delta"])
# The common-clientele sector admits no partition at ANY threshold: every C_jk is alike, so
# the graph is complete and there is nothing to separate. That is the delta ~ 0 case.
Bs = rep[rep["sector_name"] == "B"]
assert (Bs["n_groups"] == 1).all() and (Bs["largest_share"] == 1.0).all()
assert not np.isfinite(Bs["delta"].to_numpy(dtype=float)).any()
# and it has the HIGHER mean similarity -- the case the text says a mean cannot distinguish
assert float(Bs["C_bar"].iloc[0]) > float(A8["C_bar"]) and float(Bs["C_bar"].iloc[0]) > 0.9
# the rewiring null's degeneracy is REPORTED, not averaged away
assert (rep["null_degenerate"] > 0.5).all()
print(f"5 ok  the segmented sector separates at tau >= 0.7 (excess {A8['delta_excess']:+.3f}) "
      f"and chains into one group at 0.6; the common-clientele sector admits no partition at "
      f"any tau despite the HIGHER mean similarity "
      f"({float(Bs['C_bar'].iloc[0]):.3f} against {float(A8['C_bar']):.3f}), and the "
      f"rewiring null is flagged degenerate on every row")

# --- 6. the figure -----------------------------------------------------------
rep = rep[np.isfinite(rep["delta"].to_numpy(dtype=float))]
ax = plot_portfolio_decomposition({"X": rep}, tau=0.8)[0][0]
t8 = rep[rep["tau"] == 0.8]
assert [l.get_text() for l in ax.get_yticklabels()] == [str(x) for x in t8["sector_name"]]
assert len(ax.collections) == 1                       # the between/within connectors
_series = [l for l in ax.lines if l.get_linestyle() == "None"]
assert len(_series) == 3 and all(len(l.get_xdata()) == len(t8) for l in _series)
fig = ax.figure
assert len(fig.legends) == 1 and len(fig.legends[0].get_texts()) == 3
print("6 ok  the figure draws between, within and the null as three unjoined series")

print("\nall 6 gates pass")
