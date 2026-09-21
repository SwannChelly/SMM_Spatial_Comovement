"""Gate for the new concentration/commonality block: a planted economy whose answers
need no arithmetic, plus the identities the code claims."""
import json, math, numpy as np, pandas as pd, matplotlib, os
matplotlib.use("Agg"); import matplotlib.pyplot as plt

_NB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "diffusion.ipynb")
nb = json.load(open(_NB))
# located by CONTENT, not by index: a cell inserted above must not silently shift this
# gate onto another section's definitions. The section is split across three cells —
# the buyer's own portfolio, the commonality exercise, and the report that joins them
# — so all three are taken, in notebook order, and a missing one is named rather than
# surfacing later as a NameError on whichever function it held.
_ANCHORS = ('def sector_concentration(', 'def concentration_summary(',
            'def concentration_report(')
_cells = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code'
          and any(a in ''.join(c['source']) for a in _ANCHORS)]
_missing = [a for a in _ANCHORS if not any(a in c for c in _cells)]
assert not _missing, f"no notebook cell defines {_missing}"
assert len(_cells) == len(_ANCHORS), f"{len(_cells)} cells carry the section's anchors"
code = "\n".join(_cells)

S, R, THETA = 3, 12, 1.3
rng = np.random.default_rng(3)
CELL_MASK = np.zeros((S, R), bool)
for s in range(S): CELL_MASK[s, rng.choice(R, 8, replace=False)] = True
buyers = np.array([1, 2, 3, 4, 5])
D = rng.uniform(20, 600, (R, R)); np.fill_diagonal(D, 10.)
D = (D + D.T) / 2
Tcell = {s: rng.lognormal(0, .6, size=CELL_MASK[s].sum()) for s in range(S)}
N_HAT = np.array([4., 10., 30.])
ALPHA = 0.4

def sourcing_geometry(data, alpha=None, equalise_T=False):
    a = ALPHA if alpha is None else float(alpha)
    out = {}
    for s in range(S):
        cells = np.flatnonzero(CELL_MASK[s])
        T = np.ones(cells.size) if equalise_T else Tcell[s]
        d = np.maximum(D[np.ix_(cells, buyers - 1)], 1.0)
        psi = T[:, None] * d ** (-THETA * a)
        out[s] = {"cells": cells, "T_cell": T, "distance": d,
                  "rho": psi / psi.sum(0, keepdims=True)}
    return {"by_sector": out, "alpha": a, "theta": THETA, "downstream": buyers}

def _parquet_sector_index(data, sup): return sup["A129"].to_numpy().astype(int) - 1
def _region_labels(data):
    return pd.DataFrame({"index": np.arange(1, R + 1), "ze2010": np.arange(1, R + 1),
                         "ze2010_name": [f"Z{i}" for i in range(1, R + 1)]})
def _n_hat_from_diagnostics(data): return N_HAT
# buyer SIZE, deliberately unequal (one buyer is ten times the smallest), so a size
# weighting that came back uniform is caught rather than passing vacuously.
EMP_PI_R = np.array([0.40, 0.25, 0.20, 0.11, 0.04])
def _downstream_ze_index(data): return buyers
def model_theta(data): return THETA
NU_S_DEFAULT = 1.5

# a parquet drawn FROM the geometry, VARIETY BY VARIETY: each variety of a sector is
# won by one cell per buyer. `mode` controls the cross-buyer dependence, which is the
# whole content of Q vs G:
#   "independent" -> each buyer draws its own winner  => Q = G, rho = 0
#   "shared"      -> one winner serves every buyer    => Q = 1, rho -> 1
# and `unequal` switches the variety expenditure shares off equality, where
# Cauchy-Schwarz requires V > 1/N.
geom = sourcing_geometry(None)
spend_s = np.array([1.0, 2.0, 3.0])
B = 60

def build_parquet(mode="independent", unequal=False, rng=rng):
    rows = []
    for b in range(B):
        for s in range(S):
            blk = geom["by_sector"][s]
            N = int(N_HAT[s]); ncell = blk["cells"].size
            for rho in range(N):
                if mode == "shared":
                    w = np.full(len(buyers), rng.choice(ncell, p=blk["rho"][:, 0]))
                else:
                    w = np.array([rng.choice(ncell, p=blk["rho"][:, j])
                                  for j in range(len(buyers))])
                v = (rng.random(len(buyers)) if unequal
                     else np.ones(len(buyers))) if unequal else np.ones(len(buyers))
                for j, rd in enumerate(buyers):
                    rows.append((b, int(rd), int(blk["cells"][w[j]]) + 1, s + 1,
                                 rho, spend_s[s] * float(v[j])))
    df = pd.DataFrame(rows, columns=["replication", "ze2010_downstream", "ze2010",
                                     "A129", "variety", "share"])
    # equal expenditure across a buyer's varieties unless `unequal`
    return df

sup = build_parquet("independent")
data = {"S": S, "R": R, "CELL_MASK": CELL_MASK, "suppliers": sup,
        "sector_names": ["A", "B", "C"], "post_hoc_N_hat": N_HAT,
        "folder": "x", "step_dir": "step3", "emp_pi_r": EMP_PI_R}

CF_REGIMES = {"Both forces": dict(), "Distance only": dict(equalise_T=True),
              "Comparative advantage only": dict(alpha=0.0)}
UNIFORM_REGIME = "Uniform benchmark"
sim_color = (.2, .4, .7); toulouse_color = (.5, .2, .1)
CF_COLORS = {"Both forces": toulouse_color, "Distance only": sim_color,
             "Comparative advantage only": (.45, .60, .45)}
exec(code, globals())

# --- 1. the identity, and the alpha=0 control -------------------------------
sec = sector_concentration(data, verbose=False)
assert np.allclose(sec["h_bar"], sec["h_common"] + sec["m"], atol=1e-12)
ca = sec.xs("Comparative advantage only", level="regime")
assert np.allclose(ca["C"], 1.0, atol=1e-12), ca["C"].to_numpy()   # exact, a theorem
assert np.allclose(sec.loc[UNIFORM_REGIME, "n_eff_ratio"], 1.0)
assert (sec["m"] >= -1e-15).all()
assert np.allclose(sec.loc[UNIFORM_REGIME, "n_eff"], sec.loc[UNIFORM_REGIME, "n_cells"])
print("1 ok  identity, alpha=0 gives C=1 exactly, uniform gives n_eff = n_cells")

summ = concentration_summary(sec, verbose=False)
assert np.allclose(summ["h_bar"], summ["h_common"] + summ["m"])
assert abs(summ.loc["Comparative advantage only", "C"] - 1) < 1e-12
by = buyer_concentration(sec, data=data)
w = by.xs("Both forces", level="regime")
assert np.allclose(np.average(w["h"], weights=w["spend"]),
                   summ.loc["Both forces", "h_bar"])
print("2 ok  the buyer aggregation reproduces the industry number")

# --- 3. the derivatives ------------------------------------------------------
der = concentration_derivatives(data, verbose=False)
assert der.attrs["identity_residual"] < 1e-5, der.attrs["identity_residual"]
print(f"3 ok  derivative identity residual {der.attrs['identity_residual']:.2e}")

# --- 4. granularity ----------------------------------------------------------
gr = granularity_concentration(data, verbose=False)
assert gr.attrs["identity_residual"] < 1e-12
assert (gr["h_realised"] > gr["h_structural"]).all()          # Jensen, one-directional
err = np.max(np.abs(gr["h_predicted"] - gr["h_realised"]) / gr["h_realised"])
assert err < 0.03, err
# substitutes: the granular addition falls with N_s
add = gr["h_realised"] - gr["h_structural"]
assert add.iloc[0] > add.iloc[-1]
print(f"4 ok  four cells add up; E[H] formula within {err:.3%}; "
      f"granular addition {add.to_numpy().round(3)} falls with N_s")

chk = expected_network_check(data, verbose=False)
assert (chk["corr"] > .95).all()
print(f"5 ok  E[omega] vs rho: median TV {chk['tv'].median():.4f}")

zm = incidence_variance_map(data, verbose=False)
assert abs(zm["share_of_M"].sum() - 1) < 1e-12
print("6 ok  the zone attribution of M sums to one")

tab = concentration_table(data, sector=sec, granular=gr, verbose=False)
assert np.isnan(tab.loc["Comparative advantage only", "gran_share_spec"])
axes = plot_concentration_decomposition({"X": gr}, save_to="/tmp/x1.pdf")
assert len(axes[0][0].patches) == 4 * len(gr)
# The top band is three stacked things (twin ticks, twin label, title) plus a figure
# legend, and matplotlib places the title without seeing the twin — so the stacking is
# gated in DISPLAY coordinates rather than trusted to the pads that produce it.
figd = axes[0][0].figure
figd.canvas.draw(); _r = figd.canvas.get_renderer()
assert len(figd.legends) == 1 and not any(a.get_legend() for a in figd.axes)
_leg = figd.legends[0].get_window_extent(_r)
assert len(figd.legends[0].get_texts()) == 5      # 4 segments + N_s, no variety frame
for _ax in axes[0]:
    _tw = [a for a in figd.axes
           if a.get_position().bounds == _ax.get_position().bounds and a is not _ax][0]
    _t = _tw.title.get_window_extent(_r)
    _xl = _tw.xaxis.label.get_window_extent(_r)
    _tk = max(lb.get_window_extent(_r).y1 for lb in _tw.get_xticklabels()
              if lb.get_text())
    assert _tw.get_title() and not _ax.get_title()      # the title is on the TWIN
    assert _xl.y0 >= _tk - 1, "the twin label sits on its own ticks"
    assert _t.y0 >= _xl.y1, "the panel title sits on the twin label"
    assert _leg.y0 >= _t.y1, "the legend sits on a panel title"
# The default must keep the TABLE's own row order: sorting by `n_hat_s` would turn
# the y axis into a ranking of variety counts, which is not what the bars measure.
_want = [str(r.sector_name) for r in gr.itertuples()]
assert [t.get_text() for t in axes[0][0].get_yticklabels()] == _want
# the kwarg must still be live: a column the planted table is NOT already sorted on
_gr2 = gr.assign(_k=np.arange(len(gr))[::-1])
_ax2 = plot_concentration_decomposition({"X": _gr2}, order_by="_k")[0][0]
assert [t.get_text() for t in _ax2.get_yticklabels()] == _want[::-1]
axes = plot_buyer_concentration({"X": by}, save_to="/tmp/x2.pdf")
print("7 ok  table and both figures render; rows keep the table order by default")

rep = concentration_report(data, industry="gate", verbose=False)
assert set(rep) == {"sector", "summary", "derivatives", "buyers", "buyer_granular",
                    "reach", "tail", "variety", "cosourcing", "granular",
                    "granular_equal", "granular_size", "table", "zones"}
assert rep["table"] is not None and rep["variety"] is not None
assert rep["granular"] is not None and rep["granular_equal"] is not None
assert rep["granular_size"] is not None
print("8 ok  concentration_report wires every piece together")


# --- 9. the variety decomposition -------------------------------------------
pan = variety_panel(data)
vc = variety_concentration(data, panel=pan, verbose=False)
# equal expenditure shares were planted, so V must be exactly 1/N_s
assert np.allclose(vc["V"], 1.0 / N_HAT, rtol=1e-12), vc["V"].to_numpy()
assert np.allclose(vc["V_times_N"], 1.0, rtol=1e-12)
assert vc["V_buyer_range"].max() < 1e-12          # buyer-independent by construction
print("9 ok  V = 1/N_s exactly under equal variety shares, buyer-independent")

data_u = dict(data, suppliers=build_parquet("independent", unequal=True))
vcu = variety_concentration(data_u, verbose=False)
assert (vcu["V_times_N"] > 1.0 + 1e-9).all(), vcu["V_times_N"].to_numpy()
print(f"10 ok unequal shares give V x N = {vcu['V_times_N'].round(2).to_list()} > 1 "
      "(Cauchy-Schwarz)")

# Q = G when winners are drawn independently across buyers; Q = 1 when shared
co = cosourcing(data, panel=pan, verbose=False)
assert np.abs(co["Q_minus_G"]).max() < 0.02, co["Q_minus_G"].to_numpy()
data_s = dict(data, suppliers=build_parquet("shared"))
co_s = cosourcing(data_s, verbose=False)
assert (co_s["Q_off"] > 0.999).all(), co_s["Q_off"].to_numpy()
print("11 ok independent winners give Q = G; shared winners give Q = 1")

cells = granular_cells(data, panel=pan, verbose=False)
tot = cells[["struct_common", "struct_specific", "gran_common", "gran_specific"]]
assert np.allclose(tot.sum(axis=1), cells["E_H_bar"], atol=1e-12)
# with winners drawn independently across buyers the off-diagonal Q - G vanishes, so
# rho collapses onto its FLOOR: the (granular-weighted) Herfindahl of buyer spending.
assert np.allclose(cells["rho_s"], cells["rho_floor"], atol=0.02), \
    (cells["rho_s"].to_numpy(), cells["rho_floor"].to_numpy())
cells_s = granular_cells(data_s, verbose=False)
assert (cells_s["rho_s"] > 0.9).all(), cells_s["rho_s"].to_numpy()
# E[H] = V + (1-V) H^gamma, checked against the realised mean of the same draws
emp = granularity_concentration(data, verbose=False)
gap = np.nanmax(np.abs(cells["E_H_bar"] - emp["h_realised"]) / emp["h_realised"])
assert gap < 0.02, gap
# --- 12b. the equal-buyer reweighting -----------------------------------------
# The column TOTAL must survive, so the cross-SECTOR weights are untouched and the two
# decompositions differ for exactly one reason; a zero-spend buyer must stay at zero
# rather than being handed a share of the average portfolio it does not have.
_sp = _sector_spend(data)
_sp.iloc[0, 0] = 0.0                       # plant a buyer that does not buy sector 0
_eq = equal_buyer_spend(_sp)
assert np.allclose(_eq.sum(axis=0), _sp.sum(axis=0), rtol=1e-12)
assert _eq.iloc[0, 0] == 0.0
for _c in _eq.columns:
    _v = _eq[_c].to_numpy()[_eq[_c].to_numpy() > 0]
    assert _v.size and np.allclose(_v, _v[0], rtol=1e-12)
assert ((_sp > 0) == (_eq > 0)).all().all()
# rho_floor is the granular-weighted Herfindahl of buyer spending, so under equal
# weights over n positive buyers it must land exactly on 1/n -- the statistic whose
# movement between the two figures IS the customer size distribution.
_base = _sector_spend(data)
_ce = granular_cells(data, panel=pan, spend=equal_buyer_spend(_base), verbose=False)
_n = (_base > 0).sum(axis=0).reindex(_ce.index).to_numpy(dtype=float)
assert np.allclose(_ce["buyer_hhi"], 1.0 / _n, rtol=1e-12), _ce["buyer_hhi"].to_numpy()
_tot = _ce[["struct_common", "struct_specific", "gran_common", "gran_specific"]]
assert np.allclose(_tot.sum(axis=1), _ce["E_H_bar"], atol=1e-12)
# The SIZE weighting must reproduce `emp_pi_r` on the support, and must NOT coincide with
# the equal one -- the planted pi_r is deliberately unequal, so a size variant that came
# back uniform would mean the weights never reached `pi`.
_sz = size_buyer_spend(_base, data)
_w = np.asarray(data["emp_pi_r"], dtype=float).ravel()
_wi = pd.Series(_w, index=_downstream_ze_index(data)).reindex(_base.index).to_numpy()
for _c in _sz.columns:
    _m = (_base[_c] > 0).to_numpy()
    assert np.allclose(_sz[_c].to_numpy()[~_m], 0.0)
    _got, _want = _sz[_c].to_numpy()[_m], _wi[_m]
    assert np.allclose(_got / _got.sum(), _want / _want.sum(), rtol=1e-12)
assert np.allclose(_sz.sum(axis=0), _base.sum(axis=0), rtol=1e-12)
_cs = granular_cells(data, panel=pan, spend=_sz, verbose=False)
assert not np.allclose(_cs["buyer_hhi"], _ce["buyer_hhi"], rtol=1e-6), \
    "the size weighting collapsed onto the equal one"
assert (_cs["buyer_hhi"].to_numpy() > _ce["buyer_hhi"].to_numpy() - 1e-12).all()
print(f"12b ok equal weights put buyer_hhi on 1/n exactly; the size weighting reproduces "
      f"emp_pi_r on the support and is strictly more concentrated "
      f"(1/hhi {float(1/_cs['buyer_hhi'].mean()):.2f} vs {float(1/_ce['buyer_hhi'].mean()):.2f})")

print(f"12 ok four cells add up; rho hits its floor {cells['rho_floor'].mean():.2f} "
      f"when winners are independent and >0.9 when shared; the variety route "
      f"reproduces the realisation route to {gap:.1%}")

# the closed form diverges at this calibration and must say so
tail = variety_tail_index(data, verbose=False)
assert np.isclose(tail["kappa"].iloc[0], THETA / (NU_S_DEFAULT - 1))
assert not np.isfinite(_xi_of_kappa(2.0)) and _xi_of_kappa(4.0) > 1
print(f"13 ok kappa = {tail['kappa'].iloc[0]:.2f}; Xi diverges at kappa <= 2")

# the re-simulated economy reproduces the parquet's own statistics, and alpha = 0
# gives the exact controls
sim = simulate_granular_regime(data, n_rep=40, n_hat=N_HAT, seed=5)
c_sim = granular_cells(data, panel=sim, empirical=None, verbose=False)
c0 = granular_cells(data, panel=simulate_granular_regime(
    data, alpha=0.0, n_rep=20, n_hat=N_HAT, seed=6), empirical=None, verbose=False,
    structural=structural_networks(data, alpha=0.0, buyers=buyers))
assert np.allclose(c0["rho_s"], 1.0, atol=1e-12), c0["rho_s"].to_numpy()
assert np.allclose(c0["C_structural"], 1.0, atol=1e-12)
assert np.allclose(c0["C_realised"], 1.0, atol=1e-12)
assert (c_sim["gran_common"] > 0).all()     # shared draws => positive dependence
print("14 ok re-simulated regime: alpha = 0 gives Q = rho = C = 1 exactly; the "
      "estimate carries a strictly positive granular common cell")


# --- 15. the BUYER-level object: E_Omega[H_r] regime by regime ----------------
# The structure is what is gated, not a blob. Three things must hold by construction.
# (a) `Infinite varieties` IS the structural column, so its granular term is exactly
#     zero and its effective number is 1/H(gamma) — that series is the figure's point.
# (b) Granularity is Jensen: E[H_r] >= H(gamma_r) for EVERY buyer under every regime,
#     with equality only in the infinite-variety limit.
# (c) Under equal variety shares the buyer-level formula collapses to the binomial one,
#     so `h` must equal `sum_s theta_rs [H + (1-H)/N_s]` recomputed by hand — an
#     independent route that shares no code with `buyer_granular_concentration`.
bg = buyer_granular_concentration(data, panel=pan, verbose=False)
assert set(bg.index.get_level_values("regime")) == {
    "Both forces", "Distance only", "Comparative advantage only",
    UNIFORM_REGIME, INFINITE_REGIME}
inf = bg.loc[INFINITE_REGIME]
assert np.allclose(inf["gran"], 0.0, atol=1e-14) and np.allclose(inf["V"], 0.0)
assert np.allclose(inf["n_eff"], 1.0 / inf["h_struct"], rtol=1e-12)
assert (bg["gran"].dropna() >= -1e-12).all()
assert (bg.drop(index=INFINITE_REGIME, level="regime")["gran"].dropna() > 0).all()

# (c) the hand recomputation, on the estimated regime
sp = _sector_spend(data)
nets = structural_networks(data, buyers=buyers)
num = np.zeros(buyers.size); den = np.zeros(buyers.size)
for s, blk in nets["by_sector"].items():
    pi = sp.iloc[:, s].to_numpy(dtype=float)
    hg = (np.asarray(blk["W"], float) ** 2).sum(axis=1)
    num += pi * (hg + (1.0 - hg) / N_HAT[s]); den += pi
hand = pd.Series(num / den, index=buyers)
got = bg.loc["Both forces", "h"].reindex(buyers)
assert np.allclose(got.to_numpy(), hand.to_numpy(), rtol=1e-12), \
    (got.to_numpy(), hand.to_numpy())

# the buyer rows must aggregate to the sector table's own structural numbers
h_struct_ind = float((bg.loc["Both forces", "h_struct"] *
                      bg.loc["Both forces", "spend"]).sum() /
                     bg.loc["Both forces", "spend"].sum())
w = sec.loc["Both forces", "spend"]
assert abs(h_struct_ind - float((sec.loc["Both forces", "h_bar"] * w).sum() / w.sum())) < 1e-12
print(f"15 ok buyer-level E[H_r]: infinite-variety series is exactly structural, "
      f"granularity is strictly positive elsewhere, and the binomial recomputation "
      f"matches to machine precision")

# gran = V_s x slack_rs is EXACT sector by sector; at the buyer level the product of
# the two spend-weighted averages misses the cross-sector covariance by exactly
# Cov_theta(V_s, slack_rs). Both halves are gated, the second against a covariance
# recomputed BY HAND from the per-sector pieces -- the planted N_HAT = [4, 10, 30]
# makes V_s differ across sectors, so the covariance is genuinely non-zero here and
# the test is not vacuous (an earlier draft of this gate asserted it was nil, and the
# gate caught that).
assert np.allclose(bg["slack"].to_numpy(), 1.0 - bg["h_struct"].to_numpy(), rtol=0,
                   atol=1e-15)
fin = bg.drop(index=INFINITE_REGIME, level="regime")
sp_g = _sector_spend(data)
# "Both forces" reads the PLANTED panel, whose variety shares are equal, so V_s is
# exactly 1/N_s and the hand check shares NO code with the function. `V` is measured
# per (sector, buyer), so the covariance is taken over sectors with that buyer's own
# V_rs and its own input mix — the aggregate V_s of the per-sector table would be the
# wrong object wherever V varies across buyers.
nets_l = structural_networks(data, buyers=buyers)
secs = list(nets_l["by_sector"])
for b_i, b in enumerate(buyers):
    th = np.array([sp_g.iloc[b_i, s] for s in secs], dtype=float)
    th = th / th.sum()
    Vs = np.array([1.0 / N_HAT[s] for s in secs])
    sl = np.array([1.0 - (np.asarray(nets_l["by_sector"][s]["W"], float)[b_i] ** 2).sum()
                   for s in secs])
    cov = float(th @ (Vs * sl) - (th @ Vs) * (th @ sl))
    assert abs(float(bg.loc[("Both forces", b), "gran_resid"]) - cov) < 1e-9, (b, cov)
# every regime: the residual is the gap between the product of averages and the
# average of products, so it can never exceed the granular term it decomposes
assert (np.abs(fin["gran_resid"].to_numpy()) < fin["gran"].to_numpy()).all()
assert np.abs(fin["gran_resid"].to_numpy()).max() > 1e-6      # not vacuous
bs = bg.attrs["by_sector"].loc["Both forces"]
assert set(bs.columns) >= {"V_s", "slack (median)", "gran (median)", "spend share"}
assert abs(float(bs["spend share"].sum()) - 1.0) < 1e-12
assert (bg.loc[INFINITE_REGIME, "gran"] == 0).all()   # no varieties, no granularity
print(f"15b ok gran = V x slack; the buyer-level gap reproduces the hand-computed "
      f"cross-sector covariance (slack median {fin['slack'].median():.4f}, V median "
      f"{fin['V'].median():.4f})")

ax = plot_buyer_granular_concentration(bg, save_to="/tmp/x3.pdf")
# one bar per (buyer, non-baseline regime); the baseline is the zero rule, not a
# series, and `Infinite varieties` is dropped by default
n_b = bg.loc["Both forces"].shape[0]
n_reg = bg.index.get_level_values("regime").nunique()
assert len(ax.patches) == (n_reg - 2) * n_b, len(ax.patches)
assert len(ax.get_yticklabels()) == n_b
# the right-margin annotation IS the baseline level, one per row plus the header
lv = [t.get_text() for t in ax.texts]
assert len(lv) == n_b + 1, lv
base_h = bg.loc["Both forces", "n_eff"].reindex(
    [ix for ix in bg.loc["Both forces"].index])
assert sorted(float(t) for t in lv[:n_b]) == sorted(
    round(float(v), 1) for v in base_h), (lv, base_h.to_dict())
ax_lvl = plot_buyer_granular_concentration(bg, units="level", save_to="/tmp/x4.pdf")
assert len(ax_lvl.patches) == (n_reg - 1) * n_b
# drop=() restores the infinite series; an unknown regime is named, not ignored
ax_all = plot_buyer_granular_concentration(bg, drop=(), save_to="/tmp/x4b.pdf")
assert len(ax_all.patches) == (n_reg - 1) * n_b
try:
    plot_buyer_granular_concentration(bg, drop=("No such regime",))
    raise AssertionError("dropping an absent regime must raise")
except KeyError:
    pass
print("16 ok the per-buyer figure carries one bar per (buyer, regime), drops the "
      "baseline series in percentage units and the infinite-variety series by "
      "default, and annotates each row with its baseline level")


# --- 17. the REGION count, which carries no variety weighting -----------------
# `1/E[H_r]` is mostly `1/V` at any realistic N_s, so the reach is gated on the three
# things that make it a geography count rather than a variety count.
# (a) Against a BRUTE-FORCE recomputation: for each buyer, 1 - prod_s (1-gamma)^N_s
#     accumulated zone by zone with an explicit Python loop, sharing no code with the
#     vectorised scatter-add the function uses.
# (b) MONOTONE in N_s and bounded by its own support, with the infinite-variety row
#     equal to the support exactly.
# (c) INVARIANT to the expenditure shares: doubling one sector's spend changes the
#     input mix and hence E[H_r], but cannot change which zones are reached.
rc = buyer_region_reach(data, n_hat=N_HAT, verbose=False)
nets = structural_networks(data, buyers=buyers)

# (a) brute force
import collections
hand = {}
for b_i, b in enumerate(buyers):
    logfail = collections.defaultdict(float)
    for s, blk in nets["by_sector"].items():
        for c_i, c in enumerate(blk["cells"]):
            g = float(blk["W"][b_i, c_i])
            logfail[int(c)] += N_HAT[s] * math.log1p(-g)
    hand[int(b)] = sum(1.0 - math.exp(v) for v in logfail.values())
got = rc.loc["Both forces", "reach"]
assert np.allclose([hand[int(b)] for b in buyers], got.reindex(buyers).to_numpy(),
                   rtol=1e-12), (hand, got.to_dict())

# (b) support, monotonicity, the infinite row
assert (rc["reach"] <= rc["support"] + 1e-9).all()
assert np.allclose(rc.loc[INFINITE_REGIME, "reach"],
                   rc.loc[INFINITE_REGIME, "support"], rtol=0, atol=0)
assert np.allclose(rc.loc[INFINITE_REGIME, "reach_share"], 1.0)
lo = buyer_region_reach(data, n_hat=N_HAT / 4, include_infinite=False, verbose=False)
hi = buyer_region_reach(data, n_hat=N_HAT * 4, include_infinite=False, verbose=False)
assert (lo["reach"] < rc.drop(index=INFINITE_REGIME, level="regime")["reach"]).all()
assert (hi["reach"] > rc.drop(index=INFINITE_REGIME, level="regime")["reach"]).all()
assert (hi["reach"] <= hi["support"] + 1e-9).all()

# (c) the expenditure shares cannot move it, but they DO move E[H_r] — so the
#     invariance is a separating test, not a vacuous one
sp2 = _sector_spend(data).copy(); sp2.iloc[:, 0] *= 5.0
rc2 = buyer_region_reach(data, spend=sp2, n_hat=N_HAT, verbose=False)
assert np.allclose(rc2["reach"].to_numpy(), rc["reach"].to_numpy(), rtol=1e-12)
bg2 = buyer_granular_concentration(data, spend=sp2, panel=pan, verbose=False)
assert not np.allclose(bg2.loc["Both forces", "h"].to_numpy(),
                       bg.loc["Both forces", "h"].to_numpy(), rtol=1e-6)
print(f"17 ok reach matches a brute-force recomputation, rises with N_s, is capped by "
      f"its support ({rc.loc['Both forces','reach'].median():.1f} of "
      f"{rc.loc['Both forces','support'].median():.0f} zones at the estimate), and is "
      f"invariant to the expenditure shares that move E[H_r]")

ax = plot_buyer_region_reach(rc, save_to="/tmp/x5.pdf")
assert len(ax.patches) == 5 * n_b, len(ax.patches)      # levels keep the baseline
assert ax.get_xlim()[0] == 0.0                          # a count has an honest zero
ax_p = plot_buyer_region_reach(rc, units="pct", save_to="/tmp/x6.pdf")
assert len(ax_p.patches) == 4 * n_b
# the concentration figure on a log axis must refuse bars: a bar's length would
# encode the axis limits rather than the number
ax_l = plot_buyer_granular_concentration(bg, units="level", kind="point", logx=True,
                                         drop=(), save_to="/tmp/x7.pdf")
assert ax_l.get_xscale() == "log" and len(ax_l.patches) == 0
assert len(ax_l.lines) == n_reg + n_b    # one series per regime + one rule per buyer
for bad in (dict(units="level", kind="bar", logx=True), dict(units="pct", logx=True),
            dict(quantity="H")):
    try:
        plot_buyer_granular_concentration(bg, **bad); raise AssertionError(bad)
    except ValueError:
        pass
# `quantity` must draw the TABLE'S OWN column, not a transform of it: the paper is
# written in H, and `1/E[H_r]` is the reciprocal, not a rescaling, so a figure that
# silently kept drawing n_eff would invert the ranking of every regime.
for q in ("h", "n_eff"):
    ax_q = plot_buyer_granular_concentration(bg, quantity=q, units="level", drop=(),
                                             save_to=f"/tmp/x8_{q}.pdf")
    drawn = sorted(round(pt.get_width(), 12) for pt in ax_q.patches)
    assert drawn == sorted(round(v, 12) for v in bg[q]), q
    assert ("1/" in ax_q.get_xlabel()) == (q == "n_eff"), ax_q.get_xlabel()
print("18 ok the reach figure keeps an honest zero in levels; the concentration "
      "figure draws points on a log axis, refuses bars there, and draws whichever of "
      "H or 1/H the caller asked for")

# ---------------------------------------------------------------- 19
# The factor split. What it has to get right is not arithmetic but the BENCHMARK: a
# product cannot be attributed to its factors without one, so the gate pins the two
# break-evens to their defining property (each alone makes gran = H) and checks that
# the ratio built on them is the SAME number as the ratio of the terms -- which is
# what makes the comparison symmetric rather than a choice of grouping.
fs = granular_factor_decomposition(bg, regime="Both forces")
cells = fs.attrs["cells"]
assert len(cells) == len(secs) * n_b, (len(cells), len(secs), n_b)
V_c = cells["V"].to_numpy(); H_c = cells["h_struct"].to_numpy()
assert np.allclose(cells["gran"].to_numpy(), V_c * (1.0 - H_c), rtol=0, atol=1e-15)
# (a) V_s* is the variety Herfindahl at which granularity would EQUAL the network
Vst = cells["V_star"].to_numpy()
assert np.allclose(Vst * (1.0 - H_c), H_c, rtol=1e-12)
# (b) H_rs* likewise, holding V fixed
Hst = cells["H_star"].to_numpy()
assert np.allclose(V_c * (1.0 - Hst), Hst, rtol=1e-12)
# (c) the V-side ratio IS the ratio of the two terms -- so "how many times too many
#     varieties" and "how many times larger is granularity" are one statement
assert np.allclose(cells["V_over_Vstar"].to_numpy(), cells["gran_over_H"].to_numpy(),
                   rtol=1e-12)
# (d) the three exact groupings of gran/H agree, which is the point the docstring
#     makes: each factor can be made to look like the whole story
gh = cells["gran_over_H"].to_numpy()
assert np.allclose(gh, cells["V_over_H"].to_numpy() * (1.0 - H_c), rtol=1e-12)
assert np.allclose(gh, V_c * cells["slack_over_H"].to_numpy(), rtol=1e-12)
# (e) the slack channel can only shrink the ratio; the variety channel need not
assert (1.0 - H_c <= 1.0).all()
assert (cells["V_over_H"].to_numpy() > 1.0).any()
# (f) per-sector table: one row per sector, spend shares summing to one, and V_s
#     buyer-independent on the planted panel (equal variety shares => exactly 1/N_s)
assert list(fs.index) == sorted(secs), (list(fs.index), secs)
assert abs(float(fs["spend share"].sum()) - 1.0) < 1e-12
assert np.allclose(fs["V_s spread"].to_numpy(), 0.0, atol=1e-12)
assert np.allclose(fs["V_s"].to_numpy(),
                   [1.0 / N_HAT[s] for s in sorted(secs)], rtol=1e-12)
for bad in (dict(regime="Nonexistent regime"),):
    try:
        granular_factor_decomposition(bg, **bad); raise AssertionError(bad)
    except KeyError:
        pass
stripped = bg.copy(); stripped.attrs = {}
try:
    granular_factor_decomposition(stripped); raise AssertionError("no by_cell")
except KeyError:
    pass
# (g) the two aggregations of the ratio are both reported and the ratio-of-means one
#     is the buyer-level number, not the mean of ratios (they differ by Jensen, and
#     the gap is large wherever one heavy sector has a small ratio)
s19 = fs.attrs["summary"]
w19 = cells["spend"].to_numpy()
assert abs(s19["gran / H_rs (ratio of means)"]
           - np.average(cells["gran"], weights=w19)
           / np.average(cells["h_struct"], weights=w19)) < 1e-12
assert abs(s19["gran / H_rs (mean of ratios)"]
           - np.average(cells["gran_over_H"], weights=w19)) < 1e-12
print(f"19 ok the factor split is benchmarked rather than grouped: V_s/V_s* = "
      f"{s19['V_s / V_s*']:.2f} against H_rs*/H_rs = {s19['H_rs* / H_rs']:.2f}, each "
      f"defined by gran = H, and V_s/V_s* == gran/H_rs exactly")
