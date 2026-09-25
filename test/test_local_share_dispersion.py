"""Gate for Test 9 of the amplification section: the local share as a LEVEL plus a
DISPERSION.

The fixture is the planted economy of `test_concentration_identity.py` -- a parquet drawn
variety by variety FROM the geometry, with equal expenditure across a buyer's varieties, so
`V = 1/N_s` EXACTLY and every closed-form quantity has an answer that needs no arithmetic.
That is what makes the central gate real: the measured standard deviation across
replications and the closed form `sqrt(sum_s theta^2 V p(1-p))` are two routes that share no
code, and they must agree to the Monte-Carlo scale."""
import json, os, numpy as np, pandas as pd, matplotlib, warnings
matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

_NB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "diffusion.ipynb")
nb = json.load(open(_NB, encoding="utf-8"))
# located by CONTENT: the buyer-portfolio cell carries the machinery this test rides
# (`_sector_spend`, `structural_networks`, `variety_panel`, `simulate_granular_regime`,
# `_V_by_buyer`), the second carries the test itself. Both are executed, so a helper
# that moved cells is caught here rather than surfacing as a NameError in a run.
_ANCHORS = ("def sector_concentration(", "def local_share_dispersion(")
_cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"
          and any(a in "".join(c["source"]) for a in _ANCHORS)]
_missing = [a for a in _ANCHORS if not any(a in c for c in _cells)]
assert not _missing, f"no notebook cell defines {_missing}"
assert len(_cells) == len(_ANCHORS), f"{len(_cells)} cells carry the anchors"
code = "\n".join(_cells)

S, R, THETA = 3, 12, 1.3
rng = np.random.default_rng(3)
CELL_MASK = np.zeros((S, R), bool)
for s in range(S):
    CELL_MASK[s, rng.choice(R, 8, replace=False)] = True
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
def _downstream_ze_index(data): return buyers
def model_theta(data): return THETA
EMP_PI_R = np.array([0.40, 0.25, 0.20, 0.11, 0.04])
NU_S_DEFAULT = 1.5

geom = sourcing_geometry(None)
spend_s = np.array([1.0, 2.0, 3.0])
B = 200                     # sd of a sample sd is ~1/sqrt(2(B-1)) = 5% at this B


def build_parquet(rng=rng):
    """One winner per (replication, variety, buyer), drawn with probability `rho`, and
    EQUAL expenditure across a buyer's varieties -- so `V = 1/N_s` exactly."""
    rows = []
    for b in range(B):
        for s in range(S):
            blk = geom["by_sector"][s]
            N, ncell = int(N_HAT[s]), blk["cells"].size
            for rho in range(N):
                w = np.array([rng.choice(ncell, p=blk["rho"][:, j])
                              for j in range(len(buyers))])
                for j, rd in enumerate(buyers):
                    rows.append((b, int(rd), int(blk["cells"][w[j]]) + 1, s + 1, rho,
                                 float(spend_s[s])))
    return pd.DataFrame(rows, columns=["replication", "ze2010_downstream", "ze2010",
                                       "A129", "variety", "share"])


sup = build_parquet()
data = {"S": S, "R": R, "CELL_MASK": CELL_MASK, "suppliers": sup,
        "sector_names": ["A", "B", "C"], "post_hoc_N_hat": N_HAT,
        "folder": "x", "step_dir": "step3", "emp_pi_r": EMP_PI_R}

AMPLIFICATION_RADII = (100, 200)          # the Constants cell's value
CF_REGIMES = {"Both forces": dict(), "Distance only": dict(equalise_T=True),
              "Comparative advantage only": dict(alpha=0.0)}
UNIFORM_REGIME = "Uniform benchmark"
INFINITE_REGIME = "Infinite varieties"
sim_color = (.2, .4, .7); toulouse_color = (.5, .2, .1)
CF_COLORS = {"Both forces": toulouse_color, "Distance only": sim_color,
             "Comparative advantage only": (.45, .60, .45)}
exec(code, globals())

spend = _sector_spend(data)
theta = spend.div(spend.sum(axis=1), axis=0)
tab = local_share_dispersion(data, radius_km=None, verbose=False)
bs = tab.attrs["by_sector"]

# --- 1. the indicator, and the level it implies ------------------------------
# `1{l = r}` must pick the buyer's OWN zone and nothing else, and return an all-zero
# column for a buyer whose zone is not a modelled cell of that sector -- the fixture
# contains both cases, so neither branch passes vacuously.
nets = structural_networks(data, buyers=buyers)
hit_any, miss_any = 0, 0
for s, blk in nets["by_sector"].items():
    m = _local_indicator(blk, None)
    assert m.shape == (blk["cells"].size, buyers.size) and m.dtype == bool
    for j, b in enumerate(buyers):
        own = np.flatnonzero(blk["cells"] == b - 1)
        assert m[:, j].sum() == own.size
        if own.size:
            assert m[own[0], j]; hit_any += 1
        else:
            miss_any += 1
assert hit_any > 0 and miss_any > 0, (hit_any, miss_any)
# and the level is the hand-computed sum_s theta_rs gamma_rrs
lvl = np.zeros(buyers.size)
for s, blk in nets["by_sector"].items():
    W = np.asarray(blk["W"], float)
    g = np.array([W[j, np.flatnonzero(blk["cells"] == b - 1)[0]]
                  if (blk["cells"] == b - 1).any() else 0.0
                  for j, b in enumerate(buyers)])
    lvl += theta.iloc[:, s].to_numpy() * g
assert np.allclose(tab.loc["Both forces", "local"].to_numpy(), lvl, atol=1e-12)
print(f"1 ok  the own-zone indicator picks exactly the buyer's zone ({hit_any} buyers "
      f"have one in that sector, {miss_any} do not) and the level is sum_s theta gamma")

# --- 2. V = 1/N exactly, and the closed form built from it -------------------
pan = variety_panel(data)
V = _V_by_buyer(pan, buyers)
for s in range(S):
    assert np.allclose(V.loc[s].to_numpy(), 1.0 / N_HAT[s], atol=1e-12)
b0 = bs[bs["regime"] == "Both forces"]
assert np.allclose(b0["sd"], np.sqrt(b0["V"] * b0["p"] * (1 - b0["p"])), atol=1e-12)
# the SECTOR SUM squares the weights -- the one step the plan flags, gated directly
sq = (b0.assign(c=b0["theta"] ** 2 * b0["sd"] ** 2)
        .groupby("ze2010_downstream")["c"].sum().reindex(buyers).to_numpy())
assert np.allclose(tab.loc["Both forces", "sd_closed"].to_numpy() ** 2, sq, atol=1e-14)
# and the scale-free reading is EXACT and monotone: cv = sd/p = sqrt(V(1-p)/p),
# strictly decreasing in p over the whole range, so it has no interior maximum where
# the bar itself does.
m0 = b0[b0["p"] > 0]
assert np.allclose(m0["sd"] / m0["p"], np.sqrt(m0["V"] * (1 - m0["p"]) / m0["p"]),
                   atol=1e-12)
# monotone in p at a FIXED V, so the sweep is taken within one sector
o = m0[m0["sector"] == 2].sort_values("p")
assert len(o) >= 2
assert (np.diff((o["sd"] / o["p"]).to_numpy()) <= 1e-12).all()
print("2 ok  V = 1/N_s exactly on the planted panel, sd_s = sqrt(V p(1-p)), and the "
      "sector aggregation adds the variances with SQUARED weights")

# --- 3. the measured sd against the closed form (two routes, no shared code) --
t = tab.loc["Both forces"]
assert float(t["n_draws"].iloc[0]) == B
ratio = (t["sd_draws"] / t["sd_closed"]).to_numpy()
assert np.isfinite(ratio).all()
assert abs(np.median(ratio) - 1.0) < 0.10, ratio
assert np.abs(ratio - 1.0).max() < 0.35, ratio
# E[omega] = gamma is EXACT, so the realised mean must sit on the level within its own
# simulation error -- the check that the panel and the structural network are the same
# regime, which no other column would reveal.
assert np.abs(t["z"].to_numpy()).max() < 4.0, t["z"].to_numpy()
print(f"3 ok  measured vs closed-form sd: median ratio {np.median(ratio):.3f}, max "
      f"deviation {np.abs(ratio - 1).max():.3f} at B = {B}; max |z| on the level "
      f"{np.abs(t['z']).max():.2f}")

# --- 4. the two extremes, where the bar must VANISH --------------------------
# p(1-p) is zero at both ends: a radius covering the country makes every euro local
# (p = 1) and a radius below the shortest link makes none of it (p = 0). Both must
# give a bar of exactly zero on BOTH routes, the measured one included.
# The tolerance at p = 1 is sqrt(eps), not eps, and that is the arithmetic rather
# than a fudge: `p` is a column sum of `rho` and lands one ulp off one, so `p(1-p)` is
# a rounding-scale quantity whose SQUARE ROOT is the reported bar.
wide = local_share_dispersion(data, radius_km=1e6, verbose=False)
assert np.allclose(wide["local"], 1.0, atol=1e-12)
assert np.abs(wide["sd_closed"]).max() < 1e-6
assert np.allclose(wide["sd_draws"].dropna(), 0.0, atol=1e-12)
assert np.allclose(wide["local_draws"].dropna(), 1.0, atol=1e-12)
tight = local_share_dispersion(data, radius_km=1.0, verbose=False)
assert np.allclose(tight["local"], 0.0, atol=1e-12)
assert np.allclose(tight["sd_closed"], 0.0, atol=1e-12)
assert np.allclose(tight["sd_draws"].dropna(), 0.0, atol=1e-12)
# and in between the bar is strictly positive
mid = local_share_dispersion(data, radius_km=300.0, verbose=False)
assert (mid["sd_closed"] > 0).all() and (mid["sd_draws"] > 0).all()
assert ((mid["local"] > 0) & (mid["local"] < 1)).all()
print("4 ok  the bar vanishes exactly at p = 0 and p = 1 on both routes and is "
      "strictly positive in between")

# --- 5. p(1-p) is maximal at one half, ON THE DRAWS ---------------------------
# The closed form has this shape by construction, so asserting it there would be a
# tautology given gate 2. The claim with content is that the MEASURED dispersion has
# it too: sweep the radius to move `p` across its whole range and check the empirical
# sd across replications tracks sqrt(V p(1-p)) and peaks in the middle.
rows = []
for d in np.arange(60.0, 820.0, 20.0):
    b = local_share_dispersion(data, regimes={"Both forces": dict()}, panel=pan,
                               radius_km=float(d), verbose=False).attrs["by_sector"]
    rows.append(b[b["sector"] == 2][["p", "sd", "sd_draws"]])
sw = pd.concat(rows, ignore_index=True)
sw = sw[(sw["p"] > 1e-9) & (sw["p"] < 1 - 1e-9)]
assert len(sw) > 40 and sw["p"].min() < 0.15 and sw["p"].max() > 0.85, sw["p"].describe()
pred = np.sqrt((1.0 / N_HAT[2]) * sw["p"] * (1.0 - sw["p"]))
err = (sw["sd_draws"] / pred - 1.0).abs()
assert err.median() < 0.10 and err.quantile(0.95) < 0.30, err.describe()
cut = pd.cut(sw["p"], [0, .2, .4, .6, .8, 1.0])
m = sw.groupby(cut, observed=True)["sd_draws"].mean()
assert m.idxmax().left <= 0.5 <= m.idxmax().right, m
# strictly up to the peak and strictly down after it -- the shape claim itself,
# not just the location of the maximum
kmax = int(np.argmax(m.to_numpy()))
assert (np.diff(m.to_numpy()[:kmax + 1]) > 0).all(), m
assert (np.diff(m.to_numpy()[kmax:]) < 0).all(), m
assert m.iloc[0] < 0.8 * m.max() and m.iloc[-1] < 0.8 * m.max(), m
print(f"5 ok  the MEASURED dispersion tracks sqrt(V p(1-p)) to a median "
      f"{100 * err.median():.1f}% and peaks in the bin containing p = 1/2, falling to "
      f"{100 * m.iloc[0] / m.max():.0f}% and {100 * m.iloc[-1] / m.max():.0f}% of its "
      "peak at the two ends")

# --- 6. sd_fixed_V isolates gamma, and V is NOT invariant ---------------------
base = tab.loc["Both forces"]
assert np.allclose(base["sd_closed"], base["sd_fixed_V"], atol=1e-14)
moved = [lab for lab in CF_REGIMES
         if not np.allclose(tab.loc[lab, "sd_closed"], tab.loc[lab, "sd_fixed_V"],
                            rtol=1e-6)]
assert moved, "V came back identical under every regime — the column says nothing"
print(f"6 ok  sd_fixed_V equals sd_closed on the baseline by construction and differs "
      f"under {moved} — V moves with the regime, so the two columns are not one")

# --- 7. a sector with no varieties BLANKS the draws rather than reweighting ---
# pandas sums a missing sector as zero, which would silently measure a different input
# mix from the closed form. The whole regime's measured column must go missing instead.
part = {s: blk for s, blk in pan.items() if s != 1}
cut = local_share_dispersion(data, regimes={"Both forces": dict()}, panel=part,
                             radius_km=None, verbose=False)
assert cut["sd_draws"].isna().all() and cut["local_draws"].isna().all()
assert np.allclose(cut["local"], tab.loc["Both forces", "local"], atol=1e-12)
print("7 ok  a sector absent from the panel blanks the measured dispersion and leaves "
      "the closed-form level untouched")

# --- 8. the report, and the MECHANISM behind the plan's conjecture -------------
rep = local_share_report(tab, verbose=False)
assert list(rep.index) == ["Both forces"] + [r for r in CF_REGIMES if r != "Both forces"]
assert abs(rep.loc["Both forces", "d local (median)"]) < 1e-15
assert abs(rep.loc["Both forces", "d sd (median)"]) < 1e-15
for lab in CF_REGIMES:
    assert (rep.loc[lab, "n: point down, bar UP"]
            + rep.loc[lab, "n: both down"]) <= len(buyers)

# The conjecture is that cutting a force can LOWER the point and RAISE the bar. At a
# fixed V that is possible if and only if `p` moves TOWARDS one half, since
# sqrt(p(1-p)) is increasing below it and decreasing above. Gated exactly, cell by
# cell, where the identity is exact and nothing has been aggregated.
key = ["sector", "ze2010_downstream"]
b_base = bs[bs["regime"] == "Both forces"].set_index(key)
n_comp = n_cross = 0
for lab in CF_REGIMES:
    if lab == "Both forces":
        continue
    b_l = bs[bs["regime"] == lab].set_index(key).reindex(b_base.index)
    pb, pl, Vb = b_base["p"], b_l["p"], b_base["V"]
    sdb, sdl = np.sqrt(Vb * pb * (1 - pb)), np.sqrt(Vb * pl * (1 - pl))
    closer = (pb - 0.5).abs() - (pl - 0.5).abs()          # > 0: p moved towards 1/2
    ok = np.sign(np.round(sdl - sdb, 12)) == np.sign(np.round(closer, 12))
    assert ok.all(), b_base.assign(pl=pl, d_sd=sdl - sdb, closer=closer)[~ok]
    comp = (pl < pb) & (sdl > sdb)                        # point down, bar UP
    n_comp += int(comp.sum())
    assert (closer[comp] > 0).all()
    n_cross += int(((pb < 0.5) != (pl < 0.5)).sum())
assert n_comp > 0 and (pb > 0.5).any() and (pb < 0.5).any(), (n_comp, pb.describe())
print(f"8 ok  the report keeps the declared regime order; cell by cell the bar grows "
      f"if and only if p moves towards one half ({n_comp} cells lose level and gain "
      f"dispersion, {n_cross} cross 1/2), which is the plan's compensation and the "
      "condition it needs")

# --- 9. the figure ------------------------------------------------------------
ax = plot_local_share_dispersion(tab)
assert len(ax.containers) == len(CF_REGIMES)
for c in ax.containers:
    assert len(c[0].get_xdata()) == len(buyers)
order = [t.get_text() for t in ax.get_yticklabels()]
want = (tab.loc["Both forces", "local"].sort_values().index.to_numpy())
assert order == [f"Z{b}" for b in want], (order, want)
assert ax.get_xlim()[0] == 0.0                      # a share has an honest zero
# the bar IS k sigma, in data units, not a decoration -- asked for explicitly, since
# the DEFAULT band is now the 10-90 quantile range
k_test = 2.0
ax2 = plot_local_share_dispersion(tab, band="draws", k=k_test)
seg = ax2.containers[0][2][0].get_segments()
base_sorted = tab.loc["Both forces"].reindex(want)
drawn = np.array([s[1][0] - s[0][0] for s in seg]) / 2.0
assert np.allclose(drawn, k_test * base_sorted["sd_draws"].to_numpy(), atol=1e-9)
# the vertical layout the plan describes, and the per-sector view
axv = plot_local_share_dispersion(tab, orientation="v")
assert len(axv.get_xticklabels()) == len(buyers) and axv.get_ylim()[0] == 0.0
axs = plot_local_share_dispersion(tab, sector=2)
assert len(axs.containers) == len(CF_REGIMES)
for bad, kw in ((ValueError, dict(orientation="diagonal")),
                (ValueError, dict(band="guess")), (KeyError, dict(sector=99)),
                (KeyError, dict(baseline="nope"))):
    try:
        plot_local_share_dispersion(tab, **kw); raise AssertionError(f"no raise: {kw}")
    except bad:
        pass
plt.close("all")
print("9 ok  one error-bar series per regime, buyers ordered by the baseline level, an "
      "honest zero, the bar equal to k sigma in data units, both orientations and the "
      "per-sector view")

# --- 10. the quantile band and the counting regime ---------------------------
# The band the figure draws is the empirical 10-90 range, not a symmetric sd. Ordering
# is an identity of the quantiles; containment of the MEAN is not, and that is the
# point: where the law is discrete and skewed the point can sit off-centre in its own
# band, which a +/- sd cannot represent.
assert (tab["q10"] <= tab["med_draws"] + 1e-12).all()
assert (tab["med_draws"] <= tab["q90"] + 1e-12).all()
assert ((tab["q90"] - tab["q10"]) > 0).all()
assert (tab["band_skew"].abs() <= 1 + 1e-12).all()
# p x N_eff is the count of effective varieties landing locally, and 1/V its factor
assert np.allclose(tab["n_eff_var"], 1.0 / tab["V"], atol=1e-12)
assert np.allclose(tab["p_n_eff"], tab["local"] * tab["n_eff_var"], atol=1e-12)
# on the planted panel V = 1/N_s exactly, so N_eff is the spend-weighted harmonic-free
# average of the variety counts and p x N_eff is a genuine expected count
base_c = tab.loc["Both forces"]
assert (base_c["n_eff_var"] > 0).all() and np.isfinite(base_c["p_n_eff"]).all()
# the SKEW is not an artefact of the estimator: a radius where p is near one half gives
# a near-symmetric band, one where p is small gives a right-skewed one. Gated as an
# ordering, so it cannot pass by luck on a single configuration.
lowp = local_share_dispersion(data, regimes={"Both forces": dict()}, panel=pan,
                              radius_km=120.0, verbose=False)
midp = local_share_dispersion(data, regimes={"Both forces": dict()}, panel=pan,
                              radius_km=350.0, verbose=False)
assert lowp["local"].median() < midp["local"].median()
assert lowp["band_skew"].median() > midp["band_skew"].median(), \
    (lowp["band_skew"].median(), midp["band_skew"].median())
# the DEFAULT radius is the section's 200 km, not the own zone
assert LOCAL_SHARE_RADIUS_KM == AMPLIFICATION_RADII[-1]
dflt = local_share_dispersion(data, regimes={"Both forces": dict()}, panel=pan,
                              verbose=False)
at200 = local_share_dispersion(data, regimes={"Both forces": dict()}, panel=pan,
                               radius_km=float(AMPLIFICATION_RADII[-1]), verbose=False)
assert np.allclose(dflt["local"], at200["local"], atol=1e-12)
# and the figure draws THAT band: the arms are q10/q90 about the point, clipped at zero
axq = plot_local_share_dispersion(tab)
want = tab.loc["Both forces"].reindex(
    tab.loc["Both forces", "local"].sort_values().index)
segq = axq.containers[0][2][0].get_segments()
lo_drawn = np.array([g[0][0] for g in segq])
hi_drawn = np.array([g[1][0] for g in segq])
assert np.allclose(lo_drawn, np.minimum(want["q10"], want["local"]), atol=1e-9)
assert np.allclose(hi_drawn, np.maximum(want["q90"], want["local"]), atol=1e-9)
plt.close("all")
print(f"10 ok  the drawn band IS the 10-90 range of the draws about the point (skew "
      f"{tab.loc['Both forces', 'band_skew'].median():+.2f} at the default 200 km, "
      f"rising to {lowp['band_skew'].median():+.2f} at a radius where p is smaller), "
      "and p x N_eff = local/V")

print("\nall gates pass")
