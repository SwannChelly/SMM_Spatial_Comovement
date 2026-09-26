"""Gate for Test 3 bis of the amplification section: the alignment in KILOMETRES,
`Cov_rho(log T, d)` -- the rate the `Equal comparative advantage` counterfactual integrates.

This test exists because the section had none. `alignment_frame` was deleted with the
comparative-advantage section and Test 3 bis kept calling it, so the cell raised
`NameError` at run time with nothing to catch it. Gate 1 below is that regression: the
cell must define its own `alignment_frame` and `_buyer_weights`, and the copy must agree
with Test 8's."""
import ast, glob, os, sys, numpy as np, pandas as pd, matplotlib, warnings
matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _nbmod import install

# --- 1. ONE definition, not two (the regression this file exists for) ---------
# `alignment_frame` and `_buyer_weights` used to exist in two byte-identical copies --
# one per notebook section, because each section had to be runnable on its own -- and
# this gate checked that the copies had not drifted apart. The libraries remove the
# duplication rather than police it, so what is checked now is that there IS only one of
# each, across all four modules. That is the stronger statement: two copies cannot drift
# if two copies cannot exist.
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
_MODS = ["utils.py", "report_lib.py", "diffusion_lib.py", "granular_lib.py"]
_where = {}
for _m in _MODS:
    for _n in ast.parse(open(os.path.join(_ROOT, _m), encoding="utf-8").read()).body:
        if isinstance(_n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            _where.setdefault(_n.name, []).append(_m)
for _nm in ("alignment_frame", "alignment_covariance", "alignment_covariance_by_buyer",
            "_buyer_weights", "plot_alignment_covariance"):
    assert _nm in _where, f"no module defines {_nm}"
    assert len(_where[_nm]) == 1, f"{_nm} is defined in {_where[_nm]}"
_dupes = {k: v for k, v in _where.items() if len(v) > 1}
assert not _dupes, f"defined in more than one module: {_dupes}"
print(f"1 ok  one definition each, across {len(_MODS)} modules and "
      f"{len(_where)} functions -- the two copies this gate used to police are gone")

# --- the planted economy ------------------------------------------------------
S, R, BUY = 2, 10, 4
rng = np.random.default_rng(7)
CELLS = {0: np.arange(R), 1: np.arange(2, R)}
D = rng.uniform(30, 700, (R, BUY))
# sector 0: T ALIGNED with proximity (productive cells sit near the buyers) -> the
# covariance must come out negative. sector 1: T orthogonal to the geometry -> ~zero.
T = {0: np.exp(-0.004 * D.mean(axis=1)[CELLS[0]]),
     1: np.exp(rng.normal(0, .6, CELLS[1].size))}
AREAS = {s: c % BUY for s, c in CELLS.items()}
AA = [f"B{j}" for j in range(BUY)]
DOWN = np.arange(1, BUY + 1)
PI_R = np.array([0.4, 0.3, 0.2, 0.1])


def _rho(s):
    psi = T[s][:, None] * D[CELLS[s]] ** (-1.3 * 0.4)
    return psi / psi.sum(axis=0, keepdims=True)


def sourcing_geometry(data, alpha=None, equalise_T=False):
    return {"by_sector": {s: {"cells": CELLS[s], "areas": AREAS[s], "T_cell": T[s],
                              "distance": D[CELLS[s]], "rho": _rho(s)}
                          for s in range(S)},
            "alpha": 0.4, "theta": 1.3, "downstream": DOWN}


def aa_display_names(data): return AA
def _region_labels(data):
    return pd.DataFrame({"index": np.arange(1, R + 1),
                         "ze2010_name": [f"Z{i}" for i in range(1, R + 1)]})
def _parquet_sector_index(data, sup): return sup["A129"].to_numpy().astype(int) - 1
def _despine(ax=None):
    for side in ("top", "right"):
        (ax or plt.gca()).spines[side].set_visible(False)
def get_figsize(wf=1.0, hf=0.5): return [7.0 * wf, 7.0 * wf * hf]

# spending is deliberately LOPSIDED across sectors, so a spend-weighted buyer average
# cannot coincide with a plain one and gate 5 is not vacuous
SPEND = {(int(z), s): float(v) for z in DOWN for s, v in
         ((0, 9.0 + z), (1, 1.0))}
sup = pd.DataFrame([(int(z), s + 1, SPEND[(int(z), s)])
                    for z in DOWN for s in range(S)],
                   columns=["ze2010_downstream", "A129", "share"])
data = {"S": S, "R": R, "sector_names": ["A", "B"], "emp_pi_r": PI_R,
        "suppliers": sup, "industry": "test", "folder": "x", "step_dir": "step3"}
sim_color = (.2, .4, .7); toulouse_color = (.5, .2, .1); reference_color = (.3, .3, .3)

# The library, with the fixture installed where its functions resolve their globals.
import utils, granular_lib
install(globals(), [utils, granular_lib])

# --- 2. the panel ------------------------------------------------------------
fr = alignment_frame(data)
assert len(fr) == sum(CELLS[s].size for s in range(S)) * BUY
assert abs(fr["weight"].sum() - 1.0) < 1e-12          # a probability over the panel
g = fr.groupby("group", sort=False)
assert np.allclose(g["rho"].sum().to_numpy(), 1.0, atol=1e-12)
assert g.ngroups == S * BUY
# the distances ARE the geometry's, cell by cell, not a re-derivation
for s in range(S):
    for r in range(BUY):
        sub = fr[fr["group"] == f"{data['sector_names'][s]}|{r}"]
        assert np.allclose(sub["distance_km"].to_numpy(), D[CELLS[s], r], atol=1e-12)
        assert np.allclose(sub["log_T"].to_numpy(), np.log(T[s]), atol=1e-12)
one = alignment_frame(data, buyer=2)
assert one["buyer"].nunique() == 1 and len(one) == len(fr) // BUY
print(f"2 ok  the panel is {len(fr)} = (cells x buyers) rows, rho sums to one per "
      "(sector, buyer), weight to one over the panel, and the distances are the "
      "geometry's own")

# --- 3. the covariance, against an independent recomputation -----------------
cov = alignment_covariance(data)
assert len(cov) == S * BUY
for _, row in cov.iterrows():
    s = data["sector_names"].index(row["sector"]); r = AA.index(row["buyer"])
    p = _rho(s)[:, r]; p = p / p.sum()
    x, d = np.log(T[s]), D[CELLS[s], r]
    # RAW-moment route: E[xd] - E[x]E[d], algebraically distinct from the centred form
    # the function evaluates, so a sign or weighting slip cannot cancel out
    assert abs(row["cov_T_d_km"] - ((p * x * d).sum() - (p * x).sum() * (p * d).sum())) < 1e-9
    assert abs(row["d_rs_km"] - float(p @ d)) < 1e-12
# Cov(log d, d) is a covariance between two INCREASING functions of the same variable,
# so it is non-negative by Chebyshev's association inequality -- the column is a check,
# not a finding, and the docstring says so.
assert (cov["cov_d_d_km"] > 0).all(), cov["cov_d_d_km"].to_numpy()
# the two weightings answer different questions and must not silently coincide
assert not np.allclose(cov["cov_T_d_km"],
                       alignment_covariance(data, weights="cell")["cov_T_d_km"])
try:
    alignment_covariance(data, weights="equal"); raise AssertionError("no raise")
except ValueError:
    pass
print("3 ok  the covariance reproduces a raw-moment recomputation to 1e-9, "
      "Cov(log d, d) > 0 everywhere, and the two weightings differ")

# --- 4. the economics: an ALIGNED geometry is negative, an orthogonal one is not
al = cov[cov["sector"] == "A"]["cov_T_d_km"].to_numpy()
orth = cov[cov["sector"] == "B"]["cov_T_d_km"].to_numpy()
# The planted T tracks MEAN proximity across buyers, so the sign is asserted on the
# median and not buyer by buyer: an individual buyer can still face a positive
# covariance, which is not a defect of the fixture but the aerospace fact the section
# reports (the average alignment is weak because it points both ways across buyers).
assert np.median(al) < 0 and (al < 0).mean() >= 0.5, al
assert np.median(al) < np.median(orth), (al, orth)
assert np.abs(np.median(orth)) < np.abs(np.median(al)), (orth, al)
print(f"4 ok  the planted aligned sector has a negative median ({np.median(al):+.0f} km, "
      f"{100 * (al < 0).mean():.0f}% of buyers negative) against "
      f"{np.median(orth):+.0f} km for the orthogonal one — the sign the counterfactual "
      "integrates, and the per-buyer flip the aggregate hides")

# --- 5. the buyer aggregation ------------------------------------------------
by = alignment_covariance_by_buyer(data, cov=cov)
assert len(by) == BUY and set(by.index) == set(AA)
for name in AA:
    r = AA.index(name); z = int(DOWN[r])
    sub = cov[cov["buyer"] == name]
    w = np.array([SPEND[(z, data["sector_names"].index(s))] for s in sub["sector"]])
    assert abs(by.loc[name, "cov_T_d_km"]
               - float(np.average(sub["cov_T_d_km"], weights=w))) < 1e-12
# SPEND is lopsided, so the spend weighting must move the number off a plain mean
plain = cov.groupby("buyer")["cov_T_d_km"].mean().reindex(by.index)
assert not np.allclose(by["cov_T_d_km"], plain, rtol=1e-3)
# no parquet -> raise rather than fall back on equal sector weights, which would report
# a different statistic under the same name
try:
    alignment_covariance_by_buyer({**data, "suppliers": None}, cov=cov)
    raise AssertionError("no raise")
except FileNotFoundError:
    pass
# a parquet whose region index does not line up must be NAMED, not silently zero-weighted
bad = {**data, "suppliers": sup.assign(ze2010_downstream=sup["ze2010_downstream"] + 900)}
try:
    alignment_covariance_by_buyer(bad, cov=cov); raise AssertionError("no raise")
except ValueError:
    pass
print("5 ok  the buyer aggregation is the spend-weighted mean of its sectors, differs "
      "from a plain mean under lopsided spending, and refuses both a missing parquet "
      "and a non-matching region index")

# --- 6. the figure ------------------------------------------------------------
sets = [("Ind 1", data), ("Ind 2", data)]
ax, frs = plot_alignment_covariance(sets)
assert set(frs) == {"Ind 1", "Ind 2"}
assert len(ax.collections) == 2                       # one filled density per industry
assert len([ln for ln in ax.lines if ln.get_linestyle() == ":"]) == 2   # the medians
assert any(abs(ln.get_xdata()[0]) < 1e-12 for ln in ax.lines
           if len(ln.get_xdata()) == 2)               # the zero rule
assert ax.get_ylim()[0] == 0.0                        # a density has an honest zero
# ONE bin grid for both industries, else the overlay compares nothing
grids = [np.unique(np.round(np.asarray(pp.get_xy())[:, 0], 9)) for pp in ax.patches]
assert len(grids) == 2 and np.array_equal(grids[0], grids[1]), grids
assert len(ax.get_legend().get_texts()) == 2
axb, frb = plot_alignment_covariance(sets, level="buyer")
assert all(len(f) == BUY for f in frb.values())
for bad_kw in (dict(level="sector"), dict(weights="equal")):
    try:
        plot_alignment_covariance(sets, **bad_kw); raise AssertionError("no raise")
    except ValueError:
        pass
plt.close("all")
print("6 ok  one density and one median rule per industry on one panel, an honest zero, "
      "the buyer level, and a refusal for an unknown level or weighting")

print("\nall gates pass")
