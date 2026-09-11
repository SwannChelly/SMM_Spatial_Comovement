"""Gate for the new concentration/commonality block: a planted economy whose answers
need no arithmetic, plus the identities the code claims."""
import json, numpy as np, pandas as pd, matplotlib, os
matplotlib.use("Agg"); import matplotlib.pyplot as plt

_NB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "diffusion.ipynb")
nb = json.load(open(_NB))
# located by CONTENT, not by index: a cell inserted above must not silently shift this
# gate onto another section's definitions.
code = next(''.join(c['source']) for c in nb['cells']
            if c['cell_type'] == 'code'
            and 'def sector_concentration(' in ''.join(c['source']))

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

# a parquet drawn FROM the geometry: N_hat_s varieties per (sector, buyer), each won by
# one cell with probability rho, equal expenditure shares -> the granularity formula is
# exact and the four-cell identity is checkable against theory.
geom = sourcing_geometry(None)
rows, B = [], 60
spend_s = np.array([1.0, 2.0, 3.0])
for b in range(B):
    for s in range(S):
        blk = geom["by_sector"][s]
        for j, rd in enumerate(buyers):
            k = rng.multinomial(int(N_HAT[s]), blk["rho"][:, j])
            for i, n in enumerate(k):
                if n:
                    rows.append((b, int(rd), int(blk["cells"][i]) + 1, s + 1,
                                 spend_s[s] * n / N_HAT[s]))
sup = pd.DataFrame(rows, columns=["replication", "ze2010_downstream", "ze2010",
                                  "A129", "share"])
data = {"S": S, "R": R, "CELL_MASK": CELL_MASK, "suppliers": sup,
        "sector_names": ["A", "B", "C"], "post_hoc_N_hat": N_HAT,
        "folder": "x", "step_dir": "step3"}

CF_REGIMES = {"Both forces": dict(), "Distance only": dict(equalise_T=True),
              "Comparative advantage only": dict(alpha=0.0)}
UNIFORM_REGIME = "Uniform benchmark"
sim_color = (.2, .4, .7); toulouse_color = (.5, .2, .1)
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
axes = plot_buyer_concentration({"X": by}, save_to="/tmp/x2.pdf")
print("7 ok  table and both figures render")

rep = concentration_report(data, industry="gate", verbose=False)
assert set(rep) == {"sector", "summary", "derivatives", "buyers", "granular",
                    "expected_check", "table", "zones"}
assert rep["granular"] is not None
print("8 ok  concentration_report wires every piece together")
