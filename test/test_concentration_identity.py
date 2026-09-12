"""Gate for the new concentration/commonality block: a planted economy whose answers
need no arithmetic, plus the identities the code claims."""
import json, math, numpy as np, pandas as pd, matplotlib, os
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
assert set(rep) == {"sector", "summary", "derivatives", "buyers", "tail", "variety",
                    "cosourcing", "granular", "table", "zones"}
assert rep["table"] is not None and rep["variety"] is not None
assert rep["granular"] is not None
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
