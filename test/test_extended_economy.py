"""Gate for the extended parameter set and the economy it defines.

`theta+ = (Omega_L, Omega_s, A, alpha, T, N)` plus the fixed calibration, and
`simulate_economy` is the forward map from it to a realised finite-variety economy. Under
that design the counterfactuals are produced by the SAME code as the estimated economy, so
what has to be gated is (a) that the map closes -- the two CES identities, the input mix and
`D_r` -- against arithmetic recomputed here from the primitives and sharing no code with the
function, (b) that it is a deterministic function of `theta+` and the draws, (c) that
averaging it returns the closed-form `rho` the section's structural half is built on, which
is what ties the simulator to `sourcing_geometry`, and (d) that the two exact controls hold:
`alpha = 0` makes every buyer pick the same winner for a variety, and `N` does not move
across regimes.

The fixture is the planted economy of `test_concentration_identity.py` with a HAND-built
`theta+` -- the point of the exercise being that `theta+` is a plain object, so it is written
down rather than read from a run tree that does not exist in this environment.
"""
import json, os, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

_NB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "diffusion.ipynb")
nb = json.load(open(_NB, encoding="utf-8"))
_ANCHORS = ("def simulate_economy(",)
_cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"
          and any(a in "".join(c["source"]) for a in _ANCHORS)]
assert len(_cells) == 1, f"{len(_cells)} cells define simulate_economy"
code = _cells[0]

S, R, THETA, ALPHA = 3, 12, 1.3, 0.4
rng = np.random.default_rng(3)
CELL_MASK = np.zeros((S, R), bool)
for s in range(S):
    CELL_MASK[s, rng.choice(R, 8, replace=False)] = True
buyers = np.array([1, 2, 3, 4, 5])
D = rng.uniform(20, 600, (R, R)); np.fill_diagonal(D, 10.)
D = (D + D.T) / 2
Tcell = {s: rng.lognormal(0, .6, size=int(CELL_MASK[s].sum())) for s in range(S)}
N_HAT = np.array([4, 10, 30])


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


NU_S_DEFAULT, NU_ACROSS_DEFAULT, LAMBDA_DEFAULT = 1.5, 0.2, 0.5
THETA_DEFAULT = 1.0
data = {"S": S, "R": R, "CELL_MASK": CELL_MASK, "post_hoc_N_hat": N_HAT,
        "folder": "x", "step_dir": "step3"}
exec(code, globals())

# theta+ written down by hand: this IS the object, so nothing is read from a run tree.
XP = {"Omega_L": 0.31, "Omega_s": np.array([0.2, 0.3, 0.5]),
      "A": np.array([1.0, 1.4, 0.8, 1.1, 0.9]), "alpha": np.array([ALPHA]),
      "T": None, "N": N_HAT.copy(), "theta": THETA, "theta_source": "fixture",
      "nu_s": np.full(S, NU_S_DEFAULT), "nu": NU_ACROSS_DEFAULT, "lam": LAMBDA_DEFAULT,
      "epsilon": -16.0, "delta": np.ones(buyers.size), "wage": np.ones(R)}
B = 40

# --- 1. the draw-matrix column order is Julia's, derived not assumed -----------------
gs, gr = _good_order(data)
want = [(s, r) for r in range(R) for s in range(S) if CELL_MASK[s, r]]   # column-major
assert list(zip(gs.tolist(), gr.tolist())) == want, (list(zip(gs, gr))[:6], want[:6])
assert gs.size == int(CELL_MASK.sum())
# a C-order read would give sector-outer, which is the trap
wrong = [(s, r) for s in range(S) for r in range(R) if CELL_MASK[s, r]]
assert want != wrong, "the fixture cannot separate the two orders"
print(f"1 ok  the good index runs region-outer/sector-inner as Julia's column-major "
      f"findall does ({gs.size} cells), which is what aligns the draw columns")

# --- 2. the forward map closes ---------------------------------------------------------
econ = simulate_economy(data, XP, n_rep=B, seed=11)
res = economy_identities(econ, XP, verbose=False)
assert max(res.values()) < 1e-12, res
assert sorted(econ) == list(range(S))          # only sector keys, so panel.items() is safe
for s in range(S):
    blk = econ[s]
    assert blk["v"].shape == (B * N_HAT[s], buyers.size)
    assert blk["exp_val"].shape == blk["v"].shape
    assert np.array_equal(blk["buyers"], buyers)
print("2 ok  the two CES identities, the input mix and D_r all close to machine "
      f"precision (worst {max(res.values()):.1e}), and the panel carries sector keys only")

# --- 3. the value block against a HAND recomputation -----------------------------------
# P_sr is taken from the economy, everything after it is rebuilt here from theta+ alone.
V = econ.value
P_sr, P_r = V["P_sr"], V["P_r"]
h_P_r = (XP["Omega_s"][None, :, None] * P_sr ** (1 - XP["nu"])).sum(1) ** (1 / (1 - XP["nu"]))
assert np.allclose(P_r, h_P_r, rtol=0, atol=1e-13)
h_c = (XP["Omega_L"] * 1.0 + (1 - XP["Omega_L"]) * P_r ** (1 - XP["lam"])) ** (1 / (1 - XP["lam"]))
assert np.allclose(V["c_r"], h_c, atol=1e-13)
assert np.allclose(V["c_tilde_r"], h_c / XP["A"][None, :], atol=1e-13)
# D_r is the closed form the section quotes; it follows ONLY because the two CES
# identities collapse, so this single number fixes the whole expenditure chain
h_D = 1 + (1 - XP["Omega_L"]) * (P_r / V["c_r"]) ** (1 - XP["lam"])
assert np.allclose(V["D_r"], h_D, atol=1e-13)
h_mix = XP["Omega_s"][None, :, None] * (P_sr / P_r[:, None, :]) ** (1 - XP["nu"])
assert np.allclose(V["theta_rs"], h_mix, atol=1e-13)
mu = XP["epsilon"] / (XP["epsilon"] - 1)
h_p = V["c_tilde_r"] / mu
h_Pa = (h_p ** XP["epsilon"]).sum(1) ** (1 / XP["epsilon"])
assert np.allclose(V["P"], h_Pa, atol=1e-13)
assert np.allclose(V["Y_r"], h_p ** XP["epsilon"] * h_Pa[:, None] ** (-XP["epsilon"]), atol=1e-13)
assert (V["D_r"] > 1).all() and (V["Y_r"] > 0).all()
print(f"3 ok  P_r, c_r, c_tilde_r, D_r, theta_rs, P and Y_r reproduce a hand "
      f"recomputation from theta+ (D_r median {np.median(V['D_r']):.3f}) -- and D_r IS "
      "1 + (1-Omega_L)(P_r/c_r)^(1-lambda), the closed form")

# --- 4. deterministic in (theta+, draws) ----------------------------------------------
a1 = simulate_economy(data, XP, n_rep=5, seed=7)
a2 = simulate_economy(data, XP, n_rep=5, seed=7)
a3 = simulate_economy(data, XP, n_rep=5, seed=8)
for s in range(S):
    assert np.array_equal(a1[s]["winner"], a2[s]["winner"])
    assert np.array_equal(a1[s]["exp_val"], a2[s]["exp_val"])
assert np.allclose(a1.value["D_r"], a2.value["D_r"], atol=0)
assert not np.array_equal(a1[0]["winner"], a3[0]["winner"])
# and in theta+: a different head moves the value block and NOT the winners, because the
# head does not enter the Ricardian argmin at all
XP2 = dict(XP); XP2["Omega_L"] = 0.5
b1 = simulate_economy(data, XP2, n_rep=5, seed=7)
for s in range(S):
    assert np.array_equal(a1[s]["winner"], b1[s]["winner"])
assert not np.allclose(a1.value["D_r"], b1.value["D_r"])
print("4 ok  the economy is a deterministic function of (theta+, draws); the head moves "
      "the value block and leaves the winners untouched, which is why the Ricardian "
      "half needs only (alpha, T, N)")

# --- 5. averaging returns the closed-form rho ------------------------------------------
# E[omega] = gamma is exact, so the realised winner frequencies must converge to
# `sourcing_geometry`'s rho. This is what ties the simulator to the structural half.
big = simulate_economy(data, XP, n_rep=400, seed=5)
g = sourcing_geometry(data)
worst = 0.0
for s in range(S):
    cells, rho = g["by_sector"][s]["cells"], g["by_sector"][s]["rho"]
    W = big[s]["winner"].astype(int)
    n = W.shape[0]
    freq = np.stack([[(W[:, j] == c + 1).mean() for j in range(buyers.size)]
                     for c in cells])
    se = np.sqrt(np.maximum(rho * (1 - rho), 1e-12) / n)
    worst = max(worst, float(np.abs((freq - rho) / se).max()))
assert worst < 6.0, worst
print(f"5 ok  the realised winner frequencies converge to the closed-form rho: worst "
      f"|z| {worst:.2f} over {S * len(buyers) * 8} cells at 400 replications")

# --- 6. the two exact controls, and N fixed across regimes -----------------------------
# alpha = 0 makes the cost ranking common to every buyer, so one variety has ONE winner
z = simulate_economy(data, XP, alpha=0.0, n_rep=6, seed=3)
for s in range(S):
    W = z[s]["winner"]
    assert (W == W[:, [0]]).all(), f"sector {s}: alpha=0 did not give a common winner"
assert np.array_equal(z.meta["N"], XP["N"])
# equalise_T removes comparative advantage: the winner is then the nearest cell, variety
# by variety only through z, so the winner distribution must tilt towards short distances
e = simulate_economy(data, XP, equalise_T=True, n_rep=60, seed=4)
base = simulate_economy(data, XP, n_rep=60, seed=4)
for reg in (z, e, base):
    assert np.array_equal(reg.meta["N"], XP["N"]), "N moved across regimes"
d0 = g["by_sector"][0]["distance"]
cells0 = g["by_sector"][0]["cells"]
pos = {c + 1: i for i, c in enumerate(cells0)}
mean_d = lambda reg: np.mean([d0[pos[w], j] for j in range(buyers.size)
                             for w in reg[0]["winner"][:, j].astype(int)])
assert mean_d(e) > mean_d(base), (mean_d(e), mean_d(base))
print(f"6 ok  alpha=0 gives one winner per variety for every buyer (the exact control), "
      f"equalising T sends the euro further ({mean_d(base):.0f} -> {mean_d(e):.0f} km), "
      "and N is held at theta+'s value in every regime")

# --- 7. the supplied-draw path, and what it refuses ------------------------------------
# Julia hands one (N_max, n_good) matrix per replication. Built to match the internal
# draws, the two paths must agree to the bit -- that is the shape of the cross-language
# gate, with Julia's own matrix in place of this one.
n_good, N_max = gs.size, int(N_HAT.max())
rng2 = np.random.default_rng(11)
mats, want_u = [], []
for b in range(3):
    M = np.full((N_max, n_good), 0.5)
    per = {}
    for s in range(S):
        cells = np.flatnonzero(CELL_MASK[s]); N = int(N_HAT[s])
        uu = rng2.random((N, cells.size)); per[s] = uu
        cols = np.flatnonzero(gs == s)
        order = np.argsort(gr[cols])
        M[:N, cols[order]] = uu
    mats.append(M); want_u.append(per)
sup_econ = simulate_economy(data, XP, u=mats)
assert sup_econ.meta["draws"] == "julia" and sup_econ.meta["n_rep"] == 3
# reproduce it here from `want_u` with no reference to the function's internals
for s in range(S):
    cells, N = np.flatnonzero(CELL_MASK[s]), int(N_HAT[s])
    logT = np.log(Tcell[s]); logd = np.log(np.maximum(D[np.ix_(cells, buyers - 1)], 1.0))
    Wh = []
    for b in range(3):
        logz = logT[None, :] / THETA - np.log(-np.log1p(-want_u[b][s])) / THETA
        lc = ALPHA * logd.T[None, :, :] - logz[:, None, :]
        Wh.append(cells[lc.argmin(axis=2)] + 1)
    assert np.array_equal(sup_econ[s]["winner"].astype(int), np.vstack(Wh))
for bad, kw, why in (
        (ValueError, dict(u=[np.full((N_max - 1, n_good), 0.3)]), "too few rows for N_hat"),
        (ValueError, dict(u=[np.full((N_max, n_good - 1), 0.3)]), "wrong good count"),
        (ValueError, dict(u=mats, n_rep=9), "fewer matrices than replications")):
    try:
        simulate_economy(data, XP, **kw); raise AssertionError(f"no raise: {why}")
    except bad:
        pass
print("7 ok  a supplied (N_max, n_good) draw matrix reproduces a hand recomputation "
      "exactly and is refused when its shape or count disagrees with N_hat and the good "
      "order -- which is the cross-language gate with Julia's own matrix substituted")

# --- 8. theta+ refuses the states that would be silently wrong -------------------------
try:
    simulate_economy(data, dict(XP, N=np.array([4, 0, 30])), n_rep=2)
    raise AssertionError("a sector with zero varieties was accepted")
except ValueError:
    pass
try:
    simulate_economy(data, dict(XP, A=XP["A"][:3]), n_rep=2)
    raise AssertionError("a mis-sized A was accepted")
except ValueError:
    pass
EMPTY = CELL_MASK.copy(); EMPTY[1] = False
try:
    simulate_economy({**data, "CELL_MASK": EMPTY}, XP, n_rep=2,
                     geom={"by_sector": {0: sourcing_geometry(data)["by_sector"][0],
                                         1: {"cells": np.array([], int),
                                             "T_cell": np.array([]),
                                             "distance": np.zeros((0, buyers.size))},
                                         2: sourcing_geometry(data)["by_sector"][2]},
                           "alpha": ALPHA, "theta": THETA, "downstream": buyers})
    raise AssertionError("an empty sector was accepted")
except ValueError:
    pass
print("8 ok  zero varieties, a mis-sized A and an empty sector are refused by name "
      "rather than returning an infinite price index")

# --- 9. the cross-language comparator, gated on its own index arithmetic ---------------
# Julia is not in this environment, so the comparator is gated against a parquet written
# FROM `simulate_economy` in Julia's own schema: that leaves exactly the plumbing under
# test -- the 1-based replication and variety columns, the buyer -> column map, the row
# index `rep*N + variety`, and the dropped zero-share rows -- which is where a comparator
# of this shape goes wrong. It says nothing about whether Julia agrees; only a run with
# `post_hoc_u.npy` on disk can say that.
U3 = np.stack(mats, axis=2)
rows = []
for s in range(S):
    N = int(N_HAT[s])
    for b in range(3):
        for rho in range(N):
            for j, rd in enumerate(buyers):
                rows.append((b + 1, int(rd), int(sup_econ[s]["winner"][b * N + rho, j]),
                             s + 1, rho + 1, float(sup_econ[s]["exp_val"][b * N + rho, j])))
jul = pd.DataFrame(rows, columns=["replication", "ze2010_downstream", "ze2010", "A129",
                                  "variety", "share"])
gsr = np.column_stack([gs + 1, gr + 1])
d9 = {**data, "post_hoc_u": U3, "post_hoc_good_sr": gsr, "suppliers": jul}
rep9 = check_against_julia(d9, XP, verbose=False)
assert rep9.attrs["agrees"], rep9
assert int(rep9["rows"].sum()) == len(jul)
assert int(rep9["winner_mismatch"].sum()) == 0 and rep9["max_rel_exp"].max() < 1e-12
# one flipped winner and one perturbed share must both be caught, and named separately
bad_w = jul.copy(); k = 17
bad_w.loc[k, "ze2010"] = int(1 + (bad_w.loc[k, "ze2010"] % R))
r_w = check_against_julia({**d9, "suppliers": bad_w}, XP, verbose=False)
assert not r_w.attrs["agrees"] and int(r_w["winner_mismatch"].sum()) == 1
bad_e = jul.copy(); bad_e.loc[k, "share"] *= 1.05
r_e = check_against_julia({**d9, "suppliers": bad_e}, XP, verbose=False)
assert not r_e.attrs["agrees"] and int(r_e["winner_mismatch"].sum()) == 0
assert r_e["max_rel_exp"].max() > 0.04
# a column map that disagrees with Julia is refused BEFORE any number is compared
try:
    check_against_julia({**d9, "post_hoc_good_sr": gsr[::-1].copy()}, XP, verbose=False)
    raise AssertionError("a permuted column map was accepted")
except AssertionError as e:
    assert "column map" in str(e), e
for bad, kw in ((FileNotFoundError, {"post_hoc_u": None}),
                (FileNotFoundError, {"suppliers": jul.drop(columns=["replication"])}),
                (ValueError, {"post_hoc_u": U3[:, :, 0]})):
    try:
        check_against_julia({**d9, **kw}, XP, verbose=False)
        raise AssertionError(f"no raise: {kw.keys()}")
    except bad:
        pass
print("9 ok  the comparator reproduces a parquet written in Julia's schema exactly, "
      "catches a single flipped winner and a 5% share perturbation SEPARATELY, refuses a "
      "column map that disagrees with Julia before comparing anything, and refuses a "
      "missing draw file rather than falling back to its own draws")

# --- 10. theta comes from load_parameters.jl, not from a notebook constant -------------
# The failure this closes: THETA_DEFAULT sat at 1.0 while the estimator ran at 1.768, and
# nothing compared them, so every theta*alpha-scaled object would have been computed at an
# effective distance elasticity the economy was never solved at. The parser makes the Julia
# file the single source of truth; the gate checks that it reads it, prefers it, and says
# so when stats.csv disagrees.
import re as _re
from pathlib import Path as _P
_root = _P(os.path.dirname(os.path.abspath(__file__))) / ".."
# `_theta_from_julia` and `model_theta` live in the LOADER cell, which this gate does not
# execute (it would pull the whole tree reader and overwrite the fixture's stubs). The two
# definitions are sliced out by text and run in their own namespace, so what is gated is
# the shipped source rather than a copy.
_lc = [c for c in nb["cells"] if c["cell_type"] == "code"
       and "def _theta_from_julia(" in "".join(c["source"])]
assert len(_lc) == 1, f"{len(_lc)} cells define _theta_from_julia"
_ls = "".join(_lc[0]["source"])
_seg = _ls[_ls.index("def _theta_from_julia("):_ls.index("def unpack_estimated_T(")]
_ns = {"Path": _P, "np": np, "THETA_DEFAULT": THETA_DEFAULT,
       "_read_named_value": lambda df, n: (float(df.loc[df["name"] == n, "value"].iloc[0])
                                           if (df["name"] == n).any() else None)}
exec(_seg, _ns)
_theta_from_julia, model_theta = _ns["_theta_from_julia"], _ns["model_theta"]
jl = _theta_from_julia(_root)
src = (_root / "load_parameters.jl").read_text()
want = float(_re.search(r"const\s+theta\s*=\s*\$\(\s*([0-9.eE+-]+)\s*\)", src).group(1))
assert jl == want, (jl, want)
# it must WIN over a disagreeing stats.csv, since that is the value the parquet carries
coefs = pd.DataFrame({"name": ["theta"], "value": [want * 2.0]})
assert model_theta({"coefs": coefs, "base": str(_root)}) == want
# with the file out of reach it falls back to stats.csv, and to THETA_DEFAULT with neither
import os as _os
_cwd = _os.getcwd(); _os.chdir("/tmp")
try:
    assert model_theta({"coefs": coefs, "base": "/nonexistent"}) == want * 2.0
    assert model_theta({"base": "/nonexistent"}) == float(THETA_DEFAULT)
finally:
    _os.chdir(_cwd)
print(f"10 ok  theta is read from load_parameters.jl ({want:g}) and wins over a "
      "disagreeing stats.csv, so the notebook cannot drift from the economy it reports on")

print("\nall gates pass")
