"""Gate: trade costs are modular in their parametrisation (power law or binned).

`utils.log_tau` is the one port of `build_tau` (model_CP.jl); every consumer reads log tau
off `sourcing_geometry` instead of writing `alpha * log d`. This gate checks the port
against a line-by-line transcription of the Julia loop, the knots against
`load_parameters.jl`'s formula, and then the properties every consumer relies on, on a
real (fixture) run tree with N_TAU = 4:

  1. `log_tau` == the Julia loop, for K = 1, 4, 5, below the first knot and beyond the last;
  2. continuity, and d log tau / d log d == `tau_elasticity` (the segment's alpha_k);
  3. homogeneity of degree one in the alpha vector; alpha = 0 (scalar) is tau == 1;
  4. `tau_log_knots` == log of the bin centres, the last closed at max_l min_r d_lr;
  5. equal alpha_k reproduce the power law's geometry and the SAME realised winners;
  6. `No trade cost` sets every segment to zero, in the geometry and in the economy;
  7. the closed-form identities still close under binned tau (variance split, distance
     leverage path integral, concentration derivative against a central difference);
  8. the structural moment's pair elasticity is alpha_k(d_ir), and alpha under the power law.

Usage: python3 test/test_binned_tau.py
"""
import contextlib
import io
import os
import shutil
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "test")]
import matplotlib
matplotlib.use("Agg")
import utils                                                     # noqa: E402
import granular_lib                                              # noqa: E402
import report_lib                                                # noqa: E402
from _fixture_tree import build                                  # noqa: E402

rng = np.random.default_rng(3)


def julia_build_tau(alpha, ld, x):
    """model_CP.jl `build_tau`, transcribed loop for loop (1-based -> 0-based)."""
    K = len(alpha)
    if K == 1:
        return alpha[0] * ld
    lnt = alpha[0] * (min(ld, x[1]) - x[0])
    for k in range(1, K):
        if not ld > x[k]:
            break
        lnt += alpha[k] * ((ld if k == K - 1 else min(ld, x[k + 1])) - x[k])
    return lnt


# --- 1-3. the map itself --------------------------------------------------------------
for K, edges in ((1, None), (4, utils.TAU_BIN_EDGES[4]), (5, utils.TAU_BIN_EDGES[5])):
    a = np.sort(rng.uniform(0.1, 1.2, K))
    x = None if K == 1 else np.log(0.5 * (np.r_[edges, 420.0][:-1] + np.r_[edges, 420.0][1:]))
    ld = np.log(rng.uniform(0.5, 900.0, 400))                     # below x_0 and beyond x_K
    want = np.array([julia_build_tau(a, v, x) for v in ld])
    got = utils.log_tau(a, ld, x)
    assert np.allclose(got, want, atol=1e-13), K
    # slope == the segment elasticity, away from the knots
    h = 1e-6
    fd = (utils.log_tau(a, ld + h, x) - utils.log_tau(a, ld - h, x)) / (2 * h)
    far = np.ones(ld.size, bool) if x is None else (np.abs(ld[:, None] - x[None, :]) > 1e-4).all(1)
    assert np.allclose(fd[far], utils.tau_elasticity(a, ld, x)[far], atol=1e-6), K
    if x is not None:                                             # continuity at each knot
        for xk in x[1:-1]:
            assert abs(utils.log_tau(a, xk + 1e-12, x) - utils.log_tau(a, xk - 1e-12, x)) < 1e-9
    assert np.allclose(utils.log_tau(2.5 * a, ld, x), 2.5 * got)  # homogeneous of degree 1
    assert np.allclose(utils.log_tau(np.zeros(K), ld, x), 0.0)
print("1-3 ok  log_tau is the Julia loop (K = 1, 4, 5), continuous, slope alpha_k, "
      "homogeneous of degree one, and zero at alpha = 0")

# --- the run tree: the same fixture under the power law and binned ---------------------
A1 = 0.4
ALPHAS = (0.2, 0.3, 0.5, 0.8)
trees = []
try:
    T1, KW1 = build(alpha=(A1,)); trees.append(T1)
    TK, KWK = build(alpha=ALPHAS); trees.append(TK)
    TE, KWE = build(alpha=(A1,) * 4); trees.append(TE)       # binned, all segments equal
    load = lambda kw: utils.load_granular_data("test", mu=2, parts=("core", "geography"), **kw)
    d1, dK, dE = load(KW1), load(KWK), load(KWE)
    assert (d1["n_tau"], dK["n_tau"], dE["n_tau"]) == (1, 4, 4)

    # --- 4. knots ------------------------------------------------------------------
    D = np.load(dK["input_folder"] / "distances.npy")
    down = utils._downstream_ze_index(dK)
    d_max = D[:, down - 1].min(axis=1).max()
    e = np.r_[utils.TAU_BIN_EDGES[4], d_max]
    assert np.allclose(utils.tau_log_knots(dK), np.log(0.5 * (e[:-1] + e[1:])))
    assert utils.tau_log_knots(d1) is None
    print(f"4 ok  knots = bin centres {np.round(np.exp(utils.tau_log_knots(dK)), 1).tolist()} km, "
          f"the last bin closed at max_l min_r d = {d_max:.1f} km")

    # --- 5. equal alpha_k == the power law ------------------------------------------
    g1, gE = utils.sourcing_geometry(d1), utils.sourcing_geometry(dE)
    for s in g1["by_sector"]:
        assert np.allclose(g1["by_sector"][s]["rho"], gE["by_sector"][s]["rho"], atol=1e-12)
    assert np.isclose(g1["theta_alpha"], g1["theta"] * A1)
    assert np.isclose(gE["theta_alpha"], g1["theta_alpha"])
    # Equal alpha_k give tau = (d/dbar_0)^alpha: every French psi is scaled by the common
    # factor c = dbar_0^(theta*alpha), which cancels in rho but NOT against the foreign
    # competitor. In estimation Phi^F is profiled on the import share and absorbs c, so the
    # same economy has Phi^F * c under binned tau: compare at that Phi^F.
    c = np.exp(g1["theta"] * A1 * utils.tau_log_knots(dE)[0])
    with contextlib.redirect_stdout(io.StringIO()):
        xpE = utils.extended_parameters(dE, verbose=False)
    xpE["PhiF"] = np.asarray(xpE["PhiF"], dtype=float) * c
    for blk in gE["by_sector"].values():
        blk["PhiF_r"] = blk["PhiF_r"] * c
        blk["PhiF"] = blk["PhiF"] * c
    for s in g1["by_sector"]:
        assert np.allclose(g1["by_sector"][s]["f"], gE["by_sector"][s]["PhiF"]
                           / (gE["by_sector"][s]["phi_dom"] + gE["by_sector"][s]["PhiF"]),
                           atol=1e-12)
    with contextlib.redirect_stdout(io.StringIO()):
        e1 = utils.simulate_economy(d1, n_rep=5, seed=7)
        eE = utils.simulate_economy(dE, xpE, n_rep=5, seed=7, geom=gE)
    for s in e1:
        assert np.array_equal(e1[s]["winner"], eE[s]["winner"]), s
        assert np.allclose(e1[s]["v"], eE[s]["v"], atol=1e-12), s
    print("5 ok  equal alpha_k give the power law's rho (1e-12), theta_alpha, and -- with "
          "Phi^F re-expressed in binned-tau units (it absorbs dbar_0^(theta alpha)) -- the "
          "same import share, realised winners and within-sector value shares draw for draw")

    # --- 6. No trade cost --------------------------------------------------------------
    kw = utils.CF_REGIMES["No trade cost"]
    g0 = utils.sourcing_geometry(dK, **kw)
    assert np.array_equal(g0["alpha"], np.zeros(4))
    for blk in g0["by_sector"].values():
        assert np.allclose(blk["log_tau"], 0.0) and np.allclose(blk["elasticity"], 0.0)
    with contextlib.redirect_stdout(io.StringIO()):
        regs = utils.reporting_data("test", mu=2, n_rep=4, **KWK)
    assert set(regs) == set(utils.CF_REGIMES)
    assert np.array_equal(regs["No trade cost"]["economy"].meta["alpha"], np.zeros(4))
    assert np.array_equal(regs["Both forces"]["economy"].meta["alpha"], np.array(ALPHAS))
    print("6 ok  `No trade cost` sets all four alpha_k to zero (geometry and economy); "
          "every regime builds on the binned tree")

    # --- 7. identities under binned tau ---------------------------------------------
    gK = utils.sourcing_geometry(dK)
    with contextlib.redirect_stdout(io.StringIO()):
        vd = granular_lib.ca_variance_decomposition(dK)
    tot = vd["share_CA"] + vd["share_distance"] + vd["share_covariance"]
    assert np.allclose(tot, 1.0), tot
    # the distance pull is theta * log tau, which is theta*alpha*log d under the power law
    for s, blk in gK["by_sector"].items():
        if blk["cells"].size >= 2:
            x = gK["theta"] * blk["log_tau"]
            row = vd.loc[dK["sector_names"][s]]
            w = utils._buyer_weights(dK)
            assert np.isclose(row["sd_dist"], np.sqrt(np.var(x, axis=0) @ w))
    with contextlib.redirect_stdout(io.StringIO()):
        lev = granular_lib.ca_distance_leverage(dK, n_grid=801, reg=None)
    rel = np.max(np.abs(lev["check_identity"])) / max(np.abs(lev["delta_km"]).max(), 1e-12)
    assert rel < 1e-4, rel
    with contextlib.redirect_stdout(io.StringIO()):
        cd = granular_lib.concentration_derivatives(regs["Both forces"], verbose=True)
    assert cd.attrs["identity_residual"] < 1e-5, cd.attrs["identity_residual"]
    print(f"7 ok  variance split sums to one with theta*log tau as the distance term; "
          f"leverage path integral {rel:.1e}; dH/d(theta a_bar) vs central difference "
          f"{cd.attrs['identity_residual']:.1e}")

    # --- 8. the structural moment's pair elasticity -----------------------------------
    import pandas as pd
    ld = np.log(np.array([10.0, 60.0, 100.0, 150.0, 300.0]))   # segments split at 75/125/175
    panel = pd.DataFrame({"log_distance": ld})
    assert np.allclose(report_lib._pair_tau_elasticity(dK, panel), (ALPHAS[0],) + ALPHAS)
    assert np.allclose(report_lib._pair_tau_elasticity(d1, panel), A1)
    print("8 ok  the structural derivative uses alpha_k at the pair's own distance "
          f"({[ALPHAS[0]] + list(ALPHAS)} at 10/60/100/150/300 km), alpha under the power law")
finally:
    for t in trees:
        shutil.rmtree(t, ignore_errors=True)

print("\nall gates pass")
