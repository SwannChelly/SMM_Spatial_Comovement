"""
Gates for the sections added to `analysis_granular.ipynb`:
  1. Identification / sensitivity  — the N_s block re-attached to the Jacobian,
  2. Untargeted moment             — the PPML distance elasticity of comovement,
  3. Comparative advantage         — T against distance, within sector.

No Julia and no real data: each gate writes a synthetic run tree with the exact file
layout the loader expects and then EXECUTES THE NOTEBOOK'S OWN CODE CELLS against it,
so what is tested is the shipped artefact rather than a copy of it. Run with

    python test/test_analysis_granular_sections.py

Requires numpy, pandas, matplotlib, statsmodels, pyarrow (pyfixest optional).
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = Path(__file__).resolve().parent
NB_PATH = HERE.parent / "analysis_granular.ipynb"
TMP = HERE / "_granular_sections_tmp"
TMP.mkdir(exist_ok=True)


def notebook_namespace():
    """
    Every code cell of the notebook except the RUN cell, executed in order.

    Located by content rather than by index, so inserting a section above does not
    silently change what is being tested.
    """
    nb = json.load(open(NB_PATH))
    ns = {}
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if src.lstrip().startswith("%") or "RUN THE REPORTING" in src:
            continue
        exec(compile(src, f"<notebook cell {i}>", "exec"), ns)
    return ns


NS = notebook_namespace()


def gate_identification():
    """The N_s block embedded on the masked moment rows, and the thresholded figures."""
    S, n_AA, n_coef, n_tau, R_d = 3, 4, 4, 1, 4
    n_gam = 6
    moment_block_sizes = [1, S - 1, R_d - 1, n_coef, n_gam, S]
    param_block_sizes = [1, S - 1, R_d - 1, n_tau, n_gam]
    n_rows, n_par = sum(moment_block_sizes), sum(param_block_sizes)
    n_gb = n_coef + n_gam + S

    full_len = 1 + S + R_d + n_coef + S * n_AA + S
    mask = np.ones(full_len, dtype=bool)
    mask[1] = False                     # first industry share
    mask[1 + S] = False                 # first pi_r
    off_gam = 1 + S + R_d + n_coef
    gam_keep = np.zeros(S * n_AA, dtype=bool)
    gam_keep[:n_gam] = True
    mask[off_gam:off_gam + S * n_AA] = gam_keep
    assert mask.sum() == n_rows, (mask.sum(), n_rows)

    rng = np.random.default_rng(0)
    folder = TMP / "synthrun"
    (folder / "step3" / "inference").mkdir(parents=True, exist_ok=True)
    G_N = np.zeros((n_gb, S))
    for s in range(S):
        G_N[n_coef + n_gam + s, s] = -0.02 * (s + 1)     # diagonal on the G0 rows
    np.save(folder / "step3" / "inference" / "jacobian_N_s.npy", G_N)
    np.save(folder / "step3" / "inference" / "jacobian_N_s_sd.npy", np.abs(G_N) * 0.05)

    data = {
        "industry": "aero", "mu": 2, "K": 0, "granular": True, "S": S, "n_AA": n_AA,
        "n_coef": n_coef, "n_tau": n_tau, "folder": folder, "inference_step": "step3",
        "sector_names": [f"S{s}" for s in range(S)],
        "moment_mask": mask,
        "best_simulated_moments": np.abs(rng.normal(1.0, 0.1, (full_len, 2))) + 0.5,
        "granular_diagnostics": {"N_hat": np.array([12.0, 8.0, 20.0])},
        "moment_block_sizes": moment_block_sizes,
        "param_block_sizes": param_block_sizes,
        "moment_block_names": ["Labor share", "Industry shares", "Downstream sales",
                               "Extensive margin", "Regional sourcing shares",
                               "Zero-supplier share"],
        "param_block_names": ["Omega_L", "Industry shares", "Productivity",
                              "Trade cost", "Comparative advantage"],
        "moment_labels": [f"m{i}" for i in range(n_rows)],
        "param_labels": [f"p{i}" for i in range(n_par)],
        "J": rng.normal(0, 0.1, (n_rows, n_par)),
        "J_elast": rng.normal(0, 0.1, (n_rows, n_par)),
        "J_sd": np.abs(rng.normal(0, 0.01, (n_rows, n_par))),
        "J_elast_sd": np.abs(rng.normal(0, 0.01, (n_rows, n_par))),
    }

    ext = NS["with_variety_counts"](data)
    assert ext["J_elast"].shape == (n_rows, n_par + S), ext["J_elast"].shape
    assert ext["param_block_names"][-1] == "Variety count"
    assert len(ext["param_labels"]) == n_par + S
    assert len(ext["param_block_sizes"]) == len(ext["param_block_names"])

    # the N_s columns must be exactly zero outside block 6, and the elasticity must be
    # N_hat * dm/dN / m on the G0 rows
    E = ext["J_elast"][:, n_par:]
    assert np.all(E[:n_rows - S, :] == 0.0)
    m0 = data["best_simulated_moments"][:, 0][mask]
    for s in range(S):
        row = n_rows - S + s
        want = G_N[n_coef + n_gam + s, s] * 12.0 * (1 if s else 1)
        want = G_N[n_coef + n_gam + s, s] * data["granular_diagnostics"]["N_hat"][s] / m0[row]
        assert abs(E[row, s] - want) < 1e-12, (E[row, s], want)
    print("N_s block embedding OK")

    df = NS["identification_summary"](ext)
    print(df["share_above"])
    assert df["share_above"].loc["Zero-supplier share", "Variety count"] > 0
    assert df["share_exact_zero"].loc["Extensive margin", "Variety count"] == 1.0

    out = TMP / "figs"
    out.mkdir(exist_ok=True)
    NS["plot_identification_map"](ext, save_to=str(out / "map.png"))
    NS["plot_jacobian_thresholded"](ext, save_to=str(out / "thr.png"))
    NS["plot_channel_elasticities"](ext, save_to=str(out / "chan.png"))
    NS["plot_channel_elasticities"](ext, stat="mean")
    NS["plot_jacobian_full"](ext)          # the existing plots must still accept the extended dict
    NS["plot_jacobian_blocks"](ext)
    print(NS["jacobian_block_summary"](ext))

    try:
        NS["plot_channel_elasticities"](data)
    except ValueError as e:
        print("expected failure on the un-augmented dict:", str(e)[:80])
    else:
        raise AssertionError("should have raised")

    print("ALL OK")

def gate_untargeted():
    """PPML against statsmodels, then the a_ir panel and the moment end to end."""
    rng = np.random.default_rng(7)

    # ---------------------------------------------------------------------------
    # 1. ppml_absorb against statsmodels GLM Poisson with EXPLICIT group dummies
    # ---------------------------------------------------------------------------
    n, n_g = 4000, 12
    g = rng.integers(0, n_g, n)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    a = rng.normal(0, 0.7, n_g)
    eta_true = np.array([-0.35, 0.20])
    mu_true = np.exp(a[g] + x1 * eta_true[0] + x2 * eta_true[1])
    y = rng.poisson(mu_true).astype(float)

    X = np.column_stack([x1, x2])
    fit = NS["ppml_absorb"](y, X, g, cluster=None)

    D = pd.get_dummies(pd.Series(g), drop_first=False).to_numpy(dtype=float)
    Xd = np.column_stack([X, D])
    sm_fit = sm.GLM(y, Xd, family=sm.families.Poisson()).fit()
    print("beta   ours:", fit["beta"], " statsmodels:", sm_fit.params[:2])
    assert np.allclose(fit["beta"], sm_fit.params[:2], atol=1e-8), "beta mismatch"
    assert np.allclose(fit["se"], sm_fit.bse[:2], rtol=1e-6), (fit["se"], sm_fit.bse[:2])
    print("PPML == statsmodels GLM Poisson with explicit dummies (beta and classical se)")

    # weights
    w = rng.gamma(2.0, 0.5, n)
    fw = NS["ppml_absorb"](y, X, g, w=w, cluster=None)
    smw = sm.GLM(y, Xd, family=sm.families.Poisson(), freq_weights=w).fit()
    assert np.allclose(fw["beta"], smw.params[:2], atol=1e-8), (fw["beta"], smw.params[:2])
    print("weighted PPML matches too")

    # non-count outcome: PPML is a conditional-MEAN estimator, shares are legitimate
    y_share = mu_true * rng.gamma(3, 1 / 3, n)
    fs = NS["ppml_absorb"](y_share, X, g, cluster=None)
    assert np.max(np.abs(fs["beta"] - eta_true)) < 0.1, fs["beta"]
    print("continuous (share-like) outcome recovers the truth:", fs["beta"])

    # separated group: all-zero outcome in one group must be dropped, not crash
    y_sep = y.copy()
    y_sep[g == 3] = 0.0
    fsep = NS["ppml_absorb"](y_sep, X, g, cluster=g)
    assert fsep["n_dropped_groups"] == 1 and fsep["n_obs"] == int((g != 3).sum())
    print("separated group dropped:", fsep["n_dropped_groups"], "->", fsep["n_obs"], "obs")

    # clustered vcov >= classical here, and the machinery runs
    fc = NS["ppml_absorb"](y, X, g, cluster=rng.integers(0, 30, n))
    print("CRV1 se:", fc["se"], " classical se:", fit["se"], " clusters:", fc["n_clusters"])

    # ---------------------------------------------------------------------------
    # 2. End to end on a synthetic run tree with a PLANTED distance decay
    # ---------------------------------------------------------------------------
    S, R, R_d, N_rho = 4, 30, 8, 60
    folder = TMP / "synthrun_unt"
    inp = TMP / "synthinput_unt"
    folder.mkdir(parents=True, exist_ok=True)
    inp.mkdir(parents=True, exist_ok=True)

    coords = rng.uniform(0, 400, (R, 2))
    Dm = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(Dm, 0.0)
    np.save(inp / "distances.npy", Dm)
    downstream = np.sort(rng.choice(R, R_d, replace=False)) + 1     # 1-based R-indices

    ETA_PLANT = -0.45
    rows = []
    siren = 0
    for s in range(1, S + 1):
        for rho in range(1, N_rho + 1):
            l = int(rng.integers(1, R + 1))                # the winning cell
            siren += 1
            served = False
            for r in downstream:
                d = max(Dm[l - 1, r - 1], 1.0)
                p = np.clip(0.85 * (d / 50.0) ** (-0.5), 0.02, 0.95)
                if rng.random() < p:                       # extensive margin
                    served = True
                    val = np.exp(ETA_PLANT * np.log(d)) * rng.gamma(4, 0.25)
                    rows.append({"SIREN": siren, "A129": s, "ze2010": l,
                                 "ze2010_downstream": int(r), "share": val,
                                 "downstream_purchase": 1.0 + 0.1 * (r % 3),
                                 "intermediate_derivative": 0.0,
                                 "productivity": float(rng.lognormal(0, 0.5)),
                                 "sample_weight": 1.0 / N_rho})
            if not served:                                 # non-winners are not suppliers
                siren -= 1
    sup = pd.DataFrame(rows)
    sup.to_parquet(folder / "suppliers.parquet")
    print(f"\nsynthetic suppliers.parquet: {len(sup):,} linkages, "
          f"{sup.SIREN.nunique():,} suppliers, {R_d} downstream regions")

    data = {"industry": "auto", "mu": 1, "S": S, "R": R, "step_dir": "step1",
            "folder": folder, "input_folder": inp,
            "suppliers": pd.read_parquet(folder / "suppliers.parquet")}

    panel = NS["build_a_ir_panel"](data)
    n_firms = sup.SIREN.nunique()
    assert len(panel) == n_firms * R_d, (len(panel), n_firms * R_d)
    assert panel["a_ir"].min() == 0.0 and (panel["a_ir"] > 0).any()
    tot = panel.groupby("SIREN")["a_ir"].sum()
    assert np.allclose(tot.values, 1.0), tot.describe()      # shares add to one per firm
    d_check = panel.sample(200, random_state=0)
    assert np.allclose(d_check["distance"].to_numpy(),
                       Dm[d_check["ze2010"] - 1, d_check["ze2010_downstream"] - 1])
    assert (panel["log_distance"] >= 0).all()
    print("panel: zero-filled, a_ir sums to 1 per supplier, distances match distances.npy")

    res_auto = NS["estimate_untargeted_moment"](data, panel=panel)
    assert res_auto["eta"] < 0, res_auto["eta"]
    print(f"planted eta = {ETA_PLANT} (plus the extensive margin, so |eta| should exceed it)")

    # the same thing through pyfixest, if it happens to be installed
    try:
        import pyfixest as pf
        pfd = panel.rename(columns={"a_ir": "y"})
        m = pf.fepois(fml="y ~ log_distance | fe_group", data=pfd,
                      weights="sample_weight", vcov={"CRV1": "cluster"})
        print("pyfixest eta:", float(m.coef().iloc[0]), " se:", float(m.se().iloc[0]))
        assert abs(float(m.coef().iloc[0]) - res_auto["eta"]) < 1e-6
        print("matches pyfixest.fepois")
    except ImportError:
        print("(pyfixest not installed — skipped the cross-check)")

    dec = NS["decompose_untargeted_moment"](data, panel=panel)
    assert dec["extensive"]["beta"][0] < 0

    # a second industry so the joint table/figure paths are exercised
    data2 = dict(data, industry="aero")
    res_aero = NS["estimate_untargeted_moment"](data2, panel=panel)

    summ = NS["untargeted_summary"]([res_auto, res_aero])
    print("\n", summ.to_string())
    assert "delta/gamma (data)" in summ.columns and summ["eta (model)"].notna().all()

    out = TMP / "figs"
    out.mkdir(exist_ok=True)
    NS["plot_untargeted_moment"]([res_auto, res_aero], save_to=str(out / "untargeted.png"))
    NS["plot_a_ir_profile"]([res_auto, res_aero], save_to=str(out / "profile.png"))

    # controls path
    NS["estimate_untargeted_moment"](data, panel=panel, controls=("log_productivity",))

    # a missing artefact must say what is missing
    try:
        NS["build_a_ir_panel"](dict(data, suppliers=None))
    except FileNotFoundError as e:
        print("\nexpected:", str(e)[:70], "...")
    else:
        raise AssertionError("should have raised")

    print("\nALL OK")

def gate_comparative_advantage():
    """The T layout, the closed-form win probabilities, and the three measurements."""
    rng = np.random.default_rng(11)

    # ---------------------------------------------------------------- synthetic run
    S, R, n_AA, n_tau = 4, 24, 6, 1
    THETA, ALPHA = 1.768, 0.30
    inp = TMP / "synthinput_ca"
    folder = TMP / "synthrun_ca"
    (folder / "step3").mkdir(parents=True, exist_ok=True)
    inp.mkdir(parents=True, exist_ok=True)

    coords = rng.uniform(0, 500, (R, 2))
    D = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(D, 0.0)
    np.save(inp / "distances.npy", D)

    # the first n_AA regions are the downstream ones (so N_downstream flags them)
    N_down = np.zeros(R); N_down[:n_AA] = 3.0
    aa_of_ze = rng.integers(0, n_AA, R); aa_of_ze[:n_AA] = np.arange(n_AA)
    AA_ACTIVE = rng.random((S, n_AA)) < 0.8
    for s in range(S):                      # every sector needs at least two live areas
        AA_ACTIVE[s, :2] = True
    CELL_MASK = AA_ACTIVE[:, aa_of_ze]
    T_REF_AA = np.array([np.flatnonzero(AA_ACTIVE[s])[0] for s in range(S)])

    # planted T: sector 0 has one area with a large edge, sector 1 has a flat profile
    T_true = np.zeros((S, n_AA))
    for s in range(S):
        act = np.flatnonzero(AA_ACTIVE[s])
        T_true[s, act] = np.exp(rng.normal(0, 0.15 if s == 1 else 0.6, act.size))
    BIG_GAP = 3.0
    top0 = np.flatnonzero(AA_ACTIVE[0])[1]
    T_true[0, top0] = np.exp(np.median(np.log(T_true[0, AA_ACTIVE[0]])) + BIG_GAP)

    n_T = int(AA_ACTIVE.sum())
    best_params = np.concatenate([
        [0.4],                                     # Omega_L
        rng.uniform(0.5, 1.5, S),                  # Omega_s
        rng.uniform(0.5, 1.5, n_AA),               # A (R_downstream == n_AA)
        [ALPHA],                                   # alpha
        T_true[AA_ACTIVE],                         # T, s-major over (S, n_AA)
    ])

    data = {
        "industry": "aero", "mu": 2, "S": S, "R": R, "n_AA": n_AA, "n_tau": n_tau,
        "folder": folder, "input_folder": inp, "step_dir": "step3",
        "sector_names": [f"sec{s}" for s in range(S)],
        "aa_names": [f"A{a}" for a in range(n_AA)],
        "aa_of_ze": aa_of_ze, "AA_ACTIVE": AA_ACTIVE, "CELL_MASK": CELL_MASK,
        "T_REF_AA": T_REF_AA, "N_downstream": N_down,
        "emp_pi_r": rng.dirichlet(np.ones(n_AA)),
        "best_params": best_params,
        "coefs": pd.DataFrame({"value": [1.0]}),        # no `theta` entry -> the default
    }

    # ------------------------------------------------------------------ unpack
    est = NS["unpack_estimated_T"](data)
    assert abs(est["alpha"][0] - ALPHA) < 1e-12
    assert abs(est["theta"] - 1.768) < 1e-12, est["theta"]
    for s in range(S):
        ref = T_REF_AA[s]
        assert abs(est["T"][s, ref] - 1.0) < 1e-12                     # ref-normalised
        act = np.flatnonzero(AA_ACTIVE[s])
        assert np.allclose(est["T"][s, act], T_true[s, act] / T_true[s, ref])
        assert np.all(est["T"][s, ~AA_ACTIVE[s]] == 0.0)
    print("unpack_estimated_T: layout + reference normalisation OK")

    try:
        NS["unpack_estimated_T"](dict(data, best_params=best_params[:-1]))
    except ValueError as e:
        print("expected on a length mismatch:", str(e)[:60], "...")
    else:
        raise AssertionError("should have raised")

    # ------------------------------------------------------------------ geometry
    geom = NS["sourcing_geometry"](data)
    for s, blk in geom["by_sector"].items():
        assert np.allclose(blk["rho"].sum(axis=0), 1.0)                 # a proper distribution
        assert blk["rho"].shape == (blk["cells"].size, n_AA)
        # rho is the closed form, recomputed independently here
        Tc = est["T"][s, aa_of_ze[blk["cells"]]]
        d = np.maximum(D[np.ix_(blk["cells"], np.arange(n_AA))], 1.0)
        psi = Tc[:, None] * d ** (-THETA * ALPHA)
        assert np.allclose(blk["rho"], psi / psi.sum(0, keepdims=True))
    print("sourcing_geometry: win probabilities match the closed form and sum to one")

    # alpha = 0 -> the same distribution for every buyer, proportional to T
    g0 = NS["sourcing_geometry"](data, alpha=0.0)
    for s, blk in g0["by_sector"].items():
        assert np.allclose(blk["rho"], blk["rho"][:, :1])
        Tc = est["T"][s, aa_of_ze[blk["cells"]]]
        assert np.allclose(blk["rho"][:, 0], Tc / Tc.sum())
    # T equalised -> pure gravity
    gT = NS["sourcing_geometry"](data, equalise_T=True)
    for s, blk in gT["by_sector"].items():
        assert np.allclose(blk["T_cell"], 1.0)
    print("counterfactual geometries behave (alpha=0 buyer-invariant; T equalised = gravity)")

    # ------------------------------------------------------- distance equivalence
    eq = NS["ca_distance_equivalence"](data)
    ta = THETA * ALPHA
    assert abs(eq.attrs["theta_alpha"] - ta) < 1e-12
    # sector 0 carries the planted gap, in T RATIOS so the reference normalisation cancels
    assert abs(eq.loc["sec0", "dlogT_top_vs_median"] - BIG_GAP) < 1e-9, \
        eq.loc["sec0", "dlogT_top_vs_median"]
    assert abs(eq.loc["sec0", "equiv_log_d"] - BIG_GAP / ta) < 1e-9
    assert eq.loc["sec0", "CA_beats_typical_geography"]          # 3/0.53 = 5.7 log points
    assert not eq.loc["sec1", "CA_beats_typical_geography"]      # the flat sector
    # the extreme range is anchored at the own-region cell (distance floored at 1 km), so
    # it is much wider than the spread a buyer typically faces
    assert (eq["geo_log_range"] > eq["geo_log_spread"]).all()
    print(eq[["top_area", "dlogT_top_vs_median", "equiv_log_d", "geo_log_spread",
              "geo_log_range", "CA_beats_typical_geography"]].to_string())

    # --------------------------------------------------------- variance decomposition
    vd = NS["ca_variance_decomposition"](data)
    tot = vd["share_CA"] + vd["share_distance"] + vd["share_covariance"]
    assert np.allclose(tot.to_numpy(), 1.0), tot                 # the identity is exact
    assert vd.loc["sec0", "ratio_CA_over_distance"] > vd.loc["sec1", "ratio_CA_over_distance"]
    print("\n", vd.to_string())

    # ------------------------------------------------------------- counterfactuals
    cf = NS["counterfactual_sourcing"](data)
    assert set(cf.index.get_level_values("regime").unique()) == set(NS["CF_REGIMES"])
    for s in data["sector_names"]:
        n = cf.loc[(s, "Neither"), "n_cells"]
        assert abs(cf.loc[(s, "Neither"), "hhi"] - 1.0 / n) < 1e-9      # uniform benchmark
        assert abs(cf.loc[(s, "Neither"), "top_cell_share"] - 1.0 / n) < 1e-9
        # distance pulls sourcing closer than uniform does
        assert cf.loc[(s, "Distance only"), "mean_distance"] < cf.loc[(s, "Neither"), "mean_distance"]
    # the planted edge concentrates sector 0 on its top area, far above the uniform
    # benchmark, and far above the flat sector
    assert cf.loc[("sec0", "Both forces"), "top_area_share"] > \
        3 * cf.loc[("sec0", "Neither"), "top_area_share"]
    assert cf.loc[("sec0", "Both forces"), "top_area_share"] > \
        2 * cf.loc[("sec1", "Both forces"), "top_area_share"]
    # and switching distance off concentrates it further still (CA is what does the work)
    assert cf.loc[("sec0", "Comparative advantage only"), "top_area_share"] > \
        cf.loc[("sec0", "Both forces"), "top_area_share"]
    print("\n", cf.round(3).to_string())

    # ---------------------------------------------------------------------- output
    out = TMP / "figs"
    out.mkdir(exist_ok=True)
    NS["plot_ca_distribution"](data, save_to=str(out / "ca_dist.png"))
    NS["plot_ca_distance_equivalence"](data, save_to=str(out / "ca_equiv.png"))
    NS["plot_counterfactual_sourcing"](data, save_to=str(out / "ca_cf.png"))
    summ = NS["comparative_advantage_summary"](data)
    print("\n", summ.round(3).to_string())
    assert len(summ) == S

    # a binned trade cost has no single elasticity and must say so
    try:
        bad = dict(data, n_tau=2,
                   best_params=np.insert(best_params, 1 + S + n_AA + 1, 0.2))
        NS["sourcing_geometry"](bad)
    except ValueError as e:
        print("\nexpected under N_TAU > 1:", str(e)[:70], "...")
    else:
        raise AssertionError("should have raised")

    print("\nALL OK")


if __name__ == "__main__":
    for name, gate in (("identification", gate_identification),
                       ("untargeted moment", gate_untargeted),
                       ("comparative advantage", gate_comparative_advantage)):
        print("\n" + "=" * 70)
        print(f"GATE: {name}")
        print("=" * 70)
        gate()
    print("\nALL GATES PASSED")
