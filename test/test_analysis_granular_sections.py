"""
Gates for the sections added to `analysis_granular.ipynb`:
  1. Identification / sensitivity  — the variety-count columns and the noise mask,
  2. Untargeted moment             — the PPML distance elasticity of comovement,
  3. Comparative advantage         — T against distance, within sector,
  4. Amplification                 — D_r and the share of upstream sales within d km,
  5. IO benchmark                  — the TES parser and the Leontief multipliers.

No Julia and no real data: each gate writes a synthetic run tree with the exact file
layout the loader expects and then EXECUTES THE NOTEBOOK'S OWN CODE CELLS against it,
so what is tested is the shipped artefact rather than a copy of it. Run with

    python test/test_analysis_granular_sections.py

Requires numpy, pandas, matplotlib, pyfixest, pyarrow.
"""
import json
import os
import re
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


# The notebook alternates DEFINITION cells with cells that RUN the reporting — the
# per-section `for cfg in INDUSTRIES:` loops, the joint table, the Constants cell and
# the full run at the bottom. Those touch the real run tree, which does not exist here,
# so they are skipped and only the definitions are executed.
#
# A cell is a run cell when it hits one of the markers below AND defines nothing at top
# level. The second condition matters: the loader cell's DOCSTRING shows the call
# `data = load_granular_data(industry, mu=2)`, so the marker alone would skip the very
# cell that defines everything.
RUN_CELL_MARKERS = (
    "RUN THE REPORTING",
    "RUN THIS SECTION ON ITS OWN",
    "= load_granular_data(",
    "generate_combined_table(INDUSTRIES",
    "RUN_KWARGS = dict(",
)
# ... except that a definition cell may legitimately contain one of those strings in a
# docstring, which is why `notebook_namespace` also requires the cell to define nothing.


def notebook_namespace():
    """
    Every DEFINITION cell of the notebook, executed in order.

    Cells are located by CONTENT rather than by index, so inserting a section above
    does not silently change what is being tested.
    """
    nb = json.load(open(NB_PATH))
    ns = {}
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        defines = re.search(r"^(def|class)\s", src, re.M) is not None
        if src.lstrip().startswith("%") or (
                not defines and any(m in src for m in RUN_CELL_MARKERS)):
            continue
        exec(compile(src, f"<notebook cell {i}>", "exec"), ns)
    for name, value in (("THETA_DEFAULT", 1.768),      # normally set by the skipped
                        ("EMPIRICAL_MEAN_LOG_D", 5.8)):  # Constants cell
        ns.setdefault(name, value)
    return ns


NS = notebook_namespace()


def gate_identification():
    """
    The variety-count columns on the parameter axis, and the noise-to-signal mask.

    Julia appends dm/dN_s to the saved Jacobian, so the notebook receives them as
    ordinary columns; what has to hold here is that the two thresholds do different
    things — a structural zero must be reported as a zero, an entry whose Monte-Carlo
    standard deviation swamps it must be reported as unmeasured, and the two must never
    be pooled.
    """
    S, n_AA, n_coef, n_tau, R_d = 3, 4, 4, 1, 4
    n_gam = 6
    moment_block_sizes = [1, S - 1, R_d - 1, n_coef, n_gam, S]
    param_block_sizes = [1, S - 1, R_d - 1, n_tau, n_gam, S]
    n_rows, n_par = sum(moment_block_sizes), sum(param_block_sizes)

    rng = np.random.default_rng(0)
    E = rng.normal(0, 0.1, (n_rows, n_par))
    E_sd = np.abs(E) * 0.1                       # readable everywhere by default
    # the N_s block: exactly zero outside the zero-supplier rows, diagonal on them
    E[:, n_par - S:] = 0.0
    E_sd[:, n_par - S:] = 0.0
    for k in range(S):
        E[n_rows - S + k, n_par - S + k] = -0.25
        E_sd[n_rows - S + k, n_par - S + k] = 0.02
    # one entry deliberately drowned in noise: large elasticity, larger SD
    E[0, 0], E_sd[0, 0] = 0.4, 0.8

    data = {
        "industry": "aero", "mu": 2, "K": 0, "granular": True, "S": S, "n_AA": n_AA,
        "n_coef": n_coef, "n_tau": n_tau, "folder": TMP / "synthrun",
        "inference_step": "step3", "n_N_cols": S,
        "sector_names": [f"S{s}" for s in range(S)],
        "moment_block_sizes": moment_block_sizes,
        "param_block_sizes": param_block_sizes,
        "moment_block_names": ["Labor share", "Industry shares", "Downstream sales",
                               "Extensive margin", "Regional sourcing shares",
                               "Zero-supplier share"],
        "param_block_names": ["Omega_L", "Industry shares", "Productivity",
                              "Trade cost", "Comparative advantage", "Variety count"],
        "moment_labels": [f"m{i}" for i in range(n_rows)],
        "param_labels": [f"p{i}" for i in range(n_par - S)]
                        + [f"N_s[S{s}]" for s in range(S)],
        "J": E / 2.0, "J_elast": E, "J_sd": E_sd / 2.0, "J_elast_sd": E_sd,
    }

    R = NS["jacobian_noise_ratio"](data)
    assert np.all(R[:, n_par - S:][: n_rows - S] == 0.0), "structural zeros must read 0"
    assert abs(R[0, 0] - 2.0) < 1e-12, R[0, 0]
    assert abs(R[1, 1] - 0.1) < 1e-12, R[1, 1]
    mask = NS["jacobian_noise_mask"](data, noise_max=NS["NOISE_MAX"])
    assert mask[0, 0] and mask.sum() == 1, mask.sum()
    print("noise ratio: exact zeros measured, the planted noisy entry is the only mask")

    df = NS["identification_summary"](data)
    assert df["share_exact_zero"].loc["Extensive margin", "Variety count"] == 1.0
    assert df["share_above"].loc["Zero-supplier share", "Variety count"] > 0
    assert df["share_readable"].loc["Labor share", "Omega_L"] < 1.0
    assert df["share_live"].loc["Labor share", "Omega_L"] == 0.0   # large but unmeasured
    print("identification summary: structural zeros, weak channels and noise separated")

    # the channel figure must not let an unmeasurable entry produce a bar
    axes = NS["plot_channel_elasticities"](data, noise_max=NS["NOISE_MAX"])
    assert len(axes) == 3

    out = TMP / "figs"
    out.mkdir(exist_ok=True)
    for fn in ("plot_identification_map", "plot_jacobian_thresholded",
               "plot_jacobian_noise", "plot_jacobian_full", "plot_jacobian_blocks"):
        NS[fn](data, save_to=str(out / f"{fn}.png"))
    print(NS["jacobian_block_summary"](data).round(3).to_string())

    # a run whose Jacobian has no variety-count columns must say so, not plot nonsense
    plain = dict(data,
                 param_block_names=data["param_block_names"][:-1],
                 param_block_sizes=data["param_block_sizes"][:-1],
                 param_labels=data["param_labels"][:n_par - S],
                 n_N_cols=0,
                 J=data["J"][:, :n_par - S], J_elast=E[:, :n_par - S],
                 J_sd=data["J_sd"][:, :n_par - S], J_elast_sd=E_sd[:, :n_par - S])
    try:
        NS["plot_channel_elasticities"](plain)
    except ValueError as e:
        print("expected failure without the variety-count columns:", str(e)[:80])
    else:
        raise AssertionError("should have raised")

    print("ALL OK")

def gate_untargeted():
    """PPML against statsmodels, then the a_ir panel and the moment end to end."""
    rng = np.random.default_rng(7)

    # ---------------------------------------------------------------------------
    # 1. fepois_fit — the thin pyfixest wrapper — against statsmodels' Poisson GLM with
    #    EXPLICIT group dummies. Different algorithm, same estimand: what is being gated
    #    is that the formula, the weights and the absorbed fixed effect are wired the way
    #    the moment intends, not pyfixest's own arithmetic.
    # ---------------------------------------------------------------------------
    n, n_g = 4000, 12
    g = rng.integers(0, n_g, n)
    x1 = rng.normal(0, 1, n)
    a_g = rng.normal(0, 0.7, n_g)
    eta_true = -0.35
    mu_true = np.exp(a_g[g] + x1 * eta_true)
    y = rng.poisson(mu_true).astype(float)

    d = pd.DataFrame({"a_ir": y, "log_distance": x1, "fe_group": g.astype(str),
                      "sample_weight": np.ones(n), "cluster": g.astype(str)})
    fit = NS["fepois_fit"](d, "a_ir", cluster=None)
    beta = float(fit.coef()["log_distance"])

    D = pd.get_dummies(pd.Series(g), drop_first=False).to_numpy(dtype=float)
    sm_fit = sm.GLM(y, np.column_stack([x1, D]), family=sm.families.Poisson()).fit()
    print("beta   pyfixest:", beta, " statsmodels:", sm_fit.params[0])
    assert abs(beta - sm_fit.params[0]) < 1e-6, "beta mismatch"

    # a non-count outcome: PPML is a conditional-MEAN estimator, shares are legitimate
    d_share = d.assign(a_ir=mu_true * rng.gamma(3, 1 / 3, n))
    b_share = float(NS["fepois_fit"](d_share, "a_ir", cluster=None).coef()["log_distance"])
    assert abs(b_share - eta_true) < 0.1, b_share
    print("continuous (share-like) outcome recovers the truth:", b_share)

    # weights and clustering must both go through
    d_w = d.assign(sample_weight=rng.gamma(2.0, 0.5, n))
    fw = NS["fepois_fit"](d_w, "a_ir", cluster="cluster")
    print("weighted + CRV1 se:", float(fw.se()["log_distance"]),
          " classical se:", float(fit.se()["log_distance"]))

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
            "suppliers_path": folder / "suppliers.parquet",
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

    # the decomposition: the extensive margin must be negative (a more distant supplier
    # is less likely to serve at all) and the identity must close to within the
    # fixed-effect wedge it is documented to carry
    dec = NS["decompose_untargeted_moment"](data, panel=panel)
    tab = dec["table"].set_index("channel")["coefficient"]
    # the identity decomposes the REPORTED number: same fixed effect as the baseline
    assert abs(dec["table"].loc[0, "coefficient"] - res_auto["eta"]) < 1e-9
    assert tab.iloc[1] < 0, tab
    assert abs(dec["gap"]) < 0.25, dec["gap"]
    assert dec["table"].loc[2, "n_obs"] < dec["table"].loc[0, "n_obs"]   # positives only
    print("decomposition: extensive margin negative, identity closes to", round(dec["gap"], 4))

    # the margins net of theta*alpha: the identity must hold on the normalised numbers
    # too, and the panel's own facts must be the objects they are compared with
    d_ta = dict(data, best_params=None)
    facts = NS["panel_margin_facts"](panel)
    assert 0.0 <= facts["serving_rate"] <= 1.0
    assert abs(facts["serving_rate"] + facts["share_zero"] - 1.0) < 1e-12
    assert 0.0 <= facts["openness_weighted"] <= 1.0
    # the share-weighted openness is the SMALLER one: large shares carry large weight
    assert facts["openness_weighted"] <= facts["openness_served"] + 1e-12
    assert abs(facts["buyers_per_supplier"]
               - (panel["a_ir"] > 0).sum() / panel["SIREN"].nunique()) < 1e-9
    md = NS["margin_decomposition"]([dec, dict(dec, industry="aero")])
    assert list(md.index) == ["Motor Vehicles", "Aerospace"]
    if np.isfinite(dec["theta_alpha"]):
        assert abs(md.loc["Motor Vehicles", "bracket |eta|/(theta*alpha)"]
                   - abs(dec["eta"]) / dec["theta_alpha"]) < 1e-12
    NS["plot_margin_decomposition"](md, save_to=str(TMP / "figs" / "margins.png"))
    print("margins net of theta*alpha: normalised, and checked against the panel's facts")

    # the served indicator the extensive-margin regression runs on
    assert set(np.unique(panel["served"])) <= {0.0, 1.0}
    assert np.allclose(panel["served"].to_numpy(), (panel["a_ir"] > 0).astype(float))

    # a second industry so the joint table/figure paths are exercised
    data2 = dict(data, industry="aero")
    res_aero = NS["estimate_untargeted_moment"](data2, panel=panel)

    # the estimated fixed-effect specifications, and the map onto the reduced form's scale
    assert set(res_auto["specs"]) == set(NS["UNTARGETED_FE_SPECS"])
    # the baseline is the two-way effect: the reduced form's own design, and the variation
    # the appendix's derivative is taken along
    assert NS["UNTARGETED_FE_SPECS"][NS["BASELINE_FE_LABEL"]] == NS["BASELINE_FE"]
    assert set(NS["BASELINE_FE"].split(" + ")) == {"fe_group", "SIREN"}
    for label, sp in res_auto["specs"].items():
        assert abs(sp["dg"] - NS["delta_over_gamma_from_eta"](sp["eta"])) < 1e-12
        # the map is monotone and compressive, and bounded by 1/mean(log d)
        assert abs(sp["dg"]) < abs(sp["eta"]) + 1e-12
        assert abs(sp["dg"]) < 1.0 / NS["EMPIRICAL_MEAN_LOG_D"]
    assert res_auto["eta"] == res_auto["specs"][NS["BASELINE_FE_LABEL"]]["eta"]
    # the round trip through the two maps must return the elasticity it started from
    e = res_auto["eta"]
    assert abs(NS["eta_from_delta_over_gamma"](NS["delta_over_gamma_from_eta"](e)) - e) < 1e-10
    print("delta/gamma map: monotone, compressive, bounded, and invertible")

    summ = NS["untargeted_summary"]([res_auto, res_aero])
    print("\n", summ.to_string())
    assert "delta/gamma (data)" in summ.columns and summ["delta/gamma (model)"].notna().all()
    # the paper reads one scale: no elasticity column anywhere in the table
    assert "eta (model)" not in summ.columns and "se" not in summ.columns
    # ONE row per industry: the section reports one specification, and the fixed effect it
    # was run under is a column rather than a second index level
    assert len(summ) == 2 and list(summ.index.names) == ["industry"]
    assert set(summ["fixed effect"]) == {NS["BASELINE_FE_LABEL"]}

    out = TMP / "figs"
    out.mkdir(exist_ok=True)
    NS["plot_untargeted_moment"]([res_auto, res_aero], save_to=str(out / "untargeted.png"))
    NS["plot_a_ir_profile"]([res_auto, res_aero], save_to=str(out / "profile.png"))
    labels, b, se, raw = NS["a_ir_bin_profile"](panel, (0, 50, 100, 150, 200, 300, np.inf))
    assert b[0] == 0.0 and se[0] == 0.0 and abs(raw[0]) < 1e-12   # nearest bin = reference
    assert len(labels) == len(b) == len(raw)
    assert np.isfinite(b[1:]).all()
    # the fixed effect must actually change the profile: absorbed != unconditional
    assert np.nanmax(np.abs(b - raw)) > 1e-6
    print("bin profile: reference bin normalised, FE-absorbed profile differs from the raw one")

    # ---------------------------------------------------------------------------
    # 3. The specification ladder: is the gap with the reduced form a specification
    #    difference? Each rung has a property that must hold whatever the data.
    # ---------------------------------------------------------------------------
    lin = NS["linear_delta_over_gamma"](panel)
    # the closed form the markdown quotes: delta/gamma ~= eta / (1 + |eta| mean log d),
    # so a level-linear ratio can never exceed 1/mean(log d) in magnitude
    assert abs(lin["delta_over_gamma"]) < 1.0 / lin["mean_log_distance"] + 1e-9, lin
    # and the same fit read at the mean distance is strictly steeper than at d = 1 km
    assert abs(lin["elasticity_at_mean_d"]) > abs(lin["delta_over_gamma"]), lin
    # inverting the map must return the elasticity it came from
    r = NS["eta_from_delta_over_gamma"](-0.1, 5.0)
    assert abs(-0.1 - r / (1 + abs(r) * 5.0)) < 1e-12, r
    print("linear form: ratio below the 1/mean(log d) ceiling, inversion is exact")

    # pooling every variety of a (region, sector) into one firm must give strictly fewer
    # suppliers and leave the shares a proper distribution
    firms = NS["build_a_ir_panel"](data, firm_key=("ze2010", "A129"))
    assert firms["SIREN"].nunique() < panel["SIREN"].nunique()
    tot_f = firms.groupby("SIREN")["a_ir"].sum()
    assert np.allclose(tot_f.values, 1.0), tot_f.describe()
    assert len(firms) == firms["SIREN"].nunique() * R_d
    # a pooled firm serves at least as many regions as its varieties did separately
    assert (firms["a_ir"] > 0).mean() >= (panel["a_ir"] > 0).mean()
    print("multi-variety pooling: fewer suppliers, shares still sum to one, denser grid")

    ladder = NS["untargeted_specification_ladder"](data, panel=panel)
    # exactly the five rows the paper shows, and nothing else — in particular no
    # elasticity column: only the delta/gamma reading is comparable to Table 3
    assert list(ladder.index) == ["Baseline eta", "Intensive margin", "Extensive margin",
                                  "Multi-variety firms",
                                  "Empirical delta/gamma (Table 3)"], list(ladder.index)
    assert "eta" not in ladder.columns and "se" not in ladder.columns
    assert abs(ladder.loc["Baseline eta", "delta/gamma"]
               - NS["delta_over_gamma_from_eta"](res_auto["eta"])) < 1e-12
    assert ladder.loc["Empirical delta/gamma (Table 3)", "delta/gamma"] == \
        NS["EMPIRICAL_DELTA_OVER_GAMMA"]["auto"]["estimate"]
    # the level-linear cross-check is kept as an attribute, not as a rung
    assert abs(ladder.attrs["delta_over_gamma"] - lin["delta_over_gamma"]) < 1e-12
    # the multi-variety rung is estimated on the pooled panel, so on fewer observations
    assert (ladder.loc["Multi-variety firms", "n_obs"]
            < ladder.loc["Baseline eta", "n_obs"])
    # every interval is on the delta/gamma scale and brackets its point estimate
    assert (ladder["ci_lo"] <= ladder["delta/gamma"]).all()
    assert (ladder["delta/gamma"] <= ladder["ci_hi"]).all()
    assert np.isfinite(ladder["delta/gamma"]).all()
    NS["plot_untargeted_ladder"](ladder, "auto", save_to=str(out / "ladder.png"))
    print("specification ladder: five rows, delta/gamma only, baseline reproduced")

    # theta*alpha is read off the RUN's own best_params, not carried in the markdown:
    # layout [Omega_L(1) | Omega_s(S) | A(R_d) | alpha(N_TAU) | T(active (s,AA))]
    S_t, R_t, n_AA_t, alpha_t = 3, 4, 5, 0.37
    aa_active = np.zeros((S_t, n_AA_t), dtype=bool)
    aa_active[:, :2] = True
    bp = np.concatenate([[0.1], np.full(S_t, 0.2), np.full(R_t, 0.3), [alpha_t],
                         np.full(int(aa_active.sum()), 1.0)])
    d_ta = {"best_params": bp, "S": S_t, "n_tau": 1, "aa_names": ["a"] * R_t,
            "AA_ACTIVE": aa_active, "coefs": None, "folder": ".", "step_dir": "step1"}
    assert abs(NS["theta_alpha"](d_ta) - NS["THETA_DEFAULT"] * alpha_t) < 1e-12
    try:                                   # a flag mismatch must be an error, not a number
        NS["theta_alpha"](dict(d_ta, best_params=bp[:-1]))
        raise AssertionError("theta_alpha accepted a mis-sized best_params")
    except ValueError as e:
        print("expected on a mis-sized best_params:", str(e)[:70], "...")
    print("theta*alpha: read off best_params by the raw layout")

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

    # sector codes deliberately NOT in the model's 0..S order, so a table that comes
    # back sorted by code proves the ordering is applied rather than inherited
    sector_names = ["C29A", "C10Z", "C25A", "C22B"][:S]
    ze_codes = [f"{2000 + i:04d}" for i in range(R)]
    data = {
        "industry": "aero", "mu": 2, "S": S, "R": R, "n_AA": n_AA, "n_tau": n_tau,
        "folder": folder, "input_folder": inp, "step_dir": "step3",
        "sector_names": sector_names,
        "aa_names": [ze_codes[a] for a in range(n_AA)],
        "filter_N_upstream_df": pd.DataFrame(
            {"ze2010": ze_codes * S, "A129": np.repeat(sector_names, R)}),
        "france": pd.DataFrame({"ze2010": ze_codes,
                                "ze2010_name": [f"Zone {c}" for c in ze_codes]}),
        "aa_of_ze": aa_of_ze, "AA_ACTIVE": AA_ACTIVE, "CELL_MASK": CELL_MASK,
        "T_REF_AA": T_REF_AA, "N_downstream": N_down,
        "emp_pi_r": rng.dirichlet(np.ones(n_AA)),
        # observed AA-level sourcing shares: what the Sinkhorn inversion targets, and
        # the empirical half of the covariance split
        "emp_gamma_aa": np.where(AA_ACTIVE, rng.uniform(0.01, 0.4, (S, n_AA)), 0.0),
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
    S0, S1 = sector_names[0], sector_names[1]
    assert abs(eq.loc[S0, "dlogT_top_vs_median"] - BIG_GAP) < 1e-9, \
        eq.loc[S0, "dlogT_top_vs_median"]
    assert abs(eq.loc[S0, "equiv_log_d"] - BIG_GAP / ta) < 1e-9
    assert eq.loc[S0, "CA_beats_typical_geography"]          # 3/0.53 = 5.7 log points
    assert not eq.loc[S1, "CA_beats_typical_geography"]      # the flat sector
    # the extreme range is anchored at the own-region cell (distance floored at 1 km), so
    # it is much wider than the spread a buyer typically faces
    assert (eq["geo_log_range"] > eq["geo_log_spread"]).all()
    print(eq[["top_area", "dlogT_top_vs_median", "equiv_log_d", "geo_log_spread",
              "geo_log_range", "CA_beats_typical_geography"]].to_string())

    # --------------------------------------------------------- variance decomposition
    vd = NS["ca_variance_decomposition"](data)
    tot = vd["share_CA"] + vd["share_distance"] + vd["share_covariance"]
    assert np.allclose(tot.to_numpy(), 1.0), tot                 # the identity is exact
    assert vd.loc[S0, "ratio_CA_over_distance"] > vd.loc[S1, "ratio_CA_over_distance"]
    print("\n", vd.to_string())

    # ------------------------------------------------------- the exact win test
    wm = NS["ca_win_margin"](data)
    w_buy = data["emp_pi_r"] / data["emp_pi_r"].sum()
    for s in range(S):
        name = sector_names[s]
        if name not in wm.index:
            continue
        blk = geom["by_sector"][s]
        act = np.flatnonzero(AA_ACTIVE[s])
        top = int(act[np.argmax(est["T"][s, act])])
        in_top = blk["areas"] == top
        # the margin must be the closed form, recomputed here from scratch
        score = np.log(est["T"][s, blk["areas"]])[:, None] - ta * np.log(blk["distance"])
        margin = (score[in_top].max(0) - score[~in_top].max(0)) / ta
        assert abs(wm.loc[name, "mean_win_margin"] - float(margin @ w_buy)) < 1e-9
        assert abs(wm.loc[name, "share_buyers_won"] - float((margin > 0) @ w_buy)) < 1e-12
        # a positive margin must mean the argmax cell really is in the top area
        winner_in_top = in_top[blk["rho"].argmax(axis=0)]
        assert np.array_equal(margin > 0, winner_in_top)
        # the within-area handicap is pure distance: T cancels inside an area
        ld = np.log(blk["distance"])
        assert abs(wm.loc[name, "mean_within_penalty"]
                   - float((ld[in_top].max(0) - ld[in_top].min(0)) @ w_buy)) < 1e-9
    # the planted 3-log-point edge must win every buyer; the flat sector must not
    assert wm.loc[S0, "share_buyers_won"] == 1.0, wm.loc[S0, "share_buyers_won"]
    assert wm.loc[S1, "share_buyers_won"] < 1.0
    print("\n", wm.to_string())
    print("win margin: matches the closed form, agrees with the argmax, T cancels within area")

    # --------------------------------------------------- where the covariance comes from
    cb = NS["ca_covariance_benchmark"](data)
    assert np.allclose(cb["check_gamma_minus_M"].to_numpy(), 0.0, atol=1e-12), cb
    assert np.allclose((cb["cov_gamma"] - cb["cov_M"]).to_numpy(),
                       cb["cov_T"].to_numpy(), atol=1e-12)
    assert cb["share_covariance"].notna().all()
    print("\n", cb.round(4).to_string())

    # ------------------------------------------------------------- counterfactuals
    cf = NS["counterfactual_sourcing"](data)
    assert set(cf.index.get_level_values("regime").unique()) == set(NS["CF_REGIMES"])
    for s in sector_names:
        n = cf.loc[(s, "Neither"), "n_cells"]
        assert abs(cf.loc[(s, "Neither"), "hhi"] - 1.0 / n) < 1e-9      # uniform benchmark
        assert abs(cf.loc[(s, "Neither"), "top_cell_share"] - 1.0 / n) < 1e-9
        # distance pulls sourcing closer than uniform does
        assert cf.loc[(s, "Distance only"), "mean_distance"] < cf.loc[(s, "Neither"), "mean_distance"]
    # the planted edge concentrates sector 0 on its top area, far above the uniform
    # benchmark, and far above the flat sector
    assert cf.loc[(S0, "Both forces"), "top_area_share"] > \
        3 * cf.loc[(S0, "Neither"), "top_area_share"]
    assert cf.loc[(S0, "Both forces"), "top_area_share"] > \
        2 * cf.loc[(S1, "Both forces"), "top_area_share"]
    # and switching distance off concentrates it further still (CA is what does the work)
    assert cf.loc[(S0, "Comparative advantage only"), "top_area_share"] > \
        cf.loc[(S0, "Both forces"), "top_area_share"]
    print("\n", cf.round(3).to_string())

    # ---------------------------------------------------------------------- output
    out = TMP / "figs"
    out.mkdir(exist_ok=True)
    NS["plot_ca_distribution"](data, save_to=str(out / "ca_dist.png"))
    NS["plot_ca_distance_equivalence"](data, save_to=str(out / "ca_equiv.png"))
    NS["plot_ca_win_margin"](data, save_to=str(out / "ca_win_margin.png"))
    NS["plot_counterfactual_sourcing"](data, save_to=str(out / "ca_cf.png"))
    summ = NS["comparative_advantage_summary"](data)
    print("\n", summ.round(3).to_string())
    assert len(summ) == S
    # every sector-indexed object comes back in A129-CODE order, not in the model's
    # internal order and not sorted by whatever the figure ranks on
    code_order = sorted(sector_names)
    assert list(summ.index) == code_order, list(summ.index)
    assert list(eq.index) == code_order, list(eq.index)
    assert list(vd.index) == code_order, list(vd.index)
    assert list(dict.fromkeys(cf.index.get_level_values("sector"))) == code_order
    # and the top area is named by its commuting zone, not by its ZE code
    caf = NS["comparative_advantage_frame"](data)
    assert caf["area"].str.startswith("Zone").all(), caf["area"].unique()[:3]
    assert set(caf["area_code"]) <= set(data["aa_names"])
    # Test 1 is read in base-10: one unit on its axis is an order of magnitude of T, and
    # (being scale-free) 1/(theta*alpha) decades of distance
    assert np.allclose(caf["log10_T_dev"], caf["log_T_dev"] / np.log(10.0))
    assert (summ["top_area"].astype(str).str.startswith("Zone")).all()
    print("sector-indexed output is in A129-code order; areas carry region names")

    # every column of the summary has to be documented, and nothing is documented that
    # the summary does not produce — the glossary is the table's reading guide
    gloss = NS["comparative_advantage_glossary"]()
    assert set(summ.columns) <= set(gloss.index), set(summ.columns) - set(gloss.index)
    assert set(gloss.index) - set(summ.columns) == set(), \
        set(gloss.index) - set(summ.columns)
    print("every summary column is documented in comparative_advantage_glossary()")

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


def gate_amplification():
    """D_r and the share within a radius: the ratio, the grid and the CDF in d."""
    rng = np.random.default_rng(5)

    S, R, R_d, N_rho = 3, 20, 6, 40
    inp = TMP / "synthinput_amp"
    folder = TMP / "synthrun_amp"
    (folder / "step3").mkdir(parents=True, exist_ok=True)
    inp.mkdir(parents=True, exist_ok=True)

    coords = rng.uniform(0, 600, (R, 2))
    D = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(D, 0.0)
    np.save(inp / "distances.npy", D)
    downstream = np.arange(1, R_d + 1)                       # 1-based model indices

    # ZE codes / names, as the loader would build them
    codes = [f"{1000 + i:04d}" for i in range(R)]
    filter_df = pd.DataFrame({"ze2010": codes * S,
                              "A129": np.repeat([f"C{s}" for s in range(S)], R)})
    france = pd.DataFrame({"ze2010": codes, "ze2010_name": [f"Zone {c}" for c in codes]})

    # a firm-level economy with a planted distance gradient, so the local share is not
    # an artefact of a uniform draw
    rows, siren = [], 0
    TOTAL_INPUT_SHARE = 0.5          # every downstream region spends this on intermediates
    for r in downstream:
        w = np.exp(-np.maximum(D[:, r - 1], 1.0) / 120.0)
        w /= w.sum()
        for s in range(1, S + 1):
            for rho in range(1, N_rho + 1):
                l = int(rng.choice(np.arange(1, R + 1), p=w))
                siren += 1
                rows.append({"SIREN": siren, "A129": s, "ze2010": l,
                             "ze2010_downstream": int(r),
                             "share": TOTAL_INPUT_SHARE / (S * N_rho),
                             "downstream_purchase": 2.0,
                             "intermediate_derivative": 0.0,
                             "productivity": 1.0, "sample_weight": 1.0 / N_rho})
    sup = pd.DataFrame(rows)
    sup.to_parquet(folder / "suppliers.parquet")

    data = {"industry": "auto", "mu": 2, "S": S, "R": R, "step_dir": "step3",
            "folder": folder, "input_folder": inp,
            "filter_N_upstream_df": filter_df, "france": france,
            "suppliers": pd.read_parquet(folder / "suppliers.parquet")}

    # ------------------------------------------------------------------- the frame
    diff = NS["build_diffusion_frame"](data)
    assert len(diff) == R_d * R, (len(diff), R_d * R)                 # fully zero-filled
    assert (diff["upstream_sales"] >= 0).all() and (diff["upstream_sales"] == 0).any()
    chk = diff.sample(min(200, len(diff)), random_state=1)
    assert np.allclose(chk["distance"].to_numpy(),
                       D[chk["ze2010"] - 1, chk["ze2010_downstream"] - 1])
    # the sum over cells reproduces the planted intermediate share, region by region
    tot = diff.groupby("ze2010_downstream")["upstream_sales"].sum()
    assert np.allclose(tot.to_numpy(), TOTAL_INPUT_SHARE), tot
    assert diff["ze_name"].str.startswith("Zone").all()
    print("diffusion frame: zero-filled, distances by index, sales reconstructed")

    # ------------------------------------------------------------------ the summary
    summ = NS["amplification_summary"](data, radii=(100, 200), diffusion=diff)
    assert np.allclose(summ["amplification"], 1.0 + TOTAL_INPUT_SHARE)
    assert {"share_within_100km", "share_within_200km"} <= set(summ.columns)
    assert (summ["share_within_100km"] <= summ["share_within_200km"] + 1e-12).all()
    assert ((summ["share_within_200km"] >= 0) & (summ["share_within_200km"] <= 1)).all()
    # the shares are ratios of the same denominator, so they must reproduce a direct count
    for r in downstream:
        sub = diff[diff["ze2010_downstream"] == r]
        want = sub.loc[sub["distance"] <= 100, "upstream_sales"].sum() / sub["upstream_sales"].sum()
        assert abs(summ.loc[r, "share_within_100km"] - want) < 1e-12
    print("summary: D_r = 1 + upstream sales; radii nested; shares match a direct count")
    print(summ.round(3).to_string())

    # a radius beyond the country keeps everything; a radius of zero keeps only own region
    wide = NS["amplification_summary"](data, radii=(10_000,), diffusion=diff)
    assert np.allclose(wide["share_within_10000km"], 1.0)
    own = NS["amplification_summary"](data, radii=(0.0,), diffusion=diff)
    for r in downstream:
        sub = diff[(diff["ze2010_downstream"] == r) & (diff["distance"] <= 0)]
        assert set(sub["ze2010"]) == {r}          # only the shocked region is at distance 0
    print("radius limits behave (0 km = own region only, 10 000 km = everything)")

    # ------------------------------------------------------------------ the profile
    prof = NS["local_share_profile"](data, diffusion=diff)
    assert prof.index.is_monotonic_increasing
    for c in ("mean", "median", "weighted_mean", "p25", "p75"):
        assert prof[c].is_monotonic_increasing, c                    # a CDF in the radius
    assert (prof["p25"] <= prof["median"] + 1e-12).all()
    assert (prof["median"] <= prof["p75"] + 1e-12).all()
    assert prof["weighted_mean"].iloc[-1] > 0.9
    print("\nprofile is a CDF in the radius:")
    print(prof.round(3).to_string())

    # ---------------------------------------------------------------------- figures
    out = TMP / "figs"
    out.mkdir(exist_ok=True)
    NS["plot_amplification"](data, summary=summ, save_to=str(out / "amp.png"))
    NS["plot_local_share"](data, radius_km=100, summary=summ, save_to=str(out / "amp_local.png"))
    NS["plot_local_share_profile"](data, profile=prof, save_to=str(out / "amp_profile.png"))
    NS["plot_amplification_vs_local"](data, radius_km=100, summary=summ,
                                      save_to=str(out / "amp_scatter.png"))
    s2, p2 = NS["amplification_report"](data, radii=(200, 100), out_folder=str(out), show=False)
    assert "share_within_200km" in s2.columns

    # the radius must actually be free: a different one gives a different column and number
    s3 = NS["amplification_summary"](data, radii=(50,), diffusion=diff)
    assert "share_within_50km" in s3.columns
    assert s3["share_within_50km"].mean() < summ["share_within_100km"].mean()
    print("\nradius is a free parameter (50 km keeps less than 100 km)")

    # a missing artefact must say what is missing, and a wrong column too
    for kw, exc in ((dict(suppliers=None), FileNotFoundError),):
        try:
            NS["build_diffusion_frame"](dict(data, **kw))
        except exc as e:
            print("expected:", str(e)[:60], "...")
        else:
            raise AssertionError("should have raised")
    try:
        NS["build_diffusion_frame"](data, value_col="nope")
    except KeyError as e:
        print("expected on a bad value column:", str(e)[:50], "...")
    else:
        raise AssertionError("should have raised")

    print("\nALL OK")


def gate_io_benchmark():
    """The TES parser, the column normalisation, and the Leontief algebra."""
    rng = np.random.default_rng(3)

    S, R, R_d = 4, 12, 4
    inp = TMP / "synthinput_io"
    folder = TMP / "synthrun_io"
    (folder / "step1").mkdir(parents=True, exist_ok=True)
    inp.mkdir(parents=True, exist_ok=True)

    coords = rng.uniform(0, 400, (R, 2))
    D = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(D, 0.0)
    np.save(inp / "distances.npy", D)

    # --- a TES block: 8 sectors, of which 4 are the modelled ones + the downstream ---
    SECTORS = [f"C{10 + i}A" for i in range(8)]
    DOWN = "C29A"
    SECTORS[5] = DOWN
    MODELLED = [SECTORS[i] for i in (0, 1, 2, 3)]
    dom = pd.DataFrame(rng.gamma(2.0, 50.0, (8, 8)), index=SECTORS, columns=SECTORS)
    imp = pd.DataFrame(rng.gamma(1.0, 20.0, (8, 8)), index=SECTORS, columns=SECTORS)
    for name, M in (("dom", dom), ("imp", imp)):
        out = M.copy()
        out.to_csv(inp / f"TES_{name}.csv", sep=";", header=False,
                   float_format="%.4f", decimal=",")

    # --- a firm-level economy whose intermediate share is exactly 0.8 ---------------
    LABOR_SHARE = 0.2
    rows, siren = [], 0
    for r in range(1, R_d + 1):
        for s in range(1, S + 1):
            for rho in range(1, 20 + 1):
                siren += 1
                rows.append({"SIREN": siren, "A129": s, "ze2010": int(rng.integers(1, R + 1)),
                             "ze2010_downstream": r, "share": (1 - LABOR_SHARE) / (S * 20),
                             "downstream_purchase": 1.0, "intermediate_derivative": 0.0,
                             "productivity": 1.0, "sample_weight": 0.05})
    pd.DataFrame(rows).to_parquet(folder / "suppliers.parquet")

    codes = [f"{1000 + i:04d}" for i in range(R)]
    data = {"industry": "auto", "mu": 1, "S": S, "R": R, "step_dir": "step1",
            "folder": folder, "input_folder": inp, "d": DOWN,
            "agg_labor_share": LABOR_SHARE,
            "sector_names": MODELLED,
            "filter_N_upstream_df": pd.DataFrame({"ze2010": codes * S,
                                                  "A129": np.repeat(MODELLED, R)}),
            "france": pd.DataFrame({"ze2010": codes,
                                    "ze2010_name": [f"Zone {c}" for c in codes]}),
            "suppliers": pd.read_parquet(folder / "suppliers.parquet")}

    # ------------------------------------------------------------------ the parser
    io = NS["read_io_table"]("dom", inp)
    assert set(io.columns) == {"A129_1", "A129_2", "value"}
    piv = io.pivot(index="A129_1", columns="A129_2", values="value")
    assert np.allclose(piv.loc[SECTORS, SECTORS].to_numpy(), dom.to_numpy(), atol=1e-4)
    print("read_io_table: comma decimals and the (supplying x using) orientation round-trip")

    X = NS["io_flow_matrix"](data, kinds=("dom", "imp"))
    assert np.allclose(X.loc[SECTORS, SECTORS].to_numpy(),
                       (dom + imp).to_numpy(), atol=1e-3)
    print("io_flow_matrix: dom + imp add")

    # --------------------------------------------------------- technical coefficients
    A = NS["io_technical_coefficients"](NS["io_flow_matrix"](data), 0.6)
    assert np.allclose(A.sum(axis=0).to_numpy(), 0.6)          # every column sums to m
    Ap = NS["io_technical_coefficients"](NS["io_flow_matrix"](data),
                                         pd.Series(0.5, index=SECTORS))
    assert np.allclose(Ap.sum(axis=0).to_numpy(), 0.5)
    try:
        NS["io_technical_coefficients"](NS["io_flow_matrix"](data), 1.2)
    except ValueError as e:
        print("expected on m >= 1:", str(e)[:52], "...")
    else:
        raise AssertionError("should have raised")
    print("io_technical_coefficients: columns normalised to the intermediate share")

    # ------------------------------------------------------------------- Leontief
    mult = NS["leontief_multipliers"](A, max_rounds=3)
    # a column-stochastic-to-m matrix has EVERY multiplier equal to 1/(1-m) exactly:
    # sum_i (A^k)_{ij} = m^k, so the column sum of the inverse is the geometric series
    assert np.allclose(mult["total"].to_numpy(), 1.0 / (1.0 - 0.6)), mult["total"]
    assert np.allclose(mult["rounds_1"].to_numpy(), 1.0 + 0.6)
    assert np.allclose(mult["rounds_2"].to_numpy(), 1.0 + 0.6 + 0.36)
    assert (mult["rounds_1"] < mult["rounds_2"]).all() and (mult["rounds_2"] < mult["total"]).all()
    print("leontief_multipliers: geometric series recovered exactly, rounds nested")

    # --------------------------------------------------------------- the benchmark
    tab = NS["io_amplification_benchmark"](data)
    assert list(tab.index) == ["full", "subset"]
    assert tab.loc["full", "n_sectors"] == 8
    assert tab.loc["subset", "n_sectors"] == len(MODELLED) + 1
    # the model's D_r - 1 must equal the planted intermediate share, and the table must
    # say so against the labour-share moment
    assert abs(tab.attrs["implied_intermediate_share"] - (1 - LABOR_SHARE)) < 1e-9
    # coverage: the modelled sectors' share of the downstream column, from shares alone
    col = NS["io_downstream_column"](data)
    want_cov = dom.loc[MODELLED, DOWN].sum() / dom[DOWN].sum()
    assert abs(tab.attrs["coverage_modelled_dom"] - want_cov) < 1e-6, (
        tab.attrs["coverage_modelled_dom"], want_cov)
    assert abs(tab.attrs["implied_total_intermediate_share"]
               - (1 - LABOR_SHARE) / want_cov) < 1e-6
    assert col.loc[MODELLED, "modelled"].all() and not col.loc[[SECTORS[7]], "modelled"].any()
    assert abs(col["share_of_total"].sum() - 1.0) < 1e-9
    assert abs(col.loc[col["modelled"], "pi_s_io"].sum() - 1.0) < 1e-9
    # the model's simulated sector split is uniform here by construction
    assert np.allclose(col.loc[MODELLED, "share_model"].to_numpy(), 1.0 / len(MODELLED))
    assert tab.attrs["scalar_m"]
    print("coverage + downstream column: shares add up, model split recovered")
    assert abs(tab.attrs["data_intermediate_share"] - (1 - LABOR_SHARE)) < 1e-12
    assert np.allclose(tab["leontief_total"], 1.0 / LABOR_SHARE)     # m = 1 - 0.2
    assert (tab["captured_of_aggregate"] > 0).all()
    print(tab.round(3).to_string())

    # a downstream industry absent from the table must be named, not silently dropped
    try:
        NS["io_amplification_benchmark"](dict(data, d="ZZZZ"), verbose=False)
    except KeyError as e:
        print("expected on an unknown downstream industry:", str(e)[:55], "...")
    else:
        raise AssertionError("should have raised")

    # and a missing table must say where it looked
    try:
        NS["io_flow_matrix"](dict(data, input_folder=TMP / "nowhere",
                                  folder=TMP / "nowhere"))
    except FileNotFoundError as e:
        print("expected when the TES table is absent:", str(e)[:55], "...")
    else:
        raise AssertionError("should have raised")

    out = TMP / "figs"
    out.mkdir(exist_ok=True)
    NS["plot_io_benchmark"](data, benchmark=tab, save_to=str(out / "io_benchmark.png"))
    print("\nALL OK")


if __name__ == "__main__":
    for name, gate in (("identification", gate_identification),
                       ("untargeted moment", gate_untargeted),
                       ("comparative advantage", gate_comparative_advantage),
                       ("amplification", gate_amplification),
                       ("IO benchmark", gate_io_benchmark)):
        print("\n" + "=" * 70)
        print(f"GATE: {name}")
        print("=" * 70)
        gate()
    print("\nALL GATES PASSED")
