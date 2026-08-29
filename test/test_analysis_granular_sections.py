"""
Gates for the sections added to `analysis_granular.ipynb`:
  1. Identification / sensitivity  — the variety-count columns and the noise mask,
  2. Untargeted moment             — the PPML distance elasticity of comovement,
  2b. Untargeted count moment      — G_s(K) beyond the targeted K = 0,
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
    assert tab.iloc[1] < 0, tab
    assert abs(dec["gap"]) < 0.25, dec["gap"]
    assert dec["table"].loc[2, "n_obs"] < dec["table"].loc[0, "n_obs"]   # positives only
    print("decomposition: extensive margin negative, identity closes to", round(dec["gap"], 4))

    # the served indicator the extensive-margin regression runs on
    assert set(np.unique(panel["served"])) <= {0.0, 1.0}
    assert np.allclose(panel["served"].to_numpy(), (panel["a_ir"] > 0).astype(float))

    # a second industry so the joint table/figure paths are exercised
    data2 = dict(data, industry="aero")
    res_aero = NS["estimate_untargeted_moment"](data2, panel=panel)

    # the three fixed-effect specifications, and the map onto the reduced form's scale
    assert set(res_auto["specs"]) == set(NS["UNTARGETED_FE_SPECS"])
    for label, sp in res_auto["specs"].items():
        assert abs(sp["dg"] - NS["delta_over_gamma_from_eta"](sp["eta"])) < 1e-12
        # the map is monotone and compressive, and bounded by 1/mean(log d)
        assert abs(sp["dg"]) < abs(sp["eta"]) + 1e-12
        assert abs(sp["dg"]) < 1.0 / NS["EMPIRICAL_MEAN_LOG_D"]
    assert res_auto["eta"] == res_auto["specs"]["sector x buyer region"]["eta"]
    # the round trip through the two maps must return the elasticity it started from
    e = res_auto["eta"]
    assert abs(NS["eta_from_delta_over_gamma"](NS["delta_over_gamma_from_eta"](e)) - e) < 1e-10
    print("delta/gamma map: monotone, compressive, bounded, and invertible")

    summ = NS["untargeted_summary"]([res_auto, res_aero])
    print("\n", summ.to_string())
    assert "delta/gamma (data)" in summ.columns and summ["eta (model)"].notna().all()
    assert len(summ) == 2 * len(NS["UNTARGETED_FE_SPECS"])
    assert list(summ.index.names) == ["industry", "fixed effect"]

    out = TMP / "figs"
    out.mkdir(exist_ok=True)
    NS["plot_untargeted_moment"]([res_auto, res_aero], save_to=str(out / "untargeted.png"))
    NS["plot_untargeted_moment"]([res_auto, res_aero], scale="eta")
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
    head = "PPML, sector x buyer-region FE (reported eta)"
    assert head in ladder.index
    assert abs(ladder.loc[head, "eta"] - res_auto["eta"]) < 1e-9
    # every rung is also reported on the reduced form's scale, through the same map
    assert abs(ladder.loc[head, "delta_over_gamma"]
               - NS["delta_over_gamma_from_eta"](res_auto["eta"])) < 1e-12
    assert abs(ladder.loc["Level-linear on the simulated panel, delta/gamma",
                          "delta_over_gamma"] - lin["delta_over_gamma"]) < 1e-12
    assert ladder.loc["DATA delta/gamma (Table 3)", "delta_over_gamma"] == \
        NS["EMPIRICAL_DELTA_OVER_GAMMA"]["auto"]["estimate"]
    # the within-supplier rung must exist and differ from the cross-supplier one: the gap
    # between them IS the composition channel the comparison hinges on
    assert "PPML, supplier FE (within-supplier)" in ladder.index
    assert abs(ladder.loc["PPML, supplier FE (within-supplier)", "eta"]
               - ladder.loc[head, "eta"]) > 1e-6
    # every rung is on the reduced-form scale except -theta*alpha, which needs
    # best_params (absent from this synthetic panel) and is reported as missing
    assert np.isfinite(ladder["delta_over_gamma"]
                       .drop("-theta * alpha (trade-cost elasticity)")).all()
    NS["plot_untargeted_ladder"](ladder, "auto", save_to=str(out / "ladder.png"))
    NS["plot_untargeted_ladder"](ladder, "auto", scale="eta")
    print("specification ladder: every rung computed and the reported eta reproduced")

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

def gate_untargeted_counts():
    """
    G_s(K) beyond K = 0: the K* selection rule, and the count distribution the model
    predicts once N_hat has been spent on G_s(0).

    Three risks, all silent. The *selection*: the empirical curve is a step function, so
    a crossing K can sit on a plateau where the data have no mass, and a moment placed
    there is a comparison against granularity. The *estimator*: the hypergeometric
    estimator must be unbiased for Pr(Bin(N,q) <= K) and must collapse at K = 0 onto
    `gbar_cell`, the estimator the criterion matched block 6 with — otherwise the K = 0
    column of the model curve is not the fitted moment and nothing above it is
    comparable either. The *inversion*: the implied variety count must reproduce N_hat
    when it is asked the K = 0 question, or the N ratio is measuring the solver.
    """
    rng = np.random.default_rng(23)

    # ---------------------------------------------------------------------------
    # 1. The selection rule, on curves built by hand so K* is known by inspection
    # ---------------------------------------------------------------------------
    K = np.arange(0, 6)
    #            K = 0    1     2     3     4     5
    G = np.array([[0.40, 0.55, 0.72, 0.85, 0.95, 1.00],   # target .70 -> K* = 2
                  [0.20, 0.20, 0.30, 0.65, 0.90, 1.00],   # target .60 -> K* = 3
                  [0.50, 0.75, 0.75, 0.90, 0.97, 1.00],   # target .75, but G(1)=G(2):
                                                          #   crossing at K=1 has mass
                                                          #   (0.50 -> 0.75) so K* = 1
                  [0.10, 0.12, 0.15, 0.18, 0.20, 0.22],   # never reaches .55 -> none
                  [1.00, 1.00, 1.00, 1.00, 1.00, 1.00]])  # degenerate: no non-empty cell
    names = ["a", "b", "c", "d", "e"]
    sel = NS["select_untargeted_K"]({"K": K, "G": G}, sector_names=names)
    print(sel.to_string())
    assert list(sel["K_star"]) == [2, 3, 1, -1, -1], list(sel["K_star"])
    assert sel.loc["d", "status"].startswith("not reached")
    assert sel.loc["e", "status"].startswith("degenerate")
    # every selected K must clear the halfway threshold AND sit on a step with mass
    for name in ("a", "b", "c"):
        s = names.index(name)
        j = int(sel.loc[name, "K_star"])
        assert G[s, j] >= sel.loc[name, "target"] - 1e-12
        assert G[s, j] > G[s, j - 1] + 1e-12
        assert sel.loc[name, "frac_of_nonempty"] >= 0.5 - 1e-12
    # a curve whose crossing lands EXACTLY on a plateau must advance to the next rung
    G_flat = np.array([[0.40, 0.70, 0.70, 0.88, 0.95, 1.00]])   # target .70, G(1)=G(2)
    sel_flat = NS["select_untargeted_K"]({"K": K, "G": G_flat}, sector_names=["f"],
                                         verbose=False)
    assert int(sel_flat["K_star"].iloc[0]) == 1, sel_flat        # K=1 itself has mass
    G_flat2 = np.array([[0.70, 0.70, 0.70, 0.88, 0.95, 1.00]])   # target .85: K=3
    sel_flat2 = NS["select_untargeted_K"]({"K": K, "G": G_flat2}, sector_names=["f"],
                                          verbose=False)
    assert int(sel_flat2["K_star"].iloc[0]) == 3, sel_flat2
    print("selection: median of the non-empty mass, never on a plateau, "
          "truncated and degenerate curves refused")

    # ... and on a SPARSE, per-sector support, which is what G_K.csv actually gives:
    # a sector's curve is reported at the counts it realises, so the union grid skips
    # integers and "G_s(K-1)" means the sector's previous RUNG, not K-1.
    K_sp = np.array([0, 1, 2, 5, 9, 40])
    nan = np.nan
    G_sp = np.array([[0.30, nan, 0.55, 0.72, nan, 1.00],   # support {0,2,5,40}, tgt .65
                     [0.20, 0.20, nan, nan, 0.70, 1.00]])  # support {0,1,9,40}, tgt .60
    sel_sp = NS["select_untargeted_K"]({"K": K_sp, "G": G_sp}, sector_names=["p", "q"],
                                       verbose=False)
    # p: 0.55 at K=2 is below .65, 0.72 at K=5 clears it and rises -> K* = 5
    # q: K=1 is a plateau (0.20 -> 0.20) and below target anyway; K=9 rises -> K* = 9
    assert list(sel_sp["K_star"]) == [5, 9], list(sel_sp["K_star"])
    assert list(sel_sp["K_max_available"]) == [40, 40]
    print("selection on a sparse per-sector support: rungs read on the sector's own K")

    # ---------------------------------------------------------------------------
    # 2. The estimator. At K = 0 it must BE `gbar_cell`, and it must be unbiased for
    #    Pr(Bin(N,q) <= K) — which is the whole claim the section rests on.
    # ---------------------------------------------------------------------------
    def gbar_cell(k, m, n):
        """model_CP.jl's own combinatorial estimator, transcribed."""
        if n <= 0:
            return 1.0
        if n > m - k:
            return 0.0
        from math import lgamma
        lg = lambda x: lgamma(x + 1)
        return float(np.exp(lg(m - k) - lg(m - k - n) - lg(m) + lg(m - n)))

    m, N = 60, 9
    ks = np.arange(0, m + 1)
    cdf = NS["count_cdf_cells"](ks, m, N, 4)
    assert cdf.shape == (m + 1, 5)
    ref0 = np.array([gbar_cell(int(k), m, N) for k in ks])
    assert np.abs(cdf[:, 0] - ref0).max() < 1e-12, np.abs(cdf[:, 0] - ref0).max()
    print("K = 0 column reproduces gbar_cell to", f"{np.abs(cdf[:, 0] - ref0).max():.2e}")
    # a CDF: non-decreasing in K, inside [0, 1]
    assert (np.diff(cdf, axis=1) >= -1e-12).all()
    assert (cdf >= -1e-12).all() and (cdf <= 1 + 1e-12).all()
    # k = 0 (the cell wins nothing) must give probability one at every K
    assert np.allclose(cdf[0], 1.0)

    # unbiasedness: E_k[estimator] must equal the Binomial CDF at the true q. The
    # weights are formed in LOG space: an exact C(4000, k) does not fit in a float.
    def binom_pmf(n, q):
        kk = np.arange(n + 1)
        lg = NS["_log_factorials"](n)
        return np.exp(lg[n] - lg[kk] - lg[n - kk] + kk * np.log(q)
                      + (n - kk) * np.log1p(-q))

    for q in (0.05, 0.2, 0.5):
        w = binom_pmf(m, q)
        got = w @ cdf                                     # E over k ~ Bin(m, q)
        want = np.cumsum(binom_pmf(N, q)[:5])
        err = np.abs(got - want).max()
        print(f"  unbiased at q = {q}: max |E[estimator] - Pr(Bin(N,q) <= K)| = {err:.2e}")
        assert err < 1e-12, err
    # the plug-in is biased at this m and must converge to the same thing as m grows
    q0 = 0.2
    w0 = binom_pmf(m, q0)
    want0 = np.cumsum(binom_pmf(N, q0)[:5])
    plug = NS["count_cdf_cells"](ks, m, N, 4, estimator="plugin")
    d_small = np.abs((w0 @ plug) - want0).max()
    m_big = 4000
    ks_b = np.arange(0, m_big + 1)
    plug_b = NS["count_cdf_cells"](ks_b, m_big, N, 4, estimator="plugin")
    d_big = np.abs((binom_pmf(m_big, q0) @ plug_b) - want0).max()
    print(f"  plug-in bias: {d_small:.2e} at m = {m}, {d_big:.2e} at m = {m_big}")
    assert d_big < d_small
    # N > m is undefined for the subset estimator and must say so
    try:
        NS["count_cdf_cells"](ks, m, m + 1, 1)
    except ValueError as e:
        assert "exceeds the draw count" in str(e)
    else:
        raise AssertionError("N > m should have raised")

    # ---------------------------------------------------------------------------
    # 3. End to end on a synthetic run tree with a PLANTED count distribution
    # ---------------------------------------------------------------------------
    S, R, R_d, m = 4, 40, 10, 120
    folder = TMP / "synthrun_cnt"
    inp = TMP / "synthinput_cnt"
    (folder / "step1").mkdir(parents=True, exist_ok=True)
    inp.mkdir(parents=True, exist_ok=True)

    CELL_MASK = np.zeros((S, R), dtype=bool)
    for s in range(S):
        CELL_MASK[s, rng.choice(R, 25, replace=False)] = True
    N_true = np.array([6, 11, 4, 9])
    q = np.zeros((S, R))
    linkages = np.zeros((S, m, R), dtype=bool)
    for s in range(S):
        cells = np.flatnonzero(CELL_MASK[s])
        q[s, cells] = rng.uniform(0.005, 0.25, cells.size)
        for l in cells:
            linkages[s, :, l] = rng.random(m) < q[s, l]
    np.save(folder / "step1" / "suppliers.npy", linkages)
    k_true = linkages.sum(axis=1)

    # the empirical curve is the model's own, at N_true, so a correct pipeline must come
    # back with N implied by K* equal to N_hat equal to N_true
    K_max = 8
    G_plant = np.full((S, K_max + 1), np.nan)
    for s in range(S):
        G_plant[s] = NS["count_cdf_cells"](k_true[s][CELL_MASK[s]], m, int(N_true[s]),
                                           K_max).mean(axis=0)
    sectors = ["101", "205", "310", "422"]
    # N_supplier_s sets the model's own bounds N_LO = ceil(N_obs / R_d) = 2 and
    # N_HI = 20, which must bracket every planted N_true for the bisection to be free.
    # The K grid is written SPARSE and PER SECTOR, as the real G_K.csv is: each sector
    # gets its own subset of the integers, so the union grid skips values and no column
    # index can be assumed to equal its K.
    K_support = {0: [0, 1, 2, 4, 8], 1: [0, 2, 3, 5, 8], 2: [0, 1, 3, 6],
                 3: [0, 1, 2, 3, 8]}
    rows = [{"group": s + 1, "A129": sectors[s], "G": G_plant[s, kk], "K": kk,
             "N_supplier_s": 20} for s in range(S) for kk in K_support[s]]
    pd.DataFrame(rows).to_csv(inp / "G_K.csv", index=False)
    K_union = np.array(sorted({k for v in K_support.values() for k in v}))

    data = {"industry": "auto", "mu": 1, "S": S, "R": R, "n_AA": R_d,
            "step_dir": "step1", "folder": folder, "input_folder": inp,
            "CELL_MASK": CELL_MASK, "sector_names": sectors, "sector_codes": sectors,
            "granular_diagnostics": {"N_hat": N_true.astype(float),
                                     "q_hat": (k_true / m)[CELL_MASK.T.T].ravel()},
            "simulated_moments_dict": {"G0": G_plant[:, 0]}}
    # q_hat is enumerated in Julia's column-major findall order over (S, R)
    cells_rs = np.argwhere(CELL_MASK.T)
    data["granular_diagnostics"]["q_hat"] = (k_true / m)[cells_rs[:, 1], cells_rs[:, 0]]

    curve = NS["read_count_cdf"](data)
    assert list(curve["K"]) == list(K_union), (curve["K"], K_union)
    assert curve["G"].shape == (S, K_union.size)
    for s in range(S):
        on = np.isin(K_union, K_support[s])
        assert np.allclose(curve["G"][s, on], G_plant[s, K_union[on]])
        assert np.isnan(curve["G"][s, ~on]).all()      # absent rungs stay absent
    assert curve["sector_codes"] == sectors
    print("\nG_K.csv round-trip: the sparse per-sector support, in the model's order")

    # the model curve must be evaluated ON THAT GRID, column for column — reading it as
    # 0..K_max would silently misalign every point past the first gap
    G_on_grid = NS["model_count_cdf"](data, K_union, N_true)
    G_dense = NS["model_count_cdf"](data, int(K_union[-1]), N_true)
    assert G_on_grid.shape == (S, K_union.size)
    assert np.abs(G_on_grid - G_dense[:, K_union]).max() < 1e-15
    print("model curve read on the empirical grid, not on 0..K_max")

    res = NS["untargeted_count_moment"](data)
    tab = res["table"]
    # the two curves must be on the same axis, which is what the runtime failure was
    assert res["G_emp"].shape == res["G_model"].shape == (S, K_union.size)
    assert res["K"].size == res["G_model"].shape[1]
    # the K = 0 column IS block 6 — computed here from the counts, not read off the
    # moment vector — so it must equal what the criterion reported
    assert np.abs(res["G_model"][:, 0] - G_plant[:, 0]).max() < 1e-12
    # and on every reported rung the model must land on the (planted) data
    for s in range(S):
        on = np.isin(K_union, K_support[s])
        assert np.abs(res["G_model"][s, on] - G_plant[s, K_union[on]]).max() < 1e-12
    # the bisection re-run here must return the planted variety count, unclamped
    assert (res["N_hat_recomputed"] == N_true).all(), res["N_hat_recomputed"]
    assert set(res["clamp_recomputed"]) == {"none"}, res["clamp_recomputed"]
    # and it must report a bound that binds rather than silently crossing it — the
    # model's N_LO = ceil(N_obs / R_d) is a theorem, not a search convention
    n_cl, cl = NS["implied_variety_count"](k_true[2][CELL_MASK[2]], m, G_plant[2, 0], 0,
                                           n_lo=int(N_true[2]) + 2, n_hi=20)
    assert (n_cl, cl) == (int(N_true[2]) + 2, "lo"), (n_cl, cl)
    # and, the data being the model's own curve, the K* inversion must return it too
    ok = tab["K*"].notna().to_numpy()
    assert ok.all(), tab["status"].to_dict()
    assert np.abs(tab.loc[ok, "gap at K*"]).max() < 1e-12
    assert (tab.loc[ok, "N implied by K*"].to_numpy() == N_true[ok]).all()
    assert np.allclose(tab.loc[ok, "N ratio"], 1.0)
    # the model's own K* must agree with the data's when the two curves coincide
    assert (tab.loc[ok, "K* (model's own)"].to_numpy()
            == tab.loc[ok, "K*"].to_numpy()).all()
    print("planted economy: N_hat, K* and the implied N all recovered exactly")

    # a MISSPECIFIED model: give it the wrong variety count and the gap must open, with
    # the implied N pointing back at the truth
    res_bad = NS["untargeted_count_moment"](data, N_hat=N_true + 3, verbose=False)
    tb = res_bad["table"]
    assert np.abs(tb.loc[ok, "gap at K*"]).max() > 1e-3
    assert (tb.loc[ok, "N implied by K*"].to_numpy() == N_true[ok]).all()
    assert (tb.loc[ok, "N ratio"] < 1).all()
    # more varieties => fewer empty cells and fewer cells below any fixed K
    assert (res_bad["G_model"][:, 0] < res["G_model"][:, 0]).all()
    print("misspecified N: the gap opens and the implied N points back at the truth")

    # the q_hat fallback: no suppliers.npy, plug-in only, and the cell enumeration must
    # be Julia's column-major findall order or the sectors get each other's cells
    folder_q = TMP / "synthrun_cnt_q"
    (folder_q / "step1").mkdir(parents=True, exist_ok=True)
    data_q = dict(data, folder=folder_q)
    k_q, m_q, _ = NS["load_supplier_counts"](data_q)
    assert m_q is None
    assert np.abs(k_q - k_true / m).max() < 1e-12, "q_hat scattered to the wrong cells"
    G_q = NS["model_count_cdf"](data_q, K_union, N_true, estimator="plugin")
    G_plug = NS["model_count_cdf"](data, K_union, N_true, estimator="plugin")
    assert np.abs(G_q - G_plug).max() < 1e-12
    # and the moment itself must run on that path, downgrading the estimator rather
    # than asking the subset estimator for an m it does not have
    res_q = NS["untargeted_count_moment"](data_q, verbose=False)
    assert res_q["estimator"] == "plugin" and res_q["m"] is None
    assert np.abs(res_q["G_model"] - G_q).max() < 1e-12
    print("q_hat fallback: cells in Julia's order, plug-in estimator only")

    summ = NS["untargeted_count_summary"]([res, dict(res, industry="aero")])
    assert list(summ.index.names) == ["industry", "sector"] and len(summ) == 2 * S

    out = TMP / "figs"
    out.mkdir(exist_ok=True)
    NS["plot_count_cdf"](res, save_to=str(out / "count_cdf.png"))
    NS["plot_count_cdf"](res, x_max=np.inf)          # the whole grid
    NS["plot_untargeted_count"](res, save_to=str(out / "count_moment.png"))

    # a missing G_K.csv must say what is missing
    try:
        NS["read_count_cdf"](dict(data, input_folder=TMP / "nowhere"))
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
                       ("untargeted count moment", gate_untargeted_counts),
                       ("comparative advantage", gate_comparative_advantage),
                       ("amplification", gate_amplification),
                       ("IO benchmark", gate_io_benchmark)):
        print("\n" + "=" * 70)
        print(f"GATE: {name}")
        print("=" * 70)
        gate()
    print("\nALL GATES PASSED")
