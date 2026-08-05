"""
Gate for the granular / attraction-area Python reporting (`analysis_granular.ipynb`).

Standalone and data-free: it writes a SYNTHETIC baseline + reporting tree with the
exact file layout `main.jl --granular=true --ca_level=aa` produces, then executes the
NOTEBOOK'S OWN code cells against it and checks the objects the reporting depends on —
the attraction-area mapping and names, the AA-level gamma aggregation and its s-major
flattening, MOMENT_MASK, the block split of `best_simulated_moments.npy`, the mu_1 /
mu_2 folder routing, the beta -> gamma -> G standard-error split, the reference-area
reconstruction in the gamma panel, and the LaTeX table's contents.

    python3 test/test_analysis_granular.py [path/to/analysis_granular.ipynb]

Runs every gate, prints PASS/FAIL per gate and exits non-zero if any failed.
Print-only otherwise; nothing in the estimation pipeline is touched.
"""
import json, os, sys, tempfile, warnings

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

RNG = np.random.default_rng(20260805)



S, R, R_D = 4, 12, 3          # sectors, ZE, downstream regions (= n_AA)
N_REG = 4
N_STAGES = 3


def build_synthetic_tree(ROOT):
    """Write the tree and return the expectations, also dumped to ROOT/expected.json."""
    BASE = os.path.join(ROOT, "baseline_test")
    RUN = os.path.join(ROOT, "reporting_test_profiled_aa_gran_pso")
    for p in (BASE, RUN, f"{RUN}/step1", f"{RUN}/step2", f"{RUN}/step2/inference",
              f"{RUN}/step3", f"{RUN}/step3/inference"):
        os.makedirs(p, exist_ok=True)

    ze = [f"{100 + i:04d}" for i in range(R)]
    sectors = [f"C{10 + s}A" for s in range(S)]

    # ---- geography: downstream regions are ZE 0, 5, 9 (sorted index order) ----
    down_idx = np.array([0, 5, 9])
    N_downstream = np.zeros(R)
    N_downstream[down_idx] = [3.0, 5.0, 2.0]

    aa_of_ze = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 0, 1])   # each ZE -> one AA
    AA_link = np.zeros((R, R_D), dtype=np.int64)
    AA_link[np.arange(R), aa_of_ze] = 1

    # ---- filter / trade: sector 3 has an EMPTY attraction area (AA 2) ---------
    filter_N = np.zeros((S, R), dtype=np.int64)
    X_rs = np.zeros((S, R))
    for s in range(S):
        keep = RNG.random(R) < 0.75
        keep[aa_of_ze == 0] = True                      # AA 0 always populated
        if s == 3:
            keep[aa_of_ze == 2] = False                 # AA 2 inactive in sector 3
        filter_N[s, keep] = 1
        supp = keep & (RNG.random(R) < 0.8)
        supp[np.flatnonzero(keep)[0]] = True            # at least one supplier
        X_rs[s, supp] = RNG.uniform(1.0, 10.0, supp.sum())
    # control cells: filter == 1 but X_rs == 0 (goods under :aa, zero observed gamma)

    domestic_share = RNG.uniform(0.55, 0.9, S)
    emp_gamma_ls = np.zeros((S, R))
    for s in range(S):
        supp = X_rs[s] > 0
        w = RNG.uniform(0.2, 1.0, supp.sum())
        emp_gamma_ls[s, supp] = w / w.sum() * domestic_share[s]

    agg_industry_share = RNG.dirichlet(np.ones(S) * 3)
    agg_labor_share = 0.42
    epsilon = -2.5
    prior_alpha = 0.31

    X_dr = pd.DataFrame({
        "ze2010": ze,
        "X_dr": np.where(N_downstream > 0, RNG.uniform(50, 500, R), 0.0),
        "downstream": N_downstream > 0,
        "downstream_region": N_downstream > 0,
    })
    X_dr["ze2010_downstream"] = np.where(X_dr["downstream"], X_dr["ze2010"], "")
    X_dr.to_csv(f"{BASE}/X_dr.csv", index=False)

    pd.DataFrame({"stats": ["epsilon", "agg_labor_share", "reg_coef", "prior_alpha",
                            "reg_coef_cloglog_1"],
                  "value": [epsilon, agg_labor_share, 0.11, prior_alpha, 0.23]}
                 ).to_csv(f"{BASE}/stats.csv", index=False)

    pd.DataFrame([{"A129": sectors[s], "ze2010": int(ze[r])}
                  for s in range(S) for r in range(R)]
                 ).to_csv(f"{BASE}/filter_N_upstream.csv", index=False)

    np.save(f"{BASE}/input_share.npy", agg_industry_share)
    np.save(f"{BASE}/emp_gamma_ls.npy", emp_gamma_ls)
    np.save(f"{BASE}/filter_N_upstream.npy", filter_N)
    np.save(f"{BASE}/X_rs.npy", X_rs)
    np.save(f"{BASE}/domestic_share.npy", domestic_share)
    np.save(f"{BASE}/N_downstream_per_region.npy", N_downstream)
    np.save(f"{BASE}/attraction_area_linkages.npy", AA_link)
    reg_coef_emp = np.array([0.35, 0.62, 0.81, 1.05])
    np.save(f"{BASE}/reg_coef_cloglog_{N_REG}.npy", reg_coef_emp)

    # ---- G_K.csv --------------------------------------------------------------
    G_target = np.array([0.30, 0.45, 0.20, 0.55])
    N_sup = np.array([40, 26, 51, 18])
    rows = []
    for s in range(S):
        for k in range(3):
            # Real column layout: group, A129, G, K, N_supplier_s — the value column
            # is `G` (older files called it `G(K)`; both are accepted).
            rows.append({"group": s + 1, "A129": sectors[s], "G": G_target[s] + 0.1 * k,
                         "K": k, "N_supplier_s": N_sup[s]})
    pd.DataFrame(rows).to_csv(f"{BASE}/G_K.csv", index=False)

    # ---- the model-side structures the loader must rebuild ---------------------
    supplier = (filter_N == 1) & (X_rs > 0)
    AA_ACTIVE = np.zeros((S, R_D), dtype=bool)
    for s in range(S):
        AA_ACTIVE[s, aa_of_ze[supplier[s]]] = True
    CELL_MASK = (filter_N == 1) & AA_ACTIVE[:, aa_of_ze]

    emp_gamma_aa = np.zeros((S, R_D))
    for s in range(S):
        np.add.at(emp_gamma_aa[s], aa_of_ze[CELL_MASK[s]], emp_gamma_ls[s][CELL_MASK[s]])

    T_REF = np.full(S, -1)
    for s in range(S):
        act = np.flatnonzero(AA_ACTIVE[s])
        T_REF[s] = act[np.argmax(emp_gamma_aa[s, act])]

    emp_pi_r = X_dr["X_dr"].values[N_downstream != 0]
    emp_pi_r = emp_pi_r / emp_pi_r.sum()

    # ---- the SIMULATED moment vector, with a KNOWN answer ---------------------
    # gamma: per sector, an active-only split of domestic_share[s] -> the
    # reconstruction  gamma_ref = c_s - sum_{a != ref}  must return sim[s, ref] exactly.
    sim_gamma_aa = np.zeros((S, R_D))
    for s in range(S):
        act = np.flatnonzero(AA_ACTIVE[s])
        w = RNG.uniform(0.3, 1.0, act.size)
        sim_gamma_aa[s, act] = w / w.sum() * domestic_share[s]

    sim_labor = np.array([agg_labor_share * 1.05])
    sim_industry = agg_industry_share * RNG.uniform(0.9, 1.1, S)
    sim_pi = emp_pi_r * RNG.uniform(0.9, 1.1, R_D)
    sim_reg = reg_coef_emp * RNG.uniform(0.8, 1.2, N_REG)
    sim_G0 = np.clip(G_target + RNG.normal(0, 0.05, S), 0, 1)

    final = np.concatenate([sim_labor, sim_industry, sim_pi, sim_reg,
                            sim_gamma_aa.ravel(), sim_G0])
    stages = np.column_stack([final * (0.7 + 0.15 * k) for k in range(N_STAGES - 1)] + [final])
    np.save(f"{RUN}/step1/best_simulated_moments.npy", stages * 0.95)
    np.save(f"{RUN}/step3/best_simulated_moments.npy", stages)
    n_par = 1 + S + R_D + 1 + int(AA_ACTIVE.sum())
    np.save(f"{RUN}/step1/best_parameters_list.npy", RNG.random((n_par, N_STAGES)))
    np.save(f"{RUN}/step3/best_parameters_list.npy", RNG.random((n_par, N_STAGES)))

    np.save(f"{RUN}/n_reg_coef.npy", np.array(N_REG))
    np.save(f"{RUN}/n_tau.npy", np.array(1))

    # ---- inference artefacts (beta -> gamma -> G ordering) --------------------
    gam_free = AA_ACTIVE.copy()
    for s in range(S):
        gam_free[s, T_REF[s]] = False
    n_gb = N_REG + int(gam_free.sum()) + S
    A = RNG.normal(size=(n_gb, n_gb)) * 0.05
    Sigma_data = A @ A.T + np.diag(RNG.uniform(0.001, 0.01, n_gb))
    np.save(f"{RUN}/step2/Sigma_data.npy", Sigma_data)
    np.save(f"{RUN}/step2/Sigma_sim.npy", Sigma_data * 0.3)
    for st in ("step2", "step3"):
        np.save(f"{RUN}/{st}/inference/se_moments_fitted.npy",
                np.sqrt(np.diag(Sigma_data)) * (0.5 if st == "step2" else 0.4))
        np.savez(f"{RUN}/{st}/granular_diagnostics.npz",
                 N_hat=np.array([12.0, 7.0, 21.0, 4.0]),
                 N_LO=np.array([14.0, 9.0, 17.0, 6.0]),
                 N_HI=N_sup.astype(float),
                 clamped=np.array([0.0, 0.0, 0.0, -1.0]),
                 G0_fit=sim_G0, G0_target=G_target,
                 N_count=np.array([690.0, 120.0, 55.0, 33.0]),
                 q_hat=RNG.uniform(0.01, 0.3, int(CELL_MASK.sum())),
                 b_logz=np.array([-1.768]), EK=RNG.uniform(0.1, 3, S))

    # ---- the expectations the test checks against ----------------------------
    expected = {
        "S": S, "R": R, "n_AA": R_D, "n_coef": N_REG, "n_gb": n_gb,
        "aa_names": [ze[i] for i in down_idx],
        "sector_names": sectors,
        "aa_of_ze": aa_of_ze.tolist(),
        "AA_ACTIVE": AA_ACTIVE.tolist(),
        "CELL_MASK": CELL_MASK.tolist(),
        "T_REF_AA": T_REF.tolist(),
        "emp_gamma_aa": emp_gamma_aa.tolist(),
        "sim_gamma_aa": sim_gamma_aa.tolist(),
        "emp_pi_r": emp_pi_r.tolist(),
        "reg_coef": reg_coef_emp.tolist(),
        "G_target": G_target.tolist(),
        "N_supplier_s": N_sup.astype(float).tolist(),
        "sim_final": {"labor": sim_labor.tolist(), "industry": sim_industry.tolist(),
                      "pi": sim_pi.tolist(), "reg": sim_reg.tolist(), "G0": sim_G0.tolist()},
        "domestic_share": domestic_share.tolist(),
        "agg_labor_share": agg_labor_share,
        "n_gamma_kept": int(gam_free.sum()),
    }
    with open(f"{ROOT}/expected.json", "w") as f:
        json.dump(expected, f)
    print(f"synthetic tree: {ROOT}  (S={S}, R={R}, n_AA={R_D}, n_gb={n_gb}, gamma kept={int(gam_free.sum())})")
    return expected


def run_checks(ns, ROOT, exp):
    ok, fail = [], []

    def check(name, cond, detail=""):
        (ok if cond else fail).append(name)
        print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {detail}" if detail and not cond else ""))

    def close(a, b, tol=1e-12):
        return np.allclose(np.asarray(a, float), np.asarray(b, float), rtol=0, atol=tol)

    print("\n=== 1. loader: structure ===")
    d2 = ns["load_granular_data"]("test", mu=2, base=ROOT)
    check("run folder resolved", d2["folder"].name == "reporting_test_profiled_aa_gran_pso")
    check("S / R / n_AA", (d2["S"], d2["R"], d2["n_AA"]) == (exp["S"], exp["R"], exp["n_AA"]))
    check("attraction-area names from X_dr.query('downstream').ze2010_downstream",
          d2["aa_names"] == exp["aa_names"], f'{d2["aa_names"]} vs {exp["aa_names"]}')
    check("sector names (sorted A129)", d2["sector_names"] == exp["sector_names"])
    check("aa_of_ze", list(d2["aa_of_ze"]) == exp["aa_of_ze"])
    check("AA_ACTIVE", (d2["AA_ACTIVE"] == np.array(exp["AA_ACTIVE"])).all())
    check("CELL_MASK", (d2["CELL_MASK"] == np.array(exp["CELL_MASK"])).all())
    check("reference AA per sector", list(d2["T_REF_AA"]) == exp["T_REF_AA"])

    print("\n=== 2. empirical moments ===")
    check("gamma aggregated to attraction areas",
          close(d2["empirical_moments_dict"]["gamma_aa"], exp["emp_gamma_aa"]))
    check("gamma rows sum to the domestic share (active cells only)",
          close(np.array(exp["emp_gamma_aa"]).sum(1), exp["domestic_share"], 1e-12))
    check("pi_r", close(d2["empirical_moments_dict"]["emp_pi_r"], exp["emp_pi_r"]))
    check("reg_coef target", close(d2["empirical_moments_dict"]["reg_coef"], exp["reg_coef"]))
    check("G0 target from G_K.csv (K = 0 rows)",
          close(d2["empirical_moments_dict"]["G0"], exp["G_target"]))
    check("labor share", close(d2["empirical_moments_dict"]["agg_labor_share"],
                               [exp["agg_labor_share"]]))

    print("\n=== 3. moment mask (must reproduce load_parameters.jl) ===")
    mask, S, nAA = d2["moment_mask"], d2["S"], d2["n_AA"]
    off_gam = 1 + S + len(exp["emp_pi_r"]) + exp["n_coef"]
    gam_mask = mask[off_gam:off_gam + S * nAA].reshape(S, nAA)
    free = np.array(exp["AA_ACTIVE"])
    for s in range(S):
        free[s, exp["T_REF_AA"][s]] = False
    check("first industry share dropped", not mask[1])
    check("first pi_r dropped", not mask[1 + S])
    check("gamma mask = active & non-reference", (gam_mask == free).all())
    check("kept gamma count", int(gam_mask.sum()) == exp["n_gamma_kept"])
    check("G0 block never masked", mask[-S:].all())
    check("kept total = n_gb + labor + industry + pi_r pieces",
          int(mask.sum()) == 1 + (S - 1) + (len(exp["emp_pi_r"]) - 1) + exp["n_gb"])

    print("\n=== 4. simulated moments: block split ===")
    sim = d2["simulated_moments_dict"]
    check("labor", close(sim["agg_labor_share"], exp["sim_final"]["labor"]))
    check("industry", close(sim["agg_industry_share"], exp["sim_final"]["industry"]))
    check("pi_r", close(sim["emp_pi_r"], exp["sim_final"]["pi"]))
    check("reg_coef", close(sim["reg_coef"], exp["sim_final"]["reg"]))
    check("gamma (S x n_AA, s-major)", close(sim["gamma_aa"], exp["sim_gamma_aa"]))
    check("G0", close(sim["G0"], exp["sim_final"]["G0"]))

    d1 = ns["load_granular_data"]("test", mu=1, base=ROOT)
    check("mu = 1 reads step1 moments",
          close(d1["simulated_moments_dict"]["reg_coef"],
                np.array(exp["sim_final"]["reg"]) * 0.95))
    check("mu = 1 takes its SEs from step2/inference, mu = 2 from step3/inference",
          not close(d1["se_simulated"]["G0"], d2["se_simulated"]["G0"]))
    check("mu = 1 / mu = 2 folder routing",
          (d1["step_dir"], d1["inference_step"], d2["step_dir"], d2["inference_step"])
          == ("step1", "step2", "step3", "step3"))
    try:
        ns["load_granular_data"]("test", mu=3, base=ROOT)
        check("bad mu rejected", False)
    except ValueError:
        check("bad mu rejected", True)

    print("\n=== 5. standard errors (beta -> gamma -> G split) ===")
    Sig = np.load(os.path.join(ROOT, "reporting_test_profiled_aa_gran_pso", "step2", "Sigma_data.npy"))
    sd = np.sqrt(np.diag(Sig))
    check("empirical SE, beta block", close(d2["se_empirical"]["reg_coef"], sd[:exp["n_coef"]]))
    check("empirical SE, G block", close(d2["se_empirical"]["G0"], sd[-S:]))
    check("empirical SE, gamma block length",
          d2["se_empirical"]["gamma_aa"].size == exp["n_gamma_kept"])
    check("simulated (fitted) SE, beta block",
          close(d2["se_simulated"]["reg_coef"], sd[:exp["n_coef"]] * 0.4))
    check("simulated (fitted) SE, G block", close(d2["se_simulated"]["G0"], sd[-S:] * 0.4))

    print("\n=== 6. gamma scatter: reference reconstruction ===")
    x_free, y_free, x_ref, y_ref = ns["gamma_aa_points"](d2)
    emp_aa, sim_aa = np.array(exp["emp_gamma_aa"]), np.array(exp["sim_gamma_aa"])
    check("free points = active & non-reference", x_free.size == exp["n_gamma_kept"])
    check("free points carry the right values",
          close(sorted(x_free), sorted(emp_aa[free])) and close(sorted(y_free), sorted(sim_aa[free])))
    ref_x_expected = [emp_aa[s, exp["T_REF_AA"][s]] for s in range(S) if free[s].any()]
    ref_y_expected = [sim_aa[s, exp["T_REF_AA"][s]] for s in range(S) if free[s].any()]
    check("reference x = observed reference share", close(sorted(x_ref), sorted(ref_x_expected)))
    check("reference y = c_s - sum of the free simulated gammas == sim at the reference",
          close(sorted(y_ref), sorted(ref_y_expected), 1e-12))

    print("\n=== 7. WLS through the origin matches the Julia dashboard formula ===")
    b, t = ns["wls_through_origin"](x_free, y_free, x_free)
    w = x_free
    b_ref = np.sum(w * x_free * y_free) / np.sum(w * x_free ** 2)
    resid = y_free - b_ref * x_free
    s2 = np.sum(w * resid ** 2) / max(len(x_free) - 1, 1)
    t_ref = b_ref / np.sqrt(s2 / np.sum(w * x_free ** 2))
    check("coefficient", close(b, b_ref))
    check("t-stat", close(t, t_ref))
    check("perfect fit gives b = 1", close(ns["wls_through_origin"](x_free, x_free, x_free)[0], 1.0, 1e-12))

    print("\n=== 8. figures render ===")
    outdir = os.path.join(ROOT, "out")
    os.makedirs(outdir, exist_ok=True)
    for fn, kw in (("plot_gamma_aa", {}), ("plot_pi_r", {}), ("plot_reg_coef", {}), ("plot_G0", {})):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                ax = ns[fn](d2, save_to=os.path.join(outdir, f"{fn}.pdf"), **kw)
            made = os.path.getsize(os.path.join(outdir, f"{fn}.pdf")) > 0
            check(f"{fn} -> pdf", made)
        except Exception as e:
            check(f"{fn} -> pdf", False, f"{type(e).__name__}: {e}")

    check("reg bin labels for n_coef=4 drop the base bin",
          ns["reg_bin_labels"](4) == ns["REG_BIN_LABELS_5"][1:])
    check("reg bin labels for n_coef=5 keep all", len(ns["reg_bin_labels"](5)) == 5)

    print("\n=== 9. LaTeX table: labor + industry only ===")
    tex = ns["generate_combined_table"](
        [{"industry": "test", "display_name": "Synthetic"}],
        output_file=os.path.join(outdir, "moments.tex"), mu=2, base=ROOT,
        name_A129_path=os.path.join(ROOT, "does_not_exist.csv"))
    check("Panel A present", "Panel A: Aggregate Labor Share" in tex)
    check("Panel B present", "Panel B: Aggregate Industry Shares" in tex)
    check("no regression-coefficient panel", "Panel C" not in tex and "Regression Coefficients" not in tex)
    check("labor share values in the table",
          f'{exp["agg_labor_share"]:.4f}' in tex and f'{exp["sim_final"]["labor"][0]:.4f}' in tex)
    check("all S industry rows present",
          all(f'{v:.4f}' in tex for v in exp["sim_final"]["industry"]))
    check("mu is recorded in the notes", "Step~3" in tex)

    print("\n=== 10. mismatched-flags guard ===")
    try:
        ns["load_granular_data"]("test", mu=2, base=ROOT, granular=False)
        check("wrong --granular flag is caught", False)
    except (ValueError, FileNotFoundError) as e:
        check("wrong --granular flag is caught", True)

    print("\n=== 11. globals().update(data) exposes the analysis.ipynb names ===")
    g = {}
    g.update(d2)
    for name in ("input_folder", "folder", "coefs", "agg_labor_share", "epsilon",
                 "agg_industry_share", "emp_pi_r", "reg_coef", "emp_gamma_ls", "X_dr",
                 "X_rs", "filter_N_upstream_df", "idf_ze", "N_downstream", "K",
                 "empirical_moments", "empirical_moments_dict", "best_simulated_moments",
                 "best_simulated_moments_dict", "best_params", "n_coef", "industry",
                 "emp_gamma_aa", "aa_names", "AA_ACTIVE", "G_target"):
        check(f"data['{name}'] present", name in g)
    check("empirical_moments is a column vector",
          g["empirical_moments"].shape == (len(exp["emp_gamma_aa"]) * exp["n_AA"]
                                           + 1 + exp["S"] + len(exp["emp_pi_r"])
                                           + exp["n_coef"] + exp["S"], 1))
    check("best_simulated_moments_dict keeps the stage axis LAST (old [..., K] layout)",
          close(g["best_simulated_moments_dict"]["reg_coef"][:, -1], exp["sim_final"]["reg"])
          and g["best_simulated_moments_dict"]["gamma_aa"].shape[:2] == (exp["S"], exp["n_AA"]))
    check("best_params is the stage-K parameter column",
          g["best_params"] is not None and g["best_params"].ndim == 1)
    check("optional geography/panels degrade to None when absent",
          g["france"] is None and g["panel_df"] is None and g["ref"] is None)
    check("G_K.csv read through the `G` column (real layout: group, A129, G, K, N_supplier_s)",
          close(g["G_target"], exp["G_target"]))
    check("N_supplier_s read", close(g["N_supplier_s"], exp["N_supplier_s"]))

    print("\n" + "=" * 60)
    print(f"{len(ok)} passed, {len(fail)} failed")
    if fail:
        print("FAILED: " + ", ".join(fail))
        return 1
    return 0


def load_notebook_code(path):
    """Execute the notebook's code cells (skipping magics and the driver) in one namespace."""
    ns = {"__name__": "nb"}
    nb = json.load(open(path))
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if src.lstrip().startswith("%") or "RUN THE REPORTING" in src:
            continue
        exec(compile(src, f"{os.path.basename(path)}:cell{i}", "exec"), ns)
    return ns


if __name__ == "__main__":
    nb_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "analysis_granular.ipynb")
    with tempfile.TemporaryDirectory() as root:
        exp = build_synthetic_tree(root)
        ns = load_notebook_code(nb_path)
        print(f"loaded code from {nb_path}")
        sys.exit(run_checks(ns, root, exp))
