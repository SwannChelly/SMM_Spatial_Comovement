"""
Gate for the granular / attraction-area Python reporting (`utils.py` + `report_lib.py`,
the library behind `model_report.ipynb`).

Standalone and data-free: it writes a SYNTHETIC baseline + reporting tree with the
exact file layout `main.jl --granular=true --ca_level=aa` produces, then executes the
NOTEBOOK'S OWN code cells against it and checks the objects the reporting depends on —
the attraction-area mapping and names, the AA-level gamma aggregation and its s-major
flattening, MOMENT_MASK, the block split of `best_simulated_moments.npy`, the mu_1 /
mu_2 folder routing, the beta -> gamma -> G standard-error split, the reference-area
reconstruction in the gamma panel, and the LaTeX table's contents.

    python3 test/test_analysis_granular.py

Runs every gate, prints PASS/FAIL per gate and exits non-zero if any failed.
Print-only otherwise; nothing in the estimation pipeline is touched.
"""
import contextlib, io, json, os, sys, tempfile, warnings

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

RNG = np.random.default_rng(20260805)



S, R, R_D = 4, 12, 3          # sectors, ZE, downstream regions (= n_AA)
N_REG = 4
N_STAGES = 3


def build_synthetic_tree(ROOT):
    """
    Write TWO synthetic industries and return the expectations for the first.

    Two, because the reporting table is joint (aero + auto in the real notebook) and
    the two industries have DIFFERENT sector sets — the "---" path of the table is
    only exercised when the sector codes do not coincide.
    """
    exp = _build_industry(ROOT, "test", [f"C{10 + s}A" for s in range(S)])
    exp["second"] = _build_industry(ROOT, "test2", [f"C{20 + s}A" for s in range(S)])
    with open(os.path.join(ROOT, "expected.json"), "w") as f:
        json.dump(exp, f)
    return exp


def _build_industry(ROOT, industry, sectors):
    """Write one industry's baseline + reporting tree; return its expectations."""
    BASE = os.path.join(ROOT, f"baseline_{industry}")
    RUN = os.path.join(ROOT, f"reporting_{industry}_profiled_aa_gran_kappa_pso")
    for p in (BASE, RUN, f"{RUN}/step1", f"{RUN}/step2", f"{RUN}/step2/inference_world",
              f"{RUN}/step3", f"{RUN}/step3/inference_world"):
        os.makedirs(p, exist_ok=True)

    ze = [f"{100 + i:04d}" for i in range(R)]

    # ---- geography: downstream regions are ZE 0, 5, 9 (sorted index order) ----
    down_idx = np.array([0, 5, 9])
    N_downstream = np.zeros(R)
    N_downstream[down_idx] = [3.0, 5.0, 2.0]

    aa_of_ze = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 0, 1])   # each ZE -> one AA
    AA_link = np.zeros((R, R_D), dtype=np.int64)
    AA_link[np.arange(R), aa_of_ze] = 1

    # ---- filter / trade: sector 3 has an EMPTY attraction area (AA 2) ---------
    filter_N = np.zeros((S, R), dtype=np.int64)
    supp_mask = np.zeros((S, R), dtype=bool)          # observed supplier cells
    for s in range(S):
        keep = RNG.random(R) < 0.75
        keep[aa_of_ze == 0] = True                      # AA 0 always populated
        if s == 3:
            keep[aa_of_ze == 2] = False                 # AA 2 inactive in sector 3
        filter_N[s, keep] = 1
        supp = keep & (RNG.random(R) < 0.8)
        supp[np.flatnonzero(keep)[0]] = True            # at least one supplier
        supp_mask[s] = supp
        RNG.uniform(1.0, 10.0, supp.sum())             # kept so the RNG stream is unchanged
    # control cells: filter == 1 but gamma == 0 (goods under :aa, zero observed gamma)

    domestic_share = RNG.uniform(0.55, 0.9, S)
    emp_gamma_ls = np.zeros((S, R))
    for s in range(S):
        supp = supp_mask[s]
        w = RNG.uniform(0.2, 1.0, supp.sum())
        emp_gamma_ls[s, supp] = w / w.sum() * domestic_share[s]

    agg_industry_share = RNG.dirichlet(np.ones(S) * 3)
    agg_labor_share = 0.42
    epsilon = -2.5

    X_dr = pd.DataFrame({
        "ze2010": ze,
        "X_dr": np.where(N_downstream > 0, RNG.uniform(50, 500, R), 0.0),
        "downstream": N_downstream > 0,
        "downstream_region": N_downstream > 0,
    })
    X_dr["ze2010_downstream"] = np.where(X_dr["downstream"], X_dr["ze2010"], "")
    X_dr.to_csv(f"{BASE}/X_dr.csv", index=False)

    pd.DataFrame({"stats": ["epsilon", "labor_share", "reg_coef_cloglog_1"],
                  "value": [epsilon, agg_labor_share, 0.23]}
                 ).to_csv(f"{BASE}/stats.csv", index=False)

    pd.DataFrame([{"A129": sectors[s], "ze2010": int(ze[r])}
                  for s in range(S) for r in range(R)]
                 ).to_csv(f"{BASE}/filter_N_upstream.csv", index=False)

    np.save(f"{BASE}/input_share.npy", agg_industry_share)
    np.save(f"{BASE}/emp_gamma_ls.npy", emp_gamma_ls)
    np.save(f"{BASE}/filter_N_upstream.npy", filter_N)
    np.save(f"{BASE}/domestic_share.npy", domestic_share)
    np.save(f"{BASE}/N_downstream_per_region.npy", N_downstream)
    np.save(f"{BASE}/attraction_area_linkages.npy", AA_link)
    # block 4 = [beta_d (N_REG) ; beta_kappa (1)], two files as the data notebook writes
    reg_coef_emp = np.array([-0.35, -0.62, -0.81, -1.05, 0.48])
    np.save(f"{BASE}/beta_d_{N_REG}.npy", reg_coef_emp[:N_REG])
    np.save(f"{BASE}/beta_kappa_{N_REG}.npy", reg_coef_emp[N_REG:])

    # ---- G_K.csv --------------------------------------------------------------
    # G = Pr(K_ls <= K), so the curve must be non-decreasing in K and bounded by 1:
    # the reporting reads K = 0 as the targeted moment and K >= 1 as the untargeted
    # continuation, and both are checked for monotonicity.
    G_target = np.array([0.30, 0.45, 0.20, 0.55])
    N_sup = np.array([40, 26, 51, 18])
    K_ROWS = 4
    G_curve_emp = np.minimum(1.0, G_target[:, None] + 0.1 * np.arange(K_ROWS)[None, :])
    rows = []
    for s in range(S):
        for k in range(K_ROWS):
            # Real column layout: group, A129, G, K, N_supplier_s — `G` is the value
            # column, G = Pr(K_ls <= K), and the K = 0 row is the targeted moment.
            rows.append({"group": s + 1, "A129": sectors[s], "G": G_curve_emp[s, k],
                         "K": k, "N_supplier_s": N_sup[s]})
    pd.DataFrame(rows).to_csv(f"{BASE}/G_K.csv", index=False)

    # ---- G_K_var.csv: bootstrap variance of the INCREMENTS p_s(K) --------------
    # Optional file. Its K = 0 column is the SAME object as the G block of Sigma_data
    # (p_s(0) = G_s(0)), which is what makes it a free consistency check rather than an
    # extra assumption — planted consistent here, and deliberately broken in one gate.
    G_pmf_emp = np.diff(G_curve_emp, axis=1, prepend=0.0)
    G_pmf_se  = np.array([0.03, 0.05, 0.02, 0.04])[:, None] * (1.0 + 0.2 * np.arange(K_ROWS))
    pd.DataFrame([{"A129": sectors[s], "K": k, "var": G_pmf_se[s, k] ** 2}
                  for s in range(S) for k in range(K_ROWS)]
                 ).to_csv(f"{BASE}/G_K_var.csv", index=False)

    # ---- the model-side structures the loader must rebuild ---------------------
    supplier = (filter_N == 1) & (emp_gamma_ls > 0)
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

    # cells_ls.csv: n_ls (establishments) and has_sup on every simulated cell
    N_LS = RNG.integers(1, 40, (S, R))
    pd.DataFrame([{"A129": sectors[s], "ze2010": int(ze[r]), "n_ls": int(N_LS[s, r]),
                   "has_sup": int(emp_gamma_ls[s, r] > 0)}
                  for s in range(S) for r in range(R) if CELL_MASK[s, r]]
                 ).to_csv(f"{BASE}/cells_ls.csv", index=False)

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
    sim_reg = reg_coef_emp * RNG.uniform(0.8, 1.2, N_REG + 1)
    # The count moment is not free: it is `mean_l C(m-k_l, n)/C(m, n)` at the profiled
    # N_hat_s, over the cells of sector s. The reporting rebuilds the curve from the
    # `q_hat` / `N_hat` the diagnostics npz carries, so the synthetic tree has to be
    # self-consistent the way a real run is — plant the win counts first and DERIVE
    # block 6 from them, or the K = 0 identity has nothing to gate.
    N_HAT = np.array([12, 7, 21, 4])
    N_RHO = max(100, int(N_sup.max()))              # load_parameters.jl's rule
    K_COUNTS = RNG.integers(0, 30, int(CELL_MASK.sum()))
    q_hat_plant = K_COUNTS / N_RHO
    _lg = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, N_RHO + 1, dtype=float)))])

    def _lb(a, b):
        return _lg[a] - _lg[b] - _lg[a - b]

    def _gbar0(kk, n):
        out = np.zeros(np.shape(kk), float)
        ok = n <= N_RHO - kk
        out[ok] = np.exp(_lb(N_RHO - kk[ok], n) - _lb(N_RHO, n))
        return out

    _cell_s = np.flatnonzero(CELL_MASK.ravel(order="F")) % S
    sim_G0 = np.array([_gbar0(K_COUNTS[_cell_s == s], int(N_HAT[s])).mean()
                       for s in range(S)])

    final = np.concatenate([sim_labor, sim_industry, sim_pi, sim_reg,
                            sim_gamma_aa.ravel(), sim_G0])
    stages = np.column_stack([final * (0.7 + 0.15 * k) for k in range(N_STAGES - 1)] + [final])
    np.save(f"{RUN}/step1/best_simulated_moments.npy", stages * 0.95)
    np.save(f"{RUN}/step3/best_simulated_moments.npy", stages)
    n_par = 1 + S + R_D + 1 + 1 + int(AA_ACTIVE.sum())      # ... alpha | kappa | T
    PAR = {st: RNG.random((n_par, N_STAGES)) + 0.1 for st in ("step1", "step3")}
    for st in PAR:
        np.save(f"{RUN}/{st}/best_parameters_list.npy", PAR[st])

    np.save(f"{RUN}/n_reg_coef.npy", np.array(N_REG))
    np.save(f"{RUN}/n_tau.npy", np.array(1))

    gam_free = AA_ACTIVE.copy()
    for s in range(S):
        gam_free[s, T_REF[s]] = False
    n_gb = N_REG + 1 + int(gam_free.sum()) + S
    n_mom_kept = 1 + (S - 1) + (R_D - 1) + N_REG + 1 + int(gam_free.sum()) + S

    for st in ("step1", "step2", "step3"):
        np.savez(f"{RUN}/{st}/granular_diagnostics.npz",
                 N_hat=N_HAT.astype(float),
                 N_LO=np.array([14.0, 9.0, 17.0, 6.0]),
                 N_HI=N_sup.astype(float),
                 clamped=np.array([0.0, 0.0, 0.0, -1.0]),
                 G0_fit=sim_G0, G0_target=G_target,
                 N_count=np.array([690.0, 120.0, 55.0, 33.0]),
                 q_hat=q_hat_plant, EK=RNG.uniform(0.1, 3, S),
                 reg_coef_model=sim_reg, reg_coef_sd_reps=np.abs(sim_reg) * 0.1,
                 reg_coef_expected=sim_reg * 1.01, reg_fail=np.array([0.0]))

    # ---- Step 2: the fake-world residuals and the weighting of block 4 -------
    B_W, k4 = 300, N_REG + 1
    L4 = RNG.normal(size=(k4, k4)) * 0.05
    e = RNG.normal(size=(B_W, k4)) @ L4.T + 0.01        # a planted non-zero mean
    beta_sim_w = sim_reg[None, :] * 0.95 + RNG.normal(size=(B_W, k4)) * 0.01
    Omega_w = np.cov(e, rowvar=False)
    np.savez(f"{RUN}/step2/world_residuals.npz", e=e, beta_world=beta_sim_w + e,
             beta_sim=beta_sim_w, N_hat_world=np.tile(N_HAT, (B_W, 1)).astype(float),
             N_hat=N_HAT.astype(float), e_data=reg_coef_emp - sim_reg * 0.95)
    np.save(f"{RUN}/step2/Omega_world.npy", Omega_w)
    np.save(f"{RUN}/step2/W_world.npy", np.linalg.inv(Omega_w))

    # ---- inference from fake worlds: step2 (theta_1, W = I) and step3 (theta_2) ----
    t_lbl = [f"T[{sectors[s]}-AA{ze[down_idx[a]]}]" for s in range(S) for a in range(R_D)
             if AA_ACTIVE[s, a]]
    w_lbl = (["alpha[1]", "kappa"] + [f"N_s[{x}]" for x in sectors] + t_lbl
             + [f"A[{i + 1}]" for i in range(R_D)])
    B_I = 200
    W_DRAWS = {}
    for st, scale in (("step2", 1.0), ("step3", 0.6)):
        bp = PAR["step1" if st == "step2" else "step3"][:, -1]
        est = np.concatenate([[bp[1 + S + R_D], bp[2 + S + R_D]], N_HAT.astype(float),
                              RNG.uniform(0.5, 2.0, len(t_lbl)), RNG.uniform(0.8, 1.2, R_D)])
        P = est[None, :] * (1.0 + scale * 0.1 * RNG.normal(size=(B_I, est.size)))
        P[:, 2:2 + S] = np.maximum(1, np.rint(N_HAT[None, :] + scale *
                                              RNG.normal(0, 2, (B_I, S))))
        P[:5, 2 + S] = np.nan                           # an area empty in five worlds
        np.savez(f"{RUN}/{st}/inference_world/world_inference_draws.npz", params=P,
                 estimate=est, disp_logT=RNG.random((B_I, S)),
                 counterfactuals=RNG.random((B_I, 18)), counterfactuals_est=RNG.random(18),
                 loss=RNG.random(B_I), iters=np.full(B_I, 40.0),
                 thetas=RNG.random((B_I, n_par)))
        with open(f"{RUN}/{st}/inference_world/world_inference_labels.txt", "w") as f:
            f.write("\n".join(w_lbl) + "\n")
        W_DRAWS[st] = (P.tolist(), est.tolist())

    # ---- the expectations the test checks against ----------------------------
    expected = {
        "S": S, "R": R, "n_AA": R_D, "n_coef": N_REG, "n_gb": n_gb,
        "G_curve_emp": G_curve_emp.tolist(), "K_rows": list(range(K_ROWS)),
        "G_pmf_emp": G_pmf_emp.tolist(), "G_pmf_se": G_pmf_se.tolist(),
        "world_e": e.tolist(), "world_labels": w_lbl, "world_draws": W_DRAWS,
        "q_hat": q_hat_plant.tolist(), "N_LS": N_LS.tolist(),
        "N_hat": N_HAT.tolist(), "n_rho": N_RHO,
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
        "industry": industry,
        "n_mom_kept": n_mom_kept,
        "ze": ze,
    }
    print(f"synthetic {industry}: {ROOT}  (S={S}, R={R}, n_AA={R_D}, n_gb={n_gb}, gamma kept={int(gam_free.sum())})")
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
    check("run folder resolved", d2["folder"].name == "reporting_test_profiled_aa_gran_kappa_pso")
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

    print("\n=== 2b. Gbar_s(0): the figure against Julia's own reporting ===")
    # Four writings of the same moment: G_K.csv and `G0_target` on the empirical side,
    # block 6 of best_simulated_moments.npy and `G0_fit` on the simulated side. They are
    # produced by different Julia code paths at the same theta, so agreement is a real
    # cross-check — and the checker has to be able to FAIL, or it checks nothing. The
    # synthetic tree supplies both cases: step3's moment vector carries exactly the
    # `G0_fit` planted in step3/granular_diagnostics.npz, while step1's is that same
    # vector scaled by 0.95 against an unscaled step2 npz.
    d1 = ns["load_granular_data"]("test", mu=1, base=ROOT)
    g2 = ns["g0_consistency"](d2, verbose=False)
    g1 = ns["g0_consistency"](d1, verbose=False)
    check("consistency table is indexed by sector, one row each",
          list(g2.index) == exp["sector_names"])
    check("target column is G_K.csv at K = 0",
          close(g2["target (G_K.csv)"], exp["G_target"]))
    check("target agrees with the diagnostics' frozen G_TARGET",
          close(g2["d_target"], np.zeros(exp["S"])))
    check("fit column is block 6 of the moment vector",
          close(g2["fit (moment vector)"], d2["simulated_moments_dict"]["G0"]))
    check("matching moment vector and diagnostics -> agrees",
          bool(g2.attrs["agrees"]) and g2.attrs["max_abs_fit_gap"] <= 1e-12)
    check("a diagnostics file at another theta is CAUGHT, not averaged over",
          not bool(g1.attrs["agrees"]))
    check("the caught gap is the planted one (5% of G0_fit)",
          close(g1.attrs["max_abs_fit_gap"],
                np.max(np.abs(np.array(exp["sim_final"]["G0"]) * 0.05)), 1e-12))
    check("residual column is fit - target",
          close(g2["residual (fit - target)"],
                np.asarray(g2["fit (moment vector)"]) - np.asarray(g2["target (G_K.csv)"])))
    check("N_hat and clamp carried through from the diagnostics",
          list(g2["clamp"]) == ["none", "none", "none", "lo"])
    # A G_K.csv whose A129 set differs from filter_N_upstream.csv's would attach every
    # sector's zero-supplier target to another sector, silently and identically in Julia.
    gk_path = os.path.join(ROOT, "baseline_test", "G_K.csv")
    gk_orig = open(gk_path).read()
    try:
        gk = pd.read_csv(io.StringIO(gk_orig))
        gk["A129"] = gk["A129"].replace({exp["sector_names"][0]: "ZZZZ"})
        gk.to_csv(gk_path, index=False)
        raised = False
        try:
            ns["load_granular_data"]("test", mu=2, base=ROOT)
        except ValueError as e:
            raised = "do not enumerate the same sectors" in str(e)
        check("a G_K.csv covering different sectors is refused, not silently misaligned",
              raised)
    finally:
        open(gk_path, "w").write(gk_orig)

    print("\n=== 2c. the count curve Gbar_s(K), K = 0..3 ===")
    # K = 0 is block 6 of the moment vector and N_hat_s is calibrated on it; K >= 1 is
    # untargeted and exists nowhere on the Julia side, so the notebook rebuilds it from
    # `q_hat` and `N_hat`. The gate that matters is that the rebuild REPRODUCES block 6
    # at K = 0 — same estimator, same cells, same draw count — because that is the only
    # thing tying the untargeted panels to the targeted one.
    cc = ns["count_curve"](d2, K_values=(0, 1, 2, 3))
    check("curve is indexed by (sector, K)", list(cc.index.names) == ["A129", "K"] and
          len(cc) == exp["S"] * 4)
    # The panels are INCREMENTS: p_s(K) = Pr[K_ls = K] = G(K) - G(K-1). G_K.csv stores
    # the CDF, so the empirical series must be its first difference, with p_s(0) = G(0)
    # — the targeted moment is the same number under either convention.
    check("empirical curve is the INCREMENT of G_K.csv, p_s(K) = G(K) - G(K-1)",
          close(cc["empirical"].values.reshape(exp["S"], 4),
                np.diff(np.array(exp["G_curve_emp"]), axis=1, prepend=0.0)))
    check("the unbiased estimator was used (no fallback)",
          cc.attrs["estimator"].startswith("unbiased"))
    check("N_rho recovered from N_HI as load_parameters.jl fixes it",
          cc.attrs["n_rho"] == exp["n_rho"])
    check("K = 0 column reproduces block 6 of the moment vector",
          close(cc.xs(0, level="K")["simulated"].values,
                d2["simulated_moments_dict"]["G0"], 1e-12))
    check("K = 0 gap is reported and is machine zero",
          cc.attrs["K0_gap_vs_moment_vector"] <= 1e-12)
    sim_w = cc["simulated"].values.reshape(exp["S"], 4)
    # A pmf, not a CDF: no monotonicity to check, but the panels must still be
    # probabilities and their partial sums must not exceed one.
    check("simulated curve stays in [0, 1]",
          bool((sim_w >= -1e-12).all() and (sim_w <= 1 + 1e-12).all()))
    check("the increments are a sub-probability: cumulative sum <= 1",
          bool((np.cumsum(sim_w, axis=1) <= 1 + 1e-9).all()))
    check("only K = 0 is flagged as targeted",
          list(cc["targeted"].values.reshape(exp["S"], 4)[0]) == [True, False, False, False])
    check("N_hat carried through per sector",
          close(cc.xs(0, level="K")["N_hat"].values, exp["N_hat"]))
    check("residual is simulated - empirical",
          close(cc["residual"].values, cc["simulated"].values - cc["empirical"].values))
    # ---- bands on both series ------------------------------------------------
    # The simulated band is the fake-world one: the variety counts each world
    # re-calibrates (step3/inference_world for mu = 2), pushed through the closed form at
    # q_hat. The empirical band is the bootstrap variance of the increments (G_K_var.csv).
    check("empirical SE read from G_K_var.csv as sqrt(var)",
          close(cc["se_empirical"].values.reshape(exp["S"], 4),
                np.array(exp["G_pmf_se"])))
    P3 = np.array(exp["world_draws"]["step3"][0])
    Nw = np.rint(P3[:, 2:2 + exp["S"]]).astype(int)
    lgf = ns["_logfact"](exp["n_rho"])
    kc = np.rint(np.array(exp["q_hat"]) * exp["n_rho"]).astype(int)
    cs = ns["_cell_sectors"](d2)
    s0 = 1
    manual = np.array([[ns["gbar_cells"](kc[cs == s0], exp["n_rho"], n, K, lgf).mean()
                        for K in range(4)] for n in Nw[:, s0]])
    sub = cc.xs(d2["sector_names"][s0], level="A129")
    check("simulated band = 2.5/97.5% quantiles of the curve at the worlds' N_s",
          close(sub["sim_lo"].values, np.quantile(manual, 0.025, axis=0))
          and close(sub["sim_hi"].values, np.quantile(manual, 0.975, axis=0)))
    check("the band brackets the fitted K = 0 point in every sector",
          bool(np.all(cc.xs(0, level="K")["sim_lo"] <= cc.xs(0, level="K")["simulated"] + 1e-12)
               and np.all(cc.xs(0, level="K")["simulated"] <= cc.xs(0, level="K")["sim_hi"] + 1e-12)))
    check("the clamp flag is carried onto every K of that sector",
          bool((cc["clamped"].values.reshape(exp["S"], 4)[3]).all()) and
          not bool((cc["clamped"].values.reshape(exp["S"], 4)[0]).any()))
    # one INDEPENDENT figure per K, each in its own file: `save_to` is a stem and K is
    # appended before the extension, so nothing overwrites anything
    cc_axes = ns["plot_count_curve"](d2, K_values=(0, 1, 2, 3),
                                     save_to=os.path.join(ROOT, "cc.pdf"))
    check("plot_count_curve draws one figure per K",
          isinstance(cc_axes, dict) and sorted(cc_axes) == [0, 1, 2, 3]
          and len({id(a.figure) for a in cc_axes.values()}) == 4)
    check("each K is written to its own file",
          all(os.path.exists(os.path.join(ROOT, f"cc_K{K}.pdf")) for K in (0, 1, 2, 3))
          and not os.path.exists(os.path.join(ROOT, "cc.pdf")))
    # the y label names the K, since there is no title to tell the figures apart
    check("the y label carries the value of K, and there is no title",
          all(cc_axes[K].get_ylabel() == rf"$\bar{{G}}_s({K})$" for K in (0, 1, 2, 3))
          and not any(cc_axes[K].get_title() for K in (0, 1, 2, 3)))
    # and a K the curve does not carry must be named, not silently dropped
    try:
        ns["plot_count_curve"](d2, df=ns["count_curve"](d2, K_values=(0, 1)), K_values=(7,))
        check("an absent K raises", False, "no exception")
    except KeyError as e:
        check("an absent K raises", "7" in str(e))

    # p_s(0) = G_s(0), so a G_K_var.csv whose K = 0 column disagrees with the G block of
    # Sigma_data is measuring a different object — most likely the CDF's variance. That
    # must be SAID, not silently plotted.
    gkv_path = os.path.join(ROOT, "baseline_test", "G_K_var.csv")
    gkv_orig = open(gkv_path).read()

    # A sector whose bootstrap variance is 0 (or missing) must keep its POINT and simply
    # lose its bar: the value is known, only its uncertainty is not. Drawing it through
    # errorbar with a zero yerr would put a bare cap on the marker; a NaN yerr drops the
    # point entirely. Both are wrong, and the second is what makes a point look absent.
    try:
        z = pd.read_csv(io.StringIO(gkv_orig))
        z.loc[z["K"] == 2, "var"] = 0.0
        z.to_csv(gkv_path, index=False)
        d_z = ns["load_granular_data"]("test", mu=2, base=ROOT)
        cc_z = ns["count_curve"](d_z, K_values=(0, 1, 2, 3))
        check("a zero bootstrap variance keeps the empirical VALUE",
              bool(np.isfinite(cc_z.xs(2, level="K")["empirical"].values).all()) and
              close(cc_z.xs(2, level="K")["se_empirical"].values, np.zeros(exp["S"])))
        axs_z = ns["plot_count_curve"](d_z, K_values=(0, 1, 2, 3))
        ax_k2 = axs_z[2]               # one figure per K, keyed by K
        emp = [c for c in ax_k2.containers if c.get_label() == "Empirical"]
        n_pts = sum(len(c[0].get_xdata()) for c in emp)
        check("every sector is still plotted at K = 2 despite a zero variance",
              n_pts == exp["S"])
        check("and none of them carries an error bar",
              not any(getattr(c, "has_yerr", False) for c in emp))
        # the K = 0 panel, where the variances are nonzero, must still show bars
        emp0 = [c for c in axs_z[0].containers if c.get_label() == "Empirical"]
        check("a panel with nonzero variances still draws its bars",
              any(getattr(c, "has_yerr", False) for c in emp0))
    finally:
        open(gkv_path, "w").write(gkv_orig)

    # A G_K_var.csv written off G_K.csv's own schema keeps the value column called `G`,
    # which names the object measured, not the statistic. It must still be read, as a
    # variance, and the fallback must SAY so rather than guess silently.
    try:
        gcol = pd.read_csv(io.StringIO(gkv_orig)).rename(columns={"var": "G"})
        gcol.to_csv(gkv_path, index=False)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cc_g = ns["count_curve"](ns["load_granular_data"]("test", mu=2, base=ROOT),
                                     K_values=(0, 1, 2, 3))
        check("a value column named `G` is read as the variance, and the choice announced",
              "as the VARIANCE" in buf.getvalue() and
              close(cc_g["se_empirical"].values.reshape(exp["S"], 4),
                    np.array(exp["G_pmf_se"])))
    finally:
        open(gkv_path, "w").write(gkv_orig)

    # An ambiguous file (two spare columns) must refuse rather than pick one.
    try:
        amb = pd.read_csv(io.StringIO(gkv_orig)).rename(columns={"var": "G"})
        amb["other"] = 1.0
        amb.to_csv(gkv_path, index=False)
        raised = False
        try:
            ns["load_granular_data"]("test", mu=2, base=ROOT)
        except KeyError as e:
            raised = "cannot tell which column holds the variance" in str(e)
        check("an ambiguous G_K_var.csv is refused, not guessed", raised)
    finally:
        open(gkv_path, "w").write(gkv_orig)

    # k_max truncates every count-curve input at once, so raising the constant is the
    # single place that widens the reporting.
    d_k1 = ns["load_granular_data"]("test", mu=2, base=ROOT, k_max=1)
    check("k_max truncates the empirical curve and its bootstrap SE together",
          d_k1["G_curve_K"] == [0, 1] and d_k1["G_pmf"].shape[1] == 2
          and d_k1["G_pmf_se"].shape[1] == 2)
    check("k_max leaves G_target (the K = 0 targeted moment) untouched",
          close(d_k1["G_target"], d2["G_target"]))

    # Sectors are matched by VALUE on the A129 code. A file whose codes do not line up
    # (a renamed sector, or an int64 column against a string one) would otherwise hand
    # back an all-NaN band with no complaint.
    try:
        wrong = pd.read_csv(io.StringIO(gkv_orig))
        wrong["A129"] = wrong["A129"].replace({exp["sector_names"][0]: "ZZZZ"})
        wrong.to_csv(gkv_path, index=False)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cc_w = ns["count_curve"](ns["load_granular_data"]("test", mu=2, base=ROOT),
                                     K_values=(0, 1, 2, 3))
        check("an A129 code absent from G_K_var.csv is named, not silently NaN",
              "G_K_var.csv has no row for A129" in buf.getvalue() and
              bool(np.isnan(cc_w["se_empirical"].values.reshape(exp["S"], 4)[0]).all()) and
              bool(np.isfinite(cc_w["se_empirical"].values.reshape(exp["S"], 4)[1]).all()))
    finally:
        open(gkv_path, "w").write(gkv_orig)

    # Absent the optional file, the empirical band must be empty rather than guessed.
    os.rename(gkv_path, gkv_path + ".off")
    try:
        cc_nb = ns["count_curve"](ns["load_granular_data"]("test", mu=2, base=ROOT),
                                  K_values=(0, 1, 2, 3))
        check("no G_K_var.csv => empirical SE is all NaN, never invented",
              bool(np.isnan(cc_nb["se_empirical"].values).all()))
    finally:
        os.rename(gkv_path + ".off", gkv_path)

    # G is CUMULATIVE, so a (sector, K) row absent from G_K.csv means no mass was added
    # there, not that the level is unknown. Differencing a NaN would poison TWO
    # increments (K and K+1) and erase two empirical points from the figure.
    gk_path2 = os.path.join(ROOT, "baseline_test", "G_K.csv")
    gk_orig2 = open(gk_path2).read()
    try:
        g = pd.read_csv(io.StringIO(gk_orig2))
        g = g[~((g["K"] == 2) & (g["A129"] == exp["sector_names"][0]))]
        g.to_csv(gk_path2, index=False)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cc_gap = ns["count_curve"](ns["load_granular_data"]("test", mu=2, base=ROOT),
                                       K_values=(0, 1, 2, 3))
        emp_gap = cc_gap["empirical"].values.reshape(exp["S"], 4)
        ref = np.diff(np.array(exp["G_curve_emp"]), axis=1, prepend=0.0)
        check("a missing K row is read as a flat CDF (increment 0), not two NaNs",
              bool(np.isfinite(emp_gap).all()) and abs(emp_gap[0, 2]) < 1e-12
              and abs(emp_gap[0, 3] - (ref[0, 2] + ref[0, 3])) < 1e-12)
        check("and the fill is announced rather than silent",
              "the CDF is flat there" in buf.getvalue())
        check("other sectors are untouched by one sector's gap",
              close(emp_gap[1:], ref[1:]))
    finally:
        open(gk_path2, "w").write(gk_orig2)

    # A K the data pipeline never tabulated must leave the panel's empirical series
    # empty rather than silently borrowing a neighbouring K.
    cc9 = ns["count_curve"](d2, K_values=(0, 9))
    check("a K absent from G_K.csv comes back NaN, not shifted",
          bool(np.isnan(cc9.xs(9, level="K")["empirical"].values).all())
          and close(cc9.xs(0, level="K")["simulated"].values,
                    d2["simulated_moments_dict"]["G0"], 1e-12))
    # The cell -> sector map: Julia's findall over the (S, R) CELL_MASK is
    # COLUMN-major, so reading q_hat in C order would shuffle cells between sectors.
    cs = ns["_cell_sectors"](d2)
    check("cell -> sector map is column-major, matching findall(CELL_MASK)",
          list(cs) == list(np.flatnonzero(np.array(exp["CELL_MASK"]).ravel(order="F"))
                           % exp["S"]))
    check("one q_hat per simulated cell",
          cs.size == int(np.array(exp["CELL_MASK"]).sum()))

    print("\n=== 3. moment mask (must reproduce load_parameters.jl) ===")
    mask, S, nAA = d2["moment_mask"], d2["S"], d2["n_AA"]
    off_gam = 1 + S + len(exp["emp_pi_r"]) + exp["n_coef"] + 1      # + beta_kappa
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
    check("mu = 1 reads its world draws from step2/inference_world, mu = 2 from step3",
          close(d1["world_inference"]["estimate"], exp["world_draws"]["step2"][1])
          and close(d2["world_inference"]["estimate"], exp["world_draws"]["step3"][1]))
    check("mu = 1 / mu = 2 folder routing",
          (d1["step_dir"], d1["inference_step"], d2["step_dir"], d2["inference_step"])
          == ("step1", "step2", "step3", "step3"))
    try:
        ns["load_granular_data"]("test", mu=3, base=ROOT)
        check("bad mu rejected", False)
    except ValueError:
        check("bad mu rejected", True)

    print("\n=== 5. fake-world inference: block-4 band and parameter intervals ===")
    E = np.array(exp["world_e"])
    wt = ns["reg_coef_world_table"](d2)
    sim4 = d2["simulated_moments_dict"]["reg_coef"]
    check("block-4 table: one row per [beta_d ; beta_kappa], kappa last",
          len(wt) == exp["n_coef"] + 1 and wt.index[-1] == "beta_kappa")
    check("band = simulated + quantiles of e = beta_world - beta_sim",
          close(wt["band_lo"].values, sim4 + np.quantile(E, 0.025, axis=0))
          and close(wt["band_hi"].values, sim4 + np.quantile(E, 0.975, axis=0)))
    check("z centres on the world mean of e and scales by its sd",
          close(wt["z"].values, (np.array(exp["reg_coef"]) - sim4 - E.mean(0)) / E.std(0, ddof=1)))
    r4 = np.array(exp["reg_coef"]) - sim4
    check("J = e_data' W_world e_data", close(wt.attrs["J"], r4 @ d2["W_world"] @ r4, 1e-9))
    check("Omega_world and W_world loaded, block-4 sized",
          d2["Omega_world"].shape == (exp["n_coef"] + 1,) * 2
          and close(d2["W_world"] @ d2["Omega_world"], np.eye(exp["n_coef"] + 1), 1e-8))
    pt = ns["world_param_table"](d2)
    P3, est3 = (np.array(x) for x in exp["world_draws"]["step3"])
    check("parameter table rows are Julia's labels", list(pt.index) == exp["world_labels"])
    check("kappa: estimate, sd and quantiles across worlds",
          close(pt.loc["kappa", "estimate"], est3[1])
          and close(pt.loc["kappa", "sd"], P3[:, 1].std(ddof=1))
          and close(pt.loc["kappa", "q_lo"], np.quantile(P3[:, 1], 0.025)))
    t0 = exp["world_labels"][2 + S]
    check("an area empty in some worlds is counted, and excluded from its statistics",
          close(pt.loc[t0, "empty_share"], 5 / P3.shape[0])
          and close(pt.loc[t0, "mean"], np.nanmean(P3[:, 2 + S])))
    check("kappa is read from best_params, after alpha",
          close(ns["unpack_estimated_T"](d2)["kappa"], d2["best_params"][2 + S + exp["n_AA"]]))
    T_est = ns["unpack_estimated_T"](d2)
    check("T_cell = T_a(l) * n_ls^kappa on the simulated cells, zero elsewhere",
          close(T_est["T_cell"], T_est["T"][:, d2["aa_of_ze"]]
                * np.array(exp["N_LS"], float) ** T_est["kappa"] * d2["CELL_MASK"]))

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
    for fn, kw in (("plot_gamma_aa", {}), ("plot_pi_r", {}), ("plot_reg_coef", {}),
                   ("plot_G0", {}), ("plot_count_curve", {"K_values": (0, 1, 2, 3)}),
                   ("plot_parameter_ci", {}), ("plot_world_covariance", {})):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                ax = ns[fn](d2, save_to=os.path.join(outdir, f"{fn}.pdf"), **kw)
            # plot_count_curve treats save_to as a stem and writes one file per K
            written = (os.path.join(outdir, f"{fn}_K0.pdf")
                       if fn == "plot_count_curve" else os.path.join(outdir, f"{fn}.pdf"))
            made = os.path.getsize(written) > 0
            check(f"{fn} -> pdf", made)
        except Exception as e:
            check(f"{fn} -> pdf", False, f"{type(e).__name__}: {e}")

    check("reg bin labels for n_coef=4 drop the base bin",
          ns["reg_bin_labels"](4) == ns["reg_bin_labels"](5)[1:])
    check("reg bin labels for n_coef=5 keep all", len(ns["reg_bin_labels"](5)) == 5)
    check("with_kappa appends ln n", ns["reg_bin_labels"](1, with_kappa=True)[-1] == r"$\ln n_{ls}$")
    axs4 = ns["plot_reg_coef"](d2)
    sims = [c for c in axs4[1].containers if c.get_label().startswith("Simulated")]
    yerr = sims[0][2][0].get_segments()[0] if sims and sims[0][2] else None
    check("plot_reg_coef: distance panel + density panel, the band drawn on beta_kappa",
          len(axs4) == 2 and yerr is not None
          and close(sorted([yerr[0][1], yerr[1][1]]), [wt["band_lo"].iloc[-1], wt["band_hi"].iloc[-1]], 1e-9))

    print("\n=== 9. LaTeX table: JOINT (two industries), labor + industry only ===")
    tex = ns["generate_combined_table"](
        [{"industry": "test", "display_name": "Synthetic A"},
         {"industry": "test2", "display_name": "Synthetic B"}],
        output_file=os.path.join(outdir, "moments.tex"), mu=2, base=ROOT,
        name_A129_path=os.path.join(ROOT, "does_not_exist.csv"))
    check("both industries have their own column pair",
          "Synthetic A" in tex and "Synthetic B" in tex
          and tex.count(r"{Emp.} & {Sim.}") == 2)
    check("column spec has 2 pairs", tex.count("S[table-format=-1.4]") == 4)
    check("second industry's simulated values present",
          all(f"{v:.4f}" in tex for v in exp["second"]["sim_final"]["industry"]))
    check("disjoint sector sets produce `---` cells", "{---}" in tex)
    check("row count = union of both sector sets",
          sum(1 for line in tex.splitlines() if line.strip().endswith(r"\\")
              and line.split("&")[0].strip() in
              set(exp["sector_names"]) | set(exp["second"]["sector_names"])) == 2 * S)
    check("Panel A present", "Panel A: Aggregate Labor Share" in tex)
    check("Panel B present", "Panel B: Aggregate Industry Shares" in tex)
    check("Panel C: the extensive margin, beta_d bins and beta_kappa",
          "Panel C: Extensive Margin" in tex and r"$\beta_\kappa$" in tex
          and all(f"{v:.4f}" in tex for v in exp["reg_coef"]))
    check("Panel C carries the 95% fake-world band under each simulated coefficient",
          f"[{wt['band_lo'].iloc[-1]:.3f}, {wt['band_hi'].iloc[-1]:.3f}]" in tex)
    ptex = ns["generate_parameter_table"](
        [{"industry": "test", "display_name": "Synthetic A"},
         {"industry": "test2", "display_name": "Synthetic B"}],
        output_file=os.path.join(outdir, "params.tex"), mu=2, base=ROOT)
    check("parameter table: alpha, kappa and N_s with their world interval",
          r"$\kappa$" in ptex and r"$\alpha$" in ptex and r"$\hat N_s$" in ptex
          and f"[{pt.loc['kappa', 'q_lo']:.3f}, {pt.loc['kappa', 'q_hi']:.3f}]" in ptex)
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
                 "filter_N_upstream_df", "idf_ze", "N_downstream", "K",
                 "empirical_moments", "empirical_moments_dict", "best_simulated_moments",
                 "best_simulated_moments_dict", "best_params", "n_coef", "industry",
                 "emp_gamma_aa", "aa_names", "AA_ACTIVE", "G_target"):
        check(f"data['{name}'] present", name in g)
    check("empirical_moments is a column vector",
          g["empirical_moments"].shape == (len(exp["emp_gamma_aa"]) * exp["n_AA"]
                                           + 1 + exp["S"] + len(exp["emp_pi_r"])
                                           + exp["n_coef"] + 1 + exp["S"], 1))
    check("best_simulated_moments_dict keeps the stage axis LAST (old [..., K] layout)",
          close(g["best_simulated_moments_dict"]["reg_coef"][:, -1], exp["sim_final"]["reg"])
          and g["best_simulated_moments_dict"]["gamma_aa"].shape[:2] == (exp["S"], exp["n_AA"]))
    check("best_params is the stage-K parameter column",
          g["best_params"] is not None and g["best_params"].ndim == 1)
    check("optional geography degrades to None when absent",
          g["france"] is None and g["suppliers"] is None)
    check("G_K.csv read through the `G` column (real layout: group, A129, G, K, N_supplier_s)",
          close(g["G_target"], exp["G_target"]))
    check("N_supplier_s read", close(g["N_supplier_s"], exp["N_supplier_s"]))

    print("\n=== 12. labels: the Julia layout with kappa ===")
    check("no Jacobian on a current tree (the Julia code no longer writes one)", d2["J"] is None)
    check("moment blocks: labor, industry-1, pi_r-1, [beta_d ; beta_kappa], gamma free, G0",
          list(d2["moment_block_sizes"]) ==
          [1, S - 1, exp["n_AA"] - 1, exp["n_coef"] + 1, exp["n_gamma_kept"], S]
          and sum(d2["moment_block_sizes"]) == exp["n_mom_kept"])
    check("block-4 labels as MOMENT_LABELS writes them",
          [l for l in d2["moment_labels"] if l.startswith("beta")] ==
          ["beta_d[]50,100]]", "beta_d[]100,150]]", "beta_d[]150,200]]", "beta_d[>200]",
           "beta_kappa"])
    check("parameter blocks: Omega_L, Omega_s-1, A-1, [alpha ; kappa], T free",
          list(d2["param_block_sizes"]) ==
          [1, S - 1, exp["n_AA"] - 1, 2, exp["n_gamma_kept"]]
          and "kappa" in d2["param_labels"])
    check("gamma moment labels and T parameter labels enumerate the same (s, AA) pairs",
          [l.replace("gamma[", "") for l in d2["moment_labels"]
           if l.startswith("gamma[")] ==
          [l.replace("T[", "") for l in d2["param_labels"] if l.startswith("T[")])
    ws = ns["world_weight_summary"](d2)
    check("world_weight_summary returns the block-4 table", ws is not None and len(ws) == exp["n_coef"] + 1)

    print("\n" + "=" * 60)
    print(f"{len(ok)} passed, {len(fail)} failed")
    if fail:
        print("FAILED: " + ", ".join(fail))
        return 1
    return 0


def load_library_code():
    """The reporting library, as one namespace.

    This used to execute `analysis_granular.ipynb`'s definition cells and skip its run
    cells by a banner. The definitions are now modules, so they are imported: the same
    code, without a notebook's JSON in the loop, and with no need to guess which cells
    define and which run.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(here, "..")
    if root not in sys.path:
        sys.path.insert(0, root)
    import utils, report_lib
    ns = {"__name__": "nb"}
    for m in (utils, report_lib):
        for k in vars(m):
            if not k.startswith("__"):
                ns[k] = getattr(m, k)
    return ns


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as root:
        exp = build_synthetic_tree(root)
        ns = load_library_code()
        print("loaded code from utils.py + report_lib.py")
        sys.exit(run_checks(ns, root, exp))
