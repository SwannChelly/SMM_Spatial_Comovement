"""A minimal but REAL run tree, written to a temporary directory.

It carries exactly the files `parts=("core", "geography")` reads and nothing else -- no
`best_simulated_moments.npy`, no `Sigma_*`, no `jacobian_*`, no `suppliers.parquet` --
which is what makes the specialised loader and the parquet-free reporting path testable
without a run.
"""
import os, tempfile
import numpy as np
import pandas as pd


def build(seed=11):
    S, R, R_D, n_coef = 3, 8, 4, 4
    rng = np.random.default_rng(seed)
    TMP = tempfile.mkdtemp(prefix="parts_gate_")
    inp = os.path.join(TMP, "baseline_test")
    run = os.path.join(TMP, "reporting_test_profiled_aa_gran_pso")
    os.makedirs(inp); os.makedirs(os.path.join(run, "step3"))

    filt = np.zeros((S, R), dtype=int); filt[:, :6] = 1
    X_rs = np.zeros((S, R)); X_rs[:, :4] = rng.uniform(1, 5, (S, 4))
    AA = np.zeros((R, R_D), dtype=int)
    for r in range(R):
        AA[r, r % R_D] = 1
    emp_gamma = np.zeros((S, R)); emp_gamma[:, :6] = rng.uniform(0.05, 0.5, (S, 6))
    np.save(f"{inp}/filter_N_upstream.npy", filt)
    np.save(f"{inp}/X_rs.npy", X_rs)
    np.save(f"{inp}/attraction_area_linkages.npy", AA)
    np.save(f"{inp}/emp_gamma_ls.npy", emp_gamma)
    np.save(f"{inp}/input_share.npy", rng.dirichlet(np.ones(S)))
    np.save(f"{inp}/domestic_share.npy", np.full(S, 0.8))
    np.save(f"{inp}/N_downstream_per_region.npy",
            np.array([3, 4, 2, 5] + [0] * (R - R_D), dtype=float))
    np.save(f"{inp}/reg_coef_cloglog_{n_coef}.npy", rng.normal(0, 1, n_coef))
    _D = np.abs(rng.normal(300, 80, (R, R))) + 10.0
    _D = (_D + _D.T) / 2
    np.save(f"{inp}/distances.npy", _D)            # read by `sourcing_geometry`
    np.save(f"{inp}/full_distances.npy", _D)       # read by the loader's geography part
    pd.DataFrame({"name": ["epsilon", "labor"], "value": [-16.0, 0.3]}).to_csv(
        f"{inp}/stats.csv", index=False)
    pd.DataFrame({"ze2010": [f"{1000 + r}" for r in range(R)],
                  "X_dr": rng.uniform(1, 9, R),
                  "downstream": [r < R_D for r in range(R)]}).to_csv(f"{inp}/X_dr.csv",
                                                                     index=False)
    pd.DataFrame({"A129": np.repeat(["A", "B", "C"], R),
                  "ze2010": list(range(1000, 1000 + R)) * S}).to_csv(
        f"{inp}/filter_N_upstream.csv", index=False)
    pd.DataFrame({"group": 1, "A129": np.repeat(["A", "B", "C"], 4),
                  "K": list(range(4)) * S, "G": np.tile([0.4, 0.7, 0.9, 1.0], S),
                  "N_supplier_s": np.repeat([20, 24, 30], 4)}).to_csv(f"{inp}/G_K.csv",
                                                                     index=False)
    np.save(f"{run}/n_reg_coef.npy", np.array(n_coef))
    np.save(f"{run}/n_tau.npy", np.array(1))

    n_T = 0
    aa_of_ze = AA.argmax(axis=1)
    sup_cells = (filt == 1) & (X_rs > 0)
    AA_ACTIVE = np.zeros((S, R_D), dtype=bool)
    for s in range(S):
        AA_ACTIVE[s, aa_of_ze[sup_cells[s]]] = True
    n_T = int(AA_ACTIVE.sum())
    theta_hat = np.concatenate([[0.31], rng.uniform(.2, .5, S), rng.uniform(.8, 1.2, R_D),
                                [0.4], rng.lognormal(0, .3, n_T)])
    np.save(f"{run}/step3/best_parameters_list.npy", theta_hat[:, None])
    np.save(f"{run}/step3/post_hoc_N_hat.npy", np.array([4, 7, 5]))

    KW = dict(base=TMP, profile_T=True, ca_level="aa", granular=True,
              relax_n_lo=False, optimizer="pso")
    return TMP, KW


KW_TEMPLATE = dict(profile_T=True, ca_level="aa", granular=True,
                   relax_n_lo=False, optimizer="pso")
