"""A synthetic `baseline_<industry>/` in the data notebook's export format, for running
the Julia code (load_parameters.jl and everything above it) without the real data.

    python3 test/make_synthetic_baseline.py <out_dir> [S] [R] [R_D] [seed]

Writes every file load_parameters.jl reads. The geography is a 600 km square, the
attraction area of a ZE is its closest downstream region (the partition the loader
asserts), and the observed sourcing shares decay with distance so the extensive margin
has content. The beta targets are plausible numbers, not the cloglog of the planted
data: load_parameters.jl warns about the gap and carries on.
"""
import os
import sys

import numpy as np
import pandas as pd


def build(out, S=3, R=60, R_D=6, seed=7):
    rng = np.random.default_rng(seed)
    os.makedirs(out, exist_ok=True)
    xy = rng.uniform(0, 600, (R, 2))
    D = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    D[np.diag_indices(R)] = rng.uniform(5, 15, R)          # own-zone internal distance
    down = np.sort(rng.choice(R, R_D, replace=False))
    N_down = np.zeros(R)
    N_down[down] = rng.integers(2, 9, R_D)
    aa = np.argmin(D[:, down], axis=1)
    AA = np.zeros((R, R_D), dtype=np.int64)
    AA[np.arange(R), aa] = 1

    filt = (rng.random((S, R)) < 0.8).astype(np.int64)
    dmin = D[:, down].min(axis=1)
    p_sup = 1 - np.exp(-np.exp(0.8 - 0.006 * dmin))           # suppliers closer to demand
    sup = (filt == 1) & (rng.random((S, R)) < p_sup[None, :])
    for s in range(S):                                       # every area keeps a supplier
        for a in range(R_D):
            z = np.flatnonzero((aa == a) & (filt[s] == 1))
            if z.size and not sup[s, z].any():
                sup[s, z[0]] = True
    domestic = rng.uniform(0.6, 0.9, S)
    gam = np.zeros((S, R))
    for s in range(S):
        w = rng.lognormal(0, 0.8, sup[s].sum())
        gam[s, sup[s]] = w / w.sum() * domestic[s]

    ze = [f"{1000 + r}" for r in range(R)]
    sectors = [f"C{20 + s}A" for s in range(S)]
    np.save(f"{out}/distances.npy", D)
    np.save(f"{out}/filter_N_upstream.npy", filt)
    np.save(f"{out}/w_rs.npy", np.ones(R))
    np.save(f"{out}/regional_wages.npy", np.ones(R))
    np.save(f"{out}/N_downstream_per_region.npy", N_down)
    np.save(f"{out}/input_share.npy", rng.dirichlet(np.full(S, 4.0)))
    np.save(f"{out}/domestic_share.npy", domestic)
    np.save(f"{out}/emp_gamma_ls.npy", gam)
    np.save(f"{out}/attraction_area_linkages.npy", AA)
    pd.DataFrame({"stats": ["epsilon", "labor_share"], "value": [-4.0, 0.35]}).to_csv(
        f"{out}/stats.csv", index=False)
    pd.DataFrame({"ze2010": ze, "X_dr": np.where(N_down > 0, rng.uniform(50, 500, R), 0.0),
                  "downstream": N_down > 0}).to_csv(f"{out}/X_dr.csv", index=False)
    pd.DataFrame([{"A129": sectors[s], "ze2010": int(ze[r])} for s in range(S) for r in range(R)]
                 ).to_csv(f"{out}/filter_N_upstream.csv", index=False)

    # count moment: the share of kept cells with no supplier, inside active areas
    active = np.zeros((S, R_D), dtype=bool)
    for s in range(S):
        active[s, aa[sup[s]]] = True
    cells = (filt == 1) & active[:, aa]
    rows = []
    for s in range(S):
        c = cells[s]
        n_sup = int(sup[s].sum())
        g0 = 1 - sup[s][c].mean()
        for K in range(4):
            rows.append({"group": s + 1, "A129": sectors[s], "G": min(1.0, g0 + 0.15 * K),
                         "K": K, "N_supplier_s": 4 * n_sup})
    pd.DataFrame(rows).to_csv(f"{out}/G_K.csv", index=False)
    n_ls = rng.integers(1, 30, (S, R))
    pd.DataFrame([{"A129": sectors[s], "ze2010": int(ze[r]), "n_ls": int(n_ls[s, r]),
                   "has_sup": int(sup[s, r])} for s in range(S) for r in range(R) if cells[s, r]]
                 ).to_csv(f"{out}/cells_ls.csv", index=False)

    for n, bd in ((1, [-0.5]), (4, [-0.3, -0.6, -0.9, -1.2])):
        np.save(f"{out}/beta_d_{n}.npy", np.array(bd))
        np.save(f"{out}/beta_kappa_{n}.npy", np.array([0.4]))
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    build(a[0], *(int(x) for x in a[1:]))
    print("written", a[0])
