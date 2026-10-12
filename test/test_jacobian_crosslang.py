#!/usr/bin/env python3
"""
Does the Python Jacobian (jacobian_lib.py) agree with Julia's (compute_jacobian)?

PART A -- synthetic, always runs, no Julia needed. Gates the PORT on its own terms and
gates the COMPARATORS on known answers, so a PASS in part B means something:

  A1  the rebuilt layout: masked moment count, identified-parameter count
  A2  the identities the moment function must satisfy (shares sum to one, gamma sums to
      the domestic share, pi_r sums to one)
  A3  the streaming cloglog IRLS against statsmodels' GLM (CLogLog link, explicit
      fixed-effect dummies) on the SAME solved economy -- an independent route
  A4  the count estimator: E_k[gbar(k)] = (1 - q)^n exactly, summing over Binomial(m, q)
  A5  the replication: same draws -> bit-identical J; the cached path (winners and the
      regression reused across head-parameter columns) equals a brute-force central
      difference EXACTLY; the warm-started IRLS (default, fast) matches the cold start
      Julia uses; splitting a replication's columns across worker processes changes
      nothing, bit for bit; structural zeros are EXACT; N stays fixed (only block 6
      moves with N)
  A6  the comparators: Level 1 passes on a self-dump and fails on a planted 1e-5 error;
      Level 2 passes on two independent seeds and fails on a planted 50% column bias;
      Level 3 passes on two seeds and fails on a planted error
  A7  the rebuild from a real loader dict (fixture run tree): same moment mask, same
      parameter layout, N_hat read from post_hoc_N_hat.npy, a wrong area map refused

PART B -- the real comparison, run when a run tree carries Julia's dump (set
JAC_DUMP_REPS=2 before main.jl; the files land in <run>/step3/jacobian_crosslang/):

  python3 test/test_jacobian_crosslang.py --industry aero [--K 50] [--workers 8]

  B1  Level 1  same draws, every dumped replication: moments to 1e-10 (1e-7 on the
               cloglog block), J_k to 1e-6 relative, identical zero pattern
  B2  Level 2  independent draws: Bonferroni max|z|, block bias, TOST equivalence on the
               elasticity, structural zeros
  B3  Level 3  sandwich standard errors with Julia's own W_step3 and Omega
"""
import argparse
import os
import sys
from math import comb

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import jacobian_lib as jl  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


# ============================================================================
# PART A -- synthetic
# ============================================================================

def synthetic(seed=0, S=3, R=30, n_down=5, n_reg=4, n_tau=1):
    """A small economy whose cells occupy EVERY distance bin, the base category included
    -- otherwise the bin dummies are collinear with the fixed effect and the regression
    is singular (in Julia as here)."""
    for sd in range(seed, seed + 200):
        m, th, N = _synthetic(sd, S, R, n_down, n_reg, n_tau)
        cr = m.good_r
        b = m.dist_bin[cr, m.closest_dr[cr]]
        if set(range(n_reg + 1)) <= set(b.tolist()):
            return m, th, N
    raise RuntimeError("no synthetic geography occupies every distance bin")


def _synthetic(seed, S, R, n_down, n_reg, n_tau):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 450, (R, 2))
    D = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
    D[np.arange(R), np.arange(R)] = rng.uniform(5, 15, R)      # own-zone distance > 0
    Nd = np.zeros(R)
    down = rng.choice(R, n_down, replace=False)
    Nd[down] = rng.integers(1, 5, n_down)
    fil = (rng.random((S, R)) < 0.8).astype(int)
    gam = rng.random((S, R)) * fil
    gam[rng.random((S, R)) < 0.3] = 0.0                       # control cells
    for s in range(S):                                         # every sector supplied
        gam[s, rng.choice(np.flatnonzero(fil[s]), 3, replace=False)] += 0.2
    ds = rng.uniform(0.7, 0.9, S)
    gam = gam / gam.sum(1, keepdims=True) * ds[:, None]
    m = jl.EstimatorModel(distances=D, N_downstream=Nd, filter_N_upstream=fil,
                          emp_gamma_ls=gam, domestic_share=ds, regional_wages=np.ones(R),
                          epsilon=-16.0, theta=1.768, n_reg=n_reg, n_tau=n_tau,
                          ca_level="aa", granular=True,
                          G_target=np.full(S, 0.6), N_supplier_s=np.full(S, 40.0))
    th = np.r_[0.3, rng.uniform(0.5, 1.5, S), rng.uniform(0.5, 1.5, m.Rd), 0.4,
               np.exp(rng.normal(0, 0.7, m.n_T))]
    N = np.full(S, 6)
    return m, th, N


def part_a():
    print("=" * 70 + "\nPART A -- synthetic\n" + "=" * 70)
    m, th, N = synthetic()
    n_rho = 400
    u = np.random.default_rng(1).random((n_rho, m.n_good))

    # A1
    print("A1  layout")
    check("parameter vector length", th.size == m.n_theta, f"{m.n_theta}")
    check("identified columns = all minus the S+2 normalised directions",
          m.param_indices.size == m.n_theta - (m.S + 2))
    v, info = m.moments(th, u, N_fixed=N, full=True)
    check("masked moment count", v.size == int(m.moment_mask.sum()), f"{v.size}")

    # A2
    print("A2  identities")
    b = info["blocks"]
    check("industry shares sum to one", abs(np.sum(b[1]) - 1) < 1e-12)
    check("pi_r sums to one", abs(np.sum(b[2]) - 1) < 1e-12)
    check("gamma (AA level) sums to the domestic share in every sector",
          np.allclose(info["gamma_T"].sum(0), m.domestic_share, atol=1e-12))
    check("labour share in (0, 1)", 0 < b[0][0] < 1, f"{b[0][0]:.4f}")
    check("finite", np.all(np.isfinite(v)))

    # A3
    print("A3  cloglog IRLS against statsmodels GLM with explicit dummies")
    import statsmodels.api as sm
    sol = info["sol"]
    C = m._reg_cells(sol["logz_const"], n_rho)
    lzr = -(1.0 / m.theta) * np.log(-np.log(1.0 - u))
    nc = C["n_cell"]
    y = np.where(sol["linkages"][:, :nc], 0.0, 1.0).ravel(order="F")
    cols = []
    for bb in range(1, m.n_reg + 1):
        cols.append(np.repeat(((C["gcol"] == bb) * C["gval"]), n_rho))
    lz = (C["lzc"][None, :] + lzr[:, :nc]).ravel(order="F")
    X = np.column_stack(cols + [lz])
    FE = np.zeros((X.shape[0], C["n_grp"]))
    FE[np.arange(X.shape[0]), np.repeat(C["grp"], n_rho)] = 1.0
    keep = X[:, :m.n_reg].std(0) > 0
    Xd = np.column_stack([X[:, :m.n_reg][:, keep], X[:, -1], FE])
    glm = sm.GLM(y, Xd, family=sm.families.Binomial(link=sm.families.links.CLogLog()),
                 var_weights=np.full(y.size, 1.0 / n_rho)).fit(tol=1e-12, maxiter=200)
    b_ref = np.zeros(m.n_reg)
    b_ref[keep] = glm.params[:keep.sum()]
    reg, blogz = info["reg"]["coef"], info["reg"]["b_logz"]
    check("distance coefficients match GLM", np.allclose(reg[keep], b_ref[keep], atol=1e-6),
          f"max gap {np.max(np.abs(reg[keep] - b_ref[keep])):.1e}")
    check("log-z coefficient matches GLM", abs(blogz - glm.params[keep.sum()]) < 1e-6,
          f"{blogz:.4f} vs {glm.params[keep.sum()]:.4f}")

    # A4
    print("A4  count estimator")
    mm, n, q = 60, 9, 0.07
    lg = jl.gbar_logfact(mm)
    ks = np.arange(mm + 1)
    pmf = np.array([comb(mm, k) * q ** k * (1 - q) ** (mm - k) for k in ks])
    check("E[gbar] = (1-q)^n exactly", abs((pmf * jl.gbar_cells(ks, mm, n, lg)).sum()
                                         - (1 - q) ** n) < 1e-12)
    check("gbar = C(m-k,n)/C(m,n)", np.allclose(
        jl.gbar_cells(ks, mm, n, lg),
        [comb(mm - k, n) / comb(mm, n) if n <= mm - k else 0.0 for k in ks], rtol=1e-10))

    # A5
    print("A5  the replication")
    idx = m.param_indices
    # strict path (cold-start IRLS, as Julia) for the exactness checks
    J1, _, m01, dN = jl.jacobian_replication(m, th, u, idx, 1e-2, N, warm_start=False)
    J2, _, _, _ = jl.jacobian_replication(m, th, u, idx, 1e-2, N, warm_start=False)
    check("same draws -> bit-identical J", np.array_equal(J1, J2))
    Jw, _, _, _ = jl.jacobian_replication(m, th, u, idx, 1e-2, N, warm_start=True)
    r4_ = m.block_ranges[3]
    oth = np.setdiff1d(np.arange(J1.shape[0]), r4_)
    # the IRLS stops at a 1e-9 step, so ANY finite-difference Jacobian -- Julia's too --
    # resolves the cloglog rows only to ~tol/(2 delta theta_i); every other row must match
    allow = 1e-9 / (2 * 1e-2 * th[idx])[None, :]
    ratio = np.max(np.abs(Jw[r4_] - J1[r4_]) / allow)
    check("warm-started IRLS (the fast default) == cold start: other rows bit-identical",
          np.array_equal(Jw[oth], J1[oth]))
    check("... cloglog rows within the IRLS tolerance's own resolution",
          ratio < 10, f"{ratio:.2f} x tol/(2 delta theta)")
    probe = [int(np.flatnonzero(idx == 0)[0]),                     # Omega_L
             int(np.flatnonzero(idx >= 1 + m.S)[0]),               # an A
             int(np.flatnonzero(idx == m.alpha_start)[0]),         # alpha
             int(np.flatnonzero(idx >= m.T_start)[0]),             # a T
             len(idx) - 1]                                         # the last T
    err = 0.0
    for j in probe:
        i = idx[j]
        tp, tm = th.copy(), th.copy()
        tp[i] *= np.exp(1e-2)
        tm[i] *= np.exp(-1e-2)
        ref = (m.moments(tp, u, N_fixed=N) - m.moments(tm, u, N_fixed=N)) / (2e-2 * th[i])
        err = max(err, np.max(np.abs(ref - J1[:, j])))
    check("cached columns == brute-force central differences", err == 0.0, f"{err:.1e}")
    head = np.flatnonzero(idx < m.alpha_start)
    r4, r6 = m.block_ranges[3], m.block_ranges[5]
    check("head columns: block 4 and block 6 EXACTLY zero (winners do not move)",
          np.all(J1[np.ix_(r4, head)] == 0) and np.all(J1[np.ix_(r6, head)] == 0))
    tcols = np.flatnonzero(idx >= m.T_start)
    ts = m.T_red_s[idx[tcols] - m.T_start]
    off = [(r6[s2], j) for j, s in zip(tcols, ts) for s2 in range(m.S) if s2 != s]
    check("a T column moves no OTHER sector's zero-supplier share",
          all(J1[i, j] == 0 for i, j in off))
    other = np.setdiff1d(np.arange(J1.shape[0]), r6)
    check("N_s columns: zero outside block 6, diagonal inside",
          np.all(dN[other] == 0) and np.all(dN[np.ix_(r6, range(m.S))][~np.eye(m.S, dtype=bool)] == 0)
          and np.all(np.diag(dN[r6]) != 0))
    vN1 = m.moments(th, u, N_fixed=N + 1)
    check("N enters block 6 only (the economy is not re-solved for N)",
          np.array_equal(vN1[other], m01[other]))

    one = jl.python_jacobian(m, th, N, K=3, n_rho=n_rho, base_seed=77, n_workers=1,
                             n_chunks=1, verbose=False)
    many = jl.python_jacobian(m, th, N, K=3, n_rho=n_rho, base_seed=77, n_workers=3,
                              n_chunks=4, verbose=False)
    check("column chunks over worker processes == one job per replication, bit for bit",
          np.array_equal(one["J_stack"], many["J_stack"])
          and np.array_equal(one["m0_stack"], many["m0_stack"]))
    check("chunk planner: one chunk when replications >= cores, else >= 10 columns each",
          jl._plan_chunks(200, 50, 48) == 1 and jl._plan_chunks(200, 50, 200) == 4
          and jl._plan_chunks(25, 2, 64) == 2)

    # A6
    print("A6  the comparators on known answers")
    dump = {"u": u, "m0": m01, "J": J1}
    r = jl.compare_same_draws(m, th, N, dump, verbose=False)
    check("Level 1 passes on a self-dump", r["ok"])
    bad = dict(dump, m0=m01.copy())
    bad["m0"][r4[0]] *= 1 + 1e-5
    check("Level 1 catches a 1e-5 error on a cloglog coefficient",
          not jl.compare_same_draws(m, th, N, bad, verbose=False)["ok"])
    badJ = dict(dump, J=J1.copy())
    badJ["J"][m.block_ranges[4][0], tcols[0]] *= 1.001
    check("Level 1 catches a 0.1% error in one Jacobian entry",
          not jl.compare_same_draws(m, th, N, badJ, verbose=False)["ok"])

    K = 16
    a = jl.python_jacobian(m, th, N, K=K, n_rho=n_rho, base_seed=10, n_workers=4,
                           verbose=False)
    b2 = jl.python_jacobian(m, th, N, K=K, n_rho=n_rho, base_seed=10_000, n_workers=4,
                            verbose=False)
    br = m.block_ranges
    r2 = jl.compare_independent(a, b2, block_ranges=br, rel=0.25, verbose=False)
    check("Level 2 passes on two independent seeds", r2["ok"],
          f"max|z| {r2['max_abs_z']:.2f}, {len(r2['biased_groups'])} biased, "
          f"{r2['equivalence_fail']} proven off, "
          f"{r2['n_proven_equivalent']}/{r2['n_equivalence']} proven equivalent, "
          f"detectable gap {r2['mde_median']:.0%}")
    planted = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in b2.items()}
    jcol = int(np.argmax(np.abs(a["J"]).sum(0)[tcols])) + tcols[0]
    planted["J"][:, jcol] *= 1.5
    planted["J_elast"][:, jcol] *= 1.5
    planted["J_stack"][:, jcol, :] *= 1.5
    r2b = jl.compare_independent(a, planted, block_ranges=br, rel=0.10, verbose=False)
    check("Level 2 catches a 50% bias in one column", not r2b["ok"],
          f"{len(r2b['biased_groups'])} biased groups, {r2b['outliers']} outliers, "
          f"{r2b['equivalence_fail']} proven off")

    rows, cols = jl.inference_slices(m)
    W = np.eye(rows.size)
    Om = np.eye(rows.size) * 1e-4
    r3 = jl.compare_standard_errors(a, b2, W, Om, rows, cols, floor=0.10, verbose=False)
    check("Level 3 passes on two independent seeds", r3["ok"],
          f"max |log ratio| {np.abs(r3['log_ratio']).max():.3f}")
    p3 = dict(b2, J=b2["J"].copy())
    # plant an error well outside the measured Monte-Carlo noise of this tiny fixture
    shrink = float(np.exp(-max(1.0, 4 * np.abs(r3["log_ratio"]).max())))
    p3["J"][:, cols[0]] *= shrink
    p3.pop("J_stack")
    r3b = jl.compare_standard_errors(a, p3, W, Om, rows, cols, floor=0.10, verbose=False)
    check(f"Level 3 catches one column scaled by {shrink:.2f}", not r3b["ok"])


def part_a_loader():
    """A7: the rebuild from a REAL loader dict agrees with the loader's own layout."""
    print("A7  rebuild from utils.load_granular_data on a fixture run tree")
    import warnings
    from _fixture_tree import build
    from utils import load_granular_data
    tmp, kw = build()
    inp = os.path.join(tmp, "baseline_test")
    D = np.load(f"{inp}/distances.npy")
    AA = np.load(f"{inp}/attraction_area_linkages.npy")
    aa = AA.argmax(1)
    D[np.arange(D.shape[0]), aa] = 5.0      # anchor each area on the zone's NEAREST buyer,
    np.save(f"{inp}/distances.npy", D)      # the partition load_parameters.jl asserts
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = load_granular_data("test", mu=2, parts=("core", "counts"), **kw)
        m = jl.EstimatorModel.from_data(data)          # raises on a mask disagreement
    check("moment mask identical to the loader's", True)
    th, N = jl.theta_plus(data)
    check("theta+ length == the parameter layout rebuilt here", th.size == m.n_theta,
          f"{th.size} vs {m.n_theta}")
    check("N_hat read from post_hoc_N_hat.npy", N is not None and N.size == m.S)
    check("identified columns == the loader's parameter labels (before N_s)",
          m.param_indices.size == len(data["param_labels"]),
          f"{m.param_indices.size} vs {len(data['param_labels'])}")
    # this fixture puts every cell in the base distance bin: the regression is not
    # identified, and the port must stop with Julia's diagnosis, not return numbers
    u = np.random.default_rng(3).random((200, m.n_good))
    try:
        m.moments(th, u, N_fixed=N)
        check("a rank-deficient distance design is refused, as Julia refuses it", False)
    except ValueError as e:
        check("a rank-deficient distance design is refused, as Julia refuses it",
              "rank-deficient" in str(e))
    bad = dict(data, aa_of_ze=(np.asarray(data["aa_of_ze"]) + 1) % m.Rd)
    try:
        jl.EstimatorModel.from_data(bad)
        check("an area map that is not the nearest-buyer partition is refused", False)
    except ValueError:
        check("an area map that is not the nearest-buyer partition is refused", True)


# ============================================================================
# PART B -- against Julia
# ============================================================================

def part_b(args):
    print("=" * 70 + f"\nPART B -- against Julia ({args.industry}, mu = 2)\n" + "=" * 70)
    from utils import load_granular_data
    data = load_granular_data(args.industry, mu=2, base=args.base,
                              parts=("core", "counts", "moments", "inference", "jacobian"))
    dump_dir = data["folder"] / "step3" / "jacobian_crosslang"
    if not dump_dir.exists():
        print(f"  SKIP  no Julia dump in {dump_dir}: run main.jl with JAC_DUMP_REPS=2")
        return
    m = jl.EstimatorModel.from_data(data)
    th = np.load(dump_dir / "theta.npy").ravel()
    N = np.load(dump_dir / "N_fixed.npy").ravel().astype(int)
    step = float(np.load(dump_dir / "step_rel.npy").ravel()[0])
    n_rho = int(np.load(dump_dir / "n_rho.npy").ravel()[0])
    pidx = np.load(dump_dir / "param_indices.npy").ravel().astype(int) - 1
    check("Julia's parameter columns == the ones rebuilt here",
          np.array_equal(pidx, m.param_indices))
    N_hat = data.get("post_hoc_N_hat")
    if N_hat is not None:
        check("Julia pinned N at theta+ (= post_hoc_N_hat.npy)",
              np.array_equal(np.asarray(N_hat).ravel()[:m.S], N),
              f"{list(N)} vs {list(np.asarray(N_hat).ravel()[:m.S])}")

    print("B1  Level 1 -- same draws")
    reps = sorted(int(p.stem[5:]) for p in dump_dir.glob("u_rep*.npy"))
    for k in reps:
        d = {"u": np.load(dump_dir / f"u_rep{k}.npy"),
             "m0": np.load(dump_dir / f"m0_rep{k}.npy"),
             "J": np.load(dump_dir / f"J_rep{k}.npy")}
        r = jl.compare_same_draws(m, th, N, d, step=step)
        check(f"replication {k}: moments and J_k agree", r["ok"],
              "" if r["ok"] else f"first gaps at {r.get('J_bad_where')}")

    print(f"B2  Level 2 -- independent draws (Python K = {args.K}, n_rho = {n_rho})")
    py = jl.python_jacobian(m, th, N, K=args.K, n_rho=n_rho, step=step,
                            n_workers=args.workers)
    np.savez(dump_dir / "python_jacobian.npz",
             **{k: v for k, v in py.items() if isinstance(v, np.ndarray)})
    julia = {"J": data["J"], "J_sd": data["J_sd"], "J_elast": data["J_elast"],
             "n_reps": (data["jacobian_n_reps"] if data.get("jacobian_n_reps") is not None
                        else np.r_[np.full(len(pidx), 50.0), np.full(data["n_N_cols"], 10.0)]),
             "J_stack": np.load(dump_dir / "J_stack.npy")}
    r2 = jl.compare_independent(py, julia, block_ranges=m.block_ranges)
    check("Level 2: same expectation, equivalent elasticities", r2["ok"])

    print("B3  Level 3 -- standard errors")
    rows, cols = jl.inference_slices(m)
    W, Om = data.get("W_step3"), data.get("Omega")
    if W is None or Om is None or W.shape[0] != rows.size:
        print("  SKIP  no W_step3 / Omega of the inference size on disk")
        return
    r3 = jl.compare_standard_errors(py, julia, W, Om, rows, cols)
    check("Level 3: sandwich SEs agree within Monte-Carlo error", r3["ok"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--industry", default=None)
    ap.add_argument("--base", default="..")
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    part_a()
    part_a_loader()
    if args.industry:
        part_b(args)
    print("\n" + ("ALL GATES PASSED" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}"))
    sys.exit(1 if FAILS else 0)
