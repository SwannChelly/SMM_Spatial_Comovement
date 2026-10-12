"""
The estimator's moment function and its finite-difference Jacobian, in Python.

`utils.simulate_economy` reproduces the REALISED economy (N_hat varieties, weights
1/N_hat) that the reporting reads. The ESTIMATOR does not evaluate that object: its
moments are computed on `n_rho` draws per cell with flat weights 1/n_rho (the continuum
design), block 4 is a cloglog regression with a fixed effect and a log-z size control,
and block 6 is the unbiased count estimator `gbar_sector` at a FIXED N_hat. This module
is a line-by-line port of that path -- `unpack_params`, `build_tau`, `solve_network`,
`compute_moments`, `_build_reg_cells`, `_cloglog_irls_cells`, `gbar_sector`,
`concentrate_N_s`, `compute_jacobian` (model_CP.jl / tools.jl) -- acting on the RAW
parameter vector in Julia's layout

    theta = [Omega_L | Omega_s (S) | A (R_downstream) | alpha (N_TAU) | T (sum T_MASK)]

so a finite-difference column here is the same column as Julia's, index for index.

The extended parameter set theta+ is held fixed in every evaluation: T is a coordinate of
theta (the profiled T* at the estimate, never re-inverted here) and N_s is passed in
(`N_fixed`, = `post_hoc_N_hat.npy` = Julia's `estimator_N_hat`), never re-profiled.

What is deliberately NOT ported: the analytical/GMM path, `profile_T` (the alpha-only
Jacobian along T*(alpha)), the legacy random-Frechet branch, and non-flat sample weights
other than as an explicit matrix.

Index conventions (all 0-based here, 1-based in Julia):
  * goods g enumerate CELL_MASK COLUMN-MAJOR over (S, R) -- region outer, sector inner --
    exactly Julia's `findall(cell_mask)`; a draw matrix is (n_rho, n_good) in that order;
  * T and gamma are s-major over (S, T_COL_DIM), Julia's `vec(permutedims(.))`.
"""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ETA_CLAMP = 30.0

# Optional compiled kernels. The IRLS per-cell sums are ~85% of an evaluation; numba runs
# them as the same fused loop Julia does (sequential sums, ~20x the numpy speed). Without
# numba the numpy path is used: identical up to summation order, only slower.
try:
    import numba as _nb
except ImportError:            # pragma: no cover
    _nb = None

if _nb is not None:
    @_nb.njit(cache=True, parallel=True)
    def _cell_sums_nb(link, lzr, has_lzr, is_good, has_z, lzc, Acell, b_size, first,
                      wflat, eta_clamp):
        # link and lzr arrive TRANSPOSED, (cell, draw), so the inner loop over draws
        # reads contiguous memory -- Julia's column-major layout, which numpy's is not
        n_rho = link.shape[1]
        n_cell = is_good.size
        out = np.zeros((5, n_cell))
        e00 = min(max(np.log(-np.log(1.0 - 0.25)), -eta_clamp), eta_clamp)
        e01 = min(max(np.log(-np.log(1.0 - 0.75)), -eta_clamp), eta_clamp)
        for c in _nb.prange(n_cell):        # each cell's sum stays sequential
            g = is_good[c]
            hz = has_lzr and has_z[c]
            ac = Acell[c]
            lc = lzc[c]
            sw = 0.0
            swz = 0.0
            swl = 0.0
            swll = 0.0
            swlz = 0.0
            for r in range(n_rho):
                if g:
                    yi = 0.0 if link[c, r] else 1.0
                else:
                    yi = 1.0
                lzr_i = lzr[c, r] if hz else 0.0
                lz = lc + lzr_i if hz else 0.0
                if first:
                    eta = e01 if yi > 0.5 else e00
                else:
                    eta = min(max(ac + b_size * lzr_i, -eta_clamp), eta_clamp)
                ee = np.exp(eta)
                mu = min(max(1.0 - np.exp(-ee), 1e-12), 1.0 - 1e-12)
                d = ee * (1.0 - mu)
                zw = eta + (yi - mu) / d
                W = wflat * (d * d) / (mu * (1.0 - mu))
                sw += W
                swz += W * zw
                if hz:
                    swl += W * lz
                    swll += W * lz * lz
                    swlz += W * lz * zw
            out[0, c] = sw
            out[1, c] = swz
            out[2, c] = swl
            out[3, c] = swll
            out[4, c] = swlz
        return out


def distance_bin(d, n_bins):
    """tools.jl `distance_bin`, vectorised. 0 is the base category."""
    d = np.asarray(d, float)
    if n_bins == 1:
        return np.zeros(d.shape, int)
    if n_bins == 5:
        edges = [(20, 50, 1), (50, 100, 2), (100, 150, 3), (150, 200, 4)]
        last = 5
    elif n_bins == 4:
        edges = [(50, 100, 1), (100, 150, 2), (150, 200, 3)]
        last = 4
    else:
        raise ValueError(f"unsupported number of distance bins: {n_bins}")
    out = np.zeros(d.shape, int)
    for lo, hi, b in edges:
        out[(d > lo) & (d <= hi)] = b
    out[d > 200] = last
    return out


def gbar_logfact(m):
    """lg[i] = log(i!) for i = 0..m, by cumulative sum as Julia builds it."""
    return np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, m + 1, dtype=float)))])


def gbar_cells(k, m, n, lg):
    """Unbiased estimator of (1 - q)^n per cell from k wins of m draws (`gbar_cell`)."""
    k = np.asarray(k, int)
    out = np.zeros(k.shape)
    if n <= 0:
        return np.ones(k.shape)
    ok = n <= m - k
    kk = k[ok]
    out[ok] = np.exp(lg[m - kk] - lg[m - kk - n] - lg[m] + lg[m - n])
    return out


def _solve_within(A, rhs, ctx):
    """`_solve_within_normal_eq`: the k x k within-group solve, with Julia's diagnosis."""
    try:
        return np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        raise ValueError(
            f"{ctx}: the within-group normal equations are singular (rank-deficient "
            "design). Usual cause: no cell in the BASE distance category, or a bin no cell "
            "occupies, so the dummies are collinear with the (sector x nearest-downstream) "
            "fixed effect. Julia stops here too.") from None


class EstimatorModel:
    """
    The constants `load_parameters.jl` builds, rebuilt from the same primitives, plus the
    moment function. Construct with `from_data(data)` for a run tree, or directly from
    primitives (the synthetic gate does that).
    """

    def __init__(self, *, distances, N_downstream, filter_N_upstream, emp_gamma_ls,
                 domestic_share, regional_wages, epsilon, theta, n_reg, n_tau,
                 ca_level="aa", granular=True, reg_method="cloglog",
                 include_control=True, nu_s=1.5, nu=0.2, lam=0.5,
                 G_target=None, N_supplier_s=None, delta=None):
        fil = np.asarray(filter_N_upstream) == 1
        S, R = fil.shape
        self.S, self.R = S, R
        self.theta, self.epsilon = float(theta), float(epsilon)
        self.nu_s = np.full(S, float(nu_s)) if np.ndim(nu_s) == 0 else np.asarray(nu_s, float)
        self.nu, self.lam = float(nu), float(lam)
        self.n_reg, self.n_tau = int(n_reg), int(n_tau)
        self.ca_level, self.granular = ca_level, bool(granular)
        self.reg_method = reg_method
        if self.granular and ca_level != "aa":
            raise ValueError("granular=True requires ca_level='aa' (as in load_parameters.jl)")

        # ---- geography (load_parameters.jl SECTION 2c / 10)
        Nd = np.asarray(N_downstream).ravel()[:R]
        self.down = np.flatnonzero(Nd > 0)
        self.active = Nd != 0
        self.Rd = Rd = self.down.size
        D = np.asarray(distances, float)[:R, :R]
        Dd = D[:, self.down]
        self.closest_dist = Dd.min(axis=1)
        self.closest_dr = Dd.argmin(axis=1)                 # first minimiser, as Julia
        self.log_dist_down = np.log(np.maximum(Dd, 1.0))
        self.log_closest = np.log(np.maximum(self.closest_dist, 1.0))
        self.dist_bin = distance_bin(Dd, self.n_reg)
        self.delta = np.ones(R) if delta is None else np.asarray(delta, float)

        # wages: normalised to 1 where observed, 0 elsewhere (load_parameters.jl SECTION 1)
        rw = np.asarray(regional_wages, float).ravel()[:R]
        self.rw = np.where(rw != 0, 1.0, 0.0)

        # ---- cells, T columns, reference columns (SECTION 2c-5)
        egl = np.asarray(emp_gamma_ls, float)
        if egl.shape != (S, R):
            raise ValueError(f"emp_gamma_ls is {egl.shape}, expected (S, R) = {(S, R)}")
        observed = egl > 0
        supplier = fil & observed
        if ca_level == "aa":
            aa_of_ze = self.closest_dr.copy()
            n_AA = Rd
            aa_active = np.zeros((S, n_AA), bool)
            for s in range(S):
                aa_active[s, aa_of_ze[supplier[s]]] = True
            cell_mask = fil & aa_active[:, aa_of_ze]
            T_active, T_gather, T_dim = aa_active, aa_of_ze, n_AA
            emp_T = np.zeros((S, n_AA))
            for s in range(S):
                np.add.at(emp_T[s], aa_of_ze[cell_mask[s]], egl[s][cell_mask[s]])
            self.reg_control, self.reg_size = False, True
        else:
            cell_mask = supplier.copy()
            T_active, T_gather, T_dim = supplier, np.arange(R), R
            emp_T = egl.copy()
            self.reg_control, self.reg_size = bool(include_control), not include_control
        self.cell_mask, self.T_active, self.T_gather, self.T_dim = cell_mask, T_active, T_gather, T_dim
        self.T_mask = T_active.ravel()                      # s-major
        self.n_T = int(self.T_mask.sum())
        self.T_ref = np.full(S, -1, int)
        for s in range(S):
            cols = np.flatnonzero(T_active[s])
            if cols.size:
                self.T_ref[s] = cols[np.argmax(emp_T[s, cols])]

        r_idx, s_idx = np.nonzero(cell_mask.T)              # column-major over (S, R)
        self.good_s, self.good_r = s_idx, r_idx
        self.n_good = s_idx.size
        self.sector_goods = [np.flatnonzero(s_idx == s) for s in range(S)]
        cr_idx, cs_idx = np.nonzero((fil & ~observed).T)
        self.ctrl_s, self.ctrl_r = cs_idx, cr_idx
        self.domestic_share = np.asarray(domestic_share, float).ravel()[:S]

        # ---- count moment (granular)
        self.G_target = None if G_target is None else np.asarray(G_target, float)
        if N_supplier_s is not None:
            Ns = np.asarray(N_supplier_s, float)
            self.N_LO = np.ceil(Ns / Rd).astype(int)
            self.N_HI = Ns.astype(int)
        else:
            self.N_LO = self.N_HI = None

        # ---- moment layout and MOMENT_MASK (SECTION 8)
        self.n_pi = Rd
        n_gamma = S * T_dim
        sizes = [1, S, Rd, self.n_reg, n_gamma] + ([S] if self.granular else [])
        self.block_sizes_full = sizes
        mask = np.ones(sum(sizes), bool)
        mask[1] = False
        mask[1 + S] = False
        g0 = 1 + S + Rd + self.n_reg
        mask[g0:g0 + n_gamma] &= self.T_mask
        for s in range(S):
            if self.T_ref[s] >= 0:
                mask[g0 + s * T_dim + self.T_ref[s]] = False
        self.moment_mask = mask
        offs = np.cumsum([0] + sizes)
        self.block_ranges = []
        for b in range(len(sizes)):
            pos = np.flatnonzero(mask[offs[b]:offs[b + 1]])
            start = int(mask[:offs[b]].sum())
            self.block_ranges.append(np.arange(start, start + pos.size))

        # ---- parameter layout and the identified columns (SECTION 11)
        self.n_theta = 1 + S + Rd + self.n_tau + self.n_T
        self.alpha_start = 1 + S + Rd
        self.T_start = self.alpha_start + self.n_tau
        excluded = {1, S + 1}
        cum = np.cumsum(self.T_mask)
        for s in range(S):
            if self.T_ref[s] >= 0:
                fp = s * T_dim + self.T_ref[s]
                if self.T_mask[fp]:
                    excluded.add(self.T_start + int(cum[fp]) - 1)
        self.param_indices = np.array([i for i in range(self.n_theta) if i not in excluded])

        # T-column (s, c) of each reduced-T entry, s-major
        flat = np.flatnonzero(self.T_mask)
        self.T_red_s, self.T_red_c = flat // T_dim, flat % T_dim

    # ------------------------------------------------------------------ build
    @classmethod
    def from_data(cls, data, theta=None, reg_method="cloglog", include_control=True):
        """From `utils.load_granular_data`'s dict. theta defaults to `model_theta(data)`."""
        if theta is None:
            from utils import model_theta
            theta = model_theta(data)
        from utils import NU_S_DEFAULT, NU_ACROSS_DEFAULT, LAMBDA_DEFAULT
        inp = data["input_folder"]
        m = cls(distances=np.load(inp / "distances.npy"),
                N_downstream=np.load(inp / "N_downstream_per_region.npy"),
                filter_N_upstream=data["filter_N_upstream"],
                emp_gamma_ls=data["emp_gamma_ls"],
                domestic_share=data["domestic_share"],
                regional_wages=np.load(inp / "regional_wages.npy"),
                epsilon=data["epsilon"], theta=theta,
                n_reg=data["n_coef"], n_tau=data["n_tau"],
                ca_level=data["ca_level"], granular=data["granular"],
                reg_method=reg_method, include_control=include_control,
                nu_s=NU_S_DEFAULT, nu=NU_ACROSS_DEFAULT, lam=LAMBDA_DEFAULT,
                G_target=data.get("G_target"), N_supplier_s=data.get("N_supplier_s"))
        aa = data.get("aa_of_ze")
        if aa is not None and m.ca_level == "aa" and not np.array_equal(
                np.asarray(aa).ravel()[:m.R], m.closest_dr):
            raise ValueError("attraction_area_linkages.npy disagrees with the nearest "
                             "downstream region -- load_parameters.jl asserts they are the "
                             "same partition, so this tree is not one Julia would run")
        mm = data.get("moment_mask")
        if mm is not None and not np.array_equal(np.asarray(mm, bool), m.moment_mask):
            raise ValueError("the moment mask rebuilt here differs from the loader's -- "
                             "the two reconstructions of load_parameters.jl disagree")
        return m

    # --------------------------------------------------------- unpack / tau
    def unpack(self, theta):
        """`unpack_params`: (Omega_L, Omega_s, A, alpha, T_ze (S, R))."""
        th = np.asarray(theta, float)
        S, Rd = self.S, self.Rd
        Om_L = th[0]
        Om_s = th[1:1 + S] / th[1:1 + S].sum()
        A = th[1 + S:1 + S + Rd]
        A = A / A[0]
        alpha = th[self.alpha_start:self.T_start]
        T_full = np.zeros(S * self.T_dim)
        T_full[self.T_mask] = th[self.T_start:]
        T_par = T_full.reshape(S, self.T_dim)
        for s in range(S):
            c = self.T_ref[s]
            if c >= 0 and T_par[s, c] > 0:
                T_par[s] = T_par[s] / T_par[s, c]
        return Om_L, Om_s, A, alpha, T_par[:, self.T_gather]

    def tau(self, alpha):
        """`build_tau`, (R, R_downstream)."""
        if self.n_tau == 1:
            return np.exp(alpha[0] * self.log_dist_down)
        t = np.ones((self.R, self.Rd))
        b = self.dist_bin
        ok = (b > 0) & (b <= self.n_tau)
        t[ok] += alpha[b[ok] - 1]
        return t

    # -------------------------------------------------------- Ricardian block
    def _sector_winners(self, s, z_inv, tau):
        """Lowest-cost good for every (draw, buyer) of sector s: (winner g, price)."""
        gi = self.sector_goods[s]
        rs = self.good_r[gi]
        zi = z_inv[:, gi]
        n_rho = zi.shape[0]
        win = np.empty((n_rho, self.Rd), dtype=np.int64)
        price = np.empty((n_rho, self.Rd))
        for rd in range(self.Rd):
            p = (zi * tau[rs, rd][None, :]) * 1.0          # z * tau * w, w = 1
            j = p.argmin(axis=1)
            win[:, rd] = gi[j]
            price[:, rd] = p[np.arange(n_rho), j]
        return win, price

    def solve(self, theta, u, w=None, base=None):
        """
        `solve_network`. `base` is an earlier solution on the SAME draws: sectors whose
        (alpha, T row) are unchanged reuse its winners exactly, which is what makes a
        Jacobian column for a head parameter cost a value block rather than a full solve.
        """
        Om_L, Om_s, A, alpha, T = self.unpack(theta)
        u = np.asarray(u, float)
        n_rho = u.shape[0]
        flat_w = (1.0 / n_rho) if w is None else None
        th = self.theta

        same_alpha = base is not None and np.array_equal(base["alpha"], alpha)
        tau = base["tau"] if same_alpha else self.tau(alpha)

        T_g = T[self.good_s, self.good_r]
        scale = np.maximum(T_g, np.finfo(float).eps) ** (1.0 / th)
        logz_const = np.log(scale)

        win, price, recomputed = {}, {}, []
        z_inv = None
        for s in range(self.S):
            gi = self.sector_goods[s]
            if gi.size == 0:
                continue
            if same_alpha and np.array_equal(base["T_g"][gi], T_g[gi]):
                win[s], price[s] = base["win"][s], base["price"][s]
                continue
            if z_inv is None:
                z_inv = np.empty_like(u)
            z = scale[gi][None, :] * (-np.log(1.0 - u[:, gi])) ** (-1.0 / th)
            z_inv[:, gi] = 1.0 / z
            win[s], price[s] = self._sector_winners(s, z_inv, tau)
            recomputed.append(s)

        # value block, per downstream region
        S, Rd = self.S, self.Rd
        nus = self.nu_s
        X_shares = np.zeros((self.n_good, Rd))
        c_tilde = np.zeros(Rd)
        rows = np.arange(n_rho)
        for rd in range(Rd):
            r = self.down[rd]
            P_sr = np.zeros(S)
            wts = {}
            for s in win:
                g_w = win[s][:, rd]
                ws = flat_w if w is None else np.asarray(w)[rows, g_w]
                wts[s] = ws
                P_sr[s] = np.sum(ws * price[s][:, rd] ** (1 - nus[s])) ** (1 / (1 - nus[s]))
            P_r = np.sum(P_sr ** (1 - self.nu) * Om_s) ** (1 / (1 - self.nu))
            c_r = (Om_L * self.rw[r] ** (1 - self.lam)
                   + (1 - Om_L) * P_r ** (1 - self.lam)) ** (1 / (1 - self.lam))
            c_tilde[rd] = c_r / A[rd]
            for s in win:
                ev = (wts[s] * Om_s[s] * (1 - Om_L)
                      * (price[s][:, rd] / P_sr[s]) ** (1 - nus[s])
                      * (P_sr[s] / P_r) ** (1 - self.nu)
                      * (P_r / c_r) ** (1 - self.lam))
                if np.ndim(ev) == 0:
                    ev = np.full(n_rho, ev)
                X_shares[:, rd] += np.bincount(win[s][:, rd], weights=ev,
                                               minlength=self.n_good)

        eps = self.epsilon
        mu = eps / (eps - 1)
        p_r = c_tilde / mu
        dl = self.delta[self.down]
        P = np.sum(p_r ** eps * dl) ** (1 / eps)
        Y = p_r ** eps * P ** (-eps) * dl
        X_ls_flat = X_shares @ (mu * Y)

        if same_alpha and not recomputed:
            linkages = base["linkages"]
        else:
            linkages = np.zeros((n_rho, self.n_good), bool)
            for s in win:
                linkages[np.repeat(rows, Rd), win[s].ravel()] = True

        return {"alpha": alpha.copy(), "tau": tau, "T_g": T_g, "win": win, "price": price,
                "logz_const": logz_const, "linkages": linkages, "c_tilde": c_tilde,
                "Y": Y, "X_ls_flat": X_ls_flat, "n_rho": n_rho, "w": w,
                "recomputed": recomputed}

    # -------------------------------------------------------- regression
    def _reg_cells(self, logz_const, n_rho):
        """`_build_reg_cells`: one entry per (sector, region) cell."""
        n_ctrl = self.ctrl_s.size if self.reg_control else 0
        cs = np.concatenate([self.good_s, self.ctrl_s[:n_ctrl]])
        cr = np.concatenate([self.good_r, self.ctrl_r[:n_ctrl]])
        dr = self.closest_dr[cr]
        if self.n_reg == 1:
            gcol = np.ones(cs.size, int)
            gval = self.log_closest[cr]
        else:
            b = self.dist_bin[cr, dr]
            ok = (b > 0) & (b <= self.n_reg)
            gcol = np.where(ok, b, 0)
            gval = ok.astype(float)
        _, grp = np.unique(cs * self.Rd + dr, return_inverse=True)
        is_good = np.arange(cs.size) < self.n_good
        has_z = is_good & self.reg_size
        lzc = np.zeros(cs.size)
        lzc[:self.n_good] = np.where(self.reg_size, logz_const, 0.0)
        return dict(gcol=gcol, gval=gval, grp=grp, n_grp=int(grp.max()) + 1,
                    is_good=is_good, has_z=has_z, lzc=lzc, n_cell=cs.size)

    def _cell_sums(self, C, linkages, lzr, first, Acell, b_size, w, chunk=None):
        """Per-cell sums over draws of W, W z, W lz, W lz^2, W lz z (cloglog IRLS)."""
        n_rho = linkages.shape[0]
        if _nb is not None and w is None and not os.environ.get("JACOBIAN_NO_NUMBA"):
            # `linkages` / `lzr` are the transposed (cell, draw) copies made once per
            # regression (see `regression`)
            lz_arr = lzr if lzr is not None else np.zeros((1, 1))
            return _cell_sums_nb(linkages, lz_arr, lzr is not None, C["is_good"],
                                 C["has_z"], C["lzc"], Acell, float(b_size), bool(first),
                                 1.0 / n_rho, ETA_CLAMP)
        n_cell = C["n_cell"]
        chunk = chunk or max(1, int(2_000_000 // max(n_rho, 1)))
        out = np.zeros((5, n_cell))
        e00 = np.clip(np.log(-np.log(1 - 0.25)), -ETA_CLAMP, ETA_CLAMP)
        e01 = np.clip(np.log(-np.log(1 - 0.75)), -ETA_CLAMP, ETA_CLAMP)
        for a in range(0, n_cell, chunk):
            cc = np.arange(a, min(a + chunk, n_cell))
            good = cc[C["is_good"][cc]]
            y = np.ones((n_rho, cc.size))
            if good.size:
                y[:, :good.size] = np.where(linkages[:, good], 0.0, 1.0)   # not_supply
            hz = C["has_z"][cc]
            lzr_c = np.zeros((n_rho, cc.size))
            if lzr is not None and good.size:
                lzr_c[:, :good.size] = lzr[:, good]
            lzr_c[:, ~hz] = 0.0
            lz = np.where(hz[None, :], C["lzc"][cc][None, :] + lzr_c, 0.0)
            if first:
                eta = np.where(y > 0.5, e01, e00)
            else:
                eta = np.clip(Acell[cc][None, :] + b_size * lzr_c, -ETA_CLAMP, ETA_CLAMP)
            ee = np.exp(eta)
            mu = np.clip(1.0 - np.exp(-ee), 1e-12, 1.0 - 1e-12)
            d = ee * (1.0 - mu)
            zw = eta + (y - mu) / d
            if w is None:
                wi = 1.0 / n_rho
            else:
                wi = np.full((n_rho, cc.size), 1.0 / n_rho)
                if good.size:
                    wi[:, :good.size] = np.asarray(w)[:, good]
            W = wi * (d * d) / (mu * (1.0 - mu))
            out[0, cc] = W.sum(0)
            out[1, cc] = (W * zw).sum(0)
            out[2, cc] = (W * lz).sum(0)
            out[3, cc] = (W * lz * lz).sum(0)
            out[4, cc] = (W * lz * zw).sum(0)
        return out

    def _normal_eq(self, C, sw, swz, swl, swll, swlz, k, size_col):
        """Assemble the within-group k x k normal equations from per-cell sums."""
        n_grp, grp, gcol, gval, hz = C["n_grp"], C["grp"], C["gcol"], C["gval"], C["has_z"]
        Axx = np.zeros((k, k))
        bxz = np.zeros(k)
        SX = np.zeros((k, n_grp))
        has = gcol > 0
        gi = gcol[has] - 1
        np.add.at(Axx, (gi, gi), sw[has] * gval[has] ** 2)
        np.add.at(bxz, gi, gval[has] * swz[has])
        np.add.at(SX, (gi, grp[has]), gval[has] * sw[has])
        if size_col is not None:
            both = has & hz
            np.add.at(Axx, (gcol[both] - 1, np.full(both.sum(), size_col)),
                      gval[both] * swl[both])
            Axx[size_col, size_col] += swll[hz].sum()
            bxz[size_col] += swlz[hz].sum()
            np.add.at(SX[size_col], grp[hz], swl[hz])
        SW = np.bincount(grp, weights=sw, minlength=n_grp)
        Sz = np.bincount(grp, weights=swz, minlength=n_grp)
        A = np.triu(Axx) + np.triu(Axx, 1).T
        rhs = bxz.copy()
        ok = SW >= 1e-300
        A -= (SX[:, ok] / SW[ok]) @ SX[:, ok].T
        rhs -= (SX[:, ok] / SW[ok]) @ Sz[ok]
        return A, rhs, SX, SW, Sz

    def regression(self, sol, u, max_iter=50, tol=1e-9, start=None):
        """
        Block 4: `fast_cloglog_regression` (streaming) or the LPM sibling.

        Returns {"coef", "b_logz", "beta", "a", "iters"}. `start` = an earlier fit's
        {"beta", "a"} to warm-start the IRLS from (a perturbed column on the same draws);
        Julia always starts cold. Both stop at the same tolerance on the step, so the two
        agree to far below the Jacobian's own finite-difference error.
        """
        n_rho = sol["n_rho"]
        C = self._reg_cells(sol["logz_const"], n_rho)
        k = self.n_reg + (1 if self.reg_size else 0)
        size_col = self.n_reg if self.reg_size else None
        lzr = (-(1.0 / self.theta) * np.log(-np.log(1.0 - np.asarray(u, float)))
               if self.reg_size else None)
        link, w = sol["linkages"], sol["w"]

        if self.reg_method == "lpm":
            co, bl = self._lpm(C, link, lzr, w, k, size_col)
            return {"coef": co, "b_logz": bl, "beta": None, "a": None, "iters": 1}

        use_nb = _nb is not None and w is None and not os.environ.get("JACOBIAN_NO_NUMBA")
        if use_nb:                      # one transposed copy per regression, not per pass
            link_k = np.ascontiguousarray(link.T)
            lzr_k = None if lzr is None else np.ascontiguousarray(lzr.T)
        else:
            link_k, lzr_k = link, lzr

        def _acell(beta, a):
            gc = C["gcol"]
            return (np.where(gc > 0, beta[np.maximum(gc, 1) - 1] * C["gval"], 0.0)
                    + (np.where(C["has_z"], beta[size_col] * C["lzc"], 0.0)
                       if size_col is not None else 0.0)
                    + a[C["grp"]])

        if start is not None and start.get("beta") is not None:
            beta = np.asarray(start["beta"], float).copy()
            Acell = _acell(beta, np.asarray(start["a"], float))
            first = False
        else:
            beta = np.zeros(k)
            Acell = np.zeros(C["n_cell"])
            first = True
        a = np.zeros(C["n_grp"])
        it = 0
        for it in range(1, max_iter + 1):
            b_size = beta[size_col] if size_col is not None else 0.0
            sums = self._cell_sums(C, link_k, lzr_k, first, Acell, b_size, w)
            A, rhs, SX, SW, Sz = self._normal_eq(C, *sums, k, size_col)
            beta_new = _solve_within(A, rhs, "cloglog IRLS")
            a = np.where(SW >= 1e-300, (Sz - SX.T @ beta_new) / np.where(SW > 0, SW, 1), 0.0)
            Acell = _acell(beta_new, a)
            delta = np.max(np.abs(beta_new - beta))
            beta, first = beta_new, False
            if delta < tol:
                break
        return {"coef": beta[:self.n_reg],
                "b_logz": beta[size_col] if size_col is not None else np.nan,
                "beta": beta, "a": a, "iters": it}

    def _lpm(self, C, link, lzr, w, k, size_col):
        n_rho = link.shape[0]
        y = np.zeros((n_rho, C["n_cell"]))
        y[:, :self.n_good] = link
        lz = np.zeros_like(y)
        if lzr is not None:
            lz[:, :self.n_good] = C["lzc"][:self.n_good][None, :] + lzr
        W = np.full_like(y, 1.0 / n_rho)
        if w is not None:
            W[:, :self.n_good] = w
        sums = [W.sum(0), (W * y).sum(0), (W * lz).sum(0), (W * lz * lz).sum(0),
                (W * lz * y).sum(0)]
        A, rhs, *_ = self._normal_eq(C, *sums, k, size_col)
        beta = _solve_within(A, rhs, "LPM weighted LS")
        return beta[:self.n_reg], (beta[size_col] if size_col is not None else np.nan)

    # ------------------------------------------------------ count moment
    def k_counts(self, sol):
        return sol["linkages"].sum(axis=0)

    def G0(self, k, m, N):
        lg = gbar_logfact(m)
        # an empty sector is left at 0.0, as compute_moments leaves it
        return np.array([gbar_cells(k[self.sector_goods[s]], m, int(N[s]), lg).mean()
                         if self.sector_goods[s].size else 0.0 for s in range(self.S)])

    def concentrate_N_s(self, k, m):
        """`concentrate_N_s`: the bisection the ESTIMATOR runs (only to cross-check N_hat)."""
        if self.G_target is None or self.N_LO is None:
            raise ValueError("concentrate_N_s needs G_target and N_supplier_s")
        lg = gbar_logfact(m)
        N = np.zeros(self.S, int)
        for s in range(self.S):
            cells = self.sector_goods[s]
            lo0, hi0 = int(self.N_LO[s]), int(self.N_HI[s])
            if cells.size == 0:
                N[s] = lo0
                continue
            G = lambda n: gbar_cells(k[cells], m, n, lg).mean()
            t = self.G_target[s]
            if G(lo0) <= t:
                N[s] = lo0
            elif G(hi0) >= t:
                N[s] = hi0
            else:
                lo, hi = lo0, hi0
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    if G(mid) > t:
                        lo = mid
                    else:
                        hi = mid
                N[s] = lo if abs(G(lo) - t) <= abs(G(hi) - t) else hi
        return N

    # ------------------------------------------------------ moments
    def moments(self, theta, u, N_fixed=None, w=None, base=None, reg_base=None,
                full=False, warm_start=True):
        """
        The masked moment vector `moments_to_vec(compute_moments(...))`.

        `N_fixed` (granular): the theta+ variety counts. Required -- the bisection is
        available separately (`concentrate_N_s`) but never runs inside an evaluation.
        `reg_base` is an earlier (solution, regression) on the same draws: reused when the
        Ricardian block did not move (a head-parameter column), since block 4 depends on
        the winners and log z alone.
        """
        sol = self.solve(theta, u, w=w, base=base)
        Om_L = self.unpack(theta)[0]
        S, R, eps = self.S, self.R, self.epsilon
        act = self.active
        ct = np.zeros(R); ct[self.down] = sol["c_tilde"]
        Yf = np.zeros(R); Yf[self.down] = sol["Y"]

        markup = (eps - 1) / eps
        dl = self.delta
        P_idx = np.sum((ct[act] * markup) ** eps * dl[act]) ** (1 / eps)
        B = (markup / P_idx) ** (eps - 1) / P_idx
        y_r = np.zeros(R); y_r[act] = ct[act] ** (eps - 1) * dl[act] * B
        lab = np.zeros(R)
        lab[act] = Om_L * y_r[act] * (self.rw[act] / ct[act]) ** (-self.lam)
        labor = np.sum(self.rw * lab) / np.sum(ct * y_r)

        X_ls = np.zeros((R, S))
        X_ls[self.good_r, self.good_s] = sol["X_ls_flat"]
        X_s = X_ls.sum(axis=0)
        industry = X_s / X_s.sum()
        with np.errstate(invalid="ignore", divide="ignore"):
            gam = X_ls / X_s[None, :] * self.domestic_share[None, :]
        if self.ca_level == "aa":
            gout = np.zeros((self.T_dim, S))
            np.add.at(gout, (self.T_gather[self.good_r], self.good_s),
                      gam[self.good_r, self.good_s])
        else:
            gout = gam
        gamma_vec = gout.T.ravel()                          # Julia vec(.) of (T_dim, S)

        if reg_base is not None and not sol["recomputed"] \
                and np.array_equal(reg_base[0]["alpha"], sol["alpha"]):
            fit = reg_base[1]                    # winners and log z unchanged: same fit
        else:
            fit = self.regression(sol, u, start=(reg_base[1] if (reg_base is not None
                                                                 and warm_start) else None))
        reg = fit["coef"]

        pi_r = Yf[act] / Yf[act].sum()
        blocks = [[labor], industry, pi_r, reg, gamma_vec]
        k = None
        if self.granular:
            if N_fixed is None:
                raise ValueError("granular moments need N_fixed (theta+'s N_hat); the "
                                 "bisection is not re-run inside an evaluation")
            k = self.k_counts(sol)
            blocks.append(self.G0(k, sol["n_rho"], N_fixed))
        vec = np.concatenate([np.ravel(b) for b in blocks])[self.moment_mask]
        if full:
            return vec, {"sol": sol, "reg": fit, "k_counts": k,
                         "blocks": blocks, "gamma_T": gout}
        return vec


# ============================================================================
# Finite-difference Jacobian -- `compute_jacobian`, simulation path
# ============================================================================

_WORKER = {}


def _init_worker(model, theta, idx, step, N_fixed, append_N_s, warm_start=True,
                 n_threads=None):
    _WORKER.update(model=model, theta=theta, idx=idx, step=step, N_fixed=N_fixed,
                   append_N_s=append_N_s, warm_start=warm_start)
    if _nb is not None and n_threads:
        _nb.set_num_threads(max(1, int(n_threads)))


def jacobian_replication(model, theta, u, idx, step, N_fixed, w=None, append_N_s=True,
                         warm_start=True, cols=None):
    """
    One replication on one draw set: the central log-step difference for every column of
    `idx`, all on the SAME draws (common random numbers), N held at `N_fixed`.

    `cols` (positions into `idx`) restricts the work to a CHUNK of columns, so one
    replication can be spread over several workers; the other columns come back as zeros
    and the caller stitches the chunks together. Every chunk recomputes the same base
    evaluation on the same draws, so a chunked replication is bit-identical to an
    unchunked one.

    Returns (J_k raw, J_elast_k, m0_k, dN_k) where dN_k is the N_s first difference
    m(N+1) - m(N) on these draws (S columns; only block 6 can move).
    """
    theta = np.asarray(theta, float)
    m0, info = model.moments(theta, u, N_fixed=N_fixed, w=w, full=True)
    base, reg_base = info["sol"], (info["sol"], info["reg"])
    J = np.zeros((m0.size, len(idx)))
    todo = range(len(idx)) if cols is None else cols
    for j in todo:
        i = idx[j]
        tp, tm = theta.copy(), theta.copy()
        tp[i] = theta[i] * np.exp(step)
        tm[i] = theta[i] * np.exp(-step)
        mp = model.moments(tp, u, N_fixed=N_fixed, w=w, base=base, reg_base=reg_base,
                           warm_start=warm_start)
        mm = model.moments(tm, u, N_fixed=N_fixed, w=w, base=base, reg_base=reg_base,
                           warm_start=warm_start)
        J[:, j] = (mp - mm) / (2 * step * theta[i])
    th = theta[np.asarray(idx)]
    with np.errstate(divide="ignore", invalid="ignore"):
        E = np.where((np.abs(m0)[:, None] > 1e-12) & (np.abs(th)[None, :] > 1e-12),
                     th[None, :] / m0[:, None] * J, 0.0)
    dN = None
    if append_N_s and model.granular:
        k, n = info["k_counts"], base["n_rho"]
        N0 = np.asarray(N_fixed, int)
        rows = model.block_ranges[5]
        dN = np.zeros((m0.size, model.S))
        diff = model.G0(k, n, N0 + 1) - model.G0(k, n, N0)
        dN[rows, np.arange(model.S)] = diff
    return J, E, m0, dN


def _run_rep(args):
    seed, n_rho, u, cols = args
    W = _WORKER
    m = W["model"]
    if u is None:
        u = np.random.default_rng(seed).random((n_rho, m.n_good))
    return jacobian_replication(m, W["theta"], u, W["idx"], W["step"], W["N_fixed"],
                                append_N_s=W["append_N_s"], warm_start=W["warm_start"],
                                cols=cols)


def _plan_chunks(n_cols, n_reps, n_cpu, min_cols=10):
    """
    How many column chunks per replication. One when there are at least as many
    replications as cores (processes across replications already fill the machine);
    otherwise enough to give every core a job, but never chunks smaller than `min_cols`
    columns, since each chunk pays for one base evaluation of its own.
    """
    if n_reps >= n_cpu:
        return 1
    return int(max(1, min(-(-n_cpu // n_reps), n_cols // min_cols)))


def python_jacobian(model, theta, N_fixed, K=50, n_rho=10_000, step=1e-2,
                    param_indices=None, base_seed=5_000_000, u_list=None,
                    append_N_s=True, n_workers=None, n_chunks=None, warm_start=True,
                    verbose=True):
    """
    The averaged Jacobian over K independent draw sets, in `compute_jacobian`'s layout:
    rows = masked moments, columns = `param_indices` (default: Julia's
    `jacobian_param_indices`) then, under GRANULAR, the S variety counts.

    `u_list` replaces the K Python draw sets by given matrices (e.g. Julia's own), which
    turns the comparison into an exact one. Returns a dict with the mean, the
    across-replication SD (ddof = 1, as Julia's `std`), the full replication STACK
    (which gives the exact max-|t| distribution) and the per-column replication count.

    Parallelism, two levels:
      * PROCESSES over (replication, column chunk) jobs -- `n_workers`, default all
        cores. Replications are independent, and so are columns given the base
        evaluation, so this scales almost linearly up to K x n_chunks jobs. More workers
        than cores only adds contention.
      * NUMBA THREADS inside each process, over cells in the regression kernel -- given
        the cores left over (cores // workers). Only the regression is threaded; the
        Ricardian solve and the value block are single-threaded numpy, so a core spent
        on a process is worth more than a core spent on a thread.
    `n_chunks` (default: `_plan_chunks`) splits each replication's columns across jobs
    when there are more cores than replications. Each chunk repeats the base evaluation
    (~10 s at production size, against ~4 s per column), so the default keeps chunks at
    10 columns or more. Chunking is bit-identical to running unchunked.

    Memory: each worker holds a draw set and its transformed copies, about 4 x n_rho x
    n_good x 8 bytes (~0.7 GB at 10 000 draws and 2 250 cells). That, not the core count,
    is what bounds `n_workers` on a shared machine.
    """
    idx = model.param_indices if param_indices is None else np.asarray(param_indices)
    if u_list is not None:
        reps = [(None, None, np.asarray(x, float)) for x in u_list]
    else:
        reps = [(base_seed + k, n_rho, None) for k in range(1, K + 1)]
    n_cpu = os.cpu_count() or 1
    if n_chunks is None:
        n_chunks = _plan_chunks(len(idx), len(reps), n_workers or n_cpu)
    n_chunks = int(max(1, min(n_chunks, len(idx))))
    chunks = [c for c in np.array_split(np.arange(len(idx)), n_chunks) if c.size]
    jobs = [(*r, c) for r in reps for c in chunks]
    n_workers = n_workers or min(len(jobs), n_cpu)
    args = (model, np.asarray(theta, float), idx, step, N_fixed, append_N_s, warm_start,
            max(1, n_cpu // n_workers))
    if n_workers > 1:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")      # numba's OpenMP pool is not fork-safe
        with ProcessPoolExecutor(n_workers, mp_context=ctx, initializer=_init_worker,
                                 initargs=args) as ex:
            parts = list(ex.map(_run_rep, jobs))
    else:
        _init_worker(*args)
        parts = [_run_rep(j) for j in jobs]

    # stitch the chunks of each replication back together (each chunk carries zeros
    # outside its own columns, and the same m0 and dN)
    res = []
    for r in range(len(reps)):
        group = parts[r * len(chunks):(r + 1) * len(chunks)]
        J = sum(g[0] for g in group)
        E = sum(g[1] for g in group)
        res.append((J, E, group[0][2], group[0][3]))

    Jst = np.stack([r[0] for r in res], axis=2)
    Est = np.stack([r[1] for r in res], axis=2)
    m0 = np.stack([r[2] for r in res], axis=1)
    Kk = Jst.shape[2]
    out = {"J": Jst.mean(2), "J_elast": Est.mean(2),
           "J_sd": Jst.std(2, ddof=1) if Kk > 1 else np.zeros(Jst.shape[:2]),
           "J_elast_sd": Est.std(2, ddof=1) if Kk > 1 else np.zeros(Est.shape[:2]),
           "J_stack": Jst, "m0_stack": m0, "param_indices": idx,
           "n_reps": np.full(len(idx), Kk, float), "N_fixed": np.asarray(N_fixed)}
    if append_N_s and model.granular:
        Nst = np.stack([r[3] for r in res], axis=2)
        m0b = m0.mean(1)
        G = Nst.mean(2)
        Gsd = Nst.std(2, ddof=1) if Kk > 1 else np.zeros(G.shape)
        Nf = np.asarray(N_fixed, float)
        with np.errstate(divide="ignore", invalid="ignore"):
            sc = np.where(np.abs(m0b)[:, None] > 1e-12, Nf[None, :] / np.abs(m0b)[:, None], 0.0)
        EN = np.where(np.abs(m0b)[:, None] > 1e-12, Nf[None, :] / np.where(m0b == 0, 1, m0b)[:, None], 0.0) * G
        out["J"] = np.hstack([out["J"], G])
        out["J_sd"] = np.hstack([out["J_sd"], Gsd])
        out["J_elast"] = np.hstack([out["J_elast"], EN])
        out["J_elast_sd"] = np.hstack([out["J_elast_sd"], sc * Gsd])
        out["J_stack"] = np.concatenate([Jst, Nst], axis=1)
        out["n_reps"] = np.r_[out["n_reps"], np.full(model.S, Kk, float)]
        out["n_N_cols"] = model.S
    else:
        out["n_N_cols"] = 0
    if verbose:
        print(f"[python jacobian] {out['J'].shape[0]} moments x {out['J'].shape[1]} "
              f"columns, K = {Kk} draw sets of {reps[0][1] or reps[0][2].shape[0]} draws, "
              f"N held at {list(np.asarray(N_fixed))}; {len(jobs)} jobs "
              f"({len(chunks)} column chunk(s) per replication) on {n_workers} workers")
    return out


def theta_plus(data):
    """The raw theta and N_hat of theta+, as Julia's Jacobian uses them at this `mu`."""
    th = np.asarray(data["best_params"], float).ravel()
    N = data.get("post_hoc_N_hat")
    if N is None:
        from utils import _n_hat_from_diagnostics
        N = _n_hat_from_diagnostics(data)
    return th, None if N is None else np.asarray(N, int).ravel()


# ============================================================================
# Cross-language comparison
# ============================================================================

def compare_same_draws(model, theta, N_fixed, dump, step=1e-2, rtol_m=1e-10,
                       rtol_reg=1e-7, jtol=1e-6, irls_tol=1e-9, warm_start=False,
                       verbose=True):
    """
    LEVEL 1. `dump` = {"u": (n_rho, n_good), "m0": m0_k, "J": J_k raw} from ONE Julia
    replication. On identical draws there is no Monte-Carlo noise, so any gap is a
    porting error; the report names the block it sits in.
    """
    u = np.asarray(dump["u"], float)
    m_py, info = model.moments(theta, u, N_fixed=N_fixed, full=True)
    m_jl = np.asarray(dump["m0"], float).ravel()
    names = ["labor", "industry", "pi_r", "reg_coef", "gamma", "G0"][:len(model.block_ranges)]
    rep = {"moments": {}, "ok": True}
    for b, (name, rows) in enumerate(zip(names, model.block_ranges)):
        if rows.size == 0:
            continue
        d = np.abs(m_py[rows] - m_jl[rows]) / np.maximum(np.abs(m_jl[rows]), 1e-300)
        tol = rtol_reg if name == "reg_coef" else rtol_m
        ok = bool(np.nanmax(d) <= tol)
        rep["moments"][name] = (float(np.nanmax(d)), ok)
        rep["ok"] &= ok
    if "J" in dump:
        J_jl = np.asarray(dump["J"], float)
        J_py, _, _, _ = jacobian_replication(model, theta, u, model.param_indices, step,
                                             N_fixed, append_N_s=False,
                                             warm_start=warm_start)
        J_jl = J_jl[:, :J_py.shape[1]]
        scale = np.zeros_like(J_jl)
        for rows in model.block_ranges:
            if rows.size:
                scale[rows] = np.median(np.abs(J_jl[rows])) or 1.0
        # the cloglog rows pass through an IRLS stopped at a 1e-9 step: two correct
        # implementations summing in different orders can stop one step apart, which
        # shows up in a finite difference as ~tol/(2 step theta_i). Allow 10x that there.
        allow = np.zeros_like(J_jl)
        th_i = np.asarray(theta, float)[model.param_indices]
        allow[model.block_ranges[3]] = 10 * irls_tol / (2 * step * th_i)[None, :]
        bad = np.abs(J_py - J_jl) > jtol * (np.abs(J_jl) + scale) + allow
        zero_mismatch = int(((J_py == 0) != (J_jl == 0)).sum())
        rep["J_bad"] = int(bad.sum())
        rep["zero_pattern_mismatch"] = zero_mismatch
        rep["ok"] &= rep["J_bad"] == 0 and zero_mismatch == 0
        rep["J_bad_where"] = [(names[next(b for b, r in enumerate(model.block_ranges)
                                          if i in r)], int(j))
                              for i, j in zip(*np.nonzero(bad))][:10]
    if verbose:
        print("[level 1, same draws] " + "  ".join(
            f"{k} {v[0]:.1e}{'' if v[1] else ' FAIL'}" for k, v in rep["moments"].items())
            + (f"  | J: {rep.get('J_bad', 'n/a')} entries off, "
               f"{rep.get('zero_pattern_mismatch', 'n/a')} zero-pattern mismatches"
               if "J" in dump else "")
            + ("  -> PASS" if rep["ok"] else "  -> FAIL"))
    return rep


def compare_independent(a, b, block_ranges=None, alpha=0.05, rel=0.10, abs_tol=1e-3,
                        verbose=True):
    """
    LEVEL 2. Two averaged Jacobians on independent draws, each with its SD and
    replication count (dicts with J, J_sd, n_reps; optionally J_stack, J_elast).

      * outlier test   max |z| under Bonferroni, z = Welch statistic per entry;
      * bias test      per (moment block x parameter column), the mean of z;
      * equivalence    TOST on the ELASTICITY for entries significant in either:
                       |d eps| < max(abs_tol, rel |eps_b|) at level alpha.
    Structural zeros (exactly zero, zero SD) must coincide exactly.
    """
    from scipy import stats

    Ja, Jb = np.asarray(a["J"], float), np.asarray(b["J"], float)
    if Ja.shape != Jb.shape:
        raise ValueError(f"Jacobians differ in shape: {Ja.shape} vs {Jb.shape}")
    sa, sb = np.asarray(a["J_sd"], float), np.asarray(b["J_sd"], float)
    Ka = np.broadcast_to(np.asarray(a["n_reps"], float)[None, :], Ja.shape)
    Kb = np.broadcast_to(np.asarray(b["n_reps"], float)[None, :], Jb.shape)
    za_ = (Ja == 0) & (sa == 0)
    zb_ = (Jb == 0) & (sb == 0)
    zero_mismatch = int((za_ != zb_).sum())
    tested = ~(za_ & zb_)
    va, vb = sa ** 2 / Ka, sb ** 2 / Kb
    se = np.sqrt(va + vb)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (Ja - Jb) / se
    z[tested & (se == 0)] = np.where(Ja == Jb, 0.0, np.inf)[tested & (se == 0)]
    # Welch-Satterthwaite degrees of freedom
    with np.errstate(divide="ignore", invalid="ignore"):
        df = (va + vb) ** 2 / (va ** 2 / np.maximum(Ka - 1, 1) + vb ** 2 / np.maximum(Kb - 1, 1))
    df = np.where(np.isfinite(df), df, 1e6)
    m = int(tested.sum())
    crit = stats.t.isf(alpha / (2 * m), df)
    outliers = tested & (np.abs(z) > crit)

    # block bias: is the AVERAGE standardised gap over a (moment block, column) group
    # zero? Entries of one group share draws, so their z are correlated; with both
    # replication stacks the variance of the group mean is computed exactly from them,
    # otherwise it falls back to treating the entries as independent (flagged).
    bias = []
    exact = "J_stack" in a and "J_stack" in b
    if block_ranges is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(se > 0, 1.0 / se, 0.0)
        for bi, rows in enumerate(block_ranges):
            for j in range(Ja.shape[1]):
                keep = rows[tested[rows, j] & (se[rows, j] > 0)]
                if keep.size < 5:
                    continue
                # signed by the direction of the POOLED estimate, so a scale error on a
                # column whose entries have mixed signs does not cancel; under H0 and
                # equal variances (Ja - Jb) is uncorrelated with (Ja + Jb)
                sg = np.sign(Ja[keep, j] + Jb[keep, j]) * scale[keep, j]
                if exact and j < min(np.shape(a["J_stack"])[1], np.shape(b["J_stack"])[1]):
                    ga = (np.asarray(a["J_stack"])[keep, j, :] * sg[:, None]).mean(0)
                    gb = (np.asarray(b["J_stack"])[keep, j, :] * sg[:, None]).mean(0)
                    d = ga.mean() - gb.mean()
                    v = ga.var(ddof=1) / ga.size + gb.var(ddof=1) / gb.size
                else:
                    zz = z[keep, j] * np.sign(Ja[keep, j] + Jb[keep, j])
                    d, v = zz.mean(), 1.0 / keep.size
                t = d / np.sqrt(v) if v > 0 else 0.0
                bias.append((bi, j, keep.size, float(d), float(t)))
    n_bias_tests = max(len(bias), 1)
    bias_crit = stats.norm.isf(alpha / (2 * n_bias_tests))
    biased = [x for x in bias if abs(x[4]) > bias_crit]

    # equivalence on elasticities (TOST), where the elasticity is measured
    eq_fail = 0
    if "J_elast" in a and "J_elast" in b:
        Ea, Eb = np.asarray(a["J_elast"], float), np.asarray(b["J_elast"], float)
        with np.errstate(divide="ignore", invalid="ignore"):   # theta/m, per entry
            ratio = np.where(Jb != 0, np.abs(Eb / Jb),
                             np.where(Ja != 0, np.abs(Ea / Ja), 0.0))
        ratio = np.where(np.isfinite(ratio), ratio, 0.0)
        se_e = se * ratio
        sig = tested & ((np.abs(Ja) / np.maximum(np.sqrt(va), 1e-300) > 2)
                        | (np.abs(Jb) / np.maximum(np.sqrt(vb), 1e-300) > 2))
        margin = np.maximum(abs_tol, rel * np.abs(Eb))
        dE = Ea - Eb
        # Family-level FAILURE: an entry PROVEN to differ by more than the margin, at
        # Bonferroni level over the significant entries. (Requiring every entry to be
        # proven equivalent would fail by construction: each TOST with a true gap of
        # zero still misses up to 10% of the time.)
        n_sig = max(int(sig.sum()), 1)
        q_m = stats.norm.isf(alpha / n_sig)
        proven_off = sig & (np.abs(dE) - q_m * se_e > margin)
        eq_fail = int(proven_off.sum())
        # Reported beside it: the share POSITIVELY shown equivalent (TOST at alpha), among
        # the entries where the test can decide at all (interval narrower than margin).
        q = stats.norm.isf(alpha)
        decidable = sig & (2 * q * se_e < margin)
        eq = (dE + q * se_e < margin) & (dE - q * se_e > -margin)
        n_eq = int(decidable.sum())
        n_proven_eq = int((decidable & eq).sum())
        n_undecided = int((sig & ~decidable).sum())
    else:
        n_eq = n_undecided = n_proven_eq = 0

    # power: the smallest relative gap |Ja - Jb|/|J| the outlier test detects with 80%
    # probability, per significant entry -- a PASS is only as strong as this number
    with np.errstate(divide="ignore", invalid="ignore"):
        mde = (np.nanmedian(crit[tested]) + stats.norm.isf(0.2)) * se / np.abs(Jb)
    sig_any = tested & ((np.abs(Ja) / np.maximum(np.sqrt(va), 1e-300) > 2)
                        | (np.abs(Jb) / np.maximum(np.sqrt(vb), 1e-300) > 2))
    mde_med = float(np.nanmedian(mde[sig_any])) if sig_any.any() else np.nan
    mde_q10 = float(np.nanquantile(mde[sig_any], 0.1)) if sig_any.any() else np.nan

    rep = {"m": m, "zero_mismatch": zero_mismatch, "outliers": int(outliers.sum()),
           "max_abs_z": float(np.nanmax(np.abs(z[tested]))) if m else 0.0,
           "biased_groups": biased, "n_bias_groups": len(bias), "bias_exact": exact,
           "equivalence_fail": eq_fail, "n_equivalence": n_eq,
           "n_proven_equivalent": n_proven_eq, "n_undecided": n_undecided,
           "mde_median": mde_med, "mde_q10": mde_q10, "z": z}
    rep["ok"] = (zero_mismatch == 0 and rep["outliers"] == 0 and not biased
                 and eq_fail == 0)
    if verbose:
        print(f"[level 2, independent draws] {m} tested entries: max|z| "
              f"{rep['max_abs_z']:.2f} ({rep['outliers']} beyond Bonferroni); "
              f"{len(biased)}/{len(bias)} biased (block, column) groups; "
              f"{eq_fail} entries proven off by more than max({abs_tol:g}, {rel:.0%} eps); "
              f"{n_proven_eq}/{n_eq} decidable entries proven equivalent "
              f"({n_undecided} undecided); {zero_mismatch} structural-zero mismatches"
              + ("  -> PASS" if rep["ok"] else "  -> FAIL"))
        print(f"    power: a relative gap of {mde_med:.1%} (median entry; {mde_q10:.1%} for "
              "the best-measured decile) is detected with 80% probability -- smaller "
              "disagreements cannot be excluded by this comparison")
    return rep


def inference_slices(model):
    """Rows (beta -> gamma -> G) and columns (alpha + T, in Jacobian position) that
    Julia's inference uses -- `inference_moment_indices()` and `gb_cols`."""
    rows = np.concatenate([model.block_ranges[b] for b in range(3, len(model.block_ranges))])
    cols = np.flatnonzero(model.param_indices >= model.alpha_start)
    return rows, cols


def sandwich_se(J, W, Omega, rows, cols):
    """SMM sandwich standard errors on the (rows, cols) sub-Jacobian."""
    G = np.asarray(J, float)[np.ix_(rows, cols)]
    H = G.T @ W @ G
    Hi = np.linalg.pinv(H)
    V = Hi @ (G.T @ W @ Omega @ W @ G) @ Hi
    return np.sqrt(np.maximum(np.diag(V), 0.0))


def compare_standard_errors(a, b, W, Omega, rows, cols, n_sigma=3.0, floor=0.05,
                            verbose=True):
    """
    LEVEL 3. The use of the Jacobian: sandwich SEs from both, same W and Omega. The
    Monte-Carlo SE of log(se_a/se_b) is a delete-one jackknife over each side's STACK
    (when given). Pass if |log ratio| <= max(floor, n_sigma * MC se).
    """
    se_a = sandwich_se(a["J"], W, Omega, rows, cols)
    se_b = sandwich_se(b["J"], W, Omega, rows, cols)
    lr = np.log(se_a / se_b)

    def _jack(d):
        st = d.get("J_stack")
        if st is None or st.shape[2] < 3:
            return np.zeros(len(cols))
        K = st.shape[2]
        tot = st.sum(2)
        vals = np.array([np.log(sandwich_se((tot - st[:, :, k]) / (K - 1), W, Omega,
                                            rows, cols)) for k in range(K)])
        return np.sqrt((K - 1) / K * ((vals - vals.mean(0)) ** 2).sum(0))

    mc = np.sqrt(_jack(a) ** 2 + _jack(b) ** 2)
    tol = np.maximum(floor, n_sigma * mc)
    bad = np.abs(lr) > tol
    rep = {"se_a": se_a, "se_b": se_b, "log_ratio": lr, "mc_se": mc,
           "n_bad": int(bad.sum()), "ok": not bad.any()}
    if verbose:
        print(f"[level 3, standard errors] {len(cols)} parameters: median se ratio "
              f"{np.exp(np.median(lr)):.3f}, max |log ratio| {np.abs(lr).max():.3f}, "
              f"{rep['n_bad']} outside max({floor:g}, {n_sigma:g} MC se)"
              + ("  -> PASS" if rep["ok"] else "  -> FAIL"))
    return rep
