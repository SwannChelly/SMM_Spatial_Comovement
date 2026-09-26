"""
The fit of the structural model to its targets -- the library behind `model_report.ipynb`.

Moment tables, the sourcing-share and downstream-sales fit, the extensive margin and the
zero-supplier share with their standard errors, the count curve, the Jacobian and what it
says about identification, the moment covariance matrices, and the untargeted distance
elasticity of spatial comovement.
"""

import os
import re
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import statsmodels.api as sm
import pyfixest as pf          # PPML with absorbed fixed effects (ppmlhdfe)

from utils import (
    EMPIRICAL_MEAN_LOG_D, NU_S_DEFAULT, THETA_DEFAULT,
    _despine, _read_named_value, fs, get_figsize, load_granular_data, pct,
    reference_color, set_name_axis_fontsize, sim_color, toulouse_color,
    unpack_estimated_T,
)


# ============================================================================
# Moment table
# ============================================================================

def generate_combined_table(industries_config, output_file="moments_comparison_combined.tex",
                            mu=2, name_A129_path="../external/A129_name_fr_eng.csv", **load_kwargs):
    """
    LaTeX table with one {Emp., Sim.} column pair per industry.

    Panel A: aggregate labor share.  Panel B: aggregate industry shares.

    industries_config : [{'industry': 'aero', 'display_name': 'Aerospace'}, ...]
    mu                : 1 (step1 best parameters) or 2 (step3 best parameters)
    """
    all_data = {c["industry"]: load_granular_data(c["industry"], mu=mu, **load_kwargs)
                for c in industries_config}

    try:
        name_A129 = pd.read_csv(name_A129_path)
    except Exception:
        name_A129 = None

    n_ind = len(industries_config)
    col_spec = "l " + " ".join(["S[table-format=1.4] S[table-format=1.4]"] * n_ind)
    header_row1 = " & " + " & ".join(
        [f"\\multicolumn{{2}}{{c}}{{{c['display_name']}}}" for c in industries_config]) + r" \\"
    cmidrules = " ".join([f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(n_ind)])
    header_row2 = " & " + " & ".join(["{Emp.} & {Sim.}"] * n_ind) + r" \\"

    # --- Panel A -------------------------------------------------------------
    row_A = ["Aggregate Labor Share"]
    for c in industries_config:
        d = all_data[c["industry"]]
        row_A += [float(d["empirical_moments_dict"]["agg_labor_share"][0]),
                  float(d["simulated_moments_dict"]["agg_labor_share"][0])]

    # --- Panel B -------------------------------------------------------------
    all_codes = sorted({code for d in all_data.values() for code in d["sector_names"]})
    shares = {}
    for c in industries_config:
        d = all_data[c["industry"]]
        shares[c["industry"]] = pd.DataFrame({
            "A129": d["sector_names"],
            "Empirical": d["empirical_moments_dict"]["agg_industry_share"],
            "Simulated": d["simulated_moments_dict"]["agg_industry_share"],
        })

    panel_B_rows = []
    for code in all_codes:
        label = code
        if name_A129 is not None and "A129" in name_A129.columns:
            hit = name_A129.loc[name_A129["A129"].astype(str) == str(code), "name"].values
            if len(hit):
                label = hit[0]
        row = [label]
        for c in industries_config:
            df = shares[c["industry"]]
            m = df[df["A129"].astype(str) == str(code)]
            row += ([float(m.iloc[0]["Empirical"]), float(m.iloc[0]["Simulated"])]
                    if len(m) else ["---", "---"])
        panel_B_rows.append(row)

    def fmt(v, decimals=4):
        if isinstance(v, str) or (isinstance(v, float) and np.isnan(v)):
            return "{---}"
        return f"{v:.{decimals}f}"

    mu_label = r"$\hat{\mu}_1$ (Step~1)" if mu == 1 else r"$\hat{\mu}_2$ (Step~3)"
    tex = (r"\begin{table}[H]" "\n" r"\centering" "\n"
           r"\caption{Empirical and Simulated Moments: Comparison Across Industries}" "\n"
           r"\label{tab:moments_comparison_combined}" "\n"
           r"\renewcommand{\arraystretch}{1.2}" "\n" r"\small" "\n"
           r"\begin{tabular}{" + col_spec + "}\n" r"\toprule" "\n"
           + header_row1 + "\n" + cmidrules + "\n" + header_row2 + "\n")

    tex += "\\midrule\n"
    tex += f"\\multicolumn{{{1 + 2 * n_ind}}}{{l}}{{\\textbf{{Panel A: Aggregate Labor Share}}}} \\\\\n"
    tex += "\\midrule\n"
    tex += row_A[0] + " & " + " & ".join(fmt(v) for v in row_A[1:]) + " \\\\\n"

    tex += "\\midrule\n"
    tex += f"\\multicolumn{{{1 + 2 * n_ind}}}{{l}}{{\\textbf{{Panel B: Aggregate Industry Shares}}}} \\\\\n"
    tex += "\\midrule\n"
    for row in panel_B_rows:
        tex += str(row[0]) + " & " + " & ".join(fmt(v) for v in row[1:]) + " \\\\\n"

    tex += "\\bottomrule\n" + r"\end{tabular}" + "\n\n"
    tex += (r"\vspace{0.3cm}" "\n"
            r"\caption*{\footnotesize \emph{Notes}: Empirical moments against moments simulated "
            r"from the model calibrated on each industry, evaluated at " + mu_label + r". "
            r"``Emp.'' = Empirical, ``Sim.'' = Simulated. Panel~A reports the aggregate labor "
            r"share, Panel~B the sectoral industry shares. ``---'' indicates the sector is not "
            r"present in that industry's sample. The distance-bin (cloglog) coefficients and the "
            r"zero-supplier shares $\bar{G}_s(0)$ are reported in separate figures, with standard "
            r"errors.}" "\n" r"\end{table}" "\n")

    if output_file is not None:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w") as f:
            f.write(tex)
        print(f"Combined table saved to: {output_file}")
    return tex


# ============================================================================
# Fit: sourcing shares and downstream sales
# ============================================================================

def bubble_scatter(ax, x, y, xlabel, ylabel, title, size_scale=300,
                   regression_line=False, weights=None, color=None, label=None):
    """Bubble scatter of simulated against empirical, with a WLS-through-origin fit."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = x > 0
    x, y = x[keep], y[keep]
    w = np.ones_like(x) if weights is None else np.asarray(weights, float)[keep]

    lims = [min(x.min(), y.min()) * 0.9, max(x.max(), y.max()) * 1.1]
    ax.scatter(x, y, s=size_scale * x / x.max(), alpha=1, edgecolor="black",
               linewidths=0.5, color=color or toulouse_color, label=label)

    b, t = wls_through_origin(x, y, w)
    ax.text(0.98, 0.09, rf"Coefficient: ${np.round(b, 3)}$", ha="right", va="bottom",
            fontsize=fs(10), transform=ax.transAxes)
    ax.text(0.98, 0.01, rf"t-stat: ${np.round(t, 1)}$", ha="right", va="bottom",
            fontsize=fs(10), transform=ax.transAxes)

    if regression_line:
        xx = np.linspace(0, lims[1], 100)
        ax.plot(xx, b * xx, linestyle="--", color="green")
    else:
        ax.plot(lims, lims, color="black")

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(linestyle="dashed", alpha=0.5)
    _despine(ax)
    return b, t


def wls_through_origin(x, y, weights=None):
    """b and its t-stat for y = b*x (no intercept); weights default to x."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    w = x if weights is None else np.asarray(weights, float)
    denom = np.sum(w * x ** 2)
    b = np.sum(w * x * y) / denom
    resid = y - b * x
    s2 = np.sum(w * resid ** 2) / max(len(x) - 1, 1)
    se = np.sqrt(s2 / denom)
    return b, (b / se if se > 0 else np.inf)


def gamma_aa_points(data):
    """
    Split the AA-level gamma moments into the two series the dashboard panel plots.

    Returns (x_free, y_free, x_ref, y_ref):
      * free      — active, non-reference (sector, AA) pairs, i.e. the moments that
                    actually enter the criterion;
      * reference — each sector's dropped reference area, RECONSTRUCTED from the
                    within-sector adding-up constraint
                        gamma_ref,s = domestic_share_s - sum_{a != ref} gamma_{s,a},
                    applied to the SIMULATED gammas; its x is the observed share.
    """
    emp, sim = data["empirical_moments_dict"]["gamma_aa"], data["simulated_moments_dict"]["gamma_aa"]
    S, ref, act, c = data["S"], data["T_REF_AA"], data["AA_ACTIVE"], data["domestic_share"]

    free = act.copy()
    for s in range(S):
        if ref[s] >= 0:
            free[s, ref[s]] = False

    x_free, y_free = emp[free], sim[free]
    x_ref, y_ref = [], []
    for s in range(S):
        if ref[s] < 0 or not free[s].any() or emp[s, ref[s]] <= 0:
            continue
        x_ref.append(emp[s, ref[s]])
        y_ref.append(c[s] - sim[s, free[s]].sum())
    keep = x_free > 0
    return x_free[keep], y_free[keep], np.array(x_ref), np.array(y_ref)


def plot_gamma_aa(data, ax=None, save_to=None):
    """Empirical vs simulated gamma at the attraction-area level (dashboard panel 1)."""
    x_free, y_free, x_ref, y_ref = gamma_aa_points(data)
    fig, ax = (plt.subplots(figsize=get_figsize()) if ax is None else (ax.figure, ax))

    all_x = np.concatenate([x_free, x_ref]) if len(x_ref) else x_free
    all_y = np.concatenate([y_free, y_ref]) if len(y_ref) else y_free
    lims = [min(all_x.min(), all_y.min()) * 0.9, max(all_x.max(), all_y.max()) * 1.1]

    ax.scatter(x_free, y_free, s=300 * x_free / x_free.max(), alpha=0.6, edgecolor="black",
               linewidths=0.5, color=sim_color, label="Non-reference")
    if len(x_ref):
        ax.scatter(x_ref, y_ref, s=60, marker="D", alpha=0.8, edgecolor="black",
                   linewidths=0.5, color=reference_color, label="Reference")

    # WLS through the origin on the NON-REFERENCE points only.
    b, t = wls_through_origin(x_free, y_free, weights=x_free)
    ax.plot(lims, lims, color="black", linewidth=1)
    ax.text(0.98, 0.09, rf"Coefficient: ${np.round(b, 3)}$", ha="right", va="bottom",
            fontsize=fs(10), transform=ax.transAxes)
    ax.text(0.98, 0.01, rf"t-stat: ${np.round(t, 1)}$", ha="right", va="bottom",
            fontsize=fs(10), transform=ax.transAxes)

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(r"Empirical ")
    ax.set_ylabel(r"Simulated ")
    ax.grid(linestyle="dashed", alpha=0.5)
    ax.legend(loc="upper left", frameon=False)   # bottom-right is taken by the fit annotation
    _despine(ax)

    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        ax.figure.savefig(save_to, format="pdf", bbox_inches="tight")
    return ax


def plot_pi_r(data, ax=None, save_to=None):
    """Empirical vs simulated downstream sales shares pi_r."""
    fig, ax = (plt.subplots(figsize=get_figsize()) if ax is None else (ax.figure, ax))
    bubble_scatter(ax,
                   data["empirical_moments_dict"]["emp_pi_r"],
                   data["simulated_moments_dict"]["emp_pi_r"],
                   r"Empirical", r"Simulated", None, size_scale=400)
    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        ax.figure.savefig(save_to, format="pdf", bbox_inches="tight")
    return ax


# ============================================================================
# Fit with standard errors: extensive margin, zero-supplier share, count curve
# ============================================================================

REG_BIN_LABELS_5 = [r"$]0,50]$", r"$]50,100]$", r"$]100,150]$", r"$]150,200]$", r"$>200$"]


def reg_bin_labels(n_coef):
    if n_coef == 1:
        return [r"$\log d$"]
    if n_coef == 4:
        return REG_BIN_LABELS_5[1:]
    if n_coef == 5:
        return list(REG_BIN_LABELS_5)
    return [rf"$\beta_{{{i + 1}}}$" for i in range(n_coef)]


def _paired_errorbar(ax, labels, emp, sim, se_emp=None, se_sim=None, ci=1.96,
                     ylabel="", xlabel="", title=None, rotate=0):
    """
    Empirical (left) against simulated (right) at each tick, with +/- ci*SE bars.

    A point whose SE is zero, missing or NaN keeps its MARKER and simply gets no bar.
    Drawing it through `errorbar` with a zero yerr would put a bare cap on the marker,
    which reads as a measurement of zero precision rather than as an absent one; and a
    NaN yerr silently drops the point altogether, which is worse — the value is known,
    only its uncertainty is not.
    """
    x = np.arange(len(labels), dtype=float)
    off = 0.12

    def _series(xs, vals, se, **kw):
        vals = np.asarray(vals, float)
        if se is None:
            ax.errorbar(xs, vals, yerr=None, **kw)
            return
        se = np.asarray(se, float)
        has = np.isfinite(se) & (se > 0)
        # With bars, then without — one legend entry, taken from whichever is non-empty.
        if has.any():
            ax.errorbar(xs[has], vals[has], yerr=ci * se[has], **kw)
            kw = {**kw, "label": None}
        if (~has).any():
            ax.errorbar(xs[~has], vals[~has], yerr=None, **kw)

    _series(x - off, emp, se_emp,
            fmt="o", color=toulouse_color, capsize=3, markersize=7,
            markeredgecolor="black", markeredgewidth=0.5, linestyle="none",
            label="Empirical")
    _series(x + off, sim, se_sim,
            fmt="D", color=sim_color, capsize=3, markersize=6,
            markeredgecolor="black", markeredgewidth=0.5, linestyle="none",
            label="Simulated")
    #ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotate, ha="right" if rotate else "center")
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title)
    ax.grid(axis="y", linestyle="dashed", alpha=0.5)
    ax.legend(frameon=False)
    _despine(ax)
    return ax


def plot_reg_coef(data, ax=None, save_to=None, ci=1.96):
    """Empirical vs simulated cloglog distance coefficients, with SEs."""
    emp = np.asarray(data["empirical_moments_dict"]["reg_coef"], float)
    sim = np.asarray(data["simulated_moments_dict"]["reg_coef"], float)
    se_e = data["se_empirical"]["reg_coef"] if data["se_empirical"] else None
    se_s = data["se_simulated"]["reg_coef"] if data["se_simulated"] else None

    fig, ax = (plt.subplots(figsize=get_figsize(hf=0.62)) if ax is None else (ax.figure, ax))
    _paired_errorbar(ax, reg_bin_labels(data["n_coef"]), emp, sim, se_e, se_s, ci=ci,
                     ylabel=r"Extensive margin coefficient",
                     xlabel="Distance bin (km)")
    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        ax.figure.savefig(save_to, format="pdf", bbox_inches="tight")
    return ax


def plot_G0(data, ax=None, save_to=None, ci=1.96, annotate_N=True):
    """Empirical vs simulated Gbar_s(0) — the share of cells with no supplier."""
    if not data["granular"]:
        raise ValueError("Gbar_s(0) only exists under --granular=true")
    emp = np.asarray(data["empirical_moments_dict"]["G0"], float)
    sim = np.asarray(data["simulated_moments_dict"]["G0"], float)
    se_e = data["se_empirical"]["G0"] if data["se_empirical"] else None
    se_s = data["se_simulated"]["G0"] if data["se_simulated"] else None

    fig, ax = (plt.subplots(figsize=get_figsize(hf=0.62)) if ax is None else (ax.figure, ax))
    _paired_errorbar(ax, data["sector_names"], emp, sim, se_e, se_s, ci=ci,
                     ylabel=r"$\bar{G}_s(0)$", xlabel="Sector (A129)", rotate=45)

    # The profiled variety count is what the model moves to hit this moment, so
    # print it next to each sector — a clamped sector cannot close its own residual.
    diag = data.get("granular_diagnostics")
    if annotate_N and diag is not None and "N_hat" in diag:
        N_hat = np.asarray(diag["N_hat"]).ravel()
        clamped = np.asarray(diag.get("clamped", np.zeros_like(N_hat))).ravel()
        top = ax.get_ylim()[1]
        for i, (n, cl) in enumerate(zip(N_hat, clamped)):
            ax.text(i, top, rf"$\hat{{N}}_s={int(n)}$" + ("*" if cl != 0 else ""),
                    ha="center", va="bottom", fontsize=fs(12), rotation=45)
        #ax.set_ylim(ax.get_ylim()[0], top * 1.02)
    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        ax.figure.savefig(save_to, format="pdf", bbox_inches="tight")

    # The same two numbers are printed by Julia's own reporting; a gap between the
    # figure and that report is a wiring fault, not a modelling result, so say so
    # rather than leaving it to be discovered by eye.
    chk = g0_consistency(data, verbose=False)
    if chk is not None and not bool(chk.attrs.get("agrees", True)):
        print(f"[plot_G0] {data['industry']}: the figure and "
              f"{data['inference_step']}/granular_diagnostics.npz disagree on "
              "Gbar_s(0) — run g0_consistency(data) for the per-sector table.")
    return ax


G0_TOL = 1e-6


def g0_consistency(data, tol=None, verbose=True):
    """
    The four places Gbar_s(0) is written down, side by side.

    The figure above plots two numbers, and Julia's own reporting prints the same two
    out of a different file, so the pair is a cross-check rather than one source read
    twice:

    ``target``      G_K.csv at K = 0, i.e. block 6 of the empirical moment vector — the
                    red marker in the figure, and `G0_emp` in a stage `report.txt`.
    ``target_diag`` `G0_target` in `<step>/granular_diagnostics.npz`, written by
                    `report_granular` straight off `G_TARGET`.
    ``fit``         block 6 of `<step>/best_simulated_moments.npy` — the blue marker.
                    `run_reporting` evaluates it at the LAST stage folder's
                    `best_params.npy`, on `U_DRAWS`.
    ``fit_diag``    `G0_fit` in the same npz, evaluated by `granular_report` at the
                    theta the step saved (`theta_hat_1` for mu = 1, `theta_hat_2` for
                    mu = 2), also on `U_DRAWS`.

    The two thetas are the same vector — `run_optimization` returns the final
    sub-stage's parameters and writes that same vector into the last stage folder —
    and `Gbar_s(0)` is a closed form in the win counts, carrying no simulation noise of
    its own. So `fit` and `fit_diag` must agree to the inversion tolerance. They
    separate when the npz is STALE: `report_granular` writes it only when its step
    actually runs (Step 2 for mu = 1, Step 4 for mu = 2), while the post-hoc block of
    `main.jl` rewrites `best_simulated_moments.npy` on every invocation. A run resumed
    with `run_step4 = false` therefore leaves the npz at an older theta while the
    figure moves with the latest one.

    Returns a per-sector DataFrame (`None` when the npz is absent);
    `df.attrs["agrees"]` carries the verdict.
    """
    if not data["granular"]:
        raise ValueError("Gbar_s(0) only exists under --granular=true")
    tol = G0_TOL if tol is None else tol
    diag = data.get("granular_diagnostics")
    if diag is None:
        if verbose:
            print(f"[g0_consistency] {data['industry']}: no granular_diagnostics.npz in "
                  f"{data['folder'].name}/{data['inference_step']}/ — nothing to check against.")
        return None

    target = np.asarray(data["empirical_moments_dict"]["G0"], float)
    fit = np.asarray(data["simulated_moments_dict"]["G0"], float)
    nanlike = np.full_like(target, np.nan)
    target_diag = np.asarray(diag["G0_target"], float).ravel() if "G0_target" in diag else nanlike
    fit_diag = np.asarray(diag["G0_fit"], float).ravel() if "G0_fit" in diag else nanlike
    N_hat = np.asarray(diag["N_hat"], float).ravel() if "N_hat" in diag else nanlike
    clamp = np.asarray(diag["clamped"], float).ravel() if "clamped" in diag else np.zeros_like(target)

    df = pd.DataFrame({
        "target (G_K.csv)": target,
        "target (diagnostics)": target_diag,
        "d_target": target - target_diag,
        "fit (moment vector)": fit,
        "fit (diagnostics)": fit_diag,
        "d_fit": fit - fit_diag,
        "N_hat": N_hat,
        "clamp": np.where(clamp < 0, "lo", np.where(clamp > 0, "hi", "none")),
        "residual (fit - target)": fit - target,
    }, index=pd.Index(data["sector_names"], name="A129"))

    worst_t = float(np.nan_to_num(np.nanmax(np.abs(df["d_target"].values))))
    worst_f = float(np.nan_to_num(np.nanmax(np.abs(df["d_fit"].values))))
    agrees = bool(worst_t <= tol and worst_f <= tol)
    df.attrs.update(agrees=agrees, max_abs_target_gap=worst_t, max_abs_fit_gap=worst_f)

    if verbose or not agrees:
        src = (f"{data['folder'].name}: moments {data['step_dir']}/best_simulated_moments.npy "
               f"vs {data['inference_step']}/granular_diagnostics.npz")
        if agrees:
            print(f"[g0_consistency] {data['industry']} (mu = {data['mu']}) OK — {src}; "
                  f"max |gap| target {worst_t:.2e}, fit {worst_f:.2e}")
        else:
            print(f"[g0_consistency] {data['industry']} (mu = {data['mu']}) MISMATCH — {src}\n"
                  f"  max |gap| on the target {worst_t:.2e}, on the fit {worst_f:.2e} "
                  f"(tol {tol:.0e}).\n"
                  "  A TARGET gap means G_K.csv changed since the run: the figure reads the "
                  "CSV now, the npz froze G_TARGET at estimation time.\n"
                  "  A FIT gap means the npz sits at a different theta than the last stage "
                  "folder — typically a resumed run whose Step 2 / Step 4 did not re-execute. "
                  "Re-run that step, or read both numbers off the moment vector.")
    return df


# ---------------------------------------------------------------------------
# The whole count curve, Gbar_s(K) for K = 0, 1, 2, 3 — only K = 0 is targeted
# ---------------------------------------------------------------------------
# The K to show is a free parameter: the run cells pass `COUNT_CURVE_K` from the
# Constants cell, and this is the fallback when a function is called on its own.
_DEFAULT_COUNT_K = tuple(range(int(globals().get("COUNT_CURVE_K_MAX", 3)) + 1))


def _logfact(m):
    """`lg[i] = log(i!)` for i = 0..m, by cumulative sum — Julia's gbar_logfact_table."""
    lg = np.zeros(int(m) + 1)
    if m >= 1:
        lg[1:] = np.cumsum(np.log(np.arange(1, int(m) + 1, dtype=float)))
    return lg


def _log_binom(a, b, lg):
    return lg[a] - lg[b] - lg[a - b]


def gbar_cells(k, m, n, K, lg):
    """
    Unbiased per-cell estimate of `Pr(K_ls == K)` -- EXACTLY K suppliers, not at most K.

    A cell hosts a supplier for a variety it wins somewhere, so with `n` varieties
    `K_ls ~ Bin(n, q_ls)`. Take the `n` varieties to be `n` of the `m` simulated draws
    sampled WITHOUT replacement: the number of wins among them is Hypergeometric, and
    marginalising over the draws each is a win with probability `q` independently, so

        E[ C(k,K) C(m-k, n-K) / C(m,n) ]  =  C(n,K) q^K (1-q)^(n-K)

    exactly. At `K = 0` it collapses to `C(m-k,n)/C(m,n)`, which is `gbar_cell` in
    model_CP.jl verbatim -- so the targeted panel and the untargeted ones are the same
    estimator, and a discrepancy at K = 0 is a bug rather than a convention.

    INCREMENTS, not the CDF. `N_hat_s` is calibrated on the K = 0 level, so under a
    cumulative convention every panel would contain that fitted level and re-test what
    is already targeted; the increments isolate the free content -- the SHAPE of the
    supplier-count distribution (gate V8).
    """
    k = np.asarray(k, dtype=int)
    m, n, K = int(m), int(n), int(K)
    if n <= 0:
        return np.ones(k.shape) if K == 0 else np.zeros(k.shape)
    out = np.zeros(k.shape, float)
    if K < 0 or K > n:
        return out
    ok = (K <= k) & (n - K <= m - k)
    if ok.any():
        kk = k[ok]
        out[ok] = np.exp(_log_binom(kk, K, lg)
                         + _log_binom(m - kk, n - K, lg) - _log_binom(m, n, lg))
    return out


def _plugin_cells(q, n, K):
    """Plug-in fallback: the Binomial(n, q_hat) pmf at exactly K. Biased (Jensen), robust."""
    q = np.clip(np.asarray(q, float), 0.0, 1.0)
    n, K = int(n), int(K)
    if K < 0 or K > n:
        return np.zeros(q.shape, float)
    lg = _logfact(max(n, 1))
    return float(np.exp(_log_binom(n, K, lg))) * q ** K * (1 - q) ** (n - K)


def _cell_sectors(data):
    """
    Sector of each simulated cell, in the model's own good order.

    `load_parameters.jl` builds the cells as `findall(CELL_MASK)` over the `(S, R)`
    mask, and Julia's `findall` walks a matrix in COLUMN-major order — sector fastest,
    region slowest. Reading `q_hat` (which is stored in that order) with numpy's
    default C-order would silently shuffle cells between sectors.
    """
    S = data["S"]
    flat = np.flatnonzero(np.asarray(data["CELL_MASK"]).ravel(order="F"))
    return flat % S


def count_curve(data, K_values=None, tol=1e-9):
    """
    Empirical against simulated `Gbar_s(K)` = share of cells with EXACTLY K suppliers.

    **Only `K = 0` is targeted.** It is block 6 of the moment vector and the variety
    count `N_hat_s` is calibrated on it, so a good fit there is close to mechanical.
    `K >= 1` is free: nothing in the criterion asks the model to reproduce the SHAPE of
    the supplier-count distribution, which makes the rest of the curve an untargeted
    check (gate V8 of `documentation/granular_validation.md`).

    Increments, not the CDF -- see `gbar_cells`. `p_s(0) = G_s(0)`, so the targeted
    panel is the same number under either convention; only `K >= 1` changes.

    Sources, in order of preference:

    * simulated  -- `<step>/inference/G_curve_fitted.npy`, written by Julia at the same
      theta_hat and the same pinned `N_hat_s` the delta-method SE was built at. Absent,
      it is rebuilt here from `q_hat` and `N_hat_s` under the same unbiased convention,
      with `m = N_rho` RECOVERED from the diagnostics (`max(100, max N_HI)`, the rule in
      load_parameters.jl) and the recovery CHECKED twice: `q_hat * m` must be integral
      (it is a win count), and the K = 0 column must reproduce block 6 of the moment
      vector. If either fails the plug-in Binomial pmf is used and `attrs["estimator"]`
      says so.
    * empirical  -- the increments of `G_K.csv`, `p_s(K) = G(K) - G(K-1)`.

    Returns a (sector, K)-indexed DataFrame with both standard errors where available;
    `df.attrs` carries the estimator, the K = 0 gap against the moment vector, and the
    draw count used.
    """
    if not data["granular"]:
        raise ValueError("the count curve only exists under --granular=true")
    Ks = list(_DEFAULT_COUNT_K if K_values is None else K_values)
    S = data["S"]
    diag = data.get("granular_diagnostics")
    if diag is None or "q_hat" not in diag or "N_hat" not in diag:
        raise FileNotFoundError(
            f"{data['folder'].name}/{data['inference_step']}/granular_diagnostics.npz is "
            "absent or carries no q_hat/N_hat — the simulated curve is rebuilt from those "
            "two, so there is nothing to plot against. Re-run the step that writes it "
            f"({'Step 2' if data['mu'] == 1 else 'Step 4'} of main.jl).")

    q = np.asarray(diag["q_hat"], float).ravel()
    N_hat = np.rint(np.asarray(diag["N_hat"], float).ravel()).astype(int)
    N_HI = np.asarray(diag["N_HI"], float).ravel() if "N_HI" in diag else np.array([100.0])
    clamped = (np.asarray(diag["clamped"], float).ravel() if "clamped" in diag
               else np.zeros(S))
    good_s = _cell_sectors(data)
    if q.size != good_s.size:
        raise ValueError(f"q_hat has {q.size} cells but CELL_MASK has {good_s.size} — "
                         "the run's granular/ca_level flags and the ones passed here disagree.")

    n_rho = max(100, int(round(np.nanmax(N_HI))))        # load_parameters.jl's N_rho rule
    k_counts = np.rint(q * n_rho).astype(int)
    integral_gap = float(np.max(np.abs(q * n_rho - k_counts))) if q.size else 0.0
    lg = _logfact(n_rho)

    def curve(unbiased):
        out = np.full((S, len(Ks)), np.nan)
        for s in range(S):
            cells = good_s == s
            if not cells.any():
                continue
            for i, K in enumerate(Ks):
                out[s, i] = (gbar_cells(k_counts[cells], n_rho, N_hat[s], K, lg).mean()
                             if unbiased else _plugin_cells(q[cells], N_hat[s], K).mean())
        return out

    block6 = np.asarray(data["simulated_moments_dict"]["G0"], float)
    sim = curve(True)
    estimator, gap = "unbiased (gbar_cell)", np.inf
    if 0 in Ks:
        gap = float(np.nanmax(np.abs(sim[:, Ks.index(0)] - block6)))
    if integral_gap > 1e-6 or (0 in Ks and gap > 1e-6):
        sim_pi = curve(False)
        gap_pi = (float(np.nanmax(np.abs(sim_pi[:, Ks.index(0)] - block6)))
                  if 0 in Ks else np.inf)
        print(f"[count_curve] {data['industry']}: the unbiased reconstruction does not "
              f"reproduce block 6 at K = 0 (gap {gap:.2e}, q_hat*N_rho integrality "
              f"{integral_gap:.2e} at N_rho = {n_rho}); falling back to the plug-in "
              f"Binomial pmf (gap {gap_pi:.2e}). K >= 1 is then biased by Jensen "
              "— read it as indicative, and check N_rho against the run's log line.")
        sim, estimator, gap = sim_pi, "plug-in Binomial(N_hat, q_hat)", gap_pi

    # Julia's own curve wins when present: it was evaluated at the same theta_hat and
    # the same PINNED N_hat_s as the delta-method SE beside it, so the point and its
    # band cannot drift apart the way a Python reconstruction can.
    sim_se = np.full((S, len(Ks)), np.nan)
    jK = data.get("G_curve_sim_K")
    if jK is not None:
        jc, js = data.get("G_curve_sim"), data.get("G_curve_sim_se")
        for i, K in enumerate(Ks):
            if K in jK:
                c = jK.index(K)
                if js is not None and js.shape == (S, len(jK)):
                    sim_se[:, i] = js[:, c]
                if jc is not None and jc.shape == (S, len(jK)):
                    d = float(np.nanmax(np.abs(jc[:, c] - sim[:, i])))
                    if d > 1e-6:
                        print(f"[count_curve] {data['industry']}: K = {K}, Julia's fitted "
                              f"curve and the Python reconstruction differ by {d:.2e}; "
                              "using Julia's (it shares theta_hat and the pinned N_hat_s "
                              "with the SE). A large gap means N_rho was mis-recovered.")
                    sim[:, i] = jc[:, c]
        estimator += " [Julia G_curve_fitted]" if data.get("G_curve_sim") is not None else ""

    # ---- empirical: the increments of G_K.csv, and their bootstrap SE if supplied ----
    emp = np.full((S, len(Ks)), np.nan)
    emp_se = np.full((S, len(Ks)), np.nan)
    gk, gp, gse = data.get("G_curve_K"), data.get("G_pmf"), data.get("G_pmf_se")
    if gp is not None and gk is not None:
        for i, K in enumerate(Ks):
            if K in gk:
                emp[:, i] = gp[:, gk.index(K)]
                if gse is not None:
                    emp_se[:, i] = gse[:, gk.index(K)]
    missing = [K for K in Ks if gk is not None and K not in gk]
    if missing:
        print(f"[count_curve] {data['industry']}: G_K.csv has no row for K = {missing} — "
              "those panels show the simulated curve only.")

    # Free consistency gate: p_s(0) = G_s(0) IS block 6, whose bootstrap variance is
    # already the G block of Sigma_data. A supplied CSV must agree with it at K = 0.
    if 0 in Ks and gse is not None and data.get("se_empirical") is not None:
        ref = np.asarray(data["se_empirical"]["G0"], float)
        got = emp_se[:, Ks.index(0)]
        ok = np.isfinite(got) & np.isfinite(ref) & (ref > 0)
        if ok.any():
            rel = float(np.nanmax(np.abs(got[ok] - ref[ok]) / ref[ok]))
            if rel > 0.05:
                print(f"[count_curve] {data['industry']}: the supplied G_K_var.csv "
                      f"disagrees with diag(Sigma_data)'s count block at K = 0 by up to "
                      f"{rel:.1%}. They are the SAME object (p_s(0) = G_s(0)), so one of "
                      "the two bootstraps is not the one the estimator used — most "
                      "likely the CSV holds the CDF's variance rather than the "
                      "increment's, which coincide only at K = 0 if the resampling is "
                      "identical.")

    idx = pd.MultiIndex.from_product([data["sector_names"], Ks], names=["A129", "K"])
    df = pd.DataFrame({"empirical": emp.ravel(),
                       "simulated": sim.ravel(),
                       "se_empirical": emp_se.ravel(),
                       "se_simulated": sim_se.ravel(),
                       "residual": (sim - emp).ravel(),
                       "N_hat": np.repeat(N_hat, len(Ks)),
                       "clamped": np.repeat(clamped != 0, len(Ks)),
                       "targeted": np.tile([K == 0 for K in Ks], S)}, index=idx)
    df.attrs.update(estimator=estimator, K0_gap_vs_moment_vector=gap,
                    n_rho=n_rho, K_values=Ks)
    return df


def plot_count_curve(data, K_values=None, save_to=None, ci=1.96, df=None):
    """
    One INDEPENDENT figure per K: empirical against simulated `Gbar_s(K)` across sectors.

    Same marks and same layout as `plot_G0`, which IS the K = 0 case — so each K stands
    on its own, in its own file, and can be dropped into a paper or a slide without
    carrying the others with it. The y label names the K it shows (`Gbar_s(1)`,
    `Gbar_s(2)`, ...), which is what distinguishes the panels now that there is no title
    and no shared axis to read them against.

    Because the panels are INCREMENTS rather than a CDF they do not climb mechanically
    toward 1: a model whose count distribution is too concentrated puts too much mass at
    low K and too little in the tail, which reads directly as a mismatch at each K.

    Both bands are delta-method / bootstrap objects, and neither is decorative:

    * simulated — `se_G_curve_delta.npy`. At K = 0 for a FREE sector it collapses to the
      bootstrap SE of the target, because `N_hat_s` is calibrated to make the fitted
      moment EQUAL the target: the K = 0 fit is not evidence, and the band says so. For
      a CLAMPED sector `N_hat_s` cannot move, the target channel drops out, and the band
      is the alpha + gamma propagation instead. For K >= 1 it is the residualized
      structural loading plus the scaled target channel.
    * empirical — the bootstrap SE of the increments, from `G_K_var.csv`. Absent, the
      empirical points are drawn bare rather than with an invented bar.

    `save_to` is a STEM, not a file: K is appended before the extension, so
    `.../count_curve_auto_mu2.pdf` writes `..._K0.pdf`, `..._K1.pdf`, ... One call still
    produces the whole set, and nothing overwrites anything.

    `K_values` selects which K to draw; passed alongside a precomputed `df` it SUBSETS
    it, which is how a run cell draws only the untargeted K >= 1 while `plot_G0` keeps
    the targeted one (with its N_hat annotation). Returns `{K: ax}`.
    """
    df = count_curve(data, K_values=K_values) if df is None else df
    available = list(df.attrs["K_values"])
    Ks = available if K_values is None else [int(K) for K in np.atleast_1d(K_values)]
    absent = [K for K in Ks if K not in available]
    if absent:
        raise KeyError(f"K = {absent} is not in the count curve passed here "
                       f"(it carries K = {available}) — rebuild it with "
                       f"`count_curve(data, K_values={Ks})`.")
    sectors = data["sector_names"]

    def _band(v):
        # NaN and zero entries are passed through as they are: `_paired_errorbar` keeps
        # the marker and omits the bar for those, so a sector with no measured variance
        # is still visible as a point. A column with no SE anywhere gets None.
        v = np.asarray(v, float)
        return None if not np.isfinite(v).any() else v

    stem, ext = os.path.splitext(save_to) if save_to else (None, ".pdf")
    axes = {}
    for K in Ks:
        sub = df.xs(K, level="K").reindex(sectors)
        _, ax = plt.subplots(figsize=get_figsize(hf=0.62))
        _paired_errorbar(
            ax, sectors, sub["empirical"].values, sub["simulated"].values,
            _band(sub["se_empirical"].values), _band(sub["se_simulated"].values), ci=ci,
            ylabel=rf"$\bar{{G}}_s({K})$", xlabel="Sector (A129)", rotate=45)
        ax.figure.tight_layout()
        if stem:
            path = f"{stem}_K{K}{ext or '.pdf'}"
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            ax.figure.savefig(path, format="pdf", bbox_inches="tight")
        axes[K] = ax
    return axes


# ============================================================================
# Jacobian -- which parameter moves which moment
# ============================================================================

# Jacobian heatmaps. `kind="elasticity"` is what should be read; "raw" mixes units.
# The parameter axis carries the variety counts N_s when the run wrote them (see the
# markdown above); `data["n_N_cols"]` says how many columns they occupy.

JACOBIAN_BOUNDARIES = [-1, -0.1, -0.05, -0.025, -0.01, -0.001,
                       0.001, 0.01, 0.025, 0.05, 0.1, 1]

# An elasticity is unreadable when its Monte-Carlo standard deviation across the K
# Jacobian replications is at least this fraction of the elasticity itself.
NOISE_MAX = 0.5


def _binned_cmap(boundaries, coarse=False):
    from matplotlib.colors import BoundaryNorm, ListedColormap
    if coarse:
        return ListedColormap(["#2166ac", "#f0f0f0", "#b2182b"]), \
            BoundaryNorm([-1, -0.1, 0.1, 1], 3)
    n_bins = len(boundaries) - 1
    cmap = ListedColormap(plt.get_cmap("RdBu_r")(np.linspace(0, 1, n_bins)))
    return cmap, BoundaryNorm(boundaries, n_bins)


def jacobian_matrix(data, kind="elasticity"):
    """The Jacobian to plot, with a clear error when it is not on disk."""
    key = {"elasticity": "J_elast", "raw": "J", "sd": "J_sd",
           "elasticity_sd": "J_elast_sd"}[kind]
    J = data.get(key)
    if J is None:
        step = data["inference_step"]
        raise FileNotFoundError(
            f"no Jacobian in {data['folder']}/{step}/ — main.jl writes "
            f"jacobian_all*.npy at theta_hat_1 (step2) and jacobian_all_step3*.npy at "
            f"theta_hat_2 (step3); the corresponding inference step must have run.")
    return J


def jacobian_noise_ratio(data, kind="elasticity"):
    """
    sigma / |epsilon| entry by entry — the Monte-Carlo standard deviation of the
    elasticity across the K Jacobian replications, relative to the elasticity itself.

    Scale-free: the theta/m rescaling that turns dm/dtheta into an elasticity cancels
    between numerator and denominator, so this is the same number for the raw and the
    elasticity Jacobian. An EXACTLY zero entry gets ratio 0 rather than infinity: a
    structural zero (the N_s columns outside block 6) carries no simulation noise, and
    conflating it with an unmeasured entry is precisely what this diagnostic is for.
    """
    E = np.asarray(jacobian_matrix(data, kind), float)
    sd = data.get("J_elast_sd" if "elast" in kind else "J_sd")
    if sd is None:
        raise FileNotFoundError(
            f"no across-replication SD for the Jacobian in {data['folder']}/"
            f"{data['inference_step']}/ — compute_jacobian writes it as "
            "jacobian_all*_elasticity_sd.npy when K > 1.")
    sd = np.abs(np.asarray(sd, float))
    A = np.abs(E)
    with np.errstate(divide="ignore", invalid="ignore"):
        R = sd / A
    R[(A == 0) & (sd == 0)] = 0.0          # structural zero: measured, and exactly zero
    R[~np.isfinite(R)] = np.inf            # nonzero noise on a zero signal: unreadable
    return R


def jacobian_noise_mask(data, kind="elasticity", noise_max=NOISE_MAX):
    """True where the elasticity is DROWNED in simulation noise and must not be read."""
    return jacobian_noise_ratio(data, kind) >= noise_max


def _edges(sizes):
    return np.cumsum([0] + list(sizes))


def _tick_every(n, target=25):
    """Tick stride that keeps roughly `target` labels on an axis of length n."""
    return max(1, int(np.ceil(n / target)))


def _block_grid(ax, data, m_edges, p_edges, block_names=True):
    """Block separators, plus the block names on the outer margins."""
    for e in m_edges[1:-1]:
        ax.axhline(e - 0.5, color="black", linewidth=1.2, alpha=0.6)
    for e in p_edges[1:-1]:
        ax.axvline(e - 0.5, color="black", linewidth=1.2, alpha=0.6)
    if not block_names:
        return
    ax_top, ax_right = ax.secondary_xaxis("top"), ax.secondary_yaxis("right")
    ax_top.set_xticks((p_edges[:-1] + p_edges[1:]) / 2 - 0.5)
    ax_top.set_xticklabels(data["param_block_names"], fontsize=fs(10), fontweight="bold",
                           rotation=45)
    ax_right.set_yticks((m_edges[:-1] + m_edges[1:]) / 2 - 0.5)
    ax_right.set_yticklabels(data["moment_block_names"], fontsize=fs(10), fontweight="bold")


def _axis_labels(ax, data, shape):
    xs = np.arange(shape[1])[::_tick_every(shape[1])]
    ax.set_xticks(xs)
    ax.set_xticklabels(np.array(data["param_labels"])[xs], rotation=90, fontsize=fs(7))
    ax.xaxis.set_ticks_position("bottom")
    ys = np.arange(shape[0])[::_tick_every(shape[0])]
    ax.set_yticks(ys)
    ax.set_yticklabels(np.array(data["moment_labels"])[ys], fontsize=fs(7))
    ax.set_xlabel("Parameter")
    ax.set_ylabel("Moment")


def _sector_subblocks(data, block_name="Regional sourcing shares"):
    """
    The (start, stop, sector) runs INSIDE one moment block, read off the labels.

    The gamma rows are labelled `gamma[<sector>-AA<area>]` and stored s-major, so the
    sector is the text between "gamma[" and the LAST "-AA" and its rows are contiguous.
    Returns absolute row indices, so the caller can draw on the full matrix without
    knowing where the block starts. An empty list when the block is absent or its labels
    do not carry a sector, which is the right behaviour for every other block.
    """
    names = list(data["moment_block_names"])
    if block_name not in names:
        return []
    e = _edges(data["moment_block_sizes"])
    lo, hi = int(e[names.index(block_name)]), int(e[names.index(block_name) + 1])
    labels = list(data["moment_labels"])[lo:hi]

    def _sector(lab):
        if not lab.startswith("gamma[") or "-AA" not in lab:
            return None
        return lab[len("gamma["):lab.rfind("-AA")]

    runs, start, cur = [], 0, _sector(labels[0]) if labels else None
    if cur is None:
        return []
    for i, lab in enumerate(labels[1:] + [None], start=1):
        s = _sector(lab) if lab is not None else object()
        if s != cur:
            runs.append((lo + start, lo + i, cur))
            start, cur = i, s
    return runs


def _sector_axis(ax, data, block_name="Regional sourcing shares", fontsize=fs(8)):
    """
    Left y axis: one tick per sector of `block_name`, at the middle of its rows, with a
    thin separator between sectors. Every other row label is dropped -- on a matrix this
    tall the individual (sector, area) names are unreadable anyway, and what the reader
    needs from the left margin is which sector a band of rows belongs to.
    """
    runs = _sector_subblocks(data, block_name)
    if not runs:
        ax.set_yticks([])
        return runs
    for lo, hi, _ in runs[1:]:
        ax.axhline(lo - 0.5, color="black", linewidth=0.5, alpha=0.35, linestyle=":")
    ax.set_yticks([(lo + hi) / 2 - 0.5 for lo, hi, _ in runs])
    ax.set_yticklabels([s for _, _, s in runs], fontsize=fontsize)
    ax.tick_params(axis="y", length=2)
    return runs


def plot_jacobian_full(data, kind="elasticity", coarse=False, figsize=(15, 11),
                       mask=None, title=None, save_to=None):
    """
    One heatmap of the whole Jacobian, moment blocks on the rows and parameter blocks on
    the columns — the direct read of which parameter block moves which moment block.

    `mask` (boolean, same shape) blanks entries to white: pass `jacobian_noise_mask(...)`
    for the third panel of the triptych, i.e. the matrix purged of the entries whose
    elasticity is not measured precisely enough to be read.
    """
    J = np.asarray(jacobian_matrix(data, kind), float)
    cmap, norm = _binned_cmap(JACOBIAN_BOUNDARIES, coarse=coarse)
    cmap = cmap.copy()
    cmap.set_bad("white")
    M = np.ma.masked_where(np.asarray(mask, bool), J) if mask is not None else J
    m_edges, p_edges = _edges(data["moment_block_sizes"]), _edges(data["param_block_sizes"])

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.matshow(M, cmap=cmap, norm=norm, aspect="auto")
    _block_grid(ax, data, m_edges, p_edges)
    _axis_labels(ax, data, J.shape)

    if title is None:
        title = (f"Jacobian ({kind}) — {data['industry']}, "
                 rf"$\hat{{\mu}}_{data['mu']}$   ({J.shape[0]}$\times${J.shape[1]})")
        if mask is not None:
            kept = float((~np.asarray(mask, bool)).mean())
            title += f"\nnoise-purged: {100 * kept:.1f}% of entries readable"
    ax.set_title(title, pad=28)
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.04, pad=0.22,
                        ticks=(None if coarse else JACOBIAN_BOUNDARIES))
    cbar.set_label("Elasticity" if "elast" in kind else "d m / d theta")
    if not coarse:
        cbar.ax.set_xticklabels([f"{t:g}" for t in JACOBIAN_BOUNDARIES],
                                fontsize=fs(8), rotation=45)
    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        fig.savefig(save_to, bbox_inches="tight")
    return ax


def plot_jacobian_noise(data, kind="elasticity", noise_max=NOISE_MAX, figsize=(15, 11),
                        save_to=None):
    """
    The middle panel of the triptych: the noise-to-signal ratio sigma/|epsilon| alone.

    Dark = the elasticity is well measured relative to its own size; light = it is not.
    Entries at or above `noise_max` are hatched, and those are exactly the ones the
    purged panel blanks out. Reading this panel first says WHICH parts of the Jacobian
    can be interpreted at all, before any statement is made about what they show.
    """
    R = jacobian_noise_ratio(data, kind)
    bad = R >= noise_max
    m_edges, p_edges = _edges(data["moment_block_sizes"]), _edges(data["param_block_sizes"])

    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_over("black")
    im = ax.matshow(np.clip(R, 0, noise_max * 2), cmap=cmap, aspect="auto",
                    vmin=0, vmax=noise_max * 2)
    # mark the failures explicitly, so a reader is not left inferring them from a colour
    ys, xs = np.where(bad)
    ax.scatter(xs, ys, s=1.5, marker="s", color="black", alpha=0.35, linewidths=0)
    _block_grid(ax, data, m_edges, p_edges)
    _axis_labels(ax, data, R.shape)

    share = float(bad.mean())
    ax.set_title(f"Jacobian noise-to-signal $\\sigma/|\\varepsilon|$ — "
                 f"{data['industry']}, " rf"$\hat{{\mu}}_{data['mu']}$"
                 f"\n{100 * share:.1f}{pct()} of entries at or above {noise_max:g} "
                 "(black squares) — not readable", pad=28)
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.04, pad=0.22,
                        extend="max")
    cbar.set_label(r"$\sigma_{jk} / |\varepsilon_{jk}|$ across the Jacobian replications"
                   f"   (threshold {noise_max:g})")
    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        fig.savefig(save_to, bbox_inches="tight")
    return ax


def plot_jacobian_triptych(data, kind="elasticity", noise_max=NOISE_MAX,
                           figsize=(15, 11), out_folder=None, tag="jacobian",
                           show=True):
    """
    The three panels together: the elasticity Jacobian, the noise-to-signal map, and the
    Jacobian purged of the entries the map rejects. Saves `<tag>`, `<tag>_noise` and
    `<tag>_purged` under `out_folder` when one is given.
    """
    def _p(name):
        return None if out_folder is None else \
            f"{out_folder}/{name}_{data['industry']}_mu{data['mu']}.pdf"

    mask = jacobian_noise_mask(data, kind=kind, noise_max=noise_max)
    axes = [plot_jacobian_full(data, kind=kind, figsize=figsize, save_to=_p(tag))]
    if show:
        plt.show()
    axes.append(plot_jacobian_noise(data, kind=kind, noise_max=noise_max,
                                    figsize=figsize, save_to=_p(tag + "_noise")))
    if show:
        plt.show()
    axes.append(plot_jacobian_full(data, kind=kind, figsize=figsize, mask=mask,
                                   save_to=_p(tag + "_purged")))
    if show:
        plt.show()
    return axes


def plot_jacobian_blocks(data, kind="elasticity", coarse=True, ncols=3,
                         figsize=(18, 12), mask=None, save_to=None):
    """One panel per MOMENT block, columns spanning all parameters."""
    J = np.asarray(jacobian_matrix(data, kind), float)
    cmap, norm = _binned_cmap(JACOBIAN_BOUNDARIES, coarse=coarse)
    cmap = cmap.copy()
    cmap.set_bad("white")
    if mask is not None:
        J = np.ma.masked_where(np.asarray(mask, bool), J)
    m_edges, p_edges = _edges(data["moment_block_sizes"]), _edges(data["param_block_sizes"])
    blocks = np.split(J, m_edges[1:-1], axis=0)
    names = data["moment_block_names"]

    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for i, (name, block) in enumerate(zip(names, blocks)):
        ax = axes[i]
        im = ax.matshow(block, cmap=cmap, norm=norm, aspect="auto")
        for e in p_edges[1:-1]:
            ax.axvline(e - 0.5, color="black", linewidth=1.2, alpha=0.5)

        cbar = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.12)
        cbar.set_label("Elasticity", fontsize=fs(9))
        ax.set_title(f"{name}\n({block.shape[0]}x{block.shape[1]})",
                     fontsize=fs(13), fontweight="bold", pad=12)
        ax.set_xlabel("Parameter", fontsize=fs(10))

        xs = np.arange(block.shape[1])[::_tick_every(block.shape[1], 20)]
        ax.set_xticks(xs)
        ax.set_xticklabels(np.array(data["param_labels"])[xs], rotation=90, fontsize=fs(7))
        ax.xaxis.set_ticks_position("bottom")

        row_labels = np.array(data["moment_labels"][m_edges[i]:m_edges[i + 1]])
        ys = np.arange(block.shape[0])[::_tick_every(block.shape[0], 20)]
        ax.set_yticks(ys)
        ax.set_yticklabels(row_labels[ys], fontsize=fs(7))

    for ax in axes[len(names):]:
        ax.axis("off")
    fig.suptitle(f"Jacobian by moment block — {data['industry']}, "
                 rf"$\hat{{\mu}}_{data['mu']}$", fontsize=fs(18), fontweight="bold")
    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        fig.savefig(save_to, bbox_inches="tight")
    return axes[:len(names)]


def jacobian_block_summary(data, kind="elasticity", noise_max=NOISE_MAX):
    """
    Per (moment block x parameter block): the mean |elasticity|, and the share of the
    block's entries that are readable at all (noise-to-signal below `noise_max`).

    A large mean elasticity in a block whose readable share is low is a statement about
    the simulation, not about the model.
    """
    E = np.abs(np.asarray(jacobian_matrix(data, kind), float))
    try:
        ok = ~jacobian_noise_mask(data, kind=kind, noise_max=noise_max)
    except FileNotFoundError:
        ok = np.ones_like(E, dtype=bool)
    m_edges, p_edges = _edges(data["moment_block_sizes"]), _edges(data["param_block_sizes"])
    rows = []
    for i, mname in enumerate(data["moment_block_names"]):
        for j, pname in enumerate(data["param_block_names"]):
            sl = (slice(m_edges[i], m_edges[i + 1]), slice(p_edges[j], p_edges[j + 1]))
            sub, sub_ok = E[sl], ok[sl]
            rows.append({"moment_block": mname, "param_block": pname,
                         "mean_abs": sub.mean() if sub.size else np.nan,
                         "max_abs": sub.max() if sub.size else np.nan,
                         "share_readable": sub_ok.mean() if sub.size else np.nan})
    df = pd.DataFrame(rows)
    return df.set_index(["moment_block", "param_block"]).unstack("param_block") \
             .reindex(index=data["moment_block_names"]) \
             .reindex(columns=data["param_block_names"], level=1)


# ============================================================================
# Variance-covariance of the moments
# ============================================================================

# The three moment covariance matrices, and the correlations they imply.

def _gb_edges(data):
    """
    Block edges of the inference subsystem and their names.

    The three blocks are stacked in a fixed order — extensive-margin coefficients,
    sourcing shares, zero-supplier shares — on both axes of every Sigma / Omega / W on
    disk. That order is the "beta, gamma, G" ordering invariant of CLAUDE.md.
    """
    n_beta = data["n_coef"]
    n_gam = int(np.asarray(data["moment_block_sizes"])[4])
    sizes = [n_beta, n_gam] + ([data["S"]] if data["granular"] else [])
    names = ["reg_coef", "gamma", "G0"][:len(sizes)]
    return np.cumsum([0] + sizes), names


def _cov_panel(ax, M, data, title, mode="cov", pct=95):
    """One matrix panel with the block grid drawn on; `mode` is "cov" or "corr"."""
    from matplotlib.colors import TwoSlopeNorm
    edges, names = _gb_edges(data)
    if mode == "corr":
        d = np.sqrt(np.clip(np.diag(M), 1e-300, None))
        M = M / np.outer(d, d)
        im = ax.matshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    else:
        v = np.percentile(np.abs(M), pct)
        v = v if v > 0 else (np.abs(M).max() or 1.0)
        im = ax.matshow(M, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-v, vcenter=0, vmax=v))
    for e in edges[1:-1]:
        ax.axhline(e - 0.5, color="black", linewidth=0.8, alpha=0.6)
        ax.axvline(e - 0.5, color="black", linewidth=0.8, alpha=0.6)
    ticks = (edges[:-1] + edges[1:]) / 2 - 0.5
    ax.set_xticks(ticks)
    ax.set_xticklabels(names, fontsize=fs(9))
    ax.set_yticks(ticks)
    ax.set_yticklabels(names, fontsize=fs(9), rotation=90, va="center")
    ax.set_title(title, pad=14)
    return im


def _cov_matrices(data):
    return [("$\\Sigma_{sim}$ (simulation noise)", data.get("Sigma_sim")),
            ("$\\Sigma_{data}$ (bootstrap)", data.get("Sigma_data")),
            ("$\\Omega = \\Sigma_{data} + \\Sigma_{sim}$", data.get("Omega"))]


def plot_variance_covariance(data, figsize=(16, 5.5), pct=95, save_to=None):
    """Sigma_sim, Sigma_data and Omega side by side, each on its own diverging scale."""
    mats = _cov_matrices(data)
    if all(m is None for _, m in mats):
        raise FileNotFoundError(f"no Sigma_*/Omega in {data['folder']}/step2/ — "
                                "build_step3_weight_matrix writes them in Step 2.")
    fig, axes = plt.subplots(1, len(mats), figsize=figsize)
    for ax, (title, M) in zip(np.atleast_1d(axes), mats):
        if M is None:
            ax.axis("off")
            ax.set_title(title + "\n(absent)")
            continue
        im = _cov_panel(ax, M, data, title, mode="cov", pct=pct)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Moment variance-covariance — {data['industry']} "
                 "(extensive margin, sourcing shares, zero-supplier shares)", fontsize=fs(14))
    fig.tight_layout()
    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        fig.savefig(save_to, bbox_inches="tight")
    return axes


def plot_moment_correlation(data, figsize=(16, 5.5), save_to=None):
    """
    The same three matrices as CORRELATIONS, on a common [-1, 1] scale.

    Rescaling by the diagonal removes the magnitude — which the covariance figure is
    there to show — and leaves the dependence structure, so the three are comparable:
    whether the bootstrap error of the sourcing shares is correlated across areas,
    whether the simulator's noise is (it is, through the shared draws), and which of the
    two Omega inherits its structure from.
    """
    mats = _cov_matrices(data)
    if all(m is None for _, m in mats):
        raise FileNotFoundError(f"no Sigma_*/Omega in {data['folder']}/step2/")
    fig, axes = plt.subplots(1, len(mats), figsize=figsize)
    for ax, (title, M) in zip(np.atleast_1d(axes), mats):
        if M is None:
            ax.axis("off")
            ax.set_title(title + "\n(absent)")
            continue
        im = _cov_panel(ax, M, data, title, mode="corr")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Moment CORRELATION — {data['industry']} "
                 "(same three matrices, rescaled by their own diagonal)", fontsize=fs(14))
    fig.tight_layout()
    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        fig.savefig(save_to, bbox_inches="tight")
    return axes


def variance_covariance_summary(data):
    """
    How much of the weighting is simulation noise, and how well conditioned is Omega?

    `trace(Sigma_sim)/trace(Sigma_data)` is the share of total moment variance that comes
    from the simulation rather than the data — raise `--n_rho_inf` if it is not small.
    `mean_abs_offdiag_corr` is the average absolute off-diagonal correlation, the same
    thing the correlation figure shows, as one number per matrix. The condition numbers
    say whether inverting Omega to build W is safe.
    """
    out = {}
    for name in ("Sigma_data", "Sigma_sim", "Omega"):
        M = data.get(name)
        if M is None:
            continue
        ev = np.linalg.eigvalsh((M + M.T) / 2)
        d = np.sqrt(np.clip(np.diag(M), 1e-300, None))
        C = M / np.outer(d, d)
        off = ~np.eye(C.shape[0], dtype=bool)
        out[name] = {"trace": float(np.trace(M)),
                     "eig_min": float(ev.min()), "eig_max": float(ev.max()),
                     "cond": float(ev.max() / ev.min()) if ev.min() > 0 else np.inf,
                     "mean_abs_offdiag_corr": float(np.abs(C[off]).mean())}
    if "Sigma_sim" in out and "Sigma_data" in out and out["Sigma_data"]["trace"] != 0:
        out["Sigma_sim"]["share_of_data_trace"] = \
            out["Sigma_sim"]["trace"] / out["Sigma_data"]["trace"]
    return pd.DataFrame(out).T


# ============================================================================
# Identification / sensitivity
# ============================================================================

# Identification / sensitivity — the elasticity Jacobian read as the experiment
# "move one parameter, watch the moments", with two thresholds: is there a channel
# (IDENT_THRESHOLD), and is it measured well enough to see (NOISE_MAX).
#
# The variety counts N_s are part of the parameter axis: the Julia side appends
# dm/dN_s to the saved Jacobian, so no re-attachment is done here.

IDENT_THRESHOLD = 0.01


def _blocks_2d(data, M):
    """M split into (moment block x parameter block) sub-matrices."""
    m_edges, p_edges = _edges(data["moment_block_sizes"]), _edges(data["param_block_sizes"])
    return {(i, j): M[m_edges[i]:m_edges[i + 1], p_edges[j]:p_edges[j + 1]]
            for i in range(len(data["moment_block_names"]))
            for j in range(len(data["param_block_names"]))}


def _readable(data, kind="elasticity", noise_max=NOISE_MAX):
    """Boolean matrix: entries whose elasticity is precise enough to be interpreted."""
    if noise_max is None:
        return np.ones_like(np.asarray(jacobian_matrix(data, kind), float), dtype=bool)
    try:
        return ~jacobian_noise_mask(data, kind=kind, noise_max=noise_max)
    except FileNotFoundError:
        return np.ones_like(np.asarray(jacobian_matrix(data, kind), float), dtype=bool)


def identification_summary(data, threshold=IDENT_THRESHOLD, kind="elasticity",
                           noise_max=NOISE_MAX):
    """
    Per (moment block x parameter block), four numbers that answer four questions:

      share_above       is there a channel?  share of entries with |elasticity| >= threshold
      share_readable    can it be seen?      share whose sigma/|elasticity| < noise_max
      share_live        both at once         above the threshold AND readable
      share_exact_zero  is it absent by construction?  share of entries exactly 0

    The last two are the pair worth reading together: an exact zero is the model saying
    the channel does not exist (the N_s columns outside block 6, asserted in Julia),
    whereas an unreadable entry says nothing at all.
    """
    E = np.abs(np.asarray(jacobian_matrix(data, kind), float))
    ok = _readable(data, kind, noise_max)
    rows = []
    for (i, j), sub in _blocks_2d(data, E).items():
        sub_ok = _blocks_2d(data, ok)[(i, j)]
        rows.append({
            "moment_block": data["moment_block_names"][i],
            "param_block": data["param_block_names"][j],
            "share_above": float((sub >= threshold).mean()) if sub.size else np.nan,
            "share_readable": float(sub_ok.mean()) if sub.size else np.nan,
            "share_live": float(((sub >= threshold) & sub_ok).mean()) if sub.size else np.nan,
            "max_abs": float(sub.max()) if sub.size else np.nan,
            "mean_abs": float(sub.mean()) if sub.size else np.nan,
            "share_exact_zero": float((sub == 0).mean()) if sub.size else np.nan,
        })
    df = pd.DataFrame(rows)
    return df.set_index(["moment_block", "param_block"]).unstack("param_block") \
             .reindex(index=data["moment_block_names"]) \
             .reindex(columns=data["param_block_names"], level=1)


def plot_identification_map(data, threshold=IDENT_THRESHOLD, kind="elasticity",
                            noise_max=NOISE_MAX, figsize=(9, 6), save_to=None):
    """
    One cell per (moment block x parameter block), coloured by the share of entries that
    are BOTH above the threshold and readable, annotated with that share, the block's
    max |elasticity|, and — when it is not 100% — the share of the block that is readable
    at all. A near-diagonal pattern down the (Trade cost, Comparative advantage, Variety
    count) columns is the identification claim, read off directly.
    """
    E = np.abs(np.asarray(jacobian_matrix(data, kind), float))
    ok = _readable(data, kind, noise_max)
    m_names, p_names = data["moment_block_names"], data["param_block_names"]
    blocks, blocks_ok = _blocks_2d(data, E), _blocks_2d(data, ok)
    n_m, n_p = len(m_names), len(p_names)

    def _grid(fn):
        return np.array([[fn(blocks[(i, j)], blocks_ok[(i, j)])
                          if blocks[(i, j)].size else np.nan
                          for j in range(n_p)] for i in range(n_m)])

    live = _grid(lambda s, o: ((s >= threshold) & o).mean())
    mx = _grid(lambda s, o: s.max())
    read = _grid(lambda s, o: o.mean())

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.matshow(live, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(n_m):
        for j in range(n_p):
            if not np.isfinite(live[i, j]):
                continue
            txt = f"{100 * live[i, j]:.0f}{pct()}\nmax {mx[i, j]:.3g}"
            if read[i, j] < 0.999:
                txt += f"\n({100 * read[i, j]:.0f}{pct()} readable)"
            ax.text(j, i, txt, ha="center", va="center", fontsize=fs(8),
                    color="white" if live[i, j] > 0.55 else "black")
    ax.set_xticks(range(n_p))
    ax.set_xticklabels(p_names, rotation=30, ha="left", fontsize=fs(10))
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(range(n_m))
    ax.set_yticklabels(m_names, fontsize=fs(10))
    ax.set_xlabel("Parameter block")
    ax.set_ylabel("Moment block")
    ax.set_title(f"Identification map — {data['industry']}, "
                 rf"$\hat{{\mu}}_{data['mu']}$"
                 f"\nshare of $|\\varepsilon| \\geq {threshold:g}$ AND "
                 f"$\\sigma/|\\varepsilon| < {noise_max:g}$", pad=52)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("share of entries that are both large and measured")
    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        fig.savefig(save_to, bbox_inches="tight")
    return ax


def plot_jacobian_thresholded(data, threshold=IDENT_THRESHOLD, kind="elasticity",
                              noise_max=NOISE_MAX, figsize=(15, 11), save_to=None,
                              sector_block="Regional sourcing shares"):
    """
    The whole elasticity Jacobian with two kinds of entry removed: those below the
    threshold (grey — a channel too small to matter) and those drowned in simulation
    noise (white — nothing was measured). What is left is the set of channels through
    which a parameter can actually be identified, on a symmetric log colour scale so
    three orders of magnitude of live channel stay distinguishable.

    The margins carry only the block names (top and right); the left axis names the
    sectors of `sector_block`, separated by a thin rule, so a row band can be attributed
    to an industry without the figure listing every commuting zone.
    """
    from matplotlib.colors import SymLogNorm

    E = np.asarray(jacobian_matrix(data, kind), float)
    A = np.abs(E)
    ok = _readable(data, kind, noise_max)
    vmax = A[ok].max() if ok.any() and A[ok].max() > 0 else 1.0

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#f2f2f2")                     # below threshold
    small = np.ma.masked_where(A < threshold, E)
    M = np.ma.masked_where(~ok, small)          # unreadable painted white on top
    m_edges, p_edges = _edges(data["moment_block_sizes"]), _edges(data["param_block_sizes"])

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("white")
    im = ax.matshow(M, cmap=cmap, aspect="auto",
                    norm=SymLogNorm(linthresh=threshold, vmin=-vmax, vmax=vmax, base=10))
    # the noise-rejected cells, drawn white over the grey "below threshold" background
    ys, xs = np.where(~ok)
    ax.scatter(xs, ys, s=2.0, marker="s", color="white", linewidths=0)
    _block_grid(ax, data, m_edges, p_edges)
    # No per-moment / per-parameter tick labels: on a matrix this size the individual
    # (sector, area) names are unreadable and they crowd out the only labels that are
    # actually read, the block names on the top and right margins (drawn by
    # `_block_grid`). The left margin instead carries the sector bands of the regional
    # sourcing block, which is the one place the row order has content the eye can use.
    ax.set_xticks([])
    ax.xaxis.set_ticks_position("bottom")
    _sector_axis(ax, data, sector_block)
    ax.set_xlabel("Parameter")
    ax.set_ylabel("Sector (regional sourcing shares)")

    kept = float(((A >= threshold) & ok).mean())
    ax.set_title(f"Elasticity Jacobian, $|\\varepsilon| \\geq {threshold:g}$ and "
                 f"$\\sigma/|\\varepsilon| < {noise_max:g}$ — "
                 f"{data['industry']}, " rf"$\hat{{\mu}}_{data['mu']}$"
                 f"   ({100 * kept:.1f}{pct()} of {A.size} entries shown)", pad=28)
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.04, pad=0.22)
    cbar.set_label(r"$\varepsilon = \partial \log m / \partial \log \theta$"
                   "   (symmetric log; grey = below threshold, white = too noisy)")
    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        fig.savefig(save_to, bbox_inches="tight")
    return ax


def plot_channel_elasticities(data, param_blocks=("Trade cost", "Comparative advantage",
                                                  "Variety count"),
                              threshold=IDENT_THRESHOLD, kind="elasticity",
                              noise_max=NOISE_MAX, stat="max", figsize=(11, 5),
                              save_to=None):
    """
    One panel per channel: the largest READABLE |elasticity| that channel produces in
    each moment block, on a log axis, with the threshold drawn in.

    This is the identification paragraph in one picture — alpha tall on the extensive
    margin, T tall on the sourcing shares, N_s tall on the zero-supplier share and
    EXACTLY zero (no bar, marked "0") everywhere else. Entries rejected for noise are
    excluded from the reduction, so a bar cannot be produced by a single unmeasurable
    outlier; a block with nothing readable left is marked "n/m".
    """
    E = np.abs(np.asarray(jacobian_matrix(data, kind), float))
    ok = _readable(data, kind, noise_max)
    m_names, p_names = data["moment_block_names"], data["param_block_names"]
    blocks, blocks_ok = _blocks_2d(data, E), _blocks_2d(data, ok)
    reduce = {"max": np.max, "mean": np.mean}[stat]
    stat_tex = {"max": r"\max", "mean": r"\mathrm{mean}"}[stat]

    wanted = [p for p in param_blocks if p in p_names]
    missing = [p for p in param_blocks if p not in p_names]
    if missing:
        raise ValueError(f"parameter block(s) {missing} absent — the Jacobian has "
                         f"{p_names}. Under --granular=true the run must have written "
                         "the variety-count columns (compute_jacobian append_N_s).")

    vals, unmeasured = {}, {}
    for p in wanted:
        j = p_names.index(p)
        v, nm = [], []
        for i in range(len(m_names)):
            sub, sub_ok = blocks[(i, j)], blocks_ok[(i, j)]
            if sub.size == 0 or not sub_ok.any():
                v.append(0.0)
                nm.append(sub.size > 0)
            else:
                v.append(float(reduce(sub[sub_ok])))
                nm.append(False)
        vals[p], unmeasured[p] = np.array(v), np.array(nm)

    pos = np.concatenate([v[v > 0] for v in vals.values()] or [np.array([threshold])])
    bottom = (pos.min() / 4) if pos.size else threshold / 10

    fig, axes = plt.subplots(1, len(wanted), figsize=figsize, sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(m_names))
    for ax, pname in zip(axes, wanted):
        v_p, nm_p = vals[pname], unmeasured[pname]
        ax.bar(x, v_p, color=[sim_color if v >= threshold else "0.75" for v in v_p])
        ax.set_yscale("log")
        ax.set_ylim(bottom=bottom)
        ax.axhline(threshold, color=reference_color, linestyle="--", linewidth=1.2)
        for i, (v, nm) in enumerate(zip(v_p, nm_p)):
            if v == 0:
                ax.text(i, bottom * 1.35, "n/m" if nm else "0", ha="center", va="bottom",
                        fontsize=fs(9), color=reference_color, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(m_names, rotation=45, ha="right", fontsize=fs(9))
        ax.set_title(pname, fontsize=fs(13), fontweight="bold")
        _despine(ax)
    axes[0].set_ylabel(rf"${stat_tex}_{{j,k}} |\varepsilon_{{jk}}|$ in block (readable only)")
    fig.suptitle(f"Where each channel acts — {data['industry']}, "
                 rf"$\hat{{\mu}}_{data['mu']}$"
                 f"   (dashed: threshold {threshold:g}; \"0\" = exactly zero, "
                 "\"n/m\" = nothing measured)", fontsize=fs(13))
    fig.tight_layout()
    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        fig.savefig(save_to, bbox_inches="tight")
    return axes


# ============================================================================
# Untargeted moment -- the distance elasticity of spatial comovement
# ============================================================================

# The (supplier x downstream region) panel, and the PPML that reads eta off it.
# The estimator is pyfixest.fepois — ppmlhdfe's algorithm — so nothing about Poisson
# pseudo-ML is re-implemented here.

# Empirical delta/gamma from Table 3 (the spatial comovement regression): point estimate
# and 95% CI. These are DATA, not model output, and they are not read from any file in
# this repository — override them here if the reduced-form estimates change.
EMPIRICAL_DELTA_OVER_GAMMA = {
    "auto": {"estimate": -0.109, "ci_lo": -0.122, "ci_hi": -0.096,
             "display_name": "Motor Vehicles"},
    "aero": {"estimate": -0.098, "ci_lo": -0.130, "ci_hi": -0.065,
             "display_name": "Aerospace"},
}


def build_a_ir_panel(data, min_distance_km=1.0, firm_key=("SIREN",),
                     replication=None):
    """
    The (supplier x downstream region) panel the PPML runs on, ZERO-FILLED.

    `suppliers.parquet` records only realised linkages (see the column glossary above).
    The regression needs the non-linkages too — a supplier that does not serve region r
    has a_ir = 0, and that extensive margin is half of what the moment measures — so
    every supplier is crossed with every downstream region and the missing cells are
    filled at 0.

    The sample is suppliers, i.e. varieties that win somewhere, which is the
    conditioning Sup_i = 1 of equation (3) and of (B36).

    Region codes are the model's own 1..R indices for BOTH `ze2010` (the supplier's
    cell) and `ze2010_downstream` (the buyer), so the distance is read straight out of
    `distances.npy` with no name round-trip.

    `firm_key` is what counts as one supplier. The default is the parquet's own `SIREN`,
    i.e. ONE VARIETY in one cell — the model's supplier. Passing `("ze2010", "A129")`
    pools every variety a (region, sector) produces into a single multi-variety firm,
    which is closer to what a SIREN is in the data and flattens the distance profile; the
    specification ladder below uses it to measure how much.

    `min_distance_km` is a floor, not a convention: the own-region entry of the distance
    matrix is the region's own internal distance, which is POSITIVE and differs across
    regions (a large commuting zone is further from itself than a small one), exactly as
    the model uses it. The floor only guards against a data file that happens to store a
    hard zero on the diagonal, which would send log d to -infinity.
    """
    sup = data.get("suppliers")
    if sup is None:
        raise FileNotFoundError(
            f"no suppliers.parquet under {data['folder']}/{data['step_dir']}/ (or it was "
            "unreadable — pyarrow missing?). main.jl writes one per estimate in its "
            "post-hoc block, after the estimation.")

    # The parquet holds `n_rep` INDEPENDENT realisations of the finite-variety economy.
    # Two consequences, and only the first is a matter of taste.
    #
    #   * `replication=b` takes ONE economy. The default pools them, which is legitimate
    #     for the POINT estimate — they are independent draws of the same economy and the
    #     elasticity is the same in each — but NOT for the clustered standard error,
    #     which would read `n_rep` economies as `n_rep` times the information and shrink
    #     the interval by roughly sqrt(n_rep). Use `untargeted_across_replications` when
    #     the interval is what is being reported.
    #   * An aggregating `firm_key` MUST carry `replication`, or "pool the varieties of a
    #     cell into one multi-variety firm" pools across ECONOMIES and hands the firm
    #     `n_rep` times as many varieties as the model gives it. That is handled below
    #     rather than left to the caller, because the failure is silent: the ladder rung
    #     would simply report a flatter profile.
    has_rep = "replication" in sup.columns
    if replication is not None:
        if not has_rep:
            raise KeyError("`replication` was asked for but suppliers.parquet has no "
                           "`replication` column — this tree predates the finite-variety "
                           "post-hoc economy.")
        sup = sup.loc[sup["replication"] == replication]
        if sup.empty:
            raise ValueError(f"no rows for replication {replication}.")
    n_rep_pooled = int(sup["replication"].nunique()) if has_rep else 1

    dist_path = data["input_folder"] / "distances.npy"
    if not dist_path.exists():
        raise FileNotFoundError(f"{dist_path} not found — needed for log Dist_{{r'r}}.")
    D = np.load(dist_path)
    R = data["R"]
    if D.shape[0] < R or D.shape[1] < R:
        raise ValueError(f"distances.npy is {D.shape}, too small for R = {R} regions.")
    D = D[:R, :R]

    keep = ["SIREN", "A129", "ze2010", "ze2010_downstream",
            "share", "downstream_purchase", "productivity", "sample_weight"]
    df = sup[keep + (["replication"] if has_rep else [])].copy()

    # The supplier identifier. Under the default it IS the parquet's SIREN; under an
    # aggregating `firm_key` the column is rebuilt from the key, and the euro flows of the
    # pooled varieties are added before the share is formed — which is what makes the
    # aggregated firm a genuine multi-variety supplier rather than an average of shares.
    if tuple(firm_key) != ("SIREN",):
        key = list(firm_key)
        if has_rep and "replication" not in key:
            key = key + ["replication"]        # never pool two economies into one firm
        df["SIREN"] = df[key].astype(str).agg("-".join, axis=1)

    # X_{i->r} = share x (mu Y_r); a_ir = X_{i->r} / sum_r X_{i->r}
    df["X_ir"] = df["share"] * df["downstream_purchase"]
    # how many real firms one simulated supplier stands for: the MEAN of its varieties'
    # draw weights, so that pooling varieties does not multiply the weight (under the
    # flat draw designs every variety carries 1/N_rho and a pooled firm still does)
    w_firm = df.groupby("SIREN", sort=True)["sample_weight"].mean()
    df = (df.groupby(["SIREN", "ze2010_downstream"], as_index=False)
            .agg(A129=("A129", "first"), ze2010=("ze2010", "first"),
                 X_ir=("X_ir", "sum"), productivity=("productivity", "mean")))
    tot = df.groupby("SIREN", sort=False)["X_ir"].transform("sum")
    df["a_ir"] = np.where(tot > 0, df["X_ir"] / tot, 0.0)

    # --------------------------------------------------------- zero-filling ---
    firms = df.groupby("SIREN", sort=True).agg(
        A129=("A129", "first"), ze2010=("ze2010", "first"),
        productivity=("productivity", "first")).reset_index()
    firms["sample_weight"] = firms["SIREN"].map(w_firm)
    downstream = np.sort(df["ze2010_downstream"].unique())

    panel = firms.merge(pd.DataFrame({"ze2010_downstream": downstream}), how="cross")
    panel = panel.merge(df[["SIREN", "ze2010_downstream", "a_ir"]],
                        on=["SIREN", "ze2010_downstream"], how="left")
    panel["a_ir"] = panel["a_ir"].fillna(0.0)

    # ------------------------------------------------------------ regressors --
    d = D[panel["ze2010"].to_numpy() - 1, panel["ze2010_downstream"].to_numpy() - 1]
    panel["distance"] = d
    panel["log_distance"] = np.log(np.maximum(d, min_distance_km))
    panel["log_productivity"] = np.log(np.maximum(panel["productivity"].to_numpy(), 1e-300))
    panel["served"] = (panel["a_ir"].to_numpy() > 0).astype(float)
    # alpha_{r, s(i)} : sector x downstream-region
    panel["fe_group"] = (panel["A129"].astype(str) + "-"
                         + panel["ze2010_downstream"].astype(str))
    panel["cluster"] = panel["ze2010"].astype(str)     # supplier's region, as in Table 3
    # what one row is: under the default a single variety, under an aggregating key a
    # bundle of them. `attach_lambda` refuses the latter, since z is then a mean of draws.
    panel.attrs["firm_key"] = tuple(firm_key)
    # how many economies this panel pools: 1 is one realisation, more means the interval
    # below is too tight by roughly its square root (see the note at the top)
    panel.attrs["n_replications"] = n_rep_pooled
    panel.attrs["replication"] = replication
    return panel



def fepois_fit(panel, y, rhs=("log_distance",), fe="fe_group", cluster="cluster",
               weights="sample_weight"):
    """
    One `pyfixest.fepois` call: E[y | .] = exp(fixed effect + rhs'beta).

    `fe` is passed to pyfixest as written, so "fe_group" absorbs the sector x
    downstream-region effect, "SIREN" absorbs a SUPPLIER effect, and "fe_group + SIREN"
    absorbs both.

    A three-line wrapper around the library, not an estimator: it writes the formula,
    passes the Monte-Carlo draw weights as analytic weights, clusters on the supplier's
    own region and lets pyfixest drop the fixed-effect groups that are separated (a
    group with no positive outcome contributes nothing to the Poisson score, exactly as
    in `ppmlhdfe`). Returns the fitted object, so `.coef()`, `.se()`, `.tidy()` and
    `.summary()` are all available at the call site.
    """
    fml = f"{y} ~ " + " + ".join(rhs) + (f" | {fe}" if fe else "")
    return pf.fepois(fml=fml, data=panel, weights=weights, weights_type="aweights",
                     vcov=None if cluster is None else {"CRV1": cluster},
                     separation_check=["fe"])


def delta_over_gamma_from_eta(eta, mean_log_d=None):
    """
    A constant elasticity `eta` expressed as the reduced form's `delta/gamma`.

    The spatial-comovement regression is LINEAR IN THE LEVEL of the exposure — gamma on
    the demand shock, delta on the shock interacted with log distance — so it fits
    E[a|d] = gamma + delta log d. Fitting a constant-elasticity profile A d^eta that way
    puts gamma at the extrapolated intercept log d = 0 (one kilometre, far outside any
    data), and linearising around the mean gives

        delta/gamma  ~=  eta / (1 + |eta| * mean(log d)).

    Two things follow. The map is strongly COMPRESSIVE, and it is BOUNDED: whatever the
    true elasticity, |delta/gamma| can never exceed 1/mean(log d) — 0.17 at the empirical
    mean of 5.8. So a model eta and a published delta/gamma cannot be compared as they
    stand; one has to be put on the other's scale, which is what this does.

    `mean_log_d` defaults to `EMPIRICAL_MEAN_LOG_D`, the mean in the ESTIMATION SAMPLE OF
    THE REDUCED FORM — the right one here, since the compression is a property of the
    regression that produced the published number, not of the simulated geography.
    """
    m = EMPIRICAL_MEAN_LOG_D if mean_log_d is None else mean_log_d
    return float(eta) / (1.0 + abs(float(eta)) * m)


def eta_from_delta_over_gamma(ratio, mean_log_d=None):
    """The inverse map: the constant elasticity a published `delta/gamma` implies."""
    m = EMPIRICAL_MEAN_LOG_D if mean_log_d is None else mean_log_d
    r = float(ratio)
    denom = 1.0 - abs(r) * m
    return r / denom if denom > 0 else -np.inf


# The fixed effect the moment is estimated under. ONE specification is the baseline,
# and it is the two-way one: a supplier effect AND a sector x buyer-region effect.
# Two reasons, and they agree. (a) It is the reduced form's own design: the
# spatial-comovement regression of Appendix B.5 carries a firm effect alpha_i and an
# FE_{r s(i), t}. (b) It is the object the appendix DIFFERENTIATES: delta_rs is the
# derivative of a_ir in log d holding the supplier (hence its other distances) and the
# destination fixed, which is exactly what absorbing both effects leaves.
# There is no second specification: every figure, every table and the decomposition run
# at this fixed effect, so the section reports one object throughout.
BASELINE_FE_LABEL = "supplier + sector x buyer region"
BASELINE_FE = "fe_group + SIREN"


def model_nu_s(data):
    """
    The within-sector CES elasticity: from stats.csv when it carries one, else
    NU_S_DEFAULT. Same rule as `model_theta` for theta, and the same reason — it is a
    calibrated number that the run does not write out.

    Raises if the run carries a sector-specific nu_s: the granular decomposition below is
    written for a scalar, and averaging elasticities across sectors would be silent.
    """
    v = _read_named_value(data["coefs"], "nu_s") if data.get("coefs") is not None else None
    if v is None or not np.isfinite(v):
        return float(NU_S_DEFAULT)
    v = np.asarray(v, dtype=float).ravel()
    if v.size != 1:
        raise ValueError(f"stats.csv carries {v.size} values for nu_s; the granular "
                         "decomposition is written for a scalar elasticity.")
    return float(v[0])


def theta_alpha(data):
    """
    `theta * alpha_hat` for this run, read off the LOADED estimate — the scale every term
    of the moment is measured in (Appendix B.5: both margins are theta*alpha times a
    dimensionless number, so |eta_int| <= theta*alpha and |eta_ext| = theta*alpha E(Lambda)).

    `best_params` is the raw theta vector, layout
        [Omega_L(1) | Omega_s(S) | A(R_d) | alpha(N_TAU) | T(active (s,AA), s-major)],
    so alpha is the N_TAU-long slice after the A block; the length is asserted against the
    rebuilt layout, exactly as `unpack_estimated_T` does in the comparative-advantage
    section. theta follows the same rule as `model_theta` there: stats.csv when it carries
    one, else THETA_DEFAULT. Defined here so the section runs on its own.
    """
    bp = data.get("best_params")
    if bp is None:
        raise FileNotFoundError(f"no best_params in {data['folder']}/{data['step_dir']}/.")
    bp = np.asarray(bp, dtype=float).ravel()
    S, n_tau = data["S"], data["n_tau"]
    R_d = len(data["aa_names"])
    expected = 1 + S + R_d + n_tau + int(data["AA_ACTIVE"].sum())
    if bp.size != expected:
        raise ValueError(f"best_params has {bp.size} entries, layout rebuilt here is "
                         f"{expected} — the run's flags and the ones passed here disagree.")
    if n_tau != 1:
        raise ValueError(f"N_TAU = {n_tau}: tau is binned, so there is no single distance "
                         "elasticity. Re-run the reporting on an n_tau = 1 fit.")
    v = _read_named_value(data["coefs"], "theta") if data.get("coefs") is not None else None
    theta = float(v) if v is not None and np.isfinite(v) and v > 0 else THETA_DEFAULT
    return float(theta) * float(bp[1 + S + R_d])


def estimate_untargeted_moment(data, controls=(), cluster="cluster", panel=None,
                               fe=BASELINE_FE, verbose=True):
    """
    The measured moment: PPML of E[a_ir] = exp(alpha_i + alpha_{r,s(i)} + eta log Dist).

    One specification — a supplier effect and a sector x downstream-region effect — which
    is the reduced form's own design and the variation Appendix B.5 differentiates: within
    one supplier (its market access held fixed) and within one buyer cell (Y_r and P_sr
    absorbed).

    `controls` adds columns of the panel to the right-hand side (the paper's specification
    has none). `cluster=None` gives the classical Poisson variance.
    """
    panel = build_a_ir_panel(data) if panel is None else panel
    cols = ["log_distance"] + list(controls)
    fit = fepois_fit(panel, "a_ir", rhs=cols, fe=fe, cluster=cluster)
    eta = float(fit.coef()["log_distance"])
    se = float(fit.se()["log_distance"])
    emp = EMPIRICAL_DELTA_OVER_GAMMA.get(data["industry"], {})
    try:
        ta = theta_alpha(data)
    except Exception:
        ta = np.nan

    out = {"industry": data["industry"], "mu": data["mu"], "spec": "+".join(cols),
           "fe": fe, "fe_label": BASELINE_FE_LABEL, "theta_alpha": ta,
           "eta": eta, "se": se, "t": eta / se if se > 0 else np.nan,
           "ci_lo": eta - 1.96 * se, "ci_hi": eta + 1.96 * se,
           # the same estimate on the reduced form's own scale — the one that is reported
           "dg": delta_over_gamma_from_eta(eta),
           "dg_lo": delta_over_gamma_from_eta(eta - 1.96 * se),
           "dg_hi": delta_over_gamma_from_eta(eta + 1.96 * se),
           "n_obs": int(getattr(fit, "_N", len(panel))),
           "n_firms": int(panel["SIREN"].nunique()),
           "share_zero": float((panel["a_ir"].to_numpy() == 0).mean()),
           "emp_estimate": emp.get("estimate"), "emp_ci_lo": emp.get("ci_lo"),
           "emp_ci_hi": emp.get("ci_hi"), "fit": fit, "panel": panel}

    if verbose:
        print(f"[{data['industry']}]  {len(panel):,} supplier x downstream-region cells, "
              f"{out['n_firms']:,} suppliers, {100 * out['share_zero']:.1f}% zeros, "
              f"from {data['suppliers_path']}")
        print(f"  measured eta = {eta:+.4f} ({se:.4f})   ->  delta/gamma = {out['dg']:+.4f} "
              f"[{out['dg_lo']:+.4f}, {out['dg_hi']:+.4f}]   "
              f"(mean log d = {EMPIRICAL_MEAN_LOG_D}, theta*alpha = {ta:.4f})")
        if out["emp_estimate"] is not None:
            print(f"  DATA delta/gamma = {out['emp_estimate']:+.4f}  "
                  f"[{out['emp_ci_lo']:+.4f}, {out['emp_ci_hi']:+.4f}]")
    return out


def untargeted_across_replications(data, fe=BASELINE_FE, controls=(), cluster="cluster",
                                   max_replications=None, verbose=True):
    """
    The untargeted moment estimated ONCE PER REALISATION of the finite-variety economy.

    With `N_s` finite the network is a random object, so `eta_hat` is too. Pooling every
    realisation into one PPML gives a fine point estimate and an interval that is too
    tight by roughly `sqrt(n_rep)` — the clustered variance counts independent economies
    as extra information about one economy. The honest interval is the dispersion of
    `eta_hat` ACROSS realisations, which is what this returns; the pooled fit is reported
    beside it so the size of that understatement is visible rather than argued.

    A parquet with no `replication` column is one economy and comes back as one row.
    """
    sup = data.get("suppliers")
    if sup is None:
        raise FileNotFoundError("no suppliers.parquet — see build_a_ir_panel.")
    reps = (sorted(sup["replication"].unique()) if "replication" in sup.columns else [None])
    if max_replications is not None:
        reps = reps[:max_replications]

    rows = []
    for b in reps:
        panel = build_a_ir_panel(data, replication=b)
        r = estimate_untargeted_moment(data, controls=controls, cluster=cluster,
                                       panel=panel, fe=fe, verbose=False)
        rows.append({"replication": b, "eta": r["eta"], "se_within": r["se"],
                     "dg": r["dg"], "n_firms": r["n_firms"],
                     "share_zero": r["share_zero"]})
    per = pd.DataFrame(rows).set_index("replication")

    pooled = estimate_untargeted_moment(data, controls=controls, cluster=cluster,
                                        fe=fe, verbose=False)
    per.attrs["pooled_eta"] = pooled["eta"]
    per.attrs["pooled_se"] = pooled["se"]
    per.attrs["sd_across"] = float(per["eta"].std(ddof=1)) if len(per) > 1 else np.nan
    per.attrs["mean_eta"] = float(per["eta"].mean())

    if verbose:
        sd = per.attrs["sd_across"]
        head = f"[{data['industry']}]  eta_hat over {len(per)} realisation(s): "
        if len(per) > 1:
            print(head + f"mean {per.attrs['mean_eta']:.4f}, "
                         f"sd across draws {sd:.4f}")
            print(f"              pooled fit {pooled['eta']:.4f} (se {pooled['se']:.4f})"
                  f" — that se reads {len(per)} economies as information about one, so "
                  f"the sd above is the interval to report")
        else:
            print(head + f"{per.attrs['mean_eta']:.4f} (se {pooled['se']:.4f})")
    return per


def panel_facts(panel):
    """
    What the panel says about the geography, with no regression in sight.

      `serving_rate`         share of (supplier, region) cells that are served;
      `openness_weighted`    1 - sum(w a^2)/sum(w a): the SHARE-WEIGHTED portfolio openness,
                             which is the average the moment's weight takes (large shares
                             carry large weight);
      `buyers_per_supplier`  how many of the `n_buyers` downstream regions a supplier serves.

    Openness is a property of the geography of DOWNSTREAM demand — high when a supplier's
    sales are spread over many buyers, so that losing one to distance reallocates onto the
    others — and it is the object the substitution term of the granular decomposition needs.
    """
    a = panel["a_ir"].to_numpy(float)
    w = panel["sample_weight"].to_numpy(float)
    served = a > 0
    denom = float((w * a).sum())
    n_sup = int(panel["SIREN"].nunique())
    return {"serving_rate": float(np.average(served.astype(float), weights=w)),
            "openness_weighted": float(1.0 - (w * a * a).sum() / denom) if denom > 0 else np.nan,
            "buyers_per_supplier": float(served.sum() / n_sup) if n_sup else np.nan,
            "n_buyers": int(panel["ze2010_downstream"].nunique())}


# The untargeted-moment figures and the summary table.

def untargeted_summary(results):
    """
    ONE row per industry — the baseline estimate — with the fixed effect it was run under
    as a column, the moment on the reduced form's own delta/gamma scale, and the published
    number beside it.

    The elasticity is deliberately NOT a column: eta and delta/gamma are two readings of
    one profile on two scales, only the second is comparable to Table 3, and printing both
    invites the comparison of a number with itself measured differently. `theta*alpha` is
    kept, because it is the scale the structural moment is read in (see `structural_eta`).
    """
    rows = []
    for r in results:
        emp = EMPIRICAL_DELTA_OVER_GAMMA.get(r["industry"], {})
        rows.append({
            "industry": emp.get("display_name", r["industry"]),
            "fixed effect": r["fe_label"],
            "delta/gamma (model)": r["dg"],
            "95% CI (model, delta/gamma)": f"[{r['dg_lo']:+.3f}, {r['dg_hi']:+.3f}]",
            "delta/gamma (data)": r["emp_estimate"],
            "95% CI (data)": (f"[{r['emp_ci_lo']:+.3f}, {r['emp_ci_hi']:+.3f}]"
                              if r["emp_ci_lo"] is not None else "---"),
            "inside data CI": (None if r["emp_ci_lo"] is None
                               else bool(r["emp_ci_lo"] <= r["dg"] <= r["emp_ci_hi"])),
            "theta*alpha": r.get("theta_alpha", np.nan),
            "suppliers": r["n_firms"], "cells": r["n_obs"],
        })
    return pd.DataFrame(rows).set_index("industry")


def plot_untargeted_moment(results, figsize=None, save_to=None):
    """
    The model against the published reduced form, one group per industry, on the reduced
    form's own delta/gamma scale — the only scale the section reports, since it is the one
    Table 3 is on.

    One model marker per industry with its 95% interval, and a square for the published
    estimate with its own. The dotted line is the ceiling -1/mean(log d): no level-linear
    ratio can lie beyond it, so how close the markers sit to it says how much room the
    scale has left.
    """
    fig, ax = plt.subplots(figsize=figsize or get_figsize(wf=0.95, hf=0.65))
    x = np.arange(len(results))

    for i, r in enumerate(results):
        v, lo, hi = r["dg"], r["dg_lo"], r["dg_hi"]
        ax.errorbar(x[i] - 0.12, v, yerr=[[v - lo], [hi - v]], fmt="o", color=sim_color,
                    capsize=4, markersize=7,
                    label="Model (baseline)" if i == 0 else None)
        if r["emp_estimate"] is not None:
            ax.errorbar(x[i] + 0.22, r["emp_estimate"],
                        yerr=[[r["emp_estimate"] - r["emp_ci_lo"]],
                              [r["emp_ci_hi"] - r["emp_estimate"]]],
                        fmt="s", color=reference_color, capsize=4, markersize=8,
                        label="Data $\\delta/\\gamma$ (Table 3)" if i == 0 else None)

    ceil = -1.0 / EMPIRICAL_MEAN_LOG_D
    ax.axhline(ceil, color="0.4", linestyle=":", linewidth=1.1)
    ax.annotate(rf"ceiling $-1/\overline{{\log d}}$ = {ceil:.3f}", (0.01, ceil),
                xycoords=("axes fraction", "data"), fontsize=fs(8), color="0.35", va="bottom")
    ax.axhline(0.0, color="0.6", linewidth=0.8)
    ax.set_xlim(-0.5, len(results) - 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([EMPIRICAL_DELTA_OVER_GAMMA.get(r["industry"], {})
                        .get("display_name", r["industry"]) for r in results])
    ax.set_ylabel(r"$\delta/\gamma$ (reduced-form scale)")
    ax.set_title("Untargeted moment: model against the spatial-comovement regression",
                 fontsize=fs(10))
    ax.legend(frameon=False, loc="best", fontsize=fs(8))
    _despine(ax)
    fig.tight_layout()
    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        fig.savefig(save_to, bbox_inches="tight")
    return ax


# The STRUCTURAL moment: the granular derivative of Appendix B.5, computed in closed form
# on the simulated economy. Nothing here is estimated — every object entering
#     dlog a_ir/dlog d = -alpha [ theta * Lambda_ir + (nu_s - 1)(1 - a_ir) ]
# is observed, so the margin split is exact by construction and there is no identity that
# can fail to close. What IS estimated, and compared with it, is the PPML above.


def _estimated_T(data):
    """`unpack_estimated_T` with a legible error when the section is run on its own.

    The comparative-advantage definitions cell sits BELOW this one in the notebook, so a
    standalone run of the untargeted section has to execute it first. Everything else here
    is self-contained.
    """
    try:
        return unpack_estimated_T(data)
    except NameError as e:
        raise NameError(
            "`unpack_estimated_T` is defined in the comparative-advantage section, which "
            "is below this one — run its definitions cell first (the full run does)."
        ) from e


def attach_lambda(data, panel, min_distance_km=1.0):
    """
    The pair-level surprisal of winning,

        Lambda_ir = Phi_{-r', r s} (w_{r's} tau_{r'r})^theta z_i^{-theta},
        tau_{r'r} = d_{r'r}^alpha,
        Phi_{-r', r s} = Phi_{rs} - T_{a(r') s} (w_{r's} tau_{r'r})^{-theta},

    added to the panel as `lambda_ir`. This is the only genuinely new computation of the
    section, and it is the one that makes the structural moment computable rather than
    estimable: Lambda is the object the extensive margin of the appendix is written in.

    Where each piece comes from. `T_{as}` is `unpack_estimated_T` (the comparative-advantage
    section), mapped region -> area with `aa_of_ze`. `alpha` and `theta` come from
    `theta_alpha`'s own layout read of `best_params`. Distances are `distances.npy` with the
    SAME floor `build_a_ir_panel` uses, so the two objects live on one distance matrix.
    `z_i` is the panel's `productivity`.

    `w_{r's}` is the upstream cost shifter the model competes on — `w_rs.npy`. Note that
    `load_parameters.jl:29` normalises it to one wherever it is positive, so in the fitted
    model it is identically one; the file is read anyway and the same normalisation applied,
    so a future run that stops normalising is picked up rather than silently ignored. (The
    loader's `regional_wage` is the DOWNSTREAM wage and is a different object — it must not
    be used here.)

    KNOWN APPROXIMATION. `Phi_{rs}` is summed over the modelled domestic cells only: the
    foreign origin enters the model's own price index but is not recoverable from the
    artefacts loaded here. Excluding it UNDERSTATES Phi, hence understates Lambda, hence
    understates |eta_ext^struct| — the structural moment reported below is a lower bound in
    that respect.

    Raises when the panel was built with an aggregating `firm_key`: a pooled firm's
    `productivity` is a mean of draws, and Lambda is not defined pair by pair for it.
    """
    key = tuple(panel.attrs.get("firm_key", ("SIREN",)))
    if key != ("SIREN",):
        raise ValueError(
            f"the panel was built with firm_key={key}: `productivity` is then a mean over "
            "the pooled varieties and Lambda is not defined pair by pair. Build the panel "
            "with the default key for the structural moment.")

    est = _estimated_T(data)                       # T (S, n_AA), alpha, theta
    if est["alpha"].size != 1:
        raise ValueError(f"N_TAU = {est['alpha'].size}: tau is binned, so there is no "
                         "single distance elasticity to raise to the power theta.")
    alpha, theta = float(est["alpha"][0]), float(est["theta"])
    S, R = data["S"], data["R"]
    aa_of_ze = np.asarray(data["aa_of_ze"]).ravel()          # region -> area, 0-based
    CELL_MASK = np.asarray(data["CELL_MASK"], dtype=bool)    # (S, R) simulated cells

    D = np.load(data["input_folder"] / "distances.npy")[:R, :R]
    d = np.maximum(D, min_distance_km)
    tau = d ** alpha                                          # (R, R) origin x buyer

    w_path = data["input_folder"] / "w_rs.npy"
    if w_path.exists():
        w_rs = np.asarray(np.load(w_path), dtype=float).ravel()[:R]
        w_rs = np.where(w_rs > 0, 1.0, 1.0)                   # the model's own normalisation
    else:
        w_rs = np.ones(R)
        print("  note: w_rs.npy not found — the upstream cost shifter is taken as one, "
              "which is what load_parameters.jl normalises it to anyway.")

    wtau = w_rs[:, None] * tau                                # (R, R): (w_{r'} tau_{r'r})
    # Phi_{rs} = sum over the sector's simulated cells; T is at the AREA level
    T_ze = np.zeros((S, R))
    for s in range(S):
        T_ze[s] = est["T"][s, aa_of_ze]
    contrib = np.where(CELL_MASK[:, :, None], (T_ze[:, :, None]
                                               * wtau[None, :, :] ** (-theta)), 0.0)
    Phi = contrib.sum(axis=1)                                 # (S, R) over origins
    Phi_minus = Phi[:, None, :] - contrib                     # (S, R_origin, R_buyer)

    s_idx = panel["A129"].to_numpy() - 1
    o_idx = panel["ze2010"].to_numpy() - 1
    b_idx = panel["ze2010_downstream"].to_numpy() - 1
    z = np.maximum(panel["productivity"].to_numpy(float), 1e-300)
    out = panel.copy()
    out["lambda_ir"] = (np.maximum(Phi_minus[s_idx, o_idx, b_idx], 0.0)
                        * wtau[o_idx, b_idx] ** theta * z ** (-theta))
    out.attrs.update(panel.attrs)
    out.attrs.update({"alpha": alpha, "theta": theta})
    return out


def _structural_weights(panel):
    """
    The appendix's weight w = a * rho/rho_tilde, as it is realised in the panel.

    `rho/rho_tilde` is the probability of OBSERVING the pair given that the firm is a
    supplier — and the panel is a realised economy conditioned on exactly that (a variety
    is in `suppliers.parquet` only if it won somewhere), so the empirical frequency of the
    pair already carries it: each realised (i, r) row stands for `sample_weight` firms.
    `sample_weight` is flat under every draw design the model uses (`FlatWeights`, 1/N_rho
    in `model_CP.jl`), so the weight reduces to `a` up to a constant — but it is written out
    so an importance-sampled run would still be weighted correctly.
    """
    a = panel["a_ir"].to_numpy(float)
    w = a * panel["sample_weight"].to_numpy(float)
    assert np.all(w[a <= 0] == 0.0), "unserved cells must carry zero weight"
    return w


def structural_eta(data, panel, verbose=True):
    """
    The theoretical moment, from the granular derivation, with no regression.

        dlog a_ir / dlog d = -alpha [ theta * Lambda_ir + (nu_s - 1)(1 - a_ir) ]

    averaged with w = a * rho/rho_tilde over the SERVED cells (the derivative is conditional
    on winning; unserved cells carry zero weight, which is asserted rather than assumed).

    The two shares sum to one by construction — that is the point of computing rather than
    estimating them: there is no identity left to fail.
    """
    if "lambda_ir" not in panel.columns:
        raise KeyError("run `attach_lambda(data, panel)` first — the structural moment "
                       "needs the pair-level surprisal.")
    est = _estimated_T(data)
    alpha, theta = float(est["alpha"][0]), float(est["theta"])
    nu_s = model_nu_s(data)

    w = _structural_weights(panel)
    lam = panel["lambda_ir"].to_numpy(float)
    a = panel["a_ir"].to_numpy(float)
    W = w.sum()
    E_lambda = float((w * lam).sum() / W)
    E_open = float((w * (1.0 - a)).sum() / W)

    eta_ext = -theta * alpha * E_lambda
    eta_int = -(nu_s - 1.0) * alpha * E_open
    eta = eta_ext + eta_int
    out = {"industry": data["industry"], "theta": theta, "alpha": alpha, "nu_s": nu_s,
           "theta_alpha": theta * alpha, "eta_struct": eta,
           "eta_struct_ext": eta_ext, "eta_struct_int": eta_int,
           "E_lambda": E_lambda, "E_open": E_open,
           "share_ext": eta_ext / eta if eta != 0 else np.nan,
           "share_int": eta_int / eta if eta != 0 else np.nan,
           "dg_struct": delta_over_gamma_from_eta(eta)}
    out.update({f"panel_{k}": v for k, v in panel_facts(panel).items()})
    if verbose:
        print(f"  structural eta = {eta:+.4f}  =  extensive {eta_ext:+.4f} "
              f"({100 * out['share_ext']:.0f}%) + substitution {eta_int:+.4f} "
              f"({100 * out['share_int']:.0f}%)   ->  delta/gamma = {out['dg_struct']:+.4f}")
        print(f"    E_w(Lambda) = {E_lambda:.3f},  E_w(1-a) = {E_open:.3f},  "
              f"theta*alpha = {theta * alpha:.4f},  (nu_s-1)/theta = "
              f"{(nu_s - 1.0) / theta:.3f}")
    return out


def validate_structural(data, panel, sres, verbose=True):
    """
    Three numerical checks on `attach_lambda`, each comparing a closed form against a
    MEASURED FREQUENCY — never against a fitted coefficient, so nothing here can be
    confirmed by the same specification error twice.

      1. E(Lambda | win) = 1 - gamma. gamma is measured as the fraction of a cell's
         varieties that win in r, straight off the panel;
      2. Lambda is exponential across draws, so Pr(win) = E(e^{-Lambda}). The mean of
         exp(-Lambda) is therefore the UNCONDITIONAL win probability, while the panel's
         serving rate is conditional on Sup_i = 1 — the file keeps only varieties that won
         somewhere, which selects on high z. The ratio should sit somewhat BELOW one, and
         a ratio near or above one is the signal that Lambda is too small (a Phi built on
         too few competitors, or the own term subtracted twice);
      3. the substitution scale: eta_int/(theta*alpha) must equal (nu_s-1)/theta times
         E_w(1-a), by construction.

    A failure here is a coding error in `attach_lambda`, not an economic finding.
    """
    lam = panel["lambda_ir"].to_numpy(float)
    served = panel["a_ir"].to_numpy(float) > 0
    w = _structural_weights(panel)

    # 1. gamma measured cell by cell: the share of (cell, sector) varieties winning in r
    g = (panel.assign(served=served.astype(float))
              .groupby(["ze2010", "A129", "ze2010_downstream"])["served"].mean())
    gamma_pair = panel.set_index(["ze2010", "A129", "ze2010_downstream"]).index.map(g)
    gamma_w = float((w * np.asarray(gamma_pair, dtype=float)).sum() / w.sum())
    e_lambda_win = float((w * lam).sum() / w.sum())

    # 2. exponentiality
    p_hat = float(np.mean(np.exp(-lam)))
    serving = float(served.mean())

    # 3. the substitution scale
    scale = (sres["nu_s"] - 1.0) / sres["theta"]
    implied = abs(sres["eta_struct_int"]) / sres["theta_alpha"]

    df = pd.DataFrame([
        {"check": "E(Lambda | win) = 1 - gamma", "closed form": e_lambda_win,
         "measured": 1.0 - gamma_w, "ratio": e_lambda_win / max(1.0 - gamma_w, 1e-12)},
        {"check": "E(exp(-Lambda)) vs serving rate (ratio < 1: see docstring)",
         "closed form": p_hat, "measured": serving,
         "ratio": p_hat / max(serving, 1e-12)},
        {"check": "eta_int/(theta*alpha) = (nu_s-1)/theta * E(1-a)", "closed form": implied,
         "measured": scale * sres["E_open"],
         "ratio": implied / max(scale * sres["E_open"], 1e-12)},
    ]).set_index("check")
    if verbose:
        print(df.round(4).to_string())
        print("    a ratio far from one is a wiring error in attach_lambda "
              "(T, w, Phi or the distance floor), not a finding about the economy")
    return df


def decompose_gap(data, panel, res, sres, verbose=True):
    """
    R = eta_measured - eta_structural, split into the parts that can be identified.

      (a) WEIGHTING. The appendix averages the pair-level derivative with w = a rho/rho~;
          PPML's quasi-score weights each cell by its FITTED mean. The same derivative
          averaged under the fitted weights, minus the same derivative under w, is the
          weighting wedge — same object, two averages;
      (b) FUNCTIONAL FORM. The measured moment is a constant-elasticity PPML; the published
          ratio is a level-linear fit. `linear_delta_over_gamma` refits the panel in the
          reduced form's own shape, and the difference on the delta/gamma scale is mapped
          back through `eta_from_delta_over_gamma`;
      (c) COMPOSITION, the remainder. The derivative holds the origin's primitives fixed
          while moving tau; the cross-section compares DIFFERENT origin regions, whose
          T_{a(r')s}, w_{r's} and Phi_{rs} covary with distance in a finite geography.

    The sign of R is reported first and explicitly: survivor selection among distant links
    would FLATTEN the measured profile, so a negative R (measured steeper) rules that story
    out and points at (a) and (b).
    """
    est = _estimated_T(data)
    alpha, theta = float(est["alpha"][0]), float(est["theta"])
    nu_s = model_nu_s(data)
    lam = panel["lambda_ir"].to_numpy(float)
    a = panel["a_ir"].to_numpy(float)
    deriv = -alpha * (theta * lam + (nu_s - 1.0) * (1.0 - a))

    w = _structural_weights(panel)
    under_w = float((w * deriv).sum() / w.sum())

    # (a) the same derivative under PPML's own weights
    try:
        mu = np.asarray(res["fit"].predict(), dtype=float).ravel()
        if mu.size != len(panel):                       # separated groups were dropped
            mu = None
    except Exception:
        mu = None
    if mu is None:
        weighting = np.nan
        under_mu = np.nan
    else:
        mu = np.maximum(mu, 0.0)
        under_mu = float((mu * deriv).sum() / mu.sum())
        weighting = under_mu - under_w

    # (b) the functional form: the same panel in the reduced form's shape
    lin = linear_delta_over_gamma(panel)
    eta_linear = eta_from_delta_over_gamma(lin["delta_over_gamma"])
    functional = res["eta"] - eta_linear

    R = res["eta"] - sres["eta_struct"]
    composition = R - (0.0 if not np.isfinite(weighting) else weighting) - functional
    out = {"industry": data["industry"], "eta": res["eta"],
           "eta_struct": sres["eta_struct"], "R": R, "R_weighting": weighting,
           "R_functional": functional, "R_composition": composition,
           "deriv_under_w": under_w, "deriv_under_fitted": under_mu,
           "eta_linear_equivalent": eta_linear,
           "dg_linear": lin["delta_over_gamma"], "dg_ppml": res["dg"]}
    if verbose:
        sign = "measured STEEPER than structural" if R < 0 else "measured FLATTER"
        print(f"  gap R = {R:+.4f}  ({sign})")
        print(f"    weighting  {weighting:+.4f}   (derivative under fitted weights "
              f"{under_mu:+.4f} vs under a*rho/rho~ {under_w:+.4f})")
        print(f"    functional {functional:+.4f}   (level-linear delta/gamma "
              f"{lin['delta_over_gamma']:+.4f} vs PPML's {res['dg']:+.4f})")
        print(f"    composition {composition:+.4f}   (origin-level primitives covarying "
              "with distance — the term the derivative does not take)")
    return out


def within_area_check(data, panel, gap, verbose=True):
    """
    A bound on the composition term. Restricted to pairs whose ORIGIN lies in the same
    attraction area as ... itself — i.e. to sectors x areas containing several regions —
    comparative advantage T_{a(r')s} is common across the origins being compared, so the
    part of the composition residual that is cross-origin heterogeneity in T cannot operate.

    If the residual shrinks on this subsample, it is comparative advantage covarying with
    distance; if it does not, it is the rest (wages, Phi, the geometry of the area itself).

    The subsample is small wherever areas are single-region, which is why the retained pair
    count is reported beside the numbers: a noisy estimate is labelled, not quoted.
    """
    aa_of_ze = np.asarray(data["aa_of_ze"]).ravel()
    area = aa_of_ze[panel["ze2010"].to_numpy() - 1]
    key = pd.Series(list(zip(panel["A129"].to_numpy(), area)))
    # keep (sector, area) blocks that hold more than one origin region
    n_reg = (pd.DataFrame({"k": key, "r": panel["ze2010"].to_numpy()})
             .groupby("k")["r"].nunique())
    keep = key.map(n_reg).to_numpy() > 1
    sub = panel[keep].copy()
    sub.attrs.update(panel.attrs)
    n_pairs, n_sup = len(sub), sub["SIREN"].nunique() if len(sub) else 0
    if n_pairs < 500 or n_sup < 20:
        if verbose:
            print(f"  within-area check: only {n_pairs} pairs / {n_sup} suppliers survive "
                  "— too few to quote, the areas are near single-region here")
        return {"industry": data["industry"], "n_pairs": int(n_pairs),
                "n_suppliers": int(n_sup), "eta": np.nan, "eta_struct": np.nan,
                "R": np.nan, "quotable": False}
    r_sub = estimate_untargeted_moment(data, panel=sub, verbose=False)
    s_sub = structural_eta(data, sub, verbose=False)
    out = {"industry": data["industry"], "n_pairs": int(n_pairs),
           "n_suppliers": int(n_sup), "eta": r_sub["eta"],
           "eta_struct": s_sub["eta_struct"],
           "R": r_sub["eta"] - s_sub["eta_struct"], "quotable": True}
    if verbose:
        print(f"  within-area check ({n_pairs:,} pairs, {n_sup:,} suppliers): "
              f"eta {out['eta']:+.4f} vs structural {out['eta_struct']:+.4f}, "
              f"R = {out['R']:+.4f}   (against R = {gap['R']:+.4f} on the full "
              "sample — a smaller |R| here means comparative advantage covarying with "
              "distance was carrying it)")
    return out


def structural_summary(structural_results):
    """One row per industry: the theoretical moment, the measured one, and their gap."""
    rows = []
    for r in structural_results:
        emp = EMPIRICAL_DELTA_OVER_GAMMA.get(r["industry"], {})
        rows.append({
            "industry": emp.get("display_name", r["industry"]),
            "theta*alpha": r["theta_alpha"], "nu_s": r["nu_s"],
            "eta struct": r["eta_struct"], "share ext": r["share_ext"],
            "share int": r["share_int"], "E_w(Lambda)": r["E_lambda"],
            "E_w(1-a)": r["E_open"], "eta measured": r["eta"],
            "R": r["R"], "R weighting": r["R_weighting"],
            "R functional": r["R_functional"], "R composition": r["R_composition"],
            "serving rate": r["panel_serving_rate"],
            "buyers per supplier": r["panel_buyers_per_supplier"],
        })
    return pd.DataFrame(rows).set_index("industry")


def plot_structural_vs_measured(structural_results, figsize=None, save_to=None):
    """
    Per industry, two bars on the delta/gamma scale: the THEORETICAL moment, split into its
    extensive and substitution parts (they sum exactly), and the MEASURED one. The distance
    between the bar tops is R, annotated with its three parts.
    """
    fig, ax = plt.subplots(figsize=figsize or get_figsize(wf=0.95, hf=0.62))
    x = np.arange(len(structural_results))
    wdt = 0.34
    for i, r in enumerate(structural_results):
        ext = delta_over_gamma_from_eta(r["eta_struct_ext"])
        tot = delta_over_gamma_from_eta(r["eta_struct"])
        ax.bar(x[i] - wdt / 2, ext, wdt, color=sim_color, alpha=0.9,
               label="structural: extensive" if i == 0 else None)
        ax.bar(x[i] - wdt / 2, tot - ext, wdt, bottom=ext, color=(0.30, 0.55, 0.40),
               alpha=0.85, label="structural: substitution" if i == 0 else None)
        ax.bar(x[i] + wdt / 2, delta_over_gamma_from_eta(r["eta"]), wdt,
               color=reference_color, alpha=0.8,
               label="measured (PPML)" if i == 0 else None)
        ax.annotate(rf"$\theta\alpha$ = {r['theta_alpha']:.3f}", (x[i], 0.004),
                    ha="center", fontsize=fs(8), color="0.3")
        ax.annotate(f"R: weight {r['R_weighting']:+.2f}, form {r['R_functional']:+.2f}, "
                    f"comp {r['R_composition']:+.2f}",
                    (x[i], delta_over_gamma_from_eta(r["eta"])), xytext=(0, -12),
                    textcoords="offset points", ha="center", fontsize=fs(7), color="0.35")
    ax.axhline(0.0, color="0.6", linewidth=0.8)
    ceil = -1.0 / EMPIRICAL_MEAN_LOG_D
    ax.axhline(ceil, color="0.4", linestyle=":", linewidth=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels([EMPIRICAL_DELTA_OVER_GAMMA.get(r["industry"], {})
                        .get("display_name", r["industry"])
                        for r in structural_results])
    ax.set_ylabel(r"$\delta/\gamma$ (reduced-form scale)")
    ax.set_title("The moment computed against the moment estimated", fontsize=fs(10))
    ax.legend(frameon=False, fontsize=fs(8), loc="lower right")
    _despine(ax)
    fig.tight_layout()
    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        fig.savefig(save_to, bbox_inches="tight")
    return ax


# The specification ladder: the same panel and the same fixed effect, one difference at a
# time, all read on the reduced form's own delta/gamma scale — the only scale Table 3 is
# on, and the only one the section reports. See the markdown above for what each rung is.


def linear_delta_over_gamma(panel, weight="sample_weight"):
    """
    The model's `a_ir` fitted in the EMPIRICAL functional form: level-linear in log d,
    E[a | d] = gamma + delta log d, which is what a shock-response regression with a
    distance interaction estimates (gamma on the shock, delta on shock x log d).

    This is the direct version of `delta_over_gamma_from_eta`: instead of mapping the
    PPML elasticity through the closed form, it refits the simulated panel in the
    reduced form's own shape. The two answer the same question and should agree up to
    the curvature of the profile and to the difference between the SIMULATED distance
    distribution (used here) and the EMPIRICAL one (used by the closed-form map).

    Also reports the elasticity that same fit implies AT THE MEAN DISTANCE,
    `delta / (gamma + delta * mean log d)`, which is what the ratio would be if the
    empirical regression centred log d.
    """
    a = panel["a_ir"].to_numpy(float)
    ld = panel["log_distance"].to_numpy(float)
    w = np.ones_like(a) if weight is None else panel[weight].to_numpy(float)
    X = np.column_stack([np.ones_like(ld), ld])
    WX = X * w[:, None]
    gamma, delta = np.linalg.solve(X.T @ WX, WX.T @ a)
    m = float(np.average(ld, weights=w))
    return {"gamma": float(gamma), "delta": float(delta),
            "delta_over_gamma": float(delta / gamma) if gamma != 0 else np.nan,
            "elasticity_at_mean_d": float(delta / (gamma + delta * m))
            if (gamma + delta * m) != 0 else np.nan,
            "mean_log_distance": m}


def untargeted_specification_ladder(data, panel=None, structural=None, verbose=True):
    """
    The section's table: the measured moment, the two robustness readings of it, the
    THEORETICAL moment, and the published number — all on the delta/gamma scale.

      Baseline eta (PPML)     the reported moment;
      Level-linear refit      the same panel in the reduced form's own functional shape.
                              It is a rung rather than a footnote because it carries one
                              of the two identifiable parts of the gap below;
      Multi-variety firms     every variety of a (region, sector) pooled into one firm —
                              a SIREN is a bundle, a model supplier is one variety;
      Structural eta          the granular derivative, computed and not estimated, so it
                              has no confidence interval;
      Empirical delta/gamma   Table 3.

    Elasticities are not reported: only the delta/gamma reading is comparable to the
    published number.
    """
    panel = build_a_ir_panel(data) if panel is None else panel
    lin = linear_delta_over_gamma(panel)
    rows = []

    def _row(label, dg, note, lo=np.nan, hi=np.nan, n=np.nan):
        rows.append({"specification": label, "delta/gamma": dg, "ci_lo": lo, "ci_hi": hi,
                     "n_obs": n, "isolates": note})

    def _ppml(df, label, note):
        fit = fepois_fit(df, "a_ir", fe=BASELINE_FE, cluster="cluster")
        b = float(fit.coef()["log_distance"])
        se = float(fit.se()["log_distance"])
        _row(label, delta_over_gamma_from_eta(b), note,
             lo=delta_over_gamma_from_eta(b - 1.96 * se),
             hi=delta_over_gamma_from_eta(b + 1.96 * se),
             n=int(getattr(fit, "_N", len(df))))

    _ppml(panel, "Baseline eta (PPML)",
          "the measured moment: within supplier, within buyer cell")
    _row("Level-linear refit", lin["delta_over_gamma"],
         f"the reduced form's own shape, at the simulated mean log d = "
         f"{lin['mean_log_distance']:.2f}", n=len(panel))
    _ppml(build_a_ir_panel(data, firm_key=("ze2010", "A129")), "Multi-variety firms",
          "the unit: varieties of a cell pooled into one firm")
    if structural is not None:
        _row("Structural eta", structural["dg_struct"],
             "the granular derivative, computed — no interval, it is not an estimate")

    emp = EMPIRICAL_DELTA_OVER_GAMMA.get(data["industry"], {})
    if emp.get("estimate") is not None:
        _row("Empirical delta/gamma (Table 3)", emp["estimate"],
             "the published reduced form", lo=emp["ci_lo"], hi=emp["ci_hi"])

    df = pd.DataFrame(rows).set_index("specification")
    df.attrs.update(lin)
    df.attrs["mean_log_d_empirical"] = EMPIRICAL_MEAN_LOG_D
    df.attrs["ceiling_on_ratio"] = -1.0 / EMPIRICAL_MEAN_LOG_D
    if verbose:
        print(f"[{data['industry']}]  delta/gamma, the reduced form's own scale "
              f"(mean log d = {EMPIRICAL_MEAN_LOG_D}, d = "
              f"{np.exp(EMPIRICAL_MEAN_LOG_D):.0f} km; the ratio cannot exceed "
              f"{abs(df.attrs['ceiling_on_ratio']):.3f} in magnitude)")
        print(df.round(4).to_string())
    return df


def plot_untargeted_ladder(ladder, industry, figsize=None, save_to=None):
    """
    The table as a picture, on the reduced form's own delta/gamma scale — the only scale
    the section reports. One bar per rung against the band the data's delta/gamma occupies,
    with the ceiling -1/mean(log d) drawn in: a bar inside the band reproduces Table 3, and
    how close the bars sit to the ceiling says how much room the ratio scale has left.
    """
    emp = EMPIRICAL_DELTA_OVER_GAMMA.get(industry, {})
    df = ladder[::-1]
    y = np.arange(len(df))
    is_data = df.index.str.startswith("Empirical")
    is_struct = df.index.str.startswith("Structural")
    vals = df["delta/gamma"].to_numpy(float)
    lo, hi = df["ci_lo"].to_numpy(float), df["ci_hi"].to_numpy(float)

    fig, ax = plt.subplots(figsize=figsize or get_figsize(wf=1.0, hf=0.6))
    if emp.get("ci_lo") is not None:
        ax.axvspan(emp["ci_lo"], emp["ci_hi"], color=reference_color, alpha=0.15,
                   label=f"data 95{pct()} CI")
    ax.barh(y, vals, color=[reference_color if d else
                            (0.30, 0.55, 0.40) if t else sim_color
                            for d, t in zip(is_data, is_struct)], alpha=0.85)
    for i, (l, h) in enumerate(zip(lo, hi)):
        if np.isfinite(l) and np.isfinite(h):
            ax.plot([l, h], [i, i], color="black", linewidth=1)
    ceil = -1.0 / EMPIRICAL_MEAN_LOG_D
    ax.axvline(ceil, color="0.4", linestyle=":", linewidth=1.1)
    ax.annotate(rf"$-1/\overline{{\log d}}$", (ceil, len(df) - 0.4), fontsize=fs(8),
                color="0.35", ha="right")
    ax.axvline(0.0, color="0.6", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(df.index, fontsize=fs(9))
    ax.set_xlabel(r"$\delta/\gamma$ (reduced-form scale)")
    ax.set_title(f"Untargeted moment against Table 3 — {industry}", fontsize=fs(11))
    ax.legend(frameon=False, fontsize=fs(8), loc="lower left")
    _despine(ax)
    fig.tight_layout()
    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        fig.savefig(save_to, bbox_inches="tight")
    return ax
