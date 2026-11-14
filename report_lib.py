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
    _despine, _downstream_ze_index, _read_named_value, _ze_key, fs, get_figsize,
    load_granular_data, pct, reference_color, set_name_axis_fontsize, sim_color,
    toulouse_color, unpack_estimated_T, log_tau, tau_elasticity, tau_log_knots,
    sourcing_geometry,
)


# 2.5/97.5% quantiles: the fake-world intervals reported everywhere (inference.jl)
WORLD_Q = (0.025, 0.975)


# ============================================================================
# Moment table
# ============================================================================

def generate_combined_table(industries_config, output_file="moments_comparison_combined.tex",
                            mu=2, name_A129_path="../external/A129_name_fr_eng.csv", **load_kwargs):
    """
    LaTeX table with one {Emp., Sim.} column pair per industry.

    Panel A: aggregate labor share.  Panel B: aggregate industry shares.
    Panel C: the extensive margin, block 4 = [beta_d ; beta_kappa] -- the cell cloglog
    of has_sup on the distance regressor(s) and on ln n_ls. Under each simulated
    coefficient, in parentheses, its fake-world standard deviation (sd of
    e = beta_world - beta_sim across the Step-2 worlds, `reg_coef_world_table`) when
    Step 2 has run. beta_kappa is reported here only (`plot_reg_coef` draws beta_d).
    Panel D: the import share of each upstream sector, f_s = 1 - domestic_share, the
    foreign row of the sourcing-share block. It is a PROFILED moment: Phi^F_s is set so
    the closed-form share equals the target exactly, so the simulated share differs from
    it only by the Monte-Carlo error of the realised prices. Sectors that do not import
    are omitted; the panel is dropped when no industry imports.

    industries_config : [{'industry': 'aero', 'display_name': 'Aerospace'}, ...]
    mu                : 1 (step1 best parameters) or 2 (step3 best parameters)
    """
    all_data = {industry_config["industry"]: load_granular_data(industry_config["industry"], mu=mu, **load_kwargs)
                for industry_config in industries_config}

    try:
        name_A129 = pd.read_csv(name_A129_path)
    except Exception:
        name_A129 = None

    n_ind = len(industries_config)
    col_spec = "l " + " ".join(["S[table-format=-1.4] S[table-format=-1.4]"] * n_ind)
    header_row1 = " & " + " & ".join(
        [f"\\multicolumn{{2}}{{c}}{{{industry_config['display_name']}}}" for industry_config in industries_config]) + r" \\"
    cmidrules = " ".join([f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(n_ind)])
    header_row2 = " & " + " & ".join(["{Emp.} & {Sim.}"] * n_ind) + r" \\"

    # --- Panel A -------------------------------------------------------------
    row_A = ["Aggregate Labor Share"]
    for industry_config in industries_config:
        industry_data = all_data[industry_config["industry"]]
        row_A += [float(industry_data["empirical_moments_dict"]["agg_labor_share"][0]),
                  float(industry_data["simulated_moments_dict"]["agg_labor_share"][0])]

    # --- Panel B -------------------------------------------------------------
    all_codes = sorted({code for industry_data in all_data.values() for code in industry_data["sector_names"]})
    shares = {}
    for industry_config in industries_config:
        industry_data = all_data[industry_config["industry"]]
        shares[industry_config["industry"]] = pd.DataFrame({
            "A129": industry_data["sector_names"],
            "Empirical": industry_data["empirical_moments_dict"]["agg_industry_share"],
            "Simulated": industry_data["simulated_moments_dict"]["agg_industry_share"],
        })

    def _label(code):
        if name_A129 is not None and "A129" in name_A129.columns:
            name_match = name_A129.loc[name_A129["A129"].astype(str) == str(code), "name"].values
            if len(name_match):
                return name_match[0]
        return code

    # --- Panel D: import shares (the foreign row of the gamma block) ----------
    imports = {}
    for industry_config in industries_config:
        industry_data = all_data[industry_config["industry"]]
        foreign_active = np.asarray(industry_data.get("FOREIGN_ACTIVE", np.zeros(len(industry_data["sector_names"]), bool)))
        sim_f = industry_data["simulated_moments_dict"].get("foreign_share")
        imports[industry_config["industry"]] = {
            str(code): (float(industry_data["foreign_share"][s]),
                        np.nan if sim_f is None else float(sim_f[s]))
            for s, code in enumerate(industry_data["sector_names"]) if foreign_active[s]}
    import_codes = sorted({code for value in imports.values() for code in value})
    panel_D_rows = [[_label(code)] + [x for industry_config in industries_config
                                      for x in imports[industry_config["industry"]].get(code, ("---", "---"))]
                    for code in import_codes]

    panel_B_rows = []
    for code in all_codes:
        label = code
        if name_A129 is not None and "A129" in name_A129.columns:
            name_match = name_A129.loc[name_A129["A129"].astype(str) == str(code), "name"].values
            if len(name_match):
                label = name_match[0]
        row = [label]
        for industry_config in industries_config:
            df = shares[industry_config["industry"]]
            m = df[df["A129"].astype(str) == str(code)]
            row += ([float(m.iloc[0]["Empirical"]), float(m.iloc[0]["Simulated"])]
                    if len(m) else ["---", "---"])
        panel_B_rows.append(row)

    # --- Panel C: block 4 -----------------------------------------------------
    # Rows are keyed on the coefficient LABEL (bins or ln d, then ln n), so two industries
    # estimated with the same n_coef line up; a label absent from one industry is `---`.
    all_coef_labels, coef_values, coef_sds = [], {}, {}
    for industry_config in industries_config:
        industry_data = all_data[industry_config["industry"]]
        coef_labels = reg_bin_labels(industry_data["n_coef"], with_kappa=True)
        for l in coef_labels:
            if l not in all_coef_labels:
                all_coef_labels.append(l)
        emp_coef = industry_data["empirical_moments_dict"]["reg_coef"]
        sim_coef = industry_data["simulated_moments_dict"]["reg_coef"]
        world_table = reg_coef_world_table(industry_data)
        coef_values[industry_config["industry"]] = {l: (float(e), float(m)) for l, e, m in zip(coef_labels, emp_coef, sim_coef)}
        coef_sds[industry_config["industry"]] = ({} if world_table is None else
                                dict(zip(coef_labels, world_table["world_sd_e"].values)))
    # ln n last, whatever the order the industries listed the bins in
    all_coef_labels = [l for l in all_coef_labels if "n_{ls}" not in l] + [l for l in all_coef_labels if "n_{ls}" in l]

    def fmt(value, decimals=4):
        if isinstance(value, str) or (isinstance(value, float) and np.isnan(value)):
            return "{---}"
        return f"{value:.{decimals}f}"

    mu_label = r"$\hat{\mu}_1$ (Step~1)" if mu == 1 else r"$\hat{\mu}_2$ (Step~3)"
    latex = (r"\begin{table}[H]" "\n" r"\centering" "\n"
           r"\caption{Empirical and Simulated Moments: Comparison Across Industries}" "\n"
           r"\label{tab:moments_comparison_combined}" "\n"
           r"\renewcommand{\arraystretch}{1.2}" "\n" r"\small" "\n"
           r"\begin{tabular}{" + col_spec + "}\n" r"\toprule" "\n"
           + header_row1 + "\n" + cmidrules + "\n" + header_row2 + "\n")

    latex += "\\midrule\n"
    latex += f"\\multicolumn{{{1 + 2 * n_ind}}}{{l}}{{\\textbf{{Panel A: Aggregate Labor Share}}}} \\\\\n"
    latex += "\\midrule\n"
    latex += row_A[0] + " & " + " & ".join(fmt(value) for value in row_A[1:]) + " \\\\\n"

    latex += "\\midrule\n"
    latex += f"\\multicolumn{{{1 + 2 * n_ind}}}{{l}}{{\\textbf{{Panel B: Aggregate Industry Shares}}}} \\\\\n"
    latex += "\\midrule\n"
    for row in panel_B_rows:
        latex += str(row[0]) + " & " + " & ".join(fmt(value) for value in row[1:]) + " \\\\\n"

    latex += "\\midrule\n"
    latex += (f"\\multicolumn{{{1 + 2 * n_ind}}}{{l}}{{\\textbf{{Panel C: Extensive Margin "
            f"(cell cloglog)}}}} \\\\\n")
    latex += "\\midrule\n"
    any_sd = False
    for l in all_coef_labels:
        name = (r"Density, $\beta_\kappa$ on " + l if "n_{ls}" in l
                else r"Distance, $\beta_d$ " + l)
        row, sd_cells = [name], [""]
        for industry_config in industries_config:
            value = coef_values[industry_config["industry"]].get(l)
            row += ["---", "---"] if value is None else list(value)
            sd = coef_sds[industry_config["industry"]].get(l)
            ok = sd is not None and np.isfinite(sd)
            sd_cells += ["{}", f"{{({sd:.4f})}}" if ok else "{}"]
            any_sd |= ok
        latex += str(row[0]) + " & " + " & ".join(fmt(value) for value in row[1:]) + " \\\\\n"
        if any(x not in ("", "{}") for x in sd_cells):
            latex += " & ".join(sd_cells) + " \\\\\n"

    if panel_D_rows:
        latex += "\\midrule\n"
        latex += (f"\\multicolumn{{{1 + 2 * n_ind}}}{{l}}{{\\textbf{{Panel D: Import Shares "
                f"(profiled)}}}} \\\\\n")
        latex += "\\midrule\n"
        for row in panel_D_rows:
            latex += str(row[0]) + " & " + " & ".join(fmt(value) for value in row[1:]) + " \\\\\n"

    latex += "\\bottomrule\n" + r"\end{tabular}" + "\n\n"
    latex += (r"\vspace{0.3cm}" "\n"
            r"\caption*{\footnotesize \emph{Notes}: Empirical moments against moments simulated "
            r"from the model calibrated on each industry, evaluated at " + mu_label + r". "
            r"``Emp.'' = Empirical, ``Sim.'' = Simulated. Panel~A reports the aggregate labor "
            r"share, Panel~B the sectoral industry shares. ``---'' indicates the sector is not "
            r"present in that industry's sample. Panel~C reports the cell-level cloglog of the "
            r"supplier indicator on distance and on the log number of establishments $n_{ls}$, "
            r"with a sector $\times$ attraction-area fixed effect; the model counterpart is "
            r"averaged over 100 networks of $\hat N_s$ varieties."
            + (r" In parentheses: the standard deviation of the coefficient in a single "
               r"finite economy drawn from the model, i.e. of the simulated residual "
               r"$\beta_{world}-\beta_{sim}$ across the Step~2 simulations." if any_sd else "")
            + (r" Panel~D reports each upstream sector's import share $f_s = 1 - $ domestic "
               r"share (Emp.) against the simulated share of the buyers' spending on that "
               r"sector won by its foreign competitor (Sim.). It is matched exactly by the "
               r"foreign Fr\'echet scale $\Phi^F_s$ in closed form, so the gap is simulation "
               r"error only." if panel_D_rows else "")
            + r" The zero-supplier shares $\bar{G}_s(0)$ are reported in a separate figure.}"
            "\n" r"\end{table}" "\n")

    if output_file is not None:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w") as f:
            f.write(latex)
        print(f"Combined table saved to: {output_file}")
    return latex


def generate_parameter_table(industries_config, output_file="parameters_combined.tex",
                             mu=2, quantiles=WORLD_Q, **load_kwargs):
    """
    LaTeX table of the searched parameters with their fake-world 95% intervals.

    One {Estimate, 95% interval} column pair per industry; rows alpha (one per segment
    under binned tau), kappa, then N_hat_s per sector. The interval is the 2.5/97.5%
    quantiles of the parameter re-estimated in each fake world of the inference step
    (`<inf_step>/inference_world/`): Step 2 at theta_hat_1 (mu = 1), Step 4 at
    theta_hat_2 (mu = 2). Below each estimate, in parentheses, the sd across worlds.
    """
    all_data = {industry_config["industry"]: load_granular_data(industry_config["industry"], mu=mu, **load_kwargs)
                for industry_config in industries_config}
    tabs = {k: world_param_table(industry_data, quantiles=quantiles) for k, industry_data in all_data.items()}
    n_ind = len(industries_config)

    def _key_label(l, industry_data):
        if l.startswith("alpha"):
            return r"$\alpha$" if industry_data["n_tau"] == 1 else rf"$\alpha_{{{l[6:-1]}}}$"
        if l == "kappa":
            return r"$\kappa$"
        return rf"$\hat N_s$, {l[4:-1]}"

    rows = []                                   # (label, {industry: (est, sd, lo, hi)})
    for industry_config in industries_config:
        world_table, industry_data = tabs[industry_config["industry"]], all_data[industry_config["industry"]]
        if world_table is None:
            continue
        for l in world_table.index:
            if not (_is_trade_param(l) or l.startswith("N_s[")):
                continue
            key = _key_label(l, industry_data)
            row_entry = next((param_row for param_row in rows if param_row[0] == key), None)
            if row_entry is None:
                row_entry = (key, {})
                rows.append(row_entry)
            param_row = world_table.loc[l]
            row_entry[1][industry_config["industry"]] = (param_row["estimate"], param_row["sd"], param_row["q_lo"], param_row["q_hi"])

    col_spec = "l " + " ".join(["c c"] * n_ind)
    head1 = " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{industry_config['display_name']}}}"
                               for industry_config in industries_config) + r" \\"
    cmid = " ".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(n_ind))
    head2 = " & " + " & ".join([r"Estimate & 95\% interval"] * n_ind) + r" \\"
    mu_label = r"$\hat{\mu}_1$ (Step~1)" if mu == 1 else r"$\hat{\mu}_2$ (Step~3)"
    latex = (r"\begin{table}[H]" "\n" r"\centering" "\n"
           r"\caption{Parameter Estimates and Confidence Intervals}" "\n"
           r"\label{tab:parameters_combined}" "\n" r"\small" "\n"
           r"\begin{tabular}{" + col_spec + "}\n" r"\toprule" "\n"
           + head1 + "\n" + cmid + "\n" + head2 + "\n" r"\midrule" "\n")
    for i, (key, values_by_industry) in enumerate(rows):
        if i and key.startswith(r"$\hat N_s$") and not rows[i - 1][0].startswith(r"$\hat N_s$"):
            latex += r"\midrule" "\n"
        is_N = key.startswith(r"$\hat N_s$")
        cells, sd_cells = [key], [""]
        for industry_config in industries_config:
            estimates = values_by_industry.get(industry_config["industry"])
            if estimates is None:
                cells += ["---", "---"]; sd_cells += ["", ""]
                continue
            e, sd, lo, hi = estimates
            formatter = (lambda x: f"{x:.0f}") if is_N else (lambda x: f"{x:.3f}")
            cells += [formatter(e), f"[{formatter(lo)}, {formatter(hi)}]"]
            sd_cells += [f"({sd:.1f})" if is_N else f"({sd:.3f})", ""]
        latex += " & ".join(cells) + r" \\" "\n" + " & ".join(sd_cells) + r" \\" "\n"
    n_worlds_by_industry = {k: (world_table.attrs["n_worlds"] if world_table is not None else 0) for k, world_table in tabs.items()}
    latex += (r"\bottomrule" "\n" r"\end{tabular}" "\n\n" r"\vspace{0.3cm}" "\n"
            r"\caption*{\footnotesize \emph{Notes}: Estimates at " + mu_label + r". "
            r"Inference from economies simulated from the model at the estimate: in each one, "
            r"one network of $\hat N_s$ varieties with independent draws gives its own data "
            r"moments, on which the whole estimator is re-run (T and A re-inverted, "
            r"$N_s$ re-calibrated, $(\alpha,\kappa)$ re-estimated). 95\% interval: "
            r"2.5--97.5\% quantiles across simulations; standard deviation in parentheses. "
            r"Usable simulations: " + ", ".join(f"{industry_config['display_name']} {n_worlds_by_industry[industry_config['industry']]}"
                                           for industry_config in industries_config) + r".}" "\n"
            r"\end{table}" "\n")
    if output_file is not None:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w") as formatter:
            formatter.write(latex)
        print(f"Parameter table saved to: {output_file}")
    return latex


# ============================================================================
# Fit: sourcing shares and downstream sales
# ============================================================================

def bubble_scatter(ax, x, y, xlabel, ylabel, title, size_scale=300,
                   regression_line=False, weights=None, color=None, label=None):
    """Bubble scatter of simulated against empirical, with a WLS-through-origin fit."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = x > 0
    x, y = x[keep], y[keep]
    weights_used = np.ones_like(x) if weights is None else np.asarray(weights, float)[keep]

    lims = [min(x.min(), y.min()) * 0.9, max(x.max(), y.max()) * 1.1]
    ax.scatter(x, y, s=size_scale * x / x.max(), alpha=1, edgecolor="black",
               linewidths=0.5, color=color or toulouse_color, label=label)

    slope, t_stat = wls_through_origin(x, y, weights_used)
    ax.text(0.98, 0.09, rf"Coefficient: ${np.round(slope, 3)}$", ha="right", va="bottom",
            fontsize=fs(10), transform=ax.transAxes)
    ax.text(0.98, 0.01, rf"t-stat: ${np.round(t_stat, 1)}$", ha="right", va="bottom",
            fontsize=fs(10), transform=ax.transAxes)

    if regression_line:
        x_line = np.linspace(0, lims[1], 100)
        ax.plot(x_line, slope * x_line, linestyle="--", color="green")
    else:
        ax.plot(lims, lims, color="black")

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(linestyle="dashed", alpha=0.5)
    _despine(ax)
    return slope, t_stat


def wls_through_origin(x, y, weights=None):
    """b and its t-stat for y = b*x (no intercept); weights default to x."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    weights_used = x if weights is None else np.asarray(weights, float)
    denom = np.sum(weights_used * x ** 2)
    slope = np.sum(weights_used * x * y) / denom
    resid = y - slope * x
    residual_variance = np.sum(weights_used * resid ** 2) / max(len(x) - 1, 1)
    slope_se = np.sqrt(residual_variance / denom)
    return slope, (slope / slope_se if slope_se > 0 else np.inf)


def gamma_aa_points(data):
    """
    Split the AA-level gamma moments into the two series the dashboard panel plots.

    Returns (x_free, y_free, x_ref, y_ref):
      * free      — active, non-reference (sector, AA) pairs, i.e. the moments that
                    actually enter the criterion;
      * reference — each sector's dropped reference area, RECONSTRUCTED from the
                    within-sector adding-up constraint on TOTAL purchases (as Julia's
                    GAMMA_REF_MAP, c_s = 1)
                        gamma_ref,s = 1 - f_s - sum_{a != ref} gamma_{s,a},
                    applied to the SIMULATED gammas and import share f_s; its x is the
                    observed share.
    """
    emp_gamma, sim_gamma = data["empirical_moments_dict"]["gamma_aa"], data["simulated_moments_dict"]["gamma_aa"]
    S, reference_area, active = data["S"], data["T_REF_AA"], data["AA_ACTIVE"]
    f_sim = data["simulated_moments_dict"].get("foreign_share")
    domestic_total = 1.0 - (np.zeros(S) if f_sim is None else np.asarray(f_sim, float))

    free = active.copy()
    for s in range(S):
        if reference_area[s] >= 0:
            free[s, reference_area[s]] = False

    x_free, y_free = emp_gamma[free], sim_gamma[free]
    x_ref, y_ref = [], []
    for s in range(S):
        if reference_area[s] < 0 or not free[s].any() or emp_gamma[s, reference_area[s]] <= 0:
            continue
        x_ref.append(emp_gamma[s, reference_area[s]])
        y_ref.append(domestic_total[s] - sim_gamma[s, free[s]].sum())
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
    slope, t_stat = wls_through_origin(x_free, y_free, weights=x_free)
    ax.plot(lims, lims, color="black", linewidth=1)
    ax.text(0.98, 0.09, rf"Coefficient: ${np.round(slope, 3)}$", ha="right", va="bottom",
            fontsize=fs(10), transform=ax.transAxes)
    ax.text(0.98, 0.01, rf"t-stat: ${np.round(t_stat, 1)}$", ha="right", va="bottom",
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

def _paired_errorbar(ax, labels, emp, sim, se_emp=None, se_sim=None, z_critical=1.96,
                     ylabel="", xlabel="", title=None, rotate=0,
                     band_emp=None, band_sim=None, sim_label="Simulated", legend=True):
    """
    Empirical (left) against simulated (right) at each tick, with +/- ci*SE bars.

    `band_emp` / `band_sim` = (lo, hi) arrays draw an ASYMMETRIC interval in data units
    instead (a fake-world quantile band is not symmetric about the point) and take
    precedence over the corresponding SE.

    A point whose SE is zero, missing or NaN keeps its MARKER and simply gets no bar.
    Drawing it through `errorbar` with a zero yerr would put a bare cap on the marker,
    which reads as a measurement of zero precision rather than as an absent one; and a
    NaN yerr silently drops the point altogether, which is worse — the value is known,
    only its uncertainty is not.
    """
    x = np.arange(len(labels), dtype=float)
    off = 0.12

    def _series(xs, vals, se, band=None, **kw):
        vals = np.asarray(vals, float)
        if band is not None:
            lo, hi = (np.asarray(b, float) for b in band)
            has = np.isfinite(lo) & np.isfinite(hi) & (hi > lo)
            if has.any():
                yerr = np.vstack([np.maximum(vals[has] - lo[has], 0.0),
                                  np.maximum(hi[has] - vals[has], 0.0)])
                ax.errorbar(xs[has], vals[has], yerr=yerr, **kw)
                kw = {**kw, "label": None}
            if (~has).any():
                ax.errorbar(xs[~has], vals[~has], yerr=None, **kw)
            return
        if se is None:
            ax.errorbar(xs, vals, yerr=None, **kw)
            return
        se = np.asarray(se, float)
        has = np.isfinite(se) & (se > 0)
        # With bars, then without — one legend entry, taken from whichever is non-empty.
        if has.any():
            ax.errorbar(xs[has], vals[has], yerr=z_critical * se[has], **kw)
            kw = {**kw, "label": None}
        if (~has).any():
            ax.errorbar(xs[~has], vals[~has], yerr=None, **kw)

    _series(x - off, emp, se_emp, band_emp,
            fmt="o", color=toulouse_color, capsize=3, markersize=7,
            markeredgecolor="black", markeredgewidth=0.5, linestyle="none",
            label="Empirical")
    _series(x + off, sim, se_sim, band_sim,
            fmt="D", color=sim_color, capsize=3, markersize=6,
            markeredgecolor="black", markeredgewidth=0.5, linestyle="none",
            label=sim_label)
    #ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotate, ha="right" if rotate else "center")
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title)
    ax.grid(axis="y", linestyle="dashed", alpha=0.5)
    if legend:
        ax.legend(frameon=False)
    _despine(ax)
    return ax


def _legend_below_xaxis(ax, pad=0.02, ncol=2, size_at_ref=11):
    """One-row legend centred under the x-axis, just clear of its (rotated) tick labels."""
    fig = ax.figure
    fig.canvas.draw()
    axis_bbox = ax.xaxis.get_tightbbox(fig.canvas.get_renderer())
    axis_bottom = ax.transAxes.inverted().transform((0.0, axis_bbox.y0))[1] if axis_bbox is not None else -0.15
    return ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, axis_bottom - pad),
                     ncol=ncol, fontsize=fs(size_at_ref), handletextpad=0.3,
                     columnspacing=1.2)


def reg_bin_labels(n_coef, with_kappa=False):
    """Tick labels of block 4: the distance coefficients, then (optionally) ln n."""
    labels_by_n_coef = {1: [r"$\ln d$"],
         4: [r"$]50,100]$", r"$]100,150]$", r"$]150,200]$", r"$>200$"],
         5: [r"$]20,50]$", r"$]50,100]$", r"$]100,150]$", r"$]150,200]$", r"$>200$"]}
    labels = labels_by_n_coef.get(n_coef, [rf"$\beta_{{{i + 1}}}$" for i in range(n_coef)])
    return labels + ([r"$\ln n_{ls}$"] if with_kappa else [])


# ---------------------------------------------------------------------------
# Fake-world confidence bands (inference.jl). Nothing is re-estimated here.
# ---------------------------------------------------------------------------

def _sim_block4(data):
    return np.asarray(data["simulated_moments_dict"]["reg_coef"], float)


def reg_coef_world_table(data, quantiles=WORLD_Q):
    """
    Block 4 = [beta_d ; beta_kappa]: data, simulated, and the fake-world 95% band.

    `step2/world_residuals.npz` holds, for B worlds drawn at theta_hat_1, the residual
    `e_b = beta_world - beta_sim` (a world's own cell cloglog on ONE network of N_hat_s
    varieties, minus the continuum simulator at the world's recalibrated T, A, N). Its
    covariance IS the Omega_world the estimator weights block 4 with. Under the model,
    the data residual `e_data = data - simulated` is one more draw of `e`, so

        band = simulated + [q_2.5(e), q_97.5(e)]

    is where the model places the DATA moment of a single finite economy. The data point
    outside the band is a rejection of the fit for that coefficient at 5%. `z` is the
    same reading in sd units, centred on the world mean of `e` (the continuum simulator's
    bias against a finite world, the `mean e_b` column of world_weight_report.txt).

    The band is built at theta_hat_1 (Step 2 is the only step that saves the residuals);
    for mu = 2 it is re-centred on the step-3 simulated moment. Returns None when the
    residuals are absent.
    """
    world_residuals = data.get("world_residuals")
    if world_residuals is None:
        return None
    residual_draws = np.asarray(world_residuals["e"], float)
    emp_coef = np.asarray(data["empirical_moments_dict"]["reg_coef"], float)
    sim_coef = _sim_block4(data)
    lo_e, hi_e = np.nanquantile(residual_draws, quantiles[0], axis=0), np.nanquantile(residual_draws, quantiles[1], axis=0)
    mu_e, sd_e = np.nanmean(residual_draws, axis=0), np.nanstd(residual_draws, axis=0, ddof=1)
    labels = [f"beta_d[{l}]" for l in data["reg_coef_labels"][:-1]] + ["beta_kappa"]
    df = pd.DataFrame({
        "data": emp_coef, "simulated": sim_coef, "residual": emp_coef - sim_coef,
        "world_mean_e": mu_e, "world_sd_e": sd_e,
        "band_lo": sim_coef + lo_e, "band_hi": sim_coef + hi_e,
        "z": (emp_coef - sim_coef - mu_e) / sd_e,
    }, index=pd.Index(labels, name="moment"))
    df["inside_band"] = (df["data"] >= df["band_lo"]) & (df["data"] <= df["band_hi"])
    W = data.get("W_world")
    residual = emp_coef - sim_coef
    J = float(residual @ W @ residual) if W is not None and W.shape == (emp_coef.size, emp_coef.size) else np.nan
    # Under the model e_data is one more draw of e, so J is read against the same
    # quadratic form over the worlds (uncentred on both sides: the simulator's bias
    # against a finite world is part of what e_data carries too).
    J_worlds = (np.einsum("bi,ij,bj->b", residual_draws, W, residual_draws) if np.isfinite(J) else None)
    df.attrs.update(n_worlds=int(residual_draws.shape[0]), q=quantiles, J=J,
                    J_pvalue=(float(np.mean(J_worlds >= J)) if J_worlds is not None else np.nan))
    return df


def world_param_table(data, select=None, quantiles=WORLD_Q, prefixes=None):
    """
    Parameter-level inference from the fake worlds of `<inf_step>/inference_world/`.

    One row per parameter: the estimate, the mean across worlds, bias = mean - estimate,
    sd, and the 2.5/97.5% quantiles -- the same numbers as Julia's
    `world_inference.txt`. `select` is a predicate on the label (e.g.
    `lambda l: l.startswith("alpha") or l == "kappa"`), `prefixes` a tuple of label
    prefixes (e.g. `("alpha", "kappa", "N_s")`). A T entry that is NaN in a world
    is an area that won nothing there (T set to zero); `empty_share` counts them and the
    statistics are over the other worlds, as in Julia's report.
    """
    world_inference = data.get("world_inference")
    if world_inference is None:
        return None
    labels, param_draws, estimate = world_inference["labels"], np.asarray(world_inference["params"], float), np.asarray(world_inference["estimate"], float)
    keep = [j for j, l in enumerate(labels) if (select is None or select(l))
            and (prefixes is None or l.startswith(tuple(prefixes)))]
    rows = []
    for j in keep:
        draws_j = param_draws[:, j]
        finite_draws = draws_j[np.isfinite(draws_j)]
        rows.append({"parameter": labels[j], "estimate": estimate[j],
                     "mean": finite_draws.mean() if finite_draws.size else np.nan,
                     "bias": (finite_draws.mean() - estimate[j]) if finite_draws.size else np.nan,
                     "sd": finite_draws.std(ddof=1) if finite_draws.size > 1 else np.nan,
                     "q_lo": np.quantile(finite_draws, quantiles[0]) if finite_draws.size else np.nan,
                     "q_hi": np.quantile(finite_draws, quantiles[1]) if finite_draws.size else np.nan,
                     "empty_share": float(np.mean(~np.isfinite(draws_j)))})
    df = pd.DataFrame(rows).set_index("parameter")
    df.attrs.update(n_worlds=int(param_draws.shape[0]), folder=str(world_inference.get("folder", "")))
    return df


def _is_trade_param(label):
    return label.startswith("alpha") or label == "kappa"


def _n_hat_world(data):
    """(B, S) variety counts re-calibrated in each fake world, or None."""
    world_inference = data.get("world_inference")
    if world_inference is None:
        return None
    cols = [world_inference["labels"].index(f"N_s[{s}]") for s in data["sector_names"]
            if f"N_s[{s}]" in world_inference["labels"]]
    if len(cols) != data["S"]:
        return None
    return np.rint(np.asarray(world_inference["params"], float)[:, cols]).astype(int)


def plot_reg_coef(data, ax=None, save_to=None, quantiles=WORLD_Q):
    """
    The distance coefficients beta_d of block 4, empirical against simulated, with the
    95% fake-world band on the simulated point (`reg_coef_world_table`).

    beta_kappa (ln n_ls) is not drawn: it is on another scale and identifies another
    parameter, and is reported with its fake-world sd in Panel C of
    `generate_combined_table`. The data point carries no bar: the Julia code reads no
    data covariance, and the band already contains the sampling noise of one finite
    network.
    """
    n_distance = data["n_coef"]
    emp_coef = np.asarray(data["empirical_moments_dict"]["reg_coef"], float)[:n_distance]
    sim_coef = _sim_block4(data)[:n_distance]
    world_table = reg_coef_world_table(data, quantiles=quantiles)
    band = None if world_table is None else (world_table["band_lo"].values[:n_distance], world_table["band_hi"].values[:n_distance])
    sim_label = "Simulated" + ("" if band is None
                               else rf" (95\% band)" if mpl.rcParams.get("text.usetex")
                               else " (95% band)")
    if ax is None:
        fig, ax = plt.subplots(figsize=get_figsize(height_fraction=0.62))
    else:
        fig = ax.figure
    _paired_errorbar(ax, reg_bin_labels(n_distance), emp_coef, sim_coef, band_sim=band,
                     ylabel="Cell cloglog coefficient on distance",
                     xlabel="Distance bin (km)" if n_distance > 1 else "",
                     sim_label=sim_label, legend=False)
    # legend above the axes: inside, it sits on the bands of the far bins
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout()
    if world_table is not None:
        outside = world_table.iloc[:n_distance]
        outside = list(outside.index[~outside["inside_band"]])
        if outside:
            print(f"[plot_reg_coef] {data['industry']}: data outside the 95% simulation band "
                  f"for {outside}.")
    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        fig.savefig(save_to, format="pdf", bbox_inches="tight")
    return ax


# ---------------------------------------------------------------------------
# Block 4 in probability points: average marginal effects of the cell cloglog
# ---------------------------------------------------------------------------
#
# The cloglog coefficients are index effects, P(y=1) = F(a_g + x'beta) with
# F(eta) = 1 - exp(-e^eta), so they are not comparable with the LPM coefficients of the
# reduced-form figure (Fig. 4, Panel A). The AME relative to the reference bin is:
#
#   eta0_i  = a_g(i) + beta_kappa ln n_i            (every distance dummy at zero)
#   AME_b   = mean_i [F(eta0_i + beta_b) - F(eta0_i)],     AME_0 = 0,
#   AME_kap = mean_i f(eta_i) beta_kappa,   f(eta) = e^eta exp(-e^eta),
#
# over the regression set A (groups with variation in the data y). The fixed effects are
# not returned by the cloglog, and beta is either the data's or the model's, so a_g is
# set by ONE rule for both: sum_{i in g} F(a_g + x_i'beta) = sum_{i in g} y_i (data),
# i.e. the LPM-with-FE property that fitted group means equal observed ones. Empirical
# and simulated AMEs are then evaluated on the same cells and group levels, and differ
# only through beta. With n_coef = 1 the regressor is ln d and AME_d = mean f(eta) beta_d.


def _cloglog_F(eta):
    return -np.expm1(-np.exp(np.minimum(eta, 30.0)))


def _cloglog_f(eta):
    exp_eta = np.exp(np.minimum(eta, 30.0))
    return exp_eta * np.exp(-exp_eta)


def reg_cell_sample(data):
    """
    The regression set A of block 4, rebuilt as `load_parameters.jl` SECTION 10 builds
    `REG_CELL_*`: simulated cells (CELL_MASK) with a has_sup entry in cells_ls.csv and
    d > 0, d the distance to the nearest downstream region, the distance bin of that
    distance (edges 50/100/150/200 km, or 20/50/... with five bins; 0 = reference) or
    ln d when n_coef = 1, ln n_ls, and the sector x attraction-area group.

    Groups without variation in the data y are dropped (`cell_cloglog_beta` drops them;
    their fixed effect would be infinite). Returns a DataFrame, one row per cell.
    """
    input_folder = data["input_folder"]
    S, R, n_coef = data["S"], data["R"], int(data["n_coef"])
    CELL_MASK = np.asarray(data["CELL_MASK"], bool)
    cells_df = pd.read_csv(input_folder / "cells_ls.csv")
    filter_df = pd.read_csv(input_folder / "filter_N_upstream.csv")
    sector_keys = sorted(filter_df["A129"].astype(str).unique())
    ze_keys = sorted(filter_df["ze2010"].map(_ze_key).unique())
    if len(sector_keys) != S or len(ze_keys) != R:
        raise ValueError("filter_N_upstream.csv does not match the run's S x R.")
    sector_index_of = {k: i for i, k in enumerate(sector_keys)}
    ze_index_of = {k: i for i, k in enumerate(ze_keys)}
    y = np.full((S, R), np.nan)
    log_n = np.zeros((S, R))
    sector_index = cells_df["A129"].astype(str).map(sector_index_of)
    ze_index = cells_df["ze2010"].map(_ze_key).map(ze_index_of)
    ok = sector_index.notna() & ze_index.notna()
    sector_index, ze_index = sector_index[ok].astype(int).to_numpy(), ze_index[ok].astype(int).to_numpy()
    keep = CELL_MASK[sector_index, ze_index]
    sector_index, ze_index = sector_index[keep], ze_index[keep]
    y[sector_index, ze_index] = cells_df.loc[ok, "has_sup"].to_numpy(float)[keep]
    log_n[sector_index, ze_index] = np.log(cells_df.loc[ok, "n_ls"].to_numpy(float)[keep])

    distances = np.load(input_folder / "distances.npy")[:R, :R]
    distance_to_downstream = distances[:, _downstream_ze_index(data) - 1]                  # (R, R_d)
    distance = distance_to_downstream.min(axis=1)
    edges = {5: (20, 50, 100, 150, 200), 4: (50, 100, 150, 200)}.get(n_coef, ())
    distance_bin = np.zeros(R, int)
    for edge in edges:
        distance_bin += distance > edge
    area_of_ze = np.asarray(data["aa_of_ze"]).astype(int)
    if not np.array_equal(distance_to_downstream.argmin(axis=1), area_of_ze):
        # load_parameters.jl asserts it: the group is (sector, nearest downstream region)
        raise ValueError(f"{int((distance_to_downstream.argmin(axis=1) != area_of_ze).sum())} ZE whose nearest downstream "
                         "region is not their attraction area.")
    cell_sector, cell_ze = np.nonzero(CELL_MASK)          # order does not matter: groups are labels
    in_sample = (distance[cell_ze] > 0) & np.isfinite(y[cell_sector, cell_ze])
    cell_sector, cell_ze = cell_sector[in_sample], cell_ze[in_sample]
    sample = pd.DataFrame({"s": cell_sector, "ze": cell_ze, "group": cell_sector * (area_of_ze.max() + 1) + area_of_ze[cell_ze],
                        "y": y[cell_sector, cell_ze], "dist": distance[cell_ze], "bin": distance_bin[cell_ze],
                        "log_d": np.log(distance[cell_ze]), "log_n": log_n[cell_sector, cell_ze]})
    group_mean_y = sample.groupby("group")["y"].transform("mean")
    sample = sample[(group_mean_y > 0) & (group_mean_y < 1)].reset_index(drop=True)
    sample["group"] = pd.factorize(sample["group"])[0]
    return sample


def _index_no_fe(sample, beta, n_coef, drop_distance=False):
    """x_i' beta without the fixed effect; `drop_distance` sets every distance term to 0."""
    beta = np.asarray(beta, float)
    linear_index = beta[-1] * sample["log_n"].to_numpy()
    if drop_distance:
        return linear_index
    if n_coef == 1:
        return linear_index + beta[0] * sample["log_d"].to_numpy()
    bin_effects = np.concatenate([[0.0], beta[:n_coef]])
    return linear_index + bin_effects[sample["bin"].to_numpy()]


def calibrate_cloglog_fe(sample, linear_index, n_iter=100):
    """
    a_g with sum_{i in g} F(a_g + xb_i) = sum_{i in g} y_i, for every group at once.
    The left side is increasing in a_g, so the root is unique; bisection on a bracket
    that puts every eta in [-40, 40] (F saturates well inside it).
    """
    group = sample["group"].to_numpy()
    n_groups = int(group.max()) + 1
    target = np.bincount(group, weights=sample["y"].to_numpy(), minlength=n_groups)
    xmin = np.full(n_groups, np.inf); np.minimum.at(xmin, group, linear_index)
    xmax = np.full(n_groups, -np.inf); np.maximum.at(xmax, group, linear_index)
    lo, hi = -40.0 - xmax, 40.0 - xmin
    for _ in range(n_iter):
        midpoint = 0.5 * (lo + hi)
        below_target = np.bincount(group, weights=_cloglog_F(midpoint[group] + linear_index), minlength=n_groups) < target
        lo, hi = np.where(below_target, midpoint, lo), np.where(below_target, hi, midpoint)
    return 0.5 * (lo + hi)


def extensive_ame(sample, beta, n_coef):
    """
    AMEs (probability, not pp) of block 4 at `beta` = [beta_d (n_coef) ; beta_kappa]:
    `(ame_d, ame_kappa)`, `ame_d` of length n_coef + 1 with the reference bin first (0)
    when n_coef > 1, of length 1 (per unit of ln d) when n_coef = 1.
    """
    beta = np.asarray(beta, float)
    linear_index = _index_no_fe(sample, beta, n_coef)
    group_effect = calibrate_cloglog_fe(sample, linear_index)[sample["group"].to_numpy()]
    eta = group_effect + linear_index
    ame_kappa = float(np.mean(_cloglog_f(eta)) * beta[-1])
    if n_coef == 1:
        return np.array([np.mean(_cloglog_f(eta)) * beta[0]]), ame_kappa
    eta0 = group_effect + _index_no_fe(sample, beta, n_coef, drop_distance=True)
    F_reference = _cloglog_F(eta0)
    ame_distance = np.array([0.0] + [np.mean(_cloglog_F(eta0 + beta_bin) - F_reference) for beta_bin in beta[:n_coef]])
    return ame_distance, ame_kappa


def extensive_ame_table(data, quantiles=WORLD_Q, sample=None):
    """
    Block 4 as average marginal effects in PERCENTAGE POINTS, data against simulated,
    with the fake-world band: the AME of each world's `beta_sim + e_b` (a_g re-solved
    world by world), then its quantiles -- the band of `reg_coef_world_table` pushed
    through the non-linear map draw by draw, not its end points.

    Rows: the reference bin (0 by construction) and the distance bins, then ln n_ls.
    `.attrs`: the data's supply rate in the reference bin, the sample size, the number
    of groups and worlds.
    """
    sample_cells = reg_cell_sample(data) if sample is None else sample
    n_distance = int(data["n_coef"])
    emp_beta = np.asarray(data["empirical_moments_dict"]["reg_coef"], float)
    sim_beta = _sim_block4(data)
    emp_ame_d, emp_ame_kappa = extensive_ame(sample_cells, emp_beta, n_distance)
    sim_ame_d, sim_ame_kappa = extensive_ame(sample_cells, sim_beta, n_distance)
    emp_ame = 100 * np.append(emp_ame_d, emp_ame_kappa)
    sim_ame = 100 * np.append(sim_ame_d, sim_ame_kappa)
    labels = _ame_labels(n_distance)
    df = pd.DataFrame({"data": emp_ame, "simulated": sim_ame},
                      index=pd.Index(labels, name="AME (pp)"))
    world_residuals = data.get("world_residuals")
    n_worlds = 0
    if world_residuals is not None:
        residual_draws = np.asarray(world_residuals["e"], float)
        residual_draws = residual_draws[np.isfinite(residual_draws).all(axis=1)]
        n_worlds = residual_draws.shape[0]
        draws = np.array([100 * np.append(*extensive_ame(sample_cells, sim_beta + residual_draw, n_distance)) for residual_draw in residual_draws])
        df["band_lo"] = np.quantile(draws, quantiles[0], axis=0)
        df["band_hi"] = np.quantile(draws, quantiles[1], axis=0)
        df["inside_band"] = (df["data"] >= df["band_lo"]) & (df["data"] <= df["band_hi"])
        if n_distance > 1:
            df.iloc[0, df.columns.get_loc("inside_band")] = True
    reference_rate = (sample_cells.loc[sample_cells["bin"] == 0, "y"].mean() if n_distance > 1 else np.nan)
    df.attrs.update(ref_rate=float(reference_rate), n_cells=len(sample_cells),
                    n_groups=int(sample_cells["group"].nunique()), n_worlds=n_worlds, q=quantiles)
    return df


def _ame_labels(n_coef):
    dist = reg_bin_labels(n_coef)
    reference_label = {4: [r"$]0,50]$"], 5: [r"$]0,20]$"]}.get(n_coef, [r"ref."])
    return (dist if n_coef == 1 else reference_label + dist) + [r"$\ln n_{ls}$"]


def plot_extensive_ame(data, table=None, density=False, save_to=None, quantiles=WORLD_Q):
    """
    Block 4 as Fig. 4, Panel A reads it: the effect of each distance bin on the
    probability that a cell hosts a supplier, in percentage points RELATIVE TO the
    reference bin, which stays on the axis at zero (dashed line). Empirical (red) against
    simulated (blue) with the 95% fake-world band. The default is the distance panel
    alone, as in the paper's Fig. 5; `density=True` adds a narrow panel with the AME of
    ln n_ls (pp per unit of ln n). The reference-bin supply rate of the
    data is in `table.attrs["ref_rate"]` and belongs in the caption.
    """
    ame_table = extensive_ame_table(data, quantiles=quantiles) if table is None else table
    n_distance = int(data["n_coef"])
    labels = list(ame_table.index)
    has_band = "band_lo" in ame_table
    band = (ame_table["band_lo"].to_numpy(), ame_table["band_hi"].to_numpy()) if has_band else None
    sim_label = "Simulated" + ("" if not has_band
                               else rf" (95\% band)" if mpl.rcParams.get("text.usetex")
                               else " (95% band)")
    n_distance_ticks = n_distance + (1 if n_distance > 1 else 0)
    if density:
        fig, panel_axes = plt.subplots(1, 2, figsize=get_figsize(height_fraction=0.62),
                                gridspec_kw={"width_ratios": [n_distance_ticks + 0.6, 1.6]})
    else:
        fig, single_ax = plt.subplots(figsize=get_figsize(height_fraction=0.62))
        panel_axes = [single_ax]
    panel_slices = [slice(0, n_distance_ticks), slice(n_distance_ticks, n_distance_ticks + 1)]
    for k, panel_ax in enumerate(panel_axes):
        panel_slice = panel_slices[k]
        _paired_errorbar(panel_ax, labels[panel_slice], ame_table["data"].to_numpy()[panel_slice],
                         ame_table["simulated"].to_numpy()[panel_slice],
                         band_sim=None if band is None else (band[0][panel_slice], band[1][panel_slice]),
                         ylabel=("Supply probability\ndifference (in pp)" if k == 0
                                 else "pp per unit of ln n"),
                         xlabel=("Distance" if n_distance > 1 else "") if k == 0 else "Density",
                         sim_label=sim_label, legend=(k == 0))
        panel_ax.axhline(0, color="black", ls="dashed", lw=0.8, zorder=0)
        if k == 0 and n_distance_ticks >= 5:
            # five bin labels plus the reference do not fit at the default size
            panel_ax.tick_params(axis="x", labelsize=0.75 * mpl.rcParams["xtick.labelsize"]
                          if isinstance(mpl.rcParams["xtick.labelsize"], (int, float))
                          else "small")
    fig.tight_layout()
    if has_band and not ame_table["inside_band"].all():
        print(f"[plot_extensive_ame] {data['industry']}: data outside the 95% simulation "
              f"band for {list(ame_table.index[~ame_table['inside_band']])}.")
    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        fig.savefig(save_to, format="pdf", bbox_inches="tight")
    return panel_axes


def g_curve_world_band(data, K_values=(0,), quantiles=WORLD_Q):
    """
    (lo, hi) arrays (S, |K|) for the simulated count curve, from the variety counts
    re-calibrated in each fake world.

    Each world re-calibrates N_s on its own empty-cell share; pushing the B counts
    through the closed form at the estimate's win probabilities q_hat gives the spread
    of the fitted curve that the VARIETY-COUNT channel carries. It holds q_hat at
    theta_hat (the worlds' own (alpha, kappa, T) are not re-simulated here), so it is a
    lower bound on the full band. None when the worlds or the diagnostics are absent.
    """
    N_worlds = _n_hat_world(data)
    diag = data.get("granular_diagnostics")
    if N_worlds is None or diag is None or "q_hat" not in diag:
        return None
    K_list = list(K_values)
    q_hat = np.asarray(diag["q_hat"], float).ravel()
    N_HI = np.asarray(diag["N_HI"], float).ravel() if "N_HI" in diag else np.array([100.0])
    n_rho = max(100, int(round(np.nanmax(N_HI))))
    k_counts = np.rint(q_hat * n_rho).astype(int)
    log_factorial = _logfact(n_rho)
    good_s = _cell_sectors(data)
    S = data["S"]
    lo, hi = np.full((S, len(K_list)), np.nan), np.full((S, len(K_list)), np.nan)
    for s in range(S):
        cells = good_s == s
        if not cells.any():
            continue
        curve_by_N = {}
        for N_value in np.unique(N_worlds[:, s]):
            curve_by_N[N_value] = [gbar_cells(k_counts[cells], n_rho, N_value, K, log_factorial).mean() for K in K_list]
        draws = np.array([curve_by_N[N_value] for N_value in N_worlds[:, s]])          # (B, |K|)
        lo[s], hi[s] = np.quantile(draws, quantiles[0], axis=0), np.quantile(draws, quantiles[1], axis=0)
    return lo, hi


def plot_G0(data, ax=None, save_to=None, z_critical=1.96, annotate_N=True, quantiles=WORLD_Q):
    """
    Empirical vs simulated Gbar_s(0) -- the share of cells with no supplier.

    The simulated point carries the 95% band of `g_curve_world_band` (variety-count
    channel); each sector is annotated with N_hat_s and, when the fake worlds are on
    disk, its 95% interval across worlds.
    """
    if not data["granular"]:
        raise ValueError("Gbar_s(0) only exists under --granular=true")
    emp_G0 = np.asarray(data["empirical_moments_dict"]["G0"], float)
    sim_G0 = np.asarray(data["simulated_moments_dict"]["G0"], float)
    world_band = g_curve_world_band(data, K_values=(0,), quantiles=quantiles)
    band = None if world_band is None else (world_band[0][:, 0], world_band[1][:, 0])

    fig, ax = (plt.subplots(figsize=get_figsize(height_fraction=0.62)) if ax is None else (ax.figure, ax))
    _paired_errorbar(ax, data["sector_names"], emp_G0, sim_G0, z_critical=z_critical, band_sim=band,
                     ylabel=r"$\bar{G}_s(0)$", rotate=45,
                     sim_label="Simulated" if band is None else
                     ("Simulated (95\\% band)" if mpl.rcParams.get("text.usetex")
                      else "Simulated (95% band)"), legend=False)
    # Legend OUTSIDE the axes, below the rotated tick labels, one row, small: inside the
    # axes it sits on the low-G sectors. Anchored to the measured bottom of the x-axis
    # (ticks + labels) rather than a fixed offset, so it clears the labels at any font.
    _legend_below_xaxis(ax)

    # The profiled variety count is what the model moves to hit this moment, so
    # print it next to each sector — a clamped sector cannot close its own residual.
    diag = data.get("granular_diagnostics")
    if annotate_N and diag is not None and "N_hat" in diag:
        N_hat = np.asarray(diag["N_hat"]).ravel()
        clamped = np.asarray(diag.get("clamped", np.zeros_like(N_hat))).ravel()
        N_worlds = _n_hat_world(data)
        y_top = ax.get_ylim()[1]
        for i, (n, clamp_state) in enumerate(zip(N_hat, clamped)):
            annotation = rf"$\hat{{N}}_s={int(n)}$" + ("*" if clamp_state != 0 else "")
            if N_worlds is not None:
                N_world_lo, N_world_hi = np.quantile(N_worlds[:, i], quantiles)
                annotation += rf" $[{int(N_world_lo)},{int(N_world_hi)}]$"
            ax.text(i, y_top, annotation, ha="center", va="bottom", fontsize=fs(12), rotation=45)
    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        ax.figure.savefig(save_to, format="pdf", bbox_inches="tight")

    # The same two numbers are printed by Julia's own reporting; a gap between the
    # figure and that report is a wiring fault, not a modelling result, so say so
    # rather than leaving it to be discovered by eye.
    consistency = g0_consistency(data, verbose=False)
    if consistency is not None and not bool(consistency.attrs.get("agrees", True)):
        print(f"[plot_G0] {data['industry']}: the figure and "
              f"{data['inference_step']}/granular_diagnostics.npz disagree on "
              "Gbar_s(0) — run g0_consistency(data) for the per-sector table.")
    return ax


def plot_parameter_ci(data, save_to=None, quantiles=WORLD_Q):
    """
    (alpha, kappa) and N_hat_s with their 95% fake-world intervals.

    Left: the trade-cost elasticity (one per segment under binned tau) and the density
    elasticity kappa -- the two parameters the search runs over. Right: the variety count
    of each sector, re-calibrated in every world (its integer jumps are part of the
    distribution). Points are the estimate, bars the 2.5/97.5% quantiles across worlds,
    open circles the world mean (its distance to the point is the bias).
    """
    world_table = world_param_table(data, quantiles=quantiles)
    if world_table is None:
        raise FileNotFoundError(
            f"no {data['inference_step']}/inference_world/world_inference_draws.npz under "
            f"{data['folder']} — run the inference step of main.jl.")
    head = world_table[[_is_trade_param(l) for l in world_table.index]]
    N_table = world_table.loc[[f"N_s[{s}]" for s in data["sector_names"]]]

    fig, panel_axes = plt.subplots(1, 2, figsize=get_figsize(height_fraction=0.55),
                            gridspec_kw={"width_ratios": [len(head) + 0.6, len(N_table) + 0.6]})

    def _draw(a, df, labels, ylabel):
        x = np.arange(len(df), dtype=float)
        v = df["estimate"].values
        a.errorbar(x, v, yerr=np.vstack([np.maximum(v - df["q_lo"].values, 0),
                                         np.maximum(df["q_hi"].values - v, 0)]),
                   fmt="o", color=toulouse_color, capsize=3, markersize=7,
                   markeredgecolor="black", markeredgewidth=0.5, linestyle="none")
        a.plot(x, df["mean"].values, "o", mfc="none", mec="black", markersize=8)
        a.set_xticks(x)
        a.set_xticklabels(labels, rotation=45 if len(df) > 4 else 0,
                          ha="right" if len(df) > 4 else "center")
        a.set_xlim(-0.6, len(df) - 0.4)
        a.set_ylabel(ylabel)
        a.grid(axis="y", linestyle="dashed", alpha=0.5)
        _despine(a)

    head_labels = [(r"$\alpha$" if l == "alpha[1]" and data["n_tau"] == 1 else
           rf"$\alpha_{{{l[6:-1]}}}$" if l.startswith("alpha") else r"$\kappa$")
          for l in head.index]
    _draw(panel_axes[0], head, head_labels, "Estimate")
    panel_axes[0].axhline(0, color="black", linewidth=0.6, alpha=0.5)
    _draw(panel_axes[1], N_table, data["sector_names"], r"$\hat N_s$")
    fig.tight_layout()
    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        fig.savefig(save_to, format="pdf", bbox_inches="tight")
    return panel_axes


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
        source_text = (f"{data['folder'].name}: moments {data['step_dir']}/best_simulated_moments.npy "
               f"vs {data['inference_step']}/granular_diagnostics.npz")
        if agrees:
            print(f"[g0_consistency] {data['industry']} (mu = {data['mu']}) OK — {source_text}; "
                  f"max |gap| target {worst_t:.2e}, fit {worst_f:.2e}")
        else:
            print(f"[g0_consistency] {data['industry']} (mu = {data['mu']}) MISMATCH — {source_text}\n"
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
    log_factorial = np.zeros(int(m) + 1)
    if m >= 1:
        log_factorial[1:] = np.cumsum(np.log(np.arange(1, int(m) + 1, dtype=float)))
    return log_factorial


def _log_binom(a, b, log_factorial):
    return log_factorial[a] - log_factorial[b] - log_factorial[a - b]


def gbar_cells(k, m, n, K, log_factorial):
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
    pmf = np.zeros(k.shape, float)
    if K < 0 or K > n:
        return pmf
    ok = (K <= k) & (n - K <= m - k)
    if ok.any():
        k_ok = k[ok]
        pmf[ok] = np.exp(_log_binom(k_ok, K, log_factorial)
                         + _log_binom(m - k_ok, n - K, log_factorial) - _log_binom(m, n, log_factorial))
    return pmf


def _plugin_cells(q_hat, n, K):
    """Plug-in fallback: the Binomial(n, q_hat) pmf at exactly K. Biased (Jensen), robust."""
    q_hat = np.clip(np.asarray(q_hat, float), 0.0, 1.0)
    n, K = int(n), int(K)
    if K < 0 or K > n:
        return np.zeros(q_hat.shape, float)
    log_factorial = _logfact(max(n, 1))
    return float(np.exp(_log_binom(n, K, log_factorial))) * q_hat ** K * (1 - q_hat) ** (n - K)


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

    * simulated  -- rebuilt here from `q_hat` and `N_hat_s` under the same unbiased convention,
      with `m = N_rho` RECOVERED from the diagnostics (`max(100, max N_HI)`, the rule in
      load_parameters.jl) and the recovery CHECKED twice: `q_hat * m` must be integral
      (it is a win count), and the K = 0 column must reproduce block 6 of the moment
      vector. If either fails the plug-in Binomial pmf is used and `attrs["estimator"]`
      says so.
    * empirical  -- `G` of `G_K.csv`, the share of cells with EXACTLY K suppliers (read
      as is: the data pipeline writes the pmf, `G_cs` being its cumulative sum).

    The simulated curve carries the 95% band of `g_curve_world_band` (`sim_lo`,
    `sim_hi`); the empirical one carries no interval.

    Returns a (sector, K)-indexed DataFrame;
    `df.attrs` carries the estimator, the K = 0 gap against the moment vector, and the
    draw count used.
    """
    if not data["granular"]:
        raise ValueError("the count curve only exists under --granular=true")
    K_list = list(_DEFAULT_COUNT_K if K_values is None else K_values)
    S = data["S"]
    diag = data.get("granular_diagnostics")
    if diag is None or "q_hat" not in diag or "N_hat" not in diag:
        raise FileNotFoundError(
            f"{data['folder'].name}/{data['inference_step']}/granular_diagnostics.npz is "
            "absent or carries no q_hat/N_hat — the simulated curve is rebuilt from those "
            "two, so there is nothing to plot against. Re-run the step that writes it "
            f"({'Step 2' if data['mu'] == 1 else 'Step 4'} of main.jl).")

    q_hat = np.asarray(diag["q_hat"], float).ravel()
    N_hat = np.rint(np.asarray(diag["N_hat"], float).ravel()).astype(int)
    N_HI = np.asarray(diag["N_HI"], float).ravel() if "N_HI" in diag else np.array([100.0])
    clamped = (np.asarray(diag["clamped"], float).ravel() if "clamped" in diag
               else np.zeros(S))
    good_s = _cell_sectors(data)
    if q_hat.size != good_s.size:
        raise ValueError(f"q_hat has {q_hat.size} cells but CELL_MASK has {good_s.size} — "
                         "the run's granular/ca_level flags and the ones passed here disagree.")

    n_rho = max(100, int(round(np.nanmax(N_HI))))        # load_parameters.jl's N_rho rule
    k_counts = np.rint(q_hat * n_rho).astype(int)
    integral_gap = float(np.max(np.abs(q_hat * n_rho - k_counts))) if q_hat.size else 0.0
    log_factorial = _logfact(n_rho)

    def curve(unbiased):
        out = np.full((S, len(K_list)), np.nan)
        for s in range(S):
            cells = good_s == s
            if not cells.any():
                continue
            for i, K in enumerate(K_list):
                out[s, i] = (gbar_cells(k_counts[cells], n_rho, N_hat[s], K, log_factorial).mean()
                             if unbiased else _plugin_cells(q_hat[cells], N_hat[s], K).mean())
        return out

    block6 = np.asarray(data["simulated_moments_dict"]["G0"], float)
    sim_curve = curve(True)
    estimator, gap = "unbiased (gbar_cell)", np.inf
    if 0 in K_list:
        gap = float(np.nanmax(np.abs(sim_curve[:, K_list.index(0)] - block6)))
    if integral_gap > 1e-6 or (0 in K_list and gap > 1e-6):
        sim_pi = curve(False)
        gap_pi = (float(np.nanmax(np.abs(sim_pi[:, K_list.index(0)] - block6)))
                  if 0 in K_list else np.inf)
        print(f"[count_curve] {data['industry']}: the unbiased reconstruction does not "
              f"reproduce block 6 at K = 0 (gap {gap:.2e}, q_hat*N_rho integrality "
              f"{integral_gap:.2e} at N_rho = {n_rho}); falling back to the plug-in "
              f"Binomial pmf (gap {gap_pi:.2e}). K >= 1 is then biased by Jensen "
              "— read it as indicative, and check N_rho against the run's log line.")
        sim_curve, estimator, gap = sim_pi, "plug-in Binomial(N_hat, q_hat)", gap_pi

    # The fake-world band of the simulated curve (variety-count channel).
    world_band = g_curve_world_band(data, K_values=K_list)
    sim_lo, sim_hi = (np.full((S, len(K_list)), np.nan),) * 2 if world_band is None else world_band

    # ---- empirical: the shares of G_K.csv (exactly K suppliers), no interval ----
    emp_curve = np.full((S, len(K_list)), np.nan)
    K_in_file, pmf_in_file = data.get("G_curve_K"), data.get("G_pmf")
    if pmf_in_file is not None and K_in_file is not None:
        for i, K in enumerate(K_list):
            if K in K_in_file:
                emp_curve[:, i] = pmf_in_file[:, K_in_file.index(K)]
    missing = [K for K in K_list if K_in_file is not None and K not in K_in_file]
    if missing:
        print(f"[count_curve] {data['industry']}: G_K.csv has no row for K = {missing} — "
              "those panels show the simulated curve only.")

    sector_K_index = pd.MultiIndex.from_product([data["sector_names"], K_list], names=["A129", "K"])
    df = pd.DataFrame({"empirical": emp_curve.ravel(),
                       "simulated": sim_curve.ravel(),
                       "sim_lo": sim_lo.ravel(), "sim_hi": sim_hi.ravel(),
                       "residual": (sim_curve - emp_curve).ravel(),
                       "N_hat": np.repeat(N_hat, len(K_list)),
                       "clamped": np.repeat(clamped != 0, len(K_list)),
                       "targeted": np.tile([K == 0 for K in K_list], S)}, index=sector_K_index)
    df.attrs.update(estimator=estimator, K0_gap_vs_moment_vector=gap,
                    n_rho=n_rho, K_values=K_list)
    return df


def plot_count_curve(data, K_values=None, save_to=None, z_critical=1.96, df=None):
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

    One band, on the SIMULATED point only: the 2.5/97.5% quantiles of the fitted curve
    across the simulated economies of the inference step, through the variety count each
    re-calibrates (q_hat held at the estimate; `g_curve_world_band`). The empirical
    points carry NO interval (the data side has no model-based error). `z_critical` is unused
    and kept so existing calls parse.

    `save_to` is a STEM, not a file: K is appended before the extension, so
    `.../count_curve_auto_mu2.pdf` writes `..._K0.pdf`, `..._K1.pdf`, ... One call still
    produces the whole set, and nothing overwrites anything.

    `K_values` selects which K to draw; passed alongside a precomputed `df` it SUBSETS
    it, which is how a run cell draws only the untargeted K >= 1 while `plot_G0` keeps
    the targeted one (with its N_hat annotation). Returns `{K: ax}`.
    """
    df = count_curve(data, K_values=K_values) if df is None else df
    available = list(df.attrs["K_values"])
    K_list = available if K_values is None else [int(K) for K in np.atleast_1d(K_values)]
    absent = [K for K in K_list if K not in available]
    if absent:
        raise KeyError(f"K = {absent} is not in the count curve passed here "
                       f"(it carries K = {available}) — rebuild it with "
                       f"`count_curve(data, K_values={K_list})`.")
    sectors = data["sector_names"]

    stem, ext = os.path.splitext(save_to) if save_to else (None, ".pdf")
    axes = {}
    for K in K_list:
        curve_K = df.xs(K, level="K").reindex(sectors)
        _, ax = plt.subplots(figsize=get_figsize(height_fraction=0.62))
        band = ((curve_K["sim_lo"].values, curve_K["sim_hi"].values)
                if np.isfinite(curve_K["sim_lo"].values).any() else None)
        _paired_errorbar(
            ax, sectors, curve_K["empirical"].values, curve_K["simulated"].values,
            None, None, z_critical=z_critical, band_sim=band,          # no interval on the data point
            ylabel=rf"$\bar{{G}}_s({K})$", rotate=45)
        ax.figure.tight_layout()
        if stem:
            path = f"{stem}_K{K}{ext or '.pdf'}"
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            ax.figure.savefig(path, format="pdf", bbox_inches="tight")
        axes[K] = ax
    return axes


# ============================================================================
# Moment sensitivity -- one-parameter profiles (Catherine et al., Figure D.1)
# ============================================================================

def _sens(data):
    sens = data.get("sensitivity")
    if sens is None or not sens.get("profiles"):
        raise FileNotFoundError(
            f"no {data['inference_step']}/sensitivity/ profiles under {data['folder']} -- "
            "main.jl writes them at the end of Step 2 (theta_hat_1) and Step 4 "
            "(theta_hat_2) unless SENS=0; load with parts including 'sensitivity'.")
    return sens


def _moment_block_of(data, i):
    """(block index 0..5, position inside the block) of masked moment `i`."""
    block_edges = np.cumsum([0] + list(data["moment_block_sizes"]))
    block = int(np.searchsorted(block_edges, i, side="right") - 1)
    return block, i - block_edges[block]


def world_data_band(data, i, quantiles=(0.05, 0.95)):
    """
    The 5/95% band of a DATA moment across the Step-2 fake worlds -- the counterpart of
    Figure D.1's bootstrap band: where one finite economy drawn at theta_hat_1 puts that
    moment. Available for block 4 (`beta_world`) and block 6 (`G_world`); None otherwise.
    """
    world_residuals = data.get("world_residuals")
    if world_residuals is None:
        return None
    block, position = _moment_block_of(data, i)
    key = {3: "beta_world", 5: "G_world"}.get(block)
    if key is None or key not in world_residuals:
        return None
    draws = np.asarray(world_residuals[key], float)[:, position]
    draws = draws[np.isfinite(draws)]
    return None if draws.size < 2 else tuple(np.quantile(draws, quantiles))


def _moment_tex(label):
    if label.startswith("beta_d["):
        return rf"$\beta_d$ {label[7:-1]}"
    if label == "beta_kappa":
        return r"$\beta_\kappa$"
    if label.startswith("G0["):
        return rf"$\bar G_s(0)$, {label[3:-1]}"
    return label.replace("_", r"\_")


def _param_tex(label):
    if label == "alpha":
        return r"$\alpha$"
    if label.startswith("alpha"):
        return rf"$\alpha_{{{label.split('_')[-1]}}}$"
    if label == "kappa":
        return r"$\kappa$"
    if label.startswith("N_s["):
        return rf"$N_s$, {label[4:-1]}"
    return label.replace("_", r"\_")


def plot_sensitivity_profiles(data, params=None, moments=None, save_to=None, band_sd=2.0,
                              ncols=None):
    """
    One figure per swept parameter, one panel per moment: the model moment against the
    parameter value, everything else at the estimate (Figure D.1).

    * solid line: the moment on the ESTIMATION draws -- the curve the optimiser saw;
    * light band: mean +/- `band_sd` sd across the K independent draw sets -- the
      Monte-Carlo noise of the curve (a jagged solid line inside a smooth band is
      simulation noise, a jump of the band itself is an integer jump of N_hat_s);
    * red vertical line: the estimate; black horizontal line: the data moment, with the
      5/95% band of that moment across the Step-2 fake worlds in grey when available.

    For alpha and kappa, T is re-inverted and N_hat_s re-calibrated at every grid point,
    as in estimation; for N_s the other sectors' counts are pinned. `moments` overrides
    the default pairing (alpha -> the distance coefficients, kappa -> beta_kappa,
    N_s -> its own Gbar_s(0)) with a list of moment labels. `save_to` is a stem: the
    parameter is appended. Returns {parameter: array of axes}.
    """
    sens = _sens(data)
    labels = data["moment_labels"]
    wanted = list(sens["profiles"]) if params is None else list(params)
    axes_by_param = {}
    stem, ext = os.path.splitext(save_to) if save_to else (None, ".pdf")
    for param in wanted:
        profile = sens["profiles"][param]
        moment_indices = (list(profile["targets"]) if moments is None
               else [labels.index(m) for m in moments])
        n = len(moment_indices)
        n_cols = ncols or min(n, 3)
        n_rows = int(np.ceil(n / n_cols))
        fig_width, fig_height = get_figsize(height_fraction=0.62)
        fig, panel_axes = plt.subplots(n_rows, n_cols, figsize=(fig_width * min(n_cols, 2) / 1.3, fig_height * n_rows / 1.1 + 0.2),
                                squeeze=False)
        grid = np.asarray(profile["grid"], float)
        for panel_ax, i in zip(panel_axes.ravel(), moment_indices):
            ok = np.asarray(profile["converged"], float) > 0
            model_curve = np.asarray(profile["m_est"], float)[:, i]
            mu, sd = np.asarray(profile["m_mean"], float)[:, i], np.asarray(profile["m_sd"], float)[:, i]
            finite = np.isfinite(mu) & np.isfinite(sd)
            if finite.any():
                panel_ax.fill_between(grid[finite], (mu - band_sd * sd)[finite], (mu + band_sd * sd)[finite],
                               color=sim_color, alpha=0.18, linewidth=0)
            panel_ax.plot(grid[ok], model_curve[ok], color=sim_color, marker="o", markersize=3, linewidth=1.4)
            data_moment = float(np.asarray(profile["data"], float)[i])
            band = world_data_band(data, i)
            if band is not None:
                panel_ax.axhspan(band[0], band[1], color="0.6", alpha=0.3, linewidth=0)
            panel_ax.axhline(data_moment, color="black", linewidth=1.2)
            panel_ax.axvline(profile["estimate"], color="red", linewidth=1.2)
            panel_ax.set_xlabel(_param_tex(param))
            panel_ax.set_ylabel(_moment_tex(labels[i]))
            panel_ax.grid(linestyle="dashed", alpha=0.4)
            _despine(panel_ax)
        for panel_ax in panel_axes.ravel()[n:]:
            panel_ax.set_visible(False)
        fig.tight_layout()
        if stem:
            path = f"{stem}_{re.sub(r'[^A-Za-z0-9]+', '_', param).strip('_')}{ext or '.pdf'}"
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            fig.savefig(path, format="pdf", bbox_inches="tight")
        axes_by_param[param] = panel_axes
    return axes_by_param


def sensitivity_profile_table(data):
    """
    One row per (swept parameter, target moment): the slope at the estimate (central
    difference on the grid, estimation draws), the share of grid steps moving in that
    direction (`monotone`), whether the model curve crosses the data moment inside the
    grid, and the noise of the curve at the estimate (sd across draw sets) against its
    range over the grid (`noise_to_range`; near one, the profile is noise).
    """
    sens = _sens(data)
    rows = []
    for param, profile in sens["profiles"].items():
        grid = np.asarray(profile["grid"], float)
        estimate_position = int(np.argmin(np.abs(grid - profile["estimate"])))
        for i in profile["targets"]:
            model_curve = np.asarray(profile["m_est"], float)[:, i]
            sd = np.asarray(profile["m_sd"], float)[:, i]
            ok = np.isfinite(model_curve)
            slope = ((model_curve[estimate_position + 1] - model_curve[estimate_position - 1]) / (grid[estimate_position + 1] - grid[estimate_position - 1])
                     if 0 < estimate_position < grid.size - 1 else np.nan)
            steps = np.diff(model_curve[ok])
            monotone_share = (np.mean(np.sign(steps) == np.sign(slope))
                    if steps.size and np.isfinite(slope) else np.nan)
            gap_to_data = model_curve[ok] - float(profile["data"][i])
            curve_range = np.nanmax(model_curve) - np.nanmin(model_curve) if ok.any() else np.nan
            rows.append({"parameter": param, "moment": data["moment_labels"][i],
                         "estimate": profile["estimate"], "model": model_curve[estimate_position],
                         "data": float(profile["data"][i]), "slope": slope, "monotone": monotone_share,
                         "crosses_data": bool(gap_to_data.size > 1 and np.any(gap_to_data[:-1] * gap_to_data[1:] <= 0)),
                         "noise_to_range": (sd[estimate_position] / curve_range if curve_range and curve_range > 0 else np.nan)})
    return pd.DataFrame(rows).set_index(["parameter", "moment"])


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
            f"no Jacobian in {data['folder']}/{step}/sensitivity/ — sensitivity.jl writes "
            "jacobian.npz at theta_hat_1 (step2) and theta_hat_2 (step3) unless SENS=0; "
            "load with parts including 'sensitivity'.")
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
    elasticities = np.asarray(jacobian_matrix(data, kind), float)
    sd = data.get("J_elast_sd" if "elast" in kind else "J_sd")
    if sd is None:
        raise FileNotFoundError(
            f"no across-replication SD for the Jacobian in {data['folder']}/"
            f"{data['inference_step']}/ — compute_jacobian writes it as "
            "jacobian_all*_elasticity_sd.npy when K > 1.")
    sd = np.abs(np.asarray(sd, float))
    abs_elasticities = np.abs(elasticities)
    with np.errstate(divide="ignore", invalid="ignore"):
        noise_ratio = sd / abs_elasticities
    noise_ratio[(abs_elasticities == 0) & (sd == 0)] = 0.0          # structural zero: measured, and exactly zero
    noise_ratio[~np.isfinite(noise_ratio)] = np.inf            # nonzero noise on a zero signal: unreadable
    return noise_ratio


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
    x_ticks = np.arange(shape[1])[::_tick_every(shape[1])]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(np.array(data["param_labels"])[x_ticks], rotation=90, fontsize=fs(7))
    ax.xaxis.set_ticks_position("bottom")
    y_ticks = np.arange(shape[0])[::_tick_every(shape[0])]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(np.array(data["moment_labels"])[y_ticks], fontsize=fs(7))
    ax.set_xlabel("Parameter")
    ax.set_ylabel("Moment")


def _sector_subblocks(data, block_name="Regional sourcing shares"):
    """
    The (start, stop, sector) runs INSIDE one moment block, read off the labels.

    The gamma rows are labelled `gamma[<sector>-AA<area>]` and stored s-major, so the
    sector is the text between "gamma[" and the LAST "-AA" and its rows are contiguous;
    an importing sector's run ends with its import share, `f_foreign[<sector>]`.
    Returns absolute row indices, so the caller can draw on the full matrix without
    knowing where the block starts. An empty list when the block is absent or its labels
    do not carry a sector, which is the right behaviour for every other block.
    """
    names = list(data["moment_block_names"])
    if block_name not in names:
        return []
    block_edges = _edges(data["moment_block_sizes"])
    lo, hi = int(block_edges[names.index(block_name)]), int(block_edges[names.index(block_name) + 1])
    labels = list(data["moment_labels"])[lo:hi]

    def _sector(label):
        if label.startswith("f_foreign[") and label.endswith("]"):   # the import share, last row
            return label[len("f_foreign["):-1]
        if not label.startswith("gamma[") or "-AA" not in label:
            return None
        return label[len("gamma["):label.rfind("-AA")]

    runs, start, current_sector = [], 0, _sector(labels[0]) if labels else None
    if current_sector is None:
        return []
    for i, label in enumerate(labels[1:] + [None], start=1):
        s = _sector(label) if label is not None else object()
        if s != current_sector:
            runs.append((lo + start, lo + i, current_sector))
            start, current_sector = i, s
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
    shown_matrix = np.ma.masked_where(np.asarray(mask, bool), J) if mask is not None else J
    m_edges, p_edges = _edges(data["moment_block_sizes"]), _edges(data["param_block_sizes"])

    fig, ax = plt.subplots(figsize=figsize)
    image = ax.matshow(shown_matrix, cmap=cmap, norm=norm, aspect="auto")
    _block_grid(ax, data, m_edges, p_edges)
    _axis_labels(ax, data, J.shape)

    if title is None:
        title = (f"Jacobian ({kind}) — {data['industry']}, "
                 rf"$\hat{{\mu}}_{data['mu']}$   ({J.shape[0]}$\times${J.shape[1]})")
        if mask is not None:
            kept = float((~np.asarray(mask, bool)).mean())
            title += f"\nnoise-purged: {100 * kept:.1f}% of entries readable"
    ax.set_title(title, pad=28)
    cbar = fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.04, pad=0.22,
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
    noise_ratio = jacobian_noise_ratio(data, kind)
    unreadable = noise_ratio >= noise_max
    m_edges, p_edges = _edges(data["moment_block_sizes"]), _edges(data["param_block_sizes"])

    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_over("black")
    image = ax.matshow(np.clip(noise_ratio, 0, noise_max * 2), cmap=cmap, aspect="auto",
                    vmin=0, vmax=noise_max * 2)
    # mark the failures explicitly, so a reader is not left inferring them from a colour
    y_unreadable, x_unreadable = np.where(unreadable)
    ax.scatter(x_unreadable, y_unreadable, s=1.5, marker="s", color="black", alpha=0.35, linewidths=0)
    _block_grid(ax, data, m_edges, p_edges)
    _axis_labels(ax, data, noise_ratio.shape)

    share = float(unreadable.mean())
    ax.set_title(f"Jacobian noise-to-signal $\\sigma/|\\varepsilon|$ — "
                 f"{data['industry']}, " rf"$\hat{{\mu}}_{data['mu']}$"
                 f"\n{100 * share:.1f}{pct()} of entries at or above {noise_max:g} "
                 "(black squares) — not readable", pad=28)
    cbar = fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.04, pad=0.22,
                        extend="max")
    cbar.set_label(r"$\sigma_{jk} / |\varepsilon_{jk}|$ across the Jacobian replications"
                   f"   (threshold {noise_max:g})")
    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        fig.savefig(save_to, bbox_inches="tight")
    return ax


def plot_jacobian_triptych(data, kind="elasticity", noise_max=NOISE_MAX,
                           figsize=(15, 11), out_folder=None, file_tag="jacobian",
                           show=True):
    """
    The three panels together: the elasticity Jacobian, the noise-to-signal map, and the
    Jacobian purged of the entries the map rejects. Saves `<file_tag>`, `<file_tag>_noise` and
    `<file_tag>_purged` under `out_folder` when one is given.
    """
    def output_path(name):
        return None if out_folder is None else \
            f"{out_folder}/{name}_{data['industry']}_mu{data['mu']}.pdf"

    mask = jacobian_noise_mask(data, kind=kind, noise_max=noise_max)
    axes = [plot_jacobian_full(data, kind=kind, figsize=figsize, save_to=output_path(file_tag))]
    if show:
        plt.show()
    axes.append(plot_jacobian_noise(data, kind=kind, noise_max=noise_max,
                                    figsize=figsize, save_to=output_path(file_tag + "_noise")))
    if show:
        plt.show()
    axes.append(plot_jacobian_full(data, kind=kind, figsize=figsize, mask=mask,
                                   save_to=output_path(file_tag + "_purged")))
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
        image = ax.matshow(block, cmap=cmap, norm=norm, aspect="auto")
        for e in p_edges[1:-1]:
            ax.axvline(e - 0.5, color="black", linewidth=1.2, alpha=0.5)

        cbar = fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.05, pad=0.12)
        cbar.set_label("Elasticity", fontsize=fs(9))
        ax.set_title(f"{name}\n({block.shape[0]}x{block.shape[1]})",
                     fontsize=fs(13), fontweight="bold", pad=12)
        ax.set_xlabel("Parameter", fontsize=fs(10))

        x_ticks = np.arange(block.shape[1])[::_tick_every(block.shape[1], 20)]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(np.array(data["param_labels"])[x_ticks], rotation=90, fontsize=fs(7))
        ax.xaxis.set_ticks_position("bottom")

        row_labels = np.array(data["moment_labels"][m_edges[i]:m_edges[i + 1]])
        y_ticks = np.arange(block.shape[0])[::_tick_every(block.shape[0], 20)]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(row_labels[y_ticks], fontsize=fs(7))

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
    abs_elasticities = np.abs(np.asarray(jacobian_matrix(data, kind), float))
    try:
        readable = ~jacobian_noise_mask(data, kind=kind, noise_max=noise_max)
    except FileNotFoundError:
        readable = np.ones_like(abs_elasticities, dtype=bool)
    m_edges, p_edges = _edges(data["moment_block_sizes"]), _edges(data["param_block_sizes"])
    rows = []
    for i, mname in enumerate(data["moment_block_names"]):
        for j, pname in enumerate(data["param_block_names"]):
            block_slice = (slice(m_edges[i], m_edges[i + 1]), slice(p_edges[j], p_edges[j + 1]))
            block_values, block_readable = abs_elasticities[block_slice], readable[block_slice]
            rows.append({"moment_block": mname, "param_block": pname,
                         "mean_abs": block_values.mean() if block_values.size else np.nan,
                         "max_abs": block_values.max() if block_values.size else np.nan,
                         "share_readable": block_readable.mean() if block_values.size else np.nan})
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


def _cov_panel(ax, matrix, data, title, mode="cov", percentile=95):
    """One matrix panel with the block grid drawn on; `mode` is "cov" or "corr"."""
    from matplotlib.colors import TwoSlopeNorm
    edges, names = _gb_edges(data)
    if mode == "corr":
        sd_diagonal = np.sqrt(np.clip(np.diag(matrix), 1e-300, None))
        matrix = matrix / np.outer(sd_diagonal, sd_diagonal)
        image = ax.matshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    else:
        color_limit = np.percentile(np.abs(matrix), percentile)
        color_limit = color_limit if color_limit > 0 else (np.abs(matrix).max() or 1.0)
        image = ax.matshow(matrix, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-color_limit, vcenter=0, vmax=color_limit))
    for e in edges[1:-1]:
        ax.axhline(e - 0.5, color="black", linewidth=0.8, alpha=0.6)
        ax.axvline(e - 0.5, color="black", linewidth=0.8, alpha=0.6)
    ticks = (edges[:-1] + edges[1:]) / 2 - 0.5
    ax.set_xticks(ticks)
    ax.set_xticklabels(names, fontsize=fs(9))
    ax.set_yticks(ticks)
    ax.set_yticklabels(names, fontsize=fs(9), rotation=90, va="center")
    ax.set_title(title, pad=14)
    return image


def _cov_matrices(data):
    return [("$\\Sigma_{sim}$ (simulation noise)", data.get("Sigma_sim")),
            ("$\\Sigma_{data}$ (bootstrap)", data.get("Sigma_data")),
            ("$\\Omega = \\Sigma_{data} + \\Sigma_{sim}$", data.get("Omega"))]


def plot_variance_covariance(data, figsize=(16, 5.5), percentile=95, save_to=None):
    """Sigma_sim, Sigma_data and Omega side by side, each on its own diverging scale."""
    matrices = _cov_matrices(data)
    if all(m is None for _, m in matrices):
        raise FileNotFoundError(f"no Sigma_*/Omega in {data['folder']}/step2/ — "
                                "build_step3_weight_matrix writes them in Step 2.")
    fig, axes = plt.subplots(1, len(matrices), figsize=figsize)
    for ax, (title, matrix) in zip(np.atleast_1d(axes), matrices):
        if matrix is None:
            ax.axis("off")
            ax.set_title(title + "\n(absent)")
            continue
        image = _cov_panel(ax, matrix, data, title, mode="cov", percentile=percentile)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
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
    matrices = _cov_matrices(data)
    if all(m is None for _, m in matrices):
        raise FileNotFoundError(f"no Sigma_*/Omega in {data['folder']}/step2/")
    fig, axes = plt.subplots(1, len(matrices), figsize=figsize)
    for ax, (title, matrix) in zip(np.atleast_1d(axes), matrices):
        if matrix is None:
            ax.axis("off")
            ax.set_title(title + "\n(absent)")
            continue
        image = _cov_panel(ax, matrix, data, title, mode="corr")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
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
    summary = {}
    for name in ("Sigma_data", "Sigma_sim", "Omega"):
        matrix = data.get(name)
        if matrix is None:
            continue
        eigenvalues = np.linalg.eigvalsh((matrix + matrix.T) / 2)
        sd_diagonal = np.sqrt(np.clip(np.diag(matrix), 1e-300, None))
        correlation = matrix / np.outer(sd_diagonal, sd_diagonal)
        off_diagonal = ~np.eye(correlation.shape[0], dtype=bool)
        summary[name] = {"trace": float(np.trace(matrix)),
                     "eig_min": float(eigenvalues.min()), "eig_max": float(eigenvalues.max()),
                     "cond": float(eigenvalues.max() / eigenvalues.min()) if eigenvalues.min() > 0 else np.inf,
                     "mean_abs_offdiag_corr": float(np.abs(correlation[off_diagonal]).mean())}
    if "Sigma_sim" in summary and "Sigma_data" in summary and summary["Sigma_data"]["trace"] != 0:
        summary["Sigma_sim"]["share_of_data_trace"] = \
            summary["Sigma_sim"]["trace"] / summary["Sigma_data"]["trace"]
    return pd.DataFrame(summary).T


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


# ---------------------------------------------------------------------------
# The fake-world weighting of block 4 (Step 2)
# ---------------------------------------------------------------------------

def plot_world_covariance(data, figsize=None, save_to=None):
    """
    Omega_world = Cov(beta_world - beta_sim) over the Step-2 fake worlds, as a covariance
    (left, annotated) and a correlation (right). This is the whole weighting of the
    estimator: W_world = inv(Omega_world) on block 4, gamma and G being matched exactly.
    """
    from matplotlib.colors import TwoSlopeNorm
    Omega_world = data.get("Omega_world")
    if Omega_world is None:
        raise FileNotFoundError(f"no step2/Omega_world.npy under {data['folder']}")
    labels = reg_bin_labels(data["n_coef"], with_kappa=True)
    sd_diagonal = np.sqrt(np.clip(np.diag(Omega_world), 1e-300, None))
    correlation = Omega_world / np.outer(sd_diagonal, sd_diagonal)
    n_coefficients = Omega_world.shape[0]
    fig, panel_axes = plt.subplots(1, 2, figsize=figsize or get_figsize(height_fraction=0.5))
    color_limit = np.abs(Omega_world).max() or 1.0
    for panel_ax, matrix, norm, number_format in ((panel_axes[0], Omega_world, TwoSlopeNorm(vmin=-color_limit, vcenter=0, vmax=color_limit), "{:.2g}"),
                            (panel_axes[1], correlation, TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1), "{:.2f}")):
        panel_ax.matshow(matrix, cmap="RdBu_r", norm=norm)
        for i in range(n_coefficients):
            for j in range(n_coefficients):
                panel_ax.text(j, i, number_format.format(matrix[i, j]), ha="center", va="center", fontsize=fs(8))
        panel_ax.set_xticks(range(n_coefficients)); panel_ax.set_xticklabels(labels, rotation=45, ha="left", fontsize=fs(9))
        panel_ax.set_yticks(range(n_coefficients)); panel_ax.set_yticklabels(labels, fontsize=fs(9))
    panel_axes[0].set_xlabel(r"$\tilde\Omega$ (covariance)")
    panel_axes[1].set_xlabel("correlation")
    fig.tight_layout()
    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        fig.savefig(save_to, format="pdf", bbox_inches="tight")
    return panel_axes


def world_weight_summary(data):
    """
    `reg_coef_world_table` plus the global reading: J = e_data' W e_data at this step's
    simulated moments, against the distribution of e_b' W e_b over the world residuals.
    The worlds are drawn at theta_hat_1 and NOT re-estimated, so this is a reading of
    the fit at fixed parameters, not Hansen's J (which would re-estimate (alpha, kappa)
    in each world and lose N_TAU + 1 degrees of freedom). With as many block-4 entries as (alpha, kappa) the system
    is exactly identified, so J is ~0 at theta_hat_2 by construction and carries no test.
    """
    world_table = reg_coef_world_table(data)
    if world_table is None:
        return None
    n_block4 = len(world_table)
    exact = n_block4 == data["n_tau"] + 1
    print(f"{data['industry']} (mu = {data['mu']}): {world_table.attrs['n_worlds']} usable worlds; "
          f"J = {world_table.attrs['J']:.3g}, share of worlds with a larger quadratic form "
          f"{world_table.attrs['J_pvalue']:.3f}"
          + ("  [exactly identified: no over-identification test]" if exact else ""))
    return world_table


def _blocks_2d(data, matrix):
    """M split into (moment block x parameter block) sub-matrices."""
    m_edges, p_edges = _edges(data["moment_block_sizes"]), _edges(data["param_block_sizes"])
    return {(i, j): matrix[m_edges[i]:m_edges[i + 1], p_edges[j]:p_edges[j + 1]]
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
    abs_elasticities = np.abs(np.asarray(jacobian_matrix(data, kind), float))
    readable = _readable(data, kind, noise_max)
    rows = []
    for (i, j), block_values in _blocks_2d(data, abs_elasticities).items():
        block_readable = _blocks_2d(data, readable)[(i, j)]
        rows.append({
            "moment_block": data["moment_block_names"][i],
            "param_block": data["param_block_names"][j],
            "share_above": float((block_values >= threshold).mean()) if block_values.size else np.nan,
            "share_readable": float(block_readable.mean()) if block_values.size else np.nan,
            "share_live": float(((block_values >= threshold) & block_readable).mean()) if block_values.size else np.nan,
            "max_abs": float(block_values.max()) if block_values.size else np.nan,
            "mean_abs": float(block_values.mean()) if block_values.size else np.nan,
            "share_exact_zero": float((block_values == 0).mean()) if block_values.size else np.nan,
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
    abs_elasticities = np.abs(np.asarray(jacobian_matrix(data, kind), float))
    readable = _readable(data, kind, noise_max)
    m_names, p_names = data["moment_block_names"], data["param_block_names"]
    blocks, blocks_ok = _blocks_2d(data, abs_elasticities), _blocks_2d(data, readable)
    n_moment_blocks, n_param_blocks = len(m_names), len(p_names)

    def _grid(fn):
        return np.array([[fn(blocks[(i, j)], blocks_ok[(i, j)])
                          if blocks[(i, j)].size else np.nan
                          for j in range(n_param_blocks)] for i in range(n_moment_blocks)])

    live = _grid(lambda s, o: ((s >= threshold) & o).mean())
    max_abs = _grid(lambda s, o: s.max())
    readable_share = _grid(lambda s, o: o.mean())

    fig, ax = plt.subplots(figsize=figsize)
    image = ax.matshow(live, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(n_moment_blocks):
        for j in range(n_param_blocks):
            if not np.isfinite(live[i, j]):
                continue
            annotation = f"{100 * live[i, j]:.0f}{pct()}\nmax {max_abs[i, j]:.3g}"
            if readable_share[i, j] < 0.999:
                annotation += f"\n({100 * readable_share[i, j]:.0f}{pct()} readable)"
            ax.text(j, i, annotation, ha="center", va="center", fontsize=fs(8),
                    color="white" if live[i, j] > 0.55 else "black")
    ax.set_xticks(range(n_param_blocks))
    ax.set_xticklabels(p_names, rotation=30, ha="left", fontsize=fs(10))
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(range(n_moment_blocks))
    ax.set_yticklabels(m_names, fontsize=fs(10))
    ax.set_xlabel("Parameter block")
    ax.set_ylabel("Moment block")
    ax.set_title(f"Identification map — {data['industry']}, "
                 rf"$\hat{{\mu}}_{data['mu']}$"
                 f"\nshare of $|\\varepsilon| \\geq {threshold:g}$ AND "
                 f"$\\sigma/|\\varepsilon| < {noise_max:g}$", pad=52)
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
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

    elasticities = np.asarray(jacobian_matrix(data, kind), float)
    abs_elasticities = np.abs(elasticities)
    readable = _readable(data, kind, noise_max)
    vmax = abs_elasticities[readable].max() if readable.any() and abs_elasticities[readable].max() > 0 else 1.0

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#f2f2f2")                     # below threshold
    above_threshold = np.ma.masked_where(abs_elasticities < threshold, elasticities)
    shown_matrix = np.ma.masked_where(~readable, above_threshold)          # unreadable painted white on top
    m_edges, p_edges = _edges(data["moment_block_sizes"]), _edges(data["param_block_sizes"])

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("white")
    image = ax.matshow(shown_matrix, cmap=cmap, aspect="auto",
                    norm=SymLogNorm(linthresh=threshold, vmin=-vmax, vmax=vmax, base=10))
    # the noise-rejected cells, drawn white over the grey "below threshold" background
    y_unreadable, x_unreadable = np.where(~readable)
    ax.scatter(x_unreadable, y_unreadable, s=2.0, marker="s", color="white", linewidths=0)
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

    kept = float(((abs_elasticities >= threshold) & readable).mean())
    ax.set_title(f"Elasticity Jacobian, $|\\varepsilon| \\geq {threshold:g}$ and "
                 f"$\\sigma/|\\varepsilon| < {noise_max:g}$ — "
                 f"{data['industry']}, " rf"$\hat{{\mu}}_{data['mu']}$"
                 f"   ({100 * kept:.1f}{pct()} of {abs_elasticities.size} entries shown)", pad=28)
    cbar = fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.04, pad=0.22)
    cbar.set_label(r"$\varepsilon = \partial \log m / \partial \log \theta$"
                   "   (symmetric log; grey = below threshold, white = too noisy)")
    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        fig.savefig(save_to, bbox_inches="tight")
    return ax


CHANNEL_BLOCKS = ("Trade cost", "Density", "Comparative advantage", "Variety count")


def plot_channel_elasticities(data, param_blocks=None,
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
    abs_elasticities = np.abs(np.asarray(jacobian_matrix(data, kind), float))
    readable = _readable(data, kind, noise_max)
    m_names, p_names = data["moment_block_names"], data["param_block_names"]
    if param_blocks is None:              # kappa's block is the only optional channel
        param_blocks = tuple(param_block for param_block in CHANNEL_BLOCKS if param_block != "Density" or param_block in p_names)
    blocks, blocks_ok = _blocks_2d(data, abs_elasticities), _blocks_2d(data, readable)
    reduce = {"max": np.max, "mean": np.mean}[stat]
    stat_tex = {"max": r"\max", "mean": r"\mathrm{mean}"}[stat]

    wanted = [param_block for param_block in param_blocks if param_block in p_names]
    missing = [param_block for param_block in param_blocks if param_block not in p_names]
    if missing:
        raise ValueError(f"parameter block(s) {missing} absent — the Jacobian has "
                         f"{p_names}. The sensitivity Jacobian must carry these columns "
                         "(sensitivity.jl sens_columns).")

    values_by_param, unmeasured = {}, {}
    for param_block in wanted:
        param_index = p_names.index(param_block)
        channel_values, unmeasured_flags = [], []
        for i in range(len(m_names)):
            block_values, block_readable = blocks[(i, param_index)], blocks_ok[(i, param_index)]
            if block_values.size == 0 or not block_readable.any():
                channel_values.append(0.0)
                unmeasured_flags.append(block_values.size > 0)
            else:
                channel_values.append(float(reduce(block_values[block_readable])))
                unmeasured_flags.append(False)
        values_by_param[param_block], unmeasured[param_block] = np.array(channel_values), np.array(unmeasured_flags)

    positive_values = np.concatenate([channel_values[channel_values > 0] for channel_values in values_by_param.values()] or [np.array([threshold])])
    bottom = (positive_values.min() / 4) if positive_values.size else threshold / 10

    fig, axes = plt.subplots(1, len(wanted), figsize=figsize, sharey=True)
    axes = np.atleast_1d(axes)
    x_positions = np.arange(len(m_names))
    for ax, pname in zip(axes, wanted):
        param_values, param_unmeasured = values_by_param[pname], unmeasured[pname]
        ax.bar(x_positions, param_values, color=[sim_color if channel_values >= threshold else "0.75" for channel_values in param_values])
        ax.set_yscale("log")
        ax.set_ylim(bottom=bottom)
        ax.axhline(threshold, color=reference_color, linestyle="--", linewidth=1.2)
        for i, (channel_values, unmeasured_flags) in enumerate(zip(param_values, param_unmeasured)):
            if channel_values == 0:
                ax.text(i, bottom * 1.35, "n/m" if unmeasured_flags else "0", ha="center", va="bottom",
                        fontsize=fs(9), color=reference_color, fontweight="bold")
        ax.set_xticks(x_positions)
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
    suppliers = data.get("suppliers")
    if suppliers is None:
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
    has_rep = "replication" in suppliers.columns
    if replication is not None:
        if not has_rep:
            raise KeyError("`replication` was asked for but suppliers.parquet has no "
                           "`replication` column — this tree predates the finite-variety "
                           "post-hoc economy.")
        suppliers = suppliers.loc[suppliers["replication"] == replication]
        if suppliers.empty:
            raise ValueError(f"no rows for replication {replication}.")
    n_rep_pooled = int(suppliers["replication"].nunique()) if has_rep else 1

    dist_path = data["input_folder"] / "distances.npy"
    if not dist_path.exists():
        raise FileNotFoundError(f"{dist_path} not found — needed for log Dist_{{r'r}}.")
    distances = np.load(dist_path)
    R = data["R"]
    if distances.shape[0] < R or distances.shape[1] < R:
        raise ValueError(f"distances.npy is {distances.shape}, too small for R = {R} regions.")
    distances = distances[:R, :R]

    keep = ["SIREN", "A129", "ze2010", "ze2010_downstream",
            "share", "downstream_purchase", "productivity", "sample_weight"]
    df = suppliers[keep + (["replication"] if has_rep else [])].copy()

    # The supplier identifier. Under the default it IS the parquet's SIREN; under an
    # aggregating `firm_key` the column is rebuilt from the key, and the euro flows of the
    # pooled varieties are added before the share is formed — which is what makes the
    # aggregated firm a genuine multi-variety supplier rather than an average of shares.
    if tuple(firm_key) != ("SIREN",):
        pooling_key = list(firm_key)
        if has_rep and "replication" not in pooling_key:
            pooling_key = pooling_key + ["replication"]        # never pool two economies into one firm
        # one string per pooled firm, built column-wise rather than row by row
        as_text = [df[column].astype(str) for column in pooling_key]
        df["SIREN"] = as_text[0].str.cat(as_text[1:], sep="-")

    # X_{i->r} = share x (mu Y_r); a_ir = X_{i->r} / sum_r X_{i->r}
    df["X_ir"] = df["share"] * df["downstream_purchase"]
    # how many real firms one simulated supplier stands for: the MEAN of its varieties'
    # draw weights, so that pooling varieties does not multiply the weight (under the
    # flat draw designs every variety carries 1/N_rho and a pooled firm still does)
    w_firm = df.groupby("SIREN", sort=True)["sample_weight"].mean()
    df = (df.groupby(["SIREN", "ze2010_downstream"], as_index=False)
            .agg(A129=("A129", "first"), ze2010=("ze2010", "first"),
                 X_ir=("X_ir", "sum"), productivity=("productivity", "mean")))
    firm_total = df.groupby("SIREN", sort=False)["X_ir"].transform("sum")
    df["a_ir"] = np.where(firm_total > 0, df["X_ir"] / firm_total, 0.0)

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
    distance = distances[panel["ze2010"].to_numpy() - 1, panel["ze2010_downstream"].to_numpy() - 1]
    panel["distance"] = distance
    panel["log_distance"] = np.log(np.maximum(distance, min_distance_km))
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



def fepois_fit(panel, y, regressors=("log_distance",), fixed_effects="fe_group", cluster="cluster",
               weights="sample_weight"):
    """
    One `pyfixest.fepois` call: E[y | .] = exp(fixed effect + regressors'beta).

    `fixed_effects` is passed to pyfixest as written, so "fe_group" absorbs the sector x
    downstream-region effect, "SIREN" absorbs a SUPPLIER effect, and "fe_group + SIREN"
    absorbs both.

    A three-line wrapper around the library, not an estimator: it writes the formula,
    passes the Monte-Carlo draw weights as analytic weights, clusters on the supplier's
    own region and lets pyfixest drop the fixed-effect groups that are separated (a
    group with no positive outcome contributes nothing to the Poisson score, exactly as
    in `ppmlhdfe`). Returns the fitted object, so `.coef()`, `.se()`, `.tidy()` and
    `.summary()` are all available at the call site.
    """
    formula = f"{y} ~ " + " + ".join(regressors) + (f" | {fixed_effects}" if fixed_effects else "")
    return pf.fepois(fml=formula, data=panel, weights=weights, weights_type="aweights",
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
    mean_log_distance = EMPIRICAL_MEAN_LOG_D if mean_log_d is None else mean_log_d
    return float(eta) / (1.0 + abs(float(eta)) * mean_log_distance)


def eta_from_delta_over_gamma(ratio, mean_log_d=None):
    """The inverse map: the constant elasticity a published `delta/gamma` implies."""
    mean_log_distance = EMPIRICAL_MEAN_LOG_D if mean_log_d is None else mean_log_d
    ratio_value = float(ratio)
    denom = 1.0 - abs(ratio_value) * mean_log_distance
    return ratio_value / denom if denom > 0 else -np.inf


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
    nu_s_value = _read_named_value(data["coefs"], "nu_s") if data.get("coefs") is not None else None
    if nu_s_value is None or not np.isfinite(nu_s_value):
        return float(NU_S_DEFAULT)
    nu_s_value = np.asarray(nu_s_value, dtype=float).ravel()
    if nu_s_value.size != 1:
        raise ValueError(f"stats.csv carries {nu_s_value.size} values for nu_s; the granular "
                         "decomposition is written for a scalar elasticity.")
    return float(nu_s_value[0])


def theta_alpha(data):
    """
    `theta * alpha_hat` for this run, read off the LOADED estimate — the scale every term
    of the moment is measured in (Appendix B.5: both margins are theta*alpha times a
    dimensionless number, so |eta_int| <= theta*alpha and |eta_ext| = theta*alpha E(Lambda)).

    `best_params` is the raw theta vector, layout
        [chi_L(1) | chi_s(S) | A(R_d) | alpha(N_TAU) | T(active (s,AA), s-major)],
    so alpha is the N_TAU-long slice after the A block; the length is asserted against the
    rebuilt layout, exactly as `unpack_estimated_T` does in the comparative-advantage
    section. theta follows the same rule as `model_theta` there: stats.csv when it carries
    one, else THETA_DEFAULT. Defined here so the section runs on its own.
    """
    best_params = data.get("best_params")
    if best_params is None:
        raise FileNotFoundError(f"no best_params in {data['folder']}/{data['step_dir']}/.")
    best_params = np.asarray(best_params, dtype=float).ravel()
    S, n_tau = data["S"], data["n_tau"]
    R_d = len(data["aa_names"])
    expected = 1 + S + R_d + n_tau + 1 + int(data["T_ACTIVE"].sum())   # + kappa; T incl. Phi^F
    if best_params.size != expected:
        raise ValueError(f"best_params has {best_params.size} entries, layout rebuilt here is "
                         f"{expected} — the run's flags and the ones passed here disagree.")
    theta_stats = _read_named_value(data["coefs"], "theta") if data.get("coefs") is not None else None
    theta = float(theta_stats) if theta_stats is not None and np.isfinite(theta_stats) and theta_stats > 0 else THETA_DEFAULT
    if n_tau == 1:
        return float(theta) * float(best_params[1 + S + R_d])
    # binned tau: no single elasticity -- the rho- and purchase-weighted mean of the local
    # one, d ln tau / d ln d, over the competing cells (`sourcing_geometry`)
    geometry = sourcing_geometry(data)
    return float(theta) * geometry["theta_alpha"] / geometry["theta"]


def estimate_untargeted_moment(data, controls=(), cluster="cluster", panel=None,
                               fixed_effects=BASELINE_FE, verbose=True):
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
    regressors = ["log_distance"] + list(controls)
    ppml_fit = fepois_fit(panel, "a_ir", regressors=regressors, fixed_effects=fixed_effects, cluster=cluster)
    eta = float(ppml_fit.coef()["log_distance"])
    se = float(ppml_fit.se()["log_distance"])
    empirical = EMPIRICAL_DELTA_OVER_GAMMA.get(data["industry"], {})
    try:
        theta_alpha_value = theta_alpha(data)
    except Exception:
        theta_alpha_value = np.nan

    result = {"industry": data["industry"], "mu": data["mu"], "spec": "+".join(regressors),
           "fe": fixed_effects, "fe_label": BASELINE_FE_LABEL, "theta_alpha": theta_alpha_value,
           "eta": eta, "se": se, "t": eta / se if se > 0 else np.nan,
           "ci_lo": eta - 1.96 * se, "ci_hi": eta + 1.96 * se,
           # the same estimate on the reduced form's own scale — the one that is reported
           "dg": delta_over_gamma_from_eta(eta),
           "dg_lo": delta_over_gamma_from_eta(eta - 1.96 * se),
           "dg_hi": delta_over_gamma_from_eta(eta + 1.96 * se),
           "n_obs": int(getattr(ppml_fit, "_N", len(panel))),
           "n_firms": int(panel["SIREN"].nunique()),
           "share_zero": float((panel["a_ir"].to_numpy() == 0).mean()),
           "emp_estimate": empirical.get("estimate"), "emp_ci_lo": empirical.get("ci_lo"),
           "emp_ci_hi": empirical.get("ci_hi"), "fit": ppml_fit, "panel": panel}

    if verbose:
        print(f"[{data['industry']}]  {len(panel):,} supplier x downstream-region cells, "
              f"{result['n_firms']:,} suppliers, {100 * result['share_zero']:.1f}% zeros, "
              f"from {data['suppliers_path']}")
        print(f"  measured eta = {eta:+.4f} ({se:.4f})   ->  delta/gamma = {result['dg']:+.4f} "
              f"[{result['dg_lo']:+.4f}, {result['dg_hi']:+.4f}]   "
              f"(mean log d = {EMPIRICAL_MEAN_LOG_D}, theta*alpha = {theta_alpha_value:.4f})")
        if result["emp_estimate"] is not None:
            print(f"  DATA delta/gamma = {result['emp_estimate']:+.4f}  "
                  f"[{result['emp_ci_lo']:+.4f}, {result['emp_ci_hi']:+.4f}]")
    return result


def untargeted_across_replications(data, fixed_effects=BASELINE_FE, controls=(), cluster="cluster",
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
    suppliers = data.get("suppliers")
    if suppliers is None:
        raise FileNotFoundError("no suppliers.parquet — see build_a_ir_panel.")
    replications = (sorted(suppliers["replication"].unique()) if "replication" in suppliers.columns else [None])
    if max_replications is not None:
        replications = replications[:max_replications]

    rows = []
    for replication in replications:
        panel = build_a_ir_panel(data, replication=replication)
        replication_result = estimate_untargeted_moment(data, controls=controls, cluster=cluster,
                                       panel=panel, fixed_effects=fixed_effects, verbose=False)
        rows.append({"replication": replication, "eta": replication_result["eta"], "se_within": replication_result["se"],
                     "dg": replication_result["dg"], "n_firms": replication_result["n_firms"],
                     "share_zero": replication_result["share_zero"]})
    per_replication = pd.DataFrame(rows).set_index("replication")

    pooled = estimate_untargeted_moment(data, controls=controls, cluster=cluster,
                                        fixed_effects=fixed_effects, verbose=False)
    per_replication.attrs["pooled_eta"] = pooled["eta"]
    per_replication.attrs["pooled_se"] = pooled["se"]
    per_replication.attrs["sd_across"] = float(per_replication["eta"].std(ddof=1)) if len(per_replication) > 1 else np.nan
    per_replication.attrs["mean_eta"] = float(per_replication["eta"].mean())

    if verbose:
        sd = per_replication.attrs["sd_across"]
        head = f"[{data['industry']}]  eta_hat over {len(per_replication)} realisation(s): "
        if len(per_replication) > 1:
            print(head + f"mean {per_replication.attrs['mean_eta']:.4f}, "
                         f"sd across draws {sd:.4f}")
            print(f"              pooled fit {pooled['eta']:.4f} (se {pooled['se']:.4f})"
                  f" — that se reads {len(per_replication)} economies as information about one, so "
                  f"the sd above is the interval to report")
        else:
            print(head + f"{per_replication.attrs['mean_eta']:.4f} (se {pooled['se']:.4f})")
    return per_replication


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
    a_ir = panel["a_ir"].to_numpy(float)
    weights = panel["sample_weight"].to_numpy(float)
    served = a_ir > 0
    denom = float((weights * a_ir).sum())
    n_suppliers = int(panel["SIREN"].nunique())
    return {"serving_rate": float(np.average(served.astype(float), weights=weights)),
            "openness_weighted": float(1.0 - (weights * a_ir * a_ir).sum() / denom) if denom > 0 else np.nan,
            "buyers_per_supplier": float(served.sum() / n_suppliers) if n_suppliers else np.nan,
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
    for result in results:
        empirical = EMPIRICAL_DELTA_OVER_GAMMA.get(result["industry"], {})
        rows.append({
            "industry": empirical.get("display_name", result["industry"]),
            "fixed effect": result["fe_label"],
            "delta/gamma (model)": result["dg"],
            "95% CI (model, delta/gamma)": f"[{result['dg_lo']:+.3f}, {result['dg_hi']:+.3f}]",
            "delta/gamma (data)": result["emp_estimate"],
            "95% CI (data)": (f"[{result['emp_ci_lo']:+.3f}, {result['emp_ci_hi']:+.3f}]"
                              if result["emp_ci_lo"] is not None else "---"),
            "inside data CI": (None if result["emp_ci_lo"] is None
                               else bool(result["emp_ci_lo"] <= result["dg"] <= result["emp_ci_hi"])),
            "theta*alpha": result.get("theta_alpha", np.nan),
            "suppliers": result["n_firms"], "cells": result["n_obs"],
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
    fig, ax = plt.subplots(figsize=figsize or get_figsize(width_fraction=0.95, height_fraction=0.65))
    x_positions = np.arange(len(results))

    for i, result in enumerate(results):
        model_value, lo, hi = result["dg"], result["dg_lo"], result["dg_hi"]
        ax.errorbar(x_positions[i] - 0.12, model_value, yerr=[[model_value - lo], [hi - model_value]], fmt="o", color=sim_color,
                    capsize=4, markersize=7,
                    label="Model (baseline)" if i == 0 else None)
        if result["emp_estimate"] is not None:
            ax.errorbar(x_positions[i] + 0.22, result["emp_estimate"],
                        yerr=[[result["emp_estimate"] - result["emp_ci_lo"]],
                              [result["emp_ci_hi"] - result["emp_estimate"]]],
                        fmt="s", color=reference_color, capsize=4, markersize=8,
                        label="Data $\\delta/\\gamma$ (Table 3)" if i == 0 else None)

    ceiling = -1.0 / EMPIRICAL_MEAN_LOG_D
    ax.axhline(ceiling, color="0.4", linestyle=":", linewidth=1.1)
    ax.annotate(rf"ceiling $-1/\overline{{\log d}}$ = {ceiling:.3f}", (0.01, ceiling),
                xycoords=("axes fraction", "data"), fontsize=fs(8), color="0.35", va="bottom")
    ax.axhline(0.0, color="0.6", linewidth=0.8)
    ax.set_xlim(-0.5, len(results) - 0.5)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([EMPIRICAL_DELTA_OVER_GAMMA.get(result["industry"], {})
                        .get("display_name", result["industry"]) for result in results])
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


def _pair_tau_elasticity(data, panel, estimate=None):
    """d ln tau / d ln d at each panel pair's distance: alpha under the power law, the
    segment elasticity alpha_k(d_ir) under binned tau (`utils.tau_elasticity`)."""
    estimate = _estimated_T(data) if estimate is None else estimate
    return tau_elasticity(estimate["alpha"], panel["log_distance"].to_numpy(float),
                          tau_log_knots(data))


def attach_lambda(data, panel, min_distance_km=1.0):
    """
    The pair-level surprisal of winning,

        Lambda_ir = Phi_{-r', r s} (w_{r's} tau_{r'r})^theta z_i^{-theta},
        tau_{r'r} = d_{r'r}^alpha  (or the binned tau of `utils.log_tau`),
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
    firm_key = tuple(panel.attrs.get("firm_key", ("SIREN",)))
    if firm_key != ("SIREN",):
        raise ValueError(
            f"the panel was built with firm_key={firm_key}: `productivity` is then a mean over "
            "the pooled varieties and Lambda is not defined pair by pair. Build the panel "
            "with the default key for the structural moment.")

    estimate = _estimated_T(data)                       # T (S, n_AA), alpha (N_TAU), theta
    alpha, theta = estimate["alpha"], float(estimate["theta"])
    S, R = data["S"], data["R"]
    aa_of_ze = np.asarray(data["aa_of_ze"]).ravel()          # region -> area, 0-based
    CELL_MASK = np.asarray(data["CELL_MASK"], dtype=bool)    # (S, R) simulated cells

    distances = np.load(data["input_folder"] / "distances.npy")[:R, :R]
    distance = np.maximum(distances, min_distance_km)
    tau = np.exp(log_tau(alpha, np.log(distance), tau_log_knots(data)))   # (R, R) origin x buyer

    w_path = data["input_folder"] / "w_rs.npy"
    if w_path.exists():
        w_rs = np.asarray(np.load(w_path), dtype=float).ravel()[:R]
        w_rs = np.where(w_rs > 0, 1.0, 1.0)                   # the model's own normalisation
    else:
        w_rs = np.ones(R)
        print("  note: w_rs.npy not found — the upstream cost shifter is taken as one, "
              "which is what load_parameters.jl normalises it to anyway.")

    w_tau = w_rs[:, None] * tau                                # (R, R): (w_{r'} tau_{r'r})
    # Phi_{rs} = sum over the sector's simulated cells, at the CELL-level T_a n_ls^kappa
    T_ls = estimate["T_cell"]
    Phi_contribution = np.where(CELL_MASK[:, :, None], (T_ls[:, :, None]
                                               * w_tau[None, :, :] ** (-theta)), 0.0)
    Phi = Phi_contribution.sum(axis=1)                                 # (S, R) over origins
    Phi_minus = Phi[:, None, :] - Phi_contribution                     # (S, R_origin, R_buyer)

    sector_index = panel["A129"].to_numpy() - 1
    origin_index = panel["ze2010"].to_numpy() - 1
    buyer_index = panel["ze2010_downstream"].to_numpy() - 1
    productivity = np.maximum(panel["productivity"].to_numpy(float), 1e-300)
    panel_with_lambda = panel.copy()
    panel_with_lambda["lambda_ir"] = (np.maximum(Phi_minus[sector_index, origin_index, buyer_index], 0.0)
                        * w_tau[origin_index, buyer_index] ** theta * productivity ** (-theta))
    panel_with_lambda.attrs.update(panel.attrs)
    panel_with_lambda.attrs.update({"alpha": np.asarray(alpha, dtype=float), "theta": theta})
    return panel_with_lambda


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
    a_ir = panel["a_ir"].to_numpy(float)
    weights = a_ir * panel["sample_weight"].to_numpy(float)
    assert np.all(weights[a_ir <= 0] == 0.0), "unserved cells must carry zero weight"
    return weights


def structural_eta(data, panel, verbose=True):
    """
    The theoretical moment, from the granular derivation, with no regression.

        dlog a_ir / dlog d = -alpha_ir [ theta * Lambda_ir + (nu_s - 1)(1 - a_ir) ]

    with alpha_ir = d ln tau/d ln d at the pair's distance (alpha under the power law),
    averaged with w = a * rho/rho_tilde over the SERVED cells (the derivative is conditional
    on winning; unserved cells carry zero weight, which is asserted rather than assumed).

    The two shares sum to one by construction — that is the point of computing rather than
    estimating them: there is no identity left to fail.
    """
    if "lambda_ir" not in panel.columns:
        raise KeyError("run `attach_lambda(data, panel)` first — the structural moment "
                       "needs the pair-level surprisal.")
    estimate = _estimated_T(data)
    theta = float(estimate["theta"])
    nu_s = model_nu_s(data)

    weights = _structural_weights(panel)
    lambda_ir = panel["lambda_ir"].to_numpy(float)
    a_ir = panel["a_ir"].to_numpy(float)
    alpha_pair = _pair_tau_elasticity(data, panel, estimate)
    weight_total = weights.sum()
    # alpha is the w-weighted local elasticity; E_lambda and E_open are weighted by it too,
    # so eta_ext = -theta*alpha*E_lambda holds exactly (and reduces to the plain w-means
    # under the power law, where alpha_ir is constant)
    alpha = float((weights * alpha_pair).sum() / weight_total)
    E_lambda = float((weights * alpha_pair * lambda_ir).sum() / weight_total) / alpha
    E_open = float((weights * alpha_pair * (1.0 - a_ir)).sum() / weight_total) / alpha

    eta_ext = -theta * alpha * E_lambda
    eta_int = -(nu_s - 1.0) * alpha * E_open
    eta = eta_ext + eta_int
    result = {"industry": data["industry"], "theta": theta, "alpha": alpha, "nu_s": nu_s,
           "theta_alpha": theta * alpha, "eta_struct": eta,
           "eta_struct_ext": eta_ext, "eta_struct_int": eta_int,
           "E_lambda": E_lambda, "E_open": E_open,
           "share_ext": eta_ext / eta if eta != 0 else np.nan,
           "share_int": eta_int / eta if eta != 0 else np.nan,
           "dg_struct": delta_over_gamma_from_eta(eta)}
    result.update({f"panel_{k}": v for k, v in panel_facts(panel).items()})
    if verbose:
        print(f"  structural eta = {eta:+.4f}  =  extensive {eta_ext:+.4f} "
              f"({100 * result['share_ext']:.0f}%) + substitution {eta_int:+.4f} "
              f"({100 * result['share_int']:.0f}%)   ->  delta/gamma = {result['dg_struct']:+.4f}")
        print(f"    E_w(Lambda) = {E_lambda:.3f},  E_w(1-a) = {E_open:.3f},  "
              f"theta*alpha = {theta * alpha:.4f},  (nu_s-1)/theta = "
              f"{(nu_s - 1.0) / theta:.3f}")
    return result


def validate_structural(data, panel, structural, verbose=True):
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
    lambda_ir = panel["lambda_ir"].to_numpy(float)
    served = panel["a_ir"].to_numpy(float) > 0
    weights = _structural_weights(panel)

    # 1. gamma measured cell by cell: the share of (cell, sector) varieties winning in r
    win_share_by_cell = (panel.assign(served=served.astype(float))
              .groupby(["ze2010", "A129", "ze2010_downstream"])["served"].mean())
    gamma_pair = panel.set_index(["ze2010", "A129", "ze2010_downstream"]).index.map(win_share_by_cell)
    gamma_w = float((weights * np.asarray(gamma_pair, dtype=float)).sum() / weights.sum())
    e_lambda_win = float((weights * lambda_ir).sum() / weights.sum())

    # 2. exponentiality
    p_hat = float(np.mean(np.exp(-lambda_ir)))
    serving = float(served.mean())

    # 3. the substitution scale
    scale = (structural["nu_s"] - 1.0) / structural["theta"]
    implied = abs(structural["eta_struct_int"]) / structural["theta_alpha"]

    df = pd.DataFrame([
        {"check": "E(Lambda | win) = 1 - gamma", "closed form": e_lambda_win,
         "measured": 1.0 - gamma_w, "ratio": e_lambda_win / max(1.0 - gamma_w, 1e-12)},
        {"check": "E(exp(-Lambda)) vs serving rate (ratio < 1: see docstring)",
         "closed form": p_hat, "measured": serving,
         "ratio": p_hat / max(serving, 1e-12)},
        {"check": "eta_int/(theta*alpha) = (nu_s-1)/theta * E(1-a)", "closed form": implied,
         "measured": scale * structural["E_open"],
         "ratio": implied / max(scale * structural["E_open"], 1e-12)},
    ]).set_index("check")
    if verbose:
        print(df.round(4).to_string())
        print("    a ratio far from one is a wiring error in attach_lambda "
              "(T, w, Phi or the distance floor), not a finding about the economy")
    return df


def decompose_gap(data, panel, measured, structural, verbose=True):
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
    estimate = _estimated_T(data)
    theta = float(estimate["theta"])
    nu_s = model_nu_s(data)
    lambda_ir = panel["lambda_ir"].to_numpy(float)
    a_ir = panel["a_ir"].to_numpy(float)
    deriv = -_pair_tau_elasticity(data, panel, estimate) * (theta * lambda_ir + (nu_s - 1.0) * (1.0 - a_ir))

    weights = _structural_weights(panel)
    under_w = float((weights * deriv).sum() / weights.sum())

    # (a) the same derivative under PPML's own weights
    try:
        fitted_mean = np.asarray(measured["fit"].predict(), dtype=float).ravel()
        if fitted_mean.size != len(panel):                       # separated groups were dropped
            fitted_mean = None
    except Exception:
        fitted_mean = None
    if fitted_mean is None:
        weighting = np.nan
        under_mu = np.nan
    else:
        fitted_mean = np.maximum(fitted_mean, 0.0)
        under_mu = float((fitted_mean * deriv).sum() / fitted_mean.sum())
        weighting = under_mu - under_w

    # (b) the functional form: the same panel in the reduced form's shape
    linear_fit = linear_delta_over_gamma(panel)
    eta_linear = eta_from_delta_over_gamma(linear_fit["delta_over_gamma"])
    functional = measured["eta"] - eta_linear

    R = measured["eta"] - structural["eta_struct"]
    composition = R - (0.0 if not np.isfinite(weighting) else weighting) - functional
    result = {"industry": data["industry"], "eta": measured["eta"],
           "eta_struct": structural["eta_struct"], "R": R, "R_weighting": weighting,
           "R_functional": functional, "R_composition": composition,
           "deriv_under_w": under_w, "deriv_under_fitted": under_mu,
           "eta_linear_equivalent": eta_linear,
           "dg_linear": linear_fit["delta_over_gamma"], "dg_ppml": measured["dg"]}
    if verbose:
        sign = "measured STEEPER than structural" if R < 0 else "measured FLATTER"
        print(f"  gap R = {R:+.4f}  ({sign})")
        print(f"    weighting  {weighting:+.4f}   (derivative under fitted weights "
              f"{under_mu:+.4f} vs under a*rho/rho~ {under_w:+.4f})")
        print(f"    functional {functional:+.4f}   (level-linear delta/gamma "
              f"{linear_fit['delta_over_gamma']:+.4f} vs PPML's {measured['dg']:+.4f})")
        print(f"    composition {composition:+.4f}   (origin-level primitives covarying "
              "with distance — the term the derivative does not take)")
    return result


def within_area_check(data, panel, gap_result, verbose=True):
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
    # Keep the (sector, area) blocks that hold more than one origin region: only there
    # do several origins share the same area-level comparative advantage.
    origins = pd.DataFrame({"A129": panel["A129"].to_numpy(), "area": area,
                            "origin": panel["ze2010"].to_numpy()})
    n_origins = origins.groupby(["A129", "area"])["origin"].transform("nunique")
    kept = n_origins.to_numpy() > 1
    within_area_panel = panel[kept].copy()
    within_area_panel.attrs.update(panel.attrs)
    n_pairs, n_suppliers = len(within_area_panel), within_area_panel["SIREN"].nunique() if len(within_area_panel) else 0
    if n_pairs < 500 or n_suppliers < 20:
        if verbose:
            print(f"  within-area check: only {n_pairs} pairs / {n_suppliers} suppliers survive "
                  "— too few to quote, the areas are near single-region here")
        return {"industry": data["industry"], "n_pairs": int(n_pairs),
                "n_suppliers": int(n_suppliers), "eta": np.nan, "eta_struct": np.nan,
                "R": np.nan, "quotable": False}
    measured_sub = estimate_untargeted_moment(data, panel=within_area_panel, verbose=False)
    structural_sub = structural_eta(data, within_area_panel, verbose=False)
    result = {"industry": data["industry"], "n_pairs": int(n_pairs),
           "n_suppliers": int(n_suppliers), "eta": measured_sub["eta"],
           "eta_struct": structural_sub["eta_struct"],
           "R": measured_sub["eta"] - structural_sub["eta_struct"], "quotable": True}
    if verbose:
        print(f"  within-area check ({n_pairs:,} pairs, {n_suppliers:,} suppliers): "
              f"eta {result['eta']:+.4f} vs structural {result['eta_struct']:+.4f}, "
              f"R = {result['R']:+.4f}   (against R = {gap_result['R']:+.4f} on the full "
              "sample — a smaller |R| here means comparative advantage covarying with "
              "distance was carrying it)")
    return result


def structural_summary(structural_results):
    """One row per industry: the theoretical moment, the measured one, and their gap."""
    rows = []
    for result in structural_results:
        empirical = EMPIRICAL_DELTA_OVER_GAMMA.get(result["industry"], {})
        rows.append({
            "industry": empirical.get("display_name", result["industry"]),
            "theta*alpha": result["theta_alpha"], "nu_s": result["nu_s"],
            "eta struct": result["eta_struct"], "share ext": result["share_ext"],
            "share int": result["share_int"], "E_w(Lambda)": result["E_lambda"],
            "E_w(1-a)": result["E_open"], "eta measured": result["eta"],
            "R": result["R"], "R weighting": result["R_weighting"],
            "R functional": result["R_functional"], "R composition": result["R_composition"],
            "serving rate": result["panel_serving_rate"],
            "buyers per supplier": result["panel_buyers_per_supplier"],
        })
    return pd.DataFrame(rows).set_index("industry")


def plot_structural_vs_measured(structural_results, figsize=None, save_to=None):
    """
    Per industry, two bars on the delta/gamma scale: the THEORETICAL moment, split into its
    extensive and substitution parts (they sum exactly), and the MEASURED one. The distance
    between the bar tops is R, annotated with its three parts.
    """
    fig, ax = plt.subplots(figsize=figsize or get_figsize(width_fraction=0.95, height_fraction=0.62))
    x_positions = np.arange(len(structural_results))
    bar_width = 0.34
    for i, result in enumerate(structural_results):
        dg_extensive = delta_over_gamma_from_eta(result["eta_struct_ext"])
        dg_total = delta_over_gamma_from_eta(result["eta_struct"])
        ax.bar(x_positions[i] - bar_width / 2, dg_extensive, bar_width, color=sim_color, alpha=0.9,
               label="structural: extensive" if i == 0 else None)
        ax.bar(x_positions[i] - bar_width / 2, dg_total - dg_extensive, bar_width, bottom=dg_extensive, color=(0.30, 0.55, 0.40),
               alpha=0.85, label="structural: substitution" if i == 0 else None)
        ax.bar(x_positions[i] + bar_width / 2, delta_over_gamma_from_eta(result["eta"]), bar_width,
               color=reference_color, alpha=0.8,
               label="measured (PPML)" if i == 0 else None)
        ax.annotate(rf"$\theta\alpha$ = {result['theta_alpha']:.3f}", (x_positions[i], 0.004),
                    ha="center", fontsize=fs(8), color="0.3")
        ax.annotate(f"R: weight {result['R_weighting']:+.2f}, form {result['R_functional']:+.2f}, "
                    f"comp {result['R_composition']:+.2f}",
                    (x_positions[i], delta_over_gamma_from_eta(result["eta"])), xytext=(0, -12),
                    textcoords="offset points", ha="center", fontsize=fs(7), color="0.35")
    ax.axhline(0.0, color="0.6", linewidth=0.8)
    ceiling = -1.0 / EMPIRICAL_MEAN_LOG_D
    ax.axhline(ceiling, color="0.4", linestyle=":", linewidth=1.1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([EMPIRICAL_DELTA_OVER_GAMMA.get(result["industry"], {})
                        .get("display_name", result["industry"])
                        for result in structural_results])
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
    # Weighted least squares of the portfolio share on log distance, by pyfixest. The
    # sample has no missing value by construction (the panel is zero-filled), and the
    # check below makes sure no observation was dropped from it.
    sample = panel[["a_ir", "log_distance"]].astype(float)
    sample["weight"] = 1.0 if weight is None else panel[weight].to_numpy(float)
    level_fit = pf.feols("a_ir ~ log_distance", data=sample, weights="weight", vcov="iid")
    assert int(level_fit._N) == len(sample), "pyfixest dropped observations from the panel"
    gamma = float(level_fit.coef()["Intercept"])
    delta = float(level_fit.coef()["log_distance"])
    mean_log_distance = float(np.average(sample["log_distance"], weights=sample["weight"]))
    return {"gamma": gamma, "delta": delta,
            "delta_over_gamma": float(delta / gamma) if gamma != 0 else np.nan,
            "elasticity_at_mean_d": float(delta / (gamma + delta * mean_log_distance))
            if (gamma + delta * mean_log_distance) != 0 else np.nan,
            "mean_log_distance": mean_log_distance}


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
    linear_fit = linear_delta_over_gamma(panel)
    rows = []

    def _row(label, dg, note, lo=np.nan, hi=np.nan, n=np.nan):
        rows.append({"specification": label, "delta/gamma": dg, "ci_lo": lo, "ci_hi": hi,
                     "n_obs": n, "isolates": note})

    def _ppml(df, label, note):
        ppml_fit = fepois_fit(df, "a_ir", fixed_effects=BASELINE_FE, cluster="cluster")
        eta_hat = float(ppml_fit.coef()["log_distance"])
        se = float(ppml_fit.se()["log_distance"])
        _row(label, delta_over_gamma_from_eta(eta_hat), note,
             lo=delta_over_gamma_from_eta(eta_hat - 1.96 * se),
             hi=delta_over_gamma_from_eta(eta_hat + 1.96 * se),
             n=int(getattr(ppml_fit, "_N", len(df))))

    _ppml(panel, "Baseline eta (PPML)",
          "the measured moment: within supplier, within buyer cell")
    _row("Level-linear refit", linear_fit["delta_over_gamma"],
         f"the reduced form's own shape, at the simulated mean log d = "
         f"{linear_fit['mean_log_distance']:.2f}", n=len(panel))
    _ppml(build_a_ir_panel(data, firm_key=("ze2010", "A129")), "Multi-variety firms",
          "the unit: varieties of a cell pooled into one firm")
    if structural is not None:
        _row("Structural eta", structural["dg_struct"],
             "the granular derivative, computed — no interval, it is not an estimate")

    empirical = EMPIRICAL_DELTA_OVER_GAMMA.get(data["industry"], {})
    if empirical.get("estimate") is not None:
        _row("Empirical delta/gamma (Table 3)", empirical["estimate"],
             "the published reduced form", lo=empirical["ci_lo"], hi=empirical["ci_hi"])

    df = pd.DataFrame(rows).set_index("specification")
    df.attrs.update(linear_fit)
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
    empirical = EMPIRICAL_DELTA_OVER_GAMMA.get(industry, {})
    df = ladder[::-1]
    y_positions = np.arange(len(df))
    is_data = df.index.str.startswith("Empirical")
    is_struct = df.index.str.startswith("Structural")
    ratios = df["delta/gamma"].to_numpy(float)
    lo, hi = df["ci_lo"].to_numpy(float), df["ci_hi"].to_numpy(float)

    fig, ax = plt.subplots(figsize=figsize or get_figsize(width_fraction=1.0, height_fraction=0.6))
    if empirical.get("ci_lo") is not None:
        ax.axvspan(empirical["ci_lo"], empirical["ci_hi"], color=reference_color, alpha=0.15,
                   label=f"data 95{pct()} CI")
    ax.barh(y_positions, ratios, color=[reference_color if is_data_row else
                            (0.30, 0.55, 0.40) if is_struct_row else sim_color
                            for is_data_row, is_struct_row in zip(is_data, is_struct)], alpha=0.85)
    for i, (ci_lo_row, ci_hi_row) in enumerate(zip(lo, hi)):
        if np.isfinite(ci_lo_row) and np.isfinite(ci_hi_row):
            ax.plot([ci_lo_row, ci_hi_row], [i, i], color="black", linewidth=1)
    ceiling = -1.0 / EMPIRICAL_MEAN_LOG_D
    ax.axvline(ceiling, color="0.4", linestyle=":", linewidth=1.1)
    ax.annotate(rf"$-1/\overline{{\log d}}$", (ceiling, len(df) - 0.4), fontsize=fs(8),
                color="0.35", ha="right")
    ax.axvline(0.0, color="0.6", linewidth=0.8)
    ax.set_yticks(y_positions)
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
