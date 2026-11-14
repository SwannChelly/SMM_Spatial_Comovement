"""
Comparative advantage, concentration and granularity -- part of the library behind
`tests_counterfactuals.ipynb`.

Comparative advantage against distance within a sector (tests 1-6), the alignment
covariance in kilometres, the buyer's own portfolio of suppliers and its commonality
decomposition, the supplier's portfolio of customers, and the local share read as a level
against a granular dispersion.
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
import pyfixest as pf
from scipy.stats import gaussian_kde

from utils import (
    AMPLIFICATION_RADII, CF_COLORS, CF_REGIMES, INFINITE_REGIME, NU_S_DEFAULT,
    PORTFOLIO_TAUS, UNIFORM_REGIME,
    _buyer_weights, _by_sector_code, _despine, _downstream_ze_index,
    _n_hat_from_diagnostics, _parquet_sector_index, _region_labels,
    _sectors_in_code_order, aa_display_names, fs, get_figsize, model_theta,
    name_axis_fontsize, pct, reference_color, set_name_axis_fontsize, sim_color,
    simulate_economy,
    sourcing_geometry, toulouse_color, unpack_estimated_T,
)


# ============================================================================
# Comparative advantage against distance, within a sector (tests 1-6)
# ============================================================================

# Test 1 -- the within-sector spread of estimated comparative advantage.
# The sector-ordering helpers and `_buyer_weights` these tests share are in utils.



def comparative_advantage_frame(data):
    """
    One row per (sector, active attraction area): the estimated T, its deviation from the
    sector's median area in BOTH log bases, its rank, and how many REGIONS the area
    contributes to that sector's competition (the areas differ enormously in that count,
    and an area's edge is only as useful as the locations that can exploit it).

    `log10_T_dev` is the column to read: one unit is an order of magnitude of comparative
    advantage, and because the trade-off against distance is scale-free
    (`log10 d_equivalent = log10_T_dev / (theta*alpha)`), a decade of T reads directly as
    `1/(theta*alpha)` decades of distance. `log_T_dev` keeps the natural log, which is the
    unit every theta*alpha-scaled formula in the tests below is written in.
    """
    estimate = unpack_estimated_T(data)
    aa_of_ze, CELL_MASK, AA_ACTIVE = data["aa_of_ze"], data["CELL_MASK"], data["AA_ACTIVE"]
    aa_names = aa_display_names(data)          # commuting-zone NAMES, not ZE codes
    rows = []
    for s in range(data["S"]):
        active_areas = np.flatnonzero(AA_ACTIVE[s])
        if active_areas.size == 0:
            continue
        T = estimate["T"][s, active_areas]
        median_logT = np.median(np.log(np.maximum(T, 1e-300)))
        order = np.argsort(-T)
        rank = np.empty(active_areas.size, dtype=int)
        rank[order] = np.arange(1, active_areas.size + 1)
        for k, a in enumerate(active_areas):
            rows.append({
                "sector": data["sector_names"][s],
                "area": aa_names[a],
                "area_code": data["aa_names"][a],
                "T": float(T[k]),
                "log_T_dev": float(np.log(max(T[k], 1e-300)) - median_logT),
                "log10_T_dev": float((np.log(max(T[k], 1e-300)) - median_logT) / np.log(10.0)),
                "rank": int(rank[k]),
                "is_reference": bool(data["T_REF_AA"][s] == a),
                "n_cells": int(np.sum(CELL_MASK[s] & (aa_of_ze == a))),
            })
    return pd.DataFrame(rows)

def plot_ca_distribution(data, figsize=None, save_to=None, floor_ratio=1e-4):
    """
    The within-sector spread of estimated comparative advantage: one row per sector, one
    dot per active attraction area, on a LOG x axis centred at the sector's median area,
    so the axis reads in POWERS OF TEN of the ratio T_as / median_s(T) — a tick of 10
    means "ten times the median area", and one decade is one order of magnitude (and,
    divided by theta*alpha, decades of distance). The top area is marked and named — a
    sector whose best area sits far to the right of the pack is one where comparative
    advantage, not geography, decides.

    `floor_ratio` clips the left edge of the axis. `comparative_advantage_frame` guards
    T against zero with 1e-300, so a degenerate area would otherwise land at 10^-300 and
    compress every other sector into a single tick.
    """
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    ca_frame = comparative_advantage_frame(data)
    order = [s for s in _sectors_in_code_order(data) if s in set(ca_frame["sector"])][::-1]

    fig, ax = plt.subplots(figsize=figsize or get_figsize(width_fraction=0.95, height_fraction=0.75))
    for y, sector in enumerate(order):
        sector_rows = ca_frame[ca_frame["sector"] == sector]
        ratio = np.power(10.0, sector_rows["log10_T_dev"].to_numpy(float))
        ax.scatter(ratio, np.full(len(sector_rows), y), s=22, color=sim_color,
                   alpha=0.65, zorder=2)
        top_row = sector_rows.loc[sector_rows["rank"].idxmin()]
        r_top = 10.0 ** float(top_row["log10_T_dev"])
        ax.scatter([r_top], [y], s=60, marker="D", color=reference_color, zorder=3)
        ax.annotate(str(top_row["area"]), (r_top, y), fontsize=fs(8),
                    xytext=(6, 4), textcoords="offset points", color=reference_color)

    # set_xscale BEFORE margins: margins computed on a linear axis then reinterpreted in
    # log units would push the extreme points against the frame.
    ax.set_xscale("log")
    ax.margins(x=0.12)
    ax.axvline(1.0, color="0.6", linewidth=0.8)          # the median area, ratio = 1

    all_ratio = np.power(10.0, ca_frame["log10_T_dev"].to_numpy(float))
    finite = all_ratio[np.isfinite(all_ratio) & (all_ratio > 0)]
    if finite.size:
        ax.set_xlim(left=max(float(finite.min()) * 0.5, floor_ratio))

    # Plain numbers rather than 10^k: the ratios of interest sit within a few decades of
    # 1, and "30" is read faster than "3x10^1". %g would fall back to scientific notation
    # below 1e-4 on its own, hence the explicit decimal count per decade.
    def _plain(v, _):
        if v <= 0:
            return ""
        k = int(np.floor(np.log10(v)))
        return f"{v:,.0f}" if k >= 0 else f"{v:.{-k}f}"

    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(FuncFormatter(_plain))
    ax.xaxis.set_minor_formatter(NullFormatter())

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order,
                       fontsize=name_axis_fontsize(ax, len(order), size_at_ref=9))
    ax.set_xlabel(r"$\hat T_{as}$ relative to the sector's median area "
                  r"(ratio, log scale; one decade = one order of magnitude)")
    ax.set_ylabel("Sector")
    ax.set_title(f"Comparative advantage within sector — {data['industry']}, "
                 rf"$\hat\mu_{data['mu']}$")
    _despine(ax)
    fig.tight_layout()
    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        fig.savefig(save_to, bbox_inches="tight")
    return ax

# Test 2 — the edge in distance units, against the distances the geography offers.
from matplotlib.ticker import LogFormatterSciNotation
def ca_distance_equivalence(data):
    """
    How much distance a comparative-advantage gap is worth, against how much distance the
    country actually offers.

    `equiv_log_d = Delta log T / (theta*alpha)` is the log distance ratio that exactly
    offsets the gap between the top area and the median one — how much further from a
    buyer a REGION of the top area may sit than a competing REGION of the median area.
    It is compared with two benchmarks, both computed on cell-to-buyer distances:

      `geo_log_range`  the EXTREME spread, pooled max - min of log distance across all
                       competing cells and buyers. Anchored at the own-region cell, so
                       one degenerate observation sets it: it is the largest gap the
                       geography could ever offer;
      `geo_log_spread` the TYPICAL spread: p90 - p10 of log distance across the cells
                       competing for a given buyer, averaged over buyers with their
                       purchase weights.

    These two are averages over a distribution. `ca_win_margin` does the same comparison
    exactly, buyer by buyer, and is the one to quote; this table is the summary version.
    """
    geometry = sourcing_geometry(data)
    theta_alpha = geometry["theta_alpha"]        # theta*alpha; under binned tau, the mean local elasticity
    ca_frame = comparative_advantage_frame(data)
    rows = []
    for s, sector_block in geometry["by_sector"].items():
        name = data["sector_names"][s]
        sector_rows = ca_frame[ca_frame["sector"] == name].sort_values("rank")
        if sector_rows.empty:
            continue
        logT = np.log(np.maximum(sector_rows["T"].to_numpy(), 1e-300))
        d_top_med = float(logT[0] - np.median(logT))
        d_top_2nd = float(logT[0] - logT[1]) if logT.size > 1 else np.nan
        log_distance = np.log(sector_block["distance"])
        geo_range = float(log_distance.max() - log_distance.min())
        per_buyer = np.percentile(log_distance, 90, axis=0) - np.percentile(log_distance, 10, axis=0)
        geo_spread = float(per_buyer @ _buyer_weights(data))
        equiv = d_top_med / theta_alpha if theta_alpha > 0 else np.inf
        rows.append({
            "sector": name, "top_area": sector_rows["area"].iloc[0], "n_areas": len(sector_rows),
            "dlogT_top_vs_median": d_top_med, "dlogT_top_vs_2nd": d_top_2nd,
            "equiv_log_d": equiv,
            "equiv_distance_ratio": float(np.exp(min(equiv, 700))),
            "geo_log_range": geo_range, "geo_log_spread": geo_spread,
            "CA_beats_any_geography": bool(equiv > geo_range),
            "CA_beats_typical_geography": bool(equiv > geo_spread),
        })
    df = _by_sector_code(pd.DataFrame(rows).set_index("sector"), data)
    df.attrs["theta_alpha"] = theta_alpha
    return df

def plot_ca_distance_equivalence(data, figsize=None, save_to=None, reference_km=1.0):
    """
    What the top area's comparative-advantage edge is worth, in kilometres.

    One POINT per sector, on a LOG x axis. The edge is a RATIO --
    `exp(Delta log T / (theta*alpha))` is the factor by which a region of the top area
    may be further from a buyer than a competing region of the median area and still
    win. With `reference_km = 1` the point IS that factor read as kilometres:

        equiv_km = exp(Delta log T / (theta * alpha))

    Read it as: a supplier of the top area wins a buyer at that distance against a
    median area's supplier sitting one kilometre away. The axis is logarithmic because
    theta*alpha is small, so the factor spans orders of magnitude across sectors.

    A point rather than a bar because the quantity is a RATIO on a log axis: a bar's
    length is read against its origin, and on a log axis that origin is arbitrary (it is
    wherever the axis happens to start), so the bar encodes the axis limits as much as
    the number. A point encodes only its position, which is the whole of the content.

    Each point is NAMED with its TOP AREA -- the attraction area holding the edge, by its
    anchor commuting zone -- in the same form test 1 names it, beside the mark rather than
    on the axis. Without it the figure says how large the edge is and not whose it is, and
    that name is the first thing a reader of the table looks up; beside the point it is
    read WITH the number, and the y axis stays a plain list of sectors.

    The dashed line is the BILATERAL MEDIAN cell-to-buyer distance -- the distance a
    typical supplier-buyer link actually spans in this industry. It is drawn for scale
    only: it is NOT the anchor of the bars, and a bar exceeding it does not by itself
    establish that comparative advantage beats geography. Test 3 (`ca_win_margin`) is
    the buyer-by-buyer comparison that does.
    """
    from matplotlib.ticker import LogFormatterSciNotation, LogLocator, NullFormatter

    df = ca_distance_equivalence(data)[::-1]        # A129-code order, top to bottom
    geometry = sourcing_geometry(data)

    # Median over all cell-buyer pairs, pooled across sectors: a scale benchmark, not an
    # anchor. Unweighted, so each buyer counts in proportion to the cells competing for
    # it rather than to its purchases.
    median_by_sector = [float(np.median(sector_block["distance"])) for sector_block in geometry["by_sector"].values()]
    median_km = float(np.mean(median_by_sector)) if median_by_sector else np.nan

    equivalent_km = reference_km * np.exp(np.minimum(df["equiv_log_d"].to_numpy(float), 700.0))
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=figsize or get_figsize(width_fraction=0.95, height_fraction=0.62))
    ax.plot(equivalent_km, y, linestyle="none", marker="o", markersize=7, color=reference_color,
            markeredgecolor="black", markeredgewidth=0.5)
    for row, (point_km, area) in enumerate(zip(equivalent_km, df["top_area"].astype(str))):
        ax.annotate(area, (point_km, row), fontsize=fs(8), xytext=(6, 4),
                    textcoords="offset points", color=reference_color)
    ax.set_xscale("log")
    ax.set_ylim(-0.6, len(df) - 0.4)

    if np.isfinite(median_km) and median_km > 0:
        ax.axvline(median_km, color="black", ls="dashed", lw=1.0, alpha=0.7)
        # anchored to the axis top and set to the LEFT of the line: with no title there
        # is no margin above, and the line sits near the right edge, so a label placed
        # to its right runs off the figure
        ax.text(median_km, ax.get_ylim()[1],
                f"{median_km:,.0f} km ",
                fontsize=fs(8), va="top", ha="right", color="0.3")

    formatter = LogFormatterSciNotation(base=10.0)
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_minor_formatter(NullFormatter())

    # Limits must now bracket the bars AND the benchmark line: the line is no longer the
    # anchor, so nothing guarantees it falls inside the range the bars span.
    finite = equivalent_km[np.isfinite(equivalent_km) & (equivalent_km > 0)]
    anchors = [v for v in (float(finite.min()) if finite.size else np.nan, median_km)
               if np.isfinite(v) and v > 0]
    if anchors:
        ax.set_xlim(left=min(anchors) * 0.5)
    if finite.size and np.isfinite(median_km):
        # room for the NAME, not just for the number: the annotation runs to the right of
        # the widest point, and at x2 that point sits ~90% across and its label overhangs
        # the frame
        ax.set_xlim(right=max(float(finite.max()), median_km) * 4.0)

    ax.set_yticks(y)
    ax.set_yticklabels(df.index,
                       fontsize=name_axis_fontsize(ax, len(df), size_at_ref=9))
    ax.set_xlabel("Distance (km)")
    ax.grid(axis="x", which="major", linestyle="dashed", alpha=0.4)
    _despine(ax)
    fig.tight_layout()
    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        fig.savefig(save_to, bbox_inches="tight")
    return ax

# Test 3 — the exact comparison, buyer by buyer and region by region.

def ca_win_margin(data):
    """
    The decisive test, buyer by buyer and REGION by region — no averaging of distances.

    For each sector, let A* be the area with the highest T. For each downstream buyer r,
    every competing cell l has score

        log psi_lr = log T_{a(l)} - theta*alpha*log d_{lr},

    so A* takes the buyer iff its best cell outscores the best cell OUTSIDE it. Two
    numbers come out of that comparison, both in LOG-DISTANCE units (divide by
    theta*alpha), which is what makes them readable:

      win_margin_r  = [max_{l in A*} score - max_{l not in A*} score] / (theta*alpha)
                      how much FURTHER the favoured area's best region could be from
                      this buyer and still win. Positive = it wins.
      within_penalty_r = log d(worst cell of A*) - log d(best cell of A*)
                      the handicap A*'s own geography imposes on its worst region. Inside
                      an area T cancels exactly, so this comparison is PURE DISTANCE —
                      it is the variation that identifies alpha, and it is invisible to
                      any statement written in terms of area-level distances.

    Both are averaged over buyers with their purchase weights. `share_buyers_won` is the
    purchase-weighted share of buyers the favoured area actually takes; comparing
    `mean_win_margin` with `mean_within_penalty` says whether belonging to the right area
    or sitting in the right region of it matters more.
    """
    geometry = sourcing_geometry(data)
    theta_alpha, buyer_weights = geometry["theta_alpha"], _buyer_weights(data)
    est_T = geometry["T"]
    rows = []
    for s, sector_block in geometry["by_sector"].items():
        active_areas = np.flatnonzero(data["AA_ACTIVE"][s])
        if active_areas.size == 0:
            continue
        top_area = int(active_areas[np.argmax(est_T[s, active_areas])])
        in_top = sector_block["areas"] == top_area
        if not in_top.any() or in_top.all():
            continue                       # no cross-area comparison to make
        score = sector_block["log_psi"]                                   # (n_cell, R_d)
        log_distance = np.log(sector_block["distance"])
        best_top = score[in_top].max(axis=0)
        best_out = score[~in_top].max(axis=0)
        margin = (best_top - best_out) / theta_alpha if theta_alpha > 0 else np.full_like(best_top, np.inf)
        penalty = log_distance[in_top].max(axis=0) - log_distance[in_top].min(axis=0)
        rows.append({
            "sector": data["sector_names"][s],
            "top_area": aa_display_names(data)[top_area],
            "n_cells_top_area": int(in_top.sum()),
            "share_buyers_won": float((margin > 0) @ buyer_weights),
            "mean_win_margin": float(margin @ buyer_weights),
            "min_win_margin": float(margin.min()),
            "mean_within_penalty": float(penalty @ buyer_weights),
            "area_beats_own_geography": bool(float(margin @ buyer_weights) > float(penalty @ buyer_weights)),
        })
    df = pd.DataFrame(rows)
    return _by_sector_code(df.set_index("sector"), data) if len(df) else df


def plot_ca_win_margin(data, figsize=None, save_to=None):
    """
    The exact test, per sector: how much further the top area's best region could be from
    the average buyer and still win (bar), against the distance handicap that area's own
    geography already imposes on its worst region (marker). Both in log-distance units,
    so they are directly comparable — a bar shorter than its marker is a sector where
    WHICH REGION of the favoured area a supplier sits in matters more than the area's
    edge itself.
    """
    df = ca_win_margin(data)
    if not len(df):
        raise ValueError("no sector has cells both inside and outside its top area — "
                         "there is no cross-area comparison to draw.")
    df = df[::-1]
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=figsize or get_figsize(width_fraction=0.95, height_fraction=0.7))
    ax.barh(y, df["mean_win_margin"],
            color=[reference_color if beats_geography else sim_color
                   for beats_geography in df["area_beats_own_geography"]], alpha=0.85,
            label="win margin of the top area (log distance it could give away)")
    ax.scatter(df["mean_within_penalty"], y, marker="|", s=220, color="black", zorder=3,
               label="within-area distance handicap (best to worst region of that area)")
    ax.axvline(0.0, color="0.6", linewidth=0.8)
    for i, (_, r) in enumerate(df.iterrows()):
        ax.annotate(f"{100 * r['share_buyers_won']:.0f}{pct()} of buyers",
                    (max(r["mean_win_margin"], 0.0), i), fontsize=fs(8),
                    xytext=(4, -3), textcoords="offset points")
    ax.set_yticks(y)
    ax.set_yticklabels(df.index,
                       fontsize=name_axis_fontsize(ax, len(df), size_at_ref=9))
    ax.set_xlabel("log distance")
    ax.set_title(f"Does the favoured area actually win? — {data['industry']}, "
                 rf"$\hat\mu_{data['mu']}$", fontsize=fs(11))
    ax.legend(frameon=False, fontsize=fs(8), loc="upper center", bbox_to_anchor=(0.5, -0.18))
    _despine(ax)
    fig.tight_layout()
    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        fig.savefig(save_to, bbox_inches="tight")
    return ax


# Test 4 — the exact variance decomposition of log psi across competing cells.

def ca_variance_decomposition(data):
    """
    The exact split of the dispersion of log psi across competing cells.

    For a given buyer r, log psi_{l,r} = log T_l - theta*log tau_{l,r}, so

        Var_l(log psi) = Var_l(log T) + Var_l(theta log tau) - 2 Cov_l(log T, theta log tau)

    holds with no approximation; under the power law theta*log tau = theta*alpha*log d and
    this is the familiar (theta*alpha)^2 Var(log d) split. Variances are taken ACROSS CELLS at a fixed buyer —
    which is the competition the model actually runs — then averaged over buyers with
    their purchase weights. `sd_logT` against `sd_dist` is the like-for-like size of the
    two forces.

    The covariance term is NOT free of the estimator: T is the Sinkhorn image of the
    observed sourcing shares, which loads onto comparative advantage whatever distance
    takes away, so a positive Cov(log T, log d) — hence a negative contribution
    `share_covariance` — is the expected reading. `ca_covariance_benchmark` splits it.
    """
    geometry = sourcing_geometry(data)
    theta, buyer_weights = geometry["theta"], _buyer_weights(data)
    rows = []
    for s, sector_block in geometry["by_sector"].items():
        if sector_block["cells"].size < 2:
            continue
        logT = np.log(np.maximum(sector_block["T_cell"], 1e-300))
        distance_pull = theta * sector_block["log_tau"]                         # (n_cell, R_d), the distance pull
        var_logT = float(np.var(logT))
        var_distance = np.var(distance_pull, axis=0)                            # per buyer
        covariance = np.array([np.cov(logT, distance_pull[:, r], bias=True)[0, 1] for r in range(distance_pull.shape[1])])
        v_psi = var_logT + var_distance - 2 * covariance
        vd_bar, cov_bar, vpsi_bar = float(var_distance @ buyer_weights), float(covariance @ buyer_weights), float(v_psi @ buyer_weights)
        rows.append({
            "sector": data["sector_names"][s],
            "sd_logT": np.sqrt(var_logT),
            "sd_dist": np.sqrt(vd_bar),
            "ratio_CA_over_distance": np.sqrt(var_logT) / np.sqrt(vd_bar)
            if vd_bar > 0 else np.inf,
            "share_CA": var_logT / vpsi_bar if vpsi_bar > 0 else np.nan,
            "share_distance": vd_bar / vpsi_bar if vpsi_bar > 0 else np.nan,
            "share_covariance": -2 * cov_bar / vpsi_bar if vpsi_bar > 0 else np.nan,
            "sd_log_psi": np.sqrt(vpsi_bar),
        })
    return _by_sector_code(pd.DataFrame(rows).set_index("sector"), data)


# Test 5 — where the covariance term comes from, given how T was obtained.

def ca_covariance_benchmark(data):
    """
    Where the covariance term of the decomposition comes from — the data, or the
    inversion.

    Under profiling, T is chosen so the model reproduces the observed sourcing shares:
    at the area level, log T_a = log gamma^emp_a - log M_a(alpha) + const, with M the
    market-access term. Hence, cell by cell,

        Cov(log T, log d) = Cov(log gamma^emp, log d) - Cov(log M, log d),

    and since M falls with distance to markets the second term is negative, i.e. the
    inversion MECHANICALLY pushes Cov(log T, log d) up: a remote area that nonetheless
    sells must be handed a higher T. This function reports the two pieces, taking
    log M = log gamma^emp - log T as the residual (exact up to the per-sector constant,
    which drops out of a covariance). Read `cov_gamma` as the gradient in the data and
    `-cov_M` as what the inversion added on top of it.

    At alpha = 0 the market-access term is common within a sector and `cov_M` must
    vanish, leaving `cov_T == cov_gamma` — the limiting case to check the split against.
    """
    geometry = sourcing_geometry(data)
    theta_alpha, buyer_weights = geometry["theta_alpha"], _buyer_weights(data)
    emp_gamma_aa, aa_of_ze = data["emp_gamma_aa"], data["aa_of_ze"]
    rows = []
    for s, sector_block in geometry["by_sector"].items():
        cells, areas = sector_block["cells"], sector_block["areas"]
        emp_gamma = emp_gamma_aa[s, areas]
        keep = (emp_gamma > 0) & (sector_block["T_cell"] > 0)
        if keep.sum() < 2:
            continue
        logT = np.log(sector_block["T_cell"][keep])
        logG = np.log(emp_gamma[keep])
        logM = logG - logT                                  # the market-access residual
        log_distance = np.log(sector_block["distance"][keep])
        def _buyer_covariance(x):
            return float(np.array([np.cov(x, log_distance[:, r], bias=True)[0, 1]
                                   for r in range(log_distance.shape[1])]) @ buyer_weights)
        cov_T, cov_G, cov_M = _buyer_covariance(logT), _buyer_covariance(logG), _buyer_covariance(logM)
        sd_ld = float(np.sqrt(np.array([np.var(log_distance[:, r]) for r in range(log_distance.shape[1])]) @ buyer_weights))
        rows.append({
            "sector": data["sector_names"][s],
            "cov_T": cov_T, "cov_gamma": cov_G, "cov_M": cov_M,
            "check_gamma_minus_M": cov_G - cov_M - cov_T,      # 0 by construction
            "corr_T_d": cov_T / (np.std(logT) * sd_ld) if sd_ld > 0 and np.std(logT) > 0 else np.nan,
            "corr_gamma_d": cov_G / (np.std(logG) * sd_ld) if sd_ld > 0 and np.std(logG) > 0 else np.nan,
            "share_covariance": np.nan,
        })
    df = pd.DataFrame(rows).set_index("sector")
    variance_table = ca_variance_decomposition(data)
    df["share_covariance"] = variance_table["share_covariance"].reindex(df.index)
    df.attrs["theta_alpha"] = theta_alpha
    return _by_sector_code(df, data)


# Test 6 — how much sourcing DISTANCE comparative advantage is buying, in kilometres.

def _representative_buyer(data, geometry=None):
    """
    The buyer the alignment figure is drawn for: the one of MEDIAN sourcing reach.

    Per buyer, take the sourcing-weighted average distance `d_rs` within each sector and
    average across sectors; the representative buyer sits at the median of that. It is a
    buyer the model actually sources for, and it is neither the most local nor the most
    remote — which is what makes it representative rather than flattering.
    """
    geometry = sourcing_geometry(data) if geometry is None else geometry
    blocks = [b for b in geometry["by_sector"].values() if b["cells"].size >= 2]
    if not blocks:
        raise ValueError("no sector with two competing cells; nothing to draw")
    reach = np.mean([(b["rho"] * b["distance"]).sum(axis=0) for b in blocks], axis=0)
    order = np.argsort(reach)
    return int(order[order.size // 2])


def alignment_frame(data, buyer=None, geometry=None):
    """
    The panel behind the alignment regression: one row per (sector, buyer, competing cell).

    Columns are in the units the section reports. `distance_km` is the cell's distance to
    that buyer IN KILOMETRES, not its log — the object of interest is
    `d_{rs} = sum_l rho_lrs d_lr`, the average distance the euro travels, and a statement
    about `d_rs` has to be built out of `d`, not out of `log d`. `log_T` stays in logs
    because comparative advantage enters the allocation multiplicatively and is identified
    only up to a per-sector scale, which the fixed effect absorbs.

    `weight` is the sourcing probability `rho_lrs` times the buyer's purchase weight, so a
    weighted mean over the panel is the sourcing-weighted average the model performs and
    not an average over the places that could in principle have supplied.

    `buyer` selects a single downstream region (the figure); `None` keeps them all (the
    regression). The panel then has `n_cell x n_buyer` rows per sector, which is large but
    is what a within-(sector x buyer) estimand requires. `weight` is normalised to sum to
    one over the panel, so every sector carries equal weight and every buyer its purchase
    weight; the slope of a WLS is invariant to that scale, but the reported dispersions
    are not.
    """
    geometry = sourcing_geometry(data) if geometry is None else geometry
    buyer_weights = _buyer_weights(data)
    buyers = range(buyer_weights.size) if buyer is None else (int(buyer),)
    aa_names = aa_display_names(data)
    frames = []
    for s, sector_block in geometry["by_sector"].items():
        if sector_block["cells"].size < 2:
            continue
        name = data["sector_names"][s]
        for r in buyers:
            rho_buyer = sector_block["rho"][:, r]
            rho_total = rho_buyer.sum()
            if not np.isfinite(rho_total) or rho_total <= 0:
                continue
            frames.append(pd.DataFrame({
                "sector": name,
                "buyer": aa_names[r] if r < len(aa_names) else str(r),
                "group": f"{name}|{r}",
                "ze_index": sector_block["cells"] + 1,
                "area": [aa_names[a] for a in sector_block["areas"]],
                "log_T": np.log(np.maximum(sector_block["T_cell"], 1e-300)),
                "distance_km": sector_block["distance"][:, r],
                "rho": rho_buyer / rho_total,
                "weight": (rho_buyer / rho_total) * buyer_weights[r],
            }))
    if not frames:
        raise ValueError("no (sector, buyer) block with two competing cells")
    df = pd.concat(frames, ignore_index=True)
    # rho sums to one per (sector, buyer) and the purchase weights sum to one across
    # buyers, so each SECTOR carries total weight one: dividing by their number makes
    # `weight` a probability over the whole panel and states the aggregation — every
    # sector counts equally, every buyer in proportion to its purchases.
    n_sec = df["sector"].nunique()
    df["weight"] = df["weight"] / n_sec
    df.attrs["n_sectors"] = n_sec
    return df


def alignment_regression(data, frame=None, by_sector=False, cluster="buyer"):
    """
    Where comparative advantage pulls sourcing, as one weighted regression.

        distance_km = a_{s,r} + b * log T_hat + e,      weights rho_lrs * omega_r

    fitted by `pyfixest.feols` with a sector x downstream-region fixed effect. `b` is in
    KILOMETRES per log point of comparative advantage: how much closer to the buyer a
    supplier is, per unit of the capability that wins it the sale.

    The fixed effect is what makes the estimand the right one. The identity behind the
    counterfactual holds for ONE sector and ONE buyer at a time, so the object is a
    within-(sector x buyer) slope; absorbing that effect is exactly the restriction, and
    the sourcing weights are what make it the average the model performs rather than an
    average over candidate locations. Both are specification decisions, not tuning: run it
    unweighted and the same regression answers a different question, about the cells
    rather than about the sourcing.

    `by_sector=True` returns one row per sector, fitted with a buyer fixed effect inside
    that sector — the numbers `ca_distance_leverage` reports. A sector whose comparative
    advantage takes a single value is collinear with its own fixed effect and is returned
    as NaN rather than dropped silently.

    The standard error is reported because the library returns it, but it has no sampling
    interpretation here: `T_hat` and `d` are model objects, not a sample, so the dispersion
    around this line is the spread of an estimated deterministic allocation.
    """
    panel = alignment_frame(data) if frame is None else frame

    def _fit(sector_panel, fixed_effect):
        if sector_panel["log_T"].nunique() < 2:
            return {"slope_km_per_logT": np.nan, "se": np.nan,
                    "n_obs": len(sector_panel), "n_groups": sector_panel[fixed_effect].nunique()}
        wls_fit = pf.feols(f"distance_km ~ log_T | {fixed_effect}", data=sector_panel, weights="weight",
                       weights_type="aweights",
                       vcov=None if cluster is None else {"CRV1": cluster})
        return {"slope_km_per_logT": float(wls_fit.coef()["log_T"]),
                "se": float(wls_fit.se()["log_T"]),
                "n_obs": len(sector_panel), "n_groups": sector_panel[fixed_effect].nunique()}

    if not by_sector:
        result = _fit(panel, "group")
        result["sd_rho_logT"] = _weighted_sd(panel, "log_T")
        result["sd_rho_distance_km"] = _weighted_sd(panel, "distance_km")
        return pd.Series(result, name=f"{data['industry']}, mu = {data['mu']}")

    rows = {}
    for name, sector_panel in panel.groupby("sector", sort=False):
        sector_result = _fit(sector_panel, "buyer")
        sector_result["sd_rho_logT"] = _weighted_sd(sector_panel, "log_T")
        sector_result["sd_rho_distance_km"] = _weighted_sd(sector_panel, "distance_km")
        rows[name] = sector_result
    return _by_sector_code(pd.DataFrame(rows).T, data)


def _weighted_sd(panel, column):
    """Sourcing-weighted sd of `col`, demeaned WITHIN (sector x buyer) as the FE does."""
    weights = panel["weight"].to_numpy(float)
    x = panel[column].to_numpy(float)
    group = panel["group"].to_numpy()
    group_weight = pd.Series(weights).groupby(group).transform("sum").to_numpy()
    group_mean = pd.Series(weights * x).groupby(group).transform("sum").to_numpy() / np.maximum(group_weight, 1e-300)
    return float(np.sqrt((weights @ (x - group_mean) ** 2) / weights.sum())) if weights.sum() > 0 else np.nan


def ca_distance_leverage(data, n_grid=201, regression=None):
    """
    What equalising comparative advantage does to the average sourcing distance, in km.

    The object is `d_{rs} = sum_l rho_lrs d_lr`, the distance the average euro travels
    from downstream region `r` when it buys product `s`. Dialling comparative advantage
    down along `rho_l(t) ∝ T_l^t d_lr^{-theta*alpha}` moves it at exactly the rate

        d/dt E_{rho(t)}[d] = Cov_{rho(t)}(log T, d),

    an exponential-family identity, so the whole effect of the `Equal comparative advantage` regime is
    the path integral of a covariance between capability and distance IN KILOMETRES. It is
    a covariance, not a variance: a `T_hat` that varies widely but is uncorrelated with the
    buyer's geography reallocates sourcing across regions without sending the euro one
    kilometre further.

    `delta_km` is that effect, computed as two evaluations of a closed form, and the
    amplification section reaches the same object by reallocating euros — agreement is a
    cross-check of two independent routes. `check_identity` re-derives it by integrating
    the covariance over the grid, so it gates the derivative FORMULA against the mean it
    differentiates, up to the trapezoid error of the grid; it is read relative to the
    effect, not against a fixed zero.

    `slope_km_per_logT` comes from `alignment_regression`, i.e. from `pyfixest`, and is the
    line the alignment figure draws. It is the covariance divided by the sourcing-weighted
    variance of `log T_hat`, so it is the same statistic in the units the figure can show.
    """
    trapz = getattr(np, "trapezoid", None) or np.trapz
    geom = sourcing_geometry(data)
    theta_alpha, buyer_weights = geom["theta_alpha"], _buyer_weights(data)
    t_grid = np.linspace(0.0, 1.0, int(n_grid))
    rows = []
    for s, sector_block in geom["by_sector"].items():
        if sector_block["cells"].size < 2:
            continue
        logT = np.log(np.maximum(sector_block["T_cell"], 1e-300))[:, None]   # (n_cell, 1)
        d_km = sector_block["distance"]                                      # (n_cell, R_d), km
        distance_pull = -geom["theta"] * sector_block["log_tau"]                      # -theta*log tau

        def allocate(t):
            log_score = t * logT + distance_pull
            allocation = np.exp(log_score - log_score.max(axis=0, keepdims=True))
            return allocation / allocation.sum(axis=0, keepdims=True)

        cov_t, mean_distance_t = np.empty(t_grid.size), np.empty(t_grid.size)
        for k, t in enumerate(t_grid):
            allocation = allocate(t)
            mean_logT, mean_distance = (allocation * logT).sum(0), (allocation * d_km).sum(0)
            cov_t[k] = ((allocation * logT * d_km).sum(0) - mean_logT * mean_distance) @ buyer_weights     # km per log point
            mean_distance_t[k] = mean_distance @ buyer_weights                                         # km

        d_direct = float(mean_distance_t[0] - mean_distance_t[-1])          # equalising T: t = 1 -> 0
        d_path = float(-trapz(cov_t, t_grid))
        rows.append({
            "sector": data["sector_names"][s],
            "d_km": float(mean_distance_t[-1]),
            "d_km_no_CA": float(mean_distance_t[0]),
            "delta_km": d_direct,
            "cov_rho_T_d_km": float(cov_t[-1]),
            "check_identity": d_direct - d_path,
        })
    df = _by_sector_code(pd.DataFrame(rows).set_index("sector"), data)
    regression = alignment_regression(data, by_sector=True) if regression is None else regression
    for column in ("slope_km_per_logT", "sd_rho_logT", "sd_rho_distance_km"):
        df[column] = pd.to_numeric(regression[column].reindex(df.index), errors="coerce")
    df = df[["d_km", "d_km_no_CA", "delta_km", "slope_km_per_logT",
             "cov_rho_T_d_km", "sd_rho_logT", "sd_rho_distance_km", "check_identity"]]
    df.attrs["theta_alpha"] = theta_alpha
    return df


def ca_leverage_report(data, leverage=None):
    """
    The one-line reading of Test 6: what the alignment is worth in kilometres.

    Prints the sector medians, because the sector distribution is what the paper quotes,
    and flags a broken identity rather than leaving it to be found by eye.
    """
    leverage = ca_distance_leverage(data) if leverage is None else leverage
    if leverage.empty:
        print("  no sector with two competing cells; nothing to report")
        return leverage
    scale = float(np.maximum(np.abs(leverage["delta_km"]).max(), 1e-9))
    identity_gap = float(np.max(np.abs(leverage["check_identity"]))) / scale
    print(f"[{data['industry']}, mu = {data['mu']}]  theta*alpha = "
          f"{leverage.attrs['theta_alpha']:.3f}")
    print(f"  average sourcing distance d_rs = {leverage['d_km'].median():.1f} km at the "
          f"sector median; equalising T takes it to {leverage['d_km_no_CA'].median():.1f} km "
          f"({leverage['delta_km'].median():+.1f} km)")
    print(f"  slope of distance on log T (pyfixest, sector x buyer FE, rho weights): "
          f"{leverage['slope_km_per_logT'].median():+.1f} km per log point")
    print(f"  identity |direct - path| / effect = {identity_gap:.2e}"
          + ("" if identity_gap < 1e-5 else "   <-- INVESTIGATE: raise n_grid"))
    return leverage


def plot_spatial_alignment(data, buyer=None, n_label=4, figsize=None, save_to=None,
                           frame=None):
    """
    Where comparative advantage pulls sourcing.

    Each dot is an upstream commuting zone competing to supply one downstream buyer:
    its comparative advantage on `x`, its DISTANCE TO THAT BUYER IN KILOMETRES on `y`,
    with the area of the dot given by the sourcing probability `rho_lr`. The picture
    therefore answers the question the counterfactual asks — not "how unequal is T" but
    "does the buyer's euro come from places that are productive AND close".

    The line is the fit `alignment_regression` reports for this buyer, so the slope the
    eye reads IS the number annotated on the panel: kilometres of sourcing distance per
    log point of comparative advantage. A downward line means the productive suppliers
    are the near ones and equalising `T` would push the euro outward; a flat line means
    comparative advantage is orthogonal to the buyer's geography and equalising it would
    move the euro nowhere.

    Both axes are deviations from their sourcing-weighted SECTOR mean, which is what the
    fixed effect removes: `T_hat` is identified only up to a per-sector scale, so a pooled
    cloud in levels would manufacture a slope out of the normalisation. The `n_label`
    areas taking the most sourcing are named.
    """
    panel = (alignment_frame(data, buyer=_representative_buyer(data) if buyer is None
                          else buyer) if frame is None else frame)
    regression = alignment_regression(data, frame=panel, cluster=None)
    weights = panel["weight"].to_numpy(float)
    sector = panel["sector"].to_numpy()
    def _dev(col):
        x = panel[col].to_numpy(float)
        group_weight = pd.Series(weights).groupby(sector).transform("sum").to_numpy()
        group_mean = pd.Series(weights * x).groupby(sector).transform("sum").to_numpy() / np.maximum(group_weight, 1e-300)
        return x - group_mean
    x, y = _dev("log_T"), _dev("distance_km")

    fig, ax = plt.subplots(figsize=figsize or get_figsize(width_fraction=0.95, height_fraction=0.72))
    scale = weights / weights.max() if weights.max() > 0 else weights
    ax.scatter(x, y, s=18 + 320 * scale, color=reference_color, alpha=0.55,
               edgecolor="black", linewidth=0.4, zorder=2)
    slope = regression["slope_km_per_logT"]
    if np.isfinite(slope):
        x_line = np.array([x.min(), x.max()])
        ax.plot(x_line, slope * x_line, color="black", linewidth=1.4, zorder=3)
    ax.axhline(0.0, color="0.75", linewidth=0.8, zorder=1)
    ax.axvline(0.0, color="0.75", linewidth=0.8, zorder=1)

    if n_label:
        for _, zone_row in panel.assign(_x=x, _y=y).nlargest(int(n_label), "weight").iterrows():
            ax.annotate(str(zone_row["area"]), (zone_row["_x"], zone_row["_y"]), fontsize=fs(8),
                        xytext=(6, 4), textcoords="offset points", color=reference_color)
    ax.annotate(rf"${slope:+.0f}$ km per log point", (0.03, 0.05), xycoords="axes fraction",
                fontsize=fs(9), color="0.25")
    ax.set_xlabel(r"Comparative advantage $\log \hat{T}$ (deviation)")
    ax.set_ylabel("Distance to the buyer (km, deviation)")
    _despine(ax)
    fig.tight_layout()
    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        fig.savefig(save_to, bbox_inches="tight")
    return ax


# --- The alignment, buyer by buyer --------------------------------------------
#
# `ca_variance_decomposition` reports one covariance per SECTOR, already averaged
# over buyers with their purchase weights, and the paper quotes its median. That
# hides the fact the two industries actually differ on. A sector median of -0.47
# is consistent with every buyer facing the same mild alignment, and equally with
# half of them facing a strong one and half the opposite; those are different
# economies and they predict different counterfactuals. The object that separates
# them is the DISTRIBUTION of
#
#     rho_{rs} = Corr_l( log T_{a(l)s} , log d_{lr} )
#
# across (sector, buyer) pairs -- the same correlation the decomposition's
# co-location term is built from, kept disaggregated instead of averaged.
#
# Two conventions, both inherited rather than chosen. The correlation is taken
# ACROSS CELLS at a fixed buyer, which is the competition the model runs and is
# exactly `ca_variance_decomposition`'s convention, so the purchase-weighted mean
# of the covariances reported here reproduces that table's co-location term
# sector by sector (the gate asserts it). And it is in LOGS on both axes, which is
# what makes it the correlation the decomposition uses and a scale-free number:
# `T_hat` is identified only up to a per-sector scale, and a correlation is
# invariant to it. Test 6's kilometre covariance answers a different question --
# how far the euro moves -- and is not the same statistic.
#
# `weights="cell"` (the default) counts every competing cell once, which is the
# decomposition's convention. `weights="rho"` takes the correlation under the
# sourcing probabilities instead: that is the one appearing in the exponential-
# family identity, and the two come apart exactly where sourcing is concentrated.


def alignment_correlation(data, weights="cell"):
    """
    One row per (sector, downstream buyer): the correlation between comparative
    advantage and distance across the cells competing to supply that buyer.

    `corr` is the object; `cov`, `sd_logT` and `sd_logd` are the pieces it is built
    from, and `buyer_weight` is the buyer's share of purchases, so any aggregate
    quoted from this frame can be taken with the same weights the rest of the
    section uses. `weights` selects the measure the moments are taken under: "cell"
    (every competing cell once, the variance decomposition's convention) or "rho"
    (the sourcing probabilities, the convention of the identity in Section 5.1).
    """
    if weights not in ("cell", "rho"):
        raise ValueError(f"weights must be 'cell' or 'rho', not {weights!r}")
    geometry = sourcing_geometry(data)
    w_buyer = _buyer_weights(data)
    buyer_names = _region_labels(data).set_index("index")["ze2010_name"]
    rows = []
    for s, sector_block in geometry["by_sector"].items():
        if sector_block["cells"].size < 2:
            continue
        logT = np.log(np.maximum(sector_block["T_cell"], 1e-300))       # (n_cell,)
        log_distance = np.log(sector_block["distance"])                           # (n_cell, R_d)
        for r in range(log_distance.shape[1]):
            cell_weight = (sector_block["rho"][:, r] if weights == "rho"
                 else np.full(sector_block["cells"].size, 1.0 / sector_block["cells"].size))
            weight_total = cell_weight.sum()
            if not np.isfinite(weight_total) or weight_total <= 0:
                continue
            cell_weight = cell_weight / weight_total
            mean_logT, mean_logd = float(cell_weight @ logT), float(cell_weight @ log_distance[:, r])
            covariance = float(cell_weight @ ((logT - mean_logT) * (log_distance[:, r] - mean_logd)))
            sd_logT_r = float(np.sqrt(cell_weight @ (logT - mean_logT) ** 2))
            sd_logd_r = float(np.sqrt(cell_weight @ (log_distance[:, r] - mean_logd) ** 2))
            rows.append({
                "sector": data["sector_names"][s],
                "buyer": buyer_names.get(int(geometry["downstream"][r]), str(geometry["downstream"][r])),
                "cov": covariance,
                "corr": covariance / (sd_logT_r * sd_logd_r) if sd_logT_r > 0 and sd_logd_r > 0 else np.nan,
                "sd_logT": sd_logT_r,
                "sd_logd": sd_logd_r,
                "buyer_weight": float(w_buyer[r]),
                "n_cells": int(sector_block["cells"].size),
            })
    df = pd.DataFrame(rows)
    df.attrs["weights"] = weights
    df.attrs["industry"] = data["industry"]
    df.attrs["theta_alpha"] = geometry["theta_alpha"]
    return df


def alignment_covariance(data, weights="rho", frame=None):
    """
    The alignment in KILOMETRES, one row per (sector, buyer) — the object Section 5.1
    reads, as opposed to the dimensionless correlation `alignment_correlation` returns.

    `cov_T_d_km` is `Cov_rho(log T_a(l)s, d_lr)`, the rate at which dialling comparative
    advantage down moves that buyer's average sourcing distance in that sector
    (eq:alignment). `cov_d_d_km` is `Cov_rho(log d_lr, d_lr)`, the rate for trade costs
    (eq:distance_rate); it is a covariance between two increasing functions of the same
    variable, so it is POSITIVE by construction and the column exists to be checked, not
    to be read for its sign.

    Three conventions are inherited rather than chosen, and each of the three is a place
    the figure and the text can silently come apart. The moments are taken under the
    SOURCING probabilities `rho` by default, because that is the measure the identity
    holds under — `weights="cell"` reproduces the variance decomposition's convention
    instead and answers a different question. Distance enters in KILOMETRES on the second
    argument, because the statistic differentiates `d_rs = sum_l rho_lrs d_lr`; the log
    is on `T` alone. And the row is one (sector, buyer) pair, so a buyer-level statement
    needs `by_buyer=True` on the aggregate below, not a read of this frame's median.

    Built off `alignment_frame`, so the figure and the regression cannot drift apart.
    """
    panel = alignment_frame(data, geometry=None) if frame is None else frame
    if weights not in ("rho", "cell"):
        raise ValueError(f"weights must be 'rho' or 'cell', not {weights!r}")
    rows = []
    # grouped on `group`, not on (sector, buyer): `group` is built from the downstream
    # INDEX, so two buyers sharing a display name cannot be merged into one row.
    for _, pair_panel in panel.groupby("group", sort=False):
        sector, buyer = pair_panel["sector"].iloc[0], pair_panel["buyer"].iloc[0]
        log_T = pair_panel["log_T"].to_numpy(float)
        distance = pair_panel["distance_km"].to_numpy(float)
        cell_weight = (pair_panel["rho"].to_numpy(float) if weights == "rho"
             else np.full(len(pair_panel), 1.0 / len(pair_panel)))
        weight_total = cell_weight.sum()
        if not np.isfinite(weight_total) or weight_total <= 0 or len(pair_panel) < 2:
            continue
        cell_weight = cell_weight / weight_total
        log_distance = np.log(np.maximum(distance, 1.0))
        mean_logT, mean_distance, mean_log_distance = float(cell_weight @ log_T), float(cell_weight @ distance), float(cell_weight @ log_distance)
        rows.append({
            "sector": sector,
            "buyer": buyer,
            "cov_T_d_km": float(cell_weight @ ((log_T - mean_logT) * (distance - mean_distance))),
            "cov_d_d_km": float(cell_weight @ ((log_distance - mean_log_distance) * (distance - mean_distance))),
            "d_rs_km": mean_distance,
            "buyer_weight": float(pair_panel["weight"].sum()),
            "n_cells": int(len(pair_panel)),
        })
    table = pd.DataFrame(rows)
    table.attrs["weights"] = weights
    table.attrs["industry"] = data["industry"]
    return table


def alignment_covariance_by_buyer(data, covariance=None, weights="rho"):
    """
    The same covariance aggregated to the BUYER, which is the level Section 5.1 quotes.

    `d_r = sum_s theta_rs d_rs`, with `theta_rs` the buyer's input mix, so the buyer-level rate is the spend-weighted average
    of its sectors' rates — the same buyer-first aggregation `counterfactual_diffusion_frame`
    performs on the euros, and the reason a sector median and a buyer median need not
    agree (one sector can carry most of a buyer's spending).

    Spending comes from the realised economy (`suppliers.parquet`); absent it, the
    function raises rather than falling back on equal sector weights, which would report
    a different statistic under the same name.
    """
    covariance = alignment_covariance(data, weights=weights) if covariance is None else covariance
    suppliers = data.get("suppliers")
    if suppliers is None:
        raise FileNotFoundError(
            f"no suppliers.parquet under {data['folder']}/{data.get('step_dir','<step>')}/ "
            "— the buyer-level aggregation needs the realised sector-level spending.")
    geometry = sourcing_geometry(data)
    # The buyer label is rebuilt with `aa_display_names`, the SAME route `alignment_frame`
    # takes, so the join cannot silently miss: `alignment_correlation` names buyers through
    # `_region_labels` instead, and the two need not agree on every zone. The parquet's
    # `ze2010_downstream` carries the model's own region index, which is what
    # `geom["downstream"]` holds, so the key is exact rather than a name match.
    aa_names = aa_display_names(data)
    ze_of_name = {(aa_names[r] if r < len(aa_names) else str(r)): int(buyer_zone)
                  for r, buyer_zone in enumerate(geometry["downstream"])}
    spend = (suppliers.assign(_s=_parquet_sector_index(data, suppliers))
                .groupby(["ze2010_downstream", "_s"])["share"].sum())
    sector_index_of = {name: s for s, name in enumerate(data["sector_names"])}
    pair_spend = []
    for _, pair_row in covariance.iterrows():
        s, buyer_zone = sector_index_of.get(pair_row["sector"]), ze_of_name.get(pair_row["buyer"])
        pair_spend.append(0.0 if s is None or buyer_zone is None else float(spend.get((buyer_zone, s), 0.0)))
    covariance = covariance.assign(spend=pair_spend)
    if float(np.sum(pair_spend)) <= 0:
        raise ValueError(
            "every (sector, buyer) matched zero spending — the parquet's sector index or "
            "its `ze2010_downstream` does not line up with `sourcing_geometry`.")
    # `sup` pools `n_rep` realisations, so `spend` is `n_rep` times the per-economy figure;
    # unlike `counterfactual_diffusion_frame`, which allocates those euros and must divide,
    # this uses them only as WEIGHTS, and a common factor cancels out of a weighted mean.
    #
    # Summed rather than run through `groupby.apply`: apply over the grouping column warns
    # on pandas 2.x, and its `np.average` would have propagated a NaN covariance into the
    # numerator while `sum` silently kept that pair's spending in the denominator. Dropping
    # the unusable rows first is what makes the weights and the values the same set.
    usable = covariance.dropna(subset=["cov_T_d_km", "cov_d_d_km"])
    weighted_sums = (usable.assign(_a=usable["cov_T_d_km"] * usable["spend"], _b=usable["cov_d_d_km"] * usable["spend"])
             .groupby("buyer", sort=False)[["_a", "_b", "spend"]].sum())
    buyer_spend = weighted_sums["spend"].where(weighted_sums["spend"] > 0)
    table = pd.DataFrame({"cov_T_d_km": weighted_sums["_a"] / buyer_spend,
                        "cov_d_d_km": weighted_sums["_b"] / buyer_spend,
                        "spend": weighted_sums["spend"]})
    table.attrs["industry"] = data["industry"]
    return table


def plot_alignment_correlation(datasets, statistic="cov_km", weights=None, bins=24,
                               frames=None, colors=None, figsize=None, save_to=None,
                               annotate=True, level="pair"):
    """
    The distribution of the alignment statistic across (sector, buyer) pairs, one
    outline per industry.

    `statistic` decides WHICH alignment is drawn, and the two are different objects, not
    two scalings of one. `"cov_km"` (the default) is the sourcing-weighted
    `Cov_rho(log T, d)` in kilometres per log point — the quantity eq:alignment says the
    counterfactual integrates, and therefore the only one a paragraph quoting kilometres
    may cite. `"corr"` is the dimensionless `Corr_l(log T, log d)` taken over cells, the
    variance decomposition's convention: it ranks geometries but carries no distance, so
    reading a kilometre off it is a category error. `level="buyer"` spend-aggregates the
    covariance to the buyer, which is the level a sentence of the form "negative for
    every buyer" is about; `level="pair"` keeps the (sector, buyer) pairs behind it.

    `datasets` is a sequence of `(display name, data)` pairs, so both industries land
    on ONE panel: the comparison is the point, and two separate figures would leave
    the reader to align two axes by eye.

    Three things are fixed rather than left to matplotlib, each because the comparison
    would otherwise not be one. The bin EDGES are computed once on the POOLED values
    and reused, so the two histograms sit on the same grid. Each industry is a STEP
    OUTLINE rather than filled bars, since two filled series overlap and whichever is
    drawn last wins. And the counts are NORMALISED to densities, because the two
    industries need not contribute the same number of (sector, buyer) pairs.

    Zero is marked, and it is the line that carries the paragraph: mass to the left is
    capability sitting near demand. The share of pairs on the wrong side of it is
    annotated per industry -- a median says how strong the alignment is, this says
    whether it is common to every buyer or an average over buyers pulled in opposite
    directions.
    """
    if statistic not in ("cov_km", "corr"):
        raise ValueError(f"statistic must be 'cov_km' or 'corr', not {statistic!r}")
    if level not in ("pair", "buyer"):
        raise ValueError(f"level must be 'pair' or 'buyer', not {level!r}")
    if level == "buyer" and statistic != "cov_km":
        raise ValueError("the buyer-level aggregation is defined for 'cov_km' only")
    weight_measure = weights or ("rho" if statistic == "cov_km" else "cell")

    def _build(data_set):
        if statistic == "corr":
            return alignment_correlation(data_set, weights=weight_measure)
        covariance_table = alignment_covariance(data_set, weights=weight_measure)
        return covariance_table if level == "pair" else alignment_covariance_by_buyer(data_set, covariance=covariance_table)

    column = "corr" if statistic == "corr" else "cov_T_d_km"
    frames_by_industry = frames if frames is not None else {name: _build(d) for name, d in datasets}
    pooled = np.concatenate([f[column].dropna().to_numpy() for f in frames_by_industry.values()])
    if pooled.size == 0:
        raise ValueError(f"no {level} with a finite {column}.")
    edges = np.histogram_bin_edges(pooled, bins=bins)

    fig, ax = plt.subplots(figsize=figsize or get_figsize(width_fraction=0.8, height_fraction=0.6))
    palette = colors or {}
    default = [sim_color, toulouse_color, reference_color]
    for k, (name, frame) in enumerate(frames_by_industry.items()):
        values = frame[column].dropna().to_numpy()
        line_color = palette.get(name, default[k % len(default)])
        number_format = "+.1f" if statistic == "cov_km" else "+.2f"
        ax.hist(values, bins=edges, density=True, histtype="step", linewidth=1.6,
                color=line_color, label=f"{name} (median {np.median(values):{number_format}})")
        ax.axvline(np.median(values), color=line_color, linestyle=":", linewidth=1.0)
    ax.axvline(0.0, color="0.35", linewidth=0.9)
    unit = "buyer" if level == "buyer" else "(sector, buyer)"
    ax.set_xlabel(
        (r"$\mathrm{Cov}_{\rho}(\log \hat T_{a(l)s},\ d_{lr})$ (km per log point) "
         f"at one {unit}") if statistic == "cov_km" else
        (r"$\mathrm{Corr}_l(\log \hat T_{a(l)s},\ \log d_{lr})$ "
         f"at one {unit}"))
    ax.set_ylabel("Density")
    ax.legend(frameon=False, fontsize=fs(8), loc="upper left")
    if annotate:
        annotation = "\n".join(f"{name}: {100 * float((frame[column] > 0).mean()):.0f}{pct()} positive"
                        for name, frame in frames_by_industry.items())
        ax.text(0.98, 0.97, annotation, transform=ax.transAxes, fontsize=fs(8), va="top",
                ha="right", color="0.25")
    _despine(ax)
    fig.tight_layout()
    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        fig.savefig(save_to, bbox_inches="tight")
    return ax, frames_by_industry


# The six tests in one table, and the reading guide for it.

CA_SUMMARY_GLOSSARY = [
    ("top_area", "The attraction area with the highest estimated T in that sector: the "
                 "sector's favoured location, named by its anchor commuting zone."),
    ("n_areas", "How many attraction areas host at least one observed supplier of the "
                "sector — the number of distinct T values in the competition."),
    ("dlogT_top_vs_median", "log T of the top area minus log T of the median area. The "
                            "size of the edge, in log points, before any distance is "
                            "traded against it."),
    ("equiv_log_d", "That edge converted into distance: Delta log T / (theta*alpha), the "
                    "log distance ratio that exactly offsets it. A region of the top "
                    "area may be exp(.) times further from a buyer than a competing "
                    "region of the median area and still win."),
    ("geo_log_spread", "The log-distance spread a buyer typically faces across the "
                       "regions competing for it (p90 - p10, purchase-weighted). The "
                       "benchmark `equiv_log_d` has to clear for the edge to be decisive "
                       "in ordinary competition."),
    ("geo_log_range", "The extreme version of the same thing: max - min log distance over "
                      "all competing regions and buyers. Anchored at the own-region cell, "
                      "so it is the largest handicap the geography could ever impose."),
    ("CA_beats_typical_geography", "equiv_log_d > geo_log_spread: the edge beats the "
                                   "distances buyers ordinarily face."),
    ("CA_beats_any_geography", "equiv_log_d > geo_log_range: no location in the country "
                               "offsets the edge."),
    ("share_buyers_won", "The exact test, purchase-weighted: the share of downstream "
                         "buyers whose best supplier region actually belongs to the top "
                         "area. Unlike the columns above, nothing is averaged before the "
                         "comparison is made."),
    ("mean_win_margin", "How much further from the buyer the top area's best region could "
                        "be and still beat the best region outside it, in log distance, "
                        "purchase-weighted. Negative means it loses."),
    ("mean_within_penalty", "The log-distance gap between the best and the worst region "
                            "OF THE TOP AREA. Inside an area T cancels, so this is pure "
                            "geography: the handicap the area's own spread imposes."),
    ("area_beats_own_geography", "mean_win_margin > mean_within_penalty: the area's edge "
                                 "matters more than which of its regions a supplier is in."),
    ("sd_logT", "Standard deviation of log T across the cells competing for a buyer — the "
                "pull of comparative advantage."),
    ("sd_dist", "theta*alpha times the standard deviation of log distance across the same "
                "cells — the pull of geography, in the same units."),
    ("ratio_CA_over_distance", "sd_logT / sd_dist. Above one: comparative advantage "
                               "disperses the competition more than distance does."),
    ("share_CA", "Share of Var(log psi) attributable to Var(log T)."),
    ("share_distance", "Share attributable to (theta*alpha)^2 Var(log d)."),
    ("share_covariance", "Share attributable to -2 theta*alpha Cov(log T, log d). Expected "
                         "NEGATIVE: T is the Sinkhorn image of the observed sourcing "
                         "shares, so it compensates remoteness and the two forces offset. "
                         "See ca_covariance_benchmark for the split of that covariance."),
    ("d_km", "The average sourcing distance d_rs: how far the euro travels, in "
             "kilometres, when a buyer of this sector sources a unit of it."),
    ("d_km_no_CA", "The same distance with comparative advantage equalised across areas "
                   "— the `Equal comparative advantage` regime."),
    ("delta_km", "The difference, in kilometres. Positive means comparative advantage was "
                 "holding sourcing CLOSER than proximity alone would, so equalising it "
                 "sends the euro further away."),
    ("slope_km_per_logT", "Kilometres of sourcing distance per log point of comparative "
                          "advantage, from a pyfixest weighted regression of distance on "
                          "log T with a sector x downstream-region fixed effect and "
                          "sourcing weights. It is the line the alignment figure draws, "
                          "and the covariance behind `delta_km` divided by the weighted "
                          "variance of log T."),
]


def comparative_advantage_glossary():
    """The column-by-column reading guide for `comparative_advantage_summary`."""
    return pd.DataFrame(CA_SUMMARY_GLOSSARY, columns=["column", "meaning"]).set_index("column")


def comparative_advantage_summary(data):
    """
    One row per sector: the spread of T, what it buys in distance, whether it actually
    wins, and how the dispersion of log psi splits.

    `comparative_advantage_glossary()` documents every column.
    """
    equivalence = ca_distance_equivalence(data)
    win_margin = ca_win_margin(data)
    variance_table = ca_variance_decomposition(data)
    leverage = ca_distance_leverage(data)
    summary = equivalence.join(variance_table, how="outer").join(leverage, how="outer", rsuffix="_lev")
    if len(win_margin):
        summary = summary.join(win_margin.drop(columns=[c for c in ("top_area",) if c in win_margin.columns]),
                       how="outer")
    cols = [c for c, _ in CA_SUMMARY_GLOSSARY]
    return _by_sector_code(summary[[c for c in cols if c in summary.columns]], data)


# --- Test 3 bis: the alignment covariance in KILOMETRES ----------------------
# `alignment_frame`, `alignment_covariance` and `alignment_covariance_by_buyer` are
# the test-6 definitions above; only the figure is its own. They used to exist in
# two byte-identical copies -- one per notebook section -- because each section had
# to run on its own; a module removes that price entirely.




def plot_alignment_covariance(datasets, weights="rho", bins=24, frames=None, colors=None,
                              figsize=None, save_to=None, annotate=True, level="pair",
                              kde=True, bandwidth=None, grid=512):
    """
    The distribution of the alignment in KILOMETRES: one histogram outline and one
    kernel density per industry, on one panel.

    This is `plot_alignment_correlation`'s companion, not its replacement, and the two
    answer different questions. The correlation is scale-free, so it RANKS geometries
    and is the right object for Section 4, where the comparison is between two
    industries whose `log T` is identified only up to a per-sector scale. The covariance
    carries kilometres, so it is the only one of the two a paragraph quoting a distance
    may cite: by the alignment identity it IS the rate at which dialling comparative
    advantage down moves that buyer's average sourcing distance, and its path integral
    is the `Equal comparative advantage` counterfactual.

    Three conventions, each a place the figure and the text can silently come apart. The
    moments are taken under the SOURCING probabilities, because that is the measure the
    identity holds under; `weights="cell"` reproduces the variance decomposition's
    convention and answers a different question. Distance enters in kilometres on the
    second argument and the log is on `T` alone, because the statistic differentiates
    `d_rs = sum_l rho_lrs d_lr`. And a row is one (sector, buyer) pair unless
    `level="buyer"`, which spend-aggregates across sectors -- the level at which a
    sentence of the form "negative for every buyer" is a claim.

    Both the histogram and the density are drawn, deliberately. A Gaussian kernel is
    smooth and reads well in a paper, but it is an ESTIMATE with a bandwidth, and at a
    couple of hundred pairs it can put a shoulder where the data have three points; the
    outline underneath is the raw count the reader checks it against. `bw` overrides
    Scott's rule -- vary it before believing a mode, exactly as `bins` is varied on
    `plot_distance_histogram`. The kernel is drawn on the pooled support and is NOT
    truncated at the data range, so a little mass leaks past the extreme observations:
    that is the estimator, not a finding.

    Bin edges and the density grid are computed ONCE on the pooled values, so the two
    industries sit on one axis and the comparison is a comparison.
    """
    if level not in ("pair", "buyer"):
        raise ValueError(f"level must be 'pair' or 'buyer', not {level!r}")

    def _build(data_set):
        covariance_table = alignment_covariance(data_set, weights=weights)
        return covariance_table if level == "pair" else alignment_covariance_by_buyer(data_set, covariance=covariance_table)

    frames_by_industry = frames if frames is not None else {n: _build(d) for n, d in datasets}
    values_by_industry = {n: f["cov_T_d_km"].dropna().to_numpy(float) for n, f in frames_by_industry.items()}
    pooled = np.concatenate(list(values_by_industry.values()))
    if pooled.size == 0:
        raise ValueError(f"no {level} with a finite Cov(log T, d).")
    edges = np.histogram_bin_edges(pooled, bins=bins)
    margin = 0.15 * (pooled.max() - pooled.min() or 1.0)
    x_grid = np.linspace(pooled.min() - margin, pooled.max() + margin, int(grid))

    fig, ax = plt.subplots(figsize=figsize or get_figsize(width_fraction=0.8, height_fraction=0.6))
    palette = colors or {}
    default = [sim_color, toulouse_color, reference_color]
    for k, (name, v) in enumerate(values_by_industry.items()):
        line_color = palette.get(name, default[k % len(default)])
        ax.hist(v, bins=edges, density=True, histtype="step", linewidth=1.0,
                color=line_color, alpha=0.55)
        legend_label = f"{name} (median {np.median(v):+.0f} km)"
        if kde and v.size > 1 and np.ptp(v) > 0:
            density = gaussian_kde(v, bw_method=bandwidth)(x_grid)
            ax.plot(x_grid, density, color=line_color, linewidth=1.8, label=legend_label)
            ax.fill_between(x_grid, density, color=line_color, alpha=0.10)
        else:
            ax.plot([], [], color=line_color, linewidth=1.8, label=legend_label)
        ax.axvline(np.median(v), color=line_color, linestyle=":", linewidth=1.0)
    ax.axvline(0.0, color="0.35", linewidth=0.9)
    unit = "buyer" if level == "buyer" else "(sector, buyer)"
    ax.set_xlabel(r"$\mathrm{Cov}_{\rho}(\log \hat T_{a(l)s},\ d_{lr})$ "
                  f"(km per log point) at one {unit}")
    ax.set_ylabel("Density")
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False, fontsize=fs(8), loc="upper left")
    if annotate:
        annotation = "\n".join(f"{n}: {100 * float((v > 0).mean()):.0f}{pct()} positive"
                        for n, v in values_by_industry.items())
        ax.text(0.98, 0.97, annotation, transform=ax.transAxes, fontsize=fs(8), va="top",
                ha="right", color="0.25")
    _despine(ax)
    fig.tight_layout()
    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        fig.savefig(save_to, bbox_inches="tight")
    return ax, frames_by_industry


# ============================================================================
# The buyer's own portfolio of suppliers, and its commonality
# ============================================================================

# The BUYER'S OWN PORTFOLIO: how concentrated is one buyer's set of suppliers,
# and what comparative advantage, trade costs and granularity each do to it.
#
# Split out from the commonality exercise below, which is the same object read
# from the other side (do buyers agree on WHICH suppliers). Nothing here compares
# two buyers; every statistic is indexed by (sector, buyer) and aggregated with
# that buyer's own input mix.
#
# The section above reads the incidence vector POOLED across sectors and against the
# common vector `omega_bar`. Two things are wrong with that as the section's backbone,
# and both are structural rather than matters of taste.
#
# (1) A BUYER'S SHOCK HITS EVERY INPUT SECTOR, AND EACH SECTOR HAS ITS OWN SUPPLIER
#     GEOGRAPHY. Pooling them mixes a sector sourced from two hubs with one sourced
#     nationally, and because the input mix `theta_rs` is nearly common across buyers,
#     the pooled vectors are forced to look alike whatever happens inside a sector.
#     That is why the pooled cosine sits above 0.97 and separates nothing. Everything
#     below is therefore computed at the (sector x buyer) level and aggregated with
#     explicit weights.
#
# Granularity enters here and did not enter the sourcing distance, for a reason of
# ORDER: `d_r` is linear in `omega` and `E[omega_realised] = omega_expected`, so
# averaging was exact; `H` is quadratic, so `E[H] > H(E[omega])` strictly.


def _sector_spend(data, value_col="share"):
    """
    (buyer x sector) spending on the modelled upstream sectors, averaged over the
    realisations the parquet pools. Rows are 1-based downstream ZE indices.

    This is the object both weightings are built from: normalised down a COLUMN it is
    `pi_rs` (who buys sector s), normalised across a ROW it is `theta_rs` (what buyer r
    buys). The identity uses the first; the aggregation to a buyer uses the second.
    """
    suppliers = data.get("suppliers")
    if suppliers is None:
        raise FileNotFoundError(
            f"no suppliers.parquet under {data['folder']}/{data.get('step_dir','<step>')}/ "
            "— the sector split of a buyer's euro is read off the realised economy.")
    spend = (suppliers.assign(_s=_parquet_sector_index(data, suppliers))
             .groupby(["ze2010_downstream", "_s"])[value_col].sum())
    if "replication" in suppliers.columns:
        spend = spend / float(suppliers["replication"].nunique())
    spend_table = spend.unstack("_s").reindex(columns=np.arange(data["S"])).fillna(0.0)
    spend_table.columns.name = "sector"
    return spend_table




def _reweight_buyer_spend(spend, buyer_weight):
    """
    Give each sector's positive-spend buyers weights proportional to `w`, column total kept.

    Two conventions, both deliberate. Only buyers with STRICTLY POSITIVE spending on a
    sector get weight: a buyer that does not buy from `s` has no portfolio there, and
    handing it a share of the average would invent one. And each column KEEPS ITS TOTAL,
    so the `spend` column that weights SECTORS against each other is untouched and only
    the WITHIN-sector weighting moves; without that, two decompositions would differ for
    two reasons at once and neither could be read.
    """
    spend_frame = spend.copy().astype(float)
    positive_weights = (spend_frame > 0).mul(np.asarray(buyer_weight, dtype=float).ravel(), axis=0)
    column_weight = positive_weights.sum(axis=0)
    if (column_weight <= 0).any():
        empty_sectors = [str(c) for c in spend_frame.columns[column_weight <= 0]]
        raise ValueError(f"sectors {empty_sectors[:5]} have no buyer carrying positive weight.")
    return positive_weights.div(column_weight, axis=1).mul(spend_frame.sum(axis=0), axis=1)


def equal_buyer_spend(spend):
    """
    The same (buyer x sector) frame with every buyer of a sector weighted EQUALLY.

    The identity `Hbar_s = H(omega_bar_s) + M_s` holds for a weighted mean under ANY
    weights, so flattening `pi_rs` is legitimate -- it changes the QUESTION, not the
    algebra. `rho_floor` is where the change is signed in advance: it is the
    granular-weighted Herfindahl of the buyer weights, so under equal weights it lands on
    `1/n_buyers` EXACTLY, which is the check that the reweighting reached `pi` at all.
    """
    return _reweight_buyer_spend(spend, np.ones(len(spend.index)))


def size_buyer_spend(spend, data):
    """
    The same frame with every buyer weighted by its SIZE -- the empirical `pi_r` target,
    the buyer region's share of downstream purchases.

    This is the weighting the customer size distribution actually lives in, and
    `_sector_spend` is NOT it. The parquet's `share` is `exp_val`, a share of the
    DOWNSTREAM FIRM'S OWN UNIT COST, so summing it over a buyer's suppliers returns that
    buyer's intermediate cost share -- a number close to the same for every buyer,
    whatever its size. The default weights are therefore already nearly uniform (measured
    `buyer_hhi` barely above `1/n_buyers`), which is why flattening them moves almost
    nothing: the size distribution was never in the decomposition to be taken out. It
    enters here.

    `emp_pi_r` is ordered by `flatnonzero(N_downstream != 0)`, exactly as
    `_downstream_ze_index` builds the 1-based downstream ZE index, so the two align by
    construction; a buyer of `spend` absent from that index is an error rather than a
    silent zero.
    """
    buyer_size = pd.Series(np.asarray(data["emp_pi_r"], dtype=float).ravel(),
                  index=_downstream_ze_index(data))
    missing = [int(b) for b in spend.index if int(b) not in buyer_size.index]
    if missing:
        raise KeyError(f"buyers {missing[:5]} are not downstream regions of `emp_pi_r`.")
    return _reweight_buyer_spend(spend, buyer_size.reindex(spend.index).to_numpy())
def _conc_identity(W, pi_r):
    """
    The identity on one sector's incidence matrix.

    `W` is (buyer x cell) with rows summing to one, `pi` a buyer weight vector summing
    to one. Returns `(Hbar, H_common, M, per_buyer_H, per_buyer_m)`, with
    `M = Hbar - H(omega_bar) = sum_l Var_pi(omega_lrs)` — an identity, not an
    approximation, which is what makes the two halves additive in euros.
    """
    W = np.asarray(W, dtype=float)
    pi_r = np.asarray(pi_r, dtype=float)
    pi_total = pi_r.sum()
    pi_r = pi_r / pi_total if pi_total > 0 else np.full(pi_r.size, np.nan)
    omega_bar = pi_r @ W
    h_r = (W ** 2).sum(axis=1)
    m_r = ((W - omega_bar) ** 2).sum(axis=1)
    return float(pi_r @ h_r), float((omega_bar ** 2).sum()), float(pi_r @ m_r), h_r, m_r, omega_bar


def structural_networks(data, alpha=None, equalise_T=False, buyers=None, kappa=None):
    """
    The expected incidence `omega^E_{lrs}` of every modelled sector under one regime,
    as a dict `sector -> {"W": (buyer x cell), "cells": ..., "buyers": ...}`.

    `W` is `rho` transposed: `rho` is indexed (cell, buyer) and every column already
    sums to one over the cells of that sector, so the rows of `W` are incidence vectors
    with no renormalisation. `buyers` restricts and orders the columns; it defaults to
    the downstream regions the geometry carries.
    """
    # `kappa` is passed only when set, so a caller's own `sourcing_geometry` without it
    # still serves the two original regimes
    geometry = sourcing_geometry(data, alpha=alpha, equalise_T=equalise_T,
                             **({} if kappa is None else {"kappa": kappa}))
    downstream = np.asarray(geometry["downstream"]).astype(int)
    if buyers is None:
        buyers = downstream
    buyers = np.asarray(buyers).astype(int)
    column_of_buyer = {int(z): j for j, z in enumerate(downstream)}
    missing = [int(b) for b in buyers if int(b) not in column_of_buyer]
    if missing:
        raise KeyError(f"buyers {missing[:5]} are not downstream regions of the geometry.")
    columns = np.array([column_of_buyer[int(b)] for b in buyers])
    networks = {}
    for s, sector_block in geometry["by_sector"].items():
        networks[s] = {"W": sector_block["rho"][:, columns].T, "cells": sector_block["cells"], "buyers": buyers,
                  "distance": sector_block["distance"][:, columns], "T_cell": sector_block["T_cell"]}
    return {"by_sector": networks, "buyers": buyers, "alpha": geometry["alpha"],
            "theta": geometry["theta"]}


def sector_concentration(data, regimes=CF_REGIMES, value_col="share", benchmark=True,
                         spend=None, verbose=True):
    """
    The identity, one row per (regime, sector): concentration, how much of it is common,
    and both in effective numbers.

    Columns
    -------
    h_bar          spend-weighted mean of `H_rs` over buyers — how concentrated a
                   shock to the TYPICAL buyer's euro is in this sector.
    h_common       `H(omega_bar_s)` — how concentrated a shock to the whole industry is.
    m              `h_bar - h_common = sum_l Var_pi(omega_lrs)`, the buyer-specific part.
    C              `1 - m/h_bar`, the share of a buyer's concentration that aggregation
                   cannot diversify away. `C = 1` exactly when nothing is
                   buyer-specific, which under `alpha = 0` is a THEOREM: distance is the
                   only buyer-specific term in `rho`, so equal distances make every
                   buyer's incidence identical. That row is the section's internal
                   control and it should read 1.000, not 0.997.
    n_eff, n_eff_common      the reciprocals — effective numbers of destination zones.
    n_eff_ratio    `n_eff` divided by the same sector's `n_eff` under the uniform
                   allocation (`alpha = 0` AND `T` equalised), which purges the cell
                   count and the shape of the country.
    n_hat_s        the sector's variety count, for the granularity paragraph.

    AVERAGE H, THEN INVERT. Every effective number here is the reciprocal of a
    spend-weighted mean of `H`, never the mean of per-buyer reciprocals: `1/H` is convex,
    so the two differ and only the first respects the identity.
    """
    spend_frame = _sector_spend(data, value_col) if spend is None else spend
    buyers = np.asarray(spend_frame.index).astype(int)
    wanted_regimes = dict(regimes)
    if benchmark and UNIFORM_REGIME not in wanted_regimes:
        wanted_regimes[UNIFORM_REGIME] = dict(alpha=0.0, equalise_T=True)

    n_hat = data.get("post_hoc_N_hat")
    if n_hat is None:
        n_hat = _n_hat_from_diagnostics(data)
    n_hat = None if n_hat is None else np.asarray(n_hat, dtype=float).ravel()

    rows, per_buyer = [], []
    for regime, kw in wanted_regimes.items():
        networks = structural_networks(data, buyers=buyers, **kw)
        for s, sector_block in networks["by_sector"].items():
            pi_s = spend_frame.iloc[:, s].to_numpy(dtype=float)
            if pi_s.sum() <= 0:
                continue
            h_bar, h_common, m, h_r, m_r, omega_bar = _conc_identity(sector_block["W"], pi_s)
            rows.append({"regime": regime, "sector": s,
                         "sector_name": str(data["sector_names"][s]),
                         "n_cells": sector_block["W"].shape[1],
                         "spend": float(pi_s.sum()),
                         "h_bar": h_bar, "h_common": h_common, "m": m,
                         "C": 1.0 - m / h_bar if h_bar > 0 else np.nan,
                         "n_eff": 1.0 / h_bar if h_bar > 0 else np.nan,
                         "n_eff_common": 1.0 / h_common if h_common > 0 else np.nan,
                         "n_hat_s": np.nan if n_hat is None or s >= n_hat.size
                                    else float(n_hat[s])})
            per_buyer.append(pd.DataFrame({
                "regime": regime, "sector": s, "ze2010_downstream": buyers,
                "h": h_r, "m": m_r, "spend": pi_s}))
    table = pd.DataFrame(rows).set_index(["regime", "sector"]).sort_index()

    if benchmark and UNIFORM_REGIME in wanted_regimes:
        uniform_n_eff = table.loc[UNIFORM_REGIME, "n_eff"]
        table["n_eff_ratio"] = table["n_eff"] / \
            uniform_n_eff.reindex(table.index.get_level_values("sector")).to_numpy()
    else:
        table["n_eff_ratio"] = np.nan
    table.attrs["per_buyer"] = pd.concat(per_buyer, ignore_index=True)
    table.attrs["spend"] = spend_frame
    if verbose:
        print("  concentration and commonality, sector by sector "
              "(spend-weighted over buyers)")
        print(table.round(3).to_string())
    return table


def buyer_concentration(table, data=None, verbose=False):
    """
    The same identity read buyer by buyer: `H_r = sum_s theta_rs H_rs` and
    `C_r = 1 - sum_s theta_rs m_rs / H_r`, with `theta_rs` the buyer's own input mix.

    `C_r` is NOT bounded below — a buyer whose incidence is far from the common one in
    a sector where the common vector is itself concentrated can carry `m_rs > H_rs` —
    so it is a RANKING, not a share, and the appendix figure presents it as one.
    """
    per_buyer = table.attrs["per_buyer"].copy()
    per_buyer["_h"] = per_buyer["h"] * per_buyer["spend"]
    per_buyer["_m"] = per_buyer["m"] * per_buyer["spend"]
    by_buyer = per_buyer.groupby(["regime", "ze2010_downstream"], sort=False)[["_h", "_m", "spend"]].sum()
    buyer_table = pd.DataFrame({"h": by_buyer["_h"] / by_buyer["spend"], "m": by_buyer["_m"] / by_buyer["spend"],
                        "spend": by_buyer["spend"]})
    buyer_table["C"] = 1.0 - buyer_table["m"] / buyer_table["h"]
    buyer_table["n_eff"] = 1.0 / buyer_table["h"]
    if data is not None:
        names = _region_labels(data).set_index("index")["ze2010_name"]
        buyer_table["region"] = names.reindex(
            buyer_table.index.get_level_values("ze2010_downstream")).to_numpy()
    if verbose:
        print(buyer_table.round(3).to_string())
    return buyer_table


def concentration_derivatives(data, value_col="share", spend=None, check=True,
                              verbose=True):
    """
    Which force builds the concentration, as the derivative of `H` along the two dials —
    the exact analogue of the sourcing-distance identity.

    With `rho_l ∝ T_l^t d_l^{-alpha*theta}`,

        dH_rs/dt            =  2 Cov_rho(log T,  rho)
        dH_rs/d(alpha*theta) = -2 Cov_rho(log d,  rho)

    both taken under the sourcing probabilities themselves. Same weights and the same
    two forces as the distance identity; the only change is what `T` and `d` covary with
    — the SHARES rather than the distance.

    The reading is immediate. Comparative advantage raises concentration whenever the
    productive cells are the ones ALREADY holding large shares, which fails only where
    distance reverses the productivity ranking. Trade costs raise it only if a buyer's
    large suppliers are the nearby ones, and that sign is genuinely ambiguous — which is
    why `alpha = 0` can move concentration hardly at all while moving distance a great
    deal.

    The derivative is exact on `rho` and only approximate on `E[omega]`; `check=True`
    verifies it against a central difference of the closed form and prints the residual
    relative to the derivative, so the claim is measured rather than asserted.
    """
    spend_frame = _sector_spend(data, value_col) if spend is None else spend
    buyers = np.asarray(spend_frame.index).astype(int)
    geometry = sourcing_geometry(data)
    downstream = np.asarray(geometry["downstream"]).astype(int)
    buyer_columns = np.array([int(np.flatnonzero(downstream == b)[0]) for b in buyers])
    theta, alpha_vector = geometry["theta"], geometry["alpha"]
    # The trade-cost dial: log tau is homogeneous of degree one in the alpha VECTOR, so
    # scaling every alpha_k by c gives psi = T exp(-(theta*c*a_bar) L) with
    # L = log tau(alpha_hat)/a_bar -- and theta*a_bar is the dial. Under the power law
    # L = log d and the dial is theta*alpha, exactly as before.
    a_bar = float(np.mean(alpha_vector))
    if a_bar <= 0:
        raise ValueError("concentration_derivatives is taken at a geometry with trade costs.")

    rows = []
    for s, sector_block in geometry["by_sector"].items():
        rho = sector_block["rho"][:, buyer_columns]                       # (cell, buyer)
        if rho.size == 0 or rho.sum() == 0:
            continue
        logT = np.log(np.maximum(sector_block["T_cell"], 1e-300))
        log_distance_equiv = sector_block["log_tau"][:, buyer_columns] / a_bar            # = log d under the power law
        H_r = (rho ** 2).sum(axis=0)
        dH_dlogT_r = 2.0 * ((rho * logT[:, None] * rho).sum(axis=0)
                    - (rho * logT[:, None]).sum(axis=0) * H_r)
        dH_dalphatheta_r = -2.0 * ((rho * log_distance_equiv * rho).sum(axis=0)
                     - (rho * log_distance_equiv).sum(axis=0) * H_r)
        pi_s = spend_frame.iloc[:, s].to_numpy(dtype=float)
        if pi_s.sum() <= 0:
            continue
        pi_weights = pi_s / pi_s.sum()
        rows.append({"sector": s, "sector_name": str(data["sector_names"][s]),
                     "h_bar": float(pi_weights @ H_r),
                     "dH_dlogT": float(pi_weights @ dH_dlogT_r), "dH_dalphatheta": float(pi_weights @ dH_dalphatheta_r),
                     "spend": float(pi_s.sum())})
    table = pd.DataFrame(rows).set_index("sector").sort_index()

    if check:
        # central difference on alpha alone, the one dial the closed form can be
        # re-evaluated at; the T dial has no free exponent in `sourcing_geometry`.
        eps = 1e-4
        def _h(alpha_val):
            geometry_shifted = sourcing_geometry(data, alpha=alpha_val)
            H_by_sector = {}
            for s, sector_block in geometry_shifted["by_sector"].items():
                rho_buyers = sector_block["rho"][:, buyer_columns]
                pi_s = spend_frame.iloc[:, s].to_numpy(dtype=float)
                if pi_s.sum() <= 0:
                    continue
                H_by_sector[s] = float((pi_s / pi_s.sum()) @ (rho_buyers ** 2).sum(axis=0))
            return pd.Series(H_by_sector)
        step = eps / (theta * a_bar)
        finite_difference = (_h(alpha_vector * (1 + step)) - _h(alpha_vector * (1 - step))) / (2 * eps)
        table["fd_dH_dalphatheta"] = finite_difference.reindex(table.index)
        max_relative_gap = np.nanmax(np.abs(table["fd_dH_dalphatheta"] - table["dH_dalphatheta"])
                        / np.maximum(np.abs(table["dH_dalphatheta"]), 1e-12))
        table.attrs["identity_residual"] = float(max_relative_gap)
        if verbose:
            print(f"    identity check: max relative gap between the closed-form "
                  f"derivative and a central difference = {max_relative_gap:.2e}")
    if verbose:
        pi_weights = table["spend"] / table["spend"].sum()
        print("  which force builds concentration (derivatives of H under rho)")
        print(table.round(4).to_string())
        print(f"    spend-weighted: dH/dlogT = {float(pi_weights @ table['dH_dlogT']):+.4f}, "
              f"dH/d(alpha*theta) = {float(pi_weights @ table['dH_dalphatheta']):+.4f}")
    return table


# --- P6: the granular decomposition, from varieties rather than from realisations ----
#
# Everything above treats one realisation as a black box and averages. The plan's P6
# opens it: with finite varieties a buyer's incidence is
#
#     omega_lrs = sum_rho v_rho,rs * 1{ winner(rho, r) = l } ,
#
# so `H_rs = sum_{rho,rho'} v_rho v_rho' 1{same winner}` splits into the diagonal
# (same variety, winner trivially shared) and the off-diagonal (different varieties,
# which land together only through the structural geography). Under Eaton-Kortum the
# winner's IDENTITY is independent of its PRICE -- the winning price distribution is
# the same whatever region won -- so the expenditure weights `v` are independent of
# the winner indicators and the expectation factorises EXACTLY:
#
#     E[H_rs] = V_rs + (1 - V_rs) H^gamma_rs,      V_rs = E[ sum_rho v_rho,rs^2 ].
#
# `V` is a Herfindahl over VARIETIES, not over regions, and its inverse is the
# EFFECTIVE NUMBER OF VARIETIES. The familiar multinomial formula is the special case
# `V = 1/N_s`, which holds only when varieties carry equal expenditure; under CES the
# cheaper winners take more, so by Cauchy-Schwarz `V >= 1/N_s` ALWAYS and the naive
# reading of the calibrated `N_s` is optimistic.
#
# **The parametric form, and why it is not used as the estimate here.** With Fréchet
# costs `p^theta` is exponential, so `y = p^(1-nu_s)` is Fréchet with tail index
# `kappa_s = theta/(nu_s - 1)`, and `v = y/sum y` is a normalised sum of i.i.d.
# heavy-tailed weights -- the Gabaix granularity object. For `kappa > 2`,
# `V ~= Xi(kappa)/N_s` with `Xi(kappa) = Gamma(1-2/kappa)/Gamma(1-1/kappa)^2 >= 1`,
# the squared coefficient of variation of the weights. At THIS calibration
# `theta = 1` and `nu_s = 1.5` give `kappa_s = 2` exactly, where `Xi` DIVERGES: the
# variance of the variety weights is not finite and the expansion has nothing to say.
# That is a finding rather than an obstacle -- it is the statement that a single
# variety can carry a non-vanishing share of a buyer's sector spending -- and it is
# why `V` is measured from the draws instead of being read off a formula. The
# closed form is reported beside it as the reference it fails to be.
#
def _xi_of_kappa(kappa):
    """
    `Xi(kappa) = Gamma(1-2/kappa)/Gamma(1-1/kappa)^2`, the squared coefficient of
    variation of Fréchet(kappa) variety weights, and `+inf` at `kappa <= 2` where the
    second moment does not exist. The effective number of varieties is `N_s/Xi`.
    """
    k = float(kappa)
    if not np.isfinite(k) or k <= 2.0:
        return np.inf
    return math.gamma(1 - 2 / k) / math.gamma(1 - 1 / k) ** 2


def variety_tail_index(data, nu_s=None, verbose=True):
    """
    The granularity parameter of the variety weights, sector by sector.

    `kappa_s = theta/(nu_s - 1)` is what governs how unequal the expenditure shares of
    a sector's varieties are — NOT `N_s`, which only counts them. `nu_s` is CALIBRATED
    (`load_parameters.jl` sets 1.5 for every sector) and is not written to any
    artefact, so it is taken from `NU_S_DEFAULT` unless passed.
    """
    theta = model_theta(data)
    S = data["S"]
    nu_s_values = np.full(S, NU_S_DEFAULT if nu_s is None else nu_s, dtype=float) \
        if np.isscalar(nu_s) or nu_s is None else np.asarray(nu_s, dtype=float)
    n_hat = data.get("post_hoc_N_hat")
    if n_hat is None:
        n_hat = _n_hat_from_diagnostics(data)
    n_hat = np.full(S, np.nan) if n_hat is None else np.asarray(n_hat, float).ravel()
    kappa = theta / (nu_s_values - 1.0)
    xi = np.array([_xi_of_kappa(k) for k in kappa])
    table = pd.DataFrame({"sector_name": [str(c) for c in data["sector_names"]],
                        "theta": theta, "nu_s": nu_s_values, "kappa": kappa, "Xi": xi,
                        "n_hat_s": n_hat[:S], "n_eff_varieties_pred": n_hat[:S] / xi},
                       index=pd.RangeIndex(S, name="sector"))
    if verbose:
        print(f"  the variety-weight tail: kappa = theta/(nu_s - 1), theta = {theta:g}")
        print(table.round(3).to_string())
        if (kappa <= 2).any():
            print("    kappa <= 2 in at least one sector: the variance of the variety "
                  "weights is NOT finite, Xi diverges, and the closed form has no "
                  "content. V is measured from the draws instead.")
    return table


def variety_panel(data, value_col="share"):
    """
    The realised economy re-indexed by VARIETY: for each sector, who won each variety
    for each buyer, and what share of that buyer's sector spending it carries.

    Returns `sector -> {"winner": (draw x buyer) 1-based zone, "v": (draw x buyer)
    expenditure shares summing to one down each column-block of a replication,
    "buyers": ..., "replication": ...}`, where a "draw" is one (replication, variety).

    This is the object the naive realisation-by-realisation average cannot see: two
    buyers landing on the same zone through the SAME variety and through DIFFERENT
    varieties are indistinguishable in `omega`, and they are the two halves of the
    decomposition.
    """
    suppliers = data.get("suppliers")
    if suppliers is None or "replication" not in suppliers.columns or "variety" not in suppliers.columns:
        raise ValueError("suppliers.parquet carries no `replication`/`variety` column "
                         "— the variety decomposition needs the finite-variety economy "
                         "written by `write_post_hoc` (see `supplier_count_check`).")
    suppliers = suppliers.assign(_s=_parquet_sector_index(data, suppliers))
    panel = {}
    for s, sector_rows in suppliers.groupby("_s", sort=True):
        buyer_spend = sector_rows.groupby(["replication", "ze2010_downstream"])[value_col].transform("sum")
        sector_rows = sector_rows.assign(_v=sector_rows[value_col] / buyer_spend.where(buyer_spend > 0))
        winner = sector_rows.pivot_table(index=["replication", "variety"],
                            columns="ze2010_downstream", values="ze2010", aggfunc="first")
        variety_share = sector_rows.pivot_table(index=["replication", "variety"],
                            columns="ze2010_downstream", values="_v", aggfunc="sum")
        count_per_cell = sector_rows.pivot_table(index=["replication", "variety"],
                            columns="ze2010_downstream", values="_v", aggfunc="size")
        if np.nanmax(count_per_cell.to_numpy(dtype=float)) > 1:
            raise ValueError(f"sector {s}: a (replication, variety, buyer) triple has "
                             "more than one winning zone — one variety is won by exactly "
                             "one cell per buyer, so the parquet is not what it claims.")
        panel[int(s)] = {"winner": winner.to_numpy(dtype=float), "v": variety_share.to_numpy(dtype=float),
                       "buyers": np.asarray(winner.columns).astype(int),
                       "replication": np.asarray([i[0] for i in winner.index]).astype(int)}
    return panel


def variety_concentration(data, value_col="share", panel=None, spend=None,
                          verbose=True):
    """
    `V_rs = E[sum_rho v_rho,rs^2]`, the Herfindahl over VARIETIES, and its reciprocal,
    the effective number of varieties.

    Reported three ways because each answers a different question. `V` itself enters
    the decomposition. `V * N_s` is the quantity that decides whether the naive
    multinomial reading is safe: it is one under equal expenditure shares and above
    one under CES, and Cauchy-Schwarz makes it at least one always. And the CROSS-BUYER
    dispersion of `V` is a prediction of the model rather than a diagnostic: the
    Fréchet scale `Phi_rs` cancels in the normalisation, so `V` should be
    buyer-INDEPENDENT — one scalar per sector — and a visible dispersion would mean
    the independence argument does not hold in the simulated economy.
    """
    panel_used = variety_panel(data, value_col) if panel is None else panel
    spend_frame = _sector_spend(data, value_col) if spend is None else spend
    n_hat = data.get("post_hoc_N_hat")
    if n_hat is None:
        n_hat = _n_hat_from_diagnostics(data)
    n_hat = None if n_hat is None else np.asarray(n_hat, float).ravel()

    rows = []
    for s, sector_block in panel_used.items():
        variety_share, replication = sector_block["v"], sector_block["replication"]
        # sum over the varieties of ONE replication, then average over replications
        per_replication = pd.DataFrame(np.nan_to_num(variety_share) ** 2).groupby(replication).sum().to_numpy()
        v_buyer = per_replication.mean(axis=0)                       # (buyer,)
        pi_s = spend_frame.iloc[:, s].reindex(sector_block["buyers"]).fillna(0.0).to_numpy(dtype=float)
        pi_weights = pi_s / pi_s.sum() if pi_s.sum() > 0 else np.full(pi_s.size, np.nan)
        N_s = np.nan if n_hat is None or s >= n_hat.size else float(n_hat[s])
        rows.append({"sector": s, "sector_name": str(data["sector_names"][s]),
                     "n_hat_s": N_s, "V": float(pi_weights @ v_buyer),
                     "V_times_N": float(pi_weights @ v_buyer) * N_s,
                     "n_eff_varieties": 1.0 / float(pi_weights @ v_buyer),
                     "V_buyer_sd": float(np.std(v_buyer)),
                     "V_buyer_range": float(v_buyer.max() - v_buyer.min())})
    table = pd.DataFrame(rows).set_index("sector").sort_index()
    if verbose:
        print("  the effective number of VARIETIES (V = E[sum_rho v_rho^2])")
        print(table.round(4).to_string())
        print(f"    V x N_s: median {table['V_times_N'].median():.2f} — one means equal "
              "expenditure across varieties, above one means CES concentrates it and "
              "the naive 1/N_s understates granular concentration.")
        print(f"    cross-buyer dispersion of V: max range {table['V_buyer_range'].max():.4f} "
              "(the model predicts zero — the Fréchet scale cancels).")
    return table


def simulate_granular_regime(data, alpha=None, equalise_T=False, kappa=None, n_rep=None,
                             seed=20260912, value_col="share", spend=None, n_hat=None,
                             theta_plus=None, u_draws=None):
    """
    The finite-variety economy of a COUNTERFACTUAL regime — now a thin delegation to
    `simulate_economy`, the single forward map from the extended parameter set.

    It is kept as a name rather than removed because four call sites read it (the buyer
    portfolio, the commonality half, the granular table and the local share), and one
    delegation switches all four at once. What it adds is the buyer set: these consumers
    are indexed by a SPEND frame, so the panel is cut to that frame's buyers, while the
    value block stays over every downstream region.

    Two things worth knowing about what changed underneath. The Frechet branch is now
    Julia's (`-log(1-u)`, not `-log u`), so at a given seed the realised economies are
    DIFFERENT — statistically equivalent, not identical, and any gate pinned to one
    realisation moves. And the returned object carries `exp_val` and the value block
    beside `winner` and `v`, so `D_r`, the input mix and the price indices are available
    per regime; nothing here reads them yet, which is a choice rather than a limit.

    The BASELINE is now simulated here too. `suppliers.parquet` has left the reporting
    path entirely: it is the input to `check_against_julia`, which compares this port
    against `solve_network` to the bit given Julia's own draws, and nothing else. The
    estimated economy and the counterfactuals therefore share a seed, so the DIFFERENCE
    between two regimes carries common random numbers and is less noisy than two
    independent draws would make it -- while the baseline is no longer Julia's own
    realisation, and any number pinned to that realisation moves.
    """
    spend_frame = _sector_spend(data, value_col) if spend is None else spend
    return simulate_economy(data, theta_plus, alpha=alpha, equalise_T=equalise_T, kappa=kappa,
                            n_rep=n_rep,
                            seed=seed, u_draws=u_draws, n_hat=n_hat,
                            buyers=np.asarray(spend_frame.index).astype(int), verbose=False)

# --- P7: the BUYER-level object, and what each force does to it ----------------------
#
# The paragraph now reads H_rs and H_r BEFORE the industry aggregate H_s, so the
# buyer-level expectation needs to be reported on its own, regime by regime, in the
# form Figure `counterfactual_distance_region` uses for the sourcing distance.
#
# The object is the across-VARIETY split of eq. (variety_split) taken at the buyer
# level and aggregated over sectors with the buyer's own input mix:
#
#     E_Omega[ H_r(Omega) ] = sum_s theta_rs [ H(gamma_rs) + V_rs (1 - H(gamma_rs)) ] .
#
# Two readings come out of it and neither is available from the industry aggregate.
# The STRUCTURAL term sum_s theta_rs H(gamma_rs) is the N_s -> infinity limit — the
# concentration a regular Ricardian allocation would deliver — so the gap between the
# two curves IS granularity, buyer by buyer, and it is drawn as its own series rather
# than argued. And because the granular addition is proportional to 1 - H(gamma_rs),
# a buyer whose expected network is already concentrated has little left for chance to
# concentrate: the two forces are SUBSTITUTES at the buyer level, which is visible as
# the granular series shrinking exactly where the structural one is small.
#
# `V_rs` is kept disaggregated across buyers here, where `variety_concentration`
# averages it away. The model predicts it to be buyer-independent (the Frechet scale
# cancels in an expenditure share), so its cross-buyer dispersion is a restriction
# being tested rather than a nuisance, and the aggregation must not assume it.

INFINITE_REGIME = "Infinite varieties"


def _V_by_buyer(panel, buyers):
    """
    `V_rs = E[sum_j v_jrs^2]` kept DISAGGREGATED across buyers, as a (sector x buyer)
    frame whose columns are the buyers' zone codes.

    `variety_concentration` collapses this to one number per sector with the spend
    weights; the buyer-level decomposition cannot, since each buyer's `H_rs` carries
    its own `V_rs`. The sum over varieties is taken WITHIN a replication and averaged
    across replications, which is what `E[sum_j v^2]` means — pooling the replications
    first would divide every weight by the number of draws.
    """
    rows = {}
    for s, sector_block in panel.items():
        variety_share, replication = sector_block["v"], sector_block["replication"]
        per_replication = pd.DataFrame(np.nan_to_num(variety_share) ** 2).groupby(replication).sum().to_numpy()
        rows[int(s)] = pd.Series(per_replication.mean(axis=0),
                                 index=np.asarray(sector_block["buyers"]).astype(int))
    V_table = pd.DataFrame(rows).T
    V_table.index.name = "sector"
    return V_table.reindex(columns=np.asarray(buyers).astype(int))


def buyer_granular_concentration(data, regimes=None, value_col="share", spend=None,
                                 panel=None, include_infinite=True, verbose=True):
    """
    `E_Omega[H_r(Omega)]` buyer by buyer under each regime, with its structural and
    granular halves and both as effective numbers of supplier zones.

    One row per (regime, buyer). EVERY regime, the estimated economy included, comes from
    `simulate_granular_regime` -- one forward map from `theta+`, so a counterfactual is
    not a different kind of object from the baseline. (It used to read `Both forces` off
    `suppliers.parquet` and simulate the rest.) A regime whose panel cannot be built falls
    back to the structural columns with `h` left missing rather than being silently
    reported as if it were realised. A `panel=` passed for the baseline still overrides,
    which is how a caller hands in Julia's own realisation.

    Columns
    -------
    h_struct   `sum_s theta_rs H(gamma_rs)`, the N_s -> infinity limit.
    V          the buyer's spend-weighted effective-variety Herfindahl.
    h          `sum_s theta_rs [H + V(1-H)]`, the realised expectation.
    gran       `h - h_struct >= 0`, granularity in the units of `H` itself.
    slack      `1 - h_struct`, the dispersion the expected network leaves for chance
               to concentrate; `gran_VS = V * slack` and `gran_resid` its gap to
               `gran`, which IS the cross-sector covariance `Cov_theta(V_s, slack_rs)`.
               The exact (sector, buyer) form is in `.attrs["by_sector"]`.
    n_eff, n_eff_struct   the reciprocals.

    AVERAGE H, THEN INVERT, as everywhere else in this section: `1/H` is convex, so a
    mean of per-sector effective numbers is not the effective number of the mean.
    """
    spend_frame = _sector_spend(data, value_col) if spend is None else spend
    buyers = np.asarray(spend_frame.index).astype(int)
    wanted_regimes = dict(CF_REGIMES if regimes is None else regimes)
    wanted_regimes.setdefault(UNIFORM_REGIME, dict(alpha=0.0, equalise_T=True))
    names = _region_labels(data).set_index("index")["ze2010_name"]

    blocks = []
    for regime, kw in wanted_regimes.items():
        networks = structural_networks(data, buyers=buyers, **kw)
        try:
            # Every regime comes from the SAME forward map, the estimated economy
            # included: a counterfactual is then not a different KIND of object from the
            # baseline. `panel=` still overrides the baseline, which is how a caller
            # hands in a panel read off Julia's parquet.
            if panel is not None and not kw:
                variety_panel_used = panel
            else:
                variety_panel_used = simulate_granular_regime(data, value_col=value_col, spend=spend_frame, **kw)
            V = _V_by_buyer(variety_panel_used, buyers)
        except (ValueError, KeyError, FileNotFoundError) as e:
            print(f"  [buyer granular] {regime}: no variety panel "
                  f"({type(e).__name__}: {e}) — structural columns only.")
            V = None
        for s, sector_block in networks["by_sector"].items():
            pi_s = spend_frame.iloc[:, s].to_numpy(dtype=float)
            if pi_s.sum() <= 0:
                continue
            H_gamma = (np.asarray(sector_block["W"], dtype=float) ** 2).sum(axis=1)
            v_rs = (V.loc[s].to_numpy(dtype=float)
                    if V is not None and s in V.index else np.full(H_gamma.size, np.nan))
            blocks.append(pd.DataFrame({
                "regime": regime, "sector": s, "ze2010_downstream": buyers, "spend": pi_s,
                "h_struct": H_gamma, "V": v_rs, "h": H_gamma + v_rs * (1.0 - H_gamma)}))
    by_cell_long = pd.concat(blocks, ignore_index=True)

    # a sector whose V is missing must make the whole buyer row missing, not be
    # dropped: pandas sums NaN as zero, which would quietly reweight the input mix.
    by_cell_long["_bad"] = by_cell_long["h"].isna().to_numpy() * by_cell_long["spend"].to_numpy()
    for column in ("h", "h_struct", "V"):
        by_cell_long["_" + column] = by_cell_long[column].to_numpy() * by_cell_long["spend"].to_numpy()
    by_buyer = by_cell_long.groupby(["regime", "ze2010_downstream"], sort=False)[
        ["_h", "_h_struct", "_V", "spend", "_bad"]].sum()
    table = pd.DataFrame({"h": by_buyer["_h"] / by_buyer["spend"], "h_struct": by_buyer["_h_struct"] / by_buyer["spend"],
                        "V": by_buyer["_V"] / by_buyer["spend"], "spend": by_buyer["spend"]})
    table.loc[by_buyer["_bad"].to_numpy() > 0, ["h", "V"]] = np.nan

    if include_infinite and "Both forces" in wanted_regimes:
        infinite_rows = table.loc["Both forces"].copy()
        infinite_rows["h"] = infinite_rows["h_struct"]
        infinite_rows["V"] = 0.0
        infinite_rows.index = pd.MultiIndex.from_product([[INFINITE_REGIME], infinite_rows.index],
                                               names=table.index.names)
        table = pd.concat([table, infinite_rows])

    table["gran"] = table["h"] - table["h_struct"]
    table["gran_share"] = table["gran"] / table["h"]
    # THE TWO FACTORS OF THE GRANULAR TERM, which is what says WHICH force builds it.
    # Sector by sector eq. (granular_buyer) is `gran_rs = V_s (1 - H(gamma_rs))`, so
    # granularity is large either because the buyer buys few effective VARIETIES in
    # that sector (`V_s` high) or because its expected network is dispersed and leaves
    # chance room to work in (`slack_rs = 1 - H(gamma_rs)` near its ceiling of one).
    # That identity is EXACT at the (sector, buyer) level and is where the question
    # should be read; `out.attrs["by_sector"]` carries it.
    #
    # At the BUYER level the two factors are spend-weighted averages, and the product
    # of averages is not the average of products: `gran_resid = gran - V * slack` is
    # exactly the cross-sector covariance `Cov_theta(V_s, slack_rs)` under the buyer's
    # input mix — whether a buyer's many-variety sectors are also its dispersed ones.
    # It is NOT predicted to vanish (`V_s` varies with `N_s`, which ranges 12-24 on the
    # estimates; the buyer-independence the model predicts is across BUYERS at a fixed
    # sector, a different statement), so it is reported rather than assumed away.
    table["slack"] = 1.0 - table["h_struct"]
    table["gran_VS"] = table["V"] * table["slack"]
    table["gran_resid"] = table["gran"] - table["gran_VS"]

    # the (sector, buyer) view, where `gran = V x slack` holds exactly
    by_cell_long["slack"] = 1.0 - by_cell_long["h_struct"].to_numpy()
    by_cell_long["gran"] = by_cell_long["h"].to_numpy() - by_cell_long["h_struct"].to_numpy()
    # the (sector, buyer) cells themselves, where `gran = V_s x (1 - H_rs)` is an
    # identity and nothing has been averaged yet — `granular_factor_decomposition`
    # reads WHICH of the two factors carries the product off exactly this frame.
    table.attrs["by_cell"] = by_cell_long[["regime", "sector", "ze2010_downstream", "spend",
                                 "h_struct", "V", "h", "slack", "gran"]].copy()
    by_sector_table = by_cell_long.groupby(["regime", "sector"], sort=False).apply(
        lambda group_rows: pd.Series({
            "V_s": np.average(group_rows["V"], weights=group_rows["spend"]) if group_rows["spend"].sum() > 0
                   else np.nan,
            "slack (median)": group_rows["slack"].median(),
            "gran (median)": group_rows["gran"].median(),
            "H(gamma) (median)": group_rows["h_struct"].median(),
            "spend share": group_rows["spend"].sum()}), include_groups=False)
    by_sector_table["spend share"] /= by_sector_table.groupby(level="regime")["spend share"].transform("sum")
    table.attrs["by_sector"] = by_sector_table
    table["n_eff"] = 1.0 / table["h"]
    table["n_eff_struct"] = 1.0 / table["h_struct"]
    table["region"] = names.reindex(
        table.index.get_level_values("ze2010_downstream")).to_numpy()
    table.index.names = ["regime", "ze2010_downstream"]

    min_gran = float(np.nanmin(table["gran"].to_numpy())) if table["gran"].notna().any() else 0.0
    if min_gran < -1e-10:
        raise ValueError(f"granularity came out negative ({min_gran:.2e}) — E[H] >= H(gamma) "
                         "is Jensen and cannot fail; the panel and the structural "
                         "network are not the same regime.")

    if verbose:
        print("  E_Omega[H_r] buyer by buyer (spend-weighted over the buyer's sectors)")
        summary = table.groupby(level="regime").apply(
            lambda group_rows: pd.Series({
                # H first, because that is what the decomposition and the figure are
                # written in; the effective numbers are its reciprocal and are NOT
                # additive, so the granular term is `h - h_struct` and never a gap
                # between two effective numbers.
                "H (median)": group_rows["h"].median(),
                "H (min)": group_rows["h"].min(), "H (max)": group_rows["h"].max(),
                "structural H (median)": group_rows["h_struct"].median(),
                "granular H (median)": group_rows["gran"].median(),
                "n_eff (median)": group_rows["n_eff"].median(),
                "structural n_eff (median)": group_rows["n_eff_struct"].median(),
                "granular share (median)": group_rows["gran_share"].median(),
                "V (median)": group_rows["V"].median(),
                "slack 1-H(gamma) (median)": group_rows["slack"].median()}))
        print(summary.round(3).to_string())
        print("    the granular addition is proportional to 1 - H(gamma_rs): it "
              "concentrates whatever dispersion the expected network has left, so it "
              "is largest exactly where comparative advantage has done least.")
        print("\n  which of the two factors builds it — gran = V x slack")
        factor_table = table.groupby(level="regime").apply(
            lambda group_rows: pd.Series({
                "V": group_rows["V"].median(), "slack": group_rows["slack"].median(),
                "V x slack": group_rows["gran_VS"].median(), "gran": group_rows["gran"].median(),
                "cross-sector cov / gran": (group_rows["gran_resid"] / group_rows["gran"]).median(),
                "slack shortfall %": 100.0 * (1.0 - group_rows["slack"].median())}))
        print(factor_table.round(4).to_string())
        print("    `V x slack` is the product of the two spend-weighted averages and "
              "the covariance column the gap to `gran` — whether a buyer's "
              "many-variety sectors are also its dispersed ones. The identity is EXACT "
              "sector by sector; that table is in `.attrs['by_sector']`.")
        print(table.attrs["by_sector"].loc["Both forces"].round(4).to_string()
              if "Both forces" in table.attrs["by_sector"].index.get_level_values(0)
              else "")
        print("    read the SHORTFALL column: `slack` is bounded above by one, so a "
              "shortfall near zero means the expected network is so dispersed that "
              "granularity has essentially the whole unit interval to work in and "
              "`gran ~= V` — the statistic is then an effective number of VARIETIES "
              "with a region label on it, and no counterfactual on the two forces can "
              "move it. A large shortfall would mean comparative advantage has already "
              "concentrated the network and is crowding chance out.")
    return table


def granular_factor_decomposition(table, regime="Both forces"):
    """
    WHICH of the two factors of `gran_rs = V_s (1 - H_rs)` carries it, taken cell by
    cell and read against `H_rs = ||gamma_rs||^2`, the expected network.

    THE QUESTION IS NOT WELL POSED IN LEVELS, and saying so is half the answer. The
    two factors live on different scales: `V_s` is an inverse effective variety count,
    of order a tenth, while `1 - H_rs` is a dispersion capped at one and here sits at
    `0.98`. Reading "0.98 is big and 0.10 is small" attributes the product to the
    slack; dividing both by `H_rs` instead gives `slack/H = 57` against `V/H = 6` and
    attributes it to the slack even more strongly. Both readings are arithmetic
    accidents of where the yardstick is put, and a third grouping reverses them:

        gran_rs / H_rs  =  (V_s / H_rs) * (1 - H_rs)       [slack can only SHRINK it]
                        =  V_s * ((1 - H_rs) / H_rs)       [V_s can only SHRINK it]

    both exact, and each makes its own factor look like the whole story. A product
    cannot be attributed to its factors without a benchmark for each one.

    THE BENCHMARK WITH CONTENT is the value of each factor that alone would make
    granularity exactly as large as the expected network it sits on, `gran_rs = H_rs`:

        V_s^*  = H_rs / (1 - H_rs)      holding the network fixed
        H_rs^* = V_s / (1 + V_s)        holding the variety count fixed

    and the two ratios `V_s/V_s^*` and `H_rs^*/H_rs` answer the question symmetrically.
    On the estimates they come out nearly EQUAL, which is the finding: the granular
    term is large because the buyer buys few effective varieties AND because its
    expected network is dispersed, in roughly equal multiplicative measure. Neither
    force is the story on its own.

    One asymmetry survives and is worth keeping, because it is what the counterfactuals
    can act on: `1 - H_rs` is BOUNDED ABOVE BY ONE and already at `0.98` of that
    ceiling, so the dispersion channel has no room left to grow and only room to be
    taken away, which is exactly what equalising `T` does; `V_s` has no ceiling and is
    a calibration object (`N_s`), which no counterfactual in this section moves.

    Note that `1 - H_rs` and `H_rs` are ONE object, not two: a high slack and a
    dispersed expected network are the same statement, so the decomposition really has
    two free quantities (`V_s` and `H_rs`), not three.

    Returns the per-sector table; `.attrs["summary"]` carries the spend-weighted
    aggregate over cells and `.attrs["cells"]` the (sector, buyer) frame itself.
    """
    cells = table.attrs.get("by_cell")
    if cells is None:
        raise KeyError("no `by_cell` frame on the table — `buyer_granular_concentration` "
                       "stashes it in `.attrs['by_cell']`; re-run that function.")
    cells_regime = cells[cells["regime"] == regime].copy()
    if cells_regime.empty:
        raise KeyError(f"regime {regime!r} is not in the table "
                       f"(have {sorted(cells['regime'].unique())})")
    interior = (np.isfinite(cells_regime["V"].to_numpy()) & np.isfinite(cells_regime["h_struct"].to_numpy())
          & (cells_regime["h_struct"].to_numpy() > 0) & (cells_regime["h_struct"].to_numpy() < 1)
          & (cells_regime["spend"].to_numpy() > 0))
    cells_regime = cells_regime[interior]
    if cells_regime.empty:
        raise ValueError("no (sector, buyer) cell carries a finite V and an interior "
                         "H(gamma) under this regime")

    H = cells_regime["h_struct"].to_numpy(dtype=float)
    V = cells_regime["V"].to_numpy(dtype=float)
    cells_regime["V_over_H"] = V / H
    cells_regime["slack_over_H"] = cells_regime["slack"].to_numpy() / H
    cells_regime["gran_over_H"] = cells_regime["gran"].to_numpy() / H
    cells_regime["V_star"] = H / (1.0 - H)              # V that would give gran = H
    cells_regime["H_star"] = V / (1.0 + V)              # H that would give gran = H
    cells_regime["V_over_Vstar"] = V / cells_regime["V_star"].to_numpy()
    cells_regime["Hstar_over_H"] = cells_regime["H_star"].to_numpy() / H

    spend_weights = cells_regime["spend"].to_numpy(dtype=float)
    weighted_mean = lambda c: float(np.average(cells_regime[c].to_numpy(dtype=float), weights=spend_weights))

    by_sector_table = cells_regime.groupby("sector", sort=True).apply(
        lambda sector_rows: pd.Series({
            "V_s": np.average(sector_rows["V"], weights=sector_rows["spend"]),
            "V_s spread": sector_rows["V"].max() - sector_rows["V"].min(),
            "H_rs": np.average(sector_rows["h_struct"], weights=sector_rows["spend"]),
            "slack 1-H_rs": np.average(sector_rows["slack"], weights=sector_rows["spend"]),
            "gran = V x slack": np.average(sector_rows["gran"], weights=sector_rows["spend"]),
            "V_s/H_rs": np.average(sector_rows["V_over_H"], weights=sector_rows["spend"]),
            "gran/H_rs": np.average(sector_rows["gran_over_H"], weights=sector_rows["spend"]),
            "V_s/V_s*": np.average(sector_rows["V_over_Vstar"], weights=sector_rows["spend"]),
            "H_rs*/H_rs": np.average(sector_rows["Hstar_over_H"], weights=sector_rows["spend"]),
            "spend share": sector_rows["spend"].sum()}), include_groups=False)
    by_sector_table["spend share"] /= by_sector_table["spend share"].sum()

    by_sector_table.attrs["summary"] = pd.Series({
        "V_s": weighted_mean("V"),
        "H_rs = ||gamma_rs||^2": weighted_mean("h_struct"),
        "slack = 1 - H_rs": weighted_mean("slack"),
        "gran = V_s x slack": weighted_mean("gran"),
        # BOTH aggregations of the ratio, because they differ and the gap is
        # informative rather than a rounding: a mean of ratios is dominated by the
        # cells where H_rs is smallest, which are not the cells carrying the euros
        # (C30C alone is 56% of aerospace spending at a ratio of 1.4). The RATIO OF
        # MEANS is the one that reproduces the buyer-level numbers and is the headline.
        "V_s / H_rs (ratio of means)": weighted_mean("V") / weighted_mean("h_struct"),
        "slack / H_rs (ratio of means)": weighted_mean("slack") / weighted_mean("h_struct"),
        "gran / H_rs (ratio of means)": weighted_mean("gran") / weighted_mean("h_struct"),
        "gran / H_rs (mean of ratios)": weighted_mean("gran_over_H"),
        "V_s* = H/(1-H)": weighted_mean("V_star"),
        "V_s / V_s*": weighted_mean("V_over_Vstar"),
        "H_rs* = V/(1+V)": weighted_mean("H_star"),
        "H_rs* / H_rs": weighted_mean("Hstar_over_H"),
        "cells": float(len(cells_regime))})
    by_sector_table.attrs["cells"] = cells_regime
    return by_sector_table


def report_granular_factors(table, regime="Both forces"):
    """Print `granular_factor_decomposition` with the reading it licenses."""
    by_sec = granular_factor_decomposition(table, regime=regime)
    s = by_sec.attrs["summary"]
    print(f"  gran_rs = V_s x (1 - H_rs) cell by cell, regime {regime!r} "
          f"({int(s['cells'])} (sector, buyer) cells)")
    print(by_sec.round(4).to_string())
    print("\n  spend-weighted over cells")
    print(s.round(4).to_string())
    print(f"\n    In LEVELS the two factors are not comparable — V_s = "
          f"{s['V_s']:.3f} is an inverse variety count, 1 - H_rs = {s['slack = 1 - H_rs']:.3f} a "
          f"dispersion capped at one — and dividing both by H_rs only moves the "
          f"accident around (slack/H = {s['slack / H_rs (ratio of means)']:.0f} "
          f"against V/H = {s['V_s / H_rs (ratio of means)']:.1f}). A product has no "
          f"attribution without a "
          f"benchmark per factor.")
    print(f"    The benchmark with content is the value of each that ALONE would make "
          f"granularity equal the expected network (gran_rs = H_rs): "
          f"V_s* = H/(1-H) = {s['V_s* = H/(1-H)']:.4f}, against a measured "
          f"{s['V_s']:.4f}, i.e. {s['V_s / V_s*']:.1f}x too many euros on too few "
          f"varieties; and H_rs* = V/(1+V) = {s['H_rs* = V/(1+V)']:.4f}, against a "
          f"measured {s['H_rs = ||gamma_rs||^2']:.4f}, i.e. a network "
          f"{s['H_rs* / H_rs']:.1f}x too dispersed. The two are of the SAME order, so "
          f"the granular term is large on both counts at once and neither factor is "
          f"the story alone.")
    print(f"    The one asymmetry that matters for the counterfactuals: 1 - H_rs is "
          f"bounded above by one and already at {s['slack = 1 - H_rs']:.3f} of that ceiling, so "
          f"the dispersion channel can only be taken AWAY (equalising T does exactly "
          f"that), while V_s has no ceiling and is set by the calibrated N_s, which no "
          f"counterfactual here moves.")
    return by_sec

def plot_buyer_granular_concentration(table, baseline="Both forces", units="pct",
                                      order=None, annotate_level=True, figsize=None,
                                      save_to=None, xmin=None, xmax=None,
                                      kind="bar", logx=False, quantity="n_eff",
                                      drop=(INFINITE_REGIME,)):
    """
    The per-buyer figure of `E_Omega[H_r(Omega)]`, in the form
    `plot_counterfactual_distance` uses for the sourcing distance: one row per shocked
    commuting zone, one mark per regime, the estimated economy as the origin.

    `quantity` chooses what is DRAWN. `"h"` is the concentration itself,
    `E_Omega[H_r(Omega)]` — the object the decomposition is written in, so a regime's
    distance from `Infinite varieties` IS the granular term of eq. (granular_buyer) in
    its own units, and the two halves add. `"n_eff"` (the default) draws its reciprocal
    `1/E[H_r]`, an effective number of supplier zones, which is the unit a reader can
    hold but is NOT additive: `1/H` is convex, so a gap in effective numbers is not the
    granular term and cannot be compared across buyers of different size.
    `units="pct"` expresses each regime as a
    percentage deviation from `baseline`, which is what makes the comparison legible —
    the levels differ across buyers by an order of magnitude, so on a level axis the
    cross-buyer spread crowds out the counterfactual, which is the whole content. The
    baseline level is printed at the right margin, so every percentage converts back.

    `drop` removes regimes from the figure, and it defaults to `Infinite varieties`
    for a reason of scale rather than of interest. That series is `H(gamma_r)`, which
    at this calibration is an order of magnitude below the realised `H_r`, so on a
    percentage axis it sits near -90% and compresses the two force counterfactuals —
    which move by single-digit percents — into the zero rule. Granularity is the
    LARGEST of the three effects and drawing it beside them hides the other two; it
    belongs in the text as one number, with the figure left to the comparison it can
    actually resolve. Pass `drop=()` to draw it anyway.
    """
    if units not in ("pct", "level"):
        raise ValueError(f"units must be 'pct' or 'level', got {units!r}.")
    if kind not in ("bar", "point"):
        raise ValueError(f"kind must be 'bar' or 'point', got {kind!r}.")
    if logx and kind == "bar":
        raise ValueError("a log axis has no meaningful origin, so a bar's LENGTH "
                         "would encode the axis limits rather than the number; use "
                         "kind='point' with logx=True.")
    if logx and units != "level":
        raise ValueError("logx applies to levels; a percentage deviation changes sign.")
    if quantity not in ("n_eff", "h"):
        raise ValueError(f"quantity must be 'n_eff' or 'h', got {quantity!r}.")
    wide = table[quantity].unstack("regime")
    missing = [regime_to_drop for regime_to_drop in drop if regime_to_drop not in wide.columns]
    if missing:
        raise KeyError(f"cannot drop {missing}: not among {list(wide.columns)}.")
    wide = wide.drop(columns=list(drop))
    names = table["region"].groupby(level="ze2010_downstream").first().reindex(wide.index)
    if baseline not in wide.columns:
        raise KeyError(f"baseline regime {baseline!r} not among {list(wide.columns)}.")
    row_order = wide[baseline].sort_values().index if order is None else pd.Index(order)
    wide = wide.reindex(row_order)
    names = names.reindex(row_order)
    labels = [column for column in wide.columns if column != baseline]

    base = wide[baseline].astype(float)
    if units == "pct":
        wide = wide[labels].apply(lambda column: 100.0 * (column.astype(float) - base) / base)
    else:
        wide = wide[[baseline] + labels]
        labels = list(wide.columns)

    # A ROW must hold a buyer's NAME as well as its marks, so the per-row height is the
    # larger of the two demands. Sizing it off the mark count alone leaves ~12pt a row
    # once a regime is dropped, which no name fits into; 0.28 in is the row height the
    # other per-buyer figures use.
    fig, ax = plt.subplots(
        figsize=figsize or (9, max(3.5, max(0.28, 0.085 * max(len(labels), 1))
                                   * len(wide))))
    y = np.arange(len(wide))
    height = 0.8 / max(len(labels), 1)
    palette = dict(CF_COLORS)
    palette.setdefault(INFINITE_REGIME, (0.55, 0.45, 0.65))
    palette.setdefault(UNIFORM_REGIME, (0.72, 0.72, 0.72))
    if kind == "bar":
        for k, regime in enumerate(labels):
            offset = ((len(labels) - 1) / 2 - k) * height
            ax.barh(y + offset, wide[regime].to_numpy(), height=height, alpha=1.0,
                    color=palette.get(regime, toulouse_color), edgecolor="white",
                    linewidth=0.4, label=regime)
    else:
        for row_position, (_, buyer_row) in enumerate(wide.iterrows()):
            row_values = buyer_row.to_numpy(dtype=float)
            if np.isfinite(row_values).any():
                ax.plot([np.nanmin(row_values), np.nanmax(row_values)], [row_position, row_position], color="0.8",
                        linewidth=1.0, zorder=1)
        for regime in labels:
            ax.plot(wide[regime].to_numpy(), y, linestyle="none", marker="o",
                    markersize=5, color=palette.get(regime, toulouse_color), label=regime,
                    zorder=2)
    if logx:
        ax.set_xscale("log")
    plotted_values = wide.to_numpy(dtype=float)
    span = float(np.nanmax(plotted_values) - np.nanmin(plotted_values)) + 1e-9
    if logx:
        lo = float(np.nanmin(plotted_values)) / 1.15 if xmin is None else xmin
        hi = float(np.nanmax(plotted_values)) * 1.15 if xmax is None else xmax
    else:
        lo = (min(0.0, float(np.nanmin(plotted_values))) - 0.03 * span) if xmin is None else xmin
        hi = (float(np.nanmax(plotted_values)) + 0.05 * span) if xmax is None else xmax
    if units == "pct":
        ax.axvline(0.0, color="0.35", linewidth=1.0, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(names.astype(str))
    set_name_axis_fontsize(ax, len(wide))
    ax.set_ylim(-0.6, len(wide) - 0.4)
    ax.set_xlim(lo, hi)
    qsym = ("$\\mathbb{E}_\\Omega[H_r]$" if quantity == "h"
            else "$1/\\mathbb{E}_\\Omega[H_r]$")
    qname = ("Concentration of the supplier portfolio" if quantity == "h"
             else "Effective number of supplier commuting zones")
    ax.set_xlabel(f"Change in {qname.lower()} {qsym} ({pct()} of {baseline})"
                  if units == "pct" else f"{qname} {qsym}")
    ax.grid(alpha=0.2, axis="x")
    if units == "pct" and annotate_level:
        # one entry per row, like the names on the left, so it takes the same cap
        margin_size = name_axis_fontsize(ax, len(wide), size_at_ref=7.5)
        for row_position, i in enumerate(wide.index):
            ax.annotate(f"{base.loc[i]:.3f}" if quantity == "h"
                        else f"{base.loc[i]:.1f}", xy=(1.005, row_position),
                        xycoords=("axes fraction", "data"), va="center", ha="left",
                        fontsize=margin_size, color="0.35", annotation_clip=False)
        ax.annotate(f"{baseline}\n{qsym}", xy=(1.005, 1.005),
                    xycoords="axes fraction", va="bottom", ha="left",
                    fontsize=margin_size, color="0.35", annotation_clip=False)
    ax.legend(frameon=False, fontsize=fs(8), loc="lower right")
    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, bbox_inches="tight")
    return ax


# --- P8: the number of upstream REGIONS, which is not what a Herfindahl counts ---------
#
# `1/E[H_r]` is contaminated by the variety count and cannot answer "how many regions
# does this buyer source from". Decomposing eq. (granular_buyer),
#
#     E[H_rs] = V_rs + (1 - V_rs) H(gamma_rs) ,
#
# the FIRST term carries 83-91% of E[H_rs] at the measured V_s N_s = 1.81 and
# N_s = 12-24, so the reciprocal is an effective number of VARIETIES with a region
# label on it. That is also why the two counterfactuals move it by 1% and 11% while
# the infinite-variety limit moves it by 611%: comparative advantage and trade costs
# enter only through gamma, which owns the remaining tenth.
#
# The object with no variety weighting in it at all is the REACH — the number of
# distinct upstream commuting zones that win at least one variety for the buyer.
# Within sector s, region r' loses every one of the N_s varieties with probability
# (1 - gamma_r'rs)^N_s (winners independent across varieties, each won by r' with
# probability gamma), and variety sets are independent across sectors, so
#
#     E_Omega[ #{ r' : r' supplies r } ] = sum_r' [ 1 - prod_s (1 - gamma_r'rs)^{N_s} ] .
#
# Exact under the same independence that gives eq. (granular_binomial), and it needs
# NO expenditure shares: V never enters. Its N_s -> infinity limit is the SUPPORT,
# the number of zones the buyer could reach at all, so the granularity gap is a
# statement about geography rather than about how expenditure piles onto one variety.
#
# The two statistics are complements and neither replaces the other. Reach counts a
# zone winning one variety the same as one winning forty, so it says where the shock
# ARRIVES and not where the euros land; the effective number of the EXPECTED network,
# 1/H(gamma_r), weights by euros and is equally free of V. Read them as a pair.

def buyer_region_reach(data, regimes=None, value_col="share", spend=None, n_hat=None,
                       include_infinite=True, verbose=True):
    """
    `E_Omega[#{r' : r' supplies buyer r}]`, the expected number of distinct upstream
    commuting zones a shock to buyer `r` reaches, per (regime, buyer).

    Columns
    -------
    reach        the expectation above, summed over zones and across the buyer's
                 sectors (a zone supplying in two sectors is counted once).
    support      its `N_s -> infinity` limit: the number of zones with `gamma > 0` in
                 at least one of the buyer's sectors. `Infinite varieties` carries
                 exactly this in `reach`.
    reach_share  `reach / support`, the fraction of the reachable geography one
                 realisation actually touches.
    n_eff_struct `1/H(gamma_r)`, the effective number of zones of the EXPECTED
                 network — the euro-weighted region count, likewise free of `V`,
                 reported beside `reach` because the two answer different questions.

    NO EXPENDITURE WEIGHTS enter `reach`, which is the point: `1/E[H_r]` is 83-91%
    variety count at this calibration, and this statistic is a pure geography count.
    The caveat that follows from the same thing: a sector the buyer barely buys from
    contributes its full reach, so `reach` says how many places are touched and not
    how much lands in them.
    """
    sector_spend = _sector_spend(data, value_col) if spend is None else spend
    buyers = np.asarray(sector_spend.index).astype(int)
    want = dict(CF_REGIMES if regimes is None else regimes)
    want.setdefault(UNIFORM_REGIME, dict(alpha=0.0, equalise_T=True))
    names = _region_labels(data).set_index("index")["ze2010_name"]

    if n_hat is None:
        n_hat = data.get("post_hoc_N_hat")
        if n_hat is None:
            n_hat = _n_hat_from_diagnostics(data)
    if n_hat is None:
        raise ValueError("no N_hat_s available — the variety count is what turns a "
                         "sourcing share into a probability of being reached.")
    n_hat = np.asarray(n_hat, dtype=float).ravel()

    rows, sector_rows = [], []
    for regime, regime_options in want.items():
        networks = structural_networks(data, buyers=buyers, **regime_options)
        # log of the probability that a zone wins NOTHING, accumulated over sectors
        logfail = {}            # regime -> (buyer x zone) log prob of no win
        support = {}
        n_zone = int(np.max([sector_block["cells"].max() for sector_block in networks["by_sector"].values()])) + 1
        log_no_win = np.zeros((buyers.size, n_zone))
        reachable = np.zeros((buyers.size, n_zone), dtype=bool)
        for s, sector_block in networks["by_sector"].items():
            if s >= n_hat.size or not np.isfinite(n_hat[s]):
                raise KeyError(f"sector {s} has no variety count.")
            N = float(n_hat[s])
            gamma_rs = np.clip(np.asarray(sector_block["W"], dtype=float), 0.0, 1.0)   # (buyer x cell)
            cells = np.asarray(sector_block["cells"]).astype(int)
            # log1p(-g) is -inf at g = 1 (a single-cell sector), which is right:
            # that zone is reached with probability one.
            contrib = N * np.log1p(-gamma_rs)
            np.add.at(log_no_win.T, cells, contrib.T)
            reachable[:, cells] |= gamma_rs > 0
            reach_s = (1.0 - np.exp(contrib)).sum(axis=1)
            sector_rows.append(pd.DataFrame({
                "regime": regime, "sector": s, "ze2010_downstream": buyers,
                "n_hat_s": N, "n_cells": cells.size, "reach": reach_s,
                "support": int((gamma_rs > 0).any(axis=0).sum())}))
        reach = (1.0 - np.exp(log_no_win)).sum(axis=1)
        support_count = reachable.sum(axis=1).astype(float)
        # the euro-weighted region count of the same regime, for the pair reading
        herfindahl_gamma = np.zeros(buyers.size); spend_total = np.zeros(buyers.size)
        for s, sector_block in networks["by_sector"].items():
            spend_weight = sector_spend.iloc[:, s].to_numpy(dtype=float)
            herfindahl_gamma += spend_weight * (np.asarray(sector_block["W"], dtype=float) ** 2).sum(axis=1)
            spend_total += spend_weight
        rows.append(pd.DataFrame({
            "regime": regime, "ze2010_downstream": buyers, "reach": reach,
            "support": support_count, "reach_share": reach / np.where(support_count > 0, support_count, np.nan),
            "n_eff_struct": spend_total / np.where(herfindahl_gamma > 0, herfindahl_gamma, np.nan)}))

    reach_table = pd.concat(rows, ignore_index=True)
    if include_infinite and "Both forces" in want:
        infinite_rows = reach_table[reach_table["regime"] == "Both forces"].copy()
        infinite_rows["regime"] = INFINITE_REGIME
        infinite_rows["reach"] = infinite_rows["support"]
        infinite_rows["reach_share"] = 1.0
        reach_table = pd.concat([reach_table, infinite_rows], ignore_index=True)
    reach_table = reach_table.set_index(["regime", "ze2010_downstream"])
    reach_table["region"] = names.reindex(
        reach_table.index.get_level_values("ze2010_downstream")).to_numpy()
    reach_table.attrs["by_sector"] = pd.concat(sector_rows, ignore_index=True)

    excess_reach = reach_table["reach"] - reach_table["support"]
    if float(excess_reach.max()) > 1e-9:
        raise ValueError(f"reach exceeded its own support by {float(excess_reach.max()):.2e} — "
                         "a zone cannot be reached more often than it exists.")
    if verbose:
        print("  the number of distinct upstream commuting zones reached")
        summ = reach_table.groupby(level="regime").apply(lambda d: pd.Series({
            "reach (median)": d["reach"].median(), "reach (min)": d["reach"].min(),
            "reach (max)": d["reach"].max(), "support (median)": d["support"].median(),
            "reach/support (median)": d["reach_share"].median(),
            "1/H(gamma_r) (median)": d["n_eff_struct"].median()}))
        print(summ.round(2).to_string())
        print("    no expenditure weights enter `reach`: it is a count of places, not "
              "of euros, and carries none of the variety concentration that makes "
              "1/E[H_r] a variety count.")
    return reach_table


def plot_buyer_region_reach(table, baseline="Both forces", units="level", order=None,
                            annotate_level=True, figsize=None, save_to=None,
                            xmin=None, xmax=None):
    """
    The per-buyer figure of the REACH, in the form `plot_counterfactual_distance` uses.

    `units="level"` is the default here and `units="pct"` the option, the reverse of
    the concentration figure, because the reaches of the different regimes sit within
    a factor of about two of each other rather than an order of magnitude: the level
    is legible directly, and it is the number the question asks for.
    """
    if units not in ("level", "pct"):
        raise ValueError(f"units must be 'level' or 'pct', got {units!r}.")
    wide = table["reach"].unstack("regime")
    names = table["region"].groupby(level="ze2010_downstream").first().reindex(wide.index)
    if baseline not in wide.columns:
        raise KeyError(f"baseline regime {baseline!r} not among {list(wide.columns)}.")
    row_order = wide[baseline].sort_values().index if order is None else pd.Index(order)
    wide, names = wide.reindex(row_order), names.reindex(row_order)
    base = wide[baseline].astype(float)
    labels = [column for column in wide.columns if column != baseline]
    if units == "pct":
        wide = wide[labels].apply(lambda column: 100.0 * (column.astype(float) - base) / base)
    else:
        labels = [baseline] + labels
        wide = wide[labels]

    # A ROW must hold a buyer's NAME as well as its marks, so the per-row height is the
    # larger of the two demands. Sizing it off the mark count alone leaves ~12pt a row
    # once a regime is dropped, which no name fits into; 0.28 in is the row height the
    # other per-buyer figures use.
    fig, ax = plt.subplots(
        figsize=figsize or (9, max(3.5, max(0.28, 0.085 * max(len(labels), 1))
                                   * len(wide))))
    y = np.arange(len(wide))
    palette = dict(CF_COLORS)
    palette.setdefault(INFINITE_REGIME, (0.55, 0.45, 0.65))
    palette.setdefault(UNIFORM_REGIME, (0.72, 0.72, 0.72))
    height = 0.8 / max(len(labels), 1)
    for k, regime in enumerate(labels):
        offset = ((len(labels) - 1) / 2 - k) * height
        ax.barh(y + offset, wide[regime].to_numpy(), height=height, alpha=1.0,
                color=palette.get(regime, toulouse_color), edgecolor="white",
                linewidth=0.4, label=regime)
    plotted_values = wide.to_numpy(dtype=float)
    span = float(np.nanmax(plotted_values) - np.nanmin(plotted_values)) + 1e-9
    lo = (0.0 if units == "level" else min(0.0, float(np.nanmin(plotted_values))) - 0.03 * span) \
        if xmin is None else xmin
    hi = (float(np.nanmax(plotted_values)) + 0.05 * span) if xmax is None else xmax
    if units == "pct":
        ax.axvline(0.0, color="0.35", linewidth=1.0, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(names.astype(str))
    set_name_axis_fontsize(ax, len(wide))
    ax.set_ylim(-0.6, len(wide) - 0.4)
    ax.set_xlim(lo, hi)
    ax.set_xlabel("Number of upstream commuting zones supplying the buyer"
                  if units == "level" else
                  f"Change in the number of upstream commuting zones "
                  f"({pct()} of {baseline})")
    ax.grid(alpha=0.2, axis="x")
    if annotate_level and "support" in table.columns:
        support = table["support"].groupby(level="ze2010_downstream").max().reindex(wide.index)
        # one entry per row, like the names on the left, so it takes the same cap
        margin_size = name_axis_fontsize(ax, len(wide), size_at_ref=7.5)
        for row_position, i in enumerate(wide.index):
            ax.annotate(f"{support.loc[i]:.0f}", xy=(1.005, row_position),
                        xycoords=("axes fraction", "data"), va="center", ha="left",
                        fontsize=margin_size, color="0.35", annotation_clip=False)
        ax.annotate("Reachable\nzones", xy=(1.005, 1.005), xycoords="axes fraction",
                    va="bottom", ha="left", fontsize=margin_size, color="0.35",
                    annotation_clip=False)
    ax.legend(frameon=False, fontsize=fs(8), loc="lower right")
    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, bbox_inches="tight")
    return ax


# COMMONALITY: do buyers agree on which suppliers, and how much of a buyer's
# exposure survives aggregation across buyers.
#
# Everything above is one buyer at a time. Everything here compares buyers: the
# identity that splits Hbar_s into a common and a buyer-specific half, the
# two-buyer co-sourcing kernel Q against the structural overlap Gamma, the four
# cells, and the exhibits built on them. It RUNS ON the cell above, so run that
# one first.
# (2) THE STATISTIC DECOMPOSES EXACTLY, so there is no need for an overlap measure at
#     all. With buyer weights `pi_rs` (the share of sector s's purchases made by buyer
#     r) write `omega_bar_s = sum_r pi_rs omega_rs`. Then
#
#         Hbar_s  ==  sum_r pi_rs H_rs  =  H(omega_bar_s)  +  M_s ,
#         M_s     =  sum_l Var_pi(omega_lrs)  >= 0,
#         C_s     =  1 - M_s / Hbar_s .
#
#     The reading is economic, not algebraic. A shock to ONE buyer lands with
#     concentration `H_rs`. A shock to the WHOLE industry lands with `H(omega_bar_s)`.
#     So `C_s` is the share of a buyer's concentration that aggregating across buyers
#     CANNOT diversify away — the part that is common — and `1 - C_s` is the part that
#     is buyer-specific. `tv_common` measured the same idea with a statistic that does
#     not decompose; this one does, so the two halves add up and can be counted in
#     euros.
#
# The 2x2 keeps all four cells, and the fourth is not empty: low concentration with low
# commonality is DIFFUSE LOCAL SOURCING — every buyer spreads over its own
# neighbourhood — which is exactly where the `Equal comparative advantage` regime sits.
#
#        C high  |  common pool          hub-and-spoke
#        C low   |  diffuse local        local pockets
#                +------------------------------------
#                   H low                H high
#
# Two conventions, both deliberate and both departures from the paragraphs above.
#
#   WEIGHTS ARE SPENDING WEIGHTS, NOT MEDIANS. The identity holds for a weighted MEAN
#   over buyers and for no other aggregator, so the means here are spend-weighted and
#   the distributions go in the appendix. Reporting a median of `C_r` would break the
#   adding up.
#
#   "STRUCTURAL" MEANS THE EXPECTED NETWORK, NOT THE EK PROBABILITIES. `rho` is the
#   probability a cell WINS a variety; `omega` is the share of the EURO it collects, and
#   under CES a cheaper winner takes a larger value share, so the two differ by a cross
#   term. They are close in practice and `expected_network_check` measures the gap
#   rather than asserting it — but the object the identity is about is `E[omega]`.
#
def concentration_summary(table, verbose=True):
    """
    The sector table reduced to one row per regime, aggregated to the BUYER with each
    buyer's own input mix `theta_rs` and then across buyers with its spending.

    `H_r = sum_s theta_rs H_rs` is what a shock to buyer `r` delivers once its whole
    input basket is counted, so the industry number is a double spending average of the
    (sector x buyer) cells — never a median, or the identity stops adding up. The
    columns repeat the sector ones and carry the same reading.
    """
    per_buyer = table.attrs["per_buyer"].copy()
    per_buyer["_h"] = per_buyer["h"] * per_buyer["spend"]
    per_buyer["_m"] = per_buyer["m"] * per_buyer["spend"]
    regime_sums = per_buyer.groupby("regime", sort=False)[["_h", "_m", "spend"]].sum()
    summary_table = pd.DataFrame({"h_bar": regime_sums["_h"] / regime_sums["spend"], "m": regime_sums["_m"] / regime_sums["spend"]})
    summary_table["h_common"] = summary_table["h_bar"] - summary_table["m"]
    summary_table["C"] = 1.0 - summary_table["m"] / summary_table["h_bar"]
    summary_table["n_eff"] = 1.0 / summary_table["h_bar"]
    summary_table["n_eff_common"] = 1.0 / summary_table["h_common"]
    uniform_n_eff = summary_table["n_eff"].get(UNIFORM_REGIME, np.nan)
    summary_table["n_eff_ratio"] = summary_table["n_eff"] / uniform_n_eff
    summary_table = summary_table[["h_bar", "h_common", "m", "C", "n_eff", "n_eff_common", "n_eff_ratio"]]
    if verbose:
        print("  the industry aggregate (spend-weighted over sectors AND buyers)")
        print(summary_table.round(3).to_string())
        print("    C = 1 means nothing is buyer-specific; under alpha = 0 it is exact.")
    return summary_table


def realised_sector_incidence(data, value_col="share"):
    """
    The realised incidence `omega^R_{lrs}` of every (replication, sector, buyer), from
    the parquet, as a dict `sector -> {"W": (rep, buyer, cell)}` over that sector's
    modelled cells.

    Built on the SAME cell support as the structural network, so the two are comparable
    entry by entry and a realisation that reaches only a handful of cells shows up as a
    row of mostly zeros rather than as a shorter vector.
    """
    suppliers = data.get("suppliers")
    if suppliers is None or "replication" not in suppliers.columns:
        raise ValueError("the parquet carries no `replication` column — there is no "
                         "realised finite-variety economy here, only the continuum solve.")
    suppliers = suppliers.assign(_s=_parquet_sector_index(data, suppliers))
    replications = np.sort(suppliers["replication"].unique())
    replication_position = {int(b): i for i, b in enumerate(replications)}
    CELL_MASK = data["CELL_MASK"]
    by_sector = {}
    for s, sector_rows in suppliers.groupby("_s", sort=True):
        cells = np.flatnonzero(CELL_MASK[s])
        if cells.size == 0:
            continue
        cell_position = {int(cell) + 1: i for i, cell in enumerate(cells)}   # 1-based ZE -> column
        buyers = np.sort(sector_rows["ze2010_downstream"].unique())
        buyer_position = {int(b): i for i, b in enumerate(buyers)}
        realised_spend = np.zeros((replications.size, buyers.size, cells.size))
        flows = sector_rows.groupby(["replication", "ze2010_downstream", "ze2010"])[value_col].sum()
        for (b, buyer, zone), value in flows.items():
            j = cell_position.get(int(zone))
            if j is None:                       # a winner outside the modelled cells
                continue
            realised_spend[replication_position[int(b)], buyer_position[int(buyer)], j] += float(value)
        buyer_total = realised_spend.sum(axis=2, keepdims=True)
        by_sector[s] = {"W": np.divide(realised_spend, buyer_total, out=np.zeros_like(realised_spend), where=buyer_total > 0),
                  "cells": cells, "buyers": buyers, "reach": (realised_spend > 0).sum(axis=2)}
    return {"by_sector": by_sector, "replications": replications}


def expected_network_check(data, value_col="share", spend=None, verbose=True):
    """
    Is the EXPECTED network the win probability? `rho_lrs` is the probability a cell wins
    a variety; `omega_lrs` is the share of the euro it collects, and under CES a cheaper
    winner takes a larger value share, so `E[omega] = rho` only when varieties carry
    equal expenditure shares. The gap is a cross term between winning and being cheap.

    This measures it rather than assuming it away: the mean realised incidence across
    replications against `rho`, sector by sector, as a total-variation distance (the
    euro share that would have to move) and a correlation. It is the footnote the
    identity below rests on.
    """
    realised = realised_sector_incidence(data, value_col)
    networks = structural_networks(data)
    rows = []
    for s, realised_block in realised["by_sector"].items():
        structural_block = networks["by_sector"].get(s)
        if structural_block is None:
            continue
        buyer_column = {int(b): i for i, b in enumerate(structural_block["buyers"])}
        take = [buyer_column[int(b)] for b in realised_block["buyers"] if int(b) in buyer_column]
        if len(take) != realised_block["W"].shape[1]:
            continue
        mean_realised = realised_block["W"].mean(axis=0)                        # (buyer, cell)
        win_probability = structural_block["W"][take]
        rows.append({"sector": s, "sector_name": str(data["sector_names"][s]),
                     "tv": float(np.median(0.5 * np.abs(mean_realised - win_probability).sum(axis=1))),
                     "corr": float(np.corrcoef(mean_realised.ravel(), win_probability.ravel())[0, 1]),
                     "n_rep": int(realised_block["W"].shape[0])})
    check_table = pd.DataFrame(rows).set_index("sector").sort_index()
    if verbose:
        print("  E[omega] against rho (the expected network vs the win probability)")
        print(check_table.round(4).to_string())
        print(f"    median TV distance {check_table['tv'].median():.4f} — the euro share that "
              "would have to move; with finitely many draws part of it is sampling "
              "noise, so this is an upper bound on the CES cross term.")
    return check_table


def granularity_concentration(data, value_col="share", spend=None, structural=None,
                              verbose=True):
    """
    What the finite variety count adds to concentration, and to buyer-specificity.

    Because `H` is quadratic, granularity is not noise averaging out: with `N_s`
    varieties per sector each won by exactly one cell,

        E[H_rs]  ~=  H^E_rs  +  (1 - H^E_rs) / N_s ,

    exact when varieties carry equal expenditure shares. Two predictions follow, and the
    table tests both.

      (1) GRANULARITY AND COMPARATIVE ADVANTAGE ARE SUBSTITUTES as sources of
          concentration: the term it adds is proportional to `1 - H^E`, so it adds most
          exactly where the structure is diffuse and almost nothing where a hub already
          holds the mass.

      (2) GRANULARITY AND DISTANCE ARE COMPLEMENTS as sources of buyer-specificity:
          chance winners differ across buyers only in so far as the buyers are far
          enough apart to rank cells differently at all.

    Taking expectations of the identity splits `E[Hbar_s]` into FOUR additive cells,

        E[Hbar_s] = H(omega_bar^E)      structural, common
                  + M^E                 structural, buyer-specific
                  + E[H(omega_bar^R)] - H(omega_bar^E)     granular, common
                  + E[M^R] - M^E                           granular, buyer-specific

    each computed within a realisation and then averaged, so `h_bar_realised` is
    reproduced exactly by the four. The last two are UNTARGETED: bilateral links are not
    observed, so they are predictions of the model, not fitted quantities.
    """
    sector_spend = _sector_spend(data, value_col) if spend is None else spend
    realised = realised_sector_incidence(data, value_col)
    networks = structural_networks(data, buyers=np.asarray(sector_spend.index).astype(int))
    n_hat = data.get("post_hoc_N_hat")
    if n_hat is None:
        n_hat = _n_hat_from_diagnostics(data)
    n_hat = None if n_hat is None else np.asarray(n_hat, dtype=float).ravel()

    rows = []
    for s, realised_block in realised["by_sector"].items():
        structural_block = networks["by_sector"].get(s)
        if structural_block is None:
            continue
        sector_spend_column = sector_spend.iloc[:, s]
        spend_weight = sector_spend_column.reindex(realised_block["buyers"]).fillna(0.0).to_numpy(dtype=float)
        if spend_weight.sum() <= 0:
            continue
        buyer_column = {int(b): i for i, b in enumerate(structural_block["buyers"])}
        take = [buyer_column[int(b)] for b in realised_block["buyers"] if int(b) in buyer_column]
        if len(take) != realised_block["W"].shape[1]:
            continue
        h_struct, h_common_struct, m_struct, _, _, _ = _conc_identity(structural_block["W"][take], spend_weight)
        h_real = h_common_real = m_real = 0.0
        for b in range(realised_block["W"].shape[0]):
            h_rep, hc_b, m_rep, _, _, _ = _conc_identity(realised_block["W"][b], spend_weight)
            h_real += h_rep; h_common_real += hc_b; m_real += m_rep
        n_rep = realised_block["W"].shape[0]
        h_real, h_common_real, m_real = h_real / n_rep, h_common_real / n_rep, m_real / n_rep
        n_hat_sector = np.nan if n_hat is None or s >= n_hat.size else float(n_hat[s])
        rows.append({
            "sector": s, "sector_name": str(data["sector_names"][s]),
            "n_hat_s": n_hat_sector, "spend": float(spend_weight.sum()), "n_rep": n_rep,
            "reach": float(realised_block["reach"].mean()),
            "h_structural": h_struct, "h_realised": h_real,
            "h_predicted": h_struct + (1.0 - h_struct) / n_hat_sector if n_hat_sector and np.isfinite(n_hat_sector) else np.nan,
            "struct_common": h_common_struct, "struct_specific": m_struct,
            "gran_common": h_common_real - h_common_struct, "gran_specific": m_real - m_struct,
            "C_structural": 1.0 - m_struct / h_struct if h_struct > 0 else np.nan,
            "C_realised": 1.0 - m_real / h_real if h_real > 0 else np.nan,
        })
    granularity_table = pd.DataFrame(rows).set_index("sector").sort_index()
    four_cells = granularity_table[["struct_common", "struct_specific", "gran_common", "gran_specific"]]
    granularity_table["gran_share_conc"] = (granularity_table["gran_common"] + granularity_table["gran_specific"]) / granularity_table["h_realised"]
    granularity_table["gran_share_spec"] = granularity_table["gran_specific"] / \
        (granularity_table["struct_specific"] + granularity_table["gran_specific"])
    identity_residual = float(np.nanmax(np.abs(four_cells.sum(axis=1) - granularity_table["h_realised"])))
    granularity_table.attrs["identity_residual"] = identity_residual
    if verbose:
        print("  what granularity adds (the four cells of E[Hbar_s])")
        print(granularity_table.round(4).to_string())
        print(f"    the four cells reproduce h_realised to {identity_residual:.2e}")
        with_prediction = granularity_table[["h_predicted", "h_realised"]].dropna()
        if len(with_prediction):
            max_relative_gap = np.max(np.abs(with_prediction["h_predicted"] - with_prediction["h_realised"])
                         / np.maximum(with_prediction["h_realised"], 1e-12))
            print(f"    H^E + (1-H^E)/N_s against the realised mean: max relative "
                  f"gap {max_relative_gap:.3f} (equal variety shares is the approximation)")
    return granularity_table


def concentration_table(data, value_col="share", sector=None, granular=None,
                        verbose=True):
    """
    The section's table: one row per regime, the structural pair, the realised effective
    number, and the two granular shares.

    Columns (1)-(2) belong to the two architectures and to the force decomposition;
    (3)-(5) to the granularity paragraph. Column (5) is blank in the `alpha = 0` row by
    construction — with no distance there is nothing for granularity to be
    buyer-specific about, which is the internal check rather than a missing number.
    """
    sector_table = sector_concentration(data, value_col=value_col, verbose=False) \
        if sector is None else sector
    summary_table = concentration_summary(sector_table, verbose=False)
    regime_table = pd.DataFrame({
        "n_eff_rel_uniform": summary_table["n_eff_ratio"],
        "C_structural": summary_table["C"],
    })
    if granular is not None and len(granular):
        sector_weight = granular["spend"] / granular["spend"].sum()
        regime_table["n_eff_realised"] = np.nan
        regime_table["gran_share_conc"] = np.nan
        regime_table["gran_share_spec"] = np.nan
        regime_table.loc["Both forces", "n_eff_realised"] = \
            1.0 / float(sector_weight @ granular["h_realised"])
        regime_table.loc["Both forces", "gran_share_conc"] = \
            float(sector_weight @ (granular["gran_common"] + granular["gran_specific"])) / \
            float(sector_weight @ granular["h_realised"])
        regime_table.loc["Both forces", "gran_share_spec"] = \
            float(sector_weight @ granular["gran_specific"]) / \
            float(sector_weight @ (granular["struct_specific"] + granular["gran_specific"]))
    if verbose:
        print(regime_table.round(3).to_string())
    return regime_table


def incidence_variance_map(data, value_col="share", spend=None, verbose=True):
    """
    Which commuting zones GENERATE the buyer-specific term.

    `M = sum_l sum_s theta_s Var_pi(omega_lrs)` is a sum over upstream zones, so it can
    be attributed: a zone contributes when it serves some buyers heavily and others not
    at all, which is the signature of a local pocket. A zone serving everyone in
    proportion contributes nothing however large it is — which is why this is NOT the
    aggregate incidence of Figure 3 (that is `omega_bar`, the SQUARE of which is the
    common term).
    """
    sector_spend = _sector_spend(data, value_col) if spend is None else spend
    networks = structural_networks(data, buyers=np.asarray(sector_spend.index).astype(int))
    R = data["R"]
    buyer_variance = np.zeros(R)
    common_incidence = np.zeros(R)
    tot_spend = float(sector_spend.to_numpy().sum())
    for s, sector_block in networks["by_sector"].items():
        buyer_spend = sector_spend.iloc[:, s].to_numpy(dtype=float)
        if buyer_spend.sum() <= 0:
            continue
        buyer_weight = buyer_spend / buyer_spend.sum()
        incidence = sector_block["W"]
        mean_incidence = buyer_weight @ incidence
        sector_variance = buyer_weight @ ((incidence - mean_incidence) ** 2)
        buyer_variance[sector_block["cells"]] += (buyer_spend.sum() / tot_spend) * sector_variance
        common_incidence[sector_block["cells"]] += (buyer_spend.sum() / tot_spend) * mean_incidence
    region_labels = _region_labels(data)
    zone_table = (pd.DataFrame({"ze2010": np.arange(1, R + 1),
                         "var_across_buyers": buyer_variance, "common_share": common_incidence})
           .merge(region_labels.rename(columns={"index": "ze2010", "ze2010": "ze_code",
                                      "ze2010_name": "ze_name"}), on="ze2010"))
    zone_table["share_of_M"] = zone_table["var_across_buyers"] / zone_table["var_across_buyers"].sum()
    zone_table = zone_table.sort_values("var_across_buyers", ascending=False)
    if verbose:
        print("  the zones behind the buyer-specific term (top 8)")
        print(zone_table.head(8)[["ze_name", "var_across_buyers", "share_of_M",
                           "common_share"]].round(5).to_string(index=False))
    return zone_table


# The two-buyer extension needs one more object, and it is the one nothing in the
# estimation pins down: `Q_rr's`, the probability that two buyers source the SAME
# VARIETY from the same commuting zone. Against the structural overlap
# `G_rr's = sum_l gamma_lrs gamma_lr's` (two buyers landing on the same zone through
# DIFFERENT varieties), positive dependence of win events is exactly `Q >= G`, and
#
#     E[omega_rs . omega_r's] = V_rr's Q_rr's + (1 - V_rr's) G_rr's .
#
# `Q` is the pairwise-intersection term that makes the model analytically intractable,
# reported here as an economic quantity: `rho_s`, the share of the granular
# concentration that is COMMON across buyers, is an intraclass correlation of win
# events.
#
# CAVEAT, stated rather than buried: the price-winner independence is exact WITHIN a
# buyer and not across buyers, so the two-buyer formula is an interpretation device.
# Every number below is computed from the draws; the formulas supply the reading and
# the two exact internal checks (`alpha = 0` must give `Q = 1`, `C = 1`, `rho = 1`).


def cosourcing(data, value_col="share", panel=None, spend=None, structural=None,
               verbose=True):
    """
    The two-buyer objects: `Q_rr's` (same VARIETY, same zone), `G_rr's` (same zone
    through different varieties) and `V_rr's` (expenditure alignment across varieties).

    `Q >= G` is positive dependence of win events across buyers, and the gap is what
    makes granularity COMMON rather than idiosyncratic: two buyers close enough to rank
    cells alike draw the same winner for the same variety far more often than the
    structural geography alone would give. `Q = 1` exactly when nothing is
    buyer-specific, which is the `alpha = 0` control.

    Returns one row per sector with the spend-weighted off-diagonal means, and stashes
    the full matrices in `.attrs["matrices"]`.
    """
    variety_panels = variety_panel(data, value_col) if panel is None else panel
    sector_spend = _sector_spend(data, value_col) if spend is None else spend
    networks = structural_networks(data, buyers=np.asarray(sector_spend.index).astype(int)) \
        if structural is None else structural

    rows, matrices = [], {}
    for s, panel_block in variety_panels.items():
        structural_block = networks["by_sector"].get(s)
        if structural_block is None:
            continue
        buyers = panel_block["buyers"]
        buyer_column = {int(b): i for i, b in enumerate(structural_block["buyers"])}
        take = [buyer_column[int(b)] for b in buyers if int(b) in buyer_column]
        if len(take) != buyers.size:
            continue
        gamma_rs = structural_block["W"][take]                                  # (buyer, cell) structural
        G = gamma_rs @ gamma_rs.T
        winners, replication = panel_block["winner"], panel_block["replication"]
        Q = np.zeros((buyers.size, buyers.size))
        complete_rows = np.isfinite(winners).all(axis=1)
        complete_winners = winners[complete_rows]
        Q = (complete_winners[:, :, None] == complete_winners[:, None, :]).mean(axis=0)
        V_pair = np.zeros_like(Q)
        v_values = np.nan_to_num(panel_block["v"])
        for b in np.unique(replication):                            # within a replication
            in_replication = replication == b
            V_pair += v_values[in_replication].T @ v_values[in_replication]
        V_pair /= np.unique(replication).size
        buyer_spend = sector_spend.iloc[:, s].reindex(buyers).fillna(0.0).to_numpy(dtype=float)
        if buyer_spend.sum() <= 0:
            continue
        buyer_weight = buyer_spend / buyer_spend.sum()
        off_diagonal = ~np.eye(buyers.size, dtype=bool)
        pair_weight = np.outer(buyer_weight, buyer_weight)
        pair_weight_off = pair_weight * off_diagonal
        rows.append({
            "sector": s, "sector_name": str(data["sector_names"][s]),
            "Q_off": float((pair_weight_off * Q).sum() / pair_weight_off.sum()),
            "G_off": float((pair_weight_off * G).sum() / pair_weight_off.sum()),
            "V_off": float((pair_weight_off * V_pair).sum() / pair_weight_off.sum()),
            "Q_minus_G": float((pair_weight_off * (Q - G)).sum() / pair_weight_off.sum()),
            "share_pairs_Q_ge_G": float((Q[off_diagonal] >= G[off_diagonal] - 1e-12).mean()),
        })
        matrices[s] = {"Q": Q, "G": G, "V": V_pair, "buyers": buyers, "pi": buyer_weight}
    sector_table = pd.DataFrame(rows).set_index("sector").sort_index()
    sector_table.attrs["matrices"] = matrices
    if verbose:
        print("  two buyers, one variety: Q (same variety, same zone) against G "
              "(same zone, different varieties)")
        print(sector_table.round(4).to_string())
        print(f"    Q >= G in {100*sector_table['share_pairs_Q_ge_G'].min():.0f}-"
              f"{100*sector_table['share_pairs_Q_ge_G'].max():.0f}% of buyer pairs per sector "
              "— positive dependence of win events, which is what makes granularity "
              "common rather than idiosyncratic.")
    return sector_table


def granular_cells(data, value_col="share", spend=None, panel=None, structural=None,
                   empirical="auto", verbose=True):
    """
    The four cells of `E[Hbar_s]` built from `V`, `Q` and `G` — the plan's §3.5 — and
    checked against the same four cells measured realisation by realisation.

        structural common          H(gamma_bar_s)                = pi' G pi
        structural buyer-specific  Hbar^gamma_s - H(gamma_bar_s)
        granular common            sum_rr' pi pi V_rr' (Q_rr' - G_rr')
        granular buyer-specific    sum_r pi_r V_r (1 - H^gamma_rr) - granular common

    and `rho_s`, the COMMON SHARE of the granular term — an intraclass correlation of
    win events across buyers, and the one quantity in this section that no targeted
    moment pins down.

    The two routes are algebraically the same object, so their agreement is a gate on
    both: the realisation route knows nothing about varieties, the variety route knows
    nothing about realised incidence vectors.
    """
    sector_spend = _sector_spend(data, value_col) if spend is None else spend
    variety_panels = variety_panel(data, value_col) if panel is None else panel
    networks = structural_networks(data, buyers=np.asarray(sector_spend.index).astype(int)) \
        if structural is None else structural
    cosourcing_table = cosourcing(data, value_col=value_col, panel=variety_panels, spend=sector_spend,
                    structural=networks, verbose=False)
    variety_table = variety_concentration(data, value_col=value_col, panel=variety_panels, spend=sector_spend,
                               verbose=False)
    matrices = cosourcing_table.attrs["matrices"]

    rows = []
    for s, sector_matrices in matrices.items():
        G, Q, V_pair, buyer_weight = sector_matrices["G"], sector_matrices["Q"], sector_matrices["V"], sector_matrices["pi"]
        pair_weight = np.outer(buyer_weight, buyer_weight)
        h_gamma = float(buyer_weight @ np.diag(G))                       # Hbar^gamma_s
        struct_common = float((pair_weight * G).sum())                             # H(gamma_bar_s)
        struct_specific = h_gamma - struct_common
        gran_common = float((pair_weight * V_pair * (Q - G)).sum())
        V_r = np.diag(V_pair)                                       # = E[sum_rho v_r^2]
        gran_total = float(buyer_weight @ (V_r * (1.0 - np.diag(G))))
        # the FLOOR of rho: with win events independent across buyers the off-diagonal
        # terms vanish (Q = G) and only the r = r' terms survive, so rho would still be
        # positive — a buyer always co-sources with itself, and that own term is part
        # of the industry's concentration. The floor is therefore the Herfindahl of the
        # BUYER spending distribution, reweighted by each buyer's granular term, and it
        # is what rho must be read against; rho - rho_floor is the part of common
        # granularity that comes from buyers agreeing rather than from one buyer being
        # large.
        floor = float((buyer_weight ** 2) @ (V_r * (1.0 - np.diag(G))))
        rows.append({"sector": s, "sector_name": str(data["sector_names"][s]),
                     "n_hat_s": variety_table["n_hat_s"].get(s, np.nan),
                     "V": variety_table["V"].get(s, np.nan),
                     "spend": float(sector_spend.iloc[:, s].sum()),
                     "struct_common": struct_common, "struct_specific": struct_specific,
                     "gran_common": gran_common, "gran_specific": gran_total - gran_common,
                     "gran_total": gran_total, "E_H_bar": h_gamma + gran_total,
                     "rho_s": gran_common / gran_total if gran_total > 0 else np.nan,
                     "rho_floor": floor / gran_total if gran_total > 0 else np.nan,
                     "buyer_hhi": float((buyer_weight ** 2).sum()),
                     "C_structural": struct_common / h_gamma if h_gamma > 0 else np.nan,
                     "C_realised": (struct_common + gran_common) / (h_gamma + gran_total) if h_gamma + gran_total > 0
                                   else np.nan})
    cells_table = pd.DataFrame(rows).set_index("sector").sort_index()
    cells_table["gran_share"] = cells_table["gran_total"] / cells_table["E_H_bar"]

    # the realisation-by-realisation split, which shares no code path with the above
    if isinstance(empirical, str) and empirical == "auto":
        try:
            empirical = granularity_concentration(data, value_col=value_col,
                                                  spend=sector_spend, verbose=False)
        except (ValueError, KeyError, FileNotFoundError):
            empirical = None
    if isinstance(empirical, pd.DataFrame):
        empirical_aligned = empirical.reindex(cells_table.index)
        gap_E_H = np.nanmax(np.abs(cells_table["E_H_bar"] - empirical_aligned["h_realised"])
                        / np.maximum(empirical_aligned["h_realised"], 1e-12))
        gap_rho = np.nanmax(np.abs(cells_table["gran_common"] - empirical_aligned["gran_common"])
                            / np.maximum(np.abs(empirical_aligned["gran_common"]), 1e-12))
        cells_table.attrs["check_E_H"] = float(gap_E_H)
        cells_table.attrs["check_gran_common"] = float(gap_rho)

    if verbose:
        print("  the four cells of E[Hbar_s], from varieties")
        columns_shown = ["sector_name", "n_hat_s", "V", "struct_common", "struct_specific",
                "gran_common", "gran_specific", "rho_s", "rho_floor", "C_structural",
                "C_realised"]
        print(cells_table[columns_shown].round(4).to_string())
        sector_weight = cells_table["spend"] / cells_table["spend"].sum()
        print(f"    spend-weighted: granular share of E[Hbar] "
              f"{float(sector_weight @ cells_table['gran_share']):.3f}, rho (common share of the granular "
              f"term) {float((sector_weight @ cells_table['gran_common']) / (sector_weight @ cells_table['gran_total'])):.3f} "
              f"against a floor of "
              f"{float((sector_weight @ (cells_table['rho_floor']*cells_table['gran_total'])) / (sector_weight @ cells_table['gran_total'])):.3f} "
              "(what buyer size alone would give with independent win events)")
        if "check_E_H" in cells_table.attrs:
            print(f"    against the realisation-by-realisation split: E[Hbar] agrees to "
                  f"{cells_table.attrs['check_E_H']:.2e}, the granular common cell to "
                  f"{cells_table.attrs['check_gran_common']:.2e} (two independent routes)")
    return cells_table


def granular_table(data, regimes=None, value_col="share", spend=None, verbose=True):
    """
    The section's Table 3: the structural pair for every regime, and the granular
    triple beside it.

    Every row comes from `simulate_granular_regime`, the baseline included, so the table
    is built by ONE route. It used to compute `Both forces` twice -- once off
    `suppliers.parquet`, once re-simulated -- and report the agreement, which was a weak
    cross-check between the two implementations. That check is gone and is replaced by a
    stronger one: `check_against_julia` compares the port against `solve_network` to the
    bit, given Julia's own draws. Run it after changing either implementation; the
    agreement this table used to display cannot be read off it any more.
    """
    sector_spend = _sector_spend(data, value_col) if spend is None else spend
    want = dict(CF_REGIMES if regimes is None else regimes)
    want.setdefault(UNIFORM_REGIME, dict(alpha=0.0, equalise_T=True))
    sector_table = sector_concentration(data, regimes=want, value_col=value_col, spend=sector_spend,
                               benchmark=True, verbose=False)
    summary_table = concentration_summary(sector_table, verbose=False)

    rows = {}
    for regime, regime_options in want.items():
        try:
            # the structural network MUST be the one of the same regime: G_rr' is the
            # benchmark the realised co-sourcing Q is read against, and comparing a
            # re-simulated alpha = 0 economy with the estimate's gamma would put rho
            # above one (measured, not hypothetical).
            variety_panels = simulate_granular_regime(data, value_col=value_col, spend=sector_spend, **regime_options)
            networks = structural_networks(data, buyers=np.asarray(sector_spend.index).astype(int),
                                       **regime_options)
            cells = granular_cells(data, value_col=value_col, spend=sector_spend, panel=variety_panels,
                                   structural=networks,
                                   empirical="auto" if regime == "Both forces" else None,
                                   verbose=False)
        except (ValueError, KeyError, FileNotFoundError) as error:
            print(f"  [granular] {regime} skipped: {type(error).__name__}: {error}")
            continue
        sector_weight = cells["spend"] / cells["spend"].sum()
        rows[regime] = {"n_eff_realised": 1.0 / float(sector_weight @ cells["E_H_bar"]),
                     "gran_share": float(sector_weight @ cells["gran_total"]) /
                                   float(sector_weight @ cells["E_H_bar"]),
                     "rho_s": float(sector_weight @ cells["gran_common"]) /
                              float(sector_weight @ cells["gran_total"]),
                     "rho_floor": float(sector_weight @ (cells["rho_floor"] * cells["gran_total"])) /
                                  float(sector_weight @ cells["gran_total"])}
    granular_triple = pd.DataFrame(rows).T
    table3 = pd.DataFrame({
        "n_eff_rel_uniform": summary_table["n_eff_ratio"],
        "C_structural": summary_table["C"],
    }).join(granular_triple)
    if verbose:
        print("  Table 3: structural pair, then the granular triple")
        print(table3.round(3).to_string())
        print("    the `No trade cost` row must carry C = 1 and rho = 1 "
              "exactly: with equal distances every buyer draws the same winner.")
    return table3


def plot_concentration_decomposition(tables, variety=None, figsize=None,
                                     save_to=None, order_by=None):
    """
    The four cells as 100% stacked bars, one bar per upstream sector, in the table
    ORDER as given, industries in separate panels. `order_by` (a column name) sorts
    instead; it is off by default -- ordering the rows by `N_s` makes the y axis read
    as a ranking of variety counts, which is not what the bars measure.

    The bars are shares of `E[Hbar_s]`, so each has length one and there is no scale to
    exaggerate; what the figure carries is WHERE the granular block sits — in the
    common row or the buyer-specific row — and whether that tracks `N_s` or the
    industry's geography. `variety` (a dict of `variety_concentration` frames) adds the
    measured EFFECTIVE NUMBER OF VARIETIES `1/V_s` beside `N_s` on a twin axis: the
    closed form would put it at `N_s/Xi(kappa_s)`, but `kappa_s = theta/(nu_s-1) = 2`
    at this calibration, where `Xi` diverges and the approximation has no content, so
    the measured value is the only one drawn.
    """
    items = list(tables.items()) if isinstance(tables, dict) else list(tables)
    segments = ["struct_common", "struct_specific", "gran_common", "gran_specific"]
    names = ["Structural, common", "Structural, buyer-specific",
             "Granular, common", "Granular, buyer-specific"]
    colours = [sim_color, (0.62, 0.72, 0.86), toulouse_color, (0.88, 0.70, 0.62)]

    n = len(items)
    rows = max(len(ordered_table) for _, ordered_table in items)
    mark_h, mark_l = [], []
    fig, axes = plt.subplots(1, n, figsize=figsize or (6.4 * n, max(3.0, 0.34 * rows)),
                             squeeze=False)
    for ax, (label, table) in zip(axes[0], items):
        ordered_table = table.copy()
        if order_by is not None and order_by in ordered_table.columns and ordered_table[order_by].notna().any():
            ordered_table = ordered_table.sort_values(order_by, ascending=True)
        segment_share = ordered_table[segments].div(ordered_table[segments].sum(axis=1), axis=0)
        y = np.arange(len(ordered_table))
        left = np.zeros(len(ordered_table))
        for segment, segment_name, colour in zip(segments, names, colours):
            bar_share = segment_share[segment].to_numpy(dtype=float)
            ax.barh(y, bar_share, left=left, height=0.72, label=segment_name, color=colour,
                    edgecolor="white", linewidth=0.4)
            left = left + bar_share
        ax.set_yticks(y)
        ax.set_yticklabels([str(r.sector_name) for r in ordered_table.itertuples()],
                           fontsize=name_axis_fontsize(ax, len(ordered_table), size_at_ref=8))
        ax.set_ylim(-0.6, len(ordered_table) - 0.4)
        ax.set_xlim(0, 1)
        ax.set_xlabel(r"Share of $E[\bar H_s]$")
        ax.grid(alpha=0.2, axis="x")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

        variety_table = None if variety is None else (variety.get(label)
                                           if isinstance(variety, dict) else None)
        twin_axis = ax.twiny()
        twin_axis.plot(ordered_table["n_hat_s"].to_numpy(dtype=float), y, "o", ms=4, color="0.35",
                label=r"$N_s$")
        if variety_table is not None:
            inverse_V = (1.0 / variety_table["V"]).reindex(ordered_table.index).to_numpy(dtype=float)
            twin_axis.plot(inverse_V, y, "D", ms=4, mfc="white", color="0.35",
                    label=r"$1/\mathcal{V}^v_s$")
        twin_axis.set_xlim(0, max(2.0, np.nanmax(ordered_table["n_hat_s"].to_numpy(dtype=float)) * 1.15))
        # The top band stacks, from the bars upward: twin ticks, the twin's own label,
        # then the panel title -- which is therefore set on the TWIN with an explicit
        # pad. Setting it on the primary axis puts it at the primary's own top, where
        # matplotlib does not see the twin's ticks or label, and the three collide.
        twin_axis.set_xlabel("Varieties per sector", fontsize=fs(8), color="0.35", labelpad=2)
        twin_axis.set_title(label, fontsize=fs(10), pad=26)
        twin_axis.tick_params(axis="x", colors="0.35", labelsize=fs(8), pad=1)
        for side in ("left", "right"):
            twin_axis.spines[side].set_visible(False)
        if ax is axes[0][0]:
            mark_h, mark_l = twin_axis.get_legend_handles_labels()
    fig.tight_layout()
    # ONE legend for the four segments, at FIGURE level above the titles. Per-axes it
    # has to sit inside or just above the panel, where it lands on the title and on
    # the twin label; and a legend describing bars common to every panel does not
    # belong to the first one. `bbox_inches="tight"` on the save keeps it in frame.
    # The marker series join it rather than keeping their own: the bars fill [0, 1] by
    # construction, so a legend inside the panel has no empty corner to sit in and
    # lands on the bottom rows' data whatever the `loc`.
    handles, legend_labels = axes[0][0].get_legend_handles_labels()
    handles, legend_labels = handles + mark_h, legend_labels + mark_l
    fig.legend(handles, legend_labels, frameon=False, fontsize=fs(8), ncol=min(len(handles), 3 * n),
               loc="lower center", bbox_to_anchor=(0.5, 1.0), borderaxespad=0.0,
               columnspacing=1.6, handlelength=1.4)
    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        fig.savefig(save_to, bbox_inches="tight")
    return axes


def plot_buyer_concentration(tables, regime="Both forces", sort_by="n_eff",
                             figsize=None, save_to=None):
    """
    The appendix figure: every buyer's effective number of destinations and its `C_r`,
    as two panels sharing one row order.

    `C_r` is drawn as a RANKING and not as a share — it is not bounded below, since a
    buyer can be further from the common incidence than its own concentration, and a
    bar from a true zero would invite it to be read as a percentage.
    """
    items = list(tables.items()) if isinstance(tables, dict) else list(tables)
    n = len(items)
    rows = max(len(buyer_table.xs(regime, level="regime")) for _, buyer_table in items)
    fig, axes = plt.subplots(1, 2 * n, figsize=figsize or (5.0 * n, max(3.0, 0.26 * rows)),
                             squeeze=False)
    for k, (label, table) in enumerate(items):
        buyer_table = table.xs(regime, level="regime").sort_values(sort_by)
        y = np.arange(len(buyer_table))
        for j, (column, x_label) in enumerate((("n_eff", "Effective destinations"),
                                         ("C", r"$C_r$ (common share, ranking)"))):
            ax = axes[0][2 * k + j]
            ax.plot(buyer_table[column].to_numpy(dtype=float), y, "o", ms=4,
                    color=sim_color if j == 0 else toulouse_color)
            ax.set_yticks(y)
            ax.set_yticklabels(buyer_table["region"].astype(str) if "region" in buyer_table else buyer_table.index,
                               fontsize=name_axis_fontsize(ax, len(buyer_table), size_at_ref=7))
            if j:
                ax.set_yticklabels([])
                ax.axvline(1.0, color="0.4", lw=0.8, ls="--")
            ax.set_xlabel(x_label)
            ax.set_ylim(-0.6, len(buyer_table) - 0.4)
            ax.grid(alpha=0.2, axis="x")
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        axes[0][2 * k].set_title(label, fontsize=fs(10), loc="left")
    fig.tight_layout()
    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        fig.savefig(save_to, bbox_inches="tight")
    return axes




# The two halves joined: one call per industry, in the order the paragraphs read.
def concentration_report(data, value_col="share", industry="", out_folder=None,
                         verbose=True):
    """
    Everything above for one industry, in the order the paragraphs read: the identity
    sector by sector (P3), the industry aggregate (P4), the force decomposition (P5),
    then granularity (P6) — the variety tail index, the effective number of varieties,
    the two-buyer co-sourcing objects, the four cells, and the section's table.

    Returns a dict, so the run cell can collect the pieces across industries for the
    joint figures without recomputing anything.
    """
    spend = _sector_spend(data, value_col)
    sector_table = sector_concentration(data, value_col=value_col, spend=spend, verbose=verbose)
    summary_table = concentration_summary(sector_table, verbose=verbose)
    derivatives = concentration_derivatives(data, value_col=value_col, spend=spend,
                                    verbose=verbose)
    buyers = buyer_concentration(sector_table, data=data)
    tail = variety_tail_index(data, verbose=verbose)
    variety_table = cosourcing_table = cells = cells_eq = cells_sz = table3 = None
    try:
        variety_panels = variety_panel(data, value_col)
        variety_table = variety_concentration(data, value_col=value_col, panel=variety_panels, spend=spend,
                                   verbose=verbose)
        cosourcing_table = cosourcing(data, value_col=value_col, panel=variety_panels, spend=spend,
                        verbose=verbose)
        cells = granular_cells(data, value_col=value_col, spend=spend, panel=variety_panels,
                               verbose=verbose)
        table3 = granular_table(data, value_col=value_col, spend=spend, verbose=verbose)
        # the same four cells with every buyer of a sector weighted equally. The
        # pipeline is untouched -- only `spend` changes -- so the two tables differ
        # for exactly one reason, the customer size distribution.
        cells_eq = granular_cells(data, value_col=value_col, panel=variety_panels,
                                  spend=equal_buyer_spend(spend), verbose=False)
        # and weighted by buyer SIZE, which `spend` does NOT carry: `share` is a share
        # of the buyer's own unit cost, so its column sums are near-uniform whatever
        # the buyer's size. `emp_pi_r` is where the size distribution lives.
        cells_sz = granular_cells(data, value_col=value_col, panel=variety_panels,
                                  spend=size_buyer_spend(spend, data), verbose=False)
    except (ValueError, KeyError, FileNotFoundError) as error:
        print(f"  granularity block skipped: {type(error).__name__}: {error}")
    buyer_granular = None
    try:
        buyer_granular = buyer_granular_concentration(data, value_col=value_col, spend=spend,
                                             verbose=verbose)
    except (ValueError, KeyError, FileNotFoundError) as error:
        print(f"  buyer-level granular block skipped: {type(error).__name__}: {error}")
    if buyer_granular is not None and verbose:
        try:
            report_granular_factors(buyer_granular)
        except (ValueError, KeyError) as error:
            print(f"  granular factor split skipped: {type(error).__name__}: {error}")
    reach = None
    try:
        reach = buyer_region_reach(data, value_col=value_col, spend=spend,
                                   verbose=verbose)
    except (ValueError, KeyError, FileNotFoundError) as error:
        print(f"  region-reach block skipped: {type(error).__name__}: {error}")
    zones = incidence_variance_map(data, value_col=value_col, spend=spend,
                                   verbose=verbose)
    return {"sector": sector_table, "summary": summary_table, "derivatives": derivatives, "buyers": buyers,
            "buyer_granular": buyer_granular, "reach": reach, "tail": tail, "variety": variety_table,
            "cosourcing": cosourcing_table, "granular": cells, "granular_equal": cells_eq,
            "granular_size": cells_sz, "table": table3, "zones": zones}




# ============================================================================
# The supplier's portfolio of customers
# ============================================================================



def portfolio_matrix(data, sector, weights="pi_r", alpha=None, equalise_T=False,
                     geometry=None):
    """
    Sector `sector`'s (cell x buyer) SALES matrix: row `j` is supplier cell `j`'s portfolio
    of customers, `x_{jr} = pi_r * gamma_{jrs}`.

    `weights` sets the buyer sizes: `"pi_r"` is the empirical `pi_r` target (the block-3
    moment, the only object carrying buyer size), `None` counts every buyer alike. The
    choice MATTERS here and the level of `pi_r` does not: the cosine below is invariant to
    scaling a whole portfolio, so the per-sector spending level cancels, but rescaling the
    BUYER axis reweights the coordinates and changes the angle between two portfolios.
    """
    sector_geometry = sourcing_geometry(data, alpha=alpha, equalise_T=equalise_T) if geometry is None else geometry
    sector_block = sector_geometry["by_sector"][sector]
    sales = np.asarray(sector_block["rho"], dtype=float)                 # (cell, buyer), columns sum to 1
    if weights is None:
        buyer_weight = np.ones(sales.shape[1])
    elif isinstance(weights, str) and weights == "pi_r":
        buyer_weight = np.asarray(_buyer_weights(data), dtype=float).ravel()
    else:
        buyer_weight = np.asarray(weights, dtype=float).ravel()
    if buyer_weight.size != sales.shape[1]:
        raise ValueError(f"buyer weights have {buyer_weight.size} entries against {sales.shape[1]} buyers.")
    return sales * buyer_weight[None, :], np.asarray(sector_block["cells"]).astype(int)


def portfolio_similarity(X):
    """
    The cosine similarity `C_jk = <x_j, x_k> / (|x_j| |x_k|)` between every pair of supplier
    portfolios, and the rows that carry one.

    The cosine is SCALE-INVARIANT, so sales shares and raw sales give the same `C_jk` and the
    normalisation is not a modelling choice. A cell with no sales has no portfolio and no
    angle; it is dropped rather than counted as similar to everything.
    """
    X = np.asarray(X, dtype=float)
    portfolio_norm = np.linalg.norm(X, axis=1)
    has_sales = np.isfinite(portfolio_norm) & (portfolio_norm > 0)
    unit_portfolios = X[has_sales] / portfolio_norm[has_sales][:, None]
    return np.clip(unit_portfolios @ unit_portfolios.T, -1.0, 1.0), has_sales


def portfolio_groups(C, tau):
    """
    Group labels from the similarity graph: a LINK when `C_jk > tau`, then the connected
    components of that graph.

    Two limitations, both real and neither hidden. `tau` is arbitrary -- the report below
    sweeps it rather than defending one value. And connected components CHAIN: `A ~ B` and
    `B ~ C` merge A with C even when `C_AC` is far below the threshold, so a long thin path
    is read as one clientele. A modularity algorithm (Louvain/Leiden on the weighted graph)
    does not chain and would be the fix if the groups themselves became the object; here
    they are scaffolding for the decomposition, and the chaining shows up as a single giant
    component at a low `tau`, which the reported group count makes visible.
    """
    C = np.asarray(C, dtype=float)
    n = C.shape[0]
    adjacency = (C > float(tau)) & ~np.eye(n, dtype=bool)
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in zip(*np.nonzero(np.triu(adjacency, 1))):
        root_i, root_j = find(int(i)), find(int(j))
        if root_i != root_j:
            parent[root_i] = root_j
    component = np.array([find(i) for i in range(n)])
    return np.unique(component, return_inverse=True)[1], adjacency


def portfolio_decomposition(C, labels):
    """
    `C_within`, `C_between` and their gap, over ORDERED pairs `j != k`.

    Pooled over groups, so the denominators are `sum_g N_g(N_g-1)` and `sum_{g!=h} N_g N_h`
    -- the ordered-pair counts, which is what makes the two averages comparable. A SINGLETON
    contributes nothing to the within sum (its own count is zero) and its whole row to the
    between one; with every cell a singleton the within average is undefined and returns
    NaN rather than zero.
    """
    C = np.asarray(C, dtype=float)
    group = np.asarray(labels).ravel()
    same = group[:, None] == group[None, :]
    off_diagonal = ~np.eye(group.size, dtype=bool)
    within_pairs, between_pairs = same & off_diagonal, (~same) & off_diagonal
    C_within = float(C[within_pairs].mean()) if within_pairs.any() else np.nan
    C_between = float(C[between_pairs].mean()) if between_pairs.any() else np.nan
    return C_within, C_between, C_within - C_between


def _rewire(A, rng, passes=10):
    """
    A degree-preserving rewiring of the similarity graph by double-edge swaps.

    This is the null the gap needs. The groups are built FROM `C`, so `C_within > C_between`
    is true by construction and the raw gap is a separation statistic, not a test. Holding
    each cell's NUMBER of links fixed and rearranging WHICH links it has gives a graph with
    the same degrees and no clientele structure; running the same partition on it and
    scoring it on the SAME `C` says how much of the observed gap the procedure would have
    produced anyway.
    """
    rewired = np.array(A, dtype=bool, copy=True)
    edges = np.array(np.nonzero(np.triu(rewired, 1))).T
    if edges.shape[0] < 2:
        return rewired
    for _ in range(int(passes) * edges.shape[0]):
        i, j = int(rng.integers(edges.shape[0])), int(rng.integers(edges.shape[0]))
        if i == j:
            continue
        a, b = edges[i]
        c, d = edges[j]
        if rng.random() < 0.5:
            c, d = d, c
        if len({int(a), int(b), int(c), int(d)}) < 4 or rewired[a, d] or rewired[c, b]:
            continue
        rewired[a, b] = rewired[b, a] = rewired[c, d] = rewired[d, c] = False
        rewired[a, d] = rewired[d, a] = rewired[c, b] = rewired[b, c] = True
        edges[i], edges[j] = (a, d), (c, b)
    return rewired


def portfolio_null(C, A, labels, n_draws=200, seed=0):
    """
    Two null distributions for `Delta C`, because the gap needs one and neither null
    alone is adequate.

    `rewire` is the null the question suggests: hold each cell's NUMBER of links fixed and
    rearrange WHICH links it has, then partition the result and score it on the SAME `C`.
    It has a defect worth reporting rather than hiding -- a rewired graph of any density is
    almost always CONNECTED, which leaves one group, no between pairs and no gap to score.
    Those draws return NaN and `null_degenerate` counts them; a rate near one means the
    rewiring null says nothing here, NOT that the observed gap is real.

    `permute` is the null that never degenerates: keep the observed group SIZES and
    reassign which cells fill them at random. It asks whether THIS partition of `C`
    separates more than an arbitrary partition of the same shape -- which is the question
    the endogeneity objection poses, and which a sector with one common clientele (every
    `C_jk` alike, so every partition equivalent) fails while a segmented one passes.
    """
    rng = np.random.default_rng(seed)
    group = np.asarray(labels).ravel()
    delta_rewire, delta_permute = [], []
    for _ in range(int(n_draws)):
        rewired = _rewire(A, rng)
        # B is 0/1, so any threshold strictly inside (0,1) means `linked`
        delta_rewire.append(portfolio_decomposition(C, portfolio_groups(rewired.astype(float), .5)[0])[2])
        delta_permute.append(portfolio_decomposition(C, rng.permutation(group))[2])
    return np.asarray(delta_rewire, dtype=float), np.asarray(delta_permute, dtype=float)


def customer_portfolio_report(data, taus=None, weights="pi_r", n_draws=200, seed=0,
                              verbose=True):
    """
    One row per (sector, tau): the similarity level, the groups it produces, the
    within/between split and the rewiring null.

    Read `delta` against `delta_null_mean`, never against zero.
    """
    taus = PORTFOLIO_TAUS if taus is None else taus
    geometry = sourcing_geometry(data)
    rows = []
    for s, sector_block in geometry["by_sector"].items():
        if np.asarray(sector_block["cells"]).size < 3:
            continue
        sales, cells = portfolio_matrix(data, s, weights=weights, geometry=geometry)
        C, has_sales = portfolio_similarity(sales)
        if C.shape[0] < 3:
            continue
        off_diagonal = ~np.eye(C.shape[0], dtype=bool)
        for tau in taus:
            group, adjacency = portfolio_groups(C, tau)
            C_within, C_between, delta = portfolio_decomposition(C, group)
            delta_rewire, delta_permute = portfolio_null(C, adjacency, group, n_draws=n_draws, seed=seed)
            sizes = np.bincount(group)
            rows.append({
                "sector": s, "sector_name": str(data["sector_names"][s]), "tau": float(tau),
                "n_cells": int(C.shape[0]), "n_dropped": int((~has_sales).sum()),
                "C_bar": float(C[off_diagonal].mean()),
                "n_groups": int(sizes.size), "n_singletons": int((sizes == 1).sum()),
                "largest_share": float(sizes.max() / sizes.sum()),
                "C_within": C_within, "C_between": C_between, "delta": delta,
                "delta_null_mean": float(np.nanmean(delta_permute)),
                "delta_null_sd": float(np.nanstd(delta_permute)),
                "delta_rewire_mean": float(np.nanmean(delta_rewire))
                                     if np.isfinite(delta_rewire).any() else np.nan,
                "null_degenerate": float(np.mean(~np.isfinite(delta_rewire))),
                "delta_excess": delta - float(np.nanmean(delta_permute)),
            })
    # One row per (sector, tau), so the index REPEATS each sector: `_by_sector_code`
    # reindexes and would raise on the duplicates. Order the blocks instead.
    if rows:
        portfolio_table = pd.DataFrame(rows).set_index(["sector", "tau"]).reset_index("tau")
        order = [s for s in _sectors_in_code_order(data) if s in set(portfolio_table.index)]
        portfolio_table = pd.concat([portfolio_table.loc[[s]] for s in order]) if order else portfolio_table
    else:
        portfolio_table = pd.DataFrame()
    if verbose and len(portfolio_table):
        columns_shown = ["sector_name", "tau", "n_cells", "C_bar", "n_groups", "n_singletons",
                "largest_share", "C_within", "C_between", "delta", "delta_null_mean",
                "null_degenerate", "delta_excess"]
        print("  supplier customer-portfolio similarity, by sector and threshold")
        print(portfolio_table[columns_shown].round(3).to_string())
        print("    `delta` is mechanically positive -- the groups are cut out of `C` "
              "itself. What carries content is `delta_excess`, the gap against a "
              "size-preserving permutation of the group labels. `null_degenerate` is "
              "the share of degree-preserving rewirings that collapsed to one "
              "component -- near one, that second null says nothing here.")
    return portfolio_table


def plot_portfolio_decomposition(tables, tau=None, figsize=None, save_to=None):
    """
    Per sector: `C_between` and `C_within` as the ends of a segment, with the rewiring
    null's `Delta C` drawn from `C_between` as a hollow mark.

    The segment IS the decomposition and the hollow mark is what the same procedure would
    have produced on a structureless graph of identical degrees, so the distance between
    the hollow and the filled mark is the whole of the finding.
    """
    items = list(tables.items()) if isinstance(tables, dict) else list(tables)
    n = len(items)
    rows = max(len(tau_rows[tau_rows["tau"] == (tau if tau is not None else tau_rows["tau"].max())])
               for _, tau_rows in items)
    fig, axes = plt.subplots(1, n, figsize=figsize or (5.6 * n, max(3.0, 0.34 * rows)),
                             squeeze=False)
    for ax, (label, table) in zip(axes[0], items):
        tau_rows = table[table["tau"] == (tau if tau is not None else table["tau"].max())]
        y = np.arange(len(tau_rows))
        C_between = tau_rows["C_between"].to_numpy(dtype=float)
        ax.hlines(y, C_between, tau_rows["C_within"].to_numpy(dtype=float), color="0.7", lw=2, zorder=1)
        ax.plot(C_between, y, "o", ms=5, color=sim_color, label=r"$C^{between}$", zorder=2)
        ax.plot(tau_rows["C_within"].to_numpy(dtype=float), y, "o", ms=5, color=toulouse_color,
                label=r"$C^{within}$", zorder=2)
        ax.plot(C_between + tau_rows["delta_null_mean"].to_numpy(dtype=float), y, "D", ms=5, mfc="white",
                color="0.35", label="rewiring null", zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels([str(r.sector_name) for r in tau_rows.itertuples()],
                           fontsize=name_axis_fontsize(ax, len(tau_rows), size_at_ref=8))
        ax.set_ylim(-0.6, len(tau_rows) - 0.4)
        ax.set_xlabel("cosine similarity of customer portfolios")
        ax.set_title(label, fontsize=fs(10))
        ax.grid(alpha=0.2, axis="x")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    handles, legend_labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, frameon=False, fontsize=fs(8), ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 1.0), borderaxespad=0.0)
    if save_to:
        fig.savefig(save_to, bbox_inches="tight")
    return axes


# ============================================================================
# The local share: a geographic level against a granular dispersion
# ============================================================================

# Test 9 -- the LOCAL SHARE, as a level plus a dispersion.
#
# The object of test 6 is `L_r(d)`, the share of the upstream response the shocked
# region keeps. It is LINEAR in the realised network, where the Herfindahl of test 5
# bis is quadratic, and that one difference changes what a figure of it can say.
#
# Linearity plus `E_Omega[omega_lrs] = gamma_lrs` gives, for ANY indicator `h_l` over
# the upstream cells,
#
#     E_Omega[ sum_l h_l omega_lrs ] = p_rs := sum_l h_l gamma_lrs    (exactly),
#     Var_Omega( sum_l h_l omega_lrs ) = V_rs * p_rs (1 - p_rs) ,
#
# the second from the same two-moment kernel that gives eq. (granular_buyer):
# `E[omega_l omega_m] = V gamma_l 1{l=m} + (1-V) gamma_l gamma_m`, so
# `Var(sum h omega) = V [ E_gamma(h^2) - (E_gamma h)^2 ] = V Var_gamma(h)`, and the
# variance of an indicator under `gamma` is `p(1-p)`.
#
# So there is nothing "structural against granular" to split in the LEVEL: the point
# IS the continuum benchmark, and the whole of granularity sits in the spread. The
# figure therefore opposes a level (geography) to a dispersion (chance) instead of
# cutting one quantity into shares.
#
# THE INDICATOR IS THE ONLY MODELLING CHOICE. `radius_km=None` is `1{l = r}`, the
# buyer's own commuting zone; a radius takes every cell within `d` km, which is test
# 6's own `L_r(d)`. The algebra above holds verbatim for either, so the radius is a
# reporting choice and not a second derivation.
#
# WHAT THE SECTOR SUM DOES. Sectors draw their varieties independently, so the
# variances add with SQUARED weights: `sd_r = sqrt(sum_s theta_rs^2 V_rs p_rs(1-p_rs))`.
# Aggregating a buyer's ten sectors therefore diversifies most of the chance away
# before it reaches the figure -- which is why the per-sector view is kept in
# `.attrs["by_sector"]` and the plot can draw one sector on its own.

def _local_indicator(sector_block, radius_km=None):
    """
    The 0/1 `(cell x buyer)` map of what counts as LOCAL for each buyer.

    `radius_km=None` gives `1{l = r}` -- the buyer's own zone and nothing else, the
    `d -> 0+` limit of `L_r(d)`. A radius gives `1{d_lr <= radius}`, test 6's object.
    """
    cells = np.asarray(sector_block["cells"]).astype(int)            # 0-based ZE
    buyers = np.asarray(sector_block["buyers"]).astype(int)          # 1-based ZE
    if radius_km is None:
        return (cells[:, None] + 1) == buyers[None, :]
    distance = np.asarray(sector_block["distance"], dtype=float)
    if distance.shape != (cells.size, buyers.size):
        raise ValueError(f"distance is {distance.shape}, expected {(cells.size, buyers.size)} "
                         "— `structural_networks` keeps it (cell x buyer).")
    return distance <= float(radius_km)


def _realised_local(panel_block, mask, cells, buyers):
    """
    The realised local share `omega_{B,rs}(Omega)` one draw at a time: a
    `(replication x buyer)` frame read off a variety panel.

    A replication's local share is the expenditure weight of the varieties whose
    WINNER falls inside the buyer's own set, `sum_j v_jrs 1{winner_jrs in B_r}`, so
    the panel's two columns are used exactly as they are and nothing is re-derived
    from `gamma`. This is the route that shares no algebra with the closed form above,
    which is what makes the two comparable.
    """
    pan_buyers = np.asarray(panel_block["buyers"]).astype(int)
    missing_buyers = [int(b) for b in buyers if b not in set(pan_buyers.tolist())]
    if missing_buyers:
        raise KeyError(f"buyers {missing_buyers[:5]} are absent from the variety panel.")
    buyer_column = np.array([int(np.flatnonzero(pan_buyers == b)[0]) for b in buyers])

    winners = np.asarray(panel_block["winner"], dtype=float)[:, buyer_column]
    v_values = np.nan_to_num(np.asarray(panel_block["v"], dtype=float))[:, buyer_column]
    cells = np.asarray(cells).astype(int)
    max_zone = int(np.nanmax(winners)) if np.isfinite(winners).any() else 0
    cell_row_of_zone = np.full(max(int(cells.max()) + 2, max_zone + 2), -1)      # zone code -> cell row
    cell_row_of_zone[cells + 1] = np.arange(cells.size)
    cell_row = cell_row_of_zone[np.nan_to_num(winners, nan=0.0).astype(int)]
    is_local = np.where(cell_row >= 0,
                   np.asarray(mask)[np.clip(cell_row, 0, None),
                                    np.arange(buyers.size)[None, :]], False)
    local_share = pd.DataFrame(v_values * is_local, columns=np.asarray(buyers).astype(int))
    return local_share.groupby(np.asarray(panel_block["replication"])).sum()


LOCAL_SHARE_RADIUS_KM = AMPLIFICATION_RADII[-1]   # 200 km, the section's own radius


def local_share_dispersion(data, regimes=None, value_col="share", spend=None,
                           panel=None, radius_km=LOCAL_SHARE_RADIUS_KM,
                           baseline="Both forces", verbose=True):
    """
    Per (regime, buyer): the local share of the whole portfolio as a level, and the
    range of the economies that realise it.

    The reported object is GRANULARITY, not estimation uncertainty. Every replication
    is drawn at the SAME `theta+`, so the band answers "at these fundamentals, how much
    can a realised year differ from its expectation" — a statement about the economy.
    It carries nothing of the uncertainty in `(alpha_hat, T_hat, N_hat)`, so it is not
    a confidence interval and must not be captioned as one.

    Columns
    -------
    local        `sum_s theta_rs p_rs`, the continuum benchmark — exact, no granularity.
    q10, q90     the empirical 10th and 90th PERCENTILES of the realised local share
                 across replications: THE OBJECT, and what the figure draws. A standard
                 deviation would describe this distribution badly — when `p_r x N_eff_r`
                 is small the law is DISCRETE and skewed (a buyer at 5% with five
                 effective varieties realises 0 or 0.2, never 0.05), so a symmetric bar
                 implies a shape the data do not have. Quantiles impose none.
    coverage     the share of realisations that actually fall inside `[q10, q90]`. On a
                 continuous law this is 0.80 and the band means what it says. On a
                 LATTICE it is not: the realised share lives on `{0, 1/N_eff, 2/N_eff,
                 ...}`, so the interval contains AT LEAST 80% and can reach 1.0 when
                 there are few atoms, while its width jumps by whole lattice steps and
                 can collapse to zero when one atom carries more than 80% of the mass.
                 A coverage near 0.80 licenses the "8 realisations in 10" reading; well
                 above it says the width is an artefact of the variety count. This is
                 the one number that tests the band's own claim.
    p_n_eff      `local x n_eff_var` with `n_eff_var = 1/V_r`: the expected COUNT of
                 effective varieties landing locally, i.e. how fine that lattice is.
                 Read it before the band — below a few, the band is wide and skewed for
                 a counting reason and not a geographic one.
    z            `E[omega] = gamma` is EXACT, so the gap between the realised mean
                 (`local_draws`) and `local` is pure simulation error, scored against
                 `sd_draws/sqrt(n_draws)`. A `z` far from zero means the variety panel
                 and the structural network are not the same economy — i.e. the draws
                 are not draws of the right regime, which no other column would reveal.

    `.attrs["by_sector"]` carries `theta`, `p` and `V` BEFORE the sector aggregation,
    which is what `local` and `V_r` are built from.

    ONE CHOICE IS MADE SILENTLY BY THE FORMULA AND IS STATED HERE. The input mix
    `theta_rs` is held FIXED at its mean across replications, on both routes. In the
    realised economy it moves with the draw too — `share` is `exp_val`, which carries
    the realised prices — so the reported dispersion is the NETWORK's alone and not the
    network's plus the input mix's. That is what the closed form describes, and holding
    it fixed is what makes the two routes comparable; a figure of the total variation of
    the local share would be a different object and a larger one.
    """
    sector_spend = _sector_spend(data, value_col) if spend is None else spend
    buyers = np.asarray(sector_spend.index).astype(int)
    buyer_total = sector_spend.sum(axis=1)
    theta = sector_spend.div(buyer_total.where(buyer_total > 0), axis=0)          # (buyer x sector), rows sum to 1
    want = dict(CF_REGIMES if regimes is None else regimes)
    if baseline not in want:
        raise KeyError(f"baseline {baseline!r} is not among {list(want)}.")
    names = _region_labels(data).set_index("index")["ze2010_name"]

    # every panel first, so the aggregation below reads them rather than interleaving
    # simulation with it
    panels, V_by_regime = {}, {}
    for regime, regime_options in want.items():
        try:
            # Every regime comes from the SAME forward map, the estimated economy
            # included: a counterfactual is then not a different KIND of object from the
            # baseline. `panel=` still overrides the baseline, which is how a caller
            # hands in a panel read off Julia's parquet.
            if panel is not None and not regime_options:
                regime_panel = panel
            else:
                regime_panel = simulate_granular_regime(data, value_col=value_col, spend=sector_spend, **regime_options)
        except (ValueError, KeyError, FileNotFoundError) as error:
            print(f"  [local share] {regime}: no variety panel "
                  f"({type(error).__name__}: {error}) — the level only.")
            regime_panel = None
        panels[regime], V_by_regime[regime] = regime_panel, (None if regime_panel is None else _V_by_buyer(regime_panel, buyers))

    n_buyers = buyers.size
    blocks, by_sector = [], []
    for regime, regime_options in want.items():
        networks = structural_networks(data, buyers=buyers, **regime_options)
        regime_panel, V = panels[regime], V_by_regime[regime]
        local_level = np.zeros(n_buyers)
        V_r = np.zeros(n_buyers)
        parts, dropped = [], []
        for s, sector_block in networks["by_sector"].items():
            theta_rs = theta.iloc[:, s].to_numpy(dtype=float)
            mask = _local_indicator(sector_block, radius_km)                  # (cell x buyer)
            p_rs = (np.asarray(sector_block["W"], dtype=float) * mask.T).sum(axis=1)
            v_rs = (V.loc[s].to_numpy(dtype=float)
                    if V is not None and s in V.index else np.full(n_buyers, np.nan))
            local_level += theta_rs * p_rs
            V_r += theta_rs * v_rs

            drawn_local_s = None
            if regime_panel is not None and s in regime_panel:
                drawn_local_s = _realised_local(regime_panel[s], mask, sector_block["cells"], buyers)
                parts.append(drawn_local_s.mul(pd.Series(theta_rs, index=buyers), axis=1))
            elif regime_panel is not None:
                dropped.append(s)
            # `sd_draws` is the only draw-derived column kept here: it is what pins
            # the p(1-p) shape on MEASURED data rather than on the closed form, where
            # it would be a tautology. The per-sector quantiles went with the
            # per-sector figure.
            by_sector.append(pd.DataFrame({
                "regime": regime, "sector": s, "ze2010_downstream": buyers,
                "theta": theta_rs, "p": p_rs, "V": v_rs,
                "sd_draws": (drawn_local_s.std(axis=0, ddof=1).reindex(buyers).to_numpy()
                             if drawn_local_s is not None else np.nan)}))

        if dropped:
            # a sector in the geometry but not in the panel would leave the draws
            # measuring a DIFFERENT input mix from the closed form, silently. Blank
            # them instead: pandas would have summed the missing sector as a zero.
            print(f"  [local share] {regime}: sectors {dropped[:5]} carry no varieties — "
                  "the AGGREGATE measured dispersion is dropped for this regime; the "
                  "per-sector rows that do have a panel are kept.")
            parts = []
        if parts:
            common_replications = parts[0].index
            for part in parts[1:]:
                common_replications = common_replications.intersection(part.index)
            drawn_local = sum(part.reindex(index=common_replications) for part in parts)
            q10_s, q90_s = drawn_local.quantile(0.10), drawn_local.quantile(0.90)
            # the share of realisations the band actually holds. On a lattice the 10-90
            # interval contains AT LEAST 80% and can hold all of it, so this is what
            # separates "8 realisations in 10" from an artefact of the variety count.
            coverage_s = (drawn_local.ge(q10_s, axis=1) & drawn_local.le(q90_s, axis=1)).mean(axis=0)
            sd_draws = drawn_local.std(axis=0, ddof=1).reindex(buyers).to_numpy()
            mean_draws = drawn_local.mean(axis=0).reindex(buyers).to_numpy()
            q10 = q10_s.reindex(buyers).to_numpy()
            q90 = q90_s.reindex(buyers).to_numpy()
            coverage = coverage_s.reindex(buyers).to_numpy()
            n_draws = float(len(common_replications))
        else:
            sd_draws = mean_draws = q10 = q90 = coverage = np.full(n_buyers, np.nan)
            n_draws = np.nan

        blocks.append(pd.DataFrame({
            "regime": regime, "ze2010_downstream": buyers, "local": local_level,
            "q10": q10, "q90": q90, "coverage": coverage,
            "sd_draws": sd_draws, "local_draws": mean_draws, "n_draws": n_draws, "V": V_r,
            "spend": buyer_total.reindex(buyers).to_numpy()}))

    local_table = pd.concat(blocks, ignore_index=True).set_index(["regime", "ze2010_downstream"])
    local_table["width"] = local_table["q90"] - local_table["q10"]
    # how many effective varieties land locally. `p x N_eff` of order one is the regime
    # in which the realised share is a handful of discrete events, so the band is wide
    # and skewed for a counting reason and not a geographic one.
    local_table["n_eff_var"] = 1.0 / local_table["V"].where(local_table["V"] > 0)
    local_table["p_n_eff"] = local_table["local"] * local_table["n_eff_var"]
    # E[omega] = gamma is EXACT, so this is a pure simulation-error score and a free
    # check that the panel and the structural network describe the same regime.
    standard_error = local_table["sd_draws"] / np.sqrt(local_table["n_draws"])
    local_table["z"] = (local_table["local_draws"] - local_table["local"]) / standard_error.where(standard_error > 0)
    local_table["region"] = names.reindex(
        local_table.index.get_level_values("ze2010_downstream")).to_numpy()
    by_sector_table = pd.concat(by_sector, ignore_index=True)
    by_sector_table["sector_name"] = [str(data["sector_names"][int(s)]) for s in by_sector_table["sector"]]
    local_table.attrs["by_sector"] = by_sector_table
    local_table.attrs["radius_km"] = radius_km
    local_table.attrs["baseline"] = baseline

    if verbose:
        what = ("the buyer's OWN commuting zone" if radius_km is None
                else f"within {radius_km:g} km of the buyer")
        print(f"  the local share of the portfolio — {what}")
        summary_table = local_table.groupby(level="regime").apply(lambda d: pd.Series({
            "local (median)": d["local"].median(),
            "local (min)": d["local"].min(), "local (max)": d["local"].max(),
            "q10 (median)": d["q10"].median(), "q90 (median)": d["q90"].median(),
            "width (median)": d["width"].median(),
            "coverage (median)": d["coverage"].median(),
            "draws": d["n_draws"].iloc[0]}))
        print(summary_table.round(4).to_string())
        # the counting regime, which decides whether the band is wide for a granular
        # reason or simply because a handful of events are being counted
        baseline_rows = local_table.loc[baseline]
        print(f"    p x N_eff (the effective varieties landing locally): median "
              f"{baseline_rows['p_n_eff'].median():.1f}, min {baseline_rows['p_n_eff'].min():.1f}, "
              f"max {baseline_rows['p_n_eff'].max():.1f} at N_eff median "
              f"{baseline_rows['n_eff_var'].median():.1f}"
              + ("  <-- of order one: the realised share is a few discrete events, so "
                 "the band is skewed for a counting reason"
                 if baseline_rows['p_n_eff'].min() < 3 else ""))
        abs_z = local_table["z"].abs()
        if abs_z.notna().any():
            print(f"    E[omega] = gamma check: |z| median {abs_z.median():.2f}, "
                  f"max {abs_z.max():.2f} over {int(local_table['n_draws'].max())} draws"
                  + ("  <-- the panel and the structural network disagree"
                     if abs_z.max() > 4 else ""))
        coverage_column = local_table["coverage"]
        if coverage_column.notna().any():
            print(f"    coverage of [q10, q90]: median {coverage_column.median():.3f}, "
                  f"max {coverage_column.max():.3f}"
                  + ("  <-- well above 0.80: few atoms, so the interval over-covers "
                     "and its width is a lattice step rather than a dispersion"
                     if coverage_column.median() > 0.85 else
                     "  — the band holds 80% of the realised economies, as written"))
    return local_table


def _regime_order(table, baseline):
    """Regimes in the order the table was BUILT, baseline first. `unstack` sorts the
    level alphabetically, which would reorder the figure's series and the report's rows
    behind the caller's back."""
    seen = list(dict.fromkeys(table.index.get_level_values("regime")))
    if baseline not in seen:
        raise KeyError(f"baseline {baseline!r} not among {seen}.")
    return [baseline] + [r for r in seen if r != baseline]


def local_share_report(table, baseline=None, verbose=True):
    """
    What the band says, and whether it says it.

    Two readings of the band itself, each measured rather than asserted:

    (1) `p(1-p)` is maximal at `p = 1/2` and vanishes at both ends, so the buyers whose
        local sourcing is INTERMEDIATE are the ones whose realised local share says
        least. Below one half the band WIDENS with the point, above it it narrows — so
        which holds is an empirical question about where the local shares sit, and the
        share of buyers past one half is reported before anything is read into a
        movement.

    (2) Whether cutting a force lowers the point and WIDENS the band — chance filling
        the dispersion geography vacates. That needs `p` pushed TOWARDS one half; at
        `p` well below it the two move together and the compensation cannot operate on
        this statistic. The count of buyers moving each way is reported.

    Then the two guards, which are the conditions under which (1) and (2) mean
    anything: `coverage` (does the 10-90 interval hold 80% of the realisations, or is
    it over-covering a lattice) and `z` (are the realisations draws of the right
    economy at all).
    """
    baseline_regime = table.attrs.get("baseline", "Both forces") if baseline is None else baseline
    keep = _regime_order(table, baseline_regime)
    local_point = table["local"].unstack("regime")[keep]
    band_width = table["width"].unstack("regime")[keep]

    n = len(local_point)
    above = int((local_point[baseline_regime] > 0.5).sum())
    rows = []
    for regime in local_point.columns:
        change_local, change_width = local_point[regime] - local_point[baseline_regime], band_width[regime] - band_width[baseline_regime]
        rows.append({"regime": regime, "local (median)": local_point[regime].median(),
                     "width (median)": band_width[regime].median(),
                     "coverage (median)": table.loc[regime, "coverage"].median(),
                     "d local (median)": change_local.median(),
                     "d width (median)": change_width.median(),
                     "n: point down, band WIDER": int(((change_local < 0) & (change_width > 0)).sum()),
                     "n: both down": int(((change_local < 0) & (change_width < 0)).sum()),
                     "buyers past 1/2": int((local_point[regime] > 0.5).sum())})
    report_table = pd.DataFrame(rows).set_index("regime")
    if verbose:
        print("\n  the level against the band, regime by regime")
        print(report_table.round(4).to_string())
        baseline_rows = table.loc[baseline_regime]
        worst = baseline_rows["width"].idxmax() if baseline_rows["width"].notna().any() else local_point[baseline_regime].idxmax()
        lo, hi = float(baseline_rows.loc[worst, "q10"]), float(baseline_rows.loc[worst, "q90"])
        widest_name = table["region"].groupby(level="ze2010_downstream").first().get(worst, worst)
        print(f"\n    (1) the widest band is {widest_name}: {local_point[baseline_regime].loc[worst]:.3f} expected, "
              f"and 8 economies in 10 realise a share in [{lo:.3f}, {hi:.3f}] — the "
              "same fundamentals, a different year.")
        print(f"        {above} of {n} buyers sit above p = 1/2 in {baseline_regime}, so the band "
              + ("WIDENS with the point for essentially every buyer: a force that "
                 "lowers the local share lowers its dispersion too, and the "
                 "compensation the Herfindahl shows cannot operate here."
                 if above == 0 else
                 "is on both sides of its maximum and the two movements must be read "
                 "buyer by buyer."))
        for regime in local_point.columns:
            if regime == baseline_regime:
                continue
            regime_row = report_table.loc[regime]
            print(f"        (2) {regime}: {int(regime_row['n: point down, band WIDER'])} of {n} "
                  f"buyers lose local share AND gain dispersion, "
                  f"{int(regime_row['n: both down'])} lose both.")
        # --- the two guards ---------------------------------------------------
        coverage, p_n_eff = baseline_rows["coverage"], baseline_rows["p_n_eff"]
        print(f"\n    [guard] coverage: median {coverage.median():.3f}, max "
              f"{coverage.max():.3f} over {int(baseline_rows['n_draws'].iloc[0])} economies. "
              + ("the band holds 80% of the realisations, so it reads as written."
                 if coverage.median() < 0.85 else
                 "WELL ABOVE 0.80: the realised share is a lattice with few atoms, so "
                 "the interval over-covers and its width is a step of that lattice "
                 "rather than a dispersion — read p x N_eff before quoting a width."))
        print(f"    [guard] p x N_eff: {p_n_eff.median():.1f} at the median buyer, "
              f"{p_n_eff.min():.1f} at the lowest"
              + (" — fine enough for the band to read as a continuous range."
                 if p_n_eff.min() >= 3 else
                 " — BELOW 3 somewhere: there the realised share is a count of a "
                 "handful of events and the band is discrete."))
        abs_z = baseline_rows["z"].abs()
        print(f"    [guard] |z|: max {abs_z.max():.2f}"
              + (" — the realisations and the structural network are the same economy."
                 if abs_z.max() < 3 else
                 " — ABOVE 3: the variety panel and the structural network disagree, "
                 "so the band is drawn around the wrong point. Check theta and re-run "
                 "check_against_julia before reading anything below."))
    return report_table


def plot_local_share_dispersion(table, baseline=None, regimes=None, order=None,
                                orientation="h", layout="panels", figsize=None,
                                save_to=None):
    """
    The figure: one point per (buyer, regime) at the continuum level, with a band
    carrying the granularity. Buyers ordered by their local share in `baseline`, so the
    same order holds across regimes and who loses their local anchoring is read off
    directly. The axis is LINEAR and in points of local share — the object is the level,
    not a ratio, so there is nothing a log axis would straighten.

    `layout` is the one decision that changes how the figure reads, and the default
    moved for a reason. `"overlay"` puts all three regimes in ONE panel, offset within
    each buyer row: that is three marks and three bands per row, and with a band whose
    arms are of the order of the between-regime movement the marks interleave and the
    reader has to disentangle colour from vertical position. `"panels"` (the default)
    draws one INDEPENDENT FIGURE per regime — its own figure, its own file, its own
    buyer names — so a regime can be dropped into a paper or a slide on its own instead
    of carrying the other two with it. What still ties them together is the `x` WINDOW,
    pinned once across regimes from the widest band drawn, and the buyer ORDER, taken
    from `baseline`: without those two a figure read beside another would compare
    nothing. Each counterfactual figure carries the BASELINE point as a faint open mark,
    so the displacement is legible in the figure itself rather than only across figures;
    the baseline figure carries no such ghost (it would be the same mark twice).

    The bar is the empirical 10-90 range of the realised local share, drawn
    ASYMMETRICALLY about the point, and there is no other option: where `p x N_eff` is
    of order one the realised share is a count of a few events, so its law is discrete
    and skewed and a symmetric `+/- sd` draws a shape the data do not have. Read
    `coverage` in the table beside it — on a lattice the interval holds more than 80%
    and its width is a step of that lattice.

    `orientation="h"` puts the buyers on the y axis (commuting-zone names are long, and
    this is the form every other per-buyer figure of the section uses); `"v"` puts them
    on the x axis.

    Under `"panels"` the X AXIS IS NOT NAMED — neither the quantity (orientation `"h"`)
    nor the commuting zone (`"v"`). The quantity is what the caption has to state anyway,
    together with what the point and the bar ARE, and stating it on the axis of every one
    of three separate figures makes the reader parse the same legend three times; the
    zone axis is named by its own tick labels. The caption says "point: the continuum
    expectation; bar: 10-90% of realised economies", never "confidence interval": the
    replications share one `theta+`, so the band is granularity and carries no estimation
    uncertainty. The `"overlay"` layout is one exhibit and keeps its label.

    `"pair"` is the two-exhibit form, and the one for a reader comparing forces: (1) the
    estimated economy alone, in black, point and band; (2) every counterfactual on ONE
    figure, each in its own sub-row of the buyer's row with its band, and the baseline
    repeated in each sub-row as an open black ring WITHOUT a bar. The ring sits on the
    same line as the counterfactual it is read against, so each displacement is a
    horizontal distance and needs no cross-row saccade; its band is in figure (1) and
    drawing it again would double the ink on the reference. Buyer rows are separated by
    alternate shading so the sub-rows read as one buyer. Both figures share the buyer
    order (baseline local share) and the x window.

    `save_to` is a file under `"overlay"` and a STEM under `"panels"` and `"pair"`
    (`_baseline` / `_counterfactuals` appended under `"pair"`): under `"panels"` the regime is
    appended before the extension, so `.../local_share_dispersion_auto.pdf` writes
    `..._both_forces.pdf`, `..._equal_comparative_advantage.pdf`, ... One call still
    produces the whole set and nothing overwrites anything.

    Returns the `Axes` under `"overlay"`, `{regime: Axes}` under `"panels"` and
    `{"baseline": Axes, "counterfactuals": Axes}` under `"pair"`.
    """
    if orientation not in ("h", "v"):
        raise ValueError(f"orientation must be 'h' or 'v', got {orientation!r}.")
    if layout not in ("panels", "overlay", "pair"):
        raise ValueError(f"layout must be 'panels', 'overlay' or 'pair', got {layout!r}.")
    if layout == "pair" and orientation != "h":
        raise ValueError("layout='pair' is drawn with buyers on the y axis (orientation='h').")
    baseline_regime = table.attrs.get("baseline", "Both forces") if baseline is None else baseline

    keep = _regime_order(table, baseline_regime)
    local_point = table["local"].unstack("regime")[keep]
    q10_table = table["q10"].unstack("regime")[keep]
    q90_table = table["q90"].unstack("regime")[keep]
    if regimes is not None:
        columns_kept = list(regimes)
        if baseline_regime not in columns_kept:
            raise KeyError(f"baseline {baseline_regime!r} not among {columns_kept}.")
        local_point, q10_table, q90_table = local_point[columns_kept], q10_table[columns_kept], q90_table[columns_kept]
    row_order = local_point[baseline_regime].sort_values().index if order is None else pd.Index(order)
    local_point, q10_table, q90_table = local_point.reindex(row_order), q10_table.reindex(row_order), q90_table.reindex(row_order)
    names = (table["region"].groupby(level="ze2010_downstream").first()
             .reindex(row_order).astype(str))
    labels = list(local_point.columns)

    n_buyers, n_regimes = len(row_order), len(labels)
    row_position = np.arange(n_buyers)
    # the band is drawn about the POINT, which is the expectation; a skewed law can put
    # a quantile on the far fside of it, so the arm is clipped at zero rather than handed
    # to matplotlib as a negative length.
    def _arms(regime):
        point = local_point[regime].to_numpy(dtype=float)
        return point, np.vstack([np.maximum(point - q10_table[regime].to_numpy(dtype=float), 0.0),
                             np.maximum(q90_table[regime].to_numpy(dtype=float) - point, 0.0)])

    radius = table.attrs.get("radius_km")
    value_label = ("Local share of the upstream response — own zone" if radius is None
             else f"Local share of the upstream response — within {radius:g} km")
    hi_max = float(np.nanmax(q90_table.to_numpy(dtype=float)))
    x_window = (0.0, hi_max * 1.05 if np.isfinite(hi_max) and hi_max > 0 else 1.0)

    def _dress(ax, first, name_zone=True):
        # `first` governs whether the buyer names are shown, and they are hidden with
        # `tick_params` rather than by setting empty tick LABELS: an overlay-style shared
        # axis would carry a formatter set on one axes to the whole group. Under
        # `"panels"` every figure is independent and names its own buyers.
        if orientation == "h":
            ax.set_yticks(row_position); ax.set_yticklabels(names)
            set_name_axis_fontsize(ax, n_buyers)
            ax.tick_params(axis="y", labelleft=first)
            ax.set_ylim(-0.6, n_buyers - 0.4); ax.set_xlim(*x_window)
            ax.grid(alpha=0.2, axis="x")
        else:
            ax.set_xticks(row_position); ax.set_xticklabels(names, rotation=90)
            set_name_axis_fontsize(ax, n_buyers, axis="x")
            ax.tick_params(axis="x", labelbottom=first)
            ax.set_xlim(-0.6, n_buyers - 0.4); ax.set_ylim(*x_window)
            ax.grid(alpha=0.2, axis="y")

    def _draw(ax, regime, ghost):
        colour = CF_COLORS.get(regime, toulouse_color)
        if ghost is not None:
            baseline_point = local_point[ghost].to_numpy(dtype=float)
            if orientation == "h":
                ax.plot(baseline_point, row_position, "o", mfc="none", mec="0.6", ms=4.0, lw=0,
                        zorder=1, label=ghost)
            else:
                ax.plot(row_position, baseline_point, "o", mfc="none", mec="0.6", ms=4.0, lw=0,
                        zorder=1, label=ghost)
        point, arms = _arms(regime)
        errorbar_options = dict(fmt="o", markersize=4.0, color=colour, ecolor=colour, elinewidth=1.2,
                  capsize=0.0, linestyle="none", zorder=3, label=regime)
        if orientation == "h":
            ax.errorbar(point, row_position, xerr=arms, **errorbar_options)
        else:
            ax.errorbar(row_position, point, yerr=arms, **errorbar_options)

    if layout == "overlay":
        row_step = 0.8 / max(n_regimes, 1)
        if figsize is None:
            figsize = (9, max(3.5, 0.10 * n_regimes * n_buyers)) if orientation == "h" \
                else (max(7.0, 0.55 * n_buyers), 5.0)
        fig, ax = plt.subplots(figsize=figsize)
        for j, regime in enumerate(labels):
            offset = ((n_regimes - 1) / 2 - j) * row_step
            colour = CF_COLORS.get(regime, toulouse_color)
            point, arms = _arms(regime)
            errorbar_options = dict(fmt="o", markersize=4.5, color=colour, ecolor=colour, elinewidth=1.4,
                      capsize=2.5, linestyle="none", label=regime)
            if orientation == "h":
                ax.errorbar(point, row_position + offset, xerr=arms, **errorbar_options)
            else:
                ax.errorbar(row_position + offset, point, yerr=arms, **errorbar_options)
        _dress(ax, True)
        (ax.set_xlabel if orientation == "h" else ax.set_ylabel)(value_label)
        ax.legend(frameon=False, fontsize=fs(8), loc="best")
        fig.tight_layout()
        if save_to:
            fig.savefig(save_to, bbox_inches="tight")
        return ax

    if layout == "pair":
        stem, extension = os.path.splitext(save_to) if save_to else (None, ".pdf")
        extension = extension or ".pdf"
        counterfactuals = [candidate for candidate in labels if candidate != baseline_regime]
        n_counterfactuals = len(counterfactuals)
        baseline_height = max(3.5, 0.22 * n_buyers)
        figure_width = 5.6 if figsize is None else figsize[0]
        axes_pair = {}

        def _save(fig, file_tag):
            if stem:
                path = f"{stem}_{file_tag}{extension}"
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                fig.savefig(path, bbox_inches="tight")

        # (1) the estimated economy alone
        fig, ax = plt.subplots(figsize=(figure_width, baseline_height))
        _draw(ax, baseline_regime, None)
        _dress(ax, True)
        fig.tight_layout()
        _save(fig, "baseline")
        axes_pair["baseline"] = ax

        # (2) the counterfactuals, one sub-row each, the baseline ring on every sub-row.
        # The first regime is drawn at the TOP of the row, as in the legend.
        row_step = 0.78 / max(n_counterfactuals, 1)
        offsets = {regime: ((n_counterfactuals - 1) / 2 - j) * row_step for j, regime in enumerate(counterfactuals)}
        fig, ax = plt.subplots(figsize=(figure_width, max(4.0, 0.12 * n_buyers * (n_counterfactuals + 1))))
        for i in range(0, n_buyers, 2):
            ax.axhspan(i - 0.5, i + 0.5, color="0.94", lw=0, zorder=0)
        baseline_point = local_point[baseline_regime].to_numpy(dtype=float)
        for j, regime in enumerate(counterfactuals):
            y = row_position + offsets[regime]
            ax.plot(baseline_point, y, "o", mfc="none", mec="black", mew=0.8, ms=3.6, lw=0, zorder=4,
                    label=f"{baseline_regime} (estimate)" if j == 0 else None)
            colour = CF_COLORS.get(regime, toulouse_color)
            point, arms = _arms(regime)
            ax.errorbar(point, y, xerr=arms, fmt="o", markersize=3.4, color=colour, ecolor=colour,
                        elinewidth=1.0, capsize=0.0, linestyle="none", zorder=3, label=regime)
        _dress(ax, True)
        ax.set_ylim(-0.5, n_buyers - 0.5)
        ax.grid(False, axis="y")
        ax.legend(frameon=False, fontsize=fs(8), loc="lower center",
                  bbox_to_anchor=(0.5, 1.0), ncol=2, handletextpad=0.3, columnspacing=1.2)
        fig.tight_layout()
        _save(fig, "counterfactuals")
        axes_pair["counterfactuals"] = ax
        return axes_pair

    # One INDEPENDENT figure per regime. `figsize` is now the size of ONE of them, and
    # the window `lim` is shared by construction rather than by `sharex`, which is what
    # keeps three separate files comparable.
    if figsize is None:
        figsize = (4.8, max(3.5, 0.22 * n_buyers)) if orientation == "h" \
            else (max(7.0, 0.55 * n_buyers), 3.4)
    stem, extension = os.path.splitext(save_to) if save_to else (None, ".pdf")
    axes = {}
    for regime in labels:
        fig, ax = plt.subplots(figsize=figsize)
        _draw(ax, regime, None if regime == baseline_regime else baseline_regime)
        # every figure names its own buyers (there is no leftmost panel to lean on) and
        # NONE of them names the x axis -- the quantity belongs in the caption.
        _dress(ax, True, name_zone=(orientation == "h"))
        if orientation == "v":
            ax.set_ylabel(value_label)

        fig.tight_layout()
        if stem:
            slug = re.sub(r"[^a-z0-9]+", "_", str(regime).lower()).strip("_")
            path = f"{stem}_{slug}{extension or '.pdf'}"
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            fig.savefig(path, bbox_inches="tight")
        axes[regime] = ax
    return axes
