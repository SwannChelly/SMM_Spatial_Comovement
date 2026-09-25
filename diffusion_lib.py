"""
Amplification and the spatial diffusion of a downstream shock -- part of the library
behind `tests_counterfactuals.ipynb`.

The amplification coefficient D_r, the local share L_r(d) and its map, the counterfactual
regimes, the sourcing barycentre, where the response LANDS (the incidence vector, its
concentration and its commonality), and the input-output benchmark for D_r.
"""

import os
import re
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patheffects as pe
import statsmodels.api as sm

try:
    import geopandas as gpd
except Exception:            # maps are optional
    gpd = None

from utils import (
    AMPLIFICATION_RADII, CF_COLORS, CF_REGIMES, UNIFORM_REGIME,
    _as_ze_string, _despine, _n_hat_from_diagnostics, _parquet_sector_index,
    _region_labels, get_figsize, reference_color, sim_color, sourcing_geometry,
    toulouse_color,
)


# ============================================================================
# Amplification and the local share
# ============================================================================

# The two objects of Figure 6 — the amplification coefficient D_r and the share of
# the upstream sales it generates that stays within a radius of the shock. Both are
# read off the simulated firm-level economy, so `mu` selects them. The conventions
# that set the LEVEL of D_r are in the markdown above; the radius is a free
# parameter throughout.

AMPLIFICATION_RADIUS_KM = 100      # the paper's Figure 6; 200 etc. are just as valid
# The radii every function below defaults to. It is a PRIVATE copy of the Constants
# cell's `AMPLIFICATION_RADII`, so re-running that cell cannot clobber an edit here
# and the section still runs when only the definition cells have been executed.
_DEFAULT_RADII = (AMPLIFICATION_RADIUS_KM, 200)


def build_diffusion_frame(data, value_col="share", per_replication=False):
    """
    One row per (shocked downstream region, upstream cell): the upstream sales that
    one euro spent in the downstream region generates there, and the distance
    between the two.

    The grid is ZERO-FILLED over every (downstream region, region) pair — a cell that
    supplies nothing still belongs in the denominator's support and, more to the
    point, in the map. Distances come from `distances.npy` read by the model's own
    integer region indices, so there is no ZE-name round trip to get wrong.
    """
    sup = data.get("suppliers")
    if sup is None:
        raise FileNotFoundError(
            f"no suppliers.parquet under {data['folder']}/{data.get('step_dir','<step>')}/ "
            f"or {data['folder']}/ — main.jl writes one per estimate in its post-hoc block.")
    if value_col not in sup.columns:
        raise KeyError(f"`{value_col}` is not a column of suppliers.parquet "
                       f"({list(sup.columns)}).")

    R = data["R"]
    D = np.load(data["input_folder"] / "distances.npy")[:R, :R]

    # The parquet holds `n_rep` INDEPENDENT realisations of the finite-variety economy
    # (see the section header). Pooling and dividing by `n_rep` gives E[X_lr], on which
    # D_r -- linear in the euros -- is the exact expectation. `per_replication=True`
    # keeps the realisations apart instead, which is what the dispersion needs.
    keys = ["ze2010_downstream", "ze2010"]
    has_rep = "replication" in sup.columns
    n_rep = int(sup["replication"].nunique()) if has_rep else 1
    if per_replication and has_rep:
        keys = ["replication"] + keys
        scale = 1.0
    else:
        scale = 1.0 / n_rep
    agg = (sup.groupby(keys, as_index=False)
              .agg(upstream_sales=(value_col, "sum"),
                   n_links=(value_col, "size")))
    agg["upstream_sales"] *= scale
    agg["n_links"] = agg["n_links"] * scale
    shocked = np.sort(sup["ze2010_downstream"].unique())

    levels = ([np.sort(sup["replication"].unique())] if keys[0] == "replication" else []) \
        + [shocked, np.arange(1, R + 1)]
    grid = pd.MultiIndex.from_product(levels, names=keys).to_frame(index=False)
    df = grid.merge(agg, on=keys, how="left").fillna(
        {"upstream_sales": 0.0, "n_links": 0})
    df["distance"] = D[df["ze2010"].to_numpy() - 1, df["ze2010_downstream"].to_numpy() - 1]

    lab = _region_labels(data)
    df = (df.merge(lab.rename(columns={"index": "ze2010", "ze2010": "ze_code",
                                       "ze2010_name": "ze_name"}), on="ze2010")
            .merge(lab.rename(columns={"index": "ze2010_downstream",
                                       "ze2010": "shocked_code",
                                       "ze2010_name": "shocked_name"}),
                   on="ze2010_downstream"))
    df["value_col"] = value_col
    return df


def amplification_summary(data, radii=(AMPLIFICATION_RADIUS_KM,), value_col="share",
                          diffusion=None):
    """
    One row per shocked downstream region: the amplification coefficient and the
    share of the upstream sales it generates that stays within each radius.

    `radii` is a tuple, so 100 km and 200 km (or any other) come out of one pass.
    """
    df = build_diffusion_frame(data, value_col) if diffusion is None else diffusion
    tot = df.groupby("ze2010_downstream")["upstream_sales"].sum()
    out = pd.DataFrame({
        "region": df.groupby("ze2010_downstream")["shocked_name"].first(),
        "upstream_sales": tot,
        "amplification": 1.0 + tot,
        "supplier_cells": df.assign(hit=df["upstream_sales"] > 0)
                            .groupby("ze2010_downstream")["hit"].sum(),
        "mean_upstream_distance": df.eval("upstream_sales * distance")
                                    .groupby(df["ze2010_downstream"]).sum() / tot.replace(0, np.nan),
    })
    for d in radii:
        near = df.loc[df["distance"] <= d].groupby("ze2010_downstream")["upstream_sales"].sum()
        out[f"share_within_{int(d)}km"] = (near.reindex(out.index).fillna(0.0)
                                           / tot.replace(0, np.nan))
    out.attrs["value_col"] = value_col
    out.attrs["radii"] = tuple(radii)
    return out.sort_values("amplification", ascending=False)


def local_share_profile(data, radii=(25, 50, 75, 100, 150, 200, 300, 500),
                        value_col="share", diffusion=None):
    """
    The share within `d` as a FUNCTION of `d`: the mean across shocked regions, its
    interquartile band, and the sales-weighted mean. The single 100 km number of
    Figure 6 is one point on this curve, and the curve says whether that number is a
    knife edge or a plateau.
    """
    df = build_diffusion_frame(data, value_col) if diffusion is None else diffusion
    tot = df.groupby("ze2010_downstream")["upstream_sales"].sum()
    rows = []
    for d in radii:
        near = (df.loc[df["distance"] <= d].groupby("ze2010_downstream")["upstream_sales"]
                  .sum().reindex(tot.index).fillna(0.0))
        sh = (near / tot.replace(0, np.nan)).dropna()
        rows.append({"radius_km": d, "mean": sh.mean(), "median": sh.median(),
                     "p25": sh.quantile(0.25), "p75": sh.quantile(0.75),
                     "weighted_mean": float(near.sum() / tot.sum()) if tot.sum() > 0 else np.nan})
    return pd.DataFrame(rows).set_index("radius_km")


# --- The same shock, with one force switched off ----------------------------
#
# Where a shock goes is decided by the Ricardian competition, which is closed form
# (`sourcing_geometry`, in utils), so the propagation can be recomputed with either
# force removed without re-simulating or re-estimating anything. The regimes
# themselves (`CF_REGIMES`, `CF_COLORS`) are in utils, beside the economy that
# builds them.



def counterfactual_diffusion_frame(data, alpha=None, equalise_T=False,
                                   value_col="share", diffusion=None, verbose=True):
    """
    ONE CHANNEL of a counterfactual, not the counterfactual: the upstream euros
    REALLOCATED across cells at a FIXED upstream spend.

    Read `amplification_decomposition` for the full answer. This function holds the
    spending shares where the estimated economy put them and moves only the within-sector
    geography, so

        X_{lr} = sum_s spend_{sr} * rho^{regime}_{lrs},

    with `spend` read off the economy and rho from `sourcing_geometry`. Since
    sum_l rho = 1 for every (sector, buyer), the total upstream sales of a shocked region
    -- hence $D_r$ -- come out IDENTICAL under every regime here. That is a property of
    what this function holds fixed, NOT a property of the model.

    An earlier version of this docstring argued the opposite: that
    $D_r = 1 + (1-\\Omega_L)(P_r/c_r)^{1-\\lambda}$ carries no comparative-advantage
    parameter, so `T` and `alpha` could not move it. They do not appear in that
    expression, but they decide which cell wins each variety and at what delivered cost,
    so they set $P_{sr}$ and hence $P_r$. With `lambda < 1` labour and intermediates are
    COMPLEMENTS, so a higher input price index raises the intermediate expenditure share:
    equalising comparative advantage makes sourcing less efficient, $P_r$ rises, and the
    buyer spends MORE of its euro upstream. Switching distance off runs the other way.
    That response is part of how comparative advantage governs amplification and must not
    be assumed away; it is channel (a) of `amplification_decomposition`, which this
    function is channel (c) of.

    And "Both forces" is the closed-form EXPECTATION of the allocation, whereas the
    parquet is one realised finite-variety draw of it: the closed form spreads mass over
    every modelled cell, the realisation concentrates it on the varieties that happened
    to win. The two therefore differ, and the gap is simulation noise plus that
    granularity — `counterfactual_amplification` prints it rather than hiding it, and
    "Both forces" (not the realised frame) is the right baseline to read the other three
    regimes against.
    """
    base = (build_diffusion_frame(data, value_col) if diffusion is None
            else diffusion).copy()
    sup = data.get("suppliers")
    if sup is None:
        raise FileNotFoundError(
            f"no suppliers.parquet under {data['folder']}/{data.get('step_dir','<step>')}/ "
            "— the counterfactual needs the realised sector-level spending.")

    geom = sourcing_geometry(data, alpha=alpha, equalise_T=equalise_T)
    col_of_rd = {int(z): j for j, z in enumerate(geom["downstream"])}
    spend = (sup.assign(_s=_parquet_sector_index(data, sup))
                .groupby(["ze2010_downstream", "_s"])[value_col].sum())
    # `sup` pools `n_rep` realisations of the finite-variety economy, so the sector spend
    # is AVERAGED over them, exactly as `build_diffusion_frame` averages the euros it
    # reallocates. Without this every regime would allocate `n_rep` euros where the base
    # frame allocates one and $D_r$ — linear in those euros — would come out `n_rep`
    # times too large. $L_r(d)$ is a ratio within a regime and would not have shown it,
    # which is why the division is here rather than left to the reader.
    if "replication" in sup.columns:
        spend = spend / float(sup["replication"].nunique())

    R = data["R"]
    X = np.zeros((R, R))                      # [upstream region, shocked region], 0-based
    dropped = []
    for (rd, s), tot in spend.items():
        blk = geom["by_sector"].get(int(s))
        j = col_of_rd.get(int(rd))
        if blk is None or j is None:
            dropped.append((int(rd), int(s), float(tot)))
            continue
        X[blk["cells"], int(rd) - 1] += float(tot) * blk["rho"][:, j]
    if dropped and verbose:
        lost = sum(t for _, _, t in dropped)
        print(f"  [counterfactual] {len(dropped)} (buyer, sector) pairs have no modelled "
              f"cells or no downstream column and were dropped "
              f"({lost:.3g} of upstream sales): {[(r, s) for r, s, _ in dropped[:4]]}")

    out = base.drop(columns=[c for c in ("upstream_sales", "n_links") if c in base.columns])
    out["upstream_sales"] = X[out["ze2010"].to_numpy() - 1,
                              out["ze2010_downstream"].to_numpy() - 1]
    out["n_links"] = np.nan       # a closed-form allocation has no realised linkages
    out.attrs["regime"] = ("alpha=0 " if alpha == 0 else "") + ("T equalised" if equalise_T else "")
    out.attrs["value_col"] = value_col
    return out


def counterfactual_frames(data, regimes=CF_REGIMES, value_col="share", diffusion=None,
                          verbose=True):
    """One reallocated diffusion frame per regime, built once off a shared base frame."""
    base = build_diffusion_frame(data, value_col) if diffusion is None else diffusion
    return {label: counterfactual_diffusion_frame(data, value_col=value_col,
                                                  diffusion=base, verbose=verbose, **kw)
            for label, kw in regimes.items()}


def simulated_frames(economies, value_col="share"):
    """`{regime: diffusion frame}` from the SIMULATED economies.

    A drop-in replacement for `counterfactual_frames` wherever the full response is
    wanted rather than the geography channel alone: every consumer that takes `frames=`
    (`counterfactual_amplification`, `incidence_concentration_overlap`, the figures) then
    reads an economy in which the intermediate expenditure share and the sector mix have
    responded, not one in which they were pinned at the estimate.
    """
    economies = {lab: ({**v[1], "economy": v[0]} if isinstance(v, tuple) else v)
                 for lab, v in economies.items()}
    return {lab: build_diffusion_frame(dl, value_col) for lab, dl in economies.items()}


def counterfactual_amplification(data, regimes=CF_REGIMES, radii=None, value_col="share",
                                 frames=None, diffusion=None, verbose=True):
    """
    One row per (shocked region, regime): the amplification coefficient, the share of
    upstream sales within each radius, and the mean distance those sales travel.

    The realised economy is reported too, as the regime `"Realised"`, so the closed-form
    "Both forces" row can be read against the draw it is the expectation of.
    """
    radii = _DEFAULT_RADII if radii is None else tuple(radii)
    base = build_diffusion_frame(data, value_col) if diffusion is None else diffusion
    frames = (counterfactual_frames(data, regimes=regimes, value_col=value_col,
                                    diffusion=base, verbose=verbose)
              if frames is None else frames)

    rows = {"Realised": amplification_summary(data, radii=radii, value_col=value_col,
                                              diffusion=base)}
    rows.update({lab: amplification_summary(data, radii=radii, value_col=value_col,
                                            diffusion=f) for lab, f in frames.items()})
    out = pd.concat([s.assign(regime=lab) for lab, s in rows.items()])
    out["regime"] = pd.Categorical(out["regime"], categories=list(rows), ordered=True)
    out = out.reset_index().set_index(["ze2010_downstream", "regime"]).sort_index()

    if verbose:
        amp = out["amplification"].unstack("regime")
        spread = float((amp[list(frames)].max(axis=1) - amp[list(frames)].min(axis=1)).max())
        gap = float((amp["Both forces"] - amp["Realised"]).abs().max())
        col = f"share_within_{int(radii[0])}km"
        sh = out[col].unstack("regime")
        if spread < 1e-9:
            print(f"  [counterfactual] D_r is identical across regimes to {spread:.2e} "
                  "-- because these frames hold the upstream spend fixed, not because "
                  "the model says so; see `amplification_decomposition`")
        else:
            print(f"  [counterfactual] D_r RANGES {spread:.4f} across regimes: these "
                  "frames come from re-solved economies, so the intermediate "
                  "expenditure share has responded")
        print(f"  [counterfactual] and matches the realised D_r to {gap:.2e}")
        print(f"  [counterfactual] {col}: realised {sh['Realised'].mean():.3f} vs "
              f"closed-form both-forces {sh['Both forces'].mean():.3f} — the gap is "
              "granularity, not a different economy")
    return out


def counterfactual_summary(data, regimes=CF_REGIMES, radii=None, value_col="share",
                           detail=None, frames=None, diffusion=None, verbose=False):
    """
    The counterfactual in one table: per regime, how much of the shock stays within each
    radius and how far it travels on average.

    Two averages per radius, because they answer different questions: `share_within_*`
    is the mean across shocked regions (the typical region), `weighted_*` weights each
    region by the upstream sales it generates (the aggregate euro).
    """
    radii = _DEFAULT_RADII if radii is None else tuple(radii)
    det = (counterfactual_amplification(data, regimes=regimes, radii=radii,
                                        value_col=value_col, frames=frames,
                                        diffusion=diffusion, verbose=verbose)
           if detail is None else detail)
    w = det["upstream_sales"]
    out = pd.DataFrame({
        "amplification": det["amplification"].groupby(level="regime", observed=True).mean(),
        "mean_upstream_distance":
            (det["mean_upstream_distance"] * w).groupby(level="regime", observed=True).sum()
            / w.groupby(level="regime", observed=True).sum(),
    })
    for d in radii:
        c = f"share_within_{int(d)}km"
        out[c] = det[c].groupby(level="regime", observed=True).mean()
        out[f"weighted_{c}"] = ((det[c] * w).groupby(level="regime", observed=True).sum()
                                / w.groupby(level="regime", observed=True).sum())
    return out


# --- The counterfactual, with the SPENDING SHARES allowed to respond -----------------
#
# `counterfactual_diffusion_frame` above holds the upstream spending fixed and reallocates
# it. Its docstring used to argue that this was not an approximation -- that
# `D_r = 1 + (1-Omega_L)(P_r/c_r)^{1-lambda}` carries no comparative-advantage parameter,
# so `T` and `alpha` cannot move it. That is wrong, and the error matters for the section's
# own question. `T` and `alpha` do not appear in that expression, but they decide WHICH
# cell wins each variety and at what delivered cost, so they set `P_sr`, hence `P_r`. With
# `lambda < 1` labour and intermediates are COMPLEMENTS, so a higher input price index
# RAISES the intermediate expenditure share: equalising comparative advantage makes
# sourcing less efficient, `P_r` rises, and the buyer spends MORE of its euro upstream, not
# the same amount redistributed. Switching distance off runs the other way.
#
# So the amplification counterfactual has to come from the re-solved economy, which is what
# `simulate_economy` gives. The fixed-spend reallocation is kept, demoted to what it
# actually is: one CHANNEL of the answer, not the answer.
#
# The three channels are EXACT, not an approximation, because the model factorises:
#
#     X_{lr} = (D_r - 1) * sum_s theta_{rs} * rho_{lrs}
#
# with `theta_rs` the sector's share of the intermediate bill (it sums to one over
# sectors -- gated by `economy_identities`) and `rho` the within-sector geography. Hence
#
#   (a) LEVEL       D_r - 1        how much of the euro leaves for upstream at all;
#   (b) MIX         theta_rs       which upstream SECTORS it goes to;
#   (c) GEOGRAPHY   rho_lrs        where inside a sector it lands.
#
# and each statistic is moved by a known subset of them. `D_r` moves through (a) ALONE.
# `L_r(d)` and `d_r` are RATIOS in which `(D_r - 1)` cancels exactly, so they move through
# (b) and (c) alone -- and (b) is a channel the fixed-spend route also shut down, since
# equalising `T` changes relative sector price indices and sectors differ in geography.


def _regime_profile(data, geom, mix, radii):
    """`p_rs(d)` and `d_rs` per (sector, buyer), and their mix-weighted aggregates.

    `p_rs(d) = sum_{l : d_lr <= d} rho_lrs` and `d_rs = sum_l rho_lrs d_lr` are pure
    geometry; the buyer-level statistics are `sum_s mix_rs * (.)`, exact because
    `sum_l rho = 1` per (sector, buyer) so the level factor cancels.
    """
    buyers = np.asarray(geom["downstream"]).astype(int)
    nb = buyers.size
    dist = np.zeros(nb)
    near = {d: np.zeros(nb) for d in radii}
    for s, blk in geom["by_sector"].items():
        rho, dd = blk["rho"], blk["distance"]          # (n_cell, n_buyer)
        w = np.asarray(mix)[int(s)]                    # (n_buyer,)
        dist += w * (rho * dd).sum(axis=0)
        for d in radii:
            near[d] += w * np.where(dd <= d, rho, 0.0).sum(axis=0)
    out = pd.DataFrame({"ze2010_downstream": buyers,
                        "mean_upstream_distance": dist}).set_index("ze2010_downstream")
    for d in radii:
        out[f"share_within_{int(d)}km"] = near[d]
    return out


def amplification_decomposition(economies, regimes=None, radii=None,
                                baseline="Both forces", verbose=True):
    """
    The counterfactual split into its three exact channels.

    `economies` is `{regime: data_like}` as `utils.reporting_data` returns it: each entry
    carries that regime's own re-solved economy under `"economy"`. Nothing is held fixed
    that the model lets move -- in particular the intermediate expenditure share responds,
    which is the whole point of reading amplification against comparative advantage.

    Returns one row per (regime, shocked region) with

        amplification        D_r, from the regime's own value block  -- channel (a) alone
        mean_upstream_distance, share_within_*km
                             the TOTAL effect, mix and geography both at the regime
        *_geography          the same statistics with the sector MIX held at the baseline,
                             so only rho moves                       -- channel (c) alone
        *_mix                with the geometry held at the baseline, so only theta moves
                             -- channel (b) alone

    `total - baseline` is not in general `geography + mix - 2*baseline`: the two channels
    interact through the weights, and the residual is reported by `decomposition_report`
    rather than assumed away.
    """
    radii = _DEFAULT_RADII if radii is None else tuple(radii)
    regs = CF_REGIMES if regimes is None else regimes
    # `reporting_data` returns {regime: data_like} with the economy under "economy";
    # `economy_by_regime` returns {regime: (economy, data_like)}. Both are accepted, so a
    # caller never has to remember which entry point it came through.
    economies = {lab: ({**v[1], "economy": v[0]} if isinstance(v, tuple) else v)
                 for lab, v in economies.items()}
    if baseline not in economies:
        raise KeyError(f"baseline {baseline!r} not among {list(economies)}")

    geo, mix = {}, {}
    for lab in economies:
        dl = economies[lab]
        econ = dl.get("economy")
        if econ is None:
            raise KeyError(f"{lab}: no simulated economy -- `reporting_data` returns one "
                           "per regime; a plain `load_granular_data` dict does not.")
        kw = regs.get(lab, {"alpha": econ.meta.get("alpha"),
                            "equalise_T": econ.meta.get("equalise_T", False)})
        geo[lab] = sourcing_geometry(dl, **kw)
        mix[lab] = econ.value["theta_rs"].mean(axis=0)          # (S, n_buyer)

    rows = []
    for lab, dl in economies.items():
        econ = dl["economy"]
        buyers = np.asarray(econ.meta.get("value_buyers", econ.meta["buyers"])).astype(int)
        tot = _regime_profile(dl, geo[lab], mix[lab], radii)
        gch = _regime_profile(dl, geo[lab], mix[baseline], radii)      # mix held at base
        mch = _regime_profile(dl, geo[baseline], mix[lab], radii)      # geometry held
        blk = tot.copy()
        for c in tot.columns:
            blk[f"{c}_geography"] = gch[c]
            blk[f"{c}_mix"] = mch[c]
        blk["amplification"] = econ.value["D_r"].mean(axis=0)
        blk["region"] = _region_labels(dl).set_index("index")["ze2010_name"].reindex(buyers).to_numpy()
        blk["regime"] = lab
        rows.append(blk.reset_index())
    out = pd.concat(rows, ignore_index=True)
    out["regime"] = pd.Categorical(out["regime"], categories=list(economies), ordered=True)
    out = out.set_index(["ze2010_downstream", "regime"]).sort_index()
    out.attrs["radii"] = radii
    out.attrs["baseline"] = baseline
    if verbose:
        decomposition_report(out, radii=radii, baseline=baseline)
    return out


def decomposition_report(detail, radii=None, baseline="Both forces"):
    """Print the three channels per regime, and say which statistic each one can move."""
    radii = detail.attrs.get("radii", _DEFAULT_RADII) if radii is None else tuple(radii)
    amp = detail["amplification"].unstack("regime")
    base_amp = amp[baseline]
    print(f"  [decomposition] baseline = {baseline!r}; means across shocked regions\n")
    print(f"    {'regime':<28s} {'D_r':>8s} {'dD_r':>8s} | "
          f"{'total':>8s} {'geogr.':>8s} {'mix':>8s} {'resid':>8s}   (share within "
          f"{int(radii[0])} km, deviation from baseline)")
    col = f"share_within_{int(radii[0])}km"
    sh = detail[col].unstack("regime")
    shg = detail[f"{col}_geography"].unstack("regime")
    shm = detail[f"{col}_mix"].unstack("regime")
    for lab in amp.columns:
        d_tot = float((sh[lab] - sh[baseline]).mean())
        d_geo = float((shg[lab] - sh[baseline]).mean())
        d_mix = float((shm[lab] - sh[baseline]).mean())
        print(f"    {str(lab):<28s} {amp[lab].mean():8.4f} "
              f"{float((amp[lab] - base_amp).mean()):+8.4f} | "
              f"{d_tot:+8.4f} {d_geo:+8.4f} {d_mix:+8.4f} {d_tot - d_geo - d_mix:+8.4f}")
    print("\n    D_r moves through the LEVEL channel alone -- the labour-against-"
          "intermediates\n    margin, which responds because equalising T raises P_r and "
          "lambda < 1 makes\n    the two complements. The local share is a ratio in which "
          "(D_r - 1) cancels\n    exactly, so it moves through the sector MIX and the "
          "within-sector GEOGRAPHY only.")

# --- The economy is FINITE-VARIETY, and the parquet has to say so ------------
#
# The model gives sector s exactly N_s varieties; the count moment calibrates N_hat_s
# against the share of cells hosting no supplier. `N_rho` is a DIFFERENT object -- the
# number of draws integrating over the variety continuum, `max(100, max_s N_HI)` in
# load_parameters.jl, one to two orders of magnitude larger. A post-hoc economy written
# over N_rho draws is the N_s -> infinity limit of the model, not the model, and the
# symptom is visible without any reference: a shock reaches nearly EVERY commuting zone
# while the fitted count moment says three quarters of the cells host nobody.


def supplier_count_check(data, verbose=True):
    """
    Does `suppliers.parquet` carry `N_hat_s` varieties per sector, as the model says?

    This is a check on JULIA's economy. An economy simulated here is drawn at `N_hat_s`
    by construction, so running it on a simulated frame can only ever pass; the reporting
    path skips it for that reason and it is kept for the parquet.

    The exact test counts DISTINCT VARIETY INDICES within a (sector, replication): the
    model gives sector s exactly N_s of them, every one of which is won by somebody, so
    the count must equal `N_hat_s` on the nose.

    Counting distinct FIRMS cannot do that job, and the reason is worth stating because
    it is the fallback path. One variety is won by different cells for different buyers,
    so it yields between one and `R_downstream` firms; a legitimate variety economy can
    therefore show up to `R_downstream` firms per variety, and a draw-count economy
    shows `N_rho / N_hat_s`, which on the real data is of the same order. The two are
    separable only through the variety index itself. A tree written before the parquet
    carried one falls back to the firm count, and the verdict is then a BOUND: above
    `R_downstream` firms per variety the economy cannot be a variety economy at all.
    """
    sup = data.get("suppliers")
    if sup is None:
        raise FileNotFoundError(
            f"no suppliers.parquet under {data['folder']}/{data.get('step_dir', '<step>')}/.")

    s_idx = _parquet_sector_index(data, sup)
    rep = (sup["replication"].to_numpy() if "replication" in sup.columns
           else np.zeros(len(sup), dtype=int))
    exact = "variety" in sup.columns
    unit = sup["variety"] if exact else sup["SIREN"]
    frame = pd.DataFrame({"sector": s_idx, "replication": rep,
                          "unit": unit.to_numpy()})
    per = (frame.drop_duplicates(["sector", "replication", "unit"])
                .groupby(["sector", "replication"]).size()
                .groupby("sector").mean())

    n_hat = data.get("post_hoc_N_hat")
    if n_hat is None:
        n_hat = _n_hat_from_diagnostics(data)
    col = "varieties_in_parquet" if exact else "firms_per_sector"
    out = pd.DataFrame({
        "sector": [str(c) for c in data["sector_names"]],
        "N_hat": (np.asarray(n_hat, dtype=float) if n_hat is not None
                  else np.full(data["S"], np.nan)),
        col: per.reindex(range(data["S"])).to_numpy(),
    }).set_index("sector")
    out["ratio"] = out[col] / out["N_hat"]
    out["n_replications"] = int(pd.Series(rep).nunique())
    out.attrs["exact"] = exact

    R_d = int(sup["ze2010_downstream"].nunique())
    tol = 1e-9 if exact else float(R_d)
    ok = bool(out["N_hat"].notna().all()) and bool((out["ratio"] - 1.0).max() <= tol)
    if verbose:
        print(out.round(2).to_string())
        if out["N_hat"].isna().any():
            print("  [UNKNOWN] no N_hat_s on disk (post_hoc_N_hat.npy or "
                  "granular_diagnostics.npz) — nothing to check the count against.")
        elif not exact:
            print(f"  [BOUND] no `variety` column: counting FIRMS, which a variety "
                  f"economy inflates by at most R_downstream = {R_d}. "
                  + (f"ratio max {out['ratio'].max():.1f} <= {R_d}: consistent with a "
                     "variety economy, though not proof of one."
                     if ok else
                     f"ratio max {out['ratio'].max():.1f} > {R_d}: this economy was "
                     "solved on the estimation DRAWS, not on N_hat_s varieties. "
                     "Re-run main.jl's post-hoc block."))
        elif ok:
            print(f"  [OK] {out['n_replications'].iloc[0]} realisation(s), exactly "
                  "N_hat_s varieties per sector.")
        else:
            print("  [FAIL] the variety count in the parquet is not N_hat_s.")
    return out


# --- Averaging over realisations, and what granularity costs ----------------
#
# With N_s finite the network is a RANDOM object, so no single realisation is a result.
# Two different things follow, and they must not be run together.
#
# (1) The AVERAGE network. `build_diffusion_frame` divides the pooled euros by the
#     number of realisations, so every downstream object is built on E[X_lr]. D_r is
#     linear in those euros, so the averaged frame gives E[D_r] exactly.
#
# (2) The DISPERSION, which is content and not error. `L_r(d)` is a ratio, so its mean
#     across realisations is not the ratio of the means; more to the point, the SUPPORT
#     is what granularity moves — a buyer sources N_hat_s varieties per sector, so it
#     reaches a few dozen origins, not every commuting zone.
#
# And granularity itself is the deviation from the N_s -> infinity economy, which is
# `suppliers_continuum.parquet`: the SAME parameters solved on the estimation draws.
# The two differ for a reason that is not sampling noise. The within-sector index
# p_sr = [N_s^{-1} sum_rho p_rho^{1-nu}]^{1/(1-nu)} is nonlinear in the draws, so its
# EXPECTATION moves with N_s; with lambda < 1 a higher intermediate price index raises
# the intermediate share, hence D_r. That is the price-index channel.


def replication_summaries(data, radii=None, value_col="share", diffusion=None):
    """
    `amplification_summary` computed SEPARATELY for each realisation of the economy.

    Indexed by (replication, shocked region). A tree with no `replication` column comes
    back with the single realisation labelled 0, so callers need no special case.
    """
    radii = _DEFAULT_RADII if radii is None else tuple(radii)
    df = (build_diffusion_frame(data, value_col, per_replication=True)
          if diffusion is None else diffusion)
    if "replication" not in df.columns:
        df = df.assign(replication=0)
    out = []
    for b, sub in df.groupby("replication"):
        s = amplification_summary(data, radii=radii, value_col=value_col,
                                  diffusion=sub.drop(columns="replication"))
        out.append(s.assign(replication=b))
    return (pd.concat(out).set_index("replication", append=True)
              .reorder_levels(["replication", "ze2010_downstream"]).sort_index())


def granular_band(data, radii=None, value_col="share", per_rep=None):
    """
    Per shocked region, each quantity's mean and spread ACROSS realisations.

    The mean of `share_within_d` here is the mean of a RATIO — the direct estimate of
    E[L_r(d)] — where the averaged-frame figure is a ratio of means; they differ at
    O(1/n_rep). `supplier_cells` is the one that changes character: on the averaged
    frame it is the UNION of origins over all realisations, here it is how many origins
    a single shock actually reaches, which is the granular object.
    """
    radii = _DEFAULT_RADII if radii is None else tuple(radii)
    per = (replication_summaries(data, radii=radii, value_col=value_col)
           if per_rep is None else per_rep)
    cols = ["amplification", "mean_upstream_distance", "supplier_cells"] + \
           [f"share_within_{int(d)}km" for d in radii]
    g = per.groupby(level="ze2010_downstream")
    out = pd.DataFrame({"region": g["region"].first()})
    for c in cols:
        out[f"{c}"] = g[c].mean()
        out[f"{c}_sd"] = g[c].std(ddof=1)
    out["n_replications"] = g.size()
    return out


def granularity_report(data, radii=None, value_col="share", verbose=True,
                       continuum=None):
    """
    The finite-variety economy against its infinite-variety limit, quantity by quantity.

    Returns one row per quantity: the granular mean (over realisations), its spread
    across realisations, the continuum value, and the gap. The gap is granularity; the
    spread is how much a single draw can move the answer.

    `continuum` is the benchmark frame, in `suppliers.parquet`'s schema -- build it with
    `utils.continuum_data(data, xp)["suppliers"]`. Omitted, Julia's own
    `suppliers_continuum.parquet` is used if the tree carries one.
    """
    radii = _DEFAULT_RADII if radii is None else tuple(radii)
    cont = continuum if continuum is not None else data.get("suppliers_continuum")
    if cont is None:
        raise FileNotFoundError(
            "no infinite-variety benchmark. Build one with `utils.continuum_data(data, "
            "xp)` and pass its `suppliers` frame as `continuum=`, or point `data` at a "
            "tree carrying Julia's own `suppliers_continuum.parquet`.")

    per = replication_summaries(data, radii=radii, value_col=value_col)
    band = granular_band(data, radii=radii, per_rep=per)
    ref = amplification_summary(
        data, radii=radii, value_col=value_col,
        diffusion=build_diffusion_frame(dict(data, suppliers=cont), value_col))

    cols = ["amplification", "mean_upstream_distance", "supplier_cells"] + \
           [f"share_within_{int(d)}km" for d in radii]
    rows = []
    for c in cols:
        # across regions: the granular economy's mean, the spread a single draw carries,
        # and the same statistic in the N_s -> infinity economy
        rows.append({"quantity": c,
                     "granular_mean": band[c].mean(),
                     "sd_across_draws": band[f"{c}_sd"].mean(),
                     "continuum": ref[c].mean()})
    out = pd.DataFrame(rows).set_index("quantity")
    out["granularity"] = out["granular_mean"] - out["continuum"]
    out["granularity_rel"] = out["granularity"] / out["continuum"].replace(0, np.nan)
    out.attrs["n_replications"] = int(per.index.get_level_values("replication").nunique())

    if verbose:
        n = out.attrs["n_replications"]
        print(f"  granularity = the finite-variety economy minus its N_s -> infinity "
              f"limit ({n} realisations):")
        print(out.round(4).to_string())
        gap = out.loc["supplier_cells"]
        print(f"  a shock reaches {gap['granular_mean']:.0f} origins with N_hat_s "
              f"varieties against {gap['continuum']:.0f} in the continuum — the "
              "extensive margin is the whole of what granularity is for.")
    return out


# ============================================================================
# The figures
# ============================================================================

# The two Figure-6 panels (the second with every radius nested on one bar), the radius
# sweep the fixed 100 km hides, the two together in one scatter, and the same shock
# propagated with one force switched off.

def plot_amplification(data, value_col="share", summary=None, figsize=None,
                       save_to=None):
    """Top panel of Figure 6: the amplification coefficient D_r, by commuting zone."""
    s = (amplification_summary(data, value_col=value_col) if summary is None
         else summary).sort_values("amplification")
    fig, ax = plt.subplots(figsize=figsize or (8, max(3.0, 0.28 * len(s))))
    ax.barh(s["region"].astype(str), s["amplification"], color=toulouse_color)
    ax.set_xlabel(r"Amplification measure $D_r$")
    ax.set_ylabel("Commuting zone")
    ax.set_xlim(left=1.0)
    ax.grid(alpha=0.2, axis="x")
    ax.set_title(f"{data['industry']}, " rf"$\hat\mu_{data['mu']}$"
                 f"   mean $D_r$ = {s['amplification'].mean():.3f}", fontsize=11)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        fig.savefig(save_to, bbox_inches="tight")
    return ax


def _ring_colors(n):
    """
    Colours for nested radii: the headline radius in the section's own colour, every
    ring beyond it in blue, lightening outward so the order of the radii is visible
    without reading the legend.
    """
    if n <= 1:
        return [toulouse_color]
    blue = np.asarray(sim_color, dtype=float)
    rings = [tuple(1.0 - (1.0 - blue) * (1.0 - 0.30 * k / max(1, n - 2)))
             if n > 2 else tuple(blue) for k in range(n - 1)]
    return [toulouse_color] + rings


def plot_local_share(data, radii=None, value_col="share", summary=None, radius_km=None,
                     sort_by="share_within_100km", order=None, xmax=None, ax=None, figsize=None,
                     save_to=None, title=None):
    """
    Bottom panel of Figure 6, with every radius on the SAME bar.

    $L_r(d)$ is a CDF in $d$: the 200 km share CONTAINS the 100 km one. Side-by-side
    bars would hide that nesting and invite the two numbers to be read as competing
    measurements. Drawn nested instead — the widest radius first, each narrower one
    painted over it, all fully opaque — one bar says "this much stays within 100 km, and
    this much more is picked up by going out to 200 km", which is what the nesting
    means. Regions are sorted by the headline (smallest) radius, so the inner segments
    form a staircase and the blue extensions are read against it.

    `xmax` pins the axis. Left to itself it tracks the widest bar, which uses the panel
    but gives the two industries different scales; pass the same number to both when the
    figures are to be compared side by side.
    """
    if radii is None:
        radii = (radius_km,) if radius_km is not None else _DEFAULT_RADII
    radii = tuple(sorted({float(r) for r in np.atleast_1d(radii)}))
    cols = [f"share_within_{int(r)}km" for r in radii]

    s = (amplification_summary(data, radii=radii, value_col=value_col)
         if summary is None else summary)
    missing = [c for c in cols if c not in s.columns]
    if missing:
        raise KeyError(f"{missing} not in the summary — pass `radii={radii}` to "
                       "amplification_summary, or let this function build it.")
    s = s.reindex(order) if order is not None else s.sort_values(sort_by or cols[0])

    if ax is None:
        _, ax = plt.subplots(figsize=figsize or (8, max(3.0, 0.28 * len(s))))
    colors = _ring_colors(len(radii))
    y = np.arange(len(s))
    handles = []
    for i in range(len(radii) - 1, -1, -1):          # widest first, narrowest on top
        lab = (f"within {radii[0]:g} km" if i == 0
               else f"{radii[i-1]:g}–{radii[i]:g} km")
        handles.append(ax.barh(y, s[cols[i]].to_numpy(), height=0.75, color=colors[i],
                               alpha=1.0, label=lab, zorder=2 + (len(radii) - i)))
    ax.set_yticks(y)
    ax.set_yticklabels(s["region"].astype(str))
    ax.set_ylim(-0.7, len(s) - 0.3)
    # no bar comes near 1, and a mostly empty panel is harder to read, not more honest:
    # the axis tracks the widest bar unless `xmax` pins it
    top = float(s[cols[-1]].max())
    ax.set_xlim(0, xmax if xmax is not None else max(0.1, np.ceil(top * 20) / 20 + 0.05))
    # the means ride on the x label: inside the panel they would sooner or later land on
    # a bar, and in the title they would run into the legend
    ax.set_xlabel("Share of the upstream sales a shock generates\n"
                  + "mean  " + ",  ".join(f"{s[c].mean():.3f} within {r:g} km"
                                          for r, c in zip(radii, cols)))
    ax.set_ylabel("Commuting zone")
    ax.grid(alpha=0.2, axis="x")
    # the legend sits ABOVE the axes: with a bar per commuting zone there is no corner
    # inside the plot it can occupy without covering the regions it explains
    ax.legend(handles=handles[::-1], frameon=False, fontsize=9, ncol=len(radii),
              loc="lower left", bbox_to_anchor=(0.0, 1.005))
    ax.set_title(title or (f"{data['industry']}, " rf"$\hat\mu_{data['mu']}$"),
                 fontsize=10, loc="right")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.figure.tight_layout()
    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        ax.figure.savefig(save_to, bbox_inches="tight")
    return ax


# The 100 km share is a number attached to a PLACE, and the contrast between the two
# industries is a contrast of geography: aerospace concentrates on two hubs, motor
# vehicles spreads over a corridor. A bar chart sorted by value hides exactly that,
# so the same column is also drawn on the map of commuting zones.
#
# The colour scale is PINNED by default and shared by both panels. The whole content
# of the pair is that aerospace reaches 0.42 where motor vehicles never passes 0.18;
# a per-panel scale would rescale that away and paint the two maps alike.
LOCAL_SHARE_MAP_VLIM = (0.0, 0.45)
LOCAL_SHARE_MAP_VLIM_WIDE = (0.0, 0.60)   # the same scale, pinned for the 200 km map


def plot_local_share_map(data, radius_km=None, value_col="share", summary=None,
                         diffusion=None, vlim=LOCAL_SHARE_MAP_VLIM, cmap="viridis",
                         n_label=0, missing_color="0.88", ax=None, figsize=None,
                         save_to=None, title=None, cbar=True):
    """
    $L_r(d)$ on the map: every SHOCKED downstream commuting zone coloured by the share
    of the upstream sales its own shock keeps within `radius_km`.

    Only downstream zones carry a value — the object is indexed by where the shock
    STARTS, not by where the euros land — so every other zone is drawn in
    `missing_color`. That is not missing data: those zones host no downstream producer
    and there is no shock to originate there.

    `vlim` pins the colour scale so the two industries are comparable; a value outside
    it would be clipped, so an exceedance is reported rather than silently flattened.

    `n_label` names that many of the most local zones on the map. It defaults to ZERO:
    the figure's content is the SHAPE of the colour field — a couple of bright hubs
    against a dark ground, or an even mid-tone corridor — and a name pinned to the
    brightest patch pulls the eye to one zone and invites the pair to be read as a
    ranking of places rather than as two geographies. The zones that matter are named in
    the text, where they can be given their number.
    """
    fr = data.get("france")
    # a GeoDataFrame, not merely a table of codes: the loader leaves `france` at None
    # when geopandas is absent, and a run tree assembled without the geometry can still
    # carry the names, so the geometry is what is checked for
    if fr is None or not hasattr(fr, "geometry"):
        raise FileNotFoundError(
            "no commuting-zone GEOMETRY in `data['france']` (france.gpkg missing, or "
            "geopandas is not installed) — the map cannot be drawn.")

    radius_km = _DEFAULT_RADII[0] if radius_km is None else radius_km
    s = (amplification_summary(data, radii=(radius_km,), value_col=value_col,
                               diffusion=diffusion)
         if summary is None else summary).copy()
    col = f"share_within_{int(radius_km)}km"
    if col not in s.columns:
        raise KeyError(f"{col} is not in the summary — pass `radii=({radius_km},)` to "
                       "amplification_summary, or let this function build it.")

    # the summary is indexed by the MODEL's downstream region index; the map is keyed on
    # the ZE code, so it goes through the same index -> code table as every label here
    code_of = _region_labels(data).set_index("index")["ze2010"]
    s["ze2010"] = _as_ze_string(code_of.reindex(s.index))

    geo = fr.copy()
    geo["ze2010"] = _as_ze_string(geo["ze2010"])
    geo = geo.merge(s[["ze2010", col, "region"]], on="ze2010", how="left")
    shocked = geo[geo[col].notna()]
    if shocked.empty:
        raise ValueError("no shocked commuting zone matched france.gpkg on `ze2010` — "
                         "the codes are formatted differently on the two sides.")

    vmin, vmax = (float(shocked[col].min()), float(shocked[col].max())) if vlim is None \
        else (float(vlim[0]), float(vlim[1]))
    over = shocked.loc[shocked[col] > vmax, "region"]
    if len(over):
        print(f"  [map] {len(over)} zone(s) above the pinned vmax = {vmax:g} and so "
              f"clipped: {', '.join(map(str, over))} — widen LOCAL_SHARE_MAP_VLIM.")
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize or get_figsize(wf=0.5, hf=1.0))
    geo[geo[col].isna()].plot(facecolor=missing_color, ax=ax, edgecolor="white",
                              linewidth=0.25)
    shocked.plot(column=col, ax=ax, norm=norm, cmap=cmap, edgecolor="black",
                 linewidth=0.35)

    for _, row in shocked.nlargest(n_label, col).iterrows() if n_label else ():
        p = row.geometry.representative_point()
        ax.annotate(str(row["region"]), (p.x, p.y), fontsize=8, xytext=(6, 4),
                    textcoords="offset points", color="black",
                    path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])

    ax.set_xlim(-5, 10)
    ax.set_ylim(42, 52)
    ax.set_axis_off()
    if title is not None:
        ax.set_title(title, fontsize=11)
    if cbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = ax.figure.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.035,
                                pad=0.02, aspect=35)
        cb.set_label(f"Share of upstream sales within {radius_km:g} km", fontsize=9)
        cb.ax.tick_params(labelsize=8)
        cb.minorticks_off()
    ax.figure.tight_layout()
    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        ax.figure.savefig(save_to, bbox_inches="tight")
    return ax


def plot_local_share_profile(data, radii=(25, 50, 75, 100, 150, 200, 300, 500),
                             mark=(AMPLIFICATION_RADIUS_KM,), value_col="share",
                             profile=None, figsize=None, save_to=None):
    """
    The share within `d` against `d`: the median across shocked regions with its
    interquartile band, and the sales-weighted mean. Shows whether the headline
    number is robust to the radius or sits on a steep part of the curve.
    """
    p = (local_share_profile(data, radii=radii, value_col=value_col)
         if profile is None else profile)
    fig, ax = plt.subplots(figsize=figsize or get_figsize(wf=0.8, hf=0.6))
    ax.fill_between(p.index, p["p25"], p["p75"], color=sim_color, alpha=0.20,
                    label="interquartile range across regions")
    ax.plot(p.index, p["median"], marker="o", color=sim_color, label="median region")
    ax.plot(p.index, p["weighted_mean"], marker="s", color=reference_color,
            linestyle="--", label="sales-weighted mean")
    ax.set_ylim(0, 1)
    for m in mark:
        ax.axvline(m, color="0.6", linewidth=0.9, linestyle=":")
        ax.annotate(f"{m:g} km", (m, 0.98), fontsize=8, color="0.4", ha="left",
                    va="top", xytext=(3, 0), textcoords="offset points")
    ax.set_xlabel("Radius around the shocked region (km)")
    ax.set_ylabel("Share of upstream sales within the radius")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title(f"{data['industry']}, " rf"$\hat\mu_{data['mu']}$", fontsize=11)
    _despine(ax)
    fig.tight_layout()
    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        fig.savefig(save_to, bbox_inches="tight")
    return ax


def plot_amplification_vs_local(data, radius_km=AMPLIFICATION_RADIUS_KM,
                                value_col="share", summary=None, n_label=5,
                                figsize=None, save_to=None):
    """
    How much a shock amplifies against how much of it stays nearby — the two panels
    of Figure 6 in one picture. Labelled: the `n_label` regions that keep the least
    locally, and the `n_label` that amplify the most.
    """
    s = (amplification_summary(data, radii=(radius_km,), value_col=value_col)
         if summary is None else summary).copy()
    col = f"share_within_{int(radius_km)}km"
    flagged = set(s.nsmallest(n_label, col).index) | set(s.nlargest(n_label, "amplification").index)

    fig, ax = plt.subplots(figsize=figsize or get_figsize(wf=0.8, hf=0.7))
    plain = s.loc[~s.index.isin(flagged)]
    named = s.loc[s.index.isin(flagged)]
    ax.scatter(plain["amplification"], plain[col], s=30, color=toulouse_color, alpha=0.8)
    ax.scatter(named["amplification"], named[col], s=45, facecolors="none",
               edgecolors=toulouse_color, linewidths=1.2)
    texts = [ax.text(r["amplification"], r[col], str(r["region"]), fontsize=8)
             for _, r in named.iterrows()]
    try:                                   # nicer label placement when available
        from adjustText import adjust_text
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", lw=0.5, color="black"))
    except ImportError:
        pass
    ax.set_xlabel(r"Amplification measure $D_r$")
    ax.set_ylabel(f"Share of upstream sales within {radius_km:g} km")
    ax.set_title(f"{data['industry']}, " rf"$\hat\mu_{data['mu']}$", fontsize=11)
    _despine(ax)
    fig.tight_layout()
    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        fig.savefig(save_to, bbox_inches="tight")
    return ax

def plot_counterfactual_local_share(data, regimes=CF_REGIMES, radius_km=None,
                                    value_col="share", detail=None, frames=None,
                                    diffusion=None, include_realised=True, xmax=None,
                                    figsize=None, save_to=None):
    """
    The same shock, propagated with one force switched off: the share of upstream sales
    staying within `radius_km`, region by region, one bar per regime.

    Read as two switches against the estimated allocation, not against chance. The step
    from `Both forces` to `Distance only` is what comparative advantage was doing; the
    step to `Comparative advantage only` is what proximity was doing. The first is the
    one with content: $T$ pulls sourcing towards a few favoured areas wherever they
    happen to be, so removing it pushes the local share UP for most regions and down
    only for those that host a favoured area — a SIGNED prediction, which is why the two
    industries can disagree. Removing distance can only push it down.

    Regions keep the order of the previous figure (the realised share within the same
    radius), so the two are read together. $D_r$ is not re-plotted: it is identical in
    every regime by construction, which is the whole point — the counterfactual moves a
    shock in space without changing its size.
    """
    radius_km = _DEFAULT_RADII[0] if radius_km is None else radius_km
    det = (counterfactual_amplification(data, regimes=regimes, radii=(radius_km,),
                                        value_col=value_col, frames=frames,
                                        diffusion=diffusion)
           if detail is None else detail)
    col = f"share_within_{int(radius_km)}km"
    if col not in det.columns:
        raise KeyError(f"{col} is not in the detail table — pass "
                       f"`radii=({radius_km},)` to counterfactual_amplification.")

    wide = det[col].unstack("regime")
    order = wide["Realised"].sort_values().index      # the order of the previous figure
    labels = [r for r in (["Realised"] if include_realised else []) + list(regimes)
              if r in wide.columns]
    wide = wide[labels].reindex(order)
    names = det["region"].groupby(level="ze2010_downstream").first().reindex(wide.index)

    # one row per region, sized so each of the regimes keeps a legible bar
    fig, ax = plt.subplots(
        figsize=figsize or (9, max(3.5, 0.085 * len(labels) * len(wide))))
    y = np.arange(len(wide))
    height = 0.8 / len(labels)
    for k, lab in enumerate(labels):
        off = ((len(labels) - 1) / 2 - k) * height        # first label on top of the group
        ax.barh(y + off, wide[lab].to_numpy(), height=height, alpha=1.0,
                color=CF_COLORS.get(lab, toulouse_color),
                edgecolor="white", linewidth=0.4, label=lab)
    ax.set_yticks(y)
    ax.set_yticklabels(names.astype(str))
    ax.set_ylim(-0.6, len(wide) - 0.4)
    top = float(np.nanmax(wide.to_numpy()))
    ax.set_xlim(0, xmax if xmax is not None else max(0.1, np.ceil(top * 20) / 20 + 0.05))
    ax.set_xlabel(f"Share of upstream sales within {radius_km:g} km")
    ax.set_ylabel("Commuting zone")
    ax.grid(alpha=0.2, axis="x")
    ax.legend(frameon=False, fontsize=9, ncol=min(3, len(labels)),
              loc="lower left", bbox_to_anchor=(0.0, 1.005))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        fig.savefig(save_to, bbox_inches="tight")
    return ax


def plot_counterfactual_distance(data, regimes=CF_REGIMES, value_col="share",
                                 detail=None, frames=None, diffusion=None,
                                 include_realised=True, units="pct", baseline="Both forces",
                                 kind="bar", annotate_km=True, xmin=None, xmax=None,
                                 figsize=None, save_to=None):
    """
    The per-buyer companion of `plot_counterfactual_local_share`: the average sourcing
    distance `d_r` of each shocked region, one mark per regime, regions in the SAME
    order as that figure (ascending realised local share) so the two read together.

    `units` is the design decision, and it is the one that makes the figure readable.
    Every `d_r` sits between roughly 250 and 450 km, so in KILOMETRES the entire
    counterfactual lives in the last sixth of the axis and the eye reads the LEVEL of
    the sourcing distance -- which is common across regions and carries no information --
    rather than its MOVEMENT, which is the whole content. `units="pct"` (the default)
    therefore plots each regime as a percentage deviation from `baseline` ("Both forces",
    the estimated economy), so the origin IS the baseline: a bar's length is then the
    effect itself, read against a zero that means something, and the honest-zero
    objection that forced `kind="point"` in kilometres no longer applies. The level is
    not thrown away -- with `annotate_km=True` each row carries its baseline `d_r` in
    kilometres at the right margin, so the percentage can be converted back to a
    distance buyer by buyer. `units="km"` keeps the original levels figure.

    `kind` still switches bars (length from the origin) against points joined by a rule
    (position only, so a cropped window is legitimate). In percentage units the baseline
    regime is exactly zero by construction and is drawn as the zero rule, not as a
    series.
    """
    det = (counterfactual_amplification(data, regimes=regimes, value_col=value_col,
                                        frames=frames, diffusion=diffusion, verbose=False)
           if detail is None else detail)
    if kind not in ("bar", "point"):
        raise ValueError(f"kind must be 'bar' or 'point', got {kind!r}.")
    if units not in ("pct", "km"):
        raise ValueError(f"units must be 'pct' or 'km', got {units!r}.")

    wide = det["mean_upstream_distance"].unstack("regime")
    # the ordering of plot_counterfactual_local_share, so the two panels line up
    share_cols = [c for c in det.columns if c.startswith("share_within_")]
    if share_cols and "Realised" in wide.columns:
        order = det[sorted(share_cols)[0]].unstack("regime")["Realised"].sort_values().index
    else:
        order = wide.iloc[:, 0].sort_values().index
    labels = [r for r in (["Realised"] if include_realised else []) + list(regimes)
              if r in wide.columns]
    wide = wide[labels].reindex(order)
    names = det["region"].groupby(level="ze2010_downstream").first().reindex(wide.index)

    base_km = None
    if units == "pct":
        if baseline not in wide.columns:
            raise KeyError(f"baseline regime {baseline!r} not among {list(wide.columns)}.")
        base_km = wide[baseline].astype(float)
        wide = wide.drop(columns=[baseline]).apply(
            lambda c: 100.0 * (c.astype(float) - base_km) / base_km)
        labels = [l for l in labels if l != baseline]

    fig, ax = plt.subplots(
        figsize=figsize or (9, max(3.5, 0.085 * max(len(labels), 1) * len(wide))))
    y = np.arange(len(wide))
    if kind == "bar":
        height = 0.8 / max(len(labels), 1)
        for k, lab in enumerate(labels):
            off = ((len(labels) - 1) / 2 - k) * height
            ax.barh(y + off, wide[lab].to_numpy(), height=height, alpha=1.0,
                    color=CF_COLORS.get(lab, toulouse_color),
                    edgecolor="white", linewidth=0.4, label=lab)
        lo = (min(0.0, float(np.nanmin(wide.to_numpy()))) * 1.05 if units == "pct"
              else 0.0) if xmin is None else xmin
    else:
        for yi, (_, row) in enumerate(wide.iterrows()):
            v = row.to_numpy(dtype=float)
            if np.isfinite(v).any():
                ax.plot([np.nanmin(v), np.nanmax(v)], [yi, yi], color="0.8",
                        linewidth=1.0, zorder=1)
        for lab in labels:
            ax.plot(wide[lab].to_numpy(), y, linestyle="none", marker="o", markersize=5,
                    color=CF_COLORS.get(lab, toulouse_color), label=lab, zorder=2)
        span = float(np.nanmax(wide.to_numpy()) - np.nanmin(wide.to_numpy()))
        lo = (np.nanmin(wide.to_numpy()) - 0.05 * span) if xmin is None else xmin
    hi = (np.nanmax(wide.to_numpy()) * 1.02 if units == "km"
          else float(np.nanmax(wide.to_numpy())) + 0.05 * (
              float(np.nanmax(wide.to_numpy())) - float(np.nanmin(wide.to_numpy())) + 1e-9)
          ) if xmax is None else xmax
    if units == "pct":
        ax.axvline(0.0, color="0.35", linewidth=1.0, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(names.astype(str))
    ax.set_ylim(-0.6, len(wide) - 0.4)
    ax.set_xlim(lo, hi)
    if units == "pct":
        ax.set_xlabel(f"Change in the mean distance travelled by the upstream euro "
                      f"(% of {baseline})")
    else:
        ax.set_xlabel("Mean distance travelled by the upstream euro (km)")
    ax.set_ylabel("Commuting zone")
    ax.grid(alpha=0.2, axis="x")
    # the level the percentages are taken against, so the figure never hides the km
    if units == "pct" and annotate_km and base_km is not None:
        for yi, idx in enumerate(wide.index):
            ax.annotate(f"{base_km.loc[idx]:.0f} km", xy=(1.005, yi),
                        xycoords=("axes fraction", "data"), va="center", ha="left",
                        fontsize=7.5, color="0.35", annotation_clip=False)
        ax.annotate(f"{baseline}\n(km)", xy=(1.005, 1.005),
                    xycoords="axes fraction", va="bottom", ha="left",
                    fontsize=7.5, color="0.35", annotation_clip=False)
    ax.legend(frameon=False, fontsize=9, ncol=min(3, max(len(labels), 1)),
              loc="lower left", bbox_to_anchor=(0.0, 1.005))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        fig.savefig(save_to, bbox_inches="tight")
    return ax


def plot_counterfactual_profile(data, regimes=CF_REGIMES,
                                radii=(25, 50, 75, 100, 150, 200, 300, 500),
                                mark=None, value_col="share", frames=None,
                                diffusion=None, figsize=None, save_to=None):
    """
    The whole curve rather than one radius: $L_r(d)$ against $d$, median across shocked
    regions, one line per regime.

    A single radius can flatter a regime that happens to cross it steeply; the curves
    say whether a gap at 100 km is a level difference that persists or a crossing that
    closes by 200 km. The vertical distance from `Both forces` to each switched-off
    regime is what that force was contributing at every radius, and the SIGN of the first
    gap is what separates the two industries: where `Distance only` sits ABOVE `Both
    forces`, comparative advantage is sending the shock further away than proximity alone
    would.
    """
    mark = _DEFAULT_RADII if mark is None else mark
    base = build_diffusion_frame(data, value_col) if diffusion is None else diffusion
    frames = (counterfactual_frames(data, regimes=regimes, value_col=value_col,
                                    diffusion=base) if frames is None else frames)

    fig, ax = plt.subplots(figsize=figsize or get_figsize(wf=0.8, hf=0.6))
    curves = {"Realised": local_share_profile(data, radii=radii, value_col=value_col,
                                              diffusion=base)}
    curves.update({lab: local_share_profile(data, radii=radii, value_col=value_col,
                                            diffusion=f) for lab, f in frames.items()})
    for lab, p in curves.items():
        ax.plot(p.index, p["median"], marker="o", markersize=4,
                color=CF_COLORS.get(lab, "0.3"),
                linestyle="--" if lab == "Realised" else "-",
                linewidth=1.6 if lab != "Realised" else 1.2, label=lab)
    for m in mark:
        ax.axvline(m, color="0.6", linewidth=0.9, linestyle=":")
        ax.annotate(f"{m:g} km", (m, 0.99), fontsize=8, color="0.4", ha="left",
                    va="top", xytext=(3, 0), textcoords="offset points")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Radius around the shocked region (km)")
    ax.set_ylabel("Share within the radius (median region)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _despine(ax)
    fig.tight_layout()
    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        fig.savefig(save_to, bbox_inches="tight")
    return ax

def amplification_report(data, radii=None, value_col="share", out_folder=None, show=True,
                         counterfactual=True, regimes=CF_REGIMES):
    """
    Everything in this section for one industry, in one call: the two Figure-6 panels
    (the second carrying every radius on one bar), the radius sweep, the joint scatter,
    the summary table, and the same shock propagated with one force switched off.

    Returns `(summary, profile)` so the numbers can be reused without recomputing.
    `radii[0]` is the headline radius — the one the scatter and the counterfactual bars
    use, and the one the nested bars sort on.

    The counterfactual block needs `best_params` and the estimated geometry. If the run
    tree carries no parameter vector, or the fit is a binned trade cost (where there is
    no single distance elasticity to switch off), it says so and the rest of the section
    is unaffected.
    """
    radii = _DEFAULT_RADII if radii is None else tuple(radii)
    diff = build_diffusion_frame(data, value_col)
    summ = amplification_summary(data, radii=radii, value_col=value_col, diffusion=diff)
    prof = local_share_profile(data, value_col=value_col, diffusion=diff)

    ind, mu = data["industry"], data["mu"]
    tag = lambda name: (None if out_folder is None
                        else f"{out_folder}/{name}_{ind}_mu{mu}.pdf")

    # Does the economy carry N_hat_s varieties? Everything below is built on the
    # parquet, so a wrong variety count is not a detail of the input -- it is a
    # different model, and it is checked before any of it is reported.
    print(f"[{ind}, mu = {mu}]  variety count in suppliers.parquet:")
    try:
        # Only meaningful against a parquet: a simulated economy is drawn at N_hat_s
        # by construction, so the check is vacuous there and is skipped rather than
        # reported as a pass it cannot fail.
        if isinstance(data.get("suppliers_path"), (str, Path)) and \
                not str(data.get("suppliers_path", "")).startswith("<simulated"):
            supplier_count_check(data)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"  variety-count check skipped: {type(e).__name__}: {e}")

    print(f"[{ind}, mu = {mu}]  mean D_r = {summ['amplification'].mean():.3f} "
          f"(one euro of demand -> {summ['upstream_sales'].mean():.3f} upstream), "
          f"p90 = {summ['amplification'].quantile(0.9):.3f}")
    for d in radii:
        c = f"share_within_{int(d)}km"
        print(f"                 share of upstream sales within {d:g} km: "
              f"mean {summ[c].mean():.3f}, median {summ[c].median():.3f}, "
              f"min {summ[c].min():.3f}, max {summ[c].max():.3f}")

    # what the finite variety count costs, quantity by quantity
    try:
        granularity_report(data, radii=radii, value_col=value_col)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"  granularity report skipped: {type(e).__name__}: {e}")

    plot_amplification(data, summary=summ, save_to=tag("amplification"))
    if show:
        plt.show()
    plot_local_share(data, radii=radii, summary=summ, save_to=tag("local_share"))
    if show:
        plt.show()
    try:
        plot_local_share_map(data, radius_km=radii[0], summary=summ,
                             save_to=tag("local_share_map"))
        if show:
            plt.show()
        # the SAME map at the wider radius the empirical part works with. It needs its
        # own pinned scale: L_r is a CDF in d, so every value is larger at 200 km and
        # the 100 km window would clip the top of the aerospace distribution.
        if len(radii) > 1:
            plot_local_share_map(data, radius_km=radii[-1], summary=summ,
                                 vlim=LOCAL_SHARE_MAP_VLIM_WIDE,
                                 save_to=tag(f"local_share_map{int(radii[-1])}km"))
            if show:
                plt.show()
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"  local-share map skipped: {type(e).__name__}: {e}")
    plot_local_share_profile(data, profile=prof, mark=radii,
                             save_to=tag("local_share_profile"))
    if show:
        plt.show()
    plot_amplification_vs_local(data, radius_km=radii[0], summary=summ,
                                save_to=tag("amplification_vs_local"))
    if show:
        plt.show()

    if counterfactual:
        try:
            frames = counterfactual_frames(data, regimes=regimes, value_col=value_col,
                                           diffusion=diff)
            det = counterfactual_amplification(data, regimes=regimes, radii=radii,
                                               value_col=value_col, frames=frames,
                                               diffusion=diff)
            plot_counterfactual_local_share(data, regimes=regimes, radius_km=radii[0],
                                            detail=det,
                                            save_to=tag("counterfactual_local_share"))
            if show:
                plt.show()
            # the same figure at the wider radius the reduced-form part works with
            if len(radii) > 1:
                plot_counterfactual_local_share(
                    data, regimes=regimes, radius_km=radii[-1], detail=det,
                    save_to=tag(f"counterfactual_local_share{int(radii[-1])}km"))
                if show:
                    plt.show()
            plot_counterfactual_profile(data, regimes=regimes, mark=radii, frames=frames,
                                        diffusion=diff,
                                        save_to=tag("counterfactual_profile"))
            if show:
                plt.show()
            plot_distance_distribution(data, regimes=regimes, detail=det,
                                       save_to=tag("counterfactual_distance"))
            if show:
                plt.show()
            # Two alternative readings of the SAME reallocation, drawn beside the two
            # above so the choice for the paper is made on the figures, not in the
            # abstract: d_r buyer by buyer (against the local-share bars) and the
            # density of d_r across buyers (against its cumulative curve).
            #
            # Each gets its OWN guard rather than riding the block's single try, so a
            # failure in one cannot silently take the other -- or be mistaken for the
            # figure never having been added. The block's outer handler prints one note
            # for everything after the point of failure, which is exactly the case where
            # a newly added figure looks absent rather than broken.
            for _lab, _fn, _stem in (
                    ("mean sourcing distance by region", plot_counterfactual_distance,
                     "counterfactual_distance_region"),
                    ("mean sourcing distance histogram", plot_distance_histogram,
                     "counterfactual_distance_hist")):
                try:
                    _fn(data, regimes=regimes, detail=det, save_to=tag(_stem))
                    if show:
                        plt.show()
                except (FileNotFoundError, KeyError, ValueError) as e:
                    print(f"  {_lab} skipped: {type(e).__name__}: {e}")
            print("\n  the shock with one force switched off "
                  "(D_r is the same in every regime by construction):")
            print(counterfactual_summary(data, regimes=regimes, radii=radii,
                                         detail=det).round(3).to_string())
            print("\n  the distribution of d_r behind that mean, and how each force "
                  "shifts it quantile by quantile:")
            print(distance_distribution_report(data, regimes=regimes,
                                               detail=det).to_string())

            # WHERE the response lands, which both the mean distance and the barycentre
            # reduce away. Own guard, for the reason given at the two figures above: a
            # newly added block that fails should say so rather than look absent.
            try:
                inc = incidence_concentration_overlap(data, regimes=regimes,
                                                      frames=frames, diffusion=diff)
                incidence_summary(inc)
                summ.attrs["incidence"] = inc
                distance_normalisation(data, regimes=regimes, detail=det, diffusion=diff)
                # The two coordinates above are computed on `rho`, which carries no
                # N_s, so they describe the AVERAGE network. `n_eff` is convex and
                # `tv_common` is convex the other way, so a single draw of the
                # finite-variety economy is strictly MORE concentrated and strictly
                # MORE buyer-specific. Measure the gap rather than caveat it.
                try:
                    summ.attrs["realised_incidence"] = realised_incidence(
                        data, expected=inc)
                except (FileNotFoundError, KeyError, ValueError) as e:
                    print(f"  realised incidence skipped: {type(e).__name__}: {e}")
                plot_destination_composition(data, diffusion=diff,
                                             save_to=tag("destination_composition"))
                if show:
                    plt.show()
            except (FileNotFoundError, KeyError, ValueError) as e:
                print(f"  incidence concentration/overlap skipped: "
                      f"{type(e).__name__}: {e}")
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"  counterfactual propagation skipped: {type(e).__name__}: {e}")
    return summ, prof


def distance_distribution(data, regimes=CF_REGIMES, value_col="share", detail=None,
                          frames=None, diffusion=None, include_realised=True):
    """
    The mean sourcing distance of each shocked downstream region, regime by regime.

    `L_r(d)` answers how much of a shock stays near its origin; this answers how far the
    average euro travels. They are two readings of the same reallocation and they need
    not move together — mass taken off a distant hub can land partly next door and partly
    at an intermediate distance, raising the local share while leaving the mean flat.
    Reported as a distribution ACROSS shocked regions rather than as one number, because
    a force can shift the whole distribution or only its tail, and the mean alone cannot
    tell the two apart.
    """
    det = (counterfactual_amplification(data, regimes=regimes, value_col=value_col,
                                        frames=frames, diffusion=diffusion, verbose=False)
           if detail is None else detail)
    out = det["mean_upstream_distance"].unstack("regime").dropna(how="all")
    if not include_realised and "Realised" in out.columns:
        out = out.drop(columns=["Realised"])
    return out


def relative_displacement(data, regimes=CF_REGIMES, value_col="share", detail=None,
                          frames=None, diffusion=None, baseline="Both forces",
                          quantiles=(0.1, 0.25, 0.5, 0.75, 0.9)):
    """
    Each buyer's sourcing distance as a PERCENTAGE deviation from the estimated
    allocation, summarised across buyers — the numbers behind
    `plot_counterfactual_distance`, which is drawn on exactly that scale.

    `distance_distribution_report` differences the QUANTILES of `d_r`; this differences
    each BUYER and then takes quantiles of the differences. The two are not the same
    object and answer different questions: a quantile shift describes how the
    distribution moves, and says nothing about any buyer, because the buyer sitting at
    the median under one regime need not be the one sitting there under another. Only
    the per-buyer version supports a sentence of the form "half the buyers move each
    way, and the ones that move, move by x per cent".

    Percentages rather than kilometres because every `d_r` lies between roughly 250 and
    450 km, so the level — common across buyers and carrying no information — swamps the
    movement; and because a relative deviation is comparable across buyers facing
    different geographies, which a kilometre is not.

    `mean_abs_pct` is the size of the reallocation a near-zero average can hide: it is
    the average distance moved regardless of direction, so a figure far above
    `|mean_pct|` is a cancellation of large movements rather than an absence of them.
    """
    det = (counterfactual_amplification(data, regimes=regimes, value_col=value_col,
                                        frames=frames, diffusion=diffusion, verbose=False)
           if detail is None else detail)
    d = det["mean_upstream_distance"].unstack("regime").dropna(how="all")
    if baseline not in d.columns:
        raise KeyError(f"no {baseline!r} column to difference against; have {list(d.columns)}")
    pct = 100.0 * (d.div(d[baseline], axis=0) - 1.0).drop(columns=[baseline])
    # rows carry the commuting-zone NAME, not the model index: the per-buyer table exists
    # to be read buyer by buyer, and an integer index cannot be.
    if "region" in det.columns:
        name = det["region"].groupby(level=0).first()
        pct.index = [name.get(i, i) for i in pct.index]
    q = pct.quantile(list(quantiles)).T
    q.columns = [f"p{int(100 * x)}" for x in quantiles]
    out = q.assign(mean_pct=pct.mean(),
                   mean_abs_pct=pct.abs().mean(),
                   max_abs_pct=pct.abs().max(),
                   share_further=(pct > 0).mean(),
                   n_buyers=len(pct))
    out.attrs["per_buyer_pct"] = pct
    return out.round(2)


def distance_distribution_report(data, regimes=CF_REGIMES, value_col="share",
                                 detail=None, frames=None, diffusion=None,
                                 quantiles=(0.1, 0.25, 0.5, 0.75, 0.9)):
    """
    The numbers behind `plot_distance_distribution`: the quantiles of `d_r` across
    shocked regions under each regime, and the SHIFT of each quantile against
    `Both forces`.

    The shift profile is what distinguishes a force that moves every region from one
    that moves only the remote ones — a mean cannot tell them apart, and the identity
    \eqref{eq:alignment} says why it matters: the movement of `d_r` is the alignment
    covariance AT THAT BUYER, so a flat shift profile means the covariance is common
    across buyers and a steep one means it is concentrated. Also reports the share of
    regions whose distance moves by less than 5 km, i.e. the regions a force leaves
    where they were.
    """
    dist = distance_distribution(data, regimes=regimes, value_col=value_col,
                                 detail=detail, frames=frames, diffusion=diffusion)
    q = dist.quantile(list(quantiles))
    q.index = [f"p{int(100 * x)}" for x in q.index]
    out = q.T
    if "Both forces" in out.index:
        base = out.loc["Both forces"]
        shift = out.subtract(base, axis=1)
        shift.index = [f"{i} - Both forces" for i in shift.index]
        out = pd.concat([out, shift.drop(index="Both forces - Both forces")])
        d = dist.subtract(dist["Both forces"], axis=0)
        out["n_regions"] = len(dist)
        out["share_moved_lt_5km"] = [
            float((d[c].abs() < 5).mean()) if c in d else np.nan
            for c in [i.split(" - ")[0] for i in out.index]]
        out["share_moved_away"] = [
            float((d[c] > 0).mean()) if c in d else np.nan
            for c in [i.split(" - ")[0] for i in out.index]]
    return out.round(2)


def plot_distance_distribution(data, regimes=CF_REGIMES, value_col="share", detail=None,
                               frames=None, diffusion=None, include_realised=True,
                               figsize=None, save_to=None):
    """
    How far the shock travels, and how each force moves that.

    One empirical CDF per regime over the shocked downstream regions: the share of
    regions whose average upstream euro travels no further than `d`. A force that shifts
    the whole curve moves every region; a force that only bends its tail moves the
    remote ones. Medians are marked on the axis, so the level shift is readable beside
    the shape change.

    Read against `Both forces`, not against the realised draw: the realised curve is one
    finite-variety draw of the same economy and its distance from `Both forces` is
    granularity, not a force.
    """
    dist = distance_distribution(data, regimes=regimes, value_col=value_col, detail=detail,
                                 frames=frames, diffusion=diffusion,
                                 include_realised=include_realised)
    fig, ax = plt.subplots(figsize=figsize or get_figsize(wf=0.8, hf=0.6))
    for lab in dist.columns:
        v = np.sort(dist[lab].dropna().to_numpy())
        if v.size == 0:
            continue
        ax.step(v, np.arange(1, v.size + 1) / v.size, where="post",
                color=CF_COLORS.get(lab, "0.3"),
                linestyle="--" if lab == "Realised" else "-",
                linewidth=1.2 if lab == "Realised" else 1.6, label=lab)
        ax.plot([np.median(v)], [0.5], marker="o", markersize=5,
                color=CF_COLORS.get(lab, "0.3"), zorder=3)
    ax.axhline(0.5, color="0.8", linewidth=0.8, zorder=0)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean distance travelled by the upstream euro (km)")
    ax.set_ylabel("Share of shocked regions")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _despine(ax)
    fig.tight_layout()
    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        fig.savefig(save_to, bbox_inches="tight")
    return ax


def plot_distance_histogram(data, regimes=CF_REGIMES, value_col="share", detail=None,
                            frames=None, diffusion=None, include_realised=True,
                            bins=12, layout="overlay", figsize=None, save_to=None):
    """
    The density companion of `plot_distance_distribution`: the same `d_r` values, drawn
    as a histogram rather than a cumulative curve.

    The CDF is the better object for reading a SHIFT (two curves, the horizontal gap is
    the effect at every quantile); the histogram is the better object for reading a
    SHAPE — whether a regime is unimodal, whether it splits the cross-section in two,
    where the mass actually sits. Both are drawn from the identical vector, so they
    cannot disagree; they only make different features legible.

    Two things are fixed rather than left to matplotlib, because both would otherwise
    invalidate the comparison. The bin EDGES are computed once on the POOLED values
    across every regime and reused for all of them — per-regime edges would put each
    curve on its own grid and the overlay would compare nothing. And the histogram is
    drawn as a STEP outline, not filled bars: four filled series overlap and the one
    drawn last wins, which silently hides whichever regime happens to be plotted first.

    `layout="overlay"` puts every regime on one panel (the direct comparison);
    `layout="panels"` gives each its own row on a shared axis, which is the readable
    choice when the curves cross. Medians are marked in both.

    CAVEAT worth weighing before this replaces the CDF in the paper: there are only as
    many observations as there are shocked commuting zones (of order twenty), so the
    shape is sensitive to `bins` in a way the CDF is not — the CDF uses every point
    exactly, a histogram of twenty points does not. Vary `bins` before believing a mode.
    """
    dist = distance_distribution(data, regimes=regimes, value_col=value_col, detail=detail,
                                 frames=frames, diffusion=diffusion,
                                 include_realised=include_realised)
    labels = [c for c in dist.columns if dist[c].notna().any()]
    if not labels:
        raise ValueError("no regime carries a finite mean sourcing distance.")
    if layout not in ("overlay", "panels"):
        raise ValueError(f"layout must be 'overlay' or 'panels', got {layout!r}.")

    pooled = dist[labels].to_numpy(dtype=float).ravel()
    pooled = pooled[np.isfinite(pooled)]
    edges = np.histogram_bin_edges(pooled, bins=bins)   # ONE grid for every regime

    def _draw(ax, lab):
        v = dist[lab].dropna().to_numpy(dtype=float)
        col = CF_COLORS.get(lab, "0.3")
        ax.hist(v, bins=edges, histtype="step",
                linestyle="--" if lab == "Realised" else "-",
                linewidth=1.2 if lab == "Realised" else 1.6, color=col, label=lab)
        ax.axvline(np.median(v), color=col, linewidth=1.0, alpha=0.55, linestyle=":")

    if layout == "overlay":
        fig, ax = plt.subplots(figsize=figsize or get_figsize(wf=0.8, hf=0.6))
        for lab in labels:
            _draw(ax, lab)
        ax.set_xlabel("Mean distance travelled by the upstream euro (km)")
        ax.set_ylabel("Number of shocked regions")
        ax.legend(frameon=False, fontsize=8)
        _despine(ax)
        axes = ax
    else:
        fig, axs = plt.subplots(len(labels), 1, sharex=True, sharey=True,
                                figsize=figsize or get_figsize(wf=0.8,
                                                               hf=0.28 * len(labels)))
        axs = np.atleast_1d(axs)
        for ax, lab in zip(axs, labels):
            _draw(ax, lab)
            ax.set_ylabel(lab, fontsize=8, rotation=0, ha="right", va="center")
            _despine(ax)
        axs[-1].set_xlabel("Mean distance travelled by the upstream euro (km)")
        axes = axs
    fig.tight_layout()
    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        fig.savefig(save_to, bbox_inches="tight")
    return axes


# --- Where the sourcing sits: the barycentre of the shock --------------------
#
# The mean sourcing distance is a RADIUS: it says how far the average euro travels
# and nothing about where it goes. Two regions with the same mean distance can send
# their euros in opposite directions, and a region whose sourcing is spread evenly
# around it is indistinguishable, on that statistic, from one that sends everything
# to a single hub 300 km away. The barycentre is the vector the radius is the length
# of: the upstream-sales-weighted mean POSITION of a shock's suppliers,
#
#     b_r = sum_l w_lr x_l / sum_l w_lr,     w_lr = X_lr,
#
# with `x_l` the centroid of commuting zone l. Drawn as an arrow from the shocked
# zone's own centre to that point, it reads directly: a short arrow is a region that
# sources around itself, and a field of long arrows CONVERGING on one point names
# that point as a sourcing hub — which is the thing no scalar in this section reports.
#
# Two conventions worth stating, because they set what the arrow means.
#
# The barycentre is computed and drawn in a PROJECTED metric CRS (Lambert-93 by
# default), not in longitude/latitude. A weighted mean of degrees is not a mean of
# positions — a degree of longitude is ~74 km at Dunkirk and ~82 km at Perpignan —
# so averaging in lon/lat would tilt every arrow westward-of-true by an amount that
# grows with how far north the sourcing sits. Distances in kilometres are then plain
# Euclidean distances in that projection.
#
# And the arrow is NOT the mean distance. `|b_r - x_r|` is the length of the mean
# displacement vector, which is short exactly when the sourcing is directionally
# balanced; the mean distance is the mean of the lengths, which is not. The gap
# between the two IS the dispersion of directions, and `sourcing_barycentre` reports
# both so a short arrow over a large mean distance is read as symmetry rather than as
# proximity.

BARYCENTRE_CRS = "EPSG:2154"        # Lambert-93: metres, France-wide, area-true enough


def _zone_coordinates(data, crs=BARYCENTRE_CRS):
    """
    The centroid of every commuting zone, in metres, indexed by the MODEL's region
    index 1..R.

    Returns a frame with `x`, `y`, `ze2010` and `ze2010_name`. Zones the geometry does
    not cover come back with NaN coordinates rather than being dropped here — the
    caller decides what to do about them, since a missing zone is a missing WEIGHT and
    that is a fact about the barycentre, not about the table.
    """
    fr = data.get("france")
    if fr is None or not hasattr(fr, "geometry"):
        raise FileNotFoundError(
            "no commuting-zone GEOMETRY in `data['france']` (france.gpkg missing, or "
            "geopandas is not installed) — the barycentre map cannot be drawn.")
    geo = fr.copy()
    geo["ze2010"] = _as_ze_string(geo["ze2010"])
    if crs is not None and getattr(geo, "crs", None) is not None:
        geo = geo.to_crs(crs)
    cen = geo.geometry.centroid
    pts = (pd.DataFrame({"ze2010": geo["ze2010"].to_numpy(),
                         "x": np.asarray(cen.x), "y": np.asarray(cen.y)})
           .drop_duplicates("ze2010").set_index("ze2010"))
    lab = _region_labels(data)
    out = lab.join(pts, on="ze2010").set_index("index")
    out.attrs["crs"] = crs
    return out[["ze2010", "ze2010_name", "x", "y"]]


def sourcing_barycentre(data, diffusion=None, value_col="share", coords=None,
                        crs=BARYCENTRE_CRS, verbose=False):
    """
    One row per shocked downstream region: where its own centre is, where the
    barycentre of its sourcing is, and how far apart the two are.

    `displacement_km` is the length of the MEAN DISPLACEMENT VECTOR and
    `mean_distance_km` the mean of the individual distances. Their ratio
    (`concentration`) is the directional concentration of the sourcing: 1 when every
    euro leaves along the same bearing, near 0 when the pull cancels in all
    directions. A region can have a large mean distance and a barycentre sitting on
    top of itself, and the two columns are reported together so that case is legible
    rather than surprising.

    Both are measured in the SAME projected geometry, as straight-line distances
    between centroids. That is what makes the ratio a well-defined concentration
    (Jensen bounds it by one). The model's own distance matrix — the object
    `amplification_summary` averages, and which need not be a centroid distance — is
    carried alongside as `mean_model_distance_km` rather than mixed into the ratio.

    Zones the geometry does not cover cannot be placed, so their euros cannot enter a
    weighted mean of positions; they are excluded and `weight_covered` records the
    share of each region's upstream sales that survived, so a barycentre computed on
    half the euros is visible rather than silently reported as the whole.
    """
    df = (build_diffusion_frame(data, value_col) if diffusion is None else diffusion)
    co = _zone_coordinates(data, crs=crs) if coords is None else coords

    x = co["x"].reindex(df["ze2010"].to_numpy()).to_numpy(dtype=float)
    y = co["y"].reindex(df["ze2010"].to_numpy()).to_numpy(dtype=float)
    w = df["upstream_sales"].to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)

    g = df["ze2010_downstream"].to_numpy()
    tot = pd.Series(w).groupby(g).sum()
    wok = pd.Series(np.where(ok, w, 0.0))
    den = wok.groupby(g).sum()
    dist = df["distance"].to_numpy(dtype=float)

    out = pd.DataFrame({
        "region": df.groupby("ze2010_downstream")["shocked_name"].first(),
        "upstream_sales": tot,
        "weight_covered": (den / tot.replace(0, np.nan)),
        "bary_x": pd.Series(np.where(ok, w * x, 0.0)).groupby(g).sum() / den.replace(0, np.nan),
        "bary_y": pd.Series(np.where(ok, w * y, 0.0)).groupby(g).sum() / den.replace(0, np.nan),
        "mean_model_distance_km": pd.Series(w * dist).groupby(g).sum() / tot.replace(0, np.nan),
    })
    out.index.name = "ze2010_downstream"
    out["origin_x"] = co["x"].reindex(out.index).to_numpy(dtype=float)
    out["origin_y"] = co["y"].reindex(out.index).to_numpy(dtype=float)
    # the mean of the LENGTHS, in the same geometry the barycentre is taken in
    ox = out["origin_x"].reindex(g).to_numpy(dtype=float)
    oy = out["origin_y"].reindex(g).to_numpy(dtype=float)
    leg = np.hypot(x - ox, y - oy) / 1000.0
    out["mean_distance_km"] = (pd.Series(np.where(ok, w * leg, 0.0)).groupby(g).sum()
                               / den.replace(0, np.nan))
    out["dx_km"] = (out["bary_x"] - out["origin_x"]) / 1000.0
    out["dy_km"] = (out["bary_y"] - out["origin_y"]) / 1000.0
    out["displacement_km"] = np.hypot(out["dx_km"], out["dy_km"])
    # bearing of the pull, degrees clockwise from north — the direction a reader
    # would name ("north-east"), not the mathematical angle from the x axis
    out["bearing_deg"] = (np.degrees(np.arctan2(out["dx_km"], out["dy_km"])) + 360.0) % 360.0
    out["concentration"] = out["displacement_km"] / out["mean_distance_km"].replace(0, np.nan)
    out.attrs["value_col"] = value_col
    out.attrs["crs"] = crs

    miss = out["weight_covered"] < 0.999
    if verbose and miss.any():
        print(f"  [barycentre] {int(miss.sum())} region(s) have upstream sales in zones "
              f"the geometry does not cover; min covered weight "
              f"{out['weight_covered'].min():.3f}")
    return out


def barycentre_hubs(data, bary=None, n=10, radius_km=25, **kwargs):
    """
    Where the arrows POINT: the barycentres aggregated into hubs.

    Every shocked region's barycentre is assigned to the commuting zone it falls in
    (nearest centroid, within `radius_km`), and the zones are ranked by how many
    arrowheads they collect and by the upstream sales those arrows carry. This is the
    figure's claim written as a number — a bright convergence of arrows on the map
    should be the top row here, and if it is not, the eye was reading the arrow
    density of a dense part of the country rather than a hub.
    """
    b = sourcing_barycentre(data, **kwargs) if bary is None else bary
    co = _zone_coordinates(data, crs=b.attrs.get("crs", BARYCENTRE_CRS))
    cx, cy = co["x"].to_numpy(dtype=float), co["y"].to_numpy(dtype=float)
    rows = []
    for idx, r in b.dropna(subset=["bary_x"]).iterrows():
        d = np.hypot(cx - r["bary_x"], cy - r["bary_y"]) / 1000.0
        j = int(np.nanargmin(d))
        rows.append({"hub_index": co.index[j], "hub": co["ze2010_name"].iloc[j],
                     "distance_to_centroid_km": d[j], "from": r["region"],
                     "sales": r["upstream_sales"]})
    hits = pd.DataFrame(rows)
    hits = hits[hits["distance_to_centroid_km"] <= radius_km]
    out = (hits.groupby(["hub_index", "hub"])
               .agg(n_regions=("from", "size"), sales=("sales", "sum"))
               .reset_index().set_index("hub"))
    out["share_of_regions"] = out["n_regions"] / len(b)
    return out.sort_values("n_regions", ascending=False).head(n)


def plot_sourcing_barycentre(data, bary=None, value_col="share", diffusion=None,
                             coords=None, crs=BARYCENTRE_CRS, color=None,
                             min_km=0.0, ax=None, figsize=None, title=None,
                             save_to=None, annotate=True, ground_color="0.93",
                             edge_color="white"):
    """
    The barycentre map: one arrow per shocked commuting zone, from its own centre to
    the barycentre of its sourcing.

    Arrows are drawn at TRUE scale in the projected CRS (`angles="xy"`,
    `scale_units="xy"`, `scale=1`), so an arrow's length is the displacement in
    kilometres read off the same axes as the map itself. That is the whole point of
    the figure and there is deliberately no exaggeration factor: a stretched arrow
    field would make a diffuse sourcing pattern look like a hub.

    `min_km` drops arrows shorter than that, which is a readability device only — the
    dropped regions are the ones whose sourcing is centred on themselves, and the
    count is annotated rather than left implicit.
    """
    b = (sourcing_barycentre(data, diffusion=diffusion, value_col=value_col,
                             coords=coords, crs=crs) if bary is None else bary)
    fr = data.get("france")
    if fr is None or not hasattr(fr, "geometry"):
        raise FileNotFoundError(
            "no commuting-zone GEOMETRY in `data['france']` — the map cannot be drawn.")
    geo = fr.copy()
    if crs is not None and getattr(geo, "crs", None) is not None:
        geo = geo.to_crs(crs)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize or get_figsize(wf=0.5, hf=1.0))
    geo.plot(facecolor=ground_color, ax=ax, edgecolor=edge_color, linewidth=0.25)

    v = b.dropna(subset=["bary_x", "origin_x"])
    short = int((v["displacement_km"] < min_km).sum())
    v = v[v["displacement_km"] >= min_km]
    col = toulouse_color if color is None else color
    ax.quiver(v["origin_x"].to_numpy(), v["origin_y"].to_numpy(),
              (v["bary_x"] - v["origin_x"]).to_numpy(),
              (v["bary_y"] - v["origin_y"]).to_numpy(),
              angles="xy", scale_units="xy", scale=1.0, color=col,
              width=0.0035, headwidth=4.0, headlength=5.0, headaxislength=4.2,
              zorder=3)

    xmin, ymin, xmax, ymax = geo.total_bounds
    pad = 0.02 * max(xmax - xmin, ymax - ymin)
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect("equal")
    ax.set_axis_off()
    if title is not None:
        ax.set_title(title, fontsize=10)
    if annotate:
        txt = (f"median pull {v['displacement_km'].median():.0f} km"
               + (f"\n{short} region(s) below {min_km:g} km not drawn" if short else ""))
        ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=8, va="bottom",
                ha="left", color="0.25")
    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        ax.figure.savefig(save_to, bbox_inches="tight")
    return ax


def barycentre_panel(datasets, regimes=CF_REGIMES, value_col="share",
                     crs=BARYCENTRE_CRS, min_km=0.0, figsize=None, save_to=None,
                     save_each=None, include_realised=True, verbose=False):
    """
    The barycentre map for every regime and both industries in one figure: one ROW per
    scenario, one COLUMN per industry (`datasets` order — motor vehicles then
    aerospace as the section runs them).

    `datasets` is a sequence of `(column title, data)` pairs. The realised draw is the
    first row and the three closed-form regimes follow, so the reader compares each
    regime against the estimated economy directly above the fold rather than against
    the other industry.

    Every panel is drawn in the same projected CRS at the same true arrow scale and
    the axes are pinned to the same window, so an arrow in one panel is directly
    comparable with an arrow in any other — which is the only reason a grid of maps is
    worth more than four separate ones.

    `save_each` is a format string taking `industry` and `regime` (already slugged,
    e.g. `f"{out}/barycentre_{{industry}}_{{regime}}.pdf"`): every panel is ALSO written
    as its own figure, so a single scenario can be used on its own without the other
    seven coming with it. Those standalone maps are the same object drawn with the same
    scale, not a redrawn variant.

    Returns `(fig, barycentres)` with `barycentres` keyed by `(column title, regime)`,
    so the numbers behind every panel are available without recomputing them.
    """
    labels = (["Realised"] if include_realised else []) + list(regimes)
    ncol, nrow = len(datasets), len(labels)
    fig, axes = plt.subplots(nrow, ncol, squeeze=False,
                             figsize=figsize or get_figsize(wf=1.0, hf=0.30 * nrow * 2 / max(ncol, 1)))

    out = {}
    for j, (name, data) in enumerate(datasets):
        base = build_diffusion_frame(data, value_col)
        coords = _zone_coordinates(data, crs=crs)
        frames = {"Realised": base} if include_realised else {}
        frames.update(counterfactual_frames(data, regimes=regimes, value_col=value_col,
                                            diffusion=base, verbose=verbose))
        for i, lab in enumerate(labels):
            b = sourcing_barycentre(data, diffusion=frames[lab], value_col=value_col,
                                    coords=coords, crs=crs)
            out[(name, lab)] = b
            ax = axes[i][j]
            plot_sourcing_barycentre(data, bary=b, crs=crs, coords=coords, ax=ax,
                                     color=CF_COLORS.get(lab, "0.3"), min_km=min_km,
                                     title=name if i == 0 else None)
            if save_each:
                slug = lambda t: re.sub(r"[^a-z0-9]+", "_", str(t).lower()).strip("_")
                single, sax = plt.subplots(figsize=get_figsize(wf=0.5, hf=1.0))
                plot_sourcing_barycentre(data, bary=b, crs=crs, coords=coords, ax=sax,
                                         color=CF_COLORS.get(lab, "0.3"), min_km=min_km,
                                         title=f"{name} — {lab}",
                                         save_to=save_each.format(industry=slug(name),
                                                                  regime=slug(lab)))
                plt.close(single)
            if j == 0:
                # the row label goes on the axes, not in a title: a title on every panel
                # would repeat the industry name four times and the scenario name twice
                ax.text(-0.04, 0.5, lab, transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=9)
    fig.tight_layout()
    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        fig.savefig(save_to, bbox_inches="tight")
    return fig, out


# --- Do the barycentres FOLLOW their buyers? --------------------------------
#
# "Are the arrowheads clustered?" is not a question a map can settle: a tight
# cluster is what ANY diffuse sourcing pattern produces, because a weighted mean
# of positions over a near-national distribution sits near the national centre of
# mass whatever its tail does. The question with content is whether a buyer's
# barycentre MOVES WITH the buyer, and that is a regression, not a picture.
#
# Write the barycentre as a shrinkage of the buyer's own position towards a common
# point c,
#
#     b_r = (1 - lambda) c + lambda x_r + e_r,
#
# and lambda is the whole diagnostic. `lambda = 1` means every buyer sources around
# itself; `lambda = 0` means the allocation is buyer-INDEPENDENT and there is one
# national barycentre; anything between is a shrunk copy of the buyer map, and the
# shrinkage factor IS the strength of the local pull.
#
# The model predicts what lambda should be, which is what makes this a test rather
# than a summary. The only buyer-specific term in rho_lr ∝ T_l d_lr^{-theta*alpha}
# is the distance factor, so theta*alpha alone governs how much a buyer's own
# location tilts its sourcing. Over a log-distance range of about 3.8 (20 to 900 km)
# the near-far attractiveness ratio is exp(theta*alpha * 3.8): 2.4 in motor vehicles
# at 0.23, 5.7 in aerospace at 0.46. So lambda should be SMALLER in motor vehicles
# --- the industry with the stronger negative alignment --- and the counterfactual
# regimes order it too: `Comparative advantage only` sets alpha = 0, which removes
# the buyer-specific term entirely and must drive lambda to zero.
#
# Two coefficients are reported and they answer different things. `slope` is the
# POOLED 2-D projection, the rotation-invariant scalar, and is the headline.
# `slope_x` / `slope_y` are the axis-wise ones, worth reading separately because
# France is not isotropic: it is longer north-south than the sourcing geography is
# free to move, and a boundary effect on one axis shows up there and not in the
# pooled number. `dispersion_ratio` is sd(b)/sd(x) computed in two dimensions; it
# equals |slope| only when the fit is exact, and the gap between them is the part of
# the barycentre's variation that does NOT track the buyer.


def barycentre_shrinkage(bary, weights=None):
    """
    Regress the sourcing barycentre on the buyer's own position: how much of a
    buyer's location does its barycentre inherit?

    `bary` is one `sourcing_barycentre` frame (or a dict of them, keyed by regime or
    by (industry, regime), in which case one row is returned per key). `weights` names
    a column of `bary` to weight regions by — `"upstream_sales"` asks the question
    about the average EURO, `None` (the default) about the average REGION, which is
    the one the map draws.

    Returns `slope` (the pooled, rotation-invariant shrinkage), `slope_x` / `slope_y`,
    `r2`, `dispersion_ratio` = sd(b)/sd(x) in two dimensions, and `n`. `r2` comes back
    NaN in the one degenerate case that matters — a barycentre identical for every
    buyer has no variation to explain — where `slope` and `dispersion_ratio` are both
    exactly zero and say the whole of it.

    It also returns `se`, `t` and `ci_lo`/`ci_hi`, WITHOUT which the slope cannot be
    read. `slope` is a weighted regression through the origin of the stacked
    coordinate deviations, so its dispersion is the ordinary sandwich
    `Var = (sum_g s_g^2) / D^2` with `D = sum_i w_i (dx_i^2 + dy_i^2)` and the score
    `s_g = w_g (dx_g e^x_g + dy_g e^y_g)` summed WITHIN a region — the two coordinates
    of one buyer are one observation, not two, and clustering on the region is what
    stops the same scatter being counted twice. A finite-sample factor `n/(n-1)` is
    applied and the interval is normal; with of order twenty shocked regions that is
    an approximation, so read the interval as an order of magnitude, not a p-value.
    The estimated weighted means are treated as known, which understates the interval
    slightly. And what the interval measures is the CROSS-SECTIONAL scatter of an
    estimated deterministic allocation across buyers — it says whether the buyers
    agree on a common shrinkage, not how precisely the model's parameters are known.
    """
    if isinstance(bary, dict):
        rows = {k: barycentre_shrinkage(v, weights=weights).iloc[0] for k, v in bary.items()}
        out = pd.DataFrame(rows).T
        out.index = (pd.MultiIndex.from_tuples(out.index)
                     if len(out) and isinstance(next(iter(rows)), tuple) else out.index)
        return out

    v = bary.dropna(subset=["bary_x", "bary_y", "origin_x", "origin_y"])
    w = (np.ones(len(v)) if weights is None
         else v[weights].to_numpy(dtype=float))
    if len(v) < 3 or w.sum() <= 0:
        raise ValueError(f"only {len(v)} placeable region(s) with positive weight — "
                         "the shrinkage regression needs at least three.")
    w = w / w.sum()

    def dev(col):
        a = v[col].to_numpy(dtype=float)
        return a - float(w @ a)

    dx, dy = dev("origin_x"), dev("origin_y")
    bx, by = dev("bary_x"), dev("bary_y")

    # pooled: the single scalar minimising the weighted |b - lambda x|^2 over BOTH
    # coordinates at once, so it does not depend on how the map happens to be rotated
    den = float(w @ (dx**2 + dy**2))
    lam = float(w @ (dx * bx + dy * by)) / den if den > 0 else np.nan
    resid = float(w @ ((bx - lam * dx) ** 2 + (by - lam * dy) ** 2))
    tot = float(w @ (bx**2 + by**2))
    # cluster-robust on the region: the x and y deviations of one buyer are ONE
    # observation, so their scores are summed before being squared
    ex, ey = bx - lam * dx, by - lam * dy
    score = w * (dx * ex + dy * ey)
    n = len(v)
    se = (np.sqrt(float(score @ score) * n / (n - 1)) / den
          if den > 0 and n > 1 else np.nan)
    return pd.DataFrame([{
        "slope": lam,
        "se": se,
        "t": lam / se if se and np.isfinite(se) and se > 0 else np.nan,
        "ci_lo": lam - 1.96 * se, "ci_hi": lam + 1.96 * se,
        "slope_x": float(w @ (dx * bx)) / float(w @ dx**2) if float(w @ dx**2) > 0 else np.nan,
        "slope_y": float(w @ (dy * by)) / float(w @ dy**2) if float(w @ dy**2) > 0 else np.nan,
        "r2": 1.0 - resid / tot if tot > 0 else np.nan,
        "dispersion_ratio": np.sqrt(tot / den) if den > 0 else np.nan,
        "median_pull_km": float(v["displacement_km"].median()),
        "median_distance_km": float(v["mean_distance_km"].median()),
        "n": int(len(v)),
    }])


def barycentre_shrinkage_report(barys, weights=None, verbose=True):
    """
    `barycentre_shrinkage` over every (industry, regime) the panel produced, with the
    reading printed beside it.

    `barys` is `barycentre_panel`'s second return value, so the table is built on the
    SAME barycentres the figure draws rather than on a recomputation that could drift.
    """
    out = barycentre_shrinkage(barys, weights=weights)
    if isinstance(out.index, pd.MultiIndex):
        out.index.names = ["industry", "regime"]
    out = out.sort_index()
    if verbose:
        print(out.round(3).to_string())
        print("  slope = how much of its own position a buyer's barycentre inherits: "
              "1 = sources around itself, 0 = one national barycentre shared by every "
              "buyer.")
        print("  The only buyer-specific term in rho is d^(-theta*alpha), so the slope "
              "is predicted to RISE with theta*alpha (0.23 auto, 0.46 aero) and to "
              "vanish under `Comparative advantage only`, where alpha = 0.")
        print("  se/t/ci are clustered on the region: they say whether the BUYERS "
              "agree on a common shrinkage, not how precisely the parameters are known.")
    return out


def barycentre_shrinkage_contrast(table, pairs=None):
    """
    Differences between two shrinkage slopes, with a standard error — the comparison
    the table itself cannot make.

    A slope of 0.065 beside one of 0.080 says nothing until the two are differenced
    against their own dispersion. `pairs` is a list of ((industry, regime),
    (industry, regime)) index pairs from `barycentre_shrinkage_report`'s table;
    the default contrasts each industry's `Both forces` against every other regime of
    the SAME industry, and the two industries against each other under `Both forces`.

    The standard error is `sqrt(se_a^2 + se_b^2)`, i.e. the two slopes are treated as
    INDEPENDENT. Across industries that is conservative — the two allocations are
    estimated on overlapping regions and their errors are, if anything, positively
    correlated, which would shrink the true standard error. WITHIN an industry it is
    conservative for the same reason and more strongly so: two regimes are the same
    geometry with one force switched off, so their slopes move together.
    """
    if not isinstance(table.index, pd.MultiIndex):
        raise TypeError("barycentre_shrinkage_contrast needs the (industry, regime) table.")
    if pairs is None:
        inds = list(dict.fromkeys(i for i, _ in table.index))
        regs = list(dict.fromkeys(r for _, r in table.index))
        base = "Both forces" if "Both forces" in regs else regs[0]
        pairs = [((i, r), (i, base)) for i in inds for r in regs if r != base]
        pairs += [((inds[k], base), (inds[0], base)) for k in range(1, len(inds))]
    rows = []
    for a, b in pairs:
        if a not in table.index or b not in table.index:
            continue
        d = float(table.loc[a, "slope"]) - float(table.loc[b, "slope"])
        se = float(np.sqrt(float(table.loc[a, "se"]) ** 2 + float(table.loc[b, "se"]) ** 2))
        rows.append({"a": " / ".join(map(str, a)), "b": " / ".join(map(str, b)),
                     "diff": d, "se": se, "t": d / se if se > 0 else np.nan,
                     "ci_lo": d - 1.96 * se, "ci_hi": d + 1.96 * se})
    return pd.DataFrame(rows)


# ============================================================================
# Where the response LANDS: incidence, concentration, commonality
# ============================================================================

# Where the response LANDS: the incidence vector, its concentration and how much of it
# is common across buyers.
#
# `sourcing_barycentre` reduces the destination distribution to its FIRST MOMENT, and a
# first moment cannot tell one dispersed central pool from two hubs: both put the mean
# near the national centre of mass, which is why every aerospace arrow lands in the
# Massif Central and neither Toulouse nor Paris appears. The object with the answer is
# the incidence vector itself,
#
#     omega_r = ( X_{l r} / sum_l X_{l r} )_l ,
#
# the share of the upstream response to a shock in buyer `r` that lands in each upstream
# region. Two of its features carry the whole question, and they are independent:
#
#   CONCENTRATION -- is the response spread over many origins or collected by a few?
#     Measured by the Herfindahl `H_r = sum_l omega_lr^2` and its reciprocal, the
#     EFFECTIVE NUMBER of destinations `1/H_r`, which is in units a reader can hold
#     ("this shock is equivalent to spreading evenly over 40 commuting zones").
#
#   COMMONALITY -- do different buyers hit the SAME origins, or does each have its own?
#     Measured against the common incidence `omega_bar = mean_r omega_r`.
#
# Two conventions matter, and both are deliberate.
#
# (1) CONCENTRATION IS READ AGAINST THE GEOMETRY, NOT IN LEVELS. `H_r` depends on how
#     many cells the sector has and how the country is shaped before either force acts,
#     so it is reported both raw and as a RATIO to the same statistic under the uniform
#     allocation (`alpha = 0` AND `T` equalised -- nothing selects among cells). The
#     ratio is the concentration the two forces ADD, which is the economic object; the
#     level is the one a referee objects to.
#
# (2) COMMONALITY IS READ AGAINST ZERO, NOT AGAINST A BENCHMARK. The natural benchmark
#     -- the uniform allocation -- is itself almost perfectly common across buyers
#     (nothing in it is buyer-specific except the sector mix), so its overlap is
#     essentially one and differencing against it would just subtract a constant. More
#     to the point, the estimated allocation is ALREADY nearly buyer-independent: the
#     barycentre shrinkage is below 0.12 in both industries, and buyer-independence IS
#     `omega_r = omega_q`. An overlap statistic in levels therefore sits near its
#     ceiling in both industries and separates nothing -- exactly the degeneracy that
#     makes the shrinkage coefficient unreadable. What is informative is the DEFICIT,
#     how much buyer-specific mass there is, and its floor of zero is meaningful.
#
#     `tv_common` is that deficit, measured in euros:
#
#         Delta_r = (1/2) sum_l | omega_lr - omega_bar_l |  in  [0, 1],
#
#     the share of the response that would have to be MOVED to make buyer `r`'s
#     incidence identical to the common one. Zero = perfectly buyer-independent, one =
#     entirely its own pocket. The cosine `cos_common` is reported beside it because it
#     is the statistic the 2x2 framing is usually written in, but the total variation is
#     the one to quote: it is bounded, it is linear in the euros, and it does not
#     reward two vectors for sharing a direction while disagreeing about magnitude.
#
# Together the two coordinates separate the three configurations that the barycentre
# cannot:
#
#                      | spread (n_eff ratio ~ 1) | concentrated (ratio << 1)
#     -----------------+--------------------------+---------------------------
#     common (Delta~0) | dispersed common pool    | HUB-AND-SPOKE
#     specific (Delta>0)| local pockets           | local monopolists

# The geometry-free benchmark: nothing selects among the cells of a sector, so the
# incidence carries only the sector mix and the modelled cell support. It is NOT a
# counterfactual economy and is never reported as one -- it exists to divide out.
UNIFORM_REGIME = "Uniform benchmark"
CF_COLORS[UNIFORM_REGIME] = (0.72, 0.72, 0.72)


def incidence_matrix(diffusion, R=None):
    """
    The incidence vectors as a (buyer x upstream region) frame whose rows sum to one.

    Built from an ALREADY zero-filled diffusion frame, so the columns are the full
    upstream support and two regimes are directly comparable entry by entry. A buyer
    generating no upstream sales at all comes back as an all-NaN row rather than a
    row of zeros, so it is dropped by the statistics instead of counting as a
    perfectly dispersed one.
    """
    if "replication" in diffusion.columns:
        raise ValueError("pass an averaged diffusion frame — the incidence vector of a "
                         "single realisation is a different object (see `granular_band`).")
    W = diffusion.pivot_table(index="ze2010_downstream", columns="ze2010",
                              values="upstream_sales", aggfunc="sum", fill_value=0.0)
    if R is not None:
        W = W.reindex(columns=np.arange(1, R + 1), fill_value=0.0)
    tot = W.sum(axis=1)
    return W.div(tot.where(tot > 0), axis=0)


def _pairwise_overlap(W):
    """
    Median pairwise cosine and median pairwise total-variation distance over the rows.

    Both are computed on every unordered pair, not on a sample: with of order twenty
    shocked regions there are a couple of hundred pairs and no reason to approximate.
    """
    A = np.asarray(W.dropna(how="any"), dtype=float)
    n = A.shape[0]
    if n < 2:
        return np.nan, np.nan
    nrm = np.linalg.norm(A, axis=1)
    C = (A @ A.T) / np.outer(np.where(nrm > 0, nrm, np.nan), np.where(nrm > 0, nrm, np.nan))
    iu = np.triu_indices(n, k=1)
    tv = np.array([0.5 * np.abs(A[i] - A[j]).sum() for i, j in zip(*iu)])
    return float(np.nanmedian(C[iu])), float(np.nanmedian(tv))


def incidence_stats(W, weights=None):
    """
    The per-buyer statistics of one regime, from its incidence matrix alone.

    Split out of `incidence_concentration_overlap` so the arithmetic can be gated on
    planted matrices whose answer needs no model: an incidence on one region must give
    `n_eff = 1`, one spread evenly over `k` must give exactly `k`, and buyers with
    identical incidence must give `tv_common = 0`.

    `weights` is a vector over the rows (buyers) defining what "common" means; `None`
    weights every buyer equally.
    """
    A = W.to_numpy(dtype=float)
    # a buyer generating no upstream sales has an undefined incidence vector; it must
    # not drag the common vector towards zero, so it is dropped from the average rather
    # than counted as a row of zeros.
    ok = np.isfinite(A).all(axis=1)
    w = ok.astype(float) if weights is None else np.where(ok, np.asarray(weights, float), 0.0)
    w = w / w.sum() if w.sum() > 0 else w
    bar = (np.nan_to_num(A, nan=0.0) * w[:, None]).sum(axis=0)
    nb_ = np.linalg.norm(bar)
    srt = -np.sort(-np.nan_to_num(A, nan=0.0), axis=1)
    own = np.array([W.loc[r, r] if r in W.columns else np.nan for r in W.index],
                   dtype=float)
    Az = np.nan_to_num(A, nan=0.0)
    nrm = np.linalg.norm(Az, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (Az @ bar) / (nrm * (nb_ if nb_ > 0 else np.nan))
    out = pd.DataFrame({
        "hhi": np.nansum(A ** 2, axis=1),
        "top1": srt[:, 0], "top2": srt[:, :2].sum(1), "top3": srt[:, :3].sum(1),
        "own_share": own,
        "tv_common": 0.5 * np.nansum(np.abs(A - bar), axis=1),
        "cos_common": cos,
    }, index=W.index)
    out[~ok] = np.nan
    out["n_eff"] = 1.0 / out["hhi"].replace(0, np.nan)
    return out


def incidence_concentration_overlap(data, regimes=CF_REGIMES, value_col="share",
                                    frames=None, diffusion=None, include_realised=True,
                                    weights=None, benchmark=True, verbose=False):
    """
    One row per (regime, shocked buyer): how concentrated the response is and how much
    of it is buyer-specific.

    Columns
    -------
    hhi, n_eff        the Herfindahl of `omega_r` and its reciprocal, the effective
                      number of destination commuting zones.
    n_eff_ratio       `n_eff` divided by the SAME buyer's `n_eff` under the uniform
                      allocation. This is the concentration the two forces add, purged
                      of how many cells the sector has and of the shape of the country.
                      Below one means the forces concentrate; one means they do not.
    top1..top3        the share collected by the largest one, two and three origins.
    own_share         the share landing in the shocked region itself.
    tv_common         (1/2)*sum_l |omega_lr - omega_bar_l|: the share of the response
                      that would have to move for this buyer's incidence to equal the
                      common one. Read against ZERO -- see the note above on why the
                      level of an overlap statistic is uninformative here.
    cos_common        the cosine against the same common vector, for readers who want
                      the 2x2 in its usual units.

    `weights` chooses what "common" means: `None` (default) gives every buyer equal
    weight, so `omega_bar` is the typical buyer's incidence; `"sales"` weights each
    buyer by the upstream response it generates, so `omega_bar` is the aggregate
    incidence of the industry. The two answer different questions and the default is
    the one the figure draws.

    The omega matrices are stashed in `.attrs["omega"]` so `incidence_summary` can
    compute the pairwise statistics without rebuilding them.

    NOTE the index order: this table is keyed `(regime, buyer)`, because every use of it
    selects a regime first. The counterfactual detail table and `distance_normalisation`
    are keyed the other way round, `(buyer, regime)`; select a regime there with `xs`,
    since a `.loc` would look up a buyer instead of failing.
    """
    base = build_diffusion_frame(data, value_col) if diffusion is None else diffusion
    want = dict(regimes)
    if benchmark and UNIFORM_REGIME not in want:
        want[UNIFORM_REGIME] = dict(alpha=0.0, equalise_T=True)
    frames = (counterfactual_frames(data, regimes=want, value_col=value_col,
                                    diffusion=base, verbose=verbose)
              if frames is None else dict(frames))
    if benchmark and UNIFORM_REGIME not in frames:
        frames[UNIFORM_REGIME] = counterfactual_diffusion_frame(
            data, value_col=value_col, diffusion=base, verbose=verbose,
            **want[UNIFORM_REGIME])

    order = (["Realised"] if include_realised else []) + \
            [r for r in regimes if r in frames] + \
            ([UNIFORM_REGIME] if benchmark else [])
    supply = {"Realised": base, **frames}

    names = base.groupby("ze2010_downstream")["shocked_name"].first()
    omegas, rows = {}, []
    for lab in order:
        if lab not in supply:
            continue
        W = incidence_matrix(supply[lab], R=data["R"])
        omegas[lab] = W
        w = None
        if weights == "sales":
            tot = supply[lab].groupby("ze2010_downstream")["upstream_sales"].sum()
            w = tot.reindex(W.index).to_numpy(dtype=float)
        st = incidence_stats(W, weights=w)
        rows.append(st.reset_index().rename(columns={"index": "ze2010_downstream"})
                      .assign(regime=lab,
                              region=names.reindex(W.index).to_numpy()))
    out = pd.concat(rows, ignore_index=True)
    out["regime"] = pd.Categorical(out["regime"], categories=order, ordered=True)
    out = out.set_index(["regime", "ze2010_downstream"]).sort_index()

    # the geometry purge, buyer by buyer rather than through one scalar: the reference
    # is the SAME buyer under the uniform allocation, so a remote buyer is compared with
    # the dispersion that buyer's own position makes available.
    if benchmark and UNIFORM_REGIME in omegas:
        ref = out.loc[UNIFORM_REGIME, "n_eff"]
        out["n_eff_ratio"] = out["n_eff"] / \
            ref.reindex(out.index.get_level_values("ze2010_downstream")).to_numpy()
    else:
        out["n_eff_ratio"] = np.nan
    out.attrs["omega"] = omegas
    out.attrs["weights"] = weights
    return out


def incidence_summary(table, verbose=True):
    """
    The buyer table reduced to one row per regime — the two coordinates of the 2x2.

    `pairwise_cos` / `pairwise_tv` are medians over every pair of buyers, i.e. how much
    two buyers picked at random have in common. They are reported beside the
    against-the-mean statistics because a low `tv_common` could in principle come from
    every buyer sitting halfway between two disjoint groups, which the pairwise median
    would expose and the against-the-mean one would not.
    """
    g = table.groupby(level="regime", observed=True)
    out = pd.DataFrame({
        "n_eff": g["n_eff"].median(),
        "n_eff_ratio": g["n_eff_ratio"].median(),
        "hhi": g["hhi"].median(),
        "top3": g["top3"].median(),
        "own_share": g["own_share"].median(),
        "tv_common": g["tv_common"].median(),
        "cos_common": g["cos_common"].median(),
    })
    om = table.attrs.get("omega", {})
    pc, pt = {}, {}
    for lab, W in om.items():
        pc[lab], pt[lab] = _pairwise_overlap(W)
    out["pairwise_cos"] = pd.Series(pc).reindex(out.index)
    out["pairwise_tv"] = pd.Series(pt).reindex(out.index)
    if verbose:
        print("  the incidence vector: how concentrated, and how much is buyer-specific")
        print("  (n_eff_ratio < 1 = the forces concentrate; tv_common is the euro share "
              "that is buyer-specific, read against 0)")
        print(out.round(3).to_string())
    return out


def distance_normalisation(data, regimes=CF_REGIMES, value_col="share", detail=None,
                           frames=None, diffusion=None, verbose=True):
    """
    The average sourcing distance divided by the distance the SAME euro would travel
    under the uniform allocation.

    `d_r` mixes the allocation with the shape of the country: a buyer in Brest has a
    mechanically larger `d_r` than one in Bourges under ANY allocation, so the
    cross-buyer dispersion of `d_r` is partly hexagon geometry. Dividing by
    `d_r^uniform` — the same sector spending spread evenly over each sector's modelled
    cells — removes exactly that, and leaves a number with a meaningful anchor: one
    means the response travels as far as an unselective allocation would send it, below
    one means the two forces pull it closer than geometry alone.
    """
    base = build_diffusion_frame(data, value_col) if diffusion is None else diffusion
    unif = counterfactual_diffusion_frame(data, alpha=0.0, equalise_T=True,
                                          value_col=value_col, diffusion=base,
                                          verbose=False)
    ref = amplification_summary(data, radii=(), value_col=value_col,
                                diffusion=unif)["mean_upstream_distance"]
    det = (counterfactual_amplification(data, regimes=regimes, radii=_DEFAULT_RADII,
                                        value_col=value_col, frames=frames,
                                        diffusion=base, verbose=False)
           if detail is None else detail)
    out = det[["region", "mean_upstream_distance"]].copy()
    out["uniform_distance"] = ref.reindex(
        out.index.get_level_values("ze2010_downstream")).to_numpy()
    out["distance_ratio"] = out["mean_upstream_distance"] / out["uniform_distance"]
    if verbose:
        print("  d_r normalised by the uniform-allocation distance (1 = as far as an "
              "unselective allocation would send it):")
        print(out.groupby(level="regime", observed=True)["distance_ratio"]
                 .describe()[["mean", "50%", "min", "max"]].round(3).to_string())
    return out


def destination_composition(data, regime="Realised", n_hub=2, value_col="share",
                            frames=None, diffusion=None, table=None):
    """
    Each buyer's response split into {own zone, the industry's hubs, everything else}.

    The hubs are chosen from the DATA, not named by hand: the `n_hub` upstream regions
    collecting the most euros summed over every buyer. If two long blocks of the same
    colour appear under nearly every origin, the industry is hub-and-spoke; if the
    "rest" block is wide and the hub blocks are thin, it is a dispersed pool. That is
    the statement `plot_sourcing_barycentre` was drawn to make and could not.

    The own-zone block is taken FIRST, so a buyer that is itself a hub does not have its
    own sourcing counted twice; the hub column for that buyer is then zero by
    construction and the figure shows it as a missing block, which is correct.
    """
    base = build_diffusion_frame(data, value_col) if diffusion is None else diffusion
    if regime == "Realised":
        df = base
    else:
        frames = (counterfactual_frames(data, value_col=value_col, diffusion=base,
                                        verbose=False) if frames is None else frames)
        if regime not in frames:
            raise KeyError(f"regime {regime!r} is not among {list(frames)}.")
        df = frames[regime]

    W = incidence_matrix(df, R=data["R"])
    names = (df.drop_duplicates("ze2010").set_index("ze2010")["ze_name"]
               .reindex(W.columns))
    hubs = list(W.sum(axis=0).sort_values(ascending=False).index[:n_hub])

    A = W.to_numpy(dtype=float)
    cols = {c: k for k, c in enumerate(W.columns)}
    own = np.array([A[i, cols[r]] if r in cols else 0.0
                    for i, r in enumerate(W.index)], dtype=float)
    out = pd.DataFrame({"region": df.groupby("ze2010_downstream")["shocked_name"]
                                    .first().reindex(W.index).to_numpy(),
                        "Own zone": own}, index=W.index)
    for h in hubs:
        v = A[:, cols[h]].copy()
        v[W.index.to_numpy() == h] = 0.0          # already counted as the own zone
        out[str(names.get(h, h))] = v
    seg = [c for c in out.columns if c != "region"]
    out["Rest of France"] = 1.0 - out[seg].sum(axis=1)
    out.attrs["hubs"] = hubs
    out.attrs["hub_names"] = [str(names.get(h, h)) for h in hubs]
    out.attrs["regime"] = regime
    return out


def plot_destination_composition(data, regime="Realised", n_hub=2, value_col="share",
                                 composition=None, frames=None, diffusion=None,
                                 sort_by="Own zone", figsize=None, save_to=None):
    """
    The destination composition as one horizontal stacked bar per shocked buyer.

    This is the figure that replaces the barycentre arrows. Each bar is a full euro of
    upstream response, split into where it goes; the blocks are drawn in a fixed order
    (own zone, hubs, rest) with a fixed colour each, so the same block sits at the same
    place in every bar and the reader compares LENGTHS down a column rather than
    positions on a map. Nothing is scaled or exaggerated — the bars all have length one
    because they are shares of one euro, which is the honest normalisation for a
    question about composition.
    """
    comp = (destination_composition(data, regime=regime, n_hub=n_hub,
                                    value_col=value_col, frames=frames,
                                    diffusion=diffusion)
            if composition is None else composition)
    seg = [c for c in comp.columns if c != "region"]
    order = comp[sort_by].sort_values().index if sort_by in comp.columns else comp.index
    comp = comp.reindex(order)

    # own zone first, then one colour per hub, then a neutral grey for everything else:
    # the block order and the colours are FIXED, so the same segment sits at the same
    # place in every bar and in every panel of the pair.
    hub_palette = [sim_color, (0.45, 0.60, 0.45), (0.85, 0.65, 0.30), (0.55, 0.40, 0.65)]
    n_hubs = max(0, len(seg) - 2)
    colours = [toulouse_color] + hub_palette[:n_hubs] + [(0.80, 0.80, 0.80)]

    fig, ax = plt.subplots(figsize=figsize or (7.5, max(3.0, 0.28 * len(comp))))
    y = np.arange(len(comp))
    left = np.zeros(len(comp))
    for k, c in enumerate(seg):
        v = comp[c].to_numpy(dtype=float)
        ax.barh(y, v, left=left, height=0.72, label=c,
                color=colours[min(k, len(colours) - 1)],
                edgecolor="white", linewidth=0.4)
        left = left + v
    ax.set_yticks(y)
    ax.set_yticklabels(comp["region"].astype(str))
    ax.set_ylim(-0.6, len(comp) - 0.4)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of the upstream response")
    ax.set_ylabel("Shocked commuting zone")
    ax.grid(alpha=0.2, axis="x")
    ax.legend(frameon=False, fontsize=9, ncol=min(4, len(seg)),
              loc="lower left", bbox_to_anchor=(0.0, 1.005))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        fig.savefig(save_to, bbox_inches="tight")
    return ax


def realised_incidence(data, value_col="share", diffusion=None, expected=None,
                       verbose=True):
    """
    The incidence statistics on the REALISED finite-variety economy, realisation by
    realisation, beside the ones computed on the sourcing probabilities.

    **Why this exists, and it is not a robustness check.** Everything else in this
    section is computed on `rho`, which carries no `N_s`: `incidence_matrix` refuses a
    frame that still has a `replication` column, so `n_eff` and `tv_common` describe
    the AVERAGE network. Whether that is the right object depends on the ORDER of the
    functional, and the two coordinates fall on opposite sides of the line.

    `d_r = sum_l omega_lr d_lr` is LINEAR in `omega`, and `E[omega_realised] =
    omega_rho`, so averaging is exact: the realised mean distance is the closed-form
    one, and the run's 0.3 and 1.3 km gaps are that identity, not a small effect.

    `H_r = sum_l omega_lr^2` is QUADRATIC hence convex, so by Jensen
    `E[H_realised] >= H_rho`, the gap being `sum_l Var(omega_lr)`. The inequality is
    strict and one-directional: **granularity always concentrates.** And the size of it
    is bounded by construction rather than being a matter of degree --- in one draw each
    of the sector's `N_s` varieties is won by exactly ONE region, so the realised
    incidence has at most `sum_s N_hat_s` non-zero entries, a ceiling of the same order
    as the `n_eff` the closed form reports. `tv_common` is convex too, in the opposite
    direction: realised incidence vectors are FURTHER apart than averaged ones, so the
    reported buyer-specificity is a lower bound.

    **What is unaffected**, and why the section's argument does not move: the
    counterfactual comparisons are computed on `rho` with the same `N`-free machinery
    in every regime, so granularity enters each of them identically and the ratio to
    the uniform allocation, and the displacement under equalised `T`, are clean. What
    needs this table is the LEVEL statement.

    Returns one row per buyer: the mean and sd across realisations of `n_eff`, `hhi`
    and `tv_common`, and TWO references beside each. `_pooled` is the statistic of the
    POOLED frame --- the empirical mean of exactly these realisations --- against which
    the Jensen inequality is an identity of the data at hand and holds draw for draw.
    `_expected` is the closed-form value on `rho`; it is the number the section reports,
    and it coincides with `_pooled` only in so far as the parquet was drawn from that
    `rho`, which is the case on a real run and need not be on a planted one.
    `n_eff_gap = n_eff_realised / n_eff_pooled` is how much of the dispersion survives a
    single draw. `.attrs["per_replication"]` keeps the raw draw-by-draw table and
    `.attrs["variety_ceiling"]` the `sum_s N_hat_s` bound.
    """
    df = build_diffusion_frame(data, value_col, per_replication=True) \
        if diffusion is None else diffusion
    if "replication" not in df.columns:
        raise ValueError("the parquet carries no `replication` column, so there is no "
                         "realised economy to measure — this is the continuum solve.")
    if expected is None:
        expected = incidence_concentration_overlap(
            data, value_col=value_col, include_realised=False, benchmark=False,
            regimes={"Both forces": dict()})
    exp_row = expected.xs("Both forces", level="regime")

    # the empirical mean of the same realisations -- the reference the convexity
    # argument is actually about
    pooled = df.groupby(["ze2010_downstream", "ze2010"], as_index=False)\
               .agg(upstream_sales=("upstream_sales", "sum"))
    pool_row = incidence_stats(incidence_matrix(pooled, R=data["R"]))

    per = []
    for b, sub in df.groupby("replication", sort=True):
        st = incidence_stats(incidence_matrix(sub.drop(columns="replication"),
                                              R=data["R"]))
        per.append(st.assign(replication=b))
    per = pd.concat(per)
    cols = ["n_eff", "hhi", "tv_common"]
    g = per.groupby(level=0)[cols]
    out = g.mean().add_suffix("_realised").join(g.std().add_suffix("_sd"))
    out = out.join(pool_row[cols].add_suffix("_pooled"))
    out = out.join(exp_row[cols].add_suffix("_expected"))
    out["n_eff_gap"] = out["n_eff_realised"] / out["n_eff_pooled"]
    out["region"] = exp_row["region"] if "region" in exp_row else np.nan

    n_hat = data.get("post_hoc_N_hat")
    ceiling = float(np.asarray(n_hat).ravel().sum()) if n_hat is not None else np.nan
    out.attrs["per_replication"] = per
    out.attrs["variety_ceiling"] = ceiling
    out.attrs["n_replications"] = int(per["replication"].nunique())

    if verbose:
        print(f"\n  realised vs expected incidence "
              f"({out.attrs['n_replications']} realisations)")
        print(f"    variety ceiling  sum_s N_hat_s = {ceiling:.0f} "
              f"(the most origins ONE draw can reach)")
        for c in cols:
            e = out[f"{c}_expected"].median()
            p = out[f"{c}_pooled"].median()
            r = out[f"{c}_realised"].median()
            print(f"    {c:>10s}  closed form {e:8.3f}   pooled {p:8.3f}   "
                  f"one draw {r:8.3f}   ratio {r / p if p else np.nan:5.2f}")
        print("    Jensen: n_eff must fall and tv_common must rise; the counterfactual "
              "ratios are unaffected (granularity enters every regime identically).")
    return out


def _segments_cross(p, q, r, s):
    """True when the open segments p->q and r->s properly cross (shared endpoints do not count)."""
    def side(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    d1, d2 = side(p, q, r), side(p, q, s)
    d3, d4 = side(r, s, p), side(r, s, q)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def incidence_plane_crossings(moves):
    """
    Count how many pairs of displacement arrows cross.

    `moves` is a list of `(x0, y0, x1, y1)`. This is the tangle test the figure's layout
    is decided on: with twenty-odd buyers per industry a single panel is only legible if
    the arrows are close to a common direction, and "close to a common direction" is
    exactly "few crossings". Counting is O(n^2), which is nothing at this size, and it
    replaces an eyeball judgement with a number the gate can assert on.
    """
    segs = [((m[0], m[1]), (m[2], m[3])) for m in moves]
    return sum(1 for i in range(len(segs)) for j in range(i + 1, len(segs))
               if _segments_cross(segs[i][0], segs[i][1], segs[j][0], segs[j][1]))


def plot_incidence_plane(tables, regimes=("Both forces", "Distance only"),
                         layout="auto", figsize=None, save_to=None, annotate=True,
                         arrows=True, crossing_tol=0.25):
    """
    The 2x2 as a DISPLACEMENT in a plane: concentration on `x`, buyer-specificity on `y`,
    one dot per shocked buyer, and one arrow per buyer carrying it from the estimated
    economy to the counterfactual.

    `tables` is a list of `(label, incidence table)` pairs.

    **Only two regimes are drawn, and the two that are excluded are excluded for
    reasons of construction rather than taste.** The `Uniform benchmark` is the
    DENOMINATOR of the `x` axis: it sits at `(1, ~0)` by definition, so it is a fixed
    point and is drawn as the reference corner (the two rules), never as a cloud.
    `Comparative advantage only` removes the only buyer-specific term in `rho`, so
    `tv_common` must collapse there — plotting it would lay a flat line on `y ~ 0` that
    a reader would take for an economic finding rather than an identity. That leaves
    `Both forces` and `Distance only`, which is the comparison the section is about.

    **Colour is the industry, marker FILL is the regime** (filled = the estimate, hollow
    = the counterfactual), so the four clouds are two contrasts and not four categories.
    Each buyer's two points are joined by an arrow at TRUE scale: the horizontal
    component says whether equalising `T` de-concentrates the response, the vertical one
    whether it makes buyers more alike or more idiosyncratic. Opposite vertical
    displacements across industries under the SAME intervention is the finding, and it
    is a picture rather than a paragraph of numbers.

    `layout='auto'` puts everything on one panel unless the arrows tangle — measured,
    not judged: if more than `crossing_tol` of the arrow pairs of some industry cross,
    it falls back to one panel per industry SHARING ONE WINDOW (which is the only thing
    that keeps the two panels comparable). `'single'` / `'panels'` force either.
    """
    if len(regimes) != 2:
        raise ValueError("plot_incidence_plane draws exactly two regimes (an origin and "
                         f"a destination); got {list(regimes)}.")
    r0, r1 = regimes
    cols = [toulouse_color, sim_color, (0.45, 0.60, 0.45)]

    series = []
    for k, (lab, tab) in enumerate(tables):
        have = set(tab.index.get_level_values("regime"))
        for rg in (r0, r1):
            if rg not in have:
                raise KeyError(f"regime {rg!r} is not in the incidence table for {lab}.")
        a = tab.xs(r0, level="regime")
        b = tab.xs(r1, level="regime")
        common = [i for i in a.index if i in b.index]
        a, b = a.loc[common], b.loc[common]
        ok = (a[["n_eff_ratio", "tv_common"]].notna().all(axis=1)
              & b[["n_eff_ratio", "tv_common"]].notna().all(axis=1))
        a, b = a[ok], b[ok]
        moves = list(zip(a["n_eff_ratio"], a["tv_common"],
                         b["n_eff_ratio"], b["tv_common"]))
        series.append(dict(label=lab, color=cols[k % len(cols)], a=a, b=b, moves=moves))

    if layout == "auto":
        tangled = False
        for s in series:
            n = len(s["moves"])
            npair = n * (n - 1) / 2
            if npair and incidence_plane_crossings(s["moves"]) / npair > crossing_tol:
                tangled = True
        layout = "panels" if tangled else "single"
    elif layout not in ("single", "panels"):
        raise ValueError("layout must be 'auto', 'single' or 'panels'.")

    xs = np.concatenate([np.r_[s["a"]["n_eff_ratio"].values,
                               s["b"]["n_eff_ratio"].values] for s in series])
    ys = np.concatenate([np.r_[s["a"]["tv_common"].values,
                               s["b"]["tv_common"].values] for s in series])
    pad_x = 0.05 * (np.nanmax(xs) - np.nanmin(xs) + 1e-12)
    pad_y = 0.05 * (np.nanmax(ys) - np.nanmin(ys) + 1e-12)
    xlim = (min(np.nanmin(xs), 1.0) - pad_x, max(np.nanmax(xs), 1.0) + pad_x)
    ylim = (min(np.nanmin(ys), 0.0) - pad_y, np.nanmax(ys) + pad_y)

    if layout == "single":
        fig, axes = plt.subplots(figsize=figsize or get_figsize(hf=0.62))
        axes = [axes] * len(series)
        axl = [axes[0]]
    else:
        fig, arr = plt.subplots(1, len(series), sharex=True, sharey=True,
                                figsize=figsize or get_figsize(hf=0.52))
        axes = list(np.atleast_1d(arr))
        axl = axes

    for s, ax in zip(series, axes):
        c = s["color"]
        ax.scatter(s["a"]["n_eff_ratio"], s["a"]["tv_common"], s=38, color=c,
                   edgecolors="white", linewidths=0.5, zorder=4,
                   label=f"{s['label']} — {r0}")
        ax.scatter(s["b"]["n_eff_ratio"], s["b"]["tv_common"], s=38,
                   facecolors="none", edgecolors=c, linewidths=1.1, zorder=4,
                   label=f"{s['label']} — {r1}")
        if arrows:
            for x0, y0, x1, y1 in s["moves"]:
                ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=3,
                            arrowprops=dict(arrowstyle="->", color=c, lw=0.7,
                                            alpha=0.55, shrinkA=3, shrinkB=3))
        if annotate:
            for frame, mk in ((s["a"], "D"), (s["b"], "s")):
                ax.scatter([frame["n_eff_ratio"].median()], [frame["tv_common"].median()],
                           marker=mk, s=80, color=c, edgecolors="black",
                           linewidths=0.9, zorder=6)
        if layout == "panels":
            ax.set_title(s["label"], fontsize=9)

    for ax in axl:
        ax.axvline(1.0, color="0.5", lw=0.8, ls="--")
        ax.axhline(0.0, color="0.5", lw=0.8)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axl[0].set_ylabel("Share of the response that is buyer-specific")
    xlab = "Effective number of destinations, relative to the uniform allocation"
    if layout == "panels":
        fig.supxlabel(xlab, fontsize=9)
    else:
        axl[0].set_xlabel(xlab)
    axl[0].legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        fig.savefig(save_to, bbox_inches="tight")
    return axl[0] if layout == "single" else axl


# ============================================================================
# The input-output benchmark for D_r
# ============================================================================

# The input-output benchmark for D_r: the composition and coverage of the downstream
# industry's intermediate purchases, then the Leontief cascade. The caveat about the
# intermediate share the TES block cannot supply is in the markdown above, and the
# scalar path repeats it at runtime.

IO_TABLE_KINDS = ("dom", "imp")     # TES_dom.csv = domestic flows, TES_imp.csv = imports


def read_io_table(kind, folder):
    """
    One INSEE TES intermediate-flow block, long: (A129_1 supplying, A129_2 using, value).

    Same parsing as the pipeline that built `pi_s`: semicolon separated, no header,
    the first column being the row labels, and a comma decimal separator.
    """
    path = Path(folder) / f"TES_{kind}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    io = pd.read_csv(path, sep=";", header=None, index_col=0)
    io.columns = io.index
    io = io.reset_index()
    first = io.columns[0]
    io = io.melt(id_vars=first, var_name="A129_2", value_name="value")
    io = io.rename(columns={first: "A129_1"})
    io["value"] = io["value"].astype(str).str.replace(",", ".").astype(float)
    io["A129_1"] = io["A129_1"].astype(str).str.strip()
    io["A129_2"] = io["A129_2"].astype(str).str.strip()
    return io


def io_flow_matrix(data, kinds=("dom",), io_folder=None):
    """
    The square (supplying x using) flow matrix, summed over `kinds`.

    `kinds=("dom",)` is the domestic technology — the right support for a multiplier
    that is meant to stay in France, since imports leak out of the cascade.
    `("dom", "imp")` is total use, which is the support the sector shares `pi_s` were
    built on.
    """
    folders = ([Path(io_folder)] if io_folder is not None
               else [data["input_folder"], data["folder"], data["input_folder"].parent])
    total, used = None, None
    for f in folders:
        try:
            parts = [read_io_table(k, f) for k in kinds]
        except FileNotFoundError:
            continue
        used = f
        for p in parts:
            m = p.pivot(index="A129_1", columns="A129_2", values="value").fillna(0.0)
            total = m if total is None else total.add(m, fill_value=0.0)
        break
    if total is None:
        raise FileNotFoundError(
            "no TES table found. Looked for " + ", ".join(f"TES_{k}.csv" for k in kinds)
            + " in " + ", ".join(str(f) for f in folders) +
            ". Pass `io_folder=` to point at the directory that holds them.")
    idx = sorted(set(total.index) | set(total.columns))
    total = total.reindex(index=idx, columns=idx).fillna(0.0)
    total.attrs["io_folder"] = str(used)
    total.attrs["kinds"] = tuple(kinds)
    return total


def io_technical_coefficients(X, intermediate_share):
    """
    `A` from the flow block: each column normalised to sum to that sector's
    intermediate share of output.

    `intermediate_share` is a scalar applied to every sector, or a Series indexed by
    A129. A column of zeros (a sector that buys nothing in the table) stays zero.
    """
    col_tot = X.sum(axis=0).replace(0.0, np.nan)
    m = (pd.Series(float(intermediate_share), index=X.columns)
         if np.isscalar(intermediate_share)
         else pd.Series(intermediate_share).reindex(X.columns))
    if m.isna().any():
        raise ValueError(f"no intermediate share for {list(m.index[m.isna()])}")
    if (m >= 1).any():
        raise ValueError("an intermediate share >= 1 makes (I - A) singular: a sector "
                         "cannot spend its whole output on intermediates and still "
                         "close the cascade.")
    return (X / col_tot).fillna(0.0) * m


def leontief_multipliers(A, max_rounds=None):
    """
    Column sums of the Leontief inverse — total output per euro of final demand — and,
    when `max_rounds` is given, the same truncated after that many rounds.

    `(I - A)^{-1} = I + A + A^2 + ...`, so the round-1 truncation `1 + sum_i a_{id}`
    is the object the model's single upstream tier corresponds to.
    """
    n = A.shape[0]
    L = np.linalg.inv(np.eye(n) - A.to_numpy())
    if (np.diag(L) <= 0).any() or not np.isfinite(L).all():
        raise ValueError("(I - A) is not invertible or the inverse is not positive — "
                         "check the intermediate shares.")
    out = {"total": pd.Series(L.sum(axis=0), index=A.columns)}
    if max_rounds:
        P, acc = np.eye(n), np.eye(n)
        for _ in range(max_rounds):
            P = P @ A.to_numpy()
            acc = acc + P
            out[f"rounds_{_ + 1}"] = pd.Series(acc.sum(axis=0), index=A.columns)
    return pd.DataFrame(out)


def io_downstream_column(data, io_folder=None, summary=None):
    """
    Who supplies the downstream industry, in the IO table and in the model.

    Needs no gross output and no assumption: both sides are SHARES of the downstream
    industry's intermediate purchases. `pi_s_io` is the object the sector weights were
    calibrated on (dom + imp over the modelled sectors, renormalised); `share_model` is
    the simulated split of upstream sales across sectors, so the two should agree, and
    a sector where they do not is a fit problem rather than a benchmarking one.
    """
    d = data["d"]
    dom = io_flow_matrix(data, kinds=("dom",), io_folder=io_folder)
    imp = io_flow_matrix(data, kinds=("imp",), io_folder=io_folder)
    if d not in dom.columns:
        raise KeyError(f"the downstream industry {d} is not a column of the TES table.")
    col = pd.DataFrame({"dom": dom[d], "imp": imp[d].reindex(dom.index).fillna(0.0)})
    col["total"] = col["dom"] + col["imp"]
    col["modelled"] = col.index.isin(data["sector_names"])
    col["share_of_total"] = col["total"] / col["total"].sum()
    col["share_of_dom"] = col["dom"] / col["dom"].sum()
    col["pi_s_io"] = np.where(col["modelled"],
                              col["total"] / col.loc[col["modelled"], "total"].sum(), np.nan)

    sup = data.get("suppliers")
    if sup is not None:
        by_s = sup.groupby("A129")["share"].sum()
        by_s = by_s / by_s.sum()
        # the parquet keys sectors by the model's 1..S index, in A129-code order
        names = list(data["sector_names"])
        by_s.index = [names[int(i) - 1] if 1 <= int(i) <= len(names) else str(i)
                      for i in by_s.index]
        col["share_model"] = by_s.reindex(col.index)
    return col.sort_values("total", ascending=False)


def io_amplification_benchmark(data, summary=None, intermediate_share=None,
                               kinds=("dom",), io_folder=None, verbose=True):
    """
    The model's D_r against the input-output benchmarks: the first round, and the whole
    Leontief cascade — the latter on TWO supports, since restricting the economy to the
    modelled sectors is a judgement call rather than a fact.

      FULL     every A129 sector in the table: the true aggregate multiplier;
      SUBSET   the modelled sectors plus the downstream industry: the cascade the model
               could in principle represent, so the honest ceiling for it.

    Returns a one-row-per-support table plus the per-sector multipliers for the figure.
    """
    d = data["d"]                                    # C29A (auto) / C30C (aero)
    X = io_flow_matrix(data, kinds=kinds, io_folder=io_folder)
    if d not in X.columns:
        raise KeyError(f"the downstream industry {d} is not a column of the TES table "
                       f"({len(X.columns)} sectors).")
    m = (1.0 - float(data["agg_labor_share"])) if intermediate_share is None \
        else intermediate_share
    scalar_m = np.isscalar(m)

    subset = [s for s in data["sector_names"] if s in X.columns]
    if d not in subset:
        subset = subset + [d]
    missing = [s for s in data["sector_names"] if s not in X.columns]

    rows, per_sector = [], {}
    for label, cols in (("full", list(X.columns)), ("subset", subset)):
        Xs = X.loc[cols, cols]
        A = io_technical_coefficients(Xs, m)
        mult = leontief_multipliers(A, max_rounds=2)
        per_sector[label] = mult
        rows.append({
            "support": label,
            "n_sectors": len(cols),
            "round1": 1.0 + float(A[d].sum()),
            "rounds_2": float(mult.loc[d, "rounds_2"]),
            "leontief_total": float(mult.loc[d, "total"]),
        })
    tab = pd.DataFrame(rows).set_index("support")

    # --- coverage: what fraction of the true first round the model can represent,
    # from shares alone, with no output data and no assumption
    col = io_downstream_column(data, io_folder=io_folder, summary=summary)
    cov_dom = float(col.loc[col["modelled"], "dom"].sum() / col["dom"].sum())
    cov_tot = float(col.loc[col["modelled"], "total"].sum() / col["total"].sum())
    dom_share = float(col["dom"].sum() / col["total"].sum())

    # the model side: D_r - 1 IS the intermediate share of downstream unit cost
    s = amplification_summary(data) if summary is None else summary
    D_mean = float(s["amplification"].mean())
    tab["model_D_r"] = D_mean
    tab["captured_of_aggregate"] = (D_mean - 1.0) / (tab["leontief_total"] - 1.0)
    tab["model_over_round1"] = (D_mean - 1.0) / (tab["round1"] - 1.0)
    tab.attrs.update(intermediate_share=m, downstream=d, kinds=tuple(kinds),
                     io_folder=X.attrs["io_folder"], per_sector=per_sector,
                     missing_sectors=missing, downstream_column=col,
                     scalar_m=scalar_m,
                     coverage_modelled_dom=cov_dom, coverage_modelled_total=cov_tot,
                     domestic_share_of_column=dom_share,
                     implied_intermediate_share=D_mean - 1.0,
                     implied_total_intermediate_share=(D_mean - 1.0) / cov_dom
                     if cov_dom > 0 else np.nan,
                     data_intermediate_share=1.0 - float(data["agg_labor_share"]))

    if verbose:
        print(f"[{data['industry']}]  downstream industry {d}, TES from "
              f"{X.attrs['io_folder']} ({'+'.join(kinds)}), "
              f"intermediate share m = {m if np.isscalar(m) else 'per sector'}")
        print(f"  model  : mean D_r = {D_mean:.3f}  -> the modelled domestic sectors are "
              f"{D_mean - 1:.3f} of downstream unit cost")
        print(f"  IO     : those sectors are {100 * cov_dom:.1f}% of the industry's "
              f"DOMESTIC intermediate purchases ({100 * cov_tot:.1f}% of dom + imp; "
              f"the column is {100 * dom_share:.1f}% domestic)")
        print(f"           => implied TOTAL intermediate share of unit cost = "
              f"{(D_mean - 1) / cov_dom:.3f}" if cov_dom > 0 else "")
        print(f"           1 - aggregate labor share (the targeted moment) = "
              f"{1 - float(data['agg_labor_share']):.3f}")
        if scalar_m:
            print("  NOTE: a COMMON intermediate share was used, so every sector's "
                  f"Leontief total is exactly 1/(1-m) = {1 / (1 - m):.3f} and the "
                  "cascade below measures that assumption, not the network. Pass a "
                  "per-sector `intermediate_share` (value added / gross output) for an "
                  "informative comparison.")
        for lab, r in tab.iterrows():
            print(f"  IO {lab:<6}: round 1 = {r['round1']:.3f}, "
                  f"two rounds = {r['rounds_2']:.3f}, Leontief total = "
                  f"{r['leontief_total']:.3f}   "
                  f"(model captures {100 * r['captured_of_aggregate']:.0f}% of the "
                  f"aggregate, {100 * r['model_over_round1']:.0f}% of round 1)")
        if missing:
            print(f"  NOTE: {len(missing)} modelled sector(s) absent from the TES "
                  f"table: {missing}")
    return tab


def plot_io_benchmark(data, benchmark=None, figsize=None, save_to=None, **kw):
    """
    Is D_r large? The distribution of the Leontief total multiplier across ALL sectors,
    with the downstream industry, the model's mean D_r and the first-round benchmark
    marked on it. A number is high or low relative to that distribution, not in itself.
    """
    tab = io_amplification_benchmark(data, verbose=False, **kw) if benchmark is None \
        else benchmark
    mult = tab.attrs["per_sector"]["full"]["total"].sort_values()
    d, D_mean = tab.attrs["downstream"], float(tab["model_D_r"].iloc[0])
    if tab.attrs.get("scalar_m", False):
        print("plot_io_benchmark: the cross-sector distribution is degenerate under a "
              "common intermediate share (every sector at 1/(1-m)). The figure still "
              "places the model, but the spread is an artefact — pass a per-sector "
              "`intermediate_share` for a real distribution.")

    fig, ax = plt.subplots(figsize=figsize or get_figsize(wf=0.85, hf=0.6))
    ax.plot(mult.to_numpy(), np.linspace(0, 1, len(mult)), color=sim_color, lw=1.6,
            label="all A129 sectors (Leontief total)")
    for x, c, lab in ((mult.get(d, np.nan), reference_color,
                       f"{d} Leontief total = {mult.get(d, np.nan):.2f}"),
                      (float(tab.loc["full", "round1"]), "0.45",
                       f"{d} round 1 = {tab.loc['full', 'round1']:.2f}"),
                      (D_mean, toulouse_color, rf"model mean $D_r$ = {D_mean:.2f}")):
        if np.isfinite(x):
            ax.axvline(x, color=c, linestyle="--", lw=1.3, label=lab)
    q = float((mult <= D_mean).mean())
    ax.set_xlabel("Total output per euro of final demand")
    ax.set_ylabel("Share of sectors below")
    ax.set_title(f"{data['industry']}, " rf"$\hat\mu_{data['mu']}$"
                 f"   model $D_r$ at the {100 * q:.0f}th percentile of sectors",
                 fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _despine(ax)
    fig.tight_layout()
    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        fig.savefig(save_to, bbox_inches="tight")
    return ax
