"""
Shared core of the two reporting notebooks.

`model_report.ipynb` displays the fit of the structural model; `tests_counterfactuals.ipynb`
runs the tests and the counterfactuals. Everything both of them need lives here: the
figure style, the run-tree loader, the extended parameter set `theta+`, the Ricardian
geometry, and THE ECONOMY -- the forward map that turns `theta+` into a realised
finite-variety economy, estimated or counterfactual, without reading `suppliers.parquet`.

The parquet is now a VERIFICATION artefact only. `check_against_julia` compares this
port against `solve_network` to the bit, given Julia's own draws; nothing in the
reporting path reads the file. Run that check after any change to either implementation.
"""

# ============================================================================
# Imports and figure style
# ============================================================================
import os
import re
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patheffects as pe

try:
    import geopandas as gpd
except Exception:            # maps are optional in this notebook
    gpd = None
try:
    import seaborn as sns
except Exception:
    sns = None

# ---------------------------------------------------------------- style
#
# ONE style, in one place. Both notebooks import it and neither redefines it: a
# duplicated rcParams block in a notebook cell shadows this one after import, so the
# two drift and a figure's size depends on which cell was run last.
#
# Two sizes, and the distinction is what keeps a figure legible when the base moves.
# `font_size` is the BODY size -- it drives the rcParams (axis labels, ticks, legend),
# so it is what a paper's font size should be set to. Every figure-local size in the
# libraries (an annotation, a panel title, a name axis) is written as `fs(x)`, where
# `x` is that size AT THE REFERENCE BASE: `fs` rescales it PROPORTIONALLY, so raising
# the body size carries the whole family with it instead of leaving the annotations
# frozen at 8pt while the axis labels jump. At `font_size == FONT_SIZE_REF` every
# `fs(x)` is exactly `x`, so the figures this repo has already produced are unchanged.
document_width_pt = 511.0
plt.rc("font", family="serif")
toulouse_color = (132 / 255, 46 / 255, 27 / 255)
reference_color = toulouse_color#(0.75, 0.30, 0.20)
sim_color = (0.247, 0.404, 0.667)

# The base every hard-coded size in the libraries was drawn at. Do NOT change it --
# it is the anchor `fs` is calibrated against, not a setting. Change `font_size`.
FONT_SIZE_REF = 15.0
font_size = 15.0

# The LaTeX preamble used when `usetex=True`. fontenc T1 is there for the per-buyer
# figures, whose tick labels are French commuting-zone names: it gives real accented
# glyphs rather than composed ones, and correct hyphenation for accented words. utf8
# inputenc is a no-op on TeX Live 2018 and later, where it is already the default --
# MEASURED: the amsmath-only preamble renders "Ile-De-France" and "Chateauroux" without
# error on TeX Live 2023 -- and is kept only so an older installation still compiles.
LATEX_PREAMBLE = (r"\usepackage[utf8]{inputenc}"
                  r"\usepackage[T1]{fontenc}"
                  r"\usepackage{amsmath}\usepackage{amssymb}")


def _rc_from_base(size):
    """The rcParams the body size drives."""
    size = float(size)
    return {"font.size": size,
            "axes.labelsize": size,
            "axes.titlesize": size + 2,
            "xtick.labelsize": size - 3,
            "ytick.labelsize": size - 3,
            "legend.fontsize": size - 3,
            "legend.title_fontsize": size,
            "figure.titlesize": size}


plt.rcParams.update(_rc_from_base(font_size))


def fs(size_at_ref):
    """
    A figure-local font size, given at the REFERENCE base and rescaled to the current
    body size: `fs(8)` is 8pt at the default and 10.7pt once the body size is 20.

    It resolves `font_size` from this module at CALL time, so `set_paper_style` moves
    every call site at once -- including the ones in the other libraries, which import
    this function rather than the number.
    """
    return float(size_at_ref) * float(font_size) / FONT_SIZE_REF


def axes_height_inches(ax):
    """Height of `ax`'s drawing area in inches, or None if it cannot be determined."""
    try:
        return float(ax.get_position().height) * float(ax.figure.get_size_inches()[1])
    except Exception:
        return None


def axes_width_inches(ax):
    """Width of `ax`'s drawing area in inches, or None if it cannot be determined."""
    try:
        return float(ax.get_position().width) * float(ax.figure.get_size_inches()[0])
    except Exception:
        return None


def name_axis_fontsize(ax, n_rows, axis="y", size_at_ref=None, min_size=5.0,
                       fill=0.80):
    """
    Font size for a categorical axis carrying `n_rows` NAMES -- commuting zones,
    attraction areas, sectors.

    Such an axis cannot simply take the body size, and that is the whole reason this
    exists: a per-buyer figure stacks ~20 long names in a few inches, so at a 20pt body
    the labels overlap and swamp the marks they annotate. The size is therefore capped
    by the room each row ACTUALLY has -- `fill` of the per-row budget in points, a label
    needing about its own size plus leading to clear its neighbour -- and only then by
    the ceiling. With few rows the cap is slack and the axis reads at the ceiling; with
    many it tapers. Never below `min_size`, where a name stops being worth drawing and
    the caller should be dropping rows instead.

    The ceiling is `fs(size_at_ref)` when the caller names a size, and otherwise the
    CURRENT rcParams tick size for that axis -- which already tracks the body size. So
    a figure that fixed no size keeps exactly the size it had, and only gains the cap.
    """
    if size_at_ref is None:
        key = "ytick.labelsize" if axis == "y" else "xtick.labelsize"
        size = float(mpl.font_manager.font_scalings.get(plt.rcParams[key],
                                                       plt.rcParams[key])
                     if isinstance(plt.rcParams[key], str) else plt.rcParams[key])
    else:
        size = fs(size_at_ref)
    axis_length_in = axes_height_inches(ax) if axis == "y" else axes_width_inches(ax)
    n = int(n_rows or 0)
    if axis_length_in is not None and n > 0:
        size = min(size, fill * 72.0 * axis_length_in / n)
    return max(float(min_size), float(size))


def set_name_axis_fontsize(ax, n_rows, axis="y", size_at_ref=None, min_size=5.0,
                           fill=0.80):
    """`name_axis_fontsize`, applied to `ax`'s tick labels. Returns the size used."""
    size = name_axis_fontsize(ax, n_rows, axis=axis, size_at_ref=size_at_ref,
                              min_size=min_size, fill=fill)
    ax.tick_params(axis=axis, labelsize=size)
    return size


def set_paper_style(font_size=None, usetex=False, preamble=None, family="serif"):
    """
    Set the body font size, and optionally route text through LaTeX.

    This is the one call a notebook makes; it rebinds the module-level `font_size`, so
    every `fs(...)` in every library follows and the figure family stays internally
    proportioned. `usetex` is OFF by default and is never enabled at import, because a
    missing LaTeX installation would then fail every figure in the test gates rather
    than in the notebook that asked for it.

    Returns the rcParams the call applied, so a cell shows what it did.
    """
    # the parameter carries the module global's name -- which is the name to write at a
    # call site -- so the rebinding goes through `globals()` rather than `global`.
    if font_size is not None:
        globals()["font_size"] = float(font_size)
    rc_params = _rc_from_base(globals()["font_size"])
    rc_params["font.family"] = family
    rc_params["text.usetex"] = bool(usetex)
    if usetex:
        rc_params["text.latex.preamble"] = LATEX_PREAMBLE if preamble is None else preamble
    plt.rcParams.update(rc_params)
    return rc_params


def pct():
    """
    A literal per-cent sign, escaped when text goes through LaTeX.

    Under `usetex` a bare `%` opens a COMMENT, so it does not raise -- it SILENTLY
    truncates the rest of the label, which is how "(% of Both forces)" became "(" on a
    rendered axis. `\\%` is the LaTeX spelling, and a non-usetex render would print that
    backslash literally, so the choice cannot be written into the source string: it is
    made here, when the text is built, from the rcParam in force.
    """
    return r"\%" if plt.rcParams.get("text.usetex", False) else "%"


def get_figsize(document_width_pt=document_width_pt, width_fraction=1.0, height_fraction=0.5):
    """Figure size in inches from a LaTeX \\showthe\\columnwidth width in points."""
    fig_width = document_width_pt * width_fraction / 72.27
    return [fig_width, fig_width * height_fraction]


def _despine(ax=None):
    if sns is not None:
        sns.despine(ax=ax)
    else:
        for side in ("top", "right"):
            (ax or plt.gca()).spines[side].set_visible(False)


# ============================================================================
# Calibration and reporting constants
# ----------------------------------------------------------------------------
# Module-level DEFAULTS. A notebook may override any of them in its Constants
# cell and pass the value in; nothing here reads a notebook global.
# ============================================================================

# Frechet shape. Read from `stats.csv` when it carries a `theta` column and taken from
# here otherwise, so this value MUST track `load_parameters.jl` (`theta = 1.78`) — the
# economy on disk was solved at the Julia value, and every theta*alpha-scaled object here
# is off by the ratio if the two disagree. `extended_parameters` records which of the two
# it used and warns when it fell back, so the discrepancy is visible rather than silent.
THETA_DEFAULT = 1.78
# within-sector CES elasticity. CALIBRATED, not estimated and not written to any
# artefact: `load_parameters.jl` sets nu_s = 1.5 for every sector. It is the
# substitution parameter of the granular decomposition, so it belongs beside theta.
NU_S_DEFAULT = 1.5
# The two remaining calibrated elasticities, mirroring `load_parameters.jl` (lines 60-61).
# They are consts there and consts here; a run that changes one must change both, and
# `extended_parameters` carries them so the economy is never solved at a mixed
# calibration. Neither is estimated.
NU_ACROSS_DEFAULT = 0.2          # nu, ACROSS sectors
LAMBDA_DEFAULT = 0.5             # lambda, labour against intermediates

# How many economies each regime draws — every regime, the estimated one included, since
# all of them go through `simulate_economy`. It used to default to `None`, meaning "match
# the baseline": the replication count of `suppliers.parquet`. That is obsolete — the
# parquet is not on the reporting path and there is no baseline to match — so the number
# is set here outright. It governs the DISPERSION every band and `z` score rests on, and
# it is the one knob that buys precision for Python time alone.
#
# The Julia-side `POST_HOC_REPLICATIONS` in `main.jl` is a DIFFERENT quantity: how many
# economies Julia SOLVES and writes for the verification artefact. Raising that one costs
# a re-run and improves only `check_against_julia`'s sample, not anything reported.
ECONOMY_REPLICATIONS = 1000

# Highest K kept anywhere in the count-curve reporting: the panels and the empirical
# shares read out of G_K.csv are truncated here. Raise it to look further into the
# tail, but only as far as the data pipeline actually tabulates; a K absent from
# G_K.csv leaves that panel with the simulated curve only.
#
# K = 0 is the targeted moment (block 6); everything above it is a free check on the
# SHAPE of the supplier-count distribution (gate V8).
# Test 8 sweeps the similarity threshold rather than defending one value.
PORTFOLIO_TAUS = (0.6, 0.7, 0.8)

COUNT_CURVE_K_MAX = 3
COUNT_CURVE_K = tuple(range(COUNT_CURVE_K_MAX + 1))

# Radii for the amplification section; the FIRST is the headline one, used by the
# bar chart and the scatter. Any tuple works — (100,), (100, 200), (50, 100, 200)...
AMPLIFICATION_RADII = (100, 200)

# Mean of log Dist in the EMPIRICAL estimation sample of the spatial-comovement
# regression (both industries). The reduced form is linear in the level of the exposure,
# so its delta/gamma is a compressed reading of an elasticity and the compression depends
# on this number — it is what puts the model's eta and the published delta/gamma on the
# same scale. See the untargeted-moment section.
EMPIRICAL_MEAN_LOG_D = 5.8


# ============================================================================
# The regime vocabulary
# ----------------------------------------------------------------------------
# These names are shared by the diffusion, concentration and local-share
# sections, and `economy_by_regime` builds every regime in CF_REGIMES, so they
# belong beside the economy rather than inside any one section.
# ============================================================================


# --- The same shock, with one force switched off ----------------------------
#
# Where a shock goes is decided by the Ricardian competition, which is closed form
# (`sourcing_geometry`, defined in the comparative-advantage section above and reused
# here rather than duplicated). So the propagation can be recomputed with either force
# removed without re-simulating or re-estimating anything.

# The two switches, each applied alone. Turning BOTH off — the uniform 1/n draw over
# the sector's modelled cells — is still supported by `sourcing_geometry` and is what the
# gate pins the closed form against, but it is not reported: a benchmark in which nothing
# selects among cells is not a counterfactual economy, and reading the two real regimes
# against it invites the reader to treat "chance" as the null the model is tested against.
CF_REGIMES = {"Both forces": dict(),
              "Equal comparative advantage": dict(equalise_T=True),
              "No trade cost": dict(alpha=0.0)}

# One colour per regime, fixed here so a regime reads the same in every figure.
# "Both forces" is the ESTIMATED economy and is black: it is the reference the two
# counterfactuals are read against, not one of three alternatives, so it takes the
# neutral ink and each counterfactual takes a hue of its own.
CF_COLORS = {"Both forces": (0.0, 0.0, 0.0),
             "Equal comparative advantage": toulouse_color,
             "No trade cost": sim_color,
             "Neither": (0.72, 0.72, 0.72)}    # kept for a caller that asks for it

# The three forces of the estimated geography, each removed alone: the attraction-area
# layer of comparative advantage (T_a equalised, density n_ls^kappa kept), the
# within-area density gradient (kappa = 0, T_a kept), and distance (alpha = 0). Unlike
# "Equal comparative advantage", which removes BOTH layers of T at once, the first two
# split comparative advantage into its two sources.
CF_REGIMES_KAPPA = {"Both forces": dict(),
                    "Equal area advantage": dict(equalise_T="area"),
                    "No density gradient": dict(kappa=0.0),
                    "No trade cost": dict(alpha=0.0)}
CF_COLORS["Equal area advantage"] = toulouse_color
CF_COLORS["No density gradient"] = (0.80, 0.55, 0.10)

# The geometry-free benchmark: nothing selects among the cells of a sector, so the
# incidence carries only the sector mix and the modelled cell support. It is NOT a
# counterfactual economy and is never reported as one -- it exists to divide out.
UNIFORM_REGIME = "Uniform benchmark"
CF_COLORS[UNIFORM_REGIME] = (0.72, 0.72, 0.72)

# The N_s -> infinity limit of the estimated economy: the structural column with
# granularity taken out, which is what isolates granularity buyer by buyer.
INFINITE_REGIME = "Infinite varieties"


# ============================================================================
# The run tree: paths, labels and the loader
# ============================================================================

# Rebuilds, on the Python side, the moment layout `load_parameters.jl` builds under
# `--granular=true --ca_level=aa`, and reads back everything the reporting needs.
# The block layout, the mask and the meaning of `mu` are documented in the markdown
# cell above.

# mu -> (folder holding the moments, folder holding the inference)
MU_DIRS = {1: ("step1", "step2"), 2: ("step3", "step3")}


def reporting_folder(industry, base="..", profile_T=True, ca_level="aa",
                     granular=True, relax_n_lo=False, optimizer="pso",
                     kappa=True):
    """Reproduce main.jl's output_folder naming exactly.

    The current Julia code estimates one specification and names its tree
    `reporting_<ind>_profiled_aa_gran_kappa[_nlo1]_pso`. `kappa=False`
    rebuilds the pre-kappa name (`..._gran[_nlo1]_pso`), whose parameter and moment
    layouts this loader no longer reads -- it is kept only so the error is a clear
    layout mismatch rather than a missing folder.
    """
    name = f"reporting_{industry}"
    name += "_profiled" if profile_T else ""
    name += "_aa" if ca_level == "aa" else ""
    name += "_gran" if granular else ""
    name += "_kappa" if (kappa and granular) else ""
    name += "_nlo1" if (relax_n_lo and granular) else ""
    name += f"_{optimizer}"
    return Path(base) / name


def _read_named_value(coefs, name):
    """stats.csv scalar by name: a column called `name`, else a labelled row."""
    if name in coefs.columns:
        values = coefs[name].dropna().values
        if len(values):
            return float(values[0])
    if "value" in coefs.columns:
        for column in coefs.columns:
            matching_rows = coefs.index[coefs[column].astype(str).str.strip().str.lower() == name.lower()]
            if len(matching_rows):
                return float(coefs.loc[matching_rows[0], "value"])
    return None


def _ze_key(x):
    """ZE code as load_parameters.jl keys it: 4-character, zero-padded when numeric."""
    t = str(x).strip()
    try:
        return f"{int(float(t)):04d}"
    except ValueError:
        return t


SENS_KINDS = {1: "aggregate", 2: "alpha", 3: "kappa", 4: "T", 5: "N"}
SENS_BLOCK_NAMES = {1: None, 2: "Trade cost", 3: "Density", 4: "Comparative advantage",
                    5: "Variety count"}


def _sens_aggregate_block(label):
    return ("chi_L" if label == "chi_L" else
            "Industry shares" if label.startswith("chi_s") else "Productivity")


def load_sensitivity(sens_dir, n_moments=None):
    """
    Read `<step>/sensitivity/` (sensitivity.jl).

    Returns {"jacobian": {...} or None, "profiles": {label: {...}}, "report": str}.
    The Jacobian entry carries `J`, `J_est`, `J_sd` (one draw set), `J_se = J_sd/sqrt(K)`
    (the standard error of the reported mean, which is what the noise-to-signal mask
    must compare with), the elasticities `J_elast = J * scale_j / m_i` with
    `scale_j = theta_j` (alpha, T, head, N_s) or 1 for kappa (a semi-elasticity: kappa
    is an exponent that can sit at zero), their SE, the labels, and the parameter
    blocks in column order.
    """
    sens_dir = Path(sens_dir)
    result = {"jacobian": None, "profiles": {}, "report": None, "dir": sens_dir}
    jacobian_path = sens_dir / "jacobian.npz"
    if jacobian_path.exists():
        with np.load(jacobian_path) as archive:
            jacobian = {k: archive[k] for k in archive.files}
        jacobian["labels"] = [l for l in (sens_dir / "jacobian_param_labels.txt").read_text()
                        .splitlines() if l.strip()]
        jacobian["kind"] = np.rint(jacobian["kind"]).astype(int)
        J, J_sd = np.atleast_2d(jacobian["J"]), np.atleast_2d(jacobian["J_sd"])
        if n_moments is not None and J.shape[0] != n_moments:
            raise ValueError(f"{jacobian_path} has {J.shape[0]} moment rows, the layout rebuilt here "
                             f"has {n_moments}.")
        K = max(int(np.ravel(jacobian["K"])[0]), 1)
        jacobian["J_se"] = J_sd / np.sqrt(K)
        scale = np.where(jacobian["kind"] == 3, 1.0, np.asarray(jacobian["theta"], float))
        base_moments = np.asarray(jacobian["m_base"], float)
        with np.errstate(divide="ignore", invalid="ignore"):
            elasticity_factor = scale[None, :] / np.where(np.abs(base_moments) > 1e-12, base_moments, np.nan)[:, None]
        jacobian["J_elast"], jacobian["J_elast_se"] = J * elasticity_factor, jacobian["J_se"] * np.abs(elasticity_factor)
        names = [SENS_BLOCK_NAMES[k] or _sens_aggregate_block(l)
                 for k, l in zip(jacobian["kind"], jacobian["labels"])]
        block_names, block_sizes = [], []
        for name in names:
            if block_names and block_names[-1] == name:
                block_sizes[-1] += 1
            else:
                block_names.append(name); block_sizes.append(1)
        jacobian["block_names"], jacobian["block_sizes"] = block_names, block_sizes
        result["jacobian"] = jacobian
    index_path = sens_dir / "profiles_index.txt"
    if index_path.exists():
        for line in index_path.read_text().splitlines():
            if not line.strip():
                continue
            label, file_name = line.split("\t")
            with np.load(sens_dir / file_name) as archive:
                profile = {k: archive[k] for k in archive.files}
            profile["targets"] = np.rint(profile["targets"]).astype(int) - 1      # 0-based
            profile["kind"] = SENS_KINDS[int(np.ravel(profile["kind"])[0])]
            profile["estimate"] = float(np.ravel(profile["estimate"])[0])
            result["profiles"][label] = profile
    report_path = sens_dir / "profiles_report.txt"
    result["report"] = report_path.read_text() if report_path.exists() else None
    return result


def _gk_col(G_K, cands):
    """Column of G_K.csv by case-insensitive name, mirroring load_parameters.jl's _gk_col."""
    match = next((column for column in G_K.columns if column.strip().lower() in cands), None)
    if match is None:
        raise KeyError(f"G_K.csv is missing a column among {cands}; has {list(G_K.columns)}")
    return match


def _as_ze_string(col):
    """ZE codes as 4-character strings, whether the column came back as int, float or str."""
    if pd.api.types.is_numeric_dtype(col):
        return col.astype("Int64").astype(str).str.zfill(4)
    return col.astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(4)


def _aa_names(X_dr, n_AA):
    """
    Attraction-area names, in the model's AA column order.

    AA column `a` is anchored on the a-th downstream region, and the downstream
    regions are enumerated in ZE-index (sorted ze2010) order — which is the row
    order of X_dr.csv restricted to the downstream rows. The names live in
    `X_dr.query('downstream').ze2010_downstream`.
    """
    downstream_flag = next((column for column in ("downstream", "downstream_region") if column in X_dr.columns), None)
    if downstream_flag is None:
        raise KeyError("X_dr.csv has neither a `downstream` nor a `downstream_region` column")
    downstream_rows = X_dr[X_dr[downstream_flag].astype(bool)]
    ze_column = next((column for column in ("ze2010_downstream", "ze2010") if column in downstream_rows.columns), None)
    if ze_column is None:
        raise KeyError("X_dr.csv has neither `ze2010_downstream` nor `ze2010`")
    names = _as_ze_string(downstream_rows[ze_column]).tolist()
    if len(names) != n_AA:
        raise ValueError(f"{len(names)} downstream rows in X_dr.csv vs n_AA = {n_AA} "
                         "attraction areas in attraction_area_linkages.npy")
    return names


def _region_labels(data):
    """
    Model region index (1..R) -> ZE code -> display NAME.

    The model orders regions by sorted `ze2010`, exactly as `load_parameters.jl` builds
    `emp_gamma_ls`, so the l-th row of the distance matrix and the integer codes in
    `suppliers.parquet` are the l-th entry of that sorted list. `france` covers more
    commuting zones than the model does, so the join is on the CODE, and a zone with no
    name falls back to its code rather than to NaN.
    """
    filter_df = data.get("filter_N_upstream_df")
    if filter_df is None:
        codes = [str(i) for i in range(1, data["R"] + 1)]
    else:
        codes = sorted(_as_ze_string(filter_df["ze2010"]).unique())
    if len(codes) != data["R"]:
        raise ValueError(f"{len(codes)} ZE codes in filter_N_upstream.csv but R = "
                         f"{data['R']} regions in the model.")
    labels = pd.DataFrame({"index": np.arange(1, data["R"] + 1), "ze2010": codes})
    france = data.get("france")
    if france is not None and "ze2010_name" in france.columns:
        names = france[["ze2010", "ze2010_name"]].copy()
        names["ze2010"] = _as_ze_string(names["ze2010"])
        labels = labels.merge(names.drop_duplicates("ze2010"), on="ze2010", how="left")
    else:
        labels["ze2010_name"] = labels["ze2010"]
    labels["ze2010_name"] = labels["ze2010_name"].fillna(labels["ze2010"])
    return labels


def ze_name_of_code(data):
    """ZE code -> display name, for anything keyed on the code rather than the index."""
    labels = _region_labels(data)
    return dict(zip(labels["ze2010"], labels["ze2010_name"]))


def aa_display_names(data):
    """
    Attraction-area display names, in AA column order.

    Area `a` is anchored on the a-th downstream region, so its name is that commuting
    zone's name — which is what a reader recognises, unlike the bare ZE code.
    """
    name_of_code = ze_name_of_code(data)
    return [name_of_code.get(str(a), str(a)) for a in data["aa_names"]]


PARTS_ALL = ("counts", "moments", "inference", "jacobian", "sensitivity", "geography",
             "firm")

# Requesting a part pulls in what it is built from. `moments` needs the count target to
# complete the empirical vector; `inference` and `jacobian` are read against the moment
# layout, so both imply it.
PART_IMPLIES = {"moments": ("counts",),
                "inference": ("moments", "counts"),
                "jacobian": ("moments", "counts"),
                "sensitivity": ("moments", "counts")}


def resolve_parts(parts):
    """Expand a requested part set through PART_IMPLIES, and refuse an unknown name."""
    if parts is None:
        return set(PARTS_ALL)
    wanted = {parts} if isinstance(parts, str) else set(parts)
    unknown = wanted - set(PARTS_ALL) - {"core", "all"}
    if unknown:
        raise ValueError(f"unknown part(s) {sorted(unknown)}; known parts are "
                         f"{list(PARTS_ALL)} (plus 'core' and 'all')")
    if "all" in wanted:
        return set(PARTS_ALL)
    wanted.discard("core")
    for _ in range(len(PART_IMPLIES) + 1):
        grown = set(wanted)
        for part in list(wanted):
            grown |= set(PART_IMPLIES.get(part, ()))
        if grown == wanted:
            break
        wanted = grown
    return wanted


def load_granular_data(industry, mu=2, base="..", profile_T=True, ca_level="aa",
                       granular=True, relax_n_lo=False, optimizer="pso", K=-1,
                       k_max=None, parts=None, kappa=True):
    """
    Load everything the granular / attraction-area reporting needs.

    Parameters
    ----------
    industry : str          "aero", "auto", ...
    mu       : 1 or 2       1 = step1 best parameters (mu_1), 2 = step3 best
                            parameters (mu_2, the efficient estimate).
    K        : int          which stage column of best_simulated_moments to read
                            (-1 = the last, i.e. the final stage of the step).
    parts    : iterable     which optional blocks to read. `None` (the default) reads
                            everything, as this function always did. Naming a subset
                            makes the loader SPECIALISED: a part that is not asked for
                            is not read, so its files need not exist and its keys come
                            back as `None`.

                              core       always: dimensions, the attraction-area
                                         structure, the empirical targets, `best_params`,
                                         `post_hoc_N_hat`, the granular diagnostics, the
                                         moment/parameter labels and MOMENT_MASK. This is
                                         everything `extended_parameters`,
                                         `sourcing_geometry` and `simulate_economy` need.
                              counts     G_K.csv -> Gbar_s(0) and the curve
                              moments    best_simulated_moments.npy and the block dicts
                              inference  the FAKE-WORLD artefacts: step2/world_residuals.npz,
                                         Omega_world / W_world (block 4), and the
                                         parameter draws of <inf_step>/inference_world/
                              jacobian   jacobian_all*.npy and its axes (no longer
                                         written by the Julia code; None on a current tree)
                              geography  france.gpkg, regional wages
                              firm       suppliers.parquet and Julia's own draws --
                                         needed ONLY by `check_against_julia`, since the
                                         economy is simulated here from `theta+`

                            `model_report.ipynb` asks for everything but `firm`;
                            `tests_counterfactuals.ipynb` asks for
                            `("core", "geography")` and therefore opens a tree that
                            carries no moment, inference or Jacobian artefacts at all.

    Returns a flat dict meant to be splatted into the namespace, exactly like
    `load_industry_data` in analysis.ipynb:

        data = load_granular_data(industry, mu=2)
        globals().update(data)        # in a script / notebook cell

    which binds `input_folder`, `folder`, `coefs`, `agg_labor_share`, `epsilon`,
    `agg_industry_share`, `emp_pi_r`, `reg_coef`, `emp_gamma_ls`, `X_dr`,
    `filter_N_upstream_df`, `france`, `idf_ze`, `N_downstream`,
    `regional_wage`, `best_params`, `best_simulated_moments`,
    `suppliers`, `empirical_moments`, `empirical_moments_dict`
    and `best_simulated_moments_dict` under the same names the old notebook used —
    plus the granular / attraction-area additions (`emp_gamma_aa`, `AA_ACTIVE`,
    `CELL_MASK`, `aa_of_ze`, `aa_names`, `T_REF_AA`, `G_target`, `se_empirical`,
    `se_simulated`, `granular_diagnostics`, ...). The gamma blocks are at the
    ATTRACTION-AREA level and keyed `gamma_aa`, not `emp_gamma_ls`.
    """
    if mu not in MU_DIRS: # Load the best parameter index, either one or two. 
        raise ValueError(f"mu must be 1 (step1) or 2 (step3), got {mu}")
    parts = resolve_parts(parts)
    # Highest K kept from the count-curve inputs. Defaults to the notebook constant, so
    # raising COUNT_CURVE_K_MAX in the Constants cell widens everything at once.
    k_max = int(globals().get("COUNT_CURVE_K_MAX", 3)) if k_max is None else int(k_max)
    step_dir, inf_step = MU_DIRS[mu]

    input_folder = Path(base) / f"baseline_{industry}"
    folder = reporting_folder(industry, base=base, profile_T=profile_T, ca_level=ca_level,
                              granular=granular, relax_n_lo=relax_n_lo, optimizer=optimizer,
                              kappa=kappa)
    if not folder.exists():
        found = sorted(file_path.name for file_path in Path(base).glob(f"reporting_{industry}_*") if file_path.is_dir())
        hint = ("existing trees for this industry: " + ", ".join(found)) if found else \
               f"no reporting_{industry}_* tree under {Path(base).resolve()}"
        raise FileNotFoundError(f"run folder {folder} not found — {hint}. Set relax_n_lo "
                                "(_nlo1) / profile_head (_profhead) to the flags the run was "
                                "launched with; a tree without _kappa is pre-kappa and not readable.")

    n_coef = int(np.load(folder / "n_reg_coef.npy")) # Number of regression coefficients. 
    n_tau_path = folder / "n_tau.npy" 
    n_tau = int(np.load(n_tau_path)) if n_tau_path.exists() else n_coef  # Number of coefficients parametrizing the trade costs.  

    # ---------------------------------------------------------------- targets
    coefs = pd.read_csv(input_folder / "stats.csv")
    epsilon = _read_named_value(coefs, "epsilon")
    agg_labor_share = _read_named_value(coefs, "labor_share")
    if epsilon is None or agg_labor_share is None:
        raise KeyError("stats.csv must carry `epsilon` and `labor_share` "
                       "(rows: epsilon, labor_share, reg_coef_cloglog_1)")

    agg_industry_share = np.load(input_folder / "input_share.npy").ravel()
    emp_gamma_ls = np.load(input_folder / "emp_gamma_ls.npy")          # (S, R)
    filter_N_upstream = np.load(input_folder / "filter_N_upstream.npy")  # (S, R), binary
    domestic_share = np.load(input_folder / "domestic_share.npy").ravel()
    N_downstream = np.load(input_folder / "N_downstream_per_region.npy").ravel()
    S, R = filter_N_upstream.shape

    X_dr = pd.read_csv(input_folder / "X_dr.csv")
    if "ze2010" in X_dr.columns:
        X_dr["ze2010"] = _as_ze_string(X_dr["ze2010"])
    emp_pi_r = X_dr["X_dr"].values[N_downstream != 0].astype(float)
    emp_pi_r = emp_pi_r / emp_pi_r.sum()

    # Block 4 = [beta_d (n_coef) ; beta_kappa (1)]: the CELL-level cloglog of has_sup on
    # the distance regressor(s) and ln n_ls, sector x AA fixed effect -- the same files
    # load_parameters.jl reads, in the same order.
    beta_d = np.load(input_folder / f"beta_d_{n_coef}.npy").astype(float).ravel()
    beta_kappa = np.load(input_folder / f"beta_kappa_{n_coef}.npy").astype(float).ravel()
    if beta_d.size != n_coef or beta_kappa.size != 1:
        raise ValueError(f"beta_d_{n_coef}.npy has {beta_d.size} entries (expected {n_coef}) "
                         f"and beta_kappa_{n_coef}.npy {beta_kappa.size} (expected 1)")
    reg_coef = np.concatenate([beta_d, beta_kappa])
    reg_coef_labels = (["]20,50]", "]50,100]", "]100,150]", "]150,200]", ">200"]
                       if n_coef == 5 else
                       ["]50,100]", "]100,150]", "]150,200]", ">200"] if n_coef == 4
                       else ["ln d"]) + ["ln n"]

    # ------------------------------------------------- attraction-area mapping
    AA_link = np.load(input_folder / "attraction_area_linkages.npy")   # (R, R_downstream)
    if AA_link.shape[0] != R:
        raise ValueError(f"attraction_area_linkages.npy is {AA_link.shape}, expected ({R}, R_d)")
    if not np.all(AA_link.sum(axis=1) == 1):
        raise ValueError("every ZE must belong to exactly one attraction area")
    aa_of_ze = AA_link.argmax(axis=1)          # 0-based AA index per ZE
    n_AA = AA_link.shape[1]
    aa_names = _aa_names(X_dr, n_AA)

    # 𝒜⁺_s : (sector, AA) hosting at least one observed supplier; CELL_MASK: simulated cells
    # a cell has supplier sales in the data <=> its observed sourcing share is positive
    if emp_gamma_ls.shape != filter_N_upstream.shape:
        raise ValueError(f"emp_gamma_ls.npy is {emp_gamma_ls.shape} but filter_N_upstream.npy "
                         f"is {filter_N_upstream.shape}; both must be on the same (S, R) grid")
    supplier_cells = (filter_N_upstream == 1) & (emp_gamma_ls > 0)
    AA_ACTIVE = np.zeros((S, n_AA), dtype=bool)
    for s in range(S):
        AA_ACTIVE[s, aa_of_ze[supplier_cells[s]]] = True
    CELL_MASK = (filter_N_upstream == 1) & AA_ACTIVE[:, aa_of_ze]

    # gamma target at the AA level: sum over EVERY cell of the area, controls included
    emp_gamma_aa = np.zeros((S, n_AA))                                 # (S, n_AA)
    for s in range(S):
        np.add.at(emp_gamma_aa[s], aa_of_ze[CELL_MASK[s]], emp_gamma_ls[s][CELL_MASK[s]])

    # reference AA per sector = largest empirical share among the active columns
    T_REF_AA = np.full(S, -1, dtype=int)
    for s in range(S):
        active_areas = np.flatnonzero(AA_ACTIVE[s])
        if active_areas.size:
            T_REF_AA[s] = active_areas[np.argmax(emp_gamma_aa[s, active_areas])]

    # ------------------------------------------------- the foreign competitor
    # One per sector (load_parameters.jl): f_s = 1 - domestic_share[s] is the import share,
    # the LAST T column (FOREIGN_COL) holds Phi^F_s and the gamma block's foreign row is
    # f_s. A sector imports when f_s > 0 and it has an active area. The gamma target is a
    # share of TOTAL purchases: the areas sum to 1 - f_s, rescaled here as Julia does.
    foreign_share = np.clip(1.0 - domestic_share.astype(float), 0.0, 1.0)
    FOREIGN_ACTIVE = (foreign_share > 1e-12) & AA_ACTIVE.any(axis=1)
    for s in range(S):
        domestic_total = emp_gamma_aa[s].sum()
        if domestic_total > 0 and abs(domestic_total - (1.0 - foreign_share[s])) > 1e-12:
            emp_gamma_aa[s] *= (1.0 - foreign_share[s]) / domestic_total
    n_Tcol = n_AA + 1                                   # areas + the foreign column
    T_ACTIVE = np.hstack([AA_ACTIVE, FOREIGN_ACTIVE[:, None]])
    emp_gamma_T = np.hstack([emp_gamma_aa, np.where(FOREIGN_ACTIVE, foreign_share, 0.0)[:, None]])

    # ------------------------------------------------------------- count moment
    G_target = np.full(S, np.nan)
    N_supplier_s = np.full(S, np.nan)
    sector_codes = None
    G_curve, G_curve_K, G_pmf = None, None, None
    gk_path = input_folder / "G_K.csv"
    if granular and "counts" in parts:
        # G_K.csv is the data pipeline's `t_G` (Step 6 of the build): `G` is the share
        # of a sector's active cells with EXACTLY K suppliers (the pmf, summing to 1
        # over K), `G_cs` its cumulative sum when present. The K = 0 row is the
        # targeted moment Gbar_s(0).
        # Names are matched case-insensitively, as load_parameters.jl does.
        #
        # The WHOLE curve is kept, not only the targeted row: G_s(K) for K >= 1 is a
        # genuinely untargeted check on the count distribution (gate V8 of
        # documentation/granular_validation.md), and load_parameters.jl reads the K = 0
        # row alone, so nothing on the Julia side carries it.
        G_K = pd.read_csv(gk_path)
        col_sector = _gk_col(G_K, ("a129", "sector"))
        col_K = _gk_col(G_K, ("k",))
        col_G = _gk_col(G_K, ("g",))
        col_N_supplier = _gk_col(G_K, ("n_supplier_s", "n_supplier", "n_suppliers"))
        sector_codes = sorted(G_K[col_sector].unique())
        if len(sector_codes) != S:
            raise ValueError(f"G_K.csv covers {len(sector_codes)} sectors, expected S = {S}")
        G_curve_K = sorted({int(k) for k in G_K[col_K].astype(float).round()})
        col_cumulative = next((c for c in G_K.columns if c.strip().lower() == "g_cs"), None)
        # One row per (sector, K) in the file: the curve is that table laid out as
        # sectors x K, in the model's sector order. A (sector, K) pair the file does not
        # list stays missing here and is read as a zero share below.
        K_rounded = G_K[col_K].astype(float).round().astype(int)
        sectors_with_K0 = set(G_K.loc[K_rounded == 0, col_sector])
        missing_K0 = [code for code in sector_codes if code not in sectors_with_K0]
        if missing_K0:
            raise ValueError(f"G_K.csv has no K=0 row for A129={missing_K0[0]}")
        curve_table = (G_K.assign(K_rounded=K_rounded, share=G_K[col_G].astype(float))
                       .pivot_table(index=col_sector, columns="K_rounded", values="share",
                                    aggfunc="last")
                       .reindex(index=sector_codes, columns=G_curve_K))
        G_curve = curve_table.to_numpy(dtype=float)
        G_target = G_curve[:, G_curve_K.index(0)].copy()
        # The number of observed suppliers is a sector constant, repeated on every K row.
        N_supplier_s = (G_K.groupby(col_sector)[col_N_supplier].first()
                        .reindex(sector_codes).astype(float).to_numpy())

        # G is already the pmf p_s(K) = Pr[K_ls = K], which is what the reporting plots
        # (the CDF would re-test the fitted K = 0 level in every panel). It is read as
        # is -- never differenced. When the file carries G_cs, check it is the cumsum
        # of G: that is the only thing pinning down which convention `G` follows.
        if col_cumulative is not None:
            for code in sector_codes:
                sector_rows = G_K[G_K[col_sector] == code].sort_values(col_K)
                gap = np.max(np.abs(np.cumsum(sector_rows[col_G].astype(float).values)
                                    - sector_rows[col_cumulative].astype(float).values))
                if gap > 1e-8:
                    raise ValueError(f"G_K.csv, A129={code}: G_cs is not the cumulative sum "
                                     f"of G (max gap {gap:.2e}) -- the pmf reading of G "
                                     "is then not the pipeline's.")
        keep = [i for i, k in enumerate(G_curve_K) if k <= k_max]
        G_curve_K = [G_curve_K[i] for i in keep]
        G_curve = G_curve[:, keep]
        # A (sector, K) row absent from G_K.csv is a K no cell has: share 0 (the
        # pipeline's own convention, Step 13 of the build).
        missing = ~np.isfinite(G_curve)
        if missing.any():
            print(f"[load] G_K.csv has no row for "
                  f"{[(sector_codes[i], G_curve_K[j]) for i, j in zip(*np.where(missing))]}; "
                  "read as a share of 0.")
            G_curve[missing] = 0.0
        G_pmf = G_curve

    # sector names, in the model's 1..S order (sorted A129), as in load_parameters.jl
    filter_df_path = input_folder / "filter_N_upstream.csv"
    if filter_df_path.exists():
        filter_N_upstream_df = pd.read_csv(filter_df_path)
        filter_N_upstream_df["ze2010"] = filter_N_upstream_df["ze2010"].astype(str).str.zfill(4)
        sector_names = [str(a) for a in sorted(filter_N_upstream_df["A129"].unique())]
    else:
        filter_N_upstream_df = None
        sector_names = [str(s + 1) for s in range(S)]
    if len(sector_names) != S:
        sector_names = [str(s + 1) for s in range(S)]

    # G_target is ordered by G_K.csv's OWN sorted A129, the labels by
    # filter_N_upstream.csv's. load_parameters.jl does the same and asserts only that
    # the two have the same LENGTH, so a differing code SET silently assigns each
    # sector another sector's zero-supplier target — the count moment then reads as a
    # fit failure (or a suspiciously good fit) in the wrong sector, in Julia and here
    # alike. Compare the sets, not the counts.
    if granular and sector_codes is not None and filter_N_upstream_df is not None:
        count_curve_sectors = [str(a) for a in sector_codes]
        if count_curve_sectors != list(sector_names):
            raise ValueError(
                "G_K.csv and filter_N_upstream.csv do not enumerate the same sectors:\n"
                f"  G_K.csv            : {count_curve_sectors}\n"
                f"  filter_N_upstream  : {list(sector_names)}\n"
                "Gbar_s(0) would be attached to the wrong sector (in load_parameters.jl too).")

    # ------------------------------------------- n_ls: the density term of T_ls
    # T_ls = T_{a(l)s} * n_ls^kappa (load_parameters.jl SECTION 10), so the cell-level
    # comparative advantage needs ln n_ls on every simulated cell. Keyed exactly as the
    # Julia loader keys it: sectors by the sorted A129 strings of filter_N_upstream.csv,
    # ZE by the sorted 4-character codes.
    log_n_ls = None
    cells_path = input_folder / "cells_ls.csv"
    if cells_path.exists() and filter_N_upstream_df is not None:
        cells_df = pd.read_csv(cells_path)
        sector_keys = sorted(filter_N_upstream_df["A129"].astype(str).unique())
        ze_keys = sorted(filter_N_upstream_df["ze2010"].map(_ze_key).unique())
        if len(sector_keys) == S and len(ze_keys) == R:
            sector_index_of = {k: i for i, k in enumerate(sector_keys)}
            ze_index_of = {k: i for i, k in enumerate(ze_keys)}
            log_n_ls = np.zeros((S, R))
            sector_index = cells_df["A129"].astype(str).map(sector_index_of)
            ze_index = cells_df["ze2010"].map(_ze_key).map(ze_index_of)
            ok = sector_index.notna() & ze_index.notna()
            sector_index, ze_index = sector_index[ok].astype(int).values, ze_index[ok].astype(int).values
            n_firms = cells_df.loc[ok, "n_ls"].astype(float).values
            keep = CELL_MASK[sector_index, ze_index]
            log_n_ls[sector_index[keep], ze_index[keep]] = np.log(np.maximum(n_firms[keep], 1.0))
            has_n_ls = np.zeros((S, R), dtype=bool)
            has_n_ls[sector_index[keep], ze_index[keep]] = True
            if (CELL_MASK & ~has_n_ls).any():
                print(f"[load] cells_ls.csv has no n_ls for {int((CELL_MASK & ~has_n_ls).sum())} "
                      "simulated cells; ln n_ls set to 0 there (Julia refuses such a tree).")
    elif kappa:
        print(f"[load] {cells_path} not found: T_ls = T_a n_ls^kappa cannot be gathered "
              "to cells, so the Ricardian geometry and the economy are unavailable.")

    # ------------------------------------------------ FULL empirical moment vector
    blocks_emp = [np.array([agg_labor_share], dtype=float),
                  agg_industry_share.astype(float),
                  emp_pi_r,
                  reg_coef.astype(float),
                  emp_gamma_T.ravel()]             # C-order on (S, n_AA + 1) == Julia vec((T_COL_DIM, S))
    if granular:
        blocks_emp.append(G_target)
    block_sizes = [b.size for b in blocks_emp]
    empirical_moments_full = np.concatenate(blocks_emp)

    # MOMENT_MASK, exactly as load_parameters.jl builds it
    mask = np.ones(empirical_moments_full.size, dtype=bool)
    # Indices at where each moment starts.
    off_ind, off_pi = 1, 1 + S
    off_reg = off_pi + emp_pi_r.size
    off_gam = off_reg + reg_coef.size
    mask[off_ind] = False                              # first industry share
    mask[off_pi] = False                               # first pi_r
    mask[off_gam:off_gam + S * n_Tcol] = T_ACTIVE.ravel()
    for s in range(S):
        if T_REF_AA[s] >= 0:
            mask[off_gam + s * n_Tcol + T_REF_AA[s]] = False

    # ------------------------------------------------------- simulated moments
    # Read only when asked for: a notebook that reports no moment fit does not need
    # `run_reporting` to have run on this step, and used to fail here when it had not.
    best_simulated_moments = best_parameters_list = best_params = None
    emp_blocks = sim_blocks = stage_blocks = None
    keys = ["agg_labor_share", "agg_industry_share", "emp_pi_r", "reg_coef", "gamma_aa"]
    if granular:
        keys.append("G0")
    if "moments" not in parts:
        sim_vec = None
    else:
      sim_path = folder / step_dir / "best_simulated_moments.npy"
      if not sim_path.exists():
          raise FileNotFoundError(
            f"{sim_path} not found. main.jl writes it via run_reporting(<run>/{step_dir}); "
            "for mu = 2 the step-3 reporting call must have run.")
      best_simulated_moments = np.load(sim_path)
      if best_simulated_moments.ndim == 1:
        best_simulated_moments = best_simulated_moments[:, None]
      if best_simulated_moments.shape[0] != empirical_moments_full.size:
        raise ValueError(
            f"simulated moment vector has {best_simulated_moments.shape[0]} rows but the "
            f"empirical layout has {empirical_moments_full.size}. The run's "
            "granular/ca_level flags and the ones passed here disagree.")
      sim_vec = best_simulated_moments[:, K]

      edges = np.cumsum(block_sizes)[:-1]
      emp_blocks = dict(zip(keys, np.split(empirical_moments_full, edges)))
      sim_blocks = dict(zip(keys, np.split(sim_vec, edges)))
      # The gamma block is (S, n_AA + 1): the areas, then the import share.
      for blocks in (emp_blocks, sim_blocks):
          gamma_block = blocks["gamma_aa"].reshape(S, n_Tcol)
          blocks["gamma_aa"], blocks["foreign_share"] = gamma_block[:, :n_AA], gamma_block[:, n_AA]

      # All stages, block by block, with the STAGE axis last — the layout
      # analysis.ipynb's `best_simulated_moments_dict[key][..., K]` expects.
      stage_blocks = {}
      for k, b in zip(keys, np.split(best_simulated_moments, edges, axis=0)):
        if k == "gamma_aa":
          gamma_block = b.reshape(S, n_Tcol, b.shape[1])
          stage_blocks["gamma_aa"], stage_blocks["foreign_share"] = gamma_block[:, :n_AA], gamma_block[:, n_AA]
        else:
          stage_blocks[k] = b

    # `best_params` is CORE: it carries theta+'s head, so the economy needs it whatever
    # else is being reported.
    bp_path = folder / step_dir / "best_parameters_list.npy"
    if bp_path.exists():
        best_parameters_list = np.load(bp_path)
        best_params = best_parameters_list[:, K] if best_parameters_list.ndim > 1 \
            else best_parameters_list

    # ------------------------------------------- inference: fake worlds -------
    # The Julia code reads no data covariance any more: the weighting and the inference
    # come from FAKE WORLDS drawn from the model (inference.jl).
    #
    #   step2/world_residuals.npz  B worlds at theta_hat_1, each ONE network of N_hat_s
    #       varieties with i.i.d. draws: its own beta (`beta_world`), the continuum
    #       simulator's beta at the recalibrated (T, A, N) (`beta_sim`), their difference
    #       `e`, and `e_data` = data - simulated at theta_hat_1. Omega_world = Cov(e),
    #       W_world = inv(Omega_world), block 4 only.
    #   <inf_step>/inference_world/world_inference_draws.npz  the WHOLE chain
    #       re-estimated in each world (step2 at theta_hat_1 with W = I, step3 at
    #       theta_hat_2 with W_world): one row of `params` per usable world, labelled by
    #       world_inference_labels.txt (alpha, kappa, N_s, T, A, chi_L, chi_s).
    n_gamma_kept = int(mask[off_gam:off_gam + S * n_Tcol].sum())
    world_residuals = Omega_world = W_world = world_inference = None
    if "inference" in parts:
        file_path = folder / "step2" / "world_residuals.npz"
        if file_path.exists():
            with np.load(file_path) as archive:
                world_residuals = {k: archive[k] for k in archive.files}
            for k in ("e", "beta_world", "beta_sim", "N_hat_world"):
                if k in world_residuals and world_residuals[k].ndim == 1:
                    world_residuals[k] = world_residuals[k][:, None]
            if world_residuals["e"].shape[1] != reg_coef.size:
                raise ValueError(f"{file_path} carries {world_residuals['e'].shape[1]} block-4 "
                                 f"entries, the layout rebuilt here has {reg_coef.size}.")
        for name in ("Omega_world", "W_world"):
            artefact_path = folder / "step2" / f"{name}.npy"
            if artefact_path.exists():
                matrix = np.atleast_2d(np.load(artefact_path))
                if name == "Omega_world":
                    Omega_world = matrix
                else:
                    W_world = matrix
        artefact_path = folder / inf_step / "inference_world" / "world_inference_draws.npz"
        labels_path = artefact_path.parent / "world_inference_labels.txt"
        if artefact_path.exists() and labels_path.exists():
            with np.load(artefact_path) as archive:
                world_inference = {k: archive[k] for k in archive.files}
            world_inference["labels"] = [l for l in labels_path.read_text().splitlines() if l.strip()]
            param_draws = np.atleast_2d(world_inference["params"])
            if param_draws.shape[1] != len(world_inference["labels"]):
                raise ValueError(f"{artefact_path.name} has {param_draws.shape[1]} parameter columns but "
                                 f"{labels_path.name} lists {len(world_inference['labels'])}.")
            world_inference["params"] = param_draws
            world_inference["folder"] = artefact_path.parent

    # ----------------------------------------- moment and parameter labels ----
    # Both axes of the Jacobian, rebuilt exactly as load_parameters.jl SECTION 13 does.
    #   rows    : the MASKED moment vector, six blocks;
    #   columns : the IDENTIFIED parameters, layout [chi_L | chi_s(S) | A(R_d) |
    #             alpha(N_TAU) | kappa | T(active (s, AA or foreign), s-major)], minus the S+2 directions
    #             the internal normalisations kill (chi_s[1], A[1], each sector's
    #             reference T).
    moment_labels, moment_block_sizes = [], []
    moment_labels.append("labor")
    moment_block_sizes.append(1)
    moment_labels += [f"chi_s[{sector_names[s]}]" for s in range(1, S)]
    moment_block_sizes.append(S - 1)
    moment_labels += [f"pi_r[{a}]" for a in aa_names[1:]]
    moment_block_sizes.append(len(aa_names) - 1)
    moment_labels += [f"beta_d[{label}]" for label in reg_coef_labels[:-1]] + ["beta_kappa"]
    moment_block_sizes.append(reg_coef.size)
    gamma_free = T_ACTIVE.copy()
    for s in range(S):
        if T_REF_AA[s] >= 0:
            gamma_free[s, T_REF_AA[s]] = False
    moment_labels += [(f"f_foreign[{sector_names[s]}]" if a == n_AA else
                       f"gamma[{sector_names[s]}-AA{aa_names[a]}]")
                      for s in range(S) for a in range(n_Tcol) if gamma_free[s, a]]
    moment_block_sizes.append(int(gamma_free.sum()))
    if granular:
        moment_labels += [f"G0[{sector_names[s]}]" for s in range(S)]
        moment_block_sizes.append(S)

    param_labels, param_block_sizes = [], []
    param_labels.append("chi_L")
    param_block_sizes.append(1)
    param_labels += [f"chi_s[{sector_names[s]}]" for s in range(1, S)]   # chi_s[1] normalised out
    param_block_sizes.append(S - 1)
    param_labels += [f"A[{a}]" for a in aa_names[1:]]                      # A[1] normalised out
    param_block_sizes.append(len(aa_names) - 1)
    param_labels += (["alpha"] if n_tau == 1 else [f"alpha_{b + 1}" for b in range(n_tau)])
    param_labels += ["kappa"]
    param_block_sizes.append(n_tau + 1)
    param_labels += [(f"PhiF[{sector_names[s]}]" if a == n_AA else
                      f"T[{sector_names[s]}-AA{aa_names[a]}]")
                     for s in range(S) for a in range(n_Tcol) if gamma_free[s, a]]
    param_block_sizes.append(int(gamma_free.sum()))

    moment_block_names = ["Labor share", "Industry shares", "Downstream sales",
                          "Extensive margin", "Regional sourcing shares"] \
        + (["Zero-supplier share"] if granular else [])
    param_block_names = ["chi_L", "Industry shares", "Productivity",
                         "Trade cost", "Comparative advantage"]

    # ------------------------------------------------------------- Jacobian ---
    # Rows = the masked moments, columns = the parameters. The full free-parameter
    # Jacobian is saved for diagnostics at both estimates, even under profile_T (where
    # INFERENCE runs on the alpha-only profiled Jacobian instead).
    #
    # Under GRANULAR the Julia side appends the S VARIETY-COUNT columns dm/dN_s on the
    # right (compute_jacobian's `append_N_s`): N_s is calibrated by the integer bisection
    # exactly as T is calibrated by the Sinkhorn inversion, so both belong on the
    # parameter axis. `<file>_n_s_columns.npy` records how many were appended; older
    # trees have none, and the width settles it.
    # The Jacobian and the one-parameter profiles are written by sensitivity.jl under
    # <inf_step>/sensitivity/ (step2 at theta_hat_1, step3 at theta_hat_2):
    #   jacobian.npz  J (mean over K independent draw sets), J_sd (sd of ONE draw set's
    #                 derivative across those sets), J_est (estimation draws), m_base,
    #                 theta (the column's parameter value), kind (1 head, 2 alpha,
    #                 3 kappa, 4 T, 5 N_s), K, h.
    #   profile_<p>.npz  one per swept parameter (see `load_sensitivity`).
    # The Jacobian REPLACES the parameter axis rebuilt above: its columns are Julia's
    # (head without the normalised-out entries, alpha, kappa, the identified T, N_s).
    jacobian_entries = {k: None for k in ("J", "J_elast", "J_sd", "J_elast_sd",
                             "jacobian_param_indices")}
    n_N_cols = 0
    J_est = sensitivity = None
    sens_dir = folder / inf_step / "sensitivity"
    if ("jacobian" in parts or "sensitivity" in parts) and sens_dir.exists():
        sensitivity = load_sensitivity(sens_dir, n_moments=int(mask.sum()))
        jacobian = sensitivity.get("jacobian")
        if jacobian is not None:
            jacobian_entries.update(J=jacobian["J"], J_sd=jacobian["J_se"], J_elast=jacobian["J_elast"],
                       J_elast_sd=jacobian["J_elast_se"])
            J_est = jacobian["J_est"]
            param_labels = list(jacobian["labels"])
            param_block_names, param_block_sizes = jacobian["block_names"], jacobian["block_sizes"]
            n_N_cols = int(np.sum(jacobian["kind"] == 5))

    # ------------------------------------------------- granular diagnostics
    granular_diag = None
    diag_path = folder / inf_step / "granular_diagnostics.npz"
    if granular and diag_path.exists():
        with np.load(diag_path) as archive:
            granular_diag = {k: archive[k] for k in archive.files}

    # ------------------------------- optional artefacts (geography, panels) -----
    # Everything below is best-effort: it is what the DOWNSTREAM cells of the old
    # notebook reach for after `globals().update(data)`, and none of it is needed by
    # the moment reporting, so a missing file leaves a None rather than failing.
    idf_ze = ['1101', '1111', '1102', '1104', '1118', '1115', '1116', '1105',
              '1117', '1110', '1119', '1112', '1103', '1109', '1106', '1114',
              '1113', '1108', '1107']

    def _opt(fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except Exception:
            return None

    regional_wage = france = None
    if "geography" in parts:
        regional_wage = _opt(np.load, input_folder / "regional_wages.npy")
        france = None if gpd is None else _opt(
            lambda: gpd.read_file(input_folder / "france.gpkg", encoding="utf-8").sort_values(by="ze2010"))

    # Firm-level artefacts. main.jl writes them into the STEP folder of the estimate
    # they were computed at — step1 for theta_hat_1, step3 for theta_hat_2 — so `mu`
    # selects the simulated economy exactly as it selects the moments and the inference.
    # `post_hoc_N_hat` is CORE, not `firm`: the variety count is a coordinate of theta+,
    # so the economy needs it whether or not Julia's own realisation is on disk.
    n_hat_path = folder / step_dir / "post_hoc_N_hat.npy"
    post_hoc_N_hat = _opt(np.load, n_hat_path) if n_hat_path.exists() else None

    suppliers = suppliers_continuum = post_hoc_u = post_hoc_good_sr = w_srd_r = None
    post_hoc_u_foreign = None
    suppliers_path = folder / step_dir / "suppliers.parquet"
    if "firm" in parts:
      suppliers = _opt(pd.read_parquet, suppliers_path) if suppliers_path.exists() else None
    # The INFINITE-VARIETY benchmark, written beside it under GRANULAR: the same
    # economy solved on the estimation draws (`N_rho` per sector, one to two orders of
    # magnitude above the calibrated `N_hat_s`), i.e. the `N_s -> infinity` limit of the
    # same model. It is what `granularity_report` measures the finite-variety economy
    # against. Absent from a tree written before the post-hoc block drew varieties, in
    # which case `suppliers.parquet` IS that benchmark and carries no `replication`
    # column -- which is exactly how `supplier_count_check` diagnoses such a tree.
      continuum_path = folder / step_dir / "suppliers_continuum.parquet"
      suppliers_continuum = (_opt(pd.read_parquet, continuum_path)
                           if continuum_path.exists() else None)
    # The draws behind `suppliers.parquet`, stacked (N_max, n_good, n_rep), and the
    # (sector, cell) identity of each draw COLUMN. Julia's MersenneTwister generates
    # Float64 through dSFMT, which no numpy generator reproduces, so these two files are
    # what let `simulate_economy` rebuild THAT economy rather than a statistically
    # equivalent one -- i.e. what turns the comparison between the two implementations
    # into a test. Absent from a tree written before they were added.
      u_path = folder / step_dir / "post_hoc_u.npy"
      post_hoc_u = _opt(np.load, u_path) if u_path.exists() else None
      # the foreign competitor's draws, (N_max, S, n_rep): one bid per (variety, sector)
      uf_path = folder / step_dir / "post_hoc_u_foreign.npy"
      post_hoc_u_foreign = _opt(np.load, uf_path) if uf_path.exists() else None
      gsr_path = folder / step_dir / "post_hoc_good_sr.npy"
      post_hoc_good_sr = _opt(np.load, gsr_path) if gsr_path.exists() else None
      w_srd_r_path = folder / step_dir / "w_srd_r.npy"
      w_srd_r = _opt(np.load, w_srd_r_path) if w_srd_r_path.exists() else None

    return {
        # identity / paths
        "industry": industry, "mu": mu, "K": K, "parts": sorted(parts),
        "base": base,
        "input_folder": input_folder, "folder": folder,
        "step_dir": step_dir, "inference_step": inf_step,
        "granular": granular, "ca_level": ca_level,
        "post_hoc_u": post_hoc_u, "post_hoc_good_sr": post_hoc_good_sr,
        "post_hoc_u_foreign": post_hoc_u_foreign,
        # dimensions
        "S": S, "R": R, "n_AA": n_AA, "n_coef": n_coef, "n_tau": n_tau,
        "sector_names": sector_names, "sector_codes": sector_codes, "aa_names": aa_names,
        # structure
        "aa_of_ze": aa_of_ze, "AA_ACTIVE": AA_ACTIVE, "CELL_MASK": CELL_MASK,
        "T_REF_AA": T_REF_AA, "domestic_share": domestic_share,
        "foreign_share": foreign_share, "FOREIGN_ACTIVE": FOREIGN_ACTIVE,
        "T_ACTIVE": T_ACTIVE, "emp_gamma_T": emp_gamma_T,
        "filter_N_upstream": filter_N_upstream, "filter_N_upstream_df": filter_N_upstream_df,
        "X_dr": X_dr, "emp_gamma_ls": emp_gamma_ls,
        # scalars / targets
        "epsilon": epsilon, "agg_labor_share": agg_labor_share,
        "agg_industry_share": agg_industry_share, "emp_pi_r": emp_pi_r,
        "reg_coef": reg_coef, "reg_coef_labels": reg_coef_labels,
        "beta_d": beta_d, "beta_kappa": float(beta_kappa[0]), "log_n_ls": log_n_ls,
        "G_target": G_target, "N_supplier_s": N_supplier_s,
        # the WHOLE empirical count curve, Pr(K_ls <= K) by (sector, K); only the
        # K = 0 column is targeted, the rest is the untargeted fit check
        "G_curve": G_curve, "G_curve_K": G_curve_K,
        # the shares of cells with EXACTLY K suppliers, which is what is plotted
        "G_pmf": G_pmf,
        "emp_gamma_aa": emp_gamma_aa, "coefs": coefs,
        "d": "C30C" if industry == "aero" else "C29A",
        # moments
        "block_sizes": block_sizes, "moment_mask": mask,
        "empirical_moments_full": empirical_moments_full,
        "empirical_moments": empirical_moments_full.reshape(-1, 1),
        "reference_empirical_moments": blocks_emp,
        "best_simulated_moments": best_simulated_moments,
        "best_parameters_list": best_parameters_list, "best_params": best_params,
        "empirical_moments_dict": emp_blocks,
        "simulated_moments_dict": sim_blocks,
        # every stage, stage axis LAST — the analysis.ipynb layout
        # (`best_simulated_moments_dict[key][..., K]`)
        "best_simulated_moments_dict": stage_blocks,
        # geography / panels (None when the file is absent)
        "idf_ze": idf_ze, "france": france,
        "regional_wage": regional_wage, "N_downstream": N_downstream,
        "suppliers": suppliers, "suppliers_path": suppliers_path, "w_srd_r": w_srd_r,
        "suppliers_continuum": suppliers_continuum, "post_hoc_N_hat": post_hoc_N_hat,
        # labels / block layout (both Jacobian axes)
        "moment_labels": moment_labels, "moment_block_sizes": moment_block_sizes,
        "moment_block_names": moment_block_names,
        "param_labels": param_labels, "param_block_sizes": param_block_sizes,
        "param_block_names": param_block_names,
        # Jacobian (None when the file is absent)
        "J": jacobian_entries["J"], "J_elast": jacobian_entries["J_elast"], "J_sd": jacobian_entries["J_sd"],
        "J_elast_sd": jacobian_entries["J_elast_sd"], "n_N_cols": n_N_cols, "J_est": J_est,
        "sensitivity": sensitivity,
        "jacobian_param_indices": jacobian_entries["jacobian_param_indices"],
        # inference: fake worlds (None when the step has not run)
        "world_residuals": world_residuals, "Omega_world": Omega_world,
        "W_world": W_world, "world_inference": world_inference,
        # the pre-world inference objects, no longer written by the Julia code; kept as
        # None so a function that tests for them degrades instead of raising
        "Sigma_data": None, "Sigma_sim": None, "Omega": None, "W_step3": None,
        "se_empirical": None, "se_simulated": None,
        "G_curve_sim": None, "G_curve_sim_se": None, "G_curve_sim_K": None,
        "granular_diagnostics": granular_diag,
    }




# --- The Ricardian geometry: the estimated T and alpha, and the closed-form win
# probabilities -------------------------------------------------------------------
#
# These live HERE, beside the loader, rather than inside the comparative-advantage
# section, because TWO sections need them: that one, to weigh comparative advantage
# against distance, and the amplification section, to propagate a shock with one of
# the two forces switched off. A section must run on its own after the loader, so a
# helper with two consumers cannot sit in either of them.
#
# Everything is computed CELL by CELL — a cell is a (sector, region) pair. Comparative
# advantage is read off the cell's attraction area, `T[s, aa_of_ze[l]]`, which is the
# one line to change the day regions are allowed a T of their own; distance is always
# the region's own distance to the buyer.


def _theta_from_julia(root=None):
    """`theta` as `load_parameters.jl` sets it, parsed from that file.

    This is the authoritative value and not a convenience: line 63 of
    `load_parameters.jl` sets `theta` UNCONDITIONALLY, so it is what the economy on disk
    was solved at whatever `stats.csv` says. Reading it here removes a whole class of
    silent divergence — a notebook constant drifting from the estimator's — rather than
    leaving it to be noticed. Returns None when the file is not reachable.
    """
    import re
    for candidate in ([Path(root)] if root is not None else []) + [Path("."), Path(".."),
                                                              Path(__file__).parent
                                                              if "__file__" in globals()
                                                              else Path(".")]:
        julia_file = Path(candidate) / "load_parameters.jl"
        if not julia_file.exists():
            continue
        match = re.search(r"const\s+theta\s*=\s*\$\(\s*([0-9.eE+-]+)\s*\)", julia_file.read_text())
        if match:
            try:
                value = float(match.group(1))
            except ValueError:
                return None
            return value if np.isfinite(value) and value > 0 else None
    return None


def model_theta(data, verbose=False):
    """
    Frechet shape parameter, from `load_parameters.jl` first.

    Precedence is deliberate and is the opposite of what it was. `theta` is a CALIBRATED
    const in `load_parameters.jl`, so it is what every artefact in the run tree was
    produced at; `stats.csv` is an empirical input that the estimator does not read for
    this parameter. Preferring the Julia file therefore makes the notebook agree with the
    economy it is reporting on by construction. The two are compared when both exist and
    a disagreement is ANNOUNCED — it means one of the two is stale, which is exactly the
    failure that would otherwise scale every theta*alpha object by a silent ratio.

    `THETA_DEFAULT` remains as a last resort for a tree detached from the source.
    """
    julia_value = _theta_from_julia(data.get("base") if isinstance(data, dict) else None)
    stats_value = _read_named_value(data["coefs"], "theta") if data.get("coefs") is not None else None
    stats_value = float(stats_value) if stats_value is not None and np.isfinite(stats_value) and stats_value > 0 else None
    if julia_value is not None and stats_value is not None and abs(julia_value - stats_value) > 1e-12 and verbose:
        print(f"  [theta] load_parameters.jl says {julia_value:g}, stats.csv says {stats_value:g} — taking "
              "the Julia value, which is what the economy was solved at. One of the two "
              "is stale.")
    if julia_value is not None:
        return julia_value
    return stats_value if stats_value is not None else float(THETA_DEFAULT)


def unpack_estimated_T(data):
    """
    The estimated comparative advantage, reference-normalised exactly as
    `unpack_params` does, plus alpha.

    `best_params` is the RAW theta vector, layout
        [chi_L(1) | chi_s(S) | A(R_d) | alpha(N_TAU) | kappa(1) | T(active (s, col), s-major)],
    where the T block carries EVERY active (sector, column) including each sector's
    reference area and, for an importing sector, the foreign competitor Phi^F_s (the
    last column) — `unpack_params` divides the sector by its reference entry, which is
    what makes T_{ref,s} = 1 the gauge, Phi^F_s included. The length is asserted against the rebuilt
    layout, so a flag mismatch is an error rather than a silently shifted vector.

    Returns the AREA-level `T` (S, n_AA), the foreign scale `PhiF` (S, zero where the
    sector does not import) and the CELL-level `T_cell` (S, R) the model actually
    competes with, `T_ls = T_{a(l)s} * n_ls^kappa` (`gather_T_to_ze`), zero outside
    CELL_MASK. The Python economy has no foreign competitor yet: it reads `T` and
    `T_cell` only.
    """
    best_params = data.get("best_params")
    if best_params is None:
        raise FileNotFoundError(
            f"no best_params in {data['folder']}/{data['step_dir']}/ — "
            "best_parameters_list.npy is written by run_reporting.")
    best_params = np.asarray(best_params, dtype=float).ravel()

    S, n_AA, n_tau = data["S"], data["n_AA"], data["n_tau"]
    T_ACTIVE = data["T_ACTIVE"]                       # (S, n_AA + 1): areas + foreign
    R_d = len(data["aa_names"])                       # one area per downstream region
    n_T = int(T_ACTIVE.sum())
    expected = 1 + S + R_d + n_tau + 1 + n_T
    if best_params.size != expected:
        raise ValueError(
            f"best_params has {best_params.size} entries but the layout rebuilt here is "
            f"{expected} = 1 + {S} + {R_d} + {n_tau} + 1 (kappa) + {n_T} — the run's "
            "flags and the ones passed here disagree.")

    offset = 1 + S + R_d
    alpha = best_params[offset: offset + n_tau]
    kappa = float(best_params[offset + n_tau])
    T_full = np.zeros((S, n_AA + 1))
    T_full[T_ACTIVE] = best_params[offset + n_tau + 1:]           # C-order on (S, n_AA + 1) == s-major
    for s in range(S):
        reference_area = data["T_REF_AA"][s]
        if reference_area >= 0 and T_full[s, reference_area] > 0:
            T_full[s] = T_full[s] / T_full[s, reference_area]    # the gauge rescales Phi^F_s too
    T, PhiF = T_full[:, :n_AA], T_full[:, n_AA]
    log_n = data.get("log_n_ls")
    if log_n is None:
        if kappa != 0.0:
            raise FileNotFoundError("cells_ls.csv was not read, so T_ls = T_a n_ls^kappa "
                                    f"cannot be formed at kappa = {kappa:.4g}.")
        log_n = np.zeros((S, data["R"]))
    T_cell = T[:, data["aa_of_ze"]] * np.exp(kappa * log_n) * data["CELL_MASK"]
    return {"T": T, "PhiF": PhiF, "T_cell": T_cell, "alpha": alpha, "kappa": kappa,
            "log_n": log_n, "theta": model_theta(data)}


def _downstream_ze_index(data):
    """The 1..R ZE index of each attraction area, in AA column order."""
    N_downstream = np.asarray(data["N_downstream"]).ravel()
    downstream_index = np.flatnonzero(N_downstream != 0) + 1                 # 1-based, sorted — the model's order
    if downstream_index.size != data["n_AA"]:
        raise ValueError(f"{downstream_index.size} downstream regions in N_downstream_per_region.npy "
                         f"but {data['n_AA']} attraction areas.")
    return downstream_index


# --- Trade costs: ONE map from (alpha, distance) to log tau, whatever N_TAU is -------
#
# `build_tau` (model_CP.jl) is the only definition of tau, and everything below goes
# through `log_tau`, its port, so a section never writes `alpha * log d` itself:
#
#   N_TAU = 1   ln tau = alpha * ln d                          (the power law d^alpha)
#   N_TAU = K   ln tau continuous and piecewise linear in ln d, slope alpha_k between the
#               knots ln dbar_{k-1}, ln dbar_k (the bin centres), ln tau(dbar_0) = 0,
#               end segments extended at alpha_1 and alpha_K.
#
# Equal alpha_k give alpha*(ln d - ln dbar_0), i.e. d^alpha up to a factor common to a
# buyer's origins, which cancels in every share. `alpha` may be passed as a SCALAR, which
# is broadcast to every segment: `alpha=0` is therefore tau == 1 under any
# parametrisation, which is what the `No trade cost` regime means.

# Bin edges of the extensive-margin regression, by n_coef (load_parameters.jl).
TAU_BIN_EDGES = {4: (0.0, 50.0, 100.0, 150.0, 200.0),
                 5: (0.0, 20.0, 50.0, 100.0, 150.0, 200.0)}


def tau_log_knots(data):
    """
    `TAU_LOG_KNOTS` exactly as `load_parameters.jl` builds it: the log of the bin CENTRES,
    the last bin closed at the largest distance from any region to its nearest
    downstream region. `None` for the power law (N_TAU = 1). Cached on `data`.
    """
    n_tau = int(data["n_tau"])
    if n_tau == 1:
        return None
    if data.get("_tau_log_knots") is not None:
        return data["_tau_log_knots"]
    n_coef = int(data["n_coef"])
    if n_tau != n_coef or n_coef not in TAU_BIN_EDGES:
        raise ValueError(f"n_tau must be 1 or n_coef (4 or 5); got n_tau = {n_tau}, "
                         f"n_coef = {n_coef}.")
    distances = np.load(data["input_folder"] / "distances.npy")
    downstream = _downstream_ze_index(data)
    d_max = float(np.nanmax(np.nanmin(distances[:, downstream - 1], axis=1)))
    edges = np.array(TAU_BIN_EDGES[n_coef] + (d_max,))
    if d_max <= edges[-2]:
        raise ValueError(f"no region lies beyond {edges[-2]:g} km of its nearest downstream "
                         f"region (max {d_max:.1f}): the last bin is empty, as Julia refuses.")
    knots = np.log(0.5 * (edges[:-1] + edges[1:]))
    try:
        data["_tau_log_knots"] = knots
    except TypeError:
        pass
    return knots


def resolve_alpha(data, alpha=None):
    """The N_TAU-long alpha vector: the estimate (`None`), a scalar broadcast to every
    segment, or a full vector."""
    n_tau = int(data["n_tau"])
    if alpha is None:
        return np.asarray(unpack_estimated_T(data)["alpha"], dtype=float).ravel()
    alpha_vector = np.atleast_1d(np.asarray(alpha, dtype=float)).ravel()
    if alpha_vector.size == 1:
        return np.full(n_tau, float(alpha_vector[0]))
    if alpha_vector.size != n_tau:
        raise ValueError(f"alpha has {alpha_vector.size} entries against N_TAU = {n_tau}.")
    return alpha_vector


def log_tau(alpha, log_d, knots=None):
    """`ln tau` at `log_d` (any shape), the port of `build_tau`. `knots` is
    `tau_log_knots(data)`; it is ignored (and may be None) when alpha has one entry."""
    alpha_vector = np.atleast_1d(np.asarray(alpha, dtype=float)).ravel()
    log_distance = np.asarray(log_d, dtype=float)
    if alpha_vector.size == 1:
        return alpha_vector[0] * log_distance
    log_knots = np.asarray(knots, dtype=float)
    K = alpha_vector.size
    if log_knots.size != K + 1:
        raise ValueError(f"{K} segment elasticities need {K + 1} knots, got {log_knots.size}.")
    log_tau_value = alpha_vector[0] * (np.minimum(log_distance, log_knots[1]) - log_knots[0])
    for k in range(1, K):
        upper = log_distance if k == K - 1 else np.minimum(log_distance, log_knots[k + 1])
        log_tau_value = log_tau_value + np.where(log_distance > log_knots[k], alpha_vector[k] * (upper - log_knots[k]), 0.0)
    return log_tau_value


def tau_elasticity(alpha, log_d, knots=None):
    """`d ln tau / d ln d` at `log_d`: alpha for the power law, the elasticity of the
    segment `log_d` falls in when tau is binned (the left one at a knot)."""
    alpha_vector = np.atleast_1d(np.asarray(alpha, dtype=float)).ravel()
    log_distance = np.asarray(log_d, dtype=float)
    if alpha_vector.size == 1:
        return np.full(log_distance.shape, alpha_vector[0])
    log_knots = np.asarray(knots, dtype=float)
    return alpha_vector[np.searchsorted(log_knots[1:alpha_vector.size], log_distance, side="left")]




def sourcing_geometry(data, alpha=None, equalise_T=False, kappa=None):
    """
    The exact Ricardian win probabilities `rho[s][l, r]` and the pieces they are built
    from, per sector, over CELLS l = regions:

        psi_lr = T_ls * tau_lr^(-theta),   rho_lr = psi_lr / sum_l' psi_l'r,

    with `tau` from `log_tau` (the port of Julia's `build_tau`), so the same code serves
    the power law (N_TAU = 1) and binned trade costs (N_TAU = n_coef). `psi` is kept in
    LEVELS (not rescaled), in the units Julia's Phi^F_s is estimated in.

    `rho[s][:, r]` is the DOMESTIC geography: the probability that cell l wins a variety
    for buyer r GIVEN that a French cell wins it, so every column sums to one. The foreign
    competitor (Julia's Phi^F_s, a cost common to every buyer) scales every cell's
    unconditional probability by the same `1 - f_rs`; `f[s][r] = Phi^F_s / (sum_l psi_lr +
    Phi^F_s)` is carried beside it (zero for a sector without foreign competitor).

    Counterfactuals, all closed form, about the reorganisation of the DOMESTIC network:
      `alpha=0`            distance off (a scalar is broadcast to every segment, so tau == 1);
      `equalise_T=True`    every cell's T to one (both layers; Julia's `:equal_T`);
      `equalise_T="area"`  the AREA layer only, `T_ls = n_ls^kappa`;
      `kappa=k`            the density elasticity set to k (`kappa=0`: `T_ls = T_a`).
    In a regime each buyer's import share is held at its estimate by a buyer-specific
    foreign scale `PhiF_r[s][r] = Phi^F_s * Phi^dom_rs(regime) / Phi^dom_rs(estimate)`, as
    Julia's `incidence_stats` does, so no regime depends on the reference area that fixes
    the units of T.

    Per sector: `distance` (km, floored at 1 km), `log_tau`, `elasticity` = theta * d ln
    tau/d ln d, `log_psi`, `rho`, `phi_dom`, `PhiF`, `PhiF_r`, `f`. At the top level:
    `alpha` (N_TAU vector), `alpha_hat`, `knots`, and `theta_alpha`, the ONE scalar for
    statements written in theta*alpha: exactly theta*alpha under the power law, and under
    binned tau the rho- and purchase-weighted mean of the local elasticity.
    """
    if equalise_T not in (False, True, "area"):
        raise ValueError(f"equalise_T must be False, True or 'area', got {equalise_T!r}.")
    estimate = unpack_estimated_T(data)
    alpha_vector = resolve_alpha(data, alpha)
    a_hat = np.asarray(estimate["alpha"], dtype=float).ravel()
    knots = tau_log_knots(data)
    theta = estimate["theta"]
    kappa_value = estimate["kappa"] if kappa is None else float(kappa)
    if equalise_T is True and kappa is not None:
        raise ValueError("equalise_T=True already sets every T_ls to one; kappa is moot.")
    regime = bool(equalise_T) or not np.array_equal(alpha_vector, a_hat) or kappa_value != estimate["kappa"]
    estimate_geometry = sourcing_geometry(data) if regime and (estimate["PhiF"] > 0).any() else None

    distances = np.load(data["input_folder"] / "distances.npy")[:data["R"], :data["R"]]
    downstream = _downstream_ze_index(data)
    aa_of_ze, CELL_MASK = data["aa_of_ze"], data["CELL_MASK"]
    buyer_weights = np.asarray(data.get("emp_pi_r", []), dtype=float).ravel()
    buyer_weights = buyer_weights / buyer_weights.sum() if buyer_weights.size == downstream.size and buyer_weights.sum() > 0 else np.full(downstream.size, 1.0 / downstream.size)

    by_sector, theta_alpha_by_sector = {}, []
    for s in range(data["S"]):
        cells = np.flatnonzero(CELL_MASK[s])                       # 0-based ZE
        if cells.size == 0:
            continue
        # cell-level comparative advantage T_ls = T_{a(l)s} n_ls^kappa; `equalise_T`
        # sets every cell to one, as Julia's `:equal_T` regime does; "area" keeps n_ls^kappa
        if equalise_T is True:
            T_cell = np.ones(cells.size)
        else:
            density_term = np.exp(kappa_value * estimate["log_n"][s, cells])
            T_cell = density_term if equalise_T == "area" else estimate["T"][s, aa_of_ze[cells]] * density_term
        distance = np.maximum(distances[np.ix_(cells, downstream - 1)], 1.0)            # (n_cell, R_d)
        log_distance = np.log(distance)
        log_tau_lr = log_tau(alpha_vector, log_distance, knots)
        log_psi = np.log(np.maximum(T_cell, 1e-300))[:, None] - theta * log_tau_lr
        psi = np.exp(log_psi)                                       # levels: units of Phi^F
        Phi_domestic = psi.sum(axis=0, keepdims=True)
        rho = np.divide(psi, Phi_domestic, out=np.zeros_like(psi), where=Phi_domestic > 0)
        elasticity = theta * tau_elasticity(alpha_vector, log_distance, knots)
        theta_alpha_by_sector.append(float(((rho * elasticity).sum(axis=0)) @ buyer_weights))
        PhiF_s = float(estimate["PhiF"][s])
        if estimate_geometry is not None and PhiF_s > 0:
            estimate_sector = estimate_geometry["by_sector"][s]                     # hold f_rs at the estimate
            PhiF_r = PhiF_s * Phi_domestic[0] / estimate_sector["phi_dom"]
            import_share = estimate_sector["f"]
        else:
            PhiF_r = np.full(Phi_domestic.shape[1], PhiF_s)
            import_share = PhiF_s / (Phi_domestic[0] + PhiF_s)
        by_sector[s] = {"cells": cells, "areas": aa_of_ze[cells], "T_cell": T_cell,
                  "distance": distance, "log_tau": log_tau_lr, "elasticity": elasticity,
                  "log_psi": np.maximum(log_psi, np.log(1e-300)), "rho": rho,
                  "phi_dom": Phi_domestic[0], "PhiF": PhiF_s, "PhiF_r": PhiF_r, "f": import_share}
    return {"by_sector": by_sector, "alpha": alpha_vector, "theta": theta, "T": estimate["T"],
            "PhiF": estimate["PhiF"], "alpha_hat": a_hat, "knots": knots,
            "theta_alpha": float(np.mean(theta_alpha_by_sector)) if theta_alpha_by_sector else float(theta * alpha_vector.mean()),
            "downstream": downstream}


# ============================================================================
# Readers for a firm-level frame, in `suppliers.parquet`'s schema
# ----------------------------------------------------------------------------
# Both are consumed by the economy below (`extended_parameters` needs N_hat_s,
# `check_against_julia` needs the sector index) as well as by the sections, so
# they sit here rather than in either library.
# ============================================================================



def _parquet_sector_index(data, suppliers):
    """
    The model's 0-based sector index for every row of `suppliers.parquet`.

    The parquet's `A129` column holds the model's OWN index 1..S, not the A129 code —
    the column glossary says so, and `main.jl` writes `sectors = s`. It is read as such
    and then cross-checked: a column that does not fit 1..S is tried against
    `sector_names` before the function gives up, so a file written with real codes is
    NAMED rather than silently mapped onto the wrong sectors.
    """
    S = data["S"]
    a129_column = suppliers["A129"]
    if pd.api.types.is_numeric_dtype(a129_column):
        sector_index = a129_column.to_numpy().astype(int) - 1
        if sector_index.size and sector_index.min() >= 0 and sector_index.max() < S:
            return sector_index
    name_of = {str(c): s for s, c in enumerate(data["sector_names"])}
    mapped = a129_column.astype(str).str.strip().map(name_of)
    if mapped.notna().all():
        return mapped.to_numpy().astype(int)
    unplaced = sorted(set(a129_column[mapped.isna()].astype(str)))[:5]
    raise ValueError(
        f"cannot place suppliers.parquet's A129 values {unplaced} among the model's S = {S} "
        f"sectors {list(data['sector_names'])} — the column should be the model index "
        "1..S, as main.jl writes it.")


def _n_hat_from_diagnostics(data):
    """`N_hat_s` from `granular_diagnostics.npz`, the fallback when the post-hoc run
    predates `post_hoc_N_hat.npy`. It is written only when its own step ran, so it can
    be stale relative to the parquet — which is why the post-hoc file is preferred."""
    for step in (data.get("step_dir", "step3"), "step3", "step1"):
        diagnostics_path = data["folder"] / step / "granular_diagnostics.npz"
        if diagnostics_path.exists():
            with np.load(diagnostics_path, allow_pickle=True) as archive:
                for k in ("N_hat", "N_s", "N_hat_s"):
                    if k in archive.files:
                        return np.asarray(archive[k], dtype=float)
    return None


# ============================================================================
# Sector ordering and buyer weights
# ============================================================================


def _sectors_in_code_order(data):
    """
    Sector display names in A129-CODE order.

    Every sector-indexed table and figure in this section is presented in this order, so
    the reader compares the same row across figures instead of re-reading a legend that
    has been resorted by whatever the figure happens to rank on.
    """
    return [n for _, n in sorted(zip(data["sector_names"], data["sector_names"]))]


def _by_sector_code(df, data, level=None):
    """Reindex a sector-indexed frame into A129-code order, dropping absent sectors."""
    order = [s for s in _sectors_in_code_order(data)
             if s in (df.index if level is None else df.index.get_level_values(level))]
    return df.reindex(order) if level is None else df.reindex(order, level=level)


def _buyer_weights(data):
    """Downstream buyers weighted by their purchases (the empirical pi_r target)."""
    weights = np.asarray(data["emp_pi_r"], dtype=float).ravel()
    return weights / weights.sum()



# ============================================================================
# THE ECONOMY
# ============================================================================

# The EXTENDED PARAMETER SET and the economy it defines.
#
# `best_params` carries the vector the optimiser searched, which is NOT the whole
# parameter of the economy: comparative advantage `T` is the Sinkhorn image of the gamma
# target and the variety count `N_s` is an integer bisection on Gbar_s(0), so both are
# CALIBRATED INSIDE the model and neither is a coordinate of the search. They are
# parameters of the ECONOMY all the same, and writing them down beside the head is what
# makes the Julia and the Python route comparable at all:
#
#   theta+ = (chi_L, chi_s, A, alpha, T, Phi^F, N)   plus the fixed calibration
#            (theta, nu_s, nu, lambda, epsilon, delta_r, w_r).
#
# Two properties of `theta+` are worth stating because they bound what "the same
# parameter" can mean. It is GAUGE-INDETERMINATE — only `T[s,:]/T[s,ref]` is identified,
# so equality can only be asserted after the per-sector reference normalisation. And `N`
# is an INTEGER, so it is never interpolated: a counterfactual holds it fixed (the
# variety count is a technology primitive there) rather than re-running the bisection,
# which would make the exercise a re-estimation instead of a counterfactual.
#
# Given `theta+` and the draws the economy is a DETERMINISTIC FORWARD MAP with no fixed
# point — the reason the whole of it can be rebuilt here rather than only its Ricardian
# half. An upstream cell's delivered cost is `w_l tau_{l,r} / z_{rho,l}` with `w_r`
# normalised to one and NO upstream price index in it, so nothing downstream feeds back:
#
#   winners  ->  P_sr  ->  P_r  ->  c_r, c_tilde_r  ->  P, Y_r  ->  flows.
#
# `simulate_economy` is that map, step for step as `solve_network` runs it, so every
# regime — the estimated economy included — is produced by ONE implementation and the
# counterfactuals are not a different object from the baseline. The estimated economy is
# also the one Julia writes to `suppliers.parquet`, which makes it the reference the
# Python route is gated against rather than merely compared with.

# `NU_ACROSS_DEFAULT`, `LAMBDA_DEFAULT` and `NU_S_DEFAULT` live in the Constants cell
# beside the other calibration; `delta_r = 1` and regional wages collapse to one, which is
# what leaves the upstream cost free of a price index and the forward map acyclic.


class Economy(dict):
    """
    A variety panel that also carries the value block.

    It is a `dict` of `{sector: {winner, v, exp_val, buyers, replication}}` so that
    `variety_concentration`, `_V_by_buyer`, `cosourcing` and every other consumer that
    walks `panel.items()` with `int(s)` reads it UNCHANGED — the prices, `D_r`, the input
    mix and `Y_r` hang off attributes instead of extra keys, which is the same device the
    frames use with `.attrs`. A string key would have broken those loops at `int(s)`.
    """
    __slots__ = ("value", "meta")


def extended_parameters(data, n_hat=None, theta=None, verbose=True):
    """
    `theta+`: the head the optimiser searched, the two blocks the model calibrates, and
    the calibration held fixed — read from the run rather than assumed.

    `theta` is the one number with a real failure mode. It is read from `stats.csv` and
    falls back to `THETA_DEFAULT` only when that file carries none; the Julia run uses
    `load_parameters.jl`'s own value (1.768), so a silent fallback would compute every
    `theta*alpha`-scaled object at a distance elasticity the economy was never solved at.
    The source is therefore RECORDED and announced, and the gate requires it to be the
    run's own.
    """
    best_params = data.get("best_params")
    if best_params is None:
        raise FileNotFoundError(
            f"no best_params in {data['folder']}/{data['step_dir']}/ — "
            "best_parameters_list.npy is written by run_reporting.")
    best_params = np.asarray(best_params, dtype=float).ravel()
    S, n_AA, n_tau = data["S"], data["n_AA"], data["n_tau"]
    R_d = len(data["aa_names"])
    n_T = int(data["T_ACTIVE"].sum())               # areas + Phi^F of importing sectors
    expected = 1 + S + R_d + n_tau + 1 + n_T
    if best_params.size != expected:
        raise ValueError(
            f"best_params has {best_params.size} entries but the layout rebuilt here is "
            f"{expected} = 1 + {S} + {R_d} + {n_tau} + 1 (kappa) + {n_T}.")

    estimate = unpack_estimated_T(data)                 # T, gauge-fixed, plus alpha and theta
    if theta is None:
        # ONE route to theta, `model_theta`, which prefers `load_parameters.jl` — the
        # value the economy on disk was solved at. A second reading here would be a
        # second place for the notebook to drift from the estimator.
        theta = model_theta(data, verbose=verbose)
        julia_theta = _theta_from_julia(data.get("base"))
        theta_source = ("load_parameters.jl" if julia_theta is not None and abs(julia_theta - theta) <= 1e-12
               else ("stats.csv" if data.get("coefs") is not None
                     and _read_named_value(data["coefs"], "theta") is not None
                     else "THETA_DEFAULT"))
    else:
        theta_source = "caller"
    if theta_source == "THETA_DEFAULT" and verbose:
        print(f"  [theta+] WARNING theta fell back to THETA_DEFAULT = {theta:g}: neither "
              "load_parameters.jl nor stats.csv was readable, so every "
              "theta*alpha-scaled object here rests on a notebook constant rather than "
              "on the value the economy was solved at.")

    if n_hat is None:
        n_hat = data.get("post_hoc_N_hat")
        if n_hat is None:
            n_hat = _n_hat_from_diagnostics(data)
    if n_hat is None:
        raise ValueError("no N_hat_s available — the variety count is what sets the size "
                         "of a realised economy, so it is part of theta+.")
    n_hat = np.asarray(n_hat).ravel().astype(int)
    if n_hat.size < S:
        raise ValueError(f"N_hat has {n_hat.size} sectors against S = {S}.")

    theta_plus = {"chi_L": float(best_params[0]),
          "chi_s": best_params[1:1 + S].copy(),
          "A": best_params[1 + S:1 + S + R_d].copy(),
          "alpha": estimate["alpha"].copy(),
          "kappa": estimate["kappa"],
          "T": estimate["T"],                            # (S, n_AA), T[s, ref] = 1
          "T_cell": estimate["T_cell"],                  # (S, R), T_a n_ls^kappa
          "PhiF": estimate["PhiF"],                      # (S,), the foreign competitor, same gauge as T
          "N": n_hat[:S].copy(),
          "theta": float(theta), "theta_source": theta_source,
          "nu_s": np.full(S, float(NU_S_DEFAULT)),
          "nu": float(NU_ACROSS_DEFAULT), "lambda": float(LAMBDA_DEFAULT),
          "epsilon": float(data["epsilon"]),
          "delta": np.ones(R_d), "wage": np.ones(data["R"])}
    if verbose:
        print(f"  [theta+] chi_L {theta_plus['chi_L']:.4f}  alpha {np.round(theta_plus['alpha'], 4).tolist()}  "
              f"theta {theta_plus['theta']:.4f} ({theta_source})  N_s {theta_plus['N'].min()}-{theta_plus['N'].max()}  "
              f"eps {theta_plus['epsilon']:.3f}  nu {theta_plus['nu']:g}/{NU_S_DEFAULT:g}  lam {theta_plus['lambda']:g}")
    return theta_plus


def _good_order(data):
    """
    The `(sector, cell)` order Julia's draw matrix is columned by.

    `good_indices = findall(cell_mask)` on an `(S, R)` matrix walks it COLUMN-MAJOR, so
    `g` runs region-outer, sector-inner. Getting this backwards reads one cell's draws
    for another and produces a perfectly plausible wrong economy — the same class of trap
    as the s-major T convention, so it is derived here rather than assumed.
    """
    cell_mask = np.asarray(data["CELL_MASK"], dtype=bool)          # (S, R)
    good_region, good_sector = np.nonzero(cell_mask.T)                         # C-order on (R, S) == r outer
    return good_sector, good_region                                    # 0-based


def simulate_economy(data, theta_plus=None, *, alpha=None, equalise_T=False, kappa=None, n_rep=None,
                     u_draws=None, u_foreign=None, seed=20260912, geometry=None, n_hat=None,
                     buyers=None, verbose=False):
    """
    The realised finite-variety economy at `theta+`, by the forward map `solve_network`
    runs — the Ricardian block AND the value block, so `D_r`, `theta_rs`, `c_tilde_r` and
    `Y_r` come out of the same object as the winners rather than being read off a
    baseline parquet and held fixed.

    `alpha=0` and `equalise_T=True` are the two counterfactuals, and `N` is held at
    `theta+`'s value in both: the bisection would return a different `N_hat` under a
    different geometry, but re-running it would recalibrate a moment rather than answer a
    counterfactual.

    `n_rep` is how many economies are drawn; `None` takes `ECONOMY_REPLICATIONS` (1000).
    It sets the DISPERSION every band and `z` score rests on, and costs Python time
    alone. `POST_HOC_REPLICATIONS` in `main.jl` is a different quantity: how many
    economies Julia solves and writes for `check_against_julia`.

    `buyers` restricts the returned per-sector panel to those buyer zones (the consumers
    that ride a spend frame need its own buyer set). The VALUE block is deliberately not
    restricted: `P` and `Y_r` are sums over every downstream region, so computing them on
    a subset would answer a different demand system.

    `u_draws` is the uniform draw matrix. Passed (Julia's own, one `(N_max, n_good)` matrix per
    replication), the economy is Julia's to the bit and the comparison is a test rather
    than an agreement. Omitted, draws come from `default_rng(seed)` — independent across
    replications, and SHARED ACROSS BUYERS within one, because a variety is one physical
    good and that shared draw is the whole source of common granularity.

    THE FOREIGN COMPETITOR. A sector with `Phi^F_s > 0` has one foreign bid per variety,
    `p^F = Phi^F_s^(-1/theta) (-log(1 - u^F))^(1/theta)`, common to every buyer (no trade
    cost, no wage), as Julia's `solve_network` draws it. Where it is strictly below the
    best French cell the variety is IMPORTED for that buyer: it enters the price index
    `P_sr` and the buyer's spending, but it is not a French linkage, so its `winner` is 0,
    its `exp_val` and `v` are 0 and it has no row in `economy_frame` (as in the parquet).
    `u_foreign` is Julia's `(N_max, S)` matrix per replication (`post_hoc_u_foreign.npy`);
    omitted, the foreign bids come from their OWN stream, so the domestic draws (and a
    sector without imports) are exactly what they were without the foreign competitor.

    In a counterfactual (`alpha`, `equalise_T`) the foreign scale is buyer-specific,
    `Phi^F_s * Phi^dom_rs(regime)/Phi^dom_rs(estimate)` from `sourcing_geometry`, so every
    buyer's import share stays at its estimate: the regimes move the DOMESTIC network
    only. The foreign draw `u^F` of a variety is still one per sector, shared by buyers.

    The value block is DOMESTIC where the distinction matters, since the reporting reads
    the economy's French flows:
      `D_r`        = 1 + the domestic intermediate share of the buyer's cost, i.e. the
                     DOMESTIC amplification: one euro of final demand for r's output sends
                     `D_r - 1` euros to French upstream cells (the rest of the intermediate
                     bill, `M_r`, leaks to the foreign competitors);
      `theta_rs`   = sector s's share of the DOMESTIC intermediate bill (sums to one);
      `D_total_r`  = 1 + (1-chi_L)(P_r/c_r)^(1-lambda), the TOTAL intermediate share;
      `M_rs`       = imports from sector s per unit of cost, `M_r = sum_s M_rs`;
      `theta_total_rs` = chi_s (P_sr/P_r)^(1-nu), the sector mix of the TOTAL bill.
    Without imports `D_r = D_total_r` and `theta_rs = theta_total_rs` exactly.

    Returns `variety_panel`'s structure per sector (so every existing consumer reads it
    unchanged) plus the value block per (replication, buyer) and the cost shares.
    """
    if theta_plus is None:
        theta_plus = extended_parameters(data, n_hat=n_hat, verbose=verbose)
    active_geometry = (sourcing_geometry(data, alpha=alpha, equalise_T=equalise_T,
                           **({} if kappa is None else {"kappa": kappa}))
         if geometry is None else geometry)
    theta, alpha_vector = theta_plus["theta"], active_geometry["alpha"]                       # a: the N_TAU vector
    nu, lam, chi_L, chi_s = theta_plus["nu"], theta_plus["lambda"], theta_plus["chi_L"], theta_plus["chi_s"]
    S = data["S"]
    down = np.asarray(active_geometry["downstream"]).astype(int)          # 1-based ZE of each buyer
    R_d = down.size
    A = theta_plus["A"]
    if A.size != R_d:
        raise ValueError(f"A has {A.size} entries against {R_d} downstream regions.")
    wage = np.asarray(theta_plus["wage"], dtype=float)

    blocks = active_geometry["by_sector"]
    missing = [s for s in range(S) if s not in blocks or blocks[s]["cells"].size == 0]
    if missing:
        # An empty sector sends its own CES index to 0 and `P_sr^(1/(1-nu_s))` to
        # infinity, i.e. the buyer's price index is not defined. Julia `continue`s past
        # it and inherits the same infinity, so this is refused rather than reproduced.
        raise ValueError(f"sectors {missing} have no cells — the within-sector price "
                         "index is not defined there.")
    N_varieties = np.array([int(theta_plus["N"][s]) for s in range(S)])
    if (N_varieties <= 0).any():
        raise ValueError(f"a sector was given zero varieties: {N_varieties}")

    # `Nv` and the geometry are needed to validate a supplied draw matrix, so the draw
    # setup sits after them.
    PhiF_s = np.asarray(theta_plus.get("PhiF", np.zeros(S)), dtype=float)
    imports = PhiF_s > 0
    # the foreign scale per (sector, buyer): Phi^F_s at the estimate, rescaled in a
    # counterfactual so every buyer's import share stays at its estimate (`sourcing_geometry`)
    PhiF_r = {s: np.asarray(blocks[s].get("PhiF_r", np.full(R_d, PhiF_s[s])), dtype=float)
              for s in range(S) if imports[s] and s in blocks}
    if u_draws is None:
        if n_rep is None:
            n_rep = globals().get("ECONOMY_REPLICATIONS") or 1000
        rng, u_list = np.random.default_rng(seed), None
        # the foreign bids from their own stream: the domestic draws are untouched
        rng_foreign = np.random.default_rng([seed, 1]) if imports.any() else None
    else:
        u_list = [np.asarray(x, dtype=float) for x in (u_draws if isinstance(u_draws, (list, tuple)) else [u_draws])]
        if n_rep is None:
            n_rep = len(u_list)
        if len(u_list) < n_rep:
            raise ValueError(f"{len(u_list)} draw matrices supplied for {n_rep} replications.")
        good_sector, good_region = _good_order(data)
        n_good, N_max = good_sector.size, int(N_varieties.max())
        for k, draw_matrix in enumerate(u_list[:n_rep]):
            # Validated UP FRONT rather than at the slice: a matrix one column short
            # would otherwise fail as an out-of-range index deep in the sector loop, and
            # one column too MANY would not fail at all -- it would silently read the
            # wrong cell's draws, which is the trap `_good_order` exists to avoid.
            if draw_matrix.ndim != 2 or draw_matrix.shape[1] != n_good or draw_matrix.shape[0] < N_max:
                raise ValueError(
                    f"draw matrix {k} has shape {draw_matrix.shape}; Julia writes "
                    f"(N_max, n_good) = (>={N_max}, {n_good}) for this theta+.")
        rng, rng_foreign = None, None
        if imports.any():
            if u_foreign is None:
                raise ValueError("this theta+ has a foreign competitor (Phi^F > 0) but no "
                                 "foreign draws were passed: pass `u_foreign` "
                                 "(post_hoc_u_foreign.npy), or Julia's economy cannot be rebuilt.")
            u_foreign_list = [np.asarray(x, dtype=float) for x in
                       (u_foreign if isinstance(u_foreign, (list, tuple)) else [u_foreign])]
            if len(u_foreign_list) < n_rep or any(draw_matrix.ndim != 2 or draw_matrix.shape[1] != S or draw_matrix.shape[0] < N_max
                                           for draw_matrix in u_foreign_list[:n_rep]):
                raise ValueError(f"foreign draws must be {n_rep} matrices of shape "
                                 f"(>={N_max}, {S}), as post_hoc_u_foreign.npy stacks them.")

    mu_inv = theta_plus["epsilon"] / (theta_plus["epsilon"] - 1.0)           # mu = eps/(eps-1)
    economy, val_rows = Economy(), []
    accumulated = {s: {"W": [], "v": [], "rep": [], "exp": [], "z": []} for s in range(S)}

    for b in range(n_rep):
        P_sr = np.empty((S, R_d))
        price_keep, winner_keep, imported_keep = {}, {}, {}
        for s in range(S):
            sector_block = blocks[s]
            cells, N, nu_within = sector_block["cells"], N_varieties[s], float(theta_plus["nu_s"][s])
            log_T = np.log(np.maximum(sector_block["T_cell"], 1e-300))
            log_tau_lr = sector_block["log_tau"]                                     # (cell, buyer)
            if u_list is None:
                u_sector = rng.random((N, cells.size))
            else:
                # Julia's matrix is (N_max, n_good): take this sector's columns in the
                # good order, then its first N rows — the rows past N_hat carry weight
                # zero there and are clamped, so they are not part of the economy.
                cols = np.flatnonzero(good_sector == s)
                order = np.argsort(good_region[cols])                          # by ZE, as `cells` is
                u_sector = u_list[b][:N, cols[order]]
                if u_sector.shape != (N, cells.size):
                    raise ValueError(
                        f"draw matrix {b} gives {u_sector.shape} for sector {s}, expected "
                        f"{(N, cells.size)}: the good order or N_hat disagrees with Julia.")
            # Frechet inverse CDF, Julia's branch exactly: z = T^(1/theta) * (-log(1-u))^(-1/theta)
            log_z = log_T[None, :] / theta - np.log(-np.log1p(-u_sector)) / theta
            log_cost = log_tau_lr.T[None, :, :] - log_z[:, None, :]                # (var, buyer, cell)
            cheapest_cell = log_cost.argmin(axis=2)
            price = np.exp(np.take_along_axis(log_cost, cheapest_cell[:, :, None], axis=2)[:, :, 0])   # (var, buyer)
            # the winner's own productivity, which is the parquet's `productivity` column
            z_winner = np.exp(np.take_along_axis(log_z[:, None, :].repeat(cheapest_cell.shape[1], 1),
                                           cheapest_cell[:, :, None], axis=2)[:, :, 0])
            winner = cells[cheapest_cell] + 1                                        # 1-based zone
            if imports[s]:
                # the foreign bid, common to every buyer, wins where strictly cheaper
                u_foreign_sector = (rng_foreign.random(N) if u_list is None else u_foreign_list[b][:N, s])
                foreign_price = np.exp((np.log(-np.log1p(-u_foreign_sector))[:, None]
                             - np.log(PhiF_r[s])[None, :]) / theta)   # (var, buyer)
                imported = foreign_price < price                                          # (var, buyer)
                price = np.where(imported, foreign_price, price)
                winner = np.where(imported, 0, winner)
                z_winner = np.where(imported, np.nan, z_winner)
            else:
                imported = np.zeros(price.shape, dtype=bool)
            accumulated[s]["z"].append(z_winner)
            ces_term = price ** (1.0 - nu_within)
            P_sr[s] = (ces_term.mean(axis=0)) ** (1.0 / (1.0 - nu_within))         # weights are 1/N
            price_keep[s], winner_keep[s], imported_keep[s] = price, winner, imported
            # v: the variety's share of the buyer's DOMESTIC spending on s (zero if imported)
            y_dom = np.where(imported, 0.0, ces_term)
            tot_dom = y_dom.sum(axis=0, keepdims=True)
            accumulated[s]["v"].append(np.divide(y_dom, tot_dom, out=np.zeros_like(y_dom),
                                         where=tot_dom > 0))
            accumulated[s]["W"].append(winner_keep[s].astype(float))
            accumulated[s]["rep"].append(np.full(N, b))

        P_r = (chi_s[:, None] * P_sr ** (1.0 - nu)).sum(axis=0) ** (1.0 / (1.0 - nu))
        wage_buyer = wage[down - 1]
        c_r = (chi_L * wage_buyer ** (1.0 - lam) + (1.0 - chi_L) * P_r ** (1.0 - lam)) ** (1.0 / (1.0 - lam))
        c_tilde = c_r / A
        labor_substitution_factor = (1.0 - chi_L) * (P_r / c_r) ** (1.0 - lam)           # TOTAL intermediate share
        M_rs = np.zeros((S, R_d))                                        # imports per unit of cost
        for s in range(S):
            nu_within = float(theta_plus["nu_s"][s])
            e = ((1.0 / N_varieties[s]) * chi_s[s] * (1.0 - chi_L)
                 * (price_keep[s] / P_sr[s][None, :]) ** (1.0 - nu_within)
                 * (P_sr[s][None, :] / P_r[None, :]) ** (1.0 - nu)
                 * (P_r[None, :] / c_r[None, :]) ** (1.0 - lam))
            if imports[s]:
                M_rs[s] = np.where(imported_keep[s], e, 0.0).sum(axis=0)
                e = np.where(imported_keep[s], 0.0, e)                     # no French linkage
            accumulated[s]["exp"].append(e)
        # the downstream demand system: p_r = c_tilde/mu, P = (sum p^eps delta)^(1/eps)
        p_r = c_tilde / mu_inv
        P_agg = (p_r ** theta_plus["epsilon"] * theta_plus["delta"]).sum() ** (1.0 / theta_plus["epsilon"])
        Y_r = p_r ** theta_plus["epsilon"] * P_agg ** (-theta_plus["epsilon"]) * theta_plus["delta"]
        theta_total_rs = chi_s[:, None] * (P_sr / P_r[None, :]) ** (1.0 - nu)
        if imports.any():
            # domestic purchases from s per unit of cost, and their mix
            domestic_purchases = theta_total_rs * labor_substitution_factor[None, :] - M_rs
            # a buyer that imports EVERY variety of every sector in this replication has no
            # domestic bill: its mix is set to zero (and its D_r is exactly 1)
            domestic_total = domestic_purchases.sum(axis=0, keepdims=True)
            theta_rs = np.divide(domestic_purchases, domestic_total, out=np.zeros_like(domestic_purchases),
                                 where=domestic_total > 1e-10 * labor_substitution_factor[None, :])
            D_r = 1.0 + labor_substitution_factor - M_rs.sum(axis=0)
        else:
            theta_rs, D_r = theta_total_rs, 1.0 + labor_substitution_factor
        val_rows.append({"replication": b, "P_sr": P_sr.copy(), "P_r": P_r,
                         "c_r": c_r, "c_tilde_r": c_tilde, "D_r": D_r,
                         "D_total_r": 1.0 + labor_substitution_factor, "M_rs": M_rs,
                         "Y_r": Y_r, "P": float(P_agg), "theta_rs": theta_rs,
                         "theta_total_rs": theta_total_rs})

    all_buyers = down.astype(int)
    if buyers is None:
        keep, kept = slice(None), all_buyers
    else:
        kept = np.asarray(buyers).astype(int)
        buyer_position = {int(b): cheapest_cell for cheapest_cell, b in enumerate(all_buyers)}
        missing_buyers = [int(b) for b in kept if int(b) not in buyer_position]
        if missing_buyers:
            raise KeyError(f"buyers {missing_buyers} are not downstream regions of this economy.")
        keep = np.array([buyer_position[int(b)] for b in kept])
    for s in range(S):
        economy[int(s)] = {"winner": np.vstack(accumulated[s]["W"])[:, keep],
                       "v": np.vstack(accumulated[s]["v"])[:, keep],
                       "exp_val": np.vstack(accumulated[s]["exp"])[:, keep],
                       "z": np.vstack(accumulated[s]["z"])[:, keep],
                       "buyers": kept,
                       "replication": np.concatenate(accumulated[s]["rep"])}
    value_block = {k: np.stack([r[k] for r in val_rows]) for k in
           ("P_sr", "P_r", "c_r", "c_tilde_r", "D_r", "D_total_r", "M_rs", "Y_r",
            "theta_rs", "theta_total_rs")}
    value_block["P"] = np.array([r["P"] for r in val_rows])
    economy.value = value_block
    economy.meta = {"buyers": kept, "value_buyers": all_buyers, "n_rep": n_rep,
                "N": N_varieties, "alpha": alpha_vector, "PhiF": PhiF_s,
                "equalise_T": equalise_T, "kappa": kappa, "theta": theta,
                "draws": "julia" if u_list is not None else f"rng({seed})"}
    return economy


def economy_identities(econ, theta_plus, tol=1e-12, verbose=True):
    """
    The identities the forward map satisfies exactly, as a check on the port rather than
    on the economy: each one fixes a different step, so a mismatch NAMES the step.

    (1) `sum_rho exp_val + M_rs = theta_total_rs (D_total_r - 1)` — the definition of the
        within-sector index `P_sr` (the French and the imported varieties together
        exhaust the buyer's spending on s), and `v` sums to one over the DOMESTIC
        varieties of every (replication, buyer) that buys one.
    (2) `sum_s chi_s (P_sr/P_r)^(1-nu) = 1` — the across-sector index.
    (3) `theta_rs` and `theta_total_rs` sum to one over sectors, so the input mix comes OUT
        of the economy.
    (4) `D_r = 1 + sum_{l,s} exp_val` (the DOMESTIC amplification) and
        `D_total_r = D_r + M_r = 1 + (1-chi_L)(P_r/c_r)^(1-lambda)`, the closed form the
        section quotes — (1) and (2) are what collapse the two CES indices, so this is the
        one number that fixes the whole expenditure chain.
    Without imports `M = 0` and these are the identities of the domestic economy.
    """
    # The panel can be a buyer SUBSET while the value block is not (P and Y_r are sums
    # over every downstream region), so the value columns are cut to the panel's buyers
    # before any per-buyer identity is checked.
    all_buyers = np.asarray(econ.meta.get("value_buyers", econ.meta["buyers"])).astype(int)
    kept = np.asarray(econ.meta["buyers"]).astype(int)
    buyer_columns = np.array([int(np.flatnonzero(all_buyers == b)[0]) for b in kept])
    value_block = {k: (v[..., buyer_columns] if getattr(v, "ndim", 0) >= 2 else v)
           for k, v in econ.value.items()}
    sectors = sorted(econ)
    n_rep = econ.meta["n_rep"]
    gaps = {}
    max_gap_within, max_gap_v, domestic_share_sum = 0.0, 0.0, np.zeros_like(value_block["D_r"])
    intermediate_share = value_block["D_total_r"] - 1.0                                   # (rep, buyer)
    for s in sectors:
        v, e, r = econ[s]["v"], econ[s]["exp_val"], econ[s]["replication"]
        for b in range(n_rep):
            exp_sum = e[r == b].sum(axis=0)
            domestic_share_sum[b] += exp_sum
            max_gap_within = max(max_gap_within, float(np.abs(exp_sum + value_block["M_rs"][b, s] - value_block["theta_total_rs"][b, s] * intermediate_share[b]).max()))
            v_sum = v[r == b].sum(axis=0)
            buys_domestic = (econ[s]["winner"][r == b] > 0).any(axis=0)
            if buys_domestic.any():
                max_gap_v = max(max_gap_v, float(np.abs(v_sum[buys_domestic] - 1.0).max()))
    gaps["ces_within"] = max_gap_within
    gaps["v_sums_one"] = max_gap_v
    gaps["ces_across"] = float(np.abs(
        (theta_plus["chi_s"][None, :, None] * (value_block["P_sr"] / value_block["P_r"][:, None, :]) ** (1.0 - theta_plus["nu"])
         ).sum(axis=1) - 1.0).max())
    # a domestic bill to split (not one lost to rounding when every variety is imported)
    has_domestic_bill = (value_block["D_r"] - 1.0) > 1e-10 * (value_block["D_total_r"] - 1.0)
    gaps["mix_sums_one"] = float(max(np.abs(value_block["theta_rs"].sum(axis=1) - 1.0)[has_domestic_bill].max(initial=0.0),
                                    np.abs(value_block["theta_total_rs"].sum(axis=1) - 1.0).max()))
    gaps["D_r_from_shares"] = float(np.abs(value_block["D_r"] - (1.0 + domestic_share_sum)).max())
    gaps["D_total_split"] = float(np.abs(value_block["D_total_r"] - value_block["D_r"] - value_block["M_rs"].sum(axis=1)).max())
    ok = all(v <= tol for v in gaps.values())
    if verbose:
        print("  [identities] " + "  ".join(f"{k} {v:.2e}" for k, v in gaps.items())
              + ("  -> ok" if ok else "  -> FAIL"))
    if not ok:
        raise AssertionError(f"the forward map does not close: {gaps}")
    return gaps



def check_against_julia(data, theta_plus=None, econ=None, tol=1e-10, verbose=True):
    """
    The cross-language gate: rebuild the ESTIMATED economy here from `theta+` and Julia's
    own draws, then compare it with `suppliers.parquet` row by row.

    This is the one comparison that can be exact. Every other check between the two
    implementations is statistical and cannot separate a coding error from a convention
    mismatch — the Frechet branch, the column order of the draw matrix, the
    within-sector normalisation of the expenditure share. Handed the same `theta+` and
    the same `u_draws`, the forward map is deterministic, so the two must agree to the last
    few bits and anything larger is a defect with an address.

    With a foreign competitor the economy is rebuilt with Julia's foreign draws too
    (`post_hoc_u_foreign.npy`), and the number of French linkages per sector is compared
    as well (`missing_rows`): an imported (variety, buyer) is a row in neither.

    Three things are checked, in the order a failure should be read. The draw COLUMN map
    first (`post_hoc_good_sr.npy` against `_good_order`): wrong there, every number below
    is wrong in a way that still looks like an economy. Then the WINNERS, which depend on
    `(alpha, T, N, theta)` and the draws alone — a mismatch localises to the Ricardian
    block. Then `exp_val`, which additionally depends on the head and the whole price
    chain, so a mismatch there with matching winners localises to the value block.
    """
    u_draws = data.get("post_hoc_u")
    if u_draws is None:
        raise FileNotFoundError(
            f"no post_hoc_u.npy in {data['folder']}/{data['step_dir']}/ — re-run the "
            "post-hoc block of main.jl, which writes the draws beside the economy. "
            "Without them the two implementations can only be compared statistically.")
    suppliers = data.get("suppliers")
    if suppliers is None or "replication" not in getattr(suppliers, "columns", []):
        raise FileNotFoundError("no replicated suppliers.parquet to compare against.")
    u_draws = np.asarray(u_draws, dtype=float)
    if u_draws.ndim != 3:
        raise ValueError(f"post_hoc_u.npy has shape {u_draws.shape}, expected (N_max, n_good, n_rep).")

    # (1) the column map, read from the file rather than trusted
    good_sector, good_region = _good_order(data)
    julia_good_sr = data.get("post_hoc_good_sr")
    if julia_good_sr is not None:
        julia_good_sr = np.asarray(julia_good_sr, dtype=int)
        if julia_good_sr.shape != (good_sector.size, 2):
            raise ValueError(f"post_hoc_good_sr.npy is {julia_good_sr.shape}, expected {(good_sector.size, 2)}.")
        n_mismatched = int((np.abs(julia_good_sr[:, 0] - (good_sector + 1)) + np.abs(julia_good_sr[:, 1] - (good_region + 1))).sum())
        if n_mismatched:
            raise AssertionError(
                f"the draw column map disagrees with Julia on {n_mismatched} entries — "
                "`_good_order` and `findall(cell_mask)` are not walking the mask the "
                "same way, and every comparison below would be meaningless.")

    if theta_plus is None:
        theta_plus = extended_parameters(data, verbose=verbose)
    n_rep = u_draws.shape[2]
    if econ is None:
        u_foreign = data.get("post_hoc_u_foreign")
        u_foreign = None if u_foreign is None else [np.asarray(u_foreign, dtype=float)[:, :, b] for b in range(n_rep)]
        econ = simulate_economy(data, theta_plus, u_draws=[u_draws[:, :, b] for b in range(n_rep)],
                                u_foreign=u_foreign, n_rep=n_rep, verbose=verbose)

    sector = _parquet_sector_index(data, suppliers)
    replication = suppliers["replication"].to_numpy().astype(int)
    replication0 = replication - replication.min()                     # Julia writes 1..n_rep
    variety = suppliers["variety"].to_numpy().astype(int) - 1
    buyer_zone = suppliers["ze2010_downstream"].to_numpy().astype(int)
    winner_zone = suppliers["ze2010"].to_numpy().astype(int)
    share = suppliers["share"].to_numpy(dtype=float)
    position_of_buyer = {int(b): buyer_position for buyer_position, b in enumerate(np.asarray(econ.meta["buyers"]).astype(int))}

    rows = []
    for s in sorted(econ):
        N = int(econ.meta["N"][s])
        row_mask = (sector == s) & (replication0 < n_rep) & (variety < N)
        if not row_mask.any():
            rows.append({"sector": s, "rows": 0, "missing_rows": np.nan, "winner_mismatch": np.nan,
                         "max_abs_exp": np.nan, "max_rel_exp": np.nan})
            continue
        buyer_position = np.array([position_of_buyer[b] for b in buyer_zone[row_mask]])
        r = replication0[row_mask] * N + variety[row_mask]
        winner_python = econ[s]["winner"].astype(int)[r, buyer_position]
        exp_val_python = econ[s]["exp_val"][r, buyer_position]
        abs_gap = np.abs(exp_val_python - share[row_mask])
        # the linkages Python finds and the parquet lacks: an imported (variety, buyer)
        # has no row in either, so the two counts must agree
        winners_all = econ[s]["winner"][:n_rep * N]
        rows.append({"sector": s, "rows": int(row_mask.sum()),
                     "missing_rows": int((winners_all > 0).sum()) - int(row_mask.sum()),
                     "winner_mismatch": int((winner_python != winner_zone[row_mask]).sum()),
                     "max_abs_exp": float(abs_gap.max()),
                     "max_rel_exp": float((abs_gap / np.maximum(np.abs(share[row_mask]), 1e-300)).max())})
    comparison = pd.DataFrame(rows).set_index("sector")
    worst_w = int(np.nansum(comparison["winner_mismatch"])) + int(np.nansum(np.abs(comparison["missing_rows"])))
    worst_e = float(np.nanmax(comparison["max_rel_exp"]))
    comparison.attrs["agrees"] = bool(worst_w == 0 and worst_e <= tol)
    if verbose:
        print(f"  [julia] {int(comparison['rows'].sum())} linkages compared over {n_rep} "
              f"replications: {worst_w} winner mismatches, max relative gap on exp_val "
              f"{worst_e:.2e}" + ("  -> the two implementations are the same economy"
                                  if comparison.attrs["agrees"] else
                                  "  -> THEY DISAGREE; read the winners before the shares"))
    return comparison


def economy_frame(econ, theta_plus, drop_zero=False):
    """
    The economy as a frame in `suppliers.parquet`'s own schema.

    This is the step that makes the design pay. Every downstream object in the notebook —
    `build_diffusion_frame` and `D_r`, `_sector_spend` and the input mix, `variety_panel`,
    `io_downstream_column` — reads that schema, so emitting it turns the whole reporting
    stack REGIME-AGNOSTIC: a counterfactual is fed through the same code as the estimated
    economy rather than through a parallel path that has to be kept in step with it.

    `SIREN` is keyed on `(replication, cell, sector, variety)`, as Julia keys it, so two
    replications are two economies and not the same firms observed twice — anything that
    groups by firm must treat them as distinct.

    An IMPORTED (variety, buyer) has no French supplier and therefore no row, as in the
    parquet. Julia drops rows whose `share` is exactly zero (a variety past `N_hat` carries weight
    zero there and would otherwise read as a supplier that supplies nothing). Here every
    row carries a positive share by construction, since the economy is drawn at `N_hat`
    exactly, so `drop_zero` is off by default and is a convenience for comparing against
    a parquet rather than a correction.
    """
    rows = []
    mu = theta_plus["epsilon"] / (theta_plus["epsilon"] - 1.0)
    all_buyers = np.asarray(econ.meta.get("value_buyers", econ.meta["buyers"])).astype(int)
    kept = np.asarray(econ.meta["buyers"]).astype(int)
    buyer_columns = np.array([int(np.flatnonzero(all_buyers == b)[0]) for b in kept])
    Y_r_buyers = econ.value["Y_r"][:, buyer_columns]                 # (n_rep, n_buyer)
    # Julia's `intermediate_derivative` divides by the TOTAL intermediate share (the
    # labour-substitution factor), imports included
    total_share = econ.value.get("D_total_r", econ.value["D_r"])[:, buyer_columns] - 1.0
    for s in sorted(econ):
        sector_block = econ[s]
        N = int(econ.meta["N"][s])
        replication = np.asarray(sector_block["replication"]).astype(int)
        winner = np.asarray(sector_block["winner"]).astype(int)
        exp_val = np.asarray(sector_block["exp_val"], dtype=float)
        productivity = np.asarray(sector_block.get("z", np.full(exp_val.shape, np.nan)), dtype=float)
        variety = np.tile(np.arange(N), exp_val.shape[0] // N)
        n_buyers = kept.size
        rows.append(pd.DataFrame({
            "A129": s + 1,
            "ze2010": winner.ravel(order="C"),
            "ze2010_downstream": np.tile(kept, exp_val.shape[0]),
            "share": exp_val.ravel(order="C"),
            "downstream_purchase": (Y_r_buyers[replication] * mu).ravel(order="C"),
            "intermediate_derivative": (exp_val / total_share[replication]).ravel(order="C"),
            "productivity": productivity.ravel(order="C"),
            "sample_weight": 1.0 / N,
            "variety": np.repeat(variety + 1, n_buyers),
            "replication": np.repeat(replication + 1, n_buyers)}))
    frame = pd.concat(rows, ignore_index=True)
    frame = frame[frame["ze2010"] > 0].reset_index(drop=True)     # imported: no French linkage, no row
    if drop_zero:
        frame = frame[frame["share"] != 0.0].reset_index(drop=True)
    firm_key = frame[["replication", "ze2010", "A129", "variety"]].astype(int)
    frame["SIREN"] = pd.factorize(pd.MultiIndex.from_frame(firm_key))[0] + 1
    return frame[["SIREN", "A129", "ze2010", "ze2010_downstream", "share",
                "downstream_purchase", "intermediate_derivative", "productivity",
                "sample_weight", "variety", "replication"]]


def continuum_economy(data, theta_plus=None, n_var=200, n_rep=8, seed=20260913, **kw):
    """
    The `N_s -> infinity` limit of the same economy: every sector given `n_var` varieties
    instead of its calibrated `N_hat_s`.

    This replaces `suppliers_continuum.parquet`, which Julia wrote by solving the economy
    on the estimation draws (`N_rho` per sector) with flat weights -- i.e. the Monte-Carlo
    evaluation of the same limit. The ESTIMATOR is therefore unchanged; what changes is
    that it no longer costs a Julia re-run.

    Two things to hold in mind. `n_var` is a numerical knob, not a parameter: the limit is
    approached at rate 1/n_var, so read the gap against a doubled `n_var` before quoting
    it. And the cost is O(n_rep * n_var * cells * buyers) per sector, so `n_rep` defaults
    low -- the continuum quantities carry far less sampling noise than the granular ones
    they are the benchmark for, since granularity is exactly what averaging out.
    """
    if theta_plus is None:
        theta_plus = extended_parameters(data, verbose=False)
    S = data["S"]
    return simulate_economy(data, theta_plus, n_rep=n_rep, seed=seed,
                            n_hat=np.full(S, int(n_var)), **kw)


def continuum_data(data, theta_plus=None, n_var=200, n_rep=8, seed=20260913, **kw):
    """`simulated_data` for the infinite-variety limit -- the benchmark frame."""
    if theta_plus is None:
        theta_plus = extended_parameters(data, verbose=False)
    econ = continuum_economy(data, theta_plus, n_var=n_var, n_rep=n_rep, seed=seed, **kw)
    return simulated_data(data, econ, theta_plus)


def simulated_data(data, econ, theta_plus):
    """
    A shallow copy of `data` whose `suppliers` frame IS this economy.

    Everything that reads the parquet then reads the regime instead, with no argument
    threaded through it — which is what lets a counterfactual be measured by the SAME
    functions as the estimated economy rather than by re-derivations of them.
    """
    return {**data, "suppliers": economy_frame(econ, theta_plus),
            "post_hoc_N_hat": np.asarray(econ.meta["N"]).astype(float),
            "suppliers_path": f"<simulated: {econ.meta}>"}


_REPORTING_CACHE = {}


def reporting_data(industry, mu=2, parts=("core", "geography"), n_rep=None,
                   seed=20260912, regimes=None, refresh=False, verbose=True,
                   base="..", **kw):
    """
    Load a run and simulate its economies: the ONE entry point a notebook run cell needs.

    Returns `{regime: data_like}` -- for each regime in `CF_REGIMES`, a copy of `data`
    whose `suppliers` frame IS that regime's economy, plus `economy` (the `Economy`
    object, which carries the value block) and `theta_plus` (the extended parameter set it was
    built from). Every downstream function therefore reads a counterfactual through
    exactly the code it reads the estimate through, and `suppliers.parquet` is not opened
    at all -- `parts` defaults to what the tests notebook needs and excludes `firm`.

    The result is CACHED on the arguments. A notebook's sections each reload, and
    re-solving three economies per section would dominate the run; `refresh=True` drops
    the entry, which is what to do after editing `simulate_economy`.

    To compare against Julia's own realisation instead, load with `parts` including
    `"firm"` and call `check_against_julia`. That is the one thing the parquet is for.
    """
    # Resolve `n_rep` BEFORE the cache key. Otherwise `reporting_data(ind, mu)` and
    # `reporting_data(ind, mu, n_rep=ECONOMY_REPLICATIONS)` are two entries, so the
    # economy cell and the sections would each pay for a solve AND could disagree on how
    # heavily the regimes were drawn -- silently, since both are legitimate economies.
    if n_rep is None:
        n_rep = ECONOMY_REPLICATIONS
    key = (industry, mu, tuple(sorted(resolve_parts(parts))), n_rep, seed, base,
           tuple(sorted(kw.items())),
           None if regimes is None else tuple(sorted(regimes)))
    if refresh:
        _REPORTING_CACHE.pop(key, None)
    if key not in _REPORTING_CACHE:
        data = load_granular_data(industry, mu=mu, parts=parts, base=base, **kw)
        theta_plus = extended_parameters(data, verbose=verbose)
        regime_economies = economy_by_regime(data, theta_plus, regimes=regimes, n_rep=n_rep, seed=seed,
                                 verbose=verbose)
        _REPORTING_CACHE[key] = {regime: {**data_like, "economy": e, "theta_plus": theta_plus}
                                 for regime, (e, data_like) in regime_economies.items()}
    return _REPORTING_CACHE[key]


def economy_by_regime(data, theta_plus=None, regimes=None, n_rep=None, seed=20260912,
                      baseline="Both forces", verbose=True):
    """
    Read `theta+` off the run and simulate EVERY economy from it — the estimated one and
    the counterfactuals alike, by one map, with `N` held fixed throughout.

    This is the arrangement the post-hoc block is no longer needed for. `theta+` follows
    `mu` (it is read from that step's `best_params`), so the estimated economy here is the
    one the selected estimate implies rather than whichever θ̂ the post-hoc block last ran
    at — a mismatch the two-route arrangement could produce silently.

    What is GIVEN UP is the only exact check on the port, and it is worth stating plainly.
    `simulate_economy` is a hand transcription of `solve_network`; its identities are
    gated, but identities cannot catch a transcription that is self-consistently wrong.
    `suppliers.parquet` plus `post_hoc_u.npy` are what `check_against_julia` compares
    against to the bit, so the recommendation is to keep the post-hoc block as a
    VERIFICATION artefact — run it after any change to either implementation, never in the
    reporting loop — rather than to stop writing it. When the files are present this
    function runs that comparison first and says so.

    Returns `{regime: (economy, data_like)}` where `data_like` is `data` with the
    regime's own frame in place of the parquet.
    """
    if theta_plus is None:
        theta_plus = extended_parameters(data, verbose=verbose)
    regimes_used = CF_REGIMES if regimes is None else regimes
    if baseline not in regimes_used:
        raise KeyError(f"baseline {baseline!r} not among {list(regimes_used)}.")

    if data.get("post_hoc_u") is not None and data.get("suppliers") is not None:
        try:
            agree = check_against_julia(data, theta_plus, verbose=verbose)
            if not agree.attrs["agrees"] and verbose:
                print("  [julia] the port and solve_network DISAGREE — fix that before "
                      "reading anything below.")
        except (FileNotFoundError, ValueError, AssertionError) as e:
            if verbose:
                print(f"  [julia] not compared: {type(e).__name__}: {e}")
    elif verbose:
        print("  [julia] no post_hoc_u.npy / suppliers.parquet in this tree, so the port "
              "is unverified against solve_network. Run main.jl's post-hoc block once to "
              "get that check; it is not needed for anything below.")

    economies = {}
    for regime, kw in regimes_used.items():
        e = simulate_economy(data, theta_plus, n_rep=n_rep, seed=seed, verbose=False, **kw)
        economy_identities(e, theta_plus, verbose=False)
        economies[regime] = (e, simulated_data(data, e, theta_plus))
        if verbose:
            value_block = e.value
            print(f"  {regime:<28s} D_r {value_block['D_r'].mean():.4f}  P_r {value_block['P_r'].mean():.4f}  "
                  f"n_rep {e.meta['n_rep']}  draws {e.meta['draws']}")
    return economies
