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
document_width_pt = 511.0
plt.rc("font", family="serif")
toulouse_color = (132 / 255, 46 / 255, 27 / 255)
reference_color = toulouse_color#(0.75, 0.30, 0.20)
sim_color = (0.247, 0.404, 0.667)
font_size = 15
plt.rcParams.update({
    "font.size": font_size,
    "axes.labelsize": font_size,
    "axes.titlesize": font_size + 2,
    "xtick.labelsize": font_size - 3,
    "ytick.labelsize": font_size - 3,
    "legend.fontsize": font_size - 3,
    "legend.title_fontsize": font_size,
    "figure.titlesize": font_size,
})


def get_figsize(document_width_pt=document_width_pt, wf=1.0, hf=0.5):
    """Figure size in inches from a LaTeX \\showthe\\columnwidth width in points."""
    fig_width = document_width_pt * wf / 72.27
    return [fig_width, fig_width * hf]


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

# Highest K kept anywhere in the count-curve reporting: the panels, the empirical
# increments read out of G_K.csv, and the bootstrap variances read out of G_K_var.csv
# are ALL truncated here. Raise it to look further into the tail, but only as far as
# the data pipeline actually tabulates — G_K_var.csv in particular is usually written
# for a few low K only, and a K beyond it simply leaves that panel's empirical band
# empty rather than failing.
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
              "Distance only": dict(equalise_T=True),
              "Comparative advantage only": dict(alpha=0.0)}

# The realised economy is drawn beside the regimes wherever both appear, so it needs a
# colour of its own: sharing "Both forces"' would make the granularity gap invisible.
CF_COLORS = {"Realised": (0.32, 0.32, 0.32),
             "Both forces": toulouse_color,
             "Distance only": sim_color,
             "Comparative advantage only": (0.45, 0.60, 0.45),
             "Neither": (0.72, 0.72, 0.72)}    # kept for a caller that asks for it

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
                     granular=True, relax_n_lo=False, optimizer="pso"):
    """Reproduce main.jl's output_folder naming exactly."""
    name = f"reporting_{industry}"
    name += "_profiled" if profile_T else ""
    name += "_aa" if ca_level == "aa" else ""
    name += "_gran" if granular else ""
    name += "_nlo1" if (relax_n_lo and granular) else ""
    name += f"_{optimizer}"
    return Path(base) / name


def _read_named_value(coefs, name):
    """stats.csv scalar by name: a column called `name`, else a labelled row."""
    if name in coefs.columns:
        vals = coefs[name].dropna().values
        if len(vals):
            return float(vals[0])
    if "value" in coefs.columns:
        for col in coefs.columns:
            hit = coefs.index[coefs[col].astype(str).str.strip().str.lower() == name.lower()]
            if len(hit):
                return float(coefs.loc[hit[0], "value"])
    return None


def _gk_col(G_K, cands):
    """Column of G_K.csv by case-insensitive name, mirroring load_parameters.jl's _gk_col."""
    hit = next((c for c in G_K.columns if c.strip().lower() in cands), None)
    if hit is None:
        raise KeyError(f"G_K.csv is missing a column among {cands}; has {list(G_K.columns)}")
    return hit


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
    flag = next((c for c in ("downstream", "downstream_region") if c in X_dr.columns), None)
    if flag is None:
        raise KeyError("X_dr.csv has neither a `downstream` nor a `downstream_region` column")
    sub = X_dr[X_dr[flag].astype(bool)]
    col = next((c for c in ("ze2010_downstream", "ze2010") if c in sub.columns), None)
    if col is None:
        raise KeyError("X_dr.csv has neither `ze2010_downstream` nor `ze2010`")
    names = _as_ze_string(sub[col]).tolist()
    if len(names) != n_AA:
        raise ValueError(f"{len(names)} downstream rows in X_dr.csv vs n_AA = {n_AA} "
                         "attraction areas in attraction_area_linkages.npy")
    return names


def _region_labels(data):
    """
    Model region index (1..R) -> ZE code -> display NAME.

    The model orders regions by sorted `ze2010`, exactly as `load_parameters.jl` builds
    `X_rs`, so the l-th row of the distance matrix and the integer codes in
    `suppliers.parquet` are the l-th entry of that sorted list. `france` covers more
    commuting zones than the model does, so the join is on the CODE, and a zone with no
    name falls back to its code rather than to NaN.
    """
    fdf = data.get("filter_N_upstream_df")
    if fdf is None:
        codes = [str(i) for i in range(1, data["R"] + 1)]
    else:
        codes = sorted(_as_ze_string(fdf["ze2010"]).unique())
    if len(codes) != data["R"]:
        raise ValueError(f"{len(codes)} ZE codes in filter_N_upstream.csv but R = "
                         f"{data['R']} regions in the model.")
    out = pd.DataFrame({"index": np.arange(1, data["R"] + 1), "ze2010": codes})
    fr = data.get("france")
    if fr is not None and "ze2010_name" in fr.columns:
        names = fr[["ze2010", "ze2010_name"]].copy()
        names["ze2010"] = _as_ze_string(names["ze2010"])
        out = out.merge(names.drop_duplicates("ze2010"), on="ze2010", how="left")
    else:
        out["ze2010_name"] = out["ze2010"]
    out["ze2010_name"] = out["ze2010_name"].fillna(out["ze2010"])
    return out


def ze_name_of_code(data):
    """ZE code -> display name, for anything keyed on the code rather than the index."""
    lab = _region_labels(data)
    return dict(zip(lab["ze2010"], lab["ze2010_name"]))


def aa_display_names(data):
    """
    Attraction-area display names, in AA column order.

    Area `a` is anchored on the a-th downstream region, so its name is that commuting
    zone's name — which is what a reader recognises, unlike the bare ZE code.
    """
    m = ze_name_of_code(data)
    return [m.get(str(a), str(a)) for a in data["aa_names"]]


PARTS_ALL = ("counts", "moments", "inference", "jacobian", "geography", "firm")

# Requesting a part pulls in what it is built from. `moments` needs the count target to
# complete the empirical vector; `inference` and `jacobian` are read against the moment
# layout, so both imply it.
PART_IMPLIES = {"moments": ("counts",),
                "inference": ("moments", "counts"),
                "jacobian": ("moments", "counts")}


def resolve_parts(parts):
    """Expand a requested part set through PART_IMPLIES, and refuse an unknown name."""
    if parts is None:
        return set(PARTS_ALL)
    want = {parts} if isinstance(parts, str) else set(parts)
    bad = want - set(PARTS_ALL) - {"core", "all"}
    if bad:
        raise ValueError(f"unknown part(s) {sorted(bad)}; known parts are "
                         f"{list(PARTS_ALL)} (plus 'core' and 'all')")
    if "all" in want:
        return set(PARTS_ALL)
    want.discard("core")
    for _ in range(len(PART_IMPLIES) + 1):
        grown = set(want)
        for p in list(want):
            grown |= set(PART_IMPLIES.get(p, ()))
        if grown == want:
            break
        want = grown
    return want


def load_granular_data(industry, mu=2, base="..", profile_T=True, ca_level="aa",
                       granular=True, relax_n_lo=False, optimizer="pso", K=-1,
                       k_max=None, parts=None):
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
                              counts     G_K.csv / G_K_var.csv -> Gbar_s(0) and the curve
                              moments    best_simulated_moments.npy and the block dicts
                              inference  Sigma_data / Sigma_sim / Omega / W_step3 and the
                                         fitted-moment standard errors
                              jacobian   jacobian_all*.npy and its axes
                              geography  distances, france.gpkg, regional wages
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
    `agg_industry_share`, `emp_pi_r`, `reg_coef`, `emp_gamma_ls`, `X_dr`, `X_rs`,
    `filter_N_upstream_df`, `distances`, `france`, `ref`, `idf_ze`, `N_downstream`,
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
                              granular=granular, relax_n_lo=relax_n_lo, optimizer=optimizer) # Given the argument, construct the folder name. 
    if not folder.exists():
        raise FileNotFoundError(f"run folder {folder} not found — check profile_T / ca_level / "
                                "granular / relax_n_lo / optimizer against main.jl's naming")

    n_coef = int(np.load(folder / "n_reg_coef.npy")) # Number of regression coefficients. 
    n_tau_path = folder / "n_tau.npy" 
    n_tau = int(np.load(n_tau_path)) if n_tau_path.exists() else n_coef  # Number of coefficients parametrizing the trade costs.  

    # ---------------------------------------------------------------- targets
    coefs = pd.read_csv(input_folder / "stats.csv")
    epsilon = coefs.loc[0, "value"]
    agg_labor_share = coefs.loc[1, "value"]

    agg_industry_share = np.load(input_folder / "input_share.npy").ravel()
    emp_gamma_ls = np.load(input_folder / "emp_gamma_ls.npy")          # (S, R)
    filter_N_upstream = np.load(input_folder / "filter_N_upstream.npy")  # (S, R), binary
    X_rs = np.load(input_folder / "X_rs.npy")                          # (S, R)
    domestic_share = np.load(input_folder / "domestic_share.npy").ravel()
    N_downstream = np.load(input_folder / "N_downstream_per_region.npy").ravel()
    S, R = filter_N_upstream.shape

    X_dr = pd.read_csv(input_folder / "X_dr.csv")
    if "ze2010" in X_dr.columns:
        X_dr["ze2010"] = _as_ze_string(X_dr["ze2010"])
    emp_pi_r = X_dr["X_dr"].values[N_downstream != 0].astype(float)
    emp_pi_r = emp_pi_r / emp_pi_r.sum()

    if n_coef == 1:
        reg_coef = np.array([_read_named_value(coefs, "reg_coef_cloglog_1")], dtype=float)
    else:
        reg_coef = np.load(input_folder / f"reg_coef_cloglog_{n_coef}.npy").ravel()

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
    supplier_cells = (filter_N_upstream == 1) & (X_rs > 0)
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
        act = np.flatnonzero(AA_ACTIVE[s])
        if act.size:
            T_REF_AA[s] = act[np.argmax(emp_gamma_aa[s, act])]

    # ------------------------------------------------------------- count moment
    G_target = np.full(S, np.nan)
    N_supplier_s = np.full(S, np.nan)
    sector_codes = None
    G_curve, G_curve_K, G_pmf = None, None, None
    G_pmf_se = None
    gk_path = input_folder / "G_K.csv"
    if granular and "counts" in parts:
        # G_K.csv is `group, A129, G, K, N_supplier_s`: `G` is the value column,
        # G = Pr(K_ls <= K), and the K = 0 row is the targeted moment Gbar_s(0).
        # Names are matched case-insensitively, as load_parameters.jl does.
        #
        # The WHOLE curve is kept, not only the targeted row: G_s(K) for K >= 1 is a
        # genuinely untargeted check on the count distribution (gate V8 of
        # documentation/granular_validation.md), and load_parameters.jl reads the K = 0
        # row alone, so nothing on the Julia side carries it.
        G_K = pd.read_csv(gk_path)
        c_sector = _gk_col(G_K, ("a129", "sector"))
        c_K = _gk_col(G_K, ("k",))
        c_G = _gk_col(G_K, ("g",))
        c_N = _gk_col(G_K, ("n_supplier_s", "n_supplier", "n_suppliers"))
        sector_codes = sorted(G_K[c_sector].unique())
        if len(sector_codes) != S:
            raise ValueError(f"G_K.csv covers {len(sector_codes)} sectors, expected S = {S}")
        G_curve_K = sorted({int(k) for k in G_K[c_K].astype(float).round()})
        G_curve = np.full((S, len(G_curve_K)), np.nan)
        for s, code in enumerate(sector_codes):
            sub = G_K[G_K[c_sector] == code]
            k_sub = sub[c_K].astype(float).round().astype(int).values
            g_sub = sub[c_G].astype(float).values
            for kk, gg in zip(k_sub, g_sub):
                G_curve[s, G_curve_K.index(kk)] = gg
            if 0 not in k_sub:
                raise ValueError(f"G_K.csv has no K=0 row for A129={code}")
            G_target[s] = G_curve[s, G_curve_K.index(0)]
            N_supplier_s[s] = float(pd.unique(sub[c_N])[0])

        # The INCREMENTS, p_s(K) = Pr[K_ls = K] = G(K) - G(K-1), which is what the
        # reporting plots. G_K.csv stores the CDF, but N_hat_s is calibrated on the
        # K = 0 level, so under a cumulative convention every panel would contain the
        # fitted level and re-test what is already targeted. The increments isolate the
        # untargeted content -- the SHAPE of the count distribution. p_s(0) = G_s(0), so
        # the targeted panel is unchanged.
        # Truncate at k_max BEFORE differencing (the increment at K needs only K and
        # K-1, so dropping the tail changes nothing that survives). G_target was already
        # taken from the K = 0 column above and is unaffected.
        keep = [i for i, k in enumerate(G_curve_K) if k <= k_max]
        G_curve_K = [G_curve_K[i] for i in keep]
        G_curve = G_curve[:, keep]

        # A (sector, K) row absent from G_K.csv leaves a NaN in the CDF, and differencing
        # then poisons TWO increments -- K and K+1 -- so a single missing row silently
        # erases two empirical points. But G is CUMULATIVE: where no row was written, no
        # mass was added, so the CDF is simply flat there. Carry the last value forward
        # (K = 0 is required present, asserted above) and say which cells were filled,
        # so a gap is visible rather than turning into a hole in the figure.
        filled = []
        for s_i in range(S):
            for j in range(1, len(G_curve_K)):
                if not np.isfinite(G_curve[s_i, j]) and np.isfinite(G_curve[s_i, j - 1]):
                    G_curve[s_i, j] = G_curve[s_i, j - 1]
                    filled.append((sector_codes[s_i], G_curve_K[j]))
        if filled:
            print(f"[load] G_K.csv has no row for {filled}; the CDF is flat there (no mass "
                  "added), so those increments are read as 0 rather than dropped.")
        # Requires consecutive K from 0; a gap would make the difference meaningless.
        if G_curve_K == list(range(len(G_curve_K))):
            G_pmf = np.diff(G_curve, axis=1, prepend=0.0)
        else:
            G_pmf = None
            print(f"[load] G_K.csv has non-consecutive K = {G_curve_K}; the increments "
                  "p_s(K) = G(K) - G(K-1) are not well defined and are left empty.")

    # ------------------- bootstrap variance of the empirical increments (optional)
    # `G_K_var.csv` is `A129, K, var` (or `se`) and carries the BOOTSTRAP variance of
    # p_s(K) = share of cells with EXACTLY K suppliers -- the same object G_pmf holds.
    # It is optional: absent, the empirical panels are drawn without a band, exactly as
    # before. Sigma_data's count block covers K = 0 only, which is why the rest needs a
    # file at all; the K = 0 column is therefore a free CONSISTENCY CHECK against it and
    # is compared below rather than trusted.
    gkv_path = input_folder / "G_K_var.csv"
    if granular and "counts" in parts and gkv_path.exists() and sector_codes is not None:
        gkv = pd.read_csv(gkv_path)
        v_sector = _gk_col(gkv, ("a129", "sector"))
        v_K = _gk_col(gkv, ("k",))
        # The value column, by name where the name says what it is, and otherwise by
        # elimination: a file written off G_K.csv's own schema often keeps the column
        # called `G`, which names the object measured, not the statistic. Whatever is
        # left once A129 and K are removed is taken as a VARIANCE (the file's stated
        # content), and the choice is printed rather than assumed silently -- a variance
        # read as an SE would be off by orders of magnitude, which is exactly what the
        # K = 0 check against diag(Sigma_data) below is there to catch.
        _named = {c: c.strip().lower() for c in gkv.columns}
        v_col = next((c for c, n in _named.items()
                      if n in ("var", "variance", "v", "var_g", "g_var")), None)
        is_var = v_col is not None
        if v_col is None:
            v_col = next((c for c, n in _named.items()
                          if n in ("se", "sd", "std", "stderr", "se_g")), None)
        if v_col is None:
            spare = [c for c in gkv.columns if c not in (v_sector, v_K)]
            if len(spare) != 1:
                raise KeyError(
                    f"{gkv_path.name}: cannot tell which column holds the variance. "
                    f"Columns are {list(gkv.columns)}; expected one named var/variance "
                    "(or se/sd/std) beside A129 and K, or exactly one other column.")
            v_col, is_var = spare[0], True
            print(f"[load] {gkv_path.name}: reading column '{v_col}' as the VARIANCE of "
                  "the increment p_s(K) (no var/se column name to go on). Rename it to "
                  "`var` or `se` to be explicit.")
        # Sectors are matched BY VALUE on the A129 code, so the two files must agree on
        # its dtype: an int64 A129 here against a string one in G_K.csv (a leading zero,
        # a quoted field) matches nothing and would hand back an all-NaN band with no
        # complaint. Compare on the stripped string form, and say which codes missed.
        gkv_key = gkv[v_sector].astype(str).str.strip()
        G_pmf_se = np.full((S, len(G_curve_K)), np.nan)
        unmatched = []
        for s_i, code in enumerate(sector_codes):
            sub = gkv[gkv_key == str(code).strip()]
            if sub.empty:
                unmatched.append(code)
                continue
            for kk, vv in zip(sub[v_K].astype(float).round().astype(int).values,
                              sub[v_col].astype(float).values):
                if kk in G_curve_K:
                    G_pmf_se[s_i, G_curve_K.index(kk)] = (
                        np.sqrt(max(vv, 0.0)) if is_var else abs(vv))
        if unmatched:
            print(f"[load] G_K_var.csv has no row for A129 = {unmatched} (it carries "
                  f"{sorted(gkv_key.unique())[:6]}...); those sectors get no empirical "
                  "band. The A129 column must hold the same sector CODES as G_K.csv.")

    # sector names, in the model's 1..S order (sorted A129), as in load_parameters.jl
    fdf_path = input_folder / "filter_N_upstream.csv"
    if fdf_path.exists():
        filter_N_upstream_df = pd.read_csv(fdf_path)
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
        _gk = [str(a) for a in sector_codes]
        if _gk != list(sector_names):
            raise ValueError(
                "G_K.csv and filter_N_upstream.csv do not enumerate the same sectors:\n"
                f"  G_K.csv            : {_gk}\n"
                f"  filter_N_upstream  : {list(sector_names)}\n"
                "Gbar_s(0) would be attached to the wrong sector (in load_parameters.jl too).")

    # ------------------------------------------------ FULL empirical moment vector
    blocks_emp = [np.array([agg_labor_share], dtype=float),
                  agg_industry_share.astype(float),
                  emp_pi_r,
                  reg_coef.astype(float),
                  emp_gamma_aa.ravel()]            # C-order on (S, n_AA) == Julia vec((n_AA, S))
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
    mask[off_gam:off_gam + S * n_AA] = AA_ACTIVE.ravel()
    for s in range(S):
        if T_REF_AA[s] >= 0:
            mask[off_gam + s * n_AA + T_REF_AA[s]] = False

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
      emp_blocks["gamma_aa"] = emp_blocks["gamma_aa"].reshape(S, n_AA)
      sim_blocks["gamma_aa"] = sim_blocks["gamma_aa"].reshape(S, n_AA)

      # All stages, block by block, with the STAGE axis last — the layout
      # analysis.ipynb's `best_simulated_moments_dict[key][..., K]` expects.
      stage_blocks = {}
      for k, b in zip(keys, np.split(best_simulated_moments, edges, axis=0)):
        stage_blocks[k] = (b.reshape(S, n_AA, b.shape[1]) if k == "gamma_aa" else b)

    # `best_params` is CORE: it carries theta+'s head, so the economy needs it whatever
    # else is being reported.
    bp_path = folder / step_dir / "best_parameters_list.npy"
    if bp_path.exists():
        best_parameters_list = np.load(bp_path)
        best_params = best_parameters_list[:, K] if best_parameters_list.ndim > 1 \
            else best_parameters_list

    # --------------------------------------------------------- standard errors
    # The inference subsystem stacks three moment blocks in a fixed order — the
    # extensive-margin coefficients, the sourcing shares, the zero-supplier shares (the
    # "beta, gamma, G" ordering invariant of CLAUDE.md; the arrows used elsewhere in the
    # code name that stacking order, nothing causal).
    #   Sigma_data       : the bootstrap covariance of the EMPIRICAL moments — how
    #                      precisely each target is measured in the data;
    #   se_moments_fitted: sqrt(diag(G V G')), the SE of the FITTED moment. Under
    #                      profile_T, V is the SANDWICH covariance of alpha_hat and G the
    #                      PROFILED Jacobian dm/dalpha (alpha moved, T following through
    #                      the Sinkhorn inversion), so this is the delta-method
    #                      propagation of alpha_hat's sandwich SE — not an efficient SE
    #                      from the free-parameter Jacobian, which under profiling would
    #                      treat T as free when it is not.
    n_gamma_kept = int(mask[off_gam:off_gam + S * n_AA].sum())
    n_gb = reg_coef.size + n_gamma_kept + (S if granular else 0)

    def _gb_split(v, what):
        v = np.asarray(v).ravel()
        if v.size != n_gb:
            raise ValueError(f"{what} has length {v.size}, expected n_gb = {n_gb} "
                             f"(= {reg_coef.size} beta + {n_gamma_kept} gamma"
                             + (f" + {S} G0)" if granular else ")"))
        out = {"reg_coef": v[:reg_coef.size],
               "gamma_aa": v[reg_coef.size:reg_coef.size + n_gamma_kept]}
        if granular:
            out["G0"] = v[-S:]
        return out

    se_emp, se_sim = None, None
    sigma_path = folder / "step2" / "Sigma_data.npy"
    if "inference" in parts and sigma_path.exists():
        Sigma_data = np.load(sigma_path)
        se_emp = _gb_split(np.sqrt(np.maximum(np.diag(Sigma_data), 0.0)), "diag(Sigma_data)")
    else:
        Sigma_data = None
    fitted_path = folder / inf_step / "inference" / "se_moments_fitted.npy"
    if "inference" in parts and fitted_path.exists():
        se_sim = _gb_split(np.load(fitted_path), "se_moments_fitted.npy")

    # ------- the count DISTRIBUTION: Julia's own fitted curve and its delta-method SE
    # Written by `compute_profiled_T_inference`. `se_G_curve_delta.npy` is (S, |K|):
    # for a FREE sector at K = 0 it collapses to the bootstrap SE of the target (the
    # count moment is fit by construction, so its variance IS the target's); for K >= 1
    # it is the residualized alpha + gamma propagation plus the scaled target channel;
    # for a CLAMPED sector N_hat_s cannot move, so the target channel drops out at
    # every K. Absent (an older run, or a Jacobian reloaded from disk) => the simulated
    # panels fall back to the Python reconstruction with no band.
    G_curve_sim, G_curve_sim_se, G_curve_sim_K = None, None, None
    inf_dir = folder / inf_step / "inference"
    if granular and "inference" in parts:
        p_se, p_fit, p_K = (inf_dir / "se_G_curve_delta.npy",
                            inf_dir / "G_curve_fitted.npy",
                            inf_dir / "count_curve_K.npy")
        if p_se.exists():
            G_curve_sim_se = np.load(p_se)
        if p_fit.exists():
            G_curve_sim = np.load(p_fit)
        if p_K.exists():
            G_curve_sim_K = [int(round(k)) for k in np.load(p_K).ravel()]

    # ----------------------------------------- moment and parameter labels ----
    # Both axes of the Jacobian, rebuilt exactly as load_parameters.jl SECTION 13 does.
    #   rows    : the MASKED moment vector, six blocks;
    #   columns : the IDENTIFIED parameters, layout [Omega_L | Omega_s(S) | A(R_d) |
    #             alpha(N_TAU) | T(active (s,AA), s-major)], minus the S+2 directions
    #             the internal normalisations kill (Omega_s[1], A[1], each sector's
    #             reference T).
    moment_labels, moment_block_sizes = [], []
    moment_labels.append("labor")
    moment_block_sizes.append(1)
    moment_labels += [f"Omega_s[{sector_names[s]}]" for s in range(1, S)]
    moment_block_sizes.append(S - 1)
    moment_labels += [f"pi_r[{a}]" for a in aa_names[1:]]
    moment_block_sizes.append(len(aa_names) - 1)
    moment_labels += ([f"reg_coef[{b + 1}]" for b in range(n_coef)] if n_coef > 1
                      else ["reg_coef"])
    moment_block_sizes.append(n_coef)
    gam_free = AA_ACTIVE.copy()
    for s in range(S):
        if T_REF_AA[s] >= 0:
            gam_free[s, T_REF_AA[s]] = False
    moment_labels += [f"gamma[{sector_names[s]}-AA{aa_names[a]}]"
                      for s in range(S) for a in range(n_AA) if gam_free[s, a]]
    moment_block_sizes.append(int(gam_free.sum()))
    if granular:
        moment_labels += [f"G0[{sector_names[s]}]" for s in range(S)]
        moment_block_sizes.append(S)

    param_labels, param_block_sizes = [], []
    param_labels.append("Omega_L")
    param_block_sizes.append(1)
    param_labels += [f"Omega_s[{sector_names[s]}]" for s in range(1, S)]   # Omega_s[1] normalised out
    param_block_sizes.append(S - 1)
    param_labels += [f"A[{a}]" for a in aa_names[1:]]                      # A[1] normalised out
    param_block_sizes.append(len(aa_names) - 1)
    param_labels += (["alpha"] if n_tau == 1 else [f"alpha_{b + 1}" for b in range(n_tau)])
    param_block_sizes.append(n_tau)
    param_labels += [f"T[{sector_names[s]}-AA{aa_names[a]}]"
                     for s in range(S) for a in range(n_AA) if gam_free[s, a]]
    param_block_sizes.append(int(gam_free.sum()))

    moment_block_names = ["Labor share", "Industry shares", "Downstream sales",
                          "Extensive margin", "Regional sourcing shares"] \
        + (["Zero-supplier share"] if granular else [])
    param_block_names = ["Omega_L", "Industry shares", "Productivity",
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
    jac_file = "jacobian_all.npy" if mu == 1 else "jacobian_all_step3.npy"
    jac_dir = folder / inf_step
    jac = {k: None for k in ("J", "J_elast", "J_sd", "J_elast_sd",
                             "jacobian_param_indices")}
    n_N_cols = None
    if "jacobian" in parts:
      for key, suffix in (("J", ""), ("J_elast", "_elasticity"),
                        ("J_sd", "_sd"), ("J_elast_sd", "_elasticity_sd"),
                        ("jacobian_param_indices", "_param_indices")):
        p = jac_dir / jac_file.replace(".npy", f"{suffix}.npy")
        jac[key] = np.load(p) if p.exists() else None
      n_N_path = jac_dir / jac_file.replace(".npy", "_n_s_columns.npy")
      n_N_cols = int(np.load(n_N_path).ravel()[0]) if n_N_path.exists() else None
    if jac["J"] is not None:
        n_r, n_c = jac["J"].shape
        if n_N_cols is None:
            n_N_cols = n_c - len(param_labels)
        if (n_r != int(mask.sum()) or n_c != len(param_labels) + n_N_cols
                or n_N_cols not in (0, S)):
            raise ValueError(
                f"{jac_file} is {n_r}x{n_c} but the layout rebuilt here is "
                f"{int(mask.sum())} moments x {len(param_labels)} parameters "
                f"(+{n_N_cols} variety counts) — the run's flags and the ones passed "
                "here disagree.")
        if n_N_cols:
            param_labels += [f"N_s[{sector_names[s]}]" for s in range(S)]
            param_block_sizes.append(S)
            param_block_names.append("Variety count")
    else:
        n_N_cols = n_N_cols or 0

    # --------------------------------------------------- variance-covariance ---
    # Sigma_data (bootstrap, empirical) and Sigma_sim (K re-simulations) over the same
    # three-block subsystem; Omega = Sigma_data + Sigma_sim is what the estimator
    # actually weighted with, and W_step3 = inv(Omega).
    def _npy(p):
        return np.load(p) if p.exists() else None

    Sigma_sim = _npy(folder / "step2" / "Sigma_sim.npy") if "inference" in parts else None
    Omega = _npy(folder / "step2" / "Omega.npy") if "inference" in parts else None
    W_step3 = _npy(folder / "step2" / "W_step3.npy") if "inference" in parts else None
    if Omega is None and Sigma_data is not None and Sigma_sim is not None:
        Omega = Sigma_data + Sigma_sim

    # ------------------------------------------------- granular diagnostics
    granular_diag = None
    diag_path = folder / inf_step / "granular_diagnostics.npz"
    if granular and diag_path.exists():
        with np.load(diag_path) as z:
            granular_diag = {k: z[k] for k in z.files}

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

    distances = regional_wage = france = None
    if "geography" in parts:
        distances = _opt(np.load, input_folder / "full_distances.npy")
        regional_wage = _opt(np.load, input_folder / "regional_wages.npy")
        france = None if gpd is None else _opt(
            lambda: gpd.read_file(input_folder / "france.gpkg", encoding="utf-8").sort_values(by="ze2010"))
    ref = None
    if france is not None and distances is not None:
        def _build_ref():
            n = len(france)
            r = pd.DataFrame(distances[:n, :n], index=france["ze2010"].values,
                             columns=france["ze2010"].values).reset_index()
            r.rename(columns={"index": "ze2010_i"}, inplace=True)
            return r.melt(id_vars="ze2010_i", var_name="ze2010_j", value_name="M_ij")
        ref = _opt(_build_ref)

    # Firm-level artefacts. main.jl writes them into the STEP folder of the estimate
    # they were computed at — step1 for theta_hat_1, step3 for theta_hat_2 — so `mu`
    # selects the simulated economy exactly as it selects the moments and the inference.
    # `post_hoc_N_hat` is CORE, not `firm`: the variety count is a coordinate of theta+,
    # so the economy needs it whether or not Julia's own realisation is on disk.
    n_hat_path = folder / step_dir / "post_hoc_N_hat.npy"
    post_hoc_N_hat = _opt(np.load, n_hat_path) if n_hat_path.exists() else None

    suppliers = suppliers_continuum = post_hoc_u = post_hoc_good_sr = w_srd_r = None
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
        # dimensions
        "S": S, "R": R, "n_AA": n_AA, "n_coef": n_coef, "n_tau": n_tau,
        "sector_names": sector_names, "sector_codes": sector_codes, "aa_names": aa_names,
        # structure
        "aa_of_ze": aa_of_ze, "AA_ACTIVE": AA_ACTIVE, "CELL_MASK": CELL_MASK,
        "T_REF_AA": T_REF_AA, "domestic_share": domestic_share,
        "filter_N_upstream": filter_N_upstream, "filter_N_upstream_df": filter_N_upstream_df,
        "X_rs": X_rs, "X_dr": X_dr, "emp_gamma_ls": emp_gamma_ls,
        # scalars / targets
        "epsilon": epsilon, "agg_labor_share": agg_labor_share,
        "agg_industry_share": agg_industry_share, "emp_pi_r": emp_pi_r,
        "reg_coef": reg_coef, "G_target": G_target, "N_supplier_s": N_supplier_s,
        # the WHOLE empirical count curve, Pr(K_ls <= K) by (sector, K); only the
        # K = 0 column is targeted, the rest is the untargeted fit check
        "G_curve": G_curve, "G_curve_K": G_curve_K,
        # the INCREMENTS (exactly K) and their bootstrap SE, which is what is plotted
        "G_pmf": G_pmf, "G_pmf_se": G_pmf_se,
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
        "idf_ze": idf_ze, "distances": distances, "france": france, "ref": ref,
        "regional_wage": regional_wage, "N_downstream": N_downstream,
        "suppliers": suppliers, "suppliers_path": suppliers_path, "w_srd_r": w_srd_r,
        "suppliers_continuum": suppliers_continuum, "post_hoc_N_hat": post_hoc_N_hat,
        # labels / block layout (both Jacobian axes)
        "moment_labels": moment_labels, "moment_block_sizes": moment_block_sizes,
        "moment_block_names": moment_block_names,
        "param_labels": param_labels, "param_block_sizes": param_block_sizes,
        "param_block_names": param_block_names,
        # Jacobian (None when the file is absent)
        "J": jac["J"], "J_elast": jac["J_elast"], "J_sd": jac["J_sd"],
        "J_elast_sd": jac["J_elast_sd"], "n_N_cols": n_N_cols,
        "jacobian_param_indices": jac["jacobian_param_indices"],
        # inference
        "Sigma_data": Sigma_data, "Sigma_sim": Sigma_sim, "Omega": Omega,
        "W_step3": W_step3,
        "se_empirical": se_emp, "se_simulated": se_sim,
        "G_curve_sim": G_curve_sim, "G_curve_sim_se": G_curve_sim_se,
        "G_curve_sim_K": G_curve_sim_K,
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
    for cand in ([Path(root)] if root is not None else []) + [Path("."), Path(".."),
                                                              Path(__file__).parent
                                                              if "__file__" in globals()
                                                              else Path(".")]:
        f = Path(cand) / "load_parameters.jl"
        if not f.exists():
            continue
        m = re.search(r"const\s+theta\s*=\s*\$\(\s*([0-9.eE+-]+)\s*\)", f.read_text())
        if m:
            try:
                v = float(m.group(1))
            except ValueError:
                return None
            return v if np.isfinite(v) and v > 0 else None
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
    jl = _theta_from_julia(data.get("base") if isinstance(data, dict) else None)
    cv = _read_named_value(data["coefs"], "theta") if data.get("coefs") is not None else None
    cv = float(cv) if cv is not None and np.isfinite(cv) and cv > 0 else None
    if jl is not None and cv is not None and abs(jl - cv) > 1e-12 and verbose:
        print(f"  [theta] load_parameters.jl says {jl:g}, stats.csv says {cv:g} — taking "
              "the Julia value, which is what the economy was solved at. One of the two "
              "is stale.")
    if jl is not None:
        return jl
    return cv if cv is not None else float(THETA_DEFAULT)


def unpack_estimated_T(data):
    """
    The estimated comparative advantage, reference-normalised exactly as
    `unpack_params` does, plus alpha.

    `best_params` is the RAW theta vector, layout
        [Omega_L(1) | Omega_s(S) | A(R_d) | alpha(N_TAU) | T(active (s,AA), s-major)],
    where the T block carries EVERY active (sector, area) including each sector's
    reference — `unpack_params` divides the sector by its reference entry, which is
    what makes T_{ref,s} = 1 the gauge. The length is asserted against the rebuilt
    layout, so a flag mismatch is an error rather than a silently shifted vector.
    """
    bp = data.get("best_params")
    if bp is None:
        raise FileNotFoundError(
            f"no best_params in {data['folder']}/{data['step_dir']}/ — "
            "best_parameters_list.npy is written by run_reporting.")
    bp = np.asarray(bp, dtype=float).ravel()

    S, n_AA, n_tau = data["S"], data["n_AA"], data["n_tau"]
    AA_ACTIVE = data["AA_ACTIVE"]
    R_d = len(data["aa_names"])                       # one area per downstream region
    n_T = int(AA_ACTIVE.sum())
    expected = 1 + S + R_d + n_tau + n_T
    if bp.size != expected:
        raise ValueError(
            f"best_params has {bp.size} entries but the layout rebuilt here is "
            f"{expected} = 1 + {S} + {R_d} + {n_tau} + {n_T} — the run's flags and the "
            "ones passed here disagree.")

    alpha = bp[1 + S + R_d: 1 + S + R_d + n_tau]
    T = np.zeros((S, n_AA))
    T[AA_ACTIVE] = bp[1 + S + R_d + n_tau:]           # C-order on (S, n_AA) == s-major
    for s in range(S):
        ref = data["T_REF_AA"][s]
        if ref >= 0 and T[s, ref] > 0:
            T[s] = T[s] / T[s, ref]
    return {"T": T, "alpha": alpha, "theta": model_theta(data)}


def _downstream_ze_index(data):
    """The 1..R ZE index of each attraction area, in AA column order."""
    Nd = np.asarray(data["N_downstream"]).ravel()
    idx = np.flatnonzero(Nd != 0) + 1                 # 1-based, sorted — the model's order
    if idx.size != data["n_AA"]:
        raise ValueError(f"{idx.size} downstream regions in N_downstream_per_region.npy "
                         f"but {data['n_AA']} attraction areas.")
    return idx


def sourcing_geometry(data, alpha=None, equalise_T=False):
    """
    The exact Ricardian win probabilities `rho[s][l, r]` and the pieces they are built
    from, per sector, over CELLS l = regions.

    `alpha=0` switches the distance force off, `equalise_T=True` switches comparative
    advantage off — the two counterfactuals, both closed form.
    """
    est = unpack_estimated_T(data)
    if est["alpha"].size != 1:
        raise ValueError(
            f"N_TAU = {est['alpha'].size}: tau is binned, not the power law d^alpha, so "
            "there is no single distance elasticity to trade off against T. Re-run the "
            "reporting on an n_tau = 1 fit for this section.")
    a = float(est["alpha"][0]) if alpha is None else float(alpha)
    theta = est["theta"]

    D = np.load(data["input_folder"] / "distances.npy")[:data["R"], :data["R"]]
    down = _downstream_ze_index(data)
    aa_of_ze, CELL_MASK = data["aa_of_ze"], data["CELL_MASK"]

    out = {}
    for s in range(data["S"]):
        cells = np.flatnonzero(CELL_MASK[s])                       # 0-based ZE
        if cells.size == 0:
            continue
        # the ONE line that assumes comparative advantage is constant within an area
        T_cell = np.ones(cells.size) if equalise_T else est["T"][s, aa_of_ze[cells]]
        d = np.maximum(D[np.ix_(cells, down - 1)], 1.0)            # (n_cell, R_d)
        psi = T_cell[:, None] * d ** (-theta * a)
        tot = psi.sum(axis=0, keepdims=True)
        out[s] = {"cells": cells, "areas": aa_of_ze[cells], "T_cell": T_cell,
                  "distance": d, "log_psi": np.log(np.maximum(psi, 1e-300)),
                  "rho": np.divide(psi, tot, out=np.zeros_like(psi), where=tot > 0)}
    return {"by_sector": out, "alpha": a, "theta": theta, "T": est["T"],
            "alpha_hat": float(est["alpha"][0]), "downstream": down}


# ============================================================================
# Readers for a firm-level frame, in `suppliers.parquet`'s schema
# ----------------------------------------------------------------------------
# Both are consumed by the economy below (`extended_parameters` needs N_hat_s,
# `check_against_julia` needs the sector index) as well as by the sections, so
# they sit here rather than in either library.
# ============================================================================



def _parquet_sector_index(data, sup):
    """
    The model's 0-based sector index for every row of `suppliers.parquet`.

    The parquet's `A129` column holds the model's OWN index 1..S, not the A129 code —
    the column glossary says so, and `main.jl` writes `sectors = s`. It is read as such
    and then cross-checked: a column that does not fit 1..S is tried against
    `sector_names` before the function gives up, so a file written with real codes is
    NAMED rather than silently mapped onto the wrong sectors.
    """
    S = data["S"]
    col = sup["A129"]
    if pd.api.types.is_numeric_dtype(col):
        idx = col.to_numpy().astype(int) - 1
        if idx.size and idx.min() >= 0 and idx.max() < S:
            return idx
    name_of = {str(c): s for s, c in enumerate(data["sector_names"])}
    mapped = col.astype(str).str.strip().map(name_of)
    if mapped.notna().all():
        return mapped.to_numpy().astype(int)
    bad = sorted(set(col[mapped.isna()].astype(str)))[:5]
    raise ValueError(
        f"cannot place suppliers.parquet's A129 values {bad} among the model's S = {S} "
        f"sectors {list(data['sector_names'])} — the column should be the model index "
        "1..S, as main.jl writes it.")


def _n_hat_from_diagnostics(data):
    """`N_hat_s` from `granular_diagnostics.npz`, the fallback when the post-hoc run
    predates `post_hoc_N_hat.npy`. It is written only when its own step ran, so it can
    be stale relative to the parquet — which is why the post-hoc file is preferred."""
    for step in (data.get("step_dir", "step3"), "step3", "step1"):
        p = data["folder"] / step / "granular_diagnostics.npz"
        if p.exists():
            with np.load(p, allow_pickle=True) as z:
                for k in ("N_hat", "N_s", "N_hat_s"):
                    if k in z.files:
                        return np.asarray(z[k], dtype=float)
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
    w = np.asarray(data["emp_pi_r"], dtype=float).ravel()
    return w / w.sum()



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
#   theta+ = (Omega_L, Omega_s, A, alpha, T, N)   plus the fixed calibration
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
    bp = data.get("best_params")
    if bp is None:
        raise FileNotFoundError(
            f"no best_params in {data['folder']}/{data['step_dir']}/ — "
            "best_parameters_list.npy is written by run_reporting.")
    bp = np.asarray(bp, dtype=float).ravel()
    S, n_AA, n_tau = data["S"], data["n_AA"], data["n_tau"]
    R_d = len(data["aa_names"])
    n_T = int(data["AA_ACTIVE"].sum())
    expected = 1 + S + R_d + n_tau + n_T
    if bp.size != expected:
        raise ValueError(
            f"best_params has {bp.size} entries but the layout rebuilt here is "
            f"{expected} = 1 + {S} + {R_d} + {n_tau} + {n_T}.")

    est = unpack_estimated_T(data)                 # T, gauge-fixed, plus alpha and theta
    if theta is None:
        # ONE route to theta, `model_theta`, which prefers `load_parameters.jl` — the
        # value the economy on disk was solved at. A second reading here would be a
        # second place for the notebook to drift from the estimator.
        theta = model_theta(data, verbose=verbose)
        jl = _theta_from_julia(data.get("base"))
        src = ("load_parameters.jl" if jl is not None and abs(jl - theta) <= 1e-12
               else ("stats.csv" if data.get("coefs") is not None
                     and _read_named_value(data["coefs"], "theta") is not None
                     else "THETA_DEFAULT"))
    else:
        src = "caller"
    if src == "THETA_DEFAULT" and verbose:
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

    xp = {"Omega_L": float(bp[0]),
          "Omega_s": bp[1:1 + S].copy(),
          "A": bp[1 + S:1 + S + R_d].copy(),
          "alpha": est["alpha"].copy(),
          "T": est["T"],                            # (S, n_AA), T[s, ref] = 1
          "N": n_hat[:S].copy(),
          "theta": float(theta), "theta_source": src,
          "nu_s": np.full(S, float(NU_S_DEFAULT)),
          "nu": float(NU_ACROSS_DEFAULT), "lam": float(LAMBDA_DEFAULT),
          "epsilon": float(data["epsilon"]),
          "delta": np.ones(R_d), "wage": np.ones(data["R"])}
    if verbose:
        print(f"  [theta+] Omega_L {xp['Omega_L']:.4f}  alpha {float(xp['alpha'][0]):.4f}  "
              f"theta {xp['theta']:.4f} ({src})  N_s {xp['N'].min()}-{xp['N'].max()}  "
              f"eps {xp['epsilon']:.3f}  nu {xp['nu']:g}/{NU_S_DEFAULT:g}  lam {xp['lam']:g}")
    return xp


def _good_order(data):
    """
    The `(sector, cell)` order Julia's draw matrix is columned by.

    `good_indices = findall(cell_mask)` on an `(S, R)` matrix walks it COLUMN-MAJOR, so
    `g` runs region-outer, sector-inner. Getting this backwards reads one cell's draws
    for another and produces a perfectly plausible wrong economy — the same class of trap
    as the s-major T convention, so it is derived here rather than assumed.
    """
    M = np.asarray(data["CELL_MASK"], dtype=bool)          # (S, R)
    r_idx, s_idx = np.nonzero(M.T)                         # C-order on (R, S) == r outer
    return s_idx, r_idx                                    # 0-based


def simulate_economy(data, xp=None, *, alpha=None, equalise_T=False, n_rep=None,
                     u=None, seed=20260912, geom=None, n_hat=None, buyers=None,
                     verbose=False):
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

    `u` is the uniform draw matrix. Passed (Julia's own, one `(N_max, n_good)` matrix per
    replication), the economy is Julia's to the bit and the comparison is a test rather
    than an agreement. Omitted, draws come from `default_rng(seed)` — independent across
    replications, and SHARED ACROSS BUYERS within one, because a variety is one physical
    good and that shared draw is the whole source of common granularity.

    Returns `variety_panel`'s structure per sector (so every existing consumer reads it
    unchanged) plus the value block per (replication, buyer) and the cost shares.
    """
    if xp is None:
        xp = extended_parameters(data, n_hat=n_hat, verbose=verbose)
    g = sourcing_geometry(data, alpha=alpha, equalise_T=equalise_T) if geom is None else geom
    theta, a = xp["theta"], g["alpha"]
    nu, lam, Om_L, Om_s = xp["nu"], xp["lam"], xp["Omega_L"], xp["Omega_s"]
    S = data["S"]
    down = np.asarray(g["downstream"]).astype(int)          # 1-based ZE of each buyer
    R_d = down.size
    A = xp["A"]
    if A.size != R_d:
        raise ValueError(f"A has {A.size} entries against {R_d} downstream regions.")
    wage = np.asarray(xp["wage"], dtype=float)

    blocks = g["by_sector"]
    missing = [s for s in range(S) if s not in blocks or blocks[s]["cells"].size == 0]
    if missing:
        # An empty sector sends its own CES index to 0 and `P_sr^(1/(1-nu_s))` to
        # infinity, i.e. the buyer's price index is not defined. Julia `continue`s past
        # it and inherits the same infinity, so this is refused rather than reproduced.
        raise ValueError(f"sectors {missing} have no cells — the within-sector price "
                         "index is not defined there.")
    Nv = np.array([int(xp["N"][s]) for s in range(S)])
    if (Nv <= 0).any():
        raise ValueError(f"a sector was given zero varieties: {Nv}")

    # `Nv` and the geometry are needed to validate a supplied draw matrix, so the draw
    # setup sits after them.
    if u is None:
        if n_rep is None:
            n_rep = globals().get("ECONOMY_REPLICATIONS") or 1000
        rng, u_list = np.random.default_rng(seed), None
    else:
        u_list = [np.asarray(x, dtype=float) for x in (u if isinstance(u, (list, tuple)) else [u])]
        if n_rep is None:
            n_rep = len(u_list)
        if len(u_list) < n_rep:
            raise ValueError(f"{len(u_list)} draw matrices supplied for {n_rep} replications.")
        gs, gr = _good_order(data)
        n_good, N_max = gs.size, int(Nv.max())
        for k, M in enumerate(u_list[:n_rep]):
            # Validated UP FRONT rather than at the slice: a matrix one column short
            # would otherwise fail as an out-of-range index deep in the sector loop, and
            # one column too MANY would not fail at all -- it would silently read the
            # wrong cell's draws, which is the trap `_good_order` exists to avoid.
            if M.ndim != 2 or M.shape[1] != n_good or M.shape[0] < N_max:
                raise ValueError(
                    f"draw matrix {k} has shape {M.shape}; Julia writes "
                    f"(N_max, n_good) = (>={N_max}, {n_good}) for this theta+.")
        rng = None

    mu_inv = xp["epsilon"] / (xp["epsilon"] - 1.0)           # mu = eps/(eps-1)
    out, val_rows = Economy(), []
    acc = {s: {"W": [], "v": [], "rep": [], "exp": [], "z": []} for s in range(S)}

    for b in range(n_rep):
        P_sr = np.empty((S, R_d))
        p_keep, win_keep = {}, {}
        for s in range(S):
            blk = blocks[s]
            cells, N, nus = blk["cells"], Nv[s], float(xp["nu_s"][s])
            logT = np.log(np.maximum(blk["T_cell"], 1e-300))
            logd = np.log(np.maximum(blk["distance"], 1.0))          # (cell, buyer)
            if u_list is None:
                uu = rng.random((N, cells.size))
            else:
                # Julia's matrix is (N_max, n_good): take this sector's columns in the
                # good order, then its first N rows — the rows past N_hat carry weight
                # zero there and are clamped, so they are not part of the economy.
                cols = np.flatnonzero(gs == s)
                order = np.argsort(gr[cols])                          # by ZE, as `cells` is
                uu = u_list[b][:N, cols[order]]
                if uu.shape != (N, cells.size):
                    raise ValueError(
                        f"draw matrix {b} gives {uu.shape} for sector {s}, expected "
                        f"{(N, cells.size)}: the good order or N_hat disagrees with Julia.")
            # Frechet inverse CDF, Julia's branch exactly: z = T^(1/theta) * (-log(1-u))^(-1/theta)
            logz = logT[None, :] / theta - np.log(-np.log1p(-uu)) / theta
            lc = a * logd.T[None, :, :] - logz[:, None, :]            # (var, buyer, cell)
            j = lc.argmin(axis=2)
            p = np.exp(np.take_along_axis(lc, j[:, :, None], axis=2)[:, :, 0])   # (var, buyer)
            # the winner's own productivity, which is the parquet's `productivity` column
            zw = np.exp(np.take_along_axis(logz[:, None, :].repeat(j.shape[1], 1),
                                           j[:, :, None], axis=2)[:, :, 0])
            acc[s]["z"].append(zw)
            y = p ** (1.0 - nus)
            P_sr[s] = (y.mean(axis=0)) ** (1.0 / (1.0 - nus))         # weights are 1/N
            p_keep[s], win_keep[s] = p, cells[j] + 1                  # 1-based zone
            acc[s]["v"].append(y / y.sum(axis=0, keepdims=True))
            acc[s]["W"].append(win_keep[s].astype(float))
            acc[s]["rep"].append(np.full(N, b))

        P_r = (Om_s[:, None] * P_sr ** (1.0 - nu)).sum(axis=0) ** (1.0 / (1.0 - nu))
        w_d = wage[down - 1]
        c_r = (Om_L * w_d ** (1.0 - lam) + (1.0 - Om_L) * P_r ** (1.0 - lam)) ** (1.0 / (1.0 - lam))
        c_tilde = c_r / A
        lab_sub = (1.0 - Om_L) * (P_r / c_r) ** (1.0 - lam)           # D_r - 1, exactly
        for s in range(S):
            nus = float(xp["nu_s"][s])
            e = ((1.0 / Nv[s]) * Om_s[s] * (1.0 - Om_L)
                 * (p_keep[s] / P_sr[s][None, :]) ** (1.0 - nus)
                 * (P_sr[s][None, :] / P_r[None, :]) ** (1.0 - nu)
                 * (P_r[None, :] / c_r[None, :]) ** (1.0 - lam))
            acc[s]["exp"].append(e)
        # the downstream demand system: p_r = c_tilde/mu, P = (sum p^eps delta)^(1/eps)
        p_dr = c_tilde / mu_inv
        P_agg = (p_dr ** xp["epsilon"] * xp["delta"]).sum() ** (1.0 / xp["epsilon"])
        Y_r = p_dr ** xp["epsilon"] * P_agg ** (-xp["epsilon"]) * xp["delta"]
        theta_rs = Om_s[:, None] * (P_sr / P_r[None, :]) ** (1.0 - nu)
        val_rows.append({"replication": b, "P_sr": P_sr.copy(), "P_r": P_r,
                         "c_r": c_r, "c_tilde_r": c_tilde, "D_r": 1.0 + lab_sub,
                         "Y_r": Y_r, "P": float(P_agg), "theta_rs": theta_rs})

    all_buyers = down.astype(int)
    if buyers is None:
        keep, kept = slice(None), all_buyers
    else:
        kept = np.asarray(buyers).astype(int)
        pos = {int(b): j for j, b in enumerate(all_buyers)}
        miss = [int(b) for b in kept if int(b) not in pos]
        if miss:
            raise KeyError(f"buyers {miss} are not downstream regions of this economy.")
        keep = np.array([pos[int(b)] for b in kept])
    for s in range(S):
        out[int(s)] = {"winner": np.vstack(acc[s]["W"])[:, keep],
                       "v": np.vstack(acc[s]["v"])[:, keep],
                       "exp_val": np.vstack(acc[s]["exp"])[:, keep],
                       "z": np.vstack(acc[s]["z"])[:, keep],
                       "buyers": kept,
                       "replication": np.concatenate(acc[s]["rep"])}
    val = {k: np.stack([r[k] for r in val_rows]) for k in
           ("P_sr", "P_r", "c_r", "c_tilde_r", "D_r", "Y_r", "theta_rs")}
    val["P"] = np.array([r["P"] for r in val_rows])
    out.value = val
    out.meta = {"buyers": kept, "value_buyers": all_buyers, "n_rep": n_rep,
                "N": Nv, "alpha": a,
                "equalise_T": bool(equalise_T), "theta": theta,
                "draws": "julia" if u_list is not None else f"rng({seed})"}
    return out


def economy_identities(econ, xp, tol=1e-12, verbose=True):
    """
    The identities the forward map satisfies exactly, as a check on the port rather than
    on the economy: each one fixes a different step, so a mismatch NAMES the step.

    (1) `sum_rho (1/N)(p/P_sr)^(1-nu_s) = 1` — the definition of the within-sector index.
    (2) `sum_s Omega_s (P_sr/P_r)^(1-nu) = 1` — the across-sector index.
    (3) `theta_rs` sums to one over sectors, so the input mix comes OUT of the economy.
    (4) `D_r = 1 + sum_{l,s} exp_val = 1 + (1-Omega_L)(P_r/c_r)^(1-lambda)`, which is the
        closed form the section quotes — (1) and (2) are what collapse the two CES
        indices, so this is the one number that fixes the whole expenditure chain.
    """
    # The panel can be a buyer SUBSET while the value block is not (P and Y_r are sums
    # over every downstream region), so the value columns are cut to the panel's buyers
    # before any per-buyer identity is checked.
    allb = np.asarray(econ.meta.get("value_buyers", econ.meta["buyers"])).astype(int)
    kept = np.asarray(econ.meta["buyers"]).astype(int)
    col = np.array([int(np.flatnonzero(allb == b)[0]) for b in kept])
    val = {k: (v[..., col] if getattr(v, "ndim", 0) >= 2 else v)
           for k, v in econ.value.items()}
    sec = sorted(econ)
    rep = econ.meta["n_rep"]
    res = {}
    e1 = 0.0
    for s in sec:
        nus = float(xp["nu_s"][s])
        v = econ[s]["v"]
        # v IS (1/N)(p/P_sr)^(1-nu_s) up to the normalisation, so within a replication
        # it must sum to one over varieties
        r = econ[s]["replication"]
        for b in range(rep):
            e1 = max(e1, abs(v[r == b].sum(axis=0) - 1.0).max())
    res["ces_within"] = e1
    res["ces_across"] = float(np.abs(
        (xp["Omega_s"][None, :, None] * (val["P_sr"] / val["P_r"][:, None, :]) ** (1.0 - xp["nu"])
         ).sum(axis=1) - 1.0).max())
    res["mix_sums_one"] = float(np.abs(val["theta_rs"].sum(axis=1) - 1.0).max())
    tot = np.zeros_like(val["D_r"])
    for s in sec:
        e, r = econ[s]["exp_val"], econ[s]["replication"]
        for b in range(rep):
            tot[b] += e[r == b].sum(axis=0)
    res["D_r_from_shares"] = float(np.abs(val["D_r"] - (1.0 + tot)).max())
    ok = all(v <= tol for v in res.values())
    if verbose:
        print("  [identities] " + "  ".join(f"{k} {v:.2e}" for k, v in res.items())
              + ("  -> ok" if ok else "  -> FAIL"))
    if not ok:
        raise AssertionError(f"the forward map does not close: {res}")
    return res



def check_against_julia(data, xp=None, econ=None, tol=1e-10, verbose=True):
    """
    The cross-language gate: rebuild the ESTIMATED economy here from `theta+` and Julia's
    own draws, then compare it with `suppliers.parquet` row by row.

    This is the one comparison that can be exact. Every other check between the two
    implementations is statistical and cannot separate a coding error from a convention
    mismatch — the Frechet branch, the column order of the draw matrix, the
    within-sector normalisation of the expenditure share. Handed the same `theta+` and
    the same `u`, the forward map is deterministic, so the two must agree to the last
    few bits and anything larger is a defect with an address.

    Three things are checked, in the order a failure should be read. The draw COLUMN map
    first (`post_hoc_good_sr.npy` against `_good_order`): wrong there, every number below
    is wrong in a way that still looks like an economy. Then the WINNERS, which depend on
    `(alpha, T, N, theta)` and the draws alone — a mismatch localises to the Ricardian
    block. Then `exp_val`, which additionally depends on the head and the whole price
    chain, so a mismatch there with matching winners localises to the value block.
    """
    u = data.get("post_hoc_u")
    if u is None:
        raise FileNotFoundError(
            f"no post_hoc_u.npy in {data['folder']}/{data['step_dir']}/ — re-run the "
            "post-hoc block of main.jl, which writes the draws beside the economy. "
            "Without them the two implementations can only be compared statistically.")
    sup = data.get("suppliers")
    if sup is None or "replication" not in getattr(sup, "columns", []):
        raise FileNotFoundError("no replicated suppliers.parquet to compare against.")
    u = np.asarray(u, dtype=float)
    if u.ndim != 3:
        raise ValueError(f"post_hoc_u.npy has shape {u.shape}, expected (N_max, n_good, n_rep).")

    # (1) the column map, read from the file rather than trusted
    gs, gr = _good_order(data)
    ref = data.get("post_hoc_good_sr")
    if ref is not None:
        ref = np.asarray(ref, dtype=int)
        if ref.shape != (gs.size, 2):
            raise ValueError(f"post_hoc_good_sr.npy is {ref.shape}, expected {(gs.size, 2)}.")
        bad = int((np.abs(ref[:, 0] - (gs + 1)) + np.abs(ref[:, 1] - (gr + 1))).sum())
        if bad:
            raise AssertionError(
                f"the draw column map disagrees with Julia on {bad} entries — "
                "`_good_order` and `findall(cell_mask)` are not walking the mask the "
                "same way, and every comparison below would be meaningless.")

    if xp is None:
        xp = extended_parameters(data, verbose=verbose)
    n_rep = u.shape[2]
    if econ is None:
        econ = simulate_economy(data, xp, u=[u[:, :, b] for b in range(n_rep)],
                                n_rep=n_rep, verbose=verbose)

    sec = _parquet_sector_index(data, sup)
    rep = sup["replication"].to_numpy().astype(int)
    rep0 = rep - rep.min()                     # Julia writes 1..n_rep
    var = sup["variety"].to_numpy().astype(int) - 1
    buy = sup["ze2010_downstream"].to_numpy().astype(int)
    win = sup["ze2010"].to_numpy().astype(int)
    val = sup["share"].to_numpy(dtype=float)
    bpos = {int(b): j for j, b in enumerate(np.asarray(econ.meta["buyers"]).astype(int))}

    rows = []
    for s in sorted(econ):
        N = int(econ.meta["N"][s])
        m = (sec == s) & (rep0 < n_rep) & (var < N)
        if not m.any():
            rows.append({"sector": s, "rows": 0, "winner_mismatch": np.nan,
                         "max_abs_exp": np.nan, "max_rel_exp": np.nan})
            continue
        j = np.array([bpos[b] for b in buy[m]])
        r = rep0[m] * N + var[m]
        W = econ[s]["winner"].astype(int)[r, j]
        E = econ[s]["exp_val"][r, j]
        d = np.abs(E - val[m])
        rows.append({"sector": s, "rows": int(m.sum()),
                     "winner_mismatch": int((W != win[m]).sum()),
                     "max_abs_exp": float(d.max()),
                     "max_rel_exp": float((d / np.maximum(np.abs(val[m]), 1e-300)).max())})
    out = pd.DataFrame(rows).set_index("sector")
    worst_w = int(np.nansum(out["winner_mismatch"]))
    worst_e = float(np.nanmax(out["max_rel_exp"]))
    out.attrs["agrees"] = bool(worst_w == 0 and worst_e <= tol)
    if verbose:
        print(f"  [julia] {int(out['rows'].sum())} linkages compared over {n_rep} "
              f"replications: {worst_w} winner mismatches, max relative gap on exp_val "
              f"{worst_e:.2e}" + ("  -> the two implementations are the same economy"
                                  if out.attrs["agrees"] else
                                  "  -> THEY DISAGREE; read the winners before the shares"))
    return out


def economy_frame(econ, xp, drop_zero=False):
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

    Julia drops rows whose `share` is exactly zero (a variety past `N_hat` carries weight
    zero there and would otherwise read as a supplier that supplies nothing). Here every
    row carries a positive share by construction, since the economy is drawn at `N_hat`
    exactly, so `drop_zero` is off by default and is a convenience for comparing against
    a parquet rather than a correction.
    """
    rows = []
    mu = xp["epsilon"] / (xp["epsilon"] - 1.0)
    allb = np.asarray(econ.meta.get("value_buyers", econ.meta["buyers"])).astype(int)
    kept = np.asarray(econ.meta["buyers"]).astype(int)
    col = np.array([int(np.flatnonzero(allb == b)[0]) for b in kept])
    Y = econ.value["Y_r"][:, col]                 # (n_rep, n_buyer)
    Dm1 = econ.value["D_r"][:, col] - 1.0         # the labour-substitution factor
    for s in sorted(econ):
        blk = econ[s]
        N = int(econ.meta["N"][s])
        rep = np.asarray(blk["replication"]).astype(int)
        W = np.asarray(blk["winner"]).astype(int)
        E = np.asarray(blk["exp_val"], dtype=float)
        Z = np.asarray(blk.get("z", np.full(E.shape, np.nan)), dtype=float)
        var = np.tile(np.arange(N), E.shape[0] // N)
        nb_ = kept.size
        rows.append(pd.DataFrame({
            "A129": s + 1,
            "ze2010": W.ravel(order="C"),
            "ze2010_downstream": np.tile(kept, E.shape[0]),
            "share": E.ravel(order="C"),
            "downstream_purchase": (Y[rep] * mu).ravel(order="C"),
            "intermediate_derivative": (E / Dm1[rep]).ravel(order="C"),
            "productivity": Z.ravel(order="C"),
            "sample_weight": 1.0 / N,
            "variety": np.repeat(var + 1, nb_),
            "replication": np.repeat(rep + 1, nb_)}))
    out = pd.concat(rows, ignore_index=True)
    if drop_zero:
        out = out[out["share"] != 0.0].reset_index(drop=True)
    key = out[["replication", "ze2010", "A129", "variety"]].astype(int)
    out["SIREN"] = pd.factorize(pd.MultiIndex.from_frame(key))[0] + 1
    return out[["SIREN", "A129", "ze2010", "ze2010_downstream", "share",
                "downstream_purchase", "intermediate_derivative", "productivity",
                "sample_weight", "variety", "replication"]]


def continuum_economy(data, xp=None, n_var=200, n_rep=8, seed=20260913, **kw):
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
    if xp is None:
        xp = extended_parameters(data, verbose=False)
    S = data["S"]
    return simulate_economy(data, xp, n_rep=n_rep, seed=seed,
                            n_hat=np.full(S, int(n_var)), **kw)


def continuum_data(data, xp=None, n_var=200, n_rep=8, seed=20260913, **kw):
    """`simulated_data` for the infinite-variety limit -- the benchmark frame."""
    if xp is None:
        xp = extended_parameters(data, verbose=False)
    econ = continuum_economy(data, xp, n_var=n_var, n_rep=n_rep, seed=seed, **kw)
    return simulated_data(data, econ, xp)


def simulated_data(data, econ, xp):
    """
    A shallow copy of `data` whose `suppliers` frame IS this economy.

    Everything that reads the parquet then reads the regime instead, with no argument
    threaded through it — which is what lets a counterfactual be measured by the SAME
    functions as the estimated economy rather than by re-derivations of them.
    """
    return {**data, "suppliers": economy_frame(econ, xp),
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
    object, which carries the value block) and `xp` (the extended parameter set it was
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
        xp = extended_parameters(data, verbose=verbose)
        regs = economy_by_regime(data, xp, regimes=regimes, n_rep=n_rep, seed=seed,
                                 verbose=verbose)
        _REPORTING_CACHE[key] = {reg: {**dl, "economy": e, "xp": xp}
                                 for reg, (e, dl) in regs.items()}
    return _REPORTING_CACHE[key]


def economy_by_regime(data, xp=None, regimes=None, n_rep=None, seed=20260912,
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
    if xp is None:
        xp = extended_parameters(data, verbose=verbose)
    regs = CF_REGIMES if regimes is None else regimes
    if baseline not in regs:
        raise KeyError(f"baseline {baseline!r} not among {list(regs)}.")

    if data.get("post_hoc_u") is not None and data.get("suppliers") is not None:
        try:
            agree = check_against_julia(data, xp, verbose=verbose)
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

    out = {}
    for reg, kw in regs.items():
        e = simulate_economy(data, xp, n_rep=n_rep, seed=seed, verbose=False, **kw)
        economy_identities(e, xp, verbose=False)
        out[reg] = (e, simulated_data(data, e, xp))
        if verbose:
            V = e.value
            print(f"  {reg:<28s} D_r {V['D_r'].mean():.4f}  P_r {V['P_r'].mean():.4f}  "
                  f"n_rep {e.meta['n_rep']}  draws {e.meta['draws']}")
    return out
