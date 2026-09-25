"""Gates for the module layout itself, and for the two properties the refactor promised.

Three things are checked here that no section gate can check on its own.

  1. ONE definition per name across the four modules. The notebook layout forced 75
     duplicated definitions -- the whole diffusion section existed twice, and three
     helpers existed in three copies -- and the gates policed the copies for drift. A
     module removes the possibility; this is the assertion that keeps it removed.

  2. The SPECIALISED loader. `tests_counterfactuals.ipynb` reads no moment, inference or
     Jacobian artefact, and used to fail at the door of a tree that carried none. The
     gate builds a run tree with those files DELETED and requires `parts=("core",
     "geography")` to open it -- and requires asking for a part whose files are missing
     to fail, so the guard is not vacuous.

  3. The PARQUET-FREE path. `data["suppliers"]` is None and every regime, the estimated
     economy included, comes from `simulate_economy`. What has to hold is that the frame
     it emits drives the reporting stack unchanged: the amplification identity
     `D_r = 1 + sum_l X_lr` closes, and the counterfactuals move the local share while
     leaving `D_r` where it was.

No run tree and no Julia are needed; the fixture is written here.
"""
import ast, os, shutil, sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils                                                          # noqa: E402
import diffusion_lib                                                  # noqa: E402

MODS = ["utils.py", "report_lib.py", "diffusion_lib.py", "granular_lib.py"]

# --- 1. one definition per name ------------------------------------------------------
where = {}
for m in MODS:
    for n in ast.parse(open(os.path.join(ROOT, m), encoding="utf-8").read()).body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            where.setdefault(n.name, []).append(m)
dupes = {k: v for k, v in where.items() if len(v) > 1}
assert not dupes, f"defined in more than one module: {dupes}"
print(f"1 ok  {len(where)} functions across {len(MODS)} modules, none defined twice")

# --- 2. the specialised loader ------------------------------------------------------
# The fixture tree is written by `_fixture_tree.build()` -- see that file for what it
# does and does not carry. The notebook gate reuses it, so the two agree by construction.
from _fixture_tree import build as build_tree                         # noqa: E402
TMP, KW = build_tree()

# NOTHING under step3 but best_parameters_list and post_hoc_N_hat; no
# best_simulated_moments, no Sigma_*, no jacobian_*, no suppliers.parquet.
data = utils.load_granular_data("test", mu=2, parts=("core", "geography"), **KW)
for k in ("best_simulated_moments", "J", "Sigma_data", "Omega", "suppliers",
          "se_empirical", "G_curve_sim"):
    assert data[k] is None, f"{k} was read although its part was not asked for"
assert data["best_params"] is not None and data["post_hoc_N_hat"] is not None
assert data["distances"] is not None, "geography was asked for"
assert data["G_target"] is not None and np.isnan(data["G_target"]).all(), \
    "counts were not asked for, so Gbar_s(0) must come back empty rather than read"
# and asking for a part whose files are absent must FAIL rather than pass quietly
try:
    utils.load_granular_data("test", mu=2, parts=("moments",), **KW)
    raise AssertionError("a missing best_simulated_moments.npy was not reported")
except FileNotFoundError:
    pass
try:
    utils.load_granular_data("test", mu=2, parts=("core", "nope"), **KW)
    raise AssertionError("an unknown part was not refused")
except ValueError:
    pass
assert utils.resolve_parts("jacobian") == {"jacobian", "moments", "counts"}
print(f"2 ok  a tree with no moment, inference or Jacobian artefact opens under "
      f"parts=('core','geography'); asking for a part whose files are absent fails, "
      f"and 'jacobian' pulls in what it is read against")

# --- 3. the parquet-free path --------------------------------------------------------
assert data["suppliers"] is None                     # there is no parquet in this tree
xp = utils.extended_parameters(data, verbose=False)
regs = utils.economy_by_regime(data, xp, n_rep=3, verbose=False)
assert set(regs) == set(utils.CF_REGIMES)

D_of, loc_of = {}, {}
for reg, (econ, dl) in regs.items():
    assert dl["suppliers"] is not None and len(dl["suppliers"]) > 0
    utils.economy_identities(econ, xp, verbose=False)
    summ = diffusion_lib.amplification_summary(dl, radii=(200,))
    # the identity the section quotes: D_r = 1 + sum over upstream cells
    hand = 1.0 + econ.value["D_r"].mean(axis=0) - 1.0
    assert np.allclose(np.sort(summ["amplification"].to_numpy()), np.sort(hand),
                       rtol=1e-10), (summ["amplification"].to_numpy(), hand)
    D_of[reg] = float(summ["amplification"].mean())
    loc_of[reg] = float(summ["share_within_200km"].mean())

# TWO routes to a counterfactual, and they answer different questions -- which is worth
# gating precisely because the older one is the reader's default expectation.
#
#   `counterfactual_diffusion_frame` REALLOCATES the euros across cells at a fixed
#   sector spend. Since sum_l rho = 1 per (sector, buyer), the total upstream sales of a
#   shocked region -- hence D_r -- is IDENTICAL under every regime: cancelling a channel
#   moves a shock in space, not in size. That is a property of the one-tier structure.
#
#   `simulate_economy` re-solves the WHOLE forward map, so the price indices move with
#   the regime and D_r = 1 + (1-Omega_L)(P_r/c_r)^(1-lambda) moves with them. It is not
#   an inconsistency: the first holds the cost shares fixed BY CONSTRUCTION, the second
#   lets them respond. A figure that mixes the two would be reading a size change as a
#   geographic one.
assert max(loc_of.values()) - min(loc_of.values()) > 1e-6, loc_of
assert max(D_of.values()) - min(D_of.values()) > 1e-6, \
    ("the simulated regimes must move D_r -- holding it at the baseline is exactly what "
     "the two-route arrangement had to do", D_of)
base_dl = regs["Both forces"][1]
cf = {lab: diffusion_lib.counterfactual_amplification(base_dl, radii=(200,),
                                                      regimes={lab: kw}, verbose=False)
      for lab, kw in utils.CF_REGIMES.items()}
D_fixed = {lab: float(v["amplification"].mean()) for lab, v in cf.items()}
assert max(D_fixed.values()) - min(D_fixed.values()) < 1e-9, \
    ("the closed-form reallocation holds the sector spend fixed, so D_r cannot move",
     D_fixed)
# the continuum benchmark is simulated here instead of read from a second parquet
cont = utils.continuum_data(data, xp, n_var=40, n_rep=2)
rep = diffusion_lib.granularity_report(data=regs["Both forces"][1], radii=(200,),
                                       continuum=cont["suppliers"], verbose=False)
assert "granularity" in rep.columns and np.isfinite(rep["granularity"]).any()
assert rep.loc["supplier_cells", "granularity"] < 0.0, \
    "the finite-variety economy must reach FEWER origins than its continuum limit"
print(f"3 ok  every regime is built from theta+ with no parquet in the tree; D_r closes "
      f"against the value block ({D_of['Both forces']:.4f}) and now MOVES across regimes "
      f"(range {max(D_of.values()) - min(D_of.values()):.4f}) where the closed-form "
      f"reallocation holds it fixed, the local share moves under both, and the "
      f"continuum benchmark is simulated "
      f"({rep.loc['supplier_cells', 'granularity']:.1f} origins of granularity)")

shutil.rmtree(TMP, ignore_errors=True)
print("\nall gates pass")
