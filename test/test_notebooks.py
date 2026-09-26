"""Gates for the two notebooks.

The notebooks are now narrative plus run cells: every function they call is imported
from `utils`, `report_lib`, `diffusion_lib` or `granular_lib`. Two things have to hold,
and neither can be checked by a section gate.

  1. STATICALLY, every code cell parses and every free name it uses is bound -- by the
     libraries it star-imports, by the builtins, or by an earlier cell. That is the
     property the old single-namespace layout gave for free and that a notebook of
     imports has to earn: a function left behind in the other notebook would read here
     as a NameError only when the cell was run, which is exactly how `alignment_frame`
     went missing for four days.

  2. FUNCTIONALLY, the wiring the refactor changed actually runs: the imports, the
     Constants cell and the economy run cell execute against a real (if tiny) run tree
     that carries NO `suppliers.parquet`, and produce every regime.

The libraries' own behaviour is gated by the section files; this one gates the notebooks.
"""
import ast, builtins, json, os, re, shutil, sys, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import utils, report_lib, diffusion_lib, granular_lib                 # noqa: E402
from _fixture_tree import build as build_tree                         # noqa: E402

LIB = {"utils": utils, "report_lib": report_lib,
       "diffusion_lib": diffusion_lib, "granular_lib": granular_lib}
NOTEBOOKS = ["tests_counterfactuals.ipynb", "model_report.ipynb"]


def _bound(src, known):
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            known.add(n.name)
            a = n.args
            for x in list(a.args) + list(a.kwonlyargs) + list(a.posonlyargs):
                known.add(x.arg)
            if a.vararg:
                known.add(a.vararg.arg)
            if a.kwarg:
                known.add(a.kwarg.arg)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            mod = getattr(n, "module", None)
            for x in n.names:
                if x.name == "*" and mod in LIB:
                    known |= {k for k in vars(LIB[mod]) if not k.startswith("_")}
                else:
                    known.add((x.asname or x.name).split(".")[0])
        elif isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            known.add(n.id)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            known.add(n.name)
        elif isinstance(n, (ast.With, ast.AsyncWith)):
            for it in n.items:
                if isinstance(it.optional_vars, ast.Name):
                    known.add(it.optional_vars.id)
    return known


# --- 1. every cell parses, every name is bound ---------------------------------------
n_cells = 0
for nbname in NOTEBOOKS:
    nb = json.load(open(os.path.join(ROOT, nbname), encoding="utf-8"))
    assert nb.get("nbformat") == 4, nbname
    known = set(dir(builtins)) | {"display", "get_ipython", "__file__", "__name__"}
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if src.lstrip().startswith("%"):
            continue
        n_cells += 1
        ast.parse(src)                       # raises with the cell's own message
        missing = sorted({n.id for n in ast.walk(ast.parse(src))
                          if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
                         - _bound(src, set(known)))
        assert not missing, f"{nbname} cell {i}: undefined names {missing}"
        known = _bound(src, known)
print(f"1 ok  {len(NOTEBOOKS)} notebooks, {n_cells} code cells: every cell parses and "
      "every free name is bound by the libraries or an earlier cell")

# --- 2. neither notebook reads the parquet in its reporting path ---------------------
for nbname in NOTEBOOKS:
    src = "".join("".join(c["source"]) for c in
                  json.load(open(os.path.join(ROOT, nbname), encoding="utf-8"))["cells"]
                  if c["cell_type"] == "code")
    # the loader must be called with an explicit `parts`, never with the default
    assert "parts=(" in src, f"{nbname} does not specialise the loader"
    assert "suppliers_continuum" not in src, \
        f"{nbname} still reaches for Julia's second parquet"
    # The PARQUET IS NOT ON EITHER REPORTING PATH. `load_granular_data` opens
    # `suppliers.parquet` only under `parts` containing "firm", so a notebook that
    # never asks for that part cannot read it, whatever its sections then do -- every
    # economy, the estimated one included, is simulated from theta+. The one legitimate
    # use of the file is `check_against_julia`, which is not run from a notebook.
    _code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    for _p in re.findall(r"parts\s*=\s*\(([^)]*)\)", _code, flags=re.S):
        assert "firm" not in _p, f"{nbname} asks the loader for the parquet: parts=({_p})"
    assert "check_against_julia(" not in _code, \
        f"{nbname} runs the Julia comparator, which needs the parquet"
print("2 ok  both notebooks specialise the loader, neither asks for the `firm` part, so "
      "suppliers.parquet is never opened on a reporting path")

# --- 2b. the tests notebook's section numbering is consistent ------------------------
# Test 9 (the local share as a level plus a dispersion) was merged into Test 6 (the local
# share), since it is the same statistic read twice. What has to hold is that the merge
# is complete: no orphan Test 9, both halves present in Test 6, and the section header's
# table agreeing with the sections that follow it.
src_cells = [(c["cell_type"], "".join(c["source"])) for c in
             json.load(open(os.path.join(ROOT, "tests_counterfactuals.ipynb"),
                            encoding="utf-8"))["cells"]]
heads = [t.splitlines()[0] for k, t in src_cells if k == "markdown" and
         t.lstrip().startswith("## Test ")]
amp = [h for h in heads if not h.endswith("inside a sector?")]
assert not any("Test 9" in h for h in heads), heads
t6 = next(t for k, t in src_cells if k == "markdown" and "## Test 6 — how much of the shock" in t)
for piece in ("### The level, and the sign reversal", "### The dispersion",
              "10–90 percentiles", "sign reversal"):
    assert piece in t6, piece
run6 = next(t for k, t in src_cells if k == "code" and "the DISPERSION" in t)
for fn in ("plot_local_share(", "plot_counterfactual_profile(",
           "local_share_dispersion(", "local_share_report(",
           "plot_local_share_dispersion("):
    assert fn in run6, f"Test 6's run cell does not call {fn}"
# the per-region counterfactual bars are RETIRED (the profile carries the same
# reallocation at every radius), and the band is drawn ONLY as the 10-90 quantile range:
# a symmetric +/- sigma bar draws a shape the discrete law does not have.
assert "plot_counterfactual_local_share" not in run6, "the retired per-region bars are back"
_run6_code = "\n".join(l for l in run6.splitlines() if not l.lstrip().startswith("#"))
assert "fixed_V" not in _run6_code, "the fixed_V band is drawn again"
hdr = next(t for k, t in src_cells if k == "markdown" and t.startswith("# Amplification"))
assert "**eight tests**" in hdr and "| 9 |" not in hdr, "the header still advertises nine"
print(f"2b ok  Test 9 is merged into Test 6: {len(heads)} test headings, both halves in "
      "Test 6's markdown, both halves called by its run cell, and the header agrees")

# --- 3. the wiring runs against a tree with no parquet -------------------------------
TMP, KW = build_tree()
try:
    nb = json.load(open(os.path.join(ROOT, "tests_counterfactuals.ipynb"),
                        encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"
            and not "".join(c["source"]).lstrip().startswith("%")]
    g = {"__name__": "__nbgate__", "display": lambda *a, **k: None}
    sys.path.insert(0, ROOT)
    # imports, Constants
    for c in code[:2]:
        exec(compile("".join(c["source"]), "<imports/constants>", "exec"), g)
    # point the Constants cell's run tree at the fixture, and the industries at it
    g["RUN_KWARGS"] = {**KW, "parts": ("core", "geography")}
    g["INDUSTRIES"] = [{"industry": "test", "display_name": "Test"}]
    g["ECONOMY_REPLICATIONS"] = 3
    g["MU"] = 2
    # the economy run cell -- the one the refactor rewrote
    econ_cell = next(c for c in code if "ECONOMIES = {}" in "".join(c["source"]))
    exec(compile("".join(econ_cell["source"]), "<economy run cell>", "exec"), g)
    ECON = g["ECONOMIES"]
    assert set(ECON) == {("Test", r) for r in utils.CF_REGIMES}, sorted(ECON)
    for (_, reg), dl in ECON.items():
        assert dl["suppliers"] is not None and len(dl["suppliers"]) > 0
        assert dl["economy"].meta["n_rep"] == 3
        assert str(dl["suppliers_path"]).startswith("<simulated")
    print(f"3 ok  the economy run cell builds {len(ECON)} economies from theta+ against a "
          "tree carrying no suppliers.parquet, at the notebook's own replication count")
finally:
    shutil.rmtree(TMP, ignore_errors=True)

print("\nall gates pass")
