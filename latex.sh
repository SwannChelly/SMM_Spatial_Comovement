#!/usr/bin/env bash
# latex.sh — install TeX Live packages for (1) matplotlib's usetex and
# (2) the SpatialComovementPaper preamble. Safe to re-run: installed packages are skipped.
#
# Usage:
#   ./latex.sh                      # install the package list
#   ./latex.sh path/to/paper.tex    # ... then compile and auto-install anything still missing
set -euo pipefail

# TinyTeX is not always on PATH in a fresh Onyxia session
TINYTEX_BIN="$HOME/.TinyTeX/bin/x86_64-linux"
if [[ -d "$TINYTEX_BIN" ]]; then export PATH="$TINYTEX_BIN:$PATH"; fi
command -v tlmgr >/dev/null || { echo "ERROR: tlmgr not found"; exit 1; }

# sudo only if the TeX tree is not user-writable (system TeX Live); TinyTeX never needs it
TLMGR=(tlmgr)
[[ -w "$(kpsewhich -var-value TEXMFROOT)" ]] || TLMGR=(sudo tlmgr)

echo "Updating tlmgr..."
"${TLMGR[@]}" update --self || echo "WARNING: tlmgr self-update failed (continuing)"

# TeX Live package names; comments give the .sty name where it differs
PKGS=(
    # --- matplotlib usetex ---
    cm-super type1cm underscore dvipng psnfss
    # --- tools used by LaTeX Workshop (build, linter) ---
    latexmk chktex
    # --- paper preamble ---
    geometry comment threeparttable
    tools            # tabularx
    booktabs
    graphics         # graphicx, color
    caption csquotes etoolbox float
    amsmath
    amsfonts         # amssymb, amsfonts
    amscls           # amsthm
    bbm bbm-macros   # \mathbbm needs both the fonts and the macros
    xfrac setspace
    doublestroke     # dsfont
    makecell natbib
    economic         # aer.bst (and other economics-journal BibTeX styles)
    siunits          # SIunits.sty (TeX Live name is lowercase; tlmgr is case-sensitive)
    algorithms       # algorithm
    algorithmicx     # algpseudocode
    eurosym
    epigraph nextpage   # epigraph requires nextpage
    hyperref
)

# One batch call (one repository load); fall back to one-by-one only if it fails
echo "Installing ${#PKGS[@]} packages..."
if ! "${TLMGR[@]}" install "${PKGS[@]}"; then
    echo "Batch install reported errors; retrying one by one..."
    for p in "${PKGS[@]}"; do
        "${TLMGR[@]}" install "$p" >/dev/null 2>&1 || echo "  WARNING: could not install $p"
    done
fi

# Re-link binaries of newly installed packages (TinyTeX exposes them via ~/bin symlinks)
"${TLMGR[@]}" path add >/dev/null 2>&1 || true

echo
echo "Verifying binaries..."
for bin in pdflatex latex latexmk bibtex chktex dvipng gs; do
    command -v "$bin" >/dev/null && echo "  OK: $bin" || echo "  MISSING: $bin"
done

# Optional: compile the paper; for each missing file, find its package and install it.
# Catches transitive dependencies that the list above cannot anticipate.
if [[ $# -ge 1 ]]; then
    tex=$(realpath "$1"); dir=$(dirname "$tex"); base=$(basename "$tex" .tex)
    echo
    echo "Test-compiling $tex..."
    for _ in {1..15}; do
        # latexmk (not bare pdflatex) so the BibTeX pass runs and missing .bst files surface;
        # -g forces a full rerun after each install
        if out=$(cd "$dir" && latexmk -pdf -g -interaction=nonstopmode -halt-on-error "$base.tex" 2>&1); then
            echo "  OK: compiles"; break
        fi
        # missing .sty/.cls/... (from the .log), else missing .bst (from latexmk's report)
        missing=$(grep -oP "File \`\K[^']+(?=' not found)" "$dir/$base.log" | head -1 || true)
        [[ -n "$missing" ]] || missing=$(grep -A1 'Bst file not found' <<<"$out" | tail -n1 | tr -d '[:space:]' || true)
        if [[ -z "$missing" ]]; then
            echo "  Build fails for a non-package reason: see $dir/$base.log"; break
        fi
        pkg=$(tlmgr search --global --file "/$missing" \
              | awk '/^[A-Za-z0-9._-]+:$/ {sub(/:$/, ""); print; exit}' || true)
        if [[ -z "$pkg" ]]; then
            echo "  No TeX Live package provides $missing"; break
        fi
        echo "  $missing -> installing $pkg"
        "${TLMGR[@]}" install "$pkg" >/dev/null
    done
fi

echo "Done."