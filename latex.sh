#!/usr/bin/env bash
# setup_latex_matplotlib.sh — install TeX Live deps for matplotlib's usetex
set -euo pipefail

# tlmgr needs write access to the TeX tree; use sudo only if not user-mode
TLMGR="tlmgr"
if ! tlmgr option repository >/dev/null 2>&1; then
    TLMGR="sudo tlmgr"
fi

echo "Updating tlmgr..."
$TLMGR update --self

echo "Installing LaTeX dependencies..."
PKGS=(
    cm-super
    type1cm
    underscore
    geometry
    dvipng
    psnfss
)

# install one by one so a single missing package doesn't kill the run
for p in "${PKGS[@]}"; do
    echo "  -> $p"
    $TLMGR install "$p" || echo "     WARNING: could not install $p (skipping)"
done

echo
echo "Verifying binaries..."
for bin in latex dvipng gs; do
    command -v "$bin" >/dev/null && echo "  OK: $bin" || echo "  MISSING: $bin"
done

echo "Done."