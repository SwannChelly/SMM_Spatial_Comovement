```bash
#!/usr/bin/env bash

set -euo pipefail

echo "========================================"
echo " Setting up Python / LaTeX environment"
echo "========================================"

# ------------------------------------------------------------
# 1. Make sure TinyTeX is on PATH
# ------------------------------------------------------------

TINY_TEX="$HOME/.TinyTeX/bin/x86_64-linux"

if [ -d "$TINY_TEX" ]; then
    export PATH="$TINY_TEX:$PATH"
else
    echo "ERROR: TinyTeX not found at $TINY_TEX"
    exit 1
fi

echo
echo "LaTeX:"
command -v latex
latex --version | head -n 1

# ------------------------------------------------------------
# 2. Update tlmgr itself
# ------------------------------------------------------------

echo
echo "Updating tlmgr..."
tlmgr update --self

# ------------------------------------------------------------
# 3. Install LaTeX packages required by Matplotlib usetex
# ------------------------------------------------------------

echo
echo "Installing LaTeX dependencies..."

tlmgr install \
    cm-super \
    type1cm \
    underscore \
    geometry \
    inputenc \
    dvips

# ------------------------------------------------------------
# 4. Rebuild TeX filename database
# ------------------------------------------------------------

echo
echo "Updating TeX filename database..."
mktexlsr

# ------------------------------------------------------------
# 5. Verify critical packages
# ------------------------------------------------------------

echo
echo "Checking LaTeX packages..."

PACKAGES=(
    "type1ec.sty"
    "type1cm.sty"
    "underscore.sty"
    "geometry.sty"
)

for package in "${PACKAGES[@]}"; do
    if kpsewhich "$package" >/dev/null 2>&1; then
        echo "  [OK] $package -> $(kpsewhich "$package")"
    else
        echo "  [ERROR] $package not found"
        exit 1
    fi
done

# ------------------------------------------------------------
# 6. Test Matplotlib + LaTeX
# ------------------------------------------------------------

echo
echo "Testing Matplotlib + LaTeX..."

python - <<'PY'
import matplotlib
import matplotlib.pyplot as plt

print(f"Matplotlib: {matplotlib.__version__}")

plt.rcParams["text.usetex"] = True

fig = plt.figure()
plt.xlabel(r"$\alpha$")
plt.savefig("/tmp/matplotlib_latex_test.pdf")
plt.close(fig)

print("[OK] Matplotlib + LaTeX test passed")
PY

# ------------------------------------------------------------
# Done
# ------------------------------------------------------------

echo
echo "========================================"
echo " Environment ready."
echo "========================================"
```
