#!/bin/bash
#
# run.sh — three-step SMM, main specification (granular varieties, attraction-area
# comparative advantage T_a n^κ, cell-level cloglog, T profiled). Run from the parent
# directory of the repository, next to baseline_<industry>/.
#
#   ./run.sh aero
#   ./run.sh aero --n_coef=4 --n_tau=1 --n_rho_inf=10000
#   ./run.sh aero --relax_n_lo=true          # diagnostic: N_LO forced to 1
#   ./run.sh both
#
# The previous, multi-specification code is in old_version/.

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <industry> [--n_coef=N] [--n_tau=N] [--n_rho_inf=N] [--relax_n_lo=true|false]"
    echo "  industry    : aero, auto, car, both"
    echo "  --n_coef    : distance coefficients of the cell cloglog: 1 (ln d), 4 or 5 bins (default 4)"
    echo "  --n_tau     : trade-cost parameters: 1 (τ = d^α) or n_coef (one α per bin) (default 1)"
    echo "  --n_rho_inf : UNUSED since inference moved to fake worlds (kept so old calls still parse)"
    echo "  --relax_n_lo: DIAGNOSTIC — force the variety-count lower bound N_LO to 1 instead of"
    echo "                ceil(N_supplier_s / R_downstream); outputs → ..._nlo1_pso/ (default false)"
    exit 1
fi

INDUSTRY="$1"; shift
N_COEF="4"; N_TAU="1"; N_RHO_INF="10000"; RELAX_N_LO="false"
for arg in "$@"; do
    case "$arg" in
        --n_coef=*)     N_COEF="${arg#--n_coef=}" ;;
        --n_tau=*)      N_TAU="${arg#--n_tau=}" ;;
        --n_rho_inf=*)  N_RHO_INF="${arg#--n_rho_inf=}" ;;
        --relax_n_lo=*) RELAX_N_LO="${arg#--relax_n_lo=}" ;;
        *) echo "Error: unknown argument '$arg'"; exit 1 ;;
    esac
done

case "$N_COEF" in 1|4|5) ;; *) echo "Error: --n_coef must be 1, 4 or 5 (got $N_COEF)"; exit 1 ;; esac
if [ "$N_TAU" != "1" ] && [ "$N_TAU" != "$N_COEF" ]; then
    echo "Error: --n_tau must be 1 or equal to --n_coef (got $N_TAU)"; exit 1
fi
[[ "$N_RHO_INF" =~ ^[0-9]+$ ]] || { echo "Error: --n_rho_inf must be a positive integer"; exit 1; }
case "$RELAX_N_LO" in true|false) ;; *) echo "Error: --relax_n_lo must be true or false"; exit 1 ;; esac
[ "$RELAX_N_LO" = "true" ] && echo "Warning: --relax_n_lo=true drops the model's variety-count lower bound (diagnostic run)."

JULIA_SCRIPT="SMM_Spatial_Comovement/main.jl"
[ -f "$JULIA_SCRIPT" ] || { echo "Error: $JULIA_SCRIPT not found (run from the repository's parent directory)"; exit 1; }

run_industry() {
    local ind="$1" use_nohup="$2"
    local out="reporting_${ind}_profiled_aa_gran_kappa"
    [ "$RELAX_N_LO" = "true" ] && out="${out}_nlo1"
    out="${out}_pso"
    [ -d "baseline_${ind}" ] || { echo "Error: baseline_${ind} not found"; return 1; }
    mkdir -p "$out"
    local args="$ind $N_COEF $N_TAU $N_RHO_INF $RELAX_N_LO"
    echo "Starting SMM: $ind (n_coef=$N_COEF, n_tau=$N_TAU, n_rho_inf=$N_RHO_INF, relax_n_lo=$RELAX_N_LO). Logs: $out/logs.log"
    if [ "$use_nohup" = "yes" ]; then
        nohup bash -c "julia $JULIA_SCRIPT $args 2>&1 | while IFS= read -r line; do echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] \$line\"; done" >> "$out/logs.log" &
        echo "PID: $!   Monitor: tail -f $out/logs.log   Stop: pkill -f 'julia.*main'"
    else
        julia $JULIA_SCRIPT $args 2>&1 | while IFS= read -r line; do
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] $line"; done | tee -a "$out/logs.log"
    fi
}

echo "Stopping existing Julia processes..."
ps aux | grep '[j]ulia' | awk '{print $2}' | xargs kill -9 2>/dev/null || true
sleep 1

if [ "$INDUSTRY" = "both" ]; then
    run_industry aero no && run_industry auto no
else
    run_industry "$INDUSTRY" yes
fi
