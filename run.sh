#!/bin/bash
#
# run.sh — three-step SMM, main specification (granular varieties, attraction-area
# comparative advantage T_a n^κ, cell-level cloglog, T profiled). Run from the parent
# directory of the repository, next to baseline_<industry>/.
#
#   ./run.sh aero
#   ./run.sh aero --n_coef=4 --n_tau=1        # power law τ = d^α instead of one α per segment
#   ./run.sh aero --relax_n_lo=true          # diagnostic: N_LO forced to 1
#   ./run.sh both                            # auto, then aero once auto has finished
#
# The previous, multi-specification code is in old_version/.

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <industry> [--n_coef=N] [--n_tau=N] [--n_rho_inf=N] [--relax_n_lo=true|false]"
    echo "  industry    : aero, auto, car, or both (auto, then aero, one after the other)"
    echo "  --n_coef    : block-4 specification: 1 (ln d) or 4 (categories ]0,50], ]50,100], >100 km against the
                downstream ZE of the area: 3 coefficients; files beta_d_4.npy, ...) (default 4)"
    echo "  --n_tau     : trade costs: 1 (τ = d^α) or n_coef (one α per distance segment, the main spec) (default n_coef)"
    echo "  --n_rho_inf : UNUSED since inference moved to fake worlds (kept so old calls still parse)"
    echo "  --relax_n_lo: DIAGNOSTIC — force the variety-count lower bound N_LO to 1 instead of"
    echo "                ceil(N_supplier_s / R_downstream); outputs → ..._nlo1_pso/ (default false)"
    exit 1
fi

INDUSTRY="$1"; shift
N_COEF="4"; N_TAU=""; N_RHO_INF="10000"; RELAX_N_LO="false"
for arg in "$@"; do
    case "$arg" in
        --n_coef=*)     N_COEF="${arg#--n_coef=}" ;;
        --n_tau=*)      N_TAU="${arg#--n_tau=}" ;;
        --n_rho_inf=*)  N_RHO_INF="${arg#--n_rho_inf=}" ;;
        --relax_n_lo=*) RELAX_N_LO="${arg#--relax_n_lo=}" ;;
        *) echo "Error: unknown argument '$arg'"; exit 1 ;;
    esac
done

case "$N_COEF" in 1|4) ;; *) echo "Error: --n_coef must be 1 or 4 (got $N_COEF)"; exit 1 ;; esac
[ -z "$N_TAU" ] && N_TAU="$N_COEF"
if [ "$N_TAU" != "1" ] && [ "$N_TAU" != "$N_COEF" ]; then
    echo "Error: --n_tau must be 1 or equal to --n_coef (got $N_TAU)"; exit 1
fi
[[ "$N_RHO_INF" =~ ^[0-9]+$ ]] || { echo "Error: --n_rho_inf must be a positive integer"; exit 1; }
case "$RELAX_N_LO" in true|false) ;; *) echo "Error: --relax_n_lo must be true or false"; exit 1 ;; esac
[ "$RELAX_N_LO" = "true" ] && echo "Warning: --relax_n_lo=true drops the model's variety-count lower bound (diagnostic run)."

JULIA_SCRIPT="SMM_Spatial_Comovement/main.jl"
[ -f "$JULIA_SCRIPT" ] || { echo "Error: $JULIA_SCRIPT not found (run from the repository's parent directory)"; exit 1; }

# Prepares the output folder of one industry and prints the shell command that runs it,
# with every line of the Julia output time-stamped into <out>/logs.log.
industry_command() {
    local ind="$1"
    local out="reporting_${ind}_profiled_aa_gran_kappa"
    [ "$RELAX_N_LO" = "true" ] && out="${out}_nlo1"
    out="${out}_pso"
    [ -d "baseline_${ind}" ] || { echo "Error: baseline_${ind} not found" >&2; return 1; }
    mkdir -p "$out"
    echo "Prepared SMM: $ind (n_coef=$N_COEF, n_tau=$N_TAU, n_rho_inf=$N_RHO_INF, relax_n_lo=$RELAX_N_LO). Logs: $out/logs.log" >&2
    echo "echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] Starting SMM: $ind\" >> '$out/logs.log'; julia $JULIA_SCRIPT $ind $N_COEF $N_TAU $N_RHO_INF $RELAX_N_LO 2>&1 | while IFS= read -r line; do echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] \$line\"; done >> '$out/logs.log'"
}

# "both" runs auto, then aero once auto has finished (whatever its outcome), in ONE
# background chain: closing the terminal does not stop it.
case "$INDUSTRY" in
    both) INDUSTRIES="auto aero" ;;
    *)    INDUSTRIES="$INDUSTRY" ;;
esac
CHAIN=""
for ind in $INDUSTRIES; do
    command_of_industry=$(industry_command "$ind")
    CHAIN="${CHAIN:+$CHAIN; }$command_of_industry"
done

echo "Stopping existing Julia processes..."
ps aux | grep '[j]ulia' | awk '{print $2}' | xargs kill -9 2>/dev/null || true
sleep 1

nohup bash -c "$CHAIN" > /dev/null 2>&1 &
echo "Running: $INDUSTRIES (in this order). PID: $!"
echo "Monitor: tail -f reporting_<industry>_*_pso/logs.log   Stop: kill $! && pkill -f 'julia.*main'"
