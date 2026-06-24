#!/bin/bash
#
# run.sh - Launch three-step SMM or analytical GMM calibration
#
# Usage (SMM):
#   ./run.sh <industry> [--n_coef=N] [--n_tau=N] [--mode=smm|gmm] [--n_quad=N] [--draws=qmc|mc|is|sobol]
#
# Examples:
#   ./run.sh aero
#   ./run.sh aero --n_coef=4
#   ./run.sh aero --n_coef=4 --mode=gmm
#   ./run.sh aero --n_coef=4 --n_tau=1 --mode=gmm
#   ./run.sh aero --n_coef=4 --n_tau=1 --mode=gmm --n_quad=500
#   ./run.sh aero --n_coef=4 --draws=mc
#   ./run.sh both --n_coef=4 --mode=smm

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <industry> [--n_coef=N] [--n_tau=N] [--mode=smm|gmm] [--n_quad=N] [--draws=qmc|mc|is|sobol]"
    echo "  industry : aero, auto, car, both"
    echo "  --n_coef : number of regression moments: 1, 4, or 5 (default: 4)"
    echo "  --n_tau  : number of trade-cost parameters: 1, 4, or 5 (default: n_coef)"
    echo "  --mode   : smm (default) or gmm (analytical, faster, exact SEs)"
    echo "  --n_quad : Gauss-Legendre nodes for reg_coef (GMM only, default: 200)"
    echo "  --draws  : Fréchet draw method: qmc (default), mc, is, or sobol"
    exit 1
fi

INDUSTRY="$1"
shift

# Defaults
N_COEF="4"
N_TAU=""       # empty = default to n_coef inside Julia
MODE="smm"
N_QUAD="200"
DRAWS="qmc"    # Fréchet draw method: qmc (default), mc, is, sobol

for arg in "$@"; do
    case "$arg" in
        --n_coef=*) N_COEF="${arg#--n_coef=}" ;;
        --n_tau=*)  N_TAU="${arg#--n_tau=}" ;;
        --mode=*)   MODE="${arg#--mode=}" ;;
        --n_quad=*) N_QUAD="${arg#--n_quad=}" ;;
        --draws=*)  DRAWS="${arg#--draws=}" ;;
        *) echo "Warning: unknown argument '$arg' ignored" ;;
    esac
done

# If n_tau not set, default to n_coef
N_TAU="${N_TAU:-$N_COEF}"

# Validate
for val in "$N_COEF" "$N_TAU"; do
    if [ "$val" != "1" ] && [ "$val" != "4" ] && [ "$val" != "5" ]; then
        echo "Error: n_coef and n_tau must be 1, 4, or 5 (got: $val)"
        exit 1
    fi
done

if [ "$DRAWS" != "qmc" ] && [ "$DRAWS" != "mc" ] && [ "$DRAWS" != "is" ] && [ "$DRAWS" != "sobol" ]; then
    echo "Error: --draws must be qmc, mc, is, or sobol (got: $DRAWS)"
    exit 1
fi

if [ "$MODE" = "gmm" ]; then
    JULIA_SCRIPT="SMM_Spatial_Comovement/main_gmm.jl"
else
    JULIA_SCRIPT="SMM_Spatial_Comovement/main.jl"
fi

if [ ! -f "$JULIA_SCRIPT" ]; then
    echo "Error: $JULIA_SCRIPT not found"
    exit 1
fi

add_timestamp() {
    while IFS= read -r line; do
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $line"
    done
}

run_industry() {
    local ind="$1"
    local use_nohup="$2"

    if [ "$MODE" = "gmm" ]; then
        local reporting_folder="reporting_gmm_${ind}"
    else
        local reporting_folder="reporting_${ind}"
    fi
    local log_file="${reporting_folder}/logs.log"
    local baseline_folder="baseline_${ind}"

    if [ ! -d "$baseline_folder" ]; then
        echo "Error: Baseline data folder not found at $baseline_folder"
        return 1
    fi

    mkdir -p "$reporting_folder"

    # Build Julia argument string
    # main.jl    : industry n_coef n_tau K_sim draws
    # main_gmm.jl: industry n_coef n_tau n_quad draws
    if [ "$MODE" = "gmm" ]; then
        local args="$ind $N_COEF $N_TAU $N_QUAD $DRAWS"
        echo "Starting GMM for industry: $ind (n_coef=$N_COEF, n_tau=$N_TAU, n_quad=$N_QUAD, draws=$DRAWS)"
    else
        local args="$ind $N_COEF $N_TAU 10000 $DRAWS"
        echo "Starting SMM for industry: $ind (n_coef=$N_COEF, n_tau=$N_TAU, draws=$DRAWS)"
    fi
    echo "Logs: $log_file"

    if [ "$use_nohup" = "yes" ]; then
        nohup bash -c "julia $JULIA_SCRIPT $args 2>&1 | while IFS= read -r line; do echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] \$line\"; done" >> "$log_file" &
        echo "PID: $!"
    else
        julia $JULIA_SCRIPT $args 2>&1 | add_timestamp | tee -a "$log_file"
    fi
}

# Kill any existing Julia processes
echo "Stopping existing Julia processes..."
ps aux | grep '[j]ulia' | awk '{print $2}' | xargs kill -9 2>/dev/null || true
sleep 1

if [ "$INDUSTRY" = "both" ]; then
    echo "Running aero then auto sequentially (mode=$MODE)"
    run_industry "aero" "no"
    echo "--- aero completed ---"
    run_industry "auto" "no"
    echo "--- auto completed ---"
else
    run_industry "$INDUSTRY" "yes"
    echo ""
    if [ "$MODE" = "gmm" ]; then
        echo "Monitor: tail -f reporting_gmm_${INDUSTRY}/logs.log"
    else
        echo "Monitor: tail -f reporting_${INDUSTRY}/logs.log"
    fi
    echo "Stop:    pkill -f 'julia.*main'"
fi