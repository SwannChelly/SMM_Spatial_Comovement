#!/bin/bash
#
# run.sh - Launch three-step SMM or analytical GMM calibration
#
# Usage (SMM, legacy):
#   ./run.sh <industry> [n_coef] [resume] [K_sim]
#   ./run.sh aero
#   ./run.sh aero 4
#   ./run.sh aero 4 resume 10000
#   ./run.sh both 5
#
# Usage (GMM, recommended for inference):
#   ./run.sh aero 4 "" "" --mode=gmm           # n_quad=200 (default)
#   ./run.sh aero 4 "" "" --mode=gmm --n_quad=500
#

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <industry> [n_coef] [resume] [K_sim] [--mode=smm|gmm] [--n_quad=N]"
    echo "  industry : aero, auto, car, both"
    echo "  n_coef   : 1 or 4 or 5 (default: 4)"
    echo "  resume   : pass 'resume' to resume from last checkpoint"
    echo "  K_sim    : replications for Σ_sim (SMM only, default: 10000)"
    echo "  --mode   : smm (default) or gmm (analytical, faster, exact SEs)"
    echo "  --n_quad : Gauss-Legendre nodes for reg_coef (GMM only, default: 200)"
    exit 1
fi

INDUSTRY="$1"
N_COEF="${2:-4}"
RESUME="${3:-}"
K_SIM="${4:-}"
MODE="smm"
N_QUAD="200"

# Parse named flags from remaining args
for arg in "$@"; do
    case "$arg" in
        --mode=*) MODE="${arg#--mode=}" ;;
        --n_quad=*) N_QUAD="${arg#--n_quad=}" ;;
    esac
done

if [ "$N_COEF" != "4" ] && [ "$N_COEF" != "5" ] && [ "$N_COEF" != "1" ]; then
    echo "Error: n_coef must be 1, 4 or 5, got: $N_COEF"
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
    local ncoef="$2"
    local resume_arg="$3"
    local k_sim_arg="$4"
    local use_nohup="$5"
    local mode="$6"
    local n_quad_arg="$7"

    if [ "$mode" = "gmm" ]; then
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

    if [ "$mode" = "gmm" ]; then
        local args="$ind $ncoef \"\" \"\" $n_quad_arg"
        echo "Starting GMM (analytical) for industry: $ind (n_coef=$ncoef, n_quad=$n_quad_arg)"
    else
        local args="$ind $ncoef"
        [ -n "$resume_arg" ] && args="$args $resume_arg"
        [ -n "$k_sim_arg" ]  && args="$args $k_sim_arg"
        echo "Starting SMM for industry: $ind (n_coef=$ncoef)"
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
    run_industry "aero" "$N_COEF" "$RESUME" "$K_SIM" "no" "$MODE" "$N_QUAD"
    echo "--- aero completed ---"
    run_industry "auto" "$N_COEF" "$RESUME" "$K_SIM" "no" "$MODE" "$N_QUAD"
    echo "--- auto completed ---"
else
    run_industry "$INDUSTRY" "$N_COEF" "$RESUME" "$K_SIM" "yes" "$MODE" "$N_QUAD"
    echo ""
    if [ "$MODE" = "gmm" ]; then
        echo "Monitor: tail -f reporting_gmm_${INDUSTRY}/logs.log"
    else
        echo "Monitor: tail -f reporting_${INDUSTRY}/logs.log"
    fi
    echo "Stop:    pkill -f 'julia.*main'"
fi