#!/bin/bash
#
# run.sh - Launch three-step efficient SMM calibration
#
# Usage:
#   ./run.sh <industry> [n_coef] [resume] [K_sim]
#   ./run.sh aero
#   ./run.sh aero 4
#   ./run.sh aero 4 resume
#   ./run.sh aero 4 resume 10000
#   ./run.sh both 5
#

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <industry> [n_coef] [resume] [K_sim]"
    echo "  industry : aero, auto, car, both"
    echo "  n_coef   : 1 or 4 or 5 (default: 4)"
    echo "  resume   : pass 'resume' to resume from last checkpoint"
    echo "  K_sim    : number of replications for Σ_sim (default: 10000)"
    exit 1
fi

INDUSTRY="$1"
N_COEF="${2:-4}"
RESUME="${3:-}"
K_SIM="${4:-}"

if [ "$N_COEF" != "4" ] && [ "$N_COEF" != "5" ] && [ "$N_COEF" != "1" ]; then
    echo "Error: n_coef must be 4 or 5, got: $N_COEF"
    exit 1
fi

JULIA_SCRIPT="SMM_Spatial_Comovement/main.jl"
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

    local reporting_folder="reporting_${ind}"
    local log_file="${reporting_folder}/logs.log"
    local baseline_folder="baseline_${ind}"
    #local reg_coef_file="${baseline_folder}/reg_coef_${ncoef}.npy"

    if [ ! -d "$baseline_folder" ]; then
        echo "Error: Baseline data folder not found at $baseline_folder"
        return 1
    fi
    #if [ ! -f "$reg_coef_file" ]; then
    #    echo "Error: Regression coefficient file not found at $reg_coef_file"
    #    return 1
    #fi

    mkdir -p "$reporting_folder"

    # Build argument list
    local args="\"$ind\" \"$ncoef\""
    [ -n "$resume_arg" ] && args="$args \"$resume_arg\""
    [ -n "$k_sim_arg" ]  && args="$args \"$k_sim_arg\""

    echo "Starting three-step SMM for industry: $ind (n_coef=$ncoef)"
    echo "Logs: $log_file"

    if [ "$use_nohup" = "yes" ]; then
        nohup bash -c "julia $JULIA_SCRIPT $args 2>&1 | while IFS= read -r line; do echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] \$line\"; done" >> "$log_file" &
        echo "PID: $!"
    else
        julia $JULIA_SCRIPT $ind $ncoef $resume_arg $k_sim_arg 2>&1 | add_timestamp | tee -a "$log_file"
    fi
}

# Kill any existing Julia processes
echo "Stopping existing Julia processes..."
ps aux | grep '[j]ulia' | awk '{print $2}' | xargs kill -9 2>/dev/null || true
sleep 1

if [ "$INDUSTRY" = "both" ]; then
    echo "Running aero then auto sequentially"
    run_industry "aero" "$N_COEF" "$RESUME" "$K_SIM" "no"
    echo "--- aero completed ---"
    run_industry "auto" "$N_COEF" "$RESUME" "$K_SIM" "no"
    echo "--- auto completed ---"
else
    run_industry "$INDUSTRY" "$N_COEF" "$RESUME" "$K_SIM" "yes"
    echo ""
    echo "Monitor: tail -f reporting_${INDUSTRY}/logs.log"
    echo "Stop:    pkill -f 'julia.*main'"
fi