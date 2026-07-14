#!/bin/bash
#
# run_test.sh - Launch the internal-validity Monte-Carlo driver
#               (run_internal_validity.jl: synthetic-truth recovery + coverage)
#
# Usage:
#   ./run_test.sh <industry> [--n_coef=N] [--n_tau=N] [--beta0="b1 b2 ..."]
#
# Examples:
#   ./run_test.sh aero
#   ./run_test.sh aero --n_coef=4 --n_tau=1 --beta0="0.5"
#   ./run_test.sh aero --n_coef=4 --n_tau=4 --beta0="0.2 0.4 0.6 0.8"
#   ./run_test.sh both --n_coef=4 --n_tau=1 --beta0="0.5"
#
# Outputs (and logs) land under  internal_validity_<industry>/  exactly as the
# main framework writes to reporting_<industry>/. Logs are timestamped and
# appended to  internal_validity_<industry>/logs.log .

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <industry> [--n_coef=N] [--n_tau=N] [--beta0=\"b1 b2 ...\"]"
    echo "  industry : aero, auto, car, both"
    echo "  --n_coef : number of regression moments: 1, 4, or 5 (default: 4)"
    echo "  --n_tau  : number of trade-cost parameters: 1, 4, or 5 (default: n_coef)"
    echo "  --beta0  : synthetic-truth trade-cost vector (space-separated, length n_tau,"
    echo "             must be non-decreasing). If omitted, the driver uses its default."
    exit 1
fi

INDUSTRY="$1"
shift

# Defaults
N_COEF="4"
N_TAU=""       # empty = default to n_coef inside Julia
BETA0=""       # empty = driver picks a monotone default

for arg in "$@"; do
    case "$arg" in
        --n_coef=*) N_COEF="${arg#--n_coef=}" ;;
        --n_tau=*)  N_TAU="${arg#--n_tau=}" ;;
        --beta0=*)  BETA0="${arg#--beta0=}" ;;
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

JULIA_SCRIPT="SMM_Spatial_Comovement/test/run_internal_validity.jl"
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

    local output_folder="internal_validity_${ind}"
    local log_file="${output_folder}/logs.log"
    local baseline_folder="baseline_${ind}"

    if [ ! -d "$baseline_folder" ]; then
        echo "Error: Baseline data folder not found at $baseline_folder"
        return 1
    fi

    mkdir -p "$output_folder"

    # run_internal_validity.jl args: industry n_coef n_tau [beta0...]
    local args="$ind $N_COEF $N_TAU $BETA0"
    echo "Starting internal-validity test for industry: $ind (n_coef=$N_COEF, n_tau=$N_TAU, beta0=[${BETA0:-default}])"
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
    echo "Running aero then auto sequentially"
    run_industry "aero" "no"
    echo "--- aero completed ---"
    run_industry "auto" "no"
    echo "--- auto completed ---"
else
    run_industry "$INDUSTRY" "yes"
    echo ""
    echo "Monitor: tail -f internal_validity_${INDUSTRY}/logs.log"
    echo "Stop:    pkill -f 'julia.*run_internal_validity'"
fi
