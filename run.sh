#!/bin/bash
#
# run_smm.sh - Launch SMM Spatial Comovement calibration
#
# Usage:
#   ./run_smm.sh <industry> <n_coef>
#   ./run_smm.sh aero 4
#   ./run_smm.sh aero 5
#   ./run_smm.sh car 4
#   ./run_smm.sh both 5    # Runs aero then auto_23 sequentially
#

set -e  # Exit on error

# Check if industry argument is provided
if [ -z "$1" ]; then
    echo "Error: No industry specified"
    echo "Usage: $0 <industry> <n_coef>"
    echo "  industry: aero, auto_23, car, both (runs aero then auto_23)"
    echo "  n_coef: 4 or 5 (number of regression coefficients)"
    echo ""
    echo "Example: $0 aero 4"
    echo "         $0 car 5"
    echo "         $0 both 5  # runs aero then auto_23 sequentially"
    exit 1
fi

# Check if n_coef argument is provided
if [ -z "$2" ]; then
    echo "Error: No n_coef specified"
    echo "Usage: $0 <industry> <n_coef>"
    echo "  n_coef must be 4 or 5"
    exit 1
fi

INDUSTRY="$1"
N_COEF="$2"

# Validate n_coef
if [ "$N_COEF" != "4" ] && [ "$N_COEF" != "5" ]; then
    echo "Error: n_coef must be 4 or 5, got: $N_COEF"
    exit 1
fi

JULIA_SCRIPT="SMM_Spatial_Comovement/main_pso.jl"

# Check if Julia script exists
if [ ! -f "$JULIA_SCRIPT" ]; then
    echo "Error: Julia script not found at $JULIA_SCRIPT"
    exit 1
fi

# Function to run a single industry
run_industry() {
    local ind="$1"
    local ncoef="$2"
    local use_nohup="$3"  # "yes" for background, "no" for foreground
    
    local reporting_folder="reporting_${ind}"
    local log_file="${reporting_folder}/logs.log"
    local baseline_folder="baseline_${ind}"
    local reg_coef_file="${baseline_folder}/reg_coef_${ncoef}.npy"
    
    # Check if baseline data folder exists
    if [ ! -d "$baseline_folder" ]; then
        echo "Error: Baseline data folder not found at $baseline_folder"
        return 1
    fi
    
    # Check if the required reg_coef file exists
    if [ ! -f "$reg_coef_file" ]; then
        echo "Error: Regression coefficient file not found at $reg_coef_file"
        return 1
    fi
    
    # Create reporting folder if it doesn't exist
    if [ ! -d "$reporting_folder" ]; then
        echo "Creating folder: $reporting_folder"
        mkdir -p "$reporting_folder"
    else
        echo "Folder already exists: $reporting_folder"
    fi
    
    echo "Starting SMM calibration for industry: $ind with $ncoef coefficients"
    echo "Using regression coefficients from: $reg_coef_file"
    echo "Logs will be written to: $log_file"
    echo ""
    
    if [ "$use_nohup" = "yes" ]; then
        nohup julia "$JULIA_SCRIPT" "$ind" "$ncoef" >> "$log_file" 2>&1 &
        echo "Process started with PID: $!"
    else
        julia "$JULIA_SCRIPT" "$ind" "$ncoef" 2>&1 | tee -a "$log_file"
    fi
}

# Kill any existing Julia processes
echo "Stopping any existing Julia processes..."
ps aux | grep '[j]ulia' | awk '{print $2}' | xargs kill -9 2>/dev/null || true
sleep 1  # Brief pause to ensure processes are terminated

# Handle "both" option
if [ "$INDUSTRY" = "both" ]; then
    echo "=========================================="
    echo "Running aero then auto_23 sequentially"
    echo "=========================================="
    echo ""
    
    # Run aero first (foreground, wait for completion)
    echo "--- Running aero ---"
    run_industry "aero" "$N_COEF" "no"
    echo ""
    echo "--- aero completed ---"
    echo ""
    
    # Run auto_23 second (foreground, wait for completion)
    echo "--- Running auto_23 ---"
    run_industry "auto_23" "$N_COEF" "no"
    echo ""
    echo "--- auto_23 completed ---"
    echo ""
    
    echo "=========================================="
    echo "Both industries completed!"
    echo "=========================================="
    
else
    # Single industry mode (original behavior with nohup)
    run_industry "$INDUSTRY" "$N_COEF" "yes"
    
    echo ""
    echo "To monitor progress:"
    echo "  tail -f reporting_${INDUSTRY}/logs.log"
    echo ""
    echo "To stop the process:"
    echo "  pkill -f 'julia.*main_pso'"
    echo ""
    echo "To check if still running:"
    echo "  ps aux | grep '[j]ulia.*main_pso'"
fi