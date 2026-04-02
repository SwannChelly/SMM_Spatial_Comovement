#!/bin/bash
#
# run_smm.sh - Launch SMM Spatial Comovement calibration
#
# Usage:
#   ./run_smm.sh <industry> <n_coef> [optimizer]
#   ./run_smm.sh aero 4          # PSO (default)
#   ./run_smm.sh aero 4 pso      # PSO explicitly
#   ./run_smm.sh aero 4 nm       # Nelder-Mead
#   ./run_smm.sh both 5 nm       # Runs aero then auto with NM
#

set -e  # Exit on error

# Check if industry argument is provided
if [ -z "$1" ]; then
    echo "Error: No industry specified"
    echo "Usage: $0 <industry> <n_coef> [optimizer]"
    echo "  industry:  aero, auto, car, both (runs aero then auto)"
    echo "  n_coef:    4 or 5 (number of regression coefficients)"
    echo "  optimizer: pso (default) or nm (Nelder-Mead)"
    echo ""
    echo "Example: $0 aero 4"
    echo "         $0 aero 4 nm"
    echo "         $0 both 5 nm"
    exit 1
fi

# Check if n_coef argument is provided
if [ -z "$2" ]; then
    echo "Error: No n_coef specified"
    echo "Usage: $0 <industry> <n_coef> [optimizer]"
    echo "  n_coef must be 4 or 5"
    exit 1
fi

INDUSTRY="$1"
N_COEF="$2"
OPTIMIZER="${3:-pso}"

# Validate n_coef
if [ "$N_COEF" != "4" ] && [ "$N_COEF" != "5" ]; then
    echo "Error: n_coef must be 4 or 5, got: $N_COEF"
    exit 1
fi

# Select Julia script based on optimizer
if [ "$OPTIMIZER" = "nm" ] || [ "$OPTIMIZER" = "NM" ]; then
    JULIA_SCRIPT="SMM_Spatial_Comovement/main_NM.jl"
    echo "Optimizer: Nelder-Mead (main_NM.jl)"
elif [ "$OPTIMIZER" = "pso" ] || [ "$OPTIMIZER" = "PSO" ]; then
    JULIA_SCRIPT="SMM_Spatial_Comovement/main_pso.jl"
    echo "Optimizer: PSO (main_pso.jl)"
else
    echo "Error: optimizer must be 'pso' or 'nm', got: $OPTIMIZER"
    exit 1
fi

# Check if Julia script exists
if [ ! -f "$JULIA_SCRIPT" ]; then
    echo "Error: Julia script not found at $JULIA_SCRIPT"
    exit 1
fi

# Function to add timestamp to each line
add_timestamp() {
    while IFS= read -r line; do
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $line"
    done
}

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
        nohup bash -c "julia \"$JULIA_SCRIPT\" \"$ind\" \"$ncoef\" 2>&1 | while IFS= read -r line; do echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] \$line\"; done" >> "$log_file" &
        echo "Process started with PID: $!"
    else
        julia "$JULIA_SCRIPT" "$ind" "$ncoef" 2>&1 | add_timestamp | tee -a "$log_file"
    fi
}

# Kill any existing Julia processes
echo "Stopping any existing Julia processes..."
ps aux | grep '[j]ulia' | awk '{print $2}' | xargs kill -9 2>/dev/null || true
sleep 1  # Brief pause to ensure processes are terminated

# Handle "both" option
if [ "$INDUSTRY" = "both" ]; then
    echo "=========================================="
    echo "Running aero then auto sequentially"
    echo "=========================================="
    echo ""
    
    # Run aero first (foreground, wait for completion)
    echo "--- Running aero ---"
    run_industry "aero" "$N_COEF" "no"
    echo ""
    echo "--- aero completed ---"
    echo ""
    
    # Run auto second (foreground, wait for completion)
    echo "--- Running auto ---"
    run_industry "auto" "$N_COEF" "no"
    echo ""
    echo "--- auto completed ---"
    echo ""
    
    echo "=========================================="
    echo "Both industries completed!"
    echo "=========================================="
    
else
    # Single industry mode (original behavior with nohup)
    run_industry "$INDUSTRY" "$N_COEF" "yes"
    
    echo ""
    echo "To monitor progress:"
    if [ "$OPTIMIZER" = "nm" ] || [ "$OPTIMIZER" = "NM" ]; then
        echo "  tail -f reporting_${INDUSTRY}_NM/optimization.log"
    else
        echo "  tail -f reporting_${INDUSTRY}/logs.log"
    fi
    echo ""
    echo "To stop the process:"
    echo "  pkill -f 'julia.*main_'"
    echo ""
    echo "To check if still running:"
    echo "  ps aux | grep '[j]ulia.*main_'"
fi