#!/bin/bash
#
# run_smm.sh - Launch SMM Spatial Comovement calibration
#
# Usage:
#   ./run_smm.sh <industry> <n_coef>
#   ./run_smm.sh aero 4
#   ./run_smm.sh aero 5
#   ./run_smm.sh car 4
#

set -e  # Exit on error

# Check if industry argument is provided
if [ -z "$1" ]; then
    echo "Error: No industry specified"
    echo "Usage: $0 <industry> <n_coef>"
    echo "  industry: aero, car, etc."
    echo "  n_coef: 4 or 5 (number of regression coefficients)"
    echo ""
    echo "Example: $0 aero 4"
    echo "         $0 car 5"
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

REPORTING_FOLDER="reporting_${INDUSTRY}"
LOG_FILE="${REPORTING_FOLDER}/logs.log"
JULIA_SCRIPT="SMM_Spatial_Comovement/main_pso.jl"

# Check if Julia script exists
if [ ! -f "$JULIA_SCRIPT" ]; then
    echo "Error: Julia script not found at $JULIA_SCRIPT"
    exit 1
fi

# Check if baseline data folder exists
BASELINE_FOLDER="baseline_${INDUSTRY}"
if [ ! -d "$BASELINE_FOLDER" ]; then
    echo "Error: Baseline data folder not found at $BASELINE_FOLDER"
    exit 1
fi

# Check if the required reg_coef file exists
REG_COEF_FILE="${BASELINE_FOLDER}/reg_coef_${N_COEF}.npy"
if [ ! -f "$REG_COEF_FILE" ]; then
    echo "Error: Regression coefficient file not found at $REG_COEF_FILE"
    exit 1
fi

# Create reporting folder if it doesn't exist
if [ ! -d "$REPORTING_FOLDER" ]; then
    echo "Creating folder: $REPORTING_FOLDER"
    mkdir -p "$REPORTING_FOLDER"
else
    echo "Folder already exists: $REPORTING_FOLDER"
fi

# Kill any existing Julia processes
echo "Stopping any existing Julia processes..."
ps aux | grep '[j]ulia' | awk '{print $2}' | xargs kill -9 2>/dev/null || true
sleep 1  # Brief pause to ensure processes are terminated

# Launch Julia program in background
echo "Starting SMM calibration for industry: $INDUSTRY with $N_COEF coefficients"
echo "Using regression coefficients from: $REG_COEF_FILE"
echo "Logs will be written to: $LOG_FILE"
echo ""

nohup julia "$JULIA_SCRIPT" "$INDUSTRY" "$N_COEF" >> "$LOG_FILE" 2>&1 

PID=$!
echo "Process started with PID: $PID"
echo ""
echo "To monitor progress:"
echo "  tail -f $LOG_FILE"
echo ""
echo "To stop the process:"
echo "  kill $PID"
echo ""
echo "To check if still running:"
echo "  ps aux | grep '[j]ulia.*main_pso'"