#!/bin/bash
#
# run_smm.sh - Launch SMM Spatial Comovement calibration
#
# Usage:
#   ./run_smm.sh <industry>
#   ./run_smm.sh aero
#   ./run_smm.sh car
#

set -e  # Exit on error

# Check if industry argument is provided
if [ -z "$1" ]; then
    echo "Error: No industry specified"
    echo "Usage: $0 <industry>"
    echo "Example: $0 aero"
    echo "         $0 car"
    exit 1
fi

INDUSTRY="$1"
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
echo "Starting SMM calibration for industry: $INDUSTRY"
echo "Logs will be written to: $LOG_FILE"
echo ""

nohup julia "$JULIA_SCRIPT" "$INDUSTRY" > "$LOG_FILE" 2>&1 

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