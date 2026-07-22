#!/bin/bash
#
# run.sh - Launch three-step SMM or analytical GMM calibration
#
# Usage (SMM):
#   ./run.sh <industry> [--n_coef=N] [--n_tau=N] [--mode=smm|gmm] [--n_quad=N] [--draws=qmc|mc|is|sobol] [--optimizer=pso|cmaes|tiktak] [--profile_T=true|false] [--n_rho_inf=N] [--reg=cloglog|lpm] [--controls=true|false]
#
# Examples:
#   ./run.sh aero
#   ./run.sh aero --n_coef=4
#   ./run.sh aero --n_coef=4 --mode=gmm
#   ./run.sh aero --n_coef=4 --n_tau=1 --mode=gmm
#   ./run.sh aero --n_coef=4 --n_tau=1 --mode=gmm --n_quad=500
#   ./run.sh aero --n_coef=4 --draws=mc
#   ./run.sh aero --n_coef=4 --optimizer=cmaes
#   ./run.sh aero --n_coef=4 --n_tau=1 --optimizer=tiktak --profile_T=true  # multistart on the Sinkhorn-reduced space
#   ./run.sh aero --n_coef=4 --n_tau=1 --profile_T=true   # profile T out of the PSO (SMM only)
#   ./run.sh aero --n_coef=4 --reg=lpm                    # linear-probability extensive margin (default: cloglog)
#   ./run.sh aero --n_coef=4 --controls=false             # drop the no-supplier control group (adds the size control)
#   ./run.sh both --n_coef=4 --mode=smm

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <industry> [--n_coef=N] [--n_tau=N] [--mode=smm|gmm] [--n_quad=N] [--draws=qmc|mc|is|sobol] [--optimizer=pso|cmaes|tiktak] [--profile_T=true|false] [--n_rho_inf=N] [--reg=cloglog|lpm] [--controls=true|false]"
    echo "  industry   : aero, auto, car, both"
    echo "  --n_coef   : number of regression moments: 1, 4, or 5 (default: 4)"
    echo "  --n_tau    : number of trade-cost parameters: 1, 4, or 5 (default: n_coef)"
    echo "  --mode     : smm (default) or gmm (analytical, faster, exact SEs)"
    echo "  --n_quad   : Gauss-Legendre nodes for reg_coef (GMM only, default: 200)"
    echo "  --draws    : Fréchet draw method: sobol (default), mc, is, or sobol"
    echo "  --optimizer: pso (default), cmaes, or tiktak (multistart; SMM only)"
    echo "  --profile_T: true|false (default false; SMM only) — profile T out of the PSO via invert_T_ge; outputs → reporting_<ind>_profiled/"
    echo "  --n_rho_inf: draw count for inference (Jacobian + Σ_sim), decoupled from N_rho (default: 10000)"
    echo "  --reg      : extensive-margin regression link — cloglog (default, coef=αθ) or lpm (linear prob.); SMM only, target file selected to match"
    echo "  --controls : include the no-supplier control group (filter==2) — true  or false (default) (⇒ supplier pairs only, WITH the size control); SMM only"
    exit 1
fi

INDUSTRY="$1"
shift

# Defaults
N_COEF="4"
N_TAU=""       # empty = default to n_coef inside Julia
MODE="smm"
N_QUAD="200"
DRAWS="sobol"    # Fréchet draw method: sobol (default), mc, is, qmc
OPTIMIZER="pso"  # optimizer backend: pso (default) or cmaes
PROFILE_T="false"  # SMM only: profile T out of the PSO via invert_T_ge
N_RHO_INF="10000"  # draw count for inference (Jacobian + Σ_sim), decoupled from N_rho
REG="cloglog"    # SMM only: extensive-margin regression link — cloglog (default) or lpm
CONTROLS="false"  # SMM only: include the no-supplier control group (filter==2) in the reg

for arg in "$@"; do
    case "$arg" in
        --n_coef=*) N_COEF="${arg#--n_coef=}" ;;
        --n_tau=*)  N_TAU="${arg#--n_tau=}" ;;
        --mode=*)   MODE="${arg#--mode=}" ;;
        --n_quad=*) N_QUAD="${arg#--n_quad=}" ;;
        --draws=*)  DRAWS="${arg#--draws=}" ;;
        --optimizer=*) OPTIMIZER="${arg#--optimizer=}" ;;
        --profile_T=*) PROFILE_T="${arg#--profile_T=}" ;;
        --n_rho_inf=*) N_RHO_INF="${arg#--n_rho_inf=}" ;;
        --reg=*)      REG="${arg#--reg=}" ;;
        --controls=*) CONTROLS="${arg#--controls=}" ;;
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

if [ "$OPTIMIZER" != "pso" ] && [ "$OPTIMIZER" != "cmaes" ] && [ "$OPTIMIZER" != "tiktak" ]; then
    echo "Error: --optimizer must be pso, cmaes, or tiktak (got: $OPTIMIZER)"
    exit 1
fi

if [ "$PROFILE_T" != "true" ] && [ "$PROFILE_T" != "false" ]; then
    echo "Error: --profile_T must be true or false (got: $PROFILE_T)"
    exit 1
fi

if [ "$REG" != "cloglog" ] && [ "$REG" != "lpm" ]; then
    echo "Error: --reg must be cloglog or lpm (got: $REG)"
    exit 1
fi

if [ "$CONTROLS" != "true" ] && [ "$CONTROLS" != "false" ]; then
    echo "Error: --controls must be true or false (got: $CONTROLS)"
    exit 1
fi

# T-profiling is an SMM-only feature (Design A: the loss's reg_coef must stay
# simulation-based; GMM's analytical reg_coef is FKG-biased). Ignore it under GMM.
if [ "$MODE" = "gmm" ] && [ "$PROFILE_T" = "true" ]; then
    echo "Warning: --profile_T is ignored in GMM mode (SMM-only); proceeding without profiling."
    PROFILE_T="false"
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
        # Must match main.jl line 110: reporting_<ind>[_profiled]_<optimizer>
        local reporting_folder="reporting_${ind}"
        [ "$PROFILE_T" = "true" ] && reporting_folder="${reporting_folder}_profiled"
        reporting_folder="${reporting_folder}_${OPTIMIZER}"
    fi
    local log_file="${reporting_folder}/logs.log"
    local baseline_folder="baseline_${ind}"

    if [ ! -d "$baseline_folder" ]; then
        echo "Error: Baseline data folder not found at $baseline_folder"
        return 1
    fi

    mkdir -p "$reporting_folder"

    # Build Julia argument string
    # main.jl    : industry n_coef n_tau K_sim draws optimizer profile_T n_rho_inf reg controls
    # main_gmm.jl: industry n_coef n_tau n_quad draws
    if [ "$MODE" = "gmm" ]; then
        local args="$ind $N_COEF $N_TAU $N_QUAD $DRAWS"
        echo "Starting GMM for industry: $ind (n_coef=$N_COEF, n_tau=$N_TAU, n_quad=$N_QUAD, draws=$DRAWS)"
    else
        local args="$ind $N_COEF $N_TAU 10000 $DRAWS $OPTIMIZER $PROFILE_T $N_RHO_INF $REG $CONTROLS"
        echo "Starting SMM for industry: $ind (n_coef=$N_COEF, n_tau=$N_TAU, draws=$DRAWS, optimizer=$OPTIMIZER, profile_T=$PROFILE_T, n_rho_inf=$N_RHO_INF, reg=$REG, controls=$CONTROLS)"
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
        # Must match main.jl line 110: reporting_<ind>[_profiled]_<optimizer>
        monitor_folder="reporting_${INDUSTRY}"
        [ "$PROFILE_T" = "true" ] && monitor_folder="${monitor_folder}_profiled"
        monitor_folder="${monitor_folder}_${OPTIMIZER}"
        echo "Monitor: tail -f ${monitor_folder}/logs.log"
    fi
    echo "Stop:    pkill -f 'julia.*main'"
fi