#!/bin/bash
#
# run.sh - Launch three-step SMM or analytical GMM calibration
#
# Usage (SMM):
#   ./run.sh <industry> [--n_coef=N] [--n_tau=N] [--mode=smm|gmm] [--n_quad=N] [--draws=qmc|mc|is|sobol] [--optimizer=pso|cmaes|tiktak] [--profile_T=true|false] [--n_rho_inf=N] [--reg=cloglog|lpm] [--controls=true|false] [--granular=true|false] [--ca_level=ze|aa] [--n_rep=N]
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
#   ./run.sh aero --n_coef=4 --granular=false --ca_level=ze  # the legacy model (V0 reference)
#   ./run.sh aero --n_coef=4 --n_tau=1 --granular=true --ca_level=aa --n_rep=300
#   ./run.sh both --n_coef=4 --mode=smm

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <industry> [--n_coef=N] [--n_tau=N] [--mode=smm|gmm] [--n_quad=N] [--draws=qmc|mc|is|sobol] [--optimizer=pso|cmaes|tiktak] [--profile_T=true|false] [--n_rho_inf=N] [--reg=cloglog|lpm] [--controls=true|false] [--granular=true|false] [--ca_level=ze|aa] [--n_rep=N]"
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
    echo "  --controls : include the no-supplier control group — true  or false (default) (⇒ supplier pairs only, WITH the size control); SMM only"
    echo "  --granular : true|false (default false) — finite N_s varieties per sector + the count moment G_s(0); SMM only. REQUIRES --ca_level=aa"
    echo "  --ca_level : ze (default) or aa — comparative advantage (and the gamma block) at the ZE or attraction-area level; SMM only"
    echo "  --n_rep    : replications per granular loss evaluation (default 300; only used with --granular=true)"
    echo ""
    echo "  NOTE: --granular=false --ca_level=ze is the continuum ZE-level model exactly as it stood."
    echo "        --granular=true is only valid with --ca_level=aa."
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
N_RHO_INF="4000"  # draw count for inference (Jacobian + Σ_sim), decoupled from N_rho
REG="cloglog"    # SMM only: extensive-margin regression link — cloglog (default) or lpm
CONTROLS="false"  # SMM only: include the no-supplier control group in the reg
GRANULAR="false"  # SMM only: finite N_s varieties + the count moment (block 6)
CA_LEVEL="ze"     # SMM only: comparative advantage at the ZE (:ze) or attraction-area (:aa) level
N_REP="300"       # replications per granular loss evaluation

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
        --granular=*) GRANULAR="${arg#--granular=}" ;;
        --ca_level=*) CA_LEVEL="${arg#--ca_level=}" ;;
        --n_rep=*)    N_REP="${arg#--n_rep=}" ;;
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

if [ "$GRANULAR" != "true" ] && [ "$GRANULAR" != "false" ]; then
    echo "Error: --granular must be true or false (got: $GRANULAR)"
    exit 1
fi

if [ "$CA_LEVEL" != "ze" ] && [ "$CA_LEVEL" != "aa" ]; then
    echo "Error: --ca_level must be ze or aa (got: $CA_LEVEL)"
    exit 1
fi

case "$N_REP" in
    ''|*[!0-9]*) echo "Error: --n_rep must be a positive integer (got: $N_REP)"; exit 1 ;;
esac

# --granular=true REQUIRES --ca_level=aa. Under ZE-level comparative advantage only
# the supplier cells are estimated, so no cell can come out empty: the count moment
# has no content, and the estimator would be fitting T > 0 for regions whose observed
# gamma_ls is 0. The reverse (--granular=false --ca_level=aa) is the AA-level
# continuum, a legitimate intermediate, so it only warns.
if [ "$GRANULAR" = "true" ] && [ "$CA_LEVEL" != "aa" ]; then
    echo "Error: --granular=true requires --ca_level=aa (got --ca_level=$CA_LEVEL)."
    echo "       Under ZE-level comparative advantage only supplier cells are estimated,"
    echo "       so no cell can be empty: G_s(0) has no content and T > 0 would be fitted"
    echo "       for regions whose observed gamma_ls is 0."
    exit 1
fi
if [ "$GRANULAR" = "false" ] && [ "$CA_LEVEL" = "aa" ]; then
    echo "Warning: --ca_level=aa with --granular=false is the AA-level CONTINUUM (gamma"
    echo "         aggregated to areas, no count moment). Make sure the on-disk Sigma file"
    echo "         is the AA one."
fi

# Granularity and the AA layer are SMM-only: the analytical/GMM extensive margin is
# the FKG-approximated continuum object, which has no count moment.
if [ "$MODE" = "gmm" ] && { [ "$GRANULAR" = "true" ] || [ "$CA_LEVEL" = "aa" ]; }; then
    echo "Warning: --granular/--ca_level are ignored in GMM mode (SMM-only); proceeding with (false, ze)."
    GRANULAR="false"
    CA_LEVEL="ze"
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
        # Must match main.jl: reporting_<ind>[_profiled][_aa][_gran]_<optimizer>
        local reporting_folder="reporting_${ind}"
        [ "$PROFILE_T" = "true" ] && reporting_folder="${reporting_folder}_profiled"
        [ "$CA_LEVEL" = "aa" ]    && reporting_folder="${reporting_folder}_aa"
        [ "$GRANULAR" = "true" ]  && reporting_folder="${reporting_folder}_gran"
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
    # main.jl    : industry n_coef n_tau K_sim draws optimizer profile_T n_rho_inf reg controls granular ca_level n_rep
    # main_gmm.jl: industry n_coef n_tau n_quad draws
    if [ "$MODE" = "gmm" ]; then
        local args="$ind $N_COEF $N_TAU $N_QUAD $DRAWS"
        echo "Starting GMM for industry: $ind (n_coef=$N_COEF, n_tau=$N_TAU, n_quad=$N_QUAD, draws=$DRAWS)"
    else
        local args="$ind $N_COEF $N_TAU 10000 $DRAWS $OPTIMIZER $PROFILE_T $N_RHO_INF $REG $CONTROLS $GRANULAR $CA_LEVEL $N_REP"
        echo "Starting SMM for industry: $ind (n_coef=$N_COEF, n_tau=$N_TAU, draws=$DRAWS, optimizer=$OPTIMIZER, profile_T=$PROFILE_T, n_rho_inf=$N_RHO_INF, reg=$REG, controls=$CONTROLS, granular=$GRANULAR, ca_level=$CA_LEVEL, n_rep=$N_REP)"
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
        # Must match main.jl: reporting_<ind>[_profiled][_aa][_gran]_<optimizer>
        monitor_folder="reporting_${INDUSTRY}"
        [ "$PROFILE_T" = "true" ] && monitor_folder="${monitor_folder}_profiled"
        [ "$CA_LEVEL" = "aa" ]    && monitor_folder="${monitor_folder}_aa"
        [ "$GRANULAR" = "true" ]  && monitor_folder="${monitor_folder}_gran"
        monitor_folder="${monitor_folder}_${OPTIMIZER}"
        echo "Monitor: tail -f ${monitor_folder}/logs.log"
    fi
    echo "Stop:    pkill -f 'julia.*main'"
fi