##### Main with Nelder-Mead Optimization #####
# Author: Swann Chelly (CdGM-style two-step bounded Nelder-Mead)
#
# Two-step optimization with identity weight matrix:
#   Step 1: Coarse NM from 20 LHS starting points (all parameters)
#   Step 2: Refined NM from top 15 solutions (tighter convergence)
#
# Usage:
#   nohup julia SMM_Spatial_Comovement/main_NM.jl > reporting_aero/logs.log 2>&1

using Distributed
using Dates
@everywhere using NPZ
@everywhere using QuasiMonteCarlo
@everywhere using StatsPlots
@everywhere using DataFrames
@everywhere using Distributions
@everywhere using Plots
@everywhere using CSV
@everywhere using Random
@everywhere using Optim
@everywhere using Statistics
@everywhere using HaltonSequences
@everywhere using ProgressMeter
@everywhere using SharedArrays
@everywhere using Parquet
using LinearAlgebra
using Statistics, Printf
using StatsBase

# Add workers (20 cores)
available = 20
println("Using $available workers")
addprocs(max(available - 1, 0))

############## Load Parameters #################
industry = length(ARGS) >= 1 ? ARGS[1] : "auto"
n_coef = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4
if !(n_coef in [4, 5])
    error("n_coef must be 4 or 5, got: $n_coef")
end
println("Using $n_coef regression coefficients (reg_coef_$n_coef.npy)")

input_folder = "./baseline_" * industry
output_folder = "./reporting_" * industry * "_NM"
mkpath(output_folder)

coefs = CSV.read(joinpath(input_folder, "stats.csv"), DataFrame)
distances_local = NPZ.npzread(joinpath(input_folder, "distances.npy"))
filter_N_upstream_local = NPZ.npzread(joinpath(input_folder, "filter_N_upstream.npy"))
w_rs_local = NPZ.npzread(joinpath(input_folder, "w_rs.npy"))  # (R,) - regional wages, not (S,R)
regional_wages_local = NPZ.npzread(joinpath(input_folder, "regional_wages.npy"))
N_downstream_per_region_local = NPZ.npzread(joinpath(input_folder, "N_downstream_per_region.npy"))
agg_industry_share_local = NPZ.npzread(joinpath(input_folder, "input_share.npy"))
domestic_share_local = NPZ.npzread(joinpath(input_folder, "domestic_share.npy"))
X_rs_local = NPZ.npzread(joinpath(input_folder, "X_rs.npy"))
N_rs_local = NPZ.npzread(joinpath(input_folder, "N_rs.npy"))

S_, R_ = size(filter_N_upstream_local)
@everywhere const S = $(S_)
@everywhere const R = $(R_)

R_ = size(N_downstream_per_region_local[N_downstream_per_region_local .!= 0])[1]
@everywhere const R_downstream = $(R_)
@everywhere const agg_industry_share = $(agg_industry_share_local)
@everywhere const agg_labor_share = $(coefs[2, "value"])
@everywhere const domestic_share = $(domestic_share_local)
@everywhere regional_wages = $(regional_wages_local)
@everywhere const distances = $(distances_local)
@everywhere const N_downstream_per_region = $(N_downstream_per_region_local)
@everywhere const w_rs = $(w_rs_local)
@everywhere const filter_N_upstream = $(filter_N_upstream_local)
@everywhere const N_rho = $(100)
@everywhere const epsilon = $(coefs[1, "value"])
@everywhere const lambda = $(0.5)
@everywhere const nu = $(0.2)
@everywhere const nu_s = $(ones(S) .* 2.5)
@everywhere const theta = $(1.768)
@everywhere const delta_r = $(ones(R))
@everywhere const Weight_matrix = $(nothing)

if industry == "aero"
    @everywhere const T_rs_init = $(X_rs_local)
elseif industry == "auto"
    @everywhere const T_rs_init = $(X_rs_local)
end

# Mask for non-zero T_rs: only optimize T where gamma_ls > 0
T_mask_local = vec(X_rs_local) .> 0
@everywhere const T_MASK = $T_mask_local
println("T_MASK: $(sum(T_mask_local)) / $(length(T_mask_local)) non-zero T parameters")

# Precompute flat index mappings for good (s,r) pairs
R_full = size(filter_N_upstream_local, 2)  # original R before R_ gets reassigned
good_indices_local = findall(reshape(T_mask_local, S_, R_full))
n_good_local = length(good_indices_local)
GOOD_S_local = [ci[1] for ci in good_indices_local]
GOOD_R_local = [ci[2] for ci in good_indices_local]

SECTOR_GOOD_INDICES_local = [findall(GOOD_S_local .== s) for s in 1:S_]
SECTOR_GOOD_REGIONS_local = [GOOD_R_local[idx] for idx in SECTOR_GOOD_INDICES_local]

SR_TO_GOOD_local = zeros(Int, S_, R_full)
for (g, ci) in enumerate(good_indices_local)
    SR_TO_GOOD_local[ci[1], ci[2]] = g
end

W_RS_FLAT_local = [w_rs_local[GOOD_R_local[g]] for g in 1:n_good_local]  # indexed by region only

@everywhere const n_good = $n_good_local
@everywhere const GOOD_S = $GOOD_S_local
@everywhere const GOOD_R = $GOOD_R_local
@everywhere const SECTOR_GOOD_INDICES = $SECTOR_GOOD_INDICES_local
@everywhere const SECTOR_GOOD_REGIONS = $SECTOR_GOOD_REGIONS_local
@everywhere const SR_TO_GOOD = $SR_TO_GOOD_local
@everywhere const W_RS_FLAT = $W_RS_FLAT_local
println("  n_good: $n_good_local good (sector, region) pairs")

# Load empirical moments
@everywhere const emp_gamma_ls = $(permutedims(NPZ.npzread(joinpath(input_folder, "emp_gamma_ls.npy"))))
X_dr_local = CSV.read(joinpath(input_folder, "X_dr.csv"), DataFrame).X_dr
X_dr_local = X_dr_local[N_downstream_per_region .!= 0]
emp_pi_r_local = X_dr_local ./ sum(X_dr_local)
@everywhere const emp_pi_r_full = $(emp_pi_r_local)
@everywhere const emp_pi_r = $(emp_pi_r_local)  # Full vector; MOMENT_MASK handles [2:end]
@everywhere const reg_coef = $(NPZ.npzread(joinpath(input_folder, "reg_coef_" * string(n_coef) * ".npy")))
@everywhere const N_beta = $(length(NPZ.npzread(joinpath(input_folder, "reg_coef_" * string(n_coef) * ".npy"))))

# Build full empirical moments (no [2:end] drops — MOMENT_MASK handles that)
empirical_moments_local = [[agg_labor_share], vec(agg_industry_share), emp_gamma_ls, reg_coef, emp_pi_r]
empirical_moments_local = vcat([vec(empirical_moments_local[i]) for i in 1:(length(empirical_moments_local))]...)

# Moment block sizes (full)
n_labor = 1
n_industry = length(vec(agg_industry_share))       # S
n_gamma = length(vec(emp_gamma_ls))                 # R_full * S
n_reg = length(reg_coef)                            # N_beta
n_pi = length(emp_pi_r)                             # R_downstream
N_moments_full = n_labor + n_industry + n_gamma + n_reg + n_pi

# Build MOMENT_MASK: true = keep, false = drop (sum-to-1 redundancies)
moment_mask_local = trues(N_moments_full)
moment_mask_local[n_labor + 1] = false                          # first industry share
for s in 1:S_                                                   # first region per sector in gamma_ls
    moment_mask_local[n_labor + n_industry + (s - 1) * R_full + 1] = false
end
moment_mask_local[n_labor + n_industry + n_gamma + n_reg + 1] = false  # first pi_r

# Apply mask to empirical moments
empirical_moments_local = reshape(empirical_moments_local[moment_mask_local], 1, sum(moment_mask_local))
N_moments = sum(moment_mask_local)

# Identity weight matrix
Weight_matrix_custom_local = Diagonal(ones(N_moments))

@everywhere const MOMENT_MASK = $moment_mask_local
@everywhere const empirical_moments = $(empirical_moments_local)
@everywhere const Weight_matrix_custom = $Weight_matrix_custom_local
@everywhere const K_max = $(50)

@everywhere include("model_CP.jl")
@everywhere include("tools.jl")
@everywhere include("run_untargeted_validation.jl")

# Distance bins
DistBin_local = Array{Int}(undef, R, R)
for i in 1:R, j in 1:R
    DistBin_local[i, j] = distance_bin(distances_local[i, j])
end
@everywhere const DistBin = $(DistBin_local)

# Precompute closest downstream plant distance and region
closest_plant_dist_local = vec(map(x -> distances_local[x[1], x[2]], argmin(1 ./ (1 ./ distances_local .* (N_downstream_per_region_local .> 0)'), dims=2)))
closest_downstream_region_local = vec(getindex.(argmin(1 ./ (1 ./ distances_local .* (N_downstream_per_region_local .> 0)'), dims=2), 2))
@everywhere const CLOSEST_PLANT_DIST = $(closest_plant_dist_local)
@everywhere const CLOSEST_DOWNSTREAM_REGION = $(closest_downstream_region_local)

# Precompute CdGM-style stratified productivity draws
println("Generating CdGM-style stratified productivity draws...")
u_draws_local, sample_weights_local = generate_stratified_draws(N_rho)
@everywhere const U_DRAWS = $u_draws_local
@everywhere const SAMPLE_WEIGHTS = $sample_weights_local
println("  N_rho: $(length(u_draws_local)) quantiles")
println("  Weight range: [$(minimum(sample_weights_local)), $(maximum(sample_weights_local))]")


############## LOGGING SETUP ##############

log_path = joinpath(output_folder, "optimization.log")
log_io = open(log_path, "w")

function log_msg(msg; also_print=true)
    timestamp = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
    line = "[$timestamp] $msg"
    println(log_io, line)
    flush(log_io)
    if also_print
        println(line)
    end
end

function log_param_summary(params, label="")
    beta, Omega_L, Omega_s, A, T = unpack_params(params)
    log_msg("  $(label)Parameter summary:")
    log_msg("    beta = $(round.(beta, digits=6))")
    log_msg("    Omega_L = $(round(Omega_L, digits=6))")
    log_msg("    Omega_s range = [$(round(minimum(Omega_s), digits=6)), $(round(maximum(Omega_s), digits=6))]")
    log_msg("    A range = [$(round(minimum(A), digits=6)), $(round(maximum(A), digits=6))]")
    log_msg("    T range = [$(round(minimum(T), digits=6)), $(round(maximum(T), digits=6))]")
end

function log_moment_fit(params)
    try
        loss, moments = full_SMM(params; u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS)
        log_msg("  Moment fit:")
        log_msg("    Total loss: $(round(loss[1], digits=8))")
        log_msg("    Labor share: emp=$(round(agg_labor_share, digits=4)) sim=$(round(moments[1][1], digits=4))")
        log_msg("    Reg coefs: emp=$(round.(reg_coef, digits=4)) sim=$(round.(moments[4], digits=4))")
        sim_pi = moments[5]
        pi_rmse = sqrt(mean((emp_pi_r .- sim_pi).^2))
        log_msg("    pi_r RMSE: $(round(pi_rmse, digits=6))")
    catch e
        log_msg("  Moment fit: FAILED ($e)")
    end
end

log_msg("=" ^ 70)
log_msg("SMM OPTIMIZATION WITH NELDER-MEAD (CdGM-style)")
log_msg("=" ^ 70)
log_msg("Industry: $industry")
log_msg("N_beta: $N_beta, N_moments: $N_moments")
log_msg("Workers: $available")
log_msg("Weight matrix: Identity")
log_msg("S=$S_, R=$R_, R_downstream=$R_downstream, N_rho=$N_rho")


############## NM OBJECTIVE FUNCTION ##############

@everywhere function nm_objective(params)
    # Enforce beta monotonicity
    if !issorted(params[1:N_beta])
        return 1e10
    end
    # Enforce positivity
    if any(params .< 0)
        return 1e10
    end
    # Enforce Omega_L in (0, 1)
    if params[N_beta + 1] <= 0 || params[N_beta + 1] >= 1
        return 1e10
    end
    try
        loss, _ = full_SMM(params; u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS)
        return loss[1]
    catch
        return 1e10
    end
end


############## STAGE 0: LHS INITIAL BETA SEARCH ##############

log_msg("")
log_msg("=" ^ 70)
log_msg("STAGE 0: LHS initial beta search")
log_msg("=" ^ 70)

t0_stage0 = time()

N_LHS_SAMPLES = 1500
log_msg("Generating $N_LHS_SAMPLES LHS beta samples...")
lhs_betas = generate_lhs_beta(N_LHS_SAMPLES, N_beta, 0.00005, 100.0)

# Build full parameter vectors with analytical initial guesses
A = copy(emp_pi_r_full) .^ (1 / abs(epsilon)) .* regional_wages[N_downstream_per_region .!= 0]
A ./= sum(A)
T_init_nonzero = vec(T_rs_init)[T_mask_local] .+ 0.1  # Only non-zero T values
init_other = vcat([agg_labor_share], agg_industry_share, A, T_init_nonzero)
expanding_beta = [vcat(beta, init_other) for beta in lhs_betas]

log_msg("Evaluating $(length(expanding_beta)) LHS beta samples in parallel...")
results_ = @showprogress "Stage 0 LHS..." pmap(p -> parallel_SMM_safe(p; u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS), expanding_beta)

# Find beta closest to empirical regression coefficients
n_success = count(r -> r !== nothing, results_)
log_msg("  $n_success / $(length(results_)) evaluations succeeded")

reg_coefs_sim = [r !== nothing ? r[2][4] : fill(NaN, N_beta) for r in results_]
reg_distances = [any(isnan.(rc)) ? Inf : sum((reg_coef .- rc) .^ 2) for rc in reg_coefs_sim]
best_idx = argmin(reg_distances)
init_beta = expanding_beta[best_idx][1:N_beta]

log_msg("Best initial beta: $(round.(init_beta, digits=6))")
log_msg("Matched reg coefs: $(round.(reg_coefs_sim[best_idx], digits=6))")
log_msg("Empirical reg coefs: $(round.(reg_coef, digits=6))")

# Also find the overall best loss from Stage 0
losses_stage0 = [r !== nothing ? r[1][1] : Inf for r in results_]
best_loss_idx = argmin(losses_stage0)
log_msg("Best overall loss in Stage 0: $(round(losses_stage0[best_loss_idx], digits=8))")

# Use the parameter set with best regression match as starting point
x0 = copy(expanding_beta[best_idx])

log_msg("Stage 0 completed in $(round(time() - t0_stage0, digits=1))s")
log_param_summary(x0, "Initial ")

# Save Stage 0 checkpoint
mkpath(joinpath(output_folder, "stage0"))
NPZ.npzwrite(joinpath(output_folder, "stage0", "best_params.npy"), reshape(x0, :, 1))
generate_report(output_folder, "stage0", 1, nothing, x0, "";
                u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS)
log_msg("Stage 0 checkpoint saved to $(joinpath(output_folder, "stage0"))")


############## STEP 1: COARSE NELDER-MEAD (20 starts) ##############

log_msg("")
log_msg("=" ^ 70)
log_msg("STEP 1: Coarse Nelder-Mead optimization")
log_msg("=" ^ 70)

N_STARTS_STEP1 = 20
MAX_ITER_STEP1 = 5000
F_RELTOL_STEP1 = 1e-6

log_msg("Starting points: $N_STARTS_STEP1")
log_msg("Max iterations per start: $MAX_ITER_STEP1")
log_msg("Function tolerance: $F_RELTOL_STEP1")

# Build bounds around x0
d = length(x0)
lb = copy(x0) .* 0.1
ub = copy(x0) .* 10.0

# Special bounds for Omega_L (must be in (0, 1))
lb[N_beta + 1] = 0.001
ub[N_beta + 1] = 0.999

# Ensure lb < ub for all parameters
for i in 1:d
    if lb[i] >= ub[i]
        lb[i] = min(x0[i] * 0.1, x0[i] - abs(x0[i]) * 0.1)
        ub[i] = max(x0[i] * 10.0, x0[i] + abs(x0[i]) * 10.0)
    end
    # Ensure positive
    lb[i] = max(lb[i], 1e-10)
end

# Generate starting points: x0 + LHS-based perturbations
starts = [copy(x0)]
raw_lhs = QuasiMonteCarlo.sample(N_STARTS_STEP1 - 1, lb, ub, LatinHypercubeSample())
for i in 1:(N_STARTS_STEP1 - 1)
    x_start = 0.5 .* x0 .+ 0.5 .* raw_lhs[:, i]  # blend with x0
    x_start = clamp.(x_start, lb, ub)
    push!(starts, x_start)
end

log_msg("Launching $N_STARTS_STEP1 parallel Nelder-Mead runs...")
t0_step1 = time()

step1_results = @showprogress "Step 1 NM..." pmap(starts) do x_init
    try
        r = optimize(nm_objective, lb, ub, x_init,
                     Fminbox(NelderMead()),
                     Optim.Options(iterations=MAX_ITER_STEP1, f_reltol=F_RELTOL_STEP1, show_trace=false))
        return (minimizer=Optim.minimizer(r), minimum=Optim.minimum(r),
                converged=Optim.converged(r), iterations=Optim.iterations(r))
    catch e
        return nothing
    end
end

valid_step1 = filter(!isnothing, step1_results)
log_msg("  $(length(valid_step1)) / $N_STARTS_STEP1 runs succeeded")

if isempty(valid_step1)
    log_msg("ERROR: All Step 1 runs failed. Using x0 as fallback.")
    best_params = copy(x0)
    best_loss = nm_objective(x0)
else
    # Sort by objective value
    sorted_step1 = sort(valid_step1, by=r -> r.minimum)
    best_step1 = sorted_step1[1]

    log_msg("Step 1 results summary:")
    log_msg("  Best loss:   $(round(sorted_step1[1].minimum, digits=8))")
    log_msg("  Median loss: $(round(sorted_step1[div(length(sorted_step1), 2) + 1].minimum, digits=8))")
    log_msg("  Worst loss:  $(round(sorted_step1[end].minimum, digits=8))")

    for (i, r) in enumerate(sorted_step1[1:min(5, length(sorted_step1))])
        log_msg("  Run $i: loss=$(round(r.minimum, digits=8)), converged=$(r.converged), iters=$(r.iterations)")
    end

    best_params = copy(best_step1.minimizer)
    best_params[1:N_beta] = sort(best_params[1:N_beta])  # enforce monotonicity
    best_loss = best_step1.minimum
end

elapsed_step1 = round(time() - t0_step1, digits=1)
log_msg("Step 1 completed in $(elapsed_step1)s")
log_param_summary(best_params, "Step 1 best ")
log_moment_fit(best_params)

# Save Step 1 checkpoint
mkpath(joinpath(output_folder, "step1"))
NPZ.npzwrite(joinpath(output_folder, "step1", "best_params.npy"), reshape(best_params, :, 1))
generate_report(output_folder, "step1", 1, nothing, best_params, "";
                u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS)
log_msg("Step 1 checkpoint saved to $(joinpath(output_folder, "step1"))")


############## STEP 2: REFINED NELDER-MEAD (top 15 solutions) ##############

log_msg("")
log_msg("=" ^ 70)
log_msg("STEP 2: Refined Nelder-Mead optimization")
log_msg("=" ^ 70)

MAX_ITER_STEP2 = 10000
F_RELTOL_STEP2 = 1e-10

if isempty(valid_step1)
    top_starts = [copy(best_params)]
else
    top_k = min(15, length(sorted_step1))
    top_starts = [r.minimizer for r in sorted_step1[1:top_k]]
    log_msg("Re-optimizing top $top_k solutions from Step 1")
end

log_msg("Max iterations per start: $MAX_ITER_STEP2")
log_msg("Function tolerance: $F_RELTOL_STEP2")

# Tighter bounds around best solution
lb2 = best_params .* 0.5
ub2 = best_params .* 2.0
lb2[N_beta + 1] = max(0.001, best_params[N_beta + 1] * 0.5)
ub2[N_beta + 1] = min(0.999, best_params[N_beta + 1] * 2.0)

# Ensure lb < ub and positive
for i in 1:d
    lb2[i] = max(lb2[i], 1e-10)
    if lb2[i] >= ub2[i]
        lb2[i] = best_params[i] * 0.5
        ub2[i] = best_params[i] * 2.0
    end
end

log_msg("Launching $(length(top_starts)) parallel refined NM runs...")
t0_step2 = time()

step2_results = @showprogress "Step 2 NM..." pmap(top_starts) do x_init
    x_init = clamp.(x_init, lb2, ub2)
    try
        r = optimize(nm_objective, lb2, ub2, x_init,
                     Fminbox(NelderMead()),
                     Optim.Options(iterations=MAX_ITER_STEP2, f_reltol=F_RELTOL_STEP2, show_trace=false))
        return (minimizer=Optim.minimizer(r), minimum=Optim.minimum(r),
                converged=Optim.converged(r), iterations=Optim.iterations(r))
    catch
        return nothing
    end
end

valid_step2 = filter(!isnothing, step2_results)
log_msg("  $(length(valid_step2)) / $(length(top_starts)) runs succeeded")

if !isempty(valid_step2)
    sorted_step2 = sort(valid_step2, by=r -> r.minimum)
    best_step2 = sorted_step2[1]

    log_msg("Step 2 results summary:")
    log_msg("  Best loss:   $(round(sorted_step2[1].minimum, digits=10))")
    log_msg("  Median loss: $(round(sorted_step2[div(length(sorted_step2), 2) + 1].minimum, digits=10))")
    log_msg("  Worst loss:  $(round(sorted_step2[end].minimum, digits=10))")

    for (i, r) in enumerate(sorted_step2[1:min(5, length(sorted_step2))])
        log_msg("  Run $i: loss=$(round(r.minimum, digits=10)), converged=$(r.converged), iters=$(r.iterations)")
    end

    # Only update if Step 2 found something better
    if best_step2.minimum < best_loss
        best_params = copy(best_step2.minimizer)
        best_params[1:N_beta] = sort(best_params[1:N_beta])
        best_loss = best_step2.minimum
        log_msg("Step 2 improved on Step 1: $(round(best_loss, digits=10))")
    else
        log_msg("Step 2 did NOT improve on Step 1. Keeping Step 1 solution.")
    end
else
    log_msg("WARNING: All Step 2 runs failed. Keeping Step 1 solution.")
end

elapsed_step2 = round(time() - t0_step2, digits=1)
log_msg("Step 2 completed in $(elapsed_step2)s")
log_param_summary(best_params, "Final ")
log_moment_fit(best_params)

# Save Step 2 / final results
mkpath(joinpath(output_folder, "step2"))
NPZ.npzwrite(joinpath(output_folder, "step2", "best_params.npy"), reshape(best_params, :, 1))
generate_report(output_folder, "step2", 1, nothing, best_params, "";
                u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS)
log_msg("Step 2 checkpoint saved to $(joinpath(output_folder, "step2"))")

# Also save as final
mkpath(joinpath(output_folder, "final"))
NPZ.npzwrite(joinpath(output_folder, "final", "best_params.npy"), reshape(best_params, :, 1))
generate_report(output_folder, "final", 1, nothing, best_params, "";
                u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS)


############## POST-HOC ANALYSIS ##############

log_msg("")
log_msg("=" ^ 70)
log_msg("POST-HOC ANALYSIS")
log_msg("=" ^ 70)

# Run unified validation (all three models in one panel)
log_msg("Running unified validation (all three shock models)...")
try
    results_unified = validate_table2_all_models(best_params, industry, T_periods=36, time_fe_mode="resample")
    Parquet.write_parquet(joinpath(output_folder, "simulated_panel_unified.parquet"), results_unified["panel_df"])
    Parquet.write_parquet(joinpath(output_folder, "regional_sales_unified.parquet"), results_unified["regional_sales_df"])
    log_msg("Validation complete. Saved simulated_panel_unified.parquet")
catch e
    log_msg("WARNING: Validation failed: $e")
end

# Save network outputs (using sparse firm-level data)
log_msg("Computing network outputs...")
network = solve_network(best_params, return_firm_level=true, u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS)
folder = output_folder
mu = network.mu
Y_r = network.Y_r

# Reconstruct X_lrs from sparse COO for w_srd_r computation
X_lrs_sparse = zeros(R, R, S)
n_entries = length(network.firm_exp_rho)
for i in 1:n_entries
    g = network.firm_exp_g[i]
    l = GOOD_R[g]
    s_idx = network.firm_exp_s[i]
    r = network.firm_exp_r[i]
    X_lrs_sparse[l, r, s_idx] += network.firm_exp_val[i] * mu * Y_r[r]
end

w_srd_r = zeros(S, R, R)
for s in 1:S, r_prime in 1:R
    total_downstream_sales = sum(X_lrs_sparse[r_prime, :, s])
    if total_downstream_sales > 1e-10
        for r in 1:R
            w_srd_r[s, r_prime, r] = X_lrs_sparse[r_prime, r, s] / total_downstream_sales
        end
    end
end

npzwrite(joinpath(folder, "w_srd_r.npy"), w_srd_r)

# Build suppliers indicator from flat linkages
suppliers = zeros(Bool, S, N_rho, R)
for g in 1:n_good
    s = GOOD_S[g]
    r = GOOD_R[g]
    for rho in 1:N_rho
        if network.linkages_flat[rho, g] > 0
            suppliers[s, rho, r] = true
        end
    end
end
npzwrite(joinpath(folder, "suppliers.npy"), suppliers)

# Build suppliers DataFrame from sparse COO firm-level data (only non-zero entries)
sirens = Int[]
sectors = Int[]
ze2010 = Int[]
ze2010_downstream = Int[]
share = Float64[]
downstream_purchase = Float64[]
intermediate_derivative = Float64[]
productivity = Float64[]

siren_map = Dict{Tuple{Int,Int,Int}, Int}()
siren_counter = 0
for g in 1:n_good
    l = GOOD_R[g]
    s = GOOD_S[g]
    for rho in 1:N_rho
        key = (l, s, rho)
        if !haskey(siren_map, key)
            siren_counter += 1
            siren_map[key] = siren_counter
        end
    end
end

for i in 1:n_entries
    rho = network.firm_exp_rho[i]
    s = network.firm_exp_s[i]
    g = network.firm_exp_g[i]
    l = GOOD_R[g]
    r = network.firm_exp_r[i]
    siren_id = siren_map[(l, s, rho)]

    push!(sirens, siren_id)
    push!(sectors, s)
    push!(ze2010, l)
    push!(ze2010_downstream, r)
    push!(share, network.firm_exp_val[i])
    push!(downstream_purchase, Y_r[r] * mu)
    push!(intermediate_derivative, network.firm_deriv_val[i])
    push!(productivity, network.z_flat[rho, g])
end

df = DataFrame(
    SIREN = sirens,
    A129 = sectors,
    ze2010 = ze2010,
    ze2010_downstream = ze2010_downstream,
    share = share,
    downstream_purchase = downstream_purchase,
    intermediate_derivative = intermediate_derivative,
    productivity = productivity
)

Parquet.write_parquet(joinpath(folder, "suppliers.parquet"), df)

log_msg("")
log_msg("=" ^ 70)
log_msg("OPTIMIZATION COMPLETE")
log_msg("=" ^ 70)
log_msg("Final loss: $(round(best_loss, digits=10))")
log_msg("Total time: $(round(time() - t0_stage0, digits=1))s")
log_msg("Output folder: $output_folder")
log_msg("Output files:")
log_msg("  - optimization.log")
log_msg("  - stage0/best_params.npy, dashboard.png, report.txt")
log_msg("  - step1/best_params.npy, dashboard.png, report.txt")
log_msg("  - step2/best_params.npy, dashboard.png, report.txt")
log_msg("  - final/best_params.npy, dashboard.png, report.txt")
log_msg("  - simulated_panel_unified.parquet")
log_msg("  - suppliers.parquet, suppliers.npy, w_srd_r.npy")

close(log_io)
