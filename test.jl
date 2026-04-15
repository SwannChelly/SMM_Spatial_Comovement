##### Main with PSO Optimization #####
# Author: Swann Chelly (Modified for PSO)
# This replaces Halton grid search with Particle Swarm Optimization
# ps aux | grep '[j]ulia' | awk '{print $2}' | xargs kill -9
# nohup julia SMM_Spatial_Comovement/main_pso.jl > reporting_aero/logs.log 2>&1 
using Distributed
using Dates

# Add workers
available = 50#Sys.CPU_THREADS - nprocs()
println("Using "*string(available)*" workers")
addprocs(max(available-1, 0)) # Always leave one core for other tests. 
# Set seed on ALL processes (master + workers)
seed = 1234
using Random
Random.seed!(seed)
@everywhere using Random
@everywhere Random.seed!($seed)

@everywhere using NPZ
@everywhere using QuasiMonteCarlo
@everywhere using StatsPlots
@everywhere using DataFrames 
@everywhere using Distributions
@everywhere using Plots
@everywhere using CSV
@everywhere using Optim
@everywhere using Statistics
@everywhere using HaltonSequences
@everywhere using ProgressMeter
@everywhere using SharedArrays
@everywhere using Parquet
using LinearAlgebra
using Statistics, Printf
using StatsBase

@everywhere include("model_CP.jl")
@everywhere include("tools.jl")
@everywhere include("pso_integration.jl")  # PSO functions
@everywhere include("run_untargeted_validation.jl")

############## Load Parameters #################
industry = length(ARGS) >= 1 ? ARGS[1] : "auto"  # Default to "aero" if no argument
n_coef = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4  # Default to 4 coefficients
resume = length(ARGS) >= 3 && ARGS[3] == "resume"
if !(n_coef in [4, 5])
    error("n_coef must be 4 or 5, got: $n_coef")
end
println("Using $n_coef regression coefficients (reg_coef_$n_coef.npy)")

input_folder = "./baseline_"*industry
output_folder = "./reporting_"*industry
mkpath(output_folder) 

coefs = CSV.read(joinpath(input_folder,"stats.csv"), DataFrame)
distances_local = NPZ.npzread(joinpath(input_folder, "distances.npy"))
filter_N_upstream_local = NPZ.npzread(joinpath(input_folder,"filter_N_upstream.npy"))
w_rs_local = NPZ.npzread(joinpath(input_folder, "w_rs.npy"))  # (R,) - regional wages, not (S,R)
regional_wages_local = NPZ.npzread(joinpath(input_folder, "regional_wages.npy"))
N_downstream_per_region_local = NPZ.npzread(joinpath(input_folder,"N_downstream_per_region.npy"))
agg_industry_share_local = NPZ.npzread(joinpath(input_folder,"input_share.npy"))
domestic_share_local = NPZ.npzread(joinpath(input_folder,"domestic_share.npy"))
X_rs_local = NPZ.npzread(joinpath(input_folder,"X_rs.npy")) # Number of upstream per region. 
N_rs_local = NPZ.npzread(joinpath(input_folder,"N_rs.npy")) # Number of upstream per region. 

S_,R_ = size(filter_N_upstream_local)
@everywhere const S = $(S_)
@everywhere const R = $(R_)

R_ = size(N_downstream_per_region_local[N_downstream_per_region_local.!=0])[1]
@everywhere const R_downstream = $(R_)
@everywhere const agg_industry_share = $(agg_industry_share_local)
@everywhere const agg_labor_share = $(coefs[2,"value"])
@everywhere const domestic_share = $(domestic_share_local)
@everywhere regional_wages = $(regional_wages_local)
@everywhere const distances = $(distances_local)
@everywhere const N_downstream_per_region = $(N_downstream_per_region_local)     
@everywhere const w_rs = $(w_rs_local)
@everywhere const filter_N_upstream = $(filter_N_upstream_local)
@everywhere const N_rho = $(100)
@everywhere const epsilon = $(coefs[1,"value"])
@everywhere const lambda = $(0.5)
@everywhere const nu = $(0.2)
@everywhere const nu_s = $(ones(S).*2.5) 
@everywhere const theta = $(1.768) 
@everywhere const delta_r = $(ones(R))
@everywhere const Weight_matrix = $(nothing)



#if industry == "aero"
#    @everywhere const T_rs_init = $(X_rs_local)
#elseif industry == "auto"
#    @everywhere const T_rs_init = $(X_rs_local)# N_rs_local
#end

# Mask for non-zero T_rs: only optimize T where gamma_ls > 0
T_mask_local = vec(X_rs_local) .> 0
@everywhere const T_MASK = $T_mask_local
T_mask_moment_local = vec(permutedims(X_rs_local)) .> 0
@everywhere const T_MASK_MOMENT = $T_mask_moment_local

println("T_MASK: $(sum(T_mask_local)) / $(length(T_mask_local)) non-zero T parameters")

# Precompute flat index mappings for good (s,r) pairs
R_full = size(filter_N_upstream_local, 2)  # original R before R_ gets reassigned
good_indices_local = findall(reshape(T_mask_local, S_, R_full))
n_good_local = length(good_indices_local)
GOOD_S_local = [ci[1] for ci in good_indices_local]
GOOD_R_local = [ci[2] for ci in good_indices_local]

# Per-sector grouping: for sector s, which indices in the good-pair list belong to it
SECTOR_GOOD_INDICES_local = [findall(GOOD_S_local .== s) for s in 1:S_]
SECTOR_GOOD_REGIONS_local = [GOOD_R_local[idx] for idx in SECTOR_GOOD_INDICES_local]

# Reverse map: (s, r) -> good pair index (0 if inactive)
SR_TO_GOOD_local = zeros(Int, S_, R_full)
for (g, ci) in enumerate(good_indices_local)
    SR_TO_GOOD_local[ci[1], ci[2]] = g
end

# Flat w_rs for good pairs only (indexed by region only, not sector-region)
W_RS_FLAT_local = [w_rs_local[GOOD_R_local[g]] for g in 1:n_good_local]

@everywhere const n_good = $n_good_local
@everywhere const GOOD_S = $GOOD_S_local
@everywhere const GOOD_R = $GOOD_R_local
@everywhere const SECTOR_GOOD_INDICES = $SECTOR_GOOD_INDICES_local
@everywhere const SECTOR_GOOD_REGIONS = $SECTOR_GOOD_REGIONS_local
@everywhere const SR_TO_GOOD = $SR_TO_GOOD_local
@everywhere const W_RS_FLAT = $W_RS_FLAT_local
println("  n_good: $n_good_local good (sector, region) pairs")

# Load empirical moments
#@everywhere const emp_pi_r_labor = $(NPZ.npzread(joinpath(input_folder,"emp_pi_r.npy")))
@everywhere const emp_gamma_ls = $(permutedims(NPZ.npzread(joinpath(input_folder,"emp_gamma_ls.npy"))))
X_dr_local = CSV.read(joinpath(input_folder,"X_dr.csv"), DataFrame).X_dr
X_dr_local = X_dr_local[N_downstream_per_region.!=0]
emp_pi_r_local = X_dr_local./sum(X_dr_local)
@everywhere const emp_pi_r_full = $(emp_pi_r_local)
@everywhere const emp_pi_r = $(emp_pi_r_local)  # Full vector; MOMENT_MASK handles [2:end]
@everywhere const reg_coef = $(NPZ.npzread(joinpath(input_folder,"reg_coef_"*string(n_coef)*".npy")))
@everywhere const N_beta = $(length(NPZ.npzread(joinpath(input_folder,"reg_coef_"*string(n_coef)*".npy"))))

# Build full empirical moments (no [2:end] drops — MOMENT_MASK handles that)
empirical_moments_local = [[agg_labor_share], vec(agg_industry_share), emp_gamma_ls, reg_coef, emp_pi_r]
empirical_moments_local = vcat([vec(empirical_moments_local[i]) for i in 1:(length(empirical_moments_local))]...)



############# GRAVITY-BASED T INITIALIZATION (tau = 1) ##############

println("Building gravity-consistent T_init (tau = 1, max-normalized)...")

T_gravity = zeros(S, R)

for s in 1:S
    idxs = SECTOR_GOOD_INDICES[s]  # indices in flat good list
    
    # Compute raw T_sl ∝ gamma_ls * w_l^theta
    for g in idxs
        l = GOOD_R[g]  # upstream region
        
        gamma_val = emp_gamma_ls[l,s]
        w_l = w_rs[l]
        
        T_gravity[s, l] = max(gamma_val * (w_l^theta), 1e-12)
    end
    
    # Max normalization within sector
    sector_vals = T_gravity[s, GOOD_R[idxs]]
    max_val = maximum(sector_vals)
    
    if max_val > 0
        T_gravity[s, GOOD_R[idxs]] ./= max_val
    end
end
@everywhere const T_rs_init = $(T_gravity)
#@everywhere const T_rs_init = $(X_rs_local)

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
# Step A: drop all inactive (zero) gamma_ls entries.
for idx in 1:(S_ * R_full)
    if !T_mask_moment_local[idx]
        moment_mask_local[n_labor + n_industry + idx] = false
    end
end

# Step B: drop the first *active* entry per sector (sum-to-1 redundancy).
# Must come after Step A so we identify the first active r correctly.
for s in 1:S_
    sector_start = (s - 1) * R_full + 1
    sector_end   = s * R_full
    sector_slice = T_mask_moment_local[sector_start:sector_end]
    active_positions = findall(sector_slice)   # indices within this sector's R_full block
    if !isempty(active_positions)
        first_active = active_positions[1]
        moment_mask_local[n_labor + n_industry + (s - 1) * R_full + first_active] = false
    end
end

moment_mask_local[n_labor + n_industry + n_gamma + n_reg + 1] = false  # first pi_r

# Apply mask to empirical moments
empirical_moments_local = reshape(empirical_moments_local[moment_mask_local], 1, sum(moment_mask_local))
N_moments = sum(moment_mask_local)

# Load pre-built weight vector from Python (length = sum(MOMENT_MASK)).
weight_vector_local = NPZ.npzread(joinpath(input_folder, "weight_vector.npy"))

@assert length(weight_vector_local) == sum(moment_mask_local) """
Weight vector length $(length(weight_vector_local)) != N_moments $(sum(moment_mask_local)).
"""

Weight_matrix_custom_local = Diagonal(weight_vector_local)#I(length(weight_vector_local))#
@everywhere const Weight_matrix_custom = $Weight_matrix_custom_local


@everywhere const MOMENT_MASK = $moment_mask_local
@everywhere const empirical_moments = $(empirical_moments_local)
@everywhere const Weight_matrix_custom = $Weight_matrix_custom_local
@everywhere const K_max = $(50)



# Block ranges for loss decomposition (must come after MOMENT_MASK)
BLOCK_RANGES_local = compute_block_ranges(
    n_labor, n_industry, n_gamma, n_reg, n_pi, moment_mask_local
)
@everywhere const BLOCK_RANGES = $BLOCK_RANGES_local
@everywhere const BLOCK_NAMES = ("labor", "industry", "gamma_ls", "reg_coef", "pi_r")


# Precompute CdGM-style stratified productivity draws
println("Generating CdGM-style stratified productivity draws...")
u_draws_local, sample_weights_local = generate_stratified_draws(N_rho, n_good_local)
@everywhere const U_DRAWS = $u_draws_local
@everywhere const SAMPLE_WEIGHTS = $sample_weights_local
println("  U_DRAWS shape: $(size(u_draws_local)) (N_rho × n_good)")
println("  Weight range: [$(minimum(sample_weights_local)), $(maximum(sample_weights_local))]")

# Downstream region indices (maps r_d ∈ 1:R_downstream → r ∈ 1:R)
downstream_regions_local = findall(N_downstream_per_region_local .> 0)
@everywhere const DOWNSTREAM_REGIONS = $(downstream_regions_local)

# R × R_downstream distance subset
distances_downstream_local = distances_local[:, downstream_regions_local]

# Distance bins (R × R_downstream)
DistBin_local = Array{Int}(undef, R, R_downstream)
for i in 1:R, j in 1:R_downstream
    DistBin_local[i,j] = distance_bin(distances_downstream_local[i,j])
end
@everywhere const DistBin = $(DistBin_local)

# Closest downstream plant: simpler computation now all columns are downstream
closest_plant_dist_local = vec(minimum(distances_downstream_local, dims=2))
closest_downstream_region_local = vec(getindex.(argmin(distances_downstream_local, dims=2), 2))
@everywhere const CLOSEST_PLANT_DIST = $(closest_plant_dist_local)
@everywhere const CLOSEST_DOWNSTREAM_REGION = $(closest_downstream_region_local)

# PSO Configuration
N_PARTICLES = 100   # Use all available cores except one 
MAX_ITER_INITIAL = 200      # Iterations for initial full optimization
MAX_ITER_STAGE = 50         # Iterations for each refinement stage
method = "log"
max_loop = 50
full_run = true
length_range_beta = 40 # Normal is 50
BETA_SEARCH_METHOD = "log_grid"  # Options: "lhs" (default), "log_grid" (old systematic grid)
BETA_SELECTION_CRITERION = "score"  # Options: "reg_coef" (default), "score"

# Reporting configuration
REPORT_EVERY = 100  # Run reporting every X epochs (set to nothing for only at the end)


############## MAIN OPTIMIZATION ##############

# Resume state variables (set below if resuming)
resume_loop = 1
resume_substage = 1

if resume
    ############## RESUME MODE ##############
    println("\n" * "="^70)
    println("RESUME MODE: Scanning $output_folder for last completed stage...")
    println("="^70)

    resume_state = find_resume_state(output_folder)
    best_params = NPZ.npzread(joinpath(resume_state.last_folder, "best_params.npy"))
    if ndims(best_params) > 1
        best_params = best_params[:, 1]
    end
    best_fitness = NaN  # Will be updated on next PSO call

    resume_loop = resume_state.resume_loop
    resume_substage = resume_state.resume_substage
    stage = resume_state.last_stage

    println("  Last completed: $(resume_state.last_folder)")
    println("  Resuming at loop $resume_loop, sub-stage $resume_substage (global stage $stage)")
    println("="^70)
end


############# INITIAL SEARCH FOR GOOD BETA ##############

println("\n" * "="^70)
println("Method $method")
println("STAGE 0: Finding good initial beta values")
println("Beta search method: $BETA_SEARCH_METHOD")
println("Beta selection criterion: $BETA_SELECTION_CRITERION")
println("="^70)


beta_min_informed = 0.001#minimum(exp.(-reg_coef ./ theta) .- 1) * 0.3
beta_max_informed = 2#maximum(exp.(-reg_coef ./ theta) .- 1) * 3.0
#beta_min_informed = max(beta_min_informed, 1e-4)
print(beta_min_informed) 
print(beta_max_informed)
# Generate beta candidates using selected method
if BETA_SEARCH_METHOD == "log_grid"
    beta_candidates = generate_initial_betas("log_grid", N_beta, beta_min_informed, beta_max_informed;
                                                log_grid_length=length_range_beta)
    println("Generated $(length(beta_candidates)) log-grid beta combinations")
elseif BETA_SEARCH_METHOD == "lhs"
    N_LHS_SAMPLES = 20000
    beta_candidates = generate_initial_betas("lhs", N_beta, beta_min_informed, beta_max_informed;
                                                lhs_n_samples=N_LHS_SAMPLES)
    println("Generated $(length(beta_candidates)) LHS beta samples")
else
    error("Unknown BETA_SEARCH_METHOD: $BETA_SEARCH_METHOD")
end

# Use initial guess for other parameters
A = copy(emp_pi_r_full).^(1/abs(epsilon)).*regional_wages[N_downstream_per_region .!= 0]  # analytical inversion
A ./= sum(A)

# Extract only active entries (respect sparsity mask)
T_init_nonzero = vec(T_rs_init)[T_mask_local]

println("T_init stats:")
println("  min = $(minimum(T_init_nonzero))")
println("  max = $(maximum(T_init_nonzero))")
println("  mean = $(mean(T_init_nonzero))")


init_other = vcat([agg_labor_share], agg_industry_share, A, T_init_nonzero)
expanding_beta = [vcat(beta, init_other) for beta in beta_candidates]

println("Evaluating $(length(expanding_beta)) beta combinations in parallel...")
results_ = pmap(p -> parallel_SMM_safe(p; u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS), expanding_beta)

# Find best beta using selected criterion
if BETA_SELECTION_CRITERION == "reg_coef"
    reg_coefs_sim = [r !== nothing ? r[2][4] : fill(NaN, N_beta) for r in results_]
    #reg_coefs_sim = filter(row -> all(diff(row) .< 0), reg_coefs_sim)
    reg_distances = [sum((reg_coef .- rc).^2) for rc in reg_coefs_sim]
    best_idx = argmin(reg_distances)
elseif BETA_SELECTION_CRITERION == "score"
    scores = [r !== nothing ? r[1][1] : Inf for r in results_]
    best_idx = argmin(scores)
else
    error("Unknown BETA_SELECTION_CRITERION: $BETA_SELECTION_CRITERION")
end
init_beta = beta_candidates[best_idx]

println("Best initial beta: ", round.(init_beta, digits=6))
println("Related regression coefficients are: ", round.([r !== nothing ? r[2][4] : fill(NaN, N_beta) for r in results_][best_idx], digits=6))



test_beta = [0.4, 0.15315, 0.5, 0.5]
params = vcat(test_beta, init_other)
full_SMM(params; u_draws = U_DRAWS, sample_weights = SAMPLE_WEIGHTS)[2][4]

############## PSO-BASED OPTIMIZATION ##############


println("\n" * "="^70)
println("STAGE 1: Initial PSO optimization (all parameters)")
println("="^70)
println("Particles: $N_PARTICLES")
println("Iterations: $MAX_ITER_INITIAL")


# Stage 1: Optimize all parameters starting from init_beta
best_params, best_fitness, history = train_stage_pso(
    N_PARTICLES,
    MAX_ITER_INITIAL,
    init_beta = init_beta,
    variable_list = nothing,  # Optimize all parameters
    last_stage_folder = nothing,
    alpha = 0.5,
    second_stage = false,
    method = method,
    u_draws = U_DRAWS,
    sample_weights = SAMPLE_WEIGHTS
)

# Save results
stage = 0
mkpath(joinpath(output_folder, string(stage)))
NPZ.npzwrite(joinpath(output_folder, string(stage), "best_params.npy"), reshape(best_params, :, 1))
NPZ.npzwrite(joinpath(output_folder, string(stage), "pso_history.npy"), Dict(
    "best_fitness" => history["best_fitness"],
    "mean_fitness" => history["mean_fitness"]
))
generate_report(output_folder, string(stage), 1, nothing, best_params, "";
                    u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS)
run_reporting(output_folder, 0; u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS)

println("\nStage $stage complete. Best fitness: $(round(best_fitness, digits=6))")


