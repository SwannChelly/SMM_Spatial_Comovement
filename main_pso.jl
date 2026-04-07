##### Main with PSO Optimization #####
# Author: Swann Chelly (Modified for PSO)
# This replaces Halton grid search with Particle Swarm Optimization
# ps aux | grep '[j]ulia' | awk '{print $2}' | xargs kill -9
# nohup julia SMM_Spatial_Comovement/main_pso.jl > reporting_aero/logs.log 2>&1 
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

# Add workers
available = 50#Sys.CPU_THREADS - nprocs()
println("Using "*string(available)*" workers")
addprocs(max(available-1, 0)) # Always leave one core for other tests. 

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

if industry == "aero"
    @everywhere const T_rs_init = $(X_rs_local)
elseif industry == "auto"
    @everywhere const T_rs_init = $(X_rs_local)# N_rs_local
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

# Build weight matrix: upweight reg_coef by 100
weights_full = ones(N_moments_full)
reg_start_full = n_labor + n_industry + n_gamma + 1
reg_end_full = reg_start_full + n_reg - 1
weights_full[reg_start_full:reg_end_full] .= 100
weights = weights_full[moment_mask_local]
Weight_matrix_custom_local = Diagonal(weights)

@everywhere const MOMENT_MASK = $moment_mask_local
@everywhere const empirical_moments = $(empirical_moments_local)
@everywhere const Weight_matrix_custom = $Weight_matrix_custom_local
@everywhere const K_max = $(50)

@everywhere include("model_CP.jl")
@everywhere include("tools.jl")
@everywhere include("pso_integration.jl")  # PSO functions
@everywhere include("run_untargeted_validation.jl")

# Precompute CdGM-style stratified productivity draws
println("Generating CdGM-style stratified productivity draws...")
u_draws_local, sample_weights_local = generate_stratified_draws(N_rho)
@everywhere const U_DRAWS = $u_draws_local
@everywhere const SAMPLE_WEIGHTS = $sample_weights_local
println("  N_rho: $(length(u_draws_local)) quantiles")
println("  Weight range: [$(minimum(sample_weights_local)), $(maximum(sample_weights_local))]")

# Distance bins
DistBin_local = Array{Int}(undef, R,R)
for i in 1:R, j in 1:R
    DistBin_local[i,j] = distance_bin(distances_local[i,j])
end
@everywhere const DistBin = $(DistBin_local)

# Precompute closest downstream plant distance and region (constants depending only on distances and N_downstream_per_region)
closest_plant_dist_local = vec(map(x -> distances_local[x[1], x[2]], argmin(1 ./ (1 ./ distances_local .* (N_downstream_per_region_local .> 0)'), dims=2)))
closest_downstream_region_local = vec(getindex.(argmin(1 ./ (1 ./ distances_local .* (N_downstream_per_region_local .> 0)'), dims=2), 2))
@everywhere const CLOSEST_PLANT_DIST = $(closest_plant_dist_local)
@everywhere const CLOSEST_DOWNSTREAM_REGION = $(closest_downstream_region_local)

# PSO Configuration
N_PARTICLES = 100   # Use all available cores except one 
MAX_ITER_INITIAL = 200      # Iterations for initial full optimization
MAX_ITER_STAGE = 50         # Iterations for each refinement stage
method = "original"
max_loop = 50
full_run = true
length_range_beta = 20 # Normal is 50
BETA_SEARCH_METHOD = "lhs"  # Options: "lhs" (default), "log_grid" (old systematic grid)
BETA_SELECTION_CRITERION = "reg_coef"  # Options: "reg_coef" (default), "score"

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

if full_run
    if !resume
    ############# INITIAL SEARCH FOR GOOD BETA ##############

    println("\n" * "="^70)
    println("Method $method")
    println("STAGE 0: Finding good initial beta values")
    println("Beta search method: $BETA_SEARCH_METHOD")
    println("Beta selection criterion: $BETA_SELECTION_CRITERION")
    println("="^70)

    # Generate beta candidates using selected method
    if BETA_SEARCH_METHOD == "log_grid"
        beta_candidates = generate_initial_betas("log_grid", N_beta, 0.00005, 100.0;
                                                  log_grid_length=length_range_beta)
        println("Generated $(length(beta_candidates)) log-grid beta combinations")
    elseif BETA_SEARCH_METHOD == "lhs"
        N_LHS_SAMPLES = 1500
        beta_candidates = generate_initial_betas("lhs", N_beta, 0.00005, 10.0;
                                                  lhs_n_samples=N_LHS_SAMPLES)
        println("Generated $(length(beta_candidates)) LHS beta samples")
    else
        error("Unknown BETA_SEARCH_METHOD: $BETA_SEARCH_METHOD")
    end

    # Use initial guess for other parameters
    A = copy(emp_pi_r_full).^(1/abs(epsilon)).*regional_wages[N_downstream_per_region .!= 0]  # analytical inversion
    A ./= sum(A)

    T_init_nonzero = vec(T_rs_init)[T_mask_local] .+ 0.1  # Only non-zero T values
    init_other = vcat([agg_labor_share], agg_industry_share, A, T_init_nonzero)
    expanding_beta = [vcat(beta, init_other) for beta in beta_candidates]

    println("Evaluating $(length(expanding_beta)) beta combinations in parallel...")
    results_ = pmap(p -> parallel_SMM_safe(p; u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS), expanding_beta)

    # Find best beta using selected criterion
    if BETA_SELECTION_CRITERION == "reg_coef"
        reg_coefs_sim = [r !== nothing ? r[2][4] : fill(NaN, N_beta) for r in results_]
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
    run_reporting(output_folder, 0)

    println("\nStage $stage complete. Best fitness: $(round(best_fitness, digits=6))")

    end # if !resume

    if max_loop != nothing
        ############# MULTI-STAGE REFINEMENT ##############

        println("\n" * "="^70)
        println("Starting multi-stage PSO refinement (3 stages per loop)")
        println("="^70)
        loop_start = resume ? resume_loop : 1
        if !resume
            stage = 0
        end
        println("Starting at loop: $loop_start, stage: $stage")
        
        # Alpha controls search radius: starts tight, expands over time
        alpha_start, alpha_end = 0.3, 0.9
        
        for loop in loop_start:max_loop
            global stage
            global max_loop
            global best_params
            global best_fitness

            alpha = alpha_start + (loop - 1) * (alpha_end - alpha_start) / (max_loop - 1)
            
            println("\n" * "="^70)
            println("REFINEMENT LOOP $loop / $max_loop")
            println("="^70)
            println("Alpha (search radius): $alpha")
            
            past_loop_folder = loop == 1 ? output_folder : output_folder*"/epoch_"*string(loop-1)
            loop_folder = output_folder*"/epoch_"*string(loop)
            mkpath(loop_folder)
            
            # ═══════════════════════════════════════════════════════════════════
            # STAGE 1: Productivity (A_r) - MOST SENSITIVE
            # ═══════════════════════════════════════════════════════════════════
            # With ε = -16, errors in A_r are amplified 16x in π_r
            # Use tight bounds and log-space loss for π_r matching
            
            # Check if this sub-stage should be skipped (resume mode)
            skip_substage_1 = resume && loop == resume_loop && resume_substage > 1

            if skip_substage_1
                stage += 1
                println("  ⏭ Skipping Stage 1 (already completed)")
            else

            println("\n" * "-"^50)
            println("Loop $(loop) - Stage 1: PRODUCTIVITY (π_r matching)")
            println("-"^50)
            println("  Using tight bounds (alpha_A = $(round(0.7 + 0.2*alpha, digits=2)))")

            # Tighter alpha for productivity due to high sensitivity
            alpha_productivity = 0.7 + 0.2 * alpha  # Range: 0.7 to 0.9 (tight)
            #alpha_productivity = alpha
            best_params, best_fitness, history = train_stage_pso(
                N_PARTICLES,
                MAX_ITER_STAGE,
                variable_list = ["productivity"],
                last_stage_folder = joinpath(past_loop_folder, string(stage)),
                K = 1,
                alpha = alpha_productivity,
                second_stage = false,
                method = method,  # Log-space loss for π_r (handles concentration)
                u_draws = U_DRAWS,
                sample_weights = SAMPLE_WEIGHTS
            )

            stage += 1
            folder = joinpath(loop_folder, string(stage))
            mkpath(folder)
            NPZ.npzwrite(joinpath(folder, "best_params.npy"), reshape(best_params, :, 1))
            NPZ.npzwrite(joinpath(folder, "pso_history.npy"), Dict(
                "best_fitness" => history["best_fitness"],
                "mean_fitness" => history["mean_fitness"]
            ))
            generate_report(loop_folder, string(stage), 1, ["productivity"], best_params, string(alpha_productivity);
                             u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS)

            println("  ✓ Stage 1 complete. Fitness: $(round(best_fitness, digits=6))")

            end # skip_substage_1
            
            # ═══════════════════════════════════════════════════════════════════
            # STAGE 2: Spatial Structure (β, T) - MEDIUM SENSITIVITY
            # ═══════════════════════════════════════════════════════════════════
            # Trade costs affect regression coefficients
            # Fréchet scales affect sourcing shares γ_{ls}
            
            # Check if this sub-stage should be skipped (resume mode)
            skip_substage_2 = resume && loop == resume_loop && resume_substage > 2

            if skip_substage_2
                stage += 1
                println("  ⏭ Skipping Stage 2 (already completed)")
            else

            println("\n" * "-"^50)
            println("Loop $(loop) - Stage 2: SPATIAL STRUCTURE (β, T)")
            println("-"^50)
            println("  Sensitivity: MEDIUM")
            println("  Targets: regression coefficients, γ_{ls}")

            best_params, best_fitness, history = train_stage_pso(
                N_PARTICLES,
                MAX_ITER_STAGE,
                variable_list = ["beta", "T"],
                last_stage_folder = joinpath(loop_folder, string(stage)),
                K = 1,
                alpha = alpha,  # Standard alpha
                second_stage = false,
                method = method,
                u_draws = U_DRAWS,
                sample_weights = SAMPLE_WEIGHTS
            )

            stage += 1
            folder = joinpath(loop_folder, string(stage))
            mkpath(folder)
            NPZ.npzwrite(joinpath(folder, "best_params.npy"), reshape(best_params, :, 1))
            NPZ.npzwrite(joinpath(folder, "pso_history.npy"), Dict(
                "best_fitness" => history["best_fitness"],
                "mean_fitness" => history["mean_fitness"]
            ))
            generate_report(loop_folder, string(stage), 1, ["beta", "T"], best_params, string(alpha);
                             u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS)

            println("  ✓ Stage 2 complete. Fitness: $(round(best_fitness, digits=6))")

            end # skip_substage_2
            
            # ═══════════════════════════════════════════════════════════════════
            # STAGE 3: Technical Coefficients (Ω^L, Ω^s) - LOW SENSITIVITY
            # ═══════════════════════════════════════════════════════════════════
            # With λ=0.5 and ν=0.2, errors are DAMPED (×0.5 and ×0.8)
            # Can use wider bounds since optimization landscape is smooth
            
            println("\n" * "-"^50)
            println("Loop $(loop) - Stage 3: TECHNICAL COEFFICIENTS (Ω^L, Ω^s)")
            println("-"^50)
            println("  Sensitivity: LOW (damped by λ=0.5, ν=0.2)")
            println("  Using wider bounds (alpha_tech = $(round(alpha * 0.7, digits=2)))")
            
            # Wider alpha for technical coefficients (less sensitive)
            alpha_technical = alpha #* 0.7  # Allows broader exploration
            
            best_params, best_fitness, history = train_stage_pso(
                N_PARTICLES,
                MAX_ITER_STAGE,
                variable_list = ["agg_labor_share_tech", "agg_industry_share_tech"],
                last_stage_folder = joinpath(loop_folder, string(stage)),
                K = 1,
                alpha = alpha_technical,
                second_stage = false,
                method = method,
                u_draws = U_DRAWS,
                sample_weights = SAMPLE_WEIGHTS
            )
            
            stage += 1
            folder = joinpath(loop_folder, string(stage))
            mkpath(folder)
            NPZ.npzwrite(joinpath(folder, "best_params.npy"), reshape(best_params, :, 1))
            NPZ.npzwrite(joinpath(folder, "pso_history.npy"), Dict(
                "best_fitness" => history["best_fitness"],
                "mean_fitness" => history["mean_fitness"]
            ))
            generate_report(loop_folder, string(stage), 1, ["agg_labor_share_tech", "agg_industry_share_tech"], best_params, string(alpha_technical);
                             u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS)
            
            println("  ✓ Stage 3 complete. Fitness: $(round(best_fitness, digits=6))")
            
            # ═══════════════════════════════════════════════════════════════════
            # LOOP SUMMARY & CONVERGENCE CHECK
            # ═══════════════════════════════════════════════════════════════════
            
            println("\n" * "-"^50)
            println("Loop $loop COMPLETE")
            println("-"^50)
            println("  Final fitness: $(round(best_fitness, digits=6))")
            
            # Check for convergence
            if loop > 2
                prev_folder = output_folder*"/epoch_"*string(loop-1)*"/"*string(stage-3)  # 3 stages back
                prev_params = NPZ.npzread(joinpath(prev_folder, "best_params.npy"))[:,1]
                param_change = maximum(abs.(best_params .- prev_params) ./ (abs.(prev_params) .+ 1e-10))
                println("  Max parameter change from prev loop: $(round(param_change, digits=6))")
                
                if param_change < 0.000001
                    println("\n" * "="^70)
                    println("CONVERGENCE ACHIEVED!")
                    println("Parameter change < 0.0001%")
                    println("="^70)
                    max_loop = loop
                    break
                end
            end
            
            # ═══════════════════════════════════════════════════════════════════
            # PERIODIC REPORTING
            # ═══════════════════════════════════════════════════════════════════
            if REPORT_EVERY !== nothing && loop % REPORT_EVERY == 0
                println("\n" * "-"^50)
                println("Running periodic reporting (every $REPORT_EVERY epochs)")
                println("-"^50)
                run_reporting(output_folder, loop)
            end
        end

        println("\n" * "="^70)
        println("PSO OPTIMIZATION COMPLETE")
        println("="^70)
        println("Total loops completed: $loop_start to $(min(loop_start + max_loop - 1, max_loop))")
        println("Total stages: $stage")
        println("Final best fitness: $(round(best_fitness, digits=6))")
        println("Results saved to: $output_folder")

    end

    ############## FINAL REPORTING ##############
    
    println("\n" * "="^70)
    println("RUNNING FINAL REPORTING")
    println("="^70)
    
    run_reporting(output_folder, max_loop)
end

############## POST-HOC ANALYSIS ##############

last_stage_folder = find_last_stage_folder(output_folder)
println("\n" * "="^70)
println("POST-HOC ANALYSIS FOR $(uppercase(industry))")
println("="^70)
println("Loading parameters from: $last_stage_folder")

best_params = NPZ.npzread(joinpath(last_stage_folder, "best_params.npy"))
if ndims(best_params) > 1
    best_params = best_params[:, 1]
end

# Run unified validation (all three models in one panel)
println("\n>>> RUNNING UNIFIED VALIDATION (ALL THREE MODELS) <<<")
results_unified = validate_table2_all_models(best_params, industry, T_periods=36, time_fe_mode="resample")
Parquet.write_parquet(joinpath(output_folder, "simulated_panel_unified.parquet"), results_unified["panel_df"])
Parquet.write_parquet(joinpath(output_folder, "regional_sales_unified.parquet"), results_unified["regional_sales_df"])

# Save network outputs (using sparse firm-level data)
network = solve_network(best_params, return_firm_level=true, u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS)
folder = output_folder

# Reconstruct w_srd_r from sparse firm-level data
# w_srd_r[s, r_prime, r] = share of upstream (s, r_prime)'s sales going to downstream r
# We need X_lrs-like data: expenditure from upstream (l,s) to downstream r, scaled by mu*Y_r
mu = network.mu
Y_r = network.Y_r
w_srd_r = zeros(S, R, R)
# Build X_lrs from sparse COO
X_lrs_sparse = zeros(R, R, S)
n_entries = length(network.firm_exp_rho)
for i in 1:n_entries
    g = network.firm_exp_g[i]
    l = GOOD_R[g]
    s = network.firm_exp_s[i]
    r = network.firm_exp_r[i]
    X_lrs_sparse[l, r, s] += network.firm_exp_val[i] * mu * Y_r[r]
end

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

println("\nPost-hoc analysis complete for industry: $industry")
println("Results saved to: $folder")
println("\nOutput files:")
println("  - simulated_panel_unified.csv (ALL THREE MODELS)")
println("  - untargeted_summary_unified.txt")
println("  - suppliers.npy")
println("  - w_srd_r.npy")

# Build suppliers DataFrame from sparse COO firm-level data (only non-zero entries)
sirens = Int[]
sectors = Int[]
ze2010 = Int[]
ze2010_downstream = Int[]
share = Float64[]
downstream_purchase = Float64[]
intermediate_derivative = Float64[]
productivity = Float64[]

# Assign unique siren per (l, s, rho) triplet
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

