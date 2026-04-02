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
w_rs_local = NPZ.npzread(joinpath(input_folder, "w_rs.npy"))
regional_wages_local = NPZ.npzread(joinpath(input_folder, "regional_wages.npy"))
N_downstream_per_region_local = NPZ.npzread(joinpath(input_folder,"N_downstream_per_region.npy"))
filter_N_upstream_local = NPZ.npzread(joinpath(input_folder,"filter_N_upstream.npy"))
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

# Load empirical moments
#@everywhere const emp_pi_r_labor = $(NPZ.npzread(joinpath(input_folder,"emp_pi_r.npy")))
@everywhere const emp_gamma_ls = $(permutedims(NPZ.npzread(joinpath(input_folder,"emp_gamma_ls.npy"))))
X_dr_local = CSV.read(joinpath(input_folder,"X_dr.csv"), DataFrame).X_dr
X_dr_local = X_dr_local[N_downstream_per_region.!=0]
emp_pi_r_local = X_dr_local./sum(X_dr_local)
@everywhere const emp_pi_r_full = $(emp_pi_r_local)
@everywhere const emp_pi_r = $(emp_pi_r_local[2:end])
@everywhere const reg_coef = $(NPZ.npzread(joinpath(input_folder,"reg_coef_"*string(n_coef)*".npy")))
@everywhere const N_beta = $(length(NPZ.npzread(joinpath(input_folder,"reg_coef_"*string(n_coef)*".npy"))))
empirical_moments_local = [[agg_labor_share],agg_industry_share[2:end],emp_gamma_ls,reg_coef,emp_pi_r]
empirical_moments_local = vcat([vec(empirical_moments_local[i]) for i in 1:(length(empirical_moments_local))]...)   
empirical_moments_local = reshape(empirical_moments_local,1,length(empirical_moments_local))
# @everywhere const mask_emp_gamma_ls = $(NPZ.npzread(joinpath(input_folder,"mask_gamma_ls.npy"))')
@everywhere const empirical_moments = $(empirical_moments_local)
# @everywhere const empirical_moments_reduced = $(reshape(emp_gamma_ls[mask_emp_gamma_ls.!=0],(1,size(emp_gamma_ls[mask_emp_gamma_ls.!=0])[1])))
@everywhere const K_max = $(50)
#@everywhere const sigma_sr = $(NPZ.npzread(joinpath(input_folder,"sigma_sr.npy")))

# After empirical_moments_local is built
n_labor = 1
n_industry = length(agg_industry_share) -1
n_gamma = length(vec(emp_gamma_ls))  # however you reference it
n_reg = length(reg_coef)
n_pi = length(emp_pi_r)

N_moments = n_labor + n_industry + n_gamma + n_reg + n_pi
weights = ones(N_moments)

# Upweight regression coefficients (indices for reg_coef block)
reg_start = n_labor + n_industry + n_gamma + 1
reg_end = reg_start + n_reg - 1
weights[reg_start:reg_end] .= 100  # start with 10x, tune as needed


# Construct the diagonal matrix
Weight_matrix_custom_local = Diagonal(weights)

# Make it available on all processes
@everywhere const Weight_matrix_custom = $Weight_matrix_custom_local

@everywhere include("model_CP.jl")
@everywhere include("tools.jl")
@everywhere include("pso_integration.jl")  # PSO functions
@everywhere include("run_untargeted_validation.jl")

# NOTE: pso_integration.jl needs to be updated to pass u_draws/sample_weights in its internal objective wrappers

# Precompute CdGM-style stratified productivity draws
println("Generating CdGM-style stratified productivity draws...")
u_draws_local, sample_weights_local = generate_stratified_draws(S_, R_, N_rho)
@everywhere const U_DRAWS = $u_draws_local
@everywhere const SAMPLE_WEIGHTS = $sample_weights_local
println("  Firms per (s,r): $(size(u_draws_local, 1))")
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
    println("="^70)


    # First find a reasonable beta using targeted search on regression coefficients


    # LHS-based initial beta search (replaces grid search)
    N_LHS_SAMPLES = 1500
    lhs_betas = generate_lhs_beta(N_LHS_SAMPLES, N_beta, 0.00005, 100.0)

    # Use initial guess for other parameters
    A = copy(emp_pi_r_full).^(1/abs(epsilon)).*regional_wages[N_downstream_per_region .!= 0]  # analytical inversion
    A ./= sum(A)

    init_other = vcat([agg_labor_share], agg_industry_share, A, vec(T_rs_init).+0.1)
    expanding_beta = [vcat(beta, init_other) for beta in lhs_betas]

    println("Evaluating $(length(expanding_beta)) LHS beta samples in parallel...")
    results_ = pmap(p -> parallel_SMM_safe(p; u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS), expanding_beta)

    # Find beta that best matches regression coefficients
    reg_coefs_sim = [r !== nothing ? r[2][4] : fill(NaN, N_beta) for r in results_]
    reg_distances = [sum((reg_coef .- rc).^2) for rc in reg_coefs_sim]
    init_beta = expanding_beta[argmin(reg_distances)][1:N_beta]

    println("Best initial beta: ", round.(init_beta, digits=6))
    println("Related regression coefficients are: ", round.(reg_coefs_sim[argmin(reg_distances)], digits=6))

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
        method = method
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
                method = method  # Log-space loss for π_r (handles concentration)
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
                method = method
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
                method = method
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

# Save network outputs
network = solve_network(best_params, return_firm_level=true, u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS)
folder = output_folder 

w_srd_r = zeros(S, R, R)
X_lrs = network[1]
    
for s in 1:S, r_prime in 1:R
    total_downstream_sales = sum(X_lrs[r_prime, :, s])
    if total_downstream_sales > 1e-10
        for r in 1:R
            w_srd_r[s, r_prime, r] = X_lrs[r_prime, r, s] / total_downstream_sales
        end
    end
end

npzwrite(joinpath(folder, "w_srd_r.npy"), w_srd_r)

firm_expenditure_shares = network[7]
links = firm_expenditure_shares .!= 0
suppliers = reshape(sum(links, dims=4), S, N_rho, R) .!= 0
npzwrite(joinpath(folder, "suppliers.npy"), suppliers)

println("\nPost-hoc analysis complete for industry: $industry")
println("Results saved to: $folder")
println("\nOutput files:")
println("  - simulated_panel_unified.csv (ALL THREE MODELS)")
println("  - untargeted_summary_unified.txt")
println("  - suppliers.npy")
println("  - w_srd_r.npy")

siren = 0
sirens = Int[]
sectors = Int[]
ze2010 = Int[]
ze2010_downstream = Int[]
share = Float64[]
size_vec = Float64[]
downstream_purchase = Float64[]
intermediate_derivative = Float64[]
productivity = Float64[]
for l in 1:R
    for s in 1:S
        for rho in 1:N_rho
            global siren  # ADD THIS LINE
            siren += 1
            for r in 1:R
                push!(sirens,siren)
                push!(sectors, s)
                push!(ze2010, l)
                push!(ze2010_downstream, r)
                push!(share, firm_expenditure_shares[rho,s,l,r])    
                push!(downstream_purchase, network[3][r]*network[9])     
                push!(intermediate_derivative, network[8][rho, s, l, r])  
                push!(productivity, network[5][rho,l,s])  
            end       
        end
    end
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

