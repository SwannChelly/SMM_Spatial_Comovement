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

using Statistics, Printf
using StatsBase

# Add workers
available = Sys.CPU_THREADS - nprocs()
println("Using "*string(available)*" workers")
addprocs(max(available-1, 0)) # Always leave one core for other tests. 


############## Load Parameters #################

industry = length(ARGS) >= 1 ? ARGS[1] : "aero"  # Default to "aero" if no argument
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
@everywhere const N_rho = $(50)
@everywhere const epsilon = $(coefs[1,"value"])
@everywhere const lambda = $(0.5)
@everywhere const nu = $(0.2)
@everywhere const nu_s = $(ones(S).*2.5) 
@everywhere const theta = $(1.768) 
@everywhere const delta_r = $(ones(R))
@everywhere const Weight_matrix = $(nothing)
@everywhere const N_rs = $(N_rs_local)

# Load empirical moments
@everywhere const emp_gamma_ls = $(NPZ.npzread(joinpath(input_folder,"emp_gamma_ls.npy"))')
@everywhere const emp_pi_r_full = $(NPZ.npzread(joinpath(input_folder,"emp_pi_r.npy")))
@everywhere const emp_pi_r = $(NPZ.npzread(joinpath(input_folder,"emp_pi_r.npy"))[2:end])
@everywhere const reg_coef = $(NPZ.npzread(joinpath(input_folder,"reg_coef.npy")))
@everywhere const N_beta = $(length(reg_coef))
empirical_moments_local = [[agg_labor_share],agg_industry_share[2:end],emp_gamma_ls,reg_coef,emp_pi_r]
empirical_moments_local = vcat([vec(empirical_moments_local[i]) for i in 1:(length(empirical_moments_local))]...)   
empirical_moments_local = reshape(empirical_moments_local,1,length(empirical_moments_local))
# @everywhere const mask_emp_gamma_ls = $(NPZ.npzread(joinpath(input_folder,"mask_gamma_ls.npy"))')
@everywhere const empirical_moments = $(empirical_moments_local)
# @everywhere const empirical_moments_reduced = $(reshape(emp_gamma_ls[mask_emp_gamma_ls.!=0],(1,size(emp_gamma_ls[mask_emp_gamma_ls.!=0])[1])))
@everywhere const K_max = $(50)
@everywhere const sigma_sr = $(NPZ.npzread(joinpath(input_folder,"sigma_sr.npy")))

@everywhere include("model_CP.jl")
@everywhere include("tools.jl")
@everywhere include("pso_integration.jl")  # NEW: PSO functions
@everywhere include("run_untargeted_validation.jl")  

# Distance bins
DistBin_local = Array{Int}(undef, R,R)
for i in 1:R, j in 1:R
    DistBin_local[i,j] = distance_bin(distances_local[i,j])
end
@everywhere const DistBin = $(DistBin_local)

# PSO Configuration
N_PARTICLES = available-1  # Use all available cores except one 
MAX_ITER_INITIAL = 200    # Iterations for initial full optimization
MAX_ITER_STAGE = 50     # Iterations for each refinement stage
method = "original"
max_loop = 100
full_run = true
length_range_beta = 20 # Normal is 50

# Reporting configuration
REPORT_EVERY = 2  # Run reporting every X epochs (set to nothing for only at the end)

############## MAIN OPTIMIZATION ##############

if full_run
    ############# INITIAL SEARCH FOR GOOD BETA ##############

    println("\n" * "="^70)
    println("Method $method")
    println("STAGE 0: Finding good initial beta values")
    println("="^70)


    # First find a reasonable beta using targeted search on regression coefficients
    

    # Create grid with 3 categories:
    # - i: first beta (β₁)
    # - j: second beta (β₂)  
    # - k: remaining betas (β₃, β₄, β₅) - all share same value
    # if industry == "aero"
    #     range_beta = range(0.0005, stop = 3, length = length_range_beta+30) 
    #     expanding_beta = [
    #         [i, k,k,k,k]  # β₁=i, β₂=j, β₃=β₄=β₅=k
    #         for i in range_beta 
    #         for k in range_beta
    #         if i <= j <= k
    #     ]
    # else 
    #     range_beta = range(0.0005, stop = 3, length = length_range_beta) 
    #     expanding_beta = [
    #         [i,j,k,k,k]  # β₁=i, β₂=j, β₃=β₄=β₅=k
    #         for i in range_beta 
    #         for j in range_beta 
    #         for k in range_beta
    #         if i <= j <= k
    #     ]
    range_beta = range(0.0005, stop = 3, length = length_range_beta) 
    expanding_beta = [
            [i,j,k,k,k]  # β₁=i, β₂=j, β₃=β₄=β₅=k
            for i in range_beta 
            for j in range_beta 
            for k in range_beta
            if i <= j <= k
        ]
    # Use initial guess for other parameters
    A = copy(emp_pi_r_full).^(1/abs(epsilon))  # analytical inversion
    A ./= sum(A) 

    init_other = vcat([agg_labor_share], agg_industry_share, A, vec(N_rs).+0.1)
    expanding_beta = [vcat(i, init_other) for i in expanding_beta]

    println("Evaluating $(length(expanding_beta)) beta combinations in parallel...")
    results_ = pmap(parallel_SMM_safe, expanding_beta)

    # Find beta that best matches regression coefficients
    scores = [score != nothing ? score[1][1] : missing for score in results_]
    k = 1
    y0 = reg_coef[k]
    y1 = reg_coef[k+1]
    y2 = reg_coef[k+2]
    reg_coef_ = [score != nothing ? [score[2][4][k],score[2][4][k+1],score[2][4][k+2]] : missing for score in results_]
    y_flat = vcat([abs(y0-yi[1])^2+abs(y1-yi[2])^2 + abs(y2-yi[3])^2 for yi in reg_coef_]...)
    init_beta = expanding_beta[argmin(y_flat)][1:N_beta]

    println("Best initial beta: ", init_beta)
    println("Related regression coefficients are: ", reg_coef_[argmin(y_flat)])

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
    generate_report(output_folder, string(stage), 1, nothing, best_params, "")
    run_reporting(output_folder, 0)

    println("\nStage $stage complete. Best fitness: $(round(best_fitness, digits=6))")

    if max_loop != nothing
        ############# MULTI-STAGE REFINEMENT ##############

        println("\n" * "="^70)
        println("Starting multi-stage PSO refinement (3 stages per loop)")
        println("="^70)
        stage = 0
        loop_start = 1
        if loop_start == 1
            stage = 0
        else
            stage = (loop_start-1)*3  # Now 3 stages per loop
        end
        println("Starting at stage: $stage")
        
        # Alpha controls search radius: starts tight, expands over time
        alpha_start, alpha_end = 0.3, 0.9
        
        for loop in loop_start:max_loop
            global stage
            global max_loop
            global best_params
            global best_fitness

            alpha = alpha_start + (loop - loop_start) * (alpha_end - alpha_start) / (max_loop - loop_start)
            
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
            generate_report(loop_folder, string(stage), 1, ["productivity"], best_params, string(alpha_productivity))
            
            println("  ✓ Stage 1 complete. Fitness: $(round(best_fitness, digits=6))")
            
            # ═══════════════════════════════════════════════════════════════════
            # STAGE 2: Spatial Structure (β, T) - MEDIUM SENSITIVITY
            # ═══════════════════════════════════════════════════════════════════
            # Trade costs affect regression coefficients
            # Fréchet scales affect sourcing shares γ_{ls}
            
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
            generate_report(loop_folder, string(stage), 1, ["beta", "T"], best_params, string(alpha))
            
            println("  ✓ Stage 2 complete. Fitness: $(round(best_fitness, digits=6))")
            
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
            generate_report(loop_folder, string(stage), 1, ["agg_labor_share_tech", "agg_industry_share_tech"], best_params, string(alpha_technical))
            
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


        ############## PLOT CONVERGENCE ##############

        function plot_pso_convergence(output_folder, n_loops)
            all_best = Float64[]
            all_mean = Float64[]
            stage_labels = String[]
            
            # Load initial stage
            hist = NPZ.npzread(joinpath(output_folder, "0", "pso_history.npy"))
            append!(all_best, hist["best_fitness"])
            append!(all_mean, hist["mean_fitness"])
            
            # Load all refinement stages dynamically
            all_folders = find_all_stage_folders(output_folder, n_loops)
            
            for folder in all_folders[2:end]  # Skip initial stage (already loaded)
                hist_file = joinpath(folder, "pso_history.npy")
                if isfile(hist_file)
                    hist = NPZ.npzread(hist_file)
                    append!(all_best, hist["best_fitness"])
                    append!(all_mean, hist["mean_fitness"])
                end
            end
            
            # Normalize to percentage of initial
            all_best_norm = 100 * all_best / all_best[1]
            all_mean_norm = 100 * all_mean / all_mean[1]
            
            p = plot(all_best_norm, label="Best Fitness", linewidth=2, 
                    xlabel="PSO Iteration (cumulative)", ylabel="Fitness (% of initial)",
                    title="PSO Convergence", legend=:topright, color=:blue)
            plot!(p, all_mean_norm, label="Mean Fitness", linewidth=2, 
                color=:red, linestyle=:dash)
            
            # Add vertical lines at stage boundaries
            cumulative_iter = length(NPZ.npzread(joinpath(output_folder, "0", "pso_history.npy"))["best_fitness"])
            for folder in all_folders[2:end]
                hist_file = joinpath(folder, "pso_history.npy")
                if isfile(hist_file)
                    hist = NPZ.npzread(hist_file)
                    cumulative_iter += length(hist["best_fitness"])
                    vline!(p, [cumulative_iter], color=:gray, alpha=0.3, label="")
                end
            end
            
            savefig(p, joinpath(output_folder, "pso_convergence.png"))
            println("\nConvergence plot saved to: $(joinpath(output_folder, "pso_convergence.png"))")
        end
    end

    ############## FINAL REPORTING ##############
    
    println("\n" * "="^70)
    println("RUNNING FINAL REPORTING")
    println("="^70)
    
    run_reporting(output_folder, max_loop)
end


############## POST-HOC ANALYSIS ##############


# Find the last stage folder dynamically
last_stage_folder = find_last_stage_folder(output_folder)
println("\n" * "="^70)
println("POST-HOC ANALYSIS FOR $(uppercase(industry))")
println("="^70)
println("Loading parameters from: $last_stage_folder")

best_params = NPZ.npzread(joinpath(last_stage_folder, "best_params.npy"))
# Handle matrix vs vector
if ndims(best_params) > 1
    best_params = best_params[:, 1]
end
results = validate_table2(best_params, industry, T_periods=36)
CSV.write(joinpath(output_folder,"simulated_panel.csv"), results["panel_df"])



network = solve_network(best_params, return_firm_level=false)
panel_df = results["panel_df"]

function get_w_srd_r(params)
    net = solve_network(params, return_firm_level=false)
    w_srd_r = zeros(S, R, R)
    X_lrs = net[1]
    
    for s in 1:S
        for r_prime in 1:R  # Upstream region
            # Total sales from (s, r') to all downstream regions
            total_downstream_sales = sum(X_lrs[r_prime, :, s])
            
            if total_downstream_sales > 1e-10
                for r in 1:R  # Downstream region
                    w_srd_r[s, r_prime, r] = X_lrs[r_prime, r, s] / total_downstream_sales
                end
            end
        end
    end
    return w_srd_r
end

folder = output_folder * "/"  # Output folder
w_srd_r = get_w_srd_r(best_params)
npzwrite(joinpath(folder, "w_srd_r.npy"), w_srd_r)

### To display map of number of suppliers

network = solve_network(best_params, return_firm_level=true)
firm_expenditure_shares = network[7]
links = firm_expenditure_shares .!= 0  # N_rho, S, l, r
suppliers = reshape(sum(links, dims=4), S, N_rho, R) .!= 0
npzwrite(joinpath(folder, "suppliers.npy"), suppliers)

println("\nPost-hoc analysis complete for industry: $industry")
println("Results saved to: $folder")