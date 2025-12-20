##### Main with PSO Optimization #####
# Author: Swann Chelly (Modified for PSO)
# This replaces Halton grid search with Particle Swarm Optimization

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

@everywhere include("model_CP.jl")
@everywhere include("tools.jl")
@everywhere include("pso_integration.jl")  # NEW: PSO functions

############## Load Parameters #################

industry = "aero"
input_folder = "./baseline_"*industry
output_folder = "./reporting_"*industry

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

# Distance bins
DistBin_local = Array{Int}(undef, R,R)
for i in 1:R, j in 1:R
    DistBin_local[i,j] = distance_bin(distances_local[i,j])
end

R_ = size(N_downstream_per_region_local[N_downstream_per_region_local.!=0])[1]
@everywhere const R_downstream = $(R_)
@everywhere const agg_industry_share = $(agg_industry_share_local)
@everywhere const agg_labor_share = $(coefs[2,"value"])
@everywhere const domestic_share = $(domestic_share_local)
@everywhere regional_wages = $(regional_wages_local)
@everywhere const distances = $(distances_local)
@everywhere const N_downstream_per_region = $(N_downstream_per_region_local)     
@everywhere const w_rs = $(w_rs_local)
@everywhere const DistBin = $(DistBin_local)
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
@everywhere const emp_pi_r = $(NPZ.npzread(joinpath(input_folder,"emp_pi_r.npy"))[2:end])
@everywhere const reg_coef = $(NPZ.npzread(joinpath(input_folder,"reg_coef.npy")))
empirical_moments_local = [[agg_labor_share],agg_industry_share[2:end],emp_gamma_ls,reg_coef,emp_pi_r]
empirical_moments_local = vcat([vec(empirical_moments_local[i]) for i in 1:(length(empirical_moments_local))]...)   
empirical_moments_local = reshape(empirical_moments_local,1,length(empirical_moments_local))
@everywhere const mask_emp_gamma_ls = $(NPZ.npzread(joinpath(input_folder,"mask_gamma_ls.npy"))')
@everywhere const empirical_moments = $(empirical_moments_local)
@everywhere const empirical_moments_reduced = $(reshape(emp_gamma_ls[mask_emp_gamma_ls.!=0],(1,size(emp_gamma_ls[mask_emp_gamma_ls.!=0])[1])))
@everywhere const K_max = $(50)


############## INITIAL SEARCH FOR GOOD BETA ##############

println("\n" * "="^70)
println("STAGE 0: Finding good initial beta values")
println("="^70)

# First find a reasonable beta using targeted search on regression coefficients
range_beta = range(0.01, stop = 5, length = 50) 
expanding_beta = [[i,j,j,j,j] for i in range_beta for j in range_beta if i<j ]

# Use initial guess for other parameters
A = copy(N_downstream_per_region[N_downstream_per_region .!= 0])
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
reg_coef_ = [score != nothing ? [score[2][4][k],score[2][4][k+1]] : missing for score in results_]
y_flat = vcat([abs(y0-yi[1])^2+abs(y1-yi[2])^2 for yi in reg_coef_]...)
init_beta = expanding_beta[argmin(y_flat)][1:5]

println("Best initial beta: ", init_beta)
println("Related regression coefficients are: ", reg_coef_[argmin(y_flat)])


############## PSO-BASED OPTIMIZATION ##############

# PSO Configuration
N_PARTICLES = available-1  # Use all available cores except one 
MAX_ITER_INITIAL = 200    # Iterations for initial full optimization
MAX_ITER_STAGE = 30     # Iterations for each refinement stage

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
    second_stage = false
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

println("\nStage $stage complete. Best fitness: $(round(best_fitness, digits=6))")


############# MULTI-STAGE REFINEMENT ##############

println("\n" * "="^70)
println("Starting multi-stage PSO refinement")
println("="^70)
stage = 0
loop_start = 8
if loop_start == 1
    stage = 0
else
    stage = (loop_start-1)*2
end
println(stage)

for loop in loop_start:(loop_start+20)  # Reduced from 20 since PSO is more efficient
    global stage
    global best_params
    global best_fitness
    if loop >= 7 
        rescale = true
    else 
        rescale = false
    end
    println("\n" * "-"^70)
    println("REFINEMENT LOOP $loop")
    println("-"^70)
    
    past_loop_folder = loop == 1 ? output_folder : "./reporting_"*industry*"/epoch_"*string(loop-1)
    loop_folder = "./reporting_"*industry*"/epoch_"*string(loop)
    mkpath(loop_folder)
    
    # Stage 1: Refine industry shares and productivity
    println("\nLoop $(loop): Refining industry shares and productivity...")
    best_params, best_fitness, history = train_stage_pso(
        N_PARTICLES,
        MAX_ITER_STAGE,
        variable_list = ["agg_labor_share_tech","agg_industry_share_tech", "productivity"],
        last_stage_folder = joinpath(past_loop_folder, string(stage)),
        K = 1,
        alpha = 0.2,  # Narrower search for refinement
        second_stage = false,
        rescale = rescale
    )
    
    stage += 1
    folder = joinpath(loop_folder, string(stage))
    mkpath(folder)
    NPZ.npzwrite(joinpath(folder, "best_params.npy"), reshape(best_params, :, 1))
    NPZ.npzwrite(joinpath(folder, "pso_history.npy"), Dict(
        "best_fitness" => history["best_fitness"],
        "mean_fitness" => history["mean_fitness"]
    ))
    generate_report(loop_folder, string(stage), 1, ["agg_industry_share_tech", "productivity"], best_params, "0.2")
    
    # Stage 2: Refine beta and T
    println("\nLoop $(loop): Refining beta and T...")
    best_params, best_fitness, history = train_stage_pso(
        N_PARTICLES,
        MAX_ITER_STAGE,
        variable_list = ["beta", "T"],
        last_stage_folder = joinpath(loop_folder, string(stage)),
        K = 1,
        alpha = 0.2,
        second_stage = false,
        rescale = rescale
    )
    
    stage += 1
    folder = joinpath(loop_folder, string(stage))
    mkpath(folder)
    NPZ.npzwrite(joinpath(folder, "best_params.npy"), reshape(best_params, :, 1))
    NPZ.npzwrite(joinpath(folder, "pso_history.npy"), Dict(
        "best_fitness" => history["best_fitness"],
        "mean_fitness" => history["mean_fitness"]
    ))
    generate_report(loop_folder, string(stage), 1, ["beta", "T"], best_params, "0.2")
    
    println("\nLoop $loop complete. Best fitness: $(round(best_fitness, digits=6))")
    
    # Check for convergence (optional)
    if loop > 2
        prev_folder = "./reporting_"*industry*"/epoch_"*string(loop-1)*"/"*string(stage-2)
        prev_params = NPZ.npzread(joinpath(prev_folder, "best_params.npy"))[:,1]
        param_change = maximum(abs.(best_params .- prev_params) ./ (abs.(prev_params) .+ 1e-10))
        println("Maximum parameter change: $(round(param_change, digits=4))")
        
        if param_change < 0.01
            println("\nConvergence achieved! Parameter change < 1%")
            break
        end
    end
end

println("\n" * "="^70)
println("PSO OPTIMIZATION COMPLETE")
println("="^70)
println("Final best fitness: $(round(best_fitness, digits=6))")
println("Results saved to: $output_folder")


############## PLOT CONVERGENCE ##############

# Plot PSO convergence history
function plot_pso_convergence(output_folder, n_loops)
    all_best = Float64[]
    all_mean = Float64[]
    
    # Load initial stage
    hist = NPZ.npzread(joinpath(output_folder, "0", "pso_history.npy"))
    append!(all_best, hist["best_fitness"])
    append!(all_mean, hist["mean_fitness"])
    
    # Load all refinement stages
    for loop in 1:n_loops
        for stage_offset in 1:2
            folder = "./reporting_"*industry*"/epoch_"*string(loop)*"/"*string(stage_offset)
            if isdir(folder)
                hist_file = joinpath(folder, "pso_history.npy")
                if isfile(hist_file)
                    hist = NPZ.npzread(hist_file)
                    append!(all_best, hist["best_fitness"])
                    append!(all_mean, hist["mean_fitness"])
                end
            end
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
    
    savefig(p, joinpath(output_folder, "pso_convergence.png"))
    println("\nConvergence plot saved to: $(joinpath(output_folder, "pso_convergence.png"))")
end

# plot_pso_convergence(output_folder, 3)




# Function to compute scores for a given stage
function compute_scores(folder,second_stage::Bool,max_stage = nothing)
    top_score = []
    min_distances = []
    best_simulated_moments = []
    best_parameters_list = []
    
    # Initial evaluation
    params_list_stage = [best_params]
    params_list_stage, results = train_stage_one(n, nothing, params_list_stage, second_stage)
    score = [s[1] != nothing ? s[1][1] : missing for s in results]
    push!(top_score, minimum(score))
    
    # Compute min_distance
    reg_coef_ = [s != nothing ? s[2][4] : missing for s in results]
    min_distance = minimum(rmse.(reg_coef_, Ref(reg_coef)))
    push!(min_distances, min_distance)

    simulated_moments = results[1][2]
    simulated_moments = vcat([vec(simulated_moments[i]) for i in 1:(length(simulated_moments))]...)
    push!(best_simulated_moments,simulated_moments)

    push!(best_parameters_list,best_params)
    normalized_score = (top_score ./ top_score[1]) .* 100
    min_distances = (min_distances ./ min_distances[1]) .* 100
    if max_stage == nothing
        return normalized_score, min_distances,best_simulated_moments,best_parameters_list
    end

    # Loop through epochs
    stage = 1
    for loop in 1:max_stage
        for k in 1:2
            folder_stage = folder*"epoch_"*string(loop)*"/"*string(stage)
            print(folder_stage)
            best_params_stage = NPZ.npzread(joinpath(folder_stage, "best_params.npy"))
            params_list_stage = [best_params_stage]
            params_list_stage, results = train_stage_one(n, nothing, params_list_stage, second_stage)
            score = [s[1] != nothing ? s[1][1] : missing for s in results]
            push!(top_score, minimum(score))
            simulated_moments = results[argmin(score)][2]
            simulated_moments = vcat([vec(simulated_moments[i]) for i in 1:(length(simulated_moments))]...)
            push!(best_simulated_moments,simulated_moments)
            push!(best_parameters_list,best_params_stage[:,argmin(score)])
            
            # Compute min_distance
            reg_coef_ = [s != nothing ? s[2][4] : missing for s in results]
            min_distance = minimum(rmse.(reg_coef_, Ref(reg_coef)))
            push!(min_distances, min_distance)
            
            stage += 1
        end
    end
    
    # Normalize loss to percentage
    normalized_score = (top_score ./ top_score[1]) .* 100
    min_distances = (min_distances ./ min_distances[1]) .* 100
    return normalized_score, min_distances,best_simulated_moments,best_parameters_list
end


rmse(a::AbstractVector, b::AbstractVector) =
    sqrt(mean((a .- b).^2))

max_stage = 20
reporting = true
if reporting

    # Reporting
    n = 1
    folder = "./reporting_"*industry*"/" # Output folder
    #folder = "./parameters/"
    best_params = NPZ.npzread(joinpath(folder*"0/", "best_params.npy"))# Load best params.
    params_list = [best_params]

    # Compute scores for both stages
    top_score_first, min_dist_first,best_simulated_moments,best_parameters_list = compute_scores(folder,false,max_stage)
    top_score_second, min_dist_second,_,best_parameters_list = compute_scores(folder,true,max_stage)

    # Create subplots for first stage
    p1 = plot(top_score_first, marker=:circle, linewidth=2, label="Loss",
            xlabel="Iteration", ylabel="Normalized Loss (%)", 
            title="First Stage", legend=:topright, color=:blue)
    plot!(twinx(), min_dist_first, marker=:square, linewidth=2, label="Min Distance",
        ylabel="Min Distance", legend=:topleft, color=:red)

    # Create subplots for second stage
    p2 = plot(top_score_second, marker=:circle, linewidth=2, label="Loss",
            xlabel="Iteration", ylabel="Normalized Loss (%)", 
            title="Second Stage", legend=:topright, color=:blue)
    plot!(twinx(), min_dist_second, marker=:square, linewidth=2, label="Min Distance",
        ylabel="Min Distance", legend=:topleft, color=:red)

    # Combine into single figure with 2 subplots
    combined_plot = plot(p1, p2, layout=(2,1), size=(800, 800))

    # Save the figure
    # savefig(combined_plot, joinpath(folder, "loss_function_comparison.png"))
    
    npzwrite(joinpath(folder, "best_simulated_moments.npy"),hcat(best_simulated_moments...))
    npzwrite(joinpath(folder, "best_parameters_list.npy"),hcat(best_parameters_list...))
    npzwrite(joinpath(folder, "empirical_moments.npy"),empirical_moments)
end



