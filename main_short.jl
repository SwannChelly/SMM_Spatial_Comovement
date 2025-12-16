##### Main #####
# Author: Swann Chelly 
# This code can run the SMM over an Halton grid to calibrate the parameters and then compare simulated and empirical moments.

# ps aux | grep '[j]ulia' | awk '{print $2}' | xargs kill -9
# nohup julia main.jl &

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
@everywhere using ProgressMeter  # Ensure availability on all workers
@everywhere using SharedArrays

using Statistics, Printf
using StatsBase

#addprocs(100) # Number of parallel cores.
available = Sys.CPU_THREADS - nprocs()
#available = 20
println("Using "*string(available)*" workers")
addprocs(max(available, 0))
@everywhere include("model_CP.jl") # Import the model
@everywhere include("tools.jl") # Import the model

############## Load Parameters #################

industry = "aero" # Name of the industry in [aero,auto_24]
input_folder = "./baseline_"*industry # Input input_folder are stored
output_folder = "./reporting_"*industry # Output folder

coefs = CSV.read(joinpath(input_folder,"stats.csv"), DataFrame) # Sales elasticities epsilon and aggregate labor share. 
distances_local = NPZ.npzread(joinpath(input_folder, "distances.npy")) # Distance matrix: no longer used in this version
w_rs_local = NPZ.npzread(joinpath(input_folder, "w_rs.npy")) # Upstream wages 
regional_wages_local = NPZ.npzread(joinpath(input_folder, "regional_wages.npy")) # Downstream wages. Equal to 0 if no downstream firms
N_downstream_per_region_local = NPZ.npzread(joinpath(input_folder,"N_downstream_per_region.npy")) # Vector of size R that contains the number of workers per downstream region 
filter_N_upstream_local = NPZ.npzread(joinpath(input_folder,"filter_N_upstream.npy")) # Matrix of size S x R that equals to 0 if there is no supplier in region r and sector s.

agg_industry_share_local = NPZ.npzread(joinpath(input_folder,"input_share.npy")) # Industry share in total inputs (from IO tables)
domestic_share_local = NPZ.npzread(joinpath(input_folder,"domestic_share.npy")) # Domestic share per sector in total inputs (from IO tables)

S_,R_ = size(filter_N_upstream_local)
@everywhere const S = $(S_)
@everywhere const R = $(R_)

# We create the distance matrix by bins
DistBin_local = Array{Int}(undef, R,R)
for i in 1:R, j in 1:R
    DistBin_local[i,j] = distance_bin(distances_local[i,j])
end

R_ = size(N_downstream_per_region_local[N_downstream_per_region_local.!=0])[1] # Number of downstream region hosting the downstream industry
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

# Load empirical moments and reshape them.
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


#### Bellow, the model is computed over a Halton grid of size n. 

# # #### We first look at the best beta considering identical productivity
# params_list = [generate_halton_grid(2,1024,true)]
# best_params = params_list[1]
# beta,agg_labor_share_tech,agg_industry_share_tech,productivity_,T_ = unpack_params(best_params)
# range_beta = range(0.01, stop = 5, length = 50) 
# expanding_beta = [[i,j,j,j,j] for i in range_beta for j in range_beta if i<j ]
# expanding_beta = [vcat(i, best_params[6:end]) for i in expanding_beta]
# results_ = pmap(parallel_SMM_safe, expanding_beta)
# scores = [score != nothing ? score[1][1] : missing for score in results_]
# k = 1
# y0 = reg_coef[k]
# y1 = reg_coef[k+1]
# reg_coef_ = [score != nothing ? [score[2][4][k],score[2][4][k+1]] : missing for score in results_]
# y_flat = vcat([abs(y0-yi[1])^2+abs(y1-yi[2])^2 for yi in reg_coef_]...)
# init_beta = expanding_beta[argmin(y_flat)][1:5]


# n = 500000
# n = K_max
# @everywhere include("tools.jl") # Import the model
# params_list,results = train_stage_one(n,init_beta)
# save_stage_best_params(params_list,results,output_folder,"0",50)
# generate_report(output_folder,"0",n)

stage = 0
# Before loop define global variables. 

for loop in 1:10
    global stage
    global params_list
    global results
    global best_params

    print("Loop number: "*string(loop)*"\n")
    past_loop_folder =  "./reporting_"*industry*"/epoch_"*string(loop-1) # Output folder
    loop_folder =  "./reporting_"*industry*"/epoch_"*string(loop) # Output folder
    mkpath(loop_folder)
    
    n = 1000
    alpha = 0.5
    variable = "agg_labor_share_tech"
    best_params = Any[]
    print("Variable is: "*variable*" and n = "*string(n)*"\n")
    for K in 1:K_max
        if loop == 1
            params_list = generate_halton_grid(n,2000,false,nothing,joinpath(output_folder,string(stage)),K,variable,alpha)
        else
            params_list = generate_halton_grid(n,2000,false,nothing,joinpath(past_loop_folder,string(stage)),K,variable,alpha)
        end
        params_list,results = train_stage_one(n,nothing,params_list)
        score = [score[1] != nothing ? score[1][1] : missing for score in results]
        push!(best_params,params_list[argmin(score)])
    end
    stage += 1
    folder = joinpath(loop_folder, string(stage))
    mkpath(folder) 
    npzwrite(joinpath(folder, "best_params.npy"), hcat(best_params...))
    generate_report(loop_folder,string(stage),n)

    n = 10000
    #n = 2
    #stage = run_stage("productivity",n,2,stage,loop_folder,false)
    #stage = run_stage("agg_industry_share_tech",n,2,stage,loop_folder,false)
    stage = run_stage(["agg_industry_share_tech","productivity"],n,2,stage,loop_folder,false)
    stage = run_stage(["beta","T"],n,2,stage,loop_folder,false)
    #stage = run_stage("beta",n,2,stage,loop_folder,true)
                                                              
end

rmse(a::AbstractVector, b::AbstractVector) =
    sqrt(mean((a .- b).^2))

reporting = true
if reporting

    # Reporting
    n = 1
    folder = "./reporting_"*industry*"/" # Output folder
    #folder = "./parameters/"
    best_params = NPZ.npzread(joinpath(folder*"0/", "best_params.npy"))# Load best params.
    params_list = [best_params[:,K] for K in 1:50]



    # Compute scores for both stages
    top_score_first, min_dist_first = compute_scores(folder,false)
    top_score_second, min_dist_second = compute_scores(folder,true)

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
    savefig(combined_plot, joinpath(folder, "loss_function_comparison.png"))
end


# Function to compute scores for a given stage
function compute_scores(folder,second_stage::Bool)
    top_score = []
    min_distances = []
    
    # Initial evaluation
    params_list_stage = [best_params[:,K] for K in 1:50]
    params_list_stage, results = train_stage_one(n, nothing, params_list_stage, second_stage)
    score = [s[1] != nothing ? s[1][1] : missing for s in results]
    push!(top_score, minimum(score))
    
    # Compute min_distance
    reg_coef_ = [s != nothing ? s[2][4] : missing for s in results]
    min_distance = minimum(rmse.(reg_coef_, Ref(reg_coef)))
    push!(min_distances, min_distance)
    
    # Loop through epochs
    stage = 1
    for loop in 1:10
        for k in 1:3
            folder_stage = folder*"epoch_"*string(loop)*"/"*string(stage)
            print(folder_stage)
            best_params_stage = NPZ.npzread(joinpath(folder_stage, "best_params.npy"))
            params_list_stage = [best_params_stage[:,K] for K in 1:50]
            params_list_stage, results = train_stage_one(n, nothing, params_list_stage, second_stage)
            score = [s[1] != nothing ? s[1][1] : missing for s in results]
            push!(top_score, minimum(score))
            
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
    return normalized_score, min_distances
end