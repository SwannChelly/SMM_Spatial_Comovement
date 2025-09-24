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


addprocs(100) # Number of parallel cores.
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

# Load empirical moments and reshape them.
@everywhere const emp_gamma_ls = $(NPZ.npzread(joinpath(input_folder,"emp_gamma_ls.npy"))')
@everywhere const emp_pi_r = $(NPZ.npzread(joinpath(input_folder,"emp_pi_r.npy"))[2:end])
@everywhere const reg_coef = $(NPZ.npzread(joinpath(input_folder,"reg_coef.npy")))
empirical_moments_local = [[agg_labor_share],agg_industry_share[2:end],emp_gamma_ls,reg_coef,emp_pi_r]
empirical_moments_local = vcat([vec(empirical_moments_local[i]) for i in 1:(length(empirical_moments_local))]...)   
empirical_moments_local = reshape(empirical_moments_local,1,length(empirical_moments_local))
@everywhere const empirical_moments = $(empirical_moments_local)

#### Bellow, the model is computed over a Halton grid of size n. 


n = 100000
@everywhere include("tools.jl") # Import the model
params_list,results = train_stage_one(n)
save_stage_best_params(params_list,results,"0")
generate_report("0")

alpha = 0.1
variable = "productivity"
params_list = generate_halton_grid(n,2000,joinpath(output_folder,"0"),variable,alpha)
params_list,results = train_stage_one(n,params_list)
save_stage_best_params(params_list,results,"1")
generate_report("1",variable,alpha)





#################### End of the code #####################


#npzwrite(joinpath(output_folder, "best_params.npy"), vcat([0.36,2,2,3,3], best_params[6:end]))
npzwrite(joinpath(output_folder, "best_params.npy"), params_list[best_index])

### Bellow we test the sensitivity of the loss function with respect to beta
### We also search for the sensitivity of the regression coefficient with respect to beta

print("Gen expanding beta")
beta,agg_labor_share_tech,agg_industry_share_tech,productivity_,T_ = unpack_params(best_params)
range_beta = [range(beta[i]/10, stop = beta[i]*100, length = 10) for i in range(1,5)]
expanding_beta = [[i,j,k,l,m] for i in range_beta[1] for j in range_beta[2] for k in range_beta[3] for l in range_beta[4] for m in range_beta[5]]
expanding_beta = [vcat(i, best_params[6:end]) for i in expanding_beta]



results_ = pmap(parallel_SMM_safe, expanding_beta)
scores = [score != nothing ? score[1][1] : missing for score in results_]
reg_coef_ = [score != nothing ? score[2][4] : missing for score in results_]


reg_coef_[argmin(scores)]
expanding_beta[argmin(scores)]
minimum(scores)

full_SMM(vcat([0.36,2,2,3,3], best_params[6:end]))[2][4]
print("Gen expanding productivity")



beta,agg_labor_share_tech,agg_industry_share_tech,productivity_,T_ = unpack_params(best_params)

A = copy(N_downstream_per_region[N_downstream_per_region .!= 0])
A ./= sum(A)
n = 100000
lb_prod = A*0.1
ub_prod = 100*A
lb = Any[vcat(lb_prod)...]
ub = Any[vcat(ub_prod)...]
halton_samples = QuasiMonteCarlo.sample(n, lb, ub, HaltonSample())  # n rows, 8 cols
halton_samples =  [halton_samples[:,i] for i in range(1,n)]


expanding_beta = [vcat(beta,agg_labor_share_tech,agg_industry_share_tech,prod,T_) for prod in halton_samples]


results_ = pmap(parallel_SMM_safe, expanding_beta)
scores = [score != nothing ? score[1][1] : missing for score in results_]
reg_coef_ = [score != nothing ? score[2][4] : missing for score in results_]


reg_coef_[argmin(scores)]
expanding_beta[argmin(scores)]

A = copy(N_downstream_per_region[N_downstream_per_region .!= 0])
A ./= sum(A)
n = 100000
lb_prod = A*0.1
ub_prod = 100*A
lb = Any[vcat(lb_prod)...]
ub = Any[vcat(ub_prod)...]
halton_samples = QuasiMonteCarlo.sample(n, lb, ub, HaltonSample())  # n rows, 8 cols
halton_samples =  [halton_samples[:,i] for i in range(1,n)]