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
available = Sys.CPU_THREADS - nprocs()
println("Using "*string(available)*" workers")
addprocs(max(1, 0)) # Always leave one core for other tests. 


############## Load Parameters #################

industry = length(ARGS) >= 1 ? ARGS[1] : "auto_23"  # Default to "aero" if no argument
n_coef = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 5  # Default to 4 coefficients
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

R_downstream_ = length(N_downstream_per_region_local)
@everywhere const R_downstream = $(R_downstream_)
@everywhere const agg_industry_share = $(agg_industry_share_local)
@everywhere const agg_labor_share = $(coefs[2,"value"])
@everywhere const domestic_share = $(domestic_share_local)
@everywhere regional_wages = $(regional_wages_local)
regional_wages_downstream_local = NPZ.npzread(joinpath(input_folder, "regional_wages_downstream.npy"))
@everywhere const regional_wages_downstream = $(regional_wages_downstream_local)
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
@everywhere const delta_r = $(ones(R_downstream_))
@everywhere const Weight_matrix = $(nothing)

if industry == "aero"
    @everywhere const T_rs_init = $(X_rs_local)
elseif industry == "auto_23"
    @everywhere const T_rs_init = $(X_rs_local)# N_rs_local
end

# Load empirical moments


@everywhere const emp_pi_r_labor = $(NPZ.npzread(joinpath(input_folder,"emp_pi_r.npy")))


@everywhere const emp_gamma_ls = $(permutedims(NPZ.npzread(joinpath(input_folder,"emp_gamma_ls.npy"))))
X_dr_local = CSV.read(joinpath(input_folder,"X_dr.csv"), DataFrame).X_dr
emp_pi_r_local = X_dr_local ./ sum(X_dr_local)
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
@everywhere const sigma_sr = $(NPZ.npzread(joinpath(input_folder,"sigma_sr.npy")))

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
weights[reg_start:reg_end] .= 100.0  # start with 10x, tune as needed


# Construct the diagonal matrix
Weight_matrix_custom_local = Diagonal(weights)

# Make it available on all processes
@everywhere const Weight_matrix_custom = $Weight_matrix_custom_local

@everywhere include("model_CP.jl")
@everywhere include("tools.jl")
@everywhere include("pso_integration.jl")  # NEW: PSO functions
@everywhere include("run_untargeted_validation.jl")  




# Distance bins
DistBin_local = Array{Int}(undef, R, R_downstream)
for i in 1:R, j in 1:R_downstream
    DistBin_local[i,j] = distance_bin(distances_local[i,j])
end
@everywhere const DistBin = $(DistBin_local)

# PSO Configuration
N_PARTICLES = available-1  # Use all available cores except one 
MAX_ITER_INITIAL = 200    # Iterations for initial full optimization
MAX_ITER_STAGE = 50     # Iterations for each refinement stage
method = "original"
max_loop = 10
full_run = true
length_range_beta = 20 # Normal is 50

# Reporting configuration
REPORT_EVERY = 2  # Run reporting every X epochs (set to nothing for only at the end)


############## MAIN OPTIMIZATION ##############

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
range_beta = range(0.00000, stop = 0.01, length = 20) 
if n_coef == 4
    expanding_beta = [
            [i,j,k,k]  # β₁=i, β₂=j, β₃=β₄=k
            for i in range_beta 
            for j in range_beta 
            for k in range_beta
            if i <= j <= k
        ]
elseif n_coef == 5
    expanding_beta = [
            [0,0,i,j,j]  # β₁=i, β₂=j, β₃=β₄=β₅=k
            for i in range_beta
            for j in range_beta
        ]
end
# Use initial guess for other parameters
A = copy(emp_pi_r_full).^(1/abs(epsilon)) .* regional_wages_downstream  # analytical inversion
A ./= sum(A)

init_other = vcat([agg_labor_share], agg_industry_share, A, vec(T_rs_init).+0.1)
expanding_beta = [vcat(i, init_other) for i in expanding_beta]

println("Evaluating $(length(expanding_beta)) beta combinations in parallel...")
results_ = pmap(parallel_SMM_safe, expanding_beta)

# Find beta that best matches regression coefficients
scores = [score != nothing ? score[1][1] : missing for score in results_]
k = 1
reg_coef_ = [score != nothing ? [score[2][4][k],score[2][4][k+1],score[2][4][k+2],score[2][4][k+3],score[2][4][k+4]] : missing for score in results_]
y_flat = vcat([abs(reg_coef[2]-yi[2])^2 for yi in reg_coef_]...)
init_beta = expanding_beta[argmin(y_flat)][1:N_beta]

plot([beta[3] for beta in expanding_beta],[reg[2] for reg in reg_coef_])

# On veut voir reg
reg_coef