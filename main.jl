
# ps aux | grep '[j]ulia' | awk '{print $2}' | xargs kill -9

# import Pkg; Pkg.add("QuasiMonteCarlo")
# import Pkg; Pkg.add("StatsPlots")
# import Pkg; Pkg.add("DataFrames")
# import Pkg; Pkg.add("NPZ")
# import Pkg; Pkg.add("Distributions")
# import Pkg; Pkg.add("Plots")
# import Pkg; Pkg.add("CSV")

using Distributed
@everywhere using NPZ
@everywhere using QuasiMonteCarlo
@everywhere using StatsPlots
@everywhere using DataFrames
@everywhere using Distributions
@everywhere using Plots
@everywhere using CSV
@everywhere using Random
addprocs(100)
Random.seed!(1)
@everywhere include("model_CP.jl")
############## Load Parameters #################
@everywhere const low_high = $(true)


folder = "./baseline"



distances_local = NPZ.npzread(joinpath(folder, "distances.npy"))
N_downstream_per_region_local = NPZ.npzread(joinpath(folder,"N_downstream_per_region.npy")) # Should now contain the number of downstream firm per region. 
filter_N_upstream_local = NPZ.npzread(joinpath(folder,"filter_N_upstream.npy"))

S_,R_ = size(filter_N_upstream_local)

@everywhere const S = $(S_)
@everywhere const R = $(R_)

input_share_local = NPZ.npzread(joinpath(folder,"input_share.npy"))

@everywhere const input_share = $(input_share_local)
@everywhere const labor_share = $(0.12)

# Build empirical moments
test = false
if test
    empirical_moments_local = NPZ.npzread(joinpath(folder,"empirical_moments.npy"))
else
    emp_chi_js = (NPZ.npzread(joinpath(folder,"emp_chi_js.npy"))')[2:end,:]
    emp_pi_jA = NPZ.npzread(joinpath(folder,"emp_pi_jA.npy"))[2:end]
    reg_coef = [0.036]
    empirical_moments_local = [emp_chi_js,emp_pi_jA,reg_coef,input_share,[labor_share]]
    empirical_moments_local = vcat([vec(empirical_moments_local[i]) for i in 1:(length(empirical_moments_local)-1)]...)    
end
@everywhere const empirical_moments = $(empirical_moments_local) # Ajout

@everywhere regional_wages = $(ones(R))

# Then broadcast those large fixed arrays to all workers:
@everywhere const N_downstream_per_region = $(N_downstream_per_region_local)     
@everywhere const distances = $(distances_local)
@everywhere const filter_N_upstream = $(filter_N_upstream_local)
@everywhere const N_rho = $(100)

@everywhere const sigma = $(2.46)
@everywhere const lambda = $(0.5)
@everywhere const nu = $(0.001)
@everywhere const nu_s = $(ones(S).*3) 
@everywhere const theta = $(1.768) 


###### Functions #######
@everywhere function parallel_SMM(params,simulation)
    return full_SMM(params,simulation)
end


@everywhere function parallel_SMM_safe(seed,simulation = false,show_err = true)
    try
        # Perform the actual computation (replace with your actual logic)
        result = parallel_SMM(seed,simulation)

        return result
    catch e
        # If an error occurs, return a message or a placeholder result
        println("Error occurred with parameters: $seed.")
        if show_err
            println(e)
        end
        return nothing  # You can also return an error message or a custom value
    end
end


@everywhere function generate_halton_grid(n)
# beta,theta,nu_s,nu,lambda,sigma,productivity,T
    lb_beta,lb_first_nest_tech,lb_second_nest_tech,lb_prod,lb_T, = 0.5,0.8*labor_share,0.8.*input_share,0.5*ones(R),0.5*ones(S*R)
    ub_beta,ub_first_nest_tech,ub_second_nest_tech,ub_prod,ub_T, = 1.5,1.2*labor_share,1.2.*input_share,1.5*ones(R),1.5*ones(S*R)

    lb_prod = lb_prod[N_downstream_per_region.!=0]
    ub_prod = ub_prod[N_downstream_per_region.!=0]

    lb = Any[vcat(lb_beta,lb_first_nest_tech,lb_second_nest_tech,lb_prod,lb_T)...]
    ub = Any[vcat(ub_beta,ub_first_nest_tech,ub_second_nest_tech,ub_prod,ub_T)...]

    halton_samples = QuasiMonteCarlo.sample(n, lb, ub, HaltonSample())  # n rows, 8 cols

    # This will create a vector of 100 tuples, each with 8 parameters
    return [(halton_samples[1,i],halton_samples[2,i],halton_samples[3:2+(S),1]/sum(halton_samples[3:2+(S),1]),halton_samples[(S+3):(size(ub_prod)[1]+S+2),i],halton_samples[(size(ub_prod)[1]+(S+3)):(size(ub_prod)[1]+size(lb_T)[1]+S+2),i]) for i in 1:(n-1)]
end


simulation = false
n = 100
if simulation

    t1 = time()
    simulations = pmap(parallel_SMM_safe, 1:2)
    t1 = time()-t1
    print(t1)

    simulations = filter(!isnothing, simulations)
    simulations = mean(simulations)

    npzwrite(joinpath(folder, "M_ij.npy"), simulations)
else
    params_list = generate_halton_grid(n)
    t1 = time()
    results = pmap(parallel_SMM_safe, params_list)
    t1 = time()-t1
    print(t1)
end    




# Format scores
params_matrix = hcat([collect(params) for params in params_list]...)
# Create a DataFrame
param_names = ["theta", "phi_bar", "alpha", "beta", "mu_T", "sigma_T"]  # Column names for the parameters
df = DataFrame(params_matrix', :auto)  # Transpose to get parameters as rows
rename!(df, param_names)  # Rename columns to match parameter names
score = [score[1] != nothing ? score[1][1] : missing for score in results]


if !isempty(workers())
    rmprocs(workers())
end
GC.gc()

