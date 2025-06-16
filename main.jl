
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
@everywhere using Optim

@everywhere using ProgressMeter  # Ensure availability on all workers
@everywhere using SharedArrays

using Statistics, Printf


addprocs(100)

Random.seed!(1)
@everywhere include("model_CP.jl")
############## Load Parameters #################


folder = "./baseline"

distances_local = NPZ.npzread(joinpath(folder, "distances.npy"))
N_downstream_per_region_local = NPZ.npzread(joinpath(folder,"N_downstream_per_region.npy")) # Should now contain the number of downstream firm per region. 
filter_N_upstream_local = NPZ.npzread(joinpath(folder,"filter_N_upstream.npy"))
N_downstream_per_region_local[N_downstream_per_region_local.!=0] = N_downstream_per_region_local[N_downstream_per_region_local.!=0]./N_downstream_per_region_local[N_downstream_per_region_local.!=0]
S_,R_ = size(filter_N_upstream_local)

@everywhere const S = $(S_)
@everywhere const R = $(R_)

R_ = size(N_downstream_per_region_local[N_downstream_per_region_local.!=0])[1]
@everywhere const R_downstream = $(R_)

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
    reg_coef = [-0.009]
    empirical_moments_local = [emp_chi_js,emp_pi_jA,reg_coef,input_share[2:end]]
    empirical_moments_local = vcat([vec(empirical_moments_local[i]) for i in 1:(length(empirical_moments_local)-1)]...)   
    empirical_moments_local = reshape(empirical_moments_local,1,length(empirical_moments_local))
 
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
@everywhere const nu = $(0.9)
@everywhere const nu_s = $(ones(S).*3) 
@everywhere const theta = $(1.768) 


###### Functions #######
@everywhere function parallel_SMM(params,simulation)
    return full_SMM(params,simulation)
end


@everywhere function parallel_SMM_safe(params,simulation = false,show_err = true)
    try
        # Perform the actual computation (replace with your actual logic)
        result = parallel_SMM(params,simulation)

        return result
    catch e
        # If an error occurs, return a message or a placeholder result
        if show_err
            println("ERROR!!")
            println(e)
        end
        return nothing  # You can also return an error message or a custom value
    end
end


@everywhere function generate_halton_grid(n)
# beta,theta,nu_s,nu,lambda,sigma,productivity,T
    A = copy(N_downstream_per_region_local[N_downstream_per_region_local .!= 0])
    A ./= sum(A)
    #A = ones(size(A)[1])
    lb_beta,lb_first_nest_tech,lb_second_nest_tech,lb_prod,lb_T, = 0.5,0.8*labor_share,0.8.*input_share,0.5*A,0.5*ones(S*R)
    ub_beta,ub_first_nest_tech,ub_second_nest_tech,ub_prod,ub_T, = 1.5,1.2*labor_share,1.2.*input_share,1.5*A,1.5*ones(S*R)


    #lb = Any[vcat(lb_beta,lb_first_nest_tech,lb_second_nest_tech,lb_prod,lb_T)...]
    #ub = Any[vcat(ub_beta,ub_first_nest_tech,ub_second_nest_tech,ub_prod,ub_T)...]

    lb = Any[vcat(lb_beta,lb_second_nest_tech,lb_prod,lb_T)...]
    ub = Any[vcat(ub_beta,ub_second_nest_tech,ub_prod,ub_T)...]
    
    halton_samples = QuasiMonteCarlo.sample(n, lb, ub, HaltonSample())  # n rows, 8 cols
    return [halton_samples[:,i] for i in range(1,n)]
    # This will create a vector of 100 tuples, each with 8 parameters
    #return [(halton_samples[1,i],halton_samples[2,i],halton_samples[3:2+(S),1]/sum(halton_samples[3:2+(S),1]),halton_samples[(S+3):(size(ub_prod)[1]+S+2),i],halton_samples[(size(ub_prod)[1]+(S+3)):(size(ub_prod)[1]+size(lb_T)[1]+S+2),i]) for i in 1:(n-1)]
end



simulation = false
n = 50000
print("Starting simulation")
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
params_matrix = hcat([collect(unpack_params(params)) for params in params_list]...)
# Create a DataFrame
param_names = ["beta", "second_nest_tech", "prod", "T"]  # Column names for the parameters
df = DataFrame(params_matrix', :auto)  # Transpose to get parameters as rows
rename!(df, param_names)  # Rename columns to match parameter names
score = [score[1] != nothing ? score[1][1] : missing for score in results]


##### Plot output ########

df[!,"score_index"] = vec(1:length(score))
df[!, "score"] = score


# Add the new columns to the DataFrame
df[!, :score] .= map(x -> x === missing ? Inf : x, df[!, :score])

# Now sort by 'score' column
sort!(df, :score)
first_loop = true
if first_loop
    CSV.write(joinpath(folder,"parameters.csv"),df)
else
    CSV.write(joinpath(folder,"parameters_2.csv"),df)
end

###### Histograms #######

# Display the updated DataFrame
best_params = CSV.read(joinpath(folder,"parameters.csv"),DataFrame)
min_vec = [minimum(best_params[!, col]) for col in param_names]
max_vec = [maximum(best_params[!, col]) for col in param_names]
best_index = best_params[1,:score_index]



npzwrite(joinpath("./reporting", "best_params.npy"), params_list[best_index])

# Create individual histograms with LaTeX titles
p1 = histogram(vec(emp_chi_js), alpha=0.5, bins=30, label="Empirical", color=:blue, title="chi_{js}")
histogram!(p1, vec(results[best_index][2][1]), alpha=0.5, bins=30, label="Simulated", color=:red)

p2 = histogram(vec(emp_pi_jA), alpha=0.5, bins=30, label="Empirical", color=:blue, title="pi_{jA}")
histogram!(p2, vec(results[best_index][2][2]), alpha=0.5, bins=30, label="Simulated", color=:red)

p3 = histogram(input_share[2:end], alpha=0.5, bins=30, label="Empirical", color=:blue, title="pi_{sA}")
histogram!(p3, results[best_index][2][4], alpha=0.5, bins=30, label="Simulated", color=:red)

# Combine into a 2x2 subplot layout
plot(p1,p2,p3, layout=(2,2), size=(800,800))
savefig(joinpath("./reporting","dashboard.png"))


npzwrite(joinpath(folder, "pi_jA.npy"), results[best_index][2][2])
npzwrite(joinpath(folder, "productivity.npy"), unpack_params(params_list[best_index])[4])


beta,input_share_tech,productivity_,T_ = unpack_params(params_list[best_index])
data = beta,input_share_tech,productivity_,T_

@everywhere include("model_CP.jl")
low = SMM(vcat([beta*0.5]..., data[2]..., data[3]..., data[4]...),true)
npzwrite(joinpath(folder, "M_ij_low_trade_cost.npy"), low)
high = SMM(vcat([beta*1.5]..., data[2]..., data[3]..., data[4]...),true)
npzwrite(joinpath(folder, "M_ij_high_trade_cost.npy"), high)


# Report



function matrix_report(mat,include_n_zero = true)
    vals = vec(mat)
    n_zeros = count(==(0), vals)
    nonzero_vals = filter(!=(0), vals)

    if isempty(nonzero_vals)
        error("Matrix contains only zeros; cannot compute statistics on non-zero values.")
    end
    if include_n_zero
        return (
        n_zeros = n_zeros,
        q1 = quantile(nonzero_vals, 0.25),
        median = quantile(nonzero_vals, 0.50),
        q3 = quantile(nonzero_vals, 0.75),
        max_val = maximum(nonzero_vals),
    )
    else
        return (
            q1 = quantile(nonzero_vals, 0.25),
            median = quantile(nonzero_vals, 0.50),
            q3 = quantile(nonzero_vals, 0.75),
            max_val = maximum(nonzero_vals),
        )
    end
end


function add_first_element(v::Vector{Float64})
    first_element = 1.0 - sum(v)
    return [first_element; v]
end


function generate_dashboard_report(
    chi_js,pi_jA,reg_,input_share_,labor_share_,
    output_file::String = "./reporting/report.txt"
)
    # Chi_js summary table
    chi_emp,chi_sim = chi_js
    pi_jA_emp,pi_jA_sim = pi_jA
    reg_emp,reg_sim = reg_
    input_share_emp,input_share_sim=input_share_
    labor_share_emp,labor_share_sim=labor_share_

    chi_df = DataFrame(
        metric = ["Number of zeros", "Q1", "Median", "Q3", "Max value"],
        empirical = [
            chi_emp.n_zeros,
            chi_emp.q1,
            chi_emp.median,
            chi_emp.q3,
            chi_emp.max_val
        ],
        simulated = [
            chi_sim.n_zeros,
            chi_sim.q1,
            chi_sim.median,
            chi_sim.q3,
            chi_sim.max_val
        ]
    )
    pi_jA_df = DataFrame(
        metric = [ "Q1", "Median", "Q3", "Max value"],
        empirical = [
            pi_jA_emp.q1,
            pi_jA_emp.median,
            pi_jA_emp.q3,
            pi_jA_emp.max_val
        ],
        simulated = [
            pi_jA_sim.q1,
            pi_jA_sim.median,
            pi_jA_sim.q3,
            pi_jA_sim.max_val
        ]
    )

    sectors = sort(unique(CSV.read(joinpath(folder, "filter_N_upstream.csv"), DataFrame)[!, "A129"]))
    input_share_df = DataFrame(
        metric = sectors,
        empirical =input_share_emp,
        simulated = input_share_sim
    )


    open(output_file, "w") do io
        println(io, "===========================\n     MODEL DIAGNOSTICS REPORT\n===========================\n")

        println(io, ">> Chi_js (quartile are for the distribution without zeros):\n")
        for row in eachrow(chi_df)
            println(io, @sprintf("%-15s  Empirical: %8.3f  |  Simulated: %8.3f",
                row.metric, row.empirical, row.simulated))
        end

        println(io, "\n>> pi_jA :\n")
        for row in eachrow(pi_jA_df)
            println(io, @sprintf("%-15s  Empirical: %8.3f  |  Simulated: %8.3f",
                row.metric, row.empirical, row.simulated))
        end

        println(io, "\n>> Input share: \n")
        for row in eachrow(input_share_df)
            println(io, @sprintf("%-15s  Empirical: %8.3f  |  Simulated: %8.3f",
                row.metric, row.empirical, row.simulated))
        end

        
        #println(io, "\n>> Labor share:\n")
        #println(io, @sprintf("%-15s  Empirical: %8.4f  |  Simulated: %8.4f", "Coefficient", labor_share_emp, labor_share_sim))

        println(io, "\n>> Regression Coefficient:\n")
        println(io, @sprintf("%-15s  Empirical: %8.4f  |  Simulated: %8.4f", "Coefficient", reg_emp, reg_sim))


    end
end

chi_emp_result = matrix_report(emp_chi_js)
chi_sim_result = matrix_report(results[best_index][2][1])
chi_js = [chi_emp_result,chi_sim_result]

pi_jA_emp_result = matrix_report(emp_pi_jA,false)
pi_jA_sim_result = matrix_report(results[best_index][2][2])
pi_jA = [pi_jA_emp_result,pi_jA_sim_result]

reg_emp = reg_coef[1]
reg_sim = results[best_index][2][3][1]
reg_ = [reg_emp,reg_sim]


input_share_ = [input_share,add_first_element(results[best_index][2][4])]
labor_share_ = [labor_share,labor_share]



generate_dashboard_report(chi_js,pi_jA,reg_,input_share_,labor_share_)


