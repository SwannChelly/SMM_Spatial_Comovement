
# ps aux | grep '[j]ulia' | awk '{print $2}' | xargs kill -9

import Pkg; Pkg.add("QuasiMonteCarlo")
import Pkg; Pkg.add("StatsPlots")
import Pkg; Pkg.add("DataFrames")
import Pkg; Pkg.add("NPZ")
import Pkg; Pkg.add("Distributions")
import Pkg; Pkg.add("Plots")
import Pkg; Pkg.add("CSV")

using Distributed
using NPZ
using QuasiMonteCarlo
using StatsPlots
using DataFrames
using Plots
using CSV

addprocs(100)

@everywhere using NPZ
@everywhere include("model_loop.jl")
############## Load Parameters #################
@everywhere const low_high = $(false)
@everywhere const reduced = $(false)
first_loop = false


if low_high
    folder = "./bins"
else
    folder = "./baseline"
end


distances_local = NPZ.npzread(joinpath(folder, "distances.npy"))
filter_A_downstream_local = NPZ.npzread(joinpath(folder,"filter_A_downstream.npy"))
filter_N_upstream_local = NPZ.npzread(joinpath(folder,"filter_N_upstream.npy"))
filter_out_reference_region_local = NPZ.npzread(joinpath(folder,"filter_out_reference_region.npy"))

emp_chi_si = NPZ.npzread(joinpath(folder,"emp_chi_si.npy"))
emp_pi_jA = reshape(NPZ.npzread(joinpath(folder,"emp_pi_jA.npy")), (size(emp_chi_si)[2], 1))  # example R=129
emp_pi_sA = reshape(NPZ.npzread(joinpath(folder,"emp_pi_sA.npy")), (1, size(emp_chi_si)[1]))   # example S=64
W_local = NPZ.npzread(joinpath(folder,"inv_cov.npy"))
emp_pi_jA = emp_pi_jA[filter_A_downstream_local.!=0]

emp_chi_si = emp_chi_si[(filter_N_upstream_local.*filter_out_reference_region_local).!=0.0]
if low_high
    emp_rho_si_low = NPZ.npzread(joinpath(folder,"emp_rho_si_low.npy"))
    emp_rho_si_high = NPZ.npzread(joinpath(folder,"emp_rho_si_high.npy"))
    emp_rho_si_low = emp_rho_si_low[(filter_N_upstream_local.*filter_out_reference_region_local).!=0.0]
    emp_rho_si_high = emp_rho_si_high[(filter_N_upstream_local.*filter_out_reference_region_local).!=0.0]    
    empirical_moments_local = [emp_chi_si, emp_rho_si_low,emp_rho_si_high]
    empirical_moments_local = vcat([vec(item) for item in empirical_moments_local]...)'
else 
    emp_rho_si = NPZ.npzread(joinpath(folder,"emp_rho_si.npy")) 
    emp_rho_si = emp_rho_si[(filter_N_upstream_local.*filter_out_reference_region_local).!=0.0]
    
    empirical_moments_local = [emp_chi_si, emp_rho_si]
    empirical_moments_local = vcat([vec(item) for item in empirical_moments_local]...)'
end

# Then broadcast those large fixed arrays to all workers:
@everywhere const distances = $(distances_local)
@everywhere const filter_A_downstream = $(filter_A_downstream_local)
@everywhere const filter_N_upstream = $(filter_N_upstream_local)
@everywhere const filter_out_reference_region = $(filter_out_reference_region_local)
@everywhere const empirical_moments = $(empirical_moments_local)
@everywhere const omega = $(copy(emp_pi_sA))
@everywhere const share_imp_total_cost = $(0.35)
@everywhere const foreign_price = $(1)
@everywhere const sigma = $(2.46)
@everywhere const weight_matrix = $(W_local)

@everywhere function parallel_SMM(params)
    theta,phi_bar,alpha,beta,mu_T,sigma_T = params
    return full_SMM(theta, phi_bar, alpha, beta, mu_T, sigma_T)
end

@everywhere function generate_halton_grid(n)
    #    theta,phi_bar,alpha,beta,mu_T,sigma_T 
    lb = [4, 0.8, 0.9, 0.9, 0.9, 1.5]
    ub = [6, 0.9, 1.2, 1.2, 1.3, 2]
    
    halton_samples = QuasiMonteCarlo.sample(n, lb, ub, HaltonSample())  # n rows, 8 cols
    
    # This will create a vector of 100 tuples, each with 8 parameters
    return [Tuple(halton_samples[:,i]) for i in 1:(n-1)]
end

@everywhere function parallel_SMM_safe(params,show_err = true)
    try
        # Perform the actual computation (replace with your actual logic)
        result = parallel_SMM(params)

        return result
    catch e
        # If an error occurs, return a message or a placeholder result
        println("Error occurred with parameters: $params.")
        if show_err
            println(e)
        end
        return nothing  # You can also return an error message or a custom value
    end
end


# First create parameters
if first_loop
    params_list = generate_halton_grid(1000)
    # params_list = [(0.5, 0.8, 0.5, 0.5, 1.2, 1.0, 0.5) for _ in 1:2]
else 
    
    best_params = CSV.read(joinpath(folder,"parameters.csv"),DataFrame)

    center = best_params[1,["theta","phi_bar","alpha","beta","mu_T","sigma_T"]]
    params_list = Any[]
    for i in 1:6
        tmp = range(center[i]*0.95,center[i]*1.05,length = 400)
        for j in tmp
            list = Any[]
            for k in 1:6
                if k == i
                    push!(list,j)
                else
                    push!(list,center[k])
                end
            end
            push!(params_list,list)
        end

    end
end

# Then compute scores. 
t1 = time()
results = pmap(parallel_SMM_safe, params_list)
t1 = time()-t1
print(t1)

if !isempty(workers())
    rmprocs(workers())
end
GC.gc()



# Format scores

params_matrix = hcat([collect(params) for params in params_list]...)
# Create a DataFrame
param_names = ["theta", "phi_bar", "alpha", "beta", "mu_T", "sigma_T"]  # Column names for the parameters
df = DataFrame(params_matrix', :auto)  # Transpose to get parameters as rows
rename!(df, param_names)  # Rename columns to match parameter names
score = [score !== nothing ? score[1][1] : nothing for score in results]


if low_high
    delta_chi_si = [score !== nothing ? mean((score[2][1] - emp_chi_si) ./ emp_chi_si) : nothing for score in results]
    delta_rho_si_low = [score !== nothing ? mean((score[2][2] - emp_rho_si_low) ./ emp_rho_si_low) : nothing for score in results]
    delta_rho_si_high = [score !== nothing ? mean((score[2][3] - emp_rho_si_high) ./ emp_rho_si_high) : nothing for score in results]
    delta_pi_jA = [score !== nothing ? mean((score[2][4] - emp_pi_jA) ./ emp_pi_jA) : nothing for score in results]
    delta_pi_sA = [score !== nothing ? mean((score[2][5] - emp_pi_sA') ./ emp_pi_sA') : nothing for score in results]
    N_firms = [score !== nothing ? score[2][6] : nothing for score in results]
    df[!,"score_index"] = vec(1:length(score))
    df[!, "score"] = score
    df[!, "delta_chi_si"] = delta_chi_si
    df[!, "delta_rho_si_low"] = delta_rho_si_low
    df[!, "delta_rho_si_high"] = delta_rho_si_high
    df[!, "delta_pi_jA"] = delta_pi_jA
    df[!, "delta_pi_sA"] = delta_pi_sA
    df[!, "N_firms"] = N_firms
else
    delta_chi_si = [score !== nothing ? mean((score[2][1] - emp_chi_si) ./ emp_chi_si) : nothing for score in results]
    delta_rho_si = [score !== nothing ? mean((score[2][2] - emp_rho_si) ./ emp_rho_si) : nothing for score in results]
    delta_pi_jA = [score !== nothing ? mean((score[2][3] - emp_pi_jA) ./ emp_pi_jA) : nothing for score in results]
    delta_pi_sA = [score !== nothing ? mean((score[2][4] - emp_pi_sA') ./ emp_pi_sA') : nothing for score in results]
    N_firms = [score !== nothing ? score[2][5] : nothing for score in results]
    df[!,"score_index"] = vec(1:length(score))
    df[!, "score"] = score
    df[!, "delta_chi_si"] = delta_chi_si
    df[!, "delta_rho_si"] = delta_rho_si
    df[!, "delta_pi_jA"] = delta_pi_jA
    df[!, "delta_pi_sA"] = delta_pi_sA
    df[!, "N_firms"] = N_firms
end

# Add the new columns to the DataFrame
df[!, :score] .= map(x -> x === nothing ? Inf : x, df[!, :score])
df[!, :N_firms] .= map(x -> x === nothing ? Inf : x, df[!, :N_firms])

# Now sort by 'score' column
sort!(df, :score)
if first_loop
    CSV.write(joinpath(folder,"parameters.csv",df))
else
    CSV.write(joinpath(folder,"parameters_2.csv",df))
end

###### Histograms #######

# Display the updated DataFrame
best_params = CSV.read("parameters.csv",DataFrame)
min_vec = [minimum(best_params[!, col]) for col in param_names]
max_vec = [maximum(best_params[!, col]) for col in param_names]
best_index = best_params[1,:score_index]
results[best_index][2][4]
if low_high
    # Create individual histograms with LaTeX titles
    p1 = histogram(emp_chi_si, alpha=0.5, bins=30, label="Empirical", color=:blue, normalize=:pdf, title="chi_{si}")
    histogram!(p1, results[best_index][2][1], alpha=0.5, bins=30, label="Simulated", color=:red, normalize=:pdf)

    p2 = histogram(emp_rho_si_low, alpha=0.5, bins=30, label="Empirical", color=:blue, normalize=:pdf, title="rho_{si}_low")
    histogram!(p2, results[best_index][2][2], alpha=0.5, bins=30, label="Simulated", color=:red, normalize=:pdf)
    
    p3 = histogram(emp_rho_si_high, alpha=0.5, bins=30, label="Empirical", color=:blue, normalize=:pdf, title="rho_{si}_high")
    histogram!(p3, results[best_index][2][3], alpha=0.5, bins=30, label="Simulated", color=:red, normalize=:pdf)

    p4 = histogram(emp_pi_jA', alpha=0.5, bins=30, label="Empirical", color=:blue, normalize=:pdf, title="pi_{jA}")
    histogram!(p4, results[best_index][2][4], alpha=0.5, bins=30, label="Simulated", color=:red, normalize=:pdf)

    p5 = histogram(emp_pi_sA', alpha=0.5, bins=30, label="Empirical", color=:blue, normalize=:pdf, title="pi_{sA}")
    histogram!(p5, results[best_index][2][5], alpha=0.5, bins=30, label="Simulated", color=:red, normalize=:pdf)

    # Combine into a 2x2 subplot layout
    plot(p1,p2,p3,p4,p5, layout=(2,3), size=(800,800))
    savefig(joinpath(folder,"histograms.pdf"))
else
    # Create individual histograms with LaTeX titles
    p1 = histogram(emp_chi_si, alpha=0.5, bins=30, label="Empirical", color=:blue, normalize=:pdf, title="chi_{si}")
    histogram!(p1, results[best_index][2][1], alpha=0.5, bins=30, label="Simulated", color=:red, normalize=:pdf)

    p2 = histogram(emp_rho_si, alpha=0.5, bins=30, label="Empirical", color=:blue, normalize=:pdf, title="rho_{si}")
    histogram!(p2, results[best_index][2][2], alpha=0.5, bins=30, label="Simulated", color=:red, normalize=:pdf)

    p3 = histogram(emp_pi_jA, alpha=0.5, bins=30, label="Empirical", color=:blue, normalize=:pdf, title="pi_{jA}")
    histogram!(p3, results[best_index][2][4], alpha=0.5, bins=30, label="Simulated", color=:red, normalize=:pdf)

    p4 = histogram(emp_pi_sA', alpha=0.5, bins=30, label="Empirical", color=:blue, normalize=:pdf, title="pi_{sA}")
    histogram!(p4, results[best_index][2][4], alpha=0.5, bins=30, label="Simulated", color=:red, normalize=:pdf)

    # Combine into a 2x2 subplot layout
    plot(p1,p2,p3,p4, layout=(2,2), size=(800,800))
    savefig(joinpath(folder,"histograms.pdf"))
end

