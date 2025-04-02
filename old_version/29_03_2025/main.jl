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

# Load large data once on the main process:
distances_local = NPZ.npzread("./distances.npy")
filter_A_downstream_local = NPZ.npzread("./filter_A_downstream.npy")
filter_N_upstream_local = NPZ.npzread("./filter_N_upstream.npy")

emp_chi_si = NPZ.npzread("./emp_chi_si.npy")
emp_rho_si = NPZ.npzread("./emp_rho_si.npy")
emp_pi_jA = reshape(NPZ.npzread("./emp_pi_jA.npy"), (size(emp_chi_si)[1], 1))  # example R=129
emp_pi_sA = reshape(NPZ.npzread("./emp_pi_sA.npy"), (1, size(emp_chi_si)[2]))   # example S=64

emp_chi_si = emp_chi_si[filter_N_upstream_local'.!=0.0]
emp_rho_si = emp_rho_si[filter_N_upstream_local.!=0.0]
emp_pi_jA = emp_pi_jA[filter_A_downstream_local.!=0]

empirical_moments_local = [emp_chi_si, emp_pi_jA, emp_pi_sA, emp_rho_si]
empirical_moments_local = vcat([vec(item) for item in empirical_moments_local]...)'
#empirical_moments_local = vcat([vec(empirical_moments_local),vec([10990])]...)'

# Then broadcast those large fixed arrays to all workers:
@everywhere const distances = $(distances_local)
@everywhere const filter_A_downstream = $(filter_A_downstream_local)
@everywhere const filter_N_upstream = $(filter_N_upstream_local)
@everywhere const empirical_moments = $(empirical_moments_local)

# Now workers will have them once in memory.

@everywhere function parallel_SMM(params)
    eta,theta,phi_bar,alpha,beta,mu_T,sigma_T,sigma = params
    return full_SMM(eta, theta, phi_bar, alpha, beta, mu_T, sigma_T, sigma)
end

@everywhere function generate_halton_grid(n)
    lb = [0.1, 4, 0.8, 0.9, 0.9, 0.9, 1.5, 2]
    ub = [0.9, 6, 0.9, 1.2, 1.2, 1.3, 2, 2.5]
    
    halton_samples = QuasiMonteCarlo.sample(n, lb, ub, HaltonSample())  # n rows, 8 cols
    
    # This will create a vector of 100 tuples, each with 8 parameters
    return [Tuple(halton_samples[:,i]) for i in 1:(n-1)]
end


@everywhere function parallel_SMM_safe(params,show_err = false)
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
# Generate the grid and run parallel SMM
params_list = generate_halton_grid(20000)
# params_list = [(0.1, 0.5, 0.8, 0.5, 0.5, 1.2, 1.0, 0.5) for _ in 1:2]

t1 = time()
results = pmap(parallel_SMM_safe, params_list)
t1 = time()-t1
print(t1)

if !isempty(workers())
    rmprocs(workers())
end
GC.gc()
# ps aux | grep '[j]ulia' | awk '{print $2}' | xargs kill -9


# Display results
# values = [matrix[1] for matrix in results]
# histogram(values, bins=10, xlabel="Value", ylabel="Frequency", title="Histogram of Vector")

# Convert each tuple of parameters into a row vector
params_matrix = hcat([collect(params) for params in params_list]...)

# Create a DataFrame
param_names = ["eta", "theta", "phi_bar", "alpha", "beta", "mu_T", "sigma_T", "sigma"]  # Column names for the parameters

# Transpose the params_matrix so that each row represents a parameter set
df = DataFrame(params_matrix', :auto)  # Transpose to get parameters as rows
rename!(df, param_names)  # Rename columns to match parameter names

# Calculate the new columns
score = [score !== nothing ? score[1][1] : nothing for score in results]
delta_chi_si = [score !== nothing ? mean((score[2][1] - emp_chi_si) ./ emp_chi_si) : nothing for score in results]
delta_pi_jA = [score !== nothing ? mean((score[2][2] - emp_pi_jA) ./ emp_pi_jA) : nothing for score in results]
delta_pi_sA = [score !== nothing ? mean((score[2][3] - emp_pi_sA') ./ emp_pi_sA') : nothing for score in results]
delta_rho_si = [score !== nothing ? mean((score[2][4] - emp_rho_si) ./ emp_rho_si) : nothing for score in results]
N_firms = [score !== nothing ? score[2][5] : nothing for score in results]

# Add the new columns to the DataFrame
df[!,"score_index"] = vec(1:length(score))
df[!, "score"] = score
df[!, "delta_chi_si"] = delta_chi_si
df[!, "delta_pi_jA"] = delta_pi_jA
df[!, "delta_pi_sA"] = delta_pi_sA
df[!, "delta_rho_si"] = delta_rho_si
df[!, "N_firms"] = N_firms
df[!, :score] .= map(x -> x === nothing ? Inf : x, df[!, :score])
df[!, :N_firms] .= map(x -> x === nothing ? Inf : x, df[!, :N_firms])

# Now sort by 'score' column
sort!(df, :score)

# Display the updated DataFrame
best_params = first(df,50)
min_vec = [minimum(best_params[!, col]) for col in param_names]
max_vec = [maximum(best_params[!, col]) for col in param_names]

best_index = best_params[1,:score_index]

# Create individual histograms with LaTeX titles
p1 = histogram(emp_chi_si, alpha=0.5, bins=30, label="Empirical", color=:blue, normalize=:pdf, title="chi_{si}")
histogram!(p1, results[best_index][2][1], alpha=0.5, bins=30, label="Simulated", color=:red, normalize=:pdf)

p2 = histogram(emp_pi_jA, alpha=0.5, bins=30, label="Empirical", color=:blue, normalize=:pdf, title="pi_{jA}")
histogram!(p2, results[best_index][2][2], alpha=0.5, bins=30, label="Simulated", color=:red, normalize=:pdf)

p3 = histogram(emp_pi_sA', alpha=0.5, bins=30, label="Empirical", color=:blue, normalize=:pdf, title="pi_{sA}")
histogram!(p3, results[best_index][2][3], alpha=0.5, bins=30, label="Simulated", color=:red, normalize=:pdf)

p4 = histogram(emp_rho_si, alpha=0.5, bins=30, label="Empirical", color=:blue, normalize=:pdf, title="rho_{si}")
histogram!(p4, results[best_index][2][4], alpha=0.5, bins=30, label="Simulated", color=:red, normalize=:pdf)

# Combine into a 2x2 subplot layout
plot(p1, p2, p3, p4, layout=(2,2), size=(800,800))
savefig("histograms.pdf")
CSV.write("parameters.csv",best_params)

df



