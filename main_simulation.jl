
# ps aux | grep '[j]ulia' | awk '{print $2}' | xargs kill -9

#import Pkg; Pkg.add("QuasiMonteCarlo")
#import Pkg; Pkg.add("StatsPlots")
#import Pkg; Pkg.add("DataFrames")
#import Pkg; Pkg.add("NPZ")
#import Pkg; Pkg.add("Distributions")
#import Pkg; Pkg.add("Plots")
#import Pkg; Pkg.add("CSV")

using Distributed
using NPZ
using QuasiMonteCarlo
using StatsPlots
using DataFrames
using Plots
using CSV
using Random

addprocs(100)
Random.seed!(1)
@everywhere using NPZ
@everywhere include("model_loop.jl")
############## Load Parameters #################
@everywhere const low_high = $(true)


if low_high
    folder = "./bins"
else
    folder = "./baseline"
end


distances_local = NPZ.npzread(joinpath(folder, "distances.npy"))
extended_filter_A_downstream_local = NPZ.npzread(joinpath(folder,"extended_filter_A_downstream.npy"))
filter_A_downstream_local = NPZ.npzread(joinpath(folder,"filter_A_downstream.npy"))
filter_N_upstream_local = NPZ.npzread(joinpath(folder,"filter_N_upstream.npy"))
filter_out_reference_region_local = NPZ.npzread(joinpath(folder,"filter_out_reference_region.npy"))
N_si_local = NPZ.npzread(joinpath(folder,"N_si.npy"))
    

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
@everywhere const extended_filter_A_downstream = $(extended_filter_A_downstream_local)
@everywhere const N_si = $(N_si_local)     
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

@everywhere function parallel_SMM(seed)
    theta,phi_bar,alpha,beta = 6.,1,0.9,0.364115
    return SMM_simulation(seed,[theta, phi_bar, alpha, beta])
end


@everywhere function parallel_SMM_safe(seed,show_err = true)
    try
        # Perform the actual computation (replace with your actual logic)
        result = parallel_SMM(seed)

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




# Then compute scores. 
t1 = time()
simulations = pmap(parallel_SMM_safe, 1:100)
t1 = time()-t1
print(t1)




simulations = filter(!isnothing, simulations)
simulations = mean(simulations)

npzwrite(joinpath(folder, "M_ij.npy"), simulations)



# if !isempty(workers())
#     rmprocs(workers())
# end
# GC.gc()
