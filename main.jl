using Distributed
using NPZ

addprocs(100)
@everywhere using NPZ

@everywhere include("model_loop.jl")

# Load large data once on the main process:
distances_local = NPZ.npzread("./distances.npy")
filter_A_downstream_local = NPZ.npzread("./filter_A_downstream.npy")
filter_N_upstream_local = NPZ.npzread("./filter_N_upstream.npy")

emp_chi_si = NPZ.npzread("./emp_chi_si.npy")
emp_rho_si = NPZ.npzread("./emp_rho_si.npy")
emp_pi_jA = reshape(NPZ.npzread("./emp_pi_jA.npy"), (128, 1))  # example R=129
emp_pi_sA = reshape(NPZ.npzread("./emp_pi_sA.npy"), (1, 18))   # example S=64

emp_chi_si = emp_chi_si[filter_N_upstream_local'.!=0.0]
emp_rho_si = emp_rho_si[filter_N_upstream_local.!=0.0]
emp_pi_jA = emp_pi_jA[filter_A_downstream_local.!=0]

empirical_moments_local = [emp_chi_si, emp_pi_jA, emp_pi_sA, emp_rho_si]
empirical_moments_local = vcat([vec(item) for item in empirical_moments_local]...)'

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

# Then just launch your parallel runs:
eta = 0.5
theta = 1.0
phi_bar = 0.9
alpha = 1.0
beta = 1.0
mu_T = 1.35
sigma_T = 1.395
sigma = 1.0  

params_list = [(eta, theta, phi_bar, alpha, beta, mu_T, sigma_T, sigma) for _ in 1:100]
t1 = time()
results = pmap(parallel_SMM, params_list)
print(time()-t1)

if !isempty(workers())
    rmprocs(workers())
end
GC.gc()
# ps aux | grep '[j]ulia' | awk '{print $2}' | xargs kill -9