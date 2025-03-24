using Distributed
using NPZ

addprocs(2)

@everywhere include("model_loop.jl")

# Example to generate an Economy object and display its attributes:
# eco = build_economy(R=3, S=2)
distances = NPZ.npzread("./distances.npy")  # for `.npz`
filter_A_downstream = NPZ.npzread("./filter_A_downstream.npy")  # for `.npz`
filter_N_upstream = NPZ.npzread("./filter_N_upstream.npy")  # for `.npz`


emp_chi_si = NPZ.npzread("./emp_chi_si.npy")
emp_rho_si = NPZ.npzread("./emp_rho_si.npy")
emp_pi_jA = reshape(NPZ.npzread("./emp_pi_jA.npy"),(R,1))
emp_pi_sA = reshape(NPZ.npzread("./emp_pi_sA.npy"),(1,S))

emp_chi_si = emp_chi_si[filter_N_upstream'.!=0.]
emp_rho_si = emp_rho_si[filter_N_upstream.!=0.]
emp_pi_jA = emp_pi_jA[filter_A_downstream.!=0]

empirical_moments = [emp_chi_si,emp_pi_jA,emp_pi_sA,emp_rho_si]
empirical_moments = vcat([vec(item) for item in empirical_moments]...)

inv_W = empirical_moments.^(-1)
inv_W = I(length(inv_W)).*inv_W

t1 = time()
R=129
S=64
eta=0.5
omega=nothing
theta=1.0
phi_bar=0.9
w=nothing
# distances=nothing
alpha=1.0
beta=1.0
# filter_N_upstream=nothing
# filter_A_downstream=ones(Bool, R)
# filter_A_downstream[1] = 0
mu_T=0.0135*100
sigma_T=1.395
sigma=1.0
g = CES


@everywhere function some_expensive_computation(params)
    R,S,eta,omega,theta,phi_bar,w,alpha,beta,mu_T,sigma_T,sigma,distances,filter_N_upstream,filter_A_downstream,g =  params
    return SMM(R,S,eta,omega,theta,phi_bar,w,alpha,beta,mu_T,sigma_T,sigma,distances,filter_N_upstream,filter_A_downstream,g)
end

params_list = [(R,S,eta,omega,theta,phi_bar,w,alpha,beta,mu_T,sigma_T,sigma,distances,filter_N_upstream,filter_A_downstream,nothing) for i in 1:2]


t1 = time()
results = pmap(some_expensive_computation, params_list)
print(time()-t1)




if !isempty(workers())
    rmprocs(workers())
end
GC.gc() 
