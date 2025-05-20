##### SMM for Spatial Comovement #####
# Author: Swann Chelly 


##################### Packages ###################

using Distributed
using SparseArrays
using Distributions
using Random
using NPZ
using LinearAlgebra




# Hyperparameters



###### Testing environment #####
test = false 
if test


    folder = "./baseline"
    distances = NPZ.npzread(joinpath(folder, "distances.npy"))
    N_downstream_per_region = NPZ.npzread(joinpath(folder,"N_downstream_per_region.npy")) # Should now contain the number of downstream firm per region. 
    filter_N_upstream = NPZ.npzread(joinpath(folder,"filter_N_upstream.npy"))



    labor_share = 0.5 # Share of labor in variable costs
    input_share = reshape([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],1,S)
    seed = 1
    N_rho = 100
    regional_wages = ones(R)

    # Parameters
    beta = 1.
    theta = 2.
    nu_s = ones(S).*2.0 # Variety elasticity of subsitution
    nu = 2. # Input elasticity of substitution across sectors
    lambda = 2. # Labor / CI elasticity of substitution
    sigma = 2. # Demand elasticity of substitution
    productivity = ones(R)
    Random.seed!(seed) # Set seed for reproductibility across simulations. 
    T = ones(S, R) # T_sj: Region level comparative advantages drawn from a log-normal distribution
    T = T.*filter_N_upstream
end

function SMM(params,simulation = false)

    beta,productivity_,T_ = params

    # Set the parameters
    T = ones(S,R)
    T[filter_N_upstream.!=0] = T_
    productivity = ones(R)
    productivity[N_downstream_per_region.!=0] = productivity_
    
    # We initialize main variables used in the simulation
    beta = isa(beta, Float64) ? fill(beta, S) : beta
    tau = isnothing(beta) ? rand(S, R, R) : distances .^ reshape(beta, 1, 1, :)

    # Initialise the upstream firms: Draw productivities 
    frechet_rand = Frechet.(theta, T.^theta)

    # Allocate the output
    upstream_variety_productivity = zeros(S, R, N_rho)
    # Fill z with Frechet draws
    for s in 1:S, r in 1:R
        if filter_N_upstream[s,r] == 1 # Verify the indexing. 
            upstream_variety_productivity[s, r, :] = rand(frechet_rand[s, r], N_rho)
        end
    end

    upstream_variety_productivity = permutedims(upstream_variety_productivity, (3, 2, 1))
    inv_upstream_variety_productivity = upstream_variety_productivity.^(-1)

    tau_reshaped = permutedims(tau,(3,1,2))

    M_jis = zeros(R,R,S) 
    c_i_ = zeros(R)
    for i = 1:length(N_downstream_per_region) # Iterate on downstream regions. 
        if N_downstream_per_region[i] >= 1

            # We compute prices faced by downstream firms in region i
            tau_ = reshape(tau_reshaped[:,: ,i]',1,R,S)
            prices_ = inv_upstream_variety_productivity .* tau_

            # We select the lowest prices per variety and build all nests' price indices
            min_coord_rho = reshape(argmin(prices_,dims = 2),N_rho,S)
            p_si_rho = prices_[min_coord_rho]
            p_is = sum(1/N_rho .* p_si_rho.^(1 .- reshape(nu_s,1,S)),dims = 1).^(1 ./ (1 .- reshape(nu_s,1,S)))
            p_i = sum((p_is .* input_share) .^ (1 - nu)).^(1 ./ (1 - nu))
            c_i = productivity[i]*(labor_share*regional_wages[i]^(1-lambda)  + (1-labor_share)*p_i^(1-lambda))^(1/(1-lambda))
            c_i_[i] = c_i

            # Fill the flows
            for j in 1:R
                tmp = map(x -> x[2] == j ? 1 : 0, min_coord_rho) # On récupère l'ensemble des points 
                tmp = sum(tmp.* 1/N_rho .* input_share .* (1-labor_share) .* (p_si_rho./p_is).^(1 .- reshape(nu_s,1,S)) .* (p_is./p_i).^(1-nu)*(p_i/c_i).^(1-lambda)*c_i^(1-sigma),dims = 1)
                M_jis[j,i,:] = tmp
            end
        end
    end


    price_index = sum(c_i_[N_downstream_per_region.!=0].^(1-sigma)).^(1/(1-sigma))
    M_jis = M_jis/(price_index.^(1-sigma)).*reshape(N_downstream_per_region,1,R) # Since the moments are only shares it is useless.
    if simulation
        return reshape(sum(M_jis,dims = 3),R,R)
    end
    # Build moments
    # M_sj 
    # chi_js = M_{js}/M_{sA}
    M_js = reshape(sum(M_jis,dims = 2),(R,S))
    M_sA = sum(M_js,dims = 1)
    chi_js = M_js./M_sA

    # pi_sA
    pi_sA = M_sA/sum(M_sA)

    # pi_jA: Share of region $i$ in the total purchase of the aerospace industry. 
    M_is = reshape(sum(M_jis,dims = 1),(R,S))
    M_i  = sum(M_is,dims = 2)
    pi_jA = M_i/sum(M_i)
    return chi_js,pi_jA,pi_sA

end


# Then compute scores. 

function loss_function(simulated_moments)
    """
    Compute the loss function between empirical and simulated moments. Weight_matrix is the inverse of the variance covariance matrix of simulated moments. 
    """
    simulated_moments = vcat([vec(simulated_moments[i]) for i in 1:(length(simulated_moments)-1)]...)
    #simulated_moments = vcat([vec(simulated_moments),vec([N])]...)
    N = length(simulated_moments)
    simulated_moments = reshape(simulated_moments,(1,N))
    err = (empirical_moments-simulated_moments)
    # W = isnothing(W) ? I(length(empirical_moments)).*(empirical_moments).^(-1) : W 
    return err*err'
end


function full_SMM(params,simulation = false)
    """
    From the parameters, return the loss and the simulated moments (targeted and untargeted)
    """
    simulated_moments = SMM(params,simulation)
    if simulation 
        return simulated_moments
    else
        return loss_function(simulated_moments),simulated_moments
    end
end



test = false
if test
    folder = "./baseline"
    distances = NPZ.npzread(joinpath(folder, "distances.npy"))
    N_downstream_per_region = NPZ.npzread(joinpath(folder,"N_downstream_per_region.npy")) # Should now contain the number of downstream firm per region. 
    filter_N_upstream = NPZ.npzread(joinpath(folder,"filter_N_upstream.npy"))
    filter_out_reference_region = NPZ.npzread(joinpath(folder,"filter_out_reference_region.npy"))
    emp_chi_si = NPZ.npzread(joinpath(folder,"emp_chi_si.npy"))
    emp_pi_jA = reshape(NPZ.npzread(joinpath(folder,"emp_pi_jA.npy")), (size(emp_chi_si)[2], 1))  # example R=129
    emp_pi_sA = reshape(NPZ.npzread(joinpath(folder,"emp_pi_sA.npy")), (1, size(emp_chi_si)[1]))   # example S=64
    weight_matrix = NPZ.npzread(joinpath(folder,"inv_cov.npy"))
    emp_pi_jA = emp_pi_jA[N_downstream_per_region.!=0]
    emp_chi_si = emp_chi_si[(filter_N_upstream.*filter_out_reference_region).!=0.0]
    if low_high
        emp_rho_si_low = NPZ.npzread(joinpath(folder,"emp_rho_si_low.npy"))
        emp_rho_si_high = NPZ.npzread(joinpath(folder,"emp_rho_si_high.npy"))
        emp_rho_si_low = emp_rho_si_low[(filter_N_upstream.*filter_out_reference_region).!=0.0]
        emp_rho_si_high = emp_rho_si_high[(filter_N_upstream.*filter_out_reference_region).!=0.0]    
        empirical_moments = [emp_chi_si, emp_rho_si_low,emp_rho_si_high]
        empirical_moments = vcat([vec(item) for item in empirical_moments]...)'
    else 
        emp_rho_si = NPZ.npzread(joinpath(folder,"emp_rho_si.npy")) 
        emp_rho_si = emp_rho_si[(filter_N_upstream.*filter_out_reference_region).!=0.0]
        
        empirical_moments = [emp_chi_si, emp_rho_si]
        empirical_moments = vcat([vec(item) for item in empirical_moments]...)'
    end


    omega = copy(emp_pi_sA)
    share_imp_total_cost = 0.35
    foreign_price = 1
    sigma = 2.46
    
    theta,phi_bar,alpha,beta = 6.,2.5,1.,1.
    N_trial_max = 20



    # simulations = [SMM_simulation(seed,[theta,phi_bar,alpha,beta],N_trial_max) for seed in 1:1]
    # simulations = filter(!isnothing, simulations)
    # simulations = mean(simulations)

    # npzwrite(joinpath(folder, "M_ij.npy"), simulations)
    
end
#theta,phi_bar,alpha,beta,mu_T,sigma_T  = (76.0048, 0.5646939444444444, 1.16003, 1.0571728571428571, 1.200055, 1.6154096153846154)
# simulations = SMM_simulation(1,[theta,phi_bar,alpha,beta],N_trial_max) 
