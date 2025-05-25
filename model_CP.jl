##### SMM for Spatial Comovement #####
# Author: Swann Chelly 


##################### Packages ###################

using Distributed
using SparseArrays
using Distributions
using Random
using NPZ
using LinearAlgebra
using QuasiMonteCarlo
using DataFrames

using FixedEffectModels,RDatasets



# Hyperparameters



###### Testing environment #####
test = false 
if test
    folder = "./baseline"
    distances = NPZ.npzread(joinpath(folder, "distances.npy"))
    N_downstream_per_region = NPZ.npzread(joinpath(folder,"N_downstream_per_region.npy")) # Should now contain the number of downstream firm per region. 
    filter_N_upstream = NPZ.npzread(joinpath(folder,"filter_N_upstream.npy"))
    #empirical_moments = NPZ.npzread(joinpath(folder,"empirical_moments.npy"))

    S,R = size(filter_N_upstream)
    input_share = reshape(ones(S)/S,1,S)#NPZ.npzread(joinpath(folder,"input_share.npy"))
    regional_wages = ones(R)

    # Then broadcast those large fixed arrays to all workers:

    N_rho = 100
    labor_share = 0.5
    sigma = 2.46
    lambda = 0.5
    nu = 0.001
    nu_s = ones(S).*1.1 
    theta = 1.768 

    function generate_halton_grid(n)
    # beta,theta,nu_s,nu,lambda,sigma,productivity,T
        lb_beta,lb_first_nest_tech,lb_second_nest_tech,lb_prod,lb_T, = 0.5,0.8*labor_share,0.8.*input_share,0.5*ones(R),0.5*ones(S*R)
        ub_beta,ub_first_nest_tech,ub_second_nest_tech,ub_prod,ub_T, = 1.5,1.2*labor_share,1.2.*input_share,ones(S),1.5*ones(R),1.5*ones(S*R)

        lb_prod = lb_prod[N_downstream_per_region.!=0]
        ub_prod = ub_prod[N_downstream_per_region.!=0]

        lb = Any[vcat(lb_beta,lb_first_nest_tech,lb_second_nest_tech,lb_prod,lb_T)...]
        ub = Any[vcat(ub_beta,ub_first_nest_tech,ub_second_nest_tech,ub_prod,ub_T)...]

        halton_samples = QuasiMonteCarlo.sample(n, lb, ub, HaltonSample())  # n rows, 8 cols
    
        # This will create a vector of 100 tuples, each with 8 parameters
        return [(halton_samples[1,i],halton_samples[2,i],halton_samples[3:2+(S),1]/sum(halton_samples[3:2+(S),1]),halton_samples[(S+3):(size(ub_prod)[1]+S+2),i],halton_samples[(size(ub_prod)[1]+(S+3)):(size(ub_prod)[1]+size(lb_T)[1]+S+2),i]) for i in 1:(n-1)]
    end

    params = generate_halton_grid(10000)[1000]

end




function SMM(params,simulation = false)
    beta,labor_share_tech,input_share_tech,productivity_,T_ = params
    closest_plant = map(x -> distances[x[1],x[2]],argmin(1 ./(1 ./distances.*(N_downstream_per_region.>0)'),dims = 2))

    # Set the parameters
    T = reshape(T_,S,R)
    productivity = ones(R)
    productivity[N_downstream_per_region.!=0] = productivity_
    input_share_tech = reshape(input_share_tech,1,S)

    # We initialize main variables used in the simulation
    beta = isa(beta, Float64) ? fill(beta, S) : beta
    tau = isnothing(beta) ? rand(S, R, R) : distances .^ reshape(beta, 1, 1, :)

    # Initialise the upstream firms: Draw productivities 
    frechet_rand = Frechet.(theta, T.^theta)

    # Allocate the output
    upstream_variety_productivity = zeros(S, R, N_rho)
    # Fill z with Frechet draws
    for s in 1:S, r in 1:R
        upstream_variety_productivity[s, r, :] = rand(frechet_rand[s, r], N_rho)  
    end

    upstream_variety_productivity = permutedims(upstream_variety_productivity, (3, 2, 1))
    inv_upstream_variety_productivity = upstream_variety_productivity.^(-1)

    tau_reshaped = permutedims(tau,(3,1,2))

    M_jis = zeros(R,R,S) 
    c_i_ = zeros(R)
    linkages = zeros(N_rho,S,R)

    # Prepare containers
    sirens = [ i for i in  1:(S*R*N_rho) ]
    sectors = Int[]
    regions = Int[]
    suppliers = Float64[]
    distance = Float64[]
    size_i = Float64[]


    for i = 1:length(N_downstream_per_region) # Iterate on downstream regions. 
        if N_downstream_per_region[i] >= 1

            # We compute prices faced by downstream firms in region i
            tau_ = reshape(tau_reshaped[:,: ,i]',1,R,S)
            prices_ = inv_upstream_variety_productivity .* tau_

            # We select the lowest prices per variety and build all nests' price indices
            min_coord_rho = reshape(argmin(prices_,dims = 2),N_rho,S)
            p_si_rho = prices_[min_coord_rho]
            p_is = sum(1/N_rho .* p_si_rho.^(1 .- reshape(nu_s,1,S)),dims = 1).^(1 ./ (1 .- reshape(nu_s,1,S)))
            p_i = sum((p_is .* input_share_tech) .^ (1 - nu)).^(1 ./ (1 - nu))
            c_i = productivity[i]*(labor_share_tech*regional_wages[i]^(1-lambda)  + (1-labor_share_tech)*p_i^(1-lambda))^(1/(1-lambda))
            c_i_[i] = c_i

            # Fill the flows
            for j in 1:R
                tmp = map(x -> x[2] == j ? 1 : 0, min_coord_rho) # On récupère l'ensemble des points 
                linkages[:,:,j] += tmp
                tmp = sum(tmp.* 1/N_rho .* input_share_tech .* (1-labor_share_tech) .* (p_si_rho./p_is).^(1 .- reshape(nu_s,1,S)) .* (p_is./p_i).^(1-nu)*(p_i/c_i).^(1-lambda)*c_i^(1-sigma),dims = 1)
                M_jis[j,i,:] = tmp
            end
        end
    end

    id = 1
    for r in 1:R
        for s in 1:S
            for i in 1:N_rho
                push!(sectors, s)
                push!(regions, r)
                push!(suppliers, linkages[i, s, r]>0)
                push!(size_i, inv_upstream_variety_productivity[i, r, s])
                push!(distance, closest_plant[r])
                id += 1
            end
        end
    end

    df = DataFrame(
        SIREN = sirens,
        A129 = sectors,
        ze2010 = regions,
        supplier = suppliers,
        size = size_i,
        min_distance = distance/100
    )



    price_index = sum(c_i_[N_downstream_per_region.!=0].^(1-sigma)).^(1/(1-sigma))
    C_D = (sigma/(sigma-1))^(-sigma)/(price_index.^(1-sigma))
    M_jis = M_jis*C_D.*reshape(N_downstream_per_region,1,R) # Since the moments are only shares it is useless.
    if simulation
        return reshape(sum(M_jis,dims = 3),R,R)
    end

    # Build moments
    # M_sj 
    # chi_js = M_{js}/M_{sA}
    M_js = reshape(sum(M_jis,dims = 2),(R,S))
    M_sA = sum(M_js,dims = 1)
    chi_js = M_js./M_sA

    # pi_jA: Share of region $i$ in the total purchase of the aerospace industry. 
    M_is = reshape(sum(M_jis,dims = 1),(R,S))
    M_i  = sum(M_is,dims = 2)
    pi_jA = M_i/sum(M_i)
    pi_jA = pi_jA[pi_jA .!= 0]

    fixest = reg(df, @formula(supplier ~ min_distance + fe(A129)))
    reg_coef = fixest.coef[1]

    # Labor: l_i = N_i^D x Labor share x C_D x c_i^{1-σ + 1-λ}
    L = sum((N_downstream_per_region.*c_i_.^(1-sigma + 1-lambda))[N_downstream_per_region.!=0])
    M = sum(M_jis)
    labor_share = L/(L+M)
    input_share = M_sA/M
    
    return chi_js[2:end,:],pi_jA[2:end],[reg_coef],input_share[2:end],[labor_share]


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
