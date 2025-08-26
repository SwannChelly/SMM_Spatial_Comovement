##### SMM for Spatial Comovement #####
# Author: Swann Chelly 
# The function for the simulation is "SMM"

# Notations: In the paper, we use X_{r'rs} in order to describe the trade flow of goods manufactured by firms from sector s in region r' to downstream firms in region r. 
# In this code, r' is replaced by l for clarity. 


##################### Packages ###################

using Distributed
using SparseArrays
using Distributions
using Random
using NPZ
using LinearAlgebra
using QuasiMonteCarlo
using DataFrames
using Optim
using CSV
using FixedEffectModels,RDatasets,CategoricalArrays


###### Testing environment #####
## If test is set to true, we can test the model directly from this code. 
test = true 
if test
    industry = "aero"
    folder = "./baseline_"*industry
    
    coefs = CSV.read(joinpath(folder,"stats.csv"), DataFrame)
    distances = NPZ.npzread(joinpath(folder, "distances.npy"))
    N_downstream_per_region = NPZ.npzread(joinpath(folder,"N_downstream_per_region.npy")) # Should now contain the number of downstream workers per region. 
    filter_N_upstream = NPZ.npzread(joinpath(folder,"filter_N_upstream.npy"))
    #N_downstream_per_region[N_downstream_per_region.!=0] = N_downstream_per_region[N_downstream_per_region.!=0]./N_downstream_per_region[N_downstream_per_region.!=0]
    S,R = size(filter_N_upstream)
    R_downstream = size(N_downstream_per_region[N_downstream_per_region.!=0])[1]
    delta_r = ones(R)
    #empirical_moments = NPZ.npzread(joinpath(folder,"empirical_moments.npy"))

    N_rho = 50
    labor_share = 0.12
    epsilon = coefs[1,"value"]-1
    lambda = 0.5
    nu = 0.9
    nu_s = ones(S).*3.
    theta = 1.768

    regional_wages = ones(R)
    input_share = NPZ.npzread(joinpath(folder,"input_share.npy"))
    emp_chi_js = (NPZ.npzread(joinpath(folder,"emp_chi_js.npy"))')[2:end,:]
    emp_pi_jA = NPZ.npzread(joinpath(folder,"emp_pi_jA.npy"))[2:end]
    reg_coef = [coefs[3,"value"]]
    empirical_moments = [emp_chi_js,emp_pi_jA,reg_coef,input_share[2:end]]
    empirical_moments = vcat([vec(empirical_moments[i]) for i in 1:(length(empirical_moments)-1)]...)   
    empirical_moments = reshape(empirical_moments,1,length(empirical_moments))
    # Then broadcast those large fixed arrays to all workers:

    function distance_bin(d)
        if 50 < d ≤ 100
            return 1
        elseif 100 < d ≤ 150
            return 2
        elseif 150 < d ≤ 200
            return 3
        elseif d > 200
            return 4
        else
            return 0   # for ≤ 50, outside bins
        end
    end

    DistBin = Array{Int}(undef, R, R)

    for i in 1:R, j in 1:R
        DistBin[i,j] = distance_bin(distances[i,j])
    end

    function generate_halton_grid(n)
        """

        Generate a Halton grid of size P x n with P the size of the parameter set and n the number of parameter sets to test.

        """
        # beta,theta,nu_s,nu,lambda,epsilon,productivity,T
        A = copy(N_downstream_per_region[N_downstream_per_region .!= 0])
        A ./= sum(A)
        A = ones(size(A)[1])
        #lb_beta,lb_labor_share_tech,lb_input_share_tech,lb_prod,lb_T, = 0.25,0.5,0.8.*input_share,0.8*A,0.1*ones(S*R)
        #ub_beta,ub_labor_share_tech,ub_input_share_tech,ub_prod,ub_T, = 1,1,1.2.*input_share,1.2*A,20*ones(S*R)

        lb_beta,lb_labor_share_tech,lb_input_share_tech,lb_prod,lb_T, = ones(4)*0.25,0.5,0.8.*input_share,0.8*A,0.1*ones(S*R)
        ub_beta,ub_labor_share_tech,ub_input_share_tech,ub_prod,ub_T, = ones(4)*1,1,1.2.*input_share,1.2*A,20*ones(S*R)


        lb = Any[vcat(lb_beta,lb_labor_share_tech,lb_input_share_tech,lb_prod,lb_T)...]
        ub = Any[vcat(ub_beta,ub_labor_share_tech,ub_input_share_tech,ub_prod,ub_T)...]
        
        halton_samples = QuasiMonteCarlo.sample(n, lb, ub, HaltonSample())  # n rows, 8 cols
        return [halton_samples[:,i] for i in range(1,n)]
        # This will create a vector of 100 tuples, each with 8 parameters
        #return [(halton_samples[1,i],halton_samples[2,i],halton_samples[3:2+(S),1]/sum(halton_samples[3:2+(S),1]),halton_samples[(S+3):(size(ub_prod)[1]+S+2),i],halton_samples[(size(ub_prod)[1]+(S+3)):(size(ub_prod)[1]+size(lb_T)[1]+S+2),i]) for i in 1:(n-1)]
    end


    params_list = generate_halton_grid(2)
    params = params_list[1]

end



function unpack_params(params)
    """
    This function takes a vector of parameters and returns the parameters as separated variables. 
    """
    
    beta = params[1:4]
    labor_share_tech = params[5]
    input_share_tech = params[6:5+(S)]/sum(params[6:5+(S)])
    productivity_ = params[(S+6):(R_downstream+S+5)]
    T_ = params[(R_downstream+(S+6)):end]

    #input_share_tech = params[2:1+(S)]/sum(params[2:1+(S)])
    #productivity_ = params[(S+2):(R_downstream+S+1)]
    #T_ = params[(R_downstream+(S+2)):end]

    return beta,labor_share_tech,input_share_tech,productivity_,T_
end

function build_tau(beta)
    tau = ones(R, R)  # start with 1 everywhere
    for i in 1:R, j in 1:R
        b = DistBin[i,j]
        if b > 0
            tau[i,j] += beta[b]
        end
    end
    return tau
end

function SMM(params,simulation = false)

    """
    Main function to perform the Simulated Method of Moments (SMM) calibration or simulation.

    # Arguments
    - `params::Vector`: A vector containing the model parameters. These include:
        - `beta`: Exponent on distance in the trade cost, capturing the elasticity of trade with respect to distance.
        - `labor_share_tech`: Technological coefficient on labor (Ωᴸ), typically approximated by the labor share.
        - `input_share_tech`: Vector of input-specific technological coefficients (Ωˢ), usually proxied by sectoral input shares.
        - `productivity`: Region-specific productivity parameters of downstream firms.
        - `T`: Matrix (Tₛⱼ) capturing the fundamental Ricardian advantage of region j in producing good s.

    - `simulation::Bool=false`: Optional. If set to `true`, the function returns the full matrix of simulated trade flows. 
    If `false`, the function returns a set of simulated moments.

    # Returns
    - If `simulation == true`: 
        - `trade_flows::Matrix`: Simulated bilateral trade flows between regions and sectors.

    - If `simulation == false`: 
        A named tuple with the following simulated moments:
        - `chi_js`: Share of sector-s inputs sourced from region j.
        - `pi_jA`: Importance of downstream region j in the total sales of the downstream industry.
        - `reg_coef`: Estimated elasticity of supply with respect to distance (i.e., a gravity regression coefficient).
        - `input_share`: Share of purchases from sector s in the total intermediate input use of downstream industries.

    """
    # Create the matrix giving for each region the closest region with a downstream industry
    # this matrix will be used for the regression. 
    closest_plant = map(x -> distances[x[1],x[2]],argmin(1 ./(1 ./distances.*(N_downstream_per_region.>0)'),dims = 2))
    

    # Set the parameters to the right format
    Random.seed!(50) # For reproducibility 
    beta,labor_share_tech,input_share_tech,productivity_,T_ = unpack_params(params) # Unpack parameters
    #labor_share_tech = labor_share # We approximate the labor technological coefficient by the labor share.
    

    # Old version of beta
    #beta = isa(beta, Float64) ? fill(beta, S) : beta 
    #tau = isnothing(beta) ? rand(S, R, R) : distances .^ reshape(beta, 1, 1, :)
    
    productivity = ones(R)
    productivity[N_downstream_per_region.!=0] = productivity_
    input_share_tech = reshape(input_share_tech,1,S)
    T = reshape(T_,S,R)

    # Initialise the upstream firms: Draw productivities 
    frechet_rand = Frechet.(theta, T.^(1/theta))
    upstream_variety_productivity = zeros(S, R, N_rho)
    # Fill z with Frechet draws
    for s in 1:S, r in 1:R
        upstream_variety_productivity[s, r, :] = rand(frechet_rand[s, r], N_rho)  
    end

    upstream_variety_productivity = permutedims(upstream_variety_productivity, (3, 2, 1))
    inv_upstream_variety_productivity = upstream_variety_productivity.^(-1)
    tau_reshaped = permutedims(tau,(3,1,2))    

    # Bellow, we will compute the matching for each downstream region with each sector. First we create 
    # containers in order to store the results of this matching. 

    X_lrs = zeros(R,R,S)  # Trade flows from upstream firms in region l and sector s to downstream firm in region r.
    c_r = zeros(R)        # Marginal cost of production of region r
    linkages = zeros(N_rho,S,R) # Firm level linkages to downstream regions

    # Other containers used for the regression
    sirens = [ i for i in  1:(S*R*N_rho) ]
    sectors = Int[]
    regions = Int[]
    suppliers = Float64[]
    distance = Float64[]
    log_distance = Float64[]
    size_r = Float64[]

    for r = 1:length(N_downstream_per_region) # Per downstream regions. 
        if N_downstream_per_region[r] >= 1    # If there is an downstream industry in region i

            # Compute prices faced by downstream firms in region i
            tau_ = reshape(tau_reshaped[:,: ,r]',1,R,S)
            prices_ = inv_upstream_variety_productivity .* tau_

            # We select the lowest prices per variety and build all nests' price indices
            min_coord_rho = reshape(argmin(prices_,dims = 2),N_rho,S)
            p_rs_rho = prices_[min_coord_rho]
            p_rs = sum(1/N_rho .* p_rs_rho.^(1 .- reshape(nu_s,1,S)),dims = 1).^(1 ./ (1 .- reshape(nu_s,1,S)))
            p_r = sum((p_rs) .^ (1 - nu).*input_share_tech).^(1 ./ (1 - nu))
            c_r_tilde = (labor_share_tech*regional_wages[r]^(1-lambda)  + (1-labor_share_tech)*p_r^(1-lambda))^(1/(1-lambda))
            c_r[r] = c_r_tilde*(productivity[r]^(-1))

            # We create the trade flows and store the linkages.
            for l in 1:R
                tmp = map(x -> x[2] == l ? 1 : 0, min_coord_rho)  # Here tmp is a dummy variable
                linkages[:,:,l] += tmp # Here linkages is a matrix of size (N_rho,S,R) that contains an integer variable indicating if firm rho in s l suppliers the aerospace industry in region R
                tmp = sum(tmp.* 1/N_rho .* input_share_tech .* (1-labor_share_tech) .* (p_rs_rho./p_rs).^(1 .- reshape(nu_s,1,S)) .* (p_rs./p_r).^(1-nu)*(p_r/c_r_tilde).^(1-lambda)*c_r[r].^epsilon,dims = 1)
                X_lrs[l,r,:] = tmp
            end
        end
    end
    markup = (epsilon-1)/epsilon
    # Having all prices at all nest, we build the trade flows.
    price_index = sum((c_r[N_downstream_per_region.!=0]*markup).^(-epsilon).*delta_r[N_downstream_per_region.!=0]).^(-1/epsilon)
    B = ((epsilon-1)/(epsilon*price_index))^(-(1+epsilon))/price_index
    X_lrs = X_lrs.*reshape(N_downstream_per_region.*delta_r,1,R)*B # Since the moments are only shares it is useless.
    y_r = zeros(R)
    y_r[N_downstream_per_region.!=0] = c_r[N_downstream_per_region.!=0].^(-(1+epsilon)).*delta_r[N_downstream_per_region.!=0]*B

    if simulation
        return reshape(sum(X_lrs,dims = 3),R,R)
    end
    # Prepare dataframe for regression
    id = 1
    for r in 1:R
        for s in 1:S
            for i in 1:N_rho
                push!(sectors, s)
                push!(regions, r)
                push!(suppliers, linkages[i, s, r]>0)
                push!(size_r, upstream_variety_productivity[i, r, s])
                push!(distance, closest_plant[r])
                push!(log_distance,log(closest_plant[r]))
                id += 1
            end
        end
    end

    df = DataFrame(
        SIREN = sirens,
        A129 = sectors,
        ze2010 = regions,
        supplier = suppliers,
        size = size_r,
        distance = distance,
        log_distance = log_distance
    )
    # Build moments



    # 1. Aggregate labor share at the level of the industry \Gamma
    # labor_r : total employement of the downstream industry in region $r$.
    labor_r = zeros(R) 
    labor_r[N_downstream_per_region.!=0]= (c_r[N_downstream_per_region.!=0]).^(-(1+epsilon)+lambda).*delta_r[N_downstream_per_region.!=0].*labor_share_tech.*regional_wages[N_downstream_per_region.!=0].^(-lambda)*B
    
    agg_labor_share = sum(regional_wages.*labor_r)/(sum(c_r.*y_r))

    # 2. Aggregate share of sector s in the industry's input purchases.
    # Vérifier la dimension.
    X_ls = reshape(sum(X_lrs,dims = 2),(R,S))
    X_s = sum(X_ls,dims = 1)
    X = sum(X_s)
    agg_industry_share = X_s./X

    # 3. Share of the downstream input purchases of sector s sourced from region l.
    gamma_ls = X_ls./X_s

    # 4. Probability that a supplier serves the downstream industry.
    bins = [50, 100, 150, 200, Inf]   # bin edges
    df.distance_bin = cut(df.distance, bins, extend=true)
    fixest = reg(df, @formula(supplier ~ distance_bin + size + fe(A129)))
    reg_coef = fixest.coef[1:4] # Change to have the binarised version

    # 5. The share of each region the total employment. 
    pi_r = labor_r[N_downstream_per_region.!=0]./sum(labor_r[N_downstream_per_region.!=0])    
    
    return agg_labor_share, agg_industry_share[2:end],gamma_ls[2:end,:],reg_coef,pi_r[2:end]


end

# Then compute scores. 
function loss_function(simulated_moments)
    """
    Compute the loss between empirical and simulated moments. The weighting matrix is currently set to the identity.
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

