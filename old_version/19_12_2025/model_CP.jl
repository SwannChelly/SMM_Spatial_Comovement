##### SMM for Spatial Comovement #####
# Author: Swann Chelly 
# The function for the simulation is "SMM"

# Notations: In the paper, we use X_{r'rs} in order to describe the trade flow of goods manufactured by firms from sector s in region r' to downstream firms in region r. 
# In this code, r' is replaced by l for clarity. 

# This code simulate the SMM from the function 'SMM' and compare the simulated moments with the empirical one with 'loss_function'. 
# 'full_SMM' takes a set of parameters and returns the score of the loss function along with simulated moments. Weight matrix is identity. 


# The moments to be identified are: 
# - The aggregate labor share
# - The aggregate share of sector s in the industry’s input purchases
# - The share of the downstream input purchases of sector s sourced from region r′
# - The extensive regression's parameters. 
# - The share of each region in downstream sales

# R is the number of domestic regions.  
# S is the number of upstream sectors. 


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
test = false 
if test
    industry = "aero"
    input_folder = "./baseline_"*industry
    
    coefs = CSV.read(joinpath(input_folder,"stats.csv"), DataFrame)
    distances = NPZ.npzread(joinpath(input_folder, "distances.npy"))
    w_rs = NPZ.npzread(joinpath(input_folder, "w_rs.npy"))
    N_downstream_per_region = NPZ.npzread(joinpath(input_folder,"N_downstream_per_region.npy"))
    filter_N_upstream = NPZ.npzread(joinpath(input_folder,"filter_N_upstream.npy"))
    S,R = size(filter_N_upstream) # Number of upstream sectors and domestic regions. 
    R_downstream = size(N_downstream_per_region[N_downstream_per_region.!=0])[1]
    delta_r = ones(R)
    #empirical_moments = NPZ.npzread(joinpath(input_folder,"empirical_moments.npy"))

    N_rho = 50
    agg_labor_share = coefs[2,"value"]
    agg_industry_share = NPZ.npzread(joinpath(input_folder,"input_share.npy"))
    epsilon = coefs[1,"value"]
    lambda = 0.5
    nu = 0.2
    nu_s = ones(S).*2.5
    theta = 1.768

    w_rs = NPZ.npzread(joinpath(input_folder, "w_rs.npy"))
    regional_wages = NPZ.npzread(joinpath(input_folder, "regional_wages.npy"))

    domestic_share = NPZ.npzread(joinpath(input_folder,"domestic_share.npy"))
    emp_gamma_ls = (NPZ.npzread(joinpath(input_folder,"emp_gamma_ls.npy"))')
    emp_pi_r = NPZ.npzread(joinpath(input_folder,"emp_pi_r.npy"))[2:end]
    reg_coef = NPZ.npzread(joinpath(input_folder,"reg_coef.npy"))
    empirical_moments = [[agg_labor_share],agg_industry_share[2:end],emp_gamma_ls,reg_coef,emp_pi_r]
    empirical_moments = vcat([vec(empirical_moments[i]) for i in 1:(length(empirical_moments)-1)]...)   
    empirical_moments = reshape(empirical_moments,1,length(empirical_moments))
    # Then broadcast those large fixed arrays to all workers:

    function distance_bin(d)
        if 20 < d <= 50
            return 1
        elseif 50 < d <= 100
            return 2
        elseif 100 < d <= 150
            return 3
        elseif 150 < d <= 200
            return 4
        elseif d > 200
            return 5
        else
            return 0   # for ≤ 20, outside bins
        end
    end

    DistBin = Array{Int}(undef, R, R)

    for i in 1:R, j in 1:R
        DistBin[i,j] = distance_bin(distances[i,j])
    end

    function generate_halton_grid(n_needed::Int, batchsize::Int=1024)
        """

        Generate a Halton grid of size P x n with P the size of the parameter set and n the number of parameter sets to test.
        This Halton grid function allows condition on the parameters and is much faster than the previous one. 

        """
        A = copy(N_downstream_per_region[N_downstream_per_region .!= 0])
        A ./= sum(A)
        lb_beta,lb_agg_labor_share_tech,lb_agg_industry_share_tech,lb_prod,lb_T = ones(5).*2,0.8*agg_labor_share,0.8.*agg_industry_share,A,0.1*ones(S*R)
        ub_beta,ub_agg_labor_share_tech,ub_agg_industry_share_tech,ub_prod,ub_T = ones(5).*20,1.2*agg_labor_share,1.2.*agg_industry_share,10*A,100*ones(S*R)

        lb = vcat(lb_beta,lb_agg_labor_share_tech,lb_agg_industry_share_tech,lb_prod,lb_T)
        ub = vcat(ub_beta,ub_agg_labor_share_tech,ub_agg_industry_share_tech,ub_prod,ub_T)

        d = length(lb)
        accepted = Vector{Vector{Float64}}(undef, 0)

        # Create a Halton point generator in dimension d
        hp = HaltonPoint(d)  # yields a lazy sequence of points in [0,1]^d

        idx = 1
        while length(accepted) < n_needed
            # get a batch of raw Halton points
            batch_raw = collect(hp[idx : idx + batchsize - 1])  # Vector of Vectors (each length d)
            # Each point is in [0,1]^d

            for raw in batch_raw
                # scale each component
                scaled = lb .+ (ub .- lb) .* raw

                # apply your condition
                #if (scaled[1] < scaled[4] < scaled[2] < scaled[5] < scaled[3])  # Here we force the exploration of a parameter set where betas are in a specific order.
                if (scaled[1] < scaled[2]) & (scaled[1] < scaled[3]) & (scaled[1] < scaled[4]) & (scaled[1] < scaled[5])  # Condition
                    push!(accepted, scaled)
                    if length(accepted) >= n_needed
                        break
                    end
                end
            end
            idx += batchsize
        end
        return accepted
    end
    params_list = generate_halton_grid(2)
    params = params_list[1]
end



function unpack_params(params)
    """
    This function takes a vector of parameters and returns the parameters as separated variables. 
    """
    
    beta = params[1:5]
    labor_share_tech = params[6]
    input_share_tech = params[7:6+(S)]/sum(params[7:6+(S)])
    productivity_ = params[(S+7):(R_downstream+S+6)]
    T_ = params[(R_downstream+(S+7)):end]

    #input_share_tech = params[2:1+(S)]/sum(params[2:1+(S)])
    #productivity_ = params[(S+2):(R_downstream+S+1)]
    #T_ = params[(R_downstream+(S+2)):end]

    return beta,labor_share_tech,input_share_tech,productivity_,T_
end

function build_tau(beta)
    tau = ones(R,R,S)  # start with 1 everywhere
    for i in 1:R, j in 1:R
        b = DistBin[i,j]
        if b > 0
            for s in 1:S
                tau[i,j,s] += beta[b]
            end
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
        - `gamma_ls`: Share of sector-s inputs sourced from region j.
        - `pi_r`: Importance of downstream region j in the total sales of the downstream industry.
        - `reg_coef`: Estimated elasticity of supply with respect to distance (i.e., a gravity regression coefficient).
        - `input_share`: Share of purchases from sector s in the total intermediate input use of downstream industries.

    """
    # Create the matrix giving for each region the closest region with a downstream industry
    # this matrix will be used for the regression. 
    closest_plant = map(x -> distances[x[1],x[2]],argmin(1 ./(1 ./distances.*(N_downstream_per_region.>0)'),dims = 2))
    
    

    # Set the parameters to the right format
    Random.seed!(50) # For reproducibility 
    beta,labor_share_tech,input_share_tech,productivity_,T_ = unpack_params(params) # Unpack parameters

    # Old version of beta
    #beta = isa(beta, Float64) ? fill(beta, S) : beta 
    #tau = isnothing(beta) ? rand(S, R, R) : distances .^ reshape(beta, 1, 1, :)

    tau = build_tau(beta) # Now tau is built based on bins of distance and the beta coefficients. 
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
            prices_ = inv_upstream_variety_productivity .* tau_.*reshape(w_rs,(1,R,1))

            # We select the lowest prices per variety and build all nests' price indices
            min_coord_rho = reshape(argmin(prices_,dims = 2),N_rho,S)
            p_rs_rho = prices_[min_coord_rho]
            p_rs = sum(1/N_rho .* p_rs_rho.^(1 .- reshape(nu_s,1,S)),dims = 1).^(1 ./ (1 .- reshape(nu_s,1,S)))
            p_r = sum((p_rs) .^ (1 - nu).*input_share_tech).^(1 ./ (1 - nu))
            c_r_ = (labor_share_tech*regional_wages[r]^(1-lambda)  + (1-labor_share_tech)*p_r^(1-lambda))^(1/(1-lambda)) # Here regional_wages != 0 since N_downstream_per_region[r] >= 1  

            #if isinf(c_r)
            #    return p_r
            #end


            c_r[r] = c_r_*(productivity[r]^(-1))
            # We create the trade flows and store the linkages.
            for l in 1:R
                tmp = map(x -> x[2] == l ? 1 : 0, min_coord_rho)  # Here tmp is a dummy variable
                linkages[:,:,l] += tmp # Here linkages is a matrix of size (N_rho,S,R) that contains an integer variable indicating if firm rho in s l suppliers the aerospace industry in region R
                tmp = sum(tmp.* 1/N_rho .* input_share_tech .* (1-labor_share_tech) .* (p_rs_rho./p_rs).^(1 .- reshape(nu_s,1,S)) .* (p_rs./p_r).^(1-nu)*(p_r/c_r_).^(1-lambda)*c_r[r].^epsilon,dims = 1)
                X_lrs[l,r,:] = tmp
            end
            
        end
    end
    
    # Having all prices at all nest, we build the trade flows.
    markup = (epsilon-1)/epsilon
    price_index = sum((c_r[N_downstream_per_region.!=0]*markup).^(epsilon).*delta_r[N_downstream_per_region.!=0]).^(1/epsilon)
    E = 1
    B = (markup/price_index)^(epsilon-1)*E/price_index
    X_lrs = X_lrs.*reshape(N_downstream_per_region.*delta_r,1,R)*B # Since the moments are only shares it is useless.

    # We compute region level quantities
    y_r = zeros(R)
    y_r[N_downstream_per_region.!=0] = c_r[N_downstream_per_region.!=0].^(epsilon-1).*delta_r[N_downstream_per_region.!=0]*B

    if simulation
        return reshape(sum(X_lrs,dims = 3),R,R)
    end
    # Prepare dataframe for regression
    id = 1
    for r in 1:(R) 
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
    #df = filter(row -> row.ze2010 != R, df) # We don't do the regression on foreign firms.
    # 1. Aggregate labor share at the level of the industry \Gamma
    # labor_r : total employement of the downstream industry in region $r$.
    labor_r = zeros(R) 
    labor_r[N_downstream_per_region.!=0]= labor_share_tech.*y_r[N_downstream_per_region.!=0].*(regional_wages[N_downstream_per_region.!=0]./c_r[N_downstream_per_region.!=0]).^(-lambda)
    
    agg_labor_share = sum(regional_wages.*labor_r)/(sum(c_r.*y_r))

    # 2. Aggregate share of sector s in the industry's input purchases.
    # Vérifier la dimension.
    X_ls = reshape(sum(X_lrs,dims = 2),(R,S))
    X_s = sum(X_ls,dims = 1)
    X = sum(X_s)
    agg_industry_share = X_s./X

    # 3. Share of the downstream input purchases of sector s sourced from region l.
    gamma_ls = X_ls./X_s.*(reshape(domestic_share,1,S))

    # 4. Probability that a supplier serves the downstream industry.
    bins = [20,50, 100, 150, 200, Inf]   # bin edges
    df.distance_bin = cut(df.distance, bins, extend=true)
    fixest = reg(df, @formula(supplier ~ distance_bin + size + fe(A129)))
    reg_coef = fixest.coef[1:5] # Change to have the binarised version

    # 5. The share of each region the total employment. 
    pi_r = labor_r[N_downstream_per_region.!=0]./sum(labor_r[N_downstream_per_region.!=0])    
    return [agg_labor_share], agg_industry_share[2:end],gamma_ls,reg_coef,pi_r[2:end]
    


end

# Then compute scores. 
function loss_function(simulated_moments,emp,W)
    """
    Compute the loss between empirical and simulated moments. The weighting matrix is currently set to the identity.
    """
           
    #square_size = sqrt.(vcat([fill(length(vec(m)), length(vec(m))) for m in simulated_moments]...))'
    simulated_moments = vcat([vec(simulated_moments[i]) for i in 1:(length(simulated_moments))]...) 
    N = length(simulated_moments)
    simulated_moments = reshape(simulated_moments,(1,N))
    err = (emp-simulated_moments)#./square_size
    W = isnothing(W) ? I(N) : W 
    return err*W*err'
end


function full_SMM(params,simulation = false,second_stage = false)
    """
    From the parameters, return the loss and the simulated moments (targeted and untargeted)
    """
    simulated_moments = SMM(params,simulation)

    if second_stage
        emp = empirical_moments_reduced
        W = Weight_matrix
        moments = [simulated_moments[3][mask_emp_gamma_ls.!=0]]
    else
        emp = empirical_moments
        W = nothing
        moments = simulated_moments
    end
    if simulation 
        return simulated_moments
    else
        return loss_function(moments,emp,W),simulated_moments
    end
end

