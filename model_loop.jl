##### SMM for Spatial Comovement #####
# Author: Swann Chelly 
# Latest update: Build model and trade flow 24/03/2025
# List of concerns
## How to add the outside sector ? 
## Choisir un nombre d'entreprises et les tirer selon Gaubert 2021 ? 
## Trouver les distributions dans lesquels tirer (grille. )

# To Do: 
# Add the rho_si moment. 
# Import data for testing. 



using Distributed
using SparseArrays
using Distributions
using Random
using NPZ
using LinearAlgebra

############### Define functions to use ###############

 function CES(price_indices,sigma = 2)
    """
    Take price indices and return the CES demand out of those price index. 
    """
    return price_indices.^(-sigma)
end

 function create_sparse_upstream(N_upstream, S, R, N)
    """
    Starting with N_upstream that contains the number of active firm per region, we create an upstream matrix. 
    This is a sparse matrix of size (R,N,S) with N = maximum(N_upstream) that contains 1 when the firm is active. 
    """
    upstream_dense = [i <= N_upstream[s, r] ? 1 : 0 for   r in 1:R , i in 1:N,s in 1:S]
    return sparse(reshape(permutedims(upstream_dense,(3,2,1)), S,:))
end

 function compare_sparse(df1,df2,geq = true)
    """
    Compare if non-zero values of df1 are greater or equal than non-zero values of df2 if geq equals 2. Otherwise compare if those are lower strictly
    """
    rows, cols, nzvals1 = findnz(df1)
    rows, cols, nzvals2 = findnz(df2)
    if geq
        df = sparse(rows, cols, nzvals1.>=nzvals2, size(df1)...)
    else
        df = sparse(rows, cols, nzvals1.<nzvals2, size(df1)...)
    end
    return df
end

 function divide_sparse(df1,df2)
    """
    Divide non-zero values of df1 by those of df2. Built to avoid infinite values. 
    """
    rows, cols, nzvals1 = findnz(df1)
    rows, cols, nzvals2 = findnz(df2)
    df = sparse(rows, cols, nzvals1./nzvals2, size(df1)...)
    return df
end

function from_S_RNN_to_flat(df,S,R,N)
    """
    df: matrix of size (S,RRN)  
    For example contains for each sector and each downstream region the list of possible prices (those that are not equal to 0)
    We change its shape such as it has (S,R,R,N) with (Sector, Downstream region, Upstream region, Matched firms in the downstream region)
    """
    flat = reshape(df, S, R*N,R)
    flat = permutedims(flat,(3,2,1))
    return flat
end

function from_flat_to_structured(flat,S,R,N)
    structured = reshape(flat, R, N, R, S)
    structured = permutedims(structured, (3, 2, 1, 4))
    return structured
end

function remove_inf_sparse(df)
    rows, cols, vals = findnz(df)
    # Identify the positions of `Inf` values
    inf_indices = isinf.(vals)
    # Filter out the `Inf` values and their corresponding row and column indices
    filtered_rows = rows[.!inf_indices]
    filtered_cols = cols[.!inf_indices]
    filtered_vals = vals[.!inf_indices]
    df = sparse(filtered_rows, filtered_cols, filtered_vals, size(df)...)
    return df
end

function random_like_sparse(df)
    """    
    From a sparse matrix, create a similar one with random values. 
    """
    uniform = rand(length(nonzeros(df)))
    rows, cols, _ = findnz(df)
    uniform = sparse(rows, cols, uniform, size(df)...)
    return uniform
end

function from_sparse_to_SRN(df,S,N,R)
    df = reshape(df, S,N,R)
    # # println("rho: ",Base.summarysize(rho_si))
    df = permutedims(df,(3,2,1))
    df = permutedims(reshape(sum(df,dims = 2),(R,S)),(2,1))
    return df
end



# ############### Parameters for testing #####################




############### Model ###############

function SMM(seed,theta,phi_bar,alpha,beta,mu_T,sigma_T,N_trial_max  = 10)
    Times = Any[]
    t1 = time()
    # We initialize main variables used in the simulation
    S,R = size(filter_N_upstream)
    alpha = isa(alpha, Float64) ? fill(alpha, S) : alpha
    beta = isa(beta, Float64) ? fill(beta, S) : beta
    tau = isnothing(alpha) ? rand(S, R, R) : distances .^ reshape(-alpha, 1, 1, :)
    lbd = isnothing(beta) ? rand(S, R, R) : distances .^ reshape(-beta, 1, 1, :)
    seed = isnothing(seed) ? 1 : seed

    # Initialise the firms
    ## Informations related to firms are stored in a sparse matrix of size (S, RN) with N the maximum number of firm in a sector x region

    Random.seed!(seed) # Set seed for reproductibility across simulations. 
    T = exp.(randn(S, R) .* sigma_T .+ mu_T) # T_sj: Region level comparative advantes drawn from a log-normal distribution

    if mean(T)*phi_bar^(-theta) >= 30000
        return nothing
    end
    poisson_dist = Poisson.(T .* phi_bar^(-theta)) # N_si: Number of firms drawn from a Poisson distribution according to region-level comparative advantages. 


    # Used for testing
    # N_upstream = fill(4, S, R)
    # N_upstream[:,end] .= 1
    
    ## filter_N_upstream equals 1 when region sj should be considered in the simulation and 0 otherwise. 
    N_upstream = filter_N_upstream === nothing ? rand.(poisson_dist) : filter_N_upstream .* rand.(poisson_dist)

    
    N = Integer(maximum(N_upstream)) # max(N_si)
    N_firms = sum(N_upstream)        # Number of firms

    if N_firms >= 30000 # This would lead to a very large simulation and therefore we don't consider the parameter set. 
        return nothing
    end

    if N_firms <= 1000 # This would lead to a very large simulation and therefore we don't consider the parameter set. 
        return nothing
    end
    # Upstream (S,RN) sparse matrix: Contains the information of active firm per sector x region. 
    upstream = create_sparse_upstream(N_upstream, S, R, N) 

    # Generate, productivity. Construct firm level prices. Wages are equalized to 1. 
    # Draw pareto for firms and shape it as upstream (sparse (S,RN) matrix)    
    pareto_draws = rand(Pareto(phi_bar, theta), length(nonzeros(upstream)))  
    rows, cols, _ = findnz(upstream) 
    pareto_draws = sparse(rows, cols, pareto_draws, size(upstream)...)
    prices = remove_inf_sparse((pareto_draws).^(-1)) # Competitive equilibrium, prices are wages / productivity. 

    # So far tau is a (R,R,S) matrix with (i,j,s) the trade cost for shipping products of sector s from i to j. 
    # We change it into a (S,R,R) matrix with (s,i,j) the trade cost from shipping products of sectors s from i to j. 
    lbd_reshaped = permutedims(lbd,(3,1,2))
    tau_reshaped = permutedims(tau,(3,1,2))

    # We keep the coordinates in upstream of active firms. 
    rows, cols, _ = findnz(upstream)
    coords_upstream = [(r, c) for (r, c) in zip(rows, cols)]

    price_indices = copy(filter_A_downstream).*1.0
    M_sij_ = zeros((R,R,S)) # We create a blank matrix (Upstream, Downstream, Sector). For a tuple (i,j,s), it will be best serving price of region j for sector s if i is selected. Otherwise 0 
    coords = Any[] # We keep the coordinate of the best price in order to build rho_si
    for j = 1:length(filter_A_downstream) # Iterate on downstream regions. 
        if filter_A_downstream[j] == 1
            
            # For each downstream region j, we create a search cost matrix of same shape than upstream. 
            lbd_ = [lbd_reshaped[s,div.(i-1,N) +1 ,j] for (s,i) in coords_upstream]
            lbd_ = sparse(rows, cols, lbd_, size(upstream)...)
    
            r = compare_sparse(random_like_sparse(prices),lbd_) # Sparse random matching >= Search frictions | Selected set of suppliers for each sector
            N_trial = 0
            while (prod(sum(r,dims=2)) == 0) & (N_trial < N_trial_max) # Simple check to see if the matching return a potential supplier for each sector 
                r = compare_sparse(random_like_sparse(prices),lbd_) # Sparse random matching >= Search frictions | Selected set of suppliers for each sector
                N_trial = N_trial+1
            end
            if N_trial >= N_trial_max
                return nothing 
            end 
            # tau_ = repeat(tau_reshaped[:,:,j],inner = (1,N)) # Prices augmented by trade costs
    
            tau_ = [tau_reshaped[s,div.(i-1,N) +1 ,j] for (s,i) in coords_upstream]
            tau_ = sparse(rows, cols, tau_, size(upstream)...)
            prices_ = tau_.*prices # Prices augmented by trade costs
    
            # Serching for highest search cost
            matching = divide_sparse(r,prices_) # Ensure to divide when prices != 0 among selected suppliers. 
            matching_coord = argmax(matching,dims = 2) # Find the best supplier
            prices_ = prices_[matching_coord] # Extract best prices (augmented by trade costs)
            
            price_index = prod((prices_./omega').^(omega')).*((foreign_price/share_imp_total_cost)^share_imp_total_cost) # Build price index CES 
           
            i = div.(getindex.(matching_coord,2).-1,N) .+ 1# Find the region of the best supplier and update M_sij
            for s = 1:S
                M_sij_[i[s],j,s] = 1 #prices_[s] # Keep the coordinate of the region where there is a supplier for downstream sector j in sector s. 
            end
            price_indices[j] = price_index  
            push!(coords,matching_coord) # Store the coordinate of the best suppliers in the flat, upstream like, format
        end
    end
    B_A = 1.0

    # Build trade flow M_sij. In (i,j,s), so far contains the price of the firm in i that serves j (if selected) and 0 otherwise. 
    # Trade flows are w_sj * q_j p_j = w_sj * p_j^(1-\omega)* B_A
    price_indices_ = copy(price_indices)
    price_indices_[price_indices_.!=0] = price_indices_[price_indices_.!=0].^(1-sigma)
    M_sij = M_sij_.*reshape(omega,(1,1,S)).*(price_indices_).*B_A
    M_sij = ifelse.(isnan.(M_sij), 0.0, M_sij)

    # Build moments
    # M_sj 
    # chi_si = M_{si.}/M_{sA}
    M_si = reshape(sum(M_sij,dims = 2),(R,S))
    M_sA = sum(M_si,dims = 1)
    chi_si = M_si./M_sA
    chi_si = ifelse.(isnan.(chi_si), 0.0, chi_si)

    # pi_sA
    pi_sA = M_sA/sum(M_sA)

    # pi_jA: Share of region $i$ in the total purchase of the aerospace industry. 
    M_sj = reshape(sum(M_sij,dims = 1),(R,S))
    M_j  = sum(M_sj,dims = 2)
    pi_jA = M_j/sum(M_j)
    #return M_sij,price_indices,M_j

    # Using unique kill duplicated suppliers. 
    coords = unique(vcat(coords...))

    rows = [c.I[1] for c in coords]
    cols = [c.I[2] for c in coords]
    vals = ones(length(coords))  # values to place at those coordinates

    # Create sparse matrix that contains 1 when firms are suppliers. 
    rho_si = sparse(rows, cols, vals, S, R*N)

    if low_high
        pareto_median = zeros(S, R)  # Assuming S and R are the dimensions
        # Loop through each s and r in order to compute the median of productivity per region 
        for s in 1:S
            for r in 1:R
                # Get the relevant portion of pareto_draws
                pareto_slice = pareto_draws[s, (r-1)*N+1:(r-1)*N+Int(N_upstream[s,r])]
                
                # Check if pareto_slice is empty
                if nnz(pareto_slice) == 0  # nnz gives the number of stored entries in a SparseVector
                    pareto_sr = 0  # Or some other value to represent an empty slice
                else
                    pareto_sr = median(pareto_slice)
                end

                # Store the result in the matrix
                pareto_median[s, r] = pareto_sr
            end
        end
        # Identify firms within sector x region that are above of bellow the median productivity
        pareto_median = repeat(pareto_median,inner = (1,N)).*upstream
        low = compare_sparse(pareto_draws,pareto_median,false)
        high = compare_sparse(pareto_draws,pareto_median,true)

        rho_si_low = rho_si.*low # suppliers under median 
        rho_si_high = rho_si.*high # suppliers under median 

        low_N = from_sparse_to_SRN(low,S,N,R)
        high_N = from_sparse_to_SRN(high,S,N,R)
        rho_si_low = from_sparse_to_SRN(rho_si_low,S,N,R)
        rho_si_high = from_sparse_to_SRN(rho_si_high,S,N,R)

        # Compute \rho_si for each productivity bin
        rho_si_low = rho_si_low./low_N 
        rho_si_high = rho_si_high./high_N 

        rho_si_low,rho_si_high = ifelse.(isnan.(rho_si_low), 0.0, rho_si_low),ifelse.(isnan.(rho_si_high), 0.0, rho_si_high)
        rho_si_low,rho_si_high = replace(rho_si_low, Inf => 0.0),replace(rho_si_high, Inf => 0.0)

        # Filter on upstream regions and filter out reference region. 
        chi_si = chi_si[(filter_N_upstream'.*filter_out_reference_region') .!=0.0]
        rho_si_low,rho_si_high = rho_si_low[(filter_N_upstream.*filter_out_reference_region) .!=0.0], rho_si_high[(filter_N_upstream.*filter_out_reference_region) .!=0.0]
        pi_jA = pi_jA[filter_A_downstream.!=0]
        pi_sA = reshape(pi_sA,S)
        if reduced 
            mean_chi_si,var_chi_si = mean(chi_si),var(chi_si)
            mean_rho_si_low,var_rho_si_low = mean(rho_si_low),var(rho_si_low)
            mean_rho_si_high,var_rho_si_high = mean(rho_si_high),var(rho_si_high)
            return chi_si,mean_chi_si,var_chi_si,mean_rho_si_low,var_rho_si_low,mean_rho_si_high,var_rho_si_high,pi_jA,pi_sA,N_firms
        else 
            return chi_si,rho_si_low,rho_si_high,pi_jA,pi_sA,N_firms
        end
    else
        rho_si = from_sparse_to_SRN(rho_si,S,N,R)
       
        rho_si = rho_si./N_upstream
        rho_si = ifelse.(isnan.(rho_si), 0.0, rho_si)
        rho_si = replace(rho_si, Inf => 0.0)

        chi_si = chi_si[(filter_N_upstream'.*filter_out_reference_region') .!=0.0]
        rho_si = rho_si[(filter_N_upstream.*filter_out_reference_region) .!=0.0]
        #pi_jA = pi_jA[filter_A_downstream.!=0]
        pi_sA = reshape(pi_sA,S)

    # t = repeat(lbd_reshaped[:,:,1],inner = (1,N))
    # println("rho: ",Base.summarysize(rho_si)/(1024^3))
    # println("test: ",Base.summarysize(t)/(1024^3))
    # println("lbd: ",Base.summarysize(lbd)/(1024^3))
    # println("tau: ",Base.summarysize(tau)/(1024^3))
    # println("upstream: ",Base.summarysize(upstream)/(1024^3))
    # println("N_upstream: ",Base.summarysize(N_upstream)/(1024^3))
    # println("pareto_draws: ",Base.summarysize(prices)/(1024^3))
    # println("prices: ",Base.summarysize(prices)/(1024^3))   

        return chi_si ,rho_si,pi_jA,pi_sA,N_firms#,Times # dont return pi_jA since we dont calibrate it so far
    end
end


function SMM_loop(theta,phi_bar,alpha,beta,mu_T,sigma_T,N_trial_max = 10)
    simulations = [SMM(seed,theta,phi_bar,alpha,beta,mu_T,sigma_T,N_trial_max) for seed in 1:20]
    simulations = filter(!isnothing, simulations)
    if length(simulations) > 1
        moments = Any[]
        for i in 1:length(simulations[1])
            push!(moments,mean(hcat([simulation[i] for simulation in simulations]...)',dims = 1)')
        end
        return moments
    else
        return nothing              
    end
end

# Moments 
## - chi_si: the share of Aerospace industry goods of sector s purchased from region i. 
## - pi_jA: the importance of region j in the total purchase of the aerospace industry
## - pi_sA: the share of purchase of goods from sector s in the total purchase of the aerospace industry 
## - rho_si: the extensive margin | # of suppliers / # of potential suppliers
##      - For foreign, potentiel suppliers are taken from the set of firms of the same sector exporting to France. 


# Estimation procedure. 
## Guide: https://opensourceecon.github.io/CompMethods/struct_est/SMM.html

## Put more simply, you want the random draws for all the simulations to be held constant so that the only thing changing in the minimization problem is the value of the vector of parameters
### Keep seed constant through sampling. 
### Want to reduce the dimension of the moments such as to keep only values that are set to be non zeros. 

function loss_function(simulated_moments)
    # To Do: Make such that the difference is in percentage change. 
    N = simulated_moments[end]
    simulated_moments = vcat([vec(simulated_moments[i]) for i in 1:(length(simulated_moments)-3)]...)
    #simulated_moments = vcat([vec(simulated_moments),vec([N])]...)
    N = length(simulated_moments)
    simulated_moments = reshape(simulated_moments,(1,N))
    err = (empirical_moments-simulated_moments)
    # W = isnothing(W) ? I(length(empirical_moments)).*(empirical_moments).^(-1) : W 
    return err*weight_matrix*err'
end


function full_SMM(theta,phi_bar,alpha,beta,mu_T,sigma_T)
    simulated_moments = SMM_loop(theta,phi_bar,alpha,beta,mu_T,sigma_T)
    if simulated_moments != nothing
        return loss_function(simulated_moments),simulated_moments
    else
        simulated_moments = [nothing for i in 1:6]
        return nothing,simulated_moments
    end
end


###### Testing environment #####

test = false
if test
    low_high = true
    reduced = false
    if low_high
        folder = "./bins"
    else
        folder = "./baseline"
    end
    distances = NPZ.npzread(joinpath(folder, "distances.npy"))
    filter_A_downstream = NPZ.npzread(joinpath(folder,"filter_A_downstream.npy"))
    filter_N_upstream = NPZ.npzread(joinpath(folder,"filter_N_upstream.npy"))
    filter_out_reference_region = NPZ.npzread(joinpath(folder,"filter_out_reference_region.npy"))
    emp_chi_si = NPZ.npzread(joinpath(folder,"emp_chi_si.npy"))
    emp_pi_jA = reshape(NPZ.npzread(joinpath(folder,"emp_pi_jA.npy")), (size(emp_chi_si)[2], 1))  # example R=129
    emp_pi_sA = reshape(NPZ.npzread(joinpath(folder,"emp_pi_sA.npy")), (1, size(emp_chi_si)[1]))   # example S=64
    weight_matrix = NPZ.npzread(joinpath(folder,"inv_cov.npy"))
    emp_pi_jA = emp_pi_jA[filter_A_downstream.!=0]
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

    
end
#theta,phi_bar,alpha,beta,mu_T,sigma_T  = (76.0048, 0.5646939444444444, 1.16003, 1.0571728571428571, 1.200055, 1.6154096153846154)
#theta,phi_bar,alpha,beta,mu_T,sigma_T,sigma = 8.,1.,1.,1.,1.30467,1.76248,2.32003
#chi_si,rho_si_low,rho_si_high,pi_jA,pi_sA,N_firms = full_SMM(theta,phi_bar,alpha,beta,mu_T,sigma_T)


