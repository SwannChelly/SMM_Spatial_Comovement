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

 function geq_sparse(df1,df2)
    """
    Compare if non-zero values of df1 are greater than non-zero values of df2. 
    """
    rows, cols, nzvals1 = findnz(df1)
    rows, cols, nzvals2 = findnz(df2)
    df = sparse(rows, cols, nzvals1.>=nzvals2, size(df1)...)
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



############### Parameters for testing #####################

# Example to generate an Economy object and display its attributes:
# distances = NPZ.npzread("./distances.npy")  # for `.npz`
# filter_A_downstream = NPZ.npzread("./filter_A_downstream.npy")  # for `.npz`
# filter_N_upstream = NPZ.npzread("./filter_N_upstream.npy")  # for `.npz`



# t1 = time()
# R=size(distances)[1]
# S=size(filter_N_upstream)[1]
# eta=0.5
# omega=nothing
# theta=1.0
# phi_bar=0.9
# w=nothing # As we don't have an outside sector, we dont take into account the wages for now. 
# # distances=nothing
# alpha=1.0
# beta=1.0
# # R = 3
# # S = 2
# # filter_N_upstream=nothing
# # filter_A_downstream=ones(Bool, R)
# # filter_A_downstream[1] = 0
# distances = nothing
# mu_T=0.0135*100
# sigma_T=1.395
# sigma=2.0
# g = CES


# # Initialise frictions and parameters
# distances = isnothing(distances) ? begin
#     D = rand(R,R) .+1   
#     (D+D')/2         # multiply by its transpose to ensure symmetry
# end : distances  # use the provided distances matrix if available


# emp_chi_si = NPZ.npzread("./emp_chi_si.npy")
# emp_rho_si = NPZ.npzread("./emp_rho_si.npy")
# emp_pi_jA = reshape(NPZ.npzread("./emp_pi_jA.npy"), (size(emp_chi_si)[2], 1))  # example R=129
# emp_pi_sA = reshape(NPZ.npzread("./emp_pi_sA.npy"), (1, size(emp_chi_si)[1]))   # example S=64

# emp_chi_si = emp_chi_si[filter_N_upstream.!=0.0]
# emp_rho_si = emp_rho_si[filter_N_upstream.!=0.0]
# emp_pi_jA = emp_pi_jA[filter_A_downstream.!=0]

# empirical_moments_local = [emp_chi_si, emp_pi_jA, emp_pi_sA, emp_rho_si]
# empirical_moments_local = vcat([vec(item) for item in empirical_moments_local]...)'
# empirical_moments = vcat([vec(empirical_moments_local),vec([10990])]...)

# omega = emp_pi_sA
# theta,phi_bar,alpha,beta,mu_T,sigma_T,sigma = 8.,1.,1.,1.,1.30467,1.76248,2.32003
# foreign_price = 1
# share_imp_total_cost = 0.38


############### Model ###############

function SMM(seed,theta,phi_bar,alpha,beta,mu_T,sigma_T,N_trial_max  = 10)
    Times = Any[]
    t1 = time()
    # For testing
    # distances = reshape(collect(2:(R*R + 1)), R, R).*1.0
    S,R = size(filter_N_upstream)
    alpha = isa(alpha, Float64) ? fill(alpha, S) : alpha
    beta = isa(beta, Float64) ? fill(beta, S) : beta
    tau = isnothing(alpha) ? rand(S, R, R) : distances .^ reshape(-alpha, 1, 1, :)
    lbd = isnothing(beta) ? rand(S, R, R) : distances .^ reshape(-beta, 1, 1, :)
    seed = isnothing(seed) ? 1 : seed

    # Initialise the firms
    ## We will use the upstream variable in the rest of the simulation for ease of computation. 
    ## We assume that in each region there is at most N firm alive. For a region R the actual number of firms that are alive is given by self.N_upstream
    ## Then we sort them for each sector on a single line in the upstream array (of size S x 1 x RN)
    Random.seed!(seed)
    T = exp.(randn(S, R) .* sigma_T .+ mu_T) # T_sj: Region level comparative advantes
    poisson_dist = Poisson.(T .* phi_bar^(-theta))
    # Used for testing
    # N_upstream = fill(4, S, R)
    # N_upstream[:,end] .= 1
    # N_si: Number of firms drawn from a Poisson distribution according to region-level comparative advantages. 
    ## We set manually the regions where there are no firms if filter_N_upstream is given. 
    N_upstream = filter_N_upstream === nothing ? rand.(poisson_dist) : filter_N_upstream .* rand.(poisson_dist)
    N = Integer(maximum(N_upstream))
    upstream = create_sparse_upstream(N_upstream, S, R, N)
    N_firms = sum(upstream)

    # Generate wages, productivity. Construct firm level prices. 
    # w = isnothing(w) ? abs.(rand(S, R)) : w # w_sr = wage of sector s in region r
    # w_extended = repeat(w, inner=(1, N)) # Extension of this wage fro fitting upstream shape. 

    # Draw pareto for firms and shape it as upstream (sparse (S,RN) matrix)
    pareto_draws = rand(Pareto(theta), length(nonzeros(upstream))) .*phi_bar 
    rows, cols, _ = findnz(upstream) 
    pareto_draws = sparse(rows, cols, pareto_draws, size(upstream)...)
    prices = remove_inf_sparse((pareto_draws).^(-1)) # Competitive equilibrium, prices are wages / productivity. 

    # So far tau is a (R,R,S) matrix with (i,j,s) the trade cost for shipping products of sector s from i to j. 
    # We change it into a (S,R,R) matrix with (s,i,j) the trade cost from shipping products of sectors s from i to j. 
    lbd_reshaped = permutedims(lbd,(3,1,2))
    tau_reshaped = permutedims(tau,(3,1,2))

    rows, cols, _ = findnz(upstream)
    coords_upstream = [(r, c) for (r, c) in zip(rows, cols)]

    price_indices = copy(filter_A_downstream).*1.0 
    M_sij_ = zeros((R,R,S)) # We create a blank matrix (Upstream, Downstream, Sector). For a tuple (i,j,s), it will be best serving price of region j for sector s if i is selected. Otherwise 0 
    coords = Any[] # We keep the coordinate of the best price in order to build rho_si
    for j = 1:length(filter_A_downstream) # Iterate on downstream regions. 
        if filter_A_downstream[j] == 1
            # lbd_ = repeat(lbd_reshaped[:,:,j],inner = (1,N)).*upstream # Frictions to serve region j
            
            lbd_ = [lbd_reshaped[s,div.(i-1,N) +1 ,j] for (s,i) in coords_upstream]
            lbd_ = sparse(rows, cols, lbd_, size(upstream)...)
    
            r = geq_sparse(random_like_sparse(prices),lbd_) # Sparse random matching >= Search frictions | Selected set of suppliers for each sector
            N_trial = 0
            while (prod(sum(r,dims=2)) == 0) & (N_trial < N_trial_max) # Simple check to see if the matching return a potential supplier for each sector 
                r = geq_sparse(random_like_sparse(prices),lbd_) # Sparse random matching >= Search frictions | Selected set of suppliers for each sector
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

    # rho_si | So far doesn't works. Might want to check why ? 
    coords = unique(vcat(coords...))

    rows = [c.I[1] for c in coords]
    cols = [c.I[2] for c in coords]
    vals = ones(length(coords))  # values to place at those coordinates

    # Create sparse matrix
    rho_si = sparse(rows, cols, vals, S, R*N)
    rho_si = reshape(rho_si, S,N,R)
    # println("rho: ",Base.summarysize(rho_si))
    rho_si = permutedims(rho_si,(3,2,1))
    rho_si = permutedims(reshape(sum(rho_si,dims = 2),(R,S)),(2,1))
    rho_si = rho_si./N_upstream
    rho_si = ifelse.(isnan.(rho_si), 0.0, rho_si)
    rho_si = replace(rho_si, Inf => 0.0)

    chi_si = chi_si[filter_N_upstream'.!=0.]
    rho_si = rho_si[filter_N_upstream.!=0.]
    pi_jA = pi_jA[filter_A_downstream.!=0]
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

    

    return chi_si,pi_sA,rho_si,N_firms#,Times # dont return pi_jA since we dont calibrate it so far

end

# t1 = time()
# eta,theta,phi_bar,alpha,beta,mu_T,sigma_T,sigma = 0.3415,8.75738,0.6932,0.650773,1.08351,1.30467,1.76248,2.32003

# SMM(1,eta,theta,phi_bar,alpha,beta,mu_T,sigma_T,sigma)
# t1 = time()-t1
# println(t1)

function SMM_loop(theta,phi_bar,alpha,beta,mu_T,sigma_T,N_trial_max = 10)
    chi_si_ ,pi_sA_ ,rho_si_,N_firms_  = Any[],Any[],Any[],Any[],Any[]
    
    for seed = 1:20
        simulation = SMM(seed,theta,phi_bar,alpha,beta,mu_T,sigma_T,N_trial_max)
        if simulation != nothing
            chi_si,pi_sA,rho_si,N_firms = simulation
            push!(chi_si_,chi_si)
            # push!(pi_jA_,pi_jA)
            push!(pi_sA_,pi_sA)
            push!(rho_si_,rho_si)
            push!(N_firms_,N_firms)
        end
    end
    if length(chi_si_) > 1
        chi_si_ = mean(hcat(chi_si_...)',dims = 1)'
        # pi_jA_ = mean(hcat(pi_jA_...)',dims = 1)'
        pi_sA_ = mean(hcat(pi_sA_...)',dims = 1)'
        rho_si_ = mean(hcat(rho_si_...)',dims = 1)'
        N_firms = mean(N_firms_)
        return chi_si_  ,pi_sA_ ,rho_si_,N_firms
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

function loss_function(simulated_moments,W = nothing)
    # To Do: Make such that the difference is in percentage change. 
    N = simulated_moments[end]
    simulated_moments = vcat([vec(simulated_moments[i]) for i in 1:(length(simulated_moments)-1)]...)
    #simulated_moments = vcat([vec(simulated_moments),vec([N])]...)
    N = length(simulated_moments)
    simulated_moments = reshape(simulated_moments,(1,N))
    err = (empirical_moments-simulated_moments)
    #W = isnothing(W) ? I(N) : W 
    W = I(length(empirical_moments)).*(empirical_moments).^(-1)
    return err*W*err'
end


function full_SMM(theta,phi_bar,alpha,beta,mu_T,sigma_T,W = nothing)
    simulated_moments = SMM_loop(theta,phi_bar,alpha,beta,mu_T,sigma_T)
    if simulated_moments != nothing
        return loss_function(simulated_moments,W),simulated_moments
    else
        simulated_moments = [nothing for i in 1:6]
        return nothing,simulated_moments
    end
end

# simulated_moments = SMM_loop(0.1, 0.5, 0.8, 0.5, 0.5, 1.2, 1.0, 0.5)

# ps aux | grep '[j]ulia' | awk '{print $2}' | xargs kill -9
# seed = 1

# Times = Any[]
# t1 = time()
# # For testing
# #distances = reshape(collect(2:(R*R + 1)), R, R).*1.0
# S,R = size(filter_N_upstream)
# # alpha = 1.
# # beta = 1.
# alpha = isa(alpha, Float64) ? fill(alpha, S) : alpha
# beta = isa(beta, Float64) ? fill(beta, S) : beta
# tau = isnothing(alpha) ? rand(S, R, R) : distances .^ reshape(-alpha, 1, 1, :)
# lbd = isnothing(beta) ? rand(S, R, R) : distances .^ reshape(-beta, 1, 1, :)
# omega = isnothing(omega) ? ones(1,S)./S : omega
# seed = isnothing(seed) ? 1 : seed
# t1 = time()-t1
# push!(Times,t1)
# println("Initialise values: ",t1)
# t1 = time()


# # Initialise the firms
# ## We will use the upstream variable in the rest of the simulation for ease of computation. 
# ## We assume that in each region there is at most N firm alive. For a region R the actual number of firms that are alive is given by self.N_upstream
# ## Then we sort them for each sector on a single line in the upstream array (of size S x 1 x RN)
# Random.seed!(seed)
# T = exp.(randn(S, R) .* sigma_T .+ mu_T) # T_sj: Region level comparative advantes
# poisson_dist = Poisson.(T .* phi_bar^(-theta))
# # Used for testing
# # N_upstream = fill(4, S, R)
# # N_upstream[:,end] .= 1
# # N_si: Number of firms drawn from a Poisson distribution according to region-level comparative advantages. 
# ## We set manually the regions where there are no firms if filter_N_upstream is given. 
# N_upstream = filter_N_upstream === nothing ? rand.(poisson_dist) : filter_N_upstream .* rand.(poisson_dist)
# N = Integer(maximum(N_upstream))
# upstream = create_sparse_upstream(N_upstream, S, R, N)
# N_firms = sum(upstream)

# t1 = time()-t1
# push!(Times,t1)
# println("Initialise N  firms: ",t1)
# t1 = time()

# # Generate wages, productivity. Construct firm level prices. 
# # w = isnothing(w) ? abs.(rand(S, R)) : w # w_sr = wage of sector s in region r
# # w_extended = repeat(w, inner=(1, N)) # Extension of this wage fro fitting upstream shape. 

# # Draw pareto for firms and shape it as upstream (sparse (S,RN) matrix)
# pareto_draws = rand(Pareto(theta), length(nonzeros(upstream))) .*phi_bar 
# rows, cols, _ = findnz(upstream) 
# pareto_draws = sparse(rows, cols, pareto_draws, size(upstream)...)
# prices = remove_inf_sparse((pareto_draws).^(-1)) # Competitive equilibrium, prices are wages / productivity. 


# # So far tau is a (R,R,S) matrix with (i,j,s) the trade cost for shipping products of sector s from i to j. 
# # We change it into a (S,R,R) matrix with (s,i,j) the trade cost from shipping products of sectors s from i to j. 
# lbd_reshaped = permutedims(lbd,(3,1,2))
# tau_reshaped = permutedims(tau,(3,1,2))

# t1 = time()-t1
# push!(Times,t1)
# println("Initialise pareto: ",t1)
# t1 = time()


# rows, cols, _ = findnz(upstream)
# coords = [(r, c) for (r, c) in zip(rows, cols)]
# i = div.(cols.-1,N) .+ 1

# price_indices = copy(filter_A_downstream).*1.0 
# M_sij = zeros((R,R,S)) # We create a blank matrix (Upstream, Downstream, Sector). For a tuple (i,j,s), it will be best serving price of region j for sector s if i is selected. Otherwise 0 
# #coords = Any[] # We keep the coordinate of the best price in order to build rho_si
# j = 1
# t1 = time()
# # lbd_1 = repeat(lbd_reshaped[:,:,j],inner = (1,N)).*upstream # Frictions to serve region j

# lbd_ = [lbd_reshaped[s,div.(i-1,N) +1 ,j] for (s,i) in coords]
# lbd_2 = sparse(rows, cols, lbd_, size(upstream)...)

# t1 = time()-t1
# push!(Times,t1)
# println("Initialise pareto: ",t1)



# r = geq_sparse(random_like_sparse(prices),lbd_) # Sparse random matching >= Search frictions | Selected set of suppliers for each sector
# prices_ = repeat(tau_reshaped[:,:,j],inner = (1,N)).*prices # Prices augmented by trade costs

# # Serching for highest search cost
# matching = divide_sparse(r,prices_) # Ensure to divide when prices != 0 among selected suppliers. 
# matching_coord = argmax(matching,dims = 2) # Find the best supplier
# prices_ = prices_[matching_coord] # Extract best prices (augmented by trade costs)
# price_index = sum(prices_.^(1-eta).*omega).^(1/(1-eta)) # Build price index. 
# i = div.(getindex.(matching_coord,2),N) .+ 1# Find the region of the best supplier and update M_sij
# for s = 1:S
#     M_sij[i[s],j,s] = prices_[s]
# end
# price_indices[j] = price_index  
# push!(coords,matching_coord) # Store the coordinate of the best suppliers in the flat, upstream like, format

# repeat(lbd_reshaped[:,:,j],inner = (1,N))
# t1 = time()-t1
# push!(Times,t1)
# println("Initialise pareto: ",t1)

# coords