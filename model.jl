##### SMM for Spatial Comovement #####
# Author: Swann Chelly 
# Latest update: Build model and trade flow 21/03/2025
# List of concerns
## How to add the outside sector ? 

# To Do: 
# Add the rho_si moment. 
# Import data for testing. 


# import Pkg; Pkg.add("Distributions")
# import Pkg; Pkg.add("SparseArrays")

using SparseArrays
using Distributions
using Random
struct Economy
    R::Int
    S::Int
    chi_si
    pi_sA
    pi_jA
    rho_si    
end


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


function build_economy(;R=3, S=1, eta=0.5, omega=nothing, theta=1.0, phi_bar=0.9, w=nothing,
                       distances=nothing, alpha=1.0, beta=1.0, filter_N_upstream=nothing,
                       filter_A_downstream=ones(Bool, R), mu_T=10, sigma_T=1.395, sigma=1.0,g = CES)
    
    Random.seed!(1234)
    
    # Initialise frictions and parameters
    distances = isnothing(distances) ? begin
        D = rand(R,R).+1    
        (D+D')/2         # multiply by its transpose to ensure symmetry
    end : distances  # use the provided distances matrix if available
    # For testing
    #distances = reshape(collect(2:(R*R + 1)), R, R).*1.0
    alpha = isa(alpha, Float64) ? fill(alpha, S) : alpha
    beta = isa(beta, Float64) ? fill(beta, S) : beta
    tau = isnothing(alpha) ? rand(S, R, R) : distances .^ reshape(-alpha, 1, 1, :)
    lbd = isnothing(beta) ? rand(S, R, R) : distances .^ reshape(-beta, 1, 1, :)
    omega = isnothing(omega) ? rand(S, 1) : omega


    # Initialise the firms
    ## We will use the upstream variable in the rest of the simulation for ease of computation. 
    ## We assume that in each region there is at most N firm alive. For a region R the actual number of firms that are alive is given by self.N_upstream
    ## Then we sort them for each sector on a single line in the upstream array (of size S x 1 x RN)
    T = exp.(randn(S, R) .* sigma_T .+ mu_T) # T_sj: Region level comparative advantes
    poisson_dist = Poisson.(T .* phi_bar^(-theta))
    # Used for testing
    # N_upstream = fill(4, S, R)
    # N_upstream[:,end] .= 1
    # N_si: Number of firms drawn from a Poisson distribution according to region-level comparative advantages. 
    ## We set manually the regions where there are no firms if filter_N_upstream is given. 
    N_upstream = filter_N_upstream === nothing ? rand.(poisson_dist) : filter_N_upstream .* rand.(poisson_dist)
    N = maximum(N_upstream)
    upstream = create_sparse_upstream(N_upstream, S, R, N)

    # Generate wages, productivity. Construct firm level prices. 
    w = isnothing(w) ? abs.(rand(S, R)) : w
    w_extended = repeat(w, inner=(1, N))
    pareto_draws = rand(Pareto(theta), length(nonzeros(upstream))) .*phi_bar
    rows, cols, _ = findnz(upstream)
    pareto_draws = sparse(rows, cols, pareto_draws, size(upstream)...)
    prices = remove_inf_sparse(w_extended./pareto_draws)

    # Prices augmented by iceberg costs. 

    # So far it works with constant alpha across sector. Might want to verify if we use different alpha
    # Here we construct a matrix tau of size (S,RxRxN). 
    # If S = 1, R = 2 and N = 3 
    # if tau_11 is the iceberg cost from region 1 to region 1 the extended matrix will be:
    # (tau_11,tau_11,tau_11,tau_21,tau_21,tau_21,tau_12,tau_12,tau_12,tau_22,tau_22,tau_22)

    # If tau is not symmetric one should use:
    #extended_tau = reshape(permutedims(permutedims(tau,(2,1,3)), (3,2,1)),S,R*R)
    # Otherwise a fastest version in symmetric is:
    extended_tau = reshape(permutedims(tau, (3,2,1)),S,R*R)
    extended_tau = repeat(extended_tau,inner = (1,N))

    # We multiply it with the matrix of prices. Initially it is a matrix of size (S,RN) and we extend it such as
    # If p_11 is the price of the first firm in the first region then the extended price matrix will be:
    # p_11,p_12,p_13,p_21,p_22,p_23,p_11,p_12,p_13,p_21,p_22,p_23
    prices_repeated = repeat(prices, outer=(1, R))


    # The first 3 values will be the price for firms in region 1 to serve region 1. 
    # The next 3 values will be the price for firms in region 1 to serve region 2. 
    prices_repeated = extended_tau.*prices_repeated

    # Supplier set. 
    # We obtain a matrix of size (S,RRN) with the first RN values are the prices for exporting to region 1 etc...
    # So far it works with constant alpha across sector. Might want to verify if we use different alpha
    # If lbd is not symmetric one should use:
    # extended_lbd = reshape(permutedims(permutedims(lbd,(2,1,3)), (3,2,1)),S,R*R)
    # Otherwise a fastest version in symmetric is:
    extended_lbd = reshape(permutedims(lbd, (3,2,1)),S,R*R)
    extended_lbd = repeat(extended_lbd,inner = (1,N))
    extended_upstream = repeat(upstream,outer = (1,R))
    extended_lbd = extended_upstream.*extended_lbd
    rd = random_like_sparse(extended_upstream)
    # matching: a S,RRN matrix that contains the set of possible suppliers after the search frictions for each downstream region.
    matching = geq_sparse(rd,extended_lbd) 

    # Find minimal supplier. 
    # To handle 0, we first divide the matching matrix with the prices, find the maximum per downstream region. 
    ## We use the coordinate of the maximum to keep only those values in the matrix containing the prices of the potential supplier (matched_prices)
    acceptable_prices = Array(divide_sparse(matching,prices_repeated))
    acceptable_prices_flat = from_S_RNN_to_flat(acceptable_prices,S,R,N)

    matched_prices = Array(matching.*prices_repeated)
    matched_prices_flat = from_S_RNN_to_flat(matched_prices,S,R,N)
    p_sj_coord = argmax(acceptable_prices_flat, dims=2) # Coordinate of the best firm in j

    # Set the best firm coordinate's price to the matched price at that coordinate
    p_sj = zeros(size(matched_prices_flat))
    p_sj[p_sj_coord] = matched_prices_flat[p_sj_coord]
    p_sj = from_flat_to_structured(p_sj,S,R,N)

    # Build prices indices. 
    price_indices = reshape(sum(sum(p_sj,dims = 1),dims = 2),(S,R))
    omega_ = repeat(omega,outer = (1,R))
    price_indices = price_indices.^eta.^omega_
    price_indices = reshape(sum(price_indices,dims = 1),R)

    X_j = g(price_indices,2).*1.0

    # We create M_sij the trade flow from upstream firms in (s,i) to downstream firms in j
    # First we build a R x R x S matrix with (Downstream,Upstream,Sector)
    # The first matrix has R rows. If the coefficient in row r is not 0, it means that it serves region r at the in row r. 

    omega_ = reshape(permutedims(repeat(omega,inner = (1,R*R)),(2,1)),(R,1,R,S))
    M_sij = sum(p_sj,dims = 2)
    M_sij = permutedims(M_sij, (3, 2, 1, 4))
    M_sij = (M_sij./price_indices).^(1-eta).*omega_.*X_j

    # Build moments
    # M_sj 
    # chi_si = M_{si.}/M_{sA}
    M_si = reshape(sum(M_sij,dims = 1),(R,S))
    M_sA = sum(M_si,dims = 1)
    chi_si = M_si./M_sA

    # pi_sA
    pi_sA = M_sA/sum(M_sA)

    # pi_jA: Share of region $i$ in the total purchase of the aerospace industry. 
    M_sj = reshape(sum(sum(p_sj,dims = 2),dims = 1),(R,S))
    M_j  = sum(M_sj,dims = 2)
    pi_jA = M_j/sum(M_j)

    rho_si = permutedims(p_sj, (3, 2, 1, 4))
    rho_si = sum(rho_si,dims = 1)

    rho_si = abs.(rho_si./rho_si)
    rho_si = ifelse.(isnan.(rho_si), 0.0, rho_si)
    rho_si = reshape(sum(rho_si,dims = 2),(S,R))
    rho_si = rho_si./N_upstream

    return Economy(R, S, chi_si,pi_sA,pi_jA,rho_si)

end





# Example to generate an Economy object and display its attributes:
eco = build_economy(R=3, S=2)

R=3
S=2
eta=0.5
omega=nothing
theta=1.0
phi_bar=0.9
w=nothing
distances=nothing
alpha=1.0
beta=1.0
filter_N_upstream=nothing
filter_A_downstream=ones(Bool, R)
mu_T=1
sigma_T=1.395
sigma=1.0
g = CES


Random.seed!(1234)
    
# Initialise frictions and parameters
# distances = isnothing(distances) ? begin
#     D = rand(R,R) .+1   
#     (D+D')/2         # multiply by its transpose to ensure symmetry
# end : distances  # use the provided distances matrix if available
# For testing
distances = reshape(collect(2:(R*R + 1)), R, R).*1.0
alpha = isa(alpha, Float64) ? fill(alpha, S) : alpha
beta = isa(beta, Float64) ? fill(beta, S) : beta
tau = isnothing(alpha) ? rand(S, R, R) : distances .^ reshape(-alpha, 1, 1, :)
lbd = isnothing(beta) ? rand(S, R, R) : distances .^ reshape(-beta, 1, 1, :)
omega = isnothing(omega) ? rand(S, 1) : omega


# Initialise the firms
## We will use the upstream variable in the rest of the simulation for ease of computation. 
## We assume that in each region there is at most N firm alive. For a region R the actual number of firms that are alive is given by self.N_upstream
## Then we sort them for each sector on a single line in the upstream array (of size S x 1 x RN)
T = exp.(randn(S, R) .* sigma_T .+ mu_T) # T_sj: Region level comparative advantes
poisson_dist = Poisson.(T .* phi_bar^(-theta))
# Used for testing
N_upstream = fill(4, S, R)
N_upstream[:,end] .= 1
# N_si: Number of firms drawn from a Poisson distribution according to region-level comparative advantages. 
## We set manually the regions where there are no firms if filter_N_upstream is given. 
# N_upstream = filter_N_upstream === nothing ? rand.(poisson_dist) : filter_N_upstream .* rand.(poisson_dist)
N = maximum(N_upstream)
upstream = create_sparse_upstream(N_upstream, S, R, N)

# Generate wages, productivity. Construct firm level prices. 
w = isnothing(w) ? abs.(rand(S, R)) : w
w_extended = repeat(w, inner=(1, N))
pareto_draws = rand(Pareto(theta), length(nonzeros(upstream))) .*phi_bar
rows, cols, _ = findnz(upstream)
pareto_draws = sparse(rows, cols, pareto_draws, size(upstream)...)
prices = remove_inf_sparse(w_extended./pareto_draws)

# Prices augmented by iceberg costs. 

# So far it works with constant alpha across sector. Might want to verify if we use different alpha
# Here we construct a matrix tau of size (S,RxRxN). 
# If S = 1, R = 2 and N = 3 
# if tau_11 is the iceberg cost from region 1 to region 1 the extended matrix will be:
# (tau_11,tau_11,tau_11,tau_21,tau_21,tau_21,tau_12,tau_12,tau_12,tau_22,tau_22,tau_22)

# If tau is not symmetric one should use:
#extended_tau = reshape(permutedims(permutedims(tau,(2,1,3)), (3,2,1)),S,R*R)
# Otherwise a fastest version in symmetric is:
extended_tau = reshape(permutedims(tau, (3,2,1)),S,R*R)
extended_tau = repeat(extended_tau,inner = (1,N))

# We multiply it with the matrix of prices. Initially it is a matrix of size (S,RN) and we extend it such as
# If p_11 is the price of the first firm in the first region then the extended price matrix will be:
# p_11,p_12,p_13,p_21,p_22,p_23,p_11,p_12,p_13,p_21,p_22,p_23
prices_repeated = repeat(prices, outer=(1, R))


# The first 3 values will be the price for firms in region 1 to serve region 1. 
# The next 3 values will be the price for firms in region 1 to serve region 2. 
prices_repeated = extended_tau.*prices_repeated

# Supplier set. 
# We obtain a matrix of size (S,RRN) with the first RN values are the prices for exporting to region 1 etc...
# So far it works with constant alpha across sector. Might want to verify if we use different alpha
# If lbd is not symmetric one should use:
# extended_lbd = reshape(permutedims(permutedims(lbd,(2,1,3)), (3,2,1)),S,R*R)
# Otherwise a fastest version in symmetric is:
extended_lbd = reshape(permutedims(lbd, (3,2,1)),S,R*R)
extended_lbd = repeat(extended_lbd,inner = (1,N))
extended_upstream = repeat(upstream,outer = (1,R))
extended_lbd = extended_upstream.*extended_lbd
rd = random_like_sparse(extended_upstream)
# matching: a S,RRN matrix that contains the set of possible suppliers after the search frictions for each downstream region.
matching = geq_sparse(rd,extended_lbd) 

# Find minimal supplier. 
# To handle 0, we first divide the matching matrix with the prices, find the maximum per downstream region. 
## We use the coordinate of the maximum to keep only those values in the matrix containing the prices of the potential supplier (matched_prices)
acceptable_prices = Array(divide_sparse(matching,prices_repeated))
acceptable_prices_flat = from_S_RNN_to_flat(acceptable_prices,S,R,N)

matched_prices = Array(matching.*prices_repeated)
matched_prices_flat = from_S_RNN_to_flat(matched_prices,S,R,N)
p_sj_coord = argmax(acceptable_prices_flat, dims=2) # Coordinate of the best firm in j

# Set the best firm coordinate's price to the matched price at that coordinate
p_sj = zeros(size(matched_prices_flat))
p_sj[p_sj_coord] = matched_prices_flat[p_sj_coord]
p_sj = from_flat_to_structured(p_sj,S,R,N)

# Build prices indices. 
price_indices = reshape(sum(sum(p_sj,dims = 1),dims = 2),(S,R))
omega_ = repeat(omega,outer = (1,R))
price_indices = price_indices.^eta.^omega_
price_indices = reshape(sum(price_indices,dims = 1),R)

X_j = g(price_indices,2).*1.0

# We create M_sij the trade flow from upstream firms in (s,i) to downstream firms in j
# First we build a R x R x S matrix with (Downstream,Upstream,Sector)
# The first matrix has R rows. If the coefficient in row r is not 0, it means that it serves region r at the in row r. 

omega_ = reshape(permutedims(repeat(omega,inner = (1,R*R)),(2,1)),(R,1,R,S))
M_sij = sum(p_sj,dims = 2)
M_sij = permutedims(M_sij, (3, 2, 1, 4))
M_sij = (M_sij./price_indices).^(1-eta).*omega_.*X_j

# Build moments
# M_sj 
# chi_si = M_{si.}/M_{sA}
M_si = reshape(sum(M_sij,dims = 1),(R,S))
M_sA = sum(M_si,dims = 1)
chi_si = M_si./M_sA

# pi_sA
pi_sA = M_sA/sum(M_sA)

# pi_jA: Share of region $i$ in the total purchase of the aerospace industry. 
M_sj = reshape(sum(sum(p_sj,dims = 2),dims = 1),(R,S))
M_j  = sum(M_sj,dims = 2)
pi_jA = M_j/sum(M_j)

# rho_si

rho_si = permutedims(p_sj, (3, 2, 1, 4))
rho_si = sum(rho_si,dims = 1)

rho_si = abs.(rho_si./rho_si)
rho_si = ifelse.(isnan.(rho_si), 0.0, rho_si)
rho_si = reshape(sum(rho_si,dims = 2),(S,R))
rho_si = rho_si./N_upstream