##### SMM for Spatial Comovement #####
# Author: Swann Chelly 
# Latest update: Build model and trade flow 24/03/2025
# List of concerns
## How to add the outside sector ? 
## Choisir un nombre d'entreprises et les tirer selon Gaubert 2021

# To Do: 
# Add the rho_si moment. 
# Import data for testing. 


# import Pkg; Pkg.add("Distributions")
# import Pkg; Pkg.add("SparseArrays")

using SparseArrays
using Distributions
using Random
using NPZ

using Pkg



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





# Example to generate an Economy object and display its attributes:
# eco = build_economy(R=3, S=2)
distances = NPZ.npzread("./distances.npy")  # for `.npz`
filter_A_downstream = NPZ.npzread("./filter_A_downstream.npy")  # for `.npz`
filter_N_upstream = NPZ.npzread("./filter_N_upstream.npy")  # for `.npz`

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

using NPZ




Random.seed!(1234)
    
# Initialise frictions and parameters
distances = isnothing(distances) ? begin
    D = rand(R,R) .+1   
    (D+D')/2         # multiply by its transpose to ensure symmetry
end : distances  # use the provided distances matrix if available
# For testing
# distances = reshape(collect(2:(R*R + 1)), R, R).*1.0
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
N = Integer(maximum(N_upstream))
upstream = create_sparse_upstream(N_upstream, S, R, N)

# Generate wages, productivity. Construct firm level prices. 
w = isnothing(w) ? abs.(rand(S, R)) : w # w_sr = wage of sector s in region r
w_extended = repeat(w, inner=(1, N)) # Extension of this wage fro fitting upstream shape. 

# Draw pareto for firms and shape it as upstream (sparse (S,RN) matrix)
pareto_draws = rand(Pareto(theta), length(nonzeros(upstream))) .*phi_bar 
rows, cols, _ = findnz(upstream) 
pareto_draws = sparse(rows, cols, pareto_draws, size(upstream)...)
prices = remove_inf_sparse(w_extended./pareto_draws) # Competitive equilibrium, prices are wages / productivity. 


# So far tau is a (R,R,S) matrix with (i,j,s) the trade cost for shipping products of sector s from i to j. 
# We change it into a (S,R,R) matrix with (s,i,j) the trade cost from shipping products of sectors s from i to j. 
lbd_reshaped = permutedims(lbd,(3,1,2))
tau_reshaped = permutedims(tau,(3,1,2))

price_indices = copy(filter_A_downstream).*1.0 
M_sij = zeros((R,R,S)) # We create a blank matrix (Upstream, Downstream, Sector). For a tuple (i,j,s), it will be best serving price of region j for sector s if i is selected. Otherwise 0 
coords = Any[] # We keep the coordinate of the best price in order to build rho_si
for j = 1:length(filter_A_downstream) # Iterate on downstream regions. 
    if filter_A_downstream[j] == 1
        lbd_ = repeat(lbd_reshaped[:,:,j],inner = (1,N)).*upstream # Frictions to serve region j
        r = geq_sparse(random_like_sparse(prices),lbd_) # Sparse random matching >= Search frictions | Selected set of suppliers for each sector
        prices_ = repeat(tau_reshaped[:,:,j],inner = (1,N)).*prices # Prices augmented by trade costs

        # Serching for highest search cost
        matching = divide_sparse(r,prices_) # Ensure to divide when prices != 0 among selected suppliers. 
        matching_coord = argmax(matching,dims = 2) # Find the best supplier
        prices_ = prices_[matching_coord] # Extract best prices (augmented by trade costs)
        price_index = sum(prices_.^(1-eta).*omega).^(1/(1-eta)) # Build price index. 
        i = div.(getindex.(matching_coord,2),N) .+ 1# Find the region of the best supplier and update M_sij
        for s = 1:S
            M_sij[i[s],j,s] = prices_[s]
        end
        price_indices[j] = price_index  
        push!(coords,matching_coord) # Store the coordinate of the best suppliers in the flat, upstream like, format
    end
end

# Build demand. 
X_j = g(price_indices,2).*1.0

# Build trade flow M_sij. In (i,j,s), so far contains the price of the firm in i that serves j (if selected) and 0 otherwise. 
# We create trade flows using w_sj(p_sj/P_j)**(1-eta)*X_j
M_sij = (M_sij./reshape(price_indices,(1,R))).^(1-eta).*reshape(X_j,(1,R)).*reshape(omega,(1,1,S))
M_sij = ifelse.(isnan.(M_sij), 0.0, M_sij)

# Build moments
# M_sj 
# chi_si = M_{si.}/M_{sA}
M_si = reshape(sum(M_sij,dims = 2),(R,S))
M_sA = sum(M_si,dims = 1)
chi_si = M_si./M_sA

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
rho_si = permutedims(rho_si,(3,2,1))
rho_si = permutedims(reshape(sum(rho_si,dims = 2),(R,S)),(2,1))
rho_si = rho_si./N_upstream
rho_si = ifelse.(isnan.(rho_si), 0.0, rho_si)
rho_si = replace(rho_si, Inf => 0.0)


