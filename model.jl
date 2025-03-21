# using SparseArrays
# using Distributions
# using Random


struct Economy
    R::Int
    S::Int
    eta::Float64
    filter_A_downstream::Vector{Bool}
    sigma::Float64
    phi_bar::Float64
    omega::Matrix{Float64}
    distances::Matrix{Float64}
    tau::Array{Float64,3}
    lbd::Array{Float64,3}
    T::Matrix{Float64}
    N_upstream::Matrix{Int}
    N::Int
    upstream::SparseMatrixCSC{Int64, Int64}
    w::Union{Nothing, Matrix{Float64}}
    pareto_draws::Union{Nothing, Matrix{Float64}}
    prices::Union{Nothing, SparseMatrixCSC{Float64, Int64}}
    
end


function create_sparse_upstream(N_upstream, S, R, N)
    # Julia
    upstream_dense = [i <= N_upstream[s, r] ? 1 : 0 for   r in 1:R , i in 1:N,s in 1:S]
    return sparse(reshape(permutedims(upstream_dense,(3,2,1)), S,:))
end

function build_economy(;R=3, S=1, eta=0.5, omega=nothing, theta=1.0, phi_bar=0.9, w=nothing,
                       distances=nothing, alpha=1, beta=1, filter_N_upstream=nothing,
                       filter_A_downstream=ones(Bool, R), mu_T=0.095, sigma_T=1.395, sigma=1.0)
    
    Random.seed!(1234)
    # We will use the upstream variable in the rest of the simulation for ease of computation. 
    # We assume that in each region there is at most N firm alive. For a region R the actual number of firms that are alive is given by self.N_upstream
    # Then we sort them for each sector on a single line in the upstream array (of size S x 1 x RN)
        
    distances = isnothing(distances) ? begin
        D = rand(R, R)  # generate a random matrix
        #(D + D')/2          # multiply by its transpose to ensure symmetry
    end : distances  # use the provided distances matrix if available

    alpha = isa(alpha, Int) ? fill(alpha, S) : alpha
    beta = isa(beta, Int) ? fill(beta, S) : beta
    tau = isnothing(alpha) ? rand(S, R, R) : distances .^ reshape(-alpha, 1, 1, :)
    lbd = isnothing(beta) ? rand(S, R, R) : distances .^ reshape(-beta, 1, 1, :)

    T = exp.(randn(S, R) .* sigma_T .+ mu_T)
    poisson_dist = Poisson.(T .* phi_bar^(-theta))
    N_upstream = fill(4, S, R)
    N_upstream[:,end] .= 1
    #N_upstream = filter_N_upstream === nothing ? rand.(poisson_dist) : filter_N_upstream .* rand.(poisson_dist)
    N = maximum(N_upstream)

    upstream = create_sparse_upstream(N_upstream, S, R, N)



    w = isnothing(w) ? abs.(rand(S, R)) : w
    w_extended = reshape(repeat(reshape(w, S, R, 1), inner=(1, 1, N)), S, R*N)
    pareto_draws = rand(Pareto(theta), length(eco.upstream.nzval)) .*phi_bar
    omega = isnothing(omega) ? rand(S, 1) : omega
    return Economy(R, S, eta, filter_A_downstream, sigma, phi_bar, omega, distances, tau, lbd, T, N_upstream, N,
                    upstream, w,nothing,nothing)

    # w = isnothing(w) ? abs.(rand(S, R)) : w
    # w_extended = reshape(repeat(reshape(w, S, R, 1), inner=(1, 1, N)), S, R*N)

    # pareto_draws = rand(Pareto(theta), S, R*N) .* phi_bar
    # prices_dense = w_extended ./ pareto_draws
    # prices = sparse(prices_dense)

    # extended_tau = repeat(tau, inner=(1, 1, N))
    
    #prices .= prices .* reshape(extended_tau, S, R*N)

    # extended_upstream = repeat(upstream, 1, R)
    # extended_lbd = repeat(reshape(lbd, S, R, R), inner=(1, 1, N))

    # matching = sprand(S, R*N*R, 0.05)
    # matching .= matching .> (extended_upstream .* reshape(extended_lbd, S, R*N*R))
    # matching_sparse = convert(SparseMatrixCSC{Int64, Int64}, matching)

    # acceptable_prices = matching_sparse .* (1 ./ prices)
    # price_max_indices = mapslices(argmax, Array(acceptable_prices), dims=2)

    # network_sparse = spzeros(Int, S, R*N*R)
    # for (idx, max_idx) in enumerate(price_max_indices)
    #     network_sparse[idx, max_idx] = 1
    # end

    # omega = isnothing(omega) ? rand(S, 1) : omega
    # price_index = [sum((prices[i, :] .* network_sparse[i, :]).^(1 - eta) .* omega[i])^(1 / (1 - eta)) for i in 1:S]

    # return Economy(R, S, eta, filter_A_downstream, sigma, phi_bar, omega, distances, tau, lbd, T, N_upstream, N,
                #    upstream, w_extended, pareto_draws, prices, network_sparse, price_index)
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

# Example to generate an Economy object and display its attributes:
eco = build_economy(R=3, S=2)

N = eco.N
S = eco.S
R = eco.R
w = eco.w
theta = 1
phi_bar = 2
upstream = eco.upstream
tau = eco.tau
N_upstream = eco.N_upstream

w_extended = repeat(w, inner=(1, N))
pareto_draws = rand(Pareto(theta), length(nonzeros(upstream))) .*phi_bar
rows, cols, nzvals = findnz(upstream)
pareto_draws = sparse(rows, cols, pareto_draws, size(upstream)...)
prices = remove_inf_sparse(w_extended./pareto_draws)

# So far it works with constant alpha across sector. Might want to verify if we use different alpha
extended_tau = reshape(permutedims(tau, (3,2,1)),S,R*R)


# Here we construct a matrix tau of size (S,RxRxN). 
# If S = 1, R = 2 and N = 3 
# if tau_11 is the iceberg cost from region 1 to region 1 the extended matrix will be:
# (tau_11,tau_11,tau_11,tau_12,tau_12,tau_12,tau_21,tau_21,tau_21,tau_22,tau_22,tau_22)
extended_tau = repeat(extended_tau,inner = [1,N])

# We multiply it with the matrix of prices. Initially it is a matrix of size (S,RN) and we extend it such as
# If p_11 is the price of the first firm in the first region then the extended price matrix will be:
# p_11,p_12,p_13,p_11,p_12,p_13,p_21,p_22,p_23,p_21,p_22,p_23
prices_repeated = repeat(prices, outer=(1, R))


# The first 3 values will be the price for firms in region 1 to serve region 1. 
# The next 3 values will be the price for firms in region 1 to serve region 2. 
prices_repeated = extended_tau.*prices_repeated


prices_repeated


upstream


