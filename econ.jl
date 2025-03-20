using SparseArrays
using Distributions

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
    w::Array{Float64,3}
    pareto_draws::Array{Float64,3}
    prices::SparseMatrixCSC{Float64, Int64}
    network::SparseMatrixCSC{Int64, Int64}
    price_index::Vector{Float64}
end

function create_sparse_upstream(N_upstream, S, R, N)
    upstream_dense = [i < N_upstream[s, r] ? 1 : 0 for s in 1:S, r in 1:R, i in 1:N]
    return sparse(reshape(upstream_dense, S, R*N))
end

function build_economy(;R=3, S=1, eta=0.5, omega=nothing, theta=1.0, phi_bar=0.9, w=nothing,
                       distances=nothing, alpha=nothing, beta=nothing, filter_N_upstream=nothing,
                       filter_A_downstream=ones(Bool, R), mu_T=0.095, sigma_T=1.395, sigma=1.0)
    distances = isnothing(distances) ? abs.(rand(R, R)) : distances
    distances = distances + distances'  # make symmetric if random

    alpha = isa(alpha, Int) ? fill(alpha, S) : alpha
    beta = isa(beta, Int) ? fill(beta, S) : beta

    tau = isnothing(alpha) ? rand(S, R, R) : distances .^ (-reshape(alpha, S, 1, 1))
    lbd = isnothing(beta) ? rand(S, R, R) : distances .^ (-reshape(beta, S, 1, 1))

    T = exp.(randn(S, R) .* sigma_T .+ mu_T)
    poisson_dist = Poisson.(T .* phi_bar^(-theta))
    N_upstream = filter_N_upstream === nothing ? rand.(poisson_dist) : filter_N_upstream .* rand.(poisson_dist)
    N = maximum(N_upstream)

    upstream_sparse = create_sparse_upstream(N_upstream, S, R, N)

    w = isnothing(w) ? abs.(rand(S, R)) : w
    w_extended = reshape(repeat(reshape(w, S, R, 1), inner=(1, 1, N)), S, R*N)

    pareto_draws = rand(Pareto(theta), S, R*N) .* phi_bar
    prices_dense = w_extended ./ pareto_draws
    prices = sparse(prices_dense)

    extended_tau = repeat(tau, inner=(1, 1, N))
    prices .= prices .* reshape(extended_tau, S, R*N)

    extended_upstream = repeat(upstream_sparse, 1, R)
    extended_lbd = repeat(reshape(lbd, S, R, R), inner=(1, 1, N))

    matching = sprand(S, R*N*R, 0.05)
    matching .= matching .> (extended_upstream .* reshape(extended_lbd, S, R*N*R))
    matching_sparse = convert(SparseMatrixCSC{Int64, Int64}, matching)

    acceptable_prices = matching_sparse .* (1 ./ prices)
    price_max_indices = mapslices(argmax, Array(acceptable_prices), dims=2)

    network_sparse = spzeros(Int, S, R*N*R)
    for (idx, max_idx) in enumerate(price_max_indices)
        network_sparse[idx, max_idx] = 1
    end

    omega = isnothing(omega) ? rand(S, 1) : omega
    price_index = [sum((prices[i, :] .* network_sparse[i, :]).^(1 - eta) .* omega[i])^(1 / (1 - eta)) for i in 1:S]

    return Economy(R, S, eta, filter_A_downstream, sigma, phi_bar, omega, distances, tau, lbd, T, N_upstream, N,
                   upstream_sparse, w_extended, pareto_draws, prices, network_sparse, price_index)
end

# Example to generate an Economy object and display its attributes:
eco = build_economy(R=4, S=2)

# List attributes
eco |> fieldnames

# Access specific attributes, for example:
eco.R
eco.N_upstream
eco.prices


