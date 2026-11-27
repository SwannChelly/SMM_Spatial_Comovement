##### profiling.jl — T and the aggregates profiled out of the search #####
#
# The search runs over (α, κ) only. At each (α, κ), `profile_ge` returns the area-level
# comparative advantage T and the aggregates (χ^L, χ^s, A) that reproduce, exactly and
# jointly, the sourcing shares γ_{s,a} (block 5) and blocks 1–3 (labour share, sector
# shares, downstream sales shares), with the expected Eaton–Kortum prices in closed form.
# These blocks are therefore just-identified; [β_d ; β_κ] and Ḡ_s(0) are left to (α, κ)
# and N_s. Only the closed-form PRICES are used here — the extensive-margin moments
# always come from the simulation.
#
# The foreign competitor: T's last column (FOREIGN_COL) is Φ^F_s, added to every buyer's
# Φ_rs, and the γ block's foreign row is the import share. Both γ and its target are
# shares of TOTAL purchases (Σ over the T columns = 1), so the Sinkhorn update moves the
# areas' T and Φ^F_s with the same rule: the profile solves (T, Φ^F) jointly.
#
# Buyers' access to foreign suppliers: buyer r sees the foreign competitor at scale
# ψ_r Φ^F_s, so Φ_rs = Φ^dom_rs + ψ_r Φ^F_s and f_rs = ψ_r Φ^F_s / Φ_rs. ψ_r is profiled
# with (T, Φ^F) on each buyer's import share: the RAW customs shares IS_r^customs, their
# log-odds shifted by one common b at every pass so that the buyers' imports add up to the
# sectors' imports (`buyer_import_targets`, `profile_ge`).
# ψ and Φ^F enter only as a product: ψ = 1 for the buyer with the largest sales share.
#
# Notation. Variables keep the paper's symbols in roman letters, with their indices as
# suffixes: s an upstream sector, r (loop index `dr`) a downstream buyer in
# `DOWNSTREAM_REGIONS` order, l an origin cell (loop index `g` over the active cells),
# a an attraction area. The main objects are
#   T_ls (S × R)            cell-level Fréchet scale, T_{a(l)s} · n_ls^κ
#   T_area (S × T_COL_DIM)  area-level scale, plus the foreign scale Φ^F_s in FOREIGN_COL
#   psi (R_d)               ψ_r, the buyer's access to foreign suppliers (ψ_ref = 1)
#   Phi (S × R_d)           Φ_sr = Σ_l T_ls (w_l τ_lr)^{−θ} + ψ_r Φ^F_s
#   P_sr (S × R_d)          expected sector price index Γ_s Φ_sr^{−1/θ}
#   P_r, c_r (R_d)          buyer's intermediate price index and unit cost
#   ell_r (R_d)             buyer's labour cost share ℓ_r
#   s_rs (S × R_d)          sector split of a buyer's intermediate bill
#   omega_sr (S × R_d)      a buyer's spending on sector s, ω_sr
#   gamma (T_COL_DIM × S)   sourcing shares of each sector's total purchases, γ_{a,s}

using LinearAlgebra
using Printf

"""
    compute_Phi(T_ls, tau; PhiF = nothing) -> Φ (S × R_downstream),
    Φ_{s,r} = Σ_l T_ls (w_l τ_lr)^{−θ} + ψ_r Φ^F_s

`T_ls` (S × R) is the cell-level Fréchet scale and `tau` (R × R_downstream) the iceberg cost
from origin cell to buyer. `PhiF` is a vector (S), a foreign scale common to every buyer,
or a matrix (S × R_downstream), the foreign scale as each buyer sees it: ψ_r Φ^F_s at the
estimate (`foreign_scale_by_buyer`), rescaled in the counterfactuals of `incidence_stats`,
which hold each buyer's import share at its estimate.
"""
function compute_Phi(T_ls::AbstractMatrix{FT}, tau::AbstractMatrix{FT}; PhiF = nothing) where {FT}
    Phi = zeros(FT, S, R_downstream)
    # Each active cell adds its cost-adjusted productivity to every buyer's sector index.
    for g in 1:n_good
        s = GOOD_S[g]; l = GOOD_R[g]
        T_cell = T_ls[s, l]; w_cell = W_RS_FLAT[g]
        for dr in 1:R_downstream
            Phi[s, dr] += T_cell * (w_cell * tau[l, dr])^(-theta)
        end
    end
    # The foreign competitor enters every buyer's index of an importing sector.
    if PhiF isa AbstractMatrix
        for s in 1:S, dr in 1:R_downstream
            PhiF[s, dr] > 0 && (Phi[s, dr] += PhiF[s, dr])
        end
    elseif PhiF !== nothing
        for s in 1:S
            PhiF[s] > 0 || continue
            for dr in 1:R_downstream
                Phi[s, dr] += PhiF[s]
            end
        end
    end
    return Phi
end

"""
    compute_prices_analytical(χ^L, χ^s, A_r, T_ls, tau; PhiF = nothing)
        -> (P_sr, P_r, c_r, c_tilde_r, p_r, P_agg, Y_r, mu, Phi)

Closed-form EK prices and downstream sales:
- `P_sr = Γ_s Φ_sr^{−1/θ}` (S × R_d), with the foreign term in Φ;
- `P_r = (Σ_s χ_s P_sr^{1−ν})^{1/(1−ν)}`, the buyer's intermediate price index (R_d);
- `c_r = (χ^L w_r^{1−λ} + (1−χ^L) P_r^{1−λ})^{1/(1−λ)}`, the unit cost, and
  `c_tilde_r = c_r / A_r` the productivity-adjusted cost (R_d);
- `p_r = c_tilde_r / μ` with `mu = ε/(ε−1)`, the downstream price index `P_agg`, and the
  sales `Y_r = δ_r p_r^ε P_agg^{−ε}` (R_d).
"""
function compute_prices_analytical(chi_L::FT, chi_s, A_r, T_ls::AbstractMatrix,
                                   tau::AbstractMatrix; PhiF = nothing) where {FT}
    # 1) Sector price indices faced by each buyer.
    Phi = compute_Phi(T_ls, tau; PhiF = PhiF)
    P_sr = similar(Phi)
    for s in 1:S, dr in 1:R_downstream
        P_sr[s, dr] = GAMMA_FACTOR[s] * Phi[s, dr]^(-1/theta)
    end
    # 2) Each buyer aggregates its sectors into one intermediate bundle, then combines it
    #    with labour into a unit cost.
    chi_s_column = reshape(collect(chi_s), S, 1)
    P_r = vec(sum(chi_s_column .* P_sr .^ (1 - nu), dims=1)) .^ (1 / (1 - nu))
    c_r = similar(P_r)
    for dr in 1:R_downstream
        region = DOWNSTREAM_REGIONS[dr]
        c_r[dr] = (chi_L * regional_wages[region]^(1-lambda) +
                   (1-chi_L) * P_r[dr]^(1-lambda))^(1/(1-lambda))
    end
    # 3) Downstream firms price at a constant markup over their productivity-adjusted
    #    cost, and final demand allocates sales across them.
    c_tilde_r        = c_r ./ A_r
    mu               = FT(epsilon / (epsilon - 1))
    p_r              = c_tilde_r ./ mu
    delta_downstream = delta_r[DOWNSTREAM_REGIONS]
    P_agg            = sum(p_r .^ epsilon .* delta_downstream)^(1/epsilon)
    Y_r              = p_r .^ epsilon .* P_agg^(-epsilon) .* delta_downstream
    return (P_sr=P_sr, P_r=P_r, c_r=c_r, c_tilde_r=c_tilde_r,
            p_r=p_r, P_agg=P_agg, Y_r=Y_r, mu=mu, Phi=Phi)
end

"""
    gamma_T_analytical(χ^L, χ^s, A, T_area, kappa, tau; psi = ones(R_d)) -> γ (T_COL_DIM × S)

Closed-form shares of each sector's TOTAL purchases at the area level, the import share
`Σ_r ω_sr ψ_r Φ^F_s/Φ_sr / Σ_r ω_sr` on the foreign row (Σ over the columns = 1). The spending
weights come from the full price solve: `ω_sr = χ_s (P_sr/P_r)^{1−ν} (P_r/c_r)^{1−λ} (1−χ^L) μ Y_r`.
"""
function gamma_T_analytical(chi_L::Real, chi_s, A, T_area::AbstractMatrix, kappa, tau::AbstractMatrix;
                            psi::AbstractVector = ones(R_downstream))
    T_ls = gather_T_to_ze(T_area, kappa)
    PhiF = foreign_scale_by_buyer(foreign_phi(T_area), psi)
    (; P_sr, P_r, c_r, Y_r, mu, Phi) = compute_prices_analytical(chi_L, chi_s, A, T_ls, tau; PhiF = PhiF)
    # A buyer's spending on sector s: its intermediate bill split across sectors by CES.
    omega_sr = similar(Phi)
    for s in 1:S, dr in 1:R_downstream
        omega_sr[s, dr] = chi_s[s] * (P_sr[s, dr] / P_r[dr])^(1 - nu) *
                   (P_r[dr] / c_r[dr])^(1 - lambda) * (1 - chi_L) * mu * Y_r[dr]
    end
    return gamma_T_from_weights(T_ls, PhiF, tau, Phi, omega_sr)
end

"""
    gamma_T_from_weights(T_ls, PhiF_sr, tau, Phi, omega_sr) -> γ (T_COL_DIM × S)

The flows `X_ls = Σ_r T_ls (w τ)^{−θ}/Φ_sr ω_sr` (R × S) and `X_F = Σ_r ψ_r Φ^F_s/Φ_sr ω_sr` (S),
with the sector expenditure `omega_sr` (S × R_d) and the foreign scale by buyer
`PhiF_sr = ψ_r Φ^F_s` (S × R_d, `foreign_scale_by_buyer`) supplied, as shares of total
purchases. Each origin's flow is the buyer's spending times its EK probability of winning.
"""
function gamma_T_from_weights(T_ls::AbstractMatrix, PhiF::AbstractMatrix, tau::AbstractMatrix,
                              Phi::AbstractMatrix, omega_sr::AbstractMatrix)
    FT = promote_type(eltype(T_ls), eltype(omega_sr))
    # 1) Domestic flows: each cell sells to a buyer in proportion to its share of Φ_sr.
    X_ls = zeros(FT, R, S)
    @inbounds for g in 1:n_good
        s = GOOD_S[g]; l = GOOD_R[g]
        T_cell = T_ls[s, l]; w_cell = W_RS_FLAT[g]
        for dr in 1:R_downstream
            Phi[s, dr] < 1e-300 && continue
            X_ls[l, s] += T_cell * (w_cell * tau[l, dr])^(-theta) / Phi[s, dr] * omega_sr[s, dr]
        end
    end
    # 2) Imports: the foreign competitor's share of Φ_sr, buyer by buyer.
    X_F = zeros(FT, S)
    @inbounds for s in 1:S
        for dr in 1:R_downstream
            (PhiF[s, dr] > 0 && Phi[s, dr] >= 1e-300) || continue
            X_F[s] += PhiF[s, dr] / Phi[s, dr] * omega_sr[s, dr]
        end
    end
    return gamma_T_from_flows(X_ls, X_F)
end

"""
    with_T(theta_vec, T_area) -> copy of the level vector `theta_vec` with its T block set
    to the active entries of `T_area` (S × T_COL_DIM), s-major as `T_MASK`
"""
function with_T(theta_vec::AbstractVector, T_area::AbstractMatrix)
    theta_vec_new = copy(theta_vec)
    theta_vec_new[t_param_range()] = vec(permutedims(T_area))[T_MASK]
    return theta_vec_new
end

"""
    with_psi(theta_vec, psi) -> copy of the level vector `theta_vec` with its ψ block set to `psi` (R_d)
"""
function with_psi(theta_vec::AbstractVector, psi::AbstractVector)
    theta_vec_new = copy(theta_vec)
    theta_vec_new[psi_range()] = psi
    return theta_vec_new
end

# ══════════════════════════════════════════════════════════════════════════════
# THE PROFILE: T and the aggregates (χ^L, χ^s, A) at a given (α, κ)
# ══════════════════════════════════════════════════════════════════════════════
#
# Given (α, κ, T) the aggregates are just-identified by blocks 1–3: χ^L by the aggregate
# labour share (1 equation), χ^s by the sector shares of intermediate spending (S − 1
# free equations, Σχ^s = 1), A by the downstream sales shares (R_d − 1, A_1 = 1). All
# of it runs through the EXPECTED price indices P_sr = Γ_s Φ_sr^{−1/θ}, which do not
# depend on the aggregates, so the profile is deterministic:
#
#   ℓ_r  = χ^L (w_r/c_r)^{1−λ}                         regional labour share (no A)
#   s_rs = χ_s P_sr^{1−ν} / Σ_k χ_k P_kr^{1−ν}         sector split of intermediates
#   L    = Σ_r π_r ℓ_r                                 block 1 (cost ∝ sales ∝ π_r)
#   Õ_s  = Σ_r π_r (1−ℓ_r) s_rs / Σ_r π_r (1−ℓ_r)      block 2
#   A_r ∝ c_r (π_r/δ_r)^{−1/ε}                         block 3, closed form, A_1 = 1
#
# π_r is the TARGET sales share throughout: at the profile A reproduces it exactly, so
# the Sinkhorn expenditure weights ω_sr = π_r (1−ℓ_r) s_rs carry no A and A is computed
# once, at the end. Pinning sales also removes the ε channel a T map at fixed A would
# carry (raise T ⇒ cheaper inputs ⇒ with |ε| large sales jump ⇒ demand returns).
#
# (χ^L, χ^s) are solved JOINTLY by Newton on the unconstrained "aggregate coordinates"
# x = (logit χ^L, log χ_s[2:S] − log χ_s[1]): the weights π_r (1−ℓ_r) move with χ^s
# through P_r, so the two blocks are coupled. The system is S × S and costs O(S R_d) per
# evaluation, negligible beside one Φ pass, so it is solved to convergence at every
# Sinkhorn pass and the T map stays a map of T alone.

"""
    expected_sector_prices(T_ls, tau; PhiF = nothing) -> (P_sr (S × R_d), Φ (S × R_d))

The expected EK sector price indices `P_sr = Γ_s Φ_sr^{−1/θ}`, which depend on T and τ only.
"""
function expected_sector_prices(T_ls::AbstractMatrix, tau::AbstractMatrix; PhiF = nothing)
    Phi  = compute_Phi(T_ls, tau; PhiF = PhiF)
    P_sr = similar(Phi)
    @inbounds for s in 1:S, dr in 1:R_downstream
        P_sr[s, dr] = GAMMA_FACTOR[s] * Phi[s, dr]^(-1/theta)
    end
    return P_sr, Phi
end

"""
    profile_targets(; labor_share, sector_shares, sales_shares)
        -> (labor_share, sector_shares (S), sales_shares (R_d))

The targets of the profile besides γ: the aggregate labour share, the sector shares of
intermediate spending renormalised to the simplex (the model's shares sum to one; only
their profile is a target) and the downstream sales shares π_r (`DOWNSTREAM_REGIONS`
order, renormalised). Defaults: the data. (`sensitivity.jl` defines another, unrelated
method `profile_targets(c)`.)
"""
profile_targets(; labor_share = agg_labor_share, sector_shares = agg_industry_share,
                  sales_shares = emp_pi_r) =
    (labor_share = Float64(labor_share),
     sector_shares = collect(Float64, vec(sector_shares)) ./ sum(sector_shares),
     sales_shares = collect(Float64, vec(sales_shares)) ./ sum(sales_shares))

"""
    aggregate_shares(aggregate_coordinates, P_sr) -> (χ^L, χ^s, ℓ_r, s_rs, P_r, c_r)

Maps the Newton unknowns `aggregate_coordinates = (logit χ^L, log χ_s[2:S] − log χ_s[1])`
(S) back to the aggregate shares, and computes at the prices `P_sr` (S × R_d) each buyer's
sector split `s_rs` (S × R_d), intermediate price index `P_r`, unit cost `c_r` and labour
cost share `ℓ_r = χ^L (w_r/c_r)^{1−λ}` (all R_d).
"""
function aggregate_shares(aggregate_coordinates::AbstractVector, P_sr::AbstractMatrix)
    # 1) Back to levels: χ^L in (0, 1), χ^s on the simplex.
    chi_L = 1 / (1 + exp(-aggregate_coordinates[1]))
    log_chi_s_relative = vcat(0.0, aggregate_coordinates[2:end])
    chi_s = exp.(log_chi_s_relative .- maximum(log_chi_s_relative)); chi_s ./= sum(chi_s)
    # 2) CES split of the intermediate bill across sectors, and the bundle's price index.
    chi_P_power = chi_s .* P_sr .^ (1 - nu)          # χ_s P_sr^{1−ν}
    P_r_power   = vec(sum(chi_P_power; dims = 1))   # P_r^{1−ν}
    s_rs = chi_P_power ./ reshape(P_r_power, 1, :)
    P_r  = P_r_power .^ (1 / (1 - nu))
    # 3) Unit cost and labour share of each buyer (A-free: A is Hicks-neutral).
    c_r  = similar(P_r); ell_r = similar(P_r)
    @inbounds for dr in eachindex(P_r)
        w_r = regional_wages[DOWNSTREAM_REGIONS[dr]]
        c_r[dr] = (chi_L * w_r^(1 - lambda) + (1 - chi_L) * P_r[dr]^(1 - lambda))^(1 / (1 - lambda))
        ell_r[dr] = chi_L * (w_r / c_r[dr])^(1 - lambda)
    end
    return chi_L, chi_s, ell_r, s_rs, P_r, c_r
end

"""
    aggregate_residual(aggregate_coordinates, P_sr, targets) -> Vector (S): model − target for blocks 1–2

The S equations that pin (χ^L, χ^s) at given prices P_sr; the profile is the root
F(x) = 0, x the aggregate coordinates (`aggregate_shares`):
- `F[1] = Σ_r π_r ℓ_r − L̂`: the aggregate labour share. Costs are proportional to sales,
  so the regional shares ℓ_r are averaged with the target sales shares π_r;
- `F[1+k] = Õ_{k+1} − Ô_{k+1}`, k = 1..S−1: the share of sector s in total intermediate
  spending, `Õ_s = Σ_r π_r (1−ℓ_r) s_rs / Σ_r π_r (1−ℓ_r)`, i.e. each buyer's sector split
  s_rs weighted by its intermediate bill π_r (1−ℓ_r). Sector 1 is dropped: the shares sum
  to one on both sides, so its equation is implied by the others.
The labour share depends on χ^s through P_r (hence c_r), and the intermediate weights
on χ^L, which is why the two blocks are solved jointly.
"""
function aggregate_residual(aggregate_coordinates::AbstractVector, P_sr::AbstractMatrix, targets)
    _, _, ell_r, s_rs, _, _ = aggregate_shares(aggregate_coordinates, P_sr)
    intermediate_bill   = targets.sales_shares .* (1 .- ell_r)                  # π_r (1−ℓ_r)
    model_sector_shares = (s_rs * intermediate_bill) ./ sum(intermediate_bill)  # Õ_s
    return vcat(sum(targets.sales_shares .* ell_r) - targets.labor_share,
                model_sector_shares[2:end] .- targets.sector_shares[2:end])
end

"""
    aggregate_init(P_sr, targets) -> aggregate coordinates (S), the Newton starting point

The one-region closed form at the π-weighted mean log prices: `χ_s ∝ Õ_s P̄_s^{ν−1}`, then
`logit χ^L = logit L + (1−λ) log(P̄/w̄)`, with P̄ the CES bundle of the P̄_s and w̄ the
π-weighted geometric mean wage.
"""
function aggregate_init(P_sr::AbstractMatrix, targets)
    mean_log_P_s = log.(P_sr) * targets.sales_shares
    chi_s_start  = max.(targets.sector_shares, 1e-12) .* exp.((nu - 1) .* mean_log_P_s); chi_s_start ./= sum(chi_s_start)
    P_bar = sum(chi_s_start .* exp.((1 - nu) .* mean_log_P_s))^(1 / (1 - nu))
    w_bar = exp(sum(targets.sales_shares .* log.(regional_wages[DOWNSTREAM_REGIONS])))
    return vcat(log(targets.labor_share / (1 - targets.labor_share)) + (1 - lambda) * log(P_bar / w_bar),
                log.(chi_s_start[2:end]) .- log(chi_s_start[1]))
end

"""
    solve_aggregates(P_sr; aggregate_coordinates_init, targets, tol=1e-12, max_iter=50)
        -> (aggregate_coordinates, converged, max_residual, iterations)

Solves F(x) = 0 for the aggregate coordinates x = (logit χ^L, log χ_s[2:S] − log χ_s[1])
at fixed prices P_sr, where F = `aggregate_residual` (S equations: labour share, sector
shares 2..S, model minus target). The parametrisation keeps χ^L ∈ (0, 1) and χ^s on the
simplex without constraints. Without `aggregate_coordinates_init` the solve starts from
`aggregate_init`.

Newton: x ← x − J⁻¹ F(x), with J = ∂F/∂x the S × S Jacobian. J has no closed form here
(F runs through the CES aggregation of `aggregate_shares`), so it is approximated column
by column by CENTRAL differences, `J[:, j] ≈ (F(x + h e_j) − F(x − h e_j)) / 2h` (error
O(h²), h = 1e-6 · max(1, |x_j|)); this costs 2S evaluations of F per iteration, cheap at
O(S R_d) each. F itself is exact, so an approximate J only slows convergence: the root
is found to `tol` whatever the J error. Backtracking: the step is halved (up to 30 times)
until ‖F‖² decreases. A singular J (or a non-finite step) falls back to a damped
gradient step −½ J'F. Stops at max |F| < tol, or when no step improves ‖F‖².
`converged` is max |F| < 10 tol.
"""
function solve_aggregates(P_sr::AbstractMatrix; aggregate_coordinates_init = nothing,
                          targets = profile_targets(), tol::Float64 = 1e-12, max_iter::Int = 50)
    coordinates = aggregate_coordinates_init === nothing ? aggregate_init(P_sr, targets) :
                  collect(Float64, aggregate_coordinates_init)
    residual = aggregate_residual(coordinates, P_sr, targets)   # model − target of blocks 1–2 (zero at the root)
    n_unknowns = length(coordinates)
    jacobian = zeros(n_unknowns, n_unknowns)
    iterations = 0
    for it in 1:max_iter
        iterations = it
        maximum(abs, residual) < tol && break
        # 1) Central-difference Jacobian, one column per unknown.
        for j in 1:n_unknowns
            step_size = 1e-6 * max(1.0, abs(coordinates[j]))
            coordinates_plus  = copy(coordinates); coordinates_plus[j]  += step_size
            coordinates_minus = copy(coordinates); coordinates_minus[j] -= step_size
            jacobian[:, j] = (aggregate_residual(coordinates_plus, P_sr, targets) .-
                              aggregate_residual(coordinates_minus, P_sr, targets)) ./ (2step_size)
        end
        # 2) Newton step, or a damped gradient step if the Jacobian is singular.
        newton_step = try -(jacobian \ residual) catch; -0.5 .* (jacobian' * residual) end
        all(isfinite, newton_step) || (newton_step = -0.5 .* (jacobian' * residual))
        # 3) Backtracking: halve the step until the squared residual falls.
        step_length = 1.0; improved = false
        for _ in 1:30
            residual_trial = aggregate_residual(coordinates .+ step_length .* newton_step, P_sr, targets)
            if all(isfinite, residual_trial) && sum(abs2, residual_trial) < sum(abs2, residual)
                coordinates = coordinates .+ step_length .* newton_step; residual = residual_trial; improved = true
                break
            end
            step_length /= 2
        end
        improved || break
    end
    max_residual = maximum(abs, residual)
    return (aggregate_coordinates = coordinates, converged = max_residual < 10 * tol,
            max_residual = max_residual, iterations = iterations)
end

"""
    profile_A(c_r, sales_shares) -> A (R_d) with A_1 = 1

Downstream productivity in closed form from the sales shares:
`π_r ∝ δ_r (c_r/A_r)^ε ⇒ A_r ∝ c_r (π_r/δ_r)^{−1/ε}`.
"""
function profile_A(c_r::AbstractVector, sales_shares::AbstractVector)
    A = c_r .* (sales_shares ./ delta_r[DOWNSTREAM_REGIONS]) .^ (-1 / epsilon)
    return A ./ A[1]
end


"""
    logistic(z) -> 1 / (1 + e^{−z}), evaluated without overflow for large |z|
"""
logistic(z::Real) = z >= 0 ? 1 / (1 + exp(-z)) : exp(z) / (1 + exp(z))

"""
    import_logit_shift(log_odds, purchases_r, total_imports; tol = 1e-14, max_iter = 200)
        -> (shift, converged)

The common shift b of the buyers' log-odds that solves the adding-up
`g(b) = Σ_r X_r Λ(log_odds_r + b) − M = 0`, with Λ the logistic function, X_r = `purchases_r`
and M = `total_imports`. g is strictly increasing (g′ = Σ_r X_r Λ_r (1 − Λ_r) > 0) and runs
from −M to Σ_r X_r − M, so the root exists and is unique whenever 0 < M < Σ_r X_r, i.e. for
any aggregate import share in (0, 1). Newton steps, kept inside a bracket and replaced by a
bisection when they leave it; `converged = false` only if no bracket is found.
"""
function import_logit_shift(log_odds::AbstractVector, purchases_r::AbstractVector, total_imports::Real;
                            tol::Float64 = 1e-14, max_iter::Int = 200)
    adding_up_residual(shift) = sum(purchases_r[dr] * logistic(log_odds[dr] + shift) for dr in eachindex(log_odds)) - total_imports
    # 1) Bracket the root: g(lower) ≤ 0 ≤ g(upper).
    lower, upper = -1.0, 1.0
    while adding_up_residual(lower) > 0 && lower > -1e3; lower *= 2; end
    while adding_up_residual(upper) < 0 && upper < 1e3; upper *= 2; end
    (adding_up_residual(lower) <= 0 <= adding_up_residual(upper)) || return (shift = NaN, converged = false)
    # 2) Newton from b = 0 (the shift is zero when the shares already add up), safeguarded.
    shift = clamp(0.0, lower, upper)
    for _ in 1:max_iter
        residual = adding_up_residual(shift)
        abs(residual) <= tol * total_imports && return (shift = shift, converged = true)
        residual < 0 ? (lower = shift) : (upper = shift)
        slope = sum(purchases_r[dr] * logistic(log_odds[dr] + shift) * (1 - logistic(log_odds[dr] + shift))
                    for dr in eachindex(log_odds))
        shift_new = shift - residual / slope
        (isfinite(shift_new) && lower < shift_new < upper) || (shift_new = (lower + upper) / 2)
        shift = shift_new
        upper - lower < 1e-15 && break
    end
    return (shift = shift, converged = true)
end

"""
    buyer_import_targets(omega_sr, PhiF_sr, Phi, import_share_s, customs_share)
        -> (model (R_d), target (R_d), logit_shift, purchases (R_d), adding_up_gap, feasible)

The buyers' import shares in the model and their targets at the expenditure weights
`omega_sr` (S × R_d, total spending of buyer r on sector s, imports included). With
X_r = Σ_s ω_sr, X_s = Σ_r ω_sr and f_sr = ψ_r Φ^F_s / Φ_sr:

  * `model[r]  = Σ_s ω_sr f_sr / X_r`, buyer r's import share;
  * `target[r] = Λ(logit IS_r^customs + b)`, b = `logit_shift` (`import_logit_shift`): one
    common shift of the log-odds of the RAW customs shares, set so that the buyers' imports
    add up to the sectors' imports at the current weights, Σ_r X_r target_r = Σ_s X_s f̂_s.
    The customs data enter through their odds ratios only, odds(target_r)/odds(target_r') =
    odds(IS_r)/odds(IS_r'); for small shares the rule is ≈ the multiplicative c · IS_r
    (c = e^b), near one it rescales the domestic share 1 − IS_r, and every target stays in
    (0, 1). In the model odds(f_r) is proportional to ψ_r (exactly with one importing sector),
    so b acts as a common factor e^b on the buyers' access;
  * `purchases` = X_r, the weights of the adding-up;
  * `adding_up_gap` = |Σ_r X_r target_r − Σ_s X_s f̂_s| relative to the latter (0 up to the
    root's tolerance);
  * `feasible` = the shift was found (always, for shares in (0, 1) and an aggregate import
    share in (0, 1)).
"""
function buyer_import_targets(omega_sr::AbstractMatrix, PhiF_sr::AbstractMatrix, Phi::AbstractMatrix,
                              import_share_s::AbstractVector, customs_share::AbstractVector)
    purchases_r = vec(sum(omega_sr; dims = 1))
    purchases_s = vec(sum(omega_sr; dims = 2))
    imports_r = zeros(R_downstream)
    @inbounds for dr in 1:R_downstream, s in 1:S
        (PhiF_sr[s, dr] > 0 && Phi[s, dr] >= 1e-300) || continue
        imports_r[dr] += omega_sr[s, dr] * PhiF_sr[s, dr] / Phi[s, dr]
    end
    model = [purchases_r[dr] > 0 ? imports_r[dr] / purchases_r[dr] : 0.0 for dr in 1:R_downstream]
    sector_imports = sum(purchases_s .* import_share_s)
    log_odds = log.(customs_share ./ (1 .- customs_share))
    root = import_logit_shift(log_odds, purchases_r, sector_imports)
    target = logistic.(log_odds .+ root.shift)
    adding_up_gap = abs(sum(purchases_r .* target) - sector_imports) / max(sector_imports, eps())
    return (model = model, target = target, logit_shift = root.shift, purchases = purchases_r, adding_up_gap = adding_up_gap,
            feasible = root.converged && all(isfinite, target))
end

"""
    profile_ge(alpha; kappa, gamma_target=EMP_GAMMA_T_TILDE, targets=profile_targets(),
               customs_share=CUSTOMS_SHARE, T_init=copy(T_rs_init), psi_init=ones(R_d),
               aggregate_coordinates_init=nothing, max_iter=2000, tol=1e-9, damping=0.3)
        -> (T, psi, chi_L, chi_s, A, aggregate_coordinates, iterations, converged,
            max_log_step_T, max_log_step_psi, max_residual_aggregates, buyer_import_share,
            import_share_target, import_logit_shift, max_adding_up_gap)

Joint profile of T, ψ and the aggregates at (α, κ), by a damped Sinkhorn iteration on T and ψ.
Each pass: P_sr from the current (T, ψ) → (χ^L, χ^s) solved at those prices (warm-started from
the previous pass) → ω_sr = π_r (1−ℓ_r) s_rs → closed-form γ and buyer import shares f_r →
the multiplicative updates
`T⁺[s,a] = T[s,a] (γ_target[a,s] / γ_model[a,s])^damping`, renormalised to `T[s,ref] = 1`, and
`ψ⁺_r = ψ_r (odds(IS_r^target) / odds(f_r))^damping`, odds(p) = p/(1 − p) (`buyer_import_targets`),
renormalised to ψ_ref = 1
with Φ^F_s multiplied by the same ψ_ref, so the products ψ_r Φ^F_s and every f_rs are unchanged.
`gamma_target` is T_COL_DIM × S (areas then the foreign row, per sector); only its
per-sector profile matters (the reference gauge absorbs the scale). A ZERO-target column
(a fake world in which an area active in the data wins nothing) is held fixed relative to
the reference and left out of the stopping rule. `customs_share` (R_d) are the buyers' RAW
customs import shares, in (0, 1), of which only the odds ratios are used: the targets are
Λ(logit IS_r^customs + b), b re-solved at every pass (`buyer_import_targets`), so they are
always in (0, 1); a pass at which b cannot be solved stops the profile as not converged.
A is computed once at the end.
`PSI_CALIBRATED[] = false` holds ψ at `psi_init` (a test device: the model without ψ).

Returns `T` (S × T_COL_DIM, area level with Φ^F_s in FOREIGN_COL), `psi` (R_d), the
aggregates, the final aggregate coordinates, the number of Sinkhorn passes, the largest
|Δ log T| and |Δ log ψ| of the last pass, `max_residual_aggregates` (max |F| of the final
aggregate solve), the buyers' model import shares, their targets and the logit shift b at
the final point, and the largest relative adding-up gap over the passes. `converged` requires
the T and ψ iteration and the final aggregates solve to converge, b to be solved and every
buyer's import share to sit on its target (`buyer_imports_fitted`).
"""
function profile_ge(alpha; kappa::Real,
                          gamma_target::AbstractMatrix = EMP_GAMMA_T_TILDE,
                          targets = profile_targets(),
                          customs_share::AbstractVector = CUSTOMS_SHARE,
                          T_init::AbstractMatrix = copy(T_rs_init),
                          psi_init::AbstractVector = ones(R_downstream),
                          aggregate_coordinates_init = nothing,
                          max_iter::Int = 2000, tol::Float64 = 1e-9, damping::Float64 = 0.3)
    # 1) Start from T_init in the reference gauge, T[s, ref] = 1, and ψ_ref = 1.
    tau   = build_tau(alpha)
    # T_area (S × T_COL_DIM): the Fréchet scales at the AREA level, T_a,s for every attraction
    # area a and, in column FOREIGN_COL, the foreign scale Φ^F_s.
    # The cell-level T_ls = T_{a(l)s} n_ls^κ is rebuilt from it at every pass.
    T_area = Matrix{Float64}(undef, S, T_COL_DIM)
    T_area .= T_init
    for s in 1:S
        reference_area = T_REF_REGION[s]
        (reference_area > 0 && T_area[s, reference_area] > 0) && (T_area[s, :] ./= T_area[s, reference_area])
    end
    psi = collect(Float64, psi_init)
    rescale_psi_gauge!(T_area, psi)
    calibrate_psi = PSI_CALIBRATED[] && any(FOREIGN_ACTIVE)
    import_share_s = [FOREIGN_ACTIVE[s] ? gamma_target[FOREIGN_COL, s] : 0.0 for s in 1:S]
    coordinates = aggregate_coordinates_init
    max_log_step = Inf; max_log_step_psi = 0.0; iterations = 0
    max_adding_up_gap = 0.0; feasible = true
    T_area_new = similar(T_area)
    psi_new = similar(psi)
    omega_sr = zeros(S, R_downstream)
    for it in 1:max_iter
        iterations = it
        # 2) Solve the aggregates (χ^L, χ^s) at the prices implied by the current (T, ψ).
        T_ls = gather_T_to_ze(T_area, kappa)  # cell level, S × R: T_ls = T_{a(l)s} · n_ls^κ (density term)
        PhiF_sr = foreign_scale_by_buyer(foreign_phi(T_area), psi)   # ψ_r Φ^F_s; 0 where s does not import
        P_sr, Phi = expected_sector_prices(T_ls, tau; PhiF = PhiF_sr)
        coordinates = solve_aggregates(P_sr; aggregate_coordinates_init = coordinates,
                                       targets = targets).aggregate_coordinates
        # 3) Compute the model γ with the expenditure weights pinned at the target sales shares.
        _, _, ell_r, s_rs, _, _ = aggregate_shares(coordinates, P_sr)
        @inbounds for dr in 1:R_downstream, s in 1:S
            omega_sr[s, dr] = targets.sales_shares[dr] * (1 - ell_r[dr]) * s_rs[s, dr]
        end
        gamma_model = gamma_T_from_weights(T_ls, PhiF_sr, tau, Phi, omega_sr)
        # 4) Move T towards the target γ: an area that the model under-supplies relative to
        #    the data gets a higher productivity, and conversely. Then renormalise each
        #    sector to its reference area, and hold the zero-target columns where they are.
        T_area_new .= T_area
        for s in 1:S
            sector_columns = SECTOR_T_COLS[s]
            isempty(sector_columns) && continue
            for a in sector_columns
                gamma_model_as = gamma_model[a, s]; gamma_target_as = gamma_target[a, s]
                ratio = (gamma_model_as > 1e-300 && gamma_target_as > 0) ? gamma_target_as / gamma_model_as : 1.0
                T_area_new[s, a] = exp(log(T_area[s, a]) + damping * log(ratio))
            end
            reference_value = T_area_new[s, T_REF_REGION[s]]
            reference_value > 0 && (T_area_new[s, sector_columns] ./= reference_value)
            for a in sector_columns
                gamma_target[a, s] > 0 || (T_area_new[s, a] = T_area[s, a])
            end
        end
        # 5) Move ψ towards the buyers' import-share targets: a buyer that imports less in
        #    the model than its (logit-shifted) customs share gets a better access to foreign
        #    suppliers, and conversely. The shift is set at the current weights, so the
        #    buyers' imports and the sectors' imports have the same total.
        pass_max_log_step_psi = 0.0
        if calibrate_psi
            buyer_imports = buyer_import_targets(omega_sr, PhiF_sr, Phi, import_share_s, customs_share)
            max_adding_up_gap = max(max_adding_up_gap, buyer_imports.adding_up_gap)
            if !buyer_imports.feasible
                feasible = false
                break
            end
            # The update moves the ODDS of the buyer's import share: odds(f_r) is proportional
            # to ψ_r (exactly with one importing sector), so its elasticity is one at any level,
            # whereas a share ratio would stall as f_r nears one (d log f_r / d log ψ_r = 1 − f_r).
            for dr in 1:R_downstream
                model_share, target_share = buyer_imports.model[dr], buyer_imports.target[dr]
                odds_ratio = 0 < model_share < 1 ?
                             (target_share / (1 - target_share)) / (model_share / (1 - model_share)) : 1.0
                psi_new[dr] = exp(log(psi[dr]) + damping * log(odds_ratio))
            end
            rescale_psi_gauge!(T_area_new, psi_new)
            for dr in 1:R_downstream
                log_step = abs(log(psi_new[dr]) - log(psi[dr]))
                log_step > pass_max_log_step_psi && (pass_max_log_step_psi = log_step)
            end
            psi .= psi_new
        end
        # 6) Stop when no T entry with a positive target and no ψ_r moves by more than `tol` in logs.
        pass_max_log_step = 0.0
        for s in 1:S, a in SECTOR_T_COLS[s]
            gamma_target[a, s] > 0 || continue
            log_step = abs(log(T_area_new[s, a]) - log(T_area[s, a]))
            log_step > pass_max_log_step && (pass_max_log_step = log_step)
        end
        T_area .= T_area_new
        max_log_step = pass_max_log_step
        max_log_step_psi = pass_max_log_step_psi
        max(max_log_step, max_log_step_psi) < tol && break
    end
    # 7) Solve the aggregates at the final (T, ψ), and A in closed form from the sales shares.
    PhiF_sr = foreign_scale_by_buyer(foreign_phi(T_area), psi)
    P_sr, Phi = expected_sector_prices(gather_T_to_ze(T_area, kappa), tau; PhiF = PhiF_sr)
    aggregate_solution = solve_aggregates(P_sr; aggregate_coordinates_init = coordinates, targets = targets)
    chi_L, chi_s, ell_r, s_rs, _, c_r = aggregate_shares(aggregate_solution.aggregate_coordinates, P_sr)
    A = profile_A(c_r, targets.sales_shares)
    # 8) The buyers' import shares and their targets at the final point (reported).
    omega_final = [targets.sales_shares[dr] * (1 - ell_r[dr]) * s_rs[s, dr] for s in 1:S, dr in 1:R_downstream]
    buyer_imports = buyer_import_targets(omega_final, PhiF_sr, Phi, import_share_s, customs_share)
    return (T = T_area, psi = psi, chi_L = chi_L, chi_s = chi_s, A = A,
            aggregate_coordinates = aggregate_solution.aggregate_coordinates, iterations = iterations,
            converged = feasible && max(max_log_step, max_log_step_psi) < tol &&
                        aggregate_solution.converged && all(isfinite, A) && all(isfinite, psi) &&
                        (!calibrate_psi || buyer_imports_fitted(buyer_imports)),
            max_log_step_T = max_log_step, max_log_step_psi = max_log_step_psi,
            max_residual_aggregates = aggregate_solution.max_residual,
            buyer_import_share = buyer_imports.model, import_share_target = buyer_imports.target,
            import_logit_shift = buyer_imports.logit_shift, max_adding_up_gap = max_adding_up_gap)
end

"""
    buyer_imports_fitted(buyer_imports; tol = 1e-6) -> Bool

True when every buyer's model import share is finite and on its target (`buyer_import_targets`).
The logit targets are always in (0, 1), but the buyers' and the sectors' import shares must
also hold together: buyer r cannot import more of sector s than ω_rs, and the sector's
imports are X_s f̂_s. When the targets violate this joint constraint (a large buyer asked to
import almost all its purchases while a sector imports little), ψ_r diverges with Φ^F_s → 0
and the step rule alone can stop on a non-finite or off-target point: this check rejects it.
"""
buyer_imports_fitted(buyer_imports; tol::Float64 = 1e-6) =
    all(isfinite, buyer_imports.model) && maximum(abs.(buyer_imports.model .- buyer_imports.target)) < tol

"""
    rescale_psi_gauge!(T_area, psi) -> nothing

Divide every ψ_r by ψ_ref (`PSI_REF_BUYER`) and multiply every Φ^F_s (column FOREIGN_COL
of `T_area`) by it: the products ψ_r Φ^F_s, hence every import share f_rs, are unchanged.
"""
function rescale_psi_gauge!(T_area::AbstractMatrix, psi::AbstractVector)
    psi_reference = psi[PSI_REF_BUYER]
    psi_reference == 1 && return nothing
    psi ./= psi_reference
    for s in 1:S
        FOREIGN_ACTIVE[s] && (T_area[s, FOREIGN_COL] *= psi_reference)
    end
    return nothing
end

# Per-worker warm start: (T*, aggregate coordinates*, ψ*) of the last converged call. The fixed
# point is unique, so a CONVERGED result does not depend on the start; the cache is written
# on convergence only, and a failed warm attempt is retried from T_rs_init and ψ ≡ 1.
const PROFILE_WARM_CACHE = Ref{Union{Nothing, Tuple{Matrix{Float64}, Vector{Float64}, Vector{Float64}}}}(nothing)

"""
    profile_ge_warm(alpha; kwargs...) -> the result of `profile_ge`, warm-started from the cache
"""
function profile_ge_warm(alpha; kwargs...)
    # Warm start. Successive objective evaluations on one worker (`profile_theta`, e.g. the
    # PSO particles a worker scores) sit at nearby (α, κ), and the Sinkhorn fixed point converges linearly
    # (≈ 0.98 per pass near α ≈ 1, hundreds of passes from T_rs_init). Starting from the last
    # converged (T*, x*, ψ*) cuts that count; it is a SPEED device only, since a converged result
    # is the unique fixed point (up to the stopping tolerance, ≈ 1e-9 in logs, which is why
    # `sens_profile` clears the cache when bit-reproducibility is required).
    cached_start = PROFILE_WARM_CACHE[]
    if cached_start !== nothing && all(isfinite, cached_start[1])
        profile = profile_ge(alpha; T_init = copy(cached_start[1]),
                             aggregate_coordinates_init = copy(cached_start[2]),
                             psi_init = copy(cached_start[3]), kwargs...)
        profile.converged && (PROFILE_WARM_CACHE[] = (copy(profile.T), copy(profile.aggregate_coordinates), copy(profile.psi)); return profile)
    end
    profile = profile_ge(alpha; kwargs...)
    profile.converged && (PROFILE_WARM_CACHE[] = (copy(profile.T), copy(profile.aggregate_coordinates), copy(profile.psi)))
    return profile
end

"""
    with_aggregates(theta_vec, χ^L, χ^s, A) -> copy of the level vector `theta_vec` with its
    aggregates set (χ^L in position 1, χ^s (S) next, then A (R_d))
"""
function with_aggregates(theta_vec::AbstractVector, chi_L::Real, chi_s::AbstractVector, A::AbstractVector)
    theta_vec_new = collect(Float64, theta_vec)
    theta_vec_new[1] = chi_L
    theta_vec_new[2:(1 + S)] = chi_s
    theta_vec_new[(S + 2):(S + R_downstream + 1)] = A
    return theta_vec_new
end

"""
    aggregate_moment_residuals(χ^L, χ^s, A, T_ls, tau; targets, PhiF = nothing) -> (labor, industry, pi_r)

Analytical blocks 1–3 at a θ against their targets (max |model − target| per block),
from the full closed-form price solve, in which sales come from A rather than being
pinned at their target. ≈ 0 at the profile; the gap to the SIMULATED blocks is the
Monte-Carlo error of the realised price indices.
"""
function aggregate_moment_residuals(chi_L, chi_s, A, T_ls::AbstractMatrix, tau::AbstractMatrix;
                               targets = profile_targets(), PhiF = nothing)
    prices   = compute_prices_analytical(chi_L, chi_s, A, T_ls, tau; PhiF = PhiF)
    ell_r    = chi_L .* (regional_wages[DOWNSTREAM_REGIONS] ./ prices.c_r) .^ (1 - lambda)
    pi_model = prices.Y_r ./ sum(prices.Y_r)
    s_rs     = chi_s .* (prices.P_sr ./ reshape(prices.P_r, 1, :)) .^ (1 - nu)
    intermediate_bill = pi_model .* (1 .- ell_r)
    return (labor = abs(sum(pi_model .* ell_r) - targets.labor_share),
            industry = maximum(abs, (s_rs * intermediate_bill) ./ sum(intermediate_bill) .- targets.sector_shares),
            pi_r = maximum(abs, pi_model .- targets.sales_shares))
end

"""
    buyer_import_fit(theta_vec; customs_share = CUSTOMS_SHARE)
        -> (model, target, logit_shift, purchases, psi, amplification)  (each R_d but `logit_shift`)

The buyers' import shares in closed form at a level vector (expected prices, the spending
weights of the full price solve), their targets Λ(logit IS_r^customs + b) and the shift b
(`buyer_import_targets`), ψ (ψ_ref = 1) and the closed-form amplification coefficient
`D_r = 1 + (1 − ℓ_r) Σ_s s_rs (1 − f_rs)`: the sales of DOMESTIC suppliers generated by one
euro of final demand for buyer r (imports leak abroad), on top of the buyer's own euro.
"""
function buyer_import_fit(theta_vec::AbstractVector; customs_share::AbstractVector = CUSTOMS_SHARE)
    chi_L, chi_s, A, alpha, T_vec, PhiF_s, psi = unpack_params(theta_vec)
    T_ls = reshape(collect(T_vec), S, R)
    PhiF_sr = foreign_scale_by_buyer(PhiF_s, psi)
    prices = compute_prices_analytical(chi_L, chi_s, A, T_ls, build_tau(alpha); PhiF = PhiF_sr)
    # Spending of buyer r on sector s, and the shares of its unit cost.
    intermediate_share_r = (1 - chi_L) .* (prices.P_r ./ prices.c_r) .^ (1 - lambda)       # 1 − ℓ_r
    s_rs = [chi_s[s] * (prices.P_sr[s, dr] / prices.P_r[dr])^(1 - nu) for s in 1:S, dr in 1:R_downstream]
    omega_sr = s_rs .* reshape(intermediate_share_r .* prices.mu .* prices.Y_r, 1, :)
    import_share_s = [FOREIGN_ACTIVE[s] ? FOREIGN_SHARE[s] : 0.0 for s in 1:S]
    fit = buyer_import_targets(omega_sr, PhiF_sr, prices.Phi, import_share_s, customs_share)
    f_sr = [PhiF_sr[s, dr] > 0 ? PhiF_sr[s, dr] / prices.Phi[s, dr] : 0.0 for s in 1:S, dr in 1:R_downstream]
    amplification = 1 .+ intermediate_share_r .* vec(sum(s_rs .* (1 .- f_sr); dims = 1))
    return (model = fit.model, target = fit.target, logit_shift = fit.logit_shift, purchases = fit.purchases, psi = collect(psi),
            amplification = amplification)
end

"""
    profile_theta(theta_vec) -> (theta_vec, converged, max_residual, iterations, alpha)

The full level vector at the (α, κ) read from `theta_vec`: (χ^L, χ^s, A, T, ψ) come from
`profile_ge` (χ^s on the simplex, A_1 = 1, as `unpack_params` reads them), and every
other entry of the input is ignored. `max_residual` is the larger of the T step and the
aggregate residual. On `converged = false` the profiled blocks are the last iterate: the
caller must reject it, not score it.
"""
function profile_theta(theta_vec::AbstractVector)
    alpha = collect(theta_vec[alpha_range()])
    profile = profile_ge_warm(alpha; kappa = unpack_kappa(theta_vec))
    return (theta_vec = with_psi(with_T(with_aggregates(theta_vec, profile.chi_L, profile.chi_s, profile.A), profile.T), profile.psi),
            converged = profile.converged,
            max_residual = max(profile.max_log_step_T, profile.max_log_step_psi, profile.max_residual_aggregates),
            iterations = profile.iterations, alpha = alpha)
end
