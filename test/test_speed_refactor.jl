####################################################################################
# test_speed_refactor.jl — verify the performance refactor changes NOTHING numerically
#
# Gates three changes made for speed:
#   S1  fused Ricardian min/argmin in `solve_network` (no materialised price matrix)
#   S2  cached cloglog design + O(n) FE group index (`build_fe_group_rows`)
#   S3  allocation-free IRLS inner loop
#
# Each is checked against a REFERENCE implementation of the pre-refactor code, kept
# verbatim in this file, evaluated on the SAME draws and the SAME parameter vector.
# The gate is agreement to floating-point noise, and the actual max deviation is
# printed so you can see it rather than trust a threshold.
#
# It also gates the cache-invalidation logic, which is the only genuinely new risk:
#   S4  a second call at a DIFFERENT θ must give what a cold (uncached) build gives
#   S5  a call with a DIFFERENT draw set must rebuild, not reuse
#
# Run:
#   julia test/test_speed_refactor.jl <industry> <n_coef> <n_tau> <granular> <ca_level>
#   julia test/test_speed_refactor.jl auto 4 1 true aa
#
# Print-only. Touches no production artefact.
####################################################################################

using Distributed, Random, Printf, LinearAlgebra, Statistics, NPZ

industry   = length(ARGS) >= 1 ? ARGS[1] : "auto"
n_coef     = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4
n_tau      = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1
granular   = length(ARGS) >= 4 ? (lowercase(ARGS[4]) in ("true","1","yes")) : true
ca_level   = length(ARGS) >= 5 ? Symbol(ARGS[5]) : :aa
draw_method = :sobol

include("../model_CP.jl")
include("../tools.jl")
include("../model_analytical.jl")
include("../profiling.jl")
include("../load_parameters.jl")

const PASS = Ref(true)
record!(name, ok) = (ok || (PASS[] = false);
                     @printf("  %-46s %s\n", name, ok ? "PASS" : "FAIL"))

# ══════════════════════════════════════════════════════════════════════════════
# Reference implementations — the code as it stood BEFORE the refactor
# ══════════════════════════════════════════════════════════════════════════════

"""Pre-refactor IRLS: Dict/findall FE index, per-iteration copies, fancy indexing."""
function _cloglog_irls_ref(y, X, w, fe_group; max_iter::Int=50, tol::Float64=1e-9,
                           eta_clamp::Float64=30.0)
    n, k = size(X)
    yf = Float64.(y); Xf = Float64.(X); wf = Float64.(w)
    groups  = unique(fe_group)
    rows_of = Dict(g => findall(==(g), fe_group) for g in groups)
    μ = (yf .+ 0.5) ./ 2
    η = clamp.(log.(.-log.(1 .- μ)), -eta_clamp, eta_clamp)
    β = zeros(k)
    for _ in 1:max_iter
        μ     = clamp.(1 .- exp.(.-exp.(η)), 1e-12, 1 - 1e-12)
        dμdη  = exp.(η) .* (1 .- μ)
        zwork = η .+ (yf .- μ) ./ dμdη
        W     = wf .* (dμdη .^ 2) ./ (μ .* (1 .- μ))
        zt = copy(zwork); Xt = copy(Xf)
        for g in groups
            r  = rows_of[g]
            sw = sum(@view W[r])
            sw < 1e-300 && continue
            zt[r] .-= sum(W[r] .* zwork[r]) / sw
            @inbounds for j in 1:k
                Xt[r, j] .-= sum(W[r] .* Xf[r, j]) / sw
            end
        end
        WXt   = W .* Xt
        β_new = (Xt' * WXt) \ (Xt' * (W .* zt))
        resid = zwork .- Xf * β_new
        η_new = Xf * β_new
        for g in groups
            r  = rows_of[g]
            sw = sum(@view W[r])
            sw < 1e-300 && continue
            η_new[r] .+= sum(W[r] .* resid[r]) / sw
        end
        η_new = clamp.(η_new, -eta_clamp, eta_clamp)
        Δ = maximum(abs.(β_new .- β))
        β = β_new; η = η_new
        Δ < tol && break
    end
    return β
end

"""Pre-refactor cloglog: rebuilds the whole design, then calls the reference IRLS."""
function fast_cloglog_regression_ref(linkages_flat, z_flat, sample_weights::Matrix{Float64};
                                     include_control::Bool=true,
                                     include_size_control::Bool=!include_control,
                                     return_size_coef::Bool=false,
                                     max_iter::Int=50, tol::Float64=1e-9)
    n_size       = include_size_control ? 1 : 0
    n_regressors = N_REG + n_size
    size_col     = N_REG + 1
    N_rho_eff    = size(sample_weights, 1)
    n_rows_goods = n_good * N_rho_eff
    n_ctrl_eff   = include_control ? N_CONTROL : 0
    N_valid      = n_rows_goods + n_ctrl_eff * N_rho_eff

    y = Vector{Float64}(undef, N_valid); X = zeros(N_valid, n_regressors)
    w = Vector{Float64}(undef, N_valid); fe_group = Vector{Int}(undef, N_valid)

    idx = 0
    for g in 1:n_good
        s = GOOD_S[g]; r = GOOD_R[g]; dr = CLOSEST_DOWNSTREAM_REGION[r]
        group_id = (s - 1) * R_downstream + dr
        local b, log_dist
        N_REG == 1 ? (log_dist = LOG_CLOSEST_DIST[r]) : (b = DistBin[r, dr])
        for rho in 1:N_rho_eff
            idx += 1
            y[idx] = linkages_flat[rho, g] > 0 ? 0.0 : 1.0
            w[idx] = sample_weights[rho, g]
            fe_group[idx] = group_id
            N_REG == 1 ? (X[idx, 1] = log_dist) :
                         ((b > 0 && b <= N_REG) && (X[idx, b] = 1.0))
            include_size_control && (X[idx, size_col] = log(z_flat[rho, g]))
        end
    end
    if include_control
        w_ctrl = 1.0 / N_rho_eff
        for c in 1:N_CONTROL
            s = CONTROL_S[c]; r = CONTROL_R[c]; dr = CLOSEST_DOWNSTREAM_REGION[r]
            group_id = (s - 1) * R_downstream + dr
            local b, log_dist
            N_REG == 1 ? (log_dist = LOG_CLOSEST_DIST[r]) : (b = DistBin[r, dr])
            for rho in 1:N_rho_eff
                idx += 1
                y[idx] = 1.0; w[idx] = w_ctrl; fe_group[idx] = group_id
                N_REG == 1 ? (X[idx, 1] = log_dist) :
                             ((b > 0 && b <= N_REG) && (X[idx, b] = 1.0))
            end
        end
    end
    β = _cloglog_irls_ref(y, X, w, fe_group; max_iter=max_iter, tol=tol)
    return return_size_coef ?
        vcat(β[1:N_REG], include_size_control ? β[size_col] : NaN) : β[1:N_REG]
end

"""Pre-refactor Ricardian selection: materialise the price matrix, then argmin."""
function ricardian_winners_ref(z_inv_flat, tau, r_d)
    N_rho_eff = size(z_inv_flat, 1)
    winner = zeros(Int, N_rho_eff, S)
    price  = zeros(N_rho_eff, S)
    for s in 1:S
        g_indices = SECTOR_GOOD_INDICES[s]
        isempty(g_indices) && continue
        regions_s = SECTOR_GOOD_REGIONS[s]
        tau_sr = reshape(tau[regions_s, r_d], 1, :)
        w_sr   = reshape(W_RS_FLAT[g_indices], 1, :)
        prices_s = z_inv_flat[:, g_indices] .* tau_sr .* w_sr
        min_local = argmin(prices_s, dims=2)
        for rho in 1:N_rho_eff
            li = min_local[rho][2]
            winner[rho, s] = g_indices[li]
            price[rho, s]  = prices_s[rho, li]
        end
    end
    return winner, price
end

"""New path, extracted so it can be compared row-for-row against the reference."""
function ricardian_winners_new(z_inv_flat, tau, r_d)
    N_rho_eff = size(z_inv_flat, 1)
    winner = zeros(Int, N_rho_eff, S)
    price  = zeros(N_rho_eff, S)
    for s in 1:S
        g_indices = SECTOR_GOOD_INDICES[s]
        isempty(g_indices) && continue
        regions_s = SECTOR_GOOD_REGIONS[s]
        tv = tau[regions_s, r_d]; wv = W_RS_FLAT[g_indices]; n_act = length(g_indices)
        @inbounds for rho in 1:N_rho_eff
            g1 = g_indices[1]
            best = z_inv_flat[rho, g1] * tv[1] * wv[1]; bidx = 1
            if !isnan(best)
                for li in 2:n_act
                    gl = g_indices[li]
                    p  = z_inv_flat[rho, gl] * tv[li] * wv[li]
                    if isnan(p); best = p; bidx = li; break
                    elseif p < best; best = p; bidx = li; end
                end
            end
            winner[rho, s] = g_indices[bidx]; price[rho, s] = best
        end
    end
    return winner, price
end

# ══════════════════════════════════════════════════════════════════════════════
# A parameter vector to test at
# ══════════════════════════════════════════════════════════════════════════════

function load_theta()
    for p in (joinpath("reporting_$(industry)_profiled_aa_gran_pso", "step3", "theta_hat_2.npy"),
              joinpath("reporting_$(industry)_profiled_aa_gran_pso", "step1", "best_params.npy"),
              joinpath("reporting_$(industry)", "step3", "theta_hat_2.npy"))
        if isfile(p)
            θ = NPZ.npzread(p); ndims(θ) > 1 && (θ = θ[:, 1])
            println("θ loaded from $p")
            return Vector{Float64}(θ)
        end
    end
    println("No saved θ found — using the documented default (T_rs_init, Ω/A = 1, α = prior).")
    Ω_L = 0.3; Ω_s = fill(1.0 / S, S); A = ones(R_downstream)
    α   = TAU_PRIOR === nothing ? fill(0.5, N_TAU) : collect(TAU_PRIOR)
    return vcat(Ω_L, Ω_s, A, α, vec(permutedims(T_rs_init))[T_MASK])
end

θ = load_theta()
println("\n", "="^78)
println("SPEED-REFACTOR EQUIVALENCE GATES   (industry=$industry, granular=$GRANULAR, ca=$CA_LEVEL)")
println("N_rho = $N_rho | n_good = $n_good | N_REG = $N_REG | design rows = $(n_good*N_rho)")
println("="^78)

net = solve_network(θ; u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS)

# ── S1: Ricardian selection ───────────────────────────────────────────────────
println("\n── S1  fused Ricardian min/argmin ──")
let
    z_inv = net.z_flat .^ (-1)
    tau   = build_tau(unpack_params(θ)[4])
    okw = true; maxdp = 0.0
    for r_d in 1:R_downstream
        wr, pr = ricardian_winners_ref(z_inv, tau, r_d)
        wn, pn = ricardian_winners_new(z_inv, tau, r_d)
        okw &= (wr == wn)
        maxdp = max(maxdp, maximum(abs.(pr .- pn)))
    end
    @printf("  winner indices identical across all %d downstream regions : %s\n",
            R_downstream, okw ? "yes" : "NO")
    @printf("  max |Δ price|                                            : %.3e\n", maxdp)
    record!("S1 Ricardian selection unchanged", okw && maxdp == 0.0)
end

# ── S2/S3: cloglog regression ────────────────────────────────────────────────
println("\n── S2/S3  cached design + allocation-free IRLS ──")
inc_ctrl = REG_INCLUDE_CONTROL; inc_size = REG_INCLUDE_SIZE
println("  design flags: include_control=$inc_ctrl  include_size_control=$inc_size")
let
    reset_cloglog_design!()
    it = Ref(0)
    β_new = fast_cloglog_regression(net.linkages_flat, net.z_flat, SAMPLE_WEIGHTS;
                include_control=inc_ctrl, include_size_control=inc_size,
                return_size_coef=true, iters_out=it)
    β_ref = fast_cloglog_regression_ref(net.linkages_flat, net.z_flat, SAMPLE_WEIGHTS;
                include_control=inc_ctrl, include_size_control=inc_size,
                return_size_coef=true)
    d = maximum(abs.(β_new .- β_ref))
    rel = d / max(maximum(abs.(β_ref)), 1e-12)
    @printf("  β_new  = %s\n", string(round.(β_new, digits=8)))
    @printf("  β_ref  = %s\n", string(round.(β_ref, digits=8)))
    @printf("  max |Δβ| = %.3e   (relative %.3e)   IRLS iterations used: %d\n", d, rel, it[])
    it[] >= 50 && @warn "IRLS hit the 50-iteration cap — β may not be converged; " *
                        "raise max_iter or loosen tol."
    record!("S2/S3 coefficients match reference (rel < 1e-8)", rel < 1e-8)
end

# ── S4: the cache must not go stale when θ moves ─────────────────────────────
println("\n── S4  cache invalidation across θ ──")
let
    θ2 = copy(θ)
    # α block starts after [Ω^L(1) | Ω^s(S) | A(R_downstream)] in the raw layout
    a0 = 2 + S + R_downstream
    θ2[a0] *= 1.35                                   # move α ⇒ different linkages and z
    net2 = solve_network(θ2; u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS)

    # warm cache (holds the design built for θ), then evaluate at θ2
    β_warm = fast_cloglog_regression(net2.linkages_flat, net2.z_flat, SAMPLE_WEIGHTS;
                include_control=inc_ctrl, include_size_control=inc_size, return_size_coef=true)
    # cold: force a rebuild, then evaluate the same thing
    reset_cloglog_design!()
    β_cold = fast_cloglog_regression(net2.linkages_flat, net2.z_flat, SAMPLE_WEIGHTS;
                include_control=inc_ctrl, include_size_control=inc_size, return_size_coef=true)
    β_ref2 = fast_cloglog_regression_ref(net2.linkages_flat, net2.z_flat, SAMPLE_WEIGHTS;
                include_control=inc_ctrl, include_size_control=inc_size, return_size_coef=true)
    d_wc = maximum(abs.(β_warm .- β_cold))
    d_wr = maximum(abs.(β_warm .- β_ref2))
    @printf("  warm-cache vs cold-build : max |Δ| = %.3e\n", d_wc)
    @printf("  warm-cache vs reference  : max |Δ| = %.3e\n", d_wr)
    record!("S4 warm cache == cold build at a new θ", d_wc == 0.0)
    record!("S4 warm cache == reference at a new θ",
            d_wr / max(maximum(abs.(β_ref2)), 1e-12) < 1e-8)
end

# ── S5: a different draw set must force a rebuild ────────────────────────────
println("\n── S5  cache invalidation across draw sets ──")
let
    u2, w2 = generate_draws(N_rho, n_good, :mc; randomise=true, rng=MersenneTwister(4242))
    net3 = solve_network(θ; u_draws=u2, sample_weights=w2)
    β_a = fast_cloglog_regression(net3.linkages_flat, net3.z_flat, w2;
                include_control=inc_ctrl, include_size_control=inc_size, return_size_coef=true)
    β_r = fast_cloglog_regression_ref(net3.linkages_flat, net3.z_flat, w2;
                include_control=inc_ctrl, include_size_control=inc_size, return_size_coef=true)
    d = maximum(abs.(β_a .- β_r)) / max(maximum(abs.(β_r)), 1e-12)
    @printf("  new draw set, cached path vs reference : rel |Δ| = %.3e\n", d)
    # and back to the production weights: must return the original answer
    β_back = fast_cloglog_regression(net.linkages_flat, net.z_flat, SAMPLE_WEIGHTS;
                include_control=inc_ctrl, include_size_control=inc_size, return_size_coef=true)
    β_ref  = fast_cloglog_regression_ref(net.linkages_flat, net.z_flat, SAMPLE_WEIGHTS;
                include_control=inc_ctrl, include_size_control=inc_size, return_size_coef=true)
    d2 = maximum(abs.(β_back .- β_ref)) / max(maximum(abs.(β_ref)), 1e-12)
    @printf("  switching back to production draws     : rel |Δ| = %.3e\n", d2)
    record!("S5 alternating draw sets stay correct", d < 1e-8 && d2 < 1e-8)
end

# ── S6: end-to-end moment vector ─────────────────────────────────────────────
println("\n── S6  full moment vector (all six blocks) ──")
let
    _, m1 = full_SMM(θ; u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS)
    _, m2 = full_SMM(θ; u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS)
    v1 = moments_to_vec(m1); v2 = moments_to_vec(m2)
    d = maximum(abs.(v1 .- v2))
    @printf("  two identical evaluations agree to     : %.3e  (determinism)\n", d)
    record!("S6 evaluation is deterministic", d == 0.0)
end

# ── Timing ───────────────────────────────────────────────────────────────────
println("\n── Timing (single-threaded, this process) ──")
let
    reset_cloglog_design!()
    t_ref = @elapsed for _ in 1:3
        fast_cloglog_regression_ref(net.linkages_flat, net.z_flat, SAMPLE_WEIGHTS;
            include_control=inc_ctrl, include_size_control=inc_size)
    end
    fast_cloglog_regression(net.linkages_flat, net.z_flat, SAMPLE_WEIGHTS;
        include_control=inc_ctrl, include_size_control=inc_size)   # warm the cache
    t_new = @elapsed for _ in 1:3
        fast_cloglog_regression(net.linkages_flat, net.z_flat, SAMPLE_WEIGHTS;
            include_control=inc_ctrl, include_size_control=inc_size)
    end
    @printf("  cloglog regression: reference %.3f s   refactored %.3f s   speedup %.2fx\n",
            t_ref/3, t_new/3, t_ref/max(t_new, 1e-12))

    t_solve = @elapsed for _ in 1:3
        solve_network(θ; u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS)
    end
    t_full = @elapsed for _ in 1:3
        full_SMM(θ; u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS)
    end
    @printf("  solve_network %.3f s | full_SMM %.3f s | regression is %.0f%% of full_SMM\n",
            t_solve/3, t_full/3, 100 * (t_new/3) / max(t_full/3, 1e-12))
end

println("\n", "="^78)
println(PASS[] ? "ALL SPEED-REFACTOR GATES PASSED" : "SOME GATES FAILED — do not deploy")
println("="^78)
