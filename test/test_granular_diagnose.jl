##### test_granular_diagnose.jl #####
# Localise a non-finite loss. STANDALONE + PRINT-ONLY.
#
#     julia test/test_granular_diagnose.jl auto 4 1 true aa
#
# Args: industry n_coef n_tau granular ca_level [profile_T]
#
# Walks the loss evaluation at the warm start in the order the code does it and
# stops at the FIRST non-finite quantity, so the answer is a location rather than a
# guess. The chain is:
#
#   θ warm start → profiled_theta (invert_T_ge)  → T
#                → solve_network                 → z, linkages, X_ls, c_tilde, Y_r
#                → compute_moments               → each of the 5/6 blocks
#                → moments_to_vec                → the masked vector
#                → loss
#
# It also probes the parameter BOX, not just the warm start: a loss that is finite at
# the warm start but Inf for most particles is a bounds problem, not a point problem,
# so the same chain is run at random draws from the PSO box.

using Distributed
@everywhere using Random
@everywhere using NPZ
using LinearAlgebra, Statistics, Printf

@everywhere include("../model_CP.jl")
@everywhere include("../tools.jl")
@everywhere include("../model_analytical.jl")

industry  = length(ARGS) >= 1 ? ARGS[1] : "auto"
n_coef    = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4
n_tau     = length(ARGS) >= 3 && !isempty(strip(ARGS[3])) ? parse(Int, ARGS[3]) : 1
granular  = length(ARGS) >= 4 ? (lowercase(strip(ARGS[4])) in ("true","1","yes")) : true
ca_level  = length(ARGS) >= 5 && !isempty(strip(ARGS[5])) ? Symbol(strip(ARGS[5])) : :aa
profile_T = length(ARGS) >= 6 ? (lowercase(strip(ARGS[6])) in ("true","1","yes")) : true
K_sim     = 10000
reg_method = :cloglog

input_folder  = "./baseline_$industry"
output_folder = "./reporting_$industry" * (ca_level == :aa ? "_aa" : "") *
                (granular ? "_gran" : "") * "_diagnose"
mkpath(output_folder)

include("../load_parameters.jl")
include("../profiling.jl")

fin(x) = all(isfinite, x)
function status(name, x; extra="")
    v = x isa Number ? [x] : vec(collect(x))
    ok = all(isfinite, v)
    nbad = count(!isfinite, v)
    @printf("  %-26s %-8s  n=%-9d %s%s\n", name, ok ? "finite" : "NON-FINITE",
            length(v), ok ? @sprintf("range [%.3e, %.3e]", minimum(v), maximum(v)) :
                            @sprintf("%d bad (%d NaN, %d Inf)", nbad,
                                     count(isnan, v), count(isinf, v)),
            isempty(extra) ? "" : "   " * extra)
    return ok
end

println("\n" * "="^76)
println("NON-FINITE LOSS — LOCALISATION")
@printf("  industry=%s  GRANULAR=%s  CA_LEVEL=:%s  profile_T=%s\n",
        industry, GRANULAR, CA_LEVEL, profile_T)
@printf("  cells=%d  T columns=%d  moments=%d  N_rho=%d  epsilon=%.3g  nu_s=%.3g  theta=%.3g\n",
        n_good, count(T_MASK), length(empirical_moments), N_rho, epsilon, nu_s[1], theta)
println("="^76)

# ── 0. the inputs that never move ───────────────────────────────────────────
println("\n── 0. constants and targets ─────────────────────────────────────────")
status("empirical_moments", empirical_moments)
status("T_rs_init (active)", T_rs_init[T_rs_init .> 0])
status("EMP_GAMMA_T", EMP_GAMMA_T)
status("W_RS_FLAT", W_RS_FLAT)
status("U_DRAWS", U_DRAWS)
status("LOG_DIST_DOWNSTREAM", LOG_DIST_DOWNSTREAM)
@printf("  %-26s min=%.3e   (a zero here makes that cell win every variety)\n",
        "min W_RS_FLAT", minimum(W_RS_FLAT))

A_init = copy(emp_pi_r_full).^(1/abs(epsilon)) .* regional_wages[N_downstream_per_region .!= 0]
A_init ./= sum(A_init)
θ_warm = vcat([agg_labor_share], agg_industry_share, A_init,
              TAU_PRIOR !== nothing ? TAU_PRIOR : fill(P_alpha, N_TAU),
              vec(permutedims(T_rs_init))[T_MASK])

# ── the chain, at one θ ─────────────────────────────────────────────────────
function walk(θ_in; label="warm start", verbose=true)
    verbose && println("\n── $label ───────────────────────────────────────────────────────────")
    θ = θ_in

    if profile_T
        res = invert_T_ge(θ[(S + R_downstream + 2):(1 + S + R_downstream + N_TAU)],
                          θ[1],
                          (x -> x ./ sum(x))(θ[2:(1 + S)]),
                          (x -> x ./ x[1])(θ[(S + 2):(S + R_downstream + 1)]))
        verbose && @printf("  invert_T_ge: converged=%s iters=%d resid=%.3e\n",
                           res.converged, res.iters, res.resid)
        verbose && status("  T* (all entries)", res.T)
        act = [res.T[s, c] for s in 1:S for c in SECTOR_T_COLS[s]]
        verbose && status("  T* (active)", act)
        if !all(isfinite, act) || any(act .<= 0)
            verbose && println("  ⇒ STOP: the Sinkhorn image is non-finite or non-positive. " *
                               "log(T)=−Inf and log(ratio)=+Inf combine to NaN when a T column " *
                               "underflows; widen the α box or damp invert_T_ge harder.")
            return (:T, nothing)
        end
        θ = profiled_theta(θ_in)
        verbose && status("  θ after profiling", θ)
        all(isfinite, θ) || return (:theta, nothing)
    end

    ΩL, Ωs, A, α, Tvec = unpack_params(θ)
    verbose && status("  unpack: T (ZE level)", Tvec)
    verbose && status("  unpack: A", A)
    verbose && @printf("  %-26s %s\n", "alpha", string(round.(α, digits=5)))
    all(isfinite, Tvec) || return (:unpack, nothing)

    net = solve_network(θ; u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS)
    if verbose
        status("  z_flat", net.z_flat)
        status("  log z_flat", log.(max.(net.z_flat, 0.0)); extra="the size control regresses on this")
        status("  linkages_flat", net.linkages_flat)
        status("  X_ls_flat", net.X_ls_flat)
        status("  c_tilde_r (downstream)", net.c_tilde_r[DOWNSTREAM_REGIONS])
        status("  Y_r (downstream)", net.Y_r[DOWNSTREAM_REGIONS])
        @printf("  %-26s %.3e   (p_r^epsilon overflows below ~%.1e for epsilon=%.3g)\n",
                "min c_tilde_r", minimum(net.c_tilde_r[DOWNSTREAM_REGIONS]),
                (floatmax(Float64))^(1/epsilon), epsilon)
        # per-cell win share: the object q̂ and the regression outcome are built from
        nz = [count(>(0), view(net.linkages_flat, :, g)) for g in 1:n_good]
        @printf("  %-26s cells never winning: %d / %d  (their not_supply ≡ 1)\n",
                "win counts", count(==(0), nz), n_good)
        satur = 0; grp = Dict{Int,Vector{Int}}()
        for g in 1:n_good
            gid = (GOOD_S[g] - 1) * R_downstream + CLOSEST_DOWNSTREAM_REGION[GOOD_R[g]]
            push!(get!(grp, gid, Int[]), g)
        end
        for (gid, gs) in grp
            all(nz[g] == 0 for g in gs) && (satur += 1)
        end
        @printf("  %-26s %d / %d fixed-effect groups are SATURATED (y ≡ 1)\n",
                "cloglog groups", satur, length(grp))
    end
    for (nm, v) in (("z_flat", net.z_flat), ("X_ls_flat", net.X_ls_flat),
                    ("c_tilde_r", net.c_tilde_r), ("Y_r", net.Y_r))
        all(isfinite, v) || return (Symbol("network_" * nm), nothing)
    end

    m = compute_moments(net, θ)
    blocks = moment_blocks_tuple(m)
    if verbose
        for (k, nm) in enumerate(BLOCK_NAMES)
            status("  block $k $nm", blocks[k])
        end
        GRANULAR && @printf("  %-26s %s\n", "N_hat", string(m.N_hat))
        GRANULAR && @printf("  %-26s %s\n", "clamped", string(m.clamped))
        GRANULAR && status("  q_hat", m.q_hat)
        @printf("  %-26s %.5f  (theory: -theta = %.3f)\n", "b_logz", m.b_logz, -theta)
    end
    for (k, nm) in enumerate(BLOCK_NAMES)
        all(isfinite, vec(collect(blocks[k]))) || return (Symbol("block$(k)_" * nm), nothing)
    end

    mv = moments_to_vec(blocks)
    verbose && status("  masked moment vector", mv)
    all(isfinite, mv) || return (:moment_vector, nothing)

    l, _ = full_SMM(θ; u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS)
    verbose && @printf("  %-26s %.6e\n", "LOSS", l[1])
    return (isfinite(l[1]) ? :ok : :loss, l[1])
end

first_bad, lossval = walk(θ_warm; label="1. WARM START")

# ── the box, not just the point ─────────────────────────────────────────────
# A loss that is finite at the warm start but Inf for most particles is a BOUNDS
# problem. Sample the PSO box the same way train_stage builds it.
println("\n── 2. THE SEARCH BOX (50 random particles) ──────────────────────────")
alpha_anchor = TAU_PRIOR !== nothing ? TAU_PRIOR : [P_alpha]
@printf("  alpha box  = [%.4g, %.4g]   (BOUND_LO=%.3g, BOUND_HI=%.3g)\n",
        alpha_anchor[1]*BOUND_LO, alpha_anchor[1]*BOUND_HI, BOUND_LO, BOUND_HI)
lo = vcat(0.8*agg_labor_share, 0.8 .* agg_industry_share, 0.8 .* A_init, alpha_anchor .* BOUND_LO)
hi = vcat(1.2*agg_labor_share, 1.2 .* agg_industry_share, 1.2 .* A_init, alpha_anchor .* BOUND_HI)
Random.seed!(20250101)
tally = Dict{Symbol,Int}()
first_exc = nothing
for k in 1:50
    head = lo .+ rand(length(lo)) .* (hi .- lo)
    θk = vcat(head, vec(permutedims(T_rs_init))[T_MASK])
    st, _ = try
        walk(θk; verbose=false)
    catch e
        # THE point of this loop. In the optimiser this exception is invisible:
        # optimizer.jl passes show_err=false to parallel_SMM_safe, which returns
        # `nothing`, which the objective turns into Inf. So "Best fitness: Inf" in the
        # PSO log means AN EXCEPTION WAS THROWN, not that a moment was non-finite.
        global first_exc
        first_exc === nothing && (first_exc = (e, catch_backtrace()))
        (:exception, nothing)
    end
    tally[st] = get(tally, st, 0) + 1
end
println("  first non-finite stage, over 50 particles:")
for (k, v) in sort(collect(tally), by = x -> -x[2])
    @printf("    %-28s %3d / 50%s\n", string(k), v, k === :ok ? "   ← finite loss" : "")
end
if first_exc !== nothing
    println("\n  FIRST EXCEPTION (this is what the PSO reports as `Inf`):")
    showerror(stdout, first_exc[1], first_exc[2]; backtrace = true)
    println()
end

println("\n" * "="^76)
println("READING THIS")
println("="^76)
println("""
  The first NON-FINITE line above is the source. The usual suspects, in the order
  the chain hits them:

   T / theta          the GE-Sinkhorn image blew up. invert_T_ge iterates
                      T⁺ = exp(log T + δ·log(γ_target/γ_model)); if a T column
                      underflows, log T = −Inf, and if γ_model also underflows the
                      ratio → +Inf, so the sum is NaN. Widen nothing — narrow the α
                      box (BOUND_LO/BOUND_HI in load_parameters.jl) or lower the
                      invert_T_ge damping.

   z_flat / log z     z = T^{1/θ}·(−log(1−u))^{−1/θ}. Under :aa the log-z SIZE
                      CONTROL is on, so a z of exactly 0 gives log z = −Inf. Check
                      min W_RS_FLAT above too: a zero upstream wage makes p = wτ/z
                      vanish and that cell wins every variety.

   c_tilde_r / Y_r    Y_r = p_r^ε·P^{−ε}. With ε = $(round(epsilon, digits=2)) the
                      power is steep: p_r below ~$(round((floatmax(Float64))^(1/epsilon), sigdigits=3))
                      overflows, and Inf·0 = NaN follows immediately.

   block 4 reg_coef   the cloglog IRLS. Watch the SATURATED-group count above: those
                      are sector×AA groups where no cell ever wins, so not_supply ≡ 1.
                      By Lemma 1 they must drop out of the concentrated likelihood.

   block 5 / 2        γ = X_ls/X_s and X_s/X — a sector with no trade at all gives
                      0/0 = NaN.

  If stage 2 shows :ok at the warm start but mostly non-finite over the box, it is
  the BOUNDS, not the model: the α box is [×BOUND_LO, ×BOUND_HI] of the prior, and
  τ = d^α with α at the top of that box spans many orders of magnitude.

  AND NOTE: in the optimiser a thrown exception is INVISIBLE. optimizer.jl calls
  parallel_SMM_safe(x_full, false, second_stage, false; ...) — that 4th positional is
  show_err, and false makes the catch block swallow the error and return `nothing`,
  which the objective maps to Inf. So `Best fitness: Inf` in the PSO log is usually an
  EXCEPTION, not a non-finite moment. Flip that argument to `true` for one run, or
  read the FIRST EXCEPTION block above, which does not swallow anything.
""")
