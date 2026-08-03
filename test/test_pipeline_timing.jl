##### test_pipeline_timing.jl #####
# WHERE does the SMM pipeline spend its time and its memory? STANDALONE + PRINT-ONLY.
#
#     julia test/test_pipeline_timing.jl auto 4 1 true aa true
#     julia -p 8 test/test_pipeline_timing.jl auto 4 1 true aa true
#
# Args: industry n_coef n_tau granular ca_level profile_T [n_rep]
#
# One PSO iteration is `n_particles` evaluations of the objective, and the joint fit is
# 200 of those iterations. This file breaks a single evaluation into its phases and
# reports, for each, the wall time AND the bytes allocated — the two are different
# questions with different answers, and the memory column is what explains a run whose
# RSS climbs across iterations even though nothing is retained between them.
#
# It covers four things:
#
#   1. the phases of one `full_SMM` evaluation, timed and allocation-counted;
#   2. `invert_T_ge` — under `profile_T` this runs BEFORE every evaluation, so its cost
#      is additive to all of the above and is easy to miss;
#   3. how the phase costs SCALE in N_rho, which is the one knob that moves them;
#   4. the `pmap` round trip: how much of an iteration is model evaluation and how much
#      is the distributed plumbing, including the cost of what the objective closure
#      carries to the workers.
#
# Print-only. Nothing here is on the estimation path.

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
n_rep     = length(ARGS) >= 7 ? parse(Int, ARGS[7]) : 5
K_sim     = 10000
reg_method = :cloglog

input_folder  = "./baseline_$industry"
output_folder = "./reporting_$industry" * (ca_level == :aa ? "_aa" : "") *
                (granular ? "_gran" : "") * "_timing"
mkpath(output_folder)

include("../load_parameters.jl")
@everywhere include("../profiling.jl")

# ── measurement helpers ────────────────────────────────────────────────────
mb(bytes) = bytes / 1024^2

"""
Run `f` once to force compilation, then `n` more times; return the MINIMUM elapsed
time (the least noisy estimate of the cost — the maximum is whatever the GC happened
to do on that pass) alongside the mean, and the allocation count of one clean call.
"""
function bench(f, n::Int)
    f()                                    # warm up / compile
    ts = Float64[]
    for _ in 1:n
        GC.gc()                            # start each timed pass from a clean heap
        push!(ts, @elapsed f())
    end
    GC.gc()
    alloc = @allocated f()
    return (tmin = minimum(ts), tmean = mean(ts), alloc = alloc)
end

const ROWS = Tuple{String,Float64,Float64,Float64}[]   # name, tmin, alloc_MB, share
record!(name, r) = push!(ROWS, (name, r.tmin, mb(r.alloc), NaN))

function table(rows, total_t, total_a; title)
    println("\n  $title")
    println("    phase                              time (ms)    share      alloc (MB)   share")
    for (name, t, a, _) in rows
        @printf("    %-32s %9.2f  %6.1f%%   %11.1f  %6.1f%%\n",
                name, 1e3 * t, total_t > 0 ? 100 * t / total_t : NaN,
                a, total_a > 0 ? 100 * a / total_a : NaN)
    end
    @printf("    %-32s %9.2f  %6s    %11.1f  %6s\n", "── measured total", 1e3 * total_t, "",
            total_a, "")
end

println("\n" * "="^84)
println("SMM PIPELINE — TIME AND MEMORY BREAKDOWN")
@printf("  industry=%s  CA_LEVEL=:%s  GRANULAR=%s  REG_METHOD=:%s  profile_T=%s\n",
        industry, CA_LEVEL, GRANULAR, REG_METHOD, profile_T)
@printf("  n_good=%d  S=%d  R=%d  R_downstream=%d  N_rho=%d  moments=%d  T params=%d\n",
        n_good, S, R, R_downstream, N_rho, length(empirical_moments), count(T_MASK))
@printf("  reps per measurement = %d   workers = %d   REG_STREAMING = %s\n",
        n_rep, nworkers(), REG_STREAMING[])
println("="^84)

# The point of evaluation: the saved θ̂ if there is one, else the warm start.
A_init = copy(emp_pi_r_full).^(1/abs(epsilon)) .* regional_wages[N_downstream_per_region .!= 0]
A_init ./= sum(A_init)
alpha0 = TAU_PRIOR !== nothing ? TAU_PRIOR : fill(P_alpha, N_TAU)
θ = vcat([agg_labor_share], agg_industry_share, A_init, alpha0,
         vec(permutedims(T_rs_init))[T_MASK])
for cand in ("step3/theta_hat_2.npy", "step1/theta_hat_1.npy")
    p = joinpath(output_folder, cand)
    if isfile(p)
        # `global`: a top-level `for` opens a soft scope, so a bare assignment would
        # create a local and silently leave θ at the warm start.
        v = NPZ.npzread(p); global θ = ndims(v) > 1 ? v[:, 1] : v
        println("\n  evaluating at $cand"); break
    end
end
profile_T && (θ = profiled_theta(θ))

# ═══════════════════════════════════════════════════════════════════════════
# 1. ONE full_SMM EVALUATION, PHASE BY PHASE
#
#    The phases are the real functions the pipeline calls, in the order it calls
#    them, so the sum is the evaluation and nothing is a re-implementation. The only
#    thing the sum will not quite reach is `full_SMM` itself, because the network and
#    the moments are held live across the phase boundary here whereas in the real
#    call the allocator can reuse; the gap is reported.
# ═══════════════════════════════════════════════════════════════════════════
println("\n── 1. ONE EVALUATION ──────────────────────────────────────────────────")

empty!(ROWS)

r_unpack = bench(() -> unpack_params(θ), n_rep);                     record!("unpack_params", r_unpack)
alpha_θ  = unpack_params(θ)[4]
r_tau    = bench(() -> build_tau(alpha_θ), n_rep);                   record!("build_tau", r_tau)

# `need_z` is the predicate SMM uses. Its second clause (`REG_INCLUDE_SIZE && u_draws
# === nothing`) is the legacy random-Fréchet branch, and every call below passes draws,
# so only the streaming switch matters here.
need_z = !REG_STREAMING[]
t_profile = 0.0   # filled in section 2 when profile_T
r_net = bench(() -> solve_network(θ; u_draws = U_DRAWS, sample_weights = SAMPLE_WEIGHTS,
                                  return_z = need_z), n_rep)
record!("solve_network (Ricardian+CES)", r_net)

network = solve_network(θ; u_draws = U_DRAWS, sample_weights = SAMPLE_WEIGHTS, return_z = need_z)

# The extensive-margin regression, isolated from the rest of compute_moments. It is
# the one phase that is an ITERATIVE fit (IRLS) rather than a closed-form pass, so it
# is the usual answer to "what is the longest part".
reg_fun = REG_METHOD == :cloglog ? fast_cloglog_regression : fast_weighted_regression
r_reg = bench(() -> reg_fun(network.linkages_flat, network.z_flat, network.sample_weights;
                            include_control      = REG_INCLUDE_CONTROL,
                            include_size_control = REG_INCLUDE_SIZE,
                            return_size_coef     = REG_INCLUDE_SIZE,
                            u_draws              = network.u_draws,
                            logz_const           = network.logz_const,
                            inv_theta            = 1.0 / theta), n_rep)
record!("  └ block 4: $(REG_METHOD) regression", r_reg)

r_mom = bench(() -> compute_moments(network, θ), n_rep)
record!("compute_moments (all blocks)", r_mom)

moms = moment_blocks_tuple(compute_moments(network, θ))
r_vec  = bench(() -> moments_to_vec(moms), n_rep);                   record!("moments_to_vec", r_vec)
r_loss = bench(() -> loss_function(moms, empirical_moments, Weight_matrix_custom), n_rep)
record!("loss_function", r_loss)

r_full = bench(() -> full_SMM(θ; u_draws = U_DRAWS, sample_weights = SAMPLE_WEIGHTS), n_rep)

# Everything except the nested regression row, which is a component of compute_moments.
top = [r for r in ROWS if !startswith(r[1], "  └")]
tot_t = sum(r[2] for r in top); tot_a = sum(r[3] for r in top)
table(ROWS, tot_t, tot_a; title = "phases of one evaluation")
@printf("\n    full_SMM end to end:            %9.2f ms   %11.1f MB\n",
        1e3 * r_full.tmin, mb(r_full.alloc))

# Allocation sanity: solve_network's live working set is a handful of full-size arrays.
# Allocating a large multiple of that means the hot loop is churning — and by far the
# commonest cause is type instability, because a dynamically-dispatched arithmetic
# result is a BOXED Float64 allocated on the heap. The innermost Ricardian comparison
# runs N_rho × Σ_s n_active_s × R_downstream times, so a few boxed values there dominate
# everything else in the evaluation.
working_set = mb(8 * N_rho * n_good      +   # z_inv_flat
                 N_rho * n_good / 8      +   # linkages_flat (BitMatrix)
                 8 * 4 * N_rho * S       +   # the four hoisted per-region buffers
                 8 * n_good * R_downstream)  # X_shares_by_r
ric_iters = N_rho * n_good * R_downstream
@printf("\n    solve_network working set:      %11.1f MB   (allocated %.1f× that)\n",
        working_set, mb(r_net.alloc) / working_set)
@printf("    Ricardian comparisons:          %11.2f M   → %.0f bytes, %.1f ns each\n",
        ric_iters / 1e6, r_net.alloc / ric_iters, 1e9 * r_net.tmin / ric_iters)
if r_net.alloc / (working_set * 1024^2) > 3
    println("    ⚠ allocating >3× the working set: the hot loop is boxing. Check that")
    println("      `tau` and `z_inv_flat` are still bound to concrete Matrix{Float64}")
    println("      locals before the r_d loop (model_CP.jl) — `build_tau` returns")
    println("      `ones(eltype(alpha), …)` with `alpha` from an untyped `params`, so")
    println("      without the re-binding both infer as `Any`.")
end
@printf("    (phase sum vs full_SMM: %+.1f%% time, %+.1f%% alloc — the gap is measurement\n",
        100 * (tot_t - r_full.tmin) / r_full.tmin, 100 * (tot_a - mb(r_full.alloc)) / mb(r_full.alloc))
println("     overhead from holding `network` live across the phase boundary)")

# ═══════════════════════════════════════════════════════════════════════════
# 2. THE PROFILING STEP — free-standing, and paid on EVERY particle
# ═══════════════════════════════════════════════════════════════════════════
if profile_T
    println("\n── 2. T PROFILING (paid before every evaluation under profile_T) ──────")
    ΩL = θ[1]
    Ωs = θ[2:(1 + S)];                      Ωs = Ωs ./ sum(Ωs)
    Av = θ[(S + 2):(S + R_downstream + 1)]; Av = Av ./ Av[1]
    αv = θ[(S + R_downstream + 2):(1 + S + R_downstream + N_TAU)]

    inv0 = invert_T_ge(αv, ΩL, Ωs, Av)
    r_inv = bench(() -> invert_T_ge(αv, ΩL, Ωs, Av), n_rep)
    global t_profile = r_inv.tmin
    r_one = bench(() -> gamma_ls_analytical(ΩL, Ωs, Av, gather_T_to_ze(inv0.T), build_tau(αv)), n_rep)
    @printf("    invert_T_ge              %9.2f ms   %11.1f MB   (%d iterations, resid %.2e, %s)\n",
            1e3 * r_inv.tmin, mb(r_inv.alloc), inv0.iters, inv0.resid,
            inv0.converged ? "converged" : "NOT CONVERGED")
    @printf("    └ one Sinkhorn pass      %9.2f ms   %11.1f MB\n",
            1e3 * r_one.tmin, mb(r_one.alloc))
    @printf("\n    profiling / evaluation   %8.1f%% of one full_SMM\n",
            100 * r_inv.tmin / r_full.tmin)
    println("""
    Cost here is LINEAR in the iteration count, and the iteration count is exactly what
    the convergence diagnostic reports. A particle that runs to the 500-iteration cap
    pays ~$(round(Int, 500 / max(inv0.iters, 1)))× what this one did — so a swarm with a high non-convergence rate is
    slow for the same reason it is inaccurate. See test/test_T_convergence_map.jl.""")
end

# ═══════════════════════════════════════════════════════════════════════════
# 3. SCALING IN N_rho — the knob that moves all of the above
# ═══════════════════════════════════════════════════════════════════════════
println("\n── 3. SCALING IN N_rho ────────────────────────────────────────────────")
println("    N_rho    solve_network        block-4 regression      full evaluation")
println("             ms        MB        ms        MB           ms        MB")
for mult in (0.5, 1.0, 2.0)
    n = max(round(Int, mult * N_rho), 8)
    u, w = generate_draws(n, n_good, DRAW_METHOD)
    rn = bench(() -> solve_network(θ; u_draws = u, sample_weights = w, return_z = need_z), max(n_rep ÷ 2, 2))
    nk = solve_network(θ; u_draws = u, sample_weights = w, return_z = need_z)
    rr = bench(() -> reg_fun(nk.linkages_flat, nk.z_flat, nk.sample_weights;
                             include_control      = REG_INCLUDE_CONTROL,
                             include_size_control = REG_INCLUDE_SIZE,
                             return_size_coef     = REG_INCLUDE_SIZE,
                             u_draws              = nk.u_draws, logz_const = nk.logz_const,
                             inv_theta = 1.0 / theta), max(n_rep ÷ 2, 2))
    rf = bench(() -> full_SMM(θ; u_draws = u, sample_weights = w), max(n_rep ÷ 2, 2))
    @printf("    %6d  %8.1f  %8.1f  %8.1f  %8.1f     %8.1f  %8.1f\n",
            n, 1e3 * rn.tmin, mb(rn.alloc), 1e3 * rr.tmin, mb(rr.alloc),
            1e3 * rf.tmin, mb(rf.alloc))
    u = nothing; w = nothing; nk = nothing; GC.gc()
end
println("""
    Everything above is linear in N_rho except the block-4 fit, whose IRLS iteration
    count can move with the design. A row that grows FASTER than linearly is the one
    to look at before raising N_rho for the granular q̂ precision.""")

# ═══════════════════════════════════════════════════════════════════════════
# 4. THE pmap ROUND TRIP — model cost vs distributed plumbing
#
#    This is the part a single-process profile cannot see. `pmap` serializes the
#    objective CLOSURE once per TASK (batch_size = 1 ⇒ once per particle, every
#    iteration), so anything the closure captures is re-sent and re-allocated
#    n_particles × max_iter times per stage. U_DRAWS is the big one: it is already
#    `@everywhere const` on every worker, so capturing it is pure waste.
# ═══════════════════════════════════════════════════════════════════════════
println("\n── 4. pmap ROUND TRIP ─────────────────────────────────────────────────")
if nworkers() < 2
    println("    single process (run with `julia -p N`) — nothing to measure.")
else
    n_particles = 2 * nworkers()
    swarm = [θ .* (1 .+ 0.001 .* randn(length(θ))) for _ in 1:n_particles]

    # (a) the closure the objective USED to build: U_DRAWS captured by value.
    let ud = U_DRAWS, sw = SAMPLE_WEIGHTS
        f_capt = x -> (full_SMM(x; u_draws = ud, sample_weights = sw)[1][1])
        t_capt = (pmap(f_capt, swarm); @elapsed pmap(f_capt, swarm))
        # (b) the closure it builds now: the globals resolve on the worker.
        f_glob = x -> (full_SMM(x; u_draws = U_DRAWS, sample_weights = SAMPLE_WEIGHTS)[1][1])
        t_glob = (pmap(f_glob, swarm); @elapsed pmap(f_glob, swarm))
        # (c) the plumbing alone: same task count, no model.
        f_null = x -> sum(x)
        t_null = (pmap(f_null, swarm); @elapsed pmap(f_null, swarm))

        ideal = r_full.tmin * n_particles / nworkers()
        payload = mb(sizeof(U_DRAWS))
        @printf("    particles = %d over %d workers; one evaluation = %.1f ms\n",
                n_particles, nworkers(), 1e3 * r_full.tmin)
        @printf("    perfect-parallel floor              %8.1f ms\n", 1e3 * ideal)
        @printf("    pmap, globals resolved on worker    %8.1f ms   (%.2f× the floor)\n",
                1e3 * t_glob, t_glob / ideal)
        @printf("    pmap, U_DRAWS captured in closure   %8.1f ms   (%.2f× the floor)\n",
                1e3 * t_capt, t_capt / ideal)
        @printf("    pmap, empty task body               %8.1f ms\n", 1e3 * t_null)
        @printf("\n    U_DRAWS is %.1f MB. Captured, it is serialized once per task:\n", payload)
        @printf("      %.0f MB per iteration of %d particles, %.1f GB over a 200-iteration stage,\n",
                payload * n_particles, n_particles, payload * n_particles * 200 / 1024)
        println("      all of it allocated on a worker and discarded immediately. That is the")
        println("      churn that makes worker RSS climb across a stage even though the swarm")
        println("      retains nothing. optimizer.jl now keeps it out of the closure.")
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# 5. WHAT ONE STAGE COSTS
# ═══════════════════════════════════════════════════════════════════════════
println("\n── 5. EXTRAPOLATION TO ONE STAGE ──────────────────────────────────────")
# Under profiling every particle pays the inversion on top of the evaluation.
per_eval = r_full.tmin + t_profile
@printf("    cost per particle = %.1f ms evaluation%s\n", 1e3 * r_full.tmin,
        t_profile > 0 ? @sprintf(" + %.1f ms T inversion", 1e3 * t_profile) : "")
for (np, mi, tag) in ((98, 200, "Stage 1 joint fit"), (98, 50, "one refinement sub-stage"))
    n_ev = np * mi
    wall = n_ev * per_eval / max(nworkers(), 1)
    @printf("    %-26s %6d evaluations  ≈ %6.1f min wall on %d workers\n",
            tag, n_ev, wall / 60, max(nworkers(), 1))
    @printf("    %-26s %6s              ≈ %6.1f GB allocated per worker\n",
            "", "", n_ev * mb(r_full.alloc) / 1024 / max(nworkers(), 1))
end
println("""
    The allocation figure is TURNOVER, not residency: none of it is retained between
    evaluations. It matters anyway, because each worker's GC only collects when its own
    heap approaches `--heap-size-hint` (main.jl, default 2G). With 49 workers that is a
    98 GB collective ceiling reached by pure churn — which is what a memory trace that
    climbs monotonically across a 200-iteration stage looks like. The fixes are to cut
    the turnover (sections 1 and 4) or to lower the hint so each worker collects sooner:
    `WORKER_HEAP_HINT=1G ./run.sh ...`.""")

println("\n" * "="^84)
println("READING THIS")
println("="^84)
println("""
  Section 1 answers "which part is longest". Expect the two heavy rows to be
  `solve_network` — R_downstream × S × N_rho Ricardian comparisons, the only genuinely
  O(N_rho · n_good) pass — and the block-4 regression, which is an IRLS fit and so pays
  its cost per iteration. Everything else (the aggregate blocks, γ, the loss) is a
  handful of reductions over already-computed arrays and should be noise.

  Section 1's ALLOCATION column is the memory question. A phase that allocates many
  times its own working set is churning: it is building temporaries and dropping them.

  Section 4 is the part that only shows up in the distributed run. Compare the two pmap
  rows: the difference between them is pure serialization of data every worker already
  had.

  Section 5 turns per-evaluation numbers into the stage budget, and states the
  distinction that matters for reading a memory trace — allocation turnover versus
  retained memory. Nothing in the PSO loop retains anything of consequence across
  iterations; a climbing trace is deferred collection, not a leak.
""")
