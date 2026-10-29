##### optimizer.jl — staged PSO over the head [Ω^L | Ω^s | A | α | κ] #####
#
# T is never searched: every objective evaluation replaces it by T*(α, κ, Ω, A)
# (profiling.jl). A particle whose inversion does not converge is REJECTED (loss Inf),
# never scored at a T the model does not imply.
#
# Stage variables (a "stage" optimises one block, warm-started at the incumbent):
#   "productivity"                                  A
#   "alpha"                                         [α ; κ]
#   "agg_labor_share_tech", "agg_industry_share_tech"  Ω^L, Ω^s
#
# Under PROFILE_HEAD the head (Ω^L, Ω^s, A) is profiled jointly with T, so every search
# — the fresh start included — runs over [α ; κ] only.

using Distributed
using Random
using Statistics
using NPZ
using Printf

##################### PSO ###################

"""
    parallel_pso_smm(objective, lb, ub; n_particles, max_iter, warm_start_particle,
                     t_conv_probe, verbose) -> (best_x, best_f, history)

Particle swarm with a scrambled-Sobol initial swarm, reflecting walls, linearly
decreasing inertia and periodic restarts of stagnant particles. The warm start is
always one of the particles, so a stage never ends worse than its incumbent.
"""
function parallel_pso_smm(objective::Function, lb::Vector{Float64}, ub::Vector{Float64};
                          n_particles::Int = 70, max_iter::Int = 100,
                          warm_start_particle::Union{Vector{Float64}, Nothing} = nothing,
                          w_start::Float64 = 0.9, w_end::Float64 = 0.4,
                          c1::Float64 = 1.5, c2::Float64 = 2.5,
                          bounce_damping::Float64 = 0.5,
                          t_conv_probe::Union{Function, Nothing} = nothing,
                          verbose::Bool = true)
    d = length(lb)

    n_random = warm_start_particle === nothing ? n_particles : n_particles - 1
    particles = if n_random == 0
        Vector{Vector{Float64}}()
    else
        net = sobol_scrambled_net(n_random, d, Random.GLOBAL_RNG)
        [lb .+ net[i, :] .* (ub .- lb) for i in 1:n_random]
    end
    warm_start_particle === nothing ||
        push!(particles, clamp.(warm_start_particle, lb, ub))
    velocities = [0.1 * (ub .- lb) .* randn(d) for _ in 1:length(particles)]

    fitness = Vector{Float64}(pmap(objective, particles))
    p_best, p_best_fitness = copy(particles), copy(fitness)
    g_best_idx = argmin(p_best_fitness)
    g_best, g_best_fitness = copy(p_best[g_best_idx]), p_best_fitness[g_best_idx]
    history = Dict("best_fitness" => Float64[g_best_fitness],
                   "mean_fitness" => Float64[mean(filter(isfinite, fitness))])
    verbose && println("[PSO] $(length(particles)) particles, dimension $d, " *
                       "initial best $(round(g_best_fitness, digits=6))")
    last_printed = g_best_fitness

    for iter in 1:max_iter
        t_start = time()
        w = w_start - (w_start - w_end) * (iter / max_iter)
        v_max = 0.1 * (ub .- lb)
        for i in 1:length(particles)
            r1, r2 = rand(d), rand(d)
            velocities[i] = clamp.(w * velocities[i] .+ c1 * r1 .* (p_best[i] .- particles[i]) .+
                                   c2 * r2 .* (g_best .- particles[i]), -v_max, v_max)
            particles[i] = particles[i] .+ velocities[i]
            reflect_bounds!(particles[i], velocities[i], lb, ub, bounce_damping)
        end

        fitness = Vector{Float64}(pmap(objective, particles))
        for i in 1:length(particles)
            if fitness[i] < p_best_fitness[i]
                p_best[i] = copy(particles[i]); p_best_fitness[i] = fitness[i]
            end
        end
        k = argmin(p_best_fitness)
        if p_best_fitness[k] < g_best_fitness
            g_best, g_best_fitness = copy(p_best[k]), p_best_fitness[k]
        end

        # Restart particles far from the best.
        if iter > 30 && iter % 25 == 0
            for i in 1:length(particles)
                fitness[i] > g_best_fitness * 1.5 || continue
                particles[i] = lb .+ rand(d) .* (ub .- lb)
                velocities[i] = 0.1 * (ub .- lb) .* randn(d)
                p_best[i] = copy(particles[i]); p_best_fitness[i] = Inf
            end
        end

        push!(history["best_fitness"], g_best_fitness)
        push!(history["mean_fitness"], mean(filter(isfinite, fitness)))
        if verbose && (iter % 5 == 0 || iter == 1 || iter == max_iter)
            @printf("[PSO] iter %d/%d (%.2fs): best %.6g  mean %.6g  (%.2f%% since last report)\n",
                    iter, max_iter, time() - t_start, g_best_fitness,
                    mean(filter(isfinite, fitness)), 100 * (last_printed - g_best_fitness) / last_printed)
            t_conv_probe === nothing || report_t_convergence(pmap(t_conv_probe, particles))
            last_printed = g_best_fitness
            flush(stdout)
        end
    end
    return g_best, g_best_fitness, history
end

"""
    reflect_bounds!(pos, vel, lb, ub, damping)

Mirror any coordinate past a wall back inside and reverse its velocity, shrunk by
`damping` (particles bounce instead of sticking to the bounds).
"""
function reflect_bounds!(pos::Vector{Float64}, vel::Vector{Float64},
                         lb::Vector{Float64}, ub::Vector{Float64}, damping::Float64)
    @inbounds for k in eachindex(pos)
        if pos[k] < lb[k]
            pos[k] = lb[k] + (lb[k] - pos[k]); vel[k] = -damping * vel[k]
        elseif pos[k] > ub[k]
            pos[k] = ub[k] - (pos[k] - ub[k]); vel[k] = -damping * vel[k]
        end
        pos[k] = clamp(pos[k], lb[k], ub[k])
    end
    return pos, vel
end

"""
    report_t_convergence(probes; n_bins=5)

Share of the swarm whose T inversion did not converge, overall and by α bin (a rate
rising in α says the α box reaches past the solver's stability ceiling).
"""
function report_t_convergence(probes; n_bins::Int = 5)
    n_tot = length(probes)
    bad   = [!p.converged for p in probes]
    n_bad = count(bad)
    @printf("  T non-converged: %d/%d particles (%.1f%%)\n", n_bad, n_tot, 100 * n_bad / max(n_tot, 1))
    n_bad == 0 && return
    alphas = [isempty(p.alpha) ? NaN : maximum(p.alpha) for p in probes]
    ok = findall(isfinite, alphas)
    length(ok) < 2 && return
    lo, hi = extrema(alphas[ok])
    hi > lo || return
    w = (hi - lo) / n_bins
    for b in 1:n_bins
        blo = lo + (b - 1) * w; bhi = b == n_bins ? hi : lo + b * w
        idx = [i for i in ok if alphas[i] >= blo && (alphas[i] < bhi || b == n_bins)]
        isempty(idx) || @printf("    α ∈ [%.4g, %.4g)  %3d/%3d\n", blo, bhi, count(i -> bad[i], idx), length(idx))
    end
end

##################### Stages ###################

"""
    param_start(var) -> first index of a stage variable in the level vector
"""
function param_start(var::Symbol)
    var == :agg_labor_share_tech    && return 1
    var == :agg_industry_share_tech && return 2
    var == :productivity            && return 2 + S
    var == :alpha                   && return 2 + S + R_downstream
    error("Unknown stage variable :$var")
end

"""
    alpha_box(anchor) -> (lo, hi)

α box `[BOUND_LO, BOUND_HI] × anchor`, capped at `ALPHA_MAX` in levels; the floor stays
below half the ceiling so the box never collapses.
"""
function alpha_box(anchor)
    hi = min.(anchor .* BOUND_HI, ALPHA_MAX)
    return (min.(anchor .* BOUND_LO, 0.5 .* hi), hi)
end

# Binned α must be increasing (α_1 ≤ … ≤ α_K). The objective evaluates the α block
# SORTED (`build_x_full`), so it is symmetric in the order of the α coordinates and the
# returned θ is increasing. A sorted point stays in the box when the bounds are sorted
# too: with lo and hi each increasing, the k-th order statistic of any x ∈ [lo, hi] lies
# in [lo_k, hi_k]. Sorting the bounds keeps lo ≤ hi entrywise and is a no-op for an
# increasing anchor, or for N_TAU == 1.
sorted_box(lo, hi) = (sort(lo), sort(hi))

"""
    train_stage(n_particles, max_iter; init_alpha, variable_list, last_stage_folder,
                radius, weight_matrix, warm_start_override, moment_blocks)
        -> (best_params, best_fitness, history)

Fresh start (`last_stage_folder === nothing`): the whole head, boxed at `HEAD_*_BOX` × the
data-based starting values (α within `alpha_box` of its prior, κ in [KAPPA_LO, KAPPA_HI]).
Continue stage: the listed variables only, in a box of relative radius `radius` around
the incumbent (clamped to the same outer boxes); everything else fixed at the incumbent.
The returned vector carries T*(·) at the optimum.
"""
function train_stage(n_particles::Int, max_iter::Int;
                     init_alpha = nothing,
                     variable_list = nothing,
                     last_stage_folder = nothing,
                     radius = 0.1,
                     weight_matrix::Union{Nothing, AbstractMatrix} = nothing,
                     warm_start_override::Union{Nothing, Vector{Float64}} = nothing,
                     moment_blocks::Union{Nothing, Vector{Int}} = nothing)
    nhead = n_head_params()
    T_PLACEHOLDER = vec(permutedims(T_rs_init))[T_MASK]   # overwritten by profiling
    cached_tau = nothing

    if last_stage_folder === nothing
        A = copy(emp_pi_r_full).^(1/abs(epsilon)) .* regional_wages[N_downstream_per_region .!= 0]
        A ./= sum(A)
        a_lo, a_hi = sorted_box(alpha_box(TAU_PRIOR !== nothing ? TAU_PRIOR : init_alpha)...)
        lb = vcat(max(HEAD_OMEGA_L_BOX[1] * agg_labor_share, OMEGA_L_MIN),
                  HEAD_OMEGA_S_BOX[1] .* agg_industry_share, HEAD_A_BOX[1] .* A, a_lo, KAPPA_LO)
        ub = vcat(min(HEAD_OMEGA_L_BOX[2] * agg_labor_share, OMEGA_L_MAX),
                  HEAD_OMEGA_S_BOX[2] .* agg_industry_share, HEAD_A_BOX[2] .* A, a_hi, KAPPA_HI)
        best_params_prev = nothing
        var_list = String[]
        warm_start = warm_start_override === nothing ? nothing : warm_start_override[1:nhead]
        # PROFILE_HEAD: the head prefix is a placeholder (profiling overwrites it), so
        # only [α ; κ] are searched.
        n_fixed = PROFILE_HEAD ? first(alpha_range()) - 1 : 0
        head_prefix = PROFILE_HEAD ? (warm_start === nothing ? lb[1:n_fixed] : warm_start[1:n_fixed]) :
                                     Float64[]
        lb = lb[(n_fixed + 1):end]; ub = ub[(n_fixed + 1):end]
        warm_start === nothing || (warm_start = warm_start[(n_fixed + 1):end])
    else
        best_params_prev = NPZ.npzread(joinpath(last_stage_folder, "best_params.npy"))[:, 1]
        Ω_L, Ω_s, A_r, α, _ = unpack_params(best_params_prev)
        params_dict = Dict(:agg_labor_share_tech => Ω_L, :agg_industry_share_tech => Ω_s,
                           :productivity => A_r, :alpha => vcat(α, unpack_kappa(best_params_prev)))
        var_list = isa(variable_list, String) ? [variable_list] : collect(variable_list)
        ("alpha" in var_list) || (cached_tau = build_tau(α))

        lb_parts = Vector{Float64}[]; ub_parts = Vector{Float64}[]
        offset = 0
        for v in var_list
            val = params_dict[Symbol(v)]
            if v == "agg_labor_share_tech"
                lb_v = max.(val .* (1 - radius), 0.001); ub_v = min.(val .* (1 + radius), 1.0)
            elseif v == "alpha"
                κv, αv = val[end], val[1:N_TAU]
                half = (1 - radius) * (KAPPA_HI - KAPPA_LO) / 2
                κlo, κhi = max(κv - half, KAPPA_LO), min(κv + half, KAPPA_HI)
                κlo >= κhi && ((κlo, κhi) = (KAPPA_LO, KAPPA_HI))
                box_lo, box_hi = alpha_box(TAU_PRIOR !== nothing ? TAU_PRIOR : αv)
                lb_v = max.(αv .* (1 - radius), box_lo); ub_v = min.(αv .* (1 + radius), box_hi)
                bad = lb_v .>= ub_v           # incumbent above the cap: reset to the full box
                lb_v = ifelse.(bad, box_lo, lb_v); ub_v = ifelse.(bad, box_hi, ub_v)
                lb_v, ub_v = sorted_box(lb_v, ub_v)
                lb_v = vcat(lb_v, κlo); ub_v = vcat(ub_v, κhi)
            else
                lb_v = val .* radius; ub_v = val ./ radius
            end
            push!(lb_parts, lb_v isa Number ? [lb_v] : lb_v)
            push!(ub_parts, ub_v isa Number ? [ub_v] : ub_v)
            offset += length(val)
        end
        lb = vcat(lb_parts...); ub = vcat(ub_parts...)
        warm_start = vcat([params_dict[Symbol(v)] for v in var_list]...)
        @assert length(warm_start) == length(lb)
    end

    # Stage vector → full level vector (T block stale; profiling replaces it), with the
    # α block sorted (see `sorted_box`).
    function build_x_full(x_stage)
        if best_params_prev === nothing
            x_full = vcat(head_prefix, x_stage, T_PLACEHOLDER)
        else
            x_full = copy(best_params_prev)
            pos = 1
            for v in var_list
                len = length(params_dict[Symbol(v)])
                st = param_start(Symbol(v))
                x_full[st:(st + len - 1)] = x_stage[pos:(pos + len - 1)]
                pos += len
            end
        end
        sort!(view(x_full, alpha_range()))
        return x_full
    end

    function objective(x_stage)
        prof = profiled_theta_full(build_x_full(x_stage))
        prof.converged || return Inf
        result = safe_SMM(prof.theta; precomputed_tau=cached_tau,
                          W_override=weight_matrix, moment_blocks=moment_blocks)
        return (result === nothing || !isfinite(result[1][1])) ? Inf : result[1][1]
    end

    function probe(x_stage)
        try
            p = profiled_theta_full(build_x_full(x_stage))
            return (converged = p.converged, alpha = p.alpha)
        catch
            return (converged = false, alpha = Float64[])
        end
    end

    println("\n" * "="^60)
    println("PSO stage: ", isempty(var_list) ? (PROFILE_HEAD ? "alpha (head profiled)" : "all head parameters") :
            join(var_list, ", "),
            " | particles $n_particles | iterations $max_iter | dimension $(length(lb))")
    println("="^60)
    best_x, best_fitness, history = parallel_pso_smm(objective, lb, ub;
        n_particles = n_particles, max_iter = max_iter, warm_start_particle = warm_start,
        t_conv_probe = probe)

    prof = profiled_theta_full(build_x_full(best_x))
    prof.converged || @warn "invert_T_ge did NOT converge at the stage optimum (no particle " *
        "converged): the returned T is the last iterate." resid=prof.resid iters=prof.iters
    return prof.theta, best_fitness, history
end

"""
    run_optimization(; weight_matrix, warm_start_params, output_subfolder, max_loop,
                     n_particles, max_iter_initial, max_iter_stage, gamma_beta_only,
                     moments_loss_gamma_beta) -> (best_params, best_fitness)

Step 1 (`gamma_beta_only = false`): a joint fit of the whole head, then `max_loop`
block-coordinate loops (A, then [α ; κ], then Ω^L/Ω^s), with a radius annealed from
0.3 to 0.9. Under PROFILE_HEAD the fit and every loop are over [α ; κ] only, and the
loss drops blocks 1–3 (`[4, 5, 6]`): at the profile the SIMULATED blocks 1–3 differ from
their targets only by the Monte-Carlo error of the realised price indices. Step 3 (`gamma_beta_only = true`): only [α ; κ], the rest fixed at θ̂_1,
on the inference moments (β → γ → G) with `weight_matrix = W_step3`.

Every stage writes `<step>/<stage>/best_params.npy` (stage 0 = the initial fit,
`epoch_k/<stage>` in the loops) and its report; the loop stops when the parameters
move by less than 1e-6 between consecutive loops.

`n_particles` should be a multiple of `nworkers()` (pmap evaluates one particle per worker).
"""
function run_optimization(; weight_matrix::Union{Nothing, AbstractMatrix} = nothing,
                          warm_start_params::Vector{Float64},
                          output_subfolder::String = "step1",
                          max_loop::Int = 20,
                          n_particles::Int = 98,
                          max_iter_initial::Int = 200,
                          max_iter_stage::Int = 50,
                          gamma_beta_only::Bool = false,
                          moments_loss_gamma_beta::Bool = false,
                          loss_blocks::Union{Nothing, Vector{Int}} = nothing)
    loop_base = joinpath(output_folder, output_subfolder)
    mkpath(loop_base)
    # `loss_blocks` overrides: Step 3 uses [4] alone with the fake-world W (inference.jl),
    # γ and G being reproduced exactly by the T inversion and the N bisection.
    moment_blocks = loss_blocks !== nothing ? loss_blocks :
                    (moments_loss_gamma_beta || PROFILE_HEAD) ? [4, 5, 6] : nothing
    save_stage(folder, params) = (mkpath(folder);
        NPZ.npzwrite(joinpath(folder, "best_params.npy"), reshape(params, :, 1)))

    if gamma_beta_only
        seed_folder = joinpath(loop_base, "seed")
        save_stage(seed_folder, warm_start_params)
        best_params, best_fitness, _ = train_stage(n_particles, max_iter_initial;
            variable_list = ["alpha"], last_stage_folder = seed_folder, radius = 0.5,
            weight_matrix = weight_matrix, moment_blocks = moment_blocks)
    else
        best_params, best_fitness, _ = train_stage(n_particles, max_iter_initial;
            init_alpha = warm_start_params[alpha_range()], radius = 0.5,
            weight_matrix = weight_matrix, warm_start_override = warm_start_params,
            moment_blocks = moment_blocks)
    end
    stage = 0
    save_stage(joinpath(loop_base, "0"), best_params)
    generate_report(joinpath(loop_base, "0"), best_params; variable = "initial fit")

    stage_vars = (gamma_beta_only || PROFILE_HEAD) ? (["alpha"],) :
        (["productivity"], ["alpha"], ["agg_labor_share_tech", "agg_industry_share_tech"])
    radius_start, radius_end = 0.3, 0.9
    for loop in 1:max_loop
        radius = radius_start + (loop - 1) * (radius_end - radius_start) / max(max_loop - 1, 1)
        prev_folder = joinpath(loop == 1 ? loop_base : joinpath(loop_base, "epoch_$(loop-1)"), string(stage))
        loop_folder = joinpath(loop_base, "epoch_$loop")
        println("\n[$output_subfolder] LOOP $loop/$max_loop  radius=$radius")
        for vars in stage_vars
            r = vars == ["productivity"] ? 0.7 + 0.2 * radius : radius
            best_params, best_fitness, _ = train_stage(n_particles, max_iter_stage;
                variable_list = vars, last_stage_folder = prev_folder, radius = r,
                weight_matrix = weight_matrix, moment_blocks = moment_blocks)
            stage += 1
            prev_folder = joinpath(loop_folder, string(stage))
            save_stage(prev_folder, best_params)
            generate_report(prev_folder, best_params; variable = join(vars, ", "), radius = r)
        end
        println("  loop $loop done, fitness $(round(best_fitness, digits=6))")
        if loop > 2
            prev = NPZ.npzread(joinpath(loop_base, "epoch_$(loop-1)",
                                        string(stage - length(stage_vars)), "best_params.npy"))[:, 1]
            if maximum(abs.(best_params .- prev) ./ (abs.(prev) .+ 1e-10)) < 1e-6
                println("  converged: Δparams < 1e-6.")
                break
            end
        end
    end
    run_reporting(loop_base, max_loop)
    return best_params, best_fitness
end
