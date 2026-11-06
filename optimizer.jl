##### optimizer.jl — PSO over (α, κ), in stages of narrowing boxes #####
#
# The search runs over (α, κ) only. T and the aggregates (χ^L, χ^s, A) are never
# searched: every objective evaluation profiles them at the particle (profiling.jl). A
# particle whose profile does not converge is REJECTED (loss Inf), never scored at a T
# the model does not imply.

using Distributed
using Random
using Statistics
using NPZ
using Printf

##################### PSO ###################

"""
    parallel_pso_smm(objective, lb, ub; n_particles, max_iter, warm_start_particle,
                     t_conv_probe, verbose) -> (best_x, best_f, history)

Minimise `objective` over the box `[lb, ub]` by particle swarm optimisation (PSO).
The initial swarm is a scrambled Sobol net, the walls reflect, the inertia decreases
linearly and stagnant particles are periodically restarted. The warm start is always
one of the particles, so a stage never ends worse than its incumbent.

# Arguments
- `objective`: the loss of one particle; it is evaluated in parallel with `pmap`, and `Inf` marks a rejected particle.
- `lb`, `ub`: the lower and upper bounds of the search box, one entry per searched coordinate.
- `n_particles`: the swarm size, which should be a multiple of `nworkers()`.
- `max_iter`: the number of PSO iterations after the initial swarm.
- `warm_start_particle`: a point that joins the swarm as one particle (clamped to the box); `nothing` means none.
- `w_start`, `w_end`: the inertia weight at the first and last iteration (linear in between).
- `c1`: the pull of each particle towards its own best position (cognitive weight).
- `c2`: the pull of each particle towards the swarm's best position (social weight).
- `bounce_damping`: the share of its velocity a particle keeps when it bounces off a wall.
- `t_conv_probe`: an optional function that reports whether the T inversion converges at a particle; it is printed with the progress.
- `verbose`: whether to print the progress every 5 iterations.

# Returns
- `best_x`: the best position found, in the searched coordinates only.
- `best_f`: the loss at `best_x`.
- `history`: a `Dict` with the best and the mean finite loss at each iteration (entry 1 is the initial swarm).
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

    # 1) Draw the initial swarm: a scrambled Sobol net over the box, plus the warm start.
    n_random = warm_start_particle === nothing ? n_particles : n_particles - 1
    particles = if n_random == 0
        Vector{Vector{Float64}}()
    else
        net = sobol_scrambled_net(n_random, d, Random.GLOBAL_RNG)
        # The scramble comes from the global RNG (seeded in main.jl), so the swarm is reproducible.
        [lb .+ net[i, :] .* (ub .- lb) for i in 1:n_random]
    end
    warm_start_particle === nothing ||
        push!(particles, clamp.(warm_start_particle, lb, ub))
    velocities = [0.1 * (ub .- lb) .* randn(d) for _ in 1:length(particles)]

    # 2) Score the initial swarm and set each particle's best and the swarm's best.
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
        # 3) Move every particle: inertia plus pulls towards its own best and the swarm's best, with the step capped at 10% of the box and reflection at the walls.
        w = w_start - (w_start - w_end) * (iter / max_iter)
        v_max = 0.1 * (ub .- lb)
        for i in 1:length(particles)
            r1, r2 = rand(d), rand(d)
            velocities[i] = clamp.(w * velocities[i] .+ c1 * r1 .* (p_best[i] .- particles[i]) .+
                                   c2 * r2 .* (g_best .- particles[i]), -v_max, v_max)
            particles[i] = particles[i] .+ velocities[i]
            reflect_bounds!(particles[i], velocities[i], lb, ub, bounce_damping)
        end

        # 4) Score the new positions in parallel and update the personal and global bests.
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

        # 5) Every 25 iterations after the 30th, restart the particles whose loss exceeds 1.5 × the best at a random point.
        if iter > 30 && iter % 25 == 0
            for i in 1:length(particles)
                fitness[i] > g_best_fitness * 1.5 || continue
                particles[i] = lb .+ rand(d) .* (ub .- lb)
                velocities[i] = 0.1 * (ub .- lb) .* randn(d)
                p_best[i] = copy(particles[i]); p_best_fitness[i] = Inf
            end
        end

        # 6) Record the history and print the progress, with the share of particles whose T inversion failed.
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
    alpha_box(anchor) -> (lo, hi)

α box `[BOUND_LO, BOUND_HI] × anchor`, capped at `ALPHA_MAX` in levels; the floor stays
below half the ceiling so the box never collapses.
"""
function alpha_box(anchor)
    hi = min.(anchor .* BOUND_HI, ALPHA_MAX)
    return (min.(anchor .* BOUND_LO, 0.5 .* hi), hi)
end

# Binned α must be increasing (α_1 ≤ … ≤ α_K). The objective evaluates the α block
# SORTED (`stage_to_full_params`), so it is symmetric in the order of the α coordinates and the
# returned θ is increasing. A sorted point stays in the box when the bounds are sorted
# too: with lo and hi each increasing, the k-th order statistic of any x ∈ [lo, hi] lies
# in [lo_k, hi_k]. Sorting the bounds keeps lo ≤ hi entrywise and is a no-op for an
# increasing anchor, or for N_TAU == 1.
sorted_box(lo, hi) = (sort(lo), sort(hi))

"""
    train_stage(n_particles, max_iter; start, fresh, tightness, weight_matrix, moment_blocks)
        -> (best_params, best_fitness, history)

Run one PSO stage over (α, κ) and return the full parameter vector at its optimum. T and
the aggregates (χ^L, χ^s, A) are profiled at every particle (`profile_theta`).

# Arguments
- `n_particles`: the swarm size, which should be a multiple of `nworkers()`.
- `max_iter`: the number of PSO iterations.
- `start`: a full level vector whose (α, κ) joins the swarm as one particle and, unless `fresh`, centres the box; its other entries are ignored.
- `fresh`: `true` searches the whole box (α within `alpha_box` of its prior, κ in `[KAPPA_LO, KAPPA_HI]`); `false` a box around `start` set by `tightness`.
- `tightness`: t ∈ (0, 1], how narrow the box around `start` is: `[α·t, α/t]` and `κ ± (1 − t)·(KAPPA_HI − KAPPA_LO)/2`, both clipped to the whole box, so a larger t narrows the box; ignored when `fresh`.
- `weight_matrix`: the weighting matrix of the loss; `nothing` means `Weight_matrix_custom` (the identity).
- `moment_blocks`: the moment blocks that enter the loss (`[4, 5, 6]` in Step 1, `[4]` in Step 3).

# Returns
- `best_params`: the full level vector `[χ^L | χ^s | A | α | κ | T]` at the best particle, profiled there.
- `best_fitness`: the loss at that particle.
- `history`: the PSO history (`parallel_pso_smm`).
"""
function train_stage(n_particles::Int, max_iter::Int;
                     start::Vector{Float64},
                     fresh::Bool = false,
                     tightness::Real = 0.9,
                     weight_matrix::Union{Nothing, AbstractMatrix} = nothing,
                     moment_blocks::Union{Nothing, Vector{Int}} = [4, 5, 6])
    alpha0, kappa0 = start[alpha_range()], unpack_kappa(start)
    box_lo, box_hi = alpha_box(TAU_PRIOR !== nothing ? TAU_PRIOR : alpha0)

    # 1) Build the search box over (α, κ).
    if fresh
        # 1.1) Fresh start: the whole α box around its prior and the whole κ range.
        alpha_lo, alpha_hi = sorted_box(box_lo, box_hi)
        kappa_lo, kappa_hi = KAPPA_LO, KAPPA_HI
    else
        # 1.2) Continue stage: a box around `start`, narrower as the tightness rises, clipped to the whole box.
        alpha_lo = max.(alpha0 .* tightness, box_lo); alpha_hi = min.(alpha0 ./ tightness, box_hi)
        bad = alpha_lo .>= alpha_hi                 # incumbent above the cap: reset to the whole box
        alpha_lo = ifelse.(bad, box_lo, alpha_lo); alpha_hi = ifelse.(bad, box_hi, alpha_hi)
        alpha_lo, alpha_hi = sorted_box(alpha_lo, alpha_hi)
        half = (1 - tightness) * (KAPPA_HI - KAPPA_LO) / 2
        kappa_lo, kappa_hi = max(kappa0 - half, KAPPA_LO), min(kappa0 + half, KAPPA_HI)
        kappa_lo >= kappa_hi && ((kappa_lo, kappa_hi) = (KAPPA_LO, KAPPA_HI))
    end
    lb, ub = vcat(alpha_lo, kappa_lo), vcat(alpha_hi, kappa_hi)

    # 2) Map a particle (α, κ) to a full level vector, with the α block sorted (see `sorted_box`).
    function to_full(x)
        xf = copy(start)
        xf[alpha_range()] = sort(x[1:N_TAU])
        xf[kappa_index()] = x[end]
        return xf
    end
    # 3) Score a particle: profile T and the aggregates, then compute the loss on `moment_blocks`; Inf if the profile fails.
    function objective(x)
        prof = profile_theta(to_full(x))
        prof.converged || return Inf
        result = safe_SMM(prof.theta; W_override = weight_matrix, moment_blocks = moment_blocks)
        return (result === nothing || !isfinite(result[1][1])) ? Inf : result[1][1]
    end
    # 4) Diagnostic only: report whether the profile converges at a particle, without scoring it.
    function probe(x)
        try
            p = profile_theta(to_full(x))
            return (converged = p.converged, alpha = p.alpha)
        catch
            return (converged = false, alpha = Float64[])
        end
    end

    println("\n" * "="^60)
    println("PSO stage (α, κ)", fresh ? ", fresh start" : ", tightness $tightness",
            " | particles $n_particles | iterations $max_iter | dimension $(length(lb))")
    println("="^60)
    # 5) Run the PSO: each iteration scores every particle in parallel with `objective`.
    best_x, best_fitness, history = parallel_pso_smm(objective, lb, ub;
        n_particles = n_particles, max_iter = max_iter, warm_start_particle = vcat(alpha0, kappa0),
        t_conv_probe = probe)

    # 6) Profile once more at the best particle, because the PSO returns only (α, κ).
    prof = profile_theta(to_full(best_x))
    prof.converged || @warn "the profile did NOT converge at the stage optimum (no particle " *
        "converged): the returned T is the last iterate." resid=prof.resid iters=prof.iters
    return prof.theta, best_fitness, history
end

"""
    run_optimization(; start, fresh_start, weight_matrix, loss_blocks, output_subfolder,
                     max_loop, n_particles, max_iter_initial, max_iter_stage)
        -> (best_params, best_fitness)

A first PSO stage (stage 0), then `max_loop` stages, each centred on the previous optimum,
with a box tightness raised from 0.3 to 0.9 so the box narrows over the loops. Every
stage writes `<step>/0/` or `<step>/epoch_k/k/` (`best_params.npy` and its report); the
loops stop when the parameters move by less than 1e-6 between consecutive stages.

# Arguments
- `start`: the full level vector the search starts from (its (α, κ) is the warm-start particle).
- `fresh_start`: `true` (Step 1) searches the whole box at stage 0; `false` (Step 3) a box of tightness 0.5 around `start`.
- `weight_matrix`: the weighting matrix of the loss; `nothing` means the identity.
- `loss_blocks`: the moment blocks in the loss. Step 1 uses `[4, 5, 6]`: blocks 1–3 and γ are matched by the profile, and their SIMULATED values differ from the targets only by the Monte-Carlo error of the realised price indices. Step 3 uses `[4]` with the fake-world W.
- `output_subfolder`: the step folder under `output_folder`.
- `max_loop`: the maximum number of stages after stage 0.
- `n_particles`: the swarm size, which should be a multiple of `nworkers()`.
- `max_iter_initial`, `max_iter_stage`: the PSO iterations of stage 0 and of each later stage.

# Returns
- `best_params`: the full level vector at the last stage's optimum.
- `best_fitness`: its loss.
"""
function run_optimization(; start::Vector{Float64},
                          fresh_start::Bool = true,
                          weight_matrix::Union{Nothing, AbstractMatrix} = nothing,
                          loss_blocks::Vector{Int} = [4, 5, 6],
                          output_subfolder::String = "step1",
                          max_loop::Int = 20,
                          n_particles::Int = 98,
                          max_iter_initial::Int = 200,
                          max_iter_stage::Int = 50)
    loop_base = joinpath(output_folder, output_subfolder)
    mkpath(loop_base)
    function save_stage(folder, params, label)
        mkpath(folder)
        NPZ.npzwrite(joinpath(folder, "best_params.npy"), reshape(params, :, 1))
        generate_report(folder, params; variable = label)
    end

    # 1) Stage 0: the whole box (Step 1) or a box of tightness 0.5 around `start` (Step 3).
    best_params, best_fitness, _ = train_stage(n_particles, max_iter_initial; start = start,
        fresh = fresh_start, tightness = 0.5, weight_matrix = weight_matrix, moment_blocks = loss_blocks)
    save_stage(joinpath(loop_base, "0"), best_params, "initial fit")

    # 2) Later stages: centred on the previous optimum, the box narrowing as the tightness rises.
    tight_start, tight_end = 0.3, 0.9
    for loop in 1:max_loop
        tightness = tight_start + (loop - 1) * (tight_end - tight_start) / max(max_loop - 1, 1)
        println("\n[$output_subfolder] LOOP $loop/$max_loop  tightness=$tightness")
        prev = best_params
        best_params, best_fitness, _ = train_stage(n_particles, max_iter_stage; start = prev,
            tightness = tightness, weight_matrix = weight_matrix, moment_blocks = loss_blocks)
        save_stage(joinpath(loop_base, "epoch_$loop", string(loop)), best_params, "(α, κ), tightness $tightness")
        println("  loop $loop done, fitness $(round(best_fitness, digits=6))")
        # 3) Stop when the parameters no longer move.
        if loop > 2 && maximum(abs.(best_params .- prev) ./ (abs.(prev) .+ 1e-10)) < 1e-6
            println("  converged: Δparams < 1e-6.")
            break
        end
    end
    run_reporting(loop_base, max_loop)
    return best_params, best_fitness
end
