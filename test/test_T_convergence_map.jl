##### test_T_convergence_map.jl #####
# WHERE does the GE-Sinkhorn T inversion fail? STANDALONE + PRINT-ONLY.
#
#     julia test/test_T_convergence_map.jl auto 4 1 true aa
#     julia -p 16 test/test_T_convergence_map.jl auto 4 1 true aa
#
# Args: industry n_coef n_tau granular ca_level
#
# The PSO log reports one pooled number — "T non-converged: n/N particles" — and that
# number cannot tell you WHY. Two hypotheses produce the same pooled rate:
#
#   (H1) the α box reaches too high. τ = d^α, so a large α spreads trade costs over
#        many orders of magnitude, and `invert_T_ge` has to push remote-origin T
#        correspondingly hard to hold the observed γ_ls fixed. The failures should then
#        concentrate in the top α bins and the lever is ALPHA_MAX / BOUND_HI.
#
#   (H2) the failures live somewhere else in the head — Ω^L, Ω^s, A — in which case
#        narrowing α changes the rate very little and something else has to give
#        (damping, max_iter, or the bounds on that block).
#
# The two are distinguished by CONDITIONING, which is what this file does: it maps the
# convergence indicator over the same fresh-start box `train_stage` builds, reports the
# failure rate BY α, and then asks whether the other blocks separate the failures once
# α is held fixed. It also splits every failure into "slow" (residual just above tol,
# still contracting at the iteration cap) versus "not contracting" (residual orders of
# magnitude above tol, or oscillating) — those need different fixes.
#
# Print-only: nothing here touches an estimate. Writes a summary .npz and, if Plots is
# available, a rate-vs-α figure under <output_folder>/T_convergence/.

using Distributed
@everywhere using Random
@everywhere using NPZ
using LinearAlgebra, Statistics, Printf

# Plots lives only in the entry points; this file is standalone, so pull it in here and
# degrade to the .npz alone if it is unavailable.
const HAVE_PLOTS = try; @eval using Plots; true; catch; false; end

@everywhere include("../model_CP.jl")
@everywhere include("../tools.jl")
@everywhere include("../model_analytical.jl")

industry  = length(ARGS) >= 1 ? ARGS[1] : "auto"
n_coef    = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4
n_tau     = length(ARGS) >= 3 && !isempty(strip(ARGS[3])) ? parse(Int, ARGS[3]) : 1
granular  = length(ARGS) >= 4 ? (lowercase(strip(ARGS[4])) in ("true","1","yes")) : true
ca_level  = length(ARGS) >= 5 && !isempty(strip(ARGS[5])) ? Symbol(strip(ARGS[5])) : :aa
K_sim     = 10000
reg_method = :cloglog

input_folder  = "./baseline_$industry"
output_folder = "./reporting_$industry" * (ca_level == :aa ? "_aa" : "") *
                (granular ? "_gran" : "") * "_Tconv"
mkpath(output_folder)

include("../load_parameters.jl")
@everywhere include("../profiling.jl")

out_dir = joinpath(output_folder, "T_convergence"); mkpath(out_dir)

# The inversion's own settings, repeated here so the report can name them.
const INV_TOL      = 1e-9
const INV_MAX_ITER = 500
const INV_DAMPING  = 0.9

println("\n" * "="^78)
println("T-INVERSION CONVERGENCE MAP")
@printf("  industry=%s  CA_LEVEL=:%s  GRANULAR=%s  N_TAU=%d\n",
        industry, CA_LEVEL, GRANULAR, N_TAU)
@printf("  invert_T_ge: tol=%.1e  max_iter=%d  damping=%.2f\n",
        INV_TOL, INV_MAX_ITER, INV_DAMPING)
@printf("  workers=%d\n", nworkers())
println("="^78)

# ───────────────────────────────────────────────────────────────────────────
# The box, built exactly the way train_stage builds the FRESH-START box
# (optimizer.jl, `last_stage_folder === nothing`). Head only — T is profiled,
# so the search space under profile_T is [Ω^L | Ω^s | A | α].
# ───────────────────────────────────────────────────────────────────────────
A_init = copy(emp_pi_r_full).^(1/abs(epsilon)) .* regional_wages[N_downstream_per_region .!= 0]
A_init ./= sum(A_init)

alpha_anchor = TAU_PRIOR !== nothing ? TAU_PRIOR : fill(P_alpha, N_TAU)
alpha_hi = min.(alpha_anchor .* BOUND_HI, ALPHA_MAX)          # mirrors `alpha_box`
alpha_lo = min.(alpha_anchor .* BOUND_LO, 0.5 .* alpha_hi)

head_lo = vcat(0.8 * agg_labor_share, 0.8 .* agg_industry_share, 0.8 .* A_init, alpha_lo)
head_hi = vcat(1.2 * agg_labor_share, 1.2 .* agg_industry_share, 1.2 .* A_init, alpha_hi)
n_head  = length(head_lo)
@assert n_head == 1 + S + R_downstream + N_TAU

# Block slices inside the head, for the conditional analysis in section 3.
const BLK = [("Omega_L", 1:1),
             ("Omega_s", 2:(1 + S)),
             ("A",       (S + 2):(S + R_downstream + 1)),
             ("alpha",   (S + R_downstream + 2):n_head)]

T_TAIL   = vec(permutedims(T_rs_init))[T_MASK]     # inert; profiled_theta_probe ignores it
head_mid = 0.5 .* (head_lo .+ head_hi)
θ_warm   = vcat([agg_labor_share], agg_industry_share, A_init, alpha_anchor, T_TAIL)

@printf("\n  head dimension  = %d   (Ω^L 1, Ω^s %d, A %d, α %d)\n",
        n_head, S, R_downstream, N_TAU)
@printf("  α box           = [%.4g, %.4g]   (prior %.4g, BOUND_LO=%.3g, BOUND_HI=%.3g, ALPHA_MAX=%.3g)\n",
        alpha_lo[1], alpha_hi[1], alpha_anchor[1], BOUND_LO, BOUND_HI, ALPHA_MAX)
@printf("  Ω^L, Ω^s, A     = ±20%% of their analytic initialisation\n")

probe_head(head) = profiled_theta_probe(vcat(head, T_TAIL))

# ═══════════════════════════════════════════════════════════════════════════
# 1. α ALONE — every other block pinned at the warm start
#    Isolates H1: if the inversion is fine at low α and fails above some α*, that
#    threshold is the number ALPHA_MAX should be set from.
# ═══════════════════════════════════════════════════════════════════════════
println("\n── 1. α SWEEP (all other blocks at the warm start) ────────────────────")

n_alpha_grid = 40
alpha_grid = [alpha_lo[1] + (k - 1) * (alpha_hi[1] - alpha_lo[1]) / (n_alpha_grid - 1)
              for k in 1:n_alpha_grid]
# N_TAU > 1: move every component together along the grid, which is the direction the
# α-ordering constraint keeps the swarm in anyway.
sweep_heads = [vcat(θ_warm[1:(S + R_downstream + 1)], fill(a, N_TAU)) for a in alpha_grid]
sweep = pmap(probe_head, sweep_heads)

println("       α        conv   iters      resid")
for (a, p) in zip(alpha_grid, sweep)
    @printf("  %9.5f   %-5s  %5d   %.3e%s\n", a, p.converged ? "yes" : "NO",
            p.iters, p.resid, p.converged ? "" : "   ←")
end
sweep_bad = [!p.converged for p in sweep]
alpha_star = findfirst(sweep_bad)
if alpha_star === nothing
    println("\n  → α alone NEVER breaks the inversion over the whole box.")
    println("    H1 is NOT the story on this axis: whatever the swarm's failures are,")
    println("    they are not produced by α on its own. Go to sections 2 and 3.")
else
    @printf("\n  → first failure at α = %.5f (grid point %d/%d).\n",
            alpha_grid[alpha_star], alpha_star, n_alpha_grid)
    if all(sweep_bad[alpha_star:end])
        println("    Failures are a clean upper TAIL in α ⇒ H1. Setting ALPHA_MAX just")
        @printf("    below %.4f removes them at the source.\n", alpha_grid[alpha_star])
    else
        println("    Failures are INTERLEAVED with successes along α — not a clean")
        println("    threshold, so a lower ALPHA_MAX alone will not clear them.")
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# 2. THE WHOLE BOX — the rate the PSO log actually sees, decomposed by α
# ═══════════════════════════════════════════════════════════════════════════
println("\n── 2. RANDOM PARTICLES FROM THE FULL HEAD BOX ─────────────────────────")

n_draw = 400
Random.seed!(20260803)
box_heads = [head_lo .+ rand(n_head) .* (head_hi .- head_lo) for _ in 1:n_draw]
box = pmap(probe_head, box_heads)

box_bad   = [!p.converged for p in box]
box_alpha = [maximum(p.alpha) for p in box]
n_bad     = count(box_bad)
@printf("  overall: %d/%d non-converged (%.1f%%)\n", n_bad, n_draw, 100 * n_bad / n_draw)

n_bins = 8
bin_lo = [alpha_lo[1] + (b - 1) * (alpha_hi[1] - alpha_lo[1]) / n_bins for b in 1:n_bins]
bin_hi = [alpha_lo[1] + b * (alpha_hi[1] - alpha_lo[1]) / n_bins for b in 1:n_bins]
bin_rate = fill(NaN, n_bins); bin_n = zeros(Int, n_bins); bin_bad = zeros(Int, n_bins)
println("\n  failure rate by α bin:")
println("        α bin              n     bad    rate")
for b in 1:n_bins
    idx = [i for i in 1:n_draw if box_alpha[i] >= bin_lo[b] &&
                                  (box_alpha[i] < bin_hi[b] || b == n_bins)]
    bin_n[b] = length(idx); isempty(idx) && continue
    bin_bad[b]  = count(i -> box_bad[i], idx)
    bin_rate[b] = bin_bad[b] / bin_n[b]
    @printf("  [%7.4f, %7.4f)  %5d   %5d   %5.1f%%  %s\n",
            bin_lo[b], bin_hi[b], bin_n[b], bin_bad[b], 100 * bin_rate[b],
            "█"^round(Int, 20 * bin_rate[b]))
end

# Is the rate actually increasing in α, or flat? Compare the bottom and top halves.
lo_half = [i for i in 1:n_draw if box_alpha[i] < 0.5 * (alpha_lo[1] + alpha_hi[1])]
hi_half = setdiff(1:n_draw, lo_half)
rate_lo = isempty(lo_half) ? NaN : count(i -> box_bad[i], lo_half) / length(lo_half)
rate_hi = isempty(hi_half) ? NaN : count(i -> box_bad[i], hi_half) / length(hi_half)
@printf("\n  bottom half of the α box: %.1f%%   top half: %.1f%%\n",
        100 * rate_lo, 100 * rate_hi)

# ═══════════════════════════════════════════════════════════════════════════
# 3. ARE THERE OTHER REGIONS? — condition on α, look at the other blocks
#
#    Each coordinate is mapped to its POSITION IN ITS OWN BOX, u = (x−lo)/(hi−lo) ∈
#    [0,1], so blocks on wildly different scales are comparable. For each block we
#    compare the mean position of the FAILURES against that of the successes, WITHIN
#    the low-α half (where α cannot be doing the work). A block whose failures sit
#    systematically at one end of its own box is a second failure region; a block
#    whose failures look like a uniform draw is not implicated.
# ═══════════════════════════════════════════════════════════════════════════
println("\n── 3. CONDITIONING ON α — do the OTHER blocks separate the failures? ──")

upos(h) = (h .- head_lo) ./ max.(head_hi .- head_lo, eps())

sub = lo_half                       # low-α half only: α is held roughly fixed here
sub_bad = [i for i in sub if box_bad[i]]
sub_ok  = [i for i in sub if !box_bad[i]]

if length(sub_bad) < 5 || length(sub_ok) < 5
    @printf("  low-α half: %d failures / %d successes out of %d draws.\n",
            length(sub_bad), length(sub_ok), length(sub))
    println(length(sub_bad) < 5 ?
        "  Too few failures there to compare against: α is doing essentially ALL of the\n  work, so there is no second region to find. H1." :
        "  Almost EVERYTHING fails in the low-α half too, so α is not the discriminant\n  at all — the inversion is marginal across the whole box. Read section 4.")
else
    @printf("  %d failures / %d draws in the low-α half.\n", length(sub_bad), length(sub))
    println("\n  block-mean position in own box (0 = lower bound, 1 = upper bound):")
    println("    block        n_coord   failures   successes      Δ   |Δ|/se")
    for (name, rng) in BLK
        name == "alpha" && continue
        mb = mean([mean(upos(box_heads[i])[rng]) for i in sub_bad])
        mo = mean([mean(upos(box_heads[i])[rng]) for i in sub_ok])
        # se of the difference in means of a per-draw block average; the block average
        # of independent U(0,1) coordinates has variance 1/(12·n_coord).
        sb = std([mean(upos(box_heads[i])[rng]) for i in sub_bad])
        so = std([mean(upos(box_heads[i])[rng]) for i in sub_ok])
        se = sqrt(sb^2 / length(sub_bad) + so^2 / max(length(sub_ok), 1))
        z  = se > 0 ? abs(mb - mo) / se : 0.0
        @printf("    %-10s   %5d    %7.3f    %7.3f   %+6.3f   %6.2f %s\n",
                name, length(rng), mb, mo, mb - mo, z,
                z > 3 ? "← SEPARATES" : "")
    end
    println("""
      A |Δ|/se above ~3 means the failures are drawn from a different part of that
      block's box than the successes — a second failure region, independent of α.
      All small ⇒ within the low-α half the failures are scattered, i.e. the inversion
      is marginal there for reasons no single block explains (read section 4).""")
end

# One-at-a-time corners: α pinned at the prior, one block pushed to each end.
println("\n  corner probes (α at the prior, one block at a time pushed to its bound):")
corner_rows = Tuple{String,String,Bool,Int,Float64}[]
for (name, rng) in BLK
    name == "alpha" && continue
    for (tag, tgt) in (("lo", head_lo), ("hi", head_hi))
        h = copy(θ_warm[1:n_head]); h[rng] = tgt[rng]
        h[(S + R_downstream + 2):n_head] = alpha_anchor
        p = probe_head(h)
        push!(corner_rows, (name, tag, p.converged, p.iters, p.resid))
        @printf("    %-10s %-3s  %-5s  iters=%4d  resid=%.3e\n",
                name, tag, p.converged ? "conv" : "FAIL", p.iters, p.resid)
    end
end
# And α at its own ceiling, everything else at the warm start.
let h = copy(θ_warm[1:n_head])
    h[(S + R_downstream + 2):n_head] = alpha_hi
    p = probe_head(h)
    @printf("    %-10s %-3s  %-5s  iters=%4d  resid=%.3e\n",
            "alpha", "hi", p.converged ? "conv" : "FAIL", p.iters, p.resid)
end

# ═══════════════════════════════════════════════════════════════════════════
# 4. FAILURE ANATOMY — slow, or not contracting?
#    A residual just above tol at the iteration cap is a BUDGET problem (raise
#    max_iter). A residual orders of magnitude above tol, or one that stopped falling,
#    is a CONTRACTION problem (lower `damping`, or keep the particle out of that
#    region). The two have different fixes and the pooled rate hides which it is.
# ═══════════════════════════════════════════════════════════════════════════
println("\n── 4. FAILURE ANATOMY ─────────────────────────────────────────────────")

fail_idx = [i for i in 1:n_draw if box_bad[i]]
if isempty(fail_idx)
    println("  no failures over the box sample — nothing to dissect.")
else
    res_f = [box[i].resid for i in fail_idx]
    finite_f = filter(isfinite, res_f)
    n_exc = count(!isfinite, res_f)
    @printf("  %d failures: %d threw (resid = Inf), %d hit the iteration cap.\n",
            length(fail_idx), n_exc, length(finite_f))
    if !isempty(finite_f)
        srt = sort(finite_f)
        @printf("  terminal residual  min %.2e  median %.2e  max %.2e   (tol = %.1e)\n",
                srt[1], srt[cld(end, 2)], srt[end], INV_TOL)
        n_near = count(r -> r < 1e3 * INV_TOL, finite_f)
        @printf("  within 1000× tol: %d/%d\n", n_near, length(finite_f))
    end

    # Re-run a handful with a bigger budget and with heavier damping.
    n_probe = min(8, length(fail_idx))
    println("\n  re-running $(n_probe) failures with a larger budget / heavier damping:")
    println("    #   base resid   iters×4 resid   damping .5 resid   damping .3 resid")
    retry = pmap(fail_idx[1:n_probe]) do i
        h  = box_heads[i]
        ΩL = h[1]
        Ωs = h[2:(1 + S)];                      Ωs = Ωs ./ sum(Ωs)
        A  = h[(S + 2):(S + R_downstream + 1)]; A  = A ./ A[1]
        α  = h[(S + R_downstream + 2):n_head]
        f(mi, dp) = try
            r = invert_T_ge(α, ΩL, Ωs, A; max_iter = mi, damping = dp)
            (r.resid, r.converged, r.resid_hist)
        catch
            (Inf, false, Float64[])
        end
        (long = f(4 * INV_MAX_ITER, INV_DAMPING),
         d5   = f(INV_MAX_ITER, 0.5),
         d3   = f(INV_MAX_ITER, 0.3))
    end
    n_fixed_budget = 0; n_fixed_damping = 0; n_stuck = 0
    for (k, i) in enumerate(fail_idx[1:n_probe])
        r = retry[k]
        @printf("   %3d   %.3e     %.3e %-4s   %.3e %-4s   %.3e %-4s\n", k, box[i].resid,
                r.long[1], r.long[2] ? "conv" : "", r.d5[1], r.d5[2] ? "conv" : "",
                r.d3[1], r.d3[2] ? "conv" : "")
        r.long[2] && (n_fixed_budget += 1)
        (!r.long[2] && (r.d5[2] || r.d3[2])) && (n_fixed_damping += 1)
        (!r.long[2] && !r.d5[2] && !r.d3[2]) && (n_stuck += 1)
    end
    println()
    @printf("  fixed by 4× iterations : %d/%d  → raise invert_T_ge's max_iter\n",
            n_fixed_budget, n_probe)
    @printf("  fixed by more damping  : %d/%d  → lower `damping` (the map oscillates)\n",
            n_fixed_damping, n_probe)
    @printf("  still failing          : %d/%d  → genuinely non-contracting there;\n",
            n_stuck, n_probe)
    println("                                    the box, not the solver, is the lever")

    # Is the residual still falling at the cap, or has it stalled/oscillated?
    hist = [r.long[3] for r in retry if length(r.long[3]) > 20]
    if !isempty(hist)
        falling = count(h -> h[end] < h[end - 10], hist)
        @printf("\n  of %d dissected paths, %d were still DECREASING at the cap (slow),\n",
                length(hist), falling)
        @printf("  %d had stalled or were oscillating (not contracting).\n",
                length(hist) - falling)
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# 5. VERDICT + artefacts
# ═══════════════════════════════════════════════════════════════════════════
NPZ.npzwrite(joinpath(out_dir, "T_convergence_map.npz"), Dict(
    "alpha_grid"      => collect(alpha_grid),
    "alpha_sweep_bad" => Float64.(sweep_bad),
    "alpha_sweep_res" => [p.resid for p in sweep],
    "alpha_sweep_it"  => Float64.([p.iters for p in sweep]),
    "box_alpha"       => box_alpha,
    "box_bad"         => Float64.(box_bad),
    "box_resid"       => [p.resid for p in box],
    "box_heads"       => reduce(hcat, box_heads),
    "bin_lo"          => bin_lo, "bin_hi" => bin_hi,
    "bin_rate"        => bin_rate, "bin_n" => Float64.(bin_n),
    "head_lo"         => head_lo, "head_hi" => head_hi,
))

if HAVE_PLOTS
    try
        png = joinpath(out_dir, "T_nonconvergence_vs_alpha.png")
        p1 = Base.invokelatest(Plots.bar, 0.5 .* (bin_lo .+ bin_hi), 100 .* bin_rate;
            xlabel = "α", ylabel = "non-converged (%)", legend = false,
            title = "T-inversion failure rate over the head box (n=$n_draw)")
        Base.invokelatest(Plots.savefig, p1, png)
        println("\n  saved $png")
    catch e
        println("\n  (plot skipped: $e)")
    end
end
println("  saved $(joinpath(out_dir, "T_convergence_map.npz"))")

println("\n" * "="^78)
println("READING THIS")
println("="^78)
println("""
  Section 1 answers "is it α ALONE?". A clean threshold there — everything below α*
  converges, everything above fails — means ALPHA_MAX is the whole story and setting
  it just under α* removes the failures at the source, with no other cost.

  Section 2 gives the rate the PSO log reports, split by α. Compare the bottom- and
  top-half rates: a large gap is H1, a small one means α is NOT the binding factor
  even though it is the parameter you can see moving.

  Section 3 is the "are there other regions" question. It holds α in its lower half
  and asks whether the failures are drawn from a different part of any OTHER block's
  box. Ω^s and A enter the inversion through the endogenous expenditure ω — the exact
  channel `invert_T_ge` folds into the Sinkhorn map — so an A concentrated at one end
  of its box among the failures is a real second region, not noise. Ω^L enters through
  the labour/intermediate split in c_r and is the one most likely to be innocent.

  Section 4 says which knob applies. Failures that clear with 4× the iterations are a
  budget problem and cost only time; failures that clear only with heavier damping mean
  the one-step map is oscillating there (‖J_GE‖ is not small — the ν = $(round(nu, digits=2)),
  λ = $(round(lambda, digits=2)) calibration is far from Cobb-Douglas, which is exactly the
  §4 worry in profiling.jl); failures that clear with neither are regions the box should
  not contain.

  One thing this file does NOT do: a non-converged particle is not an error. The
  optimizer falls back on the warm-start T, so those particles are evaluated at a T
  that does not reproduce γ_ls — they get a bad loss and the swarm moves away from
  them. A moderate rate is survivable; what is not survivable is a rate high enough
  that the swarm's ranking is driven by fallbacks rather than by the model.
""")
