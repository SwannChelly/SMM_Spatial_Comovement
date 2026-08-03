##### test_profile_alpha_sweep.jl #####
# WHY DOES SINKHORN-PROFILING PUSH α → 0?  (standalone, print-only diagnostic)
#
#     julia test/test_profile_alpha_sweep.jl aero 4 1
#     julia test/test_profile_alpha_sweep.jl auto 1
#  Args: industry [n_coef] [n_tau] [run_folder] [reg_method] [include_control]
#        [granular] [ca_level]
#     julia test/test_profile_alpha_sweep.jl auto 4 1 ./reporting_auto_profiled_pso cloglog false
#         → cloglog link, NO control group (supplier pairs + size control), on the profiled run
#     julia test/test_profile_alpha_sweep.jl aero 4 1 \
#           ./reporting_aero_profiled_aa_gran_pso cloglog false true aa
#         → the GRANULAR / attraction-area configuration (block 6 = Ḡ_s(0) in the loss)
#
# ⚠ THE CONFIG ARGS ARE NOT OPTIONAL DECORATION. `load_parameters.jl` reads the
# modelling flags as `granular_local = (@isdefined(granular)) ? … : false` and
# `ca_level_local = (@isdefined(ca_level)) ? … : :ze`. If this script does not define
# `granular` / `ca_level` BEFORE the include, it silently loads the LEGACY continuum
# model (granular=false, ca_level=:ze) — no error — and then, because `gb_indices`
# no longer matches the on-disk W_step3, silently falls back to identity weighting
# too. The sweep would then diagnose a different model under a different metric than
# the run you are trying to explain. Pass the same flags you passed to run.sh.
#
# Observed puzzle: without profiling (T searched freely) the estimator lands at
# α ≈ 0.30 with a GOOD reg_coef fit; with profile_T=true (T = invert_T_ge(α,…) so
# γ_ls is matched exactly for every α) it collapses toward the α box floor with a BAD
# reg_coef fit. This script isolates the cause by sweeping α on a grid and, at each α,
# contrasting the two regimes at a FIXED head (Ω^L, Ω^s, A) taken from θ̂:
#
#   (A) PROFILED   : T = invert_T_ge(α)  — the Sinkhorn map re-solves T so the
#                    analytical γ_ls stays pinned to the data for every α.
#   (B) FIXED-T    : T = T̂ (the T from θ̂), held constant as α varies — mimics the
#                    joint search, where α moves reg_coef through the DIRECT τ=d^α
#                    trade-cost channel with T free to absorb γ_ls separately.
#
# WHICH BLOCK OWNS THE MINIMUM (the granular question). The step-3 loss is
# `moment_blocks = [4,5,6]` under GRANULAR (reg_coef + γ_ls + the count moment
# Ḡ_s(0)), weighted by W_step3. Under profiling the γ block is pinned to the target
# by the inversion at EVERY α, so it carries almost no α gradient — which leaves
# reg_coef and G0 to fight over α. This sweep therefore reports, at each α, the
# W-weighted quadratic form r'Wr DECOMPOSED into its per-block diagonal
# contributions r_b'W_bb r_b (cross terms shown as the remainder), so the block that
# actually owns the α minimum is read off directly rather than inferred. Under
# GRANULAR it also prints N̂_s, the number of sectors CLAMPED at a variety bound, and
# the G0 fit — a clamped sector cannot zero its own count residual with N_s, so that
# residual is pushed onto α.
#
# CONTROL-GROUP EXTENSION: where production appends control-group y=0 rows
# (REG_INCLUDE_CONTROL == true, i.e. CA_LEVEL == :ze), the sweep computes reg_coef
# BOTH with and without them. invert_T_ge pins γ_ls from supplier pairs only, so
# T*(α) is identical in both variants — only the reg_coef regressand changes. This
# tests whether the far-distance control zeros (which invert_T_ge cannot fill:
# control T≡0) break the τ/T*(α) cancellation and restore an interior α*. Under
# CA_LEVEL == :aa the control cells ARE ordinary goods (REG_INCLUDE_CONTROL is forced
# false), so the two arms would be identical and the second arm is SKIPPED.
#
# EXPECTED (the mechanism): under FIXED-T the reg_coef loss has a clear interior
# minimum near α ≈ 0.3 (α steepens reg_coef via τ, identifying it). Under PROFILED
# the reg_coef loss is (near-)monotone toward α → 0, because raising α forces
# invert_T_ge to raise T for remote origins (to keep γ_ls fixed), and that T boost
# FLATTENS reg_coef — cancelling the very τ channel that identifies α. The exact-
# γ_ls hard constraint of profiling thus profiles out most of α's leverage on
# reg_coef, so the residual objective is minimised at the α → 0 boundary (where
# τ ≈ 1 makes the inversion a clean, well-conditioned matrix scaling). This is an
# identification artefact of profiling, NOT a data feature. Under GRANULAR the count
# moment can ADD a second, independent pull toward small α: lowering α forces the
# inversion to load the whole remoteness gradient onto T, which raises the
# cross-cell DISPERSION of q̂, and since (1−q)^N is convex in q that raises
# Ḡ_s(0) = mean_l (1−q̂_l)^{N_s} toward its target. The block decomposition below is
# what separates that channel from the reg_coef one.
#
# Touches no production path (includes the model + load_parameters.jl the way
# test_ge_inversion.jl does; defines nothing global that production reads).

using Distributed
@everywhere using Random
@everywhere using NPZ
using LinearAlgebra, Statistics, Printf

@everywhere include("../model_CP.jl")
@everywhere include("../tools.jl")
@everywhere include("../model_analytical.jl")
@everywhere include("../optimizers/pso_integration.jl")

industry = length(ARGS) >= 1 ? ARGS[1] : "aero"
n_coef   = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4
n_tau    = length(ARGS) >= 3 && !isempty(strip(ARGS[3])) ? parse(Int, ARGS[3]) : n_coef
# 4th arg: reporting folder to anchor θ̂ on. DEFAULT = the NON-profiled PSO run
# (reporting_<industry>_pso), i.e. the free-T fit where α≈0.30 with a good reg_coef
# fit — that is the θ̂ whose T̂ we hold fixed in regime (B). main.jl writes the
# profiled run to reporting_<industry>_profiled_pso (plus _aa/_gran suffixes); pass
# it explicitly to anchor on the profiled (boundary-α) estimate instead.
optimizer_backend = :pso
K_sim    = 10000

# 5th/6th args: extensive-margin config for the whole sweep (target + eval_moments).
#   reg_method      (5th) : :cloglog (default) or :lpm
#   include_control (6th) : true (default) or false ⇒ supplier pairs only WITH the
#                           size control (coupled). MUST be set BEFORE load_parameters
#                           so REG_METHOD/INCLUDE_CONTROL and the empirical target match.
reg_method = length(ARGS) >= 5 && !isempty(strip(ARGS[5])) ? Symbol(strip(ARGS[5])) : :cloglog
@assert reg_method in (:lpm, :cloglog) "reg_method must be lpm|cloglog, got :$reg_method"
include_control = length(ARGS) >= 6 && !isempty(strip(ARGS[6])) ?
    (lowercase(strip(ARGS[6])) in ("true", "1", "yes")) : true

# 7th/8th args: the MODELLING flags. These MUST be defined (as `granular`/`ca_level`,
# the exact names load_parameters.jl probes with @isdefined) before the include, or
# the legacy continuum model is loaded silently. Defaults reproduce the legacy model.
granular = length(ARGS) >= 7 && !isempty(strip(ARGS[7])) ?
    (lowercase(strip(ARGS[7])) in ("true", "1", "yes")) : false
ca_level = length(ARGS) >= 8 && !isempty(strip(ARGS[8])) ? Symbol(strip(ARGS[8])) : :ze
@assert ca_level in (:ze, :aa) "ca_level must be ze|aa, got :$ca_level"

input_folder  = "./baseline_$industry"
output_folder = length(ARGS) >= 4 && !isempty(strip(ARGS[4])) ? String(ARGS[4]) :
                "./reporting_$(industry)_$(optimizer_backend)"
mkpath(output_folder)
@printf("Anchoring θ̂ / W_step3 on: %s\n", output_folder)
@printf("Config: reg_method=:%s  include_control(group)=%s  granular=%s  ca_level=:%s\n",
        reg_method, include_control, granular, ca_level)

include("../load_parameters.jl")
include("../profiling.jl")

@assert N_TAU == 1 "This sweep assumes the power-law single-α parametrization (N_TAU==1)."

println("\n" * "="^72)
println("PROFILING α-SWEEP: why does invert_T_ge push α → 0 ?")
@printf("  industry=%s  N_REG=%d  N_TAU=%d  |  ν=%.3g  λ=%.3g  θ=%.3g\n",
        industry, N_REG, N_TAU, nu, lambda, theta)
@printf("  GRANULAR=%s  CA_LEVEL=:%s  |  T_COL_DIM=%d  n_good=%d  n_T=%d\n",
        GRANULAR, CA_LEVEL, T_COL_DIM, n_good, count(T_MASK))
println("="^72)

# ── θ̂: prefer a saved estimate (same config), else a documented default ───────
function load_theta()
    expected = 1 + S + R_downstream + N_TAU + count(T_MASK)
    for rel in (("step3", "theta_hat_2.npy"), ("step1", "theta_hat_1.npy"),
                ("step1", "best_params.npy"))
        p = joinpath(output_folder, rel...)
        isfile(p) || continue
        θ = NPZ.npzread(p); ndims(θ) > 1 && (θ = θ[:, 1])
        if length(θ) == expected
            @printf("Loaded θ̂ from %s (length %d)\n", p, length(θ)); return θ
        else
            @printf("SKIP %s: length %d ≠ expected %d — wrong config for this θ̂?\n",
                    p, length(θ), expected)
        end
    end
    T_red = vec(permutedims(T_rs_init))[T_MASK]
    Ω_L   = 0.5; Ω_s = ones(S); A = ones(R_downstream); β = [0.30]
    θ = vcat(Ω_L, Ω_s, A, β, T_red)
    @printf("Using DEFAULT θ (T=T_rs_init, Ω^L=0.5, Ω^s/A=1, α=0.30)\n")
    @assert length(θ) == expected
    return θ
end

θ0                 = load_theta()
Ω_L, Ω_s, A, α̂, _  = unpack_params(θ0)
# T in the T-COLUMN space (S, T_COL_DIM) — ZE under :ze, attraction areas under :aa.
# This is the space `invert_T_ge` iterates in and the space `assemble_theta`'s
# `vec(permutedims(T_mat))[T_MASK]` expects. `unpack_params` returns the GATHERED
# ZE-level (S, R) matrix instead, which coincides with the column space only under
# :ze — passing it under :aa is a DimensionMismatch.
T_par_hat = unpack_T_par(θ0)
@assert size(T_par_hat) == (S, T_COL_DIM)
@printf("θ̂ head: Ω^L=%.4f  |Ω^s|=%d  |A|=%d  α̂=%.4f  |  T̂ block %d×%d\n",
        Ω_L, S, R_downstream, α̂[1], size(T_par_hat, 1), size(T_par_hat, 2))

# ── Efficient weight matrix for the combined β+γ(+G) loss ─────────────────────
# gb_indices MUST come from inference_moment_indices() — it appends BLOCK_RANGES[6]
# (the count moment) under GRANULAR, which a hard-coded vcat(BLOCK_RANGES[4],[5])
# would drop. Dropping it both mis-sizes W_step3 (silent identity fallback) and hides
# the block that owns the α minimum.
gb_indices          = inference_moment_indices()
gb_ranges, gb_names = inference_block_layout()
W_gb = Matrix{Float64}(I, length(gb_indices), length(gb_indices))
W_is_efficient = false
let p = joinpath(output_folder, "step2", "W_step3.npy")
    if isfile(p)
        Wt = NPZ.npzread(p)
        if size(Wt, 1) == length(gb_indices)
            global W_gb = Wt
            global W_is_efficient = true
            @printf("Loaded W_step3 (%d×%d) for the β+γ%s loss.\n",
                    size(Wt, 1), size(Wt, 2), GRANULAR ? "+G" : "")
        else
            @printf("⚠ W_step3 size %d ≠ %d β+γ%s moments — using IDENTITY. This usually\n",
                    size(Wt, 1), length(gb_indices), GRANULAR ? "+G" : "")
            println("  means the granular/ca_level flags here differ from the run that built it.")
        end
    else
        println("No W_step3 on disk — β+γ loss uses identity weighting.")
    end
end

emp_vec   = vec(empirical_moments)
reg_range = BLOCK_RANGES[4]
emp_reg   = emp_vec[reg_range]
@printf("Empirical reg_coef: %s\n", string(round.(emp_reg, digits=4)))

"""Simulated moments at a full level θ, masked. Note: full_SMM(θ; simulation=false)
returns (loss, moments_tuple); we take the tuple. (simulation=true returns the X_ls
trade matrix, not moments — do not use.)"""
function eval_moments(θ)
    _, sim = full_SMM(θ; u_draws = U_DRAWS, sample_weights = SAMPLE_WEIGHTS)
    return moments_to_vec(sim)
end

"""Decompose the weighted criterion r'Wr over the β+γ(+G) subsystem into per-block
DIAGONAL contributions r_b' W_bb r_b. The blocks do not sum to the total because W
is not block-diagonal; the gap is reported as `cross`. `unw` is the unweighted SSE
per block, for reading the raw fit alongside the weighted one."""
function block_losses(sim_vec)
    err     = emp_vec .- sim_vec
    err_gb  = err[gb_indices]
    gb_loss = err_gb' * W_gb * err_gb
    # Blocks are small (N_REG, n_γ, S), so plain slices are fine and avoid the
    # parsing fragility of `@view` inside a call argument list.
    diagc   = [ (e = err_gb[r]; e' * W_gb[r, r] * e) for r in gb_ranges ]
    unw     = [ sum(abs2, err_gb[r])                 for r in gb_ranges ]
    return gb_loss, diagc, unw, sim_vec[reg_range]
end

# reg_coef WITHOUT the control-group y=0 rows (supplier pairs only), at the SAME
# network/draws. `include_control` toggles ONLY the appended control rows. Profiling
# (invert_T_ge) pins γ_ls from the supplier pairs alone, so T*(α) is IDENTICAL with or
# without controls — the only thing that differs is the reg_coef regressand. Only
# informative where production actually HAS control rows (REG_INCLUDE_CONTROL); under
# CA_LEVEL=:aa control cells are ordinary goods and the arms coincide.
const RUN_NOCTRL_ARM = REG_INCLUDE_CONTROL
_reg_fn(link) = link == :cloglog ? fast_cloglog_regression : fast_weighted_regression
function reg_coef_without_control(θ)
    net = solve_network(θ; u_draws = U_DRAWS, sample_weights = SAMPLE_WEIGHTS)
    _reg_fn(REG_METHOD)(net.linkages_flat, net.z_flat, net.sample_weights;
                        include_control = false)
end
reg_loss_of(rc) = sum((emp_reg .- rc).^2)
rc_str(v) = string(round.(v[1:min(end, 4)], digits = 3))

if !RUN_NOCTRL_ARM
    println("\nNOTE: REG_INCLUDE_CONTROL=false (CA_LEVEL=:aa ⇒ control cells are goods),")
    println("      so the with/without-control arms would be identical. Arm SKIPPED.")
end

# ── α grid ────────────────────────────────────────────────────────────────────
# Anchored on the production search box so the boundary is actually visited:
# train_stage boxes α at [BOUND_LO, BOUND_HI] × TAU_PRIOR capped by ALPHA_MAX, with
# BOUND_LO = 0.1. A sweep that stops at 0.02/0.60 can miss both the floor the
# optimizer converges onto and α̂ itself.
α_grid = collect(range(0.02, 0.60, length = 15))
let extra = Float64[α̂[1]]
    if (@isdefined TAU_PRIOR) && TAU_PRIOR !== nothing && !isempty(TAU_PRIOR)
        push!(extra, 0.1 * TAU_PRIOR[1])          # BOUND_LO × prior = the box floor
        @printf("α box floor (BOUND_LO×prior) = %.5f ; ALPHA_MAX = %.3g\n",
                0.1 * TAU_PRIOR[1], (@isdefined ALPHA_MAX) ? ALPHA_MAX : NaN)
    end
    global α_grid = sort(unique(round.(vcat(α_grid, extra), digits = 5)))
end
@printf("α grid (%d points): %.4f … %.4f\n", length(α_grid), first(α_grid), last(α_grid))

if RUN_NOCTRL_ARM
    println("\nHYPOTHESIS: the control-group zeros (filter==2 regions, more frequent at far")
    println("distances) anchor the far end of the extensive margin at y=0 — a place")
    println("invert_T_ge CANNOT follow (control T stays 0). So the τ/T*(α) cancellation that")
    println("flattens the supplier-only reg_coef may break once controls are in, restoring an")
    println("interior α*. We test this by computing reg_coef WITH vs WITHOUT controls at each α.")
    println("NOTE: the reg_loss uses the on-disk empirical target; for the WITH-control column")
    println("to be an absolute fit, that target must itself be the with-control spec.")
end

# ── Header helpers (the granular columns only exist under GRANULAR) ───────────
blk_hdr  = join([@sprintf("%-11s", "wL:" * n) for n in gb_names], "")
gran_hdr = GRANULAR ? @sprintf("  %-6s %-6s %-9s", "clamp", "medN̂", "G0err") : ""

println("\n" * "-"^(96 + length(blk_hdr)))
println("(A) PROFILED regime  —  T = invert_T_ge(α)  (γ_ls pinned to data ∀α)")
println("    wL:<block> = r_b'W_bb r_b (diagonal contribution to the weighted criterion)")
println("-"^(96 + length(blk_hdr)))
@printf("  %-7s %-3s %-4s  %-10s %-10s %-10s %s%-10s%s  %-16s\n",
        "α", "cv", "it", "regSSE", "regSSE_nc", "TOTAL r'Wr", blk_hdr, "cross",
        gran_hdr, "reg_coef")
prof_reg_wc = Float64[]; prof_reg_wo = Float64[]; prof_gb = Float64[]
prof_blocks = Vector{Vector{Float64}}()
for a in α_grid
    res = invert_T_ge([a], Ω_L, Ω_s, A; T_init = copy(T_par_hat))
    θa  = assemble_theta(Ω_L, Ω_s, A, [a], res.T)
    sim = eval_moments(θa)
    gbl, diagc, unw, rc_wc = block_losses(sim)
    rl_wc = unw[1]
    rl_wo = RUN_NOCTRL_ARM ? reg_loss_of(reg_coef_without_control(θa)) : NaN
    push!(prof_reg_wc, rl_wc); push!(prof_reg_wo, rl_wo); push!(prof_gb, gbl)
    push!(prof_blocks, collect(diagc))

    gran_str = ""
    if GRANULAR
        g = granular_report(θa)
        gran_str = @sprintf("  %-6d %-6.1f %-9.3e",
                            count(!=(:none), g.clamped), median(g.N_hat),
                            sum(abs2, g.G0 .- g.G_target))
    end
    @printf("  %-7.4f %-3s %-4d  %-10.3e %-10.3e %-10.3e %s%-10.3e%s  %-16s\n",
            a, res.converged ? "y" : "N", res.iters, rl_wc, rl_wo, gbl,
            join([@sprintf("%-11.3e", d) for d in diagc], ""), gbl - sum(diagc),
            gran_str, rc_str(rc_wc))
end

println("\n" * "-"^(96 + length(blk_hdr)))
println("(B) FIXED-T regime  —  T = T̂ held constant (α moves reg_coef via τ=d^α)")
println("-"^(96 + length(blk_hdr)))
@printf("  %-7s  %-10s %-10s %-10s %s%-10s  %-16s\n",
        "α", "regSSE", "regSSE_nc", "TOTAL r'Wr", blk_hdr, "cross", "reg_coef")
fix_reg_wc = Float64[]; fix_reg_wo = Float64[]; fix_gb = Float64[]
for a in α_grid
    θa  = assemble_theta(Ω_L, Ω_s, A, [a], T_par_hat)
    sim = eval_moments(θa)
    gbl, diagc, unw, rc_wc = block_losses(sim)
    rl_wc = unw[1]
    rl_wo = RUN_NOCTRL_ARM ? reg_loss_of(reg_coef_without_control(θa)) : NaN
    push!(fix_reg_wc, rl_wc); push!(fix_reg_wo, rl_wo); push!(fix_gb, gbl)
    @printf("  %-7.4f  %-10.3e %-10.3e %-10.3e %s%-10.3e  %-16s\n",
            a, rl_wc, rl_wo, gbl,
            join([@sprintf("%-11.3e", d) for d in diagc], ""), gbl - sum(diagc),
            rc_str(rc_wc))
end

# ── (C) LINEAR MPL vs CLOGLOG, both size-controlled (profiled regime) ─────────
# Does the correct nonlinear link (cloglog, coef = αθ) restore the α-sensitivity that
# the linear probability model lacks? Both use include_control=false ⇒ supplier pairs
# only WITH the log-z size control, so it is an apples-to-apples link comparison on
# the same design. If the cloglog reg_coef moves with α while the MPL one is flat, the
# link — not the moment — was killing α's identification.
println("\n" * "-"^72)
println("(C) PROFILED regime — reg_coef: LINEAR MPL vs CLOGLOG (both size-controlled)")
println("-"^72)
@printf("  %-7s  %-24s  %-24s\n", "α", "MPL reg_coef (P(supplier))", "cloglog reg_coef (αθ)")
for a in α_grid
    res = invert_T_ge([a], Ω_L, Ω_s, A; T_init = copy(T_par_hat))
    θa  = assemble_theta(Ω_L, Ω_s, A, [a], res.T)
    net = solve_network(θa; u_draws = U_DRAWS, sample_weights = SAMPLE_WEIGHTS)
    rc_lin  = fast_weighted_regression(net.linkages_flat, net.z_flat, net.sample_weights;
                                       include_control = false)
    rc_clog = fast_cloglog_regression(net.linkages_flat, net.z_flat, net.sample_weights;
                                      include_control = false)
    @printf("  %-7.4f  %-24s  %-24s\n", a, rc_str(rc_lin), rc_str(rc_clog))
end

# ── Verdict ────────────────────────────────────────────────────────────────────
println("\n" * "="^72)
println("VERDICT")
println("="^72)
ia_prof_wc = argmin(prof_reg_wc)
ia_fix_wc  = argmin(fix_reg_wc)
@printf("  reg_coef-SSE minimiser α*:   PROFILED %.4f     FIXED-T %.4f\n",
        α_grid[ia_prof_wc], α_grid[ia_fix_wc])
if RUN_NOCTRL_ARM
    @printf("     (without control group)   PROFILED %.4f     FIXED-T %.4f\n",
            α_grid[argmin(prof_reg_wo)], α_grid[argmin(fix_reg_wo)])
end

# Which BLOCK owns the minimum of the weighted criterion under profiling? This is the
# question the granular configuration makes live: γ is pinned by the inversion at every
# α, so if the total criterion is minimised where reg_coef is WORST, the pull must be
# coming from another block (the count moment G0 being the candidate under GRANULAR).
ia_tot = argmin(prof_gb)
@printf("\n  PROFILED weighted criterion r'Wr minimised at α = %.4f  (value %.4e)\n",
        α_grid[ia_tot], prof_gb[ia_tot])
println("    per-block diagonal contribution at that α, and each block's own minimiser:")
for (b, nm) in enumerate(gb_names)
    series = [pb[b] for pb in prof_blocks]
    @printf("      %-10s  contrib %-11.3e   own α* = %.4f\n",
            nm, prof_blocks[ia_tot][b], α_grid[argmin(series)])
end
if !W_is_efficient
    println("\n  ⚠ W was IDENTITY (no matching W_step3 found), so the criterion above is NOT")
    println("    the one step 3 minimises. Re-run with the run folder and flags of the fit.")
end

boundary = α_grid[2]
if α_grid[ia_prof_wc] <= boundary && α_grid[ia_fix_wc] > boundary
    println("\n  ⇒ PROFILING ARTEFACT CONFIRMED on reg_coef: with T free (fixed-T regime) the")
    println("    reg_coef loss has an interior optimum, but once T = invert_T_ge(α) pins γ_ls")
    println("    the optimum collapses to the α→0 boundary — the T*(α) response cancels the")
    println("    τ channel that identifies α.")
elseif α_grid[ia_prof_wc] > boundary
    println("\n  ⇒ reg_coef retains an interior optimum even under profiling. If the TOTAL")
    println("    criterion still minimises at the boundary, another block is doing the pulling")
    println("    — read the per-block minimisers above.")
end
if GRANULAR
    println("\n  GRANULAR note: sectors CLAMPED at a variety bound cannot zero their own count")
    println("  residual through N_s, so that residual is absorbed by whatever else is free —")
    println("  here, α. Check the clamp column: if it is non-zero across the grid, α̂ is partly")
    println("  a patch for a binding N_LO/N_HI, not a trade elasticity.")
end
println("="^72)
