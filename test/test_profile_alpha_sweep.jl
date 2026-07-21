##### test_profile_alpha_sweep.jl #####
# WHY DOES SINKHORN-PROFILING PUSH α → 0?  (standalone, print-only diagnostic)
#
#     julia test/test_profile_alpha_sweep.jl aero 4 1
#     julia test/test_profile_alpha_sweep.jl auto 1
#
# Observed puzzle: without profiling (T searched freely) the estimator lands at
# α ≈ 0.30 with a GOOD reg_coef fit; with profile_T=true (T = invert_T_ge(α,…) so
# γ_ls is matched exactly for every α) it collapses to α ≈ 0.02 with a BAD reg_coef
# fit. This script isolates the cause by sweeping α on a grid and, at each α,
# contrasting the two regimes at a FIXED head (Ω^L, Ω^s, A) taken from θ̂:
#
#   (A) PROFILED   : T = invert_T_ge(α)  — the Sinkhorn map re-solves T so the
#                    analytical γ_ls stays pinned to the data for every α.
#   (B) FIXED-T    : T = T̂ (the T from θ̂), held constant as α varies — mimics the
#                    joint search, where α moves reg_coef through the DIRECT τ=d^α
#                    trade-cost channel with T free to absorb γ_ls separately.
#
# For each α and regime it prints the reg_coef loss (block 4), the combined β+γ
# loss (W_step3 if available, else identity), the reg_coef vector, and — for the
# profiled regime — invert_T_ge's convergence.
#
# CONTROL-GROUP EXTENSION: at each α it computes reg_coef BOTH with and without the
# control-group y=0 rows (filter==2 regions; fast_weighted_regression include_control
# true/false). invert_T_ge pins γ_ls from supplier pairs only, so T*(α) is identical
# in both variants — only the reg_coef regressand changes. This tests whether the
# far-distance control zeros (which invert_T_ge cannot fill: control T≡0) break the
# τ/T*(α) cancellation and restore an interior α* under profiling. The verdict reports
# the reg_coef-loss minimiser α* for each of {profiled, fixed-T} × {with, without}.
#
# EXPECTED (the mechanism): under FIXED-T the reg_coef loss has a clear interior
# minimum near α ≈ 0.3 (α steepens reg_coef via τ, identifying it). Under PROFILED
# the reg_coef loss is (near-)monotone toward α → 0, because raising α forces
# invert_T_ge to raise T for remote origins (to keep γ_ls fixed), and that T boost
# FLATTENS reg_coef — cancelling the very τ channel that identifies α. The exact-
# γ_ls hard constraint of profiling thus profiles out most of α's leverage on
# reg_coef, so the residual objective is minimised at the α → 0 boundary (where
# τ ≈ 1 makes the inversion a clean, well-conditioned matrix scaling). This is an
# identification artefact of profiling, NOT a data feature.
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
# profiled run to reporting_<industry>_profiled_pso; pass it explicitly to anchor
# on the profiled (α≈0.02) estimate instead.
optimizer_backend = :pso
K_sim    = 10000

input_folder  = "./baseline_$industry"
output_folder = length(ARGS) >= 4 && !isempty(strip(ARGS[4])) ? String(ARGS[4]) :
                "./reporting_$(industry)_$(optimizer_backend)"
mkpath(output_folder)
@printf("Anchoring θ̂ / W_step3 on: %s\n", output_folder)

include("../load_parameters.jl")
include("../profiling.jl")

@assert N_TAU == 1 "This sweep assumes the power-law single-α parametrization (N_TAU==1)."

println("\n" * "="^72)
println("PROFILING α-SWEEP: why does invert_T_ge push α → 0 ?")
@printf("  industry=%s  N_REG=%d  N_TAU=%d  |  ν=%.3g  λ=%.3g  θ=%.3g\n",
        industry, N_REG, N_TAU, nu, lambda, theta)
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
        end
    end
    T_red = vec(permutedims(T_rs_init))[T_MASK]
    Ω_L   = 0.5; Ω_s = ones(S); A = ones(R_downstream); β = [0.30]
    θ = vcat(Ω_L, Ω_s, A, β, T_red)
    @printf("Using DEFAULT θ (T=T_rs_init, Ω^L=0.5, Ω^s/A=1, α=0.30)\n")
    @assert length(θ) == expected
    return θ
end

θ0                    = load_theta()
Ω_L, Ω_s, A, α̂, T_vec = unpack_params(θ0)
T_hat                 = reshape(T_vec, S, R)          # ref-normalized per sector
T_hat_red             = vec(permutedims(T_hat))[T_MASK]
@printf("θ̂ head: Ω^L=%.4f  |Ω^s|=%d  |A|=%d  α̂=%.4f\n", Ω_L, S, R_downstream, α̂[1])

# ── Optional efficient weight matrix for the combined β+γ loss ────────────────
gb_indices = vcat(collect(BLOCK_RANGES[4]), collect(BLOCK_RANGES[5]))
W_gb = Matrix{Float64}(I, length(gb_indices), length(gb_indices))
let p = joinpath(output_folder, "step2", "W_step3.npy")
    if isfile(p)
        Wt = NPZ.npzread(p)
        if size(Wt, 1) == length(gb_indices)
            global W_gb = Wt
            println("Loaded W_step3 for the β+γ loss.")
        else
            @printf("W_step3 size %d ≠ %d β+γ moments — using identity.\n",
                    size(Wt, 1), length(gb_indices))
        end
    else
        println("No W_step3 on disk — β+γ loss uses identity weighting.")
    end
end

emp_vec   = vec(empirical_moments)
reg_range = BLOCK_RANGES[4]
gam_range = BLOCK_RANGES[5]
emp_reg   = emp_vec[reg_range]
emp_gam   = emp_vec[gam_range]
@printf("Empirical reg_coef: %s\n", string(round.(emp_reg, digits=4)))

"""Simulated moments at a full level θ, masked and block-sliced.
Note: full_SMM(θ; simulation=false) returns (loss, moments_tuple); we take the
tuple. (simulation=true returns the X_ls trade matrix, not moments — do not use.)"""
function eval_moments(θ)
    _, sim = full_SMM(θ; u_draws = U_DRAWS, sample_weights = SAMPLE_WEIGHTS)
    sim_vec = vcat([vec(sim[i]) for i in 1:5]...)[MOMENT_MASK]
    return sim_vec
end

function block_losses(sim_vec)
    err          = emp_vec .- sim_vec
    reg_loss     = sum(err[reg_range].^2)
    gam_loss     = sum(err[gam_range].^2)
    err_gb       = err[gb_indices]
    gb_loss      = (err_gb' * W_gb * err_gb)
    return reg_loss, gam_loss, gb_loss, sim_vec[reg_range]
end

# reg_coef WITHOUT the control-group y=0 rows (supplier pairs only), at the SAME
# network/draws. `include_control` toggles ONLY the appended filter==2 rows; the
# log-z size control is dropped in both variants. Profiling (invert_T_ge) pins γ_ls
# from the supplier pairs alone, so T*(α) is IDENTICAL with or without controls —
# the only thing that differs between the two variants is the reg_coef regressand.
function reg_coef_without_control(θ)
    net = solve_network(θ; u_draws = U_DRAWS, sample_weights = SAMPLE_WEIGHTS)
    fast_weighted_regression(net.linkages_flat, net.z_flat, net.sample_weights;
                             include_control = false)
end
reg_loss_of(rc) = sum((emp_reg .- rc).^2)
rc_str(v) = string(round.(v[1:min(end, 4)], digits = 3))

# ── α grid ────────────────────────────────────────────────────────────────────
α_grid = collect(range(0.02, 0.60, length = 15))

println("\nHYPOTHESIS: the control-group zeros (476 filter==2 regions, more frequent at")
println("far distances) anchor the far end of the extensive margin at y=0 — a place")
println("invert_T_ge CANNOT follow (control T stays 0). So the τ/T*(α) cancellation that")
println("flattens the supplier-only reg_coef may break once controls are in, restoring an")
println("interior α*. We test this by computing reg_coef WITH vs WITHOUT controls at each α.")
println("NOTE: the reg_loss uses the on-disk empirical target; for the WITH-control column")
println("to be an absolute fit, that target must itself be the with-control spec.")

println("\n" * "-"^72)
println("(A) PROFILED regime  —  T = invert_T_ge(α)  (γ_ls pinned to data ∀α)")
println("-"^72)
@printf("  %-5s %-4s %-4s  %-10s %-10s %-10s  %-16s %-16s\n",
        "α", "cv", "it", "regL(ctrl)", "regL(noctrl)", "βγ_loss",
        "reg_coef ctrl", "reg_coef noctrl")
prof_reg_wc = Float64[]; prof_reg_wo = Float64[]; prof_gb = Float64[]
for a in α_grid
    res = invert_T_ge([a], Ω_L, Ω_s, A; T_init = copy(T_hat))
    θa  = assemble_theta(Ω_L, Ω_s, A, [a], res.T)
    sim = eval_moments(θa)
    rl_wc, gl, gbl, rc_wc = block_losses(sim)
    rc_wo = reg_coef_without_control(θa)
    rl_wo = reg_loss_of(rc_wo)
    push!(prof_reg_wc, rl_wc); push!(prof_reg_wo, rl_wo); push!(prof_gb, gbl)
    @printf("  %-5.3f %-4s %-4d  %-10.3e %-10.3e %-10.3e  %-16s %-16s\n",
            a, res.converged ? "y" : "N", res.iters, rl_wc, rl_wo, gbl,
            rc_str(rc_wc), rc_str(rc_wo))
end

println("\n" * "-"^72)
println("(B) FIXED-T regime  —  T = T̂ held constant (α moves reg_coef via τ=d^α)")
println("-"^72)
@printf("  %-5s  %-10s %-10s %-10s  %-16s %-16s\n",
        "α", "regL(ctrl)", "regL(noctrl)", "βγ_loss", "reg_coef ctrl", "reg_coef noctrl")
fix_reg_wc = Float64[]; fix_reg_wo = Float64[]; fix_gb = Float64[]
for a in α_grid
    θa  = assemble_theta(Ω_L, Ω_s, A, [a], T_hat)
    sim = eval_moments(θa)
    rl_wc, gl, gbl, rc_wc = block_losses(sim)
    rc_wo = reg_coef_without_control(θa)
    rl_wo = reg_loss_of(rc_wo)
    push!(fix_reg_wc, rl_wc); push!(fix_reg_wo, rl_wo); push!(fix_gb, gbl)
    @printf("  %-5.3f  %-10.3e %-10.3e %-10.3e  %-16s %-16s\n",
            a, rl_wc, rl_wo, gbl, rc_str(rc_wc), rc_str(rc_wo))
end

# ── Verdict ────────────────────────────────────────────────────────────────────
println("\n" * "="^72)
println("VERDICT — does the control group restore an interior α under profiling?")
println("="^72)
ia_prof_wc = argmin(prof_reg_wc); ia_prof_wo = argmin(prof_reg_wo)
ia_fix_wc  = argmin(fix_reg_wc);  ia_fix_wo  = argmin(fix_reg_wo)
@printf("  reg_coef-loss minimiser α*:\n")
@printf("     PROFILED   with-control %.3f     without-control %.3f\n",
        α_grid[ia_prof_wc], α_grid[ia_prof_wo])
@printf("     FIXED-T    with-control %.3f     without-control %.3f\n",
        α_grid[ia_fix_wc], α_grid[ia_fix_wo])
println()
boundary = α_grid[2]                                   # "at the α→0 boundary" cutoff
prof_wc_interior = α_grid[ia_prof_wc] > boundary
prof_wo_boundary = α_grid[ia_prof_wo] <= boundary
if prof_wc_interior && prof_wo_boundary
    println("  ⇒ HYPOTHESIS SUPPORTED: without controls the profiled reg_coef optimum sits")
    println("    at the α→0 boundary, but WITH the control-group zeros it moves to an interior")
    println("    α*. The far-distance zeros (which invert_T_ge cannot fill, control T≡0) break")
    println("    the τ/T*(α) cancellation and re-identify α even under exact-γ_ls profiling.")
elseif !prof_wc_interior
    println("  ⇒ HYPOTHESIS NOT SUPPORTED: the profiled reg_coef optimum stays at/near α→0")
    println("    even WITH the control group. The control zeros are α-invariant, so they add")
    println("    a fixed anchor but do not re-create α-sensitivity strong enough to overcome")
    println("    the T*(α) flattening — profiling still profiles out α's identification.")
else
    println("  ⇒ MIXED: inspect the tables — compare the with- vs without-control reg_coef")
    println("    vectors and where each loss is minimised in the profiled regime.")
end
println("="^72)
