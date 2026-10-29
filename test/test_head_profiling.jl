##### test_head_profiling.jl #####
# Gate for PROFILE_HEAD (profiling.jl, `invert_T_head_ge`): the head (Ω^L, Ω^s, A)
# solved from blocks 1–3 at the EXPECTED price indices, jointly with T.
# STANDALONE + PRINT-ONLY; touches no production path. Run from the repo's parent dir:
#
#     julia SMM_Spatial_Comovement/test/test_head_profiling.jl aero [n_coef] [n_tau]
#
# INPUT_FOLDER overrides ./baseline_<industry>.
#
#   (1) ROUND-TRIP. Generate blocks 1–3 and γ in closed form at a planted
#       θ₀ = (Ω^L, Ω^s, A, α, κ, T), profile from a cold start, recover θ₀.
#   (2) EXACTNESS at the real targets: analytical blocks 1–3 and γ on target.
#   (3) UNIQUENESS: perturbed starts land on the same (T, head).
#   (4) STABILITY vs α: `invert_T_ge` at a FIXED head carries the ε channel; with A
#       profiled, sales are pinned and the channel is gone. Both on one α grid.
#   (5) MONTE-CARLO GAP: the SIMULATED blocks 1–3 at the profile against their
#       targets — what blocks 1–3 would add to the loss if they stayed in it.
#   (6) FAKE WORLD: `recalibrate_TA` on a world drawn at the profile returns a head
#       near the estimate, and a world equal to the analytical model returns it exactly.

using Distributed
@everywhere using Random, NPZ, DataFrames, CSV, Distributions, Plots, Statistics, Parquet,
                  QuasiMonteCarlo, LinearAlgebra, Printf, SpecialFunctions, Optim
const REPO = normpath(joinpath(@__DIR__, ".."))
@everywhere include(joinpath($REPO, "model_CP.jl"))
@everywhere include(joinpath($REPO, "profiling.jl"))
@everywhere include(joinpath($REPO, "optimizer.jl"))
@everywhere include(joinpath($REPO, "reporting.jl"))
@everywhere include(joinpath($REPO, "inference.jl"))

industry        = length(ARGS) >= 1 ? ARGS[1] : "aero"
n_coef          = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4
n_tau           = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1
n_rho_inference = 1000
relax_n_lo      = false
profile_head    = true
input_folder    = get(ENV, "INPUT_FOLDER", "./baseline_$industry")
output_folder   = mktempdir()
include(joinpath(REPO, "load_parameters.jl"))

println("\n", "="^78, "\nHEAD PROFILING GATE — S=$S  R_d=$R_downstream  N_TAU=$N_TAU")
println("ν=$nu  λ=$lambda  ε=$epsilon  θ=$theta\n", "="^78)

α0 = collect(TAU_PRIOR)
κ0 = KAPPA_PRIOR
pass = true
check(name, ok) = (global pass &= ok; @printf("  [%s] %s\n", ok ? "PASS" : "FAIL", name))
act = T_ACTIVE

# ── (1) Round-trip ──────────────────────────────────────────────────────────
println("\n(1) Round-trip from a planted θ₀")
rng = MersenneTwister(1)
ΩL0 = 0.4
Ωs0 = rand(rng, S) .+ 0.3; Ωs0 ./= sum(Ωs0)
A0  = exp.(0.3 .* randn(rng, R_downstream)); A0 ./= A0[1]
T0  = T_rs_init .* exp.(0.5 .* randn(rng, S, T_COL_DIM)) .* (T_rs_init .> 0)
for s in 1:S
    T0[s, :] ./= T0[s, T_REF_REGION[s]]
end
tau0  = build_tau(α0)
T0_ze = gather_T_to_ze(T0, κ0)
pr0   = compute_prices_analytical(ΩL0, Ωs0, A0, T0_ze, tau0)
ell0  = ΩL0 .* (regional_wages[DOWNSTREAM_REGIONS] ./ pr0.c_r) .^ (1 - lambda)
pi0   = pr0.Y_r ./ sum(pr0.Y_r)
s0    = Ωs0 .* (pr0.P_sr ./ reshape(pr0.P_r, 1, :)) .^ (1 - nu)
wI0   = pi0 .* (1 .- ell0)
tg0   = head_targets(L = sum(pi0 .* ell0), Os = (s0 * wI0) ./ sum(wI0), pi = pi0)
g0    = aggregate_gamma_to_T(gamma_ls_analytical(ΩL0, Ωs0, A0, T0_ze, tau0))
target0 = g0 ./ sum(g0; dims = 1)

rt = invert_T_head_ge(α0; kappa = κ0, target = target0, tg = tg0, tol = 1e-11, max_iter = 3000)
errT = maximum(abs.(log.(rt.T[act]) .- log.(T0[act])))
eΩs  = maximum(abs.(rt.Omega_s .- Ωs0)); eA = maximum(abs.(log.(rt.A) .- log.(A0)))
@printf("  iters=%d converged=%s  max|Δlog T|=%.1e |ΔΩL|=%.1e max|ΔΩs|=%.1e max|Δlog A|=%.1e\n",
        rt.iters, rt.converged, errT, abs(rt.Omega_L - ΩL0), eΩs, eA)
check("round-trip recovers (T, Ω^L, Ω^s, A) to 1e-6",
      rt.converged && max(errT, abs(rt.Omega_L - ΩL0), eΩs, eA) < 1e-6)

# ── (2) Exactness at the real targets ───────────────────────────────────────
println("\n(2) Exactness at the real targets (α = prior)")
res  = invert_T_head_ge(α0; kappa = κ0)
T_ze = gather_T_to_ze(res.T, κ0)
hr   = head_moment_residuals(res.Omega_L, res.Omega_s, res.A, T_ze, build_tau(α0))
gr   = aggregate_gamma_to_T(gamma_ls_analytical(res.Omega_L, res.Omega_s, res.A, T_ze, build_tau(α0)))
γerr = maximum(abs.((gr ./ sum(gr; dims = 1))[act'] .- EMP_GAMMA_T_TILDE[act']))
@printf("  iters=%d converged=%s | |block1|=%.1e max|block2|=%.1e max|block3|=%.1e max|γ̃|=%.1e\n",
        res.iters, res.converged, hr.labor, hr.industry, hr.pi_r, γerr)
@printf("  Ω^L=%.4f  Ω^s=%s\n", res.Omega_L, string(round.(res.Omega_s, digits = 4)))
check("blocks 1–3 and γ on target to 1e-7", res.converged && max(hr.labor, hr.industry, hr.pi_r, γerr) < 1e-7)

# ── (3) Uniqueness ──────────────────────────────────────────────────────────
println("\n(3) Uniqueness: 8 perturbed starts")
dmax = 0.0
for k in 1:8
    Ti = T_rs_init .* exp.(0.7 .* randn(rng, S, T_COL_DIM)) .* (T_rs_init .> 0)
    r  = invert_T_head_ge(α0; kappa = κ0, T_init = Ti, x_init = randn(rng, S), tol = 1e-11, max_iter = 3000)
    global dmax = max(dmax, maximum(abs.(log.(r.T[act]) .- log.(res.T[act]))),
                      maximum(abs.(log.(r.A) .- log.(res.A))), abs(r.Omega_L - res.Omega_L))
end
@printf("  max deviation across starts = %.1e\n", dmax)
check("unique profile (deviation < 1e-6)", dmax < 1e-6)

# ── (4) Stability vs α ──────────────────────────────────────────────────────
println("\n(4) Convergence vs α — fixed head (invert_T_ge, step-1 start) vs profiled head")
A_st  = emp_pi_r .^ (1 / abs(epsilon)) .* regional_wages[DOWNSTREAM_REGIONS]; A_st ./= A_st[1]
Os_st = agg_industry_share ./ sum(agg_industry_share)
@printf("  %6s | %-22s | %-22s\n", "α", "fixed  δ=0.3 / δ=0.9", "profiled δ=0.3 / δ=0.9")
for a in sort(unique(vcat(α0[1] .* [0.5, 1.0, 2.0, 4.0], [1.0, 1.5])))
    αv = fill(a, N_TAU)
    cell(r) = @sprintf("%s%4d", r.converged ? "✓" : "✗", r.iters)
    f = [cell(invert_T_ge(αv, agg_labor_share, Os_st, A_st; kappa = κ0, damping = δ)) for δ in (0.3, 0.9)]
    h = [cell(invert_T_head_ge(αv; kappa = κ0, damping = δ)) for δ in (0.3, 0.9)]
    @printf("  %6.3f | %-22s | %-22s\n", a, join(f, " / "), join(h, " / "))
end

# ── (5) Monte-Carlo gap ─────────────────────────────────────────────────────
println("\n(5) Simulated blocks 1–3 at the profile (N_rho = $N_rho)")
x0 = vcat(zeros(first(alpha_range()) - 1), α0, κ0, vec(permutedims(T_rs_init))[T_MASK])
p  = profiled_theta_full(x0)
check("profiled_theta_full converges and writes the profiled head",
      p.converged && p.theta[1] ≈ res.Omega_L && p.theta[2:(S + 1)] ≈ res.Omega_s)
_, m = full_SMM(p.theta)
tg = head_targets()
@printf("  |labor sim − target| = %.2e\n", abs(m[1][1] - tg.L))
@printf("  max|industry sim − target| = %.2e (scale %.2e)\n", maximum(abs.(vec(m[2]) .- tg.Os)), maximum(tg.Os))
@printf("  max|π_r sim − target| = %.2e (scale %.2e)\n", maximum(abs.(vec(m[3]) .- tg.pi)), maximum(tg.pi))

# ── (6) Fake worlds ─────────────────────────────────────────────────────────
println("\n(6) Fake world: recalibration of (T, Ω^L, Ω^s, A)")
ref = world_reference(p.theta)
pr  = compute_prices_analytical(ref.ΩL, ref.Ωs, ref.A, gather_T_to_ze(ref.T, κ0), build_tau(α0))
ellm = ref.ΩL .* (regional_wages[DOWNSTREAM_REGIONS] ./ pr.c_r) .^ (1 - lambda)
pim  = pr.Y_r ./ sum(pr.Y_r)
sm   = ref.Ωs .* (pr.P_sr ./ reshape(pr.P_r, 1, :)) .^ (1 - nu)
gm   = aggregate_gamma_to_T(gamma_ls_analytical(ref.ΩL, ref.Ωs, ref.A, gather_T_to_ze(ref.T, κ0), build_tau(α0)))
model_world = (gamma = gm ./ sum(gm; dims = 1), pi = pim, L = sum(pim .* ellm),
               Os = (sm * (pim .* (1 .- ellm))) ./ sum(pim .* (1 .- ellm)))
cm = recalibrate_TA(ref, α0, κ0, model_world)
dm = max(abs(cm.ΩL - ref.ΩL), maximum(abs.(cm.Ωs .- ref.Ωs)), maximum(abs.(log.(cm.A) .- log.(ref.A))),
         maximum(abs.(log.(cm.T[act]) .- log.(ref.T[act]))))
@printf("  world = analytical model: converged=%s  max deviation from θ̂ = %.1e\n", cm.converged, dm)
check("a world equal to the model returns the estimate (1e-6)", cm.converged && dm < 1e-6)
N_hat = granular_report(p.theta).N_hat
for b in 1:3
    w = draw_world(p.theta, N_hat, 31_000_000 + b)
    w.ref_empty && (println("  world $b: empty reference area, skipped"); continue)
    c = recalibrate_TA(ref, α0, κ0, w)
    @printf("  world %d: converged=%s  Ω^L %.4f (θ̂ %.4f)  max|ΔΩ^s| %.3f  max|Δlog A| %.3f\n",
            b, c.converged, c.ΩL, ref.ΩL, maximum(abs.(c.Ωs .- ref.Ωs)), maximum(abs.(log.(c.A) .- log.(ref.A))))
end

println("\n", pass ? "GATE: PASS" : "GATE: FAIL")
