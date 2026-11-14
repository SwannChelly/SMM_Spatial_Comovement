##### test_profiling.jl #####
# Gate for the profile (profiling.jl, `profile_ge`): T and the aggregates (χ^L, χ^s, A)
# solved jointly from γ and blocks 1–3 at the EXPECTED price indices.
# STANDALONE + PRINT-ONLY; touches no production path. Run from the repo's parent dir:
#
#     julia SMM_Spatial_Comovement/test/test_profiling.jl aero [n_coef] [n_tau]
#
# INPUT_FOLDER overrides ./baseline_<industry>.
#
#   (1) ROUND-TRIP. Generate blocks 1–3 and γ in closed form at a planted
#       θ₀ = (χ^L, χ^s, A, α, κ, T), profile from a cold start, recover θ₀.
#   (2) EXACTNESS at the real targets: analytical blocks 1–3 and γ on target.
#   (3) UNIQUENESS: perturbed starts land on the same (T, aggregates).
#   (4) STABILITY vs α: convergence of the profile over an α grid, at two dampings.
#   (5) MONTE-CARLO GAP: the SIMULATED blocks 1–3 at the profile against their
#       targets — what blocks 1–3 would add to the loss if they stayed in it.
#   (6) FAKE WORLD: `profile_world` on a world drawn at the profile returns aggregates
#       near the estimate, and a world equal to the analytical model returns them exactly.

using Distributed
@everywhere using Random, NPZ, DataFrames, CSV, Distributions, Statistics,
                  QuasiMonteCarlo, LinearAlgebra, Printf, SpecialFunctions, Optim
const REPO = normpath(joinpath(@__DIR__, ".."))
@everywhere include(joinpath($REPO, "model_CP.jl"))
@everywhere include(joinpath($REPO, "profiling.jl"))
@everywhere include(joinpath($REPO, "inference.jl"))

industry        = length(ARGS) >= 1 ? ARGS[1] : "aero"
n_coef          = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4
n_tau           = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1
n_rho_inference = 1000
relax_n_lo      = false
input_folder    = get(ENV, "INPUT_FOLDER", "./baseline_$industry")
output_folder   = mktempdir()
include(joinpath(REPO, "load_parameters.jl"))

println("\n", "="^78, "\nPROFILING GATE — S=$S  R_d=$R_downstream  N_TAU=$N_TAU")
println("ν=$nu  λ=$lambda  ε=$epsilon  θ=$theta\n", "="^78)

alpha0 = collect(TAU_PRIOR)
kappa0 = KAPPA_PRIOR
pass = true
check(name, ok) = (global pass &= ok; @printf("  [%s] %s\n", ok ? "PASS" : "FAIL", name))
act = T_ACTIVE

# ── (1) Round-trip ──────────────────────────────────────────────────────────
println("\n(1) Round-trip from a planted θ₀")
rng = MersenneTwister(1)
chi_L0 = 0.4
chi_s0 = rand(rng, S) .+ 0.3; chi_s0 ./= sum(chi_s0)
A0  = exp.(0.3 .* randn(rng, R_downstream)); A0 ./= A0[1]
T0  = T_rs_init .* exp.(0.5 .* randn(rng, S, T_COL_DIM)) .* (T_rs_init .> 0)
for s in 1:S
    T0[s, :] ./= T0[s, T_REF_REGION[s]]
end
tau0  = build_tau(alpha0)
T0_ze = gather_T_to_ze(T0, kappa0)
pr0   = compute_prices_analytical(chi_L0, chi_s0, A0, T0_ze, tau0; PhiF = foreign_phi(T0))
ell0  = chi_L0 .* (regional_wages[DOWNSTREAM_REGIONS] ./ pr0.c_r) .^ (1 - lambda)
pi0   = pr0.Y_r ./ sum(pr0.Y_r)
s0    = chi_s0 .* (pr0.P_sr ./ reshape(pr0.P_r, 1, :)) .^ (1 - nu)
wI0   = pi0 .* (1 .- ell0)
tg0   = profile_targets(labor_share = sum(pi0 .* ell0), sector_shares = (s0 * wI0) ./ sum(wI0), sales_shares = pi0)
g0    = gamma_T_analytical(chi_L0, chi_s0, A0, T0, kappa0, tau0)
target0 = g0 ./ sum(g0; dims = 1)

rt = profile_ge(alpha0; kappa = kappa0, gamma_target = target0, targets = tg0, tol = 1e-11, max_iter = 3000)
errT = maximum(abs.(log.(rt.T[act]) .- log.(T0[act])))
e_chi_s  = maximum(abs.(rt.chi_s .- chi_s0)); eA = maximum(abs.(log.(rt.A) .- log.(A0)))
@printf("  iters=%d converged=%s  max|Δlog T|=%.1e |ΔχL|=%.1e max|Δχs|=%.1e max|Δlog A|=%.1e\n",
        rt.iterations, rt.converged, errT, abs(rt.chi_L - chi_L0), e_chi_s, eA)
check("round-trip recovers (T, χ^L, χ^s, A) to 1e-6",
      rt.converged && max(errT, abs(rt.chi_L - chi_L0), e_chi_s, eA) < 1e-6)

# ── (2) Exactness at the real targets ───────────────────────────────────────
println("\n(2) Exactness at the real targets (α = prior)")
res  = profile_ge(alpha0; kappa = kappa0)
T_ze = gather_T_to_ze(res.T, kappa0)
hr   = aggregate_moment_residuals(res.chi_L, res.chi_s, res.A, T_ze, build_tau(alpha0); PhiF = foreign_phi(res.T))
gr   = gamma_T_analytical(res.chi_L, res.chi_s, res.A, res.T, kappa0, build_tau(alpha0))
gamma_err = maximum(abs.((gr ./ sum(gr; dims = 1))[act'] .- EMP_GAMMA_T_TILDE[act']))
@printf("  iters=%d converged=%s | |block1|=%.1e max|block2|=%.1e max|block3|=%.1e max|γ̃|=%.1e\n",
        res.iterations, res.converged, hr.labor, hr.industry, hr.pi_r, gamma_err)
@printf("  χ^L=%.4f  χ^s=%s\n", res.chi_L, string(round.(res.chi_s, digits = 4)))
check("blocks 1–3 and γ on target to 1e-7", res.converged && max(hr.labor, hr.industry, hr.pi_r, gamma_err) < 1e-7)

# ── (3) Uniqueness ──────────────────────────────────────────────────────────
println("\n(3) Uniqueness: 8 perturbed starts")
dmax = 0.0
for k in 1:8
    Ti = T_rs_init .* exp.(0.7 .* randn(rng, S, T_COL_DIM)) .* (T_rs_init .> 0)
    r  = profile_ge(alpha0; kappa = kappa0, T_init = Ti, aggregate_coordinates_init = randn(rng, S), tol = 1e-11, max_iter = 3000)
    global dmax = max(dmax, maximum(abs.(log.(r.T[act]) .- log.(res.T[act]))),
                      maximum(abs.(log.(r.A) .- log.(res.A))), abs(r.chi_L - res.chi_L))
end
@printf("  max deviation across starts = %.1e\n", dmax)
check("unique profile (deviation < 1e-6)", dmax < 1e-6)

# ── (4) Stability vs α ──────────────────────────────────────────────────────
println("\n(4) Convergence of the profile vs α (✓/✗ and iterations)")
@printf("  %6s | %-22s\n", "α", "δ=0.3 / δ=0.9")
for a in sort(unique(vcat(alpha0[1] .* [0.5, 1.0, 2.0, 4.0], [1.0, 1.5])))
    alpha_v = fill(a, N_TAU)
    cell(r) = @sprintf("%s%4d", r.converged ? "✓" : "✗", r.iterations)
    h = [cell(profile_ge(alpha_v; kappa = kappa0, damping = damp)) for damp in (0.3, 0.9)]
    @printf("  %6.3f | %-22s\n", a, join(h, " / "))
end

# ── (5) Monte-Carlo gap ─────────────────────────────────────────────────────
println("\n(5) Simulated blocks 1–3 at the profile (N_rho = $N_rho)")
x0 = vcat(zeros(first(alpha_range()) - 1), alpha0, kappa0, vec(permutedims(T_rs_init))[T_MASK])
p  = profile_theta(x0)
check("profile_theta converges and writes the profiled aggregates",
      p.converged && p.theta_vec[1] ≈ res.chi_L && p.theta_vec[2:(S + 1)] ≈ res.chi_s)
_, m = full_SMM(p.theta_vec)
tg = profile_targets()
@printf("  |labor sim − target| = %.2e\n", abs(m[1][1] - tg.labor_share))
@printf("  max|industry sim − target| = %.2e (scale %.2e)\n", maximum(abs.(vec(m[2]) .- tg.sector_shares)), maximum(tg.sector_shares))
@printf("  max|π_r sim − target| = %.2e (scale %.2e)\n", maximum(abs.(vec(m[3]) .- tg.sales_shares)), maximum(tg.sales_shares))

# ── (6) Fake worlds ─────────────────────────────────────────────────────────
println("\n(6) Fake world: profile of (T, χ^L, χ^s, A) on the world's moments")
ref = world_reference(p.theta_vec)
pr  = compute_prices_analytical(ref.chi_L, ref.chi_s, ref.A, gather_T_to_ze(ref.T, kappa0), build_tau(alpha0); PhiF = foreign_phi(ref.T))
ellm = ref.chi_L .* (regional_wages[DOWNSTREAM_REGIONS] ./ pr.c_r) .^ (1 - lambda)
pim  = pr.Y_r ./ sum(pr.Y_r)
sm   = ref.chi_s .* (pr.P_sr ./ reshape(pr.P_r, 1, :)) .^ (1 - nu)
gm   = gamma_T_analytical(ref.chi_L, ref.chi_s, ref.A, ref.T, kappa0, build_tau(alpha0))
model_world = (gamma = gm ./ sum(gm; dims = 1), pi = pim, L = sum(pim .* ellm),
               Os = (sm * (pim .* (1 .- ellm))) ./ sum(pim .* (1 .- ellm)))
cm = profile_world(ref, alpha0, kappa0, model_world)
dm = max(abs(cm.chi_L - ref.chi_L), maximum(abs.(cm.chi_s .- ref.chi_s)), maximum(abs.(log.(cm.A) .- log.(ref.A))),
         maximum(abs.(log.(cm.T[act]) .- log.(ref.T[act]))))
@printf("  world = analytical model: converged=%s  max deviation from θ̂ = %.1e\n", cm.converged, dm)
check("a world equal to the model returns the estimate (1e-6)", cm.converged && dm < 1e-6)
N_hat = granular_report(p.theta_vec).N_hat
for b in 1:3
    w = draw_world(p.theta_vec, N_hat, 31_000_000 + b)
    w.ref_empty && (println("  world $b: empty reference area, skipped"); continue)
    c = profile_world(ref, alpha0, kappa0, w)
    @printf("  world %d: converged=%s  χ^L %.4f (θ̂ %.4f)  max|Δχ^s| %.3f  max|Δlog A| %.3f\n",
            b, c.converged, c.chi_L, ref.chi_L, maximum(abs.(c.chi_s .- ref.chi_s)), maximum(abs.(log.(c.A) .- log.(ref.A))))
end

println("\n", pass ? "GATE: PASS" : "GATE: FAIL")
