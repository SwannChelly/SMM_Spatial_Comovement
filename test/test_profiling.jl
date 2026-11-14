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
T0_ls = gather_T_to_ze(T0, kappa0)
prices0   = compute_prices_analytical(chi_L0, chi_s0, A0, T0_ls, tau0; PhiF = foreign_phi(T0))
ell0  = chi_L0 .* (regional_wages[DOWNSTREAM_REGIONS] ./ prices0.c_r) .^ (1 - lambda)
pi0   = prices0.Y_r ./ sum(prices0.Y_r)
s_rs0    = chi_s0 .* (prices0.P_sr ./ reshape(prices0.P_r, 1, :)) .^ (1 - nu)
intermediate_bill0   = pi0 .* (1 .- ell0)
targets0   = profile_targets(labor_share = sum(pi0 .* ell0), sector_shares = (s_rs0 * intermediate_bill0) ./ sum(intermediate_bill0), sales_shares = pi0)
gamma0    = gamma_T_analytical(chi_L0, chi_s0, A0, T0, kappa0, tau0)
target0 = gamma0 ./ sum(gamma0; dims = 1)

round_trip = profile_ge(alpha0; kappa = kappa0, gamma_target = target0, targets = targets0, tol = 1e-11, max_iter = 3000)
T_error = maximum(abs.(log.(round_trip.T[act]) .- log.(T0[act])))
chi_s_error  = maximum(abs.(round_trip.chi_s .- chi_s0)); A_error = maximum(abs.(log.(round_trip.A) .- log.(A0)))
@printf("  iters=%d converged=%s  max|Δlog T|=%.1e |ΔχL|=%.1e max|Δχs|=%.1e max|Δlog A|=%.1e\n",
        round_trip.iterations, round_trip.converged, T_error, abs(round_trip.chi_L - chi_L0), chi_s_error, A_error)
check("round-trip recovers (T, χ^L, χ^s, A) to 1e-6",
      round_trip.converged && max(T_error, abs(round_trip.chi_L - chi_L0), chi_s_error, A_error) < 1e-6)

# ── (2) Exactness at the real targets ───────────────────────────────────────
println("\n(2) Exactness at the real targets (α = prior)")
profile  = profile_ge(alpha0; kappa = kappa0)
T_ls = gather_T_to_ze(profile.T, kappa0)
aggregate_residuals   = aggregate_moment_residuals(profile.chi_L, profile.chi_s, profile.A, T_ls, build_tau(alpha0); PhiF = foreign_phi(profile.T))
gamma_profile   = gamma_T_analytical(profile.chi_L, profile.chi_s, profile.A, profile.T, kappa0, build_tau(alpha0))
gamma_err = maximum(abs.((gamma_profile ./ sum(gamma_profile; dims = 1))[act'] .- EMP_GAMMA_T_TILDE[act']))
@printf("  iters=%d converged=%s | |block1|=%.1e max|block2|=%.1e max|block3|=%.1e max|γ̃|=%.1e\n",
        profile.iterations, profile.converged, aggregate_residuals.labor, aggregate_residuals.industry, aggregate_residuals.pi_r, gamma_err)
@printf("  χ^L=%.4f  χ^s=%s\n", profile.chi_L, string(round.(profile.chi_s, digits = 4)))
check("blocks 1–3 and γ on target to 1e-7", profile.converged && max(aggregate_residuals.labor, aggregate_residuals.industry, aggregate_residuals.pi_r, gamma_err) < 1e-7)

# ── (3) Uniqueness ──────────────────────────────────────────────────────────
println("\n(3) Uniqueness: 8 perturbed starts")
max_deviation = 0.0
for k in 1:8
    T_start = T_rs_init .* exp.(0.7 .* randn(rng, S, T_COL_DIM)) .* (T_rs_init .> 0)
    r  = profile_ge(alpha0; kappa = kappa0, T_init = T_start, aggregate_coordinates_init = randn(rng, S), tol = 1e-11, max_iter = 3000)
    global max_deviation = max(max_deviation, maximum(abs.(log.(r.T[act]) .- log.(profile.T[act]))),
                      maximum(abs.(log.(r.A) .- log.(profile.A))), abs(r.chi_L - profile.chi_L))
end
@printf("  max deviation across starts = %.1e\n", max_deviation)
check("unique profile (deviation < 1e-6)", max_deviation < 1e-6)

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
profiled_theta  = profile_theta(x0)
check("profile_theta converges and writes the profiled aggregates",
      profiled_theta.converged && profiled_theta.theta_vec[1] ≈ profile.chi_L && profiled_theta.theta_vec[2:(S + 1)] ≈ profile.chi_s)
_, m = full_SMM(profiled_theta.theta_vec)
targets = profile_targets()
@printf("  |labor sim − target| = %.2e\n", abs(m[1][1] - targets.labor_share))
@printf("  max|industry sim − target| = %.2e (scale %.2e)\n", maximum(abs.(vec(m[2]) .- targets.sector_shares)), maximum(targets.sector_shares))
@printf("  max|π_r sim − target| = %.2e (scale %.2e)\n", maximum(abs.(vec(m[3]) .- targets.sales_shares)), maximum(targets.sales_shares))

# ── (6) Fake worlds ─────────────────────────────────────────────────────────
println("\n(6) Fake world: profile of (T, χ^L, χ^s, A) on the world's moments")
ref = world_reference(profiled_theta.theta_vec)
pr  = compute_prices_analytical(ref.chi_L, ref.chi_s, ref.A, gather_T_to_ze(ref.T, kappa0), build_tau(alpha0); PhiF = foreign_phi(ref.T))
ell_model = ref.chi_L .* (regional_wages[DOWNSTREAM_REGIONS] ./ pr.c_r) .^ (1 - lambda)
pi_model  = pr.Y_r ./ sum(pr.Y_r)
s_rs_model   = ref.chi_s .* (pr.P_sr ./ reshape(pr.P_r, 1, :)) .^ (1 - nu)
gamma_model   = gamma_T_analytical(ref.chi_L, ref.chi_s, ref.A, ref.T, kappa0, build_tau(alpha0))
model_world = (gamma = gamma_model ./ sum(gamma_model; dims = 1), sales_shares = pi_model, labor_share = sum(pi_model .* ell_model),
               sector_shares = (s_rs_model * (pi_model .* (1 .- ell_model))) ./ sum(pi_model .* (1 .- ell_model)))
model_world_profile = profile_world(ref, alpha0, kappa0, model_world)
max_deviation_model = max(abs(model_world_profile.chi_L - ref.chi_L), maximum(abs.(model_world_profile.chi_s .- ref.chi_s)), maximum(abs.(log.(model_world_profile.A) .- log.(ref.A))),
         maximum(abs.(log.(model_world_profile.T[act]) .- log.(ref.T[act]))))
@printf("  world = analytical model: converged=%s  max deviation from θ̂ = %.1e\n", model_world_profile.converged, max_deviation_model)
check("a world equal to the model returns the estimate (1e-6)", model_world_profile.converged && max_deviation_model < 1e-6)
N_hat = granular_report(profiled_theta.theta_vec).N_hat
for b in 1:3
    w = draw_world(profiled_theta.theta_vec, N_hat, 31_000_000 + b)
    w.ref_empty && (println("  world $b: empty reference area, skipped"); continue)
    c = profile_world(ref, alpha0, kappa0, w)
    @printf("  world %d: converged=%s  χ^L %.4f (θ̂ %.4f)  max|Δχ^s| %.3f  max|Δlog A| %.3f\n",
            b, c.converged, c.chi_L, ref.chi_L, maximum(abs.(c.chi_s .- ref.chi_s)), maximum(abs.(log.(c.A) .- log.(ref.A))))
end

println("\n", pass ? "GATE: PASS" : "GATE: FAIL")
