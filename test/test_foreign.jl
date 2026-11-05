##### test_foreign.jl #####
# Gate for the foreign competitor (Φ^F_s, the T column FOREIGN_COL; f̂_s, the γ block's
# foreign row). STANDALONE + PRINT-ONLY. Run from the directory holding the baseline:
#
#     python3 SMM_Spatial_Comovement/test/make_synthetic_baseline.py ./baseline_syn
#     julia SMM_Spatial_Comovement/test/test_foreign.jl syn [n_coef]
#
# INPUT_FOLDER overrides ./baseline_<industry>; PROFILE_HEAD=true runs the head profile.
#
#   (1) LAYOUT: the γ target sums to one per sector over the T columns, its foreign row
#       is f̂_s = 1 − domestic_share, and every importing sector has a Φ^F parameter
#       paired with an f moment.
#   (2) ROUND-TRIP: γ generated in closed form at a planted (T, Φ^F) is inverted back to
#       it from the cold start T_rs_init.
#   (3) EXACTNESS at the data: at the profile the closed-form import share is f̂_s and
#       the area shares are γ̂.
#   (4) RICARDIAN CONSISTENCY: on 20 000 i.i.d. draws the simulated import share of
#       every (sector, buyer) is the closed-form win probability Φ^F_s/Φ_rs.
#   (5) SIMULATED MOMENTS at the profile on the estimation draws: γ sums to one with the
#       reference rebuilt, the import share is f̂ up to Monte-Carlo error, and the
#       French variety share q_s satisfies 1 − f̂ ≲ q_s ≤ 1 with N_HI = N^obs/q_s.
#   (6) FAKE WORLDS: a world equal to the closed-form model returns (T̂, Φ̂^F, Â); worlds
#       drawn at the profile have γ summing to one and recalibrate.

using Distributed
@everywhere using Random, NPZ, DataFrames, CSV, Distributions, Statistics, QuasiMonteCarlo,
                  LinearAlgebra, Printf, SpecialFunctions, Optim
const REPO = normpath(joinpath(@__DIR__, ".."))
@everywhere include(joinpath($REPO, "model_CP.jl"))
@everywhere include(joinpath($REPO, "profiling.jl"))
@everywhere include(joinpath($REPO, "inference.jl"))

industry        = length(ARGS) >= 1 ? ARGS[1] : "syn"
n_coef          = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4
n_tau           = 1
n_rho_inference = 1000
relax_n_lo      = false
profile_head    = get(ENV, "PROFILE_HEAD", "false") == "true"
input_folder    = get(ENV, "INPUT_FOLDER", "./baseline_$industry")
output_folder   = mktempdir()
include(joinpath(REPO, "load_parameters.jl"))

pass = true
check(name, ok) = (global pass &= ok; @printf("  [%s] %s\n", ok ? "PASS" : "FAIL", name))
F   = FOREIGN_COL
act = T_ACTIVE
imp = findall(FOREIGN_ACTIVE)
println("\n", "="^78, "\nFOREIGN COMPETITOR GATE — S=$S  R_d=$R_downstream  PROFILE_HEAD=$PROFILE_HEAD")
println("f̂_s = ", round.(FOREIGN_SHARE, digits = 4), "\n", "="^78)
isempty(imp) && error("no sector imports in this baseline: nothing to test")

# ── (1) Layout ──────────────────────────────────────────────────────────────
println("\n(1) Layout")
colsum = vec(sum(EMP_GAMMA_T; dims = 1))
check("γ target sums to one per sector (max |Σ − 1| = $(round(maximum(abs.(colsum .- 1)), sigdigits = 2)))",
      maximum(abs.(colsum .- 1)) < 1e-10)
check("foreign row of the target = f̂_s", maximum(abs.(EMP_GAMMA_T[F, imp] .- FOREIGN_SHARE[imp])) < 1e-14)
check("one Φ^F parameter and one f moment per importing sector",
      count(startswith("PhiF["), PARAM_LABELS) == length(imp) &&
      count(startswith("f_foreign["), MOMENT_LABELS) == length(imp))
check("the reference area is domestic", all(T_REF_REGION .!= F))

# ── (2) Round-trip (fixed head) ─────────────────────────────────────────────
println("\n(2) Round-trip from a planted (T, Φ^F)")
rng = MersenneTwister(3)
α0, κ0 = collect(TAU_PRIOR), KAPPA_PRIOR
ΩL0 = agg_labor_share
Ωs0 = agg_industry_share ./ sum(agg_industry_share)
A0  = exp.(0.2 .* randn(rng, R_downstream)); A0 ./= A0[1]
T0  = T_rs_init .* exp.(0.5 .* randn(rng, S, T_COL_DIM)) .* (T_rs_init .> 0)
for s in 1:S
    T0[s, :] ./= T0[s, T_REF_REGION[s]]
end
tau0 = build_tau(α0)
g0   = gamma_T_analytical(ΩL0, Ωs0, A0, T0, κ0, tau0)
@printf("  planted import shares: %s (data %s)\n", string(round.(g0[F, :], digits = 4)),
        string(round.(FOREIGN_SHARE, digits = 4)))
rt = invert_T_ge(α0, ΩL0, Ωs0, A0; kappa = κ0, target = g0, tol = 1e-12, max_iter = 5000)
eT = maximum(abs.(log.(rt.T[act]) .- log.(T0[act])))
eF = maximum(abs.(log.(rt.T[imp, F]) .- log.(T0[imp, F])))
@printf("  iters=%d converged=%s  max|Δlog T| = %.1e  max|Δlog Φ^F| = %.1e\n", rt.iters, rt.converged, eT, eF)
check("round-trip recovers T and Φ^F to 1e-7", rt.converged && max(eT, eF) < 1e-7)

# ── (3) Exactness at the data ───────────────────────────────────────────────
println("\n(3) Profile at the data (α, κ at their priors)")
x0 = vcat(agg_labor_share, agg_industry_share, ones(R_downstream), α0, κ0,
          vec(permutedims(T_rs_init))[T_MASK])
pf = profiled_theta_full(x0)
θ  = pf.theta
ΩL, Ωs, A, α, κ = normalized_head(θ)
Tp = unpack_T_par(θ)
gθ = gamma_T_analytical(ΩL, Ωs, A, Tp, κ, build_tau(α))
ef = maximum(abs.(gθ[F, imp] .- FOREIGN_SHARE[imp]))
eg = maximum(abs.(gθ[act'] .- EMP_GAMMA_T[act']))
@printf("  converged=%s iters=%d | max|f − f̂| = %.1e  max|γ − γ̂| = %.1e\n", pf.converged, pf.iters, ef, eg)
@printf("  Φ^F_s = %s ; Φ^F_s / Σ_l T_ls (wτ)^{−θ} at the median buyer = %s\n",
        string(round.(foreign_phi(Tp), sigdigits = 4)),
        string(round.([begin
            Φ = compute_Phi(gather_T_to_ze(Tp, κ), build_tau(α))[s, :]
            foreign_phi(Tp)[s] / median(Φ)
        end for s in 1:S], sigdigits = 3)))
check("closed-form import share and area shares on target (1e-7)", pf.converged && max(ef, eg) < 1e-7)

# ── (4) Ricardian consistency ───────────────────────────────────────────────
println("\n(4) Simulated import share per (sector, buyer) vs Φ^F_s/Φ_rs (20 000 i.i.d. draws)")
Nbig = 20_000
u, w = generate_draws(Nbig, n_good, :mc; randomise = true, rng = MersenneTwister(77))
net  = solve_network(θ; u_draws = u, sample_weights = w, return_firm_level = true)
D_sr = zeros(S, R_downstream)
pos  = Dict(r => i for (i, r) in enumerate(DOWNSTREAM_REGIONS))
for i in eachindex(net.firm_exp_val)
    D_sr[net.firm_exp_s[i], pos[net.firm_exp_r[i]]] += net.firm_exp_val[i]
end
f_sim = net.X_F_by_r ./ (net.X_F_by_r .+ D_sr)
Φ_all = compute_Phi(gather_T_to_ze(Tp, κ), build_tau(α); PhiF = foreign_phi(Tp))
f_cf  = [foreign_phi(Tp)[s] / Φ_all[s, dr] for s in 1:S, dr in 1:R_downstream]
dev   = maximum(abs.(f_sim[imp, :] .- f_cf[imp, :]))
for s in imp
    @printf("  sector %d: f_rs closed form in [%.3f, %.3f], simulated in [%.3f, %.3f]\n", s,
            minimum(f_cf[s, :]), maximum(f_cf[s, :]), minimum(f_sim[s, :]), maximum(f_sim[s, :]))
end
@printf("  max |f_sim − f_cf| = %.2e  (binomial sd at N = %d: ≤ %.1e)\n", dev, Nbig, 0.5 / sqrt(Nbig))
check("simulated f_rs = Φ^F/Φ_rs within 5 binomial sd", dev < 5 * 0.5 / sqrt(Nbig))
nonimp = setdiff(1:S, imp)
check("no import where the sector does not import", isempty(nonimp) || all(net.X_F_by_r[nonimp, :] .== 0))

# ── (5) Simulated moments on the estimation draws ───────────────────────────
println("\n(5) Simulated moments at the profile (N_rho = $N_rho)")
_, m = full_SMM(θ; return_full = true)
gs = m.gamma_ls
check("simulated γ sums to one per sector over the T columns",
      maximum(abs.(vec(sum(gs; dims = 1)) .- 1)) < 1e-12)
check("industry shares (total purchases) sum to one", abs(sum(m.agg_industry_share) - 1) < 1e-12)
@printf("  import share: simulated %s | target %s\n", string(round.(gs[F, :], digits = 4)),
        string(round.(FOREIGN_SHARE, digits = 4)))
check("simulated import share within 0.05 of f̂ (Monte-Carlo gap of the realised prices)",
      maximum(abs.(gs[F, imp] .- FOREIGN_SHARE[imp])) < 0.05)
gr = granular_report(θ)
@printf("  q_FR = %s ; 1 − f̂ = %s\n", string(round.(gr.q_french, digits = 3)),
        string(round.(1 .- FOREIGN_SHARE, digits = 3)))
@printf("  N_hi = %s (N^obs %s, N_HI_CAP %s) ; N̂ = %s, clamps %s\n", string(gr.N_hi), string(N_HI),
        string(N_HI_CAP), string(gr.N_hat), string(gr.clamped))
check("1 − f̂ − 0.05 ≤ q_s ≤ 1", all(gr.q_french .<= 1) && all(gr.q_french .>= 1 .- FOREIGN_SHARE .- 0.05))
check("N_hi = min(⌈N^obs/q⌉, N_HI_CAP, m) and ≥ N^obs",
      all(gr.N_hi[s] == max(min(ceil(Int, N_HI[s] / gr.q_french[s] - 1e-9), N_HI_CAP[s], N_rho), N_HI[s])
          for s in imp) && all(gr.N_hi .>= N_HI))

# ── (6) Fake worlds ─────────────────────────────────────────────────────────
println("\n(6) Fake worlds")
ref = world_reference(θ)
pr  = compute_prices_analytical(ref.ΩL, ref.Ωs, ref.A, gather_T_to_ze(ref.T, ref.kappa),
                                build_tau(ref.alpha); PhiF = foreign_phi(ref.T))
ell = ref.ΩL .* (regional_wages[DOWNSTREAM_REGIONS] ./ pr.c_r) .^ (1 - lambda)
pim = pr.Y_r ./ sum(pr.Y_r)
sm  = ref.Ωs .* (pr.P_sr ./ reshape(pr.P_r, 1, :)) .^ (1 - nu)
model_world = (gamma = gamma_T_analytical(ref.ΩL, ref.Ωs, ref.A, ref.T, ref.kappa, build_tau(ref.alpha)),
               pi = pim, L = sum(pim .* ell), Os = (sm * (pim .* (1 .- ell))) ./ sum(pim .* (1 .- ell)))
cm = recalibrate_TA(ref, ref.alpha, ref.kappa, model_world)
dm = max(maximum(abs.(log.(cm.T[act]) .- log.(ref.T[act]))), maximum(abs.(log.(cm.A) .- log.(ref.A))))
@printf("  world = closed-form model: converged=%s  max deviation (log T, log Φ^F, log A) = %.1e\n",
        cm.converged, dm)
check("a world equal to the model returns (T̂, Φ̂^F, Â) to 1e-6", cm.converged && dm < 1e-6)
for b in 1:3
    wd = draw_world(θ, gr.N_hat, 51_000_000 + b)
    c  = wd.ref_empty ? nothing : recalibrate_TA(ref, ref.alpha, ref.kappa, wd)
    @printf("  world %d: Σγ − 1 = %.1e  f_b = %s  recalibration %s\n", b,
            maximum(abs.(vec(sum(wd.gamma; dims = 1)) .- 1)), string(round.(wd.gamma[F, :], digits = 3)),
            c === nothing ? "skipped (empty reference)" :
            @sprintf("converged=%s, Φ^F/Φ̂^F = %s", c.converged,
                     string(round.(c.T[imp, F] ./ ref.T[imp, F], digits = 3))))
    check("world $b: γ sums to one", maximum(abs.(vec(sum(wd.gamma; dims = 1)) .- 1)) < 1e-12)
end

println("\n", pass ? "GATE: PASS" : "GATE: FAIL")
