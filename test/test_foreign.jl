##### test_foreign.jl #####
# Gate for the foreign competitor (Φ^F_s, the T column FOREIGN_COL; f̂_s, the γ block's
# foreign row). STANDALONE + PRINT-ONLY. Run from the directory holding the baseline:
#
#     python3 SMM_Spatial_Comovement/test/make_synthetic_baseline.py ./baseline_syn
#     julia SMM_Spatial_Comovement/test/test_foreign.jl syn [n_coef]
#
# INPUT_FOLDER overrides ./baseline_<industry>.
#
#   (1) LAYOUT: the γ target sums to one per sector over the T columns, its foreign row
#       is f̂_s = 1 − domestic_share, and every importing sector has a Φ^F parameter
#       paired with an f moment.
#   (2) ROUND-TRIP: γ and blocks 1–3 generated in closed form at a planted (T, Φ^F, χ, A)
#       are profiled back to it from the cold start T_rs_init.
#   (3) EXACTNESS at the data: at the profile the closed-form import share is f̂_s and
#       the area shares are γ̂.
#   (4) RICARDIAN CONSISTENCY: on 20 000 i.i.d. draws the simulated import share of
#       every (sector, buyer) is the closed-form win probability Φ^F_s/Φ_rs.
#   (5) SIMULATED MOMENTS at the profile on the estimation draws: γ sums to one with the
#       reference rebuilt, the import share is f̂ up to Monte-Carlo error, and the
#       French variety share q_s satisfies 1 − f̂ ≲ q_s ≤ 1 with N_HI = N^obs/q_s and
#       N_LO = N^obs/(R_d q_s).
#   (6) FAKE WORLDS: a world equal to the closed-form model returns (T̂, Φ̂^F, Â); worlds
#       drawn at the profile have γ summing to one and are profiled.

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
input_folder    = get(ENV, "INPUT_FOLDER", "./baseline_$industry")
output_folder   = mktempdir()
include(joinpath(REPO, "load_parameters.jl"))

pass = true
check(name, ok) = (global pass &= ok; @printf("  [%s] %s\n", ok ? "PASS" : "FAIL", name))
F   = FOREIGN_COL
act = T_ACTIVE
imp = findall(FOREIGN_ACTIVE)
println("\n", "="^78, "\nFOREIGN COMPETITOR GATE — S=$S  R_d=$R_downstream")
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

# ── (2) Round-trip ──────────────────────────────────────────────────────────
println("\n(2) Round-trip from a planted (T, Φ^F)")
rng = MersenneTwister(3)
alpha0, kappa0 = collect(TAU_PRIOR), KAPPA_PRIOR
chi_L0 = agg_labor_share
chi_s0 = agg_industry_share ./ sum(agg_industry_share)
A0  = exp.(0.2 .* randn(rng, R_downstream)); A0 ./= A0[1]
T0  = T_rs_init .* exp.(0.5 .* randn(rng, S, T_COL_DIM)) .* (T_rs_init .> 0)
for s in 1:S
    T0[s, :] ./= T0[s, T_REF_REGION[s]]
end
tau0 = build_tau(alpha0)
g0   = gamma_T_analytical(chi_L0, chi_s0, A0, T0, kappa0, tau0)
@printf("  planted import shares: %s (data %s)\n", string(round.(g0[F, :], digits = 4)),
        string(round.(FOREIGN_SHARE, digits = 4)))
# blocks 1–3 of the planted model, as `profile_ge` targets them (total purchases)
pr0  = compute_prices_analytical(chi_L0, chi_s0, A0, gather_T_to_ze(T0, kappa0), tau0; PhiF = foreign_phi(T0))
ell0 = chi_L0 .* (regional_wages[DOWNSTREAM_REGIONS] ./ pr0.c_r) .^ (1 - lambda)
pi0  = pr0.Y_r ./ sum(pr0.Y_r)
s0   = chi_s0 .* (pr0.P_sr ./ reshape(pr0.P_r, 1, :)) .^ (1 - nu)
wI0  = pi0 .* (1 .- ell0)
tg0  = profile_targets(labor_share = sum(pi0 .* ell0), sector_shares = (s0 * wI0) ./ sum(wI0), sales_shares = pi0)
rt = profile_ge(alpha0; kappa = kappa0, gamma_target = g0, targets = tg0, tol = 1e-12, max_iter = 5000)
eT = maximum(abs.(log.(rt.T[act]) .- log.(T0[act])))
eF = maximum(abs.(log.(rt.T[imp, F]) .- log.(T0[imp, F])))
@printf("  iters=%d converged=%s  max|Δlog T| = %.1e  max|Δlog Φ^F| = %.1e\n", rt.iterations, rt.converged, eT, eF)
eA = maximum(abs.(log.(rt.A) .- log.(A0))); e_chi = max(abs(rt.chi_L - chi_L0), maximum(abs.(rt.chi_s .- chi_s0)))
@printf("  max|Δlog A| = %.1e  max|Δχ| = %.1e\n", eA, e_chi)
check("round-trip recovers T, Φ^F, χ and A to 1e-7", rt.converged && max(eT, eF, eA, e_chi) < 1e-7)

# ── (3) Exactness at the data ───────────────────────────────────────────────
println("\n(3) Profile at the data (α, κ at their priors)")
x0 = vcat(agg_labor_share, agg_industry_share, ones(R_downstream), alpha0, kappa0,
          vec(permutedims(T_rs_init))[T_MASK])
pf = profile_theta(x0)
theta_vec  = pf.theta_vec
chi_L, chi_s, A, alpha, _ = unpack_params(theta_vec)
kappa = unpack_kappa(theta_vec)
Tp = unpack_T_area(theta_vec)
g_theta = gamma_T_analytical(chi_L, chi_s, A, Tp, kappa, build_tau(alpha))
ef = maximum(abs.(g_theta[F, imp] .- FOREIGN_SHARE[imp]))
eg = maximum(abs.(g_theta[act'] .- EMP_GAMMA_T[act']))
@printf("  converged=%s iters=%d | max|f − f̂| = %.1e  max|γ − γ̂| = %.1e\n", pf.converged, pf.iterations, ef, eg)
@printf("  Φ^F_s = %s ; Φ^F_s / Σ_l T_ls (wτ)^{−θ} at the median buyer = %s\n",
        string(round.(foreign_phi(Tp), sigdigits = 4)),
        string(round.([begin
            phi_sd = compute_Phi(gather_T_to_ze(Tp, kappa), build_tau(alpha))[s, :]
            foreign_phi(Tp)[s] / median(phi_sd)
        end for s in 1:S], sigdigits = 3)))
check("closed-form import share and area shares on target (1e-7)", pf.converged && max(ef, eg) < 1e-7)

# ── (4) Ricardian consistency ───────────────────────────────────────────────
println("\n(4) Simulated import share per (sector, buyer) vs Φ^F_s/Φ_rs (20 000 i.i.d. draws)")
Nbig = 20_000
u, w = generate_draws(Nbig, n_good, :mc; randomise = true, rng = MersenneTwister(77))
net  = solve_network(theta_vec; u_draws = u, sample_weights = w, return_firm_level = true)
D_sr = zeros(S, R_downstream)
pos  = Dict(r => i for (i, r) in enumerate(DOWNSTREAM_REGIONS))
for i in eachindex(net.firm_exp_val)
    D_sr[net.firm_exp_s[i], pos[net.firm_exp_r[i]]] += net.firm_exp_val[i]
end
f_sim = net.X_F_by_r ./ (net.X_F_by_r .+ D_sr)
phi_all = compute_Phi(gather_T_to_ze(Tp, kappa), build_tau(alpha); PhiF = foreign_phi(Tp))
f_cf  = [foreign_phi(Tp)[s] / phi_all[s, dr] for s in 1:S, dr in 1:R_downstream]
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
_, m = full_SMM(theta_vec; return_full = true)
gs = m.gamma_ls
check("simulated γ sums to one per sector over the T columns",
      maximum(abs.(vec(sum(gs; dims = 1)) .- 1)) < 1e-12)
check("industry shares (total purchases) sum to one", abs(sum(m.agg_industry_share) - 1) < 1e-12)
@printf("  import share: simulated %s | target %s\n", string(round.(gs[F, :], digits = 4)),
        string(round.(FOREIGN_SHARE, digits = 4)))
check("simulated import share within 0.05 of f̂ (Monte-Carlo gap of the realised prices)",
      maximum(abs.(gs[F, imp] .- FOREIGN_SHARE[imp])) < 0.05)
gr = granular_report(theta_vec)
@printf("  q_FR = %s ; 1 − f̂ = %s\n", string(round.(gr.q_french, digits = 3)),
        string(round.(1 .- FOREIGN_SHARE, digits = 3)))
@printf("  N_hi = %s (N^obs %s, N_HI_CAP %s) ; N̂ = %s, clamps %s\n", string(gr.N_hi), string(N_HI),
        string(N_HI_CAP), string(gr.N_hat), string(gr.clamped))
check("1 − f̂ − 0.05 ≤ q_s ≤ 1", all(gr.q_french .<= 1) && all(gr.q_french .>= 1 .- FOREIGN_SHARE .- 0.05))
@printf("  N_lo = %s (no-import bound N_LO %s)\n", string(gr.N_lo), string(N_LO))
check("N_lo = max(N_LO, ⌈N^obs/(R_d q)⌉), capped by N_hi",
      all(gr.N_lo[s] == min(max(N_LO[s], ceil(Int, N_HI[s] / (R_downstream * gr.q_french[s]) - 1e-9)), gr.N_hi[s])
          for s in imp) && all(gr.N_lo .>= N_LO .|| gr.N_lo .== gr.N_hi))
check("N_hi = min(⌈N^obs/q⌉, N_HI_CAP, m) and ≥ N^obs",
      all(gr.N_hi[s] == max(min(ceil(Int, N_HI[s] / gr.q_french[s] - 1e-9), N_HI_CAP[s], N_rho), N_HI[s])
          for s in imp) && all(gr.N_hi .>= N_HI))

# ── (6) Fake worlds ─────────────────────────────────────────────────────────
println("\n(6) Fake worlds")
ref = world_reference(theta_vec)
pr  = compute_prices_analytical(ref.chi_L, ref.chi_s, ref.A, gather_T_to_ze(ref.T, ref.kappa),
                                build_tau(ref.alpha); PhiF = foreign_phi(ref.T))
ell = ref.chi_L .* (regional_wages[DOWNSTREAM_REGIONS] ./ pr.c_r) .^ (1 - lambda)
pim = pr.Y_r ./ sum(pr.Y_r)
sm  = ref.chi_s .* (pr.P_sr ./ reshape(pr.P_r, 1, :)) .^ (1 - nu)
model_world = (gamma = gamma_T_analytical(ref.chi_L, ref.chi_s, ref.A, ref.T, ref.kappa, build_tau(ref.alpha)),
               pi = pim, L = sum(pim .* ell), Os = (sm * (pim .* (1 .- ell))) ./ sum(pim .* (1 .- ell)))
cm = profile_world(ref, ref.alpha, ref.kappa, model_world)
dm = max(maximum(abs.(log.(cm.T[act]) .- log.(ref.T[act]))), maximum(abs.(log.(cm.A) .- log.(ref.A))))
@printf("  world = closed-form model: converged=%s  max deviation (log T, log Φ^F, log A) = %.1e\n",
        cm.converged, dm)
check("a world equal to the model returns (T̂, Φ̂^F, Â) to 1e-6", cm.converged && dm < 1e-6)
for b in 1:3
    wd = draw_world(theta_vec, gr.N_hat, 51_000_000 + b)
    c  = wd.ref_empty ? nothing : profile_world(ref, ref.alpha, ref.kappa, wd)
    @printf("  world %d: Σγ − 1 = %.1e  f_b = %s  profile %s\n", b,
            maximum(abs.(vec(sum(wd.gamma; dims = 1)) .- 1)), string(round.(wd.gamma[F, :], digits = 3)),
            c === nothing ? "skipped (empty reference)" :
            @sprintf("converged=%s, Φ^F/Φ̂^F = %s", c.converged,
                     string(round.(c.T[imp, F] ./ ref.T[imp, F], digits = 3))))
    check("world $b: γ sums to one", maximum(abs.(vec(sum(wd.gamma; dims = 1)) .- 1)) < 1e-12)
end

# ── (7) Counterfactuals hold the import shares ─────────────────────────────
println("\n(7) Counterfactuals: every buyer's import share held at its estimate")
cf_est = incidence_stats(theta_vec; regime = :estimate)
for rg in (:equal_T, :no_trade_cost)
    cf = incidence_stats(theta_vec; regime = rg)
    dev = maximum(abs.(cf.import_share[imp, :] .- cf_est.import_share[imp, :]))
    @printf("  %-14s max |f_rs − f̂_rs| = %.1e ; mean dist %.1f km (estimate %.1f)\n", rg, dev,
            cf.mean_dist, cf_est.mean_dist)
    check("$rg holds f_rs at the estimate (1e-12)", dev < 1e-12)
end
check("the estimate's import share is Φ^F/Φ_rs",
      maximum(abs.(cf_est.import_share[imp, :] .- f_cf[imp, :])) < 1e-12)

println("\n", pass ? "GATE: PASS" : "GATE: FAIL")
