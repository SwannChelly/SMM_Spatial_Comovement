##### test_psi.jl #####
# Gate for the buyers' access to foreign suppliers ψ_r (profiled with T on the customs
# import shares). STANDALONE + PRINT-ONLY. Run from the directory holding the baseline:
#
#     python3 SMM_Spatial_Comovement/test/make_synthetic_baseline.py ./baseline_syn
#     julia SMM_Spatial_Comovement/test/test_psi.jl syn [n_coef]
#
# INPUT_FOLDER overrides ./baseline_<industry>.
#
#   (1) LAYOUT: the customs shares are read on the downstream zones, ψ is the last block of
#       the parameter vector, its reference is the buyer with the largest sales share.
#   (2) EXACTNESS at the data: at the profile, max_r |f_r − IS_r^target| and
#       max_s |f̃_s − f̂_s| are below 1e-7, the adding-up Σ_r X_r IS_r^target = Σ_s X_s f̂_s
#       holds at every pass, ψ_ref = 1; the per-buyer table.
#   (3) INVARIANCE: at given T, changing ψ leaves the domestic geography conditional on
#       buying in France, p_{l r s} / (1 − f_rs), unchanged (closed form), and every variety
#       bought in France under both ψ goes to the same cell (simulated, common draws).
#   (4) GAUGE: rescaling ψ by k and Φ^F by 1/k leaves the model unchanged.
#   (5) TEST DEVICE: with PSI_CALIBRATED[] = false the profile holds ψ ≡ 1.
#   (6) INFEASIBLE targets (a rescaled customs share ≥ 1) make the profile fail.
#   (7) FAKE WORLDS: the factor c of the world's targets is ≈ 1.

using Distributed
@everywhere using Random, NPZ, DataFrames, CSV, Distributions, Statistics, QuasiMonteCarlo,
                  LinearAlgebra, Printf, SpecialFunctions, Optim
const REPO = normpath(joinpath(@__DIR__, ".."))
@everywhere include(joinpath($REPO, "model_CP.jl"))
@everywhere include(joinpath($REPO, "profiling.jl"))
@everywhere include(joinpath($REPO, "inference.jl"))
include(joinpath(REPO, "reporting.jl"))

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
imp = findall(FOREIGN_ACTIVE)
println("\n", "="^78, "\nBUYERS' ACCESS TO FOREIGN SUPPLIERS ψ_r — S=$S  R_d=$R_downstream\n", "="^78)

# ── (1) Layout ──────────────────────────────────────────────────────────────
println("\n(1) Layout")
customs_raw = vec(NPZ.npzread(joinpath(input_folder, "import_share_customs_per_region.npy")))
check("customs shares = the file on the downstream zones", CUSTOMS_SHARE == customs_raw[N_downstream_per_region .> 0])
check("ψ reference = the largest sales share", PSI_REF_BUYER == argmax(emp_pi_r))
check("ψ is the last block of the parameter vector",
      last(psi_range()) == length(PARAM_LABELS) && all(startswith("psi["), PARAM_LABELS[psi_range()]))

# ── (2) Exactness at the data ───────────────────────────────────────────────
println("\n(2) Profile at the data (α, κ at their priors)")
alpha0, kappa0 = collect(TAU_PRIOR), KAPPA_PRIOR
profile = profile_ge(alpha0; kappa = kappa0)
x0 = vcat(agg_labor_share, agg_industry_share, ones(R_downstream), alpha0, kappa0,
          vec(permutedims(T_rs_init))[T_MASK], ones(R_downstream))
theta_vec = profile_theta(x0).theta_vec
fit = buyer_import_fit(theta_vec)
chi_L, chi_s, A, alpha, _, _, psi = unpack_params(theta_vec)
T_area = unpack_T_area(theta_vec)
gamma_closed_form = gamma_T_analytical(chi_L, chi_s, A, T_area, kappa0, build_tau(alpha); psi = psi)
buyer_error  = maximum(abs.(fit.model .- fit.target))
sector_error = maximum(abs.(gamma_closed_form[FOREIGN_COL, imp] .- FOREIGN_SHARE[imp]))
gamma_error  = maximum(abs.(gamma_closed_form[T_ACTIVE'] .- EMP_GAMMA_T[T_ACTIVE']))
@printf("  converged=%s iters=%d | max|f_r − IS_target| = %.1e  max|f̃_s − f̂_s| = %.1e  max|γ − γ̂| = %.1e\n",
        profile.converged, profile.iterations, buyer_error, sector_error, gamma_error)
@printf("  c = %.5f, largest relative adding-up gap over the passes = %.1e, ψ in [%.3f, %.3f]\n",
        profile.import_scale, profile.max_adding_up_gap, minimum(profile.psi), maximum(profile.psi))
check("profile converges", profile.converged)
check("buyers', sectors' import shares and γ on target (1e-7)", max(buyer_error, sector_error, gamma_error) < 1e-7)
check("adding-up Σ_r X_r IS_target = Σ_s X_s f̂_s at every pass (1e-12)", profile.max_adding_up_gap < 1e-12)
check("ψ_ref = 1", psi[PSI_REF_BUYER] == 1.0)
check("ψ moves (the customs shares are not those of ψ ≡ 1)", maximum(abs.(log.(psi))) > 1e-3)
println("\n  Per-buyer table ($industry):")
print(buyer_import_table(theta_vec; simulated_share = granular_report(theta_vec).buyer_import_share))

# ── (3) Invariance of the domestic geography ────────────────────────────────
println("\n(3) Invariance: at given T, ψ does not move p_{l r s} / (1 − f_rs)")
T_ls = gather_T_to_ze(T_area, kappa0)
tau = build_tau(alpha)
PhiF_s = foreign_phi(T_area)
function domestic_geography(psi_value)
    Phi = compute_Phi(T_ls, tau; PhiF = foreign_scale_by_buyer(PhiF_s, psi_value))
    geography = zeros(n_good, R_downstream)
    for g in 1:n_good, dr in 1:R_downstream
        s = GOOD_S[g]; l = GOOD_R[g]
        p_lrs = T_ls[s, l] * (W_RS_FLAT[g] * tau[l, dr])^(-theta) / Phi[s, dr]
        f_rs = foreign_scale_by_buyer(PhiF_s, psi_value)[s, dr] / Phi[s, dr]
        geography[g, dr] = p_lrs / (1 - f_rs)
    end
    return geography
end
psi_other = psi .* exp.(randn(MersenneTwister(9), R_downstream))
geography_gap = maximum(abs.(domestic_geography(psi) .- domestic_geography(psi_other)) ./ max.(domestic_geography(psi), 1e-300))
@printf("  max relative change of p_lrs / (1 − f_rs) under a random ψ: %.1e\n", geography_gap)
check("domestic geography independent of ψ (closed form, 1e-12)", geography_gap < 1e-12)
u_draws, sample_weights = generate_draws(4000, n_good, :mc; randomise = true, rng = MersenneTwister(71))
winners(theta_value) = (network = solve_network(theta_value; u_draws = u_draws, sample_weights = sample_weights, return_firm_level = true);
                        Dict((network.firm_exp_rho[i], network.firm_exp_s[i], network.firm_exp_r[i]) => network.firm_exp_g[i]
                             for i in eachindex(network.firm_exp_rho)))
winners_psi, winners_other = winners(theta_vec), winners(with_psi(theta_vec, psi_other))
common = intersect(keys(winners_psi), keys(winners_other))
same_winner = all(winners_psi[k] == winners_other[k] for k in common)
@printf("  simulated: %d (variety, sector, buyer) bought in France under both ψ, same cell in all: %s\n",
        length(common), same_winner)
check("same domestic winner for every variety bought in France under both ψ", same_winner && !isempty(common))

# ── (4) Gauge ───────────────────────────────────────────────────────────────
println("\n(4) Gauge: (ψ k, Φ^F / k) is the same model")
T_area_rescaled = copy(T_area); T_area_rescaled[:, FOREIGN_COL] ./= 2.5
theta_rescaled = with_psi(with_T(theta_vec, T_area_rescaled), 2.5 .* psi)
moments_base = moments_to_vec(full_SMM(theta_vec)[2]); moments_rescaled = moments_to_vec(full_SMM(theta_rescaled)[2])
gauge_gap = maximum(abs.(moments_base .- moments_rescaled))
@printf("  stored ψ × 2.5 and Φ^F / 2.5: max |Δ moments| = %.1e\n", gauge_gap)
check("moments unchanged under the gauge (1e-12)", gauge_gap < 1e-12)

# ── (5) Test device: ψ held at one ──────────────────────────────────────────
println("\n(5) PSI_CALIBRATED[] = false")
PSI_CALIBRATED[] = false
profile_fixed = profile_ge(alpha0; kappa = kappa0)
PSI_CALIBRATED[] = true
check("ψ ≡ 1 and the profile converges", profile_fixed.converged && all(profile_fixed.psi .== 1))

# ── (6) Infeasible targets ──────────────────────────────────────────────────
println("\n(6) A rescaled customs share ≥ 1")
customs_extreme = copy(CUSTOMS_SHARE); customs_extreme[1] *= 1e4
profile_infeasible = profile_ge(alpha0; kappa = kappa0, customs_share = customs_extreme)
check("the profile is rejected", !profile_infeasible.converged)

# ── (7) Fake worlds ─────────────────────────────────────────────────────────
println("\n(7) Fake worlds: the factor c of the world's buyer targets")
ref = world_reference(theta_vec)
N_hat = granular_report(theta_vec).N_hat
scales = Float64[]
for b in 1:6
    world = draw_world(theta_vec, N_hat, 61_000_000 + b)
    (world.ref_empty || !all(>(0), world.buyer_import_share)) && continue
    world_profile = profile_world(ref, ref.alpha, ref.kappa, world)
    world_profile.converged || continue
    push!(scales, world_profile.import_scale)
    @printf("  world %d: c = %.4f, ψ_b/ψ̂ in [%.3f, %.3f]\n", b, world_profile.import_scale,
            minimum(world_profile.psi ./ ref.psi), maximum(world_profile.psi ./ ref.psi))
end
check("c within 10% of one in every usable world", !isempty(scales) && all(abs.(scales .- 1) .< 0.1))

println("\n", pass ? "GATE: PASS" : "GATE: FAIL")
