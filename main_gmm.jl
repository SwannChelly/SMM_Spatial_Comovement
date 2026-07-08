##### Closed-form GMM estimation — main_gmm.jl #####
# Drop-in replacement for main.jl using analytical EK moments.
# Three-step GMM:
#   Step 1: identity-weighted PSO with analytical moments → θ̂_1
#   Step 2: W_eff = Σ_data^{-1} (Σ_sim = 0 by construction) + θ̂_1 inference
#   Step 3: efficient-weighted PSO (β+T only), warm-started at θ̂_1 → θ̂_2
#   Step 4: Jacobian at θ̂_2 + GMM inference (separable from Step 3)
#
# Usage:
#   julia main_gmm.jl aero 4            # N_REG=4, N_TAU=4, n_quad=200
#   julia main_gmm.jl aero 4 1          # N_REG=4, N_TAU=1 (over-identified)
#   julia main_gmm.jl aero 4 1 200 500  # N_REG=4, N_TAU=1, n_quad=500

using Distributed
using Dates

available = 50
println("Using $available workers")
addprocs(max(available - 1, 0))
using Random
seed = 1234
Random.seed!(seed)
@everywhere using Random
@everywhere Random.seed!($seed)

@everywhere using NPZ
@everywhere using QuasiMonteCarlo
@everywhere using StatsPlots
@everywhere using DataFrames
@everywhere using Distributions
@everywhere using Plots
@everywhere using CSV
@everywhere using Optim
@everywhere using Statistics
@everywhere using HaltonSequences
@everywhere using ProgressMeter
@everywhere using SharedArrays
@everywhere using Parquet
@everywhere using SpecialFunctions
@everywhere using FastGaussQuadrature
using LinearAlgebra
using Statistics, Printf
using StatsBase

@everywhere include("model_CP.jl")
@everywhere include("model_analytical.jl")
@everywhere include("tools.jl")
@everywhere include("pso_integration.jl")      # PSO backend
@everywhere include("cmaes_integration.jl")    # CMA-ES backend
@everywhere include("optimizer.jl")            # backend-neutral hub (defines run_optimization / run_pso_optimization)

############## Parse arguments ##############

industry = length(ARGS) >= 1 ? ARGS[1] : "auto"
n_coef   = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4
n_tau    = length(ARGS) >= 3 && !isempty(strip(ARGS[3])) ? parse(Int, ARGS[3]) : n_coef
n_quad   = length(ARGS) >= 4 && !isempty(strip(ARGS[4])) ? parse(Int, ARGS[4]) : 200
# Draw method for the simulated price solver (:qmc default, :mc, :is). Forwarded
# to load_parameters.jl (U_DRAWS) and every draw-generation site.
draw_method = length(ARGS) >= 5 && !isempty(strip(ARGS[5])) ? Symbol(strip(ARGS[5])) : :qmc
@assert draw_method in (:qmc, :mc, :is, :sobol) "draw method must be qmc|mc|is|sobol, got :$draw_method"
K = 10   # PSO loops

run_step1 = true
run_step2 = true
run_step3 = true
run_step4 = true

if !(n_coef in [1, 4, 5])
    error("n_coef must be 1, 4 or 5, got: $n_coef")
end
if !(n_tau in [1, 4, 5])
    error("n_tau must be 1, 4 or 5, got: $n_tau")
end

println("GMM mode | Industry: $industry | n_coef (N_REG): $n_coef | n_tau (N_TAU): $n_tau | n_quad: $n_quad | draws: :$draw_method")

input_folder  = "./baseline_$industry"
output_folder = "./reporting_gmm_$industry"
mkpath(output_folder)

############## Load and distribute constants ##############
include("load_parameters.jl")
NPZ.npzwrite(joinpath(output_folder, "n_reg_coef.npy"), n_coef)

############## Determine step to run ##############

step1_folder = joinpath(output_folder, "step1")
step2_W_path = joinpath(output_folder, "step2", "W_eff.npy")

############## STEP 1 — Identity-weighted GMM ##############

mkpath(step1_folder)

if run_step1
    println("\n" * "="^70)
    println("STEP 1: Identity-weighted PSO (analytical GMM)")
    println("="^70)

    theta_hat_1, _ = run_pso_optimization(;
        weight_matrix            = nothing,
        skip_initial_beta_search = false,
        warm_start_params        = nothing,
        output_subfolder         = "step1",
        max_loop                 = K,
        analytical               = true,
        n_quad                   = n_quad
    )

    NPZ.npzwrite(joinpath(output_folder, "step1", "theta_hat_1.npy"), theta_hat_1)
    # Compare the fitted T distribution to the very initial one (α in the name).
    plot_T_vs_initial(theta_hat_1, joinpath(output_folder, "step1"); label = "step1")
    println("Step 1 complete. θ̂_1 saved.")
else
    println("Step 1 skipped (resume).")
end

############## Load θ̂_1 ##############

step1_last = find_last_stage_folder(joinpath(output_folder, "step1"))
theta_hat_1 = NPZ.npzread(joinpath(step1_last, "best_params.npy"))
ndims(theta_hat_1) > 1 && (theta_hat_1 = theta_hat_1[:, 1])
println("θ̂_1 loaded from: $step1_last")
test_analytical_vs_simulated(theta_hat_1; N_rho_test=N_rho, n_quad=200)

############## STEP 2 — Efficient weight matrix (W = Σ_data^{-1}) + θ̂_1 inference ##############
# In GMM mode, Σ_sim = 0 by construction → W_eff = Σ_data^{-1} directly.
# No simulation replication needed. Also runs θ̂_1 inference for comparability with main.jl.

if run_step2
    println("\n" * "="^70)
    println("STEP 2: Building efficient weight matrix W = Σ_data^{-1}")
    println("="^70)

    step2_folder = joinpath(output_folder, "step2")
    mkpath(step2_folder)

    # Load empirical covariance matrices.
    # File selection keyed on N_REG (the moment count, not τ-parameter count N_TAU).
    sigma_file_gmm = N_REG == 1 ? "Sigma_beta_gamma_1.npy" : "Sigma_beta_gamma.npy"
    Sigma_data_full = try
        NPZ.npzread(joinpath(input_folder, sigma_file_gmm))
    catch
        # Fall back to block-diagonal from w_gamma + w_beta (β-then-γ ordering)
        w_beta  = NPZ.npzread(joinpath(input_folder, "w_beta.npy"))
        w_gamma = NPZ.npzread(joinpath(input_folder, "w_gamma.npy"))
        n_b   = size(w_beta, 1)
        n_gam = size(w_gamma, 1)
        S_full = zeros(n_b + n_gam, n_b + n_gam)
        S_full[1:n_b, 1:n_b]             = w_beta
        S_full[(n_b+1):end, (n_b+1):end] = w_gamma
        S_full
    end

    # β-then-γ ordering (invariant: BLOCK_RANGES[4] then BLOCK_RANGES[5])
    gb_indices = vcat(collect(BLOCK_RANGES[4]), collect(BLOCK_RANGES[5]))
    N_gb = length(gb_indices)

    # Reconcile the on-disk Σ with the (possibly gamma-thresholded) active set —
    # SHARED with the SMM (tools.jl build_step3_weight_matrix). A naive
    # Sigma_data_full[1:N_gb, 1:N_gb] slice silently mixes surviving and pruned
    # (s,r) γ moments whenever the file is pre-threshold (size n_gb_old > N_gb),
    # misaligning W_eff against the threshold-aware Jacobian/moment vectors.
    Sigma_data_gb = reconcile_sigma_data(Sigma_data_full, input_folder)

    # Optional regularization if near-singular
    F_gb = eigen(Symmetric(Sigma_data_gb))
    λ_gb = F_gb.values
    if minimum(λ_gb) < 0 || λ_gb[end] / max(λ_gb[1], 1e-300) > 1e10
        @warn "Σ_data ill-conditioned (cond=$(round(λ_gb[end]/max(λ_gb[1],1e-300), sigdigits=3))). Adding floor."
        floor_val = λ_gb[end] / 1e8
        Sigma_data_gb = F_gb.vectors * Diagonal(max.(λ_gb, floor_val)) * F_gb.vectors'
    end

    W_eff = Matrix(inv(Symmetric(Sigma_data_gb)))

    NPZ.npzwrite(joinpath(step2_folder, "Sigma_data_gb.npy"), Sigma_data_gb)
    NPZ.npzwrite(joinpath(step2_folder, "W_eff.npy"), W_eff)
    println("  W_eff ($(N_gb)×$(N_gb)) saved to step2/.")

    # Jacobian at θ̂_1 — exact closed-form via forward-mode AD (no FD step).
    println("\nComputing Jacobian at θ̂_1 (analytical, exact AD)...")
    J1, J1_elast, J1_sd, J1_elast_sd = compute_jacobian(
        theta_hat_1;
        param_indices = jacobian_param_indices,
        output_folder = output_folder,
        output_subdir = "step2",
        filename      = "jacobian_all.npy",
        K             = 1,
        step_rel      = 1e-4,
        step_abs      = 1e-9,
        base_seed     = 0,
        analytical    = true,
        analytical_ad = true,
        ad_validate   = true,
        n_quad        = n_quad
    )

    # Inference at θ̂_1 (GMM: Ω = Σ_data, K_sim = 0)
    _, sim_moments_1 = full_SMM(theta_hat_1; analytical=true, n_quad=n_quad)
    sim_vec_1 = vcat([vec(sim_moments_1[i]) for i in 1:5]...)[MOMENT_MASK]
    emp_vec   = vec(empirical_moments)

    n_reg_loc = length(BLOCK_RANGES[4]); n_gam_loc = length(BLOCK_RANGES[5])
    gb_block_ranges = (1:n_reg_loc, (n_reg_loc + 1):(n_reg_loc + n_gam_loc))
    gb_block_names  = ("reg_coef", "gamma_ls")

    # Restrict Jacobian columns to β+T (the only params β+γ moments identify)
    beta_T_start = 1 + S + R_downstream + 1                       # first β raw index
    gb_cols      = findall(i -> i >= beta_T_start, jacobian_param_indices)
    gb_param_idx = jacobian_param_indices[gb_cols]

    n_beta_labels_s2 = count(l -> startswith(l, "alpha") || startswith(l, "beta"), PARAM_LABELS[gb_cols])
    @assert n_beta_labels_s2 == N_TAU "Step-2 inference: expected $N_TAU β/α labels in gb_cols, found $n_beta_labels_s2"

    # Correctness check: Σ_data leading β-block must be N_REG × N_REG (moments, not params).
    Omega_gmm = Sigma_data_gb
    @assert size(Omega_gmm, 1) == N_REG + length(BLOCK_RANGES[5]) "Omega_gmm size $(size(Omega_gmm,1)) != N_REG+n_γ=$(N_REG+length(BLOCK_RANGES[5]))"

    J1_gb      = J1[gb_indices, gb_cols]
    sim_vec_gb = sim_vec_1[gb_indices]
    emp_vec_gb = emp_vec[gb_indices]

    # Step-1 weight matrix: θ̂_1 was estimated with the IDENTITY-weighted criterion
    # (run_pso_optimization called with weight_matrix=nothing → Weight_matrix_custom),
    # NOT with W_eff = Σ_data^{-1}. Inference at θ̂_1 must use that same W, otherwise
    # Var_eff = (G'W G)^{-1} describes the efficient estimator θ̂_1 is not.
    # The sandwich form (G'WG)^{-1} G'WΩWG (G'WG)^{-1} is the only valid variance here.
    W_step1 = Matrix(Weight_matrix_custom[gb_indices, gb_indices])

    # T-identification eigen-screen (diagnostic, print-only) at θ̂_1
    screen_T_identification(theta_hat_1;
        J            = J1_gb,
        W            = W_step1,
        param_labels = PARAM_LABELS[gb_cols],
        label        = "GMM step2 θ̂_1")

    compute_smm_inference(
        theta_hat_1, J1_gb, W_step1, Omega_gmm;
        param_indices         = gb_param_idx,
        empirical_moments_vec = emp_vec_gb,
        simulated_moments_vec = sim_vec_gb,
        output_folder         = joinpath(output_folder, "step2"),
        industry              = industry,
        K_sim                 = 0,
        block_ranges          = gb_block_ranges,
        block_names           = gb_block_names,
        gamma_ref_map         = GAMMA_REF_MAP,
        param_labels          = PARAM_LABELS[gb_cols],
        moment_labels         = MOMENT_LABELS[gb_indices]
    )

    println("Step 2 complete. W_eff, Jacobian, and θ̂_1 inference saved.")
else
    println("Step 2 skipped (resume). Loading W_eff...")
    W_eff = NPZ.npzread(step2_W_path)

    # Recompute gb_indices / Sigma_data_gb for use in later steps
    gb_indices = vcat(collect(BLOCK_RANGES[4]), collect(BLOCK_RANGES[5]))
    N_gb = length(gb_indices)
    Sigma_data_gb = NPZ.npzread(joinpath(output_folder, "step2", "Sigma_data_gb.npy"))
end

############## STEP 3 — Efficient-weighted GMM (β+T optimisation only) ##############

step3_folder = joinpath(output_folder, "step3")
mkpath(step3_folder)

if run_step3
    println("\n" * "="^70)
    println("STEP 3: Efficient-weighted PSO (analytical GMM)")
    println("="^70)

    theta_hat_2, _ = run_pso_optimization(;
        weight_matrix            = W_eff,
        skip_initial_beta_search = true,
        warm_start_params        = theta_hat_1,
        output_subfolder         = "step3",
        max_loop                 = K,
        gamma_beta_only          = true,
        moments_loss_gamma_beta  = true,
        analytical               = true,
        n_quad                   = n_quad
    )

    NPZ.npzwrite(joinpath(output_folder, "step3", "theta_hat_2.npy"), theta_hat_2)
    # Compare the fitted T distribution to the very initial one (α in the name).
    plot_T_vs_initial(theta_hat_2, joinpath(output_folder, "step3"); label = "step3")
    println("Step 3 complete. θ̂_2 saved.")
else
    println("Step 3 skipped (resume).")
end

############## STEP 4 — Jacobian at θ̂_2 + GMM inference ##############

if run_step4
    println("\n" * "="^70)
    println("STEP 4: Jacobian at θ̂_2 + GMM inference")
    println("="^70)

    theta_hat_2 = NPZ.npzread(joinpath(output_folder, "step3", "theta_hat_2.npy"))
    ndims(theta_hat_2) > 1 && (theta_hat_2 = theta_hat_2[:, 1])
    test_analytical_vs_simulated(theta_hat_2; N_rho_test=N_rho, n_quad=200)

    # Full-column Jacobian at θ̂_2 — exact closed-form via forward-mode AD.
    println("\nComputing Jacobian at θ̂_2 (analytical, exact AD)...")
    J2, J2_elast, J2_sd, J2_elast_sd = compute_jacobian(
        theta_hat_2;
        param_indices = jacobian_param_indices,
        output_folder = output_folder,
        output_subdir = "step3",
        filename      = "jacobian_all_step3.npy",
        K             = 1,
        step_rel      = 1e-4,
        step_abs      = 1e-9,
        base_seed     = 1_000_000,
        analytical    = true,
        analytical_ad = true,
        ad_validate   = true,
        n_quad        = n_quad
    )

    # Rank diagnostic + per-block noise report (mirroring main.jl Step 4)
    sv_J2   = svdvals(J2)
    rank_J2 = count(sv_J2 .> sv_J2[1] * 1e-8)
    println("  Rank of J2: $rank_J2 / $(size(J2, 2))")
    println("  Per-block max/mean |J2_elast| and noise (J2_elast_sd / |J2_elast|):")
    for (k, name) in enumerate(BLOCK_NAMES)
        rng = BLOCK_RANGES[k]
        isempty(rng) && continue
        block_mu = abs.(J2_elast[rng, :])
        block_sd = J2_elast_sd[rng, :]
        signif   = block_mu .> 1e-3
        noise_ratio = any(signif) ? mean(block_sd[signif] ./ block_mu[signif]) : NaN
        if !isnan(noise_ratio) && noise_ratio > 0.10
            @warn "  $name noise ratio $(round(noise_ratio, sigdigits=3)) > 0.10"
        end
        @printf("  %-12s max=%.4e  mean=%.4e  noise=%s\n",
                name, maximum(block_mu), mean(block_mu),
                isnan(noise_ratio) ? "n/a" : string(round(noise_ratio, sigdigits=3)))
    end

    # GMM inference at θ̂_2 (Σ_sim = 0, Ω = Σ_data)
    println("\nRunning GMM inference at θ̂_2...")
    W_eff     = NPZ.npzread(joinpath(output_folder, "step2", "W_eff.npy"))
    Sigma_data_gb = NPZ.npzread(joinpath(output_folder, "step2", "Sigma_data_gb.npy"))

    _, sim_moments_2 = full_SMM(theta_hat_2; analytical=true, n_quad=n_quad)
    sim_vec_2 = vcat([vec(sim_moments_2[i]) for i in 1:5]...)[MOMENT_MASK]
    emp_vec   = vec(empirical_moments)

    # β-then-γ ordering (invariant)
    gb_indices = vcat(collect(BLOCK_RANGES[4]), collect(BLOCK_RANGES[5]))
    n_reg_loc  = length(BLOCK_RANGES[4]); n_gam_loc = length(BLOCK_RANGES[5])
    gb_block_ranges = (1:n_reg_loc, (n_reg_loc + 1):(n_reg_loc + n_gam_loc))
    gb_block_names  = ("reg_coef", "gamma_ls")

    # Restrict Jacobian columns to β+T (ensures G'WG full-rank, order condition)
    beta_T_start = 1 + S + R_downstream + 1                       # first β raw index
    gb_cols      = findall(i -> i >= beta_T_start, jacobian_param_indices)
    gb_param_idx = jacobian_param_indices[gb_cols]

    # Correctness check 1: Σ_data leading β-block must be N_REG × N_REG (moments, not params).
    Omega_gmm = Sigma_data_gb
    @assert size(Omega_gmm, 1) == N_REG + length(BLOCK_RANGES[5]) "Omega_gmm size $(size(Omega_gmm,1)) != N_REG+n_γ=$(N_REG+length(BLOCK_RANGES[5]))"
    # Correctness check 2: exactly N_TAU β/α labels in the restricted param columns.
    n_beta_labels = count(l -> startswith(l, "alpha") || startswith(l, "beta"), PARAM_LABELS[gb_cols])
    @assert n_beta_labels == N_TAU "Expected $N_TAU β/α labels in gb_cols, found $n_beta_labels — beta_T_start misaligned with N_TAU"

    J2_gb      = J2[gb_indices, gb_cols]
    sim_vec_gb = sim_vec_2[gb_indices]
    emp_vec_gb = emp_vec[gb_indices]

    N_gb = length(gb_indices)
    n_gb_params = length(gb_param_idx)
    println("Step-4 inference: J2_gb is $(N_gb)×$(n_gb_params) " *
            "($(N_gb) β+γ moments × $(n_gb_params) β+T params). " *
            "df = $(N_gb - n_gb_params)")

    # T-identification eigen-screen (diagnostic, print-only) at θ̂_2
    screen_T_identification(theta_hat_2;
        J            = J2_gb,
        W            = W_eff,
        param_labels = PARAM_LABELS[gb_cols],
        label        = "GMM step4 θ̂_2")

    compute_smm_inference(
        theta_hat_2, J2_gb, W_eff, Omega_gmm;
        param_indices         = gb_param_idx,
        empirical_moments_vec = emp_vec_gb,
        simulated_moments_vec = sim_vec_gb,
        output_folder         = joinpath(output_folder, "step3"),
        industry              = industry,
        K_sim                 = 0,
        block_ranges          = gb_block_ranges,
        block_names           = gb_block_names,
        gamma_ref_map         = GAMMA_REF_MAP,
        param_labels          = PARAM_LABELS[gb_cols],
        moment_labels         = MOMENT_LABELS[gb_indices]
    )

    # Add GMM-mode note to inference summary
    note_path = joinpath(output_folder, "step3", "inference", "gmm_note.txt")
    open(note_path, "w") do io
        println(io, "GMM mode: closed-form EK moments (n_quad=$n_quad for reg_coef).")
        println(io, "Σ_sim = 0 by construction → W_eff = Σ_data^{-1}.")
        println(io, "SEs are exact delta-method without Murphy-Topel correction.")
        println(io, "Jacobian computed analytically via forward-mode AD (ForwardDiff, exact ∂m/∂θ).")
    end
    println("Step 4 inference complete. Results in step3/inference/.")
end

############## POST-HOC REPORTING ##############
run_reporting(joinpath(output_folder, "step1"), K; u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS,
              analytical=true, n_quad=n_quad)

println("\nGMM estimation complete. Results in: $output_folder")

theta_hat_2 = NPZ.npzread(joinpath(output_folder, "step3", "theta_hat_2.npy"))
# ============================================================================
# DIAGNOSTIC: price-index convergence & stratification alignment
# Append after Step 4 in main_gmm.jl. Reuses loaded globals.
# ============================================================================
function test_price_alignment(params; N_list = [2048, 8192, 32768], n_quad::Int = 200)
    ndims(params) > 1 && (params = params[:, 1])

    # Closed-form reference (the N→∞ target)
    moms_ana = compute_moments_analytical(params; n_quad = n_quad)
    Ω_L, Ω_s, A, β, T_vec = unpack_params(params)
    pr_ana = compute_prices_analytical(Ω_L, Ω_s, A, reshape(T_vec, S, R), build_tau(β))
    c_ana  = pr_ana.c_tilde_r                       # length R_downstream

    rel(sim, ana) = maximum(abs.(vec(sim) .- vec(ana)) ./ (abs.(vec(ana)) .+ 1e-12))

    println("\n" * "="^100)
    println("PRICE-INDEX ALIGNMENT TEST — block max-rel-err vs closed form (c̃_r isolates the price channel)")
    println("="^100)
    @printf("%-9s %-11s %10s %10s %10s %10s %10s %10s\n",
            "N_rho","draws","c_tilde","labor","industry","pi_r","reg_coef","gamma_ls")

    a_is = 0.5   # IS tilt (only used by the :is arm)
    for N in N_list
        uq, wq = generate_draws(N, n_good, :qmc;   randomise = false)
        um, wm = generate_draws(N, n_good, :mc;    randomise = false)
        us, ws = generate_draws(N, n_good, :sobol; randomise = false)
        ui, wi = generate_draws(N, n_good, :is;    randomise = false, a = a_is)

        # Decorrelation / ESS diagnostics for the stratified arms.
        #   • qmc/sobol have FLAT weights ⇒ ESS == N by construction (no degeneracy).
        #   • is  has bounded but non-flat weights ⇒ min ESS reported as a health signal.
        #   • columns should be decorrelated: max off-diagonal |cor(U)| ≈ 1/√N.
        ess_is = minimum(1.0 ./ vec(sum(wi .^ 2, dims = 1)))     # wi columns sum to 1
        Cq = cor(uq); Cq[diagind(Cq)] .= 0.0
        @printf("  N=%-8d qmc ESS = %.0f / %d (frac %.2f)   is ESS = %.0f / %d (frac %.2f)   max|cor(U_qmc)−I| = %.3f (≈ 1/√N = %.3f)\n",
                N, N, N, 1.0, ess_is, N, ess_is / N,
                maximum(abs.(Cq)), 1.0 / sqrt(N))

        # INTRA-sector decorrelation gate (the only check that catches a broken
        # scramble). The global max|cor(U)−I| above can look healthy while a thick
        # sector's columns are banded; restrict to columns of one sector for the
        # few largest sectors. A high value here ⇒ banding not dissolved ⇒ the
        # argmin-collapse mode the design exists to avoid. Expected ≈ 1/√N.
        thick = sort(1:S, by = s -> length(SECTOR_GOOD_INDICES[s]), rev = true)
        for s in Iterators.take(filter(s -> length(SECTOR_GOOD_INDICES[s]) >= 2, thick), 3)
            cols = SECTOR_GOOD_INDICES[s]
            for (lab, Umat) in (("qmc", uq), ("sobol", us))
                C = cor(Umat[:, cols]); C[diagind(C)] .= 0.0
                @printf("    N=%d sector %d (d=%d) intra max|cor| %-5s = %.3f (≈ 1/√N = %.3f)\n",
                        N, s, length(cols), lab, maximum(abs.(C)), 1.0 / sqrt(N))
            end
        end

        #  qmc   : stratified uniform, FLAT weights → winner-weight shortcut exact (unbiased)
        #  mc    : i.i.d. uniform, flat weights      → validates the closed form itself
        #  sobol : per-sector scrambled Sobol net    → flat weights ⇒ also unbiased (aligns ~qmc)
        #  is    : per-column importance tilt        → BIASED for min-coupled moments
        #                                              (documents the defect, does not hide it)
        for (lab, u, w) in (("qmc", uq, wq), ("mc", um, wm), ("sobol", us, ws), ("is", ui, wi))
            net  = solve_network(params; u_draws = u, sample_weights = w)
            m    = compute_moments(net, params)
            ctil = net.c_tilde_r[DOWNSTREAM_REGIONS]
            @printf("%-9d %-11s %10.2e %10.2e %10.2e %10.2e %10.2e %10.2e\n",
                    N, lab, rel(ctil, c_ana),
                    rel(m[1], moms_ana[1]), rel(m[2], moms_ana[2]), rel(m[3], moms_ana[3]),
                    rel(m[4], moms_ana[4]), rel(m[5], moms_ana[5]))
        end
        println("-"^100)
    end
    println("""
    Decision rule (default sampler is :qmc):
      • 'qmc'/'sobol' ≤ 'mc' on c_tilde/industry/pi_r/gamma_ls : flat weights make the winner-weight
                                            shortcut exact and stratification/equidistribution add
                                            variance reduction. This is the expected, correct behaviour.
      • 'sobol' aligns like 'qmc' on ALL blocks (≠ 'is') : confirms flat weights ⇒ exact shortcut ⇒
                                            unbiased, independent of the placement engine. The qmc-vs-sobol
                                            CHOICE is decided by Σ_sim variance (see test_sobol_variance).
      • 'is' biased (does NOT → 0 with N) on the min-coupled blocks : the IS tilt drops the losing
                                            columns' density ratios in solve_network's CES sum; this is a
                                            real defect, which is why :is is not the default.
      • reg_coef ~1e2 for ALL of them      : FKG/quadrature bias in the regression moment, NOT a
                                            sampling artefact (unaffected by the draw method). :is is
                                            valid here (single-column functional, no min-coupling).
      • all stall together on a block      : structural mismatch between solve_network and
                                            compute_moments_analytical (not a sampling issue).""")
end

# ============================================================================
# DIAGNOSTIC: Σ_sim variance, :qmc vs :sobol on the γ block (the DECISION gate)
# test_price_alignment validates non-bias (randomise=false, deterministic) but
# NOT the variance gain — the reason to adopt Sobol. This runs K randomise=true
# replications and compares the variance of the gamma_ls block.
# ============================================================================
function test_sobol_variance(params; K::Int = 64, N::Int = 8192)
    ndims(params) > 1 && (params = params[:, 1])

    # (i) PSO reproducibility — bloquant: randomise=false must be byte-identical.
    u1, _ = generate_draws(N, n_good, :sobol; randomise = false)
    u2, _ = generate_draws(N, n_good, :sobol; randomise = false)
    @assert u1 == u2 "Sobol randomise=false NON reproducible (master not frozen)"

    γidx = collect(BLOCK_RANGES[5])
    var_block = method -> begin
        M = reduce(hcat, pmap(1:K) do k
            u, w = generate_draws(N, n_good, method; randomise = true, rng = MersenneTwister(k))
            _, m = full_SMM(params; u_draws = u, sample_weights = w)   # (loss, moment 5-tuple)
            vcat([vec(m[i]) for i in 1:5]...)[MOMENT_MASK][γidx]
        end)'
        vec(var(M; dims = 1))
    end
    vq, vs = var_block(:qmc), var_block(:sobol)

    println("\n" * "="^100)
    println("Σ_sim VARIANCE TEST — γ-block per-moment variance over $K replications (N=$N)")
    println("="^100)
    @printf("γ-block mean Var: qmc = %.3e   sobol = %.3e   ratio = %.2f  (≤ 1 expected on thick sectors)\n",
            mean(vq), mean(vs), mean(vs) / mean(vq))
    println("""
    Interpretation:
      • ratio ≤ 1 : Sobol reduces Σ_sim variance; gain concentrates on multi-region sectors
                    (where the min has interaction content), neutral on single-region sectors.
      • (ii) cross-replication independence is implicitly tested: MersenneTwister(1) vs (2) give
             different M; if the scramble ignored the rng, inter-replication variance would collapse
             (vs anomalously small) — flag that case rather than reading it as a 'win'.""")
end

test_price_alignment(theta_hat_2; n_quad = n_quad)   # theta_hat_1 works too if Step 4 is off
test_sobol_variance(theta_hat_2)                      # qmc-vs-sobol decision gate
