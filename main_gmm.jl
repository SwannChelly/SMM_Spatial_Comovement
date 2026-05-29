##### Closed-form GMM estimation — main_gmm.jl #####
# Drop-in replacement for main.jl using analytical EK moments.
# Three-step GMM:
#   Step 1: identity-weighted PSO with analytical moments
#   Step 2: W_eff = Σ_data^{-1}  (Σ_sim = 0 by construction)
#   Step 3: efficient-weighted PSO (β+T only), warm-started at θ̂_1
#
# Usage:
#   julia main_gmm.jl aero 4          # n_quad=200 (default)
#   julia main_gmm.jl aero 4 resume 200 500   # n_quad=500 (high accuracy)

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
@everywhere include("pso_integration.jl")

############## Parse arguments ##############

industry = length(ARGS) >= 1 ? ARGS[1] : "auto"
n_coef   = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
n_quad   = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 200

K = 5   # PSO loops

if !(n_coef in [1, 4, 5])
    error("n_coef must be 1, 4 or 5, got: $n_coef")
end

println("GMM mode | Industry: $industry | n_coef: $n_coef | n_quad: $n_quad")

input_folder  = "./baseline_$industry"
output_folder = "./reporting_gmm_$industry"
mkpath(output_folder)

############## Load and distribute constants ##############
include("load_parameters.jl")
NPZ.npzwrite(joinpath(output_folder, "n_reg_coef.npy"), n_coef)

run_step1 = true
run_step3 = true

############## STEP 1 — Identity-weighted GMM ##############

step1_folder = joinpath(output_folder, "step1")
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
    println("Step 1 complete. θ̂_1 saved.")
else
    step1_last = find_last_stage_folder(joinpath(output_folder, "step1"))
    theta_hat_1 = NPZ.npzread(joinpath(step1_last, "best_params.npy"))
    ndims(theta_hat_1) > 1 && (theta_hat_1 = theta_hat_1[:, 1])
    println("Step 1 skipped. θ̂_1 loaded from: $step1_last")
end

############## Load θ̂_1 ##############

step1_last = find_last_stage_folder(joinpath(output_folder, "step1"))
theta_hat_1 = NPZ.npzread(joinpath(step1_last, "best_params.npy"))
ndims(theta_hat_1) > 1 && (theta_hat_1 = theta_hat_1[:, 1])
println("θ̂_1 loaded from: $step1_last")

############## STEP 2 — Efficient weight matrix (W = Σ_data^{-1}) ##############
# In GMM mode, Σ_sim = 0 by construction → W_eff = Σ_data^{-1} directly.
# No simulation replication needed.

println("\n" * "="^70)
println("STEP 2: Building efficient weight matrix W = Σ_data^{-1}")
println("="^70)

step2_folder = joinpath(output_folder, "step2")
mkpath(step2_folder)

# Load empirical covariance matrices
Sigma_data_full = try
    NPZ.npzread(joinpath(input_folder, "Sigma.npy"))
catch
    # Fall back to block-diagonal from w_gamma + w_beta
    w_gamma = NPZ.npzread(joinpath(input_folder, "w_gamma.npy"))
    w_beta  = NPZ.npzread(joinpath(input_folder, "w_beta.npy"))
    n_gam   = size(w_gamma, 1)
    n_b     = size(w_beta, 1)
    S_full  = zeros(n_gam + n_b, n_gam + n_b)
    S_full[1:n_gam, 1:n_gam]                   = w_gamma
    S_full[(n_gam+1):end, (n_gam+1):end]       = w_beta
    S_full
end

# The gamma+beta moment block indices (matching load_parameters.jl BLOCK_RANGES ordering)
gb_indices = vcat(collect(BLOCK_RANGES[4]), collect(BLOCK_RANGES[5]))
N_gb = length(gb_indices)

# Extract Σ_data restricted to gamma+beta moments
Sigma_data_gb = Sigma_data_full[1:N_gb, 1:N_gb]

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

# Jacobian at θ̂_1 using analytical moments
println("\nComputing Jacobian at θ̂_1 (analytical, K=1 evaluation per perturbation)...")
J1, J1_elast, J1_sd, J1_elast_sd = compute_jacobian(
    theta_hat_1;
    param_indices = jacobian_param_indices,
    output_folder = output_folder,
    output_subdir = "step2",
    filename      = "jacobian_all.npy",
    K             = 1,           # analytical is deterministic: 1 replication suffices
    step_rel      = 1e-4,
    step_abs      = 1e-9,
    base_seed     = 0,
    analytical    = true,
    n_quad        = n_quad
)

println("Step 2 complete. W_eff and Jacobian saved.")

############## STEP 3 — Efficient-weighted GMM ##############


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
    println("Step 3 complete. θ̂_2 saved.")

    # Jacobian at θ̂_2
    println("\nComputing Jacobian at θ̂_2 (analytical)...")
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
        n_quad        = n_quad
    )

    # Inference: GMM SEs are exact delta-method (Σ_sim = 0)
    _, sim_moments_2 = full_SMM(theta_hat_2; analytical=true, n_quad=n_quad)
    sim_vec_2 = vcat([vec(sim_moments_2[i]) for i in 1:5]...)[MOMENT_MASK]
    emp_vec   = vec(empirical_moments)

    # Step 3 only identifies β and T parameters (A_r/labor/industry fixed at θ̂_1).
    # Restrict Jacobian columns to β+T to ensure G'WG is full-rank (order condition).
    # β starts at raw position 1 + S_ + R_down_ + 1; everything after is T.
    beta_T_start_raw  = 1 + S_ + R_down_ + 1
    gb_param_cols     = findall(i -> i >= beta_T_start_raw, jacobian_param_indices)
    gb_param_indices_step3 = jacobian_param_indices[gb_param_cols]

    J2_gb = J2[gb_indices, gb_param_cols]
    sim_vec_gb = sim_vec_2[gb_indices]
    emp_vec_gb = emp_vec[gb_indices]

    n_reg_loc = length(BLOCK_RANGES[4]); n_gam_loc = length(BLOCK_RANGES[5])
    gb_block_ranges = (1:n_reg_loc, (n_reg_loc + 1):(n_reg_loc + n_gam_loc))
    gb_block_names  = ("reg_coef", "gamma_ls")

    println("Step-3 inference: J2_gb is $(size(J2_gb,1))×$(size(J2_gb,2)) " *
            "($(N_gb) β+γ moments × $(length(gb_param_indices_step3)) β+T params). " *
            "Order condition: $(N_gb) ≥ $(length(gb_param_indices_step3)) → $(N_gb >= length(gb_param_indices_step3) ? "satisfied" : "VIOLATED").")

    # Omega = Sigma_data (Sigma_sim = 0 in GMM mode)
    Omega_gmm = Sigma_data_gb

    compute_smm_inference(
        theta_hat_2, J2_gb, W_eff, Omega_gmm;
        param_indices         = gb_param_indices_step3,
        empirical_moments_vec = emp_vec_gb,
        simulated_moments_vec = sim_vec_gb,
        output_folder         = joinpath(output_folder, "step3"),
        industry              = industry,
        K_sim                 = 0,
        block_ranges          = gb_block_ranges,
        block_names           = gb_block_names
    )

    # Add note to inference summary
    note_path = joinpath(output_folder, "step3", "inference", "gmm_note.txt")
    open(note_path, "w") do io
        println(io, "GMM mode: closed-form EK moments (n_quad=$n_quad for reg_coef).")
        println(io, "Σ_sim = 0 by construction → W_eff = Σ_data^{-1}.")
        println(io, "SEs are exact delta-method without Murphy-Topel correction.")
        println(io, "Jacobian computed analytically (single FD replication, K=1).")
    end
    println("Step 3 inference complete. Results in step3/inference/.")
end

############## POST-HOC REPORTING ##############
run_reporting(joinpath(output_folder, "step1"), K; u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS)

println("\nGMM estimation complete. Results in: $output_folder")


# beta_min = 1e-3
# beta_max = 10
# if n_coef == 1
#     length_range_beta = 10000
# end
# beta_search_method= "log_grid"
# if beta_search_method == "log_grid"
#     beta_candidates = generate_initial_betas("log_grid", N_beta, beta_min, beta_max;
#                                                 log_grid_length=length_range_beta)
# else
#     beta_candidates = generate_initial_betas("lhs", N_beta, beta_min, beta_max;
#                                                 lhs_n_samples=20000)
# end
# println("  Generated $(length(beta_candidates)) beta candidates")

# A_init = copy(emp_pi_r_full).^(1/abs(epsilon)) .* regional_wages[N_downstream_per_region .!= 0]
# A_init ./= sum(A_init)
# T_init_nz = vec(T_rs_init)[T_MASK]
# # New layout: [Ω^L | Ω^s | A | β | T] — beta is inserted between A and T
# init_other_prefix = vcat([agg_labor_share], agg_industry_share, A_init)
# expanding_beta = [vcat(init_other_prefix, beta, T_init_nz) for beta in beta_candidates]


# results_ = pmap(p -> parallel_SMM_safe(p; u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS,W_override=Weight_matrix_custom,analytical=true, n_quad=200), expanding_beta[1,:])


# scores = [r !== nothing ? r[1][1] : Inf for r in results_]
# best_idx = argmin(scores)

# init_beta = beta_candidates[best_idx]
# println("  Best initial beta: ", round.(init_beta, digits=6))