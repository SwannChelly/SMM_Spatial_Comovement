##### Three-Step SMM — main.jl #####
# Entry point implementing the three-step efficient SMM estimator.
# Step 1: identity-weighted PSO → θ̂_1
# Step 2: build W_step3 = (Σ_data + (1+1/K)·Σ_sim)^{-1} at θ̂_1
# Step 3: efficient-weighted PSO → θ̂_2, warm-started at θ̂_1
# Legacy entry point: main_pso.jl (unchanged, identity weights only)

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
using LinearAlgebra
using Statistics, Printf
using StatsBase

@everywhere include("model_CP.jl")
@everywhere include("tools.jl")
@everywhere include("pso_integration.jl")
@everywhere include("run_untargeted_validation.jl")

############## Parse arguments ##############

industry = length(ARGS) >= 1 ? ARGS[1] : "aero"
n_coef   = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
K_sim    = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 10000 # K for Σ_sim estimation

K = 20

run_step1 = false#true
run_step2 = true
run_step3 = false


if !(n_coef in [1, 4, 5])
    error("n_coef must be 1, 4 or 5, got: $n_coef")
end

println("Industry: $industry | n_coef: $n_coef | K_sim: $K_sim")

input_folder  = "./baseline_$industry"
output_folder = "./reporting_$industry"
mkpath(output_folder)

############## Load and distribute constants ##############
include("load_parameters.jl")
NPZ.npzwrite(joinpath(output_folder, "n_reg_coef.npy"), n_coef)

############## Determine step to run ##############

step1_folder = joinpath(output_folder, "step1")
step2_W_path = joinpath(output_folder, "step2", "W_step3.npy")


############## STEP 1 — Identity-weighted optimisation ##############
# Full optimisation to fixe un-bootstrapable parameters. 

if run_step1
    println("\n" * "="^70)
    println("STEP 1: Identity-weighted PSO optimisation")
    println("="^70)

    theta_hat_1, _ = run_pso_optimization(;
        weight_matrix            = nothing,   
        skip_initial_beta_search = false,
        warm_start_params        = nothing,
        output_subfolder         = "step1",
        max_loop                 = K
    )

    NPZ.npzwrite(joinpath(output_folder, "step1", "theta_hat_1.npy"), theta_hat_1)
    println("Step 1 complete. θ̂_1 saved.")
else
    println("Step 1 skipped (resume).")
end

############## Load θ̂_1 ##############

step1_last = find_last_stage_folder(joinpath(output_folder, "step1"))
theta_hat_1 = NPZ.npzread(joinpath(step1_last, "best_params.npy"))

if ndims(theta_hat_1) > 1
    theta_hat_1 = theta_hat_1[:, 1]
end
println("θ̂_1 loaded from: $step1_last")

############## STEP 2 — Build efficient weight matrix ##############
# W_step3 is the weight matrix for the second optimisation. 
# Second optimisation only estimate T_{sr} and \beta_k. Therefore, the weight matrix is restricted to those parameters. 
# We also build the Jacobian 
#   - Used for inference. 
#   - Used to analyse if selecting only (T,\beta) is going to have an important effect on other moments. 

if run_step2
    println("\n" * "="^70)
    println("STEP 2: Building efficient weight matrix (K=$K_sim)")
    println("="^70)

    W_step3 = build_step3_weight_matrix(theta_hat_1, input_folder;
                                         K=K_sim, output_folder=output_folder)

    # Jacobian at θ̂_1 — identified parameters only
    J1, J1_elast, J1_sd, J1_elast_sd = compute_jacobian(
        theta_hat_1;
        param_indices = jacobian_param_indices,
        output_folder = output_folder,
        output_subdir = "step2",
        filename      = "jacobian_all.npy",
        K             = 50,
        step_rel      = 1e-3,
        step_abs      = 1e-8,
        base_seed     = 2_000_000   # disjoint from Σ_sim seeds (1:K_sim) and step-3 (1_000_000)
    )

    # Inference at θ̂_1 using efficient weight W_step3 and Ω from step2/
    Omega_step2 = NPZ.npzread(joinpath(output_folder, "step2", "Omega.npy"))
    _, sim_moments_1 = full_SMM(theta_hat_1; u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS)
    sim_vec_1 = vcat([vec(sim_moments_1[i]) for i in 1:5]...)[MOMENT_MASK]
    emp_vec   = vec(empirical_moments)

    gb_indices = vcat(collect(BLOCK_RANGES[4]), collect(BLOCK_RANGES[5]))   # β then γ
    n_reg_loc  = length(BLOCK_RANGES[4]); n_gam_loc = length(BLOCK_RANGES[5])
    gb_block_ranges = (1:n_reg_loc, (n_reg_loc + 1):(n_reg_loc + n_gam_loc))
    gb_block_names  = ("reg_coef", "gamma_ls")

    # Restrict Jacobian columns to β+T (the only params β+γ moments identify)
    beta_T_start = 1 + S + R_downstream + 1                       # first β raw index
    gb_cols      = findall(i -> i >= beta_T_start, jacobian_param_indices)
    gb_param_idx = jacobian_param_indices[gb_cols]

    J_gb       = J1[gb_indices, gb_cols]                          # β+γ rows × β+T cols
    sim_vec_gb = sim_vec_1[gb_indices]
    emp_vec_gb = emp_vec[gb_indices]
    Weight_matrix_inference = Weight_matrix_custom[gb_indices, gb_cols]    

    compute_smm_inference(
         theta_hat_1, J_gb, Weight_matrix_inference, Omega_step2;
         param_indices         = gb_param_idx,
         empirical_moments_vec = emp_vec_gb,
         simulated_moments_vec = sim_vec_gb,
         output_folder         = joinpath(output_folder, "step2"),
         industry              = industry,
         K_sim                 = K_sim,
         block_ranges          = gb_block_ranges,
         block_names           = gb_block_names
    )
    println("Step 2 complete. W_step3 and θ̂_1 inference saved.")
else
    println("Step 2 skipped (resume). Loading W_step3...")
    W_step3 = NPZ.npzread(step2_W_path)
end


############## STEP 3 — Efficient-weighted optimisation ##############

if run_step3
    println("\n" * "="^70)
    println("STEP 3: Efficient-weighted PSO optimisation")
    println("="^70)

    # A_r, labor share, and industry share are fixed at θ̂_1.
    # Only β and T are optimised, using the gamma+beta-only weight matrix from step 2.
    theta_hat_2, _ = run_pso_optimization(;
        weight_matrix            = W_step3,
        skip_initial_beta_search = true,
        warm_start_params        = theta_hat_1,
        output_subfolder         = "step3",
        max_loop                 = K,
        gamma_beta_only          = true,
        moments_loss_gamma_beta  = true
    )

    NPZ.npzwrite(joinpath(output_folder, "step3", "theta_hat_2.npy"), theta_hat_2)
    println("Step 3 complete. θ̂_2 saved.")

    # ── Jacobian at θ̂_2 — all parameters ────────────────────────────────────
    println("\nComputing Jacobian at θ̂_2 (base_seed=1_000_000 to avoid collision with Σ_sim seeds)...")
    J2, J2_elast, J2_sd, J2_elast_sd = compute_jacobian(
        theta_hat_2;
        param_indices = jacobian_param_indices,
        output_folder = output_folder,
        output_subdir = "step3",
        filename      = "jacobian_all_step3.npy",
        K             = 50,
        step_rel      = 1e-3,
        step_abs      = 1e-8,
        base_seed     = 1_000_000
    )

    # Rank of J2
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

    # ── SMM Inference at θ̂_2 ─────────────────────────────────────────────────
    println("\nRunning SMM inference at θ̂_2...")
    W_step3_inf = NPZ.npzread(joinpath(output_folder, "step2", "W_step3.npy"))
    Omega_inf   = NPZ.npzread(joinpath(output_folder, "step2", "Omega.npy"))

    _, sim_moments_2 = full_SMM(theta_hat_2; u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS)
    sim_vec_2 = vcat([vec(sim_moments_2[i]) for i in 1:5]...)[MOMENT_MASK]
    emp_vec   = vec(empirical_moments)

    # Restrict to β+γ moments (β then γ) for inference
    gb_indices = vcat(collect(BLOCK_RANGES[4]), collect(BLOCK_RANGES[5]))
    n_reg_loc  = length(BLOCK_RANGES[4]); n_gam_loc = length(BLOCK_RANGES[5])
    gb_block_ranges = (1:n_reg_loc, (n_reg_loc + 1):(n_reg_loc + n_gam_loc))
    gb_block_names  = ("reg_coef", "gamma_ls")

    # Restrict Jacobian columns to β+T (the only params β+γ moments identify)
    beta_T_start = 1 + S + R_downstream + 1                       # first β raw index
    gb_cols      = findall(i -> i >= beta_T_start, jacobian_param_indices)
    gb_param_idx = jacobian_param_indices[gb_cols]

    J2_gb      = J2[gb_indices, gb_cols]                          # β+γ rows × β+T cols
    sim_vec_gb = sim_vec_2[gb_indices]
    emp_vec_gb = emp_vec[gb_indices]

    compute_smm_inference(
        theta_hat_2, J2_gb, W_step3_inf, Omega_inf;
        param_indices         = gb_param_idx,
        empirical_moments_vec = emp_vec_gb,
        simulated_moments_vec = sim_vec_gb,
        output_folder         = joinpath(output_folder, "step3"),
        industry              = industry,
        K_sim                 = K_sim,
        block_ranges          = gb_block_ranges,
        block_names           = gb_block_names
    )
end

############## POST-HOC ANALYSIS ##############
run_reporting(joinpath(output_folder, "step1"), K; u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS)


last_stage_folder = find_last_stage_folder(joinpath(output_folder, "step1"))
println("\n" * "="^70)
println("POST-HOC ANALYSIS FOR $(uppercase(industry))")
println("="^70)
println("Loading parameters from: $last_stage_folder")

best_params = NPZ.npzread(joinpath(last_stage_folder, "best_params.npy"))
if ndims(best_params) > 1
    best_params = best_params[:, 1]
end

println("\n>>> RUNNING UNIFIED VALIDATION (ALL THREE MODELS) <<<")
results_unified = validate_table2_all_models(best_params, industry, T_periods=36, time_fe_mode="resample")
Parquet.write_parquet(joinpath(output_folder, "simulated_panel_unified.parquet"), results_unified["panel_df"])
Parquet.write_parquet(joinpath(output_folder, "regional_sales_unified.parquet"), results_unified["regional_sales_df"])

network = solve_network(best_params, return_firm_level=true, u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS)
folder  = output_folder

mu  = network.mu
Y_r = network.Y_r
w_srd_r = zeros(S, R, R)
X_lrs_sparse = zeros(R, R, S)
n_entries = length(network.firm_exp_rho)
for i in 1:n_entries
    g = network.firm_exp_g[i]
    l = GOOD_R[g]
    s = network.firm_exp_s[i]
    r = network.firm_exp_r[i]
    X_lrs_sparse[l, r, s] += network.firm_exp_val[i] * mu * Y_r[r]
end

for s in 1:S, r_prime in 1:R
    total = sum(X_lrs_sparse[r_prime, :, s])
    if total > 1e-10
        for r in 1:R
            w_srd_r[s, r_prime, r] = X_lrs_sparse[r_prime, r, s] / total
        end
    end
end
npzwrite(joinpath(folder, "w_srd_r.npy"), w_srd_r)

suppliers = zeros(Bool, S, N_rho, R)
for g in 1:n_good
    s = GOOD_S[g]; r = GOOD_R[g]
    for rho in 1:N_rho
        if network.linkages_flat[rho, g] > 0
            suppliers[s, rho, r] = true
        end
    end
end
npzwrite(joinpath(folder, "suppliers.npy"), suppliers)

sirens = Int[]; sectors = Int[]; ze2010 = Int[]; ze2010_downstream = Int[]
share = Float64[]; downstream_purchase = Float64[]
intermediate_derivative = Float64[]; productivity = Float64[] ; sample_weight_vec = Float64[]
siren_map = Dict{Tuple{Int,Int,Int}, Int}()
siren_counter = 0
for g in 1:n_good
    l = GOOD_R[g]; s = GOOD_S[g]
    for rho in 1:N_rho
        key = (l, s, rho)
        if !haskey(siren_map, key)
            siren_counter += 1
            siren_map[key] = siren_counter
        end
    end
end
for i in 1:n_entries
    rho = network.firm_exp_rho[i]; s = network.firm_exp_s[i]
    g   = network.firm_exp_g[i];   l = GOOD_R[g]
    r   = network.firm_exp_r[i]
    push!(sirens, siren_map[(l, s, rho)]); push!(sectors, s)
    push!(ze2010, l); push!(ze2010_downstream, r)
    push!(share, network.firm_exp_val[i])
    push!(downstream_purchase, Y_r[r] * mu)
    push!(intermediate_derivative, network.firm_deriv_val[i])
    push!(productivity, network.z_flat[rho, g])
    push!(sample_weight_vec, network.sample_weights[rho])
end
df = DataFrame(SIREN=sirens, A129=sectors, ze2010=ze2010,
               ze2010_downstream=ze2010_downstream, share=share,
               downstream_purchase=downstream_purchase,
               intermediate_derivative=intermediate_derivative,
               productivity=productivity,sample_weight = sample_weight_vec)
Parquet.write_parquet(joinpath(folder, "suppliers.parquet"), df)

println("\nPost-hoc analysis complete. Results saved to: $folder")