##### Three-step SMM — main specification #####
# Granular varieties, comparative advantage T_{a(l)s} n_ls^κ at the attraction-area
# level, cell-level cloglog extensive margin, T profiled out of the search.
#
#   Step 1  identity-weighted PSO                          → θ̂_1   (step1/)
#   Step 2  W_step3 = (Σ_data + Σ_sim)⁻¹, inference at θ̂_1          (step2/)
#   Step 3  efficient-weighted PSO on [α ; κ] (β → γ → G)   → θ̂_2   (step3/)
#   Step 4  inference at θ̂_2                                        (step3/inference/)
#   Post-hoc: the firm-level economy at θ̂_1 and θ̂_2 (suppliers.parquet, ...)
#
# Usage: julia main.jl <industry> <n_coef> <n_tau> <n_rho_inf> <relax_n_lo>
# (run.sh builds the call). Steps 2–4 need `SIGMA_DATA_FILE` in the input folder; without
# it the run stops after Step 1 and says why.

using Distributed
using Dates

n_workers = parse(Int, get(ENV, "N_WORKERS", "50"))
# Each worker is a separate process sizing its GC against total memory: cap its heap.
heap_hint = get(ENV, "WORKER_HEAP_HINT", "2G")
addprocs(max(n_workers - 1, 0); exeflags = isempty(heap_hint) ? `` : `--heap-size-hint=$heap_hint`)
println("Using $n_workers processes")

using Random
seed = 1234
Random.seed!(seed)
@everywhere using Random
@everywhere Random.seed!($seed)
@everywhere using NPZ, DataFrames, CSV, Distributions, Plots, Statistics, Parquet,
                  QuasiMonteCarlo, LinearAlgebra, Printf, SpecialFunctions

@everywhere include("model_CP.jl")
@everywhere include("profiling.jl")
@everywhere include("optimizer.jl")
@everywhere include("reporting.jl")
@everywhere include("inference.jl")

############## Arguments ##############

industry        = length(ARGS) >= 1 ? ARGS[1] : "auto"
n_coef          = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4
n_tau           = length(ARGS) >= 3 && !isempty(strip(ARGS[3])) ? parse(Int, ARGS[3]) : n_coef
n_rho_inference = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 10_000
relax_n_lo      = length(ARGS) >= 5 ? lowercase(strip(ARGS[5])) in ("true", "1", "yes") : false
n_coef in (1, 4, 5) || error("n_coef must be 1, 4 or 5, got $n_coef")
n_tau in (1, n_coef) || error("n_tau must be 1 (τ = d^α) or n_coef (one α per distance bin), got $n_tau")
n_rho_inference >= 1 || error("n_rho_inf must be ≥ 1")
println("Industry: $industry | n_coef: $n_coef | n_tau: $n_tau | n_rho_inf: $n_rho_inference | relax_n_lo: $relax_n_lo")

run_step1 = true
run_step2 = true
run_step3 = true
run_step4 = true
K_loops = 20         # block-coordinate loops per optimisation step
K_sim   = 10_000     # simulations for Σ_sim
K_jac   = 50         # draw replications averaged in the Jacobian

input_folder  = "./baseline_$industry"
output_folder = "./reporting_$(industry)_profiled_aa_gran_kappa" * (relax_n_lo ? "_nlo1" : "") * "_pso"
mkpath(output_folder)

include("load_parameters.jl")
NPZ.npzwrite(joinpath(output_folder, "n_reg_coef.npy"), n_coef)
NPZ.npzwrite(joinpath(output_folder, "n_tau.npy"), n_tau)

step1_dir = joinpath(output_folder, "step1")
step3_dir = joinpath(output_folder, "step3")
read_params(path) = (p = NPZ.npzread(path); ndims(p) > 1 ? p[:, 1] : p)

############## Step 1 — identity-weighted optimisation ##############

if run_step1
    println("\n" * "="^70 * "\nSTEP 1: identity-weighted optimisation\n" * "="^70)
    A_init = emp_pi_r_full .^ (1 / abs(epsilon)) .* regional_wages[N_downstream_per_region .!= 0]
    A_init ./= sum(A_init)
    # α₀ from the data prior; with distance bins, τ_b = 1 + α_b starts at the median of
    # d^{α₀} over the (cell, buyer) pairs of the bin.
    alpha_init = N_TAU == 1 ? [P_alpha] :
        [median(exp.(P_alpha .* LOG_DIST_DOWNSTREAM[DistBin .== b])) - 1 for b in 1:N_TAU]
    warm_start = vcat(agg_labor_share, agg_industry_share, A_init, alpha_init,
                      KAPPA_PRIOR, vec(permutedims(T_rs_init))[T_MASK])
    theta_hat_1, _ = run_optimization(; warm_start_params = warm_start, output_subfolder = "step1",
                                      max_loop = K_loops)
    NPZ.npzwrite(joinpath(step1_dir, "theta_hat_1.npy"), theta_hat_1)
    report_granular(theta_hat_1, step1_dir; industry = industry, label = "theta_1")
end
theta_hat_1 = read_params(joinpath(find_last_stage_folder(step1_dir), "best_params.npy"))

############## Steps 2–4 — need the data covariance ##############

Sigma_data, sigma_issue = load_sigma_data()
if Sigma_data === nothing
    println("\nSteps 2–4 skipped: $sigma_issue.\nThey need the bootstrap covariance of the inference " *
            "moments (β_d → β_κ → γ_{s,a} → Ḡ_s(0)) in $(joinpath(input_folder, SIGMA_DATA_FILE)).")
else
    if run_step2
        println("\n" * "="^70 * "\nSTEP 2: efficient weight matrix and inference at θ̂_1\n" * "="^70)
        W_step3 = build_weight_matrix(theta_hat_1, Sigma_data; K = K_sim)
        Omega = NPZ.npzread(joinpath(output_folder, "step2", "Omega.npy"))
        run_inference(theta_hat_1, W_step3, Omega, Sigma_data; dest = joinpath(output_folder, "step2"),
                      industry = industry, K = K_jac, base_seed = 2_000_000, K_sim = K_sim)
        report_granular(theta_hat_1, joinpath(output_folder, "step2"); industry = industry, label = "theta_1")
    end
    W_step3 = NPZ.npzread(joinpath(output_folder, "step2", "W_step3.npy"))

    if run_step3
        println("\n" * "="^70 * "\nSTEP 3: efficient-weighted optimisation of [α ; κ]\n" * "="^70)
        theta_hat_2, _ = run_optimization(; weight_matrix = W_step3, warm_start_params = theta_hat_1,
                                          output_subfolder = "step3", max_loop = K_loops,
                                          gamma_beta_only = true, moments_loss_gamma_beta = true)
        NPZ.npzwrite(joinpath(step3_dir, "theta_hat_2.npy"), theta_hat_2)
    end

    if run_step4
        println("\n" * "="^70 * "\nSTEP 4: inference at θ̂_2\n" * "="^70)
        theta_hat_2 = read_params(joinpath(step3_dir, "theta_hat_2.npy"))
        Omega = NPZ.npzread(joinpath(output_folder, "step2", "Omega.npy"))
        run_inference(theta_hat_2, W_step3, Omega, Sigma_data; dest = step3_dir,
                      industry = industry, K = K_jac, base_seed = 1_000_000, K_sim = K_sim)
        report_granular(theta_hat_2, step3_dir; industry = industry, label = "theta_2")
    end
end

############## Post-hoc firm-level economy ##############

for (path, dest) in ((joinpath(find_last_stage_folder(step1_dir), "best_params.npy"), step1_dir),
                     (joinpath(step3_dir, "theta_hat_2.npy"), step3_dir))
    isfile(path) || continue
    try
        write_post_hoc(Vector{Float64}(read_params(path)), dest)
    catch e
        @warn "post-hoc failed for $dest: $e"
    end
end
println("\nDone. Results under $output_folder")
