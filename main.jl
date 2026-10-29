##### Three-step SMM — main specification #####
# Granular varieties, comparative advantage T_{a(l)s} n_ls^κ at the attraction-area
# level, cell-level cloglog extensive margin, T profiled out of the search.
#
#   Step 1  identity-weighted PSO                          → θ̂_1   (step1/)
#   Step 2  W = Cov(β_world − β_sim)⁻¹ over fake worlds at θ̂_1     (step2/), and
#           inference at θ̂_1 from fake worlds, W = I as in Step 1  (step2/inference_world/)
#   Step 3  W-weighted PSO on [α ; κ], loss on block 4      → θ̂_2   (step3/)
#   Step 4  inference from fake worlds at θ̂_2, whole chain re-estimated
#           in each                                         (step3/inference_world/)
#   Post-hoc: the firm-level economy at θ̂_1 and θ̂_2 (suppliers.parquet, ...)
#
# Usage: julia main.jl <industry> <n_coef> <n_tau> <n_rho_inf> <relax_n_lo> <profile_head>
# profile_head = true: (Ω^L, Ω^s, A) are profiled jointly with T (profiling.jl), every
# search runs over [α ; κ] only, and blocks 1–3 leave the loss.
# (run.sh builds the call). No data covariance is used: W and the inference both come
# from worlds drawn from the model (inference.jl). World counts: ENV WORLD_B_W (1000),
# WORLD_B_INF (500).

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
                  QuasiMonteCarlo, LinearAlgebra, Printf, SpecialFunctions, Optim

@everywhere include("model_CP.jl")
@everywhere include("profiling.jl")
@everywhere include("optimizer.jl")
@everywhere include("reporting.jl")
@everywhere include("inference.jl")

############## Arguments ##############

industry        = length(ARGS) >= 1 ? ARGS[1] : "auto"
n_coef          = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4
n_tau           = length(ARGS) >= 3 && !isempty(strip(ARGS[3])) ? parse(Int, ARGS[3]) : n_coef
n_rho_inference = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 10_000   # unused since the fake-world inference
relax_n_lo      = length(ARGS) >= 5 ? lowercase(strip(ARGS[5])) in ("true", "1", "yes") : false
profile_head    = length(ARGS) >= 6 ? lowercase(strip(ARGS[6])) in ("true", "1", "yes") : false
n_coef in (1, 4, 5) || error("n_coef must be 1, 4 or 5, got $n_coef")
n_tau in (1, n_coef) || error("n_tau must be 1 (τ = d^α) or n_coef (one α per distance bin), got $n_tau")
n_rho_inference >= 1 || error("n_rho_inf must be ≥ 1")
println("Industry: $industry | n_coef: $n_coef | n_tau: $n_tau | n_rho_inf: $n_rho_inference | relax_n_lo: $relax_n_lo | profile_head: $profile_head")

run_step1 = true
run_step2 = true
run_step3 = true
run_step4 = true
K_loops = 20         # block-coordinate loops per optimisation step
B_w     = parse(Int, get(ENV, "WORLD_B_W", "1000"))    # fake worlds for W (Step 2)
B_inf   = parse(Int, get(ENV, "WORLD_B_INF", "500"))   # fake worlds for inference (Step 4)

input_folder  = "./baseline_$industry"
output_folder = "./reporting_$(industry)_profiled_aa_gran_kappa" * (relax_n_lo ? "_nlo1" : "") *
                (profile_head ? "_profhead" : "") * "_pso"
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
    # α₀ from the data prior; with distance bins, the per-segment elasticities computed from
    # the bin coefficients in load_parameters.jl (ALPHA_BIN_INIT).
    alpha_init = N_TAU == 1 ? [P_alpha] : copy(ALPHA_BIN_INIT)
    warm_start = vcat(agg_labor_share, agg_industry_share, A_init, alpha_init,
                      KAPPA_PRIOR, vec(permutedims(T_rs_init))[T_MASK])
    theta_hat_1, _ = run_optimization(; warm_start_params = warm_start, output_subfolder = "step1",
                                      max_loop = K_loops)
    NPZ.npzwrite(joinpath(step1_dir, "theta_hat_1.npy"), theta_hat_1)
    report_granular(theta_hat_1, step1_dir; industry = industry, label = "theta_1")
end
theta_hat_1 = read_params(joinpath(find_last_stage_folder(step1_dir), "best_params.npy"))

############## Steps 2–4 — fake worlds drawn from the model ##############

if run_step2
    println("\n" * "="^70 * "\nSTEP 2: weight matrix from $B_w fake worlds at θ̂_1\n" * "="^70)
    world_weight_matrix(theta_hat_1; B = B_w, dest = joinpath(output_folder, "step2"))
    # Inference on the Step-1 estimator: the same whole-chain re-estimation as Step 4, at
    # θ̂_1 and with Step 1's weighting (identity on block 4; γ and G are matched exactly,
    # A is recalibrated, Ω^L and Ω^s held at θ̂_1 as in Step 4 — re-profiled with A under
    # PROFILE_HEAD). Seeds disjoint from Step 4.
    println("\n" * "="^70 * "\nSTEP 2b: inference from $B_inf fake worlds at θ̂_1 (W = I)\n" * "="^70)
    world_inference(Vector{Float64}(theta_hat_1), Matrix(1.0I, length(BLOCK_RANGES[4]), length(BLOCK_RANGES[4]));
                    B = B_inf, dest = joinpath(output_folder, "step2", "inference_world"),
                    tcol_names = _tcol_names, sector_names = string.(SECTOR_NAMES),
                    seed_world = 31_000_000, seed_sim = 32_000_000, label = "θ̂_1, W = I")
    report_granular(theta_hat_1, joinpath(output_folder, "step2"); industry = industry, label = "theta_1")
end
W_world = NPZ.npzread(joinpath(output_folder, "step2", "W_world.npy"))

if run_step3
    println("\n" * "="^70 * "\nSTEP 3: W-weighted optimisation of [α ; κ] (block 4)\n" * "="^70)
    theta_hat_2, _ = run_optimization(; weight_matrix = W_world, warm_start_params = theta_hat_1,
                                      output_subfolder = "step3", max_loop = K_loops,
                                      gamma_beta_only = true, loss_blocks = [4])
    NPZ.npzwrite(joinpath(step3_dir, "theta_hat_2.npy"), theta_hat_2)
end

if run_step4
    println("\n" * "="^70 * "\nSTEP 4: inference from $B_inf fake worlds at θ̂_2\n" * "="^70)
    theta_hat_2 = read_params(joinpath(step3_dir, "theta_hat_2.npy"))
    world_inference(Vector{Float64}(theta_hat_2), W_world; B = B_inf,
                    dest = joinpath(step3_dir, "inference_world"),
                    tcol_names = _tcol_names, sector_names = string.(SECTOR_NAMES),
                    label = "θ̂_2, W = Ω̃⁻¹")
    report_granular(theta_hat_2, step3_dir; industry = industry, label = "theta_2")
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
