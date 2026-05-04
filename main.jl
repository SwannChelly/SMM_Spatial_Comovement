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

industry = length(ARGS) >= 1 ? ARGS[1] : "auto"
n_coef   = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4
resume   = length(ARGS) >= 3 && ARGS[3] == "resume"
K_sim    = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 10000  # K for Σ_sim estimation

if !(n_coef in [4, 5])
    error("n_coef must be 4 or 5, got: $n_coef")
end

println("Industry: $industry | n_coef: $n_coef | K_sim: $K_sim | resume: $resume")

input_folder  = "./baseline_$industry"
output_folder = "./reporting_$industry"
mkpath(output_folder)

############## Load and distribute constants ##############

coefs                         = CSV.read(joinpath(input_folder, "stats.csv"), DataFrame)
distances_local               = NPZ.npzread(joinpath(input_folder, "distances.npy"))
filter_N_upstream_local       = NPZ.npzread(joinpath(input_folder, "filter_N_upstream.npy"))
w_rs_local                    = NPZ.npzread(joinpath(input_folder, "w_rs.npy"))
regional_wages_local          = NPZ.npzread(joinpath(input_folder, "regional_wages.npy"))
N_downstream_per_region_local = NPZ.npzread(joinpath(input_folder, "N_downstream_per_region.npy"))
agg_industry_share_local      = NPZ.npzread(joinpath(input_folder, "input_share.npy"))
domestic_share_local          = NPZ.npzread(joinpath(input_folder, "domestic_share.npy"))
X_rs_local                    = NPZ.npzread(joinpath(input_folder, "X_rs.npy"))
N_rs_local                    = NPZ.npzread(joinpath(input_folder, "N_rs.npy"))

S_, R_full = size(filter_N_upstream_local)
@everywhere const S = $(S_)
@everywhere const R = $(R_full)

R_down_ = size(N_downstream_per_region_local[N_downstream_per_region_local .!= 0])[1]
@everywhere const R_downstream        = $(R_down_)
@everywhere const agg_industry_share  = $(agg_industry_share_local)
@everywhere const agg_labor_share     = $(coefs[2, "value"])
@everywhere const domestic_share      = $(domestic_share_local)
@everywhere regional_wages            = $(regional_wages_local)
@everywhere const distances           = $(distances_local)
@everywhere const N_downstream_per_region = $(N_downstream_per_region_local)
@everywhere const w_rs                = $(w_rs_local)
@everywhere const filter_N_upstream   = $(filter_N_upstream_local)
@everywhere const N_rho               = $(1000)
@everywhere const epsilon             = $(coefs[1, "value"])
@everywhere const lambda              = $(0.5)
@everywhere const nu                  = $(0.2)
@everywhere const nu_s                = $(ones(S_) .* 2.5)
@everywhere const theta               = $(1.768)
@everywhere const delta_r             = $(ones(R_full))
@everywhere const Weight_matrix       = $(nothing)

T_mask_local         = vec(X_rs_local) .> 0
T_mask_moment_local  = vec(permutedims(X_rs_local)) .> 0 # Vec flattens column per column.  So we have all region within the first sector and so on
@everywhere const T_MASK        = $T_mask_local
@everywhere const T_MASK_MOMENT = $T_mask_moment_local

good_indices_local        = findall(reshape(T_mask_local, S_, R_full))
n_good_local              = length(good_indices_local)
GOOD_S_local              = [ci[1] for ci in good_indices_local]
GOOD_R_local              = [ci[2] for ci in good_indices_local]
SECTOR_GOOD_INDICES_local = [findall(GOOD_S_local .== s) for s in 1:S_]
SECTOR_GOOD_REGIONS_local = [GOOD_R_local[idx] for idx in SECTOR_GOOD_INDICES_local]
SR_TO_GOOD_local          = zeros(Int, S_, R_full)
for (g, ci) in enumerate(good_indices_local)
    SR_TO_GOOD_local[ci[1], ci[2]] = g
end
W_RS_FLAT_local = [w_rs_local[GOOD_R_local[g]] for g in 1:n_good_local]

@everywhere const n_good               = $n_good_local
@everywhere const GOOD_S               = $GOOD_S_local
@everywhere const GOOD_R               = $GOOD_R_local
@everywhere const SECTOR_GOOD_INDICES  = $SECTOR_GOOD_INDICES_local
@everywhere const SECTOR_GOOD_REGIONS  = $SECTOR_GOOD_REGIONS_local
@everywhere const SR_TO_GOOD           = $SR_TO_GOOD_local
@everywhere const W_RS_FLAT            = $W_RS_FLAT_local

@everywhere const emp_gamma_ls   = $(permutedims(NPZ.npzread(joinpath(input_folder, "emp_gamma_ls.npy"))))
X_dr_local = CSV.read(joinpath(input_folder, "X_dr.csv"), DataFrame).X_dr
X_dr_local = X_dr_local[N_downstream_per_region_local .!= 0]
emp_pi_r_local = X_dr_local ./ sum(X_dr_local)
@everywhere const emp_pi_r_full  = $(emp_pi_r_local)
@everywhere const emp_pi_r       = $(emp_pi_r_local)
@everywhere const reg_coef       = $(NPZ.npzread(joinpath(input_folder, "reg_coef_$(n_coef).npy")))
@everywhere const N_beta         = $(length(NPZ.npzread(joinpath(input_folder, "reg_coef_$(n_coef).npy"))))

# Gravity-based T initialisation
T_gravity = zeros(S_, R_full)
for s in 1:S_
    idxs = SECTOR_GOOD_INDICES[s]
    for g in idxs
        l = GOOD_R[g]
        T_gravity[s, l] = max(emp_gamma_ls[l, s] * (w_rs_local[l]^theta), 1e-12)
    end
    vals = T_gravity[s, GOOD_R[SECTOR_GOOD_INDICES[s]]]
    m = maximum(vals)
    if m > 0; T_gravity[s, GOOD_R[SECTOR_GOOD_INDICES[s]]] ./= m; end
end
@everywhere const T_rs_init = $(T_gravity)

# Moment block sizes + MOMENT_MASK
n_labor    = 1
n_industry = length(vec(agg_industry_share_local))
n_gamma    = length(vec(permutedims(NPZ.npzread(joinpath(input_folder, "emp_gamma_ls.npy")))))
n_reg      = length(NPZ.npzread(joinpath(input_folder, "reg_coef_$(n_coef).npy")))
n_pi       = length(emp_pi_r_local)
N_moments_full = n_labor + n_industry + n_gamma + n_reg + n_pi

empirical_moments_local = vcat(
    [agg_labor_share],
    vec(agg_industry_share_local),
    vec(permutedims(NPZ.npzread(joinpath(input_folder, "emp_gamma_ls.npy")))),
    NPZ.npzread(joinpath(input_folder, "reg_coef_$(n_coef).npy")),
    emp_pi_r_local
)

moment_mask_local = trues(N_moments_full)
# Remove first industry share. 
moment_mask_local[n_labor + 1] = false 
# Remove non active gamma_ls 
for idx in 1:(S_ * R_full)
    if !T_mask_moment_local[idx]
        moment_mask_local[n_labor + n_industry + idx] = false
    end
end
# Remove first active gamma_ls per sector s. 
for s in 1:S_
    sector_start = (s - 1) * R_full + 1
    sector_end   = s * R_full
    active_positions = findall(T_mask_moment_local[sector_start:sector_end])
    if !isempty(active_positions)
        moment_mask_local[n_labor + n_industry + (s - 1) * R_full + active_positions[1]] = false
    end
end
# Remove first pi_r. 
moment_mask_local[n_labor + n_industry + n_gamma + n_reg + 1] = false

empirical_moments_local = reshape(empirical_moments_local[moment_mask_local], 1, sum(moment_mask_local))
N_moments = sum(moment_mask_local)

@everywhere const MOMENT_MASK        = $moment_mask_local
@everywhere const empirical_moments  = $(empirical_moments_local)
@everywhere const K_max              = $(50)

BLOCK_RANGES_local = compute_block_ranges(n_labor, n_industry, n_gamma, n_reg, n_pi, moment_mask_local)
@everywhere const BLOCK_RANGES = $BLOCK_RANGES_local
@everywhere const BLOCK_NAMES  = ("labor", "industry", "gamma_ls", "reg_coef", "pi_r")

w_vec = ones(N_moments)
w_vec[BLOCK_RANGES_local[4]] .= 100.0
Weight_matrix_custom_local = Diagonal(w_vec)
@everywhere const Weight_matrix_custom = $Weight_matrix_custom_local

println("Generating CdGM-style stratified draws...")
u_draws_local, sample_weights_local = generate_stratified_draws(N_rho, n_good_local)
@everywhere const U_DRAWS        = $u_draws_local
@everywhere const SAMPLE_WEIGHTS = $sample_weights_local

downstream_regions_local     = findall(N_downstream_per_region_local .> 0)
@everywhere const DOWNSTREAM_REGIONS = $(downstream_regions_local)

distances_downstream_local = distances_local[:, downstream_regions_local]
DistBin_local = Array{Int}(undef, R_full, R_down_)
for i in 1:R_full, j in 1:R_down_
    DistBin_local[i, j] = distance_bin(distances_downstream_local[i, j])
end
@everywhere const DistBin = $(DistBin_local)

closest_plant_dist_local        = vec(minimum(distances_downstream_local, dims=2))
closest_downstream_region_local = vec(getindex.(argmin(distances_downstream_local, dims=2), 2))
@everywhere const CLOSEST_PLANT_DIST        = $(closest_plant_dist_local)
@everywhere const CLOSEST_DOWNSTREAM_REGION = $(closest_downstream_region_local)

println("Constants distributed. N_moments=$N_moments, n_good=$n_good_local")

############## Determine resume state ##############

step1_folder = joinpath(output_folder, "step1")
step2_W_path = joinpath(output_folder, "step2", "W_step3.npy")

run_step1 = false#true
run_step2 = true
run_step3 = true

if resume
    if isfile(step2_W_path)
        # Step 2 complete — check if step3 has output
        step3_folder = joinpath(output_folder, "step3")
        if isdir(step3_folder) && isdir(joinpath(step3_folder, "0"))
            run_step1 = false
            run_step2 = false
            run_step3 = true  # resume step 3
            println("Resuming: Step 1+2 done. Resuming Step 3.")
        else
            run_step1 = false
            run_step2 = false
            run_step3 = true
            println("Resuming: Steps 1+2 done. Starting Step 3.")
        end
    elseif isdir(step1_folder) && isdir(joinpath(step1_folder, "0"))
        run_step1 = false  # step1 done
        run_step2 = true
        run_step3 = true
        println("Resuming: Step 1 done. Running Steps 2+3.")
    else
        println("Resume requested but no completed step found. Running all steps.")
    end
end

############## STEP 1 — Identity-weighted optimisation ##############

if run_step1
    println("\n" * "="^70)
    println("STEP 1: Identity-weighted PSO optimisation")
    println("="^70)

    theta_hat_1, _ = run_pso_optimization(;
        weight_matrix            = nothing,   # uses global Weight_matrix_custom
        skip_initial_beta_search = false,
        warm_start_params        = nothing,
        output_subfolder         = "step1",
        max_loop                 = 50
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

if run_step2
    println("\n" * "="^70)
    println("STEP 2: Building efficient weight matrix (K=$K_sim)")
    println("="^70)

    W_step3 = build_step3_weight_matrix(theta_hat_1, input_folder;
                                         K=K_sim, output_folder=output_folder,
                                         gamma_beta_only=false)

    beta_T_indices = vcat(1:N_beta, (N_beta + R_downstream + S + 2):length(theta_hat_1))# vcat(1:length(theta_hat_1))
    J_beta_T, J_beta_T_elast,_ = compute_jacobian(theta_hat_1;
                                                 param_indices = beta_T_indices,
                                                 output_folder = output_folder,
                                                 filename      = "jacobian_beta_T.npy")
    println("Step 2 complete. W_step3 saved.")
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
        max_loop                 = 50,
        gamma_beta_only          = false
    )

    NPZ.npzwrite(joinpath(output_folder, "step3", "theta_hat_2.npy"), theta_hat_2)
    println("Step 3 complete. θ̂_2 saved.")
end

############## POST-HOC ANALYSIS ##############
run_reporting(joinpath(output_folder, "step1"), 28; u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS)


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
intermediate_derivative = Float64[]; productivity = Float64[]
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
end
df = DataFrame(SIREN=sirens, A129=sectors, ze2010=ze2010,
               ze2010_downstream=ze2010_downstream, share=share,
               downstream_purchase=downstream_purchase,
               intermediate_derivative=intermediate_derivative,
               productivity=productivity)
Parquet.write_parquet(joinpath(folder, "suppliers.parquet"), df)

println("\nPost-hoc analysis complete. Results saved to: $folder")





# Test cov matrix


using LinearAlgebra, Statistics, Printf

# Center M_sim (already saved to step2/M_sim.npy by build_step3_weight_matrix)
M_sim = NPZ.npzread(joinpath(output_folder, "step2", "M_sim.npy"))
Mc = M_sim .- mean(M_sim, dims=1)
K, N_moments = size(M_sim)

# Per-moment standard deviation
sds = vec(std(M_sim, dims=1))

# Block-by-block diagnostic
for (k, name) in enumerate(BLOCK_NAMES)
    rng = BLOCK_RANGES[k]
    isempty(rng) && continue

    block = Mc[:, rng]
    sv = svdvals(block)

    @printf("%-10s  size=%4d  sd: min=%.2e  max=%.2e  median=%.2e  |  sv: max=%.2e  min=%.2e  ratio=%.2e\n",
            name, length(rng),
            minimum(sds[rng]), maximum(sds[rng]), median(sds[rng]),
            maximum(sv), minimum(sv), maximum(sv)/max(minimum(sv), 1e-300))
end

# Full-matrix singular values: tail behaviour
sv_full = svdvals(Mc)
println("\nTop 5 sv: ", round.(sv_full[1:5], sigdigits=3))
println("Bottom 10 sv: ", round.(sv_full[end-9:end], sigdigits=3))
println("Number below 1e-12 * sv_max: ", count(sv_full .< 1e-12 * sv_full[1]))

Sigma_data = NPZ.npzread(joinpath(output_folder, "step2", "Sigma_data.npy"))
Sigma_sim  = NPZ.npzread(joinpath(output_folder, "step2", "Sigma_sim.npy"))
Omega      = NPZ.npzread(joinpath(output_folder, "step2", "Omega.npy"))

for (name, M) in [("Sigma_data", Sigma_data), ("Sigma_sim", Sigma_sim), ("Omega", Omega)]
    sv = svdvals(Symmetric(M))
    tol = sv[1] * size(M,1) * eps()
    r = count(sv .> tol)
    @printf("%-10s  size=%d  rank=%d  sv_max=%.2e  sv_min=%.2e  cond=%.2e  rel_tol=%.2e\n",
            name, size(M,1), r, sv[1], sv[end], sv[1]/max(sv[end], 1e-300), tol)
end

# How many moments have ZERO contribution from Sigma_data?
zero_data_diag = findall(diag(Sigma_data) .< 1e-15)
println("\nMoments with Σ_data diagonal ≈ 0: ", length(zero_data_diag), " / ", size(Sigma_data,1))
println("Their indices: ", zero_data_diag)

println("K = ", K)
println("N_moments = ", N_moments)
println("Theoretical max rank of Sigma_sim: ", min(K-1, N_moments))

F = eigen(Symmetric(Omega))
λ = F.values
V = F.vectors

# Bottom 5 eigenvectors
for i in 1:5
    v = V[:, i]
    println("\nEigenvalue $i: ", round(λ[i], sigdigits=3))
    # Block decomposition of the eigenvector
    for (k, name) in enumerate(BLOCK_NAMES)
        rng = BLOCK_RANGES[k]
        isempty(rng) && continue
        mass = sum(v[rng].^2)
        @printf("  %-10s  ‖v_block‖² = %.4f\n", name, mass)
    end
end

w_gamma = NPZ.npzread(joinpath(input_folder, "w_gamma.npy"))
w_beta  = NPZ.npzread(joinpath(input_folder, "w_beta.npy"))

for (name, M) in [("w_gamma", w_gamma), ("w_beta", w_beta)]
    sv = svdvals(Symmetric(M))
    @printf("%-10s  size=%d  sv_max=%.2e  sv_min=%.2e  cond=%.2e\n",
            name, size(M,1), sv[1], sv[end], sv[1]/max(sv[end], 1e-300))
end

for (k, name) in enumerate(BLOCK_NAMES)
    rng = BLOCK_RANGES[k]
    isempty(rng) && continue
    d = diag(Omega)[rng]
    @printf("%-10s  diag: min=%.2e  max=%.2e  geomean=%.2e\n",
            name, minimum(d), maximum(d), exp(mean(log.(max.(d, 1e-300)))))
end

using Random
rng = MersenneTwister(1)
U, w = generate_mc_draws(N_rho, n_good, rng)
@assert size(U) == (N_rho, n_good)
@assert all(0 .< U .< 1)
@assert all(w .≈ 1/N_rho)
println("U has rank: ", rank(U))     # should equal min(N_rho, n_good)
println("U column 1 std: ", std(U[:,1]))  # should be ≈ sqrt(1/12) ≈ 0.289