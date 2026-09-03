##### Three-Step SMM — main.jl #####
# Entry point implementing the three-step efficient SMM estimator.
# Step 1: identity-weighted PSO → θ̂_1
# Step 2: build W_step3 = (Σ_data + Σ_sim)^{-1} at θ̂_1
# Step 3: efficient-weighted  PSO → θ̂_2, warm-started at θ̂_1
# ps aux | grep '[j]ulia' | awk '{print $2}' | xargs kill -9
# Optimizer backend (PSO / CMA-ES) is selected via OPTIMIZER_BACKEND; see optimizer.jl.

using Distributed
using Dates

available = 50

# Per-worker heap target. Distributed workers are separate OS processes with no shared
# memory, so peak RAM is n_workers × per-worker working set — and each worker sizes its
# GC threshold against TOTAL system memory, with no knowledge of its 48 siblings doing
# the same. The result is collective overcommit: every worker sits on its garbage at
# once. This run is churn-dominated (the IRLS kernel used to abandon tens of MB per
# iteration), so capping the heap makes each worker collect promptly instead of hoarding.
#
# Rule of thumb: ~70% of usable RAM divided by the worker count, and never below ~3× the
# live working set or the GC thrashes. Set WORKER_HEAP_HINT="" to disable.
worker_heap_hint = get(ENV, "WORKER_HEAP_HINT", "2G")
println("Using $available workers" *
        (isempty(worker_heap_hint) ? "" : " (heap-size-hint=$worker_heap_hint each)"))
addprocs(max(available - 1, 0);
         exeflags = isempty(worker_heap_hint) ? `` :
                    `--heap-size-hint=$worker_heap_hint`)
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
@everywhere include("model_analytical.jl")
@everywhere include("optimizers/pso_integration.jl")      # PSO backend
@everywhere include("optimizers/cmaes_integration.jl")    # CMA-ES backend
@everywhere include("optimizers/tiktak_integration.jl")   # TikTak (multistart) backend
@everywhere include("optimizer.jl")            # backend-neutral hub: optimize_stage, train_stage, run_optimization
@everywhere include("profiling.jl")            # T-profiling (invert_T_ge); inert unless profile_T=true
@everywhere include("test/run_untargeted_validation.jl")

############## Parse arguments ##############

industry = length(ARGS) >= 1 ? ARGS[1] : "auto"
n_coef   = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4
n_tau    = length(ARGS) >= 3 && !isempty(strip(ARGS[3])) ? parse(Int, ARGS[3]) : 1
K_sim    = length(ARGS) >= 4 && !isempty(strip(ARGS[4])) ? parse(Int, ARGS[4]) : 10000
# Draw method for the Fréchet inverse-CDF transform, used for the OPTIMISATION
# draws (U_DRAWS): :sobol (default) or :mc. Both carry flat weights, so both are
# unbiased for the min-coupled moments. Inference (Σ_sim + Jacobian) resamples with
# INFERENCE_DRAW_METHOD instead — see load_parameters.jl.
draw_method = length(ARGS) >= 5 && !isempty(strip(ARGS[5])) ? Symbol(strip(ARGS[5])) : :sobol
@assert draw_method in (:mc, :sobol) "draw method must be sobol|mc, got :$draw_method"
# Optimizer backend: :pso (default, legacy staged pattern), :cmaes (one joint
# CMA-ES per SMM step), or :tiktak (multistart Sobol + Nelder-Mead, one joint run
# per step). Read by load_parameters.jl into OPTIMIZER_BACKEND.
optimizer_backend = length(ARGS) >= 6 && !isempty(strip(ARGS[6])) ? Symbol(strip(ARGS[6])) : :pso
@assert optimizer_backend in (:pso, :cmaes, :tiktak) "optimizer must be pso|cmaes|tiktak, got :$optimizer_backend"

K = 20

run_step1 = true#true
run_step2 = true
run_step3 = true
run_step4 = true

# Reuse a previously-saved Jacobian instead of recomputing it (the expensive
# K×(2p+1) FD evaluations). When true, the Step-2 (step2/jacobian_all.npy) and
# Step-4 (step3/jacobian_all_step3.npy) calls load J + companions from disk if
# present, else fall back to computing. Set false to always recompute.
load_jacobian = false

# ── T-profiling (Design A, profiling.jl) ─────────────────────────────────────
# When true, T is NOT searched by the PSO: each particle's (Ω^L, Ω^s, A, α) head is
# mapped to T*(α,Ω,A) = invert_T_ge(...) inside the objective, collapsing the search
# by ~N_T_REDUCED dims. The γ_ls block becomes juste-identified (loss≈0); reg_coef is
# fit by (α,Ω,A) alone. Step-4 inference is UNCHANGED — it already keeps the full T
# columns, so T's CIs are preserved at the profiled point. Outputs go to a separate
# folder so the joint-search artifacts are never overwritten. false ⇒ byte-identical
# to the joint estimator. (SMM path only: the loss's reg_coef stays simulation-based.)
# 7th positional arg (run.sh --profile_T=true|false); default false.
profile_T = length(ARGS) >= 7 && !isempty(strip(ARGS[7])) ?
    (lowercase(strip(ARGS[7])) in ("true", "1", "yes")) : true
println("profile_T = $profile_T", profile_T ? " — T is profiled out of the PSO (invert_T_ge); outputs → *_profiled/" : "")

# Draw count for INFERENCE (Jacobian + Σ_sim), decoupled from the optimization draw
# count (K_sim/N_rho). Inference is simulation-noisy (the fixed-draw Jacobian and Σ_sim),
# so it benefits from many more draws than the search; picked up by load_parameters.jl
# into N_RHO_INFERENCE. 8th positional arg (run.sh --n_rho_inf=); default 10000 (= N_rho).
n_rho_inference = length(ARGS) >= 8 && !isempty(strip(ARGS[8])) ? parse(Int, strip(ARGS[8])) : 1000
@assert n_rho_inference >= 1 "n_rho_inference must be ≥ 1, got $n_rho_inference"

# Extensive-margin regression method: :cloglog (default — correct complementary-log-log
# link, distance coef = αθ) or :lpm (linear probability model). 9th positional arg
# (run.sh --reg=). Read by load_parameters.jl into REG_METHOD; the empirical target
# (reg_coef_cloglog_* vs reg_coef_*) is selected to match.
reg_method = length(ARGS) >= 9 && !isempty(strip(ARGS[9])) ? Symbol(strip(ARGS[9])) : :cloglog
@assert reg_method in (:lpm, :cloglog) "reg_method must be lpm|cloglog, got :$reg_method"

# Whether the no-supplier control group (filter==2, y=0 rows) enters the extensive-margin
# regression. false ⇒ supplier pairs only WITH the log-z size control (coupled). 10th
# positional arg (run.sh --controls=true|false). Read by load_parameters.jl into INCLUDE_CONTROL.
include_control = length(ARGS) >= 10 && !isempty(strip(ARGS[10])) ?
    (lowercase(strip(ARGS[10])) in ("true", "1", "yes")) : false

# ── Granularity + comparative-advantage level (documentation/plan_granular_aa.md) ──
# 11th positional arg (run.sh --granular=true|false), default FALSE: with
# --granular=false --ca_level=ze the estimator is the continuum ZE-level model exactly
# as it stood, which is a supported configuration, not a historical curiosity.
granular = length(ARGS) >= 11 && !isempty(strip(ARGS[11])) ?
    (lowercase(strip(ARGS[11])) in ("true", "1", "yes")) : false
# 12th positional arg (run.sh --ca_level=ze|aa), default :ze.
ca_level = length(ARGS) >= 12 && !isempty(strip(ARGS[12])) ? Symbol(strip(ARGS[12])) : :ze
@assert ca_level in (:ze, :aa) "ca_level must be ze|aa, got :$ca_level"

# ── Diagnostic: relax the variety-count lower bound to 1 ────────────────────
# 13th positional arg (run.sh --relax_n_lo=true|false), default FALSE. The model's
# bound N_LO[s] = ⌈N^obs_s / N_d⌉ is a theorem (finite_sample2.tex §3.3), and when it
# BINDS the count moment cannot be satisfied through N̂_s — under profile_T the residual
# is loaded onto α. Setting this true frees the bisection over [1, N_HI] to measure how
# much of α̂ that clamp is driving. Not the headline specification. See load_parameters.jl.
relax_n_lo = length(ARGS) >= 13 && !isempty(strip(ARGS[13])) ?
    (lowercase(strip(ARGS[13])) in ("true", "1", "yes")) : false

# Optional 2×2 noise-decomposition diagnostic (test-only). When false, behavior
# is byte-identical to today: nothing extra is computed or written.
run_2x2_test = true
n_quad       = 200

# The 2×2 test crosses a SIMULATED against an ANALYTICAL Jacobian, and the analytical
# path has no closed form for the granular count moment (block 6) — its extensive
# margin is the FKG-approximated continuum object, so it returns a 5-block vector
# against a 6-block MOMENT_MASK. Disable it here, at parse time, rather than let it
# fail after Step 2 has already spent the compute.
if run_2x2_test && granular
    @warn "run_2x2_test is not available under --granular=true: it needs an ANALYTICAL " *
          "Jacobian, and block 6 (the count moment Ḡ_s(0)) has no closed form on that " *
          "path. The test is skipped; estimation and inference are unaffected."
    run_2x2_test = false
end


if !(n_coef in [1, 4, 5])
    error("n_coef must be 1, 4 or 5, got: $n_coef")
end
if !(n_tau in [1, 4, 5])
    error("n_tau must be 1, 4 or 5, got: $n_tau")
end

println("Industry: $industry | n_coef (N_REG): $n_coef | n_tau (N_TAU): $n_tau | K_sim: $K_sim | draws: :$draw_method | optimizer: :$optimizer_backend | reg: :$reg_method | controls: $include_control | granular: $granular | ca_level: :$ca_level | relax_n_lo: $relax_n_lo")

input_folder  = "./baseline_$industry"
# profile_T ⇒ isolate all step1..4 artifacts under a distinct tree (plan §6), so the
# joint-search reporting is never overwritten and the two estimators stay comparable.
# granular / :aa likewise get their own tree, so the legacy fit stays reproducible
# side by side (the V0 gate compares the two).
output_folder = "./reporting_$industry" * (profile_T ? "_profiled" : "") *
                (ca_level == :aa ? "_aa" : "") * (granular ? "_gran" : "") *
                (relax_n_lo && granular ? "_nlo1" : "") *
                "_$optimizer_backend"
mkpath(output_folder)

############## Load and distribute constants ##############
include("load_parameters.jl")
NPZ.npzwrite(joinpath(output_folder, "n_reg_coef.npy"), n_coef)
NPZ.npzwrite(joinpath(output_folder, "n_tau.npy"), n_tau)

############## Determine step to run ##############

step1_folder = joinpath(output_folder, "step1")
step2_W_path = joinpath(output_folder, "step2", "W_step3.npy")


############## STEP 1 — Identity-weighted optimisation ##############
# Full optimisation to fixe un-bootstrapable parameters. 


A_init = copy(emp_pi_r_full).^(1/abs(epsilon)) .* regional_wages[N_downstream_per_region .!= 0]
A_init ./= sum(A_init)
T_init_nz = vec(permutedims(T_rs_init))[T_MASK]   # s-major to match T_MASK
# New layout: [Ω^L | Ω^s | A | α(N_TAU) | T] — alpha is inserted between A and T
init_other_prefix = vcat([agg_labor_share], agg_industry_share, A_init)
warm_start = vcat(init_other_prefix, P_alpha, T_init_nz)


if run_step1
    println("\n" * "="^70)
    println("STEP 1: Identity-weighted optimisation (backend = :$optimizer_backend)")
    println("="^70)

    theta_hat_1, _ = run_optimization(;
        weight_matrix            = nothing,
        skip_initial_alpha_search = true,
        warm_start_params        = warm_start,
        output_subfolder         = "step1",
        max_loop                 = K,
        max_iter_initial         = 200,
        profile_T                = profile_T
    )

    NPZ.npzwrite(joinpath(output_folder, "step1", "theta_hat_1.npy"), theta_hat_1)
    println("Step 1 complete. θ̂_1 saved.")
else
    println("Step 1 skipped (resume).")
end

############## Load θ̂_1 ##############
#output_folder = "./reporting_gmm_$industry"
step1_dir = joinpath(output_folder, "step1")
isdir(step1_dir) || error(
    "θ̂_1 cannot be loaded: $(step1_dir) does not exist. run_step1 = $(run_step1); " *
    "a fresh output tree (a new --granular/--ca_level/--optimizer combination writes " *
    "to its own folder, here $(output_folder)) has no Step-1 artefacts to resume from. " *
    "Set run_step1 = true for the first run in this tree.")
step1_last = find_last_stage_folder(step1_dir)
theta_hat_1 = NPZ.npzread(joinpath(step1_last, "best_params.npy"))

if ndims(theta_hat_1) > 1
    theta_hat_1 = theta_hat_1[:, 1]
end
println("θ̂_1 loaded from: $step1_last")

############## STEP 2 — Build efficient weight matrix ##############
# W_step3 is the weight matrix for the second optimisation. 
# Second optimisation only estimate T_{sr} and \alpha_k. Therefore, the weight matrix is restricted to those parameters. 
# We also build the Jacobian 
#   - Used for inference. 
#   - Used to analyse if selecting only (T,\alpha) is going to have an important effect on other moments. 

if run_step2
    println("\n" * "="^70)
    println("STEP 2: Building efficient weight matrix (K=$K_sim)")
    println("="^70)

    W_step3 = build_step3_weight_matrix(theta_hat_1, input_folder;
                                         K=K_sim, output_folder=output_folder)
    #W_step3 = NPZ.npzread(step2_W_path)
    # Payload slots for the count-distribution Jacobians (Ḡ_s(K), K = 0..3): filled by
    # `compute_jacobian(G_curve_out=...)` from evaluations it already makes, and read by
    # `run_profiled_inference` for the delta-method bands on the untargeted panels.
    gcurve_free_step2  = Ref{Any}(nothing)
    gcurve_alpha_step2 = Ref{Any}(nothing)

    # Jacobian at θ̂_1 — identified parameters only
    J1, J1_elast, J1_sd, J1_elast_sd = compute_jacobian(
        theta_hat_1;
        param_indices = jacobian_param_indices,
        output_folder = output_folder,
        output_subdir = "step2",
        filename      = "jacobian_all.npy",
        K             = 50,
        step_rel      = 1e-2,
        base_seed     = 2_000_000,  # disjoint from Σ_sim seeds (1:K_sim) and step-3 (1_000_000)
        # N_s is calibrated (integer bisection) exactly as T is calibrated (Sinkhorn),
        # so the FULL Jacobian carries both: the S variety-count columns are appended
        # to the right, leaving every θ column index — and hence gb_cols below —
        # untouched.
        append_N_s    = GRANULAR,
        N_s_base_seed = 7_100_000,   # disjoint from run_profiled_inference's 7_000_000
        # Differentiate the untargeted count distribution Ḡ_s(K) alongside the moments.
        # Free: it rides the SAME evaluations and is split off before anything is saved,
        # so `jacobian_all*.npy` and every consumer that slices by moment index are
        # unchanged. Needed for the delta-method bands on the K ≥ 1 panels.
        G_curve_out   = gcurve_free_step2,
        load_existing = load_jacobian
    )

    # Inference at θ̂_1 using efficient weight W_step3 and Ω from step2/
    Omega_step2 = NPZ.npzread(joinpath(output_folder, "step2", "Omega.npy"))
    _, sim_moments_1 = full_SMM(theta_hat_1; u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS)
    sim_vec_1 = moments_to_vec(sim_moments_1)
    emp_vec   = vec(empirical_moments)

    gb_indices = inference_moment_indices()   # β then γ (then G0 under GRANULAR)
    gb_block_ranges, gb_block_names = inference_block_layout()

    # Restrict Jacobian columns to α+T (the only params β+γ moments identify)
    alpha_T_start = 1 + S + R_downstream + 1                       # first α (trade-cost) raw index
    gb_cols      = findall(i -> i >= alpha_T_start, jacobian_param_indices)
    gb_param_idx = jacobian_param_indices[gb_cols]

    n_alpha_labels_s2 = count(l -> startswith(l, "alpha"), PARAM_LABELS[gb_cols])
    @assert n_alpha_labels_s2 == N_TAU "Step-2 inference: expected $N_TAU α labels in gb_cols, found $n_alpha_labels_s2"

    J_gb       = J1[gb_indices, gb_cols]                          # β+γ rows × α+T cols
    sim_vec_gb = sim_vec_1[gb_indices]
    emp_vec_gb = emp_vec[gb_indices]
    Weight_matrix_inference = Weight_matrix_custom[gb_indices, gb_indices]

    # Non-inferred head params (Ω^L, Ω^s, A): shown in the report for reference only.
    head_cols   = findall(i -> i < alpha_T_start, jacobian_param_indices)
    head_labels = PARAM_LABELS[head_cols]
    head_values = theta_hat_1[jacobian_param_indices[head_cols]]

    # T-identification eigen-screen (diagnostic, print-only) at θ̂_1
    screen_T_identification(theta_hat_1;
        J            = J_gb,
        W            = Matrix(Weight_matrix_custom[gb_indices, gb_indices]),
        param_labels = PARAM_LABELS[gb_cols],
        label        = "SMM step2 θ̂_1")

    if profile_T
        # ── Profiling path ────────────────────────────────────────────────────
        # Only α is perturbed and T follows via the Sinkhorn image T*(α,Ω,A): the
        # DIRECT profiled Jacobian dm/dα (compute_jacobian(profile_T=true) over the α
        # columns only). α CI runs on that reduced Jacobian; T CI is the correlated
        # α+γ delta method (γ = emp_gamma_ls Sinkhorn target, its data noise Σ_data).
        # Written under step2/inference/.
        alpha_raw_indices = gb_param_idx[findall(l -> startswith(l, "alpha"), PARAM_LABELS[gb_cols])]
        Jp1, _, _, _ = compute_jacobian(
            theta_hat_1;
            param_indices = alpha_raw_indices,
            profile_T     = true,
            output_folder = output_folder,
            output_subdir = "step2",
            filename      = "jacobian_profiled_alpha.npy",
            K             = 50,
            step_rel      = 1e-2,
            base_seed     = 6_000_000,  # disjoint from J1 (2e6), Σ_sim (1:K_sim), step3 (1e6)
            G_curve_out   = gcurve_alpha_step2,
            load_existing = load_jacobian)
        Sigma_data_s2 = NPZ.npzread(joinpath(output_folder, "step2", "Sigma_data.npy"))
        run_profiled_inference(
            theta_hat_1, Jp1, gb_indices, gb_param_idx,
            Matrix(Weight_matrix_inference), Omega_step2, Sigma_data_s2,
            emp_vec_gb, sim_vec_gb;
            output_folder    = joinpath(output_folder, "step2"),
            industry         = industry,
            K_sim            = K_sim,
            gb_block_ranges  = gb_block_ranges,
            gb_block_names   = gb_block_names,
            gamma_ref_map    = GAMMA_REF_MAP,
            param_labels_gb  = PARAM_LABELS[gb_cols],
            moment_labels_gb = MOMENT_LABELS[gb_indices],
            head_labels      = head_labels,
            head_values      = head_values,
            J_free_gb        = J_gb,          # free-parameter FD Jacobian ⇒ T's se_fd
            # Count-distribution Jacobians. The free one is sliced to the SAME gb
            # columns as J_free_gb so `T_pos` indexes both identically.
            G_curve_alpha    = gcurve_alpha_step2[],
            G_curve_free     = gcurve_free_step2[] === nothing ? nothing :
                               gcurve_free_step2[][:, gb_cols],
            N_s_base_seed    = 7_000_000)
    else
        inf_res_1 = compute_smm_inference(
             theta_hat_1, J_gb, Weight_matrix_inference, Omega_step2;
             param_indices         = gb_param_idx,
             empirical_moments_vec = emp_vec_gb,
             simulated_moments_vec = sim_vec_gb,
             output_folder         = joinpath(output_folder, "step2"),
             industry              = industry,
             K_sim                 = K_sim,
             block_ranges          = gb_block_ranges,
             block_names           = gb_block_names,
             gamma_ref_map   = GAMMA_REF_MAP,
             param_labels  = PARAM_LABELS[gb_cols],   # NEW: names for active params (cols of J)
             moment_labels = MOMENT_LABELS[gb_indices],    # NEW: names for kept moments (rows of J)
             display_labels = head_labels,   # NEW: non-inferred head params (value only)
             display_values = head_values
        )

        # ── Delta-method T CIs at θ̂_1 (joint estimator): Sinkhorn-pinned counterfactual
        # propagating Var(α̂_1) through ∂T*/∂α alongside the joint (T-as-free) CIs.
        compute_T_delta_inference(
            theta_hat_1, inf_res_1, gb_param_idx, PARAM_LABELS[gb_cols];
            output_folder = joinpath(output_folder, "step2"),
            industry      = industry)
    end

    # ── Optional 2×2 noise-decomposition test (isolated; off by default) ──────
    if run_2x2_test
        J_ana, _, _, _ = compute_jacobian(
            theta_hat_1;
            param_indices = jacobian_param_indices,
            output_folder = output_folder,
            output_subdir = "step2",
            filename      = "jacobian_2x2_ana.npy",
            K             = 1,
            step_rel      = 1e-2,
            base_seed     = 3_000_000,
            analytical    = true,
            analytical_ad = true,   # exact closed-form AD ⇒ the Jacobian channel is FD-noise-free
            ad_validate   = true,
            n_quad        = n_quad)
        J_ana_gb   = J_ana[gb_indices, gb_cols]
        Sigma_data = NPZ.npzread(joinpath(output_folder, "step2", "Sigma_data.npy"))
        Sigma_sim  = NPZ.npzread(joinpath(output_folder, "step2", "Sigma_sim.npy"))
        run_2x2_inference_test(
            theta_hat_1, J_gb, J_ana_gb, gb_param_idx,
            emp_vec_gb, sim_vec_gb, Sigma_data, Sigma_sim, GAMMA_REF_MAP,
            gb_block_ranges, gb_block_names, PARAM_LABELS[gb_cols],
            MOMENT_LABELS[gb_indices],
            joinpath(output_folder, "step2", "inference_2x2_test");
            label="theta_1")
    end

    # ── Granular diagnostics at θ̂_1 (N̂_s, clamps, Ḡ_s(0) fit, b_logz vs −θ) ──
    report_granular(theta_hat_1, joinpath(output_folder, "step2");
                    industry=industry, label="theta_1")

    println("Step 2 complete. W_step3 and θ̂_1 inference saved.")
else
    println("Step 2 skipped (resume). Loading W_step3...")
    W_step3 = NPZ.npzread(step2_W_path)
end


############## STEP 3 — Efficient-weighted optimisation ##############


if run_step3
    println("\n" * "="^70)
    println("STEP 3: Efficient-weighted optimisation (backend = :$optimizer_backend)")
    println("="^70)

    # A_r, labor share, and industry share are fixed at θ̂_1.
    # Only α and T are optimised, using the gamma+beta-only weight matrix from step 2.
    theta_hat_2, _ = run_optimization(;
        weight_matrix            = W_step3,
        skip_initial_alpha_search = true,
        warm_start_params        = theta_hat_1,
        output_subfolder         = "step3",
        max_loop                 = K,
        gamma_beta_only          = true,
        moments_loss_gamma_beta  = true,
        max_iter_initial         = 200,
        profile_T                = profile_T
    )

    NPZ.npzwrite(joinpath(output_folder, "step3", "theta_hat_2.npy"), theta_hat_2)
    println("Step 3 complete. θ̂_2 saved.")
end



if run_step4


    ############## Load θ̂_1 ##############

    step3_last = joinpath(output_folder, "step3")
    theta_hat_2 = NPZ.npzread(joinpath(step3_last, "theta_hat_2.npy"))

    if ndims(theta_hat_1) > 1
        theta_hat_1 = theta_hat_1[:, 1]
    end

    # ── Jacobian at θ̂_2 — all parameters ────────────────────────────────────
    # Phase 3 (T-profiling): θ̂_2 carries the profiled T*(α̂,Ω̂,Â), and this Jacobian
    # perturbs EVERY column (T included). The α+T restriction below (gb_cols) then
    # keeps the full T columns, so the standard GMM inference returns T's joint CIs
    # at the profiled point — no delta method. Inference is identical to the joint
    # estimator; only the SEARCH that produced θ̂_2 differed.
    println("\nComputing Jacobian at θ̂_2 (base_seed=1_000_000 to avoid collision with Σ_sim seeds)...")
    gcurve_free_step3  = Ref{Any}(nothing)
    gcurve_alpha_step3 = Ref{Any}(nothing)

    J2, J2_elast, J2_sd, J2_elast_sd = compute_jacobian(
        theta_hat_2;
        param_indices = jacobian_param_indices,
        output_folder = output_folder,
        output_subdir = "step3",
        filename      = "jacobian_all_step3.npy",
        K             = 50,
        step_rel      = 1e-2,
        base_seed     = 1_000_000,
        append_N_s    = GRANULAR,    # see the Step-2 call: N_s is a parameter too
        N_s_base_seed = 7_200_000,
        # Differentiate the untargeted count distribution Ḡ_s(K) alongside the moments.
        # Free: it rides the SAME evaluations and is split off before anything is saved,
        # so `jacobian_all*.npy` and every consumer that slices by moment index are
        # unchanged. Needed for the delta-method bands on the K ≥ 1 panels.
        G_curve_out   = gcurve_free_step3,
        load_existing = load_jacobian
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
    W_step3 = NPZ.npzread(joinpath(output_folder, "step2", "W_step3.npy"))
    Omega_inf   = NPZ.npzread(joinpath(output_folder, "step2", "Omega.npy"))

    _, sim_moments_2 = full_SMM(theta_hat_2; u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS)
    sim_vec_2 = moments_to_vec(sim_moments_2)
    emp_vec   = vec(empirical_moments)

    # Restrict to β+γ moments (β then γ) for inference
    gb_indices = inference_moment_indices()
    gb_block_ranges, gb_block_names = inference_block_layout()

    # Restrict Jacobian columns to α+T (the only params β+γ moments identify)
    alpha_T_start = 1 + S + R_downstream + 1                       # first α (trade-cost) raw index
    gb_cols      = findall(i -> i >= alpha_T_start, jacobian_param_indices)
    gb_param_idx = jacobian_param_indices[gb_cols]

    # Correctness check: Σ_data leading β-block must be N_REG × N_REG (moments, not params).
    @assert size(Omega_inf, 1) == length(gb_indices) "Omega_inf size $(size(Omega_inf,1)) != n_gb=$(length(gb_indices)) (N_REG + n_γ" * (GRANULAR ? " + S" : "") * ")"
    # Correctness check: exactly N_TAU α labels must appear in the restricted param columns.
    n_alpha_labels = count(l -> startswith(l, "alpha"), PARAM_LABELS[gb_cols])
    @assert n_alpha_labels == N_TAU "Expected $N_TAU α labels in gb_cols, found $n_alpha_labels — alpha_T_start misaligned with N_TAU"

    J2_gb      = J2[gb_indices, gb_cols]                          # β+γ rows × α+T cols
    sim_vec_gb = sim_vec_2[gb_indices]
    emp_vec_gb = emp_vec[gb_indices]

    # Non-inferred head params (Ω^L, Ω^s, A): shown in the report for reference only.
    head_cols   = findall(i -> i < alpha_T_start, jacobian_param_indices)
    head_labels = PARAM_LABELS[head_cols]
    head_values = theta_hat_2[jacobian_param_indices[head_cols]]

    n_gb_moments = length(gb_indices)
    n_gb_params  = length(gb_param_idx)
    println("Step-3 inference: J2_gb is $(n_gb_moments)×$(n_gb_params) " *
            "($(n_gb_moments) β+γ moments × $(n_gb_params) α+T params). " *
            "df = $(n_gb_moments - n_gb_params)")

    # T-identification eigen-screen (diagnostic, print-only) at θ̂_2
    screen_T_identification(theta_hat_2;
        J            = J2_gb,
        W            = W_step3,
        param_labels = PARAM_LABELS[gb_cols],
        label        = "SMM step4 θ̂_2")

    if profile_T
        # ── Profiling path (Phase 3) ──────────────────────────────────────────
        # Only α is perturbed and T follows via the Sinkhorn image T*(α,Ω,A): the
        # DIRECT profiled Jacobian dm/dα (compute_jacobian(profile_T=true) over the α
        # columns). The γ_ls moments are juste-identified along T*(α), so this
        # isolates α on the reg_coef τ-channel — no ∂m/∂T-as-free-parameter is
        # formed. α CI runs on that reduced Jacobian; T CI is the correlated α+γ
        # delta method: propagates Var(α̂) AND the DATA noise of the γ_ls Sinkhorn
        # target (Σ_data), with their covariance.
        alpha_raw_indices = gb_param_idx[findall(l -> startswith(l, "alpha"), PARAM_LABELS[gb_cols])]
        Jp2, _, _, _ = compute_jacobian(
            theta_hat_2;
            param_indices = alpha_raw_indices,
            profile_T     = true,
            output_folder = output_folder,
            output_subdir = "step3",
            filename      = "jacobian_profiled_alpha_step3.npy",
            K             = 50,
            step_rel      = 1e-2,
            base_seed     = 7_000_000,  # disjoint from J2 (1e6), Σ_sim (1:K_sim), step2 (6e6)
            G_curve_out   = gcurve_alpha_step3,
            load_existing = load_jacobian)
        Sigma_data_s4 = NPZ.npzread(joinpath(output_folder, "step2", "Sigma_data.npy"))
        run_profiled_inference(
            theta_hat_2, Jp2, gb_indices, gb_param_idx,
            W_step3, Omega_inf, Sigma_data_s4, emp_vec_gb, sim_vec_gb;
            output_folder    = joinpath(output_folder, "step3"),
            industry         = industry,
            K_sim            = K_sim,
            gb_block_ranges  = gb_block_ranges,
            gb_block_names   = gb_block_names,
            gamma_ref_map    = GAMMA_REF_MAP,
            param_labels_gb  = PARAM_LABELS[gb_cols],
            moment_labels_gb = MOMENT_LABELS[gb_indices],
            head_labels      = head_labels,
            head_values      = head_values,
            J_free_gb        = J2_gb,         # free-parameter FD Jacobian ⇒ T's se_fd
            G_curve_alpha    = gcurve_alpha_step3[],
            G_curve_free     = gcurve_free_step3[] === nothing ? nothing :
                               gcurve_free_step3[][:, gb_cols],
            N_s_base_seed    = 7_100_000)
    else
        inf_res = compute_smm_inference(
            theta_hat_2, J2_gb, W_step3, Omega_inf;
            param_indices         = gb_param_idx,
            empirical_moments_vec = emp_vec_gb,
            simulated_moments_vec = sim_vec_gb,
            output_folder         = joinpath(output_folder, "step3"),
            industry              = industry,
            K_sim                 = K_sim,
            block_ranges          = gb_block_ranges,
            block_names           = gb_block_names,
            gamma_ref_map   = GAMMA_REF_MAP,
            param_labels  = PARAM_LABELS[gb_cols],   # NEW: names for active params (cols of J)
            moment_labels = MOMENT_LABELS[gb_indices],    # NEW: names for kept moments (rows of J)
            display_labels = head_labels,   # NEW: non-inferred head params (value only)
            display_values = head_values)

        # ── Delta-method T CIs (joint estimator): Sinkhorn-pinned counterfactual ──
        # Propagates Var(α̂) through ∂T*/∂α alongside the joint (T-as-free) CIs above.
        compute_T_delta_inference(
            theta_hat_2, inf_res, gb_param_idx, PARAM_LABELS[gb_cols];
            output_folder = joinpath(output_folder, "step3"),
            industry      = industry)
    end

    # ── Granular diagnostics at θ̂_2 ──────────────────────────────────────────
    report_granular(theta_hat_2, joinpath(output_folder, "step3");
                    industry=industry, label="theta_2")

    # ── Optional 2×2 noise-decomposition test (isolated; off by default) ──────
    if run_2x2_test
        J_ana, _, _, _ = compute_jacobian(
            theta_hat_2;
            param_indices = jacobian_param_indices,
            output_folder = output_folder,
            output_subdir = "step3",
            filename      = "jacobian_2x2_ana.npy",
            K             = 1,
            step_rel      = 1e-2,
            base_seed     = 4_000_000,
            analytical    = true,
            analytical_ad = true,   # exact closed-form AD ⇒ the Jacobian channel is FD-noise-free
            ad_validate   = true,
            n_quad        = n_quad)
        J_ana_gb   = J_ana[gb_indices, gb_cols]


        screen_T_identification(theta_hat_2;
            J            = J_ana_gb,
            W            = W_step3,
            param_labels = PARAM_LABELS[gb_cols],
            label        = "SMM step4 θ̂_2 ana")


        Sigma_data = NPZ.npzread(joinpath(output_folder, "step2", "Sigma_data.npy"))
        Sigma_sim  = NPZ.npzread(joinpath(output_folder, "step2", "Sigma_sim.npy"))
        run_2x2_inference_test(
            theta_hat_2, J2_gb, J_ana_gb, gb_param_idx,
            emp_vec_gb, sim_vec_gb, Sigma_data, Sigma_sim, GAMMA_REF_MAP,
            gb_block_ranges, gb_block_names, PARAM_LABELS[gb_cols],
            MOMENT_LABELS[gb_indices],
            joinpath(output_folder, "step3", "inference_2x2_test");
            label="theta_2")
    end

end




############## POST-HOC ANALYSIS ##############
run_reporting(joinpath(output_folder, "step1"), K; u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS,
              analytical=false)

# Same reporting artefacts at the efficient estimate θ̂_2, so the Python reporting can
# read the simulated moments at EITHER μ_1 (step1) or μ_2 (step3). Without this, only
# the Step-1 fit is on disk as a moment vector; θ̂_2's moments exist nowhere as an
# array. Guarded: a fresh tree resumed at step ≤ 2 has no step3 stage folders yet, and
# a reporting failure must never abort the run after the estimation itself succeeded.
if isdir(joinpath(output_folder, "step3"))
    try
        run_reporting(joinpath(output_folder, "step3"), K; u_draws=U_DRAWS,
                      sample_weights=SAMPLE_WEIGHTS, analytical=false)
    catch e
        @warn "Step-3 reporting skipped: $e"
    end
end


# ─────────────────────────────────────────────────────────────────────────────
# POST-HOC ANALYSIS — the firm-level economy, at BOTH estimates.
#
# Everything below is written into the STEP folder of the estimate it was computed
# at, so the firm-level artefacts share the tree structure of the moments and the
# inference:
#
#     <run>/step1/{suppliers.parquet, suppliers.npy, w_srd_r.npy}   at θ̂_1  (mu = 1)
#     <run>/step3/{suppliers.parquet, suppliers.npy, w_srd_r.npy}   at θ̂_2  (mu = 2)
#
# so the Python reporting selects between them with the same `mu` it uses for
# `best_simulated_moments.npy` and the inference. Previously this ran at θ̂_1 only
# and wrote to the run ROOT, so `mu = 2` had no firm-level economy on disk at all
# and `mu = 1`'s did not sit where the rest of that estimate's output does.
#
# `write_post_hoc` is a FUNCTION rather than a top-level block: besides being run
# twice, a function body is hard scope, so `siren_counter += 1` inside the loop is
# an ordinary local update — at top level that same line is soft scope and silently
# binds a fresh local instead of advancing the counter.
# ─────────────────────────────────────────────────────────────────────────────
function write_post_hoc(params::Vector{Float64}, dest::AbstractString; label::AbstractString="")
    mkpath(dest)
    println("\n" * "-"^70)
    println("Post-hoc firm-level artefacts $(isempty(label) ? "" : "at $label ")-> $dest")
    println("-"^70)

    network = solve_network(params, return_firm_level=true,
                            u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS)

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
    npzwrite(joinpath(dest, "w_srd_r.npy"), w_srd_r)

    N_rho_out = size(network.linkages_flat, 1)   # pool width under GRANULAR, N_rho otherwise
    suppliers = zeros(Bool, S, N_rho_out, R)
    for g in 1:n_good
        s = GOOD_S[g]; r = GOOD_R[g]
        for rho in 1:N_rho_out
            if network.linkages_flat[rho, g] > 0
                suppliers[s, rho, r] = true
            end
        end
    end
    npzwrite(joinpath(dest, "suppliers.npy"), suppliers)

    sirens = Int[]; sectors = Int[]; ze2010 = Int[]; ze2010_downstream = Int[]
    share = Float64[]; downstream_purchase = Float64[]
    intermediate_derivative = Float64[]; productivity = Float64[]; sample_weight_vec = Float64[]
    siren_map = Dict{Tuple{Int,Int,Int}, Int}()
    siren_counter = 0
    for g in 1:n_good
        l = GOOD_R[g]; s = GOOD_S[g]
        for rho in 1:N_rho_out
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
        push!(sample_weight_vec, network.sample_weights[rho, g])
    end
    df = DataFrame(SIREN=sirens, A129=sectors, ze2010=ze2010,
                   ze2010_downstream=ze2010_downstream, share=share,
                   downstream_purchase=downstream_purchase,
                   intermediate_derivative=intermediate_derivative,
                   productivity=productivity, sample_weight=sample_weight_vec)
    Parquet.write_parquet(joinpath(dest, "suppliers.parquet"), df)

    println("  $(nrow(df)) linkages, $(siren_counter) simulated firms, " *
            "$(N_rho_out) varieties per cell")
    return df
end


println("\n" * "="^70)
println("POST-HOC ANALYSIS FOR $(uppercase(industry))")
println("="^70)

# θ̂_1 is the last Step-1 stage; θ̂_2 is Step 3's. Both are read from disk here
# rather than from the in-scope variables, so the block works whichever
# run_step{1,3,4} flags this invocation had on.
post_hoc_targets = Tuple{String,String,String}[]

step1_stage = isdir(joinpath(output_folder, "step1")) ?
    find_last_stage_folder(joinpath(output_folder, "step1")) : ""
if !isempty(step1_stage) && isfile(joinpath(step1_stage, "best_params.npy"))
    push!(post_hoc_targets,
          (joinpath(step1_stage, "best_params.npy"), joinpath(output_folder, "step1"), "θ̂_1 (mu = 1)"))
else
    @warn "no Step-1 best_params.npy under $(joinpath(output_folder, "step1")) — " *
          "skipping the mu = 1 post-hoc artefacts."
end

theta2_path = joinpath(output_folder, "step3", "theta_hat_2.npy")
if isfile(theta2_path)
    push!(post_hoc_targets, (theta2_path, joinpath(output_folder, "step3"), "θ̂_2 (mu = 2)"))
else
    @warn "no $theta2_path — skipping the mu = 2 post-hoc artefacts (Step 3 has not run in this tree)."
end

for (param_path, dest, label) in post_hoc_targets
    params = NPZ.npzread(param_path)
    if ndims(params) > 1
        params = params[:, 1]
    end
    println("\nLoading parameters from: $param_path")
    # Never let one estimate's post-hoc failure lose the other's, or the run.
    try
        write_post_hoc(Vector{Float64}(params), dest; label=label)
    catch e
        @warn "post-hoc analysis failed for $label: $e"
    end
end

# ── Untargeted Table-2 validation: DISABLED ──────────────────────────────────
# `validate_table2_all_models` exists only to produce `simulated_panel_unified`
# and `regional_sales_unified` (plus its own summary), and those panels are no
# longer used by the reporting. Uncomment the three lines to restore them; the
# `output_folder` kwarg must stay, since the run tree is named from the flags.
#
# best_params_1 = NPZ.npzread(joinpath(step1_stage, "best_params.npy"))
# results_unified = validate_table2_all_models(
#     ndims(best_params_1) > 1 ? best_params_1[:, 1] : best_params_1, industry;
#     T_periods=36, time_fe_mode="resample", output_folder=output_folder)
# Parquet.write_parquet(joinpath(output_folder, "simulated_panel_unified.parquet"), results_unified["panel_df"])
# Parquet.write_parquet(joinpath(output_folder, "regional_sales_unified.parquet"), results_unified["regional_sales_df"])

println("\nPost-hoc analysis complete. Results saved under: $output_folder")
