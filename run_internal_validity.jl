##### Internal-validity driver — run_internal_validity.jl #####
#
# Monte-Carlo internal-validity exercise for the three-step SMM estimator.
# Implements the plan in three parts on top of the EXISTING estimator code
# (model_CP.jl / tools.jl / pso_integration.jl), changing only the *targets*:
#
#   Section 0  Synthetic-truth hook. After load_parameters.jl finishes (so all
#              masks / T_REF_REGION / MOMENT_MASK / gamma_threshold are built
#              from the REAL baseline), replace ONLY the target globals
#              `empirical_moments` (← masked m(θ₀)) and `reg_coef` (← θ₀'s
#              reg_coef block). emp_gamma_ls / X_rs / emp_pi_r_full stay REAL so
#              every mask / reference / threshold is frozen. The override is an
#              in-place mutation of the (const) arrays on all workers — no const
#              rebinding, so already-compiled methods see the new values.
#
#   Section 1  Build θ₀ and diagnose it at the truth (ESS + screen_T_identification).
#
#   Experiment 1  Full-vector point recovery from 10 perturbed starts (full
#                 three-step per start). Generated with one draw seed, estimated
#                 with the production draws (a different seed) so the loss floor
#                 is genuine simulation error, not the trivial self-recovery 0.
#
#   Experiment 2  α+T coverage. A, Ωᴸ, Ωˢ fixed at θ₀; M reps draw
#                 m_b = m(θ₀)+ε_b with ε_b ~ N(0, Σ_data) on the β+γ block,
#                 re-estimate α+T only via the existing gamma_beta_only Step-3
#                 path warm-started at θ₀, then Step-4 inference. Reports
#                 coverage / bias / SD(θ̂)/mean(SE) per α+T parameter.
#
# Outputs land under  internal_validity_<industry>/ .
#
# IMPORTANT: this script was authored without a local Julia/data environment to
# execute it. It mirrors main.jl's call patterns exactly. Run with SMOKE_TEST=true
# first (tiny M / tiny PSO) to validate the wiring end-to-end before the full run.
# ─────────────────────────────────────────────────────────────────────────────

using Distributed
using Dates

available = 50
println("Using $available workers")
addprocs(max(available - 1, 0))

using Random
const BASE_SEED = 1234
Random.seed!(BASE_SEED)
@everywhere using Random
@everywhere Random.seed!($BASE_SEED)

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
@everywhere include("pso_integration.jl")      # PSO backend
@everywhere include("cmaes_integration.jl")    # CMA-ES backend
@everywhere include("optimizer.jl")            # backend-neutral hub (defines run_optimization / run_pso_optimization)
@everywhere include("run_untargeted_validation.jl")

############## Parse arguments ##############
# Usage:
#   julia run_internal_validity.jl <industry> <n_coef> <n_tau> [α₀...]
# e.g.
#   julia run_internal_validity.jl aero 4 1 0.5            # power-law, α₀ = 0.5
#   julia run_internal_validity.jl aero 4 4 0.2 0.4 0.6 0.8

industry = length(ARGS) >= 1 ? ARGS[1] : "aero"
n_coef   = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
n_tau    = length(ARGS) >= 3 && !isempty(strip(ARGS[3])) ? parse(Int, ARGS[3]) : 1

if !(n_coef in [1, 4, 5]); error("n_coef must be 1, 4 or 5, got: $n_coef"); end
if !(n_tau  in [1, 4, 5]); error("n_tau must be 1, 4 or 5, got: $n_tau"); end

# α₀ (trade-cost truth). CLI args beyond the first three; else a sensible default.
alpha0_cli = length(ARGS) >= 4 ? [parse(Float64, a) for a in ARGS[4:end]] : Float64[]

println("Industry: $industry | n_coef (N_REG): $n_coef | n_tau (N_TAU): $n_tau")

input_folder  = "./baseline_$industry"
# `output_folder` is reassigned per experiment/start because run_pso_optimization
# and generate_report read it as a (mutable) global. It is NOT const.
output_folder = "./internal_validity_$industry"
mkpath(output_folder)
const IV_ROOT = output_folder

############## Configuration ##############

# Smoke test: tiny everything so the full pipeline can be validated in minutes.
const SMOKE_TEST = true

const RUN_EXP1 = true     # 10-start full-vector point recovery
const RUN_EXP2 = true     # α+T coverage Monte Carlo

# Draw seeds. The truth m(θ₀) is generated with `randomise=true` draws seeded
# DIFFERENTLY from the production U_DRAWS (which are deterministic,
# MersenneTwister(g), see generate_stratified_draws). This guarantees the
# estimator never sees the exact draws that produced its target.
const TRUTH_DRAW_SEED = 999_000

# Experiment 1 — full three-step PSO budget (per start). Production-grade so the
# recovery test exercises the real optimiser; only 10 starts so it is affordable.
exp1_n_starts       = SMOKE_TEST ? 2   : 10
exp1_n_particles    = SMOKE_TEST ? 12  : 100
exp1_max_iter_init  = SMOKE_TEST ? 10  : 200
exp1_max_iter_stage = SMOKE_TEST ? 5   : 50
exp1_max_loop       = SMOKE_TEST ? 2   : 10
exp1_K_sim          = SMOKE_TEST ? 4   : 50    # Σ_sim re-seeds for W_step3 / Ω
exp1_K_jac          = SMOKE_TEST ? 2   : 50    # Jacobian replications at θ̂

# Experiment 2 — coverage Monte Carlo. Light-polish per-rep PSO (warm-started at
# the truth θ₀ → only a local polish is needed). Reps run SEQUENTIALLY (the 49
# workers are consumed inside each PSO iteration's pmap), so the per-rep budget
# is the binding cost. M=200 confirmed feasible only with this reduced budget.
exp2_M              = SMOKE_TEST ? 4   : 200
exp2_n_particles    = SMOKE_TEST ? 12  : 40
exp2_max_iter_init  = SMOKE_TEST ? 8   : 60
exp2_max_iter_stage = SMOKE_TEST ? 4   : 30
exp2_max_loop       = SMOKE_TEST ? 1   : 3
exp2_K_sim          = SMOKE_TEST ? 4   : 50    # Σ_sim re-seeds (W/Ω built ONCE at θ₀, reused)
exp2_K_jac          = SMOKE_TEST ? 2   : 20    # per-rep Jacobian replications
const EXP2_NOISE_SEED = 555_000

############## Load real baseline constants + masks ##############
include("load_parameters.jl")
NPZ.npzwrite(joinpath(IV_ROOT, "n_reg_coef.npy"), n_coef)

# ─────────────────────────────────────────────────────────────────────────────
# Layout / index helpers (raw parameter vector, see CLAUDE.md "Two-constant
# parameter layout" and "T flat-indexing convention").
#   [ Ω^L(1) | Ω^s(S) | A(R_downstream) | α(N_TAU) | T_active(s-major) ]
# ─────────────────────────────────────────────────────────────────────────────
const ALPHA_T_START   = 1 + S + R_downstream + 1            # first α (trade-cost) raw index
const T_PARAM_OFFSET = 1 + S + R_downstream + N_TAU        # count before first T
const T_FLAT_ACTIVE  = findall(T_MASK)                     # s-major active flat positions
@assert length(T_FLAT_ACTIVE) == sum(T_MASK)

# α+T columns within jacobian_param_indices (the only params β+γ moments identify)
const GB_COLS      = findall(i -> i >= ALPHA_T_START, jacobian_param_indices)
const GB_PARAM_IDX = jacobian_param_indices[GB_COLS]
const GB_LABELS    = PARAM_LABELS[GB_COLS]
const GB_INDICES   = vcat(collect(BLOCK_RANGES[4]), collect(BLOCK_RANGES[5]))   # β then γ moments
const N_GB         = length(GB_INDICES)
const N_REG_LOC    = length(BLOCK_RANGES[4])
const N_GAM_LOC    = length(BLOCK_RANGES[5])
const GB_BLOCK_RANGES = (1:N_REG_LOC, (N_REG_LOC + 1):(N_REG_LOC + N_GAM_LOC))
const GB_BLOCK_NAMES  = ("reg_coef", "gamma_ls")

# For each α+T identified raw param position, the per-sector reference T raw value
# in a parameter vector — used to put raw T onto the T_ref=1 (identified) gauge.
# α params map to a gauge factor of 1.0 (no normalization).
"""
    gauge_factors(theta) -> Vector (length N_GB params)

Per α+T parameter, the scale that `unpack_params` divides out:
 - α  → 1.0
 - T_sr → theta[ref-raw-index(sector s)]  (so T_sr / factor == normalized T)
"""
function gauge_factors(theta::Vector{Float64})
    f = ones(Float64, length(GB_PARAM_IDX))
    for (k, p) in enumerate(GB_PARAM_IDX)
        if p >= T_PARAM_OFFSET + 1                  # this is a T param
            t_rank   = p - T_PARAM_OFFSET           # 1-based among active T (s-major)
            flat_pos = T_FLAT_ACTIVE[t_rank]
            s        = ((flat_pos - 1) ÷ R) + 1
            ref_r    = T_REF_REGION[s]
            ref_flat = (s - 1) * R + ref_r
            ref_rank = count(@view T_MASK[1:ref_flat])
            ref_raw  = theta[T_PARAM_OFFSET + ref_rank]
            f[k]     = ref_raw
        end
    end
    return f
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — build θ₀
# ─────────────────────────────────────────────────────────────────────────────
function build_theta0()
    # A_gravity: the analytical inversion used in Stage 0 of the estimator.
    A_grav = copy(emp_pi_r_full) .^ (1 / abs(epsilon)) .* regional_wages[N_downstream_per_region .!= 0]
    A_grav ./= sum(A_grav)

    # α₀: CLI vector if supplied, else a monotone default. Enforce α₁ ≤ … ≤ α_K.
    if isempty(alpha0_cli)
        alpha0 = N_TAU == 1 ? [0.5] : collect(range(0.2, 0.8; length = N_TAU))
    else
        length(alpha0_cli) == N_TAU ||
            error("α₀ has length $(length(alpha0_cli)) but N_TAU=$N_TAU")
        alpha0 = copy(alpha0_cli)
    end
    issorted(alpha0) || error("α₀ must satisfy α₁ ≤ … ≤ α_K (got $alpha0)")

    # T_gravity active, s-major (identical to Stage-0 T_init_nz).
    T_grav_nz = vec(permutedims(T_rs_init))[T_MASK]

    theta0 = vcat([agg_labor_share], agg_industry_share, A_grav, alpha0, T_grav_nz)
    @assert length(theta0) == 1 + S + R_downstream + N_TAU + sum(T_MASK)
    return theta0, alpha0
end

# Flatten a SMM moment tuple → masked moment vector (MOMENT_MASK order).
masked_moments(sim) = vcat([vec(sim[i]) for i in 1:5]...)[MOMENT_MASK]

"""
    set_synthetic_targets!(masked_m)

In-place override of the target globals on master + all workers:
 - `empirical_moments[1, :] .= masked_m`
 - `reg_coef            .= masked_m[BLOCK_RANGES[4]]`
Both are const ARRAYS, mutated (not rebound), so every worker's compiled
`full_SMM` / Stage-0 immediately sees the new targets.
"""
function set_synthetic_targets!(masked_m::Vector{Float64})
    @assert length(masked_m) == length(empirical_moments)
    reg_block = masked_m[BLOCK_RANGES[4]]
    @everywhere begin
        empirical_moments .= reshape($masked_m, 1, :)   # in-place: 1×N const matrix
        reg_coef          .= $reg_block                 # in-place: length-N_REG const vector
    end
    return nothing
end

# Normalized (gauge-fixed) raw parameter vector, in jacobian layout order.
function normalized_raw(theta::Vector{Float64})
    OmegaL, Omega_s, A, alpha, Tvec = unpack_params(theta)   # Tvec = vec(T_mat), normalized
    Tmat = reshape(Tvec, S, R)
    T_active_smajor = vec(permutedims(Tmat))[T_MASK]
    return vcat(OmegaL, Omega_s, A, alpha, T_active_smajor)
end
identified(theta::Vector{Float64}) = normalized_raw(theta)[jacobian_param_indices]

# PSD matrix square root (eigen, negatives floored to 0).
function psd_sqrt(Sig::AbstractMatrix)
    Ssym = Symmetric((Matrix(Sig) .+ Matrix(Sig)') ./ 2)
    F = eigen(Ssym)
    return F.vectors * Diagonal(sqrt.(max.(F.values, 0.0))) * F.vectors'
end

# ─────────────────────────────────────────────────────────────────────────────
# Build θ₀, the truth draws, and the (fixed) synthetic targets m(θ₀).
# ─────────────────────────────────────────────────────────────────────────────
println("\n" * "="^70)
println("SECTION 0/1: synthetic truth θ₀ and target moments")
println("="^70)

theta0, alpha0 = build_theta0()
NPZ.npzwrite(joinpath(IV_ROOT, "theta_0.npy"), theta0)
println("θ₀ built: length $(length(theta0)), α₀ = $(round.(alpha0, digits=4))")

# Truth draws — independent of production U_DRAWS.
truth_U, truth_W = generate_stratified_draws(N_rho, n_good;
                                             randomise = true,
                                             rng = MersenneTwister(TRUTH_DRAW_SEED),
                                             verbose = true)

sim0       = SMM(theta0; u_draws = truth_U, sample_weights = truth_W)
masked_m0  = masked_moments(sim0)
NPZ.npzwrite(joinpath(IV_ROOT, "masked_m_theta0.npy"), masked_m0)

# Apply the override (fixed target for Section 1 + Experiment 1).
set_synthetic_targets!(masked_m0)
println("Targets overridden: empirical_moments ← masked m(θ₀); reg_coef ← θ₀ reg block.")

# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — diagnose θ₀ (ESS + T-identification screen at the truth)
# ─────────────────────────────────────────────────────────────────────────────
println("\nDiagnosing identification at θ₀ ...")

# ESS of the production draws (what the estimator will actually use).
min_ess = Inf
for g in 1:n_good
    s2 = sum(@view(SAMPLE_WEIGHTS[:, g]) .^ 2)
    global min_ess = min(min_ess, 1.0 / s2)
end

# Jacobian at θ₀ (β+γ rows × α+T cols), and the W-weighted T screen.
J0, _, _, _ = compute_jacobian(theta0;
    param_indices = jacobian_param_indices,
    output_folder = IV_ROOT, output_subdir = "diag_theta0",
    filename = "jacobian_theta0.npy",
    K = exp1_K_jac, step_rel = 1e-4, step_abs = 1e-9, base_seed = 7_000_000)
J0_gb = J0[GB_INDICES, GB_COLS]
W0    = Matrix(Weight_matrix_custom[GB_INDICES, GB_INDICES])

screen_diag_path = joinpath(IV_ROOT, "ess_and_screen_at_theta0.txt")
open(screen_diag_path, "w") do io
    println(io, "Internal-validity diagnostics at θ₀  ($(now()))")
    println(io, "industry=$industry  N_REG=$N_REG  N_TAU=$N_TAU  n_good=$n_good  N_rho=$N_rho")
    @printf(io, "min_g ESS_g (production draws) = %.1f / %d  (frac %.3f)\n",
            min_ess, N_rho, min_ess / N_rho)
    HTT = Symmetric(J0_gb' * W0 * J0_gb)
    # T-only sub-block: columns of GB whose label starts with "T["
    tsel = findall(l -> startswith(l, "T["), GB_LABELS)
    if !isempty(tsel)
        ev = eigvals(Symmetric(HTT[tsel, tsel]))
        @printf(io, "H[T,T] λ_min = %.6e   λ_max = %.6e   cond = %.3e\n",
                minimum(ev), maximum(ev), maximum(ev) / max(minimum(ev), 1e-300))
    else
        println(io, "H[T,T]: no T columns in α+T set.")
    end
    println(io, "\n(See stdout for full per-sector screen_T_identification output.)")
end
println("min_g ESS_g = $(round(min_ess, digits=1)) / $N_rho")
screen_T_identification(theta0; J = J0_gb, W = W0,
    param_labels = GB_LABELS, label = "IV θ₀")
println("Section-1 diagnostics written to $screen_diag_path")

# ─────────────────────────────────────────────────────────────────────────────
# A single full three-step run, seeded, into a given output folder.
# Returns (θ̂_2, se_sw_alphaT, ci95_alphaT) for the α+T identified params.
# ─────────────────────────────────────────────────────────────────────────────
function run_full_three_step(start_theta::Vector{Float64}, seed::Int, subfolder::String)
    global output_folder
    output_folder = joinpath(IV_ROOT, subfolder)
    mkpath(output_folder)
    Random.seed!(seed)
    @everywhere Random.seed!($seed)

    # Step 1: full optimisation, seeded at the perturbed start (α taken from the
    # start, so no Stage-0 α search). Never warm-started AT θ₀.
    theta_hat_1, _ = run_pso_optimization(;
        weight_matrix            = nothing,
        skip_initial_alpha_search = true,
        warm_start_params        = start_theta,
        output_subfolder         = "step1",
        n_particles              = exp1_n_particles,
        max_iter_initial         = exp1_max_iter_init,
        max_iter_stage           = exp1_max_iter_stage,
        max_loop                 = exp1_max_loop)

    step1_last  = find_last_stage_folder(joinpath(output_folder, "step1"))
    theta_hat_1 = NPZ.npzread(joinpath(step1_last, "best_params.npy"))
    ndims(theta_hat_1) > 1 && (theta_hat_1 = theta_hat_1[:, 1])

    # Step 2: efficient weight matrix.
    W_step3 = build_step3_weight_matrix(theta_hat_1, input_folder;
                                        K = exp1_K_sim, output_folder = output_folder)

    # Step 3: α+T only, efficient-weighted, warm-started at θ̂_1.
    theta_hat_2, _ = run_pso_optimization(;
        weight_matrix            = W_step3,
        skip_initial_alpha_search = true,
        warm_start_params        = theta_hat_1,
        output_subfolder         = "step3",
        gamma_beta_only          = true,
        moments_loss_gamma_beta  = true,
        n_particles              = exp1_n_particles,
        max_iter_initial         = exp1_max_iter_init,
        max_iter_stage           = exp1_max_iter_stage,
        max_loop                 = exp1_max_loop)

    # Step 4: Jacobian + inference at θ̂_2 (β+γ subsystem).
    J2, _, _, _ = compute_jacobian(theta_hat_2;
        param_indices = jacobian_param_indices,
        output_folder = output_folder, output_subdir = "step3",
        filename = "jacobian_all_step3.npy",
        K = exp1_K_jac, step_rel = 1e-4, step_abs = 1e-9, base_seed = 1_000_000)

    Omega_inf = NPZ.npzread(joinpath(output_folder, "step2", "Omega.npy"))
    _, sm2    = full_SMM(theta_hat_2; u_draws = U_DRAWS, sample_weights = SAMPLE_WEIGHTS)
    sim_vec   = masked_moments(sm2)
    emp_vec   = vec(empirical_moments)

    res = compute_smm_inference(theta_hat_2, J2[GB_INDICES, GB_COLS], W_step3, Omega_inf;
        param_indices         = GB_PARAM_IDX,
        empirical_moments_vec = emp_vec[GB_INDICES],
        simulated_moments_vec = sim_vec[GB_INDICES],
        output_folder         = joinpath(output_folder, "step3"),
        industry              = industry, K_sim = exp1_K_sim,
        block_ranges = GB_BLOCK_RANGES, block_names = GB_BLOCK_NAMES,
        gamma_ref_map = GAMMA_REF_MAP,
        param_labels = GB_LABELS, moment_labels = MOMENT_LABELS[GB_INDICES])

    return theta_hat_2, res["se_sw"], res["ci_95"]
end

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1 — full-vector point recovery, multiple perturbed starts
# ─────────────────────────────────────────────────────────────────────────────
if RUN_EXP1
    println("\n" * "="^70)
    println("EXPERIMENT 1: full-vector point recovery ($exp1_n_starts starts)")
    println("="^70)

    id0     = identified(theta0)                       # normalized identified truth
    n_id    = length(id0)
    thetahats = Matrix{Float64}(undef, length(theta0), exp1_n_starts)
    rel_err   = fill(NaN, exp1_n_starts)               # max|id(θ̂)-id0|/|id0|
    max_zSE   = fill(NaN, exp1_n_starts)               # max |Δ|/SE on α+T (normalized gauge)

    # `identified(·)` already returns T in the T_ref=1 (normalized) gauge, so the
    # POINT estimates need no further division. Only the raw inference SEs are
    # rescaled to this gauge below (se_norm = se_raw / ref_raw).
    id0_alphaT_norm = identified(theta0)[GB_COLS]       # normalized α+T truth

    for k in 1:exp1_n_starts
        sseed = BASE_SEED + 10_000 * k
        rng   = MersenneTwister(sseed)
        # Multiplicative perturbation of θ₀, clamped to the data-anchored bounds
        # implicitly enforced downstream by the PSO. ×LogUniform[0.5,2] on A,α,T;
        # ×U[0.8,1.2] on Ωᴸ,Ωˢ.
        pert = copy(theta0)
        logu(r) = exp(log(0.5) + (log(2.0) - log(0.5)) * rand(r))
        unif(r) = 0.8 + 0.4 * rand(r)
        pert[1]                                   *= unif(rng)                       # Ω^L
        for i in 2:(1 + S);            pert[i]    *= unif(rng); end                  # Ω^s
        for i in (S + 2):(ALPHA_T_START - 1); pert[i] *= logu(rng); end              # A
        for i in ALPHA_T_START:length(theta0); pert[i] *= logu(rng); end             # α, T
        # Keep α₀ monotonicity in the seed so Stage-0-skipped α start is valid.
        bsl = ALPHA_T_START:(ALPHA_T_START + N_TAU - 1)
        pert[bsl] .= sort(pert[bsl])

        println("\n--- Start $k/$exp1_n_starts (seed $sseed) ---")
        th2, _, _ = run_full_three_step(pert, sseed, joinpath("exp1", "start_$k"))
        thetahats[:, k] = th2

        idh = identified(th2)
        rel_err[k] = maximum(abs.(idh .- id0) ./ max.(abs.(id0), 1e-12))
        # |Δ|/SE on α+T is computed in the table block below by reloading the
        # start's saved SEs (normalized to the T_ref=1 gauge).
    end

    # Recompute z-scores cleanly (kept separate from the run loop for clarity):
    # reload each start's saved SEs and compute normalized-gauge z-scores.
    open(joinpath(IV_ROOT, "exp1_recovery_table.txt"), "w") do io
        println(io, "Experiment 1 — point recovery ($(now()))")
        println(io, "id0 length = $n_id   α+T params = $(length(GB_COLS))")
        @printf(io, "%-8s %16s %16s\n", "start", "max_rel_err", "max|Δ|/SE(α+T)")
        for k in 1:exp1_n_starts
            sf = joinpath(IV_ROOT, "exp1", "start_$k", "step3", "inference")
            th2 = thetahats[:, k]
            fh  = gauge_factors(th2)
            pt_norm = identified(th2)[GB_COLS]            # already normalized gauge
            zse = NaN
            se_path = joinpath(sf, "se_theta_sandwich.npy")
            if isfile(se_path)
                se_sw   = NPZ.npzread(se_path)            # raw units, α+T order
                se_norm = se_sw ./ fh                     # raw SE → normalized gauge
                zse     = maximum(abs.(pt_norm .- id0_alphaT_norm) ./ max.(se_norm, 1e-12))
            end
            max_zSE[k] = zse
            @printf(io, "%-8d %16.4e %16.4f\n", k, rel_err[k], zse)
        end
        # Cross-start dispersion (multi-modality measure).
        println(io, "\nCross-start dispersion of identified components (SD across starts):")
        ids = hcat([identified(thetahats[:, k]) for k in 1:exp1_n_starts]...)
        disp = vec(std(ids; dims = 2))
        @printf(io, "  max SD / |id0| = %.4e   (single basin if ≈ 0)\n",
                maximum(disp ./ max.(abs.(id0), 1e-12)))
        n_pass = count(max_zSE .<= 2.0)
        println(io, "\nPASS criterion (all within ~2 SE on α+T): $n_pass / $exp1_n_starts")
    end

    NPZ.npzwrite(joinpath(IV_ROOT, "exp1_thetahat_starts.npy"), thetahats)
    println("Experiment 1 complete. Recovery table written.")
end

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2 — α+T coverage Monte Carlo
# ─────────────────────────────────────────────────────────────────────────────
if RUN_EXP2
    println("\n" * "="^70)
    println("EXPERIMENT 2: α+T coverage (M=$exp2_M reps)")
    println("="^70)

    # Restore the fixed truth target before building W/Ω at θ₀.
    set_synthetic_targets!(masked_m0)

    # Build W_step3 / Ω ONCE at θ₀ (analogous to production building them at θ̂_1).
    # θ₀ is fixed across reps, so Σ_sim(θ₀) is constant — re-estimating per rep
    # would only re-draw the same object. Per-rep inference reuses this Ω.
    global output_folder
    output_folder = joinpath(IV_ROOT, "exp2")
    mkpath(output_folder)
    W_step3_0 = build_step3_weight_matrix(theta0, input_folder;
                                          K = exp2_K_sim, output_folder = output_folder)
    Sigma_data = NPZ.npzread(joinpath(output_folder, "step2", "Sigma_data.npy"))  # β+γ, β-then-γ
    Omega_0    = NPZ.npzread(joinpath(output_folder, "step2", "Omega.npy"))
    L_data     = psd_sqrt(Sigma_data)

    # Truth (normalized T_ref=1 gauge) for α+T. Point estimates come straight
    # from identified(·); only raw SEs are rescaled per rep (se_raw / ref_raw).
    truth_alphaT    = identified(theta0)[GB_COLS]
    p_alphaT        = length(GB_COLS)

    # Per-rep accumulators.
    est_norm  = Matrix{Float64}(undef, p_alphaT, exp2_M)   # normalized-gauge point estimates
    se_norm   = Matrix{Float64}(undef, p_alphaT, exp2_M)   # normalized-gauge SEs
    covered   = falses(p_alphaT, exp2_M)

    noise_rng = MersenneTwister(EXP2_NOISE_SEED)

    for b in 1:exp2_M
        global output_folder   # rebind the Main global so run_pso_optimization sees the per-rep dir
        # Draw target m_b = m(θ₀) + ε_b, ε_b ~ N(0, Σ_data) on the β+γ block.
        eps_b = L_data * randn(noise_rng, N_GB)
        m_b   = copy(masked_m0)
        m_b[GB_INDICES] .+= eps_b
        set_synthetic_targets!(m_b)

        # Re-estimate α+T only, warm-started at θ₀ (light polish).
        rep_dir = joinpath("exp2", "rep_$b")
        output_folder = joinpath(IV_ROOT, rep_dir)
        mkpath(output_folder)
        rseed = BASE_SEED + 100_000 + b
        Random.seed!(rseed); @everywhere Random.seed!($rseed)

        theta_hat_b, _ = run_pso_optimization(;
            weight_matrix            = W_step3_0,
            skip_initial_alpha_search = true,
            warm_start_params        = theta0,
            output_subfolder         = "step3",
            gamma_beta_only          = true,
            moments_loss_gamma_beta  = true,
            n_particles              = exp2_n_particles,
            max_iter_initial         = exp2_max_iter_init,
            max_iter_stage           = exp2_max_iter_stage,
            max_loop                 = exp2_max_loop)

        # Step-4 inference at θ̂_b (per-rep Jacobian; Ω reused from θ₀).
        Jb, _, _, _ = compute_jacobian(theta_hat_b;
            param_indices = jacobian_param_indices,
            output_folder = output_folder, output_subdir = "step3",
            filename = "jacobian_step3.npy",
            K = exp2_K_jac, step_rel = 1e-4, step_abs = 1e-9, base_seed = 2_000_000 + b)

        _, smb  = full_SMM(theta_hat_b; u_draws = U_DRAWS, sample_weights = SAMPLE_WEIGHTS)
        sim_vec = masked_moments(smb)

        res = compute_smm_inference(theta_hat_b, Jb[GB_INDICES, GB_COLS], W_step3_0, Omega_0;
            param_indices         = GB_PARAM_IDX,
            empirical_moments_vec = m_b[GB_INDICES],
            simulated_moments_vec = sim_vec[GB_INDICES],
            output_folder         = joinpath(output_folder, "step3"),
            industry              = industry, K_sim = exp2_K_sim,
            block_ranges = GB_BLOCK_RANGES, block_names = GB_BLOCK_NAMES,
            gamma_ref_map = GAMMA_REF_MAP,
            param_labels = GB_LABELS, moment_labels = MOMENT_LABELS[GB_INDICES])

        # Point estimate already on the T_ref=1 gauge; rescale raw SE to it.
        fb  = gauge_factors(theta_hat_b)
        ph  = identified(theta_hat_b)[GB_COLS]
        seh = res["se_sw"] ./ fb
        est_norm[:, b] = ph
        se_norm[:, b]  = seh
        covered[:, b]  = abs.(ph .- truth_alphaT) .<= 1.96 .* seh

        if b % 10 == 0 || b == exp2_M
            @printf("  rep %d/%d done (pooled coverage so far = %.3f)\n",
                    b, exp2_M, mean(covered[:, 1:b]))
        end
    end

    # ── Aggregate coverage / bias / SE-calibration ──────────────────────────
    cov_param  = vec(mean(covered; dims = 2))               # per-param coverage
    bias_param = vec(mean(est_norm; dims = 2)) .- truth_alphaT
    sd_param   = vec(std(est_norm; dims = 2))
    se_param   = vec(mean(se_norm; dims = 2))
    ratio      = sd_param ./ max.(se_param, 1e-12)

    NPZ.npzwrite(joinpath(IV_ROOT, "exp2_coverage.npy"),
                 hcat(cov_param, bias_param, sd_param, se_param, ratio))
    NPZ.npzwrite(joinpath(IV_ROOT, "exp2_estimates.npy"), est_norm)
    NPZ.npzwrite(joinpath(IV_ROOT, "exp2_se.npy"),        se_norm)

    open(joinpath(IV_ROOT, "exp2_coverage_table.txt"), "w") do io
        println(io, "Experiment 2 — α+T coverage  ($(now()))   M=$exp2_M")
        println(io, "Normalized (T_ref=1) gauge. CI₉₅ = θ̂ ± 1.96·SE_sandwich.")
        @printf(io, "%-22s %8s %12s %12s %12s %10s\n",
                "param", "cover", "bias", "SD(θ̂)", "mean(SE)", "SD/SE")
        for k in 1:p_alphaT
            @printf(io, "%-22s %8.3f %12.4e %12.4e %12.4e %10.3f\n",
                    GB_LABELS[k], cov_param[k], bias_param[k],
                    sd_param[k], se_param[k], ratio[k])
        end
        @printf(io, "\nPOOLED: coverage = %.3f   mean|bias| = %.4e   median SD/SE = %.3f\n",
                mean(covered), mean(abs.(bias_param)), median(ratio))
        @printf(io, "Coverage SE at p=0.95, M=%d: ±%.3f\n",
                exp2_M, sqrt(0.95 * 0.05 / exp2_M))
    end
    println("Experiment 2 complete. Coverage table written.")
end

# ─────────────────────────────────────────────────────────────────────────────
# One-page summary
# ─────────────────────────────────────────────────────────────────────────────
open(joinpath(IV_ROOT, "summary.txt"), "w") do io
    println(io, "Internal-validity summary — $industry  ($(now()))")
    println(io, "N_REG=$N_REG  N_TAU=$N_TAU  n_good=$n_good  N_rho=$N_rho")
    println(io, "α₀ = $(round.(alpha0, digits=4))")
    @printf(io, "min_g ESS_g (production draws) = %.1f / %d\n", min_ess, N_rho)
    println(io, "")
    println(io, "Artifacts:")
    println(io, "  theta_0.npy                  — synthetic truth parameter vector")
    println(io, "  masked_m_theta0.npy          — masked m(θ₀) (target moments)")
    println(io, "  ess_and_screen_at_theta0.txt — Section-1 identification diagnostics")
    RUN_EXP1 && println(io, "  exp1_thetahat_starts.npy / exp1_recovery_table.txt")
    RUN_EXP2 && println(io, "  exp2_coverage.npy / exp2_coverage_table.txt / exp2_estimates.npy / exp2_se.npy")
    println(io, "")
    println(io, "SMOKE_TEST = $SMOKE_TEST  (set false for the production run)")
end
println("\nAll done. Outputs under $IV_ROOT")
