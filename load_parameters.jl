# Parameter loading and constant distribution for the three-step SMM estimator.
# Expects the following variables to be defined in the calling scope:
#   input_folder  (String) — path to the industry data directory
#   n_coef        (Int)    — number of regression coefficients (4 or 5)
#
# After include(), the following locals are available in the calling scope:
#   N_moments, n_good_local, jacobian_param_indices, BLOCK_RANGES_local
#
# Expects n_tau to be defined in the calling scope (defaults to n_coef).
#   N_REG = n_coef  — reg_coef moment count (number of distance-bin regression moments)
#   N_TAU = n_tau   — trade-cost parameter count (length of the α vector in unpack_params/build_tau)
#
# All @everywhere const values are broadcast to workers here.

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — RAW DATA LOADING
# Reads the per-industry empirical inputs (wages, shares, distances, firm counts,
# sourcing matrix X_rs) from disk. Nothing is broadcast to workers yet.
# ═══════════════════════════════════════════════════════════════════════════

coefs                         = CSV.read(joinpath(input_folder, "stats.csv"), DataFrame)
distances_local               = NPZ.npzread(joinpath(input_folder, "distances.npy"))
filter_N_upstream_local       = NPZ.npzread(joinpath(input_folder, "filter_N_upstream.npy"))
w_rs_local                    = NPZ.npzread(joinpath(input_folder, "w_rs.npy")) # Upstream average wage.
regional_wages_local          = NPZ.npzread(joinpath(input_folder, "regional_wages.npy")) # Downstream region average wage for downstream firms.

# Comment if you want to use wage information. Care to comment or uncomment both so there is not too much difference between 
# upstream prices and downstream wages (would shift Omega_L to 1)
w_rs_local .= ifelse.(w_rs_local .!= 0, w_rs_local ./ w_rs_local, w_rs_local)
regional_wages_local .= ifelse.(regional_wages_local .!= 0, regional_wages_local ./ regional_wages_local, regional_wages_local)
N_downstream_per_region_local = NPZ.npzread(joinpath(input_folder, "N_downstream_per_region.npy"))
agg_industry_share_local      = NPZ.npzread(joinpath(input_folder, "input_share.npy"))
domestic_share_local          = NPZ.npzread(joinpath(input_folder, "domestic_share.npy"))
X_rs_local                    = NPZ.npzread(joinpath(input_folder, "X_rs.npy")) # X_rs has shape (S,R)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — FIXED STRUCTURAL PARAMETERS (CALIBRATION)
# Dimensions (S sectors, R regions) and all non-estimated model primitives:
# elasticities (epsilon, nu, nu_s), Fréchet dispersion (theta), labor/CI split
# (lambda), draw count (N_rho), and derived constants (GAMMA_FACTOR, delta_r).
# These are held fixed during estimation and broadcast to all workers.
# ═══════════════════════════════════════════════════════════════════════════

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
@everywhere const P_alpha             = $(coefs[4, "value"]) #Prior on alpha
@everywhere const lambda              = $(0.5)
@everywhere const nu                  = $(0.2)
@everywhere const nu_s                = $(ones(S_) .* 1.5)
@everywhere const theta               = $(1.)#1.768

println("\n N_rho = $N_rho — Entreprise par secteur x region")
println("\n Lambda = $lambda — Labor / CI share")
println("\n Epsilon = $epsilon — Sales elasticity")
println("\n nu = $nu — Across sector substituability")
println("\n nu_s = $nu_s — Within sector substituability")
println("\n theta = $theta — Frechet parameter")


# Γ((θ+1-ν_s)/θ)^{1/(1-ν_s)} — constant factor in EK closed-form price index P_sr.
# Precomputed once; used by compute_prices_analytical in model_analytical.jl (the GMM)
# Requires θ+1 > ν_s (checked here); for current calibration 2.768 > 2.5 ✓
begin
    @assert all(theta + 1 .> nu_s) "ν_s must be < θ+1 for closed-form P_sr"
    import SpecialFunctions
    _gamma_factor_local = [SpecialFunctions.gamma((theta + 1 - nu_s[s]) / theta)^(1 / (1 - nu_s[s])) for s in 1:S_]
    @everywhere const GAMMA_FACTOR = $_gamma_factor_local
end
@everywhere const delta_r             = $(ones(R_full)) # Downstream preference shifter.

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — GAMMA THRESHOLD + ACTIVE-SET PRUNING
# Defines which (sector, upstream-region) sourcing pairs are estimated. Pairs
# with a within-sector share at or below the threshold are dropped from both the
# γ_ls targets and the T active set, then survivors are renormalized to preserve
# each sector total. MUST run before T_mask_local — it shapes the active set.
# ═══════════════════════════════════════════════════════════════════════════

# ── Gamma threshold: drop small sourcing-share pairs from active set ─────
# Must precede T_mask_local so pruned pairs are excluded from T_MASK/n_good.
gamma_threshold = 0.0   # (s,r) pairs with γ_{rs} <= threshold are zeroed out
NPZ.npzwrite(joinpath(output_folder, "gamma_threshold.npy"), gamma_threshold)
emp_gamma_ls_local = permutedims(NPZ.npzread(joinpath(input_folder, "emp_gamma_ls.npy")))
# Shape: (R_full, S_) — indexed as emp_gamma_ls_local[r, s]

# ── Pre-threshold diagnostic: active regions above / below the cut, per sector ──
# Computed on the pristine matrix, before any pair is zeroed.
#   active = γ > 0 ; below = 0 < γ ≤ threshold (dropped) ; above = γ > threshold (kept)
println("\nGamma threshold = $gamma_threshold — active regions per sector:")
@printf("  %-8s %8s %8s %8s\n", "sector", "active", "above", "below")
total_active = 0; total_below = 0
for s in 1:S_
    col      = @view emp_gamma_ls_local[:, s]
    n_active = count(>(0), col)
    n_below  = count(x -> 0 < x/sum(col) <= gamma_threshold, col)
    n_above  = n_active - n_below
    global total_active += n_active; global total_below += n_below
    @printf("  %-8d %8d %8d %8d\n", s, n_active, n_above, n_below)
end
@printf("  %-8s %8d %8d %8d\n", "TOTAL", total_active, total_active - total_below, total_below)

n_dropped = 0
if gamma_threshold != 0
    for s in 1:S_
        sector_sum_before = sum(emp_gamma_ls_local[:, s])
        for r in 1:R_full
            if 0 < emp_gamma_ls_local[r, s]/sector_sum_before <= gamma_threshold
                emp_gamma_ls_local[r, s] = 0.0
                X_rs_local[s, r] = 0.0      # remove from T_MASK active set
                global n_dropped += 1
            end
        end
        # Renormalize survivors to preserve sector total
        sector_sum_after = sum(emp_gamma_ls_local[:, s])
        if sector_sum_after > 1e-15 && sector_sum_before > 1e-15
            emp_gamma_ls_local[:, s] .*= sector_sum_before / sector_sum_after
        end
        # Diagnostic: sectors collapsed to ≤ 1 surviving upstream region
        n_surv = count(>(0), @view emp_gamma_ls_local[:, s])
        if n_surv == 0
            error("γ-threshold: sector $s has NO surviving upstream region ⇒ " *
                  "X_s[s]=0, γ_ls=0/0 (NaN moment, NaN Jacobian column). " *
                  "Lower the threshold or drop this sector.")
        elseif n_surv == 1
            @warn "γ-threshold: sector $s reduced to a SINGLE upstream region. " *
                  "It is the ref region (dropped as sum-to-1 redundant), so the " *
                  "sector contributes zero γ_ls moments and T[s,·] has no free " *
                  "parameter."
        end
    end
end
println("Gamma threshold=$gamma_threshold: dropped $n_dropped (s,r) pairs")
@everywhere const emp_gamma_ls   = $(emp_gamma_ls_local)
# ──────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — T-MASK + GOOD-PAIR INDEX MAPS
# Builds the bookkeeping that links each active (s,r) comparative-advantage
# parameter T[s,r] to its position in the flattened parameter/moment vectors.
# The s-major (region-minor) flattening here is the convention the Jacobian's
# parameter axis and γ-moment row axis share (see CLAUDE.md invariant).
# ═══════════════════════════════════════════════════════════════════════════

# T-mask will be used to isolate the sector-region on which to estimate comparative advantage.
T_mask_local         = vec(permutedims(X_rs_local)) .> 0 # s-major (region-minor): identical to T_mask_moment_local / γ-moment convention
T_mask_moment_local  = vec(permutedims(X_rs_local)) .> 0 # Vec flattens column per column.  So we have all region within the first sector and so on
@everywhere const T_MASK        = $T_mask_local
@everywhere const T_MASK_MOMENT = $T_mask_moment_local


# Bellow, we flatten the vector and store the sector and region coordinates of the active regions. 
good_indices_local        = findall(permutedims(reshape(T_mask_local, R_full, S_)))  # s-major flat → (R,S) → (S,R)
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



# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — REFERENCE REGIONS
# Each sector's T and γ_ls shares are identified only up to a per-sector scale,
# so one region per sector is normalized to T=1 and its γ moment dropped. We pin
# that reference to the largest empirical sourcing share for numerical stability.
# ═══════════════════════════════════════════════════════════════════════════

# Reference region per sector: largest empirical sourcing share among active regions
# Those regions will always have T equal to one after unpack-parameters so we don't need to estimate them.
T_REF_REGION_local = Vector{Int}(undef, S_)
for s in 1:S_
    idxs = SECTOR_GOOD_INDICES_local[s]
    if !isempty(idxs)
        regions_s = GOOD_R_local[idxs]
        gamma_vals = [emp_gamma_ls[r, s] for r in regions_s]
        T_REF_REGION_local[s] = regions_s[argmax(gamma_vals)]
    else
        T_REF_REGION_local[s] = 0
    end
end
@everywhere const T_REF_REGION = $T_REF_REGION_local


# ── log-T (φ) reparameterization index maps ──────────────────────────────────
# The optimizer searches φ_i = log(T_i / T_{s,ref}) over the FREE reduced-T
# positions only; each sector's reference entry is dropped (pinned T=1) so the S
# unidentified directions never enter the search space. Reduced ordering matches
# unpack_params: flat p=(s-1)*R+r (r fastest), kept where T_MASK is true.
T_reduced_s_local = Int[]
T_reduced_r_local = Int[]
let p = 0
    for s in 1:S_, r in 1:R_full
        p += 1
        if T_mask_local[p]
            push!(T_reduced_s_local, s)
            push!(T_reduced_r_local, r)
        end
    end
end
n_T_reduced_local = length(T_reduced_s_local)
# reduced index of each sector's reference entry (T pinned to 1 there)
sector_ref_reduced_local = zeros(Int, S_)
for i in 1:n_T_reduced_local
    if T_reduced_r_local[i] == T_REF_REGION_local[T_reduced_s_local[i]]
        sector_ref_reduced_local[T_reduced_s_local[i]] = i
    end
end
@assert all(sector_ref_reduced_local[s] > 0
            for s in 1:S_ if !isempty(SECTOR_GOOD_INDICES_local[s])) "each active sector needs a reference reduced index in T_MASK"
T_free_reduced_idx_local = [i for i in 1:n_T_reduced_local if i ∉ sector_ref_reduced_local]
@everywhere const T_REDUCED_S        = $T_reduced_s_local        # sector per reduced-T position
@everywhere const SECTOR_REF_REDUCED = $sector_ref_reduced_local # reduced index of each sector's ref
@everywhere const T_FREE_REDUCED_IDX = $T_free_reduced_idx_local # reduced positions the optimizer varies
@everywhere const N_T_REDUCED        = $n_T_reduced_local        # sum(T_MASK)
@everywhere const N_T_FREE           = $(length(T_free_reduced_idx_local)) # = N_T_REDUCED - #active sectors


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — REMAINING TARGETS + PARAMETER COUNTS (pi_r, reg_coef, N_REG/N_TAU)
# Downstream market shares (pi_r) and the distance-bin regression coefficients
# (reg_coef) complete the empirical targets. N_REG fixes the reg_coef moment
# count; N_TAU the trade-cost parameter count — decoupled so a power-law τ=d^α
# (N_TAU=1) can be over-identified against N_REG>1 binned moments.
# ═══════════════════════════════════════════════════════════════════════════

X_dr_local = CSV.read(joinpath(input_folder, "X_dr.csv"), DataFrame).X_dr
X_dr_local = X_dr_local[N_downstream_per_region_local .!= 0]
emp_pi_r_local = X_dr_local ./ sum(X_dr_local)
@everywhere const emp_pi_r_full  = $(emp_pi_r_local)
@everywhere const emp_pi_r       = $(emp_pi_r_local)
if n_coef == 1
    reg_coef_local = [coefs[3, "value"]]
else
    reg_coef_local = NPZ.npzread(joinpath(input_folder, "reg_coef_$(n_coef).npy"))
end
@everywhere const reg_coef = $reg_coef_local
# N_REG: number of reg_coef moments (distance-bin regression coefficients).
# N_TAU: number of trade-cost parameters (length of α in unpack_params/build_tau).
# For standard runs n_tau == n_coef so N_TAU == N_REG; the split enables N_TAU=1
# (power-law τ = d^α) with N_REG=4 (four binned reg-coef moments, over-identified).
n_reg = length(reg_coef_local)   # actual moment count from loaded data
@everywhere N_REG = $(n_reg)
@everywhere N_TAU = $(n_tau)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7 — T STARTING VALUES (GRAVITY FALLBACK)
# Warm-starts the comparative-advantage optimisation. The gravity guess
# T ≈ γ_ls · w^θ is the value consistent with observed shares at UNIFORM trade
# costs (τ≡1) — it drops all trade-cost geometry. It is kept here only as a
# fallback: the active T_rs_init is built by the γ-inversion in SECTION 10b
# (after the geography/trade-cost precompute), which restores the τ terms.
# ═══════════════════════════════════════════════════════════════════════════

# Gravity fallback base (uniform trade costs): T ≈ γ_ls · w^θ, per (s,r).
T_gravity = zeros(S_, R_full)
for s in 1:S_
    idxs = SECTOR_GOOD_INDICES_local[s]
    for g in idxs
        l = GOOD_R_local[g]
        T_gravity[s, l] = max(emp_gamma_ls[l, s] * (w_rs_local[l]^theta), 1e-12)
    end
end
# NOTE: `@everywhere const T_rs_init` is bound in SECTION 10b, once the
# trade-cost geometry (LOG_DIST_DOWNSTREAM) it needs is available.

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8 — EMPIRICAL MOMENT VECTOR + MOMENT_MASK + BLOCK_RANGES
# Stacks the five moment blocks into one target vector and builds MOMENT_MASK,
# which drops the linearly-dependent moments (first industry/pi_r share, inactive
# and per-sector reference γ_ls) so the system is identified. BLOCK_RANGES indexes
# each block within the masked vector; Weight_matrix_custom is the Step-1 metric.
# ═══════════════════════════════════════════════════════════════════════════

# Moment block sizes + MOMENT_MASK
n_labor    = 1
n_industry = length(vec(agg_industry_share_local))
n_gamma    = length(vec(emp_gamma_ls_local))
# n_reg already computed above (== length(reg_coef_local)) for the N_REG broadcast.
n_pi       = length(emp_pi_r_local)
N_moments_full = n_labor + n_industry + n_pi + n_reg + n_gamma

empirical_moments_local = vcat(
    [agg_labor_share],
    vec(agg_industry_share_local),
    emp_pi_r_local,
    reg_coef_local,
    vec(emp_gamma_ls_local)        # thresholded+renormalized, consistent with loss residuals
)

moment_mask_local = trues(N_moments_full)
# Remove first industry share.
moment_mask_local[n_labor + 1] = false
# Remove first pi_r (sum-to-1 redundancy).
moment_mask_local[n_labor + n_industry + 1] = false
# reg_coef block: no masking needed.
# Remove non-active gamma_ls entries.
for idx in 1:(S_ * R_full)
    if !T_mask_moment_local[idx]
        moment_mask_local[n_labor + n_industry + n_pi + n_reg + idx] = false
    end
end
# Remove reference-region gamma_ls per sector (sum-to-1 redundancy, aligned with T normalization).
for s in 1:S_
    ref_r = T_REF_REGION_local[s]
    if ref_r > 0
        moment_mask_local[n_labor + n_industry + n_pi + n_reg + (s - 1) * R_full + ref_r] = false
    end
end

empirical_moments_local = reshape(empirical_moments_local[moment_mask_local], 1, sum(moment_mask_local))
N_moments = sum(moment_mask_local)

@everywhere const MOMENT_MASK        = $moment_mask_local
@everywhere const empirical_moments  = $(empirical_moments_local)
@everywhere const K_max              = $(50)


# BLOCK_RANGES : Length of each block of moment. 
BLOCK_RANGES_local = compute_block_ranges(n_labor, n_industry, n_pi, n_reg, n_gamma, moment_mask_local)
@everywhere const BLOCK_RANGES = $BLOCK_RANGES_local
@everywhere const BLOCK_NAMES  = ("labor", "industry", "pi_r", "reg_coef", "gamma_ls")

w_vec = ones(N_moments)
w_vec[BLOCK_RANGES_local[4]] .= 100.0
Weight_matrix_custom_local = Diagonal(w_vec)
@everywhere const Weight_matrix_custom = $Weight_matrix_custom_local

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 9 — SIMULATION DRAWS
# Generates the fixed common-random-numbers (U_DRAWS) and importance weights used
# by the simulated moments, so the SMM objective is smooth across parameter
# evaluations. Held constant for the whole estimation.
# ═══════════════════════════════════════════════════════════════════════════

# Draw method for the Fréchet inverse-CDF transform. The entry point (main.jl /
# main_gmm.jl) may define `draw_method`; default to :qmc (stratified uniform, flat
# weights — unbiased for the min-coupled moments). :mc and :is are alternatives.
draw_method_local = (@isdefined(draw_method)) ? draw_method : :sobol
@assert draw_method_local in (:qmc, :mc, :is, :sobol) "draw_method must be :qmc, :mc, :is or :sobol, got :$draw_method_local"
@everywhere const DRAW_METHOD = $(QuoteNode(draw_method_local))

# Optimizer backend. The entry point may define `optimizer_backend`; default :pso
# (legacy staged pattern). :cmaes runs one joint CMA-ES per SMM step; :tiktak runs
# one joint TikTak multistart (Sobol pretest + Nelder-Mead) per step. All three
# search T in log space via the φ maps above.
optimizer_backend_local = (@isdefined(optimizer_backend)) ? optimizer_backend : :pso
@assert optimizer_backend_local in (:pso, :cmaes, :tiktak) "optimizer_backend must be :pso, :cmaes or :tiktak, got :$optimizer_backend_local"
@everywhere const OPTIMIZER_BACKEND = $(QuoteNode(optimizer_backend_local))

# Draw count for INFERENCE (Jacobian + Σ_sim), decoupled from the optimization draw
# count N_rho. The optimizer's loss uses N_rho draws (U_DRAWS); the Jacobian (`compute_jacobian`)
# and the Σ_sim estimate (`build_step3_weight_matrix`) resample with N_RHO_INFERENCE draws
# instead. Inference wants MANY more draws than the optimizer (the fixed-draw Jacobian /
# Σ_sim are simulation-noisy — e.g. the reg_coef column noise), so this can be set well
# above N_rho without slowing the search. The entry point may define `n_rho_inference`;
# default = N_rho (byte-identical to before).
n_rho_inference_local = (@isdefined(n_rho_inference)) ? n_rho_inference : N_rho
@assert n_rho_inference_local >= 1 "n_rho_inference must be ≥ 1, got $n_rho_inference_local"
@everywhere const N_RHO_INFERENCE = $(n_rho_inference_local)
println("\n N_rho (optimization) = $N_rho ; N_RHO_INFERENCE (Jacobian + Σ_sim) = $N_RHO_INFERENCE")

println("Generating draws (method = :$DRAW_METHOD)...")
u_draws_local, sample_weights_local = generate_draws(N_rho, n_good_local, DRAW_METHOD)
@everywhere const U_DRAWS        = $u_draws_local
@everywhere const SAMPLE_WEIGHTS = $sample_weights_local

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 10 — DISTANCE / GEOGRAPHY PRECOMPUTATION
# Precomputes the geography the regression moments and trade costs depend on:
# distance bins, nearest downstream plant/region, and log-distances. Cached once
# here since geography is fixed across parameter evaluations.
# ═══════════════════════════════════════════════════════════════════════════

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

LOG_DIST_DOWNSTREAM_local = log.(max.(distances_downstream_local, 1.0))
LOG_CLOSEST_DIST_local    = log.(max.(closest_plant_dist_local, 1.0))
@everywhere const LOG_DIST_DOWNSTREAM = $LOG_DIST_DOWNSTREAM_local
@everywhere const LOG_CLOSEST_DIST    = $LOG_CLOSEST_DIST_local

println("Constants distributed. N_moments=$N_moments, n_good=$n_good_local")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 10b — T STARTING VALUES VIA γ_ls INVERSION (TRADE-COST AWARE)
# Inverts the closed-form sourcing-share map for the comparative-advantage
# starting point, using the α prior on trade costs and wages ≡ 1.
#
# Model (model_analytical.jl, w_val ≡ 1):
#     X_ls[r,s] = T[s,r] · M[s,r],   M[s,r] = Σ_dr τ[r,dr]^(-θ) · Ê_dr / Φ[s,dr]
#     Φ[s,dr]   = Σ_{r'} T[s,r'] · τ[r',dr]^(-θ)
#     γ_ls[r,s] ∝ T[s,r] · M[s,r]
# ⇒  T[s,r] ∝ γ_ls[r,s] / M[s,r].
#
# M depends on T only through Φ, so this is a cheap fixed point (τ from the α
# prior is fixed throughout). Destination expenditure E_{s,dr} is proxied by the
# observed market size Ê_dr = emp_pi_r_full (the destination-varying term is Y_dr;
# the residual price-index terms need the GE solve and are dropped for the init).
# At the fixed point the model reproduces emp_gamma_ls up to that proxy. T is only
# identified per sector up to scale (unpack_params normalizes T[s,:] /= T[s,ref]),
# so the inversion is likewise normalized to its reference region.
# ═══════════════════════════════════════════════════════════════════════════

# ── Read the α prior on trade costs from stats.csv (robust to layout) ────────
# Accepts either a column named "prior_alpha" or a row whose label column equals
# "prior_alpha" with the number in the "value" column. Returns nothing if absent.
function _read_prior_alpha(coefs_df)
    cols = names(coefs_df)
    if "prior_alpha" in cols
        vals = collect(skipmissing(coefs_df[!, "prior_alpha"]))
        !isempty(vals) && return Float64(vals[1])
    end
    if "value" in cols
        for cname in cols
            col = coefs_df[!, cname]
            idx = findfirst(x -> x isa AbstractString &&
                                 lowercase(strip(x)) == "prior_alpha", col)
            idx !== nothing && return Float64(coefs_df[idx, "value"])
        end
    end
    return nothing
end

"""
    invert_T_from_gamma(prior_alpha; max_iter=1000, tol=1e-11, damping=0.5)

Fixed-point inversion of the sourcing-share map for the T starting values.
Returns an (S, R_full) matrix with active (s,r) entries set to the inverted
comparative advantage (per-sector normalized to the reference region) and
inactive entries left at 0. Uses trade costs τ[r,dr] = exp(α · log max(d,1))
(the power-law N_TAU==1 form in build_tau, evaluated at the α prior) and wages ≡ 1.
"""
function invert_T_from_gamma(prior_alpha::Real; max_iter::Int=1000,
                             tol::Float64=1e-11, damping::Float64=0.5)
    θ = theta
    # τ^{-θ}[r, dr] under the power-law prior and wages ≡ 1.
    tau_negθ = exp.((-θ * prior_alpha) .* LOG_DIST_DOWNSTREAM_local)   # (R_full, R_downstream)
    Ê = emp_pi_r_local ./ sum(emp_pi_r_local)                          # (R_downstream,) market size

    T = zeros(Float64, S_, R_full)
    max_iters_used = 0
    for s in 1:S_
        regions_s = SECTOR_GOOD_REGIONS_local[s]
        isempty(regions_s) && continue
        ref = T_REF_REGION_local[s] > 0 ? T_REF_REGION_local[s] : regions_s[1]
        # Initialise at the observed shares (positive), normalized to the ref region.
        for r in regions_s
            T[s, r] = max(emp_gamma_ls_local[r, s], 1e-12)
        end
        T[s, regions_s] ./= T[s, ref]
        for it in 1:max_iter
            max_iters_used = max(max_iters_used, it)
            # Φ[dr] = Σ_{r'∈active} T[s,r'] τ^{-θ}[r',dr]
            Phi = zeros(Float64, R_down_)
            for r in regions_s, dr in 1:R_down_
                Phi[dr] += T[s, r] * tau_negθ[r, dr]
            end
            # T_new[r] ∝ γ / M[r],  M[r] = Σ_dr τ^{-θ}[r,dr] Ê_dr / Φ[dr]
            # Damped log-space update; the map is homogeneous degree 1 in T (scale
            # is a neutral direction), so renormalize to the ref region each pass.
            T_new = Dict{Int,Float64}()
            for r in regions_s
                M = 0.0
                for dr in 1:R_down_
                    Phi[dr] > 1e-300 && (M += tau_negθ[r, dr] * Ê[dr] / Phi[dr])
                end
                γ  = max(emp_gamma_ls_local[r, s], 1e-12)
                Tr = M > 1e-300 ? γ / M : T[s, r]
                T_new[r] = exp((1 - damping) * log(T[s, r]) + damping * log(Tr))
            end
            ref_val = T_new[ref]
            max_rel = 0.0
            for r in regions_s
                Tn  = T_new[r] / ref_val                       # renormalize to ref=1
                rel = abs(log(Tn) - log(T[s, r]))
                rel > max_rel && (max_rel = rel)
                T[s, r] = Tn
            end
            max_rel < tol && break
        end
    end
    return T, max_iters_used
end

_prior_alpha = _read_prior_alpha(coefs)
if _prior_alpha === nothing
    @warn "prior_alpha not found in stats.csv — falling back to gravity T init (τ≡1). " *
          "Add a `prior_alpha` column or row to enable the trade-cost-aware inversion."
    T_init_local = T_gravity ./ T_gravity   # active→1, inactive→NaN (masked out later)
else
    println("\nInverting T from γ_ls with prior_alpha = $_prior_alpha (wages ≡ 1)")
    T_init_local, iters_used = invert_T_from_gamma(_prior_alpha)
    active_vals = T_init_local[T_init_local .> 0]
    if isempty(active_vals) || !all(isfinite, active_vals)
        @warn "γ-inversion produced non-finite/empty T — falling back to gravity T init."
        T_init_local = T_gravity ./ T_gravity
    else
        @printf("  γ-inversion done (≤%d iters). T range [%.3g, %.3g], median %.3g\n",
                iters_used, minimum(active_vals), maximum(active_vals),
                sort(active_vals)[cld(length(active_vals), 2)])
    end
end
@everywhere const T_rs_init = $(T_init_local)

# Trade-cost (α) init anchor for the PSO search box. The α prior (N_TAU==1) is the
# fixed centre of the [×0.5, ×2] bound the optimizer keeps α within, in every stage
# (mirrors the T box anchored to T_rs_init). `nothing` ⇒ the optimizer falls back to
# anchoring α to each stage's starting value (see optimizer.jl train_stage).
@everywhere const TAU_PRIOR = $((_prior_alpha !== nothing && n_tau == 1) ?
                                [Float64(_prior_alpha)] : nothing)

# Init-anchored PSO search-box multipliers. α and T are boxed to
# [BOUND_LO, BOUND_HI] × their initial value in every stage (optimizer.jl
# train_stage). Single source of truth so the optimizer bounds and the
# T_best_vs_initial window lines (tools.jl plot_T_vs_initial) never drift apart.
@everywhere const BOUND_LO = 0.1
@everywhere const BOUND_HI = 10

# ── Diagnostic: gravity vs γ-inversion T starting values (ref-normalized) ────
# Saves a log-log scatter of the two initialisations (both rescaled per sector to
# their reference region) and quantifies their gap versus distance. Since both
# share the same γ_ls and wages ≡ 1, the ratio is exactly the market-access ratio
#   T_inv/T_grav = M[s,ref]/M[s,r],   M[s,r] = Σ_dr (π̂_dr/Φ_{s,dr}) · d_{r,dr}^{-θα},
# whose only origin-r dependence is d^{-θα} ⇒ T_inv/T_grav ≈ (d_r/d_ref)^{θα}.
# Master-only, guarded — never blocks estimation.
try
    # Reference-normalized gravity init (τ≡1): T_grav[s,r] = γ_ls[r,s]/γ_ls[ref,s].
    _ref_norm_gravity = begin
        G = fill(NaN, S_, R_full)
        for s in 1:S_
            regions_s = SECTOR_GOOD_REGIONS_local[s]
            isempty(regions_s) && continue
            ref  = T_REF_REGION_local[s] > 0 ? T_REF_REGION_local[s] : regions_s[1]
            gref = T_gravity[s, ref]
            for r in regions_s
                G[s, r] = gref > 0 ? T_gravity[s, r] / gref : NaN
            end
        end
        G
    end

    # Access-weighted mean distance to downstream markets (km), per region. This is
    # the faithful single-distance summary of M[s,r]=Σ_dr w_dr d^{-θα} (the ratio
    # M(ref)/M(r) is a ratio of accessibility sums, not a single power law, so the
    # nearest-plant distance under-summarizes it).
    _pihat = emp_pi_r_local ./ sum(emp_pi_r_local)
    _dbar  = distances_downstream_local * _pihat          # (R_full,)

    # Build the scatter + distance diagnostic for a given α (ref-normalized both axes).
    function _t_init_compare(alpha_val)
        T_inv_a, _ = invert_T_from_gamma(alpha_val)
        xs = Float64[]; ys = Float64[]; ds = Float64[]; ss = Int[]; rs = Int[]
        for s in 1:S_, r in SECTOR_GOOD_REGIONS_local[s]
            (isfinite(_ref_norm_gravity[s, r]) && T_inv_a[s, r] > 1e-6 &&
             _ref_norm_gravity[s, r] > 1e-6) || continue
            push!(xs, _ref_norm_gravity[s, r]); push!(ys, T_inv_a[s, r])
            push!(ds, _dbar[r]); push!(ss, s); push!(rs, r)
        end
        isempty(xs) && return
        p = Plots.scatter(xs, ys; zcolor = ds, xscale = :log10, yscale = :log10,
            xlabel = "gravity init  T/T_ref  (τ≡1)",
            ylabel = "γ-inversion init  T/T_ref  (α=$alpha_val)",
            title  = "T starting values: gravity vs γ-inversion (α=$alpha_val)",
            colorbar_title = "access-weighted mean dist to markets (km)",
            markersize = 5, markeralpha = 0.7, legend = false)
        lo = min(minimum(xs), minimum(ys)); hi = max(maximum(xs), maximum(ys))
        Plots.plot!(p, [lo, hi], [lo, hi]; color = :black, ls = :dash)
        png = joinpath(output_folder, "T_init_gravity_vs_inversion_a$(alpha_val).png")
        Plots.savefig(p, png)
        NPZ.npzwrite(joinpath(output_folder, "T_init_pairs_a$(alpha_val).npz"),
                     Dict("gravity" => xs, "inversion" => ys, "dist_km" => ds,
                          "sector" => Float64.(ss), "region" => Float64.(rs)))
        println("  saved $(png)")

        # Distance-difference approximation over d ∈ [0, 200] km.
        keep = ds .<= 200.0
        if count(keep) >= 3
            rr = ys[keep] ./ xs[keep]                   # = M(ref)/M(r), the correction
            lr = log.(rr)
            ld = log.(max.(ds[keep], 1.0))
            b  = hcat(ones(length(ld)), ld) \ lr        # OLS slope ≈ θα
            srr = sort(rr)
            med = srr[cld(length(srr), 2)]
            @printf("\n[T-init α=%.2f diagnostic, access-weighted d ≤ 200 km, n=%d active regions]\n",
                    alpha_val, count(keep))
            @printf("  correction  T_inv/T_grav = M(ref)/M(r) (only geography + α differ):\n")
            @printf("     min %.2f   median %.2f   max %.2f   mean|Δlog T| %.3f\n",
                    minimum(rr), med, maximum(rr), sum(abs.(lr)) / length(lr))
            @printf("  effective  log(T_inv/T_grav) ≈ %+.3f %+.3f·log d̄\n", b[1], b[2])
            @printf("     (functional form (d/d_ref)^{θα}=(·)^%.2f holds exactly only in the\n",
                    theta * alpha_val)
            @printf("      single-dominant-destination limit; M sums over all markets so the\n")
            @printf("      realized slope is geography-dependent, not a clean θα).\n")
            @printf("  ⇒ at α=%.2f the γ-inversion repositions the gravity T by a factor\n", alpha_val)
            @printf("    ~%.2f× (median) and up to ~%.1f× for the most market-remote regions;\n",
                    med, maximum(rr))
            @printf("    ≈1× for regions near the reference.\n")
        end
    end

    _t_init_compare(0.5)                                # the requested α = 0.5 case
    if _prior_alpha !== nothing && !isapprox(Float64(_prior_alpha), 0.5)
        _t_init_compare(Float64(_prior_alpha))          # also the actual prior in use
    end
catch e
    @warn "T-init comparison plot skipped: $e"
end

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 11 — IDENTIFIED-PARAMETER INDICES (JACOBIAN / INFERENCE)
# Selects the columns of the Jacobian that are actually identified, dropping the
# directions killed by the model's internal normalizations (Ω^s, A, and each
# sector's reference T). Without this the Jacobian would be rank-deficient.
# ═══════════════════════════════════════════════════════════════════════════

# Indices of identified parameters to use in Jacobian/inference.
# Excludes the S+2 flat directions created by internal normalizations:
#   - Ω^s[1]  (position 2)           : Omega_s ./= sum(Omega_s)
#   - A[1]    (position S+2)          : A ./= A[1]
#   - T[s, T_REF_REGION[s]] for each s: T_mat[s,:] ./= T_mat[s, ref_r] (most important regions in empirical gamma_ls)
# New parameter layout: [Ω^L(1) | Ω^s(S) | A(R_down) | α(N_TAU) | T(sum(T_MASK))]

_excluded = Set{Int}()
push!(_excluded, 2)              # Ω^s[1]: position 2 in new layout
push!(_excluded, S_ + 2)         # A[1]:   position S+2 in new layout
T_param_offset = 1 + S_ + R_down_ + N_TAU
for s in 1:S_
    ref_r = T_REF_REGION_local[s]
    ref_r == 0 && continue
    flat_pos = (s - 1) * R_full + ref_r   # s-major index in vec(permutedims(T_rs)), shape (R, S)
    if T_mask_local[flat_pos]
        t_idx = count(T_mask_local[1:flat_pos])
        push!(_excluded, T_param_offset + t_idx)
    end
end
jacobian_param_indices = [i for i in 1:(1 + S_ + R_down_ + N_TAU + sum(T_mask_local)) if i ∉ _excluded]
println("Jacobian will cover $(length(jacobian_param_indices)) identified parameters ($(length(_excluded)) normalized-out excluded).")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 12 — γ REFERENCE-REGION RECONSTRUCTION MAP
# Records, per sector, how to rebuild the dropped reference-region γ share (and
# its SE) from the retained shares via the adding-up constraint — used only for
# inference reporting, not for estimation.
# ═══════════════════════════════════════════════════════════════════════════

# ── γ_ls reference-region reconstruction map (for inference plotting) ────────
# Each retained γ moment in BLOCK_RANGES[5] corresponds to one active (s,r) pair
# that is NOT the sector's reference region. For every sector we record, in the
# *local* coordinates of the γ block, the positions of its retained regions, so
# the dropped reference share can be rebuilt as c_s − Σ γ_retained and its SE as
# 1' Var_γ 1 over those positions.
#
# c_s = domestic_share[s]  (the per-sector total the γ shares sum to).
gamma_block_local = collect(BLOCK_RANGES_local[5])   # global masked indices of γ block
gamma_block_start = isempty(gamma_block_local) ? 0 : first(gamma_block_local)

# Walk the *unmasked* γ layout (sector-major, region-minor, length S*R_full) and,
# for each kept entry, find its running position within the masked γ block.
gamma_ref_map_local = Vector{NamedTuple{(:sector, :local_positions, :c_s, :emp_ref),
                                        Tuple{Int, Vector{Int}, Float64, Float64}}}()

# offset of the γ block within the FULL (unmasked) moment vector
gamma_full_offset = n_labor + n_industry + n_pi + n_reg
local_counter = 0   # position within the masked γ block (1-based)

# Precompute, for each unmasked γ slot, whether it is kept and its local index.
local_index_of_full = Dict{Int,Int}()   # full-γ-slot (1..S*R_full) → local γ position
for slot in 1:(S_ * R_full)
    full_pos = gamma_full_offset + slot
    if moment_mask_local[full_pos]
        global local_counter
        local_counter += 1
        local_index_of_full[slot] = local_counter
    end
end

for s in 1:S_
    ref_r = T_REF_REGION_local[s]
    ref_r == 0 && continue                         # sector has no active region
    # retained regions of sector s = active, non-reference
    local_positions = Int[]
    for r in 1:R_full
        slot = (s - 1) * R_full + r
        if haskey(local_index_of_full, slot)       # kept (active & not ref)
            push!(local_positions, local_index_of_full[slot])
        end
    end
    isempty(local_positions) && continue           # ref was the only region: no free γ, skip
    c_s     = domestic_share_local[s]
    emp_ref = emp_gamma_ls_local[ref_r, s]         # already thresholded+renormalized
    push!(gamma_ref_map_local,
          (sector = s, local_positions = local_positions, c_s = c_s, emp_ref = emp_ref))
end

@everywhere const GAMMA_REF_MAP = $gamma_ref_map_local
println("γ reference-region map built for $(length(gamma_ref_map_local)) sectors " *
        "(used for inference reference-point SEs).")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 13 — MOMENT AND PARAMETER LABELS
# Builds human-readable names for each Jacobian row (moment) and column
# (parameter), aligned to MOMENT_MASK and jacobian_param_indices, for the
# inference reports. Falls back to integer ids when the name CSV is absent.
# ═══════════════════════════════════════════════════════════════════════════

# ── Human-readable labels for moments and parameters (for inference reports) ─
# Two SEPARATE axes:
#   • MOMENT_LABELS : one per kept moment, in MOMENT_MASK order (= rows of J,
#     = order of empirical_moments / simulated vectors). Blocks: labor, industry,
#     pi_r, reg_coef, gamma_ls — each already ref/sum-to-1-dropped.
#   • PARAM_LABELS  : one per identified parameter, in jacobian_param_indices
#     order (= columns of J). Layout [Ω^L | Ω^s | A | α | T], minus the S+2
#     normalized-out directions.
# Sector/ZE names come from filter_N_upstream.csv when present; otherwise we
# fall back to integer ids so a missing CSV never aborts the run.


filter_N_upstream_df = CSV.read(joinpath(input_folder,"filter_N_upstream.csv"),DataFrame)
# --- sector (A129) and downstream-region (ze2010) name maps -----------------
_sector_names = String[string(s) for s in unique(filter_N_upstream_df.A129)]                 # default: "1".."S"
_ze_names = [@sprintf("%04d", r) for r in unique(filter_N_upstream_df.ze2010)]          # default: "1".."R"
let csv_path = joinpath(input_folder, "filter_N_upstream.csv")
    if isfile(csv_path)
        _fdf = CSV.read(csv_path, DataFrame)
        if "A129" in names(_fdf)
            _a = sort(unique(_fdf.A129))
            length(_a) == S_ && (_sector_names = string.(_a))   # sorted, 1-indexed
        end
        if "ze2010" in names(_fdf)
            _z = sort(unique(_fdf.ze2010))
            length(_z) == R_full && (_ze_names = string.(_z))
        end
    else
        @warn "filter_N_upstream.csv not found; moment/param labels use integer ids."
    end
end

# --- MOMENT_LABELS: walk the FULL moment layout, keep where MOMENT_MASK is true
_moment_labels = String[]
# block 1: labor (1 moment, kept unless masked)
moment_mask_local[1] && push!(_moment_labels, "labor")
# block 2: industry shares Ω^s_1..Ω^s_S  (first one dropped by mask)
for s in 1:S_
    full_pos = n_labor + s
    moment_mask_local[full_pos] && push!(_moment_labels, "Omega_s[$(_sector_names[s])]")
end
# block 3: pi_r over downstream regions (first dropped)
for (j, r) in enumerate(downstream_regions_local)
    full_pos = n_labor + n_industry + j
    moment_mask_local[full_pos] && push!(_moment_labels, "pi_r[$(_ze_names[r])]")
end
# block 4: reg_coef (β-distance bins), never masked
for b in 1:n_reg
    full_pos = n_labor + n_industry + n_pi + b
    moment_mask_local[full_pos] &&
        push!(_moment_labels, n_reg == 1 ? "reg_coef" : "reg_coef[$b]")
end
# block 5: gamma_ls, sector-major region-minor, active & non-ref kept
for s in 1:S_, r in 1:R_full
    slot     = (s - 1) * R_full + r       # index into vec(permutedims(emp_gamma_ls))
    full_pos = n_labor + n_industry + n_pi + n_reg + slot
    moment_mask_local[full_pos] &&
        push!(_moment_labels, "gamma[$(_sector_names[s])-$(_ze_names[r])]")
end
@assert length(_moment_labels) == N_moments "moment-label count $(length(_moment_labels)) != N_moments=$N_moments"
@everywhere const MOMENT_LABELS = $_moment_labels

# --- PARAM_LABELS: full param layout, then restrict to jacobian_param_indices
# Layout: [Ω^L(1) | Ω^s(S) | A(R_down) | α(N_TAU) | T(active, sector-major)]
_param_labels_full = String[]
push!(_param_labels_full, "Omega_L")
for s in 1:S_;            push!(_param_labels_full, "Omega_s[$(_sector_names[s])]"); end
for r in 1:R_down_;       push!(_param_labels_full, "A[$(_ze_names[downstream_regions_local[r]])]"); end
for b in 1:N_TAU;         push!(_param_labels_full, N_TAU == 1 ? "alpha" : "alpha_$b"); end
# T entries follow vec(permutedims(T_rs)) s-major over (S,R) restricted to T_MASK.
# Recover (s,r) for each active T slot in the same order T_reduced is laid out.
for flat_pos in findall(T_mask_local)             # s-major: s outer, r inner
    s = ((flat_pos - 1) ÷ R_full) + 1
    r = ((flat_pos - 1) %  R_full) + 1
    push!(_param_labels_full, "T[$(_sector_names[s])-$(_ze_names[r])]")
end
@assert length(_param_labels_full) == 1 + S_ + R_down_ + N_TAU + sum(T_mask_local)
_param_labels = _param_labels_full[jacobian_param_indices]
@everywhere const PARAM_LABELS = $_param_labels
println("Built $(length(_moment_labels)) moment labels and $(length(_param_labels)) parameter labels.")