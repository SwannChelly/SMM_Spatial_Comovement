# Parameter loading and constant distribution for the three-step SMM estimator.
# Expects the following variables to be defined in the calling scope:
#   input_folder  (String) — path to the industry data directory
#   n_coef        (Int)    — number of regression coefficients (4 or 5)
#
# After include(), the following locals are available in the calling scope:
#   N_moments, n_good_local, jacobian_param_indices, BLOCK_RANGES_local, N_beta
#
# All @everywhere const values are broadcast to workers here.

############## Load and distribute constants ##############

coefs                         = CSV.read(joinpath(input_folder, "stats.csv"), DataFrame)
distances_local               = NPZ.npzread(joinpath(input_folder, "distances.npy"))
filter_N_upstream_local       = NPZ.npzread(joinpath(input_folder, "filter_N_upstream.npy"))
w_rs_local                    = NPZ.npzread(joinpath(input_folder, "w_rs.npy")) # Upstream average wage.
#w_rs_local .= ifelse.(w_rs_local .!= 0, w_rs_local ./ w_rs_local, w_rs_local)
regional_wages_local          = NPZ.npzread(joinpath(input_folder, "regional_wages.npy")) # Downstream region average wage for downstream firms.
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
@everywhere const N_rho               = $(2000)
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


# ── Gamma threshold: drop small sourcing-share pairs from active set ─────
# Insert after X_rs_local is loaded, BEFORE T_mask_local computation.
gamma_threshold = 0.00   # (s,r) pairs with γ_{rs} < threshold are zeroed out

emp_gamma_ls_local = permutedims(NPZ.npzread(joinpath(input_folder, "emp_gamma_ls.npy")))
# Shape: (R_full, S_) — indexed as emp_gamma_ls_local[r, s]

n_dropped = 0
if gamma_threshold != 0
    for s in 1:S_
        sector_sum_before = sum(emp_gamma_ls_local[:, s])
        for r in 1:R_full
            if 0 < emp_gamma_ls_local[r, s] <= gamma_threshold
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
    end
end
println("Gamma threshold=$gamma_threshold: dropped $n_dropped (s,r) pairs")
# ──────────────────────────────────────────────────────────────────────────
@everywhere const emp_gamma_ls   = $(emp_gamma_ls_local)#$(permutedims(NPZ.npzread(joinpath(input_folder, "emp_gamma_ls.npy"))))


# Reference region per sector: largest empirical sourcing share among active regions
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


X_dr_local = CSV.read(joinpath(input_folder, "X_dr.csv"), DataFrame).X_dr
X_dr_local = X_dr_local[N_downstream_per_region_local .!= 0]
emp_pi_r_local = X_dr_local ./ sum(X_dr_local)
@everywhere const emp_pi_r_full  = $(emp_pi_r_local)
@everywhere const emp_pi_r       = $(emp_pi_r_local)

# Reference downstream region for A normalization: largest empirical sales share
A_REF_REGION_local = argmax(emp_pi_r_local)
println("A_REF_REGION = $A_REF_REGION_local (π_r = $(emp_pi_r_local[A_REF_REGION_local]))")
@everywhere const A_REF_REGION = $A_REF_REGION_local

if n_coef == 1
    reg_coef_local = [coefs[3, "value"]]
else
    reg_coef_local = NPZ.npzread(joinpath(input_folder, "reg_coef_$(n_coef).npy"))
end
@everywhere const reg_coef       = $reg_coef_local
@everywhere const N_beta         = $(n_coef)

T_gravity = zeros(S_, R_full)
for s in 1:S_
    idxs = SECTOR_GOOD_INDICES_local[s]
    for g in idxs
        l = GOOD_R_local[g]
        T_gravity[s, l] = max(emp_gamma_ls[l, s] * (w_rs_local[l]^theta), 1e-12)
    end
end
@everywhere const T_rs_init = $(T_gravity)

# Moment block sizes + MOMENT_MASK
n_labor    = 1
n_industry = length(vec(agg_industry_share_local))
n_gamma    = length(vec(permutedims(NPZ.npzread(joinpath(input_folder, "emp_gamma_ls.npy")))))
n_reg      = length(reg_coef_local)
n_pi       = length(emp_pi_r_local)
N_moments_full = n_labor + n_industry + n_pi + n_reg + n_gamma

empirical_moments_local = vcat(
    [agg_labor_share],
    vec(agg_industry_share_local),
    emp_pi_r_local,
    reg_coef_local,
    vec(permutedims(NPZ.npzread(joinpath(input_folder, "emp_gamma_ls.npy"))))
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

BLOCK_RANGES_local = compute_block_ranges(n_labor, n_industry, n_pi, n_reg, n_gamma, moment_mask_local)
@everywhere const BLOCK_RANGES = $BLOCK_RANGES_local
@everywhere const BLOCK_NAMES  = ("labor", "industry", "pi_r", "reg_coef", "gamma_ls")

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

LOG_DIST_DOWNSTREAM_local = log.(max.(distances_downstream_local, 1.0))
LOG_CLOSEST_DIST_local    = log.(max.(closest_plant_dist_local, 1.0))
@everywhere const LOG_DIST_DOWNSTREAM = $LOG_DIST_DOWNSTREAM_local
@everywhere const LOG_CLOSEST_DIST    = $LOG_CLOSEST_DIST_local

println("Constants distributed. N_moments=$N_moments, n_good=$n_good_local")

# Indices of identified parameters to use in Jacobian/inference.
# Excludes the S+2 flat directions created by internal normalizations:
#   - Ω^s[1]  (position 2)           : Omega_s ./= sum(Omega_s)
#   - A[A_REF_REGION] (position S+1+A_REF_REGION) : A ./= A[A_REF_REGION]
#   - T[s, T_REF_REGION[s]] for each s: T_mat[s,:] ./= T_mat[s, ref_r] (most important regions in empirical gamma_ls)
# New parameter layout: [Ω^L(1) | Ω^s(S) | A(R_down) | β(N_beta) | T(sum(T_MASK))]

_excluded = Set{Int}()
push!(_excluded, 2)              # Ω^s[1]: position 2 in new layout
push!(_excluded, S_ + 1 + A_REF_REGION_local)  # A[A_REF_REGION]: position S+1+ref in new layout
T_param_offset = 1 + S_ + R_down_ + N_beta
for s in 1:S_
    ref_r = T_REF_REGION_local[s]
    ref_r == 0 && continue
    flat_pos = (ref_r - 1) * S_ + s   # column-major index in vec(T_rs), shape (S, R)
    if T_mask_local[flat_pos]
        t_idx = count(T_mask_local[1:flat_pos])
        push!(_excluded, T_param_offset + t_idx)
    end
end
jacobian_param_indices = [i for i in 1:(1 + S_ + R_down_ + N_beta + sum(T_mask_local)) if i ∉ _excluded]
println("Jacobian will cover $(length(jacobian_param_indices)) identified parameters ($(length(_excluded)) normalized-out excluded).")
