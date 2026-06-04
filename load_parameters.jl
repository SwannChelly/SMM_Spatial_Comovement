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
#   N_TAU = n_tau   — trade-cost parameter count (length of the β vector in unpack_params/build_tau)
#
# All @everywhere const values are broadcast to workers here.

############## Load and distribute constants ##############

coefs                         = CSV.read(joinpath(input_folder, "stats.csv"), DataFrame)
distances_local               = NPZ.npzread(joinpath(input_folder, "distances.npy"))
filter_N_upstream_local       = NPZ.npzread(joinpath(input_folder, "filter_N_upstream.npy"))
w_rs_local                    = NPZ.npzread(joinpath(input_folder, "w_rs.npy")) # Upstream average wage.
w_rs_local .= ifelse.(w_rs_local .!= 0, w_rs_local ./ w_rs_local, w_rs_local)
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

# Γ((θ+1-ν_s)/θ)^{1/(1-ν_s)} — constant factor in EK closed-form price index P_sr.
# Precomputed once; used by compute_prices_analytical in model_analytical.jl.
# Requires θ+1 > ν_s (checked here); for current calibration 2.768 > 2.5 ✓
begin
    _nu_s_local  = ones(S_) .* 2.5
    _theta_local = 1.768
    @assert all(_theta_local + 1 .> _nu_s_local) "ν_s must be < θ+1 for closed-form P_sr"
    import SpecialFunctions
    _gamma_factor_local = [SpecialFunctions.gamma((_theta_local + 1 - _nu_s_local[s]) / _theta_local)^(1 / (1 - _nu_s_local[s])) for s in 1:S_]
    @everywhere const GAMMA_FACTOR = $_gamma_factor_local
end
@everywhere const delta_r             = $(ones(R_full))
@everywhere const Weight_matrix       = $(nothing)

# ── Gamma threshold: drop small sourcing-share pairs from active set ─────
# Must precede T_mask_local so pruned pairs are excluded from T_MASK/n_good.
gamma_threshold = 0.025   # (s,r) pairs with γ_{rs} < threshold are zeroed out
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
if n_coef == 1
    reg_coef_local = [coefs[3, "value"]]
else
    reg_coef_local = NPZ.npzread(joinpath(input_folder, "reg_coef_$(n_coef).npy"))
end
@everywhere const reg_coef = $reg_coef_local
# N_REG: number of reg_coef moments (distance-bin regression coefficients).
# N_TAU: number of trade-cost parameters (length of β in unpack_params/build_tau).
# For standard runs n_tau == n_coef so N_TAU == N_REG; the split enables N_TAU=1
# (power-law τ = d^α) with N_REG=4 (four binned reg-coef moments, over-identified).
n_reg = length(reg_coef_local)   # actual moment count from loaded data
@everywhere N_REG = $(n_reg)
@everywhere N_TAU = $(n_tau)

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
n_gamma    = length(vec(emp_gamma_ls_local))
n_reg      = length(reg_coef_local)
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
#   - A[1]    (position S+2)          : A ./= A[1]
#   - T[s, T_REF_REGION[s]] for each s: T_mat[s,:] ./= T_mat[s, ref_r] (most important regions in empirical gamma_ls)
# New parameter layout: [Ω^L(1) | Ω^s(S) | A(R_down) | β(N_TAU) | T(sum(T_MASK))]

_excluded = Set{Int}()
push!(_excluded, 2)              # Ω^s[1]: position 2 in new layout
push!(_excluded, S_ + 2)         # A[1]:   position S+2 in new layout
T_param_offset = 1 + S_ + R_down_ + N_TAU
for s in 1:S_
    ref_r = T_REF_REGION_local[s]
    ref_r == 0 && continue
    flat_pos = (ref_r - 1) * S_ + s   # column-major index in vec(T_rs), shape (S, R)
    if T_mask_local[flat_pos]
        t_idx = count(T_mask_local[1:flat_pos])
        push!(_excluded, T_param_offset + t_idx)
    end
end
jacobian_param_indices = [i for i in 1:(1 + S_ + R_down_ + N_TAU + sum(T_mask_local)) if i ∉ _excluded]
println("Jacobian will cover $(length(jacobian_param_indices)) identified parameters ($(length(_excluded)) normalized-out excluded).")


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

# ── Human-readable labels for moments and parameters (for inference reports) ─
# Two SEPARATE axes:
#   • MOMENT_LABELS : one per kept moment, in MOMENT_MASK order (= rows of J,
#     = order of empirical_moments / simulated vectors). Blocks: labor, industry,
#     pi_r, reg_coef, gamma_ls — each already ref/sum-to-1-dropped.
#   • PARAM_LABELS  : one per identified parameter, in jacobian_param_indices
#     order (= columns of J). Layout [Ω^L | Ω^s | A | β | T], minus the S+2
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
# Layout: [Ω^L(1) | Ω^s(S) | A(R_down) | β(N_TAU) | T(active, sector-major)]
_param_labels_full = String[]
push!(_param_labels_full, "Omega_L")
for s in 1:S_;            push!(_param_labels_full, "Omega_s[$(_sector_names[s])]"); end
for r in 1:R_down_;       push!(_param_labels_full, "A[$(_ze_names[downstream_regions_local[r]])]"); end
for b in 1:N_TAU;         push!(_param_labels_full, N_TAU == 1 ? "alpha" : "beta_$b"); end
# T entries follow vec(T_rs) column-major over (S,R) restricted to T_MASK.
# Recover (s,r) for each active T slot in the same order T_reduced is laid out.
for flat_pos in findall(T_mask_local)             # column-major: r outer, s inner
    s = ((flat_pos - 1) % S_) + 1
    r = ((flat_pos - 1) ÷ S_) + 1
    push!(_param_labels_full, "T[$(_sector_names[s])-$(_ze_names[r])]")
end
@assert length(_param_labels_full) == 1 + S_ + R_down_ + N_TAU + sum(T_mask_local)
_param_labels = _param_labels_full[jacobian_param_indices]
@everywhere const PARAM_LABELS = $_param_labels
println("Built $(length(_moment_labels)) moment labels and $(length(_param_labels)) parameter labels.")