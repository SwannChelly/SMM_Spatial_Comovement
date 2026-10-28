# Data loading and constant distribution — main specification only:
# granular varieties (finite N_s), comparative advantage at the attraction-area level
# with T_ls = T_{a(l)s} · n_ls^κ, cell-level cloglog extensive margin, T profiled.
#
# Expects in the calling scope: input_folder, output_folder, n_coef, n_tau,
# n_rho_inference, relax_n_lo. Everything is broadcast to the workers with
# `@everywhere const`.
#
# Parameter vector: [Ω^L(1) | Ω^s(S) | A(R_downstream) | α(N_TAU) | κ(1) | T(sum(T_MASK))]
# Moment vector   : [labor | industry(S) | π_r(R_d) | [β_d ; β_κ] | γ_{s,a} | Ḡ_s(0)(S)]

# ═══════════════════════════════════════════════════════════════════════════
# 1 — RAW DATA
# ═══════════════════════════════════════════════════════════════════════════

coefs = CSV.read(joinpath(input_folder, "stats.csv"), DataFrame)

# stats.csv is read BY NAME: a column `name`, else a row whose label column is `name`
# with the number in `value`.
function _require_stat(df, name::AbstractString)
    if name in names(df)
        v = collect(skipmissing(df[!, name]))
        isempty(v) || return Float64(v[1])
    end
    if "value" in names(df)
        for c in names(df)
            i = findfirst(x -> x isa AbstractString && lowercase(strip(x)) == name, df[!, c])
            i === nothing || return Float64(df[i, "value"])
        end
    end
    error("stats.csv has no `$name` entry")
end

distances_local               = NPZ.npzread(joinpath(input_folder, "distances.npy"))
filter_N_upstream_local       = NPZ.npzread(joinpath(input_folder, "filter_N_upstream.npy"))
w_rs_local                    = NPZ.npzread(joinpath(input_folder, "w_rs.npy"))
regional_wages_local          = NPZ.npzread(joinpath(input_folder, "regional_wages.npy"))
# Wages are normalised to 1 wherever observed (upstream and downstream alike).
w_rs_local .= ifelse.(w_rs_local .!= 0, w_rs_local ./ w_rs_local, w_rs_local)
regional_wages_local .= ifelse.(regional_wages_local .!= 0, regional_wages_local ./ regional_wages_local, regional_wages_local)
N_downstream_per_region_local = NPZ.npzread(joinpath(input_folder, "N_downstream_per_region.npy"))
agg_industry_share_local      = NPZ.npzread(joinpath(input_folder, "input_share.npy"))
domestic_share_local          = NPZ.npzread(joinpath(input_folder, "domestic_share.npy"))

# ═══════════════════════════════════════════════════════════════════════════
# 2 — CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════

S_, R_full = size(filter_N_upstream_local)
R_down_    = count(!=(0), N_downstream_per_region_local)
@everywhere const S = $(S_)
@everywhere const R = $(R_full)
@everywhere const R_downstream        = $(R_down_)
@everywhere const agg_industry_share  = $(agg_industry_share_local)
@everywhere const agg_labor_share     = $(_require_stat(coefs, "labor_share"))
@everywhere const domestic_share      = $(domestic_share_local)
@everywhere regional_wages            = $(regional_wages_local)
@everywhere const distances           = $(distances_local)
@everywhere const N_downstream_per_region = $(N_downstream_per_region_local)
@everywhere const epsilon             = $(_require_stat(coefs, "epsilon"))
@everywhere const lambda              = $(0.5)
@everywhere const nu                  = $(0.2)
@everywhere const nu_s                = $(ones(S_) .* 1.5)
# Read by the Python reporting (utils._theta_from_julia): keep the `const theta = $(…)` form.
@everywhere const theta               = $(1.768)
@everywhere const delta_r             = $(ones(R_full))

# Γ((θ+1−ν_s)/θ)^{1/(1−ν_s)}: constant of the closed-form sector price index used by the
# Sinkhorn inversion (profiling.jl).
@assert all(theta + 1 .> nu_s) "ν_s must be < θ+1 for the closed-form P_sr"
_gamma_factor_local = [SpecialFunctions.gamma((theta + 1 - nu_s[s]) / theta)^(1 / (1 - nu_s[s])) for s in 1:S_]
@everywhere const GAMMA_FACTOR = $_gamma_factor_local

# κ: one density elasticity, T_ls = T_{a(l)s} n_ls^κ, searched jointly with α in an
# absolute box. Block 4 is averaged over R_CELL_REG nested networks.
@everywhere const RELAX_N_LO     = $(Bool(relax_n_lo))
@everywhere const R_CELL_REG     = 100
@everywhere const CELL_PERM_SEED = 4_242_000
@everywhere const KAPPA_LO = -1.0
@everywhere const KAPPA_HI =  3.0

# α prior: the cell-level cloglog slope on `has_sup` is −θα.
@everywhere const P_alpha = $(-Float64(vec(NPZ.npzread(joinpath(input_folder, "beta_d_1.npy")))[1]) / theta)

println("\n lambda = $lambda, epsilon = $epsilon, nu = $nu, nu_s = $(nu_s[1]), theta = $theta")
RELAX_N_LO && println(" RELAX_N_LO = true: N_LO forced to 1 (the model's bound N^obs_s/R_downstream is NOT imposed)")

# ═══════════════════════════════════════════════════════════════════════════
# 3 — GEOGRAPHY
# ═══════════════════════════════════════════════════════════════════════════

downstream_regions_local        = findall(N_downstream_per_region_local .> 0)
distances_downstream_local      = distances_local[:, downstream_regions_local]
closest_plant_dist_local        = vec(minimum(distances_downstream_local, dims=2))
closest_downstream_region_local = vec(getindex.(argmin(distances_downstream_local, dims=2), 2))

# Distance bins of the extensive-margin regression (base category: d ≤ 50 km, or
# d ≤ 20 km with five bins). With one coefficient the regressor is ln d.
function distance_bin(d, n_bins)
    edges = n_bins == 5 ? (20, 50, 100, 150, 200) : n_bins == 4 ? (50, 100, 150, 200) : ()
    b = 0
    for e in edges
        d > e && (b += 1)
    end
    return b
end

# ═══════════════════════════════════════════════════════════════════════════
# 4 — COUNT MOMENT TARGET Ḡ_s(0) AND VARIETY-COUNT BOUNDS (G_K.csv)
# N_supplier_s / R_downstream ≤ N_s ≤ N_supplier_s (finite_sample2.tex §3.3).
# ═══════════════════════════════════════════════════════════════════════════

gk_df = CSV.read(joinpath(input_folder, "G_K.csv"), DataFrame)
_gk_col(cands) = (nm = names(gk_df); i = findfirst(c -> lowercase(c) in cands, nm);
                  i === nothing && error("G_K.csv is missing a column among $cands; has $nm"); nm[i])
c_sector = _gk_col(["a129", "sector"])
c_K      = _gk_col(["k"])
c_G      = _gk_col(["g(k)", "g_k", "gk", "g"])
c_N      = _gk_col(["n_supplier_s", "n_supplier", "n_suppliers"])
gk_sectors = sort(unique(gk_df[!, c_sector]))
@assert length(gk_sectors) == S_ "G_K.csv covers $(length(gk_sectors)) sectors, expected S=$S_"
G_target_local = fill(NaN, S_)
N_LO_local     = zeros(Int, S_)
N_HI_local     = zeros(Int, S_)
for (s, a129) in enumerate(gk_sectors)
    rows = findall(==(a129), gk_df[!, c_sector])
    nsup = unique(gk_df[rows, c_N])
    @assert length(nsup) == 1 "G_K.csv: N_supplier_s is not constant within A129=$a129"
    n_obs = Int(round(Float64(nsup[1])))
    @assert n_obs >= 1 "G_K.csv: N_supplier_s = $n_obs for A129=$a129"
    k0 = findfirst(i -> Int(round(Float64(gk_df[i, c_K]))) == 0, rows)
    k0 === nothing && error("G_K.csv has no K=0 row for A129=$a129")
    G_target_local[s] = Float64(gk_df[rows[k0], c_G])
    N_HI_local[s]     = n_obs
    N_LO_local[s]     = RELAX_N_LO ? 1 : max(1, cld(n_obs, R_down_))
end
@assert all(0 .<= G_target_local .<= 1) "G_K.csv: G(0) must lie in [0,1]"
println(" Ḡ_s(0) target: ", round.(G_target_local, digits=4))
println(" variety-count bounds N_LO = $N_LO_local, N_HI = $N_HI_local")
@everywhere const G_TARGET = $G_target_local
@everywhere const N_LO     = $N_LO_local
@everywhere const N_HI     = $N_HI_local

# Draws per (sector, cell). Must cover the largest candidate variety count.
@everywhere const N_rho           = $(max(100, maximum(N_HI_local)))
@everywhere const N_RHO_INFERENCE = $(Int(n_rho_inference))
println(" N_rho (optimisation) = $N_rho ; N_RHO_INFERENCE = $N_RHO_INFERENCE")

# ═══════════════════════════════════════════════════════════════════════════
# 5 — SOURCING SHARES, ATTRACTION AREAS, CELLS AND THE T-COLUMN SPACE
#
# Two index spaces:
#  • CELLS  — the simulated (sector, ZE) cells, CELL_MASK = filter == 1 inside active
#    areas (control cells included: they inherit T_{s,a} > 0). Drives n_good, GOOD_S/R.
#  • T COLUMNS — attraction areas: the columns of the T parameter matrix AND of the γ
#    moment block (T[s,a] column ↔ γ[s,a] row, both s-major). T[s,l] = T_par[s, AA_OF_ZE[l]] n_ls^κ.
# ═══════════════════════════════════════════════════════════════════════════

emp_gamma_ls_local = permutedims(NPZ.npzread(joinpath(input_folder, "emp_gamma_ls.npy")))  # (R, S)
@everywhere const emp_gamma_ls = $(emp_gamma_ls_local)
observed_supplier_local = permutedims(emp_gamma_ls_local) .> 0                               # (S, R)
size(observed_supplier_local) == size(filter_N_upstream_local) ||
    error("emp_gamma_ls.npy and filter_N_upstream.npy are not on the same (sector, ZE) grid.")
supplier_cells_local = (filter_N_upstream_local .== 1) .& observed_supplier_local

# ZE → attraction area, anchored on the ZE's closest downstream region. The model fixed
# effect and the empirical grouping must be the SAME partition.
AA_link_local = NPZ.npzread(joinpath(input_folder, "attraction_area_linkages.npy"))
@assert size(AA_link_local) == (R_full, R_down_) "attraction_area_linkages.npy must be (R, R_downstream)"
@assert all(sum(AA_link_local, dims=2) .== 1) "every ZE must belong to exactly one attraction area"
aa_of_ze_local = vec(getindex.(argmax(AA_link_local, dims=2), 2))
@assert aa_of_ze_local == closest_downstream_region_local (
    "attraction_area_linkages.npy disagrees with the closest downstream region for " *
    "$(count(aa_of_ze_local .!= closest_downstream_region_local)) ZE")
n_AA_local = R_down_
@everywhere const AA_OF_ZE = $aa_of_ze_local
@everywhere const n_AA     = $n_AA_local

# 𝒜⁺_s: (sector, area) pairs hosting at least one observed supplier.
aa_active_local = falses(S_, n_AA_local)
for s in 1:S_, l in 1:R_full
    supplier_cells_local[s, l] && (aa_active_local[s, aa_of_ze_local[l]] = true)
end

cell_mask_local = filter_N_upstream_local .== 1
_outside = count(cell_mask_local[s, l] && !aa_active_local[s, aa_of_ze_local[l]] for s in 1:S_, l in 1:R_full)
if _outside > 0
    @warn "filter_N_upstream keeps $_outside cells whose area hosts no supplier in that sector; dropped."
    for s in 1:S_, l in 1:R_full
        aa_active_local[s, aa_of_ze_local[l]] || (cell_mask_local[s, l] = false)
    end
end
@everywhere const CELL_MASK = $cell_mask_local
@everywhere const T_COL_DIM = $n_AA_local
@everywhere const T_ACTIVE  = $aa_active_local
@everywhere const T_GATHER  = $aa_of_ze_local

T_mask_local = vec(permutedims(aa_active_local))       # s-major over (S, n_AA)
@everywhere const T_MASK = $T_mask_local

# Simulated cells ("goods"), region-major (findall over (S, R): s fastest).
good_indices_local        = findall(cell_mask_local)
n_good_local              = length(good_indices_local)
GOOD_S_local              = [ci[1] for ci in good_indices_local]
GOOD_R_local              = [ci[2] for ci in good_indices_local]
SECTOR_GOOD_INDICES_local = [findall(GOOD_S_local .== s) for s in 1:S_]
SECTOR_GOOD_REGIONS_local = [GOOD_R_local[idx] for idx in SECTOR_GOOD_INDICES_local]
W_RS_FLAT_local = [w_rs_local[GOOD_R_local[g]] for g in 1:n_good_local]
# A zero wage would make a cell win every variety: control ZE without a wage
# observation get the normalised wage 1.
let bad = findall(<=(0), W_RS_FLAT_local)
    isempty(bad) || (@warn "$(length(bad)) simulated cells have no wage; set to 1."; W_RS_FLAT_local[bad] .= 1.0)
end
@everywhere const n_good              = $n_good_local
@everywhere const GOOD_S              = $GOOD_S_local
@everywhere const GOOD_R              = $GOOD_R_local
@everywhere const SECTOR_GOOD_INDICES = $SECTOR_GOOD_INDICES_local
@everywhere const SECTOR_GOOD_REGIONS = $SECTOR_GOOD_REGIONS_local
@everywhere const W_RS_FLAT           = $W_RS_FLAT_local
@everywhere const CELLS_OF_SECTOR     = $SECTOR_GOOD_INDICES_local
SECTOR_T_COLS_local = [findall(view(aa_active_local, s, :)) for s in 1:S_]
@everywhere const SECTOR_T_COLS = $SECTOR_T_COLS_local

println("\nEstimated set per sector:")
@printf("  %-8s %10s %10s %10s %10s\n", "sector", "ZE supply", "ZE control", "ZE total", "AA active")
for s in 1:S_
    n_sup = count(view(supplier_cells_local, s, :)); n_cell = count(view(cell_mask_local, s, :))
    @printf("  %-8d %10d %10d %10d %10d\n", s, n_sup, n_cell - n_sup, n_cell, count(view(aa_active_local, s, :)))
    n_sup <= 1 && @warn "sector $s has $n_sup supplier cell(s): no free γ moment / T column."
end
println(" simulated cells: $n_good_local ; T columns: $(count(T_mask_local))")

# γ target at the area level, summed over every cell of the area (control cells
# contribute 0 in the data and a positive share in the model — the unbiased match).
for s in 1:S_, l in 1:R_full
    (!cell_mask_local[s, l] && emp_gamma_ls_local[l, s] != 0) &&
        error("emp_gamma_ls[$l, $s] ≠ 0 for a cell outside CELL_MASK")
end
emp_gamma_T_local = zeros(Float64, n_AA_local, S_)
for s in 1:S_, l in 1:R_full
    cell_mask_local[s, l] && (emp_gamma_T_local[aa_of_ze_local[l], s] += emp_gamma_ls_local[l, s])
end
# Domestic-share-balanced target (Σ_a = 1): the Sinkhorn-compatible row margin.
emp_gamma_T_tilde_local = emp_gamma_T_local ./ reshape(max.(domestic_share_local, 1e-12), 1, S_)
@everywhere const EMP_GAMMA_T       = $emp_gamma_T_local
@everywhere const EMP_GAMMA_T_TILDE = $emp_gamma_T_tilde_local

# Reference area per sector (T = 1, γ moment dropped): the largest empirical share.
T_REF_REGION_local = [isempty(c) ? 0 : c[argmax(emp_gamma_T_local[c, s])]
                      for (s, c) in enumerate(SECTOR_T_COLS_local)]
@everywhere const T_REF_REGION = $T_REF_REGION_local

# ═══════════════════════════════════════════════════════════════════════════
# 6 — REMAINING TARGETS AND PARAMETER COUNTS
# Block 4 = [β_d (N_REG) ; β_κ (1)]: the cell-level cloglog of has_sup on the distance
# regressor(s) and ln n_ls, sector × AA fixed effect (data notebook).
# ═══════════════════════════════════════════════════════════════════════════

X_dr_local = CSV.read(joinpath(input_folder, "X_dr.csv"), DataFrame).X_dr
X_dr_local = X_dr_local[N_downstream_per_region_local .!= 0]
emp_pi_r_local = X_dr_local ./ sum(X_dr_local)
@everywhere const emp_pi_r_full = $(emp_pi_r_local)
@everywhere const emp_pi_r      = $(emp_pi_r_local)

_bd = vec(Float64.(NPZ.npzread(joinpath(input_folder, "beta_d_$(n_coef).npy"))))
_bk = vec(Float64.(NPZ.npzread(joinpath(input_folder, "beta_kappa_$(n_coef).npy"))))
length(_bd) == n_coef || error("beta_d_$(n_coef).npy has $(length(_bd)) entries, expected $n_coef")
length(_bk) == 1      || error("beta_kappa_$(n_coef).npy has $(length(_bk)) entries, expected 1")
reg_coef_local = vcat(_bd, _bk)
# κ prior: the ln n coefficient of the continuous-distance specification.
_bk1_path = joinpath(input_folder, "beta_kappa_1.npy")
kappa_prior_local = clamp(Float64(vec(NPZ.npzread(isfile(_bk1_path) ? _bk1_path :
                          joinpath(input_folder, "beta_kappa_$(n_coef).npy")))[1]), -0.9, 2.9)
n_reg = length(reg_coef_local)                # N_REG + 1
@everywhere const reg_coef    = $reg_coef_local
@everywhere const N_REG       = $(n_coef)      # number of DISTANCE coefficients
@everywhere const N_TAU       = $(n_tau)       # number of trade-cost parameters
@everywhere const KAPPA_PRIOR = $(kappa_prior_local)
_dist_labels = n_coef == 5 ? ["]20,50]", "]50,100]", "]100,150]", "]150,200]", ">200"] :
               n_coef == 4 ? ["]50,100]", "]100,150]", "]150,200]", ">200"] : ["ln d"]
@everywhere const REG_COEF_LABELS = $(vcat(_dist_labels, "ln n"))
println(" block 4 target: β_d = $(round.(_bd, digits=4)), β_κ = $(round(_bk[1], digits=4)) ; " *
        "priors α₀ = $(round(P_alpha, digits=4)), κ₀ = $(round(kappa_prior_local, digits=4))")

# ═══════════════════════════════════════════════════════════════════════════
# 7 — MOMENT VECTOR, MOMENT_MASK, BLOCK_RANGES
# Dropped: the first industry share and the first π_r (sum-to-one), the inactive
# (sector, area) γ entries and each sector's reference γ.
# ═══════════════════════════════════════════════════════════════════════════

n_labor, n_industry, n_pi = 1, S_, length(emp_pi_r_local)
n_gamma = S_ * n_AA_local
n_G0    = S_
off_gamma = n_labor + n_industry + n_pi + n_reg
empirical_moments_local = vcat([agg_labor_share], vec(agg_industry_share_local), emp_pi_r_local,
                               reg_coef_local, vec(emp_gamma_T_local), G_target_local)
moment_mask_local = trues(length(empirical_moments_local))
moment_mask_local[n_labor + 1] = false
moment_mask_local[n_labor + n_industry + 1] = false
for idx in 1:n_gamma
    T_mask_local[idx] || (moment_mask_local[off_gamma + idx] = false)
end
for s in 1:S_
    T_REF_REGION_local[s] > 0 && (moment_mask_local[off_gamma + (s - 1) * n_AA_local + T_REF_REGION_local[s]] = false)
end
empirical_moments_local = reshape(empirical_moments_local[moment_mask_local], 1, :)
N_moments = length(empirical_moments_local)

_block_sizes = [n_labor, n_industry, n_pi, n_reg, n_gamma, n_G0]
_cuts = cumsum(vcat(0, _block_sizes))
BLOCK_RANGES_local = ntuple(6) do k
    base = count(moment_mask_local[1:_cuts[k]])
    (base + 1):(base + count(moment_mask_local[(_cuts[k] + 1):_cuts[k + 1]]))
end
@everywhere const MOMENT_MASK          = $moment_mask_local
@everywhere const empirical_moments    = $(empirical_moments_local)
@everywhere const BLOCK_RANGES         = $BLOCK_RANGES_local
@everywhere const BLOCK_NAMES          = ("labor", "industry", "pi_r", "reg_coef", "gamma_ls", "G0")
@everywhere const Weight_matrix_custom = $(Diagonal(ones(N_moments)))

# ═══════════════════════════════════════════════════════════════════════════
# 8 — OPTIMISATION DRAWS (frozen per-sector Sobol nets: the criterion is a
# deterministic function of θ). Inference resamples with i.i.d. draws.
# ═══════════════════════════════════════════════════════════════════════════

u_draws_local, sample_weights_local = generate_draws(N_rho, n_good_local, :sobol)
@everywhere const U_DRAWS        = $u_draws_local
@everywhere const SAMPLE_WEIGHTS = $sample_weights_local

# ═══════════════════════════════════════════════════════════════════════════
# 9 — GEOGRAPHY BROADCASTS
# ═══════════════════════════════════════════════════════════════════════════

DistBin_local = [distance_bin(distances_downstream_local[i, j], n_coef) for i in 1:R_full, j in 1:R_down_]
LOG_DIST_DOWNSTREAM_local = log.(max.(distances_downstream_local, 1.0))
@everywhere const DOWNSTREAM_REGIONS        = $(downstream_regions_local)
@everywhere const DistBin                   = $(DistBin_local)
@everywhere const CLOSEST_PLANT_DIST        = $(closest_plant_dist_local)
@everywhere const CLOSEST_DOWNSTREAM_REGION = $(closest_downstream_region_local)
@everywhere const LOG_DIST_DOWNSTREAM       = $LOG_DIST_DOWNSTREAM_local
println(" N_moments = $N_moments, n_good = $n_good_local")

# ═══════════════════════════════════════════════════════════════════════════
# 10 — CELLS: n_ls AND THE CELL-LEVEL REGRESSION SET (cells_ls.csv)
#   LOG_N_LS (S, R)  ln n_ls on every simulated cell (enters T through n_ls^κ).
#   REG_CELL_*       regression set 𝒜 = CELL_MASK ∩ {d > 0}: good index, regressors
#                    [distance bins or ln d | ln n], sector × AA fixed effect, data y.
# ZE keys are normalised to 4-character strings so "51" and "0051" match.
# ═══════════════════════════════════════════════════════════════════════════

_ze_key(x) = (t = strip(string(x)); v = tryparse(Int, t); v === nothing ? t : lpad(string(v), 4, '0'))

_cdf = CSV.read(joinpath(input_folder, "cells_ls.csv"), DataFrame)
for c in ("A129", "ze2010", "n_ls", "has_sup")
    c in names(_cdf) || error("cells_ls.csv has no `$c` column")
end
filter_N_upstream_df = CSV.read(joinpath(input_folder, "filter_N_upstream.csv"), DataFrame)
_sec_keys = sort(unique(string.(filter_N_upstream_df.A129)))
_ze_keys  = sort(unique(_ze_key.(filter_N_upstream_df.ze2010)))
(length(_sec_keys) == S_ && length(_ze_keys) == R_full) ||
    error("filter_N_upstream.csv does not match filter_N_upstream.npy ($S_ × $R_full)")
_s_of = Dict(k => i for (i, k) in enumerate(_sec_keys))
_l_of = Dict(k => i for (i, k) in enumerate(_ze_keys))

log_n_ls_local = zeros(Float64, S_, R_full)
_has_n  = falses(S_, R_full)
_y_cell = fill(NaN, S_, R_full)
for row in eachrow(_cdf)
    sk = string(row.A129); lk = _ze_key(row.ze2010)
    (haskey(_s_of, sk) && haskey(_l_of, lk)) || continue
    si = _s_of[sk]; li = _l_of[lk]
    cell_mask_local[si, li] || continue
    n = Float64(row.n_ls)
    n >= 1 || error("cells_ls.csv: n_ls = $n < 1 for cell ($sk, $lk)")
    log_n_ls_local[si, li] = log(n)
    _has_n[si, li] = true
    _y_cell[si, li] = Float64(row.has_sup)
end
_missing_n = count(cell_mask_local .& .!_has_n)
_missing_n == 0 || error("$_missing_n simulated cells have no n_ls in cells_ls.csv (T_ls = T_a n_ls^κ needs it).")

reg_cell_g_local  = Int[]
reg_cell_fe_local = Int[]
reg_cell_y_local  = Float64[]
_rows_x = Vector{Vector{Float64}}()
_fe_id  = Dict{Tuple{Int,Int},Int}()
for g in 1:n_good_local
    si = GOOD_S_local[g]; li = GOOD_R_local[g]
    d = closest_plant_dist_local[li]
    (d > 0 && isfinite(_y_cell[si, li])) || continue
    x = zeros(Float64, n_reg)
    if n_coef == 1
        x[1] = log(d)
    else
        b = DistBin_local[li, closest_downstream_region_local[li]]
        (b > 0 && b <= n_coef) && (x[b] = 1.0)
    end
    x[end] = log_n_ls_local[si, li]
    push!(reg_cell_g_local, g); push!(_rows_x, x); push!(reg_cell_y_local, _y_cell[si, li])
    push!(reg_cell_fe_local, get!(_fe_id, (si, aa_of_ze_local[li]), length(_fe_id) + 1))
end
@everywhere const LOG_N_LS        = $log_n_ls_local
@everywhere const REG_CELL_G      = $reg_cell_g_local
@everywhere const REG_CELL_FE     = $reg_cell_fe_local
@everywhere const REG_CELL_X      = $(permutedims(reduce(hcat, _rows_x)))
@everywhere const REG_CELL_N_FE   = $(maximum(reg_cell_fe_local))
@everywhere const REG_CELL_Y_DATA = $reg_cell_y_local
println(" regression set 𝒜: $(length(reg_cell_g_local)) cells ($(Int(sum(reg_cell_y_local))) with a supplier), " *
        "$(length(_fe_id)) sector × AA groups")

# The Julia kernel on the DATA outcome must reproduce the data targets: same cells,
# regressors, fixed effect and group-drop rule on both sides.
let (b, ok) = cell_cloglog_beta(reg_cell_y_local)
    println(" cell cloglog on the data outcome (Julia) vs targets:")
    for i in 1:n_reg
        @printf("   %-10s  Julia %9.4f   target %9.4f\n", REG_COEF_LABELS[i], b[i], reg_coef_local[i])
    end
    gap = maximum(abs.(b .- reg_coef_local))
    (ok && gap < 1e-3) || @warn "the Julia cell cloglog does not reproduce beta_d/beta_kappa " *
        "(max gap $(round(gap, sigdigits=3)), converged = $ok): check the regression set."
end

# ═══════════════════════════════════════════════════════════════════════════
# 11 — T STARTING VALUE: closed-form Sinkhorn inversion of γ at the priors (α₀, κ₀),
# wages ≡ 1 and destination expenditure proxied by π̂_r. T is ref-normalised per sector.
# ═══════════════════════════════════════════════════════════════════════════

function invert_T_from_gamma(alpha0::Real, kappa0::Real; max_iter::Int=1000,
                             tol::Float64=1e-11, damping::Float64=0.5)
    nfac     = exp.(kappa0 .* log_n_ls_local)
    tau_negθ = exp.((-theta * alpha0) .* LOG_DIST_DOWNSTREAM_local)
    Ê = emp_pi_r_local ./ sum(emp_pi_r_local)
    T = zeros(Float64, S_, n_AA_local)
    for s in 1:S_
        cells_s = SECTOR_GOOD_REGIONS_local[s]; cols_s = SECTOR_T_COLS_local[s]
        (isempty(cells_s) || isempty(cols_s)) && continue
        ref = T_REF_REGION_local[s]
        for c in cols_s
            T[s, c] = max(emp_gamma_T_tilde_local[c, s], 1e-12)
        end
        T[s, cols_s] ./= T[s, ref]
        for _ in 1:max_iter
            Phi = zeros(Float64, R_down_)
            for l in cells_s, dr in 1:R_down_
                Phi[dr] += T[s, aa_of_ze_local[l]] * nfac[s, l] * tau_negθ[l, dr]
            end
            M_col = zeros(Float64, n_AA_local)
            for l in cells_s
                M = 0.0
                for dr in 1:R_down_
                    Phi[dr] > 1e-300 && (M += tau_negθ[l, dr] * Ê[dr] / Phi[dr])
                end
                M_col[aa_of_ze_local[l]] += nfac[s, l] * M
            end
            T_new = Dict{Int,Float64}()
            for c in cols_s
                Tc = M_col[c] > 1e-300 ? max(emp_gamma_T_tilde_local[c, s], 1e-12) / M_col[c] : T[s, c]
                T_new[c] = exp((1 - damping) * log(T[s, c]) + damping * log(Tc))
            end
            max_rel = 0.0
            for c in cols_s
                Tn = T_new[c] / T_new[ref]
                max_rel = max(max_rel, abs(log(Tn) - log(T[s, c])))
                T[s, c] = Tn
            end
            max_rel < tol && break
        end
    end
    return T
end
T_init_local = invert_T_from_gamma(P_alpha, kappa_prior_local)
let v = T_init_local[T_init_local .> 0]
    (isempty(v) || !all(isfinite, v)) && error("γ-inversion produced a non-finite T starting value")
    @printf(" T init (γ-inversion at α₀): range [%.3g, %.3g]\n", minimum(v), maximum(v))
end
@everywhere const T_rs_init = $(T_init_local)

# Search box: α in [BOUND_LO, min(BOUND_HI, ALPHA_MAX/α₀)] × α₀ (prior anchor when N_TAU == 1),
# κ in [KAPPA_LO, KAPPA_HI].
@everywhere const TAU_PRIOR = $(n_tau == 1 ? [P_alpha] : nothing)
@everywhere const BOUND_LO  = 0.1
@everywhere const BOUND_HI  = 10
@everywhere const ALPHA_MAX = 1.5

# ═══════════════════════════════════════════════════════════════════════════
# 12 — LABELS AND INFERENCE BOOKKEEPING
# ═══════════════════════════════════════════════════════════════════════════

_sector_names = _sec_keys
_ze_names     = _ze_keys
_tcol_names   = ["AA" * _ze_names[downstream_regions_local[a]] for a in 1:n_AA_local]

_moment_labels = String["labor"]
for s in 2:S_;            push!(_moment_labels, "Omega_s[$(_sector_names[s])]"); end
for j in 2:n_pi;          push!(_moment_labels, "pi_r[$(_ze_names[downstream_regions_local[j]])]"); end
for b in 1:n_reg;         push!(_moment_labels, b <= n_coef ? "beta_d[$(REG_COEF_LABELS[b])]" : "beta_kappa"); end
for s in 1:S_, c in 1:n_AA_local
    moment_mask_local[off_gamma + (s - 1) * n_AA_local + c] && push!(_moment_labels, "gamma[$(_sector_names[s])-$(_tcol_names[c])]")
end
for s in 1:S_;            push!(_moment_labels, "G0[$(_sector_names[s])]"); end
@assert length(_moment_labels) == N_moments
@everywhere const MOMENT_LABELS = $_moment_labels
@everywhere const SECTOR_NAMES  = $_sector_names

# Parameter labels over the FULL layout, and the identified T columns (reference
# areas dropped) — the T block the inference reports.
_param_labels = vcat(["Omega_L"], ["Omega_s[$s]" for s in _sector_names],
                     ["A[$(_ze_names[r])]" for r in downstream_regions_local],
                     [n_tau == 1 ? "alpha" : "alpha_$b" for b in 1:n_tau], ["kappa"],
                     ["T[$(_sector_names[(p - 1) ÷ n_AA_local + 1])-$(_tcol_names[(p - 1) % n_AA_local + 1])]"
                      for p in findall(T_mask_local)])
_T_ref_reduced = [count(T_mask_local[1:((s - 1) * n_AA_local + T_REF_REGION_local[s])])
                  for s in 1:S_ if T_REF_REGION_local[s] > 0]
@everywhere const PARAM_LABELS   = $_param_labels
@everywhere const T_ID_REDUCED   = $([i for i in 1:count(T_mask_local) if i ∉ _T_ref_reduced])

# Per sector: positions of its retained γ moments inside the γ block, the sector total
# c_s and the empirical reference share — to rebuild the dropped reference γ.
_gamma_ref_map = NamedTuple{(:sector, :local_positions, :c_s, :emp_ref), Tuple{Int, Vector{Int}, Float64, Float64}}[]
let pos_of = Dict{Int,Int}(), k = 0
    for slot in 1:n_gamma
        moment_mask_local[off_gamma + slot] && (k += 1; pos_of[slot] = k)
    end
    for s in 1:S_
        ref = T_REF_REGION_local[s]; ref == 0 && continue
        pos = [pos_of[(s - 1) * n_AA_local + c] for c in 1:n_AA_local if haskey(pos_of, (s - 1) * n_AA_local + c)]
        isempty(pos) || push!(_gamma_ref_map, (sector = s, local_positions = pos,
                              c_s = domestic_share_local[s], emp_ref = emp_gamma_T_local[ref, s]))
    end
end
@everywhere const GAMMA_REF_MAP = $_gamma_ref_map

# On-disk data covariance of the inference moments, layout β_d (N_REG) → β_κ (1) →
# γ (active, non-reference areas, s-major) → Ḡ_s(0) (S). Steps 2–4 need it.
@everywhere const SIGMA_DATA_FILE = $("Sigma_aa_cell_cloglog" * (n_coef == 1 ? "_1" : "") * ".npy")
println(" constants distributed.")
