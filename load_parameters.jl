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
@everywhere const epsilon             = $(coefs[1, "value"])
@everywhere const P_alpha             = $(coefs[4, "value"]) #Prior on alpha
@everywhere const lambda              = $(0.5)
@everywhere const nu                  = $(0.2)
@everywhere const nu_s                = $(ones(S_) .* 1.5)
@everywhere const theta               = $(1.)#1.768

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
# SECTION 2b — MODELLING FLAGS (granularity + comparative-advantage level)
# Two INDEPENDENT switches (documentation/plan_granular_aa.md §0). With both at
# their legacy values the estimator is byte-identical to the continuum ZE-level
# model, which stays a fully supported configuration.
#
#   GRANULAR = false : continuum extensive margin, NO count moment (block 6),
#                      5-block moment vector — the model as it stood.
#   GRANULAR = true  : finite N_s varieties per sector, block 6 = the empty-cell
#                      share Ḡ_s(0), N_s profiled by closed-form integer bisection.
#
#   CA_LEVEL = :ze   : comparative advantage T (and the γ block) per (sector, ZE).
#   CA_LEVEL = :aa   : T (and the γ block) per (sector, ATTRACTION AREA); ZE inside
#                      an active area inherit T_{s,a} > 0, so their zeros become model
#                      predictions rather than exogenous.
#
# `granular = true` REQUIRES `ca_level = :aa`. Under ZE-level comparative advantage
# the estimated cell set is the supplier cells alone, so a cell can never come out
# empty: the count moment has no content, and the model would be asked to fit T > 0
# for regions whose observed γ_ls is 0. Granularity only means something once cells
# inherit their area's comparative advantage. The reverse combination
# (granular = false, ca_level = :aa) is a legitimate intermediate — the AA-level
# continuum — so it only warns.
# ═══════════════════════════════════════════════════════════════════════════

granular_local = (@isdefined(granular)) ? Bool(granular) : false
ca_level_local = (@isdefined(ca_level)) ? Symbol(ca_level) : :ze
@assert ca_level_local in (:ze, :aa) "ca_level must be :ze or :aa, got :$ca_level_local"
if granular_local && ca_level_local != :aa
    error("granular=true requires ca_level=:aa (got :$ca_level_local). Under ZE-level " *
          "comparative advantage only the supplier cells are estimated, so no cell can " *
          "come out empty: the count moment Ḡ_s(0) has no content, and the estimator " *
          "would be fitting T > 0 for regions whose observed γ_ls is 0. The extensive " *
          "margin becomes a model prediction only once cells inherit T_{s,a} from their " *
          "attraction area (finite_sample2.tex, Table 1).")
end
if !granular_local && ca_level_local == :aa
    @warn "ca_level=:aa with granular=false is the AA-level CONTINUUM: γ is aggregated to " *
          "attraction areas and control cells are simulated, but the extensive margin is " *
          "still degenerate and there is no count moment. Make sure the on-disk Σ file is " *
          "the AA one (its γ rows must match the γ moments)."
end
@everywhere const GRANULAR = $(granular_local)
@everywhere const CA_LEVEL = $(QuoteNode(ca_level_local))
println("\nGRANULAR = $granular_local ; CA_LEVEL = :$ca_level_local")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2c — GEOGRAPHY LOCALS (hoisted from SECTION 10)
# The downstream-region index set and each upstream ZE's NEAREST downstream region
# are needed early: the attraction-area map (SECTION 3b) must be asserted against
# CLOSEST_DOWNSTREAM_REGION, and the AA-level active set feeds T_MASK (SECTION 4).
# Only the locals are computed here; SECTION 10 still owns the @everywhere
# broadcasts (and the DistBin/log-distance precompute, which needs N_REG).
# ═══════════════════════════════════════════════════════════════════════════

downstream_regions_local        = findall(N_downstream_per_region_local .> 0)
distances_downstream_local      = distances_local[:, downstream_regions_local]
closest_plant_dist_local        = vec(minimum(distances_downstream_local, dims=2))
closest_downstream_region_local = vec(getindex.(argmin(distances_downstream_local, dims=2), 2))

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2d — COUNT MOMENT TARGET, VARIETY-COUNT BOUNDS, AND N_rho
# Read before N_rho is bound, because under GRANULAR the draw count per sector IS
# the variety block width and must cover the largest candidate N_s.
# ═══════════════════════════════════════════════════════════════════════════

# ── Block 6 target: the count moment Ḡ_s(0) (granular mode only) ────────────
# G_K.csv carries A129 (sector), K, G(K) = Pr(K_ls ≤ K) and the observed distinct
# supplier-firm count N_supplier_s, repeated on every row of a sector. G(0) is the
# share of ZE with ZERO suppliers; the denominator is the ZE inside ACTIVE attraction
# areas — the same 𝒜⁺ rule as AA_ACTIVE and CELL_MASK (finite_sample2.tex §3.2).
# The variety-count bounds come from the firm count (finite_sample2.tex §3.3):
#     N_supplier_s / R_downstream  ≤  N_s  ≤  N_supplier_s
G_target_local = Float64[]
N_LO_local     = Int[]
N_HI_local     = Int[]
if granular_local
    gk_path = joinpath(input_folder, "G_K.csv")
    isfile(gk_path) || error("granular=true requires $(gk_path) (columns A129, K, G(K), N_supplier_s).")
    gk_df = CSV.read(gk_path, DataFrame)
    _gk_col(cands) = begin
        nm = names(gk_df)
        hit = findfirst(c -> lowercase(c) in cands, nm)
        hit === nothing && error("G_K.csv is missing a column among $(cands); has $(nm)")
        nm[hit]
    end
    c_sector = _gk_col(["a129", "sector"])
    c_K      = _gk_col(["k"])
    c_G      = _gk_col(["g(k)", "g_k", "gk", "g"])
    c_N      = _gk_col(["n_supplier_s", "n_supplier", "n_suppliers"])
    # Sector ids in G_K.csv are the A129 labels; map them to 1..S by sorted order,
    # matching how filter_N_upstream.csv's A129 is turned into sector indices below.
    gk_sectors = sort(unique(gk_df[!, c_sector]))
    @assert length(gk_sectors) == S_ (
        "G_K.csv covers $(length(gk_sectors)) sectors, expected S=$S_ " *
        "($(gk_sectors)) — the A129 label set must match filter_N_upstream.")
    G_target_local = fill(NaN, S_)
    N_LO_local     = zeros(Int, S_)
    N_HI_local     = zeros(Int, S_)
    for (s, a129) in enumerate(gk_sectors)
        rows = findall(==(a129), gk_df[!, c_sector])
        nsup = unique(gk_df[rows, c_N])
        @assert length(nsup) == 1 "G_K.csv: N_supplier_s is not constant within A129=$a129 (got $nsup)"
        n_obs = Int(round(Float64(nsup[1])))
        @assert n_obs >= 1 "G_K.csv: N_supplier_s = $n_obs for A129=$a129 must be ≥ 1"
        k0 = findfirst(i -> Int(round(Float64(gk_df[i, c_K]))) == 0, rows)
        k0 === nothing && error("G_K.csv has no K=0 row for A129=$a129 — Ḡ_s(0) is the targeted moment.")
        G_target_local[s] = Float64(gk_df[rows[k0], c_G])
        N_HI_local[s]     = n_obs
        N_LO_local[s]     = max(1, cld(n_obs, R_down_))
    end
    @assert all(0 .<= G_target_local .<= 1) "G_K.csv: G(0) must be a share in [0,1], got $G_target_local"
    println("\nCount moment Ḡ_s(0) target: ", round.(G_target_local, digits=4))
    println("Variety-count bounds  N_LO = $N_LO_local   N_HI = $N_HI_local")
end
@everywhere const G_TARGET = $G_target_local
@everywhere const N_LO     = $N_LO_local
@everywhere const N_HI     = $N_HI_local


# ── N_rho: draws per (sector, cell) ─────────────────────────────────────────
# The number of Monte-Carlo/QMC draws integrating over the variety continuum, in
# BOTH modes. Under GRANULAR it additionally sets the precision of `q̂` and hence of
# the profiled `N̂_s`; it is NOT tied to N_HI, because no prefix of the draws is ever
# taken (see SECTION 9b).
n_rho_local = 1000
@everywhere const N_rho = $(n_rho_local)
println("\n N_rho = $n_rho_local — Entreprise par secteur x region")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — EMPIRICAL SOURCING SHARES
# Loads the observed domestic sourcing shares γ_ls. Every (sector, ZE) pair kept by
# `filter_N_upstream` is estimated; there is no share threshold.
# ═══════════════════════════════════════════════════════════════════════════

emp_gamma_ls_local = permutedims(NPZ.npzread(joinpath(input_folder, "emp_gamma_ls.npy")))
# Shape: (R_full, S_) — indexed as emp_gamma_ls_local[r, s]
# ZE-level throughout, in BOTH modes. `EMP_GAMMA_T` (SECTION 3b) is the moment/T-space
# target derived from it — the same matrix under :ze, the AA aggregate under :aa.
@everywhere const emp_gamma_ls = $(emp_gamma_ls_local)

# ── Projective (domestic-share-balanced) γ target for the Sinkhorn inversion ──
# The observed γ_ls are DOMESTIC sourcing shares: per sector they sum to
# domestic_share[s] < 1 (the rest is imported / out-of-system), NOT to 1. The
# comparative-advantage inversion (invert_T_from_gamma below, and the profiling
# invert_T_ge) is a matrix-scaling / Sinkhorn problem whose ROW margin is this γ
# and whose COLUMN margin is the expenditure distribution ω = exp_{s,dr} — which
# sums to 1. Sinkhorn's theorem requires COMPATIBLE margins (Σ row = Σ column), so
# the target must be renormalized to the same total as ω:
#
#     emp_gamma_ls_tilde[r,s] = emp_gamma_ls[r,s] / domestic_share[s]   ⇒  Σ_r = 1.
#
# This is the "version projective" γ̃ = γ / s_dom of documentation/initialisation.md
# §2.1. It is legitimate because T is identified only up to a per-sector scale
# (the reference-region gauge T[s,ref]=1 absorbs the total), so matching γ "up to a
# constant" is matching γ̃ — the returned T is unchanged, but the balanced margins
# make the Sinkhorn precondition hold exactly rather than by cancellation.
emp_gamma_ls_tilde_local = emp_gamma_ls_local ./ reshape(max.(domestic_share_local, 1e-12), 1, S_)
@everywhere const emp_gamma_ls_tilde = $(emp_gamma_ls_tilde_local)
# ──────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — T-MASK + GOOD-PAIR INDEX MAPS
# Builds the bookkeeping that links each active (s,r) comparative-advantage
# parameter T[s,r] to its position in the flattened parameter/moment vectors.
# The s-major (region-minor) flattening here is the convention the Jacobian's
# parameter axis and γ-moment row axis share (see CLAUDE.md invariant).
# ═══════════════════════════════════════════════════════════════════════════

# ── Cells with OBSERVED SUPPLIER SALES ──────────────────────────────────────
# `filter_N_upstream` is BINARY: 1 ⟺ the (sector, ZE) cell enters the optimisation,
# 0 ⟺ it is dropped (no firms). Within the kept cells,
#   supplier cell ⟺ filter == 1 AND X_rs  > 0
#   control  cell ⟺ filter == 1 AND X_rs == 0   (firms, but no observed supplier)
#
# `supplier_cells` is therefore "the cells with supplier sales IN THE DATA". It plays
# two distinct roles and it is worth keeping them apart:
#   (a) under CA_LEVEL = :ze it IS the estimated set — both the simulated cells and
#       the T/γ support (the legacy model: a cell with no sales gets T ≡ 0);
#   (b) under :aa it is NOT the estimated cell set — it only DEFINES 𝒜⁺, i.e. which
#       attraction areas host at least one supplier in a sector. The cells actually
#       put into the optimisation are all of `filter == 1` inside those areas
#       (CELL_MASK below), control cells included, because they inherit T_{s,a} > 0.
supplier_cells_local = (filter_N_upstream_local .== 1) .& (X_rs_local .> 0)  # (S, R_full)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3b — ATTRACTION AREAS, CELL SET, AND THE T-COLUMN SPACE
#
# Two distinct index spaces from here on; conflating them is the single easiest
# way to break this file.
#
#  • CELL space   — the (sector, ZE) cells actually SIMULATED through the Ricardian
#    selection ("goods"). `CELL_MASK :: BitMatrix(S, R)`. Drives n_good, GOOD_S/R,
#    SR_TO_GOOD, SECTOR_GOOD_INDICES/REGIONS, W_RS_FLAT, U_DRAWS columns.
#      :ze → the supplier cells (`active_mat`) — exactly the legacy set.
#      :aa → every kept cell (`filter == 1`), control cells INCLUDED: under
#            area-level comparative advantage they inherit T_{s,a} > 0 and have real
#            productivity draws, so their zeros are model predictions. This is the
#            endogenisation (finite_sample2.tex, Table 1, second row).
#
#  • T-COLUMN space — the columns of the comparative-advantage parameter matrix,
#    which is ALSO the column space of the γ moment block (the CLAUDE.md invariant
#    "T[s,c] parameter column ↔ γ[s,c] moment row" is preserved verbatim).
#    `T_COL_DIM` is its width and `T_ACTIVE :: BitMatrix(S, T_COL_DIM)` its support.
#      :ze → T_COL_DIM = R,      T_ACTIVE = active_mat        (legacy: cells == columns)
#      :aa → T_COL_DIM = n_AA,   T_ACTIVE = AA_ACTIVE
#    `T_GATHER[l]` maps ZE l to its T column, so the model reads
#    `T[s, l] = T_par[s, T_GATHER[l]]` — the identity under :ze.
# ═══════════════════════════════════════════════════════════════════════════

# ── ZE → attraction area (anchored on the ZE's closest downstream region) ────
if ca_level_local == :aa
    aa_path = joinpath(input_folder, "attraction_area_linkages.npy")
    isfile(aa_path) || error("ca_level=:aa requires $(aa_path) — the (R, R_downstream) " *
                             "binary ZE→attraction-area incidence matrix.")
    AA_link_local = NPZ.npzread(aa_path)
    @assert size(AA_link_local) == (R_full, R_down_) (
        "attraction_area_linkages.npy has size $(size(AA_link_local)), expected " *
        "(R, R_downstream) = ($R_full, $R_down_)")
    @assert all(sum(AA_link_local, dims=2) .== 1) (
        "attraction_area_linkages.npy: every ZE must belong to EXACTLY one attraction " *
        "area (row sums must all equal 1)")
    aa_of_ze_local = vec(getindex.(argmax(AA_link_local, dims=2), 2))
    # The decisive gate (validation V1): the model's fixed-effect partition and the
    # empirical one must be the SAME partition, else the whole alignment argument of
    # finite_sample2.tex §1.2 fails.
    @assert aa_of_ze_local == closest_downstream_region_local (
        "attraction_area_linkages.npy disagrees with CLOSEST_DOWNSTREAM_REGION for " *
        "$(count(aa_of_ze_local .!= closest_downstream_region_local)) ZE. The attraction " *
        "area must be anchored on each ZE's NEAREST downstream region — the empirical " *
        "A129_AA grouping and the model fixed effect would otherwise be different partitions.")
else
    aa_of_ze_local = copy(closest_downstream_region_local)   # inert under :ze
end
n_AA_local = R_down_
@everywhere const AA_OF_ZE = $aa_of_ze_local
@everywhere const n_AA     = $n_AA_local

# ── 𝒜⁺_s : (sector, AA) pairs hosting at least one observed supplier ─────────
# SECTORAL: an area can be active in one sector and empty in another, so the
# sector dimension must be carried through (plan §2).
aa_active_local = falses(S_, n_AA_local)
for s in 1:S_, l in 1:R_full
    if filter_N_upstream_local[s, l] == 1 && X_rs_local[s, l] > 0
        aa_active_local[s, aa_of_ze_local[l]] = true
    end
end
@everywhere const AA_ACTIVE = $aa_active_local

# ── CELL_MASK : the simulated (sector, ZE) cells ─────────────────────────────
cell_mask_local = ca_level_local == :aa ? (filter_N_upstream_local .== 1) : copy(supplier_cells_local)
if ca_level_local == :aa
    # V1b — the filter must already be restricted to 𝒜⁺: the model is never asked to
    # explain why an ENTIRE area is empty (that is the boundary the fixed effect draws).
    _outside = [(s, l) for s in 1:S_, l in 1:R_full
                if cell_mask_local[s, l] && !aa_active_local[s, aa_of_ze_local[l]]]
    if !isempty(_outside)
        @warn "filter_N_upstream keeps $(length(_outside)) (sector, ZE) cells whose " *
              "attraction area hosts NO supplier in that sector. The filter is broader " *
              "than 𝒜⁺; intersecting (those cells are dropped from CELL_MASK)."
        for (s, l) in _outside
            cell_mask_local[s, l] = false
        end
    end
end
@everywhere const CELL_MASK = $cell_mask_local

# ── T-column space ──────────────────────────────────────────────────────────
T_col_dim_local = ca_level_local == :aa ? n_AA_local : R_full
T_active_local  = ca_level_local == :aa ? aa_active_local : supplier_cells_local
T_gather_local  = ca_level_local == :aa ? aa_of_ze_local : collect(1:R_full)
@everywhere const T_COL_DIM = $T_col_dim_local
@everywhere const T_ACTIVE  = $T_active_local
@everywhere const T_GATHER  = $T_gather_local

# T-mask isolates the (sector, T-column) pairs on which comparative advantage is
# estimated. s-major (column-minor) flattening — identical to the γ-moment
# convention, so the Jacobian's parameter axis and γ-moment row axis line up.
T_mask_local         = vec(permutedims(T_active_local))
T_mask_moment_local  = vec(permutedims(T_active_local))
@everywhere const T_MASK        = $T_mask_local
@everywhere const T_MASK_MOMENT = $T_mask_moment_local


# Bellow, we flatten the vector and store the sector and region coordinates of the simulated cells.
# Region-major order (findall over the (S,R) mask): s fastest, r slowest — unchanged.
good_indices_local        = findall(cell_mask_local)
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
# Every SIMULATED cell needs a strictly positive upstream wage: the Ricardian price
# is p = w·τ/z, so a zero wage would make that cell win every variety unconditionally.
# Under :aa the cell set now includes control ZE, which may carry no wage observation
# (w_rs = 0 in the raw data); they inherit the normalised w ≡ 1 used everywhere else.
let bad = findall(<=(0), W_RS_FLAT_local)
    if !isempty(bad)
        @warn "$(length(bad)) simulated cells have w_rs ≤ 0 (no wage observation); " *
              "setting their upstream wage to the normalised value 1.0 " *
              "(a zero wage would make them win every variety)."
        W_RS_FLAT_local[bad] .= 1.0
    end
end

@everywhere const n_good               = $n_good_local
@everywhere const GOOD_S               = $GOOD_S_local
@everywhere const GOOD_R               = $GOOD_R_local
@everywhere const SECTOR_GOOD_INDICES  = $SECTOR_GOOD_INDICES_local
@everywhere const SECTOR_GOOD_REGIONS  = $SECTOR_GOOD_REGIONS_local
@everywhere const SR_TO_GOOD           = $SR_TO_GOOD_local
@everywhere const W_RS_FLAT            = $W_RS_FLAT_local
# Plan naming: the cells (good indices) of each sector — the denominator of the
# count moment Ḡ_s(0) and the prefix-regression row set.
@everywhere const CELLS_OF_SECTOR      = $SECTOR_GOOD_INDICES_local

# ── Per-sector maps in the T-COLUMN space (T parameters and γ moments) ───────
# Analogue of SECTOR_GOOD_INDICES/REGIONS but over T columns, which under :aa are
# attraction areas rather than ZE. Identical to SECTOR_GOOD_REGIONS under :ze.
SECTOR_T_COLS_local = [findall(view(T_active_local, s, :)) for s in 1:S_]
@everywhere const SECTOR_T_COLS = $SECTOR_T_COLS_local

# ── Control cells: kept by the filter but with no observed supplier ──────────
# NEW BINARY ENCODING: control cell ⟺ filter == 1 AND X_rs == 0 (the old status-2).
#   :ze — comparative advantage is null there (T ≡ 0): NOT simulated, no T/γ/π_r
#         moment; they enter the extensive-margin regression only, as extra y=0
#         observations at their distance bin (see fast_weighted_regression).
#   :aa — they inherit their area's T_{s,a} > 0 and ARE simulated as ordinary goods
#         (they are in CELL_MASK), so the appended y=0 rows would double-count them;
#         the regression's control-row path is switched off (REG_INCLUDE_CONTROL).
control_indices_local = findall((filter_N_upstream_local .== 1) .& (X_rs_local .== 0))
CONTROL_S_local       = [ci[1] for ci in control_indices_local]
CONTROL_R_local       = [ci[2] for ci in control_indices_local]
n_control_local       = length(control_indices_local)
@everywhere const CONTROL_S = $CONTROL_S_local
@everywhere const CONTROL_R = $CONTROL_R_local
@everywhere const N_CONTROL = $n_control_local
println("Control cells (filter==1 & X_rs==0): $n_control_local " *
        (ca_level_local == :aa ? "(simulated as ordinary goods under :aa)." :
                                 "(extensive-margin y=0 observations, not simulated)."))
# ── Per-sector composition of the estimated set ─────────────────────────────
# The three counts that determine what the estimator is actually fitting:
#   ZE active  — cells with observed supplier sales (X_rs > 0)
#   ZE control — cells kept by the filter with NO observed supplier. Under :ze these
#                are extensive-margin y=0 rows only; under :aa they are SIMULATED,
#                inherit T_{s,a} > 0, and their zeros become model predictions —
#                which is exactly the variation the count moment reads.
#   AA active  — attraction areas hosting at least one supplier in that sector (𝒜⁺_s),
#                i.e. the number of free T columns before the reference is dropped.
println("\nEstimated set per sector:")
@printf("  %-8s %10s %10s %10s %10s %10s\n",
        "sector", "ZE active", "ZE control", "ZE total", "AA active", "AA total")
let ta = 0, tc = 0, tt = 0, taa = 0
    for s in 1:S_
        n_act  = count(view(supplier_cells_local, s, :))
        n_cell = count(view(cell_mask_local, s, :))
        n_ctl  = n_cell - n_act
        n_aa   = count(view(aa_active_local, s, :))
        ta += n_act; tc += n_ctl; tt += n_cell; taa += n_aa
        @printf("  %-8d %10d %10d %10d %10d %10d\n", s, n_act, n_ctl, n_cell, n_aa, n_AA_local)
        n_act == 0 && @warn "sector $s has NO cell with observed supplier sales — its γ " *
                            "moments and T column are undefined. Drop the sector."
        n_act == 1 && @warn "sector $s has a SINGLE supplier cell: it is the reference " *
                            "(dropped as sum-to-1 redundant), so the sector contributes no " *
                            "γ moment and T[s,·] has no free parameter."
    end
    @printf("  %-8s %10d %10d %10d %10d %10d\n", "TOTAL", ta, tc, tt, taa, S_ * n_AA_local)
end
println("\nSimulated cells (CELL_MASK): $n_good_local ; T columns (active (s,$(ca_level_local == :aa ? "AA" : "ZE")) pairs): $(count(T_mask_local))")

# ── γ target in the T-COLUMN space ───────────────────────────────────────────
# `EMP_GAMMA_T[c, s]` is the block-5 empirical target and the Sinkhorn row margin.
#   :ze → emp_gamma_ls itself (thresholded + renormalized), (R, S).
#   :aa → the attraction-area aggregate (n_AA, S), summed over EVERY cell of the
#         area in CELL_MASK, control cells included. Control cells contribute 0 on
#         the DATA side and a strictly positive γ on the MODEL side: that is the
#         correct treatment, not a mismatch. The data are one realisation of the
#         model, so a control cell has E[γ_ls] > 0 while its realised draw is 0, and
#         E[γ̂_ls] = γ_ls holds unconditionally — excluding those cells from the model
#         sum would compare a conditional-on-activity aggregate with an unconditional
#         one, biasing the aggregate downward (plan §2).
if ca_level_local == :aa
    for s in 1:S_, l in 1:R_full
        if !cell_mask_local[s, l] && emp_gamma_ls_local[l, s] != 0
            error("emp_gamma_ls[$l, $s] = $(emp_gamma_ls_local[l, s]) ≠ 0 for a cell " *
                  "outside CELL_MASK — the AA aggregate would silently drop observed mass.")
        end
    end
    emp_gamma_T_local = zeros(Float64, n_AA_local, S_)
    for s in 1:S_, l in 1:R_full
        cell_mask_local[s, l] || continue
        emp_gamma_T_local[aa_of_ze_local[l], s] += emp_gamma_ls_local[l, s]
    end
else
    emp_gamma_T_local = copy(emp_gamma_ls_local)
end
# Domestic-share-balanced target γ̃ (Σ_c = 1) — the Sinkhorn-compatible row margin.
emp_gamma_T_tilde_local = emp_gamma_T_local ./ reshape(max.(domestic_share_local, 1e-12), 1, S_)
@everywhere const EMP_GAMMA_T       = $emp_gamma_T_local
@everywhere const EMP_GAMMA_T_TILDE = $emp_gamma_T_tilde_local



# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — REFERENCE REGIONS
# Each sector's T and γ_ls shares are identified only up to a per-sector scale,
# so one region per sector is normalized to T=1 and its γ moment dropped. We pin
# that reference to the largest empirical sourcing share for numerical stability.
# ═══════════════════════════════════════════════════════════════════════════

# Reference column per sector: largest empirical sourcing share among ACTIVE T
# columns (ZE under :ze, attraction areas under :aa). Those columns always have
# T = 1 after unpack_params, so they are not estimated. The name T_REF_REGION is
# kept (every consumer indexes the T-column space with it); T_REF_AA is an alias
# documenting the :aa reading.
T_REF_REGION_local = Vector{Int}(undef, S_)
for s in 1:S_
    cols_s = SECTOR_T_COLS_local[s]
    if !isempty(cols_s)
        gamma_vals = [emp_gamma_T_local[c, s] for c in cols_s]
        T_REF_REGION_local[s] = cols_s[argmax(gamma_vals)]
    else
        T_REF_REGION_local[s] = 0
    end
end
@everywhere const T_REF_REGION = $T_REF_REGION_local
@everywhere const T_REF_AA     = $T_REF_REGION_local   # alias (:aa reading of the same map)


# ── log-T (φ) reparameterization index maps ──────────────────────────────────
# The optimizer searches φ_i = log(T_i / T_{s,ref}) over the FREE reduced-T
# positions only; each sector's reference entry is dropped (pinned T=1) so the S
# unidentified directions never enter the search space. Reduced ordering matches
# unpack_params: flat p=(s-1)*R+r (r fastest), kept where T_MASK is true.
T_reduced_s_local = Int[]
T_reduced_r_local = Int[]
let p = 0
    for s in 1:S_, c in 1:T_col_dim_local
        p += 1
        if T_mask_local[p]
            push!(T_reduced_s_local, s)
            push!(T_reduced_r_local, c)
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
            for s in 1:S_ if !isempty(SECTOR_T_COLS_local[s])) "each active sector needs a reference reduced index in T_MASK"
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
# ── Regression method for the extensive-margin moment (block 4) ──────────────
# :lpm     — linear probability model (fast_weighted_regression); distance coef is
#            the slope of P(supplier), and the target is reg_coef_$(n_coef).npy.
# :cloglog — complementary-log-log (fast_cloglog_regression); distance coef is +αθ
#            (outcome not_supply), and the target is reg_coef_cloglog_$(n_coef).npy.
# The entry point may define `reg_method`; default :lpm (backward-compatible; GMM's
# analytical quadrature is LPM-convention). main.jl (SMM) defaults it to :cloglog.
reg_method_local = (@isdefined(reg_method)) ? Symbol(reg_method) : :lpm
@assert reg_method_local in (:lpm, :cloglog) "reg_method must be :lpm or :cloglog, got :$reg_method_local"
@everywhere const REG_METHOD = $(QuoteNode(reg_method_local))

# Whether the control group (filter==2, y=0 rows) enters the extensive-margin
# regression. false ⇒ supplier pairs only WITH the log-z size control (the two are
# coupled in fast_*_regression). The entry point may define `include_control`; default
# true (production). Consumed by compute_moments.
include_control_local = (@isdefined(include_control)) ? Bool(include_control) : true
@everywhere const INCLUDE_CONTROL = $(include_control_local)

# ── Effective regression design flags (what compute_moments actually passes) ──
# :ze — the legacy coupling: the control-group y=0 rows and the log-z size control
#       are MUTUALLY EXCLUSIVE, because control cells have no productivity draw
#       (T ≡ 0 ⇒ z ≡ −∞).
# :aa — control cells inherit T_{s,a} > 0 and ARE simulated as ordinary goods, so
#       (a) appending them again as y=0 rows would double-count them ⇒ the
#           control-row path is OFF, and
#       (b) they carry a real z ⇒ the log-z size control is ON unconditionally.
#       This is the firm-level reduced form of finite_sample2.tex Prop. 1 exactly:
#       cloglog on `not_supply`, log distance and log own productivity, with an
#       area×sector fixed effect — and the log-z coefficient (= −θ) comes back as a
#       free over-identifying diagnostic.
reg_include_control_local = ca_level_local == :aa ? false : include_control_local
reg_include_size_local    = ca_level_local == :aa ? true  : !include_control_local
@everywhere const REG_INCLUDE_CONTROL = $(reg_include_control_local)
@everywhere const REG_INCLUDE_SIZE    = $(reg_include_size_local)
println("reg_method = :$reg_method_local ; include_control (control group) = $include_control_local " *
        "⇒ effective (control rows, log-z size control) = ($reg_include_control_local, $reg_include_size_local)")

# Read a scalar stat by name from stats.csv (a column named `name`, else a row whose
# label column equals `name`, with the number in `value`). Returns nothing if absent.
function _read_named_value(coefs_df, name::AbstractString)
    cols = names(coefs_df)
    if name in cols
        vals = collect(skipmissing(coefs_df[!, name]))
        !isempty(vals) && return Float64(vals[1])
    end
    if "value" in cols
        for cname in cols
            col = coefs_df[!, cname]
            idx = findfirst(x -> x isa AbstractString &&
                                 lowercase(strip(x)) == lowercase(name), col)
            idx !== nothing && return Float64(coefs_df[idx, "value"])
        end
    end
    return nothing
end

if reg_method_local == :cloglog
    if n_coef == 1
        _rc = _read_named_value(coefs, "reg_coef_cloglog_1")
        _rc === nothing && error("reg_method=:cloglog with n_coef=1 needs a `reg_coef_cloglog_1` " *
                                 "entry (column or labeled row) in stats.csv")
        reg_coef_local = [_rc]
    else
        reg_coef_local = NPZ.npzread(joinpath(input_folder, "reg_coef_cloglog_$(n_coef).npy"))
    end
else
    if n_coef == 1
        reg_coef_local = [coefs[3, "value"]]
    else
        reg_coef_local = NPZ.npzread(joinpath(input_folder, "reg_coef_$(n_coef).npy"))
    end
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
# SECTION 8 — EMPIRICAL MOMENT VECTOR + MOMENT_MASK + BLOCK_RANGES
# Stacks the five moment blocks into one target vector and builds MOMENT_MASK,
# which drops the linearly-dependent moments (first industry/pi_r share, inactive
# and per-sector reference γ_ls) so the system is identified. BLOCK_RANGES indexes
# each block within the masked vector; Weight_matrix_custom is the Step-1 metric.
# ═══════════════════════════════════════════════════════════════════════════

# Moment block sizes + MOMENT_MASK
n_labor    = 1
n_industry = length(vec(agg_industry_share_local))
n_gamma    = length(vec(emp_gamma_T_local))       # (T_COL_DIM × S) — ZE or AA level
n_G0       = granular_local ? S_ : 0              # block 6, appended (never inserted)
# n_reg already computed above (== length(reg_coef_local)) for the N_REG broadcast.
n_pi       = length(emp_pi_r_local)
N_moments_full = n_labor + n_industry + n_pi + n_reg + n_gamma + n_G0

empirical_moments_local = vcat(
    [agg_labor_share],
    vec(agg_industry_share_local),
    emp_pi_r_local,
    reg_coef_local,
    vec(emp_gamma_T_local),        # thresholded+renormalized, consistent with loss residuals
    G_target_local                 # empty unless GRANULAR
)

moment_mask_local = trues(N_moments_full)
# Remove first industry share.
moment_mask_local[n_labor + 1] = false
# Remove first pi_r (sum-to-1 redundancy).
moment_mask_local[n_labor + n_industry + 1] = false
# reg_coef block: no masking needed.
# Remove non-active gamma entries.
for idx in 1:(S_ * T_col_dim_local)
    if !T_mask_moment_local[idx]
        moment_mask_local[n_labor + n_industry + n_pi + n_reg + idx] = false
    end
end
# Remove reference-column gamma per sector (sum-to-1 redundancy, aligned with T normalization).
for s in 1:S_
    ref_c = T_REF_REGION_local[s]
    if ref_c > 0
        moment_mask_local[n_labor + n_industry + n_pi + n_reg + (s - 1) * T_col_dim_local + ref_c] = false
    end
end
# Block 6 (G0): every sector kept — no redundancy to drop.

empirical_moments_local = reshape(empirical_moments_local[moment_mask_local], 1, sum(moment_mask_local))
N_moments = sum(moment_mask_local)

@everywhere const MOMENT_MASK        = $moment_mask_local
@everywhere const empirical_moments  = $(empirical_moments_local)
@everywhere const K_max              = $(50)


# BLOCK_RANGES : Length of each block of moment. 5 blocks in legacy mode, 6 under
# GRANULAR (block 6 = Ḡ_s(0), APPENDED so every existing index is untouched).
BLOCK_RANGES_local = compute_block_ranges(n_labor, n_industry, n_pi, n_reg, n_gamma,
                                          moment_mask_local; n_G0 = n_G0)
@everywhere const BLOCK_RANGES = $BLOCK_RANGES_local
@everywhere const BLOCK_NAMES  = $(granular_local ?
    ("labor", "industry", "pi_r", "reg_coef", "gamma_ls", "G0") :
    ("labor", "industry", "pi_r", "reg_coef", "gamma_ls"))
@everywhere const N_BLOCKS     = $(granular_local ? 6 : 5)

w_vec = ones(N_moments)
w_vec[BLOCK_RANGES_local[4]] .= 1.0
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

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 9b — WHY GRANULARITY NEEDS NO SEPARATE DRAW STRUCTURE
#
# Under the profiled design the ONLY things the variety count touches are:
#   * the count moment  G_s(0) = mean_l (1 - qhat_ls)^{Nhat_s}  -- CLOSED FORM, and
#   * Nhat_s itself, located by a monotone integer bisection on that closed form.
#
# Both are functions of `qhat` alone, and `qhat_ls` -- the probability that cell l
# wins a given variety SOMEWHERE downstream -- is free of N_s (finite_sample2.tex,
# Lemma 2) and is just a column mean of `linkages_flat` over the ordinary draws. So:
#
#   * NO prefix of the draws is ever taken,
#   * NO replicated economy is ever simulated,
#   * every moment is computed ONCE, on the same draws, exactly as in the continuum.
#
# Simulation noise is therefore handled UNIFORMLY across all moment blocks by the
# draw design (DRAW_METHOD, :sobol by default) and by N_rho -- not by a replication
# device special-cased to one block. N_rho keeps its usual meaning and value; note
# only that it now also sets the precision of qhat, hence of Nhat_s
# (s.e.(qhat) ~ sqrt(q(1-q)/N_rho) before the QMC gain), so it is worth raising
# under GRANULAR if the count moment looks noisy.
#
# The price of this design is that block 4 is the CONTINUUM-limit regression
# beta(E[y]) rather than E[betahat(N_s)], so the auxiliary estimator's finite-sample
# bias is no longer cancelled by matching a binding function. It is measured by
# validation gate V12 (2-4% of beta in data-consistent configurations) and should be
# reported alongside alphahat, or added as a local offset -- see
# documentation/granular_validation.md Part III.
# ═══════════════════════════════════════════════════════════════════════════

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

# downstream_regions_local / distances_downstream_local / closest_* were hoisted to
# SECTION 2c (the attraction-area assertion and the AA active set need them early);
# only the broadcasts and the N_REG-dependent bin/log precompute live here.
@everywhere const DOWNSTREAM_REGIONS = $(downstream_regions_local)

DistBin_local = Array{Int}(undef, R_full, R_down_)
for i in 1:R_full, j in 1:R_down_
    DistBin_local[i, j] = distance_bin(distances_downstream_local[i, j])
end
@everywhere const DistBin = $(DistBin_local)

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
# The prior is the empirical cloglog distance slope, stored as `reg_coef_cloglog_1`
# (a column or a labeled row; with θ=1 the slope equals α). Returns nothing if absent.
_read_prior_alpha(coefs_df) = _read_named_value(coefs_df, "reg_coef_cloglog_1")

"""
    invert_T_from_gamma(prior_alpha; max_iter=1000, tol=1e-11, damping=0.5)

Fixed-point inversion of the sourcing-share map for the T starting values.
Returns an (S, R_full) matrix with active (s,r) entries set to the inverted
comparative advantage (per-sector normalized to the reference region) and
inactive entries left at 0. Uses trade costs τ[r,dr] = exp(α · log max(d,1))
(the power-law N_TAU==1 form in build_tau, evaluated at the α prior) and wages ≡ 1.

The Sinkhorn row target is the DOMESTIC-SHARE-BALANCED γ̃ = emp_gamma_ls /
domestic_share (Σ_r γ̃ = 1), so it is compatible with the expenditure column
margin Ê (Σ = 1) as Sinkhorn's theorem requires (initialisation.md §2.1). The
per-sector reference gauge makes the returned T identical to the raw-γ inversion.
"""
function invert_T_from_gamma(prior_alpha::Real; max_iter::Int=1000,
                             tol::Float64=1e-11, damping::Float64=0.5)
    θ = theta
    # τ^{-θ}[l, dr] under the power-law prior and wages ≡ 1. Trade costs are ZE-level
    # in both modes; only comparative advantage moves to the area level.
    tau_negθ = exp.((-θ * prior_alpha) .* LOG_DIST_DOWNSTREAM_local)   # (R_full, R_downstream)
    Ê = emp_pi_r_local ./ sum(emp_pi_r_local)                          # (R_downstream,) market size

    T = zeros(Float64, S_, T_col_dim_local)
    max_iters_used = 0
    for s in 1:S_
        cells_s = SECTOR_GOOD_REGIONS_local[s]     # ZE of the simulated cells of sector s
        cols_s  = SECTOR_T_COLS_local[s]           # active T columns (ZE under :ze, AA under :aa)
        (isempty(cells_s) || isempty(cols_s)) && continue
        ref = T_REF_REGION_local[s] > 0 ? T_REF_REGION_local[s] : cols_s[1]
        # Initialise at the observed shares (positive), normalized to the ref column.
        # Uses the domestic-share-balanced target γ̃ (Σ_c = 1 = Σ ω) so the row/column
        # margins are Sinkhorn-compatible (initialisation.md §2.1); the ref gauge makes
        # this identical to the raw-γ inversion but keeps the precondition explicit.
        for c in cols_s
            T[s, c] = max(emp_gamma_T_tilde_local[c, s], 1e-12)
        end
        T[s, cols_s] ./= T[s, ref]
        for it in 1:max_iter
            max_iters_used = max(max_iters_used, it)
            # Φ[dr] = Σ_{l∈cells} T[s, col(l)] τ^{-θ}[l,dr]  — the sum runs over CELLS,
            # since each ZE runs its own Ricardian competition even when it shares a
            # Fréchet scale with the rest of its area (finite_sample2.tex Ass. 2).
            Phi = zeros(Float64, R_down_)
            for l in cells_s, dr in 1:R_down_
                Phi[dr] += T[s, T_gather_local[l]] * tau_negθ[l, dr]
            end
            # γ_col[c] ∝ T[s,c] · M_col[c] with M_col[c] = Σ_{l∈c} Σ_dr τ^{-θ}[l,dr] Ê_dr / Φ[dr]
            # (T factors out of the within-area sum since it is constant there)
            # ⇒ T_new[c] ∝ γ̃[c,s] / M_col[c]. Damped log-space update; the map is
            # homogeneous degree 1 in T, so renormalize to the ref column each pass.
            M_col = zeros(Float64, T_col_dim_local)
            for l in cells_s
                M = 0.0
                for dr in 1:R_down_
                    Phi[dr] > 1e-300 && (M += tau_negθ[l, dr] * Ê[dr] / Phi[dr])
                end
                M_col[T_gather_local[l]] += M
            end
            T_new = Dict{Int,Float64}()
            for c in cols_s
                γ  = max(emp_gamma_T_tilde_local[c, s], 1e-12)   # balanced target γ̃ (Σ_c=1)
                Tc = M_col[c] > 1e-300 ? γ / M_col[c] : T[s, c]
                T_new[c] = exp((1 - damping) * log(T[s, c]) + damping * log(Tc))
            end
            ref_val = T_new[ref]
            max_rel = 0.0
            for c in cols_s
                Tn  = T_new[c] / ref_val                       # renormalize to ref=1
                rel = abs(log(Tn) - log(T[s, c]))
                rel > max_rel && (max_rel = rel)
                T[s, c] = Tn
            end
            max_rel < tol && break
        end
    end
    return T, max_iters_used
end

# The Sinkhorn inversion is the T starting value in BOTH modes (:ze and :aa) — there
# is no separate gravity guess. The old "gravity" init T ∝ γ·w^θ is just this
# inversion at α = 0 (τ ≡ 1 ⇒ market access M is common within a sector and cancels),
# so when the α prior is missing we fall back to α = 0 rather than to a second,
# redundant code path.
_prior_alpha = _read_prior_alpha(coefs)
_init_alpha  = _prior_alpha === nothing ? 0.0 : Float64(_prior_alpha)
if _prior_alpha === nothing
    @warn "prior_alpha (reg_coef_cloglog_1) not found in stats.csv — inverting T at α = 0 " *
          "(τ ≡ 1), which drops the trade-cost geometry. Add the stat to use it."
end
println("\nInverting T from γ (Sinkhorn) with α = $_init_alpha (wages ≡ 1), " *
        "target = $(ca_level_local == :aa ? "AA-aggregated" : "ZE-level") γ̃")
T_init_local, iters_used = invert_T_from_gamma(_init_alpha)
let active_vals = T_init_local[T_init_local .> 0]
    (isempty(active_vals) || !all(isfinite, active_vals)) &&
        error("γ-inversion produced a non-finite or empty T starting value — check " *
              "emp_gamma_ls / domestic_share / the attraction-area map.")
    @printf("  γ-inversion done (≤%d iters). T range [%.3g, %.3g], median %.3g\n",
            iters_used, minimum(active_vals), maximum(active_vals),
            sort(active_vals)[cld(length(active_vals), 2)])
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
# ZE-level only: the diagnostic contrasts market access ACROSS ZE, which has no
# counterpart once comparative advantage lives at the area level.
ca_level_local == :ze && try
    # Baseline = the SAME inversion at α = 0 (τ ≡ 1). With uniform trade costs the
    # market-access term M is common across the ZE of a sector and cancels, so this
    # reduces exactly to T ∝ γ_ls — the old "gravity" guess, obtained without a
    # second code path. Their ratio is therefore purely the market-access correction
    #   T(α)/T(0) = M(ref)/M(r),  M[s,r] = Σ_dr (π̂_dr/Φ_{s,dr}) d_{r,dr}^{-θα}.
    _T_alpha0, _ = invert_T_from_gamma(0.0)
    _ref_norm_gravity = begin
        G = fill(NaN, S_, R_full)
        for s in 1:S_
            regions_s = SECTOR_T_COLS_local[s]
            isempty(regions_s) && continue
            for r in regions_s
                G[s, r] = _T_alpha0[s, r] > 0 ? _T_alpha0[s, r] : NaN
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
            xlabel = "γ-inversion at α=0  T/T_ref  (τ≡1)",
            ylabel = "γ-inversion init  T/T_ref  (α=$alpha_val)",
            title  = "T starting values: α=0 (τ≡1) vs γ-inversion (α=$alpha_val)",
            colorbar_title = "access-weighted mean dist to markets (km)",
            markersize = 5, markeralpha = 0.7, legend = false)
        lo = min(minimum(xs), minimum(ys)); hi = max(maximum(xs), maximum(ys))
        Plots.plot!(p, [lo, hi], [lo, hi]; color = :black, ls = :dash)
        png = joinpath(output_folder, "T_init_alpha0_vs_inversion_a$(alpha_val).png")
        Plots.savefig(p, png)
        NPZ.npzwrite(joinpath(output_folder, "T_init_pairs_a$(alpha_val).npz"),
                     Dict("alpha0" => xs, "inversion" => ys, "dist_km" => ds,
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
            @printf("  correction  T(α)/T(0) = M(ref)/M(r) (only geography + α differ):\n")
            @printf("     min %.2f   median %.2f   max %.2f   mean|Δlog T| %.3f\n",
                    minimum(rr), med, maximum(rr), sum(abs.(lr)) / length(lr))
            @printf("  effective  log(T(α)/T(0)) ≈ %+.3f %+.3f·log d̄\n", b[1], b[2])
            @printf("     (functional form (d/d_ref)^{θα}=(·)^%.2f holds exactly only in the\n",
                    theta * alpha_val)
            @printf("      single-dominant-destination limit; M sums over all markets so the\n")
            @printf("      realized slope is geography-dependent, not a clean θα).\n")
            @printf("  ⇒ at α=%.2f the trade-cost geometry repositions T by a factor\n", alpha_val)
            @printf("    ~%.2f× (median) and up to ~%.1f× for the most market-remote regions;\n",
                    med, maximum(rr))
            @printf("    ≈1× for regions near the reference.\n")
        end
    end

    _t_init_compare(0.5)                                # the requested α = 0.5 case
    if !isapprox(_init_alpha, 0.5) && _init_alpha > 0
        _t_init_compare(_init_alpha)                    # also the actual α in use
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
    flat_pos = (s - 1) * T_col_dim_local + ref_r   # s-major index in vec(permutedims(T)), shape (T_COL_DIM, S)
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
local_index_of_full = Dict{Int,Int}()   # full-γ-slot (1..S*T_COL_DIM) → local γ position
for slot in 1:(S_ * T_col_dim_local)
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
    # retained T columns of sector s = active, non-reference
    local_positions = Int[]
    for c in 1:T_col_dim_local
        slot = (s - 1) * T_col_dim_local + c
        if haskey(local_index_of_full, slot)       # kept (active & not ref)
            push!(local_positions, local_index_of_full[slot])
        end
    end
    isempty(local_positions) && continue           # ref was the only column: no free γ, skip
    c_s     = domestic_share_local[s]
    emp_ref = emp_gamma_T_local[ref_r, s]          # already thresholded+renormalized
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
# T-column names: ZE names under :ze, attraction-area names (anchored on the AA's
# downstream region) under :aa. Used for both the γ moment labels and the T params.
_tcol_names = ca_level_local == :aa ?
    ["AA" * _ze_names[downstream_regions_local[a]] for a in 1:n_AA_local] :
    _ze_names
# block 5: gamma, sector-major column-minor, active & non-ref kept
for s in 1:S_, c in 1:T_col_dim_local
    slot     = (s - 1) * T_col_dim_local + c   # index into vec(EMP_GAMMA_T)
    full_pos = n_labor + n_industry + n_pi + n_reg + slot
    moment_mask_local[full_pos] &&
        push!(_moment_labels, "gamma[$(_sector_names[s])-$(_tcol_names[c])]")
end
# block 6: G0 per sector (granular only, never masked)
if granular_local
    for s in 1:S_
        full_pos = n_labor + n_industry + n_pi + n_reg + n_gamma + s
        moment_mask_local[full_pos] && push!(_moment_labels, "G0[$(_sector_names[s])]")
    end
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
# T entries follow vec(permutedims(T)) s-major over (S, T_COL_DIM) restricted to
# T_MASK. Recover (s, column) for each active T slot in the order T_reduced is laid out.
for flat_pos in findall(T_mask_local)             # s-major: s outer, column inner
    s = ((flat_pos - 1) ÷ T_col_dim_local) + 1
    c = ((flat_pos - 1) %  T_col_dim_local) + 1
    push!(_param_labels_full, "T[$(_sector_names[s])-$(_tcol_names[c])]")
end
@assert length(_param_labels_full) == 1 + S_ + R_down_ + N_TAU + sum(T_mask_local)
_param_labels = _param_labels_full[jacobian_param_indices]
@everywhere const PARAM_LABELS = $_param_labels
println("Built $(length(_moment_labels)) moment labels and $(length(_param_labels)) parameter labels.")