##### test_t_reorder.jl #####
# Phase-2 GATE for the s-major T-parameter reorder.
#
# Loads constants exactly the way main.jl does (include the model files, then
# load_parameters.jl with a baseline_<industry> data folder present), then
# asserts the four invariants T1–T4. Run with the SAME data you estimate on:
#
#     julia test_t_reorder.jl aero 4 1
#     julia test_t_reorder.jl auto 1
#
# A SILENT transposition of T_mat runs fine but fits the wrong T; these tests
# are the only guard. If any @assert throws, the corresponding edit is wrong.

using Distributed
# Keep it light: load_parameters.jl uses @everywhere, which is a no-op-safe even
# with a single process. Add a couple of workers only if you want parity.
@everywhere using Random
@everywhere using NPZ
using LinearAlgebra, Statistics, Printf

@everywhere include("../model_CP.jl")
@everywhere include("../tools.jl")
@everywhere include("../model_analytical.jl")
@everywhere include("../optimizers/pso_integration.jl")

industry = length(ARGS) >= 1 ? ARGS[1] : "aero"
n_coef   = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4
n_tau    = length(ARGS) >= 3 && !isempty(strip(ARGS[3])) ? parse(Int, ARGS[3]) : n_coef
K_sim    = 10000   # unused here; load_parameters.jl may reference it

input_folder  = "./baseline_$industry"
output_folder = "./reporting_$industry"
mkpath(output_folder)

include("../load_parameters.jl")

println("\n" * "="^70)
println("Phase-2 GATE: s-major T reorder   (industry=$industry  N_REG=$N_REG  N_TAU=$N_TAU)")
println("="^70)

# ── T1: mask identity ───────────────────────────────────────────────────────
@assert T_MASK == T_MASK_MOMENT "T1 FAILED: T_MASK != T_MASK_MOMENT"
println("T1 PASS: T_MASK == T_MASK_MOMENT  (the two conventions are now one)")

# helper: s-major flat position of (s,r) in vec(permutedims(X_rs)) over (R,S)
smajor_flat(s, r) = (s - 1) * R_full + r
# reduced (T_MASK) index of an active (s,r)
function reduced_idx(s, r)
    fp = smajor_flat(s, r)
    @assert T_MASK[fp] "(s=$s,r=$r) is not active in T_MASK"
    return count(@view T_MASK[1:fp])
end

# ── T2: round-trip through unpack_params ────────────────────────────────────
# Build random reduced T, set each sector's REFERENCE entry to 1.0 so the
# in-unpack normalization (T_mat[s,:] ./= T_mat[s,ref]) is the identity and we
# can test EXACT recovery of every active cell. Catches EDIT 2 and EDIT 3.
nT = sum(T_MASK)
T_reduced = rand(nT) .+ 0.5                       # strictly positive
for s in 1:S_
    ref_r = T_REF_REGION[s]
    ref_r == 0 && continue
    T_reduced[reduced_idx(s, ref_r)] = 1.0        # ref cell → 1
end

# splice into a full params vector at the T-block (prefix values are irrelevant
# to T recovery; use ones so Ω^s/A normalizations stay finite)
t_start = get_param_start_index(:T)               # 1-based start of T-block
params  = vcat(ones(t_start - 1), T_reduced)
@assert length(params) == t_start - 1 + nT

_, _, _, _, T_vec = unpack_params(params)
T_mat = reshape(T_vec, S, R)                       # returned vec is region-major (S,R)

for g in 1:n_good
    s, r = GOOD_S[g], GOOD_R[g]
    expected = T_reduced[reduced_idx(s, r)]        # value spliced for this (s,r)
    @assert isapprox(T_mat[s, r], expected; atol=1e-12) (
        "T2 FAILED at (s=$s,r=$r): T_mat=$(T_mat[s,r]) expected=$expected")
end
# and every reference cell must be exactly 1 after normalization
for s in 1:S_
    ref_r = T_REF_REGION[s]
    ref_r == 0 && continue
    @assert isapprox(T_mat[s, ref_r], 1.0; atol=1e-12) "T2 FAILED: ref cell sector $s != 1"
end
println("T2 PASS: unpack_params recovers T_reduced at every active (GOOD_S,GOOD_R); refs=1")

# ── T3: PARAM_LABELS ↔ position alignment ──────────────────────────────────
# For each "T[sname-rname]" label, recover (s,r) from the NAME and assert it
# equals the (s,r) implied by its position via the new s-major flatten.
# Build name→index maps from the label vocabularies used in load_parameters.jl.
sname_to_s = Dict(_sector_names[s] => s for s in 1:S_)
zename_to_r = Dict(_ze_names[r] => r for r in 1:R_full)

# enumerate identified T params in PARAM_LABELS order = order of findall(T_MASK)
# minus the excluded reference columns (jacobian_param_indices).
t_label_pairs = Tuple{Int,Int}[]          # (s,r) parsed from each "T[..]" label
for lab in PARAM_LABELS
    startswith(lab, "T[") || continue
    body = lab[3:end-1]                    # strip "T[" ... "]"
    dash = findlast('-', body)
    sname = body[1:dash-1]; rname = body[dash+1:end]
    push!(t_label_pairs, (sname_to_s[sname], zename_to_r[rname]))
end

# independently compute the identified (non-ref) T (s,r) sequence from positions
ref_redidx = Set{Int}()
for s in 1:S_
    ref_r = T_REF_REGION[s]
    ref_r == 0 && continue
    T_MASK[smajor_flat(s, ref_r)] && push!(ref_redidx, reduced_idx(s, ref_r))
end
pos_pairs = Tuple{Int,Int}[]
for (red, fp) in enumerate(findall(T_MASK))
    red in ref_redidx && continue
    s = ((fp - 1) ÷ R_full) + 1
    r = ((fp - 1) %  R_full) + 1
    push!(pos_pairs, (s, r))
end
@assert t_label_pairs == pos_pairs "T3 FAILED: PARAM_LABELS order != position order"
println("T3 PASS: every T[s-r] label matches its position-implied (s,r)")

# ── T4: axis parallelism (the actual goal) ─────────────────────────────────
# γ rows (from MOMENT_LABELS) and T columns (from PARAM_LABELS) must enumerate
# the SAME (s,r) pairs in the SAME region order. Show one sector side-by-side.
gamma_pairs = Tuple{Int,Int}[]
for lab in MOMENT_LABELS
    startswith(lab, "gamma[") || continue
    body = lab[7:end-1]
    dash = findlast('-', body)
    sname = body[1:dash-1]; rname = body[dash+1:end]
    push!(gamma_pairs, (sname_to_s[sname], zename_to_r[rname]))
end

@assert gamma_pairs == t_label_pairs (
    "T4 FAILED: γ-row order != T-col order\n  γ: $gamma_pairs\n  T: $t_label_pairs")

# pick a sector with ≥2 active non-ref regions and print side-by-side
sel = 0
for s in 1:S_
    if count(p -> p[1] == s, gamma_pairs) >= 2
        sel = s; break
    end
end
println("T4 PASS: γ-row order == T-col order across ALL identified (s,r).")
if sel > 0
    println("\nSide-by-side for sector s=$sel ($(_sector_names[sel])):")
    @printf("%4s  %18s  %18s\n", "k", "γ row (s-r)", "T col (s-r)")
    grows = [p for p in gamma_pairs if p[1] == sel]
    tcols = [p for p in t_label_pairs if p[1] == sel]
    for (k, (gp, tp)) in enumerate(zip(grows, tcols))
        @printf("%4d  %18s  %18s\n", k,
                "$(_sector_names[gp[1]])-$(_ze_names[gp[2]])",
                "$(_sector_names[tp[1]])-$(_ze_names[tp[2]])")
    end
end

println("\n" * "="^70)
println("ALL TESTS PASS — T parameters are now s-major, aligned with γ moments.")
println("="^70)
