##### test_extensive_margin.jl #####
# Phase-2 diagnostic (GATE G2) for the exact extensive-margin refactor of the
# analytical reg_coef block (Phase 3). STANDALONE + PRINT-ONLY: it includes the
# model files and load_parameters.jl exactly the way main_gmm.jl does, then
# MEASURES the extensive-margin geometry without touching any production path.
#
#     julia test_extensive_margin.jl aero 4 1
#     julia test_extensive_margin.jl auto 1
#
# It answers the two questions Phase 3's design hinges on:
#
#   (1) How expensive is the EXACT inclusion–exclusion? The exact win-anywhere
#       probability sums over subsets of the Pareto-maximal destination set D*
#       (destinations whose win-event is not contained in another's). Cost per
#       node is 2^|D*|, so we report the |D*| histogram and the max 2^|D*|.
#       Decision rule (plan G2): max |D*| ≲ 12 → exact everywhere; else hybrid
#       (exact below a cutoff, pairwise above).
#
#   (2) How wrong is the current FKG product, and does the exact IE match the
#       truth? For a handful of (s, r_p, z) nodes we compute, with SHARED
#       competitor draws (the true estimand):
#         • MC        — Monte-Carlo win-anywhere (ground truth)
#         • FKG       — 1 − ∏_dr (1 − ρ_dr)   (what compute_regression_quadrature does)
#         • exact-IE  — Σ_{∅≠S⊆D*} (−1)^{|S|+1} exp(−zinv·Φ(S))   (Phase 3 target)
#         • pairwise  — the IE truncated at |S| ≤ 2 (Bonferroni)
#       exact-IE must track MC to MC-noise; the FKG−MC gap is the measured bias.
#
# ── Math (derived from compute_regression_quadrature / EK) ────────────────────
# For an origin good (s, r_p) with own productivity z, r_p wins destination dr
# against competitor r' iff z_{r'} ≤ z·(w_{r'}τ_{r',dr})/(w_{r_p}τ_{r_p,dr}). The
# competitor draws z_{r'} are SHARED across dr, so the destination-win events are
# positively correlated (FKG) and the product over-states the union. Define
#   Q[r',dr] = T_{r'} (w_{r'}τ_{r',dr})^{-θ} (w_{r_p}τ_{r_p,dr})^{θ}   ( ≥ 0 )
# so Φ({dr}) = Σ_{r'} Q[r',dr] = −coef_dr (matches the code's coef), and for a
# subset S the joint win probability is exp(−zinv·Φ(S)) with the running max
#   Φ(S) = Σ_{r'} max_{dr∈S} Q[r',dr].
# Win(dr) ⊆ Win(dr') ⟺ Q[·,dr] ≥ Q[·,dr'] componentwise (dr has weakly MORE
# competition), so the union only needs the Pareto-MINIMAL (least-competition)
# destinations D* — dominated destinations' win-events are already contained.

using Distributed
@everywhere using Random
@everywhere using NPZ
using LinearAlgebra, Statistics, Printf

@everywhere include("model_CP.jl")
@everywhere include("tools.jl")
@everywhere include("model_analytical.jl")
@everywhere include("pso_integration.jl")

industry = length(ARGS) >= 1 ? ARGS[1] : "aero"
n_coef   = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4
n_tau    = length(ARGS) >= 3 && !isempty(strip(ARGS[3])) ? parse(Int, ARGS[3]) : n_coef
K_sim    = 10000   # unused here; load_parameters.jl may reference it

input_folder  = "./baseline_$industry"
output_folder = "./reporting_$industry"
mkpath(output_folder)

include("load_parameters.jl")

println("\n" * "="^72)
println("Phase-2 GATE G2: extensive-margin geometry   (industry=$industry  N_REG=$N_REG  N_TAU=$N_TAU)")
println("="^72)

# ── Parameter vector: prefer a saved estimate, else a documented default ──────
function load_theta()
    # A saved θ̂ is only usable if it was estimated under the SAME (N_REG, N_TAU,
    # gamma_threshold) config as this invocation — unpack_params slices the head by
    # the current N_TAU and scatters the tail into the current T_MASK, so a length
    # mismatch means the file is from a different config and must be rejected.
    expected = 1 + S + R_downstream + N_TAU + count(T_MASK)
    for rel in (("step3", "theta_hat_2.npy"), ("step1", "theta_hat_1.npy"),
                ("step1", "best_params.npy"))
        p = joinpath(output_folder, rel...)
        isfile(p) || continue
        θ = NPZ.npzread(p)
        ndims(θ) > 1 && (θ = θ[:, 1])
        if length(θ) == expected
            @printf("Loaded θ̂ from %s (length %d)\n", p, length(θ))
            return θ
        else
            @printf("Skipping %s: length %d ≠ expected %d for this config (N_REG=%d, N_TAU=%d, active T=%d) — likely a different run\n",
                    p, length(θ), expected, N_REG, N_TAU, count(T_MASK))
        end
    end
    # Default: T from the gravity start, unit A/Ω, a mild β. Geometry (dominance
    # structure, |D*|) is what we measure; the per-node error table depends on β,
    # so this default is clearly reported and β-perturbation stability is checked.
    T_red   = vec(permutedims(T_rs_init))[T_MASK]      # s-major, T_MASK order
    Ω_L     = 0.5
    Ω_s     = ones(S)
    A       = ones(R_downstream)
    β       = N_TAU == 1 ? [0.1] : collect(range(0.05, 0.30, length = N_TAU))
    θ = vcat(Ω_L, Ω_s, A, β, T_red)
    @printf("Using DEFAULT θ (T=T_rs_init, Ω/A=1, β=%s)\n", string(round.(β; digits = 3)))
    @assert length(θ) == expected
    return θ
end

# ── Per-good competition matrix Q[comp, dr] and singleton coef ───────────────
# Returns (comps, Q, coef): comps = competitor origin-region indices (≠ r_p),
# Q is (n_comp × R_downstream), coef[dr] = T_val − Φ({dr}) (== the code's coef).
function competition_matrix(s::Int, r_p::Int, T_mat, tau)
    regs   = SECTOR_GOOD_REGIONS[s]
    comps  = filter(!=(r_p), regs)
    g_p    = SR_TO_GOOD[s, r_p]
    w_p    = W_RS_FLAT[g_p]
    T_p    = T_mat[s, r_p]
    ncomp  = length(comps)
    Q      = zeros(Float64, ncomp, R_downstream)
    for (ci, rc) in enumerate(comps), dr in 1:R_downstream
        g_c = SR_TO_GOOD[s, rc]
        w_c = W_RS_FLAT[g_c]
        Q[ci, dr] = T_mat[s, rc] * (w_c * tau[rc, dr])^(-theta) * (w_p * tau[r_p, dr])^theta
    end
    # code's coef_dr = T_val − Phi[s,dr](w_pτ_p)^θ; Phi includes r_p, whose term is
    # exactly T_val, so coef_dr = −Σ_comp Q[c,dr] = −Φ({dr}) (the r_p self-term cancels).
    coef = Float64[-sum(@view Q[:, dr]) for dr in 1:R_downstream]  # = code's coef_dr
    return comps, Q, coef, w_p, T_p
end

# ── Dedupe + Pareto-minimal (least-competition) destination pruning ──────────
# Keep destination dr unless another dr' has Q[:,dr'] ≤ Q[:,dr] everywhere and is
# strictly smaller somewhere (dr' weakly dominates the competition dr faces ⇒
# Win(dr) ⊆ Win(dr') ⇒ dr redundant for the union). Exact duplicates collapse to
# the lowest column index. Returns the retained column indices (into 1:R_downstream).
function pareto_min_columns(Q::Matrix{Float64}; atol = 1e-10)
    n = size(Q, 2)
    n == 0 && return Int[]
    # dedupe by rounded column key
    keyfun(j) = size(Q, 1) == 0 ? (0,) : Tuple(round.(@view(Q[:, j]); sigdigits = 10))
    seen = Dict{Any,Int}()
    uniq = Int[]
    for j in 1:n
        k = keyfun(j)
        if !haskey(seen, k)
            seen[k] = j
            push!(uniq, j)
        end
    end
    # Pareto-minimal frontier over the unique columns
    keep = Int[]
    for j in uniq
        dominated = false
        for k in uniq
            k == j && continue
            le = all(Q[:, k] .<= Q[:, j] .+ atol)
            lt = any(Q[:, k] .<  Q[:, j] .- atol)
            if le && lt
                dominated = true
                break
            end
        end
        dominated || push!(keep, j)
    end
    return keep
end

# ── Exact / pairwise inclusion–exclusion over a destination-column set ───────
# Qstar: (n_comp × n_star). DFS over nonempty subsets maintaining the running max
# vector; Φ(S) = Σ_comp max_{dr∈S} Qstar[comp,dr]; term (−1)^{|S|+1} exp(−zinv·Φ).
# maxlevel caps |S| (2 → Bonferroni pairwise; typemax → exact).
function ie_win_prob(Qstar::AbstractMatrix{Float64}, zinv::Float64; maxlevel::Int = typemax(Int))
    ncomp, nstar = size(Qstar)
    nstar == 0 && return 0.0
    ncomp == 0 && return 1.0                      # no competitors ⇒ always wins
    total = 0.0
    curmax = zeros(Float64, ncomp)                # Qstar ≥ 0 ⇒ 0 is the max-identity
    function dfs(start::Int, sz::Int, cmax::Vector{Float64})
        for j in start:nstar
            nm   = max.(cmax, @view Qstar[:, j])
            ΦS   = sum(nm)
            sign = isodd(sz + 1) ? 1.0 : -1.0     # (−1)^{|S|+1}, |S| = sz+1
            total += sign * exp(-zinv * ΦS)
            (sz + 1) < maxlevel && dfs(j + 1, sz + 1, nm)
        end
    end
    dfs(1, 0, curmax)
    return total
end

# ── Exact union-of-boxes via dominance-pruned recursion + memoization (DAG) ───
# The exact win-anywhere probability is the product-Fréchet measure of a UNION of
# origin-anchored boxes B_dr = {z_c ≤ z·thr[c,dr] ∀c}. In q-space a box is the
# componentwise max of a subset of the base D* columns (encoded as a bitmask over
# columns); μ(box) = exp(−zinv·Σ_c qvec_c). The measure of a union obeys
#   μ(∪rest ∪ B) = μ(∪rest) + μ(B) − μ(∪_i(B∩B_i)),   B∩B_i has mask = maskB | maskᵢ.
# Two accelerations turn the naïve 2^|D*| IE into something that can collapse:
#   • dominance pruning — a box whose q-vector is componentwise ≥ another's is
#     CONTAINED (a subset) ⇒ redundant in a union ⇒ dropped at every level;
#   • memoization — subproblems are canonical pruned mask-sets shared across the DAG.
# Crucially the recursion STRUCTURE (which subproblems arise) is independent of z
# (pruning lives entirely in z-free q-space), so we build the DAG ONCE per good and
# evaluate all quadrature nodes by a cheap topological pass — exactly the reuse
# Phase 3 would exploit. `nsub` (# distinct subproblems) is the true exact cost and
# the number G2 needs: if it stays ≪ 2^|D*| for the thick goods, exact is feasible.
function build_union_dag(Qstar::Matrix{Float64}; cap::Int = 200_000, tol = 1e-12)
    ncomp, k = size(Qstar)
    @assert k <= 62 "bitmask width exceeded"
    qvec_of = Dict{UInt,Vector{Float64}}()
    qsum_of = Dict{UInt,Float64}()
    function qvec(mask::UInt)
        get!(qvec_of, mask) do
            v = fill(-Inf, ncomp); j = 1; m = mask
            while m != 0
                if (m & UInt(1)) != 0
                    @inbounds for c in 1:ncomp
                        q = Qstar[c, j]; q > v[c] && (v[c] = q)
                    end
                end
                m >>= 1; j += 1
            end
            ncomp == 0 ? Float64[] : v
        end
    end
    qsum(mask::UInt) = get!(qsum_of, mask) do
        v = qvec(mask); s = 0.0; @inbounds for c in 1:ncomp; s += v[c]; end; s
    end
    # prune a mask list to the Pareto-min (largest) boxes: drop mask a if its box is
    # contained in another's (qvec(a) ≥ qvec(b) componentwise); equal ⇒ drop higher a.
    function prune(masks::Vector{UInt})
        vs = [qvec(m) for m in masks]
        n = length(masks); keep = trues(n)
        for a in 1:n, b in 1:n
            (a == b || !keep[a] || !keep[b]) && continue
            ge = true; gt = false
            @inbounds for c in 1:ncomp
                if vs[a][c] < vs[b][c] - tol; ge = false; break; end
                vs[a][c] > vs[b][c] + tol && (gt = true)
            end
            if ge && (gt || a > b)      # a ⊆ b (strict), or equal duplicate → drop a
                keep[a] = false
            end
        end
        return masks[keep]
    end
    id_of  = Dict{Vector{UInt},Int}()   # canonical pruned mask-set → node id (≥1)
    nodes  = Vector{Tuple{Symbol,Int,Float64,Int}}()  # (:single,_,qsum,_) | (:internal,rid,qsum_last,iid)
    capped = Ref(false)
    function build(masks::Vector{UInt})::Int
        capped[] && return -1
        pm = sort(prune(masks))
        isempty(pm) && return 0                    # id 0 = empty set → measure 0
        haskey(id_of, pm) && return id_of[pm]
        if length(pm) == 1
            push!(nodes, (:single, 0, qsum(pm[1]), 0)); id = length(nodes)
            id_of[pm] = id; return id
        end
        if length(id_of) >= cap; capped[] = true; return -1; end
        last = pm[end]; rest = pm[1:end-1]
        inter = UInt[r | last for r in rest]       # intersections = OR of masks
        rid = build(rest); iid = build(inter)
        (rid < 0 || iid < 0) && return -1
        push!(nodes, (:internal, rid, qsum(last), iid)); id = length(nodes)
        id_of[pm] = id; return id
    end
    base = UInt[UInt(1) << (j - 1) for j in 1:k]
    root = build(base)
    return (root = root, nodes = nodes, capped = capped[], nsub = length(nodes))
end

# Evaluate a prebuilt union DAG at a given zinv (topological pass; deps have lower id).
function eval_union_dag(dag, zinv::Float64)
    dag.capped && return NaN
    dag.root == 0 && return 0.0
    val = zeros(Float64, length(dag.nodes))
    @inbounds for i in 1:length(dag.nodes)
        typ, rid, qs, iid = dag.nodes[i]
        if typ === :single
            val[i] = exp(-zinv * qs)
        else
            vr = rid == 0 ? 0.0 : val[rid]
            vi = iid == 0 ? 0.0 : val[iid]
            val[i] = vr + exp(-zinv * qs) - vi
        end
    end
    return val[dag.root]
end

# ── Monte-Carlo ground truth: shared competitor draws, win at ANY destination ─
function mc_win_anywhere(s::Int, r_p::Int, z::Float64, comps::Vector{Int},
                         T_mat, tau, w_p::Float64; N::Int = 200_000, rng = MersenneTwister(12345))
    ncomp = length(comps)
    ncomp == 0 && return 1.0
    # threshold thr[c,dr] = (w_c τ_{c,dr})/(w_p τ_{r_p,dr}); win dr ⟺ z_c ≤ z·thr ∀c
    thr = zeros(Float64, ncomp, R_downstream)
    Tc  = zeros(Float64, ncomp)
    for (ci, rc) in enumerate(comps)
        w_c    = W_RS_FLAT[SR_TO_GOOD[s, rc]]
        Tc[ci] = T_mat[s, rc]
        for dr in 1:R_downstream
            thr[ci, dr] = (w_c * tau[rc, dr]) / (w_p * tau[r_p, dr])
        end
    end
    wins = 0
    zc   = zeros(Float64, ncomp)
    for _ in 1:N
        # competitor productivities: Fréchet(T_c, θ), CDF exp(−T x^{−θ})
        for ci in 1:ncomp
            u = rand(rng)
            zc[ci] = (-log(u) / Tc[ci])^(-1.0 / theta)
        end
        any_win = false
        for dr in 1:R_downstream
            win_dr = true
            @inbounds for ci in 1:ncomp
                if zc[ci] > z * thr[ci, dr]
                    win_dr = false
                    break
                end
            end
            if win_dr
                any_win = true
                break
            end
        end
        any_win && (wins += 1)
    end
    return wins / N
end

θ     = load_theta()
Ω_L, Ω_s, A, β, T_vec = unpack_params(θ)
T_mat = reshape(T_vec, S, R)
tau   = build_tau(β)
Phi   = compute_Phi(T_mat, tau)

# ── (0) Self-check: reconstructed coef == compute_regression_quadrature's coef ─
let maxerr = 0.0
    for g in 1:n_good
        s = GOOD_S[g]; r_p = GOOD_R[g]
        _, _, coef, w_p, T_p = competition_matrix(s, r_p, T_mat, tau)
        for dr in 1:R_downstream
            code_coef = T_p - Phi[s, dr] * (w_p * tau[r_p, dr])^theta
            maxerr = max(maxerr, abs(coef[dr] - code_coef))
        end
    end
    @printf("Self-check: max|reconstructed coef − code coef| = %.3e (must be ~0)\n", maxerr)
    @assert maxerr < 1e-8 "coef reconstruction disagrees with compute_regression_quadrature — Q/coef derivation is wrong"
end

# ── (1) |D*| histogram across all active goods ───────────────────────────────
println("\n── |D*| (Pareto-maximal destination set size) across all $(n_good) goods ──")
Dstar_sizes = Int[]
for g in 1:n_good
    s   = GOOD_S[g]; r_p = GOOD_R[g]
    _, Q, _, _, _ = competition_matrix(s, r_p, T_mat, tau)
    keep = pareto_min_columns(Q)
    push!(Dstar_sizes, length(keep))
end
let
    hi = maximum(Dstar_sizes)
    @printf("  R_downstream = %d   →   full-enumeration cost 2^R = %.3e\n", R_downstream, 2.0^R_downstream)
    @printf("  |D*|: min=%d  median=%.1f  mean=%.2f  max=%d\n",
            minimum(Dstar_sizes), median(Dstar_sizes), mean(Dstar_sizes), hi)
    println("  histogram (|D*| : count):")
    for k in minimum(Dstar_sizes):hi
        c = count(==(k), Dstar_sizes)
        c == 0 && continue
        @printf("    %2d : %s (%d)\n", k, "#"^min(c, 60), c)
    end
    @printf("  worst-case exact IE cost 2^max|D*| = 2^%d = %.3e\n", hi, 2.0^hi)
    if hi <= 12
        @printf("  → DECISION: max|D*|=%d ≤ 12  ⇒  EXACT inclusion–exclusion everywhere\n", hi)
    else
        @printf("  → DECISION: max|D*|=%d > 12  ⇒  HYBRID (exact below cutoff, pairwise above)\n", hi)
    end
end

# ── (2) Ground-truth error table on a handful of (s, r_p, z) nodes ───────────
println("\n── MC vs FKG vs exact-IE vs pairwise (shared competitor draws) ──")
# pick up to ~6 goods spread across the good index range
sample_goods = unique(round.(Int, range(1, n_good, length = min(6, n_good))))
# z = scale·(−log(1−u))^(−1/θ) is DECREASING in u, so SMALL u = HIGH productivity =
# the regime where the extensive margin is actually active and the FKG bias bites.
u_nodes = [0.02, 0.1, 0.3, 0.6]               # low-u = high-z tail of r_p's OWN productivity
@printf("%-22s %6s | %10s %10s %10s %10s %10s | %10s %10s\n",
        "good (s-r_p)  |D*|", "u", "MC", "FKG", "exact", "union", "pairwise", "FKG/MC", "exact−MC")
maxbias_fkg = 0.0
maxratio_fkg = 1.0
maxdiff_union = 0.0
for g in sample_goods
    s = GOOD_S[g]; r_p = GOOD_R[g]
    comps, Q, coef, w_p, T_p = competition_matrix(s, r_p, T_mat, tau)
    keep  = pareto_min_columns(Q)
    Qstar = Q[:, keep]
    dag   = build_union_dag(Qstar)                  # exact, dominance-pruned DAG
    scale = T_p^(1 / theta)
    tag   = @sprintf("%s-%s |D*|=%d", _sector_names[s], _ze_names[r_p], length(keep))
    for u in u_nodes
        u_k  = min(u, 1.0 - 1e-14)
        z    = scale * (-log(1.0 - u_k))^(-1.0 / theta)
        zinv = z^(-theta)
        ρ    = exp.(zinv .* coef)                       # per-destination win prob, all dr
        fkg  = -expm1(sum(log1p.(max.(-ρ, -1.0))))      # 1 − ∏(1−ρ_dr)  (code's estimand)
        exIE = ie_win_prob(Qstar, zinv)                 # exact over D* (brute 2^|D*| IE)
        uni  = eval_union_dag(dag, zinv)                # exact over D* (pruned DAG)
        pair = ie_win_prob(Qstar, zinv; maxlevel = 2)   # Bonferroni pairwise over D*
        mc   = mc_win_anywhere(s, r_p, z, comps, T_mat, tau, w_p; N = 200_000,
                               rng = MersenneTwister(1000 + g))
        global maxbias_fkg = max(maxbias_fkg, abs(fkg - mc))
        if isfinite(uni)
            global maxdiff_union = max(maxdiff_union, abs(uni - exIE))
        end
        if mc > 1e-6
            global maxratio_fkg = max(maxratio_fkg, fkg / mc)
        end
        @printf("%-22s %6.3f | %10.5f %10.5f %10.5f %10.5f %10.5f | %9.2fx %+10.2e\n",
                tag, u, mc, fkg, exIE, uni, pair, mc > 1e-9 ? fkg / mc : 1.0, exIE - mc)
    end
end
@printf("\nmax |FKG − MC| = %.3e   max FKG/MC ratio = %.2fx  (the FKG extensive-margin bias — level and relative)\n",
        maxbias_fkg, maxratio_fkg)
@printf("union-DAG vs brute IE: max|diff| = %.3e (must be ~machine 0 — validates the exact algorithm)\n", maxdiff_union)
@printf("(exact−MC should be within MC noise ≈ 1/√N ≈ %.1e; if not, revisit the IE derivation)\n", 1 / sqrt(200_000))
@assert maxdiff_union < 1e-9 "union DAG disagrees with brute-force inclusion–exclusion — exact algorithm is wrong"

# ── (2b) β-perturbation stability of the |D*| geometry (binned τ only) ────────
if N_TAU > 1
    println("\n── |D*| stability under β perturbation (binned τ) ──")
    for scale_β in (0.8, 1.25)
        τp   = build_tau(β .* scale_β)
        sz   = Int[]
        for g in 1:n_good
            s = GOOD_S[g]; r_p = GOOD_R[g]
            _, Q, _, _, _ = competition_matrix(s, r_p, T_mat, τp)
            push!(sz, length(pareto_min_columns(Q)))
        end
        @printf("  β×%.2f : max|D*|=%d  mean=%.2f  (baseline max=%d)\n",
                scale_β, maximum(sz), mean(sz), maximum(Dstar_sizes))
    end
else
    println("\n(power-law τ, N_TAU=1: destination dominance is monotone in distance — |D*| geometry is β-stable)")
end

# ── (3) Exact-IE feasibility: distinct subproblems under the pruned DAG ──────
# The naïve exact IE is 2^|D*| per node; the pruned DAG's cost is `nsub` (distinct
# subproblems), which is z-INDEPENDENT (pruning lives in q-space) so it is built once
# per good and evaluated per quadrature node. If nsub stays small for the |D*|=22
# goods, dominance pruning has tamed the 2^22 cost and exact-everywhere is feasible.
println("\n── (3) Exact-IE feasibility: pruned-DAG subproblem count (nsub) vs naïve 2^|D*| ──")
CAP_SUB  = 100_000   # per-good subproblem cap; goods beyond it fall back to FKG in §4
dags     = Vector{Any}(undef, n_good)
nsub_g   = zeros(Int, n_good)
feasible = trues(n_good)
for g in 1:n_good
    s = GOOD_S[g]; r_p = GOOD_R[g]
    _, Q, _, _, _ = competition_matrix(s, r_p, T_mat, tau)
    keep       = pareto_min_columns(Q)
    dag        = build_union_dag(Q[:, keep]; cap = CAP_SUB)
    dags[g]    = dag
    nsub_g[g]  = dag.nsub
    feasible[g] = !dag.capped
end
let
    @printf("  CAP=%d.  infeasible goods (hit cap): %d / %d\n", CAP_SUB, count(.!feasible), n_good)
    @printf("  nsub: min=%d  median=%.0f  mean=%.0f  max=%d   (naïve 2^max|D*| = %.2e)\n",
            minimum(nsub_g), median(nsub_g), mean(nsub_g), maximum(nsub_g), 2.0^maximum(Dstar_sizes))
    thick = findall(Dstar_sizes .>= 12)
    if !isempty(thick)
        redf = [2.0^Dstar_sizes[g] / max(nsub_g[g], 1) for g in thick]
        @printf("  thick goods (|D*|≥12, n=%d): nsub max=%d  median reduction 2^|D*|/nsub = %.2gx\n",
                length(thick), maximum(nsub_g[thick]), median(redf))
    end
    if all(feasible) && maximum(nsub_g) <= 100_000
        println("  → exact IE FEASIBLE everywhere via the pruned DAG (build once/good, eval per node)")
    else
        println("  → some goods exceed the cap ⇒ exact needs a cutoff-hybrid (exact where nsub≤cap, approx above)")
    end
end

# ── (4) reg_coef coefficient bias: FKG vs EXACT extensive margin ─────────────
# Replicates compute_regression_quadrature (same nodes / regressors / FE / weights)
# and swaps ONLY the regressand: FKG product vs the exact pruned-DAG union. The two
# coefficient vectors' gap is the number that actually decides Phase 3 — the FKG bias
# on the estimated distance coefficients, integrated over the whole productivity range.
function reg_coef_quad(; exact::Bool, n_quad::Int, dags = nothing, feasible = nothing)
    nodes_raw, weights_raw = gausslegendre(n_quad)
    un = (nodes_raw .+ 1) ./ 2; gw = weights_raw ./ 2
    N_total = n_good * n_quad; n_reg = N_REG + 1
    y = zeros(N_total); X = zeros(N_total, n_reg); w_arr = zeros(N_total); fe = zeros(Int, N_total)
    idx = 0
    for g in 1:n_good
        s = GOOD_S[g]; r_p = GOOD_R[g]
        _, Q, coef, w_p, T_p = competition_matrix(s, r_p, T_mat, tau)
        dr0  = CLOSEST_DOWNSTREAM_REGION[r_p]; gid = (s - 1) * R_downstream + dr0
        b    = N_REG == 1 ? 0 : DistBin[r_p, dr0]
        logd = N_REG == 1 ? LOG_CLOSEST_DIST[r_p] : 0.0
        scale = T_p^(1 / theta)
        use_exact = exact && feasible[g]
        for k in 1:n_quad
            idx += 1
            u_k = min(un[k], 1.0 - 1e-14)
            z_k = scale * (-log(1.0 - u_k))^(-1.0 / theta); zinv = z_k^(-theta)
            if use_exact
                y[idx] = eval_union_dag(dags[g], zinv)          # exact
            else
                lp = 0.0                                          # FKG (and exact-infeasible fallback)
                @inbounds for dr in 1:R_downstream
                    lp += log1p(max(-exp(zinv * coef[dr]), -1.0))
                end
                y[idx] = -expm1(lp)
            end
            w_arr[idx] = gw[k]; fe[idx] = gid
            if N_REG == 1
                X[idx, 1] = logd
            else
                (b > 0 && b <= N_REG) && (X[idx, b] = 1.0)
            end
            X[idx, n_reg] = log(z_k)
        end
    end
    for gid in unique(fe)
        m = fe .== gid; wg = w_arr[m]; tw = sum(wg); tw < 1e-15 && continue
        y[m] .-= sum(wg .* y[m]) / tw
        for j in 1:n_reg
            X[m, j] .-= sum(wg .* X[m, j]) / tw
        end
    end
    sw = sqrt.(w_arr)
    coefs = (sw .* X) \ (sw .* y)
    return coefs[1:N_REG]
end

println("\n── (4) reg_coef coefficient: FKG vs EXACT extensive margin (the decision number) ──")
NQ      = 100
α_fkg   = reg_coef_quad(; exact = false, n_quad = NQ)
α_exact = reg_coef_quad(; exact = true,  n_quad = NQ, dags = dags, feasible = feasible)
@printf("  n_quad=%d, exact where feasible (%d/%d goods; %d fell back to FKG)\n",
        NQ, count(feasible), n_good, count(.!feasible))
@printf("  %-4s %15s %15s %15s %9s\n", "bin", "α_FKG", "α_exact", "Δ=exact−FKG", "relΔ")
rels = zeros(N_REG)
for j in 1:N_REG
    d = α_exact[j] - α_fkg[j]
    rels[j] = abs(α_fkg[j]) > 1e-12 ? abs(d / α_fkg[j]) : 0.0
    @printf("  %-4d %15.6e %15.6e %15.6e %8.2f%%\n", j, α_fkg[j], α_exact[j], d, 100 * (α_fkg[j] != 0 ? d / α_fkg[j] : 0.0))
end
@printf("  → max |relΔ| on the reg_coef vector = %.2f%%  (FKG bias on the estimated distance coefficients)\n",
        100 * maximum(rels))

println("\n" * "="^72)
println("Phase-2 GATE G2 complete — |D*| cost, FKG−MC bias, DAG feasibility, and reg_coef bias.")
println("="^72)
