##### test_granular_aa.jl #####
# Validation gates for the granular / attraction-area estimator
# (documentation/plan_granular_aa.md + documentation/granular_validation.md).
# STANDALONE + PRINT-ONLY: includes the model files the way test_ge_inversion.jl
# does, then runs the structural gates that can be checked WITHOUT a fitted θ̂.
# Touches no production path.
#
#     julia test/test_granular_aa.jl aero 4 1 true aa      # granular + AA
#     julia test/test_granular_aa.jl aero 4 1 false ze      # legacy reference
#
# Args: industry n_coef n_tau granular ca_level [n_rep]
#
# Gates run here (the numbering follows granular_validation.md Part II):
#
#   V1   AA map — attraction_area_linkages.npy has shape (R, R_downstream), rows
#        sum to 1, and argmax_col == CLOSEST_DOWNSTREAM_REGION. The last one is the
#        decisive check: the model's fixed effect and the empirical one must be the
#        SAME partition, or the whole alignment argument fails. (Also asserted at
#        load time, in load_parameters.jl — this re-reports it explicitly.)
#   V1a  Σ layout — every Σ file carries N_REG + n_γ + S rows (β → γ → G); legacy
#        mode slices to the leading N_REG + n_γ block, granular keeps all three.
#   V1b  Filter containment — every CELL_MASK cell lies in an attraction area active
#        in its own sector (𝒜⁺).
#   V2   N_s root-find — G(s,·) monotone decreasing on [N_LO, N_HI]; the bisection
#        recovers a planted integer exactly; clamps fire at both bounds.
#   V3   AA-level Sinkhorn — round-trip recovery of a planted T from its own area
#        aggregates, mirroring test/test_ge_inversion.jl.
#   V5   Prefix stability — the winners of the first n varieties are identical
#        whether the Ricardian solve runs at width n or at pool width. This is what
#        licenses ONE solve per replication (plan D1).
#   V6   Firm ↔ champion — the log-z coefficient against −θ (Prop. 1(c)).
#   V7   Two routes to N_s — N̂_s from Ḡ_s(0) against N^count_s = N_supplier_s / Σ_l q̂.
#        A large gap is a mechanism finding, not a bug.
#   V9   Bounds not binding — clamped == :none for every sector.
#   V10  N_s-invariance of block 4 — the firm-level index contains no N_s term, so
#        reg_coef should barely move between N̂ and 2N̂.
#
# Gates needing a fitted θ̂ or an external reference (V0, V4, V8, V11, V12, V13) are
# NOT run here; see granular_validation.md.

using Distributed
@everywhere using Random
@everywhere using NPZ
using LinearAlgebra, Statistics, Printf

@everywhere include("../model_CP.jl")
@everywhere include("../tools.jl")
@everywhere include("../model_analytical.jl")

industry = length(ARGS) >= 1 ? ARGS[1] : "aero"
n_coef   = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4
n_tau    = length(ARGS) >= 3 && !isempty(strip(ARGS[3])) ? parse(Int, ARGS[3]) : n_coef
granular = length(ARGS) >= 4 ? (lowercase(strip(ARGS[4])) in ("true","1","yes")) : true
ca_level = length(ARGS) >= 5 && !isempty(strip(ARGS[5])) ? Symbol(strip(ARGS[5])) : :aa
n_rep    = length(ARGS) >= 6 && !isempty(strip(ARGS[6])) ? parse(Int, ARGS[6]) : 20
K_sim    = 10000   # unused here; load_parameters.jl may reference it
reg_method = :cloglog

input_folder  = "./baseline_$industry"
output_folder = "./reporting_$industry" * (ca_level == :aa ? "_aa" : "") *
                (granular ? "_gran" : "")
mkpath(output_folder)

include("../load_parameters.jl")
include("../profiling.jl")

println("\n" * "="^72)
println("GRANULAR / ATTRACTION-AREA VALIDATION GATES")
@printf("  industry=%s  N_REG=%d  N_TAU=%d  GRANULAR=%s  CA_LEVEL=:%s  R_REP=%d\n",
        industry, N_REG, N_TAU, GRANULAR, CA_LEVEL, R_REP)
@printf("  cells=%d  T columns=%d  moments=%d  blocks=%d\n",
        n_good, count(T_MASK), length(empirical_moments), N_BLOCKS)
println("="^72)

pass(name, ok, extra="") = @printf("  %-6s %-52s %s%s\n", name[1:min(end,6)],
                                   "", ok ? "PASS" : "FAIL", extra)

# ── θ: prefer a saved estimate, else the theory-based init ───────────────────
function load_theta()
    expected = 1 + S + R_downstream + N_TAU + count(T_MASK)
    for rel in (("step3", "theta_hat_2.npy"), ("step1", "theta_hat_1.npy"),
                ("step1", "best_params.npy"))
        p = joinpath(output_folder, rel...)
        isfile(p) || continue
        θ = NPZ.npzread(p); ndims(θ) > 1 && (θ = θ[:, 1])
        if length(θ) == expected
            @printf("\nLoaded θ from %s (length %d)\n", p, length(θ)); return θ
        end
    end
    A_init = copy(emp_pi_r_full).^(1/abs(epsilon)) .* regional_wages[N_downstream_per_region .!= 0]
    A_init ./= sum(A_init)
    θ = vcat([agg_labor_share], agg_industry_share, A_init,
             TAU_PRIOR !== nothing ? TAU_PRIOR : fill(P_alpha, N_TAU),
             vec(permutedims(T_rs_init))[T_MASK])
    @printf("\nUsing the theory init (T=T_rs_init, α=prior); length %d\n", length(θ))
    @assert length(θ) == expected
    return θ
end

θ0 = load_theta()
ΩL = θ0[1]
Ωs = θ0[2:(1 + S)];                       Ωs = Ωs ./ sum(Ωs)
A  = θ0[(S + 2):(S + R_downstream + 1)];  A  = A ./ A[1]
α  = θ0[(S + R_downstream + 2):(1 + S + R_downstream + N_TAU)]

results = Tuple{String,Bool,String}[]
record!(name, ok, note="") = push!(results, (name, ok, note))

# ═══════════════════════════════════════════════════════════════════════════
# V1 / V1a / V1b — the input-file gates
# ═══════════════════════════════════════════════════════════════════════════
println("\n── V1  attraction-area map ──────────────────────────────────────────")
if CA_LEVEL === :aa
    Amap = NPZ.npzread(joinpath(input_folder, "attraction_area_linkages.npy"))
    ok_shape = size(Amap) == (R, R_downstream)
    ok_rows  = all(sum(Amap, dims=2) .== 1)
    ok_part  = vec(getindex.(argmax(Amap, dims=2), 2)) == CLOSEST_DOWNSTREAM_REGION
    @printf("  shape (R, R_downstream) = %s : %s\n", string(size(Amap)), ok_shape ? "PASS" : "FAIL")
    @printf("  every ZE in exactly one attraction area : %s\n", ok_rows ? "PASS" : "FAIL")
    @printf("  argmax_col == CLOSEST_DOWNSTREAM_REGION : %s\n", ok_part ? "PASS" : "FAIL")
    ok_part || println("    ⚠ the model's fixed effect and the empirical A129_AA grouping are " *
                       "DIFFERENT partitions — the alignment argument of finite_sample2.tex §1.2 fails.")
    record!("V1", ok_shape && ok_rows && ok_part)
    @printf("  active (sector, AA) pairs 𝒜⁺ : %d of %d possible\n", count(AA_ACTIVE), S * n_AA)
else
    println("  (CA_LEVEL = :ze — the AA map is not used)")
end

println("\n── V1a Σ layout ─────────────────────────────────────────────────────")
let nγ = length(BLOCK_RANGES[5]), fn = sigma_beta_gamma_filename(; smm=true)
    path = joinpath(input_folder, fn)
    @printf("  selected file : %s\n", fn)
    @printf("  expected size : N_REG(%d) + n_γ(%d) + S(%d) = %d\n", N_REG, nγ, S, N_REG + nγ + S)
    if isfile(path)
        sz = size(NPZ.npzread(path), 1)
        ok = sz == N_REG + nγ + S
        @printf("  on-disk size  : %d  → %s\n", sz, ok ? "PASS" : "FAIL")
        ok || println("    ⚠ a wrong size means the file was regenerated with a different block set.")
        record!("V1a", ok)
    else
        println("  (file not present — skipped)")
    end
end

println("\n── V1b filter containment in 𝒜⁺ ─────────────────────────────────────")
let bad = [(s, l) for s in 1:S, l in 1:R if CELL_MASK[s, l] && !AA_ACTIVE[s, AA_OF_ZE[l]]]
    @printf("  cells outside an active area : %d  → %s\n", length(bad), isempty(bad) ? "PASS" : "FAIL")
    record!("V1b", isempty(bad))
end

# ═══════════════════════════════════════════════════════════════════════════
# V3 — AA-level Sinkhorn round-trip
# ═══════════════════════════════════════════════════════════════════════════
println("\n── V3  Sinkhorn round-trip (plant T, recover it from its own aggregates) ──")
let
    Random.seed!(20250101)
    T_true = copy(T_rs_init)
    for s in 1:S, c in SECTOR_T_COLS[s]
        T_true[s, c] *= exp(0.6 * randn())
    end
    for s in 1:S
        ref = T_REF_REGION[s]
        ref > 0 && (T_true[s, :] ./= T_true[s, ref])
    end
    γ_ze, _, _ = gamma_ls_analytical(ΩL, Ωs, A, gather_T_to_ze(T_true), build_tau(α))
    target     = aggregate_gamma_to_T(γ_ze)
    res = invert_T_ge(α, ΩL, Ωs, A; target=target, T_init=copy(T_rs_init),
                      tol=1e-13, max_iter=3000)
    err = maximum(abs(log(res.T[s, c]) - log(T_true[s, c]))
                  for s in 1:S for c in SECTOR_T_COLS[s])
    ok = res.converged && err < 1e-8
    @printf("  max|Δlog T| = %.3e   converged=%s   iters=%d  → %s\n",
            err, res.converged, res.iters, ok ? "PASS" : "FAIL")
    record!("V3", ok)
end

if !GRANULAR
    println("\n(GRANULAR = false — V2 / V5 / V6 / V7 / V9 / V10 are granular-only, skipped.)")
else

# ═══════════════════════════════════════════════════════════════════════════
# V2 — the N_s root-find
# ═══════════════════════════════════════════════════════════════════════════
println("\n── V2  N_s root-find (monotonicity, planted-integer recovery, clamps) ──")
let
    Random.seed!(20250102)
    q  = 0.02 .+ 0.10 .* rand(n_good)
    Gs = (s, n) -> sum((1 - q[g])^n for g in CELLS_OF_SECTOR[s]) / length(CELLS_OF_SECTOR[s])

    mono = all(Gs(s, n) > Gs(s, n + 1)
               for s in 1:S if !isempty(CELLS_OF_SECTOR[s])
               for n in N_LO[s]:(N_HI[s] - 1))
    @printf("  G(s,·) strictly decreasing on [N_LO, N_HI] : %s\n", mono ? "PASS" : "FAIL")

    exact = true
    for s in 1:S
        isempty(CELLS_OF_SECTOR[s]) && continue
        for n_true in unique((N_LO[s] + 1, (N_LO[s] + N_HI[s]) ÷ 2, max(N_LO[s], N_HI[s] - 1)))
            tgt = Gs(s, n_true)
            lo, hi = N_LO[s], N_HI[s]
            while hi - lo > 1
                mid = (lo + hi) ÷ 2
                Gs(s, mid) > tgt ? (lo = mid) : (hi = mid)
            end
            nhat = abs(Gs(s, lo) - tgt) <= abs(Gs(s, hi) - tgt) ? lo : hi
            nhat == n_true || (exact = false;
                @printf("    sector %d: planted %d, recovered %d\n", s, n_true, nhat))
        end
    end
    @printf("  bisection recovers a planted integer exactly : %s\n", exact ? "PASS" : "FAIL")

    _, c_hi = concentrate_N_s(fill(1e-9, n_good))   # q≈0 ⇒ G≈1 > target ⇒ clamp :hi
    _, c_lo = concentrate_N_s(fill(0.9,  n_good))   # q≈1 ⇒ G≈0 < target ⇒ clamp :lo
    clamps  = all(c_hi .== :hi) && all(c_lo .== :lo)
    @printf("  clamps fire at both bounds : %s  (%s / %s)\n",
            clamps ? "PASS" : "FAIL", string(c_hi), string(c_lo))
    record!("V2", mono && exact && clamps)
end

# ═══════════════════════════════════════════════════════════════════════════
# V5 — prefix stability (the property that licenses ONE solve per replication)
# ═══════════════════════════════════════════════════════════════════════════
println("\n── V5  prefix stability ─────────────────────────────────────────────")
let
    net_pool = solve_network(θ0; u_draws=U_POOL, sample_weights=POOL_WEIGHTS)
    n    = min(17, size(U_POOL, 1))
    Usub = U_POOL[1:n, :]
    Wsub = fill(1.0 / n, n, n_good)
    net_sub = solve_network(θ0; u_draws=Usub, sample_weights=Wsub)
    ok = net_sub.linkages_flat == net_pool.linkages_flat[1:n, :]
    @printf("  winners for the first %d varieties identical at width %d vs pool width %d : %s\n",
            n, n, size(U_POOL, 1), ok ? "PASS" : "FAIL")
    ok || println("    ⚠ without this, the prefix arithmetic of plan D1 is invalid and every " *
                  "candidate N_s would need its own Ricardian solve.")
    record!("V5", ok)
end

# ═══════════════════════════════════════════════════════════════════════════
# V6 / V7 / V9 / V10 — the fitted-model diagnostics
# ═══════════════════════════════════════════════════════════════════════════
println("\n── V6 / V7 / V9 / V10  granular diagnostics at θ ────────────────────")
let
    info = granular_report(θ0)
    @printf("  %-8s %8s %8s %8s %8s %10s %10s %10s\n",
            "sector", "N_LO", "N̂_s", "N_HI", "clamp", "G0_fit", "G0_target", "N^count")
    for s in 1:S
        @printf("  %-8d %8d %8d %8d %8s %10.4f %10.4f %10.1f\n",
                s, N_LO[s], info.N_hat[s], N_HI[s], string(info.clamped[s]),
                isempty(info.G0) ? NaN : info.G0[s], G_TARGET[s], info.N_count[s])
    end

    # V9 — a persistent clamp is a rejection signal for the mechanism.
    v9 = all(info.clamped .== :none)
    @printf("\n  V9  no sector clamped at a bound : %s\n", v9 ? "PASS" : "FAIL")
    v9 || println("      ⚠ :hi means the model cannot generate enough sparsity even when every " *
                  "variety is sourced from a single origin — a rejection signal, not a nuisance.")
    record!("V9", v9)

    # V6 — the log-z coefficient is a free over-identifying test (Prop. 1(c)).
    v6 = isfinite(info.b_logz) && abs(info.b_logz + theta) < 0.30 * max(theta, 1e-8)
    @printf("  V6  b_logz = %+.4f  vs  −θ = %+.4f  (rel gap %.1f%%) : %s\n",
            info.b_logz, -theta, 100 * abs(info.b_logz + theta) / max(theta, 1e-8),
            v6 ? "PASS" : "REPORT")
    println("      (a large gap invalidates the coefficient comparison and calls for " *
            "granular_validation.md §A.2 option 2 — it is reported, not enforced.)")
    record!("V6", v6, "reported")

    # V7 — the two routes to N_s should agree in order of magnitude.
    ratio = info.N_count ./ max.(info.N_hat, 1)
    v7 = all(0.2 .<= ratio .<= 5.0)
    @printf("  V7  N^count / N̂_s = %s : %s\n", string(round.(ratio, digits=2)),
            v7 ? "PASS" : "REPORT")
    println("      (a large gap is a mechanism finding, not a bug.)")
    record!("V7", v7, "reported")

    # V10 — block 4 carries no N_s term (Prop. 1), so reg_coef should barely move.
    _, sim1 = full_SMM(θ0; u_draws=U_POOL, sample_weights=POOL_WEIGHTS)
    N2 = [min(2 * n, N_BLOCK) for n in info.N_hat]
    _, sim2 = full_SMM(θ0; u_draws=U_POOL, sample_weights=POOL_WEIGHTS, N_fixed=N2)
    drift = maximum(abs.(vec(sim2[4]) .- vec(sim1[4])) ./ (abs.(vec(sim1[4])) .+ 1e-10))
    v10 = drift < 0.10
    @printf("  V10 reg_coef drift from N̂ to 2N̂ = %.3f : %s\n", drift, v10 ? "PASS" : "REPORT")
    @printf("      reg_coef(N̂)  = %s\n", string(round.(vec(sim1[4]), digits=4)))
    @printf("      reg_coef(2N̂) = %s\n", string(round.(vec(sim2[4]), digits=4)))
    println("      (should be ~0: the firm-level index contains no N_s term. A visible drift " *
            "means the union-over-buyers approximation or the firm mapping is doing something " *
            "unintended — or R_REP is too small for the binding-function average to settle.)")
    record!("V10", v10, "reported")

    # Ḡ_s(0) must be decreasing in N_s — the property the bisection relies on.
    if !isempty(sim1[6]) && !isempty(sim2[6])
        okG = all(vec(sim2[6]) .<= vec(sim1[6]) .+ 1e-12)
        @printf("  ✔   Ḡ_s(0) decreasing in N_s : %s   (%s → %s)\n", okG ? "PASS" : "FAIL",
                string(round.(vec(sim1[6]), digits=4)), string(round.(vec(sim2[6]), digits=4)))
        record!("G0-mono", okG)
    end
end

end  # if GRANULAR

# ═══════════════════════════════════════════════════════════════════════════
println("\n" * "="^72)
println("SUMMARY")
println("="^72)
for (name, ok, note) in results
    @printf("  %-10s %s%s\n", name, ok ? "PASS" : "FAIL",
            isempty(note) ? "" : "  ($note)")
end
let hard = [r for r in results if isempty(r[3])]
    @printf("\n  %d/%d enforced gates pass%s\n", count(r -> r[2], hard), length(hard),
            all(r -> r[2], hard) ? "" : "  ← INVESTIGATE")
end
println("="^72)
