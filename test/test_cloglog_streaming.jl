##### test_cloglog_streaming.jl #####
# Correctness gate for the STREAMING extensive-margin kernels (model_CP.jl), the
# memory-light replacements that are now the production path for BOTH links:
#   `_cloglog_irls_cells`  replaces the dense `_cloglog_irls`   (REG_METHOD = :cloglog)
#   `_wls_cells`           replaces the dense FWL/QR kernel     (REG_METHOD = :lpm)
# Neither ever materialises the n_cell × N_rho design.
#
# Section 1 (cloglog) is a three-way comparison on every case:
#
#   streaming  — `_cloglog_irls_cells`, which never materialises the n_cell × N_rho
#                design and never stores a per-row vector;
#   dense      — `_cloglog_irls`, the previous production kernel, on the equivalent
#                materialised design (itself gated against GLM.jl by
#                test/test_cloglog_verify.jl);
#   GLM.jl     — `glm(..., Binomial(), CloglogLink())` with the fixed effect as
#                explicit dummies, the external reference.
#
# All three must return the SAME coefficients. The weights on the production design are
# uniform (1/N_rho on every row, goods and control cells alike), so an unweighted GLM
# fit is the same MLE and no row expansion is needed here — unlike Case C of
# test_cloglog_verify.jl, which exercises non-uniform frequency weights.
#
# Section 1b does the same for the LPM sibling, against the dense FWL kernel and an
# independent explicit-dummy OLS.
#
#     julia test/test_cloglog_streaming.jl
#
# Standalone: includes only ../model_CP.jl and defines the handful of geography globals
# the design builders read, so no load_parameters / data / worker pool is needed.

using Random, LinearAlgebra, Printf
using DataFrames, CategoricalArrays
using GLM

include("../model_CP.jl")

# Both fits must reach the SAME optimum, so tolerances are tightened well past the
# production settings — this is an MLE-vs-MLE comparison, not a convergence-noise one.
const GLMKW  = (maxiter = 500, atol = 1e-12, rtol = 1e-12)
const IRLSKW = (max_iter = 300, tol = 1e-13)

glm_slope(m, name) = coef(m)[findfirst(==(name), coefnames(m))]

const RESULTS = Tuple{String, Float64, Float64}[]

# The two comparisons deserve VERY different tolerances, and the asymmetry is the
# point rather than a fudge.
#
#   streaming vs dense — both kernels are driven to the same fixed point by the same
#     stopping rule (max|Δβ| < 1e-13), so the only thing between them is the order the
#     sums are accumulated in. Expect floating-point noise, ~1e-16. TOL_DENSE is the
#     gate that would actually catch a mistake in the streaming algebra.
#
#   streaming vs GLM.jl — GLM stops on the DEVIANCE, not on β. Deviance is quadratic
#     in β at the optimum, so a relative deviance tolerance of δ pins β only to about
#     √δ: at GLM's floor (δ = 1e-12, already the tightest it accepts) that is ~1e-6.
#     GLM therefore halts measurably short of the MLE, and the residual gap is GLM's,
#     not ours — the sibling gate test/test_cloglog_verify.jl sees the same ~3e-8 for
#     the DENSE kernel and passes it at 1e-6. This comparison is an independent check
#     that we are fitting the right MODEL; it cannot resolve the last few digits.
const TOL_DENSE = 1e-10
const TOL_GLM   = 1e-5

function report(label, b_stream, b_dense, b_glm;
                tol_dense = TOL_DENSE, tol_glm = TOL_GLM)
    d_sd = maximum(abs.(b_stream .- b_dense))
    d_sg = maximum(abs.(b_stream .- b_glm))
    scale = max(maximum(abs.(b_stream)), 1e-12)
    @printf("\n[%s]\n", label)
    @printf("  %-6s %18s %18s %18s\n", "coef", "streaming", "dense", "GLM.jl")
    for i in eachindex(b_stream)
        @printf("  %-6d %18.12f %18.12f %18.12f\n", i, b_stream[i], b_dense[i], b_glm[i])
    end
    @printf("  max|stream−dense| = %.3e  (tol %.0e)  %s\n",
            d_sd, tol_dense, d_sd < tol_dense ? "PASS ✓" : "FAIL ✗")
    # NB: @printf needs a LITERAL format string — a `"..." * "..."` concatenation is an
    # expression and does not compile. Keep this on one line.
    @printf("  max|stream−GLM|   = %.3e  (tol %.0e, rel %.1e — GLM's convergence floor)  %s\n",
            d_sg, tol_glm, d_sg / scale, d_sg < tol_glm ? "PASS ✓" : "FAIL ✗")
    push!(RESULTS, (label, d_sd, d_sg))
    @assert d_sd < tol_dense (
        "streaming disagrees with the dense kernel (max|Δ|=$d_sd ≥ $tol_dense) for " *
        "$label — this is the gate that matters; the two kernels must agree to " *
        "floating-point noise.")
    @assert d_sg < tol_glm (
        "streaming disagrees with GLM.jl (max|Δ|=$d_sg ≥ $tol_glm) for $label — a gap " *
        "this large is beyond GLM's deviance-stopping floor and indicates a genuinely " *
        "different model, not a convergence difference.")
    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Build a cell-structured design and its DENSE equivalent, in the same row order
# the dense builder produces: cells 1:n_good_c are goods, then the control cells,
# each contributing n_rho consecutive rows.
#
# `continuous_geo` mirrors the N_REG == 1 branch (one CONTINUOUS log-distance
# regressor, cell-constant) as opposed to the N_REG > 1 bin-dummy branch.
# ─────────────────────────────────────────────────────────────────────────────
function build_case(; seed, n_good_c, n_ctrl_c, n_rho, n_reg, n_grp,
                      size_control, continuous_geo = false)
    rng = MersenneTwister(seed)
    n_cell = n_good_c + n_ctrl_c

    if continuous_geo
        gcol = ones(Int, n_cell)
        gval = randn(rng, n_cell)
        k_geo = 1
    else
        gcol = rand(rng, 0:n_reg, n_cell)          # 0 = base category (all-zero row)
        gval = [gcol[c] > 0 ? 1.0 : 0.0 for c in 1:n_cell]
        k_geo = n_reg
    end

    grp = rand(rng, 1:n_grp, n_cell)
    grp[1:n_grp] .= 1:n_grp                        # guarantee every group is non-empty

    has_z = [size_control && c <= n_good_c for c in 1:n_cell]
    lzc   = [has_z[c] ? 0.5 * randn(rng) - 1.0 : 0.0 for c in 1:n_cell]
    lzr   = size_control ? 0.8 .* randn(rng, n_rho, n_good_c) : nothing
    links = rand(rng, n_rho, n_good_c) .< 0.35     # BitMatrix, as solve_network returns
    wflat = 1.0 / n_rho
    sw    = FlatWeights(wflat, n_rho, n_good_c)

    k        = k_geo + (size_control ? 1 : 0)
    size_col = k_geo + 1

    cells = RegCells(n_good_c, n_ctrl_c, n_rho, gcol, gval, grp, n_grp, has_z, lzc)

    # Dense equivalent.
    N = n_cell * n_rho
    X  = zeros(N, k)
    y  = zeros(N)
    w  = zeros(N)
    fe = zeros(Int, N)
    i = 0
    for c in 1:n_cell
        is_good = c <= n_good_c
        for rho in 1:n_rho
            i += 1
            y[i]  = is_good ? (links[rho, c] > 0 ? 0.0 : 1.0) : 1.0
            w[i]  = wflat
            fe[i] = grp[c]
            gcol[c] > 0 && (X[i, gcol[c]] = gval[c])
            has_z[c] && (X[i, size_col] = lzc[c] + lzr[rho, c])
        end
    end
    @assert i == N

    return (cells = cells, links = links, sw = sw, lzr = lzr, k = k, size_col = size_col,
            X = X, y = y, w = w, fe = fe)
end

function glm_fit(C)
    df = DataFrame(y = C.y, fe = categorical(C.fe))
    names = String[]
    for j in 1:C.k
        nm = "x$j"
        df[!, nm] = C.X[:, j]
        push!(names, nm)
    end
    # Programmatic formula (StatsModels' documented idiom) — the number of regressors
    # varies by case, so @formula cannot be used here.
    f = term(:y) ~ sum(term.(Symbol.(names))) + term(:fe)
    m = glm(f, df, Binomial(), CloglogLink(); GLMKW...)
    return [glm_slope(m, nm) for nm in names]
end

println("="^78)
println("streaming cloglog  vs  dense IRLS  vs  GLM.jl — correctness gate")
println("="^78)

# ── Section 1: the kernel, across every design shape the production code can hit ──
const CASES = [
    ("bin dummies + size control, no control cells  (:aa, --controls=false)",
     (seed = 11, n_good_c = 60, n_ctrl_c = 0,  n_rho = 40, n_reg = 4, n_grp = 8,
      size_control = true)),
    ("bin dummies + control cells, no size control  (:ze, --controls=true)",
     (seed = 22, n_good_c = 50, n_ctrl_c = 25, n_rho = 30, n_reg = 4, n_grp = 6,
      size_control = false)),
    ("bin dummies + size control + control cells    (mixed)",
     (seed = 33, n_good_c = 40, n_ctrl_c = 20, n_rho = 35, n_reg = 3, n_grp = 5,
      size_control = true)),
    ("continuous log-distance (N_REG == 1) + size control",
     (seed = 44, n_good_c = 55, n_ctrl_c = 0,  n_rho = 45, n_reg = 1, n_grp = 7,
      size_control = true, continuous_geo = true)),
    ("continuous log-distance (N_REG == 1), no size control",
     (seed = 55, n_good_c = 45, n_ctrl_c = 15, n_rho = 25, n_reg = 1, n_grp = 6,
      size_control = false, continuous_geo = true)),
]

for (label, kw) in CASES
    C = build_case(; kw...)
    b_stream = _cloglog_irls_cells(C.cells, C.links, C.sw, C.lzr, C.k, C.size_col;
                                   IRLSKW...)
    b_dense  = _cloglog_irls(C.y, C.X, C.w, C.fe; IRLSKW...)
    b_glm    = glm_fit(C)
    report(label, b_stream, b_dense, b_glm)
end

# ── Section 1b: the LPM sibling (`_wls_cells` vs the dense FWL kernel) ────────
# `fast_weighted_regression` (REG_METHOD = :lpm) got the same treatment, so it needs
# the same gate. Three routes to the same weighted LS with one absorbed FE:
#   streaming  — `_wls_cells`, closed-form within transform, one pass, no design;
#   dense-FWL  — the previous kernel verbatim: demean in place, then sqrt(w)-weighted
#                pivoted-QR least squares;
#   dummy-OLS  — explicit FE dummies, no absorption at all — an independent route that
#                shares no code with either.
#
# The LPM outcome is `supplier`, the complement of the cloglog kernel's `not_supply`,
# so the design built above is reused with y flipped.
function dense_wls(y, X, w, fe)
    y = copy(y); X = copy(X)
    for g in unique(fe)
        mask = fe .== g
        w_g = w[mask]; tw = sum(w_g)
        tw < 1e-15 && continue
        y[mask] .-= sum(w_g .* y[mask]) / tw
        for j in 1:size(X, 2)
            X[mask, j] .-= sum(w_g .* X[mask, j]) / tw
        end
    end
    sq = sqrt.(w)
    return (sq .* X) \ (sq .* y)
end

function dummy_wls(y, X, w, fe)
    lev = unique(fe)
    D = Float64[fe[i] == g for i in eachindex(fe), g in lev]
    sq = sqrt.(w)
    return ((sq .* hcat(X, D)) \ (sq .* y))[1:size(X, 2)]
end

# Streaming solves the NORMAL equations where the dense kernel takes a QR of the
# sqrt-weighted design; squaring the design squares its condition number, so this is
# looser than the cloglog gate by design. It is still far tighter than anything the
# estimator can notice.
const TOL_LPM = 1e-9

for (label, kw) in CASES
    C = build_case(; kw...)
    y_lpm = 1.0 .- C.y                       # supplier = 1 − not_supply
    b_stream = _wls_cells(C.cells, C.links, C.sw, C.lzr, C.k, C.size_col)
    b_dense  = dense_wls(y_lpm, C.X, C.w, C.fe)
    b_dummy  = dummy_wls(y_lpm, C.X, C.w, C.fe)
    d_sd = maximum(abs.(b_stream .- b_dense))
    d_su = maximum(abs.(b_stream .- b_dummy))
    @printf("\n[LPM — %s]\n", label)
    @printf("  %-6s %18s %18s %18s\n", "coef", "streaming", "dense FWL", "dummy OLS")
    for i in eachindex(b_stream)
        @printf("  %-6d %18.12f %18.12f %18.12f\n", i, b_stream[i], b_dense[i], b_dummy[i])
    end
    @printf("  max|stream−dense| = %.3e   max|stream−dummy| = %.3e  (tol %.0e)  %s\n",
            d_sd, d_su, TOL_LPM, max(d_sd, d_su) < TOL_LPM ? "PASS ✓" : "FAIL ✗")
    push!(RESULTS, ("LPM — $label", d_sd, d_su))
    @assert d_sd < TOL_LPM "LPM streaming disagrees with the dense FWL kernel ($d_sd)"
    @assert d_su < TOL_LPM "LPM streaming disagrees with explicit-dummy OLS ($d_su)"
end

# ── Section 2: a degenerate group (all its weight on one bin) ────────────────
# The streaming kernel forms the within-group sum of squares as
# `Σ W X X' − Σ_p S_p S_p' / S_p^W`, a difference of two large quantities where the
# dense kernel demeans row by row. A group whose cells all share one bin is where that
# cancellation is worst — the dummy contributes NO within variation there, so the two
# terms nearly cancel exactly. This checks the two kernels still agree.
let
    C = build_case(seed = 66, n_good_c = 48, n_ctrl_c = 0, n_rho = 30,
                   n_reg = 4, n_grp = 6, size_control = true)
    gcol = copy(C.cells.gcol)
    for c in 1:length(gcol)
        C.cells.grp[c] == 1 && (gcol[c] = 2)        # group 1 becomes bin-homogeneous
    end
    cells2 = RegCells(C.cells.n_good_c, C.cells.n_ctrl_c, C.cells.n_rho,
                          gcol, C.cells.gval, C.cells.grp, C.cells.n_grp,
                          C.cells.has_z, C.cells.lzc)
    X2 = copy(C.X)
    i = 0
    for c in 1:length(gcol), rho in 1:C.cells.n_rho
        i += 1
        X2[i, 1:(C.size_col - 1)] .= 0.0
        gcol[c] > 0 && (X2[i, gcol[c]] = C.cells.gval[c])
    end
    b_stream = _cloglog_irls_cells(cells2, C.links, C.sw, C.lzr, C.k, C.size_col; IRLSKW...)
    b_dense  = _cloglog_irls(C.y, X2, C.w, C.fe; IRLSKW...)
    b_glm    = glm_fit((y = C.y, X = X2, fe = C.fe, k = C.k))
    report("bin-homogeneous fixed-effect group (worst-case cancellation)",
           b_stream, b_dense, b_glm)
end

# ── Section 3: the full fast_cloglog_regression, streaming vs dense ─────────────
# Exercises `_build_reg_cells` (the geography → cell mapping) and the log-z
# decomposition `log z = logz_const + logz_resid` against the dense path's `log(z_flat)`.
# The globals below are the ones the two design builders read; they are ordinary
# (non-const) globals so the block is self-contained without load_parameters.jl.
println("\n" * "-"^78)
println("Section 3: fast_cloglog_regression — streaming path vs dense path")
println("-"^78)

let
    rng = MersenneTwister(777)
    # 30 regions over 5 downstream regions ⇒ 6 regions share each nearest-downstream
    # region, which is what gives the distance bin room to VARY WITHIN a fixed-effect
    # group. See the DistBin comment below.
    S_, R_, Rd_ = 3, 30, 5
    goods = [(s, r) for s in 1:S_ for r in 1:R_ if (s + r) % 3 != 0]
    ctrls = [(s, r) for s in 1:S_ for r in 1:R_ if (s + r) % 3 == 0]

    global CA_LEVEL   = :aa
    global theta      = 1.3
    global N_REG      = 4
    global R_downstream = Rd_
    global n_good     = length(goods)
    global GOOD_S     = [g[1] for g in goods]
    global GOOD_R     = [g[2] for g in goods]
    global N_CONTROL  = length(ctrls)
    global CONTROL_S  = [c[1] for c in ctrls]
    global CONTROL_R  = [c[2] for c in ctrls]
    global CLOSEST_DOWNSTREAM_REGION = [1 + (r - 1) % Rd_ for r in 1:R_]
    # The synthetic bin map has to satisfy TWO conditions the real one does, or the
    # design is not estimable — for either kernel, not just the streaming one:
    #
    #  (a) the BASE category (bin 0) must be non-empty. `distance_bin` (tools.jl) returns
    #      0 below its first cutoff for exactly this reason: if every cell carried a
    #      dummy, the N_REG columns would sum to the constant, which the fixed effect has
    #      already absorbed, so (1,…,1,0) would be an exact null vector.
    #  (b) the bin must VARY WITHIN a fixed-effect group. The group is
    #      (sector × nearest-downstream region), so a bin that is a function of the
    #      nearest-downstream region alone is absorbed by the FE and demeans to zero.
    #
    # Indexing on ((r-1) ÷ Rd_) gives each group's six regions the bins 0,1,2,3,4,0 —
    # every bin represented inside every group, base category occupied.
    global DistBin    = [((r - 1) ÷ Rd_) % (N_REG + 1) for r in 1:R_, dr in 1:Rd_]
    global LOG_CLOSEST_DIST = log.(5.0 .+ collect(1:R_))

    N_rho_t = 60
    u = rand(rng, N_rho_t, n_good) .* 0.998 .+ 0.001
    swm = FlatWeights(1.0 / N_rho_t, N_rho_t, n_good)

    # z built exactly as solve_network builds it, with the matching logz_const.
    T_g  = 0.05 .+ 1.5 .* rand(rng, n_good)
    lzc  = [log(max(T_g[g], eps(Float64))^(1 / theta)) for g in 1:n_good]
    z    = [max(T_g[g], eps(Float64))^(1 / theta) * (-log(1.0 - u[rho, g]))^(-1 / theta)
            for rho in 1:N_rho_t, g in 1:n_good]
    links = rand(rng, N_rho_t, n_good) .< 0.4

    for reg_fun in (fast_cloglog_regression, fast_weighted_regression)
        fname = reg_fun === fast_cloglog_regression ? "cloglog" : "lpm"
        for (lbl, inc_ctrl, inc_size) in (("controls=false, size control on", false, true),
                                          ("controls=true,  size control off", true, false))
            reset_cloglog_design!()
            reset_logz_resid!()
            # The cloglog kernel takes IRLS controls; the LPM is a single solve.
            kw = reg_fun === fast_cloglog_regression ? IRLSKW : NamedTuple()

            REG_STREAMING[] = true
            b_stream = reg_fun(links, z, swm;
                               include_control      = inc_ctrl,
                               include_size_control = inc_size,
                               return_size_coef     = inc_size,
                               u_draws    = u,
                               logz_const = lzc,
                               inv_theta  = 1 / theta,
                               kw...)
            REG_STREAMING[] = false
            b_dense = reg_fun(links, z, swm;
                              include_control      = inc_ctrl,
                              include_size_control = inc_size,
                              return_size_coef     = inc_size,
                              kw...)
            REG_STREAMING[] = true

            d = maximum(abs.(b_stream .- b_dense))
            @printf("\n[full path — %s — %s]\n", fname, lbl)
            @printf("  %-6s %18s %18s\n", "coef", "streaming", "dense")
            for i in eachindex(b_stream)
                @printf("  %-6d %18.12f %18.12f\n", i, b_stream[i], b_dense[i])
            end
            # Looser than the kernel gate by design, for two reasons: the streaming path
            # reconstructs the size regressor as `logz_const + logz_resid` where the
            # dense path takes `log(z_flat)` (equal to the last ulp, ≈9e-16, not
            # bitwise), and the LPM streaming path solves the normal equations where the
            # dense one takes a QR. Measured gaps are ~1e-15; 1e-9 is a wide margin.
            @printf("  max|Δ| = %.3e   %s\n", d, d < 1e-9 ? "PASS ✓" : "FAIL ✗")
            push!(RESULTS, ("full path — $fname — $lbl", d, NaN))
            @assert d < 1e-9 "streaming and dense $fname regression disagree ($d)"
        end
    end
end

# ── Section 4: what the rewrite was for — allocations and time ────────────────
println("\n" * "-"^78)
println("Section 4: allocation / timing, production-shaped design")
println("-"^78)

let
    C = build_case(seed = 99, n_good_c = 1161, n_ctrl_c = 0, n_rho = 723,
                   n_reg = 4, n_grp = 30, size_control = true)
    KW = (max_iter = 8, tol = 1e-13)      # same iteration budget on both sides

    _cloglog_irls_cells(C.cells, C.links, C.sw, C.lzr, C.k, C.size_col; KW...)
    _cloglog_irls(C.y, C.X, C.w, C.fe; KW...)      # warm up both

    a_s = @allocated _cloglog_irls_cells(C.cells, C.links, C.sw, C.lzr, C.k, C.size_col; KW...)
    t_s = @elapsed  _cloglog_irls_cells(C.cells, C.links, C.sw, C.lzr, C.k, C.size_col; KW...)
    a_d = @allocated _cloglog_irls(C.y, C.X, C.w, C.fe; KW...)
    t_d = @elapsed  _cloglog_irls(C.y, C.X, C.w, C.fe; KW...)

    design_dense = (sizeof(C.X) + sizeof(C.y) + sizeof(C.w) + sizeof(C.fe)) / 2^20
    design_cells = (sizeof(C.cells.gcol) + sizeof(C.cells.gval) + sizeof(C.cells.grp) +
                    sizeof(C.cells.has_z) + sizeof(C.cells.lzc)) / 2^20
    lzr_mb = sizeof(C.lzr) / 2^20

    @printf("\n  n_good = %d, N_rho = %d  ⇒  %d dense rows, %d columns\n",
            1161, 723, length(C.y), C.k)
    @printf("  %-34s %12s %12s\n", "", "streaming", "dense")
    @printf("  %-34s %12.1f %12.1f\n", "design storage (MB)", design_cells, design_dense)
    @printf("  %-34s %12.1f %12.1f\n", "  + logz_resid, per draw set (MB)", lzr_mb, 0.0)
    @printf("  %-34s %12.1f %12.1f\n", "kernel allocations, 8 iters (MB)",
            a_s / 2^20, a_d / 2^20)
    @printf("  %-34s %12.3f %12.3f\n", "kernel time, 8 iters (s)", t_s, t_d)
    @printf("\n  design + allocations: %.1f MB → %.1f MB  (%.1f× smaller)\n",
            design_dense + a_d / 2^20, design_cells + lzr_mb + a_s / 2^20,
            (design_dense + a_d / 2^20) / (design_cells + lzr_mb + a_s / 2^20))
end

println("\n" * "="^78)
@printf("%-56s %11s %11s\n", "case", "|Δ| dense", "|Δ| GLM")
for (lbl, d1, d2) in RESULTS
    @printf("%-56s %11.2e %11s\n", first(lbl, 56), d1,
            isnan(d2) ? "-" : @sprintf("%.2e", d2))
end
@printf("\n%-56s %11.0e %11.0e\n", "tolerance", TOL_DENSE, TOL_GLM)
println("""
The dense column is the gate that matters — same fixed point, same stopping rule,
different summation order, so it must come out at floating-point noise. The GLM column
is an independent check on the MODEL, bounded below by GLM's deviance-based stopping
rule (~1e-6 on β even at its tightest setting); the same ~3e-8 shows up for the DENSE
kernel in test/test_cloglog_verify.jl.""")
println("\nALL CASES PASS ✓ — the streaming kernel reproduces the dense kernel and GLM.jl.")
println("="^78)
