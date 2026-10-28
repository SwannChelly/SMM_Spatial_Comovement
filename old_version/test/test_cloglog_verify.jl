##### test_cloglog_verify.jl #####
# Correctness check for the hand-rolled cloglog IRLS kernel `_cloglog_irls`
# (model_CP.jl) against GLM.jl's battle-tested `glm(..., Binomial(), CloglogLink())`.
# The kernel absorbs one categorical fixed effect by weighted within-group demeaning
# (Frisch–Waugh–Lovell); GLM.jl fits the same model with the FE as explicit dummies.
# On the same synthetic data the two must return the SAME slope coefficients.
#
#     julia test/test_cloglog_verify.jl
#
# Requires GLM.jl (added to pkg.jl). Standalone — includes only ../model_CP.jl to get
# `_cloglog_irls`; no load_parameters / data needed (the kernel takes a raw design).

using Random, LinearAlgebra, Printf
using DataFrames, CategoricalArrays
using GLM

include("../model_CP.jl")   # brings in _cloglog_irls

# GLM.jl cloglog slope on a named coefficient (FE as dummies via @formula).
glm_slope(m, name) = coef(m)[findfirst(==(name), coefnames(m))]

# Both fits must reach the SAME optimum. GLM.jl's default deviance tolerance
# (rtol=1e-6) stops slightly before the MLE (β off by ~1e-4); tighten it so the
# comparison is MLE-vs-MLE, not convergence-noise. Same for our IRLS.
const GLMKW  = (maxiter = 500, atol = 1e-12, rtol = 1e-12)
const IRLSKW = (max_iter = 300, tol = 1e-12)

function report(label, b_mine, b_glm; tol = 1e-6)
    d = maximum(abs.(b_mine .- b_glm))
    @printf("\n[%s]\n", label)
    @printf("  %-8s %16s %16s\n", "coef", "IRLS (mine)", "GLM.jl")
    for i in eachindex(b_mine)
        @printf("  %-8d %16.9f %16.9f\n", i, b_mine[i], b_glm[i])
    end
    @printf("  max|Δ| = %.3e   %s\n", d, d < tol ? "PASS ✓" : "FAIL ✗")
    @assert d < tol "cloglog IRLS disagrees with GLM.jl (max|Δ|=$d ≥ $tol) for $label"
    return d
end

println("="^70)
println("cloglog IRLS  vs  GLM.jl  (CloglogLink)  — correctness gate")
println("="^70)

# ── Case A: continuous regressors + FE, UNWEIGHTED ───────────────────────────
Random.seed!(123)
let
    n, G = 6000, 40
    fe = rand(1:G, n)
    a  = randn(G) .* 0.8                       # FE levels
    x1 = randn(n)                              # e.g. log-distance
    x2 = randn(n)                              # e.g. log-size control
    β  = [0.7, -0.5]
    η  = a[fe] .+ β[1] .* x1 .+ β[2] .* x2
    y  = Float64.(rand(n) .< (1 .- exp.(-exp.(η))))

    b_mine = _cloglog_irls(y, hcat(x1, x2), ones(n), fe; IRLSKW...)

    df = DataFrame(y = y, x1 = x1, x2 = x2, fe = categorical(fe))
    m  = glm(@formula(y ~ x1 + x2 + fe), df, Binomial(), CloglogLink(); GLMKW...)
    b_glm = [glm_slope(m, "x1"), glm_slope(m, "x2")]

    report("A: continuous + FE, unweighted", b_mine, b_glm)
end

# ── Case B: dummy (distance-bin) regressors + continuous size + FE ────────────
let
    n, G, K = 8000, 30, 3                      # K distance-bin dummies (base = no bin)
    fe  = rand(1:G, n)
    a   = randn(G) .* 0.6
    bin = rand(0:K, n)                          # 0 = base category
    xz  = randn(n)                              # size control
    βb  = [0.5, 0.9, 1.3]; βz = -0.4
    D   = Float64[bin[i] == b for i in 1:n, b in 1:K]
    η   = a[fe] .+ D * βb .+ βz .* xz
    y   = Float64.(rand(n) .< (1 .- exp.(-exp.(η))))

    b_mine = _cloglog_irls(y, hcat(D, xz), ones(n), fe; IRLSKW...)

    df = DataFrame(y = y, b1 = D[:,1], b2 = D[:,2], b3 = D[:,3], xz = xz,
                   fe = categorical(fe))
    m  = glm(@formula(y ~ b1 + b2 + b3 + xz + fe), df, Binomial(), CloglogLink(); GLMKW...)
    b_glm = [glm_slope(m, "b1"), glm_slope(m, "b2"), glm_slope(m, "b3"), glm_slope(m, "xz")]

    report("B: distance-bin dummies + size + FE", b_mine, b_glm)
end

# ── Case C: WEIGHTED — frequency weights vs GLM on row-expanded data ──────────
# Integer weights verified exactly by replicating rows and fitting GLM unweighted;
# pins down the analytic-weight semantics without depending on GLM's wts type.
let
    n, G = 3000, 25
    fe = rand(1:G, n)
    a  = randn(G) .* 0.7
    x1 = randn(n); x2 = randn(n)
    β  = [0.6, -0.35]
    η  = a[fe] .+ β[1] .* x1 .+ β[2] .* x2
    y  = Float64.(rand(n) .< (1 .- exp.(-exp.(η))))
    wint = Float64.(rand(1:4, n))              # integer frequency weights

    b_mine = _cloglog_irls(y, hcat(x1, x2), wint, fe; IRLSKW...)

    ridx = reduce(vcat, [fill(i, Int(wint[i])) for i in 1:n])   # expand
    dfE  = DataFrame(y = y[ridx], x1 = x1[ridx], x2 = x2[ridx], fe = categorical(fe[ridx]))
    mE   = glm(@formula(y ~ x1 + x2 + fe), dfE, Binomial(), CloglogLink(); GLMKW...)
    b_glm = [glm_slope(mE, "x1"), glm_slope(mE, "x2")]

    report("C: frequency-weighted vs expanded GLM", b_mine, b_glm)
end

println("\n" * "="^70)
println("ALL CASES PASS ✓  — _cloglog_irls matches GLM.jl to tolerance.")
println("="^70)
