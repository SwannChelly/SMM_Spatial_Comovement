##### test_sensitivity.jl #####
# Gate for sensitivity.jl (the CRN Jacobian and the one-parameter profiles).
# STANDALONE + PRINT-ONLY. Run from the directory holding the baseline:
#
#     python3 SMM_Spatial_Comovement/test/make_synthetic_baseline.py ./baseline_syn
#     julia -p 3 SMM_Spatial_Comovement/test/test_sensitivity.jl syn [n_coef]
#
#   (1) DETERMINISM: the estimation-draw Jacobian is a function of θ (two runs agree).
#   (2) STRUCTURAL ZEROS: the N_s columns are exactly zero outside blocks 4 and 6 (the
#       value blocks use the full draw set), and the head columns exactly zero on
#       blocks 4 and 6 (with wages ≡ 1 the head does not change who wins).
#   (3) PROFILES: the point at the estimate reproduces the base moments on the
#       estimation draws (α, κ: T re-inverted at θ̂ is θ̂'s own T*), and Ḡ_s(0) is
#       strictly decreasing along the N_s grid.

using Distributed
@everywhere using Random, NPZ, DataFrames, CSV, Distributions, Statistics, QuasiMonteCarlo,
                  LinearAlgebra, Printf, SpecialFunctions, Optim
const REPO = normpath(joinpath(@__DIR__, ".."))
@everywhere include(joinpath($REPO, "model_CP.jl"))
@everywhere include(joinpath($REPO, "profiling.jl"))
@everywhere include(joinpath($REPO, "inference.jl"))
@everywhere include(joinpath($REPO, "sensitivity.jl"))

industry        = length(ARGS) >= 1 ? ARGS[1] : "syn"
n_coef          = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4
n_tau           = 1
n_rho_inference = 1000
relax_n_lo      = false
profile_head    = get(ENV, "PROFILE_HEAD", "false") == "true"
input_folder    = get(ENV, "INPUT_FOLDER", "./baseline_$industry")
output_folder   = mktempdir()
include(joinpath(REPO, "load_parameters.jl"))

pass = true
check(name, ok) = (global pass &= ok; @printf("  [%s] %s\n", ok ? "PASS" : "FAIL", name))

x = vcat(agg_labor_share, agg_industry_share, ones(R_downstream), collect(TAU_PRIOR),
         KAPPA_PRIOR, vec(permutedims(T_rs_init))[T_MASK])
θ = profiled_theta_full(x).theta
d1, d2 = mktempdir(), mktempdir()
r1 = sensitivity_jacobian(θ; K = 2, dest = d1)
r2 = sensitivity_jacobian(θ; K = 2, dest = d2)

println("\n(1) Determinism")
check("estimation-draw Jacobian identical across two runs", isequal(r1.J_est, r2.J_est))
check("independent-set mean identical across two runs (seeded)", isequal(r1.J, r2.J))

println("\n(2) Structural zeros")
blk = zeros(Int, sum(MOMENT_MASK))
for (b, rg) in enumerate(BLOCK_RANGES); blk[rg] .= b; end
cols = sens_columns(θ)
iN = findall(c -> c.kind === :N, cols)
iH = findall(c -> c.kind === :head, cols)
off46 = findall(b -> !(b in (4, 6)), blk)
on46  = findall(b -> b in (4, 6), blk)
check("N_s columns exactly zero outside blocks 4 and 6", all(r1.J_est[off46, iN] .== 0))
check("N_s columns move Ḡ_s(0)", all(any(r1.J_est[collect(BLOCK_RANGES[6]), j] .!= 0) for j in iN))
check("head columns exactly zero on blocks 4 and 6", all(r1.J_est[on46, iH] .== 0))
check("every column finite on the estimation draws", all(isfinite, r1.J_est))

println("\n(3) Profiles")
dp = mktempdir()
sensitivity_profiles(θ; K = 1, G = 5, dest = dp)
m0, _ = sens_moments(θ, sens_drawset(0))
for line in readlines(joinpath(dp, "profiles_index.txt"))
    lab, fn = split(line, '\t')
    z = NPZ.npzread(joinpath(dp, fn))
    g = z["grid"]; i0 = findfirst(==(z["estimate"][1]), g)
    gap = maximum(abs.(z["m_est"][i0, :] .- m0))
    check("$lab: the point at the estimate reproduces the base moments (gap $(round(gap, sigdigits = 2)))",
          gap < 1e-6)
    if startswith(lab, "N_s")
        t = Int(z["targets"][1])
        check("$lab: Ḡ_s(0) strictly decreasing in N_s", all(diff(z["m_est"][:, t]) .< 0))
    end
end

println("\n", pass ? "ALL SENSITIVITY GATES PASS" : "SOME SENSITIVITY GATES FAILED")
cols2 = sens_columns(θ)
for k in (:head, :alpha, :kappa, :T, :N)
    j = findall(c -> c.kind === k, cols2)
    @printf("  %-6s max |ΔJ| between runs: estimation draws %.3g, mean %.3g\n", k,
            maximum(abs.(r1.J_est[:, j] .- r2.J_est[:, j])), maximum(abs.(r1.J[:, j] .- r2.J[:, j])))
end
pass || exit(1)
