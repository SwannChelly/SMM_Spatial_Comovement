##### test_sensitivity.jl #####
# Gate for sensitivity.jl (the CRN Jacobian and the one-parameter profiles).
# STANDALONE + PRINT-ONLY. Run from the directory holding the baseline:
#
#     python3 SMM_Spatial_Comovement/test/make_synthetic_baseline.py ./baseline_syn
#     julia -p 3 SMM_Spatial_Comovement/test/test_sensitivity.jl syn [n_coef]
#
#   (1) DETERMINISM: the estimation-draw Jacobian is a function of θ (two runs agree).
#   (2) STRUCTURAL ZEROS: the N_s columns are exactly zero outside blocks 4 and 6 (the
#       value blocks use the full draw set), and the aggregate columns exactly zero on
#       blocks 4 and 6 (with wages ≡ 1 the aggregates do not change who wins).
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
input_folder    = get(ENV, "INPUT_FOLDER", "./baseline_$industry")
output_folder   = mktempdir()
include(joinpath(REPO, "load_parameters.jl"))

pass = true
check(name, ok) = (global pass &= ok; @printf("  [%s] %s\n", ok ? "PASS" : "FAIL", name))

x = vcat(agg_labor_share, agg_industry_share, ones(R_downstream), collect(TAU_PRIOR),
         KAPPA_PRIOR, vec(permutedims(T_rs_init))[T_MASK])
theta_vec = profile_theta(x).theta_vec
dest1, dest2 = mktempdir(), mktempdir()
jacobian_run1 = sensitivity_jacobian(theta_vec; K = 2, dest = dest1)
jacobian_run2 = sensitivity_jacobian(theta_vec; K = 2, dest = dest2)

println("\n(1) Determinism")
check("estimation-draw Jacobian identical across two runs", isequal(jacobian_run1.J_est, jacobian_run2.J_est))
check("independent-set mean identical across two runs (seeded)", isequal(jacobian_run1.J, jacobian_run2.J))

println("\n(2) Structural zeros")
block_of_moment = zeros(Int, sum(MOMENT_MASK))
for (b, rg) in enumerate(BLOCK_RANGES); block_of_moment[rg] .= b; end
columns = sens_columns(theta_vec)
N_columns = findall(c -> c.kind === :N, columns)
aggregate_columns = findall(c -> c.kind === :aggregate, columns)
outside_blocks_4_6 = findall(b -> !(b in (4, 6)), block_of_moment)
in_blocks_4_6  = findall(b -> b in (4, 6), block_of_moment)
check("N_s columns exactly zero outside blocks 4 and 6", all(jacobian_run1.J_est[outside_blocks_4_6, N_columns] .== 0))
check("N_s columns move Ḡ_s(0)", all(any(jacobian_run1.J_est[collect(BLOCK_RANGES[6]), j] .!= 0) for j in N_columns))
check("aggregate columns exactly zero on blocks 4 and 6", all(jacobian_run1.J_est[in_blocks_4_6, aggregate_columns] .== 0))
check("every column finite on the estimation draws", all(isfinite, jacobian_run1.J_est))

println("\n(3) Profiles")
profile_dir = mktempdir()
sensitivity_profiles(theta_vec; K = 1, G = 5, dest = profile_dir)
base_moments, _ = sens_moments(theta_vec, sens_drawset(0))
for line in readlines(joinpath(profile_dir, "profiles_index.txt"))
    label, file_name = split(line, '\t')
    profile_file = NPZ.npzread(joinpath(profile_dir, file_name))
    g = profile_file["grid"]; estimate_position = findfirst(==(profile_file["estimate"][1]), g)
    gap = maximum(abs.(profile_file["m_est"][estimate_position, :] .- base_moments))
    check("$label: the point at the estimate reproduces the base moments (gap $(round(gap, sigdigits = 2)))",
          gap < 1e-6)
    if startswith(label, "N_s")
        t = Int(profile_file["targets"][1])
        check("$label: Ḡ_s(0) strictly decreasing in N_s", all(diff(profile_file["m_est"][:, t]) .< 0))
    end
end

println("\n", pass ? "ALL SENSITIVITY GATES PASS" : "SOME SENSITIVITY GATES FAILED")
columns = sens_columns(theta_vec)
for k in (:aggregate, :alpha, :kappa, :T, :N)
    j = findall(c -> c.kind === k, columns)
    @printf("  %-9s max |ΔJ| between runs: estimation draws %.3g, mean %.3g\n", k,
            maximum(abs.(jacobian_run1.J_est[:, j] .- jacobian_run2.J_est[:, j])), maximum(abs.(jacobian_run1.J[:, j] .- jacobian_run2.J[:, j])))
end
pass || exit(1)
