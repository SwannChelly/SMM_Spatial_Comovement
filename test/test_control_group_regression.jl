##### test_control_group_regression.jl #####
# Compare the extensive-margin distance regression WITH vs WITHOUT the no-supplier
# control group (filter_N_upstream status == 2), at a FIXED parameter vector.
# STANDALONE + PRINT-ONLY: includes the model files and load_parameters.jl exactly
# like main.jl / test_extensive_margin.jl, then evaluates fast_weighted_regression
# twice on the same solved network and contrasts the two coefficient vectors.
#
#     julia test/test_control_group_regression.jl auto 4 1
#     julia test/test_control_group_regression.jl auto 4 1 ./reporting_auto_profiled_pso
#
# Positional args: industry [n_coef] [n_tau] [run_folder]
#   run_folder is the estimation output tree to load θ̂ from (step3/theta_hat_2.npy,
#   else step1/best_params.npy). Defaults to the profiled-PSO run the extensive-margin
#   work targets, with a couple of fallbacks.
#
# BOTH regimes drop the log-z size/productivity control (the user's spec): the only
# difference is whether the control-group y=0 distance observations are appended.
#   • WITH control    → fast_weighted_regression(...; include_control=true)   [production]
#   • WITHOUT control  → fast_weighted_regression(...; include_control=false) [supplier pairs only]
# The gap between the two coefficient vectors is exactly the effect of pinning the
# extensive-margin zeros exogenously via the control group.

using Distributed
@everywhere using Random
@everywhere using NPZ
using LinearAlgebra, Statistics, Printf

@everywhere include("../model_CP.jl")
@everywhere include("../tools.jl")
@everywhere include("../model_analytical.jl")
@everywhere include("../optimizers/pso_integration.jl")

industry = length(ARGS) >= 1 ? ARGS[1] : "auto"
n_coef   = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4
n_tau    = length(ARGS) >= 3 && !isempty(strip(ARGS[3])) ? parse(Int, ARGS[3]) : n_coef
K_sim    = 10000   # unused here; load_parameters.jl may reference it

# Candidate run folders to load θ̂ from (first that has a usable file wins).
run_folder_arg = length(ARGS) >= 4 && !isempty(strip(ARGS[4])) ? strip(ARGS[4]) : nothing
run_candidates = run_folder_arg !== nothing ? [run_folder_arg] :
    ["./reporting_$(industry)_profiled_pso", "./reporting_$(industry)_pso",
     "./reporting_$(industry)_profiled", "./reporting_$(industry)"]

input_folder  = "./baseline_$industry"
# Scratch output for load_parameters diagnostics — do NOT clobber a real run tree.
output_folder = "./reporting_$(industry)_control_test"
mkpath(output_folder)

include("../load_parameters.jl")

println("\n" * "="^78)
println("Control-group regression contrast  (industry=$industry  N_REG=$N_REG  N_TAU=$N_TAU)")
println("  n_good (supplier pairs) = $n_good   |   N_CONTROL (filter==2 pairs) = $N_CONTROL")
println("="^78)

# ── Load θ̂ from an estimation run (full-length level θ) ──────────────────────
function load_theta()
    expected = 1 + S + R_downstream + N_TAU + count(T_MASK)
    for folder in run_candidates
        for rel in (("step3", "theta_hat_2.npy"), ("step1", "theta_hat_1.npy"))
            p = joinpath(folder, rel...)
            isfile(p) || continue
            θ = NPZ.npzread(p); ndims(θ) > 1 && (θ = θ[:, 1])
            if length(θ) == expected
                @printf("Loaded θ̂ from %s (length %d)\n", p, length(θ)); return θ
            else
                @printf("Skipping %s: length %d ≠ expected %d (different config)\n",
                        p, length(θ), expected)
            end
        end
        # step1 best_params lives under a per-stage subfolder
        s1 = joinpath(folder, "step1")
        if isdir(s1)
            for (root, _, files) in walkdir(s1), f in files
                f == "best_params.npy" || continue
                p = joinpath(root, f)
                θ = NPZ.npzread(p); ndims(θ) > 1 && (θ = θ[:, 1])
                if length(θ) == expected
                    @printf("Loaded θ̂ from %s (length %d)\n", p, length(θ)); return θ
                end
            end
        end
    end
    error("No usable θ̂ found in $(run_candidates) (expected length $(expected) = " *
          "1+S+R_downstream+N_TAU+count(T_MASK)). Pass the run folder as the 4th arg, " *
          "or run an estimation for this (industry, N_REG, N_TAU) config first.")
end

# NOTE: `theta` is a reserved global const (the Fréchet dispersion in load_parameters.jl),
# so the loaded estimate must use a different name.
theta_hat = load_theta()

# ── Solve the network once at θ̂ (shared linkages / draws for both regressions) ─
net = solve_network(theta_hat; u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS)
n_links = sum(net.linkages_flat .> 0)
@printf("\nNetwork solved: %d firm-level linkages over %d supplier pairs × %d draws.\n",
        n_links, n_good, size(net.linkages_flat, 1))

# ── The two regressions (log-z control dropped in both; only control group toggles) ─
reg_with    = fast_weighted_regression(net.linkages_flat, net.z_flat, net.sample_weights;
                                       include_control=true)
reg_without = fast_weighted_regression(net.linkages_flat, net.z_flat, net.sample_weights;
                                       include_control=false)

# ── Distance-bin composition: how many supplier vs control pairs land in each bin ─
# (N_REG > 1 only; for N_REG == 1 the single regressor is continuous log-distance.)
# `let` gives the loops a local scope so the counters aren't top-level soft-scope globals.
if N_REG > 1
    let sup_per_bin = zeros(Int, N_REG), ctrl_per_bin = zeros(Int, N_REG),
        sup_base = 0, ctrl_base = 0        # base (omitted) category, DistBin == 0
        for g in 1:n_good
            b = DistBin[GOOD_R[g], CLOSEST_DOWNSTREAM_REGION[GOOD_R[g]]]
            (b >= 1 && b <= N_REG) ? (sup_per_bin[b] += 1) : (sup_base += 1)
        end
        for c in 1:N_CONTROL
            b = DistBin[CONTROL_R[c], CLOSEST_DOWNSTREAM_REGION[CONTROL_R[c]]]
            (b >= 1 && b <= N_REG) ? (ctrl_per_bin[b] += 1) : (ctrl_base += 1)
        end
        println("\nDistance-bin composition (pair counts; 'base' = omitted category, DistBin==0):")
        @printf("  %-8s %12s %12s\n", "bin", "supplier", "control")
        @printf("  %-8s %12d %12d\n", "base", sup_base, ctrl_base)
        for b in 1:N_REG
            @printf("  %-8d %12d %12d\n", b, sup_per_bin[b], ctrl_per_bin[b])
        end
    end
end

# ── Coefficient contrast ─────────────────────────────────────────────────────
emp = reg_coef            # empirical target (may reflect a different data spec)
println("\nExtensive-margin distance coefficients (both drop the log-z control):")
@printf("  %-8s %14s %14s %14s %14s\n",
        "bin", "with control", "without", "Δ(with−wo)", "empirical")
for b in 1:N_REG
    e = b <= length(emp) ? @sprintf("%14.6f", emp[b]) : @sprintf("%14s", "-")
    @printf("  %-8d %14.6f %14.6f %14.6f %s\n",
            b, reg_with[b], reg_without[b], reg_with[b] - reg_without[b], e)
end

Δ = reg_with .- reg_without
@printf("\nSummary: max|Δ| = %.6g   mean|Δ| = %.6g   ‖Δ‖₂ = %.6g\n",
        maximum(abs.(Δ)), sum(abs.(Δ)) / length(Δ), norm(Δ))
den = norm(reg_without) > 0 ? norm(reg_without) : 1.0
@printf("Relative shift ‖Δ‖₂ / ‖without‖₂ = %.4g\n", norm(Δ) / den)
println("\n(Positive Δ ⇒ the control-group zeros make the coefficient larger/less negative;")
println(" negative Δ ⇒ they steepen the distance decay. reg_with is the production moment.)")
