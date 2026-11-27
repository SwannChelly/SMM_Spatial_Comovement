##### compare_import_access.jl #####
# Before/after table for the buyers' access to foreign suppliers ψ_r. Run from the
# directory holding the baseline and the two run trees:
#
#     julia SMM_Spatial_Comovement/test/compare_import_access.jl <industry> <before_dir> <after_dir> [n_coef] [n_tau]
#
# <before_dir> is a run without ψ (its parameter vector is R_d entries shorter): it is read
# with ψ ≡ 1, which is that model exactly (the code with ψ held at one reproduces the code
# without ψ bit for bit). Both estimates are read from `step3/theta_hat_2.npy` (ENV STEP=step1
# reads `step1/theta_hat_1.npy`). Prints and writes `<after_dir>/import_access_comparison.txt`:
# α, κ, the within-sector sd of log T, N̂_s, log T and its rank by area, and by buyer the import
# share f_r, the amplification D_r and the local share L_r(200); every rank of T and every sign
# of a counterfactual change that differs between the two is flagged.

using Distributed
using Random, NPZ, DataFrames, CSV, Distributions, Statistics, QuasiMonteCarlo,
      LinearAlgebra, Printf, SpecialFunctions, Optim
const REPO = normpath(joinpath(@__DIR__, ".."))
include(joinpath(REPO, "model_CP.jl")); include(joinpath(REPO, "profiling.jl")); include(joinpath(REPO, "inference.jl"))

industry        = ARGS[1]
before_dir, after_dir = ARGS[2], ARGS[3]
n_coef          = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 4
n_tau           = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 1
n_rho_inference = 1000
relax_n_lo      = false
input_folder    = get(ENV, "INPUT_FOLDER", "./baseline_$industry")
output_folder   = mktempdir()
include(joinpath(REPO, "load_parameters.jl"))

estimate_file = get(ENV, "STEP", "step3") == "step1" ? joinpath("step1", "theta_hat_1.npy") : joinpath("step3", "theta_hat_2.npy")
function read_estimate(dir)
    stored = NPZ.npzread(joinpath(dir, estimate_file))
    theta_vec = Float64.(ndims(stored) > 1 ? stored[:, 1] : stored)
    length(theta_vec) == length(PARAM_LABELS) - R_downstream && return vcat(theta_vec, ones(R_downstream))
    length(theta_vec) == length(PARAM_LABELS) || error("$dir: $(length(theta_vec)) parameters, expected $(length(PARAM_LABELS))")
    return theta_vec
end
estimates = (before = read_estimate(before_dir), after = read_estimate(after_dir))

# Every statistic of one estimate.
function summary_of(theta_vec)
    T_area = unpack_T_area(theta_vec)
    log_T = [T_area[s, a] > 0 ? log(T_area[s, a]) : NaN for s in 1:S, a in 1:T_COL_DIM]
    rank_T = zeros(Int, S, T_COL_DIM)
    for s in 1:S
        columns = SECTOR_DOM_COLS[s]
        order = sortperm(log_T[s, columns]; rev = true)
        rank_T[s, columns[order]] .= 1:length(columns)
    end
    stats = Dict(regime => incidence_stats(theta_vec; regime = regime) for regime in CF_REGIMES)
    fit = buyer_import_fit(theta_vec)
    return (alpha = theta_vec[alpha_range()], kappa = unpack_kappa(theta_vec), log_T = log_T, rank_T = rank_T,
            sd_log_T = [std(filter(isfinite, log_T[s, SECTOR_DOM_COLS[s]])) for s in 1:S],
            N_hat = granular_report(theta_vec).N_hat, import_share = fit.model, psi = fit.psi,
            amplification = fit.amplification, local_200 = stats[:estimate].local_share_by_buyer[2],
            mean_dist = stats[:estimate].mean_dist_by_buyer, counterfactual_row = counterfactual_row(theta_vec))
end
before, after = summary_of(estimates.before), summary_of(estimates.after)
flag(changed) = changed ? "  <-- CHANGES" : ""

io = IOBuffer()
println(io, "="^88, "\nBEFORE (ψ ≡ 1) / AFTER (ψ_r on the customs shares) — $industry, $estimate_file\n", "="^88)
@printf(io, "  %-22s %12s %12s\n", "", "before", "after")
for k in 1:N_TAU
    @printf(io, "  %-22s %12.4f %12.4f\n", N_TAU == 1 ? "α" : "α_$k", before.alpha[k], after.alpha[k])
end
@printf(io, "  %-22s %12.4f %12.4f\n", "κ", before.kappa, after.kappa)
for s in 1:S
    @printf(io, "  %-22s %12.4f %12.4f\n", "sd log T[$(SECTOR_NAMES[s])]", before.sd_log_T[s], after.sd_log_T[s])
end
for s in 1:S
    @printf(io, "  %-22s %12d %12d\n", "N̂_s[$(SECTOR_NAMES[s])]", before.N_hat[s], after.N_hat[s])
end

println(io, "\n--- log T by area (rank within sector, 1 = strongest comparative advantage) ---")
@printf(io, "  %-10s %-12s %10s %5s %10s %5s\n", "sector", "area", "before", "rank", "after", "rank")
area_names = vcat(["AA" * BUYER_NAMES[a] for a in 1:n_AA], "Foreign")
for s in 1:S, a in SECTOR_DOM_COLS[s]
    @printf(io, "  %-10s %-12s %10.4f %5d %10.4f %5d%s\n", SECTOR_NAMES[s], area_names[a], before.log_T[s, a],
            before.rank_T[s, a], after.log_T[s, a], after.rank_T[s, a], flag(before.rank_T[s, a] != after.rank_T[s, a]))
end

println(io, "\n--- by buyer: import share f_r, ψ_r, amplification D_r, local share L_r(200), mean sourcing distance ---")
@printf(io, "  %-8s %8s %8s %8s %8s %8s %8s %8s %8s %8s\n", "buyer", "f before", "f after", "ψ_r",
        "D before", "D after", "L before", "L after", "km bef.", "km aft.")
for dr in 1:R_downstream
    @printf(io, "  %-8s %8.4f %8.4f %8.4g %8.4f %8.4f %8.4f %8.4f %8.1f %8.1f\n", BUYER_NAMES[dr],
            before.import_share[dr], after.import_share[dr], after.psi[dr], before.amplification[dr], after.amplification[dr],
            before.local_200[dr], after.local_200[dr], before.mean_dist[dr], after.mean_dist[dr])
end
rank_by(values) = invperm(sortperm(values; rev = true))
for (name, b, a) in (("D_r", before.amplification, after.amplification), ("L_r(200)", before.local_200, after.local_200))
    changed = rank_by(b) != rank_by(a)
    @printf(io, "  ranking of buyers by %s: %s%s\n", name, changed ? "differs" : "unchanged", flag(changed))
end
@printf(io, "  correlation of D_r with f_r across buyers: before %.3f, after %.3f\n",
        cor(before.amplification, before.import_share), cor(after.amplification, after.import_share))

println(io, "\n--- counterfactuals (closed form): levels, and sign of the change against the estimate ---")
labels = counterfactual_labels(); n_stats = length(CF_STATS)
for (i, label) in enumerate(labels)
    change_flag = ""
    if i > n_stats
        base = (i - 1) % n_stats + 1
        change_before = before.counterfactual_row[i] - before.counterfactual_row[base]
        change_after  = after.counterfactual_row[i] - after.counterfactual_row[base]
        change_flag = sign(change_before) != sign(change_after) ? "  <-- SIGN OF THE CHANGE FLIPS" : ""
    end
    @printf(io, "  %-34s %10.4f %10.4f%s\n", label, before.counterfactual_row[i], after.counterfactual_row[i], change_flag)
end
report = String(take!(io))
print(report)
write(joinpath(after_dir, "import_access_comparison.txt"), report)
