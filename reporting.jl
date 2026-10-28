##### reporting.jl — stage reports, run summaries, granular diagnostics, post-hoc economy #####

using Printf
using Dates

##################### Stage report (dashboard + report.txt) ###################

"""
    bubble_scatter(x, y; xlabel, ylabel, title) -> plot

Empirical (x) against simulated (y), dot size ∝ x, 45° line, and the WLS slope through
the origin (weights x) with its t-statistic.
"""
function bubble_scatter(x::AbstractVector, y::AbstractVector; xlabel="Empirical",
                        ylabel="Simulated", title="", x_ref=Float64[], y_ref=Float64[])
    keep = x .> 0
    xf = Float64.(x[keep]); yf = Float64.(y[keep])
    b  = sum(xf .* xf .* yf) / sum(xf .* xf .^ 2)
    s2 = sum(xf .* (yf .- b .* xf) .^ 2) / max(length(xf) - 1, 1)
    se = sqrt(s2 / sum(xf .* xf .^ 2))
    lo = min(minimum(vcat(xf, x_ref)), minimum(vcat(yf, y_ref))) * 0.9
    hi = max(maximum(vcat(xf, x_ref)), maximum(vcat(yf, y_ref))) * 1.1
    p = scatter(xf, yf; markersize = sqrt.(300 .* xf ./ maximum(xf)) ./ 2, alpha = 0.6,
                markerstrokecolor = :black, markerstrokewidth = 0.5,
                color = RGB(0.247, 0.404, 0.667), label = isempty(x_ref) ? "" : "Non-reference",
                xlabel = xlabel, ylabel = ylabel, title = title, xlims = (lo, hi), ylims = (lo, hi),
                legend = isempty(x_ref) ? false : :bottomright, grid = true, gridstyle = :dash)
    isempty(x_ref) || scatter!(p, x_ref, y_ref; markershape = :diamond, markersize = 5, alpha = 0.7,
                               color = RGB(0.75, 0.30, 0.20), label = "Reference (reconstructed)")
    plot!(p, [lo, hi], [lo, hi]; color = :black, label = "45°", linewidth = 1)
    annotate!(p, hi * 0.95, lo + (hi - lo) * 0.12, text(@sprintf("Coef: %.3f", b), :right, 8))
    annotate!(p, hi * 0.95, lo + (hi - lo) * 0.04, text(@sprintf("t-stat: %.1f", se > 0 ? b / se : Inf), :right, 8))
    return p
end

"""
    plot_T_vs_initial(params, folder)

Area-level T at `params` against its starting value `T_rs_init` (both ref-normalised),
with the search-window lines ×BOUND_LO / ×BOUND_HI. Never blocks estimation.
"""
function plot_T_vs_initial(params, folder)
    try
        T_best = unpack_T_par(params)
        α = round(params[alpha_range()][1], digits = 3)
        idx = [(s, c) for s in 1:S for c in 1:T_COL_DIM if T_rs_init[s, c] > 1e-8 && T_best[s, c] > 1e-8]
        isempty(idx) && return
        xs = [T_rs_init[s, c] for (s, c) in idx]; ys = [T_best[s, c] for (s, c) in idx]
        lo, hi = min(minimum(xs), minimum(ys)), max(maximum(xs), maximum(ys))
        p = scatter(xs, ys; xscale = :log10, yscale = :log10, label = "",
                    xlabel = "initial T / T_ref (α₀ = $(round(P_alpha, digits=3)))",
                    ylabel = "T / T_ref (α = $α)", title = "T: initial vs current")
        plot!(p, [lo, hi], [lo, hi]; color = :black, ls = :dash, label = "y = x")
        plot!(p, [lo, hi], BOUND_HI .* [lo, hi]; color = :firebrick, ls = :dot, label = "×$BOUND_LO / ×$BOUND_HI")
        plot!(p, [lo, hi], BOUND_LO .* [lo, hi]; color = :firebrick, ls = :dot, label = "")
        savefig(p, joinpath(folder, "T_best_vs_initial_alpha$(α).png"))
        NPZ.npzwrite(joinpath(folder, "T_best_vs_initial_alpha$(α).npz"),
                     Dict("T_initial" => xs, "T_best" => ys,
                          "sector" => Float64.(first.(idx)), "region" => Float64.(last.(idx))))
    catch e
        @warn "plot_T_vs_initial skipped: $e"
    end
end

_quartiles(v) = (nz = filter(!=(0), vec(v));
                 (n_zeros = count(==(0), vec(v)), q1 = quantile(nz, 0.25), median = quantile(nz, 0.5),
                  q3 = quantile(nz, 0.75), max_val = maximum(nz)))

"""
    generate_report(folder, params; variable="", radius="")

Writes `dashboard.png` (γ_{s,a} with the reconstructed reference areas, π_r, sector
shares), the T plot, `pi_r.npy`, `productivity.npy` and `report.txt` (fit of every
block, the profiled N̂_s with its clamps, block 4 against the data).
"""
function generate_report(folder::String, params::Vector{Float64}; variable::String = "", radius = "")
    mkpath(folder)
    loss, sim = full_SMM(params)
    sim_vec = moments_to_vec(sim)
    emp_vec = vec(empirical_moments)
    rγ = collect(BLOCK_RANGES[5])
    emp_g, sim_g = emp_vec[rγ], sim_vec[rγ]
    x_ref = Float64[]; y_ref = Float64[]
    for e in GAMMA_REF_MAP
        e.emp_ref > 0 || continue
        push!(x_ref, e.emp_ref); push!(y_ref, e.c_s - sum(sim_g[e.local_positions]))
    end
    p1 = bubble_scatter(emp_g, sim_g; xlabel = "Empirical γ", ylabel = "Simulated γ",
                        title = "γ_{s,a}: Empirical vs Simulated", x_ref = x_ref, y_ref = y_ref)
    p2 = bubble_scatter(emp_pi_r, sim[3]; xlabel = "Empirical π_r", ylabel = "Simulated π_r",
                        title = "π_r: Empirical vs Simulated")
    p3 = bubble_scatter(agg_industry_share, sim[2]; xlabel = "Empirical π_s", ylabel = "Simulated π_s",
                        title = "π_s: Empirical vs Simulated")
    plot(p1, p2, p3, layout = (1, 3), size = (1500, 500), margin = 5Plots.mm)
    savefig(joinpath(folder, "dashboard.png"))
    plot_T_vs_initial(params, folder)
    npzwrite(joinpath(folder, "pi_r.npy"), sim[3])
    npzwrite(joinpath(folder, "productivity.npy"), unpack_params(params)[3])

    gr = try granular_report(params) catch e; @warn "granular section skipped: $e"; nothing end
    open(joinpath(folder, "report.txt"), "w") do io
        println(io, "Date: $(Dates.format(now(), "yyyy-mm-dd HH:MM"))")
        println(io, "Optimisation: $(isempty(variable) ? "All" : variable)   radius: $radius")
        println(io, "Score: $(loss[1])\n")
        @printf(io, ">> Labour share   Empirical: %8.4f | Simulated: %8.4f\n", agg_labor_share, sim[1][1])
        qe, qs = _quartiles(emp_g), _quartiles(sim_g)
        println(io, "\n>> γ_{s,a} (quartiles of non-zero entries):")
        for f in (:n_zeros, :q1, :median, :q3, :max_val)
            @printf(io, "%-10s Empirical: %8.3f | Simulated: %8.3f\n", f, getfield(qe, f), getfield(qs, f))
        end
        qe, qs = _quartiles(emp_pi_r), _quartiles(sim[3])
        println(io, "\n>> π_r:")
        for f in (:q1, :median, :q3, :max_val)
            @printf(io, "%-10s Empirical: %8.3f | Simulated: %8.3f\n", f, getfield(qe, f), getfield(qs, f))
        end
        println(io, "\n>> Sector input shares:")
        for s in 1:S
            @printf(io, "%-10s Empirical: %8.3f | Simulated: %8.3f\n", SECTOR_NAMES[s], agg_industry_share[s], sim[2][s])
        end
        println(io, "\n>> Block 4 — cell-level cloglog:")
        for (i, lab) in enumerate(REG_COEF_LABELS)
            @printf(io, "%-10s Empirical: %8.3f | Simulated: %8.3f\n", lab, reg_coef[i], sim[4][i])
        end
        println(io, "\n>> Count moment Ḡ_s(0) and the profiled variety count:")
        @printf(io, "%-10s %9s %9s %6s %6s %6s %6s %9s\n", "sector", "G0 data", "G0 model",
                "N_LO", "N̂_s", "N_HI", "clamp", "N_count")
        for s in 1:S
            @printf(io, "%-10s %9.4f %9.4f %6d %6s %6d %6s %9s\n", SECTOR_NAMES[s], G_TARGET[s], sim[6][s],
                    N_LO[s], gr === nothing ? "-" : string(gr.N_hat[s]), N_HI[s],
                    gr === nothing ? "-" : string(gr.clamped[s]),
                    gr === nothing ? "-" : @sprintf("%.1f", gr.N_count[s]))
        end
        println(io, "\n  clamp :lo/:hi = the variety count sits on a bound (a rejection signal, not a nuisance).")
        println(io, "  N_count = N_supplier_s / Σ_l q̂_ls is the independent second route to N_s.")
    end
end

##################### Run summaries ###################

_numbered_dirs(folder) = sort([(n, joinpath(folder, d)) for d in readdir(folder)
                               for n in (tryparse(Int, d),) if n !== nothing &&
                               isfile(joinpath(folder, d, "best_params.npy"))], by = first)
_epoch_dirs(folder) = sort([(n, joinpath(folder, d)) for d in readdir(folder) if startswith(d, "epoch_")
                            for n in (tryparse(Int, replace(d, "epoch_" => "")),) if n !== nothing], by = first)

"""
    all_stage_folders(step_folder, max_loop=nothing) -> stage "0" then every epoch's stages, in order
"""
function all_stage_folders(step_folder::String, max_loop = nothing)
    out = String[]
    isfile(joinpath(step_folder, "0", "best_params.npy")) && push!(out, joinpath(step_folder, "0"))
    for (n, ep) in _epoch_dirs(step_folder)
        (max_loop === nothing || n <= max_loop) || continue
        append!(out, last.(_numbered_dirs(ep)))
    end
    return out
end

"""
    find_last_stage_folder(step_folder) -> the last stage of the last epoch (or "0")
"""
function find_last_stage_folder(step_folder::String)
    folders = all_stage_folders(step_folder)
    isempty(folders) && error("No stage folders found in $step_folder")
    return folders[end]
end

"""
    run_reporting(step_folder, max_loop=nothing)

Evaluates every stage of a step at its own parameters and writes
`best_simulated_moments.npy` (unmasked blocks 1–6 stacked, one column per stage),
`best_parameters_list.npy`, `empirical_moments.npy` and `score_history.npy` (loss and
block-4 RMSE as % of the first stage).
"""
function run_reporting(step_folder::String, max_loop = nothing)
    folders = all_stage_folders(step_folder, max_loop)
    isempty(folders) && (println("No stage folders in $step_folder"); return nothing)
    params = [vec(NPZ.npzread(joinpath(f, "best_params.npy"))[:, 1]) for f in folders]
    results = pmap(p -> safe_SMM(p), params)
    scores  = [r === nothing ? Inf : r[1][1] for r in results]
    dists   = [r === nothing ? Inf : sqrt(mean((r[2][4] .- reg_coef) .^ 2)) for r in results]
    moments = [vcat([vec(b) for b in r[2]]...) for r in results if r !== nothing]
    isempty(moments) || npzwrite(joinpath(step_folder, "best_simulated_moments.npy"), hcat(moments...))
    npzwrite(joinpath(step_folder, "best_parameters_list.npy"), hcat(params...))
    npzwrite(joinpath(step_folder, "empirical_moments.npy"), empirical_moments)
    rel(v) = v[1] > 0 ? v ./ v[1] .* 100 : v
    npzwrite(joinpath(step_folder, "score_history.npy"),
             Dict("first_stage" => rel(scores), "min_dist_first" => rel(dists)))
    return nothing
end

##################### Granular diagnostics ###################

"""
    report_granular(theta, folder; industry="", label="")

Prints and writes (`granular_diagnostics.{npz,txt}`) N̂_s with its clamp, Ḡ_s(0)
fitted vs targeted, `N^count_s`, and block 4 as data / model `E[β̂(y)]` / sd across
networks / `β̂(E[y])`. Reporting only; never blocks estimation.
"""
function report_granular(theta::Vector{Float64}, folder::String; industry::String = "", label::String = "")
    try
        info = granular_report(theta)
        mkpath(folder)
        lines = String[]
        push!(lines, "GRANULAR DIAGNOSTICS $(industry) $(label)")
        push!(lines, @sprintf("  %-8s %6s %6s %6s %6s %9s %9s %9s", "sector", "N_LO", "N̂_s", "N_HI",
                              "clamp", "G0_fit", "G0_tgt", "N_count"))
        for s in 1:S
            push!(lines, @sprintf("  %-8s %6d %6d %6d %6s %9.4f %9.4f %9.1f", SECTOR_NAMES[s], N_LO[s],
                                  info.N_hat[s], N_HI[s], string(info.clamped[s]), info.G0[s],
                                  G_TARGET[s], info.N_count[s]))
        end
        push!(lines, "  Block 4 (R = $R_CELL_REG networks, $(info.reg_fail) failed fits)")
        push!(lines, @sprintf("  %-10s %10s %12s %10s %12s", "coef", "data", "E[β̂(y)]", "sd_j", "β̂(E[y])"))
        for (i, lab) in enumerate(REG_COEF_LABELS)
            push!(lines, @sprintf("  %-10s %10.4f %12.4f %10.4f %12.4f", lab, reg_coef[i],
                                  info.reg_coef[i], info.reg_sd[i], info.beta_expected[i]))
        end
        println("\n", join(lines, "\n"))
        write(joinpath(folder, "granular_diagnostics.txt"), join(lines, "\n") * "\n")
        NPZ.npzwrite(joinpath(folder, "granular_diagnostics.npz"),
            Dict("N_hat" => Float64.(info.N_hat), "N_LO" => Float64.(N_LO), "N_HI" => Float64.(N_HI),
                 "clamped" => [c == :none ? 0.0 : (c == :lo ? -1.0 : 1.0) for c in info.clamped],
                 "G0_fit" => Float64.(info.G0), "G0_target" => Float64.(collect(G_TARGET)),
                 "N_count" => Float64.(info.N_count), "q_hat" => Float64.(info.q_hat),
                 "reg_coef_model" => Float64.(info.reg_coef), "reg_coef_sd_reps" => Float64.(info.reg_sd),
                 "reg_coef_expected" => Float64.(info.beta_expected),
                 "reg_fail" => [Float64(info.reg_fail)], "EK" => Float64.(info.EK)))
        return info
    catch e
        @warn "granular diagnostics skipped: $e"
        return nothing
    end
end

##################### Post-hoc firm-level economy ###################

const POST_HOC_REPLICATIONS = 50

_empty_firm_cols() = (sirens = Int[], sectors = Int[], ze2010 = Int[], ze2010_downstream = Int[],
                      share = Float64[], downstream_purchase = Float64[],
                      intermediate_derivative = Float64[], productivity = Float64[],
                      sample_weight = Float64[], variety = Int[], replication = Int[])

_firm_cols_frame(c) = DataFrame(SIREN = c.sirens, A129 = c.sectors, ze2010 = c.ze2010,
                                ze2010_downstream = c.ze2010_downstream, share = c.share,
                                downstream_purchase = c.downstream_purchase,
                                intermediate_derivative = c.intermediate_derivative,
                                productivity = c.productivity, sample_weight = c.sample_weight,
                                variety = c.variety, replication = c.replication)

"""
    _firm_level_rows!(cols, network, siren_map, rep)

One row per realised (variety, cell, sector, buyer) linkage; zero-weight rows (varieties
past N̂_s) are dropped. A firm is keyed on (replication, cell, sector, variety).
"""
function _firm_level_rows!(cols, network, siren_map::Dict{NTuple{4,Int},Int}, rep::Int)
    for i in eachindex(network.firm_exp_rho)
        val = network.firm_exp_val[i]
        val == 0.0 && continue
        rho = network.firm_exp_rho[i]; s = network.firm_exp_s[i]
        g = network.firm_exp_g[i]; l = GOOD_R[g]; r = network.firm_exp_r[i]
        push!(cols.sirens, get!(siren_map, (rep, l, s, rho), length(siren_map) + 1))
        push!(cols.sectors, s); push!(cols.ze2010, l); push!(cols.ze2010_downstream, r)
        push!(cols.share, val); push!(cols.downstream_purchase, network.Y_r[r] * network.mu)
        push!(cols.intermediate_derivative, network.firm_deriv_val[i])
        push!(cols.productivity, network.z_flat[rho, g])
        push!(cols.sample_weight, network.sample_weights[rho, g])
        push!(cols.variety, rho); push!(cols.replication, rep)
    end
end

"""
    write_post_hoc(params, dest; n_rep=POST_HOC_REPLICATIONS, base_seed=8_100_000)

The economy the Python reporting checks against (`check_against_julia`):
  * `suppliers.parquet` — `n_rep` independent realisations (i.i.d. draws) of the
    N̂_s-variety economy (weights 1/N̂_s on rows ρ ≤ N̂_s, 0 beyond), with `replication`
    and `variety` columns; `post_hoc_u.npy` the draws (N_max × n_good × n_rep),
    `post_hoc_good_sr.npy` the (sector, cell) of each draw column, `post_hoc_N_hat.npy`.
  * `suppliers_continuum.parquet` — the solve on the estimation draws (N_rho per cell,
    the N_s → ∞ benchmark); `w_srd_r.npy` and `suppliers.npy` from that solve.
"""
function write_post_hoc(params::Vector{Float64}, dest::AbstractString;
                        n_rep::Int = POST_HOC_REPLICATIONS, base_seed::Int = 8_100_000)
    mkpath(dest)
    network = solve_network(params; u_draws = U_DRAWS, sample_weights = SAMPLE_WEIGHTS,
                            return_firm_level = true)
    X_lrs = zeros(R, R, S)
    for i in eachindex(network.firm_exp_rho)
        X_lrs[GOOD_R[network.firm_exp_g[i]], network.firm_exp_r[i], network.firm_exp_s[i]] +=
            network.firm_exp_val[i] * network.mu * network.Y_r[network.firm_exp_r[i]]
    end
    w_srd_r = zeros(S, R, R)
    for s in 1:S, l in 1:R
        tot = sum(X_lrs[l, :, s])
        tot > 1e-10 && (w_srd_r[s, l, :] .= X_lrs[l, :, s] ./ tot)
    end
    npzwrite(joinpath(dest, "w_srd_r.npy"), w_srd_r)
    m = size(network.linkages_flat, 1)
    suppliers = zeros(Bool, S, m, R)
    for g in 1:n_good, rho in 1:m
        network.linkages_flat[rho, g] && (suppliers[GOOD_S[g], rho, GOOD_R[g]] = true)
    end
    npzwrite(joinpath(dest, "suppliers.npy"), suppliers)
    ref_cols = _empty_firm_cols()
    _firm_level_rows!(ref_cols, network, Dict{NTuple{4,Int},Int}(), 0)
    Parquet.write_parquet(joinpath(dest, "suppliers_continuum.parquet"), _firm_cols_frame(ref_cols))

    k_counts = vec(sum(network.linkages_flat; dims = 1))
    N_hat, clamped = concentrate_N_s(k_counts, m)
    N_max = maximum(N_hat)
    npzwrite(joinpath(dest, "post_hoc_N_hat.npy"), N_hat)
    W_gran = zeros(N_max, n_good)
    for g in 1:n_good
        W_gran[1:N_hat[GOOD_S[g]], g] .= 1.0 / N_hat[GOOD_S[g]]
    end
    cols = _empty_firm_cols(); siren_map = Dict{NTuple{4,Int},Int}()
    u_stack = zeros(N_max, n_good, n_rep)
    for b in 1:n_rep
        u = generate_draws(N_max, n_good, :mc; randomise = true, rng = MersenneTwister(base_seed + b))[1]
        for g in 1:n_good, rho in (N_hat[GOOD_S[g]] + 1):N_max
            u[rho, g] = 0.5                # zero-weight rows: keep every price finite
        end
        u_stack[:, :, b] .= u
        net_b = solve_network(params; u_draws = u, sample_weights = W_gran, return_firm_level = true)
        _firm_level_rows!(cols, net_b, siren_map, b)
    end
    npzwrite(joinpath(dest, "post_hoc_u.npy"), u_stack)
    npzwrite(joinpath(dest, "post_hoc_good_sr.npy"), hcat(GOOD_S, GOOD_R))
    Parquet.write_parquet(joinpath(dest, "suppliers.parquet"), _firm_cols_frame(cols))
    @printf("  post-hoc -> %s: N̂_s %s (%d clamped), %d linkages over %d realisations; mean D_r %.4f (granular) vs %.4f (N_s → ∞)\n",
            dest, string(N_hat), count(!=(:none), clamped),
            length(cols.share), n_rep, 1 + sum(cols.share) / (R_downstream * n_rep),
            1 + sum(ref_cols.share) / R_downstream)
end
