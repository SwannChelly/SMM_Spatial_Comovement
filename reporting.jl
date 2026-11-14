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
    x_kept = Float64.(x[keep]); y_kept = Float64.(y[keep])
    slope  = sum(x_kept .* x_kept .* y_kept) / sum(x_kept .* x_kept .^ 2)
    residual_variance = sum(x_kept .* (y_kept .- slope .* x_kept) .^ 2) / max(length(x_kept) - 1, 1)
    slope_se = sqrt(residual_variance / sum(x_kept .* x_kept .^ 2))
    axis_lo = min(minimum(vcat(x_kept, x_ref)), minimum(vcat(y_kept, y_ref))) * 0.9
    axis_hi = max(maximum(vcat(x_kept, x_ref)), maximum(vcat(y_kept, y_ref))) * 1.1
    figure = scatter(x_kept, y_kept; markersize = sqrt.(300 .* x_kept ./ maximum(x_kept)) ./ 2, alpha = 0.6,
                markerstrokecolor = :black, markerstrokewidth = 0.5,
                color = RGB(0.247, 0.404, 0.667), label = isempty(x_ref) ? "" : "Non-reference",
                xlabel = xlabel, ylabel = ylabel, title = title, xlims = (axis_lo, axis_hi), ylims = (axis_lo, axis_hi),
                legend = isempty(x_ref) ? false : :bottomright, grid = true, gridstyle = :dash)
    isempty(x_ref) || scatter!(figure, x_ref, y_ref; markershape = :diamond, markersize = 5, alpha = 0.7,
                               color = RGB(0.75, 0.30, 0.20), label = "Reference (reconstructed)")
    plot!(figure, [axis_lo, axis_hi], [axis_lo, axis_hi]; color = :black, label = "45°", linewidth = 1)
    annotate!(figure, axis_hi * 0.95, axis_lo + (axis_hi - axis_lo) * 0.12, text(@sprintf("Coef: %.3f", slope), :right, 8))
    annotate!(figure, axis_hi * 0.95, axis_lo + (axis_hi - axis_lo) * 0.04, text(@sprintf("t-stat: %.1f", slope_se > 0 ? slope / slope_se : Inf), :right, 8))
    return figure
end

"""
    plot_T_vs_initial(params, folder)

Area-level T at `params` against its starting value `T_rs_init` (both ref-normalised),
with the search-window lines ×BOUND_LO / ×BOUND_HI. Never blocks estimation.
"""
function plot_T_vs_initial(params, folder)
    try
        T_best = unpack_T_area(params)
        alpha = round(params[alpha_range()][1], digits = 3)
        kappa = round(unpack_kappa(params), digits = 3)   # T* depends on (α, κ), not on α alone
        plotted_cells = [(s, a) for s in 1:S for a in 1:T_COL_DIM if T_rs_init[s, a] > 1e-8 && T_best[s, a] > 1e-8]
        isempty(plotted_cells) && return
        T_initial = [T_rs_init[s, a] for (s, a) in plotted_cells]; T_current = [T_best[s, a] for (s, a) in plotted_cells]
        axis_lo, axis_hi = min(minimum(T_initial), minimum(T_current)), max(maximum(T_initial), maximum(T_current))
        figure = scatter(T_initial, T_current; xscale = :log10, yscale = :log10, label = "",
                    xlabel = "initial T / T_ref (α₀ = $(round(P_alpha, digits=3)))",
                    ylabel = "T / T_ref (α₁ = $alpha, κ = $kappa)", title = "T: initial vs current")
        plot!(figure, [axis_lo, axis_hi], [axis_lo, axis_hi]; color = :black, ls = :dash, label = "y = x")
        plot!(figure, [axis_lo, axis_hi], BOUND_HI .* [axis_lo, axis_hi]; color = :firebrick, ls = :dot, label = "×$BOUND_LO / ×$BOUND_HI")
        plot!(figure, [axis_lo, axis_hi], BOUND_LO .* [axis_lo, axis_hi]; color = :firebrick, ls = :dot, label = "")
        savefig(figure, joinpath(folder, "T_best_vs_initial_alpha$(alpha).png"))
        NPZ.npzwrite(joinpath(folder, "T_best_vs_initial_alpha$(alpha).npz"),
                     Dict("T_initial" => T_initial, "T_best" => T_current,
                          "sector" => Float64.(first.(plotted_cells)), "region" => Float64.(last.(plotted_cells))))
    catch e
        @warn "plot_T_vs_initial skipped: $e"
    end
end

# Summary of a moment block for report.txt: number of zeros, then the quartiles and the
# maximum of the nonzero entries.
_quartiles(values) = (nonzero = filter(!=(0), vec(values));
                 (n_zeros = count(==(0), vec(values)), q1 = quantile(nonzero, 0.25), median = quantile(nonzero, 0.5),
                  q3 = quantile(nonzero, 0.75), max_val = maximum(nonzero)))

"""
    generate_report(folder, params; variable="", tightness="")

Writes `dashboard.png` (γ_{s,a} with the reconstructed reference areas, π_r, sector
shares), the T plot, `pi_r.npy`, `productivity.npy` and `report.txt` (fit of every
block, the profiled N̂_s with its clamps, block 4 against the data).
"""
function generate_report(folder::String, params::Vector{Float64}; variable::String = "", tightness = "")
    mkpath(folder)
    loss, simulated = full_SMM(params)
    sim_vec = moments_to_vec(simulated)
    emp_vec = vec(empirical_moments)
    gamma_rows = collect(BLOCK_RANGES[5])
    emp_gamma, sim_gamma = emp_vec[gamma_rows], sim_vec[gamma_rows]
    x_ref = Float64[]; y_ref = Float64[]
    # The reference area's γ is not a moment (it is pinned by the normalisation); rebuild it
    # from the sector total so the dashboard shows every area.
    for ref_entry in GAMMA_REF_MAP
        ref_entry.emp_ref > 0 || continue
        push!(x_ref, ref_entry.emp_ref); push!(y_ref, ref_entry.c_s - sum(sim_gamma[ref_entry.local_positions]))
    end
    fig_gamma = bubble_scatter(emp_gamma, sim_gamma; xlabel = "Empirical γ", ylabel = "Simulated γ",
                        title = "γ_{s,a}: Empirical vs Simulated", x_ref = x_ref, y_ref = y_ref)
    fig_pi = bubble_scatter(emp_pi_r, simulated[3]; xlabel = "Empirical π_r", ylabel = "Simulated π_r",
                        title = "π_r: Empirical vs Simulated")
    fig_sector = bubble_scatter(agg_industry_share, simulated[2]; xlabel = "Empirical π_s", ylabel = "Simulated π_s",
                        title = "π_s: Empirical vs Simulated")
    plot(fig_gamma, fig_pi, fig_sector, layout = (1, 3), size = (1500, 500), margin = 5Plots.mm)
    savefig(joinpath(folder, "dashboard.png"))
    plot_T_vs_initial(params, folder)
    npzwrite(joinpath(folder, "pi_r.npy"), simulated[3])
    npzwrite(joinpath(folder, "productivity.npy"), unpack_params(params)[3])

    granular = try granular_report(params) catch e; @warn "granular section skipped: $e"; nothing end
    open(joinpath(folder, "report.txt"), "w") do io
        println(io, "Date: $(Dates.format(now(), "yyyy-mm-dd HH:MM"))")
        println(io, "Optimisation: $(isempty(variable) ? "All" : variable)   tightness: $tightness")
        println(io, "Score: $(loss[1])")
        println(io, "α = $(round.(params[alpha_range()], digits = 4))   κ = $(round(unpack_kappa(params), digits = 4))\n")
        @printf(io, ">> Labour share   Empirical: %8.4f | Simulated: %8.4f\n", agg_labor_share, simulated[1][1])
        q_emp, q_sim = _quartiles(emp_gamma), _quartiles(sim_gamma)
        println(io, "\n>> γ_{s,a} (quartiles of non-zero entries):")
        for stat in (:n_zeros, :q1, :median, :q3, :max_val)
            @printf(io, "%-10s Empirical: %8.3f | Simulated: %8.3f\n", stat, getfield(q_emp, stat), getfield(q_sim, stat))
        end
        q_emp, q_sim = _quartiles(emp_pi_r), _quartiles(simulated[3])
        println(io, "\n>> π_r:")
        for stat in (:q1, :median, :q3, :max_val)
            @printf(io, "%-10s Empirical: %8.3f | Simulated: %8.3f\n", stat, getfield(q_emp, stat), getfield(q_sim, stat))
        end
        println(io, "\n>> Sector input shares (total purchases, imports included):")
        for s in 1:S
            @printf(io, "%-10s Empirical: %8.3f | Simulated: %8.3f\n", SECTOR_NAMES[s], agg_industry_share[s], simulated[2][s])
        end
        println(io, "\n>> Import share f_s (foreign row of the γ block; Φ^F_s profiled with T):")
        for s in 1:S
            @printf(io, "%-10s Empirical: %8.3f | Simulated: %8.3f | Φ^F_s %10.4g\n", SECTOR_NAMES[s],
                    FOREIGN_SHARE[s], simulated[5][FOREIGN_COL, s], foreign_phi(unpack_T_area(params))[s])
        end
        println(io, "\n>> Block 4 — cell-level cloglog:")
        for (i, lab) in enumerate(REG_COEF_LABELS)
            @printf(io, "%-10s Empirical: %8.3f | Simulated: %8.3f\n", lab, reg_coef[i], simulated[4][i])
        end
        println(io, "\n>> Count moment Ḡ_s(0) and the profiled variety count:")
        @printf(io, "%-10s %9s %9s %6s %6s %6s %6s %9s\n", "sector", "G0 data", "G0 model",
                "N_LO", "N̂_s", "N_HI", "clamp", "N_count")
        for s in 1:S
            @printf(io, "%-10s %9.4f %9.4f %6d %6s %6s %6s %9s\n", SECTOR_NAMES[s], G_TARGET[s], simulated[6][s],
                    granular === nothing ? N_LO[s] : granular.N_lo[s], granular === nothing ? "-" : string(granular.N_hat[s]),
                    granular === nothing ? string(N_HI[s]) : string(granular.N_hi[s]),
                    granular === nothing ? "-" : string(granular.clamped[s]),
                    granular === nothing ? "-" : @sprintf("%.1f", granular.N_count[s]))
        end
        println(io, "\n  clamp :lo/:hi = the variety count sits on a bound (a rejection signal, not a nuisance).")
        println(io, "  N_count = N_supplier_s / Σ_l q̂_ls is the independent second route to N_s.")
        println(io, "  N_HI = N^obs_s / q_s, q_s the share of varieties won by France (N^obs_s without imports).")
    end
end

##################### Run summaries ###################

# The numbered stage folders of an epoch (those holding a best_params.npy) and the
# epoch folders of a step, each as (number, path) in increasing order.
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
    results = pmap(stage_params -> safe_SMM(stage_params), params)
    scores  = [result === nothing ? Inf : result[1][1] for result in results]
    block4_rmse   = [result === nothing ? Inf : sqrt(mean((result[2][4] .- reg_coef) .^ 2)) for result in results]
    moments = [vcat([vec(b) for b in result[2]]...) for result in results if result !== nothing]
    isempty(moments) || npzwrite(joinpath(step_folder, "best_simulated_moments.npy"), hcat(moments...))
    npzwrite(joinpath(step_folder, "best_parameters_list.npy"), hcat(params...))
    npzwrite(joinpath(step_folder, "empirical_moments.npy"), empirical_moments)
    relative_to_first(values) = values[1] > 0 ? values ./ values[1] .* 100 : values
    npzwrite(joinpath(step_folder, "score_history.npy"),
             Dict("first_stage" => relative_to_first(scores), "min_dist_first" => relative_to_first(block4_rmse)))
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
        granular = granular_report(theta)
        mkpath(folder)
        lines = String[]
        push!(lines, "GRANULAR DIAGNOSTICS $(industry) $(label)")
        push!(lines, @sprintf("  %-8s %6s %6s %6s %6s %9s %9s %9s %7s %7s %7s", "sector", "N_LO", "N̂_s", "N_HI",
                              "clamp", "G0_fit", "G0_tgt", "N_count", "q_FR", "f_fit", "f_tgt"))
        for s in 1:S
            push!(lines, @sprintf("  %-8s %6d %6d %6d %6s %9.4f %9.4f %9.1f %7.3f %7.3f %7.3f", SECTOR_NAMES[s], granular.N_lo[s],
                                  granular.N_hat[s], granular.N_hi[s], string(granular.clamped[s]), granular.G0[s],
                                  G_TARGET[s], granular.N_count[s], granular.q_french[s], granular.foreign_share[s],
                                  FOREIGN_SHARE[s]))
        end
        push!(lines, "  N_LO = N^obs_s / (R_d q_FR), N_HI = N^obs_s / q_FR (q_FR: varieties won by France for ≥ 1 buyer); f: import share")
        push!(lines, "  Block 4 (R = $R_CELL_REG networks, $(granular.reg_fail) failed fits)")
        push!(lines, @sprintf("  %-10s %10s %12s %10s %12s", "coef", "data", "E[β̂(y)]", "sd_j", "β̂(E[y])"))
        for (i, lab) in enumerate(REG_COEF_LABELS)
            push!(lines, @sprintf("  %-10s %10.4f %12.4f %10.4f %12.4f", lab, reg_coef[i],
                                  granular.reg_coef[i], granular.reg_sd[i], granular.beta_expected[i]))
        end
        println("\n", join(lines, "\n"))
        write(joinpath(folder, "granular_diagnostics.txt"), join(lines, "\n") * "\n")
        NPZ.npzwrite(joinpath(folder, "granular_diagnostics.npz"),
            Dict("N_hat" => Float64.(granular.N_hat), "N_LO" => Float64.(granular.N_lo), "N_HI" => Float64.(granular.N_hi),
                 "N_obs" => Float64.(N_HI), "q_french" => Float64.(granular.q_french),
                 "foreign_share_fit" => Float64.(granular.foreign_share),
                 "foreign_share_target" => Float64.(collect(FOREIGN_SHARE)),
                 "clamped" => [clamp_state == :none ? 0.0 : (clamp_state == :lo ? -1.0 : 1.0) for clamp_state in granular.clamped],
                 "G0_fit" => Float64.(granular.G0), "G0_target" => Float64.(collect(G_TARGET)),
                 "N_count" => Float64.(granular.N_count), "q_hat" => Float64.(granular.q_hat),
                 "reg_coef_model" => Float64.(granular.reg_coef), "reg_coef_sd_reps" => Float64.(granular.reg_sd),
                 "reg_coef_expected" => Float64.(granular.beta_expected),
                 "reg_fail" => [Float64(granular.reg_fail)], "EK" => Float64.(granular.EK)))
        return granular
    catch e
        @warn "granular diagnostics skipped: $e"
        return nothing
    end
end

##################### Post-hoc firm-level economy ###################

const POST_HOC_REPLICATIONS = 50   # independent realisations of the N̂_s-variety economy

# The columns of suppliers.parquet, one vector each, filled row by row.

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
function _firm_level_rows!(cols, network, siren_map::Dict{NTuple{4,Int},Int}, replication::Int)
    for i in eachindex(network.firm_exp_rho)
        share_value = network.firm_exp_val[i]
        share_value == 0.0 && continue
        rho = network.firm_exp_rho[i]; s = network.firm_exp_s[i]
        g = network.firm_exp_g[i]; l = GOOD_R[g]; region = network.firm_exp_r[i]
        push!(cols.sirens, get!(siren_map, (replication, l, s, rho), length(siren_map) + 1))
        push!(cols.sectors, s); push!(cols.ze2010, l); push!(cols.ze2010_downstream, region)
        push!(cols.share, share_value); push!(cols.downstream_purchase, network.Y_r[region] * network.mu)
        push!(cols.intermediate_derivative, network.firm_deriv_val[i])
        push!(cols.productivity, network.z_flat[rho, g])
        push!(cols.sample_weight, network.sample_weights[rho, g])
        push!(cols.variety, rho); push!(cols.replication, replication)
    end
end

"""
    write_post_hoc(params, dest; n_rep=POST_HOC_REPLICATIONS, base_seed=8_100_000)

The economy the Python reporting checks against (`check_against_julia`):
  * `suppliers.parquet` — `n_rep` independent realisations (i.i.d. draws) of the
    N̂_s-variety economy (weights 1/N̂_s on rows ρ ≤ N̂_s, 0 beyond), with `replication`
    and `variety` columns; `post_hoc_u.npy` the cell draws (N_max × n_good × n_rep),
    `post_hoc_u_foreign.npy` the foreign competitor's (N_max × S × n_rep),
    `post_hoc_good_sr.npy` the (sector, cell) of each draw column, `post_hoc_N_hat.npy`.
    The rows are French linkages only: an imported variety creates none, so `share`
    sums to the DOMESTIC intermediate share of the buyer's cost and the printed D_r is
    1 + that share.
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
        cell_sales = sum(X_lrs[l, :, s])
        cell_sales > 1e-10 && (w_srd_r[s, l, :] .= X_lrs[l, :, s] ./ cell_sales)
    end
    npzwrite(joinpath(dest, "w_srd_r.npy"), w_srd_r)
    n_draws = size(network.linkages_flat, 1)
    suppliers = zeros(Bool, S, n_draws, R)
    for g in 1:n_good, rho in 1:n_draws
        network.linkages_flat[rho, g] && (suppliers[GOOD_S[g], rho, GOOD_R[g]] = true)
    end
    npzwrite(joinpath(dest, "suppliers.npy"), suppliers)
    ref_cols = _empty_firm_cols()
    _firm_level_rows!(ref_cols, network, Dict{NTuple{4,Int},Int}(), 0)
    Parquet.write_parquet(joinpath(dest, "suppliers_continuum.parquet"), _firm_cols_frame(ref_cols))

    k_counts = vec(sum(network.linkages_flat; dims = 1))
    N_hat, clamped = concentrate_N_s(k_counts, n_draws; q = french_variety_share(network.linkages_flat))
    N_max = maximum(N_hat)
    npzwrite(joinpath(dest, "post_hoc_N_hat.npy"), N_hat)
    weights_granular = zeros(N_max, n_good)
    for g in 1:n_good
        weights_granular[1:N_hat[GOOD_S[g]], g] .= 1.0 / N_hat[GOOD_S[g]]
    end
    cols = _empty_firm_cols(); siren_map = Dict{NTuple{4,Int},Int}()
    u_stack = zeros(N_max, n_good, n_rep); u_foreign_stack = zeros(N_max, S, n_rep)
    for replication in 1:n_rep
        u_draws_rep = generate_draws(N_max, n_good, :mc; randomise = true, rng = MersenneTwister(base_seed + replication))[1]
        for g in 1:n_good, rho in (N_hat[GOOD_S[g]] + 1):N_max
            u_draws_rep[rho, g] = 0.5                # zero-weight rows: keep every price finite
        end
        u_stack[:, :, replication] .= u_draws_rep[:, 1:n_good]; u_foreign_stack[:, :, replication] .= u_draws_rep[:, (n_good + 1):(n_good + S)]
        network_rep = solve_network(params; u_draws = u_draws_rep, sample_weights = weights_granular, return_firm_level = true)
        _firm_level_rows!(cols, network_rep, siren_map, replication)
    end
    npzwrite(joinpath(dest, "post_hoc_u.npy"), u_stack)
    npzwrite(joinpath(dest, "post_hoc_u_foreign.npy"), u_foreign_stack)
    npzwrite(joinpath(dest, "post_hoc_good_sr.npy"), hcat(GOOD_S, GOOD_R))
    Parquet.write_parquet(joinpath(dest, "suppliers.parquet"), _firm_cols_frame(cols))
    @printf("  post-hoc -> %s: N̂_s %s (%d clamped), %d linkages over %d realisations; mean domestic D_r %.4f (granular) vs %.4f (N_s → ∞)\n",
            dest, string(N_hat), count(!=(:none), clamped),
            length(cols.share), n_rep, 1 + sum(cols.share) / (R_downstream * n_rep),
            1 + sum(ref_cols.share) / R_downstream)
end
