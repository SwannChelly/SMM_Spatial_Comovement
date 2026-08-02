using Printf



"""
    generate_lhs_alpha(n_samples, n_alpha, lb, ub; seed=42)

Generate LHS samples for alpha enforcing monotonicity: alpha_1 <= alpha_2 <= ... <= alpha_K.
Method: generate LHS in [0,1]^K, sort each sample, map to [lb, ub].
"""
function generate_lhs_alpha(n_samples::Int, n_alpha::Int, lb, ub; seed=42)
    lb_vec = lb isa Number ? fill(Float64(lb), n_alpha) : Float64.(lb)
    ub_vec = ub isa Number ? fill(Float64(ub), n_alpha) : Float64.(ub)

    raw_matrix = QuasiMonteCarlo.sample(n_samples, zeros(n_alpha), ones(n_alpha),
                                         LatinHypercubeSample())

    samples = Vector{Vector{Float64}}()
    for i in 1:n_samples
        raw = sort(raw_matrix[:, i])  # Sort to enforce monotonicity
        alpha = lb_vec .+ raw .* (ub_vec .- lb_vec)
        push!(samples, alpha)
    end
    return samples
end

"""
    generate_log_grid_alpha(n_alpha, lb, ub, length_range)

Generate log-spaced alpha grid with monotonicity constraint (α₁ ≤ α₂ ≤ ... ≤ α_K).
Supports n_alpha = 4 (pattern [i,j,k,k]) or n_alpha = 5 (pattern [i,j,k,k,k]).
"""
function generate_log_grid_alpha(n_alpha::Int, lb::Real, ub::Real, length_range::Int)
    range_alpha = exp.(range(log(lb), stop=log(ub), length=length_range))

    if n_alpha == 1
        expanding_alpha = [[x] for x in range_alpha]
    elseif n_alpha == 4
        expanding_alpha = [
            [i, k, k,k]
            for i in range_alpha
            for k in range_alpha
            for k in range_alpha
            if i  <= k
        ]
    elseif n_alpha == 5
        expanding_alpha = [
            [i, j, k, k, k]
            for i in range_alpha
            for j in range_alpha
            for k in range_alpha
            if i <= j <= k
        ]
    else
        error("Log grid alpha generation not implemented for n_alpha=$n_alpha")
    end

    return expanding_alpha
end

"""
    generate_initial_alphas(method, n_alpha, lb, ub; lhs_n_samples=1500, log_grid_length=20)

Unified interface for generating initial alpha candidates.

# Arguments
- `method`: "log_grid" or "lhs"
- `n_alpha`: Number of alpha parameters
- `lb`, `ub`: Bounds for alpha values
- `lhs_n_samples`: Number of LHS samples (only for method="lhs")
- `log_grid_length`: Grid resolution (only for method="log_grid")
"""
function generate_initial_alphas(method::String, n_alpha::Int, lb::Real, ub::Real;
                                 lhs_n_samples::Int=1500, log_grid_length::Int=20)
    if method == "log_grid"
        return generate_log_grid_alpha(n_alpha, lb, ub, log_grid_length)
    elseif method == "lhs"
        return generate_lhs_alpha(lhs_n_samples, n_alpha, lb, ub)
    else
        error("Unknown alpha search method: $method. Use 'log_grid' or 'lhs'.")
    end
end



function assign_T_with_mask(true_T,sample)    
    mask = vec(mask_emp_gamma_ls)
    accept = copy(true_T)
    accept[mask.== 1] = sample
    return accept 
end

function parallel_SMM(params, simulation, second_stage;
                      precomputed_tau::Union{Nothing, Matrix{Float64}}=nothing,
                      u_draws::Union{Nothing, Matrix{Float64}}=nothing,
                      sample_weights::Union{Nothing, Matrix{Float64}}=nothing,
                      W_override::Union{Nothing, AbstractMatrix}=nothing,
                      moment_blocks::Union{Nothing, Vector{Int}}=nothing,
                      analytical::Bool=false,
                      n_quad::Int=200)
    return full_SMM(params, simulation, second_stage;
                    precomputed_tau=precomputed_tau, u_draws=u_draws, sample_weights=sample_weights,
                    W_override=W_override, moment_blocks=moment_blocks,
                    analytical=analytical, n_quad=n_quad)
end


function parallel_SMM_safe(params, simulation = false, second_stage = false, show_err = true;
                           precomputed_tau::Union{Nothing, Matrix{Float64}}=nothing,
                           u_draws::Union{Nothing, Matrix{Float64}}=nothing,
                           sample_weights::Union{Nothing, Matrix{Float64}}=nothing,
                           W_override::Union{Nothing, AbstractMatrix}=nothing,
                           moment_blocks::Union{Nothing, Vector{Int}}=nothing,
                           analytical::Bool=false,
                           n_quad::Int=200)
    try
        result = parallel_SMM(params, simulation, second_stage;
                              precomputed_tau=precomputed_tau, u_draws=u_draws, sample_weights=sample_weights,
                              W_override=W_override, moment_blocks=moment_blocks,
                              analytical=analytical, n_quad=n_quad)

        return result
    catch e
        # If an error occurs, return a message or a placeholder result
        if show_err
            println("ERROR!!")
            println(e)
        end
        return nothing
    end
end


function distance_bin(d, n_bins=N_REG)
    if n_bins == 5
        if 20 < d <= 50
            return 1
        elseif 50 < d <= 100
            return 2
        elseif 100 < d <= 150
            return 3
        elseif 150 < d <= 200
            return 4
        elseif d > 200
            return 5
        else
            return 0
        end
    elseif n_bins == 4
        # Define your 4-bin version here
        if 50 < d <= 100
            return 1
        elseif 100 < d <= 150
            return 2
        elseif 150 < d <= 200
            return 3
        elseif d > 200
            return 4
        else
            return 0
        end
    elseif n_bins == 1
        return 0
    else
        error("Unsupported number of distance bins: $n_bins")
    end
end

function train_stage_one(n, init_alpha, params_list = nothing, second_stage = false;
                        u_draws::Union{Nothing, Matrix{Float64}}=nothing,
                        sample_weights::Union{Nothing, Matrix{Float64}}=nothing,
                        analytical::Bool=false,
                        n_quad::Int=200)
    t1 = time()
    if params_list == nothing
        error("train_stage_one now requires an explicit params_list " *
              "(the legacy generate_halton_grid initialiser was removed).")
    end
    f = params -> parallel_SMM_safe(params, false, second_stage, true;
                                    u_draws=u_draws, sample_weights=sample_weights,
                                    analytical=analytical, n_quad=n_quad)
    results = pmap(f, params_list)
    return params_list,results
end

# Returns (n_zeros, q1, median, q3, max_val) when include_n_zero=true, else (q1, median, q3, max_val). Callers must be consistent with their choice.
function matrix_report(mat,include_n_zero = true)
    vals = vec(mat)
    n_zeros = count(==(0), vals)
    nonzero_vals = filter(!=(0), vals)

    if isempty(nonzero_vals)
        error("Matrix contains only zeros; cannot compute statistics on non-zero values.")
    end
    if include_n_zero
        return (
        n_zeros = n_zeros,
        q1 = quantile(nonzero_vals, 0.25),
        median = quantile(nonzero_vals, 0.50),
        q3 = quantile(nonzero_vals, 0.75),
        max_val = maximum(nonzero_vals),
    )
    else
        return (
            q1 = quantile(nonzero_vals, 0.25),
            median = quantile(nonzero_vals, 0.50),
            q3 = quantile(nonzero_vals, 0.75),
            max_val = maximum(nonzero_vals),
        )
    end
end


function generate_dashboard_report(
    n,agg_labor_share_,agg_industry_share_,gamma_ls_,reg_,pi_r,best_score,
    output_file::String,variable,alpha; G0_ = nothing, granular_info = nothing
)   

    # gamma_ls summary table
    if variable == nothing 
        variable = "All"
    end
    agg_labor_share_emp,agg_labor_share_sim=agg_labor_share_
    agg_industry_share_emp,agg_industry_share_sim=agg_industry_share_
    gamma_emp,gamma_sim = gamma_ls_
    reg_emp,reg_sim = reg_
    pi_r_emp,pi_r_sim = pi_r

    gamma_df = DataFrame(
        metric = ["Number of zeros", "Q1", "Median", "Q3", "Max value"],
        empirical = [
            gamma_emp.n_zeros,
            gamma_emp.q1,
            gamma_emp.median,
            gamma_emp.q3,
            gamma_emp.max_val
        ],
        simulated = [
            gamma_sim.n_zeros,
            gamma_sim.q1,
            gamma_sim.median,
            gamma_sim.q3,
            gamma_sim.max_val
        ]
    )
    pi_r_df = DataFrame(
        metric = [ "Q1", "Median", "Q3", "Max value"],
        empirical = [
            pi_r_emp.q1,
            pi_r_emp.median,
            pi_r_emp.q3,
            pi_r_emp.max_val
        ],
        simulated = [
            pi_r_sim.q1,
            pi_r_sim.median,
            pi_r_sim.q3,
            pi_r_sim.max_val
        ]
    )

    sectors = sort(unique(CSV.read(joinpath(input_folder, "filter_N_upstream.csv"), DataFrame)[!, "A129"]))
    agg_industry_share_df = DataFrame(
        metric = sectors,
        empirical = agg_industry_share_emp,
        simulated = agg_industry_share_sim
    )


    bin_labels = if N_REG == 5
        ["]20,50]", "]50,100]", "]100,150]", "]150,200]", ">200"]
    elseif N_REG == 4
        ["]50,100]", "]100,150]", "]150,200]", ">200"]
    else
        ["Bin $i" for i in 1:N_REG]
    end

    reg_df = DataFrame(
        bins = bin_labels,
        empirical = reg_emp,
        simulated = reg_sim
    )

    date = Dates.format(now(), "yyyy-mm-dd HH:MM")
    open(output_file, "w") do io
        println(io,"Date: $date\n")
        println(io,"Optimisation: $variable\n")
        println(io,"Learning rate: $alpha\n")
        println(io, "Score: $best_score\n") # Ajoutez cette ligne pour inclure le score
        println(io, "Size of the grid: $n\n") 
        println(io, "===========================\n     MODEL DIAGNOSTICS REPORT\n===========================\n")
        println(io, "\n>> Aggregate labor share:\n")
        println(io, @sprintf("%-15s  Empirical: %8.4f  |  Simulated: %8.4f", "Coefficient", agg_labor_share_emp, agg_labor_share_sim))
        println(io, ">> gamma_ls (quartile are for the distribution without zeros):\n")
        for row in eachrow(gamma_df)
            println(io, @sprintf("%-15s  Empirical: %8.3f  |  Simulated: %8.3f",
                row.metric, row.empirical, row.simulated))
        end

        println(io, "\n>> pi_r :\n")
        for row in eachrow(pi_r_df)
            println(io, @sprintf("%-15s  Empirical: %8.3f  |  Simulated: %8.3f",
                row.metric, row.empirical, row.simulated))
        end

        println(io, "\n>> Input share: \n")
        for row in eachrow(agg_industry_share_df)
            println(io, @sprintf("%-15s  Empirical: %8.3f  |  Simulated: %8.3f",
                row.metric, row.empirical, row.simulated))
        end

        println(io, "\n>> Regression coefficients: \n")
        for row in eachrow(reg_df)
            println(io, @sprintf("%-15s  Empirical: %8.3f  |  Simulated: %8.3f",
                row.bins, row.empirical, row.simulated))
        end

        # ── Block 6: the count moment Ḡ_s(0), granular runs only ────────────
        if G0_ !== nothing
            G0_emp, G0_sim = G0_
            println(io, "\n>> Count moment  G_s(0)  (share of cells with ZERO suppliers): \n")
            for (k, sec) in enumerate(sectors)
                k <= length(G0_emp) && k <= length(G0_sim) || continue
                println(io, @sprintf("%-15s  Empirical: %8.4f  |  Simulated: %8.4f",
                    string(sec), G0_emp[k], G0_sim[k]))
            end
            if granular_info !== nothing
                println(io, "\n>> Profiled variety count N_s (bisection on G_s(0)): \n")
                println(io, @sprintf("%-10s %8s %8s %8s %8s %10s",
                    "sector", "N_LO", "N_hat", "N_HI", "clamp", "N_count"))
                for (k, sec) in enumerate(sectors)
                    k <= length(granular_info.N_hat) || continue
                    println(io, @sprintf("%-10s %8d %8d %8d %8s %10.1f",
                        string(sec), N_LO[k], granular_info.N_hat[k], N_HI[k],
                        string(granular_info.clamped[k]), granular_info.N_count[k]))
                end
                println(io, "\n  clamp: :none = interior (good). :lo/:hi = the variety count hit a")
                println(io, "  bound — a rejection signal for the mechanism, not a numerical nuisance.")
                println(io, "  Simulated G_s(0) above is the CLOSED FORM mean_l (1-q_hat)^N_hat, which")
                println(io, "  is the exact expectation of the empty-cell share — unbiased and noise-free.")
                println(io, "  N_count = N_supplier_s / sum_l q_hat is the independent second route to")
                println(io, "  N_s (over-identifying check); a large gap is a mechanism finding.")
                println(io, @sprintf("\n  log z coefficient: %+.4f   (theory: -theta = %+.4f)",
                    granular_info.b_logz, -theta))
            end
        end
    end
end


"""
    bubble_scatter(x, y; xlabel, ylabel, title, size_scale, regression_line)

Create a bubble scatter plot with empirical (x) vs simulated (y) values.
Dot sizes are proportional to empirical values.
Includes weighted regression (WLS, no intercept, weights = x) with coefficient and t-stat annotation.
Draws a 45° reference line by default, or the regression line if `regression_line=true`.
"""
function bubble_scatter(x::AbstractVector, y::AbstractVector;
                        xlabel::String="Empirical",
                        ylabel::String="Simulated",
                        title::String="",
                        size_scale::Real=300,
                        regression_line::Bool=false)

    # Keep only observations where empirical value > 0
    mask = x .> 0
    xf = Float64.(x[mask])
    yf = Float64.(y[mask])

    # Marker sizes proportional to empirical values
    sizes = size_scale .* xf ./ maximum(xf)

    # Axis limits (with a small margin)
    lo = min(minimum(xf), minimum(yf)) * 0.9
    hi = max(maximum(xf), maximum(yf)) * 1.1
    lims = (lo, hi)

    # ── Weighted least-squares:  y = b·x  (no intercept), weights = x ──
    w = xf
    b = sum(w .* xf .* yf) / sum(w .* xf .^ 2)

    # Residuals & standard error
    resid = yf .- b .* xf
    n = length(xf)
    s2 = sum(w .* resid .^ 2) / max(n - 1, 1)
    se_b = sqrt(s2 / sum(w .* xf .^ 2))
    t_stat = se_b > 0 ? b / se_b : Inf

    # ── Build scatter plot ──
    p = scatter(xf, yf;
        markersize  = sqrt.(sizes) ./ 2,
        alpha       = 0.6,
        markerstrokecolor = :black,
        markerstrokewidth = 0.5,
        color       = RGB(0.247, 0.404, 0.667),   # steelblue-ish, similar to Toulouse color
        label       = "",
        xlabel      = xlabel,
        ylabel      = ylabel,
        title       = title,
        xlims       = lims,
        ylims       = lims,
        grid        = true,
        gridalpha   = 0.5,
        gridstyle   = :dash
    )

    # ── Reference / regression line ──
    if regression_line
        X_line = range(0, hi, length=100)
        Y_line = b .* X_line
        plot!(p, X_line, Y_line; linestyle=:dash, color=:green, label="Fit")
    else
        plot!(p, [lo, hi], [lo, hi]; color=:black, label="45°", linewidth=1)
    end

    # ── Annotate coefficient and t-stat ──
    annotate!(p,
        hi * 0.95, lo + (hi - lo) * 0.12,
        text(@sprintf("Coef: %.3f", b), :right, 8))
    annotate!(p,
        hi * 0.95, lo + (hi - lo) * 0.04,
        text(@sprintf("t-stat: %.1f", t_stat), :right, 8))

    return p
end


"""
    plot_T_vs_initial(best_params, out_folder; label="")

Scatter the current best T against the *very initial* T (`T_rs_init`, the
γ-inversion warm start built at the α prior). Called at the end of each estimation
step. Both axes are per-sector reference-normalised, so a point on the 45° line is
a region whose T was left unchanged by the fit.

- **x** = initial T at the initial α (`TAU_PRIOR`, the γ-inversion prior).
- **y** = current best T at the current best α (`unpack_params(best_params)` β block).

The current best α is embedded in the scatter title, axis labels, and the output
filename (`T_best_vs_initial_alpha<α>.png` + companion `.npz` with the raw pairs).
Fully guarded — a failure here never blocks estimation.
"""
function plot_T_vs_initial(best_params, out_folder; label::String = "")
    try
        mkpath(out_folder)
        _, _, _, beta_best, T_best_flat = unpack_params(best_params)
        # Current best α (N_TAU==1 ⇒ scalar α); initial α from the γ-inversion prior.
        cur_alpha  = length(beta_best) >= 1 ? Float64(beta_best[1]) : NaN
        init_alpha = (TAU_PRIOR !== nothing) ? Float64(TAU_PRIOR[1]) : NaN

        # Compare in the T-PARAMETER space (T_COL_DIM wide: ZE under :ze, attraction
        # areas under :aa), which is where T_rs_init lives. Both are already
        # per-sector ref-normalised → directly comparable.
        T_best = unpack_T_par(best_params)
        T_init = T_rs_init

        xs = Float64[]; ys = Float64[]; ss = Int[]; rs = Int[]
        for s in 1:S, r in 1:T_COL_DIM
            ti = T_init[s, r]; tb = T_best[s, r]
            (isfinite(ti) && isfinite(tb) && ti > 1e-8 && tb > 1e-8) || continue
            push!(xs, ti); push!(ys, tb); push!(ss, s); push!(rs, r)
        end
        isempty(xs) && return

        a_i = round(init_alpha, digits = 3)
        a_c = round(cur_alpha,  digits = 3)
        ttl = "T distribution: initial (α=$a_i) vs best (α=$a_c)" *
              (isempty(label) ? "" : "  [$label]")
        p = Plots.scatter(xs, ys; xscale = :log10, yscale = :log10,
            xlabel = "initial T / T_ref  (α=$a_i)",
            ylabel = "current best T / T_ref  (α=$a_c)",
            title  = ttl,
            markersize = 5, markeralpha = 0.7, legend = :topleft, label = "")
        lo = min(minimum(xs), minimum(ys)); hi = max(maximum(xs), maximum(ys))
        Plots.plot!(p, [lo, hi], [lo, hi]; color = :black, ls = :dash, label = "y = x (no change)")

        # Exploration-window borders. Each best-T coordinate is boxed to
        # [BOUND_LO, BOUND_HI] × its INITIAL value (optimizer.jl train_stage; the T-φ
        # box is φ_init + [log BOUND_LO, log BOUND_HI]). BOUND_LO/BOUND_HI are the
        # shared globals from load_parameters.jl — same source of truth as the optimizer.
        # Since the x-axis is the initial T, the borders are y = BOUND_LO·x and
        # y = BOUND_HI·x — straight lines parallel to the diagonal in log-log. Points
        # should sit between them.
        Plots.plot!(p, [lo, hi], BOUND_HI .* [lo, hi];
                    color = :firebrick, ls = :dot,
                    label = "×$(BOUND_LO) / ×$(BOUND_HI) search window")
        Plots.plot!(p, [lo, hi], BOUND_LO .* [lo, hi];
                    color = :firebrick, ls = :dot, label = "")

        png = joinpath(out_folder, "T_best_vs_initial_alpha$(a_c).png")
        Plots.savefig(p, png)
        NPZ.npzwrite(joinpath(out_folder, "T_best_vs_initial_alpha$(a_c).npz"),
                     Dict("T_initial" => xs, "T_best" => ys,
                          "sector" => Float64.(ss), "region" => Float64.(rs),
                          "alpha_initial" => [init_alpha], "alpha_best" => [cur_alpha]))
        println("  [T-plot] saved $(png)  (α_init=$a_i, α_best=$a_c, n=$(length(xs)))")
    catch e
        @warn "plot_T_vs_initial skipped: $e"
    end
end


function generate_report(loop_folder, stage, n, variable=nothing, best_params=nothing, alpha="";
                         u_draws::Union{Nothing, Matrix{Float64}}=nothing,
                         sample_weights::Union{Nothing, Matrix{Float64}}=nothing,
                         analytical::Bool=false,
                         n_quad::Int=200)

    folder = joinpath(loop_folder, stage)
    mkpath(folder)

    if best_params == nothing
        best_params = NPZ.npzread(joinpath(folder, "best_params.npy"))
        params_list = [best_params[:, K] for K in 1:K_max]
        params_list, results = train_stage_one(n, nothing, params_list, false;
                                               u_draws=u_draws, sample_weights=sample_weights,
                                               analytical=analytical, n_quad=n_quad)
        score = [score[1] != nothing ? score[1][1] : missing for score in results]
        best_index = argmin(score)
        best_params = best_params[:, best_index]
    else
        results = [full_SMM(best_params; u_draws=u_draws, sample_weights=sample_weights,
                            analytical=analytical, n_quad=n_quad)]
        best_index = 1
    end

    # ── Extract empirical & simulated vectors ──
    # Use the same masked gamma subset as the optimizer and compute_smm_inference:
    # apply MOMENT_MASK to the raw simulated blocks, then index BLOCK_RANGES[5].
    sim_flat_masked = moments_to_vec(results[best_index][2])
    emp_gamma = empirical_moments[collect(BLOCK_RANGES[5])]
    sim_gamma = sim_flat_masked[collect(BLOCK_RANGES[5])]

    emp_pi = vec(emp_pi_r)
    sim_pi_r = vec(results[best_index][2][3])

    emp_pi_sA = agg_industry_share
    sim_pi_sA = results[best_index][2][2]

    # ── Bubble scatter plots ──

    # p1 (γ_ls) is built directly here — NOT via bubble_scatter — so it can carry
    # the two-series color/marker distinction and legend from compute_smm_inference
    # §6b (the gamma_ref_map block). bubble_scatter is shared by p2/p3 below and
    # plots a single unlabelled series; labelling its points to drive a legend would
    # spill that legend onto p2/p3. p2/p3 stay on bubble_scatter, unchanged.
    p1 = let emp_g = Float64.(emp_gamma), sim_g = Float64.(sim_gamma)
        # Non-reference (retained) points: empirical x > 0 (as in bubble_scatter).
        keep = emp_g .> 0
        xf   = emp_g[keep]
        yf   = sim_g[keep]

        # Reconstruct each sector's dropped reference-region point from the
        # within-sector adding-up constraint γ_ref,s = c_s − Σ_{l≠ref} γ_ls.
        # sim_gamma is the γ block as a standalone 1-based vector (BLOCK_RANGES[5]
        # extracted), so entry.local_positions indexes it with ZERO offset — no
        # β-block precedes it here (unlike §6b's β-then-γ subsystem).
        x_ref = Float64[]; y_ref = Float64[]
        for entry in GAMMA_REF_MAP
            pos = entry.local_positions
            (isempty(pos) || maximum(pos) > length(sim_g)) && continue
            entry.emp_ref <= 0 && continue
            push!(x_ref, entry.emp_ref)
            push!(y_ref, entry.c_s - sum(sim_g[pos]))
        end

        # Marker sizes proportional to empirical values (matches bubble_scatter).
        sizes = 300 .* xf ./ maximum(xf)

        # WLS y = b·x (no intercept), weights = x — NON-REFERENCE points ONLY; the
        # reconstructed reference points must not enter the fit.
        w      = xf
        b      = sum(w .* xf .* yf) / sum(w .* xf .^ 2)
        resid  = yf .- b .* xf
        nobs   = length(xf)
        s2     = sum(w .* resid .^ 2) / max(nobs - 1, 1)
        se_b   = sqrt(s2 / sum(w .* xf .^ 2))
        t_stat = se_b > 0 ? b / se_b : Inf

        # Axis limits spanning BOTH series.
        all_x = vcat(xf, x_ref)
        all_y = vcat(yf, y_ref)
        lo = min(minimum(all_x), minimum(all_y)) * 0.9
        hi = max(maximum(all_x), maximum(all_y)) * 1.1
        lims = (lo, hi)

        pg = scatter(xf, yf;
            markersize        = sqrt.(sizes) ./ 2,
            alpha             = 0.6,
            markerstrokecolor = :black,
            markerstrokewidth = 0.5,
            color             = RGB(0.247, 0.404, 0.667),
            label             = "Non-reference",
            xlabel            = "Empirical γ_ls",
            ylabel            = "Simulated γ_ls",
            title             = "γ_ls: Empirical vs Simulated",
            xlims             = lims,
            ylims             = lims,
            legend            = :bottomright,
            grid              = true,
            gridalpha         = 0.5,
            gridstyle         = :dash)

        if !isempty(x_ref)
            scatter!(pg, x_ref, y_ref;
                markersize        = 5,
                markershape       = :diamond,
                alpha             = 0.7,
                markerstrokecolor = :black,
                markerstrokewidth = 0.5,
                color             = RGB(0.75, 0.30, 0.20),
                label             = "Reference (reconstructed)")
        end

        plot!(pg, [lo, hi], [lo, hi]; color=:black, label="45°", linewidth=1)

        annotate!(pg, hi * 0.95, lo + (hi - lo) * 0.12,
                  text(@sprintf("Coef: %.3f", b), :right, 8))
        annotate!(pg, hi * 0.95, lo + (hi - lo) * 0.04,
                  text(@sprintf("t-stat: %.1f", t_stat), :right, 8))
        pg
    end

    p2 = bubble_scatter(emp_pi, sim_pi_r;
        xlabel = "Empirical π_r",
        ylabel = "Simulated π_r",
        title  = "π_r: Empirical vs Simulated")

    p3 = bubble_scatter(emp_pi_sA, sim_pi_sA;
        xlabel = "Empirical π_s",
        ylabel = "Simulated π_s",
        title  = "π_s: Empirical vs Simulated")

    plot(p1, p2, p3, layout=(1, 3), size=(1500, 500), margin=5Plots.mm)
    savefig(joinpath(folder, "dashboard.png"))

    # ── Second plot: best T vs the very-initial T (α in the name) ──
    # Produced at the end of each epoch/stage report so the T distribution's drift
    # from its γ-inversion warm start is tracked alongside the moment dashboard.
    plot_T_vs_initial(best_params, folder; label = "$(stage)")

    # ── Save numpy arrays ──

    npzwrite(joinpath(folder, "pi_r.npy"), results[best_index][2][3])
    npzwrite(joinpath(folder, "productivity.npy"), unpack_params(best_params)[3])

    # ── Text report (unchanged) ──

    agg_labor_share_emp = agg_labor_share
    agg_labor_share_sim = results[best_index][2][1][1]
    agg_labor_share_ = [agg_labor_share_emp, agg_labor_share_sim]

    agg_industry_share_ = [emp_pi_sA,sim_pi_sA]

    gamma_emp_result = matrix_report(emp_gamma, true)
    gamma_sim_result = matrix_report(sim_gamma, true)
    gamma_ls_ = [gamma_emp_result, gamma_sim_result]

    reg_emp = reg_coef
    reg_sim = results[best_index][2][4]
    reg_ = [reg_emp, reg_sim]

    pi_r_emp_result = matrix_report(emp_pi, false)
    pi_r_sim_result = matrix_report(results[best_index][2][3], false)
    pi_r = [pi_r_emp_result, pi_r_sim_result]

    best_score = results[best_index][1][1]

    # Block 6 (granular only): empirical vs realised Ḡ_s(0), plus the profiled N̂_s.
    G0_ = nothing; granular_info = nothing
    if GRANULAR
        try
            G0_emp = vec(empirical_moments)[collect(BLOCK_RANGES[6])]
            G0_sim = vec(results[best_index][2][6])
            G0_    = [G0_emp, G0_sim]
            granular_info = granular_report(best_params)
        catch e
            @warn "count-moment section of report.txt skipped: $e"
        end
    end

    generate_dashboard_report(n, agg_labor_share_, agg_industry_share_, gamma_ls_, reg_, pi_r, best_score,
        folder * "/report.txt", variable, alpha; G0_ = G0_, granular_info = granular_info)
end



rmse(a::AbstractVector, b::AbstractVector) =
    sqrt(mean((a .- b).^2))



############ Reporting functions ##############

"""
    find_stage_folders(epoch_folder)

Find all stage folders within an epoch folder.
Returns sorted list of (stage_number, folder_path) tuples.
"""
function find_stage_folders(epoch_folder::String)
    if !isdir(epoch_folder)
        return Tuple{Int, String}[]
    end
    
    stage_folders = Tuple{Int, String}[]
    
    for item in readdir(epoch_folder)
        item_path = joinpath(epoch_folder, item)
        if isdir(item_path)
            # Try to parse folder name as integer (stage number)
            stage_num = tryparse(Int, item)
            if stage_num !== nothing
                # Check if it contains best_params.npy
                if isfile(joinpath(item_path, "best_params.npy"))
                    push!(stage_folders, (stage_num, item_path))
                end
            end
        end
    end
    
    # Sort by stage number
    sort!(stage_folders, by=x->x[1])
    return stage_folders
end


"""
    find_all_stage_folders(output_folder, max_loop)

Find all stage folders across initial stage and all epochs.
Returns list of folder paths in order.
"""
function find_all_stage_folders(output_folder::String, max_loop::Union{Int, Nothing}=nothing)
    all_folders = String[]
    
    # Initial stage (folder "0")
    initial_folder = joinpath(output_folder, "0")
    if isdir(initial_folder) && isfile(joinpath(initial_folder, "best_params.npy"))
        push!(all_folders, initial_folder)
    end
    
    # Find all epoch folders
    epoch_folders = Tuple{Int, String}[]
    for item in readdir(output_folder)
        if startswith(item, "epoch_")
            epoch_num = tryparse(Int, replace(item, "epoch_" => ""))
            if epoch_num !== nothing
                epoch_path = joinpath(output_folder, item)
                if isdir(epoch_path)
                    push!(epoch_folders, (epoch_num, epoch_path))
                end
            end
        end
    end
    
    # Sort by epoch number
    sort!(epoch_folders, by=x->x[1])
    
    # Apply max_loop filter if specified
    if max_loop !== nothing
        epoch_folders = filter(x -> x[1] <= max_loop, epoch_folders)
    end
    
    # Get stage folders from each epoch
    for (epoch_num, epoch_path) in epoch_folders
        stage_folders = find_stage_folders(epoch_path)
        for (stage_num, stage_path) in stage_folders
            push!(all_folders, stage_path)
        end
    end
    
    return all_folders
end


"""
    compute_scores_modular(output_folder, second_stage, max_loop=nothing)

Compute scores by dynamically finding all stage folders.
"""
function compute_scores_modular(output_folder::String, second_stage::Bool, max_loop::Union{Int, Nothing}=nothing;
                                u_draws::Union{Nothing, Matrix{Float64}}=nothing,
                                sample_weights::Union{Nothing, Matrix{Float64}}=nothing,
                                analytical::Bool=false,
                                n_quad::Int=200)
    top_score = Float64[]
    min_distances = Float64[]
    best_simulated_moments = Vector{Vector{Float64}}()
    best_parameters_list = Vector{Vector{Float64}}()

    # Find all stage folders
    all_folders = find_all_stage_folders(output_folder, max_loop)

    if isempty(all_folders)
        println("Warning: No stage folders found in $output_folder")
        return Float64[], Float64[], Vector{Vector{Float64}}(), Vector{Vector{Float64}}()
    end

    println("Found $(length(all_folders)) stage folders to evaluate")

    for (idx, folder) in enumerate(all_folders)
        # Load best params from this stage
        best_params_stage = NPZ.npzread(joinpath(folder, "best_params.npy"))

        # Handle matrix vs vector
        if ndims(best_params_stage) > 1
            best_params_stage = best_params_stage[:, 1]
        end

        params_list_stage = [best_params_stage]
        params_list_stage, results = train_stage_one(1, nothing, params_list_stage, second_stage;
                                                      u_draws=u_draws, sample_weights=sample_weights,
                                                      analytical=analytical, n_quad=n_quad)
        
        score = [s[1] !== nothing ? s[1][1] : Inf for s in results]
        push!(top_score, minimum(score))
        
        # Compute min_distance
        reg_coef_ = [s !== nothing ? s[2][4] : missing for s in results]
        valid_reg = filter(!ismissing, reg_coef_)
        if !isempty(valid_reg)
            min_distance = minimum(rmse.(valid_reg, Ref(reg_coef)))
            push!(min_distances, min_distance)
        else
            push!(min_distances, Inf)
        end
        
        # Store simulated moments
        if results[1] !== nothing && results[1][2] !== nothing
            simulated_moments = results[1][2]
            simulated_moments_vec = vcat([vec(simulated_moments[i]) for i in 1:length(simulated_moments)]...)
            push!(best_simulated_moments, simulated_moments_vec)
        end
        
        push!(best_parameters_list, best_params_stage)
        
        if idx % 10 == 0
            println("  Evaluated $idx / $(length(all_folders)) stages")
        end
    end
    
    # Normalize to percentage of initial
    if !isempty(top_score) && top_score[1] > 0
        normalized_score = (top_score ./ top_score[1]) .* 100
    else
        normalized_score = top_score
    end
    
    if !isempty(min_distances) && min_distances[1] > 0
        normalized_distances = (min_distances ./ min_distances[1]) .* 100
    else
        normalized_distances = min_distances
    end
    
    return normalized_score, normalized_distances, best_simulated_moments, best_parameters_list
end


"""
    run_reporting(output_folder, max_loop=nothing; save_plots=true)

Run full reporting: compute scores, create plots, save results.
"""
function run_reporting(output_folder::String, max_loop::Union{Int, Nothing}=nothing;
                       save_plots::Bool=true,
                       u_draws::Union{Nothing, Matrix{Float64}}=nothing,
                       sample_weights::Union{Nothing, Matrix{Float64}}=nothing,
                       analytical::Bool=false,
                       n_quad::Int=200)

    folder = output_folder * "/"

    # Compute scores for both stages
    top_score_first, min_dist_first, best_simulated_moments, best_parameters_list =
        compute_scores_modular(output_folder, false, max_loop;
                               u_draws=u_draws, sample_weights=sample_weights,
                               analytical=analytical, n_quad=n_quad)

    
    if !isempty(best_simulated_moments)
        npzwrite(joinpath(folder, "best_simulated_moments.npy"), hcat(best_simulated_moments...))
    end
    
    if !isempty(best_parameters_list)
        npzwrite(joinpath(folder, "best_parameters_list.npy"), hcat(best_parameters_list...))
    end
    
    npzwrite(joinpath(folder, "empirical_moments.npy"), empirical_moments)
    
    # Save score history
    npzwrite(joinpath(folder, "score_history.npy"), Dict(
        "first_stage" => top_score_first,
        "min_dist_first" => min_dist_first
    ))

    return Dict(
        "top_score_first" => top_score_first,
        "min_dist_first" => min_dist_first,
        "best_simulated_moments" => best_simulated_moments,
        "best_parameters_list" => best_parameters_list
    )
end




# Function to find the last epoch and stage folder dynamically
function find_last_stage_folder(base_folder::String)
    # Find all epoch folders
    epoch_folders = Tuple{Int, String}[]
    for item in readdir(base_folder)
        if startswith(item, "epoch_")
            epoch_num = tryparse(Int, replace(item, "epoch_" => ""))
            if epoch_num !== nothing
                epoch_path = joinpath(base_folder, item)
                if isdir(epoch_path)
                    push!(epoch_folders, (epoch_num, epoch_path))
                end
            end
        end
    end
    
    if isempty(epoch_folders)
        # Check for initial stage folder "0"
        initial_folder = joinpath(base_folder, "0")
        if isdir(initial_folder) && isfile(joinpath(initial_folder, "best_params.npy"))
            return initial_folder
        end
        error("No epoch or stage folders found in $base_folder")
    end
    
    # Sort by epoch number and get the last one
    sort!(epoch_folders, by=x->x[1])
    last_epoch_path = epoch_folders[end][2]
    
    # Find stage folders within the last epoch
    stage_folders = Tuple{Int, String}[]
    for item in readdir(last_epoch_path)
        item_path = joinpath(last_epoch_path, item)
        if isdir(item_path)
            stage_num = tryparse(Int, item)
            if stage_num !== nothing
                if isfile(joinpath(item_path, "best_params.npy"))
                    push!(stage_folders, (stage_num, item_path))
                end
            end
        end
    end
    
    if isempty(stage_folders)
        error("No stage folders found in $last_epoch_path")
    end
    
    # Sort by stage number and get the last one
    sort!(stage_folders, by=x->x[1])
    return stage_folders[end][2]
end

"""
    find_resume_state(base_folder)

Scan the output folder to determine the resume point for PSO optimization.
Returns a NamedTuple with:
  - `last_folder`: path to the last completed stage folder
  - `last_epoch`: epoch number (0 if only initial stage exists)
  - `last_stage`: global stage counter value of the last completed stage
  - `resume_loop`: which loop to resume in
  - `resume_substage`: which sub-stage (1, 2, or 3) to resume at within that loop
"""
function find_resume_state(base_folder::String)
    # Collect all epoch folders
    epoch_folders = Tuple{Int, String}[]
    for item in readdir(base_folder)
        if startswith(item, "epoch_")
            epoch_num = tryparse(Int, replace(item, "epoch_" => ""))
            if epoch_num !== nothing
                epoch_path = joinpath(base_folder, item)
                if isdir(epoch_path)
                    push!(epoch_folders, (epoch_num, epoch_path))
                end
            end
        end
    end

    # Case 1: No epoch folders — only the initial stage 0 exists
    if isempty(epoch_folders)
        initial_folder = joinpath(base_folder, "0")
        if isdir(initial_folder) && isfile(joinpath(initial_folder, "best_params.npy"))
            return (last_folder = initial_folder, last_epoch = 0, last_stage = 0,
                    resume_loop = 1, resume_substage = 1)
        end
        error("No completed stages found in $base_folder. Cannot resume.")
    end

    # Case 2: Find the last completed stage across all epochs
    sort!(epoch_folders, by=x->x[1])
    last_epoch_num = epoch_folders[end][1]
    last_epoch_path = epoch_folders[end][2]

    # Find stage folders within the last epoch
    stage_folders = Tuple{Int, String}[]
    for item in readdir(last_epoch_path)
        item_path = joinpath(last_epoch_path, item)
        if isdir(item_path)
            stage_num = tryparse(Int, item)
            if stage_num !== nothing && isfile(joinpath(item_path, "best_params.npy"))
                push!(stage_folders, (stage_num, item_path))
            end
        end
    end

    if isempty(stage_folders)
        error("No completed stage folders found in $last_epoch_path")
    end

    sort!(stage_folders, by=x->x[1])
    last_stage_num = stage_folders[end][1]
    last_stage_path = stage_folders[end][2]

    # Determine sub-stage within the loop: stages are (loop-1)*3 + {1,2,3}
    sub_stage_in_loop = last_stage_num - (last_epoch_num - 1) * 3  # 1, 2, or 3

    if sub_stage_in_loop == 3
        # All 3 sub-stages done for this loop → resume at next loop, sub-stage 1
        resume_loop = last_epoch_num + 1
        resume_substage = 1
    else
        # Resume within the same loop at the next sub-stage
        resume_loop = last_epoch_num
        resume_substage = sub_stage_in_loop + 1
    end

    return (last_folder = last_stage_path, last_epoch = last_epoch_num,
            last_stage = last_stage_num, resume_loop = resume_loop,
            resume_substage = resume_substage)
end



"""
    compute_block_ranges(n_labor, n_industry, n_pi, n_reg, n_gamma, mask; n_G0=0)

Compute indices into masked moment vector for each moment block.
Moment order: [labor | industry | pi_r | reg_coef | gamma_ls (| G0)]

Returns a tuple of 5 index ranges, or 6 when `n_G0 > 0` (granular mode: block 6 is
the per-sector count moment Ḡ_s(0), **appended** so every existing index into the
first five blocks is unchanged).
"""

function compute_block_ranges(n_labor, n_industry, n_pi, n_reg, n_gamma, mask; n_G0::Int = 0)
    sizes = n_G0 > 0 ? [n_labor, n_industry, n_pi, n_reg, n_gamma, n_G0] :
                       [n_labor, n_industry, n_pi, n_reg, n_gamma]
    cuts  = cumsum(vcat(0, sizes))
    nblk  = length(sizes)

    masked_ranges = ntuple(nblk) do k
        base    = count(mask[1 : cuts[k]])
        count_k = count(mask[cuts[k]+1 : cuts[k+1]])
        (base + 1):(base + count_k)
    end

    return masked_ranges
end

"""
    loss_decomposition(params) -> (c, block_totals, total)

Decompose SMM loss into per-moment contributions.

Uses global constants: U_DRAWS, SAMPLE_WEIGHTS, MOMENT_MASK,
                       empirical_moments, Weight_matrix_custom, BLOCK_RANGES
"""
function loss_decomposition(params)
    _, sim = full_SMM(params; u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS)

    sim_vec = moments_to_vec(sim)
    emp_vec = vec(empirical_moments)
    w = diag(Weight_matrix_custom)

    c = w .* (emp_vec .- sim_vec).^2   # per-moment contributions
    block_totals = [sum(c[r]) for r in BLOCK_RANGES]

    return c, block_totals, sum(c)
end

"""
    sigma_beta_gamma_filename(; smm::Bool) -> String

Name of the on-disk β+γ moment covariance to load, keyed on the extensive-margin
regression method (`REG_METHOD`), the moment count (`N_REG`), and the entry point:

  * `aa == true` (γ at the ATTRACTION-AREA level) → `Sigma_aa_beta_gamma…` PREFIX,
    else `Sigma_beta_gamma…`. Defaults from `CA_LEVEL`, since the γ level is what
    semantically drives the file: its γ rows must match the γ moments.
  * `REG_METHOD == :cloglog` → `…_cloglog…`, else nothing
  * `N_REG == 1`             → `…_1…` (single reg_coef stat), else no suffix
  * `smm == true` (SMM path) → trailing `…_f.npy`  (the SMM convention, as used by
    `build_step3_weight_matrix`); `smm == false` (GMM) → `….npy`.

Examples: LPM/SMM/N_REG=4 → `Sigma_beta_gamma_f.npy`; cloglog/SMM/N_REG=1 →
`Sigma_beta_gamma_cloglog_1_f.npy`; cloglog/GMM/N_REG=4 →
`Sigma_beta_gamma_cloglog.npy`; AA/cloglog/SMM/N_REG=4 →
`Sigma_aa_beta_gamma_cloglog_f.npy`.

Every file carries **three blocks in the order β → γ → G**, with `S` rows of
`Ḡ_s(0)`, so on disk `size(Σ) = (N_REG + n_γ + S)²`. The legacy (`!GRANULAR`) mode
slices off the trailing `G` rows/cols; see `reconcile_sigma_data`.
"""
function sigma_beta_gamma_filename(; smm::Bool, aa::Bool = (CA_LEVEL === :aa))
    prefix = aa ? "Sigma_aa_beta_gamma" : "Sigma_beta_gamma"
    link   = REG_METHOD == :cloglog ? "_cloglog" : ""
    n1     = N_REG == 1 ? "_1" : ""
    fsuf   = smm ? "_f" : ""
    return prefix * link * n1 * fsuf * ".npy"
end

"""
    report_granular(theta, output_folder; industry="", label="") -> NamedTuple

Write and print the granular diagnostics at a parameter vector: the profiled variety
count `N̂_s` with its clamp flag, the fitted vs targeted empty-cell share `Ḡ_s(0)`,
the log-z coefficient against `−θ`, the realised supplier-count distribution, and the
second route to the variety count `N^count_s = N_supplier_s / Σ_l q̂_ls`.

Reporting only — no estimate, weight matrix or inference output is affected. Guarded,
so a reporting failure never blocks estimation. A clamp is a rejection signal for the
mechanism rather than a numerical nuisance (validation gate V9), which is why it is
printed per sector; `N̂_s` vs `N^count_s` is the free over-identifying check (V7) and
`b_logz` vs `−θ` the free test of Prop. 1(c) (V6).
"""
function report_granular(theta::Vector{Float64}, output_folder::String;
                         industry::String = "", label::String = "")
    GRANULAR || return nothing
    try
        info = granular_report(theta)
        mkpath(output_folder)

        println("\n" * "="^72)
        println("GRANULAR DIAGNOSTICS" * (isempty(label) ? "" : "  [$label]"))
        println("="^72)
        @printf("  %-8s %8s %8s %8s %8s %10s %10s %10s\n",
                "sector", "N_LO", "N̂_s", "N_HI", "clamp", "G0_fit", "G0_target", "N^count")
        for s in 1:S
            @printf("  %-8d %8d %8d %8d %8s %10.4f %10.4f %10.1f\n",
                    s, N_LO[s], info.N_hat[s], N_HI[s], string(info.clamped[s]),
                    isempty(info.G0) ? NaN : info.G0[s], G_TARGET[s], info.N_count[s])
        end
        println("="^72)

        NPZ.npzwrite(joinpath(output_folder, "granular_diagnostics.npz"),
                     Dict("N_hat"    => Float64.(info.N_hat),
                          "N_LO"     => Float64.(N_LO),
                          "N_HI"     => Float64.(N_HI),
                          "clamped"  => Float64.([c == :none ? 0.0 : (c == :lo ? -1.0 : 1.0)
                                                  for c in info.clamped]),
                          "G0_fit"   => Float64.(info.G0),
                          "G0_target"=> Float64.(collect(G_TARGET)),
                          "N_count"  => Float64.(info.N_count),
                          "q_hat"    => Float64.(info.q_hat),
                          "b_logz"   => [info.b_logz],
                          "EK"       => Float64.(info.EK)))
        open(joinpath(output_folder, "granular_diagnostics.txt"), "w") do io
            println(io, "Granular diagnostics", isempty(industry) ? "" : " — $industry",
                        isempty(label) ? "" : " [$label]")
            println(io, "sector  N_LO  N_hat  N_HI  clamp  G0_fit  G0_target  N_count")
            for s in 1:S
                println(io, "$s  $(N_LO[s])  $(info.N_hat[s])  $(N_HI[s])  $(info.clamped[s])  " *
                            "$(isempty(info.G0) ? NaN : info.G0[s])  $(G_TARGET[s])  $(info.N_count[s])")
            end
            println(io, "b_logz = $(info.b_logz)  (should equal -theta)")
        end
        return info
    catch e
        @warn "granular diagnostics skipped: $e"
        return nothing
    end
end

# `theta` is both a model parameter name and the Fréchet shape constant; this keeps
# the report unambiguous where a local named `theta` shadows the global.
theta_const_for_report() = theta


"""
    inference_moment_indices() -> Vector{Int}

The moments the efficient weight matrix and the α/T inference are built on, in the
invariant **β → γ (→ G)** order: `BLOCK_RANGES[4]`, then `BLOCK_RANGES[5]`, then —
only under `GRANULAR` — the count block `BLOCK_RANGES[6]`. This is the ordering every
Σ file on disk uses; do not reverse it. Extends the historical `gb_indices` without
disturbing it (block 6 is appended, so the β and γ positions are unchanged).
"""
function inference_moment_indices()
    idx = vcat(collect(BLOCK_RANGES[4]), collect(BLOCK_RANGES[5]))
    GRANULAR && append!(idx, collect(BLOCK_RANGES[6]))
    return idx
end


"""
    inference_block_layout() -> (ranges::Tuple, names::Tuple)

Per-block ranges/names *within* the restricted β+γ(+G) subsystem, for the residual
diagnostics of `compute_smm_inference`. Local 1-based coordinates, not global
`BLOCK_RANGES`.
"""
function inference_block_layout()
    n_reg_loc = length(BLOCK_RANGES[4])
    n_gam_loc = length(BLOCK_RANGES[5])
    if GRANULAR
        n_G_loc = length(BLOCK_RANGES[6])
        return ((1:n_reg_loc,
                 (n_reg_loc + 1):(n_reg_loc + n_gam_loc),
                 (n_reg_loc + n_gam_loc + 1):(n_reg_loc + n_gam_loc + n_G_loc)),
                ("reg_coef", "gamma_ls", "G0"))
    else
        return ((1:n_reg_loc, (n_reg_loc + 1):(n_reg_loc + n_gam_loc)),
                ("reg_coef", "gamma_ls"))
    end
end


"""
    reconcile_sigma_data(Sigma_full, input_folder) -> Sigma_data

Reconcile an on-disk moment covariance with the current active set and return it
subset to the moments the estimator actually uses, in **β → γ (→ G) order**
(`BLOCK_RANGES[4]`, `BLOCK_RANGES[5]`, and `BLOCK_RANGES[6]` under `GRANULAR`).

The on-disk layout is FIXED at `N_REG + n_γ + S` (β → γ → G), so the split is a
leading-block slice, not a search: `n_γ_file = size(Σ,1) − N_REG − S`. The order of
operations is (1) split, (2) reconcile the **γ block only** against the active set,
(3) reassemble — keeping the `G` rows/columns only under `GRANULAR`. The `G` rows are
never pruned, and neither is the β block.

| mode | file | blocks kept |
|---|---|---|
| `GRANULAR = false` | ZE-level (`Sigma_beta_gamma*`) | β + γ — `G` dropped |
| `GRANULAR = true`  | AA-level (`Sigma_aa_beta_gamma*`) | β + γ + `Ḡ_s(0)` |

The γ block may have been bootstrapped on the PRE-threshold active set; if a
`gamma_threshold` pruned (s,r) pairs, the matching γ rows/cols are dropped. That
subset branch is ZE-level only (it keys on `X_rs.npy`); under `CA_LEVEL == :aa` the
file must already match the active AA-level γ moments.

NOTE: the subset branch returns Cov(raw γ); the loss uses renormalized γ (factor
`c_s ≈ sum_before/sum_after`), so subset γ rows are over-weighted by ~`c_s^2` and
the resulting T SEs run ~`c_s` too tight. For exact inference, regenerate
`Sigma_beta_gamma` with the threshold applied.

Shared by `build_step3_weight_matrix` (SMM) and `main_gmm.jl` Step 2 (GMM) so the
two paths cannot silently diverge on which moments enter the weight matrix.
"""
function reconcile_sigma_data(Sigma_full::AbstractMatrix, input_folder::String)
    # ── Moment indices in the masked vector (β then γ, then G under GRANULAR) ──
    n_beta  = N_REG
    n_gam   = length(BLOCK_RANGES[5])
    n_G     = S                                     # the file ALWAYS carries S G rows
    n_gb    = n_beta + n_gam + (GRANULAR ? length(BLOCK_RANGES[6]) : 0)

    # ── Split the fixed on-disk layout β → γ → G ─────────────────────────────
    # The file dimension is (N_REG + n_γ_file + S); n_γ_file is whatever the γ level
    # of that file implies (ZE- or AA-level), so it is derived, never searched.
    n_total_file = size(Sigma_full, 1)
    n_gamma_file = n_total_file - n_beta - n_G
    if n_gamma_file <= 0
        # Backward compatibility: a pre-granular file with no G block at all.
        n_gamma_file = n_total_file - n_beta
        n_gamma_file > 0 || error("Σ file has $(n_total_file) rows — too few for " *
            "N_REG=$n_beta plus a γ block. Expected N_REG + n_γ + S (β → γ → G).")
        GRANULAR && error("Σ file has $(n_total_file) rows = N_REG + n_γ with NO G block, " *
            "but GRANULAR=true needs the $S rows of Ḡ_s(0). Regenerate the joint bootstrap " *
            "with all three blocks (β → γ → G).")
        @warn "Σ file carries no G block ($(n_total_file) = N_REG + n_γ). Proceeding in " *
              "legacy (5-block) mode; regenerate it with the β → γ → G layout."
        has_G = false
    else
        has_G = true
    end
    beta_rows  = 1:n_beta
    gamma_rows = (n_beta + 1):(n_beta + n_gamma_file)
    G_rows     = has_G ? ((n_beta + n_gamma_file + 1):(n_beta + n_gamma_file + n_G)) : (1:0)

    # ── Reconcile the γ BLOCK ONLY against the (possibly thresholded) active set ──
    # Under :aa the γ block is AA-level and the files are generated post-hoc for the
    # current active set, so only the use-as-is branch applies; the ZE-level
    # pre-threshold subset branch keys on X_rs.npy, which has no AA counterpart.
    gamma_keep = nothing                                  # nothing ⇒ use the γ block as-is
    if n_gamma_file != n_gam
        CA_LEVEL === :aa && error(
            "Σ γ block has $(n_gamma_file) rows but the active AA-level γ moment count " *
            "is $(n_gam). Regenerate $(sigma_beta_gamma_filename(; smm=true)) for the " *
            "current active set (the pre-threshold subset branch is ZE-level only).")
        X_rs_raw          = NPZ.npzread(joinpath(input_folder, "X_rs.npy"))      # (S,R) raw
        T_mask_moment_old = vec(permutedims(X_rs_raw)) .> 0                       # sector-major
        T_mask_moment_new = collect(T_MASK)                                      # thresholded; s-major (= moment convention)

        # In the old mask, remove reference regions.
        keep_old = copy(T_mask_moment_old)                                       # active − ref/sector
        for s in 1:S
            ref_r = T_REF_REGION[s]
            ref_r > 0 && (keep_old[(s - 1) * T_COL_DIM + ref_r] = false)
        end
        gamma_old_positions = findall(keep_old)                                  # old set (without reference regions)
        survive  = T_mask_moment_new[gamma_old_positions]                        # new set, reference regions removed

        length(gamma_old_positions) == n_gamma_file || error(
            "Σ γ block has $(n_gamma_file) rows, matching neither the active count " *
            "$(n_gam) nor the pre-threshold count $(length(gamma_old_positions)). Regenerate it.")
        gamma_keep = findall(survive)
        length(gamma_keep) == n_gam || error(
            "γ subset gives $(length(gamma_keep)) rows != active n_γ = $(n_gam)")
        # NOTE: subset is Cov(raw γ); loss uses renormalized γ (factor c_s≈sum_before/
        # sum_after). Raw subset over-weights γ rows by ~c_s^2 → T SEs ~c_s too tight.
        # For exact inference, regenerate the Σ file with the threshold applied.
    end

    # ── Reassemble: β + (reconciled) γ, plus G only under GRANULAR ───────────
    keep_idx = vcat(collect(beta_rows),
                    gamma_keep === nothing ? collect(gamma_rows) : collect(gamma_rows)[gamma_keep])
    if GRANULAR
        length(BLOCK_RANGES[6]) == n_G || error(
            "block 6 has $(length(BLOCK_RANGES[6])) moments but the Σ G block has $n_G rows")
        append!(keep_idx, collect(G_rows))          # G rows are NEVER pruned
    end
    Sigma_data = Sigma_full[keep_idx, keep_idx]
    @assert size(Sigma_data, 1) == n_gb "reconciled Σ has $(size(Sigma_data,1)) rows != n_gb=$n_gb"

    @assert isapprox(Sigma_data, Sigma_data'; atol=1e-10) "Sigma is non-symmetric"
    return Sigma_data
end

function build_step3_weight_matrix(theta_hat_1::Vector{Float64}, input_folder::String;
                                   K::Int=10_000,
                                   output_folder::String=".",
                                   draw_method::Symbol=INFERENCE_DRAW_METHOD)
    N_moments = length(empirical_moments)

    # ── Estimated-moment indices in the masked vector: β, γ, and (under
    #    GRANULAR) the count block Ḡ_s(0) — the β → γ → G ordering invariant ──
    gb_indices = inference_moment_indices()
    n_gb = length(gb_indices)

    # ── Load Σ_data: joint bootstrap covariance of β+γ(+G) (β block first) ───
    # File selection keyed on N_REG (moment count) and REG_METHOD (:lpm vs :cloglog).
    # The β-block of Σ_data has N_REG rows/cols (one per reg_coef moment), independent of N_TAU.
    # SMM path ⇒ the `_f` variant (smm=true); cloglog ⇒ the `_cloglog` family.
    sigma_file = sigma_beta_gamma_filename(; smm=true)
    Sigma_full = NPZ.npzread(joinpath(input_folder, sigma_file))

    # ── Reconcile file size with the (possibly thresholded) active set ───────
    # Shared with main_gmm.jl Step 2 so SMM and GMM agree on which moments enter W.
    Sigma_data = reconcile_sigma_data(Sigma_full, input_folder)

    # ── Estimate Σ_sim via K re-seeded SMM evaluations ───────────────────────
    println("Estimating Σ_sim from K=$K SMM evaluations at θ̂_1...")
    flush(stdout)

    M_sim_rows = pmap(1:K) do k
        # Σ_sim uses the INFERENCE draw count (N_RHO_INFERENCE), decoupled from the
        # optimization draw count N_rho (full_SMM sizes itself off size(u_draws,1)).
        u_k, w_k = generate_draws(N_RHO_INFERENCE, n_good, draw_method;
                                  randomise=true, rng=MersenneTwister(k))
        _, moms = full_SMM(theta_hat_1; u_draws=u_k, sample_weights=w_k)
        return moments_to_vec(moms)[gb_indices]
    end
    M_sim = reduce(hcat, M_sim_rows)'   # (K, n_gb)

    Sigma_sim = cov(M_sim; dims=1)

    # ── Combine and invert ───────────────────────────────────────────────────
    Omega = Sigma_data .+ Sigma_sim
    Omega = (Omega .+ Omega') ./ 2

    eig_vals = eigvals(Symmetric(Omega))
    lambda_max = maximum(eig_vals)
    lambda_min = minimum(eig_vals)
    kappa = lambda_max / max(lambda_min, 1e-300)
    println("  Ω condition number: $(round(kappa, sigdigits=4))")

    if kappa > 1e10
        @warn "Ω is ill-conditioned (κ=$kappa). Applying eigenvalue floor at λ_max/1e8."
        floor_val = lambda_max / 1e8
        F = eigen(Symmetric(Omega))
        clipped = max.(F.values, floor_val)
        Omega = F.vectors * Diagonal(clipped) * F.vectors'
        Omega = (Omega .+ Omega') ./ 2
    end

    W_step3 = inv(Omega)

    # ── Save ─────────────────────────────────────────────────────────────────
    step2_dir = joinpath(output_folder, "step2")
    mkpath(step2_dir)

    NPZ.npzwrite(joinpath(step2_dir, "M_sim.npy"),       M_sim)
    NPZ.npzwrite(joinpath(step2_dir, "Sigma_data.npy"),  Sigma_data)
    NPZ.npzwrite(joinpath(step2_dir, "Sigma_sim.npy"),   Sigma_sim)
    NPZ.npzwrite(joinpath(step2_dir, "Omega.npy"),       Omega)
    NPZ.npzwrite(joinpath(step2_dir, "W_step3.npy"),     W_step3)

    open(joinpath(step2_dir, "diagnostics.txt"), "w") do io
        println(io, "K = $K")
        println(io, "lambda_max = $lambda_max")
        println(io, "lambda_min = $lambda_min")
        println(io, "condition_number = $kappa")
    end

    println("  W_step3 ($n_gb × $n_gb) saved to $step2_dir")
    return W_step3
end

"""
    compute_jacobian(theta; K, param_indices, step_rel,
                    output_folder, filename, base_seed) -> (J, J_elast, J_sd, J_elast_sd)

Central finite differences of the masked moment vector w.r.t. selected parameters,
averaged across `K` independent stratified-draw replications.

For each replication k = 1..K:
  - Generate fresh draws via `generate_draws(..., draw_method; randomise=true,
    rng=MersenneTwister(base_seed + k))`.
  - Compute J_k = ∂m(θ; u_k)/∂θ by central FD using a *single* draw configuration
    (so the difference is smooth at fixed u_k — same logic as the deterministic loss
    inside the SMM optimizer).

Returns:
  - J         : mean Jacobian across replications  (N_moments × n_perturb)
  - J_elast   : mean elasticity Jacobian            (N_moments × n_perturb)
  - J_sd      : per-entry s.d. of J across reps     — simulation-noise diagnostic
  - J_elast_sd: per-entry s.d. of J_elast across reps

Saves J, J_elast, J_sd, J_elast_sd, and the param-index map under
`<output_folder>/<output_subdir>/`. Prints per-block max/mean |elasticity| and the
mean-relative-noise ratio σ/|μ| for entries above a magnitude floor.

# Step regime (single log step for every column)
**Every** parameter is perturbed with the **same** scale-invariant **log-space central
step** — there is no additive step. Perturbed points are `θ_j·exp(±δ)` with a single
`δ = step_rel`, and the log-space central difference `(m₊ − m₋)/(2δ)` is converted **back
to raw units** by dividing by `θ_j`:
    J[:,k] = (m₊ − m₋) / (2·δ·θ_j)        # chain rule ∂m/∂θ = (∂m/∂lnθ)·(1/θ)
The log step is scale-invariant (never crosses zero, immune to any additive floor, never
straddles the Fréchet clamp) and is purely a numerical-accuracy device — the **stored
column is raw `∂m/∂θ_j`**, so `G'WG` inference is unaffected. It requires `θ_j > 0` for
every active column (asserted). `J_elast` is derived from raw `J`.

# Arguments
- `theta`         : parameter vector at which to evaluate.
- `K`             : number of independent draw replications to average (default 20).
- `param_indices` : `nothing` → all parameters; otherwise restrict to these columns.
- `step_rel`      : the single dimensionless log step `δ`, identical for every column.
- `base_seed`     : seed offset; replication k uses `MersenneTwister(base_seed + k)`.
                    Must not collide with seeds used elsewhere (e.g. Σ_sim).
- `output_subdir` : subfolder under output_folder for saved files (default "step2").
- `check_symmetry`: print a per-column forward-vs-backward asymmetry diagnostic
                    (default `true`); flags columns whose one-sided slopes diverge by
                    more than 10× the across-replication `J_sd` (nonlinear/clamped regime).
- `richardson_check`: recompute every column at `2δ` and report the relative gap to the `δ`
                    estimate (default `true`); diagnostic only, returned `J` is unchanged.
- `profile_T`     : T-profiling (SMM only). When `true`, every evaluated parameter
                    vector is routed through `profiled_theta` first, so its T block is
                    replaced by the Sinkhorn image `T*(α,Ω,A)`. Perturbing an α/head
                    column then moves T accordingly ⇒ the returned column is the TOTAL
                    derivative `dm/dα` along the profiled manifold ("only α perturbed,
                    T computed accordingly") — no ∂m/∂T-as-free-parameter is formed.
                    Restrict `param_indices` to the head/α columns (a T column is inert,
                    `profiled_theta` overwrites it). Incompatible with `analytical=true`.

# Exact analytical Jacobian (AD)
- `analytical_ad` : with `analytical=true`, compute the closed-form Jacobian by
                    forward-mode automatic differentiation (`ForwardDiff`) instead of
                    finite differences. This is the **true** `∂m/∂θ` of the analytical
                    moments — machine precision, no FD step, no truncation error. `K`,
                    `base_seed`, and the symmetry/Richardson diagnostics are ignored
                    (the moments are deterministic); `J_sd` is exactly zero. Same
                    return shape and saved files as the FD path, so it is a drop-in.
- `ad_validate`   : (with `analytical_ad=true`) run the correctness gates — compute the
                    FD analytical Jacobian once and compare (expected `O(δ²)` gap), then
                    the γ_ls adding-up structural test `Σ_r ∂γ_{r,s}/∂θ = 0`. Print-only.
"""
function compute_jacobian(theta::Vector{Float64};
                          K::Int = 50,
                          param_indices::Union{Nothing, Vector{Int}} = nothing,
                          step_rel::Float64 = 1e-2,
                          output_folder::String = ".",
                          filename::String = "jacobian.npy",
                          base_seed::Int = 0,
                          output_subdir::String = "step2",
                          analytical::Bool = false,
                          analytical_ad::Bool = false,
                          ad_validate::Bool = false,
                          n_quad::Int = 200,
                          check_symmetry::Bool = true,
                          richardson_check::Bool = true,
                          richardson_rel_tol = 0.05,
                          load_existing::Bool = false,
                          profile_T::Bool = false,
                          hold_N_s::Bool = GRANULAR,
                          draw_method::Symbol = INFERENCE_DRAW_METHOD)
    indices   = param_indices === nothing ? collect(1:length(theta)) : param_indices
    n_perturb = length(indices)

    # Fail here, not deep inside the AD tape: the analytical moment vector has no
    # block 6, so under GRANULAR it is one block short of MOMENT_MASK.
    @assert !(analytical && GRANULAR) "compute_jacobian: analytical=true is not " *
        "available under GRANULAR — the analytical extensive margin is the " *
        "FKG-approximated continuum object and has no closed form for the count " *
        "moment Ḡ_s(0) (block 6). Use the simulation Jacobian (analytical=false)."

    # ── Granular: hold N̂_s fixed across the finite-difference perturbations ──
    # N̂_s(θ) is a STEP function of the continuous parameters, so the profiled loss is
    # piecewise smooth with jumps and a central FD can straddle one. At θ̂ the correct
    # object is the derivative at fixed N̂_s anyway (N̂_s is locally constant with
    # probability one), so under GRANULAR the count is pinned at its θ̂ value for the
    # whole Jacobian.
    #
    # The pin is computed on the INFERENCE draw design (the same width the replications
    # use), and its MONTE-CARLO dispersion is measured once here: N̂_s is a bisection on
    # q̂, so independent draw sets give different N̂_s even at a FIXED θ. That dispersion
    # is a property of `N_RHO_INFERENCE`, not of θ sitting on a jump, and it is the
    # number to look at if the count moment looks unstable — raise `N_RHO_INFERENCE`.
    N_fixed = nothing
    if hold_N_s && GRANULAR && !analytical
        n_probe = min(K, 5)
        probes  = pmap(1:n_probe) do k
            u_k, w_k = generate_draws(N_RHO_INFERENCE, n_good, draw_method;
                                      randomise = true,
                                      rng       = MersenneTwister(base_seed + k))
            granular_report(theta; u_draws=u_k, sample_weights=w_k).N_hat
        end
        N_mat   = reduce(hcat, probes)                      # S × n_probe
        N_fixed = [round(Int, median(view(N_mat, s, :))) for s in axes(N_mat, 1)]

        println("Jacobian: holding N̂_s fixed at $(N_fixed) across all FD perturbations.")
        println("  N̂_s Monte-Carlo dispersion over $(n_probe) independent draw sets " *
                "(N_RHO_INFERENCE = $(N_RHO_INFERENCE)):")
        println("    sector      pinned       min       max   spread/pinned")
        worst = 0.0
        for s in axes(N_mat, 1)
            lo, hi = minimum(view(N_mat, s, :)), maximum(view(N_mat, s, :))
            rel    = N_fixed[s] > 0 ? (hi - lo) / N_fixed[s] : 0.0
            worst  = max(worst, rel)
            @printf("    %6d  %10d  %8d  %8d  %14.3f\n", s, N_fixed[s], lo, hi, rel)
        end
        if worst > 0.10
            @warn "N̂_s varies by $(round(100*worst, digits=1))% across independent draw " *
                  "sets at a FIXED θ. That is simulation noise in q̂, not a jump of " *
                  "N̂_s(θ): the count moment Ḡ_s(0) and its Jacobian row carry that " *
                  "noise. Raise N_RHO_INFERENCE (and N_rho, which sets q̂ in the loss)."
        end
        flush(stdout)
    end

    # ── T-profiling: perturb only the head/α; T follows via the Sinkhorn image ──
    # Under `profile_T`, T is NOT a free parameter — it is the deterministic image
    # T*(α,Ω,A) = invert_T_ge(...). So every evaluated parameter vector is first
    # routed through `profiled_theta` (profiling.jl), which overwrites its T block
    # with T*(α,Ω,A). A perturbation of an α (or head) column then moves T
    # accordingly, so the returned column is the TOTAL derivative dm/dα along the
    # profiled manifold — exactly "only α perturbed, T computed accordingly". No
    # ∂m/∂T-as-free-parameter is ever formed. Restrict `param_indices` to the head/α
    # columns when profiling (perturbing a T column is inert — profiled_theta
    # overwrites it — so its column would be ~0). Analytical mode is SMM-only-profiled
    # ⇒ not supported here.
    @assert !(profile_T && analytical) "compute_jacobian: profile_T is not supported with analytical=true (profiling is SMM-only)."

    # ── Optional: load a previously-saved Jacobian instead of recomputing ────────
    # Reads back the exact five files this function writes (J + _elasticity/_sd/
    # _elasticity_sd/_param_indices) from output_folder/output_subdir/filename. Same
    # return shape ⇒ drop-in for any call site. If the primary file is missing we
    # fall through and compute (with a warning), so a first run still populates it.
    if load_existing
        out_dir = joinpath(output_folder, output_subdir)
        jpath   = joinpath(out_dir, filename)
        if isfile(jpath)
            epath   = joinpath(out_dir, replace(filename, ".npy" => "_elasticity.npy"))
            sdpath  = joinpath(out_dir, replace(filename, ".npy" => "_sd.npy"))
            esdpath = joinpath(out_dir, replace(filename, ".npy" => "_elasticity_sd.npy"))
            ipath   = joinpath(out_dir, replace(filename, ".npy" => "_param_indices.npy"))
            J          = NPZ.npzread(jpath)
            J_elast    = isfile(epath)   ? NPZ.npzread(epath)   : zeros(size(J))
            J_sd       = isfile(sdpath)  ? NPZ.npzread(sdpath)  : zeros(size(J))
            J_elast_sd = isfile(esdpath) ? NPZ.npzread(esdpath) : zeros(size(J))
            @assert size(J, 2) == n_perturb (
                "Loaded Jacobian $jpath has $(size(J,2)) columns but $n_perturb params " *
                "were requested — delete the file or fix param_indices.")
            if isfile(ipath)
                saved_idx = vec(Int.(NPZ.npzread(ipath)))
                (length(saved_idx) == length(indices) && all(saved_idx .== collect(indices))) ||
                    @warn "compute_jacobian(load_existing): saved param_indices differ from " *
                          "requested; using the loaded Jacobian as-is (columns assumed aligned)."
            end
            println("Loaded Jacobian from $jpath ($(size(J,1))×$(size(J,2))) — skipping recomputation.")
            return J, J_elast, J_sd, J_elast_sd
        else
            @warn "compute_jacobian(load_existing=true) but $jpath not found — computing from scratch."
        end
    end

    # ── Exact analytical Jacobian via forward-mode AD (no finite-difference step) ──
    # When both flags are set, ForwardDiff differentiates the closed-form analytical
    # moment vector directly — eliminating the FD truncation error that the plain
    # `analytical=true` FD path still carries. Deterministic ⇒ K, base_seed, and the
    # symmetry/Richardson diagnostics are irrelevant here; J_sd is exactly zero.
    if analytical && analytical_ad
        println("Computing EXACT analytical Jacobian via ForwardDiff AD " *
                "($(n_perturb) params, no FD step)...")
        flush(stdout)
        J  = analytical_jacobian_ad(theta, indices; n_quad=n_quad)
        m0 = moments_vec_analytical(theta; n_quad=n_quad)
        N_moments = length(m0)
        @assert size(J, 1) == N_moments (
            "AD Jacobian row count $(size(J,1)) ≠ masked moment count $N_moments")

        if !all(isfinite, J)
            bad = findall(!isfinite, J)
            error("Non-finite AD analytical Jacobian: $(length(bad)) entries; first at " *
                  "(moment=$(bad[1][1]), param=$(indices[bad[1][2]])).")
        end

        J_elast = zeros(N_moments, n_perturb)
        for kk in 1:n_perturb, mm in 1:N_moments
            if abs(m0[mm]) > 1e-12 && abs(theta[indices[kk]]) > 1e-12
                J_elast[mm, kk] = (theta[indices[kk]] / m0[mm]) * J[mm, kk]
            end
        end
        J_sd       = zeros(size(J))
        J_elast_sd = zeros(size(J_elast))

        # Optional correctness gates (plan steps 4–5): AD-vs-FD + γ adding-up.
        if ad_validate
            J_fd, _, _, _ = compute_jacobian(theta;
                K = 1, param_indices = indices, step_rel = step_rel,
                output_folder = output_folder,
                filename = replace(filename, ".npy" => "_fd_check.npy"),
                output_subdir = output_subdir, analytical = true, analytical_ad = false,
                n_quad = n_quad,
                check_symmetry = false, richardson_check = false)
            validate_analytical_jacobian(theta, indices; n_quad = n_quad, J_fd = J_fd)
        end

        out_dir = joinpath(output_folder, output_subdir)
        mkpath(out_dir)
        NPZ.npzwrite(joinpath(out_dir, filename), J)
        NPZ.npzwrite(joinpath(out_dir, replace(filename, ".npy" => "_elasticity.npy")),    J_elast)
        NPZ.npzwrite(joinpath(out_dir, replace(filename, ".npy" => "_sd.npy")),            J_sd)
        NPZ.npzwrite(joinpath(out_dir, replace(filename, ".npy" => "_elasticity_sd.npy")), J_elast_sd)
        NPZ.npzwrite(joinpath(out_dir, replace(filename, ".npy" => "_param_indices.npy")),
                     collect(indices))

        println("\nMean |elasticity| by block (AD-exact analytical Jacobian):")
        for (k, name) in enumerate(BLOCK_NAMES)
            rng = BLOCK_RANGES[k]
            isempty(rng) && continue
            @printf("  %-10s : max=%.4g  mean=%.4g\n", name,
                    maximum(abs.(J_elast[rng, :])), mean(abs.(J_elast[rng, :])))
        end
        return J, J_elast, J_sd, J_elast_sd
    end

    # ── Every column takes the SAME scale-invariant LOG step ─────────────────────
    # There is no additive step: all parameters are perturbed multiplicatively
    # (θ·exp(±δ)) with a single dimensionless step δ = step_rel, and the log-space
    # central difference is converted back to raw units by the chain rule
    # ∂m/∂θ = (∂m/∂lnθ)·(1/θ). The log step never crosses zero, is immune to any
    # additive floor, and never straddles the Fréchet clamp.
    # Layout is [Ω^L(1) | Ω^s(S) | A(R_downstream) | β(N_TAU) | T]; first T flat index
    # (used only by the Richardson per-sector diagnostic below):
    T_first = 1 + S + R_downstream #+ N_TAU +1 ( uncomment to remove alpha)
    # The log step needs θ > 0 (θ·exp(±δ) stays positive and 1/θ stays finite). T_MASK
    # excludes zero T, and the head params (Ω, A, β/α) are positive at any sane estimate
    # — fail loudly if a non-positive column is ever perturbed.
    for k in 1:n_perturb
        @assert theta[indices[k]] > 0 (
            "Log-step Jacobian requires θ > 0, but theta[$(indices[k])] = " *
            "$(theta[indices[k]]); a zero/negative column cannot take a log step.")
    end

    δ = step_rel   # single log step, identical for every column

    # Pre-build perturbed parameter vectors (shared across replications)
    plus_params  = [copy(theta) for _ in 1:n_perturb]
    minus_params = [copy(theta) for _ in 1:n_perturb]
    for (k, j) in enumerate(indices)
        plus_params[k][j]  = theta[j] * exp(δ)
        minus_params[k][j] = theta[j] * exp(-δ)
    end

    println("Computing Jacobian: $K replications × $(2 * n_perturb + 1) evaluations each...")
    flush(stdout)

    # Each replication produces (J_k, J_elast_k). Distribute over k via pmap;
    # within a replication, the 2·n_perturb + 1 evaluations are sequential
    # (same draws u_k, no need to reshuffle between perturbations).
    rep_results = pmap(1:K) do k
        if analytical
            # Analytical mode: deterministic, no simulation draws needed.
            # We still loop K times for API compatibility; K=1 is recommended.
            eval_one = p -> begin
                _, m = full_SMM(p; analytical=true, n_quad=n_quad)
                moments_to_vec(m)
            end
        else
            # Jacobian replications use the INFERENCE draw count (N_RHO_INFERENCE),
            # decoupled from the optimization N_rho (full_SMM sizes off size(u_draws,1)).
            u_k, w_k = generate_draws(N_RHO_INFERENCE, n_good, draw_method;
                                                 randomise = true,
                                                 rng       = MersenneTwister(base_seed + k))
            eval_one = p -> begin
                # Under profiling, reconstruct T = T*(α,Ω,A) before evaluating, so a
                # perturbation of α/head moves T accordingly (total derivative).
                p_eval = profile_T ? profiled_theta(p) : p
                _, m = full_SMM(p_eval; u_draws=u_k, sample_weights=w_k, N_fixed=N_fixed)
                moments_to_vec(m)
            end
        end

        plus_results  = [eval_one(p) for p in plus_params]
        minus_results = [eval_one(p) for p in minus_params]
        m0_k          = eval_one(theta)

        N_moments = length(m0_k)
        J_k       = zeros(N_moments, n_perturb)
        J_elast_k = zeros(N_moments, n_perturb)

        for kk in 1:n_perturb
            # Chain-rule conversion to raw units: ∂m/∂θ = (∂m/∂lnθ)·(1/θ).
            # This division by θ_j keeps the stored column in raw `∂m/∂θ` units
            # so G'WG inference is unchanged in scale.
            J_k[:, kk] = (plus_results[kk] .- minus_results[kk]) ./
                         (2 * δ * theta[indices[kk]])
        end

        for kk in 1:n_perturb, mm in 1:N_moments
            if abs(m0_k[mm]) > 1e-12 && abs(theta[indices[kk]]) > 1e-12
                J_elast_k[mm, kk] = (theta[indices[kk]] / m0_k[mm]) * J_k[mm, kk]
            end
        end

        # Optional forward/backward slopes (raw units) for the symmetry diagnostic.
        J_fwd_k = nothing; J_bwd_k = nothing
        if check_symmetry
            J_fwd_k = zeros(N_moments, n_perturb)
            J_bwd_k = zeros(N_moments, n_perturb)
            for kk in 1:n_perturb
                dp = theta[indices[kk]] * (exp(δ) - 1)    # forward raw increment
                dm = theta[indices[kk]] * (1 - exp(-δ))   # backward raw increment
                J_fwd_k[:, kk] = (plus_results[kk] .- m0_k) ./ dp
                J_bwd_k[:, kk] = (m0_k .- minus_results[kk]) ./ dm
            end
        end

        # Optional Richardson step-doubling on log (T) columns only.
        J_rich_k = nothing
        if richardson_check
            J_rich_k = fill(NaN, N_moments, n_perturb)
            for kk in 1:n_perturb
                j = indices[kk]
                pp = copy(theta); pp[j] = theta[j] * exp(2 * δ)
                mm = copy(theta); mm[j] = theta[j] * exp(-2 * δ)
                J_rich_k[:, kk] = (eval_one(pp) .- eval_one(mm)) ./
                                  (2 * (2 * δ) * theta[j])
            end
        end

        return (J_k, J_elast_k, J_fwd_k, J_bwd_k, J_rich_k)
    end

    # Stack into 3-D arrays and reduce
    N_moments = size(rep_results[1][1], 1)
    J_stack       = Array{Float64}(undef, N_moments, n_perturb, K)
    J_elast_stack = Array{Float64}(undef, N_moments, n_perturb, K)
    for k in 1:K
        J_stack[:, :, k]       = rep_results[k][1]
        J_elast_stack[:, :, k] = rep_results[k][2]
    end

    J          = dropdims(mean(J_stack;       dims=3); dims=3)
    J_elast    = dropdims(mean(J_elast_stack; dims=3); dims=3)
    J_sd       = K > 1 ? dropdims(std(J_stack;       dims=3); dims=3) : zeros(size(J))
    J_elast_sd = K > 1 ? dropdims(std(J_elast_stack; dims=3); dims=3) : zeros(size(J_elast))

    if !all(isfinite, J)
        bad = findall(!isfinite, J)
        # Translate the offending Jacobian column back to its flat parameter index and
        # step regime so a failure names the responsible (moment, param) pair.
        msgs = String[]
        for ci in bad[1:min(end, 10)]
            mm, col = ci[1], ci[2]
            push!(msgs, "(moment=$mm, param=$(indices[col]) [log])")
        end
        error("Non-finite Jacobian: $(length(bad)) entries, first at " *
              join(msgs, ", ") * ". Likely a 0/0 moment " *
              "(collapsed sector) or a clamped tiny-T column.")
    end

    # ── Optional symmetry diagnostic (print-only; returned J unchanged) ───────
    if check_symmetry
        J_fwd = dropdims(mean(cat([rep_results[k][3] for k in 1:K]...; dims=3); dims=3); dims=3)
        J_bwd = dropdims(mean(cat([rep_results[k][4] for k in 1:K]...; dims=3); dims=3); dims=3)
        asym  = abs.(J_fwd .- J_bwd)
        # Threshold: 10× across-replication J_sd where available (K>1), else a
        # |J|-relative fallback (K==1 has no sd to compare against).
        thresh = K > 1 ? (10 .* J_sd) : (0.1 .* abs.(J))
        all_cols = 1:n_perturb
        println("\nForward/backward symmetry check (all columns take the log step):")
        if isempty(all_cols)
            println("  (no columns)")
        else
            for col in all_cols
                flagged = findall(asym[:, col] .> max.(thresh[:, col], 1e-30))
                if isempty(flagged)
                    continue
                end
                worst = flagged[argmax(asym[flagged, col])]
                println("  param $(indices[col]): $(length(flagged)) moment(s) " *
                        "with |fwd−bwd| > threshold; worst at moment $worst " *
                        "(|fwd−bwd|=$(round(asym[worst, col], sigdigits=3)), " *
                        "thresh=$(round(thresh[worst, col], sigdigits=3))) " *
                        "→ nonlinear/clamped regime.")
            end
        end
    end

    # ── Optional Richardson step-doubling diagnostic (print-only) ─────────────
    if richardson_check
        J_rich = dropdims(mean(cat([rep_results[k][5] for k in 1:K]...; dims=3); dims=3); dims=3)
        all_cols = 1:n_perturb

        # Moment→sector map (γ block only; 0 elsewhere) and param→sector map, both
        # rebuilt from structural consts so the diagnostic is self-contained.
        # Unmasked layout: [labor(1) | industry(S) | pi(R_d) | reg(N_REG) | γ(S*R)],
        # γ slot = (s-1)*R + r (s-major, region-minor) — same convention as T.
        gamma_full_offset = 1 + S + R_downstream + N_REG
        n_before_gamma    = count(MOMENT_MASK[1:gamma_full_offset])
        moment_sector     = zeros(Int, size(J, 1))
        let lg = 0
            for s in 1:S, c in 1:T_COL_DIM
                MOMENT_MASK[gamma_full_offset + (s-1)*T_COL_DIM + c] || continue
                lg += 1
                moment_sector[n_before_gamma + lg] = s
            end
        end
        active_T_flat = findall(T_MASK)                  # s-major active order
        param_sector  = idx -> begin                     # 0 for non-T params
            idx < T_first && return 0
            t = idx - T_first + 1
            (t < 1 || t > length(active_T_flat)) && return 0
            ((active_T_flat[t] - 1) ÷ T_COL_DIM) + 1
        end

        println("\nRichardson step-doubling check (J(δ) vs J(2δ), all columns):")
        println("  [worst-gap moment, its sector vs the param's, and |J| there;")
        println("   a large gap on a ~0 CROSS-sector derivative is a benign artefact]")
        if isempty(all_cols)
            println("  (no columns)")
        else
            tot_flag = 0; tot_flag_insec = 0
            for col in all_cols
                s_par = param_sector(indices[col])
                num   = abs.(J[:, col] .- J_rich[:, col])
                den   = max.(abs.(J[:, col]), 1e-30)
                rel   = num ./ den
                mm    = argmax(rel)

                s_mom = moment_sector[mm]
                cls   = s_mom == 0     ? "non-γ" :
                        s_mom == s_par ? "IN-sector" : "CROSS-sector"
                absJ   = abs(J[mm, col])
                sd_ref = K > 1 ? J_sd[mm, col] : NaN

                # Contrast: typical identifying (in-sector γ) derivative magnitude.
                in_rows = findall(==(s_par), moment_sector)
                med_in  = isempty(in_rows) ? NaN : median(abs.(J[in_rows, col]))
                ratio   = (isnan(med_in) || med_in == 0) ? NaN : absJ / med_in

                flagged      = findall(rel .> richardson_rel_tol)
                n_flag       = length(flagged)
                n_flag_insec = count(i -> moment_sector[i] == s_par, flagged)
                tot_flag      += n_flag; tot_flag_insec += n_flag_insec

                @printf("  param %d [sector %d]: max rel gap %s at moment %d [%s]  |J|=%s",
                        indices[col], s_par, string(round(rel[mm], sigdigits=3)),
                        mm, cls, string(round(absJ, sigdigits=3)))
                isnan(med_in) || @printf(" (in-sec med |J|=%s, ratio %s)",
                        string(round(med_in, sigdigits=3)),
                        isnan(ratio) ? "n/a" : string(round(ratio, sigdigits=2)))
                @printf(";  J_sd=%s;  flagged>%g: %d (%d in-sector)\n",
                        isnan(sd_ref) ? "n/a" : string(round(sd_ref, sigdigits=3)),
                        richardson_rel_tol, n_flag, n_flag_insec)
            end
            verdict = tot_flag_insec == 0 ?
                "✓ tous les moments flaggés sont CROSS-sector / non-γ (T identifié par son seul secteur)" :
                "⚠ $tot_flag_insec / $tot_flag moments flaggés sont IN-sector — l'écart n'est PAS purement un artefact hors-secteur"
            println("  → $verdict")
        end
    end

    # ── Save ────────────────────────────────────────────────────────────────
    out_dir = joinpath(output_folder, output_subdir)
    mkpath(out_dir)
    NPZ.npzwrite(joinpath(out_dir, filename), J)
    NPZ.npzwrite(joinpath(out_dir, replace(filename, ".npy" => "_elasticity.npy")),    J_elast)
    NPZ.npzwrite(joinpath(out_dir, replace(filename, ".npy" => "_sd.npy")),            J_sd)
    NPZ.npzwrite(joinpath(out_dir, replace(filename, ".npy" => "_elasticity_sd.npy")), J_elast_sd)
    NPZ.npzwrite(joinpath(out_dir, replace(filename, ".npy" => "_param_indices.npy")),
                 collect(indices))

    # ── Block-level diagnostic ──────────────────────────────────────────────
    println("\nMean |elasticity| across $K replications, by block:")
    println("  (noise = mean σ/|μ| over entries with |μ|>1e-3; high → fixed-draw J was unreliable)")
    for (k, name) in enumerate(BLOCK_NAMES)
        rng = BLOCK_RANGES[k]
        isempty(rng) && continue
        max_e  = maximum(abs.(J_elast[rng, :]))
        mean_e = mean(abs.(J_elast[rng, :]))

        # Noise ratio: avoid blowing up on near-zero entries
        block_mu = abs.(J_elast[rng, :])
        block_sd = J_elast_sd[rng, :]
        signif   = block_mu .> 1e-3
        noise_ratio = any(signif) ? mean(block_sd[signif] ./ block_mu[signif]) : NaN

        println("  $(rpad(name, 10)) : max=$(round(max_e, sigdigits=4))  " *
                "mean=$(round(mean_e, sigdigits=4))  " *
                "noise=$(isnan(noise_ratio) ? "n/a" : string(round(noise_ratio, sigdigits=3)))")
    end
    # α column mapped to the LOCAL perturbation order (`indices`), so this stays
    # in-bounds when `param_indices` is a strict subset (e.g. the profiled α-only
    # Jacobian, whose J has just N_TAU columns). Raw θ layout:
    # [Ω^L(1) | Ω^s(S) | A(R_downstream) | α(N_TAU) | T], so α raw indices are
    # (2+S+R_downstream) … (1+S+R_downstream+N_TAU).
    alpha_raw_lo = 2 + S + R_downstream
    alpha_raw_hi = 1 + S + R_downstream + N_TAU
    alpha_local  = findall(j -> alpha_raw_lo <= indices[j] <= alpha_raw_hi, 1:n_perturb)
    if !isempty(alpha_local)
        α_col = alpha_local[1]
        for (i, m) in enumerate(BLOCK_RANGES[4])
            μ = J_elast[m, α_col]; σ = J_elast_sd[m, α_col]
            @printf("reg_coef[%d] vs α : elast=%.4e  sd=%.4e  ratio=%.2f\n",
                    i, μ, σ, abs(μ) > 1e-300 ? σ/abs(μ) : NaN)
        end
    end

    # ── reg_coef Jacobian noise-to-signal (σ/|μ|) w.r.t. the T parameters ────────
    # The α report above gives the per-moment noise ratio of reg_coef vs the single
    # α column. Here we summarise the FULL reg_coef × T Jacobian block: per entry the
    # noise-to-signal ratio σ/|μ| = J_sd/|J| (the θ/m elasticity rescaling cancels, so
    # it is scale-free). High and dispersed ⇒ the fixed-draw FD Jacobian of the
    # extensive-margin (reg_coef) moments on T is simulation-noise dominated — the
    # motivation for a larger N_RHO_INFERENCE. Print-only; J is unchanged.
    # T columns mapped to the LOCAL perturbation order (`indices`); empty when the
    # Jacobian is a restricted α-only (profiled) one, so the block is skipped then.
    T_raw_lo  = 2 + S + R_downstream + N_TAU
    T_cols_ns = findall(j -> indices[j] >= T_raw_lo, 1:n_perturb)
    if !isempty(BLOCK_RANGES[4]) && !isempty(T_cols_ns)
        sub_mu = abs.(J_elast[BLOCK_RANGES[4], T_cols_ns])
        sub_sd = J_elast_sd[BLOCK_RANGES[4], T_cols_ns]
        signif = sub_mu .> 1e-3
        ns     = sub_sd[signif] ./ sub_mu[signif]
        n_tot  = length(sub_mu); n_sig = count(signif)
        println("\nreg_coef Jacobian noise-to-signal (σ/|μ|) vs T " *
                "($(length(T_cols_ns)) T cols × $(length(BLOCK_RANGES[4])) reg_coef rows):")
        @printf("  entries: %d total, %d with |elast|>1e-3 (%.1f%% usable signal)\n",
                n_tot, n_sig, 100 * n_sig / max(n_tot, 1))
        if !isempty(ns)
            qs = quantile(ns, [0.0, 0.25, 0.5, 0.75, 0.90, 0.99, 1.0])
            @printf("  min=%.3g  q25=%.3g  median=%.3g  mean=%.3g  q75=%.3g  q90=%.3g  q99=%.3g  max=%.3g\n",
                    qs[1], qs[2], qs[3], mean(ns), qs[4], qs[5], qs[6], qs[7])
            @printf("  std=%.3g   share σ/|μ|>1: %.1f%%   share >0.5: %.1f%%\n",
                    std(ns),
                    100 * count(>(1.0), ns) / length(ns),
                    100 * count(>(0.5), ns) / length(ns))
        else
            println("  (no reg_coef × T entries above the |elast|>1e-3 signal floor)")
        end
    end

    return J, J_elast, J_sd, J_elast_sd
end



"""
    compute_smm_inference(theta_hat, J, W, Omega;
                          param_indices, empirical_moments_vec,
                          simulated_moments_vec, output_folder,
                          industry, K_sim)
        -> Dict

Compute parameter SEs (efficient + sandwich), fitted-moment SEs,
moment-residual SEs, and Hansen J-test.

# Arguments
- `theta_hat`             : Vector{Float64}, full parameter vector
- `J`                     : Matrix{Float64} (N_moments × p_active), Jacobian at θ̂_2
- `W`                     : Matrix{Float64} (N_moments × N_moments), weight matrix
- `Omega`                 : Matrix{Float64} (N_moments × N_moments), Σ_data + Σ_sim
- `param_indices`         : Vector{Int}, indices of active parameters in theta_hat
- `empirical_moments_vec` : Vector{Float64}, masked empirical moments
- `simulated_moments_vec` : Vector{Float64}, masked simulated moments at theta_hat
- `output_folder`         : String, outputs go to `output_folder/inference/`
- `industry`              : String, industry label for summary header (default "")
- `K_sim`                 : Int, number of simulator replications used for Σ_sim (default 0)

# Saves (under output_folder/inference/)
- `var_theta_efficient.npy`   : (G'WG)^{-1}
- `var_theta_sandwich.npy`    : (G'WG)^{-1} G'WΩWG (G'WG)^{-1}
- `se_theta.npy`              : √diag(Var_eff)
- `se_theta_sandwich.npy`     : √diag(Var_sandwich)
- `t_stats.npy`               : θ̂_active ./ se_theta
- `ci_95.npy`                 : (p × 2) matrix of [lower, upper]
- `var_theta_regcoef.npy`     : (G'WG)^{-1} G'W Ω_β W G (G'WG)^{-1}  — reg_coef-only variance
- `se_theta_regcoef.npy`      : √diag(Var_regcoef)  — SE from the reg_coef (β) moment error alone
- `t_stats_regcoef.npy`       : θ̂_active ./ se_theta_regcoef
- `ci_95_regcoef.npy`         : (p × 2) [lower, upper] from the reg_coef-only SE
- `se_moments_fitted.npy`     : √diag(J · Var_eff · J')
- `se_moment_residuals.npy`   : √max(diag(Ω - J · Var_eff · J'), 0)
- `J_stat.txt`                : Hansen J statistic, df, p-value
- `inference_summary.txt`     : human-readable diagnostics
"""
function compute_smm_inference(theta_hat::Vector{Float64},
                               J::Matrix{Float64},
                               W::Matrix{Float64},
                               Omega::Matrix{Float64};
                               param_indices::Vector{Int},
                               empirical_moments_vec::Vector{Float64},
                               simulated_moments_vec::Vector{Float64},
                               output_folder::String = ".",
                               industry::String = "",
                               K_sim::Int = 0,
                               block_ranges = BLOCK_RANGES,
                               block_names  = BLOCK_NAMES,
                               gamma_ref_map = nothing,   # NEW: per-sector γ ref reconstruction
                               Var_m_full   = nothing,
                               param_labels  = nothing,   # NEW: names for active params (cols of J)
                               moment_labels = nothing,    # NEW: names for kept moments (rows of J))
                               display_labels = nothing,   # NEW: names of non-inferred params to display (value only)
                               display_values = nothing,   # NEW: values of those non-inferred params (no SE/CI)    # optional override; default uses local Var_m
                               # Extra NAIVE block, printed after the main table. Under the
                               # PROFILED estimator T and N_s are not columns of `J`, so
                               # they carry no SE there. `fd_G` holds their first-difference
                               # Jacobian columns (T as a FREE parameter; N_s by the unit
                               # first difference m(N+1)−m(N)); this function then applies
                               # the SAME naive machinery as the main table — (G'WG)^{-1}
                               # and the sandwich — so the whole file is one method.
                               fd_G      = nothing,
                               fd_labels = nothing,
                               fd_values = nothing,
                               fd_title  = "Naive sandwich inference on the first-difference Jacobian (α, T, N_s free)")

    inf_dir = joinpath(output_folder, "inference")
    mkpath(inf_dir)

    G = J   # (N_moments × p_active)
    p = size(G, 2)
    N_mom = size(G, 1)

    # ── 1. GtWG and its inverse ──────────────────────────────────────────────
    GtWG = Symmetric(G' * W * G)
    eig_floored = false
    floor_val   = 0.0

    GtWG_inv = try
        F = cholesky(GtWG)
        inv(F)
    catch
        @warn "GtWG is not positive-definite; applying eigenvalue floor."
        F = eigen(GtWG)
        floor_val   = F.values[end] * 1e-10
        λ_floored   = max.(F.values, floor_val)
        eig_floored = true
        Symmetric(F.vectors * Diagonal(1.0 ./ λ_floored) * F.vectors')
    end
    GtWG_inv = (Matrix(GtWG_inv) .+ Matrix(GtWG_inv)') ./ 2

    # ── 2. Variances ────────────────────────────────────────────────────────
    Var_eff      = GtWG_inv # Efficient variance. 
    middle       = G' * W * Omega * W * G
    Var_sandwich = Var_eff * middle * Var_eff # Sandwich variance
    Var_sandwich = (Var_sandwich .+ Var_sandwich') ./ 2

    # ── 3. Parameter SEs, t-stats, CIs ──────────────────────────────────────
    se_eff  = sqrt.(max.(diag(Var_eff),      0.0))
    se_sw   = sqrt.(max.(diag(Var_sandwich), 0.0))
    theta_active = theta_hat[param_indices]
    t_stats = theta_active ./ se_sw
    ci_95   = hcat(theta_active .- 1.96 .* se_sw,
                   theta_active .+ 1.96 .* se_sw)

    # ── 3b. reg_coef-only variance: isolate the β (reg_coef) moment error ────────
    # Sandwich variance with Ω replaced by Ω_β — a covariance that is ZERO on every
    # moment block except reg_coef, where it keeps only the DIAGONAL (the reg_coef
    # moment variances). This isolates the part of the parameter covariance driven
    # purely by the reg_coef (β) sampling error, holding all other moment noise
    # (γ_ls, …) at zero — the standard block-contribution decomposition of the
    # sandwich, using the SAME weight W as the main inference:
    #   Var_regcoef = (G'WG)^{-1} · G'W Ω_β W G · (G'WG)^{-1}.
    Omega_beta = zeros(size(Omega))
    rc_pos = findfirst(==("reg_coef"), collect(block_names))
    if rc_pos !== nothing
        for i in block_ranges[rc_pos]
            Omega_beta[i, i] = Omega[i, i]        # diagonal (variance) only
        end
    else
        @warn "compute_smm_inference: no 'reg_coef' block in block_names; reg_coef-only Var is 0."
    end
    middle_beta   = G' * W * Omega_beta * W * G
    Var_regcoef   = Var_eff * middle_beta * Var_eff
    Var_regcoef   = (Var_regcoef .+ Var_regcoef') ./ 2
    se_regcoef    = sqrt.(max.(diag(Var_regcoef), 0.0))
    t_regcoef     = theta_active ./ se_regcoef
    ci_regcoef    = hcat(theta_active .- 1.96 .* se_regcoef,
                         theta_active .+ 1.96 .* se_regcoef)

    # ── 4. Fitted-moment and residual SEs ────────────────────────────────────
    Var_m = G * Var_sandwich * G'
    Var_m = (Var_m .+ Var_m') ./ 2
    se_m_fitted = sqrt.(max.(diag(Var_m), 0.0))

    Var_r_diag = diag(Omega) .- diag(Var_m)
    n_clipped   = count(Var_r_diag .< 0.0)
    if n_clipped > 0
        @warn "$n_clipped negative residual variances to 0."
    end
    se_m_resid = sqrt.(max.(Var_r_diag, 0.0))

    # ── 5. Hansen J-test ─────────────────────────────────────────────────────
    r = empirical_moments_vec .- simulated_moments_vec
    J_stat = (r' * W * r)[1]
    df     = N_mom - p
    pval   = df > 0 ? (1.0 - cdf(Chisq(df), J_stat)) : NaN

    # ── 6. Save arrays ───────────────────────────────────────────────────────
    NPZ.npzwrite(joinpath(inf_dir, "var_theta_efficient.npy"),  Var_eff)
    NPZ.npzwrite(joinpath(inf_dir, "var_theta_sandwich.npy"),   Var_sandwich)
    NPZ.npzwrite(joinpath(inf_dir, "se_theta.npy"),             se_eff)
    NPZ.npzwrite(joinpath(inf_dir, "se_theta_sandwich.npy"),    se_sw)
    NPZ.npzwrite(joinpath(inf_dir, "t_stats.npy"),              t_stats)
    NPZ.npzwrite(joinpath(inf_dir, "ci_95.npy"),                ci_95)
    NPZ.npzwrite(joinpath(inf_dir, "se_moments_fitted.npy"),    se_m_fitted)
    NPZ.npzwrite(joinpath(inf_dir, "se_moment_residuals.npy"),  se_m_resid)
    # reg_coef-only (β moment error) parameter inference: sandwich with Ω_β.
    NPZ.npzwrite(joinpath(inf_dir, "var_theta_regcoef.npy"),    Var_regcoef)
    NPZ.npzwrite(joinpath(inf_dir, "se_theta_regcoef.npy"),     se_regcoef)
    NPZ.npzwrite(joinpath(inf_dir, "t_stats_regcoef.npy"),      t_regcoef)
    NPZ.npzwrite(joinpath(inf_dir, "ci_95_regcoef.npy"),        ci_regcoef)

    # ── 6b. γ_ls fitted-moment plot with SE bars (first dashboard panel) ─────
    # The subsystem passed here is β+γ in β-then-γ order, so the γ block is
    # whichever of `block_ranges` is named "gamma_ls". We index emp/sim/SE by
    # that *local* range — NOT global BLOCK_RANGES[5], which would be wrong.
    # Points and error bars all share this ordering, so they stay aligned.
    # ── 6b. γ_ls fitted-moment plot with SE bars (first dashboard panel) ─────
    # Subsystem is β+γ in β-then-γ order; γ block = whichever block is "gamma_ls".
    gam_pos = findfirst(==("gamma_ls"), collect(block_names))
    if gam_pos !== nothing && !isempty(block_ranges[gam_pos])
        try
            grng     = block_ranges[gam_pos]
            emp_gam  = empirical_moments_vec[grng]
            sim_gam  = simulated_moments_vec[grng]
            se_gam   = se_m_fitted[grng]

            # γ-block submatrix of the fitted-moment covariance (sandwich-based),
            # in LOCAL γ coordinates (1:length(grng)). Needed for ref-region SE,
            # which is a quadratic form over a sector's retained γ moments and so
            # depends on their COVARIANCES, not just their marginal SEs.
            Var_gam = Var_m[grng, grng]

            # Retained (non-reference) points: empirical x, fitted y, ±1 SE.
            keep = emp_gam .> 0
            x_ret = emp_gam[keep]
            y_ret = sim_gam[keep]
            e_ret = se_gam[keep]

            # Reference-region points reconstructed from the within-sector
            # adding-up constraint γ_ref,s = c_s − Σ_{l≠ref} γ_ls.
            x_ref = Float64[]; y_ref = Float64[]; e_ref = Float64[]
            if gamma_ref_map !== nothing
                for entry in gamma_ref_map
                    pos = entry.local_positions          # local γ indices for sector s
                    # guard: positions must lie inside this γ block
                    (isempty(pos) || maximum(pos) > length(grng)) && continue
                    # fitted reference share from the constraint
                    y_r = entry.c_s - sum(sim_gam[pos])
                    # Var(γ_ref) = 1' Var_gam[pos,pos] 1  (includes covariances)
                    v_r = sum(@view Var_gam[pos, pos])
                    se_r = sqrt(max(v_r, 0.0))
                    # only plot if the empirical reference share is positive
                    entry.emp_ref <= 0 && continue
                    push!(x_ref, entry.emp_ref)
                    push!(y_ref, y_r)
                    push!(e_ref, se_r)
                end
            end

            # Combined axis limits over both retained and reference points.
            all_x = vcat(x_ret, x_ref)
            all_lo_y = vcat(y_ret .- e_ret, y_ref .- e_ref)
            all_hi_y = vcat(y_ret .+ e_ret, y_ref .+ e_ref)
            if !isempty(all_x)
                lo = min(minimum(all_x), minimum(all_lo_y)) * 0.9
                hi = max(maximum(all_x), maximum(all_hi_y)) * 1.1

                pγ = scatter(x_ret, y_ret;
                    yerror            = e_ret,
                    markersize        = 4,
                    alpha             = 0.6,
                    markerstrokecolor = :black,
                    markerstrokewidth = 0.5,
                    color             = RGB(0.247, 0.404, 0.667),
                    label             = "Non-reference",
                    xlabel            = "Empirical γ_ls",
                    ylabel            = "Simulated γ_ls",
                    title             = "γ_ls: fitted vs empirical (±1 SE)",
                    xlims             = (lo, hi),
                    ylims             = (lo, hi),
                    grid              = true,
                    gridalpha         = 0.5,
                    gridstyle         = :dash)

                if !isempty(x_ref)
                    # Distinct colour/marker so the reconstructed reference points
                    # (different statistical object) are visually separable.
                    scatter!(pγ, x_ref, y_ref;
                        yerror            = e_ref,
                        markersize        = 5,
                        markershape       = :diamond,
                        alpha             = 0.7,
                        markerstrokecolor = :black,
                        markerstrokewidth = 0.5,
                        color             = RGB(0.75, 0.30, 0.20),
                        label             = "Reference (reconstructed)")
                end

                plot!(pγ, [lo, hi], [lo, hi]; color=:black, label="45°", linewidth=1)
                savefig(pγ, joinpath(inf_dir, "gamma_ls_fitted_se.png"))
                println("  γ_ls fitted-moment SE plot saved to: " *
                        joinpath(inf_dir, "gamma_ls_fitted_se.png") *
                        " ($(length(x_ref)) reference points reconstructed)")
            end
        catch e
            @warn "γ_ls fitted-moment plot failed; continuing." exception=e
        end
    end

    # ── 7. Identification diagnostics ────────────────────────────────────────
    eig_GtWG     = eigvals(GtWG)
    sv_G         = svdvals(G)
    cond_GtWG    = maximum(eig_GtWG) / max(minimum(eig_GtWG), 1e-300)
    rank_G       = count(sv_G .> sv_G[1] * 1e-8)

    # ── 8. Write J_stat.txt ──────────────────────────────────────────────────
    open(joinpath(inf_dir, "J_stat.txt"), "w") do io
        println(io, "Hansen J-test")
        println(io, "  J statistic : $(round(J_stat, sigdigits=6))")
        println(io, "  df          : $df")
        @printf(io, "  p-value     : %.6f\n", isnan(pval) ? -1.0 : pval)
        if isnan(pval)
            println(io, "  verdict     : df ≤ 0 — model is exactly identified")
        elseif pval < 0.05
            println(io, "  verdict     : REJECT H0 at α=0.05 — moment over-identification")
        else
            println(io, "  verdict     : fail to reject H0 at α=0.05")
        end
    end

    # ── 9. Write inference_summary.txt ───────────────────────────────────────
    # Captured by the naive-block closure below and returned, so inference_delta.txt
    # can contrast the structural SEs against the numbers this file actually prints.
    fd_se_sw_out  = nothing
    fd_se_eff_out = nothing
    open(joinpath(inf_dir, "inference_summary.txt"), "w") do io
        # Header
        println(io, "="^72)
        println(io, "SMM INFERENCE SUMMARY")
        println(io, "  Industry   : $(isempty(industry) ? "(not specified)" : industry)")
        println(io, "  θ̂_2 source : $(joinpath(output_folder, "theta_hat_2.npy"))")
        println(io, "  K_sim      : $(K_sim == 0 ? "(not recorded)" : string(K_sim))")
        println(io, "  Date       : $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
        println(io, "="^72)

        # Identification
        println(io, "\n--- Identification check ---")
        λ_min_GtWG = minimum(eig_GtWG)
        λ_max_GtWG = maximum(eig_GtWG)
        warn_eig = λ_min_GtWG < 1e-8 * λ_max_GtWG ? "  *** WARNING: near-singular ***" : ""
        warn_cond = cond_GtWG > 1e10 ? "  *** WARNING: ill-conditioned ***" : ""
        @printf(io, "  λ_min(G'WG)       = %.4e%s\n", λ_min_GtWG, warn_eig)
        @printf(io, "  λ_max(G'WG)       = %.4e\n",   λ_max_GtWG)
        @printf(io, "  cond(G'WG)        = %.4e%s\n", cond_GtWG, warn_cond)
        @printf(io, "  rank(G)           = %d / %d\n", rank_G, p)
        if eig_floored
            @printf(io, "  Eigenvalue floor applied: %.4e\n", floor_val)
        end

        # Parameter table
        println(io, "\n--- Parameter estimates (active parameters) ---")
        # `se_reg(Σβ)` isolates the β (reg_coef) moment error: sandwich with Ω_β
        # (reg_coef diagonal only, 0 elsewhere) — the SE contribution from reg_coef.
        _has_plab = param_labels !== nothing && length(param_labels) == p
        header = _has_plab ?
            @sprintf("  %-22s  %-6s  %-12s  %-12s  %-12s  %-12s  %-8s  %-8s  %-12s  %-12s",
                     "param", "idx", "theta", "se_eff", "se_sw", "se_reg(Σβ)", "ratio", "t-stat", "CI_lo", "CI_hi") :
            @sprintf("  %-6s  %-12s  %-12s  %-12s  %-12s  %-8s  %-8s  %-12s  %-12s",
                     "idx", "theta", "se_eff", "se_sw", "se_reg(Σβ)", "ratio", "t-stat", "CI_lo", "CI_hi")
        println(io, header)
        println(io, "  " * "-"^(length(header)-2))
        for i in 1:p
            ratio = se_eff[i] > 0 ? se_sw[i] / se_eff[i] : NaN
            if _has_plab
                @printf(io, "  %-22s  %-6d  %-12.6f  %-12.6f  %-12.6f  %-12.6f  %-8.4f  %-8.4f  %-12.6f  %-12.6f\n",
                        param_labels[i], param_indices[i],
                        theta_active[i], se_eff[i], se_sw[i], se_regcoef[i],
                        isnan(ratio) ? -999.0 : ratio, t_stats[i],
                        ci_95[i, 1], ci_95[i, 2])
            else
                @printf(io, "  %-6d  %-12.6f  %-12.6f  %-12.6f  %-12.6f  %-8.4f  %-8.4f  %-12.6f  %-12.6f\n",
                        param_indices[i],
                        theta_active[i], se_eff[i], se_sw[i], se_regcoef[i],
                        isnan(ratio) ? -999.0 : ratio, t_stats[i],
                        ci_95[i, 1], ci_95[i, 2])
            end
        end

        # Display-only (non-inferred) parameters: value shown for reference, but
        # inference does not target them, so SE / t-stat / CI columns are blank ("—").
        if display_labels !== nothing && display_values !== nothing &&
           length(display_labels) == length(display_values) && !isempty(display_labels)
            println(io, "\n--- Other parameters (not inferred; value only, no CI) ---")
            if _has_plab
                @printf(io, "  %-22s  %-6s  %-12s  %-12s  %-12s  %-8s  %-8s  %-12s  %-12s\n",
                        "param", "idx", "theta", "se_eff", "se_sw", "ratio", "t-stat", "CI_lo", "CI_hi")
                for j in 1:length(display_labels)
                    @printf(io, "  %-22s  %-6s  %-12.6f  %-12s  %-12s  %-8s  %-8s  %-12s  %-12s\n",
                            display_labels[j], "-", display_values[j],
                            "-", "-", "-", "-", "-", "-")
                end
            else
                @printf(io, "  %-22s  %-12s\n", "param", "theta")
                for j in 1:length(display_labels)
                    @printf(io, "  %-22s  %-12.6f\n", display_labels[j], display_values[j])
                end
            end
        end

        # NAIVE sandwich inference on the first-difference Jacobian, for parameters
        # that are not columns of the (profiled) J above. This file reports ONE
        # method — the mechanical sandwich, Jacobian × moment covariance — so T and
        # N_s are treated here exactly like any free parameter: stack their FD
        # Jacobian columns, form (G'WG)^{-1} and the sandwich. The STRUCTURAL
        # alternative, which respects that T is calibrated by (α̂, γ̂) and N_s by the
        # bisection on Ḡ_s(0), lives in inference_delta.txt.
        if fd_G !== nothing && fd_labels !== nothing && fd_values !== nothing &&
           size(fd_G, 2) == length(fd_labels) && length(fd_labels) == length(fd_values) &&
           !isempty(fd_labels)
            println(io, "\n--- $(fd_title) ---")
            println(io, "  α   : profiled Jacobian columns (as in the main table above).")
            println(io, "  T   : free-parameter FD Jacobian columns ∂m/∂T — T treated as free,")
            println(io, "        NOT as the Sinkhorn image T*(α̂,γ̂).")
            println(io, "  N_s : unit first difference ∂m/∂N_s = m(N_s+1) − m(N_s). N_s is an")
            println(io, "        INTEGER, so one variety IS the step; the SE is in varieties.")
            println(io, "  All three enter ONE G, so this is their JOINT naive variance.")
            Gf  = Matrix{Float64}(fd_G)
            Vef, fl_f = _gtwg_inv(Gf, W)
            fl_f && println(io, "  (G'WG) was not PD; eigenvalue floor applied.")
            Vsf = Vef * (Gf' * W * Omega * W * Gf) * Vef
            Vsf = (Vsf .+ Vsf') ./ 2
            se_ef = sqrt.(max.(diag(Vef), 0.0))
            se_sf = sqrt.(max.(diag(Vsf), 0.0))
            fh = @sprintf("  %-24s  %-12s  %-12s  %-12s  %-8s  %-8s  %-12s  %-12s",
                          "parameters", "estimate", "se_eff", "se_sw", "ratio", "t",
                          "CI_lo", "CI_hi")
            println(io, fh)
            println(io, "  " * "-"^(length(fh)-2))
            for j in eachindex(fd_labels)
                sj = se_sf[j]; vj = fd_values[j]
                rj = se_ef[j] > 0 ? sj / se_ef[j] : NaN
                tj = (isfinite(sj) && sj > 0) ? vj / sj : NaN
                @printf(io, "  %-24s  %-12.6f  %-12.6e  %-12.6e  %-8.4f  %-8.4f  %-12.6f  %-12.6f\n",
                        fd_labels[j], vj, se_ef[j], sj,
                        isnan(rj) ? -999.0 : rj, isnan(tj) ? -999.0 : tj,
                        vj - 1.96 * sj, vj + 1.96 * sj)
            end
            NPZ.npzwrite(joinpath(inf_dir, "se_theta_naive_fd.npy"),          se_sf)
            NPZ.npzwrite(joinpath(inf_dir, "se_theta_naive_fd_efficient.npy"), se_ef)
            fd_se_sw_out  = se_sf
            fd_se_eff_out = se_ef
        end

        # Sandwich vs efficient ratio
        println(io, "\n--- Sandwich/efficient SE ratio (mean close to 1 ⟹ W ≈ Ω^{-1}) ---")
        ratios = [se_eff[i] > 0 ? se_sw[i] / se_eff[i] : NaN for i in 1:p]
        valid  = filter(!isnan, ratios)
        @printf(io, "  Mean ratio: %.4f  |  Max ratio: %.4f\n",
                isempty(valid) ? NaN : mean(valid),
                isempty(valid) ? NaN : maximum(valid))

        # Per-block residual SEs
        println(io, "\n--- Per-block moment residual SEs ---")
        println(io, "  (residual share ≈ 0 ⟹ moment well-fit; ≈ 1 ⟹ weakly used)")
        for (name, rng) in zip(block_names, block_ranges)
            isempty(rng) && continue
            block_resid = se_m_resid[rng]
            block_omega_sd = sqrt.(max.(diag(Omega)[rng], 0.0))
            resid_share = ifelse.(block_omega_sd .> 1e-15,
                                  block_resid ./ max.(block_omega_sd, 1e-15),
                                  fill(NaN, length(rng)))
            @printf(io, "  %-12s  mean_resid_SE=%.4e  max_resid_SE=%.4e  mean_share=%.4f\n",
                    name, mean(block_resid), maximum(block_resid), mean(resid_share))
        end
        # Per-moment residual SEs (only if labels provided and lengths match)
        if moment_labels !== nothing && length(moment_labels) == length(se_m_resid)
            println(io, "\n--- Per-moment residual SE (labelled) ---")
            omega_sd = sqrt.(max.(diag(Omega), 0.0))
            for i in 1:length(se_m_resid)
                share = omega_sd[i] > 1e-15 ? se_m_resid[i] / omega_sd[i] : NaN
                @printf(io, "  %-22s  resid_SE=%.4e  omega_SD=%.4e  share=%.4f\n",
                        moment_labels[i], se_m_resid[i], omega_sd[i],
                        isnan(share) ? -999.0 : share)
            end
        end

        # Hansen J
        println(io, "\n--- Hansen J-test ---")
        @printf(io, "  J statistic : %.6f\n", J_stat)
        @printf(io, "  df          : %d\n",   df)
        if isnan(pval)
            println(io, "  p-value     : N/A (df ≤ 0 — exactly identified)")
        else
            @printf(io, "  p-value     : %.6f\n", pval)
            if pval < 0.05
                println(io, "  verdict     : REJECT H0 at α=0.05")
            else
                println(io, "  verdict     : fail to reject H0 at α=0.05")
            end
        end

        # Caveats
        println(io, "\n--- Caveats ---")
        println(io, "  * SEs are delta-method conditional on the draws used for Σ_sim estimation.")
        println(io, "    A Murphy–Topel correction would account for sequential sampling noise")
        println(io, "    across estimation steps.")
        println(io, "  * Σ_data is non-zero only on the γ_ls and reg_coef blocks. Residual SEs")
        println(io, "    on labor/industry/π_r reflect simulator variance only.")
        println(io, "  * se_reg(Σβ) isolates the reg_coef (β) moment error: it is the sandwich SE")
        println(io, "    computed with Ω replaced by Ω_β — zero on every block except reg_coef,")
        println(io, "    where only the diagonal (the reg_coef moment variances) is kept, and the")
        println(io, "    SAME weight W. It is the part of each parameter's SE attributable purely")
        println(io, "    to the reg_coef sampling error, holding all other moment noise at zero.")
        if eig_floored
            @printf(io, "  * Eigenvalue floor of %.4e was applied to GtWG during inversion.\n",
                    floor_val)
            println(io, "    Efficient and sandwich variances may diverge as a result.")
        end

        println(io, "\n" * "="^72)
    end

    println("Inference complete. Results saved to: $inf_dir")
    println("  Hansen J: stat=$(round(J_stat, sigdigits=5)), df=$df, p=$(isnan(pval) ? "N/A" : round(pval, digits=4))")

    return Dict(
        "Var_eff"       => Var_eff,
        "Var_sandwich"  => Var_sandwich,
        "se_eff"        => se_eff,
        "se_sw"         => se_sw,
        "se_regcoef"    => se_regcoef,
        "Var_regcoef"   => Var_regcoef,
        "t_regcoef"     => t_regcoef,
        "ci_regcoef"    => ci_regcoef,
        "t_stats"       => t_stats,
        "ci_95"         => ci_95,
        "se_m_fitted"   => se_m_fitted,
        "se_m_resid"    => se_m_resid,
        "J_stat"        => J_stat,
        "df"            => df,
        "pval"          => pval,
        "se_fd_sw"      => fd_se_sw_out,
        "se_fd_eff"     => fd_se_eff_out,
        "fd_labels"     => fd_labels,
    )
end


"""
    _gamma_block_rs() -> Vector{Tuple{Int,Int}}

The `(column, s)` pairs of the γ moment block (`BLOCK_RANGES[5]`), in the SAME order
as that block: sector-major, column-minor, active & non-reference. The column is an
upstream ZE under `CA_LEVEL == :ze` and an attraction area under `:aa`. Reconstructed
from `MOMENT_MASK` (the ground truth), so it stays aligned with the γ rows/cols of
`Sigma_data` and with the empirical `EMP_GAMMA_T` entries. In the full moment layout
(labor, industry, π_r, reg_coef, γ [, G0]) the γ block occupies `T_COL_DIM*S` slots
ending just before block 6; slot `(s-1)*T_COL_DIM + c` maps to `EMP_GAMMA_T[c, s]`.
"""
function _gamma_block_rs()
    # 0-based offset to the γ portion of the FULL (unmasked) vector. Under GRANULAR the
    # γ block is no longer last — block 6 (S entries of Ḡ_s(0)) is appended after it.
    off = length(MOMENT_MASK) - T_COL_DIM * S - (GRANULAR ? S : 0)
    rs  = Tuple{Int,Int}[]
    for s in 1:S, c in 1:T_COL_DIM
        MOMENT_MASK[off + (s - 1) * T_COL_DIM + c] || continue
        push!(rs, (c, s))
    end
    return rs
end


"""
    _dTstar_dalpha(theta_hat, gb_param_idx, param_labels_gb; step_rel, target)
        -> NamedTuple

Central LOG-step finite difference of the GE-Sinkhorn image `T*(α, Ω, A)` w.r.t. α,
deterministic (no draws). Perturbs `α·exp(±δ)`, re-solves `invert_T_ge`, and
chain-rules back to raw α units by `1/α`. Returns the identified-T × N_TAU Jacobian
`J_T = ∂T*/∂α` plus the bookkeeping the delta-method callers need (α/T column
positions inside the gb parameter vector, the reduced-T→identified map, the
normalized head, and the reported T̂). Shared by `compute_T_delta_inference` (the
α-only counterfactual on the joint path) and `compute_profiled_T_inference` (the
correlated α+γ CI on the profiled path), so the FD convention cannot drift between
them. Returns `nothing` if there is no α or no T column.
"""
function _dTstar_dalpha(theta_hat::Vector{Float64},
                        gb_param_idx::Vector{Int},
                        param_labels_gb::Vector{String};
                        step_rel::Float64 = 1e-2,
                        target::AbstractMatrix = EMP_GAMMA_T)
    alpha_pos = findall(l -> startswith(l, "alpha"), param_labels_gb)
    T_pos     = findall(l -> startswith(l, "T["),   param_labels_gb)
    (isempty(alpha_pos) || isempty(T_pos)) && return nothing
    n_alpha = length(alpha_pos)
    n_T     = length(T_pos)

    # Head + α at θ̂, normalized exactly as invert_T_ge / unpack_params read them.
    ΩL, Ωs, A, α, Tvec = unpack_params(theta_hat)
    T_hat_mat    = unpack_T_par(theta_hat)           # (S, T_COL_DIM) ref-normalized warm start
    T_hat_active = theta_hat[gb_param_idx[T_pos]]     # reported T̂ (CI centers)

    # `vec(permutedims(T*))[T_MASK]` gives ALL active T entries (s-major, incl. the
    # per-sector reference regions). The gb Jacobian columns (gb_param_idx[T_pos])
    # are the *identified* T params (reference regions dropped); map full-reduced-T
    # positions → identified columns via the raw layout offset.
    T_offset  = 1 + S + R_downstream + N_TAU          # raw index just before the reduced-T block
    T_red_pos = gb_param_idx[T_pos] .- T_offset        # positions within vec(permutedims(·))[T_MASK]
    @assert all(1 .<= T_red_pos .<= sum(T_MASK)) (
        "T reduced-position mapping out of range: got $(extrema(T_red_pos)), " *
        "reduced-T length $(sum(T_MASK)) — layout offset misaligned.")

    δ   = step_rel
    J_T = zeros(n_T, n_alpha)
    for jj in 1:n_alpha
        αp = copy(α); αp[jj] *= exp(δ)
        αm = copy(α); αm[jj] *= exp(-δ)
        resP = invert_T_ge(αp, ΩL, Ωs, A; target=target, T_init=copy(T_hat_mat))
        resM = invert_T_ge(αm, ΩL, Ωs, A; target=target, T_init=copy(T_hat_mat))
        (resP.converged && resM.converged) ||
            @warn "_dTstar_dalpha: invert_T_ge did not converge for α[$jj] " *
                  "(resid₊=$(round(resP.resid, sigdigits=3)), resid₋=$(round(resM.resid, sigdigits=3)))."
        dT_full = (vec(permutedims(resP.T))[T_MASK] .- vec(permutedims(resM.T))[T_MASK]) ./ (2δ * α[jj])
        J_T[:, jj] = dT_full[T_red_pos]                # restrict to the identified T columns
    end

    return (J_T = J_T, alpha_pos = alpha_pos, T_pos = T_pos, T_red_pos = T_red_pos,
            n_alpha = n_alpha, n_T = n_T, ΩL = ΩL, Ωs = Ωs, A = A, α = α,
            T_hat_mat = T_hat_mat, T_hat_active = T_hat_active)
end


"""
    compute_T_delta_inference(theta_hat, inf_result, gb_param_idx, param_labels_gb;
                              output_folder, industry="", step_rel=1e-2,
                              target=EMP_GAMMA_T) -> Dict

Delta-method confidence intervals for the comparative-advantage block **T**,
treating T as the deterministic GE-Sinkhorn image `T*(α, Ω, A)` of the OTHER
parameters (via `invert_T_ge`) rather than as a free parameter. Here **Θ = α
only**: Ω^L, Ω^s, A are held fixed at their θ̂ values — they ARE fixed in the SMM's
`gamma_beta_only` final step — so all of T's inherited uncertainty flows from
Var(α̂):

    V_T = (∂T*/∂α) · Var(α̂) · (∂T*/∂α)'

`∂T*/∂α` is a fixed-draw-free central **log-step** finite difference on
`invert_T_ge` (perturb α·exp(±δ), chain-rule back to raw α units by 1/α — the same
convention as `compute_jacobian`). Var(α̂) is the α-block of the parameter
covariance already produced by `compute_smm_inference` (passed via `inf_result`);
we propagate the **sandwich**, **efficient**, AND **reg_coef-only** α-variances
(the last from Ω_β — reg_coef diagonal only, 0 elsewhere), so `se_delta_reg`
isolates the part of T's inherited width due to the reg_coef (β) moment error
alone. `V_T` has rank ≤ N_TAU (rank 1 if N_TAU==1): with Ω,A fixed, T lives on the
N_TAU-dimensional curve traced by α.

This is the closed-form counterpart of "at zero data/sim noise, T's CI is entirely
inherited from the other parameters' CI". It is exact for the profiled estimator
(`profile_T=true`, where T̂ = T*(α̂,Ω̂,Â)); under the joint estimator T̂ may differ
from T*(α̂), so read it as "if T were pinned to the Sinkhorn image".

**ADDITIVE** — does not touch the joint (T-as-free) CIs from
`compute_smm_inference`.

# Saves (under output_folder/inference/)
- `se_theta_T_delta.npy`          : √diag(V_T) using the sandwich Var(α̂)
- `se_theta_T_delta_eff.npy`      : … using the efficient Var(α̂)
- `se_theta_T_delta_regcoef.npy`  : … using the reg_coef-only Var(α̂) (Ω_β)
- `var_theta_T_delta.npy`         : V_T (sandwich)
- `t_stats_T_delta.npy`           : T̂ ./ se_delta  (t-stat from the delta-method SE)
- `ci_95_T_delta.npy`             : (n_T × 2) [lower, upper] = T̂ ± 1.96·se_delta
- `jacobian_T_wrt_alpha.npy`      : ∂T*/∂α  (n_T × N_TAU)
- `inference_T_delta.txt`         : human-readable summary + vs-joint comparison
"""
function compute_T_delta_inference(theta_hat::Vector{Float64},
                                   inf_result::Dict,
                                   gb_param_idx::Vector{Int},
                                   param_labels_gb::Vector{String};
                                   output_folder::String = ".",
                                   industry::String = "",
                                   step_rel::Float64 = 1e-2,
                                   target::AbstractMatrix = EMP_GAMMA_T)
    inf_dir = joinpath(output_folder, "inference")
    mkpath(inf_dir)

    # ── ∂T*/∂α by central LOG-step FD on invert_T_ge (deterministic, no draws) ──
    dTa = _dTstar_dalpha(theta_hat, gb_param_idx, param_labels_gb;
                         step_rel=step_rel, target=target)
    if dTa === nothing
        @warn "compute_T_delta_inference: no α or no T columns in gb params; skipping."
        return Dict{String,Any}()
    end
    alpha_pos    = dTa.alpha_pos
    T_pos        = dTa.T_pos
    n_alpha      = dTa.n_alpha
    n_T          = dTa.n_T
    T_hat_active = dTa.T_hat_active
    J_T          = dTa.J_T

    # ── Propagate the α-variance (sandwich / efficient / reg_coef-only Ω_β) ──────
    Va_sw  = inf_result["Var_sandwich"][alpha_pos, alpha_pos]
    Va_eff = inf_result["Var_eff"][alpha_pos, alpha_pos]
    VT_sw  = J_T * Va_sw  * J_T';  VT_sw  = (VT_sw  .+ VT_sw')  ./ 2
    VT_eff = J_T * Va_eff * J_T';  VT_eff = (VT_eff .+ VT_eff') ./ 2
    se_T_sw  = sqrt.(max.(diag(VT_sw),  0.0))
    se_T_eff = sqrt.(max.(diag(VT_eff), 0.0))

    se_T_reg = fill(NaN, n_T)
    if haskey(inf_result, "Var_regcoef")
        Va_reg = inf_result["Var_regcoef"][alpha_pos, alpha_pos]
        VT_reg = J_T * Va_reg * J_T'
        se_T_reg = sqrt.(max.(diag(VT_reg), 0.0))
    end

    # t-stat and 95% CI from the delta-method SE (se_delta), centered at T̂.
    t_T_delta = [se_T_sw[i] > 0 ? T_hat_active[i] / se_T_sw[i] : NaN for i in 1:n_T]
    ci_T = hcat(T_hat_active .- 1.96 .* se_T_sw, T_hat_active .+ 1.96 .* se_T_sw)

    # Joint (T-as-free) sandwich SE for the same T params, for comparison.
    se_T_joint = inf_result["se_sw"][T_pos]

    # ── Save arrays ─────────────────────────────────────────────────────────────
    NPZ.npzwrite(joinpath(inf_dir, "se_theta_T_delta.npy"),          se_T_sw)
    NPZ.npzwrite(joinpath(inf_dir, "se_theta_T_delta_eff.npy"),      se_T_eff)
    NPZ.npzwrite(joinpath(inf_dir, "se_theta_T_delta_regcoef.npy"),  se_T_reg)
    NPZ.npzwrite(joinpath(inf_dir, "var_theta_T_delta.npy"),         VT_sw)
    NPZ.npzwrite(joinpath(inf_dir, "t_stats_T_delta.npy"),           t_T_delta)
    NPZ.npzwrite(joinpath(inf_dir, "ci_95_T_delta.npy"),             ci_T)
    NPZ.npzwrite(joinpath(inf_dir, "jacobian_T_wrt_alpha.npy"),      J_T)

    # ── Human-readable summary ──────────────────────────────────────────────────
    T_labels = param_labels_gb[T_pos]
    sv_JT    = svdvals(J_T)
    rank_JT  = count(sv_JT .> (isempty(sv_JT) ? 0.0 : sv_JT[1] * 1e-8))
    ratios   = [se_T_joint[i] > 0 ? se_T_sw[i] / se_T_joint[i] : NaN for i in 1:n_T]
    valid_r  = filter(!isnan, ratios)

    open(joinpath(inf_dir, "inference_T_delta.txt"), "w") do io
        println(io, "="^72)
        println(io, "DELTA-METHOD T INFERENCE   V_T = (∂T*/∂α) Var(α̂) (∂T*/∂α)'")
        println(io, "  Industry   : $(isempty(industry) ? "(not specified)" : industry)")
        println(io, "  Θ scope    : α only (Ω^L, Ω^s, A held fixed at θ̂)")
        println(io, "  n_T        : $n_T   n_α (N_TAU) : $n_alpha   rank(∂T*/∂α) : $rank_JT")
        println(io, "  Date       : $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
        println(io, "="^72)
        println(io, "\nT is the deterministic GE-Sinkhorn image T*(α,Ω,A) (invert_T_ge). With")
        println(io, "Ω,A fixed, T's uncertainty is entirely inherited from Var(α̂), so V_T has")
        println(io, "rank ≤ N_TAU: the T SEs below are perfectly collinear across entries when")
        println(io, "N_TAU==1. Exact for the profiled estimator (T̂ = T*(α̂)); a Sinkhorn-pinned")
        println(io, "counterfactual under the joint estimator.\n")
        println(io, "  se_delta      : √diag(V_T), sandwich Var(α̂)  — the headline delta SE")
        println(io, "  se_delta_eff  : … efficient Var(α̂)")
        println(io, "  se_delta_reg  : … reg_coef-only Var(α̂) (Ω_β: reg_coef diag, 0 elsewhere)")
        println(io, "  se_joint      : the T-as-free sandwich SE from compute_smm_inference")
        println(io, "  ratio         : se_delta / se_joint  (≪1 ⟹ pinning T to T*(α) tightens a lot)")
        println(io, "  t_delta, CI   : t-stat = T̂/se_delta and 95% CI = T̂ ± 1.96·se_delta\n")
        hdr = @sprintf("  %-24s  %-12s  %-12s  %-12s  %-12s  %-12s  %-8s  %-8s  %-12s  %-12s",
                       "T[sector-region]", "T̂", "se_delta", "se_delta_eff",
                       "se_delta_reg", "se_joint", "ratio", "t_delta", "CI_lo", "CI_hi")
        println(io, hdr)
        println(io, "  " * "-"^(length(hdr)-2))
        for i in 1:n_T
            @printf(io, "  %-24s  %-12.6f  %-12.6e  %-12.6e  %-12.6e  %-12.6e  %-8.4f  %-8.4f  %-12.6f  %-12.6f\n",
                    T_labels[i], T_hat_active[i], se_T_sw[i], se_T_eff[i],
                    se_T_reg[i], se_T_joint[i], isnan(ratios[i]) ? -999.0 : ratios[i],
                    isnan(t_T_delta[i]) ? -999.0 : t_T_delta[i], ci_T[i, 1], ci_T[i, 2])
        end
        println(io, "\n--- Summary (se_delta / se_joint over T) ---")
        @printf(io, "  mean ratio: %.4f   median: %.4f   min: %.4f   max: %.4f\n",
                isempty(valid_r) ? NaN : mean(valid_r),
                isempty(valid_r) ? NaN : median(valid_r),
                isempty(valid_r) ? NaN : minimum(valid_r),
                isempty(valid_r) ? NaN : maximum(valid_r))
        @printf(io, "  mean se_delta: %.4e   mean se_delta_reg: %.4e   mean se_joint: %.4e\n",
                mean(se_T_sw), mean(se_T_reg), mean(se_T_joint))
        println(io, "\n--- Caveats ---")
        println(io, "  * Θ = α only: Ω^L, Ω^s, A are treated as fixed (their SMM-final-step status).")
        println(io, "    A full-head delta method would add their covariance — not propagated here.")
        println(io, "  * ∂T*/∂α is a deterministic FD on invert_T_ge; simulation noise enters through")
        println(io, "    Var(α̂). se_delta_reg propagates ONLY the reg_coef (β) moment error into α̂,")
        println(io, "    isolating T's CI width due to the reg_coef error alone.")
        println(io, "\n" * "="^72)
    end

    println("Delta-method T inference saved to: $(joinpath(inf_dir, "inference_T_delta.txt"))")
    @printf("  mean se_delta/se_joint over T = %.4f  (rank ∂T*/∂α = %d)\n",
            isempty(valid_r) ? NaN : mean(valid_r), rank_JT)

    return Dict(
        "J_T_wrt_alpha" => J_T,
        "Var_T_delta"   => VT_sw,
        "se_T_delta"    => se_T_sw,
        "se_T_delta_eff"=> se_T_eff,
        "se_T_delta_reg"=> se_T_reg,
        "t_T_delta"     => t_T_delta,
        "ci_T_delta"    => ci_T,
        "se_T_joint"    => se_T_joint,
    )
end


"""
    _gtwg_inv(G, W) -> (Matrix, floored::Bool)

`(G'WG)^{-1}` — the efficient GMM variance — with the same PD guard
`compute_smm_inference` uses: Cholesky, falling back to an eigenvalue floor when the
information matrix is not positive definite. Returned symmetrised.
"""
function _gtwg_inv(G::AbstractMatrix, W::AbstractMatrix)
    GtWG = Symmetric(Matrix(G)' * Matrix(W) * Matrix(G))
    floored = false
    Vi = try
        inv(cholesky(GtWG))
    catch
        F = eigen(GtWG)
        floored = true
        Symmetric(F.vectors * Diagonal(1.0 ./ max.(F.values, F.values[end] * 1e-10)) * F.vectors')
    end
    M = Matrix(Vi)
    return (M .+ M') ./ 2, floored
end


"""
    compute_N_s_jacobian(theta; N_hat=nothing, K=10, base_seed=7_000_000,
                         draw_method=INFERENCE_DRAW_METHOD, profile_T=false)
        -> (G_N::Matrix, G_N_sd::Matrix, N_hat::Vector{Int}, diff::Vector{Float64})

`∂m/∂N_s` — the moment Jacobian w.r.t. the profiled variety counts — by a **unit
first difference** `m(N̂_s + 1) − m(N̂_s)`. `N_s` is an INTEGER, so a first difference
in one variety is the derivative; there is no step size to choose and no log step.

The two evaluations of a replication share the same draws, so every moment that does
not depend on `N_s` differences to EXACTLY zero. Under the profiled design only block
6 (`Ḡ_s(0) = mean_l (1−q̂_ls)^{N_s}`) carries an `N_s` term, and `Ḡ_s` depends on
`N_s` alone — block 4 is exactly `N_s`-free by Proposition 1 (gate V10). The Jacobian
block is therefore DIAGONAL: one nonzero per column, at that sector's `G0` row. That
lets all `S` columns be recovered from a single simultaneous bump of every `N̂_s`, and
it is **asserted**, not assumed: a nonzero difference outside block 6, or off the
sector's own `G0` row, means an `N_s` dependence the design does not have.

Returns the `(N_moments × S)` Jacobian, its across-replication SD, the `N̂_s` used,
and the raw mean difference vector.
"""
function compute_N_s_jacobian(theta::Vector{Float64};
                              N_hat::Union{Nothing,Vector{Int}} = nothing,
                              K::Int = 10,
                              base_seed::Int = 7_000_000,
                              draw_method::Symbol = INFERENCE_DRAW_METHOD,
                              profile_T::Bool = false)
    @assert GRANULAR "compute_N_s_jacobian is only defined under GRANULAR=true " *
        "(N_s exists only in the granular model)."
    theta_eval = profile_T ? profiled_theta(theta) : theta
    N0 = N_hat === nothing ?
        granular_report(theta_eval; u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS).N_hat :
        copy(N_hat)
    N1 = N0 .+ 1

    diffs = pmap(1:K) do k
        u_k, w_k = generate_draws(N_RHO_INFERENCE, n_good, draw_method;
                                  randomise = true,
                                  rng       = MersenneTwister(base_seed + k))
        _, m0 = full_SMM(theta_eval; u_draws=u_k, sample_weights=w_k, N_fixed=N0)
        _, m1 = full_SMM(theta_eval; u_draws=u_k, sample_weights=w_k, N_fixed=N1)
        moments_to_vec(m1) .- moments_to_vec(m0)
    end

    D      = reduce(hcat, diffs)                       # N_moments × K
    d_mean = vec(mean(D, dims = 2))
    d_sd   = K > 1 ? vec(std(D, dims = 2)) : zeros(length(d_mean))

    g_rng = BLOCK_RANGES[6]
    @assert length(g_rng) == S "block 6 has $(length(g_rng)) rows, expected S = $S"
    off_rows = setdiff(1:length(d_mean), collect(g_rng))
    bad = findall(i -> D[i, :] != zeros(K), off_rows)
    isempty(bad) || error("compute_N_s_jacobian: bumping N̂_s moved $(length(bad)) " *
        "moment(s) OUTSIDE block 6 (first: row $(off_rows[bad[1]])). Only Ḡ_s(0) may " *
        "depend on N_s — block 4 is N_s-free by Prop. 1 (gate V10). This is a bug in " *
        "the moment code, not a numerical tolerance issue.")

    G_N    = zeros(length(d_mean), S)
    G_N_sd = zeros(length(d_mean), S)
    for (s, row) in enumerate(g_rng)
        G_N[row, s]    = d_mean[row]
        G_N_sd[row, s] = d_sd[row]
    end
    return G_N, G_N_sd, N0, d_mean
end


"""
    compute_profiled_T_inference(theta_hat, gb_param_idx, param_labels_gb;
        dTa, G_alpha, W, Sigma_data, Var_alpha_sw, Var_alpha_eff,
        Var_alpha_reg=nothing, gb_block_ranges,
        output_folder, industry="", step_rel=1e-2, target=EMP_GAMMA_T) -> Dict

**Profiling-path** confidence intervals for the comparative-advantage block **T**,
propagating BOTH the estimation error of α̂ AND the DATA noise of the γ_ls target
(the Sinkhorn moments), with their correlation — the fully-correlated delta method.

Under profiling, T is the deterministic GE-Sinkhorn image `T̂ = T*(α̂, γ̂)` where γ̂
is the empirical `emp_gamma_ls` target of `invert_T_ge`. Both α̂ and γ̂ are functions
of the same β+γ data moments, so

    T̂ − T ≈ f_α·(α̂ − α) + f_γ·(γ̂ − γ),

with `f_α = ∂T*/∂α` (from `dTa.J_T`) and `f_γ = ∂T*/∂γ` (central log-step FD on
`invert_T_ge`, perturbing each `EMP_GAMMA_T[c,s]` target entry). The joint
covariance of `(α̂, γ̂)` is

    [ Var(α̂)            Cov(α̂, γ̂) ]
    [ Cov(α̂, γ̂)'        Σ_γγ       ]

where `Var(α̂)` is the profiled α variance (`compute_smm_inference` on the reduced
α-Jacobian `G_alpha`), `Σ_γγ = Sigma_data[γ, γ]` is the DATA (bootstrap) covariance
of the γ moments, and `Cov(α̂, γ̂) = P·Sigma_data[:, γ]` with the influence
`P = Var_eff(α̂)·G_alpha'·W` = `dα̂/dm`. This joint matrix is PSD by construction
(it is the covariance of a linear image of the independent data / simulation noise),
so `V_T = [f_α f_γ]·JointCov·[f_α f_γ]'` is a proper covariance:

    V_T = f_α Var(α̂) f_α' + f_γ Σ_γγ f_γ' + f_α Cov(α̂,γ̂) f_γ' + (·)'

The α channel uses the reported (sandwich) `Var(α̂)`; the γ channel and cross use the
DATA covariance `Sigma_data` (per "propagate the data noise of γ"). `se_T_alpha` /
`se_T_gamma` split the two marginal channels; `se_T` is the correlated total.

# Saves (under output_folder/inference/)
- `se_theta_T_delta.npy`         : √diag(V_T) — headline correlated SE (sandwich α)
- `se_theta_T_delta_eff.npy`     : … using the efficient Var(α̂) in the α channel
- `se_theta_T_delta_alpha.npy`   : α-channel-only SE  √diag(f_α Var(α̂) f_α')
- `se_theta_T_delta_gamma.npy`   : γ-channel-only SE  √diag(f_γ Σ_γγ f_γ')
- `se_theta_T_delta_regcoef.npy` : α-channel SE using the reg_coef-only Var(α̂) (Ω_β)
- `var_theta_T_delta.npy`        : V_T (correlated, sandwich α)
- `t_stats_T_delta.npy`, `ci_95_T_delta.npy` : t = T̂/se_T and T̂ ± 1.96·se_T
- `jacobian_T_wrt_alpha.npy`, `jacobian_T_wrt_gamma.npy` : f_α, f_γ
- `T_precision_vs_gamma.png`     : scatter of T precision (|t|=|T̂|/se_T) vs the
  DOMESTIC γ share γ̃=γ^F/domestic_share (Σ_r=1); + `T_precision_vs_gamma.npz`
  (γ̃, γ^F, T̂, se_T{,_alpha,_gamma}, t) for external re-plotting.
- `inference_delta.txt`          : the α / T / N_s report (first column `parameters`)
- `t_stats_T_delta_alpha.npy`, `ci_95_T_delta_alpha.npy` : t and CI from the α channel alone
"""
function compute_profiled_T_inference(theta_hat::Vector{Float64},
                                      gb_param_idx::Vector{Int},
                                      param_labels_gb::Vector{String};
                                      dTa,
                                      G_alpha::Matrix{Float64},
                                      W::Matrix{Float64},
                                      Sigma_data::Matrix{Float64},
                                      Var_alpha_sw::Matrix{Float64},
                                      Var_alpha_eff::Matrix{Float64},
                                      Var_alpha_reg::Union{Nothing,Matrix{Float64}} = nothing,
                                      gb_block_ranges,
                                      output_folder::String = ".",
                                      industry::String = "",
                                      step_rel::Float64 = 1e-2,
                                      target::AbstractMatrix = EMP_GAMMA_T,
                                      # α section of the report — the stats already in
                                      # inference_summary.txt, reproduced verbatim.
                                      alpha_labels::Vector{String} = String[],
                                      alpha_values::Vector{Float64} = Float64[],
                                      alpha_se_eff::Vector{Float64} = Float64[],
                                      alpha_se_sw::Vector{Float64} = Float64[],
                                      # N_s section. `se_N_naive` is the mechanical
                                      # (G_N'W G_N)^{-1}; the STRUCTURAL SE is built here
                                      # from q̂ and the free-parameter Jacobian.
                                      N_hat::Union{Nothing,Vector{Int}} = nothing,
                                      se_N_naive::Union{Nothing,Vector{Float64}} = nothing,
                                      q_hat::Union{Nothing,Vector{Float64}} = nothing,
                                      J_free_gb::Union{Nothing,AbstractMatrix} = nothing)
    inf_dir = joinpath(output_folder, "inference")
    mkpath(inf_dir)

    f_alpha      = dTa.J_T               # ∂T*/∂α  (n_T × N_TAU)
    T_pos        = dTa.T_pos
    T_red_pos    = dTa.T_red_pos
    n_T          = dTa.n_T
    n_alpha      = dTa.n_alpha
    ΩL, Ωs, A, α = dTa.ΩL, dTa.Ωs, dTa.A, dTa.α
    T_hat_mat    = dTa.T_hat_mat
    T_hat_active = dTa.T_hat_active

    # γ moment block within the gb vector (β-then-γ ⇒ second block) and its (r,s) map.
    gam_rng  = gb_block_ranges[2]
    n_gam    = length(gam_rng)
    gamma_rs = _gamma_block_rs()
    @assert length(gamma_rs) == n_gam (
        "γ (r,s) map length $(length(gamma_rs)) != γ moment count $n_gam — " *
        "MOMENT_MASK/BLOCK_RANGES[5] misalignment.")

    # ── f_γ = ∂T*/∂γ_target by central LOG-step FD on invert_T_ge (pmap over γ) ──
    # Perturb one empirical γ target entry at a time (EMP_GAMMA_T[c,s]), re-solve the
    # Sinkhorn inversion, chain-rule back to raw γ units by 1/γ. Only the target
    # PROFILE matters (per-sector scale is gauge), so a single-entry bump is a genuine
    # profile perturbation; the reference-region entry is gauge and left untouched.
    δ = step_rel
    fg_cols = pmap(1:n_gam) do j
        (r, s) = gamma_rs[j]
        g0 = target[r, s]
        g0 <= 0 && return zeros(n_T)
        tp = Matrix{Float64}(target); tp[r, s] = g0 * exp(δ)
        tm = Matrix{Float64}(target); tm[r, s] = g0 * exp(-δ)
        Rp = invert_T_ge(α, ΩL, Ωs, A; target=tp, T_init=copy(T_hat_mat))
        Rm = invert_T_ge(α, ΩL, Ωs, A; target=tm, T_init=copy(T_hat_mat))
        dT = (vec(permutedims(Rp.T))[T_MASK] .- vec(permutedims(Rm.T))[T_MASK]) ./ (2δ * g0)
        return dT[T_red_pos]
    end
    f_gamma = n_gam == 0 ? zeros(n_T, 0) : reduce(hcat, fg_cols)   # ∂T*/∂γ  (n_T × n_gam)

    # ── Joint (α̂, γ̂) covariance pieces ──────────────────────────────────────────
    # Influence of α̂ on the moment vector: P = (G'WG)^{-1} G'W = Var_eff(α̂)·G'·W.
    P       = Var_alpha_eff * G_alpha' * W          # (N_TAU × n_gb)  dα̂/dm
    Sig_gg  = Sigma_data[gam_rng, gam_rng]          # (n_gam × n_gam)  DATA cov of γ moments
    Sig_g   = Sigma_data[:, gam_rng]                # (n_gb × n_gam)
    Cov_ag  = P * Sig_g                             # (N_TAU × n_gam)  Cov(α̂, γ̂)

    # V_T = f_α Var(α̂) f_α' + f_γ Σ_γγ f_γ' + f_α Cov(α̂,γ̂) f_γ' + (·)'
    function assemble(Var_a)
        Vaa = f_alpha * Var_a * f_alpha'
        Vgg = f_gamma * Sig_gg * f_gamma'
        Vag = f_alpha * Cov_ag * f_gamma'
        V   = Vaa .+ Vgg .+ Vag .+ Vag'
        return (V .+ V') ./ 2
    end
    VT_sw  = assemble(Var_alpha_sw)
    VT_eff = assemble(Var_alpha_eff)

    # Marginal-channel variances (for decomposition; sandwich α).
    Vaa_only = f_alpha * Var_alpha_sw * f_alpha';  Vaa_only = (Vaa_only .+ Vaa_only') ./ 2
    Vgg_only = f_gamma * Sig_gg       * f_gamma';  Vgg_only = (Vgg_only .+ Vgg_only') ./ 2

    se_T       = sqrt.(max.(diag(VT_sw),   0.0))
    se_T_eff   = sqrt.(max.(diag(VT_eff),  0.0))
    se_T_alpha = sqrt.(max.(diag(Vaa_only), 0.0))
    se_T_gamma = sqrt.(max.(diag(Vgg_only), 0.0))

    se_T_reg = fill(NaN, n_T)
    if Var_alpha_reg !== nothing
        Vreg = f_alpha * Var_alpha_reg * f_alpha'
        se_T_reg = sqrt.(max.(diag(Vreg), 0.0))
    end

    # t-stat and 95% CI from the correlated total SE, centered at T̂.
    t_T  = [se_T[i] > 0 ? T_hat_active[i] / se_T[i] : NaN for i in 1:n_T]
    ci_T = hcat(T_hat_active .- 1.96 .* se_T, T_hat_active .+ 1.96 .* se_T)

    # ── Save arrays ─────────────────────────────────────────────────────────────
    NPZ.npzwrite(joinpath(inf_dir, "se_theta_T_delta.npy"),          se_T)
    NPZ.npzwrite(joinpath(inf_dir, "se_theta_T_delta_eff.npy"),      se_T_eff)
    NPZ.npzwrite(joinpath(inf_dir, "se_theta_T_delta_alpha.npy"),    se_T_alpha)
    NPZ.npzwrite(joinpath(inf_dir, "se_theta_T_delta_gamma.npy"),    se_T_gamma)
    NPZ.npzwrite(joinpath(inf_dir, "se_theta_T_delta_regcoef.npy"),  se_T_reg)
    NPZ.npzwrite(joinpath(inf_dir, "var_theta_T_delta.npy"),         VT_sw)
    NPZ.npzwrite(joinpath(inf_dir, "t_stats_T_delta.npy"),           t_T)
    NPZ.npzwrite(joinpath(inf_dir, "ci_95_T_delta.npy"),             ci_T)
    NPZ.npzwrite(joinpath(inf_dir, "jacobian_T_wrt_alpha.npy"),      f_alpha)
    NPZ.npzwrite(joinpath(inf_dir, "jacobian_T_wrt_gamma.npy"),      f_gamma)

    # ── T precision vs γ size scatter (γ in the RENORMALIZED, Σ_r=1 space) ───────
    # The identified T columns and the γ moment block enumerate the SAME active,
    # non-reference (s,r) pairs in s-major order, so T param i ↔ gamma_rs[i]. The
    # x-axis is the DOMESTIC sourcing share γ̃[r,s] = γ^F[r,s]/domestic_share[s]
    # (sums to 1 per sector — the economically comparable size, per the request),
    # the y-axis is T's precision = |t| = |T̂|/se_T. Expect a positive relation:
    # bigger, better-measured domestic shares ⇒ tighter T. Saves a PNG + an .npz
    # with every series so it can be re-plotted externally. Guarded — never blocks.
    gamma_tilde = fill(NaN, n_gam)   # γ̃ (Σ_r=1), aligned to the γ / T ordering
    gamma_face  = fill(NaN, n_gam)   # γ^F (with-foreign, as entered in Σ)
    for i in 1:n_gam
        (r, s) = gamma_rs[i]
        gamma_face[i]  = target[r, s]
        ds             = domestic_share[s]
        gamma_tilde[i] = ds > 0 ? target[r, s] / ds : NaN
    end
    if n_T == n_gam
        try
            keep = findall(k -> isfinite(gamma_tilde[k]) && gamma_tilde[k] > 0 &&
                                isfinite(t_T[k]), 1:n_T)
            if !isempty(keep)
                p = scatter(gamma_tilde[keep], abs.(t_T[keep]);
                    xscale            = :log10,
                    markersize        = 4,
                    alpha             = 0.6,
                    markerstrokecolor = :black,
                    markerstrokewidth = 0.5,
                    color             = RGB(0.247, 0.404, 0.667),
                    legend            = false,
                    xlabel            = "γ (domestic sourcing share, Σ_r = 1)",
                    ylabel            = "T precision  |t| = |T| / se_T",
                    title             = "T precision vs domestic γ share",
                    grid              = true, gridalpha = 0.5, gridstyle = :dash)
                hline!(p, [1.96]; color = :firebrick, ls = :dash, label = "")
                savefig(p, joinpath(inf_dir, "T_precision_vs_gamma.png"))
                println("  T-precision-vs-γ scatter saved to: " *
                        joinpath(inf_dir, "T_precision_vs_gamma.png"))
            end
        catch e
            @warn "T_precision_vs_gamma plot failed; continuing." exception=e
        end
    end
    NPZ.npzwrite(joinpath(inf_dir, "T_precision_vs_gamma.npz"),
        Dict("gamma_tilde"  => gamma_tilde,   # γ̃ (renormalized, Σ_r=1)
             "gamma_face"   => gamma_face,     # γ^F (with-foreign, Σ entered on this)
             "T_hat"        => T_hat_active,
             "se_T"         => se_T,
             "se_T_alpha"   => se_T_alpha,
             "se_T_gamma"   => se_T_gamma,
             "t_T"          => t_T))

    # ── Human-readable summary ──────────────────────────────────────────────────
    T_labels = param_labels_gb[T_pos]
    mean_share_a = mean([se_T[i] > 0 ? (se_T_alpha[i] / se_T[i])^2 : NaN for i in 1:n_T])
    mean_share_g = mean([se_T[i] > 0 ? (se_T_gamma[i] / se_T[i])^2 : NaN for i in 1:n_T])

    # ── STRUCTURAL N̂_s variance (the same idea as T's delta method) ─────────────
    # N̂_s is not estimated as a free parameter: it is CALIBRATED, solving
    #
    #     Ḡ_s(n; q̂) = Ĝ_s,        Ḡ_s(n; q̂) = mean_{l∈L_s} (1 − q̂_ls)^n,
    #
    # where Ĝ_s is the empirical count target (its bootstrap noise is the G block of
    # Sigma_data) and q̂ = q̂(α̂, T*(α̂, γ̂)) — free of N_s by Lemma 2. So N̂_s inherits
    # noise from THREE correlated sources, and the implicit function theorem gives
    #
    #     dN_s = [ dĜ_s − a_s dα̂ − b_s dγ̂ ] / D_s,
    #     D_s = ∂Ḡ_s/∂n = mean_l (1−q̂_ls)^{N̂_s} ln(1−q̂_ls)  < 0,
    #     a_s = ∂Ḡ_s/∂α  (TOTAL, along the profiled manifold) = G_alpha[G0 row s, :],
    #     b_s = ∂Ḡ_s/∂γ  = (∂Ḡ_s/∂T)·f_γ, the block-6 row of the FREE-parameter
    #           Jacobian times ∂T*/∂γ — no new simulation needed.
    #
    # Var(N̂) = L·JointCov(Ĝ, α̂, γ̂)·L' with L = [D^{-1}, −D^{-1}A, −D^{-1}B]. PSD by
    # construction (a linear image of the data/simulation noise), like V_T.
    #
    # NOTE the direction of causation: T̂ = T*(α̂, γ̂) does NOT depend on N_s at all
    # (invert_T_ge is the continuum inversion, and q̂ is N_s-free by Lemma 2), so the
    # system is TRIANGULAR — α̂ → T̂ → q̂ → N̂_s. N_s noise does not feed back into T.
    se_N_delta   = nothing
    se_N_target  = nothing   # Ĝ channel only
    se_N_alpha   = nothing   # α channel only
    se_N_gamma   = nothing   # γ channel only
    D_analytic   = nothing
    D_firstdiff  = nothing
    if GRANULAR && q_hat !== nothing && N_hat !== nothing &&
       length(gb_block_ranges) >= 3 && J_free_gb !== nothing
        G0_rng = gb_block_ranges[3]
        nS     = length(G0_rng)
        if nS == S && size(J_free_gb, 2) == length(param_labels_gb)
            # D_s: exact ∂Ḡ_s/∂n, plus the unit first difference for comparison
            # (they agree to O(q̂); the first difference is the integer-step scale).
            D_analytic  = zeros(S)
            D_firstdiff = zeros(S)
            for s in 1:S
                cells = CELLS_OF_SECTOR[s]
                isempty(cells) && continue
                acc_d = 0.0; acc_f = 0.0
                for g in cells
                    q  = q_hat[g]
                    pw = (1.0 - q)^N_hat[s]
                    acc_d += pw * log(max(1.0 - q, 1e-300))
                    acc_f += -q * pw                       # Ḡ(n+1) − Ḡ(n)
                end
                D_analytic[s]  = acc_d / length(cells)
                D_firstdiff[s] = acc_f / length(cells)
            end

            A_mat = Matrix{Float64}(G_alpha[G0_rng, :])                      # S × N_TAU
            B_mat = Matrix{Float64}(J_free_gb[G0_rng, T_pos]) * f_gamma      # S × n_gam
            Sig_GG = Sigma_data[G0_rng, G0_rng]                              # S × S
            Cov_aG = P * Sigma_data[:, G0_rng]                               # N_TAU × S
            Sig_gG = Sigma_data[gam_rng, G0_rng]                             # n_gam × S

            Dinv = Diagonal([abs(D_analytic[s]) > 1e-14 ? 1.0 / D_analytic[s] : 0.0
                             for s in 1:S])
            L = hcat(Matrix(Dinv), -(Dinv * A_mat), -(Dinv * B_mat))         # S × (S+N_TAU+n_gam)
            Joint = [ Sig_GG        Cov_aG'          Sig_gG';
                      Cov_aG        Var_alpha_sw     Cov_ag;
                      Sig_gG        Cov_ag'          Sig_gg ]
            V_N = L * Joint * L';  V_N = (V_N .+ V_N') ./ 2
            se_N_delta = sqrt.(max.(diag(V_N), 0.0))

            # Marginal channels (same loadings, one covariance block at a time).
            se_N_target = sqrt.(max.(diag(Dinv * Sig_GG * Dinv'), 0.0))
            Va = (Dinv * A_mat) * Var_alpha_sw * (Dinv * A_mat)'
            se_N_alpha  = sqrt.(max.(diag(Va), 0.0))
            Vg = (Dinv * B_mat) * Sig_gg * (Dinv * B_mat)'
            se_N_gamma  = sqrt.(max.(diag(Vg), 0.0))

            NPZ.npzwrite(joinpath(inf_dir, "var_theta_N_s_delta.npy"), V_N)
            NPZ.npzwrite(joinpath(inf_dir, "se_theta_N_s_delta.npy"),  se_N_delta)
            NPZ.npzwrite(joinpath(inf_dir, "jacobian_N_s_wrt_alpha.npy"), A_mat)
            NPZ.npzwrite(joinpath(inf_dir, "jacobian_N_s_wrt_gamma.npy"), B_mat)
            NPZ.npzwrite(joinpath(inf_dir, "dGbar_dn.npy"), D_analytic)
        end
    end

    # α-only t and CI for T: the interval T̂ would carry if γ̂ were known exactly, i.e.
    # T's precision attributable to the estimation error of α̂ alone.
    t_T_alpha  = [se_T_alpha[i] > 0 ? T_hat_active[i] / se_T_alpha[i] : NaN for i in 1:n_T]
    ci_T_alpha = hcat(T_hat_active .- 1.96 .* se_T_alpha, T_hat_active .+ 1.96 .* se_T_alpha)
    NPZ.npzwrite(joinpath(inf_dir, "t_stats_T_delta_alpha.npy"), t_T_alpha)
    NPZ.npzwrite(joinpath(inf_dir, "ci_95_T_delta_alpha.npy"),   ci_T_alpha)

    open(joinpath(inf_dir, "inference_delta.txt"), "w") do io
        println(io, "="^108)
        println(io, "PROFILED-ESTIMATOR INFERENCE — α, T, N_s")
        println(io, "  Industry   : $(isempty(industry) ? "(not specified)" : industry)")
        println(io, "  n_α (N_TAU): $n_alpha   n_T : $n_T   n_γ : $n_gam" *
                    (N_hat === nothing ? "" : "   n_N_s : $(length(N_hat))"))
        println(io, "  Date       : $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
        println(io, "="^108)
        println(io, "\nThree parameter groups, each with the variance that matches how it is estimated:")
        println(io, "  α    — estimated on the moment conditions. Stats reproduced from inference_summary.txt.")
        println(io, "  T    — NOT estimated: the deterministic GE-Sinkhorn image T*(α̂, γ̂). Correlated")
        println(io, "         delta method over (α̂, γ̂), see below.")
        println(io, "  N_s  — profiled by bisection on Ḡ_s(0), but it is NOT calibrated inside the model")
        println(io, "         the way T is, so it takes the efficient matrix like α: (G_N' W G_N)^{-1}")
        println(io, "         with G_N = ∂m/∂N_s from the unit first difference m(N_s+1) − m(N_s).")

        # ── α ────────────────────────────────────────────────────────────────────
        println(io, "\n" * "-"^108)
        println(io, "--- parameters: α  (from inference_summary.txt) ---")
        if isempty(alpha_labels)
            println(io, "  (α stats not supplied to this report — see inference_summary.txt)")
        else
            ah = @sprintf("  %-24s  %-12s  %-12s  %-12s  %-8s  %-12s  %-12s",
                          "parameters", "estimate", "se_eff", "se_sw", "t", "CI_lo", "CI_hi")
            println(io, ah)
            println(io, "  " * "-"^(length(ah)-2))
            for i in eachindex(alpha_labels)
                se_i = i <= length(alpha_se_sw) ? alpha_se_sw[i] : NaN
                ti   = (isfinite(se_i) && se_i > 0) ? alpha_values[i] / se_i : NaN
                @printf(io, "  %-24s  %-12.6f  %-12.6e  %-12.6e  %-8.4f  %-12.6f  %-12.6f\n",
                        alpha_labels[i], alpha_values[i],
                        i <= length(alpha_se_eff) ? alpha_se_eff[i] : NaN, se_i,
                        isnan(ti) ? -999.0 : ti,
                        alpha_values[i] - 1.96 * se_i, alpha_values[i] + 1.96 * se_i)
            end
            println(io, "  (t and CI use se_sw, the reported sandwich SE.)")
        end

        # ── T ────────────────────────────────────────────────────────────────────
        println(io, "\n" * "-"^108)
        println(io, "--- parameters: T  (correlated delta method) ---")
        println(io, "  V_T = f_α Var(α̂) f_α' + f_γ Σ_γγ f_γ' + f_α Cov(α̂,γ̂) f_γ' + (·)'")
        println(io, "  T = T*(α̂, γ̂) is the deterministic GE-Sinkhorn image (invert_T_ge). Its CI")
        println(io, "  propagates TWO sources of uncertainty, WITH their correlation:")
        println(io, "    (α) the estimation error of α̂ — reduced-Jacobian profiled Var(α̂), and")
        println(io, "    (γ) the DATA (bootstrap) noise of the γ_ls Sinkhorn target — Σ_γγ from")
        println(io, "        Sigma_data. Cov(α̂,γ̂)=P·Σ_data[:,γ] with P=dα̂/dm=(G'WG)^{-1}G'W.")
        println(io, "  se_T        : √diag(V_T) — headline correlated SE (sandwich α)")
        println(io, "  se_T_alpha  : α-channel only  √diag(f_α Var(α̂) f_α')")
        println(io, "  se_T_gamma  : γ-channel only  √diag(f_γ Σ_γγ f_γ')")
        println(io, "  se_T_reg    : α-channel using reg_coef-only Var(α̂) (Ω_β)")
        println(io, "  t, CI       : t = T̂/se_T and 95% CI = T̂ ± 1.96·se_T")
        println(io, "  t_α, CI_α   : the SAME statistics from the α CHANNEL ALONE (se_T_alpha) —")
        println(io, "                T's precision if the γ target were known exactly.")
        hdr = @sprintf("  %-24s  %-12s  %-12s  %-12s  %-12s  %-12s  %-8s  %-12s  %-12s  %-8s  %-12s  %-12s",
                       "parameters", "estimate", "se_T", "se_T_alpha",
                       "se_T_gamma", "se_T_reg", "t", "CI_lo", "CI_hi",
                       "t_α", "CI_lo_α", "CI_hi_α")
        println(io, hdr)
        println(io, "  " * "-"^(length(hdr)-2))
        for i in 1:n_T
            @printf(io, "  %-24s  %-12.6f  %-12.6e  %-12.6e  %-12.6e  %-12.6e  %-8.4f  %-12.6f  %-12.6f  %-8.4f  %-12.6f  %-12.6f\n",
                    T_labels[i], T_hat_active[i], se_T[i], se_T_alpha[i],
                    se_T_gamma[i], se_T_reg[i],
                    isnan(t_T[i]) ? -999.0 : t_T[i], ci_T[i, 1], ci_T[i, 2],
                    isnan(t_T_alpha[i]) ? -999.0 : t_T_alpha[i],
                    ci_T_alpha[i, 1], ci_T_alpha[i, 2])
        end

        # ── N_s ──────────────────────────────────────────────────────────────────
        println(io, "\n" * "-"^108)
        println(io, "--- parameters: N_s  (calibrated — structural delta method) ---")
        if N_hat === nothing
            println(io, "  (not applicable — N_s exists only under GRANULAR)")
        elseif se_N_delta === nothing
            println(io, "  (structural SE unavailable — needs q̂ and the free-parameter Jacobian;")
            println(io, "   only the naive SE below is reported)")
            if se_N_naive !== nothing
                for s in eachindex(se_N_naive)
                    @printf(io, "  %-24s  N̂=%-8.0f  se_naive=%-12.6e\n",
                            "N_s[sector $s]", Float64(N_hat[s]), se_N_naive[s])
                end
            end
        else
            println(io, "  N̂_s is NOT a free parameter: it is CALIBRATED, solving Ḡ_s(n; q̂) = Ĝ_s")
            println(io, "  with Ḡ_s(n; q̂) = mean_l (1−q̂_ls)^n. So — exactly as for T — its noise is")
            println(io, "  inherited, here from THREE correlated sources, via the implicit function")
            println(io, "  theorem:")
            println(io, "      dN_s = [ dĜ_s − a_s dα̂ − b_s dγ̂ ] / D_s")
            println(io, "    D_s = ∂Ḡ_s/∂n = mean_l (1−q̂)^{N̂} ln(1−q̂)   (< 0: Ḡ decreasing in n)")
            println(io, "    a_s = ∂Ḡ_s/∂α  — the TOTAL derivative along the profiled manifold")
            println(io, "    b_s = ∂Ḡ_s/∂γ  = (∂Ḡ_s/∂T)·f_γ")
            println(io, "  Var(N̂) = L·JointCov(Ĝ, α̂, γ̂)·L', L = [D⁻¹, −D⁻¹A, −D⁻¹B]. PSD by")
            println(io, "  construction. Units: VARIETIES.")
            println(io, "  se_N_target : Ĝ-channel only — the bootstrap noise of the count target")
            println(io, "                (the G block of Sigma_data). Usually dominant.")
            println(io, "  se_N_alpha  : α-channel only.   se_N_gamma : γ-channel only.")
            println(io, "  se_N_naive  : the NAIVE sandwich SE printed by inference_summary.txt")
            println(io, "                (N_s as a free parameter, jointly with α and T), shown for")
            println(io, "                contrast — it has no notion of the calibration above.")
            nh = @sprintf("  %-18s  %-10s  %-12s  %-12s  %-12s  %-12s  %-12s  %-8s  %-12s  %-12s",
                          "parameters", "estimate", "se_N", "se_N_target",
                          "se_N_alpha", "se_N_gamma", "se_N_naive", "t", "CI_lo", "CI_hi")
            println(io, nh)
            println(io, "  " * "-"^(length(nh)-2))
            for s in eachindex(se_N_delta)
                v = Float64(N_hat[s]); sd = se_N_delta[s]
                ts = (isfinite(sd) && sd > 0) ? v / sd : NaN
                @printf(io, "  %-18s  %-10.0f  %-12.6e  %-12.6e  %-12.6e  %-12.6e  %-12.6e  %-8.4f  %-12.4f  %-12.4f\n",
                        "N_s[sector $s]", v, sd, se_N_target[s], se_N_alpha[s],
                        se_N_gamma[s],
                        se_N_naive === nothing ? NaN : se_N_naive[s],
                        isnan(ts) ? -999.0 : ts, v - 1.96 * sd, v + 1.96 * sd)
            end
            println(io, "\n  ∂Ḡ_s/∂n — the denominator that converts count-moment error into")
            println(io, "  varieties. A small |D_s| means the count moment barely moves with one")
            println(io, "  more variety there, so N̂_s is weakly identified in that sector.")
            @printf(io, "  %-18s  %-16s  %-16s  %-10s\n",
                    "sector", "D_s (exact)", "Ḡ(n+1)−Ḡ(n)", "gap")
            for s in 1:S
                gap = abs(D_analytic[s]) > 1e-14 ?
                      abs(D_firstdiff[s] - D_analytic[s]) / abs(D_analytic[s]) : NaN
                @printf(io, "  %-18d  %-16.6e  %-16.6e  %-10.4f\n",
                        s, D_analytic[s], D_firstdiff[s], isnan(gap) ? -999.0 : gap)
            end
            println(io, "  (the two agree to O(q̂); the first difference is the integer-step scale,")
            println(io, "   the exact derivative is the implicit-function object used above.)")
            println(io, "\n  N.B. the CI is continuous; N_s is an integer, so read it as the set of")
            println(io, "  integer counts the count moment cannot distinguish at 95%.")
        end

        println(io, "\n--- Channel decomposition, T (mean variance share) ---")
        @printf(io, "  α-channel share: %.4f   γ-channel share: %.4f   (cross ⇒ shares need not sum to 1)\n",
                mean_share_a, mean_share_g)
        @printf(io, "  mean se_T: %.4e   mean se_T_alpha: %.4e   mean se_T_gamma: %.4e\n",
                mean(se_T), mean(se_T_alpha), mean(se_T_gamma))
        println(io, "\n--- Caveats ---")
        println(io, "  * Θ = (α̂, γ̂). Ω^L, Ω^s, A are held fixed at θ̂ (their SMM-final-step status);")
        println(io, "    a full-head delta method would add their covariance — not propagated here.")
        println(io, "  * The α channel uses the reported (sandwich) Var(α̂); the γ channel and the")
        println(io, "    cross term use the DATA covariance Sigma_data (γ target noise). The joint")
        println(io, "    (α̂,γ̂) covariance is PSD by construction, so V_T is a proper covariance.")
        println(io, "  * f_γ perturbs EMP_GAMMA_T[c,s]; Σ_γγ must be the covariance of the SAME")
        println(io, "    (thresholded/renormalized) γ moments — regenerate Sigma_beta_gamma if the")
        println(io, "    active set was gamma-thresholded (see reconcile_sigma_data caveat).")
        if se_N_delta !== nothing
            println(io, "  * The system is TRIANGULAR: α̂ → T̂ = T*(α̂,γ̂) → q̂ → N̂_s. T does NOT")
            println(io, "    depend on N_s — invert_T_ge is the continuum inversion and q̂ is")
            println(io, "    N_s-free (Lemma 2) — so N_s noise does not feed back into V_T. The")
            println(io, "    arrow runs the other way, and that transmission IS in Var(N̂) above.")
            println(io, "  * Var(N̂) holds α̂'s own variance at its reported value; it is not a")
            println(io, "    joint (α, N_s) re-estimation, so the α block above is unchanged.")
            println(io, "  * It also does not carry the Monte-Carlo dispersion of N̂_s across draw")
            println(io, "    sets (compute_jacobian reports that separately) — that is simulation")
            println(io, "    noise in q̂, removable by raising N_RHO_INFERENCE, not sampling error.")
        end
        println(io, "\n" * "="^108)
    end

    println("Profiled α/T/N_s inference saved to: $(joinpath(inf_dir, "inference_delta.txt"))")
    @printf("  mean se_T=%.4e (α-share=%.3f, γ-share=%.3f) over %d T params\n",
            mean(se_T), mean_share_a, mean_share_g, n_T)

    return Dict(
        "J_T_wrt_alpha"  => f_alpha,
        "J_T_wrt_gamma"  => f_gamma,
        "Var_T_delta"    => VT_sw,
        "se_T_delta"     => se_T,
        "se_T_delta_eff" => se_T_eff,
        "se_T_delta_alpha" => se_T_alpha,
        "se_T_delta_gamma" => se_T_gamma,
        "se_T_delta_reg" => se_T_reg,
        "t_T_delta"      => t_T,
        "ci_T_delta"     => ci_T,
        "t_T_delta_alpha"  => t_T_alpha,
        "ci_T_delta_alpha" => ci_T_alpha,
    )
end


"""
    run_profiled_inference(theta_hat, G_alpha_full, gb_indices, gb_param_idx,
        W, Omega, Sigma_data, emp_vec_gb, sim_vec_gb; kwargs...) -> (inf_alpha, inf_T)

Orchestrates the **profiling-path** inference at a profiled estimate θ̂ (where
`T̂ = T*(α̂,Ω̂,Â)`). `G_alpha_full` is the **direct profiled Jacobian** dm/dα
(all moment rows × N_TAU), computed by `compute_jacobian(...; profile_T=true,
param_indices = α-only)` — i.e. ONLY α is perturbed and T follows via the Sinkhorn
image, so no ∂m/∂T-as-free-parameter is ever formed. This routine (1) slices it to
the β+γ rows → the α-reduced Jacobian `G_α`, (2) runs `compute_smm_inference` on
`G_α` alone → the profiled α CI (T shown value-only in the report), and (3) runs
`compute_profiled_T_inference` → the correlated α+γ CI for T (using `∂T*/∂α` from
`_dTstar_dalpha` for the T delta). Both write under `output_folder/inference/`.
Used at Step 2 (θ̂_1) and Step 4 (θ̂_2) when `profile_T=true`; the joint
(non-profiled) path keeps the standard α+T inference.
"""
function run_profiled_inference(theta_hat::Vector{Float64},
                                G_alpha_full::Matrix{Float64},
                                gb_indices::Vector{Int},
                                gb_param_idx::Vector{Int},
                                W::Matrix{Float64},
                                Omega::Matrix{Float64},
                                Sigma_data::Matrix{Float64},
                                emp_vec_gb::Vector{Float64},
                                sim_vec_gb::Vector{Float64};
                                output_folder::String = ".",
                                industry::String = "",
                                K_sim::Int = 0,
                                gb_block_ranges = nothing,
                                gb_block_names  = nothing,
                                gamma_ref_map   = nothing,
                                param_labels_gb::Vector{String} = String[],
                                moment_labels_gb = nothing,
                                head_labels = String[],
                                head_values = Float64[],
                                step_rel::Float64 = 1e-2,
                                target::AbstractMatrix = EMP_GAMMA_T,
                                # The FREE-parameter first-difference Jacobian on the same
                                # β+γ(+G) rows and α+T columns. Supplied, T gets an SE in
                                # inference_summary.txt from (G'WG)^{-1} — the SE it would
                                # carry as a free parameter, alongside its delta-method SE.
                                J_free_gb::Union{Nothing,AbstractMatrix} = nothing,
                                # N_s first-difference Jacobian settings (GRANULAR only).
                                N_s_K::Int = 10,
                                N_s_base_seed::Int = 7_000_000,
                                draw_method::Symbol = INFERENCE_DRAW_METHOD)
    # ── ∂T*/∂α (identified T × N_TAU) + α/T column bookkeeping ──────────────────
    dTa = _dTstar_dalpha(theta_hat, gb_param_idx, param_labels_gb;
                         step_rel=step_rel, target=target)
    dTa === nothing && error("run_profiled_inference: no α or no T columns in gb params.")
    alpha_pos = dTa.alpha_pos
    T_pos     = dTa.T_pos
    f_alpha   = dTa.J_T

    # ── Direct profiled α Jacobian: dm/dα (only α perturbed, T follows) ──────────
    # G_alpha_full was built with profile_T=true over the α columns (in the SAME
    # order as PARAM_LABELS' α section = alpha_pos), so its N_TAU columns align with
    # alpha_pos. Slice to the β+γ rows.
    @assert size(G_alpha_full, 2) == length(alpha_pos) (
        "run_profiled_inference: G_alpha_full has $(size(G_alpha_full,2)) columns " *
        "but $(length(alpha_pos)) α params — pass a profiled α-only Jacobian.")
    G_alpha = G_alpha_full[gb_indices, :]                    # β+γ rows × N_TAU
    alpha_param_idx = gb_param_idx[alpha_pos]

    # ── The NAIVE block: one Jacobian holding α, T and N_s as FREE parameters, so
    #    inference_summary.txt reports a single method throughout (the mechanical
    #    sandwich). The structural treatment — T as the Sinkhorn image, N_s as the
    #    calibrated solution of Ḡ_s(n)=Ĝ_s — is inference_delta.txt's job.
    #    T   : columns of the free-parameter FD Jacobian.
    #    N_s : the unit first difference m(N_s+1) − m(N_s); N_s is an integer, so one
    #          variety IS the step and the SE comes out in varieties.
    T_disp_labels = param_labels_gb[T_pos]
    T_disp_values = theta_hat[gb_param_idx[T_pos]]

    J_free_ok = J_free_gb !== nothing && size(J_free_gb, 2) == length(param_labels_gb)
    if J_free_gb !== nothing && !J_free_ok
        @warn "run_profiled_inference: J_free_gb has $(size(J_free_gb,2)) columns but " *
              "$(length(param_labels_gb)) gb params — skipping the naive T block."
    end

    N_hat_v, se_N_naive, q_hat_v = nothing, nothing, nothing
    G_N_gb = nothing
    if GRANULAR
        # The GRANULAR block below writes into <output_folder>/inference/ BEFORE
        # compute_smm_inference (the only other creator of that directory) runs, so
        # create it here or the first granular run loses the Jacobian it just paid for.
        mkpath(joinpath(output_folder, "inference"))
        gr = granular_report(profiled_theta(theta_hat);
                             u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS)
        q_hat_v = gr.q_hat
        G_N, G_N_sd, N_hat_v, _ = compute_N_s_jacobian(
            theta_hat; N_hat=gr.N_hat, K=N_s_K, base_seed=N_s_base_seed,
            draw_method=draw_method, profile_T=true)
        G_N_gb = G_N[gb_indices, :]
        NPZ.npzwrite(joinpath(output_folder, "inference", "jacobian_N_s.npy"),    G_N_gb)
        NPZ.npzwrite(joinpath(output_folder, "inference", "jacobian_N_s_sd.npy"), G_N_sd[gb_indices, :])
        V_N, floored_N = _gtwg_inv(G_N_gb, W)
        floored_N && @warn "N_s (G_N'W G_N) was not PD; eigenvalue floor applied. A sector " *
                           "whose ∂Ḡ_s(0)/∂N_s ≈ 0 has no local information on N_s."
        se_N_naive = sqrt.(max.(diag(V_N), 0.0))
        NPZ.npzwrite(joinpath(output_folder, "inference", "se_theta_N_s.npy"), se_N_naive)
    end

    fd_labels = String[]; fd_values = Float64[]; fd_blocks = Matrix{Float64}[]
    if J_free_ok
        append!(fd_labels, vcat(param_labels_gb[alpha_pos], T_disp_labels))
        append!(fd_values, vcat(theta_hat[alpha_param_idx], T_disp_values))
        push!(fd_blocks, Matrix{Float64}(J_free_gb[:, vcat(alpha_pos, T_pos)]))
    end
    if G_N_gb !== nothing
        append!(fd_labels, ["N_s[sector $s]" for s in 1:size(G_N_gb, 2)])
        append!(fd_values, Float64.(N_hat_v))
        push!(fd_blocks, Matrix{Float64}(G_N_gb))
    end
    fd_G = isempty(fd_blocks) ? nothing : reduce(hcat, fd_blocks)

    # ── Profiled α inference (reduced Jacobian) — T shown value-only in report ──
    inf_alpha = compute_smm_inference(
        theta_hat, G_alpha, W, Omega;
        param_indices         = alpha_param_idx,
        empirical_moments_vec = emp_vec_gb,
        simulated_moments_vec = sim_vec_gb,
        output_folder         = output_folder,
        industry              = industry,
        K_sim                 = K_sim,
        block_ranges          = gb_block_ranges,
        block_names           = gb_block_names,
        gamma_ref_map         = gamma_ref_map,
        param_labels          = param_labels_gb[alpha_pos],
        moment_labels         = moment_labels_gb,
        display_labels        = vcat(collect(head_labels), collect(T_disp_labels)),
        display_values        = vcat(collect(head_values), collect(T_disp_values)),
        fd_G                  = fd_G,
        fd_labels             = fd_G === nothing ? nothing : fd_labels,
        fd_values             = fd_G === nothing ? nothing : fd_values)

    # ── Correlated T inference (α error + γ data noise + covariance) ────────────
    inf_T = compute_profiled_T_inference(
        theta_hat, gb_param_idx, param_labels_gb;
        dTa           = dTa,
        G_alpha       = G_alpha,
        W             = W,
        Sigma_data    = Sigma_data,
        Var_alpha_sw  = inf_alpha["Var_sandwich"],
        Var_alpha_eff = inf_alpha["Var_eff"],
        Var_alpha_reg = get(inf_alpha, "Var_regcoef", nothing),
        gb_block_ranges = gb_block_ranges,
        output_folder = output_folder,
        industry      = industry,
        step_rel      = step_rel,
        target        = target,
        alpha_labels  = collect(param_labels_gb[alpha_pos]),
        alpha_values  = collect(theta_hat[alpha_param_idx]),
        alpha_se_eff  = collect(inf_alpha["se_eff"]),
        alpha_se_sw   = collect(inf_alpha["se_sw"]),
        N_hat         = N_hat_v,
        # Prefer the JOINT naive SE that inference_summary.txt actually prints (last
        # S entries of its naive block); fall back to the marginal (G_N'W G_N)^{-1}.
        se_N_naive    = begin
            sfd = get(inf_alpha, "se_fd_sw", nothing)
            (sfd !== nothing && N_hat_v !== nothing && length(sfd) >= length(N_hat_v)) ?
                collect(sfd[end-length(N_hat_v)+1:end]) : se_N_naive
        end,
        q_hat         = q_hat_v,
        J_free_gb     = J_free_ok ? J_free_gb : nothing)

    return inf_alpha, inf_T
end


"""
    run_2x2_inference_test(theta_hat, J_sim_gb, J_ana_gb, gb_param_idx,
                           emp_vec_gb, sim_vec_gb, Sigma_data, Sigma_sim,
                           gamma_ref_map, gb_block_ranges, gb_block_names,
                           param_labels, moment_labels, out_dir; label="")

TEST-ONLY noise-decomposition diagnostic. **Off by default** (gated behind
`run_2x2_test` in `main.jl`). Fully standalone: it replicates ONLY the
linear-algebra core of `compute_smm_inference` (G'WG, efficient + sandwich
variance, se_sw, Hansen J = r'Wr) inline and does NOT call it. Writes to its
OWN tree `out_dir` (= `<step>/inference_2x2_test/`) with its OWN report format —
it never touches the production `inference/` outputs.

Crosses two axes at a fixed estimate θ̂:
  - axis 1 (Jacobian): {simulated J (`J_sim_gb`), analytical J (`J_ana_gb`)}
  - axis 2 (weighting): {data-only (Σ_data, Σ_data⁻¹), data+sim (Ω, W_full≈W_step3)}

`J_ana_gb` is now the EXACT closed-form Jacobian (forward-mode AD via
`analytical_jacobian_ad`, no finite-difference step). The Jacobian axis is thus a
clean noisy-simulation vs exact-closed-form contrast: the whole gap on axis 1 is
the Ricardian-selection simulation noise in `J_sim_gb`, with no residual FD
truncation artefact to confound it.

The (analytical, data-only) corner is the GMM-style variance evaluated at the
SMM estimate. Goal: attribute SMM parameter-variance noise to the Jacobian
channel vs the weighting channel.

`gamma_ref_map`, `gb_block_ranges`, `gb_block_names` are accepted for signature
parity with the production inference call site; this test reports se_sw only and
does not produce per-block / γ-reference plots, so they are currently unused.
"""
function run_2x2_inference_test(theta_hat::Vector{Float64},
                                J_sim_gb::Matrix{Float64},
                                J_ana_gb::Matrix{Float64},
                                gb_param_idx::Vector{Int},
                                emp_vec_gb::Vector{Float64},
                                sim_vec_gb::Vector{Float64},
                                Sigma_data::Matrix{Float64},
                                Sigma_sim::Matrix{Float64},
                                gamma_ref_map,
                                gb_block_ranges,
                                gb_block_names,
                                param_labels,
                                moment_labels,
                                out_dir::String; label::String="")

    mkpath(out_dir)

    # ── Build the two weightings internally ──────────────────────────────────
    Omega_full = Sigma_data .+ Sigma_sim
    Omega_full = (Omega_full .+ Omega_full') ./ 2
    Sigma_data_sym = (Sigma_data .+ Sigma_data') ./ 2

    W_full = inv(Symmetric(Omega_full))             # ≈ W_step3
    W_data = inv(Symmetric(Sigma_data_sym))         # GMM-style data-only weight

    p_dim = length(gb_param_idx)
    N_mom = length(emp_vec_gb)

    # ── Inline linear-algebra core (replicates compute_smm_inference) ─────────
    function compute_cell(G, W, Omega)
        p = size(G, 2); Nm = size(G, 1)

        GtWG = Symmetric(G' * W * G)
        GtWG_inv = try
            F = cholesky(GtWG)
            inv(F)
        catch
            F = eigen(GtWG)
            fl = F.values[end] * 1e-10
            λ  = max.(F.values, fl)
            Symmetric(F.vectors * Diagonal(1.0 ./ λ) * F.vectors')
        end
        GtWG_inv = (Matrix(GtWG_inv) .+ Matrix(GtWG_inv)') ./ 2

        Var_eff = GtWG_inv
        middle  = G' * W * Omega * W * G
        Var_sw  = Var_eff * middle * Var_eff
        Var_sw  = (Var_sw .+ Var_sw') ./ 2

        se_eff = sqrt.(max.(diag(Var_eff), 0.0))
        se_sw  = sqrt.(max.(diag(Var_sw),  0.0))

        r      = emp_vec_gb .- sim_vec_gb
        J_stat = (r' * W * r)[1]
        df     = Nm - p
        pval   = df > 0 ? (1.0 - cdf(Chisq(df), J_stat)) : NaN

        sv     = svdvals(G)
        rank_G = count(sv .> sv[1] * 1e-8)

        return (se_eff=se_eff, se_sw=se_sw, J_stat=J_stat, df=df, pval=pval,
                rank_G=rank_G, p=p, N_mom=Nm, ok=true)
    end

    nan_cell = (se_eff=fill(NaN, p_dim), se_sw=fill(NaN, p_dim), J_stat=NaN,
                df=-1, pval=NaN, rank_G=-1, p=p_dim, N_mom=N_mom, ok=false)

    # ── Grid (jrow, wcol) → cell — each wrapped in try/catch ──────────────────
    cell_specs = [
        ("sim", "data",    J_sim_gb, W_data, Sigma_data_sym),
        ("ana", "data",    J_ana_gb, W_data, Sigma_data_sym),
        ("sim", "dataSim", J_sim_gb, W_full, Omega_full),
        ("ana", "dataSim", J_ana_gb, W_full, Omega_full),
    ]
    grid = Dict{Tuple{String,String}, Any}()
    for (jrow, wcol, G, W, Om) in cell_specs
        grid[(jrow, wcol)] = try
            compute_cell(G, W, Om)
        catch e
            @warn "2×2 cell (J=$jrow, Ω=$wcol) failed; recording NaNs." exception=e
            nan_cell
        end
    end

    # ── Save the four se_sw vectors for external plotting ─────────────────────
    se_dict = Dict(
        "simJ_data"    => grid[("sim", "data")].se_sw,
        "anaJ_data"    => grid[("ana", "data")].se_sw,
        "simJ_dataSim" => grid[("sim", "dataSim")].se_sw,
        "anaJ_dataSim" => grid[("ana", "dataSim")].se_sw,
    )
    NPZ.npzwrite(joinpath(out_dir, "se_2x2_$(label).npy"), se_dict)

    # ── Helpers ───────────────────────────────────────────────────────────────
    safemean(v)   = (vv = filter(isfinite, v); isempty(vv) ? NaN : mean(vv))
    safemedian(v) = (vv = filter(isfinite, v); isempty(vv) ? NaN : median(vv))
    msse(jrow, wcol) = safemean(grid[(jrow, wcol)].se_sw)   # mean se_sw shortcut

    # ── Distinct 2×2 report (NOT inference_summary.txt) ───────────────────────
    report_path = joinpath(out_dir, "report_2x2_$(label).txt")
    open(report_path, "w") do io
        println(io, "#"^72)
        println(io, "# 2×2 NOISE-DECOMPOSITION TEST  (test-only, off by default)")
        println(io, "#"^72)
        println(io, "  θ̂ label  : $(isempty(label) ? "(unlabelled)" : label)")
        println(io, "  Date      : $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
        println(io, "  Output    : $out_dir")
        println(io, "  N_moments : $N_mom   |   p_params : $p_dim")
        println(io)
        println(io, "  Axis 1 (rows)    Jacobian : {sim = simulated FD, ana = analytical FD}")
        println(io, "  Axis 2 (cols)    (W, Ω)   : {data = (Σ_data, Σ_data⁻¹),")
        println(io, "                              data+sim = (Ω=Σ_data+Σ_sim, W=Ω⁻¹≈W_step3)}")
        println(io, "  Corner (ana, data) = GMM-style variance evaluated at the SMM estimate.")
        println(io)
        println(io, "  VALIDITY NOTE:")
        println(io, "    at θ̂_1 the estimator was identity-weighted, so se_eff is meaningless")
        println(io, "    and only se_sw is interpretable; at θ̂_2 the (data+sim) column is")
        println(io, "    efficient (se_eff≈se_sw) and the others are non-efficient diagnostics.")
        println(io, "    Report se_sw throughout for comparability.")
        println(io)

        # ── 2×2 tables, one per metric ───────────────────────────────────────
        # Values are pre-formatted to strings (a fixed %s layout) so @printf
        # never receives a runtime-built format string.
        fmt_e(x) = isnan(x)    ? "NaN" : @sprintf("%.6e", x)
        fmt_f(x) = isnan(x)    ? "NaN" : @sprintf("%.6f", x)
        fmt_d(x) = x < 0       ? "NaN" : string(x)

        function print_2x2(io, title, valfun)
            println(io, "-"^72)
            println(io, title)
            @printf(io, "  %-14s  %-18s  %-18s\n", "", "Ω=data", "Ω=data+sim")
            for jrow in ("sim", "ana")
                v_data = valfun(grid[(jrow, "data")])
                v_ds   = valfun(grid[(jrow, "dataSim")])
                @printf(io, "  J=%-12s  %-18s  %-18s\n", jrow, v_data, v_ds)
            end
            println(io)
        end

        print_2x2(io, ">> mean se_sw",   c -> fmt_e(safemean(c.se_sw)))
        print_2x2(io, ">> median se_sw", c -> fmt_e(safemedian(c.se_sw)))
        print_2x2(io, ">> Hansen p",     c -> fmt_f(c.pval))
        print_2x2(io, ">> rank(G)",      c -> fmt_d(c.rank_G))

        # ── Channel decomposition (on mean se_sw) ────────────────────────────
        println(io, "="^72)
        println(io, "CHANNEL DECOMPOSITION  (units: mean se_sw)")
        println(io, "="^72)
        println(io, "  Jacobian channel = (simJ − anaJ) at fixed Ω:")
        @printf(io, "    Ω=data      : %.6e\n", msse("sim","data")    - msse("ana","data"))
        @printf(io, "    Ω=data+sim  : %.6e\n", msse("sim","dataSim") - msse("ana","dataSim"))
        println(io, "  Weighting channel = (data+sim − data) at fixed J:")
        @printf(io, "    J=sim       : %.6e\n", msse("sim","dataSim") - msse("sim","data"))
        @printf(io, "    J=ana       : %.6e\n", msse("ana","dataSim") - msse("ana","data"))
        println(io, "  Total penalty = (simJ,data+sim) vs (anaJ,data) corner-to-corner:")
        @printf(io, "    total       : %.6e\n", msse("sim","dataSim") - msse("ana","data"))
        println(io)

        # ── Per-parameter se_sw, four cells side by side ─────────────────────
        println(io, "="^72)
        println(io, "PER-PARAMETER se_sw  (four cells side by side)")
        println(io, "="^72)
        has_plab = param_labels !== nothing && length(param_labels) == p_dim
        hdr = @sprintf("  %-22s  %-15s  %-15s  %-15s  %-15s",
                       "param", "simJ_data", "anaJ_data", "simJ_dataSim", "anaJ_dataSim")
        println(io, hdr)
        println(io, "  " * "-"^(length(hdr)-2))
        sd = grid[("sim","data")].se_sw
        ad = grid[("ana","data")].se_sw
        sds = grid[("sim","dataSim")].se_sw
        ads = grid[("ana","dataSim")].se_sw
        for i in 1:p_dim
            plab = has_plab ? string(param_labels[i]) : "idx_$(gb_param_idx[i])"
            @printf(io, "  %-22s  %-15.6e  %-15.6e  %-15.6e  %-15.6e\n",
                    plab, sd[i], ad[i], sds[i], ads[i])
        end
        println(io)
        println(io, "#"^72)
    end

    println("2×2 noise-decomposition test ($label) written to: $report_path")
    println("  se vectors saved to: " * joinpath(out_dir, "se_2x2_$(label).npy"))

    return grid
end

# requires compute_prices_analytical → model_analytical.jl must be included before
# this is called (both main.jl and main_gmm.jl include it before any tools usage).
"""
    screen_T_identification(params; J, W, param_labels, label)

T-identification eigen-screen. Pure diagnostic (print-only, no files, no effect on
any estimate/weight/inference output).

When a Jacobian `J` is supplied (β+γ rows × β+T cols, i.e. the `J_gb` slice), an
informativeness eigen-screen of the GMM information matrix is printed BEFORE the
per-sector analytical screen:

  a. Global: `H = Symmetric(J' W J)` with `W = I` when `W === nothing`. Prints
     λ_min(H), λ_max(H), cond = λ_max / max(λ_min, 1e-300), and rank(J) (singular
     values > sv[1]·1e-8). Lines prefixed with `label`.
  b. T-only restriction: T columns are those whose `param_labels` entry starts with
     `"T["`; if `param_labels === nothing`, falls back to columns `N_TAU+1:end`
     (gb column order is β/α first then T — asserted when labels are present).
     Prints λ_min and λ_max of the sub-block `H[T_cols, T_cols]`.

When the T sub-block is non-empty, three mechanism-attribution diagnostics follow,
all computed from the already-built `H`, `T_cols`, `J`, `W`, `param_labels`:

  1. Support of the λ_min eigenvector `v_min` of `H[T,T]`: each T column's sector is
     parsed from its `"T[sname-rname]"` label (split on the LAST '-', sector before),
     and per sector the share of squared mass `Σ v_min[sector]² / Σ v_min²` and the
     dominant sign `sign(Σ v_min[sector])` are printed. Flags CONCENTRATED (one sector
     > 0.8 of the mass → candidate mechanism 3, level-weak sector) vs SPREAD (mixed
     signs across sectors → candidate mechanism 1, inter-sectoral flat direction).
  3. W = I vs W = W_eff: `H_I = J'J` and its `H_I[T,T]` λ_min (and conditioning) are
     printed alongside the W-weighted values. λ_min weak under W_eff but fine under I
     ⇒ mechanism 2 (the W metric de-identifies the T-via-γ_ls parameters); weak under
     both ⇒ intrinsic to J.

After the per-sector analytical screen below runs (building `out`), a cross-check (2)
prints, per sector, the v_min support share/sign side-by-side with that sector's M^s
ratio λ_min/λ_max, so the worst-M^s sector can be matched to the support of the global
weak direction.

Phase-1 scale-normalized / block-diagonality screen (print-only, additive fields):

  P1a. Global normalized conditioning `Hn = D^{-1/2} H D^{-1/2}`, `D = diag(H)` — its
       `λ_min`/`λ_max`/`cond` are printed beside the raw `cond(H)`. A large raw cond
       that collapses under normalization is a scale artifact (Ω, A, β, T live on very
       different scales), not weak identification.
  P1b. Normalized T-block `Hn[T,T]` conditioning, then the per-sector normalized block
       spectrum `Hn[T_s, T_s]` (sector membership from the same `"T[sname-rname]"`
       parse), and a cross-sector coherence table
       `‖Hn[T_s,T_s′]‖₂ / √(λ_min(Hn_s)·λ_min(Hn_s′))` with its max over sector pairs.
       Coherence ≪ 1 licenses the block preconditioning / per-sector concentration of
       Phases 4–5; near/above 1 flags strong inter-sector coupling.

The per-sector analytical screen on `params` is unchanged:
M^s = Σ_dr ω_dr (diag(g_dr) − g_dr g_dr'), restricted to FREE (non-ref) active
regions; smallest eigenpair, plus each region's marginal share and curvature
contribution diag(M) = Σ_dr ω_dr γ(1−γ). Cost ~ Σ_s n_L^2 · R_downstream.

All additions are print-only (prefixed by `label`, eigen on `Symmetric` wrappers); no
estimate, weight matrix, inference output, or file is affected.

Returns `(out, global_eigen)`:
  - `out`           : the per-sector NamedTuple vector (unchanged shape).
  - `global_eigen`  : a NamedTuple summarising the global/T-only eigen-screen, or
                      `nothing` when `J === nothing`. Fields: `eval_min, eval_max,
                      cond, rank, T_eval_min, T_eval_max` (unchanged) plus the new
                      `T_evec_support::Vector{Float64}` (per-sector squared-mass share
                      of v_min), `T_evec_signs::Vector{Int}`, `T_evec_sectors::Vector{Int}`
                      (sector indices aligning the two), and `T_eval_min_identity`
                      (λ_min of H[T,T] under W = I). Phase-1 additive fields:
                      `cond_normalized` (cond of the globally normalized H),
                      `T_cond_normalized` / `T_eval_min_normalized` (Hn[T,T]),
                      `block_sectors::Vector{Int}` with aligned `block_eval_min` /
                      `block_cond` (per-sector normalized T-block λ_min / cond), and
                      `coherence_max` / `coherence_argmax` (largest cross-sector
                      off-block coherence and the attaining pair). New fields are
                      additive — existing fields and all call sites keep working.
"""
function screen_T_identification(params;
        J::Union{Nothing,Matrix{Float64}} = nothing,
        W::Union{Nothing,AbstractMatrix}  = nothing,
        param_labels = nothing,
        label::String = "")

    pfx = isempty(label) ? "" : "[$label] "

    # Mechanism-attribution accumulators (filled in the J block; always defined so the
    # post-loop M^s cross-check and the returned NamedTuple are safe when J === nothing).
    T_evec_support      = Float64[]   # per-sector share of squared v_min mass
    T_evec_signs        = Int[]       # per-sector dominant sign of v_min
    T_evec_sectors      = Int[]       # sector indices aligned with the two vectors above
    T_eval_min_identity = NaN         # λ_min(H_I[T,T]) under W = I

    # Phase-1 scale-normalized / block-diagonality accumulators (always defined so
    # the returned NamedTuple is safe when J === nothing or T_cols is empty).
    cond_normalized       = NaN       # cond of Hn = D^{-1/2} H D^{-1/2} (global)
    T_cond_normalized     = NaN       # cond of Hn[T,T]
    T_eval_min_normalized = NaN       # λ_min of Hn[T,T]
    block_sectors_out     = Int[]     # sectors with a normalized T-block spectrum
    block_eval_min_out    = Float64[] # λ_min(Hn[T_s,T_s]) per sector (aligned)
    block_cond_out        = Float64[] # cond(Hn[T_s,T_s]) per sector (aligned)
    coherence_max         = NaN       # max cross-sector off-block coherence
    coherence_argmax      = (0, 0)    # sector pair attaining coherence_max

    # ── (a)+(b) Global GMM-information eigen-screen (only when J provided) ────────
    global_eigen = nothing
    if J !== nothing
        Wmat = W === nothing ? I : W
        H = Symmetric(J' * (Wmat * J))
        Fh = eigen(H)
        λmin = Fh.values[1]; λmax = Fh.values[end]
        condH = λmax / max(λmin, 1e-300)
        sv = svdvals(J)
        rankJ = count(sv .> sv[1] * 1e-8)
        @printf("%sH=J'WJ: λ_min=%.4e  λ_max=%.4e  cond=%.4e  rank(J)=%d/%d\n",
                pfx, λmin, λmax, condH, rankJ, size(J, 2))

        # ── (P1a) Scale-normalized global conditioning ───────────────────────
        # Raw cond(H) conflates parameter units (Ω, A, β, T live on wildly
        # different scales) with genuine weak identification. Normalizing by
        # D = diag(H) gives Hn = D^{-1/2} H D^{-1/2} with unit diagonal, whose
        # conditioning is scale-free: a large raw cond that collapses under
        # normalization is a scale artifact, not weak identification.
        Hfull = Matrix(H)
        dH    = diag(Hfull)
        dsafe = sqrt.(max.(abs.(dH), 1e-300))
        Dinv  = 1.0 ./ dsafe
        Hn    = Dinv .* Hfull .* Dinv'          # D^{-1/2} H D^{-1/2}
        Fn    = eigen(Symmetric(Hn))
        cond_normalized = Fn.values[end] / max(Fn.values[1], 1e-300)
        @printf("%sNORMALIZED H: λ_min=%.4e  λ_max=%.4e  cond=%.4e   (raw cond=%.4e)\n",
                pfx, Fn.values[1], Fn.values[end], cond_normalized, condH)

        # T-only column restriction
        if param_labels !== nothing
            T_cols = findall(l -> startswith(string(l), "T["), param_labels)
            if !isempty(T_cols)
                @assert T_cols == collect(minimum(T_cols):length(param_labels)) "gb column order violated: T columns must be a contiguous suffix (β/α first, then T)"
                @assert minimum(T_cols) > 1 "gb column order violated: expected β/α column(s) before T columns"
            end
        else
            T_cols = collect((N_TAU + 1):size(J, 2))
        end

        T_emin = NaN; T_emax = NaN
        if isempty(T_cols)
            @printf("%sH[T,T]: no T columns identified — skipped\n", pfx)
        else
            Ft = eigen(Symmetric(Matrix(H)[T_cols, T_cols]))
            T_emin = Ft.values[1]; T_emax = Ft.values[end]
            @printf("%sH[T,T] (%d T cols): λ_min=%.4e  λ_max=%.4e\n",
                    pfx, length(T_cols), T_emin, T_emax)

            # Sector membership of each T column (parsed from "T[sname-rname]",
            # split on the LAST '-'). Shared by the mechanism-1 support map and
            # the Phase-1 per-sector block spectrum / coherence table below.
            col_sector = Int[]
            if param_labels !== nothing
                sname_to_s = Dict(_sector_names[s] => s for s in 1:S)
                for c in T_cols
                    body  = string(param_labels[c])[3:end-1]   # strip "T[" … "]"
                    dash  = findlast('-', body)                # region name after last '-'
                    sname = dash === nothing ? body : body[1:dash-1]
                    push!(col_sector, get(sname_to_s, sname, 0))
                end
            end

            # ── (1) Per-sector support of the λ_min eigenvector of H[T,T] ─────────
            # The weak direction v_min lives in T-column space; mapping its squared
            # mass back to sectors says WHERE the flat direction sits. Concentration
            # on one sector ⇒ that sector is level-weak (mechanism 3); spread with
            # opposite signs ⇒ an inter-sectoral flat direction (mechanism 1).
            v_min = Ft.vectors[:, 1]
            sumsq = max(sum(abs2, v_min), 1e-300)
            if param_labels !== nothing
                T_evec_sectors = sort(unique(col_sector))
                @printf("%sH[T,T] λ_min eigenvector support (share of squared mass | dominant sign):\n", pfx)
                for s in T_evec_sectors
                    mask  = findall(==(s), col_sector)
                    share = sum(abs2, @view v_min[mask]) / sumsq
                    sgn   = sum(@view v_min[mask]) >= 0 ? 1 : -1
                    push!(T_evec_support, share); push!(T_evec_signs, sgn)
                    @printf("%s    sector %d: mass=%.3f  sign=%+d\n", pfx, s, share, sgn)
                end
                if !isempty(T_evec_support)
                    mx, mi = findmax(T_evec_support)
                    if mx > 0.8
                        @printf("%s    → CONCENTRATED on sector %d (%.0f%% of v_min mass) — see M^s cross-check (mechanism 3?)\n",
                                pfx, T_evec_sectors[mi], 100 * mx)
                    else
                        mixed = length(unique(T_evec_signs)) > 1
                        @printf("%s    → SPREAD across %d sectors%s — inter-sectoral flat direction (mechanism 1?)\n",
                                pfx, length(T_evec_sectors), mixed ? " with MIXED signs" : "")
                    end
                end
            else
                @printf("%sH[T,T] λ_min eigenvector support: param_labels=nothing → sector parse skipped\n", pfx)
            end

            # ── (3) W = I vs W = W_eff on the T sub-block ─────────────────────────
            # Recompute the T-block λ_min under identity weighting. If the weak
            # direction is an artifact of the W metric it is ill-conditioned under
            # W_eff but well-conditioned under I (mechanism 2: W de-identifies the
            # T-via-γ_ls parameters); if intrinsic to J it is weak under both.
            JtJ = J' * J
            FtI = eigen(Symmetric(JtJ[T_cols, T_cols]))
            T_eval_min_identity = FtI.values[1]; T_emax_I = FtI.values[end]
            condW = T_emax   / max(T_emin,              1e-300)
            condI = T_emax_I / max(T_eval_min_identity, 1e-300)
            @printf("%sH[T,T] λ_min:  W=W_eff %.4e   W=I %.4e\n", pfx, T_emin, T_eval_min_identity)
            @printf("%sH[T,T] cond :  W=W_eff %.4e   W=I %.4e\n", pfx, condW, condI)
            if condW > 10 * condI
                @printf("%s    → weak T direction is largely a W-metric artifact (mechanism 2: W de-identifies T-via-γ_ls)\n", pfx)
            elseif condW > 1e3 && condI > 1e3
                @printf("%s    → weak T direction is intrinsic to J (present under W and I)\n", pfx)
            end

            # ── (P1b) Normalized T-block conditioning + per-sector spectrum ───────
            # Same scale-normalization applied to the T sub-block, then split by
            # sector. This is the block-diagonality measurement Phases 4–5 rely on:
            # block preconditioning and per-sector T concentration are only licensed
            # if (i) each sector's normalized T-block is well-conditioned and (ii)
            # cross-sector coupling is small.
            HnTT = Hn[T_cols, T_cols]
            Ftn  = eigen(Symmetric(HnTT))
            T_eval_min_normalized = Ftn.values[1]
            T_cond_normalized     = Ftn.values[end] / max(Ftn.values[1], 1e-300)
            @printf("%sNORMALIZED H[T,T]: λ_min=%.4e  cond=%.4e   (raw T cond=%.4e)\n",
                    pfx, T_eval_min_normalized, T_cond_normalized, condW)

            if param_labels !== nothing && !isempty(col_sector)
                HnTTm       = Matrix(HnTT)
                secs        = sort(unique(filter(!=(0), col_sector)))
                sector_local = Dict{Int,Vector{Int}}()   # sector → local T-block indices
                sector_lmin  = Dict{Int,Float64}()        # sector → λ_min(Hn[T_s,T_s])
                @printf("%sNormalized per-sector T-block spectrum:\n", pfx)
                for s in secs
                    loc = findall(==(s), col_sector)
                    isempty(loc) && continue
                    sector_local[s] = loc
                    Fs  = eigen(Symmetric(HnTTm[loc, loc]))
                    lmn = Fs.values[1]; lmx = Fs.values[end]
                    cnd = lmx / max(lmn, 1e-300)
                    sector_lmin[s] = lmn
                    push!(block_sectors_out, s)
                    push!(block_eval_min_out, lmn)
                    push!(block_cond_out, cnd)
                    @printf("%s    sector %d (%d cols): λ_min=%.4e  λ_max=%.4e  cond=%.4e\n",
                            pfx, s, length(loc), lmn, lmx, cnd)
                end

                # Cross-block coherence: ‖Hn[T_s,T_s′]‖₂ / √(λ_min(Hn_s)·λ_min(Hn_s′))
                # ≪ 1 ⇒ the T Hessian is near block-diagonal by sector ⇒ block
                # precond / per-sector concentration (Phases 4–5) are licensed.
                # Near/above 1 ⇒ strong inter-sector coupling; revisit those phases.
                coherence_max = 0.0
                if length(secs) >= 2
                    npairs = 0
                    for i in 1:length(secs)-1, j in (i+1):length(secs)
                        si, sj = secs[i], secs[j]
                        (haskey(sector_local, si) && haskey(sector_local, sj)) || continue
                        B   = HnTTm[sector_local[si], sector_local[sj]]
                        num = opnorm(B, 2)
                        den = sqrt(max(sector_lmin[si], 1e-300) * max(sector_lmin[sj], 1e-300))
                        coh = num / den
                        npairs += 1
                        if coh > coherence_max
                            coherence_max = coh
                            coherence_argmax = (si, sj)
                        end
                    end
                    @printf("%sCross-sector coherence: max=%.4e at sectors (%d,%d) over %d pairs\n",
                            pfx, coherence_max, coherence_argmax[1], coherence_argmax[2], npairs)
                    if coherence_max < 0.1
                        @printf("%s    → T Hessian near block-diagonal (coherence≪1): block precond/concentration licensed\n", pfx)
                    elseif coherence_max < 1.0
                        @printf("%s    → moderate inter-sector coupling (coherence<1): block methods viable, expect some leakage\n", pfx)
                    else
                        @printf("%s    → STRONG inter-sector coupling (coherence≥1): revisit the Phase 4–5 block assumption\n", pfx)
                    end
                end
            end
        end

        global_eigen = (eval_min=λmin, eval_max=λmax, cond=condH, rank=rankJ,
                        T_eval_min=T_emin, T_eval_max=T_emax,
                        T_evec_support=T_evec_support, T_evec_signs=T_evec_signs,
                        T_evec_sectors=T_evec_sectors, T_eval_min_identity=T_eval_min_identity,
                        cond_normalized=cond_normalized,
                        T_cond_normalized=T_cond_normalized,
                        T_eval_min_normalized=T_eval_min_normalized,
                        block_sectors=block_sectors_out,
                        block_eval_min=block_eval_min_out,
                        block_cond=block_cond_out,
                        coherence_max=coherence_max,
                        coherence_argmax=coherence_argmax)
    end

    # ── (c) Per-sector analytical screen (unchanged) ─────────────────────────────
    Ω_L, Ω_s, A, β, T_vec = unpack_params(params)
    T_mat = reshape(T_vec, S, R)
    τ     = build_tau(β)
    pr    = compute_prices_analytical(Ω_L, Ω_s, A, T_mat, τ)
    P_sr, P_r, c_r, Y_r, mu, Φ = pr.P_sr, pr.P_r, pr.c_r, pr.Y_r, pr.mu, pr.Phi

    out = NamedTuple[]
    for s in 1:S
        gidx = SECTOR_GOOD_INDICES[s]; isempty(gidx) && continue
        regs = SECTOR_GOOD_REGIONS[s]; nL = length(regs)

        # downstream weights ω_dr ∝ sector-s nominal input purchase of region dr
        w = [Ω_s[s]*(P_sr[s,dr]/P_r[dr])^(1-nu)*(P_r[dr]/c_r[dr])^(1-lambda)*
             (1-Ω_L)*mu*Y_r[dr] for dr in 1:R_downstream]
        ω = w ./ sum(w)

        # bilateral shares at the CELL level: g[l,dr] = T_l (w_l τ_{l,dr})^{-θ} / Φ_{s,dr}
        G_cell = Matrix{Float64}(undef, nL, R_downstream)
        for (li,l) in enumerate(regs), dr in 1:R_downstream
            g = SR_TO_GOOD[s,l]
            G_cell[li,dr] = T_mat[s,l]*(W_RS_FLAT[g]*τ[l,dr])^(-theta)/Φ[s,dr]
        end

        # Collapse cells onto the T-PARAMETER column space before forming M^s. The
        # estimated block is T_par[s, ·] of width T_COL_DIM — ZE under :ze (where
        # T_GATHER is the identity and this loop reproduces the cell-level matrix
        # exactly), attraction areas under :aa, where several ZE share ONE parameter.
        # M^s is the Hessian of a multinomial log-likelihood in log T, and that
        # structure survives grouping precisely because T is constant within an area,
        # so the aggregated shares g_a = Σ_{l∈a} g_l give the right block. Building it
        # in ZE space under :aa measured the identification of a parameter vector that
        # is not the one being estimated — and `ref` below is a T column, so the
        # reference row was not being dropped either.
        cols    = SECTOR_T_COLS[s]; isempty(cols) && continue
        col_pos = Dict(c => i for (i, c) in enumerate(cols))
        nC = length(cols)
        G = zeros(nC, R_downstream)
        for (li, l) in enumerate(regs)
            c = T_GATHER[l]
            haskey(col_pos, c) || continue
            @views G[col_pos[c], :] .+= G_cell[li, :]
        end

        M = zeros(nC,nC)
        for dr in 1:R_downstream
            gd = @view G[:,dr]; M .+= ω[dr] .* (Diagonal(gd) .- gd*gd')
        end
        γ_marg = G*ω; curv = diag(M)

        ref      = T_REF_REGION[s]
        free_pos = findall(c -> c != ref, cols); isempty(free_pos) && continue
        free     = free_pos
        F = eigen(Symmetric(M[free,free]))
        push!(out, (sector=s, regions=cols[free],
                    gamma_marg=γ_marg[free], curvature=curv[free],
                    eval_min=F.values[1], eval_max=F.values[end],
                    evec_min=F.vectors[:,1]))
        @printf("%ssector %d: M λ_min/λ_max = %.4e  (%d free %s of %d cells)\n",
                pfx, s, F.values[1]/F.values[end], length(free),
                CA_LEVEL === :aa ? "attraction areas" : "ZE", nL)
    end

    # ── (2) Cross-check: global v_min support vs each sector's M^s ratio ─────────
    # Side-by-side so the worst-M^s sector (level-weak, mechanism 3) can be matched
    # against where the global weak direction actually concentrates. A high support
    # share on a low M^s-ratio sector confirms mechanism 3; high support spread over
    # sectors whose M^s ratios are healthy points to mechanism 1 (inter-sectoral).
    if !isempty(T_evec_support)
        ms_ratio = Dict(o.sector => o.eval_min / max(o.eval_max, 1e-300) for o in out)
        ms_min   = Dict(o.sector => o.eval_min for o in out)
        ms_max   = Dict(o.sector => o.eval_max for o in out)

        @printf("%sv_min support  vs  M^s spectrum:\n", pfx)
        for (i, s) in enumerate(T_evec_sectors)
            @printf("%s    sector %d: support=%.3f  sign=%+d  λ_min=%.4e  λ_max=%.4e  ratio=%.4e\n",
                    pfx,
                    s,
                    T_evec_support[i],
                    T_evec_signs[i],
                    get(ms_min, s, NaN),
                    get(ms_max, s, NaN),
                    get(ms_ratio, s, NaN))
        end
    end

    return out, global_eigen
end