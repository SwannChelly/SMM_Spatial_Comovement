using Printf



"""
    generate_lhs_beta(n_samples, n_beta, lb, ub; seed=42)

Generate LHS samples for beta enforcing monotonicity: beta_1 <= beta_2 <= ... <= beta_K.
Method: generate LHS in [0,1]^K, sort each sample, map to [lb, ub].
"""
function generate_lhs_beta(n_samples::Int, n_beta::Int, lb, ub; seed=42)
    lb_vec = lb isa Number ? fill(Float64(lb), n_beta) : Float64.(lb)
    ub_vec = ub isa Number ? fill(Float64(ub), n_beta) : Float64.(ub)

    raw_matrix = QuasiMonteCarlo.sample(n_samples, zeros(n_beta), ones(n_beta),
                                         LatinHypercubeSample())

    samples = Vector{Vector{Float64}}()
    for i in 1:n_samples
        raw = sort(raw_matrix[:, i])  # Sort to enforce monotonicity
        beta = lb_vec .+ raw .* (ub_vec .- lb_vec)
        push!(samples, beta)
    end
    return samples
end

"""
    generate_log_grid_beta(n_beta, lb, ub, length_range)

Generate log-spaced beta grid with monotonicity constraint (β₁ ≤ β₂ ≤ ... ≤ β_K).
Supports n_beta = 4 (pattern [i,j,k,k]) or n_beta = 5 (pattern [i,j,k,k,k]).
"""
function generate_log_grid_beta(n_beta::Int, lb::Real, ub::Real, length_range::Int)
    range_beta = exp.(range(log(lb), stop=log(ub), length=length_range))

    if n_beta == 1
        expanding_beta = [[x] for x in range_beta]
    elseif n_beta == 4
        expanding_beta = [
            [i, k, k,k]
            for i in range_beta
            for k in range_beta
            for k in range_beta
            if i  <= k
        ]
    elseif n_beta == 5
        expanding_beta = [
            [i, j, k, k, k]
            for i in range_beta
            for j in range_beta
            for k in range_beta
            if i <= j <= k
        ]
    else
        error("Log grid beta generation not implemented for n_beta=$n_beta")
    end

    return expanding_beta
end

"""
    generate_initial_betas(method, n_beta, lb, ub; lhs_n_samples=1500, log_grid_length=20)

Unified interface for generating initial beta candidates.

# Arguments
- `method`: "log_grid" or "lhs"
- `n_beta`: Number of beta parameters
- `lb`, `ub`: Bounds for beta values
- `lhs_n_samples`: Number of LHS samples (only for method="lhs")
- `log_grid_length`: Grid resolution (only for method="log_grid")
"""
function generate_initial_betas(method::String, n_beta::Int, lb::Real, ub::Real;
                                 lhs_n_samples::Int=1500, log_grid_length::Int=20)
    if method == "log_grid"
        return generate_log_grid_beta(n_beta, lb, ub, log_grid_length)
    elseif method == "lhs"
        return generate_lhs_beta(lhs_n_samples, n_beta, lb, ub)
    else
        error("Unknown beta search method: $method. Use 'log_grid' or 'lhs'.")
    end
end


function generate_halton_grid(n_needed::Int, batchsize::Int=1024, init=false, init_beta=ones(5), last_stage_folder=nothing, K=1, variable=nothing, alpha=0.1, second_stage=false)
    """

    Generate a Halton grid of size P x n with P the size of the parameter set and n the number of parameter sets to test.
    This Halton grid function allows condition on the parameters and is much faster than the previous one. 

    """
    A = copy(N_downstream_per_region[N_downstream_per_region .!= 0])
    A ./= sum(A)
    if init
        return vcat([[agg_labor_share], agg_industry_share, A, ones(N_TAU), ones(S*R)]...)
    end
    if last_stage_folder == nothing
        lb_beta, lb_agg_labor_share_tech, lb_agg_industry_share_tech, lb_prod, lb_T = init_beta.*0.5, 0.8*agg_labor_share, 0.8.*agg_industry_share, 0.01.*A, 0.1*ones(S*R)
        ub_beta, ub_agg_labor_share_tech, ub_agg_industry_share_tech, ub_prod, ub_T = init_beta.*2, 1.2*agg_labor_share, 1.2.*agg_industry_share, A.*10, 100*ones(S*R)

        lb = vcat(lb_agg_labor_share_tech, lb_agg_industry_share_tech, lb_prod, lb_beta, lb_T)
        ub = vcat(ub_agg_labor_share_tech, ub_agg_industry_share_tech, ub_prod, ub_beta, ub_T)
        condition = true
    else
        best_params = NPZ.npzread(joinpath(last_stage_folder, "best_params.npy"))[:,K] # Load best params.
        names = [:agg_labor_share_tech, :agg_industry_share_tech, :productivity, :beta, :T]
        vals = unpack_params(best_params)
        params_dict = Dict(names .=> vals)
        
        # Handle variable as either a single string or a list
        variable_list = isa(variable, String) ? [variable] : variable
        
        # Build lb and ub by concatenating bounds for each variable in the list
        lb = vcat([params_dict[Symbol(v)] ./ alpha for v in variable_list]...)
        ub = vcat([params_dict[Symbol(v)] .* alpha for v in variable_list]...)
        
        # Set condition based on whether "beta" is in the variable list
        condition = "beta" in variable_list
        
        # Apply special constraints for specific variables
        for (var_idx, v) in enumerate(variable_list)
            if v == "agg_labor_share_tech"
                # Find the indices corresponding to this variable
                idx_start = var_idx == 1 ? 1 : sum([length(params_dict[Symbol(variable_list[i])]) for i in 1:(var_idx-1)]) + 1
                idx_end = idx_start + length(params_dict[:agg_labor_share_tech]) - 1
                lb[idx_start:idx_end] .= 0.001
                ub[idx_start:idx_end] .= 1
                condition = false
            end
        end
        
        if "T" in variable_list && second_stage
            # Find indices for T in the concatenated arrays
            t_position = findfirst(==("T"), variable_list)
            idx_before_T = t_position == 1 ? 0 : sum([length(params_dict[Symbol(variable_list[i])]) for i in 1:(t_position-1)])
            
            mask = vec(mask_emp_gamma_ls)
            t_length = length(params_dict[:T])
            t_indices = (idx_before_T + 1):(idx_before_T + t_length)
            
            lb_T = lb[t_indices][mask.==1]
            ub_T = ub[t_indices][mask.==1]
            
            # Rebuild lb and ub with masked T values
            lb = vcat(lb[1:idx_before_T], lb_T, (idx_before_T + t_length < length(lb)) ? lb[idx_before_T + t_length + 1:end] : Float64[])
            ub = vcat(ub[1:idx_before_T], ub_T, (idx_before_T + t_length < length(ub)) ? ub[idx_before_T + t_length + 1:end] : Float64[])
        end
    end
    

    
    d = length(lb)
    accepted = Vector{Vector{Float64}}(undef, 0)

    # Create a Halton point generator in dimension d
    hp = HaltonPoint(d)  # yields a lazy sequence of points in [0,1]^d

    idx = 1
    while length(accepted) < n_needed
        # get a batch of raw Halton points
        batch_raw = collect(hp[idx : idx + batchsize - 1])  # Vector of Vectors (each length d)
        # Each point is in [0,1]^d

        for raw in batch_raw
            # scale each component
            scaled = lb .+ (ub .- lb) .* raw

            # apply your condition
            if condition
                beta_start = 1 + S + R_downstream + 1
                betas = scaled[beta_start:(beta_start + N_TAU - 1)]
                if issorted(betas)
                    push!(accepted, scaled)
                    if length(accepted) >= n_needed
                        break
                    end
                end
            else
                push!(accepted, scaled)
                if length(accepted) >= n_needed
                    break
                end
            end
        end

        idx += batchsize
    end
    
    if last_stage_folder != nothing
        names = ["agg_labor_share_tech", "agg_industry_share_tech", "productivity", "beta", "T"]
        variable_list = isa(variable, String) ? [variable] : variable
        
        if "T" in variable_list && second_stage
            accepted = [assign_T_with_mask(params_dict[:T], sample) for sample in accepted]
        end
        
        keyfun(x) = isa(first(keys(params_dict)), Symbol) ? Symbol(x) : x
        
        # Helper function to get the slice of k corresponding to parameter p
        function get_param_slice(k, p)
            if !(p in variable_list)
                return params_dict[keyfun(p)]
            else
                var_idx = findfirst(==(p), variable_list)
                idx_start = var_idx == 1 ? 1 : sum([length(params_dict[Symbol(variable_list[i])]) for i in 1:(var_idx-1)]) + 1
                idx_end = idx_start + length(params_dict[keyfun(p)]) - 1
                return k[idx_start:idx_end]
            end
        end
        
        accepted = [vcat([get_param_slice(k, p) for p in names]...) for k in accepted]
        
        push!(accepted, best_params) # We add the last best parameter. 
        return accepted
    end
    return accepted
end


function assign_T_with_mask(true_T,sample)    
    mask = vec(mask_emp_gamma_ls)
    accept = copy(true_T)
    accept[mask.== 1] = sample
    return accept 
end

function parallel_SMM(params, simulation, second_stage, method;
                      precomputed_tau::Union{Nothing, Matrix{Float64}}=nothing,
                      u_draws::Union{Nothing, Matrix{Float64}}=nothing,
                      sample_weights::Union{Nothing, Matrix{Float64}}=nothing,
                      W_override::Union{Nothing, AbstractMatrix}=nothing,
                      moment_blocks::Union{Nothing, Vector{Int}}=nothing,
                      analytical::Bool=false,
                      n_quad::Int=200)
    return full_SMM(params, simulation, second_stage, method;
                    precomputed_tau=precomputed_tau, u_draws=u_draws, sample_weights=sample_weights,
                    W_override=W_override, moment_blocks=moment_blocks,
                    analytical=analytical, n_quad=n_quad)
end


function parallel_SMM_safe(params, simulation = false, second_stage = false, method = "original", show_err = true;
                           precomputed_tau::Union{Nothing, Matrix{Float64}}=nothing,
                           u_draws::Union{Nothing, Matrix{Float64}}=nothing,
                           sample_weights::Union{Nothing, Matrix{Float64}}=nothing,
                           W_override::Union{Nothing, AbstractMatrix}=nothing,
                           moment_blocks::Union{Nothing, Vector{Int}}=nothing,
                           analytical::Bool=false,
                           n_quad::Int=200)
    # Backward compatibility: convert Bool to String
    if method isa Bool
        method = method ? "normalize" : "original"
    end

    try
        result = parallel_SMM(params, simulation, second_stage, method;
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

function train_stage_one(n, init_beta, params_list = nothing, second_stage = false, method = "original";
                        u_draws::Union{Nothing, Matrix{Float64}}=nothing,
                        sample_weights::Union{Nothing, Matrix{Float64}}=nothing,
                        analytical::Bool=false,
                        n_quad::Int=200)
    # Backward compatibility: convert Bool to String
    if method isa Bool
        method = method ? "normalize" : "original"
    end

    t1 = time()
    if params_list == nothing
        params_list = generate_halton_grid(n,2000,false,init_beta)
    end
    f = params -> parallel_SMM_safe(params, false, second_stage, method, true;
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
    output_file::String,variable,alpha
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
    sim_flat_masked = vcat([vec(results[best_index][2][i]) for i in 1:length(results[best_index][2])]...)[MOMENT_MASK]
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

    generate_dashboard_report(n, agg_labor_share_, agg_industry_share_, gamma_ls_, reg_, pi_r, best_score,
        folder * "/report.txt", variable, alpha)
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
    compute_block_ranges(n_labor, n_industry, n_pi, n_reg, n_gamma, mask)

Compute indices into masked moment vector for each moment block.
Moment order: [labor | industry | pi_r | reg_coef | gamma_ls]
Returns tuple of 5 index vectors (one per block).
"""

function compute_block_ranges(n_labor, n_industry, n_pi, n_reg, n_gamma, mask)
    cuts = cumsum([0, n_labor, n_industry, n_pi, n_reg, n_gamma])

    masked_ranges = ntuple(5) do k
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

    sim_vec = vcat([vec(sim[i]) for i in 1:length(sim)]...)[MOMENT_MASK]
    emp_vec = vec(empirical_moments)
    w = diag(Weight_matrix_custom)

    c = w .* (emp_vec .- sim_vec).^2   # per-moment contributions
    block_totals = [sum(c[r]) for r in BLOCK_RANGES]

    return c, block_totals, sum(c)
end

"""
    reconcile_sigma_data(Sigma_full, input_folder) -> Sigma_data

Reconcile an on-disk β+γ moment covariance (`Sigma_beta_gamma[_1].npy`, or the
`w_beta`/`w_gamma` block-diagonal fallback) with the current, possibly
gamma-thresholded active set, and return it subset to the active β+γ moments in
**β-then-γ order** (`BLOCK_RANGES[4]` then `BLOCK_RANGES[5]`).

The file may have been bootstrapped on the PRE-threshold active set; if a
`gamma_threshold` pruned (s,r) pairs the γ block shrank, so the matching β+γ
rows/cols must be dropped. The β block (`1:N_REG`) is never pruned.

Three-way size branch:
  * `size == n_gb`     → already regenerated post-threshold, use as-is
  * `size == n_gb_old` → pre-threshold full file, subset to surviving (s,r)
  * otherwise          → error (regenerate the file)

NOTE: the subset branch returns Cov(raw γ); the loss uses renormalized γ (factor
`c_s ≈ sum_before/sum_after`), so subset γ rows are over-weighted by ~`c_s^2` and
the resulting T SEs run ~`c_s` too tight. For exact inference, regenerate
`Sigma_beta_gamma` with the threshold applied.

Shared by `build_step3_weight_matrix` (SMM) and `main_gmm.jl` Step 2 (GMM) so the
two paths cannot silently diverge on which moments enter the weight matrix.
"""
function reconcile_sigma_data(Sigma_full::AbstractMatrix, input_folder::String)
    # ── Gamma+beta moment indices in the masked vector ───────────────────────
    gb_indices = vcat(collect(BLOCK_RANGES[4]), collect(BLOCK_RANGES[5]))
    n_gb = length(gb_indices)

    # ── Reconcile file size with the (possibly thresholded) active set ───────
    X_rs_raw          = NPZ.npzread(joinpath(input_folder, "X_rs.npy"))      # (S,R) raw
    T_mask_moment_old = vec(permutedims(X_rs_raw)) .> 0                       # sector-major
    T_mask_moment_new = collect(T_MASK)                                      # thresholded; T_MASK is now s-major (= moment convention)

    # In the old mask, remove reference regions.
    keep_old = copy(T_mask_moment_old)                                       # active − ref/sector
    for s in 1:S
        ref_r = T_REF_REGION[s]
        ref_r > 0 && (keep_old[(s - 1) * R + ref_r] = false)
    end
    gamma_old_positions = findall(keep_old)                                  # Get indices of the old set (without reference regions)
    survive  = T_mask_moment_new[gamma_old_positions]                        # Get indices of the new set and remove reference regions
    keep_idx = vcat(collect(1:N_REG), N_REG .+ findall(survive))

    n_gb_old = N_REG + length(gamma_old_positions)
    if size(Sigma_full, 1) == n_gb
        Sigma_data = Sigma_full                                              # already regenerated
    elseif size(Sigma_full, 1) == n_gb_old
        Sigma_data = Sigma_full[keep_idx, keep_idx]                          # full file → subset
        @assert size(Sigma_data, 1) == N_REG + count(survive) "Sigma subset row count $(size(Sigma_data,1)) != N_REG+count(survive)=$(N_REG+count(survive))"
        @assert size(Sigma_data, 1) == n_gb "subset $(size(Sigma_data,1)) != n_gb=$n_gb"
        # NOTE: subset is Cov(raw γ); loss uses renormalized γ (factor c_s≈sum_before/
        # sum_after). Raw subset over-weights γ rows by ~c_s^2 → T SEs ~c_s too tight.
        # For exact inference, regenerate Sigma_beta_gamma with the threshold applied.
    else
        error("Sigma_beta_gamma size $(size(Sigma_full,1)) matches neither n_gb=$n_gb " *
              "nor pre-threshold n_gb_old=$n_gb_old. Regenerate it.")
    end

    @assert isapprox(Sigma_data, Sigma_data'; atol=1e-10) "Sigma is non-symmetric"
    return Sigma_data
end

function build_step3_weight_matrix(theta_hat_1::Vector{Float64}, input_folder::String;
                                   K::Int=10_000,
                                   output_folder::String=".",
                                   draw_method::Symbol=DRAW_METHOD)
    N_moments = length(empirical_moments)

    # ── Gamma+beta moment indices in the masked vector ───────────────────────
    gb_indices = vcat(collect(BLOCK_RANGES[4]), collect(BLOCK_RANGES[5]))
    n_gb = length(gb_indices)

    # ── Load Σ_data: joint bootstrap covariance of β+γ (β block first) ───────
    # File selection keyed on N_REG (the moment count, not the τ-parameter count N_TAU).
    # The β-block of Σ_data has N_REG rows/cols (one per reg_coef moment), independent of N_TAU.
    sigma_file = N_REG == 1 ? "Sigma_beta_gamma_1_f.npy" : "Sigma_beta_gamma_f.npy"
    Sigma_full = NPZ.npzread(joinpath(input_folder, sigma_file))

    # ── Reconcile file size with the (possibly thresholded) active set ───────
    # Shared with main_gmm.jl Step 2 so SMM and GMM agree on which moments enter W.
    Sigma_data = reconcile_sigma_data(Sigma_full, input_folder)

    # ── Estimate Σ_sim via K re-seeded SMM evaluations ───────────────────────
    println("Estimating Σ_sim from K=$K SMM evaluations at θ̂_1...")
    flush(stdout)

    M_sim_rows = pmap(1:K) do k
        u_k, w_k = generate_draws(N_rho, n_good, draw_method;
                                      randomise=true,
                                      rng=MersenneTwister(k))
        _, moms = full_SMM(theta_hat_1; u_draws=u_k, sample_weights=w_k)
        moms_flat = vcat([vec(moms[i]) for i in 1:5]...)[MOMENT_MASK]
        return moms_flat[gb_indices]
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
    compute_jacobian(theta; K, param_indices, step_rel, step_abs,
                    output_folder, filename, base_seed) -> (J, J_elast, J_sd, J_elast_sd)

Central finite differences of the masked moment vector w.r.t. selected parameters,
averaged across `K` independent stratified-draw replications.

For each replication k = 1..K:
  - Generate fresh stratified draws via `generate_stratified_draws(...; randomise=true,
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

# Step regimes (two)
Columns are partitioned by their flat parameter position:
  - **T columns** (strictly-positive, multiplicatively-entering trade-cost levels;
    flat index `≥ 1 + S + R_downstream + N_TAU + 1`) use a **log-space central step**:
    perturbed points are `θ_j·exp(±δ)` with `δ = step_rel`. The central difference in
    log space, `(m₊ − m₋)/(2δ)`, is converted **back to raw units** by dividing by `θ_j`:
        J[:,k] = (m₊ − m₋) / (2·δ·θ_j)        # chain rule ∂m/∂θ = (∂m/∂lnθ)·(1/θ)
    The log step is scale-invariant (never crosses zero, immune to the additive floor)
    and is purely a numerical-accuracy device — the **stored column is raw `∂m/∂θ_j`**,
    identical in meaning and units to the additive case, so `G'WG` inference is unaffected.
  - **All other columns** (`Ω^L`, `Ω^s`, `A`, `β`/`α`) use the **additive central step**
    `h_j = max(|θ_j|·step_rel, step_abs)`, unchanged.
Both regimes store derivatives in raw units `∂m/∂θ`; `J_elast` is derived from raw `J`.

# Arguments
- `theta`         : parameter vector at which to evaluate.
- `K`             : number of independent draw replications to average (default 20).
- `param_indices` : `nothing` → all parameters; otherwise restrict to these columns.
- `step_rel`/`step_abs` : additive-column FD step `h_j = max(|θ_j|·step_rel, step_abs)`.
                    For log (T) columns, `step_rel` is reinterpreted as the dimensionless
                    log step `δ`; `step_abs` is not used there.
- `base_seed`     : seed offset; replication k uses `MersenneTwister(base_seed + k)`.
                    Must not collide with seeds used elsewhere (e.g. Σ_sim).
- `output_subdir` : subfolder under output_folder for saved files (default "step2").
- `t_log_step`    : route T columns through the log step + raw-unit back-conversion
                    (default `true`). Set `false` to recover the byte-identical additive
                    behaviour for every column.
- `check_symmetry`: print a per-(T-)column forward-vs-backward asymmetry diagnostic
                    (default `false`); flags columns whose one-sided slopes diverge by
                    more than 10× the across-replication `J_sd` (nonlinear/clamped regime).
- `richardson_check`: recompute T columns at `2δ` and report the relative gap to the `δ`
                    estimate (default `false`); diagnostic only, returned `J` is unchanged.

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
                          step_rel::Float64 = 1e-4,
                          step_abs::Float64 = 1e-8,
                          t_log_step_rel = 1e-2,   # step séparé pour colonnes T
                          output_folder::String = ".",
                          filename::String = "jacobian.npy",
                          base_seed::Int = 0,
                          output_subdir::String = "step2",
                          analytical::Bool = false,
                          analytical_ad::Bool = false,
                          ad_validate::Bool = false,
                          n_quad::Int = 200,
                          t_log_step::Bool = true,
                          check_symmetry::Bool = true,
                          richardson_check::Bool = true,
                          richardson_rel_tol = 0.05,
                          draw_method::Symbol = DRAW_METHOD)
    print(DRAW_METHOD)
    indices   = param_indices === nothing ? collect(1:length(theta)) : param_indices
    n_perturb = length(indices)

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
                K = 1, param_indices = indices, step_rel = step_rel, step_abs = step_abs,
                t_log_step_rel = t_log_step_rel, output_folder = output_folder,
                filename = replace(filename, ".npy" => "_fd_check.npy"),
                output_subdir = output_subdir, analytical = true, analytical_ad = false,
                n_quad = n_quad, t_log_step = t_log_step,
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

    # ── Column classification: log-step (T) vs additive-step (everything else) ──
    # T parameters are strictly positive and enter the moments multiplicatively
    # (Fréchet scale T^{1/θ}, shares T(wτ)^{-θ}/Φ), so a scale-invariant log step is
    # both justified and immune to the additive floor. Layout is
    # [Ω^L(1) | Ω^s(S) | A(R_downstream) | β(N_TAU) | T], so the first T flat index is:
    T_first = 1 + S + R_downstream #+ N_TAU +1 ( uncomment to remove alpha)
    use_log = [t_log_step && (indices[k] >= T_first) for k in 1:n_perturb]
    # A stray zero/negative in the log set would give exp-steps of 0 or a 1/θ blow-up;
    # T_MASK already excludes zeros, so this should never fire — fail loudly if it does.
    for k in 1:n_perturb
        if use_log[k]
            @assert theta[indices[k]] > 0 (
                "Log-step Jacobian requires T_{sr} > 0, but theta[$(indices[k])] = " *
                "$(theta[indices[k]]); a zero/negative T column must not take a log step.")
        end
    end

    # Additive step (used for non-log columns); δ is the dimensionless log step.
    h = [max(abs(theta[indices[k]]) * step_rel, step_abs) for k in 1:n_perturb]
    δ = [use_log[k] ? t_log_step_rel : step_rel for k in 1:n_perturb]

    # Pre-build perturbed parameter vectors (shared across replications)
    plus_params  = [copy(theta) for _ in 1:n_perturb]
    minus_params = [copy(theta) for _ in 1:n_perturb]
    for (k, j) in enumerate(indices)
        if use_log[k]
            plus_params[k][j]  = theta[j] * exp(δ[k])
            minus_params[k][j] = theta[j] * exp(-δ[k])
        else
            plus_params[k][j]  += h[k]
            minus_params[k][j] -= h[k]
        end
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
                vcat([vec(m[i]) for i in 1:5]...)[MOMENT_MASK]
            end
        else
            u_k, w_k = generate_draws(N_rho, n_good, draw_method;
                                                 randomise = true,
                                                 rng       = MersenneTwister(base_seed + k))
            eval_one = p -> begin
                _, m = full_SMM(p; u_draws=u_k, sample_weights=w_k)
                vcat([vec(m[i]) for i in 1:5]...)[MOMENT_MASK]
            end
        end

        plus_results  = [eval_one(p) for p in plus_params]
        minus_results = [eval_one(p) for p in minus_params]
        m0_k          = eval_one(theta)

        N_moments = length(m0_k)
        J_k       = zeros(N_moments, n_perturb)
        J_elast_k = zeros(N_moments, n_perturb)

        for kk in 1:n_perturb
            if use_log[kk]
                # Chain-rule conversion to raw units: ∂m/∂θ = (∂m/∂lnθ)·(1/θ).
                # This division by θ_j keeps the stored column in raw `∂m/∂θ` units
                # so G'WG inference is unchanged in scale.
                J_k[:, kk] = (plus_results[kk] .- minus_results[kk]) ./
                             (2 * δ[kk] * theta[indices[kk]])
            else
                J_k[:, kk] = (plus_results[kk] .- minus_results[kk]) ./ (2 * h[kk])
            end
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
                if use_log[kk]
                    dp = theta[indices[kk]] * (exp(δ[kk]) - 1)    # forward raw increment
                    dm = theta[indices[kk]] * (1 - exp(-δ[kk]))   # backward raw increment
                else
                    dp = h[kk]; dm = h[kk]
                end
                J_fwd_k[:, kk] = (plus_results[kk] .- m0_k) ./ dp
                J_bwd_k[:, kk] = (m0_k .- minus_results[kk]) ./ dm
            end
        end

        # Optional Richardson step-doubling on log (T) columns only.
        J_rich_k = nothing
        if richardson_check
            J_rich_k = fill(NaN, N_moments, n_perturb)
            for kk in 1:n_perturb
                use_log[kk] || continue
                j = indices[kk]
                pp = copy(theta); pp[j] = theta[j] * exp(2 * δ[kk])
                mm = copy(theta); mm[j] = theta[j] * exp(-2 * δ[kk])
                J_rich_k[:, kk] = (eval_one(pp) .- eval_one(mm)) ./
                                  (2 * (2 * δ[kk]) * theta[j])
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
            regime  = use_log[col] ? "log" : "additive"
            push!(msgs, "(moment=$mm, param=$(indices[col]) [$regime])")
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
        log_cols = findall(use_log)
        println("\nForward/backward symmetry check (T/log columns only):")
        if isempty(log_cols)
            println("  (no log columns)")
        else
            for col in log_cols
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
        log_cols = findall(use_log)

        # Moment→sector map (γ block only; 0 elsewhere) and param→sector map, both
        # rebuilt from structural consts so the diagnostic is self-contained.
        # Unmasked layout: [labor(1) | industry(S) | pi(R_d) | reg(N_REG) | γ(S*R)],
        # γ slot = (s-1)*R + r (s-major, region-minor) — same convention as T.
        gamma_full_offset = 1 + S + R_downstream + N_REG
        n_before_gamma    = count(MOMENT_MASK[1:gamma_full_offset])
        moment_sector     = zeros(Int, size(J, 1))
        let lg = 0
            for s in 1:S, r in 1:R
                MOMENT_MASK[gamma_full_offset + (s-1)*R + r] || continue
                lg += 1
                moment_sector[n_before_gamma + lg] = s
            end
        end
        active_T_flat = findall(T_MASK)                  # s-major active order
        param_sector  = idx -> begin                     # 0 for non-T params
            idx < T_first && return 0
            t = idx - T_first + 1
            (t < 1 || t > length(active_T_flat)) && return 0
            ((active_T_flat[t] - 1) ÷ R) + 1
        end

        println("\nRichardson step-doubling check (J(δ) vs J(2δ), T columns):")
        println("  [worst-gap moment, its sector vs the param's, and |J| there;")
        println("   a large gap on a ~0 CROSS-sector derivative is a benign artefact]")
        if isempty(log_cols)
            println("  (no log columns)")
        else
            tot_flag = 0; tot_flag_insec = 0
            for col in log_cols
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
    α_col = findfirst(l -> startswith(l, "alpha"), PARAM_LABELS)
    if α_col !== nothing
        for (i, m) in enumerate(BLOCK_RANGES[4])
            μ = J_elast[m, α_col]; σ = J_elast_sd[m, α_col]
            @printf("reg_coef[%d] vs α : elast=%.4e  sd=%.4e  ratio=%.2f\n",
                    i, μ, σ, abs(μ) > 1e-300 ? σ/abs(μ) : NaN)
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
                               moment_labels = nothing)    # NEW: names for kept moments (rows of J))    # optional override; default uses local Var_m

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
    open(joinpath(inf_dir, "inference_summary.txt"), "w") do io
        # Header
        println(io, "="^72)
        println(io, "SMM INFERENCE SUMMARY")
        println(io, "  Industry   : $(isempty(industry) ? "(not specified)" : industry)")
        println(io, "  θ̂_2 source : $(joinpath(output_folder, "theta_hat_2.npy"))")
        println(io, "  K_sim      : $(K_sim == 0 ? "(not recorded)" : string(K_sim))")
        println(io, "  Date       : $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS UTC"))")
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
        _has_plab = param_labels !== nothing && length(param_labels) == p
        header = _has_plab ?
            @sprintf("  %-22s  %-6s  %-12s  %-12s  %-12s  %-8s  %-8s  %-12s  %-12s",
                     "param", "idx", "theta", "se_eff", "se_sw", "ratio", "t-stat", "CI_lo", "CI_hi") :
            @sprintf("  %-6s  %-12s  %-12s  %-12s  %-8s  %-8s  %-12s  %-12s",
                     "idx", "theta", "se_eff", "se_sw", "ratio", "t-stat", "CI_lo", "CI_hi")
        println(io, header)
        println(io, "  " * "-"^(length(header)-2))
        for i in 1:p
            ratio = se_eff[i] > 0 ? se_sw[i] / se_eff[i] : NaN
            if _has_plab
                @printf(io, "  %-22s  %-6d  %-12.6f  %-12.6f  %-12.6f  %-8.4f  %-8.4f  %-12.6f  %-12.6f\n",
                        param_labels[i], param_indices[i],
                        theta_active[i], se_eff[i], se_sw[i],
                        isnan(ratio) ? -999.0 : ratio, t_stats[i],
                        ci_95[i, 1], ci_95[i, 2])
            else
                @printf(io, "  %-6d  %-12.6f  %-12.6f  %-12.6f  %-8.4f  %-8.4f  %-12.6f  %-12.6f\n",
                        param_indices[i],
                        theta_active[i], se_eff[i], se_sw[i],
                        isnan(ratio) ? -999.0 : ratio, t_stats[i],
                        ci_95[i, 1], ci_95[i, 2])
            end
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
        "t_stats"       => t_stats,
        "ci_95"         => ci_95,
        "se_m_fitted"   => se_m_fitted,
        "se_m_resid"    => se_m_resid,
        "J_stat"        => J_stat,
        "df"            => df,
        "pval"          => pval,
    )
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

        # bilateral shares g[l,dr] = T_l (w_l τ_{l,dr})^{-θ} / Φ_{s,dr}
        G = Matrix{Float64}(undef, nL, R_downstream)
        for (li,l) in enumerate(regs), dr in 1:R_downstream
            g = SR_TO_GOOD[s,l]
            G[li,dr] = T_mat[s,l]*(W_RS_FLAT[g]*τ[l,dr])^(-theta)/Φ[s,dr]
        end

        M = zeros(nL,nL)
        for dr in 1:R_downstream
            gd = @view G[:,dr]; M .+= ω[dr] .* (Diagonal(gd) .- gd*gd')
        end
        γ_marg = G*ω; curv = diag(M)

        ref  = T_REF_REGION[s]
        free = findall(!=(ref), regs); isempty(free) && continue
        F = eigen(Symmetric(M[free,free]))
        push!(out, (sector=s, regions=regs[free],
                    gamma_marg=γ_marg[free], curvature=curv[free],
                    eval_min=F.values[1], eval_max=F.values[end],
                    evec_min=F.vectors[:,1]))
        @printf("%ssector %d: M λ_min/λ_max = %.4e\n", pfx, s, F.values[1]/F.values[end])
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