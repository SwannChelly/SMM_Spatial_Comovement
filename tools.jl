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
            [i, j, k,k]
            for i in range_beta
            for j in range_beta
            for k in range_beta
            if i <= j <= k
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
        return vcat([[agg_labor_share], agg_industry_share, A, ones(N_beta), ones(S*R)]...)
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
                betas = scaled[beta_start:(beta_start + N_beta - 1)]
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
                      sample_weights::Union{Nothing, Vector{Float64}}=nothing,
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
                           sample_weights::Union{Nothing, Vector{Float64}}=nothing,
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


function distance_bin(d, n_bins=N_beta)
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
                        sample_weights::Union{Nothing, Vector{Float64}}=nothing)
    # Backward compatibility: convert Bool to String
    if method isa Bool
        method = method ? "normalize" : "original"
    end

    t1 = time()
    if params_list == nothing
        params_list = generate_halton_grid(n,2000,false,init_beta)
    end
    f = params -> parallel_SMM_safe(params, false, second_stage, method, true;
                                    u_draws=u_draws, sample_weights=sample_weights)
    results = pmap(f, params_list)
    return params_list,results
end

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


    bin_labels = if N_beta == 5
        ["]20,50]", "]50,100]", "]100,150]", "]150,200]", ">200"]
    elseif N_beta == 4
        ["]50,100]", "]100,150]", "]150,200]", ">200"]  # Example for 4 bins
    else
        ["Bin $i" for i in 1:N_beta]
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
                         sample_weights::Union{Nothing, Vector{Float64}}=nothing)

    folder = joinpath(loop_folder, stage)
    mkpath(folder)

    if best_params == nothing
        best_params = NPZ.npzread(joinpath(folder, "best_params.npy"))
        params_list = [best_params[:, K] for K in 1:K_max]
        params_list, results = train_stage_one(n, nothing, params_list, false;
                                               u_draws=u_draws, sample_weights=sample_weights)
        score = [score[1] != nothing ? score[1][1] : missing for score in results]
        best_index = argmin(score)
        best_params = best_params[:, best_index]
    else
        results = [full_SMM(best_params; u_draws=u_draws, sample_weights=sample_weights)]
        best_index = 1
    end

    # ── Extract empirical & simulated vectors ──

    emp_gamma = vec(emp_gamma_ls)
    sim_gamma = vec(results[best_index][2][5])

    emp_pi = vec(emp_pi_r)
    sim_pi_r = vec(results[best_index][2][3])

    emp_pi_sA = agg_industry_share
    sim_pi_sA = results[best_index][2][2]

    # ── Bubble scatter plots ──

    p1 = bubble_scatter(emp_gamma, sim_gamma;
        xlabel = "Empirical γ_ls",
        ylabel = "Simulated γ_ls",
        title  = "γ_ls: Empirical vs Simulated")

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

    gamma_emp_result = matrix_report(emp_gamma_ls)
    gamma_sim_result = matrix_report(results[best_index][2][5])
    gamma_ls_ = [gamma_emp_result, gamma_sim_result]

    reg_emp = reg_coef
    reg_sim = results[best_index][2][4]
    reg_ = [reg_emp, reg_sim]

    pi_r_emp_result = matrix_report(emp_pi, false)
    pi_r_sim_result = matrix_report(results[best_index][2][3])
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
                                sample_weights::Union{Nothing, Vector{Float64}}=nothing)
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
                                                      u_draws=u_draws, sample_weights=sample_weights)
        
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
                       sample_weights::Union{Nothing, Vector{Float64}}=nothing)

    folder = output_folder * "/"

    # Compute scores for both stages
    top_score_first, min_dist_first, best_simulated_moments, best_parameters_list =
        compute_scores_modular(output_folder, false, max_loop;
                               u_draws=u_draws, sample_weights=sample_weights)

    
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
    build_step3_weight_matrix(theta_hat_1, input_folder; K, output_folder)

DEPRECATED for GMM mode. In analytical GMM (main_gmm.jl), use W = inv(Σ_data) directly;
Σ_sim = 0 by construction so no simulation replications are needed.
This function remains for legacy SMM mode (main.jl).

Assemble the efficient SMM weight matrix W_step3 = (Σ_data + Σ_sim)^{-1}
over γ_ls and reg_coef moments only.

Σ_data is loaded from Sigma_beta_gamma.npy — the joint bootstrap covariance of
reg_coef and γ_ls moments (ordering: β block first, then γ block, matching
BLOCK_RANGES[4] followed by BLOCK_RANGES[5]).
Σ_sim is estimated from K re-seeded full_SMM evaluations at theta_hat_1,
restricted to the same moment indices.

Returns W_step3 of size (N_beta + n_gamma_kept, N_beta + n_gamma_kept).
"""
function build_step3_weight_matrix(theta_hat_1::Vector{Float64}, input_folder::String;
                                   K::Int=10_000,
                                   output_folder::String=".")
    N_moments = length(empirical_moments)

    # ── Gamma+beta moment indices in the masked vector ───────────────────────
    gb_indices = vcat(collect(BLOCK_RANGES[4]), collect(BLOCK_RANGES[5]))
    n_gb = length(gb_indices)

    # ── Load Σ_data: joint bootstrap covariance of β+γ (β block first) ───────
    sigma_file = N_beta == 1 ? "Sigma_beta_gamma_1.npy" : "Sigma_beta_gamma.npy"
    Sigma_full = NPZ.npzread(joinpath(input_folder, sigma_file))

    # ── Reconcile file size with the (possibly thresholded) active set ───────
    # The on-disk Σ was bootstrapped on the PRE-threshold active set. If a
    # gamma_threshold pruned (s,r) pairs, n_gb shrank → drop matching β+γ
    # rows/cols. β block (1:N_beta) is never pruned.
    X_rs_raw          = NPZ.npzread(joinpath(input_folder, "X_rs.npy"))      # (S,R) raw
    T_mask_moment_old = vec(permutedims(X_rs_raw)) .> 0                       # sector-major
    T_mask_moment_new = vec(permutedims(reshape(collect(T_MASK), S, R)))     # thresholded

    # In the old mask, remove reference regions. 
    keep_old = copy(T_mask_moment_old)                                       # active − ref/sector
    for s in 1:S
        ref_r = T_REF_REGION[s]
        ref_r > 0 && (keep_old[(s - 1) * R + ref_r] = false)
    end
    gamma_old_positions = findall(keep_old)                                  # Get indices of the old set (without reference regions)
    survive  = T_mask_moment_new[gamma_old_positions]                        # Get indices of the new set and remove reference regions
    keep_idx = vcat(collect(1:N_beta), N_beta .+ findall(survive))

    n_gb_old = N_beta + length(gamma_old_positions)
    if size(Sigma_full, 1) == n_gb
        Sigma_data = Sigma_full                                              # already regenerated
    elseif size(Sigma_full, 1) == n_gb_old
        Sigma_data = Sigma_full[keep_idx, keep_idx]                          # full file → subset
        @assert size(Sigma_data, 1) == N_beta + count(survive)
        @assert size(Sigma_data, 1) == n_gb "subset $(size(Sigma_data,1)) != n_gb=$n_gb"
        # NOTE: subset is Cov(raw γ); loss uses renormalized γ (factor c_s≈sum_before/
        # sum_after). Raw subset over-weights γ rows by ~c_s^2 → T SEs ~c_s too tight.
        # For exact inference, regenerate Sigma_beta_gamma with the threshold applied.
    else
        error("Sigma_beta_gamma size $(size(Sigma_full,1)) matches neither n_gb=$n_gb " *
              "nor pre-threshold n_gb_old=$n_gb_old. Regenerate it.")
    end

    @assert isapprox(Sigma_data, Sigma_data'; atol=1e-10) "Sigma is non-symmetric"

    # ── Estimate Σ_sim via K re-seeded SMM evaluations ───────────────────────
    println("Estimating Σ_sim from K=$K SMM evaluations at θ̂_1...")
    flush(stdout)



    M_sim_rows = pmap(1:K) do k
        u_k, w_k = generate_stratified_draws(N_rho, n_good;
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

    NPZ.npzwrite(joinpath(step2_dir, "M_sim.npy"),      M_sim)
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
    run_pso_optimization(; kwargs...) -> (best_params, best_fitness)

Unified PSO wrapper for Steps 1 and 3 of three-step SMM.

# Keyword arguments
- `weight_matrix`: SMM weight matrix passed to full_SMM (default: uses global Weight_matrix_custom)
- `skip_initial_beta_search`: if true, skip Stage 0 LHS search (use warm_start_params beta)
- `warm_start_params`: full parameter vector to warm-start Stage 1 PSO (nothing = fresh start)
- `output_subfolder`: subfolder under output_folder for all stage outputs
- `max_loop`: number of refinement loops (default 50)
- `n_particles`, `max_iter_initial`, `max_iter_stage`: PSO configuration
- `beta_search_method`: "log_grid" or "lhs"
- `beta_selection_criterion`: "reg_coef" or "score"
"""
function run_pso_optimization(;
    weight_matrix::Union{Nothing, AbstractMatrix} = nothing,
    skip_initial_beta_search::Bool = false,
    warm_start_params::Union{Nothing, Vector{Float64}} = nothing,
    output_subfolder::String = "step1",
    max_loop::Int = 50,
    n_particles::Int = 100,
    max_iter_initial::Int = 200,
    max_iter_stage::Int = 50,
    beta_search_method::String = "log_grid",
    beta_selection_criterion::String = "reg_coef",
    length_range_beta::Int = 40,
    method::String = "original",
    gamma_beta_only::Bool = false,          # step 3: fix structural params, optimize only beta+T
    moments_loss_gamma_beta::Bool = false,  # step 3: compute loss on gamma_ls + reg_coef moments only
    analytical::Bool = false,              # GMM mode: closed-form moments (no simulation)
    n_quad::Int = 200                      # quadrature nodes for reg_coef block in analytical mode
)
    loop_base = joinpath(output_folder, output_subfolder)
    mkpath(loop_base)

    moment_blocks = moments_loss_gamma_beta ? [4, 5] : nothing   # reg_coef + gamma_ls (β+γ)

    best_params = nothing
    best_fitness = Inf
    stage = 0

    # ── Stage 0: LHS beta search ─────────────────────────────────────────────
    if !skip_initial_beta_search
        println("\n" * "="^70)
        println("[$output_subfolder] STAGE 0: Finding good initial beta values")
        println("="^70)

        beta_min = 1e-3
        beta_max = 10
        if n_coef == 1
            length_range_beta = 10000
        end
        if beta_search_method == "log_grid"
            beta_candidates = generate_initial_betas("log_grid", N_beta, beta_min, beta_max;
                                                     log_grid_length=length_range_beta)
        else
            beta_candidates = generate_initial_betas("lhs", N_beta, beta_min, beta_max;
                                                     lhs_n_samples=20000)
        end
        println("  Generated $(length(beta_candidates)) beta candidates")

        A_init = copy(emp_pi_r_full).^(1/abs(epsilon)) .* regional_wages[N_downstream_per_region .!= 0]
        A_init ./= sum(A_init)
        T_init_nz = vec(T_rs_init)[T_MASK]
        # New layout: [Ω^L | Ω^s | A | β | T] — beta is inserted between A and T
        init_other_prefix = vcat([agg_labor_share], agg_industry_share, A_init)
        expanding_beta = [vcat(init_other_prefix, beta, T_init_nz) for beta in beta_candidates]

        results_ = pmap(p -> parallel_SMM_safe(p; u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS,
                                               W_override=weight_matrix,
                                               analytical=analytical, n_quad=n_quad), expanding_beta)

        if beta_selection_criterion == "reg_coef"
            reg_coefs_sim = [r !== nothing ? r[2][4] : fill(NaN, N_beta) for r in results_]
            reg_distances  = [sum((reg_coef .- rc).^2) for rc in reg_coefs_sim]
            best_idx = argmin(reg_distances)
        else
            scores = [r !== nothing ? r[1][1] : Inf for r in results_]
            best_idx = argmin(scores)
        end
        init_beta = beta_candidates[best_idx]
        println("  Best initial beta: ", round.(init_beta, digits=6))
    else
        @assert warm_start_params !== nothing "skip_initial_beta_search=true requires warm_start_params"
        beta_start_idx = S + R_downstream + 2   # new layout: [Ω^L | Ω^s(S) | A(Rd) | β | T]
        init_beta = warm_start_params[beta_start_idx:(beta_start_idx + N_beta - 1)]
        println("\n[$output_subfolder] Skipping Stage 0: using warm_start beta $(round.(init_beta, digits=6))")
    end

    # ── Stage 1: PSO ─────────────────────────────────────────────────────────
    println("\n" * "="^70)
    println("[$output_subfolder] STAGE 1: Initial PSO")
    println("="^70)

    stage = 0

    if gamma_beta_only
        # Save warm_start_params as a seed folder so train_stage_pso can treat
        # it as a "previous stage" and restrict optimisation to beta+T only.
        @assert warm_start_params !== nothing "gamma_beta_only=true requires warm_start_params"
        seed_folder = joinpath(loop_base, "seed")
        mkpath(seed_folder)
        NPZ.npzwrite(joinpath(seed_folder, "best_params.npy"), reshape(warm_start_params, :, 1))

        println("[$output_subfolder] gamma_beta_only: optimising β+T only (A_r/labor/industry fixed at θ̂_1)")
        best_params, best_fitness, history = train_stage_pso(
            n_particles, max_iter_initial;
            variable_list     = ["beta", "T"],
            last_stage_folder = seed_folder,
            K=1, alpha=0.5, second_stage=false, method=method,
            u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS, weight_matrix=weight_matrix,
            moment_blocks=moment_blocks, analytical=analytical, n_quad=n_quad
        )
    else
        best_params, best_fitness, history = train_stage_pso(
            n_particles, max_iter_initial;
            init_beta      = init_beta,
            variable_list  = nothing,
            last_stage_folder = nothing,
            alpha          = 0.5,
            second_stage   = false,
            method         = method,
            u_draws        = U_DRAWS,
            sample_weights = SAMPLE_WEIGHTS,
            weight_matrix  = weight_matrix,
            warm_start_override = warm_start_params,
            moment_blocks  = moment_blocks,
            analytical     = analytical,
            n_quad         = n_quad
        )
    end

    stage0_folder = joinpath(loop_base, string(stage))
    mkpath(stage0_folder)
    NPZ.npzwrite(joinpath(stage0_folder, "best_params.npy"), reshape(best_params, :, 1))
    generate_report(loop_base, string(stage), 1, nothing, best_params, "";
                    u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS)

    # ── Refinement loops ─────────────────────────────────────────────────────
    alpha_start, alpha_end = 0.3, 0.9
    # gamma_beta_only: one sub-stage per loop (beta+T only); else: three sub-stages
    substages_per_loop = gamma_beta_only ? 1 : 3

    for loop in 1:max_loop
        alpha = alpha_start + (loop - 1) * (alpha_end - alpha_start) / (max_loop - 1)
        past_loop_folder = loop == 1 ? loop_base : joinpath(loop_base, "epoch_$(loop-1)")
        loop_folder = joinpath(loop_base, "epoch_$loop")
        mkpath(loop_folder)

        println("\n[$output_subfolder] LOOP $loop/$max_loop  alpha=$alpha")

        if gamma_beta_only
            # Only optimise β and T; A_r / labor / industry shares are fixed at warm start
            best_params, best_fitness, history = train_stage_pso(
                n_particles, max_iter_stage;
                variable_list     = ["beta", "T"],
                last_stage_folder = joinpath(past_loop_folder, string(stage)),
                K=1, alpha=alpha, second_stage=false, method=method,
                u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS, weight_matrix=weight_matrix,
                moment_blocks=moment_blocks, analytical=analytical, n_quad=n_quad
            )
            stage += 1
            folder = joinpath(loop_folder, string(stage)); mkpath(folder)
            NPZ.npzwrite(joinpath(folder, "best_params.npy"), reshape(best_params, :, 1))
            generate_report(loop_folder, string(stage), 1, ["beta", "T"], best_params, string(alpha);
                            u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS)
        else
            # Sub-stage 1: Productivity
            alpha_prod = 0.7 + 0.2 * alpha
            best_params, best_fitness, history = train_stage_pso(
                n_particles, max_iter_stage;
                variable_list     = ["productivity"],
                last_stage_folder = joinpath(past_loop_folder, string(stage)),
                K=1, alpha=alpha_prod, second_stage=false, method=method,
                u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS, weight_matrix=weight_matrix,
                moment_blocks=moment_blocks, analytical=analytical, n_quad=n_quad
            )
            stage += 1
            folder = joinpath(loop_folder, string(stage)); mkpath(folder)
            NPZ.npzwrite(joinpath(folder, "best_params.npy"), reshape(best_params, :, 1))
            generate_report(loop_folder, string(stage), 1, ["productivity"], best_params, string(alpha_prod);
                            u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS)

            # Sub-stage 2: Spatial structure (β, T)
            best_params, best_fitness, history = train_stage_pso(
                n_particles, max_iter_stage;
                variable_list     = ["beta", "T"],
                last_stage_folder = joinpath(loop_folder, string(stage)),
                K=1, alpha=alpha, second_stage=false, method=method,
                u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS, weight_matrix=weight_matrix,
                moment_blocks=moment_blocks, analytical=analytical, n_quad=n_quad
            )
            stage += 1
            folder = joinpath(loop_folder, string(stage)); mkpath(folder)
            NPZ.npzwrite(joinpath(folder, "best_params.npy"), reshape(best_params, :, 1))
            generate_report(loop_folder, string(stage), 1, ["beta", "T"], best_params, string(alpha);
                            u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS)

            # Sub-stage 3: Technical coefficients
            best_params, best_fitness, history = train_stage_pso(
                n_particles, max_iter_stage;
                variable_list     = ["agg_labor_share_tech", "agg_industry_share_tech"],
                last_stage_folder = joinpath(loop_folder, string(stage)),
                K=1, alpha=alpha, second_stage=false, method=method,
                u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS, weight_matrix=weight_matrix,
                moment_blocks=moment_blocks, analytical=analytical, n_quad=n_quad
            )
            stage += 1
            folder = joinpath(loop_folder, string(stage)); mkpath(folder)
            NPZ.npzwrite(joinpath(folder, "best_params.npy"), reshape(best_params, :, 1))
            generate_report(loop_folder, string(stage), 1,
                            ["agg_labor_share_tech", "agg_industry_share_tech"], best_params, string(alpha);
                            u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS)
        end

        println("  ✓ Loop $loop done. Fitness: $(round(best_fitness, digits=6))")

        # Convergence check
        if loop > 2
            prev_folder = joinpath(loop_base, "epoch_$(loop-1)", string(stage - substages_per_loop))
            prev_params = NPZ.npzread(joinpath(prev_folder, "best_params.npy"))[:, 1]
            param_change = maximum(abs.(best_params .- prev_params) ./ (abs.(prev_params) .+ 1e-10))
            if param_change < 1e-6
                println("  Convergence: Δparams < 1e-6. Stopping.")
                break
            end
        end
    end

    run_reporting(loop_base, max_loop; u_draws=U_DRAWS, sample_weights=SAMPLE_WEIGHTS)
    return best_params, best_fitness
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

# Arguments
- `theta`         : parameter vector at which to evaluate.
- `K`             : number of independent draw replications to average (default 20).
- `param_indices` : `nothing` → all parameters; otherwise restrict to these columns.
- `step_rel`/`step_abs` : FD step h_j = max(|θ_j|·step_rel, step_abs).
- `base_seed`     : seed offset; replication k uses `MersenneTwister(base_seed + k)`.
                    Must not collide with seeds used elsewhere (e.g. Σ_sim).
- `output_subdir` : subfolder under output_folder for saved files (default "step2").
"""
function compute_jacobian(theta::Vector{Float64};
                          K::Int = 50,
                          param_indices::Union{Nothing, Vector{Int}} = nothing,
                          step_rel::Float64 = 1e-2,
                          step_abs::Float64 = 1e-8,
                          output_folder::String = ".",
                          filename::String = "jacobian.npy",
                          base_seed::Int = 0,
                          output_subdir::String = "step2",
                          analytical::Bool = false,
                          n_quad::Int = 200)

    indices   = param_indices === nothing ? collect(1:length(theta)) : param_indices
    n_perturb = length(indices)
    h = [max(abs(theta[j]) * step_rel, step_abs) for j in indices]

    # Pre-build perturbed parameter vectors (shared across replications)
    plus_params  = [copy(theta) for _ in 1:n_perturb]
    minus_params = [copy(theta) for _ in 1:n_perturb]
    for (k, j) in enumerate(indices)
        plus_params[k][j]  += h[k]
        minus_params[k][j] -= h[k]
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
            u_k, w_k = generate_stratified_draws(N_rho, n_good;
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
            J_k[:, kk] = (plus_results[kk] .- minus_results[kk]) ./ (2 * h[kk])
        end

        for kk in 1:n_perturb, mm in 1:N_moments
            if abs(m0_k[mm]) > 1e-12 && abs(theta[indices[kk]]) > 1e-12
                J_elast_k[mm, kk] = (theta[indices[kk]] / m0_k[mm]) * J_k[mm, kk]
            end
        end

        return (J_k, J_elast_k)
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
        error("Non-finite Jacobian: $(length(bad)) entries, first at " *
              "(moment,param)=$(bad[1:min(end,10)]). Likely a 0/0 moment " *
              "(collapsed sector) or a clamped tiny-T column.")
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
        @warn "Clipping $n_clipped negative residual variances to 0."
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