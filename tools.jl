using Printf


# function generate_halton_grid(n_needed::Int, batchsize::Int=1024,init = false,init_beta = ones(5),last_stage_folder = nothing,K = 1 ,variable = nothing,alpha = 0.1,second_stage = false)
#     """

#     Generate a Halton grid of size P x n with P the size of the parameter set and n the number of parameter sets to test.
#     This Halton grid function allows condition on the parameters and is much faster than the previous one. 

#     """
#     A = copy(N_downstream_per_region[N_downstream_per_region .!= 0])
#     A ./= sum(A)
#     if init
#         return vcat([ones(5),[agg_labor_share],agg_industry_share,A,ones(S*R)]...)
#     end
#     if last_stage_folder == nothing
#         lb_beta,lb_agg_labor_share_tech,lb_agg_industry_share_tech,lb_prod,lb_T = init_beta.*0.5,0.8*agg_labor_share,0.8.*agg_industry_share,0.01.*A,0.1*ones(S*R)
#         ub_beta,ub_agg_labor_share_tech,ub_agg_industry_share_tech,ub_prod,ub_T = init_beta.*2,1.2*agg_labor_share,1.2.*agg_industry_share,A.*10,100*ones(S*R)
        
#         lb = vcat(lb_beta,lb_agg_labor_share_tech,lb_agg_industry_share_tech,lb_prod,lb_T)
#         ub = vcat(ub_beta,ub_agg_labor_share_tech,ub_agg_industry_share_tech,ub_prod,ub_T)
#         condition = true
#     else
#         best_params = NPZ.npzread(joinpath(last_stage_folder, "best_params.npy"))[:,K] # Load best params.
#         names = [:beta, :agg_labor_share_tech, :agg_industry_share_tech, :productivity, :T]
#         vals = unpack_params(best_params)
#         params_dict = Dict(names .=> vals)
#         lb = (1/alpha).*params_dict[Symbol(variable)]
#         ub = (alpha).*params_dict[Symbol(variable)]
#         if variable == "beta"
#             condition = true
#         elseif variable == "agg_labor_share_tech"
#             lb = 0.001
#             ub = 1
#             condition = false
#         else 
#             condition = false
#         end
#         if variable == "T" && second_stage
#             mask = vec(mask_emp_gamma_ls)
#             lb = lb[mask.==1]
#             ub = ub[mask.==1]
#         end
#     end
    

    
#     d = length(lb)
#     accepted = Vector{Vector{Float64}}(undef, 0)

#     # Create a Halton point generator in dimension d
#     hp = HaltonPoint(d)  # yields a lazy sequence of points in [0,1]^d

#     idx = 1
#     while length(accepted) < n_needed
#         # get a batch of raw Halton points
#         batch_raw = collect(hp[idx : idx + batchsize - 1])  # Vector of Vectors (each length d)
#         # Each point is in [0,1]^d

#         for raw in batch_raw
#             # scale each component
#             scaled = lb .+ (ub .- lb) .* raw

#             # apply your condition
#             #if (scaled[1] < scaled[4] < scaled[2] < scaled[5] < scaled[3])  # Here we force the exploration of a parameter set where betas are in a specific order.
#             if condition
#                 if (scaled[1]  < scaled[2]) & (scaled[2]  < scaled[3]) & (scaled[3]  < scaled[4]) & (scaled[4]  < scaled[5])  # Condition
#                     push!(accepted, scaled)
#                     if length(accepted) >= n_needed
#                         break
#                     end
#                 end
#             else
#                 push!(accepted, scaled)
#                     if length(accepted) >= n_needed
#                         break
#                 end
#             end
#         end

#         idx += batchsize
#     end
#     if last_stage_folder != nothing
#         names = ["beta", "agg_labor_share_tech", "agg_industry_share_tech", "productivity", "T"]
#         # make sure keys match the dict type
#         #return accepted,params_dict[:T]
#         if variable == "T" && second_stage
#             accepted = [assign_T_with_mask(params_dict[:T],sample) for sample in accepted ]
#         end
#         keyfun(x) = isa(first(keys(params_dict)), Symbol) ? Symbol(x) : x
#         accepted = [ vcat([ (p != variable ? params_dict[keyfun(p)] : k) for p in names ]...) for k in accepted ]
#         push!(accepted,best_params) # We add the last best parameter. 
#         return accepted
#     end
#     return accepted
# end

function generate_halton_grid(n_needed::Int, batchsize::Int=1024, init=false, init_beta=ones(5), last_stage_folder=nothing, K=1, variable=nothing, alpha=0.1, second_stage=false)
    """

    Generate a Halton grid of size P x n with P the size of the parameter set and n the number of parameter sets to test.
    This Halton grid function allows condition on the parameters and is much faster than the previous one. 

    """
    A = copy(N_downstream_per_region[N_downstream_per_region .!= 0])
    A ./= sum(A)
    if init
        return vcat([ones(N_beta), [agg_labor_share], agg_industry_share, A, ones(S*R)]...)
    end
    if last_stage_folder == nothing
        lb_beta, lb_agg_labor_share_tech, lb_agg_industry_share_tech, lb_prod, lb_T = init_beta.*0.5, 0.8*agg_labor_share, 0.8.*agg_industry_share, 0.01.*A, 0.1*ones(S*R)
        ub_beta, ub_agg_labor_share_tech, ub_agg_industry_share_tech, ub_prod, ub_T = init_beta.*2, 1.2*agg_labor_share, 1.2.*agg_industry_share, A.*10, 100*ones(S*R)
        
        lb = vcat(lb_beta, lb_agg_labor_share_tech, lb_agg_industry_share_tech, lb_prod, lb_T)
        ub = vcat(ub_beta, ub_agg_labor_share_tech, ub_agg_industry_share_tech, ub_prod, ub_T)
        condition = true
    else
        best_params = NPZ.npzread(joinpath(last_stage_folder, "best_params.npy"))[:,K] # Load best params.
        names = [:beta, :agg_labor_share_tech, :agg_industry_share_tech, :productivity, :T]
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
                betas = scaled[1:N_beta]
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
        names = ["beta", "agg_labor_share_tech", "agg_industry_share_tech", "productivity", "T"]
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

function parallel_SMM(params, simulation, second_stage, method; precomputed_tau::Union{Nothing, Array{Float64,3}}=nothing)
    return full_SMM(params, simulation, second_stage, method; precomputed_tau=precomputed_tau)
end


function parallel_SMM_safe(params, simulation = false, second_stage = false, method = "original", show_err = true; precomputed_tau::Union{Nothing, Array{Float64,3}}=nothing)
    # Backward compatibility: convert Bool to String
    if method isa Bool
        method = method ? "normalize" : "original"
    end
    
    try
        result = parallel_SMM(params, simulation, second_stage, method; precomputed_tau=precomputed_tau) # Run the SMM in parallel.

        return result
    catch e
        # If an error occurs, return a message or a placeholder result
        if show_err
            println("ERROR!!")
            println(e)
        end
        return nothing  # You can also return an error message or a custom value
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
    else
        error("Unsupported number of distance bins: $n_bins")
    end
end

function train_stage_one(n, init_beta, params_list = nothing, second_stage = false, method = "original")
    # Backward compatibility: convert Bool to String
    if method isa Bool
        method = method ? "normalize" : "original"
    end

    t1 = time()
    if params_list == nothing
        params_list = generate_halton_grid(n,2000,false,init_beta)
    end
    f = params -> parallel_SMM_safe(params, false, second_stage, method, true)
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

function print_(x)
    return x
end


function add_first_element(v::Vector{Float64})
    first_element = 1.0 - sum(v)
    return [first_element; v]
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

function save_stage_best_params(params_list,results,loop_folder,stage,K = 1,best_params = nothing)
    # Collect the results of the calibration and store them. 
    folder = joinpath(loop_folder, stage)
    mkpath(folder) 
    if best_params == nothing
        score = [score[1] != nothing ? score[1][1] : missing for score in results]
        valid_indices = findall(!ismissing, score)
        sorted_indices = sort(valid_indices, by = i -> score[i])
        top_indices = sorted_indices[1:min(K, length(sorted_indices))]
        best_params = reduce(hcat,params_list[top_indices])
    end
    npzwrite(joinpath(folder, "best_params.npy"), best_params)

end

function load_parameters_dict(folder = nothing,params = nothing)
    if params == nothing
        params = NPZ.npzread(joinpath(folder, "best_params.npy")) 
    end
    names = [:beta, :agg_labor_share_tech, :agg_industry_share_tech, :productivity, :T]
    vals = unpack_params(params)
    params_dict = Dict(names .=> vals)
    return params_dict
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


function generate_report(loop_folder, stage, n, variable=nothing, best_params=nothing, alpha="")

    folder = joinpath(loop_folder, stage)
    mkpath(folder)

    if best_params == nothing
        best_params = NPZ.npzread(joinpath(folder, "best_params.npy"))
        params_list = [best_params[:, K] for K in 1:K_max]
        params_list, results = train_stage_one(n, nothing, params_list, false)
        score = [score[1] != nothing ? score[1][1] : missing for score in results]
        best_index = argmin(score)
        best_params = best_params[:, best_index]
    else
        results = [full_SMM(best_params)]
        best_index = 1
    end

    # ── Extract empirical & simulated vectors ──

    emp_gamma = vec(emp_gamma_ls)
    sim_gamma = vec(results[best_index][2][3])

    emp_pi = vec(emp_pi_r)
    sim_pi_r = vec(results[best_index][2][5])

    emp_pi_sA = agg_industry_share[2:end]
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

    npzwrite(joinpath(folder, "pi_r.npy"), results[best_index][2][5])
    npzwrite(joinpath(folder, "productivity.npy"), unpack_params(best_params)[4])

    # ── Text report (unchanged) ──

    agg_labor_share_emp = agg_labor_share
    agg_labor_share_sim = results[best_index][2][1][1]
    agg_labor_share_ = [agg_labor_share_emp, agg_labor_share_sim]

    agg_industry_share_ = [agg_industry_share, add_first_element(results[best_index][2][2])]

    gamma_emp_result = matrix_report(emp_gamma_ls)
    gamma_sim_result = matrix_report(results[best_index][2][3])
    gamma_ls_ = [gamma_emp_result, gamma_sim_result]

    reg_emp = reg_coef
    reg_sim = results[best_index][2][4]
    reg_ = [reg_emp, reg_sim]

    pi_r_emp_result = matrix_report(emp_pi, false)
    pi_r_sim_result = matrix_report(results[best_index][2][5])
    pi_r = [pi_r_emp_result, pi_r_sim_result]

    best_score = results[best_index][1][1]

    generate_dashboard_report(n, agg_labor_share_, agg_industry_share_, gamma_ls_, reg_, pi_r, best_score,
        folder * "/report.txt", variable, alpha)
end


function run_stage(variable,n,alpha,stage,loop_folder,second_stage)
    variable_str = isa(variable, String) ? variable : join(variable, ", ")
    print("Variable is: "*variable_str*", n = "*string(n)*" and stage = "*string(stage)*"\n")
    best_params = Any[]
    for K in 1:K_max
        params_list = generate_halton_grid(n,2000,false,nothing,joinpath(loop_folder,string(stage)),K,variable,alpha)
        params_list,results = train_stage_one(n,nothing,params_list,second_stage)
        score = [score[1] != nothing ? score[1][1] : missing for score in results]
        push!(best_params,params_list[argmin(score)])
    end
    stage += 1
    folder = joinpath(loop_folder, string(stage))
    mkpath(folder) 
    npzwrite(joinpath(folder, "best_params.npy"), hcat(best_params...))
    generate_report(loop_folder,string(stage),n)    
    return stage
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
function compute_scores_modular(output_folder::String, second_stage::Bool, max_loop::Union{Int, Nothing}=nothing)
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
        params_list_stage, results = train_stage_one(1, nothing, params_list_stage, second_stage)
        
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
function run_reporting(output_folder::String, max_loop::Union{Int, Nothing}=nothing; save_plots::Bool=true)

    folder = output_folder * "/"
    
    # Compute scores for both stages
    top_score_first, min_dist_first, best_simulated_moments, best_parameters_list = 
        compute_scores_modular(output_folder, false, max_loop)

    
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


############ Old functions ##############



function old_generate_halton_grid(n)


        """

        Generate a Halton grid of size P x n with P the size of the parameter set and n the number of parameter sets to test.
        This Halton grid function doesn't allow for conditions on the parameters. 

        """
        # beta,theta,nu_s,nu,lambda,epsilon,productivity,T
        A = copy(N_downstream_per_region[N_downstream_per_region .!= 0])
        A ./= sum(A)
        #A = ones(size(A)[1])
        #lb_beta,lb_agg_labor_share_tech,lb_agg_industry_share_tech,lb_prod,lb_T, = 0.25,0.5,0.8.*agg_industry_share,0.8*A,0.1*ones(S*R)
        #ub_beta,ub_agg_labor_share_tech,ub_agg_industry_share_tech,ub_prod,ub_T, = 1,1,1.2.*agg_industry_share,1.2*A,20*ones(S*R)

        lb_beta,lb_agg_labor_share_tech,lb_agg_industry_share_tech,lb_prod,lb_T, = ones(5).*2,0.8*agg_labor_share,0.8.*agg_industry_share,A,0.1*ones(S*R)
        ub_beta,ub_agg_labor_share_tech,ub_agg_industry_share_tech,ub_prod,ub_T, = ones(5).*20,1.2*agg_labor_share,1.2.*agg_industry_share,10*A,100*ones(S*R)


        lb = Any[vcat(lb_beta,lb_agg_labor_share_tech,lb_agg_industry_share_tech,lb_prod,lb_T)...]
        ub = Any[vcat(ub_beta,ub_agg_labor_share_tech,ub_agg_industry_share_tech,ub_prod,ub_T)...]
        
        halton_samples = QuasiMonteCarlo.sample(n, lb, ub, HaltonSample())  # n rows, 8 cols
        halton_samples =  [halton_samples[:,i] for i in range(1,n)]

        return halton_samples
        # This will create a vector of 100 tuples, each with 8 parameters
        #return [(halton_samples[1,i],halton_samples[2,i],halton_samples[3:2+(S),1]/sum(halton_samples[3:2+(S),1]),halton_samples[(S+3):(size(ub_prod)[1]+S+2),i],halton_samples[(size(ub_prod)[1]+(S+3)):(size(ub_prod)[1]+size(lb_T)[1]+S+2),i]) for i in 1:(n-1)]
end