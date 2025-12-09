

using Printf


function generate_halton_grid(n_needed::Int, batchsize::Int=1024,init = false,init_beta = ones(5),last_stage_folder = nothing,K = 1 ,variable = nothing,alpha = 0.1,second_stage = false)
    """

    Generate a Halton grid of size P x n with P the size of the parameter set and n the number of parameter sets to test.
    This Halton grid function allows condition on the parameters and is much faster than the previous one. 

    """
    A = copy(N_downstream_per_region[N_downstream_per_region .!= 0])
    A ./= sum(A)
    if init
        return vcat([ones(5),[agg_labor_share],agg_industry_share,A,ones(S*R)]...)
    end
    if last_stage_folder == nothing
        lb_beta,lb_agg_labor_share_tech,lb_agg_industry_share_tech,lb_prod,lb_T = init_beta.*0.5,0.8*agg_labor_share,0.8.*agg_industry_share,0.01.*A,0.1*ones(S*R)
        ub_beta,ub_agg_labor_share_tech,ub_agg_industry_share_tech,ub_prod,ub_T = init_beta.*2,1.2*agg_labor_share,1.2.*agg_industry_share,A.*10,100*ones(S*R)
        
        lb = vcat(lb_beta,lb_agg_labor_share_tech,lb_agg_industry_share_tech,lb_prod,lb_T)
        ub = vcat(ub_beta,ub_agg_labor_share_tech,ub_agg_industry_share_tech,ub_prod,ub_T)
        condition = true
    else
        best_params = NPZ.npzread(joinpath(last_stage_folder, "best_params.npy"))[:,K] # Load best params.
        names = [:beta, :agg_labor_share_tech, :agg_industry_share_tech, :productivity, :T]
        vals = unpack_params(best_params)
        params_dict = Dict(names .=> vals)
        lb = (1/alpha).*params_dict[Symbol(variable)]
        ub = (alpha).*params_dict[Symbol(variable)]
        if variable == "beta"
            condition = true
        elseif variable == "agg_labor_share_tech"
            lb = 0.001
            ub = 1
            condition = false
        else 
            condition = false
        end
        if variable == "T" && second_stage
            mask = vec(mask_emp_gamma_ls)
            lb = lb[mask.==1]
            ub = ub[mask.==1]
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
            #if (scaled[1] < scaled[4] < scaled[2] < scaled[5] < scaled[3])  # Here we force the exploration of a parameter set where betas are in a specific order.
            if condition
                if (scaled[1]  < scaled[2]) & (scaled[2]  < scaled[3]) & (scaled[3]  < scaled[4]) & (scaled[4]  < scaled[5])  # Condition
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
        # make sure keys match the dict type
        #return accepted,params_dict[:T]
        if variable == "T" && second_stage
            accepted = [assign_T_with_mask(params_dict[:T],sample) for sample in accepted ]
        end
        keyfun(x) = isa(first(keys(params_dict)), Symbol) ? Symbol(x) : x
        accepted = [ vcat([ (p != variable ? params_dict[keyfun(p)] : k) for p in names ]...) for k in accepted ]
        push!(accepted,best_params) # We add the last best parameter. 
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

function parallel_SMM(params,simulation,second_stage)
    print(simulation,second_stage)
    return full_SMM(params,simulation,second_stage)
end


function parallel_SMM_safe(params,simulation = false,second_stage=false,show_err = true)
    try
        result = parallel_SMM(params,simulation,second_stage) # Run the SMM in parallel. 

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


function distance_bin(d)
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
        return 0   # for ≤ 20, outside bins
    end
end

function train_stage_one(n,init_beta,params_list = nothing,second_stage = false)

    t1 = time()
    if params_list == nothing
        params_list = generate_halton_grid(n,2000,false,init_beta)
    end
    f = params -> parallel_SMM_safe(params, false, second_stage, true)
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

    reg_df = DataFrame(
        bins = ["]20,50]","]50,100]", "]100,150]", "]150,200]", ">200"],
        empirical = reg_emp,
        simulated = reg_sim
    )

    date = now()
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

function generate_report(loop_folder,stage,n,K=1,variable = nothing,best_params= nothing,alpha = "")

    folder = joinpath(loop_folder, stage)
    mkpath(folder) 

    if best_params == nothing
        best_params = NPZ.npzread(joinpath(folder, "best_params.npy"))# Load best params.
        params_list = [best_params[:,K] for K in 1:50]
        params_list,results = train_stage_one(n,nothing,params_list,false)
        score = [score[1] != nothing ? score[1][1] : missing for score in results]
        best_index = argmin(score)
        best_params = best_params[:,best_index]
    else 
        results = [full_SMM(best_params)] # Get simulated moments. Since we set the seed in model_CP we ensure reproducibility.
        best_index = 1
    end
    # print(best_index,score[best_index])
    # Prepare vectors.
    # Vectorize and filter
    emp_gamma = vec(emp_gamma_ls)
    sim_gamma = vec(results[best_index][2][3])

    # Filter non-zero values
    emp_gamma_nz = emp_gamma[emp_gamma .>= 0.01]
    sim_gamma_nz = sim_gamma[sim_gamma .>= 0.01]
    emp_pi = vec(emp_pi_r)
    sim_pi_r = vec(results[best_index][2][5])
    emp_pi_sA = agg_industry_share[2:end]
    sim_pi_sA = results[best_index][2][2]

    # Define thresholds
    x_chi = quantile(emp_gamma[emp_gamma.!=0],0.9)
    x_pi_r = 0.0
    x_pi_sA = 0.0

    # Define x_vals only for strictly positive values
    xmin = minimum([minimum(emp_gamma_nz), minimum(sim_gamma_nz)])
    xmax = maximum([maximum(emp_gamma_nz), maximum(sim_gamma_nz)])
    x_vals = range(xmin, xmax, length=300)
    x_vals = x_vals[x_vals .> 0]  # avoid x=0
    # Compute cdf and survival
    F_emp = ecdf(emp_gamma_nz)
    F_sim = ecdf(sim_gamma_nz)
    ccdf_emp = F_emp.(x_vals)
    ccdf_sim = F_sim.(x_vals)
    keep = (ccdf_emp .> 0) .& (ccdf_sim .> 0) .& (x_vals .> 0) # Filter to avoid log(0)

    # Plot only where both x and y values are > 0
    p1 = plot(x_vals[keep], ccdf_emp[keep], label="Empirical", lw=2, color=:blue,
        xscale=:log10, yscale=:log10, xlabel="gamma_{ls}", ylabel="CDF",
        title="Log-Log Complementary CDF of gamma_{ls}")
    plot!(p1,x_vals[keep], ccdf_sim[keep], label="Simulated", lw=2, color=:red)

    emp_vals = emp_gamma[emp_gamma .> x_chi]
    sim_vals = sim_gamma[sim_gamma .> x_chi]
    xmin = x_chi
    xmax = maximum([maximum(emp_vals), maximum(sim_vals)])
    nbins = 30
    bin_edges = range(xmin, xmax; length=nbins+1)

    # Histogram with fixed bins
    p2 = histogram(emp_vals,
        alpha=0.5, bins=bin_edges, label="Empirical", color=:blue, title="gamma_{ls}",
        xlims=(xmin, xmax))

    histogram!(p2, sim_vals,
        alpha=0.5, bins=bin_edges, label="Simulated", color=:red)


    # Histogram 2: pi_r
    p3 = histogram(emp_pi[emp_pi .> x_pi_r],
        alpha=0.5, bins=30, label="Empirical", color=:blue, title="pi_r",
        xlims=(x_pi_r, maximum([maximum(emp_pi), maximum(sim_pi_r)])))

    histogram!(p3, sim_pi_r[sim_pi_r .> x_pi_r],
        alpha=0.5, bins=30, label="Simulated", color=:red)

    # Histogram 4: pi_s
    p4 = histogram(emp_pi_sA[emp_pi_sA .> x_pi_sA],
        alpha=0.5, bins=30, label="Empirical", color=:blue, title="pi_s",
        xlims=(x_pi_sA, maximum([maximum(emp_pi_sA), maximum(sim_pi_sA)])))

    histogram!(p4, sim_pi_sA[sim_pi_sA .> x_pi_sA],
    alpha=0.5, bins=30, label="Simulated", color=:red)

    # Combine into a 2x2 subplot layout (fourth plot left blank)
    plot(p1, p2, p3,p4 , layout=(2,2), size=(800,800))

    savefig(joinpath(folder, "dashboard.png"))

    # Bellow we store pi_r, the productivity, and trade flows in numpy to plot them with python. 
    npzwrite(joinpath(folder, "pi_r.npy"), results[best_index][2][5])
    npzwrite(joinpath(folder, "productivity.npy"), unpack_params(best_params)[4])


    beta,agg_labor_share_tech,agg_industry_share_tech,productivity_,T_ = unpack_params(best_params)
    data = beta,agg_labor_share_tech,agg_industry_share_tech,productivity_,T_
    beta = [2,9,18,5,17]
    low = SMM(vcat(beta/10..., data[2]..., data[3]..., data[4]...,data[5]...),true)
    npzwrite(joinpath(folder, "M_ij_low_trade_cost.npy"), low)
    current = SMM(vcat([1,1,1,1,1]..., data[2]..., data[3]..., data[4]...,data[5]...),true)
    npzwrite(joinpath(folder, "M_ij_trade_cost.npy"), current)
    high = SMM(vcat([2,100,100,100,100]..., data[2]..., data[3]..., data[4]...,data[5]...),true)
    npzwrite(joinpath(folder, "M_ij_high_trade_cost.npy"), high)


    agg_labor_share_emp = agg_labor_share
    agg_labor_share_sim = results[best_index][2][1][1]
    agg_labor_share_ = [agg_labor_share_emp,agg_labor_share_sim]

    agg_industry_share_ = [agg_industry_share,add_first_element(results[best_index][2][2])]

    gamma_emp_result = matrix_report(emp_gamma_ls)
    gamma_sim_result = matrix_report(results[best_index][2][3])
    gamma_ls_ = [gamma_emp_result,gamma_sim_result]

    reg_emp = reg_coef
    reg_sim = results[best_index][2][4]
    reg_ = [reg_emp,reg_sim]

    pi_r_emp_result = matrix_report(emp_pi,false)
    pi_r_sim_result = matrix_report(results[best_index][2][5])
    pi_r = [pi_r_emp_result,pi_r_sim_result]


    best_score = results[best_index][1][1]

    generate_dashboard_report(n,agg_labor_share_,agg_industry_share_,gamma_ls_,reg_,pi_r,best_score,folder*"/report.txt",variable,alpha)
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