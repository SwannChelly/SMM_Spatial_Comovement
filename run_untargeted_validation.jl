##### Integration: Run untargeted moment validation from main_pso.jl #####
#
# Usage after calibration:
#   include("run_untargeted_validation.jl")
#   results = validate_table2(best_params, industry)
#
# This file provides the interface between main_pso.jl and untargeted_moments.jl
#
# SHOCK MODELS:
#   UNIVARIATE:   Same ρ for all regions, i.i.d. shocks across regions
#   MULTIVARIATE: Region-specific ρ_r, correlated innovations via Σ
#
# Required files for MULTIVARIATE (in baseline_{industry}/):
#   - rho_r.npy: Region-specific persistence (R_downstream,)
#   - Sigma_innovations.npy: Cross-regional covariance matrix (R_downstream, R_downstream)

include("untargeted_moments.jl")


"""
    validate_table2(params, industry; shock_model=UNIVARIATE, kwargs...)

Validate calibrated model by reproducing Table 2 regression.

# Arguments
- `params`: Calibrated parameter vector
- `industry`: Industry name (e.g., "aero")
- `shock_model`: UNIVARIATE or MULTIVARIATE (default: UNIVARIATE)
- `sigma_d`: Std of downstream shocks (univariate only)
- `T_periods`: Number of time periods
- `rho_d`: AR(1) persistence (univariate only)

# Returns
Dict with network, panel_df, regression_results, config, shocks
"""
function validate_table2(
    params, 
    industry::String; 
    shock_model::ShockModel = UNIVARIATE,
    sigma_d = 0.05, 
    T_periods = 36, 
    rho_d = -0.15
)
    
    if ndims(params) > 1
        params = params[:, 1]
    end
    
    input_folder = "./baseline_" * industry
    output_folder = "./reporting_" * industry
    
    # Empirical target
    if industry == "aero"
        empirical = Dict("gamma_col5" => 0.112)
    else
        empirical = nothing
    end
    
    # Create config
    config = SimulationConfig(T_periods, sigma_d, rho_d, 42, shock_model)
    
    # Load multivariate params if needed
    multivar_params = nothing
    if shock_model == MULTIVARIATE
        multivar_params = load_multivariate_params(input_folder)
    end
    
    # Run validation
    results = run_untargeted_validation(
        params, 
        config = config, 
        empirical = empirical,
        input_folder = input_folder,
        multivar_params = multivar_params
    )
    
    # Save results
    model_suffix = shock_model == UNIVARIATE ? "" : "_multivar"
    
    if results["regression_results"]["reg1"] !== nothing
        betas = Float64[]
        for key in ["reg1", "reg2", "reg3"]
            if results["regression_results"][key] !== nothing
                push!(betas, results["regression_results"][key]["beta"])
            end
        end
        NPZ.npzwrite(joinpath(output_folder, "untargeted_betas$(model_suffix).npy"), betas)
        println("\nResults saved to: $output_folder/untargeted_betas$(model_suffix).npy")
        
        # Summary
        model_name = shock_model == UNIVARIATE ? "UNIVARIATE" : "MULTIVARIATE"
        open(joinpath(output_folder, "untargeted_summary$(model_suffix).txt"), "w") do f
            println(f, "="^60)
            println(f, "UNTARGETED VALIDATION RESULTS - $model_name")
            println(f, "="^60)
            println(f, "\nIndustry: $industry")
            println(f, "Shock model: $model_name")
            println(f, "T periods: $T_periods")
            
            if shock_model == UNIVARIATE
                println(f, "σ_d (downstream): $sigma_d")
                println(f, "ρ_d (AR1 persistence): $rho_d")
            else
                println(f, "ρ_r (region-specific persistence):")
                println(f, "  Mean: $(round(mean(multivar_params.rho_r), digits=4))")
                println(f, "  Std:  $(round(std(multivar_params.rho_r), digits=4))")
                println(f, "  Range: [$(round(minimum(multivar_params.rho_r), digits=4)), $(round(maximum(multivar_params.rho_r), digits=4))]")
                println(f, "Σ (innovation covariance):")
                println(f, "  Diagonal mean: $(round(mean(diag(multivar_params.Sigma_innovation)), digits=6))")
                off_diag = multivar_params.Sigma_innovation[.!I(size(multivar_params.Sigma_innovation,1))]
                println(f, "  Off-diagonal mean: $(round(mean(off_diag), digits=6))")
            end
            
            println(f, "σ_{sr} (other customer): loaded from global")
            println(f, "  Mean σ_{sr}: $(round(mean(sigma_sr), digits=4))")
            
            # Reg 1
            println(f, "\n" * "-"^50)
            println(f, "Reg 1: d ln x ~ downstream_growth + fe(firm)")
            r = results["regression_results"]["reg1"]
            println(f, "  β = $(round(r["beta"], digits=4))")
            println(f, "  SE = $(round(r["beta_se"], digits=4))")
            println(f, "  95% CI = [$(round(r["ci_lower"], digits=4)), $(round(r["ci_upper"], digits=4))]")
            println(f, "  R² = $(round(r["R2"], digits=4))")
            println(f, "  N = $(r["N"])")
            
            # Reg 2
            if results["regression_results"]["reg2"] !== nothing
                println(f, "\n" * "-"^50)
                println(f, "Reg 2: d ln x ~ weighted_exposure + fe(firm)")
                r = results["regression_results"]["reg2"]
                println(f, "  β = $(round(r["beta"], digits=4))")
                println(f, "  SE = $(round(r["beta_se"], digits=4))")
                println(f, "  95% CI = [$(round(r["ci_lower"], digits=4)), $(round(r["ci_upper"], digits=4))]")
                println(f, "  R² = $(round(r["R2"], digits=4))")
                println(f, "  N = $(r["N"])")
            end
            
            # Reg 3
            if results["regression_results"]["reg3"] !== nothing
                println(f, "\n" * "-"^50)
                println(f, "Reg 3: d ln x (no other customer) ~ downstream_growth + fe(firm)")
                r = results["regression_results"]["reg3"]
                println(f, "  β = $(round(r["beta"], digits=4))")
                println(f, "  SE = $(round(r["beta_se"], digits=4))")
                println(f, "  95% CI = [$(round(r["ci_lower"], digits=4)), $(round(r["ci_upper"], digits=4))]")
                println(f, "  R² = $(round(r["R2"], digits=4))")
                println(f, "  N = $(r["N"])")
            end
            
            if empirical !== nothing && haskey(empirical, "gamma_col5")
                println(f, "\n" * "="^50)
                println(f, "Empirical target: $(empirical["gamma_col5"])")
                rel_diff = abs(results["regression_results"]["reg1"]["beta"] - empirical["gamma_col5"]) / 
                           abs(empirical["gamma_col5"])
                println(f, "Relative difference (Reg 1): $(round(100*rel_diff, digits=1))%")
            end
        end
        println("Summary saved to: $output_folder/untargeted_summary$(model_suffix).txt")
    end
    
    return results
end


"""
    validate_table2_both_models(params, industry; T_periods=36, sigma_d=0.05, rho_d=-0.15)

Run validation with BOTH shock models and return both results.

This is the main function to call from main_pso.jl for comprehensive validation.

# Returns
Dict with:
- `univariate`: Results from UNIVARIATE model
- `multivariate`: Results from MULTIVARIATE model (or nothing if params not found)
"""
function validate_table2_both_models(
    params, 
    industry::String;
    T_periods = 36,
    sigma_d = 0.05,
    rho_d = -0.15
)
    println("\n" * "="^70)
    println("RUNNING BOTH SHOCK MODELS")
    println("="^70)
    
    # Run univariate
    println("\n>>> UNIVARIATE MODEL <<<")
    results_univariate = validate_table2(
        params, industry;
        shock_model = UNIVARIATE,
        T_periods = T_periods,
        sigma_d = sigma_d,
        rho_d = rho_d
    )
    
    # Try multivariate
    results_multivariate = nothing
    input_folder = "./baseline_" * industry
    
    if isfile(joinpath(input_folder, "rho_r.npy")) && isfile(joinpath(input_folder, "Sigma_innovations.npy"))
        println("\n>>> MULTIVARIATE MODEL <<<")
        results_multivariate = validate_table2(
            params, industry;
            shock_model = MULTIVARIATE,
            T_periods = T_periods
        )
    else
        println("\n>>> MULTIVARIATE MODEL: Skipped (parameters not found) <<<")
        println("  Missing: rho_r.npy and/or Sigma_innovations.npy in $input_folder")
        println("  Run extract_shock_parameters.py to generate these files")
    end
    
    # Print comparison
    println("\n" * "="^70)
    println("COMPARISON OF SHOCK MODELS")
    println("="^70)
    
    println("\nRegression 1 (β on downstream_growth):")
    if results_univariate["regression_results"]["reg1"] !== nothing
        beta_uni = results_univariate["regression_results"]["reg1"]["beta"]
        se_uni = results_univariate["regression_results"]["reg1"]["beta_se"]
        println("  UNIVARIATE:   β = $(round(beta_uni, digits=4)) (se: $(round(se_uni, digits=4)))")
    end
    if results_multivariate !== nothing && results_multivariate["regression_results"]["reg1"] !== nothing
        beta_multi = results_multivariate["regression_results"]["reg1"]["beta"]
        se_multi = results_multivariate["regression_results"]["reg1"]["beta_se"]
        println("  MULTIVARIATE: β = $(round(beta_multi, digits=4)) (se: $(round(se_multi, digits=4)))")
    end
    
    return Dict(
        "univariate" => results_univariate,
        "multivariate" => results_multivariate
    )
end


"""
    validate_table2_all_models(params, industry; T_periods=36, time_fe_mode="resample")

Run validation with ALL THREE shock models using model-specific parameters.

This is the comprehensive validation function that:
- Runs UNIVARIATE, MULTIVARIATE, and MULTIVARIATE_FE
- Uses each model's appropriate parameters
- Saves unified panel CSV with all three models
- Saves summary text file

# Arguments
- `params`: Calibrated parameter vector
- `industry`: Industry name (e.g., "aero", "auto_23")
- `T_periods`: Number of time periods (default: 36)
- `time_fe_mode`: For MULTIVARIATE_FE only - "resample", "new", or "empirical" (default: "resample")

# Returns
Dict with:
- `network`: Solved network
- `panel_df`: DataFrame with all three models' results
- `regression_results`: Dict with results for each model
- Other diagnostic info
"""
function validate_table2_all_models(
    params, 
    industry::String;
    T_periods = 36,
    time_fe_mode = "resample"
)
    if ndims(params) > 1
        params = params[:, 1]
    end
    
    input_folder = "./baseline_" * industry
    output_folder = "./reporting_" * industry
    
    # Check if all required parameter files exist
    has_univariate = isfile(joinpath(input_folder, "rho_univariate.npy"))
    has_multivariate = isfile(joinpath(input_folder, "rho_r_multivariate.npy")) && 
                       isfile(joinpath(input_folder, "Sigma_innovations_multivariate.npy"))
    has_multivariate_fe = isfile(joinpath(input_folder, "rho_r_fe.npy")) && 
                          isfile(joinpath(input_folder, "Sigma_innovations_fe.npy"))
    
    if !has_univariate || !has_multivariate || !has_multivariate_fe
        error("Missing required parameter files. Need:\n" *
              "  - UNIVARIATE: rho_univariate.npy, sigma_unconditional_univariate.npy\n" *
              "  - MULTIVARIATE: rho_r_multivariate.npy, Sigma_innovations_multivariate.npy\n" *
              "  - MULTIVARIATE_FE: rho_r_fe.npy, Sigma_innovations_fe.npy, time_fe.npy\n" *
              "Run extract_shock_parameters.py to generate these files")
    end
    
    # Empirical target
    empirical = nothing
    if industry == "aero"
        empirical = Dict("gamma_col5" => 0.112)
    end
    
    # Create config - shock_model doesn't matter since we'll call run_untargeted_validation directly
    config = SimulationConfig(T_periods, 0.05, -0.15, 42, UNIVARIATE, time_fe_mode)
    
    println("\n" * "="^70)
    println("RUNNING ALL THREE SHOCK MODELS")
    println("="^70)
    println("Industry: $industry")
    println("T_periods: $T_periods")
    println("Time FE mode: $time_fe_mode")
    
    # Run unified validation (handles all three models internally)
    results = run_untargeted_validation(
        params, 
        config = config, 
        empirical = empirical,
        input_folder = input_folder
    )
    
    # Save unified panel
    panel_file = joinpath(output_folder, "simulated_panel_unified.csv")
    CSV.write(panel_file, results["panel_df"])
    println("\nUnified panel saved to: $panel_file")

    # Save regional sales
    regional_sales_file = joinpath(output_folder, "regional_sales_unified.csv")
    CSV.write(regional_sales_file, results["regional_sales_df"])
    println("Regional sales saved to: $regional_sales_file")
    
    # Save summary
    summary_file = joinpath(output_folder, "untargeted_summary_unified.txt")
    open(summary_file, "w") do f
        println(f, "="^70)
        println(f, "UNTARGETED VALIDATION - ALL THREE MODELS")
        println(f, "="^70)
        println(f, "\nIndustry: $industry")
        println(f, "T periods: $T_periods")
        println(f, "Time FE mode: $time_fe_mode")
        println(f, "Observations: $(nrow(results["panel_df"]))")
        println(f, "Suppliers: $(sum(results["network"].linkages .> 0))")
        
        # Parameter info for each model
        println(f, "\n" * "="^70)
        println(f, "MODEL PARAMETERS")
        println(f, "="^70)
        
        # UNIVARIATE
        println(f, "\nUNIVARIATE (no FE):")
        uni_p = results["params"]["univariate"]
        if uni_p !== nothing
            println(f, "  ρ: $(round(uni_p.rho, digits=4))")
            println(f, "  σ_innovation: $(round(uni_p.sigma_innovation, digits=6))")
            println(f, "  σ_unconditional: $(round(uni_p.sigma_unconditional, digits=6))")
        end
        
        # MULTIVARIATE
        println(f, "\nMULTIVARIATE (no FE):")
        multi_p = results["params"]["multivariate"]
        println(f, "  ρ_r: mean=$(round(mean(multi_p.rho_r), digits=4)), range=[$(round(minimum(multi_p.rho_r), digits=4)), $(round(maximum(multi_p.rho_r), digits=4))]")
        println(f, "  Σ diagonal: mean=$(round(mean(diag(multi_p.Sigma_innovation)), digits=6))")
        off_diag = multi_p.Sigma_innovation[.!I(size(multi_p.Sigma_innovation,1))]
        println(f, "  Σ off-diagonal: mean=$(round(mean(off_diag), digits=6))")
        
        # MULTIVARIATE_FE
        println(f, "\nMULTIVARIATE_FE:")
        multi_fe_p = results["params"]["multivariate_fe"]
        println(f, "  ρ_r: mean=$(round(mean(multi_fe_p.rho_r), digits=4)), range=[$(round(minimum(multi_fe_p.rho_r), digits=4)), $(round(maximum(multi_fe_p.rho_r), digits=4))]")
        println(f, "  Σ_clean diagonal: mean=$(round(mean(diag(multi_fe_p.Sigma_innovation)), digits=6))")
        off_diag_fe = multi_fe_p.Sigma_innovation[.!I(size(multi_fe_p.Sigma_innovation,1))]
        println(f, "  Σ_clean off-diagonal: mean=$(round(mean(off_diag_fe), digits=6))")
        if multi_fe_p.time_fe !== nothing
            println(f, "  Time FE: mean=$(round(mean(multi_fe_p.time_fe), digits=4)), std=$(round(std(multi_fe_p.time_fe), digits=4))")
        end
        
        # Regression results
        println(f, "\n" * "="^70)
        println(f, "REGRESSION RESULTS")
        println(f, "="^70)
        
        for (reg_name, reg_key) in [("Regression 1: d ln x ~ downstream_growth + fe(firm)", "reg1"),
                                     ("Regression 2: d ln x ~ a_d_D * weighted_exposure + fe(firm)", "reg2"),
                                     ("Regression 3: d ln x (no other) ~ downstream_growth + fe(firm)", "reg3")]
            println(f, "\n" * "-"^60)
            println(f, reg_name)
            println(f, "-"^60)
            
            for model in ["UNIVARIATE", "MULTIVARIATE", "MULTIVARIATE_FE"]
                if results["regression_results"][model][reg_key] !== nothing
                    r = results["regression_results"][model][reg_key]
                    println(f, "\n$model:")
                    println(f, "  β = $(round(r["beta"], digits=4))")
                    println(f, "  SE = $(round(r["beta_se"], digits=4))")
                    println(f, "  95% CI = [$(round(r["ci_lower"], digits=4)), $(round(r["ci_upper"], digits=4))]")
                    println(f, "  R² = $(round(r["R2"], digits=4))")
                    println(f, "  N = $(r["N"])")
                end
            end
        end
        
        # Empirical comparison
        if empirical !== nothing && haskey(empirical, "gamma_col5")
            println(f, "\n" * "="^70)
            println(f, "COMPARISON TO EMPIRICAL TARGET")
            println(f, "="^70)
            println(f, "\nEmpirical target (Reg 1): $(empirical["gamma_col5"])")
            
            for model in ["UNIVARIATE", "MULTIVARIATE", "MULTIVARIATE_FE"]
                if results["regression_results"][model]["reg1"] !== nothing
                    beta = results["regression_results"][model]["reg1"]["beta"]
                    rel_diff = abs(beta - empirical["gamma_col5"]) / abs(empirical["gamma_col5"])
                    println(f, "  $model: β = $(round(beta, digits=4)), diff = $(round(100*rel_diff, digits=1))%")
                end
            end
        end
        
        println(f, "\n" * "="^70)
        println(f, "NOTES")
        println(f, "="^70)
        println(f, "Each model uses its own parameters estimated appropriately:")
        println(f, "  - UNIVARIATE: ρ, σ from ORIGINAL data (no FE)")
        println(f, "  - MULTIVARIATE: ρ_r, Σ from ORIGINAL data (includes aggregate)")
        println(f, "  - MULTIVARIATE_FE: ρ_r, Σ_clean from DEMEANED data + time FE μ_t")
    end
    
    println("Summary saved to: $summary_file")
    
    return results
end


"""
    sensitivity_analysis(params, industry; n_simulations=100, shock_model=UNIVARIATE)

Run multiple simulations with different random seeds to assess sensitivity.
"""
function sensitivity_analysis(
    params, 
    industry::String; 
    n_simulations = 100,
    shock_model::ShockModel = UNIVARIATE,
    sigma_d = 0.05, 
    T_periods = 36, 
    rho_d = -0.15
)
    if ndims(params) > 1
        params = params[:, 1]
    end
    
    input_folder = "./baseline_" * industry
    
    # Load multivariate params if needed
    multivar_params = nothing
    if shock_model == MULTIVARIATE
        multivar_params = load_multivariate_params(input_folder)
    end
    
    betas = Float64[]
    model_name = shock_model == UNIVARIATE ? "UNIVARIATE" : "MULTIVARIATE"
    
    println("\nRunning sensitivity analysis ($model_name) with $n_simulations simulations...")
    
    for sim in 1:n_simulations
        config = SimulationConfig(T_periods, sigma_d, rho_d, sim, shock_model)
        
        try
            results = run_untargeted_validation(
                params, 
                config = config, 
                empirical = nothing,
                input_folder = input_folder,
                multivar_params = multivar_params
            )
            
            if results["regression_results"]["reg1"] !== nothing
                push!(betas, results["regression_results"]["reg1"]["beta"])
            end
        catch e
            @warn "Simulation $sim failed: $e"
        end
        
        if sim % 10 == 0
            println("  Completed $sim/$n_simulations simulations")
        end
    end
    
    println("\n" * "="^60)
    println("SENSITIVITY ANALYSIS RESULTS ($model_name)")
    println("="^60)
    println("Successful simulations: $(length(betas))/$n_simulations")
    println("β mean: $(round(mean(betas), digits=4))")
    println("β std:  $(round(std(betas), digits=4))")
    println("β min:  $(round(minimum(betas), digits=4))")
    println("β max:  $(round(maximum(betas), digits=4))")
    println("β 5th percentile:  $(round(quantile(betas, 0.05), digits=4))")
    println("β 95th percentile: $(round(quantile(betas, 0.95), digits=4))")
    
    return Dict(
        "betas" => betas,
        "mean" => mean(betas),
        "std" => std(betas),
        "min" => minimum(betas),
        "max" => maximum(betas),
        "q05" => quantile(betas, 0.05),
        "q95" => quantile(betas, 0.95),
        "shock_model" => model_name
    )
end