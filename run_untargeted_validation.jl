##### Integration: Run untargeted moment validation from main_pso.jl #####
#
# Usage after calibration:
#   include("run_untargeted_validation.jl")
#   results = validate_table2(best_params, industry)
#
# This file provides the interface between main_pso.jl and untargeted_moments.jl
#
# NEW SIMULATION SETUP:
#   - Regional downstream productivity shocks z_{dr,t} ~ AR(1)
#   - Supplier sales: d ln x_{i,t} = a_{di}^D × Σ_r a_{rdi}^D × d ln x_{dr,t} + (1-a_{di}^D) × d ln x_{oi,t}
#   - a_{di}^D drawn from empirical distribution (CSV: share_dist.csv with A129, PartCa)
#   - a_{rdi}^D computed from calibrated model
#   - σ_{sr} loaded as global constant from main_pso.jl

include("untargeted_moments.jl")

"""
    validate_table2(params, industry; sigma_d=0.17, T_periods=36, rho_d=0.92)

Validate calibrated model by reproducing Table 2 regression.

The network structure is solved ONCE using solve_network() from model_CP.jl,
then shocks propagate through the fixed linkages to generate time series.

# NEW MODEL:
  d ln x_{i,t} = a_{di}^D × Σ_r a_{rdi}^D × d ln x_{dr,t} + (1-a_{di}^D) × d ln x_{oi,t}

Where:
  - a_{rdi}^D = share of supplier i's sales to downstream r in total sales to downstream
                (computed from calibrated model)
  - a_{di}^D  = share of supplier i's total sales to downstream industry
                (drawn from empirical distribution in share_dist.csv: A129 as string, PartCa)
  - d ln x_{oi,t} = other customer sales growth (i.i.d. with σ_{sr} from global)

# Required:
  - share_dist.csv in baseline_{industry} folder: columns A129 (sector as string), PartCa (exposure share)
  - sigma_sr loaded as global constant in main_pso.jl

# Arguments
- `params`: Calibrated parameter vector (can be Vector or Matrix column)
- `industry`: Industry name (e.g., "aero") for loading distributions
- `sigma_d`: Standard deviation of downstream AR(1) shocks
- `T_periods`: Number of time periods (quarters)
- `rho_d`: AR(1) persistence for downstream shocks

# Returns
Dict with:
- `network`: Baseline network solution with firm-level data
- `panel_df`: Panel DataFrame
- `regression_results`: Regression coefficients
- `config`: Configuration used
- `downstream_shocks`: Generated downstream shock matrix (R × T)
- `other_shocks`: Generated other customer shocks (N_rho × S × R × T)
- `a_d_D`: Drawn exposure to downstream (N_rho × S × R)
- `a_rdi_D`: Computed regional exposure shares (N_rho × S × R × R)
"""
function validate_table2(params, industry::String; 
                         sigma_d=0.05, T_periods=36, rho_d=-0.15)
    
    # Handle matrix input (take first column)
    if ndims(params) > 1
        params = params[:, 1]
    end
    
    # Set folders
    input_folder = "./baseline_" * industry
    output_folder = "./reporting_" * industry
    
    # Empirical coefficient from Table 2, Column (5)
    # Sup_is × a^D_is × d ln x_{s,t}
    if industry == "aero"
        empirical = Dict(
            "gamma_col5" => 0.112   # Aerospace, col (5)
        )
    else
        empirical = nothing
    end
    
    # Run validation using globals from main_pso.jl
    config = SimulationConfig(T_periods, sigma_d, rho_d, 42)
    results = run_untargeted_validation(
        params, 
        config=config, 
        empirical=empirical,
        input_folder=input_folder
    )
    
    # Save results
    if results["regression_results"]["reg1"] !== nothing
        # Save all regression betas
        betas = Float64[]
        for key in ["reg1", "reg2", "reg3"]
            if results["regression_results"][key] !== nothing
                push!(betas, results["regression_results"][key]["beta"])
            end
        end
        NPZ.npzwrite(joinpath(output_folder, "untargeted_betas.npy"), betas)
        println("\nResults saved to: $output_folder/untargeted_betas.npy")
        
        # Also save a summary
        open(joinpath(output_folder, "untargeted_summary.txt"), "w") do f
            println(f, "="^60)
            println(f, "UNTARGETED VALIDATION RESULTS")
            println(f, "="^60)
            println(f, "\nIndustry: $industry")
            println(f, "T periods: $T_periods")
            println(f, "σ_d (downstream): $sigma_d")
            println(f, "ρ_d (AR1 persistence): $rho_d")
            println(f, "σ_{sr} (other customer): loaded from global")
            println(f, "  Mean σ_{sr}: $(round(mean(sigma_sr), digits=4))")
            println(f, "  Min σ_{sr}: $(round(minimum(sigma_sr), digits=4))")
            println(f, "  Max σ_{sr}: $(round(maximum(sigma_sr), digits=4))")
            
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
        println("Summary saved to: $output_folder/untargeted_summary.txt")
    end
    
    return results
end


"""
    validate_table2_with_custom_exposure(params, industry, exposure_csv_path; kwargs...)

Run validation with a custom exposure distribution CSV file.

The CSV must have columns:
- A129: Sector identifier (string, will be sorted and assigned integer indices)
- PartCa: Exposure share a_{di}^D (float between 0 and 1)

This allows testing with different exposure distributions.
Note: sigma_sr is used from the global constant.
"""
function validate_table2_with_custom_exposure(
    params, 
    industry::String,
    exposure_csv_path::String;
    sigma_d=0.05, T_periods=36, rho_d=-0.15
)
    
    # Handle matrix input (take first column)
    if ndims(params) > 1
        params = params[:, 1]
    end
    
    # Create temporary folder with the exposure file
    temp_folder = mktempdir()
    cp(exposure_csv_path, joinpath(temp_folder, "share_dist.csv"))
    
    output_folder = "./reporting_" * industry
    
    # Empirical coefficients
    empirical = industry == "aero" ? Dict("gamma_col5" => 0.112) : nothing
    
    config = SimulationConfig(T_periods, sigma_d, rho_d, 42)
    results = run_untargeted_validation(
        params, 
        config=config, 
        empirical=empirical,
        input_folder=temp_folder
    )
    
    # Cleanup
    rm(temp_folder, recursive=true)
    
    return results
end


"""
    sensitivity_analysis(params, industry; n_simulations=100)

Run multiple simulations with different random seeds to assess sensitivity.

Returns summary statistics across simulations.
"""
function sensitivity_analysis(params, industry::String; 
                              n_simulations=100,
                              sigma_d=0.05, T_periods=36, rho_d=-0.15)
    
    if ndims(params) > 1
        params = params[:, 1]
    end
    
    input_folder = "./baseline_" * industry
    
    betas = Float64[]
    
    println("\nRunning sensitivity analysis with $n_simulations simulations...")
    
    for sim in 1:n_simulations
        config = SimulationConfig(T_periods, sigma_d, rho_d, sim)  # Different seed each time
        
        try
            results = run_untargeted_validation(
                params, 
                config=config, 
                empirical=nothing,  # Skip comparison printout
                input_folder=input_folder
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
    println("SENSITIVITY ANALYSIS RESULTS")
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
        "q95" => quantile(betas, 0.95)
    )
end