##### Integration: Run untargeted moment validation from main_pso.jl #####
#
# Usage after calibration:
#   include("run_untargeted_validation.jl")
#   results = validate_table2(best_params, industry)
#
# This file provides the interface between main_pso.jl and untargeted_moments.jl

include("untargeted_moments.jl")

"""
    validate_table2(params, industry; sigma_s=0.17, T_periods=36, rho_s=0.92)

Validate calibrated model by reproducing Table 2 regression.

The network structure is solved ONCE using solve_network() from model_CP.jl,
then demand shocks propagate through the fixed linkages to generate time series.

# Arguments
- `params`: Calibrated parameter vector (can be Vector or Matrix column)
- `industry`: "aero" or "car" (for empirical comparison)
- `sigma_s`: Standard deviation of log shocks
- `T_periods`: Number of time periods
- `rho_s`: AR(1) persistence

# Returns
Dict with:
- `network`: Baseline network solution with firm-level data
- `panel_df`: Panel DataFrame
- `regression_results`: Regression coefficients
- `config`: Configuration used
- `shocks`: Generated shock matrix
"""
function validate_table2(params, industry::String; sigma_s=0.17, T_periods=36, rho_s=0.92)
    
    # Handle matrix input (take first column)
    if ndims(params) > 1
        params = params[:, 1]
    end
    
    # Set output folder
    output_folder = "./reporting_" * industry
    
    # Empirical coefficient from Table 2, Column (5)
    # Sup_is × a^D_is × d ln x_{s,t}
    if industry == "aero"
        empirical = Dict(
            "gamma_col5" => 0.112   # Aerospace, col (5)
        )
    elseif industry == "car"
        empirical = Dict(
            "gamma_col5" => 0.161   # Car, col (1) - using Sup × d ln x as reference
        )
    else
        empirical = nothing
    end
    
    # Run validation using globals from main_pso.jl
    config = SimulationConfig(T_periods, sigma_s, rho_s, 42)
    results = run_untargeted_validation(params, config=config, empirical=empirical)
    
    # Save results
    if results["regression_results"]["firm_fe"] !== nothing
        NPZ.npzwrite(joinpath(output_folder, "untargeted_beta.npy"), 
                     [results["regression_results"]["firm_fe"]["beta"]])
        println("\nResults saved to: $output_folder/untargeted_beta.npy")
    end
    
    return results
end


"""
    validate_table2_standalone(params, industry; kwargs...)

Standalone version that loads data from disk (doesn't require main_pso.jl globals).

Use this when running outside the main_pso.jl context.
"""
# function validate_table2_standalone(params, industry::String; sigma_s=0.17, T_periods=36, rho_s=0.92)
    
#     input_folder = "./baseline_" * industry
#     output_folder = "./reporting_" * industry
    
#     # Load data
#     coefs = CSV.read(joinpath(input_folder, "stats.csv"), DataFrame)
#     distances_local = NPZ.npzread(joinpath(input_folder, "distances.npy"))
#     w_rs_local = NPZ.npzread(joinpath(input_folder, "w_rs.npy"))
#     regional_wages_local = NPZ.npzread(joinpath(input_folder, "regional_wages.npy"))
#     N_downstream_per_region_local = NPZ.npzread(joinpath(input_folder, "N_downstream_per_region.npy"))
#     filter_N_upstream_local = NPZ.npzread(joinpath(input_folder, "filter_N_upstream.npy"))
    
#     S_local, R_local = size(filter_N_upstream_local)
    
#     # Build DistBin
#     DistBin_local = zeros(Int, R_local, R_local)
#     for i in 1:R_local, j in 1:R_local
#         d = distances_local[i, j]
#         DistBin_local[i, j] = d <= 20 ? 0 : d <= 50 ? 1 : d <= 100 ? 2 : d <= 150 ? 3 : d <= 200 ? 4 : 5
#     end
    
#     # Define globals (normally done by main_pso.jl)
#     global S = S_local
#     global R = R_local
#     global R_downstream = sum(N_downstream_per_region_local .!= 0)
#     global distances = distances_local
#     global w_rs = w_rs_local
#     global regional_wages = regional_wages_local
#     global N_downstream_per_region = N_downstream_per_region_local
#     global DistBin = DistBin_local
#     global N_rho = 50
#     global theta = 1.768
#     global lambda = 0.5
#     global nu = 0.2
#     global nu_s = ones(S_local) .* 2.5
#     global epsilon = coefs[1, "value"]
#     global delta_r = ones(R_local)
    
#     # Now call the standard function
#     return validate_table2(params, industry; sigma_s=sigma_s, T_periods=T_periods, rho_s=rho_s)
# end