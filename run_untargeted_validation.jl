##### Integration: Run untargeted moment validation from main_pso.jl #####
#
# Usage after calibration:
#   include("run_untargeted_validation.jl")
#   results = validate_table2(best_params, industry)
#
# NOTE: This file expects the following global constants to be defined via @everywhere in main_pso.jl:
#   S, R, N_downstream_per_region, regional_wages, w_rs, distances, DistBin,
#   N_rho, theta, lambda, nu, nu_s, epsilon

include("untargeted_moments.jl")

"""
    validate_table2(params, industry; sigma_s=0.05, T_periods=36, rho_s=0.92)

Validate calibrated model by reproducing Table 2 regression.

The network structure is solved ONCE, then demand shocks propagate through 
the fixed linkages to generate time series variation.

Uses global constants defined in main_pso.jl (S, R, N_downstream_per_region, etc.)
"""
function validate_table2(params, industry::String; sigma_s=0.17, T_periods=36, rho_s=0.92)
    
    output_folder = "./reporting_" * industry
    
    # Empirical coefficient from Table 2, Column (5)
    # Sup × a^D × d ln x_{s,t}
    # Since all suppliers in simulation have a^D = 1, we compare β to this coefficient
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
    
    # Run validation (uses global constants from main_pso.jl)
    config = SimulationConfig(T_periods, sigma_s, rho_s, 42)
    results = run_untargeted_validation(params, config=config, empirical=empirical)
    
    # Save
    if results["regression_results"]["firm_fe"] !== nothing
        NPZ.npzwrite(joinpath(output_folder, "untargeted_beta.npy"), 
                     [results["regression_results"]["firm_fe"]["beta"]])
        println("\nResults saved to: $output_folder/untargeted_beta.npy")
    end
    
    return results
end