##### Integration: Run untargeted moment validation from main_pso.jl #####
#
# Usage after calibration:
#   include("run_untargeted_validation.jl")
#   results = validate_table2(best_params, industry)

include("untargeted_moments.jl")

"""
    validate_table2(params, industry; σ_shock=0.05, T_periods=120)

Validate calibrated model by reproducing Table 2 regression.

The network structure is solved ONCE, then demand shocks propagate through 
the fixed linkages to generate time series variation.
"""
function validate_table2(params, industry::String; σ_shock=0.17, T_periods=36, ρ_ar=0.92)
    
    input_folder = "./baseline_" * industry
    output_folder = "./reporting_" * industry
    
    # Load data
    coefs = CSV.read(joinpath(input_folder, "stats.csv"), DataFrame)
    distances_local = NPZ.npzread(joinpath(input_folder, "distances.npy"))
    w_rs_local = NPZ.npzread(joinpath(input_folder, "w_rs.npy"))
    regional_wages_local = NPZ.npzread(joinpath(input_folder, "regional_wages.npy"))
    N_downstream_per_region_local = NPZ.npzread(joinpath(input_folder, "N_downstream_per_region.npy"))
    filter_N_upstream_local = NPZ.npzread(joinpath(input_folder, "filter_N_upstream.npy"))
    
    S_local, R_local = size(filter_N_upstream_local)
    
    # Build DistBin
    DistBin_local = zeros(Int, R_local, R_local)
    for i in 1:R_local, j in 1:R_local
        d = distances_local[i, j]
        DistBin_local[i, j] = d <= 20 ? 0 : d <= 50 ? 1 : d <= 100 ? 2 : d <= 150 ? 3 : d <= 200 ? 4 : 5
    end
    
    model_globals = (
        N_downstream_per_region = N_downstream_per_region_local,
        regional_wages = regional_wages_local,
        w_rs = w_rs_local,
        distances = distances_local,
        DistBin = DistBin_local,
        N_rho = 50,
        theta = 1.768,
        lambda = 0.5,
        nu = 0.2,
        nu_s = ones(S_local) .* 2.5,
        epsilon = coefs[1, "value"],
        S = S_local,
        R = R_local
    )
    
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
    
    # Run validation
    config  = SimulationConfig(T_periods, σ_shock, ρ_ar, 42)
    results = run_untargeted_validation(params, model_globals, config=config, empirical=empirical)
    
    # Save
    if results["regression_results"]["firm_fe"] !== nothing
        NPZ.npzwrite(joinpath(output_folder, "untargeted_beta.npy"), 
                     [results["regression_results"]["firm_fe"]["beta"]])
        println("\nResults saved to: $output_folder/untargeted_beta.npy")
    end
    
    return results
end