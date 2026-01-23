##### Run Full Amplification Analysis (Section 4.3) #####
# Author: Swann
# 
# This script runs the complete amplification analysis pipeline:
#   1. Load calibrated parameters from main_pso.jl output
#   2. Compute network-derived weights (Julia)
#   3. Compute amplification coefficients and create maps (Python)
#
# Usage:
#   julia run_amplification_analysis.jl [industry]
#
# where industry is "aero" or "car"

using Distributed
using Dates
using NPZ

# Add workers for parallel computation
available = Sys.CPU_THREADS - nprocs()
println("Using $(max(available-1, 1)) workers")
addprocs(max(available-1, 0))

# Include required files
@everywhere include("model_CP.jl")
@everywhere include("tools.jl")
include("compute_amplification_weights.jl")

############## Configuration #################

# Parse command line arguments
industry = length(ARGS) >= 1 ? ARGS[1] : "aero"
println("\n" * "="^70)
println("AMPLIFICATION ANALYSIS FOR $(uppercase(industry)) INDUSTRY")
println("="^70)

# Set folders
input_folder = "./baseline_" * industry
params_folder = "./reporting_" * industry  # Where calibrated params are stored
output_folder = "./amplification_" * industry

############## Load Global Constants #################

println("\n[Setup] Loading global constants from $input_folder...")

coefs = CSV.read(joinpath(input_folder, "stats.csv"), DataFrame)
distances_local = NPZ.npzread(joinpath(input_folder, "distances.npy"))
w_rs_local = NPZ.npzread(joinpath(input_folder, "w_rs.npy"))
regional_wages_local = NPZ.npzread(joinpath(input_folder, "regional_wages.npy"))
N_downstream_per_region_local = NPZ.npzread(joinpath(input_folder, "N_downstream_per_region.npy"))
filter_N_upstream_local = NPZ.npzread(joinpath(input_folder, "filter_N_upstream.npy"))
agg_industry_share_local = NPZ.npzread(joinpath(input_folder, "input_share.npy"))
domestic_share_local = NPZ.npzread(joinpath(input_folder, "domestic_share.npy"))
N_rs_local = NPZ.npzread(joinpath(input_folder, "N_rs.npy"))

S_, R_ = size(filter_N_upstream_local)

# Define constants on all workers
@everywhere const S = $(S_)
@everywhere const R = $(R_)

# Distance bins
DistBin_local = Array{Int}(undef, R_, R_)
for i in 1:R_, j in 1:R_
    DistBin_local[i, j] = distance_bin(distances_local[i, j])
end

R_downstream_ = size(N_downstream_per_region_local[N_downstream_per_region_local .!= 0])[1]
@everywhere const R_downstream = $(R_downstream_)
@everywhere const agg_industry_share = $(agg_industry_share_local)
@everywhere const agg_labor_share = $(coefs[2, "value"])
@everywhere const domestic_share = $(domestic_share_local)
@everywhere regional_wages = $(regional_wages_local)
@everywhere const distances = $(distances_local)
@everywhere const N_downstream_per_region = $(N_downstream_per_region_local)
@everywhere const w_rs = $(w_rs_local)
@everywhere const DistBin = $(DistBin_local)
@everywhere const filter_N_upstream = $(filter_N_upstream_local)
@everywhere const N_rho = $(50)
@everywhere const epsilon = $(coefs[1, "value"])
@everywhere const lambda = $(0.5)
@everywhere const nu = $(0.2)
@everywhere const nu_s = $(ones(S_) .* 2.5)
@everywhere const theta = $(1.768)
@everywhere const delta_r = $(ones(R_))
@everywhere const Weight_matrix = $(nothing)
@everywhere const N_rs = $(N_rs_local)

# Empirical moments (for completeness)
@everywhere const emp_gamma_ls = $(NPZ.npzread(joinpath(input_folder, "emp_gamma_ls.npy"))')
@everywhere const emp_pi_r = $(NPZ.npzread(joinpath(input_folder, "emp_pi_r.npy"))[2:end])
@everywhere const reg_coef = $(NPZ.npzread(joinpath(input_folder, "reg_coef.npy")))
@everywhere const mask_emp_gamma_ls = $(NPZ.npzread(joinpath(input_folder, "mask_gamma_ls.npy"))')

empirical_moments_local = [[agg_labor_share], agg_industry_share[2:end], emp_gamma_ls, reg_coef, emp_pi_r]
empirical_moments_local = vcat([vec(empirical_moments_local[i]) for i in 1:length(empirical_moments_local)]...)
empirical_moments_local = reshape(empirical_moments_local, 1, length(empirical_moments_local))
@everywhere const empirical_moments = $(empirical_moments_local)
@everywhere const empirical_moments_reduced = $(reshape(emp_gamma_ls[mask_emp_gamma_ls .!= 0], (1, size(emp_gamma_ls[mask_emp_gamma_ls .!= 0])[1])))
@everywhere const K_max = $(50)

println("  S = $S_ sectors")
println("  R = $R_ regions")
println("  R_downstream = $R_downstream_ active downstream regions")

############## Load Calibrated Parameters #################

println("\n[Step 1] Loading calibrated parameters...")

# Try to find best parameters
params_file = nothing
search_paths = [
    joinpath(params_folder, "best_params.npy"),
    joinpath(params_folder, "0", "best_params.npy"),
]

# Add epoch folders
for epoch in 1:100
    for stage in 1:100
        push!(search_paths, joinpath(params_folder, "epoch_$epoch", "$stage", "best_params.npy"))
    end
end

for path in search_paths
    if isfile(path)
        params_file = path
        break
    end
end

if params_file === nothing
    error("Could not find calibrated parameters in $params_folder")
end

println("  Found parameters: $params_file")
params = NPZ.npzread(params_file)

# Handle matrix vs vector
if ndims(params) > 1
    params = params[:, 1]
end

println("  Parameter vector length: $(length(params))")

############## Compute Amplification Weights #################

println("\n[Step 2] Computing amplification weights...")

results = compute_amplification_weights(params, industry, output_folder=output_folder)

############## Also Save w_s_r for Python #################

println("\n[Step 3] Saving sectoral shares w_s^{r'} for Python...")

# w_rs is [R, S] - share of sector s in region r's *employment*
# We need to transpose to [S, R] for Python
w_s_r = w_rs'  # Now [S, R]

NPZ.npzwrite(joinpath(output_folder, "w_s_r.npy"), w_s_r)
println("  Saved w_s_r.npy")

############## Call Python Script #################

println("\n[Step 4] Running Python analysis...")

# Determine downstream sector index
# In the French classification, Car = C29A, Aerospace = C30C
# The exact index depends on your sector ordering
downstream_idx = 0  # Update based on your sector list

python_cmd = `python3 compute_amplification_map.py --industry $industry --amp-folder $output_folder --data-folder $output_folder --downstream-sector $downstream_idx --output $output_folder`

println("  Running: $python_cmd")
try
    run(python_cmd)
catch e
    println("\n  Warning: Python script failed or not available")
    println("  Error: $e")
    println("\n  To run manually:")
    println("    python3 compute_amplification_map.py --industry $industry --amp-folder $output_folder")
end

############## Summary #################

println("\n" * "="^70)
println("AMPLIFICATION ANALYSIS COMPLETE")
println("="^70)
println("\nOutputs saved to: $output_folder")
println("\nFiles created:")
println("  - w_d_sr.npy       : Share of (s,r') sales to downstream [S×R]")
println("  - w_r_srd.npy      : Share of downstream sales to local [S×R]")
println("  - w_s_r.npy        : Sectoral shares in region [S×R]")
println("  - gamma_ls_scaled.npy : Sourcing shares [S×R]")
println("  - X_lrs.npy        : Trade flow matrix [R×R×S]")
println("  - X_ls.npy         : Total sales to downstream [R×S]")
println("  - Y_r.npy          : Downstream sales by region [R]")
println("\nTo create the map, run:")
println("  python3 compute_amplification_map.py --industry $industry --amp-folder $output_folder")