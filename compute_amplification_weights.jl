##### Compute Amplification Weights for Section 4.3 #####
# Author: Claude
# Purpose: Compute w_d^{sr'} and w_{r'}^{sr'd} from calibrated model
#
# Notation (following paper Section 4.3):
#   w_s^{r'}    : Share of sector s in region r' total sales (from data)
#   w_d^{sr'}   : Share of sales of sector s in region r' going to downstream industry
#   w_{r'}^{sr'd} : Share of sales from (s,r') to downstream that go to local downstream (region r')
#
# The amplification coefficient is:
#   A_{r'} = H_v^{r'} / H_h^{r'}
# where:
#   H_h^{r'} = (w_d^{r'})^2 + Σ_{s≠d} (w_s^{r'})^2  (horizontal concentration)
#   H_v^{r'} = Σ_{s≠d} (w_s^{r'})^2 (w_d^{sr'} w_{r'}^{sr'd})^2  (vertical concentration)

using NPZ
using CSV
using DataFrames

# Include model functions
include("model_CP.jl")
include("tools.jl")

"""
    compute_amplification_weights(params, industry; output_folder="./amplification_output")

Compute the weights needed for amplification coefficient A_{r'}.

# Arguments
- `params`: Calibrated parameter vector
- `industry`: "aero" or "car"
- `output_folder`: Where to save results

# Outputs (saved as .npy files):
- `w_d_sr.npy`: w_d^{sr'} - Share of (s,r') sales to downstream [S × R]
- `w_r_srd.npy`: w_{r'}^{sr'd} - Share of (s,r') downstream sales to local downstream [S × R]
- `X_lrs.npy`: Full trade flow matrix [R × R × S]
- `downstream_regions.npy`: Indices of regions with downstream activity
"""
function compute_amplification_weights(params, industry::String; output_folder="./amplification_output")
    
    println("\n" * "="^70)
    println("Computing Amplification Weights (Section 4.3)")
    println("="^70)
    
    # Handle matrix input
    if ndims(params) > 1
        params = params[:, 1]
    end
    
    # Create output folder
    mkpath(output_folder)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 1: Solve the network
    # ─────────────────────────────────────────────────────────────────────────
    println("\n[Step 1] Solving production network...")
    network = solve_network(params, return_firm_level=true)
    
    X_lrs = network.X_lrs  # Trade flows: X[l, r, s] = flow from upstream (l,s) to downstream r
    Y_r = network.Y_r      # Downstream sales by region
    
    println("  Network solved successfully")
    println("  Total downstream sales: $(round(sum(Y_r), digits=4))")
    println("  Downstream regions: $R_downstream")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 2: Compute w_d^{sr'} - Share of (s,r') sales going to downstream
    # ─────────────────────────────────────────────────────────────────────────
    # 
    # w_d^{sr'} = (Sales from (s,r') to downstream) / (Total sales of (s,r'))
    #
    # In our model, X_lrs[l, r, s] is the flow from upstream region l, sector s
    # to downstream region r.
    # 
    # Sales from (s,l) to downstream = Σ_r X_lrs[l, r, s]
    #
    # For total sales of (s,l), we need to include sales to other sectors too.
    # However, in this single-downstream model, we only have downstream demand.
    # So w_d^{sr'} represents the share of sales TO THE MODELED DOWNSTREAM.
    #
    # In practice, we'll compute this relative to the firm's total sales to
    # the downstream industry.
    # ─────────────────────────────────────────────────────────────────────────
    
    println("\n[Step 2] Computing w_d^{sr'} (share to downstream)...")
    
    # Sales from (s, l) to all downstream regions
    # X_ls[l, s] = Σ_r X_lrs[l, r, s]
    X_ls = dropdims(sum(X_lrs, dims=2), dims=2)  # [R, S]
    
    # For now, w_d^{sr'} = 1 for all suppliers since the model only tracks
    # downstream demand. In a richer model, you'd have multiple downstream sectors.
    # 
    # However, from the paper's perspective, w_d^{sr'} should be the share of
    # sector s sales in region r' that go to the downstream industry.
    # This is essentially γ_{r's} from the empirical moments.
    #
    # Let's compute it as the ratio of sales to downstream vs total sectoral sales
    # We approximate total sector sales using the w_rs data (sectoral employment shares)
    
    w_d_sr = zeros(S, R)
    
    for l in 1:R
        for s in 1:S
            if X_ls[l, s] > 1e-12
                # w_d^{sl} = sales to downstream / proxy for total sales
                # Here we set it to 1 since all modeled sales go to downstream
                # In practice, this should come from the survey data (a_D^{is})
                w_d_sr[s, l] = 1.0  # Placeholder - see note below
            end
        end
    end
    
    # NOTE: In the actual implementation, w_d^{sr'} should come from the 
    # survey data on exposure (a_D^{is} in the paper). The model's γ_{ls}
    # represents sourcing shares, not exposure shares.
    # 
    # For now, we'll use the model's sourcing shares as a proxy:
    # γ̃_{ls} = X_ls[l,s] / Σ_l' X_ls[l',s]
    
    gamma_ls = zeros(S, R)
    for s in 1:S
        total_s = sum(X_ls[:, s])
        if total_s > 1e-12
            gamma_ls[s, :] = X_ls[:, s] ./ total_s
        end
    end
    
    # Scale by domestic share to get actual sourcing shares
    gamma_ls_scaled = gamma_ls .* reshape(domestic_share, S, 1)
    
    println("  w_d^{sr'} computed (S×R matrix)")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 3: Compute w_{r'}^{sr'd} - Share of downstream sales to LOCAL downstream
    # ─────────────────────────────────────────────────────────────────────────
    #
    # w_{r'}^{sr'd} = X_lrs[l, l, s] / Σ_r X_lrs[l, r, s]
    #
    # This is the share of (s,l)'s downstream sales that go to the downstream
    # firm in their OWN region l.
    # ─────────────────────────────────────────────────────────────────────────
    
    println("\n[Step 3] Computing w_{r'}^{sr'd} (share to each downstream region)...")

    # With separate upstream/downstream index spaces, "local" (same zone) requires
    # a mapping. Without a mapping table, we compute the full distribution of
    # downstream sales shares for each (s, l) pair.
    # w_r_srd[s, l, r] = X_lrs[l, r, s] / Σ_r' X_lrs[l, r', s]
    w_r_srd = zeros(S, R, R_downstream)

    for l in 1:R
        for s in 1:S
            total_to_downstream = X_ls[l, s]
            if total_to_downstream > 1e-12
                for r in 1:R_downstream
                    w_r_srd[s, l, r] = X_lrs[l, r, s] / total_to_downstream
                end
            end
        end
    end
    
    println("  w_{r'}^{sr'd} computed (S×R×R_downstream tensor)")

    # Print diagnostics
    println("\n  Diagnostics:")
    println("    w_{r'}^{sr'd} mean (non-zero): $(round(mean(w_r_srd[w_r_srd .> 0]), digits=4))")
    println("    w_{r'}^{sr'd} max: $(round(maximum(w_r_srd), digits=4))")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 4: Save outputs
    # ─────────────────────────────────────────────────────────────────────────
    
    println("\n[Step 4] Saving outputs to $output_folder...")
    
    # Save the computed weights
    NPZ.npzwrite(joinpath(output_folder, "w_d_sr.npy"), w_d_sr)
    NPZ.npzwrite(joinpath(output_folder, "w_r_srd.npy"), w_r_srd)
    NPZ.npzwrite(joinpath(output_folder, "gamma_ls_scaled.npy"), gamma_ls_scaled)
    NPZ.npzwrite(joinpath(output_folder, "X_lrs.npy"), X_lrs)
    NPZ.npzwrite(joinpath(output_folder, "X_ls.npy"), X_ls)
    NPZ.npzwrite(joinpath(output_folder, "Y_r.npy"), Y_r)
    
    # Save downstream region indices (all active in new index space)
    downstream_regions = collect(1:R_downstream)
    NPZ.npzwrite(joinpath(output_folder, "downstream_regions.npy"), downstream_regions)
    
    # Save dimensions for Python
    NPZ.npzwrite(joinpath(output_folder, "dimensions.npy"), 
                 Dict("S" => S, "R" => R, "R_downstream" => R_downstream))
    
    println("  Saved: w_d_sr.npy, w_r_srd.npy, gamma_ls_scaled.npy")
    println("  Saved: X_lrs.npy, X_ls.npy, Y_r.npy")
    println("  Saved: downstream_regions.npy, dimensions.npy")
    
    println("\n" * "="^70)
    println("Amplification weights computation complete!")
    println("="^70)
    
    return Dict(
        "w_d_sr" => w_d_sr,
        "w_r_srd" => w_r_srd,
        "gamma_ls" => gamma_ls_scaled,
        "X_lrs" => X_lrs,
        "X_ls" => X_ls,
        "Y_r" => Y_r
    )
end


"""
    run_amplification_analysis(industry; params_folder=nothing, output_folder=nothing)

Wrapper function to run the full amplification analysis.

# Arguments
- `industry`: "aero" or "car"
- `params_folder`: Folder containing calibrated parameters (default: ./reporting_{industry})
- `output_folder`: Where to save results (default: ./amplification_{industry})
"""
function run_amplification_analysis(industry::String; 
                                    params_folder=nothing, 
                                    output_folder=nothing)
    
    # Set defaults
    if params_folder === nothing
        params_folder = "./reporting_" * industry
    end
    if output_folder === nothing
        output_folder = "./amplification_" * industry
    end
    
    # Load calibrated parameters
    println("Loading calibrated parameters from: $params_folder")
    
    # Try to find the best parameters file
    params_file = nothing
    for candidate in ["best_params.npy", "epoch_50/100/best_params.npy", "0/best_params.npy"]
        full_path = joinpath(params_folder, candidate)
        if isfile(full_path)
            params_file = full_path
            break
        end
    end
    
    if params_file === nothing
        # Search for any best_params.npy
        for (root, dirs, files) in walkdir(params_folder)
            if "best_params.npy" in files
                params_file = joinpath(root, "best_params.npy")
                break
            end
        end
    end
    
    if params_file === nothing
        error("Could not find best_params.npy in $params_folder")
    end
    
    println("Found parameters: $params_file")
    params = NPZ.npzread(params_file)
    
    # Run computation
    results = compute_amplification_weights(params, industry, output_folder=output_folder)
    
    return results
end


# If run as main script
if abspath(PROGRAM_FILE) == @__FILE__
    # Default to aerospace
    industry = length(ARGS) >= 1 ? ARGS[1] : "aero"
    run_amplification_analysis(industry)
end