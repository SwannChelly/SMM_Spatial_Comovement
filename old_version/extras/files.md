# `extras/` — post-estimation analysis

Auxiliary scripts that run *after* a calibration is finished. They take the
estimated parameters and produce the economic objects used in the paper's
counterfactual and amplification sections. They are not part of the estimation
loop itself.

| File | Role |
|------|------|
| `untargeted_moments.jl` | Simulation engine for the untargeted comovement moments. Given calibrated parameters, it draws sector-level demand shocks, propagates them through the estimated supply-chain network, and returns the panel used to estimate the Table 2 regression `d ln x_{i,t} = α_i + β · d ln x_{s,t} + ε`. Called by `../test/run_untargeted_validation.jl`. |
| `compute_amplification_weights.jl` | Builds the trade-flow weights (share of sales to each downstream region, share re-spent locally) that define the regional amplification coefficients A_{r'} of Section 4.3. |
| `run_amplification_analysis.jl` | Full amplification pipeline: loads a calibrated economy, calls `compute_amplification_weights.jl`, and writes the amplification objects. Entry point for the amplification analysis. |
