# `test/` — validation and diagnostic scripts

Scripts that check the estimator is doing what it should. None of them is required
for a production run — they are run by hand when you want to *validate* a result or
*diagnose* an identification problem. Each one re-uses the same model and data
loading as `main.jl`, so it sees the exact same moments and masks.

| File | Role |
|------|------|
| `run_test.sh` | Launcher for the internal-validity Monte-Carlo (`run_internal_validity.jl`), mirroring `run.sh`. Usage: `./run_test.sh aero --n_coef=4 --n_tau=1 --beta0="0.5"`. |
| `run_internal_validity.jl` | **Does the estimator recover a known truth?** Builds a synthetic economy from a chosen parameter vector θ₀, overwrites the empirical targets with moments generated at θ₀, then re-estimates. Reports point-recovery error and confidence-interval coverage over many Monte-Carlo repetitions. |
| `run_untargeted_validation.jl` | **Out-of-sample check.** Feeds calibrated parameters into `../extras/untargeted_moments.jl` to simulate demand shocks and reproduce the paper's Table 2 comovement regression — moments that were *not* used in estimation. |
| `test_extensive_margin.jl` | Geometry screen for the analytical `reg_coef` moment: measures how large the "win-anywhere" destination set is and how much the closed-form FKG approximation biases the distance coefficient, deciding whether an exact (more expensive) computation is worth it. |
| `test_t_reorder.jl` | Guard test for the s-major flattening convention of the T parameters. Asserts the T-parameter axis and the γ-moment axis enumerate the (sector, region) pairs in the same order — a silent mismatch would fit the wrong T. |
