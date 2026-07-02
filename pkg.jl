import Pkg;
Pkg.add("QuasiMonteCarlo")
Pkg.add("StatsPlots")
Pkg.add("DataFrames")
Pkg.add("NPZ")
Pkg.add("Distributions")
Pkg.add("Plots")
Pkg.add("CSV")
Pkg.add("FixedEffectModels")
Pkg.add("RDatasets")
Pkg.add("Optim")
Pkg.add("ProgressMeter")
Pkg.add("SharedArrays")
Pkg.add("StatsBase")
Pkg.add("CategoricalArrays")
Pkg.add("HaltonSequences")
Pkg.add("Parquet")
# Required for GMM analytical mode (model_analytical.jl)
Pkg.add("SpecialFunctions")   # gamma function Γ(·)
Pkg.add("FastGaussQuadrature") # gausslegendre(n) for quadrature
Pkg.add("ForwardDiff")         # exact analytical Jacobian via forward-mode AD






