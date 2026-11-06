import Pkg
# StatsBase is pinned to >= 0.34: older versions call a Base.floatrange method that Julia
# 1.12 removed, which breaks the precompilation of Plots (via StatsBase.histrange), and the
# resolver can otherwise keep 0.33.x.
Pkg.add([[Pkg.PackageSpec(name = n) for n in ("CSV", "DataFrames", "Distributions", "NPZ",
          "Parquet", "Plots", "QuasiMonteCarlo", "SpecialFunctions", "Optim")];
         Pkg.PackageSpec(name = "StatsBase", version = "0.34")])
