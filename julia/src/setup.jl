#setup.jl
using Pkg; using Pkg.API
Pkg.add("SpecialFunctions")
Pkg.add("BenchmarkTools")
Pkg.add("DistributedArrays")
Pkg.add("Distributions")
Pkg.add("FileIO")
# Pkg.add("Flux")
Pkg.add("Optim")
Pkg.add("HDF5")
Pkg.add("IterativeSolvers")
Pkg.add("JLD2")
Pkg.add("MultivariateStats")
Pkg.add("PyPlot")
Pkg.add("SymEngine")
Pkg.pin(PackageSpec(name="SymEngine", version="0.6.0"))
Pkg.add("UnitfulAstro")
Pkg.add("Unitful")
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("FITSIO")
Pkg.add("LineSearches")
Pkg.add("PyCall")
# Pkg.add("ModelingToolkit")
# Pkg.add("UnitfulAngles")
# Pkg.add("DSP")
# Pkg.add("Interpolations")
# Pkg.add("NPZ")
Pkg.API.precompile()

# Pkg.add("CUDAdrv")
# Pkg.add("CUDAnative")
# Pkg.add("CuArrays")
# Pkg.add("GPUArrays")

Pkg.update()
Pkg.activate(".")
Pkg.instantiate()
Pkg.update()
Pkg.API.precompile()
