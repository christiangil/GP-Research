# # All of the packages I am using
using Pkg
# add SpecialFunctions JLD2 FileIO MultivariateStats HDF5 PyPlot Distributions SymEngine Flux IterativeSolvers UnitfulAstro Unitful
Pkg.activate(".")
Pkg.instantiate()
Pkg.update()