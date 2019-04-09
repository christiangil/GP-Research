# run this to have access to all of the functions Christian has written

using Pkg
# add SpecialFunctions JLD2 FileIO MultivariateStats HDF5 PyPlot Distributions SymEngine Flux IterativeSolvers UnitfulAstro Unitful
Pkg.activate(".")
# Pkg.instantiate()
# Pkg.update()


# importing functions
include("base_functions.jl")
# include("general_functions.jl")
# include("GP_functions.jl")
# include("RV_functions.jl")
# include("PCA_functions.jl")
# include("kernel_base_functions.jl")
# include("kernel_creation_functions.jl")
include("plotting_functions.jl")
include("GP_plotting_functions.jl")
include("diagnostic_functions.jl")
