# run this to have access to all of the functions Christian has written

# using Pkg
# Pkg.add("SpecialFunctions")
# Pkg.add("JLD2")
# Pkg.add("FileIO")
# Pkg.add("MultivariateStats")
# Pkg.add("HDF5")
# # Pkg.add("Rsvg")
# # Pkg.add("Plots")
# Pkg.add("PyPlot")
# # pkg"add https://github.com/eford/RvSpectraKitLearn.jl"
# Pkg.add("Optim")
# Pkg.add("Distributions")
# Pkg.add("SymEngine")
# Pkg.add("Flux")
# Pkg.add("IterativeSolvers")
# Pkg.add("UnitfulAstro")
# Pkg.add("Unitful")
# Pkg.update()

# importing functions
include("general_functions.jl")
include("GP_functions.jl")
include("RV_functions.jl")
include("PCA_functions.jl")
include("kernel_base_functions.jl")
include("kernel_creation_functions.jl")
include("plotting_functions.jl")
include("diagnostic_functions.jl")

# running testss
include("../test/runtests.jl")

# clear_variables()
