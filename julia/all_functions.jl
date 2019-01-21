# run this to have access to all of the functions Christian has written

# using Pkg
# Pkg.add("SpecialFunctions")
# Pkg.add("JLD2")
# Pkg.add("FileIO")
# Pkg.add("MultivariateStats")
# Pkg.add("HDF5")
# Pkg.add("Rsvg")
# Pkg.add("Plots")
# Pkg.add("PyPlot")
# Pkg.add("PyCall")
# Pkg.add("LaTeXStrings")
# # Pkg.add("https://github.com/eford/RvSpectraKitLearn.jl.git")
# Pkg.add("Optim")
# Pkg.add("Distributions")
# Pkg.add("SymEngine")
# Pkg.add("ForwardDiff")
# Pkg.add("StaticArrays")
# Pkg.add("DiffResults")
# # Pkg.update()

# include("reusable_code.jl")
include("GP_functions.jl")
include("general_functions.jl")
include("plotting_functions.jl")
include("diagnostic_functions.jl")

clear_variables()
