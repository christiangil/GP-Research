# run this to have access to all of the functions Christian has written

# using Pkg
# Pkg.add("SpecialFunctions")
# # Pkg.add("JLD")  # broken in Julia 1.0.0, syntax: try without catch or finally
# Pkg.add("JLD2")
# Pkg.add("MultivariateStats")
# Pkg.add("HDF5")
# Pkg.add("Rsvg")
# Pkg.add("Plots")
# # Pkg.add("PlotlyJS")  # broken in Julia 1.0.0, syntax: try without catch or finally
# # Pkg.add("Gadfly")  # broken in Julia 1.0.0, syntax: try without catch or finally
# Pkg.add("PyPlot")
# Pkg.add("PyCall")
# Pkg.add("LaTeXStrings")
# Pkg.add("https://github.com/eford/RvSpectraKitLearn.jl.git")
# Pkg.add("Optim")
# Pkg.update()

# include("reusable_code.jl")
include("GP_functions.jl")
include("general_functions.jl")
using PyPlot
# include("plotting_functions.jl")
# include("plotting_functions_PlotlyJS.jl")
include("diagnostic_functions.jl")
