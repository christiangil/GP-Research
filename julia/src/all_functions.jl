#all_functions.jl

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

using Juno
using Profile
function juno_profile(f::Function)
    Profile.clear()
    @profile f()
    Juno.profiletree()
    Juno.profiler()
    @profiler f()
end

using BenchmarkTools
# using Traceur
