# run this to have access to all of the functions Christian has written

# importing functions
include("general_functions.jl")
include("GP_functions.jl")
include("Flux_functions.jl")
# include("Optim_functions.jl")
include("RV_functions.jl")
include("PCA_functions.jl")
include("kernel_base_functions.jl")
include("kernel_creation_functions.jl")
include("plotting_functions.jl")
include("GP_plotting_functions.jl")
include("diagnostic_functions.jl")

# # running tests
# include("../test/runtests.jl")
"tries to include runtests.jl from common directories"
function run_tests()
    try
        return include("test/runtests.jl")
    catch
        return include("../test/runtests.jl")
    end
end
