# run this to have access to all of the functions Christian has written
# sans plotting, kernel creation, and testing functions

using Pkg
Pkg.activate(".")

using UnitfulAstro
using Unitful
const n_kep_parms = 6
const light_speed = uconvert(u"m/s",1u"c")
const light_speed_nu = ustrip(light_speed)

# importing functions
include("general_functions.jl")
include("chol_functions.jl")
include("problem_definition_functions.jl")
include("GP_functions.jl")
include("RV_functions.jl")
include("extra_RV_functions.jl")
include("RV_Jones_functions.jl")
include("prior_functions.jl")
include("keplerian_derivatives.jl")
# include("PCA_functions.jl")
# include("SOAP_functions.jl")
# include("kernel_base_functions.jl")
# include("kernel_creation_functions.jl")
# include("plotting_functions.jl")
# include("GP_plotting_functions.jl")
# include("diagnostic_functions.jl")
