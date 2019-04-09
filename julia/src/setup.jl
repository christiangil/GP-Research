#setup.jl
using Pkg; using Pkg.API
Pkg.activate(".")
Pkg.instantiate()
Pkg.update()
Pkg.API.precompile()
