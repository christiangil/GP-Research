# these are all custom diagnostic functions. May help with debugging
using ModelingToolkit

differentiate(func, v) = MTKsimplify(expand_derivatives(Differential(v)(func)))
function differentiate(func, v, n::Integer)
    n < 0 && throw(DomainError("n must be non-negative integer"))
    n==0 && return func
    n==1 && return differentiate(func, v)
    n > 1 && return differentiate(differentiate(func, v), v, n-1)
end


get_params!(O::ModelingToolkit.Constant, params::Vector{Variable}) = nothing
function get_params!(O::Operation, params::Vector{Variable})
    if isa(O.op, Variable)
        # if !(O.op in params)
        append!(params, [O.op])
        # end
    else
        for sub_O in O.args
            get_params!(sub_O, params)
        end
    end
end
function get_params(O::Operation; return_symbs::Bool=false, return_strings::Bool=false)
    params = Variable[]
    get_params!(O, params)
    params = unique(params)
    if return_symbs
        return [param.name for param in params]
    elseif return_strings
        return [string(param.name) for param in params]
    else
        return params
    end
end


MTKsubs(symbolic_kernel::Operation, before::Union{Operation,String}, after::Union{Operation,String}) = simplify_constants(eval(Meta.parse(replace(string(symbolic_kernel), string(before) => string(after)))))


function MTKsimplify(O::Operation)
    params = get_params(O; return_strings=true)
    for param in params
        eval(Meta.parse("$param = symbols(\"$param\")"))
    end
    O = eval(Expr(O))
    for param in params
        eval(Meta.parse("$param = Variable(:$param; known=true)()"))
    end
    return eval(Meta.parse(SymEngine.toString(O)))
end


# RANDOM USEFUL CODE TIDBITS

# # import DiffRules
# # @register DiffRules._abs_deriv(t)
# # ModelingToolkit.derivative(::typeof(DiffRules._abs_deriv), args::NTuple{1,Any}, ::Val{1}) = 0
# # dabs(t) = powers_of_negative_one(t < 0)
# import Base.abs
# @register abs(t)
# ModelingToolkit.derivative(::typeof(abs), args::NTuple{1,Any}, ::Val{1}) = dabs(args[1])
# @register dabs(t)
# ModelingToolkit.derivative(::typeof(dabs), args::NTuple{1,Any}, ::Val{1}) = 0

# @parameters δ δp se_λ p_λ

# δ = Variable(:δ; known=true)()

# P_sym = Variable(Symbol(P_sym_str); known = true)()
# eval(Meta.parse(P_sym_str * "=P_sym"))
# πval = Variable(:πval; known = true)()
# @assert occursin(periodic_var, string(symbolic_kernel_original)) "can't find periodic variable"
# symbolic_kernel_original = MTKsubs(symbolic_kernel_original, "abs($periodic_var)", 2*sin(πval*abs(δ)/P_sym))
# symbolic_kernel_original = MTKsubs(symbolic_kernel_original, periodic_var, 2*sin(πval*δ/P_sym))

# symbs = get_params(symbolic_kernel_original)
#
# δ_inds = findall(x -> x==δ.op, symbs)
# @assert length(δ_inds) == 1
# deleteat!(symbs, δ_inds[1])
# π_inds = findall(x -> x==πval.op, symbs)
# deleteat!(symbs, π_inds[1])
# symbs_str = [string(symb) for symb in symbs]


        # performing the differentiations
# symbolic_kernel = MTKsimplify(copy(symbolic_kernel_original))
# symbolic_kernel = differentiate(symbolic_kernel, δ, dorder[1])
# for j in 1:hyper_amount
#     # println(symbs[j], dorder[j+1])
#     symb = Variable(symbs[j].name; known=true)()
#     symbolic_kernel = differentiate(symbolic_kernel, symb, dorder[j+1])
# end
