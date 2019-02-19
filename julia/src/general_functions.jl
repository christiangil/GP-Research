using LinearAlgebra
using Distributed
using IterativeSolvers
using PositiveFactorizations
using Printf


"a generalized version of the built in append!() function"
function append(a::Array{T,1}, b...) where {T}
    for i in 1:length(b)
        append!(a, b[i])
    end
    return a
end


"if array is symmetric, return the symmetric version"
function symmetric_A(A::Union{Array{T,2},Symmetric{Float64,Array{Float64,2}}}; ignore_asymmetry::Bool=false) where {T}

    if size(A, 1) == size(A, 2)
        max_dif = maximum(abs.(A - transpose(A)))

        if max_dif == zero(max_dif)
            return Symmetric(A)
        # an arbitrary threshold that is meant to catch numerical errors
    elseif (max_dif < maximum([1e-6 * maximum(abs.(A)), 1e-8])) | ignore_asymmetry
            # return the symmetrized version of the matrix
            return Symmetric((A + transpose(A)) / 2)
        else
            println("Array dimensions match, but it is not symmetric")
            println(max_dif)
            return A
        end
    else
        println("Array dimensions do not match. The matrix can't be symmetric")
        return A
    end
end


"if needed, adds a ridge (or multiplies the matrix) to make a Cholesky factorization possible"
function ridge_chol(A::Union{Array{T,2},Symmetric{Float64,Array{Float64,2}}}; notification::Bool=true)  where {T} # * maximum(A))

    # only add a small ridge (based on the smallest eigenvalue) if necessary
    try
        return cholesky(Positive, A)
    catch
        smallest_eigen = IterativeSolvers.lobpcg(A, false, 1).λ[1]
        ridge = 1.10 * abs(smallest_eigen)
        println("tried adding a ridge of size 10^$(log10(ridge)) to matrix whose maximum value is 10^$(log10(maximum(abs.(A))))")
        return cholesky(Positive, A + UniformScaling(ridge))
    end

end


"gets the coefficients and differentiation orders necessary for two multiplied
functions with an arbitrary amount of parameters"
function product_rule(dorder::Array{T,1}) where {T}

    # initializing the final matrix with a single combined function with no derivatives
    total = transpose(append!([1], zeros(2 * length(dorder))))

    # Do each parameter individually
    for i in 1:length(dorder)

        # create a holding matrix so we don't have to modify the total matrix
        hold = zeros(0, size(total, 2))

        # assign the coefficients and dorders for each combination of dorders
        for j in 0:dorder[i]

            # copying the current state of the total matrix
            total_copy = copy(total)

            # modifying the current coefficient by the proper amount
            total_copy[:, 1] = total_copy[:, 1] * binomial(dorder[i], j)

            # setting the new dorders for this parameter
            current_length = size(total_copy, 1)
            total_copy[:, i + 1] = (dorder[i] - j) * ones(current_length)
            total_copy[:, i + 1 + length(dorder)] = j * ones(current_length)

            # add this to the holding matrix
            hold = vcat(hold, total_copy)
        end

        # update the total matrix
        total = hold

    end

    # total is of the following format
    # (coefficient, amount of parameters *
    # (dorder applied to first function for that parameter, dorder applied to second function))
    return total
end


"""
find differences between two arrays and set values smaller than a threshold to be zero
use isapprox instead if you care about boolean result
"""
function signficant_difference(A1::Array{Float64}, A2::Array{Float64}, dif::Float64)
    A1mA2 = abs.(A1 - A2);
    A1mA2[A1mA2 .< (maximum([maximum(A1), maximum(A2)]) * dif)] .= 0;
    return A1mA2
end


"function similar to Mathematica's Chop[]"
function chop_array(A::Array{Float64}; dif = 1e-6)
    A[abs.(A) .< dif] = 0;
    return A
end


"return approximate derivatives based on central differences"
# a bit finicky based on h values
# f is the function of x
# x is the place you want the derivative at
# n is the order of derivative
# h is the step size
function finite_differences(f, x::Float64, n::Int, h::Float64)
    return sum([(-1) ^ i * binomial(n, i) * f(x + (n / 2 - i) * h) for i in 0:n] / h ^ n)
end


"Return evenly spaced numbers over a specified interval equivalent to range but without the keywords"
linspace(start::Union{Float64,Int}, stop::Union{Float64,Int}, length::Int) = collect(range(start, stop=stop, length=length))
log_linspace(start::Union{Float64,Int}, stop::Union{Float64,Int}, length::Int) = exp.(linspace(log(start), log(stop), length))

"set all variables equal to nothing to save some memory"
function clear_variables()
    for var in names(Main)
        try
            # eval(Meta.parse("$var=0"))\
            # eval(Meta.parse("$var=nothing"))
            clear!(var)
        catch
        end
    end
    # GC.gc()
end


"Create a new array filling the non-zero entries of a template array with a vector of values"
function reconstruct_array(non_zero_entries, template_array::Array{Float64,2})
    @assert length(findall(!iszero, template_array))==length(non_zero_entries)
    new_array = zeros(size(template_array))
    new_array[findall(!iszero, template_array)] = non_zero_entries
    return new_array
end


"Log of the InverseGamma pdf. Equivalent to using Distributions; logpdf(InverseGamma(α, β), x)"
log_inverse_gamma(x::Float64, α::Float64=1., β::Float64=1.) = -(β / x) - ((1 + α) * log(x)) + (α * log(β)) - log(gamma(α))


"derivative of the Log of the InverseGamma pdf"
dlog_inverse_gamma(x::Float64, α::Float64=1., β::Float64=1.) = (β - x * (1 + α)) / (x * x)


"""
Solve a linear system of equations (optionally with variance values at each point or covariance array)
see (https://en.wikipedia.org/wiki/Generalized_least_squares#Method_outline)
"""
function general_lst_sq(design_matrix::Array{Float64,2}, data::Array{Float64,1}; covariance::Union{Symmetric{Float64,Array{Float64,2}},Array{Float64}}=ones(1))
    @assert ndims(covariance) < 3 "the covariance variable needs to be a 1D or 2D array"

    # prevent the original terms from being modified
    A = copy(design_matrix)

    # make sure term array is the right shape
    if size(A, 2) == length(data)
        A = A'
    end

    if covariance == ones(1)
        return A \ data
    else
        if ndims(covariance) == 1
            Σ = Diagonal(covariance)
        else
            Σ = ridge_chol(covariance)
        end
        return ridge_chol((A' * (Σ \ A))) \ (A' * (Σ \ data))
    end
end
