using LinearAlgebra
using Distributed

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


"""
A structure that holds a cholesky factorization and the sqrt of the amount that
the original matrix was scaled by to make the factoization possible
Works with the most important functions of applied to cholesky objects
(i.e. size, \\, inv, det, and logdet)
"""
struct scaled_chol
    chol_fact::Cholesky{Float64,Array{Float64,2}}
    scale::Union{Float64,Int}
end

import Base.size, Base.\, Base.inv, LinearAlgebra.det, LinearAlgebra.logdet
size(A::scaled_chol) = size(A.chol_fact)
(\)(A::scaled_chol, B::AbstractVecOrMat) = (A.chol_fact \ B) / A.scale
inv(A::scaled_chol) = inv(A.chol_fact) / A.scale
det(A::scaled_chol) = det(A.chol_fact) / (A.scale ^ 3)
logdet(A::scaled_chol) = logdet(A.chol_fact) - log(A.scale ^ 3)


"make it easy to get upper or lower verisons of the cholesky factorization"
function (scaled_cholesky::scaled_chol)(lower_or_upper::String)

    @assert (lower_or_upper in ["L", "U"]) "only \"L\" or \"U\" are accepted as inputs"

    if lower_or_upper=="L"
        working_array = scaled_cholesky.chol_fact.L
    elseif lower_or_upper=="U"
        working_array = scaled_cholesky.chol_fact.U
    end

    return working_array / sqrt(scaled_cholesky.scale)

end


"if needed, adds a ridge (or multiplies the matrix) to make a Cholesky factorization possible"
function ridge_chol(A::Union{Array{T,2},Symmetric{Float64,Array{Float64,2}}}; notification::Bool=true, ridge::Float64=1e-8)  where {T} # * maximum(A))

    factorization = copy(A)
    scale_magnitude = 0
    scale_factor = 1
    # only add a small ridge if necessary
    try
        factorization = cholesky(factorization)
        # factor = cholesky(copy(A))

    catch
        original_type = typeof(A)
        no_error = false
        log_max_A = log10(maximum(A))
        ridge_magnitude_min = minimum([convert(Int64, round(log_max_A + log10(ridge))), convert(Int64, round(log10(ridge)))])
        ridge_magnitude = copy(ridge_magnitude_min)

        while (!no_error) & (scale_magnitude<9)
            no_error = true
            try
                factorization = cholesky(copy(factorization) + UniformScaling(10 ^ ridge_magnitude))
            catch
                no_error = false
                ridge_magnitude += 1
                if (ridge_magnitude)>(log_max_A + scale_magnitude - 2)
                    scale_magnitude += 2
                    factorization = copy(A) .* 10 ^ scale_magnitude
                    ridge_magnitude = copy(ridge_magnitude_min)
                end
            end
        end

        if typeof(factorization)==original_type
            @warn "No amount of ridge adding or scaling could make the matrix positive definite. Returned matrix is not factorized!"
        else
            println("had to add a ridge of order 10^$ridge_magnitude and multiply matrix by 10^$scale_magnitude. log10(maximum(scaled_A)) = $(log_max_A+scale_magnitude)")
        end

        scale_factor = 10 ^ scale_magnitude

    end

    return scaled_chol(factorization, scale_factor)

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


function add_diagonal_term(original_array::Union{Symmetric{Float64,Array{Float64,2}},Array{Float64,2}}, diag_terms::Array{Float64,1}; ignore_asymmetry::Bool=false)
    @assert (size(original_array, 1) == length(diag_terms)) ["diagonal term is the wrong length"]
    # for i in 1:size(original_array, 1)
    #     original_array[i, i] +=  diag_terms[i] ^ 2
    # end
    return symmetric_A(original_array + Diagonal(diag_terms); ignore_asymmetry=ignore_asymmetry)
end


"Create a new array filling the non-zero entries of a template array with a vector of values"
function reconstruct_array(non_zero_entries::Union{Array{Any,1},Array{Float64,1}}, template_array::Array{Float64,2})
    @assert length(findall(!iszero, template_array))==length(non_zero_entries)
    new_array = zeros(size(template_array))
    new_array[findall(!iszero, template_array)] = non_zero_entries
    return new_array
end


"Log of the InverseGamma pdf. Equivalent to using Distributions; logpdf(InverseGamma(α, β), x)"
log_inverse_gamma(x::Float64, α::Float64=1., β::Float64=1.) = -(β / x) - ((1 + α) * log(x)) + (α * log(β)) - log(gamma(α))


"derivative of the Log of the InverseGamma pdf"
dlog_inverse_gamma(x::Float64, α::Float64=1., β::Float64=1.) = (β - x * (1 + α)) / (x ^ 2)
