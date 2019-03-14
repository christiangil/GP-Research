# these are all general purpose functions that aren't specifically related to
# radial velocity or GP calculations
using LinearAlgebra
using Distributed
using IterativeSolvers
using Printf


"a generalized version of the built in append!() function"
function multiple_append!(a::Array{T,1}, b...) where {T<:Real}
    for i in 1:length(b)
        append!(a, b[i])
    end
end


"if array is symmetric, return the symmetric (and opionally cholesky factorized) version"
function symmetric_A(A::Union{Array{T1,2},Symmetric{T2,Array{T2,2}}}; ignore_asymmetry::Bool=false, chol::Bool=false) where {T1<:Any, T2<:Real}

    if size(A, 1) == size(A, 2)
        max_dif = maximum(abs.(A - transpose(A)))

        if max_dif == zero(max_dif)
            A = Symmetric(A)
        # an arbitrary threshold that is meant to catch numerical errors
        elseif (max_dif < maximum([1e-6 * maximum(abs.(A)), 1e-8])) | ignore_asymmetry
            # return the symmetrized version of the matrix
            A = Symmetric((A + transpose(A)) / 2)
        else
            println("Array dimensions match, but it is not symmetric")
            println(max_dif)
            chol = false
        end
    else
        println("Array dimensions do not match. The matrix can't be symmetric")
        chol = false
    end

    if chol
        return ridge_chol(A)
    else
        return A
    end

end


"if needed, adds a ridge based on the smallest eignevalue to make a Cholesky factorization possible"
function ridge_chol(A::Union{Array{T1,2},Symmetric{T2,Array{T2,2}}}) where {T1<:Any, T2<:Real}

    # only add a small ridge (based on the smallest eigenvalue) if necessary
    try
        return cholesky(A)
    catch
        smallest_eigen = IterativeSolvers.lobpcg(A, false, 1).λ[1]
        ridge = 1.10 * abs(smallest_eigen)
        @warn "added a ridge"
        println("ridge size:          10^$(log10(ridge))")
        println("max value of array:  10^$(log10(maximum(abs.(A))))")
        return cholesky(A + UniformScaling(ridge))
    end

end

"dont do anything if an array that is already factorized is passed"
ridge_chol(A::Cholesky{T,Array{T,2}}) where {T<:Any} = A


"""
gets the coefficients and differentiation orders necessary for two multiplied
functions with an arbitrary amount of parameters
"""
function product_rule(dorder::Array{T,1}) where {T<:Any}

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
function signficant_difference(A1::Array{T1}, A2::Array{T2}, dif::Real) where {T1<:Real, T2<:Real}
    A1mA2 = abs.(A1 - A2);
    return chop_array!(A1mA2; dif=(maximum([maximum(A1), maximum(A2)]) * dif))
end


"function similar to Mathematica's Chop[]"
function chop_array!(A::Array{T}; dif::Real=1e-6) where {T<:Real}
    A[abs.(A) .< dif] = 0;
    return A
end


"""
return approximate derivatives based on central differences, a bit finicky based on h values
f is the function of x
x is the place you want the derivative at
n is the order of derivative
h is the step size
"""
function finite_differences(f, x::Real, n::Integer, h::Real)
    return sum([(-1) ^ i * binomial(n, i) * f(x + (n / 2 - i) * h) for i in 0:n] / h ^ n)
end


"Return evenly spaced numbers over a specified interval. Equivalent to range but without the keywords"
linspace(start::Real, stop::Real, length::Integer) = collect(range(start, stop=stop, length=length))
log_linspace(start::Real, stop::Real, length) = exp.(linspace(log(start), log(stop), length))


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
function reconstruct_array(non_zero_entries, template_array::Array{T,2}) where {T<:Real}
    @assert length(findall(!iszero, template_array))==length(non_zero_entries)
    new_array = zeros(size(template_array))
    new_array[findall(!iszero, template_array)] = non_zero_entries
    return new_array
end


"Log of the InverseGamma pdf. Equivalent to using Distributions; logpdf(InverseGamma(α, β), x)"
log_inverse_gamma(x::Real, α::Real=1., β::Real=1.) = -(β / x) - ((1 + α) * log(x)) + (α * log(β)) - log(gamma(α))


"derivative of the Log of the InverseGamma pdf"
dlog_inverse_gamma(x::Real, α::Real=1., β::Real=1.) = (β - x * (1 + α)) / (x * x)


"""
Solve a linear system of equations (optionally with variance values at each point or covariance array)
see (https://en.wikipedia.org/wiki/Generalized_least_squares#Method_outline)
"""
function general_lst_sq(design_matrix::Array{T1,2}, data::Array{T2,1}; Σ::Union{Cholesky{T3,Array{T3,2}},Symmetric{T4,Array{T4,2}},Array{T5}}=ones(1)) where {T1<:Real, T2<:Real, T3<:Real, T4<:Real, T5<:Real}
    @assert ndims(Σ) < 3 "the Σ variable needs to be a 1D or 2D array"

    if Σ == ones(1)
        return design_matrix \ data
    else
        if ndims(Σ) == 1
            Σ = Diagonal(Σ)
        else
            Σ = ridge_chol(Σ)
        end
        # try
        #     return ridge_chol(design_matrix' * (Σ \ design_matrix)) \ (design_matrix' * (Σ \ data))
        # catch
        return (design_matrix' * (Σ \ design_matrix)) \ (design_matrix' * (Σ \ data))
        # end
    end
end


"Return an amount of indices of local maxima of a data array"
function find_modes(data::Array{T,1}; amount::Integer=3) where {T<:Real}

    # creating index list for inds at modes
    mode_inds = [i for i in 2:(length(data)-1) if (data[i]>=data[i-1]) & (data[i]>=data[i+1])]
    if data[1] > data[2]
        prepend!(mode_inds, 1)
    end
    if data[end] > data[end-1]
        append!(mode_inds, length(data))
    end

    # get data values at each mode
    data_modes = data[mode_inds]

    # collect highest mode indices
    best_mode_inds = Array{Int64}(undef, amount)
    data_min = minimum(data)
    for i in 1:amount
        hold = argmax(data_modes)
        best_mode_inds[i] = mode_inds[hold]
        data_modes[hold] = data_min
    end
    return best_mode_inds
end


"assert all passed variables are positive"
function assert_positive(vars...)
    for i in vars
        @assert i>0 "passed a negative/0 variable that needs to be positive"
    end
end


"""
Nyquist frequency is half of the sampling rate of a discrete signal processing system
(https://en.wikipedia.org/wiki/Nyquist_frequency)
divide by another factor of 4 for uneven spacing
"""
function nyquist_frequency(times::Array{T,1}; uneven::Bool=false) where {T<:Real}
    time_span = times[end] - times[1]
    nyquist_freq = amount_of_samp_points / time_span / 2
    if uneven==true
        nyquist_freq /= 4
    end
    return nyquist_freq
end


import Base.ndims
ndims(A::Cholesky{T,Array{T,2}}) where {T<:Any} = 2


"an empty function, so that a function that requires another function to be passed can use this as a default"
do_nothing() = nothing
