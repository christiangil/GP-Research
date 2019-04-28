# these are all general purpose functions that aren't specifically related to
# radial velocity or GP calculations
using LinearAlgebra
using Distributed
using IterativeSolvers
using Printf  # for formatting print statements


"a generalized version of the built in append!() function"
function multiple_append!(a::AbstractArray{T,1}, b...) where {T<:Real}
    for i in 1:length(b)
        append!(a, b[i])
    end
end


"if array is symmetric, return the symmetric (and opionally cholesky factorized) version"
function symmetric_A(A::Union{AbstractArray{T1,2},Symmetric{T2,Array{T2,2}}}; ignore_asymmetry::Bool=false, chol::Bool=false) where {T1<:Real, T2<:Real}

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
function ridge_chol(A::Union{AbstractArray{T1,2},Symmetric{T2,Array{T2,2}}}) where {T1<:Real, T2<:Real}

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
ridge_chol(A::Cholesky{T,Array{T,2}}) where {T<:Real} = A


"""
gets the coefficients and differentiation orders necessary for two multiplied
functions with an arbitrary amount of parameters
"""
function product_rule(dorder::AbstractArray{T,1}) where {T<:Real}

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
function signficant_difference(A1::AbstractArray{T1}, A2::AbstractArray{T2}, dif::Real) where {T1<:Real, T2<:Real}
    A1mA2 = abs.(A1 - A2);
    return chop_array!(A1mA2; dif=(maximum([maximum(A1), maximum(A2)]) * dif))
end


"function similar to Mathematica's Chop[]"
function chop_array!(A::AbstractArray{T}; dif::Real=1e-6) where {T<:Real}
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
    return sum([(2 * iseven(i) - 1) * binomial(n, i) * f(x + (n / 2 - i) * h) for i in 0:n] / h ^ n)
end


"Return evenly spaced numbers over a specified interval. Equivalent to range but without the keywords"
linspace(start::Real, stop::Real, length::Integer) = range(start, stop=stop, length=length)
log_linspace(start::Real, stop::Real, length) = exp.(linspace(log(start), log(stop), length))


"Create a new array filling the non-zero entries of a template array with a vector of values"
function reconstruct_array(non_zero_entries, template_array::AbstractArray{T,2}) where {T<:Real}
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
function general_lst_sq(design_matrix::AbstractArray{T1,2}, data::AbstractArray{T2,1}; Σ::Union{Cholesky{T3,Array{T3,2}},Symmetric{T4,Array{T4,2}},AbstractArray{T5}}=ones(1)) where {T1<:Real, T2<:Real, T3<:Real, T4<:Real, T5<:Real}
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
function find_modes(data::AbstractArray{T,1}; amount::Integer=3) where {T<:Real}

    # creating index list for inds at modes
    mode_inds = [i for i in 2:(length(data)-1) if (data[i]>=data[i-1]) && (data[i]>=data[i+1])]
    if data[1] > data[2]; prepend!(mode_inds, 1) end
    if data[end] > data[end-1]; append!(mode_inds, length(data)) end

    # return highest mode indices
    return mode_inds[partialsortperm(-data[mode_inds], 1:amount)]

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
function nyquist_frequency(times::AbstractArray{T,1}; scale::Real=1) where {T<:Real}
    time_span = times[end] - times[1]
    return amount_of_samp_points / time_span / 2 / scale
end

uneven_nyquist_frequency(times; scale=4) = nyquist_frequency(times; scale=scale)


import Base.ndims
ndims(A::Cholesky{T,Array{T,2}}) where {T<:Real} = 2


"an empty function, so that a function that requires another function to be passed can use this as a default"
do_nothing() = nothing


"evaluate a polynomial. Originally provided by Eric Ford"
function eval_polynomial(x::Number, a::AbstractArray{T,1}) where T<:Number
    sum = a[end]
    for i in (length(a)-1):-1:1
        sum = a[i]+x*sum
    end
    return sum
end


"""
For distributed computing. Send a variable to a worker
stolen shamelessly from ParallelDataTransfer.jl
e.g.
sendto([1, 2], x=100, y=rand(2, 3))
z = randn(10, 10); sendto(workers(), z=z)
"""
function sendto(p::Union{T,Array{T,1}}; args...) where {T<:Integer}
    for i in p
        for (nm, val) in args
            @spawnat(i, Core.eval(Main, Expr(:(=), nm, val)))
        end
    end
end
