# these are all general purpose functions that aren't specifically related to
# radial velocity or GP calculations
using LinearAlgebra
using Distributed
using IterativeSolvers
using Printf  # for formatting print statements
using Unitful


"a generalized version of the built in append!() function"
function multiple_append!(a::Vector{T}, b...) where {T<:Real}
    for i in 1:length(b)
        append!(a, b[i])
    end
end


"if array is symmetric, return the symmetric (and optionally cholesky factorized) version"
function symmetric_A(A::Union{Matrix{T},Symmetric{T,Matrix{T}}}; ignore_asymmetry::Bool=false, chol::Bool=false) where {T<:Real}

    # an arbitrary threshold that is meant to catch numerical errors
    thres = maximum([1e-6 * maximum(abs.(A)), 1e-8])

    if size(A, 1) == size(A, 2)
        max_dif = maximum(abs.(A - transpose(A)))

        if max_dif == zero(max_dif)
            A = Symmetric(A)

        elseif (max_dif < thres) || ignore_asymmetry
            # return the symmetrized version of the matrix
            A = symmetrize_A(A)
        else
            println("Array dimensions match, but the max dif ($max_dif) is greater than the threshold ($thres)")
            chol = false
        end
    else
        println("Array dimensions do not match. The matrix can't be symmetric")
        chol = false
    end

    return chol ? ridge_chol(A) : A

end


function symmetrize_A(A::Union{Matrix{T},Symmetric{T,Matrix{T}}}) where {T<:Real}
    return Symmetric((A + transpose(A)) / 2)
end


"if needed, adds a ridge based on the smallest eignevalue to make a Cholesky factorization possible"
function ridge_chol(A::Union{Matrix{T},Symmetric{T,Matrix{T}}}) where {T<:Real}

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
ridge_chol(A::Cholesky{T,Matrix{T}}) where {T<:Real} = A


"""
gets the coefficients and differentiation orders necessary for two multiplied
functions with an arbitrary amount of parameters
"""
function product_rule(dorder::Vector{T}) where {T<:Real}

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
function signficant_difference(A1::AbstractArray{T}, A2::AbstractArray{T}, dif::Real) where {T<:Real}
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
    return sum([powers_of_negative_one(i) * binomial(n, i) * f(x + (n / 2 - i) * h) for i in 0:n] / h ^ n)
end


"Return evenly spaced numbers over a specified interval. Equivalent to range but without the keywords"
linspace(start::Real, stop::Real, length::Integer) = range(start, stop=stop, length=length)
log_linspace(start::Real, stop::Real, length) = exp.(linspace(log(start), log(stop), length))


"Create a new array filling the non-zero entries of a template array with a vector of values"
function reconstruct_array(non_zero_entries, template_array::Matrix{T}) where {T<:Real}
    @assert length(findall(!iszero, template_array))==length(non_zero_entries)
    new_array = zeros(size(template_array))
    new_array[findall(!iszero, template_array)] = non_zero_entries
    return new_array
end


"""
Solve a linear system of equations (optionally with variance values at each point or covariance array)
see (https://en.wikipedia.org/wiki/Generalized_least_squares#Method_outline)
"""
function general_lst_sq(design_matrix::Matrix{T}, data::Vector{T}; Σ::Union{Cholesky{T,Matrix{T}},Symmetric{T,Matrix{T}},Matrix{T},Vector{T}}=ones(1)) where {T<:Real}
    @assert ndims(Σ) < 3 "the Σ variable needs to be a 1D or 2D array"

    if Σ == ones(1)
        return design_matrix \ data
    else
        if ndims(Σ) == 1
            Σ = Diagonal(Σ)
        else
            Σ = ridge_chol(Σ)
        end
        return (design_matrix' * (Σ \ design_matrix)) \ (design_matrix' * (Σ \ data))
    end
end


"Return an amount of indices of local maxima of a data array"
function find_modes(data::Vector{T}; amount::Integer=3) where {T<:Real}

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
function nyquist_frequency(times::Vector{T}; scale::Real=1) where {T<:Real}
    time_span = times[end] - times[1]
    return amount_of_samp_points / time_span / 2 / scale
end

uneven_nyquist_frequency(times; scale=4) = nyquist_frequency(times; scale=scale)


import Base.ndims
ndims(A::Cholesky{T,Matrix{T}}) where {T<:Real} = 2


"an empty function, so that a function that requires another function to be passed can use this as a default"
do_nothing() = nothing


"evaluate a polynomial. Originally provided by Eric Ford"
function eval_polynomial(x::Number, a::Vector{T}) where T<:Number
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
function sendto(workers::Union{T,Vector{T}}; args...) where {T<:Integer}
    for worker in workers
        for (var_name, var_value) in args
            @spawnat(worker, Core.eval(Main, Expr(:(=), var_name, var_value)))
        end
    end
end


"""
Automatically adds as many workers as there are CPU threads minus 2 if none are
active and no number of procs to add is given
"""
function auto_addprocs(;add_procs::Integer=0)
    # only add as any processors as possible if we are on a consumer chip
    if (add_procs==0) && (nworkers()==1) && (length(Sys.cpu_info())<=16)
        add_procs = length(Sys.cpu_info()) - 2
    end
    addprocs(add_procs)
    println("added $add_procs workers")
end


"finds -1 ^ power without calling ^"
powers_of_negative_one(power::Integer) = iseven(power) ? 1 : -1


"Return the a version of the passed vector after removing all zero entries"
remove_zeros(V::Vector{T} where T<:Real) = V[findall(!iszero, V)]


"""
Compute the logarithm of the Laplace approxmation for the integral of a function
of the following form
∫ exp(-λ g(y)) h(y) dy ≈ exp(-λ g(y*)) h(y*) (2π/λ)^(d/2) |H(y*)|^(-1/2)
where y* is the value of y at the global mode and H is the (Hessian) matrix of
second order partial derivatives of g(y) (see slide 10 of
http://www.stats.ox.ac.uk/~steffen/teaching/bs2HT9/laplace.pdf). When used to
calculate evidences, one can set λ = 1, g(y) = -log-likelihood,
h(y) = model prior, and H(y) = the Fisher information matrix (FIM) or the
Hessian matrix of the negative log-likelihood. Possible to improve with methods
from Ruli et al. 2016 (https://arxiv.org/pdf/1502.06440.pdf)?

Parameters:

H (matrix): Hessian matrix of second order partial derivatives of g(y) at y*
g (float): g(y*) in above formula
logh (float): log(h(y*)) in above formula
λ (float): λ in above formula

Returns:
float: An estimate of log(∫ exp(-λ g(y)) h(y) dy)

"""
function log_laplace_approximation(
    H::Union{Symmetric{T,Matrix{T}},Matrix{T}},
    g::Real,
    logh::Real;
    λ = 1
    ) where {T<:Real}

    @assert size(H, 1) == size(H, 2)
    n = size(H, 1)

    return logh - λ * g + 0.5 * (n * log(2 * π / λ) - logdet(H))

end


function planck(λÅ::Union{Quantity, Real}, TK::Union{Quantity, Real})
    λ = convert_and_strip_units(u"m", λÅ)u"m" / 10 ^ 10
    h = u"h"
    c = (light_speed)u"m/s"
    k = u"k"
    T = convert_and_strip_units(u"K", TK)u"K"
    # W·sr−1·m−3
    return strip_units(2 * h * c ^ 2 / λ^5 / (exp(h * c / (λ * k * T)) - 1))
end


"""
trapezoidal integration, shamelessly modified from
https://github.com/dextorious/NumericalIntegration.jl/blob/master/src/NumericalIntegration.jl
"""
function trapz(x::Vector{T}, y::Vector{T}) where {T<:Real}
    @assert length(x) == length(y) "x and y vectors must be of the same length!"
    integral = zero(T)
    # @fastmath @simd for i in 1:(length(y) - 1)
    @simd for i in 1:(length(y) - 1)
        @inbounds integral += (x[i+1] - x[i]) * (y[i] + y[i+1])
    end
    return integral / 2
end


"Normalize all of the columns (second index number) integrate to the same value"
function normalize_columns_to_first_integral!(ys::Matrix{T}, x::Vector{T}) where {T<:Real}
    integrated_first = trapz(x, ys[:, 1])
    for i in 1:size(ys, 2)
        ys[:, i] *= integrated_first / trapz(x, ys[:, i])
    end
    return ys
end
