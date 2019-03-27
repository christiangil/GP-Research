# the base kernel functions that can be combined for use with GPs
using SpecialFunctions
using SymEngine


"Scale kernel. Multiplied with other kernels to modify their amplitude"
scale_kernel_base(kernel_amplitude::Union{Real, Basic}, x1, x2) where {T<:Real} = kernel_amplitude * kernel_amplitude


"Linear GP kernel"
function linear_kernel_base(hyperparameters::Union{AbstractArray{T,1}, AbstractArray{Basic,1}}, x1, x2) where {T<:Real}

    @assert length(hyperparameters) == 2 "incompatible amount of hyperparameters passed"
    sigma_b, sigma_a = hyperparameters

    return sigma_b * sigma_b * vecdot(x1, x2) + sigma_a * sigma_a
end


"Radial basis function GP kernel (aka squared exonential, ~gaussian)"
function rbf_kernel_base(kernel_length::Union{Real, Basic}, dif::Union{Basic,Real}) where {T<:Real}
    return exp(-dif * dif / (2 * (kernel_length * kernel_length)))
end


"Periodic kernel (for random cyclic functions)"
function periodic_kernel_base(hyperparameters::Union{AbstractArray{T,1}, AbstractArray{Basic,1}}, dif::Union{Basic,Real}) where {T<:Real}

    # abs_dif = sqrt(dif^2)
    # abs_dif = abs(dif)

    @assert length(hyperparameters) == 2 "incompatible amount of hyperparameters passed"
    kernel_period, kernel_length = hyperparameters

    # reframed to make it easier for symbolic derivatives to not return NaNs
    # using sin(abs(u))^2 = sin(u)^2 ( also = 1 - cos(u)^2 )
    # return exp(-2 * sin(pi * (abs_dif / kernel_period)) ^ 2 / (kernel_length ^ 2))
    sin_τ = sin(pi * (dif / kernel_period))
    return exp(-2 * sin_τ * sin_τ / (kernel_length * kernel_length))


end


"Quasi-periodic kernel"
function quasi_periodic_kernel_base(hyperparameters::Union{AbstractArray{T,1}, AbstractArray{Basic,1}}, dif::Union{Basic,Real}) where {T<:Real}

    @assert length(hyperparameters) == 3 "incompatible amount of hyperparameters passed"
    RBF_kernel_length, P_kernel_period, P_kernel_length = hyperparameters

    return rbf_kernel_base(RBF_kernel_length, dif) * periodic_kernel_base([P_kernel_period, P_kernel_length], dif)
end


"Ornstein–Uhlenbeck (Exponential) kernel"
function ou_kernel_base(kernel_length::Union{Real, Basic}, dif::Union{Basic,Real}) where {T<:Real}
    return exp(-abs(dif) / kernel_length)
end


"Exponential-periodic kernel"
function exp_periodic_kernel_base(hyperparameters::Union{AbstractArray{T,1}, AbstractArray{Basic,1}}, dif::Union{Basic,Real}) where {T<:Real}

    @assert length(hyperparameters) == 3 "incompatible amount of hyperparameters passed"
    OU_kernel_length, P_kernel_period, P_kernel_length = hyperparameters

    return ou_kernel_base(OU_kernel_length, dif) * periodic_kernel_base([P_kernel_period, P_kernel_length], dif)
end


"general Matern kernel"
function matern_kernel_base(kernel_length::Union{Real, Basic}, dif::Union{Basic,Real}, nu::Real) where {T<:Real}

    #limit of the function as it apporaches 0 (see https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function)
    if dif == 0
        return kernel_amplitude * kernel_amplitude
    else
        x = (sqrt(2 * nu) * dif) / kernel_length
        return ((2 ^ (1 - nu)) / (gamma(nu))) * x ^ nu * besselk(nu, x)
    end
end


"Matern 3/2 kernel"
function matern32_kernel_base(kernel_length::Union{Real, Basic}, dif::Union{Basic,Real}) where {T<:Real}
    x = sqrt(3) * dif / kernel_length
    return (1 + x) * exp(-x)
end


"Matern 5/2 kernel"
function matern52_kernel_base(kernel_length::Union{Real, Basic}, dif::Union{Basic,Real}) where {T<:Real}
    x = sqrt(5) * dif / kernel_length
    return (1 + x + (x * x) / 3) * exp(-x)
end


"Matern 7/2 kernel"
function matern72_kernel_base(kernel_length::Union{Real, Basic}, dif::Union{Basic,Real}) where {T<:Real}
    x = sqrt(7) * dif / kernel_length
    return (1 + x + 2 * (x * x) / 5 + (x * x * x) / 15) * exp(-x)
end


"Matern 9/2 kernel"
function matern92_kernel_base(kernel_length::Union{Real, Basic}, dif::Union{Basic,Real}) where {T<:Real}
    x = 3 * dif / kernel_length
    return (1 + x + 3 * (x * x) / 7 + 2 * (x * x * x) / 21 + (x * x * x * x) / 105) * exp(-x)
end


"""
Rational Quadratic kernel (equivalent to adding together many SE kernels
with different lengthscales. When α→∞, the RQ is identical to the SE.)
"""
function rq_kernel_base(hyperparameters::Union{AbstractArray{T,1}, AbstractArray{Basic,1}}, dif::Union{Basic,Real}) where {T<:Real}

    @assert length(hyperparameters) == 2 "incompatible amount of hyperparameters passed"
    kernel_length, alpha = hyperparameters

    return (1 + dif * dif / (2 * alpha * kernel_length * kernel_length)) ^ -alpha
end


"""
Bessel (function of the first kind) kernel
Bessel functions of the first kind, denoted as Jα(x), are solutions of Bessel's
differential equation that are finite at the origin (x = 0) for integer or positive α
http://crsouza.com/2010/03/17/kernel-functions-for-machine-learning-applications/#bessel
"""
function bessel_kernel_base(hyperparameters::Union{AbstractArray{T,1}, AbstractArray{Basic,1}}, dif::Union{Basic,Real}; nu=0) where {T<:Real}

    @assert length(hyperparameters) == 2 "incompatible amount of hyperparameters passed"
    kernel_length, n = hyperparameters

    @assert nu >= 0 "nu must be >= 0"

    return besselj(nu + 1, kernel_length * dif) / (dif ^ (-n * (nu + 1)))
end
