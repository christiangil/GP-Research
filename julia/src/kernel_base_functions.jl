# the base kernel functions that can be combined for use with GPs
using SpecialFunctions
using SymEngine


"checks the length of hyperparameters against the passed proper_length and adds a unity kernel_amplitude if necessary."
function check_hyperparameters(hyper::Union{Array{T,1},Array{Basic,1}}, proper_length::Int) where {T<:Real}
    if length(hyper) < proper_length
        @assert (length(hyper) + 1) == proper_length "incompatible amount of hyperparameters passed (too few)"
        hyper = prepend!(copy(hyper), [1.])
    end
    @assert length(hyper) == proper_length "incompatible amount of hyperparameters passed (too many)"
    return hyper
end


"Linear GP kernel"
function Linear_kernel_base(hyperparameters::Union{Array{T,1},Array{Basic,1}}, x1, x2) where {T<:Real}

    hyperparameters = check_hyperparameters(hyperparameters, 1+1)
    sigma_b, sigma_a = hyperparameters

    return sigma_b * sigma_b * vecdot(x1, x2) + sigma_a * sigma_a
end


"Radial basis function GP kernel (aka squared exonential, ~gaussian)"
function RBF_kernel_base(hyperparameters::Union{Array{T,1},Array{Basic,1}}, dif::Union{Basic,Real}) where {T<:Real}

    dif_sq = dif * dif

    hyperparameters = check_hyperparameters(hyperparameters, 1+1)
    kernel_amplitude, kernel_length = hyperparameters

    return kernel_amplitude * kernel_amplitude * exp(-dif_sq / (2 * (kernel_length * kernel_length)))
end


"Periodic kernel (for random cyclic functions)"
function Periodic_kernel_base(hyperparameters::Union{Array{T,1},Array{Basic,1}}, dif::Union{Basic,Real}) where {T<:Real}

    # abs_dif = sqrt(dif^2)
    # abs_dif = abs(dif)

    hyperparameters = check_hyperparameters(hyperparameters, 2+1)
    kernel_amplitude, kernel_period, kernel_length = hyperparameters

    # reframed to make it easier for symbolic derivatives to not return NaNs
    # using sin(abs(u))^2 = sin(u)^2 ( also = 1 - cos(u)^2 )
    # return kernel_amplitude * kernel_amplitude * exp(-2 * sin(pi * (abs_dif / kernel_period)) ^ 2 / (kernel_length ^ 2))
    sin_τ = sin(pi * (dif / kernel_period))
    return kernel_amplitude * kernel_amplitude * exp(-2 * sin_τ * sin_τ / (kernel_length * kernel_length))


end


"Quasi-periodic kernel"
function Quasi_periodic_kernel_base(hyperparameters::Union{Array{T,1},Array{Basic,1}}, dif::Union{Basic,Real}) where {T<:Real}

    hyperparameters = check_hyperparameters(hyperparameters, 3+1)
    kernel_amplitude, RBF_kernel_length, P_kernel_period, P_kernel_length = hyperparameters

    return RBF_kernel_base([RBF_kernel_length], dif) * Periodic_kernel_base([P_kernel_period, P_kernel_length], dif)
end


"Ornstein–Uhlenbeck (Exponential) kernel"
function OU_kernel_base(hyperparameters::Union{Array{T,1},Array{Basic,1}}, dif::Union{Basic,Real}) where {T<:Real}

    hyperparameters = check_hyperparameters(hyperparameters, 1+1)
    kernel_amplitude, kernel_length = hyperparameters

    return kernel_amplitude * kernel_amplitude * exp(-dif / kernel_length)
end


"general Matern kernel"
function Matern_kernel_base(hyperparameters::Union{Array{T,1},Array{Basic,1}}, dif::Union{Basic,Real}, nu::Real) where {T<:Real}

    hyperparameters = check_hyperparameters(hyperparameters, 1+1)
    kernel_amplitude, kernel_length = hyperparameters

    #limit of the function as it apporaches 0 (see https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function)
    if dif == 0
        return kernel_amplitude * kernel_amplitude
    else
        x = (sqrt(2 * nu) * dif) / kernel_length
        return kernel_amplitude * kernel_amplitude * ((2 ^ (1 - nu)) / (gamma(nu))) * x ^ nu * besselk(nu, x)
    end
end


"Matern 3/2 kernel"
function Matern32_kernel_base(hyperparameters::Union{Array{T,1},Array{Basic,1}}, dif::Union{Basic,Real}) where {T<:Real}

    hyperparameters = check_hyperparameters(hyperparameters, 1+1)
    kernel_amplitude, kernel_length = hyperparameters

    x = sqrt(3) * dif / kernel_length
    return kernel_amplitude * kernel_amplitude * (1 + x) * exp(-x)
end


"Matern 5/2 kernel"
function Matern52_kernel_base(hyperparameters::Union{Array{T,1},Array{Basic,1}}, dif::Union{Basic,Real}) where {T<:Real}

    hyperparameters = check_hyperparameters(hyperparameters, 1+1)
    kernel_amplitude, kernel_length = hyperparameters

    x = sqrt(5) * dif / kernel_length
    return kernel_amplitude * kernel_amplitude * (1 + x + (x * x) / 3) * exp(-x)
end


"""
Rational Quadratic kernel (equivalent to adding together many SE kernels
with different lengthscales. When α→∞, the RQ is identical to the SE.)
"""
function RQ_kernel_base(hyperparameters::Union{Array{T,1},Array{Basic,1}}, dif::Union{Basic,Real}) where {T<:Real}

    dif_sq = dif * dif

    hyperparameters = check_hyperparameters(hyperparameters, 2+1)
    kernel_amplitude, kernel_length, alpha = hyperparameters

    alpha = max(alpha, 0)
    return kernel_amplitude * kernel_amplitude * (1 + dif_sq / (2 * alpha * kernel_length * kernel_length)) ^ -alpha
end


"""
Bessel (function of the first kind) kernel
Bessel functions of the first kind, denoted as Jα(x), are solutions of Bessel's
differential equation that are finite at the origin (x = 0) for integer or positive α
http://crsouza.com/2010/03/17/kernel-functions-for-machine-learning-applications/#bessel
"""
function Bessel_kernel_base(hyperparameters::Union{Array{T,1},Array{Basic,1}}, dif::Union{Basic,Real}; nu=0) where {T<:Real}

    hyperparameters = check_hyperparameters(hyperparameters, 2+1)
    kernel_amplitude, kernel_length, n = hyperparameters

    @assert nu >= 0 "nu must be >= 0"

    return besselj(nu, kernel_length * dif) / (dif ^ (-n * nu))
end
