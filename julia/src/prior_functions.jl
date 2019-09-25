# these functions are related to calculating priors on the model parameters
using SpecialFunctions

"""
Log of the InverseGamma PDF. Equivalent to using Distributions; logpdf(InverseGamma(α, β), x)
https://en.wikipedia.org/wiki/Inverse-gamma_distribution
"""
function log_inverse_gamma(x::Real, α::Real=1., β::Real=1.; d::Integer=0)
    @assert 0 <= d <= 2
    if d == 0
        x > 0 ? val = -(β / x) - (1 + α) * log(x) + α * log(β) - log(gamma(α)) : val = -Inf
    elseif d == 1
        x > 0 ? val = (β / x - (1 + α)) / x : val = 0
    else
        x > 0 ? val = (-2 * β / x + (1 + α)) / (x * x) : val = 0
    end
    return val
end


function gamma_mode_std_2_alpha_theta(m::Real, s::Real)
    θ = (sqrt(m ^ 2 + 4 * s ^ 2) - m) / 2
    α = m / θ + 1
    return [α, θ]
end

"""
Log of the Gamma PDF. Equivalent to using Distributions; logpdf(Gamma(α, β), x)
https://en.wikipedia.org/wiki/Gamma_distribution
"""
function log_gamma(x::Real, parameters::Vector{<:Real}; d::Integer=0, passed_mode_std::Bool=false)
    @assert 0 <= d <= 2
    @assert length(parameters) == 2
    assert_positive(parameters)
    if passed_mode_std
        parameters = gamma_mode_std_2_alpha_theta(parameters[1], parameters[2])
    end
    α = parameters[1]
    θ = parameters[2]
    if d == 0
        x > 0 ? val = -(x / θ) + (α - 1) * log(x) - α * log(θ) - log(gamma(α)) : val = -Inf
    elseif d == 1
        x > 0 ? val = (α - 1) / x - 1 / θ : val = 0
    else
        x > 0 ? val = -(α - 1) / (x * x) : val = 0
    end
    return val
end

gauss_cdf(x::Real) = (1 + erf(x))/2

"log of the Gaussian PDF. Equivalent to using Distributions; logpdf(Gaussian(μ, σ), x)"
function log_gaussian(x::Real, parameters::Vector{<:Real}; d::Integer=0, min::Real=-Inf, max::Real=Inf)
    @assert 0 <= d <= 2
    @assert length(parameters) == 2
    μ = parameters[1]
    σ = parameters[2]
    assert_positive(σ)
    @assert min < max
    normalization = 1 - gauss_cdf(min - μ) - gauss_cdf(μ - max)
    if d == 0
        (min < x < max) ? val = -((x - μ)^2/(2 * σ * σ)) - log(sqrt(2 * π) * σ) - log(normalization) : val = -Inf
    elseif d == 1
        (min < x < max) ? val = -(x - μ)/(σ * σ) : val = 0
    else
        (min < x < max) ? val = -1 / (σ * σ) : val = 0
    end
    return val
end

"Log of the Uniform PDF."
function log_uniform(x::Real, min_max::Vector{<:Real}=[0,1]; d::Integer=0)
    @assert 0 <= d <= 2
    @assert length(min_max) == 2
    min = min_max[1]
    max = min_max[2]
    @assert min < max
    if d == 0
        min <= x <= max ? -log(max - min) : -Inf
    else
        return 0
    end
end

"""
Log of the log-Uniform PDF.
Flattens out in log space starting at shift
Also known as a (modified in shifted case) Jeffrey's prior
"""
function log_loguniform(x::Real, min_max::Vector{<:Real}; d::Integer=0, shift::Real=0)
    @assert 0 <= d <= 2
    @assert length(min_max) == 2
    min = min_max[1]
    max = min_max[2]
    @assert 0 < min + shift < max + shift
    xpshift = x + shift
    if d == 0
        min <= x <= max ? val = -log(xpshift) - log(log((max + shift)/(min + shift))) : val = -Inf
    elseif d == 1
        min <= x <= max ? val = -1 / xpshift : val = 0
    elseif d == 2
        min <= x <= max ? val = 1 / (xpshift * xpshift) : val = 0
    end
    return val
end

# Keplerian priors references
# https://arxiv.org/abs/astro-ph/0608328
# table 1

const prior_K_min = 0  # m/s
const prior_K_max = 2129  # m/s, corresponds to a maximum planet-star mass ratio of 0.01
const prior_γ_min = -prior_K_max  # m/s
const prior_γ_max = prior_K_max  # m/s
const prior_P_min = 1  # days
const prior_P_max = 1e3 * convert_and_strip_units(u"d", (1)u"yr") # days
const prior_K0 = 0.3  # * sqrt(50 / amount_of_measurements)  # m/s
const prior_e_min = 0
const prior_e_max = 1
const prior_ω_min = 0  # radians
const prior_ω_max = 2 * π  # radians
const prior_M0_min = 0  # radians
const prior_M0_max = 2 * π  # radians



function logprior_P(P::Real; d::Integer=0)
    return log_loguniform(P, [prior_P_min, prior_P_max]; d=d)
end

function logprior_e(e::Real; d::Integer=0)
    return log_uniform(e, [prior_e_min, prior_e_max]; d=d)
end

function logprior_M0(M0::Real; d::Integer=0)
    return log_uniform(M0, [prior_M0_min, prior_M0_max]; d=d)
end

function logprior_K(K::Real, P::Real; d::Integer=0)
    return log_loguniform(K, [prior_K_min, cbrt(prior_P_min / P) * prior_K_max]; d=d, shift=prior_K0)
end

function logprior_ω(ω::Real; d::Integer=0)
    return log_uniform(ω, [prior_ω_min, prior_ω_max]; d=d)
end

function logprior_γ(γ::Real; d::Integer=0)
    return log_uniform(γ, [prior_γ_min, prior_γ_max]; d=d)
end

function logprior_kepler(
    P::Real,
    e::Real,
    M0::Real,
    K::Real,
    ω::Real,
    γ::Real)

    logP = logprior_P(P)
    logP += logprior_e(e)
    logP += logprior_M0(M0)
    logP += logprior_K(K, P)
    logP += logprior_ω(ω)
    logP += logprior_γ(γ)

    return logP

end











# const prior_α = 1
# const prior_β = 5
#
# function logprior_kernel_hyperparameters(
#     n_kern_hyper::Integer,
#     total_hyperparameters::Vector{T}
#     ) where {T<:Real}
#
#     logP = 0
#     for i in 1:n_kern_hyper
#         logP += log_inverse_gamma(total_hyperparameters[end + 1 - i], prior_α, prior_β)
#     end
#     return logP
# end
#
# function nlogprior_kernel_hyperparameters(
#     n_kern_hyper::Integer,
#     total_hyperparameters::Vector{T}
#     ) where {T<:Real}
#
#     return -logprior_kernel_hyperparameters(n_kern_hyper, total_hyperparameters)
# end
#
# function logprior_kernel_hyperparameters!(
#     G::Vector{T},
#     n_kern_hyper::Integer,
#     total_hyperparameters::Vector{T}
#     ) where {T<:Real}
#
#     @assert length(findall(!iszero, total_hyperparameters)) == length(G)
#     for i in 1:n_kern_hyper
#         kernel_length = total_hyperparameters[end + 1 - i]
#         G[end + 1 - i] += log_inverse_gamma(kernel_length, prior_α, prior_β; d=1)
#     end
# end
#
# function nlogprior_kernel_hyperparameters!(
#     G::Vector{T},
#     n_kern_hyper::Integer,
#     total_hyperparameters::Vector{T}
#     ) where {T<:Real}
#
#     G = -G
#     logprior_kernel_hyperparameters!(G, n_kern_hyper,total_hyperparameters)
#     G = -G
# end
#
# function logprior_kernel_hyperparameters!(
#     H::Union{Symmetric{T,Matrix{T}},Matrix{T}},
#     n_kern_hyper::Integer,
#     total_hyperparameters::Vector{T}
#     ) where {T<:Real}
#
#     @assert length(findall(!iszero, total_hyperparameters)) == size(H, 1) == size(H, 2)
#     for i in 1:n_kern_hyper
#         H[end + 1 - i, end + 1 - i] -= log_inverse_gamma(total_hyperparameters[end + 1 - i], prior_α, prior_β; d=2)
#     end
# end
#
# function nlogprior_kernel_hyperparameters!(
#     H::Union{Symmetric{T,Matrix{T}},Matrix{T}},
#     n_kern_hyper::Integer,
#     total_hyperparameters::Vector{T}
#     ) where {T<:Real}
#
#     H = -H
#     logprior_kernel_hyperparameters!(H, n_kern_hyper, total_hyperparameters)
#     H = -H
# end
