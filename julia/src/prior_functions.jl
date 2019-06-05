# these functions are related to calculating priors on the model parameters

"Log of the InverseGamma pdf. Equivalent to using Distributions; logpdf(InverseGamma(α, β), x)"
function log_inverse_gamma(x::Real, α::Real=1., β::Real=1.; d::Integer=0)
    @assert 0 <= d <= 2
    if d==0
        x > 0 ? val = -(β / x) - (1 + α) * log(x) + α * log(β) - log(gamma(α)) : val = -Inf
    elseif d==1
        x > 0 ? val = (β / x - (1 + α)) / x : val = 0
    else
        x > 0 ? val = (-2 * β / x + (1 + α)) / (x * x) : val = 0
    end
    return val
end

"Log of the Uniform pdf."
function log_uniform(x::Real, min::Real=0, max::Real=1.; d::Integer=0)
    @assert 0 <= d <= 2
    @assert min < max
    if d==0
        min <= x <= max ? -log(max - min) : -Inf
    else
        return 0
    end
end

"""
Log of the log-Uniform pdf.
Flattens out in log space starting at shift
"""
function log_loguniform(x::Real, min::Real, max::Real; d::Integer=0, shift::Real=0)
    @assert 0 <= d <= 2
    @assert 0 < min + shift < max + shift
    xpshift = x + shift
    if d==0
        min <= x <= max ? val = -log(xpshift) - log(log((max + shift)/(min + shift))) : val = -Inf
    elseif d==1
        min <= x <= max ? val = -1 / xpshift : val = 0
    else
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
    return log_loguniform(P, prior_P_min, prior_P_max; d=d)
end

function logprior_e(e::Real; d::Integer=0)
    return log_uniform(e, prior_e_min, prior_e_max; d=d)
end

function logprior_M0(M0::Real; d::Integer=0)
    return log_uniform(M0, prior_M0_min, prior_M0_max; d=d)
end

function logprior_K(K::Real, P::Real; d::Integer=0)
    return log_loguniform(K, prior_K_min, cbrt(prior_P_min / P) * prior_K_max; d=d, shift=prior_K0)
end

function logprior_ω(ω::Real; d::Integer=0)
    return log_uniform(ω, prior_ω_min, prior_ω_max; d=d)
end

function logprior_γ(γ::Real; d::Integer=0)
    return log_uniform(γ, prior_γ_min, prior_γ_max; d=d)
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

const prior_α = 1
const prior_β = 5

function logprior_kernel_hyperparameters(
    total_hyperparameters::AbstractArray{T,1}
    ) where {T<:Real}

    logP = 0
    for i in 1:prob_def.n_kern_hyper
        logP += log_inverse_gamma(total_hyperparameters[end + 1 - i], prior_α, prior_β)
    end
    return logP
end

function logprior_kernel_hyperparameters!(
    G::AbstractArray{T,1},
    total_hyperparameters::AbstractArray{T,1}
    ) where {T<:Real}

    @assert length(total_hyperparameters) == length(G)
    for i in 1:prob_def.n_kern_hyper
        kernel_length = total_hyperparameters[end + 1 - i]
        G[end + 1 - i] += dlog_inverse_gamma(kernel_length, prior_α, prior_β)
    end
end

function nlogprior_kernel_hyperparameters!(
    G::AbstractArray{T,1},
    total_hyperparameters::AbstractArray{T,1}
    ) where {T<:Real}

    G .*= -1
    logprior_kernel_hyperparameters!(G, total_hyperparameters)
    G .*= -1
end

function logprior_kernel_hyperparameters!(
    H::AbstractArray{T,2},
    total_hyperparameters::AbstractArray{T,1}
    ) where {T<:Real}

    @assert length(total_hyperparameters) == size(H, 1) == size(H, 2)
    for i in 1:prob_def.n_kern_hyper
        H[end + 1 - i, end + 1 - i] -= d2log_inverse_gamma(total_hyperparameters[end + 1 - i], prior_α, prior_β)
    end
end

function nlogprior_kernel_hyperparameters!(
    H::AbstractArray{T,2},
    total_hyperparameters::AbstractArray{T,1}
    ) where {T<:Real}

    H .*= -1
    logprior_kernel_hyperparameters!(G, total_hyperparameters)
    H .*= -1
end
