# these functions are related to calculating priors on the model parameters
using SpecialFunctions

"""
Log of the InverseGamma PDF. Equivalent to using Distributions; logpdf(InverseGamma(α, β), x)
https://en.wikipedia.org/wiki/Inverse-gamma_distribution
"""
function log_inverse_gamma(x::Real, α::Real=1., β::Real=1.; d::Integer=0)
    @assert 0 <= d <= 2
    if d == 0
        x > 0 ? val = -(β / x) - (1 + α) * log(x) + α * log(β) - loggamma(α) : val = -Inf
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
        x > 0 ? val = -(x / θ) + (α - 1) * log(x) - α * log(θ) - loggamma(α) : val = -Inf
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
    min, max = min_max
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
    min, max = min_max
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


"""
Log of the Rayleigh PDF.
Not properly normalized because of truncation
"""
function log_Rayleigh(x::Real, σ::Real; d::Integer=0)
    @assert 0 <= d <= 2
    if d == 0
        0 <= x <= 1 ? val = -(x * x / (2 * σ * σ)) + log(x / σ / σ) : val = -Inf
    elseif d == 1
        0 <= x <= 1 ? val = 1 / x - x / σ / σ : val = 0
    elseif d == 2
        0 <= x <= 1 ? val = -1 / x / x - 1 / σ / σ : val = 0
    end
    return val
end


"""
Log of the 2D circle PDF
"""
function log_circle(x::Vector{<:Real}, min_max_r::Vector{<:Real}; d::Vector{<:Integer}=[0,0])
    @assert minimum(d) == 0
    @assert maximum(d) <= 2
    @assert sum(d) <= 2

    @assert length(x) == length(min_max_r) == length(d) == 2
    min_r, max_r = min_max_r

    # @assert min_r < sqrt(dot(x, x)) < max_r
    # @assert min_θ < atan(x[1], x[2]) < max_θ
    if all(d .== 0)
        min_r < sqrt(dot(x, x)) < max_r ? val = -log(2 * π * (max_r ^ 2 - min_r ^ 2)) : val = -Inf
    else
        return 0
    end
    return val
end


"""
Log of the 2D unit cone PDF
"""
function log_cone(x::Vector{<:Real}; d::Vector{<:Integer}=[0,0])
    @assert minimum(d) >= 0
    @assert maximum(d) <= 2
    @assert sum(d) <= 2

    @assert length(x) == length(d) == 2

    r_sq = dot(x, x)  # x^2 + y^2
    r = sqrt(r_sq)
    if d == [0,0]
        0 <= r < 1 ? val = log(3 / π * (1 - r)) : val = -Inf
    elseif d == [0,1]
        0 <= r < 1 ? val = x[2] / (r_sq - r) : val = 0
    elseif d == [0,2]
        0 <= r < 1 ? val = (-x[2] ^ 2 * r + x[1] ^ 2 * (r - 1)) /
            (r ^ 3 * (1 - 2 * r + r_sq)) : val = 0
    elseif d == [1,0]
        0 <= r < 1 ? val = x[1] / (r_sq - r) : val = 0
    elseif d == [1,1]
        0 <= r < 1 ? val = (x[1] * x[2] * (1 - 2 * r)) /
            (r ^ 3 * (1 - 2 * r + r_sq)) : val = 0
    elseif d == [2,0]
        0 <= r < 1 ? val = (-x[1] ^ 2 * r + x[2] ^ 2 * (r - 1)) /
            (r ^ 3 * (1 - 2 * r + r_sq)) : val = 0
    end
    return val
end


"""
Log of the 2D unit quadratic cone PDF
"""
function log_quad_cone(x::Vector{<:Real}; d::Vector{<:Integer}=[0,0])
    @assert minimum(d) >= 0
    @assert maximum(d) <= 2
    @assert sum(d) <= 2

    @assert length(x) == length(d) == 2

    r_sq = dot(x, x)  # x^2 + y^2
    r = sqrt(r_sq)
    if d == [0,0]
        0 <= r < 1 ? val = log(6 / π * (1 - 2 * r + r_sq)) : val = -Inf
    elseif d == [0,1]
        0 <= r < 1 ? val = 2 * x[2] / (r_sq - r) : val = 0
    elseif d == [0,2]
        0 <= r < 1 ? val = 2 * (-x[2] ^ 2 * r + x[1] ^ 2 * (r - 1)) /
            (r ^ 3 * (1 - 2 * r + r_sq)) : val = 0
    elseif d == [1,0]
        0 <= r < 1 ? val = 2 * x[1] / (r_sq - r) : val = 0
    elseif d == [1,1]
        0 <= r < 1 ? val = (2 * x[1] * x[2] * (1 - 2 * r)) /
            (r ^ 3 * (1 - 2 * r + r_sq)) : val = 0
    elseif d == [2,0]
        0 <= r < 1 ? val = 2 * (-x[1] ^ 2 * r + x[2] ^ 2 * (r - 1)) /
            (r ^ 3 * (1 - 2 * r + r_sq)) : val = 0
    end
    return val
end


"""
Log of the 2D unit cubic cone PDF
"""
function log_cubic_cone(x::Vector{<:Real}; d::Vector{<:Integer}=[0,0])
    @assert minimum(d) >= 0
    @assert maximum(d) <= 2
    @assert sum(d) <= 2

    @assert length(x) == length(d) == 2

    r_sq = dot(x, x)  # x^2 + y^2
    r = sqrt(r_sq)
    if d == [0,0]
        0 <= r < 1 ? val = log(10 / π * (1 - r)^3) : val = -Inf
    elseif d == [0,1]
        0 <= r < 1 ? val = 3 * x[2] / (r_sq - r) : val = 0
    elseif d == [0,2]
        0 <= r < 1 ? val = 3 * (-x[2] ^ 2 * r + x[1] ^ 2 * (r - 1)) /
            (r ^ 3 * (1 - 2 * r + r_sq)) : val = 0
    elseif d == [1,0]
        0 <= r < 1 ? val = 3 * x[1] / (r_sq - r) : val = 0
    elseif d == [1,1]
        0 <= r < 1 ? val = (3 * x[1] * x[2] * (1 - 2 * r)) /
            (r ^ 3 * (1 - 2 * r + r_sq)) : val = 0
    elseif d == [2,0]
        0 <= r < 1 ? val = 3 * (-x[1] ^ 2 * r + x[2] ^ 2 * (r - 1)) /
            (r ^ 3 * (1 - 2 * r + r_sq)) : val = 0
    end
    return val
end


"""
Log of the 2D rotated Rayleigh PDF that is cutoff at r=1
ONLY ROUGHLY NORMALIZED according to σ = 1/5
"""
function log_rot_Rayleigh(x::Vector{<:Real}; d::Vector{<:Integer}=[0,0], σ=1/5)
    @assert minimum(d) >= 0
    @assert maximum(d) <= 2
    @assert sum(d) <= 2

    @assert length(x) == length(d) == 2
    r_sq = dot(x, x)  # x^2 + y^2
    r = sqrt(r_sq)
    σ_sq = σ ^ 2
    log_norm = -2 * log(σ) - 0.454215
    if d == [0,0]
        0 <= r < 1 ? val = -r_sq / (2 * σ_sq) + log(r) + log_norm : val = -Inf
    elseif d == [0,1]
        0 <= r < 1 ? val = -x[2] * (r_sq - σ_sq) / (r_sq * σ_sq) : val = 0
    elseif d == [0,2]
        0 <= r < 1 ? val = -(x[1] ^ 4 + x[1] ^ 2 * (2 * x[2] ^ 2 - σ_sq) + x[2] ^ 2 * (x[2] ^ 2 + σ_sq)) / (r_sq ^ 2 * σ_sq) : val = 0
    elseif d == [1,0]
        0 <= r < 1 ? val = -x[1] * (r_sq - σ_sq) / (r_sq * σ_sq) : val = 0
    elseif d == [1,1]
        0 <= r < 1 ? val = -2 * x[1] * x[2] / r_sq ^ 2 : val = 0
    elseif d == [2,0]
        0 <= r < 1 ? val = -(x[2] ^ 4 + x[2] ^ 2 * (2 * x[1] ^ 2 - σ_sq) + x[1] ^ 2 * (x[1] ^ 2 + σ_sq)) / (r_sq ^ 2 * σ_sq) : val = 0
    end
    return val
end


"""
Log of the bivariate normal PDF
NOTE THAT THAT WHEN USING lows!=[-∞,...], THIS IS NOT PROPERLY NORMALIZED
"""
function log_bvnormal(x::Vector{T}, Σ::Cholesky{T,Matrix{T}}; μ::Vector{T}=zeros(T, length(x)), d::Vector{<:Integer}=[0,0], lows::Vector{T}=zeros(T, length(x)) .- Inf) where {T<:Real}
    @assert minimum(d) >= 0
    @assert maximum(d) <= 2
    @assert sum(d) <= 2

    @assert length(x) == length(d) == length(μ) == 2
    y = x - μ

    if sum(d) == 0
        all(x .>= lows) ? val = -nlogL(Σ, y) : val = -Inf
    elseif sum(d) == 1
        y1 = Float64.(d)
        all(x .>= lows) ? val = -dnlogLdθ(y1, Σ \ y) : val = 0
    elseif d == [0,2]
        y1 = y2 = [0, 1.]
        all(x .>= lows) ? val = -d2nlogLdθ(y2, [0,0.], Σ \ y, Σ \ y1) : val = 0
    elseif d == [1,1]
        y1 = [1, 0.]
        y2 = [0, 1.]
        all(x .>= lows) ? val = -d2nlogLdθ(y2, [0,0.], Σ \ y, Σ \ y1) : val = 0
    elseif d == [2,0]
        y1 = y2 = [1, 0.]
        all(x .>= lows) ? val = -d2nlogLdθ(y2, [0,0.], Σ \ y, Σ \ y1) : val = 0
    end
    return val
end
function bvnormal_covariance(σ11::Real, σ22::Real, ρ::Real)
    assert_positive(σ11, σ22)
    @assert 0 <= ρ <= 1
    v12 = σ11*σ22*ρ
    return ridge_chol([σ11^2 v12;v12 σ22^2])
end

# Keplerian priors references
# https://arxiv.org/abs/astro-ph/0608328
# table 1

const prior_K_min = 1e-4#u"m/s"  # m/s
const prior_K_max = 2129#u"m/s"  # m/s, corresponds to a maximum planet-star mass ratio of 0.01
const prior_γ_min = -prior_K_max  # m/s
const prior_γ_max = prior_K_max  # m/s
const prior_P_min = 1#u"d"  # days
const prior_P_max = 1e3 * convert_and_strip_units(u"d", 1u"yr")#u"d"
const prior_K0 = 0.3#u"m/s"  # * sqrt(50 / n_meas)  # m/s
const prior_e_min = 0
const prior_e_max = 1
const prior_ω_min = 0  # radians
const prior_ω_max = 2 * π  # radians
const prior_M0_min = 0  # radians
const prior_M0_max = 2 * π  # radians
const characteristic_P = 10  # days


function logprior_K(K::Unitful.Velocity; d::Integer=0, P::Unitful.Time=characteristic_P * u"d")
    return log_loguniform(convert_and_strip_units(u"m/s", K), [prior_K_min, cbrt(prior_P_min / convert_and_strip_units(u"d", P)) * prior_K_max]; d=d, shift=prior_K0)
end

function logprior_P(P::Unitful.Time; d::Integer=0)
    return log_loguniform(convert_and_strip_units(u"d", P), [prior_P_min, prior_P_max]; d=d)
end

function logprior_M0(M0::Real; d::Integer=0)
    return log_uniform(M0, [prior_M0_min, prior_M0_max]; d=d)
end

function logprior_e(e::Real; d::Integer=0)
    # return log_uniform(e, [prior_e_min, prior_e_max]; d=d)
    return log_Rayleigh(e, 1/5; d=d)
end

function logprior_ω(ω::Real; d::Integer=0)
    return log_uniform(ω, [prior_ω_min, prior_ω_max]; d=d)
end

function logprior_hk(h::Real, k::Real; d::Vector{<:Integer}=[0,0])
    # return log_quad_cone([h, k]; d=d)
    return log_rot_Rayleigh([h, k]; d=d)
end

function logprior_γ(γ::Unitful.Velocity; d::Integer=0)
    return log_uniform(convert_and_strip_units(u"m/s", γ), [prior_γ_min, prior_γ_max]; d=d)
end

function logprior_kepler(
    K::Unitful.Velocity,
    P::Unitful.Time,
    M0::Real,
    e_or_h::Real,
    ω_or_k::Real,
    γ::Unitful.Velocity;
    d::Vector{<:Integer}=[0,0,0,0,0,0],
    use_hk::Bool=false)

    @assert minimum(d) == 0
    @assert maximum(d) <= 2
    @assert sum(d) <= 2

    if use_hk
        if any(d[4:5] .!= 0) && all(d[[1,2,3,6]] .== 0); return logprior_hk(e_or_h, ω_or_k; d=d[4:5]) end
        if sum(d .!= 0) > 1; return 0 end
    else
        if sum(d .!= 0) > 1; return 0 end
        if d[4] != 0; return logprior_e(e_or_h; d=d[4]) end
        if d[5] != 0; return logprior_ω(ω_or_k; d=d[5]) end
    end
    if d[1] != 0; return logprior_K(K; d=d[1]) end
    if d[2] != 0; return logprior_P(P; d=d[2]) end
    if d[3] != 0; return logprior_M0(M0; d=d[3]) end
    if d[6] != 0; return logprior_γ(γ; d=d[6]) end

    logP = logprior_K(K)
    # logP += logprior_K(K; P=P)
    logP += logprior_P(P)
    logP += logprior_M0(M0)
    use_hk ? logP += logprior_hk(e_or_h, ω_or_k) : logP += logprior_e(e_or_h; d=d[4]) + logprior_ω(ω_or_k; d=d[5])
    logP += logprior_γ(γ)

    return logP

end
logprior_kepler(ks::Union{kep_signal, kep_signal_epicyclic, kep_signal_wright}; d::Vector{<:Integer}=zeros(Int64, n_kep_parms), use_hk::Bool=false) =
    use_hk ? logprior_kepler(ks.K, ks.P, ks.M0, ks.h, ks.k, ks.γ; d=d, use_hk=use_hk) : logprior_kepler(ks.K, ks.P, ks.M0, ks.e, ks.ω, ks.γ; d=d, use_hk=use_hk)
function logprior_kepler_tot(ks::Union{kep_signal, kep_signal_epicyclic, kep_signal_wright}; d_tot::Integer=0, use_hk::Bool=false)
    @assert 0 <= d_tot <= 2
    if d_tot == 0
        return logprior_kepler(ks; d=zeros(Int64,n_kep_parms), use_hk=use_hk)
    elseif d_tot == 1
        G = zeros(n_kep_parms)
        for i in 1:n_kep_parms
            d = zeros(Int64,n_kep_parms)
            d[i] = 1
            G[i] = logprior_kepler(ks; use_hk=use_hk, d=d)
        end
        return grad
    else
        H = zeros(n_kep_parms, n_kep_parms)
        for i in 1:n_kep_parms
            for j in 1:n_kep_parms
                if i <= j
                    d = zeros(Int64,n_kep_parms)
                    d[i] += 1
                    d[j] += 1
                    # println(typeof(d))
                    H[i, j] = logprior_kepler(ks; d=d, use_hk=use_hk)
                end
            end
        end
        return Symmetric(H)
    end
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
