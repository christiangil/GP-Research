# these functions are related to calculating RV quantities
using UnitfulAstro
using Unitful
using LinearAlgebra


"Strip Unitful units"
strip_units(quant::Quantity) = ustrip(float(quant))


"Convert Unitful units from one to another and strip the final units"
convert_and_strip_units(new_unit::Unitful.FreeUnits, quant::Quantity) = strip_units(uconvert(new_unit, quant))
convert_and_strip_units(new_unit::Unitful.FreeUnits, quant::Real) = quant


"""
Calculate velocity semi-amplitude for RV signal of planets in m/s
adapted from eq. 12 of (http://exoplanets.astro.yale.edu/workshop/EPRV/Bibliography_files/Radial_Velocity.pdf)
"""
function velocity_semi_amplitude(P::Real, m_star::Real, m_planet::Real; e::Real=0., i::Real=π/2)
    assert_positive(P, m_star, m_planet)
    comb_mass = m_star + m_planet
    K_au_yr = 2 * π * sin(i) * m_planet / (sqrt(1 - (e * e)) * cbrt(comb_mass * comb_mass * P))
    return K_au_yr * convert_and_strip_units(u"m", 1u"AU") / convert_and_strip_units(u"s", 1u"yr")
end

function velocity_semi_amplitude(P::Union{Real, Quantity}, m_star::Union{Real, Quantity}, m_planet::Union{Real, Quantity}; e::Real=0., i::Real=π/2)
    m_star = convert_and_strip_units(u"Msun", m_star)
    m_planet = convert_and_strip_units(u"Msun", m_planet)
    P = convert_and_strip_units(u"yr", P)
    return velocity_semi_amplitude(P, m_star, m_planet; e=e, i=i)
end


# unit_phase(t::Real, P::Real) = mod(t / P * 2 * π, 2 * π)
unit_phase(t::Real, P::Real) = t / P * 2 * π


"calculate mean anomaly"
mean_anomaly(t::Real, P::Real, M0::Real) = unit_phase(t, P) - M0
# mean_anomaly(t::Real, P::Real, M0::Real) = mod(unit_phase(t, P) - M0, 2 * π)
# mean_anomaly(t::Real, P::Real, M0::Real) = mod(t / P * 2 * π - M0, 2 * π)


"""   ecc_anom_init_guess_danby(M, ecc)
Initial guess for eccentric anomaly given mean anomaly (M) and eccentricity (ecc)
    Based on "The Solution of Kepler's Equations - Part Three"
    Danby, J. M. A. (1987) Journal: Celestial Mechanics, Volume 40, Issue 3-4, pp. 303-312  1987CeMec..40..303D
Julia function originally written by Eric Ford
"""
function ecc_anom_init_guess_danby(M::Real, e::Real)
    k = convert(typeof(e), 0.85)
    if M < zero(M); M += 2 * π end
    (M < π) ? M + k * e : M - k * e;
end


"""   update_ecc_anom_laguerre(E, M, ecc)
Update the current guess (E) for the solution to Kepler's equation given mean anomaly (M) and eccentricity (ecc)
    Based on "An Improved Algorithm due to Laguerre for the Solution of Kepler's Equation"
    Conway, B. A.  (1986) Celestial Mechanics, Volume 39, Issue 2, pp.199-211  1986CeMec..39..199C
Julia function originally written by Eric Ford
Basically looks for root of an nth order polynomial assuming that 1 root is
the one you are looking for (E) and the rest are evenly spaced
"""
function update_ecc_anom_laguerre(E::Real, M::Real, e::Real)
    # es = e * sin(E)
    # ec = e * cos(E)
    F = (E - e * sin(E)) - M
    Fp = one(E) - e * cos(E)
    # Fpp = es
    n = 5
    root = sqrt(abs((n - 1) * ((n - 1) * Fp * Fp - n * F * e * sin(E))))
    denom = Fp > zero(E) ? Fp + root : Fp - root
    return E - n * F / denom
end


"""
Loop to update the current estimate of the solution to Kepler's equation
Julia function originally written by Eric Ford
"""
function ecc_anomaly(t::Real, P::Real, e::Real, M0::Real; tol::Real=1e-8, max_its::Integer=200)
    @assert (0 <= e <= 1) "eccentricity has to be between 0 and 1"
    M = mean_anomaly(t, P, M0)
    E = ecc_anom_init_guess_danby(M, e)
    for i in 1:max_its
       E_old = E
       E = update_ecc_anom_laguerre(E_old, M, e)
       if abs(E - E_old) < tol; break end
    end
    return E
end


# """
# Iterative solution for eccentric anomaly from mean anomaly
# from (http://www.csun.edu/~hcmth017/master/node16.html)
# could also implement third order method from
# (http://alpheratz.net/dynamics/twobody/KeplerIterations_summary.pdf)
# """
# function ecc_anomaly(t::Real, P::Real, e::Real)
#     @assert (0 <= e <= 1) "eccentricity has to be between 0 and 1"
#     M = mean_anomaly(t, P)
#     dif_thres = 1e-8
#     dif = 1
#     ecc_anom = copy(M)
#     while dif > dif_thres
#         ecc_anom_old = copy(ecc_anom)
#         ecc_anom -= (ecc_anom - e * sin(ecc_anom) - M) / (1 - e * cos(ecc_anom))  # first order method
#         dif = abs((ecc_anom - ecc_anom_old) / ecc_anom)
#     end
#     return ecc_anom
# end


ecc_anomaly(t::Real, P::Quantity, e::Real, M0::Real; tol::Real=1e-8, max_its::Integer=200) = ecc_anomaly(t, convert_and_strip_units(u"yr", P), e, M0; tol=tol, max_its=max_its)


"""
Solution for true anomaly from eccentric anomaly
from (https://en.wikipedia.org/wiki/True_anomaly#From_the_eccentric_anomaly)
"""
ϕ(t::Real, P::Real, e::Real, M0::Real) = 2 * atan(sqrt((1 + e) / (1 - e)) * tan(ecc_anomaly(t, P, e, M0) / 2))

ϕ(t::Real, P::Quantity, e::Real, M0::Real) = ϕ(t, convert_and_strip_units(u"yr", P), e, M0)


# """
# Calculate true anomaly for small e using equation of center approximating true
# anomaly from (https://en.wikipedia.org/wiki/Equation_of_the_center) O(e^8)
# """
# function ϕ_approx(t::Real, P::Real, e::Real)
#     @assert (0 <= e <= 1) "eccentricity has to be between 0 and 1"
#     M = mean_anomaly(t, P)
#
#     term_list = [eval_polynomial(e, [0, 2, 0, - 1/4, 0, 5/96, 0, 107/4608]),
#     eval_polynomial(e, [0, 0, 5/4, 0, -11/24, 0, 17/192, 0]),
#     eval_polynomial(e, [0, 0, 0, 13/12, 0, -43/64, 0, 95/512]),
#     eval_polynomial(e, [0, 0, 0, 0, 103/96, 0, -451/480, 0]),
#     eval_polynomial(e, [0, 0, 0, 0, 0, 1097/960, 0, -5957/4608]),
#     eval_polynomial(e, [0, 0, 0, 0, 0, 0, 1223/960, 0]),
#     eval_polynomial(e, [0, 0, 0, 0, 0, 0, 0, 47273/32256])]
#
#     sin_list = [sin(i * M) for i in 1:7]
#
#     return M + dot(term_list, sin_list)
#
# end
#
# ϕ_approx(t::Real, P::Quantity, e::Real) = ϕ_approx(t, convert_and_strip_units(u"yr", P), e)


"""
kepler_rv(t, P, e, M0, K, ω; γ)
Calculate radial velocity of star due to planet following Keplerian orbit at specified time.
Inputs:
- time
- Period
- eccentricity
- mean anomaly at time=0
- RV amplitude
- argument of periapsis
Optionally:
- velocity offset
"""
function kepler_rv(t::Real, P::Real, e::Real, M0::Real, K::Real, ω::Real; γ::Real=0.)
    assert_positive(P)
    k = e * cos(ω)
    return kepler_rv_andras(t, P, e, M0, K, k, ω; γ=γ)
end


"adapted from eq. 11 of (http://exoplanets.astro.yale.edu/workshop/EPRV/Bibliography_files/Radial_Velocity.pdf)"
kepler_rv_true_anomaly(t::Real, P::Real, e::Real, M0::Real, K::Real, ω::Real; γ::Real=0.) = K * (e * cos(ω) + cos(ω + ϕ(t, P, e, M0))) + γ


"""
Radial velocity formula directly using eccentric anomaly
Based on "An analytical solution for Kepler's problem"
Pál, András, Monthly Notices of the Royal Astronomical Society, 396, 3, 1737-1742.  2009MNRAS.396.1737P
see ηdot part of eq. 19
"""
function kepler_rv_andras(t::Real, P::Real, e::Real, M0::Real, K::Real, k::Real, ω::Real; γ::Real=0.)
    E = ecc_anomaly(t, P, e, M0)
    j = sqrt(1 - e*e)
    q = e * cos(E)
    # return K * j / (1 - q) * (cos(ω + E) - (1 - j) * cos(E) * cos(ω))  # equivalent
    return K * j / (1 - q) * (cos(ω + E) - k * q / (1 + j)) + γ
end


"Convert Unitful things to proper units for RV calculations"
function kepler_rv(t::Union{Real, Quantity}, P::Union{Real, Quantity}, e::Real, M0::Real, m_star::Union{Real, Quantity}, m_planet::Union{Real, Quantity}, ω::Real; i::Real=π/2, γ::Real=0.)
    m_star = convert_and_strip_units(u"Msun", m_star)
    m_planet = convert_and_strip_units(u"Msun", m_planet)
    P = convert_and_strip_units(u"yr", P)
    assert_positive(m_star, m_planet)
    t = convert_and_strip_units(u"yr", t)
    K = velocity_semi_amplitude(P, m_star, m_planet, e=e, i=i)
    return kepler_rv(t, P, e, M0, K, ω; γ=γ)
end


"""
This differs from usual expressions so as to be differentiable, even at zero eccentricity.
Done by replacing e and ω, with h and k
h = e * sin(ω)
k = e * cos(ω)
"""
function kepler_rv_hk(t::Real, P::Real, M0::Real, K::Real, h::Real, k::Real; γ::Real=0.)
    e = sqrt(h * h + k * k)
    ω = atan(h, k)
    return kepler_rv_andras(t, P, e, M0, K, k, ω; γ=γ)
end


"""
A circular formulation of a Keplerian RV signal i.e. e-> 0 so ϕ -> M
for initial full orbit fits
coefficients[1] = K * cos(M0 - ω)
coefficients[2] = K * sin(M0 - ω)
coefficients[3] = γ
so
K = sqrt(coefficients[1]^2 + coefficients[2]^2)
M0 - ω = atan(coefficients[2], coefficients[1])
"""
function kepler_rv_circ(t, P::Union{Real, Quantity}, coefficients::AbstractArray{T,1}) where {T<:Real}
    @assert length(coefficients) == 3 "wrong number of coefficients"
    P = convert_and_strip_units(u"yr", P)
    assert_positive(P)
    t = convert_and_strip_units.(u"yr", t)
    phase = unit_phase.(t, P)
    return (coefficients[1] .* cos.(phase)) + (coefficients[2] .* sin.(phase)) + (coefficients[3] .* ones(length(t)))
end


"""
A linear formulation of a Keplerian RV signal for low eccentricity by expanding about e=0
average difference is < 1% for e < 0.1 and < 10% for e < 0.35
for initial full orbit fits
coefficients[1] = K * cos(M0 - ω)
coefficients[2] = K * sin(M0 - ω)
coefficients[3] = e * K * cos(2 * M0 - ω)
coefficients[4] = e * K * sin(2 * M0 - ω)
coefficients[5] = γ
so
K = sqrt(coefficients[1]^2 + coefficients[2]^2)
e = sqrt(coefficients[3]^2 + coefficients[4]^2) / K
M0 = atan(coefficients[4], coefficients[3]) - atan(coefficients[2], coefficients[1])
ω = atan(coefficients[4], coefficients[3]) - 2 * atan(coefficients[2], coefficients[1])
"""
function kepler_rv_linear_e(t, P::Union{Real, Quantity}, coefficients::AbstractArray{T,1}) where {T<:Real}
    @assert length(coefficients) == 5 "wrong number of coefficients"
    P = convert_and_strip_units(u"yr", P)
    assert_positive(P)
    t = convert_and_strip_units.(u"yr", t)
    phase = unit_phase.(t, P)
    return (coefficients[1] .* cos.(phase)) + (coefficients[2] .* sin.(phase)) + (coefficients[3] .* cos.(2 .* phase)) + (coefficients[4] .* sin.(2 .* phase)) + (coefficients[5] .* ones(length(t)))
end


"""
A linear formulation of a Keplerian RV signal for a given eccentricity, period and initial mean anomaly
for fitting non-(e, P, M0) components
see Wright and Howard 2009 "EFFICIENT FITTING OF MULTI-PLANET KEPLERIAN MODELS TO RADIAL VELOCITY AND ASTROMETRY DATA" (https://arxiv.org/abs/0904.3725)
coefficients[1] = K * cos(ω)
coefficients[2] = -K * sin(ω)
coefficients[3] = γ + K * e * cos(ω)
so
K = sqrt(coefficients[1]^2 + coefficients[2]^2)
ω = atan(-coefficients[2], coefficients[1])
γ = coefficients[3] - K * e * cos(ω)
"""
function kepler_rv_linear_gen(t, P::Union{Real, Quantity}, coefficients::AbstractArray{T,1}) where {T<:Real}
    @assert length(coefficients) == 3 "wrong number of coefficients"
    P = convert_and_strip_units(u"yr", P)
    assert_positive(P)
    t = convert_and_strip_units.(u"yr", t)
    ϕ_t = ϕ.(t, P, e, M0)
    return (coefficients[1] .* cos.(ϕ_t)) + (coefficients[2] .* sin.(ϕ_t)) + (coefficients[3] .* ones(length(t)))
end


"Convert the solar phase information from SOAP 2.0 into days"
function convert_SOAP_phases_to_days(phase::Real; P_rot=25.05)
    # default P_rot is the solar rotation period used by SOAP 2.0 in days
    assert_positive(P_rot)
    return phase * P_rot
end

"Convert the solar phase information from SOAP 2.0 into years"
function convert_SOAP_phases_to_years(phase::Real; P_rot = 25.05)
    return convert_and_strip_units(u"yr", convert_SOAP_phases_to_days(phase; P_rot=P_rot)u"d")
end


"Remove the best-fit circular Keplerian signal from the data"
function remove_kepler!(data::AbstractArray{T1,1}, times::AbstractArray{T2,1}, P::Real, covariance::Union{Cholesky{T3,Array{T3,2}},Symmetric{T4,Array{T4,2}},AbstractArray{T5}}) where {T1<:Real, T2<:Real, T3<:Real, T4<:Real, T5<:Real}
    assert_positive(P)
    for i in 1:ndims(covariance)
        @assert size(covariance,i)==length(data) "covariance incompatible with data"
    end
    amount_of_total_samp_points = length(data)
    amount_of_samp_points = length(times)
    kepler_rv_linear_terms = hcat(cos.(unit_phase.(times, P)), sin.(unit_phase.(times, P)), ones(length(times)))
    amount_of_total_samp_points > amount_of_samp_points ? kepler_linear_terms = vcat(kepler_rv_linear_terms, zeros(amount_of_total_samp_points - amount_of_samp_points, 3)) : kepler_linear_terms = kepler_rv_linear_terms
    x = general_lst_sq(kepler_linear_terms, data; Σ=covariance)
    data[1:amount_of_samp_points] -= kepler_rv_circ(times, P, x)
end

function remove_kepler(y_obs_w_planet::AbstractArray{T1,1}, times::AbstractArray{T2,1}, P::Real, covariance::Union{Cholesky{T3,Array{T3,2}},Symmetric{T4,Array{T4,2}},AbstractArray{T5}}) where {T1<:Real, T2<:Real, T3<:Real, T4<:Real, T5<:Real}
    y_obs_wo_planet = copy(y_obs_w_planet)
    remove_kepler!(y_obs_wo_planet, times, P, covariance)
    return y_obs_wo_planet
end


"""
Evaluate the likelihood function with data after removing the best-fit circular
Keplerian orbit for given period.

    kep_signal_likelihood(likelihood_func::Function, period::Real, times_obs::AbstractArray{T2,1}, signal_data::AbstractArray{T3,1}, covariance::Union{Cholesky{T4,Array{T4,2}},Symmetric{T5,Array{T5,2}},AbstractArray{T6}})

likelihood_func is a wrapper function handle that returns the likelihood given a single input of the data without the best-fit Kperleian signal
period is the orbital period that you want to attempt to remove
times_obs are the times of the measurements
signal_data is your data including the planetary signal
covariance is either the covariance matrix relating all of your data points, or a vector of noise measuremnets
"""
function kep_signal_likelihood(likelihood_func::Function, period::Real, times_obs::AbstractArray{T2,1}, signal_data::AbstractArray{T3,1}, covariance::Union{Cholesky{T4,Array{T4,2}},Symmetric{T5,Array{T5,2}},AbstractArray{T6}}) where {T2<:Real, T3<:Real, T4<:Real, T5<:Real, T6<:Real}
    return likelihood_func(remove_kepler(signal_data, times_obs, period, covariance))
end

"""
Evaluate the likelihood function with data after removing the best-fit circular
Keplerian orbit for many periods without unnecessary memory reallocations
"""
function kep_signal_likelihoods(likelihood_func::Function, period_grid::AbstractArray{T1,1}, times_obs::AbstractArray{T2,1}, signal_data::AbstractArray{T3,1}, covariance::Union{Cholesky{T4,Array{T4,2}},Symmetric{T5,Array{T5,2}},AbstractArray{T6}}) where {T1<:Real, T2<:Real, T3<:Real, T4<:Real, T5<:Real, T6<:Real}
    # @warn "make sure that period_grid and time_obs are in the same units!"
    likelihoods = zeros(length(period_grid))
    new_data = zeros(length(signal_data))
    for i in 1:length(period_grid)
        new_data .= signal_data
        remove_kepler!(new_data, times_obs, period_grid[i], covariance)
        likelihoods[i] = likelihood_func(new_data)
    end
    return likelihoods
end
