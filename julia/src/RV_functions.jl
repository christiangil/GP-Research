# these functions are related to calculating RV quantities
using UnitfulAstro
using Unitful
# using UnitfulAngles
using LinearAlgebra

"Strip Unitful units"
strip_units(quant::Quantity) = ustrip(float(quant))


"Convert Unitful units from one to another and strip the final units"
convert_and_strip_units(new_unit::Unitful.FreeUnits, quant::Quantity) = strip_units(uconvert(new_unit, quant))
convert_and_strip_units(new_unit::Unitful.FreeUnits, quant::Unitful.Time) = quant


"""
Calculate velocity semi-amplitude for RV signal of planets in m/s
adapted from eq. 12 of (http://exoplanets.astro.yale.edu/workshop/EPRV/Bibliography_files/Radial_Velocity.pdf)
"""
function velocity_semi_amplitude(
    m_star::Unitful.Mass,
    m_planet::Unitful.Mass,
    P::Unitful.Time;
    e::Real=0.,
    i::Real=π/2)

    # m_star = convert_and_strip_units(u"Msun", m_star)
    # m_planet = convert_and_strip_units(u"Msun", m_planet)
    # P = convert_and_strip_units(u"yr", P)
    assert_positive(P, m_star, m_planet)
    comb_mass = m_star + m_planet
    K_au_yr = (2 * π * sin(i) * m_planet / (sqrt(1 - (e * e)) * cbrt(comb_mass * comb_mass * P)))u"AU/yr"
    return uconvert(u"m/s", K_au_yr)
end


# unit_phase(t::Unitful.Time, P::Unitful.Time) = mod(t / P * 2 * π, 2 * π)
unit_phase(t::Unitful.Time, P::Unitful.Time) = t / P * 2 * π


"calculate mean anomaly"
mean_anomaly(t::Unitful.Time, P::Unitful.Time, M0::Real) = unit_phase(t, P) - M0
# mean_anomaly(t::Unitful.Time, P::Unitful.Time, M0::Real) = mod(unit_phase(t, P) - M0, 2 * π)
# mean_anomaly(t::Unitful.Time, P::Unitful.Time, M0::Real) = mod(t / P * 2 * π - M0, 2 * π)


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
function ecc_anomaly(
    t::Unitful.Time,
    P::Unitful.Time,
    M0::Number,
    e::Number;
    tol::Number=1e-8,
    max_its::Integer=200)

    # P = convert_and_strip_units(u"yr", P)
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
# function ecc_anomaly(t::Unitful.Time, P::Unitful.Time, e::Real)
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


"""
Solution for true anomaly from eccentric anomaly
from (https://en.wikipedia.org/wiki/True_anomaly#From_the_eccentric_anomaly)
"""
ϕ(
    t::Unitful.Time,
    P::Unitful.Time,
    M0::Real,
    e::Real
    ) = 2 * atan(sqrt((1 + e) / (1 - e)) * tan(ecc_anomaly(t, P, M0, e) / 2))


# """
# Calculate true anomaly for small e using equation of center approximating true
# anomaly from (https://en.wikipedia.org/wiki/Equation_of_the_center) O(e^8)
# """
# function ϕ_approx(t::Unitful.Time, P::Unitful.Time, e::Real)
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
# ϕ_approx(t::Unitful.Time, P::Unitful.Time, e::Real) = ϕ_approx(t, convert_and_strip_units(u"yr", P), e)


"""
kepler_rv(t, K, P, M0, e, ω; γ)
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
function kepler_rv(
    t::Unitful.Time,
    K::Unitful.Velocity,
    P::Unitful.Time,
    M0::Real,
    e::Real,
    ω::Real;
    γ::Unitful.Velocity=0.)

    # P = convert_and_strip_units(u"yr", P)
    # t = convert_and_strip_units(u"yr", t)
    # K = convert_and_strip_units(u"m/s", K)
    # γ = convert_and_strip_units(u"m/s", γ)
    # assert_positive(P)
    return kepler_rv_ecc_anom(t, K, P, M0, e, ω; γ=γ)
    # return kepler_rv_true_anom(t, P, M0, e, K, ω; γ=γ)
end


function kepler_rv(
    t::Unitful.Time,
    m_star::Unitful.Mass,
    m_planet::Unitful.Mass,
    P::Unitful.Time,
    M0::Real,
    e::Real,
    ω::Real;
    i::Real=π/2,
    γ::Unitful.Velocity=0.)

    K = velocity_semi_amplitude(P, m_star, m_planet, e=e, i=i)
    return kepler_rv(t, K, P, M0, e, ω; γ=γ)
end


struct kep_signal
    K::Unitful.Velocity
    P::Unitful.Time
    M0::Real  # ::Unitful.NoDims
    e::Real
    ω::Real  # ::Unitful.NoDims

    γ::Unitful.Velocity

    h::Real  # ::Unitful.NoDims
    k::Real  # ::Unitful.NoDims

    kep_signal(K, P, M0) = kep_signal(K, P, M0, 0, 0)
    kep_signal(K, P, M0, e, ω) = kep_signal(K, P, M0, e, ω, 0u"m/s")
    kep_signal(K, P, M0, e, ω, γ) = kep_signal(K, P, M0, e, ω, γ, e*cos(ω), e*sin(ω))
    function kep_signal(
        K::Unitful.Velocity,
        P::Unitful.Time,
        M0::Real,
        e::Real,
        ω::Real,
        γ::Unitful.Velocity,
        h::Real,
        k::Real)

        @assert 0 <= e < 1 "orbit needs to be bound"
        M0 = mod2pi(M0)
        ω = mod2pi(ω)
        assert_positive(P)
        return new(K, P, M0, e, ω, γ, h, k)
    end
end
kepler_rv(t::Unitful.Time, ks::kep_signal) = kepler_rv(t, ks.K, ks.P, ks.M0, ks.e, ks.ω; γ=ks.γ)
(ks::kep_signal)(t::Unitful.Time) = kepler_rv(t, ks)

"""
Simple radial velocity formula using true anomaly
adapted from eq. 11 of (http://exoplanets.astro.yale.edu/workshop/EPRV/Bibliography_files/Radial_Velocity.pdf)
"""
kepler_rv_true_anom(
    t::Unitful.Time,
    K::Unitful.Velocity,
    P::Unitful.Time,
    M0::Real,
    e::Real,
    ω::Real;
    γ::Unitful.Velocity=0.
    ) = K * (e * cos(ω) + cos(ω + ϕ(t, P, M0, e))) + γ


"""
Radial velocity formula directly using eccentric anomaly
Based on simplification of "An analytical solution for Kepler's problem"
Pál, András, Monthly Notices of the Royal Astronomical Society, 396, 3, 1737-1742.  2009MNRAS.396.1737P
see ηdot part of eq. 19
"""
function kepler_rv_ecc_anom(
    t::Unitful.Time,
    K::Unitful.Velocity,
    P::Unitful.Time,
    M0::Real,
    e::Real,
    ω::Real;
    γ::Unitful.Velocity=0.)

    k = e * cos(ω)
    E = ecc_anomaly(t, P, M0, e)
    j = sqrt(1 - e*e)
    q = e * cos(E)
    # return K * j / (1 - q) * (cos(ω + E) - (1 - j) * cos(E) * cos(ω)) + γ  # equivalent
    return K * j / (1 - q) * (cos(ω + E) - k * q / (1 + j)) + γ
end


# """
# Replacing e and ω, with h and k
# A uniform prior on h and k (on the unit circle) leads to a uniform prior on ω
# and a linearly increasing prior on e
# he = e * sin(ω)
# ke = e * cos(ω)
# so
# e = sqrt(h^2 + k^2)
# ω = atan(h, k)
# sin(ω) = h / e
# cos(ω) = k / e
# """
# function kepler_rv_hk1(
#     t::Unitful.Time,
#     K::Unitful.Velocity,
#     P::Unitful.Time,
#     M0::Real,
#     h::Real,
#     k::Real;
#     γ::Unitful.Velocity=0.)
#
#     e_sq = h * h + k * k
#     e = sqrt(e_sq)
#     E = ecc_anomaly(t, P, M0, e)
#     cosE = cos(E)
#     j = sqrt(1 - e_sq)
#
#     return K * j / (e - e_sq * cosE) * (j * k * cosE - h * sin(E)) + γ
# end
#
#
# """
# Replacing e and ω, with h and k
# A uniform prior on h and k (on the unit circle) leads to a uniform prior on e
# and ω
# hp = sqrt(e) * sin(ω)
# kp = sqrt(e) * cos(ω)
# so
# e = hp^2 + kp^2
# ω = atan(h, k)
# sin(ω) = hp / sqrt(e)
# cos(ω) = kp / sqrt(e)
# """
# function kepler_rv_hk2(
#     t::Unitful.Time,
#     K::Unitful.Velocity,
#     P::Unitful.Time,
#     M0::Real,
#     hp::Real,
#     kp::Real;
#     γ::Unitful.Velocity=0.)
#
#     e = hp * hp + kp * kp
#     E = ecc_anomaly(t, P, M0, e)
#     cosE = cos(E)
#     j = sqrt(1 - e * e)
#
#     return K * j / (sqrt(e) * (1 - e * cosE)) * (j * kp * cosE - hp * sin(E)) + γ
# end


# """
# A circular formulation of a Keplerian RV signal i.e. e-> 0 so ϕ -> M
# for initial full orbit fits
# coefficients[1] = K * cos(M0 - ω)
# coefficients[2] = K * sin(M0 - ω)
# coefficients[3] = γ
# so
# K = sqrt(coefficients[1]^2 + coefficients[2]^2)
# M0 - ω = atan(coefficients[2], coefficients[1])
# """
# function kepler_rv_circ(
#     t::Unitful.Time,
#     P::Unitful.Time,
#     coefficients::Vector{Real}
#     ) where
#
#     @assert length(coefficients) == 3 "wrong number of coefficients"
#     # P = convert_and_strip_units(u"yr", P)
#     # assert_positive(P)
#     # t = convert_and_strip_units.(u"yr", t)
#     phase = unit_phase.(t, P)
#     return (coefficients[1] .* cos.(phase)) + (coefficients[2] .* sin.(phase)) + (coefficients[3] .* ones(length(t)))
# end

# function kepler_rv_circ_orbit_params(
#     coefficients::Vector{Real};
#     print_params::Bool=false
#     ) where
#
#     @assert length(coefficients) == 3 "wrong number of coefficients"
#     K = sqrt(coefficients[1]^2 + coefficients[2]^2)
#     M0minusω = mod2pi(atan(coefficients[2], coefficients[1]))
#     γ = coefficients[3]
#     if print_params; println("K: $K, M0-ω: $M0minusω, γ: $γ") end
#     return K, M0minusω, γ
# end


struct kep_signal_circ
    K::Unitful.Velocity
    P::Unitful.Time
    M0minusω::Real  # ::Unitful.NoDims
    γ::Unitful.Velocity
    coefficients::Vector{T} where T<:Unitful.Velocity

    kep_signal_circ(K::Unitful.Velocity, P::Unitful.Time) = kep_signal_circ(K, P, 0)
    kep_signal_circ(K, P, M0minusω) = kep_signal_circ(K, P, M0minusω, 0u"m/s")
    kep_signal_circ(K, P, M0minusω, γ) = kep_signal_circ(K, P, M0minusω, γ, [K * cos(M0minusω), K * sin(M0minusω), γ])
    function kep_signal_circ(
        P::Unitful.Time,
        coefficients::Vector{T} where T<:Unitful.Velocity)

        return kep_signal_circ(
            sqrt(coefficients[1]^2 + coefficients[2]^2),
            P,
            mod2pi(atan(coefficients[2], coefficients[1])),
            coefficients[3],
            coefficients)
    end
    function kep_signal_circ(
        K::Unitful.Velocity,
        P::Unitful.Time,
        M0minusω::Real,  # ::Unitful.NoDims
        γ::Unitful.Velocity,
        coefficients::Vector{T} where T<:Unitful.Velocity)

        M0minusω = mod2pi(M0minusω)
        assert_positive(P)
        @assert length(coefficients) == 3 "wrong number of coefficients"
        return new(K, P, M0minusω, γ, coefficients)
    end
end
function kepler_rv_circ(t::Unitful.Time, ks::kep_signal_circ)
    phase = unit_phase.(t, ks.P)
    return (ks.coefficients[1] .* cos.(phase)) + (ks.coefficients[2] .* sin.(phase)) .+ ks.coefficients[3]
end
kepler_rv(t::Unitful.Time, ks::kep_signal_circ) = kepler_rv_circ(t, ks)
(ks::kep_signal_circ)(t::Unitful.Time) = kepler_rv(t, ks)

# """
# A linear formulation of a Keplerian RV signal for low eccentricity by expanding about e=0
# average difference is < 1% for e < 0.1 and < 10% for e < 0.35
# for initial full orbit fits
# coefficients[1] = K * cos(M0 - ω)
# coefficients[2] = K * sin(M0 - ω)
# coefficients[3] = e * K * cos(2 * M0 - ω)
# coefficients[4] = e * K * sin(2 * M0 - ω)
# coefficients[5] = γ
# so
# K = sqrt(coefficients[1]^2 + coefficients[2]^2)
# e = sqrt(coefficients[3]^2 + coefficients[4]^2) / K
# M0 = atan(coefficients[4], coefficients[3]) - atan(coefficients[2], coefficients[1])
# ω = atan(coefficients[4], coefficients[3]) - 2 * atan(coefficients[2], coefficients[1])
# """
# function kepler_rv_epicyclic(
#     t::Unitful.Time,
#     P::Unitful.Time,
#     coefficients::Vector{Real}
#     ) where
#
#     @assert length(coefficients) == 5 "wrong number of coefficients"
#     P = convert_and_strip_units(u"yr", P)
#     assert_positive(P)
#     t = convert_and_strip_units.(u"yr", t)
#     phase = unit_phase.(t, P)
#     return (coefficients[1] .* cos.(phase)) + (coefficients[2] .* sin.(phase)) + (coefficients[3] .* cos.(2 .* phase)) + (coefficients[4] .* sin.(2 .* phase)) + (coefficients[5] .* ones(length(t)))
# end

# function kepler_rv_epicyclic_orbit_params(
#     coefficients::Vector{Real};
#     print_params::Bool=false
#     ) where
#
#     @assert length(coefficients) == 5 "wrong number of coefficients"
#     K = sqrt(coefficients[1]^2 + coefficients[2]^2)
#     e = sqrt(coefficients[3]^2 + coefficients[4]^2) / K
#     if e > 0.35; @warn "an orbit with this eccentricity would be very different from the output of kepler_rv_epicyclic" end
#     M0 = mod2pi(atan(coefficients[4], coefficients[3]) - atan(coefficients[2], coefficients[1]))
#     ω = mod2pi(atan(coefficients[4], coefficients[3]) - 2 * atan(coefficients[2], coefficients[1]))
#     γ = coefficients[5]
#     if print_params; println("K: $K, e: $e, M0: $M0, ω: $ω, γ: $γ") end
#     return K, e, M0, ω, γ
# end


struct kep_signal_epicyclic
    K::Unitful.Velocity
    P::Unitful.Time
    M0::Real
    e::Real
    ω::Real  # ::Unitful.NoDims
    γ::Unitful.Velocity
    coefficients::Vector{T} where T<:Unitful.Velocity

    kep_signal_epicyclic(K, P, M0) = kep_signal_epicyclic(K, P, M0, 0, 0)
    kep_signal_epicyclic(K, P, M0, e, ω) = kep_signal_epicyclic(K, P, M0, e, ω, 0u"m/s")
    kep_signal_epicyclic(K, P, M0, e, ω, γ) = kep_signal_epicyclic(K, P, M0, e, ω, γ,
        [K * cos(M0 - ω), K * sin(M0 - ω), e * K * cos(2 * M0 - ω), e * K * sin(2 * M0 - ω), γ])
    function kep_signal_epicyclic(P, coefficients)
        K = sqrt(coefficients[1]^2 + coefficients[2]^2)
        e = sqrt(coefficients[3]^2 + coefficients[4]^2) / K
        if e > 0.35; @warn "an orbit with this eccentricity would be very different from the output of kepler_rv_epicyclic" end
        M0 = mod2pi(atan(coefficients[4], coefficients[3]) - atan(coefficients[2], coefficients[1]))
        ω = mod2pi(atan(coefficients[4], coefficients[3]) - 2 * atan(coefficients[2], coefficients[1]))
        γ = coefficients[5]
        return kep_signal_epicyclic(K, P, M0, e, ω, γ, coefficients)
    end
    function kep_signal_epicyclic(
        K::Unitful.Velocity,
        P::Unitful.Time,
        M0::Real,
        e::Real,
        ω::Real,  # ::Unitful.NoDims
        γ::Unitful.Velocity,
        coefficients::Vector{T} where T<:Unitful.Velocity)

        M0 = mod2pi(M0)
        ω = mod2pi(ω)
        assert_positive(P)
        @assert length(coefficients) == 5 "wrong number of coefficients"
        return new(K, P, M0, e, ω, γ, coefficients)
    end
end
function kepler_rv_epicyclic(t::Unitful.Time, ks::kep_signal_epicyclic)
    phase = unit_phase.(t, ks.P)
    return return (ks.coefficients[1] .* cos.(phase)) + (ks.coefficients[2] .* sin.(phase)) + (ks.coefficients[3] .* cos.(2 .* phase)) + (ks.coefficients[4] .* sin.(2 .* phase)) .+ ks.coefficients[5]
end
kepler_rv(t::Unitful.Time, ks::kep_signal_epicyclic) = kepler_rv_epicyclic(t, ks)
(ks::kep_signal_epicyclic)(t::Unitful.Time) = kepler_rv(t, ks)

# """
# A linear formulation of a Keplerian RV signal for a given eccentricity, period and initial mean anomaly
# for fitting non-(e, P, M0) components
# see Wright and Howard 2009 "EFFICIENT FITTING OF MULTI-PLANET KEPLERIAN MODELS TO RADIAL VELOCITY AND ASTROMETRY DATA" (https://arxiv.org/abs/0904.3725)
# coefficients[1] = K * cos(ω)
# coefficients[2] = -K * sin(ω)
# coefficients[3] = γ + K * e * cos(ω)
# so
# K = sqrt(coefficients[1]^2 + coefficients[2]^2)
# ω = atan(-coefficients[2], coefficients[1])
# γ = coefficients[3] - K * e * cos(ω)
# """
# function kepler_rv_wright(
#     t::Unitful.Time,
#     P::Unitful.Time,
#     M0::Real,
#     e::Real,
#     coefficients::Vector{Real}
#     ) where
#
#     @assert length(coefficients) == 3 "wrong number of coefficients"
#     P = convert_and_strip_units(u"yr", P)
#     assert_positive(P)
#     t = convert_and_strip_units.(u"yr", t)
#     ϕ_t = ϕ.(t, P, M0, e)
#     return (coefficients[1] .* cos.(ϕ_t)) + (coefficients[2] .* sin.(ϕ_t)) + (coefficients[3] .* ones(length(t)))
# end

# function kepler_rv_wright_orbit_params(
#     coefficients::Vector{Real},
#     e::Real,
#     print_params::Bool=false
#     ) where
#
#     @assert length(coefficients) == 3 "wrong number of coefficients"
#     K = sqrt(coefficients[1]^2 + coefficients[2]^2)
#     ω = mod2pi(atan(-coefficients[2], coefficients[1]))
#     γ = coefficients[3] - K * e * cos(ω)
#     if print_params; println("K: $K, ω: $ω, γ: $γ") end
#     return K, ω, γ
# end


struct kep_signal_wright
    K::Unitful.Velocity
    P::Unitful.Time
    M0::Real
    e::Real
    ω::Real  # ::Unitful.NoDims
    γ::Unitful.Velocity
    coefficients::Vector{T} where T<:Unitful.Velocity

    kep_signal_wright(K, P, M0) = kep_signal_wright(K, P, M0, 0, 0)
    kep_signal_wright(K, P, M0, e, ω) = kep_signal_wright(K, P, M0, e, ω, 0u"m/s")
    kep_signal_wright(K, P, M0, e, ω, γ) = kep_signal_wright(K, P, M0, e, ω, γ,
        [K * cos(ω), -K * sin(ω), γ + K * e * cos(ω)])
    function kep_signal_wright(P, M0, e, coefficients)
        K = sqrt(coefficients[1]^2 + coefficients[2]^2)
        ω = mod2pi(atan(-coefficients[2], coefficients[1]))
        γ = coefficients[3] - K * e * cos(ω)
        return kep_signal_wright(K, P, M0, e, ω, γ, coefficients)
    end
    function kep_signal_wright(
        K::Unitful.Velocity,
        P::Unitful.Time,
        M0::Real,
        e::Real,
        ω::Real,  # ::Unitful.NoDims
        γ::Unitful.Velocity,
        coefficients::Vector{T} where T<:Unitful.Velocity)

        M0 = mod2pi(M0)
        ω = mod2pi(ω)
        assert_positive(P)
        @assert length(coefficients) == 3 "wrong number of coefficients"
        return new(K, P, M0, e, ω, γ, coefficients)
    end
end
function kepler_rv_wright(t::Unitful.Time, ks::kep_signal_wright)
    ϕ_t = ϕ.(t, ks.P, ks.M0, ks.e)
    return return (ks.coefficients[1] .* cos.(ϕ_t)) + (ks.coefficients[2] .* sin.(ϕ_t)) + (ks.coefficients[3] .* ones(length(t)))
end
kepler_rv(t::Unitful.Time, ks::kep_signal_wright) = kepler_rv_wright(t, ks)
(ks::kep_signal_wright)(t::Unitful.Time) = kepler_rv(t, ks)


"Convert the solar phase information from SOAP 2.0 into days"
function convert_SOAP_phases(new_unit::Unitful.FreeUnits, phase::Real; P_rot::Unitful.Time=25.05u"d")
    # default P_rot is the solar rotation period used by SOAP 2.0 in days
    assert_positive(P_rot)
    return uconvert(new_unit, phase * P_rot)
end

# "Convert the solar phase information from SOAP 2.0 into days"
# function convert_SOAP_phases_to_days(phase::Real; P_rot::Unitful.Time=25.05u"d")
#     # default P_rot is the solar rotation period used by SOAP 2.0 in days
#     assert_positive(P_rot)
#     return phase * P_rot
# end
#
# "Convert the solar phase information from SOAP 2.0 into years"
# function convert_SOAP_phases_to_years(phase::Real; P_rot::Unitful.Time=25.05u"d")
#     return convert_and_strip_units(u"yr", convert_SOAP_phases_to_days(phase; P_rot=P_rot)u"d")
# end

function fit_kepler_epicyclic(
    data::Vector{Real},
    times::Vector{T2} where T2<:Unitful.Time,
    P::Unitful.Time,
    covariance::Union{Cholesky{T,Matrix{T}},Symmetric{T,Matrix{T}},Matrix{T},Vector{T}}
    ) where T<:Real

    # if P==0; return kep_signal_epicyclic(p, P, M0, e, ω, γ, coefficients) end
    assert_positive(P)
    for i in 1:ndims(covariance)
        @assert size(covariance,i)==length(data) "covariance incompatible with data"
    end
    amount_of_total_samp_points = length(data)
    amount_of_samp_points = length(times)
    phases = unit_phase.(times, P)
    kepler_epicyclic_terms = hcat(cos.(phases), sin.(phases), cos.(2 .* phases), sin.(2 .* phases), ones(length(times)))
    amount_of_total_samp_points > amount_of_samp_points ? kepler_epicyclic_terms = vcat(kepler_rv_linear_terms, zeros(amount_of_total_samp_points - amount_of_samp_points, size(kepler_rv_linear_terms, 2))) : kepler_epicyclic_terms = kepler_rv_linear_terms
    x = general_lst_sq(kepler_epicyclic_terms, data; Σ=covariance)
    return kep_signal_epicyclic(P, x)
end


# "Remove the best-fit epicyclic (linearized in e) Keplerian signal from the data"
# function remove_kepler!(
#     data::Vector{Real},
#     times::Vector{T2} where T2<:Unitful.Time,
#     P::Unitful.Time,
#     covariance::Union{Cholesky{T,Matrix{T}},Symmetric{T,Matrix{T}},Matrix{T},Vector{Real}};
#     return_params::Bool=false
#     ) where
#
#     if return_params
#         orbit_fit, K, e, M0, ω, γ = fit_kepler_epicyclic(data, times, P, covariance; return_params=return_params)
#         data[1:length(times)] -= orbit_fit
#         return K, e, M0, ω, γ
#     else
#         data[1:length(times)] -= fit_kepler_epicyclic(data, times, P, covariance; return_params=return_params)
#     end
# end
#
# function remove_kepler(
#     y_obs_w_planet::Vector{Real},
#     times::Vector{T2} where T2<:Unitful.Time,
#     P::Unitful.Time,
#     covariance::Union{Cholesky{T,Matrix{T}},Symmetric{T,Matrix{T}},Matrix{T},Vector{Real}};
#     return_params::Bool=false
#     ) where
#
#     y_obs_wo_planet = copy(y_obs_w_planet)
#     if return_params
#         K, e, M0, ω, γ = remove_kepler!(y_obs_wo_planet, times, P, covariance; return_params=return_params)
#         return y_obs_wo_planet, K, e, M0, ω, γ
#     else
#         remove_kepler!(y_obs_wo_planet, times, P, covariance; return_params=return_params)
#         return y_obs_wo_planet
#     end
# end


function add_kepler_to_Jones_problem_definition(
    prob_def::Jones_problem_definition,
    ks::kep_signal)

    if ustrip(ks.K) == 0
        return prob_def.y_obs
    end

    amount_of_samp_points = length(prob_def.x_obs)
    planet_rvs = ks.(prob_def.time)
    y_obs_w_planet = copy(prob_def.y_obs)
    y_obs_w_planet[1:amount_of_samp_points] += planet_rvs / (prob_def.normals[1] * prob_def.rv_unit)
    return y_obs_w_planet
end

# function add_kepler_to_Jones_problem_definition(
#     prob_def::Jones_problem_definition,
#     P::Unitful.Time,
#     e::Real,
#     M0::Real,
#     m_star::Union{Real, Quantity},
#     m_planet::Unitful.Time,
#     ω::Real;
#     normalization::Real=1,
#     i::Real=π/2,
#     γ::Unitful.Velocity=0.)
#
#     K = velocity_semi_amplitude(P, m_star, m_planet, e=e, i=i)
#     return add_kepler_to_Jones_problem_definition(prob_def, P, e, M0, K, ω; normalization=normalization, γ=γ)
# end
