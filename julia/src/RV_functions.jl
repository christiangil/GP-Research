# these functions are related to calculating RV quantities
using UnitfulAstro
using Unitful
# using UnitfulAngles
using LinearAlgebra
using PyPlot
using LineSearches

"Convert Unitful units from one to another and strip the final units"
convert_and_strip_units(new_unit::Unitful.FreeUnits, quant::Quantity) = ustrip(uconvert(new_unit, quant))
convert_and_strip_units(new_unit::Unitful.FreeUnits, quant) = quant


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
    # @assert (0 <= e <= 1) "eccentricity has to be between 0 and 1"
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
    e_or_h::Real,
    ω_or_k::Real;
    γ::Unitful.Velocity=0u"m/s",
    use_hk::Bool=false)

    # P = convert_and_strip_units(u"yr", P)
    # t = convert_and_strip_units(u"yr", t)
    # K = convert_and_strip_units(u"m/s", K)
    # γ = convert_and_strip_units(u"m/s", γ)
    # assert_positive(P)
    return kepler_rv_ecc_anom(t, K, P, M0, e_or_h, ω_or_k; γ=γ, use_hk=use_hk)
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
    γ::Unitful.Velocity=0u"m/s")

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

    kep_signal(; K=0u"m/s", P=1u"d", M0=0, e_or_h=0, ω_or_k=0, γ=0u"m/s", use_hk::Bool=false) = kep_signal(K, P, M0, e_or_h, ω_or_k, γ; use_hk=use_hk)
    function kep_signal(K, P, M0, e_or_h, ω_or_k, γ; use_hk::Bool=false)
        if use_hk
            h = e_or_h
            k = ω_or_k
            e, ω = hk_2_eω(h, k)
        else
            e = e_or_h
            ω = ω_or_k
            h, k = eω_2_hk(e, ω)
        end
        return kep_signal(K, P, M0, e, ω, γ, h, k)
    end
    function kep_signal(
        K::Unitful.Velocity,
        P::Unitful.Time,
        M0::Real,
        e::Real,
        ω::Real,
        γ::Unitful.Velocity,
        h::Real,
        k::Real)

        # @assert 0 <= e < 1 "orbit needs to be bound"
        M0 = mod2pi(M0)
        ω = mod2pi(ω)
        # assert_positive(P)
        return new(K, P, M0, e, ω, γ, h, k)
    end
end
kepler_rv(t::Unitful.Time, ks::kep_signal) = kepler_rv(t, ks.K, ks.P, ks.M0, ks.e, ks.ω; γ=ks.γ)
(ks::kep_signal)(t::Unitful.Time) = kepler_rv(t, ks)

mutable struct kep_buffer{T1<:Real}
    ks::kep_signal
    rm_kep::Vector{T1}
    nprior::T1
end

function ∇nlogL_kep(
    data::Vector{T},
    times::Vector{T2} where T2<:Unitful.Time,
    covariance::Union{Cholesky{T,Matrix{T}},Symmetric{T,Matrix{T}},Matrix{T},Vector{T}},
    ks::kep_signal,
    y::Vector{T};
    data_unit::Unitful.Velocity=1u"m/s") where T<:Real

    α = covariance \ y
    G = zeros(n_kep_parms)
    d = zeros(Int64, n_kep_parms)
    for i in 1:length(G)
        d[:] = zeros(Int64, n_kep_parms)
        d[i] = 1
        G[i] = dnlogLdθ(remove_kepler(data, times, ks; data_unit=data_unit, d=d), α)
    end
    return G
end
∇nlogL_kep(data, times, covariance, ks; data_unit=1u"m/s") =
    ∇nlogL_kep(data, times, covariance, ks, remove_kepler(data, times, ks; data_unit=data_unit); data_unit=data_unit)

function fit_kepler(
    data::Vector{T},
    times::Vector{T2} where T2<:Unitful.Time,
    covariance::Union{Cholesky{T,Matrix{T}},Symmetric{T,Matrix{T}},Matrix{T},Vector{T}},
    init_ks::kep_signal;
    data_unit::Unitful.Velocity=1u"m/s"
    ) where T<:Real

    # if P==0; return kep_signal_epicyclic(p, P, M0, e, ω, γ, coefficients) end
    assert_positive(init_ks.P)
    for i in 1:ndims(covariance)
        @assert size(covariance,i)==length(data) "covariance incompatible with data"
    end
    amount_of_total_samp_points = length(data)
    amount_of_samp_points = length(times)
    normalization = logdet(covariance) + amount_of_total_samp_points * log(2 * π)
    K_u = unit(init_ks.K)
    P_u = unit(init_ks.P)
    γ_u = unit(init_ks.γ)

    current_x = ustrip.([init_ks.K, init_ks.P, init_ks.M0, init_ks.h, init_ks.k, init_ks.γ])
    buffer = kep_buffer(init_ks, remove_kepler(data, times, init_ks; data_unit=data_unit), -logprior_kepler(init_ks; use_hk=true))
    last_x = similar(current_x)

    function calculate_common!(x::Vector{T}, last_x::Vector{T}, buffer::kep_buffer) where {T<:Real}
        if x != last_x
            copy!(last_x, x)
            buffer.ks = ks_from_vec(x, K_u, P_u, γ_u; use_hk=true)
            buffer.nprior = -logprior_kepler(buffer.ks; use_hk=true)
            buffer.rm_kep = remove_kepler(data, times, buffer.ks; data_unit=data_unit)
        end
        # println("buffer: ", kep_parms_str(buffer.ks))
    end

    function f(x::Vector{T}, buffer::kep_buffer, last_x::Vector{T}) where {T<:Real}
        calculate_common!(x, last_x, buffer)
        # println(kep_parms_str(buffer.ks))
        # println(buffer.nprior)
        if buffer.nprior == Inf
            return buffer.nprior
        else
            return nlogL(covariance, buffer.rm_kep, nlogL_normalization=normalization) + buffer.nprior
        end
    end
    f(x) = f(x, buffer, last_x)

    function g!(G::Vector{T}, x::Vector{T}, buffer::kep_buffer, last_x::Vector{T}) where {T<:Real}
        calculate_common!(x, last_x, buffer)
        if buffer.nprior == Inf
            G[:] = zeros(length(G))
        else
            G[:] = ∇nlogL_kep(data, times, covariance, buffer.ks, buffer.rm_kep; data_unit=data_unit)
            d = zeros(Int64, length(G))
            for i in 1:length(G)
                d[:] .= 0
                d[i] = 1
                G[i] -= logprior_kepler(buffer.ks; d=d, use_hk=true)
            end
            # println(G)
        end
    end
    g!(G, x) = g!(G, x, buffer, last_x)

    attempts = 0
    in_saddle = true
    while attempts < 10 && in_saddle
        attempts += 1
        if attempts > 1;
            println("found saddle point. starting attempt $attempts with a perturbation")
            current_x += 1e-1 .* (rand(rng, length(current_x)) .- 0.5)
            current_x[1] = maximum([current_x[1], 0.01])
            current_x[3] = mod2pi(current_x[3])
            e, ω = hk_2_eω(current_x[4], current_x[5])
            e = maximum([minimum([e, 0.75]), 0])
            ω = mod2pi(ω)
            current_x[4:5] .= eω_2_hk(e, ω)
        end
        println(current_x)
        result = optimize(f, g!, current_x, LBFGS(alphaguess=LineSearches.InitialStatic(alpha=1e-2))) # 27s
        current_x = copy(result.minimizer)
        ks = ks_from_vec(current_x, K_u, P_u, γ_u; use_hk=true)
        println(kep_parms_str(ks))
        new_det = det(∇∇nlogL_kep(data, times, covariance, ks; data_unit=data_unit))
        println(new_det)
        in_saddle = new_det <= 0
        if !in_saddle; return ks end
    end
    @error "no non-saddle point soln found"
end


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
    γ::Unitful.Velocity=0u"m/s"
    ) = K * (e * cos(ω) + cos(ω + ϕ(t, P, M0, e))) + γ


function eω_2_hk(e, ω)
    return e * sin(ω), e * cos(ω)
end
function hk_2_eω(h, k)
    return sqrt(h * h + k * k), atan(h, k)
end


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
    e_or_h::Real,
    ω_or_k::Real;
    γ::Unitful.Velocity=0u"m/s",
    use_hk::Bool=false)

    if use_hk
        h = e_or_h
        k = ω_or_k
        e, ω = hk_2_eω(h, k)
    else
        e = e_or_h
        ω = ω_or_k
        h, k = eω_2_hk(e, ω)
    end

    # k = e * cos(ω)
    E = ecc_anomaly(t, P, M0, e)
    j = sqrt(1 - e*e)
    q = e * cos(E)
    # return K * j / (1 - q) * (cos(ω + E) - (1 - j) * cos(E) * cos(ω)) + γ  # equivalent
    return K * j / (1 - q) * (cos(ω + E) - k * q / (1 + j)) + γ
end


struct kep_signal_circ
    K::Unitful.Velocity
    P::Unitful.Time
    M0minusω::Real  # ::Unitful.NoDims
    γ::Unitful.Velocity
    coefficients::Vector{T} where T<:Unitful.Velocity

    kep_signal_circ(; K=0u"m/s", P=0u"d", M0minusω=0, γ=0u"m/s") = kep_signal_circ(K, P, M0minusω, γ)
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
#     coefficients::Vector{<:Real}
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
#     coefficients::Vector{<:Real};
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

    h::Real
    k::Real  # ::Unitful.NoDims
    coefficients::Vector{T} where T<:Unitful.Velocity

    kep_signal_epicyclic(; K=0u"m/s", P=1u"d", M0=0, e_or_h=0, ω_or_k=0, γ=0u"m/s", use_hk::Bool=false) = kep_signal_epicyclic(K, P, M0, e_or_h, ω_or_k, γ; use_hk=use_hk)
    function kep_signal_epicyclic(K, P, M0, e_or_h, ω_or_k, γ; use_hk::Bool=false)
        if use_hk
            h = e_or_h
            k = ω_or_k
            e, ω = hk_2_eω(h, k)
        else
            e = e_or_h
            ω = ω_or_k
            h, k = eω_2_hk(e, ω)
        end
        return kep_signal_epicyclic(K, P, M0, e, ω, γ, h, k)
    end
    kep_signal_epicyclic(K, P, M0, e, ω, γ, h, k) = kep_signal_epicyclic(K, P, M0, e, ω, γ, h, k,
        [K * cos(M0 - ω), K * sin(M0 - ω), e * K * cos(2 * M0 - ω), e * K * sin(2 * M0 - ω), γ])
    function kep_signal_epicyclic(P, coefficients)
        K = sqrt(coefficients[1]^2 + coefficients[2]^2)
        e = sqrt(coefficients[3]^2 + coefficients[4]^2) / K
        # if e > 0.35; @warn "an orbit with this eccentricity would be very different from the output of kepler_rv_epicyclic" end
        M0 = mod2pi(atan(coefficients[4], coefficients[3]) - atan(coefficients[2], coefficients[1]))
        ω = mod2pi(atan(coefficients[4], coefficients[3]) - 2 * atan(coefficients[2], coefficients[1]))
        γ = coefficients[5]
        h, k = eω_2_hk(e, ω)
        return kep_signal_epicyclic(K, P, M0, e, ω, γ, h, k, coefficients)
    end
    function kep_signal_epicyclic(
        K::Unitful.Velocity,
        P::Unitful.Time,
        M0::Real,
        e::Real,
        ω::Real,  # ::Unitful.NoDims
        γ::Unitful.Velocity,
        h::Real,
        k::Real,
        coefficients::Vector{T} where T<:Unitful.Velocity)

        M0 = mod2pi(M0)
        ω = mod2pi(ω)
        assert_positive(P)
        @assert length(coefficients) == 5 "wrong number of coefficients"
        return new(K, P, M0, e, ω, γ, h, k, coefficients)
    end
end
function kepler_rv_epicyclic(t::Unitful.Time, ks::kep_signal_epicyclic)
    phase = unit_phase.(t, ks.P)
    return return (ks.coefficients[1] .* cos.(phase)) + (ks.coefficients[2] .* sin.(phase)) + (ks.coefficients[3] .* cos.(2 .* phase)) + (ks.coefficients[4] .* sin.(2 .* phase)) .+ ks.coefficients[5]
end
kepler_rv(t::Unitful.Time, ks::kep_signal_epicyclic) = kepler_rv_epicyclic(t, ks)
(ks::kep_signal_epicyclic)(t::Unitful.Time) = kepler_rv(t, ks)
function fit_kepler_epicyclic(
    data::Vector{T},
    times::Vector{T2} where T2<:Unitful.Time,
    covariance::Union{Cholesky{T,Matrix{T}},Symmetric{T,Matrix{T}},Matrix{T},Vector{T}},
    P::Unitful.Time;
    data_unit::Unitful.Velocity=1u"m/s"
    ) where T<:Real


    # if P==0; return kep_signal_epicyclic(p, P, M0, e, ω, γ, coefficients) end
    assert_positive(P)
    for i in 1:ndims(covariance)
        @assert size(covariance,i)==length(data) "covariance incompatible with data"
    end
    amount_of_total_samp_points = length(data)
    amount_of_samp_points = length(times)
    phases = unit_phase.(times, P)
    design_matrix = hcat(cos.(phases), sin.(phases), cos.(2 .* phases), sin.(2 .* phases), ones(length(times)))
    amount_of_total_samp_points > amount_of_samp_points ? design_matrix = vcat(design_matrix, zeros(amount_of_total_samp_points - amount_of_samp_points, size(design_matrix, 2))) : design_matrix = design_matrix
    x = general_lst_sq(design_matrix, data; Σ=covariance)
    return kep_signal_epicyclic(P, x .* data_unit)
end
fit_kepler(data, times, covariance, ks::kep_signal_epicyclic; data_unit=1u"m/s") = fit_kepler_epicyclic(data, times, covariance, ks.P; data_unit=data_unit)


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
#     coefficients::Vector{<:Real}
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
#     coefficients::Vector{<:Real},
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

    kep_signal_wright(; K=0u"m/s", P=0u"d", M0=0, e=0, ω=0, γ=0u"m/s") = kep_signal_wright(K, P, M0, e, ω, γ)
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


# "Remove the best-fit epicyclic (linearized in e) Keplerian signal from the data"
# function remove_kepler!(
#     data::Vector{<:Real},
#     times::Vector{T2} where T2<:Unitful.Time,
#     P::Unitful.Time,
#     covariance::Union{Cholesky{T,Matrix{T}},Symmetric{T,Matrix{T}},Matrix{T},Vector{<:Real}};
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
#     y_obs_w_planet::Vector{<:Real},
#     times::Vector{T2} where T2<:Unitful.Time,
#     P::Unitful.Time,
#     covariance::Union{Cholesky{T,Matrix{T}},Symmetric{T,Matrix{T}},Matrix{T},Vector{<:Real}};
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


function add_kepler_to_Jones_problem_definition!(
    prob_def::Jones_problem_definition,
    ks::kep_signal)

    if ustrip(ks.K) == 0
        return prob_def.y_obs
    end

    amount_of_samp_points = length(prob_def.x_obs)
    planet_rvs = ks.(prob_def.time)
    prob_def.y_obs[1:amount_of_samp_points] += planet_rvs / (prob_def.normals[1] * prob_def.rv_unit)
    prob_def.rv[:] += planet_rvs
    # return prob_def
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
#     γ::Unitful.Velocity=0u"m/s")
#
#     K = velocity_semi_amplitude(P, m_star, m_planet, e=e, i=i)
#     return add_kepler_to_Jones_problem_definition(prob_def, P, e, M0, K, ω; normalization=normalization, γ=γ)
# end


function remove_kepler(
    data::Vector{<:Real},
    times::Vector{T2} where T2<:Unitful.Time,
    ks::Union{kep_signal, kep_signal_epicyclic, kep_signal_wright, kep_signal_circ};
    data_unit::Unitful.Velocity=1u"m/s",
    d::Vector{<:Integer}=zeros(Int64,n_kep_parms))

    validate_kepler_dorder(d)

    if all(d .== 0)
        y = copy(data)
        if ustrip(ks.K) != 0; y[1:length(times)] -= uconvert.(unit(data_unit), ks.(times)) ./ data_unit end
    else
        @assert typeof(ks) == kep_signal
        y = zeros(length(data))
        kep_deriv_simple(t) = kep_deriv(ks, t, d)
        y[1:length(times)] -= ustrip.(kep_deriv_simple.(times)) ./ convert_and_strip_units(unit(ks.K), data_unit)
    end
    return y

end
remove_kepler(
    prob_def::Jones_problem_definition,
    ks::Union{kep_signal, kep_signal_epicyclic, kep_signal_wright, kep_signal_circ};
    d::Vector{<:Integer}=zeros(Int64,n_kep_parms)) =
    remove_kepler(prob_def.y_obs, prob_def.time, ks; data_unit=prob_def.rv_unit*prob_def.normals[1], d=d)


function add_kepler(
    data::Vector{<:Real},
    times::Vector{T2} where T2<:Unitful.Time,
    ks::Union{kep_signal, kep_signal_epicyclic, kep_signal_wright, kep_signal_circ};
    data_unit::Unitful.Velocity=1u"m/s")

    y = copy(data)
    if ustrip(ks.K) != 0; y[1:length(times)] += uconvert.(unit(data_unit), ks.(times)) ./ data_unit end
    return y
end
add_kepler(
    prob_def::Jones_problem_definition,
    ks::Union{kep_signal, kep_signal_epicyclic, kep_signal_wright, kep_signal_circ}) =
    add_kepler(prob_def.y_obs, prob_def.time, ks; data_unit=prob_def.rv_unit*prob_def.normals[1])


# function fit_and_remove_kepler_epi(
#     data::Vector{<:Real},
#     times::Vector{T2} where T2<:Unitful.Time,
#     Σ_obs::Union{Cholesky{T,Matrix{T}},Symmetric{T,Matrix{T}},Matrix{T},Vector{T}},
#     P::Unitful.Time;
#     data_unit::Unitful.Velocity=1u"m/s") where T<:Real
#
#     ks = fit_kepler_epicyclic(data, times, Σ_obs, P; data_unit=data_unit)
#     return remove_kepler(data, times, ks; data_unit=data_unit)
# end
# function fit_and_remove_kepler_epi(
#     prob_def::Jones_problem_definition,
#     Σ_obs::Union{Cholesky{T,Matrix{T}},Symmetric{T,Matrix{T}},Matrix{T},Vector{T}},
#     P::Unitful.Time;
#     data_unit::Unitful.Velocity=1u"m/s") where T<:Real)
#     return fit_and_remove_kepler_epi(prob_def.y_obs, prob_def.time, Σ_obs, P; data_unit=prob_def.rv_unit*prob_def.normals[1])
# end


function fit_and_remove_kepler(
    data::Vector{<:Real},
    times::Vector{T2} where T2<:Unitful.Time,
    Σ_obs::Union{Cholesky{T,Matrix{T}},Symmetric{T,Matrix{T}},Matrix{T},Vector{T}},
    ks::Union{kep_signal, kep_signal_epicyclic};
    data_unit::Unitful.Velocity=1u"m/s") where T<:Real

    ks = fit_kepler(data, times, Σ_obs, ks; data_unit=data_unit)
    return remove_kepler(data, times, ks; data_unit=data_unit)
end
fit_and_remove_kepler(
    prob_def::Jones_problem_definition,
    Σ_obs::Union{Cholesky{T,Matrix{T}},Symmetric{T,Matrix{T}},Matrix{T},Vector{T}},
    ks::Union{kep_signal, kep_signal_epicyclic}) where T<:Real =
    fit_and_remove_kepler(prob_def.y_obs, prob_def.time, Σ_obs, ks; data_unit=prob_def.rv_unit*prob_def.normals[1])

fit_kepler(
    prob_def::Jones_problem_definition,
    covariance::Union{Cholesky{T,Matrix{T}},Symmetric{T,Matrix{T}},Matrix{T},Vector{T}},
    ks::Union{kep_signal, kep_signal_epicyclic}) where T<:Real =
    fit_kepler(prob_def.y_obs, prob_def.time, covariance, ks; data_unit=prob_def.rv_unit*prob_def.normals[1])



# TODO
# add linear parts to GP fitting epicyclic and full kepler fitting

kep_parms_str(ks::Union{kep_signal, kep_signal_epicyclic, kep_signal_wright}) =
    "K: $(round(convert_and_strip_units(u"m/s", ks.K), digits=2))" * L"^m/_s" * "  P: $(round(convert_and_strip_units(u"d", ks.P), digits=2))" * L"d" * "  M0: $(round(ks.M0,digits=2))  e: $(round(ks.e,digits=2))  ω: $(round(ks.ω,digits=2)) γ: $(round(convert_and_strip_units(u"m/s", ks.γ), digits=2))" * L"^m/_s"
kep_parms_str_short(ks::Union{kep_signal, kep_signal_epicyclic, kep_signal_wright}) =
    "K: $(round(convert_and_strip_units(u"m/s", ks.K), digits=2))" * L"^m/_s" * "  P: $(round(convert_and_strip_units(u"d", ks.P), digits=2))" * L"d" * "  e: $(round(ks.e,digits=2))"


using DataFrames, CSV


function save_nlogLs(
    seed::Integer,
    sim_id::Integer,
    likelihoods::Vector{T},
    hyperparameters::Vector{T},
    og_ks::Union{kep_signal, kep_signal_epicyclic, kep_signal_wright},
    fit_ks::Union{kep_signal, kep_signal_epicyclic, kep_signal_wright},
    save_loc::String
    ) where {T<:Real}


    likelihood_strs = ["L", "uE", "E"]
    num_likelihoods= length(likelihood_strs)
    @assert length(likelihoods) == 3 * num_likelihoods
    orbit_params_strs = ["K", "P", "M0", "e", "ω", "γ"]
    orbit_params= [og_ks.K, og_ks.P, og_ks.M0, og_ks.e, og_ks.ω, og_ks.γ, fit_ks.K, fit_ks.P, fit_ks.M0, fit_ks.e, fit_ks.ω, fit_ks.γ]
    num_hyperparameters = Int(length(hyperparameters) / 2)
    # file_name = "csv_files/$(kernel_name)_logLs.csv"
    file_name = save_loc * "logL.csv"

    df = DataFrame(seed=seed, sim_id=sim_id, date=today())

    for i in 1:length(likelihoods)
        df[!, Symbol(string(likelihood_strs[(i-1)%num_likelihoods + 1]) * string(Int(1 + floor((i-1)//num_likelihoods))))] .= likelihoods[i]
    end
    # df[!, Symbol("E_wp")] .= likelihoods[end]
    for i in 1:length(hyperparameters)
        df[!, Symbol("H" * string(((i-1)%num_hyperparameters) + 1) * "_" * string(Int(1 + floor((i-1)//num_hyperparameters))))] .= hyperparameters[i]
    end
    for i in 1:length(orbit_params)
        df[!, Symbol(string(orbit_params_strs[(i-1)%n_kep_parms + 1]) * string(Int(1 + floor((i-1)//n_kep_parms))))] .= orbit_params[i]
    end

    # if isfile(file_name); append!(df, CSV.read(file_name)) end

    CSV.write(file_name, df)

end


# TODO properly fix jank
function ∇∇nlogL_kep(
    data::Vector{T},
    times::Vector{T2} where T2<:Unitful.Time,
    covariance::Union{Cholesky{T,Matrix{T}},Symmetric{T,Matrix{T}},Matrix{T},Vector{T}},
    ks::kep_signal;
    data_unit::Unitful.Velocity=1u"m/s",
    fix_jank::Bool=false) where T<:Real

    # prob_def -> data, times, data_unit
    # prob_def.y_obs, prob_def.time, ks; data_unit=prob_def.rv_unit*prob_def.normals[1]
    y = remove_kepler(data, times, ks; data_unit=data_unit)
    α = covariance \ y
    H = zeros(n_kep_parms, n_kep_parms)
    for i in 1:n_kep_parms
        for j in 1:n_kep_parms
            if i <= j
                d = zeros(Int64, n_kep_parms)
                d[j] += 1
                y2 = remove_kepler(data, times, ks; data_unit=data_unit, d=d)
                d[i] += 1
                y12 = remove_kepler(data, times, ks; data_unit=data_unit, d=d)
                d[j] -= 1
                y1 = remove_kepler(data, times, ks; data_unit=data_unit, d=d)
                H[i, j] = d2nlogLdθ(y2, y12, α, covariance \ y1)
            end
        end
    end

    # for some reason dhdk and (dk)^2 aren't quite working
    if fix_jank
        est_H = est_∇∇nlogL_kep(data, times, covariance, ks; data_unit=data_unit)
        H[5, 5] = est_H[5, 5]
        H[4, 5] = est_H[4, 5]
    end

    return Symmetric(H)
end


function ∇∇nlogL_Jones_and_planet!(
    workspace::nlogL_matrix_workspace,
    prob_def::Jones_problem_definition,
    total_hyperparameters::Vector{<:Real},
    ks::kep_signal)

    calculate_shared_∇nlogL_matrices!(workspace, prob_def, total_hyperparameters)

    non_zero_inds = copy(prob_def.non_zero_hyper_inds)
    n_hyper = length(non_zero_inds)
    full_H = zeros(n_kep_parms + n_hyper, n_kep_parms + n_hyper)
    full_H[1:n_hyper, 1:n_hyper] = ∇∇nlogL_Jones(
        prob_def, total_hyperparameters; Σ_obs=workspace.Σ_obs, y_obs=remove_kepler(prob_def, ks))

    full_H[n_hyper+1:end,n_hyper+1:end] = ∇∇nlogL_kep(prob_def.y_obs, prob_def.time, workspace.Σ_obs, ks; data_unit=prob_def.rv_unit*prob_def.normals[1], fix_jank=true)

    # TODO allow y and α to be passed to ∇∇nlogL_kep
    y = remove_kepler(prob_def, ks)
    α = workspace.Σ_obs \ y
    for (i, nzind1) in enumerate(non_zero_inds)
        for j in 1:n_kep_parms
            d = zeros(Int64, n_kep_parms)
            d[j] += 1
            y1 = remove_kepler(prob_def, ks; d=d)
            full_H[i, j + n_hyper] = d2nlogLdθ(y, y1, α, workspace.Σ_obs \ y1, workspace.βs[i])
        end
    end

    return Symmetric(full_H)
end


function validate_kepler_dorder(d::Vector{<:Integer})
	@assert sum(d) < 3
	@assert minimum(d) == 0
	@assert length(d) == 6
end


function ks_from_vec(vector::Vector{<:Real}, K_u::Unitful.VelocityFreeUnits, P_u::Unitful.TimeFreeUnits, γ_u::Unitful.VelocityFreeUnits; use_hk::Bool=false)
    @assert length(vector) == n_kep_parms
    return kep_signal(vector[1] * K_u, vector[2] * P_u, vector[3], vector[4], vector[5], vector[6] * γ_u; use_hk=use_hk)
end
