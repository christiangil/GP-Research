using PhysicalConstants
using UnitfulAstro
using Unitful
using LinearAlgebra


"Strip Unitful units"
strip_units(quant::Union{Quantity,PhysicalConstants.Constant}) = ustrip(float(quant))
convert_and_strip_units(new_unit::Unitful.FreeUnits, quant::Quantity) = strip_units(uconvert(new_unit, quant))
convert_and_strip_units(new_unit::Unitful.FreeUnits, quant::Float64) = quant


"Calculate velocity semi-amplitude for RV signal of planets in m/s"
function velocity_semi_amplitude(P::Union{Float64, Quantity}, m_star::Union{Float64, Quantity}, m_planet::Union{Float64, Quantity}; e::Float64=0., i::Float64=pi/2)
    m_star = convert_and_strip_units(u"kg", m_star)
    m_planet = convert_and_strip_units(u"kg", m_planet)
    P = convert_and_strip_units(u"s", P)
    assert_positive(P, m_star, m_planet)
    G = strip_units(PhysicalConstants.CODATA2014.G)  # SI units
    sma = cbrt(P * P * G * (m_star + m_planet) / (4 * pi * pi))
    return sqrt(G/(1 - e * e)) * m_planet * sin(i) / sqrt(sma * (m_star + m_planet))
end


"""
Iterative solution for true anomaly from mean anomaly from (http://www.csun.edu/~hcmth017/master/node16.html)
for small e, can use equation of center approximating true anomaly from (https://en.wikipedia.org/wiki/Equation_of_the_center) O(e^8)
"""
function ϕ(t::Float64, P::Float64; e::Float64=0., iter::Bool=true)
    @assert (0 <= e <= 1) "eccentricity has to be between 0 and 1"
    M = 2 * pi * ((t / P) % 1)
    if e == 0.
        return M
    elseif iter
        dif_thres = 1e-8
        dif = 1
        true_anom = copy(M)
        while dif > dif_thres
            true_anom_old = copy(true_anom)
            true_anom -= (true_anom - e * sin(true_anom) - M) / (1 - e * cos(true_anom))
            dif = abs((true_anom - true_anom_old) / true_anom)
        end
        return true_anom
    else
        e_list = zeros(7)
        e_list = [e_list[i] * e for i in 1:7]
        term_list = [2 * e - 1/4 * e_list[3] + 5/96 * e_list[5] + 107/4608 * e_list[7],
        5/4 * e_list[2] - 11/24 * e_list[4] + 17/192 * e_list[5],
        13/12 * e_list[3] - 43/64 * e_list[5] + 95/512 * e_list[7],
        103/96 * e_list[4] - 451/480 * e_list[6],
        1097/960 * e_list[5] - 5957/4608 * e_list[7],
        1223/960 * e_list[6],
        47273/32256 * e_list[7]]
        sin_list = [sin(i * M) for i in 1:7]
        return M + dot(term_list, sin_list)
    end
end


function ϕ(t::Float64, P::Quantity; e::Float64=0., iter::Bool=true)
    return ϕ(t, convert_and_strip_units(u"s", P); e=e, iter=iter)
end


"Radial velocity formula"
function kepler_rv(K::Float64, t::Float64, P::Float64; e::Float64=0., i::Float64=pi/2, ω::Float64=0., γ::Float64=0.)
    return K * (e * cos(ω) + cos(ω + ϕ(t, P, e=e))) + γ
end


"RV fomula that deals with Unitful planet and star masses"
function kepler_rv(t::Union{Float64, Quantity}, P::Union{Float64, Quantity}, m_star::Union{Float64, Quantity}, m_planet::Union{Float64, Quantity}; e::Float64=0., ω::Float64=0., i::Float64=pi/2, γ::Float64=0.)
    m_star = convert_and_strip_units(u"kg", m_star)
    m_planet = convert_and_strip_units(u"kg", m_planet)
    P = convert_and_strip_units(u"s", P)
    assert_positive(P, m_star, m_planet)
    t = convert_and_strip_units(u"s", t)
    K = velocity_semi_amplitude(P, m_star, m_planet, e=e, i=i)
    return kepler_rv(K, t, P; e=e, i=i, ω=ω, γ=γ)
end


"""
A linear formulation of a Keplerian RV signal where:
coefficients[1] = K * cos(ω)
coefficients[2] = -K * sin(ω)
coefficients[3] = K * e * cos(ω) + γ
"""
function kepler_rv_linear(t, P::Union{Float64, Quantity}, coefficients::Array{Float64,1})
    @assert length(coefficients) == 3 "wrong number of coefficients"
    P = convert_and_strip_units(u"s", P)
    t = convert_and_strip_units.(u"s", t)
    return (coefficients[1] .* cos.(ϕ.(t, P))) + (coefficients[2] .* sin.(ϕ.(t, P))) + (coefficients[3] .* ones(length(t)))
end


"Convert the solar phase information from SOAP 2.0 into days"
function convert_phases_to_days(phase::Float64; P_rot = 25.05)
    # default P_rot is the solar rotation period used by SOAP 2.0 in days
    return phase / (2 * pi / P_rot)
end


"Convert the solar phase information from SOAP 2.0 into seconds"
function convert_phases_to_seconds(phase::Float64; P_rot = 25.05)
    return convert_and_strip_units(u"s", convert_phases_to_days(phase; P_rot = P_rot)u"d")
end


"Remove the best-fit circular Keplerian signal from the Jones multivariate time series"
function remove_kepler(times_obs::Array{Float64,1}, period::Float64, y_obs_w_planet::Array{Float64,1}, covariance::Union{Symmetric{Float64,Array{Float64,2}},Array{Float64}})
    for i in 1:ndims(covariance)
        @assert size(covariance,i)==length(y_obs_w_planet) "covariance incompatible with y_obs_w_planet"
    end
    amount_of_total_samp_points = length(y_obs_w_planet)
    amount_of_samp_points = length(times_obs)
    kepler_rv_linear_terms = hcat(cos.(ϕ.(times_obs, period)), sin.(ϕ.(times_obs, period)), ones(length(times_obs)))
    kepler_linear_terms = vcat(kepler_rv_linear_terms, zeros(amount_of_total_samp_points - amount_of_samp_points, 3))
    x = general_lst_sq(kepler_linear_terms, y_obs_w_planet; covariance=covariance)
    new_data = copy(y_obs_w_planet)
    new_data[1:amount_of_samp_points] -= kepler_rv_linear(times_obs, period, x)
    return new_data
end
