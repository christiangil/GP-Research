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
    G = strip_units(PhysicalConstants.CODATA2014.G)  # SI units
    sma = cbrt(P * P * G * (m_star + m_planet) / (4 * pi * pi))
    return sqrt(G/(1 - e * e)) * m_planet * sin(i) / sqrt(sma * (m_star + m_planet))
end


"Equation of center approximating true anomaly from (https://en.wikipedia.org/wiki/Equation_of_the_center) O(e^8)"
function ϕ(t::Float64, P::Float64; e::Float64=0.)
    @assert (0 <= e <= 1) "eccentricity has to be between 0 and 1"
    M = 2 * pi * ((t / P) % 1)
    if e == 0.
        return M
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


"Radial velocity formula"
function rv(K::Float64, t::Float64, P::Union{Float64, Quantity}; e::Float64=0., i::Float64=pi/2, ω::Float64=0., γ::Float64=0.)
    P = convert_and_strip_units(u"s", P)
    return K * (e * cos(ω) + cos(ω + ϕ(t, P, e=e))) + γ
end


"RV fomula that deals with Unitful planet and star masses"
function rv(t::Union{Float64, Quantity}, P::Union{Float64, Quantity}, m_star::Union{Float64, Quantity}, m_planet::Union{Float64, Quantity}; e::Float64=0., ω::Float64=0., i::Float64=pi/2, γ::Float64=0.)
    m_star = convert_and_strip_units(u"kg", m_star)
    m_planet = convert_and_strip_units(u"kg", m_planet)
    P = convert_and_strip_units(u"s", P)
    t = convert_and_strip_units(u"s", t)
    K = velocity_semi_amplitude(P, m_star, m_planet, e=e, i=i)
    return rv(K, t, P; e=e, i=i, ω=ω, γ=γ)
end
