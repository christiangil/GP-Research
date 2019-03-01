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
function velocity_semi_amplitude(P::Real, m_star::Real, m_planet::Real; e::Real=0., i::Real=pi/2)
    assert_positive(P, m_star, m_planet)
    comb_mass = m_star + m_planet
    K_au_yr = 2 * pi * sin(i) * m_planet / (sqrt(1 - (e * e)) * cbrt(comb_mass * comb_mass * P))
    return K_au_yr * convert_and_strip_units(u"m", 1u"AU") / convert_and_strip_units(u"s", 1u"yr")
end

function velocity_semi_amplitude(P::Union{Real, Quantity}, m_star::Union{Real, Quantity}, m_planet::Union{Real, Quantity}; e::Real=0., i::Real=pi/2)
    m_star = convert_and_strip_units(u"Msun", m_star)
    m_planet = convert_and_strip_units(u"Msun", m_planet)
    P = convert_and_strip_units(u"yr", P)
    return velocity_semi_amplitude(P, m_star, m_planet; e=e, i=i)
end


"calculate mean anomaly"
mean_anomaly(t::Real, P::Real) = 2 * pi * ((t / P) % 1)


"""
Iterative solution for true anomaly from mean anomaly
from (http://www.csun.edu/~hcmth017/master/node16.html)
"""
function ϕ(t::Real, P::Real; e::Real=0.)
    @assert (0 <= e <= 1) "eccentricity has to be between 0 and 1"
    M = mean_anomaly(t, P)
    if e == 0.
        return M
    else
        dif_thres = 1e-8
        dif = 1
        true_anom = copy(M)
        while dif > dif_thres
            true_anom_old = copy(true_anom)
            true_anom -= (true_anom - e * sin(true_anom) - M) / (1 - e * cos(true_anom))
            dif = abs((true_anom - true_anom_old) / true_anom)
        end
        return true_anom
    end
end

function ϕ(t::Real, P::Quantity; e::Real=0.)
    return ϕ(t, convert_and_strip_units(u"yr", P); e=e, iter=iter)
end

"""
Calculate true anomaly for small e using equation of center approximating true
anomaly from (https://en.wikipedia.org/wiki/Equation_of_the_center) O(e^8)
"""
function ϕ_approx(t::Real, P::Real; e::Real=0.)
    @assert (0 <= e <= 1) "eccentricity has to be between 0 and 1"
    M = mean_anomaly(t, P)
    if e == 0.
        return M
    else
        e_list = [e ^ i for i in 1:7]
        term_list = [2 * e_list[1] - 1/4 * e_list[3] + 5/96 * e_list[5] + 107/4608 * e_list[7],
        5/4 * e_list[2] - 11/24 * e_list[4] + 17/192 * e_list[6],
        13/12 * e_list[3] - 43/64 * e_list[5] + 95/512 * e_list[7],
        103/96 * e_list[4] - 451/480 * e_list[6],
        1097/960 * e_list[5] - 5957/4608 * e_list[7],
        1223/960 * e_list[6],
        47273/32256 * e_list[7]]
        sin_list = [sin(i * M) for i in 1:7]
        return M + dot(term_list, sin_list)
    end
end

function ϕ_approx(t::Real, P::Quantity; e::Real=0.)
    return ϕ_approx(t, convert_and_strip_units(u"yr", P); e=e, iter=iter)
end


"""
Radial velocity formula
adapted from eq. 11 of (http://exoplanets.astro.yale.edu/workshop/EPRV/Bibliography_files/Radial_Velocity.pdf)
"""
function kepler_rv(K::Real, t::Real, P::Real; e::Real=0., i::Real=pi/2, ω::Real=0., γ::Real=0.)
    assert_positive(P)
    return K * (e * cos(ω) + cos(ω + ϕ(t, P, e=e))) + γ
end

function kepler_rv(t::Union{Real, Quantity}, P::Union{Real, Quantity}, m_star::Union{Real, Quantity}, m_planet::Union{Real, Quantity}; e::Real=0., i::Real=pi/2, ω::Real=0., γ::Real=0.)
    m_star = convert_and_strip_units(u"Msun", m_star)
    m_planet = convert_and_strip_units(u"Msun", m_planet)
    P = convert_and_strip_units(u"yr", P)
    assert_positive(m_star, m_planet)
    t = convert_and_strip_units(u"yr", t)
    K = velocity_semi_amplitude(P, m_star, m_planet, e=e, i=i)
    return kepler_rv(K, t, P; e=e, i=i, ω=ω, γ=γ)
end


"""
A linear formulation of a Keplerian RV signal where:
coefficients[1] = K * cos(ω)
coefficients[2] = -K * sin(ω)
coefficients[3] = K * e * cos(ω) + γ
"""
function kepler_rv_linear(t, P::Union{Real, Quantity}, coefficients::Array{T,1}) where {T<:Real}
    @assert length(coefficients) == 3 "wrong number of coefficients"
    P = convert_and_strip_units(u"yr", P)
    assert_positive(P)
    t = convert_and_strip_units.(u"yr", t)
    return (coefficients[1] .* cos.(ϕ.(t, P))) + (coefficients[2] .* sin.(ϕ.(t, P))) + (coefficients[3] .* ones(length(t)))
end


"Convert the solar phase information from SOAP 2.0 into days"
function convert_phases_to_days(phase::Real; P_rot=25.05)
    # default P_rot is the solar rotation period used by SOAP 2.0 in days
    assert_positive(P_rot)
    return phase / (2 * pi / P_rot)
end

"Convert the solar phase information from SOAP 2.0 into years"
function convert_phases_to_years(phase::Real; P_rot = 25.05)
    return convert_and_strip_units(u"yr", convert_phases_to_days(phase; P_rot=P_rot)u"d")
end


"Remove the best-fit circular Keplerian signal from the data"
function remove_kepler!(data::Array{T1,1}, times::Array{T2,1}, P::Real, covariance::Union{Cholesky{T3,Array{T3,2}},Symmetric{T4,Array{T4,2}},Array{T5}}) where {T1<:Real, T2<:Real, T3<:Real, T4<:Real, T5<:Real}
    assert_positive(P)
    for i in 1:ndims(covariance)
        @assert size(covariance,i)==length(data) "covariance incompatible with data"
    end
    amount_of_total_samp_points = length(data)
    amount_of_samp_points = length(times)
    kepler_rv_linear_terms = hcat(cos.(ϕ.(times, P)), sin.(ϕ.(times, P)), ones(length(times)))
    if amount_of_total_samp_points > amount_of_samp_points
        kepler_linear_terms = vcat(kepler_rv_linear_terms, zeros(amount_of_total_samp_points - amount_of_samp_points, 3))
    else
        kepler_linear_terms = kepler_rv_linear_terms
    end
    x = general_lst_sq(kepler_linear_terms, data; Σ=covariance)
    data[1:amount_of_samp_points] -= kepler_rv_linear(times, P, x)
end

function remove_kepler(y_obs_w_planet::Array{T1,1}, times::Array{T2,1}, P::Real, covariance::Union{Cholesky{T3,Array{T3,2}},Symmetric{T4,Array{T4,2}},Array{T5}}) where {T1<:Real, T2<:Real, T3<:Real, T4<:Real, T5<:Real}
    y_obs_wo_planet = copy(y_obs_w_planet)
    remove_kepler!(y_obs_wo_planet, times, P, covariance)
    return y_obs_wo_planet
end


"Find GP likelihoods for best fit Keplerian orbit of a specified period"
function kep_signal_likelihood(period_grid::Array{T1,1}, fake_data::Array{T2,1}, problem_definition::Jones_problem_definition, total_hyperparameters::Array{T3,1}) where {T1<:Real, T2<:Real, T3<:Real}
    K_obs = K_observations(problem_definition, total_hyperparameters)
    times_obs = convert_phases_to_years.(problem_definition.x_obs)
    likelihoods = zeros(length(period_grid))
    new_data = zeros(length(fake_data))
    for i in 1:length(period_grid)
        new_data .= fake_data
        remove_kepler!(new_data, times_obs, period_grid[i], K_obs)
        likelihoods[i] = nlogL_Jones(problem_definition, total_hyperparameters, y_obs=new_data)
    end
    return likelihoods
end
