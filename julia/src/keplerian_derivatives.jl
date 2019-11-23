"""
Script created by build_kep_deriv.jl.

Derivative of transformed Keplerian parameters defined in Pal 2019
"An analytical solution for Kepler's problem"

Parameters:
k (real): eccentrcity * cos(argument of periastron)
l (real): 1 - sqrt(1 - eccentricty ^ 2)
q (real): eccentrcity * cos(eccentric anomaly)
an (real): velocity semi-amplitude / sqrt(1 - eccentricty ^ 2)
γ (real): velocity offset
λ (real): mean anomaly + argument of periastron
dorder (vector of integers): A vector of how many partial derivatives you want
    to take with respect to each variable in the following order Basic[k, l, q, an, γ, λ]
    Can only take up to 2 partials (aka sum(dorder) < 3).

Returns:
float: The derivative specfied with dorder

"""
function kep_deriv_pal(
	k::T,
	l::T,
	q::T,
	an::T,
	γ::T,
	λ::T,
	dorder::Vector{<:Integer}
	) where {T<:Real}

	@assert sum(dorder) < 3
	@assert minimum(dorder) == 0
    @assert length(dorder) == 6

    if dorder==[0, 0, 0, 0, 0, 2]
        func = -an*cos(λ + sqrt(l*(2 - l) - q^2))/(1 - q)
    end

    if dorder==[0, 0, 0, 0, 1, 1]
        func = 0
    end

    if dorder==[0, 0, 0, 1, 0, 1]
        func = -sin(λ + sqrt(l*(2 - l) - q^2))/(1 - q)
    end

    if dorder==[0, 0, 1, 0, 0, 1]
        func = -an*sin(λ + sqrt(l*(2 - l) - q^2))/(1 - q)^2 + q*an*cos(λ + sqrt(l*(2 - l) - q^2))/((1 - q)*sqrt(l*(2 - l) - q^2))
    end

    if dorder==[0, 1, 0, 0, 0, 1]
        func = (-1/2)*an*(-l + 2 - l)*cos(λ + sqrt(l*(2 - l) - q^2))/((1 - q)*sqrt(l*(2 - l) - q^2))
    end

    if dorder==[1, 0, 0, 0, 0, 1]
        func = 0
    end

    if dorder==[0, 0, 0, 0, 0, 1]
        func = -an*sin(λ + sqrt(l*(2 - l) - q^2))/(1 - q)
    end

    if dorder==[0, 0, 0, 0, 2, 0]
        func = 0
    end

    if dorder==[0, 0, 0, 1, 1, 0]
        func = 0
    end

    if dorder==[0, 0, 1, 0, 1, 0]
        func = 0
    end

    if dorder==[0, 1, 0, 0, 1, 0]
        func = 0
    end

    if dorder==[1, 0, 0, 0, 1, 0]
        func = 0
    end

    if dorder==[0, 0, 0, 0, 1, 0]
        func = 1
    end

    if dorder==[0, 0, 0, 2, 0, 0]
        func = 0
    end

    if dorder==[0, 0, 1, 1, 0, 0]
        func = (-k*q/(2 - l) + cos(λ + sqrt(l*(2 - l) - q^2)))/(1 - q)^2 + (-k/(2 - l) + q*sin(λ + sqrt(l*(2 - l) - q^2))/sqrt(l*(2 - l) - q^2))/(1 - q)
    end

    if dorder==[0, 1, 0, 1, 0, 0]
        func = (-k*q/(2 - l)^2 + (-1/2)*(-l + 2 - l)*sin(λ + sqrt(l*(2 - l) - q^2))/sqrt(l*(2 - l) - q^2))/(1 - q)
    end

    if dorder==[1, 0, 0, 1, 0, 0]
        func = -q/((1 - q)*(2 - l))
    end

    if dorder==[0, 0, 0, 1, 0, 0]
        func = (-k*q/(2 - l) + cos(λ + sqrt(l*(2 - l) - q^2)))/(1 - q)
    end

    if dorder==[0, 0, 2, 0, 0, 0]
        func = 2*an*(-k*q/(2 - l) + cos(λ + sqrt(l*(2 - l) - q^2)))/(1 - q)^3 + 2*an*(-k/(2 - l) + q*sin(λ + sqrt(l*(2 - l) - q^2))/sqrt(l*(2 - l) - q^2))/(1 - q)^2 + an*(sin(λ + sqrt(l*(2 - l) - q^2))/sqrt(l*(2 - l) - q^2) + q^2*sin(λ + sqrt(l*(2 - l) - q^2))/(l*(2 - l) - q^2)^(3/2) - q^2*cos(λ + sqrt(l*(2 - l) - q^2))/(l*(2 - l) - q^2))/(1 - q)
    end

    if dorder==[0, 1, 1, 0, 0, 0]
        func = an*(-k*q/(2 - l)^2 + (-1/2)*(-l + 2 - l)*sin(λ + sqrt(l*(2 - l) - q^2))/sqrt(l*(2 - l) - q^2))/(1 - q)^2 + an*(-k/(2 - l)^2 + (-1/2)*q*(-l + 2 - l)*sin(λ + sqrt(l*(2 - l) - q^2))/(l*(2 - l) - q^2)^(3/2) + (1/2)*q*(-l + 2 - l)*cos(λ + sqrt(l*(2 - l) - q^2))/(l*(2 - l) - q^2))/(1 - q)
    end

    if dorder==[1, 0, 1, 0, 0, 0]
        func = -an/((1 - q)*(2 - l)) - q*an/((1 - q)^2*(2 - l))
    end

    if dorder==[0, 0, 1, 0, 0, 0]
        func = an*(-k*q/(2 - l) + cos(λ + sqrt(l*(2 - l) - q^2)))/(1 - q)^2 + an*(-k/(2 - l) + q*sin(λ + sqrt(l*(2 - l) - q^2))/sqrt(l*(2 - l) - q^2))/(1 - q)
    end

    if dorder==[0, 2, 0, 0, 0, 0]
        func = an*(sin(λ + sqrt(l*(2 - l) - q^2))/sqrt(l*(2 - l) - q^2) - 2*k*q/(2 - l)^3 + (1/4)*(-l + 2 - l)^2*sin(λ + sqrt(l*(2 - l) - q^2))/(l*(2 - l) - q^2)^(3/2) + (-1/4)*(-l + 2 - l)^2*cos(λ + sqrt(l*(2 - l) - q^2))/(l*(2 - l) - q^2))/(1 - q)
    end

    if dorder==[1, 1, 0, 0, 0, 0]
        func = -q*an/((1 - q)*(2 - l)^2)
    end

    if dorder==[0, 1, 0, 0, 0, 0]
        func = an*(-k*q/(2 - l)^2 + (-1/2)*(-l + 2 - l)*sin(λ + sqrt(l*(2 - l) - q^2))/sqrt(l*(2 - l) - q^2))/(1 - q)
    end

    if dorder==[2, 0, 0, 0, 0, 0]
        func = 0
    end

    if dorder==[1, 0, 0, 0, 0, 0]
        func = -q*an/((1 - q)*(2 - l))
    end

    if dorder==[0, 0, 0, 0, 0, 0]
        func = γ + an*(-k*q/(2 - l) + cos(λ + sqrt(l*(2 - l) - q^2)))/(1 - q)
    end

    return float(func)

end
