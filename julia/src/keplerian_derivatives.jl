"""
Derivative of transformed Keplerian parameters defined in Pal 2019
"An analytical solution for Kepler's problem"

Parameters:
K (real): velocity semi-amplitude
h * (real): eccentricity * sin(argument of periastron)
k * (real): eccentricity * cos(argument of periastron)
M0 (real): initial mean anomaly
γ (real): velocity offset
P (real): period
λ (real): mean anomaly + argument of periastron
dorder (vector of integers): A vector of how many partial derivatives you want
	to take with respect to each variable in the following order Basic[k, l, q, an, γ, λ]
	Can only take up to 2 * partials * (aka sum(dorder) < 3).

Returns:
float: The derivative specfied with dorder

"""
function kep_deriv(
	K::Unitful.Velocity,
	P::Unitful.Time,
	M0::Real,
	h::Real,
	k::Real,
	γ::Unitful.Velocity,
	t::Unitful.Time,
	dorder::Vector{<:Integer})

	validate_kepler_dorder(dorder)

	esq = h * h + k * k
	e = sqrt(esq)
	EA = ecc_anomaly(t, P, M0, e)
	q = e * cos(EA)
	p = e * sin(EA)
	ω = atan(h, k)
	# λ = mean_anomaly(t, P, M0) + atan(h, k)
	# c = cos(λ + p)
	# s = sin(λ + p)
	c = cos(EA + ω)
	s = sin(EA + ω)
	jsq = 1 - esq
	j = sqrt(jsq)
	qmod = 1-q
	jmod = 1+j

	if dorder==[0, 2, 0, 0, 0, 0]
		func = (1 / (jmod * P^4 * qmod^5)) * 4 * j * π * t * (-jmod * qmod * s * (P *
			qmod^2 - 3 * p * π * t) + c * jmod * (-p * P * qmod^2 + 3 * p^2 * π *
			t - π * qmod * t) + k * (p * P * qmod^2 - 3 * p^2 * π * t + π * q *
			qmod * t)) * K
	end

	if dorder==[0, 1, 0, 0, 0, 1]
		func = 0
	end

	if dorder==[0, 1, 1, 0, 0, 0]
		func = (2 * j * π * (k * (-3 * p^2 + q - q^2) + c * jmod * (3 * p^2 -
			qmod) + 3 * jmod * p * qmod * s) * t * K) / (jmod * P^2 * qmod^5)
	end

	if dorder==[0, 1, 0, 0, 1, 0]
		func = (1 / (esq * jmod^2 * j * P^2 * qmod^5)) * 2 * π * (3 * c^2 * esq *
			jmod^2 * jsq * p + c * jmod * (h * jmod * jsq * (3 * p^2 - qmod) -
			esq * k * p * (6 * jsq + qmod^2 + j * (3 * jsq + qmod^2)) + 2 * esq *
			jmod * jsq * (2 - 3 * q + q^2) * s) - h * jmod * jsq * (3 * k * p^2 -
			k * q * qmod - 3 * jmod * p * qmod * s) + esq * ((-1 + h^2 - j) * j *
			p * qmod^2 + jmod * k * qmod^2 * (k * p - jmod * qmod * s) + jmod *
			jsq * (3 * k^2 * p - (2 + j) * k * qmod * s - 2 * jmod * p * qmod *
			s^2))) * t * K
	end

	if dorder==[0, 1, 0, 1, 0, 0]
		func = -(1 / (esq * jmod^2 * j * P^2 * qmod^5)) * 2 * π * (c^2 * esq *
			jmod^2 * jsq * (2 - 3 * q + q^2) - jmod * jsq * k * (3 * k * p^2 -
			k * q * qmod - 3 * jmod * p * qmod * s) + c * jmod * (esq * h * jmod *
			p * qmod^2 + jsq * k * (jmod * (3 * p^2 - qmod) - esq * qmod) + esq *
			jmod * jsq * p * (3 * h + (-5 + 2 * q) * s)) - esq * (jmod * jsq * s *
			(-3 * k * p + jmod * (2 - 3 * q + q^2) * s) + h * (k * p * (3 * jsq +
			qmod^2 - j * qmod^2 + j * (3 * jsq + qmod^2)) - jmod^2 * qmod *
			(jsq + qmod^2) * s))) * t * K
	end

	if dorder==[1, 1, 0, 0, 0, 0]
		func = (2 * j * π * (c * jmod * p - k * p + jmod * qmod * s) * t) /
			(jmod * P^2 * qmod^3)
	end

	if dorder==[0, 1, 0, 0, 0, 0]
		func = (2 * j * π * (c * jmod * p - k * p - jmod * -qmod * s) *
			t * K) / (jmod * P^2 * qmod^3)
	end

	if dorder==[0, 0, 0, 0, 0, 2]
		func = 0
	end

	if dorder==[0, 0, 1, 0, 0, 1]
		func = 0
	end

	if dorder==[0, 0, 0, 0, 1, 1]
		func = 0
	end

	if dorder==[0, 0, 0, 1, 0, 1]
		func = 0
	end

	if dorder==[1, 0, 0, 0, 0, 1]
		func = 0
	end

	if dorder==[0, 0, 0, 0, 0, 1]
		func = 1
	end

	if dorder==[0, 0, 2, 0, 0, 0]
		func = (j * (k * (-3 * p^2 + q - q^2) + c * jmod * (3 * p^2 - qmod) + 3 *
			jmod * p * qmod * s) * K) / (jmod * qmod^5)
	end

	if dorder==[0, 0, 1, 0, 1, 0]
		func = (1 / (esq * jmod^2 * j * qmod^5)) * (3 * c^2 * esq * jmod^2 * jsq *
			p + c * jmod * (h * jmod * jsq * (3 * p^2 - qmod) - esq * k * p * (6 *
			jsq + qmod^2 + j * (3 * jsq + qmod^2)) + 2 * esq * jmod * jsq *
			(2 - 3 * q + q^2) * s) - h * jmod * jsq * (3 * k * p^2 - k * q * qmod -
			3 * jmod * p * qmod * s) + esq * ((-1 + h^2 - j) * j * p * qmod^2 +
			jmod * k * qmod^2 * (k * p - jmod * qmod * s) + jmod * jsq * (3 * k^2 *
			p - (2 + j) * k * qmod * s - 2 * jmod * p * qmod * s^2))) * K
	end

	if dorder==[0, 0, 1, 1, 0, 0]
		func = -(1 / (esq * jmod^2 * j * qmod^5)) * (c^2 * esq * jmod^2 * jsq *
			(2 - 3 * q + q^2) - jmod * jsq * k * (3 * k * p^2 - k * q * qmod -
			3 * jmod * p * qmod * s) + c * jmod * (esq * h * jmod * p * qmod^2 +
			jsq * k * (jmod * (3 * p^2 - qmod) - esq * qmod) + esq * jmod * jsq *
			p * (3 * h + (-5 + 2 * q) * s)) - esq * (jmod * jsq * s * (-3 * k * p +
			jmod * (2 - 3 * q + q^2) * s) + h * (k * p * (3 * jsq + qmod^2 - j *
			qmod^2 + j * (3 * jsq + qmod^2)) - jmod^2 * qmod * (jsq + qmod^2) * s))) * K
	end

	if dorder==[1, 0, 1, 0, 0, 0]
		func = -((j * (-c * jmod * p + k * p - jmod * qmod * s)) / (jmod * qmod^3))
	end

	if dorder==[0, 0, 1, 0, 0, 0]
		func = -(j * (-c * jmod * p + k * p + jmod * -qmod * s) * K) / (jmod *
			qmod^3)
	end

	if dorder==[0, 0, 0, 0, 2, 0]
		func = -(((-jmod^2 * jsq^3 * (c * esq - esq * k + h * p)^2 * (c + c * j - k * q) -
	   		esq * jmod^2 * jsq^2 * k * (-c * esq + esq * k - h * p) * (c + c * j -
		  	k * q) * qmod^2 + esq^2 * jmod^2 * jsq^2 * (c + c * j - k * q) * qmod^4 -
	   		3 * esq * j * jmod * jsq * k * qmod^3 * (j * jmod * k * (c * esq - esq * k + h * p) -
		  	esq * (-1 + h^2 - j) * q * qmod + j * jmod^2 * s * (-h + esq * s)) -
	   		jmod^2 * jsq^3 * (c + c * j - k * q) * qmod * (h * (h * q - 2 * k * p * qmod) +
		  	2 * esq * h * s + esq^2 * (-qmod - s^2)) -
	   		2 * jmod * jsq^2 * (c * esq - esq * k +
		  	h * p) * (jmod * jsq * (c * esq - esq * k + h * p) * (c + c * j - k * q) -
		  	esq * jmod * k * (c + c * j - k * q) * qmod^2 -
		  	j * qmod * (j * jmod * k * (c * esq - esq * k + h * p) -
			esq * (-1 + h^2 - j) * q * qmod + j * jmod^2 * s * (-h + esq * s))) -
	   		esq * jmod * jsq * k * qmod^2 * (jmod * jsq * (c * esq - esq * k + h * p) * (c +
			c * j - k * q) - esq * jmod * k * (c + c * j - k * q) * qmod^2 -
		  	j * qmod * (j * jmod * k * (c * esq - esq * k + h * p) -
			esq * (-1 + h^2 - j) * q * qmod + j * jmod^2 * s * (-h + esq * s))) +
	   		j * qmod * (jmod^2 * j^5 * k * (c * esq - esq * k + h * p)^2 -
		  	esq * jmod * jsq^2 * k^2 * (-c * esq + esq * k - h * p) * qmod^2 +
		  	esq * jmod * (-h^2 + jmod) * jsq^2 * (c * esq - esq * k + h * p) * qmod^2 +
		  	esq * jmod^2 * j^5 * (c * esq - esq * k + h * p) * qmod^2 +
		  	esq^2 * jmod * (-h^2 + jmod) * jsq * k * q * qmod^3 -
		  	2 * esq^2 * (-1 + h^2 - j) * j^3 * k * q * qmod^3 -
		  	esq^2 * jmod * j^3 * k * q * qmod^3 +
		  	c * jmod^3 * j^5 * qmod * (h - esq * s)^2 +
		  	jmod^3 * j^5 * (c * esq - esq * k + h * p) * s * (-h + esq * s) +
		  	jmod^3 * j^5 * qmod * s * (2 * h * k * qmod + c * esq * (-h + esq * s)) +
		  	jmod^2 * j^5 * k * qmod * (h * (h * q - 2 * k * p * qmod) + 2 * esq * h * s +
			esq^2 * (-qmod - s^2)))) * K) / (jmod^3 * j^5 * (esq - esq * q)^2 * qmod^3))
	end

	if dorder==[0, 0, 0, 1, 1, 0]
		func = -(((2 * jmod * jsq^2 * (c * esq - esq * k +
			h * p) * (esq * h * jmod * (c + c * j - k * q) * qmod^2 +
          	jmod * jsq * (c + c * j - k * q) * (k * p + esq * (h - s)) +
          	j * qmod * (esq * h * k * q * qmod - j * jmod * k * (k * p + esq * (h - s)) -
            j * jmod^2 * (c * esq - k) * s)) +
       		esq * jmod * jsq * k * qmod^2 * (esq * h * jmod * (c + c * j - k * q) * qmod^2 +
          	jmod * jsq * (c + c * j - k * q) * (k * p + esq * (h - s)) +
          	j * qmod * (esq * h * k * q * qmod - j * jmod * k * (k * p + esq * (h - s)) -
            j * jmod^2 * (c * esq - k) * s)) -
       		j * (esq * h * j * jmod^2 * jsq * (c * esq - esq * k + h * p) * (c + c * j -
            k * q) * qmod^2 -
          	jmod^2 * j^5 * (c * esq - esq * k + h * p) * (c + c * j - k * q) * (k * p +
            esq * (h - s)) +
          	2 * esq * j * jmod^2 * jsq * k * (c + c * j - k * q) * qmod^2 * (k * p +
            esq * (h - s)) +
          	jmod * jsq^2 * (c * esq - esq * k + h * p) * qmod * (esq * h * k * q * qmod -
            j * jmod * k * (k * p + esq * (h - s)) - j * jmod^2 * (c * esq - k) * s) +
          	2 * esq * jmod * jsq * k * qmod^3 * (esq * h * k * q * qmod -
            j * jmod * k * (k * p + esq * (h - s)) - j * jmod^2 * (c * esq - k) * s) +
          	esq * h * jmod * jsq * qmod^3 * (j * jmod * k * (c * esq - esq * k + h * p) -
            esq * (-1 + h^2 - j) * q * qmod + j * jmod^2 * s * (-h + esq * s)) +
          	jmod * jsq^2 * qmod * (k * p +
            esq * (h - s)) * (j * jmod * k * (c * esq - esq * k + h * p) -
            esq * (-1 + h^2 - j) * q * qmod + j * jmod^2 * s * (-h + esq * s)) -
          	jmod^2 * j^5 * (c + c * j - k * q) * qmod * ((h^2 - k^2) * p * qmod +
            c * esq * (h - esq * s) + k * (h * q + esq * s)) -
          	qmod * (-esq * h * jmod * jsq^2 * k * (-c * esq + esq * k - h * p) * qmod^2 +
            esq^2 * h * jmod * jsq^2 * q * qmod^3 +
            esq^2 * h * jmod * jsq * k^2 * q * qmod^3 +
            2 * esq^2 * h * j^3 * k^2 * q * qmod^3 +
            jmod^2 * j^5 * k * (-c * esq + esq * k - h * p) * (k * p + esq * (h - s)) -
            esq * jmod^2 * j^5 * qmod^2 * (k * p + esq * (h - s)) -
            esq * jmod * jsq^2 * k^2 * qmod^2 * (k * p + esq * (h - s)) -
            jmod^3 * j^5 * (c * esq - k) * (c * esq - esq * k + h * p) * s -
            c * jmod^3 * j^5 * (c * esq - k) * qmod * (-h + esq * s) +
            jmod^3 * j^5 *
            qmod * s * ((h^2 - k^2) * qmod + esq * s * (-h + esq * s)) -
            jmod^2 * j^5 *
            k * qmod * ((h^2 - k^2) * p * qmod + c * esq * (h - esq * s) +
            k * (h * q + esq * s))))) * K) / (jmod^3 * j^5 * (esq - esq * q)^2 * qmod^3))
	end

	# if dorder==[0, 0, 2, 0, 0, 0]
	#	 func = -(((-jmod^2 * jsq^3 * (c * esq - esq * k + h * p)^2 * (c + c *
	# 		j - k * q) - esq * jmod^2 * jsq^2 * k * (-c * esq + esq * k - h *
	# 		p) * (c + c * j - k * q) * qmod^2 + esq^2 * jmod^2 * jsq^2 * (c +
	# 		c * j - k * q) * qmod^4 - 3 * esq * j * jmod * jsq * k * qmod^3 *
	# 		(j * jmod * k * (c * esq - esq * k + h * p) - esq * (-1 + h^2 - j) *
	# 		q * qmod + j * jmod^2 * s * (-h + esq * s)) - jmod^2 * jsq^3 * (c +
	# 		c * j - k * q) * qmod * (h * (h * q - 2 * k * p * qmod) + 2 * esq *
	# 		h * s + esq^2 * (-qmod - s^2)) - 2 * jmod * jsq^2 * (c * esq -
	# 		esq * k + h * p) * (jmod * jsq * (c * esq - esq * k + h * p) *
	# 		(c + c * j - k * q) - esq * jmod * k * (c + c * j - k * q) * qmod^2 -
	# 		j * qmod * (j * jmod * k * (c * esq - esq * k + h * p) - esq *
	# 		(-1 + h^2 - j) * q * qmod + j * jmod^2 * s * (-h + esq * s))) -
	# 		esq * jmod * jsq * k * qmod^2 * (jmod * jsq * (c * esq - esq *
	# 		k + h * p) * (c + c * j - k * q) - esq * jmod * k * (c + c * j - k *
	# 		q) * qmod^2 - j * qmod * (j * jmod * k * (c * esq - esq * k + h *
	# 		p) - esq * (-1 + h^2 - j) * q * qmod + j * jmod^2 * s * (-h + esq *
	# 		s))) + j * qmod * (jmod^2 * j^5 * k * (c * esq - esq * k + h * p)^2 -
	# 		esq * jmod * jsq^2 * k^2 * (-c * esq + esq * k - h * p) * qmod^2 +
	# 		esq * jmod * (-h^2 + jmod) * jsq^2 * (c * esq - esq * k + h * p) *
	# 		qmod^2 + esq * jmod^2 * j^5 * (c * esq - esq * k + h * p) * qmod^2 +
	# 		esq^2 * jmod * (-h^2 + jmod) * jsq * k * q * qmod^3 - 2 * esq^2 *
	# 		(-1 + h^2 - j) * j^3 * k * q * qmod^3 - esq^2 * jmod * j^3 * k * q *
	# 		qmod^3 + c * jmod^3 * j^5 * qmod * (h - esq * s)^2 + jmod^3 * j^5 *
	# 		(c * esq - esq * k + h * p) * s * (-h + esq * s) + jmod^3 * j^5 *
	# 		qmod * s * (2 * h * k * qmod + c * esq * (-h + esq * s)) + jmod^2 *
	# 		j^5 * k * qmod * (h * (h * q - 2 * k * p * qmod) + 2 * esq * h * s +
	# 		esq^2 * (-qmod - s^2)))) * K) / (jmod^3 * j^5 * (esq - esq * q)^2 *
	# 		qmod^3))
	# end
	#
	# if dorder==[0, 1, 1, 0, 0, 0]
	#	 func = -(((2 * jmod * jsq^2 * (c * esq - esq * k + h * p) * (esq * h *
	# 		jmod * (c + c * j - k * q) * qmod^2 + jmod * jsq * (c + c * j - k *
	# 		q) * (k * p + esq * (h - s)) + j * qmod * (esq * h * k * q * qmod -
	# 		j * jmod * k * (k * p + esq * (h - s)) - j * jmod^2 * (c * esq - k) *
	# 		s)) + esq * jmod * jsq * k * qmod^2 * (esq * h * jmod * (c + c *
	# 		j - k * q) * qmod^2 + jmod * jsq * (c + c * j - k * q) * (k * p +
	# 		esq * (h - s)) + j * qmod * (esq * h * k * q * qmod - j * jmod * k *
	# 		(k * p + esq * (h - s)) - j * jmod^2 * (c * esq - k) * s)) - j *
	# 		(esq * h * j * jmod^2 * jsq * (c * esq - esq * k + h * p) * (c +
	# 		c * j - k * q) * qmod^2 - jmod^2 * j^5 * (c * esq - esq * k + h *
	# 		p) * (c + c * j - k * q) * (k * p + esq * (h - s)) + 2 * esq * j *
	# 		jmod^2 * jsq * k * (c + c * j - k * q) * qmod^2 * (k * p + esq *
	# 		(h - s)) + jmod * jsq^2 * (c * esq - esq * k + h * p) * qmod *
	# 		(esq * h * k * q * qmod - j * jmod * k * (k * p + esq * (h - s)) -
	# 		j * jmod^2 * (c * esq - k) * s) + 2 * esq * jmod * jsq * k *
	# 		qmod^3 * (esq * h * k * q * qmod - j * jmod * k * (k * p + esq *
	# 		(h - s)) - j * jmod^2 * (c * esq - k) * s) + esq * h * jmod * jsq *
	# 		qmod^3 * (j * jmod * k * (c * esq - esq * k + h * p) - esq * (-1 +
	# 		h^2 - j) * q * qmod + j * jmod^2 * s * (-h + esq * s)) + jmod *
	# 		jsq^2 * qmod * (k * p + esq * (h - s)) * (j * jmod * k * (c * esq -
	# 		esq * k + h * p) - esq * (-1 + h^2 - j) * q * qmod + j * jmod^2 *
	# 		s * (-h + esq * s)) - jmod^2 * j^5 * (c + c * j - k * q) * qmod *
	# 		((h^2 - k^2) * p * qmod + c * esq * (h - esq * s) + k * (h * q +
	# 		esq * s)) - qmod * (-esq * h * jmod * jsq^2 * k * (-c * esq +
	# 		esq * k - h * p) * qmod^2 + esq^2 * h * jmod * jsq^2 * q * qmod^3 +
	# 		esq^2 * h * jmod * jsq * k^2 * q * qmod^3 + 2 * esq^2 * h * j^3 *
	# 		k^2 * q * qmod^3 + jmod^2 * j^5 * k * (-c * esq + esq * k - h * p) *
	# 		(k * p + esq * (h - s)) - esq * jmod^2 * j^5 * qmod^2 * (k * p +
	# 		esq * (h - s)) - esq * jmod * jsq^2 * k^2 * qmod^2 * (k * p + esq *
	# 		(h - s)) - jmod^3 * j^5 * (c * esq - k) * (c * esq - esq * k + h *
	# 		p) * s - c * jmod^3 * j^5 * (c * esq - k) * qmod * (-h + esq * s) +
	# 		jmod^3 * j^5 * qmod * s * ((h^2 - k^2) * qmod + esq * s * (-h + esq *
	# 		s)) - jmod^2 * j^5 * k * qmod * ((h^2 - k^2) * p * qmod + c * esq *
	# 		(h - esq * s) + k * (h * q + esq * s))))) * K) / (jmod^3 * j^5 *
	# 		(esq - esq * q)^2 * qmod^3))
	# end

	if dorder==[1, 0, 0, 0, 1, 0]
		func = ((jsq * (c - k + (h * p) / esq) * (c - (k * q) / jmod)) / qmod -
			(k * (c + c * j - k * q) * qmod) / jmod + jsq * qmod * (((-1 + h^2 -
			j) * q) / (jmod^2 * j) - (k * (c * esq - esq * k + h * p)) / (esq *
			jmod * qmod) - (s * (-h + esq * s)) / (esq * qmod))) / (j * qmod^2)
	end

	if dorder==[0, 0, 0, 0, 1, 0]
		func = (((k * -qmod * (c + c * j - k * q)) / jmod + (jsq * (c - k +
			(h * p) / esq) * (c - (k * q) / jmod)) / qmod + jsq * qmod * ((k *
			(c - k + (h * p) / esq)) / (jmod * -qmod) + ((-1 + h^2 - j) * q)  /
			(jmod^2 * j) - (s * (-(h / esq) + s)) / qmod)) * K) / (j * qmod^2)
	end

	if dorder==[0, 0, 0, 2, 0, 0]
		func = -(((esq^2 * jmod^2 * jsq^2 * (c + c * j - k * q) * qmod^4 -
			jmod^2 * jsq^3 * (c + c * j - k * q) * qmod * (2 * c * esq * k + esq^2 *
			(-c^2 - qmod) - k * (k * q - 2 * h * p * qmod)) - esq * h * jmod^2 * jsq^2 *
			(c + c * j - k * q) * qmod^2 * (k * p + esq * (h - s)) - jmod^2 * jsq^3 *
			(c + c * j - k * q) * (k * p + esq * (h - s))^2 - 3 * esq * h * j *
			jmod * jsq * qmod^3 * (esq * h * k * q * qmod - j * jmod * k * (k * p +
			esq * (h - s)) - j * jmod^2 * (c * esq - k) * s) + esq * h * jmod *
			jsq * qmod^2 * (esq * h * jmod * (c + c * j - k * q) * qmod^2 + jmod *
			jsq * (c + c * j - k * q) * (k * p + esq * (h - s)) + j * qmod * (esq *
			h * k * q * qmod - j * jmod * k * (k * p + esq * (h - s)) - j * jmod^2 *
			(c * esq - k) * s)) - 2 * jmod * jsq^2 * (k * p + esq * (h - s)) *
			(esq * h * jmod * (c + c * j - k * q) * qmod^2 + jmod * jsq * (c + c *
			j - k * q) * (k * p + esq * (h - s)) + j * qmod * (esq * h * k * q *
			qmod - j * jmod * k * (k * p + esq * (h - s)) - j * jmod^2 * (c * esq -
			k) * s)) + j * qmod * (c * jmod^3 * j^5 * (-c * esq + k)^2 * qmod +
			esq^2 * h^2 * jmod * jsq * k * q * qmod^3 + 2 * esq^2 * h^2 * j^3 * k *
			q * qmod^3 + esq^2 * jmod * jsq^2 * k * q * qmod^3 + jmod^2 * j^5 * k *
			qmod * (2 * c * esq * k + esq^2 * (-c^2 - qmod) - k * (k * q - 2 * h *
			p * qmod)) - 2 * esq * h * jmod * jsq^2 * k * qmod^2 * (k * p + esq *
			(h - s)) + jmod^2 * j^5 * k * (k * p + esq * (h - s))^2 - jmod^3 * j^5 *
			qmod * s * (2 * h * k * qmod + esq * (c * esq - k) * s) - jmod^3 * j^5 *
			(c * esq - k) * s * (-k * p + esq * (-h + s)))) * K) / (jmod^3 * j^5 *
			(esq - esq * q)^2 * qmod^3))
	end

	if dorder==[1, 0, 0, 1, 0, 0]
		func = (-((h * (c + c * j - k * q) * qmod) / jmod) + (jsq * (c - (k * q)
			/ jmod) * (-h - (k * p) / esq + s)) / qmod + (-jmod * jsq * k *
			(-k * p + s + j * s) + esq * (h * jmod * jsq * k - h * j * k * q * qmod +
			jmod * jsq * (c + c * j - k) * s)) / (esq * jmod^2)) / (j * qmod^2)
	end

	if dorder==[0, 0, 0, 1, 0, 0]
		func = (((h * -qmod * (c + c * j - k * q)) / jmod + (jsq * (c -
			(k * q) / jmod) * (-h - (k * p) / esq + s)) / qmod + jsq * qmod *
			(-((h * k * q) / (jmod^2 * j)) + ((c - k / esq) * s) / qmod + (k *
			(-h - (k * p) / esq + s)) / (jmod * -qmod))) * K) / (j * qmod^2)
	end

	if dorder==[2, 0, 0, 0, 0, 0]
		func = 0
	end

	if dorder==[1, 0, 0, 0, 0, 0]
		func = (j * (c - (k * q) / jmod)) / qmod
	end

	if dorder==[0, 0, 0, 0, 0, 0]
		func = (j * (c - (k * q) / jmod) * K) / qmod + γ
	end

	return float(func)

end


function kep_grad(K1::T, P1::T, M01::T, h1::T, k1::T, γ1::T, t1::T) where {T<:Real}

    nparms = 6
    grad = zeros(nparms)
    for i in 1:nparms
        dorder = zeros(Int64, nparms)
        dorder[i] = 1
        grad[i] = kep_deriv(K1, h1, k1, M01, γ1, P1, t1, dorder)
    end

    return grad

end


function kep_hess(K1::T, h1::T, k1::T, M01::T, γ1::T, P1::T, t1::T) where {T<:Real}

    nparms = 6
    hess = zeros(nparms, nparms)
    for i in 1:nparms
        for j in 1:nparms
            dorder = zeros(Int64, nparms)
            dorder[i] += 1
            dorder[j] += 1
            hess[i,j] = kep_deriv(K1, h1, k1, M01, γ1, P1, t1, dorder)
        end
    end

    return hess

end


kep_deriv(ks::kep_signal, t::Unitful.Time, dorder::Vector{<:Integer}) =
	kep_deriv(ks.K, ks.P, ks.M0, ks.h, ks.k, ks.γ, t, dorder)
