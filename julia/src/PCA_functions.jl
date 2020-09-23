# functions related to calculating the PCA scores of time series spectra
# using JLD2, FileIO
# using HDF5
using Distributions
light_speed_nu = ustrip(uconvert(u"m/s", 1u"c"))

"""
modified code shamelessly stolen from RvSpectraKitLearn.jl/src/deriv_spectra_simple.jl
Estimate the derivatives of a vector
"""
function calc_deriv_RVSKL(x::Vector{<:Real})
    @assert length(x)>=3
    dx = similar(x)
    dx[1] = x[2]-x[1]
    dx[end] = x[end]-x[end-1]
    for i in 2:(length(x)-1)
        dx[i] = (x[i+1]-x[i-1])/2
    end
    return dx
end


"""
modified code shamelessly stolen from RvSpectraKitLearn.jl/src/deriv_spectra_simple.jl
Function to estimate the derivative(s) of the mean spectrum
doppler_comp = λ * dF/dλ -> units of flux
"""
function calc_doppler_component_RVSKL(lambda::Vector{T}, flux::Vector{T}) where {T<:Real}
    @assert length(lambda) == length(flux)
    dlambdadpix = calc_deriv_RVSKL(lambda);
    dfluxdpix = calc_deriv_RVSKL(flux);
    return dfluxdpix .* (lambda ./ dlambdadpix)  # doppler basis
end

function calc_doppler_component_RVSKL(lambda::Vector{T}, flux::Matrix{T}) where {T<:Real}
    return calc_doppler_component_RVSKL(lambda, vec(mean(flux, dims=2)))
end


"""
modified code shamelessly stolen from RvSpectraKitLearn.jl/src/generalized_pca.jl
Compute the PCA component with the largest eigenvalue
X is data, r is vector of random numbers, s is preallocated memory; r && s  are of same length as each data point
"""
function compute_pca_component_RVSKL!(X::Matrix{T}, r::AbstractArray{T, 1}, s::Vector{T}; tol::Float64=1e-8, max_it::Int64=20) where {T<:Real}
	num_lambda = size(X, 1)
    num_spectra = size(X, 2)
    @assert length(r) == num_lambda
    #rand!(r)  # assume r is already randomized
    last_mag_s = 0.0
    for j in 1:max_it
		s[:] = zeros(T, num_lambda)
		for i in 1:num_spectra
			BLAS.axpy!(dot(view(X, :, i), r), view(X, :, i), s)  # s += dot(X[:,i],r)*X[:,i]
		end
		mag_s = norm(s)
		r[:]  = s / mag_s
		if abs(mag_s - last_mag_s) < (tol * mag_s); break end
		last_mag_s = mag_s
	end
	return r
end


"""
modified code shamelessly stolen from RvSpectraKitLearn.jl/src/generalized_pca.jl
Compute first num_components basis vectors for PCA, after subtracting projection onto fixed_comp
"""
function fit_gen_pca_rv_RVSKL(X::Matrix{T}, fixed_comp::Vector{T}; mu::Vector{T}=vec(mean(X, dims=2)), num_components::Integer=4, tol::Float64=1e-12, max_it::Int64=20) where {T<:Real}

	# initializing relevant quantities
	num_lambda = size(X, 1)
    num_spectra = size(X, 2)
    M = rand(T, (num_lambda, num_components))  # random initialization is part of algorithm (i.e., not zeros)
    s = zeros(T, num_lambda)  # pre-allocated memory for compute_pca_component
    scores = zeros(num_components, num_spectra)
	fracvar = zeros(num_components)

    Xtmp = X .- mu  # perform PCA after subtracting off mean
    totalvar = sum(abs2, Xtmp)

	# doppler component calculations
	M[:, 1] = fixed_comp  # Force fixed (i.e., Doppler) component to replace first PCA component
    fixed_comp_norm2 = sum(abs2, fixed_comp)
    for i in 1:num_spectra
        scores[1, i] = z = dot(view(Xtmp, :, i), fixed_comp) / fixed_comp_norm2  # Normalize differently, so scores are z (i.e., doppler shift)
	    Xtmp[:, i] -= z * fixed_comp
    end
	fracvar[1] = sum(abs2, Xtmp) / totalvar
    # println("# j = ", 1, " sum(abs2, Xtmp) = ", sum(abs2, Xtmp), " frac_var_remain= ", fracvar[1] )

	# remaining component calculations
    for j in 2:num_components
        compute_pca_component_RVSKL!(Xtmp, view(M, :, j), s, tol=tol, max_it=max_it)
	    for i in 1:num_spectra
			scores[j, i] = dot(view(Xtmp, :, i), view(M, :, j)) #/sum(abs2,view(M,:,j-1))
			Xtmp[:,i] .-= scores[j, i] * view(M, :, j)
		end
		fracvar[j] = sum(abs2,Xtmp)/totalvar
		# println("# j = ", j, " sum(abs2, Xtmp) = ", sum(abs2,Xtmp), " frac_var_remain= ", fracvar[j] )
	end

	# calculating radial velocities (in m/s) from redshifts
	rvs = light_speed_nu * scores[1, :]  # c * z

	return (mu, M, scores, fracvar, rvs)
end



function fit_gen_pca(X::Matrix{T}; mu::Vector{T}=vec(mean(X, dims=2)), num_components::Integer=4, tol::Float64=1e-12, max_it::Int64=20) where {T<:Real}

	# initializing relevant quantities
	num_lambda = size(X, 1)
    num_spectra = size(X, 2)
    M = rand(T, (num_lambda, num_components))  # random initialization is part of algorithm (i.e., not zeros)
    s = zeros(T, num_lambda)  # pre-allocated memory for compute_pca_component
    scores = zeros(num_components, num_spectra)
	fracvar = zeros(num_components)

    Xtmp = X .- mu  # perform PCA after subtracting off mean
    totalvar = sum(abs2, Xtmp)
	# remaining component calculations
    for j in 1:num_components
        compute_pca_component_RVSKL!(Xtmp, view(M, :, j), s, tol=tol, max_it=max_it)
	    for i in 1:num_spectra
			scores[j, i] = dot(view(Xtmp, :, i), view(M, :, j)) #/sum(abs2,view(M,:,j-1))
			Xtmp[:,i] .-= scores[j, i] * view(M, :, j)
		end
		fracvar[j] = sum(abs2,Xtmp)/totalvar
		# println("# j = ", j, " sum(abs2, Xtmp) = ", sum(abs2,Xtmp), " frac_var_remain= ", fracvar[j] )
	end

	return (mu, M, scores, fracvar)
end


function score_covariances(many_scores::Array{<:Real,3})
	n_out = size(many_scores, 2)
	n_meas = size(many_scores, 3)
	covariances = zeros(n_meas, n_out, n_out)
	for i in 1:n_meas
		covariances[i, :, :] = cov(many_scores[:, :, i]; dims=1)
	end
	return covariances
end


"""
Generate a noisy permutation of the data by drawing from a multivariate Gaussian
based on the covariance of many data draws
"""
function noisy_scores_from_samples(mean_scores::Matrix{T}, many_scores::Array{T,3}; rng::AbstractRNG=Random.GLOBAL_RNG) where {T<:Real}
	@assert size(mean_scores, 1) == size(many_scores, 2)  # amount of score dimensions
	@assert size(mean_scores, 2) == size(many_scores, 3)  # amount of time points
	noisy_scores = zeros(size(mean_scores))
	covariance_hold = zeros(size(many_scores, 2), size(many_scores, 2))
	for i in 1:size(mean_scores, 2)
		covariance_hold[:, :] = cov(many_scores[:, :, i]; dims=1)
	    noisy_scores[:, i] = rand(rng, MvNormal(mean_scores[:, i], covariance_hold))
	end
	return noisy_scores
end


"""
Generate a noisy permutation of the data by drawing from a multivariate Gaussian
based on the covariance of many data draws
"""
function noisy_scores_from_covariance(mean_scores::Matrix{T}, covariances::Array{T,3}; rng::AbstractRNG=Random.GLOBAL_RNG) where {T<:Real}
	@assert size(mean_scores, 1) == size(covariances, 2) == size(covariances, 3)  # amount of score dimensions
	@assert size(mean_scores, 2) == size(covariances, 1)  # amount of time points
	noisy_scores = zeros(size(mean_scores))
	for i in 1:size(mean_scores, 2)
	    noisy_scores[:, i] = rand(rng, MvNormal(mean_scores[:, i], covariances[i, :, :]))
	end
	return noisy_scores
end
