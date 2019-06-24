# functions related to calculating the PCA scores of time series spectra
using JLD2, FileIO
using HDF5
using Distributions


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
    return calc_doppler_component_simple(lambda, vec(mean(flux, dims=2)))
end


"""
modified code shamelessly stolen from RvSpectraKitLearn.jl/src/generalized_pca.jl
Compute the PCA component with the largest eigenvalue
X is data, r is vector of random numbers, s is preallocated memory; r && s  are of same length as each data point
"""
function compute_pca_component_RVSKL!(X::Matrix{T}, r::Vector{T}, s::Vector{T}; tol::Float64=1e-8, max_it::Int64=20) where {T<:Real}
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
        scores[1, i] = z = (dot(view(Xtmp, :, i), fixed_comp) / fixed_comp_norm2)  # Normalize differently, so scores are z (i.e., doppler shift)
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
	rvs = light_speed * scores[1, :]  # c * z

	return (mu, M, scores, fracvar, rvs)
end


function make_noisy_SOAP_spectra(time_series_spectra::Matrix{T}, λs::Vector{T}; SNR::Real=100, temperature::Real=5700) where {T<:Real}
	noisy_spectra = zero(time_series_spectra)
	plancks = planck.(λs, temperature)
	photons = u"h" * uconvert(u"m / s", (1)u"c") ./ ((λs)u"m" / 10 ^ 10)
	mask = findall(!iszero, time_series_spectra[:, 1])
	mu = vec(mean(time_series_spectra, dims=2))
	noises = sqrt.(strip_units.(plancks .* mu .* photons))
	norm = mean(noises[mask] ./ mu[mask])
	ratios = noises / (norm * 100)
	ratios[mask] ./= mu[mask]  # prevent NaNs from dividing by zero
	for i in 1:size(time_series_spectra, 2)
		# noises = sqrt.(strip_units.(plancks .* time_series_spectra[:, i] .* photons))
		# norm = mean(noises[mask] ./ time_series_spectra[:, i][mask])
		# ratios = norm * noises ./ time_series_spectra[:, i] ./ SNR
		noisy_spectra[:, i] = time_series_spectra[:, i] .* (1 .+ (ratios .* randn(length(λs))))
	end
	return noisy_spectra
end


"""
Generate a noisy permutation of the data by recalculating PCA components and scores
after adding a noise-to-signal ratio amount of Gaussian noise to each flux bin
"""
function noisy_scores_from_SOAP_spectra(time_series_spectra::Matrix{T}, λs::Vector{T}, M::Matrix{T}) where {T<:Real}
	num_components = size(M, 2)
	num_spectra = size(time_series_spectra, 2)
	noisy_scores = zeros(num_components, num_spectra)
	time_series_spectra_tmp = make_noisy_SOAP_spectra(time_series_spectra, λs)
	time_series_spectra_tmp .-= vec(mean(time_series_spectra_tmp, dims=2))
	fixed_comp_norm2 = sum(abs2, view(M, :, 1))
	for i in 1:num_spectra
		noisy_scores[1, i] = (dot(view(time_series_spectra_tmp, :, i), view(M, :, 1)) / fixed_comp_norm2)  # Normalize differently, so scores are z (i.e., doppler shift)
		time_series_spectra_tmp[:, i] -= noisy_scores[1, i] * view(M, :, 1)
	end
	for j in 2:num_components
		for i in 1:num_spectra
			noisy_scores[j, i] = dot(view(time_series_spectra_tmp, :, i), view(M, :, j)) #/sum(abs2,view(M,:,j-1))
			time_series_spectra_tmp[:, i] .-= noisy_scores[j, i] * view(M, :, j)
		end
	end
	return noisy_scores
end


"bootstrapping for errors in PCA scores. Takes ~28s per bootstrap on my computer"
function bootstrap_SOAP_errors(time_series_spectra::Matrix{T}, λs::Vector{T}, hdf5_filename::AbstractString; boot_amount::Integer=10) where {T<:Real}

    @load hdf5_filename * "_rv_data.jld2" M scores

	scores_mean = copy(scores)  # saved to ensure that the scores are paired with the proper rv_data

    num_lambda = size(time_series_spectra, 1)
    num_spectra = size(time_series_spectra, 2)

    num_components = size(M, 2)
    scores_tot_new = zeros(boot_amount, num_components, num_spectra)

    for k in 1:boot_amount
        scores = noisy_scores_from_SOAP_spectra(time_series_spectra, λs, M)
        scores_tot_new[k, :, :] = scores
    end

	close(fid)

	save_filename = hdf5_filename * "_bootstrap.jld2"

    if isfile(save_filename)
        @load save_filename scores_tot
        scores_tot = vcat(scores_tot, scores_tot_new)
    else
        scores_tot = scores_tot_new
    end

    error_ests = zeros(num_components, num_spectra)

	# est_point_error(a) = fit_mle(Normal, a).σ
    # std_uncorr(a) = std(a; corrected=false)

    for i in 1:num_components
		# # produce same results
        # error_ests[i, :] = mapslices(est_point_error, scores_tot[:, i, :]; dims=1)
        # error_ests[i,:] = mapslices(std_uncorr, scores_tot[:, i, :]; dims=1)

        error_ests[i,:] = mapslices(std, scores_tot[:, i, :]; dims=1)
    end

    @save save_filename scores_mean scores_tot error_ests
end


"""
Generate a noisy permutation of the data by drawing from a multivariate Gaussian
based on the covariance of many data draws
"""
function noisy_scores_from_covariance(mean_scores::Matrix{T}, many_scores::AbstractArray{T,3}) where {T<:Real}
	@assert size(mean_scores, 1) == size(many_scores, 2)  # amount of score dimensions
	@assert size(mean_scores, 2) == size(many_scores, 3)  # amount of time points
	noisy_scores = zeros(size(mean_scores))
	for i in 1:size(mean_scores, 2)
	    noisy_scores[:, i] = rand(MvNormal(mean_scores[:, i], cov(many_scores[:, :, i]; dims=1)))
	end
	return noisy_scores
end
