# functions related to calculating the PCA scores of time series spectra
using JLD2, FileIO
using HDF5
using Distributions


"""
modified code shamelessly stolen from RvSpectraKitLearn.jl/src/deriv_spectra_simple.jl
Estimate the derivatives of a vector
"""
function calc_deriv_RVSKL(x::AbstractArray{T,1}) where {T<:Real}
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
function calc_doppler_component_RVSKL(lambda::AbstractArray{T1,1}, flux::AbstractArray{T2,1}) where {T1<:Real, T2<:Real}
    @assert length(lambda) == length(flux)
    dlambdadpix = calc_deriv_RVSKL(lambda);
    dfluxdpix = calc_deriv_RVSKL(flux);
    return dfluxdpix .* (lambda ./ dlambdadpix)  # doppler basis
end

function calc_doppler_component_RVSKL(lambda::AbstractArray{T1,1}, flux::AbstractArray{T2,2}) where {T1<:Real, T2<:Real}
    return calc_doppler_component_simple(lambda, vec(mean(flux, dims=2)))
end


"""
modified code shamelessly stolen from RvSpectraKitLearn.jl/src/generalized_pca.jl
Compute the PCA component with the largest eigenvalue
X is data, r is vector of random numbers, s is preallocated memory; r & s  are of same length as each data point
"""
function compute_pca_component_RVSKL!(X::AbstractArray{T,2}, r::AbstractArray{T,1}, s::AbstractArray{T,1}; tol::Float64=1e-8, max_it::Int64=20) where {T<:Real}
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
		if abs(mag_s - last_mag_s) < (tol * mag_s) break end
		last_mag_s = mag_s
	end
	return r
end


"""
modified code shamelessly stolen from RvSpectraKitLearn.jl/src/generalized_pca.jl
Compute first num_components basis vectors for PCA, after subtracting projection onto fixed_comp
"""
function fit_gen_pca_rv_RVSKL(X::AbstractArray{T,2}, fixed_comp::AbstractArray{T,1}; mu::AbstractArray{T,1}=vec(mean(X, dims=2)), num_components::Integer=4, tol::Float64=1e-12, max_it::Int64=20) where {T<:Real}

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
    println("# j = ", 1, " sum(abs2, Xtmp) = ", sum(abs2, Xtmp), " frac_var_remain= ", fracvar[1] )

	# calculating radial velocities (in m/s) from redshifts
	rvs = 299792458 * scores[1, :]  # c * z

	# remaining component calculations
    for j in 2:num_components
        compute_pca_component_RVSKL!(Xtmp, view(M, :, j), s, tol=tol, max_it=max_it)
	    for i in 1:num_spectra
			scores[j, i] = dot(view(Xtmp, :, i), view(M, :, j)) #/sum(abs2,view(M,:,j-1))
			Xtmp[:,i] .-= scores[j, i] * view(M, :, j)
		end
		fracvar[j] = sum(abs2,Xtmp)/totalvar
		println("# j = ", j, " sum(abs2, Xtmp) = ", sum(abs2,Xtmp), " frac_var_remain= ", fracvar[j] )
	end

	return (mu, M, scores, fracvar, rvs)
end


"bootstrapping for errors in PCA scores. Takes ~28s per bootstrap on my computer"
function bootstrap_errors(time_series_spectra::AbstractArray{T,2}; boot_amount::Integer=10, save_filename::String="jld2_files/bootstrap.jld2") where {T<:Real}

    # @load "jld2_files/rv_data.jld2" lambda phases quiet doppler_comp mu M scores fracvar rvs
    @load "jld2_files/rv_data.jld2" M scores
    scores0 = copy(scores)

    SNR = 1e-2
    num_lambda = size(time_series_spectra, 1)
    num_spectra = size(time_series_spectra, 2)

    num_components = size(M, 2)
    scores_tot_new = zeros(boot_amount, num_components, num_spectra)

    # code adapted from fit_gen_pca_rv
    # https://github.com/eford/RvSpectraKitLearn.jl/blob/master/src/generalized_pca.jl
    for k in 1:boot_amount
        scores = zeros(num_components, num_spectra)
        time_series_spectra_tmp = time_series_spectra .* (1 .+ (SNR .* randn(size(time_series_spectra))))
        time_series_spectra_tmp .-= vec(mean(time_series_spectra_tmp, dims=2))
        fixed_comp_norm2 = sum(abs2, view(M, :, 1))
        for i in 1:num_spectra
            scores[1, i] = (dot(view(time_series_spectra_tmp, :, i), view(M, :, 1)) / fixed_comp_norm2)  # Normalize differently, so scores are z (i.e., doppler shift)
            time_series_spectra_tmp[:, i] -= scores[1, i] * view(M, :, 1)
        end
        for j in 2:num_components
            for i in 1:num_spectra
                scores[j, i] = dot(view(time_series_spectra_tmp, :, i), view(M, :, j)) #/sum(abs2,view(M,:,j-1))
                time_series_spectra_tmp[:, i] .-= scores[j, i] * view(M, :, j)
            end
        end
        scores_tot_new[k, :, :] = scores
    end

    if isfile(save_filename)
        @load save_filename scores_tot
        scores_tot = vcat(scores_tot, scores_tot_new)
    else
        scores_tot = scores_tot_new
    end

    error_ests = zeros(num_components, num_spectra)
    est_point_error(a::AbstractArray{T,1}) where {T<:Real} = fit_mle(Normal, a).σ

    for i in 1:num_components
        error_ests[i, :] = mapslices(est_point_error, scores_tot[:, i, :]; dims=1)
        # error_ests[i,:] = mapslices(std, scores_tot[:, i, :]; dims=1)
    end

    @save save_filename scores_tot error_ests
end
