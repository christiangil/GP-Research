# SOAP_functions.jl
using Unitful
using UnitfulAstro

"""
Get flat, active SOAP spectra time series from HDF5 file, impose a Planck
distribution, and normalize so that the brightness stays the same.
"""
function prep_SOAP_spectra(fid::HDF5File; return_quiet::Bool=false, T::Unitful.Temperature=5700u"K")
	λs = fid["lambdas"][:]u"nm"/10
	actives, normalization = normalize_columns_to_first_integral!(fid["active"][:, :] .* planck.(λs, T), ustrip.(λs); return_normalization=true)
	if return_quiet
		return actives, λs, fid["quiet"][:] .* planck.(λs, T) ./ normalization
	else
		return actives, λs
	end
end


function spectra_noise_levels(time_series_spectrum::Vector{T} ; photons::Real = ustrip.(u"h" * uconvert(u"m / s", 1u"c") ./ ((λ_nu[mask])u"m" / (10 ^ 10)))) where {T<:Real}

function make_noisy_SOAP_spectra(time_series_spectra::Matrix{T}, λs::Vector{T}; SNR::Real=100, sampling::Integer=1, return_noise::Bool=false) where {T<:Real}
	per_pixel_SNR = SNR / sqrt(sampling)
	noisy_spectra = zero(time_series_spectra)

	# a point 1e-300 in size was sneaking through with findall!(!iszero, ...)
	mask = time_series_spectra[:, 1] .> 1e-100

	photons = ustrip.(u"h" * uconvert(u"m / s", 1u"c") ./ ((λs[mask])u"m" / (10 ^ 10)))
	ratios = zeros(sum(mask))
	for i in 1:size(time_series_spectra, 2)
		time_series_spectrum = time_series_spectra[mask, i]
		noises = sqrt.(time_series_spectrum ./ photons)
		normalization = mean(noises ./ time_series_spectra[mask, i])
		ratios[:] = noises ./ time_series_spectrum ./ (normalization * per_pixel_SNR)
		noisy_spectra[mask, i] = time_series_spectrum .* (1 .+ (ratios .* randn(sum(mask))))
	end
	if return_noise
		noise = zeros(size(time_series_spectra, 1))
		noise[mask] = ratios
		return noisy_spectra, noise
	else
		return noisy_spectra
	end
end



"""
Generate a noisy permutation of the data by recalculating PCA scores
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


# "bootstrapping for errors in PCA scores. Takes ~28s per bootstrap on my computer"
# function bootstrap_SOAP_errors(
# 	time_series_spectra::Matrix{T},
# 	λs::Vector{T},
# 	hdf5_filename::AbstractString;
# 	boot_amount::Integer=10
# 	) where {T<:Real}
#
#     @load hdf5_filename * "_rv_data.jld2" M scores
#
# 	scores_mean = copy(scores)  # saved to ensure that the scores are paired with the proper rv_data
#
#     num_lambda = size(time_series_spectra, 1)
#     num_spectra = size(time_series_spectra, 2)
#
#     num_components = size(M, 2)
#     scores_tot_new = zeros(boot_amount, num_components, num_spectra)
#
#     for k in 1:boot_amount
#         scores = noisy_scores_from_SOAP_spectra(time_series_spectra, λs, M)
#         scores_tot_new[k, :, :] = scores
#     end
#
# 	close(fid)
#
# 	save_filename = hdf5_filename * "_bootstrap.jld2"
#
#     if isfile(save_filename)
#         @load save_filename scores_tot
#         scores_tot = vcat(scores_tot, scores_tot_new)
#     else
#         scores_tot = scores_tot_new
#     end
#
#     error_ests = zeros(num_components, num_spectra)
#
# 	# est_point_error(a) = fit_mle(Normal, a).σ
#     # std_uncorr(a) = std(a; corrected=false)
#
#     for i in 1:num_components
# 		# # produce same results
#         # error_ests[i, :] = mapslices(est_point_error, scores_tot[:, i, :]; dims=1)
#         # error_ests[i,:] = mapslices(std_uncorr, scores_tot[:, i, :]; dims=1)
#
#         error_ests[i,:] = mapslices(std, scores_tot[:, i, :]; dims=1)
#     end
#
#     @save save_filename scores_mean scores_tot error_ests
# end
