# SOAP_functions.jl


"""
Get flat, active SOAP spectra time series from HDF5 file, impose a Planck
distribution, and normalize so that the brightness stays the same.
"""
function prep_SOAP_spectra(fid::HDF5File)
	λs = fid["lambdas"][:]
	return normalize_columns_to_first_integral!(fid["active"][:, :] .* planck.(λs, 5700), λs), λs
end


function make_noisy_SOAP_spectra(time_series_spectra::Matrix{T}, λs::Vector{T}; SNR::Real=100, temperature::Real=5700) where {T<:Real}
	noisy_spectra = zero(time_series_spectra)
	photons = strip_units.(u"h" * uconvert(u"m / s", (1)u"c") ./ ((λs)u"m" / 10 ^ 10))
	mask = findall(!iszero, time_series_spectra[:, 1])
	for i in 1:size(time_series_spectra, 2)
		noises = sqrt.(time_series_spectra[:, i] .* photons)
	    normalization = mean(noises[mask] ./ time_series_spectra[mask, i])
	    ratios = noises[mask] ./ time_series_spectra[mask, i] / (normalization * SNR)
	    noisy_spectra[mask, i] = time_series_spectra[mask, i] .* (1 .+ (ratios .* randn(length(mask))))
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
function bootstrap_SOAP_errors(
	time_series_spectra::Matrix{T},
	λs::Vector{T},
	hdf5_filename::AbstractString;
	boot_amount::Integer=10
	) where {T<:Real}

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


function init_problem_definition(
	hdf5_filename::String;
	sub_sample::Integer=0,
	n_out::Integer=3,
	n_dif::Integer=3,
	save_prob_def::Bool=true,
	save_str::String="full",
	on_off::Real=0)

    assert_positive(n_out, n_dif)

    @load hdf5_filename * "_rv_data.jld2" phases scores
    @load hdf5_filename * "_bootstrap.jld2" scores_tot scores_mean error_ests
    @assert isapprox(scores, scores_mean)

    noisy_scores = noisy_scores_from_covariance(scores, scores_tot)

	phases_days = convert_SOAP_phases_to_days.(phases)
    inds = collect(1:size(noisy_scores, 2))
    if sub_sample !=0
		# if a on-off cadence is specified, remove all observations during the
		# off-phase and shift by a random phase
		if on_off != 0
			inds = findall(iseven, convert.(Int64, floor.((inds / on_off) .- rand())))
		end
		inds = sort(sample(inds, sub_sample; replace=false))
	end

    amount_of_measurements = length(inds)
    total_amount_of_measurements = amount_of_measurements * n_out

    # getting proper slice of data and converting to days
    x_obs = convert_SOAP_phases_to_days.(phases[inds])
    x_obs_units = u"d"
    y_obs_hold = noisy_scores[1:n_out, inds]
    measurement_noise_hold = error_ests[1:n_out, inds]
    y_obs_hold[1, :] *= light_speed  # convert scores from redshifts to radial velocities in m/s
    measurement_noise_hold[1, :] *= light_speed  # convert score errors from redshifts to radial velocities in m/s

    # rearranging the data into one column (not sure reshape() does what I want)
    # and normalizing the data (for numerical purposes)
    y_obs = zeros(total_amount_of_measurements)
    measurement_noise = zeros(total_amount_of_measurements)

    normals = ones(n_out)
    for i in 1:n_out
        y_obs[((i - 1) * amount_of_measurements + 1):(i * amount_of_measurements)] = y_obs_hold[i, :]
        measurement_noise[((i - 1) * amount_of_measurements + 1):(i * amount_of_measurements)] = measurement_noise_hold[i, :]
    end
    y_obs_units = u"m / s"

    # a0 = ones(n_out, n_dif) / 20
    a0 = zeros(n_out, n_dif)
    a0[1,1] = 0.03; a0[2,1] = 0.3; a0[1,2] = 0.3; a0[3,2] = 0.3; a0[2,3] = 0.075; a0  #  /= 20

    problem_def_base = init_problem_definition(n_dif, n_out, x_obs, x_obs_units, a0; y_obs=y_obs, y_obs_units=y_obs_units, normals=normals, noise=measurement_noise)

	if save_prob_def; @save hdf5_filename * "_problem_def_" * save_str * "_base.jld2" problem_def_base end

	return problem_def_base

    # kernel_function, num_kernel_hyperparameters = include("src/kernels/quasi_periodic_kernel.jl")
    # problem_def = init_problem_definition(kernel_function, num_kernel_hyperparameters, problem_def_base)
    # @save "jld2_files/" * hdf5_filename * "_problem_def_" * save_str * ".jld2" problem_def
end
