# Problem_definition_functions.jl
using SpecialFunctions
using LinearAlgebra
using JLD2, FileIO
using Dates
using Unitful
using UnitfulAstro
using Random


"""
A structure that holds all of the relevant information for constructing the
model used in the Jones et al. 2017+ paper (https://arxiv.org/pdf/1711.01318.pdf).
"""
struct Jones_problem_definition{T1<:Real, T2<:Integer}
	kernel::Function  # kernel function
	n_kern_hyper::T2  # amount of hyperparameters for the kernel function
	n_dif::T2  # amount of times you are differenting the base kernel
	n_out::T2  # amount of scores you are jointly modelling
	x_obs::Vector{T1} # the observation times/phases
	time::Vector{T} where T<:Unitful.Time
	time_unit::Unitful.FreeUnits  # the units of x_obs
	y_obs::Vector{T1}  # the flattened, observed data
	rv::Vector{T} where T<:Unitful.Velocity
	rv_unit::Unitful.FreeUnits  # the units of the RV section of y_obs
	noise::Vector{T1}  # the measurement noise at all observations
	rv_noise::Vector{T} where T<:Unitful.Velocity # the measurement noise at all observations
	normals::Vector{T1}  # the normalization of each section of y_obs
	a0::Matrix{T1}  # the meta kernel coefficients
	non_zero_hyper_inds::Vector{T2}  # the indices of the non-zero hyperparameters
	# The powers that each a0 coefficient
	# is taken to for each part of the matrix construction
	# used for constructing differentiated versions of the kernel
	coeff_orders::AbstractArray{T2,6}
	coeff_coeffs::AbstractArray{T2,4}
	covariance::Array{T1, 3}  # the measurement covariance at all observations
	has_covariance::Bool

	function Jones_problem_definition(
		kernel::Function,
		n_kern_hyper::T,
		hdf5_filenames::Vector{String};
		sub_sample::T=0,
		n_out::T=3,
		n_dif::T=3,
		on_off::Unitful.Time=0u"d",
		rng::AbstractRNG=Random.GLOBAL_RNG
		) where T<:Integer

		assert_positive(n_out, n_dif)

		time_unit = u"d"
		if length(hdf5_filenames) > 1
			time = Vector{typeof(1. * time_unit)}()
			noisy_scores = zeros(n_out, 0)
			selected_error_ests = zeros(n_out, 0)
			selected_covariances = zeros(0, n_out, n_out)
		end
		for i in 1:length(hdf5_filenames)
			hdf5_filename = hdf5_filenames[i]
			@load hdf5_filename * "_rv_data.jld2" phases scores
			@load hdf5_filename * "_bootstrap.jld2" scores_tot scores_mean error_ests
			@assert isapprox(scores, scores_mean)
			len_phase = length(phases)
			@assert len_phase == size(scores, 2)
			@assert size(scores) == size(error_ests)

			if length(hdf5_filenames) == 1
				selected_covariances = score_covariances(scores_tot[:, 1:n_out, :])
				selected_error_ests = error_ests[1:n_out, :]
				noisy_scores = noisy_scores_from_covariance(scores[1:n_out,:], selected_covariances; rng=rng)
				time = convert_SOAP_phases.(time_unit, phases)
			else
				fraction_unobservable = 1 / 4
				shift = Int(floor(rand(rng) * len_phase * fraction_unobservable))
				kept_inds = shift + 1 : shift + Int(floor(len_phase * (1 - fraction_unobservable)))  # skip 3 months of the year. helps join together disparate simulations
				println(kept_inds)
				selected_covariances_holder = score_covariances(scores_tot[:, 1:n_out, kept_inds])
				noisy_scores = cat(noisy_scores, noisy_scores_from_covariance(scores[1:n_out, kept_inds], selected_covariances_holder; rng=rng); dims=2)
				selected_error_ests = cat(selected_error_ests, error_ests[1:n_out, kept_inds]; dims=2)
				selected_covariances = cat(selected_covariances, selected_covariances_holder; dims=1)
				all_times = convert_SOAP_phases.(time_unit, phases)
				i == 1 ? reshift = 0 * time_unit : reshift = fraction_unobservable * (all_times[end] - all_times[1]) + time[end]
				append!(time, (reshift - all_times[kept_inds[1]]) .+ all_times[kept_inds])
			end
		end
		inds = collect(1:size(noisy_scores, 2))
		if sub_sample !=0
			# if a on-off cadence is specified, remove all observations during the
			# off-phase and shift by a random phase
			if on_off != 0u"d"
				inds = findall(iseven, convert.(Int64, floor.((time / on_off) .- rand(rng))))
				inds = sort(sample(rng, inds, sub_sample; replace=false))
				println(inds[1:10])
			else
				shift = convert(Int64, floor(rand(rng) * (length(inds) - sub_sample)))
				inds = collect((1 + shift):(sub_sample + shift))
			end
		end

		n_meas = length(inds)
		total_n_meas = n_meas * n_out

		# getting proper slice of data and converting to days

		selected_covariances = selected_covariances[inds, :, :]
		selected_covariances[:, 1, :] *= light_speed
		selected_covariances[:, :, 1] *= light_speed
		time = time[inds]
		time .-= mean(time)  # minimizing period derivatives
		x_obs = ustrip.(time)
		y_obs_hold = noisy_scores[:, inds]
		measurement_noise_hold = selected_error_ests[:, inds]
		y_obs_hold[1, :] *= light_speed  # convert scores from redshifts to radial velocities in m/s
		measurement_noise_hold[1, :] *= light_speed  # convert score errors from redshifts to radial velocities in m/s

		# rearranging the data into one column (not sure reshape() does what I want)
		# and normalizing the data (for numerical purposes)
		y_obs = zeros(total_n_meas)
		measurement_noise = zeros(total_n_meas)

		normals = ones(n_out)
		for i in 1:n_out
			# y_obs[((i - 1) * n_meas + 1):(i * n_meas)] = y_obs_hold[i, :]
			# measurement_noise[((i - 1) * n_meas + 1):(i * n_meas)] = measurement_noise_hold[i, :]
			y_obs[i:n_out:end] = y_obs_hold[i, :]
			measurement_noise[i:n_out:end] = measurement_noise_hold[i, :]
		end
		rv_unit = u"m/s"
        rv = y_obs[1:n_out:end] * rv_unit
        rv_noise = measurement_noise[1:n_out:end] * rv_unit
		# y_obs *= rvs_unit
		# measurement_noise *= rvs_unit

		# a0 = ones(n_out, n_dif) / 20
		a0 = zeros(n_out, n_dif)
		if n_dif == n_out == 3
			a0[1,1] = 0.03; a0[2,1] = 0.3; a0[1,2] = 0.3; a0[3,2] = 0.3; a0[2,3] = 0.075;
		elseif n_dif == 3 && n_out == 2
			a0[1,1] = 0.03; a0[2,1] = 0.3; a0[1,2] = 0.3; a0[2,3] = 0.075;
		elseif n_dif == 3 && n_out == 1
			a0[1,1] = 0.03; a0[1,2] = 0.3;
		else
			a0[:,:] = ones(n_out, n_dif) / 20
		end

		return Jones_problem_definition(kernel, n_kern_hyper, n_dif, n_out, x_obs, time, time_unit, y_obs, rv, rv_unit, measurement_noise, rv_noise, normals, a0; covariance=selected_covariances)
	end
	Jones_problem_definition(
		kernel::Function,
		n_kern_hyper::T,
		hdf5_filename::String;
		sub_sample::T=0,
		n_out::T=3,
		n_dif::T=3,
		on_off::Unitful.Time=0u"d",
		rng::AbstractRNG=Random.GLOBAL_RNG
		) where T<: Real = Jones_problem_definition(kernel, n_kern_hyper, [hdf5_filename]; sub_sample=sub_sample, n_out=n_out, n_dif=n_dif, on_off=on_off, rng=rng)
	function Jones_problem_definition(kernel, n_kern_hyper, n_dif, n_out, x_obs, time, time_unit, y_obs, rv, rv_unit, measurement_noise, rv_noise, normals, a0; covariance=zeros(length(x_obs), n_out, n_out), non_zero_hyper_inds=append!(findall(!iszero, collect(Iterators.flatten(a0))), collect(1:n_kern_hyper) .+ length(a0)))
		coeff_orders, coeff_coeffs = coefficient_orders(n_out, n_dif, a=a0)
		has_covariance = (covariance != zeros(length(x_obs), n_out, n_out))
		return Jones_problem_definition(kernel, n_kern_hyper, n_dif, n_out, x_obs, time, time_unit, y_obs, rv, rv_unit, measurement_noise, rv_noise, normals, a0, non_zero_hyper_inds, coeff_orders, coeff_coeffs, covariance, has_covariance)
	end
	function Jones_problem_definition(
		kernel::Function,  # kernel function
		n_kern_hyper::T2,  # amount of hyperparameters for the kernel function
		n_dif::T2,  # amount of times you are differenting the base kernel
		n_out::T2,  # amount of scores you are jointly modelling
		x_obs::Vector{T1}, # the observation times/phases
		time::Vector{T} where T<:Unitful.Time,
		time_unit::Unitful.FreeUnits,  # the units of x_obs
		y_obs::Vector{T1},  # the flattened, observed data
		rv::Vector{T} where T<:Unitful.Velocity,
		rv_unit::Unitful.FreeUnits,  # the units of the RV section of y_obs
		noise::Vector{T1},  # the measurement noise at all observations
		rv_noise::Vector{T} where T<:Unitful.Velocity, # the measurement noise at all observations
		normals::Vector{T1},  # the normalization of each section of y_obs
		a0::Matrix{T1},
		non_zero_hyper_inds::Vector{T2},
		coeff_orders::Array{T2,6},
		coeff_coeffs::Array{T2,4},
		covariance::Array{T1, 3},
		has_covariance::Bool) where {T1<:Real, T2<:Integer}

		@assert isfinite(kernel(ones(num_kernel_hyperparameters), randn(); dorder=zeros(Int64, 2 + num_kernel_hyperparameters)))  # make sure the kernel is valid by testing a sample input
		@assert n_dif>0
		@assert n_out>0
		# @assert dimension(time_unit) == dimension(u"s")
		# @assert dimension(rvs_unit) == dimension(u"m / s")
		n_meas = length(x_obs)
		@assert (n_meas * n_out) == length(y_obs) == length(noise)
		@assert n_meas == length(rv) == length(rv_noise) == size(covariance, 1)
		@assert time_unit == unit(time[1])
		@assert rv_unit == unit(rv[1]) == unit(rv_noise[1])
		@assert length(normals) == n_out
		@assert size(a0) == (n_out, n_dif)
		@assert size(coeff_orders) == (n_out, n_out, n_dif, n_dif, n_out, n_dif)  # maybe unnecessary due to the fact that we construct it
		@assert size(coeff_coeffs) == (n_out, n_out, n_dif, n_dif)  # maybe unnecessary due to the fact that we construct it
		@assert length(non_zero_hyper_inds) == length(findall(!iszero, collect(Iterators.flatten(a0)))) + n_kern_hyper
		@assert n_out == size(covariance, 2) == size(covariance, 3)
		@assert (covariance != zeros(n_meas, n_out, n_out)) == has_covariance

		return new{typeof(x_obs[1]),typeof(n_kern_hyper)}(kernel, n_kern_hyper, n_dif, n_out, x_obs, time, time_unit, y_obs, rv, rv_unit, noise, rv_noise, normals, a0, non_zero_hyper_inds, coeff_orders, coeff_coeffs, covariance, has_covariance)
	end
end


function normalize_problem_definition!(prob_def::Jones_problem_definition)
	n_obs = length(prob_def.x_obs)
	renorms = ones(prob_def.n_out)
	for i in 1:prob_def.n_out
		inds = (i:prob_def.n_out:length(prob_def.y_obs))
		prob_def.y_obs[inds] .-= mean(prob_def.y_obs[inds])
		renorms[i] = std(prob_def.y_obs[inds])
		# println(renorm)
		prob_def.normals[i] *= renorms[i]
		prob_def.y_obs[inds] /= renorms[i]
		prob_def.noise[inds] /= renorms[i]
	end
	if prob_def.has_covariance
		renorm_mat = renorms .* transpose(renorms)
		for i in 1:n_obs
			prob_def.covariance[i, :, :] ./= renorm_mat
		end
	end
end
