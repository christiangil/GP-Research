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
    x_obs_units::Unitful.FreeUnits  # the units of x_obs
    y_obs::Vector{T1}  # the flattened, observed data
    y_obs_units::Unitful.FreeUnits  # the units of the RV section of y_obs
    noise::Vector{T1}  # the measurement noise at all observations
    normals::Vector{T1}  # the normalization of each section of y_obs
    a0::Matrix{T1}  # the meta kernel coefficients
    # The powers that each a0 coefficient
    # is taken to for each part of the matrix construction
    # used for constructing differentiated versions of the kernel
    coeff_orders::AbstractArray{T2,6}
    coeff_coeffs::AbstractArray{T2,4}
end


"Jones_problem_definition without kernel information"
struct Jones_problem_definition_base{T1<:Real, T2<:Integer}
    n_dif::Integer  # amount of times you are differenting the base kernel
    n_out::Integer  # amount of scores you are jointly modelling
    x_obs::Vector{T1} # the observation times/phases
    x_obs_units::Unitful.FreeUnits  # the units of x_bs
    y_obs::Vector{T1}  # the flattened, observed data
    y_obs_units::Unitful.FreeUnits  # the units of the RV section of y_obs
    noise::Vector{T1}  # the measurement noise at all observations
    normals::Vector{T1}  # the normalization of each section of y_obs
    a0::Matrix{T1}  # the meta kernel coefficients
    # The powers that each a0 coefficient
    # is taken to for each part of the matrix construction
    # used for constructing differentiated versions of the kernel
    coeff_orders::AbstractArray{T2,6}
    coeff_coeffs::AbstractArray{T2,4}
end


"Ensure that the passed problem definition parameters are what we expect them to be"
function check_problem_definition(
    n_dif::Integer,
    n_out::Integer,
    x_obs::Vector{T1},
    x_obs_units::Unitful.FreeUnits,
    y_obs::Vector{T1},
    y_obs_units::Unitful.FreeUnits,
    noise::Vector{T1},
    normals::Vector{T1},
    a0::Matrix{T1},
    coeff_orders::AbstractArray{T2,6},
    coeff_coeffs::AbstractArray{T2,4}
    ) where {T1<:Real, T2<:Integer}

    @assert n_dif>0
    @assert n_out>0
    @assert dimension(x_obs_units) == dimension(u"s")
    @assert dimension(y_obs_units) == dimension(u"m / s")
    @assert (length(x_obs) * n_out) == length(y_obs)
    @assert length(y_obs) == length(noise)
    @assert length(normals) == n_out
    @assert size(a0) == (n_out, n_dif)
    @assert size(coeff_orders) == (n_out, n_out, n_dif, n_dif, n_out, n_dif)  # maybe unnecessary due to the fact that we construct it
    @assert size(coeff_coeffs) == (n_out, n_out, n_dif, n_dif)  # maybe unnecessary due to the fact that we construct it
end


"Ensure that Jones_problem_definition_base is constructed correctly"
function init_problem_definition(
    n_dif::Integer,
    n_out::Integer,
    x_obs::Vector{T1},
    x_obs_units::Unitful.FreeUnits,
    y_obs::Vector{T1},
    y_obs_units::Unitful.FreeUnits,
    noise::Vector{T1},
    normals::Vector{T1},
    a0::Matrix{T1},
    coeff_orders::AbstractArray{T2,6},
    coeff_coeffs::AbstractArray{T2,4},
    ) where {T1<:Real, T2<:Integer}

    check_problem_definition(n_dif, n_out, x_obs, x_obs_units, y_obs, y_obs_units, noise, normals, a0, coeff_orders, coeff_coeffs)
    return Jones_problem_definition_base(n_dif, n_out, x_obs, x_obs_units, y_obs, y_obs_units, noise, normals, a0, coeff_orders, coeff_coeffs)
end

"Calculate the coeffficient orders for Jones_problem_definition_base construction if they weren't passed"
function init_problem_definition(
    n_dif::Integer,
    n_out::Integer,
    x_obs::Vector{T},
    x_obs_units::Unitful.FreeUnits,
    a0::Matrix{T};
    y_obs::Vector{T}=zeros(length(x_obs) * n_out),
    y_obs_units::Unitful.FreeUnits=u"m / s",
    noise::Vector{T}=zeros(length(x_obs) * n_out),
    normals::Vector{T}=ones(n_out)
    ) where {T<:Real}

    coeff_orders, coeff_coeffs = coefficient_orders(n_out, n_dif, a=a0)
    return init_problem_definition(n_dif, n_out, x_obs, x_obs_units, y_obs, y_obs_units, noise, normals, a0, coeff_orders, coeff_coeffs)
end

"Construct Jones_problem_definition by adding kernel information to Jones_problem_definition_base"
function init_problem_definition(
    kernel_func::Function,
    num_kernel_hyperparameters::Integer,
    prob_def_base::Jones_problem_definition_base)

    @assert isfinite(kernel_func(ones(num_kernel_hyperparameters), randn(); dorder=zeros(Int64, 2 + num_kernel_hyperparameters)))  # make sure the kernel is valid by testing a sample input
    check_problem_definition(prob_def_base.n_dif, prob_def_base.n_out, prob_def_base.x_obs, prob_def_base.x_obs_units, prob_def_base.y_obs, prob_def_base.y_obs_units, prob_def_base.noise, prob_def_base.normals, prob_def_base.a0, prob_def_base.coeff_orders, prob_def_base.coeff_coeffs)  # might be unnecessary
    return Jones_problem_definition(kernel_func, num_kernel_hyperparameters, prob_def_base.n_dif, prob_def_base.n_out, prob_def_base.x_obs, prob_def_base.x_obs_units, prob_def_base.y_obs, prob_def_base.y_obs_units, prob_def_base.noise, prob_def_base.normals, prob_def_base.a0, prob_def_base.coeff_orders, prob_def_base.coeff_coeffs)
end


function init_problem_definition(
	hdf5_filename::String;
	sub_sample::Integer=0,
	n_out::Integer=3,
	n_dif::Integer=3,
	save_prob_def::Bool=true,
	save_str::String="full",
	on_off::Real=0,
	rng::AbstractRNG=Random.GLOBAL_RNG)

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
			inds = findall(iseven, convert.(Int64, floor.((phases_days / on_off) .- rand(rng))))
			inds = sort(sample(rng, inds, sub_sample; replace=false))
			println(inds[1:10])
		else
			shift = convert(Int64, floor(rand() * (length(inds) - sub_sample)))
			inds = collect((1 + shift):(sub_sample + shift))
		end
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
	if n_dif == n_out == 3
		a0[1,1] = 0.03; a0[2,1] = 0.3; a0[1,2] = 0.3; a0[3,2] = 0.3; a0[2,3] = 0.075;
	else
		a0[:,:] = ones(n_out, n_dif) / 20
	end

    problem_def_base = init_problem_definition(n_dif, n_out, x_obs, x_obs_units, a0; y_obs=y_obs, y_obs_units=y_obs_units, normals=normals, noise=measurement_noise)

	if save_prob_def; @save hdf5_filename * "_problem_def_" * save_str * "_base.jld2" problem_def_base end

	return problem_def_base

    # kernel_function, num_kernel_hyperparameters = include("src/kernels/quasi_periodic_kernel.jl")
    # problem_def = init_problem_definition(kernel_function, num_kernel_hyperparameters, problem_def_base)
    # @save "jld2_files/" * hdf5_filename * "_problem_def_" * save_str * ".jld2" problem_def
end


# function init_problem_definition(
# 	rvs::Vector{Real},
# 	noises::Vector{Real},
# 	rv_units::Unitful.FreeUnits,
# 	phases::Vector{Real},
# 	phase_units::Unitful.FreeUnits;
# 	n_dif::Integer=3,
# 	save_str::String="full",
# 	rng::AbstractRNG=Random.GLOBAL_RNG)
#
#     assert_positive(n_out, n_dif)
#
# 	phases_days = convert_and_strip_units("days", phases * phase_units)
#
#     amount_of_measurements = length(rvs)
#
#     # getting proper slice of data and converting to days
#     x_obs = convert_SOAP_phases_to_days.(phases[inds])
#     x_obs_units = u"d"
#     y_obs_hold = noisy_scores[1:n_out, inds]
#     measurement_noise_hold = error_ests[1:n_out, inds]
#     y_obs_hold[1, :] *= light_speed  # convert scores from redshifts to radial velocities in m/s
#     measurement_noise_hold[1, :] *= light_speed  # convert score errors from redshifts to radial velocities in m/s
#
#     # rearranging the data into one column (not sure reshape() does what I want)
#     # and normalizing the data (for numerical purposes)
#     y_obs = zeros(amount_of_measurements)
#     measurement_noise = zeros(amount_of_measurements)
#
#     normals = ones(n_out)
#     for i in 1:n_out
#         y_obs[((i - 1) * amount_of_measurements + 1):(i * amount_of_measurements)] = y_obs_hold[i, :]
#         measurement_noise[((i - 1) * amount_of_measurements + 1):(i * amount_of_measurements)] = measurement_noise_hold[i, :]
#     end
#     y_obs_units = u"m / s"
#
#     # a0 = ones(n_out, n_dif) / 20
#     a0 = zeros(n_out, n_dif)
# 	a0[1,1] = 0.03;
# 	a0[2,1] = 0.3; a0[1,2] = 0.3; a0[3,2] = 0.3; a0[2,3] = 0.075; a0
#
#     problem_def_base = init_problem_definition(n_dif, n_out, x_obs, x_obs_units, a0; y_obs=y_obs, y_obs_units=y_obs_units, normals=normals, noise=measurement_noise)
#
# 	if save_prob_def; @save hdf5_filename * "_problem_def_" * save_str * "_base.jld2" problem_def_base end
#
# 	return problem_def_base
#
#     # kernel_function, num_kernel_hyperparameters = include("src/kernels/quasi_periodic_kernel.jl")
#     # problem_def = init_problem_definition(kernel_function, num_kernel_hyperparameters, problem_def_base)
#     # @save "jld2_files/" * hdf5_filename * "_problem_def_" * save_str * ".jld2" problem_def
# end


function normalize_problem_definition!(prob_def::Jones_problem_definition)
    n_obs = length(prob_def.x_obs)
    for i in 1:prob_def.n_out
        inds = 1 + (i - 1) * n_obs : i * n_obs
        prob_def.y_obs[inds] .-= mean(prob_def.y_obs[inds])
		renorm = std(prob_def.y_obs[inds])
		# println(renorm)
        prob_def.normals[i] *= renorm
        prob_def.y_obs[inds] /= renorm
        prob_def.noise[inds] /= renorm
    end
end
