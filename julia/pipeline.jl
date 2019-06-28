# errors
# include("src/setup.jl")
include("src/all_functions.jl")

old_dir = pwd()
cd(@__DIR__)

#################################
# Importing from SOAP HDF5 file #
#################################

hdf5_loc = "C:/Users/chris/Downloads/res-1000-lambda-3923-6664-1years_1579spots_diffrot_id11.h5"
# hdf5_loc = "D:/Christian/Downloads/res-1000-lambda-3923-6664-0years_1spots_diffrot_id2.h5"

hdf5_filename = string(split(hdf5_loc,"/")[end])[1:end-3]
fid = h5open(hdf5_loc, "r")

@elapsed obs, λs = prep_SOAP_spectra(fid)

phases = fid["phases"][:]
quiet = fid["quiet"][:]

###########################
# Calculating RVs and PCA #
###########################

average_spectra = vec(mean(obs, dims=2))
doppler_comp = calc_doppler_component_RVSKL(λs, average_spectra)

# Compute first num_components (can be specified) basis vectors for PCA, after
# subtracting projection onto fixed_comp (the doppler component we passed)
# returns yhe doppler basis component, mean spectra, basis vectors, PCA scores,
# fractional variance remaining after each component is taken out, and the radial velocities
# M[:, 1] is not a unit vector, this is so scores[1, :] are actually redshifts
mu, M, scores, fracvar, rvs = @time fit_gen_pca_rv_RVSKL(obs, doppler_comp, mu=average_spectra, num_components=6)

@save "jld2_files/" * hdf5_filename * "_rv_data.jld2" λs phases quiet doppler_comp mu M scores fracvar rvs

##########################################
# Finding bootstrap estimates for errors #
##########################################

boot_amount = 10
for i in 1:1
    bootstrap_SOAP_errors(obs, λs, "jld2_files/" * hdf5_filename; boot_amount=boot_amount)
    println("Completed $(i * boot_amount) bootstraps resamples so far")
end

################################
# Creating problem definitions #
################################

@load "jld2_files/" * hdf5_filename * "_rv_data.jld2" phases scores
@load "jld2_files/" * hdf5_filename * "_bootstrap.jld2" scores_tot scores_mean error_ests
@assert isapprox(scores, scores_mean)

abs.(scores ./ error_ests)

noisy_scores = noisy_scores_from_covariance(scores, scores_tot)

# how many components you will use
n_out = 3
# how many differentiated versions of the original GP you will use
n_dif = 3

function init_problem_definition(inds, save_str)
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
    @save "jld2_files/" * hdf5_filename * "_problem_def_" * save_str * "_base.jld2" problem_def_base

    # kernel_function, num_kernel_hyperparameters = include("src/kernels/quasi_periodic_kernel.jl")
    # problem_def = init_problem_definition(kernel_function, num_kernel_hyperparameters, problem_def_base)
    # @save "jld2_files/" * hdf5_filename * "_problem_def_" * save_str * ".jld2" problem_def
end

inds = collect(1:size(noisy_scores, 2))
init_problem_definition(inds, "full")
init_problem_definition(sort(sample(inds, 70; replace=false)), "sample")

cd(old_dir)
