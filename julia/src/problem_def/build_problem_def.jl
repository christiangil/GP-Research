include("../all_functions.jl")

using JLD2, FileIO

use_sample=true

use_sample ? save_str = "sample" : save_str = "full"

old_dir = pwd()
cd(@__DIR__)

@load "../../jld2_files/rv_data.jld2" lambda phases quiet doppler_comp mu M scores fracvar rvs
@load "../../jld2_files/bootstrap.jld2" scores_tot scores_mean
@assert isapprox(scores, scores_mean)

noisy_scores = noisy_scores_from_covariance(scores, scores_tot)

noisy_scores[1, :] *= light_speed  # convert scores from redshifts to radial velocities in m/s

# how many components you will use
n_out = 3
# how many differentiated versions of the original GP you will use
n_dif = 3

# Setting up all of the data things
# how much of the data you want to use (on time domain)
use_sample ? start_ind = 100 : start_ind = 1
use_sample ? end_ind = 170 : end_ind = 730
amount_of_measurements = end_ind - start_ind + 1
total_amount_of_measurements = amount_of_measurements * n_out

# getting proper slice of data and converting to days
x_obs = convert_SOAP_phases_to_days.(phases[start_ind:end_ind])
x_obs_units = "Time (days)"
y_obs_hold = noisy_scores[1:n_out, start_ind:end_ind]
@load "../../jld2_files/bootstrap.jld2" error_ests
measurement_noise_hold = error_ests[1:n_out, start_ind:end_ind]
measurement_noise_hold[1, :] *= light_speed  # convert score errors from redshifts to radial velocities in m/s

# rearranging the data into one column (not sure reshape() does what I want)
# and normalizing the data (for numerical purposes)
y_obs = zeros(total_amount_of_measurements)
measurement_noise = zeros(total_amount_of_measurements)
normals = std(y_obs_hold, dims=2)'[:]
for i in 1:n_out
    y_obs[((i - 1) * amount_of_measurements + 1):(i * amount_of_measurements)] = y_obs_hold[i, :] / normals[i]
    measurement_noise[((i - 1) * amount_of_measurements + 1):(i * amount_of_measurements)] = measurement_noise_hold[i, :] / normals[i]
end
y_obs_units = "Normalized noisy RV + PCA scores"

# a0 = ones(n_out, n_dif) / 20
a0 = zeros(n_out, n_dif)
a0[1,1] = 0.03; a0[2,1] = 0.3; a0[1,2] = 0.3; a0[3,2] = 0.3; a0[2,3] = 0.075; a0  #  /= 20

problem_def_base = build_problem_definition(n_dif, n_out, x_obs, x_obs_units, y_obs, y_obs_units, measurement_noise, a0)
@save "../../jld2_files/problem_def_" * save_str * "_base.jld2" problem_def_base normals

kernel_function, num_kernel_hyperparameters = include("../kernels/quasi_periodic_kernel.jl")
problem_def = build_problem_definition(kernel_function, num_kernel_hyperparameters, problem_def_base)
@save "../../jld2_files/problem_def_" * save_str * ".jld2" problem_def normals

cd(old_dir)
