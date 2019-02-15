#adding in custom functions
include("../src/all_functions.jl")

# can use this if you want to replicate results
# srand(1234)

# loading in data
using JLD2, FileIO
# @load "jld2_files/sunspot_data.jld2" lambda phases quiet
# @load "jld2_files/rv_data.jld2" doppler_comp genpca_out rvs_out
# mu, M, scores = genpca_out
# scores[:, 1] ./ 3e8
# scores = scores'
#
# # how many components you will use
# n_out = 3
# # how many differentiated versions of the original GP you will use
# n_dif = 3
#
# # Setting up all of the data things
# # how much of the data you want to use (on time domain)
# start_ind = 100
# end_ind = 170    # <= 730 70 datapoints is like 5 weeks
# amount_of_measurements = end_ind - start_ind + 1
# total_amount_of_measurements = amount_of_measurements * n_out
#
# # getting proper slice of data
# x_obs = phases[start_ind:end_ind]
# y_obs_hold = scores[1:n_out, start_ind:end_ind]
# @load "jld2_files/bootstrap.jld2" error_ests
# measurement_noise_hold = error_ests[1:n_out, start_ind:end_ind]
#
# # rearranging the data into one column (not sure reshape() does what I want)
# # and normalizing the data (for numerical purposes)
# y_obs = zeros(total_amount_of_measurements)
# measurement_noise = zeros(total_amount_of_measurements)
# normals = mean(abs.(y_obs_hold), dims=2)'[:]
# for i in 1:n_out
#     y_obs[((i - 1) * amount_of_measurements + 1):(i * amount_of_measurements)] = y_obs_hold[i, :] / normals[i]
#     measurement_noise[((i - 1) * amount_of_measurements + 1):(i * amount_of_measurements)] = measurement_noise_hold[i, :] / normals[i]
# end
#
# # # setting noise to 10% of max measurements
# for i in 1:n_out
#     measurement_noise[((i - 1) * amount_of_measurements + 1):(i * amount_of_measurements)] .= 0.10 * maximum(abs.(y_obs[i, :]))
# end
#
# a0 = zeros(n_out, n_dif)
# a0[1,1] = -0.53477; a0[2,1] = -1.54269; a0[1,2] = -1.96109; a0[3,2] = -2.08949; a0[2,3] = 0.170251; a0    #    /= 20
#
# num_kernel_hyperparameters = include_kernel("Quasi_periodic_kernel")    # sets correct num_kernel_hyperparameters
# problem_def_528 = build_problem_definition(Quasi_periodic_kernel, num_kernel_hyperparameters, n_dif, n_out, x_obs, y_obs, measurement_noise, a0)
# @save "jld2_files/problem_def_528.jld2" problem_def_528

include_kernel("Quasi_periodic_kernel")
@load "jld2_files/problem_def_528.jld2" problem_def_528
problem_definition = problem_def_528

##############################################################################

# kernel hyper parameters
kernel_lengths = [0.583594, 1.83475, 2.58466]
total_hyperparameters = append!(collect(Iterators.flatten(problem_definition.a0)), kernel_lengths)

# how finely to sample the domain
amount_of_samp_points = 50

x_samp = sort(minimum(problem_definition.x_obs) .+ (maximum(problem_definition.x_obs) - minimum(problem_definition.x_obs)) .* rand(amount_of_samp_points))

# total amount of output points
amount_of_total_samp_points = amount_of_samp_points * problem_definition.n_out

# Finding how correlated the sampled inputs are to each other
# (aka getting the covariance matrix by evaluating the kernel function at all
# pairs of points)
K_samp = covariance(problem_definition, x_samp, x_samp, total_hyperparameters)
L_samp = ridge_chol(K_samp).L
fake_data = (L_samp * randn(amount_of_total_samp_points))

P = 6u"d"
m_star = 1u"Msun"
m_planet = 1u"Mjup"
times = convert_phases_to_seconds.(x_samp)
planet_rvs = rv.(times, P, m_star, m_planet)
fake_data[1:amount_of_samp_points] += planet_rvs

span_x = x_samp[end] - x_samp[1]
# Nyquist frequency, named after electronic engineer Harry Nyquist, is half of
# the sampling rate of a discrete signal processing system
# (https://en.wikipedia.org/wiki/Nyquist_frequency)
nyquist_samp = span_x / amount_of_samp_points
# if the frequency locations are unknown, then it is necessary to sample at
# least at twice the Nyquist criteria; in other words, you must pay at least a
# factor of 2 for not knowing the location of the spectrum. Note that minimum
# sampling requirements do not necessarily guarantee stability.
# https://en.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem#Nonuniform_sampling
uneven_nyquist_samp = nyquist_samp / 2
period_grid = collect(uneven_nyquist_samp:uneven_nyquist_samp:(span_x / 2))

test = rand()
ϕ(test, 2.; e=0.01, iter=true)
ϕ(test, 2.; e=0.01)
