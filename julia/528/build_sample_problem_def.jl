using JLD2, FileIO

@load "jld2_files/sunspot_data.jld2" lambda phases quiet
@load "jld2_files/rv_data.jld2" doppler_comp genpca_out rvs_out
mu, M, scores = genpca_out
scores[:, 1] .* 3e8
scores = scores'

# how many components you will use
n_out = 3
# how many differentiated versions of the original GP you will use
n_dif = 3

# Setting up all of the data things
# how much of the data you want to use (on time domain)
start_ind = 100
end_ind = 170    # <= 730 70 datapoints is like 5 weeks
amount_of_measurements = end_ind - start_ind + 1
total_amount_of_measurements = amount_of_measurements * n_out

# getting proper slice of data
x_obs = phases[start_ind:end_ind]
y_obs_hold = scores[1:n_out, start_ind:end_ind]
@load "jld2_files/bootstrap.jld2" error_ests
measurement_noise_hold = error_ests[1:n_out, start_ind:end_ind]

# rearranging the data into one column (not sure reshape() does what I want)
# and normalizing the data (for numerical purposes)
y_obs = zeros(total_amount_of_measurements)
measurement_noise = zeros(total_amount_of_measurements)
normals = mean(abs.(y_obs_hold), dims=2)'[:]
for i in 1:n_out
    y_obs[((i - 1) * amount_of_measurements + 1):(i * amount_of_measurements)] = y_obs_hold[i, :] / normals[i]
    # measurement_noise[((i - 1) * amount_of_measurements + 1):(i * amount_of_measurements)] = measurement_noise_hold[i, :] / normals[i]
end

# setting noise to 10% of max measurements
for i in 1:n_out
    measurement_noise[((i - 1) * amount_of_measurements + 1):(i * amount_of_measurements)] .= 0.10 * maximum(abs.(y_obs[i, :]))
end

a0 = zeros(n_out, n_dif)
a0[1,1] = -0.53477; a0[2,1] = -1.54269; a0[1,2] = -1.96109; a0[3,2] = -2.08949; a0[2,3] = 0.170251; a0    #    /= 20

num_kernel_hyperparameters = include_kernel("Quasi_periodic_kernel")    # sets correct num_kernel_hyperparameters
problem_def_528 = build_problem_definition(Quasi_periodic_kernel, num_kernel_hyperparameters, n_dif, n_out, x_obs, y_obs, measurement_noise, a0)
@save "jld2_files/problem_def_528.jld2" problem_def_528 normals
