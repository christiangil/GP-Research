#adding in custom functions
include("../general_functions.jl")
include("../GP_functions.jl")
include("../diagnostic_functions.jl")

# loading in data
using JLD2, FileIO
@load "sunspot_data.jld2" lambda phases quiet
@load "rv_data.jld2" doppler_comp genpca_out rvs_out
mu, M, scores = genpca_out
scores[:, 1] = rvs_out
scores = scores'

# how many components you will use
n_out = 3
# how many differentiated versions of the original GP you will use
n_dif = 3

# Setting up all of the data things
# how much of the data you want to use (on time domain)
start_ind = 100
end_ind = 120  # 1070
amount_of_measurements = end_ind - start_ind + 1
total_amount_of_measurements = amount_of_measurements * n_out

# getting proper slice of data
x_obs = phases[start_ind:end_ind]
y_obs_hold = scores[1:n_out, start_ind:end_ind]

# rearranging the data into one column (not sure reshape() does what I want)
y_obs = zeros(total_amount_of_measurements)
for i in 1:n_out
    y_obs[((i - 1) * amount_of_measurements + 1):(i * amount_of_measurements)] = y_obs_hold[i, :]
end

measurement_noise = ones(total_amount_of_measurements)
# currently set to 5 percent of total amplitude at every point. should be done with bootstrapping
for i in 1:n_out
    measurement_noise[((i - 1) * amount_of_measurements + 1):(i * amount_of_measurements)] *= 0.05 * maximum(abs.(y_obs_hold[i, :]))
end

# a0 = ones(n_out, n_dif) / 20
a0 = zeros(n_out, n_dif)
a0[1,1] = 1; a0[1,2] = 1; a0[2,1] = 1; a0[2,3] = 1; a0[3,2] = 1; a0 /= 20

include("../kernels/Quasi_periodic_kernel.jl")  # sets correct num_kernel_hyperparameters
build_problem_definition(Quasi_periodic_kernel, num_kernel_hyperparameters, n_dif, n_out, x_obs, y_obs, measurement_noise, a0)

things = append!(collect(Iterators.flatten(a0)), [1, 1, 1] / 1.5)
# initializing Cholesky factorization storage
chol_storage = chol_struct(things, ridge_chol(K_observations(problem_definition, copy(things))))

##########################################

@testset "taking gradients" begin
   @test test_grad(rand(3); print_stuff=false)  # "Gradient of nLogL is incorrect"
   @test est_dKdθ(problem_definition, rand(3); return_bool=true, print_stuff=false)  # "partial derivatives of covariance function are incorrect"
end
