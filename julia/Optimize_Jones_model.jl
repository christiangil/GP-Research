#adding in custom functions
include("all_functions.jl")

# can use this if you want to replicate results
# srand(1234)

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
end_ind = 140  # 1070
amount_of_measurements = end_ind - start_ind + 1
total_amount_of_measurements = amount_of_measurements * n_out

# getting proper slice of data
x_obs = phases[start_ind:end_ind]
y_obs_hold = scores[1:n_out, start_ind:end_ind]

# normalizing the data (for numerical purposes)
# normals = maximum(abs.(y_obs_hold), dims=2)'[:]
# for i in 1:n_out
#     y_obs_hold[i, :] /= normals[i]
# end

# rearranging the data into one column (not sure reshape() does what I want)
y_obs = zeros(total_amount_of_measurements)
for i in 1:n_out
    y_obs[((i - 1) * amount_of_measurements + 1):(i * amount_of_measurements)] = y_obs_hold[i, :]
end

# Uncertainty in the data (AFFECTS SPREAD OF DATA AND HOW TIGHTLY THE GP's WILL
# TRY TO HUG THE DATA) aka how much noise is added to measurements and
# measurement covariance function
# a vector of length = total_amount_of_measurements
measurement_noise = ones(total_amount_of_measurements)
# currently set to 20 percent of total amplitude at every point. should be done with bootstrapping
for i in 1:n_out
    measurement_noise[((i - 1) * amount_of_measurements + 1):(i * amount_of_measurements)] *= 0.05 * maximum(abs.(y_obs_hold[i, :]))
end

a0 = ones(n_out, n_dif) / 20
# a0 = zeros(n_out, n_dif)
# a0[1,1] = 1; a0[2,1] = 1; a0[1,2] = 1000; a0[3,2] = 1000; a0[2,3] = 100; a0 /= 20

include("kernels/Quasi_periodic_kernel.jl")  # sets correct num_kernel_hyperparameters
build_problem_definition(Quasi_periodic_kernel, num_kernel_hyperparameters, n_dif, n_out, x_obs, y_obs, measurement_noise, a0)

# kernel hyper parameters
kernel_lengths = [1, 1, 1] / 1.5
total_hyperparameters = append!(collect(Iterators.flatten(a0)), kernel_lengths)

# # initializing Cholesky factorization storage
# chol_storage = chol_struct(copy(total_hyperparameters), ridge_chol(K_observations(problem_definition, copy(total_hyperparameters))))

##############################################################################

# how finely to sample the domain (for plotting)
amount_of_samp_points = 500

# creating many inputs to sample the eventual gaussian process on (for plotting)
x_samp = linspace(minimum(x_obs),maximum(x_obs), amount_of_samp_points[1])

# total amount of output points
amount_of_total_samp_points = amount_of_samp_points * n_out

# Finding how correlated the sampled inputs are to each other
# (aka getting the covariance matrix by evaluating the kernel function at all
# pairs of points)
K_samp = covariance(problem_definition, x_samp, x_samp, total_hyperparameters)
plot_im(K_samp, file="test.pdf")

# getting the Cholesky factorization of the covariance matrix (for drawing GPs)
L_samp = ridge_chol(K_samp).L  # usually has to add a ridge

# showing a heatmap of the covariance matrix
# plot_im(K_samp; file="figs/gp/initial_covariance.pdf")
# plot_im(L_samp)

custom_line_plot(x_samp, L_samp, x_obs, y_obs, output=1, file="figs/gp/initial_gp_0.pdf")
custom_line_plot(x_samp, L_samp, x_obs, y_obs, output=2, file="figs/gp/initial_gp_1.pdf")
custom_line_plot(x_samp, L_samp, x_obs, y_obs, output=3, file="figs/gp/initial_gp_2.pdf")

# calculate posterior quantities
mean_post, return_vec = GP_posteriors(problem_definition, x_samp, total_hyperparameters, return_K=true, return_L=true, return_σ=true)
σ, K_post, L_post = return_vec

# plot the posterior covariance matrix (colors show how correlated points are to each other)
# plot_im(K_post)
# plot_im(L_post)

# plotting randomly drawn Gaussian processes with the mean function and covariance of the posterior
# much closer to the data, no?

# for 1D GPs
custom_line_plot(x_samp, L_post, x_obs, y_obs, σ=σ, mean=mean_post, output=1, file="figs/gp/cond_initial_gp_1.pdf")
custom_line_plot(x_samp, L_post, x_obs, y_obs, σ=σ, mean=mean_post, output=2, file="figs/gp/cond_initial_gp_2.pdf")
custom_line_plot(x_samp, L_post, x_obs, y_obs, σ=σ, mean=mean_post, output=3, file="figs/gp/cond_initial_gp_3.pdf")

# numerically maximize the likelihood to find the best hyperparameters
using Optim

# storing initial hyperparameters
initial_x = total_hyperparameters[findall(!iszero, total_hyperparameters)]

# # initializing Cholesky factorization storage
# chol_storage = chol_struct(initial_x, ridge_chol(K_observations(problem_definition, initial_x)))



# @elapsed result = optimize(nlogL_Jones, ∇nlogL_Jones, initial_x, LBFGS(), Optim.Options(show_trace=true))
# @elapsed result = optimize(nlogL_Jones, ∇nlogL_Jones, initial_x, LBFGS())
@elapsed result = optimize(Optim.only_fg!(nlogL_Jones_fg!), initial_x, LBFGS())

final_total_hyperparameters = reconstruct_total_hyperparameters(problem_definition, result.minimizer)

println("old hyperparameters")
println(total_hyperparameters)
# println(nlogL_Jones(initial_x))

println("new hyperparameters")
println(final_total_hyperparameters)
println(result.minimum)
# new hyperparameters
# no penalty
# [-0.38062, -7785.22, 0.0, -0.289081, 0.0, -31.1518, 0.0, 7.02291, 0.0, -72.7, -0.0870051, 0.14647]
# 323.6028338641281
# penalty
# [0.547235, 0.53716, 0.488929, 50.577, 0.488929, 50.4002, 0.488929, 5.51546, 0.488929, 0.520252, 0.286278, 1.07373]
# 5311.9462030734985

# reruning analysis of posterior with the "most likley" kernel amplitude and lengthscale
# recalculating posterior covariance and mean function
mean_post, return_vec = GP_posteriors(problem_definition, x_samp, final_total_hyperparameters, return_K=true, return_L=true, return_σ=true)
σ, K_post, L_post = return_vec

# plot the posterior covariance of the "most likely" posterior matrix
# (colors show how correlated points are to each other)
# plot_im(K_post)

# for 1D GPs
custom_line_plot(x_samp, L_post, x_obs, y_obs, σ=σ, mean=mean_post, output=1, file="figs/gp/fit_gp_1.pdf")
custom_line_plot(x_samp, L_post, x_obs, y_obs, σ=σ, mean=mean_post, output=2, file="figs/gp/fit_gp_2.pdf")
custom_line_plot(x_samp, L_post, x_obs, y_obs, σ=σ, mean=mean_post, output=3, file="figs/gp/fit_gp_3.pdf")
