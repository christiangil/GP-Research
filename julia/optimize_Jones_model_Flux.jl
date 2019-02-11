#adding in custom functions
include("src/all_functions.jl")

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
# end_ind = 170    # 1070
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
#         y_obs[((i - 1) * amount_of_measurements + 1):(i * amount_of_measurements)] = y_obs_hold[i, :] / normals[i]
#         measurement_noise[((i - 1) * amount_of_measurements + 1):(i * amount_of_measurements)] = measurement_noise_hold[i, :] / normals[i]
# end
#
# # # setting noise to 10% of max measurements
# for i in 1:n_out
#         measurement_noise[((i - 1) * amount_of_measurements + 1):(i * amount_of_measurements)] .= 0.10 * maximum(abs.(y_obs[i, :]))
# end
#
# # normals
# # y_obs
# # measurement_noise
#
# # a0 = ones(n_out, n_dif) / 20
# a0 = zeros(n_out, n_dif)
# a0[1,1] = 0.03; a0[2,1] = 0.3; a0[1,2] = 0.3; a0[3,2] = 0.3; a0[2,3] = 0.075; a0    #    /= 20
#
# num_kernel_hyperparameters = include_kernel("Quasi_periodic_kernel")    # sets correct num_kernel_hyperparameters
# sample_problem_def = build_problem_definition(Quasi_periodic_kernel, num_kernel_hyperparameters, n_dif, n_out, x_obs, y_obs, measurement_noise, a0)
# @save "jld2_files/sample_problem_def.jld2" sample_problem_def
# # const problem_definition = build_problem_definition(Quasi_periodic_kernel, num_kernel_hyperparameters, n_dif, n_out, x_obs, y_obs, measurement_noise, a0)

include_kernel("Quasi_periodic_kernel")
@load "jld2_files/sample_problem_def.jld2" sample_problem_def
problem_definition = sample_problem_def

##############################################################################

# kernel hyper parameters
kernel_lengths = 2 * ones(problem_definition.n_kern_hyper)
total_hyperparameters = append!(collect(Iterators.flatten(problem_definition.a0)), kernel_lengths)

# how finely to sample the domain (for plotting)
amount_of_samp_points = 500

# creating many inputs to sample the eventual gaussian process on (for plotting)
x_samp = linspace(minimum(problem_definition.x_obs), maximum(problem_definition.x_obs), amount_of_samp_points[1])

# total amount of output points
amount_of_total_samp_points = amount_of_samp_points * problem_definition.n_out

# Finding how correlated the sampled inputs are to each other
# (aka getting the covariance matrix by evaluating the kernel function at all
# pairs of points)
K_samp = covariance(problem_definition, x_samp, x_samp, total_hyperparameters)
plot_im(K_samp, file="test.pdf")

# getting the Cholesky factorization of the covariance matrix (for drawing GPs)
L_samp = ridge_chol(K_samp).L

# showing a heatmap of the covariance matrix
# plot_im(K_samp; file="figs/gp/initial_covariance.pdf")
# plot_im(L_samp)

# custom_line_plot(x_samp, L_samp, problem_definition.x_obs, problem_definition.y_obs, output=1, file="figs/gp/initial_gp_0.pdf")
# custom_line_plot(x_samp, L_samp, problem_definition.x_obs, problem_definition.y_obs, output=2, file="figs/gp/initial_gp_1.pdf")
# custom_line_plot(x_samp, L_samp, problem_definition.x_obs, problem_definition.y_obs, output=3, file="figs/gp/initial_gp_2.pdf")

# calculate posterior quantities
mean_post, return_vec = GP_posteriors(problem_definition, x_samp, total_hyperparameters, return_K=true, return_L=true, return_σ=true)
σ, K_post, L_post = return_vec

# plot the posterior covariance matrix (colors show how correlated points are to each other)
# plot_im(K_post)
# plot_im(L_post)

# plotting randomly drawn Gaussian processes with the mean function and covariance of the posterior
# much closer to the data, no?

# for 1D GPs
# custom_line_plot(x_samp, L_post, problem_definition; σ=σ, mean=mean_post, output=1, file="figs/gp/cond_initial_gp_1.pdf")
# custom_line_plot(x_samp, L_post, problem_definition; σ=σ, mean=mean_post, output=2, file="figs/gp/cond_initial_gp_2.pdf")
# custom_line_plot(x_samp, L_post, problem_definition; σ=σ, mean=mean_post, output=3, file="figs/gp/cond_initial_gp_3.pdf")


using Flux; using Flux.Tracker: track, @grad, data

# Allowing Flux to use the analytical gradients we have calculated
nLogL_custom(non_zero_hyper) = nlogL_Jones(problem_definition, non_zero_hyper)
nLogL_custom(non_zero_hyper::TrackedArray) = track(nLogL_custom, non_zero_hyper)
@grad nLogL_custom(non_zero_hyper) = nLogL_custom(data(non_zero_hyper)), Δ -> tuple(Δ .* ∇nlogL_Jones(problem_definition, data(non_zero_hyper)))

# Setting model parameters for Flux
non_zero_hyper_param = param(total_hyperparameters[findall(!iszero, total_hyperparameters)])
ps = Flux.params(non_zero_hyper_param)

# Final function wrapper for Flux
nLogL_custom() = nLogL_custom(non_zero_hyper_param)

# Initializing other training things
iteration_amount = 100
flux_data = Iterators.repeated((), iteration_amount)    # the function is called $iteration_amount times with no arguments
opt = ADAM(0.1)

# global current_iteration = 0
# function update_iteration!()
#     global current_iteration += 1
# end
#
# #callback function to observe training
# callback_func_expens = function ()
#     LogL = data(nLogL_custom())
#     total_hyperparameters = reconstruct_total_hyperparameters(problem_definition, data(non_zero_hyper_param))
#     mean_post, return_vec = GP_posteriors(problem_definition, x_samp, total_hyperparameters, return_K=true, return_L=true, return_σ=true)
#     σ, K_post, L_post = return_vec
#     update_iteration!()
#     custom_line_plot(x_samp, L_post, problem_definition; σ=σ, mean=mean_post, output=1, file="figs/gp/training/training_gp1_$current_iteration.png", LogL=LogL)
#     custom_line_plot(x_samp, L_post, problem_definition; σ=σ, mean=mean_post, output=2, file="figs/gp/training/training_gp2_$current_iteration.png", LogL=LogL)
#     custom_line_plot(x_samp, L_post, problem_definition; σ=σ, mean=mean_post, output=3, file="figs/gp/training/training_gp3_$current_iteration.png", LogL=LogL)
# end

callback_func_simp = function ()
    println("Current nLogL: ", data(nLogL_custom()))
    # println(data(non_zero_hyper_param))
    println("Current gradient norm: ", norm(∇nlogL_Jones(problem_definition, data(non_zero_hyper_param))))
end


Flux.train!(nLogL_custom, ps, flux_data, opt, cb=Flux.throttle(callback_func_simp, 10))
# Flux.train!(nLogL_custom, ps, flux_data, opt)

final_total_hyperparameters = reconstruct_total_hyperparameters(problem_definition, data(non_zero_hyper_param))

println("old hyperparameters")
println(total_hyperparameters)
println(nlogL_Jones(problem_definition, total_hyperparameters))

println("new hyperparameters")
println(final_total_hyperparameters)
println(nlogL_Jones(problem_definition, final_total_hyperparameters))

# K_post = covariance(problem_definition, x_samp, x_samp, final_total_hyperparameters)
# plot_im(K_post, file="test.pdf")

# reruning analysis of posterior with the "most likley" kernel amplitude and lengthscale
# recalculating posterior covariance and mean function
mean_post, return_vec = GP_posteriors(problem_definition, x_samp, final_total_hyperparameters, return_K=true, return_L=true, return_σ=true)
σ, K_post, L_post = return_vec

# plot the posterior covariance of the "most likely" posterior matrix
# (colors show how correlated points are to each other)
# plot_im(K_post)

# for 1D GPs
custom_line_plot(x_samp, L_post, problem_definition; σ=σ, mean=mean_post, output=1, file="figs/gp/fit_gp_1.pdf")
custom_line_plot(x_samp, L_post, problem_definition; σ=σ, mean=mean_post, output=2, file="figs/gp/fit_gp_2.pdf")
custom_line_plot(x_samp, L_post, problem_definition; σ=σ, mean=mean_post, output=3, file="figs/gp/fit_gp_3.pdf")
