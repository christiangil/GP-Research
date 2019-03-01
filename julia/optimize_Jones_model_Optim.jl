#adding in custom functions
include("src/all_functions.jl")

# can use this if you want to replicate results
# srand(1234)

# loading in data
using JLD2, FileIO

include_kernel("quasi_periodic_kernel")
@load "jld2_files/sample_problem_def.jld2" sample_problem_def normals
problem_definition = sample_problem_def
##############################################################################

# kernel hyper parameters
kernel_lengths = 2 * ones(num_kernel_hyperparameters)
total_hyperparameters = append!(collect(Iterators.flatten(a0)), kernel_lengths)

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
L_samp = ridge_chol(K_samp).L

# showing a heatmap of the covariance matrix
# plot_im(K_samp; file="figs/gp/initial_covariance.pdf")
# plot_im(L_samp)

custom_line_plot(x_samp, L_samp, problem_definition, output=1, file="figs/gp/initial_gp_0.pdf")
custom_line_plot(x_samp, L_samp, problem_definition, output=2, file="figs/gp/initial_gp_1.pdf")
custom_line_plot(x_samp, L_samp, problem_definition, output=3, file="figs/gp/initial_gp_2.pdf")

# calculate posterior quantities
mean_post, return_vec = GP_posteriors(problem_definition, x_samp, total_hyperparameters, return_K=true, return_L=true, return_σ=true)
σ, K_post, L_post = return_vec

# plot the posterior covariance matrix (colors show how correlated points are to each other)
# plot_im(K_post)
# plot_im(L_post)

# plotting randomly drawn Gaussian processes with the mean function and covariance of the posterior
# much closer to the data, no?

# for 1D GPs
custom_line_plot(x_samp, L_post, problem_definition, σ=σ, mean=mean_post, output=1, file="figs/gp/cond_initial_gp_1.pdf")
custom_line_plot(x_samp, L_post, problem_definition, σ=σ, mean=mean_post, output=2, file="figs/gp/cond_initial_gp_2.pdf")
custom_line_plot(x_samp, L_post, problem_definition, σ=σ, mean=mean_post, output=3, file="figs/gp/cond_initial_gp_3.pdf")


# numerically maximize the likelihood to find the best hyperparameters
using Optim

# storing initial hyperparameters
initial_x = total_hyperparameters[findall(!iszero, total_hyperparameters)]

custom_fg!(F, G, non_zero_hyperparameters::Array{Float64,1}) = nlogL_Jones_fg!(problem_definition, F, G, non_zero_hyperparameters)
lower = [-1000., -1000, -1000, -1000, -1000, 0, 0, 0]
upper = [1000., 1000, 1000, 1000, 1000, 50, 50, 50]
@elapsed result = optimize(Optim.only_fg!(custom_fg!), lower, upper, initial_x, Fminbox(LBFGS()))
# @elapsed result2 = optimize(Optim.only_fg!(custom_fg!), initial_x, LBFGS())

final_total_hyperparameters = reconstruct_total_hyperparameters(problem_definition, result.minimizer)

println("old hyperparameters")
println(total_hyperparameters)
println(nlogL_Jones(problem_definition, total_hyperparameters))

println("new hyperparameters")
println(final_total_hyperparameters)
# println(result.minimum)

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
custom_line_plot(x_samp, L_post, problem_definition, σ=σ, mean=mean_post, output=1, file="figs/gp/fit_gp_1.pdf")
custom_line_plot(x_samp, L_post, problem_definition, σ=σ, mean=mean_post, output=2, file="figs/gp/fit_gp_2.pdf")
custom_line_plot(x_samp, L_post, problem_definition, σ=σ, mean=mean_post, output=3, file="figs/gp/fit_gp_3.pdf")
