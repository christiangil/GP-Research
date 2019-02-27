#adding in custom functions
include("src/all_functions.jl")

# can use this if you want to replicate results
# srand(1234)

# loading in data
using JLD2, FileIO

include_kernel("Quasi_periodic_kernel")
@load "jld2_files/sample_problem_def.jld2" sample_problem_def normals
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
iteration_amount = 200
flux_data = Iterators.repeated((), iteration_amount)    # the function is called $iteration_amount times with no arguments
opt = ADAM(0.1)

global current_iteration = 0
function update_iteration!()
    global current_iteration += 1
end

#callback function to observe training
callback_func_expens = function ()
    LogL = data(nLogL_custom())
    total_hyperparameters = reconstruct_total_hyperparameters(problem_definition, data(non_zero_hyper_param))
    mean_post, return_vec = GP_posteriors(problem_definition, x_samp, total_hyperparameters, return_K=true, return_L=true, return_σ=true)
    σ, K_post, L_post = return_vec
    update_iteration!()
    custom_line_plot(x_samp, L_post, problem_definition; σ=σ, mean=mean_post, output=1, file="figs/gp/training/training_gp1_$current_iteration.png", LogL=LogL)
    custom_line_plot(x_samp, L_post, problem_definition; σ=σ, mean=mean_post, output=2, file="figs/gp/training/training_gp2_$current_iteration.png", LogL=LogL)
    custom_line_plot(x_samp, L_post, problem_definition; σ=σ, mean=mean_post, output=3, file="figs/gp/training/training_gp3_$current_iteration.png", LogL=LogL)
end

callback_func_simp = function ()
    println("Current nLogL: ", data(nLogL_custom()))
    # println(data(non_zero_hyper_param))
    println("Current gradient norm: ", norm(∇nlogL_Jones(problem_definition, data(non_zero_hyper_param))))
end


Flux.train!(nLogL_custom, ps, flux_data, opt, cb=Flux.throttle(callback_func_simp, 5))
# Flux.train!(nLogL_custom, ps, flux_data, opt)

final_total_hyperparameters = reconstruct_total_hyperparameters(problem_definition, data(non_zero_hyper_param))

println("old hyperparameters")
println(total_hyperparameters)
println(nlogL_Jones(problem_definition, total_hyperparameters))

println("new hyperparameters")
println(final_total_hyperparameters)
LogL = nlogL_Jones(problem_definition, final_total_hyperparameters)
println(LogL)

# reruning analysis of posterior with the "most likley" kernel amplitude and lengthscale
# recalculating posterior covariance and mean function
mean_post, return_vec = GP_posteriors(problem_definition, x_samp, final_total_hyperparameters, return_K=true, return_L=true, return_σ=true)
σ, K_post, L_post = return_vec

waves = final_total_hyperparameters[end-problem_definition.n_kern_hyper+1:end]
# for 1D GPs
custom_line_plot(x_samp, L_post, problem_definition; σ=σ, mean=mean_post, output=1, file="figs/gp/fit_gp_1.png", LogL=LogL, waves=waves)
custom_line_plot(x_samp, L_post, problem_definition; σ=σ, mean=mean_post, output=2, file="figs/gp/fit_gp_2.png", LogL=LogL, waves=waves)
custom_line_plot(x_samp, L_post, problem_definition; σ=σ, mean=mean_post, output=3, file="figs/gp/fit_gp_3.png", LogL=LogL, waves=waves)
