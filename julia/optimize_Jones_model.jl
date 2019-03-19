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
time_span = maximum(problem_definition.x_obs) - minimum(problem_definition.x_obs)
kernel_lengths = time_span/10 * ones(problem_definition.n_kern_hyper)
total_hyperparameters = append!(collect(Iterators.flatten(problem_definition.a0)), kernel_lengths)

# how finely to sample the domain (for plotting)
amount_of_samp_points = 500

# total amount of output points
amount_of_total_samp_points = amount_of_samp_points * problem_definition.n_out

Jones_line_plots(amount_of_samp_points, problem_definition, total_hyperparameters; file="figs/gp/initial_gp", find_post=false, plot_K=true)
Jones_line_plots(amount_of_samp_points, problem_definition, total_hyperparameters; file="figs/gp/post_gp", plot_K=true)

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

# final functions for observing training
custom_g() = ∇nlogL_Jones(problem_definition, data(non_zero_hyper_param))
training_plots() = Jones_line_plots(amount_of_samp_points, problem_definition, reconstruct_total_hyperparameters(problem_definition, data(non_zero_hyper_param)); file="figs/gp/training/iteration_$(iter_num)_gp")

flux_train_to_target!(nLogL_custom, custom_g, ps; outer_cb=training_plots)
final_total_hyperparameters = reconstruct_total_hyperparameters(problem_definition, data(non_zero_hyper_param))

println("starting hyperparameters")
println(total_hyperparameters)
println(nlogL_Jones(problem_definition, total_hyperparameters), "\n")

println("ending hyperparameters")
println(final_total_hyperparameters)
println(nlogL_Jones(problem_definition, final_total_hyperparameters), "\n")

Jones_line_plots(amount_of_samp_points, problem_definition, final_total_hyperparameters; file="figs/gp/fit_gp", plot_K=true)
