#adding in custom functions
include("src/all_functions.jl")
# run_tests()

# can use this if you want to replicate results
# srand(1234)

# loading in data
using JLD2, FileIO

@load "jld2_files/problem_def_base_full.jld2" problem_def_base_full normals

# kernel_names = ["quasi_periodic_kernel", "periodic_kernel", "rbf_kernel", "exponential_kernel", "exp_periodic_kernel", "matern32_kernel", "matern52_kernel", "rq_kernel"]
kernel_names = ["quasi_periodic_kernel", "rbf_kernel", "rq_kernel"]
if length(ARGS)>0
    kernel_name = kernel_names[parse(Int, ARGS[1])]
else
    kernel_name = kernel_names[3]
end

mkpath("figs/gp/$kernel_name/training")

kernel_function, num_kernel_hyperparameters = include_kernel(kernel_name)
problem_definition = build_problem_definition(kernel_function, num_kernel_hyperparameters, problem_def_base)

##############################################################################
# kernel hyper parameters
time_span = maximum(problem_definition.x_obs) - minimum(problem_definition.x_obs)
kernel_lengths = time_span/10 * ones(problem_definition.n_kern_hyper)
total_hyperparameters = append!(collect(Iterators.flatten(problem_definition.a0)), kernel_lengths)

# how finely to sample the domain (for plotting)
amount_of_samp_points = 500

# total amount of output points
amount_of_total_samp_points = amount_of_samp_points * problem_definition.n_out

Jones_line_plots(amount_of_samp_points, problem_definition, total_hyperparameters; file="figs/gp/$kernel_name/initial_gp", find_post=false, plot_K=true)
Jones_line_plots(amount_of_samp_points, problem_definition, total_hyperparameters; file="figs/gp/$kernel_name/post_gp", plot_K=true)

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
training_plots() = Jones_line_plots(amount_of_samp_points, problem_definition, reconstruct_total_hyperparameters(problem_definition, data(non_zero_hyper_param)); file="figs/gp/$kernel_name/training/iteration_$(iter_num)_gp")

flux_train_to_target!(nLogL_custom, custom_g, ps; outer_cb=training_plots)
# flux_train_to_target!(nLogL_custom, custom_g, ps)
final_total_hyperparameters = reconstruct_total_hyperparameters(problem_definition, data(non_zero_hyper_param))

println("starting hyperparameters")
println(total_hyperparameters)
println(nlogL_Jones(problem_definition, total_hyperparameters), "\n")

println("ending hyperparameters")
println(final_total_hyperparameters)
println(nlogL_Jones(problem_definition, final_total_hyperparameters), "\n")

Jones_line_plots(amount_of_samp_points, problem_definition, final_total_hyperparameters; file="figs/gp/$kernel_name/fit_gp", plot_K=true)
