#adding in custom functions
include("src/setup.jl")
# include("test/runtests.jl")
include("src/all_functions.jl")


# can use this if you want to replicate results
# srand(1234)

# loading in data
using JLD2, FileIO
using Flux; using Flux.Tracker: track, @grad, data

kernel_names = ["quasi_periodic_kernel", "se_kernel", "rq_kernel", "matern52_kernel"]

# if called from terminal with an argument, use a full dataset. Otherwise, use a smaller testing set
if length(ARGS)>0
    kernel_name = kernel_names[parse(Int, ARGS[1])]
    println("optimizing the full problem using the $kernel_name")
    @load "jld2_files/problem_def_full_base.jld2" problem_def_base normals
    kernel_function, num_kernel_hyperparameters = include_kernel(kernel_name)
    problem_definition = build_problem_definition(kernel_function, num_kernel_hyperparameters, problem_def_base)
    flux_cb_delay = 3600 / 2
    grad_norm_thres = 1e1
    opt = ADAM(0.1)
else
    kernel_name = kernel_names[4]
    @load "jld2_files/problem_def_sample_base.jld2" problem_def_base normals
    kernel_function, num_kernel_hyperparameters = include_kernel(kernel_name)
    problem_definition = build_problem_definition(kernel_function, num_kernel_hyperparameters, problem_def_base)
    flux_cb_delay = 3600 / 500
    grad_norm_thres = 1e0
    opt = ADAM(0.2)
end

mkpath("figs/gp/$kernel_name/training")

##############################################################################
# kernel hyper parameters
time_span = maximum(problem_definition.x_obs) - minimum(problem_definition.x_obs)
# kernel_lengths = time_span/3 * ones(problem_definition.n_kern_hyper)
kernel_lengths = maximum([10, time_span/10]) * ones(problem_definition.n_kern_hyper)
total_hyperparameters = append!(collect(Iterators.flatten(problem_definition.a0)), kernel_lengths)

# how finely to sample the domain (for plotting)
amount_of_samp_points = convert(Int, max(500, round(2 * sqrt(2) * length(problem_definition.x_obs))))

# total amount of output points
amount_of_total_samp_points = amount_of_samp_points * problem_definition.n_out

Jones_line_plots(amount_of_samp_points, problem_definition, total_hyperparameters; file="figs/gp/$kernel_name/initial_gp", find_post=false, plot_K=true, plot_K_profile=true)
Jones_line_plots(amount_of_samp_points, problem_definition, total_hyperparameters; file="figs/gp/$kernel_name/post_gp", plot_K=true, plot_K_profile=true)

# Allowing Flux to use the analytical gradients we have calculated
f_custom(non_zero_hyper) = nlogL_Jones(problem_definition, non_zero_hyper)
f_custom(non_zero_hyper::TrackedArray) = track(f_custom, non_zero_hyper)
g_custom() = ∇nlogL_Jones(problem_definition, data(non_zero_hyper_param))
@grad f_custom(non_zero_hyper) = f_custom(data(non_zero_hyper)), Δ -> tuple(Δ .* g_custom())

# Setting model parameters for Flux
non_zero_hyper_param = param(total_hyperparameters[findall(!iszero, total_hyperparameters)])
ps = Flux.params(non_zero_hyper_param)

# Final function wrapper for Flux
f_custom() = f_custom(non_zero_hyper_param)

# setting things for Flux to use
flux_data = Iterators.repeated((), 500)  # use at most 500 iterations

# save plots as we are training every flux_cb_delay seconds
# stop training if our gradient norm gets small enough
@warn "global training_time variable created/reassigned"; global training_time = 0
flux_cb = function ()
    training_time_str = @sprintf "%.3fh" training_time/3600
    Jones_line_plots(amount_of_samp_points, problem_definition, reconstruct_total_hyperparameters(problem_definition, data(non_zero_hyper_param)); file="figs/gp/$kernel_name/training/trained_" * training_time_str * "_gp", plot_K_profile=true)
    grad_norm = norm(g_custom())
    println("Training time: " * training_time_str * " score: ", data(f_custom()), " with gradient norm ", grad_norm)
    println("Current hyperparameters: " * string(data(non_zero_hyper_param)))
    global training_time += flux_cb_delay
    if grad_norm < grad_norm_thres
        println("Gradient threshold of $grad_norm_thres reached!")
        Flux.stop()
    end
    println()
end

@profiler Flux.train!(f_custom, ps, flux_data, opt, cb=Flux.throttle(flux_cb, flux_cb_delay))

# flux_train_to_target!(nLogL_custom, custom_g, ps)
final_total_hyperparameters = reconstruct_total_hyperparameters(problem_definition, data(non_zero_hyper_param))

println("starting hyperparameters")
println(total_hyperparameters)
println(nlogL_Jones(problem_definition, total_hyperparameters), "\n")

println("ending hyperparameters")
println(final_total_hyperparameters)
println(nlogL_Jones(problem_definition, final_total_hyperparameters), "\n")

Jones_line_plots(amount_of_samp_points, problem_definition, final_total_hyperparameters; file="figs/gp/$kernel_name/fit_gp", plot_K=true, plot_K_profile=true)

coeffs = final_total_hyperparameters[1:end - problem_definition.n_kern_hyper]
coeff_array = reconstruct_array(coeffs[findall(!iszero, coeffs)], problem_definition.a0)

f_corner(input) = nlogL_Jones(problem_definition, input)
input_labels = [L"a_{11}", L"a_{21}", L"a_{12}", L"a_{32}", L"a_{23}", L"\lambda_{M52}"]
@elapsed @profiler corner_plot(f_corner, data(non_zero_hyper_param), "figs/gp/$kernel_name/corner_$kernel_name.png"; input_labels=input_labels)
