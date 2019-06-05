# adding in custom functions
include("src/setup.jl")
# include("test/runtests.jl")
include("src/all_functions.jl")

###################################
# Loading data and setting kernel #
###################################

using Flux; using Flux.Tracker: track, @grad, data

kernel_names = ["quasi_periodic_kernel", "se_kernel", "rq_kernel", "matern52_kernel"]

# if called from terminal with an argument, use a full dataset. Otherwise, use a smaller testing set
if length(ARGS)>0
    kernel_name = kernel_names[parse(Int, ARGS[1])]
    println("optimizing the full problem using the $kernel_name")
    @load "jld2_files/res-1000-lambda-3923-6664-1years_1579spots_diffrot_id11_problem_def_full_base.jld2" problem_def_base
    kernel_function, num_kernel_hyperparameters = include_kernel(kernel_name)
    problem_definition = build_problem_definition(kernel_function, num_kernel_hyperparameters, problem_def_base)
    flux_cb_delay = 3600 / 2
    grad_norm_thres = 5e0
    opt = ADAM(0.1)
else
    kernel_name = kernel_names[1]
    @load "jld2_files/res-1000-lambda-3923-6664-1years_1579spots_diffrot_id11_problem_def_sample_base.jld2" problem_def_base
    kernel_function, num_kernel_hyperparameters = include_kernel(kernel_name)
    problem_definition = build_problem_definition(kernel_function, num_kernel_hyperparameters, problem_def_base)
    flux_cb_delay = 3600 / 500
    grad_norm_thres = 5e-1
    opt = ADAM(0.2)
end

# creating path to save figures
mkpath("figs/gp/$kernel_name/training")

# allowing covariance matrix to be calculated in parallel
prep_parallel_covariance(kernel_name)

########################################
# Adding planet and normalizing scores #
########################################

P = (30)u"d"
e = 0.1
M0 = 5.57  # 2 * π * rand()
K = 2  # m/s
ω = 3.87  # 2 * π * rand()
problem_definition.y_obs[:] = add_kepler_to_Jones_problem_definition(
    problem_definition, P, e, M0, K, ω;
    normalization=problem_definition.normals[1])

n_obs = length(problem_definition.x_obs)
for i in 1:problem_definition.n_out
    problem_definition.normals[i] = std(problem_definition.y_obs[1 + (i - 1) * n_obs : i * n_obs])
    problem_definition.y_obs[1 + (i - 1) * n_obs : i * n_obs] /= problem_definition.normals[i]
    problem_definition.noise[1 + (i - 1) * n_obs : i * n_obs] /= problem_definition.normals[i]
end

#####################################
# Initial hyperparameters and plots #
#####################################

# kernel hyper parameters
time_span = maximum(problem_definition.x_obs) - minimum(problem_definition.x_obs)
# kernel_lengths = time_span/3 * ones(problem_definition.n_kern_hyper)
kernel_lengths = maximum([10, time_span/10]) * ones(problem_definition.n_kern_hyper)
total_hyperparameters = append!(collect(Iterators.flatten(problem_definition.a0)), kernel_lengths)

# how finely to sample the domain (for plotting)
amount_of_samp_points = convert(Int64, max(500, round(2 * sqrt(2) * length(problem_definition.x_obs))))

Jones_line_plots(amount_of_samp_points, problem_definition, total_hyperparameters; file="figs/gp/$kernel_name/initial_gp")  # , find_post=false, plot_K=true, plot_K_profile=true)
Jones_line_plots(amount_of_samp_points, problem_definition, total_hyperparameters; file="figs/gp/$kernel_name/post_gp")  # , plot_K=true, plot_K_profile=true)

#################################
# Jones model fitting with FLux #
#################################

# Allowing Flux to use the analytical gradients we have calculated
f_custom(non_zero_hyper) = nlogL_Jones(problem_definition, non_zero_hyper)
f_custom(non_zero_hyper::TrackedArray) = track(f_custom, non_zero_hyper)
g_custom() = ∇nlogL_Jones(problem_definition, data(non_zero_hyper_param))
@grad f_custom(non_zero_hyper) = f_custom(data(non_zero_hyper)), Δ -> tuple(Δ .* g_custom())

# initializing save of fit
current_params = initialize_optimize_Jones_model_jld2!(kernel_name, total_hyperparameters[findall(!iszero, total_hyperparameters)])

# Setting model parameters for Flux
non_zero_hyper_param = param(current_params)
ps = Flux.params(non_zero_hyper_param)

# Final function wrapper for Flux
f_custom() = f_custom(non_zero_hyper_param)

# setting things for Flux to use
flux_data = Iterators.repeated((), 2000)  # use at most 500 iterations

# save plots as we are training every flux_cb_delay seconds
# stop training if our gradient norm gets small enough
flux_cb = function ()
    training_time_str = Dates.format(now(),"yyyy_mm_dd_HH_MM_SS")
    Jones_line_plots(amount_of_samp_points, problem_definition, reconstruct_total_hyperparameters(problem_definition, data(non_zero_hyper_param)); file="figs/gp/$kernel_name/training/trained_" * training_time_str * "_gp", plot_K_profile=true)
    grad_norm = norm(g_custom())
    println("Current time: " * training_time_str * " score: ", data(f_custom()), " with gradient norm ", grad_norm)
    println("Current hyperparameters: " * string(data(non_zero_hyper_param)))
    if grad_norm < grad_norm_thres
        println("Gradient threshold of $grad_norm_thres reached!")
        Flux.stop()
    end
    update_optimize_Jones_model_jld2!(kernel_name, non_zero_hyper_param)
    println()
end

# @profiler
@elapsed Flux.train!(f_custom, ps, flux_data, opt, cb=Flux.throttle(flux_cb, flux_cb_delay))

update_optimize_Jones_model_jld2!(kernel_name, non_zero_hyper_param)

final_total_hyperparameters = reconstruct_total_hyperparameters(problem_definition, data(non_zero_hyper_param))

println("starting hyperparameters")
println(total_hyperparameters)
println(nlogL_Jones(problem_definition, total_hyperparameters), "\n")

println("ending hyperparameters")
println(final_total_hyperparameters)
println(nlogL_Jones(problem_definition, final_total_hyperparameters), "\n")

Jones_line_plots(amount_of_samp_points, problem_definition, final_total_hyperparameters; file="figs/gp/$kernel_name/fit_gp", plot_K=true, plot_K_profile=true)

# coeffs = final_total_hyperparameters[1:end - problem_definition.n_kern_hyper]
# coeff_array = reconstruct_array(coeffs[findall(!iszero, coeffs)], problem_definition.a0)

# ################
# # Corner plots #
# ################
#
# possible_labels = [L"a_{11}" L"a_{21}" L"a_{12}" L"a_{32}" L"a_{23}" L"\lambda_{SE}" L"\lambda_{P}" L"\tau_P";
#     L"a_{11}" L"a_{21}" L"a_{12}" L"a_{32}" L"a_{23}" L"\lambda_{SE}" L" " L" ";
#     L"a_{11}" L"a_{21}" L"a_{12}" L"a_{32}" L"a_{23}" L"\alpha" L"\lambda_{RQ}" L" ";
#     L"a_{11}" L"a_{21}" L"a_{12}" L"a_{32}" L"a_{23}" L"\lambda_{M52}" L" " L" "]
#
# @load "jld2_files/optimize_Jones_model_$kernel_name.jld2" current_params
# actual_labels = possible_labels[1, 1:length(current_params)]
# f_corner(input) = nlogL_Jones(problem_definition, input)
# @elapsed corner_plot(f_corner, current_params, "figs/gp/$kernel_name/corner_$kernel_name.png"; input_labels=actual_labels)

###########################################################################
# Evaluating GP likelihoods after taking out planets at different periods #
###########################################################################

amount_of_periods = 2048

# sample linearly in frequency space so that we get periods from the 1 / uneven Nyquist
# frequency to 4 times the total timespan of the data
freq_grid = linspace(1 / (problem_definition.x_obs[end] - problem_definition.x_obs[1]) / 4, uneven_nyquist_frequency(problem_definition.x_obs), amount_of_periods)
period_grid = 1 ./ reverse(freq_grid)

K_obs = K_observations(problem_definition, final_total_hyperparameters)

# making necessary variables local to all workers
# sendto(workers(), kernel_name=kernel_name)
# @everywhere include_kernel(kernel_name)
sendto(workers(), problem_definition=problem_definition, final_total_hyperparameters=final_total_hyperparameters, K_obs=K_obs)
@everywhere kep_likelihood_distributed(period::Real) = nlogL_Jones(problem_definition, final_total_hyperparameters; K_obs=K_obs, P=period)

# parallelize with DistributedArrays
@everywhere using DistributedArrays
period_grid_dist = distribute(period_grid)
parallel_time = @elapsed likelihoods = collect(map(kep_likelihood_distributed, period_grid_dist))

begin
    ax = init_plot()
    ticklabel_format(style="sci", axis="y", scilimits=(0,0))
    fig = semilogx(period_grid, -likelihoods, color="black")
    xlabel("Periods (days)")
    ylabel("negative GP likelihoods")
    axvline(x=convert_and_strip_units(u"d", P))
    axhline(y=-nlogL_Jones(problem_definition, final_total_hyperparameters), color="k")
    title_string = @sprintf "%.0f day, %.2f m/s" convert_and_strip_units(u"d",P) K
    title(title_string, fontsize=30)
    save_PyPlot_fig("figs/rv/$(kernel_name)_$(amount_of_periods).png")
    PyPlot.close_figs()
end

# three best periods
best_period_grid = period_grid[find_modes(-likelihoods)]

# plot after subtracing best period signal
Jones_line_plots(amount_of_samp_points, problem_definition, final_total_hyperparameters; file="figs/gp/$kernel_name/after", y_obs=remove_kepler(problem_definition.y_obs, problem_definition.x_obs, best_period_grid[1], K_obs))

# #################################################
# # Refitting GP with best fit signals subtracted #
# #################################################
#
# # Allowing Flux to use the analytical gradients we have calculated
# f_custom2(non_zero_hyper) = nlogL_Jones(problem_definition, non_zero_hyper; P=best_period_grid[1])
# f_custom2(non_zero_hyper::TrackedArray) = track(f_custom2, non_zero_hyper)
# g_custom2() = ∇nlogL_Jones(problem_definition, data(non_zero_hyper_param); P=best_period_grid[1])
# @grad f_custom2(non_zero_hyper) = f_custom2(data(non_zero_hyper)), Δ -> tuple(Δ .* g_custom2())
#
# # Setting model parameters for Flux
# non_zero_hyper_param2 = param(copy(current_params))
# ps2 = Flux.params(non_zero_hyper_param2)
#
# # Final function wrapper for Flux
# f_custom2() = f_custom2(non_zero_hyper_param2)
#
# # setting things for Flux to use
# flux_data = Iterators.repeated((), 100)  # use at most 500 iterations
#
# # save plots as we are training every flux_cb_delay seconds
# # stop training if our gradient norm gets small enough
# flux_cb2 = function ()
#     training_time_str = Dates.format(now(),"yyyy_mm_dd_HH_MM_SS")
#     # Jones_line_plots(amount_of_samp_points, problem_definition, reconstruct_total_hyperparameters(problem_definition, data(non_zero_hyper_param)); file="figs/gp/$kernel_name/training/trained_" * training_time_str * "_gp", plot_K_profile=true)
#     grad_norm = norm(g_custom2())
#     println("Current time: " * training_time_str * " score: ", data(f_custom2()), " with gradient norm ", grad_norm)
#     println("Current hyperparameters: " * string(data(non_zero_hyper_param)))
#     if grad_norm < grad_norm_thres
#         println("Gradient threshold of $grad_norm_thres reached!")
#         Flux.stop()
#     end
#     # update_optimize_Jones_model_jld2!(kernel_name, non_zero_hyper_param)
#     println()
# end
#
# # @profiler
# @elapsed Flux.train!(f_custom2, ps, flux_data, opt, cb=Flux.throttle(flux_cb2, flux_cb_delay))
#
# update_optimize_Jones_model_jld2!(kernel_name, non_zero_hyper_param)
#
# final_total_hyperparameters = reconstruct_total_hyperparameters(problem_definition, data(non_zero_hyper_param))
#
# println("starting hyperparameters")
# println(total_hyperparameters)
# println(nlogL_Jones(problem_definition, total_hyperparameters), "\n")
#
# println("ending hyperparameters")
# println(final_total_hyperparameters)
# println(nlogL_Jones(problem_definition, final_total_hyperparameters), "\n")
#
# Jones_line_plots(amount_of_samp_points, problem_definition, final_total_hyperparameters; file="figs/gp/$kernel_name/fit_gp", plot_K=true, plot_K_profile=true)

# ##########################
# # Evidence approximation #
# ##########################
#
# H = ∇∇nlogL_Jones(problem_definition, final_total_hyperparameters)
# log_laplace_approximation(H, nlogL_Jones(problem_definition, current_params), 0)
