# adding in custom functions
# include("src/setup.jl")
include("test/runtests.jl")
include("src/all_functions.jl")

###################################
# Loading data and setting kernel #
###################################

using Flux; using Flux.Tracker: track, @grad, data

# kernel_names = ["quasi_periodic_kernel", "se_kernel", "rq_kernel", "matern52_kernel"]
kernel_names = ["quasi_periodic_kernel", "se_kernel", "matern52_kernel"]

# if called from terminal with an argument, use a full dataset. Otherwise, use a smaller testing set
if length(ARGS)>0
    kernel_name = kernel_names[parse(Int, ARGS[1])]
    println("optimizing the full problem using the $kernel_name")
    @load "jld2_files/res-1000-lambda-3923-6664-1years_1579spots_diffrot_id11_problem_def_full_base.jld2" problem_def_base
    kernel_function, num_kernel_hyperparameters = include_kernel(kernel_name)
    problem_definition = init_problem_definition(kernel_function, num_kernel_hyperparameters, problem_def_base)
    flux_cb_delay = 3600 / 2
    grad_norm_thres = 5e0
    opt = ADAM(0.1)
else
    kernel_name = kernel_names[2]
    @load "jld2_files/res-1000-lambda-3923-6664-1years_1579spots_diffrot_id11_problem_def_sample_base.jld2" problem_def_base
    # @load "jld2_files/res-1000-lambda-3923-6664-1years_1579spots_diffrot_id11_problem_def_full_base.jld2" problem_def_base
    kernel_function, num_kernel_hyperparameters = include_kernel(kernel_name)
    problem_definition = init_problem_definition(kernel_function, num_kernel_hyperparameters, problem_def_base)
    flux_cb_delay = 3600 / 120
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

P = (20 + 3 * randn())u"d"
e = rand() / 5
M0 = 2 * π * rand()
length(ARGS) > 1 ? K = parse(Int, ARGS[2]) / 10 : K = 0.5  # m/s
ω = 2 * π * rand()
γ = 0
problem_definition.y_obs[:] = add_kepler_to_Jones_problem_definition(
    problem_definition, P, e, M0, K, ω; γ = γ,
    normalization=problem_definition.normals[1])

normalize_problem_definition!(problem_definition)

# mean(problem_definition.noise[1:70])
# mean(problem_definition.noise[71:140])
# mean(problem_definition.noise[141:210])

#####################################
# Initial hyperparameters and plots #
#####################################

# kernel hyper parameters
time_span = maximum(problem_definition.x_obs) - minimum(problem_definition.x_obs)
# kernel_lengths = time_span/3 * ones(problem_definition.n_kern_hyper)
kernel_lengths = maximum([10, time_span/10]) .* ones(problem_definition.n_kern_hyper)
total_hyperparameters = append!(collect(Iterators.flatten(problem_definition.a0)), kernel_lengths)

# how finely to sample the domain (for plotting)
amount_of_samp_points = convert(Int64, max(500, round(2 * sqrt(2) * length(problem_definition.x_obs))))

Jones_line_plots(amount_of_samp_points, problem_definition, total_hyperparameters; file="figs/gp/$kernel_name/initial_gp", find_post=false)  # , plot_Σ=true, plot_Σ_profile=true)
Jones_line_plots(amount_of_samp_points, problem_definition, total_hyperparameters; file="figs/gp/$kernel_name/post_gp")  # , plot_Σ=true, plot_Σ_profile=true)

#################################
# Jones model fitting with FLux #
#################################

workspace = init_nlogL_Jones_matrix_workspace(problem_definition, total_hyperparameters)

# Allowing Flux to use the analytical gradients we have calculated
f_custom(non_zero_hyper) = nlogL_Jones(workspace, problem_definition, non_zero_hyper)
f_custom(non_zero_hyper::TrackedArray) = track(f_custom, non_zero_hyper)
g_custom() = ∇nlogL_Jones(workspace, problem_definition, data(non_zero_hyper_param))
@grad f_custom(non_zero_hyper) = f_custom(data(non_zero_hyper)), Δ -> tuple(Δ .* g_custom())

# initializing save of fit, or getting the most recent version, if it exists
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
    # Jones_line_plots(amount_of_samp_points, problem_definition, reconstruct_total_hyperparameters(problem_definition, data(non_zero_hyper_param)); file="figs/gp/$kernel_name/training/trained_" * training_time_str * "_gp")  # , plot_Σ_profile=true)
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

fit1_total_hyperparameters = reconstruct_total_hyperparameters(problem_definition, data(non_zero_hyper_param))

println("starting hyperparameters")
println(total_hyperparameters)
println(nlogL_Jones(problem_definition, total_hyperparameters), "\n")

println("ending hyperparameters")
println(fit1_total_hyperparameters)
println(nlogL_Jones(problem_definition, fit1_total_hyperparameters), "\n")

Jones_line_plots(amount_of_samp_points, problem_definition, fit1_total_hyperparameters; file="figs/gp/$kernel_name/fit_gp")  # , plot_Σ=true, plot_Σ_profile=true)

# coeffs = fit1_total_hyperparameters[1:end - problem_definition.n_kern_hyper]
# coeff_array = reconstruct_array(coeffs[findall(!iszero, coeffs)], problem_definition.a0)

# ################
# # Corner plots #
# ################

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

Σ_obs = Σ_observations(problem_definition, fit1_total_hyperparameters)

# making necessary variables local to all workers
# sendto(workers(), kernel_name=kernel_name)
# @everywhere include_kernel(kernel_name)
sendto(workers(), problem_definition=problem_definition, fit1_total_hyperparameters=fit1_total_hyperparameters, Σ_obs=Σ_obs)
@everywhere kep_likelihood_distributed(period::Real) = -nlogL_Jones(problem_definition, fit1_total_hyperparameters; Σ_obs=Σ_obs, P=period)

# parallelize with DistributedArrays
@everywhere using DistributedArrays
period_grid_dist = distribute(period_grid)
likelihoods = collect(map(kep_likelihood_distributed, period_grid_dist))

begin
    ax = init_plot()
    fig = plot(period_grid, likelihoods, color="black")
    xscale("log")
    ticklabel_format(style="sci", axis="y", scilimits=(0,0))
    xlabel("Periods (days)")
    ylabel("GP log likelihoods")
    axvline(x=convert_and_strip_units(u"d", P))
    axhline(y=-nlogL_Jones(problem_definition, fit1_total_hyperparameters), color="k")
    title_string = @sprintf "%.0f day, %.2f m/s" convert_and_strip_units(u"d",P) K
    title(title_string, fontsize=30)
    save_PyPlot_fig("figs/rv/$(kernel_name)_$(amount_of_periods).png")
end

# three best periods
best_period_grid = period_grid[find_modes(likelihoods; amount=10)]
println(best_period_grid)

################################################################
# Refitting GP with planet signals at found periods subtracted #
################################################################

# find first period that uses a bound eccentricity
best_period = best_period_grid[findfirst(y -> isless(y, 1), [fit_linear_kepler(problem_definition.y_obs, problem_definition.x_obs, period, Σ_obs; return_params=true)[3] for period in best_period_grid])]

println("original period: $(strip_units(P)) days")
println("found period:    $best_period days")

# Allowing Flux to use the analytical gradients we have calculated
f_custom2(non_zero_hyper) = nlogL_Jones(workspace, problem_definition, non_zero_hyper; P=best_period)
f_custom2(non_zero_hyper::TrackedArray) = track(f_custom2, non_zero_hyper)
g_custom2() = ∇nlogL_Jones(workspace, problem_definition, data(non_zero_hyper_param2); P=best_period)
@grad f_custom2(non_zero_hyper) = f_custom2(data(non_zero_hyper)), Δ -> tuple(Δ .* g_custom2())

# Setting model parameters for Flux
non_zero_hyper_param2 = param(data(non_zero_hyper_param))
ps2 = Flux.params(non_zero_hyper_param2)

# Final function wrapper for Flux
f_custom2() = f_custom2(non_zero_hyper_param2)

# setting things for Flux to use
flux_data = Iterators.repeated((), 500)  # use at most 500 iterations

# save plots as we are training every flux_cb_delay seconds
# stop training if our gradient norm gets small enough
flux_cb2 = function ()
    training_time_str = Dates.format(now(),"yyyy_mm_dd_HH_MM_SS")
    # Jones_line_plots(amount_of_samp_points, problem_definition, reconstruct_total_hyperparameters(problem_definition, data(non_zero_hyper_param)); file="figs/gp/$kernel_name/training/trained_" * training_time_str * "_gp", plot_Σ_profile=true)
    grad_norm = norm(g_custom2())
    println("Current time: " * training_time_str * " score: ", data(f_custom2()), " with gradient norm ", grad_norm)
    println("Current hyperparameters: " * string(data(non_zero_hyper_param2)))
    if grad_norm < grad_norm_thres
        println("Gradient threshold of $grad_norm_thres reached!")
        Flux.stop()
    end
    # update_optimize_Jones_model_jld2!(kernel_name, non_zero_hyper_param)
    println()
end

# @profiler
@elapsed Flux.train!(f_custom2, ps2, flux_data, opt, cb=Flux.throttle(flux_cb2, flux_cb_delay))

fit2_total_hyperparameters = reconstruct_total_hyperparameters(problem_definition, data(non_zero_hyper_param2))

println("starting hyperparameters")
println(total_hyperparameters)
println(nlogL_Jones(problem_definition, total_hyperparameters), "\n")

println("fit hyperparameters")
println(fit1_total_hyperparameters)
println(nlogL_Jones(problem_definition, fit1_total_hyperparameters), "\n")

println("refit hyperparameters")
println(fit2_total_hyperparameters)
println(nlogL_Jones(problem_definition, fit2_total_hyperparameters; P=best_period), "\n")

hold = copy(problem_definition.y_obs)
problem_definition.y_obs[:] = remove_kepler(problem_definition.y_obs, problem_definition.x_obs, best_period, Σ_obs)
Jones_line_plots(amount_of_samp_points, problem_definition, fit2_total_hyperparameters; file="figs/gp/$kernel_name/after")
problem_definition.y_obs[:] = hold

##########################
# Evidence approximation #
##########################

H1 = nlogprior_kernel_hyperparameters!(∇∇nlogL_Jones(problem_definition, fit1_total_hyperparameters), problem_definition.n_kern_hyper, fit1_total_hyperparameters)
nlogL_val1 = nlogL_Jones(problem_definition, fit1_total_hyperparameters) + nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, fit1_total_hyperparameters)
println("evidence for Jones model: " * string(log_laplace_approximation(H1, nlogL_val1, 0)))

H2 = nlogprior_kernel_hyperparameters!(∇∇nlogL_Jones(problem_definition, fit2_total_hyperparameters; P=best_period), problem_definition.n_kern_hyper, fit2_total_hyperparameters)
nlogL_val2 = nlogL_Jones(problem_definition, fit2_total_hyperparameters; P=best_period) + nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, fit2_total_hyperparameters)
Σ_obs1 = Σ_observations(problem_definition, fit2_total_hyperparameters)
println("best fit keplerian")
K1, e1, M01, ω1, γ1 = fit_linear_kepler(problem_definition.y_obs, problem_definition.x_obs, best_period, Σ_obs1; return_params=true, print_params=true)[2:end]
println("\noriginial injected keplerian")
println("K: $K, e: $e, M0: $M0, ω: $ω, γ: $γ")

llH2 = log_laplace_approximation(H2, nlogL_val2, 0)
println("evidence for Jones + planet model (no prior): " * string(llH2))
println("evidence for Jones + planet model: " * string(llH2 + logprior_kepler(best_period, e1, M01, K1, ω1, γ1)))
