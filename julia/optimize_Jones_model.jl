# adding in custom functions
include("src/setup.jl")
# include("test/runtests.jl")
include("src/all_functions.jl")

###################################
# Loading data and setting kernel #
###################################

using Flux; using Flux.Tracker: track, @grad, data

# kernel_names = ["quasi_periodic_kernel", "se_kernel", "rq_kernel", "matern52_kernel"]
kernel_names = ["quasi_periodic_kernel", "se_kernel", "matern52_kernel", "rq_kernel"]

# if called from terminal with an argument, use a full dataset. Otherwise, use a smaller testing set
if length(ARGS)>0
    kernel_name = kernel_names[parse(Int, ARGS[1])]
    # id = parse(Int, ARGS[2])
    id = sample(1:50)
    println("optimizing the full problem using the $kernel_name")
    # @load "jld2_files/res-1000-1years_full_id$(id)_problem_def_full_base.jld2" problem_def_base
    problem_def_base = init_problem_definition("jld2_files/res-1000-1years_full_id$id"; save_prob_def=false, sub_sample=100, on_off=14)
    kernel_function, num_kernel_hyperparameters = include_kernel(kernel_name)
    problem_definition = init_problem_definition(kernel_function, num_kernel_hyperparameters, problem_def_base)
    flux_cb_delay = 3600 / 30  # 2
    grad_norm_thres = 5e-1
    opt = ADAM(0.1)
else
    kernel_name = kernel_names[3]
    id = 41
    # @load "jld2_files/res-1000-lambda-3923-6664-1years_1579spots_diffrot_id11_problem_def_sample_base.jld2" problem_def_base
    # @load "jld2_files/res-1000-lambda-3923-6664-1years_1579spots_diffrot_id11_problem_def_full_base.jld2" problem_def_base
    problem_def_base = init_problem_definition("jld2_files/res-1000-1years_full_id$id"; save_prob_def=false, sub_sample=100, on_off=14)
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

# P = (20 + 3 * randn())u"d"
# e = rand() / 5
# M0 = 2 * π * rand()
# length(ARGS) > 1 ? K = parse(Float64, ARGS[2]) : K = 0.5  # m/s
# ω = 2 * π * rand()
# γ = 0
# problem_definition.y_obs[:] = add_kepler_to_Jones_problem_definition(
#     problem_definition, P, e, M0, K, ω; γ = γ,
#     normalization=problem_definition.normals[1])

normalize_problem_definition!(problem_definition)

# mean(problem_definition.noise[71:140])
# mean(problem_definition.noise[141:210])

#####################################
# Initial hyperparameters and plots #
#####################################

# kernel hyper parameters
time_span = maximum(problem_definition.x_obs) - minimum(problem_definition.x_obs)
coeff_hyperparameters = collect(Iterators.flatten(problem_definition.a0))
coeff_hyperparameters[findall(!iszero, coeff_hyperparameters)] .= 1
total_hyperparameters = append!(coeff_hyperparameters, minimum([7, time_span/10]) .* ones(problem_definition.n_kern_hyper))

# how finely to sample the domain (for plotting)
amount_of_samp_points = convert(Int64, max(500, round(2 * sqrt(2) * length(problem_definition.x_obs))))

# Jones_line_plots(amount_of_samp_points, problem_definition, total_hyperparameters, "figs/gp/$kernel_name/initial_gp"; find_post=false)  # , plot_Σ=true, plot_Σ_profile=true)
# Jones_line_plots(amount_of_samp_points, problem_definition, total_hyperparameters, "figs/gp/$kernel_name/post_gp")  # , plot_Σ=true, plot_Σ_profile=true)

#######################
# Jones model fitting #
#######################

workspace = init_nlogL_Jones_matrix_workspace(problem_definition, total_hyperparameters)

# #############
# # With FLux #
# #############
#
# # Allowing Flux to use the analytical gradients we have calculated
# f_custom(non_zero_hyper) = nlogL_Jones(workspace, problem_definition, non_zero_hyper)
# f_custom(non_zero_hyper::TrackedArray) = track(f_custom, non_zero_hyper)
# g_custom() = ∇nlogL_Jones(workspace, problem_definition, data(non_zero_hyper_param))
# @grad f_custom(non_zero_hyper) = f_custom(data(non_zero_hyper)), Δ -> tuple(Δ .* g_custom())
#
# # initializing save of fit, or getting the most recent version, if it exists
# # current_params = initialize_optimize_Jones_model_jld2!(kernel_name, total_hyperparameters[findall(!iszero, total_hyperparameters)])
# current_params = remove_zeros(total_hyperparameters)
#
# # Setting model parameters for Flux
# non_zero_hyper_param = param(current_params)
# ps = Flux.params(non_zero_hyper_param)
#
# # Final function wrapper for Flux
# f_custom() = f_custom(non_zero_hyper_param)
#
# # setting things for Flux to use
# flux_data = Iterators.repeated((), 2000)  # use at most 500 iterations
#
# # save plots as we are training every flux_cb_delay seconds
# # stop training if our gradient norm gets small enough
# flux_cb = function ()
#     training_time_str = Dates.format(now(),"yyyy_mm_dd_HH_MM_SS")
#     # Jones_line_plots(amount_of_samp_points, problem_definition, reconstruct_total_hyperparameters(problem_definition, data(non_zero_hyper_param)), "figs/gp/$kernel_name/training/trained_" * training_time_str * "_gp")  # ; plot_Σ_profile=true)
#     grad_norm = norm(g_custom())
#     println("Current time: " * training_time_str * " score: ", data(f_custom()), " with gradient norm ", grad_norm)
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
# @elapsed Flux.train!(f_custom, ps, flux_data, opt, cb=Flux.throttle(flux_cb, flux_cb_delay))  # 180-300s
#
# # update_optimize_Jones_model_jld2!(kernel_name, non_zero_hyper_param)
#
# fit1_total_hyperparameters = reconstruct_total_hyperparameters(problem_definition, data(non_zero_hyper_param))

##############
# With Optim #
##############

using Optim

# PosDefException means it tried a negative lengthscale

# storing initial hyperparameters
initial_x = remove_zeros(total_hyperparameters)

function f(non_zero_hyper::Vector{T}) where {T<:Real}
    println(non_zero_hyper)
    return nlogL_Jones!(workspace, problem_definition, non_zero_hyper)
end

function g!(G::Vector{T}, non_zero_hyper::Vector{T}) where {T<:Real}
    G[:] = ∇nlogL_Jones!(workspace, problem_definition, non_zero_hyper)
end

function h!(H::Matrix{T}, non_zero_hyper::Vector{T}) where {T<:Real}
    H[:, :] = ∇∇nlogL_Jones!(workspace, problem_definition, non_zero_hyper)
end

start_time = [time()]
# ends optimization if true
function optim_cb(x)
    if x.iteration == 0
        println()
        start_time[1] = time()
    else
        so_far =  time() - start_time[1]
        println("Iteration:     ", x.iteration)
        println("Time so far:   ", so_far)
        println("nLogL:         ", x.value)
        println("Gradient Norm: ", x.g_norm)
        println()
        # update_optimize_Jones_model_jld2!(kernel_name, non_zero_hyper_param)
    end
    # Jones_line_plots(amount_of_samp_points, problem_definition, reconstruct_total_hyperparameters(problem_definition, data(non_zero_hyper_param)), "figs/gp/$kernel_name/training/trained_" * training_time_str * "_gp")  # ; plot_Σ_profile=true)
    return false
end

# # second order
# # @elapsed result = optimize(f, g!, h!, initial_x, Newton())  # PosDefException, makes lengthscale negative in first iteration
# @elapsed result = optimize(f, g!, h!, initial_x, NewtonTrustRegion(), Optim.Options(callback=optim_cb)) # 27s
#
# # # first order
# # @elapsed result = optimize(f, g!, initial_x, ConjugateGradient(), Optim.Options(callback=optim_cb))  # 44s
# # @elapsed result = optimize(f, g!, initial_x, LBFGS(), Optim.Options(callback=optim_cb))  # 34s
#
# # # constrained second order
# # lx = append!(-10 * ones(length(remove_zeros(total_hyperparameters)) - problem_definition.n_kern_hyper), zeros(problem_definition.n_kern_hyper))
# # ux = append!(10 * ones(length(remove_zeros(total_hyperparameters)) - problem_definition.n_kern_hyper), time_span * ones(problem_definition.n_kern_hyper))
# # @elapsed result = optimize(TwiceDifferentiable(f, g!, h!, initial_x), TwiceDifferentiableConstraints(lx, ux), initial_x, IPNewton(), Optim.Options(callback=optim_cb)) # 50s
#
# # using preconditioning
try
    result = optimize(f, g!, initial_x, LBFGS(P = precond(length(initial_x))), Optim.Options(callback=optim_cb))  # 21s
catch
    result = optimize(f, g!, initial_x, ConjugateGradient(P = precond(length(initial_x))), Optim.Options(callback=optim_cb))  # 25s
end

fit1_total_hyperparameters = reconstruct_total_hyperparameters(problem_definition, result.minimizer)

############
# Post fit #
############

println("starting hyperparameters")
println(total_hyperparameters)
println(nlogL_Jones(problem_definition, total_hyperparameters), "\n")

println("ending hyperparameters")
println(fit1_total_hyperparameters)
fit_nLogL = nlogL_Jones(problem_definition, fit1_total_hyperparameters)
println(nlogL_Jones(problem_definition, fit1_total_hyperparameters), "\n")

# save_nlogLs!(fit_nLogL, id, fit1_total_hyperparameters, kernel_name)

Jones_line_plots(amount_of_samp_points, problem_definition, fit1_total_hyperparameters, "figs/gp/$kernel_name/id$(id)_fit_gp")  # , plot_Σ=true, plot_Σ_profile=true)

# # coeffs = fit1_total_hyperparameters[1:end - problem_definition.n_kern_hyper]
# # coeff_array = reconstruct_array(coeffs[findall(!iszero, coeffs)], problem_definition.a0)

# # ################
# # # Corner plots #
# # ################
#
# # possible_labels = [L"a_{11}" L"a_{21}" L"a_{12}" L"a_{32}" L"a_{23}" L"\lambda_{SE}" L"\lambda_{P}" L"\tau_P";
# #     L"a_{11}" L"a_{21}" L"a_{12}" L"a_{32}" L"a_{23}" L"\lambda_{SE}" L" " L" ";
# #     L"a_{11}" L"a_{21}" L"a_{12}" L"a_{32}" L"a_{23}" L"\alpha" L"\lambda_{RQ}" L" ";
# #     L"a_{11}" L"a_{21}" L"a_{12}" L"a_{32}" L"a_{23}" L"\lambda_{M52}" L" " L" "]
# #
# # @load "jld2_files/optimize_Jones_model_$kernel_name.jld2" current_params
# # actual_labels = possible_labels[1, 1:length(current_params)]
# # f_corner(input) = nlogL_Jones(problem_definition, input)
# # @elapsed corner_plot(f_corner, current_params, "figs/gp/$kernel_name/corner_$kernel_name.png"; input_labels=actual_labels)

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

# #############
# # With FLux #
# #############
#
# # Allowing Flux to use the analytical gradients we have calculated
# f_custom(non_zero_hyper) = nlogL_Jones(workspace, problem_definition, non_zero_hyper; P=best_period)
# f_custom(non_zero_hyper::TrackedArray) = track(f_custom, non_zero_hyper)
# g_custom() = ∇nlogL_Jones(workspace, problem_definition, data(non_zero_hyper_param2); P=best_period)
# @grad f_custom(non_zero_hyper) = f_custom(data(non_zero_hyper)), Δ -> tuple(Δ .* g_custom())
#
# # Setting model parameters for Flux
# non_zero_hyper_param2 = param(data(non_zero_hyper_param))
# ps2 = Flux.params(non_zero_hyper_param2)
#
# # Final function wrapper for Flux
# f_custom() = f_custom(non_zero_hyper_param2)
#
# # @profiler
# @elapsed Flux.train!(f_custom, ps2, flux_data, opt, cb=Flux.throttle(flux_cb, flux_cb_delay))
#
# fit2_total_hyperparameters = reconstruct_total_hyperparameters(problem_definition, data(non_zero_hyper_param2))

##############
# With Optim #
##############

# storing initial hyperparameters
initial_x = remove_zeros(fit1_total_hyperparameters)

function f(non_zero_hyper::Vector{T}) where {T<:Real}
    println(non_zero_hyper)
    return nlogL_Jones!(workspace, problem_definition, non_zero_hyper; P=best_period)
end

function g!(G::Vector{T}, non_zero_hyper::Vector{T}) where {T<:Real}
    G[:] = ∇nlogL_Jones!(workspace, problem_definition, non_zero_hyper; P=best_period)
end

function h!(H::Matrix{T}, non_zero_hyper::Vector{T}) where {T<:Real}
    H[:, :] = ∇∇nlogL_Jones!(workspace, problem_definition, non_zero_hyper; P=best_period)
end

# # second order
# @elapsed result = optimize(f, g!, h!, initial_x, NewtonTrustRegion(), Optim.Options(callback=optim_cb)) # 27s

# # using preconditioning
try
    result = optimize(f, g!, initial_x, LBFGS(P = precond(length(initial_x))), Optim.Options(callback=optim_cb))  # 21s
catch
    result = optimize(f, g!, initial_x, ConjugateGradient(P = precond(length(initial_x))), Optim.Options(callback=optim_cb))  # 25s
end


fit2_total_hyperparameters = reconstruct_total_hyperparameters(problem_definition, result.minimizer)

# ###################
# # Post planet fit #
# ###################
#
# println("starting hyperparameters")
# println(total_hyperparameters)
# println(nlogL_Jones(problem_definition, total_hyperparameters), "\n")
#
# println("fit hyperparameters")
# println(fit1_total_hyperparameters)
# println(nlogL_Jones(problem_definition, fit1_total_hyperparameters), "\n")
#
# println("refit hyperparameters")
# println(fit2_total_hyperparameters)
# println(nlogL_Jones(problem_definition, fit2_total_hyperparameters; P=best_period), "\n")
#
# hold = copy(problem_definition.y_obs)
# problem_definition.y_obs[:] = remove_kepler(problem_definition.y_obs, problem_definition.x_obs, best_period, Σ_obs)
# Jones_line_plots(amount_of_samp_points, problem_definition, fit2_total_hyperparameters, "figs/gp/$kernel_name/after")
# problem_definition.y_obs[:] = hold

# ##########################
# # Evidence approximation #
# ##########################
#
# H1 = nlogprior_kernel_hyperparameters!(∇∇nlogL_Jones(problem_definition, fit1_total_hyperparameters), problem_definition.n_kern_hyper, fit1_total_hyperparameters)
# nlogL_val1 = nlogL_Jones(problem_definition, fit1_total_hyperparameters) + nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, fit1_total_hyperparameters)
# println("evidence for Jones model: " * string(log_laplace_approximation(H1, nlogL_val1, 0)))
#
# H2 = nlogprior_kernel_hyperparameters!(∇∇nlogL_Jones(problem_definition, fit2_total_hyperparameters; P=best_period), problem_definition.n_kern_hyper, fit2_total_hyperparameters)
# nlogL_val2 = nlogL_Jones(problem_definition, fit2_total_hyperparameters; P=best_period) + nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, fit2_total_hyperparameters)
# Σ_obs1 = Σ_observations(problem_definition, fit2_total_hyperparameters)
# println("best fit keplerian")
# K1, e1, M01, ω1, γ1 = fit_linear_kepler(problem_definition.y_obs, problem_definition.x_obs, best_period, Σ_obs1; return_params=true, print_params=true)[2:end]
# println("\noriginial injected keplerian")
# println("K: $K, e: $e, M0: $M0, ω: $ω, γ: $γ")
#
# # diag(H2)
# # nlogL_val2
#
# llH2 = log_laplace_approximation(H2, nlogL_val2, 0)
# println("evidence for Jones + planet model (no prior): " * string(llH2))
# println("evidence for Jones + planet model: " * string(llH2 + logprior_kepler(best_period, e1, M01, K1, ω1, γ1)))
