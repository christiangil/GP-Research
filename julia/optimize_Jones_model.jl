include("src/all_functions.jl")

called_from_terminal = length(ARGS) > 0

###################################
# Loading data and setting kernel #
###################################

kernel_names = ["pp", "se", "m52", "rq", "rm52", "qp", "m52x2"]
initial_hypers = [[12.], [12], [12], [4, 12], [4, 12], [1, 20, 40], [12, 24, 1]]

# if called from terminal with an argument, use a full dataset. Otherwise, use a smaller testing set
sample_size = 100
if called_from_terminal
    kernel_choice = parse(Int, ARGS[1])
    kernel_name = kernel_names[kernel_choice]
    seed = parse(Int, ARGS[2])
    use_planet = length(ARGS) > 2
    # if isfile("csv_files/$(kernel_name)_logL_$seed.csv")
    #     println("results aleady calculated")
    #     exit()
    # end
else
    kernel_choice = 3
    kernel_name = kernel_names[kernel_choice]
    seed = 2
    use_planet = true
end

# allowing covariance matrix to be calculated in parallel
if !called_from_terminal
    try
        prep_parallel_covariance(kernel_name)
    catch
        prep_parallel_covariance(kernel_name)
    end
end

rng = MersenneTwister(seed)
sim_id = sample(rng, 6:20)
filename = "res-1000-1years_long_id$sim_id"
println("optimizing on $filename using the $kernel_name")
if called_from_terminal
    problem_def_base = init_problem_definition("jld2_files/" * filename; save_prob_def=false, sub_sample=sample_size, on_off=14, rng=rng)
else
    problem_def_base = init_problem_definition("../../../OneDrive/Desktop/jld2_files/" * filename; save_prob_def=false, sub_sample=sample_size, on_off=14, rng=rng)
    # problem_def_base = init_problem_definition("jld2_files/" * filename; save_prob_def=false, sub_sample=sample_size, on_off=14, rng=rng)
end

kernel_function, num_kernel_hyperparameters = include_kernel(kernel_name)
problem_definition = init_problem_definition(kernel_function, num_kernel_hyperparameters, problem_def_base)

########################################
# Adding planet and normalizing scores #
########################################

if use_planet
    P = (8 + 1 * randn())u"d"  # draw over more periods?
    e = rand() / 5
    M0 = 2 * π * rand()
    length(ARGS) > 1 ? K = parse(Float64, ARGS[3]) : K = 3.  # m/s
    ω = 2 * π * rand()
    γ = 0
    problem_definition.y_obs[:] = add_kepler_to_Jones_problem_definition(
        problem_definition, P, e, M0, K, ω; γ = γ,
        normalization=copy(problem_definition.normals[1]))
    results_dir = "results/$(kernel_name)/K_$(string(K))/seed_$(seed)/"
else
    results_dir = "results/$(kernel_name)/K_0.0/seed_$(seed)/"
end

begin
    mkpath(results_dir)
    normalize_problem_definition!(problem_definition)
    println("Score errors:")
    println(mean(problem_definition.noise[1:sample_size]) * problem_definition.normals[1], " m/s")
    for i in 2:problem_definition.n_out
        println(mean(problem_definition.noise[(i-1)*sample_size+1:i*sample_size]))
        @assert mean(problem_definition.noise[(i-1)*sample_size+1:i*sample_size]) < 2
    end
    println()
end


if kernel_name in ["pp", "se", "m52"]
    # normals
    parameters = gamma_mode_std_2_alpha_theta(10, 10)
    function kernel_hyper_priors(hps::Vector{<:Real}, d::Integer)
        return [log_gamma(hps[1], parameters; d=d)]
    end
elseif kernel_name in ["rq", "rm52"]
    # rationals
    paramsα = gamma_mode_std_2_alpha_theta(4, 10)
    paramsμ = gamma_mode_std_2_alpha_theta(10, 10)
    function kernel_hyper_priors(hps::Vector{<:Real}, d::Integer)
        return [log_gamma(hps[1], paramsα; d=d), log_gamma(hps[2], paramsμ; d=d)]
    end
elseif  kernel_name == "qp"
    # qp
    paramsλp = gamma_mode_std_2_alpha_theta(1, 1)
    paramsP = gamma_mode_std_2_alpha_theta(20, 10)
    paramsλse = gamma_mode_std_2_alpha_theta(40, 20)
    function kernel_hyper_priors(hps::Vector{<:Real}, d::Integer)
        return [log_gamma(hps[1], paramsλp; d=d), log_gamma(hps[2], paramsP; d=d), log_gamma(hps[3], paramsλse; d=d)]
    end
elseif kernel_name == "m52x2"
    # m52x2
    paramsλ1 = gamma_mode_std_2_alpha_theta(10, 10)
    paramsλ2 = gamma_mode_std_2_alpha_theta(20, 10)
    function kernel_hyper_priors(hps::Vector{<:Real}, d::Integer)
        return [log_gamma(hps[1], paramsλ1; d=d), log_gamma(hps[2], paramsλ2; d=d), log_gaussian(hps[3], [0, 3]; d=d)]
    end
end

function nlogprior_kernel_hyperparameters(n_kern_hyper::Integer, total_hyperparameters::Vector{<:Real}, d::Integer)
    hps = total_hyperparameters[(end - (n_kern_hyper - 1)):end]
    if d == 0
        return -sum(kernel_hyper_priors(hps, d))
    elseif d == 1
        return append!(zeros(length(total_hyperparameters) - n_kern_hyper), -kernel_hyper_priors(hps, d))
    elseif d == 2
        H = zeros(length(total_hyperparameters), length(total_hyperparameters))
        H[(end - (n_kern_hyper - 1)):end, (end - (n_kern_hyper - 1)):end] -= Diagonal(kernel_hyper_priors(hps, d))
        return H
    end
end


#####################################
# Initial hyperparameters and plots #
#####################################

# kernel hyper parameters
time_span = maximum(problem_definition.x_obs) - minimum(problem_definition.x_obs)
coeff_hyperparameters = collect(Iterators.flatten(problem_definition.a0))
coeff_hyperparameters[findall(!iszero, coeff_hyperparameters)] .= 1
total_hyperparameters = append!(coeff_hyperparameters, initial_hypers[kernel_choice] .* (0.8 .+ (0.4 .* rand(problem_definition.n_kern_hyper))))

# nlogL_Jones(problem_definition, remove_zeros(total_hyperparameters))

# how finely to sample the domain (for plotting)
amount_of_samp_points = convert(Int64, max(500, round(2 * sqrt(2) * length(problem_definition.x_obs))))

# Jones_line_plots(amount_of_samp_points, problem_definition, total_hyperparameters, "figs/gp/$kernel_name/seed$(seed)_initial_gp"; find_post=false)  # , plot_Σ=true, plot_Σ_profile=true)
Jones_line_plots(amount_of_samp_points, problem_definition, total_hyperparameters, results_dir * "init")  # ; plot_Σ=true, plot_Σ_profile=true)

#######################
# Jones model fitting #
#######################

workspace = init_nlogL_Jones_matrix_workspace(problem_definition, total_hyperparameters)

# #############
# # With FLux #
# #############
#
# using Flux; using Flux.Tracker: track, @grad, data
#
# flux_cb_delay = 3600 / 120
# grad_norm_thres = 5e-1
# opt = ADAM(0.2)
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

try
    using Optim
catch
    using Optim
end

# PosDefException means it tried a negative lengthscale

# storing initial hyperparameters
initial_x = remove_zeros(total_hyperparameters)

function f_no_print(non_zero_hyper::Vector{T}) where {T<:Real}
    # return nlogL_Jones!(workspace, problem_definition, non_zero_hyper)
    prior = nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, non_zero_hyper, 0)
    if prior == Inf
        return prior
    else
        return nlogL_Jones!(workspace, problem_definition, non_zero_hyper) + prior
    end
end

function f(non_zero_hyper::Vector{T}) where {T<:Real}
    println(non_zero_hyper)
    global current_hyper = copy(non_zero_hyper)
    return f_no_print(non_zero_hyper)
end

function g!(G::Vector{T}, non_zero_hyper::Vector{T}) where {T<:Real}
    # G[:] = ∇nlogL_Jones!(workspace, problem_definition, non_zero_hyper)
    G[:] = (∇nlogL_Jones!(workspace, problem_definition, non_zero_hyper)
        + nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, non_zero_hyper, 1))
end

function h!(H::Matrix{T}, non_zero_hyper::Vector{T}) where {T<:Real}
    # H[:, :] = ∇∇nlogL_Jones!(workspace, problem_definition, non_zero_hyper)
    H[:, :] = (∇∇nlogL_Jones!(workspace, problem_definition, non_zero_hyper)
     + nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, non_zero_hyper, 2))
end

# # creating path to save training figures
# mkpath("figs/gp/$kernel_name/training")

# ends optimization if true
function optim_cb(x::OptimizationState)
    println()
    if x.iteration > 0
        println("Iteration:     ", x.iteration)
        println("Time so far:   ", x.metadata["time"], " s")
        println("nlogL:         ", x.value)
        println("Gradient Norm: ", x.g_norm)
        println()
        # update_optimize_Jones_model_jld2!(kernel_name, non_zero_hyper_param)
    end
    # Jones_line_plots(amount_of_samp_points, problem_definition, reconstruct_total_hyperparameters(problem_definition, data(non_zero_hyper_param)), "figs/gp/$kernel_name/training/training_$(x.iteration)_gp")  # ; plot_Σ_profile=true)
    return false
end

# # second order
# @elapsed result = optimize(f, g!, h!, initial_x, NewtonTrustRegion(), Optim.Options(callback=optim_cb, iterations=1)) # 27s

try
    global result = optimize(f, g!, h!, initial_x, NewtonTrustRegion(), Optim.Options(callback=optim_cb, g_tol = 1e-6, iterations=200)) # 27s
    # global result = optimize(f, g!, initial_x, LBFGS(), Optim.Options(callback=optim_cb, g_tol = 1e-6, iterations=200)) # 27s
catch
    println("retrying fit")
    global result = optimize(f, g!, h!, current_hyper, NewtonTrustRegion(), Optim.Options(callback=optim_cb, g_tol = 1e-6, iterations=200))
    # global result = optimize(f, g!, current_hyper, LBFGS(), Optim.Options(callback=optim_cb, g_tol = 1e-6, iterations=200)) # 27s
end

# # second order
# @elapsed result = optimize(f, g!, h!, initial_x, NewtonTrustRegion(), Optim.Options(callback=optim_cb))  # 44s

# # first order
# @elapsed result = optimize(f, g!, initial_x, ConjugateGradient(), Optim.Options(callback=optim_cb))  # 44s
# @elapsed result = optimize(f, g!, initial_x, LBFGS(), Optim.Options(callback=optim_cb))  # 34s

# # constrained second order
# lx = append!(-10 * ones(length(remove_zeros(total_hyperparameters)) - problem_definition.n_kern_hyper), zeros(problem_definition.n_kern_hyper))
# ux = append!(10 * ones(length(remove_zeros(total_hyperparameters)) - problem_definition.n_kern_hyper), time_span * ones(problem_definition.n_kern_hyper))
# @elapsed result = optimize(TwiceDifferentiable(f, g!, h!, initial_x), TwiceDifferentiableConstraints(lx, ux), initial_x, IPNewton(), Optim.Options(callback=optim_cb)) # 50s

println(result)

fit1_total_hyperparameters = reconstruct_total_hyperparameters(problem_definition, result.minimizer)

############
# Post fit #
############

println("starting hyperparameters")
println(total_hyperparameters)
initial_nlogL = nlogL_Jones(problem_definition, total_hyperparameters)
println(initial_nlogL, "\n")

println("ending hyperparameters")
println(fit1_total_hyperparameters)
fit_nlogL = nlogL_Jones!(workspace, problem_definition, fit1_total_hyperparameters)
println(fit_nlogL, "\n")

Jones_line_plots(amount_of_samp_points, problem_definition, fit1_total_hyperparameters, results_dir * "fit")  # , plot_Σ=true, plot_Σ_profile=true)

# # coeffs = fit1_total_hyperparameters[1:end - problem_definition.n_kern_hyper]
# # coeff_array = reconstruct_array(coeffs[findall(!iszero, coeffs)], problem_definition.a0)

################
# Corner plots #
################

if called_from_terminal
    possible_labels = [
        [L"\lambda_{pp}"],
        [L"\lambda_{se}"],
        [L"\lambda_{m52}"],
        [L"\alpha" L"\mu"],
        [L"\alpha" L"\mu"],
        [L"\lambda_{p}" L"\tau_p" L"\lambda_{se}"],
        [L"\lambda_{1}" L"\lambda_{2}" L"\sqrt{ratio}"]]

    actual_labels = append!([L"a_{11}", L"a_{21}", L"a_{12}", L"a_{32}", L"a_{23}"], possible_labels[kernel_choice])
    corner_plot(f_no_print, remove_zeros(fit1_total_hyperparameters), results_dir * "corner.png"; input_labels=actual_labels)
    # corner_plot(f_no_print, remove_zeros(fit1_total_hyperparameters), "figs/gp/$kernel_name/seed$(seed)_corner.png"; input_labels=actual_labels, steps=3)
end

if use_planet

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
        title_string = @sprintf "%.1f day, %.2f m/s" convert_and_strip_units(u"d",P) K
        title(title_string, fontsize=30)

        save_PyPlot_fig(results_dir * "periodogram.png")
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

    function f_no_print(non_zero_hyper::Vector{T}) where {T<:Real}
        # return nlogL_Jones!(workspace, problem_definition, non_zero_hyper; P=best_period)
        prior = nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, non_zero_hyper, 0)
        if prior == Inf
            return prior
        else
            return nlogL_Jones!(workspace, problem_definition, non_zero_hyper; P=best_period) + prior
        end
    end

    function f(non_zero_hyper::Vector{T}) where {T<:Real}
        println(non_zero_hyper)
        global current_hyper = copy(non_zero_hyper)
        return f_no_print(non_zero_hyper)
    end

    function g!(G::Vector{T}, non_zero_hyper::Vector{T}) where {T<:Real}
        # G[:] = ∇nlogL_Jones!(workspace, problem_definition, non_zero_hyper; P=best_period)
        G[:] = (∇nlogL_Jones!(workspace, problem_definition, non_zero_hyper; P=best_period)
            + nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, non_zero_hyper, 1))
    end

    function h!(H::Matrix{T}, non_zero_hyper::Vector{T}) where {T<:Real}
        # H[:, :] = ∇∇nlogL_Jones!(workspace, problem_definition, non_zero_hyper; P=best_period)
        H[:, :] = (∇∇nlogL_Jones!(workspace, problem_definition, non_zero_hyper; P=best_period)
         + nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, non_zero_hyper, 2))
    end

    # second order
    try
        global result = optimize(f, g!, h!, initial_x, NewtonTrustRegion(), Optim.Options(callback=optim_cb, g_tol = 1e-6, iterations=200)) # 27s
    catch
        println("retrying fit")
        global result = optimize(f, g!, h!, current_hyper, NewtonTrustRegion(), Optim.Options(callback=optim_cb, g_tol = 1e-6, iterations=200))
    end

    println(result)

    fit2_total_hyperparameters = reconstruct_total_hyperparameters(problem_definition, result.minimizer)

    ###################
    # Post planet fit #
    ###################

    println("starting hyperparameters")
    println(total_hyperparameters)
    println(initial_nlogL, "\n")

    println("fit hyperparameters")
    println(fit1_total_hyperparameters)
    println(fit_nlogL, "\n")

    println("refit hyperparameters")
    println(fit2_total_hyperparameters)
    refit_nlogL = nlogL_Jones!(workspace, problem_definition, fit2_total_hyperparameters; P=best_period)
    println(refit_nlogL, "\n")

    hold = copy(problem_definition.y_obs)
    problem_definition.y_obs[:] = remove_kepler(problem_definition.y_obs, problem_definition.x_obs, best_period, Σ_obs)

    Jones_line_plots(amount_of_samp_points, problem_definition, fit2_total_hyperparameters, results_dir * "fit_planet")

    problem_definition.y_obs[:] = hold

    if called_from_terminal
        possible_labels = [
            [L"\lambda_{pp}"],
            [L"\lambda_{se}"],
            [L"\lambda_{m52}"],
            [L"\alpha" L"\mu"],
            [L"\alpha" L"\mu"],
            [L"\lambda_{p}" L"\tau_p" L"\lambda_{se}"],
            [L"\lambda_{1}" L"\lambda_{2}" L"\sqrt{ratio}"]]

        actual_labels = append!([L"a_{11}", L"a_{21}", L"a_{12}", L"a_{32}", L"a_{23}"], possible_labels[kernel_choice])
        corner_plot(f_no_print, remove_zeros(fit2_total_hyperparameters), results_dir * "corner_planet.png"; input_labels=actual_labels)
    end

    ##########################
    # Evidence approximation #
    ##########################

    H1 = (∇∇nlogL_Jones!(workspace, problem_definition, fit1_total_hyperparameters)
        + nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, remove_zeros(fit1_total_hyperparameters), 2))
    # H1 = ∇∇nlogL_Jones(problem_definition, fit1_total_hyperparameters)
    nlogL_val1 = fit_nlogL + nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, fit1_total_hyperparameters, 0)
    llH1 = log_laplace_approximation(H1, nlogL_val1, 0)

    H2 = (∇∇nlogL_Jones!(workspace, problem_definition, fit2_total_hyperparameters; P=best_period)
        + nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, remove_zeros(fit2_total_hyperparameters), 2))
    # H2 = ∇∇nlogL_Jones(problem_definition, fit2_total_hyperparameters; P=best_period)
    nlogL_val2 = refit_nlogL + nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, fit2_total_hyperparameters, 0)
    Σ_obs_final = Σ_observations(problem_definition, fit2_total_hyperparameters)
    println("best fit keplerian")
    K2, e2, M02, ω2, γ2 = fit_linear_kepler(problem_definition.y_obs, problem_definition.x_obs, best_period, Σ_obs_final; return_params=true, print_params=true)[2:end]
    println("originial injected keplerian")
    println("K: $K, e: $e, M0: $M0, ω: $ω, γ: $γ")

    # diag(H2)
    # nlogL_val2

    llH2 = log_laplace_approximation(H2, nlogL_val2, 0)
    llH2_wp = llH2 + logprior_kepler(best_period, e2, M02, K2, ω2, γ2)
    println("\nevidence for Jones model: " * string(llH1))
    println("evidence for Jones + planet model (no prior): " * string(llH2))
    println("evidence for Jones + planet model: " * string(llH2_wp))

    likelihoods = [-nlogL_val1, llH1, -nlogL_val2, llH2, llH2_wp]
    orbit_params = [K, e, M0, ω, γ, K2, e2, M02, ω2, γ2]
    save_nlogLs(seed, sim_id, likelihoods, append!(copy(fit1_total_hyperparameters), fit2_total_hyperparameters), orbit_params, results_dir)
end
