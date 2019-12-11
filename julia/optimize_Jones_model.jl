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
else
    kernel_choice = 3
    kernel_name = kernel_names[kernel_choice]
    seed = 2
end

# allowing covariance matrix to be calculated in parallel
if !called_from_terminal
    try
        prep_parallel_covariance(kernel_name)
    catch
        prep_parallel_covariance(kernel_name)
    end
end

kernel_function, num_kernel_hyperparameters = include_kernel(kernel_name)

rng = MersenneTwister(seed)
sim_id = sample(rng, 6:20)
fname = "res-1000-1years_long_id$sim_id"
println("optimizing on $fname using the $kernel_name")
if called_from_terminal
    problem_definition = Jones_problem_definition(kernel_function, num_kernel_hyperparameters,"jld2_files/" * fname; sub_sample=sample_size, on_off=14u"d", rng=rng)
else
    # problem_definition = Jones_problem_definition(kernel_function, num_kernel_hyperparameters,"../../../OneDrive/Desktop/jld2_files/" * fname; sub_sample=sample_size, on_off=14u"d", rng=rng, n_out=2)
    problem_definition = Jones_problem_definition(kernel_function, num_kernel_hyperparameters,"jld2_files/" * fname; sub_sample=sample_size, on_off=14u"d", rng=rng, n_out=2)
end

########################################
# Adding planet and normalizing scores #
########################################

length(ARGS) > 1 ? K_val = parse(Float64, ARGS[3]) : K_val = 3.  # m/s
# draw over more periods?
original_ks = kep_signal(K_val * u"m/s", (8 + 1 * randn())u"d", 2 * π * rand(), rand() / 5, 2 * π * rand(), 0u"m/s")
add_kepler_to_Jones_problem_definition!(problem_definition, original_ks)
results_dir = "results/$(kernel_name)/K_$(string(ustrip(original_ks.K)))/seed_$(seed)/"

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
total_hyperparameters = append!(coeff_hyperparameters, initial_hypers[kernel_choice] .* (0.8 .+ (0.4 .* rand(problem_definition.n_kern_hyper))))

# how finely to sample the domain (for plotting)
amount_of_samp_points = convert(Int64, max(500, round(2 * sqrt(2) * length(problem_definition.x_obs))))

# Jones_line_plots(amount_of_samp_points, problem_definition, total_hyperparameters, "figs/gp/$kernel_name/seed$(seed)_initial_gp"; find_post=false)  # , plot_Σ=true, plot_Σ_profile=true)
Jones_line_plots(amount_of_samp_points, problem_definition, total_hyperparameters, results_dir * "init")

#######################
# Jones model fitting #
#######################

workspace = init_nlogL_matrix_workspace(problem_definition, total_hyperparameters)

try
    using Optim
catch
    using Optim
end

# storing initial hyperparameters
initial_x = remove_zeros(total_hyperparameters)

function f_no_print(non_zero_hyper::Vector{T}) where {T<:Real}
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
    G[:] = (∇nlogL_Jones!(workspace, problem_definition, non_zero_hyper)
        + nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, non_zero_hyper, 1))
end

function h!(H::Matrix{T}, non_zero_hyper::Vector{T}) where {T<:Real}
    # H[:, :] = ∇∇nlogL_Jones!(workspace, problem_definition, non_zero_hyper)
    H[:, :] = (∇∇nlogL_Jones!(workspace, problem_definition, non_zero_hyper)
     + nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, non_zero_hyper, 2))
end

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

try
    global result = optimize(f, g!, h!, initial_x, NewtonTrustRegion(), Optim.Options(callback=optim_cb, g_tol=1e-6, iterations=200)) # 27s
    # global result = optimize(f, g!, initial_x, LBFGS(), Optim.Options(callback=optim_cb, g_tol=1e-6, iterations=200)) # 27s
catch
    println("retrying fit")
    global result = optimize(f, g!, h!, current_hyper, NewtonTrustRegion(), Optim.Options(callback=optim_cb, g_tol=1e-6, iterations=200))
    # global result = optimize(f, g!, current_hyper, LBFGS(), Optim.Options(callback=optim_cb, g_tol=1e-6, iterations=200)) # 27s
end

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


###########################################################################
# Evaluating GP likelihoods after taking out planets at different periods #
###########################################################################

amount_of_periods = 2048

# sample linearly in frequency space so that we get periods from the 1 / uneven Nyquist
# frequency to 4 times the total timespan of the data
freq_grid = linspace(1 / (problem_definition.time[end] - problem_definition.time[1]) / 4, uneven_nyquist_frequency(problem_definition.time), amount_of_periods)
period_grid = 1 ./ reverse(freq_grid)

Σ_obs = Σ_observations(problem_definition, fit1_total_hyperparameters)

# making necessary variables local to all workers
sendto(workers(), problem_definition=problem_definition, fit1_total_hyperparameters=fit1_total_hyperparameters, Σ_obs=Σ_obs)
@everywhere kep_likelihood_distributed(P::Unitful.Time) =
    -nlogL_Jones(
        problem_definition,
        fit1_total_hyperparameters;
        Σ_obs=Σ_obs,
        y_obs=fit_and_remove_kepler(problem_definition, Σ_obs, kep_signal_epicyclic(P=P)))

# parallelize with DistributedArrays
@everywhere using DistributedArrays
period_grid_dist = distribute(period_grid)
likelihoods = collect(map(kep_likelihood_distributed, period_grid_dist))

begin
    ax = init_plot()
    fig = plot(ustrip.(period_grid), likelihoods, color="black")
    xscale("log")
    ticklabel_format(style="sci", axis="y", scilimits=(0,0))
    xlabel("Periods (days)")
    ylabel("GP log likelihoods")
    axvline(x=convert_and_strip_units(u"d", original_ks.P))
    axhline(y=-nlogL_Jones(problem_definition, fit1_total_hyperparameters), color="k")
    title_string = @sprintf "%.1f day, %.2f m/s" convert_and_strip_units(u"d", original_ks.P) convert_and_strip_units(u"m/s",original_ks.K)
    title(title_string, fontsize=30)

    save_PyPlot_fig(results_dir * "periodogram.png")
end

best_period_grid = period_grid[find_modes(likelihoods; amount=10)]

################################################################
# Refitting GP with planet signals at found periods subtracted #
################################################################

# find first period that uses a bound eccentricity
best_period = best_period_grid[findfirst(y -> isless(y, 0.3), [fit_kepler(
    problem_definition, Σ_obs, kep_signal_epicyclic(;P=period)
    ).e for period in best_period_grid])]

println("original period: $(ustrip(original_ks.P)) days")
println("found period:    $(ustrip(best_period)) days")

# storing initial hyperparameters
initial_x = remove_zeros(fit1_total_hyperparameters)

fit_and_remove_kepler_epi_short() = fit_and_remove_kepler(
    problem_definition,
    workspace.Σ_obs,
    kep_signal_epicyclic(P=best_period))

function f_no_print(non_zero_hyper::Vector{T}) where {T<:Real}
    prior = nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, non_zero_hyper, 0)
    if prior == Inf
        return prior
    else
        return nlogL_Jones!(workspace, problem_definition, non_zero_hyper; y_obs=fit_and_remove_kepler_epi_short()) + prior
    end
end

function f(non_zero_hyper::Vector{T}) where {T<:Real}
    println(non_zero_hyper)
    global current_hyper = copy(non_zero_hyper)
    return f_no_print(non_zero_hyper)
end

function g!(G::Vector{T}, non_zero_hyper::Vector{T}) where {T<:Real}
    G[:] = (∇nlogL_Jones!(workspace, problem_definition, non_zero_hyper; y_obs=fit_and_remove_kepler_epi_short())
        + nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, non_zero_hyper, 1))
end

function h!(H::Matrix{T}, non_zero_hyper::Vector{T}) where {T<:Real}
    H[:, :] = (∇∇nlogL_Jones!(workspace, problem_definition, non_zero_hyper; y_obs=fit_and_remove_kepler_epi_short())
     + nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, non_zero_hyper, 2))
end

# second order
try
    global result = optimize(f, g!, h!, initial_x, NewtonTrustRegion(), Optim.Options(callback=optim_cb, g_tol=1e-6, iterations=200)) # 27s
catch
    println("retrying fit")
    global result = optimize(f, g!, h!, current_hyper, NewtonTrustRegion(), Optim.Options(callback=optim_cb, g_tol=1e-6, iterations=200))
end

println(result)

fit2_total_hyperparameters = reconstruct_total_hyperparameters(problem_definition, result.minimizer)

###################
# Post planet fit #
###################

println("fit hyperparameters")
println(fit1_total_hyperparameters)
println(fit_nlogL, "\n")

println("epicyclic hyperparameters")
println(fit2_total_hyperparameters)
refit_nlogL = nlogL_Jones!(workspace, problem_definition, fit2_total_hyperparameters; y_obs=fit_and_remove_kepler_epi_short())
println(refit_nlogL, "\n")

Σ_obs_final = Σ_observations(problem_definition, fit2_total_hyperparameters)

epi_ks = fit_kepler(problem_definition, Σ_obs_final, kep_signal_epicyclic(;P=best_period))
Jones_line_plots(amount_of_samp_points, problem_definition, fit2_total_hyperparameters, results_dir * "fit_epi"; fit_ks=epi_ks)

###################################################################
# Refitting GP with full planet signal at found period subtracted #
###################################################################

# storing initial hyperparameters
initial_x = remove_zeros(fit2_total_hyperparameters)
global current_ks = fit_kepler(problem_definition, workspace.Σ_obs, kep_signal(epi_ks.K, epi_ks.P, epi_ks.M0, epi_ks.e, epi_ks.ω, epi_ks.γ))
global hyper_for_ks = copy(workspace.nlogL_hyperparameters)

function fit_and_remove_kepler_short()
    if workspace.nlogL_hyperparameters!=hyper_for_ks
        global hyper_for_ks[:] = workspace.nlogL_hyperparameters[:]
        global current_ks = fit_kepler(problem_definition, workspace.Σ_obs, current_ks)
    end
    return remove_kepler(problem_definition, current_ks)
end

function f_no_print(non_zero_hyper::Vector{T}) where {T<:Real}
    prior = nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, non_zero_hyper, 0)
    if prior == Inf
        return prior
    else
        return nlogL_Jones!(workspace, problem_definition, non_zero_hyper; y_obs=fit_and_remove_kepler_short()) + prior
    end
end

function f(non_zero_hyper::Vector{T}) where {T<:Real}
    println(non_zero_hyper)
    global current_hyper = copy(non_zero_hyper)
    return f_no_print(non_zero_hyper)
end

function g!(G::Vector{T}, non_zero_hyper::Vector{T}) where {T<:Real}
    G[:] = (∇nlogL_Jones!(workspace, problem_definition, non_zero_hyper; y_obs=fit_and_remove_kepler_short())
        + nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, non_zero_hyper, 1))
end

function h!(H::Matrix{T}, non_zero_hyper::Vector{T}) where {T<:Real}
    H[:, :] = (∇∇nlogL_Jones!(workspace, problem_definition, non_zero_hyper; y_obs=fit_and_remove_kepler_short())
     + nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, non_zero_hyper, 2))
end

# second order
try
    global result = optimize(f, g!, h!, initial_x, NewtonTrustRegion(), Optim.Options(callback=optim_cb, g_tol=1e-6, f_tol=1e-10, iterations=200)) # 27s
catch
    println("retrying fit")
    global result = optimize(f, g!, h!, current_hyper, NewtonTrustRegion(), Optim.Options(callback=optim_cb, g_tol=1e-6, f_tol=1e-10, iterations=200))
end

println(result)

fit3_total_hyperparameters = reconstruct_total_hyperparameters(problem_definition, result.minimizer)

###################
# Post planet fit #
###################

println("epicyclic hyperparameters")
println(fit2_total_hyperparameters)
println(refit_nlogL, "\n")

println("full kepler hyperparameters")
println(fit3_total_hyperparameters)
refit_nlogL2 = nlogL_Jones!(workspace, problem_definition, fit3_total_hyperparameters; y_obs=fit_and_remove_kepler_short())
println(refit_nlogL2, "\n")

Σ_obs_final2 = Σ_observations(problem_definition, fit3_total_hyperparameters)

full_ks = kep_signal(current_ks.K, current_ks.P, current_ks.M0, current_ks.e, current_ks.ω, current_ks.γ)
Jones_line_plots(amount_of_samp_points, problem_definition, fit3_total_hyperparameters, results_dir * "fit_full"; fit_ks=full_ks)

println("best fit keplerian")
println(kep_parms_str(full_ks))
println("originial injected keplerian")
println(kep_parms_str(original_ks))

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
    corner_plot(f_no_print, remove_zeros(fit3_total_hyperparameters), results_dir * "corner_planet.png"; input_labels=actual_labels)
end

##########################
# Evidence approximation #
##########################

# no planet
H1 = (∇∇nlogL_Jones!(workspace, problem_definition, fit1_total_hyperparameters)
    + nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, remove_zeros(fit1_total_hyperparameters), 2))
nlogL_val1 = fit_nlogL + nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, fit1_total_hyperparameters, 0)
llH1 = log_laplace_approximation(H1, nlogL_val1, 0)

# planet
H2 = ∇∇nlogL_Jones_and_planet!(workspace, problem_definition, fit3_total_hyperparameters, full_ks)
n_hyper = length(remove_zeros(fit3_total_hyperparameters))
H2[diagind(H2)[1:n_hyper]] += diag(nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, remove_zeros(fit3_total_hyperparameters), 2))
H2[diagind(H2)[n_hyper+1:n_hyper+n_kep_parms]] -= diag(logprior_kepler_tot(full_ks; d_tot=2, use_hk=true))
nlogL_val2 = refit_nlogL2 + nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, fit3_total_hyperparameters, 0) - logprior_kepler(full_ks; use_hk=true)
llH2 = log_laplace_approximation(H2, nlogL_val2, 0)

println("\nevidence for Jones model: " * string(llH1))
println("evidence for Jones + planet model: " * string(llH2))

likelihoods = [-nlogL_val1, llH1, -nlogL_val2, llH2]
save_nlogLs(seed, sim_id, likelihoods, append!(copy(fit1_total_hyperparameters), fit3_total_hyperparameters), original_ks, full_ks, results_dir)
