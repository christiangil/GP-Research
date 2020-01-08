include("../src/all_functions.jl")

called_from_terminal = length(ARGS) > 0

###################################
# Loading data and setting kernel #
###################################

kernel_names = ["pp", "se", "m52", "qp", "m52x2", "rq", "rm52"]
initial_hypers = [[30.], [30], [30], [30, 60, 1], [30, 60, 1], [4, 12], [4, 12]]

# if called from terminal with an argument, use a full dataset. Otherwise, use a smaller testing set
sample_size = 100
if called_from_terminal
    kernel_choice = parse(Int, ARGS[1])
    kernel_name = kernel_names[kernel_choice]
    seed = parse(Int, ARGS[2])
else
    kernel_choice = 1
    kernel_name = kernel_names[kernel_choice]
    seed = 1
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
    problem_definition = Jones_problem_definition(kernel_function, num_kernel_hyperparameters,"../../../OneDrive/Desktop/jld2_files/" * fname; sub_sample=sample_size, on_off=14u"d", rng=rng)
    # problem_definition = Jones_problem_definition(kernel_function, num_kernel_hyperparameters,"jld2_files/" * fname; sub_sample=sample_size, on_off=14u"d", rng=rng)
end

########################################
# Adding planet and normalizing scores #
########################################

length(ARGS) > 1 ? K_val = parse(Float64, ARGS[3]) : K_val = 0.0  # m/s
# draw over more periods?
original_ks = kep_signal(K_val * u"m/s", (8 + 1 * randn(rng))u"d", 2 * π * rand(rng), rand(rng) / 5, 2 * π * rand(rng), 0u"m/s")
add_kepler_to_Jones_problem_definition!(problem_definition, original_ks)

results_dir = "results/$(kernel_name)/K_$(string(ustrip(original_ks.K)))/seed_$(seed)/"

try
    rm(results_dir, recursive=true)
    mkpath(results_dir)
catch
    mkpath(results_dir)
end
normalize_problem_definition!(problem_definition)
println("Score errors:")
println(mean(problem_definition.noise[1:sample_size]) * problem_definition.normals[1], " m/s")
for i in 1:problem_definition.n_out
    println(mean(problem_definition.noise[(i-1)*sample_size+1:i*sample_size]))
    @assert mean(problem_definition.noise[(i-1)*sample_size+1:i*sample_size]) < 2
end
println()


if kernel_name in ["pp", "se", "m52"]
    # normals
    parameters = gamma_mode_std_2_alpha_theta(30, 15)
    function kernel_hyper_priors(hps::Vector{<:Real}, d::Integer)
        return [log_gamma(hps[1], parameters; d=d)]
    end
elseif kernel_name in ["rq", "rm52"]
    # rationals
    paramsα = gamma_mode_std_2_alpha_theta(4, 10)
    paramsμ = gamma_mode_std_2_alpha_theta(30, 15)
    function kernel_hyper_priors(hps::Vector{<:Real}, d::Integer)
        return [log_gamma(hps[1], paramsα; d=d), log_gamma(hps[2], paramsμ; d=d)]
    end
elseif  kernel_name == "qp"
    # qp
    paramsλp = gamma_mode_std_2_alpha_theta(1, 1)
    σP = 15; σse = 30; ρ = .9
    Σ_qp_prior = bvnormal_covariance(σP, σse, ρ)
    μ_qp_prior = [30, 60.]
    function nlogprior_kernel_hyperparameters(n_kern_hyper::Integer, total_hyperparameters::Vector{<:Real}, d::Integer)
        hps = total_hyperparameters[(end - (n_kern_hyper - 1)):end]
        if d == 0
            return -(log_bvnormal(hps[1:2], Σ_qp_prior; μ=μ_qp_prior, lows=[0,0.]) + log_gamma(hps[3], paramsλp))
        elseif d == 1
            return append!(zeros(length(total_hyperparameters) - n_kern_hyper), -[log_bvnormal(hps[1:2], Σ_qp_prior; μ=μ_qp_prior, lows=[0,0.], d=[1,0]), log_bvnormal(hps[1:2], Σ_qp_prior; μ=μ_qp_prior, lows=[0,0.], d=[0,1]), log_gamma(hps[3], paramsλp; d=d)])
        elseif d == 2
            H = zeros(length(total_hyperparameters), length(total_hyperparameters))
            H[end-2,end-2] = log_bvnormal(hps[1:2], Σ_qp_prior; μ=μ_qp_prior, lows=[0,0.], d=[2,0])
            H[end-1,end-1] = log_bvnormal(hps[1:2], Σ_qp_prior; μ=μ_qp_prior, lows=[0,0.], d=[0,2])
            H[end,end] = log_gamma(hps[3], paramsλp; d=d)
            H[end-2,end-1] = log_bvnormal(hps[1:2], Σ_qp_prior; μ=μ_qp_prior, lows=[0,0.], d=[1,1])
            return Symmetric(H)
        end
    end
elseif kernel_name == "m52x2"
    # m52x2
    paramsλ1 = gamma_mode_std_2_alpha_theta(30, 15)
    paramsλ2 = gamma_mode_std_2_alpha_theta(60, 30)
    function kernel_hyper_priors(hps::Vector{<:Real}, d::Integer)
        return [log_gamma(hps[1], paramsλ1; d=d), log_gamma(hps[2], paramsλ2; d=d), log_gaussian(hps[3], [1, 1]; d=d)]
    end
end

if !(kernel_name in ["qp"])
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
end

#####################################
# Initial hyperparameters and plots #
#####################################

# kernel hyper parameters
time_span = maximum(problem_definition.x_obs) - minimum(problem_definition.x_obs)
coeff_hyperparameters = collect(Iterators.flatten(problem_definition.a0))
total_hyperparameters = append!(coeff_hyperparameters, initial_hypers[kernel_choice] .* (0.8 .+ (0.4 .* rand(rng, problem_definition.n_kern_hyper))))

# how finely to sample the domain (for plotting)
amount_of_samp_points = convert(Int64, max(500, round(2 * sqrt(2) * length(problem_definition.x_obs))))

# Jones_line_plots(amount_of_samp_points, problem_definition, total_hyperparameters, "figs/gp/$kernel_name/seed$(seed)_initial_gp"; find_post=false)  # , plot_Σ=true, plot_Σ_profile=true)

possible_labels = [
    [L"\lambda_{pp}"],
    [L"\lambda_{se}"],
    [L"\lambda_{m52}"],
    [L"\tau_p" L"\lambda_{se}" L"^1/_{\lambda_{p}}"],
    [L"\lambda_{1}" L"\lambda_{2}" L"\sqrt{ratio}"],
    [L"\alpha" L"\mu"],
    [L"\alpha" L"\mu"]]

global hp_string = ""
for i in 1:problem_definition.n_kern_hyper
    global hp_string = hp_string * possible_labels[kernel_choice][i] * ": $(round(total_hyperparameters[end-problem_definition.n_kern_hyper+i], digits=3))  "
end

Jones_line_plots(amount_of_samp_points, problem_definition, total_hyperparameters, results_dir * "init"; hyper_param_string=hp_string)

#######################
# Jones model fitting #
#######################

workspace = nlogL_matrix_workspace(problem_definition, total_hyperparameters)

try
    using Optim
catch
    using Optim
end

# storing initial hyperparameters
initial_x = remove_zeros(total_hyperparameters)

function f_no_print(non_zero_hyper::Vector{T}) where {T<:Real}
    nprior = nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, non_zero_hyper, 0)
    if nprior == Inf
        return nprior
    else
        return nlogL_Jones!(workspace, problem_definition, non_zero_hyper) + nprior
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
        println("Iteration:             ", x.iteration)
        println("Time so far:           ", x.metadata["time"], " s")
        println("Unnormalized evidence: ", x.value)
        println("Gradient Norm:         ", x.g_norm)
        println()
        # update_optimize_Jones_model_jld2!(kernel_name, non_zero_hyper_param)
    end
    # Jones_line_plots(amount_of_samp_points, problem_definition, reconstruct_total_hyperparameters(problem_definition, data(non_zero_hyper_param)), "figs/gp/$kernel_name/training/training_$(x.iteration)_gp")  # ; plot_Σ_profile=true)
    return false
end


function fit_GP!(initial_x::Vector{<:Real}; g_tol=1e-6, iterations=200)
    attempts = 0
    in_saddle = true
    global current_hyper = copy(initial_x)
    while attempts < 10 && in_saddle
        attempts += 1
        if attempts > 1;
            println("found saddle point. starting attempt $attempt with a perturbation")
            global current_hyper += 3e-2 * (rand(rng, length(current_hyper)) .- 0.5)
        end
        # println("fitting with Hessian")
        try
            global result = optimize(f, g!, h!, current_hyper, NewtonTrustRegion(), Optim.Options(callback=optim_cb, g_tol=g_tol, iterations=iterations)) # 27s
        catch
            println("retrying fit")
            global result = optimize(f, g!, h!, current_hyper, NewtonTrustRegion(), Optim.Options(callback=optim_cb, g_tol=g_tol, iterations=iterations))
        end
        # println("fitting without Hessian")
        # try
        #     global result = optimize(f, g!, current_hyper, LBFGS(), Optim.Options(callback=optim_cb, g_tol=g_tol, iterations=iterations)) # 27s
        # catch
        #     println("retrying fit")
        #     global result = optimize(f, g!, current_hyper, LBFGS(), Optim.Options(callback=optim_cb, g_tol=g_tol, iterations=iterations))
        # end
        in_saddle = det(h!(zeros(length(initial_x), length(initial_x)), current_hyper)) <= 0
    end
end

fit_GP!(initial_x)

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
fit_nlogL1 = nlogL_Jones!(workspace, problem_definition, fit1_total_hyperparameters)
println(fit_nlogL1, "\n")

global hp_string = ""
for i in 1:problem_definition.n_kern_hyper
    global hp_string = hp_string * possible_labels[kernel_choice][i] * ": $(round(fit1_total_hyperparameters[end-problem_definition.n_kern_hyper+i], digits=3))  "
end

Jones_line_plots(amount_of_samp_points, problem_definition, fit1_total_hyperparameters, results_dir * "fit"; hyper_param_string=hp_string)  # , plot_Σ=true, plot_Σ_profile=true)

################
# Corner plots #
################

if called_from_terminal
    actual_labels = append!([L"a_{11}", L"a_{21}", L"a_{12}", L"a_{32}", L"a_{23}"], possible_labels[kernel_choice])
    corner_plot(f_no_print, remove_zeros(fit1_total_hyperparameters), results_dir * "corner.png"; input_labels=actual_labels)
end


###########################################################################
# Evaluating GP likelihoods after taking out planets at different periods #
###########################################################################

amount_of_periods = 2^13

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

best_period_grid = period_grid[find_modes(likelihoods; amount=10)]

################################################################
# Refitting GP with planet signals at found periods subtracted #
################################################################

# find first period that uses a bound eccentricity
estimated_eccentricities = [fit_kepler(
    problem_definition, Σ_obs, kep_signal_epicyclic(;P=period)
    ).e for period in best_period_grid]

try
    global best_period = best_period_grid[findfirst(y -> isless(y, 0.3), estimated_eccentricities)]
catch
    try
        global best_period = best_period_grid[findfirst(y -> isless(y, 0.5), estimated_eccentricities)]
    catch
        global best_period = best_period_grid[argmin(estimated_eccentricities)]
        @warn "no low-eccentricity period found"
    end
end

println("original period: $(ustrip(original_ks.P)) days")
println("found period:    $(ustrip(best_period)) days")


fit_and_remove_kepler_epi_short() = fit_and_remove_kepler(
    problem_definition,
    workspace.Σ_obs,
    kep_signal_epicyclic(P=best_period))

function f_no_print(non_zero_hyper::Vector{T}) where {T<:Real}
    nprior = nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, non_zero_hyper, 0)
    if nprior == Inf
        return nprior
    else
        return nlogL_Jones!(workspace, problem_definition, non_zero_hyper; y_obs=fit_and_remove_kepler_epi_short()) + nprior
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

fit_GP!(remove_zeros(fit1_total_hyperparameters))

println(result)

fit2_total_hyperparameters = reconstruct_total_hyperparameters(problem_definition, result.minimizer)

######################
# Post epicyclic fit #
######################

println("fit hyperparameters")
println(fit1_total_hyperparameters)
println(fit_nlogL1, "\n")

println("epicyclic hyperparameters")
println(fit2_total_hyperparameters)
refit_nlogL1 = nlogL_Jones!(workspace, problem_definition, fit2_total_hyperparameters; y_obs=fit_and_remove_kepler_epi_short())
println(refit_nlogL1, "\n")

Σ_obs_final = Σ_observations(problem_definition, fit2_total_hyperparameters)

epi_ks = fit_kepler(problem_definition, Σ_obs_final, kep_signal_epicyclic(;P=best_period))

dif = 1e-7


ks = kep_signal(epi_ks.K, epi_ks.P, epi_ks.M0, epi_ks.h, epi_ks.k, epi_ks.γ; use_hk=true)

test_∇nlogL_kep(problem_definition.y_obs, problem_definition.time, workspace.Σ_obs, ks; data_unit=problem_definition.rv_unit*problem_definition.normals[1])

test_∇∇nlogL_kep(problem_definition.y_obs, problem_definition.time, workspace.Σ_obs, ks; data_unit=problem_definition.rv_unit*problem_definition.normals[1])

using BenchmarkTools

@btime est_∇∇nlogL_kep(problem_definition.y_obs, problem_definition.time, workspace.Σ_obs, ks; data_unit=problem_definition.rv_unit*problem_definition.normals[1])

@btime ∇∇nlogL_kep(problem_definition.y_obs, problem_definition.time, workspace.Σ_obs, ks; data_unit=problem_definition.rv_unit*problem_definition.normals[1])
