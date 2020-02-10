include("src/all_functions.jl")

called_from_terminal = length(ARGS) > 0

###################################
# Loading data and setting kernel #
###################################

kernel_names = ["pp", "se", "m52", "qp", "m52x2", "sex2", "rq", "rm52"]
initial_hypers = [[30.], [30], [30], [30, 60, 1], [30, 60, 1], [30, 60, 1], [4, 30], [4, 30]]

# if called from terminal with an argument, use a full dataset. Otherwise, use a smaller testing set

if called_from_terminal
    kernel_choice = parse(Int, ARGS[1])
    length(ARGS) > 1 ? K_val = parse(Float64, ARGS[3]) : K_val = 0.3  # m/s
    length(ARGS) > 2 ? seed = parse(Int, ARGS[2]) : seed = 0  # m/s
    length(ARGS) > 3 ? use_long = Bool(parse(Int, ARGS[4])) : use_long = true
    length(ARGS) > 4 ? sample_size = parse(Int, ARGS[5]) : sample_size = 100
    length(ARGS) > 5 ? n_out = parse(Int, ARGS[5]) : n_out = 3
else
    kernel_choice = 3
    K_val = 0.5
    seed = 0
    use_long = true
    sample_size = 100
    n_out = 3
end
kernel_name = kernel_names[kernel_choice]

# allowing covariance matrix to be calculated in parallel
if !called_from_terminal
    try
        prep_parallel_covariance(kernel_name)
    catch
        prep_parallel_covariance(kernel_name)
    end
end

kernel_function, num_kernel_hyperparameters = include_kernel(kernel_name)

sample_size < 120 ? n_sims_needed = 1 : n_sims_needed = Int(ceil(sample_size/90))
rng = MersenneTwister(seed)
if use_long
    # sim_ids = sample(rng, 6:20, n_sims_needed; replace=false)
    sim_ids = sample(rng, 6:55, n_sims_needed; replace=false)
    fnames = ["res-1000-1years_long_id$sim_id" for sim_id in sim_ids]
else
    sim_ids = sample(rng, 1:50, n_sims_needed; replace=false)
    fnames = ["res-1000-1years_full_id$sim_id" for sim_id in sim_ids]
end

println("optimizing on $fnames using the $kernel_name")
if called_from_terminal
    problem_definition = Jones_problem_definition(kernel_function, num_kernel_hyperparameters,"jld2_files/" .* fnames; sub_sample=sample_size, on_off=14u"d", rng=rng, n_out=n_out)
else
    # problem_definition = Jones_problem_definition(kernel_function, num_kernel_hyperparameters,"../../../OneDrive/Desktop/jld2_files/" .* fnames; sub_sample=sample_size, on_off=14u"d", rng=rng, n_out=n_out)
    problem_definition = Jones_problem_definition(kernel_function, num_kernel_hyperparameters,"jld2_files/" .* fnames; sub_sample=sample_size, on_off=14u"d", rng=rng, n_out=n_out)
    # problem_definition = Jones_problem_definition(kernel_function, num_kernel_hyperparameters,"jld2_files/" .* fnames; sub_sample=sample_size, rng=rng, n_out=n_out)
end

########################################
# Adding planet and normalizing scores #
########################################

# draw over more periods?
original_ks = kep_signal(K_val * u"m/s", (8 + 1 * randn(rng))u"d", 2 * π * rand(rng), rand(rng) / 5, 2 * π * rand(rng), 0u"m/s")
add_kepler_to_Jones_problem_definition!(problem_definition, original_ks)
if use_long
    results_dir = "results/long/$n_out/$sample_size/$(kernel_name)/K_$(string(ustrip(original_ks.K)))/seed_$(seed)/"
else
    results_dir = "results/short/$n_out/$sample_size/$(kernel_name)/K_$(string(ustrip(original_ks.K)))/seed_$(seed)/"
end
try
    # rm(results_dir, recursive=true)
    mkpath(results_dir)
catch
    sleep(1)
    mkpath(results_dir)
end
normalize_problem_definition!(problem_definition)
println("Score errors:")
println(mean(problem_definition.noise[1:problem_definition.n_out:end]) * problem_definition.normals[1], " m/s")
for i in 1:problem_definition.n_out
    println(mean(problem_definition.noise[i:problem_definition.n_out:end]))
    @assert mean(problem_definition.noise[i:problem_definition.n_out:end]) < 2
end
println()

if kernel_name in ["pp", "se", "m52"]
    # normals
    parameters = gamma_mode_std_2_alpha_theta(30, 15)
    function kernel_hyper_priors(hps::Vector{<:Real}, d::Integer)
        return [log_gamma(hps[1], parameters; d=d)]
    end
    function add_kick!(hps::Vector{<:Real})
        @assert length(hps) == 1
        hps .*= centered_rand(rng, length(hps); center=1, scale=0.5)
        return hps
    end
elseif kernel_name in ["rq", "rm52"]
    # rationals
    paramsα = gamma_mode_std_2_alpha_theta(4, 10)
    paramsμ = gamma_mode_std_2_alpha_theta(30, 15)
    function kernel_hyper_priors(hps::Vector{<:Real}, d::Integer)
        return [log_gamma(hps[1], paramsα; d=d), log_gamma(hps[2], paramsμ; d=d)]
    end
    function add_kick!(hps::Vector{<:Real})
        @assert length(hps) == 2
        hps .*= centered_rand(rng, length(hps); center=1, scale=0.5)
        return hps
    end
elseif  kernel_name == "qp"
    # qp
    paramsλp = gamma_mode_std_2_alpha_theta(1, 0.4)
    σP = 15; σse = 15; ρ = .9
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
            return Symmetric(-H)
        end
    end
    function add_kick!(hps::Vector{<:Real})
        @assert length(hps) == 3
        hps[1] *= centered_rand(rng; center=1.0, scale=0.4)
        hps[2] *= centered_rand(rng; center=1.2, scale=0.4)
        hps[3] *= centered_rand(rng; center=0.8, scale=0.4)
        return hps
    end
elseif kernel_name in ["sex2", "m52x2"]
    # m52x2
    paramsλ1 = gamma_mode_std_2_alpha_theta(30, 15)
    paramsλ2 = gamma_mode_std_2_alpha_theta(60, 15)
    function kernel_hyper_priors(hps::Vector{<:Real}, d::Integer)
        return [log_gamma(hps[1], paramsλ1; d=d), log_gamma(hps[2], paramsλ2; d=d), log_gaussian(hps[3], [1, 1]; d=d)]
    end
    function add_kick!(hps::Vector{<:Real})
        @assert length(hps) == 3
        if hps[1] > hps[2]; hps[3] = 1 / hps[3] end
        hold = sort(hps[1:2])
        hps[1] = hold[1] * centered_rand(rng; center=0.8, scale=0.4)
        hps[2] = hold[2] * centered_rand(rng; center=1.2, scale=0.4)
        hps[3] *= centered_rand(rng; center=1, scale=0.4)
        return hps
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
total_hyperparameters = append!(coeff_hyperparameters, initial_hypers[kernel_choice] .* centered_rand(rng, problem_definition.n_kern_hyper; center=1, scale=0.4))

possible_labels = [
    [L"\lambda_{pp}"],
    [L"\lambda_{se}"],
    [L"\lambda_{m52}"],
    [L"\tau_p" L"\lambda_{se}" L"^1/_{\lambda_{p}}"],
    [L"\lambda_{m52_1}" L"\lambda_{m52_2}" L"\sqrt{ratio}"],
    [L"\lambda_{se_1}" L"\lambda_{se_2}" L"\sqrt{ratio}"],
    [L"\alpha" L"\mu"],
    [L"\alpha" L"\mu"]]

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
    global current_hyper[:] = non_zero_hyper
    return f_no_print(non_zero_hyper)
end

function g!(G::Vector{T}, non_zero_hyper::Vector{T}) where {T<:Real}
    if nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, non_zero_hyper, 0) == Inf
        G[:] = zeros(length(G))
    else
        G[:] = (∇nlogL_Jones!(workspace, problem_definition, non_zero_hyper)
            + nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, non_zero_hyper, 1))
    end
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
    # try
    #     global hp_string = ""
    #     for i in 1:problem_definition.n_kern_hyper
    #         global hp_string = hp_string * possible_labels[kernel_choice][i] * ": $(round(current_hyper[end-problem_definition.n_kern_hyper+i], digits=3))  "
    #     end
    #     Jones_line_plots(problem_definition, reconstruct_total_hyperparameters(problem_definition, current_hyper), "figs/gp/$(kernel_name)_training_$(x.iteration)", hyper_param_string=hp_string)  # ; plot_Σ_profile=true)
    # catch
    #     nothing
    # end
    return false
end


function do_gp_fit_gridsearch!(non_zero_hyper::Vector{<:Real}, ind::Integer)
    println("gridsearching over hp$ind")
    current_hp = non_zero_hyper[ind]
    spread = max(0.5, abs(current_hp) / 4)
    possible_hp = linspace(current_hp - spread, current_hp + spread, 11)
    new_hp = current_hp
    new_f = f_no_print(non_zero_hyper)
    println("starting at hp = ", new_hp, " -> ", new_f)
    println("searching from ", current_hp - spread, " to ", current_hp + spread)
    for hp in possible_hp
        hold = copy(non_zero_hyper)
        hold[ind] = hp
        possible_f = f_no_print(hold)
        if possible_f < new_f
            new_hp = hp
            new_f = possible_f
        end
    end
    non_zero_hyper[ind] = new_hp
    println("ending at hp   = ", new_hp, " -> ", new_f)
    return non_zero_hyper
end


function fit_GP!(initial_x::Vector{<:Real}; g_tol=1e-6, iterations=200)
    time0 = Libc.time()
    attempts = 0
    in_saddle = true
    global current_hyper = copy(initial_x)
    while attempts < 10 && in_saddle
        attempts += 1
        if attempts > 1;
            println("found saddle point. starting attempt $attempts with a perturbation")
            global current_hyper[1:end-problem_definition.n_kern_hyper] += centered_rand(rng, length(current_hyper) - problem_definition.n_kern_hyper)
            global current_hyper[end-problem_definition.n_kern_hyper+1:end] = add_kick!(current_hyper[end-problem_definition.n_kern_hyper+1:end])
        end
        if kernel_name == "qp"
            gridsearch_every = 100
            iterations = Int(ceil(iterations * 1.5))
            converged = false
            i = 0
            before_grid = zeros(length(current_hyper))
            global current_hyper = do_gp_fit_gridsearch!(current_hyper, length(current_hyper) - 2)
            try
                while i < Int(ceil(iterations / gridsearch_every)) && !converged
                    global result = optimize(f, g!, h!, current_hyper, NewtonTrustRegion(), Optim.Options(callback=optim_cb, g_tol=g_tol, iterations=gridsearch_every)) # 27s
                    before_grid[:] = current_hyper
                    global current_hyper = do_gp_fit_gridsearch!(current_hyper, length(current_hyper) - 2)
                    converged = result.g_converged && isapprox(before_grid, current_hyper)
                    i += 1
                end
            catch
                println("retrying fit")
                i = 0
                while i < Int(ceil(iterations / gridsearch_every)) && !converged
                    global result = optimize(f, g!, h!, current_hyper, NewtonTrustRegion(), Optim.Options(callback=optim_cb, g_tol=g_tol, iterations=gridsearch_every)) # 27s
                    before_grid[:] = current_hyper
                    global current_hyper = do_gp_fit_gridsearch!(current_hyper, length(current_hyper) - 2)
                    converged = result.g_converged && isapprox(before_grid, current_hyper)
                    i += 1
                end
            end
        else
            try
                global result = optimize(f, g!, h!, current_hyper, NewtonTrustRegion(), Optim.Options(callback=optim_cb, g_tol=g_tol, iterations=iterations)) # 27s
            catch
                println("retrying fit")
                global result = optimize(f, g!, h!, current_hyper, NewtonTrustRegion(), Optim.Options(callback=optim_cb, g_tol=g_tol, iterations=iterations))
            end
        end
        current_det = det(h!(zeros(length(initial_x), length(initial_x)), current_hyper))
        println(current_det)
        in_saddle = current_det <= 0
    end
    return Libc.time() - time0
end

time1 = fit_GP!(initial_x)
println(result)

fit1_total_hyperparameters = reconstruct_total_hyperparameters(problem_definition, result.minimizer)

############
# Post fit #
############

println("starting hyperparameters")
println(total_hyperparameters)
initial_nlogL = nlogL_Jones(problem_definition, total_hyperparameters)
initial_uE = -initial_nlogL - nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, total_hyperparameters, 0)
println(initial_uE, "\n")

println("ending hyperparameters")
println(fit1_total_hyperparameters)
fit_nlogL1 = nlogL_Jones!(workspace, problem_definition, fit1_total_hyperparameters)
uE1 = -fit_nlogL1 - nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, fit1_total_hyperparameters, 0)
println(uE1, "\n")

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
@everywhere function kep_likelihood_distributed(P::Unitful.Time)
    ks = fit_kepler(problem_definition, Σ_obs, kep_signal_epicyclic(P=P))
    # ks = kep_signal(ks.K, ks.P, ks.M0, minimum([ks.e, 0.3]), ks.ω, ks.γ)
    return -nlogL_Jones(
        problem_definition,
        fit1_total_hyperparameters;
        Σ_obs=Σ_obs,
        y_obs=remove_kepler(problem_definition, ks))
end


fit1_total_hyperparameters_nz = remove_zeros(fit1_total_hyperparameters)
nlogprior_kernel = nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, fit1_total_hyperparameters_nz, 0)
sendto(workers(), fit1_total_hyperparameters_nz=fit1_total_hyperparameters_nz, nlogprior_kernel=nlogprior_kernel)
@everywhere function kep_unnormalized_evidence_distributed(P::Unitful.Time)
    ks = fit_kepler(problem_definition, Σ_obs, kep_signal_epicyclic(P=P))
    # ks = kep_signal(ks.K, ks.P, ks.M0, minimum([ks.e, 0.3]), ks.ω, ks.γ)
    return logprior_kepler(ks; use_hk=true) - nlogprior_kernel - nlogL_Jones(
        problem_definition,
        fit1_total_hyperparameters;
        Σ_obs=Σ_obs,
        y_obs=remove_kepler(problem_definition, ks))
end

# parallelize with DistributedArrays
@everywhere using DistributedArrays
period_grid_dist = distribute(period_grid)
@time likelihoods = collect(map(kep_likelihood_distributed, period_grid_dist))
@time evidences = collect(map(kep_unnormalized_evidence_distributed, period_grid_dist))

# @everywhere function kep_full_unnormalized_evidence_distributed(P::Unitful.Time)  # 400x slower than kep_unnormalized_evidence_distributed
#     ks = fit_kepler(problem_definition, Σ_obs, kep_signal_epicyclic(P=P))
#     ks = fit_kepler(problem_definition, Σ_obs, kep_signal_wright(maximum([0.1u"m/s", ks.K]), ks.P, ks.M0, minimum([ks.e, 0.3]), ks.ω, ks.γ); print_stuff=false)
#     return logprior_kepler(ks; use_hk=true) - nlogprior_kernel - nlogL_Jones(
#         problem_definition,
#         fit1_total_hyperparameters;
#         Σ_obs=Σ_obs,
#         y_obs=remove_kepler(problem_definition, ks))
# end

# period_grid_dist_small = distribute(period_grid[1:10])
# @time evidences_full = collect(map(kep_full_unnormalized_evidence_distributed, period_grid_dist_small))
# @time evidences_full = kep_full_unnormalized_evidence_distributed.(period_grid[1:10])

best_period_grid = period_grid[find_modes(likelihoods; amount=10)]

# find first period that uses a bound eccentricity
estimated_eccentricities = [fit_kepler(
    problem_definition, Σ_obs, kep_signal_epicyclic(;P=period)
    ).e for period in best_period_grid]

try
    global best_period = best_period_grid[findfirst(y -> isless(y, 0.33), estimated_eccentricities)]
catch
    try
        global best_period = best_period_grid[findfirst(y -> isless(y, 0.5), estimated_eccentricities)]
    catch
        global best_period = best_period_grid[argmin(estimated_eccentricities)]
        @warn "no low-eccentricity period found"
    end
end

# kep_signal_wright(0.1u"m/s", test_ks.P + 0.1u"d", test_ks.M0, 0.3, test_ks.ω, test_ks.γ)
#
#
# # test_ks = fit_kepler(problem_definition, Σ_obs, kep_signal_epicyclic(P=best_period))
# @time test_ks1 = fit_kepler(problem_definition, Σ_obs, kep_signal_wright(0.1u"m/s", test_ks.P + 0.1u"d", test_ks.M0, 0.1, test_ks.ω, test_ks.γ))
# @time test_ks2 = fit_kepler(problem_definition, Σ_obs, kep_signal(maximum([0.1u"m/s", test_ks.K]), test_ks.P, test_ks.M0, minimum([test_ks.e, 0.3]), test_ks.ω, test_ks.γ); print_stuff=false)
#
#
#
# test_ks
# Jones_line_plots(problem_definition, fit1_total_hyperparameters, results_dir * "fart"; fit_ks=test_ks)
# test_ks1
# Jones_line_plots(problem_definition, fit1_total_hyperparameters, results_dir * "fart1"; fit_ks=test_ks1)
# test_ks2
# Jones_line_plots(problem_definition, fit1_total_hyperparameters, results_dir * "fart2"; fit_ks=test_ks2)


println("original period: $(ustrip(original_ks.P)) days")
println("found period:    $(ustrip(best_period)) days")

begin
    ax = init_plot()
    fig = plot(ustrip.(period_grid), likelihoods, color="black")
    xscale("log")
    ticklabel_format(style="sci", axis="y", scilimits=(0,0))
    xlabel("Periods (days)")
    ylabel("GP log likelihoods")
    ylim(-fit_nlogL1 - 3, maximum(likelihoods) + 3)
    axhline(y=-fit_nlogL1, color="k")
    axvline(x=convert_and_strip_units(u"d", best_period), color="red", linestyle="--")
    if original_ks.K != 0u"m/s"
        axvline(x=convert_and_strip_units(u"d", original_ks.P), color="blue", linestyle="--")
        title_string = @sprintf "%.1f day, %.2f m/s" convert_and_strip_units(u"d", original_ks.P) convert_and_strip_units(u"m/s",original_ks.K)
        title(title_string, fontsize=30)
    end
    save_PyPlot_fig(results_dir * "periodogram.png")

    ax = init_plot()
    fig = plot(ustrip.(period_grid), evidences, color="black")
    xscale("log")
    ticklabel_format(style="sci", axis="y", scilimits=(0,0))
    xlabel("Periods (days)")
    ylabel("GP log unnormalized evidences")
    axhline(y=uE1, color="k")
    axvline(x=convert_and_strip_units(u"d", best_period), color="red", linestyle="--")
    if original_ks.K != 0u"m/s"
        axvline(x=convert_and_strip_units(u"d", original_ks.P), color="blue", linestyle="--")
        title_string = @sprintf "%.1f day, %.2f m/s" convert_and_strip_units(u"d", original_ks.P) convert_and_strip_units(u"m/s",original_ks.K)
        title(title_string, fontsize=30)
    end
    save_PyPlot_fig(results_dir * "periodogram_ev.png")
end

if original_ks.K != 0u"m/s"
    ax = init_plot()
    inds = (original_ks.P / 1.5).<period_grid.<(2.5 * original_ks.P)
    fig = plot(ustrip.(period_grid[inds]), likelihoods[inds], color="black")
    xscale("log")
    ticklabel_format(style="sci", axis="y", scilimits=(0,0))
    xlabel("Periods (days)")
    ylabel("GP log likelihoods")
    axhline(y=-fit_nlogL1, color="k")
    axvline(x=convert_and_strip_units(u"d", best_period), color="red", linestyle="--")
    axvline(x=convert_and_strip_units(u"d", original_ks.P), color="blue", linestyle="--")
    title_string = @sprintf "%.1f day, %.2f m/s" convert_and_strip_units(u"d", original_ks.P) convert_and_strip_units(u"m/s",original_ks.K)
    title(title_string, fontsize=30)
    save_PyPlot_fig(results_dir * "periodogram_zoom.png")

    ax = init_plot()
    inds = (original_ks.P / 1.5).<period_grid.<(2.5 * original_ks.P)
    fig = plot(ustrip.(period_grid[inds]), evidences[inds], color="black")
    xscale("log")
    ticklabel_format(style="sci", axis="y", scilimits=(0,0))
    xlabel("Periods (days)")
    ylabel("GP log unnormalized evidences")
    axhline(y=uE1, color="k")
    axvline(x=convert_and_strip_units(u"d", best_period), color="red", linestyle="--")
    axvline(x=convert_and_strip_units(u"d", original_ks.P), color="blue", linestyle="--")
    title_string = @sprintf "%.1f day, %.2f m/s" convert_and_strip_units(u"d", original_ks.P) convert_and_strip_units(u"m/s",original_ks.K)
    title(title_string, fontsize=30)
    save_PyPlot_fig(results_dir * "periodogram_ev_zoom.png")
end

################################################################
# Refitting GP with planet signals at found periods subtracted #
################################################################

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
    global current_hyper[:] = non_zero_hyper
    return f_no_print(non_zero_hyper)
end

function g!(G::Vector{T}, non_zero_hyper::Vector{T}) where {T<:Real}
    if nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, non_zero_hyper, 0) == Inf
        G[:] = zeros(length(G))
    else
        G[:] = (∇nlogL_Jones!(workspace, problem_definition, non_zero_hyper; y_obs=fit_and_remove_kepler_epi_short())
            + nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, non_zero_hyper, 1))
    end
end

function h!(H::Matrix{T}, non_zero_hyper::Vector{T}) where {T<:Real}
    H[:, :] = (∇∇nlogL_Jones!(workspace, problem_definition, non_zero_hyper; y_obs=fit_and_remove_kepler_epi_short())
     + nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, non_zero_hyper, 2))
end

time2 = fit_GP!(remove_zeros(fit1_total_hyperparameters))
println(result)

fit2_total_hyperparameters = reconstruct_total_hyperparameters(problem_definition, result.minimizer)

######################
# Post epicyclic fit #
######################

Σ_obs_final = Σ_observations(problem_definition, fit2_total_hyperparameters)

epi_ks = fit_kepler(problem_definition, Σ_obs_final, kep_signal_epicyclic(;P=best_period))

global hp_string = ""
for i in 1:problem_definition.n_kern_hyper
    global hp_string = hp_string * possible_labels[kernel_choice][i] * ": $(round(fit2_total_hyperparameters[end-problem_definition.n_kern_hyper+i], digits=3))  "
end
Jones_line_plots(problem_definition, fit2_total_hyperparameters, results_dir * "fit_epi"; fit_ks=epi_ks, hyper_param_string=hp_string)

###################################################################
# Refitting GP with full planet signal at found period subtracted #
###################################################################

function f_no_print(non_zero_hyper::Vector{T}) where {T<:Real}
    nprior = nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, non_zero_hyper, 0)
    # println(prior)
    # println(kep_parms_str(current_ks))
    if nprior == Inf
        return nprior
    else
        return nlogL_Jones!(workspace, problem_definition, non_zero_hyper; y_obs=current_y) + nprior
    end
end

function f(non_zero_hyper::Vector{T}) where {T<:Real}
    println(non_zero_hyper)
    global current_hyper[:] = non_zero_hyper
    return f_no_print(non_zero_hyper)
end

function g!(G::Vector{T}, non_zero_hyper::Vector{T}) where {T<:Real}
    if nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, non_zero_hyper, 0) == Inf
        G[:] = zeros(length(G))
    else
        G[:] = (∇nlogL_Jones!(workspace, problem_definition, non_zero_hyper; y_obs=current_y)
            + nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, non_zero_hyper, 1))
    end
end

function h!(H::Matrix{T}, non_zero_hyper::Vector{T}) where {T<:Real}
    H[:, :] = (∇∇nlogL_Jones!(workspace, problem_definition, non_zero_hyper; y_obs=current_y)
     + nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, non_zero_hyper, 2))
end

begin
    time0 = Libc.time()
    global current_hyper = remove_zeros(fit2_total_hyperparameters)

    global current_ks = kep_signal(epi_ks.K, epi_ks.P, epi_ks.M0, epi_ks.e, epi_ks.ω, epi_ks.γ)
    println("before full fit: ", kep_parms_str(current_ks))
    # det(∇∇nlogL_kep(problem_definition.y_obs, problem_definition.time, workspace.Σ_obs, current_ks; data_unit=problem_definition.rv_unit*problem_definition.normals[1]))

    results = [Inf, Inf]
    result_change = Inf
    global num_iter = 0
    while result_change > 1e-6 && num_iter < 100
        global current_ks = fit_kepler(problem_definition, workspace.Σ_obs, current_ks)
        println(kep_parms_str(current_ks))
        global current_y = remove_kepler(problem_definition, current_ks)
        fit_GP!(current_hyper)
        results[:] = [results[2], copy(result.minimum)]
        thing = results[1] - results[2]
        if thing < 0; @warn "result increased occured on iteration $num_iter" end
        global result_change = abs(thing)
        global num_iter += 1
        println("change on joint fit $num_iter: ", result_change)
    end
    time3 = Libc.time() - time0
end

println(result)

fit3_total_hyperparameters = reconstruct_total_hyperparameters(problem_definition, result.minimizer)

Σ_obs_final2 = Σ_observations(problem_definition, fit3_total_hyperparameters)

full_ks = kep_signal(current_ks.K, current_ks.P, current_ks.M0, current_ks.e, current_ks.ω, current_ks.γ)

###################
# Post planet fit #
###################

println("fit hyperparameters")
println(fit1_total_hyperparameters)
println(uE1, "\n")

println("kepler hyperparameters")
println(fit3_total_hyperparameters)
fit_nlogL2 = nlogL_Jones!(workspace, problem_definition, fit3_total_hyperparameters; y_obs=current_y)
uE2 = -fit_nlogL2 - nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, fit3_total_hyperparameters, 0) + logprior_kepler(full_ks; use_hk=true)
println(uE2, "\n")

global hp_string = ""
for i in 1:problem_definition.n_kern_hyper
    global hp_string = hp_string * possible_labels[kernel_choice][i] * ": $(round(fit3_total_hyperparameters[end-problem_definition.n_kern_hyper+i], digits=3))  "
end
Jones_line_plots(problem_definition, fit3_total_hyperparameters, results_dir * "fit_full"; fit_ks=full_ks, hyper_param_string=hp_string)

println("best fit keplerian")
println(kep_parms_str(full_ks))
println("originial injected keplerian")
println(kep_parms_str(original_ks))

##########################
# refitting after planet #
##########################

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
    global current_hyper[:] = non_zero_hyper
    return f_no_print(non_zero_hyper)
end

function g!(G::Vector{T}, non_zero_hyper::Vector{T}) where {T<:Real}
    if nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, non_zero_hyper, 0) == Inf
        G[:] = zeros(length(G))
    else
        G[:] = (∇nlogL_Jones!(workspace, problem_definition, non_zero_hyper)
            + nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, non_zero_hyper, 1))
    end
end

function h!(H::Matrix{T}, non_zero_hyper::Vector{T}) where {T<:Real}
    # H[:, :] = ∇∇nlogL_Jones!(workspace, problem_definition, non_zero_hyper)
    H[:, :] = (∇∇nlogL_Jones!(workspace, problem_definition, non_zero_hyper)
     + nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, non_zero_hyper, 2))
end

time1 += fit_GP!(remove_zeros(fit3_total_hyperparameters))
println(result)

fit1_total_hyperparameters_temp = reconstruct_total_hyperparameters(problem_definition, result.minimizer)

############
# Post fit #
############

println("first fit hyperparameters")
println(fit1_total_hyperparameters)
println(uE1, "\n")

println("fit after planet hyperparameters")
println(fit1_total_hyperparameters_temp)
fit_nlogL1_temp = nlogL_Jones!(workspace, problem_definition, fit1_total_hyperparameters_temp)
uE1_temp = -fit_nlogL1_temp - nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, fit1_total_hyperparameters_temp, 0)
println(uE1_temp, "\n")

if uE1_temp > uE1
    println("new fit is better, switching hps")
    fit1_total_hyperparameters = fit1_total_hyperparameters_temp
    fit_nlogL1 = fit_nlogL1_temp
    uE1 = uE1_temp
end

global hp_string = ""
for i in 1:problem_definition.n_kern_hyper
    global hp_string = hp_string * possible_labels[kernel_choice][i] * ": $(round(fit1_total_hyperparameters[end-problem_definition.n_kern_hyper+i], digits=3))  "
end

Jones_line_plots(problem_definition, fit1_total_hyperparameters, results_dir * "fit"; hyper_param_string=hp_string)  # , plot_Σ=true, plot_Σ_profile=true)


################
# Corner plots #
################

# if called_from_terminal
if true
    actual_labels = append!([L"a_{11}", L"a_{21}", L"a_{12}", L"a_{32}", L"a_{23}"], possible_labels[kernel_choice])

    corner_plot(f_no_print, remove_zeros(fit1_total_hyperparameters), results_dir * "corner.png"; input_labels=actual_labels)
    # corner_plot(f_no_print, remove_zeros(fit1_total_hyperparameters), results_dir * "corner.png"; input_labels=actual_labels, steps=3)

    y_obs = remove_kepler(problem_definition, full_ks)
    function f2(non_zero_hyper::Vector{T}) where {T<:Real}
        nprior = nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, non_zero_hyper, 0)
        if nprior == Inf
            return nprior
        else
            return nlogL_Jones!(workspace, problem_definition, non_zero_hyper; y_obs=y_obs) + nprior
        end
    end
    corner_plot(f2, remove_zeros(fit3_total_hyperparameters), results_dir * "corner_planet.png"; input_labels=actual_labels)
    # corner_plot(f2, remove_zeros(fit3_total_hyperparameters), results_dir * "corner_planet.png"; input_labels=actual_labels, steps=3)
end


####################################################
# fitting with true planet removed (for reference) #
####################################################

y_obs_og = remove_kepler(problem_definition, original_ks)

function f_no_print(non_zero_hyper::Vector{T}) where {T<:Real}
    nprior = nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, non_zero_hyper, 0)
    if nprior == Inf
        return nprior
    else
        return nlogL_Jones!(workspace, problem_definition, non_zero_hyper; y_obs=y_obs_og) + nprior
    end
end

function f(non_zero_hyper::Vector{T}) where {T<:Real}
    println(non_zero_hyper)
    global current_hyper[:] = non_zero_hyper
    return f_no_print(non_zero_hyper)
end

function g!(G::Vector{T}, non_zero_hyper::Vector{T}) where {T<:Real}
    if nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, non_zero_hyper, 0) == Inf
        G[:] = zeros(length(G))
    else
        G[:] = (∇nlogL_Jones!(workspace, problem_definition, non_zero_hyper; y_obs=y_obs_og)
            + nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, non_zero_hyper, 1))
    end
end

function h!(H::Matrix{T}, non_zero_hyper::Vector{T}) where {T<:Real}
    H[:, :] = (∇∇nlogL_Jones!(workspace, problem_definition, non_zero_hyper; y_obs=y_obs_og)
     + nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, non_zero_hyper, 2))
end

original_ks.K < 0.5u"m/s" ? time4 = fit_GP!(remove_zeros(fit1_total_hyperparameters)) : time4 = fit_GP!(remove_zeros(fit3_total_hyperparameters))

println(result)

fit4_total_hyperparameters = reconstruct_total_hyperparameters(problem_definition, result.minimizer)

############
# Post fit #
############

println("with no planet hyperparameters")
println(fit4_total_hyperparameters)
fit_nlogL3 = nlogL_Jones!(workspace, problem_definition, fit4_total_hyperparameters; y_obs=remove_kepler(problem_definition, original_ks))
uE3 = -fit_nlogL3 - nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, fit4_total_hyperparameters, 0) + logprior_kepler(original_ks; use_hk=true)
println(fit_nlogL3, "\n")


global hp_string = ""
for i in 1:problem_definition.n_kern_hyper
    global hp_string = hp_string * possible_labels[kernel_choice][i] * ": $(round(fit4_total_hyperparameters[end-problem_definition.n_kern_hyper+i], digits=3))  "
end
Jones_line_plots(problem_definition, fit4_total_hyperparameters, results_dir * "truth"; fit_ks=original_ks, hyper_param_string=hp_string)  # , plot_Σ=true, plot_Σ_profile=true)

##########################
# Evidence approximation #
##########################

# no planet
H1 = (∇∇nlogL_Jones!(workspace, problem_definition, fit1_total_hyperparameters)
    + nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, remove_zeros(fit1_total_hyperparameters), 2))
try
    global E1 = log_laplace_approximation(H1, -uE1, 0)
catch
    println("Laplace approximation failed for initial GP fit")
    println("det(H1): $(det(H1)) (should've been positive)")
    global E1 = 0
end


# planet
H2 = Matrix(∇∇nlogL_Jones_and_planet!(workspace, problem_definition, fit3_total_hyperparameters, full_ks))
n_hyper = length(remove_zeros(fit3_total_hyperparameters))
H2[1:n_hyper, 1:n_hyper] += nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, remove_zeros(fit3_total_hyperparameters), 2)
H2[n_hyper+1:n_hyper+n_kep_parms, n_hyper+1:n_hyper+n_kep_parms] -= logprior_kepler_tot(full_ks; d_tot=2, use_hk=true)
try
    global E2 = log_laplace_approximation(Symmetric(H2), -uE2, 0)
catch
    println("Laplace approximation failed for planet fit")
    println("det(H2): $(det(H2)) (should've been positive)")
    global E2 = 0
end


# true planet
H3 = Matrix(∇∇nlogL_Jones_and_planet!(workspace, problem_definition, fit4_total_hyperparameters, original_ks))
H3[1:n_hyper, 1:n_hyper] += nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, remove_zeros(fit4_total_hyperparameters), 2)
H3[n_hyper+1:n_hyper+n_kep_parms, n_hyper+1:n_hyper+n_kep_parms] -= logprior_kepler_tot(original_ks; d_tot=2, use_hk=true)
try
    # often the following often wont work because the keplerian paramters haven't been optimized on the data
    global E3 = log_laplace_approximation(Symmetric(H3), -uE3, 0)
    # equivalent to fitting original dataset with no planet
    # global llH3 = log_laplace_approximation(Symmetric(H3[1:6,1:6]), fit_nlogL3 + nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, fit4_total_hyperparameters, 0), 0)
catch
    println("Laplace approximation failed for original planet fit (expected most of the time)")
    println("det(H3): $(det(H3)) (could've been positive)")
    global E3 = 0
end

println("\nlog likelihood for Jones model: " * string(-fit_nlogL1))
println("log likelihood for Jones + planet model: " * string(-fit_nlogL2))
println("log likelihood for Jones + true planet model: " * string(-fit_nlogL3))

println("\nunnormalized evidence for Jones model: " * string(uE1))
println("unnormalized evidence for Jones + planet model: " * string(uE2))
println("unnormalized evidence for Jones + true planet model: " * string(uE3))

println("\nevidence for Jones model: " * string(E1))
println("evidence for Jones + planet model: " * string(E2))
println("evidence for Jones + true planet model: " * string(E3))

saved_likelihoods = [-fit_nlogL1, uE1, E1, -fit_nlogL2, uE2, E2, -fit_nlogL3, uE3, E3]
save_nlogLs(seed, [time1, time2 + time3, time4] ./ 3600, saved_likelihoods, append!(copy(fit1_total_hyperparameters), fit3_total_hyperparameters), original_ks, full_ks, results_dir)
# sum([time1, time2 + time3, time4] ./ 3600)
