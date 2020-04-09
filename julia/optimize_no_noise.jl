include("src/all_functions.jl")

called_from_terminal = length(ARGS) > 0

###################################
# Loading data and setting kernel #
###################################

# if called from terminal with an argument, use a full dataset. Otherwise, use a smaller testing set
# vertical line - fit w/o stellar activity
if called_from_terminal
    length(ARGS) > 1 ? seed = parse(Int, ARGS[2]) : seed = 0  # m/s
    length(ARGS) > 2 ? K_val = parse(Float64, ARGS[3]) : K_val = 0.3  # m/s
    length(ARGS) > 3 ? star_choice = parse(Int, ARGS[4]) : star_choice = 1  # short, long, or none
    length(ARGS) > 4 ? sample_size = parse(Int, ARGS[5]) : sample_size = 100
else
    K_val = 0.2
    seed = 48
    star_choice = 2
    sample_size = 300
end
kernel_name = "zero"
n_out = 1

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
if star_choice == 1
    # sim_ids = sample(rng, 6:20, n_sims_needed; replace=false)
    sim_ids = sample(rng, 6:55, n_sims_needed; replace=false)
    fnames = ["res-1000-1years_long_id$sim_id" for sim_id in sim_ids]
else
    sim_ids = sample(rng, 1:50, n_sims_needed; replace=false)
    fnames = ["res-1000-1years_full_id$sim_id" for sim_id in sim_ids]
end

println("optimizing on $fnames using the $kernel_name")
if called_from_terminal
    problem_definition = Jones_problem_definition(kernel_function, num_kernel_hyperparameters,"jld2_files/" .* fnames; sub_sample=sample_size, on_off=14u"d", rng=rng, n_out=n_out, n_dif=1)
else
    # problem_definition = Jones_problem_definition(kernel_function, num_kernel_hyperparameters,"../../../OneDrive/Desktop/jld2_files/" .* fnames; sub_sample=sample_size, on_off=14u"d", rng=rng, n_out=n_out)
    problem_definition = Jones_problem_definition(kernel_function, num_kernel_hyperparameters,"jld2_files/" .* fnames; sub_sample=sample_size, on_off=14u"d", rng=rng, n_out=n_out, n_dif=1)
    # problem_definition = Jones_problem_definition(kernel_function, num_kernel_hyperparameters,"jld2_files/" .* fnames; sub_sample=sample_size, rng=rng, n_out=n_out)
end
# problem_definition.a0 .= 0

########################################
# Adding planet and normalizing scores #
########################################

if star_choice == 2
    new_y = problem_definition.noise .* randn(sample_size)
    problem_definition.y_obs .= new_y
    problem_definition.rv .= new_y * problem_definition.rv_unit
end

# draw over more periods?
original_ks = kep_signal(K_val * u"m/s", (8 + 1 * randn(rng))u"d", 2 * π * rand(rng), rand(rng) / 5, 2 * π * rand(rng), 0u"m/s")
add_kepler_to_Jones_problem_definition!(problem_definition, original_ks)
if star_choice == 0
    results_dir = "results/short/$n_out/$sample_size/$(kernel_name)/K_$(string(ustrip(original_ks.K)))/seed_$(seed)/"
end
if star_choice == 1
    results_dir = "results/long/$n_out/$sample_size/$(kernel_name)/K_$(string(ustrip(original_ks.K)))/seed_$(seed)/"
end
if star_choice == 2
    results_dir = "results/none/$n_out/$sample_size/$(kernel_name)/K_$(string(ustrip(original_ks.K)))/seed_$(seed)/"
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
println()

total_hyperparameters = collect(Iterators.flatten(problem_definition.a0))
fit_nlogL1 = nlogL_Jones(problem_definition, total_hyperparameters)
uE1 = -fit_nlogL1

# sample linearly in frequency space so that we get periods from the 1 / uneven Nyquist
freq_grid = autofrequency(problem_definition.time; samples_per_peak=11)
period_grid = 1 ./ reverse(freq_grid)
amount_of_periods = length(period_grid)

Σ_obs = Σ_observations(problem_definition, total_hyperparameters)

# making necessary variables local to all workers
sendto(workers(), problem_definition=problem_definition, total_hyperparameters=total_hyperparameters, Σ_obs=Σ_obs)
@everywhere function fit_kep_hold_P(P::Unitful.Time)  # 40x slower than just epicyclic fit. neither optimized including priors
    ks = fit_kepler(problem_definition, Σ_obs, kep_signal_epicyclic(P=P))
    return fit_kepler(problem_definition, Σ_obs, kep_signal_wright(maximum([0.1u"m/s", ks.K]), ks.P, ks.M0, minimum([ks.e, 0.3]), ks.ω, ks.γ); print_stuff=false, hold_P=true, avoid_saddle=false)
end
@everywhere function kep_unnormalized_evidence_distributed(P::Unitful.Time)  # 40x slower than just epicyclic fit. neither optimized including priors
    ks = fit_kep_hold_P(P)
    if ks==nothing
        return [-Inf, -Inf]
    else
        val = nlogL_Jones(
            problem_definition,
            total_hyperparameters;
            Σ_obs=Σ_obs,
            y_obs=remove_kepler(problem_definition, ks))
        return [-val, logprior_kepler(ks; use_hk=true) - val]
    end
end

pmap(x->kep_unnormalized_evidence_distributed(x), [3.,4] * u"d", batch_size=1)
@elapsed holder = pmap(x->kep_unnormalized_evidence_distributed(x), period_grid, batch_size=Int(floor(amount_of_periods / nworkers()) + 1))
likelihoods = [holder[i][1] for i in 1:length(holder)]
unnorm_evidences = [holder[i][2] for i in 1:length(holder)]

# @time holder_lin = collect(map(kep_unnormalized_evidence_lin_distributed, period_grid_dist))
# likelihoods_lin = [holder_lin[i][1] for i in 1:length(holder_lin)]
# unnorm_evidences_lin = [holder_lin[i][2] for i in 1:length(holder_lin)]

best_period = period_grid[find_modes(unnorm_evidences; amount=1)][1]

println("original period: $(ustrip(original_ks.P)) days")
println("found period:    $(ustrip(best_period)) days")


function periodogram_plot(vals::Vector{T} where T<:Real; likelihoods::Bool=true, zoom::Bool=false, linear::Bool=true)
    ax = init_plot()
    if zoom
        inds = (minimum([original_ks.P, best_period]) / 1.5).<period_grid.<(1.5 * maximum([original_ks.P, best_period]))
        fig = plot(ustrip.(period_grid[inds]), vals[inds], color="black")
    else
        fig = plot(ustrip.(period_grid), vals, color="black")
    end
    xscale("log")
    ticklabel_format(style="sci", axis="y", scilimits=(0,0))
    xlabel("Periods (days)")
    if likelihoods
        ylabel("GP log likelihoods")
        axhline(y=-fit_nlogL1, color="k")
        ylim(-fit_nlogL1 - 3, maximum(vals) + 3)
    else
        ylabel("GP log unnormalized evidences")
        axhline(y=uE1, color="k")
    end
    axvline(x=convert_and_strip_units(u"d", best_period), color="red", linestyle="--")
    if original_ks.K != 0u"m/s"
        axvline(x=convert_and_strip_units(u"d", original_ks.P), color="blue", linestyle="--")
        title_string = @sprintf "%.1f day, %.2f m/s" convert_and_strip_units(u"d", original_ks.P) convert_and_strip_units(u"m/s",original_ks.K)
        title(title_string, fontsize=30)
    end
    file_name_add = ""
    if linear; file_name_add *= "_lin" end
    if !likelihoods; file_name_add *= "_ev" end
    if zoom; file_name_add *= "_zoom" end
    save_PyPlot_fig(results_dir * "periodogram" * file_name_add * ".png")
end

periodogram_plot(likelihoods; likelihoods=true, zoom=false, linear=false)
# periodogram_plot(likelihoods_lin; likelihoods=true, zoom=false, linear=true)
periodogram_plot(unnorm_evidences; likelihoods=false, zoom=false, linear=false)
# periodogram_plot(unnorm_evidences_lin; likelihoods=false, zoom=false, linear=true)
if original_ks.K != 0u"m/s"
    periodogram_plot(likelihoods; likelihoods=true, zoom=true, linear=false)
    # periodogram_plot(likelihoods_lin; likelihoods=true, zoom=true, linear=true)
    periodogram_plot(unnorm_evidences; likelihoods=false, zoom=true, linear=false)
    # periodogram_plot(unnorm_evidences_lin; likelihoods=false, zoom=true, linear=true)
end

current_ks = fit_kep_hold_P(best_period)
full_ks = kep_signal(current_ks.K, current_ks.P, current_ks.M0, current_ks.e, current_ks.ω, current_ks.γ)
full_ks = fit_kepler(problem_definition, Σ_obs, full_ks; print_stuff=false)
current_y = remove_kepler(problem_definition, full_ks)

fit_nlogL2 = nlogL_Jones(problem_definition, total_hyperparameters; y_obs=current_y)
uE2 = -fit_nlogL2 + logprior_kepler(full_ks; use_hk=true)

fit_nlogL3 = nlogL_Jones(problem_definition, total_hyperparameters; y_obs=remove_kepler(problem_definition, original_ks))
uE3 = -fit_nlogL3 + logprior_kepler(original_ks; use_hk=true)

Jones_line_plots(problem_definition, total_hyperparameters, results_dir * "fit")
Jones_line_plots(problem_definition, total_hyperparameters, results_dir * "fit_full"; fit_ks=full_ks)
Jones_line_plots(problem_definition, total_hyperparameters, results_dir * "truth"; fit_ks=original_ks)

##########################
# Evidence approximation #
##########################
E1 = copy(uE1)

H2 = Matrix(∇∇nlogL_kep(problem_definition.y_obs, problem_definition.time, Σ_obs, full_ks; data_unit=problem_definition.rv_unit*problem_definition.normals[1], fix_jank=true, include_priors=true))
try
    global E2 = log_laplace_approximation(Symmetric(H2), -uE2, 0)
catch
    println("Laplace approximation failed for planet fit")
    println("det(H2): $(det(H2)) (should've been positive)")
    global E2 = 0
end

# true planet
H3 = Matrix(∇∇nlogL_kep(problem_definition.y_obs, problem_definition.time, Σ_obs, original_ks; data_unit=problem_definition.rv_unit*problem_definition.normals[1], fix_jank=true, include_priors=true))
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
save_nlogLs(seed, [0, 0, 0] ./ 3600, saved_likelihoods, append!(copy(total_hyperparameters), total_hyperparameters), original_ks, full_ks, results_dir)
