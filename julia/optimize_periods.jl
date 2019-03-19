#adding in custom functions
include("src/all_functions.jl")
run_tests()

if length(ARGS)>0
    amount_of_periods = parse(Int, ARGS[1])
else
    amount_of_periods = 300
end

# loading in data
using JLD2, FileIO

include_kernel("quasi_periodic_kernel")
@load "jld2_files/problem_def_528.jld2" problem_def_528 normals

# original kernel hyper parameters
kernel_lengths = [0.6, 2, 2.5]
total_hyperparameters_og = append!(collect(Iterators.flatten(problem_def_528.a0)), kernel_lengths)

# adding some noise so we aren't using original values
total_hyperparameters = zeros(length(total_hyperparameters_og))
for i in 1:length(total_hyperparameters)
    if total_hyperparameters_og[i]!=0
        total_hyperparameters[i] = total_hyperparameters_og[i] * (1 + 0.2 * randn())
    end
end

amount_of_samp_points = length(problem_def_528.x_obs)
amount_of_total_samp_points = amount_of_samp_points * problem_def_528.n_out

P = 30u"d"
m_star = 1u"Msun"
m_planet = 50u"Mearth"
times_obs = convert_and_strip_units.(u"yr", (problem_def_528.x_obs)u"d")
planet_rvs = kepler_rv.(times_obs, P, m_star, m_planet)
fake_data = copy(problem_def_528.y_obs)
fake_data[1:amount_of_samp_points] += planet_rvs/normals[1]

# sample linearly in frequency space so that we get periods from the 1 / uneven Nyquist
# frequency to 4 times the total timespan of the data
freq_grid = linspace(1 / (times_obs[end] - times_obs[1]) / 4, uneven_nyquist_frequency(times_obs), amount_of_periods)
period_grid = 1 ./ reverse(freq_grid)

likelihoods = kep_signal_likelihood(period_grid, times_obs, fake_data, problem_def_528, total_hyperparameters)

begin
    ax = init_plot()
    ticklabel_format(style="sci", axis="y", scilimits=(0,0))
    fig = semilogx(period_grid .* convert_and_strip_units(u"d", 1u"yr"), -likelihoods, color="black")
    xlabel("Periods (days)")
    ylabel("GP likelihoods")
    axvline(x=convert_and_strip_units(u"d", P))
    title_string = @sprintf "%.0f day, %.2f Earth masses" convert_and_strip_units(u"d",P) convert_and_strip_units(u"Mearth",m_planet)
    title(title_string, fontsize=30)
    savefig("figs/rv/test$amount_of_periods.png")
    PyPlot.close_figs()
end

# # three best periods
# best_period_grid = period_grid[find_modes(-likelihoods)]
# K_obs = K_observations(problem_def_528, total_hyperparameters)
# fake_data - remove_kepler(fake_data, times_obs, best_period_grid[1], K_obs)
#
# kepler_rv.(times_obs, best_period_grid[1], m_star, m_planet)
# Jones_line_plots(amount_of_samp_points, problem_definition, final_total_hyperparameters; file="figs/gp/fit_gp", plot_K=true)
#
# best_period_grid * convert_and_strip_units(u"d", 1u"yr")  # in days instead of years
# K_obs = K_observations(problem_def_528, total_hyperparameters)
#
# using Flux; using Flux: @epochs; using Flux.Tracker: track, @grad, data
#
# for period in best_period_grid
#
#     new_y_obs = remove_kepler(fake_data, times_obs, period, K_obs)
#
#     # Allowing Flux to use the analytical gradients we have calculated
#     nLogL_custom(non_zero_hyper) = nlogL_Jones(problem_def_528, non_zero_hyper; y_obs=new_y_obs)
#     nLogL_custom(non_zero_hyper::TrackedArray) = track(nLogL_custom, non_zero_hyper)
#     @grad nLogL_custom(non_zero_hyper) = nLogL_custom(data(non_zero_hyper)), Δ -> tuple(Δ .* ∇nlogL_Jones(problem_def_528, data(non_zero_hyper); y_obs=new_y_obs))
#
#     # Setting model parameters for Flux
#     non_zero_hyper_param = param(total_hyperparameters[findall(!iszero, total_hyperparameters)])
#     ps = Flux.params(non_zero_hyper_param)
#
#     # Final function wrapper for Flux
#     nLogL_custom() = nLogL_custom(non_zero_hyper_param)
#
#     # Initializing other training things
#     iteration_amount = 10
#     flux_data = Iterators.repeated((), iteration_amount)    # the function is called $iteration_amount times with no arguments
#     opt = ADAM(0.1)
#
#     global grad_norm = 1e4
#     global epoch_num = 0
#     while grad_norm>1e2
#         global epoch_num += 10
#         Flux.train!(nLogL_custom, ps, flux_data, opt)
#         global grad_norm = norm(∇nlogL_Jones(problem_def_528, data(non_zero_hyper_param); y_obs=new_y_obs))
#         println("Epoch $epoch_num gradient norm: ", grad_norm)
#     end
#
#     final_total_hyperparameters = reconstruct_total_hyperparameters(problem_def_528, data(non_zero_hyper_param))
#
#     println("starting hyperparameters")
#     println(total_hyperparameters)
#     println(nlogL_Jones(problem_def_528, total_hyperparameters), "\n")
#
#     println("ending hyperparameters")
#     println(final_total_hyperparameters)
#     println(nlogL_Jones(problem_def_528, final_total_hyperparameters), "\n")
#
#     println("original hyperparameters")
#     println(total_hyperparameters_og)
#     println(nlogL_Jones(problem_def_528, total_hyperparameters_og), "\n")
#
# end
