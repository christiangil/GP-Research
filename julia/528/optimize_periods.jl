#adding in custom functions
include("../src/all_functions.jl")

# loading in data
using JLD2, FileIO

include_kernel("Quasi_periodic_kernel")
@load "jld2_files/problem_def_528.jld2" problem_def_528 normals

# kernel hyper parameters
kernel_lengths = [0.6, 2, 2.5]
total_hyperparameters = append!(collect(Iterators.flatten(problem_def_528.a0)), kernel_lengths)

amount_of_samp_points = length(problem_def_528.x_obs)
amount_of_total_samp_points = amount_of_samp_points * problem_def_528.n_out

P = 6u"d"
m_star = 1u"Msun"
m_planet = 1u"Mjup"
times_obs = convert_phases_to_seconds.(problem_def_528.x_obs)
planet_rvs = kepler_rv.(times_obs, P, m_star, m_planet)
fake_data = copy(problem_def_528.y_obs)
fake_data[1:amount_of_samp_points] += planet_rvs

time_span = times_obs[end] - times_obs[1]
# Nyquist frequency is half of the sampling rate of a discrete signal processing system
# (https://en.wikipedia.org/wiki/Nyquist_frequency)
# if the frequency locations are unknown, then it is necessary to sample at
# least at twice the Nyquist criteria; in other words, you must pay at least a
# factor of 2 for not knowing the location of the spectrum. Note that minimum
# sampling requirements do not necessarily guarantee stability.
# https://en.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem#Nonuniform_sampling
uneven_nyquist_spacing = (time_span / amount_of_samp_points)

period_grid = log_linspace(uneven_nyquist_spacing, time_span / 2, 100)

K_obs = K_observations(problem_def_528, total_hyperparameters)
likelihoods = zeros(length(period_grid))
for i in 1:length(period_grid)
    period = period_grid[i]
    kepler_rv_linear_terms = hcat(cos.(ϕ.(times_obs, period)), sin.(ϕ.(times_obs, period)), ones(length(times_obs)))
    kepler_linear_terms = vcat(kepler_rv_linear_terms, zeros(amount_of_total_samp_points - amount_of_samp_points, 3))
    x = general_lst_sq(kepler_linear_terms, fake_data; covariance=K_obs)
    new_data = copy(fake_data)
    new_data[1:amount_of_samp_points] -= kepler_rv_linear(times_obs, period, x)
    likelihoods[i] = nlogL_Jones(problem_def_528, total_hyperparameters, y_obs=new_data)
end

ax = init_plot()
ticklabel_format(style="sci", axis="y", scilimits=(0,0))
fig = semilogx(period_grid .* convert_and_strip_units(u"d", 1u"s"), -likelihoods, color="black")
xlabel("Periods (days)")
ylabel("GP likelihoods")
axvline(x=convert_and_strip_units(u"d", P))
title_string = @sprintf "%.0f day, %.2f Earth masses" convert_and_strip_units(u"d",P) convert_and_strip_units(u"Mearth",m_planet)
title(title_string, fontsize=30)
savefig("test.pdf")
