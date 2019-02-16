#adding in custom functions
include("../src/all_functions.jl")

# loading in data
using JLD2, FileIO

include_kernel("Quasi_periodic_kernel")
@load "jld2_files/problem_def_528.jld2" problem_def_528 normals
problem_definition = problem_def_528

# kernel hyper parameters
kernel_lengths = [0.583594, 1.83475, 2.58466]
total_hyperparameters = append!(collect(Iterators.flatten(problem_definition.a0)), kernel_lengths)

# how finely to sample the domain
amount_of_samp_points = 50

x_samp = sort(minimum(problem_definition.x_obs) .+ (maximum(problem_definition.x_obs) - minimum(problem_definition.x_obs)) .* rand(amount_of_samp_points))

# total amount of output points
amount_of_total_samp_points = amount_of_samp_points * problem_definition.n_out

# Finding how correlated the sampled inputs are to each other
# (aka getting the covariance matrix by evaluating the kernel function at all
# pairs of points)
K_samp = covariance(problem_definition, x_samp, x_samp, total_hyperparameters)
L_samp = ridge_chol(K_samp).L
fake_data = (L_samp * randn(amount_of_total_samp_points))

measurement_noise = zeros(amount_of_total_samp_points)
# setting noise to 10% of max measurements
for i in 1:problem_definition.n_out
    measurement_noise[((i - 1) * amount_of_samp_points + 1):(i * amount_of_samp_points)] .= 0.10 * maximum(abs.(fake_data[i, :]))
end

P = 6u"d"
m_star = 1u"Msun"
m_planet = 1u"Mjup"
times_obs = convert_phases_to_seconds.(x_samp)
planet_rvs = kepler_rv.(times_obs, P, m_star, m_planet)
fake_data[1:amount_of_samp_points] += planet_rvs


# 
# function length_nyquist_sampling(x)
#     maxim




span_x = x_samp[end] - x_samp[1]
# Nyquist frequency is half of the sampling rate of a discrete signal processing system
# (https://en.wikipedia.org/wiki/Nyquist_frequency)
nyquist_samp = span_x / amount_of_samp_points
# if the frequency locations are unknown, then it is necessary to sample at
# least at twice the Nyquist criteria; in other words, you must pay at least a
# factor of 2 for not knowing the location of the spectrum. Note that minimum
# sampling requirements do not necessarily guarantee stability.
# https://en.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem#Nonuniform_sampling
uneven_nyquist_samp = nyquist_samp / 2
period_grid = collect(uneven_nyquist_samp:uneven_nyquist_samp:(span_x / 2))

kepler_rv_linear_terms = hcat(cos.(ϕ.(times, P)), sin.(ϕ.(times, P)), ones(length(times)))



x = solve_linear_system(kepler_rv_linear_terms, fake_data[1:amount_of_samp_points]; noise=measurement_noise[1:amount_of_samp_points])
kepler_rv_linear(times, P, x)
