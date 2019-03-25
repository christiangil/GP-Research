# getting packages ready and making sure they are up to date
include("src/setup.jl")
include("src/all_functions.jl")

using Distributed
# nworkers()
addprocs(7)

#adding custom functions to all processes
@everywhere include("src/all_functions.jl")

if length(ARGS)>0
    amount_of_periods = parse(Int, ARGS[1])
else
    amount_of_periods = 128
end

# loading in data
using JLD2, FileIO

include_kernel("quasi_periodic_kernel")
@load "jld2_files/problem_def_528.jld2" problem_def_528 normals

# original kernel hyper parameters
kernel_lengths = [0.6, 2, 2.5]
total_hyperparameters_og = append!(collect(Iterators.flatten(problem_def_528.a0)), kernel_lengths)

# adding some noise so we aren't using original values
total_hyperparameters = total_hyperparameters_og .* (1 .+ 0.2 * randn(length(total_hyperparameters_og)))

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

K_obs = K_observations(problem_def_528, total_hyperparameters)

# @time likelihoods = kep_signal_likelihoods(period_grid, times_obs, fake_data, problem_def_528, K_obs)

# making necessary variables local to all workers
@sync @everywhere include_kernel("quasi_periodic_kernel")
for i in workers()
    remotecall_fetch(()->times_obs, i)
    remotecall_fetch(()->fake_data, i)
    remotecall_fetch(()->problem_def_528, i)
    remotecall_fetch(()->K_obs, i)
    remotecall_fetch(()->total_hyperparameters, i)
end

# InteractiveUtils.varinfo()
# @fetchfrom 2 InteractiveUtils.varinfo()

@sync @everywhere kep_signal_likelihood_distributed(period::Real) = nlogL_Jones(problem_def_528, total_hyperparameters, y_obs=remove_kepler(fake_data, times_obs, period, K_obs))
@sync @everywhere kep_signal_likelihood_distributed(4)

pmap_time = @elapsed likelihoods_pmap = pmap(x->kep_signal_likelihood_distributed(x), period_grid, batch_size=floor(amount_of_periods / nworkers()) + 1)

@everywhere using SharedArrays
likelihoods_shared = SharedArray{Float64}(length(period_grid))
@elapsed @sync @distributed for i in 1:length(period_grid)
    likelihoods_shared[i] = kep_signal_likelihood_distributed(period_grid[i])
end

@everywhere using DistributedArrays
@time likelihoods_dist = collect(map(kep_signal_likelihood_distributed, distribute(period_grid)))
