using Distributed
if nworkers() < 2
    addprocs(1)
end

@everywhere old_dir = pwd()
@everywhere cd(@__DIR__)
@everywhere cd("..")
@everywhere include("src/base_functions.jl")
@everywhere cd(old_dir)

# loading in data
using JLD2, FileIO

kernel_name="quasi_periodic_kernel"
include_kernel(kernel_name)
@load "../jld2_files/problem_def_528.jld2" problem_def_528 normals

# original kernel hyper parameters
kernel_lengths = [0.6, 2, 2.5]
total_hyperparameters = append!(collect(Iterators.flatten(problem_def_528.a0)), kernel_lengths)

amount_of_samp_points = length(problem_def_528.x_obs)
amount_of_total_samp_points = amount_of_samp_points * problem_def_528.n_out

P = 30u"d"
m_star = 1u"Msun"
m_planet = 1u"Mjup"
times_obs = convert_and_strip_units.(u"yr", (problem_def_528.x_obs)u"d")
planet_rvs = kepler_rv.(times_obs, P, m_star, m_planet)
fake_data = copy(problem_def_528.y_obs)
fake_data[1:amount_of_samp_points] += planet_rvs/normals[1]

period_grid = [1, 30., 400] ./ 365

K_obs = K_observations(problem_def_528, total_hyperparameters)

likelihood_func(new_data::AbstractArray{T,1}) where{T<:Real} = nlogL_Jones(problem_def_528, total_hyperparameters, y_obs=new_data)
likelihoods_serial = kep_signal_likelihoods(likelihood_func, period_grid, times_obs, fake_data, K_obs)

# making necessary variables local to all workers
sendto(workers(), kernel_name=kernel_name)
@everywhere include_kernel(kernel_name)
sendto(workers(), times_obs=times_obs, fake_data=fake_data, problem_def_528=problem_def_528, total_hyperparameters=total_hyperparameters, K_obs=K_obs)
@everywhere likelihood_func(new_data::AbstractArray{T,1}) where{T<:Real} = nlogL_Jones(problem_def_528, total_hyperparameters, y_obs=new_data)
@everywhere kep_signal_likelihood_distributed(period::Real) = kep_signal_likelihood(likelihood_func, period, times_obs, fake_data, K_obs)

# parallelize with pmap
likelihoods_pmap = pmap(x->kep_signal_likelihood_distributed(x), period_grid, batch_size=floor(length(period_grid) / nworkers()) + 1)

# parallelize with SharedArrays
@everywhere using SharedArrays
likelihoods_shared = SharedArray{Float64}(length(period_grid))
@sync @distributed for i in 1:length(period_grid)
    likelihoods_shared[i] = kep_signal_likelihood_distributed(period_grid[i])
end
likelihoods_shared = convert(Array, likelihoods_shared)

# parallelize with DistributedArrays
@everywhere using DistributedArrays
likelihoods_dist = collect(map(kep_signal_likelihood_distributed, distribute(period_grid)))

@testset "parallel likelihood calculations equivalent" begin
    @test isapprox(likelihoods_serial, likelihoods_pmap)
    @test isapprox(likelihoods_serial, likelihoods_shared)
    @test isapprox(likelihoods_serial, likelihoods_dist)
    println()
end

@testset "GP likelihood highest at planet period" begin
    @test likelihoods_serial[2] < min(likelihoods_serial[1], likelihoods_serial[3])
    println()
end
