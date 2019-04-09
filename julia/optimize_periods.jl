# getting packages ready and making sure they are up to date
include("src/setup.jl")
include("src/all_functions.jl")
# include("test/runtests.jl")

using Distributed
if (nworkers()==1) & (length(Sys.cpu_info())<18)  # only add processors if we are on a consumer chip
    addprocs(length(Sys.cpu_info()) - 2)
end

#adding custom functions to all processes
@everywhere include("src/base_functions.jl")

if length(ARGS)>0
    amount_of_periods = parse(Int, ARGS[1])
else
    amount_of_periods = 512
end

# loading in data
using JLD2, FileIO

kernel_name = "se_kernel"
@load "jld2_files/problem_def_full_base.jld2" problem_def_full_base normals
kernel_function, num_kernel_hyperparameters = include_kernel(kernel_name)
problem_def_528 = build_problem_definition(kernel_function, num_kernel_hyperparameters, problem_def_full_base)

# best-fit se_kernel hyper parameters on original data
total_hyperparameters = [0.234374, 0.97434, 0.0, 0.501636, 0.0, 0.510968, 0.0, -0.000116937, 0.0, 0.540127]

amount_of_samp_points = length(problem_def_528.x_obs)
amount_of_total_samp_points = amount_of_samp_points * problem_def_528.n_out

P = 30u"d"
m_star = 1u"Msun"
m_planet = 1u"Mearth"
times_obs = convert_and_strip_units.(u"yr", (problem_def_528.x_obs)u"d")
planet_rvs = kepler_rv.(times_obs, P, m_star, m_planet)
fake_data = copy(problem_def_528.y_obs)
fake_data[1:amount_of_samp_points] += planet_rvs/normals[1]

# sample linearly in frequency space so that we get periods from the 1 / uneven Nyquist
# frequency to 4 times the total timespan of the data
freq_grid = linspace(1 / (times_obs[end] - times_obs[1]) / 4, uneven_nyquist_frequency(times_obs), amount_of_periods)
period_grid = 1 ./ reverse(freq_grid)

K_obs = K_observations(problem_def_528, total_hyperparameters)
likelihood_func(new_data::AbstractArray{T,1}) where{T<:Real} = nlogL_Jones(problem_def_528, total_hyperparameters, y_obs=new_data)


if length(ARGS)>1
    parallelize = parse(Int, ARGS[2])
    @assert parallelize >= 0
else
    parallelize = 2
end

if parallelize == 0

    kep_signal_likelihoods(likelihood_func, period_grid[1:2], times_obs, fake_data, K_obs)
    serial_time = @elapsed likelihoods = kep_signal_likelihoods(likelihood_func, period_grid, times_obs, fake_data, K_obs)
    println("Serial likelihood calculation took $(serial_time)s")

else

    # making necessary variables local to all workers
    @sync @everywhere include_kernel("quasi_periodic_kernel")
    @sync sendto(workers(), times_obs=times_obs, fake_data=fake_data, problem_def_528=problem_def_528, total_hyperparameters=total_hyperparameters)

    # @sync for i in workers()
    #     remotecall_fetch(()->times_obs, i)
    #     remotecall_fetch(()->fake_data, i)
    #     remotecall_fetch(()->problem_def_528, i)
    #     remotecall_fetch(()->K_obs, i)
    #     remotecall_fetch(()->total_hyperparameters, i)
    # end

    @everywhere kep_signal_likelihood_distributed(period::Real) = kep_signal_likelihood(likelihood_func, period, times_obs, fake_data, K_obs)
    @sync @everywhere kep_signal_likelihood_distributed(4)  # make sure everything is compiled


    if parallelize == 2

        # parallelize with pmap
        pmap(x->kep_signal_likelihood_distributed(x), [1, 2], batch_size=2)  # make sure everything is compiled
        parallel_time = @elapsed likelihoods = pmap(x->kep_signal_likelihood_distributed(x), period_grid, batch_size=floor(amount_of_periods / nworkers()) + 1)

    elseif parallelize == 3

        # parallelize with SharedArrays
        @everywhere using SharedArrays
        likelihoods = SharedArray{Float64}(length(period_grid))
        @sync @distributed for i in 1:2  # make sure everything is compiled
            likelihoods[i] = kep_signal_likelihood_distributed(period_grid[i])
        end
        parallel_time = @elapsed @sync @distributed for i in 1:length(period_grid)
            likelihoods[i] = kep_signal_likelihood_distributed(period_grid[i])
        end

    else

        # parallelize with DistributedArrays
        @everywhere using DistributedArrays
        collect(map(kep_signal_likelihood_distributed, distribute([1, 2])))  # make sure everything is compiled
        period_grid_dist = distribute(period_grid)
        parallel_time = @elapsed likelihoods = collect(map(kep_signal_likelihood_distributed, period_grid_dist))

    end

    println("Parallel likelihood calculation took $(parallel_time)s")

end

# begin
#     ax = init_plot()
#     ticklabel_format(style="sci", axis="y", scilimits=(0,0))
#     fig = semilogx(period_grid .* convert_and_strip_units(u"d", 1u"yr"), -likelihoods, color="black")
#     xlabel("Periods (days)")
#     ylabel("GP likelihoods")
#     axvline(x=convert_and_strip_units(u"d", P))
#     title_string = @sprintf "%.0f day, %.2f Earth masses" convert_and_strip_units(u"d",P) convert_and_strip_units(u"Mearth",m_planet)
#     title(title_string, fontsize=30)
#     savefig("figs/rv/test$amount_of_periods.png")
#     PyPlot.close_figs()
# end


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
