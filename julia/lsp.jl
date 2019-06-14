# adding in custom functions
# include("src/setup.jl")
# include("test/runtests.jl")
include("src/all_functions.jl")

using LombScargle
###################################
# Loading data and setting kernel #
###################################

kernel_names = ["quasi_periodic_kernel", "se_kernel", "rq_kernel", "matern52_kernel"]
kernel_name = kernel_names[4]
# @load "jld2_files/res-1000-lambda-3923-6664-1years_1579spots_diffrot_id11_problem_def_sample_base.jld2" problem_def_base
@load "jld2_files/res-1000-lambda-3923-6664-1years_1579spots_diffrot_id11_problem_def_full_base.jld2" problem_def_base
kernel_function, num_kernel_hyperparameters = include_kernel(kernel_name)
problem_definition = build_problem_definition(kernel_function, num_kernel_hyperparameters, problem_def_base)

# allowing covariance matrix to be calculated in parallel
prep_parallel_covariance(kernel_name)

n_obs = length(problem_definition.x_obs)
for i in 1:problem_definition.n_out
    problem_definition.normals[i] = std(problem_definition.y_obs[1 + (i - 1) * n_obs : i * n_obs])
    problem_definition.y_obs[1 + (i - 1) * n_obs : i * n_obs] /= problem_definition.normals[i]
    problem_definition.noise[1 + (i - 1) * n_obs : i * n_obs] /= problem_definition.normals[i]
end

# https://github.com/JuliaAstro/LombScargle.jl/blob/master/docs/src/index.md
i=1
# for i in 1:problem_definition.n_out
plan = LombScargle.plan(problem_definition.x_obs, problem_definition.y_obs[1 + (i - 1) * n_obs : i * n_obs], problem_definition.noise[1 + (i - 1) * n_obs : i * n_obs])

# Compute the periodogram
pgram = lombscargle(plan)
maxpow = findmaxpower(pgram)
prob(pgram, findmaxpower(pgram))  # probability that the periodogram power p can exceed the value p0
probinv(pgram, 0.001)  # periodogram power corresponding to a probability
 # FAP is the probability that at least one out of M independent power values in
 # a prescribed search band of a power spectrum computed from a white-noise time
 # series is expected to be as large as or larger than a given value
fap(pgram, findmaxpower(pgram))
fapinv(pgram, 0.001)
plan2 = LombScargle.plan(problem_definition.x_obs, problem_definition.y_obs[1 + (i - 1) * n_obs : i * n_obs], problem_definition.noise[1 + (i - 1) * n_obs : i * n_obs])
lspboot = LombScargle.bootstrap(2000, plan2)
# fap(lspboot, maxpow)
fapinv(lspboot, 0.001)


findmaxfreq(pgram)
findmaxperiod(pgram)

begin
    ax = init_plot()
    fig = plot(periodpower(pgram)...)
    xscale("log")
    xlabel("Periods (days)")
    ylabel("Normalized Power")
    title("L-S Periodogram $(i-1)", fontsize=30)
    save_PyPlot_fig("lsp$(i-1).png")
end
# end
