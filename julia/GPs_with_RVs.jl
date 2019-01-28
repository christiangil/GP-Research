#adding in custom functions
include("all_functions.jl")

# can use this if you want to replicate results
# srand(1234)

# loading in data
using JLD2, FileIO
@load "sunspot_data.jld2" lambda phases quiet
@load "rv_data.jld2" doppler_comp genpca_out rvs_out
mu, M, scores = genpca_out
scores[:, 1] = rvs_out
scores = scores'

# # plot pca scores
# for i in 1:3
#     init_plot()
#     fig = plot(phases, scores[i, :])
#     xlabel("phases (days?)")
#     ylabel("pca scores")
#     title("PCA " * string(i - 1))
#     savefig("figs/pca/pca_score_" * string(i - 1) * ".pdf")
# end

# how many components you will use
n_out = 3
# how many differentiated versions of the original GP you will use
n_dif = 3

# Setting up all of the data things
# how much of the data you want to use (on time domain)
start_ind = 100
end_ind = 130  # 1070
amount_of_measurements = end_ind - start_ind + 1
total_amount_of_measurements = amount_of_measurements * n_out

# getting proper slice of data
x_obs = phases[start_ind:end_ind]
y_obs_hold = scores[1:n_out, start_ind:end_ind]

# # plot pca scores for chosen fit section
# for i in 1:3
#     init_plot()
#     fig = plot(x_obs, y_obs_hold[i, :])
#     xlabel("phases (days?)")
#     ylabel("pca scores")
#     title("PCA " * string(i - 1) * " fit section")
#     savefig("figs/pca/pca_score_" * string(i - 1) * "_section.pdf")
# end

# normalizing the data (for numerical purposes)
normals = maximum(abs.(y_obs_hold), dims=2)'[:]
for i in 1:n_out
    y_obs_hold[i, :] /= normals[i]
end

# rearranging the data into one column (not sure reshape() does what I want)
y_obs = zeros(total_amount_of_measurements)
for i in 1:n_out
    y_obs[((i - 1) * amount_of_measurements + 1):(i * amount_of_measurements)] = y_obs_hold[i, :]
end

# # plot subsection that will be fit
# for i in 1:3
#     init_plot()
#     fig = plot(x_obs, y_obs_hold[i, :])
#     xlabel("phases (days?)")
#     ylabel("pca scores")
#     title("PCA " * string(i - 1) * " fit section")
#     savefig("figs/pca/pca_score_" * string(i - 1) * "_section.pdf")
# end

# Uncertainty in the data (AFFECTS SPREAD OF DATA AND HOW TIGHTLY THE GP's WILL
# TRY TO HUG THE DATA) aka how much noise is added to measurements and
# measurement covariance function
# a vector of length = total_amount_of_measurements
measurement_noise = ones(total_amount_of_measurements)
# currently set to 5 percent of total amplitude at every point. should be done with bootstrapping
for i in 1:n_out
    measurement_noise[((i - 1) * amount_of_measurements + 1):(i * amount_of_measurements)] *= 0.05 * maximum(abs.(y_obs_hold[i, :]))
end


a0 = ones(n_out, n_dif) / 20

include("kernels/Quasi_periodic_kernel.jl")  # sets correct num_kernel_hyperparameters
build_problem_definition(Quasi_periodic_kernel, num_kernel_hyperparameters, n_dif, n_out, x_obs, y_obs, measurement_noise, a0)


##############################################################################

# kernel hyper parameters
# AFFECTS SHAPE OF THE GP's
# make sure you have the right amount!

kernel_lengths = [1, 1, 1] / 1.5
total_hyperparameters = append!(collect(Iterators.flatten(a0)), kernel_lengths)
# hyperparameters=rand(n_out * n_dif + 3)

# how finely to sample the domain (for plotting)
amount_of_samp_points = max(100, convert(Int64, round(sqrt(2) * 3 * amount_of_measurements)))

# creating many inputs to sample the eventual gaussian process on (for plotting)
x_samp = linspace(minimum(x_obs),maximum(x_obs), amount_of_samp_points[1])

# total amount of output points
amount_of_total_samp_points = amount_of_samp_points * n_out

# Finding how correlated the sampled inputs are to each other
# (aka getting the covariance matrix by evaluating the kernel function at all
# pairs of points)
K_samp = covariance(problem_definition, x_samp, x_samp, total_hyperparameters)

# getting the Cholesky factorization of the covariance matrix (for drawing GPs)
L_samp = ridge_chol(K_samp).L  # usually has to add a ridge


function plot_im(A; file::String="")
    init_plot()
    fig = imshow(A[:,:])
    colorbar()
    title("Heatmap")
    if file != ""
        savefig(file)
    end
end


# showing a heatmap of the covariance matrix
# plot_im(K_samp; file="figs/gp/initial_covariance.pdf")
# plot_im(L_samp)


# quick and dirty function for creating plots that show what I want
function custom_line_plot(x_samp::Array{Float64,1}, L, x_obs::Array{Float64,1}, y_obs::Array{Float64,1}; output::Int=1, draws::Int=5000, σ::Array{Float64,1}=zeros(1), mean::Array{Float64,1}=zeros(amount_of_total_samp_points), show::Int=10, file::String="")

    # same curves are drawn every time?
    # srand(100)

    # initialize figure size
    init_plot()

    # the indices corresponding to the proper output
    output_indices = (amount_of_samp_points * (output - 1) + 1):(amount_of_samp_points * output)

    # geting the y values for the proper output
    y = y_obs[(amount_of_measurements * (output - 1) + 1):(amount_of_measurements * output)]

    # initializing storage for example GPs to be plotted
    show_curves = zeros(show, amount_of_samp_points)

    # if no analytical variance is passed, estimate it with sampling
    if σ == zeros(1)

        # calculate a bunch of GP draws
        storage = zeros((draws, amount_of_total_samp_points))
        for i in 1:draws
            storage[i, :] = (L * randn(amount_of_total_samp_points)) + mean
        end

        # ignore the outputs that aren't meant to be plotted
        storage = storage[:, output_indices]

        #
        show_curves[:, :] = storage[1:show, :]
        storage = sort(storage, dims=1)

        # filling the 5-95th percentile with a transparent orange
        fill_between(x_samp, storage[convert(Int64, 0.95 * draws), :], storage[convert(Int64, 0.05 * draws), :], alpha=0.3, color="orange")

        # needs to be in both leaves of the if statement
        mean = mean[output_indices]

    else

        storage = zeros((show, amount_of_total_samp_points))
        for i in 1:show
            storage[i,:] = L * randn(amount_of_total_samp_points) + mean
        end
        show_curves[:, :] = storage[:, output_indices]

        σ = σ[output_indices]

        mean = mean[output_indices]


        # filling the 5-95th percentile with a transparent orange
        fill_between(x_samp, mean + 1.96 * σ, mean - 1.96 * σ, alpha=0.3, color="orange")

    end


    plot(x_samp, mean, color="black", zorder=2)
    for i in 1:show
        plot(x_samp, show_curves[i, :], alpha=0.5, zorder=1)
    end
    scatter(x_obs, y, color="black", zorder=2)

    xlabel("phases (days?)")
    ylabel("pca scores")
    title("PCA " * string(output-1))

    if file!=""
        savefig(file)
    end

end


# custom_line_plot(x_samp, L_samp, x_obs, y_obs, output=1, file="figs/gp/initial_gp_0.pdf")
# custom_line_plot(x_samp, L_samp, x_obs, y_obs, output=2, file="figs/gp/initial_gp_1.pdf")
# custom_line_plot(x_samp, L_samp, x_obs, y_obs, output=3, file="figs/gp/initial_gp_2.pdf")


# calculate posterior quantities
mean_post, return_vec = GP_posteriors(problem_definition, x_samp, total_hyperparameters, return_K=true, return_L=true, return_σ=true)
σ, K_post, L_post = return_vec

# plot the posterior covariance matrix (colors show how correlated points are to each other)
# plot_im(K_post)
# plot_im(L_post)

# plotting randomly drawn Gaussian processes with the mean function and covariance of the posterior
# much closer to the data, no?

# for 1D GPs
# custom_line_plot(x_samp, L_post, x_obs, y_obs, σ=σ, mean=mean_post, output=1, file="figs/gp/cond_initial_gp_1.pdf")
# custom_line_plot(x_samp, L_post, x_obs, y_obs, σ=σ, mean=mean_post, output=2, file="figs/gp/cond_initial_gp_2.pdf")
# custom_line_plot(x_samp, L_post, x_obs, y_obs, σ=σ, mean=mean_post, output=3, file="figs/gp/cond_initial_gp_3.pdf")

# numerically maximize the likelihood to find the best hyperparameters
using Optim

# storing initial hyperparameters
initial_x = copy(total_hyperparameters)

# initializing Cholesky factorization storage
chol_storage = chol_struct(initial_x, ridge_chol(K_observations(problem_definition, initial_x)))
# chol_storage = chol_struct(zeros(1), ridge_chol(hcat([1,.1],[.1,1])))

# lower = zeros(length(hyperparameters))
# lower = -5 * ones(length(hyperparameters))
# lower[(num_coefficients + 1):length(hyperparameters)] = zeros(length(hyperparameters) - num_coefficients)
# upper = 5 * ones(length(hyperparameters))

# can optimize with or without using the analytical gradient
# times listed are for optimizing 2 outputs with first order derivatives (7 hyperparameters)
# the optimization currently doesn't work because it looks for kernel lengths that are either negative (unphysical) or way too large kernel lengths (flatten covariance and prevents positice definiteness)

# use gradient
# http://julianlsolvers.github.io/Optim.jl/stable/#user/gradientsandhessians/
# result = optimize(nlogL, ∇nlogL, lower, upper, initial_x, Fminbox(GradientDescent()))  # 272.3 s
# @elapsed result = optimize(nlogL, ∇nlogL, initial_x, ConjugateGradient())  # 54.0 s, gave same result as Fminbox
# @elapsed result = optimize(nlogL, ∇nlogL, initial_x, GradientDescent(), Optim.Options(iterations=2, show_trace=true))  # 116.8 s, gave same result as SA
@elapsed result = optimize(nlogL_Jones, ∇nlogL_Jones, initial_x, LBFGS(), Optim.Options(show_trace=true))
  # 19.8 s, unique result, takes around 5 mins with 50 real data points and 3 outputs for 3 hyperparameter model (9 total)
# @elapsed result = optimize(nlogL_penalty, ∇nlogL_penalty, initial_x, LBFGS())
# @elapsed result = optimize(nlogL, ∇nlogL, initial_x)
# result = optimize(nlogL_penalty, ∇nlogL_penalty, initial_x, LBFGS())  # same as above but with a penalty


# don't use gradient
# @elapsed result = optimize(nlogL, lower, upper, initial_x, SAMIN(), Optim.Options(iterations=10^6))  # 253.4 s

final_total_hyperparameters = result.minimizer

println("old hyperparameters")
println(initial_x)
println(nlogL(initial_x))

println("new hyperparameters")
println(final_total_hyperparameters)
println(result.minimum)


# reruning analysis of posterior with the "most likley" kernel amplitude and lengthscale
# recalculating posterior covariance and mean function
mean_post, return_vec = GP_posteriors(problem_definition, x_samp, final_total_hyperparameters, return_K=true, return_L=true, return_σ=true)
σ, K_post, L_post = return_vec

# plot the posterior covariance of the "most likely" posterior matrix
# (colors show how correlated points are to each other)
# plot_im(K_post)


# for 1D GPs
# custom_line_plot(x_samp, L_post, x_obs, y_obs, σ=σ, mean=mean_post, output=1, file="figs/gp/fit_gp_1.pdf")
# custom_line_plot(x_samp, L_post, x_obs, y_obs, σ=σ, mean=mean_post, output=2, file="figs/gp/fit_gp_2.pdf")
# custom_line_plot(x_samp, L_post, x_obs, y_obs, σ=σ, mean=mean_post, output=3, file="figs/gp/fit_gp_3.pdf")
