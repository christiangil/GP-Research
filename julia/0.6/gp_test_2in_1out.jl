# importing packages
# Pkg.add("PlotlyJS")
using PlotlyJS

# a generalized version of the built in append!() function
function append(a, b...)
    for i in 1:length(b)
        append!(a, b[i])
    end
    return a
end


# defining kernels
# Linear kernel
function linear_kernel(hyperparameters, x1, x2)
    sigma_b, sigma_a = hyperparameters
    return sigma_b ^ 2 * vecdot(x1, x2) + sigma_a ^ 2
end


# Radial basis function kernel(aka squared exonential, ~gaussian)
function RBF_kernel(hyperparameters, dif_sq)

    if length(hyperparameters) > 1
        kernel_amplitude, kernel_length = hyperparameters
    else
        kernel_amplitude = 1
        kernel_length = hyperparameters
    end

    return kernel_amplitude ^ 2 * exp(-dif_sq / (2 * kernel_length ^ 2))
end


# Differentiated Radial basis function kernel (aka squared exonential, ~gauss)
# ONLY WORKS FOR 1D TIME AS INPUTS
# HAS A CONSTANT AMPLITUDE
function dRBF_kernel(hyperparameters, dif, dorder)
    for i in length(dorder)
        dorder[i] = max(0,convert(Int64, dorder[i]))
    end

    value = RBF_kernel(hyperparameters, dif ^ 2)

    if length(hyperparameters) > 1
        kernel_amplitude, kernel_length = hyperparameters
    else
        kernel_amplitude = 1
        kernel_length = hyperparameters
    end

    # first coefficients are triangular numbers. second coefficients are
    # tri triangular numbers
    if sum(dorder) > 0
        T1 = dif / kernel_length ^ 2
        if sum(dorder) == 1
            value = value * T1
            if dorder == [1, 0]
                value = -value
            # elseif dorder == [0 ,1]
            end
        elseif sum(dorder) == 2
            value = value * (T1 ^ 2 - 1 / kernel_length ^ 2)
            # if dorder == [2, 0] || dorder == [0, 2]
            if dorder==[1, 1]
                value =  -value
            end
        elseif sum(dorder) == 3
            value = value * (T1 ^ 3 - 3 * dif / (kernel_length ^ 4))
            # if dorder == [2, 1]
            if dorder == [1, 2]
                value =  -value
            end
        elseif sum(dorder) == 4
            value = value * (T1 ^ 4 - 6 * dif ^ 2 / (kernel_length ^ 6)
                + 3 / (kernel_length ^ 4))
            # if dorder == [2, 2]
        end
    end
    return value
end


#Ornstein–Uhlenbeck (Exponential) kernel
function OU_kernel(hyperparameters, dif)
    kernel_amplitude, kernel_length = hyperparameters
    return kernel_amplitude ^ 2 * exp(-dif / kernel_length)
end


# Periodic kernel
function periodic_kernel(hyperparameters, dif)
    kernel_amplitude, kernel_length, kernel_period = hyperparameters
    return kernel_amplitude ^ 2 * exp(-2 * sin(pi * dif / kernel_period) ^ 2 / (kernel_length ^ 2))
end


using SpecialFunctions


#Matern kernel (not sure if I implemented this correctly. Gets nans when dif==0)
function Matern_kernel(hyperparameters, dif, nu)
    kernel_amplitude, kernel_length = hyperparameters
    #limit of the function as it apporaches 0 (see https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function)
    if dif == 0
        return kernel_amplitude ^ 2
    else
        return kernel_amplitude ^ 2 * ((2 ^ (1 - nu)) / (gamma(nu))) * ((sqrt(2 * nu) * dif) / kernel_length) ^ nu * besselk(nu, (sqrt(2 * nu) * dif) / kernel_length)
    end
end


#Rational Quadratic kernel (equivalent to adding together many SE kernels
#with different lengthscales. When α→∞, the RQ is identical to the SE.)
function RQ_kernel(hyperparameters, dif_sq)
    kernel_amplitude, kernel_length, alpha = hyperparameters
    alpha = max(alpha, 0)
    return kernel_amplitude ^ 2 * (1 + dif_sq / (2 * alpha * kernel_length ^ 2)) ^ -alpha
end


# Creating a custom kernel (possibly by adding and multiplying other kernels?)
# x1 and x2 are single data points
function kernel(hyperparameters, x1, x2; dorder=[0,0])
    # finding squared differences between inputs
    dif_vec = x1 - x2
    dif_sq_vec = dif_vec .^ 2  # element-wise squaring
    dif_sq_tot = sum(dif_sq_vec)
    dif_tot = sqrt(dif_sq_tot)

    # final = RBF_kernel(hyperparameters[1:2], dif_sq_tot)
    # final = RBF_kernel(append!([1],hyperparameters[1]), dif_sq_tot) #constant amplitude RBF
    # final = OU_kernel(hyperparameters[1:2], dif_tot)
    # final = linear_kernel(hyperparameters[1:2], x1a, x2a)
    # final = periodic_kernel(hyperparameters[1:3], dif_tot)
    # final = RQ_kernel(hyperparameters[1:3], dif_sq_tot)
    # final = Matern_kernel(hyperparameters[1:2], dif_tot, 3/2)  # doesn't seem to be working right now

    # example of adding kernels
    # final = periodic_kernel(hyperparameters[1:3], dif_tot) + RBF_kernel(hyperparameters[4:5], dif_sq_tot)

    # example of multiplying kernels
    # final = periodic_kernel(hyperparameters[1:3], dif_tot) * RBF_kernel(hyperparameters[4:5], dif_sq_tot)

    # example of independent multivariate kernel
    final = RBF_kernel(hyperparameters[1:2], dif_sq_vec[1]) + RBF_kernel(hyperparameters[3:4], dif_sq_vec[2])
    # final = periodic_kernel(hyperparameters[1:3], dif_vec[1]) + periodic_kernel(hyperparameters[4:6], dif_vec[2])

    # example of dependent multivariate kernel
    # final = RBF_kernel(hyperparameters[1:2], dif_sq_tot)

    #complicated somesuch
    # final = dRBF_kernel(append!([1.],[hyperparameters[length(hyperparameters)]]), dif_tot, dorder)
    # final = dRBF_kernel(hyperparameters[length(hyperparameters)], dif_tot, dorder)
    return final
end


# Creates the covariance matrix by evaluating the kernel function for each pair
# of passed inputs. Generic. Complex covariances accounted for in the
# efficient_covariance function
function covariance(x1list, x2list, hyperparameters; equal=false, dorder=[0,0], symmetric=false)
    K = zeros((size(x1list)[1], size(x2list)[1]))

    if equal
        kernline = zeros(size(x1list)[1])
        for i in 1:size(x1list)[1]
            kernline[i] = kernel(hyperparameters, x1list[1,:], x2list[i,:], dorder=dorder)
        end
        for i in 1:size(x1list)[1]
            for j in 1:size(x2list)[1]
                if i <= j
                    K[i, j] = kernline[j - i + 1]
                end
            end
        end
        return Symmetric(K)
    elseif symmetric
        for i in 1:size(x1list)[1]
            for j in 1:size(x2list)[1]
                if i <= j
                    K[i, j] = kernel(hyperparameters, x1list[i,:], x2list[j,:])
                end
            end
        end
        return Symmetric(K)
    else
        for i in 1:size(x1list)[1]
            for j in 1:size(x2list)[1]
                K[i, j] = kernel(hyperparameters, x1list[i,:], x2list[j,:])
            end
        end
        return K
    end
end


# calculating the covariance between all outputs for a combination of dependent GPs
# in this case, for outputs = a1 * GP[t] + a2 * GP'[t]
# written so that the intermediate K's don't have to be calculated over and over again
# multi output GP paper
# https://lra.le.ac.uk/bitstream/2381/31763/2/multi-output-gp-v4.pdf
function efficient_covariance(x1list, x2list, hyperparameters; equal=false, symmetric=false)

    # # comment everything excpet bottom line out if not using this specific function
    # point_amount = [size(x1list)[1], size(x2list)[1]]
    # K = zeros((n_out * point_amount[1], n_out * point_amount[2]))
    #
    # A = Array{Any}(n_dif, n_dif)
    # for k1 in 1:n_dif
    #     for k2 in 1:n_dif
    #         dorder = [k1 - 1, k2 - 1]
    #         A[k1, k2] = covariance(x1list, x2list, hyperparameters; equal=equal, dorder=dorder, symmetric=symmetric)
    #     end
    # end
    #
    # a = reshape(hyperparameters[1:n_out * n_dif], (n_out, n_dif))
    # for i in 1:n_out
    #     for j in 1:n_out
    #         for k in 1:n_dif
    #             for l in 1:n_dif
    #                 K[((i - 1) * point_amount[1] + 1):(i * point_amount[1]),
    #                     ((j - 1) * point_amount[2] + 1):(j * point_amount[2])] +=
    #                     a[i, k] * a[j, l] * A[k, l]
    #             end
    #         end
    #     end
    # end

    # comment this out when using the more complicated function
    K = covariance(x1list, x2list, hyperparameters; equal=false, dorder=[0,0], symmetric=false)
    return K
end

# kernel hyper parameters
# (AFFECTS SHAPE OF GP's, make sure you have the right amount!)
# a = [[1  0.5];[0.5  1]] / sqrt(50)
# kernel_length = [0.8]
# hyperparameters = append!(collect(Iterators.flatten(a)), kernel_length)
hyperparameters = [0.25, 0.3, 0.25, 0.5]


# how finely to sample the domain
GP_sample_amount = [30, 30]

# creating many inputs to sample the eventual gaussian process on
# how wide the measurement domain is
domain = [8, 8]

x1_samp = linspace(0, domain[1], GP_sample_amount[1])
x2_samp = linspace(0, domain[2], GP_sample_amount[2])

# # for 1D GPs
# x_samp = x1_samp

# for 2D GPs
# creating a list with all pairs of a uniformly sampled domain
# Pkg.add("IterTools")
using IterTools
x_samp = collect(product(x1_samp, x2_samp))
# converting the weird output into an array
x_samp = hcat([x_samp[i][1] for i in 1:length(x_samp)],[x_samp[i][2] for i in 1:length(x_samp)])
n_out = 1

# # size(a)[1] is how many outputs there will be
# n_out = size(a)[1]
# # size(a)[2] is how many differentiated versions of the GP there will be
# n_dif = size(a)[2]


# Finding how correlated the sampled inputs are to each other
# (aka getting the covariance matrix by evaluating the kernel function at all
# pairs of points)
K_samp = efficient_covariance(x_samp, x_samp, hyperparameters, equal=true)

# figure out how to plot images
plot(heatmap(z=K_samp, colorscale="Viridis"))

# noise to add to gaussians (to simulate observations)
GP_noise = .1

# how many GPs to plot
amount_of_GPs = 1

amount_of_samp_points = size(x_samp)[1] * n_out


#convert an Any type array to a trace array
function trace_list(a)
    traces = []
    for i in 1:size(a)[1]
        if i == 1
            traces = [a[1]]
        else
            append!(traces, [a[i]])
        end
    end

    return traces
end


# use for 1D GP plotting
# lines can't have opacity :(
function line_trace(x, y)
    return scatter(; x=x, y=y, mode="lines",
        line_width=1)
end


# use for 2D GP plotting
function surface_trace(x, y, z; opacity=1)
    return surface(z=reshape(z, (GP_sample_amount[1], GP_sample_amount[2])),
        x=reshape(x, (GP_sample_amount[1], GP_sample_amount[2])),
        y=reshape(y, (GP_sample_amount[1], GP_sample_amount[2])),
        showscale=false, opacity=opacity, colorscale="Viridis")
end


# generic trace collection wrapper function
function traces(coords...; opacity=1)

    x = coords[1]
    y = coords[2]

    all_traces = []
    if length(coords) > 2
        z = coords[3]
        if length(size(z)) > 1
            all_traces = []
            for i in 1:size(z)[1]
                append!(all_traces, [surface_trace(x, y, z[i,:], opacity=opacity)])
            end
            all_traces = trace_list(all_traces)
        else
            all_traces = surface_trace(x, y, z)
        end
    else
        if length(size(y)) > 1
            for i in 1:size(y)[1]
                append!(all_traces, [line_trace(x, y[i,:])])
            end
            all_traces = trace_list(traces)
        else
            all_traces = [line_trace(x, y)]
        end
    end

    return all_traces
end


# # this is broken for some reason
# function traces(coords...; opacity=1)
#
#     x = coords[1]
#     if length(coords) > 2
#         y = coords[2]
#         func(response) = surface_trace(x, y, response, opacity=opacity)
#     else
#         func(response) = line_trace(x, response)
#     end
#
#     all_traces = []
#     dims = length(coords)
#     responses = coords[dims]
#     if length(size(responses)) > 1
#         for i in 1:size(responses)[1]
#             append!(all_traces, [func(responses[i,:])])
#         end
#         all_traces = trace_list(all_traces)
#     else
#         all_traces = func(responses)
#     end
#
#     return all_traces
# end


# plotting amount_of_GPs randomly drawn Gaussian processes using the kernel
# function to correlate points
GP_funcs = zeros((amount_of_GPs, amount_of_samp_points))
GP_obs = zeros((amount_of_GPs, amount_of_samp_points))

# variables initialized in for loops or if statements are not saved
for i in 1:amount_of_GPs
    # sampled possible GP function values
    GP_func = K_samp * randn(amount_of_samp_points)
    GP_funcs[i, :] = GP_func
    # sampled possible GP observations (that have noise)
    GP_obs[i, :] = GP_func + GP_noise  *randn(amount_of_samp_points)
end

layout = Layout(; title="Gaussian Processes",
    xaxis=attr(title="x (time)",
        tickcolor="rgb(127, 127, 127)",
        ticks="outside"),
    yaxis=attr(title="y (flux or something lol)",
        tickcolor="rgb(127, 127, 127)",
        ticks="outside"),
    showlegend=false)

# # for 1D GPs
# plot(traces(x_samp,GP_funcs), layout)

# # 2 out stuff
# plot(traces(x_samp, GP_funcs[:, 1:size(x_samp)[1]]), layout)
# plot(traces(x_samp, GP_funcs[:, size(x_samp)[1] + 1:2 * size(x_samp)[1]]), layout)

# for 2D GPs
plot(traces(x_samp[:, 1], x_samp[:, 2], GP_funcs), layout)

# "true" underlying function for the fake observations
function observations(x, measurement_noise)
    # a phase shifted sine curve with noise
    if length(size(x)) > 1
        shift = 2 * pi * rand(size(x)[2])
        return [sum(sin.(pi / 2 * x[i,:] + shift)) for i in 1:size(x)[1]] + measurement_noise.^2 .* randn(size(x)[1])
    else
        shift = 2 * pi * rand()
        return [sum(sin.(pi / 2 * x[i,:] + shift)) for i in 1:length(x)] + measurement_noise.^2 .* randn(length(x))
    end
end


# creating observations to test methods on
amount_of_measurements = 36

# Uncertainty in the data (AFFECTS SPREAD OF DATA AND HOW TIGHTLY THE GP's WILL
# TRY TO HUG THE DATA) aka how much noise is added to measurements and
# measurement covariance function can be a single Float64 or a vector of
# length = amount_of_measurements
measurement_noise = 0.2 * ones(amount_of_measurements)

# x_obs = linspace(0,domain,amount_of_measurements)  # observation inputs
x_obs = domain[1] * rand((amount_of_measurements))

if length(domain) > 1
    for i in 2:length(domain)
        x_obs = hcat(x_obs, domain[i] * rand(amount_of_measurements))
    end
    x_obs = sortrows(x_obs, by=x->(x[1]))
else
    x_obs = sort(x_obs)
end
y_obs = observations(x_obs, measurement_noise)  # observation outputs
for i in 2:n_out
    y_obs = vcat(y_obs, observations(x_obs, measurement_noise))
end

# plotting some more randomly drawn Gaussian processes before data influences
# the posterior
draws = 500
storage = zeros((draws, amount_of_samp_points))
for i in 1:draws
    storage[i,:] = K_samp * randn(amount_of_samp_points)
end

# drawing lots of GPs (to get a good estimate of 5-95th percentile.
# Haven't found the analytical way to do it

# # for 1D GPs
#
#
# # quick and dirty function for creating plots that show what I want
# function custom_line_plot(x_samp, storage, x_obs, y_obs)
#     # only showing some of the curves
#     show = 10
#     show_curves = storage[1:show, :]
#     storage = sort(storage, 1)
#     # filling the 5-95th percentile with a transparent orange
#     upper = scatter(;x=x_samp, y=storage[convert(Int64,0.95 * draws),:], mode="lines", line_width=0)
#     lower = scatter(;x=x_samp, y=storage[convert(Int64,0.05 * draws),:], fill="tonexty", mode="lines", line_width=0)
#     median = scatter(;x=x_samp, y=storage[convert(Int64,0.5 * draws),:], mode="lines", line_width=4, line_color="rgb(0, 0, 0)")
#     data_trace = scatter(;x=x_obs, y=y_obs, mode="markers", marker_size=12, marker_color="rgb(0, 0, 0)")
#     plot(append([upper, lower, median], traces(x_samp, show_curves), [data_trace]), layout)
# end
#
#
# custom_line_plot(x_samp, storage[:, 1:convert(Int, amount_of_samp_points / n_out)],
#     x_obs, y_obs[1:amount_of_measurements])
# custom_line_plot(x_samp, storage[:, (convert(Int, amount_of_samp_points / n_out) + 1):amount_of_samp_points],
#     x_obs, y_obs[amount_of_measurements + 1:2 * amount_of_measurements])

# for 2D GPs


# another quick and dirty function for creating plots that show what I want
function custom_surface_plot(x_samp, storage, x_obs, y_obs)
    storage = sort(storage, 1)
    bounds = traces(x_samp[:, 1], x_samp[:, 2],
        transpose(hcat(storage[convert(Int64, 0.95 * draws),:],storage[convert(Int64, 0.05 * draws),:])),
        opacity=0.4)
    median = traces(x_samp[:, 1], x_samp[:, 2],
        storage[convert(Int64, 0.5 * draws),:], opacity=0.9)
    data_trace = scatter3d(;x=x_obs[:, 1], y=x_obs[:, 2], z=y_obs,
            mode="markers", opacity=1,
            marker_size=5, marker_color="rgba(0, 0, 0)")
    plot(append!(bounds, [median, data_trace]), layout)
end


custom_surface_plot(x_samp, storage, x_obs, y_obs)


function K_observations(x_obs, measurement_noise, hyperparameters)
    # creating additional covariance matrices (as defined on pages 13-15 of Eric
    # Schulz's tutorial)
    for i in 2:n_out
        measurement_noise = vcat(measurement_noise, measurement_noise)
    end
    amount_of_measurements = size(x_obs)[1]
    amount_of_outputs = amount_of_measurements * n_out
    noise_I = zeros((amount_of_outputs, amount_of_outputs))
    for i in 1:amount_of_outputs
        noise_I[i, i] =  measurement_noise[i] ^ 2
    end
    K_obs = efficient_covariance(x_obs, x_obs, hyperparameters, symmetric=true) + noise_I
    return K_obs
end


function GP_posteriors(x_obs, x_samp, measurement_noise, hyperparameters)

    K_samp = efficient_covariance(x_samp, x_samp, hyperparameters, equal=true)
    K_obs = K_observations(x_obs, measurement_noise, hyperparameters)
    K_samp_obs = efficient_covariance(x_samp, x_obs, hyperparameters)
    K_obs_samp = efficient_covariance(x_obs, x_samp, hyperparameters)

    # mean of the conditional distribution p(f_samp|x_obs,t_obs,x_samp) (as defined on page 14)
    mean_post = K_samp_obs * (inv(K_obs) * y_obs)
    # covariance matrix of the conditional distribution p(f_samp|x_obs,t_obs,x_samp) (as defined on page 14)
    K_post = K_samp - (K_samp_obs * (inv(K_obs) * K_obs_samp))
    return mean_post, K_post
end


mean_post, K_post = GP_posteriors(x_obs, x_samp, measurement_noise, hyperparameters);

# plot the posterior covariance matrix (colors show how correlated points are to each other)
# possible colors found here (https://plot.ly/julia/heatmaps/)
plot(heatmap(z=K_post, colorscale="Viridis"))

# plotting randomly drawn Gaussian processes with the mean function and covariance of the posterior
# much closer to the data, no?
storage = zeros((draws, amount_of_samp_points))
for i in 1:draws
    storage[i, :] = K_post * randn(amount_of_samp_points) + mean_post
end

# # for 1D GPs
# custom_line_plot(x_samp, storage[:, 1:convert(Int, amount_of_samp_points / n_out)],
#     x_obs, y_obs[1:amount_of_measurements])
# custom_line_plot(x_samp, storage[:, (convert(Int, amount_of_samp_points / n_out) + 1):amount_of_samp_points],
#     x_obs, y_obs[amount_of_measurements + 1:2 * amount_of_measurements])

# for 2D GPs
custom_surface_plot(x_samp, storage, x_obs, y_obs)

# negative log likelihood of the data given the current kernel parameters (as seen on page 19)
# (negative because scipy has a minimizer instead of a maximizer)
function nlogL(hyperparameter_list...)
    # println(typeof(hyperparameter_list))
    hyperparameters = []
    for i in 1:length(hyperparameter_list)
        append!(hyperparameters, hyperparameter_list[i])
    end
    n=length(y_obs)

    # a weirdly necessary dummy variable
    measurement_noise_dummy = measurement_noise
    K_obs = K_observations(x_obs, measurement_noise_dummy, hyperparameters)
    # goodness of fit term
    data_fit = -1 / 2 * (transpose(y_obs) * (inv(K_obs) * y_obs))
    # complexity penalization term
    penalty = -1 / 2 * log(det(K_obs))
    # normalization term (functionally useless)
    normalization = -n / 2 * log(2 * pi)
    return -1 * (data_fit + penalty + normalization)
end


# numerically maximize the likelihood to find the best hyperparameters

# # Pkg.add("JuMP")
# # Pkg.add("Ipopt")
# # Pkg.add("Mosek")  # "Unsupported feature Hess" error
#
# using JuMP, Ipopt
# m = Model(solver=IpoptSolver())
# @variable(m, x[i=1:length(hyperparameters)] >= 0, start = hyperparameters[i])
# # @variable(m, x[i=1:length(hyperparameters)] >= 0)
# JuMP.register(m, :nlogL, 5, nlogL, autodiff=true)
# @NLobjective(m, Min, nlogL(x[1],x[2],x[3],x[4],x[5])) # or Min
# print(m)
# solve(m)

# Pkg.add("Optim")
using Optim

# can't do anything more sophisticated without a gradient for a multivariate function apparently
# http://julianlsolvers.github.io/Optim.jl/stable/user/minimization/
result = optimize(nlogL, hyperparameters, BFGS())
hyperparameters=result.minimizer

print("'Best-fit' hyperparameters")
print(hyperparameters)

# reruning analysis of posterior with the "most likley" kernel amplitude and lengthscale

# recalculating posterior covariance and mean function
mean_post, K_post = GP_posteriors(x_obs, x_samp, measurement_noise, hyperparameters);

# plot the posterior covariance of the "most likely" posterior matrix
# (colors show how correlated points are to each other)
plot(heatmap(z=K_post, colorscale="Viridis"))

# plotting randomly drawn Gaussian processes with the mean function and covariance of the posterior
# much closer to the data, no?
storage = zeros((draws, amount_of_samp_points))
for i in 1:draws
    storage[i, :] = K_post * randn(amount_of_samp_points) + mean_post
end

# # for 1D GPs
# custom_line_plot(x_samp, storage[:, 1:convert(Int, amount_of_samp_points / n_out)],
#     x_obs, y_obs[1:amount_of_measurements])
# custom_line_plot(x_samp, storage[:, (convert(Int, amount_of_samp_points / n_out) + 1):amount_of_samp_points],
#     x_obs, y_obs[amount_of_measurements + 1:2 * amount_of_measurements])

# for 2D GPs
custom_surface_plot(x_samp, storage, x_obs, y_obs)




# a statistical exploration
test = 450

#prediction interval (for future draws)
# Pkg.add("Distributions")
using Distributions
a = fit_mle(Normal, storage[:,test])
println(a.σ)

#confidence (frequentists) / credible (Bayes)? interval (on mean) (1/2 width of smallest 68.3% credible interval, centered on mean)
println(sqrt.(diag(K_post)[test]))

plot(histogram(x=storage[:,test], opacity=0.75))
