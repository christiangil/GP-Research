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
    kernel_amplitude, kernel_length = hyperparameters
    return kernel_amplitude ^ 2 * exp(-dif_sq / (2 * kernel_length ^ 2))
end


#Ornstein–Uhlenbeck (Exponential) kernel
function OU_kernel(hyperparameters,dif)
    kernel_amplitude, kernel_length = hyperparameters
    return kernel_amplitude ^ 2 * exp(-dif / kernel_length)
end


# Periodic kernel
function periodic_kernel(hyperparameters, dif)
    kernel_amplitude, kernel_length, kernel_period = hyperparameters
    return kernel_amplitude ^ 2 * exp(-2 * sin(pi * dif / kernel_period) ^ 2 / (kernel_length ^ 2))
end


#Matern kernel (not sure if I implemented this correctly. Gets nans when dif==0)
function Matern_kernel(hyperparameters,dif,nu)
    kernel_amplitude, kernel_length = hyperparameters
    #limit of the function as it apporaches 0 (see https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function)
    if dif==0
        return kernel_amplitude ^ 2
    else
        return kernel_amplitude ^ 2 * ((2^ (1 - nu)) / (gamma(nu))) * ((sqrt(2 * nu) * dif) / kernel_length) ^ nu * besselk(nu, (sqrt(2 * nu) * dif) / kernel_length)
    end
end


#Rational Quadratic kernel (equivalent to adding together many SE kernels
#with different lengthscales. When α→∞, the RQ is identical to the SE.)
function RQ_kernel(hyperparameters,dif_sq)
    kernel_amplitude, kernel_length, alpha = hyperparameters
    alpha=max(alpha,0)
    return kernel_amplitude ^ 2 * (1 + dif_sq / (2 * alpha * kernel_length ^ 2)) ^ -alpha
end


# Creating a custom kernel (possibly by adding and multiplying other kernels?)
# x1 and x2 are single data points
function kernel(hyperparameters, x1, x2)
    # finding squared differences between inputs
    dif_vec = x1 - x2
    dif_sq_vec = dif_vec .^ 2  # element-wise squaring
    dif_sq_tot = sum(dif_sq_vec)
    dif_tot = sqrt(dif_sq_tot)

    # final = RBF_kernel(hyperparameters[1:2], dif_sq_tot)
    # final = OU_kernel(hyperparameters[1:2], dif_tot)
    # final = linear_kernel(hyperparameters[1:2], x1a ,x2a)
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

    return final
end

# Creates the covariance matrix by evaluating the kernel function for each pair
# of passed inputs
function covariance(x1list, x2list, hyperparameters)
    K = zeros((size(x1list)[1], size(x2list)[1]))
    for i in 1:size(x1list)[1]
        for j in 1:size(x2list)[1]
            K[i, j] = kernel(hyperparameters, x1list[i,:], x2list[j,:])
        end
    end
    return K
end

using SpecialFunctions

# kernel hyper parameters
# (AFFECTS SHAPE OF GP's, make sure you have the right amount!)
hyperparameters = [0.25, 0.3, 0.25, 0.5]
# hyperparameters = [0.25, 0.3, 5, 0.25, 0.5, 4]


# how finely to sample the domain
GP_sample_amount = [30, 30]

# creating many inputs to sample the eventual gaussian process on
domain = [8, 8]  # how wide the measurement domain is

x1_samp = linspace(0, domain[1], GP_sample_amount[1])
x2_samp = linspace(0, domain[2], GP_sample_amount[2])

# creating a list with all pairs of a uniformly sampled domain
# Pkg.add("IterTools")
using IterTools
x_samp = collect(product(x1_samp, x2_samp))

# converting the weird output into an array
x_samp = hcat([x_samp[i][1] for i in 1:length(x_samp)],[x_samp[i][2] for i in 1:length(x_samp)])

# Finding how correlated the sampled inputs are to each other
# (aka getting the covariance matrix by evaluating the kernel function at all
# pairs of points)
K_samp = covariance(x_samp, x_samp, hyperparameters);

# figure out how to plot images
plot(heatmap(z=K_samp, colorscale="Viridis"))

# noise to add to gaussians (to simulate observations)
GP_noise = .1

# how many GPs to plot
amount_of_GPs = 1

amount_of_samp_points = size(x_samp)[1]
# plotting amount_of_GPs randomly drawn Gaussian processes using the kernel
# function to correlate points
GP_funcs = zeros((amount_of_GPs, amount_of_samp_points))
GP_obs = zeros((amount_of_GPs, amount_of_samp_points))


#convert an Any type array to a trace array
function trace_list(a)
    traces = []
    for i in 1:size(a)[1]
        if i == 1
            traces = [a[1]]
        else
            append!(traces,[a[i]])
        end
    end

    return traces
end


# use for 1D GP plotting
# lines can't have opacity :(
function line_traces(x,y)


    function line_trace(x,y)
        return scatter(; x=x, y=y, mode="lines",
            line_width=1)
    end


    traces = []
    for i in 1:size(y)[1]
        append!(traces,[line_trace(x,y[i,:])])
    end

    traces = trace_list(traces)

    return traces
end


# use for 2D GP plotting
function surface_traces(x,y,z)

    function surface_trace(x,y,z)
        return surface(z=reshape(z,(GP_sample_amount[1],GP_sample_amount[2])), x=reshape(x,(GP_sample_amount[1],GP_sample_amount[2])), y=reshape(y,(GP_sample_amount[1],GP_sample_amount[2])),
            showscale=false, opacity=1, colorscale="Viridis")
    end


    traces = []
    for i in 1:size(z)[1]
        append!(traces,[surface_trace(x,y,z[i,:])])
    end

    traces = trace_list(traces)

    return traces
end

# variables initialized in for loops or if statements are not saved
for i in 1:amount_of_GPs
    # sampled possible GP function values
    GP_func = K_samp * randn(amount_of_samp_points)
    GP_funcs[i, :] = GP_func
    # sampled possible GP observations (that have noise)
    GP_obs[i, :] = GP_func + GP_noise*randn(amount_of_samp_points)
end

layout = Layout(; title="Gaussian Processes",
    xaxis=attr(title="x (time)",
        tickcolor="rgb(127, 127, 127)",
        ticks="outside"),
    yaxis=attr(title="y (flux or something lol)",
        tickcolor="rgb(127, 127, 127)",
        ticks="outside"),
    showlegend=false)

x = x_samp[:, 1];
y = x_samp[:, 2];
# plot(surface_traces(x, y, GP_funcs), layout)

# plot(line_traces(x_samp,GP_funcs), layout)


# "true" underlying function for the fake observations
function observations(x, measurement_noise)
    # a phase shifted sine curve with noise
    shift = 2 * pi * rand(size(x)[2])
    return [sum(sin.(pi / 2 * x[i,:] + shift)) for i in 1:size(x)[1]] + measurement_noise.^2 .* randn(size(x)[1])
end


# creating observations to test methods on
amount_of_measurements = 6^2

# Uncertainty in the data (AFFECTS SPREAD OF DATA AND HOW TIGHTLY THE GP's WILL
# TRY TO HUG THE DATA) aka how much noise is added to measurements and
# measurement covariance function can be a single Float64 or a vector of
# length = amount_of_measurements
measurement_noise = 0.2*ones(amount_of_measurements)

# x_obs = linspace(0,domain,amount_of_measurements)  # observation inputs
x_obs = domain[1]*rand((amount_of_measurements))
for i in 2:length(domain)
    x_obs = hcat(x_obs,domain[i]*rand((amount_of_measurements)))
end
y_obs = observations(x_obs,measurement_noise)  # observation outputs

# plotting some more randomly drawn Gaussian processes before data influences
# the posterior
# draws = 5000
# show = 20
draws = 500
storage = zeros((draws, amount_of_samp_points))


# # for 1D GPs
# # drawing lots of GPs (to get a good estimate of 5-95th percentile.
# # Haven't found the analytical way to do it
# for i in 1:draws
#     storage[i,:] = K_samp * randn(amount_of_samp_points)
# end
# # only showing some of the curves
# show_curves = storage[1:show,:]
# storage = sort(storage, 1)
# # filling the 5-95th percentile with a transparent orange
# upper = scatter(;x=x_samp, y=storage[convert(Int64,0.95*draws),:], mode="lines", line_width=0)
# lower = scatter(;x=x_samp, y=storage[convert(Int64,0.05*draws),:], fill="tonexty", mode="lines", line_width=0)
# median = scatter(;x=x_samp, y=storage[convert(Int64,0.5*draws),:], mode="lines", line_width=4, line_color="rgb(0, 0, 0)")
# data_trace = scatter(;x=x_obs, y=y_obs, mode="markers", marker_size=12, marker_color="rgb(0, 0, 0)")
# plot(append([median, upper, lower], line_traces(x_samp,show_curves),[data_trace]), layout)

# for 2D GPs
# drawing lots of GPs (to get a good estimate of 5-95th percentile.
# Haven't found the analytical way to do it
for i in 1:draws
    storage[i,:] = K_samp * randn(amount_of_samp_points)
end
storage = sort(storage, 1)

# bounding the 5-95th percentile with transparent surfaces
upper = surface(z=reshape(storage[convert(Int64,0.95*draws),:],(GP_sample_amount[1],GP_sample_amount[2])), x=reshape(x,(GP_sample_amount[1],GP_sample_amount[2])), y=reshape(y,(GP_sample_amount[1],GP_sample_amount[2])),
            showscale=false, opacity=0.4, colorscale="Viridis")
lower = surface(z=reshape(storage[convert(Int64,0.05*draws),:],(GP_sample_amount[1],GP_sample_amount[2])), x=reshape(x,(GP_sample_amount[1],GP_sample_amount[2])), y=reshape(y,(GP_sample_amount[1],GP_sample_amount[2])),
            showscale=false, opacity=0.4, colorscale="Viridis")
median = surface(z=reshape(storage[convert(Int64,0.5*draws),:],(GP_sample_amount[1],GP_sample_amount[2])), x=reshape(x,(GP_sample_amount[1],GP_sample_amount[2])), y=reshape(y,(GP_sample_amount[1],GP_sample_amount[2])),
            showscale=false, opacity=0.9, colorscale="Viridis")
data_trace = scatter3d(;x=x_obs[:, 1], y=x_obs[:, 2], z=y_obs,
        mode="markers", opacity=1,
        marker_size=5, marker_color="rgba(0, 0, 0)")
# plot(append([median, upper, lower, data_trace]), layout)


function GP_posteriors(x_obs, x_samp, measurement_noise, hyperparameters)

    K_samp=covariance(x_samp,x_samp,hyperparameters)

    # creating additional covariance matrices (as defined on pages 13-15 of Eric
    # Schulz's tutorial)
    amount_of_measurements = size(x_obs)[1]
    noise_I = zeros((amount_of_measurements, amount_of_measurements))
    for i in 1:amount_of_measurements
        noise_I[i,i] =  measurement_noise[i] ^ 2
    end
    K_obs = covariance(x_obs, x_obs, hyperparameters) + noise_I
    K_samp_obs = covariance(x_samp, x_obs, hyperparameters)
    K_obs_samp = covariance(x_obs, x_samp, hyperparameters)

    # mean of the conditional distribution p(f_samp|x_obs,t_obs,x_samp) (as defined on page 14)
    mean_post=K_samp_obs*(inv(K_obs)*y_obs)
    # covariance matrix of the conditional distribution p(f_samp|x_obs,t_obs,x_samp) (as defined on page 14)
    K_post=K_samp-(K_samp_obs*(inv(K_obs)*K_obs_samp))
    return mean_post, K_post
end


mean_post, K_post = GP_posteriors(x_obs, x_samp, measurement_noise, hyperparameters);

# plot the posterior covariance matrix (colors show how correlated points are to each other)
# possible colors found here (https://plot.ly/julia/heatmaps/)
# plot(heatmap(z=K_post, colorscale="Electric"))

# plotting randomly drawn Gaussian processes with the mean function and covariance of the posterior
# much closer to the data, no?

storage = zeros((draws, amount_of_samp_points))

# # for 1D GPs
# for i in 1:draws
#     storage[i,:] = K_post * randn(GP_sample_amount) + mean_post
# end
# # only showing some of the curves
# show_curves = storage[1:show,:]
# storage = sort(storage, 1)
# # filling the 5-95th percentile prediction interval with a transparent orange
# upper = scatter(;x=x_samp, y=storage[convert(Int64,0.95*draws),:], mode="lines", line_width=0)
# lower = scatter(;x=x_samp, y=storage[convert(Int64,0.05*draws),:], fill="tonexty", mode="lines", line_width=0)
# data_trace = scatter(;x=x_obs, y=y_obs, mode="markers", marker_size=12, marker_color="rgb(0, 0, 0)")
# mean_func_trace = scatter(;x=x_samp, y=mean_post, mode="line", line_size=5, line_color="rgb(0, 0, 0)")
# plot(append([upper, lower], line_traces(x_samp,show_curves),[mean_func_trace, data_trace]), layout)


# for 2D GPs
for i in 1:draws
    storage[i,:] = K_samp * randn(amount_of_samp_points) + mean_post
end
storage = sort(storage, 1)
# bounding the 5-95th percentile prediction interval with transparent surfaces
upper = surface(z=reshape(storage[convert(Int64,0.95*draws),:],(GP_sample_amount[1],GP_sample_amount[2])), x=reshape(x,(GP_sample_amount[1],GP_sample_amount[2])), y=reshape(y,(GP_sample_amount[1],GP_sample_amount[2])),
            showscale=false, opacity=0.4, colorscale="Viridis")
lower = surface(z=reshape(storage[convert(Int64,0.05*draws),:],(GP_sample_amount[1],GP_sample_amount[2])), x=reshape(x,(GP_sample_amount[1],GP_sample_amount[2])), y=reshape(y,(GP_sample_amount[1],GP_sample_amount[2])),
            showscale=false, opacity=0.4, colorscale="Viridis")
mean_func_trace = surface(z=reshape(mean_post,(GP_sample_amount[1],GP_sample_amount[2])), x=reshape(x,(GP_sample_amount[1],GP_sample_amount[2])), y=reshape(y,(GP_sample_amount[1],GP_sample_amount[2])),
            showscale=false, opacity=0.9, colorscale="Viridis")
data_trace = scatter3d(;x=x_obs[:, 1], y=x_obs[:, 2], z=y_obs,
        mode="markers", opacity=1,
        marker_size=5, marker_color="rgba(0, 0, 0)")
plot(append([mean_func_trace, upper, lower, data_trace]), layout)

# negative log likelihood of the data given the current kernel parameters (as seen on page 19)
# negative log likelihood of the data given the current kernel parameters (as seen on page 19)
# (negative because scipy has a minimizer instead of a maximizer)
function nlogL(hyperparameter_list...)
    hyperparameters=[]
    for i in 1:length(hyperparameter_list)
        append!(hyperparameters,hyperparameter_list[i])
    end
    n=length(y_obs)

    noise_I = zeros((amount_of_measurements, amount_of_measurements))
    for i in 1:amount_of_measurements
        noise_I[i,i] =  measurement_noise[i] ^ 2
    end
    # println(hyperparameters)
    # println(length(x_obs))
    K_obs=covariance(x_obs,x_obs,hyperparameters)+noise_I

    # goodness of fit term
    data_fit = -1 / 2 * (transpose(y_obs) * (inv(K_obs) * y_obs))
    # complexity penalization term
    penalty = -1 / 2 * log(det(K_obs))
    # normalization term (functionally useless)
    normalization = -n / 2 * log(2 * pi)
    return -1 * (data_fit + penalty + normalization)
end

# # Pkg.add("JuMP")
# # Pkg.build("Mosek")
# using JuMP, Mosek

# spook(x,y) = (x-1)^2+(y-2)^2

# m = Model(solver = MosekSolver())

# JuMP.register(m, :spook, 2, spook, autodiff=true)
# @variable(m, x[1:2] >= 0)
# @NLobjective(m, Min, (x[1]-1)^2+(x[2]-2)^2)

# println(m)

# status = solve(m)
# println()

# println("Objective value: ", getobjectivevalue(m))
# println("x = ", getvalue(x))

# trying to use JuMP. Doesn't work with custom functions "No method Hess error"

# Pkg.add("JuMP")
# Pkg.build("Mosek")
# using JuMP, Mosek

# spook(x,y) = (x-1)^2+(y-2)^2

# m = Model(solver = MosekSolver())

# JuMP.register(m, :spook, 2, spook, autodiff=true)
# @variable(m, x[1:2] >= 0)
# @NLobjective(m, Min, spook(x[1],x[2]))
# print(m)

# status = solve(m)

# println("Objective value: ", getobjectivevalue(m))
# println("x = ", getvalue(x))

# Pkg.add("Optim")
using Optim

# lower = [0, 0]
# upper = [Inf, Inf]
# initial_x = hyperparameters
# inner_optimizer = GradientDescent()
# results = optimize(f, g!, lower, upper, initial_x, Fminbox(inner_optimizer))

# numerically maximize the likelihood to find the best hyperparameters

# can't do anything more sophisticated without a gradient for a multivariate function apparently
# http://julianlsolvers.github.io/Optim.jl/stable/user/minimization/
result = optimize(nlogL, hyperparameters, BFGS())

# reruning analysis of posterior with the "most likley" kernel amplitude and lengthscale
hyperparameters=result.minimizer

print("'Best-fit' hyperparameters")
print(hyperparameters)

# recalculating posterior covariance and mean function
mean_post, K_post = GP_posteriors(x_obs, x_samp, measurement_noise, hyperparameters);

# plot the posterior covariance of the "most likely" posterior matrix
# (colors show how correlated points are to each other)
# plot(heatmap(z=K_post, colorscale="Viridis"))

# plotting randomly drawn Gaussian processes with the mean function and covariance of the posterior
# much closer to the data, no?
storage=zeros((draws,amount_of_samp_points))

# # for 1D GPs
# for i in 1:draws
#     storage[i,:] = K_post * randn(GP_sample_amount) + mean_post
# end
# # only showing some of the curves
# show_curves = storage[1:show,:]
# storage = sort(storage, 1)
# # filling the 5-95th percentile prediction interval with a transparent orange
# upper = scatter(;x=x_samp, y=storage[convert(Int64,0.95*draws),:], mode="lines", line_width=0)
# lower = scatter(;x=x_samp, y=storage[convert(Int64,0.05*draws),:], fill="tonexty", mode="lines", line_width=0)
# data_trace = scatter(;x=x_obs, y=y_obs, mode="markers", marker_size=12, marker_color="rgb(0, 0, 0)")
# mean_func_trace = scatter(;x=x_samp, y=mean_post, mode="line", line_size=5, line_color="rgb(0, 0, 0)")
# plot(append([upper, lower], line_traces(x_samp, show_curves), [mean_func_trace, data_trace]), layout)


# for 2D GPs
for i in 1:draws
    storage[i,:] = K_samp * randn(amount_of_samp_points) + mean_post
end
storage = sort(storage, 1)
# bounding the 5-95th percentile prediction interval with transparent surfaces
upper = surface(z=reshape(storage[convert(Int64,0.95*draws),:],(GP_sample_amount[1],GP_sample_amount[2])), x=reshape(x,(GP_sample_amount[1],GP_sample_amount[2])), y=reshape(y,(GP_sample_amount[1],GP_sample_amount[2])),
            showscale=false, opacity=0.4, colorscale="Viridis")
lower = surface(z=reshape(storage[convert(Int64,0.05*draws),:],(GP_sample_amount[1],GP_sample_amount[2])), x=reshape(x,(GP_sample_amount[1],GP_sample_amount[2])), y=reshape(y,(GP_sample_amount[1],GP_sample_amount[2])),
            showscale=false, opacity=0.4, colorscale="Viridis")
mean_func_trace = surface(z=reshape(mean_post,(GP_sample_amount[1],GP_sample_amount[2])), x=reshape(x,(GP_sample_amount[1],GP_sample_amount[2])), y=reshape(y,(GP_sample_amount[1],GP_sample_amount[2])),
            showscale=false, opacity=0.9, colorscale="Viridis")
data_trace = scatter3d(;x=x_obs[:, 1], y=x_obs[:, 2], z=y_obs,
        mode="markers", opacity=1,
        marker_size=5, marker_color="rgba(0, 0, 0)")
plot(append([mean_func_trace, upper, lower, data_trace]), layout)





test = 450

#prediction interval (for future draws)
# Pkg.add("Distributions")
using Distributions
a = fit_mle(Normal, storage[:,test])
println(a.σ)

#confidence (frequentists) / credible (Bayes)? interval (on mean) (1/2 width of smallest 68.3% credible interval, centered on mean)
println(sqrt.(diag(K_post)[test]))

plot(histogram(x=storage[:,test], opacity=0.75))
