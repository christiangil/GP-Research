# importing packages
# Pkg.add("Plots")
# using Plots
# Pkg.add("SpecialFunctions")
using SpecialFunctions
# Pkg.add("PlotlyJS")
using PlotlyJS


function append(a, b...)
    for i in 1:length(b)
        append!(a,b[i])
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
function kernel(hyperparameters, x1, x2)
    # finding squared differences between inputs
    dif_vec = x1 - x2
    dif_sq = dot(dif_vec, dif_vec)
    dif = sqrt(dif_sq)

    # final = RBF_kernel(hyperparameters, dif_sq)
    # final = OU_kernel(hyperparameters, dif)
    # final = linear_kernel(hyperparameters, x1a ,x2a)
    # final = periodic_kernel(hyperparameters, dif)
    # final = RQ_kernel(hyperparameters, dif_sq)
    final = Matern_kernel(hyperparameters, dif, 3/2)  # doesn't seem to be working right now

    # example of adding kernels
    # final = periodic_kernel(hyperparameters[0:3], dif) + RBF_kernel(hyperparameters[3:5], dif_sq)

    # example of multiplying kernels
    # final = periodic_kernel(hyperparameters[0:3], dif) * RBF_kernel(hyperparameters[3:5], dif_sq)

    return final
end

# Creates the covariance matrix by evaluating the kernel function for each pair
# of passed inputs
function covariance(x1list, x2list, hyperparameters)
    K = zeros((length(x1list), length(x2list)))
    for (i, x1) in enumerate(x1list)
        for (j, x2) in enumerate(x2list)
            K[i, j] = kernel(hyperparameters, x1, x2)
        end
    end
    return K
end

# kernel hyper parameters
# (AFFECTS SHAPE OF GP's, make sure you have the right amount!)
# hyperparameters = [0.8, 0.3, 2 ,0.5, .3]
hyperparameters = [0.25, 0.3]

# how finely to sample the domain
GP_sample_amount = 500 + 1

# creating many inputs to sample the eventual gaussian process on
domain = 8 # how wide the measurement domain is
x_samp=linspace(0, domain, GP_sample_amount)

# Finding how correlated the sampled inputs are to each other
# (aka getting the covariance matrix by evaluating the kernel function at all
# pairs of points)
K_samp=covariance(x_samp, x_samp,hyperparameters)

# figure out how to plot images
plot(heatmap(z=K_samp, colorscale="Electric"))

# noise to add to gaussians (to simulate observations)
GP_noise = .1

# how many GPs to plot
amount_of_GPs = 10

# plotting amount_of_GPs randomly drawn Gaussian processes using the kernel
# function to correlate points
GP_funcs = zeros((amount_of_GPs, length(x_samp)))
GP_obs = zeros((amount_of_GPs, length(x_samp)))


# lines can't have opacity :(
function line_traces(x,y)


    function line_trace(x,y)
        return scatter(; x=x, y=y, mode="lines",
            line_width=1)
    end


    traces = []
    for i in 1:size(y)[1]
        if i == 1
            traces = [line_trace(x,y[i,:])]
        else
            append!(traces,[line_trace(x,y[i,:])])
        end
    end

    return traces
end


# variables initialized in for loops or if statements are not saved
for i in 1:amount_of_GPs
    # sampled possible GP function values
    GP_func = K_samp * randn(GP_sample_amount)
    GP_funcs[i, :] = GP_func
    # sampled possible GP observations (that have noise)
    GP_obs[i, :] = GP_func + GP_noise*randn(GP_sample_amount)
end

layout = Layout(; title="Gaussian Processes",
    xaxis=attr(title="x (time)",
        tickcolor="rgb(127, 127, 127)",
        ticks="outside"),
    yaxis=attr(title="y (flux or something lol)",
        tickcolor="rgb(127, 127, 127)",
        ticks="outside"),
    showlegend=false)

plot(line_traces(x_samp,GP_funcs), layout)
plot(line_traces(x_samp,GP_obs), layout)


# "true" underlying function for the fake observations
function observations(x,measurement_noise)
    # a phase shifted sine curve with noise
    shift = 2 * pi * rand()
    return sin(pi / 2 * x + shift) + measurement_noise.^2 .* randn(length(x))
end


# creating observations to test methods on
amount_of_measurements = 25

# Uncertainty in the data (AFFECTS SPREAD OF DATA AND HOW TIGHTLY THE GP's WILL
# TRY TO HUG THE DATA) aka how much noise is added to measurements and
# measurement covariance function can be a single Float64 or a vector of
# length = amount_of_measurements
measurement_noise = 0.2*ones(amount_of_measurements)

x_obs = linspace(0,domain,amount_of_measurements)  # observation inputs
y_obs = observations(x_obs,measurement_noise)  # observation outputs

# plotting some more randomly drawn Gaussian processes before data influences
# the posterior
draws = 5000
show = 20
storage = zeros((draws, length(x_samp)))

# drawing lots of GPs (to get a good estimate of 5-95th percentile.
# Haven't found the analytical way to do it
for i in 1:draws
    storage[i,:] = K_samp * randn(GP_sample_amount)
end
# only showing some of the curves
show_curves = storage[1:show,:]
storage = sort(storage, 1)
# filling the 5-95th percentile with a transparent orange
upper = scatter(;x=x_samp, y=storage[convert(Int64,0.95*draws),:], mode="lines", line_width=0)
lower = scatter(;x=x_samp, y=storage[convert(Int64,0.05*draws),:], fill="tonexty", mode="lines", line_width=0)
plot(append!([upper, lower], line_traces(x_samp,show_curves)), layout)


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


mean_post, K_post = GP_posteriors(x_obs, x_samp, measurement_noise, hyperparameters)

# plot the posterior covariance matrix (colors show how correlated points are to each other)
# still dont know how to do this
# possible colors found here (https://plot.ly/julia/heatmaps/)
plot(heatmap(z=K_post, colorscale="Electric"))

# plotting randomly drawn Gaussian processes with the mean function and covariance of the posterior
# much closer to the data, no?
storage=zeros((draws,GP_sample_amount))
for i in 1:draws
    storage[i,:] = K_post * randn(GP_sample_amount) + mean_post
end
# only showing some of the curves
show_curves = storage[1:show,:]
storage = sort(storage, 1)
# filling the 5-95th percentile with a transparent orange
upper = scatter(;x=x_samp, y=storage[convert(Int64,0.95*draws),:], mode="lines", line_width=0)
lower = scatter(;x=x_samp, y=storage[convert(Int64,0.05*draws),:], fill="tonexty", mode="lines", line_width=0)
data_trace = scatter(;x=x_obs, y=y_obs, mode="markers", marker_size=12, marker_color="rgb(0, 0, 0)")
mean_func_trace = scatter(;x=x_samp, y=mean_post, mode="line", line_size=5, line_color="rgb(0, 0, 0)")
plot(append([upper, lower], line_traces(x_samp,show_curves),[mean_func_trace, data_trace]), layout)


# negative log likelihood of the data given the current kernel parameters (as seen on page 19)
# (negative because scipy has a minimizer instead of a maximizer)
function nlogL(hyperparameters)
    n=length(y_obs)

    noise_I = zeros((amount_of_measurements, amount_of_measurements))
    for i in 1:amount_of_measurements
        noise_I[i,i] =  measurement_noise[i] ^ 2
    end
    println(hyperparameters)
    println(length(x_obs))
    K_obs=covariance(x_obs,x_obs,hyperparameters)+noise_I

    # goodness of fit term
    data_fit = -1 / 2 * (transpose(y_obs) * (inv(K_obs) * y_obs))
    # complexity penalization term
    penalty = -1 / 2 * log(det(K_obs))
    # normalization term (functionally useless)
    normalization = -n / 2 * log(2 * pi)
    return -1 * (data_fit + penalty + normalization)
end


Pkg.add("Optim")
using Optim

# numerically maximize the likelihood to find the best hyperparameters
result = optimize(nlogL, hyperparameters, BFGS())

# reruning analysis of posterior with the "most likley" kernel amplitude and lengthscale
# hyperparameters=result.minimizer

Pkg.add("JuMP")
Pkg.add("Clp")
using Compat
using JuMP, Clp

m = Model(solver = ClpSolver())

@variable(m, 0 <= x <= 2)
@variable(m, 0 <= y <= 30)

@objective(m, Max, 5x + 3y)
@constraint(m, 1x + 5y <= 3.0)

print(m)

status = solve(m)

println("Objective value: ", getobjectivevalue(m))
println("x = ", getvalue(x))
println("y = ", getvalue(y))



print("'Best-fit' hyperparameters")
print(hyperparameters)

# recalculating posterior covariance and mean function
mean_post, K_post = GP_posteriors(x_obs, x_samp, measurement_noise, hyperparameters)

# plot the posterior covariance of the "most likely" posterior matrix
# (colors show how correlated points are to each other)
plot(heatmap(z=K_post, colorscale="Electric"))

# plotting randomly drawn Gaussian processes with the mean function and covariance of the posterior
# much closer to the data, no?
storage=zeros((draws,GP_sample_amount))
for i in 1:draws
    storage[i,:] = K_post * randn(GP_sample_amount) + mean_post
end
# only showing some of the curves
show_curves = storage[1:show,:]
storage = sort(storage, 1)
# filling the 5-95th percentile with a transparent orange
upper = scatter(;x=x_samp, y=storage[convert(Int64,0.95*draws),:], mode="lines", line_width=0)
lower = scatter(;x=x_samp, y=storage[convert(Int64,0.05*draws),:], fill="tonexty", mode="lines", line_width=0)
data_trace = scatter(;x=x_obs, y=y_obs, mode="markers", marker_size=12, marker_color="rgb(0, 0, 0)")
mean_func_trace = scatter(;x=x_samp, y=mean_post, mode="line", line_size=5, line_color="rgb(0, 0, 0)")
plot(append([upper, lower], line_traces(x_samp, show_curves), [mean_func_trace, data_trace]), layout)
