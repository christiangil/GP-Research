#adding in custom functions
include("all_functions.jl")

# can use this if you want to replicate results
# srand(1234)

# Creating a custom kernel (possibly by adding and multiplying other kernels?)
# x1 and x2 are single data points
function kernel(hyperparameters, x1, x2; dorder=[0,0], dKdθ=0, products = 0)

    # finding required differences between inputs
    dif_vec = x1 - x2  # raw difference vectors
    # abs_dif_vec = abs(dif_vec)  # absolute value of difference vectors
    # dif_sq_vec = dif_vec .^ 2  # element-wise squaring
    # dif_sq_tot = sum(dif_sq_vec)  # sum of squared distances
    # dif_tot = sqrt(dif_sq_tot)  # separation

    # a bunch of different kernels can be used
    # final = RBF_kernel(hyperparameters[1:2], dif_sq_tot)
    # final = RBF_kernel(append!([1],hyperparameters[1]), dif_sq_tot) #constant amplitude RBF
    # final = OU_kernel(hyperparameters[1:2], dif_tot)
    # final = linear_kernel(hyperparameters[1:2], x1a, x2a)
    # final = Periodic_kernel(hyperparameters[1:3], dif_tot)
    # final = RQ_kernel(hyperparameters[1:3], dif_sq_tot)
    # final = Matern_kernel(hyperparameters[1:2], dif_tot, 3/2)
    # final = Matern32_kernel(hyperparameters[1:2], dif_tot)
    # final = Matern52_kernel(hyperparameters[1:2], dif_tot)

    # example of adding independent kernels
    # final = Periodic_kernel(hyperparameters[1:3], dif_tot) + RBF_kernel(hyperparameters[4:5], dif_sq_tot)

    # example of multiplying independent kernels
    # final = Periodic_kernel(hyperparameters[1:3], dif_tot) * RBF_kernel(hyperparameters[4:5], dif_sq_tot)

    # examples of multivariate kernels
    # final = RBF_kernel(hyperparameters[1:2], dif_sq_vec[1]) + RBF_kernel(hyperparameters[3:4], dif_sq_vec[2])
    # final = Periodic_kernel(hyperparameters[1:3], abs_dif_vec[1]) + Periodic_kernel(hyperparameters[4:6], abs_dif_vec[2])
    # final = RBF_kernel(hyperparameters[1:2], dif_sq_tot)

    #complicated somesuch (my custom kernel that can return differentiated versions of itself)


    # this function mostly exists so I have to write fewer characters lol

    #
    function kernel_piece(hyper, dif, products_line)

        # amount of possible derivatives on each function
        dorders = length(hyper) + length(dorder)

        # get the derivative orders for functions 1 and 2
        dorder1 = floats2ints(products_line[2:(dorders+1)]; allow_negatives=false)
        dorder2 = floats2ints(products_line[(dorders + 2):(2 * dorders+1)]; allow_negatives=false)

        # return 0 if you know that that portion will equal 0
        # this is when you are deriving one of the kernels by a hyperparameter
        # of the other kernel
        if (((dorder1[length(dorder) + 2] == 1) | (dorder1[length(dorder) + 3] == 1))
            | (dorder2[length(dorder) + 1] == 1))

            return 0

        else

            # use the properly differentiated version of kernel function 1
            if dorder1[length(dorder) + 1] == 1
                func1 = dRBFdλ_kernel(hyper[1], dif, dorder1[1:length(dorder)])
            # elseif ((dorder1[length(dorder) + 2] == 1) | (dorder1[length(dorder) + 3] == 1))
            #     func1 = 0
            else
                func1 = dRBFdt_kernel(hyper[1], dif, dorder1[1:length(dorder)])
            end

            # use the properly differentiated version of kernel function 2
            # if dorder2[length(dorder) + 1] == 1
            #     func2 = 0
            if dorder2[length(dorder) + 2] == 1
                func2 = dPeriodicdλ_kernel(hyper[2:3], dif, dorder2[1:length(dorder)])
            elseif dorder2[length(dorder) + 3] == 1
                func2 = dPeriodicdp_kernel(hyper[2:3], dif, dorder2[1:length(dorder)])
            else
                func2 = dPeriodicdt_kernel(hyper[2:3], dif, dorder2[1:length(dorder)])
            end

            return func1 * func2

        end
    end

    # calculate the product rule coefficients and dorders if they aren't passed
    if products == 0
        non_coefficient_hyperparameters = length(hyperparameters) - total_coefficients
        new_dorder = append!(copy(dorder), zeros(non_coefficient_hyperparameters))
        # differentiate by RBF kernel length
        if dKdθ == length(hyperparameters) - 2
            new_dorder[length(dorder) + 1] = 1
        # differentiate by Periodic kernel length
        elseif dKdθ == length(hyperparameters) - 1
            new_dorder[length(dorder) + 2] = 1
        # differentiate by Periodic kernel period
        elseif dKdθ == length(hyperparameters)
            new_dorder[length(dorder) + 3] = 1
        end
        products = product_rule(new_dorder)
    end

    # add all of the differentiated kernels together according to the product rule
    final = sum([products[i, 1] * kernel_piece(hyperparameters[(length(hyperparameters) - 2):length(hyperparameters)], dif_vec[1], products[i,:]) for i in 1:size(products, 1)])

    return final

end


# calculating the covariance between all outputs for a combination of dependent GPs
# written so that the intermediate K's don't have to be calculated over and over again
function total_covariance(x1list, x2list, hyperparameters; dKdθ=0, coeff_orders=0)

    # calculating the total size of the multi-output covariance matrix
    point_amount = [size(x1list, 1), size(x2list, 1)]
    K = zeros((n_out * point_amount[1], n_out * point_amount[2]))

    non_coefficient_hyperparameters = length(hyperparameters) - total_coefficients

    # calculating all of the sub-matrices explicitly
    # A = Array{Any}(n_dif, n_dif)
    # for k in 1:n_dif
    #     for l in 1:n_dif
    #         dorder = [k - 1, l - 1]
    #         # things that have been differentiated an even amount of times are symmetric
    #         if iseven(k + l)
    #             A[k, l] = covariance(x1list, x2list, hyperparameters; dorder=dorder, symmetric=true, dKdθ=dKdθ)
    #         else
    #             A[k, l] = covariance(x1list, x2list, hyperparameters; dorder=dorder, dKdθ=dKdθ)
    #         end
    #     end
    # end
    #
    # save_A(A)

    # only calculating each sub-matrix once and using the fact that they should
    # be basically the same if the kernel has been differentiated the same amount of times
    A_list = Array{Any}(nothing, 2 * n_dif - 1)
    for k in 1:(2 * n_dif - 1)
        dorder = [0, k - 1]

        new_dorder = append!(copy(dorder), zeros(non_coefficient_hyperparameters))
        # differentiate by RBF kernel length
        if dKdθ == length(hyperparameters) - 2
            new_dorder[length(dorder) + 1] = 1
        # differentiate by Periodic kernel length
        elseif dKdθ == length(hyperparameters) - 1
            new_dorder[length(dorder) + 2] = 1
        # differentiate by Periodic kernel period
        elseif dKdθ == length(hyperparameters)
            new_dorder[length(dorder) + 3] = 1
        end
        products = product_rule(new_dorder)


        # things that have been differentiated an even amount of times are symmetric
        if isodd(k)
            A_list[k] = covariance(x1list, x2list, hyperparameters; dorder=dorder, symmetric=true, dKdθ=dKdθ, products=products)
        else
            A_list[k] = covariance(x1list, x2list, hyperparameters; dorder=dorder, dKdθ=dKdθ, products=products)
        end
    end


    # return the properly negative differentiated A matrix from the list
    # make it negative or not based on how many times it has been differentiated in the x1 direction
    A_mat(k, l, A_list) = ((-1) ^ (k - 1)) * A_list[k + l - 1]


    # reshaping the list into a format consistent with the explicit calculation
    # A = Array{Any}(n_dif, n_dif)
    # for k in 1:n_dif
    #     for l in 1:n_dif
    #         A[k, l] =  A_mat(k, l, A_list)
    #     end
    # end
    #
    # save_A(A)
    # save_A(A_list)

    # assembling the total covariance matrix
    a = reshape(hyperparameters[1:total_coefficients], (n_out, n_dif))
    # if we aren't differentiating by one of the coefficient hyperparameters
    # assemble the covariance matrix in the expected way
    if dKdθ == 0 || dKdθ > total_coefficients
        for i in 1:n_out
            for j in 1:n_out
                for k in 1:n_dif
                    for l in 1:n_dif
                        if (i == j) & isodd(k + l)
                            # the cross terms (of odd differentiation orders) cancel each other out in diagonal matrices
                        else
                            K[((i - 1) * point_amount[1] + 1):(i * point_amount[1]),
                                ((j - 1) * point_amount[2] + 1):(j * point_amount[2])] +=
                                # a[i, k] * a[j, l] * A[k, l]
                                a[i, k] * a[j, l] *  A_mat(k, l, A_list)
                        end
                    end
                end
            end
        end
    # if we are differentiating by one of the coefficient hyperparameters
    # we have to assemble the covariance matrix in a different way
    else
        # ((output pair), (which A matrix to use), (which a coefficent to use))
        # get all of the coefficients for coefficient hyperparameters based on
        # the amount of outputs and differentiations
        coeff = dif_coefficients(n_out, n_dif, dKdθ; coeff_orders=coeff_orders, a=a)

        for i in 1:n_out
            for j in 1:n_out
                for k in 1:n_dif
                    for l in 1:n_dif
                        for m in 1:n_out
                            for n in 1:n_dif
                                if coeff[i, j, k, l, m, n] != 0
                                    K[((i - 1) * point_amount[1] + 1):(i * point_amount[1]),
                                        ((j - 1) * point_amount[2] + 1):(j * point_amount[2])] +=
                                        # coeff[i, j, k, l, m, n] * a[m, n] * A[k, l]
                                        coeff[i, j, k, l, m, n] * a[m, n] * A_mat(k, l, A_list)
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    # this would just be a wrapper function for a less complicated kernel
    # all you would need is the following line
    # K = covariance(x1list, x2list, hyperparameters)

    # return the symmetrized version of the covariance matrix
    # function corrects for numerical errors and notifies us if our matrix isn't
    # symmetric like it should be
    return symmetric_A(K)

end

# kernel hyper parameters
# AFFECTS SHAPE OF THE GP's
# make sure you have the right amount!


# loading in data
using JLD2, FileIO
@load "sunspot_data.jld2" lambda phases quiet
@load "rv_data.jld2" doppler_comp genpca_out rvs_out
mu, M, scores = genpca_out
scores[:, 1] = rvs_out
scores = scores'

# for i in 1:3
#     figure(figsize=(10,6))
#     fig = plot(phases, scores[i, :])
#     xlabel("phases (days?)")
#     ylabel("pca scores")
#     title("PCA " * string(i - 1))
#     savefig("figs/pca/pca_score_" * string(i - 1) * ".pdf")
# end

# how many components you will use
n_out = 3
# how many differentiated versions of the original GP you will use
n_dif = 2

total_coefficients = n_out * n_dif

# Setting up all of the data things
# how much of the data you want to use (on time domain)
start_ind = 900
end_ind = 940  # 1070
amount_of_measurements = end_ind - start_ind + 1
total_amount_of_measurements = amount_of_measurements * n_out

# getting proper slice of data
x_obs = phases[start_ind:end_ind]
y_obs_hold = scores[1:n_out, start_ind:end_ind]

# for i in 1:3
#     figure(figsize=(10,6))
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

# Uncertainty in the data (AFFECTS SPREAD OF DATA AND HOW TIGHTLY THE GP's WILL
# TRY TO HUG THE DATA) aka how much noise is added to measurements and
# measurement covariance function
# a vector of length = total_amount_of_measurements
measurement_noise = ones(total_amount_of_measurements)
for i in 1:n_out
    measurement_noise[((i - 1) * amount_of_measurements + 1):(i * amount_of_measurements)] *= 0.05 * maximum(abs.(y_obs_hold[i, :]))
end

# kernel hyper parameters
# AFFECTS SHAPE OF THE GP's
# make sure you have the right amount!
a = ones(n_out, n_dif) / 20
kernel_lengths = [1, 1, 1] / 1.5
hyperparameters = append!(collect(Iterators.flatten(a)), kernel_lengths)
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
K_samp = total_covariance(x_samp, x_samp, hyperparameters)

# getting the Cholesky factorization of the covariance matrix (for drawing GPs)
L_samp = ridge_chol(K_samp).L  # usually has to add a ridge


function plot_im(A; file=0)
    figure(figsize=(10,6))
    fig = imshow(A[:,:])
    colorbar()
    title("Heatmap")
    if file != 0
        savefig(file)
    end
end


# showing a heatmap of the covariance matrix
plot_im(K_samp; file="figs/gp/initial_covariance.pdf")
plot_im(L_samp)


# quick and dirty function for creating plots that show what I want
function custom_line_plot(x_samp, L, x_obs, y_obs; output=1, draws=5000, σ=1, mean=zeros(amount_of_total_samp_points), show=10)

    # same curves are drawn every time?
    # srand(100)

    # initialize figure size
    figure(figsize=(10,6))

    # the indices corresponding to the proper output
    output_indices = (amount_of_samp_points * (output - 1) + 1):(amount_of_samp_points * output)

    # geting the y values for the proper output
    y = y_obs[(amount_of_measurements * (output - 1) + 1):(amount_of_measurements * output)]

    # initializing storage for example GPs to be plotted
    show_curves = zeros(show, amount_of_samp_points)

    # if no analytical variance is passed, estimate it with sampling
    if σ == 1

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

end


fig = custom_line_plot(x_samp, L_samp, x_obs, y_obs, output=1)
savefig("figs/gp/initial_gp_0.pdf")
fig = custom_line_plot(x_samp, L_samp, x_obs, y_obs, output=2)
savefig("figs/gp/initial_gp_1.pdf")
fig = custom_line_plot(x_samp, L_samp, x_obs, y_obs, output=3)
savefig("figs/gp/initial_gp_2.pdf")

# calculate posterior quantities
mean_post, return_vec = GP_posteriors(x_obs, x_samp, measurement_noise, hyperparameters, return_K=true, return_L=true, return_σ=true)
σ, K_post, L_post = return_vec

# plot the posterior covariance matrix (colors show how correlated points are to each other)
# plot_im(K_post)
# plot_im(L_post)

# plotting randomly drawn Gaussian processes with the mean function and covariance of the posterior
# much closer to the data, no?

# for 1D GPs
fig = custom_line_plot(x_samp, L_post, x_obs, y_obs, σ=σ, mean=mean_post, output=1)
savefig("figs/gp/cond_initial_gp_1.pdf")
fig = custom_line_plot(x_samp, L_post, x_obs, y_obs, σ=σ, mean=mean_post, output=2)
savefig("figs/gp/cond_initial_gp_2.pdf")
fig = custom_line_plot(x_samp, L_post, x_obs, y_obs, σ=σ, mean=mean_post, output=3)
savefig("figs/gp/cond_initial_gp_3.pdf")

# numerically maximize the likelihood to find the best hyperparameters
using Optim

# storing initial hyperparameters
initial_x = copy(hyperparameters)

# initializing Cholesky factorization storage
chol_storage = chol_struct(initial_x, ridge_chol(K_observations(x_obs, measurement_noise, initial_x)))

# lower = zeros(length(hyperparameters))
# lower = -5 * ones(length(hyperparameters))
# lower[(total_coefficients + 1):length(hyperparameters)] = zeros(length(hyperparameters) - total_coefficients)
# upper = 5 * ones(length(hyperparameters))

# can optimize with or without using the analytical gradient
# times listed are for optimizing 2 outputs with first order derivatives (7 hyperparameters)
# the optimization currently doesn't work because it looks for kernel lengths that are either negative (unphysical) or way too large kernel lengths (flatten covariance and prevents positice definiteness)

# use gradient
# result = optimize(nlogL, ∇nlogL, lower, upper, initial_x, Fminbox(GradientDescent()))  # 272.3 s
# @elapsed result = optimize(nlogL, ∇nlogL, initial_x, ConjugateGradient())  # 54.0 s, gave same result as Fminbox
# @elapsed result = optimize(nlogL, ∇nlogL, initial_x, GradientDescent(), Optim.Options(iterations=2, show_trace=true))  # 116.8 s, gave same result as SA
@elapsed result = optimize(nlogL, ∇nlogL, initial_x, LBFGS(), Optim.Options(show_trace=true))  # 19.8 s, unique result, takes around a minute with 50 real data points and 3 outputs for 3 hyperparameter model (9 total)
# @elapsed result = optimize(nlogL_penalty, ∇nlogL_penalty, initial_x, LBFGS())
# @elapsed result = optimize(nlogL, ∇nlogL, initial_x)
# result = optimize(nlogL_penalty, ∇nlogL_penalty, initial_x, LBFGS())  # same as above but with a penalty


# don't use gradient
# @elapsed result = optimize(nlogL, lower, upper, initial_x, SAMIN(), Optim.Options(iterations=10^6))  # 253.4 s

final_hyperparameters = result.minimizer

println("old hyperparameters")
println(initial_x)
println(nlogL(initial_x))

println("new hyperparameters")
println(final_hyperparameters)
println(result.minimum)


# reruning analysis of posterior with the "most likley" kernel amplitude and lengthscale
# recalculating posterior covariance and mean function
mean_post, return_vec = GP_posteriors(x_obs, x_samp, measurement_noise, final_hyperparameters, return_K=true, return_L=true, return_σ=true)
σ, K_post, L_post = return_vec

# plot the posterior covariance of the "most likely" posterior matrix
# (colors show how correlated points are to each other)
plot_im(K_post)


# for 1D GPs
fig = custom_line_plot(x_samp, L_post, x_obs, y_obs, σ=σ, mean=mean_post, output=1)
savefig("figs/gp/fit_gp_1.pdf")
fig = custom_line_plot(x_samp, L_post, x_obs, y_obs, σ=σ, mean=mean_post, output=2)
savefig("figs/gp/fit_gp_2.pdf")
fig = custom_line_plot(x_samp, L_post, x_obs, y_obs, σ=σ, mean=mean_post, output=3)
savefig("figs/gp/fit_gp_3.pdf")
