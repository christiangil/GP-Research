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
    final = sum([products[i, 1] * kernel_piece(hyperparameters[(length(hyperparameters) - 2):length(hyperparameters)], dif_vec[1], products[i,:]) for i in 1:size(products)[1]])

    return final

end


# calculating the covariance between all outputs for a combination of dependent GPs
# written so that the intermediate K's don't have to be calculated over and over again
function total_covariance(x1list, x2list, hyperparameters; dKdθ=0, coeff_orders=0)

    # calculating the total size of the multi-output covariance matrix
    point_amount = [size(x1list)[1], size(x2list)[1]]
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
    # A_list = Array{Any}(2 * n_dif - 1)  # depreciated in 1.0
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
# a = [[1 .2];]
a = [[1 .2];[2 .3]]
# a = [[1  0.7 3];[0.4  1.5 3];[1 1 1]]
kernel_lengths = [2, 2, 2]
hyperparameters = append!(collect(Iterators.flatten(a)), kernel_lengths)

# how finely to sample the domain (for plotting)
amount_of_samp_points = 100

# how wide the measurement domain is
domain = 8

# creating many inputs to sample the eventual gaussian process on (for plotting)
x1_samp = linspace(0, domain[1], amount_of_samp_points[1])
# x2_samp = linspace(0, domain[2], amount_of_samp_points[2])

# for 1D GPs
x_samp = x1_samp

# how many outputs there will be
n_out = size(a)[1]
# how many differentiated versions of the original GP there will be
n_dif = size(a)[2]

total_coefficients = n_out * n_dif

# total amount of output points
amount_of_total_samp_points = amount_of_samp_points * n_out

# Finding how correlated the sampled inputs are to each other
# (aka getting the covariance matrix by evaluating the kernel function at all
# pairs of points)
K_samp = total_covariance(x_samp, x_samp, hyperparameters)

# getting the Cholesky factorization of the covariance matrix (for drawing GPs)
L_samp = ridge_chol(K_samp).L


# showing a heatmap of the covariance matrix
# plot(heatmap(z=K_samp, colorscale="Viridis"))  # PlotlyJS
function plot_K(K)
    figure(figsize=(8, 8))
    imshow(K[:,:])
    colorbar()
end


plot_K(K_samp)
# plot_K_trace(K_samp)


# creating observations to test methods on
amount_of_measurements = 20

# Uncertainty in the data (AFFECTS SPREAD OF DATA AND HOW TIGHTLY THE GP's WILL
# TRY TO HUG THE DATA) aka how much noise is added to measurements and
# measurement covariance function
# a vector of length = amount_of_measurements
measurement_noise = 0.3 * ones(amount_of_measurements)

# x_obs = linspace(0,domain,amount_of_measurements)  # observation inputs
# srand(1234)
# observations taken at random points in the measurement domain
x_obs = domain[1] * rand((amount_of_measurements))

# sorting inputs
if length(domain) > 1
    for i in 2:length(domain)
        x_obs = hcat(x_obs, domain[i] * rand(amount_of_measurements))
    end
    x_obs = sortrows(x_obs, by=x->(x[1]))
else
    x_obs = sort(x_obs)
end

# getting simulated measurement values
y_obs = zeros(n_out * amount_of_measurements)
for i in 1:n_out
    y_obs[(amount_of_measurements * (i - 1) + 1):(amount_of_measurements * i)] = observations(x_obs, measurement_noise)
end


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

    xlabel("x (time)")
    ylabel("y (flux or something lol)")
    title("test")

end


# # quick and dirty function for creating plots that show what I want
# function custom_line_plot(x_samp, L, x_obs, y_obs; output=1, draws=5000, σ=1, mean=zeros(amount_of_total_samp_points), show=10)
#
#     # same curves are drawn every time?
#     # srand(100)
#
#     y = y_obs[(amount_of_measurements * (output - 1) + 1):(amount_of_measurements * output)]
#
#     # if no analytical variance is passed, estimate it with sampling
#     if σ == 1
#
#         storage = zeros((draws, amount_of_total_samp_points))
#         for i in 1:draws
#             storage[i, :] = L * randn(amount_of_total_samp_points) + mean
#         end
#
#         storage = storage[:, (amount_of_samp_points * (output - 1) + 1):(amount_of_samp_points * output)]
#
#         show_curves = storage[1:show, :]
#         storage = sort(storage, 1)
#
#         # filling the 5-95th percentile with a transparent orange
#         upper = scatter(;x=x_samp, y=storage[convert(Int64, 0.95 * draws),:], mode="lines", line_width=0)
#         lower = scatter(;x=x_samp, y=storage[convert(Int64, 0.05 * draws),:], fill="tonexty", mode="lines", line_width=0)
#
#         mean = mean[(amount_of_samp_points * (output - 1) + 1):(amount_of_samp_points * output)]
#
#     else
#
#         show_curves = zeros((show, amount_of_total_samp_points))
#         for i in 1:show
#             show_curves[i,:] = L * randn(amount_of_total_samp_points) + mean
#         end
#
#         show_curves = show_curves[:, (amount_of_samp_points * (output - 1) + 1):(amount_of_samp_points * output)]
#
#         mean = mean[(amount_of_samp_points * (output - 1) + 1):(amount_of_samp_points * output)]
#         σ = σ[(amount_of_samp_points * (output - 1) + 1):(amount_of_samp_points * output)]
#
#         # filling the 5-95th percentile with a transparent orange
#         upper = scatter(;x=x_samp, y=mean + 1.96 * σ, mode="lines", line_width=0)
#         lower = scatter(;x=x_samp, y=mean - 1.96 * σ, fill="tonexty", mode="lines", line_width=0)
#
#     end
#
#     median = scatter(;x=x_samp, y=mean, mode="lines", line_width=4, line_color="rgb(0, 0, 0)")
#     data_trace = scatter(;x=x_obs, y=y, mode="markers", marker_size=12, marker_color="rgb(0, 0, 0)")
#     return plot(append([upper, lower, median], traces(x_samp, show_curves), [data_trace]), layout)
#
# end


plt = custom_line_plot(x_samp, L_samp, x_obs, y_obs, output=1)
# savefig(plt, "figs/GP/initial_gp_1.pdf")
plt = custom_line_plot(x_samp, L_samp, x_obs, y_obs, output=2)
# savefig(plt, "figs/GP/initial_gp_2.pdf")
# plt = custom_line_plot(x_samp, L_samp, x_obs, y_obs, output=3)
# savefig(plt, "figs/GP/initial_gp_3.pdf")

# calculate posterior quantities
mean_post, return_vec = GP_posteriors(x_obs, x_samp, measurement_noise, hyperparameters, return_K=true, return_L=true, return_σ=true)
σ, K_post, L_post = return_vec


# plot the posterior covariance matrix (colors show how correlated points are to each other)
# possible colors found here (https://plot.ly/julia/heatmaps/)
# plot(heatmap(z=K_post, colorscale="Viridis"))

# plotting randomly drawn Gaussian processes with the mean function and covariance of the posterior
# much closer to the data, no?

# for 1D GPs
plt = custom_line_plot(x_samp, L_post, x_obs, y_obs, σ=σ, mean=mean_post, output=1)
# savefig(plt, "figs/GP/cond_initial_gp_1.pdf")
plt = custom_line_plot(x_samp, L_post, x_obs, y_obs, σ=σ, mean=mean_post, output=2)
# savefig(plt, "figs/GP/cond_initial_gp_2.pdf")
# plt = custom_line_plot(x_samp, L_post, x_obs, y_obs, σ=σ, mean=mean_post, output=3)
# savefig(plt, "figs/GP/cond_initial_gp_3.pdf")

# numerically maximize the likelihood to find the best hyperparameters
# Pkg.add("Optim")
using Optim

# storing initial hyperparameters
initial_x = copy(hyperparameters)

# initializing Cholesky factorization storage
chol_storage = chol_struct(initial_x, ridge_chol(K_observations(x_obs, measurement_noise, initial_x)))

# lower = zeros(length(hyperparameters))
# lower = -4 * ones(length(hyperparameters))
# lower[(total_coefficients + 1):length(hyperparameters)] = 0
# upper = 7 * ones(length(hyperparameters))

# can optimize with or without using the analytical gradient
# times listed are for optimizing 2 outputs with first order derivatives (7 hyperparameters)

# use gradient
# tic(); result1 = optimize(nlogL, ∇nlogL, lower, upper, initial_x, Fminbox(GradientDescent())); toc()  # 272.3 s
# tic(); result2 = optimize(nlogL, ∇nlogL, initial_x, ConjugateGradient()); toc()  # 54.0 s, gave same result as Fminbox
# tic(); result3 = optimize(nlogL, ∇nlogL, initial_x, GradientDescent()); toc()  # 116.8 s, gave same result as SA
@elapsed result = optimize(nlogL, ∇nlogL, initial_x, LBFGS())
# tic(); result = optimize(nlogL_penalty, ∇nlogL_penalty, initial_x, LBFGS()); toc()  # 19.8 s, unique result

# don't use gradient
# tic(); result5 = @time optimize(nlogL, lower, upper, initial_x, SAMIN(), Optim.Options(iterations=10^6)); toc()  # 253.4 s

# println(result1.minimizer)
# println(result1.minimum)
# println(result2.minimizer)
# println(result2.minimum)
# println(result3.minimizer)
# println(result3.minimum)
# println(result4.minimizer)
# println(result4.minimum)
# println(result5.minimizer)
# println(result5.minimum)

println("old hyperparameters")
println(initial_x)

println("new hyperparameters")
println(result.minimizer)
println(result.minimum)

final_hyperparameters = result.minimizer

# reruning analysis of posterior with the "most likley" kernel amplitude and lengthscale
# recalculating posterior covariance and mean function
mean_post, return_vec = GP_posteriors(x_obs, x_samp, measurement_noise, final_hyperparameters, return_K=true, return_L=true, return_σ=true)
σ, K_post, L_post = return_vec

# plot the posterior covariance of the "most likely" posterior matrix
# (colors show how correlated points are to each other)
# plot(heatmap(z=K_post, colorscale="Viridis"))


# for 1D GPs
plt = custom_line_plot(x_samp, L_post, x_obs, y_obs, σ=σ, mean=mean_post, output=1)
# savefig(plt, "figs/GP/fit_gp_1.pdf")
plt = custom_line_plot(x_samp, L_post, x_obs, y_obs, σ=σ, mean=mean_post, output=2)
# savefig(plt, "figs/GP/fit_gp_2.pdf")
# plt = custom_line_plot(x_samp, L_post, x_obs, y_obs, σ=σ, mean=mean_post, output=3)
# savefig(plt, "figs/GP/fit_gp_3.pdf")











# create a contour plot that is smooth and puts an x on the [minx, miny] point
function custom_contour(x, y, z, min_x, min_y)

    main = contour(;z=z, x=x, y=y, showscale=false, colorscale="Viridis", contours_coloring="heatmap")
    best_point = scatter(;x=[min_x], y=[min_y],
        mode="markers", marker_size=12, marker_symbol="x", marker_color="rgb(0, 0, 0)",
        showlegend=false)
    return plot([main, best_point])

# end


# creating corner plots
function corner_plot(; steps=30+1)
    data = [1]
    spread = 1
    for k in 1:length(final_hyperparameters)
        for l in 1:length(final_hyperparameters)

            hold = copy(final_hyperparameters)
            if k == l
                # x = linspace(maximum([0, final_hyperparameters[k] - spread]), final_hyperparameters[k] + spread, steps)
                x = linspace(final_hyperparameters[k] - spread, final_hyperparameters[k] + spread, steps)
                hold[k] = x[1]
                y = [nlogL(hold)]
                for i in 2:length(x)
                    hold[k] = x[i]
                    y = append!(y,[nlogL(hold)])
                end
                plt = plot([line_trace(x, y), line_trace([final_hyperparameters[k], final_hyperparameters[k]], [minimum(y), maximum(y)], color="black")])
            elseif k < l
                # x = linspace(maximum([0, final_hyperparameters[k] - spread]), final_hyperparameters[k] + spread, steps)
                # y = linspace(maximum([0, final_hyperparameters[l] - spread]), final_hyperparameters[l] + spread, steps)
                x = linspace(final_hyperparameters[k] - spread, final_hyperparameters[k] + spread, steps)
                y = linspace(final_hyperparameters[l] - spread, final_hyperparameters[l] + spread, steps)

                z = zeros((steps,steps))
                for i in 1:steps
                    for j in 1:steps
                        hold[k] = x[i]
                        hold[l] = y[j]
                        z[i,j] = nlogL(hold)
                    end
                end
                plt = custom_contour(x, y, z, final_hyperparameters[k], final_hyperparameters[l])
            else
                plt = plot()
            end

            data = hcat(data,[plt])

        end
    end
    data = data[2:length(data)]

    return return_plot_matrix(data)

end


data = corner_plot()
savefig(data, "figs/GP/corner_plot.pdf")
