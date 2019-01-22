using SpecialFunctions
using LinearAlgebra


"""
All of these functions are enabled by a generic kernel function of the following form

function kernel(hyperparameters, t1::Float64, t2::Float64; dorder=[0, 0], dKdθ=0)
    ...
    return value
end
"""
# function kernel1(hyperparameters, x1, x2; dorder=[0,0], dKdθ=0, products = 0)
#
#     # finding required differences between inputs
#     dif_vec = x1 - x2  # raw difference vectors
#
#     function kernel_piece(hyper, dif, products_line)
#
#         # amount of possible derivatives on each function
#         dorders = length(hyper) + length(dorder)
#
#         # get the derivative orders for functions 1 and 2
#         dorder1 = convert(Array{Int64,1}, products_line[2:(dorders+1)])
#         dorder2 = convert(Array{Int64,1}, products_line[(dorders + 2):(2 * dorders+1)])
#
#         # return 0 if you know that that portion will equal 0
#         # this is when you are deriving one of the kernels by a hyperparameter
#         # of the other kernel
#         if (((dorder1[length(dorder) + 2] == 1) | (dorder1[length(dorder) + 3] == 1))
#             | (dorder2[length(dorder) + 1] == 1))
#
#             return 0
#
#         else
#
#             # use the properly differentiated version of kernel function 1
#             if dorder1[length(dorder) + 1] == 1
#                 func1 = dRBFdλ_kernel([hyper[1]], dif, dorder1[1:length(dorder)])
#             # elseif ((dorder1[length(dorder) + 2] == 1) | (dorder1[length(dorder) + 3] == 1))
#             #     func1 = 0
#             else
#                 func1 = dRBFdt_kernel([hyper[1]], dif, dorder1[1:length(dorder)])
#             end
#
#             # use the properly differentiated version of kernel function 2
#             # if dorder2[length(dorder) + 1] == 1
#             #     func2 = 0
#             if dorder2[length(dorder) + 2] == 1
#                 func2 = dPeriodicdλ_kernel(hyper[2:3], dif, dorder2[1:length(dorder)])
#             elseif dorder2[length(dorder) + 3] == 1
#                 func2 = dPeriodicdp_kernel(hyper[2:3], dif, dorder2[1:length(dorder)])
#             else
#                 func2 = dPeriodicdt_kernel(hyper[2:3], dif, dorder2[1:length(dorder)])
#             end
#
#             return func1 * func2
#
#         end
#     end
#
#     # calculate the product rule coefficients and dorders if they aren't passed
#     if products == 0
#         non_coefficient_hyperparameters = length(hyperparameters) - total_coefficients
#         new_dorder = append!(copy(dorder), zeros(non_coefficient_hyperparameters))
#         # differentiate by RBF kernel length
#         if dKdθ == length(hyperparameters) - 2
#             new_dorder[length(dorder) + 1] = 1
#         # differentiate by Periodic kernel length
#         elseif dKdθ == length(hyperparameters) - 1
#             new_dorder[length(dorder) + 2] = 1
#         # differentiate by Periodic kernel period
#         elseif dKdθ == length(hyperparameters)
#             new_dorder[length(dorder) + 3] = 1
#         end
#         products = product_rule(new_dorder)
#     end
#
#     # add all of the differentiated kernels together according to the product rule
#     final = sum([products[i, 1] * kernel_piece(hyperparameters[(length(hyperparameters) - 2):length(hyperparameters)], dif_vec[1], products[i,:]) for i in 1:size(products, 1)])
#
#     return final
#
# end



"""
Creates the covariance matrix by evaluating the kernel function for each pair
of passed inputs. Generic. Complicated covariances accounted for in the
total_covariance function
the symmetric parameter asks whether or note the kernel used will be symmetric about dif=0 (only guarunteed for undifferentiated kernels)
"""
function covariance(x1list, x2list, kernel_hyperparameters; dorder=[0, 0], symmetric=false, dKdθ=0)

    # are the list of x's passed identical
    same_x = (x1list == x2list)

    # are the x's passed identical and equally spaced
    if same_x
        spacing = [x1list[i]-x1list[i-1] for i in 2:length(x1list)]
        equal_spacing = all([abs(spacing[i] - spacing[1]) < 1e-8 for i in 2:length(spacing)])
    else
        equal_spacing = false
    end

    x1_length = size(x1list, 1)
    x2_length = size(x2list, 1)
    K = zeros((x1_length, x2_length))

    if equal_spacing && symmetric
        kernline = zeros(x1_length)
        for i in 1:x1_length
            kernline[i] = kernel(kernel_hyperparameters, x1list[1,:], x1list[i,:], dorder=dorder, dKdθ=dKdθ)
        end
        for i in 1:x1_length
            for j in 1:x1_length
                if i <= j
                    K[i, j] = kernline[j - i + 1]
                end
            end
        end
        return Symmetric(K)
    elseif same_x && symmetric
        for i in 1:x1_length
            for j in 1:x1_length
                if i <= j
                    K[i, j] = kernel(kernel_hyperparameters, x1list[i,:], x1list[j,:], dorder=dorder, dKdθ=dKdθ)
                end
            end
        end
        return Symmetric(K)
    else
        for i in 1:x1_length
            for j in 1:x2_length
                K[i, j] = kernel(kernel_hyperparameters, x1list[i,:], x2list[j,:], dorder=dorder, dKdθ=dKdθ)
            end
        end
        return K
    end
end


# """
# Calculating the covariance between all outputs for a combination of dependent GPs
# written so that the intermediate K's don't have to be calculated over and over again
# """
# function dependent_covariance(x1list, x2list, hyperparameters; dKdθ=0, coeff_orders=0)
#
#     # calculating the total size of the multi-output covariance matrix
#     point_amount = [size(x1list, 1), size(x2list, 1)]
#     K = zeros((n_out * point_amount[1], n_out * point_amount[2]))
#
#     non_coefficient_hyperparameters = length(hyperparameters) - total_coefficients
#
#     # calculating all of the sub-matrices explicitly
#     # A = Array{Any}(n_dif, n_dif)
#     # for k in 1:n_dif
#     #     for l in 1:n_dif
#     #         dorder = [k - 1, l - 1]
#     #         # things that have been differentiated an even amount of times are symmetric
#     #         if iseven(k + l)
#     #             A[k, l] = covariance(x1list, x2list, hyperparameters; dorder=dorder, symmetric=true, dKdθ=dKdθ)
#     #         else
#     #             A[k, l] = covariance(x1list, x2list, hyperparameters; dorder=dorder, dKdθ=dKdθ)
#     #         end
#     #     end
#     # end
#     #
#     # save_A(A)
#
#     # only calculating each sub-matrix once and using the fact that they should
#     # be basically the same if the kernel has been differentiated the same amount of times
#     A_list = Array{Any}(nothing, 2 * n_dif - 1)
#     for k in 1:(2 * n_dif - 1)
#         dorder = [0, k - 1]
#
#         new_dorder = append!(copy(dorder), zeros(non_coefficient_hyperparameters))
#         # differentiate by RBF kernel length
#         if dKdθ == length(hyperparameters) - 2
#             new_dorder[length(dorder) + 1] = 1
#         # differentiate by Periodic kernel length
#         elseif dKdθ == length(hyperparameters) - 1
#             new_dorder[length(dorder) + 2] = 1
#         # differentiate by Periodic kernel period
#         elseif dKdθ == length(hyperparameters)
#             new_dorder[length(dorder) + 3] = 1
#         end
#         products = product_rule(new_dorder)
#
#
#         # things that have been differentiated an even amount of times are symmetric
#         if isodd(k)
#             A_list[k] = covariance(x1list, x2list, hyperparameters; dorder=dorder, symmetric=true, dKdθ=dKdθ, products=products)
#         else
#             A_list[k] = covariance(x1list, x2list, hyperparameters; dorder=dorder, dKdθ=dKdθ, products=products)
#         end
#     end
#
#
#     # return the properly negative differentiated A matrix from the list
#     # make it negative or not based on how many times it has been differentiated in the x1 direction
#     A_mat(k, l, A_list) = ((-1) ^ (k - 1)) * A_list[k + l - 1]
#
#
#     # reshaping the list into a format consistent with the explicit calculation
#     # A = Array{Any}(n_dif, n_dif)
#     # for k in 1:n_dif
#     #     for l in 1:n_dif
#     #         A[k, l] =  A_mat(k, l, A_list)
#     #     end
#     # end
#     #
#     # save_A(A)
#     # save_A(A_list)
#
#     # assembling the total covariance matrix
#     a = reshape(hyperparameters[1:total_coefficients], (n_out, n_dif))
#     # if we aren't differentiating by one of the coefficient hyperparameters
#     # assemble the covariance matrix in the expected way
#     if dKdθ == 0 || dKdθ > total_coefficients
#         for i in 1:n_out
#             for j in 1:n_out
#                 for k in 1:n_dif
#                     for l in 1:n_dif
#                         if (i == j) & isodd(k + l)
#                             # the cross terms (of odd differentiation orders) cancel each other out in diagonal matrices
#                         else
#                             K[((i - 1) * point_amount[1] + 1):(i * point_amount[1]),
#                                 ((j - 1) * point_amount[2] + 1):(j * point_amount[2])] +=
#                                 # a[i, k] * a[j, l] * A[k, l]
#                                 a[i, k] * a[j, l] *  A_mat(k, l, A_list)
#                         end
#                     end
#                 end
#             end
#         end
#     # if we are differentiating by one of the coefficient hyperparameters
#     # we have to assemble the covariance matrix in a different way
#     else
#         # ((output pair), (which A matrix to use), (which a coefficent to use))
#         # get all of the coefficients for coefficient hyperparameters based on
#         # the amount of outputs and differentiations
#         coeff = dif_coefficients(n_out, n_dif, dKdθ; coeff_orders=coeff_orders, a=a)
#
#         for i in 1:n_out
#             for j in 1:n_out
#                 for k in 1:n_dif
#                     for l in 1:n_dif
#                         for m in 1:n_out
#                             for n in 1:n_dif
#                                 if coeff[i, j, k, l, m, n] != 0
#                                     K[((i - 1) * point_amount[1] + 1):(i * point_amount[1]),
#                                         ((j - 1) * point_amount[2] + 1):(j * point_amount[2])] +=
#                                         # coeff[i, j, k, l, m, n] * a[m, n] * A[k, l]
#                                         coeff[i, j, k, l, m, n] * a[m, n] * A_mat(k, l, A_list)
#                                 end
#                             end
#                         end
#                     end
#                 end
#             end
#         end
#     end
#
#     # this would just be a wrapper function for a less complicated kernel
#     # all you would need is the following line
#     # K = covariance(x1list, x2list, hyperparameters)
#
#     # return the symmetrized version of the covariance matrix
#     # function corrects for numerical errors and notifies us if our matrix isn't
#     # symmetric like it should be
#     return symmetric_A(K)
#
# end


# "\"true\" underlying for the fake observations"
# function observations(x, measurement_noise)
#     # a phase shifted sine curve with measurement noise and inherent noise
#     measurement_noise += 0.2 * ones(length(measurement_noise))  # adding a noise component inherent to the activity
#     if length(size(x)) > 1
#         shift = 2 * pi * rand(size(x, 2))
#         return [sum(sin.(pi / 2 * x[i,:] + [shift])) for i in 1:size(x, 1)] + measurement_noise .^ 2 .* randn(size(x, 1))
#     else
#         shift = 2 * pi * rand()
#         return [sum(sin.(pi / 2 * x[i,:] + [shift])) for i in 1:length(x)] + measurement_noise .^ 2 .* randn(length(x))
#     end
# end


"adding measurement noise to K_obs"
function K_observations(x_obs, measurement_noise, total_hyperparameters; ignore_asymmetry=false)
    K_obs = dependent_covariance(x_obs, x_obs, total_hyperparameters)
    @assert (size(K_obs, 1) == length(measurement_noise)) ["measurement_noise is the wrong length"]
    for i in 1:size(K_obs, 1)
        K_obs[i, i] +=  measurement_noise[i] ^ 2
    end
    return symmetric_A(K_obs; ignore_asymmetry=ignore_asymmetry)
end


"calculating the variance at each GP posterior point"
function get_σ(L_obs, K_obs_samp, K_samp)

    v = L_obs \ K_obs_samp
    V = zeros(size(K_samp, 1))
    for i in 1:size(K_samp, 1)
        V[i] = K_samp[i, i] - transpose(v[:, i]) * v[:, i]

        # this should only happen at plot points that are very close to the observation points
        if V[i] < 0
            println("ignored a negative value in variance calculation")
            println(V[i])
            V[i] = 0
        end

    end
    σ = sqrt.(V)
    return σ

end


"conditions a GP with data"
function GP_posteriors(x_obs, x_samp, measurement_noise, total_hyperparameters; return_σ=false, return_K=false, return_L=false)

    return_vec = []

    K_samp = dependent_covariance(x_samp, x_samp, total_hyperparameters)
    K_obs = K_observations(x_obs, measurement_noise, total_hyperparameters)

    K_samp_obs = dependent_covariance(x_samp, x_obs, total_hyperparameters)
    # K_samp_obs = (K_samp_obs + transpose(dependent_covariance(x_obs, x_samp, hyperparameters)) / 2
    K_obs_samp = transpose(K_samp_obs)

    # (RW alg. 2.1)

    # tells Julia to perform and store the Cholesky factorization
    L_fact = ridge_chol(K_obs)

    # actual lower triangular matrix values
    L = L_fact.L

    # these are all equivalent but have different computational costs
    # α = inv(L_fact) * y_obs
    # α = transpose(L) \ (L \ y_obs)
    α = L_fact \ y_obs

    mean_post = K_samp_obs * α

    if return_σ
        σ = get_σ(L, K_obs_samp, K_samp)
        append!(return_vec, [σ])
    end

    if return_L || return_K
        K_post = symmetric_A(K_samp - (K_samp_obs * (L_fact \ K_obs_samp)))
        if return_K
            append!(return_vec, [K_post])
        end
        if return_L
            L_post = ridge_chol(K_post).L
            append!(return_vec, [L_post])
        end
    end

    return mean_post, return_vec
end


"""negative log likelihood of the data given the current kernel parameters (as seen on page 19)
(negative because Optim has a minimizer)"""
function nlogL(hyperparameter_list...)

    hyper = []
    for i in 1:length(hyperparameter_list)
        append!(hyper, hyperparameter_list[i])
    end
    n=length(y_obs)

    # a weirdly necessary dummy variable
    measurement_noise_dummy = measurement_noise
    K_obs = K_observations(x_obs, measurement_noise_dummy, hyper; ignore_asymmetry=true)
    L_fact = stored_chol(chol_storage, hyper, K_obs; notification=false)
    # inv_K_obs = inv(L_fact)  # ~35% faster than inv(K_obs)
    # det_K_obs = det(L_fact)  # ~8% faster than det(K_obs)

    # goodness of fit term
    data_fit = -1 / 2 * (transpose(y_obs) * (L_fact \ y_obs))
    # complexity penalization term
    # penalty = -1 / 2 * log(det(L_fact))
    penalty = -1 / 2 * logdet(L_fact)  # half memory but twice the time
    # normalization term (functionally useless)
    normalization = -n / 2 * log(2 * pi)

    return -1 * (data_fit + penalty + normalization)
end


"""http://www.gaussianprocess.org/gpml/chapters/RW5.pdf
gradient of negative log likelihood of the data given the current kernel parameters (as seen on page 19)"""
function ∇nlogL(G, hyperparameter_list...)

    hyper = []
    for i in 1:length(hyperparameter_list)
        append!(hyper, hyperparameter_list[i])
    end

    # a weirdly necessary dummy variable
    measurement_noise_dummy = measurement_noise
    K_obs = K_observations(x_obs, measurement_noise_dummy, hyper; ignore_asymmetry=true)
    L_fact = stored_chol(chol_storage, hyper, K_obs; notification=false)
    # inv_K_obs = inv(L_fact)


    function grad(dK_dθj)
        # derivative of goodness of fit term
        data_fit = 1 / 2 * (transpose(y_obs) * (L_fact \ (dK_dθj * (L_fact \ y_obs))))
        # derivative of complexity penalization term
        penalty = -1 / 2 * tr(L_fact \ dK_dθj)
        return -1 * (data_fit + penalty)
    end

    # taking some burden off of recalculating the coefficient orders used by the automatic differentiation
    coeff_orders = coefficient_orders(n_out, n_dif)

    for i in 1:(length(hyper))
        G[i] = grad(dependent_covariance(x_obs, x_obs, hyper; dKdθ=i, coeff_orders=coeff_orders))
    end

end


"creating a Cholesky factorization storage structure"
struct chol_struct
    total_hyperparameters::Array{Float64,1}
    cholesky_object::Cholesky{Float64,Array{Float64,2}}
end


"An effort to avoid recalculating Cholesky factorizations"
function stored_chol(chol_storage, new_hyper, A; return_values=false, notification=true, ridge=1e-6)

    # if the Cholesky factorization wasn't just calculated, calculate a new one.
    if chol_storage.total_hyperparameters != new_hyper
        chol_storage = chol_struct(copy(new_hyper), ridge_chol(A; notification=notification, ridge=ridge))
    end
    return chol_storage.cholesky_object

end


"""find the powers that each coefficient is taken to for each part of the matrix construction
used for constructing differentiated versions of the kernel"""
function coefficient_orders(n_out, n_dif; a=ones(n_out, n_dif))

    coeff_orders = zeros(n_out,n_out,n_dif,n_dif,n_out,n_dif)
    for i in 1:n_out
        for j in 1:n_out
            for k in 1:n_dif
                for l in 1:n_dif
                    if (i == j) & isodd(k + l)
                        # the cross terms (of odd differentiation orders) cancel each other out in diagonal matrices
                    else
                        for m in 1:n_out
                            for n in 1:n_dif
                                if a[m, n] != 0
                                    if [m, n] == [i, k]
                                        coeff_orders[i, j, k, l, m, n] += 1
                                    end
                                    if [m, n] == [j, l]
                                        coeff_orders[i, j, k, l, m, n] += 1
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    return coeff_orders

end


"a small function to get indices that make sense from Julia's reshaping routine"
proper_index(i::Int) = [convert(Int64, rem(i - 1, n_out)) + 1, convert(Int64, floor((i -1) / n_out)) + 1]


"""getting the coefficients for constructing differentiated versions of the kernel
using the powers that each coefficient is taken to for each part of the matrix construction"""
function dif_coefficients(n_out, n_dif, dKdθ; coeff_orders=0, a=ones(n_out, n_dif))

    if coeff_orders == 0
        coeff_orders = coefficient_orders(n_out, n_dif, a=a)
    end

    # ((output pair), (which A matrix to use), (which a coefficent to use))
    coeff = zeros(n_out,n_out,n_dif,n_dif,n_out,n_dif)
    proper_indices = proper_index(dKdθ)
    for i in 1:n_out
        for j in 1:n_out
            for k in 1:n_dif
                for l in 1:n_dif
                    if coeff_orders[i, j, k, l, proper_indices[1], proper_indices[2]] == 1
                        coeff[i, j, k, l, :, :] = coeff_orders[i, j, k, l, :, :]
                        coeff[i, j, k, l, proper_indices[1], proper_indices[2]] = 0
                    elseif coeff_orders[i, j, k, l, proper_indices[1], proper_indices[2]] == 2
                        coeff[i, j, k, l, :, :] = coeff_orders[i, j, k, l, :, :]
                    end
                end
            end
        end
    end

    return coeff
end
