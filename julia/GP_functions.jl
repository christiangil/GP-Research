using SpecialFunctions
using LinearAlgebra
using Test


struct Jones_problem_definition
    kernel
    n_kern_hyper::Int
    n_dif::Int
    n_out::Int
    x_obs::Array{Float64,1}
    y_obs::Array{Float64,1}
    noise::Array{Float64,1}
    a0::Array{Float64,2}
    coeff_orders::Array{Float64,6}
end


function build_problem_definition(kernel_func, num_kernel_hyperparameters::Int, n_dif::Int, n_out::Int, x_obs::Array{Float64,1}, y_obs::Array{Float64,1}, noise::Array{Float64,1}, a0::Array{Float64,2}, coeff_orders::Array{Float64,6})
    @assert isfinite(kernel_func(ones(num_kernel_hyperparameters), randn(); dorder=zeros(2 + num_kernel_hyperparameters)))  # make sure the kernel is valid by testing a sample input
    @assert n_dif>0
    @assert n_out>0
    @assert (length(x_obs) * n_out) == length(y_obs)
    @assert length(y_obs) == length(noise)
    @assert size(a0) == (n_out, n_dif)
    @assert size(coeff_orders) == (n_out, n_out, n_dif, n_dif, n_out, n_dif)
    global problem_definition = Jones_problem_definition(kernel_func, num_kernel_hyperparameters, n_dif, n_out, x_obs, y_obs, noise, a0, coeff_orders)
end


function build_problem_definition(kernel_func, num_kernel_hyperparameters::Int, n_dif::Int, n_out::Int, x_obs::Array{Float64,1}, y_obs::Array{Float64,1}, noise::Array{Float64,1}, a0::Array{Float64,2})
    build_problem_definition(kernel_func, num_kernel_hyperparameters, n_dif, n_out, x_obs, y_obs, noise, a0, coefficient_orders(n_out, n_dif, a=a0))
end


# include("kernels/RBF_kernel.jl")  # sets correct num_kernel_hyperparameters
# build_problem_definition(RBF_kernel, num_kernel_hyperparameters, 1, 1, zeros(1), zeros(1,1), zeros(1,1))


# Creating a custom kernel (possibly by adding and multiplying other kernels?)
# t1 and t2 are single time points
# kernel_hyperparameters = total_hyperparameters[(num_coefficients + 1):end] = the hyperparameters for the base kernel (e.g. [kernel_period, kernel_length])
# total_hyperparameters = kernel_hyperparameters appended to a flattened list of coefficients
# num_coefficients is the number of coefficient "hyperparameters"
function kernel(kernel_func, kernel_hyperparameters::Union{Array{Any,1},Array{Float64,1}}, t1::Union{Float64,Array{Float64,1}}, t2::Union{Float64,Array{Float64,1}}; dorder::Union{Array{Int,1},Array{Float64,1}}=zeros(2), dKdθ_kernel::Int=0)

    @assert dKdθ_kernel <= length(kernel_hyperparameters) "Asking to differentiate by hyperparameter that the kernel doesn't have"

    dif = (t1 - t2)[1]
    dorder_tot = append!(copy(dorder), zeros(length(kernel_hyperparameters)))

    if dKdθ_kernel > 0
        dorder_tot[2 + dKdθ_kernel] = 1
    end

    return kernel_func(kernel_hyperparameters, dif; dorder=dorder_tot)

end


function kernel(prob_def::Jones_problem_definition, kernel_hyperparameters::Union{Array{Any,1},Array{Float64,1}}, t1::Union{Float64,Array{Float64,1}}, t2::Union{Float64,Array{Float64,1}}; dorder::Union{Array{Int,1},Array{Float64,1}}=zeros(2), dKdθ_kernel::Int=0)
    return kernel(prob_def.kernel, kernel_hyperparameters, t1, t2; dorder=dorder, dKdθ_kernel=dKdθ_kernel)
end


"""
Creates the covariance matrix by evaluating the kernel function for each pair
of passed inputs. Generic. Complicated covariances accounted for in the
total_covariance function
the symmetric parameter asks whether or note the kernel used will be symmetric about dif=0 (only guarunteed for undifferentiated kernels)
"""
function covariance(kernel_func, x1list::Union{Array{Float64,1},Array{Float64,2}}, x2list::Union{Array{Float64,1},Array{Float64,2}}, kernel_hyperparameters::Union{Array{Any,1},Array{Float64,1}}; dorder::Union{Array{Float64,1},Array{Int,1}}=[0, 0], symmetric::Bool=false, dKdθ_kernel::Int=0)

    # @assert dKdθ <= length()
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
            kernline[i] = kernel(kernel_func, kernel_hyperparameters, x1list[1,:], x1list[i,:], dorder=dorder, dKdθ_kernel=dKdθ_kernel)
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
                    K[i, j] = kernel(kernel_func, kernel_hyperparameters, x1list[i,:], x1list[j,:], dorder=dorder, dKdθ_kernel=dKdθ_kernel)
                end
            end
        end
        return Symmetric(K)
    else
        for i in 1:x1_length
            for j in 1:x2_length
                K[i, j] = kernel(kernel_func, kernel_hyperparameters, x1list[i,:], x2list[j,:], dorder=dorder, dKdθ_kernel=dKdθ_kernel)
            end
        end
        return K
    end
end


"""
Calculating the covariance between all outputs for a combination of dependent GPs
written so that the intermediate K's don't have to be calculated over and over again
"""
function covariance(prob_def::Jones_problem_definition, x1list::Union{Array{Float64,1},Array{Float64,2}}, x2list::Union{Array{Float64,1},Array{Float64,2}}, total_hyperparameters::Union{Array{Any,1},Array{Float64,1}}; dKdθ_total::Int=0)

    @assert dKdθ_total >= 0
    @assert length(total_hyperparameters) == prob_def.n_kern_hyper + length(prob_def.a0)
    num_coefficients = length(total_hyperparameters) - prob_def.n_kern_hyper
    n_out = prob_def.n_out
    n_dif = prob_def.n_dif
    dKdθ_kernel = dKdθ_total - num_coefficients
    kernel_hyperparameters = total_hyperparameters[(num_coefficients + 1):end]
    # println(length(kernel_hyperparameters))

    # calculating the total size of the multi-output covariance matrix
    point_amount = [size(x1list, 1), size(x2list, 1)]
    K = zeros((n_out * point_amount[1], n_out * point_amount[2]))

    # non_coefficient_hyperparameters = length(total_hyperparameters) - num_coefficients

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

        # CHANGE THIS TO MAKE MORE SENSE WITH NEW KERNEL SCHEME
        # VERY HACKY
        # dorder = [0, k - 1]
        total_time_derivatives = k - 1
        dorder = [2 * div(total_time_derivatives - 1, 2), rem(total_time_derivatives - 1, 2) + 1]

        # new_dorder = append!(copy(dorder), zeros(non_coefficient_hyperparameters))
        # # differentiate by RBF kernel length
        # if dKdθ == length(hyperparameters) - 2
        #     new_dorder[length(dorder) + 1] = 1
        # # differentiate by Periodic kernel length
        # elseif dKdθ == length(hyperparameters) - 1
        #     new_dorder[length(dorder) + 2] = 1
        # # differentiate by Periodic kernel period
        # elseif dKdθ == length(hyperparameters)
        #     new_dorder[length(dorder) + 3] = 1
        # end
        # products = product_rule(new_dorder)


        # things that have been differentiated an even amount of times are symmetric about t1-t2==0
        if isodd(k)
            A_list[k] = covariance(prob_def.kernel, x1list, x2list, kernel_hyperparameters; dorder=dorder, symmetric=true, dKdθ_kernel=dKdθ_kernel)
            # A_list[k] = covariance(x1list, x2list, hyperparameters; dorder=dorder, symmetric=true, dKdθ=dKdθ, products=products)
        else
            A_list[k] = covariance(prob_def.kernel, x1list, x2list, kernel_hyperparameters; dorder=dorder, dKdθ_kernel=dKdθ_kernel)
            # A_list[k] = covariance(x1list, x2list, hyperparameters; dorder=dorder, dKdθ=dKdθ, products=products)
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
    a = reshape(total_hyperparameters[1:num_coefficients], (n_out, n_dif))
    # if we aren't differentiating by one of the coefficient hyperparameters
    # assemble the covariance matrix in the expected way
    if dKdθ_total == 0 || dKdθ_kernel > 0
        for i in 1:n_out
            for j in 1:n_out
                for k in 1:n_dif
                    for l in 1:n_dif
                        if false  # (i == j) & isodd(k + l)
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
        coeff = dif_coefficients(n_out, n_dif, dKdθ_total; coeff_orders=prob_def.coeff_orders, a=a)

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


function covariance(prob_def::Jones_problem_definition, total_hyperparameters::Union{Array{Any,1},Array{Float64,1}}; dKdθ_total::Int=0)
    return covariance(prob_def, prob_def.x_obs, prob_def.x_obs, total_hyperparameters; dKdθ_total=dKdθ_total)
end


"adding measurement noise to K_obs"
function K_observations(kernel_func, x_obs::Array{Float64,1}, measurement_noise::Array{Float64,1}, kernel_hyperparameters::Union{Array{Any,1},Array{Float64,1}}; ignore_asymmetry::Bool=false)
    K_obs = covariance(kernel_func, x_obs, x_obs, kernel_hyperparameters)
    return add_diagonal_term(K_obs, measurement_noise; ignore_asymmetry=ignore_asymmetry)
end


"adding measurement noise to K_obs"
function K_observations(prob_def::Jones_problem_definition, total_hyperparameters::Union{Array{Any,1},Array{Float64,1}}; ignore_asymmetry::Bool=false)
    K_obs = covariance(prob_def, total_hyperparameters)
    return add_diagonal_term(K_obs, measurement_noise; ignore_asymmetry=ignore_asymmetry)
end


"calculating the variance at each GP posterior point"
function get_σ(L_obs, K_obs_samp, K_samp)
# function get_σ(L_obs::Cholesky{Float64,Array{Float64,2}}, K_obs_samp::Array{Float64,2}, K_samp::Array{Float64,2})

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


function covariance_permutations(x_obs::Array{Float64,1}, x_samp::Array{Float64,1}, measurement_noise::Array{Float64,1}, kernel_hyperparameters::Array{Float64,1})
    K_samp = covariance(x_samp, x_samp, kernel_hyperparameters)
    K_obs = K_observations(x_obs, measurement_noise, kernel_hyperparameters)

    K_samp_obs = covariance(x_samp, x_obs, kernel_hyperparameters)
    # K_samp_obs = (K_samp_obs + transpose(dependent_covariance(x_obs, x_samp, hyperparameters)) / 2
    K_obs_samp = transpose(K_samp_obs)
    return K_samp, K_obs, K_samp_obs, K_obs_samp
end


function covariance_permutations(prob_def::Jones_problem_definition, x_samp::Array{Float64,1}, total_hyperparameters::Array{Float64,1})
    K_samp = covariance(prob_def, x_samp, x_samp, total_hyperparameters)
    K_obs = K_observations(prob_def, total_hyperparameters)

    K_samp_obs = covariance(prob_def, x_samp, prob_def.x_obs, total_hyperparameters)
    # K_samp_obs = (K_samp_obs + transpose(covariance(x_obs, x_samp, hyperparameters)) / 2
    K_obs_samp = transpose(K_samp_obs)
    return K_samp, K_obs, K_samp_obs, K_obs_samp
end


function GP_posteriors_from_covariances(K_samp::Union{Symmetric{Float64,Array{Float64,2}},Array{Float64,2}}, K_obs::Union{Symmetric{Float64,Array{Float64,2}},Array{Float64,2}}, K_samp_obs::Union{Symmetric{Float64,Array{Float64,2}},Array{Float64,2}}, K_obs_samp::Union{Transpose{Float64,Array{Float64,2}},Symmetric{Float64,Array{Float64,2}},Array{Float64,2}}; return_σ::Bool=false, return_K::Bool=false, return_L::Bool=false)
    return_vec = []

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


"conditions a GP with data"
function GP_posteriors(x_obs::Array{Float64,1}, x_samp::Array{Float64,1}, measurement_noise::Array{Float64,1}, total_hyperparameters::Array{Float64,1}; return_σ::Bool=false, return_K::Bool=false, return_L::Bool=false)

    (K_samp, K_obs, K_samp_obs, K_obs_samp) = covariance_permutations(x_obs, x_samp, measurement_noise, total_hyperparameters)
    mean_post, return_vec = GP_posteriors_from_covariances(K_samp, K_obs, K_samp_obs, K_obs_samp; return_σ=return_σ, return_K=return_σ, return_L=return_σ)
    return mean_post, return_vec

end


"conditions a GP with data"
function GP_posteriors(prob_def::Jones_problem_definition, x_samp::Array{Float64,1}, total_hyperparameters::Array{Float64,1}; return_σ::Bool=false, return_K::Bool=false, return_L::Bool=false)

    (K_samp, K_obs, K_samp_obs, K_obs_samp) = covariance_permutations(prob_def, x_samp, total_hyperparameters)
    mean_post, return_vec = GP_posteriors_from_covariances(K_samp, K_obs, K_samp_obs, K_obs_samp; return_σ=return_σ, return_K=return_σ, return_L=return_σ)
    return mean_post, return_vec

end


# "creating a Cholesky factorization storage structure"
# struct chol_struct
#     total_hyperparameters::Union{Array{Any,1},Array{Float64,1}}
#     cholesky_object::Cholesky{Float64,Array{Float64,2}}
# end


# # initializing Cholesky factorization storage
# chol_storage = chol_struct(zeros(1), ridge_chol(hcat([1,.1],[.1,1])))


# "An effort to avoid recalculating Cholesky factorizations"
# function stored_chol(chol_stored::chol_struct, new_hyper::Union{Array{Any,1},Array{Float64,1}}, A::Union{Symmetric{Float64,Array{Float64,2}},Array{Float64,2}}; return_values::Bool=false, notification::Bool=true, ridge::Float64=1e-6)
#
#     new_chol = chol_stored.cholesky_object
#     # if the Cholesky factorization wasn't just calculated, calculate a new one.
#     if chol_stored.total_hyperparameters != new_hyper
#         new_chol = ridge_chol(A; notification=notification, ridge=ridge)
#         global chol_storage = chol_struct(copy(new_hyper), new_chol)  # reassigning the gloabl variable
#     end
#     return new_chol
#
# end


"""
find the powers that each coefficient is taken to for each part of the matrix construction
used for constructing differentiated versions of the kernel
"""
function coefficient_orders(n_out::Int, n_dif::Int; a::Union{Array{Float64,2},Array{Any,2}}=ones(1,1))

    if a == ones(1,1)
        a = ones(n_out, n_dif)
    else
        @assert size(a) == (n_out, n_dif)
    end

    coeff_orders = zeros(n_out, n_out, n_dif, n_dif, n_out, n_dif)
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


"""
getting the coefficients for constructing differentiated versions of the kernel
using the powers that each coefficient is taken to for each part of the matrix construction
"""
function dif_coefficients(n_out::Int, n_dif::Int, dKdθ_total::Int; coeff_orders::Array{Float64,6}=zeros(1,1,1,1,1,1), a::Union{Array{Any,2},Array{Float64,2}}=ones(1,1))

    @assert dKdθ_total>0 "Can't get differential coefficients when you aren't differentiating "
    @assert dKdθ_total<=(n_out*n_dif) "Can't get differential coefficients fpr non-coefficient hyperparameters"

    if coeff_orders == zeros(1,1,1,1,1,1)
        coeff_orders = coefficient_orders(n_out, n_dif, a=a)
        # println("found new coeff_orders")
    else
        @assert size(coeff_orders) == (n_out, n_out, n_dif, n_dif, n_out, n_dif)
    end

    # ((output pair), (which A matrix to use), (which a coefficent to use))
    coeff = zeros(n_out, n_out, n_dif, n_dif, n_out, n_dif)

    # "a small function to get indices that make sense from Julia's reshaping routine"
    # proper_index(i::Int) = [convert(Int64, rem(i - 1, n_out)) + 1, convert(Int64, floor((i -1) / n_out)) + 1]
    # proper_index(i::Int) = [((dKdθ_total - 1) % n_out) + 1, div(dKdθ_total - 1, n_out) + 1]

    proper_indices = [((dKdθ_total - 1) % n_out) + 1, div(dKdθ_total - 1, n_out) + 1]
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


# """
# negative log likelihood of the data given the current kernel parameters (as seen on page 19)
# (negative because Optim has a minimizer)
# """
# function nlogL(hyperparameter_list...)
#
#     hyper = []
#     for i in 1:length(hyperparameter_list)
#         append!(hyper, hyperparameter_list[i])
#     end
#     n=length(y_obs)
#
#     # a weirdly necessary dummy variable
#     measurement_noise_dummy = measurement_noise
#     K_obs = K_observations(x_obs, measurement_noise_dummy, hyper; ignore_asymmetry=true)
#     L_fact = stored_chol(chol_storage, hyper, K_obs; notification=false)
#     # inv_K_obs = inv(L_fact)  # ~35% faster than inv(K_obs)
#     # det_K_obs = det(L_fact)  # ~8% faster than det(K_obs)
#
#     # goodness of fit term
#     data_fit = -1 / 2 * (transpose(y_obs) * (L_fact \ y_obs))
#     # complexity penalization term
#     # penalty = -1 / 2 * log(det(L_fact))
#     penalty = -1 / 2 * logdet(L_fact)  # half memory but twice the time
#     # normalization term (functionally useless)
#     normalization = -n / 2 * log(2 * pi)
#
#     return -1 * (data_fit + penalty + normalization)
# end


# """
# negative log likelihood of the data given the current kernel parameters (as seen on page 19)
# (negative because Optim has a minimizer)
# """
# function nlogL_Jones(hyperparameter_list...)
#
#     hyper = []
#     for i in 1:length(hyperparameter_list)
#         append!(hyper, hyperparameter_list[i])
#     end
#
#
#     K_obs = K_observations(problem_definition, hyper; ignore_asymmetry=true)
#     L_fact = stored_chol(chol_storage, hyper, K_obs; notification=false)
#     # inv_K_obs = inv(L_fact)  # ~35% faster than inv(K_obs)
#     # det_K_obs = det(L_fact)  # ~8% faster than det(K_obs)
#
#     y_obs = problem_definition.y_obs
#     n=length(problem_definition.y_obs)
#
#     # goodness of fit term
#     data_fit = -1 / 2 * (transpose(y_obs) * (L_fact \ y_obs))
#     # complexity penalization term
#     # complexity_penalty = -1 / 2 * log(det(L_fact))
#     complexity_penalty = -1 / 2 * logdet(L_fact)  # half memory but twice the time
#     # normalization term (functionally useless)
#     normalization = -n / 2 * log(2 * pi)
#
#     custom_penalty = 0
#     for i in hyper[(end - problem_definition.n_kern_hyper):end]
#         if i < 0
#             custom_penalty -= 10000 * (i ^ 2)
#         end
#     end
#
#     return -1 * (data_fit + complexity_penalty + normalization + custom_penalty)
# end


# """
# http://www.gaussianprocess.org/gpml/chapters/RW5.pdf
# gradient of negative log likelihood of the data given the current kernel parameters (as seen on page 19)
# """
# function ∇nlogL(G, hyperparameter_list...)
#
#     hyper = []
#     for i in 1:length(hyperparameter_list)
#         append!(hyper, hyperparameter_list[i])
#     end
#
#     # a weirdly necessary dummy variable
#     measurement_noise_dummy = measurement_noise
#     K_obs = K_observations(x_obs, measurement_noise_dummy, hyper; ignore_asymmetry=true)
#     L_fact = stored_chol(chol_storage, hyper, K_obs; notification=false)
#     # inv_K_obs = inv(L_fact)
#
#
#     function grad(dK_dθj)
#         # derivative of goodness of fit term
#         data_fit = 1 / 2 * (transpose(y_obs) * (L_fact \ (dK_dθj * (L_fact \ y_obs))))
#         # derivative of complexity penalization term
#         penalty = -1 / 2 * tr(L_fact \ dK_dθj)
#         return -1 * (data_fit + penalty)
#     end
#
#     # taking some burden off of recalculating the coefficient orders used by the automatic differentiation
#     coeff_orders = coefficient_orders(n_out, n_dif)
#
#     for i in 1:(length(hyper))
#         G[i] = grad(dependent_covariance(x_obs, x_obs, hyper; dKdθ=i, coeff_orders=coeff_orders))
#     end
#
# end


# """
# http://www.gaussianprocess.org/gpml/chapters/RW5.pdf
# gradient of negative log likelihood of the data given the current kernel parameters (as seen on page 19)
# """
# function ∇nlogL_Jones(G, hyperparameter_list...)
#
#     hyper = []
#     for i in 1:length(hyperparameter_list)
#         append!(hyper, hyperparameter_list[i])
#     end
#
#     K_obs = K_observations(problem_definition, hyper; ignore_asymmetry=true)
#     L_fact = stored_chol(chol_storage, hyper, K_obs; notification=false)
#     # inv_K_obs = inv(L_fact)
#     y_obs = problem_definition.y_obs
#
#
#     function grad(dK_dθj::Union{Array{Float64,2},Symmetric{Float64,Array{Float64,2}}})
#         # derivative of goodness of fit term
#         data_fit = 1 / 2 * (transpose(y_obs) * (L_fact \ (dK_dθj * (L_fact \ y_obs))))
#         # derivative of complexity penalization term
#         complexity_penalty = -1 / 2 * tr(L_fact \ dK_dθj)
#
#         custom_penalty = 0
#         for i in hyper[(end - problem_definition.n_kern_hyper):end]
#             if i < 0
#                 custom_penalty -= 2 * 10000 * i
#             end
#         end
#
#         return -1 * (data_fit + complexity_penalty + custom_penalty)
#     end
#
#
#     for i in 1:(length(hyper))
#         G[i] = grad(covariance(problem_definition, hyper; dKdθ_total=i))
#         # G[i] = grad(covariance(problem_definition, hyper; dKdθ_total=i))
#     end
#
# end


function nlogL_Jones_fg!(F, G, hyperparameter_list...)

    non_zero_hyperparameters = []
    for i in 1:length(hyperparameter_list)
        append!(non_zero_hyperparameters, hyperparameter_list[i])
    end

    # this allows us to prevent the optimizer from seeing the constant zero coefficients
    total_hyperparameters = reconstruct_total_hyperparameters(problem_definition, non_zero_hyperparameters)

    K_obs = K_observations(problem_definition, total_hyperparameters; ignore_asymmetry=true)
    # L_fact = stored_chol(chol_storage, hyper, K_obs; notification=false)
    L_fact = ridge_chol(K_obs)
    # inv_K_obs = inv(L_fact)

    y_obs = problem_definition.y_obs
    L_fact_solve_y_obs = L_fact \ y_obs

    penalty_lower = 0
    penalty_upper = 10
    penalty_amplitude = 10000

    if G != nothing

        function GP_grad(dK_dθj::Union{Array{Float64,2},Symmetric{Float64,Array{Float64,2}}})
            # derivative of goodness of fit term
            data_fit = 1 / 2 * (transpose(y_obs) * (L_fact \ (dK_dθj * (L_fact_solve_y_obs))))
            # derivative of complexity penalization term
            complexity_penalty = -1 / 2 * tr(L_fact \ dK_dθj)

            custom_penalty = 0

            for i in total_hyperparameters[(end - problem_definition.n_kern_hyper):end]
                if i < penalty_lower
                    custom_penalty -= 2 * penalty_amplitude * (i - penalty_lower)
                # elseif i > penalty_upper
                #     custom_penalty += 2 * penalty_amplitude * (i - penalty_upper)
                end
            end

            return -1 * (data_fit + complexity_penalty + custom_penalty)
        end

        j = 1
        for i in 1:(length(total_hyperparameters))
            if total_hyperparameters[i]!=0
                G[j] = GP_grad(covariance(problem_definition, total_hyperparameters; dKdθ_total=i))
                j += 1
            end
            # G[i] = grad(covariance(problem_definition, hyper; dKdθ_total=i))
        end

    end
    if F != nothing

        n=length(problem_definition.y_obs)

        # goodness of fit term
        data_fit = -1 / 2 * (transpose(y_obs) * (L_fact_solve_y_obs))
        # complexity penalization term
        # complexity_penalty = -1 / 2 * log(det(L_fact))
        complexity_penalty = -1 / 2 * logdet(L_fact)  # half memory but twice the time
        # normalization term (functionally useless)
        normalization = -n / 2 * log(2 * pi)

        custom_penalty = 0
        for i in total_hyperparameters[(end - problem_definition.n_kern_hyper):end]
            if i < penalty_lower
                custom_penalty -= penalty_amplitude * ((i - penalty_lower) ^ 2)
            # elseif i > penalty_upper
            #     custom_penalty += penalty_amplitude * ((i - penalty_upper) ^ 2)
            end
        end

        return -1 * (data_fit + complexity_penalty + normalization + custom_penalty)
    end
end


""
function reconstruct_total_hyperparameters(prob_def::Jones_problem_definition, non_zero_hyperparameters::Union{Array{Any,1},Array{Float64,1}})
    new_coeff_array = reconstruct_array(non_zero_hyperparameters[1:end - prob_def.n_kern_hyper], prob_def.a0)
    coefficient_hyperparameters = collect(Iterators.flatten(new_coeff_array))
    total_hyperparameters = append!(coefficient_hyperparameters, non_zero_hyperparameters[end - problem_definition.n_kern_hyper + 1:end])
    @assert length(total_hyperparameters)==(prob_def.n_kern_hyper + length(prob_def.a0))
    return total_hyperparameters
end
