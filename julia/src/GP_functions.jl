# these functions are directly related to calculating GP quantities
using SpecialFunctions
using LinearAlgebra
using Test


"""
A structure that holds all of the relevant information for doing the analysis in
the Jones et al. 2017+ paper (https://arxiv.org/pdf/1711.01318.pdf).
"""
struct Jones_problem_definition{T1<:Real, T2<:Real, T3<:Real, T4<:Real, T5<:Real}
    kernel::Function  # kernel function
    n_kern_hyper::Integer  # amount of hyperparameters for the kernel function
    n_dif::Integer  # amount of times you are differenting the base kernel
    n_out::Integer  # amount of scores you are jointly modelling
    x_obs::Array{T1,1} # the observation times/phases
    x_obs_units::String  # the units of x_bs
    y_obs::Array{T2,1}  # the flattened, observed data
    y_obs_units::String  # the units of y_obs
    noise::Array{T3,1}  # the measurement noise at all observations
    a0::Array{T4,2}  # the meta kernel coefficients
    # The powers that each a0 coefficient
    # is taken to for each part of the matrix construction
    # used for constructing differentiated versions of the kernel
    coeff_orders::Array{T5,6}
end

"Ensure that Jones_problem_definition is constructed correctly"
function build_problem_definition(kernel_func::Function, num_kernel_hyperparameters::Integer, n_dif::Integer, n_out::Integer, x_obs::Array{T1,1}, x_obs_units::String, y_obs::Array{T2,1}, y_obs_units::String, noise::Array{T3,1}, a0::Array{T4,2}, coeff_orders::Array{T5,6}) where {T1<:Real, T2<:Real, T3<:Real, T4<:Real, T5<:Real}
    @assert isfinite(kernel_func(ones(num_kernel_hyperparameters), randn(); dorder=zeros(2 + num_kernel_hyperparameters)))  # make sure the kernel is valid by testing a sample input
    @assert n_dif>0
    @assert n_out>0
    @assert (length(x_obs) * n_out) == length(y_obs)
    @assert length(y_obs) == length(noise)
    @assert size(a0) == (n_out, n_dif)
    @assert size(coeff_orders) == (n_out, n_out, n_dif, n_dif, n_out, n_dif)
    return Jones_problem_definition(kernel_func, num_kernel_hyperparameters, n_dif, n_out, x_obs, x_obs_units, y_obs, y_obs_units, noise, a0, coeff_orders)
end

"construct the coefficient_orders for Jones_problem_definition if they weren't passed"
function build_problem_definition(kernel_func::Function, num_kernel_hyperparameters::Integer, n_dif::Integer, n_out::Integer, x_obs::Array{T1,1}, x_obs_units::String, y_obs::Array{T2,1}, y_obs_units::String, noise::Array{T3,1}, a0::Array{T4,2}) where {T1<:Real, T2<:Real, T3<:Real, T4<:Real}
    build_problem_definition(kernel_func, num_kernel_hyperparameters, n_dif, n_out, x_obs, x_obs_units, y_obs, y_obs_units, noise, a0, coefficient_orders(n_out, n_dif, a=a0))
end

"build_problem_definition setting empty values for y_obs and measurement noise"
function build_problem_definition(kernel_func::Function, num_kernel_hyperparameters::Integer, n_dif::Integer, n_out::Integer, x_obs::Array{T1,1}, x_obs_units::String, a0::Array{T2,2}) where {T1<:Real, T2<:Real}
    build_problem_definition(kernel_func, num_kernel_hyperparameters, n_dif, n_out, x_obs, x_obs_units, zeros(length(x_obs) * n_out), "", zeros(length(x_obs) * n_out), a0)
end


"""
The basic kernel function evaluator
t1 and t2 are single time points
kernel_hyperparameters = the hyperparameters for the base kernel (e.g. [kernel_period, kernel_length])
dorder = the amount of derivatives to take wrt t1 and t2
dKdθ_kernel = which hyperparameter to take a derivative wrt
"""
function kernel(kernel_func::Function, kernel_hyperparameters::Union{Array{Any,1},Array{T1,1}}, t1::Union{T2,Array{T3,1}}, t2::Union{T4,Array{T5,1}}; dorder::Array{T6,1}=zeros(2), dKdθ_kernel::Integer=0) where {T1<:Real, T2<:Real, T3<:Real, T4<:Real, T5<:Real, T6<:Real}

    @assert dKdθ_kernel <= length(kernel_hyperparameters) "Asking to differentiate by hyperparameter that the kernel doesn't have"

    dif = (t1 - t2)[1]
    dorder_tot = append!(copy(dorder), zeros(length(kernel_hyperparameters)))

    if dKdθ_kernel > 0
        dorder_tot[2 + dKdθ_kernel] = 1
    end

    return kernel_func(kernel_hyperparameters, dif; dorder=dorder_tot)

end

kernel(prob_def::Jones_problem_definition, kernel_hyperparameters, t1, t2; dorder=zeros(2), dKdθ_kernel=0) = kernel(prob_def.kernel, kernel_hyperparameters, t1, t2; dorder=dorder, dKdθ_kernel=dKdθ_kernel)


"""
Creates the covariance matrix by evaluating the kernel function for each pair of passed inputs
symmetric = a parameter stating whether the covariance is guarunteed to be symmetric about the diagonal
"""
function covariance(kernel_func::Function, x1list::Union{Array{T1,1},Array{T2,2}}, x2list::Union{Array{T3,1},Array{T4,2}}, kernel_hyperparameters::Union{Array{Any,1},Array{T5,1}}; dorder::Array{T6,1}=[0, 0], symmetric::Bool=false, dKdθ_kernel::Integer=0) where {T1<:Real, T2<:Real, T3<:Real, T4<:Real, T5<:Real, T6<:Real}

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
function covariance(prob_def::Jones_problem_definition, x1list::Union{Array{T1,1},Array{T2,2}}, x2list::Union{Array{T3,1},Array{T4,2}}, total_hyperparameters::Union{Array{Any,1},Array{T5,1}}; dKdθ_total::Integer=0, chol::Bool=false) where {T1<:Real, T2<:Real, T3<:Real, T4<:Real, T5<:Real}

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
        total_time_derivatives = k - 1
        dorder = [2 * div(total_time_derivatives - 1, 2), rem(total_time_derivatives - 1, 2) + 1]

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
    return symmetric_A(K; chol=chol)

end

function covariance(prob_def::Jones_problem_definition, total_hyperparameters::Union{Array{Any,1},Array{T,1}}; dKdθ_total::Integer=0) where {T<:Real}
    return covariance(prob_def, prob_def.x_obs, prob_def.x_obs, total_hyperparameters; dKdθ_total=dKdθ_total)
end


"adding measurement noise to K_obs"
function K_observations(kernel_func::Function, x_obs::Array{T1,1}, measurement_noise::Array{T2,1}, kernel_hyperparameters::Union{Array{Any,1},Array{T3,1}}; ignore_asymmetry::Bool=false) where {T1<:Real, T2<:Real, T3<:Real}
    K_obs = covariance(kernel_func, x_obs, x_obs, kernel_hyperparameters)
    return symmetric_A(K_obs + Diagonal(measurement_noise); ignore_asymmetry=ignore_asymmetry, chol=true)
end

"adding measurement noise to K_obs"
function K_observations(prob_def::Jones_problem_definition, total_hyperparameters::Union{Array{Any,1},Array{T,1}}; ignore_asymmetry::Bool=false) where {T<:Real}
    K_obs = covariance(prob_def, total_hyperparameters)
    return symmetric_A(K_obs + Diagonal(prob_def.noise); ignore_asymmetry=ignore_asymmetry, chol=true)
end


"calculating the standard deviation at each GP posterior point. Algorithm from RW alg. 2.1"
function get_σ(L_obs::LowerTriangular{T1,Array{T1,2}}, K_obs_samp::Union{Transpose{T2,Array{T2,2}},Symmetric{T3,Array{T3,2}},Array{T4,2}}, diag_K_samp::Array{T5,1}) where {T1<:Real, T2<:Real, T3<:Real, T4<:Real, T5<:Real}
    v = L_obs \ K_obs_samp
    return sqrt.(diag_K_samp - [dot(v[:, i], v[:, i]) for i in 1:length(diag_K_samp)])  # σ
end

function get_σ(prob_def::Jones_problem_definition, x_samp::Array{T1,1}, total_hyperparameters::Array{T2,1}) where {T1<:Real, T2<:Real}
    (K_samp, K_obs, K_samp_obs, K_obs_samp) = covariance_permutations(prob_def, x_samp, total_hyperparameters)
    return get_σ(ridge_chol(K_obs).L, K_obs_samp, diag(K_samp))
end


"calcuate all of the different versions of the covariance matrices for measured and sampled points"
function covariance_permutations(x_obs::Array{T1,1}, x_samp::Array{T2,1}, measurement_noise::Array{T3,1}, kernel_hyperparameters::Array{T4,1}) where {T1<:Real, T2<:Real, T3<:Real, T4<:Real}
    K_samp = covariance(x_samp, x_samp, kernel_hyperparameters)
    K_obs = K_observations(x_obs, measurement_noise, kernel_hyperparameters)
    K_samp_obs = covariance(x_samp, x_obs, kernel_hyperparameters)
    K_obs_samp = transpose(K_samp_obs)
    return K_samp, K_obs, K_samp_obs, K_obs_samp
end

"calcuate all of the different versions of the covariance matrices for measured and sampled points"
function covariance_permutations(prob_def::Jones_problem_definition, x_samp::Array{T1,1}, total_hyperparameters::Array{T2,1}) where {T1<:Real, T2<:Real}
    K_samp = covariance(prob_def, x_samp, x_samp, total_hyperparameters)
    K_obs = K_observations(prob_def, total_hyperparameters)
    K_samp_obs = covariance(prob_def, x_samp, prob_def.x_obs, total_hyperparameters)
    K_obs_samp = transpose(K_samp_obs)
    return K_samp, K_obs, K_samp_obs, K_obs_samp
end


"Condition the GP on data"
function GP_posteriors_from_covariances(y_obs::Array{T1,1}, K_samp::Union{Cholesky{T2,Array{T2,2}},Symmetric{T3,Array{T3,2}},Array{T4,2}}, K_obs::Cholesky{T5,Array{T5,2}}, K_samp_obs::Union{Symmetric{T6,Array{T6,2}},Array{T7,2}}, K_obs_samp::Union{Transpose{T8,Array{T8,2}},Symmetric{T9,Array{T9,2}},Array{T10,2}}; return_σ::Bool=false, return_K::Bool=true, chol::Bool=false) where {T1<:Real, T2<:Real, T3<:Real, T4<:Real, T5<:Real, T6<:Real, T7<:Real, T8<:Real, T9<:Real, T10<:Real}

    # posterior mean calcuation from RW alg. 2.1

    # these are all equivalent but have different computational costs
    # α = inv(K_obs) * y_obs
    # α = transpose(L) \ (L \ y_obs)
    α = K_obs \ y_obs

    mean_post = K_samp_obs * α

    # posterior standard deviation calcuation from RW alg. 2.1
    σ = get_σ(K_obs.L, K_obs_samp, diag(K_samp))

    # posterior covariance calculation is from eq. 2.24 of RW
    if return_K
        K_post = symmetric_A(K_samp - (K_samp_obs * (K_obs \ K_obs_samp)), chol=chol)
        return mean_post, σ, K_post
    else
        return mean_post, σ
    end

end

function GP_posteriors(x_obs::Array{T1,1}, y_obs::Array{T2,1}, x_samp::Array{T3,1}, measurement_noise::Array{T4,1}, total_hyperparameters::Array{T5,1}; return_K::Bool=true, chol::Bool=false) where {T1<:Real, T2<:Real, T3<:Real, T4<:Real, T5<:Real}
    (K_samp, K_obs, K_samp_obs, K_obs_samp) = covariance_permutations(x_obs, x_samp, measurement_noise, total_hyperparameters)
    return GP_posteriors_from_covariances(y_obs, K_samp, K_obs, K_samp_obs, K_obs_samp; return_K=return_K, chol=chol)
end

function GP_posteriors(prob_def::Jones_problem_definition, x_samp::Array{T1,1}, total_hyperparameters::Array{T2,1}; return_K::Bool=true, chol::Bool=false) where {T1<:Real, T2<:Real}
    (K_samp, K_obs, K_samp_obs, K_obs_samp) = covariance_permutations(prob_def, x_samp, total_hyperparameters)
    return GP_posteriors_from_covariances(prob_def.y_obs, K_samp, K_obs, K_samp_obs, K_obs_samp; return_K=return_K, chol=chol)
end


"""
find the powers that each Jones coefficient is taken to for each part of the
matrix construction. Used for constructing differentiated versions of the kernel
"""
function coefficient_orders(n_out::Integer, n_dif::Integer; a::Union{Array{T,2},Array{Any,2}}=ones(1,1)) where {T<:Real}

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
Getting the coefficients for constructing differentiated versions of the kernel
using the powers that each coefficient is taken to for each part of the matrix construction
"""
function dif_coefficients(n_out::Integer, n_dif::Integer, dKdθ_total::Integer; coeff_orders::Array{T1,6}=zeros(1,1,1,1,1,1), a::Union{Array{Any,2},Array{T2,2}}=ones(1,1)) where {T1<:Real, T2<:Real}

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
    # proper_index(i::Integer) = [convert(Int64, rem(i - 1, n_out)) + 1, convert(Int64, floor((i -1) / n_out)) + 1]
    # proper_index(i::Integer) = [((dKdθ_total - 1) % n_out) + 1, div(dKdθ_total - 1, n_out) + 1]

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


"Calculates the quantities shared by the LogL and ∇LogL calculations"
function calculate_shared_nLogL_Jones(prob_def::Jones_problem_definition, non_zero_hyperparameters::Union{Array{Any,1},Array{T1,1}}; y_obs::Array{T2,1}=zeros(1)) where {T1<:Real, T2<:Real}

    # this allows us to prevent the optimizer from seeing the constant zero coefficients
    total_hyperparameters = reconstruct_total_hyperparameters(prob_def, non_zero_hyperparameters)

    K_obs = K_observations(prob_def, total_hyperparameters; ignore_asymmetry=true)

    if y_obs == zeros(1)
        y_obs = prob_def.y_obs
    end

    α = K_obs \ y_obs

    prior_alpha = 10.
    prior_beta = 10.
    prior_params = (prior_alpha, prior_beta)

    return total_hyperparameters, K_obs, y_obs, α, prior_params

end


"generic GP nLogL"
function nlogL(K_obs::Cholesky{T1,Array{T1,2}}, y_obs::Array{T2,1}, α::Array{T3,1}) where {T1<:Real, T2<:Real, T3<:Real}

    n = length(y_obs)

    # goodness of fit term
    data_fit = -1 / 2 * (transpose(y_obs) * (α))
    # complexity penalization term
    # complexity_penalty = -1 / 2 * log(det(K_obs))
    complexity_penalty = -1 / 2 * logdet(K_obs)  # half memory but twice the time
    # normalization term (functionally useless)
    normalization = -n / 2 * log(2 * pi)

    return -1 * (data_fit + complexity_penalty + normalization)

end

nlogL(K_obs, y_obs) = nlogL(K_obs, y_obs, K_obs \ y_obs)


"nlogL for Jones GP"
function nlogL_Jones(prob_def::Jones_problem_definition, total_hyperparameters::Array{T1,1}, K_obs::Cholesky{T2,Array{T2,2}}, y_obs::Array{T3,1}, α::Array{T4,1}, prior_params::Tuple) where {T1<:Real, T2<:Real, T3<:Real, T4<:Real}

    nLogL_val = nlogL(K_obs, y_obs, α)

    prior_alpha, prior_beta = prior_params

    # adding prior for physical length scales
    prior_term = 0
    for i in 1:prob_def.n_kern_hyper
        kernel_length = total_hyperparameters[end + 1 - i]
        prior_term += log_inverse_gamma(kernel_length, prior_alpha, prior_beta)
    end

    return nLogL_val - prior_term

end

function nlogL_Jones(prob_def::Jones_problem_definition, total_hyperparameters::Array{T1,1}; y_obs::Array{T2,1}=zeros(1)) where {T1<:Real, T2<:Real}
    non_zero_hyperparameters = total_hyperparameters[findall(!iszero, total_hyperparameters)]
    total_hyperparameters, K_obs, y_obs, α, prior_params = calculate_shared_nLogL_Jones(prob_def, non_zero_hyperparameters, y_obs=y_obs)
    return nlogL_Jones(prob_def, total_hyperparameters, K_obs, y_obs, α, prior_params)
end


"Partial derivative of GP nLogL w.r.t. a given hyperparameter"
function dnlogLdθ(dK_dθj::Union{Array{T1,2},Symmetric{T2,Array{T2,2}}}, K_obs::Cholesky{T3,Array{T3,2}}, y_obs::Array{T4,1}, α::Array{T5,1}) where {T1<:Real, T2<:Real, T3<:Real, T4<:Real, T5<:Real}

    # derivative of goodness of fit term
    data_fit = 1 / 2 * (transpose(y_obs) * (K_obs \ (dK_dθj * (α))))
    # derivative of complexity penalization term
    complexity_penalty = -1 / 2 * tr(K_obs \ dK_dθj)

    return -1 * (data_fit + complexity_penalty)

end


"Replaces G with gradient of nLogL for non-zero hyperparameters"
function ∇nlogL_Jones!(G::Array{T1,1}, prob_def::Jones_problem_definition, total_hyperparameters::Array{T2,1}, K_obs::Cholesky{T3,Array{T3,2}}, y_obs::Array{T4,1}, α::Array{T5,1}, prior_params::Tuple) where {T1<:Real, T2<:Real, T3<:Real, T4<:Real, T5<:Real}

    j = 1
    for i in 1:(length(total_hyperparameters))
        if total_hyperparameters[i]!=0
            G[j] = dnlogLdθ(covariance(prob_def, total_hyperparameters; dKdθ_total=i), K_obs, y_obs, α)
            j += 1
        end
    end

    prior_alpha, prior_beta = prior_params

    # adding prior for physical length scales
    for i in 1:prob_def.n_kern_hyper
        kernel_length = total_hyperparameters[end + 1 - i]
        prior_term = dlog_inverse_gamma(kernel_length, prior_alpha, prior_beta)
        G[end + 1 - i] -= prior_term
    end

end


"Returns gradient of nLogL for non-zero hyperparameters"
function ∇nlogL_Jones(prob_def::Jones_problem_definition, total_hyperparameters::Array{T1,1}, K_obs::Cholesky{T2,Array{T2,2}}, y_obs::Array{T3,1}, α::Array{T4,1}, prior_params::Tuple) where {T1<:Real, T2<:Real, T3<:Real, T4<:Real}
    G = zeros(length(total_hyperparameters[findall(!iszero, total_hyperparameters)]))
    ∇nlogL_Jones!(G, prob_def, total_hyperparameters, K_obs, y_obs, α, prior_params)
    return G
end


"Replaces G with gradient of nLogL for non-zero hyperparameters"
function ∇nlogL_Jones!(G::Array{T1,1}, prob_def::Jones_problem_definition, total_hyperparameters::Array{T2,1}; y_obs::Array{T3,1}=zeros(1)) where {T1<:Real, T2<:Real, T3<:Real}
    non_zero_hyperparameters = total_hyperparameters[findall(!iszero, total_hyperparameters)]
    total_hyperparameters, K_obs, y_obs, α, prior_params = calculate_shared_nLogL_Jones(prob_def, non_zero_hyperparameters; y_obs=y_obs)
    ∇nlogL_Jones!(G, prob_def, total_hyperparameters, K_obs, y_obs, α, prior_params)
end


"Returns gradient of nLogL for non-zero hyperparameters"
function ∇nlogL_Jones(prob_def::Jones_problem_definition, total_hyperparameters::Array{T1,1}; y_obs::Array{T2,1}=zeros(1)) where {T1<:Real, T2<:Real}
    G = zeros(length(total_hyperparameters[findall(!iszero, total_hyperparameters)]))
    ∇nlogL_Jones!(G, prob_def, total_hyperparameters; y_obs=y_obs)
    return G
end


"reinsert the zero coefficients into the non-zero hyperparameter list"
function reconstruct_total_hyperparameters(prob_def::Jones_problem_definition, non_zero_hyperparameters::Union{Array{Any,1},Array{T,1}}) where {T<:Real}

    new_coeff_array = reconstruct_array(non_zero_hyperparameters[1:end - prob_def.n_kern_hyper], prob_def.a0)

    coefficient_hyperparameters = collect(Iterators.flatten(new_coeff_array))
    total_hyperparameters = append!(coefficient_hyperparameters, non_zero_hyperparameters[end - prob_def.n_kern_hyper + 1:end])
    @assert length(total_hyperparameters)==(prob_def.n_kern_hyper + length(prob_def.a0))

    return total_hyperparameters

end


"imports the specified kernel and returns the number of hyperparameters it uses"
function include_kernel(kernel_name::String)
    try
        return include("src/kernels/$kernel_name.jl")
    catch
        return include("../src/kernels/$kernel_name.jl")
    end
end
