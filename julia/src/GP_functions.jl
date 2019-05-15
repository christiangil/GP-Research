# these functions are directly related to calculating GP quantities
using SpecialFunctions
using LinearAlgebra
using Test
using Memoize
using SharedArrays
using Distributed
using JLD2, FileIO
using Dates


"""
A structure that holds all of the relevant information for constructing the
model used in the Jones et al. 2017+ paper (https://arxiv.org/pdf/1711.01318.pdf).
"""
struct Jones_problem_definition{T1<:Real, T2<:Integer}
    kernel::Function  # kernel function
    n_kern_hyper::Integer  # amount of hyperparameters for the kernel function
    n_dif::Integer  # amount of times you are differenting the base kernel
    n_out::Integer  # amount of scores you are jointly modelling
    x_obs::AbstractArray{T1,1} # the observation times/phases
    x_obs_units::AbstractString  # the units of x_bs
    y_obs::AbstractArray{T1,1}  # the flattened, observed data
    y_obs_units::AbstractString  # the units of y_obs
    noise::AbstractArray{T1,1}  # the measurement noise at all observations
    a0::AbstractArray{T1,2}  # the meta kernel coefficients
    # The powers that each a0 coefficient
    # is taken to for each part of the matrix construction
    # used for constructing differentiated versions of the kernel
    coeff_orders::AbstractArray{T2,6}
    coeff_coeffs::AbstractArray{T2,4}
end


"Jones_problem_definition without kernel information"
struct Jones_problem_definition_base{T1<:Real, T2<:Integer}
    n_dif::Integer  # amount of times you are differenting the base kernel
    n_out::Integer  # amount of scores you are jointly modelling
    x_obs::AbstractArray{T1,1} # the observation times/phases
    x_obs_units::AbstractString  # the units of x_bs
    y_obs::AbstractArray{T1,1}  # the flattened, observed data
    y_obs_units::AbstractString  # the units of y_obs
    noise::AbstractArray{T1,1}  # the measurement noise at all observations
    a0::AbstractArray{T1,2}  # the meta kernel coefficients
    # The powers that each a0 coefficient
    # is taken to for each part of the matrix construction
    # used for constructing differentiated versions of the kernel
    coeff_orders::AbstractArray{T2,6}
    coeff_coeffs::AbstractArray{T2,4}
end


"Ensure that the passed problem definition parameters are what we expect them to be"
function check_problem_definition(
    n_dif::Integer,
    n_out::Integer,
    x_obs::AbstractArray{T1,1},
    x_obs_units::AbstractString,
    y_obs::AbstractArray{T2,1},
    y_obs_units::AbstractString,
    noise::AbstractArray{T3,1},
    a0::AbstractArray{T4,2},
    coeff_orders::AbstractArray{T5,6},
    coeff_coeffs::AbstractArray{T5,4}
    ) where {T1<:Real, T2<:Real, T3<:Real, T4<:Real, T5<:Integer}

    @assert n_dif>0
    @assert n_out>0
    @assert (length(x_obs) * n_out) == length(y_obs)
    @assert length(y_obs) == length(noise)
    @assert size(a0) == (n_out, n_dif)
    @assert size(coeff_orders) == (n_out, n_out, n_dif, n_dif, n_out, n_dif)  # maybe unnecessary due to the fact that we construct it
    @assert size(coeff_coeffs) == (n_out, n_out, n_dif, n_dif)  # maybe unnecessary due to the fact that we construct it
end


"Ensure that Jones_problem_definition_base is constructed correctly"
function build_problem_definition(
    n_dif::Integer,
    n_out::Integer,
    x_obs::AbstractArray{T1,1},
    x_obs_units::AbstractString,
    y_obs::AbstractArray{T2,1},
    y_obs_units::AbstractString,
    noise::AbstractArray{T3,1},
    a0::AbstractArray{T4,2},
    coeff_orders::AbstractArray{T5,6},
    coeff_coeffs::AbstractArray{T5,4},
    ) where {T1<:Real, T2<:Real, T3<:Real, T4<:Real, T5<:Integer}

    check_problem_definition(n_dif, n_out, x_obs, x_obs_units, y_obs, y_obs_units, noise, a0, coeff_orders, coeff_coeffs)
    return Jones_problem_definition_base(n_dif, n_out, x_obs, x_obs_units, y_obs, y_obs_units, noise, a0, coeff_orders, coeff_coeffs)
end

"Calculate the coeffficient orders for Jones_problem_definition_base construction if they weren't passed"
function build_problem_definition(
    n_dif::Integer,
    n_out::Integer,
    x_obs::AbstractArray{T1,1},
    x_obs_units::AbstractString,
    y_obs::AbstractArray{T2,1},
    y_obs_units::AbstractString,
    noise::AbstractArray{T3,1},
    a0::AbstractArray{T4,2}
    ) where {T1<:Real, T2<:Real, T3<:Real, T4<:Real}

    coeff_orders, coeff_coeffs = coefficient_orders(n_out, n_dif, a=a0)
    return build_problem_definition(n_dif, n_out, x_obs, x_obs_units, y_obs, y_obs_units, noise, a0, coeff_orders, coeff_coeffs)
end

"build_problem_definition setting empty values for y_obs and measurement noise"
build_problem_definition(
    n_dif::Integer,
    n_out::Integer,
    x_obs::AbstractArray{T1,1},
    x_obs_units::AbstractString,
    a0::AbstractArray{T2,2}
    ) where {T1<:Real, T2<:Real} = build_problem_definition(n_dif, n_out, x_obs, x_obs_units, zeros(length(x_obs) * n_out), "", zeros(length(x_obs) * n_out), a0)

"Construct Jones_problem_definition by adding kernel information to Jones_problem_definition_base"
function build_problem_definition(
    kernel_func::Function,
    num_kernel_hyperparameters::Integer,
    prob_def_base::Jones_problem_definition_base)

    @assert isfinite(kernel_func(ones(num_kernel_hyperparameters), randn(); dorder=zeros(Int64, 2 + num_kernel_hyperparameters)))  # make sure the kernel is valid by testing a sample input
    check_problem_definition(prob_def_base.n_dif, prob_def_base.n_out, prob_def_base.x_obs, prob_def_base.x_obs_units, prob_def_base.y_obs, prob_def_base.y_obs_units, prob_def_base.noise, prob_def_base.a0, prob_def_base.coeff_orders, prob_def_base.coeff_coeffs)  # might be unnecessary
    return Jones_problem_definition(kernel_func, num_kernel_hyperparameters, prob_def_base.n_dif, prob_def_base.n_out, prob_def_base.x_obs, prob_def_base.x_obs_units, prob_def_base.y_obs, prob_def_base.y_obs_units, prob_def_base.noise, prob_def_base.a0, prob_def_base.coeff_orders, prob_def_base.coeff_coeffs)
end


"""
The basic kernel function evaluator
t1 and t2 are single time points
kernel_hyperparameters = the hyperparameters for the base kernel (e.g. [kernel_period, kernel_length])
dorder = the amount of derivatives to take wrt t1 and t2
dKdθ_kernel = which hyperparameter to take a derivative wrt
"""
function kernel(
    kernel_func::Function,
    kernel_hyperparameters::AbstractArray{T1,1},
    t1::Real,
    t2::Real;
    dorder::AbstractArray{T2,1}=zeros(Int64, 2),
    dKdθs_kernel::AbstractArray{T3,1}=Int64[]
    ) where {T1<:Real, T2<:Real, T3<:Integer}

    @assert all(dKdθs_kernel .<= length(kernel_hyperparameters)) "Asking to differentiate by hyperparameter that the kernel doesn't have"
    @assert length(dKdθs_kernel) <= 2 "Only two kernel hyperparameter derivatives are currently supported"

    dif = (t1 - t2)
    dorder_tot = append!(copy(dorder), zeros(Int64, length(kernel_hyperparameters)))

    for dKdθ_kernel in dKdθs_kernel
        if dKdθ_kernel > 0; dorder_tot[2 + dKdθ_kernel] += 1 end
    end

    return kernel_func(kernel_hyperparameters, dif; dorder=dorder_tot)

end

kernel(
    prob_def::Jones_problem_definition,
    kernel_hyperparameters,
    t1,
    t2;
    dorder=zeros(Int64, 2),
    dKdθs_kernel=Int64[]
    ) = kernel(prob_def.kernel, kernel_hyperparameters, t1, t2; dorder=dorder, dKdθs_kernel=dKdθs_kernel)


"""
Creates the covariance matrix by evaluating the kernel function for each pair of passed inputs
symmetric = a parameter stating whether the covariance is guarunteed to be symmetric about the diagonal
"""
function covariance!(
    K::AbstractArray{T1,2},
    kernel_func::Function,
    x1list::AbstractArray{T2,1},
    x2list::AbstractArray{T3,1},
    kernel_hyperparameters::AbstractArray{T4,1};
    dorder::AbstractArray{T5,1}=zeros(Int64, 2),
    symmetric::Bool=false,
    dKdθs_kernel::AbstractArray{T6,1}=Int64[]
    ) where {T1<:Real, T2<:Real, T3<:Real, T4<:Real, T5<:Integer, T6<:Integer}

    @assert issorted(x1list)

    # are the list of x's passed identical
    same_x = (x1list == x2list)

    # are the x's passed identical and equally spaced
    if same_x
        spacing = x1list[2:end]-x1list[1:end-1]
        equal_spacing = all((spacing .- spacing[1]) .< 1e-8)
    else
        @assert issorted(x2list)
        equal_spacing = false
    end

    x1_length = length(x1list)
    x2_length = length(x2list)

    @assert size(K, 1) == x1_length
    @assert size(K, 2) == x2_length

    if equal_spacing && symmetric
        # this section is so fast, it isn't worth parallelizing
        kernline = zeros(x1_length)
        for i in 1:x1_length
            kernline[i] = kernel(kernel_func, kernel_hyperparameters, x1list[1], x1list[i], dorder=dorder, dKdθs_kernel=dKdθs_kernel)
        end
        for i in 1:x1_length
            K[i, i:end] = kernline[1:(x1_length + 1 - i)]
        end
        K = Symmetric(K)
    elseif same_x && symmetric
        sendto(workers(), kernel_func=kernel_func, kernel_hyperparameters=kernel_hyperparameters, x1list=x1list, dorder=dorder, dKdθs_kernel=dKdθs_kernel)
        @sync @distributed for i in 1:length(x1list)
            for j in 1:length(x1list)
                if i <= j; K[i, j] = kernel(kernel_func, kernel_hyperparameters, x1list[i], x1list[j], dorder=dorder, dKdθs_kernel=dKdθs_kernel) end
            end
        end
        K = Symmetric(K)
    else
        sendto(workers(), kernel_func=kernel_func, kernel_hyperparameters=kernel_hyperparameters, x1list=x1list, x2list=x2list, dorder=dorder, dKdθs_kernel=dKdθs_kernel)
        @sync @distributed for i in 1:length(x1list)
            for j in 1:length(x2list)
                K[i, j] = kernel(kernel_func, kernel_hyperparameters, x1list[i], x2list[j], dorder=dorder, dKdθs_kernel=dKdθs_kernel)
            end
        end
        return K
    end
end

function covariance(
    kernel_func::Function,
    x1list::AbstractArray{T1,1},
    x2list::AbstractArray{T2,1},
    kernel_hyperparameters::AbstractArray{T3,1};
    dorder::AbstractArray{T4,1}=zeros(Int64, 2),
    symmetric::Bool=false,
    dKdθs_kernel::AbstractArray{T5,1}=Int64[]
    ) where {T1<:Real, T2<:Real, T3<:Real, T4<:Integer, T5<:Integer}

    K_share = SharedArray{Float64}(length(x1list), length(x2list))
    return covariance!(K_share, kernel_func, x1list, x2list, kernel_hyperparameters; dorder=dorder, symmetric=symmetric, dKdθs_kernel=dKdθs_kernel)
end

# Calculating the covariance between all outputs for a combination of dependent GPs
# written so that the intermediate K's don't have to be calculated over and over again
@memoize function covariance(
    prob_def::Jones_problem_definition,
    x1list::AbstractArray{T1,1} where T1<:Real,
    x2list::AbstractArray{T2,1} where T2<:Real,
    total_hyperparameters::AbstractArray{T3,1} where T3<:Real;
    dKdθs_total::AbstractArray{T3,1} where T3<:Integer=Int64[],
    chol::Bool=false)

    @assert all(dKdθs_total .>= 0)
    @assert length(total_hyperparameters) == prob_def.n_kern_hyper + length(prob_def.a0)

    num_coefficients = length(total_hyperparameters) - prob_def.n_kern_hyper
    n_out = prob_def.n_out
    n_dif = prob_def.n_dif
    dKdθs_kernel = dKdθs_total .- num_coefficients
    kernel_hyperparameters = total_hyperparameters[(num_coefficients + 1):end]
    # println(length(kernel_hyperparameters))

    # calculating the total size of the multi-output covariance matrix
    x1_length = length(x1list)
    x2_length = length(x2list)
    K = zeros((n_out * x1_length, n_out * x2_length))

    # only calculating each sub-matrix once and using the fact that they should
    # be basically the same if the kernel has been differentiated the same amount of times
    A_list = Array{Any}(nothing, 2 * n_dif - 1)
    for i in 0:(2 * n_dif - 2)

        # CHANGE THIS TO MAKE MORE SENSE WITH NEW KERNEL SCHEME
        # VERY HACKY
        dorder = [rem(i - 1, 2) + 1, 2 * div(i - 1, 2)]

        # things that have been differentiated an even amount of times are symmetric about t1-t2==0
        iseven(i) ? A_list[i + 1] = covariance(prob_def.kernel, x1list, x2list, kernel_hyperparameters; dorder=dorder, symmetric=true, dKdθs_kernel=dKdθs_kernel) : A_list[i + 1] = covariance(prob_def.kernel, x1list, x2list, kernel_hyperparameters; dorder=dorder, dKdθs_kernel=dKdθs_kernel)

    end

    # return the properly negative differentiated A matrix from the list
    # make it negative or not based on how many times it has been differentiated in the x1 direction
    A_mat(k::Integer, l::Integer, A_list) = powers_of_negative_one(l + 1) * A_list[k + l - 1]

    # assembling the coefficient matrix
    a = reshape(total_hyperparameters[1:num_coefficients], (n_out, n_dif))

    coeff_orders = copy(prob_def.coeff_orders)
    coeff_coeffs = copy(prob_def.coeff_coeffs)
    dif_coefficients!(n_out, n_dif, dKdθs_total, coeff_orders, coeff_coeffs)
    for i in 1:n_out
        for j in 1:n_out
            for k in 1:n_dif
                for l in 1:n_dif
                    # if the coefficient for the Jones coefficients is non-zero
                    if coeff_coeffs[i, j, k, l] != 0
                        A_mat_coeff = coeff_coeffs[i, j, k, l]
                        for m in 1:n_out
                            for n in 1:n_dif
                                A_mat_coeff *= a[m, n] ^ coeff_orders[i, j, k, l, m, n]
                            end
                        end
                        K[((i - 1) * x1_length + 1):(i * x1_length),
                            ((j - 1) * x2_length + 1):(j * x2_length)] +=
                            A_mat_coeff * A_mat(k, l, A_list)
                    end
                end
            end
        end
    end

    # return the symmetrized version of the covariance matrix
    # function corrects for numerical errors and notifies us if our matrix isn't
    # symmetric like it should be
    return symmetric_A(K; chol=chol)

end

covariance(
    prob_def::Jones_problem_definition,
    total_hyperparameters::AbstractArray{T1,1};
    dKdθs_total::AbstractArray{T2,1}=Int64[],
    ) where {T1<:Real, T2<:Integer} = covariance(prob_def, prob_def.x_obs, prob_def.x_obs, total_hyperparameters; dKdθs_total=dKdθs_total)


"adding measurement noise to K_obs"
function K_observations(
    kernel_func::Function,
    x_obs::AbstractArray{T1,1},
    measurement_noise::AbstractArray{T2,1},
    kernel_hyperparameters::AbstractArray{T3,1};
    ignore_asymmetry::Bool=false
    ) where {T1<:Real, T2<:Real, T3<:Real}

    K_obs = covariance(kernel_func, x_obs, x_obs, kernel_hyperparameters)
    return symmetric_A(K_obs + Diagonal(measurement_noise); ignore_asymmetry=ignore_asymmetry, chol=true)
end

"adding measurement noise to K_obs"
function K_observations(
    prob_def::Jones_problem_definition,
    total_hyperparameters::AbstractArray{T,1};
    ignore_asymmetry::Bool=false
    ) where {T<:Real}

    K_obs = covariance(prob_def, total_hyperparameters)
    return symmetric_A(K_obs + Diagonal(prob_def.noise); ignore_asymmetry=ignore_asymmetry, chol=true)
end


"calculating the standard deviation at each GP posterior point. Algorithm from RW alg. 2.1"
function get_σ(
    L_obs::LowerTriangular{T1,Array{T1,2}},
    K_obs_samp::Union{Transpose{T2,Array{T2,2}},Symmetric{T3,Array{T3,2}},AbstractArray{T4,2}},
    diag_K_samp::AbstractArray{T5,1}
    ) where {T1<:Real, T2<:Real, T3<:Real, T4<:Real, T5<:Real}

    v = L_obs \ K_obs_samp
    return sqrt.(diag_K_samp - [dot(v[:, i], v[:, i]) for i in 1:length(diag_K_samp)])  # σ
end

function get_σ(
    prob_def::Jones_problem_definition,
    x_samp::AbstractArray{T1,1},
    total_hyperparameters::AbstractArray{T2,1}
    ) where {T1<:Real, T2<:Real}

    (K_samp, K_obs, K_samp_obs, K_obs_samp) = covariance_permutations(prob_def, x_samp, total_hyperparameters)
    return get_σ(ridge_chol(K_obs).L, K_obs_samp, diag(K_samp))
end


"calcuate all of the different versions of the covariance matrices for measured and sampled points"
function covariance_permutations(
    x_obs::AbstractArray{T1,1},
    x_samp::AbstractArray{T2,1},
    measurement_noise::AbstractArray{T3,1},
    kernel_hyperparameters::AbstractArray{T4,1}
    ) where {T1<:Real, T2<:Real, T3<:Real, T4<:Real}

    K_samp = covariance(x_samp, x_samp, kernel_hyperparameters)
    K_obs = K_observations(x_obs, measurement_noise, kernel_hyperparameters)
    K_samp_obs = covariance(x_samp, x_obs, kernel_hyperparameters)
    K_obs_samp = transpose(K_samp_obs)
    return K_samp, K_obs, K_samp_obs, K_obs_samp
end

"calcuate all of the different versions of the covariance matrices for measured and sampled points"
function covariance_permutations(
    prob_def::Jones_problem_definition,
    x_samp::AbstractArray{T1,1},
    total_hyperparameters::AbstractArray{T2,1}
    ) where {T1<:Real, T2<:Real}

    K_samp = covariance(prob_def, x_samp, x_samp, total_hyperparameters)
    K_obs = K_observations(prob_def, total_hyperparameters)
    K_samp_obs = covariance(prob_def, x_samp, prob_def.x_obs, total_hyperparameters)
    K_obs_samp = transpose(K_samp_obs)
    return K_samp, K_obs, K_samp_obs, K_obs_samp
end


"Condition the GP on data"
function GP_posteriors_from_covariances(
    y_obs::AbstractArray{T1,1},
    K_samp::Union{Cholesky{T2,Array{T2,2}},Symmetric{T3,Array{T3,2}},AbstractArray{T4,2}},
    K_obs::Cholesky{T5,Array{T5,2}},
    K_samp_obs::Union{Symmetric{T6,Array{T6,2}},AbstractArray{T7,2}},
    K_obs_samp::Union{Transpose{T8,Array{T8,2}},Symmetric{T9,Array{T9,2}},AbstractArray{T10,2}};
    return_σ::Bool=false,
    return_K::Bool=true,
    chol::Bool=false
    ) where {T1<:Real, T2<:Real, T3<:Real, T4<:Real, T5<:Real, T6<:Real, T7<:Real, T8<:Real, T9<:Real, T10<:Real}

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

function GP_posteriors(
    x_obs::AbstractArray{T1,1},
    y_obs::AbstractArray{T2,1},
    x_samp::AbstractArray{T3,1},
    measurement_noise::AbstractArray{T4,1},
    total_hyperparameters::AbstractArray{T5,1};
    return_K::Bool=true,
    chol::Bool=false
    ) where {T1<:Real, T2<:Real, T3<:Real, T4<:Real, T5<:Real}

    (K_samp, K_obs, K_samp_obs, K_obs_samp) = covariance_permutations(x_obs, x_samp, measurement_noise, total_hyperparameters)
    return GP_posteriors_from_covariances(y_obs, K_samp, K_obs, K_samp_obs, K_obs_samp; return_K=return_K, chol=chol)
end

function GP_posteriors(
    prob_def::Jones_problem_definition,
    x_samp::AbstractArray{T1,1},
    total_hyperparameters::AbstractArray{T2,1};
    return_K::Bool=true,
    chol::Bool=false) where {T1<:Real, T2<:Real}

    (K_samp, K_obs, K_samp_obs, K_obs_samp) = covariance_permutations(prob_def, x_samp, total_hyperparameters)
    return GP_posteriors_from_covariances(prob_def.y_obs, K_samp, K_obs, K_samp_obs, K_obs_samp; return_K=return_K, chol=chol)
end


"""
Find the powers that each Jones coefficient is taken to for each part of the
matrix construction before differentiating by any hyperparameters.

Parameters:

n_out (int): Amount of dimensions being fit
n_dif (int): Amount of GP time derivatives are in the Jones model being used
a (matrix): The coefficients for the Jones model

Returns:
6D matrix: Filled with integers for what power each coefficient is taken to in
    the construction of a given block of the total covariance matrix.
    For example, coeff_orders[1,1,2,3,:,:] would tell you the powers of each
    coefficient (:,:) that are multiplied by the covariance matrix constructed
    by evaluating the partial derivative of the kernel (once by t1 and twice by
    t2) at every pair of time points (2,3) in the construction of the first
    block of the total covariance matrix (1,1)
4D matrix: Filled with ones anywhere that coeff_orders indicates that
    coefficients exists to multiply a given covariance matrix for a given block

"""
function coefficient_orders(
    n_out::Integer,
    n_dif::Integer;
    a::AbstractArray{T,2}=ones(n_out, n_dif)
    ) where {T<:Real}

    @assert size(a) == (n_out, n_dif)

    # ((output pair), (which A matrix to use), (which a coefficent to use))
    coeff_orders = zeros(Int64, n_out, n_out, n_dif, n_dif, n_out, n_dif)
    coeff_coeffs = zeros(Int64, n_out, n_out, n_dif, n_dif)
    for i in 1:n_out
        for j in 1:n_out
            for k in 1:n_dif
                for l in 1:n_dif
                    for m in 1:n_out
                        for n in 1:n_dif
                            if a[m, n] != 0
                                if [m, n] == [i, k]; coeff_orders[i, j, k, l, m, n] += 1 end
                                if [m, n] == [j, l]; coeff_orders[i, j, k, l, m, n] += 1 end
                            end
                        end
                    end
                    # There should be two coefficients being applied to every
                    # matrix. If there are less than two, that means one of the
                    # coefficients was zero, so we should just set them both to
                    # zero
                    if sum(coeff_orders[i, j, k, l, :, :]) != 2
                        coeff_orders[i, j, k, l, :, :] .= 0
                    else
                        coeff_coeffs[i, j, k, l] = 1
                    end
                end
            end
        end
    end

    return coeff_orders, coeff_coeffs

end


"""
Getting the coefficients for constructing differentiated versions of the kernel
using the powers that each coefficient is taken to for each part of the matrix
construction

Parameters:

n_out (int): Amount of dimensions being fit
n_dif (int): Amount of GP time derivatives are in the Jones model being used
dKdθ_total (matrix): The coefficients for the Jones model
coeff_orders (6D matrix): Filled with integers for what power each coefficient
    is taken to in the construction of a given block of the total covariance
    matrix. See coefficient_orders()
coeff_coeffs (4D matrix): Filled with ones and zeros based on which time
    differentiated covariance matrices are added to which blocks

Returns:
Only modifies the passed coeff_orders and coeff_coeffs matrices

"""
function dif_coefficients!(
    n_out::Integer,
    n_dif::Integer,
    dKdθ_total::Integer,
    coeff_orders::AbstractArray{T1,6},
    coeff_coeffs::AbstractArray{T2,4}
    ) where {T1<:Integer, T2<:Integer}

    @assert size(coeff_orders) == (n_out, n_out, n_dif, n_dif, n_out, n_dif)
    @assert size(coeff_orders) == (n_out, n_out, n_dif, n_dif, n_out, n_dif)

    # only do something if a derivative is being taken
    if dKdθ_total > 0 && dKdθ_total <= (n_out*n_dif)
        proper_indices = [((dKdθ_total - 1) % n_out) + 1, div(dKdθ_total - 1, n_out) + 1]
        for i in 1:n_out
            for j in 1:n_out
                for k in 1:n_dif
                    for l in 1:n_dif
                        if coeff_orders[i, j, k, l, proper_indices[1], proper_indices[2]] != 0
                            coeff_coeffs[i, j, k, l] *= coeff_orders[i, j, k, l, proper_indices[1], proper_indices[2]]
                            coeff_orders[i, j, k, l, proper_indices[1], proper_indices[2]] -= 1
                        else
                            coeff_coeffs[i, j, k, l] = 0
                            coeff_orders[i, j, k, l, :, :] .= 0
                        end
                    end
                end
            end
        end
    end
end

function dif_coefficients!(
    n_out::Integer,
    n_dif::Integer,
    dKdθs_total::AbstractArray{T1,1},
    coeff_orders::AbstractArray{T2,6},
    coeff_coeffs::AbstractArray{T3,4}
    ) where {T1<:Integer, T2<:Integer, T3<:Integer}

    for dKdθ_total in dKdθs_total
        dif_coefficients!(n_out, n_dif, dKdθ_total, coeff_orders, coeff_coeffs)
    end
end


# "Calculates the quantities shared by the LogL and ∇LogL calculations"
# can't use docstrings with @memoize macro :(
# TODO rework this
@memoize function calculate_shared_nlogL_Jones(
    prob_def::Jones_problem_definition,
    non_zero_hyperparameters::AbstractArray{T1,1} where T1<:Real;
    y_obs::AbstractArray{T2,1} where T2<:Real=prob_def.y_obs,
    K_obs::Cholesky{T3,Array{T3,2}} where T3<:Real=K_observations(prob_def, reconstruct_total_hyperparameters(prob_def, non_zero_hyperparameters), ignore_asymmetry=true),
    P::Real=0)
# function calculate_shared_nlogL_Jones(prob_def::Jones_problem_definition, non_zero_hyperparameters::AbstractArray{T1,1} where T1<:Real; y_obs::AbstractArray{T2,1}  where T2<:Real=prob_def.y_obs, K_obs::Cholesky{T3,Array{T3,2}}  where T3<:Real=K_observations(prob_def, reconstruct_total_hyperparameters(prob_def, non_zero_hyperparameters); ignore_asymmetry=true))

    # this allows us to prevent the optimizer from seeing the constant zero coefficients
    total_hyperparameters = reconstruct_total_hyperparameters(prob_def, non_zero_hyperparameters)

    # remove the best fit planet with the period given
    # only changes things for a non-zero period
    remove_kepler!(y_obs, prob_def.x_obs, P, K_obs)

    α = K_obs \ y_obs

    return total_hyperparameters, K_obs, y_obs, α

end


"""
GP negative log marginal likelihood (see Algorithm 2.1 in Rasmussen and Williams 2006)

Parameters:

K_obs (Cholesky factorized object): The covariance matrix constructed by
    evaulating the kernel at each pair of observations and adding measurement
    noise.
y_obs (vector): The observations at each time point
α (vector): inv(K_obs) * y_obs

Returns:
float: the negative log marginal likelihood

"""
function nlogL(
    K_obs::Cholesky{T1,Array{T1,2}},
    y_obs::AbstractArray{T2,1},
    α::AbstractArray{T3,1}
    ) where {T1<:Real, T2<:Real, T3<:Real}

    n = length(y_obs)

    # 2 times negative goodness of fit term
    data_fit = transpose(y_obs) * α
    # 2 times negative complexity penalization term
    # complexity_penalty = log(det(K_obs))
    complexity_penalty = logdet(K_obs)  # half memory but twice the time
    # 2 times negative normalization term (doesn't affect fitting)
    normalization = n * log(2 * π)

    return 0.5 * (data_fit + complexity_penalty + normalization)

end

nlogL(K_obs, y_obs) = nlogL(K_obs, y_obs, K_obs \ y_obs)


"""
First partial derivative of the GP negative log marginal likelihood
(see eq. 5.9 in Rasmussen and Williams 2006)

Parameters:

y_obs (vector): The observations at each time point
α (vector): inv(K_obs) * y_obs
β (matrix): inv(K_obs) * dK_dθ where dK_dθ is the partial derivative of the
    covariance matrix K_obs w.r.t. a hyperparameter

Returns:
float: the partial derivative of the negative log marginal likelihood w.r.t. the
    hyperparameter used in the calculation of β

"""
function dnlogLdθ(
    y_obs::AbstractArray{T1,1},
    α::AbstractArray{T2,1},
    β::AbstractArray{T3,2}
    ) where {T1<:Real, T2<:Real, T3<:Real}

    # 2 times negative derivative of goodness of fit term
    data_fit = -(transpose(y_obs) * β * α)
    # 2 times negative derivative of complexity penalization term
    complexity_penalty = tr(β)

    # return -1 / 2 * tr((α * transpose(α) - inv(K_obs)) * dK_dθj)
    return 0.5 * (data_fit + complexity_penalty)

end


"""
Second partial derivative of the GP negative log marginal likelihood.
Calculated with help from rules found on page 7 of the matrix cookbook
(https://www.ics.uci.edu/~welling/teaching/KernelsICS273B/MatrixCookBook.pdf)

Parameters:

y_obs (vector): The observations at each time point
α (vector): inv(K_obs) * y_obs
β1 (matrix): inv(K_obs) * dK_dθ1 where dK_dθ1 is the partial derivative of the
    covariance matrix K_obs w.r.t. a hyperparameter
β2 (matrix): inv(K_obs) * dK_dθ2 where dK_dθ2 is the partial derivative of the
    covariance matrix K_obs w.r.t. another hyperparameter
β12 (matrix): inv(K_obs) * d2K_dθ1dθ2 where d2K_dθ1dθ2 is the partial
    derivative of the covariance matrix K_obs w.r.t. both of the hyperparameters
    being considered

Returns:
float: the partial derivative of the negative log marginal likelihood w.r.t. the
    hyperparameters used in the calculation of β1, β2, and β12

"""
function d2nlogLdθ2(
    y_obs::AbstractArray{T1,1},
    α::AbstractArray{T2,1},
    β1::AbstractArray{T3,2},
    β2::AbstractArray{T4,2},
    β12::AbstractArray{T5,2}
    ) where {T1<:Real, T2<:Real, T3<:Real, T4<:Real, T5<:Real}

    β12mβ2β1 = β12 - β2 * β1

    # 2 times negative second derivative of goodness of fit term
    data_fit = -(transpose(y_obs) * (β12mβ2β1 - β1 * β2) * α)
    # 2 times negative derivative of complexity penalization term
    complexity_penalty = tr(β12mβ2β1)

    return 0.5 * (data_fit + complexity_penalty)

end


"nlogL for Jones GP"
function nlogL_Jones(
    prob_def::Jones_problem_definition,
    total_hyperparameters::AbstractArray{T1,1},
    K_obs::Cholesky{T2,Array{T2,2}},
    y_obs::AbstractArray{T3,1},
    α::AbstractArray{T4,1}
    ) where {T1<:Real, T2<:Real, T3<:Real, T4<:Real}

    nlogL_val = nlogL(K_obs, y_obs, α)

    return nlogL_val

end

function nlogL_Jones(
    prob_def::Jones_problem_definition,
    total_hyperparameters::AbstractArray{T1,1};
    y_obs::AbstractArray{T2,1}=prob_def.y_obs,
    K_obs::Cholesky{T3,Array{T3,2}}=K_observations(prob_def, reconstruct_total_hyperparameters(prob_def, total_hyperparameters); ignore_asymmetry=true),
    P::Real=0
    ) where {T1<:Real, T2<:Real, T3<:Real}

    non_zero_hyperparameters = remove_zeros(total_hyperparameters)
    total_hyperparameters, K_obs, y_obs, α = calculate_shared_nlogL_Jones(prob_def, non_zero_hyperparameters, y_obs=y_obs, K_obs=K_obs, P=P)
    return nlogL_Jones(prob_def, total_hyperparameters, K_obs, y_obs, α)
end


"Replaces G with gradient of nlogL for non-zero hyperparameters"
function ∇nlogL_Jones!(
    G::AbstractArray{T1,1},
    prob_def::Jones_problem_definition,
    total_hyperparameters::AbstractArray{T2,1},
    K_obs::Cholesky{T3,Array{T3,2}},
    y_obs::AbstractArray{T4,1},
    α::AbstractArray{T5,1}
    ) where {T1<:Real, T2<:Real, T3<:Real, T4<:Real, T5<:Real}

    j = 1
    for i in 1:(length(total_hyperparameters))
        if total_hyperparameters[i]!=0
            G[j] = dnlogLdθ(y_obs, α, K_obs \ covariance(prob_def, total_hyperparameters; dKdθs_total=[i]))
            j += 1
        end
    end

end


"Returns gradient of nlogL for non-zero hyperparameters"
function ∇nlogL_Jones(
    prob_def::Jones_problem_definition,
    total_hyperparameters::AbstractArray{T1,1},
    K_obs::Cholesky{T2,Array{T2,2}},
    y_obs::AbstractArray{T3,1},
    α::AbstractArray{T4,1}
    ) where {T1<:Real, T2<:Real, T3<:Real, T4<:Real}

    G = zeros(length(findall(!iszero, total_hyperparameters)))
    ∇nlogL_Jones!(G, prob_def, total_hyperparameters, K_obs, y_obs, α)
    return G
end


"Replaces G with gradient of nlogL for non-zero hyperparameters"
function ∇nlogL_Jones!(
    G::AbstractArray{T1,1},
    prob_def::Jones_problem_definition,
    total_hyperparameters::AbstractArray{T2,1};
    y_obs::AbstractArray{T3,1}=prob_def.y_obs,
    P::Real=0
    ) where {T1<:Real, T2<:Real, T3<:Real}

    non_zero_hyperparameters = remove_zeros(total_hyperparameters)
    total_hyperparameters, K_obs, y_obs, α = calculate_shared_nlogL_Jones(prob_def, non_zero_hyperparameters; y_obs=y_obs, P=P)
    ∇nlogL_Jones!(G, prob_def, total_hyperparameters, K_obs, y_obs, α)
end


"Returns gradient of nlogL for non-zero hyperparameters"
function ∇nlogL_Jones(
    prob_def::Jones_problem_definition,
    total_hyperparameters::AbstractArray{T1,1};
    y_obs::AbstractArray{T2,1}=prob_def.y_obs
    ) where {T1<:Real, T2<:Real}

    G = zeros(length(findall(!iszero, total_hyperparameters)))
    ∇nlogL_Jones!(G, prob_def, total_hyperparameters; y_obs=y_obs)
    return G
end


"Replaces H with Hessian of nlogL for non-zero hyperparameters"
function ∇∇nlogL_Jones!(
    H::AbstractArray{T1,2},
    prob_def::Jones_problem_definition,
    total_hyperparameters::AbstractArray{T2,1},
    K_obs::Cholesky{T3,Array{T3,2}},
    y_obs::AbstractArray{T4,1},
    α::AbstractArray{T5,1}
    ) where {T1<:Real, T2<:Real, T3<:Real, T4<:Real, T5<:Real}

    k = 1
    l = 1
    for i in 1:length(total_hyperparameters)
        if total_hyperparameters[i]!=0
            for j in 1:length(total_hyperparameters)
                if total_hyperparameters[j]!=0
                    H[k, l] = d2nlogLdθ2(y_obs, α, K_obs \ covariance(prob_def, total_hyperparameters; dKdθs_total=[i]), K_obs \ covariance(prob_def, total_hyperparameters; dKdθs_total=[j]), K_obs \ covariance(prob_def, total_hyperparameters; dKdθs_total=[i, j]))
                    l += 1
                end
            end
            l = 1
            k += 1
        end
    end

end


"reinsert the zero coefficients into the non-zero hyperparameter list if needed"
function reconstruct_total_hyperparameters(
    prob_def::Jones_problem_definition,
    hyperparameters::AbstractArray{T,1}
    ) where {T<:Real}

    if length(hyperparameters)!=(prob_def.n_kern_hyper + length(prob_def.a0))
        new_coeff_array = reconstruct_array(hyperparameters[1:end - prob_def.n_kern_hyper], prob_def.a0)
        coefficient_hyperparameters = collect(Iterators.flatten(new_coeff_array))
        total_hyperparameters = append!(coefficient_hyperparameters, hyperparameters[end - prob_def.n_kern_hyper + 1:end])
    else
        total_hyperparameters = copy(hyperparameters)
    end

    @assert length(total_hyperparameters)==(prob_def.n_kern_hyper + length(prob_def.a0))

    return total_hyperparameters

end


"""
Tries to include the specified kernel from common directories
Returns the number of hyperparameters it uses
"""
function include_kernel(kernel_name::AbstractString)
    try
        return include("src/kernels/$kernel_name.jl")
    catch
        return include("../src/kernels/$kernel_name.jl")
    end
end


"""
Make it easy to run the covariance calculations on many processors
Automatically adds as many workers as there are CPU threads minus 2 if none are
active and no number of procs to add is given
"""
function prep_parallel_covariance(
    kernel_name::AbstractString;
    add_procs::Integer=0)

    auto_addprocs(;add_procs=add_procs)
    @everywhere include("src/base_functions.jl")
    sendto(workers(), kernel_name=kernel_name)
    @everywhere include_kernel(kernel_name)
end


"Iitialize an optimize_Jones_model_jld2"
function initialize_optimize_Jones_model_jld2!(
    kernel_name::AbstractString,
    current_params::AbstractArray{T,1}
    ) where {T<:Real}

    current_fit_time = now()
    if isfile("jld2_files/optimize_Jones_model_$kernel_name.jld2")
        @load "jld2_files/optimize_Jones_model_$kernel_name.jld2" initial_time total_fit_time current_params
        @save "jld2_files/optimize_Jones_model_$kernel_name.jld2" initial_time current_fit_time total_fit_time current_params
    else
        initial_time = now()
        total_fit_time = Millisecond(0)
        @save "jld2_files/optimize_Jones_model_$kernel_name.jld2" initial_time current_fit_time total_fit_time current_params
    end
    return current_params
end


"Update an optimize_Jones_model_jld2 with the fit status"
function update_optimize_Jones_model_jld2!(
    kernel_name::AbstractString,
    non_zero_hyper_param)

    @load "jld2_files/optimize_Jones_model_$kernel_name.jld2" initial_time current_fit_time total_fit_time
    total_fit_time += current_fit_time - now()
    current_fit_time = now()
    current_params = data(non_zero_hyper_param)
    @save "jld2_files/optimize_Jones_model_$kernel_name.jld2" initial_time current_fit_time total_fit_time current_params
end


function logGP_prior(
    total_hyperparameters::AbstractArray{T,1};
    alpha::Real=1,
    beta::Real=1
    ) where {T<:Real}

    logprior = 0

    # adding prior for physical length scales
    for i in 1:prob_def.n_kern_hyper
        kernel_length = total_hyperparameters[end + 1 - i]
        logprior += log_inverse_gamma(kernel_length, alpha, beta)
    end

    return logprior

end


function ∇logGP_prior(
    total_hyperparameters::AbstractArray{T,1};
    alpha::Real=1,
    beta::Real=1
    ) where {T<:Real}

    ∇logprior = zero(total_hyperparameters)

    # adding prior for physical length scales
    for i in 1:prob_def.n_kern_hyper
        kernel_length = total_hyperparameters[end + 1 - i]
        ∇logprior[end + 1 - i] -= dlog_inverse_gamma(kernel_length, alpha, beta)
    end

    return ∇logprior

end
