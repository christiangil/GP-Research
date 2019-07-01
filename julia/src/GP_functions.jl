# these functions are directly related to calculating GP quantities
using SpecialFunctions
using LinearAlgebra
using Test
# using Memoize
using SharedArrays
using Distributed
using JLD2, FileIO
using Dates
using Unitful
using UnitfulAstro


"""
A structure that holds all of the relevant information for constructing the
model used in the Jones et al. 2017+ paper (https://arxiv.org/pdf/1711.01318.pdf).
"""
struct Jones_problem_definition{T1<:Real, T2<:Integer}
    kernel::Function  # kernel function
    n_kern_hyper::T2  # amount of hyperparameters for the kernel function
    n_dif::T2  # amount of times you are differenting the base kernel
    n_out::T2  # amount of scores you are jointly modelling
    x_obs::Vector{T1} # the observation times/phases
    x_obs_units::Unitful.FreeUnits  # the units of x_obs
    y_obs::Vector{T1}  # the flattened, observed data
    y_obs_units::Unitful.FreeUnits  # the units of the RV section ogy_obs
    noise::Vector{T1}  # the measurement noise at all observations
    normals::Vector{T1}  # the normalization of each section of y_obs
    a0::Matrix{T1}  # the meta kernel coefficients
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
    x_obs::Vector{T1} # the observation times/phases
    x_obs_units::Unitful.FreeUnits  # the units of x_bs
    y_obs::Vector{T1}  # the flattened, observed data
    y_obs_units::Unitful.FreeUnits  # the units of y_obs
    noise::Vector{T1}  # the measurement noise at all observations
    normals::Vector{T1}  # the normalization of each section of y_obs
    a0::Matrix{T1}  # the meta kernel coefficients
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
    x_obs::Vector{T1},
    x_obs_units::Unitful.FreeUnits,
    y_obs::Vector{T1},
    y_obs_units::Unitful.FreeUnits,
    noise::Vector{T1},
    normals::Vector{T1},
    a0::Matrix{T1},
    coeff_orders::AbstractArray{T2,6},
    coeff_coeffs::AbstractArray{T2,4}
    ) where {T1<:Real, T2<:Integer}

    @assert n_dif>0
    @assert n_out>0
    @assert dimension(x_obs_units) == dimension(u"s")
    @assert dimension(y_obs_units) == dimension(u"m / s")
    @assert (length(x_obs) * n_out) == length(y_obs)
    @assert length(y_obs) == length(noise)
    @assert length(normals) == n_out
    @assert size(a0) == (n_out, n_dif)
    @assert size(coeff_orders) == (n_out, n_out, n_dif, n_dif, n_out, n_dif)  # maybe unnecessary due to the fact that we construct it
    @assert size(coeff_coeffs) == (n_out, n_out, n_dif, n_dif)  # maybe unnecessary due to the fact that we construct it
end


"Ensure that Jones_problem_definition_base is constructed correctly"
function init_problem_definition(
    n_dif::Integer,
    n_out::Integer,
    x_obs::Vector{T1},
    x_obs_units::Unitful.FreeUnits,
    y_obs::Vector{T1},
    y_obs_units::Unitful.FreeUnits,
    noise::Vector{T1},
    normals::Vector{T1},
    a0::Matrix{T1},
    coeff_orders::AbstractArray{T2,6},
    coeff_coeffs::AbstractArray{T2,4},
    ) where {T1<:Real, T2<:Integer}

    check_problem_definition(n_dif, n_out, x_obs, x_obs_units, y_obs, y_obs_units, noise, normals, a0, coeff_orders, coeff_coeffs)
    return Jones_problem_definition_base(n_dif, n_out, x_obs, x_obs_units, y_obs, y_obs_units, noise, normals, a0, coeff_orders, coeff_coeffs)
end

"Calculate the coeffficient orders for Jones_problem_definition_base construction if they weren't passed"
function init_problem_definition(
    n_dif::Integer,
    n_out::Integer,
    x_obs::Vector{T},
    x_obs_units::Unitful.FreeUnits,
    a0::Matrix{T};
    y_obs::Vector{T}=zeros(length(x_obs) * n_out),
    y_obs_units::Unitful.FreeUnits=u"m / s",
    noise::Vector{T}=zeros(length(x_obs) * n_out),
    normals::Vector{T}=ones(n_out)
    ) where {T<:Real}

    coeff_orders, coeff_coeffs = coefficient_orders(n_out, n_dif, a=a0)
    return init_problem_definition(n_dif, n_out, x_obs, x_obs_units, y_obs, y_obs_units, noise, normals, a0, coeff_orders, coeff_coeffs)
end

"Construct Jones_problem_definition by adding kernel information to Jones_problem_definition_base"
function init_problem_definition(
    kernel_func::Function,
    num_kernel_hyperparameters::Integer,
    prob_def_base::Jones_problem_definition_base)

    @assert isfinite(kernel_func(ones(num_kernel_hyperparameters), randn(); dorder=zeros(Int64, 2 + num_kernel_hyperparameters)))  # make sure the kernel is valid by testing a sample input
    check_problem_definition(prob_def_base.n_dif, prob_def_base.n_out, prob_def_base.x_obs, prob_def_base.x_obs_units, prob_def_base.y_obs, prob_def_base.y_obs_units, prob_def_base.noise, prob_def_base.normals, prob_def_base.a0, prob_def_base.coeff_orders, prob_def_base.coeff_coeffs)  # might be unnecessary
    return Jones_problem_definition(kernel_func, num_kernel_hyperparameters, prob_def_base.n_dif, prob_def_base.n_out, prob_def_base.x_obs, prob_def_base.x_obs_units, prob_def_base.y_obs, prob_def_base.y_obs_units, prob_def_base.noise, prob_def_base.normals, prob_def_base.a0, prob_def_base.coeff_orders, prob_def_base.coeff_coeffs)
end


function normalize_problem_definition!(prob_def::Jones_problem_definition)
    n_obs = length(prob_def.x_obs)
    for i in 1:prob_def.n_out
        inds = 1 + (i - 1) * n_obs : i * n_obs
        prob_def.y_obs[inds] .-= mean(prob_def.y_obs[inds])
        prob_def.normals[i] = stdm(prob_def.y_obs[inds], 0)
        prob_def.y_obs[inds] /= prob_def.normals[i]
        prob_def.noise[inds] /= prob_def.normals[i]
    end
end


"""
The basic kernel function evaluator
t1 and t2 are single time points
kernel_hyperparameters = the hyperparameters for the base kernel (e.g. [kernel_period, kernel_length])
dorder = the amount of derivatives to take wrt t1 and t2
dΣdθ_kernel = which hyperparameter to take a derivative wrt
"""
function kernel(
    kernel_func::Function,
    kernel_hyperparameters::Vector{<:Real},
    t1::Real,
    t2::Real;
    dorder::Vector{T}=zeros(Int64, 2),
    dΣdθs_kernel::Vector{T}=Int64[]
    ) where {T<:Integer}

    @assert all(dΣdθs_kernel .<= length(kernel_hyperparameters)) "Asking to differentiate by hyperparameter that the kernel doesn't have"
    @assert length(dΣdθs_kernel) <= 2 "Only two kernel hyperparameter derivatives are currently supported"

    dif = (t1 - t2)
    dorder_tot = append!(copy(dorder), zeros(Int64, length(kernel_hyperparameters)))

    for dΣdθ_kernel in dΣdθs_kernel
        if dΣdθ_kernel > 0; dorder_tot[2 + dΣdθ_kernel] += 1 end
    end

    return kernel_func(kernel_hyperparameters, dif; dorder=dorder_tot)

end

kernel(
    prob_def::Jones_problem_definition,
    kernel_hyperparameters,
    t1,
    t2;
    dorder=zeros(Int64, 2),
    dΣdθs_kernel=Int64[]
    ) = kernel(prob_def.kernel, kernel_hyperparameters, t1, t2; dorder=dorder, dΣdθs_kernel=dΣdθs_kernel)


"""
Creates the covariance matrix by evaluating the kernel function for each pair of passed inputs
symmetric = a parameter stating whether the covariance is guarunteed to be symmetric about the diagonal
"""
function covariance!(
    Σ::DenseArray{T1,2},
    kernel_func::Function,
    x1list::Vector{T1},
    x2list::Vector{T1},
    kernel_hyperparameters::Vector{T1};
    dorder::Vector{T2}=zeros(Int64, 2),
    symmetric::Bool=false,
    dΣdθs_kernel::Vector{T2}=Int64[]
    ) where {T1<:Real, T2<:Integer}

    @assert issorted(x1list)

    # are the list of x's passed identical
    same_x = (x1list == x2list)

    # are the x's passed identical and equally spaced
    if same_x
        spacing = x1list[2:end]-x1list[1:end-1]
        equal_spacing = all((spacing .- spacing[1]) .< (1e-6 * spacing[1]))
    else
        @assert issorted(x2list)
        equal_spacing = false
    end

    x1_length = length(x1list)
    x2_length = length(x2list)

    @assert size(Σ, 1) == x1_length
    @assert size(Σ, 2) == x2_length

    if equal_spacing && symmetric
        # this section is so fast, it isn't worth parallelizing
        kernline = zeros(x1_length)
        for i in 1:x1_length
            kernline[i] = kernel(kernel_func, kernel_hyperparameters, x1list[1], x1list[i], dorder=dorder, dΣdθs_kernel=dΣdθs_kernel)
        end
        for i in 1:x1_length
            Σ[i, i:end] = kernline[1:(x1_length + 1 - i)]
        end
        return Symmetric(Σ)
    else
        covariance!(Σ, kernel_func, x1list, x2list, kernel_hyperparameters, dorder, symmetric, dΣdθs_kernel, same_x, equal_spacing)
    end
end

function covariance!(
    Σ::SharedArray{T1,2},
    kernel_func::Function,
    x1list::Vector{T1},
    x2list::Vector{T1},
    kernel_hyperparameters::Vector{T1},
    dorder::Vector{T2},
    symmetric::Bool,
    dΣdθs_kernel::Vector{T2},
    same_x::Bool,
    equal_spacing::Bool
    ) where {T1<:Real, T2<:Integer}

    x1_length = length(x1list)
    x2_length = length(x2list)

    @assert size(Σ) == (x1_length, x2_length)

    if same_x && symmetric
        sendto(workers(), kernel_func=kernel_func, kernel_hyperparameters=kernel_hyperparameters, x1list=x1list, dorder=dorder, dΣdθs_kernel=dΣdθs_kernel)
        @sync @distributed for i in 1:length(x1list)
            for j in 1:length(x1list)
                if i <= j; Σ[i, j] = kernel(kernel_func, kernel_hyperparameters, x1list[i], x1list[j], dorder=dorder, dΣdθs_kernel=dΣdθs_kernel) end
            end
        end
        return Symmetric(Σ)
    else
        sendto(workers(), kernel_func=kernel_func, kernel_hyperparameters=kernel_hyperparameters, x1list=x1list, x2list=x2list, dorder=dorder, dΣdθs_kernel=dΣdθs_kernel)
        @sync @distributed for i in 1:length(x1list)
            for j in 1:length(x2list)
                Σ[i, j] = kernel(kernel_func, kernel_hyperparameters, x1list[i], x2list[j], dorder=dorder, dΣdθs_kernel=dΣdθs_kernel)
            end
        end
        return Σ
    end
end

function covariance!(
    Σ::Matrix{T1},
    kernel_func::Function,
    x1list::Vector{T1},
    x2list::Vector{T1},
    kernel_hyperparameters::Vector{T1},
    dorder::Vector{T2},
    symmetric::Bool,
    dΣdθs_kernel::Vector{T2},
    same_x::Bool,
    equal_spacing::Bool
    ) where {T1<:Real, T2<:Integer}

    x1_length = length(x1list)
    x2_length = length(x2list)

    @assert size(Σ) == (x1_length, x2_length)

    if same_x && symmetric
        for i in 1:length(x1list)
            for j in 1:length(x1list)
                if i <= j; Σ[i, j] = kernel(kernel_func, kernel_hyperparameters, x1list[i], x1list[j], dorder=dorder, dΣdθs_kernel=dΣdθs_kernel) end
            end
        end
        return Symmetric(Σ)
    else
        for i in 1:length(x1list)
            for j in 1:length(x2list)
                Σ[i, j] = kernel(kernel_func, kernel_hyperparameters, x1list[i], x2list[j], dorder=dorder, dΣdθs_kernel=dΣdθs_kernel)
            end
        end
        return Σ
    end
end


function covariance(
    kernel_func::Function,
    x1list::Vector{T1},
    x2list::Vector{T1},
    kernel_hyperparameters::Vector{T1};
    dorder::Vector{T2}=zeros(Int64, 2),
    symmetric::Bool=false,
    dΣdθs_kernel::Vector{T2}=Int64[]
    ) where {T1<:Real, T2<:Integer}

    if nworkers()>1
        return covariance!(SharedArray{Float64}(length(x1list), length(x2list)), kernel_func, x1list, x2list, kernel_hyperparameters; dorder=dorder, symmetric=symmetric, dΣdθs_kernel=dΣdθs_kernel)
    else
        return covariance!(zeros(length(x1list), length(x2list)), kernel_func, x1list, x2list, kernel_hyperparameters; dorder=dorder, symmetric=symmetric, dΣdθs_kernel=dΣdθs_kernel)
    end

end


"""
Calculating the covariance between all outputs for a combination of dependent GPs
written so that the intermediate Σ's don't have to be calculated over and over again
"""
function covariance(
    prob_def::Jones_problem_definition,
    x1list::Vector{<:Real},
    x2list::Vector{<:Real},
    total_hyperparameters::Vector{<:Real};
    dΣdθs_total::Vector{<:Integer}=Int64[],
    chol::Bool=false)

    @assert all(dΣdθs_total .>= 0)
    @assert length(total_hyperparameters) == prob_def.n_kern_hyper + length(prob_def.a0)

    num_coefficients = length(total_hyperparameters) - prob_def.n_kern_hyper
    n_out = prob_def.n_out
    n_dif = prob_def.n_dif
    dΣdθs_kernel = dΣdθs_total .- num_coefficients
    kernel_hyperparameters = total_hyperparameters[(num_coefficients + 1):end]
    # println(length(kernel_hyperparameters))

    x1_length = length(x1list)
    x2_length = length(x2list)

    if nworkers()>1
        holder = SharedArray{Float64}(length(x1list), length(x2list))
    else
        holder = zeros(length(x1list), length(x2list))
    end

    # only calculating each sub-matrix once and using the fact that they should
    # be basically the same if the kernel has been differentiated the same amount of times
    A_list = Vector{AbstractArray{<:Real,2}}(undef, 2 * n_dif - 1)
    for i in 0:(2 * n_dif - 2)

        # CHANGE THIS TO MAKE MORE SENSE WITH NEW KERNEL SCHEME
        # VERY HACKY
        dorder = [rem(i - 1, 2) + 1, 2 * div(i - 1, 2)]

        # things that have been differentiated an even amount of times are symmetric about t1-t2==0
        if iseven(i)
            A_list[i + 1] = copy(covariance!(holder, prob_def.kernel, x1list, x2list, kernel_hyperparameters; dorder=dorder, symmetric=true, dΣdθs_kernel=dΣdθs_kernel))
        else
            A_list[i + 1] = copy(covariance!(holder, prob_def.kernel, x1list, x2list, kernel_hyperparameters; dorder=dorder, dΣdθs_kernel=dΣdθs_kernel))
        end

    end

    # assembling the coefficient matrix
    a = reshape(total_hyperparameters[1:num_coefficients], (n_out, n_dif))

    # initializing the multi-output covariance matrix
    Σ = zeros((n_out * x1_length, n_out * x2_length))

    coeff_orders = copy(prob_def.coeff_orders)
    coeff_coeffs = copy(prob_def.coeff_coeffs)
    dif_coefficients!(n_out, n_dif, dΣdθs_total, coeff_orders, coeff_coeffs)
    for i in 1:n_out
        for j in 1:n_out
            # if i <= j  # only need to calculate one set of the off-diagonal blocks
                for k in 1:n_dif
                    for l in 1:n_dif
                        # if the coefficient for the Jones coefficients is non-zero
                        if coeff_coeffs[i, j, k, l] != 0
                            # make it negative or not based on how many times it has been differentiated in the x1 direction
                            A_mat_coeff = coeff_coeffs[i, j, k, l] * powers_of_negative_one(l - 1)
                            for m in 1:n_out
                                for n in 1:n_dif
                                    A_mat_coeff *= a[m, n] ^ coeff_orders[i, j, k, l, m, n]
                                end
                            end
                            # add the properly negative differentiated A matrix from the list
                            Σ[((i - 1) * x1_length + 1):(i * x1_length),
                                ((j - 1) * x2_length + 1):(j * x2_length)] +=
                                A_mat_coeff * A_list[k + l - 1]
                        end
                    end
                end
            # end
        end
    end

    # return the symmetrized version of the covariance matrix
    # function corrects for numerical errors and notifies us if our matrix isn't
    # symmetric like it should be
    return symmetric_A(Σ; chol=chol)
    # if x1list == x2list
    #     if chol
    #         return ridge_chol(Symmetric(Σ))
    #     else
    #         return Symmetric(Σ)
    #     end
    # else
    #     return Σ
    # end

end

covariance(
    prob_def::Jones_problem_definition,
    total_hyperparameters::Vector{<:Real};
    dΣdθs_total::Vector{<:Integer}=Int64[]
    ) = covariance(prob_def, prob_def.x_obs, prob_def.x_obs, total_hyperparameters; dΣdθs_total=dΣdθs_total)


"adding measurement noise to Σ_obs"
function Σ_observations(
    kernel_func::Function,
    x_obs::Vector{T},
    measurement_noise::Vector{T},
    kernel_hyperparameters::Vector{T};
    ignore_asymmetry::Bool=false,
    return_both::Bool=false
    ) where {T<:Real}

    Σ = symmetric_A(covariance(kernel_func, x_obs, x_obs, kernel_hyperparameters); ignore_asymmetry=ignore_asymmetry)
    Σ_obs = symmetric_A(Σ + Diagonal(measurement_noise .* measurement_noise); ignore_asymmetry=ignore_asymmetry, chol=true)
    return return_both ? (Σ_obs, Σ) : Σ_obs
end


"adding measurement noise to Σ_obs"
function Σ_observations(
    prob_def::Jones_problem_definition,
    total_hyperparameters::Vector{T};
    ignore_asymmetry::Bool=false,
    return_both::Bool=false
    ) where {T<:Real}

    Σ = symmetric_A(covariance(prob_def, total_hyperparameters); ignore_asymmetry=ignore_asymmetry)
    Σ_obs = symmetric_A(Σ + Diagonal(prob_def.noise .* prob_def.noise); ignore_asymmetry=ignore_asymmetry, chol=true)
    return return_both ? (Σ_obs, Σ) : Σ_obs
end


"calculating the standard deviation at each GP posterior point. Algorithm from RW alg. 2.1"
function get_σ(
    L_obs::LowerTriangular{T,Matrix{T}},
    Σ_obs_samp::Union{Transpose{T,Matrix{T}},Symmetric{T,Matrix{T}},Matrix{T}},
    diag_Σ_samp::Vector{T}
    ) where {T<:Real}

    v = L_obs \ Σ_obs_samp
    return sqrt.(diag_Σ_samp - [dot(v[:, i], v[:, i]) for i in 1:length(diag_Σ_samp)])  # σ
    # return sqrt.(diag_Σ_samp - diag(Σ_samp_obs * (Σ_obs \ Σ_obs_samp)))  # much slower
end

function get_σ(
    prob_def::Jones_problem_definition,
    x_samp::Vector{T},
    total_hyperparameters::Vector{T}
    ) where {T<:Real}

    (Σ_samp, Σ_obs, Σ_samp_obs, Σ_obs_samp) = covariance_permutations(prob_def, x_samp, total_hyperparameters)
    return get_σ(ridge_chol(Σ_obs).L, Σ_obs_samp, diag(Σ_samp))
end


"calcuate all of the different versions of the covariance matrices for measured and sampled points"
function covariance_permutations(
    kernel_func::Function,
    x_obs::Vector{T},
    x_samp::Vector{T},
    measurement_noise::Vector{T},
    kernel_hyperparameters::Vector{T};
    return_both::Bool=false
    ) where {T<:Real}

    Σ_samp = covariance(kernel_func, x_samp, x_samp, kernel_hyperparameters)
    Σ_samp_obs = covariance(kernel_func, x_samp, x_obs, kernel_hyperparameters)
    Σ_obs_samp = transpose(Σ_samp_obs)
    if return_both
        Σ_obs, Σ_obs_raw = Σ_observations(kernel_func, x_obs, measurement_noise, kernel_hyperparameters; return_both=return_both)
        return Σ_samp, Σ_obs, Σ_samp_obs, Σ_obs_samp, Σ_obs_raw
    else
        Σ_obs = Σ_observations(kernel_func, x_obs, measurement_noise, kernel_hyperparameters; return_both=return_both)
        return Σ_samp, Σ_obs, Σ_samp_obs, Σ_obs_samp
    end
end

"calcuate all of the different versions of the covariance matrices for measured and sampled points"
function covariance_permutations(
    prob_def::Jones_problem_definition,
    x_samp::Vector{T},
    total_hyperparameters::Vector{T};
    return_both::Bool=false
    ) where {T<:Real}

    Σ_samp = covariance(prob_def, x_samp, x_samp, total_hyperparameters)
    Σ_samp_obs = covariance(prob_def, x_samp, prob_def.x_obs, total_hyperparameters)
    Σ_obs_samp = transpose(Σ_samp_obs)
    if return_both
        Σ_obs, Σ_obs_raw = Σ_observations(prob_def, total_hyperparameters; return_both=return_both)
        return Σ_samp, Σ_obs, Σ_samp_obs, Σ_obs_samp, Σ_obs_raw
    else
        Σ_obs = Σ_observations(prob_def, total_hyperparameters; return_both=return_both)
        return Σ_samp, Σ_obs, Σ_samp_obs, Σ_obs_samp
    end
end


"Condition the GP on data"
function GP_posteriors_from_covariances(
    y_obs::Vector{T},
    Σ_samp::Union{Cholesky{T,Matrix{T}},Symmetric{T,Matrix{T}},Matrix{T}},
    Σ_obs::Cholesky{T,Matrix{T}},
    Σ_samp_obs::Union{Symmetric{T,Matrix{T}},Matrix{T}},
    Σ_obs_samp::Union{Transpose{T,Matrix{T}},Symmetric{T,Matrix{T}},Matrix{T}},
    Σ_obs_raw::Symmetric{T,Matrix{T}};
    return_σ::Bool=false,
    return_Σ::Bool=true,
    chol::Bool=false,
    ) where {T<:Real}

    # posterior mean calcuation from RW alg. 2.1

    # these are all equivalent but have different computational costs
    # α = inv(Σ_obs) * y_obs
    # α = transpose(L) \ (L \ y_obs)
    α = Σ_obs \ y_obs

    mean_post = Σ_samp_obs * α
    mean_post_obs = Σ_obs_raw * α

    # posterior standard deviation calcuation from RW alg. 2.1
    σ = get_σ(Σ_obs.L, Σ_obs_samp, diag(Σ_samp))

    # posterior covariance calculation is from eq. 2.24 of RW
    if return_Σ
        Σ_post = symmetric_A(Σ_samp - (Σ_samp_obs * (Σ_obs \ Σ_obs_samp)), chol=chol)
        return mean_post, σ, mean_post_obs, Σ_post
    else
        return mean_post, σ, mean_post_obs
    end

end


"Condition the GP on data"
function GP_posteriors_from_covariances(
    y_obs::Vector{T},
    Σ_samp::Union{Cholesky{T,Matrix{T}},Symmetric{T,Matrix{T}},Matrix{T}},
    Σ_obs::Cholesky{T,Matrix{T}},
    Σ_samp_obs::Union{Symmetric{T,Matrix{T}},Matrix{T}},
    Σ_obs_samp::Union{Transpose{T,Matrix{T}},Symmetric{T,Matrix{T}},Matrix{T}};
    return_σ::Bool=false,
    return_Σ::Bool=true,
    chol::Bool=false,
    ) where {T<:Real}

    # posterior mean calcuation from RW alg. 2.1

    # these are all equivalent but have different computational costs
    # α = inv(Σ_obs) * y_obs
    # α = transpose(L) \ (L \ y_obs)
    α = Σ_obs \ y_obs

    mean_post = Σ_samp_obs * α

    # posterior standard deviation calcuation from RW alg. 2.1
    σ = get_σ(Σ_obs.L, Σ_obs_samp, diag(Σ_samp))

    # posterior covariance calculation is from eq. 2.24 of RW
    if return_Σ
        Σ_post = symmetric_A(Σ_samp - (Σ_samp_obs * (Σ_obs \ Σ_obs_samp)), chol=chol)
        return mean_post, σ, Σ_post
    else
        return mean_post, σ
    end

end


function GP_posteriors(
    kernel_func::Function,
    x_obs::Vector{T},
    y_obs::Vector{T},
    x_samp::Vector{T},
    measurement_noise::Vector{T},
    total_hyperparameters::Vector{T};
    return_Σ::Bool=true,
    chol::Bool=false,
    return_mean_obs::Bool=false
    ) where {T<:Real}

    if return_mean_obs
        (Σ_samp, Σ_obs, Σ_samp_obs, Σ_obs_samp, Σ_obs_raw) = covariance_permutations(kernel_func, x_obs, x_samp, measurement_noise, total_hyperparameters; return_both=return_mean_obs)
        return GP_posteriors_from_covariances(y_obs, Σ_samp, Σ_obs, Σ_samp_obs, Σ_obs_samp, Σ_obs_raw; return_Σ=return_Σ, chol=chol)
    else
        (Σ_samp, Σ_obs, Σ_samp_obs, Σ_obs_samp) = covariance_permutations(kernel_func, x_obs, x_samp, measurement_noise, total_hyperparameters; return_both=return_mean_obs)
        return GP_posteriors_from_covariances(y_obs, Σ_samp, Σ_obs, Σ_samp_obs, Σ_obs_samp; return_Σ=return_Σ, chol=chol)
    end
end

function GP_posteriors(
    prob_def::Jones_problem_definition,
    x_samp::Vector{T},
    total_hyperparameters::Vector{T};
    return_Σ::Bool=true,
    chol::Bool=false,
    return_mean_obs::Bool=false
    ) where {T<:Real}

    if return_mean_obs
        (Σ_samp, Σ_obs, Σ_samp_obs, Σ_obs_samp, Σ_obs_raw) = covariance_permutations(prob_def, x_samp, total_hyperparameters; return_both=return_mean_obs)
        return GP_posteriors_from_covariances(prob_def.y_obs, Σ_samp, Σ_obs, Σ_samp_obs, Σ_obs_samp, Σ_obs_raw; return_Σ=return_Σ, chol=chol)
    else
        (Σ_samp, Σ_obs, Σ_samp_obs, Σ_obs_samp) = covariance_permutations(prob_def, x_samp, total_hyperparameters; return_both=return_mean_obs)
        return GP_posteriors_from_covariances(prob_def.y_obs, Σ_samp, Σ_obs, Σ_samp_obs, Σ_obs_samp; return_Σ=return_Σ, chol=chol)
    end

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
    a::Matrix{T}=ones(n_out, n_dif)
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
dΣdθ_total (matrix): The coefficients for the Jones model
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
    dΣdθ_total::Integer,
    coeff_orders::AbstractArray{T,6},
    coeff_coeffs::AbstractArray{T,4}
    ) where {T<:Integer}

    @assert size(coeff_orders) == (n_out, n_out, n_dif, n_dif, n_out, n_dif)
    @assert size(coeff_orders) == (n_out, n_out, n_dif, n_dif, n_out, n_dif)

    # only do something if a derivative is being taken
    if dΣdθ_total > 0 && dΣdθ_total <= (n_out*n_dif)
        proper_indices = [((dΣdθ_total - 1) % n_out) + 1, div(dΣdθ_total - 1, n_out) + 1]
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
    dΣdθs_total::Vector{T},
    coeff_orders::AbstractArray{T,6},
    coeff_coeffs::AbstractArray{T,4}
    ) where {T<:Integer}

    for dΣdθ_total in dΣdθs_total
        dif_coefficients!(n_out, n_dif, dΣdθ_total, coeff_orders, coeff_coeffs)
    end
end


"""
GP negative log marginal likelihood (see Algorithm 2.1 in Rasmussen and Williams 2006)

Parameters:

Σ_obs (Cholesky factorized object): The covariance matrix constructed by
    evaulating the kernel at each pair of observations and adding measurement
    noise.
y_obs (vector): The observations at each time point
α (vector): inv(Σ_obs) * y_obs

Returns:
float: the negative log marginal likelihood

"""
function nlogL(
    Σ_obs::Cholesky{T,Matrix{T}},
    y_obs::Vector{T},
    α::Vector{T}
    ) where {T<:Real}

    n = length(y_obs)

    # 2 times negative goodness of fit term
    data_fit = transpose(y_obs) * α
    # 2 times negative complexity penalization term
    # complexity_penalty = log(det(Σ_obs))
    complexity_penalty = logdet(Σ_obs)  # half memory but twice the time
    # 2 times negative normalization term (doesn't affect fitting)
    normalization = n * log(2 * π)

    return 0.5 * (data_fit + complexity_penalty + normalization)

end

nlogL(Σ_obs, y_obs) = nlogL(Σ_obs, y_obs, Σ_obs \ y_obs)


"""
First partial derivative of the GP negative log marginal likelihood
(see eq. 5.9 in Rasmussen and Williams 2006)

Parameters:

y_obs (vector): The observations at each time point
α (vector): inv(Σ_obs) * y_obs
β (matrix): inv(Σ_obs) * dΣ_dθ where dΣ_dθ is the partial derivative of the
    covariance matrix Σ_obs w.r.t. a hyperparameter

Returns:
float: the partial derivative of the negative log marginal likelihood w.r.t. the
    hyperparameter used in the calculation of β

"""
function dnlogLdθ(
    y_obs::Vector{T},
    α::Vector{T},
    β::Matrix{T}
    ) where {T<:Real}

    # 2 times negative derivative of goodness of fit term
    data_fit = -(transpose(y_obs) * β * α)
    # 2 times negative derivative of complexity penalization term
    complexity_penalty = tr(β)

    # return -1 / 2 * tr((α * transpose(α) - inv(Σ_obs)) * dΣ_dθj)
    return 0.5 * (data_fit + complexity_penalty)

end


"Returns gradient of nlogL"
∇nlogL(y_obs::Vector{T}, α::Vector{T}, βs::Vector{Matrix{T}}) where {T<:Real} = [dnlogLdθ(y_obs, α, β) for β in βs]


"""
Second partial derivative of the GP negative log marginal likelihood.
Calculated with help from rules found on page 7 of the matrix cookbook
(https://www.ics.uci.edu/~welling/teaching/KernelsICS273B/MatrixCookBook.pdf)

Parameters:

y_obs (vector): The observations at each time point
α (vector): inv(Σ_obs) * y_obs
β1 (matrix): inv(Σ_obs) * dΣ_dθ1 where dΣ_dθ1 is the partial derivative of the
    covariance matrix Σ_obs w.r.t. a hyperparameter
β2 (matrix): inv(Σ_obs) * dΣ_dθ2 where dΣ_dθ2 is the partial derivative of the
    covariance matrix Σ_obs w.r.t. another hyperparameter
β12 (matrix): inv(Σ_obs) * d2Σ_dθ1dθ2 where d2Σ_dθ1dθ2 is the partial
    derivative of the covariance matrix Σ_obs w.r.t. both of the hyperparameters
    being considered

Returns:
float: the partial derivative of the negative log marginal likelihood w.r.t. the
    hyperparameters used in the calculation of β1, β2, and β12

"""
function d2nlogLdθ(
    y_obs::Vector{T},
    α::Vector{T},
    β1::Matrix{T},
    β2::Matrix{T},
    β12::Matrix{T}
    ) where {T<:Real}

    β12mβ2β1 = β12 - β2 * β1

    # 2 times negative second derivative of goodness of fit term
    data_fit = -(transpose(y_obs) * (β12mβ2β1 - β1 * β2) * α)
    # 2 times negative derivative of complexity penalization term
    complexity_penalty = tr(β12mβ2β1)

    return 0.5 * (data_fit + complexity_penalty)

end


struct nlogL_Jones_matrix_workspace{T<:Real}
    nlogL_hyperparameters::Vector{T}
    Σ_obs::Cholesky{T,Matrix{T}}
    y_obs::Vector{T}
    α::Vector{T}
    ∇nlogL_hyperparameters::Vector{T}
    βs::Vector{Matrix{T}}
end


# "Ensure that the passed matrix workspace parameters are what we expect them to be"
# function check_matrix_workspace(
#     nlogL_hyperparameters::Vector{T}
#     Σ_obs::Cholesky{T,Matrix{T}}
#     y_obs::Vector{T}
#     α::Vector{T}
#     ∇nlogL_hyperparameters::Vector{T}
#     βs::Vector{Matrix{T}}
#     ) where {T<:Real}
#
#     @assert length(y_obs) == length(α) == size(Σ_obs, 1) == size(Σ_obs, 2) = size(βs[1], 1) == size(βs[1], 2)
#     @assert length(nlogL_hyperparameters) == length(∇nlogL_hyperparameters)
#     @assert length(findall(!iszero, nlogL_hyperparameters)) == length(findall(!iszero, ∇nlogL_hyperparameters)) == size(βs, 1)
# end


function init_nlogL_Jones_matrix_workspace(
    prob_def::Jones_problem_definition,
    total_hyperparameters::Vector{<:Real})

    total_hyper = reconstruct_total_hyperparameters(prob_def, total_hyperparameters)
    Σ_obs = Σ_observations(prob_def, total_hyper; ignore_asymmetry=true)
    return nlogL_Jones_matrix_workspace(
        total_hyper,
        Σ_obs,
        copy(prob_def.y_obs),
        Σ_obs \ prob_def.y_obs,
        copy(total_hyper),
        [Σ_obs \ covariance(prob_def, total_hyper; dΣdθs_total=[i]) for i in findall(!iszero, total_hyper)])
end


"Calculates the quantities shared by the nlogL and ∇nlogL calculations"
function calculate_shared_nlogL_Jones(
    prob_def::Jones_problem_definition,
    non_zero_hyperparameters::Vector{<:Real};
    Σ_obs::Cholesky{T,Matrix{T}} where T<:Real=Σ_observations(prob_def, reconstruct_total_hyperparameters(prob_def, non_zero_hyperparameters); ignore_asymmetry=true),
    P::Union{Real, Quantity}=0)

    # this allows us to prevent the optimizer from seeing the constant zero coefficients
    total_hyperparameters = reconstruct_total_hyperparameters(prob_def, non_zero_hyperparameters)

    y_obs = copy(prob_def.y_obs)

    if P!=0
        y_obs = remove_kepler(y_obs, prob_def.x_obs, convert_and_strip_units(prob_def.x_obs_units, P), Σ_obs)
    end

    α = Σ_obs \ y_obs

    return total_hyperparameters, Σ_obs, y_obs, α

end


function calculate_shared_nlogL_Jones!(
    workspace::nlogL_Jones_matrix_workspace,
    prob_def::Jones_problem_definition,
    non_zero_hyperparameters::Vector{<:Real};
    P::Union{Real, Quantity}=0)

    # this allows us to prevent the optimizer from seeing the constant zero coefficients
    total_hyperparameters = reconstruct_total_hyperparameters(prob_def, non_zero_hyperparameters)

    recalc_α = false

    if !isapprox(workspace.nlogL_hyperparameters, total_hyperparameters)
        workspace.nlogL_hyperparameters[:] = total_hyperparameters
        workspace.Σ_obs.factors[:,:] = Σ_observations(prob_def, total_hyperparameters).factors
        recalc_α = true
    end

    P != 0 ? y_obs = remove_kepler(prob_def.y_obs, prob_def.x_obs, convert_and_strip_units(prob_def.x_obs_units, P), workspace.Σ_obs) : y_obs = copy(prob_def.y_obs)

    if !isapprox(workspace.y_obs, y_obs)
        workspace.y_obs[:] = y_obs
        recalc_α = true
    end

    if recalc_α; workspace.α[:] = workspace.Σ_obs \ workspace.y_obs end

end


"Calculates the quantities shared by the ∇nlogL and ∇∇nlogL calculations"
function calculate_shared_∇nlogL_Jones(
    prob_def::Jones_problem_definition,
    non_zero_hyperparameters::Vector{<:Real};
    Σ_obs::Cholesky{T,Matrix{T}} where T<:Real=Σ_observations(prob_def, reconstruct_total_hyperparameters(prob_def, non_zero_hyperparameters), ignore_asymmetry=true),
    P::Union{Real, Quantity}=0)

    total_hyperparameters, Σ_obs, y_obs, α = calculate_shared_nlogL_Jones(prob_def, non_zero_hyperparameters; Σ_obs=Σ_obs, P=P)

    βs = [Σ_obs \ covariance(prob_def, total_hyperparameters; dΣdθs_total=[i]) for i in findall(!iszero, total_hyperparameters)]

    return total_hyperparameters, Σ_obs, y_obs, α, βs

end


function calculate_shared_∇nlogL_Jones!(
    workspace::nlogL_Jones_matrix_workspace,
    prob_def::Jones_problem_definition,
    non_zero_hyperparameters::Vector{<:Real};
    P::Union{Real, Quantity}=0)

    calculate_shared_nlogL_Jones!(workspace, prob_def, non_zero_hyperparameters; P=P)

    if !isapprox(workspace.∇nlogL_hyperparameters, workspace.nlogL_hyperparameters)
        workspace.∇nlogL_hyperparameters[:] = workspace.nlogL_hyperparameters
        workspace.βs[:] = [workspace.Σ_obs \ covariance(prob_def, workspace.∇nlogL_hyperparameters; dΣdθs_total=[i]) for i in findall(!iszero, workspace.∇nlogL_hyperparameters)]
    end

end


"nlogL for Jones GP"
function nlogL_Jones(
    prob_def::Jones_problem_definition,
    total_hyperparameters::Vector{T};
    Σ_obs::Cholesky{T,Matrix{T}}=Σ_observations(prob_def, reconstruct_total_hyperparameters(prob_def, total_hyperparameters); ignore_asymmetry=true),
    P::Union{Real, Quantity}=0
    ) where {T<:Real}

    total_hyperparameters, Σ_obs, y_obs, α = calculate_shared_nlogL_Jones(
        prob_def, remove_zeros(total_hyperparameters); Σ_obs=Σ_obs, P=P)
    return nlogL(Σ_obs, y_obs, α)
end


"nlogL for Jones GP"
function nlogL_Jones(
    workspace::nlogL_Jones_matrix_workspace,
    prob_def::Jones_problem_definition,
    total_hyperparameters::Vector{T};
    P::Union{Real, Quantity}=0
    ) where {T<:Real}

    calculate_shared_nlogL_Jones!(workspace, prob_def, total_hyperparameters; P=P)
    return nlogL(workspace.Σ_obs, workspace.y_obs, workspace.α)
end


"Returns gradient of nlogL for non-zero hyperparameters"
function ∇nlogL_Jones(
    prob_def::Jones_problem_definition,
    total_hyperparameters::Vector{T};
    Σ_obs::Cholesky{T,Matrix{T}}=Σ_observations(prob_def, reconstruct_total_hyperparameters(prob_def, total_hyperparameters); ignore_asymmetry=true),
    P::Union{Real, Quantity}=0
    ) where {T<:Real}

    total_hyperparameters, Σ_obs, y_obs, α, βs = calculate_shared_∇nlogL_Jones(
        prob_def, remove_zeros(total_hyperparameters); Σ_obs=Σ_obs, P=P)

    return ∇nlogL(y_obs, α, βs)

end


"Returns gradient of nlogL for non-zero hyperparameters"
function ∇nlogL_Jones(
    worskpace::nlogL_Jones_matrix_workspace,
    prob_def::Jones_problem_definition,
    total_hyperparameters::Vector{T};
    P::Union{Real, Quantity}=0
    ) where {T<:Real}

    calculate_shared_∇nlogL_Jones!(workspace, prob_def, total_hyperparameters; P=P)

    return ∇nlogL(workspace.y_obs, workspace.α, workspace.βs)

end


"Replaces H with Hessian of nlogL for non-zero hyperparameters"
function ∇∇nlogL_Jones(
    prob_def::Jones_problem_definition,
    total_hyperparameters::Vector{T},
    Σ_obs::Cholesky{T,Matrix{T}},
    y_obs::Vector{T},
    α::Vector{T},
    βs::Array{Matrix{T},1}
    ) where {T<:Real}

    non_zero_inds = findall(!iszero, total_hyperparameters)
    H = zeros(length(non_zero_inds), length(non_zero_inds))

    for (i, nzind1) in enumerate(non_zero_inds)
        for (j, nzind2) in enumerate(non_zero_inds)
            if i <= j
                H[i, j] = d2nlogLdθ(y_obs, α, βs[i], βs[j], Σ_obs \ covariance(prob_def, total_hyperparameters; dΣdθs_total=[nzind1, nzind2]))
            end
        end
    end

    H = Symmetric(H)

end


"Replaces H with Hessian of nlogL for non-zero hyperparameters"
function ∇∇nlogL_Jones(
    prob_def::Jones_problem_definition,
    total_hyperparameters::Vector{T};
    Σ_obs::Cholesky{T,Matrix{T}}=Σ_observations(prob_def, reconstruct_total_hyperparameters(prob_def, total_hyperparameters); ignore_asymmetry=true),
    P::Union{Real, Quantity}=0
    ) where {T<:Real}

    total_hyperparameters, Σ_obs, y_obs, α, βs = calculate_shared_∇nlogL_Jones(
        prob_def, remove_zeros(total_hyperparameters); Σ_obs=Σ_obs, P=P)

    return ∇∇nlogL_Jones(prob_def, total_hyperparameters, Σ_obs, y_obs, α, βs)

end


"reinsert the zero coefficients into the non-zero hyperparameter list if needed"
function reconstruct_total_hyperparameters(
    prob_def::Jones_problem_definition,
    hyperparameters::Vector{T}
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
    current_params::Vector{T}
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
    total_hyperparameters::Vector{T};
    alpha::Real=1,
    beta::Real=1
    ) where {T<:Real}

    logP = 0

    # adding prior for physical length scales
    for i in 1:prob_def.n_kern_hyper
        kernel_length = total_hyperparameters[end + 1 - i]
        logP += log_inverse_gamma(kernel_length, alpha, beta)
    end

    return logP

end

function ∇logGP_prior(
    total_hyperparameters::Vector{T};
    alpha::Real=1,
    beta::Real=1
    ) where {T<:Real}

    ∇logP = zero(total_hyperparameters)

    # adding prior for physical length scales
    for i in 1:prob_def.n_kern_hyper
        kernel_length = total_hyperparameters[end + 1 - i]
        ∇logP[end + 1 - i] -= log_inverse_gamma(kernel_length, alpha, beta; d=1)
    end

    return ∇logP

end

function ∇∇logGP_prior(
    total_hyperparameters::Vector{T};
    alpha::Real=1,
    beta::Real=1
    ) where {T<:Real}

    ∇∇logP = zeros(length(total_hyperparameters), length(total_hyperparameters))

    # adding prior for physical length scales
    for i in 1:prob_def.n_kern_hyper
        kernel_length = total_hyperparameters[end + 1 - i]
        ∇∇logP[end + 1 - i, end + 1 - i] -= log_inverse_gamma(kernel_length, alpha, beta; d=2)
    end

    return ∇∇logP

end
