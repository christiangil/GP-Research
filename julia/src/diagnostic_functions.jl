# these are all custom diagnostic functions. May help with debugging
using Statistics


"""
Estimate the covariance derivatives with forward differences
"""
function est_dKdθ(prob_def::Jones_problem_definition, kernel_hyperparameters::AbstractArray{T,1}; return_est::Bool=true, return_anal::Bool=false, return_dif::Bool=false, return_bool::Bool=false, dif::Real=1e-6, print_stuff::Bool=true) where {T<:Real}

    total_hyperparameters = append!(collect(Iterators.flatten(prob_def.a0)), kernel_hyperparameters)

    if print_stuff
        println()
        println("est_dKdθ: Check that our analytical dKdθ are close to numerical estimates")
        println("hyperparameters: ", total_hyperparameters)
    end

    x = prob_def.x_obs
    return_vec = []
    coeff_orders = coefficient_orders(prob_def.n_out, prob_def.n_dif, a=prob_def.a0)

    # construct estimated dKdθs
    if return_est || return_dif || return_bool
        val = covariance(prob_def, total_hyperparameters)
        est_dKdθs = zeros(length(total_hyperparameters), prob_def.n_out * length(x), prob_def.n_out * length(x))
        for i in 1:length(total_hyperparameters)
            if total_hyperparameters[i]!=0
                hold = copy(total_hyperparameters)
                hold[i] += dif
                est_dKdθs[i, :, :] =  (covariance(prob_def, hold) - val) / dif
            end
        end
        if return_est; append!(return_vec, [est_dKdθs]) end
    end

    # construct analytical dKdθs
    if return_anal || return_dif || return_bool
        anal_dKdθs = zeros(length(total_hyperparameters), prob_def.n_out * length(x), prob_def.n_out * length(x))
        for i in 1:length(total_hyperparameters)
            anal_dKdθs[i, :, :] =  covariance(prob_def, total_hyperparameters; dKdθs_total=[i])
        end
        if return_anal; append!(return_vec, [anal_dKdθs]) end
    end

    if return_dif || return_bool
        difs = est_dKdθs - anal_dKdθs
        append!(return_vec, [difs])
    end

    # find whether the analytical and estimated dKdθs are approximately the same
    if return_bool
        no_differences = true
        min_thres = 5e-3
        max_val = 0
        for ind in 1:length(total_hyperparameters)
            est = est_dKdθs[ind,:,:]
            est[est .== 0] .= 1e-8
            val = mean(abs.(difs[ind,:,:] ./ est))
            max_val = maximum([max_val, val])
            if val>min_thres
                no_differences = false
                if print_stuff; println("dK/dθ$ind has a average ratioed difference of: ", val) end
            end
        end
        if print_stuff; println("Maximum average dK/dθ ratioed difference of: ", max_val) end
        return no_differences
    end

    return return_vec

end


"""
Estimate the gradient of nlogL_Jones() with forward differences

Parameters:

prob_def (Jones_problem_definition): A structure that holds all of the relevant
    information for constructing the model used in the Jones et al. 2017+ paper
total_hyperparameters (vector): The hyperparameters for the GP model, including
    both the coefficient hyperparameters and the kernel hyperparameters
dif (float): How much to perturb the hyperparameters

Returns:
vector: An estimate of the gradient

"""
function est_grad(prob_def::Jones_problem_definition, total_hyperparameters::AbstractArray{T,1}; dif::Real=1e-4) where {T<:Real}

    # original value
    val = nlogL_Jones(prob_def, total_hyperparameters)

    #estimate gradient
    j = 1
    grad = zeros(length(findall(!iszero, total_hyperparameters)))
    for i in 1:length(total_hyperparameters)
        if total_hyperparameters[i]!=0
            hold = copy(total_hyperparameters)
            hold[i] += dif
            grad[j] =  (nlogL_Jones(prob_def, hold) - val) / dif
            j += 1
        end
    end
    return grad
end


"""
Test that the analytical and numerically estimated ∇nlogL_Jones() are approximately the same

Parameters:

prob_def (Jones_problem_definition): A structure that holds all of the relevant
    information for constructing the model used in the Jones et al. 2017+ paper
kernel_hyperparameters (vector): The kernel hyperparameters for the GP model
dif (float): How much to perturb the hyperparameters
print_stuff (bool): if true, prints extra information about the output

Returns:
bool: Whether or not every part of the analytical and numerical gradients match

"""
function test_grad(prob_def::Jones_problem_definition, kernel_hyperparameters::AbstractArray{T,1}; dif::Real=1e-4, print_stuff::Bool=true) where {T<:Real}

    total_hyperparameters = append!(collect(Iterators.flatten(prob_def.a0)), kernel_hyperparameters)
    G = ∇nlogL_Jones(prob_def, total_hyperparameters)
    est_G = est_grad(prob_def, total_hyperparameters; dif=dif)

    if print_stuff
        println()
        println("test_grad: Check that our analytical ∇nLogL_Jones() is close to numerical estimates")
        println("only values for non-zero hyperparameters are shown!")
        println("hyperparameters: ", total_hyperparameters)
        println("analytical: ", G)
        println("numerical : ", est_G)
    end

    no_mismatch = true
    for i in 1:length(G)
        if (!(isapprox(G[i], est_G[i], rtol=2e-1))) && (G[i] != 0)
            println("mismatch dnlogL/dθ" * string(i))
            no_mismatch = false
        end
    end

    return no_mismatch
end


"""
Estimate the Hessian of nlogL_Jones() with forward differences

Parameters:

prob_def (Jones_problem_definition): A structure that holds all of the relevant
    information for constructing the model used in the Jones et al. 2017+ paper
total_hyperparameters (vector): The hyperparameters for the GP model, including
    both the coefficient hyperparameters and the kernel hyperparameters
dif (float): How much to perturb the hyperparameters

Returns:
matrix: An estimate of the hessian

"""
function est_hess(prob_def::Jones_problem_definition, total_hyperparameters::AbstractArray{T,1}; dif::Real=1e-4) where {T<:Real}

    # original value
    val = ∇nlogL_Jones(prob_def, total_hyperparameters)

    #estimate hessian
    j = 1
    hess = zeros(length(val), length(val))
    for i in 1:length(total_hyperparameters)
        if total_hyperparameters[i]!=0
            hold = copy(total_hyperparameters)
            hold[i] += dif
            hess[j, :] =  (∇nlogL_Jones(prob_def, hold) - val) / dif
            j += 1
        end
    end
    return hess
end


"""
Test that the analytical and numerically estimated ∇∇nlogL_Jones() are approximately the same

Parameters:

prob_def (Jones_problem_definition): A structure that holds all of the relevant
    information for constructing the model used in the Jones et al. 2017+ paper
kernel_hyperparameters (vector): The kernel hyperparameters for the GP model
dif (float): How much to perturb the hyperparameters
print_stuff (bool): if true, prints extra information about the output

Returns:
bool: Whether or not every part of the analytical and numerical Hessians match

"""
function test_hess(prob_def::Jones_problem_definition, kernel_hyperparameters::AbstractArray{T,1}; dif::Real=1e-4, print_stuff::Bool=true) where {T<:Real}

    total_hyperparameters = append!(collect(Iterators.flatten(prob_def.a0)), kernel_hyperparameters)
    H = ∇∇nlogL_Jones(prob_def, total_hyperparameters)
    est_H = est_grad(prob_def, total_hyperparameters; dif=dif)

    if print_stuff
        println()
        println("test_hess: Check that our analytical ∇∇nLogL_Jones() is close to numerical estimates")
        println("only values for non-zero hyperparameters are shown!")
        println("hyperparameters: ", total_hyperparameters)
        println("analytical: ", H)
        println("numerical : ", est_H)
    end

    no_mismatch = true
    for i in 1:size(H, 1)
        for j in 1:size(H, 2)
            if (!(isapprox(H[i, j], est_H[i, j], rtol=2e-1))) && (H[i, j] != 0)
                println("mismatch d2nlogL/dθ" * string(i) * "dθ" * string(j))
                no_mismatch = false
            end
        end
    end

    return no_mismatch
end
