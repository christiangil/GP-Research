# these are all custom diagnostic functions. May help with debugging
using Statistics


"""
Estimate the covariance derivatives with forward differences
"""
function est_dΣdθ(prob_def::Jones_problem_definition, kernel_hyperparameters::Vector{T}; return_est::Bool=true, return_anal::Bool=false, return_dif::Bool=false, return_bool::Bool=false, dif::Real=1e-6, print_stuff::Bool=true) where {T<:Real}

    total_hyperparameters = append!(collect(Iterators.flatten(prob_def.a0)), kernel_hyperparameters)

    if print_stuff
        println()
        println("est_dΣdθ: Check that our analytical dΣdθ are close to numerical estimates")
        println("hyperparameters: ", total_hyperparameters)
    end

    x = prob_def.x_obs
    return_vec = []
    coeff_orders = coefficient_orders(prob_def.n_out, prob_def.n_dif, a=prob_def.a0)

    # construct estimated dΣdθs
    if return_est || return_dif || return_bool
        val = covariance(prob_def, total_hyperparameters)
        est_dΣdθs = zeros(length(total_hyperparameters), prob_def.n_out * length(x), prob_def.n_out * length(x))
        for i in 1:length(total_hyperparameters)
            if total_hyperparameters[i]!=0
                hold = copy(total_hyperparameters)
                hold[i] += dif
                est_dΣdθs[i, :, :] =  (covariance(prob_def, hold) - val) / dif
            end
        end
        if return_est; append!(return_vec, [est_dΣdθs]) end
    end

    # construct analytical dΣdθs
    if return_anal || return_dif || return_bool
        anal_dΣdθs = zeros(length(total_hyperparameters), prob_def.n_out * length(x), prob_def.n_out * length(x))
        for i in 1:length(total_hyperparameters)
            anal_dΣdθs[i, :, :] =  covariance(prob_def, total_hyperparameters; dΣdθs_total=[i])
        end
        if return_anal; append!(return_vec, [anal_dΣdθs]) end
    end

    if return_dif || return_bool
        difs = est_dΣdθs - anal_dΣdθs
        append!(return_vec, [difs])
    end

    # find whether the analytical and estimated dΣdθs are approximately the same
    if return_bool
        no_differences = true
        min_thres = 5e-3
        max_val = 0
        for ind in 1:length(total_hyperparameters)
            est = est_dΣdθs[ind,:,:]
            est[est .== 0] .= 1e-8
            val = mean(abs.(difs[ind,:,:] ./ est))
            max_val = maximum([max_val, val])
            if val>min_thres
                no_differences = false
                if print_stuff; println("dΣ/dθ$ind has a average ratioed difference of: ", val) end
            end
        end
        if print_stuff; println("Maximum average dΣ/dθ ratioed difference of: ", max_val) end
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
function est_grad(prob_def::Jones_problem_definition, total_hyperparameters::Vector{T}; dif::Real=1e-4) where {T<:Real}

    # original value
    val = nlogL_Jones(prob_def, total_hyperparameters)

    #estimate gradient
    j = 1
    grad = zeros(length(prob_def.non_zero_hyper_inds))
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
function test_grad(prob_def::Jones_problem_definition, kernel_hyperparameters::Vector{T}; dif::Real=1e-4, print_stuff::Bool=true) where {T<:Real}

    total_hyperparameters = append!(collect(Iterators.flatten(prob_def.a0)), kernel_hyperparameters)
    G = ∇nlogL_Jones(prob_def, total_hyperparameters)
    est_G = est_grad(prob_def, total_hyperparameters; dif=dif)

    if print_stuff
        println()
        println("test_grad: Check that our analytical ∇nlogL_Jones() is close to numerical estimates")
        println("only values for non-zero hyperparameters are shown!")
        println("hyperparameters: ", total_hyperparameters)
        println("analytical: ", G)
        println("numerical : ", est_G)
    end

    no_mismatch = true
    for i in 1:length(G)
        if !isapprox(G[i], est_G[i], rtol=5e-2)
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
function est_hess(prob_def::Jones_problem_definition, total_hyperparameters::Vector{T}; dif::Real=1e-4) where {T<:Real}

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
function test_hess(prob_def::Jones_problem_definition, kernel_hyperparameters::Vector{T}; dif::Real=1e-4, print_stuff::Bool=true) where {T<:Real}

    total_hyperparameters = append!(collect(Iterators.flatten(prob_def.a0)), kernel_hyperparameters)
    H = ∇∇nlogL_Jones(prob_def, total_hyperparameters)
    est_H = est_hess(prob_def, total_hyperparameters; dif=dif)

    if print_stuff
        println()
        println("test_hess: Check that our analytical ∇∇nlogL_Jones() is close to numerical estimates")
        println("only values for non-zero hyperparameters are shown!")
        println("hyperparameters: ", total_hyperparameters)
        println("analytical:")
        for i in 1:size(H, 1)
            println(H[i, :])
        end
        println("numerical:")
        for i in 1:size(est_H, 1)
            println(est_H[i, :])
        end
    end

    no_mismatch = true
    matches = fill(1, size(H))
    for i in 1:size(H, 1)
        for j in 1:size(H, 2)
            if !(isapprox(H[i, j], est_H[i, j], rtol=1e-1))
                # println("mismatch d2nlogL/dθ" * string(i) * "dθ" * string(j))
                matches[i, j] = 0
                no_mismatch = false
            end
        end
    end

    if !no_mismatch
        println("mismatches at 0s")
        for i in 1:size(H, 1)
            println(matches[i, :])
        end
    end

    return no_mismatch
end


function est_kep_grad(K1::T, h1::T, k1::T, M01::T, γ1::T, P1::T, t1::T; dif::Real=1e-6) where {T<:Real}

    tester(parms) = kep_deriv(parms[1], parms[2], parms[3], parms[4], parms[5], parms[6], t1, [0,0,0,0,0,0])
    parms = [K1, h1, k1, M01, γ1, P1]
    # original value
    val = tester(parms)

    #estimate gradient
    grad = zeros(length(parms))
    for i in 1:length(parms)
        hold = copy(parms)
        hold[i] += dif
        grad[i] = (tester(hold) - val) / dif
    end
    return grad
end


function test_kep_grad(K1::T, h1::T, k1::T, M01::T, γ1::T, P1::T, t1::T; dif::Real=1e-6, print_stuff::Bool=true) where {T<:Real}

    pamrs_str = ["K", "h", "k", "M0", "γ", "P"]
    parms = [K1, h1, k1, M01, γ1, P1]
    G = kep_grad(K1, h1, k1, M01, γ1, P1, t1)
    est_G = est_kep_grad(K1, h1, k1, M01, γ1, P1, t1; dif=dif)

    if print_stuff
        println()
        println("test_kep_grad: Check that our analytical kep_grad() is close to numerical estimates")
        println("parameters: ", parms)
        println("analytical: ", G)
        println("numerical : ", est_G)
    end

    no_mismatch = true
    for i in 1:length(G)
        if !isapprox(G[i], est_G[i], rtol=5e-2)
            println("mismatch dkep/d" * pamrs_str[i])
            no_mismatch = false
        end
    end

    return no_mismatch
end


function est_kep_hess(K1::T, h1::T, k1::T, M01::T, γ1::T, P1::T, t1::T; dif::Real=1e-8) where {T<:Real}

    tester(parms) = kep_grad(parms[1], parms[2], parms[3], parms[4], parms[5], parms[6], t1)
    parms = [K1, h1, k1, M01, γ1, P1]
    # original value
    val = tester(parms)

    #estimate hessian
    hess = zeros(length(parms), length(parms))
    for i in 1:length(parms)
        hold = copy(parms)
        hold[i] += dif
        hess[i, :] =  (tester(hold) - val) / dif
    end
    return hess
end


function test_kep_hess(K1::T, h1::T, k1::T, M01::T, γ1::T, P1::T, t1::T; dif::Real=1e-6, print_stuff::Bool=true) where {T<:Real}

    parms = [K1, h1, k1, M01, γ1, P1]
    H = kep_hess(K1, h1, k1, M01, γ1, P1, t1)
    est_H = est_kep_hess(K1, h1, k1, M01, γ1, P1, t1; dif=dif)

    if print_stuff
        println()
        println("test_kep_hess: Check that our analytical kep_hess() is close to numerical estimates")
        println("parameters: ", parms)
        println("analytical:")
        for i in 1:size(H, 1)
            println(H[i, :])
        end
        println("numerical:")
        for i in 1:size(est_H, 1)
            println(est_H[i, :])
        end
    end

    no_mismatch = true
    matches = fill(1, size(H))
    for i in 1:size(H, 1)
        for j in 1:size(H, 2)
            if !(isapprox(H[i, j], est_H[i, j], rtol=5e-1))
                # println("mismatch d2nlogL/dθ" * string(i) * "dθ" * string(j))
                matches[i, j] = 0
                no_mismatch = false
            end
        end
    end

    if !no_mismatch
        println("mismatches at 0s")
        for i in 1:size(H, 1)
            println(matches[i, :])
        end
    end

    return no_mismatch
end
