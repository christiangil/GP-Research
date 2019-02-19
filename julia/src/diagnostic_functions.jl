# these are all custom diagnostic functions. May help with debugging
using Statistics

"estimate the gradient of nlogL with forward differences"
function est_grad(prob_def::Jones_problem_definition, total_hyperparameters::Array{Float64,1}; dif::Float64=0.0001)

    # original value
    val = nlogL_Jones(prob_def, total_hyperparameters)

    #estimate gradient
    grad = zeros(length(total_hyperparameters))
    for i in 1:length(total_hyperparameters)
        if total_hyperparameters[i]!=0
            hold = copy(total_hyperparameters)
            hold[i] += dif
            non_zero_hyperparameters = total_hyperparameters[findall(!iszero, total_hyperparameters)]
            total_hyperparameters, L_fact, y_obs, L_fact_solve_y_obs = calculate_shared_nLogL_Jones(prob_def, non_zero_hyperparameters)
            grad[i] =  (nlogL_Jones(prob_def, hold) - val) / dif
        end
    end
    return grad[findall(!iszero, grad)]
end


"test that the analytical and numerically estimated ∇nlogL are approximately the same"
function test_grad(prob_def::Jones_problem_definition, kernel_hyperparameters::Array{Float64,1}; dif::Float64=1e-7, print_stuff::Bool=true)

    total_hyperparameters = append!(collect(Iterators.flatten(prob_def.a0)), kernel_hyperparameters)
    G = ∇nlogL_Jones(prob_def, total_hyperparameters)
    est_G = est_grad(prob_def, total_hyperparameters; dif=dif)

    if print_stuff
        println()
        println("test_grad: Check that our analytical ∇nLogL is close to numerical estimates")
        println("only values for non-zero hyperparameters are shown!")
        println("hyperparameters: ", total_hyperparameters)
        println("analytical: ", G)
        println("numerical : ", est_G)
    end

    no_mismatch = true
    for i in 1:length(G)
        if (!(isapprox(G[i], est_G[i], rtol=2e-1))) & (G[i] != 0)
            println("mismatch dnlogL/dθ" * string(i))
            no_mismatch = false
        end
    end

    return no_mismatch
end


"estimate the covariance derivatives with forward differences"
function est_dKdθ(prob_def::Jones_problem_definition, kernel_hyperparameters::Array{Float64,1}; return_est::Bool=true, return_anal::Bool=false, return_dif::Bool=false, return_bool::Bool=false, dif::Float64=1e-6, print_stuff::Bool=true)

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
    if return_est | return_dif | return_bool
        val = covariance(prob_def, total_hyperparameters)
        est_dKdθs = zeros(length(total_hyperparameters), prob_def.n_out * length(x), prob_def.n_out * length(x))
        for i in 1:length(total_hyperparameters)
            if total_hyperparameters[i]!=0
                hold = copy(total_hyperparameters)
                hold[i] += dif
                est_dKdθs[i, :, :] =  (covariance(prob_def, hold) - val) / dif
            end
        end
        if return_est
            append!(return_vec, [est_dKdθs])
        end
    end

    # construct analytical dKdθs
    if return_anal | return_dif | return_bool
        anal_dKdθs = zeros(length(total_hyperparameters), prob_def.n_out * length(x), prob_def.n_out * length(x))
        for i in 1:length(total_hyperparameters)
            anal_dKdθs[i, :, :] =  covariance(prob_def, total_hyperparameters; dKdθ_total=i)
        end
        if return_anal
            append!(return_vec, [anal_dKdθs])
        end
    end

    if return_dif | return_bool
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
                if print_stuff
                    println("dK/dθ$ind has a average ratioed difference of: ", val)
                end
            end
        end
        if print_stuff
            println("Maximum average dK/dθ ratioed difference of: ", max_val)
        end
        return_vec = no_differences
    end

    return return_vec

end


"Compare the performance of two different kernel functions"
function func_comp(n::Int, kernel1, kernel2; hyperparameters::Array{Float64,1}=[1.,2], dif::Float64=0.1)
    @time [kernel1(hyperparameters, dif) for i in 1:n]
    @time [kernel2(hyperparameters, dif) for i in 1:n]
end
