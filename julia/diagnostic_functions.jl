# these are all custom diagnostic functions. May help with debugging


"estimate the gradient of nlogL with forward differences"
function est_grad(hyper::Array{Float64,1}; dif::Float64=0.0001)
    val = nlogL_Jones(hyper)
    grad = zeros(length(hyper))
    for i in 1:length(hyper)
        hold = copy(hyper)
        hold[i] += dif
        # println(hold)
        # println(nlogL(hold))
        grad[i] =  (nlogL_Jones(hold) - val) / dif
    end
    # println(nlogL(hyper))
    return grad
end


"prints analytical and numerically estimated ∇nlogL"
function test_grad(kernel_hyperparameters::Array{Float64,1}; dif::Float64=1e-6, print_stuff::Bool=true)
    total_hyperparameters = append!(collect(Iterators.flatten(problem_definition.a0)), kernel_hyperparameters)

    # println(nlogL_Jones(hyper))
    G = zeros(length(total_hyperparameters))
    ∇nlogL_Jones(G, total_hyperparameters)
    est_G = est_grad(total_hyperparameters; dif=dif)

    if print_stuff
        println()
        println("hyperparameters: ", total_hyperparameters)
        println("analytical: ", G)
        println("numerical : ", est_G)
    end

    no_mismatch = true
    for i in 1:length(total_hyperparameters)
        if (!(isapprox(G[i], est_G[i], rtol=1e-2))) & (G[i] != 0)
            # println("mismatch dnlogL/dθ" * string(i))
            no_mismatch = false
        end
    end
    return no_mismatch
end


"estimate the covariance derivatives with forward differences"
function est_dKdθ(prob_def::Jones_problem_definition, kernel_hyperparameters::Array{Float64,1}; return_est::Bool=true, return_act::Bool=false, return_dif::Bool=false, return_bool::Bool=false, dif::Float64=1e-6, print_stuff::Bool=true)

    total_hyperparameters = append!(collect(Iterators.flatten(prob_def.a0)), kernel_hyperparameters)
    x = prob_def.x_obs
    return_vec = []
    coeff_orders = coefficient_orders(prob_def.n_out, prob_def.n_dif, a=prob_def.a0)

    # println(total_hyperparameters)
    if return_est | return_dif | return_bool

        val = covariance(prob_def, total_hyperparameters; coeff_orders=coeff_orders)
        est_dKdθs = zeros(length(total_hyperparameters), prob_def.n_out * length(x), prob_def.n_out * length(x))
        for i in 1:length(total_hyperparameters)
            hold = copy(total_hyperparameters)
            hold[i] += dif
            est_dKdθs[i, :, :] =  (covariance(prob_def, hold; coeff_orders=coeff_orders) - val) / dif
        end
        if return_est
            append!(return_vec, [est_dKdθs])
        end
    end

    if return_act | return_dif | return_bool

        act_dKdθs = zeros(length(total_hyperparameters), problem_definition.n_out * length(x), problem_definition.n_out * length(x))
        for i in 1:length(total_hyperparameters)
            act_dKdθs[i, :, :] =  covariance(problem_definition, total_hyperparameters; dKdθ_total=i, coeff_orders=coeff_orders)
        end
        if return_act
            append!(return_vec, [act_dKdθs])
        end
    end

    if return_dif | return_bool

        # difs = signficant_difference(est_dKdθs, act_dKdθs, dif)
        difs = est_dKdθs - act_dKdθs

        append!(return_vec, [difs])

        # println()
        # for i in 1:length(hyper)
        #     if difs[i, :, :] != zeros(problem_definition.n_out * length(x), problem_definition.n_out * length(x))
        #         println("significant differences in dKdθ" * string(i))
        #     end
        # end
    end

    if return_bool
        no_differences = true
        min_thres = 0.05
        for ind in 1:length(total_hyperparameters)
            est = est_dKdθs[ind,:,:]
            est[est .== 0] .= 1e-8
            val = maximum(abs.(difs[ind,:,:] ./ est))
            if (val>min_thres) & (val!=1)
                no_differences = false
                if print_stuff
                    println("hyperparameter $ind has a maximum ratioed difference of: ", val)
                end
            end
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
