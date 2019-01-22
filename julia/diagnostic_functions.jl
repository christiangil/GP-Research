# these are all custom diagnostic functions. May help with debugging


"estimate the gradient of nlogL with forward differences"
function est_grad(hyper; dif=0.0001)
    val = nlogL(hyper)
    grad = zeros(length(hyper))
    for i in 1:length(hyper)
        hold = copy(hyper)
        hold[i] += dif
        # println(hold)
        # println(nlogL(hold))
        grad[i] =  (nlogL(hold) - val) / dif
    end
    # println(nlogL(hyper))
    return grad
end


"prints analytical and numerically estimated ∇nlogL"
function test_grad(; dif=0.001, hyper=rand(length(hyperparameters)))
    println()
    println(nlogL(hyper))
    println(hyper)
    G = zeros(length(hyper))
    ∇nlogL(G, hyper)
    println(G)
    println(est_grad(hyper; dif=dif))
end


"estimate the covariance derivatives with forward differences"
function est_dKdθ(hyper; x=[1,2], return_est=true, return_act=false, return_dif=false, dif=0.0001, outputs=2)

    return_vec = []

    if return_est | return_dif

        val = total_covariance(x, x, hyper)
        est_dKdθs = zeros(length(hyper), outputs * length(x), outputs * length(x))
        for i in 1:length(hyper)
            hold = copy(hyper)
            hold[i] += dif
            est_dKdθs[i, :, :] =  (total_covariance(x, x, hold) - val) / dif
        end
        if return_est
            append!(return_vec, [est_dKdθs])
        end
    end

    if return_act | return_dif

        act_dKdθs = zeros(length(hyper), outputs * length(x), outputs * length(x))
        for i in 1:length(hyper)
            act_dKdθs[i, :, :] =  total_covariance(x, x, hyper; dKdθ=i)
        end
        if return_act
            append!(return_vec, [act_dKdθs])
        end
    end

    if return_dif

        difs = signficant_difference(est_dKdθs, act_dKdθs, dif)
        append!(return_vec, [difs])

        println()
        for i in 1:length(hyper)
            if difs[i, :, :] != zeros(outputs * length(x), outputs * length(x))
                println("significant differences in dKdθ" * string(i))
            end
        end
    end

    return return_vec

end


"Compare the performance of two different kernel functions"
function func_comp(n, kernel1, kernel2; hyperparameters::Array{Float64,1}=[1.,2], dif::Float64=0.1)
    @time [kernel1(hyperparameters, dif) for i in 1:n]
    @time [kernel2(hyperparameters, dif) for i in 1:n]
end
