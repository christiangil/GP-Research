# these are all custom diagnostic functions. May help with debugging

# save all of the A matrices calculated in the total covariance function when using
# multiple GP outputs and GP models that include derivatives of themselves
function save_A(A)
    for i in 1:size(A)[1]
        if length(size(A)) == 2
            for j in 1:size(A)[2]
                save("A" * string(i) * string(j) * ".jld", "data", A[i,j])
            end
        else
            save("A" * string(i) * "1.jld", "data", A[i])
        end
    end
end


# plot a trace of all of the A matrices calculated in the total covariance function when using
# multiple GP outputs and GP models that include derivatives of themselves
function plot_A(;dims=[n_dif, n_dif])

    # intializing the data matrix as something other than an Any matrix
    data = [1]
    for i in 1:dims[1]
        for j in 1:dims[2]
            test = load("A" * string(i) * string(j) * ".jld")["data"]
            # plt = plot(heatmap(z=test, colorscale="Viridis", showscale=false))
            plt = plot(line_trace(x_samp, test[1:convert(Int64, GP_sample_amount), convert(Int64, GP_sample_amount/2)]), layout)
            data = hcat(data,[plt])
        end
    end
    data = data[2:length(data)]

    return return_plot_matrix(data)

end

# plot a trace of all of the K sub-matrices
function plot_K(K_samp)

    # intializing the data matrix as something other than an Any matrix
    data = [1]
    GP_sample_amount = size(K_samp)[1] / n_out
    for i in 1:n_out
        for j in 1:n_out
            test = (K_samp[((i - 1) * GP_sample_amount + 1):(i * GP_sample_amount),
                ((j - 1) * GP_sample_amount + 1):(j * GP_sample_amount)])
            # plt = plot(heatmap(z=test, colorscale="Viridis", showscale=false))
            plt = plot(line_trace(x_samp, test[1:convert(Int64, GP_sample_amount), convert(Int64, GP_sample_amount/2)]), layout)
            data = hcat(data,[plt])
        end
    end
    data = data[2:length(data)]

    return return_plot_matrix(data)

end


# estimate the gradient of nlogL with forward differences
# could use code like this to compare the analytical version with the numerical version
#
# G = zeros(length(hyperparameters))
# ∇nlogL(G, hyperparameters)
# println(G)
# println(est_grad(hyperparameters; dif=0.0001))
#
function est_grad(hyper; dif=0.0001)
    val = nlogL(hyper)
    grad = zeros(length(hyper))
    for i in 1:length(hyper)
        hold = copy(hyper)
        hold[i] += dif
        grad[i] =  (nlogL(hold) - val) / dif
    end
    return grad
end


# estimate the covariance derivatives with forward differences
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


function plot_dfdhyper_differences(dorder; dif=0.0001, x=linspace(-2,2,1000))
    # dorder = [2, 2]  # dorder is of this form
    y1 = [dPeriodicdλ_kernel([1, 1], i, dorder) for i in x]
    y2 = [(dPeriodicdt_kernel([1 + dif / 2, 1], i, dorder) - dPeriodicdt_kernel([1 - dif / 2, 1], i, dorder)) / dif for i in x]
    y1my2 = signficant_difference(y1, y2, dif)
    y1 = [dPeriodicdλ_kernel([1, 1], i, dorder) for i in x]
    y2 = [(dPeriodicdt_kernel([1 + dif / 2, 1], i, dorder) - dPeriodicdt_kernel([1 - dif / 2, 1], i, dorder)) / dif for i in x]
    y1my2 = signficant_difference(y1, y2, dif)
    y3 = [dPeriodicdp_kernel([1, 1], i, dorder) for i in x]
    y4 = [(dPeriodicdt_kernel([1, 1 + dif / 2], i, dorder) - dPeriodicdt_kernel([1, 1 - dif / 2], i, dorder)) / dif for i in x]
    y3my4 = signficant_difference(y3, y4, dif)
    return [plot([line_trace(x, y1), line_trace(x, y2)]) plot([line_trace(x, y3), line_trace(x, y4)])
         plot(line_trace(x, y1my2)) plot(line_trace(x, y3my4))]
end


function plot_dfdt_differences(f, n; dif=0.001, x=linspace(-2,2,1000))
    # n is just an integer derivative order e.g. n = 1
    # f is a function where the two inputs are the x and the derivative order e.g.
    # f(x, n) = dRBFdt_kernel([1, 1], x, [n, 0])

    y1 = [f(i, n) for i in x]
    f_mod(x) = f(x, 0)
    y2 = [finite_differences(f_mod, i, n, dif) for i in x]
    y1my2 = signficant_difference(y1, y2, dif)
    return [plot([line_trace(x, y1), line_trace(x, y2)]), plot(line_trace(x, y1my2))]

end
