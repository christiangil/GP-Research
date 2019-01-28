include("all_functions.jl")


function kernel1(hyperparameters, x1, x2; dorder=[0,0], dKdθ=0, products = 0)

    # finding required differences between inputs
    dif_vec = x1 - x2  # raw difference vectors

    function kernel_piece(hyper, dif, products_line)

        # amount of possible derivatives on each function
        dorders = length(hyper) + length(dorder)

        # get the derivative orders for functions 1 and 2
        dorder1 = convert(Array{Int64,1}, products_line[2:(dorders+1)])
        dorder2 = convert(Array{Int64,1}, products_line[(dorders + 2):(2 * dorders+1)])

        # return 0 if you know that that portion will equal 0
        # this is when you are deriving one of the kernels by a hyperparameter
        # of the other kernel
        if (((dorder1[length(dorder) + 2] == 1) | (dorder1[length(dorder) + 3] == 1))
            | (dorder2[length(dorder) + 1] == 1))

            return 0

        else

            # use the properly differentiated version of kernel function 1
            if dorder1[length(dorder) + 1] == 1
                func1 = dRBFdλ_kernel([hyper[1]], dif, dorder1[1:length(dorder)])
            # elseif ((dorder1[length(dorder) + 2] == 1) | (dorder1[length(dorder) + 3] == 1))
            #     func1 = 0
            else
                func1 = dRBFdt_kernel([hyper[1]], dif, dorder1[1:length(dorder)])
            end

            # use the properly differentiated version of kernel function 2
            # if dorder2[length(dorder) + 1] == 1
            #     func2 = 0
            if dorder2[length(dorder) + 2] == 1
                func2 = dPeriodicdλ_kernel(hyper[2:3], dif, dorder2[1:length(dorder)])
            elseif dorder2[length(dorder) + 3] == 1
                func2 = dPeriodicdp_kernel(hyper[2:3], dif, dorder2[1:length(dorder)])
            else
                func2 = dPeriodicdt_kernel(hyper[2:3], dif, dorder2[1:length(dorder)])
            end

            return func1 * func2

        end
    end

    # calculate the product rule coefficients and dorders if they aren't passed
    if products == 0
        non_coefficient_hyperparameters = length(hyperparameters) - total_coefficients
        new_dorder = append!(copy(dorder), zeros(non_coefficient_hyperparameters))
        # differentiate by RBF kernel length
        if dKdθ == length(hyperparameters) - 2
            new_dorder[length(dorder) + 1] = 1
        # differentiate by Periodic kernel length
        elseif dKdθ == length(hyperparameters) - 1
            new_dorder[length(dorder) + 2] = 1
        # differentiate by Periodic kernel period
        elseif dKdθ == length(hyperparameters)
            new_dorder[length(dorder) + 3] = 1
        end
        products = product_rule(new_dorder)
    end

    # add all of the differentiated kernels together according to the product rule
    final = sum([products[i, 1] * kernel_piece(hyperparameters[(length(hyperparameters) - 2):length(hyperparameters)], dif_vec[1], products[i,:]) for i in 1:size(products, 1)])

    return final

end


function kernel2(hyperparameters, x1, x2; dorder=[0,0], dKdθ=0, products = 0)

    # finding required differences between inputs
    dif_vec = x1 - x2  # raw difference vectors

    function kernel_piece(hyper, dif, products_line)

        # amount of possible derivatives on each function
        dorders = length(hyper) + length(dorder)

        # get the derivative orders for functions 1 and 2
        dorder1 = convert(Array{Int64,1}, products_line[2:(dorders+1)])
        dorder2 = convert(Array{Int64,1}, products_line[(dorders + 2):(2 * dorders+1)])

        # return 0 if you know that that portion will equal 0
        # this is when you are deriving one of the kernels by a hyperparameter
        # of the other kernel
        if (((dorder1[length(dorder) + 2] == 1) | (dorder1[length(dorder) + 3] == 1))
            | (dorder2[length(dorder) + 1] == 1))

            return 0

        else

            # use the properly differentiated version of kernel function 1
            func1 = RBF_kernel([hyper[1]], dif; dorder=dorder1[1:3])
            # if dorder1[length(dorder) + 1] == 1
            #     func1 = dRBFdλ_kernel([hyper[1]], dif, dorder1[1:length(dorder)])
            # else
            #     func1 = dRBFdt_kernel([hyper[1]], dif, dorder1[1:length(dorder)])
            # end

            # use the properly differentiated version of kernel function 2\
            dorder2
            func2 = Periodic_kernel(hyper[2:3], dif; dorder=append(dorder2[1:2], [dorder2[5]], [dorder2[4]]))
            # if dorder2[length(dorder) + 2] == 1
            #     func2 = dPeriodicdλ_kernel(hyper[2:3], dif, dorder2[1:length(dorder)])
            # elseif dorder2[length(dorder) + 3] == 1
            #     func2 = dPeriodicdp_kernel(hyper[2:3], dif, dorder2[1:length(dorder)])
            # else
            #     func2 = dPeriodicdt_kernel(hyper[2:3], dif, dorder2[1:length(dorder)])
            # end

            return func1 * func2

        end
    end

    # calculate the product rule coefficients and dorders if they aren't passed
    if products == 0
        non_coefficient_hyperparameters = length(hyperparameters) - total_coefficients
        new_dorder = append!(copy(dorder), zeros(non_coefficient_hyperparameters))
        # differentiate by RBF kernel length
        if dKdθ == length(hyperparameters) - 2
            new_dorder[length(dorder) + 1] = 1
        # differentiate by Periodic kernel length
        elseif dKdθ == length(hyperparameters) - 1
            new_dorder[length(dorder) + 2] = 1
        # differentiate by Periodic kernel period
        elseif dKdθ == length(hyperparameters)
            new_dorder[length(dorder) + 3] = 1
        end
        products = product_rule(new_dorder)
    end

    # add all of the differentiated kernels together according to the product rule
    final = sum([products[i, 1] * kernel_piece(hyperparameters[(length(hyperparameters) - 2):length(hyperparameters)], dif_vec[1], products[i,:]) for i in 1:size(products, 1)])

    return final

end


function kernel3(hyperparameters, t1::Float64, t2::Float64; dorder=zeros(2), dKdθ=0)

    dif = t1 - t2
    dorder_tot = append!(dorder, zeros(length(hyperparameters) - total_coefficients))
    if dKdθ > total_coefficients
        dorder_tot[2 + dKdθ - total_coefficients] = 1
    end

    final = Quasi_periodic_kernel(hyperparameters[(total_coefficients + 1):end], dif; dorder=dorder_tot)

    return final

end

# how many components you will use
n_out = 3
# how many differentiated versions of the original GP you will use
n_dif = 3

total_coefficients = n_out * n_dif

a = ones(n_out, n_dif) / 20
kernel_lengths = [1, 1, 1] / 1.5
hyperparameters = append!(collect(Iterators.flatten(a)), kernel_lengths)

include("kernels/Quasi_periodic_kernel.jl")

x=collect(1:1000)./200 .- 2.50001
function plot_dKdθ(dKdθ)
    init_plot()
    plot(x, [kernel1(hyperparameters, i, 0.; dorder=[1,0], dKdθ=dKdθ) for i in x])
    plot(x, [kernel2(hyperparameters, i, 0.; dorder=[1,0], dKdθ=dKdθ) for i in x])
    savefig("test.pdf")
end

function plot_approx_dKdθ(dKdθ)
    init_plot()
    dif = 1e-6
    f = kernel3
    y1 = [f(hyperparameters, i, 0.; dorder=[0,0], dKdθ=0) for i in x]
    hyper = copy(hyperparameters)
    hyper[dKdθ] += dif
    y2 = [f(hyper, i, 0.; dorder=[0,0], dKdθ=0) for i in x]
    plot(x, (y2.-y1)./dif)
    # plot(x, y1)
    plot(x, [f(hyperparameters, i, 0.; dorder=[0,0], dKdθ=dKdθ) for i in x])
    savefig("test.pdf")
end

plot_approx_dKdθ(10)


@elapsed [kernel1(hyperparameters, i, 0.; dorder=[1,0], dKdθ=12) for i in x]
@elapsed [kernel2(hyperparameters, i, 0.; dorder=[1,0], dKdθ=12) for i in x]
@elapsed [kernel3(hyperparameters, i, 0.; dorder=[1,0], dKdθ=12) for i in x]

##########################################

hyper = [2.4,3.5]
dif = 1.5
include("kernels/RBF_kernel.jl")
# dorder = [2,2]
# dRBFdλ_kernel([hyper[1]], dif, dorder)
# RBF_kernel([hyper[1]], dif; dorder=append!(copy(dorder),[1]))
# dRBFdt_kernel([hyper[1]], dif, dorder)
# RBF_kernel([hyper[1]], dif; dorder=append!(copy(dorder),[0]))

include("kernels/Periodic_kernel.jl")
dorder = [2,2]
dPeriodicdλ_kernel(hyper, dif, dorder)
Periodic_kernel(hyper, dif; dorder=append!(copy(dorder),[1,0]))
dPeriodicdp_kernel(hyper, dif, dorder)
Periodic_kernel(hyper, dif; dorder=append!(copy(dorder),[0,1]))
dPeriodicdt_kernel(hyper, dif, dorder)
Periodic_kernel(hyper, dif; dorder=append!(copy(dorder),[0,0]))

# use the properly differentiated version of kernel function 2
# if dorder2[length(dorder) + 1] == 1
#     func2 = 0
if dorder2[length(dorder) + 2] == 1
func2 = dPeriodicdλ_kernel(hyper[2:3], dif, dorder2[1:length(dorder)])
elseif dorder2[length(dorder) + 3] == 1
func2 = dPeriodicdp_kernel(hyper[2:3], dif, dorder2[1:length(dorder)])
else
func2 = dPeriodicdt_kernel(hyper[2:3], dif, dorder2[1:length(dorder)])
