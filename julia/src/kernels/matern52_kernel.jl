

"""
matern52_kernel function created by kernel_coder(). Requires 1 hyperparameters. Likely created using matern52_kernel_base() as an input. 
Use with include("kernels/matern52_kernel.jl").
"""
function matern52_kernel(hyperparameters::Union{Array{T1,1},Array{Any,1}}, dif::Real; dorder::Array{T2,1}=zeros(1)) where {T1<:Real, T2<:Real}

    @assert length(hyperparameters)==1 "hyperparameters is the wrong length"
    if dorder==zeros(1)
        dorder = zeros(length(hyperparameters) + 2)
    else
        @assert length(dorder)==(length(hyperparameters) + 2) "dorder is the wrong length"
    end
    dorder = convert(Array{Int64,1}, dorder)

    kernel_length = hyperparameters[3-2]

    if dorder==[2, 2, 1]
        func = -400.0*exp(-2.23606797749979*dif/kernel_length)/kernel_length^5 + 223.606797749979*exp(-2.23606797749979*dif/kernel_length)*dif/kernel_length^6 - 67.0820393249937*exp(-2.23606797749979*dif/kernel_length)*(-3.33333333333333*dif/kernel_length^2 - 2.23606797749979*kernel_length^(-1))/kernel_length^4 + 67.0820393249937*exp(-2.23606797749979*dif/kernel_length)*(3.33333333333333*dif/kernel_length^2 + 2.23606797749979*kernel_length^(-1))/kernel_length^4 - 22.3606797749979*exp(-2.23606797749979*dif/kernel_length)*(-6.66666666666667*dif/kernel_length^3 - 2.23606797749979*kernel_length^(-2))/kernel_length^3 + 22.3606797749979*exp(-2.23606797749979*dif/kernel_length)*(6.66666666666667*dif/kernel_length^3 + 2.23606797749979*kernel_length^(-2))/kernel_length^3 + 25.0*exp(-2.23606797749979*dif/kernel_length)*(-2.23606797749979*dif/kernel_length^2 - 3.33333333333333*dif^2/kernel_length^3)/kernel_length^4 - 100.0*exp(-2.23606797749979*dif/kernel_length)*(1 + 2.23606797749979*dif/kernel_length + 1.66666666666667*dif^2/kernel_length^2)/kernel_length^5 + 50.0*exp(-2.23606797749979*dif/kernel_length)*dif*(-3.33333333333333*dif/kernel_length^2 - 2.23606797749979*kernel_length^(-1))/kernel_length^5 - 50.0*exp(-2.23606797749979*dif/kernel_length)*dif*(3.33333333333333*dif/kernel_length^2 + 2.23606797749979*kernel_length^(-1))/kernel_length^5 + 55.9016994374948*exp(-2.23606797749979*dif/kernel_length)*dif*(1 + 2.23606797749979*dif/kernel_length + 1.66666666666667*dif^2/kernel_length^2)/kernel_length^6
    end

    if dorder==[1, 2, 1]
        func = 67.0820393249937*exp(-2.23606797749979*dif/kernel_length)/kernel_length^4 - 50.0*exp(-2.23606797749979*dif/kernel_length)*dif/kernel_length^5 + 20.0*exp(-2.23606797749979*dif/kernel_length)*(-3.33333333333333*dif/kernel_length^2 - 2.23606797749979*kernel_length^(-1))/kernel_length^3 - 10.0*exp(-2.23606797749979*dif/kernel_length)*(3.33333333333333*dif/kernel_length^2 + 2.23606797749979*kernel_length^(-1))/kernel_length^3 + 5.0*exp(-2.23606797749979*dif/kernel_length)*(-6.66666666666667*dif/kernel_length^3 - 2.23606797749979*kernel_length^(-2))/kernel_length^2 - 10.0*exp(-2.23606797749979*dif/kernel_length)*(6.66666666666667*dif/kernel_length^3 + 2.23606797749979*kernel_length^(-2))/kernel_length^2 - 11.180339887499*exp(-2.23606797749979*dif/kernel_length)*(-2.23606797749979*dif/kernel_length^2 - 3.33333333333333*dif^2/kernel_length^3)/kernel_length^3 + 33.5410196624969*exp(-2.23606797749979*dif/kernel_length)*(1 + 2.23606797749979*dif/kernel_length + 1.66666666666667*dif^2/kernel_length^2)/kernel_length^4 - 22.3606797749979*exp(-2.23606797749979*dif/kernel_length)*dif*(-3.33333333333333*dif/kernel_length^2 - 2.23606797749979*kernel_length^(-1))/kernel_length^4 + 11.180339887499*exp(-2.23606797749979*dif/kernel_length)*dif*(3.33333333333333*dif/kernel_length^2 + 2.23606797749979*kernel_length^(-1))/kernel_length^4 - 25.0*exp(-2.23606797749979*dif/kernel_length)*dif*(1 + 2.23606797749979*dif/kernel_length + 1.66666666666667*dif^2/kernel_length^2)/kernel_length^5
    end

    if dorder==[0, 2, 1]
        func = -6.66666666666667*exp(-2.23606797749979*dif/kernel_length)/kernel_length^3 + 7.4535599249993*exp(-2.23606797749979*dif/kernel_length)*dif/kernel_length^4 - 4.47213595499958*exp(-2.23606797749979*dif/kernel_length)*(-3.33333333333333*dif/kernel_length^2 - 2.23606797749979*kernel_length^(-1))/kernel_length^2 + 4.47213595499958*exp(-2.23606797749979*dif/kernel_length)*(6.66666666666667*dif/kernel_length^3 + 2.23606797749979*kernel_length^(-2))/kernel_length + 5.0*exp(-2.23606797749979*dif/kernel_length)*(-2.23606797749979*dif/kernel_length^2 - 3.33333333333333*dif^2/kernel_length^3)/kernel_length^2 - 10.0*exp(-2.23606797749979*dif/kernel_length)*(1 + 2.23606797749979*dif/kernel_length + 1.66666666666667*dif^2/kernel_length^2)/kernel_length^3 + 10.0*exp(-2.23606797749979*dif/kernel_length)*dif*(-3.33333333333333*dif/kernel_length^2 - 2.23606797749979*kernel_length^(-1))/kernel_length^3 + 11.180339887499*exp(-2.23606797749979*dif/kernel_length)*dif*(1 + 2.23606797749979*dif/kernel_length + 1.66666666666667*dif^2/kernel_length^2)/kernel_length^4
    end

    if dorder==[2, 1, 1]
        func = -67.0820393249937*exp(-2.23606797749979*dif/kernel_length)/kernel_length^4 + 50.0*exp(-2.23606797749979*dif/kernel_length)*dif/kernel_length^5 - 10.0*exp(-2.23606797749979*dif/kernel_length)*(-3.33333333333333*dif/kernel_length^2 - 2.23606797749979*kernel_length^(-1))/kernel_length^3 + 20.0*exp(-2.23606797749979*dif/kernel_length)*(3.33333333333333*dif/kernel_length^2 + 2.23606797749979*kernel_length^(-1))/kernel_length^3 - 10.0*exp(-2.23606797749979*dif/kernel_length)*(-6.66666666666667*dif/kernel_length^3 - 2.23606797749979*kernel_length^(-2))/kernel_length^2 + 5.0*exp(-2.23606797749979*dif/kernel_length)*(6.66666666666667*dif/kernel_length^3 + 2.23606797749979*kernel_length^(-2))/kernel_length^2 + 11.180339887499*exp(-2.23606797749979*dif/kernel_length)*(-2.23606797749979*dif/kernel_length^2 - 3.33333333333333*dif^2/kernel_length^3)/kernel_length^3 - 33.5410196624969*exp(-2.23606797749979*dif/kernel_length)*(1 + 2.23606797749979*dif/kernel_length + 1.66666666666667*dif^2/kernel_length^2)/kernel_length^4 + 11.180339887499*exp(-2.23606797749979*dif/kernel_length)*dif*(-3.33333333333333*dif/kernel_length^2 - 2.23606797749979*kernel_length^(-1))/kernel_length^4 - 22.3606797749979*exp(-2.23606797749979*dif/kernel_length)*dif*(3.33333333333333*dif/kernel_length^2 + 2.23606797749979*kernel_length^(-1))/kernel_length^4 + 25.0*exp(-2.23606797749979*dif/kernel_length)*dif*(1 + 2.23606797749979*dif/kernel_length + 1.66666666666667*dif^2/kernel_length^2)/kernel_length^5
    end

    if dorder==[1, 1, 1]
        func = 6.66666666666667*exp(-2.23606797749979*dif/kernel_length)/kernel_length^3 - 7.4535599249993*exp(-2.23606797749979*dif/kernel_length)*dif/kernel_length^4 + 2.23606797749979*exp(-2.23606797749979*dif/kernel_length)*(-3.33333333333333*dif/kernel_length^2 - 2.23606797749979*kernel_length^(-1))/kernel_length^2 - 2.23606797749979*exp(-2.23606797749979*dif/kernel_length)*(3.33333333333333*dif/kernel_length^2 + 2.23606797749979*kernel_length^(-1))/kernel_length^2 + 2.23606797749979*exp(-2.23606797749979*dif/kernel_length)*(-6.66666666666667*dif/kernel_length^3 - 2.23606797749979*kernel_length^(-2))/kernel_length - 2.23606797749979*exp(-2.23606797749979*dif/kernel_length)*(6.66666666666667*dif/kernel_length^3 + 2.23606797749979*kernel_length^(-2))/kernel_length - 5.0*exp(-2.23606797749979*dif/kernel_length)*(-2.23606797749979*dif/kernel_length^2 - 3.33333333333333*dif^2/kernel_length^3)/kernel_length^2 + 10.0*exp(-2.23606797749979*dif/kernel_length)*(1 + 2.23606797749979*dif/kernel_length + 1.66666666666667*dif^2/kernel_length^2)/kernel_length^3 - 5.0*exp(-2.23606797749979*dif/kernel_length)*dif*(-3.33333333333333*dif/kernel_length^2 - 2.23606797749979*kernel_length^(-1))/kernel_length^3 + 5.0*exp(-2.23606797749979*dif/kernel_length)*dif*(3.33333333333333*dif/kernel_length^2 + 2.23606797749979*kernel_length^(-1))/kernel_length^3 - 11.180339887499*exp(-2.23606797749979*dif/kernel_length)*dif*(1 + 2.23606797749979*dif/kernel_length + 1.66666666666667*dif^2/kernel_length^2)/kernel_length^4
    end

    if dorder==[0, 1, 1]
        func = exp(-2.23606797749979*dif/kernel_length)*(6.66666666666667*dif/kernel_length^3 + 2.23606797749979*kernel_length^(-2)) + 2.23606797749979*exp(-2.23606797749979*dif/kernel_length)*(-2.23606797749979*dif/kernel_length^2 - 3.33333333333333*dif^2/kernel_length^3)/kernel_length - 2.23606797749979*exp(-2.23606797749979*dif/kernel_length)*(1 + 2.23606797749979*dif/kernel_length + 1.66666666666667*dif^2/kernel_length^2)/kernel_length^2 + 2.23606797749979*exp(-2.23606797749979*dif/kernel_length)*dif*(-3.33333333333333*dif/kernel_length^2 - 2.23606797749979*kernel_length^(-1))/kernel_length^2 + 5.0*exp(-2.23606797749979*dif/kernel_length)*dif*(1 + 2.23606797749979*dif/kernel_length + 1.66666666666667*dif^2/kernel_length^2)/kernel_length^3
    end

    if dorder==[2, 0, 1]
        func = -6.66666666666667*exp(-2.23606797749979*dif/kernel_length)/kernel_length^3 + 7.4535599249993*exp(-2.23606797749979*dif/kernel_length)*dif/kernel_length^4 + 4.47213595499958*exp(-2.23606797749979*dif/kernel_length)*(3.33333333333333*dif/kernel_length^2 + 2.23606797749979*kernel_length^(-1))/kernel_length^2 - 4.47213595499958*exp(-2.23606797749979*dif/kernel_length)*(-6.66666666666667*dif/kernel_length^3 - 2.23606797749979*kernel_length^(-2))/kernel_length + 5.0*exp(-2.23606797749979*dif/kernel_length)*(-2.23606797749979*dif/kernel_length^2 - 3.33333333333333*dif^2/kernel_length^3)/kernel_length^2 - 10.0*exp(-2.23606797749979*dif/kernel_length)*(1 + 2.23606797749979*dif/kernel_length + 1.66666666666667*dif^2/kernel_length^2)/kernel_length^3 - 10.0*exp(-2.23606797749979*dif/kernel_length)*dif*(3.33333333333333*dif/kernel_length^2 + 2.23606797749979*kernel_length^(-1))/kernel_length^3 + 11.180339887499*exp(-2.23606797749979*dif/kernel_length)*dif*(1 + 2.23606797749979*dif/kernel_length + 1.66666666666667*dif^2/kernel_length^2)/kernel_length^4
    end

    if dorder==[1, 0, 1]
        func = exp(-2.23606797749979*dif/kernel_length)*(-6.66666666666667*dif/kernel_length^3 - 2.23606797749979*kernel_length^(-2)) - 2.23606797749979*exp(-2.23606797749979*dif/kernel_length)*(-2.23606797749979*dif/kernel_length^2 - 3.33333333333333*dif^2/kernel_length^3)/kernel_length + 2.23606797749979*exp(-2.23606797749979*dif/kernel_length)*(1 + 2.23606797749979*dif/kernel_length + 1.66666666666667*dif^2/kernel_length^2)/kernel_length^2 + 2.23606797749979*exp(-2.23606797749979*dif/kernel_length)*dif*(3.33333333333333*dif/kernel_length^2 + 2.23606797749979*kernel_length^(-1))/kernel_length^2 - 5.0*exp(-2.23606797749979*dif/kernel_length)*dif*(1 + 2.23606797749979*dif/kernel_length + 1.66666666666667*dif^2/kernel_length^2)/kernel_length^3
    end

    if dorder==[0, 0, 1]
        func = exp(-2.23606797749979*dif/kernel_length)*(-2.23606797749979*dif/kernel_length^2 - 3.33333333333333*dif^2/kernel_length^3) + 2.23606797749979*exp(-2.23606797749979*dif/kernel_length)*dif*(1 + 2.23606797749979*dif/kernel_length + 1.66666666666667*dif^2/kernel_length^2)/kernel_length^2
    end

    if dorder==[2, 2, 0]
        func = 100.0*exp(-2.23606797749979*dif/kernel_length)/kernel_length^4 + 22.3606797749979*exp(-2.23606797749979*dif/kernel_length)*(-3.33333333333333*dif/kernel_length^2 - 2.23606797749979*kernel_length^(-1))/kernel_length^3 - 22.3606797749979*exp(-2.23606797749979*dif/kernel_length)*(3.33333333333333*dif/kernel_length^2 + 2.23606797749979*kernel_length^(-1))/kernel_length^3 + 25.0*exp(-2.23606797749979*dif/kernel_length)*(1 + 2.23606797749979*dif/kernel_length + 1.66666666666667*dif^2/kernel_length^2)/kernel_length^4
    end

    if dorder==[1, 2, 0]
        func = -22.3606797749979*exp(-2.23606797749979*dif/kernel_length)/kernel_length^3 - 10.0*exp(-2.23606797749979*dif/kernel_length)*(-3.33333333333333*dif/kernel_length^2 - 2.23606797749979*kernel_length^(-1))/kernel_length^2 + 5.0*exp(-2.23606797749979*dif/kernel_length)*(3.33333333333333*dif/kernel_length^2 + 2.23606797749979*kernel_length^(-1))/kernel_length^2 - 11.180339887499*exp(-2.23606797749979*dif/kernel_length)*(1 + 2.23606797749979*dif/kernel_length + 1.66666666666667*dif^2/kernel_length^2)/kernel_length^3
    end

    if dorder==[0, 2, 0]
        func = 3.33333333333333*exp(-2.23606797749979*dif/kernel_length)/kernel_length^2 + 4.47213595499958*exp(-2.23606797749979*dif/kernel_length)*(-3.33333333333333*dif/kernel_length^2 - 2.23606797749979*kernel_length^(-1))/kernel_length + 5.0*exp(-2.23606797749979*dif/kernel_length)*(1 + 2.23606797749979*dif/kernel_length + 1.66666666666667*dif^2/kernel_length^2)/kernel_length^2
    end

    if dorder==[2, 1, 0]
        func = 22.3606797749979*exp(-2.23606797749979*dif/kernel_length)/kernel_length^3 + 5.0*exp(-2.23606797749979*dif/kernel_length)*(-3.33333333333333*dif/kernel_length^2 - 2.23606797749979*kernel_length^(-1))/kernel_length^2 - 10.0*exp(-2.23606797749979*dif/kernel_length)*(3.33333333333333*dif/kernel_length^2 + 2.23606797749979*kernel_length^(-1))/kernel_length^2 + 11.180339887499*exp(-2.23606797749979*dif/kernel_length)*(1 + 2.23606797749979*dif/kernel_length + 1.66666666666667*dif^2/kernel_length^2)/kernel_length^3
    end

    if dorder==[1, 1, 0]
        func = -3.33333333333333*exp(-2.23606797749979*dif/kernel_length)/kernel_length^2 - 2.23606797749979*exp(-2.23606797749979*dif/kernel_length)*(-3.33333333333333*dif/kernel_length^2 - 2.23606797749979*kernel_length^(-1))/kernel_length + 2.23606797749979*exp(-2.23606797749979*dif/kernel_length)*(3.33333333333333*dif/kernel_length^2 + 2.23606797749979*kernel_length^(-1))/kernel_length - 5.0*exp(-2.23606797749979*dif/kernel_length)*(1 + 2.23606797749979*dif/kernel_length + 1.66666666666667*dif^2/kernel_length^2)/kernel_length^2
    end

    if dorder==[0, 1, 0]
        func = exp(-2.23606797749979*dif/kernel_length)*(-3.33333333333333*dif/kernel_length^2 - 2.23606797749979*kernel_length^(-1)) + 2.23606797749979*exp(-2.23606797749979*dif/kernel_length)*(1 + 2.23606797749979*dif/kernel_length + 1.66666666666667*dif^2/kernel_length^2)/kernel_length
    end

    if dorder==[2, 0, 0]
        func = 3.33333333333333*exp(-2.23606797749979*dif/kernel_length)/kernel_length^2 - 4.47213595499958*exp(-2.23606797749979*dif/kernel_length)*(3.33333333333333*dif/kernel_length^2 + 2.23606797749979*kernel_length^(-1))/kernel_length + 5.0*exp(-2.23606797749979*dif/kernel_length)*(1 + 2.23606797749979*dif/kernel_length + 1.66666666666667*dif^2/kernel_length^2)/kernel_length^2
    end

    if dorder==[1, 0, 0]
        func = exp(-2.23606797749979*dif/kernel_length)*(3.33333333333333*dif/kernel_length^2 + 2.23606797749979*kernel_length^(-1)) - 2.23606797749979*exp(-2.23606797749979*dif/kernel_length)*(1 + 2.23606797749979*dif/kernel_length + 1.66666666666667*dif^2/kernel_length^2)/kernel_length
    end

    if dorder==[0, 0, 0]
        func = exp(-2.23606797749979*dif/kernel_length)*(1 + 2.23606797749979*dif/kernel_length + 1.66666666666667*dif^2/kernel_length^2)
    end

    return float(func)

end


return matern52_kernel, 1  # the function handle and the number of kernel hyperparameters
