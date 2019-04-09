

"""
matern52_kernel function created by kernel_coder(). Requires 1 hyperparameters. Likely created using matern52_kernel_base() as an input. 
Use with include("src/kernels/matern52_kernel.jl").
hyperparameters == ["kernel_length"]
"""
function matern52_kernel(hyperparameters::AbstractArray{T1,1}, dif::Real; dorder::AbstractArray{T2,1}=zeros(length(hyperparameters) + 2)) where {T1<:Real, T2<:Real}

    @assert length(hyperparameters)==1 "hyperparameters is the wrong length"
    @assert length(dorder)==(length(hyperparameters) + 2) "dorder is the wrong length"
    dorder = convert(Array{Int64,1}, dorder)
    even_time_derivative = 2 * iseven(dorder[2]) - 1
    @assert maximum(dorder) < 3 "No more than two time derivatives for either t1 or t2 can be calculated"

    dorder = append!([sum(dorder[1:2])], dorder[3:end])

    kernel_length = hyperparameters[1]

    dif_positive = (dif >= 0)  # store original sign of dif (for correcting odd kernel derivatives)
    dif = abs(dif)  # this is corected for later

    if dorder==[4, 1]
        func = -400.0*exp(-2.23606797749979*dif/kernel_length)/kernel_length^5 + 223.606797749979*exp(-2.23606797749979*dif/kernel_length)*dif/kernel_length^6 - 100.0*exp(-2.23606797749979*dif/kernel_length)*(1 + 2.23606797749979*dif*(1 + 0.74535599249993*dif/kernel_length)/kernel_length)/kernel_length^5 - 44.7213595499958*exp(-2.23606797749979*dif/kernel_length)*(0.0 - 5.0*dif/kernel_length^3 - 2.23606797749979*(1 + 0.74535599249993*dif/kernel_length)/kernel_length^2)/kernel_length^3 + 134.164078649987*exp(-2.23606797749979*dif/kernel_length)*(0.0 + 1.66666666666667*dif/kernel_length^2 + 2.23606797749979*(1 + 0.74535599249993*dif/kernel_length)/kernel_length)/kernel_length^4 + 25.0*exp(-2.23606797749979*dif/kernel_length)*(0.0 - 1.66666666666667*dif^2/kernel_length^3 - 2.23606797749979*dif*(1 + 0.74535599249993*dif/kernel_length)/kernel_length^2)/kernel_length^4 + 55.9016994374948*exp(-2.23606797749979*dif/kernel_length)*dif*(1 + 2.23606797749979*dif*(1 + 0.74535599249993*dif/kernel_length)/kernel_length)/kernel_length^6 - 100.0*exp(-2.23606797749979*dif/kernel_length)*dif*(0.0 + 1.66666666666667*dif/kernel_length^2 + 2.23606797749979*(1 + 0.74535599249993*dif/kernel_length)/kernel_length)/kernel_length^5
    end

    if dorder==[3, 1]
        func = 67.0820393249937*exp(-2.23606797749979*dif/kernel_length)/kernel_length^4 - 50.0*exp(-2.23606797749979*dif/kernel_length)*dif/kernel_length^5 + 33.5410196624969*exp(-2.23606797749979*dif/kernel_length)*(1 + 2.23606797749979*dif*(1 + 0.74535599249993*dif/kernel_length)/kernel_length)/kernel_length^4 + 15.0*exp(-2.23606797749979*dif/kernel_length)*(0.0 - 5.0*dif/kernel_length^3 - 2.23606797749979*(1 + 0.74535599249993*dif/kernel_length)/kernel_length^2)/kernel_length^2 - 30.0*exp(-2.23606797749979*dif/kernel_length)*(0.0 + 1.66666666666667*dif/kernel_length^2 + 2.23606797749979*(1 + 0.74535599249993*dif/kernel_length)/kernel_length)/kernel_length^3 - 11.180339887499*exp(-2.23606797749979*dif/kernel_length)*(0.0 - 1.66666666666667*dif^2/kernel_length^3 - 2.23606797749979*dif*(1 + 0.74535599249993*dif/kernel_length)/kernel_length^2)/kernel_length^3 - 25.0*exp(-2.23606797749979*dif/kernel_length)*dif*(1 + 2.23606797749979*dif*(1 + 0.74535599249993*dif/kernel_length)/kernel_length)/kernel_length^5 + 33.5410196624969*exp(-2.23606797749979*dif/kernel_length)*dif*(0.0 + 1.66666666666667*dif/kernel_length^2 + 2.23606797749979*(1 + 0.74535599249993*dif/kernel_length)/kernel_length)/kernel_length^4
    end

    if dorder==[2, 1]
        func = -6.66666666666667*exp(-2.23606797749979*dif/kernel_length)/kernel_length^3 + 7.4535599249993*exp(-2.23606797749979*dif/kernel_length)*dif/kernel_length^4 - 10.0*exp(-2.23606797749979*dif/kernel_length)*(1 + 2.23606797749979*dif*(1 + 0.74535599249993*dif/kernel_length)/kernel_length)/kernel_length^3 - 4.47213595499958*exp(-2.23606797749979*dif/kernel_length)*(0.0 - 5.0*dif/kernel_length^3 - 2.23606797749979*(1 + 0.74535599249993*dif/kernel_length)/kernel_length^2)/kernel_length + 4.47213595499958*exp(-2.23606797749979*dif/kernel_length)*(0.0 + 1.66666666666667*dif/kernel_length^2 + 2.23606797749979*(1 + 0.74535599249993*dif/kernel_length)/kernel_length)/kernel_length^2 + 5.0*exp(-2.23606797749979*dif/kernel_length)*(0.0 - 1.66666666666667*dif^2/kernel_length^3 - 2.23606797749979*dif*(1 + 0.74535599249993*dif/kernel_length)/kernel_length^2)/kernel_length^2 + 11.180339887499*exp(-2.23606797749979*dif/kernel_length)*dif*(1 + 2.23606797749979*dif*(1 + 0.74535599249993*dif/kernel_length)/kernel_length)/kernel_length^4 - 10.0*exp(-2.23606797749979*dif/kernel_length)*dif*(0.0 + 1.66666666666667*dif/kernel_length^2 + 2.23606797749979*(1 + 0.74535599249993*dif/kernel_length)/kernel_length)/kernel_length^3
    end

    if dorder==[1, 1]
        func = exp(-2.23606797749979*dif/kernel_length)*(0.0 - 5.0*dif/kernel_length^3 - 2.23606797749979*(1 + 0.74535599249993*dif/kernel_length)/kernel_length^2) + 2.23606797749979*exp(-2.23606797749979*dif/kernel_length)*(1 + 2.23606797749979*dif*(1 + 0.74535599249993*dif/kernel_length)/kernel_length)/kernel_length^2 - 2.23606797749979*exp(-2.23606797749979*dif/kernel_length)*(0.0 - 1.66666666666667*dif^2/kernel_length^3 - 2.23606797749979*dif*(1 + 0.74535599249993*dif/kernel_length)/kernel_length^2)/kernel_length - 5.0*exp(-2.23606797749979*dif/kernel_length)*dif*(1 + 2.23606797749979*dif*(1 + 0.74535599249993*dif/kernel_length)/kernel_length)/kernel_length^3 + 2.23606797749979*exp(-2.23606797749979*dif/kernel_length)*dif*(0.0 + 1.66666666666667*dif/kernel_length^2 + 2.23606797749979*(1 + 0.74535599249993*dif/kernel_length)/kernel_length)/kernel_length^2
    end

    if dorder==[0, 1]
        func = exp(-2.23606797749979*dif/kernel_length)*(0.0 - 1.66666666666667*dif^2/kernel_length^3 - 2.23606797749979*dif*(1 + 0.74535599249993*dif/kernel_length)/kernel_length^2) + 2.23606797749979*exp(-2.23606797749979*dif/kernel_length)*dif*(1 + 2.23606797749979*dif*(1 + 0.74535599249993*dif/kernel_length)/kernel_length)/kernel_length^2
    end

    if dorder==[4, 0]
        func = 100.0*exp(-2.23606797749979*dif/kernel_length)/kernel_length^4 + 25.0*exp(-2.23606797749979*dif/kernel_length)*(1 + 2.23606797749979*dif*(1 + 0.74535599249993*dif/kernel_length)/kernel_length)/kernel_length^4 - 44.7213595499958*exp(-2.23606797749979*dif/kernel_length)*(0.0 + 1.66666666666667*dif/kernel_length^2 + 2.23606797749979*(1 + 0.74535599249993*dif/kernel_length)/kernel_length)/kernel_length^3
    end

    if dorder==[3, 0]
        func = -22.3606797749979*exp(-2.23606797749979*dif/kernel_length)/kernel_length^3 - 11.180339887499*exp(-2.23606797749979*dif/kernel_length)*(1 + 2.23606797749979*dif*(1 + 0.74535599249993*dif/kernel_length)/kernel_length)/kernel_length^3 + 15.0*exp(-2.23606797749979*dif/kernel_length)*(0.0 + 1.66666666666667*dif/kernel_length^2 + 2.23606797749979*(1 + 0.74535599249993*dif/kernel_length)/kernel_length)/kernel_length^2
    end

    if dorder==[2, 0]
        func = 3.33333333333333*exp(-2.23606797749979*dif/kernel_length)/kernel_length^2 + 5.0*exp(-2.23606797749979*dif/kernel_length)*(1 + 2.23606797749979*dif*(1 + 0.74535599249993*dif/kernel_length)/kernel_length)/kernel_length^2 - 4.47213595499958*exp(-2.23606797749979*dif/kernel_length)*(0.0 + 1.66666666666667*dif/kernel_length^2 + 2.23606797749979*(1 + 0.74535599249993*dif/kernel_length)/kernel_length)/kernel_length
    end

    if dorder==[1, 0]
        func = exp(-2.23606797749979*dif/kernel_length)*(0.0 + 1.66666666666667*dif/kernel_length^2 + 2.23606797749979*(1 + 0.74535599249993*dif/kernel_length)/kernel_length) - 2.23606797749979*exp(-2.23606797749979*dif/kernel_length)*(1 + 2.23606797749979*dif*(1 + 0.74535599249993*dif/kernel_length)/kernel_length)/kernel_length
    end

    if dorder==[0, 0]
        func = exp(-2.23606797749979*dif/kernel_length)*(1 + 2.23606797749979*dif*(1 + 0.74535599249993*dif/kernel_length)/kernel_length)
    end

    return (2 * (dif_positive | iseven(dorder[1])) - 1) * even_time_derivative * float(func)  # correcting for use of abs_dif and amount of t2 derivatives

end


return matern52_kernel, 1  # the function handle and the number of kernel hyperparameters
