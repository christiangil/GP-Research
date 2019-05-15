

"""
matern52_kernel function created by kernel_coder(). Requires 1 hyperparameters. Likely created using matern52_kernel_base() as an input. 
Use with include("src/kernels/matern52_kernel.jl").
hyperparameters == ["kernel_length"]
"""
function matern52_kernel(
    hyperparameters::AbstractArray{T1,1}, 
    dif::Real; 
    dorder::AbstractArray{T2,1}=zeros(Int64, length(hyperparameters) + 2) 
    ) where {T1<:Real, T2<:Integer}

    @assert length(hyperparameters)==1 "hyperparameters is the wrong length"
    @assert length(dorder)==(length(hyperparameters) + 2) "dorder is the wrong length"
    even_time_derivative = 2 * iseven(dorder[2]) - 1
    @assert maximum(dorder) < 3 "No more than two time derivatives for either t1 or t2 can be calculated"

    dorder = append!([sum(dorder[1:2])], dorder[3:end])

    kernel_length = hyperparameters[1]

    dif_positive = (dif >= 0)  # store original sign of dif (for correcting odd kernel derivatives)
    dif = abs(dif)  # this is corected for later

    if dorder==[4, 2]
        func = 2000.0*exp(-2.23606797749979*dif/kernel_length)/kernel_length^6 - 2236.06797749979*exp(-2.23606797749979*dif/kernel_length)*dif/kernel_length^7 + 500.0*exp(-2.23606797749979*dif/kernel_length)*dif^2/kernel_length^8 + 500.0*exp(-2.23606797749979*dif/kernel_length)*(1 + 2.23606797749979*dif*(1 + 0.74535599249993*dif/kernel_length)/kernel_length)/kernel_length^6 - 44.7213595499958*exp(-2.23606797749979*dif/kernel_length)*(0.0 + 16.6666666666667*dif/kernel_length^4 + 4.47213595499958*(1 + 0.74535599249993*dif/kernel_length)/kernel_length^3)/kernel_length^3 + 268.328157299975*exp(-2.23606797749979*dif/kernel_length)*(0.0 - 5.0*dif/kernel_length^3 - 2.23606797749979*(1 + 0.74535599249993*dif/kernel_length)/kernel_length^2)/kernel_length^4 - 536.65631459995*exp(-2.23606797749979*dif/kernel_length)*(0.0 + 1.66666666666667*dif/kernel_length^2 + 2.23606797749979*(1 + 0.74535599249993*dif/kernel_length)/kernel_length)/kernel_length^5 + 25.0*exp(-2.23606797749979*dif/kernel_length)*(0.0 + 6.66666666666667*dif^2/kernel_length^4 + 4.47213595499958*dif*(1 + 0.74535599249993*dif/kernel_length)/kernel_length^3)/kernel_length^4 - 200.0*exp(-2.23606797749979*dif/kernel_length)*(0.0 - 1.66666666666667*dif^2/kernel_length^3 - 2.23606797749979*dif*(1 + 0.74535599249993*dif/kernel_length)/kernel_length^2)/kernel_length^5 - 559.016994374948*exp(-2.23606797749979*dif/kernel_length)*dif*(1 + 2.23606797749979*dif*(1 + 0.74535599249993*dif/kernel_length)/kernel_length)/kernel_length^7 - 200.0*exp(-2.23606797749979*dif/kernel_length)*dif*(0.0 - 5.0*dif/kernel_length^3 - 2.23606797749979*(1 + 0.74535599249993*dif/kernel_length)/kernel_length^2)/kernel_length^5 + 800.0*exp(-2.23606797749979*dif/kernel_length)*dif*(0.0 + 1.66666666666667*dif/kernel_length^2 + 2.23606797749979*(1 + 0.74535599249993*dif/kernel_length)/kernel_length)/kernel_length^6 + 111.80339887499*exp(-2.23606797749979*dif/kernel_length)*dif*(0.0 - 1.66666666666667*dif^2/kernel_length^3 - 2.23606797749979*dif*(1 + 0.74535599249993*dif/kernel_length)/kernel_length^2)/kernel_length^6 + 125.0*exp(-2.23606797749979*dif/kernel_length)*dif^2*(1 + 2.23606797749979*dif*(1 + 0.74535599249993*dif/kernel_length)/kernel_length)/kernel_length^8 - 223.606797749979*exp(-2.23606797749979*dif/kernel_length)*dif^2*(0.0 + 1.66666666666667*dif/kernel_length^2 + 2.23606797749979*(1 + 0.74535599249993*dif/kernel_length)/kernel_length)/kernel_length^7
    end

    if dorder==[3, 2]
        func = -268.328157299975*exp(-2.23606797749979*dif/kernel_length)/kernel_length^5 + 400.0*exp(-2.23606797749979*dif/kernel_length)*dif/kernel_length^6 - 111.80339887499*exp(-2.23606797749979*dif/kernel_length)*dif^2/kernel_length^7 - 134.164078649987*exp(-2.23606797749979*dif/kernel_length)*(1 + 2.23606797749979*dif*(1 + 0.74535599249993*dif/kernel_length)/kernel_length)/kernel_length^5 + 15.0*exp(-2.23606797749979*dif/kernel_length)*(0.0 + 16.6666666666667*dif/kernel_length^4 + 4.47213595499958*(1 + 0.74535599249993*dif/kernel_length)/kernel_length^3)/kernel_length^2 - 60.0*exp(-2.23606797749979*dif/kernel_length)*(0.0 - 5.0*dif/kernel_length^3 - 2.23606797749979*(1 + 0.74535599249993*dif/kernel_length)/kernel_length^2)/kernel_length^3 + 90.0*exp(-2.23606797749979*dif/kernel_length)*(0.0 + 1.66666666666667*dif/kernel_length^2 + 2.23606797749979*(1 + 0.74535599249993*dif/kernel_length)/kernel_length)/kernel_length^4 - 11.180339887499*exp(-2.23606797749979*dif/kernel_length)*(0.0 + 6.66666666666667*dif^2/kernel_length^4 + 4.47213595499958*dif*(1 + 0.74535599249993*dif/kernel_length)/kernel_length^3)/kernel_length^3 + 67.0820393249937*exp(-2.23606797749979*dif/kernel_length)*(0.0 - 1.66666666666667*dif^2/kernel_length^3 - 2.23606797749979*dif*(1 + 0.74535599249993*dif/kernel_length)/kernel_length^2)/kernel_length^4 + 200.0*exp(-2.23606797749979*dif/kernel_length)*dif*(1 + 2.23606797749979*dif*(1 + 0.74535599249993*dif/kernel_length)/kernel_length)/kernel_length^6 + 67.0820393249937*exp(-2.23606797749979*dif/kernel_length)*dif*(0.0 - 5.0*dif/kernel_length^3 - 2.23606797749979*(1 + 0.74535599249993*dif/kernel_length)/kernel_length^2)/kernel_length^4 - 201.246117974981*exp(-2.23606797749979*dif/kernel_length)*dif*(0.0 + 1.66666666666667*dif/kernel_length^2 + 2.23606797749979*(1 + 0.74535599249993*dif/kernel_length)/kernel_length)/kernel_length^5 - 50.0*exp(-2.23606797749979*dif/kernel_length)*dif*(0.0 - 1.66666666666667*dif^2/kernel_length^3 - 2.23606797749979*dif*(1 + 0.74535599249993*dif/kernel_length)/kernel_length^2)/kernel_length^5 - 55.9016994374948*exp(-2.23606797749979*dif/kernel_length)*dif^2*(1 + 2.23606797749979*dif*(1 + 0.74535599249993*dif/kernel_length)/kernel_length)/kernel_length^7 + 75.0*exp(-2.23606797749979*dif/kernel_length)*dif^2*(0.0 + 1.66666666666667*dif/kernel_length^2 + 2.23606797749979*(1 + 0.74535599249993*dif/kernel_length)/kernel_length)/kernel_length^6
    end

    if dorder==[2, 2]
        func = 20.0*exp(-2.23606797749979*dif/kernel_length)/kernel_length^4 - 44.7213595499958*exp(-2.23606797749979*dif/kernel_length)*dif/kernel_length^5 + 16.6666666666667*exp(-2.23606797749979*dif/kernel_length)*dif^2/kernel_length^6 + 30.0*exp(-2.23606797749979*dif/kernel_length)*(1 + 2.23606797749979*dif*(1 + 0.74535599249993*dif/kernel_length)/kernel_length)/kernel_length^4 - 4.47213595499958*exp(-2.23606797749979*dif/kernel_length)*(0.0 + 16.6666666666667*dif/kernel_length^4 + 4.47213595499958*(1 + 0.74535599249993*dif/kernel_length)/kernel_length^3)/kernel_length + 8.94427190999916*exp(-2.23606797749979*dif/kernel_length)*(0.0 - 5.0*dif/kernel_length^3 - 2.23606797749979*(1 + 0.74535599249993*dif/kernel_length)/kernel_length^2)/kernel_length^2 - 8.94427190999916*exp(-2.23606797749979*dif/kernel_length)*(0.0 + 1.66666666666667*dif/kernel_length^2 + 2.23606797749979*(1 + 0.74535599249993*dif/kernel_length)/kernel_length)/kernel_length^3 + 5.0*exp(-2.23606797749979*dif/kernel_length)*(0.0 + 6.66666666666667*dif^2/kernel_length^4 + 4.47213595499958*dif*(1 + 0.74535599249993*dif/kernel_length)/kernel_length^3)/kernel_length^2 - 20.0*exp(-2.23606797749979*dif/kernel_length)*(0.0 - 1.66666666666667*dif^2/kernel_length^3 - 2.23606797749979*dif*(1 + 0.74535599249993*dif/kernel_length)/kernel_length^2)/kernel_length^3 - 67.0820393249937*exp(-2.23606797749979*dif/kernel_length)*dif*(1 + 2.23606797749979*dif*(1 + 0.74535599249993*dif/kernel_length)/kernel_length)/kernel_length^5 - 20.0*exp(-2.23606797749979*dif/kernel_length)*dif*(0.0 - 5.0*dif/kernel_length^3 - 2.23606797749979*(1 + 0.74535599249993*dif/kernel_length)/kernel_length^2)/kernel_length^3 + 40.0*exp(-2.23606797749979*dif/kernel_length)*dif*(0.0 + 1.66666666666667*dif/kernel_length^2 + 2.23606797749979*(1 + 0.74535599249993*dif/kernel_length)/kernel_length)/kernel_length^4 + 22.3606797749979*exp(-2.23606797749979*dif/kernel_length)*dif*(0.0 - 1.66666666666667*dif^2/kernel_length^3 - 2.23606797749979*dif*(1 + 0.74535599249993*dif/kernel_length)/kernel_length^2)/kernel_length^4 + 25.0*exp(-2.23606797749979*dif/kernel_length)*dif^2*(1 + 2.23606797749979*dif*(1 + 0.74535599249993*dif/kernel_length)/kernel_length)/kernel_length^6 - 22.3606797749979*exp(-2.23606797749979*dif/kernel_length)*dif^2*(0.0 + 1.66666666666667*dif/kernel_length^2 + 2.23606797749979*(1 + 0.74535599249993*dif/kernel_length)/kernel_length)/kernel_length^5
    end

    if dorder==[1, 2]
        func = exp(-2.23606797749979*dif/kernel_length)*(0.0 + 16.6666666666667*dif/kernel_length^4 + 4.47213595499958*(1 + 0.74535599249993*dif/kernel_length)/kernel_length^3) - 4.47213595499958*exp(-2.23606797749979*dif/kernel_length)*(1 + 2.23606797749979*dif*(1 + 0.74535599249993*dif/kernel_length)/kernel_length)/kernel_length^3 - 2.23606797749979*exp(-2.23606797749979*dif/kernel_length)*(0.0 + 6.66666666666667*dif^2/kernel_length^4 + 4.47213595499958*dif*(1 + 0.74535599249993*dif/kernel_length)/kernel_length^3)/kernel_length + 4.47213595499958*exp(-2.23606797749979*dif/kernel_length)*(0.0 - 1.66666666666667*dif^2/kernel_length^3 - 2.23606797749979*dif*(1 + 0.74535599249993*dif/kernel_length)/kernel_length^2)/kernel_length^2 + 20.0*exp(-2.23606797749979*dif/kernel_length)*dif*(1 + 2.23606797749979*dif*(1 + 0.74535599249993*dif/kernel_length)/kernel_length)/kernel_length^4 + 4.47213595499958*exp(-2.23606797749979*dif/kernel_length)*dif*(0.0 - 5.0*dif/kernel_length^3 - 2.23606797749979*(1 + 0.74535599249993*dif/kernel_length)/kernel_length^2)/kernel_length^2 - 4.47213595499958*exp(-2.23606797749979*dif/kernel_length)*dif*(0.0 + 1.66666666666667*dif/kernel_length^2 + 2.23606797749979*(1 + 0.74535599249993*dif/kernel_length)/kernel_length)/kernel_length^3 - 10.0*exp(-2.23606797749979*dif/kernel_length)*dif*(0.0 - 1.66666666666667*dif^2/kernel_length^3 - 2.23606797749979*dif*(1 + 0.74535599249993*dif/kernel_length)/kernel_length^2)/kernel_length^3 - 11.180339887499*exp(-2.23606797749979*dif/kernel_length)*dif^2*(1 + 2.23606797749979*dif*(1 + 0.74535599249993*dif/kernel_length)/kernel_length)/kernel_length^5 + 5.0*exp(-2.23606797749979*dif/kernel_length)*dif^2*(0.0 + 1.66666666666667*dif/kernel_length^2 + 2.23606797749979*(1 + 0.74535599249993*dif/kernel_length)/kernel_length)/kernel_length^4
    end

    if dorder==[0, 2]
        func = exp(-2.23606797749979*dif/kernel_length)*(0.0 + 6.66666666666667*dif^2/kernel_length^4 + 4.47213595499958*dif*(1 + 0.74535599249993*dif/kernel_length)/kernel_length^3) - 4.47213595499958*exp(-2.23606797749979*dif/kernel_length)*dif*(1 + 2.23606797749979*dif*(1 + 0.74535599249993*dif/kernel_length)/kernel_length)/kernel_length^3 + 4.47213595499958*exp(-2.23606797749979*dif/kernel_length)*dif*(0.0 - 1.66666666666667*dif^2/kernel_length^3 - 2.23606797749979*dif*(1 + 0.74535599249993*dif/kernel_length)/kernel_length^2)/kernel_length^2 + 5.0*exp(-2.23606797749979*dif/kernel_length)*dif^2*(1 + 2.23606797749979*dif*(1 + 0.74535599249993*dif/kernel_length)/kernel_length)/kernel_length^4
    end

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

    return (2 * (dif_positive || iseven(dorder[1])) - 1) * even_time_derivative * float(func)  # correcting for use of abs_dif and amount of t2 derivatives

end


return matern52_kernel, 1  # the function handle and the number of kernel hyperparameters
