

"""
rbf_kernel function created by kernel_coder(). Requires 1 hyperparameters. Likely created using rbf_kernel_base() as an input. 
Use with include("src/kernels/rbf_kernel.jl").
hyperparameters == ["kernel_length"]
"""
function rbf_kernel(hyperparameters::AbstractArray{T1,1}, dif::Real; dorder::AbstractArray{T2,1}=zeros(length(hyperparameters) + 2)) where {T1<:Real, T2<:Real}

    @assert length(hyperparameters)==1 "hyperparameters is the wrong length"
    @assert length(dorder)==(length(hyperparameters) + 2) "dorder is the wrong length"
    dorder = convert(Array{Int64,1}, dorder)
    even_time_derivative = 2 * iseven(dorder[2]) - 1
    @assert maximum(dorder) < 3 "No more than two time derivatives for either t1 or t2 can be calculated"

    dorder = append!([sum(dorder[1:2])], dorder[3:end])

    kernel_length = hyperparameters[1]

    if dorder==[4, 1]
        func = -12*exp((-1/2)*dif^2/kernel_length^2)/kernel_length^5 + 39*exp((-1/2)*dif^2/kernel_length^2)*dif^2/kernel_length^7 - 14*exp((-1/2)*dif^2/kernel_length^2)*dif^4/kernel_length^9 + exp((-1/2)*dif^2/kernel_length^2)*dif^6/kernel_length^11
    end

    if dorder==[3, 1]
        func = -12*exp((-1/2)*dif^2/kernel_length^2)*dif/kernel_length^5 + 9*exp((-1/2)*dif^2/kernel_length^2)*dif^3/kernel_length^7 - exp((-1/2)*dif^2/kernel_length^2)*dif^5/kernel_length^9
    end

    if dorder==[2, 1]
        func = 2*exp((-1/2)*dif^2/kernel_length^2)/kernel_length^3 - 5*exp((-1/2)*dif^2/kernel_length^2)*dif^2/kernel_length^5 + exp((-1/2)*dif^2/kernel_length^2)*dif^4/kernel_length^7
    end

    if dorder==[1, 1]
        func = 2*exp((-1/2)*dif^2/kernel_length^2)*dif/kernel_length^3 - exp((-1/2)*dif^2/kernel_length^2)*dif^3/kernel_length^5
    end

    if dorder==[0, 1]
        func = exp((-1/2)*dif^2/kernel_length^2)*dif^2/kernel_length^3
    end

    if dorder==[4, 0]
        func = 3*exp((-1/2)*dif^2/kernel_length^2)/kernel_length^4 - 6*exp((-1/2)*dif^2/kernel_length^2)*dif^2/kernel_length^6 + exp((-1/2)*dif^2/kernel_length^2)*dif^4/kernel_length^8
    end

    if dorder==[3, 0]
        func = 3*exp((-1/2)*dif^2/kernel_length^2)*dif/kernel_length^4 - exp((-1/2)*dif^2/kernel_length^2)*dif^3/kernel_length^6
    end

    if dorder==[2, 0]
        func = -exp((-1/2)*dif^2/kernel_length^2)/kernel_length^2 + exp((-1/2)*dif^2/kernel_length^2)*dif^2/kernel_length^4
    end

    if dorder==[1, 0]
        func = -exp((-1/2)*dif^2/kernel_length^2)*dif/kernel_length^2
    end

    if dorder==[0, 0]
        func = exp((-1/2)*dif^2/kernel_length^2)
    end

    return even_time_derivative * float(func)  # correcting for amount of t2 derivatives

end


return rbf_kernel, 1  # the function handle and the number of kernel hyperparameters
