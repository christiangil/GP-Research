

"""
RBF_kernel function created by kernel_coder(). Requires 1 hyperparameters. Likely created using RBF_kernel_base() as an input.
Use with include("kernels/RBF_kernel.jl").
"""
function RBF_kernel(hyperparameters::Union{Array{Float64,1},Array{Any,1}}, dif::Float64; dorder::Union{Array{Int,1},Array{Float64,1}}=zeros(1))

    @assert length(hyperparameters)==1 "hyperparameters is the wrong length"
    if dorder==zeros(1)
        dorder = zeros(length(hyperparameters) + 2)
    else
        @assert length(dorder)==(length(hyperparameters) + 2) "dorder is the wrong length"
    end
    dorder = convert(Array{Int64,1}, dorder)

    kernel_length = hyperparameters[3-2]

    if dorder==[2, 2, 1]
        func = 0.0 - 12.0*exp((-1/2)*dif^2/kernel_length^2)/kernel_length^5 + 39.0*exp((-1/2)*dif^2/kernel_length^2)*dif^2/kernel_length^7 - 14.0*exp((-1/2)*dif^2/kernel_length^2)*dif^4/kernel_length^9 + 1.0*exp((-1/2)*dif^2/kernel_length^2)*dif^6/kernel_length^11
    end

    if dorder==[1, 2, 1]
        func = 0.0 - 12.0*exp((-1/2)*dif^2/kernel_length^2)*dif/kernel_length^5 + 9.0*exp((-1/2)*dif^2/kernel_length^2)*dif^3/kernel_length^7 - 1.0*exp((-1/2)*dif^2/kernel_length^2)*dif^5/kernel_length^9
    end

    if dorder==[0, 2, 1]
        func = 0.0 + 2.0*exp((-1/2)*dif^2/kernel_length^2)/kernel_length^3 - 5.0*exp((-1/2)*dif^2/kernel_length^2)*dif^2/kernel_length^5 + 1.0*exp((-1/2)*dif^2/kernel_length^2)*dif^4/kernel_length^7
    end

    if dorder==[2, 1, 1]
        func = 0.0 + 12.0*exp((-1/2)*dif^2/kernel_length^2)*dif/kernel_length^5 - 9.0*exp((-1/2)*dif^2/kernel_length^2)*dif^3/kernel_length^7 + 1.0*exp((-1/2)*dif^2/kernel_length^2)*dif^5/kernel_length^9
    end

    if dorder==[1, 1, 1]
        func = 0.0 - 2.0*exp((-1/2)*dif^2/kernel_length^2)/kernel_length^3 + 5.0*exp((-1/2)*dif^2/kernel_length^2)*dif^2/kernel_length^5 - 1.0*exp((-1/2)*dif^2/kernel_length^2)*dif^4/kernel_length^7
    end

    if dorder==[0, 1, 1]
        func = -2.0*exp((-1/2)*dif^2/kernel_length^2)*dif/kernel_length^3 + 1.0*exp((-1/2)*dif^2/kernel_length^2)*dif^3/kernel_length^5
    end

    if dorder==[2, 0, 1]
        func = 0.0 + 2.0*exp((-1/2)*dif^2/kernel_length^2)/kernel_length^3 - 5.0*exp((-1/2)*dif^2/kernel_length^2)*dif^2/kernel_length^5 + 1.0*exp((-1/2)*dif^2/kernel_length^2)*dif^4/kernel_length^7
    end

    if dorder==[1, 0, 1]
        func = 2.0*exp((-1/2)*dif^2/kernel_length^2)*dif/kernel_length^3 - 1.0*exp((-1/2)*dif^2/kernel_length^2)*dif^3/kernel_length^5
    end

    if dorder==[0, 0, 1]
        func = 1.0*exp((-1/2)*dif^2/kernel_length^2)*dif^2/kernel_length^3
    end

    if dorder==[2, 2, 0]
        func = 0.0 + 3.0*exp((-1/2)*dif^2/kernel_length^2)/kernel_length^4 - 6.0*exp((-1/2)*dif^2/kernel_length^2)*dif^2/kernel_length^6 + 1.0*exp((-1/2)*dif^2/kernel_length^2)*dif^4/kernel_length^8
    end

    if dorder==[1, 2, 0]
        func = 0.0 + 3.0*exp((-1/2)*dif^2/kernel_length^2)*dif/kernel_length^4 - 1.0*exp((-1/2)*dif^2/kernel_length^2)*dif^3/kernel_length^6
    end

    if dorder==[0, 2, 0]
        func = -1.0*exp((-1/2)*dif^2/kernel_length^2)/kernel_length^2 + 1.0*exp((-1/2)*dif^2/kernel_length^2)*dif^2/kernel_length^4
    end

    if dorder==[2, 1, 0]
        func = 0.0 - 3.0*exp((-1/2)*dif^2/kernel_length^2)*dif/kernel_length^4 + 1.0*exp((-1/2)*dif^2/kernel_length^2)*dif^3/kernel_length^6
    end

    if dorder==[1, 1, 0]
        func = 1.0*exp((-1/2)*dif^2/kernel_length^2)/kernel_length^2 - 1.0*exp((-1/2)*dif^2/kernel_length^2)*dif^2/kernel_length^4
    end

    if dorder==[0, 1, 0]
        func = 1.0*exp((-1/2)*dif^2/kernel_length^2)*dif/kernel_length^2
    end

    if dorder==[2, 0, 0]
        func = -1.0*exp((-1/2)*dif^2/kernel_length^2)/kernel_length^2 + 1.0*exp((-1/2)*dif^2/kernel_length^2)*dif^2/kernel_length^4
    end

    if dorder==[1, 0, 0]
        func = -1.0*exp((-1/2)*dif^2/kernel_length^2)*dif/kernel_length^2
    end

    if dorder==[0, 0, 0]
        func = 1.0*exp((-1/2)*dif^2/kernel_length^2)
    end

    return float(func)

end


return 1  # the number of kernel hyperparameters