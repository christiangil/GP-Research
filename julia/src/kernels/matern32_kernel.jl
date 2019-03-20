

"""
matern32_kernel function created by kernel_coder(). Requires 1 hyperparameters. Likely created using matern32_kernel_base() as an input. 
Use with include("kernels/matern32_kernel.jl").
"""
function matern32_kernel(hyperparameters::Union{Array{T1,1},Array{Any,1}}, dif::Real; dorder::Array{T2,1}=zeros(1)) where {T1<:Real, T2<:Real}

    @assert length(hyperparameters)==1 "hyperparameters is the wrong length"
    if dorder==zeros(1)
        dorder = zeros(length(hyperparameters) + 2)
    else
        @assert length(dorder)==(length(hyperparameters) + 2) "dorder is the wrong length"
    end
    dorder = convert(Array{Int64,1}, dorder)

    kernel_length = hyperparameters[3-2]

    if dorder==[2, 2, 1]
        func = 144.0*exp(-1.73205080756888*dif/kernel_length)/kernel_length^5 - 77.9422863405995*exp(-1.73205080756888*dif/kernel_length)*dif/kernel_length^6 - 36.0*exp(-1.73205080756888*dif/kernel_length)*(1 + 1.73205080756888*dif/kernel_length)/kernel_length^5 + 15.5884572681199*exp(-1.73205080756888*dif/kernel_length)*dif*(1 + 1.73205080756888*dif/kernel_length)/kernel_length^6
    end

    if dorder==[1, 2, 1]
        func = -46.7653718043597*exp(-1.73205080756888*dif/kernel_length)/kernel_length^4 + 36.0*exp(-1.73205080756888*dif/kernel_length)*dif/kernel_length^5 + 15.5884572681199*exp(-1.73205080756888*dif/kernel_length)*(1 + 1.73205080756888*dif/kernel_length)/kernel_length^4 - 9.0*exp(-1.73205080756888*dif/kernel_length)*dif*(1 + 1.73205080756888*dif/kernel_length)/kernel_length^5
    end

    if dorder==[0, 2, 1]
        func = 12.0*exp(-1.73205080756888*dif/kernel_length)/kernel_length^3 - 15.5884572681199*exp(-1.73205080756888*dif/kernel_length)*dif/kernel_length^4 - 6.0*exp(-1.73205080756888*dif/kernel_length)*(1 + 1.73205080756888*dif/kernel_length)/kernel_length^3 + 5.19615242270663*exp(-1.73205080756888*dif/kernel_length)*dif*(1 + 1.73205080756888*dif/kernel_length)/kernel_length^4
    end

    if dorder==[2, 1, 1]
        func = 46.7653718043597*exp(-1.73205080756888*dif/kernel_length)/kernel_length^4 - 36.0*exp(-1.73205080756888*dif/kernel_length)*dif/kernel_length^5 - 15.5884572681199*exp(-1.73205080756888*dif/kernel_length)*(1 + 1.73205080756888*dif/kernel_length)/kernel_length^4 + 9.0*exp(-1.73205080756888*dif/kernel_length)*dif*(1 + 1.73205080756888*dif/kernel_length)/kernel_length^5
    end

    if dorder==[1, 1, 1]
        func = -12.0*exp(-1.73205080756888*dif/kernel_length)/kernel_length^3 + 15.5884572681199*exp(-1.73205080756888*dif/kernel_length)*dif/kernel_length^4 + 6.0*exp(-1.73205080756888*dif/kernel_length)*(1 + 1.73205080756888*dif/kernel_length)/kernel_length^3 - 5.19615242270663*exp(-1.73205080756888*dif/kernel_length)*dif*(1 + 1.73205080756888*dif/kernel_length)/kernel_length^4
    end

    if dorder==[0, 1, 1]
        func = 1.73205080756888*exp(-1.73205080756888*dif/kernel_length)/kernel_length^2 - 6.0*exp(-1.73205080756888*dif/kernel_length)*dif/kernel_length^3 - 1.73205080756888*exp(-1.73205080756888*dif/kernel_length)*(1 + 1.73205080756888*dif/kernel_length)/kernel_length^2 + 3.0*exp(-1.73205080756888*dif/kernel_length)*dif*(1 + 1.73205080756888*dif/kernel_length)/kernel_length^3
    end

    if dorder==[2, 0, 1]
        func = 12.0*exp(-1.73205080756888*dif/kernel_length)/kernel_length^3 - 15.5884572681199*exp(-1.73205080756888*dif/kernel_length)*dif/kernel_length^4 - 6.0*exp(-1.73205080756888*dif/kernel_length)*(1 + 1.73205080756888*dif/kernel_length)/kernel_length^3 + 5.19615242270663*exp(-1.73205080756888*dif/kernel_length)*dif*(1 + 1.73205080756888*dif/kernel_length)/kernel_length^4
    end

    if dorder==[1, 0, 1]
        func = -1.73205080756888*exp(-1.73205080756888*dif/kernel_length)/kernel_length^2 + 6.0*exp(-1.73205080756888*dif/kernel_length)*dif/kernel_length^3 + 1.73205080756888*exp(-1.73205080756888*dif/kernel_length)*(1 + 1.73205080756888*dif/kernel_length)/kernel_length^2 - 3.0*exp(-1.73205080756888*dif/kernel_length)*dif*(1 + 1.73205080756888*dif/kernel_length)/kernel_length^3
    end

    if dorder==[0, 0, 1]
        func = -1.73205080756888*exp(-1.73205080756888*dif/kernel_length)*dif/kernel_length^2 + 1.73205080756888*exp(-1.73205080756888*dif/kernel_length)*dif*(1 + 1.73205080756888*dif/kernel_length)/kernel_length^2
    end

    if dorder==[2, 2, 0]
        func = -36.0*exp(-1.73205080756888*dif/kernel_length)/kernel_length^4 + 9.0*exp(-1.73205080756888*dif/kernel_length)*(1 + 1.73205080756888*dif/kernel_length)/kernel_length^4
    end

    if dorder==[1, 2, 0]
        func = 15.5884572681199*exp(-1.73205080756888*dif/kernel_length)/kernel_length^3 - 5.19615242270663*exp(-1.73205080756888*dif/kernel_length)*(1 + 1.73205080756888*dif/kernel_length)/kernel_length^3
    end

    if dorder==[0, 2, 0]
        func = -6.0*exp(-1.73205080756888*dif/kernel_length)/kernel_length^2 + 3.0*exp(-1.73205080756888*dif/kernel_length)*(1 + 1.73205080756888*dif/kernel_length)/kernel_length^2
    end

    if dorder==[2, 1, 0]
        func = -15.5884572681199*exp(-1.73205080756888*dif/kernel_length)/kernel_length^3 + 5.19615242270663*exp(-1.73205080756888*dif/kernel_length)*(1 + 1.73205080756888*dif/kernel_length)/kernel_length^3
    end

    if dorder==[1, 1, 0]
        func = 6.0*exp(-1.73205080756888*dif/kernel_length)/kernel_length^2 - 3.0*exp(-1.73205080756888*dif/kernel_length)*(1 + 1.73205080756888*dif/kernel_length)/kernel_length^2
    end

    if dorder==[0, 1, 0]
        func = -1.73205080756888*exp(-1.73205080756888*dif/kernel_length)/kernel_length + 1.73205080756888*exp(-1.73205080756888*dif/kernel_length)*(1 + 1.73205080756888*dif/kernel_length)/kernel_length
    end

    if dorder==[2, 0, 0]
        func = -6.0*exp(-1.73205080756888*dif/kernel_length)/kernel_length^2 + 3.0*exp(-1.73205080756888*dif/kernel_length)*(1 + 1.73205080756888*dif/kernel_length)/kernel_length^2
    end

    if dorder==[1, 0, 0]
        func = 1.73205080756888*exp(-1.73205080756888*dif/kernel_length)/kernel_length - 1.73205080756888*exp(-1.73205080756888*dif/kernel_length)*(1 + 1.73205080756888*dif/kernel_length)/kernel_length
    end

    if dorder==[0, 0, 0]
        func = exp(-1.73205080756888*dif/kernel_length)*(1 + 1.73205080756888*dif/kernel_length)
    end

    return float(func)

end


return matern32_kernel, 1  # the function handle and the number of kernel hyperparameters
