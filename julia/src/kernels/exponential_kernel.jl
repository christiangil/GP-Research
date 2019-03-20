

"""
exponential_kernel function created by kernel_coder(). Requires 1 hyperparameters. Likely created using exponential_kernel_base() as an input. 
Use with include("kernels/exponential_kernel.jl").
"""
function exponential_kernel(hyperparameters::Union{Array{T1,1},Array{Any,1}}, dif::Real; dorder::Array{T2,1}=zeros(1)) where {T1<:Real, T2<:Real}

    @assert length(hyperparameters)==1 "hyperparameters is the wrong length"
    if dorder==zeros(1)
        dorder = zeros(length(hyperparameters) + 2)
    else
        @assert length(dorder)==(length(hyperparameters) + 2) "dorder is the wrong length"
    end
    dorder = convert(Array{Int64,1}, dorder)

    OU_kernel_length = hyperparameters[3-2]

    if dorder==[2, 2, 1]
        func = -4.0*exp(-abs(dif)/OU_kernel_length)/OU_kernel_length^5 - 3.0*exp(-abs(dif)/OU_kernel_length)/((dif^2)^(3/2)*OU_kernel_length^2) - 3.0*exp(-abs(dif)/OU_kernel_length)/(abs(dif)*OU_kernel_length^4) + 18.0*exp(-abs(dif)/OU_kernel_length)*dif^2/((dif^2)^(5/2)*OU_kernel_length^2) + 12.0*exp(-abs(dif)/OU_kernel_length)*dif^2/((dif^2)^(3/2)*OU_kernel_length^4) - 15.0*exp(-abs(dif)/OU_kernel_length)*dif^4/((dif^2)^(7/2)*OU_kernel_length^2) - 9.0*exp(-abs(dif)/OU_kernel_length)*dif^4/((dif^2)^(5/2)*OU_kernel_length^4) + exp(-abs(dif)/OU_kernel_length)*dif^4/((dif^2)^(3/2)*OU_kernel_length^6)
    end

    if dorder==[1, 2, 1]
        func = -exp(-abs(dif)/OU_kernel_length)*dif/OU_kernel_length^5 - 2.0*exp(-abs(dif)/OU_kernel_length)*abs(dif)/(dif*OU_kernel_length^4) - 3.0*exp(-abs(dif)/OU_kernel_length)*dif/((dif^2)^(3/2)*OU_kernel_length^2) + 3.0*exp(-abs(dif)/OU_kernel_length)*dif/(abs(dif)*OU_kernel_length^4) + 3.0*exp(-abs(dif)/OU_kernel_length)*dif^3/((dif^2)^(5/2)*OU_kernel_length^2) + 2.0*exp(-abs(dif)/OU_kernel_length)*dif^3/((dif^2)^(3/2)*OU_kernel_length^4)
    end

    if dorder==[0, 2, 1]
        func = -2.0*exp(-abs(dif)/OU_kernel_length)/OU_kernel_length^3 + exp(-abs(dif)/OU_kernel_length)/(abs(dif)*OU_kernel_length^2) - exp(-abs(dif)/OU_kernel_length)*dif^2/((dif^2)^(3/2)*OU_kernel_length^2) + exp(-abs(dif)/OU_kernel_length)*dif^2/(abs(dif)*OU_kernel_length^4)
    end

    if dorder==[2, 1, 1]
        func = exp(-abs(dif)/OU_kernel_length)*dif/OU_kernel_length^5 + 2.0*exp(-abs(dif)/OU_kernel_length)*abs(dif)/(dif*OU_kernel_length^4) + 3.0*exp(-abs(dif)/OU_kernel_length)*dif/((dif^2)^(3/2)*OU_kernel_length^2) - 3.0*exp(-abs(dif)/OU_kernel_length)*dif/(abs(dif)*OU_kernel_length^4) - 3.0*exp(-abs(dif)/OU_kernel_length)*dif^3/((dif^2)^(5/2)*OU_kernel_length^2) - 2.0*exp(-abs(dif)/OU_kernel_length)*dif^3/((dif^2)^(3/2)*OU_kernel_length^4)
    end

    if dorder==[1, 1, 1]
        func = 2.0*exp(-abs(dif)/OU_kernel_length)/OU_kernel_length^3 - exp(-abs(dif)/OU_kernel_length)/(abs(dif)*OU_kernel_length^2) + exp(-abs(dif)/OU_kernel_length)*dif^2/((dif^2)^(3/2)*OU_kernel_length^2) - exp(-abs(dif)/OU_kernel_length)*dif^2/(abs(dif)*OU_kernel_length^4)
    end

    if dorder==[0, 1, 1]
        func = exp(-abs(dif)/OU_kernel_length)*dif/OU_kernel_length^3 - exp(-abs(dif)/OU_kernel_length)*dif/(abs(dif)*OU_kernel_length^2)
    end

    if dorder==[2, 0, 1]
        func = -2.0*exp(-abs(dif)/OU_kernel_length)/OU_kernel_length^3 + exp(-abs(dif)/OU_kernel_length)/(abs(dif)*OU_kernel_length^2) - exp(-abs(dif)/OU_kernel_length)*dif^2/((dif^2)^(3/2)*OU_kernel_length^2) + exp(-abs(dif)/OU_kernel_length)*dif^2/(abs(dif)*OU_kernel_length^4)
    end

    if dorder==[1, 0, 1]
        func = -exp(-abs(dif)/OU_kernel_length)*dif/OU_kernel_length^3 + exp(-abs(dif)/OU_kernel_length)*dif/(abs(dif)*OU_kernel_length^2)
    end

    if dorder==[0, 0, 1]
        func = exp(-abs(dif)/OU_kernel_length)*abs(dif)/OU_kernel_length^2
    end

    if dorder==[2, 2, 0]
        func = exp(-abs(dif)/OU_kernel_length)/OU_kernel_length^4 + 3.0*exp(-abs(dif)/OU_kernel_length)/((dif^2)^(3/2)*OU_kernel_length) + 2.0*exp(-abs(dif)/OU_kernel_length)/(abs(dif)*OU_kernel_length^3) - 18.0*exp(-abs(dif)/OU_kernel_length)*dif^2/((dif^2)^(5/2)*OU_kernel_length) - 6.0*exp(-abs(dif)/OU_kernel_length)*dif^2/((dif^2)^(3/2)*OU_kernel_length^3) + 15.0*exp(-abs(dif)/OU_kernel_length)*dif^4/((dif^2)^(7/2)*OU_kernel_length) + 4.0*exp(-abs(dif)/OU_kernel_length)*dif^4/((dif^2)^(5/2)*OU_kernel_length^3)
    end

    if dorder==[1, 2, 0]
        func = 3.0*exp(-abs(dif)/OU_kernel_length)*dif/((dif^2)^(3/2)*OU_kernel_length) - 3.0*exp(-abs(dif)/OU_kernel_length)*dif^3/((dif^2)^(5/2)*OU_kernel_length) - exp(-abs(dif)/OU_kernel_length)*dif^3/((dif^2)^(3/2)*OU_kernel_length^3)
    end

    if dorder==[0, 2, 0]
        func = exp(-abs(dif)/OU_kernel_length)/OU_kernel_length^2 - exp(-abs(dif)/OU_kernel_length)/(abs(dif)*OU_kernel_length) + exp(-abs(dif)/OU_kernel_length)*dif^2/((dif^2)^(3/2)*OU_kernel_length)
    end

    if dorder==[2, 1, 0]
        func = -3.0*exp(-abs(dif)/OU_kernel_length)*dif/((dif^2)^(3/2)*OU_kernel_length) + 3.0*exp(-abs(dif)/OU_kernel_length)*dif^3/((dif^2)^(5/2)*OU_kernel_length) + exp(-abs(dif)/OU_kernel_length)*dif^3/((dif^2)^(3/2)*OU_kernel_length^3)
    end

    if dorder==[1, 1, 0]
        func = -exp(-abs(dif)/OU_kernel_length)/OU_kernel_length^2 + exp(-abs(dif)/OU_kernel_length)/(abs(dif)*OU_kernel_length) - exp(-abs(dif)/OU_kernel_length)*dif^2/((dif^2)^(3/2)*OU_kernel_length)
    end

    if dorder==[0, 1, 0]
        func = exp(-abs(dif)/OU_kernel_length)*dif/(abs(dif)*OU_kernel_length)
    end

    if dorder==[2, 0, 0]
        func = exp(-abs(dif)/OU_kernel_length)/OU_kernel_length^2 - exp(-abs(dif)/OU_kernel_length)/(abs(dif)*OU_kernel_length) + exp(-abs(dif)/OU_kernel_length)*dif^2/((dif^2)^(3/2)*OU_kernel_length)
    end

    if dorder==[1, 0, 0]
        func = -exp(-abs(dif)/OU_kernel_length)*dif/(abs(dif)*OU_kernel_length)
    end

    if dorder==[0, 0, 0]
        func = exp(-abs(dif)/OU_kernel_length)
    end

    return float(func)

end


return exponential_kernel, 1  # the function handle and the number of kernel hyperparameters
