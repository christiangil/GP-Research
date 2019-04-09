

"""
rq_kernel function created by kernel_coder(). Requires 2 hyperparameters. Likely created using rq_kernel_base() as an input. 
Use with include("src/kernels/rq_kernel.jl").
hyperparameters == ["alpha", "kernel_length"]
"""
function rq_kernel(hyperparameters::AbstractArray{T1,1}, dif::Real; dorder::AbstractArray{T2,1}=zeros(length(hyperparameters) + 2)) where {T1<:Real, T2<:Real}

    @assert length(hyperparameters)==2 "hyperparameters is the wrong length"
    @assert length(dorder)==(length(hyperparameters) + 2) "dorder is the wrong length"
    dorder = convert(Array{Int64,1}, dorder)
    even_time_derivative = 2 * iseven(dorder[2]) - 1
    @assert maximum(dorder) < 3 "No more than two time derivatives for either t1 or t2 can be calculated"

    dorder = append!([sum(dorder[1:2])], dorder[3:end])

    alpha = hyperparameters[1]
    kernel_length = hyperparameters[2]

    if dorder==[4, 0, 1]
        func = 12*(-1 - alpha)*(1 + (1/2)*dif^2/(alpha*kernel_length^2))^(-2 - alpha)/(alpha*kernel_length^5) + 39*dif^2*(-1 - alpha)*(-2 - alpha)*(1 + (1/2)*dif^2/(alpha*kernel_length^2))^(-3 - alpha)/(alpha^2*kernel_length^7) + 14*dif^4*(-3 - alpha)*(-1 - alpha)*(-2 - alpha)*(1 + (1/2)*dif^2/(alpha*kernel_length^2))^(-4 - alpha)/(alpha^3*kernel_length^9) + dif^6*(-3 - alpha)*(-4 - alpha)*(-1 - alpha)*(-2 - alpha)*(1 + (1/2)*dif^2/(alpha*kernel_length^2))^(-5 - alpha)/(alpha^4*kernel_length^11)
    end

    if dorder==[3, 0, 1]
        func = 12*dif*(-1 - alpha)*(1 + (1/2)*dif^2/(alpha*kernel_length^2))^(-2 - alpha)/(alpha*kernel_length^5) + 9*dif^3*(-1 - alpha)*(-2 - alpha)*(1 + (1/2)*dif^2/(alpha*kernel_length^2))^(-3 - alpha)/(alpha^2*kernel_length^7) + dif^5*(-3 - alpha)*(-1 - alpha)*(-2 - alpha)*(1 + (1/2)*dif^2/(alpha*kernel_length^2))^(-4 - alpha)/(alpha^3*kernel_length^9)
    end

    if dorder==[2, 0, 1]
        func = 2*(1 + (1/2)*dif^2/(alpha*kernel_length^2))^(-1 - alpha)/kernel_length^3 + 5*dif^2*(-1 - alpha)*(1 + (1/2)*dif^2/(alpha*kernel_length^2))^(-2 - alpha)/(alpha*kernel_length^5) + dif^4*(-1 - alpha)*(-2 - alpha)*(1 + (1/2)*dif^2/(alpha*kernel_length^2))^(-3 - alpha)/(alpha^2*kernel_length^7)
    end

    if dorder==[1, 0, 1]
        func = 2*dif*(1 + (1/2)*dif^2/(alpha*kernel_length^2))^(-1 - alpha)/kernel_length^3 + dif^3*(-1 - alpha)*(1 + (1/2)*dif^2/(alpha*kernel_length^2))^(-2 - alpha)/(alpha*kernel_length^5)
    end

    if dorder==[0, 0, 1]
        func = dif^2*(1 + (1/2)*dif^2/(alpha*kernel_length^2))^(-1 - alpha)/kernel_length^3
    end

    if dorder==[4, 1, 0]
        func = 3*(1 + (1/2)*dif^2/(alpha*kernel_length^2))^(-2 - alpha)/(alpha*kernel_length^4) + 3*(-1 - alpha)*(1 + (1/2)*dif^2/(alpha*kernel_length^2))^(-2 - alpha)/(alpha^2*kernel_length^4) - 3*(-1 - alpha)*(1 + (1/2)*dif^2/(alpha*kernel_length^2))^(-2 - alpha)*((-1/2)*dif^2*(-2 - alpha)/(alpha^2*(1 + (1/2)*dif^2/(alpha*kernel_length^2))*kernel_length^2) - log(1 + (1/2)*dif^2/(alpha*kernel_length^2)))/(alpha*kernel_length^4) + 6*dif^2*(-2 - alpha)*(1 + (1/2)*dif^2/(alpha*kernel_length^2))^(-3 - alpha)/(alpha^2*kernel_length^6) + 6*dif^2*(-1 - alpha)*(1 + (1/2)*dif^2/(alpha*kernel_length^2))^(-3 - alpha)/(alpha^2*kernel_length^6) + 12*dif^2*(-1 - alpha)*(-2 - alpha)*(1 + (1/2)*dif^2/(alpha*kernel_length^2))^(-3 - alpha)/(alpha^3*kernel_length^6) + dif^4*(-3 - alpha)*(-2 - alpha)*(1 + (1/2)*dif^2/(alpha*kernel_length^2))^(-4 - alpha)/(alpha^3*kernel_length^8) + dif^4*(-3 - alpha)*(-1 - alpha)*(1 + (1/2)*dif^2/(alpha*kernel_length^2))^(-4 - alpha)/(alpha^3*kernel_length^8) + dif^4*(-1 - alpha)*(-2 - alpha)*(1 + (1/2)*dif^2/(alpha*kernel_length^2))^(-4 - alpha)/(alpha^3*kernel_length^8) - 6*dif^2*(-1 - alpha)*(-2 - alpha)*(1 + (1/2)*dif^2/(alpha*kernel_length^2))^(-3 - alpha)*((-1/2)*dif^2*(-3 - alpha)/(alpha^2*(1 + (1/2)*dif^2/(alpha*kernel_length^2))*kernel_length^2) - log(1 + (1/2)*dif^2/(alpha*kernel_length^2)))/(alpha^2*kernel_length^6) + 3*dif^4*(-3 - alpha)*(-1 - alpha)*(-2 - alpha)*(1 + (1/2)*dif^2/(alpha*kernel_length^2))^(-4 - alpha)/(alpha^4*kernel_length^8) - dif^4*(-3 - alpha)*(-1 - alpha)*(-2 - alpha)*(1 + (1/2)*dif^2/(alpha*kernel_length^2))^(-4 - alpha)*((-1/2)*dif^2*(-4 - alpha)/(alpha^2*(1 + (1/2)*dif^2/(alpha*kernel_length^2))*kernel_length^2) - log(1 + (1/2)*dif^2/(alpha*kernel_length^2)))/(alpha^3*kernel_length^8)
    end

    if dorder==[3, 1, 0]
        func = 3*dif*(1 + (1/2)*dif^2/(alpha*kernel_length^2))^(-2 - alpha)/(alpha*kernel_length^4) + 3*dif*(-1 - alpha)*(1 + (1/2)*dif^2/(alpha*kernel_length^2))^(-2 - alpha)/(alpha^2*kernel_length^4) + dif^3*(-2 - alpha)*(1 + (1/2)*dif^2/(alpha*kernel_length^2))^(-3 - alpha)/(alpha^2*kernel_length^6) + dif^3*(-1 - alpha)*(1 + (1/2)*dif^2/(alpha*kernel_length^2))^(-3 - alpha)/(alpha^2*kernel_length^6) - 3*dif*(-1 - alpha)*(1 + (1/2)*dif^2/(alpha*kernel_length^2))^(-2 - alpha)*((-1/2)*dif^2*(-2 - alpha)/(alpha^2*(1 + (1/2)*dif^2/(alpha*kernel_length^2))*kernel_length^2) - log(1 + (1/2)*dif^2/(alpha*kernel_length^2)))/(alpha*kernel_length^4) + 2*dif^3*(-1 - alpha)*(-2 - alpha)*(1 + (1/2)*dif^2/(alpha*kernel_length^2))^(-3 - alpha)/(alpha^3*kernel_length^6) - dif^3*(-1 - alpha)*(-2 - alpha)*(1 + (1/2)*dif^2/(alpha*kernel_length^2))^(-3 - alpha)*((-1/2)*dif^2*(-3 - alpha)/(alpha^2*(1 + (1/2)*dif^2/(alpha*kernel_length^2))*kernel_length^2) - log(1 + (1/2)*dif^2/(alpha*kernel_length^2)))/(alpha^2*kernel_length^6)
    end

    if dorder==[2, 1, 0]
        func = -(1 + (1/2)*dif^2/(alpha*kernel_length^2))^(-1 - alpha)*((-1/2)*dif^2*(-1 - alpha)/(alpha^2*(1 + (1/2)*dif^2/(alpha*kernel_length^2))*kernel_length^2) - log(1 + (1/2)*dif^2/(alpha*kernel_length^2)))/kernel_length^2 + dif^2*(1 + (1/2)*dif^2/(alpha*kernel_length^2))^(-2 - alpha)/(alpha*kernel_length^4) + dif^2*(-1 - alpha)*(1 + (1/2)*dif^2/(alpha*kernel_length^2))^(-2 - alpha)/(alpha^2*kernel_length^4) - dif^2*(-1 - alpha)*(1 + (1/2)*dif^2/(alpha*kernel_length^2))^(-2 - alpha)*((-1/2)*dif^2*(-2 - alpha)/(alpha^2*(1 + (1/2)*dif^2/(alpha*kernel_length^2))*kernel_length^2) - log(1 + (1/2)*dif^2/(alpha*kernel_length^2)))/(alpha*kernel_length^4)
    end

    if dorder==[1, 1, 0]
        func = -dif*(1 + (1/2)*dif^2/(alpha*kernel_length^2))^(-1 - alpha)*((-1/2)*dif^2*(-1 - alpha)/(alpha^2*(1 + (1/2)*dif^2/(alpha*kernel_length^2))*kernel_length^2) - log(1 + (1/2)*dif^2/(alpha*kernel_length^2)))/kernel_length^2
    end

    if dorder==[0, 1, 0]
        func = (1 + (1/2)*dif^2/(alpha*kernel_length^2))^(-alpha)*((1/2)*dif^2/(alpha*(1 + (1/2)*dif^2/(alpha*kernel_length^2))*kernel_length^2) - log(1 + (1/2)*dif^2/(alpha*kernel_length^2)))
    end

    if dorder==[4, 0, 0]
        func = -3*(-1 - alpha)*(1 + (1/2)*dif^2/(alpha*kernel_length^2))^(-2 - alpha)/(alpha*kernel_length^4) - 6*dif^2*(-1 - alpha)*(-2 - alpha)*(1 + (1/2)*dif^2/(alpha*kernel_length^2))^(-3 - alpha)/(alpha^2*kernel_length^6) - dif^4*(-3 - alpha)*(-1 - alpha)*(-2 - alpha)*(1 + (1/2)*dif^2/(alpha*kernel_length^2))^(-4 - alpha)/(alpha^3*kernel_length^8)
    end

    if dorder==[3, 0, 0]
        func = -3*dif*(-1 - alpha)*(1 + (1/2)*dif^2/(alpha*kernel_length^2))^(-2 - alpha)/(alpha*kernel_length^4) - dif^3*(-1 - alpha)*(-2 - alpha)*(1 + (1/2)*dif^2/(alpha*kernel_length^2))^(-3 - alpha)/(alpha^2*kernel_length^6)
    end

    if dorder==[2, 0, 0]
        func = -(1 + (1/2)*dif^2/(alpha*kernel_length^2))^(-1 - alpha)/kernel_length^2 - dif^2*(-1 - alpha)*(1 + (1/2)*dif^2/(alpha*kernel_length^2))^(-2 - alpha)/(alpha*kernel_length^4)
    end

    if dorder==[1, 0, 0]
        func = -dif*(1 + (1/2)*dif^2/(alpha*kernel_length^2))^(-1 - alpha)/kernel_length^2
    end

    if dorder==[0, 0, 0]
        func = (1 + (1/2)*dif^2/(alpha*kernel_length^2))^(-alpha)
    end

    return even_time_derivative * float(func)  # correcting for amount of t2 derivatives

end


return rq_kernel, 2  # the function handle and the number of kernel hyperparameters
