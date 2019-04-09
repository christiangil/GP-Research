

"""
periodic_kernel function created by kernel_coder(). Requires 2 hyperparameters. Likely created using periodic_kernel_base() as an input. 
Use with include("src/kernels/periodic_kernel.jl").
hyperparameters == ["kernel_period", "kernel_length"]
"""
function periodic_kernel(hyperparameters::AbstractArray{T1,1}, dif::Real; dorder::AbstractArray{T2,1}=zeros(length(hyperparameters) + 2)) where {T1<:Real, T2<:Real}

    @assert length(hyperparameters)==2 "hyperparameters is the wrong length"
    @assert length(dorder)==(length(hyperparameters) + 2) "dorder is the wrong length"
    dorder = convert(Array{Int64,1}, dorder)
    even_time_derivative = 2 * iseven(dorder[2]) - 1
    @assert maximum(dorder) < 3 "No more than two time derivatives for either t1 or t2 can be calculated"

    dorder = append!([sum(dorder[1:2])], dorder[3:end])

    kernel_period = hyperparameters[1]
    kernel_length = hyperparameters[2]

    if dorder==[4, 0, 1]
        func = 32*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*sin(pi*dif/kernel_period)^2/(kernel_period^4*kernel_length^3) - 256*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*sin(pi*dif/kernel_period)^4/(kernel_period^4*kernel_length^5) + 192*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*sin(pi*dif/kernel_period)^6/(kernel_period^4*kernel_length^7) - 32*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*cos(pi*dif/kernel_period)^2/(kernel_period^4*kernel_length^3) - 192*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*cos(pi*dif/kernel_period)^4/(kernel_period^4*kernel_length^5) + 1472*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*sin(pi*dif/kernel_period)^2*cos(pi*dif/kernel_period)^2/(kernel_period^4*kernel_length^5) + 2496*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*sin(pi*dif/kernel_period)^2*cos(pi*dif/kernel_period)^4/(kernel_period^4*kernel_length^7) - 3712*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*sin(pi*dif/kernel_period)^4*cos(pi*dif/kernel_period)^2/(kernel_period^4*kernel_length^7) - 3584*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*sin(pi*dif/kernel_period)^4*cos(pi*dif/kernel_period)^4/(kernel_period^4*kernel_length^9) + 1536*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*sin(pi*dif/kernel_period)^6*cos(pi*dif/kernel_period)^2/(kernel_period^4*kernel_length^9) + 1024*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*sin(pi*dif/kernel_period)^6*cos(pi*dif/kernel_period)^4/(kernel_period^4*kernel_length^11)
    end

    if dorder==[3, 0, 1]
        func = -32*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*sin(pi*dif/kernel_period)*cos(pi*dif/kernel_period)/(kernel_period^3*kernel_length^3) - 192*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*sin(pi*dif/kernel_period)*cos(pi*dif/kernel_period)^3/(kernel_period^3*kernel_length^5) + 256*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*sin(pi*dif/kernel_period)^3*cos(pi*dif/kernel_period)/(kernel_period^3*kernel_length^5) + 576*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*sin(pi*dif/kernel_period)^3*cos(pi*dif/kernel_period)^3/(kernel_period^3*kernel_length^7) - 192*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*sin(pi*dif/kernel_period)^5*cos(pi*dif/kernel_period)/(kernel_period^3*kernel_length^7) - 256*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*sin(pi*dif/kernel_period)^5*cos(pi*dif/kernel_period)^3/(kernel_period^3*kernel_length^9)
    end

    if dorder==[2, 0, 1]
        func = -8*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^2*sin(pi*dif/kernel_period)^2/(kernel_period^2*kernel_length^3) + 16*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^2*sin(pi*dif/kernel_period)^4/(kernel_period^2*kernel_length^5) + 8*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^2*cos(pi*dif/kernel_period)^2/(kernel_period^2*kernel_length^3) - 80*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^2*sin(pi*dif/kernel_period)^2*cos(pi*dif/kernel_period)^2/(kernel_period^2*kernel_length^5) + 64*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^2*sin(pi*dif/kernel_period)^4*cos(pi*dif/kernel_period)^2/(kernel_period^2*kernel_length^7)
    end

    if dorder==[1, 0, 1]
        func = 8*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi*sin(pi*dif/kernel_period)*cos(pi*dif/kernel_period)/(kernel_period*kernel_length^3) - 16*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi*sin(pi*dif/kernel_period)^3*cos(pi*dif/kernel_period)/(kernel_period*kernel_length^5)
    end

    if dorder==[0, 0, 1]
        func = 4*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*sin(pi*dif/kernel_period)^2/kernel_length^3
    end

    if dorder==[4, 1, 0]
        func = 64*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*sin(pi*dif/kernel_period)^2/(kernel_period^5*kernel_length^2) - 192*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*sin(pi*dif/kernel_period)^4/(kernel_period^5*kernel_length^4) - 64*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*cos(pi*dif/kernel_period)^2/(kernel_period^5*kernel_length^2) - 192*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*cos(pi*dif/kernel_period)^4/(kernel_period^5*kernel_length^4) + 1408*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*sin(pi*dif/kernel_period)^2*cos(pi*dif/kernel_period)^2/(kernel_period^5*kernel_length^4) + 1536*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*sin(pi*dif/kernel_period)^2*cos(pi*dif/kernel_period)^4/(kernel_period^5*kernel_length^6) - 1536*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*sin(pi*dif/kernel_period)^4*cos(pi*dif/kernel_period)^2/(kernel_period^5*kernel_length^6) - 1024*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*sin(pi*dif/kernel_period)^4*cos(pi*dif/kernel_period)^4/(kernel_period^5*kernel_length^8) + 64*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^5*dif*sin(pi*dif/kernel_period)*cos(pi*dif/kernel_period)/(kernel_period^6*kernel_length^2) + 960*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^5*dif*sin(pi*dif/kernel_period)*cos(pi*dif/kernel_period)^3/(kernel_period^6*kernel_length^4) + 960*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^5*dif*sin(pi*dif/kernel_period)*cos(pi*dif/kernel_period)^5/(kernel_period^6*kernel_length^6) - 960*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^5*dif*sin(pi*dif/kernel_period)^3*cos(pi*dif/kernel_period)/(kernel_period^6*kernel_length^4) - 4480*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^5*dif*sin(pi*dif/kernel_period)^3*cos(pi*dif/kernel_period)^3/(kernel_period^6*kernel_length^6) - 2560*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^5*dif*sin(pi*dif/kernel_period)^3*cos(pi*dif/kernel_period)^5/(kernel_period^6*kernel_length^8) + 960*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^5*dif*sin(pi*dif/kernel_period)^5*cos(pi*dif/kernel_period)/(kernel_period^6*kernel_length^6) + 2560*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^5*dif*sin(pi*dif/kernel_period)^5*cos(pi*dif/kernel_period)^3/(kernel_period^6*kernel_length^8) + 1024*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^5*dif*sin(pi*dif/kernel_period)^5*cos(pi*dif/kernel_period)^5/(kernel_period^6*kernel_length^10)
    end

    if dorder==[3, 1, 0]
        func = -48*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*sin(pi*dif/kernel_period)*cos(pi*dif/kernel_period)/(kernel_period^4*kernel_length^2) - 144*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*sin(pi*dif/kernel_period)*cos(pi*dif/kernel_period)^3/(kernel_period^4*kernel_length^4) + 144*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*sin(pi*dif/kernel_period)^3*cos(pi*dif/kernel_period)/(kernel_period^4*kernel_length^4) + 192*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*sin(pi*dif/kernel_period)^3*cos(pi*dif/kernel_period)^3/(kernel_period^4*kernel_length^6) + 16*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*dif*sin(pi*dif/kernel_period)^2/(kernel_period^5*kernel_length^2) - 48*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*dif*sin(pi*dif/kernel_period)^4/(kernel_period^5*kernel_length^4) - 16*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*dif*cos(pi*dif/kernel_period)^2/(kernel_period^5*kernel_length^2) - 48*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*dif*cos(pi*dif/kernel_period)^4/(kernel_period^5*kernel_length^4) + 352*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*dif*sin(pi*dif/kernel_period)^2*cos(pi*dif/kernel_period)^2/(kernel_period^5*kernel_length^4) + 384*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*dif*sin(pi*dif/kernel_period)^2*cos(pi*dif/kernel_period)^4/(kernel_period^5*kernel_length^6) - 384*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*dif*sin(pi*dif/kernel_period)^4*cos(pi*dif/kernel_period)^2/(kernel_period^5*kernel_length^6) - 256*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*dif*sin(pi*dif/kernel_period)^4*cos(pi*dif/kernel_period)^4/(kernel_period^5*kernel_length^8)
    end

    if dorder==[2, 1, 0]
        func = -8*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^2*sin(pi*dif/kernel_period)^2/(kernel_period^3*kernel_length^2) + 8*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^2*cos(pi*dif/kernel_period)^2/(kernel_period^3*kernel_length^2) - 32*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^2*sin(pi*dif/kernel_period)^2*cos(pi*dif/kernel_period)^2/(kernel_period^3*kernel_length^4) - 16*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*dif*sin(pi*dif/kernel_period)*cos(pi*dif/kernel_period)/(kernel_period^4*kernel_length^2) - 48*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*dif*sin(pi*dif/kernel_period)*cos(pi*dif/kernel_period)^3/(kernel_period^4*kernel_length^4) + 48*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*dif*sin(pi*dif/kernel_period)^3*cos(pi*dif/kernel_period)/(kernel_period^4*kernel_length^4) + 64*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*dif*sin(pi*dif/kernel_period)^3*cos(pi*dif/kernel_period)^3/(kernel_period^4*kernel_length^6)
    end

    if dorder==[1, 1, 0]
        func = 4*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi*sin(pi*dif/kernel_period)*cos(pi*dif/kernel_period)/(kernel_period^2*kernel_length^2) - 4*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^2*dif*sin(pi*dif/kernel_period)^2/(kernel_period^3*kernel_length^2) + 4*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^2*dif*cos(pi*dif/kernel_period)^2/(kernel_period^3*kernel_length^2) - 16*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^2*dif*sin(pi*dif/kernel_period)^2*cos(pi*dif/kernel_period)^2/(kernel_period^3*kernel_length^4)
    end

    if dorder==[0, 1, 0]
        func = 4*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi*dif*sin(pi*dif/kernel_period)*cos(pi*dif/kernel_period)/(kernel_period^2*kernel_length^2)
    end

    if dorder==[4, 0, 0]
        func = -16*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*sin(pi*dif/kernel_period)^2/(kernel_period^4*kernel_length^2) + 48*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*sin(pi*dif/kernel_period)^4/(kernel_period^4*kernel_length^4) + 16*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*cos(pi*dif/kernel_period)^2/(kernel_period^4*kernel_length^2) + 48*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*cos(pi*dif/kernel_period)^4/(kernel_period^4*kernel_length^4) - 352*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*sin(pi*dif/kernel_period)^2*cos(pi*dif/kernel_period)^2/(kernel_period^4*kernel_length^4) - 384*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*sin(pi*dif/kernel_period)^2*cos(pi*dif/kernel_period)^4/(kernel_period^4*kernel_length^6) + 384*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*sin(pi*dif/kernel_period)^4*cos(pi*dif/kernel_period)^2/(kernel_period^4*kernel_length^6) + 256*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*sin(pi*dif/kernel_period)^4*cos(pi*dif/kernel_period)^4/(kernel_period^4*kernel_length^8)
    end

    if dorder==[3, 0, 0]
        func = 16*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*sin(pi*dif/kernel_period)*cos(pi*dif/kernel_period)/(kernel_period^3*kernel_length^2) + 48*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*sin(pi*dif/kernel_period)*cos(pi*dif/kernel_period)^3/(kernel_period^3*kernel_length^4) - 48*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*sin(pi*dif/kernel_period)^3*cos(pi*dif/kernel_period)/(kernel_period^3*kernel_length^4) - 64*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*sin(pi*dif/kernel_period)^3*cos(pi*dif/kernel_period)^3/(kernel_period^3*kernel_length^6)
    end

    if dorder==[2, 0, 0]
        func = 4*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^2*sin(pi*dif/kernel_period)^2/(kernel_period^2*kernel_length^2) - 4*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^2*cos(pi*dif/kernel_period)^2/(kernel_period^2*kernel_length^2) + 16*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^2*sin(pi*dif/kernel_period)^2*cos(pi*dif/kernel_period)^2/(kernel_period^2*kernel_length^4)
    end

    if dorder==[1, 0, 0]
        func = -4*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi*sin(pi*dif/kernel_period)*cos(pi*dif/kernel_period)/(kernel_period*kernel_length^2)
    end

    if dorder==[0, 0, 0]
        func = exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)
    end

    return even_time_derivative * float(func)  # correcting for amount of t2 derivatives

end


return periodic_kernel, 2  # the function handle and the number of kernel hyperparameters