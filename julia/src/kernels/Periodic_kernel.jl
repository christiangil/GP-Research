

"""
Periodic_kernel function created by kernel_coder(). Requires 2 hyperparameters. Likely created using Periodic_kernel_base() as an input.
Use with include("kernels/Periodic_kernel.jl").
"""
function Periodic_kernel(hyperparameters::Union{Array{Float64,1},Array{Any,1}}, dif::Float64; dorder::Union{Array{Int,1},Array{Float64,1}}=zeros(1))

    @assert length(hyperparameters)==2 "hyperparameters is the wrong length"
    if dorder==zeros(1)
        dorder = zeros(length(hyperparameters) + 2)
    else
        @assert length(dorder)==(length(hyperparameters) + 2) "dorder is the wrong length"
    end
    dorder = convert(Array{Int64,1}, dorder)

    kernel_period = hyperparameters[3-2]
    kernel_length = hyperparameters[4-2]

    if dorder==[2, 2, 0, 1]
        func = 0.0 + 32.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*sin(pi*dif/kernel_period)^2/(kernel_period^4*kernel_length^3) - 256.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*sin(pi*dif/kernel_period)^4/(kernel_period^4*kernel_length^5) + 192.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*sin(pi*dif/kernel_period)^6/(kernel_period^4*kernel_length^7) - 32.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*cos(pi*dif/kernel_period)^2/(kernel_period^4*kernel_length^3) - 192.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*cos(pi*dif/kernel_period)^4/(kernel_period^4*kernel_length^5) + 1472.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*sin(pi*dif/kernel_period)^2*cos(pi*dif/kernel_period)^2/(kernel_period^4*kernel_length^5) + 2496.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*sin(pi*dif/kernel_period)^2*cos(pi*dif/kernel_period)^4/(kernel_period^4*kernel_length^7) - 3712.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*sin(pi*dif/kernel_period)^4*cos(pi*dif/kernel_period)^2/(kernel_period^4*kernel_length^7) - 3584.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*sin(pi*dif/kernel_period)^4*cos(pi*dif/kernel_period)^4/(kernel_period^4*kernel_length^9) + 1536.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*sin(pi*dif/kernel_period)^6*cos(pi*dif/kernel_period)^2/(kernel_period^4*kernel_length^9) + 1024.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*sin(pi*dif/kernel_period)^6*cos(pi*dif/kernel_period)^4/(kernel_period^4*kernel_length^11)
    end

    if dorder==[1, 2, 0, 1]
        func = 0.0 - 32.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*sin(pi*dif/kernel_period)*cos(pi*dif/kernel_period)/(kernel_period^3*kernel_length^3) - 192.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*sin(pi*dif/kernel_period)*cos(pi*dif/kernel_period)^3/(kernel_period^3*kernel_length^5) + 256.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*sin(pi*dif/kernel_period)^3*cos(pi*dif/kernel_period)/(kernel_period^3*kernel_length^5) + 576.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*sin(pi*dif/kernel_period)^3*cos(pi*dif/kernel_period)^3/(kernel_period^3*kernel_length^7) - 192.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*sin(pi*dif/kernel_period)^5*cos(pi*dif/kernel_period)/(kernel_period^3*kernel_length^7) - 256.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*sin(pi*dif/kernel_period)^5*cos(pi*dif/kernel_period)^3/(kernel_period^3*kernel_length^9)
    end

    if dorder==[0, 2, 0, 1]
        func = 0.0 - 8.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^2*sin(pi*dif/kernel_period)^2/(kernel_period^2*kernel_length^3) + 16.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^2*sin(pi*dif/kernel_period)^4/(kernel_period^2*kernel_length^5) + 8.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^2*cos(pi*dif/kernel_period)^2/(kernel_period^2*kernel_length^3) - 80.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^2*sin(pi*dif/kernel_period)^2*cos(pi*dif/kernel_period)^2/(kernel_period^2*kernel_length^5) + 64.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^2*sin(pi*dif/kernel_period)^4*cos(pi*dif/kernel_period)^2/(kernel_period^2*kernel_length^7)
    end

    if dorder==[2, 1, 0, 1]
        func = 0.0 + 32.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*sin(pi*dif/kernel_period)*cos(pi*dif/kernel_period)/(kernel_period^3*kernel_length^3) + 192.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*sin(pi*dif/kernel_period)*cos(pi*dif/kernel_period)^3/(kernel_period^3*kernel_length^5) - 256.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*sin(pi*dif/kernel_period)^3*cos(pi*dif/kernel_period)/(kernel_period^3*kernel_length^5) - 576.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*sin(pi*dif/kernel_period)^3*cos(pi*dif/kernel_period)^3/(kernel_period^3*kernel_length^7) + 192.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*sin(pi*dif/kernel_period)^5*cos(pi*dif/kernel_period)/(kernel_period^3*kernel_length^7) + 256.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*sin(pi*dif/kernel_period)^5*cos(pi*dif/kernel_period)^3/(kernel_period^3*kernel_length^9)
    end

    if dorder==[1, 1, 0, 1]
        func = 0.0 + 8.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^2*sin(pi*dif/kernel_period)^2/(kernel_period^2*kernel_length^3) - 16.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^2*sin(pi*dif/kernel_period)^4/(kernel_period^2*kernel_length^5) - 8.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^2*cos(pi*dif/kernel_period)^2/(kernel_period^2*kernel_length^3) + 80.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^2*sin(pi*dif/kernel_period)^2*cos(pi*dif/kernel_period)^2/(kernel_period^2*kernel_length^5) - 64.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^2*sin(pi*dif/kernel_period)^4*cos(pi*dif/kernel_period)^2/(kernel_period^2*kernel_length^7)
    end

    if dorder==[0, 1, 0, 1]
        func = -8.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi*sin(pi*dif/kernel_period)*cos(pi*dif/kernel_period)/(kernel_period*kernel_length^3) + 16.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi*sin(pi*dif/kernel_period)^3*cos(pi*dif/kernel_period)/(kernel_period*kernel_length^5)
    end

    if dorder==[2, 0, 0, 1]
        func = 0.0 - 8.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^2*sin(pi*dif/kernel_period)^2/(kernel_period^2*kernel_length^3) + 16.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^2*sin(pi*dif/kernel_period)^4/(kernel_period^2*kernel_length^5) + 8.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^2*cos(pi*dif/kernel_period)^2/(kernel_period^2*kernel_length^3) - 80.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^2*sin(pi*dif/kernel_period)^2*cos(pi*dif/kernel_period)^2/(kernel_period^2*kernel_length^5) + 64.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^2*sin(pi*dif/kernel_period)^4*cos(pi*dif/kernel_period)^2/(kernel_period^2*kernel_length^7)
    end

    if dorder==[1, 0, 0, 1]
        func = 8.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi*sin(pi*dif/kernel_period)*cos(pi*dif/kernel_period)/(kernel_period*kernel_length^3) - 16.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi*sin(pi*dif/kernel_period)^3*cos(pi*dif/kernel_period)/(kernel_period*kernel_length^5)
    end

    if dorder==[0, 0, 0, 1]
        func = 4.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*sin(pi*dif/kernel_period)^2/kernel_length^3
    end

    if dorder==[2, 2, 1, 0]
        func = 0.0 + 64.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*sin(pi*dif/kernel_period)^2/(kernel_period^5*kernel_length^2) - 192.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*sin(pi*dif/kernel_period)^4/(kernel_period^5*kernel_length^4) - 64.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*cos(pi*dif/kernel_period)^2/(kernel_period^5*kernel_length^2) - 192.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*cos(pi*dif/kernel_period)^4/(kernel_period^5*kernel_length^4) + 1408.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*sin(pi*dif/kernel_period)^2*cos(pi*dif/kernel_period)^2/(kernel_period^5*kernel_length^4) + 1536.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*sin(pi*dif/kernel_period)^2*cos(pi*dif/kernel_period)^4/(kernel_period^5*kernel_length^6) - 1536.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*sin(pi*dif/kernel_period)^4*cos(pi*dif/kernel_period)^2/(kernel_period^5*kernel_length^6) - 1024.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*sin(pi*dif/kernel_period)^4*cos(pi*dif/kernel_period)^4/(kernel_period^5*kernel_length^8) + 64.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^5*dif*sin(pi*dif/kernel_period)*cos(pi*dif/kernel_period)/(kernel_period^6*kernel_length^2) + 960.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^5*dif*sin(pi*dif/kernel_period)*cos(pi*dif/kernel_period)^3/(kernel_period^6*kernel_length^4) + 960.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^5*dif*sin(pi*dif/kernel_period)*cos(pi*dif/kernel_period)^5/(kernel_period^6*kernel_length^6) - 960.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^5*dif*sin(pi*dif/kernel_period)^3*cos(pi*dif/kernel_period)/(kernel_period^6*kernel_length^4) - 4480.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^5*dif*sin(pi*dif/kernel_period)^3*cos(pi*dif/kernel_period)^3/(kernel_period^6*kernel_length^6) - 2560.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^5*dif*sin(pi*dif/kernel_period)^3*cos(pi*dif/kernel_period)^5/(kernel_period^6*kernel_length^8) + 960.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^5*dif*sin(pi*dif/kernel_period)^5*cos(pi*dif/kernel_period)/(kernel_period^6*kernel_length^6) + 2560.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^5*dif*sin(pi*dif/kernel_period)^5*cos(pi*dif/kernel_period)^3/(kernel_period^6*kernel_length^8) + 1024.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^5*dif*sin(pi*dif/kernel_period)^5*cos(pi*dif/kernel_period)^5/(kernel_period^6*kernel_length^10)
    end

    if dorder==[1, 2, 1, 0]
        func = 0.0 - 48.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*sin(pi*dif/kernel_period)*cos(pi*dif/kernel_period)/(kernel_period^4*kernel_length^2) - 144.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*sin(pi*dif/kernel_period)*cos(pi*dif/kernel_period)^3/(kernel_period^4*kernel_length^4) + 144.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*sin(pi*dif/kernel_period)^3*cos(pi*dif/kernel_period)/(kernel_period^4*kernel_length^4) + 192.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*sin(pi*dif/kernel_period)^3*cos(pi*dif/kernel_period)^3/(kernel_period^4*kernel_length^6) + 16.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*dif*sin(pi*dif/kernel_period)^2/(kernel_period^5*kernel_length^2) - 48.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*dif*sin(pi*dif/kernel_period)^4/(kernel_period^5*kernel_length^4) - 16.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*dif*cos(pi*dif/kernel_period)^2/(kernel_period^5*kernel_length^2) - 48.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*dif*cos(pi*dif/kernel_period)^4/(kernel_period^5*kernel_length^4) + 352.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*dif*sin(pi*dif/kernel_period)^2*cos(pi*dif/kernel_period)^2/(kernel_period^5*kernel_length^4) + 384.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*dif*sin(pi*dif/kernel_period)^2*cos(pi*dif/kernel_period)^4/(kernel_period^5*kernel_length^6) - 384.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*dif*sin(pi*dif/kernel_period)^4*cos(pi*dif/kernel_period)^2/(kernel_period^5*kernel_length^6) - 256.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*dif*sin(pi*dif/kernel_period)^4*cos(pi*dif/kernel_period)^4/(kernel_period^5*kernel_length^8)
    end

    if dorder==[0, 2, 1, 0]
        func = 0.0 - 8.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^2*sin(pi*dif/kernel_period)^2/(kernel_period^3*kernel_length^2) + 8.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^2*cos(pi*dif/kernel_period)^2/(kernel_period^3*kernel_length^2) - 32.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^2*sin(pi*dif/kernel_period)^2*cos(pi*dif/kernel_period)^2/(kernel_period^3*kernel_length^4) - 16.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*dif*sin(pi*dif/kernel_period)*cos(pi*dif/kernel_period)/(kernel_period^4*kernel_length^2) - 48.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*dif*sin(pi*dif/kernel_period)*cos(pi*dif/kernel_period)^3/(kernel_period^4*kernel_length^4) + 48.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*dif*sin(pi*dif/kernel_period)^3*cos(pi*dif/kernel_period)/(kernel_period^4*kernel_length^4) + 64.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*dif*sin(pi*dif/kernel_period)^3*cos(pi*dif/kernel_period)^3/(kernel_period^4*kernel_length^6)
    end

    if dorder==[2, 1, 1, 0]
        func = 0.0 + 48.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*sin(pi*dif/kernel_period)*cos(pi*dif/kernel_period)/(kernel_period^4*kernel_length^2) + 144.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*sin(pi*dif/kernel_period)*cos(pi*dif/kernel_period)^3/(kernel_period^4*kernel_length^4) - 144.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*sin(pi*dif/kernel_period)^3*cos(pi*dif/kernel_period)/(kernel_period^4*kernel_length^4) - 192.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*sin(pi*dif/kernel_period)^3*cos(pi*dif/kernel_period)^3/(kernel_period^4*kernel_length^6) - 16.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*dif*sin(pi*dif/kernel_period)^2/(kernel_period^5*kernel_length^2) + 48.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*dif*sin(pi*dif/kernel_period)^4/(kernel_period^5*kernel_length^4) + 16.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*dif*cos(pi*dif/kernel_period)^2/(kernel_period^5*kernel_length^2) + 48.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*dif*cos(pi*dif/kernel_period)^4/(kernel_period^5*kernel_length^4) - 352.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*dif*sin(pi*dif/kernel_period)^2*cos(pi*dif/kernel_period)^2/(kernel_period^5*kernel_length^4) - 384.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*dif*sin(pi*dif/kernel_period)^2*cos(pi*dif/kernel_period)^4/(kernel_period^5*kernel_length^6) + 384.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*dif*sin(pi*dif/kernel_period)^4*cos(pi*dif/kernel_period)^2/(kernel_period^5*kernel_length^6) + 256.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*dif*sin(pi*dif/kernel_period)^4*cos(pi*dif/kernel_period)^4/(kernel_period^5*kernel_length^8)
    end

    if dorder==[1, 1, 1, 0]
        func = 0.0 + 8.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^2*sin(pi*dif/kernel_period)^2/(kernel_period^3*kernel_length^2) - 8.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^2*cos(pi*dif/kernel_period)^2/(kernel_period^3*kernel_length^2) + 32.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^2*sin(pi*dif/kernel_period)^2*cos(pi*dif/kernel_period)^2/(kernel_period^3*kernel_length^4) + 16.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*dif*sin(pi*dif/kernel_period)*cos(pi*dif/kernel_period)/(kernel_period^4*kernel_length^2) + 48.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*dif*sin(pi*dif/kernel_period)*cos(pi*dif/kernel_period)^3/(kernel_period^4*kernel_length^4) - 48.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*dif*sin(pi*dif/kernel_period)^3*cos(pi*dif/kernel_period)/(kernel_period^4*kernel_length^4) - 64.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*dif*sin(pi*dif/kernel_period)^3*cos(pi*dif/kernel_period)^3/(kernel_period^4*kernel_length^6)
    end

    if dorder==[0, 1, 1, 0]
        func = -4.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi*sin(pi*dif/kernel_period)*cos(pi*dif/kernel_period)/(kernel_period^2*kernel_length^2) + 4.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^2*dif*sin(pi*dif/kernel_period)^2/(kernel_period^3*kernel_length^2) - 4.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^2*dif*cos(pi*dif/kernel_period)^2/(kernel_period^3*kernel_length^2) + 16.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^2*dif*sin(pi*dif/kernel_period)^2*cos(pi*dif/kernel_period)^2/(kernel_period^3*kernel_length^4)
    end

    if dorder==[2, 0, 1, 0]
        func = 0.0 - 8.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^2*sin(pi*dif/kernel_period)^2/(kernel_period^3*kernel_length^2) + 8.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^2*cos(pi*dif/kernel_period)^2/(kernel_period^3*kernel_length^2) - 32.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^2*sin(pi*dif/kernel_period)^2*cos(pi*dif/kernel_period)^2/(kernel_period^3*kernel_length^4) - 16.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*dif*sin(pi*dif/kernel_period)*cos(pi*dif/kernel_period)/(kernel_period^4*kernel_length^2) - 48.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*dif*sin(pi*dif/kernel_period)*cos(pi*dif/kernel_period)^3/(kernel_period^4*kernel_length^4) + 48.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*dif*sin(pi*dif/kernel_period)^3*cos(pi*dif/kernel_period)/(kernel_period^4*kernel_length^4) + 64.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*dif*sin(pi*dif/kernel_period)^3*cos(pi*dif/kernel_period)^3/(kernel_period^4*kernel_length^6)
    end

    if dorder==[1, 0, 1, 0]
        func = 4.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi*sin(pi*dif/kernel_period)*cos(pi*dif/kernel_period)/(kernel_period^2*kernel_length^2) - 4.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^2*dif*sin(pi*dif/kernel_period)^2/(kernel_period^3*kernel_length^2) + 4.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^2*dif*cos(pi*dif/kernel_period)^2/(kernel_period^3*kernel_length^2) - 16.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^2*dif*sin(pi*dif/kernel_period)^2*cos(pi*dif/kernel_period)^2/(kernel_period^3*kernel_length^4)
    end

    if dorder==[0, 0, 1, 0]
        func = 4.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi*dif*sin(pi*dif/kernel_period)*cos(pi*dif/kernel_period)/(kernel_period^2*kernel_length^2)
    end

    if dorder==[2, 2, 0, 0]
        func = 0.0 - 16.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*sin(pi*dif/kernel_period)^2/(kernel_period^4*kernel_length^2) + 48.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*sin(pi*dif/kernel_period)^4/(kernel_period^4*kernel_length^4) + 16.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*cos(pi*dif/kernel_period)^2/(kernel_period^4*kernel_length^2) + 48.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*cos(pi*dif/kernel_period)^4/(kernel_period^4*kernel_length^4) - 352.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*sin(pi*dif/kernel_period)^2*cos(pi*dif/kernel_period)^2/(kernel_period^4*kernel_length^4) - 384.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*sin(pi*dif/kernel_period)^2*cos(pi*dif/kernel_period)^4/(kernel_period^4*kernel_length^6) + 384.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*sin(pi*dif/kernel_period)^4*cos(pi*dif/kernel_period)^2/(kernel_period^4*kernel_length^6) + 256.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^4*sin(pi*dif/kernel_period)^4*cos(pi*dif/kernel_period)^4/(kernel_period^4*kernel_length^8)
    end

    if dorder==[1, 2, 0, 0]
        func = 0.0 + 16.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*sin(pi*dif/kernel_period)*cos(pi*dif/kernel_period)/(kernel_period^3*kernel_length^2) + 48.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*sin(pi*dif/kernel_period)*cos(pi*dif/kernel_period)^3/(kernel_period^3*kernel_length^4) - 48.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*sin(pi*dif/kernel_period)^3*cos(pi*dif/kernel_period)/(kernel_period^3*kernel_length^4) - 64.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*sin(pi*dif/kernel_period)^3*cos(pi*dif/kernel_period)^3/(kernel_period^3*kernel_length^6)
    end

    if dorder==[0, 2, 0, 0]
        func = 4.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^2*sin(pi*dif/kernel_period)^2/(kernel_period^2*kernel_length^2) - 4.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^2*cos(pi*dif/kernel_period)^2/(kernel_period^2*kernel_length^2) + 16.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^2*sin(pi*dif/kernel_period)^2*cos(pi*dif/kernel_period)^2/(kernel_period^2*kernel_length^4)
    end

    if dorder==[2, 1, 0, 0]
        func = 0.0 - 16.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*sin(pi*dif/kernel_period)*cos(pi*dif/kernel_period)/(kernel_period^3*kernel_length^2) - 48.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*sin(pi*dif/kernel_period)*cos(pi*dif/kernel_period)^3/(kernel_period^3*kernel_length^4) + 48.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*sin(pi*dif/kernel_period)^3*cos(pi*dif/kernel_period)/(kernel_period^3*kernel_length^4) + 64.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^3*sin(pi*dif/kernel_period)^3*cos(pi*dif/kernel_period)^3/(kernel_period^3*kernel_length^6)
    end

    if dorder==[1, 1, 0, 0]
        func = -4.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^2*sin(pi*dif/kernel_period)^2/(kernel_period^2*kernel_length^2) + 4.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^2*cos(pi*dif/kernel_period)^2/(kernel_period^2*kernel_length^2) - 16.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^2*sin(pi*dif/kernel_period)^2*cos(pi*dif/kernel_period)^2/(kernel_period^2*kernel_length^4)
    end

    if dorder==[0, 1, 0, 0]
        func = 4.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi*sin(pi*dif/kernel_period)*cos(pi*dif/kernel_period)/(kernel_period*kernel_length^2)
    end

    if dorder==[2, 0, 0, 0]
        func = 4.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^2*sin(pi*dif/kernel_period)^2/(kernel_period^2*kernel_length^2) - 4.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^2*cos(pi*dif/kernel_period)^2/(kernel_period^2*kernel_length^2) + 16.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi^2*sin(pi*dif/kernel_period)^2*cos(pi*dif/kernel_period)^2/(kernel_period^2*kernel_length^4)
    end

    if dorder==[1, 0, 0, 0]
        func = -4.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)*pi*sin(pi*dif/kernel_period)*cos(pi*dif/kernel_period)/(kernel_period*kernel_length^2)
    end

    if dorder==[0, 0, 0, 0]
        func = 1.0*exp(-2*sin(pi*dif/kernel_period)^2/kernel_length^2)
    end

    return float(func)

end


num_kernel_hyperparameters = 2
