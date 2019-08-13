

"""
periodic_kernel function created by kernel_coder(). Requires 2 hyperparameters. Likely created using periodic_kernel_base() as an input. 
Use with include("src/kernels/periodic_kernel.jl").
hyperparameters == ["P", "λ"]
"""
function periodic_kernel(
    hyperparameters::Vector{<:Real}, 
    δ::Real; 
    dorder::Vector{<:Integer}=zeros(Int64, length(hyperparameters) + 2))

    @assert length(hyperparameters)==2 "hyperparameters is the wrong length"
    @assert length(dorder)==(2 + 2) "dorder is the wrong length"
    even_time_derivative = powers_of_negative_one(dorder[2])
    @assert maximum(dorder) < 3 "No more than two time derivatives for either t1 or t2 can be calculated"

    dorder[1] = sum(dorder[1:2])
    dorder[2:(end - 1)] = dorder[3:end]

    deleteat!(dorder, length(dorder))

    P = hyperparameters[1]
    λ = hyperparameters[2]

    if dorder==[4, 0, 2]
        func = 768*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*sin(pi*δ/P)^8/(P^4*λ^10) - 2368*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*sin(pi*δ/P)^6/(P^4*λ^8) + 1408*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*sin(pi*δ/P)^4/(P^4*λ^6) + 960*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*cos(pi*δ/P)^4/(P^4*λ^6) - 96*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*sin(pi*δ/P)^2/(P^4*λ^4) + 96*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*cos(pi*δ/P)^2/(P^4*λ^4) + 4096*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*sin(pi*δ/P)^8*cos(pi*δ/P)^4/(P^4*λ^14) - 25600*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*sin(pi*δ/P)^6*cos(pi*δ/P)^4/(P^4*λ^12) + 6144*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*sin(pi*δ/P)^8*cos(pi*δ/P)^2/(P^4*λ^12) + 42240*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*sin(pi*δ/P)^4*cos(pi*δ/P)^4/(P^4*λ^10) - 28672*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*sin(pi*δ/P)^6*cos(pi*δ/P)^2/(P^4*λ^10) - 18240*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*sin(pi*δ/P)^2*cos(pi*δ/P)^4/(P^4*λ^8) + 31872*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*sin(pi*δ/P)^4*cos(pi*δ/P)^2/(P^4*λ^8) - 7488*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*sin(pi*δ/P)^2*cos(pi*δ/P)^2/(P^4*λ^6)
    end

    if dorder==[3, 0, 2]
        func = -1024*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^3*sin(pi*δ/P)^7*cos(pi*δ/P)^3/(P^3*λ^12) + 4608*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^3*sin(pi*δ/P)^5*cos(pi*δ/P)^3/(P^3*λ^10) - 768*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^3*sin(pi*δ/P)^7*cos(pi*δ/P)/(P^3*λ^10) - 4800*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^3*sin(pi*δ/P)^3*cos(pi*δ/P)^3/(P^3*λ^8) + 2368*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^3*sin(pi*δ/P)^5*cos(pi*δ/P)/(P^3*λ^8) + 960*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^3*sin(pi*δ/P)*cos(pi*δ/P)^3/(P^3*λ^6) - 1408*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^3*sin(pi*δ/P)^3*cos(pi*δ/P)/(P^3*λ^6) + 96*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^3*sin(pi*δ/P)*cos(pi*δ/P)/(P^3*λ^4)
    end

    if dorder==[2, 0, 2]
        func = 64*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^2*sin(pi*δ/P)^6/(P^2*λ^8) - 112*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^2*sin(pi*δ/P)^4/(P^2*λ^6) + 24*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^2*sin(pi*δ/P)^2/(P^2*λ^4) - 24*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^2*cos(pi*δ/P)^2/(P^2*λ^4) + 256*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^2*sin(pi*δ/P)^6*cos(pi*δ/P)^2/(P^2*λ^10) - 768*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^2*sin(pi*δ/P)^4*cos(pi*δ/P)^2/(P^2*λ^8) + 432*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^2*sin(pi*δ/P)^2*cos(pi*δ/P)^2/(P^2*λ^6)
    end

    if dorder==[1, 0, 2]
        func = -64*exp(-2*sin(pi*δ/P)^2/λ^2)*pi*sin(pi*δ/P)^5*cos(pi*δ/P)/(P*λ^8) + 112*exp(-2*sin(pi*δ/P)^2/λ^2)*pi*sin(pi*δ/P)^3*cos(pi*δ/P)/(P*λ^6) - 24*exp(-2*sin(pi*δ/P)^2/λ^2)*pi*sin(pi*δ/P)*cos(pi*δ/P)/(P*λ^4)
    end

    if dorder==[0, 0, 2]
        func = 16*exp(-2*sin(pi*δ/P)^2/λ^2)*sin(pi*δ/P)^4/λ^6 - 12*exp(-2*sin(pi*δ/P)^2/λ^2)*sin(pi*δ/P)^2/λ^4
    end

    if dorder==[4, 1, 1]
        func = -768*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*sin(pi*δ/P)^6/(P^5*λ^7) + 1024*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*sin(pi*δ/P)^4/(P^5*λ^5) + 768*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*cos(pi*δ/P)^4/(P^5*λ^5) - 128*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*sin(pi*δ/P)^2/(P^5*λ^3) + 128*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*cos(pi*δ/P)^2/(P^5*λ^3) - 4096*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*sin(pi*δ/P)^6*cos(pi*δ/P)^4/(P^5*λ^11) + 14336*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*sin(pi*δ/P)^4*cos(pi*δ/P)^4/(P^5*λ^9) - 6144*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*sin(pi*δ/P)^6*cos(pi*δ/P)^2/(P^5*λ^9) - 9984*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*sin(pi*δ/P)^2*cos(pi*δ/P)^4/(P^5*λ^7) + 14848*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*sin(pi*δ/P)^4*cos(pi*δ/P)^2/(P^5*λ^7) - 5888*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*sin(pi*δ/P)^2*cos(pi*δ/P)^2/(P^5*λ^5) + 4096*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^5*δ*sin(pi*δ/P)^7*cos(pi*δ/P)^5/(P^6*λ^13) - 20480*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^5*δ*sin(pi*δ/P)^5*cos(pi*δ/P)^5/(P^6*λ^11) + 10240*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^5*δ*sin(pi*δ/P)^7*cos(pi*δ/P)^3/(P^6*λ^11) + 24320*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^5*δ*sin(pi*δ/P)^3*cos(pi*δ/P)^5/(P^6*λ^9) - 38400*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^5*δ*sin(pi*δ/P)^5*cos(pi*δ/P)^3/(P^6*λ^9) + 3840*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^5*δ*sin(pi*δ/P)^7*cos(pi*δ/P)/(P^6*λ^9) - 5760*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^5*δ*sin(pi*δ/P)*cos(pi*δ/P)^5/(P^6*λ^7) + 30720*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^5*δ*sin(pi*δ/P)^3*cos(pi*δ/P)^3/(P^6*λ^7) - 9600*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^5*δ*sin(pi*δ/P)^5*cos(pi*δ/P)/(P^6*λ^7) - 3840*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^5*δ*sin(pi*δ/P)*cos(pi*δ/P)^3/(P^6*λ^5) + 4096*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^5*δ*sin(pi*δ/P)^3*cos(pi*δ/P)/(P^6*λ^5) - 128*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^5*δ*sin(pi*δ/P)*cos(pi*δ/P)/(P^6*λ^3)
    end

    if dorder==[3, 1, 1]
        func = -192*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*δ*sin(pi*δ/P)^6/(P^5*λ^7) + 256*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*δ*sin(pi*δ/P)^4/(P^5*λ^5) + 192*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*δ*cos(pi*δ/P)^4/(P^5*λ^5) - 32*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*δ*sin(pi*δ/P)^2/(P^5*λ^3) + 32*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*δ*cos(pi*δ/P)^2/(P^5*λ^3) + 768*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^3*sin(pi*δ/P)^5*cos(pi*δ/P)^3/(P^4*λ^9) - 1728*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^3*sin(pi*δ/P)^3*cos(pi*δ/P)^3/(P^4*λ^7) + 576*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^3*sin(pi*δ/P)^5*cos(pi*δ/P)/(P^4*λ^7) + 576*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^3*sin(pi*δ/P)*cos(pi*δ/P)^3/(P^4*λ^5) - 768*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^3*sin(pi*δ/P)^3*cos(pi*δ/P)/(P^4*λ^5) + 96*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^3*sin(pi*δ/P)*cos(pi*δ/P)/(P^4*λ^3) - 1024*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*δ*sin(pi*δ/P)^6*cos(pi*δ/P)^4/(P^5*λ^11) + 3584*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*δ*sin(pi*δ/P)^4*cos(pi*δ/P)^4/(P^5*λ^9) - 1536*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*δ*sin(pi*δ/P)^6*cos(pi*δ/P)^2/(P^5*λ^9) - 2496*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*δ*sin(pi*δ/P)^2*cos(pi*δ/P)^4/(P^5*λ^7) + 3712*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*δ*sin(pi*δ/P)^4*cos(pi*δ/P)^2/(P^5*λ^7) - 1472*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*δ*sin(pi*δ/P)^2*cos(pi*δ/P)^2/(P^5*λ^5)
    end

    if dorder==[2, 1, 1]
        func = -32*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^2*sin(pi*δ/P)^4/(P^3*λ^5) + 16*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^2*sin(pi*δ/P)^2/(P^3*λ^3) - 16*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^2*cos(pi*δ/P)^2/(P^3*λ^3) - 128*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^2*sin(pi*δ/P)^4*cos(pi*δ/P)^2/(P^3*λ^7) + 160*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^2*sin(pi*δ/P)^2*cos(pi*δ/P)^2/(P^3*λ^5) + 256*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^3*δ*sin(pi*δ/P)^5*cos(pi*δ/P)^3/(P^4*λ^9) - 576*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^3*δ*sin(pi*δ/P)^3*cos(pi*δ/P)^3/(P^4*λ^7) + 192*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^3*δ*sin(pi*δ/P)^5*cos(pi*δ/P)/(P^4*λ^7) + 192*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^3*δ*sin(pi*δ/P)*cos(pi*δ/P)^3/(P^4*λ^5) - 256*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^3*δ*sin(pi*δ/P)^3*cos(pi*δ/P)/(P^4*λ^5) + 32*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^3*δ*sin(pi*δ/P)*cos(pi*δ/P)/(P^4*λ^3)
    end

    if dorder==[1, 1, 1]
        func = -16*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^2*δ*sin(pi*δ/P)^4/(P^3*λ^5) + 8*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^2*δ*sin(pi*δ/P)^2/(P^3*λ^3) - 8*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^2*δ*cos(pi*δ/P)^2/(P^3*λ^3) + 16*exp(-2*sin(pi*δ/P)^2/λ^2)*pi*sin(pi*δ/P)^3*cos(pi*δ/P)/(P^2*λ^5) - 8*exp(-2*sin(pi*δ/P)^2/λ^2)*pi*sin(pi*δ/P)*cos(pi*δ/P)/(P^2*λ^3) - 64*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^2*δ*sin(pi*δ/P)^4*cos(pi*δ/P)^2/(P^3*λ^7) + 80*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^2*δ*sin(pi*δ/P)^2*cos(pi*δ/P)^2/(P^3*λ^5)
    end

    if dorder==[0, 1, 1]
        func = 16*exp(-2*sin(pi*δ/P)^2/λ^2)*pi*δ*sin(pi*δ/P)^3*cos(pi*δ/P)/(P^2*λ^5) - 8*exp(-2*sin(pi*δ/P)^2/λ^2)*pi*δ*sin(pi*δ/P)*cos(pi*δ/P)/(P^2*λ^3)
    end

    if dorder==[4, 0, 1]
        func = 192*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*sin(pi*δ/P)^6/(P^4*λ^7) - 256*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*sin(pi*δ/P)^4/(P^4*λ^5) - 192*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*cos(pi*δ/P)^4/(P^4*λ^5) + 32*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*sin(pi*δ/P)^2/(P^4*λ^3) - 32*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*cos(pi*δ/P)^2/(P^4*λ^3) + 1024*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*sin(pi*δ/P)^6*cos(pi*δ/P)^4/(P^4*λ^11) - 3584*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*sin(pi*δ/P)^4*cos(pi*δ/P)^4/(P^4*λ^9) + 1536*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*sin(pi*δ/P)^6*cos(pi*δ/P)^2/(P^4*λ^9) + 2496*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*sin(pi*δ/P)^2*cos(pi*δ/P)^4/(P^4*λ^7) - 3712*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*sin(pi*δ/P)^4*cos(pi*δ/P)^2/(P^4*λ^7) + 1472*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*sin(pi*δ/P)^2*cos(pi*δ/P)^2/(P^4*λ^5)
    end

    if dorder==[3, 0, 1]
        func = -256*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^3*sin(pi*δ/P)^5*cos(pi*δ/P)^3/(P^3*λ^9) + 576*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^3*sin(pi*δ/P)^3*cos(pi*δ/P)^3/(P^3*λ^7) - 192*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^3*sin(pi*δ/P)^5*cos(pi*δ/P)/(P^3*λ^7) - 192*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^3*sin(pi*δ/P)*cos(pi*δ/P)^3/(P^3*λ^5) + 256*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^3*sin(pi*δ/P)^3*cos(pi*δ/P)/(P^3*λ^5) - 32*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^3*sin(pi*δ/P)*cos(pi*δ/P)/(P^3*λ^3)
    end

    if dorder==[2, 0, 1]
        func = 16*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^2*sin(pi*δ/P)^4/(P^2*λ^5) - 8*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^2*sin(pi*δ/P)^2/(P^2*λ^3) + 8*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^2*cos(pi*δ/P)^2/(P^2*λ^3) + 64*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^2*sin(pi*δ/P)^4*cos(pi*δ/P)^2/(P^2*λ^7) - 80*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^2*sin(pi*δ/P)^2*cos(pi*δ/P)^2/(P^2*λ^5)
    end

    if dorder==[1, 0, 1]
        func = -16*exp(-2*sin(pi*δ/P)^2/λ^2)*pi*sin(pi*δ/P)^3*cos(pi*δ/P)/(P*λ^5) + 8*exp(-2*sin(pi*δ/P)^2/λ^2)*pi*sin(pi*δ/P)*cos(pi*δ/P)/(P*λ^3)
    end

    if dorder==[0, 0, 1]
        func = 4*exp(-2*sin(pi*δ/P)^2/λ^2)*sin(pi*δ/P)^2/λ^3
    end

    if dorder==[4, 2, 0]
        func = 960*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*sin(pi*δ/P)^4/(P^6*λ^4) + 960*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*cos(pi*δ/P)^4/(P^6*λ^4) - 320*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*sin(pi*δ/P)^2/(P^6*λ^2) + 320*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*cos(pi*δ/P)^2/(P^6*λ^2) + 960*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^6*δ^2*sin(pi*δ/P)^6/(P^8*λ^6) - 960*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^6*δ^2*cos(pi*δ/P)^6/(P^8*λ^6) - 960*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^6*δ^2*sin(pi*δ/P)^4/(P^8*λ^4) - 960*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^6*δ^2*cos(pi*δ/P)^4/(P^8*λ^4) + 64*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^6*δ^2*sin(pi*δ/P)^2/(P^8*λ^2) - 64*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^6*δ^2*cos(pi*δ/P)^2/(P^8*λ^2) + 5120*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*sin(pi*δ/P)^4*cos(pi*δ/P)^4/(P^6*λ^8) - 7680*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*sin(pi*δ/P)^2*cos(pi*δ/P)^4/(P^6*λ^6) + 7680*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*sin(pi*δ/P)^4*cos(pi*δ/P)^2/(P^6*λ^6) - 7040*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*sin(pi*δ/P)^2*cos(pi*δ/P)^2/(P^6*λ^4) + 4096*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^6*δ^2*sin(pi*δ/P)^6*cos(pi*δ/P)^6/(P^8*λ^12) - 15360*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^6*δ^2*sin(pi*δ/P)^4*cos(pi*δ/P)^6/(P^8*λ^10) + 15360*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^6*δ^2*sin(pi*δ/P)^6*cos(pi*δ/P)^4/(P^8*λ^10) + 11520*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^6*δ^2*sin(pi*δ/P)^2*cos(pi*δ/P)^6/(P^8*λ^8) - 43520*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^6*δ^2*sin(pi*δ/P)^4*cos(pi*δ/P)^4/(P^8*λ^8) + 11520*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^6*δ^2*sin(pi*δ/P)^6*cos(pi*δ/P)^2/(P^8*λ^8) + 22080*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^6*δ^2*sin(pi*δ/P)^2*cos(pi*δ/P)^4/(P^8*λ^6) - 22080*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^6*δ^2*sin(pi*δ/P)^4*cos(pi*δ/P)^2/(P^8*λ^6) + 6016*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^6*δ^2*sin(pi*δ/P)^2*cos(pi*δ/P)^2/(P^8*λ^4) - 10240*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^5*δ*sin(pi*δ/P)^5*cos(pi*δ/P)^5/(P^7*λ^10) + 25600*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^5*δ*sin(pi*δ/P)^3*cos(pi*δ/P)^5/(P^7*λ^8) - 25600*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^5*δ*sin(pi*δ/P)^5*cos(pi*δ/P)^3/(P^7*λ^8) - 9600*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^5*δ*sin(pi*δ/P)*cos(pi*δ/P)^5/(P^7*λ^6) + 44800*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^5*δ*sin(pi*δ/P)^3*cos(pi*δ/P)^3/(P^7*λ^6) - 9600*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^5*δ*sin(pi*δ/P)^5*cos(pi*δ/P)/(P^7*λ^6) - 9600*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^5*δ*sin(pi*δ/P)*cos(pi*δ/P)^3/(P^7*λ^4) + 9600*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^5*δ*sin(pi*δ/P)^3*cos(pi*δ/P)/(P^7*λ^4) - 640*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^5*δ*sin(pi*δ/P)*cos(pi*δ/P)/(P^7*λ^2)
    end

    if dorder==[3, 2, 0]
        func = 384*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*δ*sin(pi*δ/P)^4/(P^6*λ^4) + 384*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*δ*cos(pi*δ/P)^4/(P^6*λ^4) - 128*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*δ*sin(pi*δ/P)^2/(P^6*λ^2) + 128*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*δ*cos(pi*δ/P)^2/(P^6*λ^2) - 768*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^3*sin(pi*δ/P)^3*cos(pi*δ/P)^3/(P^5*λ^6) + 576*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^3*sin(pi*δ/P)*cos(pi*δ/P)^3/(P^5*λ^4) - 576*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^3*sin(pi*δ/P)^3*cos(pi*δ/P)/(P^5*λ^4) + 192*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^3*sin(pi*δ/P)*cos(pi*δ/P)/(P^5*λ^2) - 1024*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^5*δ^2*sin(pi*δ/P)^5*cos(pi*δ/P)^5/(P^7*λ^10) + 2560*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^5*δ^2*sin(pi*δ/P)^3*cos(pi*δ/P)^5/(P^7*λ^8) - 2560*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^5*δ^2*sin(pi*δ/P)^5*cos(pi*δ/P)^3/(P^7*λ^8) - 960*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^5*δ^2*sin(pi*δ/P)*cos(pi*δ/P)^5/(P^7*λ^6) + 4480*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^5*δ^2*sin(pi*δ/P)^3*cos(pi*δ/P)^3/(P^7*λ^6) - 960*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^5*δ^2*sin(pi*δ/P)^5*cos(pi*δ/P)/(P^7*λ^6) - 960*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^5*δ^2*sin(pi*δ/P)*cos(pi*δ/P)^3/(P^7*λ^4) + 960*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^5*δ^2*sin(pi*δ/P)^3*cos(pi*δ/P)/(P^7*λ^4) - 64*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^5*δ^2*sin(pi*δ/P)*cos(pi*δ/P)/(P^7*λ^2) + 2048*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*δ*sin(pi*δ/P)^4*cos(pi*δ/P)^4/(P^6*λ^8) - 3072*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*δ*sin(pi*δ/P)^2*cos(pi*δ/P)^4/(P^6*λ^6) + 3072*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*δ*sin(pi*δ/P)^4*cos(pi*δ/P)^2/(P^6*λ^6) - 2816*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*δ*sin(pi*δ/P)^2*cos(pi*δ/P)^2/(P^6*λ^4)
    end

    if dorder==[2, 2, 0]
        func = 24*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^2*sin(pi*δ/P)^2/(P^4*λ^2) - 24*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^2*cos(pi*δ/P)^2/(P^4*λ^2) + 48*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*δ^2*sin(pi*δ/P)^4/(P^6*λ^4) + 48*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*δ^2*cos(pi*δ/P)^4/(P^6*λ^4) - 16*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*δ^2*sin(pi*δ/P)^2/(P^6*λ^2) + 16*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*δ^2*cos(pi*δ/P)^2/(P^6*λ^2) + 96*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^2*sin(pi*δ/P)^2*cos(pi*δ/P)^2/(P^4*λ^4) + 256*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*δ^2*sin(pi*δ/P)^4*cos(pi*δ/P)^4/(P^6*λ^8) - 384*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*δ^2*sin(pi*δ/P)^2*cos(pi*δ/P)^4/(P^6*λ^6) + 384*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*δ^2*sin(pi*δ/P)^4*cos(pi*δ/P)^2/(P^6*λ^6) - 352*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*δ^2*sin(pi*δ/P)^2*cos(pi*δ/P)^2/(P^6*λ^4) - 384*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^3*δ*sin(pi*δ/P)^3*cos(pi*δ/P)^3/(P^5*λ^6) + 288*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^3*δ*sin(pi*δ/P)*cos(pi*δ/P)^3/(P^5*λ^4) - 288*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^3*δ*sin(pi*δ/P)^3*cos(pi*δ/P)/(P^5*λ^4) + 96*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^3*δ*sin(pi*δ/P)*cos(pi*δ/P)/(P^5*λ^2)
    end

    if dorder==[1, 2, 0]
        func = 16*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^2*δ*sin(pi*δ/P)^2/(P^4*λ^2) - 16*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^2*δ*cos(pi*δ/P)^2/(P^4*λ^2) - 8*exp(-2*sin(pi*δ/P)^2/λ^2)*pi*sin(pi*δ/P)*cos(pi*δ/P)/(P^3*λ^2) - 64*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^3*δ^2*sin(pi*δ/P)^3*cos(pi*δ/P)^3/(P^5*λ^6) + 48*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^3*δ^2*sin(pi*δ/P)*cos(pi*δ/P)^3/(P^5*λ^4) - 48*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^3*δ^2*sin(pi*δ/P)^3*cos(pi*δ/P)/(P^5*λ^4) + 16*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^3*δ^2*sin(pi*δ/P)*cos(pi*δ/P)/(P^5*λ^2) + 64*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^2*δ*sin(pi*δ/P)^2*cos(pi*δ/P)^2/(P^4*λ^4)
    end

    if dorder==[0, 2, 0]
        func = 4*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^2*δ^2*sin(pi*δ/P)^2/(P^4*λ^2) - 4*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^2*δ^2*cos(pi*δ/P)^2/(P^4*λ^2) + 16*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^2*δ^2*sin(pi*δ/P)^2*cos(pi*δ/P)^2/(P^4*λ^4) - 8*exp(-2*sin(pi*δ/P)^2/λ^2)*pi*δ*sin(pi*δ/P)*cos(pi*δ/P)/(P^3*λ^2)
    end

    if dorder==[4, 1, 0]
        func = -192*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*sin(pi*δ/P)^4/(P^5*λ^4) - 192*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*cos(pi*δ/P)^4/(P^5*λ^4) + 64*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*sin(pi*δ/P)^2/(P^5*λ^2) - 64*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*cos(pi*δ/P)^2/(P^5*λ^2) - 1024*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*sin(pi*δ/P)^4*cos(pi*δ/P)^4/(P^5*λ^8) + 1536*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*sin(pi*δ/P)^2*cos(pi*δ/P)^4/(P^5*λ^6) - 1536*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*sin(pi*δ/P)^4*cos(pi*δ/P)^2/(P^5*λ^6) + 1408*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*sin(pi*δ/P)^2*cos(pi*δ/P)^2/(P^5*λ^4) + 1024*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^5*δ*sin(pi*δ/P)^5*cos(pi*δ/P)^5/(P^6*λ^10) - 2560*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^5*δ*sin(pi*δ/P)^3*cos(pi*δ/P)^5/(P^6*λ^8) + 2560*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^5*δ*sin(pi*δ/P)^5*cos(pi*δ/P)^3/(P^6*λ^8) + 960*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^5*δ*sin(pi*δ/P)*cos(pi*δ/P)^5/(P^6*λ^6) - 4480*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^5*δ*sin(pi*δ/P)^3*cos(pi*δ/P)^3/(P^6*λ^6) + 960*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^5*δ*sin(pi*δ/P)^5*cos(pi*δ/P)/(P^6*λ^6) + 960*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^5*δ*sin(pi*δ/P)*cos(pi*δ/P)^3/(P^6*λ^4) - 960*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^5*δ*sin(pi*δ/P)^3*cos(pi*δ/P)/(P^6*λ^4) + 64*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^5*δ*sin(pi*δ/P)*cos(pi*δ/P)/(P^6*λ^2)
    end

    if dorder==[3, 1, 0]
        func = -48*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*δ*sin(pi*δ/P)^4/(P^5*λ^4) - 48*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*δ*cos(pi*δ/P)^4/(P^5*λ^4) + 16*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*δ*sin(pi*δ/P)^2/(P^5*λ^2) - 16*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*δ*cos(pi*δ/P)^2/(P^5*λ^2) + 192*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^3*sin(pi*δ/P)^3*cos(pi*δ/P)^3/(P^4*λ^6) - 144*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^3*sin(pi*δ/P)*cos(pi*δ/P)^3/(P^4*λ^4) + 144*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^3*sin(pi*δ/P)^3*cos(pi*δ/P)/(P^4*λ^4) - 48*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^3*sin(pi*δ/P)*cos(pi*δ/P)/(P^4*λ^2) - 256*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*δ*sin(pi*δ/P)^4*cos(pi*δ/P)^4/(P^5*λ^8) + 384*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*δ*sin(pi*δ/P)^2*cos(pi*δ/P)^4/(P^5*λ^6) - 384*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*δ*sin(pi*δ/P)^4*cos(pi*δ/P)^2/(P^5*λ^6) + 352*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*δ*sin(pi*δ/P)^2*cos(pi*δ/P)^2/(P^5*λ^4)
    end

    if dorder==[2, 1, 0]
        func = -8*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^2*sin(pi*δ/P)^2/(P^3*λ^2) + 8*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^2*cos(pi*δ/P)^2/(P^3*λ^2) - 32*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^2*sin(pi*δ/P)^2*cos(pi*δ/P)^2/(P^3*λ^4) + 64*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^3*δ*sin(pi*δ/P)^3*cos(pi*δ/P)^3/(P^4*λ^6) - 48*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^3*δ*sin(pi*δ/P)*cos(pi*δ/P)^3/(P^4*λ^4) + 48*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^3*δ*sin(pi*δ/P)^3*cos(pi*δ/P)/(P^4*λ^4) - 16*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^3*δ*sin(pi*δ/P)*cos(pi*δ/P)/(P^4*λ^2)
    end

    if dorder==[1, 1, 0]
        func = -4*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^2*δ*sin(pi*δ/P)^2/(P^3*λ^2) + 4*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^2*δ*cos(pi*δ/P)^2/(P^3*λ^2) + 4*exp(-2*sin(pi*δ/P)^2/λ^2)*pi*sin(pi*δ/P)*cos(pi*δ/P)/(P^2*λ^2) - 16*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^2*δ*sin(pi*δ/P)^2*cos(pi*δ/P)^2/(P^3*λ^4)
    end

    if dorder==[0, 1, 0]
        func = 4*exp(-2*sin(pi*δ/P)^2/λ^2)*pi*δ*sin(pi*δ/P)*cos(pi*δ/P)/(P^2*λ^2)
    end

    if dorder==[4, 0, 0]
        func = 48*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*sin(pi*δ/P)^4/(P^4*λ^4) + 48*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*cos(pi*δ/P)^4/(P^4*λ^4) - 16*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*sin(pi*δ/P)^2/(P^4*λ^2) + 16*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*cos(pi*δ/P)^2/(P^4*λ^2) + 256*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*sin(pi*δ/P)^4*cos(pi*δ/P)^4/(P^4*λ^8) - 384*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*sin(pi*δ/P)^2*cos(pi*δ/P)^4/(P^4*λ^6) + 384*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*sin(pi*δ/P)^4*cos(pi*δ/P)^2/(P^4*λ^6) - 352*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^4*sin(pi*δ/P)^2*cos(pi*δ/P)^2/(P^4*λ^4)
    end

    if dorder==[3, 0, 0]
        func = -64*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^3*sin(pi*δ/P)^3*cos(pi*δ/P)^3/(P^3*λ^6) + 48*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^3*sin(pi*δ/P)*cos(pi*δ/P)^3/(P^3*λ^4) - 48*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^3*sin(pi*δ/P)^3*cos(pi*δ/P)/(P^3*λ^4) + 16*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^3*sin(pi*δ/P)*cos(pi*δ/P)/(P^3*λ^2)
    end

    if dorder==[2, 0, 0]
        func = 4*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^2*sin(pi*δ/P)^2/(P^2*λ^2) - 4*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^2*cos(pi*δ/P)^2/(P^2*λ^2) + 16*exp(-2*sin(pi*δ/P)^2/λ^2)*pi^2*sin(pi*δ/P)^2*cos(pi*δ/P)^2/(P^2*λ^4)
    end

    if dorder==[1, 0, 0]
        func = -4*exp(-2*sin(pi*δ/P)^2/λ^2)*pi*sin(pi*δ/P)*cos(pi*δ/P)/(P*λ^2)
    end

    if dorder==[0, 0, 0]
        func = exp(-2*sin(pi*δ/P)^2/λ^2)
    end

    return even_time_derivative * float(func)  # correcting for amount of t2 derivatives

end


return periodic_kernel, 2  # the function handle and the number of kernel hyperparameters
