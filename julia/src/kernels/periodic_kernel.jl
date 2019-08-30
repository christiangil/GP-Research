

"""
periodic_kernel function created by kernel_coder(). Requires 2 hyperparameters. Likely created using periodic_kernel_base() as an input. 
Use with include("src/kernels/periodic_kernel.jl").
hyperparameters == ["λ", "se_P"]
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

    λ = hyperparameters[1]
    se_P = hyperparameters[2]

    if dorder==[4, 0, 2]
        func = 960*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*sin(pi*δ/se_P)^4/(λ^4*se_P^6) + 960*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*cos(pi*δ/se_P)^4/(λ^4*se_P^6) - 320*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*sin(pi*δ/se_P)^2/(λ^2*se_P^6) + 320*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*cos(pi*δ/se_P)^2/(λ^2*se_P^6) + 5120*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*sin(pi*δ/se_P)^4*cos(pi*δ/se_P)^4/(λ^8*se_P^6) - 7680*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*sin(pi*δ/se_P)^2*cos(pi*δ/se_P)^4/(λ^6*se_P^6) + 7680*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*sin(pi*δ/se_P)^4*cos(pi*δ/se_P)^2/(λ^6*se_P^6) - 7040*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*sin(pi*δ/se_P)^2*cos(pi*δ/se_P)^2/(λ^4*se_P^6) + 960*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^6*δ^2*sin(pi*δ/se_P)^6/(λ^6*se_P^8) - 960*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^6*δ^2*cos(pi*δ/se_P)^6/(λ^6*se_P^8) - 960*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^6*δ^2*sin(pi*δ/se_P)^4/(λ^4*se_P^8) - 960*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^6*δ^2*cos(pi*δ/se_P)^4/(λ^4*se_P^8) + 64*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^6*δ^2*sin(pi*δ/se_P)^2/(λ^2*se_P^8) - 64*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^6*δ^2*cos(pi*δ/se_P)^2/(λ^2*se_P^8) - 10240*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^5*δ*sin(pi*δ/se_P)^5*cos(pi*δ/se_P)^5/(λ^10*se_P^7) + 25600*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^5*δ*sin(pi*δ/se_P)^3*cos(pi*δ/se_P)^5/(λ^8*se_P^7) - 25600*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^5*δ*sin(pi*δ/se_P)^5*cos(pi*δ/se_P)^3/(λ^8*se_P^7) - 9600*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^5*δ*sin(pi*δ/se_P)*cos(pi*δ/se_P)^5/(λ^6*se_P^7) + 44800*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^5*δ*sin(pi*δ/se_P)^3*cos(pi*δ/se_P)^3/(λ^6*se_P^7) - 9600*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^5*δ*sin(pi*δ/se_P)^5*cos(pi*δ/se_P)/(λ^6*se_P^7) - 9600*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^5*δ*sin(pi*δ/se_P)*cos(pi*δ/se_P)^3/(λ^4*se_P^7) + 9600*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^5*δ*sin(pi*δ/se_P)^3*cos(pi*δ/se_P)/(λ^4*se_P^7) - 640*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^5*δ*sin(pi*δ/se_P)*cos(pi*δ/se_P)/(λ^2*se_P^7) + 4096*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^6*δ^2*sin(pi*δ/se_P)^6*cos(pi*δ/se_P)^6/(λ^12*se_P^8) - 15360*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^6*δ^2*sin(pi*δ/se_P)^4*cos(pi*δ/se_P)^6/(λ^10*se_P^8) + 15360*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^6*δ^2*sin(pi*δ/se_P)^6*cos(pi*δ/se_P)^4/(λ^10*se_P^8) + 11520*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^6*δ^2*sin(pi*δ/se_P)^2*cos(pi*δ/se_P)^6/(λ^8*se_P^8) - 43520*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^6*δ^2*sin(pi*δ/se_P)^4*cos(pi*δ/se_P)^4/(λ^8*se_P^8) + 11520*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^6*δ^2*sin(pi*δ/se_P)^6*cos(pi*δ/se_P)^2/(λ^8*se_P^8) + 22080*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^6*δ^2*sin(pi*δ/se_P)^2*cos(pi*δ/se_P)^4/(λ^6*se_P^8) - 22080*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^6*δ^2*sin(pi*δ/se_P)^4*cos(pi*δ/se_P)^2/(λ^6*se_P^8) + 6016*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^6*δ^2*sin(pi*δ/se_P)^2*cos(pi*δ/se_P)^2/(λ^4*se_P^8)
    end

    if dorder==[3, 0, 2]
        func = -768*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^3*sin(pi*δ/se_P)^3*cos(pi*δ/se_P)^3/(λ^6*se_P^5) + 576*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^3*sin(pi*δ/se_P)*cos(pi*δ/se_P)^3/(λ^4*se_P^5) - 576*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^3*sin(pi*δ/se_P)^3*cos(pi*δ/se_P)/(λ^4*se_P^5) + 192*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^3*sin(pi*δ/se_P)*cos(pi*δ/se_P)/(λ^2*se_P^5) + 384*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*δ*sin(pi*δ/se_P)^4/(λ^4*se_P^6) + 384*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*δ*cos(pi*δ/se_P)^4/(λ^4*se_P^6) - 128*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*δ*sin(pi*δ/se_P)^2/(λ^2*se_P^6) + 128*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*δ*cos(pi*δ/se_P)^2/(λ^2*se_P^6) + 2048*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*δ*sin(pi*δ/se_P)^4*cos(pi*δ/se_P)^4/(λ^8*se_P^6) - 3072*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*δ*sin(pi*δ/se_P)^2*cos(pi*δ/se_P)^4/(λ^6*se_P^6) + 3072*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*δ*sin(pi*δ/se_P)^4*cos(pi*δ/se_P)^2/(λ^6*se_P^6) - 2816*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*δ*sin(pi*δ/se_P)^2*cos(pi*δ/se_P)^2/(λ^4*se_P^6) - 1024*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^5*δ^2*sin(pi*δ/se_P)^5*cos(pi*δ/se_P)^5/(λ^10*se_P^7) + 2560*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^5*δ^2*sin(pi*δ/se_P)^3*cos(pi*δ/se_P)^5/(λ^8*se_P^7) - 2560*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^5*δ^2*sin(pi*δ/se_P)^5*cos(pi*δ/se_P)^3/(λ^8*se_P^7) - 960*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^5*δ^2*sin(pi*δ/se_P)*cos(pi*δ/se_P)^5/(λ^6*se_P^7) + 4480*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^5*δ^2*sin(pi*δ/se_P)^3*cos(pi*δ/se_P)^3/(λ^6*se_P^7) - 960*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^5*δ^2*sin(pi*δ/se_P)^5*cos(pi*δ/se_P)/(λ^6*se_P^7) - 960*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^5*δ^2*sin(pi*δ/se_P)*cos(pi*δ/se_P)^3/(λ^4*se_P^7) + 960*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^5*δ^2*sin(pi*δ/se_P)^3*cos(pi*δ/se_P)/(λ^4*se_P^7) - 64*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^5*δ^2*sin(pi*δ/se_P)*cos(pi*δ/se_P)/(λ^2*se_P^7)
    end

    if dorder==[2, 0, 2]
        func = 24*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^2*sin(pi*δ/se_P)^2/(λ^2*se_P^4) - 24*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^2*cos(pi*δ/se_P)^2/(λ^2*se_P^4) + 96*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^2*sin(pi*δ/se_P)^2*cos(pi*δ/se_P)^2/(λ^4*se_P^4) + 48*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*δ^2*sin(pi*δ/se_P)^4/(λ^4*se_P^6) + 48*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*δ^2*cos(pi*δ/se_P)^4/(λ^4*se_P^6) - 16*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*δ^2*sin(pi*δ/se_P)^2/(λ^2*se_P^6) + 16*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*δ^2*cos(pi*δ/se_P)^2/(λ^2*se_P^6) - 384*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^3*δ*sin(pi*δ/se_P)^3*cos(pi*δ/se_P)^3/(λ^6*se_P^5) + 288*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^3*δ*sin(pi*δ/se_P)*cos(pi*δ/se_P)^3/(λ^4*se_P^5) - 288*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^3*δ*sin(pi*δ/se_P)^3*cos(pi*δ/se_P)/(λ^4*se_P^5) + 96*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^3*δ*sin(pi*δ/se_P)*cos(pi*δ/se_P)/(λ^2*se_P^5) + 256*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*δ^2*sin(pi*δ/se_P)^4*cos(pi*δ/se_P)^4/(λ^8*se_P^6) - 384*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*δ^2*sin(pi*δ/se_P)^2*cos(pi*δ/se_P)^4/(λ^6*se_P^6) + 384*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*δ^2*sin(pi*δ/se_P)^4*cos(pi*δ/se_P)^2/(λ^6*se_P^6) - 352*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*δ^2*sin(pi*δ/se_P)^2*cos(pi*δ/se_P)^2/(λ^4*se_P^6)
    end

    if dorder==[1, 0, 2]
        func = -8*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi*sin(pi*δ/se_P)*cos(pi*δ/se_P)/(λ^2*se_P^3) + 16*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^2*δ*sin(pi*δ/se_P)^2/(λ^2*se_P^4) - 16*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^2*δ*cos(pi*δ/se_P)^2/(λ^2*se_P^4) + 64*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^2*δ*sin(pi*δ/se_P)^2*cos(pi*δ/se_P)^2/(λ^4*se_P^4) - 64*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^3*δ^2*sin(pi*δ/se_P)^3*cos(pi*δ/se_P)^3/(λ^6*se_P^5) + 48*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^3*δ^2*sin(pi*δ/se_P)*cos(pi*δ/se_P)^3/(λ^4*se_P^5) - 48*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^3*δ^2*sin(pi*δ/se_P)^3*cos(pi*δ/se_P)/(λ^4*se_P^5) + 16*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^3*δ^2*sin(pi*δ/se_P)*cos(pi*δ/se_P)/(λ^2*se_P^5)
    end

    if dorder==[0, 0, 2]
        func = 4*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^2*δ^2*sin(pi*δ/se_P)^2/(λ^2*se_P^4) - 4*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^2*δ^2*cos(pi*δ/se_P)^2/(λ^2*se_P^4) - 8*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi*δ*sin(pi*δ/se_P)*cos(pi*δ/se_P)/(λ^2*se_P^3) + 16*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^2*δ^2*sin(pi*δ/se_P)^2*cos(pi*δ/se_P)^2/(λ^4*se_P^4)
    end

    if dorder==[4, 1, 1]
        func = -768*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*sin(pi*δ/se_P)^6/(λ^7*se_P^5) + 1024*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*sin(pi*δ/se_P)^4/(λ^5*se_P^5) + 768*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*cos(pi*δ/se_P)^4/(λ^5*se_P^5) - 128*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*sin(pi*δ/se_P)^2/(λ^3*se_P^5) + 128*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*cos(pi*δ/se_P)^2/(λ^3*se_P^5) - 4096*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*sin(pi*δ/se_P)^6*cos(pi*δ/se_P)^4/(λ^11*se_P^5) + 14336*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*sin(pi*δ/se_P)^4*cos(pi*δ/se_P)^4/(λ^9*se_P^5) - 6144*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*sin(pi*δ/se_P)^6*cos(pi*δ/se_P)^2/(λ^9*se_P^5) - 9984*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*sin(pi*δ/se_P)^2*cos(pi*δ/se_P)^4/(λ^7*se_P^5) + 14848*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*sin(pi*δ/se_P)^4*cos(pi*δ/se_P)^2/(λ^7*se_P^5) - 5888*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*sin(pi*δ/se_P)^2*cos(pi*δ/se_P)^2/(λ^5*se_P^5) + 4096*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^5*δ*sin(pi*δ/se_P)^7*cos(pi*δ/se_P)^5/(λ^13*se_P^6) - 20480*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^5*δ*sin(pi*δ/se_P)^5*cos(pi*δ/se_P)^5/(λ^11*se_P^6) + 10240*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^5*δ*sin(pi*δ/se_P)^7*cos(pi*δ/se_P)^3/(λ^11*se_P^6) + 24320*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^5*δ*sin(pi*δ/se_P)^3*cos(pi*δ/se_P)^5/(λ^9*se_P^6) - 38400*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^5*δ*sin(pi*δ/se_P)^5*cos(pi*δ/se_P)^3/(λ^9*se_P^6) + 3840*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^5*δ*sin(pi*δ/se_P)^7*cos(pi*δ/se_P)/(λ^9*se_P^6) - 5760*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^5*δ*sin(pi*δ/se_P)*cos(pi*δ/se_P)^5/(λ^7*se_P^6) + 30720*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^5*δ*sin(pi*δ/se_P)^3*cos(pi*δ/se_P)^3/(λ^7*se_P^6) - 9600*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^5*δ*sin(pi*δ/se_P)^5*cos(pi*δ/se_P)/(λ^7*se_P^6) - 3840*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^5*δ*sin(pi*δ/se_P)*cos(pi*δ/se_P)^3/(λ^5*se_P^6) + 4096*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^5*δ*sin(pi*δ/se_P)^3*cos(pi*δ/se_P)/(λ^5*se_P^6) - 128*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^5*δ*sin(pi*δ/se_P)*cos(pi*δ/se_P)/(λ^3*se_P^6)
    end

    if dorder==[3, 1, 1]
        func = 768*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^3*sin(pi*δ/se_P)^5*cos(pi*δ/se_P)^3/(λ^9*se_P^4) - 1728*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^3*sin(pi*δ/se_P)^3*cos(pi*δ/se_P)^3/(λ^7*se_P^4) + 576*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^3*sin(pi*δ/se_P)^5*cos(pi*δ/se_P)/(λ^7*se_P^4) + 576*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^3*sin(pi*δ/se_P)*cos(pi*δ/se_P)^3/(λ^5*se_P^4) - 768*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^3*sin(pi*δ/se_P)^3*cos(pi*δ/se_P)/(λ^5*se_P^4) + 96*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^3*sin(pi*δ/se_P)*cos(pi*δ/se_P)/(λ^3*se_P^4) - 192*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*δ*sin(pi*δ/se_P)^6/(λ^7*se_P^5) + 256*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*δ*sin(pi*δ/se_P)^4/(λ^5*se_P^5) + 192*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*δ*cos(pi*δ/se_P)^4/(λ^5*se_P^5) - 32*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*δ*sin(pi*δ/se_P)^2/(λ^3*se_P^5) + 32*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*δ*cos(pi*δ/se_P)^2/(λ^3*se_P^5) - 1024*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*δ*sin(pi*δ/se_P)^6*cos(pi*δ/se_P)^4/(λ^11*se_P^5) + 3584*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*δ*sin(pi*δ/se_P)^4*cos(pi*δ/se_P)^4/(λ^9*se_P^5) - 1536*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*δ*sin(pi*δ/se_P)^6*cos(pi*δ/se_P)^2/(λ^9*se_P^5) - 2496*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*δ*sin(pi*δ/se_P)^2*cos(pi*δ/se_P)^4/(λ^7*se_P^5) + 3712*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*δ*sin(pi*δ/se_P)^4*cos(pi*δ/se_P)^2/(λ^7*se_P^5) - 1472*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*δ*sin(pi*δ/se_P)^2*cos(pi*δ/se_P)^2/(λ^5*se_P^5)
    end

    if dorder==[2, 1, 1]
        func = -32*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^2*sin(pi*δ/se_P)^4/(λ^5*se_P^3) + 16*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^2*sin(pi*δ/se_P)^2/(λ^3*se_P^3) - 16*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^2*cos(pi*δ/se_P)^2/(λ^3*se_P^3) - 128*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^2*sin(pi*δ/se_P)^4*cos(pi*δ/se_P)^2/(λ^7*se_P^3) + 160*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^2*sin(pi*δ/se_P)^2*cos(pi*δ/se_P)^2/(λ^5*se_P^3) + 256*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^3*δ*sin(pi*δ/se_P)^5*cos(pi*δ/se_P)^3/(λ^9*se_P^4) - 576*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^3*δ*sin(pi*δ/se_P)^3*cos(pi*δ/se_P)^3/(λ^7*se_P^4) + 192*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^3*δ*sin(pi*δ/se_P)^5*cos(pi*δ/se_P)/(λ^7*se_P^4) + 192*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^3*δ*sin(pi*δ/se_P)*cos(pi*δ/se_P)^3/(λ^5*se_P^4) - 256*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^3*δ*sin(pi*δ/se_P)^3*cos(pi*δ/se_P)/(λ^5*se_P^4) + 32*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^3*δ*sin(pi*δ/se_P)*cos(pi*δ/se_P)/(λ^3*se_P^4)
    end

    if dorder==[1, 1, 1]
        func = 16*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi*sin(pi*δ/se_P)^3*cos(pi*δ/se_P)/(λ^5*se_P^2) - 8*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi*sin(pi*δ/se_P)*cos(pi*δ/se_P)/(λ^3*se_P^2) - 16*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^2*δ*sin(pi*δ/se_P)^4/(λ^5*se_P^3) + 8*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^2*δ*sin(pi*δ/se_P)^2/(λ^3*se_P^3) - 8*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^2*δ*cos(pi*δ/se_P)^2/(λ^3*se_P^3) - 64*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^2*δ*sin(pi*δ/se_P)^4*cos(pi*δ/se_P)^2/(λ^7*se_P^3) + 80*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^2*δ*sin(pi*δ/se_P)^2*cos(pi*δ/se_P)^2/(λ^5*se_P^3)
    end

    if dorder==[0, 1, 1]
        func = 16*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi*δ*sin(pi*δ/se_P)^3*cos(pi*δ/se_P)/(λ^5*se_P^2) - 8*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi*δ*sin(pi*δ/se_P)*cos(pi*δ/se_P)/(λ^3*se_P^2)
    end

    if dorder==[4, 0, 1]
        func = -192*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*sin(pi*δ/se_P)^4/(λ^4*se_P^5) - 192*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*cos(pi*δ/se_P)^4/(λ^4*se_P^5) + 64*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*sin(pi*δ/se_P)^2/(λ^2*se_P^5) - 64*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*cos(pi*δ/se_P)^2/(λ^2*se_P^5) - 1024*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*sin(pi*δ/se_P)^4*cos(pi*δ/se_P)^4/(λ^8*se_P^5) + 1536*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*sin(pi*δ/se_P)^2*cos(pi*δ/se_P)^4/(λ^6*se_P^5) - 1536*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*sin(pi*δ/se_P)^4*cos(pi*δ/se_P)^2/(λ^6*se_P^5) + 1408*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*sin(pi*δ/se_P)^2*cos(pi*δ/se_P)^2/(λ^4*se_P^5) + 1024*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^5*δ*sin(pi*δ/se_P)^5*cos(pi*δ/se_P)^5/(λ^10*se_P^6) - 2560*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^5*δ*sin(pi*δ/se_P)^3*cos(pi*δ/se_P)^5/(λ^8*se_P^6) + 2560*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^5*δ*sin(pi*δ/se_P)^5*cos(pi*δ/se_P)^3/(λ^8*se_P^6) + 960*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^5*δ*sin(pi*δ/se_P)*cos(pi*δ/se_P)^5/(λ^6*se_P^6) - 4480*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^5*δ*sin(pi*δ/se_P)^3*cos(pi*δ/se_P)^3/(λ^6*se_P^6) + 960*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^5*δ*sin(pi*δ/se_P)^5*cos(pi*δ/se_P)/(λ^6*se_P^6) + 960*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^5*δ*sin(pi*δ/se_P)*cos(pi*δ/se_P)^3/(λ^4*se_P^6) - 960*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^5*δ*sin(pi*δ/se_P)^3*cos(pi*δ/se_P)/(λ^4*se_P^6) + 64*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^5*δ*sin(pi*δ/se_P)*cos(pi*δ/se_P)/(λ^2*se_P^6)
    end

    if dorder==[3, 0, 1]
        func = 192*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^3*sin(pi*δ/se_P)^3*cos(pi*δ/se_P)^3/(λ^6*se_P^4) - 144*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^3*sin(pi*δ/se_P)*cos(pi*δ/se_P)^3/(λ^4*se_P^4) + 144*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^3*sin(pi*δ/se_P)^3*cos(pi*δ/se_P)/(λ^4*se_P^4) - 48*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^3*sin(pi*δ/se_P)*cos(pi*δ/se_P)/(λ^2*se_P^4) - 48*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*δ*sin(pi*δ/se_P)^4/(λ^4*se_P^5) - 48*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*δ*cos(pi*δ/se_P)^4/(λ^4*se_P^5) + 16*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*δ*sin(pi*δ/se_P)^2/(λ^2*se_P^5) - 16*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*δ*cos(pi*δ/se_P)^2/(λ^2*se_P^5) - 256*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*δ*sin(pi*δ/se_P)^4*cos(pi*δ/se_P)^4/(λ^8*se_P^5) + 384*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*δ*sin(pi*δ/se_P)^2*cos(pi*δ/se_P)^4/(λ^6*se_P^5) - 384*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*δ*sin(pi*δ/se_P)^4*cos(pi*δ/se_P)^2/(λ^6*se_P^5) + 352*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*δ*sin(pi*δ/se_P)^2*cos(pi*δ/se_P)^2/(λ^4*se_P^5)
    end

    if dorder==[2, 0, 1]
        func = -8*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^2*sin(pi*δ/se_P)^2/(λ^2*se_P^3) + 8*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^2*cos(pi*δ/se_P)^2/(λ^2*se_P^3) - 32*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^2*sin(pi*δ/se_P)^2*cos(pi*δ/se_P)^2/(λ^4*se_P^3) + 64*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^3*δ*sin(pi*δ/se_P)^3*cos(pi*δ/se_P)^3/(λ^6*se_P^4) - 48*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^3*δ*sin(pi*δ/se_P)*cos(pi*δ/se_P)^3/(λ^4*se_P^4) + 48*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^3*δ*sin(pi*δ/se_P)^3*cos(pi*δ/se_P)/(λ^4*se_P^4) - 16*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^3*δ*sin(pi*δ/se_P)*cos(pi*δ/se_P)/(λ^2*se_P^4)
    end

    if dorder==[1, 0, 1]
        func = 4*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi*sin(pi*δ/se_P)*cos(pi*δ/se_P)/(λ^2*se_P^2) - 4*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^2*δ*sin(pi*δ/se_P)^2/(λ^2*se_P^3) + 4*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^2*δ*cos(pi*δ/se_P)^2/(λ^2*se_P^3) - 16*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^2*δ*sin(pi*δ/se_P)^2*cos(pi*δ/se_P)^2/(λ^4*se_P^3)
    end

    if dorder==[0, 0, 1]
        func = 4*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi*δ*sin(pi*δ/se_P)*cos(pi*δ/se_P)/(λ^2*se_P^2)
    end

    if dorder==[4, 2, 0]
        func = 768*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*sin(pi*δ/se_P)^8/(λ^10*se_P^4) - 2368*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*sin(pi*δ/se_P)^6/(λ^8*se_P^4) + 1408*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*sin(pi*δ/se_P)^4/(λ^6*se_P^4) + 960*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*cos(pi*δ/se_P)^4/(λ^6*se_P^4) - 96*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*sin(pi*δ/se_P)^2/(λ^4*se_P^4) + 96*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*cos(pi*δ/se_P)^2/(λ^4*se_P^4) + 4096*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*sin(pi*δ/se_P)^8*cos(pi*δ/se_P)^4/(λ^14*se_P^4) - 25600*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*sin(pi*δ/se_P)^6*cos(pi*δ/se_P)^4/(λ^12*se_P^4) + 6144*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*sin(pi*δ/se_P)^8*cos(pi*δ/se_P)^2/(λ^12*se_P^4) + 42240*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*sin(pi*δ/se_P)^4*cos(pi*δ/se_P)^4/(λ^10*se_P^4) - 28672*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*sin(pi*δ/se_P)^6*cos(pi*δ/se_P)^2/(λ^10*se_P^4) - 18240*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*sin(pi*δ/se_P)^2*cos(pi*δ/se_P)^4/(λ^8*se_P^4) + 31872*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*sin(pi*δ/se_P)^4*cos(pi*δ/se_P)^2/(λ^8*se_P^4) - 7488*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*sin(pi*δ/se_P)^2*cos(pi*δ/se_P)^2/(λ^6*se_P^4)
    end

    if dorder==[3, 2, 0]
        func = -1024*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^3*sin(pi*δ/se_P)^7*cos(pi*δ/se_P)^3/(λ^12*se_P^3) + 4608*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^3*sin(pi*δ/se_P)^5*cos(pi*δ/se_P)^3/(λ^10*se_P^3) - 768*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^3*sin(pi*δ/se_P)^7*cos(pi*δ/se_P)/(λ^10*se_P^3) - 4800*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^3*sin(pi*δ/se_P)^3*cos(pi*δ/se_P)^3/(λ^8*se_P^3) + 2368*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^3*sin(pi*δ/se_P)^5*cos(pi*δ/se_P)/(λ^8*se_P^3) + 960*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^3*sin(pi*δ/se_P)*cos(pi*δ/se_P)^3/(λ^6*se_P^3) - 1408*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^3*sin(pi*δ/se_P)^3*cos(pi*δ/se_P)/(λ^6*se_P^3) + 96*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^3*sin(pi*δ/se_P)*cos(pi*δ/se_P)/(λ^4*se_P^3)
    end

    if dorder==[2, 2, 0]
        func = 64*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^2*sin(pi*δ/se_P)^6/(λ^8*se_P^2) - 112*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^2*sin(pi*δ/se_P)^4/(λ^6*se_P^2) + 24*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^2*sin(pi*δ/se_P)^2/(λ^4*se_P^2) - 24*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^2*cos(pi*δ/se_P)^2/(λ^4*se_P^2) + 256*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^2*sin(pi*δ/se_P)^6*cos(pi*δ/se_P)^2/(λ^10*se_P^2) - 768*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^2*sin(pi*δ/se_P)^4*cos(pi*δ/se_P)^2/(λ^8*se_P^2) + 432*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^2*sin(pi*δ/se_P)^2*cos(pi*δ/se_P)^2/(λ^6*se_P^2)
    end

    if dorder==[1, 2, 0]
        func = -64*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi*sin(pi*δ/se_P)^5*cos(pi*δ/se_P)/(λ^8*se_P) + 112*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi*sin(pi*δ/se_P)^3*cos(pi*δ/se_P)/(λ^6*se_P) - 24*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi*sin(pi*δ/se_P)*cos(pi*δ/se_P)/(λ^4*se_P)
    end

    if dorder==[0, 2, 0]
        func = 16*exp(-2*sin(pi*δ/se_P)^2/λ^2)*sin(pi*δ/se_P)^4/λ^6 - 12*exp(-2*sin(pi*δ/se_P)^2/λ^2)*sin(pi*δ/se_P)^2/λ^4
    end

    if dorder==[4, 1, 0]
        func = 192*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*sin(pi*δ/se_P)^6/(λ^7*se_P^4) - 256*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*sin(pi*δ/se_P)^4/(λ^5*se_P^4) - 192*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*cos(pi*δ/se_P)^4/(λ^5*se_P^4) + 32*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*sin(pi*δ/se_P)^2/(λ^3*se_P^4) - 32*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*cos(pi*δ/se_P)^2/(λ^3*se_P^4) + 1024*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*sin(pi*δ/se_P)^6*cos(pi*δ/se_P)^4/(λ^11*se_P^4) - 3584*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*sin(pi*δ/se_P)^4*cos(pi*δ/se_P)^4/(λ^9*se_P^4) + 1536*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*sin(pi*δ/se_P)^6*cos(pi*δ/se_P)^2/(λ^9*se_P^4) + 2496*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*sin(pi*δ/se_P)^2*cos(pi*δ/se_P)^4/(λ^7*se_P^4) - 3712*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*sin(pi*δ/se_P)^4*cos(pi*δ/se_P)^2/(λ^7*se_P^4) + 1472*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*sin(pi*δ/se_P)^2*cos(pi*δ/se_P)^2/(λ^5*se_P^4)
    end

    if dorder==[3, 1, 0]
        func = -256*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^3*sin(pi*δ/se_P)^5*cos(pi*δ/se_P)^3/(λ^9*se_P^3) + 576*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^3*sin(pi*δ/se_P)^3*cos(pi*δ/se_P)^3/(λ^7*se_P^3) - 192*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^3*sin(pi*δ/se_P)^5*cos(pi*δ/se_P)/(λ^7*se_P^3) - 192*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^3*sin(pi*δ/se_P)*cos(pi*δ/se_P)^3/(λ^5*se_P^3) + 256*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^3*sin(pi*δ/se_P)^3*cos(pi*δ/se_P)/(λ^5*se_P^3) - 32*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^3*sin(pi*δ/se_P)*cos(pi*δ/se_P)/(λ^3*se_P^3)
    end

    if dorder==[2, 1, 0]
        func = 16*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^2*sin(pi*δ/se_P)^4/(λ^5*se_P^2) - 8*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^2*sin(pi*δ/se_P)^2/(λ^3*se_P^2) + 8*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^2*cos(pi*δ/se_P)^2/(λ^3*se_P^2) + 64*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^2*sin(pi*δ/se_P)^4*cos(pi*δ/se_P)^2/(λ^7*se_P^2) - 80*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^2*sin(pi*δ/se_P)^2*cos(pi*δ/se_P)^2/(λ^5*se_P^2)
    end

    if dorder==[1, 1, 0]
        func = -16*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi*sin(pi*δ/se_P)^3*cos(pi*δ/se_P)/(λ^5*se_P) + 8*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi*sin(pi*δ/se_P)*cos(pi*δ/se_P)/(λ^3*se_P)
    end

    if dorder==[0, 1, 0]
        func = 4*exp(-2*sin(pi*δ/se_P)^2/λ^2)*sin(pi*δ/se_P)^2/λ^3
    end

    if dorder==[4, 0, 0]
        func = 48*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*sin(pi*δ/se_P)^4/(λ^4*se_P^4) + 48*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*cos(pi*δ/se_P)^4/(λ^4*se_P^4) - 16*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*sin(pi*δ/se_P)^2/(λ^2*se_P^4) + 16*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*cos(pi*δ/se_P)^2/(λ^2*se_P^4) + 256*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*sin(pi*δ/se_P)^4*cos(pi*δ/se_P)^4/(λ^8*se_P^4) - 384*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*sin(pi*δ/se_P)^2*cos(pi*δ/se_P)^4/(λ^6*se_P^4) + 384*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*sin(pi*δ/se_P)^4*cos(pi*δ/se_P)^2/(λ^6*se_P^4) - 352*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^4*sin(pi*δ/se_P)^2*cos(pi*δ/se_P)^2/(λ^4*se_P^4)
    end

    if dorder==[3, 0, 0]
        func = -64*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^3*sin(pi*δ/se_P)^3*cos(pi*δ/se_P)^3/(λ^6*se_P^3) + 48*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^3*sin(pi*δ/se_P)*cos(pi*δ/se_P)^3/(λ^4*se_P^3) - 48*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^3*sin(pi*δ/se_P)^3*cos(pi*δ/se_P)/(λ^4*se_P^3) + 16*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^3*sin(pi*δ/se_P)*cos(pi*δ/se_P)/(λ^2*se_P^3)
    end

    if dorder==[2, 0, 0]
        func = 4*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^2*sin(pi*δ/se_P)^2/(λ^2*se_P^2) - 4*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^2*cos(pi*δ/se_P)^2/(λ^2*se_P^2) + 16*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi^2*sin(pi*δ/se_P)^2*cos(pi*δ/se_P)^2/(λ^4*se_P^2)
    end

    if dorder==[1, 0, 0]
        func = -4*exp(-2*sin(pi*δ/se_P)^2/λ^2)*pi*sin(pi*δ/se_P)*cos(pi*δ/se_P)/(λ^2*se_P)
    end

    if dorder==[0, 0, 0]
        func = exp(-2*sin(pi*δ/se_P)^2/λ^2)
    end

    return even_time_derivative * float(func)  # correcting for amount of t2 derivatives

end


return periodic_kernel, 2  # the function handle and the number of kernel hyperparameters
