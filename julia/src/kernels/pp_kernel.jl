"""
pp_kernel function created by kernel_coder(). Requires 1 hyperparameters. Likely created using pp_kernel_base() as an input.
Use with include("src/kernels/pp_kernel.jl").
hyperparameters == ["λ"]
"""
function pp_kernel(
    hyperparameters::Vector{<:Real},
    δ::Real;
    dorder::Vector{<:Integer}=zeros(Int64, length(hyperparameters) + 2))

    @assert length(hyperparameters)==1 "hyperparameters is the wrong length"
    @assert length(dorder)==(1 + 2) "dorder is the wrong length"
    even_time_derivative = powers_of_negative_one(dorder[2])
    @assert maximum(dorder) < 3 "No more than two time derivatives for either t1 or t2 can be calculated"

    dorder = append!([sum(dorder[1:2])], dorder[3:end])

    λ = hyperparameters[1]

    if abs(δ) > λ
        return 0
    end

    dabs_δ = powers_of_negative_one(δ < 0)  # store derivative of abs()
    abs_δ = abs(δ)

    if dorder==[4, 2]
        func = 38400*(1 - abs_δ/λ)^3/λ^6 + 11520*abs_δ^2*(1 - abs_δ/λ)/λ^8 - 57600*abs_δ*(1 - abs_δ/λ)^2/λ^7 + 120*(48*abs_δ^2/λ^4 + 10*abs_δ/λ^3)*(1 - abs_δ/λ)/λ^4 + 240*(-16*abs_δ^2/λ^3 - 5*abs_δ/λ^2)*abs_δ/λ^6 - 960*(-16*abs_δ^2/λ^3 - 5*abs_δ/λ^2)*(1 - abs_δ/λ)/λ^5 - 1200*(1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*abs_δ/λ^7 + 2400*(1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*(1 - abs_δ/λ)/λ^6 - 480*abs_δ^2*dabs_δ^3*(5*dabs_δ/λ + 16*abs_δ*dabs_δ/λ^2)/λ^7 - 2880*dabs_δ^3*(1 - abs_δ/λ)^2*(5*dabs_δ/λ + 16*abs_δ*dabs_δ/λ^2)/λ^5 + 1440*dabs_δ^3*(1 - abs_δ/λ)^2*(-5*dabs_δ/λ^2 - 32*abs_δ*dabs_δ/λ^3)/λ^4 - 240*dabs_δ^3*(1 - abs_δ/λ)^2*(10*dabs_δ/λ^3 + 96*abs_δ*dabs_δ/λ^4)/λ^3 + 3840*abs_δ*dabs_δ^3*(1 - abs_δ/λ)*(5*dabs_δ/λ + 16*abs_δ*dabs_δ/λ^2)/λ^6 - 960*abs_δ*dabs_δ^3*(1 - abs_δ/λ)*(-5*dabs_δ/λ^2 - 32*abs_δ*dabs_δ/λ^3)/λ^5
    end

    if dorder==[3, 2]
        func = -2880*dabs_δ^3*(1 - abs_δ/λ)^4/λ^5 + 360*(1 - abs_δ/λ)^3*(5*dabs_δ/λ + 16*abs_δ*dabs_δ/λ^2)/λ^4 - 240*(1 - abs_δ/λ)^3*(-5*dabs_δ/λ^2 - 32*abs_δ*dabs_δ/λ^3)/λ^3 + 60*(1 - abs_δ/λ)^3*(10*dabs_δ/λ^3 + 96*abs_δ*dabs_δ/λ^4)/λ^2 - 2880*abs_δ^2*dabs_δ^3*(1 - abs_δ/λ)^2/λ^7 + 7680*abs_δ*dabs_δ^3*(1 - abs_δ/λ)^3/λ^6 + 360*abs_δ^2*(1 - abs_δ/λ)*(5*dabs_δ/λ + 16*abs_δ*dabs_δ/λ^2)/λ^6 - 1080*abs_δ*(1 - abs_δ/λ)^2*(5*dabs_δ/λ + 16*abs_δ*dabs_δ/λ^2)/λ^5 + 360*abs_δ*(1 - abs_δ/λ)^2*(-5*dabs_δ/λ^2 - 32*abs_δ*dabs_δ/λ^3)/λ^4 - 60*(48*abs_δ^2/λ^4 + 10*abs_δ/λ^3)*dabs_δ^3*(1 - abs_δ/λ)^2/λ^3 + 360*(-16*abs_δ^2/λ^3 - 5*abs_δ/λ^2)*dabs_δ^3*(1 - abs_δ/λ)^2/λ^4 - 120*(1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*abs_δ^2*dabs_δ^3/λ^7 - 720*(1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*dabs_δ^3*(1 - abs_δ/λ)^2/λ^5 - 240*(-16*abs_δ^2/λ^3 - 5*abs_δ/λ^2)*abs_δ*dabs_δ^3*(1 - abs_δ/λ)/λ^5 + 960*(1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*abs_δ*dabs_δ^3*(1 - abs_δ/λ)/λ^6
    end

    if dorder==[2, 2]
        func = 96*(1 - abs_δ/λ)^5/λ^4 + 320*abs_δ^2*(1 - abs_δ/λ)^3/λ^6 - 480*abs_δ*(1 - abs_δ/λ)^4/λ^5 + 20*(48*abs_δ^2/λ^4 + 10*abs_δ/λ^3)*(1 - abs_δ/λ)^3/λ^2 - 80*(-16*abs_δ^2/λ^3 - 5*abs_δ/λ^2)*(1 - abs_δ/λ)^3/λ^3 + 120*(1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*(1 - abs_δ/λ)^3/λ^4 - 20*dabs_δ*(1 - abs_δ/λ)^4*(5*dabs_δ/λ + 16*abs_δ*dabs_δ/λ^2)/λ^3 + 20*dabs_δ*(1 - abs_δ/λ)^4*(-5*dabs_δ/λ^2 - 32*abs_δ*dabs_δ/λ^3)/λ^2 - 10*dabs_δ*(1 - abs_δ/λ)^4*(10*dabs_δ/λ^3 + 96*abs_δ*dabs_δ/λ^4)/λ + 120*(-16*abs_δ^2/λ^3 - 5*abs_δ/λ^2)*abs_δ*(1 - abs_δ/λ)^2/λ^4 + 120*(1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*abs_δ^2*(1 - abs_δ/λ)/λ^6 - 360*(1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*abs_δ*(1 - abs_δ/λ)^2/λ^5 - 120*abs_δ^2*dabs_δ*(1 - abs_δ/λ)^2*(5*dabs_δ/λ + 16*abs_δ*dabs_δ/λ^2)/λ^5 + 160*abs_δ*dabs_δ*(1 - abs_δ/λ)^3*(5*dabs_δ/λ + 16*abs_δ*dabs_δ/λ^2)/λ^4 - 80*abs_δ*dabs_δ*(1 - abs_δ/λ)^3*(-5*dabs_δ/λ^2 - 32*abs_δ*dabs_δ/λ^3)/λ^3
    end

    if dorder==[1, 2]
        func = (1 - abs_δ/λ)^5*(10*dabs_δ/λ^3 + 96*abs_δ*dabs_δ/λ^4) + 20*abs_δ^2*(1 - abs_δ/λ)^3*(5*dabs_δ/λ + 16*abs_δ*dabs_δ/λ^2)/λ^4 - 10*abs_δ*(1 - abs_δ/λ)^4*(5*dabs_δ/λ + 16*abs_δ*dabs_δ/λ^2)/λ^3 + 10*abs_δ*(1 - abs_δ/λ)^4*(-5*dabs_δ/λ^2 - 32*abs_δ*dabs_δ/λ^3)/λ^2 - 5*(48*abs_δ^2/λ^4 + 10*abs_δ/λ^3)*dabs_δ*(1 - abs_δ/λ)^4/λ + 10*(-16*abs_δ^2/λ^3 - 5*abs_δ/λ^2)*dabs_δ*(1 - abs_δ/λ)^4/λ^2 - 10*(1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*dabs_δ*(1 - abs_δ/λ)^4/λ^3 - 40*(-16*abs_δ^2/λ^3 - 5*abs_δ/λ^2)*abs_δ*dabs_δ*(1 - abs_δ/λ)^3/λ^3 - 60*(1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*abs_δ^2*dabs_δ*(1 - abs_δ/λ)^2/λ^5 + 80*(1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*abs_δ*dabs_δ*(1 - abs_δ/λ)^3/λ^4
    end

    if dorder==[0, 2]
        func = (48*abs_δ^2/λ^4 + 10*abs_δ/λ^3)*(1 - abs_δ/λ)^5 + 10*(-16*abs_δ^2/λ^3 - 5*abs_δ/λ^2)*abs_δ*(1 - abs_δ/λ)^4/λ^2 + 20*(1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*abs_δ^2*(1 - abs_δ/λ)^3/λ^4 - 10*(1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*abs_δ*(1 - abs_δ/λ)^4/λ^3
    end

    if dorder==[4, 1]
        func = -7680*(1 - abs_δ/λ)^3/λ^5 + 5760*abs_δ*(1 - abs_δ/λ)^2/λ^6 + 120*(-16*abs_δ^2/λ^3 - 5*abs_δ/λ^2)*(1 - abs_δ/λ)/λ^4 + 120*(1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*abs_δ/λ^6 - 480*(1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*(1 - abs_δ/λ)/λ^5 + 720*dabs_δ^3*(1 - abs_δ/λ)^2*(5*dabs_δ/λ + 16*abs_δ*dabs_δ/λ^2)/λ^4 - 240*dabs_δ^3*(1 - abs_δ/λ)^2*(-5*dabs_δ/λ^2 - 32*abs_δ*dabs_δ/λ^3)/λ^3 - 480*abs_δ*dabs_δ^3*(1 - abs_δ/λ)*(5*dabs_δ/λ + 16*abs_δ*dabs_δ/λ^2)/λ^5
    end

    if dorder==[3, 1]
        func = 720*dabs_δ^3*(1 - abs_δ/λ)^4/λ^4 - 120*(1 - abs_δ/λ)^3*(5*dabs_δ/λ + 16*abs_δ*dabs_δ/λ^2)/λ^3 + 60*(1 - abs_δ/λ)^3*(-5*dabs_δ/λ^2 - 32*abs_δ*dabs_δ/λ^3)/λ^2 - 960*abs_δ*dabs_δ^3*(1 - abs_δ/λ)^3/λ^5 + 180*abs_δ*(1 - abs_δ/λ)^2*(5*dabs_δ/λ + 16*abs_δ*dabs_δ/λ^2)/λ^4 - 60*(-16*abs_δ^2/λ^3 - 5*abs_δ/λ^2)*dabs_δ^3*(1 - abs_δ/λ)^2/λ^3 + 180*(1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*dabs_δ^3*(1 - abs_δ/λ)^2/λ^4 - 120*(1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*abs_δ*dabs_δ^3*(1 - abs_δ/λ)/λ^5
    end

    if dorder==[2, 1]
        func = -32*(1 - abs_δ/λ)^5/λ^3 + 80*abs_δ*(1 - abs_δ/λ)^4/λ^4 + 20*(-16*abs_δ^2/λ^3 - 5*abs_δ/λ^2)*(1 - abs_δ/λ)^3/λ^2 - 40*(1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*(1 - abs_δ/λ)^3/λ^3 + 10*dabs_δ*(1 - abs_δ/λ)^4*(5*dabs_δ/λ + 16*abs_δ*dabs_δ/λ^2)/λ^2 - 10*dabs_δ*(1 - abs_δ/λ)^4*(-5*dabs_δ/λ^2 - 32*abs_δ*dabs_δ/λ^3)/λ + 60*(1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*abs_δ*(1 - abs_δ/λ)^2/λ^4 - 40*abs_δ*dabs_δ*(1 - abs_δ/λ)^3*(5*dabs_δ/λ + 16*abs_δ*dabs_δ/λ^2)/λ^3
    end

    if dorder==[1, 1]
        func = (1 - abs_δ/λ)^5*(-5*dabs_δ/λ^2 - 32*abs_δ*dabs_δ/λ^3) + 5*abs_δ*(1 - abs_δ/λ)^4*(5*dabs_δ/λ + 16*abs_δ*dabs_δ/λ^2)/λ^2 - 5*(-16*abs_δ^2/λ^3 - 5*abs_δ/λ^2)*dabs_δ*(1 - abs_δ/λ)^4/λ + 5*(1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*dabs_δ*(1 - abs_δ/λ)^4/λ^2 - 20*(1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*abs_δ*dabs_δ*(1 - abs_δ/λ)^3/λ^3
    end

    if dorder==[0, 1]
        func = (-16*abs_δ^2/λ^3 - 5*abs_δ/λ^2)*(1 - abs_δ/λ)^5 + 5*(1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*abs_δ*(1 - abs_δ/λ)^4/λ^2
    end

    if dorder==[4, 0]
        func = 1920*(1 - abs_δ/λ)^3/λ^4 + 120*(1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*(1 - abs_δ/λ)/λ^4 - 240*dabs_δ^3*(1 - abs_δ/λ)^2*(5*dabs_δ/λ + 16*abs_δ*dabs_δ/λ^2)/λ^3
    end

    if dorder==[3, 0]
        func = -240*dabs_δ^3*(1 - abs_δ/λ)^4/λ^3 + 60*(1 - abs_δ/λ)^3*(5*dabs_δ/λ + 16*abs_δ*dabs_δ/λ^2)/λ^2 - 60*(1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*dabs_δ^3*(1 - abs_δ/λ)^2/λ^3
    end

    if dorder==[2, 0]
        func = 16*(1 - abs_δ/λ)^5/λ^2 + 20*(1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*(1 - abs_δ/λ)^3/λ^2 - 10*dabs_δ*(1 - abs_δ/λ)^4*(5*dabs_δ/λ + 16*abs_δ*dabs_δ/λ^2)/λ
    end

    if dorder==[1, 0]
        func = (1 - abs_δ/λ)^5*(5*dabs_δ/λ + 16*abs_δ*dabs_δ/λ^2) - 5*(1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*dabs_δ*(1 - abs_δ/λ)^4/λ
    end

    if dorder==[0, 0]
        func = (1 + 8*abs_δ^2/λ^2 + 5*abs_δ/λ)*(1 - abs_δ/λ)^5
    end

    return even_time_derivative * float(func)  # correcting for amount of t2 derivatives

end


return pp_kernel, 1  # the function handle and the number of kernel hyperparameters
