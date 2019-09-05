

"""
m52x2_kernel function created by kernel_coder(). Requires 3 hyperparameters. Likely created using m52x2_kernel_base() as an input. 
Use with include("src/kernels/m52x2_kernel.jl").
hyperparameters == ["λ1", "λ2", "ratio"]
"""
function m52x2_kernel(
    hyperparameters::Vector{<:Real}, 
    δ::Real; 
    dorder::Vector{<:Integer}=zeros(Int64, length(hyperparameters) + 2))

    @assert length(hyperparameters)==3 "hyperparameters is the wrong length"
    @assert length(dorder)==(3 + 2) "dorder is the wrong length"
    even_time_derivative = powers_of_negative_one(dorder[2])
    @assert maximum(dorder) < 3 "No more than two time derivatives for either t1 or t2 can be calculated"

    dorder = append!([sum(dorder[1:2])], dorder[3:end])

    λ1 = hyperparameters[1]
    λ2 = hyperparameters[2]
    ratio = hyperparameters[3]

    dabs_δ = powers_of_negative_one(δ < 0)  # store derivative of abs()
    abs_δ = abs(δ)

    dabs_ratio = powers_of_negative_one(ratio < 0)  # store derivative of abs()
    abs_ratio = abs(ratio)

    if dorder==[4, 0, 0, 2]
        func = 0.0
    end

    if dorder==[3, 0, 0, 2]
        func = 0.0
    end

    if dorder==[2, 0, 0, 2]
        func = 0.0
    end

    if dorder==[1, 0, 0, 2]
        func = 0.0
    end

    if dorder==[0, 0, 0, 2]
        func = 0
    end

    if dorder==[4, 0, 1, 1]
        func = -100.0*exp(-2.23606797749979*abs_δ/λ2)*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)*dabs_ratio/λ2^5 + 25.0*exp(-2.23606797749979*abs_δ/λ2)*(-1.66666666666667*abs_δ^2/λ2^3 - 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2^2)*dabs_ratio/λ2^4 - 60.0*exp(-2.23606797749979*abs_δ/λ2)*(3.33333333333333*λ2^(-2))*dabs_ratio/λ2^3 + 30.0*exp(-2.23606797749979*abs_δ/λ2)*(-6.66666666666667*λ2^(-3))*dabs_ratio/λ2^2 + 55.9016994374948*exp(-2.23606797749979*abs_δ/λ2)*abs_δ*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)*dabs_ratio/λ2^6 + 67.0820393249937*exp(-2.23606797749979*abs_δ/λ2)*abs_δ*(3.33333333333333*λ2^(-2))*dabs_ratio/λ2^4 + 134.164078649987*exp(-2.23606797749979*abs_δ/λ2)*dabs_δ^3*dabs_ratio*(1.66666666666667*abs_δ*dabs_δ/λ2^2 + 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2^4 - 44.7213595499958*exp(-2.23606797749979*abs_δ/λ2)*dabs_δ^3*dabs_ratio*(-5.0*abs_δ*dabs_δ/λ2^3 - 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2^2)/λ2^3 - 100.0*exp(-2.23606797749979*abs_δ/λ2)*abs_δ*dabs_δ^3*dabs_ratio*(1.66666666666667*abs_δ*dabs_δ/λ2^2 + 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2^5
    end

    if dorder==[3, 0, 1, 1]
        func = -30.0*exp(-2.23606797749979*abs_δ/λ2)*dabs_ratio*(1.66666666666667*abs_δ*dabs_δ/λ2^2 + 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2^3 + 15.0*exp(-2.23606797749979*abs_δ/λ2)*dabs_ratio*(-5.0*abs_δ*dabs_δ/λ2^3 - 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2^2)/λ2^2 + 33.5410196624969*exp(-2.23606797749979*abs_δ/λ2)*abs_δ*dabs_ratio*(1.66666666666667*abs_δ*dabs_δ/λ2^2 + 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2^4 + 33.5410196624969*exp(-2.23606797749979*abs_δ/λ2)*dabs_δ^3*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)*dabs_ratio/λ2^4 - 11.180339887499*exp(-2.23606797749979*abs_δ/λ2)*dabs_δ^3*(-1.66666666666667*abs_δ^2/λ2^3 - 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2^2)*dabs_ratio/λ2^3 + 6.70820393249937*exp(-2.23606797749979*abs_δ/λ2)*(3.33333333333333*λ2^(-2))*dabs_δ*dabs_ratio/λ2^2 - 6.70820393249937*exp(-2.23606797749979*abs_δ/λ2)*dabs_δ*(-6.66666666666667*λ2^(-3))*dabs_ratio/λ2 - 25.0*exp(-2.23606797749979*abs_δ/λ2)*abs_δ*dabs_δ^3*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)*dabs_ratio/λ2^5 - 15.0*exp(-2.23606797749979*abs_δ/λ2)*abs_δ*(3.33333333333333*λ2^(-2))*dabs_δ*dabs_ratio/λ2^3
    end

    if dorder==[2, 0, 1, 1]
        func = exp(-2.23606797749979*abs_δ/λ2)*(-6.66666666666667*λ2^(-3))*dabs_ratio - 10.0*exp(-2.23606797749979*abs_δ/λ2)*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)*dabs_ratio/λ2^3 + 5.0*exp(-2.23606797749979*abs_δ/λ2)*(-1.66666666666667*abs_δ^2/λ2^3 - 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2^2)*dabs_ratio/λ2^2 + 11.180339887499*exp(-2.23606797749979*abs_δ/λ2)*abs_δ*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)*dabs_ratio/λ2^4 + 2.23606797749979*exp(-2.23606797749979*abs_δ/λ2)*abs_δ*(3.33333333333333*λ2^(-2))*dabs_ratio/λ2^2 + 4.47213595499958*exp(-2.23606797749979*abs_δ/λ2)*dabs_δ*dabs_ratio*(1.66666666666667*abs_δ*dabs_δ/λ2^2 + 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2^2 - 4.47213595499958*exp(-2.23606797749979*abs_δ/λ2)*dabs_δ*dabs_ratio*(-5.0*abs_δ*dabs_δ/λ2^3 - 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2^2)/λ2 - 10.0*exp(-2.23606797749979*abs_δ/λ2)*abs_δ*dabs_δ*dabs_ratio*(1.66666666666667*abs_δ*dabs_δ/λ2^2 + 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2^3
    end

    if dorder==[1, 0, 1, 1]
        func = exp(-2.23606797749979*abs_δ/λ2)*dabs_ratio*(-5.0*abs_δ*dabs_δ/λ2^3 - 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2^2) + 2.23606797749979*exp(-2.23606797749979*abs_δ/λ2)*abs_δ*dabs_ratio*(1.66666666666667*abs_δ*dabs_δ/λ2^2 + 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2^2 + 2.23606797749979*exp(-2.23606797749979*abs_δ/λ2)*dabs_δ*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)*dabs_ratio/λ2^2 - 2.23606797749979*exp(-2.23606797749979*abs_δ/λ2)*dabs_δ*(-1.66666666666667*abs_δ^2/λ2^3 - 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2^2)*dabs_ratio/λ2 - 5.0*exp(-2.23606797749979*abs_δ/λ2)*abs_δ*dabs_δ*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)*dabs_ratio/λ2^3
    end

    if dorder==[0, 0, 1, 1]
        func = exp(-2.23606797749979*abs_δ/λ2)*(-1.66666666666667*abs_δ^2/λ2^3 - 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2^2)*dabs_ratio + 2.23606797749979*exp(-2.23606797749979*abs_δ/λ2)*abs_δ*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)*dabs_ratio/λ2^2
    end

    if dorder==[4, 1, 0, 1]
        func = 0
    end

    if dorder==[3, 1, 0, 1]
        func = 0
    end

    if dorder==[2, 1, 0, 1]
        func = 0
    end

    if dorder==[1, 1, 0, 1]
        func = 0
    end

    if dorder==[0, 1, 0, 1]
        func = 0
    end

    if dorder==[4, 0, 0, 1]
        func = 25.0*exp(-2.23606797749979*abs_δ/λ2)*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)*dabs_ratio/λ2^4 + 30.0*exp(-2.23606797749979*abs_δ/λ2)*(3.33333333333333*λ2^(-2))*dabs_ratio/λ2^2 - 44.7213595499958*exp(-2.23606797749979*abs_δ/λ2)*dabs_δ^3*dabs_ratio*(1.66666666666667*abs_δ*dabs_δ/λ2^2 + 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2^3
    end

    if dorder==[3, 0, 0, 1]
        func = 15.0*exp(-2.23606797749979*abs_δ/λ2)*dabs_ratio*(1.66666666666667*abs_δ*dabs_δ/λ2^2 + 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2^2 - 11.180339887499*exp(-2.23606797749979*abs_δ/λ2)*dabs_δ^3*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)*dabs_ratio/λ2^3 - 6.70820393249937*exp(-2.23606797749979*abs_δ/λ2)*(3.33333333333333*λ2^(-2))*dabs_δ*dabs_ratio/λ2
    end

    if dorder==[2, 0, 0, 1]
        func = exp(-2.23606797749979*abs_δ/λ2)*(3.33333333333333*λ2^(-2))*dabs_ratio + 5.0*exp(-2.23606797749979*abs_δ/λ2)*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)*dabs_ratio/λ2^2 - 4.47213595499958*exp(-2.23606797749979*abs_δ/λ2)*dabs_δ*dabs_ratio*(1.66666666666667*abs_δ*dabs_δ/λ2^2 + 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2
    end

    if dorder==[1, 0, 0, 1]
        func = exp(-2.23606797749979*abs_δ/λ2)*dabs_ratio*(1.66666666666667*abs_δ*dabs_δ/λ2^2 + 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2) - 2.23606797749979*exp(-2.23606797749979*abs_δ/λ2)*dabs_δ*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)*dabs_ratio/λ2
    end

    if dorder==[0, 0, 0, 1]
        func = exp(-2.23606797749979*abs_δ/λ2)*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)*dabs_ratio
    end

    if dorder==[4, 0, 2, 0]
        func = 500.0*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2^6 - 200.0*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*(-1.66666666666667*abs_δ^2/λ2^3 - 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2^2)/λ2^5 + 180.0*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*(3.33333333333333*λ2^(-2))/λ2^4 + 25.0*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*(6.66666666666667*abs_δ^2/λ2^4 + 4.47213595499958*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2^3)/λ2^4 - 120.0*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*(-6.66666666666667*λ2^(-3))/λ2^3 + 30.0*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*(20.0*λ2^(-4))/λ2^2 + 125.0*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*abs_δ^2*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2^8 - 559.016994374948*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*abs_δ*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2^7 + 111.80339887499*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*abs_δ*(-1.66666666666667*abs_δ^2/λ2^3 - 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2^2)/λ2^6 + 150.0*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*abs_δ^2*(3.33333333333333*λ2^(-2))/λ2^6 - 402.492235949962*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*abs_δ*(3.33333333333333*λ2^(-2))/λ2^5 - 536.65631459995*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*dabs_δ^3*(1.66666666666667*abs_δ*dabs_δ/λ2^2 + 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2^5 + 134.164078649987*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*abs_δ*(-6.66666666666667*λ2^(-3))/λ2^4 + 268.328157299975*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*dabs_δ^3*(-5.0*abs_δ*dabs_δ/λ2^3 - 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2^2)/λ2^4 - 44.7213595499958*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*dabs_δ^3*(16.6666666666667*abs_δ*dabs_δ/λ2^4 + 4.47213595499958*dabs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2^3)/λ2^3 - 223.606797749979*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*abs_δ^2*dabs_δ^3*(1.66666666666667*abs_δ*dabs_δ/λ2^2 + 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2^7 + 800.0*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*abs_δ*dabs_δ^3*(1.66666666666667*abs_δ*dabs_δ/λ2^2 + 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2^6 - 200.0*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*abs_δ*dabs_δ^3*(-5.0*abs_δ*dabs_δ/λ2^3 - 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2^2)/λ2^5
    end

    if dorder==[3, 0, 2, 0]
        func = 90.0*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*(1.66666666666667*abs_δ*dabs_δ/λ2^2 + 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2^4 - 60.0*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*(-5.0*abs_δ*dabs_δ/λ2^3 - 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2^2)/λ2^3 + 15.0*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*(16.6666666666667*abs_δ*dabs_δ/λ2^4 + 4.47213595499958*dabs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2^3)/λ2^2 + 75.0*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*abs_δ^2*(1.66666666666667*abs_δ*dabs_δ/λ2^2 + 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2^6 - 201.246117974981*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*abs_δ*(1.66666666666667*abs_δ*dabs_δ/λ2^2 + 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2^5 - 134.164078649987*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*dabs_δ^3*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2^5 + 67.0820393249937*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*abs_δ*(-5.0*abs_δ*dabs_δ/λ2^3 - 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2^2)/λ2^4 + 67.0820393249937*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*dabs_δ^3*(-1.66666666666667*abs_δ^2/λ2^3 - 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2^2)/λ2^4 - 11.180339887499*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*dabs_δ^3*(6.66666666666667*abs_δ^2/λ2^4 + 4.47213595499958*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2^3)/λ2^3 - 13.4164078649987*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*(3.33333333333333*λ2^(-2))*dabs_δ/λ2^3 + 13.4164078649987*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*dabs_δ*(-6.66666666666667*λ2^(-3))/λ2^2 - 6.70820393249937*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*(20.0*λ2^(-4))*dabs_δ/λ2 - 55.9016994374948*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*abs_δ^2*dabs_δ^3*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2^7 + 200.0*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*abs_δ*dabs_δ^3*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2^6 - 50.0*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*abs_δ*dabs_δ^3*(-1.66666666666667*abs_δ^2/λ2^3 - 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2^2)/λ2^5 - 33.5410196624969*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*abs_δ^2*(3.33333333333333*λ2^(-2))*dabs_δ/λ2^5 + 60.0*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*abs_δ*(3.33333333333333*λ2^(-2))*dabs_δ/λ2^4 - 30.0*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*abs_δ*dabs_δ*(-6.66666666666667*λ2^(-3))/λ2^3
    end

    if dorder==[2, 0, 2, 0]
        func = exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*(20.0*λ2^(-4)) + 30.0*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2^4 - 20.0*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*(-1.66666666666667*abs_δ^2/λ2^3 - 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2^2)/λ2^3 + 5.0*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*(6.66666666666667*abs_δ^2/λ2^4 + 4.47213595499958*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2^3)/λ2^2 + 25.0*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*abs_δ^2*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2^6 - 67.0820393249937*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*abs_δ*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2^5 + 22.3606797749979*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*abs_δ*(-1.66666666666667*abs_δ^2/λ2^3 - 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2^2)/λ2^4 + 5.0*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*abs_δ^2*(3.33333333333333*λ2^(-2))/λ2^4 - 4.47213595499958*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*abs_δ*(3.33333333333333*λ2^(-2))/λ2^3 - 8.94427190999916*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*dabs_δ*(1.66666666666667*abs_δ*dabs_δ/λ2^2 + 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2^3 + 4.47213595499958*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*abs_δ*(-6.66666666666667*λ2^(-3))/λ2^2 + 8.94427190999916*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*dabs_δ*(-5.0*abs_δ*dabs_δ/λ2^3 - 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2^2)/λ2^2 - 4.47213595499958*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*dabs_δ*(16.6666666666667*abs_δ*dabs_δ/λ2^4 + 4.47213595499958*dabs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2^3)/λ2 - 22.3606797749979*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*abs_δ^2*dabs_δ*(1.66666666666667*abs_δ*dabs_δ/λ2^2 + 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2^5 + 40.0*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*abs_δ*dabs_δ*(1.66666666666667*abs_δ*dabs_δ/λ2^2 + 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2^4 - 20.0*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*abs_δ*dabs_δ*(-5.0*abs_δ*dabs_δ/λ2^3 - 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2^2)/λ2^3
    end

    if dorder==[1, 0, 2, 0]
        func = exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*(16.6666666666667*abs_δ*dabs_δ/λ2^4 + 4.47213595499958*dabs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2^3) + 5.0*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*abs_δ^2*(1.66666666666667*abs_δ*dabs_δ/λ2^2 + 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2^4 - 4.47213595499958*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*abs_δ*(1.66666666666667*abs_δ*dabs_δ/λ2^2 + 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2^3 - 4.47213595499958*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*dabs_δ*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2^3 + 4.47213595499958*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*abs_δ*(-5.0*abs_δ*dabs_δ/λ2^3 - 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2^2)/λ2^2 + 4.47213595499958*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*dabs_δ*(-1.66666666666667*abs_δ^2/λ2^3 - 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2^2)/λ2^2 - 2.23606797749979*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*dabs_δ*(6.66666666666667*abs_δ^2/λ2^4 + 4.47213595499958*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2^3)/λ2 - 11.180339887499*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*abs_δ^2*dabs_δ*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2^5 + 20.0*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*abs_δ*dabs_δ*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2^4 - 10.0*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*abs_δ*dabs_δ*(-1.66666666666667*abs_δ^2/λ2^3 - 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2^2)/λ2^3
    end

    if dorder==[0, 0, 2, 0]
        func = exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*(6.66666666666667*abs_δ^2/λ2^4 + 4.47213595499958*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2^3) + 5.0*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*abs_δ^2*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2^4 - 4.47213595499958*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*abs_δ*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2^3 + 4.47213595499958*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*abs_δ*(-1.66666666666667*abs_δ^2/λ2^3 - 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2^2)/λ2^2
    end

    if dorder==[4, 1, 1, 0]
        func = 0
    end

    if dorder==[3, 1, 1, 0]
        func = 0
    end

    if dorder==[2, 1, 1, 0]
        func = 0
    end

    if dorder==[1, 1, 1, 0]
        func = 0
    end

    if dorder==[0, 1, 1, 0]
        func = 0
    end

    if dorder==[4, 0, 1, 0]
        func = -100.0*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2^5 + 25.0*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*(-1.66666666666667*abs_δ^2/λ2^3 - 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2^2)/λ2^4 - 60.0*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*(3.33333333333333*λ2^(-2))/λ2^3 + 30.0*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*(-6.66666666666667*λ2^(-3))/λ2^2 + 55.9016994374948*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*abs_δ*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2^6 + 67.0820393249937*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*abs_δ*(3.33333333333333*λ2^(-2))/λ2^4 + 134.164078649987*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*dabs_δ^3*(1.66666666666667*abs_δ*dabs_δ/λ2^2 + 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2^4 - 44.7213595499958*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*dabs_δ^3*(-5.0*abs_δ*dabs_δ/λ2^3 - 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2^2)/λ2^3 - 100.0*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*abs_δ*dabs_δ^3*(1.66666666666667*abs_δ*dabs_δ/λ2^2 + 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2^5
    end

    if dorder==[3, 0, 1, 0]
        func = -30.0*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*(1.66666666666667*abs_δ*dabs_δ/λ2^2 + 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2^3 + 15.0*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*(-5.0*abs_δ*dabs_δ/λ2^3 - 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2^2)/λ2^2 + 33.5410196624969*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*abs_δ*(1.66666666666667*abs_δ*dabs_δ/λ2^2 + 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2^4 + 33.5410196624969*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*dabs_δ^3*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2^4 - 11.180339887499*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*dabs_δ^3*(-1.66666666666667*abs_δ^2/λ2^3 - 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2^2)/λ2^3 + 6.70820393249937*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*(3.33333333333333*λ2^(-2))*dabs_δ/λ2^2 - 6.70820393249937*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*dabs_δ*(-6.66666666666667*λ2^(-3))/λ2 - 25.0*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*abs_δ*dabs_δ^3*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2^5 - 15.0*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*abs_δ*(3.33333333333333*λ2^(-2))*dabs_δ/λ2^3
    end

    if dorder==[2, 0, 1, 0]
        func = exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*(-6.66666666666667*λ2^(-3)) - 10.0*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2^3 + 5.0*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*(-1.66666666666667*abs_δ^2/λ2^3 - 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2^2)/λ2^2 + 11.180339887499*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*abs_δ*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2^4 + 2.23606797749979*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*abs_δ*(3.33333333333333*λ2^(-2))/λ2^2 + 4.47213595499958*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*dabs_δ*(1.66666666666667*abs_δ*dabs_δ/λ2^2 + 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2^2 - 4.47213595499958*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*dabs_δ*(-5.0*abs_δ*dabs_δ/λ2^3 - 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2^2)/λ2 - 10.0*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*abs_δ*dabs_δ*(1.66666666666667*abs_δ*dabs_δ/λ2^2 + 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2^3
    end

    if dorder==[1, 0, 1, 0]
        func = exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*(-5.0*abs_δ*dabs_δ/λ2^3 - 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2^2) + 2.23606797749979*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*abs_δ*(1.66666666666667*abs_δ*dabs_δ/λ2^2 + 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2^2 + 2.23606797749979*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*dabs_δ*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2^2 - 2.23606797749979*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*dabs_δ*(-1.66666666666667*abs_δ^2/λ2^3 - 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2^2)/λ2 - 5.0*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*abs_δ*dabs_δ*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2^3
    end

    if dorder==[0, 0, 1, 0]
        func = exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*(-1.66666666666667*abs_δ^2/λ2^3 - 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2^2) + 2.23606797749979*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*abs_δ*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2^2
    end

    if dorder==[4, 2, 0, 0]
        func = 500.0*exp(-2.23606797749979*abs_δ/λ1)*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1)/λ1^6 - 200.0*exp(-2.23606797749979*abs_δ/λ1)*(-1.66666666666667*abs_δ^2/λ1^3 - 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1^2)/λ1^5 + 180.0*exp(-2.23606797749979*abs_δ/λ1)*(3.33333333333333*λ1^(-2))/λ1^4 + 25.0*exp(-2.23606797749979*abs_δ/λ1)*(6.66666666666667*abs_δ^2/λ1^4 + 4.47213595499958*abs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1^3)/λ1^4 - 120.0*exp(-2.23606797749979*abs_δ/λ1)*(-6.66666666666667*λ1^(-3))/λ1^3 + 30.0*exp(-2.23606797749979*abs_δ/λ1)*(20.0*λ1^(-4))/λ1^2 + 125.0*exp(-2.23606797749979*abs_δ/λ1)*abs_δ^2*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1)/λ1^8 - 559.016994374948*exp(-2.23606797749979*abs_δ/λ1)*abs_δ*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1)/λ1^7 + 111.80339887499*exp(-2.23606797749979*abs_δ/λ1)*abs_δ*(-1.66666666666667*abs_δ^2/λ1^3 - 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1^2)/λ1^6 + 150.0*exp(-2.23606797749979*abs_δ/λ1)*abs_δ^2*(3.33333333333333*λ1^(-2))/λ1^6 - 402.492235949962*exp(-2.23606797749979*abs_δ/λ1)*abs_δ*(3.33333333333333*λ1^(-2))/λ1^5 - 536.65631459995*exp(-2.23606797749979*abs_δ/λ1)*dabs_δ^3*(1.66666666666667*abs_δ*dabs_δ/λ1^2 + 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1)/λ1^5 + 134.164078649987*exp(-2.23606797749979*abs_δ/λ1)*abs_δ*(-6.66666666666667*λ1^(-3))/λ1^4 + 268.328157299975*exp(-2.23606797749979*abs_δ/λ1)*dabs_δ^3*(-5.0*abs_δ*dabs_δ/λ1^3 - 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1^2)/λ1^4 - 44.7213595499958*exp(-2.23606797749979*abs_δ/λ1)*dabs_δ^3*(16.6666666666667*abs_δ*dabs_δ/λ1^4 + 4.47213595499958*dabs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1^3)/λ1^3 - 223.606797749979*exp(-2.23606797749979*abs_δ/λ1)*abs_δ^2*dabs_δ^3*(1.66666666666667*abs_δ*dabs_δ/λ1^2 + 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1)/λ1^7 + 800.0*exp(-2.23606797749979*abs_δ/λ1)*abs_δ*dabs_δ^3*(1.66666666666667*abs_δ*dabs_δ/λ1^2 + 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1)/λ1^6 - 200.0*exp(-2.23606797749979*abs_δ/λ1)*abs_δ*dabs_δ^3*(-5.0*abs_δ*dabs_δ/λ1^3 - 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1^2)/λ1^5
    end

    if dorder==[3, 2, 0, 0]
        func = 90.0*exp(-2.23606797749979*abs_δ/λ1)*(1.66666666666667*abs_δ*dabs_δ/λ1^2 + 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1)/λ1^4 - 60.0*exp(-2.23606797749979*abs_δ/λ1)*(-5.0*abs_δ*dabs_δ/λ1^3 - 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1^2)/λ1^3 + 15.0*exp(-2.23606797749979*abs_δ/λ1)*(16.6666666666667*abs_δ*dabs_δ/λ1^4 + 4.47213595499958*dabs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1^3)/λ1^2 + 75.0*exp(-2.23606797749979*abs_δ/λ1)*abs_δ^2*(1.66666666666667*abs_δ*dabs_δ/λ1^2 + 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1)/λ1^6 - 201.246117974981*exp(-2.23606797749979*abs_δ/λ1)*abs_δ*(1.66666666666667*abs_δ*dabs_δ/λ1^2 + 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1)/λ1^5 - 134.164078649987*exp(-2.23606797749979*abs_δ/λ1)*dabs_δ^3*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1)/λ1^5 + 67.0820393249937*exp(-2.23606797749979*abs_δ/λ1)*abs_δ*(-5.0*abs_δ*dabs_δ/λ1^3 - 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1^2)/λ1^4 + 67.0820393249937*exp(-2.23606797749979*abs_δ/λ1)*dabs_δ^3*(-1.66666666666667*abs_δ^2/λ1^3 - 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1^2)/λ1^4 - 11.180339887499*exp(-2.23606797749979*abs_δ/λ1)*dabs_δ^3*(6.66666666666667*abs_δ^2/λ1^4 + 4.47213595499958*abs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1^3)/λ1^3 - 13.4164078649987*exp(-2.23606797749979*abs_δ/λ1)*(3.33333333333333*λ1^(-2))*dabs_δ/λ1^3 + 13.4164078649987*exp(-2.23606797749979*abs_δ/λ1)*dabs_δ*(-6.66666666666667*λ1^(-3))/λ1^2 - 6.70820393249937*exp(-2.23606797749979*abs_δ/λ1)*(20.0*λ1^(-4))*dabs_δ/λ1 - 55.9016994374948*exp(-2.23606797749979*abs_δ/λ1)*abs_δ^2*dabs_δ^3*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1)/λ1^7 + 200.0*exp(-2.23606797749979*abs_δ/λ1)*abs_δ*dabs_δ^3*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1)/λ1^6 - 50.0*exp(-2.23606797749979*abs_δ/λ1)*abs_δ*dabs_δ^3*(-1.66666666666667*abs_δ^2/λ1^3 - 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1^2)/λ1^5 - 33.5410196624969*exp(-2.23606797749979*abs_δ/λ1)*abs_δ^2*(3.33333333333333*λ1^(-2))*dabs_δ/λ1^5 + 60.0*exp(-2.23606797749979*abs_δ/λ1)*abs_δ*(3.33333333333333*λ1^(-2))*dabs_δ/λ1^4 - 30.0*exp(-2.23606797749979*abs_δ/λ1)*abs_δ*dabs_δ*(-6.66666666666667*λ1^(-3))/λ1^3
    end

    if dorder==[2, 2, 0, 0]
        func = exp(-2.23606797749979*abs_δ/λ1)*(20.0*λ1^(-4)) + 30.0*exp(-2.23606797749979*abs_δ/λ1)*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1)/λ1^4 - 20.0*exp(-2.23606797749979*abs_δ/λ1)*(-1.66666666666667*abs_δ^2/λ1^3 - 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1^2)/λ1^3 + 5.0*exp(-2.23606797749979*abs_δ/λ1)*(6.66666666666667*abs_δ^2/λ1^4 + 4.47213595499958*abs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1^3)/λ1^2 + 25.0*exp(-2.23606797749979*abs_δ/λ1)*abs_δ^2*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1)/λ1^6 - 67.0820393249937*exp(-2.23606797749979*abs_δ/λ1)*abs_δ*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1)/λ1^5 + 22.3606797749979*exp(-2.23606797749979*abs_δ/λ1)*abs_δ*(-1.66666666666667*abs_δ^2/λ1^3 - 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1^2)/λ1^4 + 5.0*exp(-2.23606797749979*abs_δ/λ1)*abs_δ^2*(3.33333333333333*λ1^(-2))/λ1^4 - 4.47213595499958*exp(-2.23606797749979*abs_δ/λ1)*abs_δ*(3.33333333333333*λ1^(-2))/λ1^3 - 8.94427190999916*exp(-2.23606797749979*abs_δ/λ1)*dabs_δ*(1.66666666666667*abs_δ*dabs_δ/λ1^2 + 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1)/λ1^3 + 4.47213595499958*exp(-2.23606797749979*abs_δ/λ1)*abs_δ*(-6.66666666666667*λ1^(-3))/λ1^2 + 8.94427190999916*exp(-2.23606797749979*abs_δ/λ1)*dabs_δ*(-5.0*abs_δ*dabs_δ/λ1^3 - 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1^2)/λ1^2 - 4.47213595499958*exp(-2.23606797749979*abs_δ/λ1)*dabs_δ*(16.6666666666667*abs_δ*dabs_δ/λ1^4 + 4.47213595499958*dabs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1^3)/λ1 - 22.3606797749979*exp(-2.23606797749979*abs_δ/λ1)*abs_δ^2*dabs_δ*(1.66666666666667*abs_δ*dabs_δ/λ1^2 + 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1)/λ1^5 + 40.0*exp(-2.23606797749979*abs_δ/λ1)*abs_δ*dabs_δ*(1.66666666666667*abs_δ*dabs_δ/λ1^2 + 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1)/λ1^4 - 20.0*exp(-2.23606797749979*abs_δ/λ1)*abs_δ*dabs_δ*(-5.0*abs_δ*dabs_δ/λ1^3 - 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1^2)/λ1^3
    end

    if dorder==[1, 2, 0, 0]
        func = exp(-2.23606797749979*abs_δ/λ1)*(16.6666666666667*abs_δ*dabs_δ/λ1^4 + 4.47213595499958*dabs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1^3) + 5.0*exp(-2.23606797749979*abs_δ/λ1)*abs_δ^2*(1.66666666666667*abs_δ*dabs_δ/λ1^2 + 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1)/λ1^4 - 4.47213595499958*exp(-2.23606797749979*abs_δ/λ1)*abs_δ*(1.66666666666667*abs_δ*dabs_δ/λ1^2 + 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1)/λ1^3 - 4.47213595499958*exp(-2.23606797749979*abs_δ/λ1)*dabs_δ*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1)/λ1^3 + 4.47213595499958*exp(-2.23606797749979*abs_δ/λ1)*abs_δ*(-5.0*abs_δ*dabs_δ/λ1^3 - 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1^2)/λ1^2 + 4.47213595499958*exp(-2.23606797749979*abs_δ/λ1)*dabs_δ*(-1.66666666666667*abs_δ^2/λ1^3 - 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1^2)/λ1^2 - 2.23606797749979*exp(-2.23606797749979*abs_δ/λ1)*dabs_δ*(6.66666666666667*abs_δ^2/λ1^4 + 4.47213595499958*abs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1^3)/λ1 - 11.180339887499*exp(-2.23606797749979*abs_δ/λ1)*abs_δ^2*dabs_δ*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1)/λ1^5 + 20.0*exp(-2.23606797749979*abs_δ/λ1)*abs_δ*dabs_δ*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1)/λ1^4 - 10.0*exp(-2.23606797749979*abs_δ/λ1)*abs_δ*dabs_δ*(-1.66666666666667*abs_δ^2/λ1^3 - 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1^2)/λ1^3
    end

    if dorder==[0, 2, 0, 0]
        func = exp(-2.23606797749979*abs_δ/λ1)*(6.66666666666667*abs_δ^2/λ1^4 + 4.47213595499958*abs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1^3) + 5.0*exp(-2.23606797749979*abs_δ/λ1)*abs_δ^2*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1)/λ1^4 - 4.47213595499958*exp(-2.23606797749979*abs_δ/λ1)*abs_δ*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1)/λ1^3 + 4.47213595499958*exp(-2.23606797749979*abs_δ/λ1)*abs_δ*(-1.66666666666667*abs_δ^2/λ1^3 - 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1^2)/λ1^2
    end

    if dorder==[4, 1, 0, 0]
        func = -100.0*exp(-2.23606797749979*abs_δ/λ1)*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1)/λ1^5 + 25.0*exp(-2.23606797749979*abs_δ/λ1)*(-1.66666666666667*abs_δ^2/λ1^3 - 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1^2)/λ1^4 - 60.0*exp(-2.23606797749979*abs_δ/λ1)*(3.33333333333333*λ1^(-2))/λ1^3 + 30.0*exp(-2.23606797749979*abs_δ/λ1)*(-6.66666666666667*λ1^(-3))/λ1^2 + 55.9016994374948*exp(-2.23606797749979*abs_δ/λ1)*abs_δ*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1)/λ1^6 + 67.0820393249937*exp(-2.23606797749979*abs_δ/λ1)*abs_δ*(3.33333333333333*λ1^(-2))/λ1^4 + 134.164078649987*exp(-2.23606797749979*abs_δ/λ1)*dabs_δ^3*(1.66666666666667*abs_δ*dabs_δ/λ1^2 + 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1)/λ1^4 - 44.7213595499958*exp(-2.23606797749979*abs_δ/λ1)*dabs_δ^3*(-5.0*abs_δ*dabs_δ/λ1^3 - 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1^2)/λ1^3 - 100.0*exp(-2.23606797749979*abs_δ/λ1)*abs_δ*dabs_δ^3*(1.66666666666667*abs_δ*dabs_δ/λ1^2 + 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1)/λ1^5
    end

    if dorder==[3, 1, 0, 0]
        func = -30.0*exp(-2.23606797749979*abs_δ/λ1)*(1.66666666666667*abs_δ*dabs_δ/λ1^2 + 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1)/λ1^3 + 15.0*exp(-2.23606797749979*abs_δ/λ1)*(-5.0*abs_δ*dabs_δ/λ1^3 - 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1^2)/λ1^2 + 33.5410196624969*exp(-2.23606797749979*abs_δ/λ1)*abs_δ*(1.66666666666667*abs_δ*dabs_δ/λ1^2 + 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1)/λ1^4 + 33.5410196624969*exp(-2.23606797749979*abs_δ/λ1)*dabs_δ^3*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1)/λ1^4 - 11.180339887499*exp(-2.23606797749979*abs_δ/λ1)*dabs_δ^3*(-1.66666666666667*abs_δ^2/λ1^3 - 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1^2)/λ1^3 + 6.70820393249937*exp(-2.23606797749979*abs_δ/λ1)*(3.33333333333333*λ1^(-2))*dabs_δ/λ1^2 - 6.70820393249937*exp(-2.23606797749979*abs_δ/λ1)*dabs_δ*(-6.66666666666667*λ1^(-3))/λ1 - 25.0*exp(-2.23606797749979*abs_δ/λ1)*abs_δ*dabs_δ^3*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1)/λ1^5 - 15.0*exp(-2.23606797749979*abs_δ/λ1)*abs_δ*(3.33333333333333*λ1^(-2))*dabs_δ/λ1^3
    end

    if dorder==[2, 1, 0, 0]
        func = exp(-2.23606797749979*abs_δ/λ1)*(-6.66666666666667*λ1^(-3)) - 10.0*exp(-2.23606797749979*abs_δ/λ1)*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1)/λ1^3 + 5.0*exp(-2.23606797749979*abs_δ/λ1)*(-1.66666666666667*abs_δ^2/λ1^3 - 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1^2)/λ1^2 + 11.180339887499*exp(-2.23606797749979*abs_δ/λ1)*abs_δ*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1)/λ1^4 + 2.23606797749979*exp(-2.23606797749979*abs_δ/λ1)*abs_δ*(3.33333333333333*λ1^(-2))/λ1^2 + 4.47213595499958*exp(-2.23606797749979*abs_δ/λ1)*dabs_δ*(1.66666666666667*abs_δ*dabs_δ/λ1^2 + 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1)/λ1^2 - 4.47213595499958*exp(-2.23606797749979*abs_δ/λ1)*dabs_δ*(-5.0*abs_δ*dabs_δ/λ1^3 - 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1^2)/λ1 - 10.0*exp(-2.23606797749979*abs_δ/λ1)*abs_δ*dabs_δ*(1.66666666666667*abs_δ*dabs_δ/λ1^2 + 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1)/λ1^3
    end

    if dorder==[1, 1, 0, 0]
        func = exp(-2.23606797749979*abs_δ/λ1)*(-5.0*abs_δ*dabs_δ/λ1^3 - 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1^2) + 2.23606797749979*exp(-2.23606797749979*abs_δ/λ1)*abs_δ*(1.66666666666667*abs_δ*dabs_δ/λ1^2 + 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1)/λ1^2 + 2.23606797749979*exp(-2.23606797749979*abs_δ/λ1)*dabs_δ*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1)/λ1^2 - 2.23606797749979*exp(-2.23606797749979*abs_δ/λ1)*dabs_δ*(-1.66666666666667*abs_δ^2/λ1^3 - 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1^2)/λ1 - 5.0*exp(-2.23606797749979*abs_δ/λ1)*abs_δ*dabs_δ*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1)/λ1^3
    end

    if dorder==[0, 1, 0, 0]
        func = exp(-2.23606797749979*abs_δ/λ1)*(-1.66666666666667*abs_δ^2/λ1^3 - 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1^2) + 2.23606797749979*exp(-2.23606797749979*abs_δ/λ1)*abs_δ*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1)/λ1^2
    end

    if dorder==[4, 0, 0, 0]
        func = 25.0*exp(-2.23606797749979*abs_δ/λ1)*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1)/λ1^4 + 30.0*exp(-2.23606797749979*abs_δ/λ1)*(3.33333333333333*λ1^(-2))/λ1^2 - 44.7213595499958*exp(-2.23606797749979*abs_δ/λ1)*dabs_δ^3*(1.66666666666667*abs_δ*dabs_δ/λ1^2 + 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1)/λ1^3 + 25.0*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2^4 + 30.0*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*(3.33333333333333*λ2^(-2))/λ2^2 - 44.7213595499958*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*dabs_δ^3*(1.66666666666667*abs_δ*dabs_δ/λ2^2 + 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2^3
    end

    if dorder==[3, 0, 0, 0]
        func = 15.0*exp(-2.23606797749979*abs_δ/λ1)*(1.66666666666667*abs_δ*dabs_δ/λ1^2 + 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1)/λ1^2 - 11.180339887499*exp(-2.23606797749979*abs_δ/λ1)*dabs_δ^3*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1)/λ1^3 - 6.70820393249937*exp(-2.23606797749979*abs_δ/λ1)*(3.33333333333333*λ1^(-2))*dabs_δ/λ1 + 15.0*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*(1.66666666666667*abs_δ*dabs_δ/λ2^2 + 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2^2 - 11.180339887499*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*dabs_δ^3*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2^3 - 6.70820393249937*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*(3.33333333333333*λ2^(-2))*dabs_δ/λ2
    end

    if dorder==[2, 0, 0, 0]
        func = exp(-2.23606797749979*abs_δ/λ1)*(3.33333333333333*λ1^(-2)) + 5.0*exp(-2.23606797749979*abs_δ/λ1)*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1)/λ1^2 + exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*(3.33333333333333*λ2^(-2)) - 4.47213595499958*exp(-2.23606797749979*abs_δ/λ1)*dabs_δ*(1.66666666666667*abs_δ*dabs_δ/λ1^2 + 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1)/λ1 + 5.0*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2^2 - 4.47213595499958*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*dabs_δ*(1.66666666666667*abs_δ*dabs_δ/λ2^2 + 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2
    end

    if dorder==[1, 0, 0, 0]
        func = exp(-2.23606797749979*abs_δ/λ1)*(1.66666666666667*abs_δ*dabs_δ/λ1^2 + 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1) + exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*(1.66666666666667*abs_δ*dabs_δ/λ2^2 + 2.23606797749979*dabs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2) - 2.23606797749979*exp(-2.23606797749979*abs_δ/λ1)*dabs_δ*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1)/λ1 - 2.23606797749979*exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*dabs_δ*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)/λ2
    end

    if dorder==[0, 0, 0, 0]
        func = exp(-2.23606797749979*abs_δ/λ1)*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ1)/λ1) + exp(-2.23606797749979*abs_δ/λ2)*abs_ratio*(1 + 2.23606797749979*abs_δ*(1 + 0.74535599249993*abs_δ/λ2)/λ2)
    end

    return even_time_derivative * float(func)  # correcting for amount of t2 derivatives

end


return m52x2_kernel, 3  # the function handle and the number of kernel hyperparameters
