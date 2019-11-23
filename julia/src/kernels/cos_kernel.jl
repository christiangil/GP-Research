"""
cos_kernel function created by kernel_coder(). Requires 1 hyperparameters. Likely created using cos_kernel_base() as an input.
Use with include("src/kernels/cos_kernel.jl").
hyperparameters == ["λ"]
"""
function cos_kernel(
    hyperparameters::Vector{<:Real},
    δ::Real;
    dorder::Vector{<:Integer}=zeros(Int64, length(hyperparameters) + 2))

    @assert length(hyperparameters)==1 "hyperparameters is the wrong length"
    @assert length(dorder)==(1 + 2) "dorder is the wrong length"
    even_time_derivative = powers_of_negative_one(dorder[2])
    @assert maximum(dorder) < 3 "No more than two time derivatives for either t1 or t2 can be calculated"

    dorder = append!([sum(dorder[1:2])], dorder[3:end])

    λ = hyperparameters[1]

    if dorder==[4, 2]
        func = 31170.9091308808*cos(6.28318530717959*δ/λ)/λ^6 - 97926.29913129*δ*sin(6.28318530717959*δ/λ)/λ^7 - 61528.9083888195*δ^2*cos(6.28318530717959*δ/λ)/λ^8
    end

    if dorder==[3, 2]
        func = 2976.60256130878*sin(6.28318530717959*δ/λ)/λ^5 + 12468.3636523523*δ*cos(6.28318530717959*δ/λ)/λ^6 - 9792.629913129*δ^2*sin(6.28318530717959*δ/λ)/λ^7
    end

    if dorder==[2, 2]
        func = -236.870505626145*cos(6.28318530717959*δ/λ)/λ^4 + 1488.30128065439*δ*sin(6.28318530717959*δ/λ)/λ^5 + 1558.54545654404*δ^2*cos(6.28318530717959*δ/λ)/λ^6
    end

    if dorder==[1, 2]
        func = -12.5663706143592*sin(6.28318530717959*δ/λ)/λ^3 - 157.91367041743*δ*cos(6.28318530717959*δ/λ)/λ^4 + 248.050213442399*δ^2*sin(6.28318530717959*δ/λ)/λ^5
    end

    if dorder==[0, 2]
        func = -12.5663706143592*δ*sin(6.28318530717959*δ/λ)/λ^3 - 39.4784176043574*δ^2*cos(6.28318530717959*δ/λ)/λ^4
    end

    if dorder==[4, 1]
        func = -6234.18182617615*cos(6.28318530717959*δ/λ)/λ^5 + 9792.629913129*δ*sin(6.28318530717959*δ/λ)/λ^6
    end

    if dorder==[3, 1]
        func = -744.150640327196*sin(6.28318530717959*δ/λ)/λ^4 - 1558.54545654404*δ*cos(6.28318530717959*δ/λ)/λ^5
    end

    if dorder==[2, 1]
        func = 78.9568352087149*cos(6.28318530717959*δ/λ)/λ^3 - 248.050213442399*δ*sin(6.28318530717959*δ/λ)/λ^4
    end

    if dorder==[1, 1]
        func = 6.28318530717959*sin(6.28318530717959*δ/λ)/λ^2 + 39.4784176043574*δ*cos(6.28318530717959*δ/λ)/λ^3
    end

    if dorder==[0, 1]
        func = 6.28318530717959*δ*sin(6.28318530717959*δ/λ)/λ^2
    end

    if dorder==[4, 0]
        func = 1558.54545654404*cos(6.28318530717959*δ/λ)/λ^4
    end

    if dorder==[3, 0]
        func = 248.050213442399*sin(6.28318530717959*δ/λ)/λ^3
    end

    if dorder==[2, 0]
        func = -39.4784176043574*cos(6.28318530717959*δ/λ)/λ^2
    end

    if dorder==[1, 0]
        func = -6.28318530717959*sin(6.28318530717959*δ/λ)/λ
    end

    if dorder==[0, 0]
        func = cos(6.28318530717959*δ/λ)
    end

    return even_time_derivative * float(func)  # correcting for amount of t2 derivatives

end


return cos_kernel, 1  # the function handle and the number of kernel hyperparameters
