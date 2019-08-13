

"""
se_kernel function created by kernel_coder(). Requires 1 hyperparameters. Likely created using se_kernel_base() as an input. 
Use with include("src/kernels/se_kernel.jl").
hyperparameters == ["λ"]
"""
function se_kernel(
    hyperparameters::Vector{<:Real}, 
    δ::Real; 
    dorder::Vector{<:Integer}=zeros(Int64, length(hyperparameters) + 2))

    @assert length(hyperparameters)==1 "hyperparameters is the wrong length"
    @assert length(dorder)==(1 + 2) "dorder is the wrong length"
    even_time_derivative = powers_of_negative_one(dorder[2])
    @assert maximum(dorder) < 3 "No more than two time derivatives for either t1 or t2 can be calculated"

    dorder[1] = sum(dorder[1:2])
    dorder[2:(end - 1)] = dorder[3:end]

    deleteat!(dorder, length(dorder))

    λ = hyperparameters[1]

    if dorder==[4, 2]
        func = 60*exp((-1/2)*δ^2/λ^2)/λ^6 - 285*exp((-1/2)*δ^2/λ^2)*δ^2/λ^8 + 165*exp((-1/2)*δ^2/λ^2)*δ^4/λ^10 - 25*exp((-1/2)*δ^2/λ^2)*δ^6/λ^12 + exp((-1/2)*δ^2/λ^2)*δ^8/λ^14
    end

    if dorder==[3, 2]
        func = 60*exp((-1/2)*δ^2/λ^2)*δ/λ^6 - 75*exp((-1/2)*δ^2/λ^2)*δ^3/λ^8 + 18*exp((-1/2)*δ^2/λ^2)*δ^5/λ^10 - exp((-1/2)*δ^2/λ^2)*δ^7/λ^12
    end

    if dorder==[2, 2]
        func = -6*exp((-1/2)*δ^2/λ^2)/λ^4 + 27*exp((-1/2)*δ^2/λ^2)*δ^2/λ^6 - 12*exp((-1/2)*δ^2/λ^2)*δ^4/λ^8 + exp((-1/2)*δ^2/λ^2)*δ^6/λ^10
    end

    if dorder==[1, 2]
        func = -6*exp((-1/2)*δ^2/λ^2)*δ/λ^4 + 7*exp((-1/2)*δ^2/λ^2)*δ^3/λ^6 - exp((-1/2)*δ^2/λ^2)*δ^5/λ^8
    end

    if dorder==[0, 2]
        func = -3*exp((-1/2)*δ^2/λ^2)*δ^2/λ^4 + exp((-1/2)*δ^2/λ^2)*δ^4/λ^6
    end

    if dorder==[4, 1]
        func = -12*exp((-1/2)*δ^2/λ^2)/λ^5 + 39*exp((-1/2)*δ^2/λ^2)*δ^2/λ^7 - 14*exp((-1/2)*δ^2/λ^2)*δ^4/λ^9 + exp((-1/2)*δ^2/λ^2)*δ^6/λ^11
    end

    if dorder==[3, 1]
        func = -12*exp((-1/2)*δ^2/λ^2)*δ/λ^5 + 9*exp((-1/2)*δ^2/λ^2)*δ^3/λ^7 - exp((-1/2)*δ^2/λ^2)*δ^5/λ^9
    end

    if dorder==[2, 1]
        func = 2*exp((-1/2)*δ^2/λ^2)/λ^3 - 5*exp((-1/2)*δ^2/λ^2)*δ^2/λ^5 + exp((-1/2)*δ^2/λ^2)*δ^4/λ^7
    end

    if dorder==[1, 1]
        func = 2*exp((-1/2)*δ^2/λ^2)*δ/λ^3 - exp((-1/2)*δ^2/λ^2)*δ^3/λ^5
    end

    if dorder==[0, 1]
        func = exp((-1/2)*δ^2/λ^2)*δ^2/λ^3
    end

    if dorder==[4, 0]
        func = 3*exp((-1/2)*δ^2/λ^2)/λ^4 - 6*exp((-1/2)*δ^2/λ^2)*δ^2/λ^6 + exp((-1/2)*δ^2/λ^2)*δ^4/λ^8
    end

    if dorder==[3, 0]
        func = 3*exp((-1/2)*δ^2/λ^2)*δ/λ^4 - exp((-1/2)*δ^2/λ^2)*δ^3/λ^6
    end

    if dorder==[2, 0]
        func = -exp((-1/2)*δ^2/λ^2)/λ^2 + exp((-1/2)*δ^2/λ^2)*δ^2/λ^4
    end

    if dorder==[1, 0]
        func = -exp((-1/2)*δ^2/λ^2)*δ/λ^2
    end

    if dorder==[0, 0]
        func = exp((-1/2)*δ^2/λ^2)
    end

    return even_time_derivative * float(func)  # correcting for amount of t2 derivatives

end


return se_kernel, 1  # the function handle and the number of kernel hyperparameters
