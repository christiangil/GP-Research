# these are all general purpose functions that aren't specifically related to
# radial velocity or GP calculations
using LinearAlgebra
using IterativeSolvers

"if needed, adds a ridge based on the smallest eignevalue to make a Cholesky factorization possible"
function ridge_chol(A::Union{Matrix{T},Symmetric{T,Matrix{T}}}) where {T<:Real}

    # only add a small ridge (based on the smallest eigenvalue) if necessary
    try
        return cholesky(A)
    catch
        smallest_eigen = IterativeSolvers.lobpcg(A, false, 1).λ[1]
        ridge = 1.10 * abs(smallest_eigen)
        @warn "added a ridge"
        println("ridge size:          10^$(log10(ridge))")
        println("max value of array:  10^$(log10(maximum(abs.(A))))")
        return cholesky(A + UniformScaling(ridge))
    end

end

"dont do anything if an array that is already factorized is passed"
ridge_chol(A::Cholesky{T,Matrix{T}}) where {T<:Real} = A

"""
Solve a linear system of equations (optionally with variance values at each point or covariance array)
see (https://en.wikipedia.org/wiki/Generalized_least_squares#Method_outline)
"""
function general_lst_sq(
    dm::Matrix{T},
    data::Vector;
    Σ::Union{Cholesky{T,Matrix{T}},Symmetric{T,Matrix{T}},Matrix{T},Vector{T}}=ones(1),
    return_ϵ_inv::Bool=false) where {T<:Real}
    @assert ndims(Σ) < 3 "the Σ variable needs to be a 1D or 2D array"

    # if Σ == ones(1)
    #     return dm \ data
    # else
    if ndims(Σ) == 1
        Σ = Diagonal(Σ)
    else
        Σ = ridge_chol(Σ)
    end
    if return_ϵ_inv
        ϵ_int = Σ \ dm
        ϵ_inv = dm' * ϵ_int
        return ϵ_inv \ (dm' * (Σ \ data)), ϵ_int, ϵ_inv
    else
        return (dm' * (Σ \ dm)) \ (dm' * (Σ \ data))
    end
    # end
end

import Base.ndims
ndims(A::Cholesky{T,Matrix{T}}) where {T<:Real} = 2
