include("../all_functions.jl")

# process for se_kernel_base
@vars δ λ
kernel_coder(se_kernel_base(λ, δ), "se_kernel")

# process for periodic_kernel_base
@vars δ P λ
kernel_coder(periodic_kernel_base([P, λ], δ), "periodic_kernel")
@vars δ λ
kernel_coder(se_kernel_base(λ, δ), "se"; periodic=true)

# process for quasi_periodic_kernel_base
@vars δ SE_λ P_P P_λ
kernel_coder(quasi_periodic_kernel_base([SE_λ, P_P, P_λ], δ), "quasi_periodic_kernel")

#process for matern52_kernel_base

@vars abs_δ λ
kernel_coder(matern52_kernel_base(λ, abs_δ), "matern52_kernel")

#process for rq_kernel_base

# @vars δ k θ
# kernel_coder(rq_kernel_base([k, θ], δ), "rq_kernel")
# @vars δ α β
# kernel_coder(rq_kernel_base([α, β], δ), "rq_kernel")
@vars δ α μ
kernel_coder(rq_kernel_base([α, μ], δ), "rq_kernel")

@vars abs_δ α μ
kernel_coder(rm52_kernel_base([α, μ], abs_δ), "rm52_kernel")

@vars δ, P, λ, α
kernel_coder(periodic_rq_kernel_base([P, λ, α], δ), "periodic_rq_kernel")
