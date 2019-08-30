include("../all_functions.jl")

# process for se_kernel_base
@vars δ λ
kernel_coder(se_kernel_base(λ, δ), "se")

# process for periodic_kernel_base
@vars δ se_P λ
kernel_coder(periodic_kernel_base([se_P, λ], δ), "periodic")
@vars δ λ
kernel_coder(se_kernel_base(λ, δ), "se"; periodic_var="δ")

# process for quasi_periodic_kernel_base
@vars δ se_λ qp_P p_λ
kernel_coder(quasi_periodic_kernel_base([se_λ, qp_P, p_λ], δ), "quasi_periodic")
@vars δ δp se_λ p_λ
kernel_coder(se_kernel_base(se_λ, δ) * se_kernel_base(p_λ, δp), "qp"; periodic_var="δp")

#process for matern52_kernel_base

@vars δ λ
kernel_coder(matern52_kernel_base(λ, δ), "m52_test")

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

@vars abs_δ, λ
kernel_coder(pp_kernel_base(λ, abs_δ), "pp"; cutoff_var="λ")
