include("../all_functions.jl")

# #define a kernel like so:
#
# "Radial basis function GP kernel (aka squared exonential, ~gaussian)"
# function rbf_kernel_base(hyperparameters, dif)
#
#     dif_sq = dif ^ 2
#
#     hyperparameters = check_hyperparameters(hyperparameters, 1+1)
#     kernel_length = hyperparameters[1]
#
#     return exp(-dif_sq / (2 * (kernel_length ^ 2)))
# end
#
# # calculate the necessary derivative versions like so:
#
# @vars t1 t2 kernel_length
# symbolic_kernel = rbf_kernel_base([kernel_length], t1 - t2)
# kernel_name = "rbf_kernel"
# kernel_coder(symbolic_kernel, kernel_name)
#
# # the function is saved in kernels/\$kernel_name.jl, so you can use it with a command akin to this:
#
# include("kernels/" * kernel_name * ".jl")


#########################################################

# process for se_kernel_base

@vars dif kernel_length
kernel_coder(se_kernel_base(kernel_length, dif), "se_kernel")

# process for periodic_kernel_base

@vars dif kernel_period kernel_length
kernel_coder(periodic_kernel_base([kernel_period, kernel_length], dif), "periodic_kernel")

# process for quasi_periodic_kernel_base

@vars dif SE_kernel_length P_kernel_period P_kernel_length
kernel_coder(quasi_periodic_kernel_base([SE_kernel_length, P_kernel_period, P_kernel_length], dif), "quasi_periodic_kernel")

#process for matern52_kernel_base

@vars abs_dif kernel_length
kernel_coder(matern52_kernel_base(kernel_length, abs_dif), "matern52_kernel")

#process for rq_kernel_base

@vars dif kernel_length alpha
kernel_coder(rq_kernel_base([kernel_length, alpha], dif), "rq_kernel")
