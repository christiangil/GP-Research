

# #define a kernel like so:
#
# "Radial basis function GP kernel (aka squared exonential, ~gaussian)"
# function rbf_kernel_base(hyperparameters, dif)
#
#     dif_sq = dif ^ 2
#
#     hyperparameters = check_hyperparameters(hyperparameters, 1+1)
#     kernel_amplitude, kernel_length = hyperparameters
#
#     return kernel_amplitude ^ 2 * exp(-dif_sq / (2 * (kernel_length ^ 2)))
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

# process for RBF_kernel_base

@vars t1 t2 kernel_length
symbolic_kernel = rbf_kernel_base([kernel_length], t1 - t2)
kernel_name = "rbf_kernel"
kernel_coder(symbolic_kernel, kernel_name)

# process for Periodic_kernel_base

@vars t1 t2 kernel_period kernel_length
symbolic_kernel = periodic_kernel_base([kernel_period, kernel_length], t1 - t2)
kernel_name = "periodic_kernel"
kernel_coder(symbolic_kernel, kernel_name)

# process for Quasi_periodic_kernel_base

@vars t1 t2 RBF_kernel_length P_kernel_period P_kernel_length
symbolic_kernel = quasi_periodic_kernel_base([RBF_kernel_length, P_kernel_period, P_kernel_length], t1 - t2)
kernel_name = "quasi_periodic_kernel"
kernel_coder(symbolic_kernel, kernel_name)

#process for Exp_periodic_kernel_base

@vars t1 t2 OU_kernel_length P_kernel_period P_kernel_length
symbolic_kernel = exp_periodic_kernel_base([OU_kernel_length, P_kernel_period, P_kernel_length], t1 - t2)
kernel_name = "exp_periodic_kernel"
kernel_coder(symbolic_kernel, kernel_name)
