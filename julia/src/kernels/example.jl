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

# process for rbf_kernel_base

@vars t1 t2 kernel_length
symbolic_kernel = rbf_kernel_base(kernel_length, t1 - t2)
kernel_name = "rbf_kernel"
kernel_coder(symbolic_kernel, kernel_name)

# process for periodic_kernel_base

@vars t1 t2 kernel_period kernel_length
symbolic_kernel = periodic_kernel_base([kernel_period, kernel_length], t1 - t2)
kernel_name = "periodic_kernel"
kernel_coder(symbolic_kernel, kernel_name)

# process for quasi_periodic_kernel_base

@vars t1 t2 RBF_kernel_length P_kernel_period P_kernel_length
symbolic_kernel = quasi_periodic_kernel_base([RBF_kernel_length, P_kernel_period, P_kernel_length], t1 - t2)
kernel_name = "quasi_periodic_kernel"
kernel_coder(symbolic_kernel, kernel_name)

#process for matern52_kernel_base

@vars t1 t2 kernel_length
symbolic_kernel = matern52_kernel_base(kernel_length, t1 - t2)
kernel_name = "matern52_kernel"
kernel_coder(symbolic_kernel, kernel_name)

#process for matern92_kernel_base

@vars t1 t2 kernel_length
symbolic_kernel = matern92_kernel_base(kernel_length, t1 - t2)
kernel_name = "matern92_kernel"
kernel_coder(symbolic_kernel, kernel_name)

#process for rq_kernel_base

@vars t1 t2 kernel_length alpha
symbolic_kernel = rq_kernel_base([kernel_length, alpha], t1 - t2)
kernel_name = "rq_kernel"
kernel_coder(symbolic_kernel, kernel_name)
