using SpecialFunctions
using SymEngine


"checks the length of hyperparameters against the passed proper_length and adds a unity kernel_amplitude if necessary."
function check_hyperparameters(hyper::Union{Array{Float64,1},Array{Basic,1}}, proper_length::Int)
    if length(hyper) < proper_length
        @assert (length(hyper) + 1) == proper_length "incompatible amount of hyperparameters passed (too few)"
        hyper = prepend!(copy(hyper), [1.])
    end
    @assert length(hyper) == proper_length "incompatible amount of hyperparameters passed (too many)"
    return hyper
end


"Linear GP kernel"
function Linear_kernel_base(hyperparameters, x1, x2)

    hyperparameters = check_hyperparameters(hyperparameters, 1+1)
    sigma_b, sigma_a = hyperparameters

    return sigma_b ^ 2 * vecdot(x1, x2) + sigma_a ^ 2
end


"Radial basis function GP kernel (aka squared exonential, ~gaussian)"
function RBF_kernel_base(hyperparameters, dif)

    dif_sq = dif ^ 2

    hyperparameters = check_hyperparameters(hyperparameters, 1+1)
    kernel_amplitude, kernel_length = hyperparameters

    return kernel_amplitude ^ 2 * exp(-dif_sq / (2 * (kernel_length ^ 2)))
end


"Periodic kernel (for random cyclic functions)"
function Periodic_kernel_base(hyperparameters, dif)

    # abs_dif = sqrt(dif^2)
    # abs_dif = abs(dif)

    hyperparameters = check_hyperparameters(hyperparameters, 2+1)
    kernel_amplitude, kernel_period, kernel_length = hyperparameters

    # reframed to make it easier for symbolic derivatives to not return NaNs
    # using sin(abs(u))^2 = sin(u)^2 ( also = 1 - cos(u)^2 )
    # return kernel_amplitude ^ 2 * exp(-2 * sin(pi * (abs_dif / kernel_period)) ^ 2 / (kernel_length ^ 2))
    return kernel_amplitude ^ 2 * exp(-2 * sin(pi * (dif / kernel_period)) ^ 2 / (kernel_length ^ 2))


end


"Quasi-periodic kernel"
function Quasi_periodic_kernel_base(hyperparameters, dif)

    hyperparameters = check_hyperparameters(hyperparameters, 3+1)
    kernel_amplitude, RBF_kernel_length, P_kernel_period, P_kernel_length = hyperparameters

    return RBF_kernel_base([RBF_kernel_length], dif) * Periodic_kernel_base([P_kernel_period, P_kernel_length], dif)
end


"Ornstein–Uhlenbeck (Exponential) kernel"
function OU_kernel_base(hyperparameters, dif)

    hyperparameters = check_hyperparameters(hyperparameters, 1+1)
    kernel_amplitude, kernel_length = hyperparameters

    return kernel_amplitude ^ 2 * exp(-dif / kernel_length)
end


"general Matern kernel"
function Matern_kernel_base(hyperparameters, dif, nu)

    hyperparameters = check_hyperparameters(hyperparameters, 1+1)
    kernel_amplitude, kernel_length = hyperparameters

    #limit of the function as it apporaches 0 (see https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function)
    if dif == 0
        return kernel_amplitude ^ 2
    else
        x = (sqrt(2 * nu) * dif) / kernel_length
        return kernel_amplitude ^ 2 * ((2 ^ (1 - nu)) / (gamma(nu))) * x ^ nu * besselk(nu, x)
    end
end


"Matern 3/2 kernel"
function Matern32_kernel_base(hyperparameters, dif)

    hyperparameters = check_hyperparameters(hyperparameters, 1+1)
    kernel_amplitude, kernel_length = hyperparameters

    x = sqrt(3) * dif / kernel_length
    return kernel_amplitude ^ 2 * (1 + x) * exp(-x)
end


"Matern 5/2 kernel"
function Matern52_kernel_base(hyperparameters, dif)

    hyperparameters = check_hyperparameters(hyperparameters, 1+1)
    kernel_amplitude, kernel_length = hyperparameters

    x = sqrt(5) * dif / kernel_length
    return kernel_amplitude ^ 2 * (1 + x + (x ^ 2) / 3) * exp(-x)
end


"""
Rational Quadratic kernel (equivalent to adding together many SE kernels
with different lengthscales. When α→∞, the RQ is identical to the SE.)
"""
function RQ_kernel_base(hyperparameters, dif_sq)

    hyperparameters = check_hyperparameters(hyperparameters, 2+1)
    kernel_amplitude, kernel_length, alpha = hyperparameters

    alpha = max(alpha, 0)
    return kernel_amplitude ^ 2 * (1 + dif_sq / (2 * alpha * kernel_length ^ 2)) ^ -alpha
end


"""
Bessel (function of he first kind) kernel
Bessel functions of the first kind, denoted as Jα(x), are solutions of Bessel's
differential equation that are finite at the origin (x = 0) for integer or positive α
http://crsouza.com/2010/03/17/kernel-functions-for-machine-learning-applications/#bessel
"""
function Bessel_kernel_base(hyperparameters, dif; nu=0)

    hyperparameters = check_hyperparameters(hyperparameters, 2+1)
    kernel_amplitude, kernel_length, n = hyperparameters

    @assert nu >= 0 "nu must be >= 0"

    return besselj(nu, kernel_length * dif) / (dif ^ (-n * nu))
end

######################### OLD ##########################


# # # Radial basis function kernel (aka squared exonential, ~gaussian)
# # function RBF_kernel(hyperparameters, dif_sq)
# #
# #     if length(hyperparameters) > 1
# #         kernel_amplitude, kernel_length = hyperparameters
# #     else
# #         kernel_amplitude = 1
# #         kernel_length = hyperparameters
# #     end
# #
# #     return kernel_amplitude ^ 2 * exp(-dif_sq / (2 * kernel_length ^ 2))
# # end
#
#
# # Differentiated Radial basis function kernel (aka squared exonential, ~gauss)
# # ONLY WORKS FOR 1D TIME AS INPUTS
# function dRBFdt_kernel(hyperparameters, dif, dorder)
#
#     dorder = convert(Array{Int64,1}, dorder)
#
#     if length(hyperparameters) > 1
#         kernel_amplitude, kernel_length = hyperparameters
#     else
#         kernel_length = hyperparameters[1]
#     end
#
#     RBF = RBF_kernel_base(hyperparameters, dif)
#
#     sum_dorder = sum(dorder)
#
#     # first coefficients are triangular numbers. second coefficients are
#     # tri triangular numbers
#     if sum_dorder > 0
#
#         T1 = dif / kernel_length ^ 2
#         if sum_dorder == 1
#             value = (T1)
#         elseif sum_dorder == 2
#             value = (T1 ^ 2 - 1 / kernel_length ^ 2)
#         elseif sum_dorder == 3
#             value = (T1 ^ 3 - 3 * T1 / (kernel_length ^ 2))
#         elseif sum_dorder == 4
#             value = (T1 ^ 4 - 6 * T1 ^ 2 / (kernel_length ^ 2)
#                 + 3 / (kernel_length ^ 4))
#         end
#
#         if isodd(convert(Int64, dorder[1]))
#             value = -value
#         end
#
#         return RBF * value
#
#     else
#
#         return RBF
#     end
#
# end
#
#
# # kernel length differentiated Radial basis function kernel (aka squared exonential, ~gauss)
# # ONLY WORKS FOR 1D TIME AS INPUTS
# function dRBFdλ_kernel(hyperparameters, dif, dorder)
#
#     dorder = convert(Array{Int64,1}, dorder)
#
#     if length(hyperparameters) > 1
#         kernel_amplitude, kernel_length = hyperparameters
#     else
#         kernel_length = hyperparameters[1]
#     end
#
#     RBF = RBF_kernel_base(hyperparameters, dif)
#
#     sum_dorder = sum(dorder)
#
#     T1 = dif / kernel_length ^ 2
#     if sum_dorder == 0
#         value = (T1 ^ 2)
#     elseif sum_dorder == 1
#         value = (T1 ^ 3 - 2 * T1 / (kernel_length ^ 2))
#     elseif sum_dorder == 2
#         value = (T1 ^ 4 - 5 * T1 ^ 2 / (kernel_length ^ 2)
#             + 2 / (kernel_length ^ 4))
#     elseif sum_dorder == 3
#         value = (T1 ^ 5 - 9 * T1 ^ 3 / (kernel_length ^ 2)
#             + 12 * T1 / (kernel_length ^ 4))
#     elseif sum_dorder == 4
#         value = (T1 ^ 6 - 14 * T1 ^ 4 / (kernel_length ^ 2)
#             + 39 * T1 ^ 2 / (kernel_length ^ 4) - 12 / (kernel_length ^ 6))
#     end
#
#     if isodd(convert(Int64, dorder[1]))
#         value = -value
#     end
#
#     return RBF * kernel_length * value
#
# end
#
#
# # # Periodic kernel
# # function Periodic_kernel(hyperparameters, abs_dif)
# #
# #     if length(hyperparameters) > 2
# #         kernel_amplitude, kernel_length, kernel_period = hyperparameters
# #     else
# #         kernel_amplitude = 1
# #         kernel_length, kernel_period = hyperparameters
# #     end
# #
# #     return kernel_amplitude ^ 2 * exp(-2 * sin(pi * abs_dif / kernel_period) ^ 2 / (kernel_length ^ 2))
# # end
#
#
# # Differentiated periodic kernel
# # ONLY WORKS FOR 1D TIME AS INPUTS
# function dPeriodicdt_kernel(hyperparameters, dif, dorder)
#
#     dorder = convert(Array{Int64,1}, dorder)
#
#     if length(hyperparameters) > 2
#         kernel_amplitude, kernel_length, kernel_period = hyperparameters
#     else
#         kernel_length, kernel_period = hyperparameters
#     end
#
#     Periodic = Periodic_kernel_base(hyperparameters, dif)
#
#     sum_dorder = sum(dorder)
#
#     if sum_dorder > 0
#
#         theta = 2 * pi * dif / kernel_period
#         Sin = sin(theta)
#         Cos_tab = [cos(i * theta) for i in 1:(2 * floor(sum_dorder/2))]
#
#         if sum_dorder == 1
#             value = (Sin)
#         elseif sum_dorder == 2
#             value = -1 * (-1 + 2 * kernel_length ^ 2 * Cos_tab[1] + Cos_tab[2])
#         elseif sum_dorder == 3
#             value = (-2 * Sin * (-1 + 2 * kernel_length ^ 4
#                 + 6 * kernel_length ^ 2 * Cos_tab[1] + Cos_tab[2]))
#         elseif sum_dorder == 4
#             value = (3 - 4 * kernel_length ^ 4
#                 + 4 * kernel_length ^ 2 * (-3 + 2 * kernel_length ^ 4) * Cos_tab[1]
#                 + 4 * (-1 + 7 * kernel_length ^ 4) * Cos_tab[2]
#                 + 12 * kernel_length ^ 2 * Cos_tab[3]
#                 + Cos_tab[4])
#         end
#         if isodd(convert(Int64, dorder[1]))
#             value = -value
#         end
#
#         constant = (pi / (kernel_period * kernel_length ^ 2)) ^ sum_dorder
#         return 2 * constant * Periodic * value
#     else
#         return Periodic
#     end
#
# end
#
#
# # kernel length differentiated periodic kernel
# # ONLY WORKS FOR 1D TIME AS INPUTS
# function dPeriodicdλ_kernel(hyperparameters, dif, dorder)
#
#     dorder = convert(Array{Int64,1}, dorder)
#
#     if length(hyperparameters) > 2
#         kernel_amplitude, kernel_length, kernel_period = hyperparameters
#     else
#         kernel_length, kernel_period = hyperparameters
#     end
#
#     Periodic = Periodic_kernel_base(hyperparameters, dif)
#
#     sum_dorder = sum(dorder)
#
#     theta = pi * dif / kernel_period
#     Sin_tab = [sin(i * theta) for i in 1:maximum([2, sum_dorder + 1])]
#     Cos_tab = [cos(2 * i * theta) for i in 1:(1 + 2 * floor(sum_dorder/2))]
#
#     if sum_dorder == 0
#         value = (Sin_tab[1] ^ 2)
#     elseif sum_dorder == 1
#         value = (-Sin_tab[2] * (-1 + kernel_length ^ 2 + Cos_tab[1]))
#     elseif sum_dorder == 2
#         value = (1 / 2 * (2 - 2 * kernel_length ^ 2
#             + (-1 - 4 * kernel_length ^ 2 + 4 * kernel_length ^ 4) * Cos_tab[1]
#             + (-2 + 6 * kernel_length ^ 2) * Cos_tab[2]
#             + Cos_tab[3]))
#     elseif sum_dorder == 3
#         value = (Sin_tab[2] * (2 - 4 * kernel_length ^ 4 + 4 * kernel_length ^ 6
#             + (-1 - 12 * kernel_length ^ 2 + 28 * kernel_length ^ 4) * Cos_tab[1]
#             + 2 * (-1 + 6 * kernel_length ^ 2) * Cos_tab[2]
#             + Cos_tab[3]))
#     elseif sum_dorder == 4
#         value = (-1 / 2 * (-6 + 12 * kernel_length ^ 2 + 8 * kernel_length ^ 4 - 8 * kernel_length ^ 6
#             + 2 * (1 + 12 * kernel_length ^ 2 - 26 * kernel_length ^ 4 - 8 * kernel_length ^ 6 + 8 * kernel_length ^ 8) * Cos_tab[1]
#             + 8 * (1 - 4 * kernel_length ^ 2 - 7 * kernel_length ^ 4 + 15 * kernel_length ^ 6) * Cos_tab[2]
#             + (-3 - 24 * kernel_length ^ 2 + 100 * kernel_length ^ 4) * Cos_tab[3]
#             + 2 * (-1 + 10 * kernel_length ^ 2) * Cos_tab[4]
#             + Cos_tab[5]))
#     end
#
#     if isodd(convert(Int64, dorder[1]))
#         value = -value
#     end
#
#     constant = (pi / (kernel_period * kernel_length ^ 2)) ^ sum_dorder
#     return 4 / kernel_length ^ 3 * constant * Periodic * value
#
# end
#
#
# # kernel period differentiated periodic kernel
# # ONLY WORKS FOR 1D TIME AS INPUTS
# function dPeriodicdp_kernel(hyperparameters, dif, dorder)
#
#     dorder = convert(Array{Int64,1}, dorder)
#
#     if length(hyperparameters) > 2
#         kernel_amplitude, kernel_length, kernel_period = hyperparameters
#     else
#         kernel_length, kernel_period = hyperparameters
#     end
#
#     Periodic = Periodic_kernel_base(hyperparameters, dif)
#
#     sum_dorder = sum(dorder)
#
#     theta = 2 * pi *dif / kernel_period
#     Sin_tab = [sin(i * theta) for i in 1:maximum([3,(1 + 2 * floor(sum_dorder/2))])]
#     Cos_tab = [cos(i * theta) for i in 1:(2 * floor(sum_dorder/2 + 1))]
#
#     if sum_dorder == 0
#         value = (pi * dif * Sin_tab[1])
#     elseif sum_dorder == 1
#         value = (-1 * (pi * dif * (-1 + 2 * kernel_length ^ 2 * Cos_tab[1] + Cos_tab[2])
#             + kernel_period * kernel_length ^ 2 * Sin_tab[1]))
#     elseif sum_dorder == 2
#         value = (-1 * (-2 * kernel_period * kernel_length ^ 2 * (-1 + 2 * kernel_length ^ 2 * Cos_tab[1] + Cos_tab[2])
#             + 2 * pi * dif * (-1 + 2 * kernel_length ^ 4 + 6 * kernel_length ^ 2 * Cos_tab[1] + Cos_tab[2]) * Sin_tab[1]))
#     elseif sum_dorder == 3
#         value = (-pi * dif * (-3 + 4 * kernel_length ^ 4)
#             + 4 * pi * dif * kernel_length ^ 2 * (-3 + 2 * kernel_length ^ 4) * Cos_tab[1]
#             + 4 * pi * dif * (-1 + 7 * kernel_length ^ 4) * Cos_tab[2]
#             + 12 * pi * dif * kernel_length ^ 2 * Cos_tab[3]
#             + pi * dif * Cos_tab[4]
#             + 3 * kernel_period * kernel_length ^ 2 * (-3 + 4 * kernel_length ^ 4) * Sin_tab[1]
#             + 18 * kernel_period * kernel_length ^ 4 * Sin_tab[2]
#             + 3 * kernel_period * kernel_length ^ 2 * Sin_tab[3])
#     elseif sum_dorder == 4
#         value = (-4 * kernel_period * kernel_length ^ 2 * (3
#             - 4 * kernel_length ^ 4 + 4 * kernel_length ^ 2 * (-3 + 2 * kernel_length ^ 4) * Cos_tab[1]
#             + 4 * (-1 + 7 * kernel_length ^ 4) * Cos_tab[2]
#             + 12 * kernel_length ^ 2 * Cos_tab[3]
#             + Cos_tab[4])
#             + 2 * pi * dif * Sin_tab[1] * (3 + 20 * kernel_length ^ 4 + 8 * kernel_length ^ 8
#             + 20 * kernel_length ^ 2 * (-1 + 6 * kernel_length ^ 4) * Cos_tab[1]
#             + 4 * (-1 + 25 * kernel_length ^ 4) * Cos_tab[2]
#             + 20 * kernel_length ^ 2 * Cos_tab[3]
#             + Cos_tab[4]))
#     end
#
#     if isodd(convert(Int64, dorder[1]))
#         value = -value
#     end
#
#     constant = (pi / (kernel_period * kernel_length ^ 2)) ^ sum_dorder
#     return 2 / (kernel_period ^ 2 * kernel_length ^ 2) * constant * Periodic * value
#
# end
