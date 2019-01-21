include("all_functions.jl")

using SymEngine

kernel_length = 1.
kernel_period = 2
dif = 3.5
exp(-2*sin(pi*abs(dif)/kernel_period)^2/kernel_length^2)
Periodic_kernel_base([kernel_period, kernel_length], abs(dif))
kernel_name = "Periodic_kernel"
include("kernels/" * kernel_name * ".jl")
Periodic_kernel_test([kernel_period, kernel_length], abs(dif))
exp(-2*sin(pi*abs(dif)/ kernel_period)^2/(kernel_length^2))

hyper=[2,3]
dorders = zeros(9 * (2 ^ length(hyper)), 2 + length(hyper))
tot = size(dorders,1)
for i in 1:tot
      quant = copy(tot-i)
      dorders[i, 1] = rem(quant, 3)
      quant = div(quant, 3)
      dorders[i, 2] = rem(quant, 3)
      quant = div(quant, 3)
      for j in 1:length(hyper)
            dorders[i, j+2] = rem(quant, 2)
            quant = div(quant, 2)
      end
end

div(42,4)
