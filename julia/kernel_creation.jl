include("all_functions.jl")

using SymEngine

"holy crap this is dumb"
function kernel_coder(symbolic_kernel_original::Basic, kernel_name::String)

    symbols = free_symbols(symbolic_kernel_original)
    sym_amount = length(symbols)
    symbols_str = [string(symbols[i]) for i in 1:sym_amount]
    @assert symbols_str[1]=="t1" "The first symbol needs to be t1"
    @assert symbols_str[2]=="t2" "The second symbol needs to be t2"

    file_loc = "kernels/" * kernel_name * ".jl"
    io = open(file_loc, "w")

    write(io, "\n\n\"" * kernel_name * " function created by kernel_coder(). Requires $(sym_amount-2) hyperparameters.\"\n")
    write(io, "function " * kernel_name * "(hyperparameters::Array{Float64,1}, dif::Float64; dorder::Array{}=zeros(1))\n\n")
    write(io, "    @assert length(hyperparameters)==$(sym_amount-2) \"hyperparameters is the wrong length\"\n")
    write(io, "    if dorder==zeros(1)\n")
    write(io, "        dorder = zeros(length(hyperparameters) + 2)\n")
    write(io, "    else\n")
    write(io, "        @assert length(dorder)==(length(hyperparameters) + 2) \"dorder is the wrong length\"\n")
    write(io, "    end\n")
    write(io, "    dorder = convert(Array{Int64,1}, dorder)\n\n")


    # hyperparameters, dif; dorder=zeros(4)
    for i in 3:sym_amount
        write(io, "    " * symbols_str[i] * " = hyperparameters[$i-2]" * "\n")
    end

    write(io, "\n")

    dorders = zeros(9 * (2 ^ (sym_amount - 2)), sym_amount)
    amount_of_dorders = size(dorders,1)
    for i in 1:amount_of_dorders
          quant = amount_of_dorders - i
          dorders[i, 1] = rem(quant, 3)
          quant = div(quant, 3)
          dorders[i, 2] = rem(quant, 3)
          quant = div(quant, 3)
          for j in 1:(sym_amount - 2)
                dorders[i, j+2] = rem(quant, 2)
                quant = div(quant, 2)
          end
    end

    for i in 1:amount_of_dorders

        dorder = convert(Array{Int64,1}, dorders[i, :])
        symbolic_kernel = copy(symbolic_kernel_original)

        write(io, "    if dorder==" * string(dorder) * "\n")

        # performing the differentiations
        for j in 1:sym_amount
            symbolic_kernel = diff(symbolic_kernel, symbols[j], dorder[j])
        end
        @vars dif
        symbolic_kernel = subs(symbolic_kernel, t1=>dif, t2=>0)
        symbolic_kernel_str = SymEngine.toString(symbolic_kernel)
        symbolic_kernel_str = replace(symbolic_kernel_str, "sqrt(dif^2)"=>"abs(dif)")
        write(io, string("        func = " * symbolic_kernel_str * "\n    end\n\n"))

    end


    write(io, string("    return float(func)\n\n"))
    write(io, "end")
    close(io)

end




"Radial basis function GP kernel (aka squared exonential, ~gaussian)"
function RBF_kernel_base(hyperparameters, dif_sq)

    if length(hyperparameters) > 1
        kernel_amplitude, kernel_length = hyperparameters
    else
        kernel_amplitude = 1
        kernel_length = hyperparameters
    end

    return kernel_amplitude ^ 2 * exp(-dif_sq / (2 * kernel_length ^ 2))
end



@vars t1 t2 kernel_length
dpdt = RBF_kernel_base(kernel_length, (t1 - t2) ^ 2)
kernel_name = "RBF_kernel_sym"
kernel_coder(dpdt, kernel_name)
include("kernels/" * kernel_name * ".jl")


function func_comp(n)
    hyperparameters = [1.,2]
    dif = 0.1
    dorder2 = [2, 0]

    #julia --help for optimization parameters
    @time [Periodic_kernel_sym(hyperparameters, dif, dorder=vcat(dorder2, [0, 0])) for i in 1:n]
    @time [Periodic_kernel(hyperparameters, dif, dorder=vcat(dorder2, [0, 0])) for i in 1:n]
    @time [Periodic_kernel_sym(hyperparameters, dif, dorder=vcat(dorder2, [1, 0])) for i in 1:n]
    @time [Periodic_kernel(hyperparameters, dif, dorder=vcat(dorder2, [1, 0])) for i in 1:n]
    @time [Periodic_kernel_sym(hyperparameters, dif, dorder=vcat(dorder2, [0, 1])) for i in 1:n]
    @time [Periodic_kernel(hyperparameters, dif, dorder=vcat(dorder2, [0, 1])) for i in 1:n]
end

func_comp(1000)
