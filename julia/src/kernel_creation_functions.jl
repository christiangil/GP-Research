using SymEngine

"""
Creates the necessary differentiated versions of base kernels required by the Jones et al. 2017 paper (https://arxiv.org/abs/1711.01318) method.
You must pass it a SymEngine Basic object with the variables already declared with the @vars command. dif or abs_dif must be the first declared variables.
The created function will look like this

    \$kernel_name(hyperparameters::AbstractArray{T1,1}, dif::Real; dorder::AbstractArray{T2,1}=zeros(1)) where {T1<:Real, T2<:Real}

For example, you could define a kernel like so:

    "Radial basis function GP kernel (aka squared exonential, ~gaussian)"
    function rbf_kernel_base(kernel_length::Union{Real, Basic}, dif::Union{Basic,Real}) where {T<:Real}
        return exp(-dif * dif / (2 * (kernel_length * kernel_length)))
    end

And then calculate the necessary derivative versions like so:

    @vars dif kernel_length
    kernel_coder(rbf_kernel_base(kernel_length, dif), "rbf_kernel")

The function is saved in src/kernels/\$kernel_name.jl, so you can use it with a command akin to this:

    include("src/kernels/" * kernel_name * ".jl")

"""
function kernel_coder(symbolic_kernel_original::Basic, kernel_name::String, manual_simplifications::Bool=true)

    # get the symbols of the passed function and check that t1 and t2 are first
    symbols = free_symbols(symbolic_kernel_original)
    sym_amount = length(symbols)
    symbols_str = [string(symbol) for symbol in symbols]
    @assert symbols_str[1] in ["dif", "abs_dif"] "The first symbol needs to be dif or abs_dif"

    # open the file we will write to
    file_loc = "src/kernels/" * kernel_name * ".jl"
    io = open(file_loc, "w")

    num_kernel_hyperparameters = sym_amount - 1
    # begin to write the function including assertions that the amount of hyperparameters are correct
    write(io, "\n\n\"\"\"\n" * kernel_name * " function created by kernel_coder(). Requires $num_kernel_hyperparameters hyperparameters. Likely created using $kernel_name" * "_base() as an input. \nUse with include(\"src/kernels/$kernel_name.jl\").\nhyperparameters == $(symbols_str[2:end])\n\"\"\"\n")
    write(io, "function " * kernel_name * "(\n    hyperparameters::AbstractArray{T1,1}, \n    dif::Real; \n    dorder::AbstractArray{T2,1}=zeros(Int64, length(hyperparameters) + 2) \n    ) where {T1<:Real, T2<:Integer}\n\n")
    write(io, "    @assert length(hyperparameters)==$num_kernel_hyperparameters \"hyperparameters is the wrong length\"\n")
    write(io, "    @assert length(dorder)==(length(hyperparameters) + 2) \"dorder is the wrong length\"\n")
    write(io, "    even_time_derivative = 2 * iseven(dorder[2]) - 1\n")
    write(io, "    @assert maximum(dorder) < 3 \"No more than two time derivatives for either t1 or t2 can be calculated\"\n\n")
    write(io, "    dorder = append!([sum(dorder[1:2])], dorder[3:end])\n\n")

    # map the hyperparameters that will be passed to this function to the symbol names
    for i in 1:(sym_amount - 1)
        write(io, "    " * symbols_str[i + 1] * " = hyperparameters[$i]" * "\n")
    end
    write(io, "\n")

    uses_abs_dif = (symbols_str[1] == "abs_dif")
    if uses_abs_dif
        write(io, "    dif_positive = (dif >= 0)  # store original sign of dif (for correcting odd kernel derivatives)\n")
        write(io, "    dif = abs(dif)  # this is corected for later\n\n")
        @vars dif
        symbolic_kernel_original = subs(symbolic_kernel_original, abs_dif=>dif)
        symbols = free_symbols(symbolic_kernel_original)
    end

    # diff has 5 derivatives (0-4) and the other symbols have 3 (0-2)
    max_diff_derivs = 5  # (0-4)
    max_hyper_derivs = 3  # (0-2)
    # calculate all of the necessary derivations we need for the Jones model
    # for two symbols (dif and a hyperparameter), dorders is of the form:
    # [4 2; 3 2; 2 2; 1 2; 0 2; ... ]
    # where the dorders[n, :]==[dorder of dif, dorder of symbol 2]
    # can be made for any number of symbols

    dorders = zeros(Int64, max_diff_derivs * (max_hyper_derivs ^ (sym_amount - 1)), sym_amount)
    amount_of_dorders = size(dorders,1)
    for i in 1:amount_of_dorders
          quant = amount_of_dorders - i
          dorders[i, 1] = rem(quant, max_diff_derivs)
          quant = div(quant, max_diff_derivs)
          for j in 1:(sym_amount - 1)
                dorders[i, j+1] = rem(quant, max_hyper_derivs)
                quant = div(quant, max_hyper_derivs)
          end
    end

    # for each differentiated version of the kernel
    for i in 1:amount_of_dorders

        dorder = convert(AbstractArray{Int64,1}, dorders[i, :])

        # only record another differentiated version of the function if we will actually use it
        # i.e. no instances where differentiations of multiple, non-time symbols are asked for
        if sum(dorder[2:end]) < max_hyper_derivs
            symbolic_kernel = copy(symbolic_kernel_original)

            write(io, "    if dorder==" * string(dorder) * "\n")

            # performing the differentiations
            for j in 1:sym_amount
                symbolic_kernel = diff(symbolic_kernel, symbols[j], dorder[j])
            end

            symbolic_kernel_str = SymEngine.toString(symbolic_kernel)
            symbolic_kernel_str = string("        func = " * symbolic_kernel_str * "\n    end\n\n")

            if manual_simplifications
                # replacing equivalent expressions to increase readability and decrease useless computations
                symbolic_kernel_str = replace(symbolic_kernel_str, "sqrt(dif^2)"=>"abs(dif)")
                symbolic_kernel_str = replace(symbolic_kernel_str, " 0.0 - "=>" -")
                symbolic_kernel_str = replace(symbolic_kernel_str, " 0.0 + "=>" ")
                symbolic_kernel_str = replace(symbolic_kernel_str, " 1.0*"=>" ")
                symbolic_kernel_str = replace(symbolic_kernel_str, "-1.0*"=>"-")
            end

            write(io, symbolic_kernel_str)
        end

    end

    # # error catching, should only do things for weird derivative behaviors from abs(t1-t2) stuff
    # write(io, string("    if isnan(func)\n"))
    # write(io, string("        func = 0\n"))
    # write(io, string("    end\n\n"))
    uses_abs_dif ? write(io, "    return (2 * (dif_positive || iseven(dorder[1])) - 1) * even_time_derivative * float(func)  # correcting for use of abs_dif and amount of t2 derivatives\n\n") : write(io, "    return even_time_derivative * float(func)  # correcting for amount of t2 derivatives\n\n")
    write(io, "end\n\n\n")
    write(io, "return $kernel_name, $num_kernel_hyperparameters  # the function handle and the number of kernel hyperparameters\n")
    close(io)

    @warn "The order of hyperparameters in the function may be different from the order given when you made them in @vars. Check the created function file (or the print statement below) to see what order you need to actually use."
    println("$kernel_name() created at $file_loc")
    println("hyperparameters == ",symbols_str[2:end])
end
