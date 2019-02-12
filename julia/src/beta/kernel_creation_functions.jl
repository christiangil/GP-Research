using SymEngine

"""
Creates the necessary differentiated versions of base kernels required by the Jones et al. 2017 paper (https://arxiv.org/abs/1711.01318).
You must pass it a SymEngine Basic object with the variables already declared with the @vars command. t1 and t2 must be the first declared variables.
The created functions will looks like this

    \$kernel_name(hyperparameters::Union{Array{Float64,1},Array{Any,1}}, dif::Float64; dorder::Union{Array{Int,1},Array{Float64,1}}=zeros(1))

For example, you could define a kernel like so:

    "Radial basis function GP kernel (aka squared exonential, ~gaussian)"
    function RBF_kernel_base(hyperparameters, dif)

        dif_sq = dif ^ 2

        hyperparameters = check_hyperparameters(hyperparameters, 1+1)
        kernel_amplitude, kernel_length = hyperparameters

        return kernel_amplitude ^ 2 * exp(-dif_sq / (2 * (kernel_length ^ 2)))
    end

And then calculate the necessary derivative versions like so:

    @vars t1 t2 kernel_length
    symbolic_kernel = RBF_kernel_base(kernel_length, t1 - t2)
    kernel_name = "RBF_kernel"
    kernel_coder(symbolic_kernel, kernel_name)

The function is saved in src/kernels/\$kernel_name.jl, so you can use it with a command akin to this:

    include("src/kernels/" * kernel_name * ".jl")

"""
function kernel_coder(symbolic_kernel_original::Basic, kernel_name::String, manual_simplifications::Bool=true)

    # get the symbols of the passed function and check that t1 and t2 are first
    symbols = free_symbols(symbolic_kernel_original)
    sym_amount = length(symbols)
    symbols_str = [string(symbols[i]) for i in 1:sym_amount]
    @assert symbols_str[1]=="t1" "The first symbol needs to be t1"
    @assert symbols_str[2]=="t2" "The second symbol needs to be t2"

    # open the file we will write to
    file_loc = "src/kernels/" * kernel_name * ".jl"
    io = open(file_loc, "w")

    num_kernel_hyperparameters = sym_amount-2
    # begin to write the function including assertions that the amount of hyperparameters are correct
    write(io, "\n\n\"\"\"\n" * kernel_name * " function created by kernel_coder(). Requires $num_kernel_hyperparameters hyperparameters. Likely created using $kernel_name" * "_base() as an input. \nUse with include(\"kernels/$kernel_name.jl\").\n\"\"\"\n")
    write(io, "function " * kernel_name * "(hyperparameters::Array{Any,1}, dif::Float64; dorder::Union{Array{Int,1},Array{Float64,1}}=zeros(1))\n\n")
    write(io, "    @assert length(hyperparameters)==$num_kernel_hyperparameters \"hyperparameters is the wrong length\"\n")
    write(io, "    if dorder==zeros(1)\n")
    write(io, "        dorder = zeros(length(hyperparameters) + 2)\n")
    write(io, "    else\n")
    write(io, "        @assert length(dorder)==(length(hyperparameters) + 2) \"dorder is the wrong length\"\n")
    write(io, "    end\n")
    write(io, "    dorder = convert(Array{Int64,1}, dorder)\n\n")


    # map the hyperparameters that will be passed to this function to the symbol names
    for i in 3:sym_amount
        write(io, "    " * symbols_str[i] * " = hyperparameters[$i-2]" * "\n")
    end

    write(io, "\n")

    # calculate all of the necessary derivations we need for the Jones model
    # for four symbols, dorders is of the form: [2.0 2.0 1.0 1.0; 1.0 2.0 1.0 1.0; 0.0 2.0 1.0 1.0; 2.0 1.0 1.0 1.0; ... ]
    # where the dorders[n, :]==[dorder of t1 (0-2), dorder of t2 (0-2), dorder of symbol 3 (0-1), dorder of symbol 4 (0-1)]
    # can be made for any number of symbols
    # this could be optimized to remove the instances where differentiations of multiple, non-time symbols are asked for
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

    # for each differentiated version of the kernel
    for i in 1:amount_of_dorders

        dorder = convert(Array{Int64,1}, dorders[i, :])

        # only record another differentiated version of the function if we will actually use it
        # i.e. no instances where differentiations of multiple, non-time symbols are asked for
        if sum(dorder[3:end]) < 2
            symbolic_kernel = copy(symbolic_kernel_original)

            write(io, "    if dorder==" * string(dorder) * "\n")

            # performing the differentiations
            for j in 1:sym_amount
                symbolic_kernel = diff(symbolic_kernel, symbols[j], dorder[j])
            end

            # make a simplification based on t1-t2=dif
            @vars dif
            symbolic_kernel = subs(symbolic_kernel, t1=>dif, t2=>0)

            symbolic_kernel_str = SymEngine.toString(symbolic_kernel)
            symbolic_kernel_str = string("        func = " * symbolic_kernel_str * "\n    end\n\n")

            if manual_simplifications
                # replacing equivalent expressions to increase readability and decrease useless computations
                symbolic_kernel_str = replace(symbolic_kernel_str, "sqrt(dif^2)"=>"abs(dif)")
                symbolic_kernel_str = replace(symbolic_kernel_str, " 0.0 - "=>" -")
                symbolic_kernel_str = replace(symbolic_kernel_str, " 0.0 + "=>" ")
                symbolic_kernel_str = replace(symbolic_kernel_str, " 1.0*"=>" ")

                # replacing all calls to ^
                symbols_str = symbols_str[2:end]
                symbols_str[1] = "dif"
                for symb in symbols_str
                    replace(symbolic_kernel_str, symb=>"($symb)")
                end
                pow_index = findfirst(isequal(")^"), symbolic_kernel_str)
                while typeof(pow_index)!=Nothing
                    paren_count = 1
                    begin_index = copy(pow_index) - 1
                    while paren_count>0
                        if symbolic_kernel_str[begin_index]==")"
                            paren_count+=1
                        elseif symbolic_kernel_str[begin_index]=="("
                            paren_count-=1
                        end
                        begin_index-=1
                    end
                    write(io, symbolic_kernel_str)
                    replace(symbolic_kernel_str, symbolic_kernel_str=>"($symb)")
                    pow_index = findfirst(isequal(")^"), symbolic_kernel_str)
                end
            end

            write(io, symbolic_kernel_str)
        end

    end

    # # error catching, should only do things for weird derivative behaviors from abs(t1-t2) stuff
    # write(io, string("    if isnan(func)\n"))
    # write(io, string("        func = 0\n"))
    # write(io, string("    end\n\n"))

    write(io, string("    return float(func)\n\n"))
    write(io, "end\n\n\n")
    write(io, "return $num_kernel_hyperparameters  # the number of kernel hyperparameters\n")
    close(io)

    @warn "The order of hyperparameters in the function may be different from the order given when you made them in @vars. Check the created function file (or the print statement below) to see what order you need to actually use."
    println("$kernel_name() created at $file_loc")
    println("hyperparameters == ",symbols_str[3:end])
end
