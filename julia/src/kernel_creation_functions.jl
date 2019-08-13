using SymEngine

"""
Creates the necessary differentiated versions of base kernels required by the Jones et al. 2017 paper (https://arxiv.org/abs/1711.01318) method.
You must pass it a SymEngine Basic object with the variables already declared with the @vars command. δ or abs_δ must be the first declared variables.
The created function will look like this

    \$kernel_name(hyperparameters::Vector{<:Real}, δ::Real; dorder::Vector{<:Integer}=zeros(Int64, length(hyperparameters) + 2))

For example, you could define a kernel like so:

    "Radial basis function GP kernel (aka squared exonential, ~gaussian)"
    function rbf_kernel_base(λ::Number, δ::Number)
        return exp(-δ ^ 2 / (2 * λ ^ 2))
    end

And then calculate the necessary derivative versions like so:

    @vars δ λ
    kernel_coder(rbf_kernel_base(λ, δ), "rbf_kernel")

The function is saved in src/kernels/\$kernel_name.jl, so you can use it with a command akin to this:

    include("src/kernels/" * kernel_name * ".jl")

"""
function kernel_coder(symbolic_kernel_original::Basic, kernel_name::String, manual_simplifications::Bool=true)

    # get the symbols of the passed function and check that δ is first
    symbols = free_symbols(symbolic_kernel_original)
    symbols_str = [string(symbol) for symbol in symbols]
    δ_inds = findall(x -> x=="abs_δ", symbols_str)
    @assert length(δ_inds) < 2
    uses_abs_δ = (length(δ_inds)==1)
    if uses_abs_δ
        @vars δ
        δ_sym = free_symbols(δ)[1]
        symbolic_kernel_original = subs(symbolic_kernel_original, abs_δ=>δ)
    else
        δ_inds = findall(x -> x=="δ", symbols_str)
        @assert 0<length(δ_inds)<2 "A single δ or abs_δ symbol must be passed"
        δ_sym = symbols[δ_inds[1]]
    end
    δ_ind = δ_inds[1]
    deleteat!(symbols, δ_ind)
    deleteat!(symbols_str, δ_ind)
    # println(symbols_str)
    # println(symbols)
    hyper_amount = length(symbols)

    # open the file we will write to
    file_loc = "src/kernels/" * kernel_name * ".jl"
    io = open(file_loc, "w")

    num_kernel_hyperparameters = hyper_amount
    # begin to write the function including assertions that the amount of hyperparameters are correct
    write(io, "\n\n\"\"\"\n" * kernel_name * " function created by kernel_coder(). Requires $num_kernel_hyperparameters hyperparameters. Likely created using $kernel_name" * "_base() as an input. \nUse with include(\"src/kernels/$kernel_name.jl\").\nhyperparameters == $symbols_str\n\"\"\"\n")
    write(io, "function " * kernel_name * "(\n    hyperparameters::Vector{<:Real}, \n    δ::Real; \n    dorder::Vector{<:Integer}=zeros(Int64, length(hyperparameters) + 2))\n\n")
    write(io, "    @assert length(hyperparameters)==$num_kernel_hyperparameters \"hyperparameters is the wrong length\"\n")
    write(io, "    @assert length(dorder)==($num_kernel_hyperparameters + 2) \"dorder is the wrong length\"\n")
    write(io, "    even_time_derivative = powers_of_negative_one(dorder[2])\n")
    write(io, "    @assert maximum(dorder) < 3 \"No more than two time derivatives for either t1 or t2 can be calculated\"\n\n")
    write(io, "    dorder[1] = sum(dorder[1:2])\n")
    write(io, "    dorder[2:(end - 1)] = dorder[3:end]\n\n")
    write(io, "    deleteat!(dorder, length(dorder))\n\n")

    # map the hyperparameters that will be passed to this function to the symbol names
    for i in 1:(hyper_amount)
        write(io, "    " * symbols_str[i] * " = hyperparameters[$i]" * "\n")
    end
    write(io, "\n")

    if uses_abs_δ
        write(io, "    δ_positive = (δ >= 0)  # store original sign of δ (for correcting odd kernel derivatives)\n")
        write(io, "    δ = abs(δ)  # this is corrected for later\n\n")
    end

    # δf has 5 derivatives (0-4) and the other symbols have 3 (0-2)
    max_δ_derivs = 5  # (0-4)
    max_hyper_derivs = 3  # (0-2)
    # calculate all of the necessary derivations we need for the Jones model
    # for two symbols (δ and a hyperparameter), dorders is of the form:
    # [4 2; 3 2; 2 2; 1 2; 0 2; ... ]
    # where the dorders[n, :]==[dorder of δ, dorder of symbol 2]
    # can be made for any number of symbols

    dorders = zeros(Int64, max_δ_derivs * (max_hyper_derivs ^ hyper_amount), hyper_amount + 1)
    amount_of_dorders = size(dorders,1)
    for i in 1:amount_of_dorders
          quant = amount_of_dorders - i
          dorders[i, 1] = rem(quant, max_δ_derivs)
          quant = div(quant, max_δ_derivs)
          for j in 1:(hyper_amount)
                dorders[i, j+1] = rem(quant, max_hyper_derivs)
                quant = div(quant, max_hyper_derivs)
          end
    end

    # for each differentiated version of the kernel
    for i in 1:amount_of_dorders

        # dorder = convert(Vector{Int64}, dorders[i, :])
        dorder = dorders[i, :]

        # only record another differentiated version of the function if we will actually use it
        # i.e. no instances where differentiations of multiple, non-time symbols are asked for
        if sum(dorder[2:end]) < max_hyper_derivs
            symbolic_kernel = copy(symbolic_kernel_original)

            write(io, "    if dorder==" * string(dorder) * "\n")

            # performing the differentiations
            symbolic_kernel = diff(symbolic_kernel, δ_sym, dorder[1])
            for j in 1:hyper_amount
                # println(symbols[j], dorder[j+1])
                symbolic_kernel = diff(symbolic_kernel, symbols[j], dorder[j+1])
            end

            symbolic_kernel_str = SymEngine.toString(symbolic_kernel)
            symbolic_kernel_str = string("        func = " * symbolic_kernel_str * "\n    end\n\n")

            if manual_simplifications
                # replacing equivalent expressions to increase readability and decrease useless computations
                symbolic_kernel_str = replace(symbolic_kernel_str, "sqrt(δ^2)"=>"abs(δ)")
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
    uses_abs_δ ? write(io, "    return -powers_of_negative_one(δ_positive || iseven(dorder[1])) * even_time_derivative * float(func)  # correcting for use of abs_δ and amount of t2 derivatives\n\n") : write(io, "    return even_time_derivative * float(func)  # correcting for amount of t2 derivatives\n\n")
    write(io, "end\n\n\n")
    write(io, "return $kernel_name, $num_kernel_hyperparameters  # the function handle and the number of kernel hyperparameters\n")
    close(io)

    @warn "The order of hyperparameters in the function may be different from the order given when you made them in @vars. Check the created function file (or the print statement below) to see what order you need to actually use."
    println("$kernel_name() created at $file_loc")
    println("hyperparameters == ", symbols_str)
end
