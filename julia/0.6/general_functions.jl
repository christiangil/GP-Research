# include("reusable_code.jl")
# using LinearAlgebra


# a generalized version of the built in append!() function
function append(a, b...)
    for i in 1:length(b)
        append!(a, b[i])
    end
    return a
end


# if array is symmetric, return the symmetric version
function symmetric_A(A)

    if size(A, 1) == size(A, 2)
        max_dif = maximum(abs.(A - transpose(A)))

        if max_dif == zero(max_dif)
            return Symmetric(A)
        # an arbitrary threshold that is meant to catch numerical errors
        elseif max_dif < 1e-6
            # return the symmetrized version of the matrix
            return Symmetric((A + transpose(A)) / 2)
        else
            println("Array dimensions match, but it is not symmetric")
            println(max_dif)
            return A
        end
    else
        println("Array dimensions do not match. The matrix can't be symmetric")
        return A
    end
end


# # Returns the type of cholesky factorization process that is desired
# # either the raw values of the matrix or the factorization type
# # depreciated in 1.0
# function genernal_chol(A; return_values=false)
#
#     factor = cholfact(A, :L)
#     if return_values
#         return LowerTriangular(factor[:L])
#     else
#         return factor
#     end
# end


# adds a ridge to a Cholesky factorization if necessary
function ridge_chol(A; notification=true, ridge=1e-10)

    # only add a small ridge if necessary
    try
        factor = cholesky(A)
    catch
        if notification
            println("had to add a ridge")
        end
        # factor = genernal_chol(A + ridge * eye(size(A, 1)), return_values=return_values)  # depreciated in 1.0
        factor = cholesky(A + UniformScaling(ridge))
    end

end


# gets the coefficients and differentiation orders necessary for two multiplied
# functions with an arbitrary amount of parameters
function product_rule(dorder)

    # initializing the final matrix with a single combined function with no derivatives
    total = transpose(append!([1], zeros(2 * length(dorder))))

    # Do each parameter individually
    for i in 1:length(dorder)

        # create a holding matrix so we don't have to modify the total matrix
        hold = zeros(0, size(total, 2))

        # assign the coefficients and dorders for each combination of dorders
        for j in 0:dorder[i]

            # copying the current state of the total matrix
            total_copy = copy(total)

            # modifying the current coefficient by the proper amount
            total_copy[:, 1] = total_copy[:, 1] * binomial(dorder[i], j)

            # setting the new dorders for this parameter
            # total_copy[:, i + 1] = dorder[i] - j  # depreciated in 1.0
            # total_copy[:, i + 1 + length(dorder)] = j  # depreciated in 1.0
            current_length = size(total_copy, 1)
            total_copy[:, i + 1] = (dorder[i] - j) * ones(current_length)
            total_copy[:, i + 1 + length(dorder)] = j * ones(current_length)

            # add this to the holding matrix
            hold = vcat(hold, total_copy)
        end

        # update the total matrix
        total = hold

    end

    # total is of the following format
    # (coefficient, amount of parameters *
    # (dorder applied to first function for that parameter, dorder applied to second function))
    return total
end


# convert all floats in an array to ints
function floats2ints(v; allow_negatives=true)
    for i in length(v)
        if v[i] < 0 & !allow_negatives
            v[i] = 0
            println("one of your floats was negative!")
        else
            v[i] = convert(Int64, v[i])
        end
    end
    return v
end


# find differences between two arrays and set values smaller than a threshold
# to be zero
# use isapprox instead if you care about whether about boolean result
function signficant_difference(A1, A2, dif)
    A1mA2 = abs.(A1 - A2);
    A1mA2[A1mA2 .< (max(A1, A2) * ones(size(A1mA2)) * dif)] = 0;
    # A1mA2[A1mA2 .< (ones(size(A1mA2)) * sqrt(dif))] = 0;
    return A1mA2
end


# function similar to Mathematica's Chop[]
function chop_array(A; dif = 1e-6)
    A[abs.(A) .< dif] = 0;
    return A
end


# return approximate derivatives based on central differences
# a bit finicky based on h values
# f is the function of x
# x is the place you want the derivative at
# n is the order of derivative
# h is the step size
function finite_differences(f, x, n, h)
    return sum([(-1) ^ i * binomial(n, i) * f(x + (n / 2 - i) * h) for i in 0:n] / h ^ n)
end


# linspace(start, stop, length) = range(start, stop=stop, length=length)
