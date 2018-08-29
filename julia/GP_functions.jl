using SpecialFunctions


# Linear kernel
function Linear_kernel(hyperparameters, x1, x2)
    sigma_b, sigma_a = hyperparameters
    return sigma_b ^ 2 * vecdot(x1, x2) + sigma_a ^ 2
end


# Radial basis function kernel (aka squared exonential, ~gaussian)
function RBF_kernel(hyperparameters, dif_sq)

    if length(hyperparameters) > 1
        kernel_amplitude, kernel_length = hyperparameters
    else
        kernel_amplitude = 1
        kernel_length = hyperparameters
    end

    return kernel_amplitude ^ 2 * exp(-dif_sq / (2 * kernel_length ^ 2))
end


# Differentiated Radial basis function kernel (aka squared exonential, ~gauss)
# ONLY WORKS FOR 1D TIME AS INPUTS
function dRBFdt_kernel(hyperparameters, dif, dorder)

    dorder = floats2ints(dorder; allow_negatives=false)

    if length(hyperparameters) > 1
        kernel_amplitude, kernel_length = hyperparameters
    else
        kernel_length = hyperparameters
    end

    RBF = RBF_kernel(hyperparameters, dif ^ 2)

    sum_dorder = sum(dorder)

    # first coefficients are triangular numbers. second coefficients are
    # tri triangular numbers
    if sum_dorder > 0

        T1 = dif / kernel_length ^ 2
        if sum_dorder == 1
            value = (T1)
        elseif sum_dorder == 2
            value = (T1 ^ 2 - 1 / kernel_length ^ 2)
        elseif sum_dorder == 3
            value = (T1 ^ 3 - 3 * T1 / (kernel_length ^ 2))
        elseif sum_dorder == 4
            value = (T1 ^ 4 - 6 * T1 ^ 2 / (kernel_length ^ 2)
                + 3 / (kernel_length ^ 4))
        end

        if isodd(convert(Int64, dorder[1]))
            value = -value
        end

        return RBF * value

    else

        return RBF
    end

end


# kernel length differentiated Radial basis function kernel (aka squared exonential, ~gauss)
# ONLY WORKS FOR 1D TIME AS INPUTS
function dRBFdλ_kernel(hyperparameters, dif, dorder)

    dorder = floats2ints(dorder; allow_negatives=false)

    if length(hyperparameters) > 1
        kernel_amplitude, kernel_length = hyperparameters
    else
        kernel_length = hyperparameters
    end

    RBF = RBF_kernel(hyperparameters, dif ^ 2)

    sum_dorder = sum(dorder)

    T1 = dif / kernel_length ^ 2
    if sum_dorder == 0
        value = (T1 ^ 2)
    elseif sum_dorder == 1
        value = (T1 ^ 3 - 2 * T1 / (kernel_length ^ 2))
    elseif sum_dorder == 2
        value = (T1 ^ 4 - 5 * T1 ^ 2 / (kernel_length ^ 2)
            + 2 / (kernel_length ^ 4))
    elseif sum_dorder == 3
        value = (T1 ^ 5 - 9 * T1 ^ 3 / (kernel_length ^ 2)
            + 12 * T1 / (kernel_length ^ 4))
    elseif sum_dorder == 4
        value = (T1 ^ 6 - 14 * T1 ^ 4 / (kernel_length ^ 2)
            + 39 * T1 ^ 2 / (kernel_length ^ 4) - 12 / (kernel_length ^ 6))
    end

    if isodd(convert(Int64, dorder[1]))
        value = -value
    end

    return RBF * kernel_length * value

end


# Ornstein–Uhlenbeck (Exponential) kernel
function OU_kernel(hyperparameters, dif)
    kernel_amplitude, kernel_length = hyperparameters
    return kernel_amplitude ^ 2 * exp(-dif / kernel_length)
end


# Periodic kernel
function Periodic_kernel(hyperparameters, abs_dif)

    if length(hyperparameters) > 2
        kernel_amplitude, kernel_length, kernel_period = hyperparameters
    else
        kernel_amplitude = 1
        kernel_length, kernel_period = hyperparameters
    end

    return kernel_amplitude ^ 2 * exp(-2 * sin(pi * abs_dif / kernel_period) ^ 2 / (kernel_length ^ 2))
end


# Differentiated periodic kernel
# ONLY WORKS FOR 1D TIME AS INPUTS
function dPeriodicdt_kernel(hyperparameters, dif, dorder)

    dorder = floats2ints(dorder; allow_negatives=false)

    if length(hyperparameters) > 2
        kernel_amplitude, kernel_length, kernel_period = hyperparameters
    else
        kernel_length, kernel_period = hyperparameters
    end

    Periodic = Periodic_kernel(hyperparameters, abs(dif))

    sum_dorder = sum(dorder)

    if sum_dorder > 0

        theta = 2 * pi * dif / kernel_period
        Sin = sin(theta)
        Cos_tab = [cos(i * theta) for i in 1:(2 * floor(sum_dorder/2))]

        if sum_dorder == 1
            value = (Sin)
        elseif sum_dorder == 2
            value = -1 * (-1 + 2 * kernel_length ^ 2 * Cos_tab[1] + Cos_tab[2])
        elseif sum_dorder == 3
            value = (-2 * Sin * (-1 + 2 * kernel_length ^ 4
                + 6 * kernel_length ^ 2 * Cos_tab[1] + Cos_tab[2]))
        elseif sum_dorder == 4
            value = (3 - 4 * kernel_length ^ 4
                + 4 * kernel_length ^ 2 * (-3 + 2 * kernel_length ^ 4) * Cos_tab[1]
                + 4 * (-1 + 7 * kernel_length ^ 4) * Cos_tab[2]
                + 12 * kernel_length ^ 2 * Cos_tab[3]
                + Cos_tab[4])
        end
        if isodd(convert(Int64, dorder[1]))
            value = -value
        end

        constant = (pi / (kernel_period * kernel_length ^ 2)) ^ sum_dorder
        return 2 * constant * Periodic * value
    else
        return Periodic
    end

end


# kernel length differentiated periodic kernel
# ONLY WORKS FOR 1D TIME AS INPUTS
function dPeriodicdλ_kernel(hyperparameters, dif, dorder)

    dorder = floats2ints(dorder; allow_negatives=false)

    if length(hyperparameters) > 2
        kernel_amplitude, kernel_length, kernel_period = hyperparameters
    else
        kernel_length, kernel_period = hyperparameters
    end

    Periodic = Periodic_kernel(hyperparameters, abs(dif))

    sum_dorder = sum(dorder)

    theta = pi * dif / kernel_period
    Sin_tab = [sin(i * theta) for i in 1:maximum([2, sum_dorder + 1])]
    Cos_tab = [cos(2 * i * theta) for i in 1:(1 + 2 * floor(sum_dorder/2))]

    if sum_dorder == 0
        value = (Sin_tab[1] ^ 2)
    elseif sum_dorder == 1
        value = (-Sin_tab[2] * (-1 + kernel_length ^ 2 + Cos_tab[1]))
    elseif sum_dorder == 2
        value = (1 / 2 * (2 - 2 * kernel_length ^ 2
            + (-1 - 4 * kernel_length ^ 2 + 4 * kernel_length ^ 4) * Cos_tab[1]
            + (-2 + 6 * kernel_length ^ 2) * Cos_tab[2]
            + Cos_tab[3]))
    elseif sum_dorder == 3
        value = (Sin_tab[2] * (2 - 4 * kernel_length ^ 4 + 4 * kernel_length ^ 6
            + (-1 - 12 * kernel_length ^ 2 + 28 * kernel_length ^ 4) * Cos_tab[1]
            + 2 * (-1 + 6 * kernel_length ^ 2) * Cos_tab[2]
            + Cos_tab[3]))
    elseif sum_dorder == 4
        value = (-1 / 2 * (-6 + 12 * kernel_length ^ 2 + 8 * kernel_length ^ 4 - 8 * kernel_length ^ 6
            + 2 * (1 + 12 * kernel_length ^ 2 - 26 * kernel_length ^ 4 - 8 * kernel_length ^ 6 + 8 * kernel_length ^ 8) * Cos_tab[1]
            + 8 * (1 - 4 * kernel_length ^ 2 - 7 * kernel_length ^ 4 + 15 * kernel_length ^ 6) * Cos_tab[2]
            + (-3 - 24 * kernel_length ^ 2 + 100 * kernel_length ^ 4) * Cos_tab[3]
            + 2 * (-1 + 10 * kernel_length ^ 2) * Cos_tab[4]
            + Cos_tab[5]))
    end

    if isodd(convert(Int64, dorder[1]))
        value = -value
    end

    constant = (pi / (kernel_period * kernel_length ^ 2)) ^ sum_dorder
    return 4 / kernel_length ^ 3 * constant * Periodic * value

end


# kernel period differentiated periodic kernel
# ONLY WORKS FOR 1D TIME AS INPUTS
function dPeriodicdp_kernel(hyperparameters, dif, dorder)

    dorder = floats2ints(dorder; allow_negatives=false)

    if length(hyperparameters) > 2
        kernel_amplitude, kernel_length, kernel_period = hyperparameters
    else
        kernel_length, kernel_period = hyperparameters
    end

    Periodic = Periodic_kernel(hyperparameters, abs(dif))

    sum_dorder = sum(dorder)

    theta = 2 * pi *dif / kernel_period
    Sin_tab = [sin(i * theta) for i in 1:maximum([3,(1 + 2 * floor(sum_dorder/2))])]
    Cos_tab = [cos(i * theta) for i in 1:(2 * floor(sum_dorder/2 + 1))]

    if sum_dorder == 0
        value = (pi * dif * Sin_tab[1])
    elseif sum_dorder == 1
        value = (-1 * (pi * dif * (-1 + 2 * kernel_length ^ 2 * Cos_tab[1] + Cos_tab[2])
            + kernel_period * kernel_length ^ 2 * Sin_tab[1]))
    elseif sum_dorder == 2
        value = (-1 * (-2 * kernel_period * kernel_length ^ 2 * (-1 + 2 * kernel_length ^ 2 * Cos_tab[1] + Cos_tab[2])
            + 2 * pi * dif * (-1 + 2 * kernel_length ^ 4 + 6 * kernel_length ^ 2 * Cos_tab[1] + Cos_tab[2]) * Sin_tab[1]))
    elseif sum_dorder == 3
        value = (-pi * dif * (-3 + 4 * kernel_length ^ 4)
            + 4 * pi * dif * kernel_length ^ 2 * (-3 + 2 * kernel_length ^ 4) * Cos_tab[1]
            + 4 * pi * dif * (-1 + 7 * kernel_length ^ 4) * Cos_tab[2]
            + 12 * pi * dif * kernel_length ^ 2 * Cos_tab[3]
            + pi * dif * Cos_tab[4]
            + 3 * kernel_period * kernel_length ^ 2 * (-3 + 4 * kernel_length ^ 4) * Sin_tab[1]
            + 18 * kernel_period * kernel_length ^ 4 * Sin_tab[2]
            + 3 * kernel_period * kernel_length ^ 2 * Sin_tab[3])
    elseif sum_dorder == 4
        value = (-4 * kernel_period * kernel_length ^ 2 * (3
            - 4 * kernel_length ^ 4 + 4 * kernel_length ^ 2 * (-3 + 2 * kernel_length ^ 4) * Cos_tab[1]
            + 4 * (-1 + 7 * kernel_length ^ 4) * Cos_tab[2]
            + 12 * kernel_length ^ 2 * Cos_tab[3]
            + Cos_tab[4])
            + 2 * pi * dif * Sin_tab[1] * (3 + 20 * kernel_length ^ 4 + 8 * kernel_length ^ 8
            + 20 * kernel_length ^ 2 * (-1 + 6 * kernel_length ^ 4) * Cos_tab[1]
            + 4 * (-1 + 25 * kernel_length ^ 4) * Cos_tab[2]
            + 20 * kernel_length ^ 2 * Cos_tab[3]
            + Cos_tab[4]))
    end

    if isodd(convert(Int64, dorder[1]))
        value = -value
    end

    constant = (pi / (kernel_period * kernel_length ^ 2)) ^ sum_dorder
    return 2 / (kernel_period ^ 2 * kernel_length ^ 2) * constant * Periodic * value

end


# general Matern kernel
function Matern_kernel(hyperparameters, dif, nu)
    kernel_amplitude, kernel_length = hyperparameters
    #limit of the function as it apporaches 0 (see https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function)
    if dif == 0
        return kernel_amplitude ^ 2
    else
        x = (sqrt(2 * nu) * dif) / kernel_length
        return kernel_amplitude ^ 2 * ((2 ^ (1 - nu)) / (gamma(nu))) * x ^ nu * besselk(nu, x)
    end
end


# Matern 3/2 kernel
function Matern32_kernel(hyperparameters, dif, nu)
    kernel_amplitude, kernel_length = hyperparameters
    x = sqrt(3) * dif / kernel_length
    return kernel_amplitude ^ 2 * (1 + x) * exp(-x)
end


# Matern 5/2 kernel
function Matern52_kernel(hyperparameters, dif, nu)
    kernel_amplitude, kernel_length = hyperparameters
    x = sqrt(5) * dif / kernel_length
    return kernel_amplitude ^ 2 * (1 + x + (x ^ 2) / 3) * exp(-x)
end


# Rational Quadratic kernel (equivalent to adding together many SE kernels
# with different lengthscales. When α→∞, the RQ is identical to the SE.)
function RQ_kernel(hyperparameters, dif_sq)
    kernel_amplitude, kernel_length, alpha = hyperparameters
    alpha = max(alpha, 0)
    return kernel_amplitude ^ 2 * (1 + dif_sq / (2 * alpha * kernel_length ^ 2)) ^ -alpha
end


# Creates the covariance matrix by evaluating the kernel function for each pair
# of passed inputs. Generic. Complicated covariances accounted for in the
# total_covariance function
# the symmetric parameter asks whether or note the kernel used will be symmetric about d=0
function covariance(x1list, x2list, hyperparameters; dorder=[0, 0], symmetric=false, dKdθ=0, products=0)

    # are the list of x's passed identical
    same_x = (x1list == x2list)

    # are the x's passed identical and equally spaced
    spacing = [x1list[i]-x1list[i-1] for i in 2:length(x1list)]
    all([spacing[i]==spacing[1] for i in 2:length(spacing)])
    if same_x
        spacing = [x1list[i]-x1list[i-1] for i in 2:length(x1list)]
        equal_spacing = all([abs(spacing[i] - spacing[1]) < 1e-8 for i in 2:length(spacing)])
    else
        equal_spacing = false
    end

    x1_length = size(x1list, 1)
    x2_length = size(x2list, 1)
    K = zeros((x1_length, x2_length))

    if equal_spacing && symmetric
        kernline = zeros(x1_length)
        for i in 1:x1_length
            kernline[i] = kernel(hyperparameters, x1list[1,:], x1list[i,:], dorder=dorder, dKdθ=dKdθ, products=products)
        end
        for i in 1:x1_length
            for j in 1:x1_length
                if i <= j
                    K[i, j] = kernline[j - i + 1]
                end
            end
        end
        return Symmetric(K)
    elseif same_x && symmetric
        for i in 1:x1_length
            for j in 1:x1_length
                if i <= j
                    K[i, j] = kernel(hyperparameters, x1list[i,:], x1list[j,:], dorder=dorder, dKdθ=dKdθ, products=products)
                end
            end
        end
        return Symmetric(K)
    else
        for i in 1:x1_length
            for j in 1:x2_length
                K[i, j] = kernel(hyperparameters, x1list[i,:], x2list[j,:], dorder=dorder, dKdθ=dKdθ, products=products)
            end
        end
        return K
    end
end


# "true" underlying for the fake observations
function observations(x, measurement_noise)
    # a phase shifted sine curve with measurement noise and inherent noise
    measurement_noise += 0.2 * ones(length(measurement_noise))  # adding a noise component inherent to the activity
    if length(size(x)) > 1
        shift = 2 * pi * rand(size(x, 2))
        return [sum(sin.(pi / 2 * x[i,:] + [shift])) for i in 1:size(x, 1)] + measurement_noise .^ 2 .* randn(size(x, 1))
    else
        shift = 2 * pi * rand()
        return [sum(sin.(pi / 2 * x[i,:] + [shift])) for i in 1:length(x)] + measurement_noise .^ 2 .* randn(length(x))
    end
end


# adding measurement noise to K_obs
function K_observations(x_obs, measurement_noise, hyperparameters)
    K_obs = total_covariance(x_obs, x_obs, hyperparameters)
    total_amount_of_measurements = size(x_obs, 1)
    for i in 1:total_amount_of_measurements
        K_obs[i, i] +=  measurement_noise[i] ^ 2
    end
    # noise_I = zeros((total_amount_of_measurements, total_amount_of_measurements))
    # for i in 1:total_amount_of_measurements
    #     noise_I[i, i] =  measurement_noise[i] ^ 2
    # end
    # K_obs = K_obs + noise_I
    return symmetric_A(K_obs)
end


# calculating the variance at each point
function get_σ(L_obs, K_obs_samp, K_samp)

    v = L_obs \ K_obs_samp
    V = zeros(size(K_samp, 1))
    for i in 1:size(K_samp, 1)
        V[i] = K_samp[i, i] - transpose(v[:, i]) * v[:, i]
    end
    σ = sqrt.(V)
    return σ
end


# produces equivalent variances as the other version, but ~46% faster
function GP_posteriors(x_obs, x_samp, measurement_noise, hyperparameters; return_σ=false, return_K=false, return_L=false)


    return_vec = []

    K_samp = total_covariance(x_samp, x_samp, hyperparameters)
    K_obs = K_observations(x_obs, measurement_noise, hyperparameters)
    K_samp_obs = total_covariance(x_samp, x_obs, hyperparameters)
    # K_obs_samp = total_covariance(x_obs, x_samp, hyperparameters)
    K_obs_samp = transpose(K_samp_obs)

    # (RW alg. 2.1)

    # tells Julia to perform and store the Cholesky factorization
    L_fact = ridge_chol(K_obs)

    # actual lower triangular matrix values)
    # L = ridge_chol(K_obs, return_values=true)
    # L = LowerTriangular(L_fact[:L])  # depreciated in 1.0
    L = L_fact.L

    # these are all equivalent but have different computational costs
    # α = inv(L_fact) * y_obs
    # α = transpose(L) \ (L \ y_obs)
    α = L_fact \ y_obs

    mean_post = K_samp_obs * α

    # python implementation
    # # Compute the mean at our test points.
    # v = np.linalg.solve(L, K_obs_samp)
    # mean_post = np.dot(v.T, np.linalg.solve(L, y_obs)).reshape((n,))
    #
    # V = np.diag(K_samp) - np.sum(v**2, axis=0)
    # stdv = np.sqrt(V)

    if return_σ
        σ = get_σ(L, K_obs_samp, K_samp)
        append!(return_vec, [σ])
    end

    if return_L || return_K
        K_post = symmetric_A(K_samp - (K_samp_obs * (L_fact \ K_obs_samp)))
        if return_K
            append!(return_vec, [K_post])
        end
        if return_L
            L_post = ridge_chol(K_post).L
            append!(return_vec, [L_post])
        end
    end

    return mean_post, return_vec
end


# negative log likelihood of the data given the current kernel parameters (as seen on page 19)
# (negative because scipy has a minimizer instead of a maximizer)
function nlogL(hyperparameter_list...)

    hyperparameters = []
    for i in 1:length(hyperparameter_list)
        append!(hyperparameters, hyperparameter_list[i])
    end
    n=length(y_obs)

    # a weirdly necessary dummy variable
    measurement_noise_dummy = measurement_noise
    K_obs = K_observations(x_obs, measurement_noise_dummy, hyperparameters)

    # inv_K_obs = inv(K_obs)
    # inv_K_obs = inv(L_fact)  # ~35% faster than inv(K_obs)
    # det_K_obs = det(L_fact)  # ~8% faster than det(K_obs)
    L_fact = stored_chol(chol_storage, hyperparameters, K_obs; notification=false)

    # goodness of fit term
    data_fit = -1 / 2 * (transpose(y_obs) * (L_fact \ y_obs))
    # complexity penalization term
    # penalty = -1 / 2 * log(det_K_obs)
    penalty = -1 / 2 * logdet(L_fact)  # half memory but twice the time
    # normalization term (functionally useless)
    normalization = -n / 2 * log(2 * pi)

    return -1 * (data_fit + penalty + normalization)
end


# http://www.gaussianprocess.org/gpml/chapters/RW5.pdf
# gradient of negative log likelihood of the data given the current kernel parameters (as seen on page 19)
# (negative because scipy has a minimizer instead of a maximizer)
function ∇nlogL(G, hyperparameter_list...)

    hyperparameters = []
    for i in 1:length(hyperparameter_list)
        append!(hyperparameters, hyperparameter_list[i])
    end

    # a weirdly necessary dummy variable
    measurement_noise_dummy = measurement_noise
    K_obs = K_observations(x_obs, measurement_noise_dummy, hyperparameters)
    L_fact = stored_chol(chol_storage, hyperparameters, K_obs; notification=false)
    # inv_K_obs = inv(L_fact)


    function grad(dK_dθj)
        # derivative of goodness of fit term
        data_fit = 1 / 2 * (transpose(y_obs) * (L_fact \ (dK_dθj * (L_fact \ y_obs))))
        # derivative of complexity penalization term
        penalty = -1 / 2 * tr(L_fact \ dK_dθj)
        return -1 * (data_fit + penalty)
    end

    # taking some burden off of recalculating the coefficient orders used by the automatic differentiation
    coeff_orders = coefficient_orders(n_out, n_dif)

    for i in 1:(length(hyperparameters))
        G[i] = grad(total_covariance(x_obs, x_obs, hyperparameters; dKdθ=i, coeff_orders=coeff_orders))
    end

end


# negative log likelihood of the data given the current kernel parameters (as seen on page 19)
# (negative because scipy has a minimizer instead of a maximizer)
function nlogL_penalty(hyperparameter_list...)

    hyperparameters = []
    for i in 1:length(hyperparameter_list)
        append!(hyperparameters, hyperparameter_list[i])
    end
    n=length(y_obs)

    # a weirdly necessary dummy variable
    measurement_noise_dummy = measurement_noise
    K_obs = K_observations(x_obs, measurement_noise_dummy, hyperparameters)

    # inv_K_obs = inv(K_obs)
    # inv_K_obs = inv(L_fact)  # ~35% faster than inv(K_obs)
    # det_K_obs = det(L_fact)  # ~8% faster than det(K_obs)
    L_fact = stored_chol(chol_storage, hyperparameters, K_obs; notification=false)

    # goodness of fit term
    data_fit = -1 / 2 * (transpose(y_obs) * (L_fact \ y_obs))
    # complexity penalization term
    # penalty = -1 / 2 * log(det_K_obs)
    penalty = -1 / 2 * logdet(L_fact)  # half memory but twice the time
    # normalization term (functionally useless)
    normalization = -n / 2 * log(2 * pi)

    # lower = zeros(length(hyperparameters))
    lower = -10 * ones(length(hyperparameters))
    lower[(total_coefficients + 1):length(hyperparameters)] = 0
    upper = 10 * ones(length(hyperparameters))

    add_penalty = 0
    for i in 1:length(hyperparameters)
        if hyperparameters[i] > upper[i]
            add_penalty += hyperparameters[i] - upper[i]
        elseif hyperparameters[i] < lower[i]
            add_penalty += lower[i] - hyperparameters[i]
        end
    end

    return -1 * (data_fit + penalty + normalization) + add_penalty
end


# http://www.gaussianprocess.org/gpml/chapters/RW5.pdf
# gradient of negative log likelihood of the data given the current kernel parameters (as seen on page 19)
# (negative because scipy has a minimizer instead of a maximizer)
function ∇nlogL_penalty(G, hyperparameter_list...)

    hyperparameters = []
    for i in 1:length(hyperparameter_list)
        append!(hyperparameters, hyperparameter_list[i])
    end

    # a weirdly necessary dummy variable
    measurement_noise_dummy = measurement_noise
    K_obs = K_observations(x_obs, measurement_noise_dummy, hyperparameters)
    L_fact = stored_chol(chol_storage, hyperparameters, K_obs; notification=false)
    # inv_K_obs = inv(L_fact)

    # lower = zeros(length(hyperparameters))
    lower = -10* ones(length(hyperparameters))
    lower[(total_coefficients + 1):length(hyperparameters)] = 0
    upper = 10 * ones(length(hyperparameters))

    function grad(dK_dθj)
        # derivative of goodness of fit term
        data_fit = 1 / 2 * (transpose(y_obs) * (L_fact \ (dK_dθj * (L_fact \ y_obs))))
        # derivative of complexity penalization term
        penalty = -1 / 2 * tr(L_fact \ dK_dθj)

        add_penalty = 0
        for i in 1:length(hyperparameters)
            if hyperparameters[i] > upper[i]
                add_penalty = 1
            elseif hyperparameters[i] < lower[i]
                add_penalty = -1
            end
        end

        return -1 * (data_fit + penalty) + add_penalty
    end

    # taking some burden off of recalculating the coefficient orders used by the automatic differentiation
    coeff_orders = coefficient_orders(n_out, n_dif)

    for i in 1:(length(hyperparameters))
        G[i] = grad(total_covariance(x_obs, x_obs, hyperparameters; dKdθ=i, coeff_orders=coeff_orders))
    end


end


# creating a Cholesky factorization storage structure
struct chol_struct
    hyperparameters
    cholesky_object
end


# An effort to avoid recalculating Cholesky factorizations
function stored_chol(chol_storage, new_hyper, A; return_values=false, notification=true)

    # if the Cholesky factorization wasn'y just calculated, calculate a new one.
    if chol_storage.hyperparameters != new_hyper
        chol_storage = chol_struct(new_hyper, ridge_chol(A; notification=notification))
    end
    return chol_storage.cholesky_object

end


# find the powers that each coefficient is taken to for each part of the matrix construction
# used for constructing differentiated versions of the kernel
function coefficient_orders(n_out, n_dif; a=ones(n_out, n_dif))

    coeff_orders = zeros(n_out,n_out,n_dif,n_dif,n_out,n_dif)
    for i in 1:n_out
        for j in 1:n_out
            for k in 1:n_dif
                for l in 1:n_dif
                    if (i == j) & isodd(k + l)
                        # the cross terms (of odd differentiation orders) cancel each other out in diagonal matrices
                    else
                        for m in 1:n_out
                            for n in 1:n_dif
                                if a[m, n] != 0
                                    if [m, n] == [i, k]
                                        coeff_orders[i, j, k, l, m, n] += 1
                                    end
                                    if [m, n] == [j, l]
                                        coeff_orders[i, j, k, l, m, n] += 1
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    return coeff_orders

end


# a small function to get indices that make sense from Julia's reshaping routine
proper_index(i) = [convert(Int64, rem(i - 1, n_out)) + 1, convert(Int64, floor((i -1) / n_out)) + 1]


# getting the coefficients for constructing differentiated versions of the kernel
# using the powers that each coefficient is taken to for each part of the matrix construction
function dif_coefficients(n_out, n_dif, dKdθ; coeff_orders=0, a=ones(n_out, n_dif))

    if coeff_orders == 0
        coeff_orders = coefficient_orders(n_out, n_dif, a=a)
    end

    # ((output pair), (which A matrix to use), (which a coefficent to use))
    coeff = zeros(n_out,n_out,n_dif,n_dif,n_out,n_dif)
    proper_indices = proper_index(dKdθ)
    for i in 1:n_out
        for j in 1:n_out
            for k in 1:n_dif
                for l in 1:n_dif
                    if coeff_orders[i, j, k, l, proper_indices[1], proper_indices[2]] == 1
                        coeff[i, j, k, l, :, :] = coeff_orders[i, j, k, l, :, :]
                        coeff[i, j, k, l, proper_indices[1], proper_indices[2]] = 0
                    elseif coeff_orders[i, j, k, l, proper_indices[1], proper_indices[2]] == 2
                        coeff[i, j, k, l, :, :] = coeff_orders[i, j, k, l, :, :]
                    end
                end
            end
        end
    end

    return coeff
end


# old handcoded version of dif_coefficients function
# function dif_coefficients_old(n_out, n_dif, dKdθ)
#     # ((output pair), (which A matrix to use), (which a coefficent to use))
#     coeff = zeros(n_out, n_out, n_dif, n_dif, n_out, n_dif)
#     if (n_out == 2) & (n_dif > 1)
#         if dKdθ == 1  # dK/da11
#             coeff[1, 1, 1, 1, 1, 1] = 2
#
#             coeff[1, 2, 1, 1, 2, 1] = 1
#             coeff[1, 2, 1, 2, 2, 2] = 1
#
#             coeff[2, 1, 1, 1, 2, 1] = 1
#             coeff[2, 1, 2, 1, 2, 2] = 1
#         elseif dKdθ == 2  # dK/da21
#             coeff[1, 2, 1, 1, 1, 1] = 1
#             coeff[1, 2, 2, 1, 1, 2] = 1
#
#             coeff[2, 1, 1, 1, 1, 1] = 1
#             coeff[2, 1, 1, 2, 1, 2] = 1
#
#             coeff[2, 2, 1, 1, 2, 1] = 2
#         elseif dKdθ == 3  # dK/da12
#             coeff[1, 1, 2, 2, 1, 2] = 2
#
#             coeff[1, 2, 2, 1, 2, 1] = 1
#             coeff[1, 2, 2, 2, 2, 2] = 1
#
#             coeff[2, 1, 1, 2, 2, 1] = 1
#             coeff[2, 1, 2, 2, 2, 2] = 1
#         elseif dKdθ == 4  # dK/da22
#             coeff[1, 2, 1, 2, 1, 1] = 1
#             coeff[1, 2, 2, 2, 1, 2] = 1
#
#             coeff[2, 1, 2, 1, 1, 1] = 1
#             coeff[2, 1, 2, 2, 1, 2] = 1
#
#             coeff[2, 2, 2, 2, 2, 2] = 2
#         end
#         if n_dif > 2
#             if dKdθ == 1  # dK/da11
#                 coeff[1, 1, 3, 1, 1, 3] = 1
#                 coeff[1, 1, 1, 3, 1, 3] = 1
#
#                 coeff[1, 2, 1, 3, 2, 3] = 1
#
#                 coeff[2, 1, 3, 1, 2, 3] = 1
#             elseif dKdθ == 2  # dK/da21
#                 coeff[1, 2, 3, 1, 1, 3] = 1
#
#                 coeff[2, 1, 1, 3, 1, 3] = 1
#
#                 coeff[2, 2, 3, 1, 2, 3] = 1
#                 coeff[2, 2, 1, 3, 2, 3] = 1
#             elseif dKdθ == 3  # dK/da12
#                 coeff[1, 2, 2, 3, 2, 3] = 1
#
#                 coeff[2, 1, 3, 2, 2, 3] = 1
#             elseif dKdθ == 4  # dK/da22
#                 coeff[1, 2, 3, 2, 1, 3] = 1
#
#                 coeff[2, 1, 2, 3, 1, 3] = 1
#             elseif dKdθ == 5  # dK/da13
#                 coeff[1, 1, 3, 3, 1, 3] = 2
#                 coeff[1, 1, 3, 1, 1, 1] = 1
#                 coeff[1, 1, 1, 3, 1, 1] = 1
#
#                 coeff[1, 2, 3, 1, 2, 1] = 1
#                 coeff[1, 2, 3, 2, 2, 2] = 1
#                 coeff[1, 2, 3, 3, 2, 3] = 1
#
#                 coeff[2, 1, 1, 3, 2, 1] = 1
#                 coeff[2, 1, 2, 3, 2, 2] = 1
#                 coeff[2, 1, 3, 3, 2, 3] = 1
#             elseif dKdθ == 6  # dK/da23
#                 coeff[1, 2, 1, 3, 1, 1] = 1
#                 coeff[1, 2, 2, 3, 1, 2] = 1
#                 coeff[1, 2, 3, 3, 1, 3] = 1
#
#                 coeff[2, 1, 3, 1, 1, 1] = 1
#                 coeff[2, 1, 3, 2, 1, 2] = 1
#                 coeff[2, 1, 3, 3, 1, 3] = 1
#
#                 coeff[2, 2, 3, 3, 2, 3] = 2
#                 coeff[2, 2, 3, 1, 2, 1] = 1
#                 coeff[2, 2, 1, 3, 2, 1] = 1
#             end
#         end
#     end
#     return coeff
# end
