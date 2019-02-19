using JLD2, FileIO

# how many components you will use
n_out = 3
# how many differentiated versions of the original GP you will use
n_dif = 3

amount_of_samp_points = 100
amount_of_total_samp_points = amount_of_samp_points * n_out

x_obs = sort(10 .* rand(amount_of_samp_points))

a0 = zeros(n_out, n_dif)
a0[1,1] = -1; a0[2,1] = -1.5; a0[1,2] = -2; a0[3,2] = -2; a0[2,3] = 0.2; a0    #    /= 20

num_kernel_hyperparameters = include_kernel("Quasi_periodic_kernel")    # sets correct num_kernel_hyperparameters
temp_prob_def = build_problem_definition(Quasi_periodic_kernel, num_kernel_hyperparameters, n_dif, n_out, x_obs, a0)

##########################################################

# kernel hyper parameters
kernel_lengths = [0.6, 2, 2.5]
total_hyperparameters = append!(collect(Iterators.flatten(problem_definition.a0)), kernel_lengths)

K_samp = symmetric_A(covariance(temp_prob_def, x_obs, x_obs, total_hyperparameters))
L_samp = ridge_chol(K_samp).L
y_obs = (L_samp * randn(amount_of_total_samp_points))
normals = [mean(abs.(y_obs[((i - 1) * amount_of_samp_points + 1):(i * amount_of_samp_points)])) for i in 1:n_out]

for i in 1:n_out
    y_obs[((i - 1) * amount_of_samp_points + 1):(i * amount_of_samp_points)] /= normals[i]
end

measurement_noise = zeros(amount_of_total_samp_points)
for i in 1:n_out
    measurement_noise[((i - 1) * amount_of_samp_points + 1):(i * amount_of_samp_points)] .= 0.10 * maximum(abs.(y_obs[i, :]))
end

y_obs += measurement_noise .* randn(amount_of_total_samp_points)
problem_def_528 = build_problem_definition(Quasi_periodic_kernel, num_kernel_hyperparameters, n_dif, n_out, x_obs, y_obs, measurement_noise, a0)
@save "jld2_files/problem_def_528.jld2" problem_def_528 normals
