using JLD2, FileIO

old_dir = pwd()
cd(@__DIR__)
@load "../jld2_files/sample_problem_def.jld2" sample_problem_def normals
cd(old_dir)

# how many components you will use
n_out = sample_problem_def.n_out
# how many differentiated versions of the original GP you will use
n_dif = sample_problem_def.n_dif

amount_of_samp_points = 100
amount_of_total_samp_points = amount_of_samp_points * n_out

# amount_of_samp_points random observations over a 120 day period
x_obs = sort(120 .* rand(amount_of_samp_points))
x_obs_units = "days"
a0 = zeros(n_out, n_dif)

# best fit params of AIC 1 jones model from a subsection of SOAP 2.0 data saved in sample_problem_def.jld2 (with wrong size spots but whatever)
# [0.222321, 1.48745, 0.0, 2.59429, 0.0, 2.56856, 0.0, -0.899077, 0.0, 1.27324, 4.25597, 2.12614]  # compare this to total_hyperparameters
a0[1,1] = 0.22; a0[2,1] = 1.49; a0[1,2] = 2.59; a0[3,2] = 2.57; a0[2,3] = -0.90; a0

num_kernel_hyperparameters = include_kernel(string(sample_problem_def.kernel))    # sets correct num_kernel_hyperparameters
temp_prob_def = build_problem_definition(sample_problem_def.kernel, sample_problem_def.n_kern_hyper, n_dif, n_out, x_obs, x_obs_units, a0)

##########################################################

# kernel hyper parameters
kernel_lengths = [1.27, 4.26, 2.13]
total_hyperparameters = append!(collect(Iterators.flatten(temp_prob_def.a0)), kernel_lengths)

L_samp = covariance(temp_prob_def, x_obs, x_obs, total_hyperparameters, chol=true).L
y_obs = (L_samp * randn(amount_of_total_samp_points))
# normals = [std(y_obs[((i - 1) * amount_of_samp_points + 1):(i * amount_of_samp_points)]) for i in 1:n_out]

# measurement_noise = zeros(amount_of_total_samp_points)
# for i in 1:n_out
#     measurement_noise[((i - 1) * amount_of_samp_points + 1):(i * amount_of_samp_points)] .= 0.10 * maximum(abs.(y_obs[i, :]))
# end
amount_of_measurements = length(sample_problem_def.x_obs)
measurement_noise_mag = [mean(sample_problem_def.noise[((i - 1) * amount_of_measurements + 1):(i * amount_of_measurements)]) for i in 1:n_out]
measurement_noise = zeros(amount_of_total_samp_points)
for i in 1:n_out
    measurement_noise[((i - 1) * amount_of_samp_points + 1):(i * amount_of_samp_points)] .= measurement_noise_mag[i]
end
y_obs += measurement_noise .* randn(amount_of_total_samp_points)

problem_def_528 = build_problem_definition(sample_problem_def.kernel, sample_problem_def.n_kern_hyper, n_dif, n_out, x_obs, x_obs_units, y_obs, "Normalized RV + PCA scores", measurement_noise, a0)

cd(@__DIR__)
@save "../jld2_files/problem_def_528.jld2" problem_def_528 normals
cd(old_dir)
