include("src/all_functions.jl")

possible_labels = [L"a_{11}" L"a_{21}" L"a_{12}" L"a_{32}" L"a_{23}" L"\lambda_{SE}" L"\lambda_{P}" L"\tau_P";
    L"a_{11}" L"a_{21}" L"a_{12}" L"a_{32}" L"a_{23}" L"\lambda_{SE}" L" " L" ";
    L"a_{11}" L"a_{21}" L"a_{12}" L"a_{32}" L"a_{23}" L"\alpha" L"\lambda_{RQ}" L" ";
    L"a_{11}" L"a_{21}" L"a_{12}" L"a_{32}" L"a_{23}" L"\lambda_{M52}" L" " L" "]
kernel_names = ["quasi_periodic_kernel", "se_kernel", "rq_kernel", "matern52_kernel"]

if length(ARGS)>0
    user_input = parse(Int, ARGS[1])
    kernel_name = kernel_names[user_input]
    @load "jld2_files/problem_def_full_base.jld2" problem_def_base normals
    kernel_function, num_kernel_hyperparameters = include_kernel(kernel_name)
    problem_definition = build_problem_definition(kernel_function, num_kernel_hyperparameters, problem_def_base)
else
    user_input = 1
    kernel_name = kernel_names[user_input]
    @load "jld2_files/problem_def_sample_base.jld2" problem_def_base normals
    kernel_function, num_kernel_hyperparameters = include_kernel(kernel_name)
    problem_definition = build_problem_definition(kernel_function, num_kernel_hyperparameters, problem_def_base)
end

@load "jld2_files/optimize_Jones_model_$kernel_name.jld2" current_params
actual_labels = possible_labels[user_input, 1:length(current_params)]

prep_parallel_covariance(kernel_name)

f_corner(input) = nlogL_Jones(problem_definition, input)

@elapsed corner_plot(f_corner, current_params, "figs/gp/$kernel_name/corner_$kernel_name.png"; input_labels=actual_labels)
