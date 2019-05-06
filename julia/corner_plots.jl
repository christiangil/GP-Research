length(ARGS)>0 ? user_input = parse(Int, ARGS[1]) : user_input = 1

possible_labels = [L"a_{11}" L"a_{21}" L"a_{12}" L"a_{32}" L"a_{23}" L"\lambda_{SE}" L"\lambda_{P}" L"\tau_P";
    L"a_{11}" L"a_{21}" L"a_{12}" L"a_{32}" L"a_{23}" L"\lambda_{SE}" L" " L" ";
    L"a_{11}" L"a_{21}" L"a_{12}" L"a_{32}" L"a_{23}" L"\alpha" L"\lambda_{RQ}" L" ";
    L"a_{11}" L"a_{21}" L"a_{12}" L"a_{32}" L"a_{23}" L"\lambda_{M52}" L" " L" "]
kernel_names = ["quasi_periodic_kernel", "se_kernel", "rq_kernel", "matern52_kernel"]

kernel_name = kernel_names[user_input]
@load "jld2_files/optimize_Jones_model_$kernel_name.jld2" current_params
prep_parallel_covariance(kernel_name)
actual_labels = possible_labels[user_input, 1:length(current_params)]
f_corner(input) = nlogL_Jones(problem_definition, input)
corner_plot(f_corner, data(current_params), "figs/gp/$kernel_name/corner_$kernel_name.png"; input_labels=actual_labels)
