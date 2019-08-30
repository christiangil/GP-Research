# errors
# include("src/setup.jl")
include("src/all_functions.jl")

kernel_names = ["quasi_periodic_kernel", "se_kernel", "matern52_kernel", "rq_kernel"]
nice_kernel_names = ["Quasi-Periodic", "Squared Exponential", "Matérn " * L"^5/_2", "Rational Quadratic"]


function get_df(kernel_name)
    file_name = "csv_files/$(kernel_name)_logLs.csv"
    df = CSV.read(file_name)
end

function return_logL(kernel_name; filter::Bool=true)
    df = get_df(kernel_name)
    logL = -df.nLogL
    if filter; logL = logL[logL .> -600] end
    if filter; logL = logL[logL .< -200] end
    return logL
end

# plot each kernel dist
for i in 1:3  # length(kernel_names)
    kernel_name = kernel_names[i]
    logL = return_logL(kernel_name)
    ax = init_plot()
    hs = hist(logL; density=true, bins=convert(Int64, floor(length(logL)/10)))
    # ax.axvline(x=-300 / 2 * (log(2 * π) + 1), color="black", linewidth=4)
    l_act_theta = L"\ell_{act}(\theta|t,s)"
    xlabel(l_act_theta)
    title("$(nice_kernel_names[i]) " * l_act_theta * ", n = 100 x 3", fontsize=45)
    text(minimum(hs[2]), 0.9 * maximum(hs[1]), L"\overline{\ell_{act}}: " * string(convert(Int64, round(mean(logL)))), fontsize=40)
    text(minimum(hs[2]), 0.8 * maximum(hs[1]), L"\sigma_{\ell_{act}}: " * string(convert(Int64, round(std(logL)))), fontsize=40)
    save_PyPlot_fig("$(kernel_name)_test")
end

# plot differences from qp
begin
    ax = init_plot()
    logL1 = return_logL(kernel_names[1]; filter=false)
    logL2 = return_logL(kernel_names[2]; filter=false)
    hist(logL2-logL1; density=true, bins=convert(Int64, floor(length(logL2)/10)), label=nice_kernel_names[2], alpha=0.5)
    logL3 = return_logL(kernel_names[3]; filter=false)
    hist(logL3-logL1; density=true, bins=convert(Int64, floor(length(logL3)/10)), label=nice_kernel_names[3], alpha=0.5)
    l_act_theta = L"\ell_{act}(\theta|t,s)"
    xlabel(l_act_theta * " difference")
    legend(;fontsize=30)
    title("Comparing " * l_act_theta * " difference between kernels", fontsize=45)
    save_PyPlot_fig("difference_test")
end

# shared plot
begin
    ax = init_plot()
    logL2 = return_logL(kernel_names[2])
    hist(logL2; density=true, bins=convert(Int64, floor(length(logL2)/10)), label=nice_kernel_names[2], alpha=0.3)
    logL1 = return_logL(kernel_names[1])
    hist(logL1; density=true, bins=convert(Int64, floor(length(logL1)/10)), label=nice_kernel_names[1], alpha=0.3)
    logL3 = return_logL(kernel_names[3])
    hist(logL3; density=true, bins=convert(Int64, floor(length(logL3)/10)), label=nice_kernel_names[3], alpha=0.3)
    l_act_theta = L"\ell_{act}(\theta|t,s)"
    xlabel(l_act_theta)
    legend(;fontsize=30)
    title("Comparing " * l_act_theta * " between kernels", fontsize=45)
    save_PyPlot_fig("compare_test")
end
