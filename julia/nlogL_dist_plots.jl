# errors
# include("src/setup.jl")
include("src/all_functions.jl")

kernel_names = ["pp", "se", "m52", "rq", "rm52", "qp_periodic", "m52x2"]
nice_kernel_names = ["Piecewise Polynomial", "Squared Exponential", "Matérn " * L"^5/_2", "Rational Quadratic", "Rational Matérn " * L"^5/_2", "Quasi-Periodic", "Two Matérn " * L"^5/_2"]
# kernel_names = ["pp", "se", "m52", "rm52", "qp_periodic"]
# nice_kernel_names = ["Piecewise Polynomial", "Squared Exponential", "Matérn " * L"^5/_2", "Rational Matérn " * L"^5/_2", "Quasi-Periodic"]

kernel_amount = length(kernel_names)


function get_df(kernel_name)
    file_name = "csv_files/$(kernel_name)_logLs.csv"
    df = CSV.read(file_name)
end

for i in kernel_names
    df = get_df(i)
    println(i * ": ", length(df[!, 1]))
end

df = get_df(kernel_names[2])
mean(df.E1 - df.E2)
mean(df.L1)

df = get_df(kernel_names[3])
mean(df.E1 - df.E2)
mean(df.L1)
# df.E_wp
# df.L1 - df.L2
df = get_df(kernel_names[6])
mean(df.E1 - df.E2)
mean(df.L1)

df = get_df(kernel_names[7])
mean(df.E1 - df.E2)
mean(df.L1)




function return_logL(kernel_name; filter::Bool=true)
    df = get_df(kernel_name)
    logL = -df.nlogL
    if filter; logL = logL[logL .> -600] end
    if filter; logL = logL[logL .< -200] end
    return logL
end

# plot each kernel dist
for i in 1:kernel_amount  # length(kernel_names)
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
    qp_df = get_df("qp_periodic")
    for i in 1:kernel_amount
        if kernel_names[i] != "qp_periodic"
            logL1 = Float64[]
            logL2 = Float64[]
            kernel_df = get_df(kernel_names[i])
            for j in 1:length(kernel_df[!, 1])
                if kernel_df.seed[j] in qp_df.seed
                    append!(logL1, [kernel_df.nlogL[j]])
                    append!(logL2, qp_df.nlogL[qp_df.seed .== kernel_df.seed[j]])
                end
            end
            hist(logL2-logL1; density=true, bins=convert(Int64, floor(length(logL2)/10)), label=nice_kernel_names[i], alpha=1/(kernel_amount-1))
        end
    end
    l_act_theta = L"\ell_{act}(\theta|t,s)"
    xlabel(l_act_theta * " difference")
    legend(;fontsize=30)
    title("Comparing " * l_act_theta * " difference between kernels", fontsize=45)
    save_PyPlot_fig("difference_test")
end



# shared plot
begin
    ax = init_plot()
    for i in 1:kernel_amount
        logL = return_logL(kernel_names[i])
        hist(logL; density=true, bins=convert(Int64, floor(length(logL)/10)), label=nice_kernel_names[i], alpha=1/kernel_amount)
    end
    l_act_theta = L"\ell_{act}(\theta|t,s)"
    xlabel(l_act_theta)
    legend(;fontsize=30)
    title("Comparing " * l_act_theta * " between kernels", fontsize=45)
    save_PyPlot_fig("compare_test")
end

using KernelDensity
begin
    df = get_df("qp_periodic")
    init_plot()
    # hist(df[!, end-1]; density=true, bins=convert(Int64, floor(length(df[!, end-1])/4)))
    x = collect(linspace(10, 35, 1000))
    data = df[!, end-1]
    plot(x, pdf(kde(data), x))
    # plot(x, pdf(kde(data; bandwidth=KernelDensity.default_bandwidth(data)/2), x))
    xlabel("Time (days)")
    title("QP period hyperparameter distribution (" * L"\tau_{med}" * "=30 days)", fontsize=45)
    save_PyPlot_fig("test")
end

begin
    kernel_names = ["pp", "se", "m52", "rq", "rm52", "qp_periodic", "m52x2"]
    df = get_df(kernel_names[3])
    for i in 1:100
        if !(i in df.seed)
            println(i)
        end
    end
end

println(get_df(kernel_names[1])[1, :])
