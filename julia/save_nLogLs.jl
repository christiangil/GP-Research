include("src/setup.jl")
include("src/all_functions.jl")

dir = "csv_files/"
kernel_names = ["quasi_periodic_kernel", "se_kernel", "matern52_kernel", "rq_kernel"]

for j in 1:3
    # kernel_name = kernel_names[parse(Int, ARGS[1])]
    kernel_name = kernel_names[j]
    LogLs = sort(filter(s->occursin(Regex("$(kernel_name)_logL_"), s), readdir(dir)))

    df = DataFrame()
    for i in 1:length(LogLs)
        append!(df, CSV.read(dir * LogLs[i]))
    end

    CSV.write(dir * "$(kernel_name)_logLs.csv", df)
end
