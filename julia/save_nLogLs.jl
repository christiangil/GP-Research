include("src/all_functions.jl")

dir = "csv_files/"
kernel_names = ["pp", "se", "m52", "rq", "rm52", "qp_periodic", "m52x2"]

for j in 1:length(kernel_names)
    # kernel_name = kernel_names[parse(Int, ARGS[1])]
    kernel_name = kernel_names[j]
    LogLs = sort(filter(s->occursin(Regex("^$(kernel_name)_logL_"), s), readdir(dir)))

    df = DataFrame()
    for i in 1:length(LogLs)
        append!(df, CSV.read(dir * LogLs[i]))
    end

    sort!(df, :seed)

    CSV.write(dir * "$(kernel_name)_logLs.csv", df)
end
