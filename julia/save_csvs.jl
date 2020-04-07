include("src/general_functions.jl")
using Statistics
using CSV, DataFrames

kernel_names = ["pp", "se", "m52", "qp", "m52_m52", "se_se", "zero", "white"]

Ks = [string(round(i, digits=2)) for i in (collect(0:10) / 10)]
seeds_rest = [string(i) for i in 1:50]
seeds_0 = [string(i) for i in 1:200]
star_strs = ["long", "short"]
ns = ["100", "300"]
for star_str in star_strs
    for n in ns
        for kernel_name in kernel_names
            kernel_name in ["zero", "white"] ? n_out = 1 : n_out = 3
            for i in 1:length(Ks)
                K = Ks[i]
                i==1 ? seeds = seeds_0 : seeds = seeds_rest
                global df = DataFrame()
                for j in 1:length(seeds)
                    seed = seeds[j]
                    try
                        append!(df, CSV.read("results/" * star_str * "/$n_out/" * n * "/$(kernel_name)/K_$(string(K))/seed_$(seed)/logL.csv"))
                    catch
                        println(kernel_name, " ", K, " ", seed, " failed")
                    end
                end
                sort!(df, :seed)
                CSV.write("saved_csv/$(star_str)_$(n)_$(kernel_name)_$(string(K)).csv", df)
            end
        end
    end
end

for kernel_name in kernel_names
    global df = DataFrame()
    for star_str in star_strs
        for n in ns
            for K in Ks
                df2 = CSV.read("saved_csv/$(star_str)_$(n)_$(kernel_name)_$K.csv")
                insertcols!(df2, 1, no_weirdness=(df2.E2 .!= 0) + (df2.E1 .!= 0) .== 2)
                insertcols!(df2, 1, K=parse(Float64, K))
                insertcols!(df2, 1, n=parse(Int64, n))
                insertcols!(df2, 1, star=star_str)
                append!(df, df2)
            end
        end
    end
    CSV.write("saved_csv/$(kernel_name)_results.csv", df)
end
