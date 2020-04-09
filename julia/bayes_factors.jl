include("src/general_functions.jl")
include("src/plotting_functions.jl")
using Statistics
using CSV, DataFrames


function get_sub_df(tot_df::DataFrame, K::Real, n::Integer, star_str::String)
    df = tot_df[tot_df.K .== K, :]
    df = df[df.n .== n, :]
    return df[df.star .== star_str, :]
end

quantiles(thing::Vector{<:Real}) = median(thing), quantile(thing, 0.84), quantile(thing, 0.16)

M = 2^13
F = 0.5
σ = 0.24
best_case(n) = σ * sqrt(2 * ((M/F)^(2/(n - 3)) - 1))


kernel_names = ["pp", "se", "m52", "qp", "m52_m52", "se_se", "white", "zero", "no_activity"]
nice_kernel_names = ["Piecewise Polynomial", "Squared Exponential", "Matérn " * L"^5/_2", "Quasi-Periodic", "Two Matérn " * L"^5/_2", "Two Squared Exponential", "White Noise", "None", "Best Case"]

Ks = [round(i, digits=2) for i in (collect(0:10) / 10)]
seeds_rest = [string(i) for i in 1:50]
seeds_0 = [string(i) for i in 1:200]
star_strs = ["long", "short"]
ns = [100, 300]
detect_thres = 0.95
detection = zeros(length(kernel_names))

ideal_total = length(star_strs) * length(ns) * ((length(Ks) - 1) * length(seeds_rest) + length(seeds_0))

meds = zeros(length(kernel_names), length(Ks))
errs_high = copy(meds)
errs_low = copy(meds)
ers_detection = copy(meds)

tot_dfs = [CSV.read("saved_csv/$(kernel_name)_results.csv") for kernel_name in kernel_names]

for ii in 1:length(star_strs)
    star_str = star_strs[ii]
    for jj in 1:length(ns)
        n = ns[jj]
        for k in 1:length(kernel_names)
            kernel_name = kernel_names[k]
            tot_df = tot_dfs[k]
            non_fail = size(tot_df)[1]
            tot_df = tot_df[tot_df.no_weirdness .== true, :]

            if ii==1 && jj==1
                println("$kernel_name failed $(ideal_total - non_fail)/$ideal_total times")
                println("$kernel_name had weird evidence another $(non_fail - size(tot_df)[1])/$non_fail times")
            end

            for i in 1:length(Ks)
                K = Ks[i]
                kernel_name == "no_activity" ? df = get_sub_df(tot_df, K, n, "none") : df = get_sub_df(tot_df, K, n, star_str)
                seed_ers = remove_zeros(df.E2 - df.E1)
                meds[k, i], errs_high[k, i], errs_low[k, i] = quantiles(seed_ers)
                if i==1; detection[k] = quantile(seed_ers, detect_thres) end
                ers_detection[k, i] = sum(seed_ers .>= detection[k]) / length(seed_ers)
            end
        end
        add_str = star_str * "_" * string(n)
        # n=100
        # ax.axvline(x=.25*sqrt(50/n) * 10, color="black", linewidth=3)  # should be 0.25?

        ax = init_plot()
        for i in 1:length(kernel_names)
            plot(Ks, meds[i,:], color="C$(i-1)", zorder=2, label=nice_kernel_names[i])
            fill_between(Ks, errs_high[i,:], errs_low[i,:], alpha=0.1, color="C$(i-1)")
            scatter(Ks, meds[i,:], color="C$(i-1)", zorder=2)
            ax.axhline(y=detection[i], color="C$(i-1)", linestyle="--", linewidth=1)
        end
        legend(;loc="upper left", fontsize=15)
        xlabel("Injected K " * L"(^m/_s)")
        ylabel("Evidence ratios")
        title("Evidence ratios of models fit with\nand without a planet signal", fontsize=45)
        ylim(minimum(errs_low[1:length(kernel_names)-1,1])-15, maximum(errs_high[1:length(kernel_names)-1,end])+15)
        save_PyPlot_fig("ERs_" * add_str * ".png")

        ax = init_plot()
        for i in 1:length(kernel_names)
            plot(Ks, ers_detection[i,:], color="C$(i-1)", zorder=2, label=nice_kernel_names[i])
            scatter(Ks, ers_detection[i,:], color="C$(i-1)", zorder=2)
        end
        ax.axhline(y=0.5, color="black", linewidth=3)
        ax.axvline(x=best_case(n), linestyle="--", color="grey", linewidth=3)
        legend(;loc="lower right", fontsize=15)
        xlabel("Injected K " * L"(^m/_s)")
        title("Proportion of evidence ratios > the " * string(Int(round(detect_thres * 100))) * "th percentile", fontsize=45)
        save_PyPlot_fig("detections_ERs_" * add_str * ".png")
    end
end

## periods

for star_str in star_strs
    for n in ns
        for k in 1:length(kernel_names)
            kernel_name = kernel_names[k]
            tot_df = tot_dfs[k]
            tot_df = tot_df[tot_df.no_weirdness .== true, :]
            for i in 1:length(Ks)
                kernel_name == "no_activity" ? df = get_sub_df(tot_df, Ks[i], n, "none") : df = get_sub_df(tot_df, Ks[i], n, star_str)
                P1 = [parse(Float64, df.P1[i][1:end-2]) for i in 1:size(df)[1]]
                P2 = [parse(Float64, df.P2[i][1:end-2]) for i in 1:size(df)[1]]
                desired_quant = (P2 - P1) ./ P1
                meds[k, i], errs_high[k, i], errs_low[k, i] = quantiles(desired_quant)
            end
        end
        add_str = star_str * "_" * string(n)

        ax = init_plot()
        for i in 1:length(kernel_names)
            plot(Ks, meds[i,:], color="C$(i-1)", zorder=2, label=nice_kernel_names[i])
            fill_between(Ks, errs_high[i,:], errs_low[i,:], alpha=0.1, color="C$(i-1)")
            scatter(Ks, meds[i,:], color="C$(i-1)", zorder=2)
        end
        ylim(-1,0.5)
        ax.axhline(y=0, color="black", linewidth=3)
        legend(;loc="lower right", fontsize=20)
        xlabel("Injected K " * L"(^m/_s)")
        ylabel("Fractional error between\ninjected and found periods")
        title("Period errors of found planets", fontsize=45)
        save_PyPlot_fig("Ps_" * add_str * ".png")
    end
end

## Ks
for star_str in star_strs
    for n in ns
        for k in 1:length(kernel_names)
            kernel_name = kernel_names[k]
            tot_df = tot_dfs[k]
            tot_df = tot_df[tot_df.no_weirdness .== true, :]
            for i in 1:length(Ks)
                kernel_name == "no_activity" ? df = get_sub_df(tot_df, Ks[i], n, "none") : df = get_sub_df(tot_df, Ks[i], n, star_str)
                K1 = [parse(Float64, df.K1[i][1:end-7]) for i in 1:size(df)[1]]
                K2 = [parse(Float64, df.K2[i][1:end-7]) for i in 1:size(df)[1]]
                desired_quant = K2 - K1
                meds[k, i], errs_high[k, i], errs_low[k, i] = quantiles(desired_quant)
            end
        end
        add_str = star_str * "_" * string(n)

        ax = init_plot()
        for i in 1:length(kernel_names)
            plot(Ks, meds[i,:], color="C$(i-1)", zorder=2, label=nice_kernel_names[i])
            fill_between(Ks, errs_high[i,:], errs_low[i,:], alpha=0.1, color="C$(i-1)")
            scatter(Ks, meds[i,:], color="C$(i-1)", zorder=2)
        end
        ylim(-0.25,0.25)
        ax.axhline(y=0, color="black", linewidth=3)
        legend(;loc="lower right", fontsize=20)
        xlabel("Injected K " * L"(^m/_s)")
        ylabel("Difference between injected\nand found K " * L"(^m/_s)")
        title("Velocity semi-amplitude errors of found planets", fontsize=45)
        save_PyPlot_fig("Ks_" * add_str * ".png")
    end
end
## es

for star_str in star_strs
    for n in ns
        for k in 1:length(kernel_names)
            kernel_name = kernel_names[k]
            tot_df = tot_dfs[k]
            tot_df = tot_df[tot_df.no_weirdness .== true, :]
            for i in 1:length(Ks)
                kernel_name == "no_activity" ? df = get_sub_df(tot_df, Ks[i], n, "none") : df = get_sub_df(tot_df, Ks[i], n, star_str)
                desired_quant = df.e2 - df.e1
                meds[k, i], errs_high[k, i], errs_low[k, i] = quantiles(desired_quant)
            end
        end
        add_str = star_str * "_" * string(n)

        ax = init_plot()
        for i in 1:length(kernel_names)
            plot(Ks, meds[i,:], color="C$(i-1)", zorder=2, label=nice_kernel_names[i])
            fill_between(Ks, errs_high[i,:], errs_low[i,:], alpha=0.1, color="C$(i-1)")
            scatter(Ks, meds[i,:], color="C$(i-1)", zorder=2)
        end
        ylim(0,0.4)
        ax.axhline(y=0, color="black", linewidth=3)
        legend(;loc="upper right", fontsize=20)
        xlabel("Injected K " * L"(^m/_s)")
        ylabel("Difference between injected\nand found e")
        title("Eccentricity errors of found planets", fontsize=45)
        save_PyPlot_fig("es_" * add_str * ".png")
    end
end
