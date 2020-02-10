include("src/general_functions.jl")
include("src/plotting_functions.jl")
using Statistics
using CSV, DataFrames

# kernel_names = ["pp", "se", "m52", "rq", "rm52", "qp", "m52x2"]
# nice_kernel_names = ["Piecewise Polynomial", "Squared Exponential", "Matérn " * L"^5/_2", "Rational Quadratic", "Rational Matérn " * L"^5/_2", "Quasi-Periodic", "Two Matérn " * L"^5/_2"]
# kernel_names = ["pp", "se", "m52", "qp", "m52x2"]
# nice_kernel_names = ["Piecewise Polynomial", "Squared Exponential", "Matérn " * L"^5/_2", "Quasi-Periodic", "Two Matérn " * L"^5/_2"]
kernel_names = ["se", "m52", "qp"]
nice_kernel_names = ["Squared Exponential", "Matérn " * L"^5/_2", "Quasi-Periodic"]

Ks = [string(round(i, digits=2)) for i in (collect(0:10) / 10)]
seeds_rest = [string(i) for i in 1:50]
seeds_0 = [string(i) for i in 1:200]
detect_thres = 0.95
detection = zeros(length(kernel_names), 3)

lrs = zeros(length(kernel_names), length(Ks))
lrs_errs_high = copy(lrs)
lrs_errs_low = copy(lrs)
lrs_detection = copy(lrs)
uers = copy(lrs)
uers_errs_high = copy(lrs)
uers_errs_low = copy(lrs)
uers_detection = copy(lrs)
ers = copy(lrs)
ers_errs_high = copy(lrs)
ers_errs_low = copy(lrs)
ers_detection = copy(lrs)

for star_str in ["long", "short"]
    for n in ["100", "300"]
        for k in 1:length(kernel_names)
            failures = 0
            l_failures = 0
            ue_failures = 0
            e_failures = 0
            kernel_name = kernel_names[k]
            for i in 1:length(Ks)
                K = Ks[i]
                i==1 ? seeds = seeds_0 : seeds = seeds_rest
                seed_lrs = zeros(length(seeds))
                seed_uers = zeros(length(seeds))
                seed_ers = zeros(length(seeds))
                for j in 1:length(seeds)
                    seed = seeds[j]
                    try
                        # use_long ? star_str = "long/" : star_str = "short/"
                        df = CSV.read("results/" * star_str * "/" * n * "/$(kernel_name)/K_$(string(K))/seed_$(seed)/logL.csv")
                        if df.L2[1] != 0 && df.L1[1] != 0
                            seed_lrs[j] = (df.L2-df.L1)[1]
                        else
                            println(kernel_name, " ", K, " ", seed, " has weird likelihoods")
                            l_failures += 1
                        end
                        if df.uE2[1] != 0 && df.uE1[1] != 0
                            seed_uers[j] = (df.uE2-df.uE1)[1]
                        else
                            println(kernel_name, " ", K, " ", seed, " has weird unnormalized evidence")
                            ue_failures += 1
                        end
                        if df.E2[1] != 0 && df.E1[1] != 0
                            seed_ers[j] = (df.E2-df.E1)[1]
                        else
                            println(kernel_name, " ", K, " ", seed, " has weird evidence")
                            e_failures += 1
                        end
                    catch
                        println(kernel_name, " ", K, " ", seed, " failed")
                        failures += 1
                    end
                end
                seed_lrs = remove_zeros(seed_lrs)
                lrs[k, i] = median(seed_lrs)
                lrs_errs_high[k, i] = quantile(seed_lrs, 0.84)
                lrs_errs_low[k, i] = quantile(seed_lrs, 0.16)
                if i==1; detection[k, 1] = quantile(seed_lrs, detect_thres) end
                lrs_detection[k, i] = sum(seed_lrs .>= detection[k, 1]) / length(seed_lrs)

                seed_uers = remove_zeros(seed_uers)
                uers[k, i] = median(seed_uers)
                uers_errs_high[k, i] = quantile(seed_uers, 0.84)
                uers_errs_low[k, i] = quantile(seed_uers, 0.16)
                if i==1; detection[k, 2] = quantile(seed_uers, detect_thres) end
                uers_detection[k, i] = sum(seed_uers .>= detection[k, 2]) / length(seed_uers)

                seed_ers = remove_zeros(seed_ers)
                ers[k, i] = median(seed_ers)
                ers_errs_high[k, i] = quantile(seed_ers, 0.84)
                ers_errs_low[k, i] = quantile(seed_ers, 0.16)
                if i==1; detection[k, 3] = quantile(seed_ers, detect_thres) end
                ers_detection[k, i] = sum(seed_ers .>= detection[k, 3]) / length(seed_ers)

            end
            println(kernel_name, " failed ", failures, " times")
            println(kernel_name, " had weird likelihoods an additional ", l_failures, " times")
            println(kernel_name, " had weird unnormalized evidence an additional ", ue_failures, " times")
            println(kernel_name, " had weird evidence an additional ", e_failures, " times")
        end
        add_str = star_str * "_" * n
        # n=100
        # ax.axvline(x=.25*sqrt(50/n) * 10, color="black", linewidth=3)  # should be 0.25?

        ax = init_plot()
        for i in 1:length(kernel_names)
            plot(Ks, lrs[i,:], color="C$(i-1)", zorder=2, label=nice_kernel_names[i])
            fill_between(Ks, lrs_errs_high[i,:], lrs_errs_low[i,:], alpha=0.1, color="C$(i-1)")
            scatter(Ks, lrs[i,:], color="C$(i-1)", zorder=2)
            ax.axhline(y=detection[i, 1], color="C$(i-1)", linestyle="--", linewidth=1)
        end
        legend(;loc="upper left", fontsize=30)
        xlabel("K " * L"(^m/_s)")
        ylabel("Likelihood ratios")
        title("Likelihood ratios of models fit with\nand without a planet signal", fontsize=45)
        save_PyPlot_fig("LRs_" * add_str * ".png")

        ax = init_plot()
        for i in 1:length(kernel_names)
            plot(Ks, uers[i,:], color="C$(i-1)", zorder=2, label=nice_kernel_names[i])
            fill_between(Ks, uers_errs_high[i,:], uers_errs_low[i,:], alpha=0.1, color="C$(i-1)")
            scatter(Ks, uers[i,:], color="C$(i-1)", zorder=2)
            ax.axhline(y=detection[i, 2], color="C$(i-1)", linestyle="--", linewidth=1)
        end
        legend(;loc="upper left", fontsize=30)
        xlabel("K " * L"(^m/_s)")
        ylabel("Unormalized evidence ratios")
        title("Unormalized evidence ratios of models fit with\nand without a planet signal", fontsize=45)
        save_PyPlot_fig("uERs_" * add_str * ".png")

        ax = init_plot()
        for i in 1:length(kernel_names)
            plot(Ks, ers[i,:], color="C$(i-1)", zorder=2, label=nice_kernel_names[i])
            fill_between(Ks, ers_errs_high[i,:], ers_errs_low[i,:], alpha=0.1, color="C$(i-1)")
            scatter(Ks, ers[i,:], color="C$(i-1)", zorder=2)
            ax.axhline(y=detection[i, 3], color="C$(i-1)", linestyle="--", linewidth=1)
        end
        legend(;loc="upper left", fontsize=30)
        xlabel("K " * L"(^m/_s)")
        ylabel("Evidence ratios")
        title("Evidence ratios of models fit with\nand without a planet signal", fontsize=45)
        save_PyPlot_fig("ERs_" * add_str * ".png")


        ax = init_plot()
        for i in 1:length(kernel_names)
            plot(Ks, lrs_detection[i,:], color="C$(i-1)", zorder=2, label=nice_kernel_names[i])
            scatter(Ks, lrs_detection[i,:], color="C$(i-1)", zorder=2)
        end
        ax.axhline(y=0.5, color="black", linewidth=3)
        legend(;loc="lower right", fontsize=30)
        xlabel("K " * L"(^m/_s)")
        title("Proportion of likelihood ratios > the " * string(Int(round(detect_thres * 100))) * "th percentile", fontsize=45)
        save_PyPlot_fig("detections_LRs_" * add_str * ".png")

        ax = init_plot()
        for i in 1:length(kernel_names)
            plot(Ks, uers_detection[i,:], color="C$(i-1)", zorder=2, label=nice_kernel_names[i])
            scatter(Ks, uers_detection[i,:], color="C$(i-1)", zorder=2)
        end
        ax.axhline(y=0.5, color="black", linewidth=3)
        legend(;loc="lower right", fontsize=30)
        xlabel("K " * L"(^m/_s)")
        title("Proportion of unnormalized evidence ratios > the " * string(Int(round(detect_thres * 100))) * "th percentile", fontsize=45)
        save_PyPlot_fig("detections_uERs_" * add_str * ".png")

        ax = init_plot()
        for i in 1:length(kernel_names)
            plot(Ks, ers_detection[i,:], color="C$(i-1)", zorder=2, label=nice_kernel_names[i])
            scatter(Ks, ers_detection[i,:], color="C$(i-1)", zorder=2)
        end
        ax.axhline(y=0.5, color="black", linewidth=3)
        legend(;loc="lower right", fontsize=30)
        xlabel("K " * L"(^m/_s)")
        title("Proportion of evidence ratios > the " * string(Int(round(detect_thres * 100))) * "th percentile", fontsize=45)
        save_PyPlot_fig("detections_ERs_" * add_str * ".png")
    end
end
