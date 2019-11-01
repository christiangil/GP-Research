include("src/all_functions.jl")

# kernel_names = ["pp", "se", "m52", "rq", "rm52", "qp", "m52x2"]
kernel_names = ["pp", "se", "m52", "qp", "m52x2"]
# nice_kernel_names = ["Piecewise Polynomial", "Squared Exponential", "Matérn " * L"^5/_2", "Rational Quadratic", "Rational Matérn " * L"^5/_2", "Quasi-Periodic", "Two Matérn " * L"^5/_2"]
nice_kernel_names = ["Piecewise Polynomial", "Squared Exponential", "Matérn " * L"^5/_2", "Quasi-Periodic", "Two Matérn " * L"^5/_2"]
Ks = [string(round(i, digits=2)) for i in (collect(0:10) / 10)]
seeds = [string(i) for i in 1:10]

using CSV, DataFrames

begin
    ax = init_plot()
    println()
    for k in 1:length(kernel_names)
        kernel_name = kernel_names[k]
        factors = zeros(length(Ks))
        factor_errs = zeros(length(Ks))
        for i in 1:length(Ks)
            K = Ks[i]
            seed_factors = zeros(length(seeds))
            for j in 1:length(seeds)
                seed = seeds[j]
                results_dir = "results/$(kernel_name)/K_$(string(K))/seed_$(seed)/"
                try
                    df = CSV.read(results_dir * "logL.csv")
                    seed_factors[j] = (df.E2-df.E1)[1]
                    # seed_factors[j] = (df.E_wp-df.E1)[1]
                catch
                    println(kernel_name, " ", K, " ", seed, " failed")
                    seed_factors[j] = 0
                end
            end
            factors[i] = mean(seed_factors)
            factor_errs[i] = std(seed_factors)
        end
        plot(Ks, factors, color="C$(k-1)", zorder=2, label=nice_kernel_names[k])
        fill_between(Ks, factors + factor_errs, factors - factor_errs, alpha=0.1, color="C$(k-1)")
        # errorbar(Ks, factors, yerr=[factor_errs';factor_errs'], color="C$(k-1)", fmt="o", zorder=2)
        scatter(Ks, factors, color="C$(k-1)", zorder=2)
    end
    n=100
    ax.axvline(x=.25*sqrt(50/n) * 10, color="black", linewidth=3)  # should be 0.25?
    legend(;loc="upper left", fontsize=30)
    xlabel("K " * L"(^m/_s)")
    ylabel("Evidence ratio")
    title("Evidence ratio of models fit with\nand without a planet signal", fontsize=45)
    save_PyPlot_fig("test.png")
end
