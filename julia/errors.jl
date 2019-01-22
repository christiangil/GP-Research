# errors
include("all_functions.jl")
clear_variables()

using JLD2, FileIO
using HDF5
using Distributions

# test = "D:/Christian/Downloads/lambda-3923-6664-3years_174spots_diffrot_id9.h5"
test = "/Users/cjg66/Downloads/lambda-3923-6664-3years_174spots_diffrot_id9.h5"
fid = h5open(test, "r")

act = fid["active"]
lam = fid["lambdas"]
quiet = fid["quiet"]  # flux in ergs?
quiet_array = quiet[:]

# h = 6.6260755e-27  # erg s
# c =  2.99792458e10 * 1e8  # cm / s -> A / s
# photon_energy = vec(h * c ./ lam[:])
# N = quiet_array[:] ./ photon_energy
# errors = photon_energy .* sqrt.(N)
errors = quiet_array * 0.01
# error_floor = 1  # max is around 60, prevents 0 error points
# errors[errors .< error_floor] = error_floor * ones(length(errors[errors .< error_floor]))

# function error_plot(y)
#     init_plot()
#     xlabel("wavelengths (angstrom)")
#     plot(lam[:], y)
# end
#
#
# error_plot(quiet[:])
# title("quiet spectrum")
# ylabel("flux (ergs?)")
# savefig("figs/errors/quiet_spectrum.pdf")
#
# error_plot(N)
# title("quiet spectrum")
# ylabel("photon count?")
# savefig("figs/errors/quiet_spectrum_N.pdf")
#
# error_plot(errors)
# title("quiet spectrum errors")
# ylabel("flux (ergs?)")
# savefig("figs/errors/quiet_spectrum_shot_errors.pdf")


# bootstrapping for errors in scores

println("break")
boot_amount = 100
subset = 2190  # needs to be <= 2190
@load "rv_data.jld2" doppler_comp genpca_out rvs_out
mu, M, scores0 = genpca_out
scores0 = scores0[1:subset, :]

@elapsed X = act[:, 1:subset] # ~60 s
num_lambda = size(X, 1)
num_spectra = size(X, 2)
num_components = size(M, 2)
scores_tot = zeros(boot_amount, num_spectra, num_components)
for k in 1:boot_amount
    scores = zeros(num_spectra, num_components)
    Xtmp = X .+ (errors .* rand(Normal(), num_lambda))
    # Xtmp_lt_0 = Xtmp .< 0  # ~15 s
    # Xtmp[Xtmp_lt_0] = zeros(sum(Xtmp_lt_0))  # ~30 s
    mu = vec(mean(Xtmp, dims=2))
    Xtmp .-= mu  # ~60s
    fixed_comp = M[:,1]
    fixed_comp_norm = 1/sum(abs2, fixed_comp)
    for i in 1:num_spectra
        scores[i, 1] = (dot(view(Xtmp, :, i), fixed_comp) * fixed_comp_norm)  # Normalize differently, so scores are z (i.e., doppler shift)
        Xtmp[:, i] -= scores[i, 1] * fixed_comp
    end
    for j in 2:num_components
        for i in 1:num_spectra
            scores[i, j] = dot(view(Xtmp, :, i), view(M, :, j)) #/sum(abs2,view(M,:,j-1))
            Xtmp[:, i] .-= scores[i, j] * view(M, :, j)
        end
    end
    scores_tot[k, :, :] = scores
end


# for ind in 1:num_components
#     init_plot()
#     # comp = scores[:, ind] - scores0[:, ind]
#     # plot(collect(1:num_spectra), comp)
#     plot(collect(1:num_spectra), scores[:, ind])
#     plot(collect(1:num_spectra), scores0[:, ind])
#     # legend(["dif", "with error", "base"])
#     legend(["with error", "base"])
#     savefig("figs/errors/PCA_" * string(ind) * "_new_comparison.pdf")
# end
#
#
# println("score ratios")
# for i in 1:num_components
#     println("PC " * string(i))
#     println("variance")
#     println(var(scores0[:, i])/var(scores[:, i]))
#     println("mean")
#     println(mean(abs.(scores0[:, i]))/mean(abs.(scores[:, i])))
# end

@save "boot.jld2" scores_tot

test = mapslices(fit_mle, Normal, scores_tot[:, :, 1]; dims=2)
test = mapslices(std, scores_tot[:, :, 1]; dims=2)
# a = fit_mle(Normal, storage[:,test])
# println(a.σ)
