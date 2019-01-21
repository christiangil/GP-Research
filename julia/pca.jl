include("all_functions.jl")

using JLD2, FileIO
# using RvSpectraKitLearn
using MultivariateStats
using HDF5

# # path_to_spectra = "/gpfs/group/ebf11/default/SOAP_output/May_runs"
# # test = "/Users/cjg66/Downloads/lambda-3923-6664-3years_174spots_diffrot_id9.h5"
# test = "D:/Christian/Downloads/lambda-3923-6664-3years_174spots_diffrot_id9.h5"
# test = "C:/Users/Christian/Dropbox/GP_research/star_spots/SOAP2_RawSpectra/Example/test_runs/lambda-3923-6664-0years_18spots_diffrot_id1.h5"
test = "D:/Christian/Downloads/lambda-3923-6664-1years_1582spots_diffrot_id1.h5"
fid = h5open(test, "r")
# objects = names(fid)
# println(objects)
thing1 = fid["n_spots"][:]
thing2 = fid["msh_covered"][:]

init_plot()
plot(thing2)
savefig("test.pdf")

act = fid["active"]
# lam = fid["lambdas"]
# phase = fid["phases"]
# quiet = fid["quiet"]

lam_slice = :
phase_slice = :  # 1:convert(Int64, round(2190))
@elapsed obs = act[lam_slice, phase_slice]

# lambda = lam[lam_slice]
# phases = phase[phase_slice]
# quiet = quiet[:]
# average_spectra = vec(mean(obs, dims=2))
# first_spectra = obs[:, 1]
# @save "sunspot_data.jld2" lambda phases quiet average_spectra first_spectra
@load "sunspot_data.jld2" lambda phases quiet average_spectra first_spectra

# pca_out = @time fit_pca_default(obs, max_num_components=6);
# pca_scores = transform(pca_out, obs) # pca scores, save this later
# @save "pca_out_2.jld2" pca_out pca_scores

# pca_out.mean
# pca_out.proj  # projection matrix: of size d x p
# pca_out.prinvars  # principal variances: of length p
# pca_out.tprinvar  # total principal variance, i.e. sum(prinvars)
# pca_out.tvar  # total variance

# doppler component basis vector? Based on how adjacent bins would grow if there was a shift?
# can use simple derivatives or derivatives of fitted GPs?
# returns doppler basis vector
# doppler_comp = @time calc_doppler_component_simple(lambda, obs)
# # doppler_comp = calc_doppler_component_gp(lambda,obs)

# Compute first num_components (can be specified) basis vectors for PCA, after
# subtracting projection onto fixed_comp (the doppler component we passed)
# returns mu, M, scores (mean spectra, basis vectors, and PCA scores)
# genpca_out = @time fit_gen_pca_rv(obs, doppler_comp)

# converting scores for doppler component basis vector into actual RV units (m/s)
# rvs_out = est_rvs_from_pc(obs, genpca_out[1], genpca_out[2][:,1])

# @save "rv_data.jld2" doppler_comp genpca_out rvs_out


@load "pca_out_2.jld2" pca_out pca_scores

# # this is the same as pca_out.prinvars
using Statistics
prinvars = mapslices(var, pca_scores; dims=2)
prinvars = pca_out.prinvars


@load "rv_data.jld2" doppler_comp genpca_out rvs_out
mu, M, scores = genpca_out
# M = M'
# M[1, :] = M[1, :] / sum(abs2, M[1, :])
scores = scores'
scores[1,:] *= norm(doppler_comp)  # accounting for doppler_comp not having a norm of 1
prinvars = mapslices(var, scores; dims=2)


# variance plotting

remaining_variance = zeros(length(prinvars))
for i in 1:length(prinvars)
    remaining_variance[i] = 1 - sum(prinvars[1:i]) / max(pca_out.tvar, sum(prinvars))
end
println(remaining_variance)
println(log10.(remaining_variance))


# function variance_plot()
#     figure(figsize=(10,6))
#     ax = subplot(111)
#     set_font_sizes(ax)
# end
#
#
# variance_plot()
# plot(collect(0:(length(remaining_variance)-2)), log10.(remaining_variance[1:(length(remaining_variance)-1)]), linewidth=4.0)
# xlabel("Number of Principal Components")
# ylabel("log_10(Fraction of Variance Explained)")
# savefig("figs/pca/pca_logvariance.pdf")
#
# variance_plot()
# plot(collect(0:(length(remaining_variance)-1)), remaining_variance, linewidth=4.0)
# xlabel("Number of Principal Components")
# ylabel("Fraction of Variance Explained")
# savefig("figs/pca/pca_variance.pdf")


# HDF5 manipulation

# # fid=h5open(test,"r")
# names(fid)  # Names of datasets
# names(attrs(fid))  # Names of attributes (for file)
# Temp = read(attrs(fid)["Temp"])  # Read a file attribute value
# names(attrs(fid["lambdas"]))  # Get names of attributes (for dataset)
# read(attrs(fid["lambdas"])["units"])  # Read a dataset attribute value
#
# lambdas = read(fid["lambdas"]) # Read wavelengths  from a dataset
# phases = read(fid["phases"]) # Read phases from a dataset
# # active = read(fid["active"]) # Read flux from a dataset # Warning this is slow
# nlambdas = length(read(fid["lambdas"]))
# nphases =  length(read(fid["phases"]))
# first_spectra = fid["active"][1:nlambdas,1] # First spectrum as a 2-d array
# mask = .!iszero.(first_spectra)
# first_spectra = reshape(fid["active"][1:nlambdas,1][mask],(sum(mask))) # First spectra's non-zero values as a 1-d array
