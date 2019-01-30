#use Julia 0.6 for this script
include("all_functions.jl")

# using JLD
using JLD2, FileIO
using RvSpectraKitLearn
using MultivariateStats
using HDF5
# using Rsvg

# path_to_spectra = "/gpfs/group/ebf11/default/SOAP_output/May_runs"
# test = "/Users/cjg66/Downloads/lambda-3923-6664-3years_174spots_diffrot_id980.h5"
test = "D:/Christian/Downloads/lambda-3923-6664-3years_174spots_diffrot_id9.h5"
# test = "C:/Users/chris/Downloads/lambda-3923-6664-3years_174spots_diffrot_id9.h5"
fid = h5open(test, "r")
objects = names(fid)
println(objects)

act = fid["active"]
# read(act)
single_time = collect(Base.Iterators.flatten(act[1:523732, 1]))
single_wavelength = collect(Base.Iterators.flatten(act[1, 1:2190]))

lam = fid["lambdas"]
read(lam)
lam[:]

phase = fid["phases"]
read(phase)
phase[:]

quiet = fid["quiet"]
read(quiet)
quiet[:]

# lam_slice = :
# phase_slice = :  # 1:convert(Int64, round(2190))
# obs = @time act[lam_slice, phase_slice];

# lambda = lam[lam_slice]
# save("lambda.jld", "data", lambda)
# lambda = load("lambda.jld")["data"]
# phases = phase[phase_slice]
# save("phases.jld", "data", phases)
# phases = load("phases.jld")["data"]
# quiet = quiet[:]
# save("quiet.jld", "data", quiet)
# quiet = load("quiet.jld")["data"]
# @save "sunspot_data.jld2" lambda phases quiet
@load "sunspot_data.jld2" lambda phases quiet

# pca_out = @time fit_pca_default(obs, max_num_components=6);
#
#
# # pca_out.mean
# # pca_out.proj  # projection matrix: of size d x p
# # pca_out.prinvars  # principal variances: of length p
# # pca_out.tprinvar  # total principal variance, i.e. sum(prinvars)
# # pca_out.tvar  # total variance
# remaining_variance = zeros(length(pca_out.prinvars))
# for i in 1:length(pca_out.prinvars)
#     remaining_variance[i] = 1 - sum(pca_out.prinvars[1:i])/pca_out.tvar
# end
# println(remaining_variance)
# plt = plot(traces(collect(1:length(pca_variances)), log10.(remaining_variance)))
# # savefig(plt, "figs/PCA/pca_logvariance.pdf")
# plt = plot(traces(collect(1:length(pca_variances)), remaining_variance))
# # savefig(plt, "figs/PCA/pca_variance.pdf")
#
# pca_scores = @time transform(pca_out, obs);   # compute scores using Julia's PCA routine for comparison's sake
# plt = plot(traces(phases, pca_scores))
# # savefig(plt, "figs/PCA/pca_scores.pdf")
#
# times = zeros(2,10)
# for i in 1:10
#     tic(); fit_pca_default(obs, max_num_components=i); times[1, i] = toc()
#     tic(); fit_pca_eford(obs, num_components=i); times[2, i] = toc()
#     println(i)
# end
# plt = plot(traces(collect(1:10), times))
# # savefig(plt, "figs/PCA/pca_times.pdf")
#
#
# pca_eford_out = @time fit_pca_eford(obs, num_components=length(pca_out.prinvars));   # compute first few scores using Eric's iterative PCA routine
# mu, M, scores = pca_eford_out
# plt = plot(traces(phases, -scores'))
# savefig(plt, "figs/PCA/eric_pca_scores.pdf")
# plt = plot(traces(phases, abs.(pca_scores[1:6,:]) - abs.(scores')))
# # savefig(plt, "figs/PCA/pca_score_comparison.pdf")

# doppler component basis vector? Based on how adjacent bins would grow if there was a shift?
# can use simple derivatives or derivatives of fitted GPs?
# returns doppler basis vector
doppler_comp = @time calc_doppler_component_simple(lambda, obs)
# doppler_comp = calc_doppler_component_gp(lambda,obs)
# save("doppler_comp.jld", "data", doppler_comp)
# doppler_comp = load("doppler_comp.jld")["data"]


# Compute first num_components (can be specified) basis vectors for PCA, after
# subtracting projection onto fixed_comp (the doppler component we passed)
# returns mu, M, scores (mean spectra, basis vectors, and PCA scores)
genpca_out = @time fit_gen_pca_rv(obs, doppler_comp)
# save("genpca_out.jld", "data", genpca_out)
# genpca_out = load("genpca_out.jld")["data"]
mu, M, scores = genpca_out
M = M'
M[1, :] = M[1, :] / sum(abs2, M[1, :])
scores = scores'

# converting scores for doppler component basis vector into actual RV units (m/s)
rvs_out = est_rvs_from_pc(obs, genpca_out[1], genpca_out[2][:,1])
# save("rvs_out.jld", "data", rvs_out)
# rvs_out = load("rvs_out.jld")["data"]


@save "rv_data.jld2" doppler_comp genpca_out rvs_out
# @load "rv_data.jld2" doppler_comp genpca_out rvs_out


plot(line_trace(phases, rvs_out))
plot(traces(lambda, M))
plot(traces(phases, scores))



rms = sqrt(sum(abs2,rvs_out)/length(rvs_out))
# println(" rms = ", rms)

rvs_out = chop_array(rvs_out; dif = 5e-3)
plot(line_trace(phases, rvs_out))
phases[length(phases)]


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
