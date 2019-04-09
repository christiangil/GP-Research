# this script converts a modified SOAP generated HDF5 file into JLD2 files used for later analysis
include("../setup.jl")
include("../all_functions.jl")

using JLD2, FileIO
using MultivariateStats
using HDF5

hdf5_loc = "D:/Christian/Downloads/lambda-3923-6664-1years_1586spots_diffrot_id21.h5"
fid = h5open(hdf5_loc, "r")
# objects = names(fid)
# println(objects)

act = fid["active"]
lam = fid["lambdas"]
phase = fid["phases"]
quiet = fid["quiet"]

@elapsed obs = act[:, :]

lambda = lam[:]
phases = phase[:]
quiet = quiet[:]
average_spectra = vec(mean(obs, dims=2))
doppler_comp = calc_doppler_component_RVSKL(lambda, average_spectra)

# Compute first num_components (can be specified) basis vectors for PCA, after
# subtracting projection onto fixed_comp (the doppler component we passed)
# returns yhe doppler basis component, mean spectra, basis vectors, PCA scores,
# fractional variance remaining after each component is taken out, and the radial velocities
# M[:, 1] is not a unit vector, this is so scores[1, :] are actually redshifts
mu, M, scores, fracvar, rvs = @time fit_gen_pca_rv_RVSKL(obs, doppler_comp, mu=average_spectra, num_components=6)

old_dir = pwd()
cd(@__DIR__)
@save "../../jld2_files/rv_data.jld2" lambda phases quiet doppler_comp mu M scores fracvar rvs
cd(old_dir)
