# importing packages
# Pkg.clone("https://github.com/eford/RvSpectraKitLearn.jl.git")
# Pkg.add("MultivariateStats")
include("all_functions.jl")

using RvSpectraKitLearn
using MultivariateStats
using HDF5
using Rsvg

# path_to_spectra = "/gpfs/group/ebf11/default/SOAP_output/May_runs"
# test = "/Users/cjg66/Downloads/lambda-3923-6664-3years_174spots_diffrot_id980.h5"
test = "D:/Christian/Downloads/lambda-3923-6664-3years_174spots_diffrot_id9.h5"
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

lam_slice = :
phase_slice = 1:convert(Int64, round(2190))
obs = @time act[lam_slice, phase_slice];
lambda = lam[lam_slice];
phases = phase[phase_slice]0

# doppler component basis vector? Based on how adjacent bins would grow if there was a shift?
# can use simple derivatives or derivatives of fitted GPs?
# returns doppler basis vector
doppler_comp = @time calc_doppler_component_simple(lambda, obs)
# doppler_comp = calc_doppler_component_gp(lambda,obs)

# Compute first num_components (can be specified) basis vectors for PCA, after
# subtracting projection onto fixed_comp (the doppler component we passed)
# returns mu, M, scores (mean spectra, basis vectors, and PCA scores)
genpca_out = @time fit_gen_pca_rv(obs, doppler_comp)

# converting scores for doppler component basis vector into actual RV units (m/s)
rvs_out = est_rvs_from_pc(obs, genpca_out[1], genpca_out[2][:,1])

plot(line_trace(phases, rvs_out))
# plot(line_trace(lambda, doppler_comp))



# rms = sqrt(sum(abs2,rvs_out)/length(rvs_out))
# println(" rms = ", rms)


# fid=h5open(test,"r")
# names(fid)  # Names of datasets
names(attrs(fid))  # Names of attributes (for file)
Temp = read(attrs(fid)["Temp"])  # Read a file attribute value
names(attrs(fid["lambdas"]))  # Get names of attributes (for dataset)
read(attrs(fid["lambdas"])["units"])  # Read a dataset attribute value

lambdas = read(fid["lambdas"]) # Read wavelengths  from a dataset
phases = read(fid["phases"]) # Read phases from a dataset
# active = read(fid["active"]) # Read flux from a dataset # Warning this is slow
nlambdas = length(read(fid["lambdas"]))
nphases =  length(read(fid["phases"]))
first_spectra = fid["active"][1:nlambdas,1] # First spectrum as a 2-d array
mask = .!iszero.(first_spectra)
first_spectra = reshape(fid["active"][1:nlambdas,1][mask],(sum(mask))) # First spectra's non-zero values as a 1-d array
