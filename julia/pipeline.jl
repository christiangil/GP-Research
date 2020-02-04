# SOAP spectra conversion pipeline

# include("src/setup.jl")
include("src/all_functions.jl")

#################################
# Importing from SOAP HDF5 file #
#################################

hdf5_loc = "C:/Users/chris/Downloads/res-1000-lambda-3923-6664-1years_1579spots_diffrot_id11.h5"
# hdf5_loc = "D:/Christian/Downloads/res-1000-lambda-3923-6664-1years_1584spots_diffrot_id12.h5"
# length(ARGS)>0 ? id = ARGS[1] : id = "11"
# hdf5_loc = "/gpfs/group/ebf11/default/SOAP_output/Christian_runs/res-1000-1years_full_id$id.h5"

hdf5_filename = "jld2_files/" * string(split(hdf5_loc,"/")[end])[1:end-3]
fid = h5open(hdf5_loc, "r")

@elapsed obs, λs = prep_SOAP_spectra(fid)
λs = ustrip.(λs * 10)
phases = fid["phases"][:]
quiet = fid["quiet"][:]

###########################
# Calculating RVs and PCA #
###########################

average_spectra = vec(mean(obs, dims=2))
doppler_comp = calc_doppler_component_RVSKL(λs, average_spectra)

# Compute first num_components (can be specified) basis vectors for PCA, after
# subtracting projection onto fixed_comp (the doppler component we passed)
# returns yhe doppler basis component, mean spectra, basis vectors, PCA scores,
# fractional variance remaining after each component is taken out, and the radial velocities
# M[:, 1] is not a unit vector, this is so scores[1, :] are actually redshifts
mu, M, scores, fracvar, rvs = @time fit_gen_pca_rv_RVSKL(obs, doppler_comp, mu=average_spectra, num_components=6)

@save hdf5_filename * "_rv_data.jld2" λs phases quiet doppler_comp mu M scores fracvar rvs

##########################################
# Finding bootstrap estimates for errors #
##########################################

boot_amount = 10
for i in 1:10
    bootstrap_SOAP_errors(obs, λs, hdf5_filename; boot_amount=boot_amount)
    println("Completed $(i * boot_amount) bootstraps resamples so far")
end

# ################################
# # Creating problem definitions #
# ################################
#
# init_problem_definition(hdf5_filename)
# init_problem_definition(hdf5_filename; sub_sample=70, save_str="sample")
