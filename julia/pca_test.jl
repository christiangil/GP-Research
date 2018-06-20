# importing packages
# Pkg.clone("https://github.com/eford/RvSpectraKitLearn.jl.git")
using RvSpectraKitLearn
using MultivariateStats
# include(joinpath(Pkg.dir("RvSpectraKitLearn"),"path_to_spectra.jl"))
path_to_spectra = "/Users/cjg66/Downloads/planet_10ms_150k"

cd(path_to_spectra)

(datafiles, phases) = get_filenames(); # num_max_files=4);
(lambda, obs_noisefree) = read_filelist(datafiles);

sampling = 3.0
planet_amplitude = 10.0
rvs_true = -planet_amplitude*cos.(2pi*phases)

snr = 100.0
obs = RvSpectraKitLearn.make_noisy_spectra(obs_noisefree,snr,sampling=sampling)

pca_out = fit_pca_default(obs);
pca_scores = transform(pca_out,obs)   # compute scores using Julia's PCA routine for comparison's sake
pca_eford_out = fit_pca_eford(obs);   # compute first few scores using Eric's itterative PCA routine

doppler_comp_simple = calc_doppler_component_simple(lambda,obs);
genpca_simple_out = fit_gen_pca_rv(obs,doppler_comp_simple)
rvs_out_simple = est_rvs_from_pc(obs,genpca_simple_out[1],genpca_simple_out[2][:,1])


doppler_comp_gp = calc_doppler_component_gp(lambda,obs);
genpca_gp_out = fit_gen_pca_rv(obs,doppler_comp_gp)
rvs_out_gp = est_rvs_from_pc(obs,genpca_gp_out[1],genpca_gp_out[2][:,1])
rms_simple = sqrt(sum(abs2,rvs_out_simple.-rvs_true)/length(rvs_true))
rms_gp = sqrt(sum(abs2,rvs_out_gp.-rvs_true)/length(rvs_true))
println("snr = ", snr, " rms = ", rms_simple, "  ", rms_gp)

# Pkg.add("PyPlot")
using PyPlot
plot(phases,rvs_true,"b-")
 plot(phases,rvs_out_simple,"r.")
 plot(phases,rvs_out_gp,"g.")
 plot(phases,rvs_out_simple.-rvs_true,"r.")
 plot(phases,rvs_out_gp.-rvs_true,"g.")
 xlabel("Phase")
 ylabel("Velocity (m/s)")
