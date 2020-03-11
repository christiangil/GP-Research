include("src/all_functions.jl")

##########################
# Importing David's data #
##########################

spectra_home = "D:/Christian/Downloads/config4_1000_grid/config4_1000_grid/"
spot_files = readdir(spectra_home)[2:end]

# get the phases from the file names
n_spectra = length(spot_files)
spot_phase = zeros(n_spectra)
for i in 1:n_spectra
  str_tmp = split(spot_files[i],"phase_")[2]
  spot_phase[i] = parse(Float64, split(str_tmp,".c")[1])
end

# look at the first spectra to get the number of wavelengths
tmp = CSV.read(spectra_home * spot_files[1])
n_wavelengths = length(tmp.Wavelength)
wavelengths = tmp.Wavelength

# build matrix with rows corresponding to observations and columns to wavelength
spot_data = zeros((n_spectra,n_wavelengths))
for i in 1:n_spectra
  spot_data[i, :] = CSV.read(spectra_home * spot_files[i]).Intensity
end

n_obs = 100
phase_index = sort(sample(1:125,n_obs,replace=false))

# reorder the data by phase; might not be necessary now that we jiggle them below anyway
ind = sortperm(spot_phase)[phase_index]
spot_data = spot_data[ind,:]
spot_phase = spot_phase[ind]
n_raw = length(spot_phase)

#########################
# Using renormalization #
#########################

λs = wavelengths * u"nm"/10
obs = normalize_columns_to_first_integral!(spot_data' .* planck.(λs, 5700u"K"), ustrip.(λs))
λs = copy(wavelengths)
obs_noisy = make_noisy_SOAP_spectra(obs, λs; SNR=500)

average_spectra = vec(mean(obs_noisy, dims=2))
average_spectra2 = vec(mean(obs, dims=2))

doppler_comp = calc_doppler_component_RVSKL(λs, average_spectra)
doppler_comp2 = calc_doppler_component_RVSKL(λs, average_spectra2)

mu, M, scores, fracvar, rvs = @time fit_gen_pca_rv_RVSKL(obs_noisy, doppler_comp, mu=average_spectra, num_components=6)
mu2, M2, scores2, fracvar2, rvs2 = @time fit_gen_pca_rv_RVSKL(obs, doppler_comp2, mu=average_spectra2, num_components=6)

init_plot()
subplot(221)
scatter(spot_phase, scores[1, :] .* light_speed_nu)
scatter(spot_phase, scores2[1, :] .* light_speed_nu)
for i in 2:4
	subplot(220 + i)
    factor =
    # scatter(spot_phase, hmm.proj[:, i-1] / maximum(abs.(hmm.proj[:, i-1])))
	# scatter(spot_phase, scores[i, :] / maximum(abs.(scores[i, :])))
	scatter(spot_phase, scores[i, :])
	scatter(spot_phase, scores2[i, :])
end
save_PyPlot_fig("renormalization.png")

#############################
# Not using renormalization #
#############################

λs = wavelengths * u"nm"/10
obs = spot_data' .* planck.(λs, 5700u"K")
λs = copy(wavelengths)
obs_noisy = make_noisy_SOAP_spectra(obs, λs; SNR=500)

average_spectra = vec(mean(obs_noisy, dims=2))
average_spectra2 = vec(mean(obs, dims=2))

doppler_comp = calc_doppler_component_RVSKL(λs, average_spectra)
doppler_comp2 = calc_doppler_component_RVSKL(λs, average_spectra2)

mu, M, scores, fracvar, rvs = @time fit_gen_pca_rv_RVSKL(obs_noisy, doppler_comp, mu=average_spectra, num_components=6)
mu2, M2, scores2, fracvar2, rvs2 = @time fit_gen_pca_rv_RVSKL(obs, doppler_comp2, mu=average_spectra2, num_components=6)

init_plot()
subplot(221)
scatter(spot_phase, scores[1, :] .* light_speed_nu)
scatter(spot_phase, scores2[1, :] .* light_speed_nu)
for i in 2:4
	subplot(220 + i)
    # scatter(spot_phase, hmm.proj[:, i-1] / maximum(abs.(hmm.proj[:, i-1])))
	# scatter(spot_phase, scores[i, :] / maximum(abs.(scores[i, :])))
	scatter(spot_phase, scores[i, :])
	scatter(spot_phase, scores2[i, :])
end
save_PyPlot_fig("no_renormalization.png")
