include("../src/general_functions.jl")
include("../src/PCA_functions.jl")
include("../src/SOAP_functions.jl")
include("../src/plotting_functions.jl")
include("../src/interpolation_functions.jl")
include("../src/RV_functions.jl")
const light_speed = uconvert(u"m/s",1u"c")
const light_speed_nu = ustrip(light_speed)

using NPZ

## Getting precomputed airmasses and observation times
valid_obs = npzread("telfitting/valid_obs_res-1000-lambda-3923-6664-1years_1579spots_diffrot_id11.npy")
valid_inds = [i for i in 1:length(valid_obs) if valid_obs[i]]
times = npzread("telfitting/times_res-1000-lambda-3923-6664-1years_1579spots_diffrot_id11.npy")[valid_inds] .* 1u"d"
airmasses = npzread("telfitting/airmasses_res-1000-lambda-3923-6664-1years_1579spots_diffrot_id11.npy")[valid_inds]


## Setup
photon_noise = true
stellar_activity = false
bary_velocity = false
SNR = 1000
K = 10u"m/s"
T = 5700u"K"
min_wav = 615  # nm
max_wav = 665  # nm
obs_resolution = 200000


## Reading in O2 and H2O lines from HITRAN
old_dir = pwd()
cd(@__DIR__)
f = open("outputtransitionsdata3.par", "r")
n_lines = countlines(open("outputtransitionsdata3.par", "r"))
cd(old_dir)
is_h2o = [false for i in 1:n_lines]
is_o2 = [false for i in 1:n_lines]
intensities = zeros(n_lines)
wavelengths = zeros(n_lines)
for i in 1:n_lines
    new_line = readline(f)
    if length(new_line) > 0
        is_h2o[i] = new_line[2] == '1'
        is_o2[i] = new_line[2] == '7'
        intensities[i] = parse(Float64, new_line[17:25])
        wavelengths[i] = 1e7 / parse(Float64, new_line[4:15])
    end
end

intens_h2o = intensities[is_h2o]
wavelengths_h2o = wavelengths[is_h2o]
max_intens_h2o_inds = sortperm(intens_h2o)[end-9:end][end:-1:1]
wavelengths_h2o[max_intens_h2o_inds]
intens_h2o[max_intens_h2o_inds]

intens_o2 = intensities[is_o2]
wavelengths_o2 = wavelengths[is_o2]
max_intens_o2_inds = sortperm(intens_o2)[end-1:end][end:-1:1]
wavelengths_o2[max_intens_o2_inds]
intens_o2[max_intens_o2_inds]

## Simulating observations

fid = h5open("D:/Christian/Downloads/res-1000-lambda-3923-6664-1years_1579spots_diffrot_id11.h5")

#######
# Recreating prep_SOAP_spectra functionality which isn't working for some reason
# TODO fix no method iterate(::Nothing) MethodError
# actives, λ, quiet = prep_SOAP_spectra(fid; return_quiet=true)

λ = fid["lambdas"][:]u"nm"/10
# actives, normalization = normalize_columns_to_first_integral!(fid["active"][:, 1:2] .* planck.(λ, T), ustrip.(λ); return_normalization=true)
if stellar_activity
    ys = fid["active"][:, :] .* planck.(λ, T)
else
    thing = fid["quiet"][:] .* planck.(λ, T)
    ys = zeros(length(thing), 730)
    for i in 1:730
        ys[:, i] = thing
    end
end
x = ustrip.(λ)
integrated_first = trapz(x, ys[:, 1])
for i in 1:size(ys, 2)
    ys[:, i] *= integrated_first / trapz(x, ys[:, i])
end
actives = ys
normalization = integrated_first
quiet = fid["quiet"][:] .* planck.(λ, T)
quiet*= integrated_first / trapz(ustrip.(λ), quiet)

#######
λ = ustrip.(λ)
actives = actives[:, valid_inds]

average_spectra = vec(mean(actives, dims=2))
doppler_comp = calc_doppler_component_RVSKL(λ, average_spectra)
x, x, x, x, rvs_activ = @time fit_gen_pca_rv_RVSKL(actives, doppler_comp, mu=average_spectra, num_components=1)

if photon_noise
    @time active = make_noisy_SOAP_spectra(actives, λ; SNR=SNR)
else
    active = copy(actives)
end

inds = [λ1 > (min_wav - 1) && λ1 < (max_wav + 1) for λ1 in λ]
psf_width = 50 / sum(inds) * 6  # 4 for NEID, 8-12 for EXPRESS

## Creating telluric mask

inds2 = [λ1 > 664.3 && λ1 < 664.4 for λ1 in λ]
# inds2 = [λ1 > 635.8 && λ1 < 635.9 for λ1 in λ]
# init_plot()
# plot(λ[inds2], quiet[inds2])
# save_PyPlot_fig("test.png")
quiet_line = quiet[inds2]
quiet_λ = λ[inds2]

HM = (maximum(quiet_line) + minimum(quiet_line)) / 2

hmm = quiet_λ[quiet_line .< HM]
FWHM = hmm[end] - hmm[1]
solar_sigma = FWHM / (2 * sqrt(2 * log(2)))
telluric_sigma = solar_sigma / sqrt(2)

n_mask = Int(round((max_wav - min_wav) / (sqrt((max_wav^2 + min_wav^2) / 2) / obs_resolution)))
obs_waves = log_linspace(min_wav, max_wav, n_mask)

gauss(x::Real; a::Real=1, loc::Real=0, sigma::Real=1) = a * exp(-((x-loc) / sigma)^2/2)

global h2omask = ones(n_mask)

for wave in wavelengths_h2o[max_intens_h2o_inds]
    global h2omask .*= 1 .- gauss.(obs_waves, a=0.3 + 0.5*rand(), loc=wave, sigma=telluric_sigma)
end
wavelengths_h2o[max_intens_h2o_inds]

global o2mask = ones(n_mask)

for wave in wavelengths_o2[max_intens_o2_inds]
    o2mask .*= 1 .- gauss.(obs_waves, a=0.2 + 0.6*rand(), loc=wave, sigma=telluric_sigma)
end

function tellurics(airmass::Real; add_rand::Bool=true)
    o2scale = airmass / 2
    h2oscale = airmass / 2 * (0.7 + 0.6*(rand()-0.5)*add_rand)
    return (o2mask .^ o2scale) .* (h2omask .^ h2oscale)
end


## Bringing observations into observer frame and multiplying by telluric mask

λ_obs = λ[inds]
planet_ks = kep_signal(K=K, e_or_h=0.1, P=sqrt(800)u"d")
bary_ks = kep_signal(K=15u"km/s", e_or_h=0.016, P=1u"yr", M0=rand()*2π, ω_or_k=rand()*2π)
vs = planet_ks.(times) + (Int(bary_velocity) .* bary_ks.(times))

valid_obs = zeros(length(obs_waves), length(valid_inds))
true_tels = zeros(length(obs_waves), length(valid_inds))
@time for i in 1:length(valid_inds)
    true_tels[:, i] = tellurics(airmasses[i])
    valid_obs[:, i] = spectra_interpolate(obs_waves,
        λ_obs .* sqrt((1.0+vs[i]/light_speed)/(1.0-vs[i]/light_speed)),
        active[:, i]) .* true_tels[:, i]
end

## Estimating tellurics

average_spectra = vec(mean(valid_obs, dims=2))
doppler_comp = calc_doppler_component_RVSKL(obs_waves, average_spectra)
x, x, x, x, rvs_naive = @time fit_gen_pca_rv_RVSKL(valid_obs, doppler_comp, mu=average_spectra, num_components=1)

log_valid_obs = log.(valid_obs)

mu, M, scores, fracvar = fit_gen_pca(log_valid_obs; num_components=2)
init_plot()
plot(obs_waves, exp.(-M[:,1]), label="e^PCA1", alpha=0.5)
plot(obs_waves, exp.(-M[:,2]), label="e^PCA2", alpha=0.5)
plot(obs_waves, tellurics(mean(airmasses); add_rand=false), label="Telluric mask", alpha=0.5)
# plot(obs_waves, doppler_comp ./ norm(doppler_comp), label="Analytical Doppler Component")
legend(;fontsize=20)
save_PyPlot_fig("components_vs_tellurics_SNR$(SNR)_K$(ustrip(K)).png")

est_tels = zeros(size(log_valid_obs))
for (j, s) in enumerate(scores[1, :])
    est_tels[:, j] += M[:, 1] .* s
end
# for (j, s) in enumerate(scores[2, :])
#     est_tels[:, j] += M[:, 2] .* s
# end

## Taking out estimated tellurics?

new_spectra = exp.(log_valid_obs .- est_tels)
average_spectra = vec(mean(new_spectra, dims=2))
doppler_comp = calc_doppler_component_RVSKL(obs_waves, average_spectra)
x, x, x, x, rvs_notel = @time fit_gen_pca_rv_RVSKL(new_spectra, doppler_comp, mu=average_spectra, num_components=1)

init_plot()
times_mod = ustrip.(times .% ks.P)
scatter(times_mod, ustrip.(rvs_true) + rvs_activ, label="Planet + Activity")
scatter(times_mod, -rvs_naive, label="Naive")
scatter(times_mod, -rvs_notel, label="Tels subtracted")
scatter(times_mod, ustrip.(rvs_true), label="Input")
legend(;fontsize=20)
save_PyPlot_fig("rv_comparison_SNR$(SNR)_K$(ustrip(K)).png")

init_plot()
scatter(ustrip.(times), rvs_activ, label="Activity")
scatter(ustrip.(times), ustrip.(rvs_true)+rvs_naive, label="Naive")
scatter(ustrip.(times), ustrip.(rvs_true)+rvs_notel, label="Tels subtracted")
axhline(color="black")
legend(;fontsize=20)
save_PyPlot_fig("rv_comparison_resid_SNR$(SNR)_K$(ustrip(K)).png")

println("RVs from activity:                       ", std(rvs_activ))
println("Estimated RVs:                           ", std(ustrip.(rvs_true) + rvs_naive))
println("Estimated RVs (trying to take out tels): ", std(ustrip.(rvs_true) + rvs_notel))
# 
# init_plot()
# plot(obs_waves, true_tels[:, 5], label = "true")
# plot(obs_waves, exp.(est_tels[:, 5]), label = "est")
# legend(;fontsize=20)
# save_PyPlot_fig("test.png")
