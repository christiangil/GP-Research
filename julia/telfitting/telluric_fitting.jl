include("../src/general_functions.jl")
include("../src/PCA_functions.jl")
include("../src/SOAP_functions.jl")
include("../src/plotting_functions.jl")
include("../src/interpolation_functions.jl")
include("../src/RV_functions.jl")
const light_speed = uconvert(u"m/s",1u"c")
const light_speed_nu = ustrip(light_speed)

using NPZ

valid_obs = npzread("telfitting/valid_obs_res-1000-lambda-3923-6664-1years_1579spots_diffrot_id11.npy")
valid_inds = [i for i in 1:length(valid_obs) if valid_obs[i]]
times = npzread("telfitting/times_res-1000-lambda-3923-6664-1years_1579spots_diffrot_id11.npy")[valid_inds] .* 1u"d"
airmasses = npzread("telfitting/airmasses_res-1000-lambda-3923-6664-1years_1579spots_diffrot_id11.npy")[valid_inds]

blackbody = true
photon_noise = true
stellar_activity = true
velocity_offset = 14000.0u"m/s"
SNR = 1000
K = 10u"m/s"
T = 5700u"K"
min_wav = 615  # nm
max_wav = 665  # nm
obs_resolution = 200000

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

fid = h5open("D:/Christian/Downloads/res-1000-lambda-3923-6664-1years_1579spots_diffrot_id11.h5")
# fid = h5open("C:/Users/chris/Downloads/res-1000-lambda-3923-6664-1years_1579spots_diffrot_id11.h5")

# TODO fix no method iterate(::Nothing) MethodError
# actives, λ, quiet = prep_SOAP_spectra(fid; return_quiet=true)
########################################################################


λ = fid["lambdas"][:]u"nm"/10
# actives, normalization = normalize_columns_to_first_integral!(fid["active"][:, 1:2] .* planck.(λ, T), ustrip.(λ); return_normalization=true)
if stellar_activity
    ys = fid["active"][:, :] .* (planck.(λ, T) .* Int(blackbody))
else
    thing = fid["quiet"][:] .* (planck.(λ, T) .* Int(blackbody))
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
quiet = fid["quiet"][:] .* (planck.(λ, T) .* Int(blackbody))
quiet*= integrated_first / trapz(ustrip.(λ), quiet)

########################################################################
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

inds2 = [λ1 > 664.3 && λ1 < 664.4 for λ1 in λ]
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
    return (o2scale .* o2mask .+ (1 - o2scale)) .* (h2oscale .* h2omask .+ (1 - h2oscale))
end


λ_obs = λ[inds]
ks = kep_signal(K=K, e_or_h=0.1, P=sqrt(800)u"d")
rvs_true = ks.(times)
vs = velocity_offset .+ rvs_true

valid_obs = zeros(length(obs_waves), length(valid_inds))
@time for i in 1:length(valid_inds)
    valid_obs[:, i] = spectra_interpolate(obs_waves,
        λ_obs .* sqrt((1.0+vs[i]/light_speed)/(1.0-vs[i]/light_speed)),
        actives[:, i]) .* tellurics(airmasses[i])
end

# normalize_columns_to_first_integral!(valid_obs, obs_waves)

average_spectra = vec(mean(valid_obs, dims=2))
doppler_comp = calc_doppler_component_RVSKL(obs_waves, average_spectra)
x, x, x, x, rvs_naive = @time fit_gen_pca_rv_RVSKL(valid_obs, doppler_comp, mu=average_spectra, num_components=1)

mu, M, scores, fracvar = fit_gen_pca(valid_obs; num_components=2)
init_plot()
plot(obs_waves, -M[:,1], label="PCA Component 1")
plot(obs_waves, -M[:,2], label="PCA Component 2")
tels = tellurics(mean(airmasses); add_rand=false) .- 1
tels /= norm(tels)
plot(obs_waves, tels, label="Telluric mask")
plot(obs_waves, doppler_comp ./ norm(doppler_comp), label="Analytical Doppler Component")
legend(;fontsize=20)
save_PyPlot_fig("components_vs_tellurics_SNR$(SNR)_K$(ustrip(K)).png")

tel_spectra = zeros(size(valid_obs))
# for (j, s) in enumerate(scores[1, :])
#     tel_spectra[:, j] = (M[:, 1] .* s) + mu
# end
# new_spectra = (valid_obs ./ tel_spectra) .* mu
for (j, s) in enumerate(scores[1, :])
    tel_spectra[:, j] = M[:, 1] .* s
end
init_plot()
plot(obs_waves, tel_spectra[:,150])
save_PyPlot_fig("test.png")

sum(scores[1, :] .< 0)

new_spectra = valid_obs .- tel_spectra
average_spectra = vec(mean(new_spectra, dims=2))
doppler_comp = calc_doppler_component_RVSKL(obs_waves, average_spectra)
x, x, x, x, rvs_notel = @time fit_gen_pca_rv_RVSKL(new_spectra, doppler_comp, mu=average_spectra, num_components=1)

init_plot()
scatter(ustrip.(times .% ks.P), ustrip.(rvs_true) + rvs_activ, label="Planet + Activity")
scatter(ustrip.(times .% ks.P), -rvs_naive, label="Naive")
scatter(ustrip.(times .% ks.P), -rvs_notel, label="Tels subtracted")
scatter(ustrip.(times .% ks.P), ustrip.(rvs_true), label="Input")
legend(;fontsize=20)
save_PyPlot_fig("rv_comparison_SNR$(SNR)_K$(ustrip(K)).png")

init_plot()
scatter(ustrip.(times), rvs_activ, label="Activity")
scatter(ustrip.(times), ustrip.(rvs_true)+rvs_naive, label="Naive")
scatter(ustrip.(times), ustrip.(rvs_true)+rvs_notel, label="Tels subtracted")
axhline(color="black")
legend(;fontsize=20)
save_PyPlot_fig("rv_comparison_resid_SNR$(SNR)_K$(ustrip(K)).png")


print("RVs from activity:                       ", std(rvs_activ))
print("Estimated RVs:                           ", std(ustrip.(rvs_true) + rvs_naive))
print("Estimated RVs (trying to take out tels): ", std(ustrip.(rvs_true) + rvs_notel))
