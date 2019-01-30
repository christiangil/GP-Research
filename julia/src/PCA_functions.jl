using JLD2, FileIO
using HDF5
using Distributions

"bootstrapping for errors in PCA scores. Takes about 28s per bootstrap on my computer"
function bootstrap_errors(; boot_amount::Int=10, time_series_spectra::Array{Float64,2}=zeros(2,2), hdf5_loc::String="", save_filename::String="bootstrap.jld2")

    @assert ((time_series_spectra != zeros(2,2)) | (hdf5_loc != "")) "either time_series_spectra or hdf5_loc have to be defined"
    if time_series_spectra==zeros(2,2)
        fid = h5open(hdf5_loc, "r")
        act = fid["active"]
        time_series_spectra = act[:, :]
    end

    @load "rv_data.jld2" doppler_comp genpca_out rvs_out
    mu, M, scores0 = genpca_out
    # scores0 = scores0[:, :]

    errors = 1e-2
    num_lambda = size(time_series_spectra, 1)
    num_spectra = size(time_series_spectra, 2)

    @assert size(time_series_spectra)==(num_lambda, num_spectra)

    num_components = size(M, 2)
    scores_tot_new = zeros(boot_amount, num_spectra, num_components)
    for k in 1:boot_amount
        scores = zeros(num_spectra, num_components)
        time_series_spectra_tmp = time_series_spectra .* (1 .+ (errors .* randn(num_lambda)))
        mu = vec(mean(time_series_spectra_tmp, dims=2))
        time_series_spectra_tmp .-= mu  # ~60s
        fixed_comp = M[:,1]
        fixed_comp_norm = 1/sum(abs2, fixed_comp)
        for i in 1:num_spectra
            scores[i, 1] = (dot(view(time_series_spectra_tmp, :, i), fixed_comp) * fixed_comp_norm)  # Normalize differently, so scores are z (i.e., doppler shift)
            time_series_spectra_tmp[:, i] -= scores[i, 1] * fixed_comp
        end
        for j in 2:num_components
            for i in 1:num_spectra
                scores[i, j] = dot(view(time_series_spectra_tmp, :, i), view(M, :, j)) #/sum(abs2,view(M,:,j-1))
                time_series_spectra_tmp[:, i] .-= scores[i, j] * view(M, :, j)
            end
        end
        scores_tot_new[k, :, :] = scores
    end


    if isfile(save_filename)
        @load save_filename scores_tot
        scores_tot = vcat(scores_tot, scores_tot_new)
    else
        scores_tot = scores_tot_new
    end

    error_ests = zeros(num_components, num_spectra)
    fit_normal(a::Array{Float64,1}) = fit_mle(Normal, a).σ

    for i in 1:num_components
        error_ests[i,:] = mapslices(fit_normal, scores_tot[:, :, i]; dims=1)
        # error_ests[i,:] = mapslices(std, scores_tot_new[:, :, i]; dims=1)
    end

    @save save_filename scores_tot error_ests
end
