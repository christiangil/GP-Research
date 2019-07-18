# all of the functions to simplify using PyPlot
# https://github.com/JuliaPy/PyPlot.jl

using PyPlot


# quick and dirty function for creating plots that show what I want
function custom_GP_plot(x_samp::Vector{T}, show_curves::Matrix{T}, x_obs::Vector{T}, y_obs::Vector{T}, σ::Vector{T}, mean::Vector{T}; errors::Vector{T}=zero(x_obs), mean_obs::Vector{T}=zero(x_obs)) where {T<:Real}

    @assert size(show_curves, 2)==length(x_samp)==length(mean)==length(σ)
    @assert length(x_obs)==length(y_obs)==length(errors)

    if mean_obs == zero(x_obs)

        ax = init_plot()

        # filling the ±1 σ with a transparent orange
        fill_between(x_samp, mean + σ, mean - σ, alpha=0.5, color="orange")

        plot(x_samp, mean, color="black", zorder=2)

        for i in 1:size(show_curves, 1)
            plot(x_samp, show_curves[i, :], alpha=0.5, zorder=1)
        end

        scatter(x_obs, y_obs, color="black", zorder=2)
        errorbar(x_obs, y_obs, yerr=[errors';errors'], fmt="o", color="black", zorder=2)

        return [ax]
    else

        # initialize figure size
        init_plot(; hspace=0)

        ax1 = subplot2grid((4, 1), (0, 0), rowspan=3)
        set_font_sizes(ax1)
        ax2 = subplot2grid((4, 1), (3, 0))
        set_font_sizes(ax2)

        # filling the ±1 σ with a transparent orange
        ax1.fill_between(x_samp, mean + σ, mean - σ, alpha=0.5, color="orange")

        ax1.plot(x_samp, mean, color="black", zorder=2)

        for i in 1:size(show_curves, 1)
            ax1.plot(x_samp, show_curves[i, :], alpha=0.5, zorder=1)
        end

        ax1.scatter(x_obs, y_obs, color="black", zorder=2)
        ax1.errorbar(x_obs, y_obs, yerr=[errors';errors'], fmt="o", color="black", zorder=2)

        resids = y_obs - mean_obs
        ax2.scatter(x_obs, resids, color="black")
        ax2.errorbar(x_obs, resids, yerr=[errors';errors'], fmt="o", color="black")
        ax2.fill_between(x_samp, σ, -σ, alpha=0.5, color="orange")
        ax2.axhline(y=0, color="black")

        return [ax1, ax2]
    end

end


function Jones_line_plots(amount_of_samp_points::Integer, prob_def::Jones_problem_definition, total_hyperparameters::Vector{T}, file::AbstractString; show::Integer=0, find_post::Bool=true, plot_Σ::Bool=false, plot_Σ_profile::Bool=false, filetype::AbstractString="png") where {T<:Real}

    x_samp = collect(linspace(minimum(prob_def.x_obs), maximum(prob_def.x_obs), amount_of_samp_points))
    amount_of_total_samp_points = amount_of_samp_points * prob_def.n_out
    amount_of_obs = length(prob_def.x_obs)

    show_curves = zeros(show, amount_of_total_samp_points)

    # calculate mean, σ, and show_curves
    if find_post
        show_resids = true
        mean, σ, mean_obs, Σ = GP_posteriors(prob_def, x_samp, total_hyperparameters; return_mean_obs=true)
        if plot_Σ; plot_im(Σ, file = file * "_K_post." * filetype) end
        L = ridge_chol(Σ).L
        for i in 1:show
            show_curves[i,:] = L * randn(amount_of_total_samp_points) + mean
        end
    # if no posterior is being calculated, estimate σ with sampling
    else
        show_resids = false
        mean = zeros(amount_of_total_samp_points)
        Σ = covariance(prob_def, x_samp, x_samp, total_hyperparameters)
        if plot_Σ
            plot_im(Σ, file = file * "_K_prior." * filetype)
        end
        L = ridge_chol(Σ).L

        # calculate a bunch of GP draws for a σ estimation
        draws = 5000
        storage = zeros((draws, amount_of_total_samp_points))
        for i in 1:draws
            storage[i, :] = (L * randn(amount_of_total_samp_points)) + mean
        end
        show_curves[:, :] = storage[1:show, :]
        storage = sort(storage, dims=1)
        σ = storage[Int(round(0.84135 * draws)), :] - storage[Int(round(0.15865 * draws)), :] ./ 2
    end

    for output in 1:prob_def.n_out

        if plot_Σ_profile
            init_plot()
            fig = plot(collect(1:(prob_def.n_out * amount_of_samp_points)) / amount_of_samp_points, Σ[convert(Int64, round((output - 1 / 2) * amount_of_samp_points)),:])
            axvline(x=1, color="black")
            axvline(x=2, color="black")
            ylabel("Covariance")
            title("Covariance profile of Output $(output-1)", fontsize=30)
            save_PyPlot_fig(file * "_K_profile_$output." * filetype)
        end

        # the indices corresponding to the proper output
        sample_output_indices = (amount_of_samp_points * (output - 1) + 1):(amount_of_samp_points * output)
        obs_output_indices = (amount_of_obs * (output - 1) + 1):(amount_of_obs * output)

        # getting the y values for the proper output
        y_o = prob_def.y_obs[obs_output_indices]
        obs_noise_o = prob_def.noise[obs_output_indices]
        show_curves_o = show_curves[:, sample_output_indices]
        σ_o = σ[sample_output_indices]
        mean_o = mean[sample_output_indices]
        if output==1
            y_o ./= prob_def.normals[output]
            obs_noise_o ./= prob_def.normals[output]
            show_curves_o ./= prob_def.normals[output]
            σ_o ./= prob_def.normals[output]
            mean_o ./= prob_def.normals[output]
        end

        show_resids ? axs = custom_GP_plot(x_samp, show_curves_o, prob_def.x_obs, y_o, σ_o, mean_o; errors=obs_noise_o, mean_obs=mean_obs[obs_output_indices]) : axs = custom_GP_plot(x_samp, show_curves_o, prob_def.x_obs, y_o, σ_o, mean_o; errors=obs_noise_o)

        if output==1
            y_str = "RVs (" * string(prob_def.y_obs_units) * ")"
            title_string = "Apparent RVs"
        else
            y_str = "Normalized scores"
            title_string = "DCPCA Component " * string(output-1)
        end
        axs[1].set_ylabel(y_str)
        axs[1].set_title(title_string, fontsize=45)

        if find_post
            # put log likelihood on plot
            LogL = -nlogL_Jones(prob_def, total_hyperparameters)
            show > 0 ? max_val = maximum([maximum(y_o + obs_noise_o), maximum(show_curves_o), maximum(mean_o + σ_o)]) : max_val = maximum([maximum(y_o + obs_noise_o), maximum(mean_o + σ_o)])
            axs[1].text(minimum(prob_def.x_obs), 0.9 * max_val, L"\ell_{act}(\theta|t,s): " * string(convert(Int64, round(LogL))), fontsize=30)
        end

        # put kernel lengths on plot
        kernel_lengths = total_hyperparameters[end-prob_def.n_kern_hyper+1:end]
        show > 0 ? min_val = minimum([minimum(y_o - obs_noise_o), minimum(show_curves_o), minimum(mean_o - σ_o)]) : min_val = minimum([minimum(y_o - obs_noise_o), minimum(mean_o - σ_o)])
        axs[1].text(minimum(prob_def.x_obs), 1 * min_val, "Hyperparameters: " * string(kernel_lengths), fontsize=30)
        # text(minimum(prob_def.x_obs), 1 *minimum(y_o), "Wavelengths: " * string(kernel_lengths), fontsize=30)

        @assert 0<length(axs)<3
        axs[end].set_xlabel("Time (" * string(prob_def.x_obs_units) * ")")
        if length(axs)==2; axs[end].set_ylabel("Residuals") end

        save_PyPlot_fig(file * "_$output." * filetype)

    end

    PyPlot.close("all")

end
