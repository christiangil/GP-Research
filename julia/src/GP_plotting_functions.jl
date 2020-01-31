# all of the functions to simplify using PyPlot
# https://github.com/JuliaPy/PyPlot.jl

using PyPlot


# quick and dirty function for creating plots that show what I want
function custom_GP_plot(
    x_samp::Vector{T},
    show_curves::Matrix{T},
    x_obs::Vector{T},
    y_obs::Vector{T},
    σ::Vector{T},
    mean::Vector{T};
    errors::Vector{T}=zero(x_obs),
    mean_obs::Vector{T}=zero(x_obs)
    ) where {T<:Real}

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

        scatter(x_obs, y_obs, color="black", zorder=2) # could comment out?
        errorbar(x_obs, y_obs, yerr=[errors';errors'], fmt="o", color="black", zorder=2)

        return [ax]
    else

        # initialize figure size
        init_plot(; hspace=0, wspace=0)

        ax1 = subplot2grid((4, 6), (0, 0), rowspan=3, colspan=5)
        set_font_sizes(ax1)
        ax2 = subplot2grid((4, 6), (3, 0), colspan=5)
        set_font_sizes(ax2)
        ax3 = subplot2grid((4, 6), (3, 5))
        set_font_sizes(ax3)

        # filling the ±1 σ with a transparent orange
        ax1.fill_between(x_samp, mean + σ, mean - σ, alpha=0.5, color="orange")

        ax1.plot(x_samp, mean, color="black", zorder=2)

        for i in 1:size(show_curves, 1)
            ax1.plot(x_samp, show_curves[i, :], alpha=0.5, zorder=1)
        end

        ax1.scatter(x_obs, y_obs, color="black", zorder=2)
        ax1.errorbar(x_obs, y_obs, yerr=[errors';errors'], fmt="o", color="black", zorder=2)
        ax1.get_xaxis().set_visible(false)

        resids = y_obs - mean_obs
        ax2.scatter(x_obs, resids, color="black")
        ax2.errorbar(x_obs, resids, yerr=[errors';errors'], fmt="o", color="black")
        ax2.fill_between(x_samp, σ, -σ, alpha=0.5, color="orange")
        ax2.axhline(y=0, color="black")

        ax3.hist(resids, bins=convert(Int64, round(length(resids) / 10)), orientation="horizontal")
        ax3.axis("off")

        return [ax1, ax2]
    end

end


function Jones_line_plots(
    prob_def::Jones_problem_definition,
    total_hyperparameters::Vector{T},
    file::AbstractString;
    show::Integer=0,
    plot_Σ::Bool=false,
    plot_Σ_profile::Bool=false,
    filetype::AbstractString="png",
    fit_ks::Union{kep_signal, kep_signal_epicyclic, kep_signal_wright, kep_signal_circ}=kep_signal(;K=0u"m/s"),
    hyper_param_string::String="",
    n_samp_points::Integer = convert(Int64, max(500, round(2 * sqrt(2) * length(prob_def.x_obs))))
    ) where {T<:Real}

    x_samp = collect(linspace(minimum(prob_def.x_obs), maximum(prob_def.x_obs), n_samp_points))
    n_total_samp_points = n_samp_points * prob_def.n_out
    n_meas = length(prob_def.x_obs)

    show_curves = zeros(show, n_total_samp_points)

    # calculate mean, σ, and show_curves
    # if find_post
    show_resids = true
    mean_GP, σ, mean_GP_obs, Σ = GP_posteriors(prob_def, x_samp, total_hyperparameters; return_mean_obs=true, y_obs=remove_kepler(prob_def, fit_ks))

    # init_plot()
    # plot(x_samp, mean_GP[1:n_samp_points])
    # plot(prob_def.x_obs, mean_GP_obs[1:n_meas])
    # save_PyPlot_fig("test.png")

    if plot_Σ; plot_im(Σ, file = file * "_K_post." * filetype) end
    L = ridge_chol(Σ).L
    for i in 1:show
        show_curves[i,:] = L * randn(n_total_samp_points) + mean_GP
    end
    # if no posterior is being calculated, estimate σ with sampling
    # else
    #     show_resids = false
    #     mean_GP = zeros(n_total_samp_points)
    #     Σ = covariance(prob_def, x_samp, x_samp, total_hyperparameters)
    #     if plot_Σ
    #         plot_im(Σ, file = file * "_K_prior." * filetype)
    #     end
    #     L = ridge_chol(Σ).L
    #
    #     # calculate a bunch of GP draws for a σ estimation
    #     draws = 5000
    #     storage = zeros((draws, n_total_samp_points))
    #     for i in 1:draws
    #         storage[i, :] = (L * randn(n_total_samp_points)) + mean
    #     end
    #     show_curves[:, :] = storage[1:show, :]
    #     storage = sort(storage, dims=1)
    #     σ = storage[Int(round(0.84135 * draws)), :] - storage[Int(round(0.15865 * draws)), :] ./ 2
    # end

    mean = add_kepler(mean_GP, x_samp .* prob_def.time_unit, fit_ks; data_unit=prob_def.rv_unit*prob_def.normals[1])
    # mean = copy(mean_GP)
    mean_obs = add_kepler(mean_GP_obs, prob_def.time, fit_ks; data_unit=prob_def.rv_unit*prob_def.normals[1])
    # mean_obs = copy(mean_GP_obs)

    for output in 1:prob_def.n_out

        if plot_Σ_profile
            init_plot()
            fig = plot(collect(1:(prob_def.n_out * n_samp_points)) / n_samp_points, Σ[convert(Int64, round((output - 1 / 2) * n_samp_points)),:])
            axvline(x=1, color="black")
            axvline(x=2, color="black")
            ylabel("Covariance")
            title("Covariance profile of Output $(output-1)", fontsize=30)
            save_PyPlot_fig(file * "_K_profile_$output." * filetype)
        end

        # the indices corresponding to the proper output
        # sample_output_indices = (n_samp_points * (output - 1) + 1):(n_samp_points * output)
        # obs_output_indices = (n_meas * (output - 1) + 1):(n_meas * output)
        sample_output_indices = output:prob_def.n_out:n_total_samp_points
        obs_output_indices = output:prob_def.n_out:length(prob_def.y_obs)

        # getting the y values for the proper output
        y_o = prob_def.y_obs[obs_output_indices]
        obs_noise_o = prob_def.noise[obs_output_indices]
        show_curves_o = show_curves[:, sample_output_indices]
        σ_o = σ[sample_output_indices]
        mean_o = mean[sample_output_indices]
        # if find_post
        mean_obs_o = mean_obs[obs_output_indices]
        # end

        if output==1
            y_o .*= prob_def.normals[output]
            obs_noise_o .*= prob_def.normals[output]
            show_curves_o .*= prob_def.normals[output]
            σ_o .*= prob_def.normals[output]
            mean_o .*= prob_def.normals[output]
            # if find_post
            mean_obs_o .*= prob_def.normals[output]
            # end
        end

        show_resids ? axs = custom_GP_plot(x_samp, show_curves_o, prob_def.x_obs, y_o, σ_o, mean_o; errors=obs_noise_o, mean_obs=mean_obs_o) : axs = custom_GP_plot(x_samp, show_curves_o, prob_def.x_obs, y_o, σ_o, mean_o; errors=obs_noise_o)

        if output==1
            y_str = "RVs (" * string(prob_def.rv_unit) * ")"
            title_string = "Apparent RVs"
        else
            y_str = "Normalized scores"
            title_string = "DCPCA Component " * string(output-1)
        end
        axs[1].set_ylabel(y_str)
        axs[1].set_title(title_string, fontsize=45)

        # if find_post
        # put log likelihood on plot
        logL = -nlogL_Jones(prob_def, total_hyperparameters; y_obs=remove_kepler(prob_def, fit_ks))
        show > 0 ? max_val = maximum([maximum(y_o + obs_noise_o), maximum(show_curves_o), maximum(mean_o + σ_o)]) : max_val = maximum([maximum(y_o + obs_noise_o), maximum(mean_o + σ_o)])
        axs[1].text(minimum(prob_def.x_obs), 0.9 * max_val, L"\ell_{act}(\theta|t,s): " * string(convert(Int64, round(logL))), fontsize=30)
        # end

        # put kernel lengths on plot
        # kernel_lengths = total_hyperparameters[end-prob_def.n_kern_hyper+1:end]
        show > 0 ? min_val = minimum([minimum(y_o - obs_noise_o), minimum(show_curves_o), minimum(mean_o - σ_o)]) : min_val = minimum([minimum(y_o - obs_noise_o), minimum(mean_o - σ_o)])

        axs[1].text(minimum(prob_def.x_obs), 1 * min_val, hyper_param_string, fontsize=30)
        # text(minimum(prob_def.x_obs), 1 *minimum(y_o), "Wavelengths: " * string(kernel_lengths), fontsize=30)

        @assert 0<length(axs)<3
        axs[end].set_xlabel("Time (" * string(prob_def.time_unit) * ")")
        if length(axs)==2; axs[end].set_ylabel("Residuals") end

        save_PyPlot_fig(file * "_$output." * filetype)

    end

    if fit_ks.K != 0u"m/s"
        mod_x_obs = convert_and_strip_units.(u"d", mod.(prob_def.time, fit_ks.P))
        samp_x = collect(linspace(0, ustrip(fit_ks.P), 1000)) * unit(fit_ks.P)
        keps = fit_ks.(samp_x)
        # ys = ustrip.(prob_def.rv) - mean_GP_obs[1:prob_def.n_out:end] .* prob_def.normals[1]
        ys = (prob_def.y_obs[1:prob_def.n_out:end] - mean_GP_obs[1:prob_def.n_out:end]) .* prob_def.normals[1]
        # noises = convert_and_strip_units.(unit(prob_def.rv[1]), prob_def.rv_noise)
        noises = prob_def.noise[1:prob_def.n_out:end] .* prob_def.normals[1]

        init_plot()
        plot(convert_and_strip_units.(u"d", samp_x), ustrip.(keps), linewidth=4)
        scatter(mod_x_obs, ys, color="black")
        errorbar(mod_x_obs, ys, yerr=[noises';noises'], fmt="o", color="black")
        xlabel("Time (" * string(prob_def.time_unit) * ")")
        ylabel("RVs (" * string(prob_def.rv_unit) * ")")
        title("Folded (GP subtracted) apparent RVs", fontsize=45)
        text(0, 0.9 * maximum([maximum(ys + noises), maximum(ustrip.(keps))]), kep_parms_str_short(fit_ks), fontsize=30)
        save_PyPlot_fig(file * "_planet." * filetype)
    end

    PyPlot.close("all")

end
