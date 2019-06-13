# all of the functions to simplify using PyPlot
# https://github.com/JuliaPy/PyPlot.jl

using PyPlot


# quick and dirty function for creating plots that show what I want
function custom_GP_plot(x_samp::Vector{T}, show_curves::Matrix{T}, x_obs::Vector{T}, y_obs::Vector{T}, σ::Vector{T}, mean::Vector{T}; errors::Vector{T}=zero(x_obs)) where {T<:Real}

    @assert size(show_curves, 2)==length(x_samp)==length(mean)==length(σ)
    @assert length(x_obs)==length(y_obs)==length(errors)

    # initialize figure size
    init_plot()

    # filling the 5-95th percentile with a transparent orange
    fill_between(x_samp, mean + 1.96 * σ, mean - 1.96 * σ, alpha=0.5, color="orange")

    plot(x_samp, mean, color="black", zorder=2)

    for i in 1:size(show_curves, 1)
        plot(x_samp, show_curves[i, :], alpha=0.5, zorder=1)
    end

    scatter(x_obs, y_obs, color="black", zorder=2)
    errorbar(x_obs, y_obs, yerr=[errors';errors'], fmt="o", color="black", zorder=2)

end


function Jones_line_plots(amount_of_samp_points::Integer, prob_def::Jones_problem_definition, total_hyperparameters::Vector{T}; show::Integer=3, file::AbstractString="", find_post::Bool=true, plot_Σ::Bool=false, plot_Σ_profile::Bool=false, filetype::AbstractString="png", y_obs::Vector{T}=prob_def.y_obs) where {T<:Real}

    x_samp = collect(linspace(minimum(prob_def.x_obs), maximum(prob_def.x_obs), amount_of_samp_points))
    amount_of_total_samp_points = amount_of_samp_points * prob_def.n_out
    amount_of_obs = length(prob_def.x_obs)

    show_curves = zeros(show, amount_of_total_samp_points)

    # calculate mean, σ, and show_curves
    if find_post
        mean, σ, Σ = GP_posteriors(prob_def, x_samp, total_hyperparameters; y_obs=y_obs)
        if plot_Σ; plot_im(Σ, file = file * "_K_post." * filetype) end
        L = ridge_chol(Σ).L
        for i in 1:show
            show_curves[i,:] = L * randn(amount_of_total_samp_points) + mean
        end
    # if no posterior is being calculated, estimate σ with sampling
    else
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

        # geting the y values for the proper output
        y_o = y_obs[obs_output_indices]
        show_curves_o = show_curves[:, sample_output_indices]
        σ_o = σ[sample_output_indices]
        mean_o = mean[sample_output_indices]
        obs_noise_o = prob_def.noise[obs_output_indices]

        custom_GP_plot(x_samp, show_curves_o, prob_def.x_obs, y_o, σ_o, mean_o; errors=obs_noise_o)

        xlabel("Time (" * string(prob_def.x_obs_units) * ")")
        if output==1
            y_str = "RVs (" * string(prob_def.y_obs_units) * ")"
            title_string = "Apparent RVs"
        else
            y_str = "Scores"
            title_string = "DCPCA Component " * string(output-1)
        end
        ylabel(y_str)
        title(title_string, fontsize=45)

        if find_post
            # put log likelihood on plot
            LogL = -nlogL_Jones(prob_def, total_hyperparameters; y_obs=y_obs)
            text(minimum(prob_def.x_obs), 0.9 * maximum([maximum(y_o), maximum(show_curves_o)]), L"l_{act}(\theta|t,s): " * string(round(LogL)), fontsize=30)
        end

        # put kernel lengths on plot
        kernel_lengths = total_hyperparameters[end-prob_def.n_kern_hyper+1:end]
        text(minimum(prob_def.x_obs), 1 * minimum([minimum(y_o), minimum(show_curves_o)]), "Hyperparameters: " * string(kernel_lengths), fontsize=30)
        # text(minimum(prob_def.x_obs), 1 *minimum(y_o), "Wavelengths: " * string(kernel_lengths), fontsize=30)

        if file!=""; save_PyPlot_fig(file * "_$output." * filetype) end

    end

    PyPlot.close("all")

end
