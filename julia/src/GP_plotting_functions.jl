# all of the functions to simplify using PyPlot
# https://github.com/JuliaPy/PyPlot.jl

using PyPlot


# quick and dirty function for creating plots that show what I want
function custom_GP_plot(x_samp::Array{T1,1}, show_curves::Array{T2,2}, x_obs::Array{T3,1}, y_obs::Array{T4,1}, σ::Array{T5,1}, mean::Array{T6,1}; errors::Array{T7,1}=zeros(length(x_obs))) where {T1<:Real, T2<:Real, T3<:Real, T4<:Real, T5<:Real, T6<:Real, T7<:Real}

    @assert size(show_curves, 2)==length(x_samp)==length(mean)==length(σ)
    @assert length(x_obs)==length(y_obs)==length(errors)

    # initialize figure size
    init_plot()

    # filling the 5-95th percentile with a transparent orange
    fill_between(x_samp, mean + 1.96 * σ, mean - 1.96 * σ, alpha=0.3, color="orange")

    plot(x_samp, mean, color="black", zorder=2)

    for i in 1:size(show_curves, 1)
        plot(x_samp, show_curves[i, :], alpha=0.5, zorder=1)
    end

    scatter(x_obs, y_obs, color="black", zorder=2)
    errorbar(x_obs, y_obs, yerr=[errors';errors'], fmt="o", color="black", zorder=2)

end


function Jones_line_plots(amount_of_samp_points::Integer, prob_def::Jones_problem_definition, total_hyperparameters::Array{T,1}; show::Integer=5, file::String="", find_post::Bool=true, plot_K::Bool=false, filetype::String="png") where {T<:Real}

    x_samp = collect(linspace(minimum(prob_def.x_obs), maximum(prob_def.x_obs), amount_of_samp_points))
    amount_of_total_samp_points = amount_of_samp_points * prob_def.n_out
    amount_of_measurements = length(prob_def.x_obs)

    show_curves = zeros(show, amount_of_total_samp_points)

    # calculate mean, σ, and show_curves
    if find_post
        mean, σ, K_post = GP_posteriors(prob_def, x_samp, total_hyperparameters)
        L = ridge_chol(K_post).L
        for i in 1:show
            show_curves[i,:] = L * randn(amount_of_total_samp_points) + mean
        end
        if plot_K
            plot_im(K_post, file = file * "_K_post." * filetype)
        end
    # if no posterior is being calculated, estimate σ with sampling
    else
        mean = zeros(amount_of_total_samp_points)
        K_samp = covariance(prob_def, x_samp, x_samp, total_hyperparameters)
        L = ridge_chol(K_samp).L

        # calculate a bunch of GP draws for a σ estimation
        draws = 5000
        storage = zeros((draws, amount_of_total_samp_points))
        for i in 1:draws
            storage[i, :] = (L * randn(amount_of_total_samp_points)) + mean
        end
        show_curves[:, :] = storage[1:show, :]
        storage = sort(storage, dims=1)
        σ = storage[Int(round(0.84135 * draws)), :] - storage[Int(round(0.15865 * draws)), :] ./ 2
        if plot_K
            plot_im(K_samp, file = file * "_K_prior." * filetype)
        end
    end

    for output in 1:prob_def.n_out

        # the indices corresponding to the proper output
        sample_output_indices = (amount_of_samp_points * (output - 1) + 1):(amount_of_samp_points * output)
        obs_output_indices = (amount_of_measurements * (output - 1) + 1):(amount_of_measurements * output)

        # geting the y values for the proper output
        y_o = prob_def.y_obs[obs_output_indices]
        show_curves_o = show_curves[:, sample_output_indices]
        σ_o = σ[sample_output_indices]
        mean_o = mean[sample_output_indices]
        measurement_noise_o = prob_def.noise[obs_output_indices]

        custom_GP_plot(x_samp, show_curves_o, prob_def.x_obs, y_o, σ_o, mean_o; errors=measurement_noise_o)

        xlabel(prob_def.x_obs_units)
        ylabel(prob_def.y_obs_units)
        title("Output " * string(output-1), fontsize=45)

        if find_post
            # put log likelihood on plot
            LogL = nlogL_Jones(prob_def, total_hyperparameters)
            text(minimum(prob_def.x_obs), 0.9 * maximum([maximum(y_o), maximum(show_curves_o)]), "nLogL: " * string(round(LogL)), fontsize=30)
        end

        # put kernel lengths on plot
        kernel_lengths = total_hyperparameters[end-prob_def.n_kern_hyper+1:end]
        text(minimum(prob_def.x_obs), 1 * minimum([minimum(y_o), minimum(show_curves_o)]), "Lengthscales: " * string(kernel_lengths), fontsize=30)
        # text(minimum(prob_def.x_obs), 1 *minimum(y_o), "Wavelengths: " * string(kernel_lengths), fontsize=30)

        if file!=""
            savefig(file * "_" * string(output) * "." * filetype)
        end
    end

    PyPlot.close_figs()

end


# # quick and dirty function for creating plots that show what I want
# function custom_line_plot(x_samp::Array{T1,1}, L::LowerTriangular{T2,Array{T2,2}}, x_obs::Array{T3,1}, y_obs::Array{T4,1}; output::Integer=1, draws::Integer=5000, σ::Array{T5,1}=zeros(1), mean::Array{T6,1}=zeros(1), show::Integer=5, file::String="", LogL::Real=0., waves::Array{T7,1}=zeros(1)) where {T1<:Real, T2<:Real, T3<:Real, T4<:Real, T5<:Real, T6<:Real, T7<:Real}
#
#     amount_of_samp_points = length(x_samp)
#     amount_of_total_samp_points = length(x_samp) * convert(Int, length(y_obs)/length(x_obs))
#     amount_of_measurements = length(x_obs)
#
#     if mean==zeros(1)
#         mean = zeros(amount_of_total_samp_points)
#     end
#
#     # same curves are drawn every time?
#     # srand(100)
#
#     # initialize figure size
#     init_plot()
#
#     # the indices corresponding to the proper output
#     output_indices = (amount_of_samp_points * (output - 1) + 1):(amount_of_samp_points * output)
#
#     # geting the y values for the proper output
#     y = y_obs[(amount_of_measurements * (output - 1) + 1):(amount_of_measurements * output)]
#
#     # initializing storage for example GPs to be plotted
#     show_curves = zeros(show, amount_of_samp_points)
#
#     # if no analytical variance is passed, estimate it with sampling
#     if σ == zeros(1)
#
#         # calculate a bunch of GP draws
#         storage = zeros((draws, amount_of_total_samp_points))
#         for i in 1:draws
#             storage[i, :] = (L * randn(amount_of_total_samp_points)) + mean
#         end
#
#         # ignore the outputs that aren't meant to be plotted
#         storage = storage[:, output_indices]
#
#         #
#         show_curves[:, :] = storage[1:show, :]
#         storage = sort(storage, dims=1)
#
#         # filling the 5-95th percentile with a transparent orange
#         fill_between(x_samp, storage[convert(Int64, 0.975 * draws), :], storage[convert(Int64, 0.025 * draws), :], alpha=0.3, color="orange")
#
#         # needs to be in both leaves of the if statement
#         mean = mean[output_indices]
#
#     else
#
#         storage = zeros((show, amount_of_total_samp_points))
#         for i in 1:show
#             storage[i,:] = L * randn(amount_of_total_samp_points) + mean
#         end
#         show_curves[:, :] = storage[:, output_indices]
#
#         σ = σ[output_indices]
#
#         mean = mean[output_indices]
#
#
#         # filling the 5-95th percentile with a transparent orange
#         fill_between(x_samp, mean + 1.96 * σ, mean - 1.96 * σ, alpha=0.3, color="orange")
#
#     end
#
#
#     plot(x_samp, mean, color="black", zorder=2)
#     for i in 1:show
#         plot(x_samp, show_curves[i, :], alpha=0.5, zorder=1)
#     end
#     scatter(x_obs, y, color="black", zorder=2)
#
#     xlabel("phases (days?)")
#     ylabel("pca scores")
#     title("PCA " * string(output-1), fontsize=30)
#
#     if LogL != 0.
#         text(minimum(x_samp), 0.9 * maximum([maximum(y), maximum(show_curves)]), "nLogL: " * string(round(LogL)), fontsize=20)
#     end
#
#     if waves != zeros(1)
#         text(minimum(x_samp), 1. * minimum([minimum(y), minimum(show_curves)]), "Wavelengths: " * string(waves), fontsize=20)
#     end
#
#     if file!=""
#         savefig(file)
#     end
#
#     PyPlot.close_figs()
#
# end
