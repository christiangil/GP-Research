# all of the functions to simplify using PyPlot
# https://github.com/JuliaPy/PyPlot.jl

using PyPlot


"set axes and tick label font sizes for PyPlot plots"
function set_font_sizes(ax; axes::Union{Float64,Int}=20., ticks::Union{Float64,Int}=15., title::Union{Float64,Int}=30.)

    # this doesn't work. set title size at plot creating
    # setp(ax[:title], fontsize=title)

    setp(ax[:xaxis][:label], fontsize=axes)
    setp(ax[:yaxis][:label], fontsize=axes)

    setp(ax[:get_xticklabels](), fontsize=ticks)
    setp(ax[:get_yticklabels](), fontsize=ticks)

    # old reference stuff
    # font = Dict("fontsize"=>20)
    # xlabel("Number of Principal Components", fontdict=font)
    # for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels())
    #     item.set_fontsize(20)
    # end

end


"create a nice, large PyPlot with large text sizes (excluding title)"
function init_plot()
    figure(figsize=(10,6))
    ax = subplot(111)
    set_font_sizes(ax)
end


function plot_im(A; file::String="")
    init_plot()
    fig = imshow(A[:,:])
    colorbar()
    # title("Heatmap")
    if file != ""
        savefig(file)
    end
    PyPlot.close_figs()
end


# quick and dirty function for creating plots that show what I want
function custom_line_plot(x_samp::Array{Float64,1}, L::LowerTriangular{Float64,Array{Float64,2}}, prob_def::Jones_problem_definition; output::Int=1, draws::Int=5000, σ::Array{Float64,1}=zeros(1), mean::Array{Float64,1}=zeros(1), show::Int=5, file::String="", LogL::Float64=0., waves::Array{Float64,1}=zeros(1) )

    amount_of_samp_points = length(x_samp)
    amount_of_total_samp_points = amount_of_samp_points * prob_def.n_out
    amount_of_measurements = length(prob_def.x_obs)

    if mean==zeros(1)
        mean = zeros(amount_of_total_samp_points)
    end

    # same curves are drawn every time?
    # srand(100)

    # initialize figure size
    init_plot()

    # the indices corresponding to the proper output
    output_indices = (amount_of_samp_points * (output - 1) + 1):(amount_of_samp_points * output)

    # geting the y values for the proper output
    y = prob_def.y_obs[(amount_of_measurements * (output - 1) + 1):(amount_of_measurements * output)]

    # initializing storage for example GPs to be plotted
    show_curves = zeros(show, amount_of_samp_points)

    # if no analytical variance is passed, estimate it with sampling
    if σ == zeros(1)

        # calculate a bunch of GP draws
        storage = zeros((draws, amount_of_total_samp_points))
        for i in 1:draws
            storage[i, :] = (L * randn(amount_of_total_samp_points)) + mean
        end

        # ignore the outputs that aren't meant to be plotted
        storage = storage[:, output_indices]

        #
        show_curves[:, :] = storage[1:show, :]
        storage = sort(storage, dims=1)

        # filling the 5-95th percentile with a transparent orange
        fill_between(x_samp, storage[convert(Int64, 0.95 * draws), :], storage[convert(Int64, 0.05 * draws), :], alpha=0.3, color="orange")

        # needs to be in both leaves of the if statement
        mean = mean[output_indices]

    else

        storage = zeros((show, amount_of_total_samp_points))
        for i in 1:show
            storage[i,:] = L * randn(amount_of_total_samp_points) + mean
        end
        show_curves[:, :] = storage[:, output_indices]

        σ = σ[output_indices]

        mean = mean[output_indices]


        # filling the 5-95th percentile with a transparent orange
        fill_between(x_samp, mean + 1.96 * σ, mean - 1.96 * σ, alpha=0.3, color="orange")

    end


    plot(x_samp, mean, color="black", zorder=2)
    for i in 1:show
        plot(x_samp, show_curves[i, :], alpha=0.5, zorder=1)
    end
    scatter(prob_def.x_obs, y, color="black", zorder=2)

    xlabel("phases (days?)")
    ylabel("pca scores")
    title("PCA " * string(output-1), fontsize=30)

    if LogL != 0.
        text(minimum(prob_def.x_obs), 0.9 * maximum([maximum(y), maximum(show_curves)]), "nLogL: " * string(round(LogL)), fontsize=20)
    end

    if waves != zeros(1)
        text(minimum(prob_def.x_obs), 1. * minimum([minimum(y), minimum(show_curves)]), "Wavelengths: " * string(waves), fontsize=20)
    end

    if file!=""
        savefig(file)
    end

    PyPlot.close_figs()

end
