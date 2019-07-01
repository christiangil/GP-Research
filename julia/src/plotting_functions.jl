# all of the functions to simplify using PyPlot
# https://github.com/JuliaPy/PyPlot.jl

using PyPlot


# make it so that plot windowsa don't appear
matplotlib.use("Agg")


"set axes and tick label font sizes for PyPlot plots"
function set_font_sizes(ax; axes::Real=30., ticks::Real=24.)

    # this doesn't work. set title size at plot creating
    # setp(ax[:title], fontsize=title)

    setp(ax.xaxis.label, fontsize=axes)
    setp(ax.yaxis.label, fontsize=axes)

    setp(ax.get_xticklabels(), fontsize=ticks)
    setp(ax.get_yticklabels(), fontsize=ticks)

    # old reference stuff
    # font = Dict("fontsize"=>20)
    # xlabel("Number of Principal Components", fontdict=font)
    # for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels())
    #     item.set_fontsize(20)
    # end

end


"create a nice, large PyPlot with large text sizes (excluding title)"
function init_plot(;figsize=(16,9), hspace::Real=-1)
    fig = figure(figsize=figsize)
    ax = subplot(111)
    set_font_sizes(ax)
    if hspace != -1; fig.subplots_adjust(hspace=hspace) end
    return ax
end


"create a nice, large set of PyPlots with large text sizes (excluding title)"
function init_subplots(;nrows::Integer=1, ncols::Integer=1, sharex::Bool=false, sharey::Bool=false, figsize=(16,16), hspace::Real=-1)
    fig, axs = subplots(nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey, figsize=figsize)
    if hspace != -1; fig.subplots_adjust(hspace=hspace) end
    for ax in axs; set_font_sizes(ax) end
    return axs
end


"save current PyPlot figure and close all figures"
function save_PyPlot_fig(filename::AbstractString)
    savefig(filename)
    PyPlot.close("all")
end


function plot_im(A; file::AbstractString="", figsize=(9,9))
    init_plot(figsize=figsize)
    fig = imshow(A[:,:], origin="lower")
    colorbar()
    # title("Heatmap")
    if file != ""; save_PyPlot_fig(file) end
end


"""
Create a corner plot by evaluating a function around its current input values
should scale as steps^2*n(n-1)/2 in time
"""
function corner_plot(
    f::Function,
    input::Vector{<:Real},
    filename::AbstractString;
    steps::Integer=15+1,
    min_spread::Real=1/2,
    input_labels::Vector{<:AbstractString}=repeat([L" "],length(input)),
    n::Integer=length(input))

    assert_positive(min_spread, steps, n)
    @assert n <= length(input)
    @assert length(input_labels) == n

    figsize = 16/3 * n
    axs = init_subplots(nrows=n, ncols=n, figsize=(figsize,figsize))

    for k in 1:n
        for l in 1:n

            holder = copy(input)

            xspread = max(min_spread, abs(input[k]) / 10)
            yspread = max(min_spread, abs(input[l]) / 10)

            # create function profiles on diagonals
            if k == l
                x = linspace(input[k] - xspread, input[k] + xspread, steps)
                y = zero(x)
                for i in 1:length(x)
                    holder[k] = x[i]
                    y[i] = f(holder)
                end
                axs[k, k].set_title(input_labels[k], fontsize=10*n)
                axs[k, k].plot(x, y, linewidth=16/n)
                axs[k, k].axvline(x=input[k], color="black", linewidth=16/n)
                if abs(input[k]) < xspread; axs[k, k].axvline(x=0, color="grey", linewidth=16/n) end

            # create function heatmaps elsewhere
            elseif k < l
                xmin, xmax =  (input[k] - xspread, input[k] + xspread)
                ymin, ymax =  (input[l] - yspread, input[l] + yspread)
                x = linspace(xmin, xmax, steps)
                y = linspace(ymin, ymax, steps)
                Z = zeros((steps,steps))
                for i in 1:steps
                    for j in 1:steps
                        holder[k] = x[i]
                        holder[l] = y[j]
                        Z[i,j] = f(holder)
                    end
                end
                X = repeat(x', length(y), 1)
                Y = repeat(y, 1, length(x))
                axs[l, k].contour(X, Y, Z, colors="k", linewidths=16/n)
                axs[l, k].imshow(Z, interpolation="bilinear", origin="lower", extent=(xmin, xmax, ymin, ymax))
                axs[l, k].scatter(input[k], input[l], marker="X", c="k", s=1200/n)
                if abs(input[k]) < xspread; axs[l, k].axvline(x=0, color="grey", linewidth=32/n) end
                if abs(input[l]) < yspread; axs[l, k].axhline(y=0, color="grey", linewidth=32/n) end

                # setting image aspect ratio to make it square
                # xleft, xright = axs[l, k].get_xlim()
                # ybottom, ytop = axs[l, k].get_ylim()
                axs[l, k].set_aspect(xspread / yspread)

            # remove plots above the diagonal
            else
                axs[l, k].axis("off")
            end
        end
    end

    # remove plot labels for everything that isn't on the outside or diagonal
    for i in 1:n
        for j in 1:n
            if i!=j; axs[j,i].label_outer(); end
        end
    end

    save_PyPlot_fig(filename)

end
