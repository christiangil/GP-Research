# all of the functions to simplify using PyPlot
# https://github.com/JuliaPy/PyPlot.jl

using PyPlot


"set axes and tick label font sizes for PyPlot plots"
function set_font_sizes(ax; axes::Real=30., ticks::Real=24., title::Real=45.)

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
function init_plot(;figsize=(16,9))
    figure(figsize=figsize)
    ax = subplot(111)
    set_font_sizes(ax)
    return ax
end


function plot_im(A; file::String="")
    init_plot(figsize=(9,9))
    fig = imshow(A[:,:])
    colorbar()
    # title("Heatmap")
    if file != ""
        savefig(file)
    end
    PyPlot.close_figs()
end


"make the figure show up in Atom plot pane"
draw_fig_atom() = fig[:canvas][:draw]()
