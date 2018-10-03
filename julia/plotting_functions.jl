# all of the functions to simplify using PyPlot
# https://github.com/JuliaPy/PyPlot.jl

using PyPlot


function set_font_sizes(ax; title=30, axes=20, ticks=15)

    setp(ax[:title], fontsize=title)

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


function init_plot()
    figure(figsize=(10,6))
    ax = subplot(111)
    set_font_sizes(ax)
end
