# errors
include("src/setup.jl")
include("src/all_functions.jl")

@load "jld2_files/res-1000-1years_full_id41_rv_data.jld2"

using Dierckx

# M
# inds = 9000:9080
inds = 139460:139600
x = λs[inds]
y = mu[inds]
new_x = linspace(x[1], x[end], 10*length(inds))
new_y = Spline1D(x, y)(new_x)

for i in 1:6
    colors = Spline1D(x, M[inds, i])(new_x)
    colors .-= minimum(colors)
    colors ./= maximum(colors)

    ax = init_plot(figsize=(16,16))
    # scatter(new_x, new_y; c=colors, cmap="viridis")
    scatter(new_x, new_y; c=colors, cmap="bwr")

    title("PC$(i-1)", fontsize=60)

    # for spine in plt.gca().spines.values()
    #     spine.set_visible(False)
    # end
    # fig.patch.set_visible(False)
    ax.axis("off")
    tick_params(top="off", bottom="off", left="off", right="off", labelleft="off", labelbottom="off")

    save_PyPlot_fig("test$i")
end

for i in 1:6
    colors = M[inds, i]
    colors .-= minimum(colors)
    colors ./= maximum(colors)

    ax = init_plot()
    plot(x, colors)
    save_PyPlot_fig("test$i")
end
