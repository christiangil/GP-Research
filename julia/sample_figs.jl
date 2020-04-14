include("src/setup.jl")
# include("test/runtests.jl")
include("src/all_functions.jl")

hdf5_filename = "jld2_files/res-1000-1years_long_id10"
@load hdf5_filename * "_rv_data.jld2" phases scores
@load hdf5_filename * "_bootstrap.jld2" error_ests

x_obs = ustrip.(convert_SOAP_phases.(u"d", phases))

init_plot()
# plot(ustrip.(convert_SOAP_phases.(u"d", phases)), scores[1, :] * light_speed_nu)
y_obs = scores[1, :] * light_speed_nu
errors = error_ests[1, :] * light_speed_nu
scatter(x_obs, y_obs, color="black", zorder=2) # could comment out?
errorbar(x_obs, y_obs, yerr=[errors';errors'], fmt="o", color="black", zorder=2)
ylabel("Apparent RV (m/s)")
xlabel("Time (days)")
title("RVs from activity", fontsize=45)
save_PyPlot_fig("long_id10_0.png")

init_plot()
# plot(ustrip.(convert_SOAP_phases.(u"d", phases)), scores[1, :] * light_speed_nu)
y_obs = scores[1, :] * light_speed_nu + 1/2 .* sin.(x_obs./12)
errors = error_ests[1, :] * light_speed_nu
scatter(x_obs, y_obs, color="black", zorder=2) # could comment out?
errorbar(x_obs, y_obs, yerr=[errors';errors'], fmt="o", color="black", zorder=2)
ylabel("Apparent RV (m/s)")
xlabel("Time (days)")
title("RVs from activity and mystery planet", fontsize=45)
save_PyPlot_fig("long_id10_0_wp.png")

init_plot()
# plot(ustrip.(convert_SOAP_phases.(u"d", phases)), scores[1, :] * light_speed_nu)
y_obs = scores[2, :]
errors = error_ests[2, :]
scatter(x_obs, y_obs, color="black", zorder=2) # could comment out?
errorbar(x_obs, y_obs, yerr=[errors';errors'], fmt="o", color="black", zorder=2)
ylabel("Score")
xlabel("Time (days)")
title("Doppler PCA component 1", fontsize=45)
save_PyPlot_fig("long_id10_1.png")

init_plot()
# plot(ustrip.(convert_SOAP_phases.(u"d", phases)), scores[1, :] * light_speed_nu)
y_obs = scores[3, :]
errors = error_ests[3, :]
scatter(x_obs, y_obs, color="black", zorder=2) # could comment out?
errorbar(x_obs, y_obs, yerr=[errors';errors'], fmt="o", color="black", zorder=2)
ylabel("Score")
xlabel("Time (days)")
title("Doppler PCA component 2", fontsize=45)
save_PyPlot_fig("long_id10_2.png")


hdf5_filename = "jld2_files/res-1000-1years_full_id1"
@load hdf5_filename * "_rv_data.jld2" phases scores
init_plot()
plot(ustrip.(convert_SOAP_phases.(u"d", phases)), scores[1, :] * light_speed_nu)
ylabel("Apparent RV (m/s)")
xlabel("Time (days)")
save_PyPlot_fig("short_id1.png")

###############################

filenames = ["long_id10", "full_id1"]

function abc(name)
    hdf5_filename = "D:/Christian/Downloads/res-1000-1years_" * name * ".h5"
    thing = h5open(hdf5_filename)
    a = thing["msh_covered"][:] ./ 1e6
    b = thing["msh_visible"][:] ./ 1e6
    c = thing["msh_vis_proj"][:] ./ 1e6
    times = ustrip.(convert_SOAP_phases.(u"d", thing["phases"][:]))
    return a, b, c, times
end


inds = 280:340

begin
    fig, axs = init_subplots(;nrows=2, ncols=2, sharey="row", sharex="col", figsize=(18,9), hspace=0.1, wspace=0.1, ticks=20)
    # ylabel("Proportion covered")
    # xlabel("Time (days)")

    a, b, c, times = abc(filenames[2])

    ax1 = axs[1,1]
    ax2 = axs[1,2]
    ax3 = axs[2,1]
    ax4 = axs[2,2]

    ax1.plot(times, a ./ 2, label="Total star")
    ax1.plot(times, b, label="Visible hemisphere")
    ax1.plot(times, c .* 4, label="Visible disk")

    ax2.plot(times[inds], a[inds] ./ 2, label="Total star")
    ax2.plot(times[inds], b[inds], label="Visible hemisphere")
    ax2.plot(times[inds], c[inds] .* 4, label="Visible disk")

    a, b, c, times = abc(filenames[1])

    ax3.plot(times, a ./ 2, label="Total star")
    ax3.plot(times, b, label="Visible hemisphere")
    ax3.plot(times, c .* 4, label="Visible disk")
    ax3.legend(;loc="upper left", fontsize=20)

    ax4.plot(times[inds], a[inds] ./ 2, label="Total star")
    ax4.plot(times[inds], b[inds], label="Visible hemisphere")
    ax4.plot(times[inds], c[inds] .* 4, label="Visible disk")

    fig.text(0.5, 0.02, "Time (days)", ha="center", fontsize=30)
    fig.text(0.04, 0.5, "Proportion covered", va="center", rotation="vertical", fontsize=30)

    save_PyPlot_fig("test.png")
end
