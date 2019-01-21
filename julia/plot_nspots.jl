include("all_functions.jl")

using JLD2, FileIO
# using RvSpectraKitLearn
using MultivariateStats
using HDF5

# # path_to_spectra = "/gpfs/group/ebf11/default/SOAP_output/May_runs"
# # test = "/Users/cjg66/Downloads/lambda-3923-6664-3years_174spots_diffrot_id9.h5"
# test = "D:/Christian/Downloads/lambda-3923-6664-3years_174spots_diffrot_id9.h5"
# test = "C:/Users/Christian/Dropbox/GP_research/star_spots/SOAP2_RawSpectra/Example/test_runs/lambda-3923-6664-0years_18spots_diffrot_id1.h5"
test = "D:/Christian/Downloads/lambda-3923-6664-1years_1582spots_diffrot_id1.h5"
fid = h5open(test, "r")
# objects = names(fid)
# println(objects)
thing1 = fid["n_spots"][:]
thing2 = fid["msh_covered"][:]

init_plot()
plot(thing1)
xlabel("Phases")
ylabel("Number of spots")
title("Spots/time", fontsize=30)
savefig("n_spots.pdf")

init_plot()
plot(thing2)
xlabel("Phases")
ylabel("MSH covered")
title("MSH/time", fontsize=30)
savefig("msh.pdf")
