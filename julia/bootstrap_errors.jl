# errors
include("src/all_functions.jl")

hdf5_locs = ["D:/Christian/Downloads/lambda-3923-6664-1years_1574spots_diffrot_id6.h5",
    "D:/Christian/Downloads/lambda-3923-6664-0years_7spots_diffrot_id123.h5",
    "D:/Christian/Downloads/lambda-3923-6664-0years_1spots_diffrot_id1.h5"]
file_addns = ["", "_singles", "_const"]

file_choice = 1
hdf5_loc = hdf5_locs[file_choice]
file_str =  file_addns[file_choice]

fid = h5open(hdf5_loc, "r")
act = fid["active"]
@elapsed time_series_spectra = act[:, :]
boot_amount = 10
for i in 1:5
    bootstrap_errors(time_series_spectra; boot_amount=boot_amount, save_filename="jld2_files/bootstrap" * file_str * ".jld2")
    println("Completed $(i * boot_amount) bootstraps resamples so far")
end
