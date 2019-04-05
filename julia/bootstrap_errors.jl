# errors
include("src/all_functions.jl")


hdf5_loc = "D:/Christian/Downloads/lambda-3923-6664-1years_1586spots_diffrot_id21.h5"
fid = h5open(hdf5_loc, "r")
act = fid["active"]
@elapsed time_series_spectra = act[:, :]
boot_amount = 10
for i in 1:5
    bootstrap_errors(time_series_spectra; boot_amount=boot_amount)
    println("Completed $(i * boot_amount) bootstraps resamples so far")
end
