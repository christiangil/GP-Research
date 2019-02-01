# errors
include("src/all_functions.jl")


hdf5_loc = "D:/Christian/Downloads/lambda-3923-6664-1years_1582spots_diffrot_id1.h5"
fid = h5open(hdf5_loc, "r")
act = fid["active"]
@elapsed time_series_spectra = act[:, :]
for i in 1:12
    bootstrap_errors(; boot_amount=5, time_series_spectra=time_series_spectra)
end
