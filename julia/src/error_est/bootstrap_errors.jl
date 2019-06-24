# errors
include("../all_functions.jl")

old_dir = pwd()
cd(@__DIR__)

length(ARGS)>0 ? hdf5_loc = parse(String, ARGS[1]) : hdf5_loc = "C:/Users/chris/Downloads/res-1000-lambda-3923-6664-1years_1579spots_diffrot_id11.h5"

hdf5_filename = string(split(hdf5_loc,"/")[end])[1:end-3]

fid = h5open(hdf5_loc, "r")
time_series_spectra = fid["active"][:, :]

λs = fid["lambdas"][:]
boot_amount = 10
for i in 1:5
    bootstrap_SOAP_errors(time_series_spectra, λs, "../../jld2_files/" * hdf5_filename; boot_amount=boot_amount)
    println("Completed $(i * boot_amount) bootstraps resamples so far")
end

cd(old_dir)
