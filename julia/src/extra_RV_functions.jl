# these functions are related to calculating RV quantities
using UnitfulAstro
using Unitful
# using UnitfulAngles
using LinearAlgebra
using PyPlot
using DataFrames, CSV

kep_parms_str(ks::Union{kep_signal, kep_signal_epicyclic, kep_signal_wright}) =
    "K: $(round(convert_and_strip_units(u"m/s", ks.K), digits=2))" * L"^m/_s" * "  P: $(round(convert_and_strip_units(u"d", ks.P), digits=2))" * L"d" * "  M0: $(round(ks.M0,digits=2))  e: $(round(ks.e,digits=2))  ω: $(round(ks.ω,digits=2)) γ: $(round(convert_and_strip_units(u"m/s", ks.γ), digits=2))" * L"^m/_s"
kep_parms_str_short(ks::Union{kep_signal, kep_signal_epicyclic, kep_signal_wright}) =
    "K: $(round(convert_and_strip_units(u"m/s", ks.K), digits=2))" * L"^m/_s" * "  P: $(round(convert_and_strip_units(u"d", ks.P), digits=2))" * L"d" * "  e: $(round(ks.e,digits=2))"

function save_nlogLs(
    seed::Integer,
    times::Vector{T},
    likelihoods::Vector{T},
    hyperparameters::Vector{T},
    og_ks::Union{kep_signal, kep_signal_epicyclic, kep_signal_wright},
    fit_ks::Union{kep_signal, kep_signal_epicyclic, kep_signal_wright},
    save_loc::String
    ) where {T<:Real}


    likelihood_strs = ["L", "uE", "E"]
    num_likelihoods= length(likelihood_strs)
    @assert num_likelihoods == length(times)
    @assert length(likelihoods) == 3 * num_likelihoods
    orbit_params_strs = ["K", "P", "M0", "e", "ω", "γ"]
    orbit_params= [og_ks.K, og_ks.P, og_ks.M0, og_ks.e, og_ks.ω, og_ks.γ, fit_ks.K, fit_ks.P, fit_ks.M0, fit_ks.e, fit_ks.ω, fit_ks.γ]
    num_hyperparameters = Int(length(hyperparameters) / 2)
    # file_name = "csv_files/$(kernel_name)_logLs.csv"
    file_name = save_loc * "logL.csv"

    df = DataFrame(seed=seed, date=today())

    for i in 1:length(times)
        df[!, Symbol("t$(Int(i))")] .= times[i]
    end
    for i in 1:length(likelihoods)
        df[!, Symbol(string(likelihood_strs[(i-1)%num_likelihoods + 1]) * string(Int(1 + floor((i-1)//num_likelihoods))))] .= likelihoods[i]
    end
    # df[!, Symbol("E_wp")] .= likelihoods[end]
    for i in 1:length(hyperparameters)
        df[!, Symbol("H" * string(((i-1)%num_hyperparameters) + 1) * "_" * string(Int(1 + floor((i-1)//num_hyperparameters))))] .= hyperparameters[i]
    end
    for i in 1:length(orbit_params)
        df[!, Symbol(string(orbit_params_strs[(i-1)%n_kep_parms + 1]) * string(Int(1 + floor((i-1)//n_kep_parms))))] .= orbit_params[i]
    end

    # if isfile(file_name); append!(df, CSV.read(file_name)) end

    CSV.write(file_name, df)

end
