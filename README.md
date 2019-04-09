## 528 Final Project (Soon to be KepLikelihood.jl?)

This software allows the user to take a (possibly multivariate) time series of data and evalute a custom log likelihood function after the removal of a "best-fit" generalized sinusoidal signal (interpretted as a circular Keplerian orbit in this case).

The main functions are kep_signal_likelihood() and its vectorized verion kep_signal_likelihoods().

```julia
"""
Evaluate the likelihood function with data after removing the best-fit circular
Keplerian orbit for given period.

    kep_signal_likelihood(likelihood_func::Function, period::Real, times_obs::AbstractArray{T2,1}, signal_data::AbstractArray{T3,1}, covariance::Union{Cholesky{T4,Array{T4,2}},Symmetric{T5,Array{T5,2}},AbstractArray{T6}})
    
likelihood_func is a wrapper function handle that returns the likelihood given a single input of the data without the best-fit Kperleian signal
period is the orbital period that you want to attempt to remove
times_obs are the times of the measurements
signal_data is your data including the planetary signal
covariance is either the covariance matrix relating all of your data points, or a vector of noise measuremnets
"""
function kep_signal_likelihood(likelihood_func::Function, period::Real, times_obs::AbstractArray{T2,1}, signal_data::AbstractArray{T3,1}, covariance::Union{Cholesky{T4,Array{T4,2}},Symmetric{T5,Array{T5,2}},AbstractArray{T6}})
    return likelihood_func(remove_kepler(signal_data, times_obs, period, covariance))
end
```

These functions can be called like so

```julia
likelihood_func(new_data::AbstractArray{T,1}) where{T<:Real} = some_likelihood_function(x...)  # at least one of the inputs should depend on new_data 
likelihoods = kep_signal_likelihoods(likelihood_func, period_grid, times_obs, fake_data, covariance)
```

An example script called optimize_periods.jl is given in the julia directory.

While this isn't an official package yet, the directory setup (once inside the julia folder) is the same.

From within the julia folder, you can import all of the functions and run all of the tests using

```julia
include("src/setup.jl")
include("src/all_functions.jl")
include("test/runtests.jl")
```
