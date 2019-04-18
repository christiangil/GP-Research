## 528 Final Project (Soon to be KepLikelihood.jl?)

This software allows the user to take a (possibly multivariate) time series of data and evalute a custom log likelihood function after the removal of a "best-fit" generalized sinusoidal signal (interpretted as a circular Keplerian orbit in this case).

- Choice of portion of code to parallelize (1 point)

I chose to optimize this likelihood evaluation because, if a non-constant covariance matrix is assumed, kep_signal_likelihoods() dominates the runtime of optimize_periods.jl after checking only 64 periods. In my research, we will likely be checking many more.

- Choice of approach for parallelizing code (1 point)

I chose to parallelize initially with shared memory arrays with distributed for loops. I also parallelized with distributed memory using both parallel mapping and mapping the function onto distributed arrays. I looked at all of these methods because it was not clear to me which would be the best, and our computation is easily generalized onto these methods

- Code performs proposed tasks (2 point)

The code is able to remove generalized sinusoidal signals and evaluate the likelihood of the remainging signal. See julia/figs/\*periods.png

- Unit/regression tests comparing serial & parallel versions (1 point)

These are performed using julia/test/parallel_rv_test.jl, which is called when running julia/test/runtests.jl

- Code passes tests (1 point)

From within the julia folder, you can import all of the functions and run all of the tests using

```julia
include("src/setup.jl")
include("test/runtests.jl")
```

- General code efficiency (1 point)

See functions in julia/src

- Implementation/optimization of second type of parallelism (2 points)

See lines 65+ in julia/optimize_periods.jl

- Significant performance improvement (1 point)

Yep! see optimize_period_scaling_l.png and optimize_period_scaling_t.png in the julia/figs folder


The main functions are kep_signal_likelihood() and its vectorized version kep_signal_likelihoods().

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
include("test/runtests.jl")
```
