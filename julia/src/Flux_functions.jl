# these functions are Flux wrappers
using Flux

"an empty function, so that a function that requires another function to be passed can use this as a default"
do_nothing() = nothing


"a wrapper for Flux.train! that allows me to easily train down to a gradient norm threshold"
function flux_train_to_target!(f::Function, g::Function, ps::Flux.Tracker.Params; grad_norm_thres::Real=1e2, max_iter::Integer=500, opt=ADAM(0.1), flux_data=Iterators.repeated((), 50), cb::Function=do_nothing, cb_delay::Real=10)
    @warn "grad_norm and iter_num global variables created/reassigned"
    global grad_norm = 10 * grad_norm_thres
    global iter_num = 0
    while ((grad_norm>grad_norm_thres) & (iter_num<max_iter))
        global iter_num += length(flux_data)
        Flux.train!(f, ps, flux_data, opt, cb=Flux.throttle(cb, cb_delay))
        global grad_norm = norm(g())
        println("Epoch $iter_num score: ", data(f()), " with gradient norm ", grad_norm)
    end
end
