# these functions are Flux wrappers
using Flux


"a wrapper for Flux.train! that allows me to easily train down to a gradient norm threshold"
function flux_train_to_target!(f::Function, g::Function, ps::Flux.Tracker.Params; grad_norm_thres::Real=1e2, max_iter::Integer=500, opt=ADAM(0.2), flux_data=Iterators.repeated((), 50), flux_cb::Function=do_nothing, flux_cb_delay::Real=10, outer_cb::Function=do_nothing)
    @warn "grad_norm and iter_num global variables created/reassigned"
    global grad_norm = 10 * grad_norm_thres
    global iter_num = 0
    while ((grad_norm>grad_norm_thres) & (iter_num<max_iter))
        global iter_num += length(flux_data)
        Flux.train!(f, ps, flux_data, opt, cb=Flux.throttle(flux_cb, flux_cb_delay))
        global grad_norm = norm(g())
        println("Epoch $iter_num score: ", data(f()), " with gradient norm ", grad_norm)
        outer_cb()
    end
end
