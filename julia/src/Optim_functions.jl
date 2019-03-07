# these functions are Optim wrappers
using Optim


"allowing shared quanities to be shared"
function nlogL_Jones_fg!(prob_def::Jones_problem_definition, F, G, non_zero_hyperparameters::Array{T1,1}; y_obs::Array{T2,1}=zeros(1)) where {T1<:Real, T2<:Real}

    total_hyperparameters, L_fact, y_obs, α, prior_params = calculate_shared_nLogL_Jones(prob_def, non_zero_hyperparameters; y_obs=y_obs)
    # println(total_hyperparameters)
    if G != nothing
        ∇nlogL_Jones!(G, prob_def, total_hyperparameters, L_fact, y_obs, α, prior_params)
    end
    if F != nothing
        return nlogL_Jones(prob_def, total_hyperparameters, L_fact, y_obs, α, prior_params)
    end

end
