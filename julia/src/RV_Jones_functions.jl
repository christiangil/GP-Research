fit_kepler_wright_linear_step(
    prob_def::Jones_problem_definition,
    covariance::Union{Cholesky{T,Matrix{T}},Symmetric{T,Matrix{T}},Matrix{T},Vector{T}} where T<:Real,
    P::Unitful.Time,
    M0::Real,
    e::Real;
    data_unit::Unitful.Velocity=1u"m/s",
    return_extra::Bool=false) =
    fit_kepler_wright_linear_step(prob_def.y_obs, prob_def.time, covariance, P, M0, e; data_unit=prob_def.rv_unit*prob_def.normals[1], return_extra=return_extra)

function add_kepler_to_Jones_problem_definition!(
    prob_def::Jones_problem_definition,
    ks::kep_signal)

    if ustrip(ks.K) == 0
        return prob_def.y_obs
    end

    n_samp_points = length(prob_def.x_obs)
    planet_rvs = ks.(prob_def.time)
    # prob_def.y_obs[1:n_samp_points] += planet_rvs / (prob_def.normals[1] * prob_def.rv_unit)
    prob_def.y_obs[1:prob_def.n_out:end] += planet_rvs / (prob_def.normals[1] * prob_def.rv_unit)
    prob_def.rv[:] += planet_rvs
    # return prob_def
end

remove_kepler(
    prob_def::Jones_problem_definition,
    ks::Union{kep_signal, kep_signal_epicyclic, kep_signal_wright, kep_signal_circ};
    d::Vector{<:Integer}=zeros(Int64,n_kep_parms)) =
    remove_kepler(prob_def.y_obs, prob_def.time, ks; data_unit=prob_def.rv_unit*prob_def.normals[1], d=d)

add_kepler(
    prob_def::Jones_problem_definition,
    ks::Union{kep_signal, kep_signal_epicyclic, kep_signal_wright, kep_signal_circ}) =
    add_kepler(prob_def.y_obs, prob_def.time, ks; data_unit=prob_def.rv_unit*prob_def.normals[1])

fit_and_remove_kepler(
    prob_def::Jones_problem_definition,
    Σ_obs::Union{Cholesky{T,Matrix{T}},Symmetric{T,Matrix{T}},Matrix{T},Vector{T}},
    ks::Union{kep_signal, kep_signal_epicyclic}) where T<:Real =
    fit_and_remove_kepler(prob_def.y_obs, prob_def.time, Σ_obs, ks; data_unit=prob_def.rv_unit*prob_def.normals[1])

fit_kepler(
    prob_def::Jones_problem_definition,
    covariance::Union{Cholesky{T,Matrix{T}},Symmetric{T,Matrix{T}},Matrix{T},Vector{T}},
    ks::Union{kep_signal, kep_signal_wright};
    print_stuff::Bool=true) where T<:Real =
    fit_kepler(prob_def.y_obs, prob_def.time, covariance, ks; data_unit=prob_def.rv_unit*prob_def.normals[1], print_stuff=print_stuff)
fit_kepler(
    prob_def::Jones_problem_definition,
    covariance::Union{Cholesky{T,Matrix{T}},Symmetric{T,Matrix{T}},Matrix{T},Vector{T}},
    ks::kep_signal_wright;
    print_stuff::Bool=true,
    hold_P::Bool=false,
    avoid_saddle=true) where T<:Real =
    fit_kepler(prob_def.y_obs, prob_def.time, covariance, ks; data_unit=prob_def.rv_unit*prob_def.normals[1], print_stuff=print_stuff, hold_P=hold_P, avoid_saddle=avoid_saddle)
fit_kepler(
    prob_def::Jones_problem_definition,
    covariance::Union{Cholesky{T,Matrix{T}},Symmetric{T,Matrix{T}},Matrix{T},Vector{T}},
    ks::kep_signal_epicyclic) where T<:Real =
    fit_kepler(prob_def.y_obs, prob_def.time, covariance, ks; data_unit=prob_def.rv_unit*prob_def.normals[1])

function ∇∇nlogL_Jones_and_planet!(
    workspace::nlogL_matrix_workspace,
    prob_def::Jones_problem_definition,
    total_hyperparameters::Vector{<:Real},
    ks::kep_signal;
    include_kepler_priors::Bool=false)

    calculate_shared_∇nlogL_matrices!(workspace, prob_def, total_hyperparameters)

    non_zero_inds = copy(prob_def.non_zero_hyper_inds)
    n_hyper = length(non_zero_inds)
    full_H = zeros(n_kep_parms + n_hyper, n_kep_parms + n_hyper)
    full_H[1:n_hyper, 1:n_hyper] = ∇∇nlogL_Jones(
        prob_def, total_hyperparameters; Σ_obs=workspace.Σ_obs, y_obs=remove_kepler(prob_def, ks))

    full_H[n_hyper+1:end,n_hyper+1:end] = ∇∇nlogL_kep(prob_def.y_obs, prob_def.time, workspace.Σ_obs, ks; data_unit=prob_def.rv_unit*prob_def.normals[1], fix_jank=true, include_priors=include_kepler_priors)

    # TODO allow y and α to be passed to ∇∇nlogL_kep
    y = remove_kepler(prob_def, ks)
    α = workspace.Σ_obs \ y
    for (i, nzind1) in enumerate(non_zero_inds)
        for j in 1:n_kep_parms
            d = zeros(Int64, n_kep_parms)
            d[j] += 1
            y1 = remove_kepler(prob_def, ks; d=d)
            full_H[i, j + n_hyper] = d2nlogLdθ(y, y1, α, workspace.Σ_obs \ y1, workspace.βs[i])
        end
    end

    return Symmetric(full_H)
end
