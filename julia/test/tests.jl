using JLD2, FileIO

old_dir = pwd()
cd(@__DIR__)
pwd()

@testset "hyperparameter gradients" begin
    include_kernel("Quasi_periodic_kernel")
    @load "../jld2_files/sample_problem_def.jld2" sample_problem_def
    @test est_dKdθ(sample_problem_def, 1 .+ rand(3); return_bool=true)
    @test test_grad(sample_problem_def, 1 .+ rand(3))
    println()
end

@testset "velocity semi-amplitudes" begin
    # testing Jupiter and Earth's radial velocity amplitudes
    # https://en.wikipedia.org/wiki/Doppler_spectroscopy
    m_star = 1u"Msun"
    P = 11.86u"yr"
    m_planet = 1u"Mjup"
    @test isapprox(velocity_semi_amplitude(P, m_star, m_planet), 12.4; rtol=1e-2)
    P = 1u"yr"
    m_planet = 1u"Mearth"
    @test isapprox(velocity_semi_amplitude(P, m_star, m_planet), 0.09; rtol=1e-2)
    println()
end

@testset "true anomaly" begin
    # see if the low eccentricity approximation is working
    test_time = rand()
    @test isapprox(ϕ(test_time, 2.; e=0.01), ϕ(test_time, 2.; e=0.01, iter=false); rtol=1e-2)
    println()
end

@testset "radial velocities" begin
    # making sure RVs are being calculated sensibly
    m_star = 1u"Msun"
    P = 1u"yr"
    m_planet = 1u"Mearth"
    @test isapprox(kepler_rv(0., P, m_star, m_planet), -kepler_rv(1/2 * P, P, m_star, m_planet))
    @test isapprox(kepler_rv(1/4 * P, P, m_star, m_planet), 0; atol=1e-8)
    @test isapprox(kepler_rv(0., P, m_star, m_planet, i=0.), 0)
    @test isapprox(kepler_rv(0., P, m_star, m_planet, i=pi/4), 1 / sqrt(2) * kepler_rv(0., P, m_star, m_planet))
    println()
end

@testset "estimating keplerian params" begin
    # see if my paper math for solving for the linear equations is working
    amount_of_samp_points = 50
    true_coeffs = rand(3)
    x_samp = 5 .* pi .* sort(rand(amount_of_samp_points))
    fake_data = true_coeffs[1] .* cos.(x_samp) .+ true_coeffs[2] .* sin.(x_samp) .+ true_coeffs[3]
    noise_mag =  0.01 * maximum(fake_data)
    fake_data += noise_mag .* randn(amount_of_samp_points)
    measurement_noise = noise_mag .* ones(amount_of_samp_points)

    A = hcat(cos.(x_samp), sin.(x_samp), ones(length(x_samp)))

    est_coeffs = general_lst_sq(A, fake_data; covariance=measurement_noise)

    # println("Estimating keplerian params")
    # println("true parameters:      ", true_coeffs)
    # println("estimated parameters: ", est_coeffs)

    @test isapprox(est_coeffs, true_coeffs, rtol=1e-2)
    println()
end

cd(old_dir)
clear_variables()
