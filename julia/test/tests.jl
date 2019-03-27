using JLD2, FileIO
using Test

@testset "cholesky factorizations" begin
    A = [4. 12 -16; 12 37 -43; -16 -43 98]
    chol_A = ridge_chol(A)
    @test chol_A==cholesky(A)==symmetric_A(A, chol=true)
    @test chol_A.L==LowerTriangular([2. 0 0;6 1 0;-8 5 3])
    A = [4. 12 -16; 12 37 -43; -16 -43 88]
    @test_logs (:warn, "added a ridge")  ridge_chol(A)
    println()
end

@testset "hyperparameter gradients" begin
    old_dir = pwd()
    cd(@__DIR__)
    include_kernel("quasi_periodic_kernel")
    @load "../jld2_files/problem_def_sample.jld2" problem_def_sample
    cd(old_dir)

    @test est_dKdθ(problem_def_sample, 1 .+ rand(3); return_bool=true, print_stuff=false)
    @test test_grad(problem_def_sample, 1 .+ rand(3), print_stuff=false)
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
    P=2.
    ecc=0.01
    @test isapprox(ϕ(test_time, P, ecc), ϕ_approx(test_time, P, ecc); rtol=1e-2)
    @test ϕ(test_time, P, ecc)!=mean_anomaly(test_time, P)
    @test ϕ_approx(test_time, P, ecc)!=mean_anomaly(test_time, P)
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
    noise_vect = noise_mag .* randn(amount_of_samp_points)
    fake_data += noise_vect
    measurement_noise = noise_mag .* ones(amount_of_samp_points)

    A = hcat(cos.(x_samp), sin.(x_samp), ones(length(x_samp)))

    est_coeffs = general_lst_sq(A, fake_data; Σ=measurement_noise)

    @test isapprox(est_coeffs, true_coeffs, rtol=1e-2)
    @test isapprox(std(noise_vect), std(remove_kepler(fake_data, x_samp, 2*pi, measurement_noise)); rtol=5e-1)
    println()
end

old_dir = pwd()
cd(@__DIR__)
include("parallel_rv_test.jl")
cd(old_dir)
