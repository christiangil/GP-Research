using JLD2, FileIO
using Test

# import activate Project.toml in top folder and import functions
old_dir = pwd()
cd(@__DIR__)
cd("..")
include("../src/all_functions.jl")
cd(old_dir)

@testset "cholesky factorizations" begin
    A = [4. 12 -16; 12 37 -43; -16 -43 98]
    chol_A = ridge_chol(A)
    @test chol_A==cholesky(A)==symmetric_A(A, chol=true)
    @test chol_A.L==LowerTriangular([2. 0 0;6 1 0;-8 5 3])
    A = [4. 12 -16; 12 37 -43; -16 -43 88]
    @test_logs (:warn, "added a ridge")  ridge_chol(A)
    println()
end

@testset "nlogL derivatives" begin
    old_dir = pwd()
    cd(@__DIR__)
    include_kernel("quasi_periodic_kernel")
    @load "../jld2_files/problem_def_sample.jld2" problem_def
    cd(old_dir)

    @test est_dΣdθ(problem_def, 1 .+ rand(3); return_bool=true, print_stuff=false)
    @test test_grad(problem_def, 1 .+ rand(3), print_stuff=false)
    @test test_hess(problem_def, 1 .+ rand(3), print_stuff=false)
    println()
end

@testset "velocity semi-amplitudes" begin
    # testing Jupiter and Earth's radial velocity amplitudes
    # https://en.wikipedia.org/wiki/Doppler_spectroscopy
    m_star = (1)u"Msun"
    P = (11.86)u"yr"
    m_planet = (1)u"Mjup"
    @test isapprox(velocity_semi_amplitude(P, m_star, m_planet), 12.4; rtol=1e-2)
    P = (1)u"yr"
    m_planet = (1)u"Mearth"
    @test isapprox(velocity_semi_amplitude(P, m_star, m_planet), 0.09; rtol=1e-2)
    println()
end

@testset "true anomaly" begin
    # see if the low eccentricity approximation is working
    test_time = rand()
    P=2.
    ecc=0.01
    M0=1
    # @test isapprox(ϕ(test_time, P, ecc, M0), ϕ_approx(test_time, P, ecc); rtol=1e-2)
    @test ϕ(test_time, P, ecc, M0) != mean_anomaly(test_time, P, M0)
    # @test ϕ_approx(test_time, P, ecc, M0) != mean_anomaly(test_time, P, M0)
    println()
end

@testset "radial velocities" begin
    # making sure RVs are being calculated sensibly
    m_star = (1)u"Msun"
    P = (1)u"yr"
    m_planet = (1)u"Mearth"
    e = 0.
    ω = 0
    M0 = 0

    @test isapprox(kepler_rv(0., P, e, M0, m_star, m_planet, ω), -kepler_rv(1/2 * P, P, e, M0, m_star, m_planet, ω))
    @test isapprox(kepler_rv(1/4 * P, P, e, M0, m_star, m_planet, ω), 0; atol=1e-8)
    @test isapprox(kepler_rv(0., P, e, M0, m_star, m_planet, ω; i=0.), 0)
    @test isapprox(kepler_rv(0., P, e, M0, m_star, m_planet, ω; i=pi/4), 1 / sqrt(2) * kepler_rv(0., P, e, M0, m_star, m_planet, ω))

    # making sure our two RV equations produce the same results
    K = 1.
    times = linspace(0,1,10)
    h1 = 0.0
    k1 = 0.99
    P = 1.
    M0 = pi / 2
    γ = 0.1
    e = sqrt(h1 * h1 + k1 * k1)
    ω = atan(h1, k1)
    h2 = sqrt(e) * sin(ω)
    k2 = sqrt(e) * cos(ω)

    @test isapprox(kepler_rv.(times, P, e, M0, K, ω; γ=γ), kepler_rv_hk1.(times, P, M0, K, h1, k1; γ=γ))
    @test isapprox(kepler_rv.(times, P, e, M0, K, ω; γ=γ), kepler_rv_hk2.(times, P, M0, K, h2, k2; γ=γ))
    @test isapprox(kepler_rv.(times, P, e, M0, K, ω; γ=γ), kepler_rv_true_anomaly.(times, P, e, M0, K, ω; γ=γ))

    println()
end

@testset "estimating keplerian params" begin
    # see if my paper math for solving for the linear equations is working
    amount_of_samp_points = 50
    true_coeffs = rand(3)
    x_samp = 5 .* pi .* sort(rand(amount_of_samp_points))
    fake_data = kepler_rv_circ(x_samp, 2 * π, true_coeffs)
    # fake_data = true_coeffs[1] .* cos.(x_samp) .+ true_coeffs[2] .* sin.(x_samp) .+ true_coeffs[3]
    noise_mag =  0.01 * maximum(fake_data)
    noise_vect = noise_mag .* randn(amount_of_samp_points)
    fake_data += noise_vect
    measurement_noise = noise_mag .* ones(amount_of_samp_points)

    A = hcat(cos.(x_samp), sin.(x_samp), ones(length(x_samp)))

    est_coeffs = general_lst_sq(A, fake_data; Σ=measurement_noise)

    @test isapprox(est_coeffs, true_coeffs, rtol=1e-2)
    @test isapprox(std(noise_vect), std(remove_kepler(fake_data, x_samp, 2*pi, measurement_noise)); rtol=3e-1)
    println()
end

# old_dir = pwd()
# cd(@__DIR__)
# include("parallel_rv_test.jl")
# cd(old_dir)

@testset "A \\ b == x working as intended?" begin

    # defining a simple colvariance function
    rbf(kernel_length, dif) = exp((-1/2)*dif^2/kernel_length^2)

    # getting some basis values
    x_samp = (0:200)/40
    x_length = length(x_samp)

    # calculating the covariance matrix
    Σ_samp = zeros((x_length, x_length))
    for i in 1:x_length
        for j in 1:x_length
            Σ_samp[i, j] = rbf(1, x_samp[i] - x_samp[j])
        end
    end
    Σ_samp += Diagonal(1e-5 * ones(x_length))

    y_samp = randn(length(x_samp))

    # make sure inv is working as intended
    @test isapprox(Σ_samp * inv(Σ_samp), Matrix(I, x_length, x_length); rtol=1e-5)
    @test isapprox(inv(Σ_samp) * Σ_samp, Matrix(I, x_length, x_length))

    sol_inv = inv(Σ_samp) * y_samp
    sol_chol = cholesky(Σ_samp) \ y_samp
    sol_lmd = Σ_samp \ y_samp
    sol_cg = IterativeSolvers.cg(Σ_samp, y_samp)

    # all methods should give the same results
    @test isapprox(sol_inv, sol_chol)
    @test isapprox(sol_inv, sol_lmd)
    @test isapprox(sol_inv, sol_cg)

    # y_samp should be recoverable
    @test isapprox(Σ_samp * sol_inv, y_samp; rtol=1e-4)
    @test isapprox(Σ_samp * sol_chol, y_samp)
    @test isapprox(Σ_samp * sol_lmd, y_samp)
    @test isapprox(Σ_samp * sol_cg, y_samp)
    println()
end
