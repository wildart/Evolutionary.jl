module TestCMAES
    using Evolutionary
    using Base.Test

    terminate(σ) = σ < 1e-10

    # Rosenbrock function
    function rosenbrock{T <: Vector}(x::T)
        return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
    end
    N = 2
    initialValue = zeros(N)
    CMAstrategy = strategy(τ = sqrt(N), τ_c = N^2, τ_σ = sqrt(N))
    result, fitness, cnt = cmaes(rosenbrock, initialValue, CMAstrategy;
        termination = terminate, μ = 3, λ = 12, iterations = 100_000)
    println("(3/3,12)-CMA-ES (Rosenbrock) => F: $(fitness), C: $(cnt), OBJ: $(result)")

    @test_approx_eq_eps result [1.0, 1.0] 1e-2
    @test_approx_eq_eps fitness 0.0 1e-5

    # Schwefel's ellipsoid test function, a moderately conditioned ellipsoid with a dominating isolated eigenvalue
    function schwefel{T <: Vector}(x::T)
        out = 0
        for i = 1:length(x)
            out += sumabs2(x[1:i])
        end
        return out
    end
    N = 30
    initialValue = ones(N)
    CMAstrategy = strategy(τ = sqrt(N), τ_c = N^2, τ_σ = sqrt(N))
    result, fitness, cnt = cmaes(schwefel, initialValue, CMAstrategy;
        termination = terminate, μ = 3, λ = 12, iterations = 1000, verbose=false)
    println("(3/3,12)-CMA-ES (Schwefel) => F: $(fitness), C: $(cnt)")

    @test_approx_eq_eps result zeros(N) 1e-5
    @test_approx_eq_eps fitness 0.0 1e-5

end