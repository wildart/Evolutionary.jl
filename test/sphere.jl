@testset "Sphere" begin

    function test_result(result::Vector, fitness::Float64, N::Int, tol::Float64)
        @test length(result) == N
        @test ≈(fitness, 0.0, atol=tol)
    end

    # Objective function
    sphere(x::T) where {T <: Vector} = sum(x.*x)

    # Parameters
    N = 30
    P = 25
    terminate(strategy) = strategy[:σ] < 1e-10
    initial = ones(N)

    # Testing: (μ/μ_I, λ)-σ-Self-Adaptation-ES
    # with isotropic mutation operator y' := y + σ(N_1(0, 1), ..., N_N(0, 1))
    result, fitness, cnt = es(sphere, N;
        initStrategy = strategy(σ = 1.0, τ = 1/sqrt(2*N)),
        recombination = average, srecombination = averageSigma1,
        mutation = isotropic, smutation = isotropicSigma,
        termination = terminate, selection=:comma,
        μ = 3, λ = P, iterations = 1000)
    println("(3/3,$(P))-σ-SA-ES => F: $(fitness), C: $(cnt)")
    test_result(result, fitness, N, 1e-5)

    # Testing: GA
    result, fitness, cnt =
        ga( sphere, N;
            populationSize = 4P,
            mutationRate = 0.05,
            ɛ = 0.1,
            selection = sus,
            crossover = intermediate(0.25),
            mutation = domainrange(fill(0.5,N)),
            tol = 1e-5, tolIter = 15)
    println("GA(pop=$(P),xover=0.8,μ=0.1) => F: $(fitness), C: $(cnt)")
    test_result(result, fitness, N, 1e-2)

end
