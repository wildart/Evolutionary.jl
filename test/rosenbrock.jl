@testset "Rosenbrock" begin

    function test_result(result::Evolutionary.EvolutionaryOptimizationResults, N::Int, tol::Float64)
        fitness = minimum(result)
        extremum = Evolutionary.minimizer(result)
        @test sum(abs2,extremum.-ones(N)) ≈ 0.0 atol=tol
        @test fitness ≈ 0.0 atol=tol
    end

    # Objective function
    rosenbrock(x::AbstractVector) = (1.0 - x[1])^2 + 100 * (x[2] - x[1]^2)^2

    # Parameters
    N = 2

    # Testing: (15,3+100)-ES
    # with isotropic mutation operator y' := y + σ(N_1(0, 1), ..., N_N(0, 1))
    result = es(rosenbrock, N; # old api
        initPopulation = rand(N,25),
        initStrategy = strategy(σ = 1.0), srecombination = averageSigma1,
        recombination = average, mutation = isotropic,
        μ = 15, ρ = 3, λ = 100, iterations = 1000, tolIter=30)
    println("(15/3+100)-ES => F: $(minimum(result)), C: $(Evolutionary.iterations(result))")
    test_result(result, N, 1e-1)

    settings = [
        :isotropic=>(isotropic, isotropicSigma, strategy(σ = 1.0, τ = 1/sqrt(2*N)))
        :anisotropic=>(anisotropic, anisotropicSigma, strategy(σ = .5ones(N), τ = 1/sqrt(2*N), τ0 = 1/sqrt(N)))
    ]
    selections = [:plus, :comma]
    @testset "ES settings" for (sn,ss) in settings, sel in selections
        result = Evolutionary.optimize( rosenbrock, (() -> rand(N)),
            ES(
                initStrategy = ss[3],
                recombination = average, srecombination = averageSigma,
                mutation = ss[1], smutation = ss[2],
                μ = 10, ρ = 3, λ = 100, selection=sel
            ), Evolutionary.Options(iterations=1000, successive_f_tol=25)
        )
        println("(15/3$(sel == :plus ? "+" : ",")100)-ES:$(sn) => F: $(minimum(result)), C: $(Evolutionary.iterations(result))")
        test_result(result, N, sel == :plus ? 0.1 : 0.5)
    end

    # Testing: CMA-ES
    result = Evolutionary.optimize(rosenbrock, (() -> rand(Float32,N)), CMAES(μ = 5, λ = 100))
    println("(5/5,100)-CMA-ES => F: $(minimum(result)), C: $(Evolutionary.iterations(result))")
    test_result(result, N, 1e-2)

    # Testing: GA
    result = Evolutionary.optimize(rosenbrock, (() -> rand(N)),
        GA(
            populationSize = 100,
            ɛ = 0.1,
            selection = rouletteinv,
            crossover = intermediate(0.25),
            mutation = domainrange(fill(0.5,N))
        ))
    println("GA(p=100,x=0.8,μ=0.1,ɛ=0.1) => F: $(minimum(result)), C: $(Evolutionary.iterations(result))")
    test_result(result, N, 1e-1)

end
