@testset "Rosenbrock" begin

    rng = StableRNG(42)

    function test_result(result::Evolutionary.EvolutionaryOptimizationResults, N::Int, tol::Float64)
        fitness = minimum(result)
        extremum = Evolutionary.minimizer(result)
        @test sum(abs2,extremum.-ones(N)) ≈ 0.0 atol=tol
        @test fitness ≈ 0.0 atol=tol
    end

    # Objective function
    rosenbrock(x::AbstractVector) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2

    # Parameters
    N = 2

    # Testing: (15,3/(+,)100)-ES
    settings = [
        :isotropic=>(gaussian, gaussian, IsotropicStrategy(N)),
        :isotropic=>(cauchy, gaussian, IsotropicStrategy(N)),
        :anisotropic=>(gaussian, gaussian, AnisotropicStrategy(N))
    ]
    selections = [:plus, :comma]
    opts = Evolutionary.Options(iterations=1000, successive_f_tol=25, rng=rng)
    @testset "ES settings" for (sn,ss) in settings, sel in selections
        Random.seed!(rng, 42)
        m = ES(
            initStrategy = ss[3],
            recombination = average, srecombination = average,
            mutation = ss[1], smutation = ss[2],
            μ = 15, ρ = 3, λ = 100, selection=sel
        )
        result = Evolutionary.optimize( rosenbrock, (() -> rand(rng, N)), m, opts)
        println("$(summary(m)):$(sn) => F: $(minimum(result)), C: $(Evolutionary.iterations(result))")
        test_result(result, N, sel == :plus ? 0.1 : 0.5)
    end
    Random.seed!(rng, 42)
    m = ES( initStrategy = IsotropicStrategy(N),
                recombination = average, srecombination = average,
                mutation = gaussian, smutation = gaussian,
                μ = 15, ρ = 3, λ = 100)
    result = Evolutionary.optimize(rosenbrock, BoxConstraints(0.0, 0.5, N), (() -> rand(rng, N)), m, opts)
    println("$(summary(m)) [box] => F: $(minimum(result)), C: $(Evolutionary.iterations(result))")
    @test Evolutionary.minimizer(result) ≈ [0.5, 0.25] atol=1e-1

    # Testing: CMA-ES
    opts = Evolutionary.Options(rng=rng)
    Random.seed!(rng, 42)
    result = Evolutionary.optimize(rosenbrock, (() -> rand(rng, Float32,N)), CMAES(mu = 5, lambda = 100, weights=zeros(Float32,100)), opts)
    println("(5/5,100)-CMA-ES => F: $(minimum(result)), C: $(Evolutionary.iterations(result))")
    test_result(result, N, 1e-2)

    Random.seed!(rng, 42)
    bc = BoxConstraints(fill(0.0, N), fill(0.5, N))
    result = Evolutionary.optimize(rosenbrock, bc, (() -> rand(rng, N)), CMAES(mu = 5, lambda = 100), opts)
    println("(5/5,100)-CMA-ES [box] => F: $(minimum(result)), C: $(Evolutionary.iterations(result))")
    @test Evolutionary.minimizer(result) ≈ [0.5, 0.25] atol=1e-5

    Random.seed!(rng, 42)
    con_c!(x) = [sum(x)]
    c = PenaltyConstraints(100.0, fill(0.0, 5N), Float64[], [1.0], [1.0], con_c!)
    result = Evolutionary.optimize(rosenbrock, c, (() -> rand(rng, 5N)), CMAES(mu = 40, lambda = 100), opts)
    println("(5/5,100)-CMA-ES [penalty] => F: $(minimum(result)), C: $(Evolutionary.iterations(result))")
    @test Evolutionary.minimizer(result) |> sum ≈ 1.0 atol=0.1

    Random.seed!(rng, 42)
    c = PenaltyConstraints(100.0, fill(0.0, 2N), fill(0.5, 2N), [1.0], [1.0], con_c!)
    result = Evolutionary.optimize(rosenbrock, c, (() -> rand(rng, 2N)), CMAES(mu = 8, lambda = 100), opts)
    println("(5/5,100)-CMA-ES [penalty] => F: $(minimum(result)), C: $(Evolutionary.iterations(result))")
    @test Evolutionary.minimizer(result) |> sum ≈ 1.0 atol=0.1
    @test all(0.0 <= x+0.01 && x-0.01 <= 0.5 for x in abs.(Evolutionary.minimizer(result)))

    # Testing: GA
    Random.seed!(rng, 42)
    opts = Evolutionary.Options(rng=rng)
    m = GA(
        populationSize = 100,
        ɛ = 0.1,
        selection = rouletteinv,
        crossover = IC(0.2),
        mutation = BGA(fill(0.5,N))
    )
    result = Evolutionary.optimize(rosenbrock, (() -> rand(rng, N)), m, opts)
    println("GA(p=100,x=0.8,μ=0.1,ɛ=0.1) => F: $(minimum(result)), C: $(Evolutionary.iterations(result))")
    test_result(result, N, 0.1)
    result = Evolutionary.optimize(rosenbrock, BoxConstraints(0.0, 0.5, N), (() -> rand(N)), m, opts)
    println("GA(p=100,x=0.8,μ=0.1,ɛ=0.1)[box] => F: $(minimum(result)), C: $(Evolutionary.iterations(result))")
    @test Evolutionary.minimizer(result) ≈ [0.5, 0.25] atol=0.1

    # Testing: DE
    Random.seed!(rng, 42)
    opts = Evolutionary.Options(rng=rng)
    result = Evolutionary.optimize(rosenbrock, (() -> rand(rng, N)), DE(populationSize = 100), opts)
    println("DE/rand/1/bin(F=1.0,Cr=0.5) => F: $(minimum(result)), C: $(Evolutionary.iterations(result))")
    test_result(result, N, 1e-1)
    result = Evolutionary.optimize(rosenbrock, BoxConstraints(0.0, 0.5, N), (() -> rand(rng, Float32,N)), DE(populationSize = 100), opts)
    println("DE/rand/1/bin(F=1.0,Cr=0.5)[box] => F: $(minimum(result)), C: $(Evolutionary.iterations(result))")
    @test Evolutionary.minimizer(result) ≈ [0.5, 0.25] atol=1e-1

end

