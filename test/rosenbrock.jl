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

    # Testing: (15,3/(+,)100)-ES
    settings = [
        :isotropic=>(gaussian, gaussian, IsotropicStrategy(N)),
        :isotropic=>(cauchy, gaussian, IsotropicStrategy(N)),
        :anisotropic=>(gaussian, gaussian, AnisotropicStrategy(N))
    ]
    selections = [:plus, :comma]
    @testset "ES settings" for (sn,ss) in settings, sel in selections
        result = Evolutionary.optimize( rosenbrock, (() -> rand(N)),
            ES(
                initStrategy = ss[3],
                recombination = average, srecombination = average,
                mutation = ss[1], smutation = ss[2],
                μ = 10, ρ = 3, λ = 100, selection=sel
            ), Evolutionary.Options(iterations=1000, successive_f_tol=25)
        )
        println("(15/3$(sel == :plus ? "+" : ",")100)-ES:$(sn) => F: $(minimum(result)), C: $(Evolutionary.iterations(result))")
        test_result(result, N, sel == :plus ? 0.1 : 0.5)
    end

    # Testing: CMA-ES
    result = Evolutionary.optimize(rosenbrock, (() -> rand(Float32,N)), CMAES(mu = 5, lambda = 100))
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

    # Testing: DE
    result = Evolutionary.optimize(rosenbrock, (() -> rand(N)), DE(populationSize = 100))
    println("DE/rand/1/bin(F=1.0,Cr=0.5) => F: $(minimum(result)), C: $(Evolutionary.iterations(result))")
    test_result(result, N, 1e-1)

end


@testset "clip bounds" begin
    function test_result(result::Evolutionary.EvolutionaryOptimizationResults, N::Int, tol::Float64)
        fitness = minimum(result)
        extremum = Evolutionary.minimizer(result)
        @test sum(abs2,extremum.-ones(N)) ≈ 0.0 atol=tol
        @test fitness ≈ 0.0 atol=tol
    end

    N = 2
    lb, ub = -1., 2.

    # Objective function
    function rosenbrock1(x::AbstractVector)
        if any(xi -> (xi < lb || xi > ub), x)
            error("bounds error!")
        end
        (1.0 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
    end

    # Testing: CMA-ES-No Bounds
    @test_throws ErrorException Evolutionary.ClipBounds(2.2, -0.9)
    @test_throws ErrorException Evolutionary.optimize(rosenbrock1, [1.1, 0.93], CMAES(mu = 5, lambda = 100))


    # Testing: CMA-ES-Scalar Bounds
    bounds = ClipBounds(lb, ub)

    result = Evolutionary.optimize(rosenbrock1, [1.1, 0.93], CMAES(mu = 5, lambda = 100), Evolutionary.Options(bounds=bounds))
    println("(5/5,100)-CMA-ES => F: $(minimum(result)), C: $(Evolutionary.iterations(result)), miminimzer = $(result.minimizer)")
    test_result(result, N, 1e-2)

    # Testing: CMA-ES-Scalar Vector Bounds
    bounds = ClipBounds([lb, lb], [ub, ub])

    result = Evolutionary.optimize(rosenbrock1, [1.1, 0.93], CMAES(mu = 5, lambda = 100), Evolutionary.Options(bounds=bounds))
    println("(5/5,100)-CMA-ES => F: $(minimum(result)), C: $(Evolutionary.iterations(result)), miminimzer = $(result.minimizer)")
    test_result(result, N, 1e-2)
end