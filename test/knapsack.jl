@testset "Knapsack" begin
    rng = StableRNG(42)

    mass    = [1, 5, 3, 7, 2, 10, 5]
    utility = [1, 3, 5, 2, 5,  8, 3]

    fitnessFun = n -> (sum(mass .* n) <= 20) ? sum(utility .* n) : 0

    Random.seed!(rng, 42)
    initpop = ()->rand(rng, Bool, length(mass))
    result = Evolutionary.optimize(
        x -> -fitnessFun(x),
        initpop,
        GA(
            selection = roulette,
            mutation = swap2,
            crossover = SPX,
            mutationRate = 0.05,
            crossoverRate = 0.85,
            ɛ = 0.05,                                # Elitism
            populationSize = 100,
           ), Evolutionary.Options(show_trace=false, rng=rng)
       );
    println("GA:RLT:SWP:SPX (-objfun) => F: $(minimum(result)), C: $(Evolutionary.iterations(result))")
    @test abs(Evolutionary.minimum(result)) == 21.
    @test sum(mass .* Evolutionary.minimizer(result)) <= 20


    # with a constraint
    fitnessFun = n -> sum(utility .* n)

    cf(n) = [ sum(mass .* n) ] # constraint function
    lc   = [0] # lower bound for constraint function
    uc   = [20]   # upper bound for constraint function
    con = WorstFitnessConstraints(Int[], Int[], lc, uc, cf)

    Random.seed!(rng, 42)
    initpop = BitVector(rand(rng, Bool, length(mass)))
    result = Evolutionary.optimize(
        x -> -fitnessFun(x), con,
        initpop,
        GA(
            selection = tournament(3),
            mutation = inversion,
            crossover = SPX,
            mutationRate = 0.05,
            crossoverRate = 0.85,
            ɛ = 0.1,                                # Elitism
            populationSize = 50,
           ), Evolutionary.Options(show_trace=false, rng=rng, successive_f_tol=10));
    println("GA:TRN3:INV:SPX (-objfun) => F: $(minimum(result)), C: $(Evolutionary.iterations(result))")
    @test abs(Evolutionary.minimum(result)) == 21.
    @test sum(mass .* Evolutionary.minimizer(result)) <= 20

end
