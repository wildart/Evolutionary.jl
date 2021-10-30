@testset "Knapsack" begin
    Random.seed!(42)

    mass    = [1, 5, 3, 7, 2, 10, 5]
    utility = [1, 3, 5, 2, 5,  8, 3]

    fitnessFun = n -> (sum(mass .* n) <= 20) ? sum(utility .* n) : 0

    initpop = collect(rand(Bool,length(mass)))

    result = Evolutionary.optimize(
        x -> -fitnessFun(x),
        initpop,
        GA(
            selection = tournament(3),
            mutation = inversion,
            crossover = SPX,
            mutationRate = 0.9,
            crossoverRate = 0.2,
            ɛ = 0.1,                                # Elitism
            populationSize = 100,
        ));
    println("GA:RLT:INV:SP (-objfun) => F: $(minimum(result)), C: $(Evolutionary.iterations(result))")
    @test abs(Evolutionary.minimum(result)) == 21.
    @test sum(mass .* Evolutionary.minimizer(result)) <= 20


    # with a constraint
    fitnessFun = n -> sum(utility .* n)

    cf(n) = [ sum(mass .* n) ] # constraint function
    lc   = [0] # lower bound for constraint function
    uc   = [20]   # upper bound for constraint function
    con = WorstFitnessConstraints(Int[], Int[], lc, uc, cf)

    initpop = BitVector(rand(Bool,length(mass)))

    result = Evolutionary.optimize(
        x -> -fitnessFun(x), con,
        initpop,
        GA(
            selection = roulette,
            mutation = flip,
            crossover = SPX,
            mutationRate = 0.9,
            crossoverRate = 0.1,
            ɛ = 0.1,                                # Elitism
            populationSize = 50,
        ));
    println("GA:RLT:FL:SP (-objfun) => F: $(minimum(result)), C: $(Evolutionary.iterations(result))")
    @test abs(Evolutionary.minimum(result)) == 21.
    @test sum(mass .* Evolutionary.minimizer(result)) <= 20
end
