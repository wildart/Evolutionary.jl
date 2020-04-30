@testset "Knapsack" begin
    mass    = [1, 5, 3, 7, 2, 10, 5]
    utility = [1, 3, 5, 2, 5,  8, 3]

    fitnessFun = n -> (sum(mass .* n) <= 20) ? sum(utility .* n) : 0

    initpop = collect(rand(Bool,length(mass)))

    # old api
    result = ga(
        x -> 1 / fitnessFun(x),  # Function to MINIMISE
        length(initpop),         # Length of chromosome
        initPopulation = initpop,
        selection = rouletteinv,
        mutation = inversion,
        crossover = singlepoint,
        mutationRate = 0.2,
        crossoverRate = 0.5,
        ɛ = 0.1,                 # Elitism
        iterations = 20,
        tolIter = 20,
        populationSize = 50,
        interim = true);
    println("GA:RLT:INV:SP (1/objfun) => F: $(minimum(result)), C: $(Evolutionary.iterations(result))")
    @test 1. /Evolutionary.minimum(result) == 21.
    @test sum(mass .* Evolutionary.minimizer(result)) <= 20

    # new api
    result = Evolutionary.optimize(
        x -> -fitnessFun(x),
        rand(Bool,length(mass)),
        GA(
            selection = roulette,
            mutation = inversion,
            crossover = singlepoint,
            mutationRate = 0.2,
            crossoverRate = 0.5,
            ɛ = 0.1,                                # Elitism
            populationSize = 50,
        ));
    println("GA:RLT:INV:SP (-objfun) => F: $(minimum(result)), C: $(Evolutionary.iterations(result))")
    @test abs(Evolutionary.minimum(result)) == 21.
    @test sum(mass .* Evolutionary.minimizer(result)) <= 20
end
