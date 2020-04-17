@testset "OneMax" begin

    # Initial population
    individualSize = 100
    initpop = (N -> BitArray(rand(Bool, N)))

    best, invbestfit, generations, tolerance, history = optimize(
        x -> 1 / sum(x),                 # Function to MINIMIZE
        GA(
            N=individualSize,            # Length of chromosome
            initPopulation = initpop,
            selection = tournament(3),
            mutation =  flip,
            crossover = singlepoint,
            mutationRate = 0.1,
            crossoverRate = 0.1,
            tolIter = 20,
            populationSize = 100,
            interim = true
        ),
        iterations = 3000, debug=false);

    @test sum(best) == individualSize
end
