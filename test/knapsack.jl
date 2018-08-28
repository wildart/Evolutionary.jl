@testset "Knapsack" begin
    mass    = [1, 5, 3, 7, 2, 10, 5]
    utility = [1, 3, 5, 2, 5,  8, 3]

    function fitness(n::AbstractVector)
        total_mass = sum(mass .* n)
        return (total_mass <= 20) ? sum(utility .* n) : 0
    end

    initpop = collect(rand(Bool,length(mass)))

    best, invbestfit, generations, tolerance, history = ga(
        x -> 1 / fitness(x),                    # Function to MINIMISE
        length(initpop),                        # Length of chromosome
        initPopulation = initpop,
        selection = roulette,                   # Options: sus
        mutation = inversion,                   # Options:
        crossover = singlepoint,                # Options:
        mutationRate = 0.2,
        crossoverRate = 0.5,
        ɛ = 0.1,                                # Elitism
        iterations = 20,
        populationSize = 50,
        interim = true);

    @test fitness(best) == 21.
    @test 1. /invbestfit == 21.
    @test sum(mass .* best) <= 20
end
