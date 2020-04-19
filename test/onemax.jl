@testset "OneMax" begin

    # Initial population
    individualSize = 100
    initpop = (N -> BitArray(rand(Bool, N)))
    initpop2 = (() -> BitArray(rand(Bool, individualSize)))

    function Evolutionary.trace!(record::Dict{String,Any}, objfun, state, population, method::GA, options)
        idx = sortperm(state.fitpop)
        record["fitpop"] = state.fitpop[idx[1:5]]
    end

    # old api
    res = ga(
        x -> -sum(x),                           # Function to MINIMISE
        individualSize,                         # Length of chromosome
        initPopulation = initpop2,
        selection = uniformranking(3),
        mutation = flip,
        crossover = singlepoint,
        mutationRate = 0.6,
        crossoverRate = 0.2,
        iterations = 1500,
        tolIter = 20,
        populationSize = 100,
        interim = false, verbose=false);
    # show(res)
    println("GA:UR(3):FLP:SP (OneMax: -sum) => F: $(minimum(res)), C: $(Evolutionary.iterations(res))")
    @test sum(Evolutionary.minimizer(res)) == individualSize
    @test abs(minimum(res)) == individualSize

    # new api
    res = Evolutionary.optimize(
        x -> 1/sum(x),                 # Function to MINIMIZE
        initpop2,
        GA(
            selection = uniformranking(3),
            mutation =  flip,
            crossover = twopoint,
            mutationRate = 0.6,
            crossoverRate = 0.2,
            populationSize = 100,
        ),
        Evolutionary.Options(successive_f_tol = 20,iterations = 1500, store_trace=true));
    println("GA:UR(3):FLP:SP (OneMax: 1/sum) => F: $(minimum(res)), C: $(Evolutionary.iterations(res))")
    @test sum(Evolutionary.minimizer(res)) == individualSize
    @test 1/minimum(res) == individualSize
    @test Evolutionary.trace(res)[end].metadata["fitpop"][1] == minimum(res)

end
