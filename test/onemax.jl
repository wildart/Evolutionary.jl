@testset "OneMax" begin

    rng = StableRNG(42)

    # Initial population
    N = 100
    initpop = (() -> BitArray(rand(rng, Bool, N)))

    function Evolutionary.trace!(record::Dict{String,Any}, objfun, state, population, method::GA, options)
        idx = sortperm(state.fitpop)
        record["fitpop"] = state.fitpop[idx[1:5]]
    end

    res = Evolutionary.optimize(
        x -> -sum(x),                 # Function to MINIMIZE
        initpop,
        GA(
            selection = tournament(3),
            mutation =  flip,
            crossover = TPX,
            mutationRate = 0.05,
            crossoverRate = 0.85,
            populationSize = N,
        ),
        Evolutionary.Options(rng=rng, store_trace=true));
    println("GA:TOUR(3):FLP:TPX (OneMax: 1/sum) => F: $(minimum(res)), C: $(Evolutionary.iterations(res))")
    @test sum(Evolutionary.minimizer(res)) >= N-3
    @test abs(minimum(res)) >= N-3
    @test Evolutionary.trace(res)[end].metadata["fitpop"][1] == minimum(res)

end
