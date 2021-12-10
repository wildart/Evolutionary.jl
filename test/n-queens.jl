@testset "n-Queens" begin
    rng = StableRNG(42)

    N = 8
    P = 100
    generatePositions = ()->randperm(rng, N)

    # Vector of N cols filled with numbers from 1:N specifying row position
    function nqueens(queens::Vector{Int})
        n = length(queens)
        fitness = 0
        for i=1:(n-1)
            for j=(i+1):n
                k = abs(queens[i] - queens[j])
                if (j-i) == k || k == 0
                    fitness += 1
                end
                # println("$(i),$(queens[i]) <=> $(j),$(queens[j]) : $(fitness)")
            end
        end
        return fitness
    end
    @test nqueens([2,4,1,3]) == 0
    @test nqueens([3,1,2]) == 1

    # Testing: GA solution with various mutations
    for muts in [inversion, insertion, swap2, scramble, shifting],
        xo in [PMX, OX1, CX, OX2, POS]
        Random.seed!(rng, 42)
        result = Evolutionary.optimize(nqueens,
            generatePositions,
            GA(
                populationSize = P,
                selection = tournament(5),
                crossover = xo,
                crossoverRate = 0.89,
                mutation = muts,
                mutationRate = 0.06,
               ), Evolutionary.Options(rng=rng, successive_f_tol=30));
        # show(result)
        println("GA:$(string(xo)):$(string(muts))(N=$(N), P=$(P)) => F: $(minimum(result)), C: $(Evolutionary.iterations(result))")
        @test nqueens(Evolutionary.minimizer(result)) <= 1
        @test Evolutionary.iterations(result) < 500 # Test early stopping
    end

    # Testing: ES
    μ = 20
    for muts in [inversion, insertion, swap2, scramble, shifting]
        for sel in [:plus, :comma]
            Random.seed!(rng, 42)
            result = Evolutionary.optimize(nqueens, generatePositions,
                ES(
                   mutation = mutationwrapper(muts),
                   μ=μ, ρ=1, λ=P,
                   selection=sel
                  ), Evolutionary.Options(show_trace=false, rng=rng))
            println("($μ$(sel == :plus ? "+" : ",")$(P))-ES:$(string(muts)) => F: $(minimum(result)), C: $(Evolutionary.iterations(result))")
            # show(result)
            @test nqueens(Evolutionary.minimizer(result)) == 0
        end
    end

end

