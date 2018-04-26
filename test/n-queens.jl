@testset "n-Queens" begin

    N = 8
    P = 50
    generatePositions(N::Int) = collect(1:N)[randperm(N)]

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
    for muts in [inversion, insertion, swap2, scramble, shifting]
        result, fitness, cnt = ga(nqueens, N;
            initPopulation = generatePositions,
            populationSize = P,
            selection = sus,
            crossover = pmx,
            mutation = muts)
        println("GA:PMX:$(string(muts))(N=$(N), P=$(P)) => F: $(fitness), C: $(cnt), OBJ: $(result)")
        @test nqueens(result) == 0
    end

    # Testing: ES
    for muts in [inversion, insertion, swap2, scramble, shifting]
        result, fitness, cnt = es(nqueens, N;
            initPopulation = generatePositions,
            mutation = mutationwrapper(muts),
            μ = 15, ρ = 1, λ = P)
        println("(15+$(P))-ES:$(string(muts)) => F: $(fitness), C: $(cnt), OBJ: $(result)")
        @test nqueens(result) == 0
    end

end
