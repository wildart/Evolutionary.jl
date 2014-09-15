module TestNQueens
    using Evolutionary
    using Base.Test

    N = 8
    initial = rand(1:N,N)

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

end