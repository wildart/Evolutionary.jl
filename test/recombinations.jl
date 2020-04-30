@testset "Recombinations" begin

    compare(iter1, iter2) = length(setdiff(Set(iter1), Set(iter2))) == 0

    pop = [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0]
    ]
    @testset "ES" begin
        @test average(pop) ≈ [1/3.,1/3.]
        @test compare( (marriage(pop) for i in 1:100), ([p1,p2] for p1 in [0,1.], p2 in [0,1.]))
    end

    @testset "ES Strategies" begin
        @test averageSigma([strategy(σ = 1.0), strategy(σ = 2.0), strategy(σ = 3.0)]) == strategy(σ = 2.0)
        @test averageSigma([strategy(σ = 1.0), strategy(), strategy(σ = 3.0)]) == strategy(σ = 2.0)
        @test averageSigma([strategy(σ = fill(1.0,3)), strategy(σ =fill(2.0,3)), strategy(σ = fill(3.0,3))]) == strategy(σ = fill(2.0,3))
        @test averageSigma([strategy(σ = fill(1.0,3)), strategy(), strategy(σ = fill(3.0,3))]) == strategy(σ = fill(2.0,3))
    end

    @testset "GA" begin
        @test identity(pop[1], pop[2]) == (pop[1], pop[2])
        @test compare((singlepoint(pop[1], pop[2]) for i in 1:100),
                      [([0.0, 1.0],[1.0, 0.0]),([1.0, 1.0],[0.0, 0.0])])
        @test compare((twopoint(pop[1], pop[2]) for i in 1:100),
                      [([0.0, 1.0],[1.0, 0.0]),([1.0, 1.0],[0.0, 0.0]),([0.0, 0.0], [1.0, 1.0])])
        @test compare((uniform(pop[1], pop[2]) for i in 1:100),
                      [([0.0, 1.0],[1.0, 0.0]),([1.0, 1.0],[0.0, 0.0]),([0.0, 0.0], [1.0, 1.0]),([1.0, 0.0], [0.0, 1.0])])
        @test compare((discrete(pop[1], pop[2]) for i in 1:100),
                      ([p1,p2],[p3,p4]) for p1 in [0,1.], p2 in [0,1.], p3 in [0,1.], p4 in [0,1.])

        @test waverage([1.0,0.5])(pop[1], pop[2]) == ([1.0, 2.0], [1.0, 2.0])

        @test pmx(pop[1], pop[2]) == ([0.0, 1.0], [1.0, 0.0])
        @test sum(pmx(collect(1:9), collect(9:-1:1))) == fill(10,9)
        @test sum(ox1(collect(1:9), collect(9:-1:1))) == fill(10,9)
        @test sum(ox2(collect(1:9), collect(9:-1:1))) == fill(10,9)
        @test sum(cx(collect(1:9), collect(9:-1:1))) == fill(10,9)
        @test sum(pos(collect(1:9), collect(9:-1:1))) == fill(10,9)
        @test cx([8,4,7,3,6,2,5,1,9,0],collect(0:9)) == ([8,1,2,3,4,5,6,7,9,0],[0,4,7,3,6,2,5,1,8,9])
    end

end