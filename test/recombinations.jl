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
        ss = [IsotropicStrategy{Float64}(1.0,1,0),
              IsotropicStrategy{Float64}(2.0,0,0), IsotropicStrategy{Float64}(3.0,0,0)]
        @test average(ss).σ == ss[2].σ
        @test gaussian(ss[1]).σ == ss[1].σ
        @test ss[1].σ != 1.0
        ss = [AnisotropicStrategy{Float64}(fill(1.0,3),1,1),
              AnisotropicStrategy{Float64}(fill(2.0,3),0,0),
              AnisotropicStrategy{Float64}(fill(3.0,3),0,0)]
        @test average(ss).σ == ss[2].σ
        @test gaussian(ss[1]).σ == ss[1].σ
        @test ss[1].σ != fill(1.0,3)
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

        @test map(sum, HX(pop[1], pop[2])) |> collect ≈ [1.0, 1.0]
        @test map(abs∘first∘diff, LX()(pop[1], pop[2])) |> collect ≈ [1.0, 1.0]
        @test mapslices(diff, hcat(MILX(1., 0., 1.)(Real[0, 0.0], Real[1, 0.0])...), dims=2) |> vec ≈ [1.0, 0.0]

        @test PMX(pop[1], pop[2]) == ([0.0, 1.0], [1.0, 0.0])
        @test sum(PMX(collect(1:9), collect(9:-1:1))) == fill(10,9)
        @test sum(OX1(collect(1:9), collect(9:-1:1))) == fill(10,9)
        @test sum(OX2(collect(1:9), collect(9:-1:1))) == fill(10,9)
        @test sum(CX(collect(1:9), collect(9:-1:1))) == fill(10,9)
        @test sum(POS(collect(1:9), collect(9:-1:1))) == fill(10,9)
        @test CX([8,4,7,3,6,2,5,1,9,0],collect(0:9)) == ([8,1,2,3,4,5,6,7,9,0],[0,4,7,3,6,2,5,1,8,9])
    end

    @testset "DE" begin
        v1 = collect(1:10)
        v2 = fill(0,10)

        xvr = uniformbin(0.0)
        @test first( xvr(v1, v2) )== v1
        xvr = uniformbin(1.0)
        @test last( xvr(v1, v2) )== v1
        xvr = uniformbin(0.5)
        m1, m2 = xvr(v1, v2)
        @test m1.+m2 == v1

        xvr = exponential(0.0)
        @test sum(first( xvr(v1, v2) )) <= 10
        xvr = exponential(1.0)
        @test first( xvr(v1, v2) )== v1
        xvr = exponential(0.05)
        @test sum(first( xvr(v1, v2) )) <= 19
    end

end