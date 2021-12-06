@testset "Mutations" begin

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
        lx = [0.0, 0.0]
        ux = [2.0, 1.0]
        @test PM(lx, ux, Inf)([1.0, 2.0])[1] ∈ [0.0, 2.0]
        @test PM(lx, ux, Inf)([1.0, 2.0])[2] == 1.0
        @test MIPM(lx, ux, Inf, 1.0)(Real[1.0, 2])[1] ∈ [0.0, 2.0]
        @test MIPM(lx, ux, 10.0, Inf)(Real[1.0, 2])[2] == 1.0

        mut = uniform(3.0)
        @test all(mut(ones(3)) .!= 1)
        @test all(map(sum ∘ mut, [ones(3) for i in 1:10]) .!= 3)
        mut = uniform(0.0)
        @test all(map(sum ∘ mut, [ones(3) for i in 1:10]) .== 3)

        mut = gaussian(3.0)
        @test all(mut(ones(3)) .!= 1)
        @test all(map(sum ∘ mut, [ones(3) for i in 1:10]) .!= 3)
        mut = gaussian(0.0)
        @test all(map(sum ∘ mut, [ones(3) for i in 1:10]) .== 3)

        ex = PLM(ones(1000), pm=1.0)(zeros(1000)) |> extrema
        @test ex[1] >= -1
        @test ex[2] <= 1

    end

    @testset "DE" begin
        rec = ones(2)
        mut = [[0.5, 1.0], [1.0, 0.5]]
        @test_throws AssertionError Evolutionary.differentiation(rec, mut[1:1])
        @test Evolutionary.differentiation(rec, mut) == [0.5, 1.5]
    end

    @testset "GP" begin
        Random.seed!(9874984737482)
        H = 2
        tr = TreeGP(maxdepth=H, initialization=:grow);
        ex = rand(tr, H)

        # subtree mutation does not produce offspring expression longer then the parent
        mut = subtree(tr)
        off = [mut(copy(ex)) for i in 1:10]
        @testset "Offspring Height" for i in 1:10
            @test all(o->Evolutionary.height(o) <= H, map(mut, off))
        end

        # subtree mutation does not produce offspring expression longer then the parent
        mut = subtree(tr; growth=0.6)
        off = [mut(copy(ex)) for i in 1:10]
        @testset "Offspring Height" for i in 1:5
            @test all(o->Evolutionary.height(o) <= 3i, map(mut, off))
        end

        # hoist
        mut = subtree(tr; growth=0.6)
        off = [mut(copy(ex)) for i in 1:10]
        for i in 1:5; map(mut, off); end
        mut = hoist(tr)
        @testset "Hoist Mutation" for i in 1:10
            n = length(off[i])
            m = length(mut(off[i]))
            @test n > m
        end

        # shrink
        mut = subtree(tr; growth=0.6)
        off = [mut(copy(ex)) for i in 1:10]
        for i in 1:5; map(mut, off); end
        mut = shrink(tr)
        @testset "Shrink Mutation" for i in 1:10
            n = length(off[i])
            m = length(mut(off[i]))
            @test n >= m
        end

        # point
        mut = point(tr)
        @testset "Point Mutation" for i in 1:10
            mex = mut(copy(ex))
            @test length(ex) == length(mex) == 3
            @test sum(x == y for (x,y) in zip(ex.args, mex.args)) >= 2
        end

    end
end

