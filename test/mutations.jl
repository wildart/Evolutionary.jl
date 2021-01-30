@testset "Mutations" begin

    @testset "GA" begin
        lx = [0.0, 0.0]
        ux = [2.0, 1.0]
        @test PM(lx, ux, Inf)([1.0, 2.0])[1] ∈ [0.0, 2.0]
        @test PM(lx, ux, Inf)([1.0, 2.0])[2] == 1.0
        @test MIPM(lx, ux, Inf, 1.0)(Real[1.0, 2])[1] ∈ [0.0, 2.0]
        @test MIPM(lx, ux, 10.0, Inf)(Real[1.0, 2])[2] == 1.0
    end

    @testset "DE" begin
        rec = ones(2)
        mut = [[0.5, 1.0], [1.0, 0.5]]
        @test_throws AssertionError Evolutionary.differentiation(rec, mut[1:1])
        @test Evolutionary.differentiation(rec, mut) == [0.5, 1.5]
    end

    using Evolutionary, Test, Random

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