@testset "Selections" begin

    @testset "Rank" begin
        Random.seed!(2);
        fitness = [5, 1, 3, 4, 2.0]
        s = ranklinear(1.0)
        @test s(fitness, 2) == [2,3]

        s = ranklinear(2.0)
        @test s([1,2], 2) == [2,2]
    end

    @testset "Uniform" begin
        Random.seed!(2);
        s = uniformranking(2)
        @test sort(unique(s([1.0,2.0,3.0], 10))) == [1,2]
        @test sort(unique(s([5,2,3], 5))) == [2,3]
        @test_throws AssertionError s([1.], 4)
    end

    @testset "Roulette" begin
        Random.seed!(2);
        @test roulette([30.0, -0.1,-0.2, -30.0], 2) == [1, 4]
        @test roulette([0.0,0.0,3.0], 2) == [3, 3]
        @test roulette([0,2,0], 2) == [2,2]
        @test rouletteinv([0.0,0.0,3.0], 2) == [1, 1]
        @test rouletteinv([1,2,0], 2) == [3,3]
    end

    @testset "Truncation" begin
        @test truncation([0.0,2.0,3.0], 2) == [1,2]
        @test truncation([0,2,1], 1) == [1]
        @test_throws AssertionError truncation([1.,2.], 5)
    end

    @testset "Tournament" begin
        Random.seed!(2);
        @test_throws AssertionError tournament(0)
        t = tournament(2)
        @test t([0,2,0],2) == [2,2]
        @test t([0.0,0.0,1.0],2) == [3,3]
    end

    @testset "SUS" begin
        @test sort(unique(sus([1.0,0.0,2.0], 5))) == [1,3]
        @test sort(unique(sus([0,1,2], 5))) == [2,3]
    end

    @testset "DE" begin
        N = 5
        rng = collect(1:10)
        idxs = random(rng, N)
        @test length(idxs) == N
        @test all(i-> i ∈ rng, idxs)
        @test !all(length(unique(random(rng, N))) == N for i in 1:100)

        idxs = randomoffset(rng, N)
        @test length(idxs) == N
        @test all(i-> i ∈ rng, idxs)
        @test all( let idxs = randomoffset(rng, N); let l = (idxs[1] - 1 + N)%10; idxs[N] == (l == 0 ? 10 : l) end end for i in 1:100 )

        idxs = best(reverse(rng), N)
        @test length(idxs) == N
        @test all(i-> i ∈ rng, idxs)
        @test idxs == fill(10, N)

        idxs = permutation(rng, N)
        @test length(idxs) == N
        @test all(i-> i ∈ rng, idxs)
        @test all(length(unique(permutation(rng, N))) == N for i in 1:100)

        @testset "Mutually Exclusive Indices" for i in idxs
            targets = Evolutionary.randexcl(rng, [i], 4)
            @test length(unique(push!(targets, i))) == N
        end
    end

end
