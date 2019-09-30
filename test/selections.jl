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
        @test sort(unique(rankuniform([1.0,2.0,3.0], 2))) == [2,3]
        @test sort(unique(rankuniform([1,2,3], 2))) == [2,3]
        @test_throws AssertionError rankuniform([1.,2.], 3)
    end

    @testset "Roulette" begin
        @test roulette([0.0,0.0,3.0], 2) == [3, 3]
        @test roulette([0,2,0], 2) == [2,2]
    end

    @testset "Truncation" begin
        @test truncation([0.0,2.0,3.0], 2) == [3,2]
        @test truncation([0,2,1], 1) == [2]
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

end

