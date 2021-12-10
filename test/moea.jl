@testset "Multi-objective EA" begin

    rng = StableRNG(42)
    opts = Evolutionary.Options(rng=rng)

    # domination
    P =[ [0,0,0], [0,1,0], [-1,0,0], [0,-1,0] ]
    @test Evolutionary.dominate(P[1], P[2]) == 1
    @test Evolutionary.dominate(P[1], P[1]) == 0
    @test Evolutionary.dominate(P[1], P[3]) == -1
    @test Evolutionary.dominate(P[3], P[4]) == 0
    @test Evolutionary.dominations(P)[:, 1] == [0, -1, 1, 1]

    # convergence
    R = reshape([10,0,6,1,2,2,1,6,0,10],2,5)
    A = reshape([4,2,3,3,2,4], 2, 3)
    B = reshape([8,2,4,4,2,8], 2, 3)
    @test Evolutionary.igd(A,R) ≈ 3.707092031609239
    @test Evolutionary.igd(B,R) ≈ 2.59148346584763
    @test Evolutionary.spread(R) ≈ 0.0
    @test Evolutionary.spread(R[1:1,1:4]) ≈ 3.75

    # Schaffer F2

    schafferf2(x::AbstractVector) = [ x[1]^2,  (x[1]-2)^2 ]
    Random.seed!(rng, 1)
    result = Evolutionary.optimize(schafferf2, [5.0], NSGA2(), opts)
    println("NSGA2:2RLT:SBX:PLM => F: $(minimum(result)), C: $(Evolutionary.iterations(result))")
    @test isnan(Evolutionary.minimum(result))
    mvs = vcat(Evolutionary.minimizer(result)...)
    @test 1-sum(0 .<= mvs .<= 2)/length(mvs) ≈ 0.0 atol=0.1 # PO ∈ [0,2]

    function schafferf2!(F, x::AbstractVector) # in-place update
        F[1] = x[1]^2
        F[2] = (x[1]-2)^2
        F
    end
    Random.seed!(rng, 2)
    result = Evolutionary.optimize(schafferf2!, zeros(2), [5.0], NSGA2(), opts)
    println("NSGA2:2RLT:SBX:PLM => F: $(minimum(result)), C: $(Evolutionary.iterations(result))")
    @test isnan(Evolutionary.minimum(result))
    mvs = vcat(Evolutionary.minimizer(result)...)
    @test 1-sum(0 .<= mvs .<= 2)/length(mvs) ≈ 0.0 atol=0.1 # PO ∈ [0,2]

end
