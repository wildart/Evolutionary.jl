@testset "Mutations" begin
    lx = [0.0, 0.0]
    ux = [2.0, 1.0]
    @test PM(lx, ux, Inf)([1.0, 2.0])[1] ∈ [0.0, 2.0]
    @test PM(lx, ux, Inf)([1.0, 2.0])[2] == 1.0
    @test MIPM(lx, ux, Inf, 1.0)(Real[1.0, 2])[1] ∈ [0.0, 2.0]
    @test MIPM(lx, ux, 10.0, Inf)(Real[1.0, 2])[2] == 1.0
end