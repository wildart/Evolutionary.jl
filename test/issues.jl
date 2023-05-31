@testset "Issues" begin
    # issue #105
    res = Evolutionary.optimize(sum, ones(5), ES(μ = 40, λ = 100))
    @test minimum(res) == 5

end