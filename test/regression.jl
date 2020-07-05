@testset "Regression" begin
    ## data
    m, n, n2 = 9, 6, 3
    X, A, E = randn(m, n), randn(n, n2), randn(m, n2) * 0.1
    Y = X * A + E
    r = 0.1

    linear(w) = sum(abs2, Y .- X*w)
    ridge(w) = linear(w) + r*sum(abs2, w)
    ridgevec(w) = ridge(reshape(w,n,n2))

    # using vector individual
    result = Evolutionary.optimize(ridgevec, randn(n*n2), CMAES(μ=100, c_1=0.05))
    @test sum(abs, inv(X'X + r * I)*X'*Y .- reshape(Evolutionary.minimizer(result), n, n2)) ≈ 0.001 atol=1e-3

    # using matrix individual
    res1 = Evolutionary.optimize(linear, randn(n,n2), CMAES(μ=100, c_1=0.05))
    res2 = Evolutionary.optimize(ridge, randn(n,n2), CMAES(μ=100, c_1=0.05))
    @test sum(abs, inv(X'X )*X'*Y .- Evolutionary.minimizer(res1)) ≈ 0.001 atol=1e-3
    @test sum(abs, inv(X'X + r * I)*X'*Y .- Evolutionary.minimizer(res2)) ≈ 0.001 atol=1e-3
end
