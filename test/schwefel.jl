module TestCMAES
    using Evolutionary
    using Base.Test

    # Schwefel's ellipsoid test function:
    # a moderately conditioned ellipsoid with a dominating isolated eigenvalue
    function schwefel{T <: Vector}(x::T)
        out = 0
        for i = 1:length(x)
            out += sumabs2(x[1:i])
        end
        return out
    end

    # Parameters
    N = 30

    # Testing: CMA-ES
    result, fitness, cnt = cmaes(schwefel, N; μ = 3, λ = 12, iterations = 1000)
    println("(3/3,12)-CMA-ES (Schwefel) => F: $(fitness), C: $(cnt)")

    @test_approx_eq_eps result zeros(N) 1e-5
    @test_approx_eq_eps fitness 0.0 1e-5

end