@testset "Schwefel CMA-ES" begin

    # Schwefel's ellipsoid test function:
    # a moderately conditioned ellipsoid with a dominating isolated eigenvalue
    function schwefel(x::AbstractVector{T}) where {T<:AbstractFloat}
        out = zero(T)
        for i = 1:length(x)
            out += sum(abs2, x[1:i])
        end
        return out
    end

    # Parameters
    N = 30

    # Testing: CMA-ES
    result, fitness, cnt = cmaes(schwefel, N; μ = 3, λ = 12, iterations = 1000)
    println("(3/3,12)-CMA-ES (Schwefel) => F: $(fitness), C: $(cnt)")

    @test ≈(result, zeros(N), atol=1e-5)
    @test ≈(fitness, 0.0, atol=1e-5)
end
