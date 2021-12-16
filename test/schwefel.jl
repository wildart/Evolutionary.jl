@testset "Schwefel CMA-ES" begin

    rng = StableRNG(42)

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
    function Evolutionary.trace!(record::Dict{String,Any}, objfun, state, population, method::CMAES, options)
        record["σ"] = state.σ
    end

    # Testing: CMA-ES
    opts = Evolutionary.Options(rng=rng)
    result = Evolutionary.optimize(schwefel, ()->rand(rng, N), CMAES(mu = 3, lambda = 12, c_1=0.05, weights=[ones(6)./6; -ones(6)./6]), opts)
    println("(3/3,12)-CMA-ES => F: $(minimum(result)), C: $(Evolutionary.iterations(result))")
    @test Evolutionary.converged(result)
    @test Evolutionary.minimizer(result) ≈ zeros(N) atol=1e-5
    @test minimum(result) ≈ 0.0 atol=1e-5

    bounds = Evolutionary.ConstraintBounds(fill(-1.0f0,N),fill(1.0f0,N),[],[])
    opts = Evolutionary.Options(store_trace=true, iterations=10, rng=rng)
    Random.seed!(rng, 42)
    result = Evolutionary.optimize(schwefel, bounds, CMAES(mu = 3, lambda = 12, weights=zeros(Float32,12)), opts)
    println("(3/3,12)-CMA-ES [bounds] => F: $(minimum(result)), C: $(Evolutionary.iterations(result))")
    @test Evolutionary.iterations(result) == 10
    @test !Evolutionary.converged(result)
    @test haskey(Evolutionary.trace(result)[end].metadata, "σ")
end
