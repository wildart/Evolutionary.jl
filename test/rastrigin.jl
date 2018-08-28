@testset "Rastrigin" begin

    function test_result(result::Vector, fitness::Float64, N::Int, tol::Float64)
        if round(fitness) == 0
            @test ≈(result, zeros(N), atol=tol)
            @test ≈(fitness,0.0, atol=tol)
        else
            @warn("Found local minimum!!!")
            @test sum(round.(abs.(result))) < N
        end
    end

    # Objective function
    function rastrigin(x::AbstractVector{T}) where {T <: AbstractFloat}
        n = length(x)
        return 10n + sum([ x[i]^2 - 10cos(convert(T,2π*x[i])) for i in 1:n ])
    end

    # Parameters
    N = 3
    P = 100

    # Testing: (μ/μ_I,λ)-σ-Self-Adaptation-ES
    # with non-isotropic mutation operator y' := y + (σ_1 N_1(0, 1), ..., σ_N N_N(0, 1))
    result, fitness, cnt = es( rastrigin, N;
            initStrategy = strategy(σ = .5ones(N), τ = 1/sqrt(2*N), τ0 = 1/sqrt(N)),
            recombination = average, srecombination = averageSigmaN,
            mutation = anisotropic, smutation = anisotropicSigma,
            selection=:comma,
            μ = 15, λ = P,
            iterations = 1000)
    println("(15/15,$(P))-σ-SA-ES-AS => F: $(fitness), C: $(cnt), OBJ: $(result)")
    test_result(result, fitness, N, 1e-1)

    # Testing: CMA-ES
    result, fitness, cnt = cmaes( rastrigin, N; μ = 15, λ = P, tol = 1e-8)
    println("(15/15,$(P))-CMA-ES => F: $(fitness), C: $(cnt), OBJ: $(result)")
    test_result(result, fitness, N, 1e-1)

    # Testing: GA
    selections = [roulette, sus, ranklinear(1.5)]
    crossovers = [discrete, intermediate(0.), intermediate(0.25), line(0.2)]
    mutations = [domainrange(fill(0.5,N)), domainrange(fill(1.0,N))]

    @testset "GA settings" for ss in selections, xovr in crossovers, ms in mutations
        result, fitness, cnt = ga( rastrigin, N;
                populationSize = P,
                ɛ = 0.1,
                selection = ss,
                crossover = xovr,
                mutation = ms,
                tol = 1e-5)
        # println("GA(p=$(P),x=.8,μ=.1,ɛ=0.1) => F: $(fitness), C: $(cnt), OBJ: $(result)")
        test_result(result, fitness, N, 1e-1)
    end
end
