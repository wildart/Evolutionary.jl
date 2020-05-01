@testset "Rastrigin" begin

    function test_result(result::Evolutionary.EvolutionaryOptimizationResults, N::Int, tol::Float64)
        fitness = minimum(result)
        extremum = Evolutionary.minimizer(result)
        if round(fitness) == 0
            @test extremum ≈ zeros(N) atol=tol
            @test fitness ≈ 0.0 atol=tol
        else
            # @warn("Found local minimum!!!")
            @test sum(abs, extremum) < N
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
    initState = ()->rand(N)

    # Testing: (μ/μ_I,λ)-σ-Self-Adaptation-ES
    # with non-isotropic mutation operator y' := y + (σ_1 N_1(0, 1), ..., σ_N N_N(0, 1))
    result = Evolutionary.optimize( rastrigin,
        initState,
        ES(
            initStrategy = AnisotropicStrategy(N),
            recombination = average, srecombination = average,
            mutation = gaussian, smutation = gaussian,
            selection=:comma,
            μ = 15, λ = P
        ),Evolutionary.Options(iterations=1000, show_trace=false)
    )
    println("(15,$(P))-σ-SA-ES => F: $(minimum(result)), C: $(Evolutionary.iterations(result))")
    test_result(result, N, 1e-1)

    # Testing: CMA-ES
    result = Evolutionary.optimize(rastrigin, initState, CMAES(μ = 15, λ = P))
    println("(15/15,$(P))-CMA-ES => F: $(minimum(result)), C: $(Evolutionary.iterations(result))")
    test_result(result, N, 1e-1)

    # Testing: GA
    selections = [:roulette=>rouletteinv, :sus=>susinv, :rank=>ranklinear(1.5)]
    crossovers = [:discrete=>discrete, :intermediate0=>intermediate(0.), :intermediate0_25=>intermediate(0.25), :line=>line(0.2)]
    mutations = [:domrng0_5=>domainrange(fill(0.5,N)), :domrng1=>domainrange(fill(1.0,N))]

    @testset "GA settings" for (sn,ss) in selections, (xn,xovr) in crossovers, (mn,ms) in mutations
        result = Evolutionary.optimize( rastrigin, initState,
            GA(
                populationSize = P,
                ɛ = 0.1,
                selection = ss,
                crossover = xovr,
                mutation = ms
            ), Evolutionary.Options(iterations=1000, successive_f_tol=25)
        )
        println("GA:$(sn):$(xn):$(mn)(N=$(N),P=$(P),x=.8,μ=.1,ɛ=0.1) => F: $(minimum(result)), C: $(Evolutionary.iterations(result))")
        test_result(result, N, 1e-1)
    end
end
