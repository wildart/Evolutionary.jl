@testset "Sphere" begin

    # Objective function
    sphere(x::AbstractVector) = sum(x.*x)

    # Parameters
    N = 30
    P = 25
    initial = ones(N)

    function Evolutionary.trace!(record::Dict{String,Any}, objfun, state, population, method::ES, options)
        record["fitpop"] = state.fitness
        record["σ"] = strategy(state).σ
    end

    function Evolutionary.terminate(state::Evolutionary.ESState)
        strategy(state).σ < 1e-10
    end

    # Testing: (μ/μ_I, λ)-σ-Self-Adaptation-ES
    # with isotropic mutation operator y' := y + σ(N_1(0, 1), ..., N_N(0, 1))
    result = Evolutionary.optimize(
        sphere,
        initial,
        ES(
            initStrategy = IsotropicStrategy(N),
            recombination = average, srecombination = average,
            mutation = gaussian, smutation = gaussian,
            selection=:comma,
            μ = 3, λ = P
        ), Evolutionary.Options(show_trace=false,iterations=1000));
    # show(result)
    println("(3/3,$(P))-σ-SA-ES => F: $(minimum(result)), C: $(Evolutionary.iterations(result))")
    @test minimum(result) ≈ 0.0 atol=1e-3
    @test sum(x->x.^2, Evolutionary.minimizer(result)) ≈ 0.0 atol=1e-3
    @test length(Evolutionary.minimizer(result)) == N

    # disable custom functions
    Evolutionary.terminate(state::Evolutionary.ESState) = false
    Evolutionary.trace!(record::Dict{String,Any}, objfun, state, population, method::ES, options) = ()

    # Testing: GA
    result = Evolutionary.optimize(
        sphere,
        initial,
        GA(
            populationSize = 4P,
            mutationRate = 0.15,
            ɛ = 0.1,
            selection = susinv,
            crossover = intermediate(0.25),
            mutation = domainrange(fill(0.5,N)),
        ));
    # show(result)
    println("GA:INTER:DOMRNG:(N=$(N), P=$(P)) => F: $(minimum(result)), C: $(Evolutionary.iterations(result))")
    @test minimum(result) ≈ 0.0 atol=1e-3
    @test sum(x->x.^2, Evolutionary.minimizer(result)) ≈ 0.0 atol=1e-3
    @test length(Evolutionary.minimizer(result)) == N

end
