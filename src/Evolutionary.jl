module Evolutionary
    using Random, LinearAlgebra, Statistics
    import NLSolversBase: AbstractObjective, NonDifferentiable, value, value!
    import Base: length, push!, show, @kwdef, minimum, copy, identity
    import UnPack.@unpack
    import NLSolversBase: NonDifferentiable, f_calls
    export AbstractStrategy, strategy, mutationwrapper,
           IsotropicStrategy, AnisotropicStrategy, NoStrategy,
           # ES mutations
           gaussian, cauchy,
           # GA mutations
           flip, bitinversion, domainrange, inversion, insertion, swap2, scramble, shifting,
           # ES recombinations
           average, marriage,
           # GA recombinations
           singlepoint, twopoint, uniform,
           discrete, waverage, intermediate, line,
           pmx, ox1, cx, ox2, pos,
           # GA selections
           ranklinear, uniformranking, roulette, rouletteinv, sus, susinv, tournament, truncation,
           # Optimization methods
           ES, CMAES, GA,
           es, cmaes, ga,
           NonDifferentiable

    # optimize API
    include("api/types.jl")
    include("api/results.jl")
    include("api/utilities.jl")
    include("api/optimize.jl")

    # ES & GA recombination functions
    include("recombinations.jl")

    # ES & GA mutation functions
    include("mutations.jl")

    # GA selection functions
    include("selections.jl")

    # Evolution Strategy
    include("es.jl")
    include("cmaes.jl")

    # Genetic Algorithms
    include("ga.jl")

    # deprecations
    @deprecate ga(f, N; initPopulation::Individual=ones(N), populationSize=50, crossoverRate=0.8, mutationRate=0.1, ɛ=0, selection=((x,n)->1:n), crossover=((x,y)->(y,x)), mutation=(x->x), iterations=100*N, tol=1e-10, tolIter=10, verbose=false, debug=false, interim=false) Evolutionary.optimize(f, initPopulation, GA(populationSize=populationSize, crossoverRate=crossoverRate, mutationRate=mutationRate, ɛ=ɛ, selection=selection, crossover=crossover, mutation=mutation), Evolutionary.Options(iterations=iterations,abstol=tol,successive_f_tol=tolIter,store_trace=interim,show_trace=verbose))
    @deprecate es(f, N; iterations=N*100, initPopulation::Individual=ones(N), initStrategy=NoStrategy(), recombination=(rs->rs[1]), srecombination= (ss->ss[1]), mutation=((r,m)->r), smutation=(s->s), μ=1, ρ=μ, λ=1, selection=:plus, tol=1e-10, tolIter=10, interim=false, verbose=false, debug=false) Evolutionary.optimize(f, initPopulation, ES(initStrategy=initStrategy, recombination=recombination, srecombination=srecombination, mutation=mutation, smutation=smutation, μ=μ, ρ=ρ, λ=λ, selection=selection), Evolutionary.Options(iterations=iterations,abstol=tol,successive_f_tol=tolIter,store_trace=interim, show_trace=verbose))
    @deprecate cmaes(f,N; initPopulation::Individual=ones(N), τ=sqrt(N), τ_c=N^2, τ_σ=sqrt(N), μ=1, λ=1, iterations=1000, tol=1e-10, verbose=false) Evolutionary.optimize(f, initPopulation, CMAES(μ=μ, λ=λ, τ=τ, τ_c=τ_c, τ_σ=τ_σ), Evolutionary.Options(iterations=iterations, abstol=tol, show_trace=verbose))
    @deprecate isotropic(recombinant, strategy) gaussian(recombinant, strategy)
    @deprecate anisotropic(recombinant, strategy) gaussian(recombinant, strategy)
    @deprecate isotropicSigma(strategy) gaussian(strategy)
    @deprecate anisotropicSigma(strategy) gaussian(strategy)
    @deprecate averageSigma(strategy) average(strategy)
    @deprecate strategy(kwargs...) IsotropicStrategy(N) false
end
