module Evolutionary
    using Random, LinearAlgebra, Statistics
    import Base.@kwdef
    import UnPack.@unpack
    export Strategy, strategy, inverse, mutationwrapper,
           # ES mutations
           isotropic, anisotropic, isotropicSigma, anisotropicSigma,
           # GA mutations
           flip, domainrange, inversion, insertion, swap2, scramble, shifting,
           # ES recombinations
           average, marriage, averageSigma1, averageSigmaN,
           # GA recombinations
           singlepoint, twopoint, uniform,
           discrete, waverage, intermediate, line,
           pmx, ox1, cx, ox2, pos,
           # GA selections
           ranklinear, rankuniform, roulette, sus, tournament, truncation,
           # Optimization methods
           ES, CMAES, GA,
           es, cmaes, ga

    const Strategy = Dict{Symbol,Any}
    const Individual = Union{AbstractArray, Function, Nothing}

    # Wrapping function for strategy
    function strategy(; kwargs...)
        result = Dict{Symbol,Any}()
        for (k, v) in kwargs
            result[k] = v
        end
        return result
    end

    # Inverse function for reversing optimization direction
    function inverseFunc(f::Function)
        function fitnessFunc(x::T) where {T <: AbstractVector}
            return 1.0/(f(x)+eps())
        end
        return fitnessFunc
    end

    # Obtain individual
    function getIndividual(init::Individual, N::Int)
        if isa(init, Vector)
            @assert length(init) == N "Dimensionality of initial population must be $(N)"
            individual = init
        elseif isa(init, Matrix)
            @assert size(init, 1) == N "Dimensionality of initial population must be $(N)"
            individual = init[:, 1]
        elseif isa(init, Function) # Creation function
            individual = init(N)
        else
            individual = ones(N)
        end
        return  individual
    end

    # Collecting interim values
    function keep(interim, v, vv, col)
        if interim
            if !haskey(col, v)
                col[v] = typeof(vv)[]
            end
            push!(col[v], vv)
        end
    end

    abstract type Optimizer end

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
    @deprecate ga(f, N; initPopulation::Individual=ones(N), lowerBounds=nothing, upperBounds=nothing, populationSize=50, crossoverRate=0.8, mutationRate=0.1, ɛ=0, selection=((x,n)->1:n), crossover=((x,y)->(y,x)), mutation=(x->x), iterations=100*N, tol=0.0, tolIter=10, verbose=false, debug=false, interim=false) Evolutionary.optimize(f, GA(N=N, initPopulation=initPopulation, lowerBounds=lowerBounds, upperBounds=upperBounds, populationSize=populationSize, crossoverRate=crossoverRate, mutationRate=mutationRate, ɛ=ɛ, selection=selection, crossover=crossover, mutation=mutation), iterations=iterations, tol=tol, tolIter=tolIter, interim=interim, verbose=verbose, debug=debug)
    @deprecate es(f, N; iterations=N*100, initPopulation::Individual=ones(N), initStrategy::Strategy=strategy(), recombination=(rs->rs[1]), srecombination= (ss->ss[1]), mutation=((r,m)->r), smutation=(s->s), termination=(x->false), μ=1, ρ=μ, λ=1, selection=:plus, interim=false, verbose=false, debug=false) Evolutionary.optimize(f, ES(N=N, initPopulation=initPopulation, initStrategy=initStrategy, recombination=recombination, srecombination=srecombination, mutation=mutation, smutation=smutation, termination=termination, μ=μ, ρ=ρ, λ=λ, selection=selection), iterations=iterations, interim=interim, verbose=verbose, debug=debug)
    @deprecate cmaes(f,N; initPopulation::Individual=ones(N), initStrategy=strategy(τ=sqrt(N), τ_c=N^2, τ_σ=sqrt(N)), μ=1, λ=1, iterations=1000, tol=1e-10, verbose=false) Evolutionary.optimize(f, CMAES(N=N, initPopulation=initPopulation, initStrategy=initStrategy, μ=μ, λ=λ), iterations=iterations, tol=tol, verbose=verbose)
end
