module Evolutionary
    using Random, LinearAlgebra, Statistics
    using Parameters: @unpack, @with_kw
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
           ranklinear, uniformranking, roulette, sus, tournament, truncation,
           # Optimization methods
           optimize, ES, CMAES, GA,
           es, cmaes, ga

    const Strategy = Dict{Symbol,Any}
    const Individual = Union{Vector, Matrix, Function, Nothing}

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
        function fitnessFunc(x::T) where {T <: Vector}
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

    ga(f,N;iterations::Integer = 100*N,tol = 0.0,verbose = false,debug = false,kwargs...) = optimize(f,GA(N=N,kwargs...), iterations = iterations, tol = tol, verbose = verbose, debug = debug)
    es(f,N;iterations::Integer = N*100,verbose = false, debug = false, kwargs...) = optimize(f, ES(N=N,kwargs...), iterations = iterations, verbose = verbose, debug = debug)
    cmaes(f,N;iterations::Integer = 1_000,tol::Float64 = 1e-10,verbose = false) = optimize(f, CAMES(N=N,kwargs...), iterations = iterations, tol = tol, verbose = verbose)
end
