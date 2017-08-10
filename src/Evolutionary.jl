module Evolutionary
using Random
    export Strategy, strategy, inverse, mutationwrapper,
           # ES mutations
           isotropic, anisotropic, isotropicSigma, anisotropicSigma,
           # GA mutations
           flip, domainrange, inversion, insertion, swap2, scramble, shifting,
           pm, mipm,
           # ES recombinations
           average, marriage, averageSigma1, averageSigmaN,
           # GA recombinations
           singlepoint, twopoint, uniform,
           discrete, waverage, intermediate, line, lx, milx,
           pmx, ox1, cx, ox2, pos,
           # GA selections
           ranklinear, uniformranking, roulette, sus, tournament, #truncation
           # Optimization methods
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
        function fitnessFunc(x::T, V...) where {T <: Vector}
            return 1.0/(f(x, V...)+eps())
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
            populationSize = size(init, 2)
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

    # Truncate integer values
    # Rules for truncation are taken from:
    # Kusum Deep, Krishna Pratap Singh, M. L. Kansal, and C. Mohan, A real coded
    # genetic algorithm for solving integer and mixed integer optimization
    # problems. Appl. Math. Comput. 212 (2009) 505-518
    function truncfunc(t::Type, o::Real)
        if t <: Integer
            # truncate only if o is not already an integer
            return isa(o, Integer) ? o : (rand() > 0.5 ? trunc(t, o) : trunc(t, o) + 1)
        else
            # no integer -> don't truncate
            return o
        end
    end

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

end
