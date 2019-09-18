module Evolutionary
using Random
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
           ranklinear, rankuniform, roulette, sus, tournament, #truncation
           # Optimization methods
           es, cmaes, ga

    const Strategy = Dict{Symbol,Any}

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
        function fitnessFunc(x)
            return 1.0/(f(x)+eps())
        end
        return fitnessFunc
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
