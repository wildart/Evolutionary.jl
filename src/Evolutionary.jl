module Evolutionary

    export Strategy, strategy, inverse,
           # ES mutations
           isotropic, anisotropic, isotropicSigma, anisotropicSigma,
           # GA mutations
           flip, domainrange, inversion, insertion, swap2, scramble, shifting,
           # ES recombinations
           average, marriage, averageSigma1, averageSigmaN,
           # GA recombinations
           singlepoint, twopoint, uniform,
           discrete, waverage, intermediate, line,
           pmx, #ox1, cx, ox2, pos
           # GA selections
           ranklinear, uniformranking, roulette, sus, #truncation, tournament
           # Optimization methods
           es, cmaes, ga

    typealias Strategy Dict{Symbol,Any}

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
      function fitnessFunc{T <: Vector}(x::T)
        return 1.0/(f(x)+eps())
      end
      return fitnessFunc
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
