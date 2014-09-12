module Evolutionary

    export Strategy, strategy,
           # mutations
           isotropic, anisotropic, isotropicSigma, anisotropicSigma,
           # recombinations
           average, averageSigma1, averageSigmaN,
           es,
           cmaes

    typealias Strategy Dict{Symbol,Any}

    # Wrapping function for strategy
    function strategy(; kwargs...)
        result = Dict{Symbol,Any}()
        for (k, v) in kwargs
            result[k] = v
        end
        return result
    end

    # recombination functions
    include("recombinations.jl")

    # mutation functions
    include("mutations.jl")

    # Evolution Strategy
    include("es.jl")
    include("cmaes.jl")
end
