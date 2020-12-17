module Evolutionary
    using Random, LinearAlgebra, Statistics
    using Base: @kwdef
    using UnPack: @unpack
    using NLSolversBase: AbstractObjective, ConstraintBounds, AbstractConstraints, nconstraints_x, nconstraints

    import NLSolversBase: NonDifferentiable, f_calls, value, value!
    import Base: show, copy, minimum, summary, identity, getproperty, rand, getindex, length, copyto!, setindex!

    export AbstractStrategy, strategy, mutationwrapper,
           IsotropicStrategy, AnisotropicStrategy, NoStrategy,
           isfeasible, BoxConstraints, apply!, penalty, penalty!,
           PenaltyConstraints, WorstFitnessConstraints, MixedTypePenaltyConstraints,
           # ES mutations
           gaussian, cauchy,
           # GA mutations
           flip, bitinversion, domainrange, inversion, insertion, swap2, scramble, shifting, PM, MIPM,
           # ES recombinations
           average, marriage,
           # GA recombinations
           singlepoint, twopoint, uniform,
           discrete, waverage, intermediate, line, HX, LX, MILX,
           PMX, OX1, CX, OX2, POS,
           # GA selections
           ranklinear, uniformranking, roulette, rouletteinv, sus, susinv, tournament, truncation,
           # DE selections
           random, permutation, randomoffset, best,
           # DE recombinations
           uniformbin, exponential,
           # GP exports
           Terminal,
           # Optimization methods
           ES, CMAES, GA, DE, TreeGP,
           # re-export
           NonDifferentiable, value, value!

    # optimize API
    include("api/types.jl")
    include("api/results.jl")
    include("api/utilities.jl")
    include("api/constraints.jl")
    include("api/optimize.jl")
    include("api/expressions.jl")

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

    # Differential Evolution
    include("de.jl")

    # Genetic Programming
    include("gp.jl")

    # deprecations
    @deprecate isotropic(recombinant, strategy) gaussian(recombinant, strategy)
    @deprecate anisotropic(recombinant, strategy) gaussian(recombinant, strategy)
    @deprecate isotropicSigma(strategy) gaussian(strategy)
    @deprecate anisotropicSigma(strategy) gaussian(strategy)
    @deprecate averageSigma(strategy) average(strategy)
    @deprecate strategy(kwargs...) IsotropicStrategy(N) false
end
