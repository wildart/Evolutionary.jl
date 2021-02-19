
module Evolutionary

using Random, Base.Threads
using Distributed
using DistributedArrays, DistributedArrays.SPMD
using Printf
using Dates

export
    # Optimization methods
    es, cmaes, ga,
    # Constants
    GAVector, Individual, AbstractGene

####################################################################

"""
Abstract Type that represents all types of genes supported.
"""
abstract type AbstractGene end

const Strategy = Dict{Symbol,Any}

const Individual = Vector{AbstractGene}

const GAVector = Union{T, BitVector} where T <: Vector

####################################################################

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
        populationSize = size(init, 2)
        individual = init[:, 1]
    elseif isa(init, Function) # Creation function
        individual = init(N)
    else
        individual = ones(N)
    end
    return individual
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

# General Structures
include("structs.jl")

# ES & GA recombination functions
include("recombinations.jl")

# ES & GA mutation functions
include("mutations.jl")

# GA selection functions
include("selections.jl")

# Evolution Strategy
include("es.jl")
include("cmaes.jl")

# Backup functions
include("backup.jl")

# Genetic Algorithms
include("ga.jl")

end
