##### structs.jl #####

# In this file you will find all the structures needed to generalize code.
# You will find also some functions to help create the global structures.

####################################################################

export AbstractGene
export BinaryGene, IntegerGene, FloatGene
export Crossover, Selection
export selection, crossover, bin

####################################################################

"""
Abstract Type to represent all types of genes supported.
"""
abstract type AbstractGene end

####################################################################

"""
    BinaryGene(value ::Bool)

Creates a `BinaryGene` structure. This gene represents a `Bool` variable (1 or 0). Useful to make decisions (will a reactor work or not, will a SMB pathway be used or not, etc.).
"""
mutable struct BinaryGene <: AbstractGene
    value ::Bool
    
    BinaryGene(value ::Bool) = new(value)
end

"""
    BinaryGene()

Creates a `BinaryGene` structure with a random `Bool` value.
"""
BinaryGene() = BinaryGene(rand(Bool))

####################################################################

"""
    IntegerGene(value ::BitVector, mutation ::Symbol)

Creates a `IntegerGene` structure. This gene represents an integer variable as `BitVector`. To convert the `BitVector` in an integer, just look at the `bin` function from this package by typing `?bin` on the command prompt. `mutation` is a symbol that represents the type of mutation used for the bit vector. The below table shows all the mutation types supported:

| Symbol | Algorithm |
|-----|-----|
| :FM | Flip Mutation |
| :InvM | Inversion Mutation |
| :InsM | Insertion Mutation |
| :SwM | Swap Mutation |
| :ScrM | Scramble Mutation |
| :ShM | Shifting Mutation |
"""
mutable struct IntegerGene <: AbstractGene
    value    ::BitVector
    mutation ::Symbol
    
    function IntegerGene(value ::BitVector, mutation ::Symbol)
        int_func = mutate(mutation)
        @eval mutate(gene ::IntegerGene) = $int_func(gene.value)
        return new(value, mutation)
    end
end

"""
    IntegerGene(value ::BitVector)

Creates a `IntegerGene` structure with the default mutation being Flip Mutation.
"""
function IntegerGene(value ::BitVector) 
    return IntegerGene(value, :FM)
end

"""
    IntegerGene(n ::Int64)

Creates a `IntegerGene` structure in which the `BitVector` is of length `n` with random `Bool` values.
"""
function IntegerGene(n ::Int64)
    value = BitVector(undef, n)
    for i in 1:n
        value[i] = rand(Bool)
    end
    return IntegerGene(value, :FM)
end

"""
    bin(gene ::IntegerGene)

Returns the integer number represented by the `BitVector` of `gene`.
"""
function bin(gene ::IntegerGene)
    bin_val = 0
    for (i,j) in enumerate(gene.value)
        bin_val += j ? 2^(i-1) : 0
    end
    return bin_val
end

####################################################################

"""
    FloatGene(value ::Vector{Float64}, range ::Vector{Float64}, m ::Int64)

Creates a `FloatGene` structure. `value` is a vector with the variables to be changed. `range` is a vector with the minimum and maximum values a variable can take. `m` is just a parameter that changes how much in a mutation the variables change, the bigger the value, the bigger the change in each mutation. If the range of a variable is 0.5, then the biggest mutation a variable can suffer is 0.5 for instance.
"""
mutable struct FloatGene <: AbstractGene
    value ::Vector{Float64}
    range ::Vector{Float64}
    m     ::Int64
    
    function FloatGene( value ::Vector{Float64} ,
                        range ::Vector{Float64} ,
                        m     ::Int64           )
        if length(value) != length(range)
            error("vectors must have the same length")
        end
        return new(value, range, m)
    end
end

"""
    FloatGene(value ::Float64, range ::Float64; m ::Int64 = 20)

Creates a `FloatGene` structure. Handy for creating just one real number variable.
"""
function FloatGene(value ::Float64, range ::Float64; m ::Int64 = 20)
    return FloatGene(Float64[value], Float64[range], m)
end

"""
    FloatGene(value ::Vector{Float64}, range ::Float64; m ::Int64 = 20)

Creates a `FloatGene` structure. Handy for creating a vector of real numbers with the same range.
"""
function FloatGene(value ::Vector{Float64}, range ::Float64; m ::Int64 = 20)
    vec = Float64[range for i in value]
    return FloatGene(value, vec, m)
end

"""
    FloatGene(value ::Vector{Float64}; m ::Int64 = 20)

Creates a `FloatGene` structure. Handy for creating a vector of real numbers with a random range.
"""
function FloatGene(value ::Vector{Float64}; m ::Int64 = 20)
    range = rand(Float64, length(value))
    return FloatGene(value, range, m)
end

"""
    FloatGene(value ::Float64; m ::Int64 = 20)

Creates a `FloatGene` structure. Handy for creating one variable with a random range.
"""
function FloatGene(value ::Float64; m ::Int64 = 20)
    return FloatGene(value, rand(); m=m)
end

"""
    FloatGene(n ::Int64)

Creates a `FloatGene` structure. Creates a vector of length `n` with random variables and random ranges. Used particularly for testing purposes.
"""
function FloatGene(n ::Int64)
    value = rand(Float64, n)
    range = rand(Float64, n)
    return FloatGene(value, range, 20)
end

####################################################################

# :SPX - Single Point Crossover               - singlepoint
# :TPX - Two Point Crossover                  - twopoint
# :UX  - Uniform Crossover                    - uniform
# :DX  - Discrete Crossover                   - discrete
# :WMX - Weighted Mean Crossover              - waverage(w ::Vector{Float64})
# :IRX - Intermediate Recombination Crossover - intermediate(d ::Float64)
# :LRX - Line Recombination Crossover         - line(d ::Float64)
# :PMX - Partially Mapped Crossover           - pmx
# :O1X - Order 1 Crossover                    - ox1
# :O2X - Order 2 Crossover                    - ox2
# :CX  - Cycle Crossover                      - cx
# :PX  - Position-based Crossover             - pos

"""
    Crossover(cross ::Symbol                                    ;
              w     ::Union{Nothing, Vector{Float64}} = nothing ,
              d     ::Union{Nothing, Float64        } = nothing )

Creates a `Crossover` structure. `cross` is a Symbol that represents the type of crossover that would be used. `w` and `d` are not mandatory but need to be set for some types of crossovers. All algorithms will be shown in the table below:

| Symbol | Algorithm | Optional Arguments |
|----|----|---|
| :SPX | Single Point Crossover | not needed |
| :TPX | Two Point Crossover | not needed |
| :UX | Uniform Crossover | not needed |
| :DX | Discrete Crossover | not needed |
| :WMX | Weighted Mean Crossover | needs `w` |
| :IRX | Intermediate Recombination Crossover | needs `d` |
| :LRX | Line Recombination Crossover | needs `d` |
| :PMX | Partially Mapped Crossover | not needed |
| :O1X | Order 1 Crossover | not needed |
| :O2X | Order 2 Crossover | not needed |
| :CX | Cycle Crossover | not needed |
| :PX | Position-based Crossover | not needed |
"""
mutable struct Crossover
    cross ::Symbol
    w     ::Union{Nothing, Vector{Float64}}
    d     ::Union{Nothing, Float64        }
    
    function Crossover(cross ::Symbol                                    ;
                       w     ::Union{Nothing, Vector{Float64}} = nothing ,
                       d     ::Union{Nothing, Float64        } = nothing )
            cross_func = nothing
            if cross == :SPX
                cross_func = singlepoint
            elseif cross == :TPX
                cross_func = twopoint
            elseif cross == :UX
                cross_func = uniform
            elseif cross == :DX
                cross_func = discrete
            elseif cross == :WMX
                if isnothing(w)
                    error("value `w` must be given a value")
                end
                cross_func = waverage(w)
            elseif cross == :IRX
                if isnothing(d)
                    error("value `d` must be given a value")
                end
                cross_func = intermediate(d)
            elseif cross == :LRX
                if isnothing(d)
                    error("value `d` must be given a value")
                end
                cross_func = line(d)
            elseif cross == :PMX
                cross_func = pmx
            elseif cross == :O1X
                cross_func = ox1
            elseif cross == :O2X
                cross_func = ox2
            elseif cross == :CX
                cross_func = cx
            elseif cross == :PX
                cross_func = pos
            end
        
            @eval begin
                function crossover(v1::T, v2 ::T) where {T <: AbstractVector}    
                    return $cross_func(v1, v2)
                end
            end
        return new(cross, w, d)
    end
end

####################################################################

# :RBS  - Rank-Based Selection
# :URS  - Uniform-Ranking Selection
# :RWS  - Roulette Wheel Selection
# :SUSS - Stochastic Universal Sampling Selection
# :TrS  - Truncation Selection
# :ToS  - Tournament Selection

"""
    Selection( select    ::Symbol                            ;
               sp        ::Union{Nothing, Float64} = nothing ,
               μ         ::Union{Nothing,   Int64} = nothing ,
               groupsize ::Union{Nothing,   Int64} = nothing )

Creates a `Selection` structure. `select` is a symbol that represents the type of selection that will be used. `sp`, `μ` and `groupsize` are optional but need to be set for some types of selections. All algorithms will be shown in the table below:

| Symbol | Algorithm | Optional Arguments |
|----|----|----|
| :RBS | Rank-based Selection | needs `sp` |
| :URS | Uniform-Ranking Selection | needs `μ` |
| :RWS | Roulette Wheel Selection | not needed |
| :SUSS | Stochastic Universal Sampling Selection | not needed |
| :TrS | Truncation Selection | not needed |
| :ToS | Tournament Selection | needs `groupsize` |
"""
mutable struct Selection
    select ::Symbol
    sp     ::Union{Nothing, Float64}
    μ      ::Union{Nothing,   Int64}
    gsize  ::Union{Nothing,   Int64}
    
    function Selection( select    ::Symbol                            ;
                        sp        ::Union{Nothing, Float64} = nothing ,
                        μ         ::Union{Nothing,   Int64} = nothing ,
                        groupsize ::Union{Nothing,   Int64} = nothing )
        selec_func = nothing
        if select == :RBS
            if isnothing(sp)
                error("need to specify `sp` value")
            end
            selec_func = ranklinear(sp)
        elseif select == :URS
            if isnothing(μ)
                error("need to specify `μ` value")
            end
            selec_func = uniformranking(μ)
        elseif select == :RWS
            selec_func = roulette
        elseif select == :SUSS
            selec_func = sus
        elseif select == :TrS
            selec_func = truncation
        elseif select == :ToS
            if isnothing(groupsize)
                error("need to specify `groupsize` value")
            end
            selec_func = tournament(groupsize)
        else
            error("Unknown parameter " * string(select))
        end
        
        @eval begin
            function selection(fit ::Vector{<:Real}, N ::Int)
                return $selec_func(fit, N)
            end
        end
        return new(select, sp, μ, groupsize)
    end
end
