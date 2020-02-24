
export BinaryGene, IntegerGene, FloatGene
export Crossover, Selection, Chromossome

####################################################################

abstract type AbstractGene end

####################################################################

mutable struct BinaryGene <: AbstractGene
    value ::Bool
    
    BinaryGene(value ::Bool) = new(value)
end

function BinaryGene()
    return BinaryGene(rand(Bool))
end

####################################################################

mutable struct IntegerGene <: AbstractGene
    value    ::BitVector
    mutation ::Symbol
    
    function IntegerGene(value ::BitVector, mutation ::Symbol)
        int_func = int_mutate(mutation)
        @eval mutate(gene ::IntegerGene) = $int_func(gene.value)
        return new(value, mutation)
    end
end

function IntegerGene(value ::BitVector) 
    return IntegerGene(value, :FM)
end

function IntegerGene(n ::Int64)
    value = BitVector(undef, n)
    return IntegerGene(value, :FM)
end

####################################################################

mutable struct FloatGene <: AbstractGene
    value ::Vector{Float64}
    range ::Vector{Float64}
    m     ::Int64
    
    function FloatGene(value ::Vector{Float64} ,
                       range ::Vector{Float64} ,
                       m     ::Int64           )
        if length(value) != length(range)
            error("vectors mush have the same length")
        end
        return new(value, range, m)
    end
end

function FloatGene(value ::Float64, range ::Float64; m ::Int64 = 20)
    return FloatGene(Float64[value], Float64[range], m)
end

function FloatGene(value ::Vector{Float64}, range ::Float64; m ::Int64 = 20)
    vec = Float64[range for i in value]
    return FloatGene(value, vec, m)
end

function FloatGene(value ::Vector{Float64}; m ::Int64 = 20)
    range = rand(Float64, length(value))
    return FloatGene(value, range, m)
end

function FloatGene(value ::Float64; m ::Int64 = 20)
    return FloatGene(value, rand(); m=m)
end

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
mutable struct Crossover
    cross ::Symbol
    func  ::Function
    w     ::Union{Nothing, Vector{Float64}}
    d     ::Union{Nothing, Float64        }
    
    function Crossover(cross ::Symbol                                    ;
                       w     ::Union{Nothing, Vector{Float64}} = nothing ,
                       d     ::Union{Nothing, Float64        } = nothing )
        function crossover_func( cross ::Symbol                           ;
                                 w     ::Union{Nothing        ,
                                               Vector{Float64}} = nothing ,
                                 d     ::Union{Nothing        ,
                                               Float64        } = nothing )
            if cross == :SPX
                return singlepoint
            elseif cross == :TPX
                return twopoint
            elseif cross == :UX
                return uniform
            elseif cross == :DX
                return discrete
            elseif cross == :WMX
                if isnothing(w)
                    error("value `w` must be given a value")
                end
                return waverage(w)
            elseif cross == :IRX
                if isnothing(d)
                    error("value `d` must be given a value")
                end
                return intermediate(d)
            elseif cross == :LRX
                if isnothing(d)
                    error("value `d` must be given a value")
                end
                return line(d)
            elseif cross == :PMX
                return pmx
            elseif cross == :O1X
                return ox1
            elseif cross == :O2X
                return ox2
            elseif cross == :CX
                return cx
            elseif cross == :PX
                return pos
            end
            return nothing
        end
        func = crossover_func(cross; w=w, d=d)
        return new(cross, func, w, d)
    end
end

####################################################################

# :RBS  - Rank-Based Selection
# :URS  - Uniform-Ranking Selection
# :RWS  - Roulette Wheel Selection
# :SUSS - Stochastic Universal Sampling Selection
# :TrS  - Truncation Selection
# :ToS  - Tournament Selection
mutable struct Selection
    select ::Symbol
    func   ::Function
    sp     ::Union{Nothing, Float64}
    μ      ::Union{Nothing, Int64  }
    gsize  ::Union{Nothing, Int64  }
    
    function Selection( select    ::Symbol                            ;
                        sp        ::Union{Nothing, Float64} = nothing ,
                        μ         ::Union{Nothing, Int64  } = nothing ,
                        groupsize ::Union{Nothing, Int64  } = nothing )
        function selection( select    ::Symbol                            ;
                            sp        ::Union{Nothing, Float64} = nothing ,
                            μ         ::Union{Nothing, Int64  } = nothing ,
                            groupsize ::Union{Nothing, Int64  } = nothing )
            if select == :RBS
                if isnothing(sp)
                    error("need to specify `sp` value")
                end
                return ranklinear(sp)
            elseif select == :URS
                if isnothing(μ)
                    error("need to specify `μ` value")
                end
                return uniformranking(μ)
            elseif select == :RWS
                return roulette
            elseif select == :SUSS
                return sus
            elseif select == :TrS
                return truncation
            elseif select == :ToS
                if isnothing(groupsize)
                    error("need to specify `groupsize` value")
                end
                return tournament(groupsize)
            else
                error("Unknown parameter " * string(select))
            end
        end
        func = selection(select; sp=sp, μ=μ, groupsize=groupsize)
        return new(select, func, sp, μ, groupsize)
    end
end

####################################################################

mutable struct Chromossome
    n           ::Int64
    chromossome ::Vector{<:AbstractGene}
    
    function Chromossome(chromossome ::Vector{<:AbstractGene})
        n = length(chromossome)        
        return new(n, chromossome, cross, select)
    end
end

function Chromossome(gene ::AbstractGene)
    chromossome = AbstractGene[gene]
    return Chromossome(chromossome)
end
