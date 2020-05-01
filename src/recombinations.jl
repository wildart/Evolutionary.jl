# Recombinations
# ==============
"""
    average(population)

Returns an offspring of a multi-parent recombination by averaging `population`.
"""
function average(population::Vector{T}) where {T <: AbstractVector}
    obj = zeros(eltype(T), length(population[1]))
    l = length(population)
    for i in 1:l
        obj += population[i]
    end
    return obj./l
end

"""
    marriage(population)

Returns an offspring of a multi-parent recombination by random copying from `population`.
"""
function marriage(population::Vector{T}) where {T <: AbstractVector}
    s = length(population)
    l = length(population[1])
    obj = zeros(eltype(T), l)
    for i in 1:l
        obj[i] = population[rand(1:s)][i]
    end
    return obj
end

# Strategy recombinations
# =======================

"""
    average(ss::Vector{<:AbstractStrategy})

Returns the average value of the mutation parameter ``\\sigma`` of strategies `ss`.
"""
function average(ss::Vector{<:AbstractStrategy})
    s = copy(first(ss))
    l = length(ss)
    s.σ = mapreduce(s->s.σ/l, +, ss)
    return s
end

# ==================
# Genetic algorithms
# ==================

identity(v1::T, v2::T) where {T <: AbstractVector} = (v1,v2)

# Binary crossovers
# -----------------

"""
    singlepoint(v1, v2)

Single point crossover between `v1` and `v2`
"""
function singlepoint(v1::T, v2::T) where {T <: AbstractVector}
    l = length(v1)
    c1 = copy(v1)
    c2 = copy(v2)
    pos = rand(1:l)
    for i in pos:l
        vswap!(c1, c2, i)
    end
    return c1, c2
end

"""
    twopoint(v1, v2)

Two point crossover between `v1` and `v2`
"""
function twopoint(v1::T, v2::T) where {T <: AbstractVector}
    l = length(v1)
    c1 = copy(v1)
    c2 = copy(v2)
    from, to = rand(1:l, 2)
    from, to = from > to ? (to, from)  : (from, to)
    for i in from:to
        vswap!(c1, c2, i)
    end
    return c1, c2
end

"""
    uniform(v1, v2)

Uniform crossover between `v1` and `v2`
"""
function uniform(v1::T, v2::T) where {T <: AbstractVector}
    l = length(v1)
    c1 = copy(v1)
    c2 = copy(v2)
    xch = rand(Bool, l)
    for i in 1:l
        if xch[i]
            vswap!(c1, c2, i)
        end
    end
    return c1, c2
end


# Real valued crossovers
# ----------------------

"""Discrete recombination"""
function discrete(v1::T, v2::T) where {T <: AbstractVector}
    l = length(v1)
    c1 = similar(v1)
    c2 = similar(v2)
    sltc = rand(Bool, 2, l)
    for i in 1:l
        c1[i] = sltc[1,i] ? v1[i] : v2[i]
        c2[i] = sltc[2,i] ? v2[i] : v1[i]
    end
    return c1, c2
end

"""Weighted arithmetic mean recombination"""
function waverage(w::Vector{Float64})
    function wavexvr(v1::T, v2::T) where {T <: AbstractVector}
        c1 = (v1+v2)./w
        return c1, copy(c1)
    end
    return wavexvr
end

"""Intermediate recombination"""
function intermediate(d::Float64 = 0.0)
    function intermxvr(v1::T, v2::T) where {T <: AbstractVector}
        l = length(v1)
        α = (1.0+2d) * rand(l) .- d
        c1 = v2 .+ α .* (v1 - v2)
        α = (1.0+2d) * rand(l) .- d
        c2 = v1 .+ α .* (v2 - v1)
        return c1, c2
    end
    return intermxvr
end

"""Line recombination"""
function line(d::Float64 = 0.0)
    function linexvr(v1::T, v2::T) where {T <: AbstractVector}
        α1, α2 = (1.0+2d) * rand(2) .- d
        c1 = v2 .+ α2 * (v1 - v2)
        c2 = v1 .+ α1 * (v2 - v1)
        return c1, c2
    end
    return linexvr
end


# Permutation crossovers
# ----------------------

"""Partially mapped crossover"""
function pmx(v1::T, v2::T) where {T <: AbstractVector}
    s = length(v1)
    from, to = rand(1:s, 2)
    from, to = from > to ? (to, from)  : (from, to)
    c1 = similar(v1)
    c2 = similar(v2)

    # Swap
    c1[from:to] = v2[from:to]
    c2[from:to] = v1[from:to]

    # Fill in from parents
    for i in vcat(1:from-1, to+1:s)
        # Check conflicting offspring
        in1 = inmap(v1[i], c1, from, to)
        if in1 == 0
            c1[i] = v1[i]
        else
            tmpin = in1
            while tmpin > 0
                tmpin = inmap(c2[in1], c1, from, to)
                in1 = tmpin > 0 ? tmpin : in1
            end
            c1[i] = v1[in1]
        end

        in2 = inmap(v2[i], c2, from, to)
        if in2 == 0
            c2[i] = v2[i]
        else
            tmpin = in2
            while tmpin > 0
                tmpin = inmap(c1[in2], c2, from, to)
                in2 = tmpin > 0 ? tmpin : in2
            end
            c2[i] = v2[in2]
        end
    end
    return c1, c2
end

"""Order crossover"""
function ox1(v1::T, v2::T) where {T <: AbstractVector}
    s = length(v1)
    from, to = rand(1:s, 2)
    from, to = from > to ? (to, from)  : (from, to)
    c1 = zero(v1)
    c2 = zero(v2)
    # Swap
    c1[from:to] = v2[from:to]
    c2[from:to] = v1[from:to]
    # Fill in from parents
    k = to+1 > s ? 1 : to+1 #child1 index
    j = to+1 > s ? 1 : to+1 #child2 index
    for i in vcat(to+1:s,1:from-1)
        while in(v1[k],c1)
            k = k+1 > s ? 1 : k+1
        end
        c1[i] = v1[k]
        while in(v2[j],c2)
            j = j+1 > s ? 1 : j+1
        end
        c2[i] = v2[j]
    end
    return c1, c2
end

"""Cycle crossover"""
function cx(v1::T, v2::T) where {T <: AbstractVector}
    s = length(v1)
    c1 = zero(v1)
    c2 = zero(v2)
    c1i = fill(true, length(v1))
    c2i = fill(true, length(v2))
    Z = zero(eltype(T))

    f1 = true #switch
    k = 1
    while k > 0
        idx = k
        if f1
            #cycle from v1
            while c1i[idx]
                c1[idx] = v1[idx]
                c2[idx] = v2[idx]
                c1i[idx] = false
                c2i[idx] = false
                idx = inmap(v2[idx],v1,1,s)
            end
        else
            #cycle from v2
            while c2i[idx]
                c1[idx] = v2[idx]
                c2[idx] = v1[idx]
                c1i[idx] = false
                c2i[idx] = false
                idx = inmap(v1[idx],v2,1,s)
            end
        end
        f1 ⊻= true
        k = inmap(true,c2i,1,s)
    end
    return c1,c2
end

"""Order-based crossover"""
function ox2(v1::T, v2::T) where {T <: AbstractVector}
    s = length(v1)
    c1 = copy(v1)
    c2 = copy(v2)
    Z = zero(eltype(T))

    for i in 1:s
        if rand(Bool)
            idx1 = inmap(v2[i],v1,1,s)
            idx2 = inmap(v1[i],v2,1,s)
            c1[idx1] = Z
            c2[idx2] = Z
        end
    end

    for i in 1:s
        if !in(v2[i],c1)
            tmpin = inmap(Z,c1,1,s)
            c1[tmpin] = v2[i]
        end
        if !in(v1[i],c2)
            tmpin = inmap(Z,c2,1,s)
            c2[tmpin] = v1[i]
        end
    end
    return c1,c2
end

"""Position-based crossover"""
function pos(v1::T, v2::T) where {T <: AbstractVector}
    s = length(v1)
    c1 = zero(v1)
    c2 = zero(v2)
    Z = zero(eltype(T))

    for i in 1:s
        if rand(Bool)
            c1[i] = v2[i]
            c2[i] = v1[i]
        end
    end

    for i in 1:s
        if !in(v1[i],c1)
            tmpin = inmap(Z,c1,1,s)
            c1[tmpin] = v1[i]
        end
        if !in(v2[i],c2)
            tmpin = inmap(Z,c2,1,s)
            c2[tmpin] = v2[i]
        end
    end
    return c1,c2
end

# Utils
# =====
function vswap!(v1::T, v2::T, idx::Int) where {T <: AbstractVector}
    val = v1[idx]
    v1[idx] = v2[idx]
    v2[idx] = val
end

function inmap(v::T, c::AbstractVector{T}, from::Int, to::Int) where {T}
    exists = 0
    for j in from:to
        if exists == 0 && v == c[j]
            exists = j
        end
    end
    return exists
end
