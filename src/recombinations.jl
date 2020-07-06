# Recombinations
# ==============
"""
    average(population)

Returns an *one* offspring individual of a multi-parent recombination by averaging `population`.
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

Returns an *one* offspring individual of a multi-parent recombination by random copying from `population`.
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

"""
    identity(v1, v2)

Returns the same parameter individuals `v1` and `v2` as an offspring pair.
"""
identity(v1::T, v2::T) where {T <: AbstractVector} = (v1,v2)

# Binary crossovers
# -----------------

"""
    singlepoint(v1, v2)

Single point crossover between `v1` and `v2` individuals.
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

Two point crossover between `v1` and `v2` individuals.
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

Uniform crossover between `v1` and `v2` individuals.
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

"""
    uniformbin(Cr::Real=0.5)

Returns a uniform (binomial) crossover function, see [Recombination Interface](@ref), function with the propbabilty `Cr` [^2].

The crossover probability value must be in unit interval, ``Cr \\in [0,1]``.
"""
function uniformbin(Cr::Real = 0.5)
    function binxvr(v1::T, v2::T) where {T <: AbstractVector}
        l = length(v1)
        c1 = copy(v1)
        c2 = copy(v2)
        j = rand(1:l)
        for i in (((1:l).+j.-2).%l).+1
            if rand() <= Cr
                vswap!(c1, c2, i)
            end
        end
        return c1, c2
    end
    return binxvr
end

"""
    exponential(Cr::Real=0.5)

Returns an exponential crossover function, see [Recombination Interface](@ref), function with the propbabilty `Cr` [^2].

The crossover probability value must be in unit interval, ``Cr \\in [0,1]``.
"""
function exponential(Cr::Real = 0.5)
    function expxvr(v1::T, v2::T) where {T <: AbstractVector}
        l = length(v1)
        c1 = copy(v1)
        c2 = copy(v2)
        j = rand(1:l)
        switch = true
        for i in (((1:l).+j.-2).%l).+1
            i == j && continue
            if switch && rand() <= Cr
                c1[i] = v1[i]
            else
                switch = false
                c1[i] = v2[i]
            end
        end
        return c1, copy(c1)
    end
    return expxvr
end


# Real valued crossovers
# ----------------------

"""
    discrete(v1, v2)

Returs a randomly assembled offspring and its inverse from the elements of parents `v1` and `v2`.
"""
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

"""
    intermediate(d::Real=0.0)

Returns an extended intermediate recombination operation, see [Recombination Interface](@ref), which generates offspring `u` and `v` as

- ``u_i = x_i + \\alpha_i (y_i - x_i)``
- ``v_i = y_i + \\alpha_i (x_i - y_i)``

where ``\\alpha_i`` is chosen uniform randomly in the interval ``[-d;d+1]``.

"""
function intermediate(d::Real = 0.0)
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

"""
    line(d::Real=0.0)

Returns a extended line recombination operation, see [Recombination Interface](@ref), which generates offspring `u` and `v` as

- ``u_i = x_i + \\alpha (y_i - x_i)``
- ``v_i = y_i + \\alpha (x_i - y_i)``

where ``\\alpha`` is chosen uniform randomly in the interval ``[-d;d+1]``.

"""
function line(d::Real = 0.0)
    function linexvr(v1::T, v2::T) where {T <: AbstractVector}
        α1, α2 = (1.0+2d) * rand(2) .- d
        c1 = v2 .+ α2 * (v1 - v2)
        c2 = v1 .+ α1 * (v2 - v1)
        return c1, c2
    end
    return linexvr
end

"""
    HX(x, y)

Heuristic crossover (HX) recombination operation[^3] generates offspring `u` and `v` as

- ``u = x + r (x - y)``
- ``v = y + r (y - x)``

where ``r`` is chosen uniform randomly in the interval ``[0;1)``.
"""
function HX(v1::T, v2::T) where {T <: Vector}
    c1 = v1 .+ rand()*(v1 .- v2)
    c2 = v2 .+ rand()*(v2 .- v1)
    return c1, c2
end

"""
    LX(μ::Real = 0.0, b::Real = 0.2)

Returns a Laplace crossover (LX) recombination operation[^4], see [Recombination Interface](@ref).
"""
function LX(μ::Real = 0.0, b::Real = 0.2) # location μ, scale b > 0
    function lxxvr(v1::T, v2::T) where {T <: Vector}
        u = rand()
        β = u > 0.5 ? μ + b * log(u) : μ - b * log(u)
        S = β * abs.(v1 - v2)
        c1 = v1 + S
        c2 = v2 + S
        return c1, c2
    end
    return lxxvr
end

"""
    MILX(μ::Real = 0.0, b_real::Real = 0.15, b_int::Real = 0.35)

Returns a mixed integer Laplace crossover (MI-LX) recombination operation[^5], see [Recombination Interface](@ref).
"""
function MILX(μ::Real = 0.0, b_real::Real = 0.15, b_int::Real = 0.35) # location μ, scale b > 0
    function milxxvr(v1::T, v2::T) where {T <: Vector}
        @assert all([typeof(a) == typeof(b) for (a, b) in zip(v1, v2)]) "Types of variables in vectors do not match"
        l = length(v1)
        U, R = rand(l), rand(l)
        B = (isa(x, Integer) ? b_int : b_real for x in v1)
        βs = broadcast((u, r, b) -> r > 0.5 ? μ + b * log.(u) : μ - b * log.(u), U, R, B)
        S = βs .* abs.(v1 - v2)
        c1 = v1 + S
        c2 = v2 + S
        return c1, c2
    end
    return milxxvr
end


# Permutation crossovers
# ----------------------

"""Partially mapped crossover"""
function PMX(v1::T, v2::T) where {T <: AbstractVector}
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
function OX1(v1::T, v2::T) where {T <: AbstractVector}
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
function CX(v1::T, v2::T) where {T <: AbstractVector}
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
function OX2(v1::T, v2::T) where {T <: AbstractVector}
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
function POS(v1::T, v2::T) where {T <: AbstractVector}
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
