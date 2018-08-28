# Recombinations
# ==============
function average(population::Vector{T}) where {T <: Vector}
    obj = zeros(eltype(T), length(population[1]))
    l = length(population)
    for i in 1:l
        obj += population[i]
    end
    return obj./l
end

function marriage(population::Vector{T}) where {T <: Vector}
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
function averageSigma1(ss::Vector{S}) where {S <: Strategy}
    s = copy(ss[1])
    σ = 0.0
    l = length(ss)
    for i in 1:l
        σ += ss[i][:σ]
    end
    s[:σ] = σ/l
    return s
end

function averageSigmaN(ss::Vector{S}) where {S <: Strategy}
    s = copy(ss[1])
    σ = zeros(length(ss[1][:σ]))
    l = length(ss)
    for i in 1:l
        σ += ss[i][:σ]
    end
    s[:σ] = σ./l
    return s
end

# ==================
# Genetic algorithms
# ==================

# Binary crossovers
# -----------------

# Single point crossover
function singlepoint(v1::T, v2::T) where {T <: Vector}
    l = length(v1)
    c1 = copy(v1)
    c2 = copy(v2)
    pos = rand(1:l)
    for i in pos:l
        vswap!(c1, c2, i)
    end
    return c1, c2
end

# Two point crossover
function twopoint(v1::T, v2::T) where {T <: Vector}
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

# Uniform crossover
function uniform(v1::T, v2::T) where {T <: Vector}
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

function discrete(v1::T, v2::T) where {T <: Vector}
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

# Weighted arithmetic mean
function waverage(w::Vector{Float64})
    function wavexvr(v1::T, v2::T) where {T <: Vector}
        c1 = (v1+v2)./w
        return c1, copy(c1)
    end
    return wavexvr
end

# Intermediate recombination
function intermediate(d::Float64 = 0.0)
    function intermxvr(v1::T, v2::T) where {T <: Vector}
        l = length(v1)
        α = (1.0+2d) * rand(l) .- d
        c1 = v2 .+ α .* (v1 - v2)
        α = (1.0+2d) * rand(l) .- d
        c2 = v1 .+ α .* (v2 - v1)
        return c1, c2
    end
    return intermxvr
end

# Line recombination
function line(d::Float64 = 0.0)
    function linexvr(v1::T, v2::T) where {T <: Vector}
        α1, α2 = (1.0+2d) * rand(2) .- d
        c1 = v2 .+ α2 * (v1 - v2)
        c2 = v1 .+ α1 * (v2 - v1)
        return c1, c2
    end
    return linexvr
end


# Permutation crossovers
# ----------------------

# Partially mapped crossover
function pmx(v1::T, v2::T) where {T <: Vector}
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

# Order crossover
function ox1(v1::T, v2::T) where {T <: Vector}
    s = length(v1)
    from, to = rand(1:s, 2)
    from, to = from > to ? (to, from)  : (from, to)
    c1 = zeros(v1)
    c2 = zeros(v2)
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

# Cycle crossover
function cx(v1::T, v2::T) where {T <: Vector}
    s = length(v1)
    c1 = zeros(v1)
    c2 = zeros(v2)

    f1 = true #switch
    k = 1
    while k > 0
        idx = k
        if f1
            #cycle from v1
            while c1[idx] == zero(T)
            c1[idx] = v1[idx]
            c2[idx] = v2[idx]
            idx = inmap(v2[idx],v1,1,s)
            end
        else
            #cycle from v2
            while c2[idx] == zero(T)
            c1[idx] = v2[idx]
            c2[idx] = v1[idx]
            idx = inmap(v1[idx],v2,1,s)
            end
        end
        f1 $= true
        k = inmap(zero(T),c2,1,s)
    end
    return c1,c2
end

# Order-based crossover
function ox2(v1::T, v2::T) where {T <: Vector}
    s = length(v1)
    c1 = copy(v1)
    c2 = copy(v2)

    for i in 1:s
        if rand(Bool)
            idx1 = inmap(v2[i],v1,1,s)
            idx2 = inmap(v1[i],v2,1,s)
            c1[idx1] = zero(T)
            c2[idx2] = zero(T)
        end
    end

    for i in 1:s
        if !in(v2[i],c1)
            tmpin = inmap(zero(T),c1,1,s)
            c1[tmpin] = v2[i]
        end
        if !in(v1[i],c2)
            tmpin = inmap(zero(T),c2,1,s)
            c2[tmpin] = v1[i]
        end
    end
    return c1,c2
end

# Position-based crossover
function pos(v1::T, v2::T) where {T <: Vector}
    s = length(v1)
    c1 = zeros(v1)
    c2 = zeros(v2)

    for i in 1:s
        if rand(Bool)
            c1[i] = v2[i]
            c2[i] = v1[i]
        end
    end

    for i in 1:s
        if !in(v1[i],c1)
            tmpin = inmap(zero(T),c1,1,s)
            c1[tmpin] = v1[i]
        end
        if !in(v2[i],c2)
            tmpin = inmap(zero(T),c2,1,s)
            c2[tmpin] = v2[i]
        end
    end
    return c1,c2
end

# Utils
# =====
function vswap!(v1::T, v2::T, idx::Int) where {T <: Vector}
    val = v1[idx]
    v1[idx] = v2[idx]
    v2[idx] = val
end

function inmap(v::T, c::Vector{T}, from::Int, to::Int) where{T}
    exists = 0
    for j in from:to
        if exists == 0 && v == c[j]
            exists = j
        end
    end
    return exists
end
