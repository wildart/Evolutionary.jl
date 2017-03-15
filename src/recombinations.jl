# Recombinations
# ==============
function average{T <: Vector}(population::Vector{T})
    obj = zeros(eltype(T), length(population[1]))
    l = length(population)
    for i in 1:l
        obj += population[i]
    end
    return obj./l
end

function marriage{T <: Vector}(population::Vector{T})
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
function averageSigma1{S <: Strategy}(ss::Vector{S})
    s = copy(ss[1])
    σ = 0.0
    l = length(ss)
    for i in 1:l
        σ += ss[i][:σ]
    end
    s[:σ] = σ/l
    return s
end

function averageSigmaN{S <: Strategy}(ss::Vector{S})
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
function singlepoint{T <: Vector}(v1::T, v2::T)
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
function twopoint{T <: Vector}(v1::T, v2::T)
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
function uniform{T <: Vector}(v1::T, v2::T)
    l = length(v1)
    c1 = copy(v1)
    c2 = copy(v2)
    xch = randbool(l)
    for i in 1:l
        if xch[i]
            vswap!(c1, c2, i)
        end
    end
    return c1, c2
end


# Real valued crossovers
# ----------------------

function discrete{T <: Vector}(v1::T, v2::T)
    l = length(v1)
    c1 = similar(v1)
    c2 = similar(v2)
    sltc = randbool(2,l)
    for i in 1:l
        c1[i] = sltc[1,i] ? v1[i] : v2[i]
        c2[i] = sltc[2,i] ? v2[i] : v1[i]
    end
    return c1, c2
end

# Weighted arithmetic mean
function waverage(w::Vector{Float64})
    function wavexvr{T <: Vector}(v1::T, v2::T)
        c1 = (v1+v2)./w
        return c1, copy(c1)
    end
    return wavexvr
end

# Intermediate recombination
function intermediate(d::Float64 = 0.0)
    function intermxvr{T <: Vector}(v1::T, v2::T)
        l = length(v1)
        α = (1.0+2d)*rand(l)-d
        c1 = v2 .+ α .* (v1 - v2)
        α = (1.0+2d)*rand(l)-d
        c2 = v1 .+ α .* (v2 - v1)
        return c1, c2
    end
    return intermxvr
end

# Line recombination
function line(d::Float64 = 0.0)
    function linexvr{T <: Vector}(v1::T, v2::T)
        α1, α2 = (1.0+2d)*rand(2)-d
        c1 = v2 .+ α2 * (v1 - v2)
        c2 = v1 .+ α1 * (v2 - v1)
        return c1, c2
    end
    return linexvr
end


# Permutation crossovers
# ----------------------

# Partially mapped crossover
function pmx{T <: Vector}(v1::T, v2::T)
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
function ox1{T <: Vector}(v1::T, v2::T)
    s = length(v1)
    from, to = rand(1:s, 2)
    from, to = from > to ? (to, from)  : (from, to)
    c1 = similar(v1)
    c2 = similar(v2)
    # Swap
    c1[from:to] = v2[from:to]
    c2[from:to] = v1[from:to]
    # Fill in from parents
    return c1, c2
end

# Cycle crossover
function cx{T <: Vector}(v1::T, v2::T)
  s = length(v1)
  c1 = similar(v1)
  c2 = similar(v2)
  f1 = true

  for i in 1:s
    #Establish priority of parents
    d1,d2 = f1 ? (v1,v2) : (v2,v1)
    #Check existence in child1
    in1 = inmap(d1[i],c1,1,s)
    if in1 == 0
      #fill
      c1[i] = d1[i]
      tmpin = inmap(c1[i],d2,1,s)
      #cycle
      while !in(d1[tmpin],c1)
        c1[tmpin] = d1[tmpin]
        tmpin = inmap(c1[tmpin],d2,1,s)
      end
      #reverse priority
      f1 = false
    end
    #Check existence in child2
    in2 = inmap(d2[i],c2,1,s)
    if in2 == 0
      #fill
      c2[i] = d2[i]
      tmpin = inmap(c2[i],d1,1,s)
      #cycle
      while !in(d2[tmpin],c2)
        c2[tmpin] = d2[tmpin]
        tmpin = inmap(c2[tmpin],d1,1,s)
      end
    end
  end
  return c1,c2
end

# Order-based crossover
function ox2{T <: Vector}(v1::T, v2::T)
    # TODO
end

# Position-based crossover
function pos{T <: Vector}(v1::T, v2::T)
    # TODO
end


# Utils
# =====
function vswap!{T <: Vector}(v1::T, v2::T, idx::Int)
    val = v1[idx]
    v1[idx] = v2[idx]
    v2[idx] = val
end

function inmap{T}(v::T, c::Vector{T}, from::Int, to::Int)
    exists = 0
    for j in from:to
        if exists == 0 && v == c[j]
            exists = j
        end
    end
    return exists
end
