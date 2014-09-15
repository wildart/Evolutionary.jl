# Mutation operators
# ==================

# Isotropic mutation operator y' := y + σ(N_1(0, 1), ..., N_N(0, 1))
function isotropic{T <: Vector, S <: Strategy}(recombinant::T, s::S)
    vals = randn(length(recombinant)) * s[:σ]
    recombinant += vals
    return recombinant
end

# Anisotropic mutation operator y' := y + σ(N_1(0, 1), ..., N_N(0, 1))
function anisotropic{T <: Vector, S <: Strategy}(recombinant::T, s::S)
    @assert length(s[:σ]) == length(recombinant) "Sigma parameters must be defined for every dimension of objective parameter"
    vals = randn(length(recombinant)) .* s[:σ]
    recombinant += vals
    return recombinant
end


# Strategy mutation operators
# ===========================

# Isotropic strategy mutation σ' := σ exp(τ N(0, 1))
function isotropicSigma{S <: Strategy}(s::S)
    @assert :σ ∈ keys(s) && :τ ∈ keys(s) "Strategy must have parameters: σ, τ"
    return strategy(σ = s[:σ] * exp(s[:τ]*randn()), τ = s[:τ])
end

# Anisotropic strategy mutation σ' := exp(τ0 N_0(0, 1))(σ_1 exp(τ N_1(0, 1)), ..., σ_N exp(τ N_N(0, 1)))
function anisotropicSigma{S <: Strategy}(s::S)
    @assert :σ ∈ keys(s) && :τ ∈ keys(s) && :τ0 ∈ keys(s) "Strategy must have parameters: σ, τ0, τ"
    @assert isa(s[:σ], Vector) "Sigma must be a vector of parameters"
    #σ = exp(s[:τ0]*randn())*exp(s[:τ]*randn(length(s[:σ])))
    σ = exp(s[:τ]*randn(length(s[:σ])))
    return strategy(σ = σ, τ = s[:τ], τ0 = s[:τ0])
end


# Genetic mutations
# =================

# Binary mutations
# ----------------
function flip{T <: BitArray}(recombinant::T)
    s = length(recombinant)
    pos = rand(1:s)
    recombinant[pos] = !recombinant[pos]
    return recombinant
end

# Combinatorial mutations (applicable to binary vectors)
# ------------------------------------------------------

function inversion{T <: Vector}(recombinant::T)
    l = length(recombinant)
    from, to = rand(1:l, 2)
    from, to = from > to ? (to, from)  : (from, to) 
    l = int((to - from)/2)  
    for i in 0:(l-1)
        swap!(recombinant, from+i, to-i)
    end
    return recombinant
end

function insertion{T <: Vector}(recombinant::T)
    l = length(recombinant)
    from, to = rand(1:l, 2)
    val = recombinant[from]
    deleteat!(recombinant, from)
    return insert!(recombinant, to, val)
end

function swap2{T <: Vector}(recombinant::T)
    l = length(recombinant)
    from, to = rand(1:l, 2)
    swap!(recombinant, from, to)
    return recombinant
end

function scramble{T <: Vector}(recombinant::T)
    l = length(recombinant)
    from, to = rand(1:l, 2)
    from, to = from > to ? (to, from)  : (from, to)
    diff = to - from + 1
    if diff > 1
        patch = recombinant[from:to]
        idx = randperm(diff)
        # println("$(from)-$(to), P: $(patch), I: $(idx)")
        for i in 1:(diff)
            recombinant[from+i-1] = patch[idx[i]]
        end
    end
    return recombinant
end

function shifting{T <: Vector}(recombinant::T)
    l = length(recombinant)
    from, to, where = sort(rand(1:l, 3))    
    patch = recombinant[from:to]
    diff = where - to
    if diff > 0
        # move values after tail of patch to the patch head position
        println([from, to, where, diff])
        for i in 1:diff
            recombinant[from+i-1] = recombinant[to+i]
        end
        # place patch values in order
        start = from + diff
        for i in 1:length(patch)
            recombinant[start+i-1] = patch[i]
        end
    end
    return recombinant
end

# Utils
# =====
function swap!{T <: Vector}(v::T, from::Int, to::Int)
    val = v[from]
    v[from] = v[to]
    v[to] = val
end

function mutationwrapper(gamutation::Function)
    wrapper{T <: Vector, S <: Strategy}(recombinant::T, s::S) = gamutation(recombinant)
    return wrapper
end