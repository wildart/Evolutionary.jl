# Mutation operators
# ==================

# Isotropic mutation operator y' := y + σ(N_1(0, 1), ..., N_N(0, 1))
function isotropic(recombinant::T, s::S) where {T <: AbstractVector, S <: Strategy}
    vals = randn(length(recombinant)) * s[:σ]
    recombinant += vals
    return recombinant
end

# Anisotropic mutation operator y' := y + σ(N_1(0, 1), ..., N_N(0, 1))
function anisotropic(recombinant::T, s::S) where {T <: AbstractVector, S <: Strategy}
    @assert length(s[:σ]) == length(recombinant) "Sigma parameters must be defined for every dimension of objective parameter"
    vals = randn(length(recombinant)) .* s[:σ]
    recombinant += vals
    return recombinant
end


# Strategy mutation operators
# ===========================

# Isotropic strategy mutation σ' := σ exp(τ N(0, 1))
function isotropicSigma(s::S) where {S <: Strategy}
    @assert :σ ∈ keys(s) && :τ ∈ keys(s) "Strategy must have parameters: σ, τ"
    return strategy(σ = s[:σ] * exp(s[:τ]*randn()), τ = s[:τ])
end

# Anisotropic strategy mutation σ' := exp(τ0 N_0(0, 1))(σ_1 exp(τ N_1(0, 1)), ..., σ_N exp(τ N_N(0, 1)))
function anisotropicSigma(s::S) where {S <: Strategy}
    @assert :σ ∈ keys(s) && :τ ∈ keys(s) && :τ0 ∈ keys(s) "Strategy must have parameters: σ, τ0, τ"
    @assert isa(s[:σ], Vector) "Sigma must be a vector of parameters"
    σ = exp.(s[:τ0]*randn())*exp.(s[:τ]*randn(length(s[:σ])))
    # σ = exp.(s[:τ]*randn(length(s[:σ])))
    return strategy(σ = σ, τ = s[:τ], τ0 = s[:τ0])
end


# Genetic mutations
# =================

"""
    flip(recombinant)

Returns a binary `recombinant` with a bit flips at random positions.
"""
function flip(recombinant::T) where {T <: BitVector}
    s = length(recombinant)
    pos = rand(1:s)
    recombinant[pos] = !recombinant[pos]
    return recombinant
end

"""
    bitinversion(recombinant)

Returns a binary `recombinant` with its bits inverted.
"""
bitinversion(recombinant::T) where {T <: BitVector} = map(!, recombinant)


"""
    domainrange(valrange, m = 20)

Returns a real valued mutation function with the mutation range `valrange` and the mutation probability `1/m` [^1].
"""
function domainrange(valrange::Vector, m::Int = 20)
    prob = 1.0 / m
    function mutation(recombinant::T) where {T <: AbstractVector}
        d = length(recombinant)
        @assert length(valrange) == d "Range matrix must have $(d) columns"
        δ = zeros(m)
        for i in 1:length(recombinant)
            for j in 1:m
                δ[j] = (rand() < prob) ? δ[j] = 2.0^(-j) : 0.0
            end
            if rand() > 0.5
                recombinant[i] += sum(δ)*valrange[i]
            else
                recombinant[i] -= sum(δ)*valrange[i]
            end
        end
        return recombinant
    end
    return mutation
end


# Combinatorial mutations (applicable to binary vectors)
# ------------------------------------------------------
function inversion(recombinant::T) where {T <: AbstractVector}
    l = length(recombinant)
    from, to = rand(1:l, 2)
    from, to = from > to ? (to, from)  : (from, to)
    l = round(Int,(to - from)/2)
    for i in 0:(l-1)
        swap!(recombinant, from+i, to-i)
    end
    return recombinant
end

function insertion(recombinant::T) where {T <: AbstractVector}
    l = length(recombinant)
    from, to = rand(1:l, 2)
    val = recombinant[from]
    deleteat!(recombinant, from)
    return insert!(recombinant, to, val)
end

function swap2(recombinant::T) where {T <: AbstractVector}
    l = length(recombinant)
    from, to = rand(1:l, 2)
    swap!(recombinant, from, to)
    return recombinant
end

function scramble(recombinant::T) where {T <: AbstractVector}
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

function shifting(recombinant::T) where {T <: AbstractVector}
    l = length(recombinant)
    from, to, where = sort(rand(1:l, 3))
    patch = recombinant[from:to]
    diff = where - to
    if diff > 0
        # move values after tail of patch to the patch head position
        #println([from, to, where, diff])
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
function swap!(v::T, from::Int, to::Int) where {T <: AbstractVector}
    val = v[from]
    v[from] = v[to]
    v[to] = val
end

function mutationwrapper(gamutation::Function)
    wrapper(recombinant::T, s::S) where {T <: AbstractVector, S <: Strategy} =  gamutation(recombinant)
    return wrapper
end
