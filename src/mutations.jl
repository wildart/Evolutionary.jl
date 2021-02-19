##### mutations.jl #####

# In this file you will several functions that correspond to specific types of mutations.
# Has functions for both Evolution Strategies and Genetic Algorithms.

####################################################################

export mutate

####################################################################

# Mutation operators
# ==================

# Isotropic mutation operator y' := y + σ(N_1(0, 1), ..., N_N(0, 1))
function isotropic(recombinant::T, s::S) where {T <: Vector, S <: Strategy}
    vals = randn(length(recombinant)) * s[:σ]
    recombinant += vals
    return recombinant
end

# Anisotropic mutation operator y' := y + σ(N_1(0, 1), ..., N_N(0, 1))
function anisotropic(recombinant::T, s::S) where {T <: Vector, S <: Strategy}
    @assert length(s[:σ]) == length(recombinant) "Sigma parameters must be defined for every dimension of objective parameter"
    vals = randn(length(recombinant)) .* s[:σ]
    recombinant += vals
    return recombinant
end

####################################################################

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
    #σ = exp(s[:τ0]*randn())*exp(s[:τ]*randn(length(s[:σ])))
    σ = exp.(s[:τ]*randn(length(s[:σ])))
    return strategy(σ = σ, τ = s[:τ], τ0 = s[:τ0])
end

####################################################################

# Genetic mutations
# =================

# Binary mutations
# ----------------
function flip(recombinant ::BitVector)
    s = length(recombinant)
    pos = rand(1:s)
    @inbounds recombinant[pos] = !recombinant[pos]
    return recombinant
end

function singleflip(recombinant ::Bool)
    if rand() > 0.5
        recombinant = !recombinant
    end
    return recombinant
end

# Real valued mutation
# Mühlenbein, H. and Schlierkamp-Voosen, D.: Predictive Models for the Breeder Genetic Algorithm: I. Continuous Parameter Optimization. Evolutionary Computation, 1 (1), pp. 25-49, 1993.
# --------------------
function domainrange(valrange:: Vector{Float64}, m ::Int = 20)
    prob = 1.0 / m
    function mutation(recombinant ::Vector{Float64})
        d = length(recombinant)
        @assert length(valrange) == d "Range matrix must have $(d) columns"
        δ = zeros(m)
        for i in 1:length(recombinant)
            for j in 1:m
                @inbounds δ[j] = (rand() < prob) ? 2.0^(-j) : 0.0
            end
            if rand() > 0.5
                @inbounds recombinant[i] += sum(δ)*valrange[i]
            else
                @inbounds recombinant[i] -= sum(δ)*valrange[i]
            end
        end
        return recombinant
    end
    return mutation
end

# Combinatorial mutations (applicable to binary vectors)
# ------------------------------------------------------
function inversion(recombinant ::BitVector)
    l = length(recombinant)
    from, to = rand(1:l, 2)
    from, to = from > to ? (to, from)  : (from, to)
    l = round(Int,(to - from)/2)
    for i in 0:(l-1)
        swap!(recombinant, from+i, to-i)
    end
    return recombinant
end

function insertion(recombinant ::BitVector)
    l = length(recombinant)
    from, to = rand(1:l, 2)
    val = recombinant[from]
    deleteat!(recombinant, from)
    return insert!(recombinant, to, val)
end

function swap2(recombinant ::BitVector)
    l = length(recombinant)
    from, to = rand(1:l, 2)
    swap!(recombinant, from, to)
    return recombinant
end

function scramble(recombinant ::BitVector)
    l = length(recombinant)
    from, to = rand(1:l, 2)
    from, to = from > to ? (to, from)  : (from, to)
    diff = to - from + 1
    if diff > 1
        @inbounds patch = recombinant[from:to]
        idx = randperm(diff)
        # println("$(from)-$(to), P: $(patch), I: $(idx)")
        for i in 1:(diff)
            @inbounds recombinant[from+i-1] = patch[idx[i]]
        end
    end
    return recombinant
end

function shifting(recombinant ::BitVector)
    l = length(recombinant)
    from, to, where = sort(rand(1:l, 3))
    @inbounds patch = recombinant[from:to]
    diff = where - to
    if diff > 0
        # move values after tail of patch to the patch head position
        #println([from, to, where, diff])
        for i in 1:diff
            @inbounds recombinant[from+i-1] = recombinant[to+i]
        end
        # place patch values in order
        start = from + diff
        for i in 1:length(patch)
            @inbounds recombinant[start+i-1] = patch[i]
        end
    end
    return recombinant
end

# Utils
# =====
function swap!(v ::T, from ::Int, to ::Int) where {T <: Vector}
    @inbounds begin
        val = v[from]
        v[from] = v[to]
        v[to] = val
    end
end

function mutationwrapper(gamutation ::Function)
    wrapper(recombinant::T, s::S) where {T <: Vector, S <: Strategy} =
        gamutation(recombinant) 
    return wrapper
end

####################################################################

# This function serves only to choose the mutation function for binary functions. The actual `mutate`
# function is created in the `IntegerGene` structure.
# FM   - Flip Mutation
# InvM - Inversion Mutation
# InsM - Insertion Mutation
# SwM  - Swap Mutation
# ScrM - Scramble Mutation
# ShM  - Shifting Mutation

"""
    mutate(gene ::IntegerGene)

Mutates `gene` according to the mutation function chosen in the `IntegerGene` structure.
"""
function mutate(mutatetype ::Symbol)
    mut_func = nothing
    if mutatetype == :FM
        mut_func = flip
    elseif mutatetype == :InvM
        mut_func = inversion
    elseif mutatetype == :InsM
        mut_func = insertion
    elseif mutatetype == :SwM
        mut_func = swap2
    elseif mutatetype == :ScrM
        mut_func = scramble
    elseif mutatetype == :ShM
        mut_func = shifting
    else
        error("Unknown mutation type")
    end
    return mut_func
end

"""
    mutate(gene ::BinaryGene)

Mutates `gene` using the Single Flip Mutation.
"""
function mutate(gene ::BinaryGene)
    gene.value = singleflip(gene.value)
    return nothing
end

"""
    mutate(gene ::FloatGene)

Mutates `gene` using Real Valued Mutation.
"""
function mutate(gene ::FloatGene)
    prob = 1.0 / gene.m
    δ = zeros(gene.m)
    
    function mutation_help(value ::Float64, range ::Float64)
        for j in 1:gene.m
            δ[j] = (rand() < prob) ? 2.0^(-j) : 0.0
        end
        if rand() > 0.5
            value += sum(δ)*range
        else
            value -= sum(δ)*range
        end
        return value
    end

    for i in 1:length(gene.value)
        gene.value[i] = mutation_help(gene.value[i], gene.range[i])
        while !isbound(gene, i)
            gene.value[i] = mutation_help(gene.value[i], gene.range[i])
        end
    end
        
    return nothing
end

"""
    mutate(chromossome ::Vector{<:AbstractGene}, rate ::Float64)

Mutates each entry of `chromossome` according to the mutations chosen.
"""
function mutate(chromossome ::Individual, rate ::Float64)
    for gene in chromossome
        if rand() < rate
            mutate(gene)
        else
            while !isbound(gene)
                mutate(gene)
            end
        end
    end
    return nothing
end
