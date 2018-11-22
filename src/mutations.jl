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


# Genetic mutations
# =================

# Binary mutations
# ----------------
function flip(recombinant::Vector{Bool})
    s = length(recombinant)
    pos = rand(1:s)
    recombinant[pos] = !recombinant[pos]
    return recombinant
end

# Real valued mutation
# Mühlenbein, H. and Schlierkamp-Voosen, D.: Predictive Models for the Breeder Genetic Algorithm: I. Continuous Parameter Optimization. Evolutionary Computation, 1 (1), pp. 25-49, 1993.
# --------------------
function domainrange(valrange::Vector, m::Int = 20)
    prob = 1.0 / m
    function mutation(recombinant::T) where {T <: Vector}
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

function mipmmutation(lowerBounds::Vector, upperBounds::Vector, p_real::Float64, p_int::Union{Nothing, Float64} = nothing)
    function mutation(recombinant::T) where {T <: Vector}
        d = length(recombinant)
        @assert length(lowerBounds) == d "Bounds vector must have $(d) columns"
        @assert length(upperBounds) == d "Bounds vector must have $(d) columns"
        @assert p_int != nothing || all(!isa(x, Integer) for x in recombinant) "Need to set p_int for integer variables"
        u = rand()
        P = [isa(x, Integer) ? p_int : p_real for x in recombinant]
        S = u .^ (1 ./ P) # random var following power distribution
        D = (recombinant - lowerBounds) ./ (upperBounds - lowerBounds)
        recombinant = broadcast((x, l, u, s, d) -> d < rand() ? x - s * (x - l) : x + s * (u - x), recombinant, lowerBounds, upperBounds, S, D)
        return recombinant
    end
    return mutation
end

# Power Mutation (PM) operator
# K. Deep, M. Thakur, A new mutation operator for real coded genetic algorithms,
# Applied Mathematics and Computation 193 (2007) 211–230
# Note: The implementation is a degenerate case of Mixed Integer Power Mutation
function pm(lowerBounds::Vector, upperBounds::Vector, p::Float64 = 0.25) # index of distribution p
    return mipmmutation(lowerBounds, upperBounds, p)
end

# Mixed Integer Power Mutation (MI-PM) operator
# Kusum Deep, Krishna Pratap Singh, M. L. Kansal, and C. Mohan, A real coded
# genetic algorithm for solving integer and mixed integer optimization problems.
# Appl. Math. Comput. 212 (2009) 505-518
function mipm(lowerBounds::Vector, upperBounds::Vector, p_real::Float64 = 10.0, p_int::Float64 = 4.0) # index of distribution p
    return mipmmutation(lowerBounds, upperBounds, p_real, p_int)
end

# Combinatorial mutations (applicable to binary vectors)
# ------------------------------------------------------
function inversion(recombinant::T) where {T <: Vector}
    l = length(recombinant)
    from, to = rand(1:l, 2)
    from, to = from > to ? (to, from)  : (from, to)
    l = round(Int,(to - from)/2)
    for i in 0:(l-1)
        swap!(recombinant, from+i, to-i)
    end
    return recombinant
end

function insertion(recombinant::T) where {T <: Vector}
    l = length(recombinant)
    from, to = rand(1:l, 2)
    val = recombinant[from]
    deleteat!(recombinant, from)
    return insert!(recombinant, to, val)
end

function swap2(recombinant::T) where {T <: Vector}
    l = length(recombinant)
    from, to = rand(1:l, 2)
    swap!(recombinant, from, to)
    return recombinant
end

function scramble(recombinant::T) where {T <: Vector}
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

function shifting(recombinant::T) where {T <: Vector}
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
function swap!(v::T, from::Int, to::Int) where {T <: Vector}
    val = v[from]
    v[from] = v[to]
    v[to] = val
end

function mutationwrapper(gamutation::Function)
    wrapper(recombinant::T, s::S) where {T <: Vector, S <: Strategy} =  gamutation(recombinant) 
    return wrapper
end
