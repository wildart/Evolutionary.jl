# Mutation operators
# ==================

# Evolutionary mutations
# ======================

"""
    gaussian(x, s::IsotropicStrategy)

Performs Gaussian isotropic mutation of the recombinant `x` given the strategy `s`  by adding Gaussian noise as follows:

- ``x_i^\\prime = x_i + s.\\sigma \\mathcal{N}_i(0,1)``

"""
function gaussian(recombinant::AbstractVector, s::IsotropicStrategy)
    vals = randn(length(recombinant)) * s.σ
    recombinant += vals
    return recombinant
end

"""
    gaussian(x, s::AnisotropicStrategy)

Performs Gaussian anisotropic mutation of the recombinant `x` given the strategy `s`  by adding Gaussian noise as follows:

- ``x_i^\\prime = x_i + s.\\sigma_i \\mathcal{N}_i(0,1)``

"""
function gaussian(recombinant::AbstractVector, s::AnisotropicStrategy)
    @assert length(s.σ) == length(recombinant) "Parameter `σ` must be defined for every dimension of objective parameter"
    vals = randn(length(recombinant)) .* s.σ
    recombinant += vals
    return recombinant
end

"""
    cauchy(x, s::IsotropicStrategy)

Performs isotropic mutation of the recombinant `x` given the strategy `s`  by adding a noise from the Cauchy distribution as follows:

- ``x_i^\\prime = x_i + s.\\sigma_i \\delta_i``

where ``\\delta`` is a Cauchy random variable with the scale parameter ``t = 1`` [^2].

"""
function cauchy(recombinant::AbstractVector, s::IsotropicStrategy)
    l = length(recombinant)
    vals = s.σ * randn(l)./(randn(l).+eps())
    recombinant += vals
    return recombinant
end


# Strategy mutation operators
# ===========================

"""
    gaussian(s::IsotropicStrategy)

Performs in-place mutation of the isotropic strategy `s` modifying its mutated strategy parameter ``\\sigma`` with Gaussian noise as follows:

- ``\\sigma^\\prime = \\sigma \\exp(\\tau_0 \\mathcal{N}(0,1))``

"""
function gaussian(s::IsotropicStrategy)
    s.σ *= exp(s.τ₀*randn())
    return s
end


"""
    gaussian(s::AnisotropicStrategy)

Performs in-place mutation of the anisotropic strategy `s` modifying its mutated strategy parameter ``\\sigma`` with Gaussian noise as follows:

- ``\\sigma_i^\\prime = \\sigma_i \\exp(\\tau_0 \\mathcal{N}(0,1) + \\tau_i \\mathcal{N}(0,1))``

"""
function gaussian(s::AnisotropicStrategy)
    s.σ .*= exp.(s.τ₀*randn())*exp.(s.τ*randn(length(s.σ)))
    return s
end


# Genetic mutations
# =================

# Binary mutations
# ---------------------

"""
    flip(recombinant)

Returns an in-place mutated binary `recombinant` with a bit flips at random positions.
"""
function flip(recombinant::T) where {T <: BitVector}
    s = length(recombinant)
    pos = rand(1:s)
    recombinant[pos] = !recombinant[pos]
    return recombinant
end

"""
    bitinversion(recombinant)

Returns an in-place mutated binary `recombinant` with its bits inverted.
"""
bitinversion(recombinant::T) where {T <: BitVector} = map!(!, recombinant, recombinant)


# Real-valued mutations
# ---------------------

"""
    uniform(r = 1.0)

Returns an in-place real valued mutation function that performs the uniform distributed mutation [^1].

The mutation operator randomly chooses a number ``z`` in from the uniform distribution on the interval ``[-r,r]``, the mutation range.
The mutated individual is given by

- ``x_i^\\prime = x_i + z_i``

"""
function uniform(r::Real = 1.0)
    function mutation(recombinant::T) where {T <: AbstractVector}
        d = length(recombinant)
        recombinant += 2r.*rand(d).-r
        return recombinant
    end
    return mutation
end

"""
    gaussian(σ = 1.0)

Returns an in-place real valued mutation function that performs the normal distributed mutation [^1].

The mutation operator randomly chooses a number ``z`` in from the normal distribution ``\\mathcal{N}(0,\\sigma)`` with standard deviation ``\\sigma``.
The mutated individual is given by

- ``x_i^\\prime = x_i + z_i``

"""
function gaussian(σ::Real = 1.0)
    function mutation(recombinant::T) where {T <: AbstractVector}
        d = length(recombinant)
        recombinant += σ.*randn(d)
        return recombinant
    end
    return mutation
end

"""
    domainrange(valrange, m = 20)

Returns an in-place real valued mutation function that performs the BGA mutation scheme with the mutation range `valrange` and the mutation probability `1/m` [^1].
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

"""
    PM(lower, upper, p = 2)

Returns an in-place real valued mutation function that performs the Power Mutation (PM) scheme within `lower` and `upper` bound, and an index of
mutation `p`[^3].

*Note:* The implementation is a degenerate case of Mixed Integer Power Mutation ([`MIPM`](@ref))
"""
function PM(lower::Vector, upper::Vector, p::Float64 = 5.0) # index of distribution p
    return mipmmutation(lower, upper, p)
end

"""
    MIPM(lower, upper, p_real = 10, p_int = 4)

Returns an in-place real valued mutation function that performs the Mixed Integer Power Mutation (MI-PM) scheme within `lower` and `upper` bound, and an index of mutation `p_real` for real value and `p_int` for integer values[^4].
"""
function MIPM(lowerBounds::Vector, upperBounds::Vector, p_real::Float64 = 10.0, p_int::Float64 = 4.0) # index of distribution p
    return mipmmutation(lowerBounds, upperBounds, p_real, p_int)
end

function mipmmutation(lowerBounds::Vector, upperBounds::Vector, p_real::Float64, p_int::Union{Nothing, Float64} = nothing)
    function mutation(recombinant::T) where {T <: Vector}
        d = length(recombinant)
        @assert length(lowerBounds) == d "Bounds vector must have $(d) columns"
        @assert length(upperBounds) == d "Bounds vector must have $(d) columns"
        @assert p_int != nothing || all(!isa(x, Integer) for x in recombinant) "Need to set p_int for integer variables"
        u = rand()
        P = (isa(x, Integer) ? p_int : p_real for x in recombinant)
        S = u .^ (1 ./ P) # random var following power distribution
        D = (recombinant - lowerBounds) ./ (upperBounds - lowerBounds)
        broadcast!((x, l, u, s, d) -> d < rand() ? x - s * (x - l) : x + s * (u - x), recombinant, recombinant, lowerBounds, upperBounds, S, D)
        return recombinant
    end
    return mutation
end

# Combinatorial mutations (applicable to binary vectors)
# ------------------------------------------------------

"""
    inversion(recombinant)

Returns an in-place mutated individual with a random arbitrary length segment of the genome in the reverse order.
"""
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

"""
    swap2(recombinant)

Returns an in-place mutated individual with a two random elements of the genome are swapped.
"""
function swap2(recombinant::T) where {T <: AbstractVector}
    l = length(recombinant)
    from, to = rand(1:l, 2)
    swap!(recombinant, from, to)
    return recombinant
end

"""
    scramble(recombinant)

Returns an in-place mutated individual with elements, on a random arbitrary length segment of the genome, been scrambled.
"""
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

"""
    shifting(recombinant)

Returns an in-place mutated individual with a random arbitrary length segment of the genome been shifted to an arbitrary position.
"""
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

# Differential Evolution
# ======================

"""
    differentiation(recombinant, mutators; F = 1.0)

Returns an in-place differentially mutated individual ``x^\\prime`` from `recombinant` ``x``  by `mutators` ``\\{\\xi_1, \\ldots, \\xi_n \\}`` as follows

- ``x^\\prime = x + \\sum_{i=1}^{n/2} F (\\xi_{2i-1} - \\xi_{2i})``

"""
function differentiation(recombinant::T, mutators::AbstractVector{T}; F::Real = 1.0) where {T <: AbstractVector}
    m = length(mutators)
    @assert m%2 == 0 "Must be even number of target mutators"
    for i in 1:2:m
        recombinant .+= F.*(mutators[i] .- mutators[i+1])
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
    wrapper(recombinant::T, s::S) where {T <: AbstractVector, S <: AbstractStrategy} =  gamutation(recombinant)
    return wrapper
end
