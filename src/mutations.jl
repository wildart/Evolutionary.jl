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
function flip(recombinant::T) where {T <: AbstractVector{Bool}}
    s = length(recombinant)
    pos = rand(1:s)
    recombinant[pos] = !recombinant[pos]
    return recombinant
end

"""
    bitinversion(recombinant)

Returns an in-place mutated binary `recombinant` with its bits inverted.
"""
bitinversion(recombinant::T) where {T <: AbstractVector{Bool}} = map!(!, recombinant, recombinant)


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
        recombinant .+= 2r.*rand(d).-r
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
        recombinant .+= σ.*randn(d)
        return recombinant
    end
    return mutation
end

"""
    BGA(valrange, m = 20)

Returns an in-place real valued mutation function that performs the BGA mutation scheme with the mutation range `valrange` and the mutation probability `1/m` [^1].
"""
function BGA(valrange::Vector, m::Int = 20)
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
    function pm_mutation(x, l, u, s, d)
        x̄ = d < rand() ? x - s * (x - l) : x + s * (u - x)
        if isa(x, Integer)
            if isinteger(x̄)
                Int(x̄)
            else
                floor(Int, x̄) + (rand() > 0.5)
            end
        else
            x̄
        end
    end
    function mutation(recombinant::T) where {T <: Vector}
        d = length(recombinant)
        @assert length(lowerBounds) == d "Bounds vector must have $(d) columns"
        @assert length(upperBounds) == d "Bounds vector must have $(d) columns"
        @assert !(p_int === nothing && any(isa(x, Integer) for x in recombinant)) "Need to set p_int for integer variables"
        u = rand()
        P = (isa(x, Integer) ? p_int : p_real for x in recombinant)
        S = u .^ (1 ./ P) # random var following power distribution
        D = (recombinant - lowerBounds) ./ (upperBounds - lowerBounds)
        broadcast!(pm_mutation, recombinant, recombinant, lowerBounds, upperBounds, S, D)
        return recombinant
    end
    return mutation
end

"""
    PLM(lower, upper, η = 2)

Returns an in-place real valued mutation function that performs the Polynomial Mutation (PLM) scheme
within `lower` and `upper` bounds, and a mutation distribution index `η`[^9].
"""
function PLM(Δ::Union{Real, Vector}=1.0; η=2, pm::Real=NaN) # index of distribution p
    function mutation(recombinant::T; rng::AbstractRNG=Random.GLOBAL_RNG) where {T <: AbstractVector}
        d = length(recombinant)
        pm = isnan(pm) ? 1/d : pm
        mask = rand(rng, d) .< pm
        u = rand(rng, d)
        mask_p = u .<= 0.5
        mask_n = u .> 0.5
        δpw = 1/(η+1)
        δ = similar(recombinant)
        δ[mask_p] .= (2* u[mask_p]).^δpw .- 1
        δ[mask_n] .= 1 .- (2(1 .- u[mask_n])).^δpw
        recombinant[mask] .+= (Δ.*δ)[mask]
        return recombinant
    end
    return mutation
end
PLM(lower::Vector, upper::Vector; η::Real = 2, pm::Real=NaN) = PLM(upper-lower; η=η, pm=pm)


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

"""
    insertion(recombinant)

Returns an in-place mutated individual with an arbitrary element of the genome moved in a random position.
"""
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
        for i in 1:diff
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

# Genetic Programming
# ======================

"""
    subtree(method::TreeGP; growth::Real = 0.1)

Returns an in-place expression mutation function that performs mutation of an arbitrary expression subtree with a randomly generated one [^5].

Parameters:
- `growth`: Growth restriction on the offspring in comparison to the parent.
            The offspring cannot be more than `growth`% deeper than its parent.  (default: `0.0`)
"""
function subtree(method::TreeGP; growth::Real = 0.0)
    function mutation(recombinant::Expr)
        i = rand(1:nodes(recombinant)-1)
        th = depth(recombinant, recombinant[i])
        maxh = if growth > 0
            rh = height(recombinant)
            mh = max(method.maxdepth, rand(rh:rh*(1+growth)))
            round(Int, mh)
        else
            method.maxdepth
        end
        recombinant[i] = rand(method, max(0, maxh-th))
        recombinant
    end
    return mutation
end

"""
    point(method::TreeGP)

Returns an in-place expression mutation function that replaces an arbitrary node in the tree by the randomly selected one.
Node replacement mutation is similar to bit string mutation in that it randomly changes a point in the individual.
To ensure the tree remains legal, the replacement node has the same number of arguments as the node it is replacing [^6].
"""
function point(method::TreeGP)
    function mutation(recombinant::Expr)
        i = rand(0:nodes(recombinant)-1)
        nd = recombinant[i]
        if isa(nd, Expr)
            aty = length(nd.args)-1
            atyfnc = filter(kv -> kv[2] == aty, method.functions)
            if length(atyfnc) > 0
                nd.args[1] = atyfnc |> keys |> rand
            end
        else
            recombinant[i] = randterm(method)
        end
        recombinant
    end
    return mutation
end

"""
    hoist(method::TreeGP)

Returns an in-place expression mutation function that creates a new offspring individual which is copy
of a randomly chosen subtree of the parent. Thus, the offspring will be smaller than the parent
and will have a different root node [^7].
"""
function hoist(method::TreeGP)
    function mutation(recombinant::Expr)
        rnodes = nodes(recombinant)
        stsize = 0
        ch = recombinant
        while stsize < 2 && rnodes > 3
            i = rand(1:rnodes-1)
            ch = recombinant[i]
            stsize = length(ch)
        end
        copyto!(recombinant, ch)
    end
    return mutation
end

"""
    shrink(method::TreeGP)

Returns an in-place expression mutation function that replaces a randomly chosen subtree with a randomly
created terminal. This is a special case of subtree mutation where the replacement tree is a terminal.
As with hoist mutation, it is motivated by the desire to reduce program size [^8].
"""
function shrink(method::TreeGP)
    function mutation(recombinant::Expr)
        i = rand(1:nodes(recombinant)-1)
        recombinant[i] = randterm(method)
        recombinant
    end
    return mutation
end


# Utilities
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
