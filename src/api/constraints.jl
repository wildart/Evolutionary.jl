# Public interface for AbstractConstraints

"""
    value(c::AbstractConstraints, f::AbstractObjective, x)

Return a value of the function `f` for a variable `x` given the set of constraints `c`.
"""
value(c::AbstractConstraints, f::AbstractObjective, x) = value(f, x)

"""
    value(c::AbstractConstraints, x)

Return a value of a variable with respect to the constraints `c`.
"""
value(c::AbstractConstraints, x) = x

# Auxillary functions

"""
    isfeasible(constraints, x) -> Bool
    isfeasible(bounds, x) -> Bool

Return `true` if point `x` is feasible, given the `constraints` which
specify bounds `lx`, `ux`. `x` is feasible if

    lx[i] <= x[i] <= ux[i]

    for all possible `i`.
"""
function isfeasible(bounds::ConstraintBounds, x)
    isf = true
    for (i,j) in enumerate(bounds.eqx)
        isf &= x[j] == bounds.valx[i]
    end
    for (i,j) in enumerate(bounds.ineqx)
        isf &= bounds.σx[i]*(x[j] - bounds.bx[i]) >= 0
    end
    isf
end
isfeasible(constraints::AbstractConstraints, x) = isfeasible(constraints.bounds, x)

function clip!(bounds::ConstraintBounds, x)
    for (i,j) in enumerate(bounds.eqx)
        x[j] = bounds.valx[i]
    end
    for (i,j) in enumerate(bounds.ineqx)
        if bounds.σx[i]*(x[j] - bounds.bx[i]) < 0
            x[j] = bounds.bx[i]
        end
    end
    x
end

# Implementations

"""Type for an empty set of constratins"""
struct NoConstraints <: AbstractConstraints end
isfeasible(constraints::NoConstraints, x)  = true

"""Box constraints"""
struct BoxConstraints{T} <: AbstractConstraints
    bounds::ConstraintBounds{T}
end
BoxConstraints(lower::AbstractVector{T}, upper::AbstractVector{T}) where {T} =
    BoxConstraints(ConstraintBounds(lower, upper, [], []))
BoxConstraints(lower::T, upper::T, size) where {T} =
    BoxConstraints(fill(lower, size), fill(upper, size))
value(constraints::BoxConstraints, x) = clip!(constraints.bounds, x)

"""
Transfinite encoding of bounds as a smooth (surjective) function `h : Rⁿ → [lᵢ , uᵢ ], e.g.

h : xᵢ → lᵢ + (uᵢ − lᵢ)/2 · (1 + tanh(xᵢ))

and optimize the composite function g(x) = f(h(x))
"""
struct TransfiniteConstraints{T} <: AbstractConstraints
    bounds::ConstraintBounds{T}
end
TransfiniteConstraints(lower::AbstractVector{T}, upper::AbstractVector{T}) where {T} =
    TransfiniteConstraints(ConstraintBounds(lower, upper, [], []))
TransfiniteConstraints(lower::T, upper::T, size) where {T} =
    TransfiniteConstraints(fill(lower, size), fill(upper, size))
function value(constraints::TransfiniteConstraints{T}, x) where {T}
    bounds = constraints.bounds
    y = zeros(T, size(x))
    for (i,j) in enumerate(bounds.eqx)
        y[j] = bounds.valx[i]
    end
    l, u = -Inf, Inf
    for (i,j) in enumerate(bounds.ineqx)
        if isinf(l) && bounds.σx[i] > 0
            l = bounds.bx[i]
            continue
        end
        if isinf(u) && bounds.σx[i] < 0
            u = bounds.bx[i]
            y[j] = l + (u-l)*(1+tanh(x[j]))/2
            l, u = -Inf, Inf
        end
    end
    y
end


"""
Penalty constraints type encodes constraints as an additional penalty for an objective function.
"""
struct PenaltyConstraints{T} <: AbstractConstraints
    penalty::T
    bounds::ConstraintBounds{T}
end
PenaltyConstraints(penalty::T, lower::AbstractVector{T}, upper::AbstractVector{T}) where {T} =
    PenaltyConstraints(penalty, ConstraintBounds(lower, upper, [], []))
