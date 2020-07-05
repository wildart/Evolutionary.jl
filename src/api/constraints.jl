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

Only accepts bounds and equality constraints.

    lx[i] <= x[i] <= ux[i]
    lc[i] <= c(x)[i] <= uc[i]
"""
struct PenaltyConstraints{T,F} <: AbstractConstraints
    coef::Vector{T}
    c!::F   # c!(x) returns the value of the constraint-functions at x
    bounds::ConstraintBounds{T}
    PenaltyConstraints(penalty::Vector{T}, bounds::ConstraintBounds{T}, c::F) where {T,F} =
        new{T,F}(penalty, c, bounds)
end
PenaltyConstraints(penalty::T, bounds::ConstraintBounds{T}, cf=(x)->nothing) where {T} =
    PenaltyConstraints(fill(penalty, nconstraints(bounds)+nconstraints_x(bounds)), bounds, cf)
PenaltyConstraints(penalty::T, lx::AbstractVector{T}, ux::AbstractVector{T},
                   lc::AbstractVector{T}=T[], uc::AbstractVector{T}=T[],
                   cf=(x)->nothing) where {T} =
    PenaltyConstraints(penalty, ConstraintBounds(lx, ux, lc, uc), cf)
function value(constraints::PenaltyConstraints{T}, f::AbstractObjective, x) where {T}
    bounds = constraints.bounds
    coef = constraints.coef
    xc = nconstraints_x(bounds)
    penalty = zero(T)
    for (i,j) in enumerate(bounds.eqx)
        penalty += coef[j]*(x[j] - bounds.valx[i])^2
    end
    for (i,j) in enumerate(bounds.ineqx)
        penalty += coef[j]*max(zero(T), bounds.σx[i]*(bounds.bx[i]-x[j]))^2
    end
    c = constraints.c!(x)
    for (i,j) in enumerate(bounds.eqc)
        penalty += coef[xc+j]*(c[j] - bounds.valc[i])^2
    end
    for (i,j) in enumerate(bounds.ineqc)
        penalty += coef[xc+j]*max(zero(T), bounds.σc[i]*(bounds.bc[i]-c[j]))^2
    end
    value(f, x) + penalty
end
