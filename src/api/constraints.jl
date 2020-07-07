# Public interface for AbstractConstraints

"""
    constraints(c::AbstractConstraints, f::AbstractObjective, x)

Return a values of constraints for a variable `x` given the set of constraints object `c`.
"""
value(c::AbstractConstraints, x) = nothing

"""
    penalty(constraints, x)

Calculate a penalty for the variable `x` given the set of `constraints`.
"""
penalty(c::AbstractConstraints, x) = 0

"""
    penalty!(fitness, constraints, population)

Set a penalty to the `fitness` given the set of `constraints` and `population`.
"""
penalty!(fitness, c::AbstractConstraints, population) = fitness

"""
    clip!(c::AbstractConstraints, x)

Clip a variable `x` values to satisfy the bound constraints `c`.
"""
apply!(c::AbstractConstraints, x) = x

# Auxillary functions

"""
    isfeasible(constraints, x) -> Bool
    isfeasible(bounds, x) -> Bool

Return `true` if point `x` is feasible, given the `constraints` which
specify bounds `lx`, `ux`, `lc`, and `uc`. `x` is feasible if

    lx[i] <= x[i] <= ux[i]
    lc[i] <= c(x)[i] <= uc[i]

for all possible `i`.
"""
function isfeasible(bounds::ConstraintBounds, x, c::Union{Vector,Nothing}=nothing)
    isf = true
    for (i,j) in enumerate(bounds.eqx)
        isf &= x[j] == bounds.valx[i]
    end
    for (i,j) in enumerate(bounds.ineqx)
        isf &= bounds.σx[i]*(x[j] - bounds.bx[i]) >= 0
    end
    if c !== nothing
        for (i,j) in enumerate(bounds.eqc)
            isf &= c[j] == bounds.valc[i]
        end
        for (i,j) in enumerate(bounds.ineqc)
            isf &= bounds.σc[i]*(c[j] - bounds.bc[i]) >= 0
        end
    end
    isf
end
isfeasible(c::AbstractConstraints, x) = isfeasible(c.bounds, x, value(c, x))

# Implementations

"""Type for an empty set of constratins"""
struct NoConstraints <: AbstractConstraints end
isfeasible(c::NoConstraints, x)  = true
getproperty(c::NoConstraints, s::Symbol) =
    s == :bounds ? ConstraintBounds(Float64[], Float64[], Float64[], Float64[]) : nothing

"""Box constraints"""
struct BoxConstraints{T} <: AbstractConstraints
    bounds::ConstraintBounds{T}
end
BoxConstraints(lower::AbstractVector{T}, upper::AbstractVector{T}) where {T} =
    BoxConstraints(ConstraintBounds(lower, upper, [], []))
BoxConstraints(lower::T, upper::T, size) where {T} =
    BoxConstraints(fill(lower, size), fill(upper, size))
apply!(c::BoxConstraints, x) = clip!(c.bounds, x)

"""
This type encodes constraints as a following additional penalty for an objective function:

``p(x) = \\sum^n_{i=1} r_i max(0, g_i(x))^2``

where ``g_i(x)`` is an inequality constraint. The equality constraints ``h_i(x) = 0`` are transformed
to inequality constraints as follows:

``h(x) - \\epsilon  \\leq 0``
"""
struct PenaltyConstraints{T,F} <: AbstractConstraints
    coef::Vector{T}
    constraints::F   # constraints(x) returns the value of the constraint-functions at x
    bounds::ConstraintBounds{T}
    PenaltyConstraints(penalty::Vector{T}, bounds::ConstraintBounds{T}, c::F) where {T,F} =
        new{T,F}(penalty, c, bounds)
end
PenaltyConstraints(penalty::T, bounds::ConstraintBounds{T}, cf=(x)->nothing) where {T} =
    PenaltyConstraints(fill(penalty, nconstraints(bounds)+nconstraints_x(bounds)), bounds, cf)
PenaltyConstraints(penalty::AbstractVector{T}, lx::AbstractVector{T}, ux::AbstractVector{T},
                   lc::AbstractVector{T}, uc::AbstractVector{T}, cf=(x)->nothing) where {T} =
    PenaltyConstraints(penalty, ConstraintBounds(lx, ux, lc, uc), cf)
PenaltyConstraints(penalty::T, lx::AbstractVector{T}, ux::AbstractVector{T},
                   lc::AbstractVector{T}=T[], uc::AbstractVector{T}=T[],
                   cf=(x)->nothing) where {T} =
    PenaltyConstraints(penalty, ConstraintBounds(lx, ux, lc, uc), cf)
PenaltyConstraints(lx::AbstractVector{T}, ux::AbstractVector{T},
                   lc::AbstractVector{T}=T[], uc::AbstractVector{T}=T[],
                   cf=(x)->nothing) where {T} =
    PenaltyConstraints(one(T), ConstraintBounds(lx, ux, lc, uc), cf)
value(c::PenaltyConstraints, x) = c.constraints(x)
penalty(c::PenaltyConstraints, x) = penalty(c.bounds, c.coef, x, value(c, x))
function penalty!(fitness::AbstractVector{T}, c::PenaltyConstraints{T,F}, population) where {T,F}
    for (i,x) in enumerate(population)
        fitness[i] += penalty(c, x)
    end
    return fitness
end

"""
This type encodes constraints as a following additional penalty for an objective function:

``p(x) = f_{worst} + \\sum^n_{i=1} |g_i(x)|``

if ``x`` is not feasible, otherwise no penalty is applied.
"""
struct WorstFitnessConstraints{T,F} <: AbstractConstraints
    bounds::ConstraintBounds{T}
    constraints::F   # constraints(x) returns the value of the constraint-functions at x
end
value(c::WorstFitnessConstraints, x) = c.constraints(x)
apply!(c::WorstFitnessConstraints, x) = clip!(c.bounds, x)
function penalty!(fitness::AbstractVector{T}, c::WorstFitnessConstraints{T,F}, population) where {T,F}
    worst = maximum(fitness)
    for (i,x) in enumerate(population)
        cv = value(c, x)
        p = zeros(size(cv))
        if !isfeasible(c.bounds, x, cv)
            for (i,j) in enumerate(c.bounds.eqc)
                p[j] = cv[j] - c.bounds.valc[i] - eps()
            end
            for (i,j) in enumerate(c.bounds.ineqc)
                p[j] = c.bounds.σc[i]*(c.bounds.bc[i]-cv[j])
            end
            fitness[i] = worst + sum(abs, p)
        end
    end
    return fitness
end

"""
This type provides an additional type constraints on the varaibles required for mixed integer optimization problmes.
"""
struct MixedTypePenaltyConstraints{C<:AbstractConstraints} <: AbstractConstraints
    penalty::C
    types::Vector{DataType}
end
value(c::MixedTypePenaltyConstraints, x) = value(c.penalty, x)
penalty!(fitness, c::MixedTypePenaltyConstraints, population) = penalty!(fitness, c.penalty, population)
function apply!(c::MixedTypePenaltyConstraints, x)
    y = apply!(c.penalty, x)
    for (i,t) in enumerate(c.types)
        if t <: Integer
            y[i] = round(t, y[i])
        end
    end
    return y
end

# Utilities

function clip!(bounds::ConstraintBounds{T}, x) where {T}
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

function penalty(bounds::ConstraintBounds{T}, coeff::AbstractVector{T},
                 x, c::Union{AbstractVector,Nothing}=nothing) where {T}
    penalty = 0
    xc = nconstraints_x(bounds)
    for (i,j) in enumerate(bounds.eqx)
        penalty += coeff[j]*(x[j] - bounds.valx[i] - eps())^2
    end
    for (i,j) in enumerate(bounds.ineqx)
        penalty += coeff[j]*max(0, bounds.σx[i]*(bounds.bx[i]-x[j]))^2
    end
    if c !== nothing
        for (i,j) in enumerate(bounds.eqc)
            penalty += coeff[xc+j]*max(0, c[j] - bounds.valc[i] - eps())^2
        end
        for (i,j) in enumerate(bounds.ineqc)
            penalty += coeff[xc+j]*max(0, bounds.σc[i]*(bounds.bc[i]-c[j]))^2
        end
    end
    return penalty
end
