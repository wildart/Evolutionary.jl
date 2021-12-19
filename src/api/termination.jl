#######################
# Convergence Metrics #
#######################

# Utilities

abschange(curr, prev) = Float64(abs(curr - prev))
relchange(curr, prev) = Float64(abs(curr - prev)/abs(curr))

maxdiff(x::AbstractArray, y::AbstractArray) = mapreduce((a, b) -> abs(a - b), max, x, y)
abschange(curr::T, prev) where {T<:AbstractArray} = maxdiff(curr, prev)
relchange(curr::T, prev) where {T<:AbstractArray} = maxdiff(curr, prev)/maximum(abs, curr)

# Single value

"""
Estimate absolute difference convergence.
"""
mutable struct AbsDiff{T} <: ConvergenceMetric
    tol::Float64
    Δ::Float64
    value::T
end
AbsDiff{T}(tol::Float64) where {T} = AbsDiff(tol, Inf, zero(T))
AbsDiff() = AbsDiff{Float64}(1e-12)
description(m::AbsDiff) = "|f(x) - f(x')|"
function assess!(m::AbsDiff, state::AbstractOptimizerState)
    val = value(state)
    m.Δ = abschange(val, m.value)
    m.value = val
    converged(m)
end

"""
Estimate relative difference convergence.
"""
mutable struct RelDiff{T} <: ConvergenceMetric
    tol::Float64
    Δ::Float64
    value::T
end
RelDiff{T}(tol::Float64) where {T} = RelDiff(tol, Inf, zero(T))
RelDiff() = RelDiff{Float64}(1e-12)
description(m::RelDiff) = "|f(x) - f(x')|/|f(x')|"
function assess!(m::RelDiff, state::AbstractOptimizerState)
    val = value(state)
    m.Δ = relchange(val, m.value)
    m.value = val
    converged(m)
end


"""
    gd(A,R)

Calculate a generational distance between set `A` and the reference set `R`.
This metric measures the convergence, i.e. closeness of the non-dominated solutions
to the Pareto front, of a population.

*Note:* Parameters are column-major matrices.
"""
function gd(A::AbstractMatrix, R::AbstractMatrix)
    da, na = size(A)
    dr, nr = size(R)
    (na == 0 || nr == 0) && return Inf
    sum = 0
    for a in eachcol(A)
        sum += minimum(norm(a-r) for r in eachcol(R))
    end
    sum/na
end

"""
    igd(S,R)

Calculate an inverted generational distance, [`gd`](@ref), between set `S` and the reference set `R`.
Parameters are column-major matrices.
"""
igd(S,R) = gd(R,S)

mutable struct GD{T} <: ConvergenceMetric
    tol::Float64
    Δ::Float64
    value::Matrix{T}
    inverted::Bool
end
GD{T}(tol::Float64, inv=false) where {T} = GD(tol, Inf, zeros(T,0,0), inv)
GD(inv=false) = GD{Float64}(1e-5, inv)
function description(m::GD)
    prefix = m.inverted ? "I" : "" 
    "|$(prefix)GD(P) - $(prefix)GD(P')|"
end
function assess!(m::GD, state::AbstractOptimizerState)
    val = value(state)
    m.Δ = m.inverted ? igd(val, m.value) : gd(val, m.value)
    m.value = val
    converged(m)
end


"""
    spread(S,R)

Returns a diversity metric of a population of set `S` to the reference set `R`.
"""
function spread(S::AbstractMatrix, R::AbstractMatrix)
    n = size(S,2)
    m = size(R,2)
    Δₖ = [ minimum(norm(view(S,:,i)-view(R,:,j)) for i in 1:n if view(S,:,i) != view(R,:,j)) for j in 1:m ]
    Δ = mean(Δₖ)
    sum(abs.(Δₖ.-Δ))/m*Δ
end

function spread(S::AbstractMatrix)
    n = size(S,2)
    n == 1 && return NaN
    Δₖ = [ minimum(norm(view(S,:,i)-view(S,:,j)) for j in 1:n if view(S,:,i) != view(S,:,j)) for i in 1:n ]
    Δ = mean(Δₖ)
    sum(abs.(Δₖ.-Δ))/n*Δ
end

##########################
# Convergence Assessment #
##########################

assess_convergence(state::AbstractOptimizerState, method) =
    any(assess!(cm, state) for cm in metrics(method))

