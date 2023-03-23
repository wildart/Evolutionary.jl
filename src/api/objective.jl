"""
Wrapper around an objective function (compatible with NLSolversBase).
"""
mutable struct EvolutionaryObjective{TC,TF,TX,TP} <: AbstractObjective
    f::TC
    F::TF
    x_f::TX
    f_calls::Int
end

"""
    EvolutionaryObjective(f, x[, F])

Constructor for an objective function object around the function `f` with initial parameter `x`, and objective value `F`.
"""
function EvolutionaryObjective(f::TC, x::AbstractArray,
                               F::Union{Real, AbstractArray{<:Real}} = zero(f(x));
                               eval::Symbol = :serial) where {TC}
    defval = default_values(x)
    # convert function into the in-place one
    TF = typeof(F)
    fn, TN = if funargnum(f) == 2 && F isa AbstractArray
        ff = (Fv,xv) -> (Fv .= f(xv))
        ff, typeof(ff)
    else
        f, TC
    end
    EvolutionaryObjective{TN,TF,typeof(x),Val{eval}}(fn, F, defval, 0)
end

"""
    EvolutionaryObjective(f, x::Expr[, F])

Constructor for an objective object for a Julia evaluatable expression.
"""
function EvolutionaryObjective(f::TC, x::Expr, F::TF = zero(f(x));
                               eval::Symbol = :serial) where {TC,TF<:Real}
    EvolutionaryObjective{TC,TF,typeof(x),Val{eval}}(f, F, :(), 0)
end

f_calls(obj::EvolutionaryObjective) = obj.f_calls

"""
    ismultiobjective(objfun)

Return `true` if the function is multi-objective objective.
"""
ismultiobjective(obj) = obj.F isa AbstractArray

function value(obj::EvolutionaryObjective{TC,TF,TX,TP}, x::TX) where {TC,TF,TX,TP}
    obj.f_calls += 1
    obj.f(x)::TF
end

function value(obj::EvolutionaryObjective{TC,TF,TX,TP}, x::TX, pop::AbstractVector{TX}) where {TC,TF,TX,TP}
    obj.f_calls += 1
    obj.f(x, pop)::TF
end

function value!(obj::EvolutionaryObjective{TC,TF,TX,TP}, x::TX) where {TC,TF,TX,TP}
    obj.F = value(obj, x)
    obj.F
end

function value!!(obj::EvolutionaryObjective{TC,TF,TX,TP}, x::TX) where {TC,TF,TX,TP}
    copyto!(obj.x_f, x)
    value!(obj, x)
end

function value(obj::EvolutionaryObjective{TC,TF,TX,TP}, F, x::TX) where {TC,TF,TX,TP}
    obj.f_calls += 1
    obj.f(F, x)
end
value(obj::EvolutionaryObjective{TC,TF,TX,TP}, x::TX) where {TC,TF<:AbstractArray,TX,TP} = value(obj, copy(obj.F), x)

function value(obj::EvolutionaryObjective{TC,TF,TX,TP}, F, x::TX, pop::AbstractVector{TX}) where {TC,TF,TX,TP}
    obj.f_calls += 1
    obj.f(F, x, pop)
end
value(obj::EvolutionaryObjective{TC,TF,TX,TP}, x::TX, pop::AbstractVector{TX}) where {TC,TF<:AbstractArray,TX,TP} = value(obj, copy(obj.F), x, pop)

function value!(obj::EvolutionaryObjective{TC,TF,TX,TP}, F, x::TX) where {TC,TF,TX,TP}
    obj.F = value(obj, F, x)
end

function value!!(obj::EvolutionaryObjective{TC,TF,TX,TP}, F, x::TX) where {TC,TF,TX,TP}
    copyto!(obj.x_f, x)
    value!(obj, F, x)
end

function value!(obj::EvolutionaryObjective{TC,TF,TX,Val{:serial}},
                F::AbstractVector, xs::AbstractVector{TX}, pop_dependent::Bool=false) where {TC,TF<:Real,TX}
    if pop_dependent
        broadcast!(x->value(obj,x,xs), F, xs)
    else
        broadcast!(x->value(obj,x), F, xs)
    end
    F
end

function value!(obj::EvolutionaryObjective{TC,TF,TX,Val{:thread}},
                F::AbstractVector, xs::AbstractVector{TX}, pop_dependent::Bool=false) where {TC,TF<:Real,TX}
    n = length(xs)
    Threads.@threads for i in 1:n
        if pop_dependent
            F[i] = value(obj, xs[i], xs)
        else
            F[i] = value(obj, xs[i])
        end
    end
    F
end

function value!(obj::EvolutionaryObjective{TC,TF,TX,Val{:serial}},
                F::AbstractMatrix, xs::AbstractVector{TX}, pop_dependent::Bool=false) where {TC,TF,TX}
    n = length(xs)
    for i in 1:n
        fv = view(F, :, i)
        if pop_dependent
            value(obj, fv, xs[i], xs)
        else
            value(obj, fv, xs[i])
        end
    end
    F
end

function value!(obj::EvolutionaryObjective{TC,TF,TX,Val{:thread}},
                F::AbstractMatrix, xs::AbstractVector{TX}, pop_dependent::Bool=false) where {TC,TF,TX}
    n = length(xs)
    @Threads.threads for i in 1:n
        fv = view(F, :, i)

        if pop_dependent
            value(obj, fv, xs[i], xs)
        else
            value(obj, fv, xs[i])
        end
    end
    end
    F
end
