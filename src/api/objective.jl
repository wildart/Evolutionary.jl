"""
Wrapper around an objective function (compatible with NLSolversBase).
"""
mutable struct EvolutionaryObjective{TC,TF,TX,TP} <: AbstractObjective
    f::TC
    F::TF
    x_f::TX
    f_calls::Int
end

function EvolutionaryObjective(f, x::AbstractArray,
                               F::Union{Real, AbstractArray{<:Real}} = zero(f(x));
                               eval::Symbol = :serial)
    defval = default_values(x)
    # convert function into the in-place one
    fn = if funargnum(f) == 2 && F isa AbstractArray
        (Fv,xv) -> (Fv .= f(xv))
    else
        f
    end
    EvolutionaryObjective{typeof(fn),typeof(F),typeof(x),Val{eval}}(fn, F, defval, 0)
end

# Constructor for non-differentiable function for expression
function EvolutionaryObjective(f, x::Expr, F::TF = zero(f(x));
                               eval::Symbol = :serial) where {TF<:Real}
    EvolutionaryObjective{typeof(f),TF,typeof(x),Val{eval}}(f, F, :(), 0)
end

f_calls(obj::EvolutionaryObjective) = obj.f_calls

"""
    ismultiobjective(objfun)

Return `true` if the function is multiobjective objective.
"""
ismultiobjective(obj) = obj.F isa AbstractArray

function value(obj::EvolutionaryObjective{TC,TF,TX,TP}, x::TX) where {TC,TF,TX,TP}
    obj.f_calls += 1
    obj.f(x)::TF
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

function value!(obj::EvolutionaryObjective{TC,TF,TX,TP}, F, x::TX) where {TC,TF,TX,TP}
    obj.F = value(obj, F, x)
end

function value!!(obj::EvolutionaryObjective{TC,TF,TX,TP}, F, x::TX) where {TC,TF,TX,TP}
    copyto!(obj.x_f, x)
    value!(obj, F, x)
end

function value!(obj::EvolutionaryObjective{TC,TF,TX,Val{:serial}},
                F::AbstractVector, xs::AbstractVector{TX}) where {TC,TF<:Real,TX}
    broadcast!(x->value(obj,x), F, xs)
    F
end

function value!(obj::EvolutionaryObjective{TC,TF,TX,Val{:thread}},
                F::AbstractVector, xs::AbstractVector{TX}) where {TC,TF<:Real,TX}
    n = length(xs)
    Threads.@threads for i in 1:n
        F[i] = value(obj, xs[i])
    end
    F
end

function value!(obj::EvolutionaryObjective{TC,TF,TX,Val{:serial}},
                F::AbstractMatrix, xs::AbstractVector{TX}) where {TC,TF,TX}
    n = length(xs)
    for i in 1:n
        fv = view(F, :, i)
        value(obj, fv, xs[i])
    end
    F
end

function value!(obj::EvolutionaryObjective{TC,TF,TX,Val{:thread}},
                F::AbstractMatrix, xs::AbstractVector{TX}) where {TC,TF,TX}
    n = length(xs)
    @Threads.threads for i in 1:n
        fv = view(F, :, i)
        value(obj, fv, xs[i])
    end
    F
end
