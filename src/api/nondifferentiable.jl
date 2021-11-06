# Constructor for non-differentiable function with Integer-derived value
NonDifferentiable(f, x::AbstractVector{TX}, F::TF = zero(f(x))) where {TF<:Real, TX<:Integer} =
    NonDifferentiable{TF,typeof(x)}(f, F, zeros(TX, length(x)), [0,])

NonDifferentiable(f, x::AbstractVector{TX}, F::TF) where {TF<:AbstractVector, TX<:Integer} =
    NonDifferentiable{TF,typeof(x)}(f, F, zeros(TX, length(x)), [0,])

# Constructor for non-differentiable function over bit vector
NonDifferentiable(f, x::AbstractVector{Bool}) = NonDifferentiable(f, BitVector(x))

function NonDifferentiable(f, x::BitArray)
    xs = similar(x)
    fzval = f(xs)
    NonDifferentiable{typeof(fzval),typeof(xs)}(f, fzval, xs, [0,])
end

# Constructor for non-differentiable function for expression
function NonDifferentiable(f, x::Expr, F::TF = zero(f(x))) where {TF<:Real}
    NonDifferentiable{TF,typeof(x)}(f, F, :(), [0,])
end

"""
    ismmo(objfun)

Return `true` if the function is multiobjective objective.
"""
ismmo(f::AbstractObjective) = f.F isa AbstractArray


##############
# EVALUATION #
##############

function value(obj::NonDifferentiable{TF, TX}, x) :: TF where {TF, TX}
    @inbounds obj.f_calls[1] += 1
    return obj.f(x)
end

function value(obj::NonDifferentiable{TF, TX}, F::TF, x) :: TF where {TF<:AbstractArray, TX}
    @inbounds obj.f_calls[1] += 1
    return obj.f(F, x)
end

function value!!(obj::NonDifferentiable{TF, TX}, F::TF, x) where {TF<:AbstractArray, TX}
    obj.f(F, x)
    copyto!(obj.x_f, x)
    @inbounds obj.f_calls[1] += 1
    F
end

function value!(obj::NonDifferentiable{TF, TX}, F::TF, x) :: TF where {TF<:AbstractArray, TX}
    if x != obj.x_f
        value!!(obj, F, x)
    end
    value(obj)
end

function value!(::Val{:serial}, fitness::AbstractVector, objfun, population::AbstractVector{IT}) where {IT}
    for i in 1:length(population)
        fitness[i] = value(objfun, population[i])
    end
end

function value!(::Val{:serial}, fitness, objfun, population::AbstractVector{IT}) where {IT}
    for i in 1:length(population)
        fv = view(fitness, :, i)
        value(objfun, fv, population[i])
    end
end

function value!(::Val{:thread}, fitness::AbstractVector, objfun, population::AbstractVector{IT}) where {IT}
    Threads.@threads for i in 1:length(population)
        fitness[i] = value(objfun, population[i])
    end
end

function value!(::Val{:thread}, fitness, objfun, population::AbstractVector{IT}) where {IT}
    Threads.@threads for i in 1:length(population)
        fv = view(fitness, :, i)
        value(objfun, fv, population[i])
    end
end

