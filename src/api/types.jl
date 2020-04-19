"""
Abstract evolutionary optimizer algorithm
"""
abstract type AbstractOptimizer end

function print_header(method::AbstractOptimizer)
    println("Iter     Function value")
end
population_size(method::AbstractOptimizer) = error("`population_size` is not implemented for $(summary(method)).")


# Options
"""
Configurable options with defaults:
```
abstol::Float64 = 1e-32
reltol::Float64 = 1e-32
successive_f_tol::Integer = 10
iterations::Integer = 1000
store_trace::Bool = false
show_trace::Bool  = false
show_every::Integer = 1
callback::TCallback = nothing
```
"""
@kwdef struct Options{TCallback <: Union{Nothing, Function}}
    abstol::Float64 = 1e-32
    reltol::Float64 = 1e-32
    successive_f_tol::Integer = 10
    iterations::Integer = 1000
    store_trace::Bool = false
    show_trace::Bool  = false
    show_every::Integer = 1
    callback::TCallback = nothing
end
function Base.show(io::IO, o::Options)
    for k in fieldnames(typeof(o))
        v = getfield(o, k)
        if v === nothing
            print(io, lpad("$(k)",24) *" = nothing\n")
        else
            print(io, lpad("$(k)",24) *" = $v\n")
        end
    end
end


# Optimizer State
"""
Abstract type for defining an optimizer state

Every algorithm have to implement a state type derived from this abstract type.
"""
abstract type AbstractOptimizerState end

"""
    value(state)

Returns a minimum value of the current `state`.
"""
value(state::AbstractOptimizerState) = error("`value` is not implemented for $(state).")

"""
    value(state)

Returns a minimizer object in the current `state`.
"""
minimizer(state::AbstractOptimizerState) = error("`minimizer` is not implemented for $(state).")


"""
    terminate(state)

Returns `true` if the `state` requires early termination.
"""
terminate(state::AbstractOptimizerState) = false


# Optimization Trace

struct OptimizationTraceRecord{T, O <: AbstractOptimizer}
    iteration::Int
    value::T
    metadata::Dict{String,Any}
end

function Base.show(io::IO, t::OptimizationTraceRecord)
    print(io, lpad("$(t.iteration)",6))
    print(io, "   ")
    print(io, lpad("$(t.value)",14))
    for (key, value) in t.metadata
        print(io, "\n * $key: $value")
    end
    return
end

const OptimizationTrace{T,O} = Vector{OptimizationTraceRecord{T,O}}

function Base.show(io::IO, tr::OptimizationTrace)
    print(io, "Iter     Function value\n")
    print(io, "------   --------------\n")
    for rec in tr
        show(io, rec)
        print(io, "\n")
    end
    return
end


# Miscellaneous


# Constructor for nondifferentiable function over bit vector
NonDifferentiable(f, x::AbstractVector{<:Integer}) =
    NonDifferentiable{Integer,typeof(x)}(f, zero(Integer), zeros(Integer, length(x)), [0,])
NonDifferentiable(f, x::AbstractVector{Bool}) = NonDifferentiable(f, BitVector(x))
function NonDifferentiable(f, x::BitArray)
    xs = BitArray(zero(eltype(x)) for i = 1:length(x))
    NonDifferentiable{Real,typeof(xs)}(f, f(xs), xs, [0,])
end
length(d::NonDifferentiable) = length(d.x_f)


const Strategy = Dict{Symbol,Any}
const Individual = Union{AbstractArray, Function, Nothing}

# Wrapping function for strategy
function strategy(; kwargs...)
    result = Dict{Symbol,Any}()
    for (k, v) in kwargs
        result[k] = v
    end
    return result
end
