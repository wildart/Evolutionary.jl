const Terminal = Union{Symbol, Real, Function}

"""
Implementation of Koza-type (tree-based) Genetic Programming

The constructor takes following keyword arguments:

- `populationSize`: The size of the population
- `terminals`: A dictionary of terminals with their their corresponding dimensionality
    - This dictionary contains (`Terminal`, `Int`) pairs
    - The terminals can be any symbols (variables), constat values, or 0-arity functions.
- `functions`: A collection of functions with their corresponding arity.
    - This dictionary contains (`Function`, `Int`) pairs
- `initialization`: A strategy for population initialization (default: `:grow`)
    - Possible values: `:grow` and `:full`
- `mindepth`: Minimal depth of the expression (default: `0`)
- `maxdepth`: Maximal depth of the expression (default: `3`)
- `simplify`: An expression simplification function (default: `:nothing`)
- `optimizer`: An evolutionary optimizer used for evolving the expressions (default: [`GA`](@ref))
    - Use `mutation` and `crossover` parameters to specify GP-related mutation operation.
    - Use `selection` parameter to specify the offspring selection procedure
"""
@kwdef struct TreeGP <: AbstractOptimizer
    populationSize::Integer = 50
    terminals::Dict{Terminal, Int} = Dict(:x=>1, rand=>1)
    functions::Dict{Function, Int} = Dict( f=>2 for f in [+,-,*,pdiv] )
    mindepth::Int = 0
    maxdepth::Int = 3
    initialization::Symbol = :grow
    simplify::Union{Nothing, Function} = nothing
    optimizer::AbstractOptimizer = GA()
end
function TreeGP(pop::Integer, term::Vector{Terminal}, func::Vector{Function}; kwargs...)
    terminals = Dict(t=>1 for t in term)
    functions = Dict(f=>2 for f in func)
    TreeGP(;populationSize=pop, terminals=terminals, functions=functions, kwargs...)
end

population_size(method::TreeGP) = method.populationSize
default_options(method::TreeGP) = (iterations=1000, abstol=1e-15)
terminals(m::TreeGP) = Symbol[t for t in keys(m.terminals) if isa(t,Symbol)] |> sort!
function summary(m::TreeGP)
    par = join(terminals(m),",")
    "TreeGP[P=$(m.populationSize),Parameter[$(par)],$(keys(m.functions))]"
end
show(io::IO,m::TreeGP) = print(io, summary(m))

"""
    randterm(t::TreeGP)

Returns a random terminal given the specification from the `TreeGP` object `t`.
"""
function randterm(t::TreeGP)
    term = rand(keys(t.terminals))
    if isa(term, Symbol)
        term
    else
        dim = t.terminals[term]
        dim == 1 ? rand() : rand(dim)
    end
end

"""
    rand(t::TreeGP, maxdepth=2; mindepth=maxdepth-1)::Expr

Create a ranodm expression tree given the specification from the `TreeGP` object `t`.
"""
function rand(t::TreeGP, maxdepth::Int=2; mindepth::Int=maxdepth-1)
    @assert maxdepth > mindepth "`maxdepth` must be larger then `mindepth`"
    tl = length(t.terminals)
    fl = length(t.functions)
    root = if (maxdepth == 0  || ( t.initialization == :grow && rand() < tl/(tl+fl) ) ) && mindepth <= 0
        randterm(t)
    else
        rand(keys(t.functions))
    end
    if isa(root, Function)
        args = Any[]
        for i in 1:t.functions[root]
            arg = rand(t, maxdepth-1, mindepth=mindepth-1)
            push!(args, arg)
        end
        Expr(:call, root, args...)
    else
        return root
    end
end

"""
    initial_population(m::TreeGP, expr::{Expr,Nothing}=nothing)

Initialize a random population of expressions derived from `expr`.
"""
function initial_population(m::TreeGP, expr::Union{Expr,Nothing}=nothing)
    n = population_size(m)
    return [
        expr === nothing ? rand(m, m.maxdepth, mindepth=m.mindepth) : deepcopy(expr)
        for i in 1:n
    ]
end

mutable struct GPState{T,IT} <: Evolutionary.AbstractOptimizerState
    ga::GAState{T,IT}
end
value(s::GPState) = s.ga.fitness
minimizer(s::GPState) = s.ga.fittest

"""Initialization of GP algorithm state"""
function initial_state(method::TreeGP, options, objfun, population)
    return GPState(initial_state(method.optimizer, options, objfun, population))
end

function update_state!(objfun, constraints, state::GPState, population::AbstractVector{IT}, method, options, itr) where {IT}
    # perform GA step
    res = update_state!(objfun, constraints, state.ga, population, method.optimizer, options, itr)
    # simplify expressions
    if method.simplify !== nothing
        for i in 1:length(population)
            method.simplify(population[i])
        end
    end
    return res
end

# Custom optimization call
function optimize(f, method::TreeGP, options::Options = Options(;default_options(method)...))
    method.optimizer.mutation =  subtree(method)
    method.optimizer.crossover = crosstree
    optimize(f, NoConstraints(), nothing, method, options)
end
