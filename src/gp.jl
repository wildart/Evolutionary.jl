"""
Implementation of Koza-type (tree-based) Genetic Programming

The constructor takes following keyword arguments:

- `populationSize`: The size of the population
- `initialization`: A strategy for population initialization (default: `:half`)
  - Possible values: `:grow`, `:full`, `:half`
- `evaluate`: A collection of fitness cases required for evaluation of a fitness function

"""
@kwdef struct TreeGP{T} <: AbstractOptimizer
    populationSize::Integer = 50
    terminals::Vector{Symbol} = [:x, :rand]
    functions::Vector{Symbol} = [:+, :-, :*, :/, :^]
    mindepth::Int = 0
    maxdepth::Int = 10
    initialization::Symbol = :grow
    evaluate::AbstractVector{T} = zeros(T, 0)
end

"""
    initial_population(method::TreeGP, expr::{Expr,Nothing})

Initialize a random population of expressions derived from `expr`.
"""
function initial_population(method::GP, expr::{Expr,Nothing}))
    n = population_size(method)

    return [genrandexpr(method.terminals, method.functions,
                        mindepth=method.mindepth, maxdepth=method.maxdepth)
            for i in 1:n]
end


arity(sym) = 2
function genrandexpr(terms, funcs; maxdepth=2, mindepth=1, method=:grow)
    # println("depth: $maxdepth => $mindepth")
    tl = length(terms)
    fl = length(funcs)
    expr = if (maxdepth <= 0 || ( method == :grow && rand() < tl/(tl+fl) )) && mindepth <= 0
        termsym = rand(terms)
        term = termsym == :rand ? rand() : termsym
        # println("term: $term")
        term
    else
        func = rand(funcs)
        # println("func: $func, ")
        args = Any[]
        for i in 1:arity(func)
            arg = genrandexpr(terms, funcs,
                              maxdepth=maxdepth-1, mindepth=mindepth-1, method=method)
            # println("arg_$i: $arg ")
            push!(args, arg)
        end
        Expr(:call, func, args...)
    end
    return expr
end

terminals = [:x, :rand, 1, 2, 3, 4]
functions = [:+, :-, :*, :/, :^]


ex = genrandexpr(terminals, functions, maxdepth=3, mindepth=2)
exfun = eval(Expr(:->, :x, ex))
exfun(1)

depth(ex) = isa(ex, Expr) ? maximum( depth(e) for e in ex.args )+1 : 0
children(ex) = isa(ex, Expr) ? sum( children(e) for e in ex.args) : 1

children(ex)
depth(ex)
dump(ex)

function subtree(root, idx)
    m, left, right = if !isa(root, Expr) || length(root.args) < 2
        1, root, root
    else
        left, right = root.args[2:3] # left & right nodes
        children(left)+1, left, right
    end
    if m == idx
        return root
    elseif m > idx
        return subtree(left, idx)
    else
        return subtree(right, idx - m)
    end
end
randsubtree(ex) = subtree(ex, rand(1:children(ex)))

[subtree(ex, i) for i in 1:children(ex)]


