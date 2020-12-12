const Terminal = Union{Symbol, Function}

"""
Implementation of Koza-type (tree-based) Genetic Programming

The constructor takes following keyword arguments:

- `populationSize`: The size of the population
- `initialization`: A strategy for population initialization (default: `:half`)
  - Possible values: `:grow`, `:full`, `:half`

"""
@kwdef struct TreeGP <: AbstractOptimizer
    populationSize::Integer = 50
    terminals::Dict{Terminal, Int} = Dict(:x=>1, rand=>1)
    functions::Dict{Function, Int} = Dict( f=>2 for f in [+,-,*,/] )
    mindepth::Int = 0
    maxdepth::Int = 10
    initialization::Symbol = :grow
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

function rand(t::TreeGP; maxdepth=2, mindepth=maxdepth-1, method=:grow)
    # println("depth: $maxdepth => $mindepth")
    @assert maxdepth > mindepth "`maxdepth` must be larger then `mindepth`"
    tl = length(t.terminals)
    fl = length(t.functions)
    root = if (maxdepth == 0  || ( method == :grow && rand() < tl/(tl+fl) ) ) && mindepth <= 0
        term = rand(keys(t.terminals))
        if isa(term, Symbol)
            term
        else
            dim = t.terminals[term]
            dim == 1 ? rand() : rand(dim)
        end
    else
        rand(keys(t.functions))
    end
    if isa(root, Function)
        args = Any[]
        for i in 1:t.functions[root]
            arg = rand(t, maxdepth=maxdepth-1, mindepth=mindepth-1, method=method)
            # println("arg_$i: $arg ")
            push!(args, arg)
        end
        Expr(:call, root, args...)
    else
        return root
    end
end

"""
    initial_population(method::TreeGP, expr::{Expr,Nothing}=nothing)

Initialize a random population of expressions derived from `expr`.
"""
function initial_population(method::TreeGP, expr::Union{Expr,Nothing}=nothing)
    n = population_size(method)
    return [rand(method, mindepth=method.mindepth, maxdepth=method.maxdepth) for i in 1:n]
end

height(ex) = isa(ex, Expr) ? maximum( height(e) for e in ex.args )+1 : 0
nodes(ex) = !isa(ex, Expr) ? 1 : length(ex.args) > 0 ? sum( nodes(e) for e in ex.args ) : 0
length(ex) = nodes(ex)

function depth(root, ex; d=0)
    return if root == ex
        d
    elseif isa(root, Expr)
        maximum( depth(e, ex, d=d+1) for e in root.args )
    else
        -1
    end
end

function copyto!(ex1::Expr, ex2::Expr)
    ex1.head = ex2.head
    ex1.args = deepcopy(ex2.args)
end

function randsubtree(ex)
    !isa(ex, Expr) && return ex
    csize = map(nodes, ex.args[2:end])
    cidx = rand(1:sum(csize)+1)-1
    if cidx == 0
        ex
    else
        nidx = findfirst(i->cidx<=i, cumsum(csize))
        randsubtree(ex.args[nidx+1])
    end
end

function getindex(ex::Expr, idx::Int)
    if idx == 0
        ex
    else
        csize = cumsum(map(nodes, ex.args[2:end]))
        nidx = findfirst(i->i>=idx, csize)
        nex = ex.args[nidx+1]
        !isa(nex, Expr) ? nex : nex[csize[nidx]-idx]
    end
end

function setindex!(ex::Expr, subex, idx::Int)
    csize = cumsum(map(nodes, ex.args[2:end]))
    nidx = findfirst(i->i>=idx, csize)
    nex = ex.args[nidx+1]
    if !isa(nex, Expr) || csize[nidx]-idx == 0
        ex.args[nidx+1] = subex
    else
        nex[csize[nidx]-idx] = subex
    end
end

function crosstree(t1::Expr, t2::Expr)
    tt1, tt2 = copy(t1), copy(t2)
    i, j = rand(1:nodes(t1)-1), rand(1:nodes(t2)-1)
    ex1 = tt1[i]
    ex2 = tt2[j]
    tt1[i] = ex2
    tt2[j] = ex1
    tt1, tt2
end

function mutatetree(method::TreeGP)
    function mutation(recombinant::Expr)
        i = rand(1:nodes(recombinant)-1)
        th = depth(recombinant, recombinant[i])
        recombinant[i] = rand(method, maxdepth=max(0, method.maxdepth-th))
        recombinant
    end
    return mutation
end

"""
- `fitness`: A collection of fitness cases required for evaluation of a fitness function
"""
mutable struct GPState{T,IT} <: Evolutionary.AbstractOptimizerState
    ga::GAState{T,IT}
end
value(s::GPState) = s.ga.fitness
minimizer(s::GPState) = s.ga.fittest

"""Initialization of GA algorithm state"""
function initial_state(method::TreeGP, options, objfun, population)
    # setup initial state
    return GPState(initial_state(method.optimizer, options, objfun, population))
end

function update_state!(objfun, constraints, state::GPState, population::AbstractVector{IT}, method, itr) where {IT}
    # perform GA step
    res = update_state!(objfun, constraints, state.ga, population, method.optimizer, itr)
    # simplify expressions
    for i in 1:length(population)
        simplify!(population[i])
    end
    return res
end

function evaluate(ex::Expr, params, vals)
    exprm = ex.args
    exvals = (isa(nex, Expr) || isa(nex, Symbol) ? evaluate(nex, params, vals) : nex for nex in exprm[2:end])
    exprm[1](exvals...)
end

function evaluate(ex::Symbol, params, vals)
    pidx = findfirst(isequal(ex), params)
    vals[pidx]
end

iszeronum(root) = isa(root, Number) && iszero(root)
isonenum(root) = isa(root, Number) && isone(root)

function simplify!(root)
    # return non-expression
    !isa(root, Expr) && return root
    if root.head == :call
        for i in 2:3
            !isa(root.args[i], Expr) && continue
            root.args[i] = simplify!(root.args[i])
        end
        # simplification rules
        if root.args[1] == (-) && root.args[2] == root.args[3] # additive inverse
            root = 0
        elseif root.args[1] == (/) && (root.args[2] == root.args[3] || iszeronum(root.args[3])) # look for multiplicative inverse
            root = 1
        elseif (root.args[1] == (*) || root.args[1] == (/)) &&  (iszeronum(root.args[2]) || iszeronum(root.args[3])) # look for 0
            root = 0
        elseif (root.args[1] == (+) || root.args[1] == (-)) && iszeronum(root.args[3]) # look for 0
            root = root.args[2]
        elseif root.args[1] == (+) && iszeronum(root.args[2]) # look for 0
            root = root.args[3]
        elseif (root.args[1] == (*) || root.args[1] == (/)) && isonenum(root.args[3]) # x*1 = x || x/1 = x
            root = root.args[2]
        elseif root.args[1] == (*) && isonenum(root.args[2]) # 1*x = x
            root = root.args[3]
        else
            if root.args[1] == (+) && root.args[2] == root.args[3] # + into *
                root.args[1] = (*)
                root.args[2] = 2
            end
            if root.args[1] == (*) && # (z/d)*(x*y) = x*z (if y = d) || y*z (if x = d)
                isa(root.args[2], Expr) && root.args[2].args[1] == (/) &&
                isa(root.args[3], Expr) && root.args[3].args[1] == (*)
                x = root.args[3].args[2]
                y = root.args[3].args[3]
                z = root.args[2].args[2]
                d = root.args[2].args[3]
                if x == d
                    root.args[2] = z
                    root.args[3] = y
                elseif y == d
                    root.args[2] = z
                    root.args[3] = x
                elseif isa(d, Number) && isa(x, Number) # (z/d)*(x*y) = (y*z)*(x/d) = (y*z)*w
                    root.args[3] = x/d
                    root.args[2].args[1] = *
                    root.args[2].args[2] = y
                    root.args[2].args[3] = z
                elseif isa(d, Number) && isa(y, Number)  # (z/d)*(x*y) = (x*z)*(y/d) = (x*z)*w
                    root.args[3] = y/d
                    root.args[2].args[1] = *
                    root.args[2].args[2] = x
                    root.args[2].args[3] = z
                end
            end
            if root.args[1] == (*) && # (x*y)*(z/d) = x*z (if y = d) || y*z (if x = d)
                isa(root.args[2], Expr) && root.args[2].args[1] == (*) &&
                isa(root.args[3], Expr) && root.args[3].args[1] == (/)
                x = root.args[2].args[2]
                y = root.args[2].args[3]
                z = root.args[3].args[2]
                d = root.args[3].args[3]
                if x == d
                    root.args[2] = y
                    root.args[3] = z
                elseif y == d
                    root.args[2] = x
                    root.args[3] = z
                elseif isa(d, Number) && isa(x, Number) # (x*y)*(z/d) = (x/d)*(y*z) = w*(y*z)
                    root.args[2] = x/d
                    root.args[3].args[1] = *
                    root.args[3].args[2] = y
                    root.args[3].args[3] = z
                elseif isa(d, Number) && isa(y, Number)  # (x*y)*(z/d) = (y/d)*(x*z) = w*(x*z)
                    root.args[2] = y/d
                    root.args[3].args[1] = *
                    root.args[3].args[2] = x
                    root.args[3].args[3] = z
                end
            end
            if (root.args[1] == (+) || root.args[1] == (-)) && # x ± (y ± z) = (x ± y) ± z = w ± z
                isa(root.args[2], Number) &&
                isa(root.args[3], Expr) && (root.args[3].args[1] == (+) || root.args[3].args[1] == (-)) &&
                isa(root.args[3].args[2], Number)
                root.args[2] = root.args[1](root.args[2], root.args[3].args[2])
                root.args[1] = root.args[3].args[1]
                root.args[3] = root.args[3].args[3]
            end
            if (root.args[1] == (+) || root.args[1] == (-)) && # (x ± y) ± z = x ± (y ± z) = x ± w
                isa(root.args[3], Number) &&
                isa(root.args[2], Expr) && (root.args[2].args[1] == (+) || root.args[2].args[1] == (-)) &&
                isa(root.args[2].args[3], Number)
                root.args[3] = root.args[1](root.args[3], root.args[2].args[3])
                root.args[1] = root.args[2].args[1]
                root.args[2] = root.args[2].args[2]
            end
            if all(e->!(isa(e, Symbol) || isa(e, Expr)), root.args[2:3]) # arithmetic
                root = root.args[1](root.args[2:3]...)
            end
        end
    end
    return root
end

function infix(root; digits=3)
    if isa(root, Number)
        print(round(root, digits=digits))
    elseif isa(root, Expr) && root.head == :call
        print("(")
        infix(root.args[2])
        infix(root.args[1])
        infix(root.args[3])
        print(")")
    else
        print(root)
    end
end

function latex(root; digits=3)
    if isa(root, Number)
        print(round(root, digits=digits))
    elseif isa(root, Expr) && root.head == :call
        if root.args[1] == (/)
            print("\\frac{")
            latex(root.args[2])
            print("}{")
            latex(root.args[3])
            print("}")
        else
            print("\\left(")
            latex(root.args[2])
            latex(root.args[1])
            latex(root.args[3])
            print("\\right)")
        end
    else
        print(root)
    end
end

# Custom optimization call
function optimize(f, method::TreeGP, options::Options = Options(;default_options(method)...))
    method.optimizer.mutation =  mutatetree(method)
    method.optimizer.crossover = crosstree
    optimize(f, NoConstraints(), nothing, method, options)
end
