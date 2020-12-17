"""
Julia expression wrapper
"""
struct Expression
    expr::Expr
    syms::Vector{Symbol}
end
Expression(ex::Expr) = Expression(ex, sort!(symbols(ex)))
show(io::IO, e::Expression) = infix(io, e.expr)
(e::Expression)(val) = evaluate(val, e.expr, e.syms)

function symbols(ex::Expr)
    syms = Symbol[]
    for e in ex.args
        isa(e, Symbol) && push!(syms, e)
        isa(e, Expr) && append!(syms, symbols(e))
    end
    unique!(syms)
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
    ex1
end

function randsubexpr(ex)
    !isa(ex, Expr) && return ex
    csize = map(nodes, ex.args[2:end])
    cidx = rand(1:sum(csize)+1)-1
    if cidx == 0
        ex
    else
        nidx = findfirst(i->cidx<=i, cumsum(csize))
        randsubexpr(ex.args[nidx+1])
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

"""
Special eveluation conditions for some common functions.
"""
function specialfunc(f::Function, vals...)
    if f == (log)
        vl = first(vals)
        iszero(vl) ? one(vl) : log(abs(vl))
    elseif f == (/)
        iszero(last(vals)) ? one(last(vals)) : (/)(vals...)
    elseif f == (sin)
        vl = first(vals)
        isinf(vl) ? one(vl) : f(vl)
    elseif f == (cos)
        vl = first(vals)
        isinf(vl) ? one(vl) : f(vl)
    elseif f == (^)
        vl = real(complex(first(vals))^last(vals))
        isinf(vl) || isnan(vl) ? zero(first(vals)) : vl
    else
        f(vals...)
    end
end

function evaluate(val, ex::Expr, psyms::Vector{Symbol})
    exprm = ex.args
    exvals = (isa(nex, Expr) || isa(nex, Symbol) ? evaluate(val, nex, psyms) : nex for nex in exprm[2:end])
    specialfunc(exprm[1], exvals...)
end

function evaluate(val, ex::Symbol, psyms::Vector{Symbol})
    pidx = findfirst(isequal(ex), psyms)
    val[pidx]
end

iszeronum(root) = isa(root, Number) && iszero(root)
isonenum(root) = isa(root, Number) && isone(root)

"""
    simplify!(expr)

Simplify an algebraic expression
"""
function simplify!(root)
    # return if not an expression
    !isa(root, Expr) && return root
    # return if not a function
    root.head != :call && return root

    # recursively simplify arguments
    for i in 2:length(root.args)
        !isa(root.args[i], Expr) && continue
        root.args[i] = simplify!(root.args[i])
    end

    # some elementary simplification rules
    if root.args[1] == (-) && root.args[2] == root.args[3] # additive inverse: x-x = 0
        root = 0
    elseif root.args[1] == (/) && (root.args[2] == root.args[3] || iszeronum(root.args[3])) # look for multiplicative inverse x/x = 1
        root = 1
    elseif (root.args[1] == (*) || root.args[1] == (/)) &&  (iszeronum(root.args[2]) || iszeronum(root.args[3])) # look for 0: x*0 = 0*x = 0
        root = 0
    elseif (root.args[1] == (+) || root.args[1] == (-)) && iszeronum(root.args[3]) # look for 0: x ± 0 = 0
        root = root.args[2]
    elseif root.args[1] == (+) && iszeronum(root.args[2]) # look for 0: 0 ± x = 0
        root = root.args[3]
    elseif (root.args[1] == (*) || root.args[1] == (/)) && isonenum(root.args[3]) # x*1 = x || x/1 = x
        root = root.args[2]
    elseif root.args[1] == (*) && isonenum(root.args[2]) # 1*x = x
        root = root.args[3]
    else
        # x+x = 2x
        if root.args[1] == (+) && root.args[2] == root.args[3]
            root.args[1] = (*)
            root.args[2] = 2
        end
        # evaluate numerical arithmetic
        if all(e->!(isa(e, Symbol) || isa(e, Expr)), root.args[2:end])
            root = specialfunc(root.args[1], root.args[2:end]...)
        end
    end
    return root
end

function infix(io::IO, root; digits=3)
    if isa(root, Number)
        print(io, round(root, digits=digits))
    elseif isa(root, Expr) && root.head == :call
        if root.args[1] ∈ [+, -, *, /, ^]
            print(io, "(")
            infix(io, root.args[2])
            infix(io, root.args[1])
            infix(io, root.args[3])
            print(io, ")")
        else
            infix(io, root.args[1])
            print(io, "(")
            for ex in root.args[2:end]
                infix(io, ex)
            end
            print(io, ")")
        end
    else
        print(io, root)
    end
end
show(io::IO, ::MIME"text/html", e::Expression) = infix(io, e.expr)

latex(io::IO, root::Number; digits=3) = print(io, round(root, digits=digits))
latex(io::IO, root::Symbol; kwargs...) = print(io, root)
function latex(io::IO, root::Expr; digits=3)
    root.head != :call && print(io, expr)
    if root.args[1] == (/)
        print(io, "\\frac{")
        latex(io, root.args[2], digits=digits)
        print(io, "}{")
        latex(io, root.args[3], digits=digits)
        print(io, "}")
    elseif root.args[1] ∈ [+, -, *]
        print(io, "\\left(")
        latex(io, root.args[2], digits=digits)
        print(io, root.args[1])
        latex(io, root.args[3], digits=digits)
        print(io, "\\right)")
    elseif root.args[1] == (^)
        print(io, "{")
        latex(io, root.args[2], digits=digits)
        print(io, "}^{")
        latex(io, root.args[3], digits=digits)
        print(io, "}")
    else
        print(io, "\\")
        print(io, root.args[1])
        print(io, "\\left(")
        for ex in root.args[2:end]
            latex(io, ex, digits=digits)
        end
        print(io, "\\right)")
    end
end
show(io::IO, ::MIME"text/latex", e::Expression) = latex(io, e.expr)
