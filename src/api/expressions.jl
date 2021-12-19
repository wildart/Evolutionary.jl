"""
Julia expression wrapper
"""
struct Expression
    expr::Expr
    syms::Dict{Symbol,Int}
end
function Expression(ex::Expr)
    syms = Dict( s=>i for (i,s) in pairs(sort!(symbols(ex))) )
    Expression(ex, syms)
end
show(io::IO, e::Expression) = infix(io, e.expr)
(e::Expression)(vals::T...) where {T} = evaluate(e.expr, e.syms, vals...)

function symbols(ex::Expr)
    syms = Symbol[]
    for e in ex.args
        isa(e, Symbol) && push!(syms, e)
        isa(e, Expr) && append!(syms, symbols(e))
    end
    unique!(syms)
end

function compile(ex::Expr, params::Vector{Symbol})
    tprm = Expr(:tuple, params...)
    Expr(:->, tprm, ex)
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

function copyto!(dst::Expr, src::Expr)
    dst.head = src.head
    dst.args = deepcopy(src.args)
    dst
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

isnum(ex) = isa(ex, Number)
iszeronum(root) = isnum(root) && iszero(root)
isonenum(root) = isnum(root) && isone(root)
isdiv(ex) = ex in [/, div, pdiv, aq]
isexpr(ex) = isa(ex, Expr)
issym(ex) = isa(ex, Symbol)

function evaluate(ex::Expr, psyms::Dict{Symbol,Int}, vals::T...)::T where {T}
    exprm = ex.args
    exvals = (isexpr(nex) || issym(nex) ? evaluate(nex, psyms, vals...) : nex for nex in exprm[2:end])
    exprm[1](exvals...)
end

function evaluate(ex::Symbol, psyms::Dict{Symbol,Int}, vals::T...)::T where {T}
    pidx = get(psyms, ex, 0)
    if pidx == 0
        @error "Undefined symbol: $ex"
    end
    return T(vals[pidx])
end


function simplifyunary!(root)
    fn, op = root.args
    if isexpr(op)
        fn2, op2 = op.args
        if fn == log && fn2 == exp
            return op2
        elseif fn == exp && fn2 == log
            return op2
        end
    end
    return root
end

function simplifybinary!(root)
    fn, op1, op2 = root.args

    # some elementary simplification rules
    if fn == (-) && op1 == op2 # additive inverse: x-x = 0
        root = 0
    elseif isdiv(fn) && (op1 == op2 || iszeronum(op2)) # look for multiplicative inverse x/x = 1
        root = 1
    elseif (fn == (*) || isdiv(fn)) &&  (iszeronum(op1) || iszeronum(op2)) # look for 0: x*0 = 0*x = 0
        root = 0
    elseif (fn == (+) || fn == (-)) && iszeronum(op2) # look for 0: x ± 0 = x
        root = op1
    elseif fn == (+) && iszeronum(op1) # look for 0: 0 + x = x
        root = op2
    elseif (fn == (*) || isdiv(fn)) && isonenum(op2) # x*1 = x || x/1 = x
        root = op1
    elseif fn == (*) && isonenum(op1) # 1*x = x
        root = op2
    elseif (fn == (+)) && (op1 == op2) # x+x = 2x
        root.args[1] = (*)
        root.args[2] = 2
    elseif (fn == (+) || fn == (-))
        # n1+(n2+x) = n1+(x+n2) = (n2+x)+n1 = (x+n2)+n1 = x+n3, s.t. n3=n1+n2
        if (isexpr(op1) && isnum(op2)) || (isnum(op1) && isexpr(op2))
            # swap so op1 is expr
            if isnum(op1) && isexpr(op2)
                op1, op2 =  op2, op1
            end
            #println("ex1: $fn ($op1, $op2)"
            # op2 is binexpr
            if isexpr(op1) && length(op1) == 3
                fn2, op21, op22 = op1.args
                # some operand has to be num
                if (fn2 == (+) || fn2 == (-)) && (isnum(op21) || isnum(op22))
                    var, n2 = isnum(op21) ? (op22, op21) : (op21, op22)
                    n3 = fn(n2, op2)
                    root.args[1] = fn2
                    root.args[2] = var
                    root.args[3] = n3
                end
            end
        end
    end

    # evaluate numerical arithmetic
    isnum(op1) && isnum(op2) && return fn(op1, op2)

    return root
end

"""
    simplify!(expr)

Simplify an algebraic expression
"""
function simplify!(root)
    # return if not an expression
    !isa(root, Expr) && return root
    # return if not a function
    root.head != :call && return root

    n = length(root.args)
    # recursively simplify arguments
    for i in 2:n
        !isexpr(root.args[i]) && continue
        root.args[i] = simplify!(root.args[i])
    end

    n == 2 && return simplifyunary!(root)
    n == 3 && return simplifybinary!(root)
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
            for (i, ex) in enumerate(root.args[2:end])
                i > 1 && print(io, ", ")
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

