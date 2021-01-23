@testset "Genetic Programming" begin
    Random.seed!(9874984737486);
    pop = 10
    terms = Terminal[:x, :y, rand]
    funcs = Function[+,-,*,/]

    t = TreeGP(pop, terms, funcs, maxdepth=2)
    @test Evolutionary.population_size(t) == pop
    @test sort(Evolutionary.terminals(t)) == [:x, :y]
    @testset for (term, dim) in t.terminals
        @test dim == 1
    end
    @testset for (func, arity) in t.functions
        @test arity == 2
    end
    @test_skip summary(t) == "TreeGP[P=10,Parameter[x,y],Function[*, +, /, -]]"

    # population initialization
    popexp = Evolutionary.initial_population(t);
    @test length(popexp) == pop
    popexp = Evolutionary.initial_population(t, :(x + 1));
    @test popexp[1] == :(x + 1)

    # recursive helper functions
    Random.seed!(9874984737482)
    gt = rand(TreeGP(pop, terms, funcs, maxdepth=2, initialization=:grow), 3)
    @test Evolutionary.nodes(gt) < 15
    @test Evolutionary.height(gt) <= 3
    @test length(gt) < 15
    ft = rand(TreeGP(pop, terms, funcs, maxdepth=2, initialization=:full), 3)
    @test Evolutionary.nodes(ft) == 15
    @test Evolutionary.height(ft) == 3
    @test length(ft) == 15
    @test Evolutionary.depth(ft, :y) == 3
    ft[2] = :z
    @test Evolutionary.depth(ft, :z) == 3
    @test Evolutionary.depth(ft, ft) == 0
    @test Evolutionary.depth(ft, ft[3]) > 0
    @test Evolutionary.depth(ft, :w) == -1
    @test Evolutionary.evaluate([1.0, 2.0], :y, [:y, :z]) == 1.0
    copyto!(ft, gt)
    @test ft == gt
    @test Evolutionary.symbols(ft) == [:y, :x]

    # simplification
    @test Expr(:call, -, :x, :x) |> Evolutionary.simplify! == 0
    @test Expr(:call, /, :x, :x) |> Evolutionary.simplify! == 1
    @test Expr(:call, *, 0, :x) |> Evolutionary.simplify! == 0
    @test Expr(:call, *, :x, 0) |> Evolutionary.simplify! == 0
    @test Expr(:call, /, 0, :x) |> Evolutionary.simplify! == 0
    @test Expr(:call, /, 1, 0) |> Evolutionary.simplify! == 1
    @test Expr(:call, +, 0, :x) |> Evolutionary.simplify! == :x
    @test Expr(:call, +, :x, 0) |> Evolutionary.simplify! == :x
    @test Expr(:call, -, :x, 0) |> Evolutionary.simplify! == :x
    @test Expr(:call, +, :x, :x) |> Evolutionary.simplify! == Expr(:call, *, 2, :x)
    @test Expr(:call, -, 2, 1) |> Evolutionary.simplify! == 1

    # evaluation
    ex = Expr(:call, +, 1, :x) |> Evolutionary.Expression
    xs = rand(10)
    @test ex(xs[1]) == xs[1]+1
    @test ex.(xs) == xs.+1
    io = IOBuffer()
    show(io, "text/latex", ex)
    @test String(take!(io)) == "\\left(1.0+x\\right)"

    depth = 5
    fitfun(x) = x*x + x + 1.0
    function fitobj(expr)
        rg = -5.0:0.5:5.0
        ex = Evolutionary.Expression(expr)
        sum(v->isnan(v) ? 1.0 : v, abs2.(fitfun.(rg) - ex.(rg)) )/length(rg) |> sqrt
    end

    Random.seed!(9874984737426);
    res = Evolutionary.optimize(fitobj,
        TreeGP(50, Terminal[:x, randn], Function[+,-,*,/],
            mindepth=1,
            maxdepth=depth,
            optimizer = GA(
                selection = uniformranking(5),
                ɛ = 0.1,
                mutationRate = 0.8,
                crossoverRate = 0.2,
            ),
        )
    )
    @test minimum(res) < 1

end
