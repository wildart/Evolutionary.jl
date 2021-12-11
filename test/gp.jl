@testset "Genetic Programming" begin
    rng = StableRNG(42)

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
    popexp = Evolutionary.initial_population(t, rng=rng);
    @test length(popexp) == pop
    popexp = Evolutionary.initial_population(t, :(x + 1));
    @test popexp[1] == :(x + 1)

    # recursive helper functions
    gtr = TreeGP(pop, terms, funcs, maxdepth=2, initialization=:grow)
    Random.seed!(rng, 1)
    tmp = rand(rng, gtr, 3)
    Random.seed!(rng, 1)
    gt = rand(rng, gtr, 3)
    @test tmp == gt
    @test Evolutionary.nodes(gt) < 15
    @test Evolutionary.height(gt) <= 3
    @test length(gt) < 15
    ft = rand(rng, TreeGP(pop, terms, funcs, maxdepth=2, initialization=:full), 3)
    @test Evolutionary.nodes(ft) == 15
    @test Evolutionary.height(ft) == 3
    @test length(ft) == 15
    # @test Evolutionary.depth(ft, :x) == 3
    # ft[3] = :z
    # @test Evolutionary.depth(ft, :z) == 3
    @test Evolutionary.depth(ft, ft) == 0
    @test Evolutionary.depth(ft, ft[3]) > 0
    @test Evolutionary.depth(ft, :w) == -1
    @test Evolutionary.evaluate([1.0, 2.0], :y, [:y, :z]) == 1.0
    copyto!(ft, gt)
    @test ft == gt
    # @test Evolutionary.symbols(ft) |> sort == [:x, :y]

    # simplification
    using Evolutionary: simplify!
    @test Expr(:call, -, :x, :x) |> simplify! == 0
    @test Expr(:call, /, :x, :x) |> simplify! == 1
    @test Expr(:call, *, 0, :x) |> simplify! == 0
    @test Expr(:call, *, :x, 0) |> simplify! == 0
    @test Expr(:call, /, 0, :x) |> simplify! == 0
    @test Expr(:call, /, 0,  1) |> simplify! == 0
    @test Expr(:call, +, 0, :x) |> simplify! == :x
    @test Expr(:call, +, :x, 0) |> simplify! == :x
    @test Expr(:call, -, :x, 0) |> simplify! == :x
    @test Expr(:call, +, :x, :x) |> simplify! == Expr(:call, *, 2, :x)
    @test Expr(:call, -, 2, 1) |> simplify! == 1
    @test Expr(:call, exp, Expr(:call, log, 1)) |> simplify! == 1
    @test Expr(:call, log, Expr(:call, exp, 1)) |> simplify! == 1
    @test Expr(:call, -, Expr(:call, +, :x, 1), 2) |> simplify! == Expr(:call, +, :x, -1)
    @test Expr(:call, -, Expr(:call, +, 1, :x), 2) |> simplify! == Expr(:call, +, :x, -1)
    @test Expr(:call, +, 2, Expr(:call, +, 1, :x)) |> simplify! == Expr(:call, +, :x, 3)
    @test Expr(:call, +, 2, Expr(:call, +, :x, 1)) |> simplify! == Expr(:call, +, :x, 3)

    # evaluation
    ex = Expr(:call, +, 1, :x) |> Evolutionary.Expression
    xs = rand(rng, 10)
    @test ex(xs[1]) == xs[1]+1
    @test ex.(xs) == xs.+1
    io = IOBuffer()
    show(io, "text/latex", ex)
    @test String(take!(io)) == "\\left(1.0+x\\right)"

    # symreg
    fitfun(x) = x*x + x + 1.0
    rg = -5.0:0.1:5.0
    ys = fitfun.(rg)
    function fitobj(expr)
        ex = Evolutionary.Expression(expr)
        sum(v->isnan(v) ? 1.0 : v, abs2.(ys - ex.(rg)) )/length(rg)
    end

    Random.seed!(rng, 42)
    res = Evolutionary.optimize(fitobj,
        TreeGP(25, Terminal[:x, randn], Function[+,-,*,Evolutionary.aq],
            mindepth=1,
            maxdepth=3,
            simplify = simplify!,
            optimizer = GA(
                selection = tournament(3),
                mutationRate = 0.2,
                crossoverRate = 0.9,
            ),
        ),
        Evolutionary.Options(show_trace=false, rng=rng, iterations=50)
    )
    @test minimum(res) < 1.1

end
