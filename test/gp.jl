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
    @test summary(t) == "TreeGP[P=10,Parameter[x,y],Function[*, +, /, -]]"

    popexp = Evolutionary.initial_population(t);
    @test length(popexp) == pop

    Random.seed!(9874984737484)
    ft = rand(t, method=:full)
    @test Evolutionary.nodes(ft) == 7
    @test Evolutionary.height(ft) == 2
    gt = rand(t, method=:grow)
    @test Evolutionary.nodes(gt) == 5
    @test Evolutionary.height(gt) == 2
    @test gt[0] == gt
    @test gt[1] == :x
    @test gt[2] == :x
    @test gt[4] == Expr(:call, /, :x, :x)


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
    @test Expr(:call, +, 1, Expr(:call, -, 1, :x) ) |> Evolutionary.simplify! == Expr(:call, -, 2, :x)
    @test Expr(:call, -, 1, Expr(:call, +, 1, :x) ) |> Evolutionary.simplify!  == Expr(:call, +, 0, :x)
    @test Expr(:call, *, Expr(:call, *, :x, :y), Expr(:call, /, :z, :x)) |> Evolutionary.simplify! == Expr(:call, *, :y, :z)
    @test Expr(:call, *, Expr(:call, *, :x, :y), Expr(:call, /, :z, :y)) |> Evolutionary.simplify! == Expr(:call, *, :x, :z)
    @test Expr(:call, *, Expr(:call, /, :z, :x), Expr(:call, *, :x, :y)) |> Evolutionary.simplify! == Expr(:call, *, :z, :y)
    @test Expr(:call, *, Expr(:call, /, :z, :y), Expr(:call, *, :x, :y)) |> Evolutionary.simplify! == Expr(:call, *, :z, :x)
    @test Expr(:call, *, Expr(:call, *, 2, :y), Expr(:call, /, :z, 2)) |> Evolutionary.simplify! == Expr(:call, *, :y, :z)
    @test Expr(:call, *, Expr(:call, *, :y, 2), Expr(:call, /, :z, 2)) |> Evolutionary.simplify! ==  Expr(:call, *, :y, :z)
    @test Expr(:call, *, Expr(:call, /, :z, 2), Expr(:call, *, 2, :y)) |> Evolutionary.simplify! == Expr(:call, *, :z, :y)
    @test Expr(:call, *, Expr(:call, /, :z, 2), Expr(:call, *, :y, 2)) |> Evolutionary.simplify! ==  Expr(:call, *, :z, :y)
    @test Expr(:call, +, Expr(:call, -, :x, 1), 1 ) |> Evolutionary.simplify! == Expr(:call, -, :x, 2)
    @test Expr(:call, -, Expr(:call, +, :x, 1), 1 ) |> Evolutionary.simplify! == Expr(:call, +, :x, 0)

    depth = 5
    fitfun(x) = x*x + x + 1.0
    function fitobj(expr)
        rg = -5.0:0.5:5.0
        sum(v->isnan(v) ? 1.0 : v, (abs(fitfun(x) - Evolutionary.evaluate(expr, [:x], [x])) for x in rg))
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
    @test minimum(res) < 21

end
