@testset "API" begin

    #########
    # SETUP #
    #########

    using Evolutionary: AbstractOptimizer, AbstractOptimizerState, Options, value!,
                        f_calls, NonDifferentiable, BoxConstraints, PenaltyConstraints
    import Evolutionary: value, population_size, default_options, initial_state, update_state!

    # objectvive function
    dimension = 5
    individual = ones(dimension)
    minval = 1.0
    func = x->sum(x)+1
    objfun = NonDifferentiable(func, individual)

    # state
    mutable struct TestOptimizerState <: AbstractOptimizerState
        individual
        fitness
    end
    Evolutionary.value(state::TestOptimizerState) = state.fitness
    Evolutionary.minimizer(state::TestOptimizerState) = state.individual
    st = TestOptimizerState(individual, minval);

    # optimizer
    struct TestOptimizer <: AbstractOptimizer end
    Evolutionary.population_size(method::TestOptimizer) = 5
    Evolutionary.default_options(method::TestOptimizer) = Dict(:iterations=>10, :abstol=>1e-10)
    Evolutionary.initial_state(method, options, d, population) = TestOptimizerState(population[end], value(d, population[end]))
    function Evolutionary.update_state!(d, constraints, state::TestOptimizerState, population::AbstractVector, method, itr)
        i = rand(1:population_size(method))
        state.individual = population[i]
        state.fitness = value!(d, state.individual)
        return false
    end
    mthd = TestOptimizer()
    cnstr= Evolutionary.NoConstraints()

    # options
    opts = Evolutionary.Options(store_trace=true; default_options(mthd)...)
    show(IOBuffer(), opts)

    # population
    ppl_size = 7
    ppl = [rand(dimension) for i in 1:ppl_size]

    #########
    # TRACE #
    #########
    @test value(st) == minval
    val = value!(objfun, rand(dimension))
    @test Evolutionary.abschange(objfun, st) == val - minval
    @test Evolutionary.relchange(objfun, st) == (val - minval)/val

    tr = Evolutionary.OptimizationTrace{typeof(value(objfun)), typeof(mthd)}()
    @test !Evolutionary.trace!(tr, 1, objfun, st, ppl, mthd, opts)
    @test tr[1].iteration == 1
    @test tr[1].value == val
    @test tr[1].metadata["time"] <= time()
    @test !update_state!(objfun, cnstr, st, ppl, mthd, 10)
    @test !Evolutionary.trace!(tr, 2, objfun, st, ppl, mthd, opts)
    @test tr[2].iteration == 2
    @test tr[2].value >= 1
    @test tr[2].metadata["time"] >= tr[1].metadata["time"]
    show(IOBuffer(), tr)

    # callback on state
    opts = Evolutionary.Options(store_trace=false, callback=(s->s.iteration == 2))
    tr = Evolutionary.OptimizationTrace{typeof(value(objfun)), typeof(mthd)}()
    @test Evolutionary.trace!(tr, 1, objfun, st, ppl, mthd, opts) == false
    @test Evolutionary.trace!(tr, 2, objfun, st, ppl, mthd, opts) == true

    # callback on trace
    opts = Evolutionary.Options(store_trace=true, callback=(tr->length(tr)>1))
    tr = Evolutionary.OptimizationTrace{typeof(value(objfun)), typeof(mthd)}()
    @test Evolutionary.trace!(tr, 1, objfun, st, ppl, mthd, opts) == false
    @test Evolutionary.trace!(tr, 2, objfun, st, ppl, mthd, opts) == true

    # custom trace
    function Evolutionary.trace!(record::Dict{String,Any}, objfun, state::TestOptimizerState, population, method, options)
        record["ppl_size"] = length(population)
    end
    tr = Evolutionary.OptimizationTrace{typeof(value(objfun)), typeof(mthd)}()
    @test !Evolutionary.trace!(tr, 1, objfun, st, ppl, mthd, opts)
    @test tr[1].iteration == 1
    @test tr[1].metadata["ppl_size"] == length(ppl)


    ##########
    # RESULT #
    ##########
    res = Evolutionary.EvolutionaryOptimizationResults(
        mthd, individual, value(objfun),
        opts.iterations, opts.show_trace, opts.store_trace,
        opts.abstol, tr, f_calls(objfun)
    );
    @test summary(res) == summary(mthd)
    @test Evolutionary.minimizer(res) == individual
    @test minimum(res) == value(objfun)
    @test Evolutionary.iterations(res) == opts.iterations
    @test Evolutionary.iteration_limit_reached(res) == opts.show_trace
    @test Evolutionary.tol(res) == opts.abstol
    @test Evolutionary.converged(res) == opts.store_trace
    @test f_calls(res) == f_calls(objfun)
    @test length(Evolutionary.trace(res)) >= 1
    show(IOBuffer(), res)


    ##############
    # POPULATION #
    ##############
    pop = Evolutionary.initial_population(mthd, BitVector(ones(dimension)))
    @test length(pop) == population_size(mthd)
    @test size(first(pop)) == (dimension,)

    @test_throws AssertionError Evolutionary.initial_population(mthd, fill(BitVector(ones(dimension)),4))
    pop = Evolutionary.initial_population(mthd, fill(BitVector(ones(dimension)),5))
    @test length(pop) == population_size(mthd)
    @test size(first(pop)) == (dimension,)

    pop = Evolutionary.initial_population(mthd, BitMatrix(ones(dimension,6)))
    @test length(pop) == population_size(mthd)
    @test size(first(pop)) == (dimension,6)

    pop = Evolutionary.initial_population(mthd, (()->rand(Bool,dimension)))
    @test length(pop) == population_size(mthd)
    @test size(first(pop)) == (dimension,)

    lb = [0,1,2,-1]
    ub = [0,3,2,1]
    cb = Evolutionary.ConstraintBounds(lb,ub,[],[])
    pop = Evolutionary.initial_population(mthd, cb)
    @test length(pop) == population_size(mthd)
    @test map(i->i[1],pop) == fill(ub[1], population_size(mthd))
    @test all(map(i->lb[2] <= i[2] <= ub[2],pop))
    @test map(i->i[3],pop) == fill(ub[3], population_size(mthd))
    @test all(map(i->lb[4] <= i[4] <= ub[4],pop))

    ###########
    # OPTIONS #
    ###########
    dopts = default_options(mthd)
    opts = Options(;store_trace=true, dopts...)
    @test opts.abstol == dopts[:abstol]

    #############
    # OBJECTIVE #
    #############
    objfun= NonDifferentiable(sum, BitVector(ones(dimension)))
    @test value(objfun) == 0.0
    @test 0 <= value(objfun,rand(Bool,dimension)) <= dimension
    @test f_calls(objfun) == 1
    @test 0 <= value!(objfun,rand(Bool,dimension)) <= dimension
    @test f_calls(objfun) == 2
    v = value!(objfun,rand(Bool,dimension))
    @test 0 <= v <= dimension
    @test value(objfun) == v


    ###############
    # CONSTRAINTS #
    ###############
    c = Evolutionary.NoConstraints()
    @test isfeasible(c, 1)
    @test isa(c.bounds, Evolutionary.ConstraintBounds)
    @test value(c, 1) === nothing
    @test apply!(c, 1) == 1
    @test penalty(c, 1) == 0

    c = BoxConstraints(lb, ub)
    @test isfeasible(c, [0, 1, 2, 0])
    @test value(c, [1,4,2,-2]) === nothing
    @test apply!(c, [1,4,2,-2]) == [0,3,2,-1]
    @test penalty(c, [1,4,2,-2]) == 0

    con_c!(x) = [sum(x)]
    cb = Evolutionary.ConstraintBounds(fill(0.0, 3), fill(1.0, 3), [1.0], [1.0])
    c = PenaltyConstraints(1.0, cb, con_c!)
    objfun = NonDifferentiable(sum, zeros(3))
    x, y = [0, 1, 0], [0, -1, 3]
    @test isfeasible(c, x)    # feasible
    @test !isfeasible(c, y) # not feasible
    @test apply!(c, x) == x
    @test apply!(c, y) == y
    @test value(c, x) == [sum(x)]
    @test value(c, y) == [sum(y)]
    @test penalty(c, x) == 0.0 # c penalty
    @test penalty(c, y) == 1+2^2+1.0 # c penalty
    @test penalty!([1.0, 2.0], c, [x,y]) == [1, 2 + penalty(c, y)]


    ############
    # OPTIMIZE #
    ############
    opts = Options(;store_trace=true, iterations=7, abstol=-1.0, reltol=-1.0)
    res =  Evolutionary.optimize(sum, (()->BitVector(rand(Bool,dimension))), mthd, opts)
    @test Evolutionary.minimum(res) <= dimension
    @test length(Evolutionary.minimizer(res)) == dimension
    @test Evolutionary.iterations(res) == 7
    @test !Evolutionary.converged(res)
    @test length(Evolutionary.trace(res)) > 1

end
