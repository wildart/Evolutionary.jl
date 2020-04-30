@testset "API" begin

    #########
    # SETUP #
    #########

    using Evolutionary: AbstractOptimizer, AbstractOptimizerState, Options, value!, f_calls, NonDifferentiable
    import Evolutionary: value, population_size, default_options, initial_state, update_state!

    # objectvive function
    dimension = 5
    individual = ones(dimension)
    minval = 1.0
    func = x->sum(x)+1
    objfun = NonDifferentiable(func, individual)

    # state
    mutable struct TestOptimizerState <: Evolutionary.AbstractOptimizerState
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
    function Evolutionary.update_state!(d, state::TestOptimizerState, population::AbstractVector, method)
        i = rand(1:population_size(method))
        state.individual = population[i]
        state.fitness = value!(d, state.individual)
        return false
    end
    mthd = TestOptimizer()

    # options
    opts = Evolutionary.Options(store_trace=true; default_options(mthd)...)

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
    @test !update_state!(objfun, st, ppl, mthd)
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

    @test_throws AssertionError Evolutionary.initial_population(mthd, BitMatrix(ones(dimension,4)))
    pop = Evolutionary.initial_population(mthd, BitMatrix(ones(dimension,6)))
    @test length(pop) == population_size(mthd)

    pop = Evolutionary.initial_population(mthd, (()->rand(Bool,dimension)))
    @test length(pop) == population_size(mthd)

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
