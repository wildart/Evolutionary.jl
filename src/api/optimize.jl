# Optimization interface

"""
    optimize(f[, F], indiv, algo[, opts])
    optimize(f[, F], constr, algo[, opts])
    optimize(f[, F], constr, indiv, algo[, opts])
    optimize(f[, F], constr, algo, poplt[, opts])

Perform optimization of the function `f` using the alorithm `algo` with the population, composed of
the initial population `poplt`, or individuals similar to the original individual `indiv`,
or generated from the constraints `constr`, with the options `opts`.
- For multi-objective optimization, the objective value `F` *must* be provided.
"""
optimize(f, individual, method::M,
         opts::Options = Options(;default_options(method)...)) where {M<:AbstractOptimizer} =
    optimize(f, NoConstraints(), individual, method, opts)
optimize(f, F::AbstractVector, individual, method::M,
         opts::Options = Options(;default_options(method)...)) where {M<:AbstractOptimizer} =
    optimize(f, F, NoConstraints(), individual, method, opts)
optimize(f, bounds::ConstraintBounds, method::M,
         opts::Options = Options(;default_options(method)...)) where {M<:AbstractOptimizer} =
    optimize(f, BoxConstraints(bounds), method, opts)
optimize(f, F, bounds::ConstraintBounds, method::M,
         opts::Options = Options(;default_options(method)...)) where {M<:AbstractOptimizer} =
    optimize(f, F, BoxConstraints(bounds), method, opts)
function optimize(f, constraints::C, method::M,
                  opts::Options = Options(;default_options(method)...)
                  ) where {M<:AbstractOptimizer, C<:AbstractConstraints}
    population = initial_population(method, bounds(constraints), rng=opts.rng)
    optimize(f, constraints, method, population, opts)
end
function optimize(f, F, constraints::C, method::M,
                  opts::Options = Options(;default_options(method)...)
                  ) where {M<:AbstractOptimizer, C<:AbstractConstraints}
    population = initial_population(method, bounds(constraints), rng=opts.rng)
    optimize(f, F, constraints, method, population, opts)
end
function optimize(f, constraints::C, individual, method::M,
                  opts::Options = Options(;default_options(method)...)
                  ) where {M<:AbstractOptimizer, C<:AbstractConstraints}
    population = initial_population(method, individual, rng=opts.rng)
    optimize(f, constraints, method, population, opts)
end
function optimize(f, F, constraints::C, individual, method::M,
                  opts::Options = Options(;default_options(method)...)
                  ) where {M<:AbstractOptimizer, C<:AbstractConstraints}
    population = initial_population(method, individual, rng=opts.rng)
    optimize(f, F, constraints, method, population, opts)
end
function optimize(f, constraints::C, method::M, population,
                  opts::Options = Options(;default_options(method)...)
                 ) where {M<:AbstractOptimizer, C<:AbstractConstraints}
    @assert length(population) > 0 "Population is empty"
    objfun = EvolutionaryObjective(f, first(population); eval=opts.parallelization)
    optimize(objfun, constraints, method, population, opts)
end
function optimize(f, F, constraints::C, method::M, population,
                  opts::Options = Options(;default_options(method)...)
                 ) where {M<:AbstractOptimizer, C<:AbstractConstraints}
    @assert length(population) > 0 "Population is empty"
    objfun = EvolutionaryObjective(f, first(population), F; eval=opts.parallelization)
    optimize(objfun, constraints, method, population, opts)
end

function optimize(objfun::D, constraints::C, method::M, population::AbstractArray,
                  options::Options = Options(;default_options(method)...),
                  state = initial_state(method, options, objfun, population)
                 ) where {D<:AbstractObjective, C<:AbstractConstraints, M<:AbstractOptimizer}
    # setup trace
    tr = OptimizationTrace{typeof(value(state)), typeof(method)}()
    tracing = options.store_trace || options.show_trace || options.callback !== nothing

    # prepare iteration counter (used to make "initial state" trace entry)
    iteration = 0
    t0 = time()
    stopped, stopped_by_callback = false, false
    converged, counter_tol = false, 0 # tolerance convergence
    is_moo = ismultiobjective(objfun)

    options.show_trace && print_header(method)
    trace!(tr, iteration, objfun, state, population, method, options, time()-t0)

    _time = time()
    while !converged && !stopped && iteration < options.iterations
        iteration += 1

        # perform state update
        update_state!(objfun, constraints, state, population, method, options, iteration) && break

        # evaluate convergence
        converged = assess_convergence(state, method)

        # check convergence persistence
        counter_tol = converged ? counter_tol+1 : 0
        converged = converged && (counter_tol > options.successive_f_tol)

        # custom state-based termination
        converged = converged || terminate(state)

        if tracing
            # update trace; callbacks can stop routine early by returning true
            stopped_by_callback = trace!(tr, iteration, objfun, state, population, method, options, time()-t0)
        end

        _time = time()
        stopped_by_time_limit = _time-t0 > options.time_limit

        if stopped_by_callback || stopped_by_time_limit
            stopped = true
        end
    end

    after_while!(objfun, state, method, options)

    return EvolutionaryOptimizationResults(
        method,
        minimizer(state),
        is_moo ? NaN : value(state),
        iteration,
        iteration == options.iterations,
        converged,
        metrics(method),
        tr,
        f_calls(objfun),
        options.time_limit,
        _time-t0,
        is_moo,
    )
end

after_while!(objfun, state, method, options) = nothing
