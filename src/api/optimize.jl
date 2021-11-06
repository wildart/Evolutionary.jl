after_while!(objfun, state, method, options) = nothing

# Optimization interface
function optimize(f, lower, upper, method::M,
                  options::Options = Options(;default_options(method)...); kwargs...
                 ) where {M<:AbstractOptimizer}
    bounds = ConstraintBounds(lower,upper,[],[])
    optimize(f, bounds, method, options)
end

"""
    optimize(f, indiv, algo)

Perform optimization of the function `f` using aloruthm `algo` with the population composed of
individuals similar to the original individual `indiv`.
"""
function optimize(f, individual, method::M,
                  options::Options = Options(;default_options(method)...); kwargs...
                 ) where {M<:AbstractOptimizer}
    optimize(f, NoConstraints(), individual, method, options; kwargs...)
end

function optimize(f, individual::ConstraintBounds, method::M,
                  options::Options = Options(;default_options(method)...); kwargs...
                 ) where {M<:AbstractOptimizer}
    optimize(f, BoxConstraints(individual), individual, method, options; kwargs...)
end
function optimize(f, lower, upper, individual, method::M,
                  options::Options = Options(;default_options(method)...); kwargs...
                 ) where {M<:AbstractOptimizer}
    optimize(f, BoxConstraints(lower,upper), individual, method, options; kwargs...)
end
function optimize(f, constraints::C, method::M,
                  options::Options = Options(;default_options(method)...); kwargs...
                 ) where {M<:AbstractOptimizer, C<:AbstractConstraints}
    optimize(f, constraints, constraints.bounds, method, options; kwargs...)
end
function optimize(f, constraints::C, individual, method::M,
                  options::Options = Options(;default_options(method)...); kwargs...
                 ) where {M<:AbstractOptimizer, C<:AbstractConstraints}
    population = initial_population(method, individual)
    @assert length(population) > 0 "Population is empty"
    val = first(population)
    objfun = try
        nd = NonDifferentiable(f, val)
        value(nd, val)
        nd
    catch
        params = Dict(kwargs...)
        @assert getkey(params, :F, nothing) !== nothing "Specify a sample of a multi-objective function return value in the `F` parameter."
        try
            nd = NonDifferentiable(f, val, params[:F])
            value(nd, val)
            nd
        catch
            error("Multi-objective function must have two parameters")
        end
    end
    optimize(objfun, constraints, population, method, options; kwargs...)
end

function optimize(objfun::D, constraints::C, population::AbstractArray,
                  method::M, options::Options = Options(;default_options(method)...),
                  state = initial_state(method, options, objfun, population); kwargs...
                 ) where {D<:AbstractObjective, C<:AbstractConstraints, M<:AbstractOptimizer}

    # setup trace
    tr = OptimizationTrace{typeof(value(objfun)), typeof(method)}()
    tracing = options.store_trace || options.show_trace || options.callback !== nothing

    # prepare iteration counter (used to make "initial state" trace entry)
    iteration = 0
    t0 = time()
    stopped, stopped_by_callback = false, false
    converged, counter_tol = false, 0 # tolerance convergence
    is_moo = ismmo(objfun)

    options.show_trace && print_header(method)
    trace!(tr, iteration, objfun, state, population, method, options, time()-t0)

    _time = time()
    while !converged && !stopped && iteration < options.iterations
        iteration += 1

        # perform state update
        update_state!(objfun, constraints, state, population, method, options, iteration) && break

        # evaluate convergence
        converged = assess_convergence(objfun, state, method, options)

        # update the function value
        !is_moo && value!(objfun, minimizer(state))

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
        is_moo ? NaN : value(objfun),
        iteration,
        iteration == options.iterations,
        converged,
        options.abstol,
        options.reltol,
        abschange(objfun, state),
        relchange(objfun, state),
        tr,
        f_calls(objfun),
        options.time_limit,
        _time-t0,
    )
end
