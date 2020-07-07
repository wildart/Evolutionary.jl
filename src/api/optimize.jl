after_while!(objfun, state, method, options) = nothing

# Optimization interface
function optimize(f, lower, upper, method::M,
                  options::Options = Options(;default_options(method)...)
                 ) where {M<:AbstractOptimizer}
    bounds = ConstraintBounds(lower,upper,[],[])
    optimize(f, bounds, method, options)
end
function optimize(f, individual, method::M,
                  options::Options = Options(;default_options(method)...)
                 ) where {M<:AbstractOptimizer}
    optimize(f, NoConstraints(), individual, method, options)
end
function optimize(f, individual::ConstraintBounds, method::M,
                  options::Options = Options(;default_options(method)...)
                 ) where {M<:AbstractOptimizer}
    optimize(f, BoxConstraints(individual), individual, method, options)
end
function optimize(f, lower, upper, individual, method::M,
                  options::Options = Options(;default_options(method)...)
                 ) where {M<:AbstractOptimizer}
    optimize(f, BoxConstraints(lower,upper), individual, method, options)
end
function optimize(f, constraints::C, method::M,
                  options::Options = Options(;default_options(method)...)
                 ) where {M<:AbstractOptimizer, C<:AbstractConstraints}
    optimize(f, constraints, constraints.bounds, method, options)
end
function optimize(f, constraints::C, individual, method::M,
                  options::Options = Options(;default_options(method)...)
                 ) where {M<:AbstractOptimizer, C<:AbstractConstraints}
    population = initial_population(method, individual)
    @assert length(population) > 0 "Population is empty"
    objfun = NonDifferentiable(f, first(population))
    optimize(objfun, constraints, population, method, options)
end

function optimize(objfun::D, constraints::C, population::AbstractArray,
                  method::M, options::Options = Options(;default_options(method)...),
                  state = initial_state(method, options, objfun, population)
                 ) where {D<:AbstractObjective, C<:AbstractConstraints, M<:AbstractOptimizer}

    # setup trace
    tr = OptimizationTrace{typeof(value(objfun)), typeof(method)}()
    tracing = options.store_trace || options.show_trace || options.callback !== nothing

    # prepare iteration counter (used to make "initial state" trace entry)
    iteration = 0
    t0 = time()
    stopped, stopped_by_callback = false, false
    converged, counter_tol = false, 0 # tolerance convergence

    options.show_trace && print_header(method)
    trace!(tr, iteration, objfun, state, population, method, options, time()-t0)

    while !converged && !stopped && iteration < options.iterations
        iteration += 1

        # perform state update
        update_state!(objfun, constraints, state, population, method, iteration) && break

        # evaluate convergence
        converged = assess_convergence(objfun, state, method, options)

        # update the function value
        value!(objfun, minimizer(state))

        # check covergence persistence
        counter_tol = converged ? counter_tol+1 : 0
        converged = converged && (counter_tol > options.successive_f_tol)

        # custom state-based termination
        converged = converged || terminate(state)

        if tracing
            # update trace; callbacks can stop routine early by returning true
            stopped_by_callback = trace!(tr, iteration, objfun, state, population, method, options, time()-t0)
        end

        if stopped_by_callback
            stopped = true
        end
    end
    after_while!(objfun, state, method, options)

    return EvolutionaryOptimizationResults(
        method,
        minimizer(state),
        value(objfun),
        iteration,
        iteration == options.iterations,
        converged,
        options.abstol,
        tr,
        f_calls(objfun)
    )
end
