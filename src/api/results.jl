"""
Abstract evolutionary optimization result type
"""
abstract type OptimizationResults end

"""
    summary(result)

Shows the optimization algorithm that produced this `result`.
"""
summary(or::OptimizationResults) = summary(or.method)

"""
    minimizer(result)

A minimizer object from the optimization `result`.
"""
minimizer(r::OptimizationResults) = r.minimizer

"""
    minimum(result)

A minimum value from the optimization `result`.
"""
minimum(r::OptimizationResults) = r.minimum

"""
    iterations(result)

A number of iterations to reach the minimum.
"""
iterations(r::OptimizationResults) = r.iterations

"""
    iteration_limit_reached(result)

Returns `true` if the iteration limit was reached.
"""
iteration_limit_reached(r::OptimizationResults) = r.iteration_converged

"""
    trace(result)

Returns a trace of optimization states from the optimization `result`.
"""
trace(r::OptimizationResults) = length(r.trace) > 0 ? r.trace : error("No trace in optimization results. To get a trace, run optimize() with store_trace = true.")

"""
    f_calls(result)

Returns a number of an objective function calls.
"""
f_calls(r::OptimizationResults) = r.f_calls

"""
    tol(result)

Returns an absolute tollerance value of the optimization `result`.
"""
tol(r::OptimizationResults) = error("tol is not implemented for $(summary(r)).")

"""
Evolutionary optimization result type
"""
mutable struct EvolutionaryOptimizationResults{O<:AbstractOptimizer, T, Tx, Tf} <: OptimizationResults
    method::O
    minimizer::Tx
    minimum::Tf
    iterations::Int
    iteration_converged::Bool
    converged::Bool
    abstol::T
    trace::OptimizationTrace
    f_calls::Int
end

"""
    converged(result)

Returns `true` if the optimization sucesfully coverged to a minimum value.
"""
converged(r::EvolutionaryOptimizationResults) = r.converged

tol(r::EvolutionaryOptimizationResults) = r.abstol

function show(io::IO, r::EvolutionaryOptimizationResults)
    failure_string = "failure"
    if iteration_limit_reached(r)
        failure_string *= " (reached maximum number of iterations)"
    end
    print(io, "\n")
    print(io, " * Status: ", converged(r) ? "success" : failure_string, "\n\n")
    print(io, " * Candidate solution\n")
    nx = length(minimizer(r))
    str_x_elements = ["$_x" for _x in Iterators.take(minimizer(r), min(nx, 3))]
    if nx >= 4
        push!(str_x_elements, " ...")
    end

    print(io, "    Minimizer:  [", join(str_x_elements, ", "),  "]\n")
    print(io, "    Minimum:    $(minimum(r))\n")
    print(io, "    Iterations: $(iterations(r))\n")
    print(io, "\n")
    print(io, " * Found with\n")
    print(io, "    Algorithm: $(summary(r))\n")
    return
end
