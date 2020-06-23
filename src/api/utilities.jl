#########
# TRACE #
#########

function update!(tr::OptimizationTrace{T,O}, state::S,
                 iteration::Integer, val::T, dt::Dict,
                 store_trace::Bool, show_trace::Bool, show_every::Int = 1,
                 callback = nothing) where {S<:AbstractOptimizerState, T, O}
    os = OptimizationTraceRecord{T,O}(iteration, val, dt)
    store_trace && push!(tr, os)
    if show_trace
        if iteration % show_every == 0
            show(os)
            print("\n")
            flush(stdout)
        end
    end
    if callback != nothing && (iteration % show_every == 0)
        if store_trace
            stopped = callback(tr)
        else
            stopped = callback(os)
        end
    else
        stopped = false
    end
    stopped
end

function trace!(tr, iteration, objfun, state, population, method, options, curr_time=time())
    dt = Dict{String,Any}()
    dt["time"] = curr_time
    # set additional trace value
    trace!(dt, objfun, state, population, method, options)
    update!(tr,
            state,
            iteration,
            value(objfun),
            dt,
            options.store_trace,
            options.show_trace,
            options.show_every,
            options.callback)
end

"""
    trace!(record::Dict{String,Any}, objfun, state, population, method, options)

Update the trace `record`. This function allows to supplement an additional information into the optimization algorithm trace by modifing a trace `record`. It can be overwiden by specifing particular parameter types.
"""
trace!(record::Dict{String,Any}, objfun, state, population, method, options) = ()


#########
# STATE #
#########

abschange(objfun::O, state::S) where {O<:AbstractObjective, S<:AbstractOptimizerState} =
    abschange(value(objfun), value(state))
abschange(curr::T, prev) where T = abs(curr - prev)
relchange(objfun::O, state::S) where {O<:AbstractObjective, S<:AbstractOptimizerState} =
    relchange(value(objfun), value(state))
relchange(curr::T, prev) where T = abs(curr - prev)/abs(curr)

function assess_convergence(objfun, state::AbstractOptimizerState, method, options::Options)
    converged = false

    if abschange(objfun, state) ≤ options.abstol
        converged = true
    end
    if relchange(objfun, state) ≤ options.reltol
        converged = true
    end

    return converged
end


##############
# POPULATION #
##############

"""
    initial_population(method, individual::AbstractVector)

Initialize population by replicating the `inividual` vector.
"""
initial_population(method::M, individual::I) where {M<:AbstractOptimizer, I<:AbstractVector} =
    [copy(individual) for i in 1:population_size(method)]

"""
    initial_population(method, individuals::AbstractVector{<:AbstractVector})

Initialize population from the collection of `inividuals` vectors.
"""
function initial_population(method::M, individuals::AbstractVector{I}) where {M<:AbstractOptimizer, I<:AbstractVector}
    n = population_size(method)
    @assert length(individuals) ==  n "Size of initial population must be $n"
    individuals
end

"""
    initial_population(method, individual::Function)

Initialize population from the `inividual` function which returns an individual object.
"""
initial_population(method::M, individualFunc::Function) where {M<:AbstractOptimizer} =
    [individualFunc() for i in 1:population_size(method)]

"""
    initial_population(method, individual::AbstractMatrix)

Initialize population by replicating the `inividual` matrix.
"""
function initial_population(method::M, individual::I) where {M<:AbstractOptimizer, I<:AbstractMatrix}
    [copy(individual) for i in 1:population_size(method)]
end

"""
    initial_population(method, bounds::ConstraintBounds)

Initialize a random population within the individual `bounds`.
"""
function initial_population(method::M, bounds::ConstraintBounds) where {M<:AbstractOptimizer}
    n = population_size(method)
    cn = nconstraints_x(bounds)
    indv = rand(cn, n)
    if length(bounds.eqx) > 0
        indv[bounds.eqx,:] .= bounds.valx
    end
    T = eltype(bounds)
    rngs = Dict(i=>zeros(T,2) for i in bounds.ineqx)
    for (j,v,s) in zip(bounds.ineqx, bounds.bx, bounds.σx)
        rngs[j][1] -= s*v
        if s > 0
            rngs[j][2] += v
        end
    end
    for i in 1:n
        for (j,(r,l)) in rngs
            indv[j,i] *= r
            indv[j,i] += l
        end
    end
    initial_population(method,  [collect(i) for i in  eachcol(indv)])
end
