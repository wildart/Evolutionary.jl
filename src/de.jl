"""
Implementation of Differential Evolution: DE/**selection**/n/**recombination**

The constructor takes following keyword arguments:

- `populationSize`: The size of the population
- `F`: the differentiation (mutation) scale factor (default: 0.9). It's usually defined in range ``F \\in (0, 1+]``
- `n`: the number of differences used in the perturbation (default: 1)
- `selection`: the selection strategy function (default: [`random`](@ref))
- `recombination`: the recombination functions (default: [`uniformbin(0.5)`](@ref))
- `K`: the recombination scale factor (default: 0.5*(F+1))
"""
@kwdef struct DE <: AbstractOptimizer
    populationSize::Integer = 50
    F::Real = 0.9
    n::Integer = 1
    K::Real = 0.5*(F+1)
    selection::Function = random
    recombination::Function = uniformbin(0.5)
end
population_size(method::DE) = method.populationSize
default_options(method::DE) = (abstol=1e-10,)
summary(m::DE) = "DE/$(m.selection)/$(m.n)/$(m.recombination)"

mutable struct DEState{T,IT} <: AbstractOptimizerState
    N::Int
    fitness::Vector{T}
    fittest::IT
end
value(s::DEState) = minimum(s.fitness)
minimizer(s::DEState) = s.fittest

"""Initialization of ES algorithm state"""
function initial_state(method::DE, options, objfun, population)
    T = typeof(value(objfun))
    individual = first(population)
    N = length(individual)
    fitness = fill(maxintfloat(T), method.populationSize)

    # setup initial state
    return DEState(N, fitness, copy(individual))
end

function update_state!(objfun, constraints, state, population::AbstractVector{IT}, method::DE, itr) where {IT}

    # setup
    Np = method.populationSize
    n = method.n
    F = method.F

    offspring = Array{IT}(undef, Np)

    # select base vectors
    bases = method.selection(state.fitness, Np)

    # select target vectors
    for (i,b) in enumerate(bases)
        # mutation
        base = population[b]
        offspring[i] = copy(base)
        # println("$i => base:", offspring[i])

        targets = randexcl(1:Np, [i], 2*n)
        offspring[i] = differentiation(offspring[i], @view population[targets]; F=F)
        # println("$i => mutated:", offspring[i], ", targets:", targets)

        # recombination
        offspring[i], _ = method.recombination(offspring[i], base)
        # println("$i => recombined:", offspring[i])
    end

    # Create new generation
    fitidx = 0
    minfit = Inf
    for i in 1:Np
        o = apply!(constraints, offspring[i])
        v = value(objfun, o) + penalty(constraints, o)
        if (v <= state.fitness[i])
            population[i] = o
            state.fitness[i] = v
            if v < minfit
                minfit = v
                fitidx = i
            end
        end
    end

    # set best individual
    if fitidx > 0
        state.fittest = population[fitidx]
    end

    return false
end
