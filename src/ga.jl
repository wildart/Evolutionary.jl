"""
Implementation of Genetic Algorithm

The constructor takes following keyword arguments:

- `populationSize`: The size of the population
- `crossoverRate`: The fraction of the population at the next generation, not including elite children, that is created by the crossover function.
- `mutationRate`: Probability of chromosome to be mutated
- `ɛ`/`epsilon`: Positive integer specifies how many individuals in the current generation are guaranteed to survive to the next generation. Floating number specifies fraction of population.
- `selection`: [Selection](@ref) function
- `crossover`: [Crossover](@ref) function (default: `identity`)
- `mutation`: [Mutation](@ref) function (default: `identity`)
"""
struct GA <: AbstractOptimizer
    populationSize::Int
    crossoverRate::Float64
    mutationRate::Float64
    ɛ::Real
    selection::Function
    crossover::Function
    mutation::Function

    GA(; populationSize::Int=50, crossoverRate::Float64=0.8, mutationRate::Float64=0.1,
        ɛ::Real=0, epsilon::Real=ɛ,
        selection::Function = ((x,n)->1:n),
        crossover::Function = identity, mutation::Function = identity) =
        new(populationSize, crossoverRate, mutationRate, epsilon, selection, crossover, mutation)
end
population_size(method::GA) = method.populationSize
default_options(method::GA) = (iterations=1000, abstol=1e-15)
summary(m::GA) = "GA[P=$(m.populationSize),x=$(m.crossoverRate),μ=$(m.mutationRate),ɛ=$(m.ɛ)]"

mutable struct GAState{T,IT} <: AbstractOptimizerState
    N::Int
    eliteSize::Int
    fitness::T
    fitpop::Vector{T}
    fittest::IT
end
value(s::GAState) = s.fitness
minimizer(s::GAState) = s.fittest

"""Initialization of GA algorithm state"""
function initial_state(method::GA, options, objfun, population)
    T = typeof(value(objfun))
    N = length(first(population))
    fitness = zeros(T, method.populationSize)

    # setup state values
    eliteSize = isa(method.ɛ, Int) ? method.ɛ : round(Int, method.ɛ * method.populationSize)

    # Evaluate population fitness
    fitness = map(i -> value(objfun, i), population)
    minfit, fitidx = findmin(fitness)

    # setup initial state
    return GAState(N, eliteSize, minfit, fitness, copy(population[fitidx]))
end

function update_state!(objfun, constraints, state, population::AbstractVector{IT}, method::GA, itr) where {IT}
    @unpack populationSize,crossoverRate,mutationRate,ɛ,selection,crossover,mutation = method

    offspring = similar(population)

    # Select offspring
    selected = selection(state.fitpop, populationSize)

    # Perform matingstate.fitness
    offidx = randperm(populationSize)
    offspringSize = populationSize - state.eliteSize
    for i in 1:2:offspringSize
        j = (i == offspringSize) ? i-1 : i+1
        if rand() < crossoverRate
            offspring[i], offspring[j] = crossover(population[selected[offidx[i]]], population[selected[offidx[j]]])
        else
            offspring[i], offspring[j] = population[selected[i]], population[selected[j]]
        end
    end

    # Elitism (copy population individuals before they pass to the offspring & get mutated)
    fitidxs = sortperm(state.fitpop)
    for i in 1:state.eliteSize
        subs = offspringSize+i
        offspring[subs] = copy(population[fitidxs[i]])
    end

    # Perform mutation
    for i in 1:offspringSize
        if rand() < mutationRate
            mutation(offspring[i])
        end
    end

    # Create new generation & evaluate it
    for i in 1:populationSize
        o = apply!(constraints, offspring[i])
        population[i] = o
        state.fitpop[i] = value(objfun, o)
    end
    # apply penalty to fitness
    penalty!(state.fitpop, constraints, population)

    # find the best individual
    minfit, fitidx = findmin(state.fitpop)
    state.fittest = population[fitidx]
    state.fitness = state.fitpop[fitidx]

    return false
end
