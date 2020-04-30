"""
Implementation of Genetic Algorithm

The constructor takes following keyword arguments:

- `populationSize`: Size of the population
- `crossoverRate`: The fraction of the population at the next generation, not including elite children, that is created by the crossover function.
- `mutationRate`: Probability of chromosome to be mutated
- `ɛ`: Positive integer specifies how many individuals in the current generation are guaranteed to survive to the next generation. Floating number specifies fraction of population.
- `selection`: [Selection](@ref) function
- `crossover`: [Crossover](@ref) function (default: `identity`)
- `mutation`: [Mutation](@ref) function (default: `identity`)
"""
@kwdef struct GA <: AbstractOptimizer
    populationSize::Int = 50
    crossoverRate::Float64 = 0.8
    mutationRate::Float64 = 0.1
    ɛ::Real = 0
    selection::Function = ((x,n)->1:n)
    crossover::Function = identity
    mutation::Function = identity
end
population_size(method::GA) = method.populationSize
default_options(method::GA) = Dict(:iterations=>1000, :abstol=>1e-10)

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

function update_state!(objfun, state, population::AbstractVector{IT}, method::GA) where {IT}
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
        population[i] = offspring[i]
        state.fitpop[i] = value(objfun, offspring[i])
    end
    minfit, fitidx = findmin(state.fitpop)
    state.fittest = population[fitidx]
    state.fitness = state.fitpop[fitidx]

    return false
end
