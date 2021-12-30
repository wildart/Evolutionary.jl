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
- `metrics` is a collection of convergence metrics.
"""
struct GA{T1,T2,T3} <: AbstractOptimizer
    populationSize::Int
    crossoverRate::Float64
    mutationRate::Float64
    ɛ::Real
    selection::T1
    crossover::T2
    mutation::T3
    metrics::ConvergenceMetrics

    GA(; populationSize::Int=50, crossoverRate::Float64=0.8, mutationRate::Float64=0.1,
        ɛ::Real=0, epsilon::Real=ɛ,
        selection::T1=((x,n)->1:n),
        crossover::T2=identity, mutation::T3=identity,
        metrics = ConvergenceMetric[AbsDiff(1e-12)]) where {T1, T2, T3} =
        new{T1,T2,T3}(populationSize, crossoverRate, mutationRate, epsilon, selection, crossover, mutation, metrics)
end
population_size(method::GA) = method.populationSize
default_options(method::GA) = (iterations=1000,)
summary(m::GA) = "GA[P=$(m.populationSize),x=$(m.crossoverRate),μ=$(m.mutationRate),ɛ=$(m.ɛ)]"
show(io::IO,m::GA) = print(io, summary(m))

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

function update_state!(objfun, constraints, state, parents::AbstractVector{IT}, method::GA, options, itr) where {IT}
    populationSize = method.populationSize
    evaltype = options.parallelization
    rng = options.rng
    offspring = similar(parents)

    # select offspring
    selected = method.selection(state.fitpop, populationSize, rng=rng)

    # perform mating
    offspringSize = populationSize - state.eliteSize
    recombine!(offspring, parents, selected, method, offspringSize, rng=rng)

    # Elitism (copy population individuals before they pass to the offspring & get mutated)
    fitidxs = sortperm(state.fitpop)
    for i in 1:state.eliteSize
        subs = offspringSize+i
        offspring[subs] = copy(parents[fitidxs[i]])
    end

    # perform mutation
    mutate!(offspring, method, constraints, rng=rng)

    # calculate fitness of the population
    evaluate!(objfun, state.fitpop, offspring, constraints)

    # select the best individual
    minfit, fitidx = findmin(state.fitpop)
    state.fittest = offspring[fitidx]
    state.fitness = state.fitpop[fitidx]
    
    # replace population
    parents .= offspring

    return false
end

function recombine!(offspring, parents, selected, method, n=length(selected);
                    rng::AbstractRNG=Random.default_rng())
    mates = ((i,i == n ? i-1 : i+1) for i in 1:2:n)
    for (i,j) in mates
        p1, p2 = parents[selected[i]], parents[selected[j]]
        if rand(rng) < method.crossoverRate
            offspring[i], offspring[j] = method.crossover(p1, p2, rng=rng)
        else
            offspring[i], offspring[j] = p1, p2
        end
    end

end

function mutate!(population, method, constraints;
                 rng::AbstractRNG=Random.default_rng())
    n = length(population)
    for i in 1:n
        if rand(rng) < method.mutationRate
            method.mutation(population[i], rng=rng)
        end        
        apply!(constraints, population[i])
    end
end

function evaluate!(objfun, fitness, population, constraints)
    # calculate fitness of the population
    value!(objfun, fitness, population)
    # apply penalty to fitness
    penalty!(fitness, constraints, population)
end

