"""
Implementation of Evolution Strategy: (μ/ρ(+/,)λ)-ES

The constructor takes following keyword arguments:

- `initStrategy`: an initial strategy description, (default: empty)
- `recombination`: ES recombination function for population (default: `first`), see [Crossover](@ref)
- `srecombination`: ES recombination function for strategies (default: `first`), see [Crossover](@ref)
- `mutation`: [Mutation](@ref) function for population (default: `first`)
- `smutation`: [Mutation](@ref) function for strategies (default: `identity`)
- `μ`/`mu`: the number of parents
- `ρ`/`rho`: the mixing number, ρ ≤ μ, (i.e., the number of parents involved in the procreation of an offspring)
- `λ`/`lambda`: the number of offspring
- `selection`: the selection strategy `:plus` or `:comma` (default: `:plus`)
"""
struct ES <: AbstractOptimizer
    initStrategy::AbstractStrategy
    recombination::Function
    srecombination::Function
    mutation::Function
    smutation::Function
    μ::Integer
    ρ::Integer
    λ::Integer
    selection::Symbol

    ES(; initStrategy::AbstractStrategy = NoStrategy(),
        recombination::Function = first,
        srecombination::Function = first,
        mutation::Function = ((r,s) -> r),
        smutation::Function = identity,
        μ::Integer = 1,
        mu::Integer = μ,
        ρ::Integer = μ,
        rho::Integer = ρ,
        λ::Integer = 1,
        lambda::Integer = λ,
        selection::Symbol = :plus) =
        new(initStrategy, recombination, srecombination, mutation, smutation,
            mu, rho, lambda, selection)
end
population_size(method::ES) = method.μ
default_options(method::ES) = (iterations=1000, abstol=1e-10)
summary(m::ES) = "($(m.μ)/$(m.ρ)$(m.selection == :plus ? '+' : ',')$(m.λ))-ES"

mutable struct ESState{T,IT,ST} <: AbstractOptimizerState
    N::Int
    fitness::Vector{T}
    strategies::Vector{ST}
    fittest::IT
end
value(s::ESState) = first(s.fitness)
minimizer(s::ESState) = s.fittest
strategy(s::ESState) = first(s.strategies)

"""Initialization of ES algorithm state"""
function initial_state(method::ES, options, objfun, population)
    T = typeof(value(objfun))
    individual = first(population)
    N = length(individual)

    # Populate fitness and strategies
    fitness = map(i -> value(objfun, i), population)
    strategies = Array{AbstractStrategy}(undef, method.μ)
    for i in 1:method.μ
        strategies[i] = copy(method.initStrategy)
    end

    # setup initial state
    return ESState(N, fitness, strategies, copy(individual))
end

function update_state!(objfun, constraints, state, population::AbstractVector{IT}, method::ES, itr) where {IT}
    @unpack initStrategy,recombination,srecombination,mutation,smutation,μ,ρ,λ,selection = method

    @assert ρ <= μ "Number of parents involved in the procreation of an offspring should be no more then total number of parents"
    if selection == :comma
        @assert μ < λ "Offspring population must be larger then parent population"
    end
    offspring = Array{IT}(undef, λ)
    fitoff = fill(Inf, λ)
    stgoff = Array{AbstractStrategy}(undef, λ)

    for i in 1:λ
        # Recombine the ρ selected parents to form a recombinant individual
        if ρ == 1
            j = rand(1:μ)
            recombinantStrategy = state.strategies[j]
            recombinant = copy(population[j])
        else
            idx = randperm(μ)[1:ρ]
            recombinantStrategy = srecombination(state.strategies[idx])
            recombinant = recombination(population[idx])
        end

        # Mutate the strategy parameter set of the recombinant
        stgoff[i] = smutation(recombinantStrategy)

        # Mutate the objective parameter set of the recombinant using the mutated strategy parameter set
        # to control the statistical properties of the object parameter mutation
        off = mutation(recombinant, stgoff[i])

        # Apply constraints
        offspring[i] = apply!(constraints, off)

        # Evaluate fitness
        fitoff[i] = value(objfun, offspring[i])
    end
    # apply penalty to fitness
    penalty!(fitoff, constraints, offspring)

    # Select new parent population
    if selection == :plus
        idxs = sortperm(vcat(state.fitness, fitoff))[1:μ]
        skip = idxs[idxs.<=μ]
        for i = 1:μ
            if idxs[i] ∉ skip
                ii = idxs[i] - μ
                population[i] = offspring[ii]
                state.strategies[i] = stgoff[ii]
                state.fitness[i] = fitoff[ii]
            end
        end
    else
        idxs = sortperm(fitoff)[1:μ]
        for (i,j) in enumerate(idxs)
            population[i] = offspring[j]
            state.strategies[i] = stgoff[j]
            state.fitness[i] = fitoff[j]
        end
    end

    # indicate a fittest individual
    state.fittest = first(population)

    return false
end
