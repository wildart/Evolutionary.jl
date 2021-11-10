"""
Nondominated Sorting Genetic Algorithm (NSGA-II) for Multi-objective Optimization

The constructor takes following keyword arguments:

- `populationSize`: The size of the population
- `crossoverRate`: The fraction of the population at the next generation, that is created by the crossover function
- `mutationRate`: Probability of chromosome to be mutated
- `selection`: [Selection](@ref) function (default: `tournament`)
- `crossover`: [Crossover](@ref) function (default: `SBX`)
- `mutation`: [Mutation](@ref) function (default: `PLM`)
"""
mutable struct NSGA2 <: AbstractOptimizer
    populationSize::Int
    crossoverRate::Float64
    mutationRate::Float64
    selection::Function
    crossover::Function
    mutation::Function

    NSGA2(; populationSize::Int=50, crossoverRate::Float64=0.9, mutationRate::Float64=0.1,
        selection::Function = tournament(2, select=twowaycomp),
        crossover::Function = SBX(),
        mutation::Function = PLM()) =
        new(populationSize, crossoverRate, mutationRate, selection, crossover, mutation)
end
population_size(method::NSGA2) = method.populationSize
default_options(method::NSGA2) = (iterations=1000, abstol=1e-15)
summary(m::NSGA2) = "NSGA-II[P=$(m.populationSize),x=$(m.crossoverRate),μ=$(m.mutationRate)]"
show(io::IO,m::NSGA2) = print(io, summary(m))

mutable struct NSGAState{T,IT} <: AbstractOptimizerState
    N::Int                      # population size
    fitness::AbstractMatrix{T}  # fitness of the fittest individuals
    fitpop::AbstractMatrix{T}   # fitness of the whole population (including offspring)
    fittest::AbstractVector{IT} # fittest individuals
    offspring::AbstractArray    # offspring cache
    population::AbstractArray   # combined population (parents + offspring)
    ranks::Vector{Int}          # individual ranks
    crowding::Vector{T}         # individual crowding distance
end
value(s::NSGAState) = s.fitness
minimizer(s::NSGAState) = s.fittest

"""Initialization of NSGA2 algorithm state"""
function initial_state(method::NSGA2, options, objfun, parents)

    v = value(objfun) # objective function value
    T = eltype(v)     # objective function value type
    d = length(v)     # objective function value dimension
    N = length(first(parents)) # parents dimension
    IT = eltype(parents)       # individual type
    offspring = similar(parents) # offspring cache

    # construct fitness array that covers total population,
    # i.e. parents + offspring
    fitpop = fill(typemax(T), d, method.populationSize*2)

    # Evaluate parents fitness
    value!(objfun, fitpop, parents)

    # setup initial state
    allpop = StackView(parents, offspring)
    ranks = vcat(fill(1, method.populationSize), fill(2, method.populationSize))
    crowding = vcat(fill(zero(T), method.populationSize), fill(typemax(T), method.populationSize))
    return NSGAState(N, zeros(T,d,0), fitpop, IT[], offspring, allpop, ranks, crowding)
end

function update_state!(objfun, constraints, state, parents::AbstractVector{IT}, method::NSGA2, options, itr) where {IT}
    @unpack populationSize,crossoverRate,mutationRate,selection,crossover,mutation = method

    # Select offspring
    specFit = StackView(state.ranks, state.crowding, dims=1)
    selected = selection(view(specFit, :, 1:populationSize), populationSize)

    # Perform mating
    offidx = randperm(populationSize)
    for i in 1:2:populationSize
        j = (i == populationSize) ? i-1 : i+1
        if rand() < crossoverRate
            state.offspring[i], state.offspring[j] = crossover(parents[selected[offidx[i]]], parents[selected[offidx[j]]])
        else
            state.offspring[i], state.offspring[j] = parents[selected[i]], parents[selected[j]]
        end
    end

    # Perform mutation
    for i in 1:populationSize
        if rand() < mutationRate
            mutation(state.offspring[i])
        end
        apply!(constraints, state.offspring[i])
    end

    # calculate fitness of the offspring
    offfit = @view state.fitpop[:, populationSize+1:end]
    value!(objfun, offfit, state.offspring)
    # apply penalty to fitness
    penalty!(offfit, constraints, state.offspring)

    # calculate ranks & crowding for population
    F = nondominatedsort!(state.ranks, state.fitpop)
    crowding_distance!(state.crowding, state.fitpop, F)

    # select best individuals
    fitidx = Int[]
    for f in F
        if length(fitidx) + length(f) > populationSize
            idxs = sortperm(view(state.crowding,f))
            append!(fitidx, idxs[1:(populationSize-length(fitidx))])
            break
        else
            append!(fitidx, f)
        end
    end
    # designate non-dominated individual as fittest
    state.fittest = state.population[F[1]]
    # and keep their fitness
    state.fitness = state.fitpop[:,F[1]]
    # construct new parent population
    parents .= state.population[fitidx]

    return false
end
