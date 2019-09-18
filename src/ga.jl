# Genetic Algorithms
# ==================
#        objfun: Objective fitness function
#             N: Search space dimensionality
#    population: Initial population values as vector
# crossoverRate: The fraction of the population at the next generation, not including elite children,
#                that is created by the crossover function.
#  mutationRate: Probability of chromosome to be mutated
#             ɛ: Positive integer specifies how many individuals in the current generation
#                are guaranteed to survive to the next generation.
#                Floating number specifies fraction of population.
#
function ga(objfun::Function, N::Int, population::Vector{T};
            crossoverRate::Float64 = 0.8,
            mutationRate::Float64 = 0.1,
            ɛ::Real = 0,
            selection::Function = ((x, n) -> 1:n),
            crossover::Function = ((x, y) -> (y, x)),
            mutation::Function = (x -> x),
            iterations::Integer = 100*N,
            tol = 0.0,
            tolIter = 10,
            verbose = false,
            debug = false,
            interim = false) where {T}

    store = Dict{Symbol,Any}()

    # Setup parameters
    populationSize = length(population)
    elite = isa(ɛ, Int) ? ɛ : round(Int, ɛ * populationSize)
    fitFunc = inverseFunc(objfun)
    fitness = fitFunc.(population)
    offspring = similar(population)
    fitidx = sortperm(fitness, rev = true)

    keep(interim, :fitness, copy(fitness), store)

    # Generate and evaluate offspring
    itr = 1
    bestFitness = 0.0
    bestIndividual = 0
    fittol = 0.0
    fittolitr = 1
    while true
        debug && println("BEST: $(fitidx)")

        # Select offspring
        selected = selection(fitness, populationSize)

        # Perform mating
        offidx = randperm(populationSize)
        for i in 1:2:populationSize
            j = (i == populationSize) ? i-1 : i+1
            if rand() < crossoverRate
                debug && println("MATE $(offidx[i])+$(offidx[j])>: $(population[selected[offidx[i]]]) : $(population[selected[offidx[j]]])")
                offspring[i], offspring[j] = crossover(population[selected[offidx[i]]], population[selected[offidx[j]]])
                debug && println("MATE >$(offidx[i])+$(offidx[j]): $(offspring[i]) : $(offspring[j])")
            else
                offspring[i], offspring[j] = population[selected[i]], population[selected[j]]
            end
        end

        # Perform mutation
        for i in 1:populationSize
            if rand() < mutationRate
                debug && println("MUTATED $(i)>: $(offspring[i])")
                mutation(offspring[i])
                debug && println("MUTATED >$(i): $(offspring[i])")
            end
        end

        # Elitism
        if elite > 0
            for i in 1:elite
                subs = rand(1:populationSize)
                debug && println("ELITE $(fitidx[i])=>$(subs): $(population[fitidx[i]]) => $(offspring[subs])")
                offspring[subs] = population[fitidx[i]]
            end
        end

        # New generation
        for i in 1:populationSize
            population[i] = offspring[i]
            fitness[i] = fitFunc(offspring[i])
            debug && println("FIT $(i): $(fitness[i])")
        end
        fitidx = sortperm(fitness, rev = true)
        bestIndividual = fitidx[1]
        curGenFitness = Float64(objfun(population[bestIndividual]))
        fittol = abs(bestFitness - curGenFitness)
        bestFitness = curGenFitness

        keep(interim, :fitness, copy(fitness), store)
        keep(interim, :bestFitness, bestFitness, store)

        # Verbose step
        verbose && println("BEST: $(bestFitness): $(population[bestIndividual]), G: $(itr)")

        # Terminate:
        #  if fitness tolerance is met for specified number of steps
        if fittol <= tol
            if fittolitr > tolIter
                break
            else
                fittolitr += 1
            end
        else
            fittolitr = 1
        end
        # if number of iterations more then specified
        if itr >= iterations
            break
        end
        itr += 1
    end

    return population[bestIndividual], bestFitness, itr, fittol, store
end

# Create population from creation function, with an option to partially initialise with specific individuals (default)
function ga(objfun::Function, N::Int; creation::Function = (n -> rand(n)), population::Vector{T}=[], populationSize::Integer=50, kwargs...) where {T}
    remaining = populationSize - length(population)
    population = vcat(population, [creation(N) for _ in 1:remaining])
    return ga(objfun, N, population; kwargs...)
end

# Spawn population from one individual
function ga(objfun::Function, N::Int, individual::Vector{E}; populationSize::Integer=50, kwargs...) where {E<:Real}
    N = length(individual)
    population = [individual .* rand(E, N) for _ in 1:populationSize]
    return ga(objfun, N, population; kwargs...)
end

# Spawn population from matrix of individuals
function ga(objfun::Function, N::Int, population::Matrix{E}; kwargs...) where {E<:Real}
    return ga(objfun, N, [population[:,i] for i in axes(population, 2)]; kwargs...)
end