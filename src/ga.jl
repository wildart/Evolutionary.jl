# Genetic Algorithms
# ==================
#         objfun: Objective fitness function
#              N: Search space dimensionality
# initPopulation: Search space dimension ranges as a vector, or initial population values as matrix,
#                 or generation function which produce individual population entities.
# populationSize: Size of the population
#  crossoverRate: The fraction of the population at the next generation, not including elite children,
#                 that is created by the crossover function.
#   mutationRate: Probability of chromosome to be mutated
#              ɛ: Positive integer specifies how many individuals in the current generation
#                 are guaranteed to survive to the next generation.
#                 Floating number specifies fraction of population.
#
@with_kw struct GA <: Optimizer
    N::Int;
    initPopulation::Individual = ones(N)
    lowerBounds::Union{Nothing, Vector} = nothing
    upperBounds::Union{Nothing, Vector} = nothing
    populationSize::Int = 50
    crossoverRate::Float64 = 0.8
    mutationRate::Float64 = 0.1
    ɛ::Real = 0
    selection::Function = ((x,n)->1:n)
    crossover::Function = ((x,y)->(y,x))
    mutation::Function = (x->x)
    tolIter = 10
    interim = false
end

function optimize(objfun::Function, opt::GA;
                    iterations::Integer = 100*opt.N,
                    tol = 0.0,
                    verbose = false,
                    debug = false)
    @unpack N,initPopulation,lowerBounds,upperBounds,populationSize,crossoverRate,mutationRate,ɛ,selection,crossover,mutation,tolIter,interim = opt
    store = Dict{Symbol,Any}()

    # Setup parameters
    eliteSize = isa(ɛ, Int) ? ɛ : round(Int, ɛ * populationSize)
    debug && println("Elite Size: $eliteSize")
    fitFunc = inverseFunc(objfun)

    # Initialize population
    individual = getIndividual(initPopulation, N)
    fitness = zeros(populationSize)
    population = Array{typeof(individual)}(undef, populationSize)
    offspring = similar(population)
    debug && println("Offspring Size: $(length(offspring))")

    # Generate population
    for i in 1:populationSize
        if isa(initPopulation, Vector)
            population[i] = initPopulation.*rand(eltype(initPopulation), N)
        elseif isa(initPopulation, Matrix)
            population[i] = initPopulation[:, i]
        elseif isa(initPopulation, Function)
            population[i] = initPopulation(N) # Creation function
        else
            error("Cannot generate population")
        end
        fitness[i] = fitFunc(population[i])
        debug && println("INIT $(i): $(population[i]) : $(fitness[i])")
    end
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
        debug && println("POP: $(population)")

        # Select offspring
        selected = selection(fitness, populationSize)

        # Perform mating
        offidx = randperm(populationSize)
        offspringSize = populationSize-eliteSize
        for i in 1:2:offspringSize
            j = (i == offspringSize) ? i-1 : i+1
            if rand() < crossoverRate
                debug && println("MATE $(offidx[i])+$(offidx[j])>: $(population[selected[offidx[i]]]) : $(population[selected[offidx[j]]])")
                offspring[i], offspring[j] = crossover(population[selected[offidx[i]]], population[selected[offidx[j]]])
                debug && println("MATE >$(offidx[i])+$(offidx[j]): $(offspring[i]) : $(offspring[j])")
            else
                offspring[i], offspring[j] = population[selected[i]], population[selected[j]]
            end
        end

        # Elitism (copy population individuals before they pass to the offspring & get mutated)
        for i in 1:eliteSize
            subs = offspringSize+i
            debug && println("ELITE $(fitidx[i])=>$(subs): $(population[fitidx[i]])")
            offspring[subs] = copy(population[fitidx[i]])
        end

        # Perform mutation
        for i in 1:offspringSize
            if rand() < mutationRate
                debug && println("MUTATED $(i)>: $(offspring[i])")
                mutation(offspring[i])
                debug && println("MUTATED >$(i): $(offspring[i])")
            end
        end

        debug && println("OFF: $(offspring)")

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
