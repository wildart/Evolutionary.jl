##### ga.jl #####

# In this file you will find the Genetic Algorithm.

####################################################################

export ga

####################################################################

# Genetic Algorithms
# ==================
#         objfun : Objective fitness function
#              N : Search space dimensionality
# initPopulation : Search space dimension ranges as a vector, or initial population values as matrix,
#                  or generation function which produce individual population entities.
# populationSize : Size of the population
#  crossoverRate : The fraction of the population at the next generation, not including elite
#                  children, that is created by the crossover function.
#   mutationRate : Probability of chromosome to be mutated
#              ϵ : Positive integer specifies how many individuals in the current generation
#                  are guaranteed to survive to the next generation.
#                  Floating number specifies fraction of population.
#
"""
    ga( objfun         ::Function                          ,
        initpopulation ::Vector{<:AbstractGene}            ,
        populationSize ::Int64                             ;
        lowerBounds    ::Union{Nothing, Vector } = nothing ,
        upperBounds    ::Union{Nothing, Vector } = nothing ,
        crossoverRate  ::Float64                 = 0.5     ,
        mutationRate   ::Float64                 = 0.5     ,
        ϵ              ::Real                    = 0       ,
        iterations     ::Integer                 = 100     ,
        tol            ::Real                    = 0.0     ,
        tolIter        ::Int64                   = 10      ,
        verbose        ::Bool                    = false   ,
        debug          ::Bool                    = false   ,
        interim        ::Bool                    = false   ,
        parallel       ::Bool                    = false   )

Runs the Genetic Algorithm using the objective function `objfun`, the initial population `initpopulation` and the population size `populationSize`. `objfun` is the function to MINIMIZE. 
"""
function ga( objfun         ::Function                          ,
             initpopulation ::Vector{Vector{<:AbstractGene}}    ,
             populationSize ::Int64                             ;
             lowerBounds    ::Union{Nothing, Vector } = nothing ,
             upperBounds    ::Union{Nothing, Vector } = nothing ,
             crossoverRate  ::Float64                 = 0.5     ,
             mutationRate   ::Float64                 = 0.5     ,
             ϵ              ::Bool                    = false   ,
             iterations     ::Integer                 = 100     ,
             tol            ::Real                    = 0.0     ,
             tolIter        ::Int64                   = 10      ,
             verbose        ::Bool                    = false   ,
             debug          ::Bool                    = false   ,
             interim        ::Bool                    = false   ,
             parallel       ::Bool                    = false   )

    store = Dict{Symbol,Any}()

    # Initialize population
    fitness = zeros(populationSize)
    population = Vector{Vector{<:AbstractGene}}(undef, populationSize)
    offspring = similar(population)

    for i in 1:populationSize
        population[i] = initpopulation[i]
        fitness[i] = objfun(population[i])
        debug && println("INIT $(i): $(population[i]) : $(fitness[i])")
    end
    fitidx = sortperm(fitness)
    keep(interim, :fitness, copy(fitness), store)
    
    # Generate and evaluate offspring
    generations = 1
    bestFitness = 0.0
    bestIndividual = 0
    full_fitness = Vector{Float64}(undef, 2*populationSize)
    full_pop = Vector{Vector{<:AbstractGene}}(undef, 2*populationSize)
    for iter in 1:iterations
        debug && println("BEST: $(fitidx)")

        # Select offspring
        selected = selection(fitness, populationSize)

        # Perform mating
        offidx = randperm(populationSize)
        for i in 1:2:populationSize
            j = (i == populationSize) ? i-1 : i+1
            if rand() < crossoverRate
                debug &&
                    println( "MATE $(offidx[i])+$(offidx[j])>: "     *
                             "$(population[selected[offidx[i]]]) : " *
                             "$(population[selected[offidx[j]]])"    )
                offspring[i], offspring[j] =
                    crossover(population[selected[offidx[i]]], population[selected[offidx[j]]])
                debug &&
                    println("MATE >$(offidx[i])+$(offidx[j]): $(offspring[i]) : $(offspring[j])")
            else
                offspring[i], offspring[j] =
                    population[selected[i]], population[selected[j]]
            end
        end
        
        # Perform mutation
        for i in 1:populationSize
            if rand() < mutationRate
                debug && println("MUTATED $(i)>: $(offspring[i])")
                mutate(offspring[i])
                debug && println("MUTATED >$(i): $(offspring[i])")
            end
        end
        
        # Elitism
        # When true, always picks N best individuals from the full population
        # (parents+offspring), which is size 2*N.
        # When false, does everything randomly
        if ϵ
            full_pop[1:populationSize] = population
            full_pop[populationSize+1:end] = offspring
            full_fitness = objfun.(full_pop)
            fitidx = sortperm(full_fitness)
            for i in 1:populationSize
                population[i] = full_pop[fitidx[i]]
                fitness[i] = objfun(population[i])
            end
        else
            for i in 1:populationSize
                population[i] = offspring[i]
                fitness[i] = objfun(population[i])
                debug && println("FIT $(i): $(fitness[i])")
            end
        end

        bestFitness, bestIndividual = findmin(fitness)

        keep(interim, :fitness, copy(fitness), store)
        keep(interim, :bestFitness, bestFitness, store)

        # Verbose step
        verbose &&
            println("BEST: $(round(bestFitness, digits=3)): " *
                    "$(population[bestIndividual]), G: $(iter)")

        generations = iter
        if bestFitness <= tol
            break
        end
    end

    # result presentation
    printstyled("\nRESULTS :\n", color=:bold)
    println("number of generations = " * string(generations))
    println("best Fitness          = " * string(bestFitness))
    println("")
    println("genes of best individual :")
    for gene in population[bestIndividual]
        if isa(gene, FloatGene)
            for (i,j) in enumerate(gene.value)
                println("$(gene.name[i]) = $j")
            end
        elseif isa(gene, IntegerGene)
            println("$(gene.name) = $(bin(gene))")
        else
            println("$(gene.name) = $(gene.value)")
        end
    end
    println("")
    
    return population[bestIndividual], bestFitness, generations, store
end
