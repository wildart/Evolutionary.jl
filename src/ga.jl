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
#              ϵ : Boolean to decide if the N best ones will surely survive or
#                  it's all random
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
function ga( objfun         ::Function                        ,
             population     ::Vector{Individual}              ;
             crossoverRate  ::Float64                 = 0.5   ,
             mutationRate   ::Float64                 = 0.5   ,
             ϵ              ::Bool                    = true  ,
             iterations     ::Integer                 = 100   ,
             tol            ::Real                    = 0.0   ,
             parallel       ::Bool                    = false )

    # Initialize population
    N = length(population)
    fitness = Vector{Float64}(undef, N)

    for i in 1:N
        @inbounds fitness[i] = objfun(population[i])
    end
    fitidx = sortperm(fitness)

    # save optional arguments in a dictionary
    # to pass to generation function
    pars = Dict{Symbol, Any}()
    pars[:crossoverRate] = crossoverRate
    pars[:mutationRate ] = mutationRate
    pars[:ϵ            ] = ϵ
    pars[:iterations   ] = iterations
    pars[:tol          ] = tol
    
    # Generate and evaluate offspring
    if parallel
        population = distribute(population)
        fitness    = distribute(fitness)
        elapsed_time = @elapsed begin
            spmd(generations_parallel, objfun,
                 population, fitness, pars)
        end
    else
        elapsed_time = @elapsed begin
            generations(objfun, population, N, fitness, pars)
        end
    end

    bestFitness, bestIndividual = findmin(fitness)
    if bestFitness <= tol
        isfit = true
    else
        isfit = false
    end

    # result presentation
    data_presentation( population[bestIndividual], iterations,
                       bestFitness, isfit, elapsed_time )
    
    return population[bestIndividual], bestFitness
end

####################################################################

function generations( objfun     ::Function           ,
                      population ::Vector{Individual} ,
                      N          ::Integer            ,
                      fitness    ::Vector{Float64}    ,
                      pars       ::Dict{Symbol,Any}   )
    
    # Variable initialization
    generations    = 1
    bestIndividual = 0
    bestFitness    = 0.0
    offspring      = Vector{Individual}(undef,   N)
    full_pop       = Vector{Individual}(undef, 2*N)
    full_fitness   = Vector{Float64   }(undef, 2*N)

    for iter in 1:pars[:iterations]
        
        # Select offspring
        selected = selection(fitness, N)
        
        # Perform mating
        offidx = randperm(N)
        for i in 1:2:N
            j = (i == N) ? i-1 : i+1
            if rand() < pars[:crossoverRate]
                @inbounds begin
                    offspring[i], offspring[j] =
                        crossover( population[selected[offidx[i]]] ,
                                   population[selected[offidx[j]]] )
                end
            else
                @inbounds begin
                    offspring[i], offspring[j] =
                        population[selected[i]], population[selected[j]]
                end
            end
        end
        
        # Perform mutation
        for i in 1:N
            if rand() < pars[:mutationRate]
                @inbounds mutate(offspring[i])
            end
        end
        
        # Elitism
        # When true, always picks N best individuals from the full population
        # (parents+offspring), which is size 2*N.
        # When false, does everything randomly
        if pars[:ϵ]
            @inbounds begin
                full_pop[  1:  N] = population
                full_pop[N+1:2*N] = offspring
                full_fitness = objfun.(full_pop)
                fitidx = sortperm(full_fitness)
            end
            for i in 1:N
                @inbounds begin
                    population[i] = full_pop[fitidx[i]]
                       fitness[i] = objfun(population[i])
                end
            end
        else
            for i in 1:N
                @inbounds begin
                    population[i] = offspring[i]
                       fitness[i] = objfun(population[i])
                end
            end
        end
    end

    return nothing
end

####################################################################

function generations_parallel( objfun ::Function                                ,
                               popul  ::DArray{Individual,1,Vector{Individual}} ,
                               fit    ::DArray{Float64,1,Vector{Float64}}       ,
                               pars   ::Dict{Symbol,Any}                        )

    # Variable initialization
    population     = popul[:L]
    fitness        =   fit[:L]
    N              = length(population)
    generations    = 1
    bestIndividual = 0
    bestFitness    = 0.0
    offspring      = Vector{Individual}(undef,   N)
    full_pop       = Vector{Individual}(undef, 2*N)
    full_fitness   = Vector{Float64   }(undef, 2*N)

    # Generate and evaluate offspring
    for iter in 1:pars[:iterations]
        
        # Select offspring
        selected = selection(fitness, N)
        
        # Perform mating
        offidx = randperm(N)
        for i in 1:2:N
            j = (i == N) ? i-1 : i+1
            if rand() < pars[:crossoverRate]
                @inbounds begin
                    offspring[i], offspring[j] =
                        crossover( population[selected[offidx[i]]] ,
                                   population[selected[offidx[j]]] )
                end
            else
                @inbounds begin
                    offspring[i], offspring[j] =
                        population[selected[i]], population[selected[j]]
                end
            end
        end
        
        # Perform mutation
        for i in 1:N
            if rand() < pars[:mutationRate]
                @inbounds mutate(offspring[i])
            end
        end
        
        # Elitism
        # When true, always picks N best individuals from the full population
        # (parents+offspring), which is size 2*N.
        # When false, does everything randomly
        if pars[:ϵ]
            @inbounds begin
                full_pop[1  :  N] = population
                full_pop[1+N:2*N] = offspring
                full_fitness      = objfun.(full_pop)
                fitidx            = sortperm(full_fitness)
            end
            for i in 1:N
                @inbounds begin
                    population[i] = full_pop[fitidx[i]]
                       fitness[i] = objfun(population[i])
                end
            end
        else
            for i in 1:N
                @inbounds begin
                    population[i] = offspring[i]
                       fitness[i] = objfun(population[i])
                end
            end
        end
    end

    return nothing
end

####################################################################

function data_presentation( individual   ::Individual ,
                            generations  ::Integer    ,
                            bestFitness  ::Float64    ,
                            isfit        ::Bool       ,
                            elapsed_time ::Float64    )

    optim_time = round(elapsed_time, digits=3)
    
    params = Vector{AbstractString}(undef, 0)
    values = Vector{Real}(undef, 0)
    for gene in individual
        if isa(gene, FloatGene)
            for (i,j) in enumerate(gene.value)
                push!(params, gene.name[i])
                push!(values, j)
            end
        elseif isa(gene, IntegerGene)
            push!(params, gene.name)
            push!(values, bin(gene))
        else
            push!(params, gene.name)
            push!(values, gene.value)
        end
    end
    
    table = string( "| parameter | value |\n" ,
                    "|-----------|-------|\n" )
    for (i,j) in enumerate(params)
        table *= "| $j | $(values[i]) |\n"
    end
    @doc table present
    
    printstyled("\nRESULTS :\n", color=:bold)
    println("number of generations = " * string(generations))
    println("best Fitness          = " * string(bestFitness))
    println("Run time              = $optim_time seconds")
    println("")
    printstyled("GENES OF BEST INDIVIDUAL :\n", color=:bold)
    display(@doc present)
    println("")
    if isfit
        printstyled("OPTIMIZATION SUCCESSFUL\n"  , color=:bold)
    else
        printstyled("OPTIMIZATION UNSUCCESSFUL\n", color=:bold)
    end

    return nothing
end

