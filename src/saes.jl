# Population initialazer
initialize(rng::AbstractRNG) = Individual()

# Recombine the selected parents to form a recombinant individual
recombine(population::Vector{Individual}) = population[1]

# Mutate the strategy parameter set of the recombinant
mutateStrategy(recombinant::Individual) = recombinant

# Mutate the objective parameter set of the recombinant
mutate(recombinant::Individual) = recombinant

#
# (μ/ρ(+/,)λ)-Self-Adaptation Evolution Strategy
#
# μ is the number of parents
# ρ ≤ μ the mixing number (i.e., the number of parents involved in the procreation of an offspring)
# λ is the number of offspring.
#
# Comma-selection (μ<λ must hold): parents are deterministically selected from the set of the offspring
# Plus-selection: parents are deterministically selected from the set of both the parents and offspring
#
function saes(objfun::Function;
              μ::Integer = 1,
              ρ::Integer = 1,
              λ::Integer = 1,
              selection::Symbol = :comma
              iterations::Integer = 1_000,
              tol::Real = 1e-8)

    @assert ρ <= μ "Number of parents involved in the procreation of an offspring should be no more then total number of parents" 

    rng = MersenneTwister(time_ns())
    population = Array(Individual, μ)
    fitness = Array(Float64, μ)
    offspring = Array(Individual, λ)

    # Initialize parent population    
    for i in 1:μ
        population[i] = initialize(rng)
    end

    count = 0
    bestFitness = NaN
    while true

        for i in 1:λ
            # Recombine the ρ selected parents a to form a recombinant individual
            idx = randperm(μ)[1:ρ]
            recombinant = (ρ == 1) ? population[idx[1]] : recombine(population[idx])
                
            # Mutate the strategy parameter set of the recombinant
            recombinant = mutateStrategy(recombinant)
            
            # Mutate the objective parameter set of the recombinant using the mutated strategy parameter set 
            # to control the statistical properties of the object parameter mutation
            offspring[i] = mutate(recombinant)
        end
        
        # Select new parent population
        if selection == :plus
            population = [population, offspring]
        end



        # termination condition
        count += 1
        if count == iterations || (bestFitness - best) < tol
            break
        end
    end
end
