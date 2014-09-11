# Evolution Strategy
# ==================
#
# Implementation: (μ/ρ(+/,)λ)-CMA-ES
#
# μ is the number of parents
# ρ ≤ μ the mixing number (i.e., the number of parents involved in the procreation of an offspring)
# λ is the number of offspring.
#
# Comma-selection (μ<λ must hold): parents are deterministically selected from the set of the offspring
# Plus-selection: parents are deterministically selected from the set of both the parents and offspring
#
function cmaes{T}(objfun::Function, initval::T, initstg::Strategy;
                  recombination::Function = (x->x[1]),
                  srecombination::Function = (x->x[1]),
                  mutation::Function = (x->x),
                  smutation::Function = (x->x),
                  termination::Function = (x->false),
                  μ::Integer = 1,
                  ρ::Integer = 1,
                  λ::Integer = 1,
                  selection::Symbol = :plus,
                  iterations::Integer = 1_000,
                  verbose = false)

    @assert ρ <= μ "Number of parents involved in the procreation of an offspring should be no more then total number of parents"
    if selection == :comma
        @assert μ < λ "Offspring population must be larger then parent population"
    end

    # Initialize parent population
    population = fill(initval, μ)
    offspring = Array(T, λ)
    fitpop = fill(objfun(initval), μ)
    fitoff = fill(Inf, λ)
    stgpop = fill(initstg, μ)
    stgoff = fill(initstg, λ)

    count = 0
    while true

        for i in 1:λ            
            # Recombine the ρ selected parents to form a recombinant individual            
            if ρ == 1
                j = rand(1:μ)
                recombinantStrategy = stgpop[j]
                recombinant = population[j] 
            else                
                idx = randperm(μ)[1:ρ]
                recombinantStrategy = srecombination(stgpop[idx])
                recombinant = recombination(population[idx])
            end
            
            # Mutate the strategy parameter set of the recombinant
            stgoff[i] = smutation(recombinantStrategy)
            
            # Mutate the objective parameter set of the recombinant using the mutated strategy parameter set
            # to control the statistical properties of the object parameter mutation
            offspring[i] = mutation(recombinant, stgoff[i])

            # Evaluate fitness
            fitoff[i] = objfun(offspring[i])
        end

        # Select new parent population
        if selection == :plus
            idx = sortperm([fitpop, fitoff])[1:μ]
            skip = idx[idx.<=μ]
            for i = 1:μ
                if idx[i] ∉ skip
                    ii = idx[i] - μ
                    population[i] = offspring[ii]
                    stgpop[i] = stgoff[ii]
                    fitpop[i] = fitoff[ii]
                end
            end
        else
            idx = sortperm(fitoff)[1:μ]
            population = offspring[idx]
            stgpop = stgoff[idx]
            fitpop = fitoff[idx]
        end

        # termination condition
        count += 1
        if count == iterations || termination(stgpop[1])
            break
        end
        #verbose && println("BEST: $(fitpop[1]): $(population[1]): $(stgpop[1])")
        verbose && println("BEST: $(fitpop[1]): $(stgpop[1])")
    end

    return population[1], fitpop[1], count

end