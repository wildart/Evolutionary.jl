
using Evolutionary

nworks = 2
distributed_ga(localcpu=nworks)

gene = IntegerGene(10, "index")

npop = 10 * nworks
pop = Vector{Individual}(undef, npop)

for i in 1:npop
    pop[i] = AbstractGene[gene]
end

@everywhere IntegerGene(:FM)
@everywhere Crossover(:SPX)
@everywhere Selection(:RWS)

println("Creating objfun function...\n")
@everywhere function objfun(chrom ::Individual)
    return abs( bin(chrom[1]) - 501 )
end

println("Starting ga...")
bestGene, bestFit = ga( objfun, pop,
                        parallel   = true ,
                        nworkers   = nworks ,
                        iterations = 10 )


nothing
