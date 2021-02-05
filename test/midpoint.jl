
using Evolutionary

nworks = 2
distributed_ga(localcpu=nworks)

@everywhere using SpecialFunctions
@everywhere using DataAnalysis

x = -5.0:0.01:5.0
y = erf.(x)

gene = IntegerGene(10, "index")

npop = 100 * nworks
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
bestFit, bestGene = ga( objfun, pop,
                        parallel = true,
                        nworkers = nworks)

nothing
