
using Evolutionary

nworks = 2
distributed_ga(localcpu=nworks)

gene = IntegerGene(10, "index")

npop = 2 * nworks
warmup_pop = Vector{Individual}(undef, npop)
for i in 1:npop
    warmup_pop[i] = AbstractGene[gene]
end

npop = 10 * nworks
pop = Vector{Individual}(undef, npop)
for i in 1:npop
    pop[i] = AbstractGene[gene]
end

@everywhere IntegerGene(:FM)
@everywhere Crossover(:SPX)
@everywhere Selection(:RWS)

@everywhere function objfun(chrom ::Individual)
    return abs( bin(chrom[1]) - 501 )
end

for i in 1:2
    res = ga( objfun, warmup_pop,
              parallel   = true ,
              nworkers   = nworks ,
              iterations = 2 )
end

res = ga( objfun, pop,
          parallel   = true ,
          nworkers   = nworks ,
          iterations = 10 )

nothing
