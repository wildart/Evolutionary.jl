using BenchmarkTools
using Evolutionary
using Random
Random.seed!(2);

rosenbrock(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2

N = 3
cmaes = CMAES(;mu=20, lambda=100)
ga = GA(populationSize=100, ɛ=0.1, selection=rouletteinv, crossover=intermediate(0.25), mutation=domainrange(fill(0.5,N)))
es = ES(initStrategy=IsotropicStrategy(N), recombination=average, srecombination=average, mutation=gaussian, smutation=gaussian, μ=10, ρ=3, λ=100, selection=:plus)
de = DE(populationSize = 100)

suite = BenchmarkGroup()

suite["methods"] = BenchmarkGroup(["ga", "cmaes", "es", "de"])
suite["methods"]["cmaes"] = @benchmarkable Evolutionary.optimize(rosenbrock, randn($N), $cmaes)
suite["methods"]["ga"] = @benchmarkable Evolutionary.optimize(rosenbrock, randn($N), $ga)
suite["methods"]["es"] = @benchmarkable Evolutionary.optimize(rosenbrock, randn($N), $es)
suite["methods"]["de"] = @benchmarkable Evolutionary.optimize(rosenbrock, randn($N), $de)

# parameters
loadparams!(suite, BenchmarkTools.load("bmsetup.json")[1], :evals, :samples);
# tune!(suite);
# BenchmarkTools.save("bmsetup.json", params(suite));
# exit()

# benchmark
results = run(suite, verbose = true)
BenchmarkTools.save("results.json", results)
