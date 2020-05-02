# Evolutionary.jl

The package __Evolutionary__ aims to provide a library for evolutionary optimization. It provides implementation of $(\mu/\rho \; \stackrel{+}{,} \;\lambda)$-Evolution Strategy, $(\mu/\mu_I, \;\lambda)$-Covariance Matrix Adaptation Evolution Strategy, Genetic Algorithm, and Differential Evolution as well as a rich set of mutation, recombination, crossover and selection functions.

## Getting started

To install the package just type

```julia
] add Evolutionary
```

A simple example of using the [`GA`](@ref) algorithm to find minium of the [Sphere function](https://www.sfu.ca/~ssurjano/spheref.html).

```@repl
using Evolutionary
result = Evolutionary.optimize(
      x -> sum(x.^2), ones(3),
      GA(populationSize = 100, selection = susinv,
         crossover = discrete, mutation = domainrange(ones(3))))
```
