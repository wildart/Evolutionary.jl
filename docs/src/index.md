# Evolutionary.jl

The package __Evolutionary__ aims to provide a library for evolutionary optimization. It provides implementation of $(\mu/\rho \; \stackrel{+}{,} \;\lambda)$-Evolution Strategy, $(\mu/\mu_I, \;\lambda)$-Covariance Matrix Adaptation Evolution Strategy and Genetic Algorithms, as well as a rich set of mutation, recombination, crossover and selection functions.

## Getting started

To install the package just type

```julia
] add Evolutionary
```

A simple example of using the [`GA`](@ref) algorithm to solve a benchmark problem, **OneMax** (a simple problem consisting in maximizing the number of ones of a bitstring).

```@repl
using Evolutionary, Random
Random.seed!(9874984737484);
individualSize = 100 # chromosome size
best, _ = optimize(
        x -> 1 / sum(x),                 # Function to MINIMIZE
        GA(                              # Algorithm: GA
            N = individualSize,          # Length of chromosome
            initPopulation = (N -> BitArray(rand(Bool, N))),
            selection = tournament(3),
            mutation =  flip,
            crossover = singlepoint,
            mutationRate = 0.1,
            crossoverRate = 0.1,
            tolIter = 20,
            populationSize = 100
        ),
        iterations = 3000);
sum(best) == individualSize
```

## Methods

| Methods | Description |
|:--------|:------------|
|[`GA`](@ref)| Isometric mapping |
|[`ES`](@ref)| Evolution Strategy |
|[`CMAES`](@ref)| Covariance Matrix Adaptation Evolution Strategy |
