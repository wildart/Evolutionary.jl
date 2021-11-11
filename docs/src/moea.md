# Multi-objective EA

```@docs
NSGA2
```

## Description

[Multi-objective optimization](https://en.wikipedia.org/wiki/Multi-objective_optimization) is an area of multiple criteria decision making that is concerned with mathematical optimization problems involving more than one objective function to be optimized simultaneously. Evolutionary algorithms are popular approaches to generating Pareto optimal solutions to a multi-objective optimization problem by appling Pareto-based ranking schemes, such as the Non-dominated Sorting Genetic Algorithm-II (NSGA-II)[^1].

## Auxiliary Functions

```@docs
Evolutionary.nondominatedsort!
Evolutionary.dominate
Evolutionary.dominations
Evolutionary.crowding_distance!
Evolutionary.gd
Evolutionary.igd
Evolutionary.spread
```

## References

[^1]: Deb, K. et al., "A fast and elitist multiobjective genetic algorithm: NSGA-II". IEEE Transactions on Evolutionary Computation, 2002.
