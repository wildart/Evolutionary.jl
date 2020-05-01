# Crossover

In genetic algorithms and evolutionary computation, **crossover**, also called **recombination**, is a genetic operator used to combine the genetic information of two parents to generate new offspring.

## Recombination Interface

All recombination operations have following call interface: `recombination(i1, i2)` where `i1` and `i2` are the same type individuals that involved in recombination to produce an offspring. The recombination function returns pair of recombined individuals.

**Note:** Some of the selection algorithms implemented as function closures, in order to provide additional parameters for the specified above recombination interface.

## Operations


List of the ES strategy recombination operations:

```@docs
average(::Vector{<:AbstractStrategy})
```

List of the ES population recombination operations:

```@docs
average(population::Vector{T}) where {T <: AbstractVector}
marriage
```

Binary crossovers:

```@docs
singlepoint
twopoint
uniform
```

Real valued crossovers:
```@docs
identity
discrete
waverage
intermediate
line
```

Permutation crossovers:
```@docs
pmx
ox1
cx
ox2
pos
```

## References

[^1]: Mühlenbein, H. and Schlierkamp-Voosen, D.: Predictive Models for the Breeder Genetic Algorithm: I. Continuous Parameter Optimization. Evolutionary Computation, 1 (1), pp. 25-49, 1993.
