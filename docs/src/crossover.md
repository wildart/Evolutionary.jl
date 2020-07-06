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
uniformbin
exponential
```

Real valued crossovers:
```@docs
identity
discrete
waverage
intermediate
line
HX
LX
MILX
```

Permutation crossovers:
```@docs
PMX
OX1
CX
OX2
POS
```

## References

[^1]: H. Mühlenbein, D. Schlierkamp-Voosen, "Predictive Models for the Breeder Genetic Algorithm: I. Continuous Parameter Optimization". Evolutionary Computation, 1 (1), pp. 25-49, 1993.

[^2]: K. V. Price and R. M. Storn and J. A. Lampinen, "Differential evolution: A practical approach to global optimization", Springer, 2005.

[^3]: Z. Michalewicz, T. Logan,  S. Swaminathan. "Evolutionary operators for continuous convex parameter spaces." Proceedings of the 3rd Annual conference on Evolutionary Programming, 1994.

[^4]: K. Deep, M. Thakur, "A new crossover operator for real coded genetic algorithms", Applied Mathematics and Computation 188, 2007, 895–912

[^5]: K. Deep, K. P. Singh, M. L. Kansal, and C. Mohan, "A real coded  genetic algorithm for solving integer and mixed integer optimization problems.", Appl. Math. Comput. 212, 505-518, 2009
