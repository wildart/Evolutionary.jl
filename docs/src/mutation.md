# Mutation

In genetic algorithms and evolutionary computation, **mutation** is a genetic operator used to maintain a diversity from one generation of a population to the next. It is analogous to biological mutation. Mutation alters one or more gene values in a chromosome from its initial state.
The purpose of mutation is to introduce diversity into the sampled population.

## Mutation Interface

All mutation operations have following call interface `mutation(individual)` where `individual` is the member of population. The `mutation` function returns an **in-place mutated** individual.

## Evolutionary Strategy

See [Strategies](@ref) section for detailed description of ES strategies.

List of ES mutation operations:

```@docs
gaussian(::AbstractVector, ::IsotropicStrategy)
gaussian(::AbstractVector, ::AnisotropicStrategy)
cauchy(::AbstractVector, ::IsotropicStrategy)
```

List of ES strategy mutation operations:

```@docs
gaussian(::IsotropicStrategy)
gaussian(::AnisotropicStrategy)
```


## Genetic Algorithm

### Binary Mutations

```@docs
flip
bitinversion
```

### Real-valued Mutations

```@docs
uniform(::Real)
gaussian(::Real)
domainrange
PM
MIPM
```

### Combinatorial Mutations

*Note: The combinatorial mutation operations are applicable to binary vectors.*

```@docs
inversion
insertion
swap2
scramble
shifting
```

## Differential Evolution

```@docs
Evolutionary.differentiation
```

## References

[^1]: Mühlenbein, H. and Schlierkamp-Voosen, D.: Predictive Models for the Breeder Genetic Algorithm: I. Continuous Parameter Optimization. Evolutionary Computation, 1 (1), pp. 25-49, 1993.

[^2]: Yao, Xin, and Yong Liu. "Fast evolution strategies." In International Conference on Evolutionary Programming, pp. 149-161. Springer, Berlin, Heidelberg, 1997.

[^3]: K. Deep, M. Thakur, "A new crossover operator for real coded genetic algorithms", Applied Mathematics and Computation 188, 2007, 895–912

[^4]: K. Deep, K. P. Singh, M. L. Kansal, and C. Mohan, "A real coded  genetic algorithm for solving integer and mixed integer optimization problems.", Appl. Math. Comput. 212, 505-518, 2009
