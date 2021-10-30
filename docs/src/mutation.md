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
BGA
PM
MIPM
PLM
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

## Genetic Programming

```@docs
subtree
point
hoist
shrink
```

## References

[^1]: Mühlenbein, H. and Schlierkamp-Voosen, D., "Predictive Models for the Breeder Genetic Algorithm: I. Continuous Parameter Optimization", Evolutionary Computation, 1 (1), 25-49, 1993.

[^2]: Yao, Xin, and Yong Liu, "Fast evolution strategies", In International Conference on Evolutionary Programming, 149-161, Springer, 1997.

[^3]: K. Deep, M. Thakur, "A new crossover operator for real coded genetic algorithms", Applied Mathematics and Computation 188, 895-912, 2007.

[^4]: K. Deep, K. P. Singh, M. L. Kansal, and C. Mohan, "A real coded  genetic algorithm for solving integer and mixed integer optimization problems", Appl. Math. Comput. 212, 505-518, 2009

[^5]: K. E. Kinnear, Jr., "Evolving a sort: Lessons in genetic programming", In Proceedings of the 1993 International Conference on Neural Networks, vol 2, 881-888, IEEE Press, 1993.

[^6]: B. McKay, M. J. Willis, and G. W. Barton., "Using a tree structured genetic algorithm to perform symbolic regression", GALESIA, vol 414, 487-492, 1995.

[^7]: K. E. Kinnear, Jr., "Fitness landscapes and difficulty in genetic programming", In Proceedings of the 1994 IEEE World Conference on Computational Intelligence, vol 1, 142-147, IEEE Press, 1994.

[^8]: P. J. Angeline, "An investigation into the sensitivity of genetic programming to the frequency of leaf selection during subtree crossover", Genetic Programming 1996: Proceedings of the First Annual Conference, 21–29, 1996.

[^9]: K. Deb, R. B. Agrawal, "Simulated Binary Crossover for Continuous Search Space", Complex Syst., 9., 1995