# Mutation

In genetic algorithms and evolutionary computation, **mutation** is a genetic operator used to maintain a diversity from one generation of a population to the next. It is analogous to biological mutation. Mutation alters one or more gene values in a chromosome from its initial state.
The purpose of mutation is to introduce diversity into the sampled population.

**Note:** All mutations are in-place operations, i.e. they modify a parameter object.

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

List of the binary mutation operations:


```@docs
flip
bitinversion
```

List of the real valued mutation operations:

```@docs
domainrange
```

List of the combinatorial mutation operations (applicable to binary vectors):

```@docs
inversion
insertion
swap2
scramble
shifting
```

## References

[^1]: Mühlenbein, H. and Schlierkamp-Voosen, D.: Predictive Models for the Breeder Genetic Algorithm: I. Continuous Parameter Optimization. Evolutionary Computation, 1 (1), pp. 25-49, 1993.

[^2]: Yao, Xin, and Yong Liu. "Fast evolution strategies." In International Conference on Evolutionary Programming, pp. 149-161. Springer, Berlin, Heidelberg, 1997.