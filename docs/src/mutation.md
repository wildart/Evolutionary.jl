# Mutation

In genetic algorithms and evolutionary computation, **mutation** is a genetic operator used to maintain a diversity from one generation of a population to the next. It is analogous to biological mutation. Mutation alters one or more gene values in a chromosome from its initial state.
The purpose of mutation is to introduce diversity into the sampled population.

List of the evolutionary strategy strategy mutation operations:

```@docs
isotropicSigma
anisotropicSigma
```

List of the evolutionary strategy population mutation operations:

```@docs
isotropic
anisotropic
```

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
