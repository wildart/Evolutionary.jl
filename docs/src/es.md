# Evolution Strategies

```@docs
ES
```

## Description

The [Evolution Strategy](https://en.wikipedia.org/wiki/Evolution_strategy)  is  is an optimization technique based on ideas of evolution.

Evolution strategies use natural problem-dependent representations, and primarily [Mutation](@ref) and [Selection](@ref), as search operators.

The canonical versions of the ES are denoted by **(μ/ρ,λ)-ES** and **(μ/ρ+λ)-ES**, respectively. Here **μ** denotes the number of parents, **ρ ≤ μ** the mixing number (i.e., the number of parents involved in the procreation of an offspring), and **λ** the number of offspring. The parents are deterministically selected (i.e., deterministic survivor selection) from the (multi-)set of either the offspring, referred to as **comma**-selection (μ<λ must hold), or both the parents and offspring, referred to as **plus**-selection [^1].

## References

[^1]: [http://www.scholarpedia.org/article/Evolution_strategies](http://www.scholarpedia.org/article/Evolution_strategies)
