# Differential Evolution

```@docs
DE
```

## Description

The [Differential Evolution](https://en.wikipedia.org/wiki/Differential_evolution) is used for multidimensional real-valued functions but does not use the gradient of the problem being optimized, which means DE does not require the optimization problem to be differentiable, as is required by classic optimization methods such as gradient descent and quasi-newton methods. DE can therefore also be used on optimization problems that are not even continuous, are noisy, change over time, etc [^1].

## References

[^1]: K. V. Price and R. M. Storn and J. A. Lampinen, "Differential evolution: A practical approach to global optimization", Springer, 2005.
