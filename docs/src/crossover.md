# Crossover

In genetic algorithms and evolutionary computation, **crossover**, also called **recombination**, is a genetic operator used to combine the genetic information of two parents to generate new offspring.


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
