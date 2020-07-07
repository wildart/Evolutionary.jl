# Development

```@meta
CurrentModule = Evolutionary
```

If you are contributing a new algorithm to this package, you need to know an internal API which allows to add a new algorithm without considerable changes to overall structure of the package.

## Adding an algorithm

If you're contributing a new algorithm, you shouldn't need to touch any of the code in src/api/optimize.jl. You should rather add a file named (solver is the name of the solver) `algo.jl` in src, and make sure that you define an optimizer **parameters** and **state** types, `initial_population` that initializes a population of `individual` objects, a state type that holds all variables that are (re)used throughout the iterative procedure, an `initial_state` that initializes such a state, and an `update_state!` method that does the actual work.


### Algorithm

Every optimization algorithm have to implement an algorithm parameters type derived from
 `AbstractOptimizer` type,  e.g. `struct Algo <: AbstractOptimizer end`, with appropriate fields, a default constructor with a keyword for each field.

Function `initial_state` returns an initial state for the algorithm, see [State](#state) section.
Function `update_state!` returns a `Bool` value. If the state update is *successfully* completed then the function returns `false`, otherwise `true`.

```@docs
AbstractOptimizer
initial_state
```

### State

Every optimization algorithm have to implement a state type derived from `AbstractOptimizerState` type, e.g. `struct AlgoState <: AbstractOptimizerState end`. All derived types should implement `value` and `minimizer` functions

```@docs
AbstractOptimizerState
value(::AbstractOptimizerState)
minimizer(::AbstractOptimizerState)
terminate(::AbstractOptimizerState)
```

### Population

The evolutionary algorithms require a collection of individuals, **population**, which the algorithm constantly modifies. The population collection type must be derived from the `AbstractVector` type. Function `initial_population` is used for implementing a strategy of population collection initialization.

The `initial_population` must accept two parameters:
- `method`, an algorithm object derived from `AbstractOptimizer` type
- `individual`, a description of an individual template used to create the population


Following population initialization strategies are available:

```@docs
initial_population
```

### Constraints

All constraints derived from the `AbstractConstraints` abstract type.
Usually the derived type wraps a `ConstraintBounds` object, so the

Following methods can be overridden for the derived types:

```@docs
value(::AbstractConstraints, ::AbstractObjective, x)
value(::AbstractConstraints, x)
```

Following auxillary functions are available for every derived type of `AbstractConstraints`.

```@docs
isfeasible(::AbstractConstraints, x)
```

Package provides following constrains implementations.

```@docs
Evolutionary.NoConstraints
BoxConstraints
PenaltyConstraints
WorstFitnessConstraints
MixedTypePenaltyConstraints
```
