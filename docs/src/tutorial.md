```@meta
CurrentModule = Evolutionary
```

```@setup result
using Evolutionary
```

## Optimization
To show how the **Evolutionary** package can be used, we minimize the
[Rosenbrock function](http://en.wikipedia.org/wiki/Rosenbrock_function),
a classical test problem for numerical optimization. We'll assume that you've already
installed the **Evolutionary** package using Julia's package manager.

First, we load **Evolutionary** and define the Rosenbrock function:

```julia
using Evolutionary
f(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
```

Once we've defined this function, we can find the minimizer (the input that minimizes the objective) and the minimum (the value of the objective at the minimizer) using any of our favorite optimization algorithms. With a function defined,
we just specify a form of an individual `x` of the population for an evolutionary algorithm, and call `optimize` with a starting individual `x0` and a particular optimization algorithm, e.g. [`CMAES()`](@ref):

```julia
x0 = [0.0, 0.0];
Evolutionary.optimize(f, x0, CMAES())
```

## Configurable options

There are several options that simply take on some default values if the user doesn't provide any.

### Algorithm options

There are different algorithms available in `Evolutionary`, and they are all listed below. Notice that the constructors are written without input here, but they generally take keywords to tweak the way they work. See the pages describing each solver for more detail.

- [`GA()`](@ref)
- [`ES()`](@ref)
- [`CMAES()`](@ref)
- [`DE()`](@ref)

### General options

In addition to the algorithm, you can alter the behavior of the optimization procedure by using the following `Options` keyword arguments:
```@docs
Options
```

We currently recommend the statically dispatched interface by using the `Evolutionary.Options` constructor:

```julia
res = Evolutionary.optimize(x->-sum(x),
                            BitVector(zeros(3)),
                            GA(),
                            Evolutionary.Options(iterations=10))
```


## Obtaining results

After we have our results in `res` object, we can use the API for getting optimization results. This consists of a collection of functions. They are not exported, so they have to be prefixed by `Evolutionary.`. Say we do the following optimization:

```@repl result
res = Evolutionary.optimize(x->-sum(x), BitVector(zeros(3)), GA())
```

You can inspect the result by using a collection of the auxiliary functions, e.g. the `minimizer` and `minimum` of the objective functions, which can be found using

```@repl result
Evolutionary.minimizer(res)
Evolutionary.minimum(res)
```

#### Complete list of functions

An `OptimizationResults` interface for representing an optimization result.

```@docs
OptimizationResults
summary(::OptimizationResults)
minimizer(::OptimizationResults)
minimum(::OptimizationResults)
iterations(::OptimizationResults)
iteration_limit_reached(::OptimizationResults)
trace(::OptimizationResults)
f_calls(::OptimizationResults)
tol(::OptimizationResults)
```

An implementation of the result object for evolutionary optimizations.
```@docs
EvolutionaryOptimizationResults
converged(::EvolutionaryOptimizationResults)
tol(::EvolutionaryOptimizationResults)
```

## Trace

When `store_trace` and/or `show_trace` options are set to `true` in the `Option`(@ref) object, an optimization trace is either captured and/or shown on the screen. By default, only the current state minimum value is displayed in the trace. In order to extend trace record, you need to override [`trace!`](@ref) function providing specialize function behavior on one of specific parameters.

```@docs
trace!(::Dict{String,Any}, Any, Any, Any, Any, Any)
```

Commonly, you would define a specializations of a `state`, `population`, or `method` parameters of `trace!` function, e.g.

```julia
function trace!(record::Dict{String,Any}, objfun, state, population, method::CMAES, options)
    record["σ"] = state.σ
end
```
