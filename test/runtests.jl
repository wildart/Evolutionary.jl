using Evolutionary
using Test
using Random
using LinearAlgebra
using Statistics
using StableRNGs

# Guard against accidental piracy from `import`
@test Evolutionary.contains !== Base.contains

for tests in [
    "types.jl",
    "objective.jl",
    "interface.jl",
    "selections.jl",
    "recombinations.jl",
    "mutations.jl",
    "sphere.jl",
    "rosenbrock.jl",
    "schwefel.jl",
    "rastrigin.jl",
    "n-queens.jl",
    "knapsack.jl",
    "onemax.jl",
    "moea.jl",
    "regression.jl",
    "gp.jl"
]
    include(tests)
end
