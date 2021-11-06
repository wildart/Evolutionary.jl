using Evolutionary
using Test
using Random
using LinearAlgebra
using Statistics
Random.seed!(9874984737486)

for tests in [
    "nondifferentiable.jl",
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
