using Evolutionary
using Test
using Random
using LinearAlgebra
Random.seed!(9874984737484)

for tests in [
            "sphere.jl",
            "rosenbrock.jl",
            "schwefel.jl",
            "rastrigin.jl",
            "n-queens.jl",
            "knapsack.jl"
]
    include(tests)
end
