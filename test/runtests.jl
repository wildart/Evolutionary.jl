using Evolutionary
using Base.Test

srand(9874984737484)

@testset "Evolutionary" for tests in [
            "sphere.jl",
            "rosenbrock.jl",
            "schwefel.jl",
            "rastrigin.jl",
            "n-queens.jl",
            "knapsack.jl"
]
    include(tests)
end
