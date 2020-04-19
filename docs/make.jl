using Documenter, Evolutionary

makedocs(
    modules = [Evolutionary],
    doctest = false,
    clean = true,
    sitename = "Evolutionary.jl",
    pages = [
        "Home" => "index.md",
        "Tutorial" => "tutorial.md",
        "Algorithms" => [
            "Genetic Algorithm" => "ga.md",
            "Evolution Strategy" => "es.md",
            "CMA-ES" => "cmaes.md",
        ],
        "Operations" => [
            "Selection" => "selection.md",
            "Crossover" => "crossover.md",
            "Mutation" => "mutation.md",
        ],
        "Development" => "dev.md",
    ]
)

deploydocs(repo = "github.com/wildart/Evolutionary.jl.git")
