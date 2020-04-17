using Documenter, Evolutionary

makedocs(
	modules = [Evolutionary],
	doctest = false,
	clean = true,
	sitename = "Evolutionary.jl",
	pages = [
		"Home" => "index.md",
		# "Interface" => "interface.md",
		# "Methods" => [
		# 	"Genetic Algorithms" => "ga.md",
		# 	"Evolution Strategy" => "es.md",
		#  	"CMAES" => "cmaes.md",
		# ],
	]
)

# deploydocs(repo = "github.com/wildart/Evolutionary.jl.git")
