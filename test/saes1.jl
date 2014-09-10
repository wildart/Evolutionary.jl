module TestSAES1
	using Evolutionary
	using Base.Test

	# Implementation: (15/5+100)-ES
	# with isotropic mutation operator y' := y + σ(N_1(0, 1), ..., N_N(0, 1))

	solution = [1.0, 1.0]
	initial  = [0.0, 0.0]
	strategy = Dict([:σ],[0.1])

	function rosenbrock{T <: Vector}(x::T)
	    return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
	end

	function recombine{T <: Vector, S}(population::Vector{T}, stgpop::Vector{S})
		stgy = stgpop[1]
		obj = Array(eltype(T), length(population[1]))
		l = length(population)
		for i = 1:l
			obj += population[i]
		end
		return obj./l, stgy
	end

	function mutate{T <: Vector}(recombinant::T, strategy::Dict)
		vals = randn(length(recombinant)) * strategy[:σ]
		recombinant += vals
		return recombinant, strategy
	end

	result, fitness, cnt = es(rosenbrock, initial, strategy;
		recombination = recombine, mutation = mutate,
		μ = 15, ρ = 5, λ = 100, iterations = 1000)

	@test_approx_eq_eps result solution 1e-2
	@test_approx_eq_eps fitness 0.0 1e-2
end