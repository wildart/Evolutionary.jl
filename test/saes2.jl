module TestSAES2
	using Evolutionary
	using Base.Test

	# Implementation: (15/15+100)-σ-Self-Adaptation-ES
	# with isotropic mutation operator y' := y + σ(N_1(0, 1), ..., N_N(0, 1))

	solution = [1.0, 1.0]
	initial  = [0.0, 0.0]
	strategy = Dict([:σ, :τ],[1.0, 1/sqrt(2*length(initial))]) # self-adaptation learning rate

	function rosenbrock{T <: Vector}(x::T)
	    return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
	end

	function recombine{T <: Vector, S}(population::Vector{T}, stgpop::Vector{S})
		σ = 0.0
		obj = Array(eltype(T), length(population[1]))
		l = length(population)
		for i = 1:l
			obj += population[i]
			σ += stgpop[i][:σ]
		end
		return obj./l, Dict([:σ, :τ],[σ/l, stgpop[1][:τ]])
	end

	function mutate{T <: Vector}(recombinant::T, strategy::Dict)
		σ = strategy[:σ] * exp(strategy[:τ]*randn())
		vals = randn(length(recombinant)) * σ
		recombinant += vals
		return recombinant, Dict([:σ, :τ],[σ, strategy[:τ]])
	end

	result, fitness, cnt = es(rosenbrock, initial, strategy;
       	recombination = recombine, mutation = mutate, selection=:plus,
       	μ = 15, ρ = 15, λ = 100, iterations = 1000)

	@test_approx_eq_eps result solution 1e-3
	@test_approx_eq_eps fitness 0.0 1e-5
end