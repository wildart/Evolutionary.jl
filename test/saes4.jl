module TestSAES4
	using Evolutionary
	using Base.Test

	# Implementation: (μ/μ_I, λ)-σ-Self-Adaptation-ES with μ = 3, λ = 12
	# with isotropic mutation operator y' := y + σ(N_1(0, 1), ..., N_N(0, 1))

	sphere{T <: Vector}(x::T) = sum(x.*x)

	terminate(strategy) = strategy[:σ] < 1e-10	

	function recombine{T <: Vector, S}(population::Vector{T}, stgpop::Vector{S})
		σ = 0.0
		obj = zeros(eltype(T), length(population[1]))
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
		return recombinant + vals, Dict([:σ, :τ],[σ, strategy[:τ]])
	end

	# Initial parameters
	n = 30
	solution = [1.0, 1.0]
	initial  = ones(n)
	strategy = Dict([:σ, :τ],[1.0, 1/sqrt(2*length(initial))]) # self-adaptation learning rate

	# Optimization step
	result, fitness, cnt = es(sphere, initial, strategy;
       	recombination = recombine, mutation = mutate, termination = terminate, selection=:comma,
       	μ = 3, ρ = 3, λ = 12, iterations = 1000)

	@test length(result) == n
	@test cnt < 1000
	@test_approx_eq_eps fitness 0.0 1e-15
end