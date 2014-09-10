module TestSAES3
	using Evolutionary
	using Base.Test

	# Implementation: (15/15+100)-σ-Self-Adaptation-ES
	# with non-isotropic mutation operator y' := y + (σ_1 N_1(0, 1), ..., σ_N N_N(0, 1))

	function create_strategy(σ::Vector{Float64}, τ::Float64)
		strategy = Dict{Symbol,Any}()
		strategy[:σ] = σ # strategy parameter per dimension
		strategy[:τ] = τ # self-adaptation learning rate
		return strategy
	end

	function rosenbrock{T <: Vector}(x::T)
	    return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
	end

	function recombine{T <: Vector, S}(population::Vector{T}, stgpop::Vector{S})
		s = length(population[1])
		obj = Array(eltype(T), s)
		σ = zeros(Float64, s)
		l = length(population)
		for i = 1:l
			obj += population[i]
			σ += stgpop[i][:σ]
		end
		return obj./l, create_strategy(σ./l, stgpop[1][:τ])
	end

	function mutate{T <: Vector}(recombinant::T, strategy::Dict)
		σ = strategy[:σ] * exp(strategy[:τ]*randn())
		vals = randn(length(recombinant)) .* σ
		recombinant += vals
		return recombinant, create_strategy(σ, strategy[:τ])
	end

	# Initial parameters
	solution = [1.0, 1.0]
	initial  = [0.0, 0.0]
	strategy = create_strategy(fill(1.0, length(initial)), 1/sqrt(2*length(initial)))

	# Optimization step
	result, fitness, cnt = es(rosenbrock, initial, strategy;
       	recombination = recombine, mutation = mutate, selection=:plus,
       	μ = 15, ρ = 15, λ = 100, iterations = 1000)

	@test_approx_eq_eps result solution 1e-5
	@test_approx_eq_eps fitness 0.0 1e-9
end