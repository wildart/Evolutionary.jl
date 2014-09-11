module TestSAES
	using Evolutionary
	using Base.Test
	
	solution = [1.0, 1.0]
	initial  = [0.0, 0.0]
	N = length(initial)

	function rosenbrock{T <: Vector}(x::T)
	    return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
	end

	# Implementation: (15/5+100)-ES
	# with isotropic mutation operator y' := y + σ(N_1(0, 1), ..., N_N(0, 1))
	result, fitness, cnt = es(rosenbrock, initial, strategy(σ = 0.1);
		recombination = average, mutation = isotropic,
		μ = 15, ρ = 5, λ = 100, iterations = 1000)
	println("(15/5+100)-ES => F: $(fitness), C: $(cnt), OBJ: $(result)")

	@test_approx_eq_eps result solution 1e-2
	@test_approx_eq_eps fitness 0.0 1e-5

	# Implementation: (15/15+100)-σ-Self-Adaptation-ES
	# with isotropic mutation operator y' := y + σ(N_1(0, 1), ..., N_N(0, 1))
	result, fitness, cnt = es(rosenbrock, initial, strategy(σ = 1.0, τ = 1/sqrt(2*N));
		recombination = average, srecombination = averageSigma1, 
		mutation = isotropic, smutation = isotropicSigma,
		μ = 15, ρ = 15, λ = 100, iterations = 1000)
	println("(15/15+100)-σ-SA-ES-IS => F: $(fitness), C: $(cnt), OBJ: $(result)")

	@test_approx_eq_eps result solution 1e-3
	@test_approx_eq_eps fitness 0.0 1e-5

	# Implementation: (15/15+100)-σ-Self-Adaptation-ES
	# with non-isotropic mutation operator y' := y + (σ_1 N_1(0, 1), ..., σ_N N_N(0, 1))	
	result, fitness, cnt = es(rosenbrock, initial, strategy(σ = .5ones(N), τ = 1/sqrt(2*N), τ0 = 1/sqrt(N));
		recombination = average, srecombination = averageSigmaN, 
		mutation = anisotropic, smutation = anisotropicSigma,
		μ = 15, ρ = 15, λ = 100, iterations = 1000)
	println("(15/15+100)-σ-SA-ES-AS => F: $(fitness), C: $(cnt), OBJ: $(result)")

	@test_approx_eq_eps result solution 1e-1
	@test_approx_eq_eps fitness 0.0 1e-2


	# Implementation: (μ/μ_I, λ)-σ-Self-Adaptation-ES with μ = 3, λ = 12
	# with isotropic mutation operator y' := y + σ(N_1(0, 1), ..., N_N(0, 1))
	N = 30
	sphere{T <: Vector}(x::T) = sum(x.*x)
	terminate(strategy) = strategy[:σ] < 1e-10
	result, fitness, cnt = es(sphere, ones(N), strategy(σ = 1.0, τ = 1/sqrt(2*N));
		recombination = average, srecombination = averageSigma1, 
		mutation = isotropic, smutation = isotropicSigma,
       	termination = terminate, selection=:comma,
       	μ = 3, ρ = 3, λ = 12, iterations = 1000)
	println("(μ/μ_I,λ)-σ-SA-ES => F: $(fitness), C: $(cnt)")

	@test length(result) == N
	@test cnt < 1000
	@test_approx_eq_eps fitness 0.0 1e-15
end