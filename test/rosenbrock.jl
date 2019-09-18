@testset "Rosenbrock" begin

    function test_result(result::Vector, fitness::Float64, N::Int, tol::Float64)
        @test length(result) == N
    	@test ≈(result, ones(N), atol=tol)
        @test ≈(fitness, 0.0, atol=tol)
    end

	# Objective function
	rosenbrock(x::Vector{Float64}) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2

	# Parameters
	N = 2
	terminate(σ) = σ < 1e-10

	# Testing: (15/5+100)-ES
	# with isotropic mutation operator y' := y + σ(N_1(0, 1), ..., N_N(0, 1))
	result, fitness, cnt = es(rosenbrock, Vector{Float64}(undef, N);
        initStrategy = strategy(σ = 1.0),
        recombination = average, mutation = isotropic,
        μ = 15, ρ = 5, λ = 100, iterations = 1000)
	println("(15/5+100)-ES => F: $(fitness), C: $(cnt), OBJ: $(result)")
	test_result(result, fitness, N, 1e-1)

	# Testing: (15/15+100)-σ-Self-Adaptation-ES
	# with isotropic mutation operator y' := y + σ(N_1(0, 1), ..., N_N(0, 1))
	result, fitness, cnt = es(rosenbrock, Vector{Float64}(undef, N);
		initStrategy = strategy(σ = 1.0, τ = 1/sqrt(2*N)),
		creation = (n -> 0.5 .* rand(Float64, n)),
		recombination = average, srecombination = averageSigma1,
		mutation = isotropic, smutation = isotropicSigma,
		μ = 15, λ = 100, iterations = 1000)
	println("(15/15+100)-σ-SA-ES-IS => F: $(fitness), C: $(cnt), OBJ: $(result)")
	test_result(result, fitness, N, 1e-1)

	# Testing: (15/15+100)-σ-Self-Adaptation-ES
	# with non-isotropic mutation operator y' := y + (σ_1 N_1(0, 1), ..., σ_N N_N(0, 1))
	result, fitness, cnt = es(rosenbrock, Vector{Float64}(undef, N);
		initPopulation = [rand(N) for _ in 1:15],
        initStrategy = strategy(σ = .5ones(N), τ = 1/sqrt(2*N), τ0 = 1/sqrt(N)),
		recombination = average, srecombination = averageSigmaN,
		mutation = anisotropic, smutation = anisotropicSigma,
		μ = 15, λ = 100, iterations = 1000)
	println("(15/15+100)-σ-SA-ES-AS => F: $(fitness), C: $(cnt), OBJ: $(result)")
	test_result(result, fitness, N, 1e-1)

	# Testing: CMA-ES
	result, fitness, cnt = cmaes(rosenbrock, Vector{Float64}(undef, N);
    	μ = 3, λ = 12, iterations = 100_000, tol = 1e-3)
    println("(3/3,12)-CMA-ES => F: $(fitness), C: $(cnt), OBJ: $(result)")
    test_result(result, fitness, N, 1e-1)

    # Testing: GA
    result, fitness, cnt = ga(rosenbrock, Vector{Float64}(undef, N);
    	populationSize = 100,
		ɛ = 0.1,
        selection = sus,
        crossover = intermediate(0.25),
        mutation = domainrange(fill(0.5,N)))
    println("GA(p=50,x=0.8,μ=0.1) => F: $(fitness), C: $(cnt), OBJ: $(result)")
    test_result(result, fitness, N, 3e-1)

end
