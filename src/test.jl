include("cmaes.new.jl")
using Test

rosenbrock(x::Vector{Float64}) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2

function test_result(result::Vector, fitness::Float64, N::Int, tol::Float64)
    @test length(result) == N
	@test ≈(result, ones(N), atol=tol)
    @test ≈(fitness, 0.0, atol=tol)
end

struct RosenBrock<:AbstractOptProblem
	N::Int
end
ellength(rb::RosenBrock) = rb.N
loss(::RosenBrock, x::AbstractVector) = rosenbrock(x)

@testset "CMA-ES RosenBrock" begin
	# Testing: CMA-ES
	N = 2
	# termination condition
	ci = CMAESIter(rosenbrock, randn(2), num_offsprings=12, num_population=3)
	for (count, cr) in enumerate(ci)
		if count == 1_1000 || cr.σ<1e-10 break end
		println("BEST: $(cr.fitpop[1]): $(cr.σ)")
	end
	test_result(best(ci.cr)..., N, 1e-1)

	rb = RosenBrock(N)
	ci = CMAESIter(rb, rand_individual(rb), num_offsprings=12, num_population=3)
	for (count, cr) in enumerate(ci)
		if count == 1_1000 || cr.σ<1e-10 break end
		println("BEST: $(cr.fitpop[1]): $(cr.σ)")
	end
	test_result(best(ci.cr)..., N, 1e-1)
end
