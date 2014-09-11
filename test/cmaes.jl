module TestCMAES
    using Evolutionary
    using Base.Test
    
    # Rosenbrock function
    function rosenbrock{T <: Vector}(x::T)
        return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
    end

    N = 2
    # result, fitness, cnt = cmaes(rosenbrock, zeros(N), strategy(σ = 1.0, τ = 1/sqrt(2*N));
    #     recombination = average, srecombination = averageSigma1, 
    #     mutation = isotropic, smutation = isotropicSigma,
    #     termination = terminate, 
    #     μ = 3, λ = 12, iterations = 1000)
    
    # Schwefel's ellipsoid test function, a moderately conditioned ellipsoid with a dominating isolated eigenvalue
    function schwefel{T <: Vector}(x::T)
        out = 0
        for i = 1:length(x)
            out += sumabs2(x[1:i])
        end
        return out
    end
    terminate(strategy) = strategy[:σ] < 1e-10

    N = 30
    # result, fitness, cnt = cmaes(schwefel, ones(N), strategy(σ = 1.0, τ = 1/sqrt(2*N));
    #     recombination = average, srecombination = averageSigma1, 
    #     mutation = isotropic, smutation = isotropicSigma,
    #     termination = terminate, 
    #     μ = 3, λ = 12, iterations = 1000)

end