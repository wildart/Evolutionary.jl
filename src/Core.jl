export AbstractOptProblem, AbstractRuntime, loss, ellength, rand_individual, AbstractRuntime, GeneralOptProblem
"""
    AbstractOptProblem

optimization problem, required interfaces are `ellength`, `loss`.
optional interfaces are `rand_individual`, `populate!`.
"""
abstract type AbstractOptProblem end

"""
    ellength(prob::AbstractOptProblem) -> Int

Length of an individual as the input of an optimization problem.
"""
function ellength end

"""
    loss(prob::AbstractOptProblem, x) -> Real

Loss function for an optimization problem.
"""
function loss end

struct GeneralOptProblem{FT} <: AbstractOptProblem
	lossfunc::FT
	N::Int
end
ellength(opt::GeneralOptProblem) = opt.N
loss(opt::GeneralOptProblem, x) = opt.lossfunc(x)

# where you can implement parallism
function populate!(fitoff::AbstractVector, offspring::AbstractVector, prob::AbstractOptProblem)
    for i in 1:length(fitoff)
        fitoff[i] = loss(prob, offspring[i]) # Evaluate fitness
    end
end

"""randomly generate an individual"""
rand_individual(prob::AbstractOptProblem) = randn(prob |> ellength)

"""
    AbstractRuntime

runtime information for optimization, required interfaces are `best`.
"""
abstract type AbstractRuntime end

"""
    best(prob::AbstractRuntime) -> (individual, cost)

The best suited individual, and its cost.
"""
function best end
