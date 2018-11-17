using LinearAlgebra
using Statistics

include("Core.jl")
include("CMAESCore.jl")

export CMAESIter

############################ New Interface ############################
# * less input parameters, like number of iterations, tol, verbose.
# * clean `call_back` support, with full access to runtime information.
# * support to implement parallism.

struct CMAESIter{PT<:AbstractOptProblem, T}
	cr::CMAESRuntime{T}
	prob::PT
end
function CMAESIter(prob::AbstractOptProblem, individual; num_population::Integer, num_offsprings::Integer)
	cr = CMAESRuntime(ellength(prob), individual, num_population, num_offsprings)
	CMAESIter(cr, prob)
end

function CMAESIter(lossfunc::Function, individual; num_population::Integer, num_offsprings::Integer)
	prob = GeneralOptProblem(lossfunc, length(individual))
	CMAESIter(prob, individual, num_population=num_population, num_offsprings=num_offsprings)
end

function Base.iterate(ci::CMAESIter, state=1)
	cmaes_step!(ci.cr, ci.prob, τ=init_τ(ci.prob), τ_c=init_τ_c(ci.prob), τ_σ=init_τ_σ(ci.prob)), state+1
end
