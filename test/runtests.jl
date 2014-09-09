using Evolutionary
using Base.Test

# write your own tests here
@test 1 == 1


function rosenbrock(x::Vector)
    return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end


function Evolutionary.initialize(rng::AbstractRNG)
    return Individual([0.0, 0.0])
end

function Evolutionary.recombine(population::Vector{Individual})
end

function Evolutionary.mutate(recombinant::Individual)
end


initial = [0.0, 0.0]
solutions = {[1.0, 1.0]}