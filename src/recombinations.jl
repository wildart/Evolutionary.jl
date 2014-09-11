# Recombinations
# ==============
function average{T <: Vector}(population::Vector{T})
    obj = zeros(eltype(T), length(population[1]))
    l = length(population)
    for i = 1:l
        obj += population[i]
    end
    return obj./l
end

# Strategy recombinations
# =======================
function averageSigma1{S <: Strategy}(ss::Vector{S})
    s = copy(ss[1])
    σ = 0.0
    l = length(ss)
    for i = 1:l
        σ += ss[i][:σ]
    end
    s[:σ] = σ/l
    return s
end

function averageSigmaN{S <: Strategy}(ss::Vector{S})
    s = copy(ss[1])
    σ = zeros(length(ss[1][:σ]))
    l = length(ss)
    for i = 1:l
        σ += ss[i][:σ]
    end
    s[:σ] = σ./l
    return s
end