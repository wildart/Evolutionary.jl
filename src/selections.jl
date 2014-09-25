# GA seclections
# ==============

# Rank-based fitness assignment
# sp - selective linear presure in [1.0, 2.0]
function ranklinear(sp::Float64)
    @assert 1.0 <= sp <= 2.0 "Selective pressure has to be in range [1.0, 2.0]."
    function rank(fitness::Vector{Float64}, N::Int)
        λ = float(length(fitness))
        idx = sortperm(fitness)
        ranks = similar(fitness)
        for i in 1:λ
            ranks[i] = (sp - 2.0*(sp - 1.0)*(i - 1.0) / (λ - 1.0)) / λ
        end
        return pselection(ranks[idx], N)
    end
    return rank
end

# (μ, λ)-uniform ranking sellection
function uniformranking(μ::Int)
    function uniformrank(fitness::Vector{Float64}, N::Int)
        λ = length(fitness)
        @assert μ < λ "μ should be less then $(λ)"
        ranks = zeros(fitness)
        ranks[1:μ] = 1/μ
        return pselection(ranks, N)
    end
    return uniformrank
end

# Roulette wheel (proportionate selection) selection
function roulette(fitness::Vector{Float64}, N::Int)
    prob = fitness./sum(fitness)
    return pselection(prob, N)
end

# Stochastic universal sampling (SUS)
function sus(fitness::Vector{Float64}, N::Int)
    F = sum(fitness)
    P = F/N
    start = P*rand()
    pointers = [start+P*i for i = 0:(N-1)]
    selected = Array(Int,N)
    i = c = 1
    for P in pointers
        while sum(fitness[1:i]) < P
            i += 1
        end
        selected[c] = i
        c += 1
    end
    return selected
end

# Truncation selection
function truncation{T <: Vector}(population::Vector{T}, N::Int)
    #TODO
end

# Tournament selection
function tournament{T <: Vector}(population::Vector{T}, N::Int)
    #TODO
end


# Utils: selection
function pselection(prob::Vector{Float64}, N::Int)
    cp = cumsum(prob)
    selected = Array(Int,N)
    for i in 1:N
        j = 1
        r = rand()
        while cp[j] < r
            j += 1
        end
        selected[i] = j
    end
    return selected
end

