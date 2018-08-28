# GA seclections
# ==============

# Rank-based fitness assignment
# sp - selective linear presure in [1.0, 2.0]
function ranklinear(sp::Float64)
    @assert 1.0 <= sp <= 2.0 "Selective pressure has to be in range [1.0, 2.0]."
    function rank(fitness::Vector{Float64}, N::Int)
        λ = length(fitness)
        idx = sortperm(fitness)
        ranks = zeros(λ)
        for i in 1:λ
            ranks[i] = ( 2.0- sp + 2.0*(sp - 1.0)*(idx[i] - 1.0) / (λ - 1.0) ) / λ
        end
        return pselection(ranks, N)
    end
    return rank
end

# (μ, λ)-uniform ranking selection
function uniformranking(μ::Int)
    function uniformrank(fitness::Vector{Float64}, N::Int)
        λ = length(fitness)
        idx = sortperm(fitness, rev=true)
        @assert μ < λ "μ should be less then $(λ)"
        ranks = zeros(fitness)
        for i in 1:μ
            ranks[idx[i]] = 1/μ
        end
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
    selected = Array{Int}(undef, N)
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
function truncation(population::Vector{T}, N::Int) where {T <: Vector}
    #TODO
end

# Tournament selection
function tournament(groupSize :: Int)
    groupSize <= 0 && error("Group size needs to be positive")
    function tournamentN(fitness::Vector{Float64}, N::Int)
        selection = Array{Int}(N)

        nFitness = length(fitness)

        for i in 1:N
            contender = unique(rand(1:nFitness, groupSize))
            while length(contender) < groupSize
                contender = unique(vcat(contender, rand(1:nFitness, groupSize - length(contender))))
            end

            winner = first(contender)
            winnerFitness = fitness[winner]
            for idx = 2:groupSize
                c = contender[idx]
                if winnerFitness < fitness[c]
                    winner = c
                    winnerFitness = fitness[c]
                end
            end

            selection[i] = winner
        end
        return selection
    end
    return tournamentN
end


# Utils: selection
function pselection(prob::Vector{Float64}, N::Int)
    cp = cumsum(prob)
    selected = Array{Int}(undef, N)
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
