# GA selections
# ==============

# Rank-based fitness assignment
# sp - selective linear presure in [1.0, 2.0]
function ranklinear(sp::Float64)
    @assert 1.0 <= sp <= 2.0 "Selective pressure has to be in range [1.0, 2.0]."
    function rank(fitness::Vector{<:Real}, N::Int)
        λ = length(fitness)
        rank = sortperm(fitness)
        
        prob = Vector{Float64}(undef, λ)
        for i in 1:λ
            prob[i] = ( 2.0- sp + 2.0*(sp - 1.0)*(rank[i] - 1.0) / (λ - 1.0) ) / λ
        end

        return pselection(prob, N)
    end
    return rank
end

# (μ, λ)-uniform ranking selection
function rankuniform(fitness::Vector{<:Real}, N::Int)
    μ = N
    λ = length(fitness)
    @assert μ <= λ "μ must be ≤ λ = $(λ)"
    
    prob = zeros(λ)
    rank = sortperm(fitness, rev=true)
    prob[rank[1:μ]] .= 1/μ

    return pselection(prob, N)
end

# Roulette wheel (proportionate selection) selection
function roulette(fitness::Vector{<:Real}, N::Int)
    prob = fitness./sum(fitness)
    return pselection(prob, N)
end

# Stochastic universal sampling (SUS)
function sus(fitness::Vector{<:Real}, N::Int)
    selected = Vector{Int}(undef, N)
    
    F = sum(fitness)
    P = F/N
    
    start = P*rand()
    pointers = [start+P*i for i = 0:(N-1)]
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
function truncation(fitness::Vector{<:Real}, N::Int)
    λ = length(fitness)
    @assert λ >= N "Cannot select more then $(λ) elements"
    idx = sortperm(fitness, rev=true)
    return idx[1:N]
end

# Tournament selection
function tournament(groupSize::Int)
    @assert groupSize > 0 "Group size must be positive"
    function tournamentN(fitness::Vector{<:Real}, N::Int)
        selection = Vector{Int}(undef, N)

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
function pselection(prob::Vector{<:Real}, N::Int)
    selected = Vector{Int}(undef, N)

    cp = cumsum(prob)
    @assert cp[end] ≈ 1 "Sum of probability vector must equal 1"

    for i in 1:N
        selected[i] = vlookup(cp, rand())
    end
    return selected
end

# Utils: vlookup
function vlookup(range::Vector{<:Number}, value::Number)
    for i in eachindex(range)
        if range[i] >= value
            return i
        end
    end

    return -1
end
