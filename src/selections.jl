##### selections.jl #####

# In this file you will all the functions regarding population selection.
# All functions can be used for both Evolution Strategies and Genetic Algorithms.

####################################################################

# GA selections
# ==============

# Rank-based fitness assignment (RBS)
# sp - selective linear presure in [1.0, 2.0]
function ranklinear(sp ::Float64)
    @assert 1.0 <= sp <= 2.0 "Selective pressure has to be in range [1.0, 2.0]."
    function rank(fitness ::Vector{<:Real}, N ::Int)
        λ = length(fitness)
        idx = sortperm(fitness)
        ranks = zeros(λ)
        for i in 1:λ
            @inbounds ranks[i] = ( 2 - sp + 2*(sp-1)*(idx[i]-1) / (λ-1) ) / λ
        end
        return pselection(ranks, N)
    end
    return rank
end

# (μ, λ)-uniform ranking selection (URS)
function uniformranking(μ ::Int)
    function uniformrank(fitness ::Vector{<:Real}, N ::Int)
        λ = length(fitness)
        idx = sortperm(fitness, rev=true)
        @assert μ < λ "μ should be less then $(λ)"
        ranks = similar(fitness, Float64)
        for i in 1:μ
            @inbounds ranks[idx[i]] = 1/μ
        end
        return pselection(ranks, N)
    end
    return uniformrank
end

# Roulette wheel (proportionate selection) selection (RWS)
function roulette(fitness ::Vector{<:Real}, N ::Int)
    prob = fitness ./ sum(fitness)
    return pselection(prob, N)
end

# Stochastic universal sampling selection (SUSS)
function sus(fitness ::Vector{<:Real}, N ::Int)
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

# Truncation selection (TrS)
function truncation(fitness ::Vector{<:Real}, N ::Int)
    λ = length(fitness)
    @assert λ >= N "Cannot select more than $(λ) elements"
    idx = sortperm(fitness, rev=true)
    return idx[1:N]
end

# Tournament selection (ToS)
function tournament(groupSize ::Int)
    @assert groupSize > 0 "Group size must be positive"
    function tournamentN(fitness ::Vector{<:Real}, N ::Int)
        selection = fill(0,N)

        nFitness = length(fitness)

        for i in 1:N
            contender = unique(rand(1:nFitness, groupSize))
            while length(contender) < groupSize
                contender = unique(vcat(contender, rand(1:nFitness, groupSize - length(contender))))
            end

            winner        = first(contender)
            winnerFitness = fitness[winner]
            for idx = 2:groupSize
                c = contender[idx]
                if winnerFitness < fitness[c]
                    winner        = c
                    @inbounds winnerFitness = fitness[c]
                end
            end

            selection[i] = winner
        end
        return selection
    end
    return tournamentN
end

####################################################################

# Utils: selection
function pselection(prob ::Vector{<:Real}, N ::Int)
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

