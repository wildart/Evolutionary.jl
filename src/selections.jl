# GA seclections
# ==============

"""
    ranklinear(sp::Real)

Returns a rank-based fitness selection function, see [Selection Interface](@ref), with the selective pressure value `sp`.

In rank-based fitness selection, the population is sorted according to the objective values. The fitness assigned to each individual depends only on its position in the individuals rank and not on the actual objective value [^1].

Consider ``M`` the number of individuals in the population, ``P`` the position of an individual in this population (least fit individual has ``P = 1``, the fittest individual ``P = M``) and ``SP`` the selective pressure. The fitness value for an individual is calculated as:

``Fitness(P) = 2 - SP + \\frac{2(SP - 1)(P - 1)}{(M - 1)}``

Linear ranking allows values of selective pressure in [1.0, 2.0].

"""
function ranklinear(sp::Real)
    @assert 1.0 <= sp <= 2.0 "Selective pressure has to be in range [1.0, 2.0]."
    function rank(fitness::Vector{<:Real}, N::Int)
        λ = length(fitness)
        idx = sortperm(fitness)
        ranks = zeros(λ)
        for i in 1:λ
            ranks[i] = ( 2 - sp + 2*(sp - 1)*(idx[i] - 1) / (λ - 1) ) / λ
        end
        return pselection(ranks, N)
    end
    return rank
end

"""
    uniformranking(μ)

Returns a (μ, λ)-uniform ranking selection function, see [Selection Interface](@ref) with the best individuals parameter `μ`.

In uniform ranking, the best ``\\mu`` individuals are assigned a selection probability of ``1/\\mu`` while the rest them are discarded [^2].
"""
function uniformranking(μ::Int)
    function uniformrank(fitness::Vector{<:Real}, N::Int)
        λ = length(fitness)
        idx = sortperm(fitness)
        @assert μ <= λ "μ should be no larger then $(λ)"
        ranks = zeros(length(fitness))
        for i in 1:μ
            ranks[idx[i]] = 1/μ
        end
        return pselection(ranks, N)
    end
    return uniformrank
end

"""
    roulette(fitness)

Roulette wheel (fitness proportionate, FPS) selection from `fitness` collection.

In roulette selection, the fitness level is used to associate a probability of selection with each individual. If ``f_i`` is the fitness of individual ``i`` in the population, its probability of being selected is ``p_i = \\frac{f_i}{\\Sigma_{j=1}^{M} f_j}``, where ``M`` is the number of individuals in the population.

*Note:* Best used in maximization context.

"""
function roulette(fitness::Vector{<:Real}, N::Int)
    absf = abs.(fitness)
    prob = absf./sum(absf)
    return pselection(prob, N)
end

"""
    rouletteinv(fitness)

Fitness proportionate selection (FPS) or roulette wheel for inverse `fitness` values. Best used in minimization context.
"""
rouletteinv(fitness::Vector{<:Real}, N::Int) = roulette(1.0 ./ fitness, N)

"""
    sus(fitness, N)

Stochastic universal sampling (SUS) provides zero bias and minimum spread [^3]. SUS is a development of fitness proportionate selection (FPS).
Using a comb-like ruler, SUS starts from a small random number, and chooses the next candidates from the rest of population remaining, not allowing the fittest members to saturate the candidate space. The individuals are mapped to contiguous segments of a line, such that each individual's segment is equal in size to its fitness exactly as in roulette-wheel selection. Here equally spaced pointers are placed over the line as many as there are individuals to be selected.

Consider ``N`` the number of individuals to be selected, then the distance between the pointers are ``1/N`` and the position of the first pointer is given by a randomly generated number in the range ``[0, 1/N]``.

*Note*: Best used in maximization context.

"""
function sus(fitness::Vector{<:Real}, N::Int)
    F = sum(abs, fitness)
    P = F/N
    start = P*rand()
    pointers = [start+P*i for i = 0:(N-1)]
    selected = Array{Int}(undef, N)
    i = c = 1
    for P in pointers
        while sum(abs, fitness[1:i]) < P
            i += 1
        end
        selected[c] = i
        c += 1
    end
    return selected
end

"""
    susinv(fitness)

Inverse fitness SUS. Best used in minimization context.
"""
susinv(fitness::Vector{<:Real}, N::Int) = sus(1.0 ./ fitness, N)

"""
    truncation(fitness, N)

Truncation selection returns first `N` of best `fitness` inividuals
"""
function truncation(fitness::Vector{<:Real}, N::Int)
    λ = length(fitness)
    @assert λ >= N "Cannot select more then $(λ) elements"
    idx = sortperm(fitness)
    return idx[1:N]
end

"""Tournament selection"""
function tournament(groupSize :: Int)
    @assert groupSize > 0 "Group size must be positive"
    function tournamentN(fitness::Vector{<:Real}, N::Int)
        selection = fill(0,N)

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

"""
    random(fitness, N)

Returns a colection on size `N` of uniformly selected individuals from the population.
"""
random(fitness::Vector{<:Real}, N::Int) = rand(1:length(fitness),N)

"""
    permutation(fitness, N)

Returns a permutation on size `N` of the individuals from the population.
"""
function permutation(fitness::Vector{<:Real}, N::Int)
    λ = length(fitness)
    @assert λ >= N "Cannot select more then $(λ) elements"
    return randperm(λ)[1:N]
end

"""
    randomoffset(fitness, N)

Returns a cycle selection on size `N` from an arbitrary position.
"""
function randomoffset(fitness::Vector{<:Real}, N::Int)
    λ = length(fitness)
    @assert λ >= N "Cannot select more then $(λ) elements"
    rg = rand(1:λ)
    return [(i+rg)%λ+1 for i in 1:N]
end

"""
    best(fitness, N)

Returns a collection of best individuals of size `N`.
"""
best(fitness::Vector{<:Real}, N::Int) = fill(last(findmin(fitness)),N)


# Utils: selection
function pselection(prob::Vector{<:Real}, N::Int)
    cp = cumsum(prob)
    # println(cp)
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

function randexcl(itr, exclude, dims)
    idxs = Int[]
    while length(idxs) < dims
        j = rand(itr)
        (j ∈ exclude || j ∈ idxs) && continue
        push!(idxs, j)
    end
    return idxs
end
