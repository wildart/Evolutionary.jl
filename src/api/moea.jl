"""
    dominate(p, q)

Returns `1` if `p` is dominated by `q`, `-1` if otherwise, and `0` if dominance cannot be determined.
"""
function dominate(p::T, q::T) where {T <: AbstractArray}
    ret = 0
    for (i,j) in zip(p,q)
        if i < j
            ret == -1 && return 0
            ret = 1
        elseif j < i
            ret == 1 && return 0
            ret = -1
        end
    end
    return ret
end

"""
dominations(P::AbstractVector)

Returns a domination matrix of all elements in the input collection `P`.
"""
function dominations(P::AbstractVector{T}) where {T <: AbstractArray}
    l = length(P)
    D = zeros(Int8, l, l)
    for i in 1:l
        for j in (i+1):l
            D[i,j] = dominate(P[i],P[j])
            D[j,i] = -D[i,j]
        end
    end
    D
end

"""
    nondominatedsort!(R, F)

Calculate fronts for fitness values `F`, and store ranks of the individuals into `R`.
"""
function nondominatedsort!(R, P)
    n = size(P,2)
    @assert length(R) == n "Ranks must be defined for the whole population"

    Sₚ = Dict(i=>Set() for i in 1:n)
    C = zeros(Int, n)

    # construct first front
    F =[Int[]]
    for i in 1:n
        for j in i+1:n
            r = dominate(view(P,:,i), view(P,:,j)) #M[i,j]
            if r == 1
                push!(Sₚ[i], j)
                C[j] += 1
            elseif r == -1
                push!(Sₚ[j], i)
                C[i] += 1
            end
        end
        if C[i] == 0
            R[i] = 1
            push!(F[1], i)
        end
    end

    # construct rest of the fronts
    while !isempty(last(F))
        Q = Int[]
        for i in last(F)
            for j in Sₚ[i]
                C[j] -= 1
                if C[j] == 0
                    push!(Q, j)
                    R[j] = length(F) + 1
                end
            end
        end
        push!(F, Q)
    end
    isempty(last(F)) && pop!(F)

    F #, R #, Sₚ
end

"""
    crowding_distance!((C, F, fronts)

Calculate crowding distance for individuals and save the results into `C`
given the fitness values `F` and collection of `fronts`.
"""
function crowding_distance!(C::AbstractVector, F::AbstractMatrix{T}, fronts) where {T}
    for f in fronts
        cf = @view C[f]
        if length(cf) <= 2
            cf .= typemax(T)
        else
            # sort front by each objective value
            SF = F[:, f]
            d = size(SF,1)
            IX = zeros(Int, size(SF))
            IIX = zeros(Int, size(SF))
            for i in 1:d
                irow, iirow, row = view(IX,i,:), view(IIX,i,:), view(SF,i,:)
                sortperm!(irow, row)
                sortperm!(iirow, irow)
                permute!(row, irow)
            end
            nrm = SF[:,end] - SF[:,1]
            dst = (hcat(SF, fill(typemax(T), d)) - hcat(fill(typemin(T), d), SF)) ./ nrm
            dst[isnan.(dst)] .= zero(T)
            ss = sum(mapslices(v->diag(dst[:,v]) + diag(dst[:,v.+1]), IIX, dims=1), dims=1)
            cf .= vec(ss)/d
        end
    end
    C
end

