# Mutation operators
# ==================

# Isotropic mutation operator y' := y + σ(N_1(0, 1), ..., N_N(0, 1))
function isotropic{T <: Vector, S <: Strategy}(recombinant::T, s::S)
    vals = randn(length(recombinant)) * s[:σ]
    recombinant += vals
    return recombinant
end

# Anisotropic mutation operator y' := y + σ(N_1(0, 1), ..., N_N(0, 1))
function anisotropic{T <: Vector, S <: Strategy}(recombinant::T, s::S)
    @assert length(s[:σ]) == length(recombinant) "Sigma parameters must be defined for every dimension of objective parameter"
    vals = randn(length(recombinant)) .* s[:σ]
    recombinant += vals
    return recombinant
end


# Strategy mutation operators
# ===========================

# Isotropic strategy mutation σ' := σ exp(τ N(0, 1))
function isotropicSigma{S <: Strategy}(s::S)
    @assert :σ ∈ keys(s) && :τ ∈ keys(s) "Strategy must have parameters: σ, τ"
    return strategy(σ = s[:σ] * exp(s[:τ]*randn()), τ = s[:τ])
end

# Anisotropic strategy mutation σ' := exp(τ0 N_0(0, 1))(σ_1 exp(τ N_1(0, 1)), ..., σ_N exp(τ N_N(0, 1)))
function anisotropicSigma{S <: Strategy}(s::S)
    @assert :σ ∈ keys(s) && :τ ∈ keys(s) && :τ0 ∈ keys(s) "Strategy must have parameters: σ, τ0, τ"
    @assert isa(s[:σ], Vector) "Sigma must be a vector of parameters"
    #σ = exp(s[:τ0]*randn())*exp(s[:τ]*randn(length(s[:σ])))
    σ = exp(s[:τ]*randn(length(s[:σ])))
    return strategy(σ = σ, τ = s[:τ], τ0 = s[:τ0])
end
