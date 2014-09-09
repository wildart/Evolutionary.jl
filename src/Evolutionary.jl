module Evolutionary

    include("types.jl")

    # Self-Adaptation Evolution Strategy
    include("saes.jl")

    # End-User Facing Wrapper Functions
    include("optimize.jl")

end
