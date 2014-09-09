type Individual
    objective::Vector
    strategy::Dict
    fitness::Real    
end
Individual() = Individual(Array(Any,0), Dict(), NaN)
Individual(obj::Vector) = Individual(obj, Dict(), NaN)
Individual(obj::Vector, s::Dict) = Individual(obj, s, NaN)