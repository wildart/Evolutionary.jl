# Genetic Algorithms
# ==================
function ga{T}(objfun::Function, initValue::T;
              population::Int = 25,
              crossoverRate::Float64 = 0.8,
              mutationRate::Float64 = 0.01,
              selection::Function = (x->x[1]),
              crossover::Function = ((x,y)->(x,y)),
              mutation::Function = (x->x),              
              termination::Function = (x->false),
              iterations::Integer = 1_000,
              verbose = false)

end