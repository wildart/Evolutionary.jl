function optimize(f::Function,
                  initial_x::Array;
                  method::Symbol = :nelder_mead,
                  xtol::Real = 1e-32,
                  ftol::Real = 1e-8,
                  grtol::Real = 1e-8,
                  iterations::Integer = 1_000)
end