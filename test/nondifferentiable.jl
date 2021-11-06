@testset "NonDifferentiable" begin
    dim = 5

    # SOO

    objfun = NonDifferentiable(sum, ones(dim))
    @test objfun.F isa Float64
    @test objfun.F == 0.0
    @test all(isnan.(objfun.x_f))
    @test value(objfun) == 0.0
    @test 0 <= value(objfun,rand(dim)) <= dim
    @test f_calls(objfun) == 1
    @test 0 <= value!(objfun,rand(dim)) <= dim
    @test f_calls(objfun) == 2
    v = value!(objfun, ones(dim))
    @test v == dim
    @test value(objfun) == v
    vs = [rand(dim) for i in 1:dim]
    fitness = fill(-1.0, dim)
    value!(Val(:serial), fitness, objfun, vs)
    @test all(fitness.>=0)
    fitness = fill(-1.0, dim)
    value!(Val(:thread), fitness, objfun, vs)
    @test all(fitness.>=0)

    objfun = NonDifferentiable(sum, BitVector(ones(dim)))
    @test objfun.F isa Int64
    @test objfun.F == 0
    @test all(iszero.(objfun.x_f))
    @test value(objfun) == 0
    @test 0 <= value(objfun,rand(Bool,dim)) <= dim
    @test f_calls(objfun) == 1
    @test 0 <= value!(objfun,rand(Bool,dim)) <= dim
    @test f_calls(objfun) == 2
    v = value!(objfun,rand(Bool,dim))
    @test 0 <= v <= dim
    @test value(objfun) == v
    fitness = fill(-1, dim)
    value!(Val(:serial), fitness, objfun, [rand(Bool,dim) for i in 1:dim])
    @test all(fitness.>=0)
    fitness = fill(-1, dim)
    value!(Val(:thread), fitness, objfun, [rand(Bool,dim) for i in 1:dim])
    @test all(fitness.>=0)

    # MOO

    Fv = zeros(dim)
    mof(F,x) = copyto!(F,x.+1)
    objfun = NonDifferentiable(mof, fill(-1.0, dim), Fv)
    @test all(iszero.(objfun.F))
    @test all(isnan.(objfun.x_f))
    @test value(objfun) == Fv
    @test value(objfun, zeros(dim)) == ones(dim)
    @test f_calls(objfun) == 1
    @test all(iszero.(objfun.F))
    @test all(iszero.(Fv))
    @test value(objfun, Fv, zeros(dim)) == ones(dim)
    @test f_calls(objfun) == 2
    @test Fv == ones(dim)
    @test value!(objfun, Fv, zeros(dim)) == ones(dim)
    @test f_calls(objfun) == 3
    @test objfun.F == ones(dim)
    @test all(iszero.(objfun.x_f))
    F = zeros(dim)
    @test value(objfun, F, zeros(dim)) == ones(dim)
    @test F == ones(dim)
    fitness = zeros(dim, dim)
    value!(Val(:serial), fitness, objfun, [zeros(dim) for i in 1:dim])
    @test fitness == ones(dim, dim)
    fitness = zeros(dim, dim)
    value!(Val(:thread), fitness, objfun, [zeros(dim) for i in 1:dim])
    @test fitness == ones(dim, dim)

    Fv = zeros(Int, dim)
    mof2(F,x) = copyto!(F,x.+1)
    objfun = NonDifferentiable(mof2, fill(false, dim), Fv)
    @test all(iszero.(objfun.F))
    @test all(.!(objfun.x_f))
    @test value(objfun) == Fv
    @test value(objfun, zeros(Bool, dim)) == ones(dim)
    @test f_calls(objfun) == 1
    @test all(iszero.(objfun.F))
    @test all(iszero.(Fv))
    @test value(objfun, Fv, zeros(Bool, dim)) == ones(dim)
    @test f_calls(objfun) == 2
    @test Fv == ones(dim)
    @test value!(objfun, Fv, ones(Bool,dim)) == ones(dim)*2
    @test f_calls(objfun) == 3
    @test objfun.F == ones(dim)*2
    @test all(.!iszero.(objfun.x_f))
    F = zeros(Int, dim)
    @test value(objfun, F, zeros(Bool, dim)) == ones(dim)
    @test F == ones(Int, dim)
    fitness = zeros(Int, dim, dim)
    value!(Val(:serial), fitness, objfun, [zeros(Bool, dim) for i in 1:dim])
    @test fitness == ones(dim, dim)
    fitness = zeros(Int, dim, dim)
    value!(Val(:thread), fitness, objfun, [zeros(Bool, dim) for i in 1:dim])
    @test fitness == ones(dim, dim)

end
