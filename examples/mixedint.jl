# Mixed Integer Problems
# from Kusum Deep, Krishna Pratap Singh, M. L. Kansal, and C. Mohan,
# "A real coded  genetic algorithm for solving integer and mixed integer optimization problems."
# Appl. Math. Comput. 212 (2009) 505-518

using Evolutionary, Test, Random

f1(x) = 2*x[1]+x[2]
lx = [0.0, 0]
ux = [1.6, 1]

cons(x) = [1.25-x[1]^2-x[2], x[1]+x[2]]
lc = [-Inf, -Inf]
uc = [0.0, 1.6]

tc = [Float64, Int]

cb = Evolutionary.ConstraintBounds(lx,ux,lc,uc)
c = MixedTypePenaltyConstraints(PenaltyConstraints(1e3, cb, cons), tc)
c = MixedTypePenaltyConstraints(WorstFitnessConstraints(cb, cons), tc)
init = ()->Real[rand(Float64), rand(0:1)]

opts = Evolutionary.Options(iterations=500, abstol=1e-5)
mthd = GA(populationSize=40, crossoverRate=0.8, mutationRate=0.05, selection=susinv, crossover=MILX(), mutation=MIPM(lx,ux))

Random.seed!(10);
result = Evolutionary.optimize(f1, c, init, mthd, opts)
@test minimum(result) ≈ 2.0 atol=1e-1
@test Evolutionary.minimizer(result) ≈ [0.5, 1] atol=1e-1

#-----------------------------------------------------------------

f8(z) = sum((z[4:end-1].-1).^2) - log(z[end]+1) + sum((z[i]-i)^2 for i in 1:3)

@test f8(Real[0.2, 1.280624, 1.954483, 1, 0, 0, 1]) ≈ 3.55746 atol=1e-6

cons(z) = [
    sum(z[1:end-1]),
    sum(v->v^2, z[[1,2,3,6]]),
    z[1]+z[4],
    z[2]+z[5],
    z[3]+z[6],
    z[1]+z[7],
    z[2]^2+z[5]^2,
    z[3]^2+z[6]^2,
    z[2]^2+z[6]^2,
]

lx = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
ux = [Inf, Inf, Inf, 1.0, 1.0, 1.0, 1.0]

lc = Float64[]
uc = [5.0, 5.5, 1.2, 1.8, 2.5, 1.2, 1.64, 4.25, 4.64]

tc = [Float64, Float64, Float64, Int, Int, Int, Int]

cb = Evolutionary.ConstraintBounds(lx,ux,lc,uc)
c = MixedTypePenaltyConstraints(PenaltyConstraints(1e2, cb, cons), tc)
c = MixedTypePenaltyConstraints(WorstFitnessConstraints(cb, cons), tc)
init = ()->Real[rand(Float64,3); rand(0:1,4)]

opts = Evolutionary.Options(iterations=2000)
mthd = GA(populationSize=100, ɛ=0.05, crossoverRate=0.8, mutationRate=0.01, selection=susinv, crossover=MILX(0.0,0.5,0.3), mutation=MIPM(lx,ux))

Random.seed!(235);
result = Evolutionary.optimize(f8, c, init, mthd, opts)
@test minimum(result) < 4

# [Evolutionary.optimize(f8, c, init, mthd, opts) |> minimum for i in 1:100] |> minimum
# uc .- (Evolutionary.minimizer(result) |> cons)
# Evolutionary.isfeasible(c.penalty, Evolutionary.minimizer(result))
# Evolutionary.minimizer(result) |> f8
