# Welded beam design problem
# from Kalyanmoy Deb, "An effcient constraint handling method for genetic algorithms", 2000

using Evolutionary, Test, Random

beam(x) = 1.10471*x[2]*x[1]^2+0.04811*x[3]*x[4]*(14+x[2])
lx = [0.125, 0.1, 0.1, 0.125]
ux = [5.0, 10.0, 10.0, 5.0]

P = 6000; L = 14; E = 30e+6; G = 12e+6;
τₘₐₓ = 13600; σₘₐₓ = 30000; δₘₐₓ = 0.25;
M(x) = P*(L+x[2]/2)
R(x) = sqrt(0.25*(x[2]^2+(x[1]+x[3])^2))
J(x) = 2/sqrt(2)*x[1]*x[2]*(x[2]^2/12+0.25*(x[1]+x[3])^2)
P_c(x) = (4.013*E/(6*L^2))*x[3]*x[4]^3*(1-0.25*x[3]*sqrt(E/G)/L)
τ₁(x) = P/(sqrt(2)*x[1]*x[2])
τ₂(x) = M(x)*R(x)/J(x)
τ(x) = sqrt(τ₁(x)^2+τ₁(x)*τ₂(x)*x[2]/R(x)+τ₂(x)^2)
σ(x) = 6*P*L/(x[4]*x[3]^2)
δ(x) = 4*P*L^3/(E*x[4]*x[3]^3)

con_c1(x) = [τ(x), σ(x), x[4]-x[1], 0.10471*x[1]^2+0.04811*x[3]*x[4]*(14.0+x[2]), P_c(x), δ(x)]
lc = [-Inf, -Inf, 0.0, -Inf,   P, -Inf]
uc = [τₘₐₓ, σₘₐₓ, Inf,  5.0, Inf, δₘₐₓ]

con_c2(x) = [τ(x)-τₘₐₓ, σ(x)-σₘₐₓ, x[1]-x[4], 0.10471*x[1]^2+0.04811*x[3]*x[4]*(14.0+x[2])-5.0, P-P_c(x), δ(x)-δₘₐₓ]
lc = fill(-Inf, 6)
uc = fill(0.0, 6)

opts = Evolutionary.Options(iterations=2000, successive_f_tol=40)
mthd = CMAES(mu = 90, c_mu=0.3)
mthd = GA(populationSize = 120, ɛ = 0.03, crossoverRate=0.8, mutationRate=0.01, selection=rouletteinv, crossover=LX(0.0,4.0), mutation = PM(lx,ux,1.0))
mthd = DE(n=2)

c = PenaltyConstraints([1e3, 1e3, 1e3, 1e3, 1e2, 1e1, 1e3, 1e1, 1e1, 1e0], lx, ux, lc, uc, con_c1)
c = PenaltyConstraints([1e3, 1e3, 1e3, 1e3, 1e1, 1e1, 1e3, 1e1, 1e1, 1e0], Evolutionary.ConstraintBounds(lx,ux,lc,[]), con_c2)
c = WorstFitnessConstraints(Evolutionary.ConstraintBounds(lx,ux,lc,uc), con_c1)
c = WorstFitnessConstraints(Evolutionary.ConstraintBounds(lx,ux,lc,uc), con_c2)

Random.seed!(0);
result = Evolutionary.optimize(beam, c, mthd, opts)
Evolutionary.minimizer(result) |> con_c1
Evolutionary.minimizer(result) |> con_c2
[Evolutionary.optimize(beam, c, mthd, opts) |> minimum for i in 1:100] |> minimum
