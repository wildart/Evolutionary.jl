# Welded beam design problem
# from Kalyanmoy Deb, "An effcient constraint handling method for genetic algorithms", 2000

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

con_c2(x) = [τₘₐₓ-τ(x), σₘₐₓ-σ(x), x[4]-x[1], 5.0-0.10471*x[1]^2+0.04811*x[3]*x[4]*(14.0+x[2]), P_c(x)-P, δₘₐₓ-δ(x)]
lc = [0.0, 0, 0.0, 0.0, 0.0, 0.0]
uc = [Inf, Inf, Inf,  5.0, Inf, δₘₐₓ]

con_c3(x) = [τ(x), σ(x), x[4]-x[1], P_c(x), δ(x)]
lc = [-Inf, -Inf, 0.0, P, -Inf]
uc = [τₘₐₓ, σₘₐₓ, Inf, Inf, δₘₐₓ]

opts = Evolutionary.Options(iterations=1500, successive_f_tol=25)
mthd = CMAES(mu = 50, c_mu=0.3)
mthd = GA(populationSize = 50, ɛ = 0.03, selection = susinv, crossover = intermediate(0.7), mutation = gaussian(2))
mthd = DE(n=2)

c = PenaltyConstraints(1e3, lx, ux, lc, uc, con_c1)
c = PenaltyConstraints([1e3, 1e3, 1e3, 1e3, 1e0, 1e0, 1e2, 1e0, 1e1, 1e0], Evolutionary.ConstraintBounds(lx,ux,lc,[]), con_c2)
c = PenaltyConstraints([1e3, 1e3, 1e3, 1e3, 1e1, 1e1, 1e4, 1e1, 1e1], Evolutionary.ConstraintBounds(lx,ux,lc,uc), con_c3)

result = Evolutionary.optimize(beam, c, c.bounds, mthd, opts)
m = Evolutionary.minimizer(result)
minimum(result)
m |> con_c!
m |> con_c2
